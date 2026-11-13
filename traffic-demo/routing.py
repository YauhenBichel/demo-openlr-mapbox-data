"""
Traffic-aware routing module.

Demonstrates the first Mapbox Traffic Data use case:
  "Powering custom routing engines with traffic-aware ETAs
   and finding the most efficient routes around traffic."

Features:
- Time-dependent Dijkstra using weekly speed profiles (2016 intervals)
- ETA calculation for any departure time
- Route comparison: fastest (traffic-aware) vs shortest (distance-only)
- What-if analysis: block a segment, see rerouting impact
- Alternative routes via penalty method

In production, this maps to Valhalla/OSRM with a traffic-speed overlay
updated from Flink streaming aggregation.
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from map_matching import SimpleRoadNetwork, RoadEdge
from sample_data import DC_SEGMENTS, generate_weekly_speeds


@dataclass
class Route:
    edge_ids: list[int]
    total_distance_m: float
    total_time_sec: float
    edge_times_sec: list[float]

    @property
    def avg_speed_kmh(self) -> float:
        if self.total_time_sec <= 0:
            return 0.0
        return (self.total_distance_m / 1000) / (self.total_time_sec / 3600)


@dataclass
class RouteComparison:
    fastest: Route
    shortest: Route
    time_saved_sec: float
    extra_distance_m: float


@dataclass
class WhatIfResult:
    baseline: Optional[Route]
    rerouted: Optional[Route]
    blocked_edge_id: int
    additional_time_sec: float
    additional_distance_m: float


class TrafficRouter:
    """
    Traffic-aware router using time-dependent Dijkstra.

    Uses weekly speed profiles (2016 five-minute intervals) from sample_data
    to compute travel times that vary by departure time. This is the key
    insight for the Mapbox Traffic team: the same road has very different
    speeds at 8 AM vs 8 PM.
    """

    def __init__(self, network: SimpleRoadNetwork,
                 speed_profiles: dict[int, np.ndarray] | None = None):
        self.network = network
        self.speed_profiles = speed_profiles or {}

    def get_speed_at_time(self, edge_id: int, time_index: int) -> float:
        """Get speed for an edge at a specific weekly time index (0-2015)."""
        if edge_id in self.speed_profiles:
            idx = time_index % 2016
            return float(self.speed_profiles[edge_id][idx])
        return self.network.edges[edge_id].speed_limit

    def travel_time_sec(self, edge: RoadEdge, speed_kmh: float) -> float:
        """Travel time in seconds: distance / speed."""
        if speed_kmh <= 0:
            return float('inf')
        return edge.length_m / (speed_kmh * 1000 / 3600)

    def find_route(self, from_edge_id: int, to_edge_id: int,
                   departure_time_index: int = 0,
                   blocked_edges: set[int] | None = None,
                   cost_penalties: dict[int, float] | None = None) -> Optional[Route]:
        """
        Time-dependent Dijkstra with travel-time cost and path reconstruction.

        Args:
            from_edge_id: Starting edge
            to_edge_id: Destination edge
            departure_time_index: Weekly time index (0-2015)
            blocked_edges: Edges to exclude (for what-if)
            cost_penalties: Edge cost multipliers (for alternative routes)
        """
        blocked = blocked_edges or set()
        penalties = cost_penalties or {}

        from_edge = self.network.edges.get(from_edge_id)
        to_edge = self.network.edges.get(to_edge_id)
        if from_edge is None or to_edge is None:
            return None

        start_node = from_edge.to_node
        end_node = to_edge.from_node

        if start_node == end_node:
            return Route([], 0.0, 0.0, [])

        dist = {start_node: 0.0}
        prev_edge: dict[int, int] = {}
        pq = [(0.0, start_node)]

        while pq:
            cost, u = heapq.heappop(pq)
            if u == end_node:
                break
            if cost > dist.get(u, float('inf')):
                continue

            for eid in self.network.adjacency.get(u, []):
                if eid in blocked:
                    continue
                edge = self.network.edges[eid]
                speed = self.get_speed_at_time(eid, departure_time_index)
                edge_time = self.travel_time_sec(edge, speed)
                penalty = penalties.get(eid, 1.0)
                new_cost = cost + edge_time * penalty

                if new_cost < dist.get(edge.to_node, float('inf')):
                    dist[edge.to_node] = new_cost
                    prev_edge[edge.to_node] = eid
                    heapq.heappush(pq, (new_cost, edge.to_node))

        if end_node not in dist:
            return None

        # Reconstruct path
        path = []
        node = end_node
        while node != start_node:
            if node not in prev_edge:
                return None
            eid = prev_edge[node]
            path.append(eid)
            node = self.network.edges[eid].from_node
        path.reverse()

        # Compute actual times (without penalties)
        edge_times = []
        total_dist = 0.0
        for eid in path:
            edge = self.network.edges[eid]
            speed = self.get_speed_at_time(eid, departure_time_index)
            edge_times.append(self.travel_time_sec(edge, speed))
            total_dist += edge.length_m

        return Route(path, total_dist, sum(edge_times), edge_times)

    def find_alternative(self, from_edge_id: int, to_edge_id: int,
                          departure_time_index: int = 0) -> Optional[Route]:
        """Find alternative route via penalty method (inflate primary path costs 3x)."""
        primary = self.find_route(from_edge_id, to_edge_id, departure_time_index)
        if primary is None:
            return None
        penalties = {eid: 3.0 for eid in primary.edge_ids}
        return self.find_route(from_edge_id, to_edge_id, departure_time_index,
                               cost_penalties=penalties)

    def compare_routes(self, from_edge_id: int, to_edge_id: int,
                       departure_time_index: int = 0) -> Optional[RouteComparison]:
        """Compare fastest (traffic-aware) vs shortest (distance-only)."""
        fastest = self.find_route(from_edge_id, to_edge_id, departure_time_index)
        if fastest is None:
            return None

        # For shortest distance, use speed_limit (no profiles)
        saved = self.speed_profiles
        self.speed_profiles = {}
        shortest = self.find_route(from_edge_id, to_edge_id, departure_time_index)
        self.speed_profiles = saved

        if shortest is None:
            return None

        return RouteComparison(
            fastest=fastest,
            shortest=shortest,
            time_saved_sec=shortest.total_time_sec - fastest.total_time_sec,
            extra_distance_m=fastest.total_distance_m - shortest.total_distance_m,
        )

    def simulate_closure(self, blocked_edge_id: int,
                          from_edge_id: int, to_edge_id: int,
                          departure_time_index: int = 0) -> WhatIfResult:
        """What-if: block a segment and measure routing impact."""
        baseline = self.find_route(from_edge_id, to_edge_id, departure_time_index)
        rerouted = self.find_route(from_edge_id, to_edge_id, departure_time_index,
                                    blocked_edges={blocked_edge_id})

        add_time = 0.0
        add_dist = 0.0
        if baseline and rerouted:
            add_time = rerouted.total_time_sec - baseline.total_time_sec
            add_dist = rerouted.total_distance_m - baseline.total_distance_m

        return WhatIfResult(
            baseline=baseline,
            rerouted=rerouted,
            blocked_edge_id=blocked_edge_id,
            additional_time_sec=add_time,
            additional_distance_m=add_dist,
        )


def build_router_with_profiles(seed: int = 42) -> tuple[SimpleRoadNetwork, TrafficRouter]:
    """Build a router with the DC sample network and weekly speed profiles."""
    network = SimpleRoadNetwork.from_dc_segments()
    rng = np.random.default_rng(seed)
    profiles = {}
    for eid, seg in enumerate(DC_SEGMENTS):
        _, _, _, _, _, free_flow, road_type = seg
        profiles[eid] = generate_weekly_speeds(free_flow, road_type, rng)
    return network, TrafficRouter(network, profiles)


def demo_routing():
    """CLI demo showing traffic-aware routing features."""
    print("=" * 65)
    print("  TRAFFIC-AWARE ROUTING DEMO")
    print("=" * 65)

    network, router = build_router_with_profiles()
    print(f"\n  Network: {len(network.nodes)} nodes, {len(network.edges)} edges")
    print(f"  Speed profiles: {len(router.speed_profiles)} edges with weekly data")

    # Find two connected edges for routing demo
    # Try to find a route between segments that are connected
    route_found = False
    for from_id in range(len(DC_SEGMENTS)):
        for to_id in range(len(DC_SEGMENTS)):
            if from_id == to_id:
                continue
            r = router.find_route(from_id, to_id, 0)
            if r and len(r.edge_ids) >= 2:
                _run_routing_demo(router, network, from_id, to_id)
                route_found = True
                break
        if route_found:
            break

    if not route_found:
        print("\n  No multi-hop routes available in sample network.")

    print("\n" + "=" * 65)


def _run_routing_demo(router: TrafficRouter, network: SimpleRoadNetwork,
                       from_id: int, to_id: int):
    """Run the full routing demo between two edges."""
    from_name = network.edges[from_id].name
    to_name = network.edges[to_id].name
    print(f"\n  Route: {from_name} -> {to_name}")

    # 1. Time-dependent routing: morning rush (8:15 AM Mon) vs midnight (Sun 12 AM)
    morning_rush = 99   # Monday 8:15 AM = (0*288 + 8*12 + 3)
    midnight = 0        # Sunday midnight

    print("\n  --- Time-Dependent Routing ---")
    route_rush = router.find_route(from_id, to_id, morning_rush)
    route_night = router.find_route(from_id, to_id, midnight)

    if route_rush:
        print(f"  Morning rush (Mon 8:15 AM):")
        print(f"    Time: {route_rush.total_time_sec:.0f}s | "
              f"Distance: {route_rush.total_distance_m:.0f}m | "
              f"Avg speed: {route_rush.avg_speed_kmh:.1f} km/h")
        print(f"    Path: {' -> '.join(network.edges[e].name for e in route_rush.edge_ids)}")

    if route_night:
        print(f"  Midnight (Sun 12:00 AM):")
        print(f"    Time: {route_night.total_time_sec:.0f}s | "
              f"Distance: {route_night.total_distance_m:.0f}m | "
              f"Avg speed: {route_night.avg_speed_kmh:.1f} km/h")

    if route_rush and route_night:
        diff = route_rush.total_time_sec - route_night.total_time_sec
        print(f"  Rush hour penalty: +{diff:.0f}s ({diff/60:.1f} min)")

    # 2. Route comparison
    print("\n  --- Fastest vs Shortest ---")
    comp = router.compare_routes(from_id, to_id, morning_rush)
    if comp:
        print(f"  Fastest:  {comp.fastest.total_time_sec:.0f}s, "
              f"{comp.fastest.total_distance_m:.0f}m")
        print(f"  Shortest: {comp.shortest.total_time_sec:.0f}s, "
              f"{comp.shortest.total_distance_m:.0f}m")
        if comp.time_saved_sec > 0:
            print(f"  Traffic-aware routing saves {comp.time_saved_sec:.0f}s")

    # 3. What-if: block a segment on the route
    if route_rush and route_rush.edge_ids:
        blocked = route_rush.edge_ids[0]
        print(f"\n  --- What-If: Close '{network.edges[blocked].name}' ---")
        result = router.simulate_closure(blocked, from_id, to_id, morning_rush)
        if result.rerouted:
            print(f"  Rerouted path: {' -> '.join(network.edges[e].name for e in result.rerouted.edge_ids)}")
            print(f"  Additional time: {result.additional_time_sec:+.0f}s")
            print(f"  Additional distance: {result.additional_distance_m:+.0f}m")
        else:
            print("  No alternative route available!")


if __name__ == "__main__":
    demo_routing()

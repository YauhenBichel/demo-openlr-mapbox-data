"""
HMM-based map matching using the Viterbi algorithm.

Implements the approach from Newson & Krumm (2009):
"Hidden Markov Map Matching Through Noise and Sparseness"

Given a sequence of noisy GPS observations, finds the most likely sequence
of road segments the vehicle traveled along. This is THE core algorithm
used by the Mapbox Traffic team to convert raw GPS telemetry into road-level
speed measurements.

States:      candidate road segments near each GPS observation
Observations: GPS points (lon, lat, timestamp)
Emission:    P(obs | state) = Gaussian(distance_to_segment, sigma)
Transition:  P(state_t | state_{t-1}) ∝ exp(-|route_dist - gc_dist| / beta)
"""

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from traffic_parser import haversine
from spatial_index import GridSpatialIndex, point_to_segment_distance
from sample_data import DC_SEGMENTS


@dataclass
class GPSPoint:
    lon: float
    lat: float
    timestamp: float  # seconds
    speed_kmh: float = -1.0  # -1 means unknown
    vehicle_id: str = ""


@dataclass
class RoadNode:
    id: int
    lon: float
    lat: float


@dataclass
class RoadEdge:
    id: int
    from_node: int
    to_node: int
    length_m: float
    name: str = ""
    frc: int = 3
    speed_limit: float = 50.0


@dataclass
class MatchResult:
    """Result of matching a single GPS point to the road network."""
    gps_point: GPSPoint
    edge_id: int
    snap_lon: float
    snap_lat: float
    distance_m: float  # distance from GPS to snapped position
    confidence: float


class SimpleRoadNetwork:
    """
    In-memory road network graph using adjacency list representation.

    Nodes = intersections, Edges = road segments (directed).
    Supports Dijkstra shortest path computation, which is needed for
    the transition probability in the HMM map matcher.
    """

    def __init__(self):
        self.nodes: dict[int, RoadNode] = {}
        self.edges: dict[int, RoadEdge] = {}
        self.adjacency: dict[int, list[int]] = defaultdict(list)  # node -> [edge_ids]

    def add_node(self, node: RoadNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: RoadEdge):
        self.edges[edge.id] = edge
        self.adjacency[edge.from_node].append(edge.id)

    def shortest_path_distance(self, from_edge_id: int, to_edge_id: int) -> float:
        """
        Compute shortest path distance between end of from_edge and start of to_edge.

        Uses Dijkstra's algorithm with a priority queue.
        Returns distance in meters, or float('inf') if unreachable.
        """
        from_edge = self.edges[from_edge_id]
        to_edge = self.edges[to_edge_id]

        start_node = from_edge.to_node
        end_node = to_edge.from_node

        if start_node == end_node:
            return 0.0

        # Dijkstra
        dist = {start_node: 0.0}
        pq = [(0.0, start_node)]
        max_search_dist = 5000.0  # limit search to 5km to avoid expensive queries

        while pq:
            d, u = heapq.heappop(pq)

            if u == end_node:
                return d

            if d > dist.get(u, float('inf')):
                continue

            if d > max_search_dist:
                break

            for edge_id in self.adjacency.get(u, []):
                edge = self.edges[edge_id]
                new_dist = d + edge.length_m
                if new_dist < dist.get(edge.to_node, float('inf')):
                    dist[edge.to_node] = new_dist
                    heapq.heappush(pq, (new_dist, edge.to_node))

        return float('inf')

    @classmethod
    def from_dc_segments(cls) -> 'SimpleRoadNetwork':
        """
        Build a road network from the DC sample segments.
        Creates nodes at segment endpoints and connects them.
        """
        network = cls()
        node_map = {}  # (rounded_lon, rounded_lat) -> node_id
        next_node_id = 0
        next_edge_id = 0

        def get_or_create_node(lon: float, lat: float) -> int:
            nonlocal next_node_id
            # Round to ~10m precision for node deduplication
            key = (round(lon, 4), round(lat, 4))
            if key not in node_map:
                node_map[key] = next_node_id
                network.add_node(RoadNode(next_node_id, lon, lat))
                next_node_id += 1
            return node_map[key]

        for lon1, lat1, lon2, lat2, name, speed, road_type in DC_SEGMENTS:
            from_id = get_or_create_node(lon1, lat1)
            to_id = get_or_create_node(lon2, lat2)
            length = haversine(lat1, lon1, lat2, lon2) * 1000  # km to m

            frc = {"highway": 0, "arterial": 2, "local": 4}.get(road_type, 3)

            network.add_edge(RoadEdge(
                id=next_edge_id,
                from_node=from_id,
                to_node=to_id,
                length_m=length,
                name=name,
                frc=frc,
                speed_limit=speed,
            ))
            next_edge_id += 1

        # Add connector segments to create a connected subgraph for routing demos
        connectors = [
            # (from_key, to_key, name, frc, speed_limit)
            ((-77.009, 38.905), (-77.02, 38.905), "K-to-H Connector", 2, 40),
            ((-77.05, 38.8977), (-77.0722, 38.9007), "Const-to-M Connector", 2, 35),
            ((-77.005, 38.905), (-77.01, 38.92), "H-to-Bladensburg Connector", 4, 30),
            ((-77.05, 38.9), (-77.0365, 38.8977), "Conn-to-Const Connector", 2, 40),
            ((-77.04, 38.9007), (-77.0369, 38.9072), "M-to-K Connector", 4, 35),
            ((-77.05, 38.91), (-77.05, 38.93), "Calvert-to-Conn Connector", 4, 30),
        ]
        for from_key, to_key, cname, cfrc, cspeed in connectors:
            from_key_r = (round(from_key[0], 4), round(from_key[1], 4))
            to_key_r = (round(to_key[0], 4), round(to_key[1], 4))
            if from_key_r in node_map and to_key_r in node_map:
                fid = node_map[from_key_r]
                tid = node_map[to_key_r]
                fn = network.nodes[fid]
                tn = network.nodes[tid]
                length = haversine(fn.lat, fn.lon, tn.lat, tn.lon) * 1000
                network.add_edge(RoadEdge(
                    id=next_edge_id, from_node=fid, to_node=tid,
                    length_m=length, name=cname, frc=cfrc, speed_limit=cspeed,
                ))
                next_edge_id += 1

        print(f"Road network: {len(network.nodes)} nodes, "
              f"{len(network.edges)} edges")
        return network


@dataclass
class _Candidate:
    """Internal: a candidate road segment for a GPS observation."""
    edge_id: int
    snap_lon: float
    snap_lat: float
    distance_m: float  # distance from GPS point to snapped position


class HMMMapMatcher:
    """
    Hidden Markov Model map matcher using the Viterbi algorithm.

    Parameters:
        sigma: GPS noise standard deviation in meters (default 10m for phone GPS).
               Higher values make the matcher more tolerant of GPS noise.
        beta:  Transition tolerance in meters (default 30m).
               Controls how much the route distance can differ from the
               straight-line distance between consecutive GPS points.
        max_candidates: Maximum candidate segments per observation (default 5).
        search_radius_deg: Spatial search radius in degrees (default 0.003 ≈ 333m).
    """

    def __init__(self, road_network: SimpleRoadNetwork,
                 spatial_index: GridSpatialIndex,
                 sigma: float = 10.0,
                 beta: float = 30.0,
                 max_candidates: int = 5,
                 search_radius_deg: float = 0.003):
        self.network = road_network
        self.index = spatial_index
        self.sigma = sigma
        self.beta = beta
        self.max_candidates = max_candidates
        self.search_radius = search_radius_deg

    def _emission_log_prob(self, distance_m: float) -> float:
        """
        Log emission probability: how likely is this GPS observation
        given the candidate road segment?

        Uses a Gaussian distribution centered on the road segment.
        P(obs | segment) = N(distance; 0, sigma^2)
        log P = -0.5 * (d / sigma)^2  (ignoring constant normalization)
        """
        return -0.5 * (distance_m / self.sigma) ** 2

    def _transition_log_prob(self, route_dist_m: float,
                             gc_dist_m: float) -> float:
        """
        Log transition probability: how likely is the transition between
        two consecutive candidate segments?

        Based on the difference between the shortest-path route distance
        and the great-circle distance between consecutive GPS observations.
        Penalizes cases where the route distance diverges significantly
        from the straight-line distance (suggesting an unlikely detour).

        P(s_t | s_{t-1}) ∝ exp(-|route_dist - gc_dist| / beta)
        """
        if route_dist_m == float('inf'):
            return -100.0  # very unlikely if unreachable
        diff = abs(route_dist_m - gc_dist_m)
        return -diff / self.beta

    def _find_candidates(self, gps: GPSPoint) -> list[_Candidate]:
        """Find candidate road segments near a GPS point."""
        nearby = self.index.query_radius(gps.lon, gps.lat, self.search_radius)

        candidates = []
        for seg, dist_km, snap_lon, snap_lat in nearby[:self.max_candidates]:
            # Find the edge_id that matches this segment
            # The spatial index stores RoadSegment objects from the parser;
            # we need to match them to our network edges by coordinates
            edge_id = self._find_matching_edge(seg)
            if edge_id is not None:
                candidates.append(_Candidate(
                    edge_id=edge_id,
                    snap_lon=snap_lon,
                    snap_lat=snap_lat,
                    distance_m=dist_km * 1000,
                ))

        return candidates

    def _find_matching_edge(self, segment) -> int | None:
        """Find the network edge matching a parsed RoadSegment."""
        if not segment.points:
            return None

        seg_start = (round(segment.start.lon, 4), round(segment.start.lat, 4))
        seg_end = (round(segment.end.lon, 4), round(segment.end.lat, 4))

        for edge_id, edge in self.network.edges.items():
            from_node = self.network.nodes[edge.from_node]
            to_node = self.network.nodes[edge.to_node]
            edge_start = (round(from_node.lon, 4), round(from_node.lat, 4))
            edge_end = (round(to_node.lon, 4), round(to_node.lat, 4))

            if edge_start == seg_start and edge_end == seg_end:
                return edge_id
        return None

    def match(self, trace: list[GPSPoint]) -> list[MatchResult]:
        """
        Match a GPS trace to the road network using the Viterbi algorithm.

        Algorithm overview:
        1. For each GPS point, find K candidate road segments within search radius
        2. Initialize Viterbi trellis with emission probabilities for first point
        3. Forward pass: for each subsequent point, compute best predecessor
           for each candidate using emission + transition probabilities
        4. Backtrack from the best final state to recover the optimal path

        Time complexity:  O(T * K^2 * D) where T=observations, K=candidates,
                          D=Dijkstra cost per query
        Space complexity: O(T * K) for the Viterbi trellis
        """
        if len(trace) < 2:
            return []

        T = len(trace)

        # Step 1: Find candidates for each observation
        all_candidates = [self._find_candidates(gps) for gps in trace]

        # Skip observations with no candidates
        valid = [(i, cands) for i, cands in enumerate(all_candidates) if cands]
        if not valid:
            return []

        # Step 2: Viterbi forward pass
        # viterbi[t] = list of (log_prob, candidate_index) for each candidate at step t
        viterbi = []
        backpointer = []

        # Initialize first valid observation
        first_idx, first_cands = valid[0]
        viterbi.append([self._emission_log_prob(c.distance_m) for c in first_cands])
        backpointer.append([-1] * len(first_cands))

        # Forward pass
        for step in range(1, len(valid)):
            obs_idx, curr_cands = valid[step]
            prev_obs_idx, prev_cands = valid[step - 1]
            prev_scores = viterbi[-1]

            # Great-circle distance between consecutive GPS observations
            gc_dist_m = haversine(
                trace[prev_obs_idx].lat, trace[prev_obs_idx].lon,
                trace[obs_idx].lat, trace[obs_idx].lon,
            ) * 1000

            curr_scores = []
            curr_backptrs = []

            for k, curr_cand in enumerate(curr_cands):
                emit_lp = self._emission_log_prob(curr_cand.distance_m)
                best_score = float('-inf')
                best_prev = 0

                for j, prev_cand in enumerate(prev_cands):
                    route_dist = self.network.shortest_path_distance(
                        prev_cand.edge_id, curr_cand.edge_id,
                    )
                    trans_lp = self._transition_log_prob(route_dist, gc_dist_m)
                    score = prev_scores[j] + trans_lp + emit_lp

                    if score > best_score:
                        best_score = score
                        best_prev = j

                curr_scores.append(best_score)
                curr_backptrs.append(best_prev)

            viterbi.append(curr_scores)
            backpointer.append(curr_backptrs)

        # Step 3: Backtrack
        # Find best final state
        final_scores = viterbi[-1]
        best_final = max(range(len(final_scores)), key=lambda k: final_scores[k])

        # Trace back through backpointers
        path_indices = [best_final]
        for step in range(len(valid) - 1, 0, -1):
            path_indices.append(backpointer[step][path_indices[-1]])
        path_indices.reverse()

        # Build results
        results = []
        for step, (obs_idx, cands) in enumerate(valid):
            cand = cands[path_indices[step]]
            confidence = math.exp(min(viterbi[step][path_indices[step]], 0))

            results.append(MatchResult(
                gps_point=trace[obs_idx],
                edge_id=cand.edge_id,
                snap_lon=cand.snap_lon,
                snap_lat=cand.snap_lat,
                distance_m=cand.distance_m,
                confidence=min(1.0, confidence),
            ))

        return results


def build_demo_matcher() -> tuple[SimpleRoadNetwork, GridSpatialIndex, HMMMapMatcher]:
    """Build a complete map matcher from the DC sample data."""
    from traffic_parser import load_traffic_data, RoadSegment as ParsedSegment
    from pathlib import Path

    # Build road network
    network = SimpleRoadNetwork.from_dc_segments()

    # Build spatial index from the same segments
    # We need parsed RoadSegment objects for the spatial index
    sample_file = Path(__file__).parent / "data" / "sample-Washington_DC.csv.gz"
    if sample_file.exists():
        segments = load_traffic_data(sample_file)
    else:
        # Build minimal segments for the spatial index
        from sample_data import generate_sample_file
        generate_sample_file(sample_file)
        segments = load_traffic_data(sample_file)

    spatial = GridSpatialIndex(cell_size_deg=0.001)
    spatial.build_from_segments(segments)

    matcher = HMMMapMatcher(network, spatial)
    return network, spatial, matcher


def demo_map_matching():
    """Run a map matching demo with a synthetic GPS trace."""
    network, spatial, matcher = build_demo_matcher()

    # Generate a synthetic GPS trace along Constitution Ave → I-395
    # Constitution Ave: (-77.0365, 38.8977) → (-77.0500, 38.8977)
    # Then turn onto I-395: (-77.0147, 38.8990) → (-77.0147, 38.8810)
    rng = np.random.default_rng(42)
    trace = []
    t = 0.0

    # Drive along Constitution Ave
    for i in range(10):
        frac = i / 9
        lon = -77.0365 + frac * (-77.0500 - (-77.0365))
        lat = 38.8977
        # Add GPS noise (~8m standard deviation)
        lon += rng.normal(0, 0.00008)
        lat += rng.normal(0, 0.00008)
        trace.append(GPSPoint(lon=lon, lat=lat, timestamp=t, speed_kmh=45.0))
        t += 2.0

    print(f"\nMap matching {len(trace)} GPS points...")
    results = matcher.match(trace)

    print(f"\nMatched {len(results)} points:")
    print(f"{'#':<4} {'GPS (lon, lat)':<28} {'Snapped (lon, lat)':<28} "
          f"{'Dist(m)':<8} {'Edge':<6} {'Conf'}")
    print("-" * 85)

    for i, r in enumerate(results):
        edge = network.edges.get(r.edge_id)
        edge_name = edge.name if edge else "?"
        print(f"{i:<4} ({r.gps_point.lon:.4f}, {r.gps_point.lat:.4f})   "
              f"({r.snap_lon:.4f}, {r.snap_lat:.4f})   "
              f"{r.distance_m:>6.1f}  {r.edge_id:<6} {r.confidence:.3f}  "
              f"{edge_name}")


if __name__ == "__main__":
    demo_map_matching()

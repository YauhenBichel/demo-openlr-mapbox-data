"""
Traffic scenario simulation module.

Demonstrates the second Mapbox Traffic Data use case:
  "Traffic analysis and simulation for urban planning, route planning
   at scale, supply and demand analysis, and simulation of billions
   of potential routes."

Features:
- Network-wide KPI computation (avg speed, total delay, congestion %)
- OD (Origin-Destination) matrix generation
- Capacity analysis with Volume/Capacity ratio and Level of Service
- What-if scenario simulation (closures, speed reductions)
- Time-of-day comparison (rush hour vs off-peak)

In production, this maps to offline analytics jobs (Spark/Presto)
running over the historical speed tile archive.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from map_matching import SimpleRoadNetwork, RoadEdge
from sample_data import DC_SEGMENTS, generate_weekly_speeds, INTERVALS_PER_DAY


@dataclass
class NetworkKPIs:
    avg_speed_kmh: float
    total_delay_hours: float
    congested_pct: float
    avg_congestion_index: float
    total_vkt_km: float  # vehicle-km traveled


@dataclass
class ODEntry:
    origin_edge: int
    dest_edge: int
    volume: int


@dataclass
class CapacityResult:
    edge_id: int
    name: str
    volume: int
    capacity: int
    vc_ratio: float
    level_of_service: str  # A-F


@dataclass
class ScenarioResult:
    name: str
    baseline: NetworkKPIs
    scenario: NetworkKPIs
    impact_summary: str


class TrafficSimulator:
    """Network-level traffic simulation and analysis."""

    def __init__(self, network: SimpleRoadNetwork,
                 speed_profiles: dict[int, np.ndarray]):
        self.network = network
        self.speed_profiles = speed_profiles

    def compute_network_kpis(self, time_index: int = 0,
                              profiles_override: dict[int, np.ndarray] | None = None) -> NetworkKPIs:
        """Compute network-wide KPIs at a given time."""
        profiles = profiles_override or self.speed_profiles
        speeds = []
        congestion_indices = []
        total_delay = 0.0
        total_vkt = 0.0

        for eid, edge in self.network.edges.items():
            if eid in profiles:
                speed = float(profiles[eid][time_index % 2016])
            else:
                speed = edge.speed_limit
            speeds.append(speed)

            # Congestion index: 1 - (actual / free_flow)
            ci = max(0, 1.0 - speed / edge.speed_limit) if edge.speed_limit > 0 else 0
            congestion_indices.append(ci)

            # Delay: extra time vs free flow
            if speed > 0 and edge.speed_limit > 0:
                actual_time_h = (edge.length_m / 1000) / speed
                ff_time_h = (edge.length_m / 1000) / edge.speed_limit
                total_delay += max(0, actual_time_h - ff_time_h)

            # VKT: assume 1 vehicle per segment for baseline
            total_vkt += edge.length_m / 1000

        return NetworkKPIs(
            avg_speed_kmh=float(np.mean(speeds)) if speeds else 0,
            total_delay_hours=total_delay,
            congested_pct=sum(1 for ci in congestion_indices if ci > 0.3)
                          / max(len(congestion_indices), 1) * 100,
            avg_congestion_index=float(np.mean(congestion_indices)) if congestion_indices else 0,
            total_vkt_km=total_vkt,
        )

    def generate_od_matrix(self, num_trips: int = 100,
                            seed: int = 42) -> list[ODEntry]:
        """Generate synthetic OD matrix weighted by road class."""
        rng = np.random.default_rng(seed)
        edge_ids = list(self.network.edges.keys())
        if len(edge_ids) < 2:
            return []

        # Highways attract more trips
        weights = []
        for eid in edge_ids:
            frc = self.network.edges[eid].frc
            weights.append(3.0 if frc <= 1 else 1.0)
        weights = np.array(weights) / sum(weights)

        od = []
        for _ in range(num_trips):
            o = int(rng.choice(edge_ids, p=weights))
            d = int(rng.choice(edge_ids, p=weights))
            if o != d:
                od.append(ODEntry(o, d, 1))
        return od

    def capacity_analysis(self, volumes: dict[int, int]) -> list[CapacityResult]:
        """Compute V/C ratio and Level of Service per edge."""
        results = []
        for eid, edge in self.network.edges.items():
            vol = volumes.get(eid, 0)
            # Capacity heuristic based on road class (FRC)
            if edge.frc <= 1:
                capacity = 2000  # highway
            elif edge.frc <= 3:
                capacity = 1200  # arterial
            else:
                capacity = 600   # local

            vc = vol / capacity if capacity > 0 else 0
            los = _vc_to_los(vc)
            results.append(CapacityResult(eid, edge.name, vol, capacity, vc, los))

        results.sort(key=lambda r: r.vc_ratio, reverse=True)
        return results

    def simulate_scenario(self, name: str,
                           speed_overrides: dict[int, float] | None = None,
                           blocked_edges: set[int] | None = None,
                           time_index: int = 0) -> ScenarioResult:
        """Run a what-if scenario by modifying speeds or blocking edges."""
        baseline = self.compute_network_kpis(time_index)

        # Create modified profiles
        modified = {}
        for eid in self.speed_profiles:
            if blocked_edges and eid in blocked_edges:
                modified[eid] = np.full(2016, 5.0)  # near-standstill
            elif speed_overrides and eid in speed_overrides:
                modified[eid] = np.full(2016, speed_overrides[eid])
            else:
                modified[eid] = self.speed_profiles[eid]

        scenario = self.compute_network_kpis(time_index, modified)

        delay_change = scenario.total_delay_hours - baseline.total_delay_hours
        speed_change = scenario.avg_speed_kmh - baseline.avg_speed_kmh
        summary = (f"Avg speed: {speed_change:+.1f} km/h, "
                   f"Total delay: {delay_change:+.3f}h, "
                   f"Congested: {scenario.congested_pct:.0f}% "
                   f"(was {baseline.congested_pct:.0f}%)")

        return ScenarioResult(name, baseline, scenario, summary)


def _vc_to_los(vc: float) -> str:
    """Convert volume/capacity ratio to Level of Service (HCM standard)."""
    if vc <= 0.35:
        return "A"
    elif vc <= 0.54:
        return "B"
    elif vc <= 0.77:
        return "C"
    elif vc <= 0.93:
        return "D"
    elif vc <= 1.0:
        return "E"
    else:
        return "F"


def build_simulator(seed: int = 42) -> tuple[SimpleRoadNetwork, TrafficSimulator]:
    """Build simulator with DC network and weekly speed profiles."""
    network = SimpleRoadNetwork.from_dc_segments()
    rng = np.random.default_rng(seed)
    profiles = {}
    for eid, seg in enumerate(DC_SEGMENTS):
        _, _, _, _, _, free_flow, road_type = seg
        profiles[eid] = generate_weekly_speeds(free_flow, road_type, rng)
    return network, TrafficSimulator(network, profiles)


def demo_simulation():
    """CLI demo of traffic simulation and analysis features."""
    print("=" * 65)
    print("  TRAFFIC ANALYSIS & SIMULATION DEMO")
    print("=" * 65)

    network, sim = build_simulator()
    print(f"\n  Network: {len(network.nodes)} nodes, {len(network.edges)} edges")

    # 1. Time-of-day comparison
    morning_rush = 99    # Mon 8:15 AM
    evening_rush = 210   # Mon 5:30 PM
    midnight = 0         # Sun midnight
    weekend_noon = 1872  # Sat noon (6.5 * 288)

    print("\n  --- Network KPIs by Time of Day ---")
    for label, idx in [("Monday 8:15 AM (rush)", morning_rush),
                        ("Monday 5:30 PM (rush)", evening_rush),
                        ("Sunday midnight", midnight),
                        ("Saturday noon", weekend_noon)]:
        kpis = sim.compute_network_kpis(idx)
        print(f"  {label:30s}  "
              f"Avg: {kpis.avg_speed_kmh:5.1f} km/h | "
              f"Delay: {kpis.total_delay_hours:.3f}h | "
              f"Congested: {kpis.congested_pct:4.0f}%")

    # 2. OD Matrix
    print("\n  --- OD Matrix (top trips) ---")
    od = sim.generate_od_matrix(50)
    print(f"  Generated {len(od)} OD pairs")
    # Aggregate volumes per edge (simplified)
    volumes: dict[int, int] = {}
    for entry in od:
        volumes[entry.origin_edge] = volumes.get(entry.origin_edge, 0) + entry.volume
        volumes[entry.dest_edge] = volumes.get(entry.dest_edge, 0) + entry.volume

    # 3. Capacity analysis
    print("\n  --- Capacity Analysis (top segments by V/C) ---")
    cap = sim.capacity_analysis(volumes)
    for r in cap[:5]:
        print(f"  {r.name:30s}  V/C: {r.vc_ratio:.2f}  LOS: {r.level_of_service}  "
              f"({r.volume}/{r.capacity} veh/hr)")

    # 4. Scenario: close a highway
    print("\n  --- Scenario: Close I-395 SB ---")
    # Find I-395 SB edge
    i395_id = None
    for eid, edge in network.edges.items():
        if "I-395 SB" in edge.name:
            i395_id = eid
            break

    if i395_id is not None:
        result = sim.simulate_scenario(
            "Close I-395 SB",
            blocked_edges={i395_id},
            time_index=morning_rush,
        )
        print(f"  {result.impact_summary}")
    else:
        print("  I-395 SB not found in network")

    # 5. Scenario: reduce speed on all local roads
    print("\n  --- Scenario: 20 km/h Zone on Local Roads ---")
    local_overrides = {}
    for eid, edge in network.edges.items():
        if edge.frc >= 4:
            local_overrides[eid] = 20.0
    result = sim.simulate_scenario(
        "20 km/h local zone",
        speed_overrides=local_overrides,
        time_index=morning_rush,
    )
    print(f"  {result.impact_summary}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    demo_simulation()

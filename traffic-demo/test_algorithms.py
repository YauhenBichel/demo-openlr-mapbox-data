"""
Unit tests for core traffic algorithms.

Tests spatial index, map matching (Viterbi), and geo utilities.
Run with: python -m pytest test_algorithms.py -v
"""

import math

import numpy as np
import pytest

from traffic_parser import RoadSegment, LocationPoint, haversine
from spatial_index import GridSpatialIndex, point_to_segment_distance, BoundingBox
from map_matching import (
    SimpleRoadNetwork, RoadNode, RoadEdge, HMMMapMatcher, GPSPoint,
)
from sample_data import DC_SEGMENTS


# --- Helpers ---

def make_segment(lon1, lat1, lon2, lat2, openlr="test") -> RoadSegment:
    """Create a minimal RoadSegment for testing."""
    return RoadSegment(
        openlr_raw=openlr,
        points=[LocationPoint(lon1, lat1), LocationPoint(lon2, lat2)],
        speeds=np.zeros(2016, dtype=int),
    )


# --- GeoUtils tests ---

class TestHaversine:
    def test_same_point(self):
        assert haversine(38.9, -77.0, 38.9, -77.0) == 0.0

    def test_known_distance(self):
        # DC to Baltimore: ~56 km
        dist = haversine(38.9072, -77.0369, 39.2904, -76.6122)
        assert 50 < dist < 60

    def test_symmetry(self):
        d1 = haversine(38.9, -77.0, 39.0, -76.9)
        d2 = haversine(39.0, -76.9, 38.9, -77.0)
        assert abs(d1 - d2) < 1e-10


class TestPointToSegmentDistance:
    def test_perpendicular_projection(self):
        # Point directly above the midpoint of a horizontal segment
        # Segment from (0, 0) to (0.01, 0), point at (0.005, 0.001)
        dist, snap_lon, snap_lat = point_to_segment_distance(
            0.005, 0.001,  # point
            0.0, 0.0,      # segment start
            0.01, 0.0,     # segment end
        )
        assert abs(snap_lon - 0.005) < 0.0001
        assert abs(snap_lat - 0.0) < 0.0001
        assert dist > 0

    def test_endpoint_projection(self):
        # Point beyond segment end — should snap to nearest endpoint
        dist, snap_lon, snap_lat = point_to_segment_distance(
            0.02, 0.0,     # point far beyond end
            0.0, 0.0,      # segment start
            0.01, 0.0,     # segment end
        )
        assert abs(snap_lon - 0.01) < 0.0001
        assert abs(snap_lat - 0.0) < 0.0001

    def test_degenerate_segment(self):
        # Zero-length segment
        dist, snap_lon, snap_lat = point_to_segment_distance(
            0.001, 0.001,
            0.0, 0.0,
            0.0, 0.0,
        )
        assert dist > 0
        assert snap_lon == 0.0
        assert snap_lat == 0.0


# --- Spatial Index tests ---

class TestGridSpatialIndex:
    def test_insert_and_query(self):
        index = GridSpatialIndex(cell_size_deg=0.001)
        seg = make_segment(-77.036, 38.907, -77.009, 38.905)
        index.insert(seg)
        assert index.segment_count == 1

        # Query near the segment
        results = index.query_radius(-77.02, 38.906, radius_deg=0.03)
        assert len(results) >= 1
        assert results[0][0] is seg

    def test_empty_query(self):
        index = GridSpatialIndex()
        results = index.query_radius(0, 0, 0.01)
        assert results == []

    def test_query_bbox(self):
        index = GridSpatialIndex(cell_size_deg=0.001)
        seg1 = make_segment(-77.04, 38.90, -77.03, 38.90, "s1")
        seg2 = make_segment(-77.01, 38.92, -77.00, 38.92, "s2")
        index.insert(seg1)
        index.insert(seg2)

        # BBox covering only seg1
        bbox = BoundingBox(-77.05, 38.89, -77.02, 38.91)
        results = index.query_bbox(bbox)
        assert len(results) == 1
        assert results[0].openlr_raw == "s1"

    def test_nearest(self):
        index = GridSpatialIndex(cell_size_deg=0.001)
        close = make_segment(-77.036, 38.907, -77.035, 38.907, "close")
        far = make_segment(-77.070, 38.920, -77.069, 38.920, "far")
        index.insert(close)
        index.insert(far)

        results = index.query_nearest(-77.036, 38.907, k=1)
        assert len(results) >= 1
        assert results[0][0].openlr_raw == "close"


# --- Road Network tests ---

class TestRoadNetwork:
    def setup_method(self):
        """Build a simple L-shaped network for testing."""
        self.network = SimpleRoadNetwork()
        # Nodes: 0 -- 1 -- 2 (horizontal), then 2 -- 3 (vertical)
        self.network.add_node(RoadNode(0, -77.04, 38.90))
        self.network.add_node(RoadNode(1, -77.03, 38.90))
        self.network.add_node(RoadNode(2, -77.02, 38.90))
        self.network.add_node(RoadNode(3, -77.02, 38.91))

        self.network.add_edge(RoadEdge(0, 0, 1, 870, "Seg A"))
        self.network.add_edge(RoadEdge(1, 1, 2, 870, "Seg B"))
        self.network.add_edge(RoadEdge(2, 2, 3, 1110, "Seg C"))

    def test_shortest_path_adjacent(self):
        # Adjacent edges: distance should be 0 (share a node)
        dist = self.network.shortest_path_distance(0, 1)
        assert dist == 0.0

    def test_shortest_path_one_hop(self):
        # Edge 0 → Edge 2: need to go through node 1 → node 2 via edge 1
        dist = self.network.shortest_path_distance(0, 2)
        assert dist == 870.0  # edge 1's length

    def test_shortest_path_unreachable(self):
        # Reverse direction: edge 2 → edge 0 (directed graph, no reverse edges)
        dist = self.network.shortest_path_distance(2, 0)
        assert dist == float('inf')


# --- Map Matching tests ---

class TestHMMMapMatcher:
    def setup_method(self):
        """Build a simple network and matcher for testing."""
        self.network = SimpleRoadNetwork()
        # Two parallel horizontal roads + one connecting vertical road
        #
        # Node layout:
        #   0 ---- 1 ---- 2    (top road, lat=38.91)
        #                 |
        #   3 ---- 4 ---- 5    (bottom road, lat=38.90)
        #
        self.network.add_node(RoadNode(0, -77.04, 38.91))
        self.network.add_node(RoadNode(1, -77.03, 38.91))
        self.network.add_node(RoadNode(2, -77.02, 38.91))
        self.network.add_node(RoadNode(3, -77.04, 38.90))
        self.network.add_node(RoadNode(4, -77.03, 38.90))
        self.network.add_node(RoadNode(5, -77.02, 38.90))

        # Top road
        self.network.add_edge(RoadEdge(0, 0, 1, 870, "Top-A"))
        self.network.add_edge(RoadEdge(1, 1, 2, 870, "Top-B"))
        # Bottom road
        self.network.add_edge(RoadEdge(2, 3, 4, 870, "Bot-A"))
        self.network.add_edge(RoadEdge(3, 4, 5, 870, "Bot-B"))
        # Connector
        self.network.add_edge(RoadEdge(4, 2, 5, 1110, "Connector"))

        # Build spatial index with segments that match the edges
        self.index = GridSpatialIndex(cell_size_deg=0.001)
        for eid, edge in self.network.edges.items():
            fn = self.network.nodes[edge.from_node]
            tn = self.network.nodes[edge.to_node]
            seg = make_segment(fn.lon, fn.lat, tn.lon, tn.lat, f"edge_{eid}")
            self.index.insert(seg)

        self.matcher = HMMMapMatcher(
            self.network, self.index,
            sigma=15.0, beta=50.0, max_candidates=5,
        )

    def test_match_along_top_road(self):
        """GPS trace along top road should match to edges 0 and 1."""
        trace = [
            GPSPoint(-77.039, 38.9101, 0, 40),
            GPSPoint(-77.035, 38.9099, 1, 40),
            GPSPoint(-77.031, 38.9101, 2, 40),
            GPSPoint(-77.027, 38.9100, 3, 40),
            GPSPoint(-77.023, 38.9099, 4, 40),
        ]
        results = self.matcher.match(trace)

        assert len(results) > 0
        # All matched edges should be from the top road (edges 0 or 1)
        matched_edges = {r.edge_id for r in results}
        assert matched_edges.issubset({0, 1}), f"Expected top road, got edges {matched_edges}"

    def test_match_returns_snapped_coords(self):
        """Snapped coordinates should be closer to road than GPS points."""
        trace = [
            GPSPoint(-77.039, 38.912, 0, 40),  # ~200m north of top road
            GPSPoint(-77.035, 38.911, 1, 40),
            GPSPoint(-77.031, 38.912, 2, 40),
        ]
        results = self.matcher.match(trace)

        for r in results:
            # Snapped lat should be closer to 38.91 (the road)
            # than the original GPS lat
            snap_dist = abs(r.snap_lat - 38.91)
            gps_dist = abs(r.gps_point.lat - 38.91)
            assert snap_dist <= gps_dist + 0.001

    def test_empty_trace(self):
        results = self.matcher.match([])
        assert results == []

    def test_single_point_trace(self):
        results = self.matcher.match([GPSPoint(-77.03, 38.91, 0, 40)])
        assert results == []


# --- Integration test ---

class TestDCNetworkIntegration:
    def test_build_dc_network(self):
        """Verify the DC sample network builds successfully."""
        network = SimpleRoadNetwork.from_dc_segments()
        assert len(network.nodes) > 0
        assert len(network.edges) >= len(DC_SEGMENTS)  # includes connectors


# --- Routing tests ---

from routing import TrafficRouter, Route

class TestTrafficRouter:
    def setup_method(self):
        """Build network with two paths: top (fast) and bottom (slow)."""
        self.network = SimpleRoadNetwork()
        self.network.add_node(RoadNode(0, -77.04, 38.91))
        self.network.add_node(RoadNode(1, -77.03, 38.91))
        self.network.add_node(RoadNode(2, -77.02, 38.91))
        self.network.add_node(RoadNode(3, -77.04, 38.90))
        self.network.add_node(RoadNode(4, -77.03, 38.90))
        self.network.add_node(RoadNode(5, -77.02, 38.90))

        # Top path: fast
        self.network.add_edge(RoadEdge(0, 0, 1, 870, "Top-A", speed_limit=60))
        self.network.add_edge(RoadEdge(1, 1, 2, 870, "Top-B", speed_limit=60))
        # Bottom path: slower, longer
        self.network.add_edge(RoadEdge(2, 0, 3, 1200, "Bot-A", speed_limit=40))
        self.network.add_edge(RoadEdge(3, 3, 4, 1200, "Bot-B", speed_limit=40))
        self.network.add_edge(RoadEdge(4, 4, 5, 1200, "Bot-C", speed_limit=40))
        self.network.add_edge(RoadEdge(5, 5, 2, 1200, "Bot-D", speed_limit=40))
        # Cross connectors
        self.network.add_edge(RoadEdge(6, 1, 4, 1110, "Cross-1", speed_limit=40))
        self.network.add_edge(RoadEdge(7, 5, 1, 1110, "Cross-2", speed_limit=40))

        self.router = TrafficRouter(self.network)

    def test_finds_route(self):
        route = self.router.find_route(0, 1)
        assert route is not None
        assert route.total_time_sec >= 0

    def test_unreachable_returns_none(self):
        # No reverse edges
        assert self.router.find_route(1, 0) is None

    def test_travel_time_positive(self):
        # Edge 0 ends at node 1, edge 5 starts at node 5 -> must traverse edges
        route = self.router.find_route(0, 5)
        assert route is not None
        assert route.total_distance_m > 0
        assert route.avg_speed_kmh > 0

    def test_closure_forces_reroute(self):
        # Route 0->5 via top requires edge 1; blocking it forces bottom path
        result = self.router.simulate_closure(1, 0, 5)
        if result.rerouted:
            assert 1 not in result.rerouted.edge_ids

    def test_time_dependent_speed(self):
        # Slow profile on edge 1 should make route through it slower
        profiles = {1: np.full(2016, 10.0)}
        router = TrafficRouter(self.network, profiles)
        slow = router.find_route(0, 5)
        fast = self.router.find_route(0, 5)
        assert slow is not None and fast is not None
        assert slow.total_time_sec >= fast.total_time_sec


# --- Simulation tests ---

from traffic_simulation import TrafficSimulator, NetworkKPIs

class TestTrafficSimulator:
    def setup_method(self):
        self.network = SimpleRoadNetwork()
        self.network.add_node(RoadNode(0, -77.04, 38.91))
        self.network.add_node(RoadNode(1, -77.03, 38.91))
        self.network.add_node(RoadNode(2, -77.02, 38.91))
        self.network.add_edge(RoadEdge(0, 0, 1, 870, "Seg-A", speed_limit=60))
        self.network.add_edge(RoadEdge(1, 1, 2, 870, "Seg-B", speed_limit=60))

        self.profiles = {
            0: np.full(2016, 50.0),  # slightly below limit
            1: np.full(2016, 30.0),  # congested
        }
        self.sim = TrafficSimulator(self.network, self.profiles)

    def test_network_kpis(self):
        kpis = self.sim.compute_network_kpis()
        assert kpis.avg_speed_kmh == pytest.approx(40.0, abs=0.1)
        assert kpis.total_delay_hours > 0
        assert 0 <= kpis.congested_pct <= 100

    def test_od_matrix_generation(self):
        od = self.sim.generate_od_matrix(20)
        assert len(od) > 0
        for entry in od:
            assert entry.origin_edge != entry.dest_edge

    def test_scenario_simulation(self):
        result = self.sim.simulate_scenario(
            "Block Seg-A", blocked_edges={0}, time_index=0)
        assert result.baseline.avg_speed_kmh > result.scenario.avg_speed_kmh
        assert "Avg speed" in result.impact_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

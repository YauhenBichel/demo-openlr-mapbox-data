"""
Grid-based spatial index for fast nearest-segment queries.

Used by the map matching algorithm to efficiently find candidate road segments
near each GPS observation point. Uses a fixed-grid approach where the world is
divided into cells of configurable size (default ~111m at equator).

In production at Mapbox, this would be an R-tree or S2 cell index.
The grid approach is simpler and O(1) for lookup, which is sufficient for
urban-scale road networks with relatively uniform segment density.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

from traffic_parser import RoadSegment, haversine


@dataclass
class BoundingBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, lon: float, lat: float) -> bool:
        return (self.min_lon <= lon <= self.max_lon and
                self.min_lat <= lat <= self.max_lat)


def point_to_segment_distance(px: float, py: float,
                               ax: float, ay: float,
                               bx: float, by: float) -> tuple[float, float, float]:
    """
    Compute the distance from point P(px, py) to line segment A(ax, ay)-B(bx, by).

    Uses equirectangular projection for short distances, then Haversine for the
    final distance calculation.

    Returns (distance_km, snap_lon, snap_lat)
    """
    # Scale longitude by cos(latitude) for equirectangular approximation
    cos_lat = math.cos(math.radians((ay + by) / 2))
    dx = (bx - ax) * cos_lat
    dy = by - ay

    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        # Degenerate segment (zero length)
        return haversine(py, px, ay, ax), ax, ay

    # Project P onto line AB: t = dot(AP, AB) / dot(AB, AB)
    t = ((px - ax) * cos_lat * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    # Snapped point on segment
    snap_lon = ax + t * (bx - ax)
    snap_lat = ay + t * (by - ay)

    dist = haversine(py, px, snap_lat, snap_lon)
    return dist, snap_lon, snap_lat


class GridSpatialIndex:
    """
    Fixed-grid spatial index for road segments.

    Divides the coordinate space into cells. Each cell stores references to
    segments whose bounding box overlaps that cell. Supports:
    - Nearest-segment query (for map matching)
    - Bounding-box query (for tile serving)
    - Radius query
    """

    def __init__(self, cell_size_deg: float = 0.001):
        """
        Args:
            cell_size_deg: Cell size in degrees. ~0.001 deg ≈ 111m at equator.
        """
        self.cell_size = cell_size_deg
        self.grid: dict[tuple[int, int], list[RoadSegment]] = defaultdict(list)
        self.segment_count = 0

    def _cell_key(self, lon: float, lat: float) -> tuple[int, int]:
        return (int(math.floor(lon / self.cell_size)),
                int(math.floor(lat / self.cell_size)))

    def insert(self, segment: RoadSegment):
        """Insert a segment into all grid cells its bounding box overlaps."""
        if not segment.points:
            return

        lons = [p.lon for p in segment.points]
        lats = [p.lat for p in segment.points]
        min_cx, min_cy = self._cell_key(min(lons), min(lats))
        max_cx, max_cy = self._cell_key(max(lons), max(lats))

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                self.grid[(cx, cy)].append(segment)

        self.segment_count += 1

    def query_radius(self, lon: float, lat: float,
                     radius_deg: float = 0.005) -> list[tuple[RoadSegment, float, float, float]]:
        """
        Find all segments within radius of a point.

        Returns list of (segment, distance_km, snap_lon, snap_lat) sorted by distance.
        """
        cx, cy = self._cell_key(lon, lat)
        cell_radius = int(math.ceil(radius_deg / self.cell_size))

        seen = set()
        results = []

        for dcx in range(-cell_radius, cell_radius + 1):
            for dcy in range(-cell_radius, cell_radius + 1):
                for seg in self.grid.get((cx + dcx, cy + dcy), []):
                    seg_id = id(seg)
                    if seg_id in seen:
                        continue
                    seen.add(seg_id)

                    dist, snap_lon, snap_lat = point_to_segment_distance(
                        lon, lat,
                        seg.start.lon, seg.start.lat,
                        seg.end.lon, seg.end.lat,
                    )
                    if dist <= radius_deg * 111.0:  # rough degree-to-km conversion
                        results.append((seg, dist, snap_lon, snap_lat))

        results.sort(key=lambda x: x[1])
        return results

    def query_nearest(self, lon: float, lat: float,
                      k: int = 5) -> list[tuple[RoadSegment, float, float, float]]:
        """Find the k nearest segments to a point."""
        # Start with small radius, expand if not enough results
        for radius in [0.002, 0.005, 0.01, 0.02, 0.05]:
            results = self.query_radius(lon, lat, radius)
            if len(results) >= k:
                return results[:k]
        return results[:k]

    def query_bbox(self, bbox: BoundingBox) -> list[RoadSegment]:
        """Return all segments overlapping a bounding box."""
        min_cx, min_cy = self._cell_key(bbox.min_lon, bbox.min_lat)
        max_cx, max_cy = self._cell_key(bbox.max_lon, bbox.max_lat)

        seen = set()
        results = []

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                for seg in self.grid.get((cx, cy), []):
                    seg_id = id(seg)
                    if seg_id not in seen:
                        seen.add(seg_id)
                        results.append(seg)

        return results

    def build_from_segments(self, segments: list[RoadSegment]):
        """Bulk insert all segments."""
        for seg in segments:
            self.insert(seg)
        print(f"Spatial index built: {self.segment_count} segments in "
              f"{len(self.grid)} cells")

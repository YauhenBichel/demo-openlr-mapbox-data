"""
Parse Mapbox Traffic Data CSV.gz files with OpenLR location references.

Handles both:
- OpenLR variant: column 0 = base64 OpenLR, columns 1-2016 = speeds
- OSM variant: columns 0,1 = node IDs, columns 2-2017 = speeds

Mapbox typical-speed files contain 2,016 speed values per segment,
representing every 5-minute interval of a typical week.
"""

import base64
import csv
import gzip
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

INTERVALS_PER_WEEK = 2016
INTERVALS_PER_DAY = 288
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


@dataclass
class LocationPoint:
    """A single point in an OpenLR location reference."""
    lon: float
    lat: float
    frc: int = 0  # Functional Road Class (0=motorway ... 7=other)
    fow: int = 0  # Form of Way (0=undefined, 1=motorway, 2=multi-carriageway, 3=single)
    bearing: float = 0.0


@dataclass
class RoadSegment:
    """A decoded road segment with its traffic speed data."""
    openlr_raw: str
    points: list  # list of LocationPoint
    speeds: np.ndarray  # 2,016 speed values in km/h

    @property
    def start(self) -> LocationPoint:
        return self.points[0]

    @property
    def end(self) -> LocationPoint:
        return self.points[-1]

    @property
    def start_coords(self) -> tuple:
        return (self.start.lat, self.start.lon)

    @property
    def end_coords(self) -> tuple:
        return (self.end.lat, self.end.lon)

    @property
    def length_km(self) -> float:
        """Approximate segment length using Haversine formula."""
        return haversine(self.start.lat, self.start.lon,
                         self.end.lat, self.end.lon)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def decode_openlr_binary(data: str) -> list[LocationPoint]:
    """
    Decode a base64-encoded OpenLR binary location reference.

    Returns a list of LocationPoints representing the location reference.
    OpenLR binary format v3: status byte + location reference points.
    """
    raw = base64.b64decode(data)
    if len(raw) < 13:
        raise ValueError(f"OpenLR data too short: {len(raw)} bytes")

    points = []

    # First point: absolute coordinates (24-bit each)
    lon_int = (raw[1] << 16) | (raw[2] << 8) | raw[3]
    lat_int = (raw[4] << 16) | (raw[5] << 8) | raw[6]

    lon = (lon_int / (1 << 24)) * 360.0 - 180.0
    lat = (lat_int / (1 << 24)) * 180.0 - 90.0

    # First point attributes
    attr1 = raw[7]
    frc = (attr1 >> 5) & 0x07
    fow = (attr1 >> 2) & 0x07
    bearing_sector = ((attr1 & 0x03) << 3)

    attr2 = raw[8]
    bearing_sector |= (attr2 >> 5) & 0x07
    bearing = bearing_sector * 360.0 / 32.0

    points.append(LocationPoint(lon=lon, lat=lat, frc=frc, fow=fow, bearing=bearing))

    # Parse intermediate and last points
    offset = 9
    while offset + 4 <= len(raw):
        # Relative coordinates (16-bit signed)
        dlon = struct.unpack_from('>h', raw, offset)[0]
        dlat = struct.unpack_from('>h', raw, offset + 2)[0]
        offset += 4

        lon2 = lon + (dlon / (1 << 24)) * 360.0
        lat2 = lat + (dlat / (1 << 24)) * 180.0

        # Last point attributes (if available)
        frc2, fow2, bearing2 = 0, 0, 0.0
        if offset < len(raw):
            attr_last = raw[offset]
            frc2 = (attr_last >> 5) & 0x07
            fow2 = (attr_last >> 2) & 0x07
            offset += 1

        points.append(LocationPoint(lon=lon2, lat=lat2, frc=frc2, fow=fow2, bearing=bearing2))

        # Update reference for next relative point
        lon, lat = lon2, lat2

    return points


def time_index(day: str, hour: int, minute: int = 0) -> int:
    """
    Convert day/time to column index in Mapbox typical-speed data.

    Day 0 = Sunday in the file's time zone.

    Args:
        day: Day name ("Sun", "Mon", ..., "Sat") or int (0=Sun, 6=Sat)
        hour: Hour (0-23)
        minute: Minute (0-59, will be rounded to nearest 5-min interval)

    Returns:
        Index into the 2,016-column speed array.
    """
    if isinstance(day, str):
        day_map = {d: i for i, d in enumerate(DAYS)}
        day_num = day_map[day[:3].capitalize()]
    else:
        day_num = day

    interval = minute // 5
    return day_num * INTERVALS_PER_DAY + hour * 12 + interval


def index_to_time(idx: int) -> str:
    """Convert a speed array index back to a human-readable day/time string."""
    day_num = idx // INTERVALS_PER_DAY
    remainder = idx % INTERVALS_PER_DAY
    hour = remainder // 12
    minute = (remainder % 12) * 5
    return f"{DAYS[day_num]} {hour:02d}:{minute:02d}"


def load_traffic_data(filepath: str | Path) -> list[RoadSegment]:
    """
    Load a Mapbox typical-speed CSV.gz file and return decoded RoadSegments.

    Auto-detects OpenLR vs OSM format based on column count.
    """
    filepath = Path(filepath)
    segments = []

    open_fn = gzip.open if filepath.suffix == '.gz' else open
    with open_fn(filepath, 'rt') as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader):
            if len(row) < 100:
                continue  # skip malformed rows

            # Detect format: OpenLR has 2017 columns (1 openlr + 2016 speeds)
            # OSM has 2018 columns (2 node IDs + 2016 speeds)
            try:
                int(row[1])
                is_osm = len(row) >= 2018
            except ValueError:
                is_osm = False

            if is_osm:
                openlr_raw = f"{row[0]}_{row[1]}"  # combine node IDs
                speed_start = 2
            else:
                openlr_raw = row[0]
                speed_start = 1

            speeds = np.array([int(s) for s in row[speed_start:speed_start + INTERVALS_PER_WEEK]])

            # Decode OpenLR to get coordinates
            if not is_osm:
                try:
                    points = decode_openlr_binary(openlr_raw)
                except Exception as e:
                    print(f"Warning: Failed to decode OpenLR on row {row_num}: {e}")
                    continue
            else:
                # For OSM format, we don't have coordinates from the file
                # In production, you'd look up node coordinates from OSM data
                points = []

            segments.append(RoadSegment(
                openlr_raw=openlr_raw,
                points=points,
                speeds=speeds,
            ))

    print(f"Loaded {len(segments)} road segments from {filepath.name}")
    return segments


def segments_to_dataframe(segments: list[RoadSegment]) -> pd.DataFrame:
    """Convert segments to a DataFrame for analysis."""
    records = []
    for i, seg in enumerate(segments):
        record = {
            'segment_id': i,
            'openlr': seg.openlr_raw,
            'start_lat': seg.start.lat if seg.points else None,
            'start_lon': seg.start.lon if seg.points else None,
            'end_lat': seg.end.lat if seg.points else None,
            'end_lon': seg.end.lon if seg.points else None,
            'frc': seg.start.frc if seg.points else None,
            'length_km': seg.length_km if seg.points else None,
        }
        # Add all speed columns
        for j in range(min(len(seg.speeds), INTERVALS_PER_WEEK)):
            record[f't{j}'] = seg.speeds[j]
        records.append(record)

    return pd.DataFrame(records)

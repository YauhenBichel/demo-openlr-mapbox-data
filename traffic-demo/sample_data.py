"""
Generate sample Mapbox-format traffic data for demo purposes.

Simulates typical-speed CSV files in OpenLR format:
- Column 0: Base64-encoded OpenLR binary location reference
- Columns 1-2016: Speed (km/h) for each 5-min interval of the week

The sample uses real road segments around Washington DC (Mapbox HQ area).
"""

import base64
import csv
import gzip
import math
import struct
from pathlib import Path

import numpy as np

# 2,016 intervals = 7 days * 24 hours * 12 (five-minute slots)
INTERVALS_PER_WEEK = 2016
INTERVALS_PER_DAY = 288  # 24 * 12


def encode_openlr_binary(lon1: float, lat1: float, bearing1: int, frc1: int,
                         fow1: int, lon2: float, lat2: float, bearing2: int,
                         frc2: int, fow2: int) -> str:
    """
    Encode a minimal OpenLR line location reference (2 points) into base64.

    OpenLR binary format (version 3):
    - Byte 0: status byte (version=3, has attributes, point type=line)
    - Bytes 1-3: first point longitude (24-bit, relative to -180)
    - Bytes 4-6: first point latitude (24-bit, relative to -90)
    - Byte 7: first point attributes (FRC, FOW, bearing sector)
    - Byte 8: first point attributes (LFRCNP, distance to next)
    - Bytes 9-10: last point relative longitude (16-bit signed)
    - Bytes 11-12: last point relative latitude (16-bit signed)
    - Byte 13: last point attributes
    """
    # Status byte: version 3, has attributes, area flag=0, is not area
    status = 0b00001011  # version 3, line location

    # Encode absolute longitude: (lon + 180) / 360 * 2^24
    lon1_int = int(round((lon1 + 180.0) / 360.0 * (1 << 24)))
    lat1_int = int(round((lat1 + 90.0) / 180.0 * (1 << 24)))

    # Encode relative longitude/latitude for second point (16-bit signed)
    dlon = int(round((lon2 - lon1) / 360.0 * (1 << 24)))
    dlat = int(round((lat2 - lat1) / 180.0 * (1 << 24)))
    # Clamp to 16-bit signed
    dlon = max(-32768, min(32767, dlon))
    dlat = max(-32768, min(32767, dlat))

    # First point attributes byte: FRC(3 bits) | FOW(3 bits) | bearing sector(5 bits)
    bear_sector1 = bearing1 * 32 // 360
    attr1 = ((frc1 & 0x07) << 5) | ((fow1 & 0x07) << 2) | ((bear_sector1 >> 3) & 0x03)
    attr2 = ((bear_sector1 & 0x07) << 5) | ((frc1 & 0x07) << 2) | 0x01  # LFRCNP + DNP

    # Last point attributes
    bear_sector2 = bearing2 * 32 // 360
    attr_last = ((frc2 & 0x07) << 5) | ((fow2 & 0x07) << 2) | ((bear_sector2 >> 3) & 0x03)

    data = bytearray()
    data.append(status)
    data.append((lon1_int >> 16) & 0xFF)
    data.append((lon1_int >> 8) & 0xFF)
    data.append(lon1_int & 0xFF)
    data.append((lat1_int >> 16) & 0xFF)
    data.append((lat1_int >> 8) & 0xFF)
    data.append(lat1_int & 0xFF)
    data.append(attr1)
    data.append(attr2)
    data.extend(struct.pack('>h', dlon))
    data.extend(struct.pack('>h', dlat))
    data.append(attr_last)

    return base64.b64encode(bytes(data)).decode('ascii')


# Sample road segments around Washington DC area
# (start_lon, start_lat, end_lon, end_lat, name, free_flow_speed, road_type)
DC_SEGMENTS = [
    (-77.0369, 38.9072, -77.0090, 38.9050, "K St NW (downtown)", 45, "arterial"),
    (-77.0365, 38.8977, -77.0500, 38.8977, "Constitution Ave", 50, "arterial"),
    (-77.0722, 38.9007, -77.0400, 38.9007, "M St NW (Georgetown)", 35, "local"),
    (-77.0147, 38.8990, -77.0147, 38.8810, "I-395 SB (downtown)", 95, "highway"),
    (-77.0147, 38.8810, -77.0147, 38.8990, "I-395 NB (downtown)", 95, "highway"),
    (-77.0889, 38.9200, -77.0500, 38.9072, "Wisconsin Ave NW", 40, "arterial"),
    (-77.0320, 38.9120, -77.0120, 38.9120, "U St NW", 35, "local"),
    (-77.0500, 38.9300, -77.0500, 38.9000, "Connecticut Ave NW", 45, "arterial"),
    (-77.0060, 38.8720, -77.0060, 38.8950, "I-295 NB", 100, "highway"),
    (-77.0060, 38.8950, -77.0060, 38.8720, "I-295 SB", 100, "highway"),
    (-77.0220, 38.9170, -77.0220, 38.8900, "North Capitol St", 50, "arterial"),
    (-77.0450, 38.8630, -77.0150, 38.8630, "I-495 EB (Beltway)", 105, "highway"),
    (-77.0150, 38.8630, -77.0450, 38.8630, "I-495 WB (Beltway)", 105, "highway"),
    (-77.0700, 38.8800, -77.0500, 38.8800, "Columbia Pike", 45, "arterial"),
    (-77.1000, 38.8850, -77.0700, 38.8850, "Lee Hwy", 55, "arterial"),
    (-77.0400, 38.8550, -77.0100, 38.8550, "Duke St (Alexandria)", 50, "arterial"),
    (-77.0600, 38.9400, -77.0300, 38.9200, "16th St NW", 40, "arterial"),
    (-77.0200, 38.9050, -77.0050, 38.9050, "H St NE", 35, "local"),
    (-77.0100, 38.9200, -77.0100, 38.8900, "Bladensburg Rd", 45, "arterial"),
    (-77.0800, 38.9100, -77.0500, 38.9100, "Calvert St NW", 30, "local"),
]


def generate_weekly_speeds(free_flow: int, road_type: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate realistic weekly speed profile for a road segment.

    Models:
    - Rush hour slowdowns (weekday 7-9 AM, 5-7 PM)
    - Overnight free-flow speeds
    - Weekend patterns (lighter traffic)
    - Random noise
    """
    speeds = np.full(INTERVALS_PER_WEEK, free_flow, dtype=float)

    for day in range(7):
        day_start = day * INTERVALS_PER_DAY
        is_weekend = day in (0, 6)  # Sunday=0, Saturday=6

        for interval in range(INTERVALS_PER_DAY):
            hour = interval / 12.0  # fractional hour
            idx = day_start + interval

            if is_weekend:
                # Weekend: mild slowdown midday
                if 10 <= hour <= 18:
                    factor = 0.80 + 0.10 * math.cos((hour - 14) * math.pi / 8)
                elif hour < 6:
                    factor = 1.0
                else:
                    factor = 0.90
            else:
                # Weekday pattern
                if 7 <= hour <= 9.5:
                    # Morning rush: peak slowdown at 8:15 AM
                    severity = math.exp(-((hour - 8.25) ** 2) / 0.5)
                    if road_type == "highway":
                        factor = 1.0 - 0.55 * severity  # highways congest heavily
                    elif road_type == "arterial":
                        factor = 1.0 - 0.45 * severity
                    else:
                        factor = 1.0 - 0.35 * severity
                elif 16.5 <= hour <= 19:
                    # Evening rush: peak at 5:30 PM
                    severity = math.exp(-((hour - 17.5) ** 2) / 0.6)
                    if road_type == "highway":
                        factor = 1.0 - 0.60 * severity
                    elif road_type == "arterial":
                        factor = 1.0 - 0.50 * severity
                    else:
                        factor = 1.0 - 0.40 * severity
                elif 11 <= hour <= 14:
                    # Lunch mild slowdown
                    factor = 0.85
                elif hour < 5:
                    # Late night
                    factor = 1.0
                else:
                    factor = 0.88

            speeds[idx] = free_flow * factor

    # Add random noise (±5%)
    noise = rng.normal(1.0, 0.03, INTERVALS_PER_WEEK)
    speeds = speeds * noise

    # Clamp to reasonable range
    speeds = np.clip(speeds, 5, free_flow * 1.05)

    return np.round(speeds).astype(int)


def generate_sample_file(output_path: Path, segments=None, seed: int = 42):
    """Generate a sample Mapbox-format traffic CSV.gz file."""
    if segments is None:
        segments = DC_SEGMENTS

    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, 'wt', newline='') as f:
        writer = csv.writer(f)
        for lon1, lat1, lon2, lat2, name, free_flow, road_type in segments:
            # Compute bearing from start to end
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            bearing = int(math.degrees(math.atan2(dlon, dlat))) % 360
            bearing_rev = (bearing + 180) % 360

            # Map road type to FRC (Functional Road Class)
            frc = {"highway": 0, "arterial": 2, "local": 4}.get(road_type, 3)
            fow = {"highway": 1, "arterial": 3, "local": 3}.get(road_type, 3)

            openlr = encode_openlr_binary(lon1, lat1, bearing, frc, fow,
                                          lon2, lat2, bearing_rev, frc, fow)

            speeds = generate_weekly_speeds(free_flow, road_type, rng)
            row = [openlr] + speeds.tolist()
            writer.writerow(row)

    print(f"Generated sample data: {output_path} ({len(segments)} segments)")
    return output_path


if __name__ == "__main__":
    output = Path(__file__).parent / "data" / "sample-Washington_DC.csv.gz"
    generate_sample_file(output)

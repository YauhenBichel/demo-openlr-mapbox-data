# Mapbox Traffic Data Demo — OpenLR Pipeline

A demo project showing how to process, analyze, and visualize Mapbox Traffic Data
using OpenLR-encoded road segment references.

## What This Demonstrates

- **OpenLR decoding**: Parse base64-encoded OpenLR binary location references into coordinates
- **Traffic data parsing**: Read Mapbox typical-speed CSV.gz files (2,016 five-minute intervals per week)
- **Congestion analysis**: Identify rush-hour bottlenecks, speed patterns, and congestion hotspots
- **Geospatial visualization**: Interactive Folium map with color-coded traffic speeds
- **Time-series analysis**: Weekly speed profiles per road segment

## Architecture

```
Mapbox .csv.gz files (OpenLR format)
        │
        ▼
┌─────────────────────┐
│  traffic_parser.py   │  Read & decode OpenLR references
│  - gzip decompress   │  Extract coordinates from binary OpenLR
│  - CSV parsing       │  Parse 2,016 speed columns
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  traffic_analysis.py │  Compute analytics
│  - Rush hour speeds  │  Congestion scoring
│  - Weekly profiles   │  Peak vs off-peak comparison
│  - Segment ranking   │  Find worst bottlenecks
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  traffic_visualizer  │  Generate outputs
│  - Folium map (HTML) │  Color-coded speed map
│  - Matplotlib charts │  Weekly speed profiles
│  - Summary stats     │  Congestion report
└─────────────────────┘
```

## Data Format

Mapbox typical-speed files (OpenLR variant):
- Column 0: Base64-encoded OpenLR binary string
- Columns 1–2016: Speed in km/h for each 5-min interval of the week
- Index mapping: `day * 288 + hour * 12 + (minute // 5)`
- Day 0 = Sunday (in the file's time zone)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# With sample data (no Mapbox license needed)
python main.py --sample

# With real Mapbox data
python main.py --file data/0320122-America-New_York.csv.gz
```

## Output

- `output/traffic_map.html` — Interactive map with speed-colored segments
- `output/speed_profiles.png` — Weekly speed profile charts
- `output/congestion_report.txt` — Top congested segments and stats

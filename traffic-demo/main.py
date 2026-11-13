"""
Mapbox Traffic Data Demo — Main Entry Point

Demonstrates processing, analyzing, and visualizing Mapbox traffic data
encoded with OpenLR location references.

Usage:
    python main.py --sample              # Generate and analyze sample data
    python main.py --file <path.csv.gz>  # Analyze a real Mapbox traffic file
    python main.py --streaming           # Run real-time streaming simulation
    python main.py --mapmatch            # Run map matching demo
"""

import argparse
import sys
from pathlib import Path

from sample_data import generate_sample_file
from traffic_parser import load_traffic_data, segments_to_dataframe
from traffic_analysis import analyze_all, generate_congestion_report
from traffic_visualizer import (
    create_traffic_map,
    plot_speed_profiles,
    plot_rush_hour_comparison,
    plot_congestion_heatmap,
)

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"


def run_demo(data_path: Path):
    """Full pipeline: parse → analyze → visualize → report."""

    print("\n" + "=" * 60)
    print("  MAPBOX TRAFFIC DATA DEMO — OpenLR Pipeline")
    print("=" * 60)

    # Step 1: Parse traffic data
    print("\n[1/5] Parsing traffic data...")
    segments = load_traffic_data(data_path)
    if not segments:
        print("ERROR: No segments loaded. Check your data file.")
        sys.exit(1)

    # Show a sample decoded segment
    seg = segments[0]
    if seg.points:
        print(f"  Sample segment: ({seg.start.lat:.4f}, {seg.start.lon:.4f}) → "
              f"({seg.end.lat:.4f}, {seg.end.lon:.4f})")
        print(f"  FRC: {seg.start.frc}, Length: {seg.length_km:.2f} km")
    print(f"  Speed data: {len(seg.speeds)} intervals "
          f"(7 days × 288 five-min slots)")
    print(f"  OpenLR: {seg.openlr_raw[:40]}...")

    # Step 2: Analyze
    print("\n[2/5] Analyzing traffic patterns...")
    analyses = analyze_all(segments)

    # Step 3: Generate congestion report
    print("\n[3/5] Generating congestion report...")
    report = generate_congestion_report(analyses)
    report_path = OUTPUT_DIR / "congestion_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report)
    print(f"\nReport saved: {report_path}")

    # Step 4: Create interactive map
    print("\n[4/5] Creating visualizations...")

    # Speed map
    map_path = create_traffic_map(analyses, OUTPUT_DIR / "traffic_map_speed.html",
                                  mode="speed")

    # Congestion map
    create_traffic_map(analyses, OUTPUT_DIR / "traffic_map_congestion.html",
                       mode="congestion")

    # Step 5: Generate charts
    print("\n[5/5] Generating charts...")
    plot_speed_profiles(analyses, OUTPUT_DIR / "speed_profiles.png", top_n=6)
    plot_rush_hour_comparison(analyses, OUTPUT_DIR / "rush_hour_comparison.png")
    plot_congestion_heatmap(analyses, OUTPUT_DIR / "congestion_heatmap.png")

    # Export DataFrame for further analysis
    df = segments_to_dataframe(segments)
    csv_path = OUTPUT_DIR / "segments_summary.csv"
    summary_cols = ['segment_id', 'start_lat', 'start_lon', 'end_lat', 'end_lon',
                    'frc', 'length_km']
    df[summary_cols].to_csv(csv_path, index=False)
    print(f"Segment summary CSV saved: {csv_path}")

    # Done
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE — Output files:")
    print("=" * 60)
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size
        if size > 1024:
            print(f"  {f.name:<40} {size/1024:.1f} KB")
        else:
            print(f"  {f.name:<40} {size} B")

    print(f"\nOpen {OUTPUT_DIR / 'traffic_map_speed.html'} in a browser "
          f"to explore the interactive map.")


def main():
    parser = argparse.ArgumentParser(
        description="Mapbox Traffic Data Demo — OpenLR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --sample                          Generate sample DC data and analyze
  python main.py --file data/0320122-*.csv.gz      Analyze a real Mapbox file
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample', action='store_true',
                       help='Generate and use sample data (no Mapbox license needed)')
    group.add_argument('--file', type=str,
                       help='Path to a Mapbox typical-speed CSV.gz file')
    group.add_argument('--streaming', action='store_true',
                       help='Run real-time streaming pipeline simulation')
    group.add_argument('--mapmatch', action='store_true',
                       help='Run HMM map matching demo')
    group.add_argument('--routing', action='store_true',
                       help='Run traffic-aware routing demo')
    group.add_argument('--simulation', action='store_true',
                       help='Run traffic simulation and what-if analysis')

    args = parser.parse_args()

    if args.routing:
        from routing import demo_routing
        demo_routing()
    elif args.simulation:
        from traffic_simulation import demo_simulation
        demo_simulation()
    elif args.streaming:
        from streaming_simulator import StreamingDemo
        demo = StreamingDemo(num_vehicles=8, duration_sec=30)
        demo.run()
    elif args.mapmatch:
        from map_matching import demo_map_matching
        demo_map_matching()
    elif args.sample:
        sample_path = DATA_DIR / "sample-Washington_DC.csv.gz"
        if not sample_path.exists():
            print("Generating sample traffic data...")
            generate_sample_file(sample_path)
        run_demo(sample_path)
    else:
        data_path = Path(args.file)
        if not data_path.exists():
            print(f"ERROR: File not found: {data_path}")
            sys.exit(1)
        run_demo(data_path)


if __name__ == "__main__":
    main()

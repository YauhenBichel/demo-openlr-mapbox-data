"""
Traffic data analysis module.

Computes congestion metrics, identifies bottlenecks, compares rush-hour vs
off-peak speeds, and generates statistical summaries from Mapbox traffic data.
"""

from dataclasses import dataclass

import numpy as np

from traffic_parser import (
    RoadSegment, INTERVALS_PER_DAY, INTERVALS_PER_WEEK, DAYS,
    time_index, index_to_time,
)


@dataclass
class SegmentAnalysis:
    """Analysis results for a single road segment."""
    segment: RoadSegment
    segment_id: int

    # Overall stats
    avg_speed: float
    min_speed: float
    max_speed: float
    free_flow_speed: float  # estimated from night/low-traffic speeds
    std_dev: float

    # Congestion metrics
    congestion_index: float  # 0=free flow, 1=completely congested
    worst_congestion_time: str
    worst_congestion_speed: float

    # Rush hour analysis
    am_rush_avg: float  # weekday 7-9 AM average
    pm_rush_avg: float  # weekday 5-7 PM average
    off_peak_avg: float  # weekday 10 AM - 3 PM average
    weekend_avg: float

    # Time lost
    am_delay_per_km_sec: float  # extra seconds per km vs free flow
    pm_delay_per_km_sec: float


def estimate_free_flow_speed(speeds: np.ndarray) -> float:
    """
    Estimate free-flow speed from the speed profile.

    Uses the 85th percentile of nighttime speeds (Sun-Thu 11 PM - 5 AM)
    as the free-flow estimate, which is standard practice in traffic engineering.
    """
    night_indices = []
    for day in range(7):
        # 11 PM to midnight
        start = day * INTERVALS_PER_DAY + 23 * 12
        end = day * INTERVALS_PER_DAY + INTERVALS_PER_DAY
        night_indices.extend(range(start, min(end, INTERVALS_PER_WEEK)))
        # midnight to 5 AM (next day)
        if day < 6:
            start2 = (day + 1) * INTERVALS_PER_DAY
            end2 = (day + 1) * INTERVALS_PER_DAY + 5 * 12
            night_indices.extend(range(start2, min(end2, INTERVALS_PER_WEEK)))

    night_speeds = speeds[night_indices]
    return float(np.percentile(night_speeds, 85))


def compute_congestion_index(speed: float, free_flow: float) -> float:
    """
    Compute congestion index: 0 = free flow, 1 = standstill.

    Formula: 1 - (actual_speed / free_flow_speed)
    Clamped to [0, 1].
    """
    if free_flow <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - speed / free_flow))


def delay_per_km(actual_speed: float, free_flow: float) -> float:
    """
    Calculate extra travel time per km in seconds compared to free flow.

    delay = (1/actual - 1/freeflow) * 3600 seconds per km
    """
    if actual_speed <= 0 or free_flow <= 0:
        return 0.0
    actual_time = 3600.0 / actual_speed  # seconds to travel 1 km
    free_time = 3600.0 / free_flow
    return max(0.0, actual_time - free_time)


def get_weekday_period_indices(start_hour: float, end_hour: float) -> list[int]:
    """Get indices for a time period across weekdays (Mon=1 through Fri=5)."""
    indices = []
    for day in range(1, 6):  # Mon-Fri (0=Sun)
        start_idx = day * INTERVALS_PER_DAY + int(start_hour * 12)
        end_idx = day * INTERVALS_PER_DAY + int(end_hour * 12)
        indices.extend(range(start_idx, end_idx))
    return indices


def get_weekend_indices() -> list[int]:
    """Get all indices for Saturday and Sunday."""
    indices = []
    for day in [0, 6]:  # Sun, Sat
        start = day * INTERVALS_PER_DAY
        end = start + INTERVALS_PER_DAY
        indices.extend(range(start, end))
    return indices


def analyze_segment(segment: RoadSegment, segment_id: int) -> SegmentAnalysis:
    """Perform full analysis on a single road segment."""
    speeds = segment.speeds

    # Overall stats
    avg_speed = float(np.mean(speeds))
    min_speed = float(np.min(speeds))
    max_speed = float(np.max(speeds))
    std_dev = float(np.std(speeds))
    free_flow = estimate_free_flow_speed(speeds)

    # Worst congestion point
    worst_idx = int(np.argmin(speeds))
    worst_time = index_to_time(worst_idx)
    worst_speed = float(speeds[worst_idx])
    congestion_index = compute_congestion_index(avg_speed, free_flow)

    # Rush hour averages
    am_indices = get_weekday_period_indices(7.0, 9.0)
    pm_indices = get_weekday_period_indices(17.0, 19.0)
    off_peak_indices = get_weekday_period_indices(10.0, 15.0)
    weekend_indices = get_weekend_indices()

    am_rush_avg = float(np.mean(speeds[am_indices]))
    pm_rush_avg = float(np.mean(speeds[pm_indices]))
    off_peak_avg = float(np.mean(speeds[off_peak_indices]))
    weekend_avg = float(np.mean(speeds[weekend_indices]))

    return SegmentAnalysis(
        segment=segment,
        segment_id=segment_id,
        avg_speed=avg_speed,
        min_speed=min_speed,
        max_speed=max_speed,
        free_flow_speed=free_flow,
        std_dev=std_dev,
        congestion_index=congestion_index,
        worst_congestion_time=worst_time,
        worst_congestion_speed=worst_speed,
        am_rush_avg=am_rush_avg,
        pm_rush_avg=pm_rush_avg,
        off_peak_avg=off_peak_avg,
        weekend_avg=weekend_avg,
        am_delay_per_km_sec=delay_per_km(am_rush_avg, free_flow),
        pm_delay_per_km_sec=delay_per_km(pm_rush_avg, free_flow),
    )


def analyze_all(segments: list[RoadSegment]) -> list[SegmentAnalysis]:
    """Analyze all segments and return sorted by congestion (worst first)."""
    analyses = [analyze_segment(seg, i) for i, seg in enumerate(segments)]
    analyses.sort(key=lambda a: a.congestion_index, reverse=True)
    return analyses


def generate_congestion_report(analyses: list[SegmentAnalysis]) -> str:
    """Generate a human-readable congestion report."""
    lines = []
    lines.append("=" * 70)
    lines.append("TRAFFIC CONGESTION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary statistics
    avg_ci = np.mean([a.congestion_index for a in analyses])
    lines.append(f"Total segments analyzed: {len(analyses)}")
    lines.append(f"Average congestion index: {avg_ci:.2f}")
    lines.append(f"Segments with heavy congestion (CI > 0.3): "
                 f"{sum(1 for a in analyses if a.congestion_index > 0.3)}")
    lines.append("")

    # Top 10 most congested
    lines.append("-" * 70)
    lines.append("TOP CONGESTED SEGMENTS")
    lines.append("-" * 70)
    lines.append(f"{'#':<4} {'Coords':<30} {'CI':<6} {'Avg':<6} {'Free':<6} "
                 f"{'AM Rush':<8} {'PM Rush':<8} {'Worst At'}")
    lines.append("-" * 70)

    for i, a in enumerate(analyses[:10]):
        if a.segment.points:
            coords = f"({a.segment.start.lat:.4f}, {a.segment.start.lon:.4f})"
        else:
            coords = "N/A"
        lines.append(
            f"{i+1:<4} {coords:<30} {a.congestion_index:.3f} "
            f"{a.avg_speed:>5.1f} {a.free_flow_speed:>5.1f} "
            f"{a.am_rush_avg:>7.1f} {a.pm_rush_avg:>7.1f} {a.worst_congestion_time}"
        )

    lines.append("")

    # Rush hour impact
    lines.append("-" * 70)
    lines.append("RUSH HOUR IMPACT ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"{'#':<4} {'Segment':<30} {'AM Delay':<12} {'PM Delay':<12} "
                 f"{'Worst Period'}")
    lines.append("-" * 70)

    delay_sorted = sorted(analyses, key=lambda a: max(a.am_delay_per_km_sec,
                                                       a.pm_delay_per_km_sec), reverse=True)
    for i, a in enumerate(delay_sorted[:10]):
        if a.segment.points:
            seg_desc = f"({a.segment.start.lat:.3f},{a.segment.start.lon:.3f})"
        else:
            seg_desc = f"Segment {a.segment_id}"
        worst = "AM" if a.am_delay_per_km_sec > a.pm_delay_per_km_sec else "PM"
        lines.append(
            f"{i+1:<4} {seg_desc:<30} "
            f"{a.am_delay_per_km_sec:>7.1f}s/km  {a.pm_delay_per_km_sec:>7.1f}s/km  "
            f"{worst} rush"
        )

    lines.append("")
    lines.append("-" * 70)
    lines.append("SPEED DISTRIBUTION")
    lines.append("-" * 70)
    all_avg_speeds = [a.avg_speed for a in analyses]
    lines.append(f"  Network average speed: {np.mean(all_avg_speeds):.1f} km/h")
    lines.append(f"  Fastest segment avg:   {max(all_avg_speeds):.1f} km/h")
    lines.append(f"  Slowest segment avg:   {min(all_avg_speeds):.1f} km/h")

    # Speed brackets
    brackets = [(0, 20, "Very Slow"), (20, 40, "Slow"), (40, 60, "Moderate"),
                (60, 80, "Fast"), (80, 200, "Very Fast")]
    lines.append("")
    for low, high, label in brackets:
        count = sum(1 for s in all_avg_speeds if low <= s < high)
        bar = "#" * count
        lines.append(f"  {label:<12} ({low:>3}-{high:>3} km/h): {count:>3} segments {bar}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)

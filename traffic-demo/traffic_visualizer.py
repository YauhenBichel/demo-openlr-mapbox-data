"""
Traffic data visualization module.

Generates:
1. Interactive Folium map with color-coded road segments by speed/congestion
2. Matplotlib weekly speed profile charts
3. Rush hour comparison charts
"""

from pathlib import Path

import folium
import branca.colormap as cm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from traffic_parser import RoadSegment, INTERVALS_PER_DAY, DAYS, index_to_time
from traffic_analysis import SegmentAnalysis


# Speed → color mapping (traffic-style: green=fast, red=slow)
SPEED_COLORMAP = cm.LinearColormap(
    colors=['#8B0000', '#FF0000', '#FF6600', '#FFCC00', '#00CC00', '#006600'],
    index=[0, 20, 35, 50, 70, 100],
    vmin=0,
    vmax=100,
    caption='Speed (km/h)',
)

CONGESTION_COLORMAP = cm.LinearColormap(
    colors=['#006600', '#00CC00', '#FFCC00', '#FF6600', '#FF0000', '#8B0000'],
    index=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
    vmin=0.0,
    vmax=0.7,
    caption='Congestion Index (0=free, 1=standstill)',
)


def create_traffic_map(analyses: list[SegmentAnalysis],
                       output_path: Path,
                       mode: str = "speed") -> Path:
    """
    Create an interactive Folium map with color-coded road segments.

    Args:
        analyses: List of segment analysis results.
        output_path: Where to save the HTML file.
        mode: "speed" for average speed coloring, "congestion" for congestion index.
    """
    # Find map center from segment coordinates
    lats, lons = [], []
    for a in analyses:
        if a.segment.points:
            lats.append(a.segment.start.lat)
            lons.append(a.segment.start.lon)
            lats.append(a.segment.end.lat)
            lons.append(a.segment.end.lon)

    if not lats:
        print("No segments with coordinates to map.")
        return output_path

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles='cartodbpositron')

    colormap = SPEED_COLORMAP if mode == "speed" else CONGESTION_COLORMAP

    for a in analyses:
        if not a.segment.points:
            continue

        coords = [(p.lat, p.lon) for p in a.segment.points]

        if mode == "speed":
            color = colormap(a.avg_speed)
            weight = 4
        else:
            color = colormap(min(a.congestion_index, 0.7))
            weight = 3 + int(a.congestion_index * 8)  # thicker = more congested

        # Popup with detailed info
        popup_html = f"""
        <div style="font-family: monospace; font-size: 12px; width: 280px;">
            <b>Segment {a.segment_id}</b><br>
            <hr style="margin: 4px 0;">
            <b>Speeds:</b><br>
            &nbsp; Average: {a.avg_speed:.1f} km/h<br>
            &nbsp; Free flow: {a.free_flow_speed:.1f} km/h<br>
            &nbsp; Range: {a.min_speed:.0f} – {a.max_speed:.0f} km/h<br>
            <br>
            <b>Rush Hour:</b><br>
            &nbsp; AM (7-9): {a.am_rush_avg:.1f} km/h
                (delay: +{a.am_delay_per_km_sec:.1f}s/km)<br>
            &nbsp; PM (5-7): {a.pm_rush_avg:.1f} km/h
                (delay: +{a.pm_delay_per_km_sec:.1f}s/km)<br>
            &nbsp; Off-peak: {a.off_peak_avg:.1f} km/h<br>
            &nbsp; Weekend: {a.weekend_avg:.1f} km/h<br>
            <br>
            <b>Congestion Index:</b> {a.congestion_index:.3f}<br>
            <b>Worst time:</b> {a.worst_congestion_time}
                ({a.worst_congestion_speed:.0f} km/h)<br>
            <br>
            <b>FRC:</b> {a.segment.start.frc}
            &nbsp; <b>Length:</b> {a.segment.length_km:.2f} km<br>
            <b>OpenLR:</b> <code>{a.segment.openlr_raw[:30]}...</code>
        </div>
        """

        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Seg {a.segment_id}: {a.avg_speed:.0f} km/h (CI: {a.congestion_index:.2f})",
        ).add_to(m)

        # Add markers at segment endpoints
        folium.CircleMarker(
            coords[0],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.9,
        ).add_to(m)

    # Add colormap legend
    colormap.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Traffic map saved: {output_path}")
    return output_path


def plot_speed_profiles(analyses: list[SegmentAnalysis],
                        output_path: Path,
                        top_n: int = 6) -> Path:
    """
    Plot weekly speed profiles for the most congested segments.

    Shows 2,016 data points (7 days × 288 intervals) per segment.
    """
    # Select top N most congested segments
    selected = analyses[:top_n]

    fig, axes = plt.subplots(len(selected), 1, figsize=(16, 3 * len(selected)),
                              sharex=True)
    if len(selected) == 1:
        axes = [axes]

    x = np.arange(INTERVALS_PER_DAY * 7)

    for ax, a in zip(axes, selected):
        speeds = a.segment.speeds

        # Color the line by congestion level (green to red)
        for day in range(7):
            day_start = day * INTERVALS_PER_DAY
            day_end = day_start + INTERVALS_PER_DAY
            day_x = x[day_start:day_end]
            day_speeds = speeds[day_start:day_end]

            is_weekend = day in (0, 6)
            color = '#2196F3' if is_weekend else '#333333'
            alpha = 0.6 if is_weekend else 0.8
            ax.plot(day_x, day_speeds, color=color, alpha=alpha, linewidth=0.8)

        # Highlight rush hours (weekday only)
        for day in range(1, 6):  # Mon-Fri
            am_start = day * INTERVALS_PER_DAY + 7 * 12
            am_end = day * INTERVALS_PER_DAY + 9 * 12
            pm_start = day * INTERVALS_PER_DAY + 17 * 12
            pm_end = day * INTERVALS_PER_DAY + 19 * 12
            ax.axvspan(am_start, am_end, alpha=0.08, color='red')
            ax.axvspan(pm_start, pm_end, alpha=0.08, color='red')

        # Reference lines
        ax.axhline(y=a.free_flow_speed, color='green', linestyle='--',
                    alpha=0.5, linewidth=1, label=f'Free flow: {a.free_flow_speed:.0f}')
        ax.axhline(y=a.avg_speed, color='orange', linestyle=':',
                    alpha=0.5, linewidth=1, label=f'Average: {a.avg_speed:.0f}')

        # Labels
        if a.segment.points:
            title = (f"Segment {a.segment_id} "
                     f"({a.segment.start.lat:.4f}, {a.segment.start.lon:.4f}) — "
                     f"CI: {a.congestion_index:.3f}")
        else:
            title = f"Segment {a.segment_id} — CI: {a.congestion_index:.3f}"

        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        ax.set_ylabel('km/h', fontsize=9)
        ax.set_ylim(0, a.free_flow_speed * 1.15)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    # X-axis labels: day names
    day_positions = [d * INTERVALS_PER_DAY + INTERVALS_PER_DAY // 2 for d in range(7)]
    axes[-1].set_xticks([d * INTERVALS_PER_DAY for d in range(8)])
    axes[-1].set_xticklabels(DAYS + [DAYS[0]], fontsize=9)
    axes[-1].set_xlabel('Day of Week (shaded = rush hour, blue = weekend)', fontsize=9)

    plt.suptitle('Weekly Speed Profiles — Most Congested Segments',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Speed profiles saved: {output_path}")
    return output_path


def plot_rush_hour_comparison(analyses: list[SegmentAnalysis],
                              output_path: Path,
                              top_n: int = 10) -> Path:
    """Bar chart comparing AM rush, PM rush, off-peak, and weekend speeds."""
    selected = sorted(analyses, key=lambda a: a.congestion_index, reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(14, 6))

    labels = []
    for a in selected:
        if a.segment.points:
            labels.append(f"Seg {a.segment_id}\n({a.segment.start.lat:.3f})")
        else:
            labels.append(f"Seg {a.segment_id}")

    x = np.arange(len(selected))
    width = 0.2

    am_speeds = [a.am_rush_avg for a in selected]
    pm_speeds = [a.pm_rush_avg for a in selected]
    offpeak_speeds = [a.off_peak_avg for a in selected]
    weekend_speeds = [a.weekend_avg for a in selected]

    bars1 = ax.bar(x - 1.5 * width, am_speeds, width, label='AM Rush (7-9)',
                   color='#FF6B6B', edgecolor='white')
    bars2 = ax.bar(x - 0.5 * width, pm_speeds, width, label='PM Rush (5-7)',
                   color='#CC3333', edgecolor='white')
    bars3 = ax.bar(x + 0.5 * width, offpeak_speeds, width, label='Off-Peak (10-3)',
                   color='#4ECDC4', edgecolor='white')
    bars4 = ax.bar(x + 1.5 * width, weekend_speeds, width, label='Weekend',
                   color='#45B7D1', edgecolor='white')

    # Add free-flow markers
    for i, a in enumerate(selected):
        ax.plot([i - 2 * width, i + 2 * width],
                [a.free_flow_speed, a.free_flow_speed],
                'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Segment')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Rush Hour vs Off-Peak Speed Comparison (Most Congested Segments)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Rush hour comparison saved: {output_path}")
    return output_path


def plot_congestion_heatmap(analyses: list[SegmentAnalysis],
                            output_path: Path,
                            top_n: int = 10) -> Path:
    """
    Heatmap showing congestion level by segment × time of day.

    Rows = segments, Columns = hours of the day (averaged across weekdays).
    """
    selected = sorted(analyses, key=lambda a: a.congestion_index, reverse=True)[:top_n]

    # Build heatmap data: weekday average speed per hour
    hours = 24
    heatmap = np.zeros((len(selected), hours))

    for i, a in enumerate(selected):
        for h in range(hours):
            # Average across Mon-Fri for this hour
            hour_speeds = []
            for day in range(1, 6):
                start = day * INTERVALS_PER_DAY + h * 12
                end = start + 12
                hour_speeds.extend(a.segment.speeds[start:end])
            avg_speed = np.mean(hour_speeds)
            # Convert to congestion ratio
            heatmap[i, h] = 1.0 - (avg_speed / a.free_flow_speed) if a.free_flow_speed > 0 else 0

    fig, ax = plt.subplots(figsize=(14, max(4, len(selected) * 0.6)))

    im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.6)

    # Labels
    ylabels = [f"Seg {a.segment_id}" for a in selected]
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_xlabel('Hour of Day (Weekday Average)')
    ax.set_title('Congestion Heatmap by Segment × Hour', fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Congestion (0=free, 1=standstill)')

    # Highlight rush hours
    ax.axvline(x=6.5, color='white', linestyle=':', alpha=0.5)
    ax.axvline(x=9.5, color='white', linestyle=':', alpha=0.5)
    ax.axvline(x=16.5, color='white', linestyle=':', alpha=0.5)
    ax.axvline(x=19.5, color='white', linestyle=':', alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Congestion heatmap saved: {output_path}")
    return output_path

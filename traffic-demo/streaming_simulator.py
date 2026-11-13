"""
Real-time traffic telemetry streaming simulator.

Simulates the full Mapbox Traffic pipeline without external dependencies:
1. TelemetryGenerator — vehicles driving routes, emitting noisy GPS pings
2. StreamProcessor   — consumes pings, runs map matching, aggregates speeds
3. StreamingDemo     — orchestrates pipeline with live console dashboard

Architecture mirrors what a production system would do:
- TelemetryGenerator → Kafka producer (mobile SDK → Kinesis in Mapbox)
- queue.Queue        → Kafka topic (bounded buffer = back-pressure)
- StreamProcessor    → Flink job (map match → aggregate → emit)
- current_speeds     → Redis/DynamoDB (serving layer)

Demonstrates: producer-consumer, back-pressure, thread pools, rolling
time windows, concurrent data structures — all patterns relevant to
the Mapbox SDE II Traffic position.
"""

import math
import queue
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from map_matching import (
    GPSPoint, HMMMapMatcher, MatchResult, SimpleRoadNetwork,
    build_demo_matcher,
)
from sample_data import DC_SEGMENTS


@dataclass
class SpeedObservation:
    edge_id: int
    speed_kmh: float
    timestamp: float


class TelemetryGenerator:
    """
    Simulates vehicles driving along road segments, emitting GPS pings.

    Each vehicle:
    1. Picks a random road segment from the network
    2. Drives along it at a speed with some variation
    3. Emits GPS pings at regular intervals with configurable noise

    Writes pings to a bounded queue. If the queue is full (consumer can't
    keep up), pings are dropped — this is the back-pressure mechanism.
    In production, this maps to Kafka producer with a bounded buffer.
    """

    def __init__(self, output_queue: queue.Queue,
                 num_vehicles: int = 10,
                 ping_interval_sec: float = 1.0,
                 gps_noise_m: float = 8.0,
                 seed: int = 42):
        self.queue = output_queue
        self.num_vehicles = num_vehicles
        self.ping_interval = ping_interval_sec
        self.gps_noise_deg = gps_noise_m / 111_000  # rough m to deg
        self.rng = np.random.default_rng(seed)
        self._stop = threading.Event()
        self.generated_count = 0
        self.dropped_count = 0

    def stop(self):
        self._stop.set()

    def _drive_vehicle(self, vehicle_id: str):
        """Simulate one vehicle driving along a random segment."""
        while not self._stop.is_set():
            # Pick a random segment
            seg = DC_SEGMENTS[self.rng.integers(len(DC_SEGMENTS))]
            lon1, lat1, lon2, lat2, name, free_flow, road_type = seg

            # Drive speed with variation
            speed_kmh = free_flow * self.rng.uniform(0.6, 1.0)
            speed_deg_per_sec = (speed_kmh / 3600) / 111.0  # rough conversion

            # Number of pings along this segment
            seg_length_deg = math.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)
            if speed_deg_per_sec <= 0:
                continue
            travel_time = seg_length_deg / speed_deg_per_sec
            num_pings = max(2, int(travel_time / self.ping_interval))

            for i in range(num_pings):
                if self._stop.is_set():
                    return

                frac = i / max(1, num_pings - 1)
                true_lon = lon1 + frac * (lon2 - lon1)
                true_lat = lat1 + frac * (lat2 - lat1)

                # Add GPS noise
                noisy_lon = true_lon + self.rng.normal(0, self.gps_noise_deg)
                noisy_lat = true_lat + self.rng.normal(0, self.gps_noise_deg)

                ping = GPSPoint(
                    lon=noisy_lon,
                    lat=noisy_lat,
                    timestamp=time.time(),
                    speed_kmh=speed_kmh + self.rng.normal(0, 3),
                    vehicle_id=vehicle_id,
                )

                try:
                    self.queue.put_nowait(ping)
                    self.generated_count += 1
                except queue.Full:
                    self.dropped_count += 1

                time.sleep(self.ping_interval)

    def start(self) -> list[threading.Thread]:
        """Start vehicle simulation threads."""
        threads = []
        for i in range(self.num_vehicles):
            t = threading.Thread(
                target=self._drive_vehicle,
                args=(f"vehicle_{i:03d}",),
                daemon=True,
            )
            t.start()
            threads.append(t)
        return threads


class StreamProcessor:
    """
    Consumes GPS pings, runs map matching, and aggregates rolling speed windows.

    Architecture:
    1. Drain batch from queue (= Kafka consumer poll)
    2. Group by vehicle_id to form traces
    3. Map match each trace (parallel via thread pool)
    4. Feed matched results into speed aggregation windows
    5. Update current_speeds (= write to Redis)

    Uses a ThreadPoolExecutor for parallel map matching — CPU-bound work.
    Uses threading.Lock to protect the shared speed window state.
    """

    def __init__(self, matcher: HMMMapMatcher,
                 window_size_sec: float = 60.0,
                 batch_size: int = 50,
                 min_trace_length: int = 3):
        self.matcher = matcher
        self.window_size = window_size_sec
        self.batch_size = batch_size
        self.min_trace_length = min_trace_length

        # Per-vehicle ping buffers — accumulate across batch drains
        self._vehicle_buffers: dict[str, list[GPSPoint]] = defaultdict(list)
        self._buffer_lock = threading.Lock()

        # Rolling speed windows per segment
        self._speed_windows: dict[int, deque[SpeedObservation]] = defaultdict(
            lambda: deque(maxlen=200)
        )
        self._lock = threading.Lock()

        # Current aggregated speeds (read by serving layer)
        self.current_speeds: dict[int, float] = {}

        # Metrics
        self.processed_count = 0
        self.matched_count = 0
        self.match_pool = ThreadPoolExecutor(max_workers=4)

    def process_batch(self, pings: list[GPSPoint]):
        """Process a batch of GPS pings through the pipeline."""
        if not pings:
            return

        # Accumulate pings into per-vehicle buffers
        with self._buffer_lock:
            for ping in pings:
                self._vehicle_buffers[ping.vehicle_id].append(ping)

            # Collect traces that have enough points for map matching
            ready_traces: list[list[GPSPoint]] = []
            for vehicle_id, buffer in self._vehicle_buffers.items():
                if len(buffer) >= self.min_trace_length:
                    # Sort by timestamp and take the trace
                    buffer.sort(key=lambda p: p.timestamp)
                    ready_traces.append(list(buffer))
                    # Keep the last point as overlap for continuity
                    self._vehicle_buffers[vehicle_id] = buffer[-1:]

        # Map match each ready trace (parallel)
        futures = []
        for trace in ready_traces:
            futures.append(self.match_pool.submit(self._match_trace, trace))

        # Collect results
        for future in futures:
            try:
                results = future.result(timeout=5.0)
                for r in results:
                    self._add_speed_observation(r)
            except Exception:
                pass

        self.processed_count += len(pings)

    def _match_trace(self, trace: list[GPSPoint]) -> list[MatchResult]:
        """Map match a single vehicle's trace."""
        results = self.matcher.match(trace)
        self.matched_count += len(results)
        return results

    def _add_speed_observation(self, result: MatchResult):
        """Add a speed observation to the rolling window for a segment."""
        if result.gps_point.speed_kmh <= 0:
            return

        obs = SpeedObservation(
            edge_id=result.edge_id,
            speed_kmh=result.gps_point.speed_kmh,
            timestamp=result.gps_point.timestamp,
        )

        with self._lock:
            window = self._speed_windows[result.edge_id]
            window.append(obs)

            # Evict old observations
            now = time.time()
            while window and now - window[0].timestamp > self.window_size:
                window.popleft()

            # Recompute average speed for this segment
            if window:
                speeds = [o.speed_kmh for o in window]
                self.current_speeds[result.edge_id] = sum(speeds) / len(speeds)

    def get_all_speeds(self) -> dict[int, float]:
        """Get current speed estimates for all segments (thread-safe read)."""
        with self._lock:
            return dict(self.current_speeds)


class StreamingDemo:
    """
    Orchestrates the full streaming pipeline with a live console dashboard.

    Shows real-time metrics: throughput, queue depth, speeds per segment,
    match rate — demonstrating operational visibility into the pipeline.
    """

    def __init__(self, num_vehicles: int = 8, duration_sec: int = 30):
        self.num_vehicles = num_vehicles
        self.duration = duration_sec

    def run(self):
        print("=" * 65)
        print("  STREAMING TRAFFIC PIPELINE SIMULATOR")
        print("=" * 65)
        print(f"\n  Vehicles:     {self.num_vehicles}")
        print(f"  Duration:     {self.duration}s")
        print(f"  Architecture: Generators → Queue → MapMatcher → Aggregator")
        print()

        # Build the pipeline components
        print("Initializing road network and map matcher...")
        network, spatial, matcher = build_demo_matcher()

        # Bounded queue (= Kafka topic)
        telemetry_queue = queue.Queue(maxsize=500)

        # Generator (= mobile SDK telemetry)
        generator = TelemetryGenerator(
            telemetry_queue,
            num_vehicles=self.num_vehicles,
            ping_interval_sec=0.5,
        )

        # Processor (= Flink job)
        processor = StreamProcessor(matcher, window_size_sec=30.0)

        # Start generators
        print(f"Starting {self.num_vehicles} vehicle simulators...")
        gen_threads = generator.start()

        # Processing loop with live dashboard
        print("\nPipeline running. Live dashboard:\n")
        start_time = time.time()

        try:
            while time.time() - start_time < self.duration:
                # Drain batch from queue
                batch = []
                for _ in range(processor.batch_size):
                    try:
                        ping = telemetry_queue.get_nowait()
                        batch.append(ping)
                    except queue.Empty:
                        break

                # Process the batch
                if batch:
                    processor.process_batch(batch)

                # Print dashboard every second
                elapsed = time.time() - start_time
                if int(elapsed * 2) % 2 == 0:
                    self._print_dashboard(
                        elapsed, generator, processor, telemetry_queue,
                        network,
                    )

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            generator.stop()

        # Final summary
        self._print_summary(generator, processor, network)

    def _print_dashboard(self, elapsed: float,
                         generator: TelemetryGenerator,
                         processor: StreamProcessor,
                         q: queue.Queue,
                         network: SimpleRoadNetwork):
        """Print a live dashboard line."""
        speeds = processor.get_all_speeds()
        throughput = processor.processed_count / max(elapsed, 0.1)
        match_rate = (processor.matched_count / max(processor.processed_count, 1)) * 100
        queue_pct = q.qsize() / q.maxsize * 100

        line = (
            f"\r  [{elapsed:5.1f}s] "
            f"Ingested: {generator.generated_count:>5} | "
            f"Processed: {processor.processed_count:>5} | "
            f"Queue: {queue_pct:>4.0f}% | "
            f"Throughput: {throughput:>5.1f}/s | "
            f"Match rate: {match_rate:>4.0f}% | "
            f"Segments w/data: {len(speeds):>3}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def _print_summary(self, generator: TelemetryGenerator,
                       processor: StreamProcessor,
                       network: SimpleRoadNetwork):
        """Print final pipeline summary."""
        speeds = processor.get_all_speeds()

        print("\n\n" + "=" * 65)
        print("  PIPELINE SUMMARY")
        print("=" * 65)
        print(f"  Total generated:  {generator.generated_count}")
        print(f"  Total dropped:    {generator.dropped_count} "
              f"({generator.dropped_count / max(generator.generated_count, 1) * 100:.1f}%)")
        print(f"  Total processed:  {processor.processed_count}")
        print(f"  Total matched:    {processor.matched_count}")
        print(f"  Segments w/data:  {len(speeds)} / {len(network.edges)}")

        if speeds:
            print(f"\n  {'Edge':<6} {'Name':<30} {'Speed (km/h)'}")
            print("  " + "-" * 55)
            for edge_id, speed in sorted(speeds.items()):
                edge = network.edges.get(edge_id)
                name = edge.name if edge else "?"
                limit = edge.speed_limit if edge else 0
                bar_len = int(speed / 2)
                bar = "#" * bar_len
                print(f"  {edge_id:<6} {name:<30} {speed:>5.1f} / {limit:.0f}  {bar}")

        print("\n" + "=" * 65)


if __name__ == "__main__":
    demo = StreamingDemo(num_vehicles=8, duration_sec=30)
    demo.run()

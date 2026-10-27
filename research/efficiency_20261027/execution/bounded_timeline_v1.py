"""Bounded Chrome activity sample and complete-recording interval attribution.

Summaries use every recorded CUDA interval. The Chrome view is explicitly
sampled across the recording; it cannot establish full CUDA idle coverage.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
import json
import math
import time
import torch
from research.efficiency_20261027.root.compare_host_parallelism import need


def merged_intervals(intervals, start, end):
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals if b > start and a < end and b > a)
    merged = []
    for a, b in clipped:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def interval_summary(intervals, start, end):
    merged = merged_intervals(intervals, start, end)
    cursor = start
    gaps = []
    for a, b in merged:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = b
    if cursor < end:
        gaps.append((cursor, end))
    busy = sum(b-a for a, b in merged)
    return dict(recorded_busy_seconds=busy/1e9, recording_window_seconds=(end-start)/1e9,
        recorded_busy_fraction=busy/(end-start) if end > start else None,
        gap_count=len(gaps), largest_gaps=[dict(start_us=(a-start)/1000, duration_us=(b-a)/1000)
            for a, b in sorted(gaps, key=lambda p: p[1]-p[0], reverse=True)[:20]])


def activity_kind(event):
    if 'CUDA' not in str(event.device_type()):
        return 'CPU'
    name = event.name().lower()
    if 'memcpy' in name:
        if 'htod' in name or 'h2d' in name:
            return 'H2D'
        if 'dtoh' in name or 'd2h' in name:
            return 'D2H'
        return 'other_copy'
    if 'memset' in name:
        return 'memset'
    return 'kernel'


class DiagnosticProfile:
    """Host-stage clocks only; no added stage synchronization or peak reset."""
    def __init__(self, tracing):
        self.tracing = tracing
        self.timings = {}
        self.marks = Counter()

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        annotation = torch.profiler.record_function('treeloom/' + name) if self.tracing else nullcontext()
        try:
            with annotation:
                yield
        finally:
            duration = time.perf_counter()-started
            row = self.timings.setdefault(name, dict(calls=0, host_seconds=0., maximum_host_seconds=0.))
            row['calls'] += 1
            row['host_seconds'] += duration
            row['maximum_host_seconds'] = max(row['maximum_host_seconds'], duration)

    def mark(self, name, **values):
        self.marks[name] += 1

    def snapshot(self):
        return dict(stage_host_timings=self.timings, mark_counts=dict(self.marks),
            stage_clock_is_not_CUDA_elapsed=True, added_stage_synchronization=False)


def save_timeline(profile, path, *, max_bytes=12*2**20, event_quota=30000):
    need(type(max_bytes) is int and 1024 <= max_bytes <= 12*2**20
         and type(event_quota) is int and 4 <= event_quota <= 30000, 'Finite explicit trace bounds')
    events = profile.profiler.kineto_results.events()
    roots = [event for event in events if event.name() == 'diagnostic_learned_update']
    need(len(roots) == 1, 'One complete learned forward/decoder/backward recording window')
    start, end = roots[0].start_ns(), roots[0].end_ns()
    need(end > start, 'Positive complete recorded interval')
    counts = Counter()
    intervals = defaultdict(list)
    duration_by_name = defaultdict(float)
    count_by_name = Counter()
    copy_bytes = Counter()
    copy_bytes_unknown = Counter()
    for event in events:
        kind = activity_kind(event)
        counts[kind] += 1
        name = event.name()
        duration_by_name[(kind, name)] += event.duration_ns()/1e9
        count_by_name[(kind, name)] += 1
        if kind != 'CPU':
            intervals[kind].append((event.start_ns(), event.end_ns()))
            if kind in ('H2D', 'D2H', 'other_copy'):
                amount = int(event.nbytes())
                if amount > 0:
                    copy_bytes[kind] += amount
                else:
                    copy_bytes_unknown[kind] += 1
    all_cuda = [interval for group in intervals.values() for interval in group]
    occupancy = interval_summary(all_cuda, start, end)
    summaries = {kind: interval_summary(values, start, end) for kind, values in intervals.items()}
    # Half the view budget is CPU; half is CUDA. Sample each across its entire
    # recording. Always keep the root and stage annotations when the byte cap allows.
    cpu_stride = max(1, math.ceil(counts['CPU']/max(1, event_quota//2)))
    gpu_count = sum(value for kind, value in counts.items() if kind != 'CPU')
    gpu_stride = max(1, math.ceil(gpu_count/max(1, event_quota//2)))
    indices = Counter()
    kept = Counter()
    skipped_bytes = 0
    prefix = b'{"displayTimeUnit":"ms","traceEvents":['
    suffix = b']}'
    used = len(prefix)
    emitted = 0
    with path.open('xb') as stream:
        stream.write(prefix)
        for event in events:
            kind = activity_kind(event)
            group = 'CPU' if kind == 'CPU' else 'CUDA'
            index = indices[group]
            indices[group] += 1
            annotation = event.is_user_annotation() and (event.name().startswith('treeloom/')
                or event.name() == 'diagnostic_learned_update')
            stride = cpu_stride if group == 'CPU' else gpu_stride
            if not annotation and index % stride:
                continue
            row = dict(name=event.name(), cat=kind, ph='X', ts=(event.start_ns()-start)/1000,
                dur=event.duration_ns()/1000, pid=1 if kind == 'CPU' else 1000+event.device_index(),
                tid=event.start_thread_id() if kind == 'CPU' else event.device_resource_id(),
                args=dict(correlation_id=event.correlation_id(), linked_correlation_id=event.linked_correlation_id()))
            raw = ((',' if emitted else '')+json.dumps(row, separators=(',', ':'), allow_nan=False)).encode()
            if used+len(raw)+len(suffix) > max_bytes:
                skipped_bytes += 1
                continue
            stream.write(raw)
            used += len(raw)
            emitted += 1
            kept[kind] += 1
        stream.write(suffix)
    need(path.stat().st_size <= max_bytes, 'Trace output byte cap')
    top = sorted(duration_by_name, key=duration_by_name.get, reverse=True)
    return dict(recorded_event_counts=dict(counts), recorded_CUDA_union=occupancy,
        recorded_activity_unions=summaries, reported_copy_bytes=dict(copy_bytes),
        copy_events_without_positive_byte_metadata=dict(copy_bytes_unknown),
        top_CPU_events=[dict(name=name, calls=count_by_name[(kind, name)], summed_duration_seconds=duration_by_name[(kind, name)])
            for kind, name in top if kind == 'CPU'][:30],
        top_CUDA_events=[dict(kind=kind, name=name, calls=count_by_name[(kind, name)], summed_duration_seconds=duration_by_name[(kind, name)])
            for kind, name in top if kind != 'CPU'][:30],
        Chrome_trace=dict(path=str(path), max_bytes=max_bytes, actual_bytes=path.stat().st_size,
            kept_event_counts=dict(kept), cpu_stride=cpu_stride, cuda_stride=gpu_stride,
            skipped_due_to_byte_cap=skipped_bytes, sampled=True),
        limitations=['Union attribution uses all recorded activities, while the bounded Chrome file samples events across the recording.',
            'Recorded CUDA gaps describe this process in the profiled window, not whole-device utilization or proof of CPU causation.',
            'Torch profiler overhead can change gaps and duration. Profiled calls are excluded from performance medians.',
            'Summed CPU durations overlap nested events and must not be interpreted as wall time or CPU busy fraction.',
            'Copy bytes are emitted only from positive Kineto byte metadata; missing metadata is counted explicitly.'])

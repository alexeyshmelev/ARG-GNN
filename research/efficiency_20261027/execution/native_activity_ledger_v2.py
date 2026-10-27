"""Stream native Kineto categories into a finite complete physical-CUDA ledger.

Only native categories kernel, gpu_memcpy and gpu_memset enter activity unions.
Annotation and synthetic ranges are counted separately and never fill gaps.
The reader runs as one visible project-Python subprocess of its own profiler.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
PHYSICAL = frozenset(('kernel', 'gpu_memcpy', 'gpu_memset'))
LIMIT = 8*2**20


def need(ok, message):
    if not ok:
        raise ValueError(message)


def native_events(stream):
    """Parse one bounded JSON event at a time; never retain the native trace."""
    buffer = ''
    while '"traceEvents"' not in buffer:
        chunk = stream.read(65536)
        need(bool(chunk), 'Native traceEvents array required')
        buffer += chunk
        need(len(buffer) <= 2**20, 'Bounded native trace header')
    index = buffer.index('"traceEvents"') + len('"traceEvents"')
    while '[' not in buffer[index:]:
        chunk = stream.read(65536)
        need(bool(chunk), 'Native event array begins')
        buffer += chunk
    buffer = buffer[buffer.index('[', index)+1:]
    decoder = json.JSONDecoder(parse_float=Decimal)
    while True:
        buffer = buffer.lstrip()
        if buffer.startswith(','):
            buffer = buffer[1:].lstrip()
        if buffer.startswith(']'):
            while stream.read(65536):
                pass
            return
        try:
            value, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk = stream.read(65536)
            need(bool(chunk), 'Complete native trace event required')
            buffer += chunk
            need(len(buffer) <= 2*2**20, 'Individual native event exceeds2MiB bound')
            continue
        need(isinstance(value, dict), 'Native event object required')
        yield value
        buffer = buffer[end:]


def native_kind(event):
    category = event.get('cat', '')
    if category not in PHYSICAL or event.get('ph') != 'X':
        return None
    if category == 'kernel':
        return 'kernel'
    if category == 'gpu_memset':
        return 'memset'
    name = event.get('name', '').lower()
    if 'htod' in name or 'h2d' in name:
        return 'H2D'
    if 'dtoh' in name or 'd2h' in name:
        return 'D2H'
    return 'other_copy'


def interval_union(intervals, start, end):
    spans = sorted((max(a, start), min(b, end)) for a, b in intervals if b > start and a < end and b > a)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    cursor = start
    gaps = []
    for a, b in merged:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = b
    if cursor < end:
        gaps.append((cursor, end))
    busy = sum(b-a for a, b in merged)
    return dict(busy_seconds=busy/1e9, window_seconds=(end-start)/1e9,
        busy_fraction=busy/(end-start), gaps=len(gaps),
        largest_gaps=[dict(start_us=(a-start)/1000, duration_us=(b-a)/1000)
            for a, b in sorted(gaps, key=lambda z: z[1]-z[0], reverse=True)[:20]])


def consume(stream, ledger_path, summary_path):
    categories, excluded = Counter(), Counter()
    excluded_names = Counter()
    physical, names, name_indices = [], [], {}
    roots = []
    charged = 0
    for event in native_events(stream):
        cat = event.get('cat', '')
        categories[cat] += 1
        kind = native_kind(event)
        if event.get('name') == 'diagnostic_learned_update' and cat == 'user_annotation' and event.get('ph') == 'X':
            roots.append((int(Decimal(event['ts'])*1000), int(Decimal(event['dur'])*1000)))
        if kind is None:
            excluded[cat] += 1
            if cat == 'gpu_user_annotation':
                excluded_names[event.get('name', '')] += 1
            need(not (cat.startswith('gpu_') and cat not in ('gpu_user_annotation',)),
                 'Unreviewed native GPU category: '+cat)
            continue
        name = event['name']
        if name not in name_indices:
            need(len(names) < 2048, 'Bounded native kernel/copy name table')
            name_indices[name] = len(names)
            names.append(name)
            charged += len(name.encode())+16
        args = event.get('args', {})
        start_ns, duration_ns = int(Decimal(event['ts'])*1000), int(Decimal(event['dur'])*1000)
        need(duration_ns >= 0, 'Finite nonnegative physical activity duration')
        row = [kind, start_ns, duration_ns, name_indices[name], int(args.get('device', event.get('pid', -1))),
            int(args.get('stream', event.get('tid', -1))), int(args.get('correlation', -1)),
            int(args.get('bytes', 0))]
        charged += len(json.dumps(row, separators=(',', ':')))+1
        need(charged <= LIMIT-65536 and len(physical) < 200000, 'Complete physical CUDA ledger cap exceeded')
        physical.append(row)
    need(len(roots) == 1 and roots[0][1] > 0, 'One native CPU learned annotation window')
    start, duration = roots[0]
    end = start+duration
    intervals = defaultdict(list)
    counts, byte_counts, missing_bytes = Counter(), Counter(), Counter()
    totals = defaultdict(float)
    for row in physical:
        kind, when, duration, name_index, device, stream, correlation, nbytes = row
        intervals[kind].append((when, when+duration))
        counts[kind] += 1
        totals[(kind, name_index)] += duration/1e9
        if kind in ('H2D','D2H','other_copy'):
            if nbytes > 0:
                byte_counts[kind] += nbytes
            else:
                missing_bytes[kind] += 1
        row[1] -= start
    ledger = dict(schema='TreeLoom_native_physical_CUDA_ledger_v2', complete_physical_event_ledger=True,
        native_category_allowlist=sorted(PHYSICAL), excluded_native_categories=dict(excluded),
        excluded_GPU_annotation_names=dict(excluded_names), window_start_native_ns=start,
        window_duration_ns=end-start, names=names,
        fields=['kind','start_relative_ns','duration_ns','name_index','device','stream','correlation','bytes'], events=physical)
    raw = json.dumps(ledger, separators=(',', ':'), allow_nan=False).encode()
    need(len(raw) <= LIMIT, 'Final complete ledger byte cap')
    with ledger_path.open('xb') as output:
        output.write(raw)
    all_intervals = [span for group in intervals.values() for span in group]
    top = sorted(totals, key=totals.get, reverse=True)[:30]
    summary = dict(schema='TreeLoom_native_activity_summary_v2', classification='explicit_native_category_allowlist',
        native_category_counts=dict(categories), excluded_category_counts=dict(excluded),
        excluded_GPU_annotation_names=dict(excluded_names), physical_event_counts=dict(counts),
        physical_CUDA_union=interval_union(all_intervals,start,end),
        physical_category_unions={kind:interval_union(values,start,end) for kind,values in intervals.items()},
        top_physical_events=[dict(kind=kind,name=names[index],summed_seconds=totals[(kind,index)]) for kind,index in top],
        reported_copy_bytes=dict(byte_counts), copy_events_without_positive_byte_metadata=dict(missing_bytes),
        complete_ledger_bytes=len(raw), complete_ledger_max_bytes=LIMIT,
        limitations=['Activity union covers recorded physical events of this process only, not device-wide utilization.',
            'Profiler overhead can change timing and gaps. These calls are not speed measurements.',
            'GPU user annotations and every non-allowlisted native category are excluded from physical unions.'])
    with summary_path.open('x') as output:
        json.dump(summary,output,indent=2,allow_nan=False)
    return summary


def export_native(profile, prefix, *, timeout_seconds=60):
    """Readable own-process streaming export; no native trace is written to disk."""
    prefix = Path(prefix).resolve()
    need(prefix.is_relative_to(ROOT/'research/efficiency_20261027'), 'Contained diagnostic output')
    fifo = prefix.with_name(prefix.name+'_native.fifo')
    ledger = prefix.with_name(prefix.name+'_physical_ledger.json')
    summary = prefix.with_name(prefix.name+'_activity_summary.json')
    log_path = prefix.with_name(prefix.name+'_reader.log')
    need(not any(path.exists() for path in (fifo,ledger,summary,log_path)), 'Fresh owned export paths')
    os.mkfifo(fifo,0o600)
    command = [str(ROOT/'.runtime/env/bin/python3.11'),'-B',str(Path(__file__).resolve()),
        '--fifo',str(fifo),'--ledger',str(ledger),'--summary',str(summary)]
    child = None
    try:
        with log_path.open('xb') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
            profile.export_chrome_trace(str(fifo))
            code = child.wait(timeout=timeout_seconds)
        need(code == 0, 'Native activity reader failed; preserve reader log and partial evidence')
        result = json.loads(summary.read_text())
        result['export'] = dict(reader_argv=command,actual_returncode=code,owned_reader_reaped=True,
            ledger_path=str(ledger),summary_path=str(summary),native_disk_trace_written=False)
        return result
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=15)
        if fifo.exists():
            fifo.unlink()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fifo',required=True)
    parser.add_argument('--ledger',required=True)
    parser.add_argument('--summary',required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    paths = [Path(name).resolve() for name in (args.fifo,args.ledger,args.summary)]
    need(all(path.is_relative_to(ROOT/'research/efficiency_20261027') for path in paths), 'Contained stream paths')
    with paths[0].open() as stream:
        consume(stream,paths[1],paths[2])


if __name__ == '__main__':
    main()

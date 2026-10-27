"""Bounded direct Kineto CUDA ledger; explicit user annotations are excluded.

The actual tiny CUDA probe verified these API annotation flags against native
categories. No Chrome exporter, FIFO, subprocess or raw disk trace is used.
"""
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
from .native_activity_ledger_v2 import ROOT, PHYSICAL, LIMIT, need, native_kind, interval_union


def api_events(profile):
    for event in profile.profiler.kineto_results.events():
        device = str(event.device_type())
        annotated = event.is_user_annotation()
        need(type(annotated) is bool, 'Explicit native API annotation flag required')
        name = event.name()
        if device == 'DeviceType.CUDA':
            if annotated:
                category = 'gpu_user_annotation'
            elif name.startswith('Memcpy '):
                category = 'gpu_memcpy'
            elif name.startswith('Memset '):
                category = 'gpu_memset'
            else:
                category = 'kernel'
                need(not name.startswith(('treeloom/', 'diagnostic_', 'probe_')),
                     'An annotation-like CUDA event must carry its native API flag')
        elif device == 'DeviceType.CPU' and annotated:
            category = 'user_annotation'
        else:
            category = 'excluded_API_non_CUDA'
        yield dict(cat=category, ph='X', name=name,
                   ts=Decimal(event.start_ns())/1000, dur=Decimal(event.duration_ns())/1000,
                   args=dict(device=event.device_index(), stream=event.device_resource_id(),
                             correlation=event.correlation_id(), bytes=0))


def consume_API(profile, ledger_path, summary_path):
    categories, excluded = Counter(), Counter()
    excluded_names = Counter()
    physical, names, name_indices = [], [], {}
    roots = []
    causal = []
    charged = 0
    for event in api_events(profile):
        cat = event.get('cat', '')
        categories[cat] += 1
        kind = native_kind(event)
        if cat == 'user_annotation':
            causal.append([event['name'], int(Decimal(event['ts'])*1000), int(Decimal(event['dur'])*1000)])
            need(len(causal) <= 8192, 'Bounded complete CPU annotation causality ledger')
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
    ledger = dict(schema='TreeLoom_direct_API_physical_CUDA_ledger_v3', complete_physical_event_ledger=True,
        derived_API_categories=sorted(PHYSICAL), native_API_annotation_flag_filter=True, excluded_native_categories=dict(excluded),
        excluded_GPU_annotation_names=dict(excluded_names), window_start_native_ns=start,
        window_duration_ns=end-start, names=names,
        CPU_annotation_envelopes_for_causality_only=[[name, when-start, span] for name, when, span in causal],
        CPU_annotation_fields=['name','start_relative_ns','duration_ns'],
        fields=['kind','start_relative_ns','duration_ns','name_index','device','stream','correlation','bytes'], events=physical)
    raw = json.dumps(ledger, separators=(',', ':'), allow_nan=False).encode()
    need(len(raw) <= LIMIT, 'Final complete ledger byte cap')
    with ledger_path.open('xb') as output:
        output.write(raw)
    all_intervals = [span for group in intervals.values() for span in group]
    top = sorted(totals, key=totals.get, reverse=True)[:30]
    summary = dict(schema='TreeLoom_direct_API_activity_summary_v3', classification='CUDA_API_events_excluding_is_user_annotation_true',
        native_category_counts=dict(categories), excluded_category_counts=dict(excluded),
        excluded_GPU_annotation_names=dict(excluded_names), physical_event_counts=dict(counts),
        physical_CUDA_union=interval_union(all_intervals,start,end),
        physical_category_unions={kind:interval_union(values,start,end) for kind,values in intervals.items()},
        top_physical_events=[dict(kind=kind,name=names[index],summed_seconds=totals[(kind,index)]) for kind,index in top],
        reported_copy_bytes=dict(byte_counts), copy_events_without_positive_byte_metadata=dict(missing_bytes),
        complete_ledger_bytes=len(raw), complete_ledger_max_bytes=LIMIT,
        limitations=['Activity union covers recorded physical events of this process only, not device-wide utilization.',
            'Profiler overhead can change timing and gaps. These calls are not speed measurements.',
            'Native API is_user_annotation=True events are excluded before classifying CUDA physical activity.',
            'Copy direction uses native API names; the API exposes no byte metadata, so copy bytes are unavailable.'])
    with summary_path.open('x') as output:
        json.dump(summary,output,indent=2,allow_nan=False)
    return summary


def export_API(profile, prefix):
    prefix = Path(prefix).resolve()
    need(prefix.is_relative_to(ROOT/'research/efficiency_20261027'), 'Contained diagnostic output')
    ledger = prefix.with_name(prefix.name+'_physical_ledger.json')
    summary = prefix.with_name(prefix.name+'_activity_summary.json')
    result = consume_API(profile, ledger, summary)
    result['export'] = dict(native_disk_trace_written=False, subprocesses=0,
                            ledger_path=str(ledger), summary_path=str(summary))
    return result

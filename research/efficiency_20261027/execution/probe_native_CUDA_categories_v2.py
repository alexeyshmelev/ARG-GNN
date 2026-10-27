"""Prepared tiny actual-CUDA category probe; ROOT admission is required."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, GIB, need, ref, rss, write
from .native_activity_ledger_v2 import export_native
from .deferred_gpu_gate import wait_before_cuda


def verify_pins(digest):
    path = ROOT/'research/efficiency_20261027/execution/source_pins_native_activity_v2.json'
    need(ref(path)['sha256'] == digest, 'Exact reviewed native-activity manifest')
    sources = json.loads(path.read_text())['source_refs']
    need(all(ref(item['path']) == item for item in sources), 'All frozen sources exact')
    return dict(source_manifest_ref=ref(path), source_refs=sources)


def raw_API_rows(profile):
    rows = []
    for event in profile.profiler.kineto_results.events():
        if 'CUDA' not in str(event.device_type()):
            continue
        rows.append(dict(name=event.name(), device_type=str(event.device_type()),
            device_index=event.device_index(), device_resource_id=event.device_resource_id(),
            is_user_annotation=event.is_user_annotation(), scope=event.scope(),
            correlation_id=event.correlation_id(), linked_correlation_id=event.linked_correlation_id(),
            start_ns=event.start_ns(), duration_ns=event.duration_ns()))
    need(len(rows) <= 1000 and len(json.dumps(rows).encode()) <= 512*2**10, 'Tiny complete CUDA API record bound')
    return rows


def run_probe(output):
    cpu = torch.arange(65536, dtype=torch.float32)
    warm = torch.zeros(65536, device='cuda:0')
    torch.cuda.synchronize()
    del warm
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False) as profile:
        with torch.profiler.record_function('diagnostic_learned_update'):
            with torch.profiler.record_function('probe_outer_annotation'):
                gpu = cpu.to('cuda:0')
                value = gpu.square()
                torch.cuda.synchronize()
                time.sleep(.02)
                with torch.profiler.record_function('probe_inner_annotation'):
                    value.add_(1)
                    zero = torch.empty_like(value)
                    zero.zero_()
                    result = value.cpu()
                torch.cuda.synchronize()
    need(torch.equal(result,cpu.square()+1), 'Exact tiny observed probe arithmetic')
    raw = raw_API_rows(profile)
    write(output/'raw_CUDA_API.json',dict(rows=raw,complete=True))
    summary = export_native(profile,output/'probe')
    need(summary['physical_event_counts'].get('kernel',0)>0
         and summary['physical_event_counts'].get('H2D',0)>0
         and summary['physical_event_counts'].get('D2H',0)>0, 'Actual native kernel and bidirectional copy categories')
    need(summary['excluded_category_counts'].get('gpu_user_annotation',0)>0, 'Actual GPU user-annotation category exercised')
    need(any(item['name']=='probe_outer_annotation' for item in raw),
         'Actual GPU annotation API record established')
    need(all(not row['name'].startswith('probe_') for row in summary['top_physical_events']), 'Probe annotations never counted as physical kernels')
    need(max(row['duration_us'] for row in summary['physical_CUDA_union']['largest_gaps']) >= 15000,
         'Deliberate20ms CPU gap must not be filled by GPU annotation envelope')
    return dict(status='passed',category_summary=summary,raw_API_ref=ref(output/'raw_CUDA_API.json'),
        ledger_ref=ref(output/'probe_physical_ledger.json'),
        GPU_annotation_API_flags={item['name']:item['is_user_annotation'] for item in raw if item['name'].startswith('probe_')},
        classifier_uses_native_categories_not_API_flag=True,native_physical_category_allowlist_verified=True,
        deliberate_CPU_gap_preserved=True,learned_training_executed=False)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-pins-sha256',required=True)
    parser.add_argument('--gpu-predecessor-directory',required=True)
    args=parser.parse_args()
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') in ('3','4','5')
         and os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
         and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
         and os.environ.get('ARGGNN_GPU_MEMORY_POLICY')=='full_device','ROOT-admitted single GPU')
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pins=verify_pins(args.source_pins_sha256)
    output=Path(args.output).resolve()
    need(output.is_relative_to(ROOT/'research/efficiency_20261027/root/runs') and not output.exists(),'Fresh ROOT output')
    output.mkdir(parents=True)
    started=time.perf_counter()
    def check(stage):
        need(time.perf_counter()-started <= 120 and rss() <= 4*GIB,'Finite tiny probe wall/host guard')
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved() <= 2*GIB,'Tiny2GiB CUDA budget')
    try:
        gate=wait_before_cuda(torch,check,predecessor_directory=args.gpu_predecessor_directory,
            output=output,gpu_index=int(os.environ['ARGGNN_PHYSICAL_GPU']),gpu_budget_bytes=2*GIB,
            timeout_seconds=30,poll_seconds=5)
        torch.cuda.init()
        result=run_probe(output)
        check('probe_complete')
        verify_pins(args.source_pins_sha256)
        write(output/'completed.json',dict(**result,**pins,gpu_predecessor_gate=gate,
            physical_gpu=os.environ['ARGGNN_PHYSICAL_GPU'],total_seconds=time.perf_counter()-started))
    except BaseException as error:
        write(output/'failed.json',dict(error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    main()

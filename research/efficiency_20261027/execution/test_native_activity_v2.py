"""CPU native-export transport and adversarial annotation-exclusion tests."""
from io import StringIO
import json
from pathlib import Path
import torch
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, need, ref, write
from .native_activity_ledger_v2 import consume, native_kind, export_native

BASE=ROOT/'research/efficiency_20261027/execution'


def main():
    need(not torch.cuda.is_initialized(),'CPU-only source proof')
    output=BASE/'native_activity_CPU_v2'
    output.mkdir()
    events=[dict(cat='user_annotation',ph='X',name='diagnostic_learned_update',ts=0,dur=100),
        dict(cat='gpu_user_annotation',ph='X',name='treeloom/looks_like_kernel',ts=0,dur=100),
        dict(cat='kernel',ph='X',name='real_kernel',ts=10,dur=10,args=dict(device=0,stream=7)),
        dict(cat='gpu_memcpy',ph='X',name='Memcpy HtoD',ts=40,dur=10,args=dict(device=0,stream=7,bytes=128)),
        dict(cat='cuda_runtime',ph='X',name='cudaLaunchKernel',ts=0,dur=100)]
    summary=consume(StringIO(json.dumps(dict(traceEvents=events))),output/'fixture_ledger.json',output/'fixture_summary.json')
    need(summary['physical_CUDA_union']['busy_fraction']==.2
         and summary['physical_event_counts']==dict(kernel=1,H2D=1),'Only physical intervals enter union')
    need(summary['excluded_GPU_annotation_names']=={'treeloom/looks_like_kernel':1},'GPU user envelope excluded despite kernel-like name')
    need(native_kind(dict(cat='user_annotation',ph='X',name='Memcpy HtoD')) is None,'Copy-like annotation name cannot bypass category allowlist')
    need(native_kind(dict(cat='gpu_user_annotation',ph='X',name='Memset')) is None,'Memset-like annotation excluded')
    ledger=json.loads((output/'fixture_ledger.json').read_text())
    need(ledger['complete_physical_event_ledger'] and len(ledger['events'])==2,'Complete physical ledger remains reanalyzable')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False,
            profile_memory=False,with_stack=False) as profile:
        with torch.profiler.record_function('diagnostic_learned_update'):
            x=torch.arange(8,dtype=torch.float32,requires_grad=True)
            x.square().sum().backward()
    actual=export_native(profile,output/'actual_CPU')
    need(actual['physical_event_counts']=={} and actual['physical_CUDA_union']['busy_fraction']==0,
         'Native CPU export cannot invent CUDA activity')
    need(actual['export']['actual_returncode']==0 and actual['export']['owned_reader_reaped'],
         'Visible reader process reaches exact exit and is reaped')
    need(not list(output.glob('*.fifo')),'Only own ephemeral FIFO removed after success')
    need(not torch.cuda.is_initialized(),'No GPU work launched')
    result=dict(status='passed',schema='TreeLoom_native_activity_CPU_test_v2',
        source_refs=[ref(BASE/'native_activity_ledger_v2.py'),ref(BASE/'test_native_activity_v2.py')],
        synthetic_physical_union=summary['physical_CUDA_union'],native_CPU_export=actual,
        complete_physical_ledger=True,annotation_name_spoofing_rejected=True,CUDA_initialized=False,
        actual_GPU_API_probe_pending=True)
    write(BASE/'NATIVE_ACTIVITY_CPU_v2.json',result)
    print(json.dumps(dict(status='passed',result_ref=ref(BASE/'NATIVE_ACTIVITY_CPU_v2.json'))),flush=True)


if __name__=='__main__':
    main()

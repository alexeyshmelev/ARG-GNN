"""Explicit full-length GPU preflight or gated complete-parent training worker."""
from pathlib import Path
import argparse
import gc
import json
import os
import resource
import sys
import time
import traceback
from .candidate_common import (
    ROOT, Guard, need, load_request, read_ref, file_ref, write_json,
    identity, gpu_inventory, source_identity, numerical_identity, scope_bytes,
)


def environment(output, request):
    need(Path.cwd() == ROOT, "Candidate must run in the project repository")
    os.environ['CUDA_VISIBLE_DEVICES'] = request['gpu_uuid']
    os.environ['ARGGNN_PHYSICAL_GPU'] = '0'
    os.environ['ARGGNN_MAX_CUDA_RESERVED_BYTES'] = str(request['max_cuda_reserved_bytes'])
    os.environ['CUDA_CACHE_MAXSIZE'] = str(16*2**20)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH'):
        path = output/'runtime'/name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*2**20,8*2**20))
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')


def metadata_bindings(job, barrier):
    from arggnn.benchmarking import treeloom_fresh as fresh
    from arggnn.benchmarking.treeloom_training import SETTINGS, LIMITS, _bindings
    spec = job['call']['spec']['training_spec']
    need(spec['settings'] == SETTINGS and spec['preparation_limits'] == LIMITS,
         "Exact frozen 30-epoch, 600-update training settings required")
    registry = fresh.FreshParentRegistry.from_ref(spec['parent_registry_ref'])
    _bindings(registry,spec['train_inputs'])
    parents = registry.ids('train')
    need(barrier.get('schema') == 'treeloom_fresh_all20_observation_barrier_v1'
         and barrier.get('parent_registry_ref') == registry.reference
         and barrier.get('all20_observations_sealed_before_truth') is True
         and barrier.get('private_truth_opened') is False
         and set(barrier.get('parents',{})) == set(parents)
         and set(barrier.get('train_parent_ids',[])) == set(parents),
         "Inherited complete immutable observed-input barrier required")
    plans = {}
    for parent in parents:
        seal = barrier['parents'][parent]
        plan = read_ref(seal['plan_ref'])
        need(plan.get('parent_id') == parent and plan.get('private_truth_opened') is False
             and plan.get('observation_only') is True
             and seal.get('focal_identity_sha256')
             and all(plan.get(key) == seal.get(key) == spec['train_inputs'][parent][key]
                     for key in fresh.INPUT_REFS),
             "Every sealed parent must retain the exact observed refs")
        plans[parent] = dict(plan, plan_ref=seal['plan_ref'])
    return fresh, spec, registry, plans


def configure_torch():
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def enable_gpu(torch, guard, request_ref, profile):
    profile.mark('waiting_for_explicit_GPU_gate')
    profile.flush()
    gate_ref = guard.wait_gate('gpu_gate.json',request_ref)
    inventory = gpu_inventory(guard.request['gpu_uuid'])
    need(inventory['free_bytes'] >= guard.request['minimum_free_gpu_bytes'],
         "Shared GPU0 free-memory floor not satisfied")
    need(not torch.cuda.is_initialized(), "No CUDA initialization before the ROOT gate")
    need(torch.cuda.device_count() == 1, "Only admitted physical GPU0 may be visible")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(guard.request['max_cuda_reserved_bytes']/total,0)
    guard.cuda = True
    torch.cuda.reset_peak_memory_stats(0)
    profile.mark('GPU_gate_accepted', gpu_uuid=guard.request['gpu_uuid'])
    profile.flush()
    return gate_ref, inventory


def preflight(request, request_ref, job, barrier, output, guard):
    from .candidate_gru import install_chunked_gru
    from .progress import StageProgress
    from ..runtime.binding import ReadableTreeLoom
    torch = configure_torch()
    _, spec, registry, plans = metadata_bindings(job,barrier)
    need(not torch.cuda.is_initialized(), "Preflight metadata must remain CPU only")
    holds = max(int(plan['native_holds']) for plan in plans.values())
    runtime = ReadableTreeLoom(registry)
    model = install_chunked_gru(runtime.build_model(), chunk_holds=request['gru_chunk_holds'])
    profile = StageProgress(output/'preflight_live.json',parent_id='synthetic_full_sequence')
    try:
        profile.mark('CPU_ready', native_holds=holds, haplotypes=32)
        write_json(output/'ready_for_gpu.json',dict(request_ref=request_ref,worker_identity=identity(),
            phase='preflight',native_holds=holds,haplotypes=32,CUDA_initialized=False))
        gate_ref, inventory = enable_gpu(torch,guard,request_ref,profile)
        with profile.stage('full_sequence_GRU_forward_backward'):
            started = time.monotonic()
            model = model.to(device='cuda:0',dtype=torch.float32)
            torch.manual_seed(70019)
            context = torch.randn((holds,32,32),device='cuda:0',dtype=torch.float32,requires_grad=True)
            geometry = torch.ones((holds,3),device='cuda:0',dtype=torch.float32)
            rows = torch.arange(10000,device='cuda:0') % holds
            columns = torch.arange(10000,device='cuda:0') % 32
            with profile.stage('forward_direction'):
                forward,_ = model.scan_context(context,geometry,direction='forward')
            with profile.stage('reverse_direction'):
                reverse,_ = model.scan_context(context,geometry,direction='reverse')
            logits = model.classifier(torch.cat((context[rows,columns],forward[rows,columns],
                                                 reverse[rows,columns]),-1))
            loss = torch.nn.functional.cross_entropy(logits,torch.arange(10000,device='cuda:0')%3)
            with profile.stage('full_BPTT'):
                loss.backward()
            torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            names = ('forward_gru.','reverse_gru.','classifier.')
            need(context.grad is not None and bool(torch.isfinite(context.grad).all())
                 and all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                         for name,parameter in model.named_parameters() if name.startswith(names)),
                 "Finite complete-sequence decoder and input gradients required")
            guard.check('preflight_complete')
        result = dict(schema='treeloom_full_sequence_GPU_preflight_v1',status='passed',
            request_ref=request_ref,worker_identity=identity(),gate_ref=gate_ref,
            gpu_uuid=request['gpu_uuid'],physical_gpu=0,source_identity=source_identity(request['source_refs']),
            numerical_identity=numerical_identity(request['source_refs']),
            native_holds=holds,haplotypes=32,gru_chunk_holds=request['gru_chunk_holds'],
            full_BPTT=True,hidden_state_detached=False,CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),elapsed_seconds=elapsed,
            optimizer_updates=0,scientific_observations_loaded=False,scientific_truth_loaded=False,
            ancestry_accuracy=None,stage_timings=profile.snapshot()['stage_timings'],
            torch_version=torch.__version__,cudnn_version=torch.backends.cudnn.version())
        write_json(output/'preflight_passed.json',result)
        profile.mark('preflight_passed',elapsed_seconds=elapsed)
    finally:
        profile.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--supervisor-pid',type=int,required=True)
    parser.add_argument('--supervisor-start-ticks',type=int,required=True)
    args = parser.parse_args()
    request,request_ref,job,barrier,output = load_request(args.request)
    need(request_ref['sha256'] == args.request_sha256, "Supervisor-pinned request changed")
    need(os.getppid() == args.supervisor_pid
         and identity(args.supervisor_pid) == dict(pid=args.supervisor_pid,start_ticks=args.supervisor_start_ticks),
         "Exact directly owned supervisor identity required")
    need(output.is_dir() and not (output/'worker_identity.json').exists(), "Fresh one-shot candidate scope")
    write_json(output/'worker_identity.json',dict(request_ref=request_ref,worker_identity=identity(),
        supervisor_identity=identity(args.supervisor_pid),phase=request['phase']))
    environment(output,request)
    guard = Guard(request,output)
    try:
        if request['phase'] == 'preflight':
            preflight(request,request_ref,job,barrier,output,guard)
        else:
            from .candidate_fit import fit
            fit(request,request_ref,job,barrier,output,guard)
        guard.check('candidate_worker_complete')
    except BaseException as error:
        try:
            write_json(output/'worker_failed.json',dict(request_ref=request_ref,worker_identity=identity(),
                exception=type(error).__name__,message=str(error)[:2000],
                traceback=traceback.format_exc()[-14000:],complete_schedule_claimed=False))
        finally:
            raise


if __name__ == '__main__':
    main()

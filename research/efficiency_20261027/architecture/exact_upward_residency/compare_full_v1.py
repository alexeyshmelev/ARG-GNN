"""One complete TRAIN000 pair: boundary fusion versus upward/direct GPU residency.

Standalone numerical and resource qualification. Never writes a production
checkpoint, changes the campaign, truncates genomic history, or opens TEST.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import torch
from research.efficiency_20261027.root.compare_host_parallelism import (
    ROOT, CHECKPOINT, GIB, need, ref, rss, write, prepare)
from research.efficiency_20261027.root.compare_selected_full import MemoryProfile, failure_record
from research.efficiency_20261027.execution.compare_full_context import compare_state
from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
from research.efficiency_20261027.architecture.exact_boundary_fusion.large_boundary_executor_v1 import build_executor as baseline_factory
from .large_resident_executor_v1 import build_executor as fused_factory
from .memory_v1 import residency_plan, DEFAULT_SCRATCH_BYTES
from .full_release_v1 import verify_prerequisites, require_comparison
from research.efficiency_20261027.execution.deferred_gpu_gate import wait_before_cuda

KINDS = ('boundary_fused_CPU512', 'upward_direct_resident_CPUcontext512')
MANIFEST = ROOT / 'research/efficiency_20261027/architecture/exact_upward_residency/source_pins_full_v1.json'
DEVICE_BUDGET_BYTES = 74*GIB
EXPECTED_NATIVE_NODES = 205369272


def verify_pins(digest):
    need(ref(MANIFEST)['sha256'] == digest, 'Exact reviewed source manifest')
    sources = json.loads(MANIFEST.read_text())['source_refs']
    need(len(sources) == len({item['path'] for item in sources}), 'Unique source paths')
    need(all(ref(item['path']) == item for item in sources), 'Reviewed sources unchanged')
    return dict(source_manifest_ref=ref(MANIFEST), source_refs=sources)


def run_update(kind, rt, saved, bundle, metadata, labels, frozen, cache, check, output):
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import (
        decode_parent_backward, geometry_from_spans)
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    need(kind in KINDS and torch.get_num_threads() == 8, 'CPU8 and same512 partition')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    profile = MemoryProfile(torch, check, journal=output/'stage_memory.jsonl',
        identity=dict(kind=kind, phase='full_first_call', repeat=0), GPU_budget_bytes=DEVICE_BUDGET_BYTES)
    model = optimizer = executor = hook = context = vjp = None
    logits = []
    wall_start, started = time.time(), time.perf_counter()
    try:
        with profile.stage('profile_restore'):
            model = rt.build_model()
            model.load_state_dict(saved['model_state'], strict=True)
            model = model.to(device='cuda:0', dtype=torch.float32)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                betas=(.9, .999), eps=1e-8, maximize=False, amsgrad=False,
                foreach=False, fused=False, capturable=False, differentiable=False)
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']))
            restore_random_state(saved['random_state'])
        hook = model.classifier.register_forward_hook(
            lambda module, inputs, value: logits.append(value.detach().cpu()))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with profile.stage('profile_layout'):
            executor = (baseline_factory if kind == KINDS[0] else fused_factory)(
                model, bundle, metadata, layout_cache=cache, frozen_module=frozen,
                holds_per_block=512, haplotypes_per_gru=64, history_holds_per_window=1024,
                resource_check=check, profile=profile)
        bank_plan = None
        if kind == KINDS[1]:
            bank_plan = residency_plan(executor)
            need(bank_plan["native_nodes"] == EXPECTED_NATIVE_NODES
                 and bank_plan["scratch_reserve_bytes"] == DEFAULT_SCRATCH_BYTES
                 and torch.cuda.memory_allocated()+bank_plan["initial_additional_capacity_bytes"] <= DEVICE_BUDGET_BYTES,
                 "Actual two-bank lifetime plus16GiB scratch must fit74GiB worker budget before forward")
        with torch.autocast(device_type='cuda', enabled=False):
            with profile.stage('profile_context_forward'):
                context = executor.context_forward()
                need(context.shape == (77472, 1600, 32) and context.device.type == 'cpu',
                     'Complete original CPU context')
            with profile.stage('profile_decoder_forward_backward'):
                loss, vjp, decoder = decode_parent_backward(model, context,
                    geometry_from_spans(bundle.bank.spans), bundle.focal_hold,
                    bundle.focal_sample_column, labels, tile_haplotypes=64,
                    profile=profile, resource_check=check)
                need(vjp.shape == context.shape and vjp.device == context.device,
                     'Complete original context VJP')
            with profile.stage('profile_trunk_backward'):
                context.backward(vjp)
        with profile.stage('profile_gradient_capture'):
            gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
            need(len(gradients) == 49 and all(bool(torch.isfinite(v).all()) for v in gradients.values()),
                 'All49 original finite gradients')
        with profile.stage('profile_clip_optimizer'):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        update_seconds, wall_end = time.perf_counter()-started, time.time()
        check('full_upward_residency_update_complete')
        with profile.stage('profile_state_capture'):
            parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            moments = {str(i): {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in row.items()} for i, row in optimizer.state_dict()['state'].items()}
            need(len(moments) == 49 and all(int(row['step'].item()) == 242 for row in moments.values()),
                 'All49 original Adam counters become242')
            numerical = dict(logits=torch.cat(logits), loss=loss.detach().cpu(), gradients=gradients,
                parameters=parameters, moments=moments)
            need(numerical['logits'].shape == (10000, 3), 'All10000 original targets')
        row = dict(kind=kind, update_seconds=update_seconds, update_wall_start_unix=wall_start,
            update_wall_end_unix=wall_end, CPU_threads=8, holds_per_block=512,
            loss=float(loss), gradient_norm=float(norm), decoder=decoder, stages=profile.snapshot(),
            torch_peak_allocated_bytes=profile.run_allocated_peak,
            torch_peak_reserved_bytes=profile.run_reserved_peak,
            RSS_bytes=rss(), process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cache_snapshot=cache.snapshot(), explicit_vjp=executor.explicit_vjp_snapshot(),
            full_history=True, CPU_context=True, hotpack=False,
            native_nodes=bundle.bank.parent.numel(),resident_bank_plan=bank_plan,
            device_soft_budget_bytes=DEVICE_BUDGET_BYTES,parameter_count=49,optimizer_step=242)
        return row, numerical
    except BaseException as error:
        write(output/('failed_'+kind+'.json'), failure_record(error, kind=kind,
            elapsed_seconds=time.perf_counter()-started, stages=profile.snapshot()))
        raise
    finally:
        if hook is not None:
            hook.remove()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if executor is not None:
            executor.release_layout()
        model = optimizer = executor = hook = context = vjp = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-pins-sha256', required=True)
    parser.add_argument('--prefix-run-directory', required=True)
    parser.add_argument('--prefix-completed-sha256', required=True)
    parser.add_argument('--boundary-full-run-directory', required=True)
    parser.add_argument('--boundary-full-completed-sha256', required=True)
    args = parser.parse_args()
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') in ('3', '4', '5')
         and os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
         and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
         and os.environ.get('ARGGNN_GPU_MEMORY_POLICY') == 'full_device',
         'ROOT-admitted single GPU environment')
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    pins = verify_pins(args.source_pins_sha256)
    need(not torch.cuda.is_initialized(), "Prerequisite validation precedes CUDA")
    prerequisites = verify_prerequisites(args)
    output = Path(args.output).resolve()
    need(output.is_relative_to(ROOT/'research/efficiency_20261027/root/runs')
         and not output.exists(), 'Fresh ROOT qualification output')
    output.mkdir(parents=True)
    started = time.perf_counter()
    def check(stage):
        need(time.perf_counter()-started <= 7200, 'Complete worker7200second limit at '+stage)
        need(rss() <= 256*GIB, '256GiB worker RSS limit at '+stage)
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved() <= DEVICE_BUDGET_BYTES, '74GiB finite device envelope at '+stage)
    write(output/'configuration.json', dict(arguments=vars(args), **pins,
        protocol=list(KINDS), preparation_workers=4, CPU_threads=8, worker_RSS_bytes=256*GIB,
        supervisor_group_PSS_bytes=288*GIB, device_envelope_bytes=DEVICE_BUDGET_BYTES,
        prerequisites=prerequisites,expected_native_nodes=EXPECTED_NATIVE_NODES,
        resident_scratch_bytes=DEFAULT_SCRATCH_BYTES,
        max_seconds=7200, numerical_repetitions=1, profiler=False,
        note='One fixed-order complete call per arm, not a steady-state throughput estimate.'))
    try:
        from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh, training_v2 as validation
        from research.preprocessing_optimization_20261026.integration.hooks import install_tree
        need(ref(CHECKPOINT)['sha256'] == '12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24',
             'Immutable checkpoint241')
        saved = torch.load(ROOT/CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
        registry = fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref'])
        rt = fresh.FreshTreeLoom(registry)
        validation.validate_saved_state(saved, rt.build_model(), registry, expected_steps=241)
        install_tree(preparation_workers=4, max_rss_bytes=256*GIB)
        need(not torch.cuda.is_initialized(), 'Prepare before CUDA initialization')
        bundle, metadata, labels, frozen, scope = prepare(SimpleNamespace(mode='full'), rt, saved, check)
        need(bundle.native_holds == 77472 and len(labels) == 10000 and bundle.original_sites == 227474
             and bundle.bank.parent.numel() == EXPECTED_NATIVE_NODES
             and sum(metadata['episode_metadata']['native_nodes_per_tree']) == EXPECTED_NATIVE_NODES,
             'Exact complete original TRAIN000 observation')
        bank_bytes = bundle.bank.parent.numel()*32*4
        need(2*bank_bytes+DEFAULT_SCRATCH_BYTES <= DEVICE_BUDGET_BYTES,
             'Exact full-parent two-bank payload and16GiB scratch fit declared device budget')
        scope = dict(scope,native_nodes=bundle.bank.parent.numel(),resident_bank_bytes=bank_bytes,
            maximum_two_bank_bytes=2*bank_bytes,minimum_scratch_bytes=DEFAULT_SCRATCH_BYTES)
        write(output/'scope.json', scope)
        need(verify_prerequisites(args) == prerequisites, 'Pinned successful prerequisites remain unchanged')
        gate = wait_before_cuda(torch,check,predecessor_directory=args.boundary_full_run_directory,
            output=output,gpu_index=int(os.environ['ARGGNN_PHYSICAL_GPU']),
            gpu_budget_bytes=DEVICE_BUDGET_BYTES,timeout_seconds=60,poll_seconds=5)
        write(output/'gpu_predecessor_gate.json', gate)
        rows, baseline = [], None
        for kind in KINDS:
            verify_pins(args.source_pins_sha256)
            cache = LargeBlockLayoutCache(max_bytes=8*GIB, admit=lambda amount, stage:
                need(rss()+amount <= 256*GIB, 'Layout working reserve at '+stage))
            row, numerical = run_update(kind, rt, saved, bundle, metadata, labels, frozen, cache, check, output)
            if baseline is None:
                baseline = numerical
            row['comparison'] = compare_state(baseline, numerical)
            write(output/(kind+'.json'), row)
            require_comparison(row['comparison'])
            rows.append(row)
            del cache
            gc.collect()
            print(json.dumps(dict(event='full_upward_residency_update_complete', kind=kind,
                update_seconds=row['update_seconds'], comparison_passed=True)), flush=True)
        verify_pins(args.source_pins_sha256)
        write(output/'completed.json', dict(status='completed', schema='TreeLoom_upward_direct_residency_full_v1',
            **pins, checkpoint_ref=ref(CHECKPOINT), scope=scope, records=rows,
            prerequisites=prerequisites,gpu_predecessor_gate=gate,
            all49_gradient_and_parameter_tensors_checked=True,all147_Adam_tensors_checked=True,
            device_soft_budget_bytes=DEVICE_BUDGET_BYTES,worker_RSS_bytes=256*GIB,family_PSS_bytes=288*GIB,
            all_strict_comparisons_passed=True, physical_gpu=os.environ['ARGGNN_PHYSICAL_GPU'],
            total_seconds=time.perf_counter()-started,
            limitations=['One full call per arm in fixed order, with separate cold layout caches.',
                'Original full parent and all targets retained, parameters reset from241 per arm.',
                'FP32 tolerance equivalence is not bitwise equality over future training.',
                'One exact full pair; no added prefix, warmup, profiler or repeated performance calls.',
                'Inherited synchronized stage-memory accounting is retained; it is not a Torch profiler.',
                'No held-out accuracy score or sustained80percent utilization claim.']))
    except BaseException as error:
        write(output/'failed.json', failure_record(error, elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()

"""Prepared-only1153-hold upward/direct-residency qualification; no profiler.

Six restored checkpoint241 calls: warmup AB, timed AB, timed BA.
Compare qualified boundary fusion against its separate residency successor.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import time
import torch
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, CHECKPOINT, GIB, need, ref, rss, write
from research.efficiency_20261027.execution.compare_full_context import compare_state
from research.efficiency_20261027.execution.compare_explicit_resident_vjp import failure_record
from research.efficiency_20261027.execution.prepare_large_slice_v2 import prepare_large_prefix
from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
from research.efficiency_20261027.architecture.exact_boundary_fusion.large_boundary_executor_v1 import build_executor as baseline_factory
from .large_resident_executor_v1 import build_executor as fused_factory
from research.efficiency_20261027.execution.deferred_gpu_gate import wait_before_cuda
from research.efficiency_20261027.execution.bounded_timeline_v1 import DiagnosticProfile

BASE = ROOT / 'research/efficiency_20261027/architecture/exact_upward_residency'
KINDS = ('boundary_fused_CPU512', 'upward_direct_resident_CPUcontext512')


def schedule():
    return [('warmup', 0, KINDS), ('timed', 1, KINDS), ('timed', 2, tuple(reversed(KINDS)))]


def verify_pins(digest):
    manifest_path = BASE / 'source_pins_prefix_v1.json'
    need(ref(manifest_path)['sha256'] == digest, 'Exact reviewed runner source manifest')
    sources = json.loads(manifest_path.read_text())['source_refs']
    need(len(sources) == len({item['path'] for item in sources})
         and all(ref(item['path']) == item for item in sources), 'Complete immutable reviewed source set')
    return dict(source_manifest_ref=ref(manifest_path), source_refs=sources)


def run_update(kind, phase, repeat, args, rt, saved, bundle, metadata, labels, frozen, cache, check, output):
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward, geometry_from_spans
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    need(kind in KINDS and torch.get_num_threads() == 8, 'Selected CPU8/512 boundary fusion or upward/direct residency')
    need(phase in ('warmup', 'timed'), 'No profiled calls in residency qualification')
    profile = DiagnosticProfile(False)
    identity = dict(kind=kind, phase=phase, repeat=repeat)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = optimizer = executor = hook = context = vjp = None
    logits = []
    try:
        with profile.stage('model_optimizer_restore'):
            model = rt.build_model()
            model.load_state_dict(saved['model_state'], strict=True)
            model = model.to(device='cuda:0', dtype=torch.float32)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                betas=(.9, .999), eps=1e-8, maximize=False, amsgrad=False,
                foreach=False, fused=False, capturable=False, differentiable=False)
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']))
            restore_random_state(saved['random_state'])
        hook = model.classifier.register_forward_hook(lambda module, inputs, value: logits.append(value.detach().cpu()))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with profile.stage('layout'):
            executor = (baseline_factory if kind == KINDS[0] else fused_factory)(model, bundle, metadata,
                layout_cache=cache, frozen_module=frozen, holds_per_block=512, haplotypes_per_gru=64,
                history_holds_per_window=1024, resource_check=check, profile=profile)
        # The learned-region clock excludes model/layout setup and numerical capture.
        # Original .to/.cpu boundaries remain; no stage-level synchronization is added.
        torch.cuda.synchronize()
        learned_started = time.perf_counter()
        with torch.autocast(device_type='cuda', enabled=False):
            with profile.stage('context_forward'):
                context = executor.context_forward()
                need(context.shape == (1153, 1600, 32) and context.device.type == 'cpu', 'Complete original CPU context')
            with profile.stage('decoder_forward_backward'):
                loss, vjp, decoder = decode_parent_backward(model, context, geometry_from_spans(bundle.bank.spans),
                    bundle.focal_hold, bundle.focal_sample_column, labels,
                    tile_haplotypes=64, profile=profile, resource_check=check)
                need(vjp.shape == context.shape and vjp.device == context.device, 'Complete original CPU context VJP')
            with profile.stage('trunk_backward'):
                context.backward(vjp)
            torch.cuda.synchronize()
        learned_seconds = time.perf_counter()-learned_started
        with profile.stage('gradient_capture'):
            gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()}
            need(len(gradients) == 49 and all(bool(torch.isfinite(g).all()) for g in gradients.values()), 'All49 finite connected gradients')
        with profile.stage('clip_optimizer'):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        update_seconds = time.perf_counter()-started
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        check('upward_residency_update_complete')
        with profile.stage('state_capture'):
            parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            moments = {str(i): {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in row.items()} for i, row in optimizer.state_dict()['state'].items()}
            need(len(moments) == 49 and all(int(row['step'].item()) == 242 for row in moments.values()), 'Every49 Adam counter242')
            numerical = dict(logits=torch.cat(logits), loss=loss.detach().cpu(), gradients=gradients,
                parameters=parameters, moments=moments)
            need(numerical['logits'].shape == (len(labels), 3), 'Every selected actual TRAIN target once')
        row = dict(**identity, profiled=False, performance_sample=phase == 'timed',
            update_seconds=update_seconds, learned_region_seconds=learned_seconds,
            timing_definition='update: restore through gradient capture and Adam step; learned: forward, decoder backward, trunk backward plus boundary synchronization.',
            profiler_start_stop_overhead_in_learned_time=False,
            loss=float(loss), gradient_norm=float(norm), decoder=decoder, stages=profile.snapshot(),
            torch_peak_allocated_bytes=peak_allocated, torch_peak_reserved_bytes=peak_reserved,
            RSS_bytes=rss(), process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cache_snapshot=cache.snapshot(), explicit_vjp=executor.explicit_vjp_snapshot(),
            CPU_context=True, hotpack=False, parameter_count=49, optimizer_step=242)
        return row, numerical
    except BaseException as error:
        write(output/('failed_'+phase+'_'+str(repeat)+'_'+kind+'.json'), failure_record(error, **identity,
            elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        if hook is not None:
            hook.remove()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if executor is not None:
            executor.release_layout()
        context = vjp = model = optimizer = executor = hook = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-pins-sha256', required=True)
    parser.add_argument('--gpu-predecessor-directory', required=True)
    args = parser.parse_args()
    args.slice_holds = 1153
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') in ('3', '4', '5')
         and os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
         and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
         and os.environ.get('ARGGNN_GPU_MEMORY_POLICY') == 'full_device', 'ROOT-admitted singleGPU environment')
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    need(not torch.cuda.is_initialized(), 'CPU preparation precedes CUDA')
    pins = verify_pins(args.source_pins_sha256)
    output = Path(args.output).resolve()
    need(output.is_relative_to(ROOT/'research/efficiency_20261027/root/runs') and not output.exists(), 'Fresh ROOT output')
    output.mkdir(parents=True)
    started = time.perf_counter()
    def check(stage):
        need(time.perf_counter()-started <= 1800, 'Complete worker wall1800second cap at '+stage)
        need(rss() <= 32*GIB, '32GiB host RSS cap at '+stage)
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved() <= 24*GIB, '24GiB GPU reserved cap at '+stage)
    write(output/'configuration.json', dict(arguments=vars(args), **pins, protocol=schedule(),
        host_cap_bytes=32*GIB, device_soft_cap_bytes=24*GIB, max_worker_seconds=1800,
        profiler_enabled=False, preparation_workers=4, CPU_threads=8, hotpack=False,
        original_FP32_math=True, independent_layout_caches=True, standalone_prefix_claim_only=True))
    try:
        from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh, training_v2 as validation
        need(ref(CHECKPOINT)['sha256'] == '12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24', 'Exact checkpoint241')
        saved = torch.load(ROOT/CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
        registry = fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref'])
        rt = fresh.FreshTreeLoom(registry)
        validation.validate_saved_state(saved, rt.build_model(), registry, expected_steps=241)
        bundle, metadata, labels, frozen, scope = prepare_large_prefix(args, rt, saved, check)
        write(output/'scope.json', scope)
        gate = wait_before_cuda(torch, check, predecessor_directory=args.gpu_predecessor_directory,
            output=output, gpu_index=int(os.environ['ARGGNN_PHYSICAL_GPU']), gpu_budget_bytes=24*GIB,
            timeout_seconds=60, poll_seconds=5)
        write(output/'gpu_predecessor_gate.json', gate)
        caches = {kind: LargeBlockLayoutCache(max_bytes=256*2**20, admit=lambda amount, stage:
            need(rss()+amount <= 32*GIB, 'Layout working reserve at '+stage)) for kind in KINDS}
        baseline = None
        rows = []
        for phase, repeat, order in schedule():
            verify_pins(args.source_pins_sha256)
            for position, kind in enumerate(order):
                row, numerical = run_update(kind, phase, repeat, args, rt, saved, bundle, metadata, labels,
                    frozen, caches[kind], check, output)
                if baseline is None:
                    baseline = numerical
                row['comparison'] = compare_state(baseline, numerical)
                row['order_position'] = position
                rows.append(row)
                write(output/(phase+'_'+str(repeat)+'_'+kind+'.json'), row)
                need(row['comparison']['passed'], 'Strict loss/logits49gradients/parameters/Adam and zeroargmax difference')
                print(json.dumps(dict(event='upward_residency_update_complete', kind=kind, phase=phase,
                    repeat=repeat, update_seconds=row['update_seconds'], learned_seconds=row['learned_region_seconds'],
                    comparison_passed=True)), flush=True)
        verify_pins(args.source_pins_sha256)
        summary = {kind: dict(timed_calls=2,
            median_update_seconds=statistics.median(r['update_seconds'] for r in rows if r['kind'] == kind and r['phase'] == 'timed'),
            median_learned_seconds=statistics.median(r['learned_region_seconds'] for r in rows if r['kind'] == kind and r['phase'] == 'timed'),
            maximum_allocated_bytes=max(r['torch_peak_allocated_bytes'] for r in rows if r['kind'] == kind),
            maximum_reserved_bytes=max(r['torch_peak_reserved_bytes'] for r in rows if r['kind'] == kind)) for kind in KINDS}
        write(output/'completed.json', dict(status='completed', schema='TreeLoom_upward_direct_residency_prefix_GPU_qualification_v1',
            **pins, checkpoint_ref=ref(CHECKPOINT), scope=scope, records=rows, summaries=summary,
            total_seconds=time.perf_counter()-started, gpu_predecessor_gate=gate,
            physical_gpu=os.environ['ARGGNN_PHYSICAL_GPU'], visible_UUID=os.environ['CUDA_VISIBLE_DEVICES'],
            all_strict_comparisons_passed=True, actual_49gradient_and_Adam_checks=True, hotpack=False,
            limitations=['This is a1153hold prefix with restricted history, not a full30Mb qualification.',
                'Only two unprofiled timed observations per arm; no statistical certainty or full-parent speed claim.',
                'No Torch profiler is active and no timeline is collected.',
                'Actual-count capacity admission and16GiB scratch do not qualify full-parent peak memory.',
                'Both arms restore the same checkpoint241 each call; no trained state is saved to production.']))
    except BaseException as error:
        write(output/'failed.json', failure_record(error, elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()

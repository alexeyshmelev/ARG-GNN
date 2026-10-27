"""Prepared-only strict observed128 Transport execution qualification.

ROOT owns admission and external resource supervision. Each isolated trial
starts from identical fresh weights and performs one real TRAIN-loss backward
and clipped AdamW update. This is not an accuracy or full-parent benchmark.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import resource
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OWNED = ROOT / 'research/efficiency_20261027/architecture'
MANIFEST = OWNED / 'staged_gpu_128_qualification_sources_v2.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    need(path.is_file() and not path.is_symlink(), 'Ordinary frozen file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest())


def clock():
    return dict(unix_seconds=time.time(), monotonic_seconds=time.monotonic(),
                UTC=datetime.now(timezone.utc).isoformat())


def rss():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


class StageMeter:
    """Synchronized interval peaks; instrumentation is included in trial timings."""
    def __init__(self, torch, check):
        self.torch, self.check = torch, check
        self.current = None
        self.rows = []

    def switch(self, name):
        self.close()
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        self.current = dict(stage=name, start=clock(), start_RSS_bytes=rss())

    def close(self):
        if self.current is None:
            return
        self.torch.cuda.synchronize()
        row = self.current
        row.update(end=clock(), end_RSS_bytes=rss(),
                   GPU_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                   GPU_peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
        row['seconds'] = row['end']['monotonic_seconds'] - row['start']['monotonic_seconds']
        self.rows.append(row)
        self.current = None
        self.check(row['stage'])

    def callback(self, name):
        self.check(name)
        if name.startswith('staged_transport_') and (name.endswith('_begin') or name.endswith('_complete')):
            self.switch(name)


def compare(torch, expected, actual):
    checks = []
    maximum = {}
    rules = {'logits': (1e-6, 0.), 'loss': (1e-6, 0.),
             'gradients': (1e-6, 1e-4), 'clipped_gradients': (1e-6, 1e-4),
             'parameters': (1e-6, 1e-4), 'moments': (1e-6, 1e-4),
             'gradient_norm': (1e-6, 1e-4)}
    for category, (atol, rtol) in rules.items():
        maximum[category] = 0.
        def visit(a, b, path):
            if isinstance(a, dict):
                need(isinstance(b, dict) and set(a) == set(b), 'Identical numerical keys: ' + path)
                for key in a:
                    visit(a[key], b[key], path + '.' + str(key))
            elif torch.is_tensor(a):
                need(a.shape == b.shape and a.dtype == b.dtype, 'Identical tensor contract: ' + path)
                finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                error = (float((a - b).abs().max()) if a.numel() else 0.) if finite else None
                accepted = finite and bool(torch.isclose(b, a, atol=atol, rtol=rtol).all())
                if error is not None:
                    maximum[category] = max(maximum[category], error)
                checks.append(dict(path=path, passed=accepted, finite=finite, max_abs_error=error,
                                   atol=atol, rtol=rtol))
            else:
                checks.append(dict(path=path, passed=a == b))
        visit(expected[category], actual[category], category)
    changed = int((expected['logits'].argmax(-1) != actual['logits'].argmax(-1)).sum())
    return dict(passed=all(row['passed'] for row in checks) and changed == 0,
                argmax_changes=changed, category_max_abs_error=maximum, checks=checks,
                note='Gradient gates use mixed absolute and relative tolerances; passing does not imply maximum absolute error below 1e-6.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-rss-gib', type=float, default=32.)
    parser.add_argument('--max-gpu-reserved-gib', type=float, default=24.)
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args()
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') == '5' and
         os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-') and
         ',' not in os.environ['CUDA_VISIBLE_DEVICES'] and
         os.environ.get('ARGGNN_GPU_MEMORY_POLICY') == 'full_device',
         'ROOT must admit physical GPU5 by UUID and its existing full-device policy')
    need(0 < args.max_rss_gib <= 32 and 0 < args.max_gpu_reserved_gib <= 24 and
         0 < args.max_seconds <= 1200, 'Do not enlarge prepared diagnostic resource allowances')
    output = (ROOT / args.output).resolve()
    need(output.is_relative_to(OWNED) and not output.exists(), 'Fresh architecture-owned output required')
    output.mkdir(parents=True)
    started = clock()
    deadline = started['monotonic_seconds'] + args.max_seconds
    records = []
    def write(name, value):
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return reference(output / name)
    try:
        manifest_ref = reference(MANIFEST)
        manifest = json.loads(MANIFEST.read_text())
        pins = manifest['source_refs'] + manifest['input_refs']
        for pin in pins:
            need(reference(pin['path']) == pin, 'Frozen dependency changed before qualification: ' + pin['path'])
        import setproctitle
        setproctitle.setproctitle('AI Auto Research Pipeline Test')
        import torch
        from torch.nn import functional as F
        from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
        from research.scalability_20261024.profiling.run_profile import prepare
        from .treeloom_transport import TreeLoomTransport, prepare_transport_plan
        from .staged_transport import StagedTransport
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        need(torch.cuda.device_count() == 1, 'One admitted logical CUDA device required')
        def check(stage):
            need(time.monotonic() < deadline, 'Diagnostic wall budget exceeded at ' + stage)
            need(rss() <= args.max_rss_gib * 2**30, 'Diagnostic RSS budget exceeded at ' + stage)
            if torch.cuda.is_initialized():
                need(torch.cuda.memory_reserved() <= args.max_gpu_reserved_gib * 2**30,
                     'Diagnostic GPU reserved budget exceeded at ' + stage)
        schedule = [('first_call', 0, ('monolithic', 'staged_uncached', 'staged_cached')),
                    ('warm_libraries', 1, ('staged_uncached', 'staged_cached', 'monolithic')),
                    ('warm_libraries', 2, ('staged_cached', 'monolithic', 'staged_uncached')),
                    ('warm_libraries', 3, ('monolithic', 'staged_uncached', 'staged_cached'))]
        optimizer_config = dict(lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8,
                                maximize=False, amsgrad=False, foreach=False, fused=False,
                                capturable=False, differentiable=False)
        write('invocation.json', dict(arguments=vars(args), source_manifest_ref=manifest_ref,
              source_refs=pins, physical_gpu=os.environ['ARGGNN_PHYSICAL_GPU'],
              visible_UUID=os.environ['CUDA_VISIBLE_DEVICES'], start=started,
              external_resource_guardian_required=True, hard_allocator_fraction_cap=False,
              CPU_threads=4, seed=70001, FP32=True, tree_block=128, haplotype_tile=128, scan_block=1024,
              schedule=schedule, optimizer=optimizer_config, gradient_clip_norm=1.,
              optimizer_updates_per_independent_trial=1, continued_training=False,
              stage_timing_includes_synchronization=True))
        checkpoint = 'research/repairs/TreeLoom_resume94_RAM_headroom_v1/checkpoint.pt'
        need(reference(checkpoint)['sha256'] == '5ceebcd75ae978689f987a702ea5a157d6ddc2bd7d057e6aded56627bbf45276',
             'Original94 observation-barrier lineage required')
        saved = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=True, mmap=True)
        registry = fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref'])
        runtime = fresh.FreshTreeLoom(registry)
        bundle, metadata, labels, _, scope = prepare(SimpleNamespace(mode='slice'), runtime, saved, check)
        del saved
        need(bundle.native_holds == 128 and bundle.bank.sample_index.shape == (128, 1600) and
             labels.shape == bundle.focal_hold.shape == bundle.focal_sample_column.shape == (22,),
             'Exactly original observed128-hold window and22 frozen TRAIN focals required')
        cpu_plan = prepare_transport_plan(bundle.bank.parent, bundle.episodes.flags.node_flags)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(70001)
            initial = TreeLoomTransport()
        weights = {name: value.detach().clone() for name, value in initial.state_dict().items()}
        del initial
        reference_numerical = None
        write('scope.json', scope)
        for phase, repeat, kinds in schedule:
            for position, kind in enumerate(kinds):
                check('before_' + kind)
                torch.cuda.synchronize()
                trial_started = clock()
                meter = StageMeter(torch, check)
                meter.switch('source_model_optimizer_transfer')
                model = TreeLoomTransport()
                model.load_state_dict(weights, strict=True)
                model = model.to('cuda:0').train()
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
                optimizer.zero_grad(set_to_none=True)
                execution = None
                if kind == 'monolithic':
                    device_bank = bundle.bank.to('cuda:0')
                    device_plan = cpu_plan.to('cuda:0')
                    focal_hold = bundle.focal_hold.to('cuda:0')
                    focal_column = bundle.focal_sample_column.to('cuda:0')
                else:
                    meter.switch('staged_execution_setup')
                    execution = StagedTransport(model, bundle.bank, bundle.episodes.flags.node_flags,
                        metadata['episode_metadata']['native_nodes_per_tree'], holds_per_block=128,
                        haplotypes_per_tile=128, scan_holds_per_block=1024,
                        cache_first_nodes=kind == 'staged_cached', resource_check=meter.callback)
                device_labels = labels.to('cuda:0')
                meter.switch('forward_logits_and_loss')
                forward_start = clock()
                with torch.autocast(device_type='cuda', enabled=False):
                    if kind == 'monolithic':
                        logits = model.focal_logits(device_bank, device_plan, focal_hold, focal_column)
                    else:
                        logits = execution.forward_focals(bundle.focal_hold, bundle.focal_sample_column)
                    loss = F.cross_entropy(logits, device_labels)
                torch.cuda.synchronize()
                forward_end = clock()
                meter.switch('loss_backward')
                backward_start = clock()
                loss.backward()
                if execution is not None:
                    need(logits.grad is not None and bool(torch.isfinite(logits.grad).all()), 'Finite complete logits VJP required')
                    need(all(p.grad is None for p in model.parameters()), 'Parameter gradients must start in explicit VJPs')
                    execution.backward_focals(logits.grad)
                    need(execution.state is None, 'Explicit stage state must be consumed')
                torch.cuda.synchronize()
                backward_end = clock()
                meter.switch('capture_unclipped_gradients')
                need(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)), 'Finite logits and TRAIN loss required')
                gradients = {}
                for name, parameter in model.named_parameters():
                    need(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), 'Finite connected gradient required: ' + name)
                    gradients[name] = parameter.grad.detach().cpu().clone()
                need(len(gradients) == 55, 'All55 parameter gradients required')
                need(all(torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in weights.items()),
                     'Weights must remain identical before explicit optimizer update')
                meter.switch('clip_and_AdamW_update')
                update_start = clock()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                update_end = clock()
                meter.switch('capture_updated_state')
                numerical = dict(logits=logits.detach().cpu().clone(), loss=loss.detach().cpu().clone(),
                    gradients=gradients, gradient_norm=norm.detach().cpu().clone(),
                    clipped_gradients={name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()},
                    parameters={name: p.detach().cpu().clone() for name, p in model.named_parameters()},
                    moments={str(i): {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in row.items()}
                             for i, row in optimizer.state_dict()['state'].items()})
                need(len(numerical['moments']) == 55 and
                     all(int(row['step']) == 1 for row in numerical['moments'].values()), 'One fresh AdamW update for all55 tensors required')
                meter.close()
                numerical_path = output / f'{phase}_{repeat}_{kind}_numerical.pt'
                need(not numerical_path.exists(), 'Fresh numerical output required')
                torch.save(numerical, numerical_path)
                if reference_numerical is None:
                    reference_numerical = numerical
                    comparison = compare(torch, numerical, numerical)
                    comparison['reference'] = True
                else:
                    comparison = compare(torch, reference_numerical, numerical)
                record = dict(kind=kind, phase=phase, repeat=repeat, schedule_position=position,
                    start=trial_started, end=clock(), loss=float(loss), gradient_norm=float(norm),
                    forward_start=forward_start, forward_end=forward_end,
                    backward_start=backward_start, backward_end=backward_end,
                    update_start=update_start, update_end=update_end,
                    forward_seconds=forward_end['monotonic_seconds'] - forward_start['monotonic_seconds'],
                    backward_seconds=backward_end['monotonic_seconds'] - backward_start['monotonic_seconds'],
                    clip_optimizer_seconds=update_end['monotonic_seconds'] - update_start['monotonic_seconds'],
                    GPU_peak_allocated_bytes=max(row['GPU_peak_allocated_bytes'] for row in meter.rows),
                    GPU_peak_reserved_bytes=max(row['GPU_peak_reserved_bytes'] for row in meter.rows),
                    process_RSS_bytes=rss(), process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    stages=meter.rows, numerical_ref=reference(numerical_path), comparison=comparison,
                    all55_gradients_finite=True, optimizer_updates=1)
                records.append(record)
                write(f'{phase}_{repeat}_{kind}.json', record)
                print(json.dumps({k: v for k, v in record.items() if k not in ('comparison', 'stages')}), flush=True)
                need(comparison['passed'], 'Strict numerical/argmax/update qualification failed; inspect saved tensors and comparison')
                if kind == 'monolithic':
                    del device_bank, device_plan, focal_hold, focal_column
                else:
                    execution.release()
                del execution, model, optimizer, logits, loss, numerical, gradients, norm, device_labels, meter
                gc.collect()
                torch.cuda.empty_cache()
                check('after_' + kind)
        for pin in pins:
            need(reference(pin['path']) == pin, 'Frozen dependency changed during qualification: ' + pin['path'])
        need(reference(MANIFEST) == manifest_ref, 'Source manifest changed during qualification')
        write('completed.json', dict(schema='TreeLoom_Transport_staged_observed128_GPU_qualification_v1',
            status='passed', source_manifest_ref=manifest_ref, scope=scope, records=records,
            start=started, end=clock(), strict_numerical_gates_passed=True,
            original_TRAIN_focals=22, TRAIN_targets_used=True, optimizer_created=True,
            optimizer_updates_per_independent_trial=1, total_independent_updates=len(records),
            seed=70001, fresh_Transport_parameters=29315, VAL_truth_opened=False, TEST_opened=False,
            full_original_parent=False, ancestry_predictive_quality_established=False,
            useful_GPU_duty_requires_external_telemetry=True,
            limitations=['Same fresh Transport model and true TRAIN targets, three execution routes.',
                         'Each trial resets weights and AdamW; twelve one-step checks are not a training trajectory.',
                         '128-tree prefix truncates external genomic history and contains22 original frozen focals.',
                         'Warm trials rotate all three routes through all positions; allocator cache is emptied between trials.',
                         'Stage synchronization is included in timings; external NVML clocks are required to assess GPU duty.',
                         'No full-parent resource, accuracy-preservation or >80-percent predictive performance claim.']))
        print(json.dumps(dict(status='passed', output=str(output))), flush=True)
    except BaseException as error:
        write('failed.json', dict(status='failed', error_type=type(error).__name__, message=str(error),
              traceback=traceback.format_exc(), start=started, end=clock(), completed_trial_count=len(records)))
        raise


if __name__ == '__main__':
    main()

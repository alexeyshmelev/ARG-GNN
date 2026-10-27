"""CPU proof of bounded valid trace output, union accounting and launch protocol."""
import ast
import json
from types import SimpleNamespace
import torch
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, need, ref, write
from .bounded_timeline_v1 import interval_summary, save_timeline, DiagnosticProfile
from .compare_boundary_timeline_v1 import schedule, KINDS, baseline_factory, fused_factory
from .launch_boundary_timeline_v1 import argv_for
from .large_block_executors import _LargeInitialization, LargeExplicitVJPExecution
from research.efficiency_20261027.architecture.exact_boundary_fusion.large_boundary_executor_v1 import LargeBoundaryFusedExecution

BASE = ROOT / 'research/efficiency_20261027/execution'


def main():
    need(not torch.cuda.is_initialized(), 'CPU-only proof')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    union = interval_summary([(0, 20), (15, 50), (70, 120)], 10, 100)
    need(abs(union['recorded_busy_fraction']-70/90) < 1e-12 and union['gap_count'] == 1
         and union['largest_gaps'][0]['duration_us'] == .02, 'Clip, union overlapping intervals and gaps without double counting')
    empty = interval_summary([], 10, 100)
    need(empty['recorded_busy_fraction'] == 0 and empty['gap_count'] == 1, 'Empty recording activity is explicit')
    diagnostic = DiagnosticProfile(True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False,
            profile_memory=False, with_stack=False) as profile:
        with torch.profiler.record_function('diagnostic_learned_update'):
            with diagnostic.stage('CPU_forward'):
                value = torch.arange(128, dtype=torch.float32, requires_grad=True)
                loss = value.sin().square().sum()
            with diagnostic.stage('CPU_backward'):
                loss.backward()
    need(value.grad is not None and torch.isfinite(value.grad).all(), 'Real CPU forward/backward profiler recording')
    trace_path = BASE / 'bounded_timeline_CPU_trace_v1.json'
    summary = save_timeline(profile, trace_path, max_bytes=2048, event_quota=10)
    parsed = json.loads(trace_path.read_text())
    need(trace_path.stat().st_size <= 2048 and len(parsed['traceEvents']) > 0, 'Strict finite bytes and valid completed Chrome JSON')
    need(summary['recorded_event_counts']['CPU'] > sum(summary['Chrome_trace']['kept_event_counts'].values()),
         'All-recording summary is separate from sampled bounded view')
    need(summary['recorded_CUDA_union']['recorded_busy_fraction'] == 0
         and summary['copy_events_without_positive_byte_metadata'] == {}, 'CPU test cannot invent CUDA activity or transfer bytes')
    orders = schedule()
    need(len(orders) == 4 and sum(len(order) for phase, repeat, order in orders) == 8, 'Finite eight calls')
    timed = [order for phase, repeat, order in orders if phase == 'timed']
    need(timed == [KINDS, tuple(reversed(KINDS))], 'Two unprofiled balanced timed pairs')
    need([phase for phase, repeat, order in orders][-1] == 'profiled', 'Profiles after independent unprofiled timing')
    need(LargeExplicitVJPExecution.__init__ is _LargeInitialization.__init__
         and LargeBoundaryFusedExecution.__init__ is _LargeInitialization.__init__, 'Same admitted512 initializer')
    for name in ('_block', '_sites', '_second'):
        need(getattr(LargeBoundaryFusedExecution, name) is getattr(LargeExplicitVJPExecution, name), 'Inherited original method '+name)
    args = SimpleNamespace(directory='research/efficiency_20261027/root/runs/boundary_timeline_review_placeholder_v1',
        gpu=5, predecessor_directory='research/efficiency_20261027/root/runs/hot_block_transfer_gpu5_v1',
        peer_directory=['research/efficiency_20261027/root/runs/hot_block_transfer_gpu5_v1'], source_pins_sha256='0'*64)
    command = argv_for(args)
    need(command[command.index('--host-gib')+1] == '32' and command[command.index('--device-gib')+1] == '24'
         and command[command.index('--run-seconds')+1] == '1800', 'Finite reviewed guardian budgets')
    need(not (ROOT/args.directory).exists(), 'Argv review creates no output')
    names = ('bounded_timeline_v1.py', 'compare_boundary_timeline_v1.py', 'launch_boundary_timeline_v1.py', 'test_bounded_timeline_v1.py')
    for name in names:
        tree = ast.parse((BASE/name).read_text())
        need(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('eval','exec','compile')
            for node in ast.walk(tree)), 'Readable direct source execution')
    need(not torch.cuda.is_initialized(), 'No CUDA initialized or GPU launched')
    result = dict(status='passed', schema='TreeLoom_bounded_timeline_CPU_protocol_v1',
        source_refs=[ref(BASE/name) for name in names], trace_ref=ref(trace_path), trace_summary=summary,
        interval_union_proof=union, protocol=orders, same512_initializer=True,
        original_block_sites_second_inherited=True, launcher_argv_reviewed_without_launch=True,
        CPU_profile_used_real_forward_backward=True, CUDA_initialized=False,
        actual1153_learned_GPU_qualification=False, no_GPU_performance_claim=True)
    write(BASE/'bounded_timeline_CPU_v1.json', result)
    print(json.dumps(dict(status='passed', result_ref=ref(BASE/'bounded_timeline_CPU_v1.json'),
        trace_bytes=trace_path.stat().st_size, recorded_CPU_events=summary['recorded_event_counts']['CPU'], CUDA_initialized=False)), flush=True)


if __name__ == '__main__':
    main()

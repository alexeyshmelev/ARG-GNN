"""Read-only prerequisites for the separate complete-parent residency pair.

Require pinned successful prefix and full-boundary numerical receipts and
actual reaping of each exact ROOT-owned process family. No waiting, signals,
launches, source edits, scientific payloads or CUDA initialization occur here.
"""
import json
from pathlib import Path
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, need, ref, CHECKPOINT
from research.efficiency_20261027.execution.deferred_gpu_gate import validate_completion, owned_group_members
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity

PREFIX_MANIFEST = 'research/efficiency_20261027/architecture/exact_upward_residency/source_pins_prefix_v1.json'
BOUNDARY_MANIFEST = 'research/efficiency_20261027/root/source_pins_boundary_full_v1.json'
PREFIX_KINDS = ('boundary_fused_CPU512', 'upward_direct_resident_CPUcontext512')
BOUNDARY_KINDS = ('selected_explicit_CPU512', 'boundary_fused_CPU512')


def require_comparison(value):
    need(type(value) is dict and value.get('passed') is True
         and value.get('argmax_equal') is True and value.get('argmax_disagreements') == 0,
         'Strict numerical parity and zero argmax disagreement required')
    expected = dict(logits=1, loss=1, gradients=49, parameters=49, moments=147)
    categories = value.get('categories', {})
    need(set(categories) == set(expected), 'All original numerical categories required')
    for name, count in expected.items():
        row = categories[name]
        need(row.get('passed') is True and row.get('tensors') == count
             and row.get('mismatched_elements') == 0 and row.get('mismatched_paths') == []
             and row.get('structural_errors') == [] and row.get('atol') == 1e-6
             and row.get('rtol') == (1e-5 if name in ('logits', 'loss') else 1e-4),
             'Exact original tolerance and tensor coverage required for '+name)


def validate_numerical_release(value, kind, *, checkpoint_ref, source_manifest_ref):
    need(kind in ('prefix', 'boundary_full') and type(value) is dict,
         'Declared qualification prerequisite')
    schema = ('TreeLoom_upward_direct_residency_prefix_GPU_qualification_v1'
              if kind == 'prefix' else 'TreeLoom_boundary_full_v1')
    need(value.get('status') == 'completed' and value.get('schema') == schema
         and value.get('all_strict_comparisons_passed') is True
         and value.get('checkpoint_ref') == checkpoint_ref
         and value.get('source_manifest_ref') == source_manifest_ref,
         'Pinned successful original-checkpoint qualification required: '+kind)
    scope = value.get('scope', {})
    need(scope.get('parent_id') == 'argformer_american_train_000'
         and scope.get('haplotypes') == 1600 and scope.get('heldout_truth_opened') is False,
         'Actual original TRAIN000 observations without held-out truth required')
    rows = value.get('records')
    need(type(rows) is list, 'Complete prerequisite numerical records required')
    if kind == 'prefix':
        expected = [('warmup', 0, PREFIX_KINDS[0]), ('warmup', 0, PREFIX_KINDS[1]),
                    ('timed', 1, PREFIX_KINDS[0]), ('timed', 1, PREFIX_KINDS[1]),
                    ('timed', 2, PREFIX_KINDS[1]), ('timed', 2, PREFIX_KINDS[0])]
        need(scope.get('native_holds') == 1153 and scope.get('full_parent_claim') is False
             and value.get('actual_49gradient_and_Adam_checks') is True
             and [(r.get('phase'), r.get('repeat'), r.get('kind')) for r in rows] == expected,
             'Exactly the reviewed six-call1153hold prefix must pass')
        need(all(r.get('optimizer_step') == 242 and r.get('parameter_count') == 49
                 and r.get('profiled') is False for r in rows),
             'Original49-parameter Adam242 prefix with no profiler required')
    else:
        need(scope.get('mode') == 'full' and scope.get('full_parent_claim') is True
             and scope.get('native_holds') == 77472 and scope.get('sites') == 227474
             and scope.get('supervised_original_focals') == 10000
             and scope.get('physical_support_bp') == [0, 30000000]
             and [r.get('kind') for r in rows] == list(BOUNDARY_KINDS),
             'Complete original30Mb/all10000target boundary pair must pass')
    for row in rows:
        need(row.get('CPU_context') is True and row.get('hotpack') is False,
             'Original CPU-context/no-hotpack prerequisite')
        require_comparison(row.get('comparison'))
    return dict(kind=kind, schema=schema, numerical_records=len(rows),
        strict_gradient_tensors=49, strict_parameter_tensors=49,
        strict_Adam_tensors=147, argmax_disagreements=0)


def read_metadata(path, maximum=8*2**20):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
         'Bounded ordinary prerequisite metadata required')
    reference = ref(path)
    value = json.loads(path.read_text())
    need(ref(path) == reference, 'Prerequisite metadata changed while reading')
    return value, reference


def release_from_directory(directory, digest, kind):
    need(type(digest) is str and len(digest) == 64
         and all(ch in '0123456789abcdef' for ch in digest),
         'Exact ROOT-selected completion SHA256 required')
    directory = (ROOT/Path(directory)).resolve()
    need(directory.is_relative_to(ROOT/'research/efficiency_20261027/root/runs'),
         'ROOT-owned qualification run directory required')
    completed_path = directory/'output/completed.json'
    completed, completed_ref = read_metadata(completed_path)
    need(completed_ref['sha256'] == digest, 'Exact selected completed qualification changed')
    manifest_path = PREFIX_MANIFEST if kind == 'prefix' else BOUNDARY_MANIFEST
    summary = validate_numerical_release(completed, kind,
        checkpoint_ref=ref(CHECKPOINT), source_manifest_ref=ref(manifest_path))
    launch, launch_ref = read_metadata(directory/'launch.json')
    request, request_ref = read_metadata(directory/'request.json')
    wait, wait_ref = read_metadata(directory/'actual_wait.json')
    need(type(launch.get('pid')) is int and launch['pid'] > 1
         and type(launch.get('process_start_ticks')) is str and launch['process_start_ticks'].isdigit()
         and launch.get('actual_argv') == request.get('command'),
         'Matching exact ROOT launch, request and process-start identity required')
    command = request['command']
    need(type(command) is list and command.count('--output') == 1,
         'Readable original qualification output argument required')
    output_index = command.index('--output')+1
    need(output_index < len(command) and (ROOT/command[output_index]).resolve() == completed_path.parent,
         'Numerical completion must belong to the exact reaped worker output')
    validate_completion(launch, wait, identity(launch['pid']), owned_group_members(launch['pid']))
    return dict(summary, directory=str(directory.relative_to(ROOT)), completed_ref=completed_ref,
        launch_ref=launch_ref, request_ref=request_ref, actual_wait_ref=wait_ref,
        actual_wait=True, owned_group_absent=True, actual_returncode=0)


def verify_prerequisites(args):
    return dict(
        prefix=release_from_directory(args.prefix_run_directory, args.prefix_completed_sha256, 'prefix'),
        boundary_full=release_from_directory(args.boundary_full_run_directory,
                                            args.boundary_full_completed_sha256, 'boundary_full'))

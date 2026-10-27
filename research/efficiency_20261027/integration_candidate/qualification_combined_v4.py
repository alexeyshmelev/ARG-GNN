"""Fail-closed release of the combined sources after independent terminal proofs."""
import math
from pathlib import Path
from arggnn.benchmarking import contracts as c
from .configuration_combined_v4 import configuration, PINS, GIB
from .envelope_combined_v4 import validate_declaration, predecessor_request
from . import entrypoint_v3 as original_entry
from research.efficiency_20261027.architecture.exact_upward_residency.full_release_v2 import (
    require_comparison, validate_numerical_release)
from research.efficiency_20261027.execution.deferred_gpu_gate import validate_completion, owned_group_members
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity

need = original_entry.need
read = original_entry.read
BASE = 'research/efficiency_20261027/'
BOUNDARY_MANIFEST = BASE+'root/source_pins_boundary_full_v1.json'
RESIDENT_MANIFEST = BASE+'architecture/exact_upward_residency/source_pins_full_v2.json'
PREPARATION_MANIFEST = BASE+'execution/source_pins_two_parent_comparison_v2.json'
PREPARATION_SPEC = BASE+'execution/TWO_PARENT_COMPARISON_SPEC_v1.json'
ADAPTIVE_MANIFEST = BASE+'integration_candidate/source_pins_adaptive_v2.json'


def authenticate_manifest(reference):
    manifest = read(reference)
    rows = manifest.get('source_refs')
    need(type(rows) is list and rows and len(rows) == len({r['path'] for r in rows}),
         'Nonempty unique immutable source manifest required')
    for row in rows:
        need(c.file_ref(row['path']) == row, 'Qualification source dependency changed')
    return manifest


def qualification_paths(record, command):
    # The ROOT launcher may write absolute --output paths while c.file_ref
    # serializes relative paths. Preserve ordinary-path and traversal guards
    # before comparing canonical project locations.
    ordinary = original_entry.previous._ordinary_path
    directory = ordinary(record['launch_ref']['path'], regular_file=True).parent
    need(directory.is_relative_to(c.ROOT/BASE/'root/runs')
         and ordinary(record['actual_wait_ref']['path'], regular_file=True).parent == directory
         and ordinary(record['request_ref']['path'], regular_file=True).parent == directory,
         'One ROOT-owned exact qualification family required')
    need(type(command) is list and command.count('--output') == 1
         and command.index('--output')+1 < len(command), 'Explicit worker output argument required')
    output = ordinary(command[command.index('--output')+1])
    completed = ordinary(record['completed_ref']['path'], regular_file=True)
    need(output.is_dir() and output.parent == directory and completed == output/'completed.json',
         'Completion must belong to the exact launched ordinary worker output')
    return directory, output


def terminal_run(record, module, manifest_path):
    need(type(record) is dict and set(record) == {'completed_ref', 'request_ref', 'launch_ref', 'actual_wait_ref'},
         'Exact completed/request/launch/actual-wait references required')
    completed, request, launch, wait = (read(record[key]) for key in
        ('completed_ref', 'request_ref', 'launch_ref', 'actual_wait_ref'))
    command = request.get('command')
    need(type(command) is list and command == launch.get('actual_argv')
         and command.count('-m') == command.count('--output') == 1
         and command[command.index('-m')+1] == module,
         'Exact readable reviewed worker launch required')
    qualification_paths(record, command)
    validate_completion(launch, wait, identity(launch['pid']), owned_group_members(launch['pid']))
    manifest_ref = c.file_ref(manifest_path)
    need(completed.get('source_manifest_ref') == manifest_ref,
         'Qualification must bind the exact independently reviewed source manifest')
    authenticate_manifest(manifest_ref)
    if 'source_refs' in completed:
        need(completed['source_refs'] == read(manifest_ref)['source_refs'], 'Complete source list must match its manifest')
    return completed


def validate_resident(value, boundary_ref):
    need(value.get('schema') == 'TreeLoom_upward_direct_residency_full_v1'
         and value.get('status') == 'completed' and value.get('all_strict_comparisons_passed') is True
         and value.get('all49_gradient_and_parameter_tensors_checked') is True
         and value.get('all147_Adam_tensors_checked') is True
         and value.get('device_soft_budget_bytes') == 74*GIB
         and value.get('prerequisites', {}).get('boundary_full', {}).get('completed_ref') == boundary_ref,
         'Completed full-residency chain with all49 gradients/parameters and147 Adam tensors required')
    scope = value.get('scope', {})
    need(scope.get('parent_id') == 'argformer_american_train_000'
         and scope.get('mode') == 'full' and scope.get('full_parent_claim') is True
         and scope.get('native_holds') == 77472 and scope.get('native_nodes') == 205369272
         and scope.get('sites') == 227474 and scope.get('supervised_original_focals') == 10000
         and scope.get('haplotypes') == 1600 and scope.get('heldout_truth_opened') is False
         and scope.get('physical_support_bp') == [0, 30000000],
         'Exact complete original30Mb TRAIN000 residency qualification required')
    rows = value.get('records')
    need(type(rows) is list and [r.get('kind') for r in rows] ==
         ['boundary_fused_CPU512', 'upward_direct_resident_CPUcontext512'],
         'One complete boundary/resident pair required')
    for row in rows:
        require_comparison(row.get('comparison'))
        need(row.get('CPU_threads') == 8 and row.get('holds_per_block') == 512
             and row.get('CPU_context') is True and row.get('full_history') is True
             and row.get('hotpack') is False and row.get('parameter_count') == 49
             and row.get('optimizer_step') == 242
             and 0 < row.get('torch_peak_allocated_bytes', 0) <= row.get('torch_peak_reserved_bytes', 0) <= 74*GIB,
             'Qualified original CPU-context partitions and finite measured device envelope required')
    resident = rows[1]
    bank = 205369272*32*4
    plan = resident.get('resident_bank_plan', {})
    receipt = resident.get('explicit_vjp', {})
    lifetime = receipt.get('residency_lifetime', {})
    need(plan.get('upward_bank_bytes') == plan.get('direct_upward_gradient_bank_bytes') == bank
         and plan.get('maximum_simultaneous_bank_bytes') == 2*bank
         and plan.get('scratch_reserve_bytes') == 16*GIB
         and lifetime.get('peak_bank_bytes') == 2*bank
         and lifetime.get('direct_gradient_bytes') == lifetime.get('future_bank_bytes') == 0
         and receipt.get('first_forcing_forward_device_fused') is True
         and receipt.get('first_forcing_backward_device_fused') is True
         and receipt.get('full_history_backward_calls') == 1,
         'Both complete resident banks, exact scratch floor and released direct VJP lifetime required')


def validate_preparation(value, spec):
    need(value.get('schema') == 'TreeLoom_two_parent_preparation_comparison_v2'
         and value.get('status') == 'passed'
         and value.get('exact_serial_concurrent_signatures') is True
         and value.get('saved_CPU_RNG_unchanged') is True
         and value.get('actual_waits_and_all_tracked_families_absent') is True
         and value.get('training_executed') is False and value.get('truth_opened') is False
         and value.get('spec_ref') == c.file_ref(PREPARATION_SPEC),
         'Completed observation-only exact real serial/concurrent comparison required')
    from research.efficiency_20261027.execution.preparation_evidence_v2 import validate_signature, SCHEMA
    need(value.get('signature_encoding') == 'canonical_complete_JSON_SHA256_v2',
         'Complete canonical bounded signature encoding required')
    rows = value.get('arms')
    need(type(rows) is list and [r.get('arm') for r in rows] ==
         ['current_serial_pair', 'adaptive_concurrent_pair'], 'Exactly one real pair per arm required')
    for index, row in enumerate(rows):
        seconds = row.get('pair_prepare_pack_seconds')
        monitor = row.get('family_monitor', {})
        peaks = monitor.get('peaks', {})
        need(row.get('parents') == spec['parents'] and type(seconds) in (int, float)
             and math.isfinite(seconds) and seconds > 0
             and row.get('supply_seconds_per_parent') == seconds/2
             and type(monitor.get('samples')) is int and monitor['samples'] > 0
             and 0 < peaks.get('main_RSS_bytes', 0) <= 288*GIB
             and 0 < peaks.get('family_PSS_or_RSS_bytes', 0) <= 288*GIB
             and 0 < peaks.get('encoder_children', 0) <= (4 if index == 0 else 8),
             'Measured complete-pair timing and actual bounded main/family/child samples required')
    signatures = []
    for row in rows:
        need(row.get('signature_schema') == SCHEMA
             and type(row.get('signature_refs')) is dict
             and set(row['signature_refs']) == set(spec['parents']),
             'Every complete parent requires a durable bounded exact signature')
        signatures.append({parent: validate_signature(read(row['signature_refs'][parent]))
                           for parent in spec['parents']})
    need(signatures[0] == signatures[1], 'All arrays, scalars, structures and provenance must match')
    need(rows[1]['family_monitor']['peaks']['encoder_children'] > 4,
         'The real comparison must actually observe concurrent encoder families')
    ratio = rows[0]['pair_prepare_pack_seconds']/rows[1]['pair_prepare_pack_seconds']
    need(value.get('pair_speed_ratio') == ratio and ratio > 1.,
         'This combined adaptive2 selection requires a measured positive preparation supply benefit')


def validate_release(reference):
    release = read(reference)
    pins = c.file_ref(PINS)
    need(release.get('schema') == 'TreeLoom_combined_execution_release_v1'
         and release.get('status') == 'independently_approved'
         and release.get('source_pins_ref') == pins,
         'ROOT-sealed independently reviewed combined candidate release required')
    selected = configuration(release.get('configuration'))
    authenticate_manifest(pins)
    adaptive_review = read(c.file_ref(BASE+'fidelity/ADAPTIVE_PREFETCH_V2_SOURCE_REVIEW.json'))
    preparation_review = read(c.file_ref(BASE+'fidelity/TWO_PARENT_PREPARATION_V2_SOURCE_REVIEW.json'))
    need(adaptive_review.get('passed') is True
         and adaptive_review.get('candidate_manifest_ref') == c.file_ref(ADAPTIVE_MANIFEST)
         and preparation_review.get('passed') is True
         and preparation_review.get('source_manifest_ref') == c.file_ref(PREPARATION_MANIFEST),
         'Previously reviewed adaptive joint accounting and actual comparison protocol must remain bound')
    prior = validate_declaration(release.get('resource_declaration'))
    old = predecessor_request(prior)
    need(release.get('original_execution_release_ref') == old['execution_release_ref'],
         'Retain the exact attempt7 original numerical-anchor release')
    original = original_entry.validate_release(release['original_execution_release_ref'])
    need(all(selected[key] == value for key, value in original['configuration'].items()),
         'Original selected execution partition must remain unchanged')
    boundary = terminal_run(release['boundary_full'], BASE.replace('/', '.')+'root.compare_boundary_full_v1', BOUNDARY_MANIFEST)
    validate_numerical_release(boundary, 'boundary_full', checkpoint_ref=boundary['checkpoint_ref'],
        source_manifest_ref=c.file_ref(BOUNDARY_MANIFEST))
    resident = terminal_run(release['residency_full'],
        BASE.replace('/', '.')+'architecture.exact_upward_residency.compare_full_v2', RESIDENT_MANIFEST)
    need(resident.get('checkpoint_ref') == boundary['checkpoint_ref'], 'Same qualification checkpoint for boundary and residency')
    validate_resident(resident, release['boundary_full']['completed_ref'])
    preparation = terminal_run(release['preparation_comparison'],
        BASE.replace('/', '.')+'execution.compare_two_parent_preparation_v2', PREPARATION_MANIFEST)
    validate_preparation(preparation, read(c.file_ref(PREPARATION_SPEC)))
    acceptance = release.get('acceptance')
    need(acceptance == dict(residency_full_numerically_and_operationally_accepted=True,
         preparation_supply_benefit_accepted=True, one_fixed_order_pair_limitation_accepted=True,
         training_contention_speedup_claimed=False, sustained_GPU_utilization_claimed=False,
         deployment_uses_then_latest_checkpoint=True), 'Explicit bounded acceptance of actual evidence required')
    review = read(release['independent_integration_review_ref'])
    need(review.get('status') in ('pass', 'passed') and review.get('candidate_source_pins_ref') == pins,
         'Independent combined source review must bind this exact immutable candidate')
    result_review = read(release['independent_results_review_ref'])
    need(result_review.get('status') in ('pass', 'passed')
         and result_review.get('candidate_source_pins_ref') == pins
         and result_review.get('boundary_full_ref') == release['boundary_full']['completed_ref']
         and result_review.get('residency_full_ref') == release['residency_full']['completed_ref']
         and result_review.get('preparation_comparison_ref') == release['preparation_comparison']['completed_ref'],
         'Independent terminal-evidence review must bind every selected completed result')
    from . import training_combined_v4 as training
    training.requirements(selected)
    return release

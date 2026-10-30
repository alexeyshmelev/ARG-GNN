"""Read-only consumer validation after root publication; report native outcome honestly."""
import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import successor_v2_gpu as gpu
from research.joint_evaluation_20261029.root import finish_coordinator_storage_v2 as finish

BASE = 'research/joint_evaluation_20261029'
FOLDER = BASE + '/queue/gpu_priority_after_fits_20261030_v1'
CANDIDATE = FOLDER + '/release_candidate_v1.json'
AMENDMENT = FOLDER + '/policy_amendment_v1.json'
EXPECTED_CANDIDATE = '2422c456731bcdbad4bf2fd81007fc44ea024cd444016c2e8874394d4380ef8c'
EXPECTED_AMENDMENT = 'e11dabe5edcf0919395e3aa4e03a94ca4897d2d7632dd430aed1d7eb8468db7c'


def need(value, message):
    if not value:
        raise ValueError(message)


def blocked_write(*args, **kwargs):
    raise AssertionError('Consumer validation must not write or publish')


candidate_ref = r.ref(CANDIDATE)
amendment_ref = r.ref(AMENDMENT)
need(candidate_ref['sha256'] == EXPECTED_CANDIDATE, 'Candidate hash changed')
need(amendment_ref['sha256'] == EXPECTED_AMENDMENT, 'Amendment hash changed')
candidate = r.read(candidate_ref)
amendment = r.read(amendment_ref)
need(candidate['amendment_ref'] == amendment_ref, 'Exact amendment binding required')
need(candidate['priority_only'] and not candidate['all_predictions_complete'], 'Honest priority-only scope required')
need(candidate['argmix_scope'] == 'fit_only', 'ARGMix fit-only scope required')
need(candidate['remaining_ARGMix_predictions_preserved'] == 9, 'Remaining prediction work preserved')
production_gate = r.path(amendment['publication_destination'])
need(production_gate.samefile(r.path(CANDIDATE)), 'Published gate must be the same reviewed inode')
published_ref = r.ref(production_gate)
need(published_ref['sha256'] == EXPECTED_CANDIDATE, 'Published gate content changed')
publication_ref = r.ref(FOLDER + '/publication_receipt_v1.json')
publication = r.read(publication_ref)
need(publication['release_sha256'] == EXPECTED_CANDIDATE, 'Publication binding changed')

for source in amendment['source_refs']:
    finish.check_reference(source)
plan = finish.verify_reference(amendment['finish_plan_ref'])
config = r.read(plan['successor_config_ref'])
template = dict(r.read(config['successor_template_ref']), handoff_config_ref=plan['successor_config_ref'])
candidate_template = dict(template, root_gpu_priority_release_path=CANDIDATE)
native_plan = finish.verify_reference(amendment['native_execution_plan_ref'])
task = next(row for row in native_plan['tasks'] if row['id'] == amendment['immediate_target_task_id'])
need(task['argv'] == amendment['original_native_task_argv'], 'Original native argv changed')
for key, value in amendment['original_native_task_limits'].items():
    need(task[key] == value, 'Original native resource changed: ' + key)
for source in task['source_refs']:
    finish.check_reference(source)
finish.waited(amendment['fit_inputs_actual_wait_ref'])
for reference in candidate['completed_models'].values():
    finish.waited(reference)
need(template['root_gpu_priority_release_path'] == amendment['publication_destination'], 'Destination changed')
need(gpu.LOCK_PATH == amendment['original_GPU3_mutex_path'], 'Shared GPU3 lock changed')

with patch.object(finish, 'immutable', blocked_write), patch.object(r, 'save', blocked_write):
    candidate_release_ref = gpu.root_release(candidate_template)
    actual_release_ref = gpu.root_release(template)
    allowed = gpu.allowed(template)
    chains = finish.chain_completions(plan)
    candidate_finisher_release = finish.maybe_release(plan, candidate_template, chains)
    actual_finisher_release = finish.maybe_release(plan, template, chains)
need(candidate_release_ref == candidate_ref == candidate_finisher_release, 'Candidate consumer mismatch')
need(actual_release_ref == published_ref == actual_finisher_release, 'Published gate consumer mismatch')
need(allowed == [3], 'Current admission is not GPU3-only')
need(set(chains) == {'american_joint', 'european_four_joint'}, 'Unexpected full-chain completion map')

state = json.loads(r.path(BASE + '/queue/successor_records_v2/state.json').read_text())
need(state['controller_pid'] == amendment['existing_native_owner_pid'], 'Native owner changed')
need(state['plan_ref'] == amendment['native_execution_plan_ref'], 'Native owner plan changed')
native_row = state['tasks'][task['id']]
need(native_row['command'] == task['argv'] and native_row['physical_gpu'] == 3, 'Original task not launched on GPU3')
need(native_row['status'] == 'failed' and native_row['actual_returncode'] == 1, 'Native failure outcome changed; inspect')
need(native_row['actual_wait'] and native_row['owned_process_group_absent'], 'Missing genuine failed wait')
need(not processes.session_members(native_row['pid']), 'Native failed worker session still live')
native_wait_ref = r.ref(r.path(task['output_directory']) / 'actual_wait.json')
need(r.read(native_wait_ref) == native_row, 'Saved failed wait differs from scheduler state')
log_ref = r.ref(native_row['log_path'])
need('RuntimeError: Invalid device argument' in r.path(log_ref['path']).read_text(), 'Unexpected failure reason')
need(r.ref(production_gate) == published_ref, 'Published gate changed during validation')

receipt = dict(
    schema='GPU_priority_published_consumer_validation_v2',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    admission_validation_passed=True, training_complete=False,
    candidate_ref=candidate_ref, amendment_ref=amendment_ref, publication_receipt_ref=publication_ref,
    validation_source_ref=r.ref(__file__), source_refs=amendment['source_refs'],
    native_plan_ref=amendment['native_execution_plan_ref'], native_source_refs_checked=len(task['source_refs']),
    genuine_priority_wait_refs=candidate['completed_models'], fit_inputs_wait_ref=amendment['fit_inputs_actual_wait_ref'],
    allowed_physical_GPUs=allowed, full_chain_completions=chains,
    consumer_root_release_ref=actual_release_ref, consumer_finisher_release_ref=actual_finisher_release,
    all_predictions_complete=False, ARGMix_full_GPU_chain_complete=False,
    same_inode_as_reviewed_candidate=True, existing_owner_admitted_native_task=True,
    native_task_outcome='failed_before_training_at_cuda_reset_peak_memory_stats',
    native_actual_wait_ref=native_wait_ref, native_log_ref=log_ref,
    native_controller_pid=state['controller_pid'], native_state_observed_updated_unix=state['updated_unix'],
    launches_by_validation=0, signals_by_validation=0, production_gate_writes_by_validation=0,
    source_plan_state_rewrites_by_validation=0, only_output=FOLDER + '/consumer_validation_v2.json')
with r.path(receipt['only_output']).open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'admission_validation_passed': True, 'receipt_ref': r.ref(receipt['only_output']),
                  'allowed': allowed, 'full_chain_completions': sorted(chains),
                  'native_task_outcome': receipt['native_task_outcome'], 'native_actual_wait_ref': native_wait_ref}, indent=2))

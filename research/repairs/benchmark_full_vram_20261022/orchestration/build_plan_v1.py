"""Prepare unchanged benchmark science with full GPU memory and preserved live jobs."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
from . import storage_workflow_v1 as facade
from research.repairs.benchmark_resume_20261021.native import plan_paths_v1 as paths

queue = facade.queue
c = queue.c
HERE = Path(__file__).parent
ACTUAL = HERE / 'actual_v1'
OUTPUT = 'experiments/results/benchmark_campaign_20261018_workflow/release_20261022_full_vram_v1'
PLAN = 'configs/benchmarks/executable_workflow_20261022_full_vram_v1.json'
RECORDS = '.runtime/benchmarking/workflow_20261022_full_vram_v1'
AM = 'AM_treeloom_fit'
AP = 'AP__all_train__treeloom_fit'
GPU_UUIDS = {2: 'GPU-976a5cd3-df12-2535-fa51-e3ea063688ba', 3: 'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786', 5: 'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'}


def bind_active(task, entry, old_ref, snapshot_ref, source_state_path):
    supervisor = queue.shared_gpu.identity(entry['pid'])
    queue.need(supervisor is not None, 'Preserved supervisor must remain live at snapshot')
    identity_ref = c.file_ref(c.contained_path(task['output_directory']) / 'supervision_identity.json')
    identity = c.read_ref(identity_ref)['child_identity']
    binding = dict(source_plan_ref=old_ref, source_state_path=str(source_state_path.relative_to(c.ROOT)),
        source_state_snapshot_ref=snapshot_ref, source_task_id=task['id'], source_task_fingerprint=c.fingerprint(task),
        original_job_ref=entry['job_ref'], original_supervision_identity_ref=identity_ref,
        original_supervisor_pid=entry['pid'], original_supervisor_start_ticks=supervisor['start_ticks'],
        owned_process_group_id=identity['pid'], owned_group_leader_start_ticks=identity['start_ticks'],
        owned_group_leader_session=identity['session'], max_rss_bytes=task['max_rss_bytes'],
        max_output_bytes=task['max_output_bytes'], resource=task['resource'], output_directory=task['output_directory'])
    queue.support.predecessor._source(binding)
    return binding


def build():
    policy = c.read_json(c.POLICY_PATH)
    old_ref = policy['approved_recovery_continuation_workflow_ref']
    queue.need(policy.get('recovery_continuation_workflow_drain_ref') == old_ref
               and policy.get('gpu_memory_policy') == 'full_device', 'Explicit full VRAM authority and future-dispatch drain required')
    old = c.read_ref(old_ref)
    source_state_path = c.contained_path(old['records_directory']) / 'state.json'
    state = c.read_json(source_state_path)
    queue.need(state['plan_ref'] == old_ref, 'Current canonical state required')
    ACTUAL.mkdir(exist_ok=False)
    snapshot_ref = queue.save(ACTUAL / 'source_state_snapshot.json', state, exclusive=True)
    plan = deepcopy(old)
    by_id = {task['id']: task for task in old['tasks']}
    plan['adopted_state_snapshot_ref'] = snapshot_ref
    plan['adopted_completed_task_fingerprints'] = {identifier: c.fingerprint(by_id[identifier])
        for identifier, entry in state['tasks'].items() if entry['status'] == 'succeeded'}
    active_ids = {identifier for identifier, entry in state['tasks'].items() if entry['status'] == 'running'}
    queue.need(active_ids <= {'C4__all_train__c4_treeloom_fit', 'AM_argformer_american_validation_003_argformer_embeddings'},
               'Reconcile unexpected active work before rebasing')
    bindings = deepcopy(plan['workflow_external_predecessors'])
    failed_jobs = {}
    failure_retention = []
    for identifier in (AM, AP):
        rows = [b for b in bindings if b['source_task_id'] == identifier]
        queue.need(len(rows) == 1, 'Exact failed predecessor binding required')
        binding = rows[0]
        _, entry = queue.support.predecessor._source(binding)
        terminal = queue.support.predecessor._drained(binding, entry)
        queue.need(entry['status'] == 'failed' and entry.get('actual_controller_wait') == 1
                   and terminal is not None and terminal[1]['actual_direct_returncode'] == 1,
                   'Retry only a genuinely failed and drained scientific worker')
        job = c.read_ref(binding['original_job_ref'])
        failed_jobs[identifier] = job
        retained = queue.scope_bytes(c.contained_path(binding['output_directory']))
        queue.need(retained == entry['retained_bytes'], 'Closed failure scope must preserve its actual recorded charge')
        evidence = queue.save(ACTUAL / (identifier + '_retained_failure.json'),
            dict(original_binding=binding, original_wait_ref=terminal[0], retained_bytes=retained,
                 files_deleted=False, scientific_settings_changed=False), exclusive=True)
        plan['external_storage'].append(dict(cap_bytes=retained, scopes=[binding['output_directory']],
            completion_evidence=[], evidence_ref=evidence, reason='Preserve drained failure evidence and completed checkpoints'))
        failure_retention.append(evidence)
        bindings.remove(binding)
    active_bindings = {}
    for identifier in sorted(active_ids):
        binding = bind_active(by_id[identifier], state['tasks'][identifier], old_ref, snapshot_ref, source_state_path)
        queue.need(not any(b['source_task_id'] == identifier for b in bindings), 'No duplicate live predecessor')
        bindings.append(binding)
        active_bindings[identifier] = binding
    plan['workflow_external_predecessors'] = bindings
    new_tasks = []
    for original in old['tasks']:
        identifier = original['id']
        entry = state['tasks'][identifier]
        if identifier in active_ids:
            task = dict(id=identifier, dependencies=deepcopy(original['dependencies']), resource='external',
                        workflow_predecessor=active_bindings[identifier])
        elif identifier in (AM, AP):
            task = deepcopy(failed_jobs[identifier]['task'])
            task['factory_module'] = ('research.repairs.benchmark_full_vram_20261022.training.fit_v1' if identifier == AM
                                      else 'research.repairs.benchmark_full_vram_20261022.training.ap_fit_v1')
            task['output_directory'] = OUTPUT + '/other/' + identifier
            task.pop('max_cuda_reserved_bytes', None)
            if identifier == AM:
                latest = c.read_json(c.contained_path(failed_jobs[identifier]['call']['spec']['output_directory']) / 'latest_checkpoint.json')
                queue.need(latest['complete_parent_updates'] == 6, 'Use the actual six-update checkpoint, never replay it')
                task['factory_arguments']['resume_checkpoint_ref'] = latest['checkpoint_ref']
                task['minimum_free_gpu_bytes'] = 64 * 2**30
            else:
                queue.need('resume_checkpoint_ref' not in task['factory_arguments'], 'European API has no resume argument')
                task['minimum_free_gpu_bytes'] = 64 * 2**30
                task['physical_gpu'] = 5
                task['gpu_uuid'] = GPU_UUIDS[5]
                task['factory_arguments']['physical_gpu'] = 5
                task['factory_arguments']['gpu_uuid'] = GPU_UUIDS[5]
            task['gpu_memory_policy'] = 'full_device'
        else:
            task = deepcopy(original)
            if entry['status'] != 'succeeded' and task['resource'] != 'external':
                queue.need(not entry.get('job_ref') and not entry.get('reservation_bytes'), 'Reconcile every launched scope before rebasing')
                if task.get('factory_module') != paths.COMPETITORS:
                    task['output_directory'] = OUTPUT + '/other/' + identifier
                    queue.need(not c.contained_path(task['output_directory']).exists(), 'Fresh future output required')
        new_tasks.append(task)
    plan['tasks'] = new_tasks
    # Preserve historical successes. Resolve only never-launched native paths.
    plan, path_receipt = paths.repair_plan_paths(plan, state, OUTPUT)
    plan['retention_owner_rebinding'] = paths.rebind_owner_paths(plan, state)
    for task in plan['tasks']:
        identifier = task['id']
        if state['tasks'][identifier]['status'] == 'succeeded' or task['resource'] == 'external':
            continue
        if task['resource'] == 'gpu0' or task.get('gpu_after_preparation') is True:
            task['gpu_memory_policy'] = 'full_device'
            task.pop('max_cuda_reserved_bytes', None)
            if identifier != AM and identifier != AP:
                task['physical_gpu'] = 3
                task['gpu_uuid'] = GPU_UUIDS[3]
                if 'physical_gpu' in task.get('factory_arguments', {}):
                    task['factory_arguments']['physical_gpu'] = 3
                    task['factory_arguments']['gpu_uuid'] = GPU_UUIDS[3]
    plan['records_directory'] = RECORDS
    plan['continuation_approval_key'] = 'approved_full_vram_workflow_ref'
    plan['operational_successor_of'] = old_ref
    plan['execution_authority'] = 'Explicit user full-device VRAM permission on physical GPUs 2, 3 and 5'
    plan['gpu_memory_policy'] = 'full_device'
    plan['inherited_supplemental_archive_credit'] = dict(original_plan_ref=old_ref,
        original_state_snapshot_ref=snapshot_ref, completed_reduction_bytes=20789874688)
    plan['full_vram_failure_retention_refs'] = failure_retention
    plan['full_vram_live_predecessors'] = sorted(active_ids)
    plan['full_vram_scientific_settings_changed'] = False
    plan['full_vram_assignments'] = dict(American_fit=2, European_AP_fit=5, future_native_and_prediction=3,
        preserved_running_workers_keep_original_devices=True)
    plan['recovery_paths_evidence_ref'] = queue.save(ACTUAL / 'path_rebinding.json', path_receipt, exclusive=True)
    new_modules = ['research.repairs.benchmark_full_vram_20261022.training.fit_v1',
                   'research.repairs.benchmark_full_vram_20261022.training.ap_fit_v1']
    plan['readable_factory_modules'] = sorted(set(plan.get('readable_factory_modules', []) + new_modules))
    references = {ref['path']: ref for ref in old['sources']}
    for directory in [HERE.parent / 'runtime', HERE.parent / 'training', HERE]:
        for path in sorted(directory.glob('*.py')):
            ref = c.file_ref(path)
            references[ref['path']] = ref
    plan['sources'] = [references[key] for key in sorted(references)]
    reference = queue.save(PLAN, plan, exclusive=True)
    queue.load_plan(reference)
    initial = queue.initial_state(plan, reference)
    projection = queue.storage_projection(plan, initial)
    queue.need(projection < plan['storage_ceiling_bytes'], 'Conservative inherited storage charge must remain under ceiling')
    queue.need(facade.historical_credit(plan, initial) == 20789874688, 'Retain genuine completed archive credit exactly once')
    receipt = dict(plan_ref=reference, source_state_snapshot_ref=snapshot_ref,
        counts=dict(Counter(entry['status'] for entry in initial['tasks'].values())), tasks=len(plan['tasks']),
        healthy_workers_preserved=sorted(active_ids), storage_projection_bytes=projection,
        genuine_historical_archive_credit_bytes=20789874688, original_AM_updates_restored=6,
        remaining_AM_updates=594, original_AP_budget=90, full_gpu_memory=True,
        scientific_settings_changed=False, scientific_launches=0, signals_sent=0)
    queue.save(ACTUAL / 'preparation_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, indent=2), flush=True)

if __name__ == '__main__':
    build()

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
OUTPUT = 'experiments/results/benchmark_campaign_20261018_workflow/release_20261022_gpu345_v1'
PLAN = 'configs/benchmarks/executable_workflow_20261022_gpu345_v1.json'
RECORDS = '.runtime/benchmarking/workflow_20261022_gpu345_v1'
AM = 'AM_treeloom_fit'
AP = 'AP__all_train__treeloom_fit'
GPU_UUIDS = {4: 'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71', 3: 'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786', 5: 'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'}


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
    old_ref = policy['approved_full_vram_workflow_ref']
    queue.need(policy.get('full_vram_workflow_drain_ref') == old_ref
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
    queue.need(active_ids == {'AM_argformer_american_validation_004_argformer_mlm', AP}, 'Preserve the two exact healthy local GPU3/5 jobs')
    bindings = deepcopy(plan['workflow_external_predecessors'])
    failed_jobs = {}
    failure_retention = []
    for identifier in (AM,):
        entry = state['tasks'][identifier]
        queue.need(entry['status'] == 'failed' and entry.get('actual_controller_wait') == 1
                   and entry.get('owned_process_group_absent') is True, 'Exact failed or cancelled local attempt must be drained')
        job = c.read_ref(entry['job_ref'])
        task = by_id[identifier]
        wait_ref = c.file_ref(c.contained_path(task['output_directory']) / 'actual_wait.json')
        wait = c.read_ref(wait_ref)
        queue.need(job['plan_ref'] == old_ref and job['task'] == task
                   and job['task_fingerprint'] == c.fingerprint(task) and wait['job_ref'] == entry['job_ref'],
                   'Failed attempt must bind the exact original plan, task, fingerprint and actual wait')
        queue.need(wait.get('actual_direct_wait') is True and wait.get('owned_process_group_absent') is True
                   and wait.get('actual_direct_returncode') in (1, -15), 'Genuine failed or cancelled worker wait required')
        control = c.read_json(HERE / 'GPU2_AM_control_receipt.json')
        queue.need(control.get('signal_dispatched') is True and control['job_ref'] == entry['job_ref'], 'Exact user-directed GPU2 migration cancellation required')
        binding = dict(output_directory=task['output_directory'], original_job_ref=entry['job_ref'])
        terminal = (wait_ref, wait)
        failed_jobs[identifier] = job
        retained = queue.scope_bytes(c.contained_path(binding['output_directory']))
        queue.need(retained == entry['retained_bytes'], 'Closed failure scope must preserve its actual recorded charge')
        evidence = queue.save(ACTUAL / (identifier + '_retained_failure.json'),
            dict(original_binding=binding, original_wait_ref=terminal[0], retained_bytes=retained,
                 files_deleted=False, scientific_settings_changed=False), exclusive=True)
        plan['external_storage'].append(dict(cap_bytes=retained, scopes=[binding['output_directory']],
            completion_evidence=[], evidence_ref=evidence, reason='Preserve drained failure evidence and completed checkpoints'))
        failure_retention.append(evidence)
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
        elif identifier == AM:
            task = deepcopy(failed_jobs[identifier]['task'])
            task['factory_module'] = 'research.repairs.benchmark_gpu345_20261022.training.fit_v1'
            task['output_directory'] = OUTPUT + '/other/' + identifier
            task.pop('max_cuda_reserved_bytes', None)
            old_native = c.contained_path(failed_jobs[identifier]['call']['spec']['output_directory'])
            latest = old_native / 'latest_checkpoint.json'
            if latest.is_file():
                saved = c.read_json(latest)
                checkpoint = saved['checkpoint_ref']
                completed_updates = saved['complete_parent_updates']
            else:
                checkpoint = failed_jobs[identifier]['call']['spec']['resume_checkpoint_ref']
                completed_updates = control['checkpoint_before_signal']['complete_parent_updates']
            queue.need(checkpoint is not None and c.file_ref(checkpoint['path']) == checkpoint, 'Retain the exact latest completed-update checkpoint')
            task['factory_arguments']['resume_checkpoint_ref'] = checkpoint
            task['factory_arguments']['physical_gpu'] = 4
            task['factory_arguments']['gpu_uuid'] = GPU_UUIDS[4]
            task['physical_gpu'] = 4
            task['gpu_uuid'] = GPU_UUIDS[4]
            task['minimum_free_gpu_bytes'] = 64 * 2**30
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
        if task['resource'] == 'cpu' and task.get('gpu_after_preparation') is not True:
            for unused in ('physical_gpu', 'gpu_uuid', 'minimum_free_gpu_bytes', 'max_cuda_reserved_bytes'):
                task.pop(unused, None)
        if task['resource'] == 'gpu0' or task.get('gpu_after_preparation') is True:
            task['gpu_memory_policy'] = 'full_device'
            task.pop('max_cuda_reserved_bytes', None)
            if identifier != AM:
                task['physical_gpu'] = 3
                task['gpu_uuid'] = GPU_UUIDS[3]
                if 'physical_gpu' in task.get('factory_arguments', {}):
                    task['factory_arguments']['physical_gpu'] = 3
                    task['factory_arguments']['gpu_uuid'] = GPU_UUIDS[3]
    plan['records_directory'] = RECORDS
    plan['continuation_approval_key'] = 'approved_gpu345_workflow_ref'
    plan['operational_successor_of'] = old_ref
    plan['execution_authority'] = 'User migration from GPU2 to GPU4. Only GPUs3/4/5 may execute new work, with full-device VRAM.'
    plan['physical_gpus'] = [3, 4, 5]
    plan['gpu_memory_policy'] = 'full_device'
    plan['inherited_supplemental_archive_credit'] = deepcopy(old['inherited_supplemental_archive_credit'])
    plan['GPU4_migration_retention_refs'] = failure_retention
    plan['GPU4_migration_control_ref'] = c.file_ref(HERE / 'GPU2_AM_control_receipt.json')
    plan['GPU4_migration_source_checkpoint_ref'] = checkpoint
    plan['GPU4_migration_completed_updates'] = completed_updates
    plan['full_vram_live_predecessors'] = sorted(active_ids | {'C4__all_train__c4_treeloom_fit'})
    plan['full_vram_scientific_settings_changed'] = False
    plan['full_vram_assignments'] = dict(American_fit=4, European_AP_fit=5, future_native_and_prediction=3,
        preserved_running_workers_keep_original_devices=True)
    plan['recovery_paths_evidence_ref'] = queue.save(ACTUAL / 'path_rebinding.json', path_receipt, exclusive=True)
    new_modules = ['research.repairs.benchmark_gpu345_20261022.training.fit_v1']
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
        healthy_workers_preserved=plan['full_vram_live_predecessors'], storage_projection_bytes=projection,
        genuine_historical_archive_credit_bytes=20789874688, original_AM_updates_restored=completed_updates,
        remaining_AM_updates=600-completed_updates, original_AP_budget=90, full_gpu_memory=True,
        scientific_settings_changed=False, scientific_launches=0, signals_sent=0)
    queue.save(ACTUAL / 'preparation_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, indent=2), flush=True)

if __name__ == '__main__':
    build()

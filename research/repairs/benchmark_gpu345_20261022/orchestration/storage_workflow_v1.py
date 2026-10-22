"""Full-device workflow with authenticated historical lossless storage credit."""
from research.repairs.benchmark_gpu345_20261022.runtime import readable_workflow_v1 as queue
from research.repairs.benchmark_repair_20261022_afternoon.storage_ap_train001_v1 import external_archive_v1 as archive

MODULE = 'research.repairs.benchmark_gpu345_20261022.orchestration.storage_workflow_v1'
SOURCE = 'research/repairs/benchmark_gpu345_20261022/orchestration/storage_workflow_v1.py'
BASE_LOAD = queue.load_plan
BASE_PROJECTION = queue.storage_projection


def historical_credit(plan, state=None):
    proof = plan['inherited_supplemental_archive_credit']
    original = queue.c.read_ref(proof['original_plan_ref'])
    snapshot = queue.c.read_ref(proof['original_state_snapshot_ref'])
    queue.need(snapshot['plan_ref'] == proof['original_plan_ref'], 'Authentic archive plan and state required')
    by_id = {task['id']: task for task in plan['tasks']}
    original_by_id = {task['id']: task for task in original['tasks']}
    identifier = original[archive.FIELD]['task_id']
    owner = archive.OWNER
    queue.need(plan[archive.FIELD] == original[archive.FIELD]
               and by_id[identifier] == original_by_id[identifier]
               and by_id[owner] == original_by_id[owner], 'Original archive task and charged owner must stay exact')
    adopted = queue.c.read_ref(plan['adopted_state_snapshot_ref'])
    queue.need(identifier in plan['adopted_completed_task_fingerprints'], 'Original archive must be adopted')
    for key in ('status', 'result_ref', 'job_ref', 'actual_controller_wait', 'owned_process_group_absent'):
        queue.need(adopted['tasks'][identifier].get(key) == snapshot['tasks'][identifier].get(key), 'Archive adoption chain must retain exact original completion evidence')
    if state is not None:
        entry = state['tasks'][identifier]
        original_entry = snapshot['tasks'][identifier]
        for key in ('status', 'result_ref', 'job_ref', 'actual_controller_wait', 'owned_process_group_absent'):
            queue.need(entry.get(key) == original_entry.get(key), 'Adopted archive evidence changed')
    credit = archive.completed_storage_reduction(original, snapshot)
    queue.need(type(credit) is int and credit == proof['completed_reduction_bytes'], 'Exact verified historical lossless reduction required')
    return credit


def load_plan(reference, *, approved=False):
    plan = BASE_LOAD(reference, approved=approved)
    historical_credit(plan)
    return plan


def storage_projection(plan, state, new_cap=0):
    original = BASE_PROJECTION(plan, state, new_cap)
    credit = historical_credit(plan, state)
    queue.need(0 <= credit <= original, 'Storage credit cannot exceed conservative charged storage')
    return original - credit


queue.MODULE = MODULE
queue.SOURCE = SOURCE
queue.load_plan = load_plan
queue.storage_projection = storage_projection

if __name__ == '__main__':
    queue.main()

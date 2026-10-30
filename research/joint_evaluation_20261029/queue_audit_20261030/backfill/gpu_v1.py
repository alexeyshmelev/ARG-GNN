"""TEST009-only GPU3 eligibility after authenticated root and native completion."""
import json
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import successor_v2_gpu as shared
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import boundary_v1 as boundary
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import support_v1 as support

acquire = shared.acquire
release = shared.release


def allowed(plan):
    if shared.root_release(support.accounting_template(plan)) is None:
        return [4, 5]
    spec = r.read(plan['operational_continuation_ref'])
    if not treeloom_gpu_complete(spec):
        return [4, 5]
    registry = r.read(r.ref(spec['native_active_owner_registry_path']))
    boundary.need(registry['schema'] == 'native_argmix_retry_active_owner_v1' and registry['root_confirmed'] is True,
                  'Root-confirmed native GPU owner registry required')
    native = r.read(registry['active_plan_ref'])
    state = json.loads(r.path(registry['state_path']).read_text())
    boundary.need(state.get('operational_plan_ref', state['plan_ref']) == registry['active_plan_ref'], 'Native GPU owner plan changed')
    for task in native['tasks']:
        if task['kind'] != 'gpu':
            continue
        row = state['tasks'][task['id']]
        if row['status'] not in ('succeeded', 'failed', 'blocked'):
            return [4, 5]
        if 'pid' in row:
            boundary.need(row.get('actual_wait') is True and row.get('owned_process_group_absent') is True,
                          'Terminal native GPU worker requires its genuine wait')
            boundary.need(r.read(r.ref(r.path(task['output_directory']) / 'actual_wait.json')) == row,
                          'Native GPU row differs from actual wait')
            if processes.session_members(row['pid']):
                return [4, 5]
        else:
            boundary.need(row['status'] == 'blocked', 'Native terminal task lacks worker evidence')
    return [3, 4, 5]


def allowed_for_task(plan, task, allowed_gpus):
    return allowed_gpus if task['id'].startswith(boundary.PREFIX) else [gpu for gpu in allowed_gpus if gpu != 3]


def treeloom_gpu_complete(spec):
    manifest = r.read(spec['treeloom_manifest_ref'])
    plan = r.read(manifest['plan_ref'])
    for task in plan['tasks']:
        if task['kind'] != 'gpu':
            continue
        receipt = r.path(manifest['records_directory']) / task['id'] / 'actual_wait.json'
        if not receipt.is_file():
            return False
        wait = r.read(r.ref(receipt))
        boundary.need(wait.get('task_id') == task['id'] and wait.get('plan_ref') == manifest['plan_ref'] and
                      wait.get('actual_wait') is True and wait.get('owned_process_group_absent') is True and
                      type(wait.get('actual_returncode')) is int, 'TreeLoom GPU stage needs its exact genuine wait')
        if processes.session_members(wait['pid']):
            return False
        if wait.get('status') == 'succeeded':
            boundary.need(wait['actual_returncode'] == 0 and
                          r.read(r.ref(r.path(task['output_directory']) / 'actual_wait.json')) == wait,
                          'TreeLoom GPU completed stage lacks matching colocated wait')
            r.read(wait['result_ref'])
        else:
            boundary.need(wait.get('status') == 'failed', 'Unknown TreeLoom GPU terminal state')
    return True

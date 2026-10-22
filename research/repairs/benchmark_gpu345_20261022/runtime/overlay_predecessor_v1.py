"""Authenticate original jobs supervised by the separately approved prior overlay.

Only the named read-only metadata function is bound in a fresh recovery process.
Original on-disk helpers and every already-running interpreter remain unchanged.
"""
from arggnn.benchmarking import workflow_predecessor_v5 as original
from . import contracts_recovery_v1 as c

_ORIGINAL_SOURCE=original._source


def need(value,message):
    if not value:raise ValueError(message)


def source(binding):
    if 'source_controller_overlay_ref' not in binding:return _ORIGINAL_SOURCE(binding)
    overlay_ref=binding['source_controller_overlay_ref']
    policy=c.read_json(c.POLICY_PATH)
    need(policy.get('approved_parallel_resource_overlay_ref')==overlay_ref,'Exact prior overlay approval must remain unchanged')
    overlay=c.read_ref(overlay_ref)
    need(overlay.get('schema')=='parallel_controller_resource_overlay_v1'
         and overlay['base_plan_ref']==binding['source_plan_ref'],'Exact original overlay/base plan required')
    plan=c.read_ref(binding['source_plan_ref'])
    tasks=[task for task in plan['tasks'] if task['id']==binding['source_task_id']]
    need(len(tasks)==1 and c.fingerprint(tasks[0])==binding['source_task_fingerprint'],
         'Exact original scientific predecessor descriptor required')
    task=tasks[0]
    need(task['resource']==binding.get('resource','cpu') and task['resource'] in ('cpu','gpu0')
         and task['max_output_bytes']==binding['max_output_bytes'] and task['max_rss_bytes']==binding['max_rss_bytes']
         and c.contained_path(task['output_directory'])==c.contained_path(binding['output_directory']),
         'Original predecessor resource and output bounds required')
    job=c.read_ref(binding['original_job_ref'])
    need(job.get('plan_ref')==binding['source_plan_ref'] and job.get('task')==task
         and job.get('task_fingerprint')==binding['source_task_fingerprint'],'Exact original predecessor job required')
    receipt=c.read_ref(binding['original_supervision_identity_ref']);root=receipt['child_identity']
    need(receipt['job_ref']==binding['original_job_ref']
         and root['pid']==root['pgid']==binding['owned_process_group_id']
         and root['session']==binding['owned_group_leader_session']
         and root['start_ticks']==binding['owned_group_leader_start_ticks']
         and root['ppid']==binding['original_supervisor_pid'],'Preserve exact original durable Popen identity')
    expected_state=c.contained_path(overlay['records_directory'])/'state.json'
    need(c.contained_path(binding['source_state_path'])==expected_state,'Read only the approved overlay controller state')
    frozen=c.read_ref(binding['source_state_snapshot_ref'])
    need(frozen['plan_ref']==binding['source_plan_ref'],'Exact frozen overlay state plan required')
    captured=frozen['tasks'][binding['source_task_id']]
    need(captured.get('status')=='running' and captured.get('job_ref')==binding['original_job_ref']
         and captured.get('pid')==binding['original_supervisor_pid'],'Exact frozen healthy job and supervisor required')
    state=c.read_json(expected_state)
    need(state['plan_ref']==binding['source_plan_ref'],'Original overlay controller plan changed')
    entry=state['tasks'][binding['source_task_id']]
    need(entry.get('job_ref')==binding['original_job_ref'] and entry.get('pid')==binding['original_supervisor_pid'],
         'Original job and supervisor identities changed')
    return task,entry


def install():
    if original._source is source:return
    need(original._source is _ORIGINAL_SOURCE,'Original predecessor metadata function changed before recovery installation')
    original._source=source

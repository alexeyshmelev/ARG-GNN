"""Exact failed-branch sidecar admission; never mutate the original owner or waits."""
import copy
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import successor_v2_support as original
from research.joint_evaluation_20261029.queue import storage_credit_v1 as credit

need = original.need
external_result = original.external_result
external_evidence = original.external_evidence
task_fits_scope = original.task_fits_scope


def manifest(plan):
    return r.read(plan['retry_manifest_ref'])


def verify_original(plan):
    spec = manifest(plan)
    for reference in spec['operational_source_refs']:
        need(r.ref(reference['path']) == reference, 'Pinned operational retry source changed')
    state = json.loads(r.path(spec['original_state_path']).read_text())
    need(state['plan_ref'] == spec['original_plan_ref'] and state['controller_pid'] == spec['original_controller_pid'],
         'Original ownership identity changed; re-audit before continuation')
    for name, expected in spec['preserved_original_rows'].items():
        need(state['tasks'][name] == expected, 'Original terminal branch row changed: ' + name)
    need(r.ref(spec['failed_actual_wait_ref']['path']) == spec['failed_actual_wait_ref'], 'Original failure wait changed')
    need(r.ref(spec['failed_log_ref']['path']) == spec['failed_log_ref'], 'Original failure log changed')
    failed = r.read(spec['failed_actual_wait_ref'])
    need(failed['actual_wait'] and failed['actual_returncode'] == 1 and failed['owned_process_group_absent'],
         'Exact genuine failed native fit is required')
    need(not processes.session_members(failed['pid']), 'Failed native fit session still live')
    for name, row in spec['preserved_original_rows'].items():
        if name == spec['retry_task_id']:
            need(row == failed and row['status'] == 'failed', 'Failed fit must remain failed in original state')
        else:
            need(row['status'] == 'blocked' and row.get('error') == 'Required predecessor failed' and 'pid' not in row,
                 'Only never-launched descendants may be owned by retry sidecar')
    return spec, state


def initial_rows(plan):
    spec, source = verify_original(plan)
    original_plan = r.read(spec['original_plan_ref'])
    originals = {task['id']: task for task in original_plan['tasks']}
    selected = {task['id']: task for task in plan['tasks'] if task['kind'] != 'external'}
    need(set(selected) == set(spec['preserved_original_rows']) and len(selected) == 20,
         'Exact twenty-task native branch required')
    affected = {spec['retry_task_id']}
    while True:
        found = {task['id'] for task in originals.values() if affected.intersection(task['dependencies'])}
        if found <= affected:
            break
        affected |= found
    expected = {name for name in affected if source['tasks'][name]['status'] != 'superseded'}
    need(set(selected) == expected, 'Every active descendant must follow the genuine retry')
    for name, task in selected.items():
        prior = originals[name]
        amended = copy.deepcopy(prior)
        if name == spec['retry_task_id']:
            amended['output_directory'] = spec['retry_output_directory']
            amended['argv'][3] = spec['wrapper_module']
            amended['argv'][amended['argv'].index('--output') + 1] = spec['retry_output_directory']
        elif name.endswith('.argmix_prediction'):
            amended['argv'][amended['argv'].index('--selected-model') + 1] = spec['retry_output_directory'] + '/completed.json'
        need(task == amended, 'Only reviewed CUDA wrapper/output routing may differ: ' + name)
        need(not r.path(task['output_directory']).exists(), 'No replay or existing output: ' + name)
    external_ids = {dep for task in selected.values() for dep in task['dependencies']} - set(selected)
    external_tasks = {task['id']: task for task in plan['tasks'] if task['kind'] == 'external'}
    need(set(external_tasks) == external_ids, 'Only exact original branch dependencies may be external')
    for name, task in external_tasks.items():
        prior = originals[name]
        if prior['kind'] == 'external':
            need(task == prior, 'Existing external producer contract changed')
        else:
            need(task == dict(id=name, kind='external', dependencies=[], sidecar_plan_ref=spec['original_plan_ref'],
                             state_path=spec['original_state_path'], source_task_id=name),
                 'External native producer must retain exact original plan and genuine wait')
    rows = {}
    for task in plan['tasks']:
        name = task['id']
        row = dict(status='pending', task_id=name, operational_retry_manifest_ref=plan['retry_manifest_ref'])
        if name == spec['retry_task_id']:
            row.update(attempt=5, preserved_failed_actual_wait_ref=spec['failed_actual_wait_ref'])
        elif name in selected:
            row['attempt'] = source['tasks'][name].get('attempt', 1)
        else:
            row['external_owner'] = True
        rows[name] = row
    return rows


def metadata_within_bound(plan):
    spec, state = verify_original(plan)
    return original.metadata_within_bound(plan) and r.scope_bytes(r.path(spec['repair_directory'])) <= spec['metadata_max_bytes']


def admission_accounting(plan):
    spec, state = verify_original(plan)
    prior = r.read(spec['original_plan_ref'])
    accounting = credit.commitment(prior, prior['storage_credit_ref'])
    accounting['items']['Additive native CUDA retry metadata'] = spec['metadata_max_bytes']
    accounting['total_bytes'] = sum(accounting['items'].values())
    accounting['admitted'] = accounting['total_bytes'] <= accounting['ceiling_bytes']
    accounting['retry_outputs_within_existing_ARGMix_scope_cap'] = True
    return accounting


def foreign_owner_growth(plan):
    spec, state = verify_original(plan)
    tasks = {task['id']: task for task in r.read(spec['original_plan_ref'])['tasks']}
    growth = 0
    for name, row in state['tasks'].items():
        if row['status'] != 'running':
            continue
        stat = Path('/proc') / str(row['pid']) / 'stat'
        if not stat.exists():
            continue
        fields = stat.read_text().rsplit(')', 1)[1].split()
        need(fields[19] == row['process_start_ticks'], 'Running original worker identity changed')
        growth += max(0, tasks[name]['max_rss_bytes'] - processes.pss(row['pid']))
    return growth

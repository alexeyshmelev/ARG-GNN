"""Authenticate an idle native retry continuation without replaying completed fit."""
import copy
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue.native_argmix_cuda_retry_20261030_v1 import support_v1 as prior

from research.joint_evaluation_20261029.queue_audit_20261030.resources.probe_guard_v1 import ProbeUnavailable

need = prior.need
external_result = prior.external_result
external_evidence = prior.external_evidence
metadata_within_bound = prior.metadata_within_bound
admission_accounting = prior.admission_accounting
task_fits_scope = prior.task_fits_scope


def validate_continuation(plan, *, require_owner_absent):
    spec = r.read(plan['operational_continuation_ref'])
    need(spec['schema'] == 'native_argmix_probe_safe_continuation_v1', 'Exact continuation schema')
    need(spec['original_retry_manifest_ref'] == plan['retry_manifest_ref'], 'Original retry lineage unchanged')
    prior.verify_original(plan)
    for reference in spec['source_refs']:
        need(r.ref(reference['path']) == reference, 'Continuation source changed')
    old_plan = r.read(spec['predecessor_plan_ref'])
    need(plan.get('existing_workers') == [] and old_plan.get('existing_workers') == [],
         'This continuation cannot adopt pre-existing workers')
    launch = r.read(spec['predecessor_launch_ref'])
    need(launch['pid'] == spec['predecessor_controller']['pid'] and
         launch['process_start_ticks'] == spec['predecessor_controller']['start_ticks'] and
         launch['plan_ref'] == spec['predecessor_plan_ref'], 'Pinned predecessor launch identity')
    state = json.loads(r.path(spec['predecessor_state_path']).read_text())
    need(state['plan_ref'] == spec['predecessor_plan_ref'] and state['controller_pid'] == spec['predecessor_controller']['pid'],
         'Original idle retry ownership changed')
    need(state['tasks'] == spec['expected_predecessor_rows'], 'Predecessor tasks changed; re-audit instead of replay')
    need(not any(row['status'] == 'running' for row in state['tasks'].values()), 'No running work may be handed off')
    expected = copy.deepcopy(old_plan)
    expected['records_directory'] = plan['records_directory']
    expected['operational_continuation_ref'] = plan['operational_continuation_ref']
    expected['controller_source_refs'] = old_plan['controller_source_refs'] + spec['source_refs']
    changed = []
    for task in expected['tasks']:
        if task['id'].endswith('.argmix_prediction'):
            need(state['tasks'][task['id']]['status'] == 'pending' and 'pid' not in state['tasks'][task['id']],
                 'Only never-launched predictions receive process naming')
            task['argv'][3] = spec['prediction_wrapper_module']
            changed.append(task['id'])
    need(sorted(changed) == spec['prediction_task_ids'] and len(changed) == 8, 'Exactly eight native predictions')
    need(plan == expected, 'Only explicit continuation metadata and prediction title routing may change')
    tasks = {task['id']:task for task in old_plan['tasks']}
    for name,row in state['tasks'].items():
        task = tasks[name]
        need(row['status'] in ('pending','succeeded'), 'No failed or unknown row may be adopted')
        if row['status'] == 'succeeded':
            r.read(row['result_ref'])
            if task['kind'] == 'external':
                need(prior.external_result(task) == row['result_ref'], 'Authenticated original external result')
                evidence = prior.external_evidence(task)
                need(all(row.get(key) == value for key,value in evidence.items()), 'Authenticated external wait lineage')
            if task['kind'] != 'external':
                wait_path = r.path(task['output_directory']) / task['completion_file']
                receipt = wait_path.parent / 'actual_wait.json'
                need(r.ref(receipt) == spec['completed_wait_refs'][name] and r.read(r.ref(receipt)) == row,
                     'Reuse only exact original successful actual wait')
                need(row['actual_wait'] is True and row['actual_returncode'] == 0 and
                     row['owned_process_group_absent'] is True and row['result_ref'] == r.ref(wait_path),
                     'Actual completed producer and product binding')
                need(not processes.session_members(row['pid']), 'Completed fit session still active')
        elif task['kind'] != 'external':
            need('pid' not in row and not r.path(task['output_directory']).exists(), 'No launched or partial pending work')
    identity = spec['predecessor_controller']
    stat = Path('/proc') / str(identity['pid']) / 'stat'
    if require_owner_absent and stat.exists():
        fields = stat.read_text().rsplit(')',1)[1].split()
        need(fields[19] != identity['start_ticks'] or fields[0] == 'Z', 'Predecessor owner still live')
    return spec, state


def initial_rows(plan):
    # Called only after the unchanged lifetime owner lock has been acquired.
    spec, state = validate_continuation(plan, require_owner_absent=True)
    folder = r.path(plan['records_directory'])
    target = folder / 'predecessor_state_sealed.json'
    need(not target.exists(), 'Never replace a prior handoff snapshot')
    with target.open('x') as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.write('\n')
    r.save(folder / 'adoption_audit.json', dict(schema='native_retry_complete_row_adoption_v1',
        operational_continuation_ref=plan['operational_continuation_ref'], predecessor_state_ref=r.ref(target),
        completed_wait_refs=spec['completed_wait_refs'], completed_fit_retrained=False,
        original_waits_modified=False, exact_original_task_ids_preserved=True))
    return copy.deepcopy(state['tasks'])


def foreign_owner_growth(plan):
    spec, state = prior.verify_original(plan)
    tasks = {task['id']: task for task in r.read(spec['original_plan_ref'])['tasks']}
    growth = 0
    for name, row in state['tasks'].items():
        if row['status'] != 'running':
            continue
        stat = Path('/proc') / str(row['pid']) / 'stat'
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            # A normally exiting process may disappear between two observations.
            continue
        except PermissionError as error:
            raise ProbeUnavailable('Foreign owner process observation denied') from error
        need(fields[19] == row['process_start_ticks'], 'Running original worker identity changed')
        growth += max(0, tasks[name]['max_rss_bytes'] - processes.pss(row['pid']))
    return growth

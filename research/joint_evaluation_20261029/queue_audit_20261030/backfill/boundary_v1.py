"""Read-only, fail-closed authentication of an ARGformer idle boundary."""
import copy
import json
from pathlib import Path
import psutil
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import claims
from research.joint_evaluation_20261029.queue.future_merge import validate_wait

PREFIX = 'paper_american_argformer_american_test_009_'
PREPARE = PREFIX + 'prepare'
OLD_EDGE = 'paper_american_argformer_american_test_007_retire'
IDENTITY_KEYS = ('pid', 'process_start_ticks', 'command', 'physical_gpu',
                 'reservation_bytes', 'started_unix', 'log_path')


def need(value, message):
    if not value:
        raise ValueError(message)


def controller(spec, *, frozen=False):
    old = spec['predecessor_controller']
    root = Path('/proc') / str(old['pid'])
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    need(fields[19] == old['process_start_ticks'], 'Predecessor PID was reused')
    need(fields[0] not in ('Z', 'X'), 'Predecessor already exited; inspect before adoption')
    need((root / 'cwd').resolve() == r.ROOT, 'Predecessor working directory changed')
    need((root / 'exe').resolve() == r.path(old['command'][0]).resolve(), 'Predecessor interpreter changed')
    command = (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    need(command == old['command'], 'Exact predecessor command changed')
    children = [child.pid for child in psutil.Process(old['pid']).children(recursive=True)]
    need(not children, 'Active predecessor descendants; preserve all workers')
    threads = {entry.name: (entry / 'stat').read_text().rsplit(')', 1)[1].split()[0]
               for entry in (root / 'task').iterdir()}
    if frozen:
        need(threads and all(value in ('T', 't') for value in threads.values()), 'Every predecessor thread must be frozen')
    return dict(pid=old['pid'], process_start_ticks=fields[19], state=fields[0],
                command=command, children=children, thread_states=threads)


def expected_plan(spec_ref):
    spec = r.read(spec_ref)
    plan = copy.deepcopy(r.read(spec['original_plan_ref']))
    task = next(task for task in plan['tasks'] if task['id'] == PREPARE)
    need(task['dependencies'] == [OLD_EDGE], 'Exact operational retirement dependency required')
    task['dependencies'] = []
    plan['operational_continuation_ref'] = spec_ref
    plan['controller_source_refs'] += spec['source_refs']
    return plan


def validate_plan(plan):
    spec_ref = plan['operational_continuation_ref']
    spec = r.read(spec_ref)
    need(spec['schema'] == 'argformer_idle_boundary_continuation_v1', 'Exact continuation schema required')
    need(plan == expected_plan(spec_ref), 'Only TEST009 operational retirement edge and explicit provenance may change')
    need(plan['existing_workers'] == [], 'No worker adoption is allowed')
    for source in plan['controller_source_refs']:
        need(r.ref(source['path']) == source, 'Pinned controller source changed: ' + source['path'])
    return spec


def pending_output(plan, task, row):
    need(row['status'] == 'pending' and 'pid' not in row, 'Pending task was already launched')
    output = r.path(task['output_directory'])
    if output.exists():
        need(claims.permit_claimed_output(plan, task, row), 'Pending output already exists')
    directory = r.path(plan['records_directory'])
    need(not list((directory / 'logs').glob(task['id'] + '*.log')), 'Pending task has an existing launch log')
    need(not list((directory / 'jobs').glob(task['id'] + '*.json')), 'Pending task has a prepared launch record')


def terminal_wait(task, row, plan):
    need(row['status'] in ('succeeded', 'failed'), 'Actual worker must have a terminal row')
    need(row.get('actual_wait') is True and row.get('owned_process_group_absent') is True,
         'Genuine consumed wait and absent process session required')
    need(type(row.get('actual_returncode')) is int, 'Actual return code is missing')
    need(type(row.get('pid')) is int and str(row.get('process_start_ticks', '')).isdigit(), 'Exact worker identity required')
    suffix = '' if row.get('attempt', 1) == 1 else '.attempt' + str(row['attempt'])
    job = r.path(plan['records_directory']) / 'jobs' / (task['id'] + suffix + '.json')
    expected = [str(r.path(task['python'])), '-B', '-s', '-m',
                'arggnn.benchmarking.paper_campaign.runtime', '--task', str(job)]
    need(row['command'] == expected, 'Actual wait task command differs')
    need(row.get('reservation_bytes') == task['max_output_bytes'], 'Original full output cap changed')
    need(row.get('physical_gpu') in ((4, 5) if task['kind'] == 'gpu' else (None,)), 'Original worker placement changed')
    receipt = r.path(task['output_directory']) / 'actual_wait.json'
    reference = r.ref(receipt)
    need(r.read(reference) == row, 'Actual wait and task row must match exactly')
    need(not processes.session_members(row['pid']), 'Historical worker session is still live')
    if row['status'] == 'succeeded':
        validate_wait(task['id'], row, task)
    else:
        need('error' in row, 'Failed genuine wait requires its original error')
    return reference


def reconcile(spec, source):
    plan = r.read(spec['original_plan_ref'])
    baseline = r.read(spec['baseline_state_ref'])
    need(source['plan_ref'] == spec['original_plan_ref'], 'Original public plan binding changed')
    need(source['controller_pid'] == spec['predecessor_controller']['pid'], 'Original owner changed')
    tasks = {task['id']: task for task in plan['tasks']}
    need(set(source['tasks']) == set(tasks) == set(baseline['tasks']), 'Exact original task namespace required')
    need(source['tasks'][PREPARE]['status'] == 'pending' and 'pid' not in source['tasks'][PREPARE],
         'TEST009 already started; backfill opportunity closed without replay')
    rows = copy.deepcopy(source['tasks'])
    wait_refs, repaired = {}, []
    for name, row in rows.items():
        prior = baseline['tasks'][name]
        if prior['status'] in ('succeeded', 'failed', 'blocked'):
            need(row == prior, 'Previously terminal history changed: ' + name)
        if prior['status'] == 'running':
            need(all(row.get(key) == prior.get(key) for key in IDENTITY_KEYS), 'Reviewed running worker identity changed: ' + name)
        if row['status'] == 'running':
            wait = r.read(r.ref(r.path(tasks[name]['output_directory']) / 'actual_wait.json'))
            need(all(wait.get(key) == row.get(key) for key in IDENTITY_KEYS), 'Terminal wait does not belong to the stale running row')
            terminal_wait(tasks[name], wait, plan)
            rows[name] = row = wait
            repaired.append(name)
        if 'pid' in row:
            wait_refs[name] = terminal_wait(tasks[name], row, plan)
        elif row['status'] == 'pending':
            pending_output(plan, tasks[name], row)
        else:
            need(row['status'] == 'blocked' and 'error' in row, 'Unknown unlaunched task state')
    return dict(rows=rows, actual_wait_refs=wait_refs, reconciled_stale_running_ids=repaired)


def inspect_boundary(spec):
    before = controller(spec)
    source = json.loads(r.path(spec['original_state_path']).read_text())
    merged = reconcile(spec, source)
    return dict(controller=before, source=source, **merged)

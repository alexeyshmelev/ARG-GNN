"""Read-only owner-aware queue analysis; never launch, signal, score or publish."""
from pathlib import Path
import json
import hashlib

BASE = Path('research/joint_evaluation_20261029')


def read_with_ref(path):
    path = Path(path)
    payload = path.read_bytes()
    return json.loads(payload), dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


def read(path):
    return read_with_ref(path)[0]


def reference(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def merge_owner(nodes, plan, state, owner, owned_only=False):
    for task in plan['tasks']:
        if owned_only and task['kind'] == 'external':
            continue
        name = task['id']
        row = state['tasks'][name]
        nodes[name] = dict(owner=owner, kind=task['kind'], dependencies=list(task['dependencies']),
                           status=row['status'], waiting_reason=row.get('waiting_reason'),
                           error=row.get('error'), pid=row.get('pid'), physical_gpu=row.get('physical_gpu'))


def blockers(nodes, name, trail=()):
    if name in trail:
        return [dict(task=name, status='dependency_cycle', path=list(trail)+( [name] ))]
    if name not in nodes:
        return [dict(task=name, status='external_gate_not_in_owner_map')]
    node = nodes[name]
    if node['status'] == 'succeeded':
        return []
    pending = [parent for parent in node['dependencies'] if parent not in nodes or nodes[parent]['status'] != 'succeeded']
    if node['status'] in ('running', 'failed', 'blocked', 'superseded') or not pending:
        return [dict(task=name, owner=node['owner'], status=node['status'], waiting_reason=node.get('waiting_reason'), pid=node.get('pid'))]
    result = []
    for parent in pending:
        result.extend(blockers(nodes, parent, trail+(name,)))
    unique = {json.dumps(row, sort_keys=True): row for row in result}
    return list(unique.values())


def cycles(nodes):
    visited, active, found = set(), [], []
    def visit(name):
        if name in active:
            found.append(active[active.index(name):]+[name]); return
        if name in visited or name not in nodes:
            return
        active.append(name)
        for parent in nodes[name]['dependencies']:
            visit(parent)
        active.pop(); visited.add(name)
    for name in nodes:
        visit(name)
    return found


def snapshot():
    nodes, sources = {}, []
    specs = [('successor', BASE/'queue/successor_records_v2/plan.json', BASE/'queue/successor_records_v2/state.json', False),
             ('native_ARGformer_sidecar', BASE/'queue/sidecar_plan.json', BASE/'queue/records_v1/state.json', True),
             ('native_ARGMix_preparation_sidecar', BASE/'queue/argmix_plan.json', BASE/'queue/argmix_records_v1/state.json', True)]
    registry, registry_ref = read_with_ref(BASE/'queue/native_argmix_retry_active_owner_v1.json')
    assert registry['root_confirmed'] is True
    specs.append(('native_ARGMix_active_retry', Path(registry['active_plan_ref']['path']), Path(registry['state_path']), True))
    for owner, plan_path, state_path, owned_only in specs:
        plan, plan_ref = read_with_ref(plan_path)
        state, state_ref = read_with_ref(state_path)
        assert state['plan_ref'] == plan_ref
        if owner == 'native_ARGMix_active_retry':
            assert plan_ref == registry['active_plan_ref']
            assert str(Path(plan['records_directory'])/'state.json') == str(state_path)
        merge_owner(nodes, plan, state, owner, owned_only)
        sources.append(dict(owner=owner, plan_ref=plan_ref, state_ref=state_ref,
                            recorded_unix=state.get('updated_unix'), controller_pid=state.get('controller_pid')))
    plan_path = BASE/'argmix/recovery_20261030_queue/task_plan_v1.json'
    plan, plan_ref = read_with_ref(plan_path)
    phase, phase_ref = read_with_ref(BASE/'argmix/recovery_20261030_queue/records_v1/state.json')
    assert phase['plan_ref'] == plan_ref
    wait_refs = {}
    for task in plan['tasks']:
        wait_path = BASE/'argmix/recovery_20261030_queue/records_v1'/task['id']/'actual_wait.json'
        wait = None
        if wait_path.exists():
            wait, wait_refs[task['id']] = read_with_ref(wait_path)
        status = 'pending'
        if wait:
            assert wait['task_id'] == task['id'] and wait['plan_ref'] == plan_ref
            assert wait['actual_wait'] is True and wait['owned_process_group_absent'] is True
            status = 'succeeded' if wait['actual_returncode'] == 0 and wait.get('products_complete') else 'failed'
        elif phase['status'] == 'running' and phase['reason'] == task['id']:
            status = 'running'
        nodes[task['id']] = dict(owner='TreeLoom_ARGMix', kind=task['kind'], dependencies=task['dependencies'],
                                status=status, waiting_reason=phase.get('reason') if status=='pending' else None)
    sources.append(dict(owner='TreeLoom_ARGMix', plan_ref=plan_ref, state_ref=phase_ref,
                        controller_pid=phase.get('controller_pid'), actual_wait_refs=wait_refs))
    sources.append(dict(owner='native_ARGMix_active_registry', registry_ref=registry_ref))
    return nodes, sources


def observability_projection(value):
    """Proposed stable signal for blocker changes, excluding telemetry timestamps."""
    unavailable = (value.get('monitor_health') or {}).get('unavailable', {})
    compact_unavailable = {key: {field: row[field] for field in ('reason', 'error', 'error_type') if field in row}
                           if isinstance(row, dict) else row for key, row in unavailable.items()}
    return dict(waiting={name: row.get('waiting_reason') for name, row in value.get('tasks', {}).items() if row['status']=='pending'},
                monitor_status=(value.get('monitor_health') or {}).get('status'), unavailable=compact_unavailable)

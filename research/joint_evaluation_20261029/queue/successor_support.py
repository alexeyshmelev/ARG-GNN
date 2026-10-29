"""Verified sidecar aliases, selective continuation, and closed writer accounting.

No function here signals a process, rewrites original state, or invents a wait.
"""
import copy
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue.future_merge import validate_wait

MAX_SECONDS = 9223372036854775807
ARGMIX_SCOPE = 'experiments/paper_argmix_1611_rebuild_20261024_v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def verify_original(state, config):
    need(state['controller_pid'] == config['prior_controller']['pid'], 'Original controller changed')
    need(state['plan_ref'] == config['original_identity']['plan_ref'], 'Original plan changed')
    need(state.get('plan_revision', 0) == config['original_identity']['plan_revision'], 'Original revision changed')
    need(state.get('applied_repairs') == config['original_identity']['applied_repairs'], 'Original repairs changed')


def reset_rows(original, tasks, aliases):
    """Only the explicit owned aliases and their claim-blocked descendants reset."""
    need(not any(row['status'] == 'running' for row in original['tasks'].values()), 'Original still owns running work')
    rows = copy.deepcopy(original['tasks'])
    prior = {}
    affected = set(aliases)
    for name in aliases:
        old = rows[name]
        need(old['status'] in ('pending', 'blocked') and 'pid' not in old, 'Cannot replace launched attempt: ' + name)
        if old['status'] == 'blocked':
            need(old.get('error') in ('Output already exists, no overwrite', 'Required predecessor failed'), 'Unrelated alias failure')
        prior[name] = copy.deepcopy(old)
        rows[name] = dict(status='pending', external_owner=True)
    changed = True
    while changed:
        changed = False
        for name, row in list(rows.items()):
            if row['status'] != 'blocked' or row.get('error') != 'Required predecessor failed' or 'pid' in row:
                continue
            dependencies = tasks[name]['dependencies']
            if not affected.intersection(dependencies):
                continue
            if any(rows[d]['status'] in ('failed', 'blocked') for d in dependencies):
                continue
            prior[name] = copy.deepcopy(row)
            rows[name] = dict(status='pending', attempt=row.get('attempt', 1), claim_block_released=True)
            affected.add(name)
            changed = True
    for name, old in original['tasks'].items():
        if old['status'] == 'failed':
            need(rows[name] == old, 'Genuine failed history must be unchanged')
    return rows, prior


def initial_rows(plan):
    handoff = r.read(plan['handoff_ref'])
    source = r.read(handoff['sealed_original_state_ref'])
    config = r.read(plan['handoff_config_ref'])
    verify_original(source, config)
    identity = config['prior_controller']
    stat = Path('/proc') / str(identity['pid']) / 'stat'
    if stat.exists():
        fields = stat.read_text().rsplit(')', 1)[1].split()
        need(fields[19] != identity['start_ticks'] or fields[0] == 'Z', 'Original controller is still alive')
    need(not processes.session_members(identity['pid']), 'Original controller session not drained')
    need(r.ref(config['original_state_path']) == handoff['original_state_ref'], 'Original state changed after sealed handoff')
    tasks = {t['id']: t for t in plan['tasks']}
    need(set(tasks) == set(source['tasks']), 'Exact original task namespace')
    rows, prior = reset_rows(source, tasks, plan['alias_task_ids'])
    for name, row in rows.items():
        task = tasks[name]
        if row['status'] == 'succeeded':
            r.read(row['result_ref'])
        elif row['status'] == 'pending' and task['kind'] != 'external':
            need(not r.path(task['output_directory']).exists(), 'Pending output already exists: ' + name)
        if 'pid' in row:
            need(row.get('actual_wait') is True and row.get('owned_process_group_absent') is True, 'Historical worker lacks consumed wait')
            need(not processes.session_members(row['pid']), 'Historical worker session is live')
    r.save(r.path(plan['records_directory']) / 'adoption_audit.json', dict(
        schema='selective_campaign_adoption_v1', original_state_ref=handoff['sealed_original_state_ref'],
        prior_rows=prior, genuine_failed_rows={n:v for n,v in source['tasks'].items() if v['status']=='failed'},
        sidecar_aliases=plan['alias_task_ids'], historical_waits_rewritten=False))
    return rows


def external_result(task):
    if 'sidecar_plan_ref' not in task:
        if task.get('result_ref'):
            r.read(task['result_ref'])
            return task['result_ref']
        source = json.loads(r.path(task['state_path']).read_text())
        row = source['tasks'][task['source_task_id']]
        if row['status'] == 'succeeded':
            r.read(row['result_ref'])
            return row['result_ref']
        if row['status'] in ('failed', 'blocked', 'unknown'):
            raise RuntimeError('External predecessor ' + row['status'])
        return None
    sideplan = r.read(task['sidecar_plan_ref'])
    source = json.loads(r.path(task['state_path']).read_text())
    need(source['plan_ref'] == task['sidecar_plan_ref'], 'Sidecar plan identity changed')
    selected = next(t for t in sideplan['tasks'] if t['id'] == task['source_task_id'])
    row = source['tasks'][selected['id']]
    if row['status'] == 'succeeded':
        validate_wait(selected['id'], row, selected)
        return row['result_ref']
    if row['status'] in ('failed', 'blocked', 'unknown'):
        raise RuntimeError('Authenticated external predecessor ' + row['status'])
    need(row['status'] in ('pending', 'running'), 'Unknown sidecar task status')
    return None


def external_evidence(task):
    if 'sidecar_plan_ref' not in task:
        return {}
    plan = r.read(task['sidecar_plan_ref'])
    selected = next(t for t in plan['tasks'] if t['id'] == task['source_task_id'])
    state = json.loads(r.path(task['state_path']).read_text())
    row = state['tasks'][selected['id']]
    wait_ref = validate_wait(selected['id'], row, selected)
    return dict(external_actual_wait_ref=wait_ref, external_plan_ref=task['sidecar_plan_ref'],
                external_state_path=task['state_path'], external_source_task_id=selected['id'])


def commitment(plan):
    """Charge every foreign writer and the entire ARGMix scope before admission.

    The old campaign growth envelope is replaced, never subtracted as reclaimed
    files. Historical inventory itself remains fully charged, conservatively.
    """
    config = r.read(plan['handoff_config_ref'])
    ledger = r.read(config['storage_ledger_ref'])
    american = r.read(config['american_allocation_ref'])
    joint_argmix = r.read(config['argmix_joint_allocation_ref'])
    need(joint_argmix['additional_total_bytes'] == 8*2**30 and joint_argmix['owned_RAM_max_bytes'] == 64*2**30, 'New ARGMix reviewed allocation changed')
    need(ledger['writer_closure_confirmed'] is True, 'Writer closure required')
    need(american['additional_total_bytes'] == 64*2**30 and american['owned_RAM_max_bytes'] == 64*2**30, 'American reviewed allocation changed')
    items = {x['name']: x['reserved_bytes'] for x in ledger['base_allowances']
             if x['name'] != 'Entire original campaign post-baseline growth envelope, conservative double count'}
    need(len(items) == 4, 'Exact four noncampaign base allowances')
    items['American full additive allocation'] = american['additional_total_bytes']
    items['New ARGMix full additive allocation'] = joint_argmix['additional_total_bytes']
    items['CNN additional conservative reserve'] = 640*2**20
    items['Entire original ARGMix scope cap'] = 128*2**30
    items['Successor metadata reserve'] = 256*2**20
    items['Original controller records retained'] = r.scope_bytes(r.path(config['original_state_path']).parent)
    native_plan = r.read(config['sidecars'][0]['plan_ref'])
    state = json.loads(r.path(config['sidecars'][0]['state_path']).read_text())
    need(state['plan_ref'] == config['sidecars'][0]['plan_ref'], 'Native sidecar identity changed')
    native_bytes = 0
    released = {}
    for task in native_plan['tasks']:
        row = state['tasks'][task['id']]
        charged = task['max_output_bytes']
        if row['status'] == 'succeeded':
            validate_wait(task['id'], row, task)
            observed = r.scope_bytes(r.path(task['output_directory']))
            need(observed <= task['max_output_bytes'], 'Completed sidecar exceeded cap')
            charged = observed
            released[task['id']] = task['max_output_bytes'] - observed
        native_bytes += charged
    items['ARGformer outstanding caps and verified retained outputs'] = native_bytes
    items['ARGformer metadata cap'] = native_plan['controller_metadata_allowance_bytes']
    items['ARGMix sidecar metadata cap'] = r.read(config['sidecars'][1]['plan_ref'])['controller_metadata_allowance_bytes']
    original = r.read(config['effective_plan_ref'])
    aliases = set(plan['alias_task_ids'])
    items['Retained original outputs outside ARGMix and sidecar ownership'] = sum(
        r.scope_bytes(r.path(t['output_directory'])) for t in original['tasks']
        if t['kind'] != 'external' and t['id'] not in aliases
        and not r.path(t['output_directory']).is_relative_to(r.path(ARGMIX_SCOPE)))
    total = sum(items.values())
    need(r.scope_bytes(r.path(ARGMIX_SCOPE)) <= 128*2**30, 'Existing ARGMix scope exceeds original cap')
    return dict(total_bytes=total, ceiling_bytes=1_200_000_000_000,
                admitted=total <= 1_200_000_000_000, items=items,
                released_unused_successful_stage_reservations=released,
                observed_only_declared_scopes=True)


def task_fits_scope(plan, state, task, active):
    for limit in plan.get('scope_limits', []):
        scope = r.path(limit['path'])
        if not r.path(task['output_directory']).is_relative_to(scope):
            continue
        charged = r.scope_bytes(scope)
        seen = set()
        for candidate in [*active, task]:
            output = r.path(candidate['output_directory'])
            if output.is_relative_to(scope):
                charged += max(0, candidate['max_output_bytes'] - r.scope_bytes(output))
            for extra in candidate.get('additional_mutable_scopes', []):
                path = r.path(extra['path'])
                if path.is_relative_to(scope) and str(path) not in seen:
                    charged += max(0, extra['max_bytes'] - r.scope_bytes(path))
                    seen.add(str(path))
        if charged > limit['max_bytes']:
            return False
    return True


def metadata_within_bound(plan):
    return r.scope_bytes(r.path(plan['records_directory'])) <= plan['controller_metadata_allowance_bytes']

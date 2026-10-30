"""Authenticated producer claims and complete two-parent storage reservations."""
import copy
import json
import os
import stat
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import successor_v2_support as original, storage_credit_v1 as credit
from research.joint_evaluation_20261029.queue_audit_20261030.resources.probe_guard_v1 import ProbeUnavailable

MARKER = 'SIDECAR_CLAIM.json'
SUFFIXES = {'generated', 'relate_inputs', 'native_relate', 'inferred', 'support', 'features', 'treeloom_observation', 'release_new_scratch'}
need = original.need
external_result = original.external_result
external_evidence = original.external_evidence


def manifest(plan):
    return r.read(plan['producer_manifest_ref'])


def original_state(plan):
    spec = manifest(plan)
    state = json.loads(r.path(spec['original_state_path']).read_text())
    need(state['plan_ref'] == spec['original_plan_ref'] and state['controller_pid'] == spec['original_controller']['pid'], 'Original main identity changed')
    return spec, state


def marker_value(plan, task):
    spec = manifest(plan)
    return dict(schema='exact_scope_sidecar_claim_v1', owner=spec['claim_owner'], task_id=task['id'],
                original_plan_ref=spec['original_plan_ref'], output_directory=task['output_directory'])


def validate_contract(plan):
    spec, state = original_state(plan)
    need(spec['schema'] == 'native_relate_parallel_producer_v1', 'Exact producer manifest required')
    for reference in spec['source_refs']:
        need(r.ref(reference['path']) == reference, 'Producer source changed')
    old = r.read(spec['original_plan_ref'])
    old_tasks = {task['id']: task for task in old['tasks']}
    selected = {task['id']: task for task in plan['tasks'] if task['kind'] != 'external'}
    need(set(selected) == set(spec['owned_task_ids']) and len(selected) == 48, 'Exact forty-eight producer stages required')
    need(set(spec['claim_task_ids']) == {name for name in selected if name.endswith('.generated')}, 'Six generated roots are the ownership gates')
    need(len(spec['claim_task_ids']) == 6 and len(spec['parents']) == 6, 'Exact six future parents')
    expected_parents = ['published_recon_iid_'+str(i).zfill(3) for i in range(3)] + ['published_recon_misspec_'+str(i).zfill(3) for i in range(5)]
    need(spec['all_test_parents'] == expected_parents and spec['parents'] == expected_parents[2:], 'Exact future held-out parent order')
    need(set(selected) == {'paper_argmix_'+parent+'.'+suffix for parent in spec['parents'] for suffix in SUFFIXES}, 'Only producer stages for the six held-out parents')
    expected_tasks = []
    for task in old['tasks']:
        if task['id'] not in selected:
            continue
        amended = copy.deepcopy(task)
        if task['id'] in spec['claim_task_ids']:
            change = spec['dependency_changes'][task['id']]
            need(task['dependencies'] == change['before'], 'Original resource dependency changed')
            parent = task['id'].removeprefix('paper_argmix_').rsplit('.', 1)[0]
            index = expected_parents.index(parent)
            expected_dependencies = [] if index == 2 else ['paper_argmix_'+expected_parents[index-2]+'.release_new_scratch']
            need(change['after'] == expected_dependencies, 'Only rolling two-parent resource edges may change')
            amended['dependencies'] = change['after']
            amended['argv'][3] = spec['generation_wrapper_module']
        need(selected[task['id']] == amended, 'Producer task differs outside exact claim and resource dependency amendments')
        expected_tasks.append(amended)
    external_id = spec['original_release_task_id']
    external = dict(id=external_id, kind='external', dependencies=[], state_path=spec['original_state_path'],
                    sidecar_plan_ref=spec['original_plan_ref'], source_task_id=external_id)
    expected_tasks.append(external)
    expected = copy.deepcopy(old)
    expected.update(tasks=expected_tasks, records_directory=spec['records_directory'],
                    controller_metadata_allowance_bytes=spec['metadata_max_bytes'], producer_manifest_ref=plan['producer_manifest_ref'])
    expected['controller_source_refs'] = old['controller_source_refs'] + spec['source_refs']
    need(plan == expected, 'Producer plan changes are explicitly bounded')
    need(plan.get('existing_workers') == [], 'No existing worker may be adopted')
    return spec, state


def validate_preclaim(plan, *, allow_claimed=False):
    spec, state = validate_contract(plan)
    identity = spec['original_controller']
    fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    need(fields[19] == identity['start_ticks'] and fields[0] not in ('Z', 'X'), 'Original controller must remain live')
    need(state['tasks'][spec['original_release_task_id']]['status'] == 'pending', 'Original release gate must remain unfinished')
    guard = state['tasks'][spec['guard_task_id']]
    need(guard['status'] == 'running' and guard['pid'] == spec['guard_pid'] and guard['process_start_ticks'] == spec['guard_start_ticks'], 'Original guard task changed')
    fields = (Path('/proc') / str(guard['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    need(fields[19] == guard['process_start_ticks'] and fields[0] not in ('Z', 'X'), 'Current Relate guard must remain live')
    for task in plan['tasks']:
        if task['kind'] == 'external':
            continue
        need(state['tasks'][task['id']]['status'] == 'pending' and 'pid' not in state['tasks'][task['id']], 'Only never-launched main tasks may be claimed')
        need(state['tasks'][task['id']] == spec['expected_pending_rows'][task['id']], 'Original pending row changed during claim protocol')
        output = r.path(task['output_directory'])
        if allow_claimed and task['id'] in spec['claim_task_ids'] and output.exists():
            permit_existing_output(plan, task, {'status':'pending'})
        else:
            need(not output.exists(), 'Producer output must be absent before root claim')
    return spec, state


def verify_claims(plan, *, empty=False):
    spec = manifest(plan)
    tasks = {task['id']: task for task in plan['tasks']}
    refs = []
    for name in spec['claim_task_ids']:
        task = tasks[name]
        output = r.path(task['output_directory'])
        need(output.is_dir() and not output.is_symlink(), 'Claim root must remain an ordinary directory')
        marker = output / MARKER
        need(json.loads(marker.read_text()) == marker_value(plan, task), 'Exact permanent claim marker required')
        if empty:
            need(sorted(path.name for path in output.iterdir()) == [MARKER], 'Fresh claim may contain only marker')
        refs.append(r.ref(marker))
    return refs


def initial_rows(plan):
    spec, state = validate_preclaim(plan, allow_claimed=True)
    refs = verify_claims(plan, empty=True)
    receipt = json.loads(r.path(spec['claims_receipt_path']).read_text())
    need(receipt['schema'] == 'native_relate_producer_claims_v1' and receipt['root_confirmed'] is True, 'Root-confirmed claims receipt required')
    need(receipt['plan_ref'] == r.ref(spec['plan_path']) and receipt['manifest_ref'] == plan['producer_manifest_ref'] and
         receipt['claims'] == refs, 'Claims receipt must bind exact plan, manifest and six markers')
    for task in plan['tasks']:
        if task['kind'] == 'external':
            continue
        old = state['tasks'][task['id']]
        need(old['status'] in ('pending', 'blocked') and 'pid' not in old, 'Original task was already launched')
        if task['id'] not in spec['claim_task_ids']:
            need(not r.path(task['output_directory']).exists(), 'Unclaimed descendant output already exists')
    r.save(r.path(plan['records_directory']) / 'adoption_audit.json', dict(schema='native_relate_producer_claim_adoption_v1',
        claims=refs, original_plan_ref=spec['original_plan_ref'], owned_task_ids=spec['owned_task_ids'],
        original_controller_untouched=True, old_rows_rewritten=False, actual_waits_fabricated=False))
    return {task['id']: dict(status='pending', task_id=task['id'], producer_manifest_ref=plan['producer_manifest_ref']) for task in plan['tasks']}


def state_metadata(plan, plan_path):
    return dict(producer_manifest_ref=plan['producer_manifest_ref'], original_plan_ref=manifest(plan)['original_plan_ref'])


def permit_existing_output(plan, task, row):
    spec = manifest(plan)
    if task['id'] not in spec['claim_task_ids'] or row['status'] != 'pending':
        return False
    output = r.path(task['output_directory'])
    need(output.is_dir() and not output.is_symlink(), 'Ordinary claimed generation root required')
    need(sorted(path.name for path in output.iterdir()) == [MARKER], 'Never overwrite generated data')
    need(json.loads((output/MARKER).read_text()) == marker_value(plan, task), 'Exact generation gate required')
    return True


def metadata_within_bound(plan):
    spec = manifest(plan)
    return r.scope_bytes(r.path(spec['producer_directory'])) <= spec['metadata_max_bytes']


def admission_accounting(plan):
    spec, state = original_state(plan)
    old = r.read(spec['original_plan_ref'])
    accounting = credit.commitment(old, old['storage_credit_ref'])
    accounting['items']['Native retry metadata conservative reserve'] = 16*2**20
    accounting['items']['Parallel Relate producer metadata reserve'] = spec['metadata_max_bytes']
    accounting['items']['Queue audit and launcher metadata reserve'] = 16*2**20
    accounting['items']['ARGformer boundary metadata reserve'] = 16*2**20
    accounting['total_bytes'] = sum(accounting['items'].values())
    accounting['admitted'] = accounting['total_bytes'] <= accounting['ceiling_bytes']
    return accounting


def foreign_owner_growth(plan):
    spec, state = original_state(plan)
    tasks = {task['id']: task for task in r.read(spec['original_plan_ref'])['tasks']}
    growth = 0
    for name, row in state['tasks'].items():
        if row['status'] != 'running':
            continue
        try:
            fields = (Path('/proc')/str(row['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise ProbeUnavailable('Original process observation denied') from error
        need(fields[19] == row['process_start_ticks'], 'Original worker identity changed')
        growth += max(0, tasks[name]['max_rss_bytes'] - processes.pss(row['pid']))
    return growth


def scope_inventory(scope):
    """One bounded declared-family inventory keeps total and subscopes consistent."""
    sizes = {}
    def fail(error):
        raise error
    for directory, directories, filenames in os.walk(scope, followlinks=False, onerror=fail):
        for name in directories:
            need(not (Path(directory)/name).is_symlink(), 'No symlink inside declared family scope')
        for name in filenames:
            path = Path(directory)/name
            try:
                observed = path.lstat()
            except FileNotFoundError:
                continue
            need(not stat.S_ISLNK(observed.st_mode), 'No symlink inside declared family scope')
            sizes[path] = observed.st_size
    return sizes


def completion_reservation(plan, state, candidate, *, sizes=None):
    spec, old_state = original_state(plan)
    old_plan = r.read(spec['original_plan_ref'])
    original_tasks = {task['id']: task for task in old_plan['tasks']}
    scope = r.path(spec['scope_path'])
    sizes = scope_inventory(scope) if sizes is None else sizes
    def actual(path):
        target = r.path(path)
        return sum(size for file, size in sizes.items() if file == target or file.is_relative_to(target))
    rows = dict(old_state['tasks'])
    rows.update({name: row for name, row in state['tasks'].items() if name in spec['owned_task_ids']})
    all_parents = spec['all_test_parents']
    opened = []
    candidate_parent = candidate['id'].removeprefix('paper_argmix_').rsplit('.', 1)[0]
    for parent in all_parents:
        prefix = 'paper_argmix_'+parent+'.'
        started = rows[prefix+'generated']['status'] in ('running', 'succeeded')
        released = rows[prefix+'release_new_scratch']['status'] == 'succeeded'
        if (started and not released) or (parent == candidate_parent and candidate['id'].endswith('.generated')):
            opened.append(parent)
    commitments = {}
    for parent in opened:
        prefix = 'paper_argmix_'+parent+'.'
        for suffix in SUFFIXES:
            task = original_tasks[prefix+suffix]
            row = rows[task['id']]
            path = r.path(task['output_directory'])
            desired = actual(path) if row['status'] == 'succeeded' else max(actual(path), task['max_output_bytes'])
            commitments[path] = max(commitments.get(path, 0), desired)
            for auxiliary in task.get('additional_mutable_scopes', []):
                path = r.path(auxiliary['path'])
                commitments[path] = max(commitments.get(path, 0), auxiliary['max_bytes'], actual(path))
    total = sum(sizes.values()) + sum(max(0, desired-actual(path)) for path, desired in commitments.items())
    # Only authenticated sealed comparator outputs release their unused caps.
    retry = r.read(spec['native_retry_plan_ref'])
    comparator_extra = 0
    for task in retry['tasks']:
        if task['kind'] == 'external':
            continue
        output = r.path(task['output_directory'])
        sealed = spec['sealed_comparator_outputs'].get(task['id'])
        if sealed is not None:
            wait = r.read(sealed['actual_wait_ref'])
            need(wait['status'] == 'succeeded' and wait['actual_wait'] is True and wait['actual_returncode'] == 0 and
                 wait['owned_process_group_absent'] is True and wait['result_ref'] == sealed['result_ref'], 'Sealed comparator evidence changed')
            r.read(sealed['result_ref'])
            continue
        if output.is_relative_to(scope):
            comparator_extra += max(0, task['max_output_bytes']-actual(output))
    total += comparator_extra
    allowed = len(opened) <= 2 and total + spec['completion_guard_bytes'] <= spec['scope_cap_bytes']
    return dict(admitted=allowed, open_parents=opened, observed_bytes=sum(sizes.values()),
                complete_remaining_commitment_bytes=total, comparator_future_reserve_bytes=comparator_extra,
                guard_bytes=spec['completion_guard_bytes'], scope_cap_bytes=spec['scope_cap_bytes'])


def task_fits_scope(plan, state, task, active):
    verify_claims(plan)
    proof = completion_reservation(plan, state, task)
    r.save(r.path(plan['records_directory'])/'completion_reservation.json', proof)
    return proof['admitted'] and original.task_fits_scope(plan, state, task, active)

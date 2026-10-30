"""Read-only authentication of the exact six-parent parallel producer owner."""
import copy
import hashlib
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue.future_merge import validate_wait

BASE = 'research/joint_evaluation_20261029'
REGISTRY = BASE + '/queue/native_relate_producer_active_owner_v1.json'
PARENTS = ['published_recon_iid_002'] + ['published_recon_misspec_' + str(i).zfill(3) for i in range(5)]
SUFFIXES = ('generated', 'relate_inputs', 'native_relate', 'inferred', 'support', 'features', 'treeloom_observation', 'release_new_scratch')
OWNED_IDS = frozenset('paper_argmix_' + parent + '.' + suffix for parent in PARENTS for suffix in SUFFIXES)
NATIVE_ALIASES = frozenset(name for name in OWNED_IDS if name.endswith(('.support', '.features')))
ORIGINAL_RELEASE = 'paper_argmix_published_recon_iid_001.release_new_scratch'


def need(value, message):
    if not value:
        raise ValueError(message)


def read_live(path):
    path = r.path(path)
    payload = path.read_bytes()
    return json.loads(payload), dict(path=str(path.relative_to(r.ROOT)), sha256=hashlib.sha256(payload).hexdigest())


def validate_contract(registry, manifest, plan, original, state):
    """Pure contract validation; no owner can expand the registered namespace."""
    need(registry.get('schema') == 'native_relate_producer_active_owner_v1' and registry.get('root_confirmed') is True,
         'Explicit root-confirmed parallel producer registry required')
    need(manifest.get('schema') == 'native_relate_parallel_producer_v1', 'Exact producer manifest schema')
    need(set(registry['owned_task_ids']) == OWNED_IDS and len(registry['owned_task_ids']) == 48,
         'Registry must own exactly the registered forty-eight producer tasks')
    need(manifest['owned_task_ids'] == sorted(OWNED_IDS) and manifest['parents'] == PARENTS,
         'Manifest producer namespace differs')
    need(registry['original_plan_ref'] == manifest['original_plan_ref'], 'Original producer plan differs')
    need(state['plan_ref'] == registry['producer_plan_ref'], 'Producer state has another plan')
    need(r.path(registry['state_path']) == r.path(plan['records_directory']) / 'state.json', 'Producer state path differs')
    need(plan['producer_manifest_ref'] == registry['producer_manifest_ref'], 'Producer plan manifest differs')
    originals = {task['id']: task for task in original['tasks']}
    tasks = {task['id']: task for task in plan['tasks']}
    need(len(tasks) == len(plan['tasks']) and set(tasks) == OWNED_IDS | {ORIGINAL_RELEASE}, 'Exact producer plan namespace')
    need(set(state['tasks']) == set(tasks), 'Bound producer state must include every planned row')
    generated = sorted(name for name in OWNED_IDS if name.endswith('.generated'))
    need(manifest['claim_task_ids'] == generated and set(manifest['dependency_changes']) == set(generated), 'Exact six generated roots')
    predecessor_parents = [None, 'published_recon_iid_001', *PARENTS[:4]]
    for parent, predecessor in zip(PARENTS, predecessor_parents):
        name = 'paper_argmix_' + parent + '.generated'
        expected_dependencies = [] if predecessor is None else ['paper_argmix_' + predecessor + '.release_new_scratch']
        change = manifest['dependency_changes'][name]
        need(change == dict(before=originals[name]['dependencies'], after=expected_dependencies), 'Only the reviewed two-parent dependency window is allowed')
    for name in OWNED_IDS:
        expected = copy.deepcopy(originals[name])
        if name.endswith('.generated'):
            expected['dependencies'] = manifest['dependency_changes'][name]['after']
            expected['argv'][3] = manifest['generation_wrapper_module']
        need(tasks[name] == expected, 'Producer scientific command, output or resource contract changed: ' + name)
        need(tasks[name]['kind'] == 'cpu', 'Producer may own only CPU tasks')
    alias = dict(id=ORIGINAL_RELEASE, kind='external', dependencies=[], sidecar_plan_ref=registry['original_plan_ref'],
                 state_path=manifest['original_state_path'], source_task_id=ORIGINAL_RELEASE)
    need(tasks[ORIGINAL_RELEASE] == alias, 'Only the original IID001 release may be external')
    return tasks



def validate_activation(registry, state, activation, claims):
    need(activation.get('schema') == 'native_relate_producer_root_activation_v1' and activation.get('root_confirmed') is True,
         'Explicit root-confirmed producer launch activation required')
    need(activation['plan_ref'] == registry['producer_plan_ref'] and
         activation['manifest_ref'] == registry['producer_manifest_ref'], 'Activation producer contract differs')
    need(type(activation['pid']) is int and activation['pid'] > 1 and activation['pid'] == state['controller_pid'],
         'Activation must bind the actual producer controller PID')
    need(isinstance(activation['process_start_ticks'], str) and activation['process_start_ticks'].isdigit(),
         'Activation requires exact process start identity')
    command = activation['command']
    need(isinstance(command, list) and '--plan' in command and
         r.path(command[command.index('--plan') + 1]) == r.path(registry['producer_plan_ref']['path']),
         'Activation command must execute the registered plan')
    need('-m' in command and command[command.index('-m') + 1] ==
         'research.joint_evaluation_20261029.queue_audit_20261030.critical_path.producer_v1.scheduler_v1',
         'Activation must invoke the reviewed producer scheduler')
    need(claims.get('schema') == 'native_relate_producer_claims_v1' and claims.get('root_confirmed') is True,
         'Root-confirmed producer claims receipt required')
    need(claims['plan_ref'] == registry['producer_plan_ref'] and claims['manifest_ref'] == registry['producer_manifest_ref'],
         'Claims receipt and activation must bind the same producer')
    need(len(claims['claims']) == 6 and len({item['path'] for item in claims['claims']}) == 6,
         'Exactly six distinct generation claims required')


def load_active_owner(registry_path=REGISTRY, *, required=False, expected_plan_ref=None, expected_manifest_ref=None):
    path = r.path(registry_path)
    if not path.exists():
        need(not required, 'Parallel producer registry is not active')
        return None
    registry, registry_ref = read_live(path)
    if expected_plan_ref is not None:
        need(registry['producer_plan_ref'] == expected_plan_ref, 'Producer activation changed from reviewed consumer plan')
    if expected_manifest_ref is not None:
        need(registry['producer_manifest_ref'] == expected_manifest_ref, 'Producer manifest changed from reviewed consumer plan')
    manifest = r.read(registry['producer_manifest_ref'])
    plan = r.read(registry['producer_plan_ref'])
    original = r.read(registry['original_plan_ref'])
    state, state_ref = read_live(registry['state_path'])
    tasks = validate_contract(registry, manifest, plan, original, state)
    for source in manifest['source_refs']:
        need(r.ref(source['path']) == source, 'Producer source changed: ' + source['path'])
    activation = r.read(registry['activation_ref'])
    claims = r.read(activation['claims_ref'])
    validate_activation(registry, state, activation, claims)
    for name, claim_ref in zip(manifest['claim_task_ids'], claims['claims']):
        task = tasks[name]
        need(r.path(claim_ref['path']) == r.path(task['output_directory']) / 'SIDECAR_CLAIM.json', 'Claim marker path differs')
        claim = r.read(claim_ref)
        need(claim == dict(schema='exact_scope_sidecar_claim_v1', owner=manifest['claim_owner'], task_id=name,
             original_plan_ref=registry['original_plan_ref'], output_directory=task['output_directory']), 'Claim marker identity differs')
    return dict(registry=registry, registry_ref=registry_ref, manifest=manifest, plan=plan, original=original,
                state=state, state_ref=state_ref, tasks=tasks,
                owned_rows={name: state['tasks'][name] for name in sorted(OWNED_IDS)})


def validated_wait(owner, name):
    need(name in OWNED_IDS, 'Wait is outside exact registered producer ownership')
    row, task = owner['state']['tasks'][name], owner['tasks'][name]
    need(row['status'] == 'succeeded' and row.get('task_id') == name, 'Successful producer task identity required')
    need(row.get('command') == task['argv'], 'Producer wait command differs from reviewed task')
    need(row.get('products_complete', True) is True, 'Producer products are incomplete')
    if 'plan_ref' in row:
        need(row['plan_ref'] == owner['registry']['producer_plan_ref'], 'Producer wait plan differs')
    wait_ref = validate_wait(name, row, task)
    return dict(result_ref=row['result_ref'], external_actual_wait_ref=wait_ref,
                external_plan_ref=owner['registry']['producer_plan_ref'],
                external_state_path=owner['registry']['state_path'], external_source_task_id=name,
                external_producer_registry_ref=owner['registry_ref'])


def resolve_alias(task, owner):
    name = task.get('source_task_id')
    need(task.get('kind') == 'external' and name in NATIVE_ALIASES and task['id'] == name, 'Exact native support/features alias required')
    need(task['sidecar_plan_ref'] == owner['registry']['original_plan_ref'] and
         r.path(task['state_path']) == r.path(owner['manifest']['original_state_path']), 'Consumer original owner binding differs')
    row = owner['state']['tasks'][name]
    if row['status'] == 'succeeded':
        return validated_wait(owner, name)
    if row['status'] in ('failed', 'blocked', 'unknown'):
        raise RuntimeError('Authenticated parallel producer ' + row['status'] + ': ' + name)
    need(row['status'] in ('pending', 'running'), 'Unknown parallel producer status')
    return None

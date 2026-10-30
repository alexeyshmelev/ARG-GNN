"""Idle native continuation with exact registered producer alias routing."""
import copy
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue.native_argmix_cuda_retry_20261030_v1.probe_safe_continuation_v1 import support_v1 as prior
from research.joint_evaluation_20261029.queue_audit_20261030.common import probe_scheduler_v1 as engine
from research.joint_evaluation_20261029.queue.future_merge import validate_wait
from research.joint_evaluation_20261029.queue_audit_20261030.resources.probe_guard_v1 import ProbeUnavailable
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import producer_owner_v1 as producer

need = producer.need
task_fits_scope = prior.task_fits_scope
ACTIVE_SPEC = None


def active_owner(spec):
    try:
        return producer.load_active_owner(spec['producer_registry_path'], required=True,
            expected_plan_ref=spec['producer_plan_ref'], expected_manifest_ref=spec['producer_manifest_ref'])
    except OSError as error:
        raise ProbeUnavailable('Parallel producer metadata temporarily unavailable') from error



def metadata_within_bound(plan):
    spec = r.read(plan['operational_continuation_ref'])
    return (prior.metadata_within_bound(plan) and
            r.scope_bytes(r.path(spec['producer_metadata_scope'])) <= spec['producer_metadata_max_bytes'] and
            r.scope_bytes(r.path(spec['audit_metadata_scope'])) <= spec['audit_metadata_max_bytes'])


def admission_accounting(plan):
    spec = r.read(plan['operational_continuation_ref'])
    result = prior.admission_accounting(plan)
    result['items']['Parallel native producer metadata'] = spec['producer_metadata_max_bytes']
    result['items']['Queue audit metadata'] = spec['audit_metadata_max_bytes']
    result['items']['ARGformer boundary continuation metadata'] = spec['argformer_metadata_max_bytes']
    result['total_bytes'] = sum(result['items'].values())
    result['admitted'] = result['total_bytes'] <= result['ceiling_bytes']
    return result


def foreign_owner_growth(plan):
    spec = r.read(plan['operational_continuation_ref'])
    owner = active_owner(spec)
    growth = prior.foreign_owner_growth(plan)
    for name, row in owner['owned_rows'].items():
        if row['status'] != 'running':
            continue
        if row.get('process_start_ticks') is None:
            raise ProbeUnavailable('Parallel producer start identity is pending observation')
        stat = Path('/proc') / str(row['pid']) / 'stat'
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ProbeUnavailable('Parallel producer process observation unavailable') from error
        need(fields[19] == row['process_start_ticks'], 'Running parallel producer PID identity changed')
        growth += max(0, owner['tasks'][name]['max_rss_bytes'] - processes.pss(row['pid']))
    return growth


def external_result(task):
    if task.get('source_task_id') not in producer.NATIVE_ALIASES:
        return prior.external_result(task)
    need(ACTIVE_SPEC is not None, 'Bridge must authenticate its continuation before resolving aliases')
    evidence = producer.resolve_alias(task, active_owner(ACTIVE_SPEC))
    return evidence['result_ref'] if evidence else None


def external_evidence(task):
    if task.get('source_task_id') not in producer.NATIVE_ALIASES:
        return prior.external_evidence(task)
    need(ACTIVE_SPEC is not None, 'Bridge continuation is not bound')
    evidence = producer.resolve_alias(task, active_owner(ACTIVE_SPEC))
    need(evidence is not None, 'Producer successful wait disappeared')
    return {key: value for key, value in evidence.items() if key != 'result_ref'}


def validate_continuation(plan, *, require_owner_absent, require_producer_active=True):
    spec = r.read(plan['operational_continuation_ref'])
    need(spec['schema'] == 'native_argmix_producer_bridge_continuation_v1', 'Exact native bridge continuation required')
    need(spec['original_retry_manifest_ref'] == plan['retry_manifest_ref'], 'Original retry lineage unchanged')
    prior.prior.verify_original(plan)
    registry = r.read(spec['predecessor_registry_ref'])
    need(registry['root_confirmed'] is True and registry['active_plan_ref'] == spec['predecessor_plan_ref'] and
         registry['state_path'] == spec['predecessor_state_path'], 'Predecessor must remain the root-activated native owner')
    need(spec['producer_metadata_max_bytes'] == spec['audit_metadata_max_bytes'] == spec['argformer_metadata_max_bytes'] == 16*2**20, 'Exact metadata reservation required')
    for source in spec['source_refs']:
        need(r.ref(source['path']) == source, 'Bridge source changed: ' + source['path'])
    old_plan = r.read(spec['predecessor_plan_ref'])
    need(registry['active_manifest_ref'] == spec['predecessor_manifest_ref'] == old_plan['operational_continuation_ref'],
         'Predecessor operational continuation binding changed')
    state, state_ref = producer.read_live(spec['predecessor_state_path'])
    need(state['plan_ref'] == spec['predecessor_plan_ref'] and state['controller_pid'] == spec['predecessor_controller']['pid'],
         'Predecessor native ownership changed')
    need(state['tasks'] == spec['expected_predecessor_rows'], 'Native task rows changed; rebuild and review the idle continuation')
    need(not any(row['status'] == 'running' for row in state['tasks'].values()), 'No active native task may transfer')
    need(old_plan.get('existing_workers') == [] and plan.get('existing_workers') == [], 'Bridge cannot adopt workers')
    expected = copy.deepcopy(old_plan)
    expected['records_directory'] = plan['records_directory']
    expected['operational_continuation_ref'] = plan['operational_continuation_ref']
    expected['controller_source_refs'] = old_plan['controller_source_refs'] + spec['source_refs']
    need(plan == expected, 'Only explicit continuation metadata may change; every native task stays identical')
    tasks = {task['id']: task for task in old_plan['tasks']}
    owned = {name for name, task in tasks.items() if task['kind'] != 'external'}
    need(owned == set(prior.prior.manifest(plan)['preserved_original_rows']) and len(owned) == 20,
         'Exact twenty native consumers retained')
    aliases = {name for name, task in tasks.items() if task['kind'] == 'external' and name in producer.NATIVE_ALIASES}
    need(aliases == producer.NATIVE_ALIASES and spec['bridged_alias_ids'] == sorted(aliases), 'Exactly twelve support/features aliases required')
    for name, row in state['tasks'].items():
        task = tasks[name]
        need(row['status'] in ('pending', 'succeeded'), 'No failed or unknown native row may transfer')
        if row['status'] == 'succeeded':
            if task['kind'] == 'external':
                need(name not in aliases, 'Transferred producer alias already completed under another owner')
                need(prior.external_result(task) == row['result_ref'], 'Completed original external result differs')
                evidence = prior.external_evidence(task)
                need(all(row.get(key) == value for key, value in evidence.items()), 'External evidence lineage changed')
            else:
                wait_ref = validate_wait(name, row, task)
                need(wait_ref == spec['completed_wait_refs'][name], 'Completed genuine wait changed')
                need(row.get('task_id') == name and row.get('command') == engine.resolve(task['argv'], state),
                     'Completed native task command changed')
        elif task['kind'] != 'external':
            need('pid' not in row and not r.path(task['output_directory']).exists(), 'Pending consumer must never have launched')
    identity = spec['predecessor_controller']
    stat = Path('/proc') / str(identity['pid']) / 'stat'
    if stat.exists():
        fields = stat.read_text().rsplit(')', 1)[1].split()
        if require_owner_absent:
            need(fields[19] != identity['start_ticks'] or fields[0] == 'Z', 'Predecessor native owner is still live')
        else:
            need(fields[19] == identity['start_ticks'], 'Predecessor PID identity changed')
    if require_producer_active:
        active_owner(spec)
    else:
        producer_plan = r.read(spec['producer_plan_ref'])
        producer_manifest = r.read(spec['producer_manifest_ref'])
        need(producer_plan['producer_manifest_ref'] == spec['producer_manifest_ref'], 'Candidate producer binding differs')
        need(producer_manifest['owned_task_ids'] == sorted(producer.OWNED_IDS), 'Candidate exact producer namespace')
    return spec, state, state_ref


def initial_rows(plan):
    global ACTIVE_SPEC
    # The unchanged native lifetime lock is held by the reused scheduler here.
    spec, state, state_ref = validate_continuation(plan, require_owner_absent=True)
    folder = r.path(plan['records_directory'])
    target = folder / 'predecessor_state_sealed.json'
    with target.open('x') as stream:
        json.dump(state, stream, indent=2, sort_keys=True); stream.write('\n')
    r.save(folder / 'adoption_audit.json', dict(schema='native_producer_bridge_adoption_v1',
        operational_continuation_ref=plan['operational_continuation_ref'], predecessor_observed_state_ref=state_ref,
        predecessor_state_sealed_ref=r.ref(target), completed_wait_refs=spec['completed_wait_refs'],
        bridged_alias_ids=spec['bridged_alias_ids'], completed_fit_retrained=False,
        original_waits_modified=False, exact_original_task_ids_preserved=True))
    ACTIVE_SPEC = spec
    return copy.deepcopy(state['tasks'])


def state_metadata(plan, plan_path):
    return dict(operational_retry_manifest_ref=plan['retry_manifest_ref'],
                operational_continuation_ref=plan['operational_continuation_ref'])


def permit_existing_output(plan, task, row):
    return False

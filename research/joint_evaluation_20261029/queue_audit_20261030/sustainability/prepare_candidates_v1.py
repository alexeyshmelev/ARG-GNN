"""Prepare immutable native/reporting candidates; never launch or signal."""
import copy
import json
from pathlib import Path
from datetime import datetime, timezone
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import producer_owner_v1 as producer

BASE = Path('research/joint_evaluation_20261029')
HERE = BASE / 'queue_audit_20261030/sustainability'
BRIDGE = BASE / 'queue/native_argmix_cuda_retry_20261030_v1/producer_bridge_continuation_v3'
PRODUCER = BASE / 'queue_audit_20261030/critical_path/producer_v1'


def immutable(path, value):
    path = r.path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        producer.need(json.loads(path.read_text()) == value, 'Existing candidate differs; preserve it and use a new version')
        return r.ref(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True);stream.write('\n')
    return r.ref(path)


def main():
    native_registry_path = BASE / 'queue/native_argmix_retry_active_owner_v1.json'
    registry_ref = r.ref(native_registry_path)
    registry = r.read(registry_ref)
    old_plan = r.read(registry['active_plan_ref'])
    state, state_ref = producer.read_live(registry['state_path'])
    producer.need(state['plan_ref'] == registry['active_plan_ref'] and not any(row['status']=='running' for row in state['tasks'].values()),
                  'Native owner must be idle while preparing the candidate')
    identity = (Path('/proc')/str(state['controller_pid'])/'stat').read_text().rsplit(')',1)[1].split()
    producer.need(identity[0] not in ('Z','X'), 'Current native controller must have its exact live identity')
    producer_plan_ref, producer_manifest_ref = r.ref(PRODUCER/'plan_v1.json'), r.ref(PRODUCER/'manifest_v1.json')
    producer_plan, producer_manifest = r.read(producer_plan_ref), r.read(producer_manifest_ref)
    for source in producer_manifest['source_refs']:
        producer.need(r.ref(source['path']) == source, 'Producer sources must be sealed before preparing consumer')
    native_sources = [HERE/name for name in ('producer_owner_v1.py','native_support_v1.py','native_scheduler_v1.py')]
    native_sources.append(BASE/'queue_audit_20261030/common/probe_scheduler_v1.py')
    source_refs = [r.ref(path) for path in native_sources]
    tasks = {task['id']:task for task in old_plan['tasks']}
    waits = {name:r.ref(r.path(tasks[name]['output_directory'])/'actual_wait.json') for name,row in state['tasks'].items()
             if row['status']=='succeeded' and tasks[name]['kind']!='external'}
    predecessor_registry_snapshot_ref = immutable(BRIDGE/'predecessor_active_registry.json', registry)
    spec = dict(schema='native_argmix_producer_bridge_continuation_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        original_retry_manifest_ref=old_plan['retry_manifest_ref'], predecessor_registry_ref=predecessor_registry_snapshot_ref,
        predecessor_plan_ref=registry['active_plan_ref'], predecessor_manifest_ref=registry['active_manifest_ref'],
        predecessor_state_path=registry['state_path'],preparation_state_ref=state_ref,
        predecessor_controller=dict(pid=state['controller_pid'],start_ticks=identity[19]),
        expected_predecessor_rows=state['tasks'],completed_wait_refs=waits,
        producer_registry_path=producer.REGISTRY, producer_plan_ref=producer_plan_ref, producer_manifest_ref=producer_manifest_ref,
        bridged_alias_ids=sorted(producer.NATIVE_ALIASES), source_refs=source_refs,
        producer_metadata_scope=producer_manifest['producer_directory'],producer_metadata_max_bytes=16*2**20,
        audit_metadata_scope=str(BASE/'queue_audit_20261030'),audit_metadata_max_bytes=16*2**20,argformer_metadata_max_bytes=16*2**20,
        unchanged_lifetime_lock_path=str(BASE/'queue/native_argmix_cuda_retry_20261030_v1/sidecar_owner.lock'),
        exact_native_task_contracts_unchanged=True,original_waits_modified=False,completed_fit_retrained=False)
    spec_ref = immutable(BRIDGE/'manifest_v1.json',spec)
    plan = copy.deepcopy(old_plan)
    plan['records_directory'] = str(BRIDGE/'records_v1')
    plan['operational_continuation_ref'] = spec_ref
    plan['controller_source_refs'] += source_refs
    plan_ref = immutable(BRIDGE/'plan_v1.json',plan)
    report_sources = [r.ref(HERE/name) for name in ('producer_owner_v1.py','effective_state_v2.py','publish_results_v2.py','finish_coordinator_v2.py')]
    old_finish_ref=r.ref(BASE/'root/reporting_20261030_v1/finish_plan_candidate_v1.json')
    finish=copy.deepcopy(r.read(old_finish_ref))
    finish['publisher_argv'][4]='research.joint_evaluation_20261029.queue_audit_20261030.sustainability.publish_results_v2'
    finish['source_refs']+=report_sources
    finish['reporting_continuation_of_plan_ref']=old_finish_ref
    finish_ref=immutable(HERE/'finish_plan_candidate_v3.json',finish)
    old_final_ref=r.ref(BASE/'root/reporting_20261030_v1/final_review_plan_candidate_v1.json')
    final=copy.deepcopy(r.read(old_final_ref))
    final['publisher_argv']=finish['publisher_argv']
    final['finish_plan_ref']=finish_ref
    final['source_refs']+=report_sources
    final['reporting_continuation_of_plan_ref']=old_final_ref
    final_ref=immutable(HERE/'final_review_plan_candidate_v3.json',final)
    print(json.dumps(dict(native_plan_ref=plan_ref,native_manifest_ref=spec_ref,finish_plan_ref=finish_ref,final_review_plan_ref=final_ref),sort_keys=True))


if __name__=='__main__':
    main()

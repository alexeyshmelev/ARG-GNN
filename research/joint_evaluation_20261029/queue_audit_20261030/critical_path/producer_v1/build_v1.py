"""Prepare exact producer contracts only; never claim directories or launch jobs."""
import copy
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r


def main():
    folder = Path(__file__).parent
    original_plan_ref = r.ref('research/joint_evaluation_20261029/queue/successor_records_v2/plan.json')
    old = r.read(original_plan_ref)
    state_path = 'research/joint_evaluation_20261029/queue/successor_records_v2/state.json'
    state = json.loads(r.path(state_path).read_text())
    all_parents = ['published_recon_iid_'+str(i).zfill(3) for i in range(3)] + ['published_recon_misspec_'+str(i).zfill(3) for i in range(5)]
    parents = all_parents[2:]
    suffixes = {'generated','relate_inputs','native_relate','inferred','support','features','treeloom_observation','release_new_scratch'}
    selected = [copy.deepcopy(t) for t in old['tasks'] if t['id'].removeprefix('paper_argmix_').rsplit('.',1)[0] in parents and t['id'].rsplit('.',1)[1] in suffixes]
    assert len(selected) == 48
    module = 'research.joint_evaluation_20261029.queue_audit_20261030.critical_path.producer_v1.stage_v1'
    changes = {}
    for task in selected:
        if not task['id'].endswith('.generated'):
            continue
        parent = task['id'].removeprefix('paper_argmix_').rsplit('.',1)[0]
        index = all_parents.index(parent)
        dependencies = [] if index == 2 else ['paper_argmix_'+all_parents[index-2]+'.release_new_scratch']
        changes[task['id']] = dict(before=task['dependencies'], after=dependencies)
        task['dependencies'] = dependencies
        task['argv'][3] = module
    original_release = 'paper_argmix_published_recon_iid_001.release_new_scratch'
    selected.append(dict(id=original_release,kind='external',dependencies=[],state_path=state_path,
                         sidecar_plan_ref=original_plan_ref,source_task_id=original_release))
    owned = sorted(t['id'] for t in selected if t['kind'] != 'external')
    claims = sorted(changes)
    guard_id = 'paper_argmix_published_recon_iid_001.native_relate'
    guard = state['tasks'][guard_id]
    assert guard['status'] == 'running'
    identity = (Path('/proc')/str(state['controller_pid'])/'stat').read_text().rsplit(')',1)[1].split()[19]
    native_registry = json.loads(r.path('research/joint_evaluation_20261029/queue/native_argmix_retry_active_owner_v1.json').read_text())
    native_plan = r.read(native_registry['active_plan_ref'])
    native_state = json.loads(r.path(native_registry['state_path']).read_text())
    sealed = {}
    for task in native_plan['tasks']:
        row = native_state['tasks'][task['id']]
        if task['kind'] != 'external' and row['status'] == 'succeeded':
            sealed[task['id']] = dict(actual_wait_ref=r.ref(r.path(task['output_directory'])/'actual_wait.json'), result_ref=row['result_ref'])
    sources = [r.ref(folder/name) for name in ['support_v1.py','stage_v1.py','scheduler_v1.py','build_v1.py']]
    sources += [r.ref('research/joint_evaluation_20261029/queue_audit_20261030/common/probe_scheduler_v1.py'),
                r.ref('research/joint_evaluation_20261029/queue_audit_20261030/resources/probe_guard_v1.py')]
    spec = dict(schema='native_relate_parallel_producer_v1', original_plan_ref=original_plan_ref,
        original_state_path=state_path,original_controller=dict(pid=state['controller_pid'],start_ticks=identity),
        guard_task_id=guard_id,guard_pid=guard['pid'],guard_start_ticks=guard['process_start_ticks'],
        original_release_task_id=original_release,owned_task_ids=owned,claim_task_ids=claims,parents=parents,all_test_parents=all_parents,
        expected_pending_rows={name:state['tasks'][name] for name in owned},source_refs=sources,dependency_changes=changes,
        generation_wrapper_module=module,claim_owner='native_relate_parallel_producer_20261030_v1',
        producer_directory=str(folder.relative_to(r.ROOT)),records_directory=str((folder/'records_v1').relative_to(r.ROOT)),
        plan_path=str((folder/'plan_v1.json').relative_to(r.ROOT)),claims_receipt_path=str((folder/'claims_v1.json').relative_to(r.ROOT)),
        scope_path='experiments/paper_argmix_1611_rebuild_20261024_v1',scope_cap_bytes=137438953472,
        completion_guard_bytes=2**30,metadata_max_bytes=16*2**20,native_retry_plan_ref=native_registry['active_plan_ref'],
        sealed_comparator_outputs=sealed,native_algorithm_seeds_output_paths_and_caps_unchanged=True,
        maximum_unretired_parents=2,root_claims_and_activation_required=True)
    plan = copy.deepcopy(old)
    for path in [folder/'manifest_v1.json', folder/'plan_v1.json']:
        if path.exists():
            raise ValueError('Existing candidate must be preserved before a reviewed revision')
    r.save(folder/'manifest_v1.json',spec)
    plan.update(tasks=selected,records_directory=spec['records_directory'],controller_metadata_allowance_bytes=spec['metadata_max_bytes'],
                producer_manifest_ref=r.ref(folder/'manifest_v1.json'))
    plan['controller_source_refs'] = old['controller_source_refs']+sources
    r.save(folder/'plan_v1.json',plan)
    print(json.dumps(dict(manifest_ref=r.ref(folder/'manifest_v1.json'),plan_ref=r.ref(folder/'plan_v1.json')),sort_keys=True))


if __name__ == '__main__':
    main()

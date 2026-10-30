"""Focused continuation checks; all prospective waits exist only in mock memory."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
from . import continue_v1 as q
from research.joint_evaluation_20261029.root import finish_coordinator_storage_v2 as finish


def main():
    c=q.c
    reference=c.ref(q.HERE/'continuation_v1.json')
    spec,manifest,plan,prefix,binding,host_minimum=q.validate(reference)
    passed=['real_prefix_waits_products_and_original_sources_authenticated']
    assert len(q.remaining_tasks(plan))==13 and q.remaining_tasks(plan)[0]['id']=='joint_argmix.train'
    assert all(task['id'] not in q.PREFIX_IDS for task in q.remaining_tasks(plan))
    passed.append('no_cache_or_qualification_replay_among_exact_thirteen_remaining_tasks')
    assert q.c.stage.__module__=='research.joint_evaluation_20261029.argmix.recovery_20261030_queue.controller_v1'
    passed.append('unchanged_original_stage_function_imported_directly')
    task=q.remaining_tasks(plan)[0]
    required=q.gpu_requirement(spec,task)
    assert spec['measured_peak_reserved_bytes']>task['minimum_free_gpu_bytes']==40*2**30
    assert required==81522589696
    passed.append('actual_7392GiB_peak_above40GiB_floor_uses_measured_peak_plus2GiB')
    for free,apps,expected,name in [
        (81153,'',True,'empty80GiB_GPU_with_sufficient_measured_headroom_accepted'),
        (77000,'',False,'insufficient_measured_headroom_waits'),
        (81153,'9999\n',False,'active_compute_process_rejected_even_with_free_memory')]:
        with patch.object(c.subprocess,'check_output',side_effect=[f'3, GPU-mock, {free}\n',apps]):
            assert c.gpu_capacity(3,required)==('GPU-mock',expected)
        passed.append(name)
    original_binding={k:v for k,v in binding.items() if k!='operational_continuation_ref'}
    def bad_spec(name,mutate):
        changed=copy.deepcopy(spec);mutate(changed)
        try:q.authenticate_prefix(changed,manifest,plan,original_binding)
        except ValueError:passed.append(name)
        else:raise AssertionError(name+' accepted')
    bad_spec('reversed_prefix_waits_rejected',lambda s:s['prefix_actual_wait_refs'].reverse())
    bad_spec('reduced_headroom_rejected',lambda s:s.update(gpu_headroom_bytes=0))
    bad_spec('stale_qualification_peak_rejected',lambda s:s.update(measured_peak_reserved_bytes=40*2**30))
    real_read=c.read
    qual_path=real_read(prefix[1])['result_ref']['path']
    for field,value,name in [('passed',False,'failed_qualification_rejected'),('physical_gpu',4,'wrong_GPU_qualification_rejected'),('sampled_only',True,'sampled_qualification_rejected')]:
        def altered(ref):
            row=copy.deepcopy(real_read(ref))
            if ref['path']==qual_path:row[field]=value
            return row
        with patch.object(c,'read',side_effect=altered):
            try:q.authenticate_prefix(spec,manifest,plan,original_binding)
            except ValueError:passed.append(name)
            else:raise AssertionError(name+' accepted')
    def failed_wait(ref):
        row=copy.deepcopy(real_read(ref))
        if ref['path']==prefix[1]['path']:row['actual_returncode']=1
        return row
    with patch.object(c,'read',side_effect=failed_wait):
        try:q.authenticate_prefix(spec,manifest,plan,original_binding)
        except ValueError:passed.append('nonzero_actual_prefix_wait_rejected')
        else:raise AssertionError('Failed actual wait accepted')
    with patch.object(c,'historical_session_absent',return_value=False):
        try:q.validate(reference)
        except ValueError:passed.append('live_previous_owner_or_worker_session_rejected')
        else:raise AssertionError('Live historical owner accepted')
    gpu_tasks=[t for t in plan['tasks'] if t['kind']=='gpu']
    folder=Path(manifest['records_directory'])
    all_refs=[dict(path=str((folder/t['id']/'actual_wait.json').relative_to(c.ROOT)),sha256='a'*64) for t in plan['tasks']]
    all_refs[:2]=prefix
    gpu_refs=[r for r,t in zip(all_refs,plan['tasks']) if t['kind']=='gpu']
    rows={}
    for r,t in zip(all_refs,plan['tasks']):
        rows[r['path']]=dict(task_id=t['id'],plan_ref=manifest['plan_ref'],command=t['argv_by_physical_gpu']['3'] if t['kind']=='gpu' else t['argv'],
                            actual_wait=True,actual_returncode=0,owned_process_group_absent=True,products_complete=True,pid=999999999,
                            process_start_ticks='0',recovery_binding=binding)
    with patch.object(c,'read',side_effect=lambda r:rows[r['path']]):
        chain=q.chain_record(manifest,reference,binding,gpu_refs,gpu_tasks,gpu=True)
        final=q.chain_record(manifest,reference,binding,all_refs,plan['tasks'])
        try:q.chain_record(manifest,reference,binding,gpu_refs[1:],gpu_tasks,gpu=True)
        except ValueError:passed.append('missing_original_qualification_in_GPU_chain_rejected')
        else:raise AssertionError('Missing qualification accepted')
    assert len(chain['stage_actual_wait_refs'])==11 and chain['stage_actual_wait_refs'][0]==prefix[1]
    assert len(final['stage_actual_wait_refs'])==15 and final['stage_actual_wait_refs'][:2]==prefix
    passed.append('full11_GPU_and15_total_chains_retain_original_success_prefix')
    finish_plan=real_read(c.ref(c.BASE.parent/'root/finish_plan_v5.json'))
    finish_plan['model_chains']={'argmix_joint':finish_plan['model_chains']['argmix_joint']}
    chain_path=finish_plan['model_chains']['argmix_joint']['wait_path']
    rows[chain_path]=chain
    mock_refs={r['path']:r for r in all_refs};mock_refs[chain_path]=dict(path=chain_path,sha256='a'*64)
    original_path=finish.r.path
    class MockPath(type(Path())):
        def exists(self):return str(self.relative_to(c.ROOT)) in rows
    def mock_ref(path):return mock_refs[str(original_path(path).relative_to(c.ROOT))]
    with patch.object(finish.r,'path',side_effect=lambda p:MockPath(original_path(p))), patch.object(finish.r,'ref',side_effect=mock_ref), patch.object(finish.r,'read',side_effect=lambda r:rows[r['path']]), patch.object(finish.processes,'session_members',return_value=[]):
        assert finish.chain_completions(finish_plan)=={'argmix_joint':mock_refs[chain_path]}
        passed.append('actual_unchanged_finish_chain_completions_accepts_exact_original_bindings_plus_continuation')
        rows[gpu_refs[-1]['path']]['command']=['wrong']
        try:finish.chain_completions(finish_plan)
        except ValueError:passed.append('actual_finish_consumer_still_rejects_changed_scientific_command')
        else:raise AssertionError('Wrong command accepted by actual finish consumer')
    receipt=dict(schema='argmix_gpu_capacity_continuation_tests_v1',passed=True,tests=passed,
                 source_ref=c.ref(q.HERE/'continue_v1.py'),test_source_ref=c.ref(__file__),continuation_ref=reference,
                 actual_finish_source_ref=c.ref(c.BASE.parent/'root/finish_coordinator_storage_v2.py'),
                 production_launched=False,signals_sent=0,prospective_waits_only_in_mock_memory=True)
    p=q.HERE/'test_results_v1.json'
    with p.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(passed=True,tests=len(passed),result_ref=c.ref(p))))


if __name__=='__main__':main()

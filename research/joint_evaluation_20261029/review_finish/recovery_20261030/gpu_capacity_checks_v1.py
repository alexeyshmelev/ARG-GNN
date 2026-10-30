"""Independent continuation checks using genuine public metadata and synthetic terminal rows."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import copy,json,sys
from research.joint_evaluation_20261029.argmix.recovery_20261030_gpu_capacity_v1 import continue_v1 as v

c=v.c
ROOT=c.ROOT
REVIEW=ROOT/'research/joint_evaluation_20261029/review_finish/recovery_20261030'
checks=[]
def rejects(action):
    try:action()
    except ValueError:return
    raise AssertionError('Unsafe continuation accepted')

def main():
    continuation_ref=c.ref(sys.argv[1])
    spec,manifest,plan,prefix,binding,host=v.validate(continuation_ref)
    original_binding=dict(binding);original_binding.pop('operational_continuation_ref')
    assert len(prefix)==2 and binding['operational_continuation_ref']==continuation_ref and host==448*2**30
    assert spec['minimum_free_gpu_bytes']==81522589696 and spec['measured_peak_reserved_bytes']==79375106048
    checks.append('actual_successful_cache_and_qualification_waits_and_all_source_bindings_authenticate')
    assert v.remaining_tasks(plan)==plan['tasks'][2:] and len(v.remaining_tasks(plan))==13
    assert v.remaining_tasks(plan)[0]['id']=='joint_argmix.train'
    checks.append('exact13unfinished_stages_preserved_without_cache_or_qualification_replay')
    for task in plan['tasks']:
        if task['kind']=='gpu':assert v.gpu_requirement(spec,task)==max(task['minimum_free_gpu_bytes'],79375106048+2*2**30)
    assert v.gpu_requirement(spec,dict(minimum_free_gpu_bytes=90*2**30))==90*2**30
    checks.append('GPU_requirement_is_measured_peak_plus2GiB_admission_floor_never_a_maximum_cap')
    bad=copy.deepcopy(plan);bad['tasks'][:2]=list(reversed(bad['tasks'][:2]))
    rejects(lambda:v.remaining_tasks(bad))
    checks.append('changed_successful_prefix_order_rejected')
    for mutation in ('swap','truncate','hash'):
        bad=copy.deepcopy(spec)
        if mutation=='swap':bad['prefix_actual_wait_refs'].reverse()
        elif mutation=='truncate':bad['prefix_actual_wait_refs']=bad['prefix_actual_wait_refs'][:1]
        else:bad['prefix_actual_wait_refs'][0]['sha256']='0'*64
        rejects(lambda bad=bad:v.authenticate_prefix(bad,manifest,plan,original_binding))
    checks.append('swapped_missing_or_changed_original_wait_references_rejected')
    actual_read=c.read
    for key,value in [('actual_returncode',1),('products_complete',False),('owned_process_group_absent',False)]:
        def broken_read(reference):
            row=actual_read(reference)
            if reference==prefix[0]:row=dict(row,**{key:value})
            return row
        with patch.object(c,'read',side_effect=broken_read):
            rejects(lambda:v.authenticate_prefix(spec,manifest,plan,original_binding))
    checks.append('failed_incomplete_or_undrained_prefix_cannot_be_adopted')
    for key in ('gpu_headroom_bytes','measured_peak_reserved_bytes','minimum_free_gpu_bytes'):
        bad=dict(spec);bad[key]-=1
        rejects(lambda bad=bad:v.authenticate_prefix(bad,manifest,plan,original_binding))
    checks.append('tampered_measured_peak_headroom_or_derived_floor_rejected')
    folder=Path(manifest['records_directory'])
    tasks=plan['tasks'];gpu_tasks=[task for task in tasks if task['kind']=='gpu']
    references=[dict(path=str((folder/task['id']/'actual_wait.json').relative_to(ROOT)),sha256='synthetic') for task in tasks]
    references[:2]=prefix
    gpu_refs=[reference for task,reference in zip(tasks,references) if task['kind']=='gpu']
    with patch.object(c,'read',return_value=dict(pid=2147483600,process_start_ticks='synthetic')):
        gpu=v.chain_record(manifest,continuation_ref,binding,gpu_refs,gpu_tasks,gpu=True)
        all_stages=v.chain_record(manifest,continuation_ref,binding,references,tasks)
        assert gpu['stage_actual_wait_refs'][0]==prefix[1] and len(gpu['stage_actual_wait_refs'])==11
        assert all_stages['stage_actual_wait_refs'][:2]==prefix and len(all_stages['stage_actual_wait_refs'])==15
        assert gpu['manifest_ref']==binding['manifest_ref'] and gpu['plan_ref']==manifest['plan_ref']
        assert gpu['operational_continuation_ref']==continuation_ref and all_stages['operational_continuation_ref']==continuation_ref
        for wrong in [gpu_refs[:-1],list(reversed(gpu_refs)),gpu_refs[:-1]+[gpu_refs[0]]]:
            rejects(lambda wrong=wrong:v.chain_record(manifest,continuation_ref,binding,wrong,gpu_tasks,gpu=True))
    checks.append('final11GPU_and15total_chain_records_preserve_original_bindings_and_explicit_new_provenance')
    checks.append('missing_reordered_or_duplicate_stage_chain_references_rejected')
    state_path=folder/'state.json'
    state=actual_read(c.ref(state_path))
    assert state['status']=='failed' and c.ref(state_path)['sha256']==spec['preserved_failed_state_ref']['sha256']
    assert not (folder/'joint_argmix.train/started.json').exists()
    checks.append('original_controller_failure_preserved_and_no_training_started_during_review')
    output=REVIEW/'gpu_capacity_checks_result_v1.json'
    receipt=dict(schema='independent_ARGMix_GPU_admission_continuation_checks_v1',passed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),
        source_ref=c.ref(v.__file__),continuation_ref=continuation_ref,original_controller_source_ref=c.ref(c.__file__),test_source_ref=c.ref(__file__),
        original_manifest_ref=binding['manifest_ref'],original_plan_ref=manifest['plan_ref'],checks=checks,
        production_launched=False,signals_sent=0,heldout_truth_opened=False,synthetic_chain_rows_written=False)
    with output.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,checks=len(checks),receipt_ref=c.ref(output)),sort_keys=True))

if __name__=='__main__':main()

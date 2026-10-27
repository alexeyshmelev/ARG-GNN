"""CPU metadata/AST contracts only; no child, model, CUDA or scientific arrays."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import fit_cli_v1 as cli,fit_gate_v1 as gate,progress_v1 as progress
from research.efficiency_20261027.architecture.european_cohort_train import driver_v2 as driver

HERE=Path(__file__).parent

def rejected(action):
    try:action()
    except (ValueError,RuntimeError,KeyError,IndexError):return
    raise AssertionError('Expected fail-closed contract rejection')


def main():
    checks=[]
    source=gate.ROOT/'research/efficiency_20261027/root/run_guarded_profile_with_peer.py'
    expected=source.read_text().replace('Run one isolated profile when live benchmark growth allowances leave room.',
        'Guard one isolated fixed90 FIT with an explicit2GiB whole-run allowance.')
    expected=expected.replace('    output_cap = 64 * 2**20','    output_cap = 2 * GIB')
    expected=expected.replace("if not directory.is_relative_to(ROOT / 'research/efficiency_20261027'):",
        "if directory.parent != ROOT / 'research/efficiency_20261027/root/runs':")
    expected=expected.replace("raise ValueError('Profile output must remain inside the efficiency study')",
        "raise ValueError('FIT output must be one fresh ROOT run directory')")
    assert (HERE/'run_guarded_fit_v1.py').read_text()==expected
    checks.append('Guardian source differs only in docstring,2GiB allowance,and immediate ROOT-run confinement')
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    checks.append('All authored source parses; no executable payload encoding')
    runtime=driver.read_small(driver.ORIGINAL_JOB_REF)['call']['spec']['scientific_spec']
    plan=driver.original._parent_plan(runtime['sampling_plan_ref'],runtime['protocol_id'])
    qualification=dict(schema='European_cohort8_full_parent_qualification_v2',passed=True,
        source_manifest_ref=gate.SOURCE_MANIFEST_REF,scientific_condition=driver.CONDITION,all_original_rows_used=True,
        qualified_parent=plan['train_parent_ids'][0],prepared_parents=plan['train_parent_ids'],tolerances=driver.TOLERANCES,
        gradient_tensors=49,parameter_tensors=49,Adam_tensors=147,sampled_rows=16667,samples=952,argmax_changes=0,
        TEST_truth_read=False,no_qualification_checkpoint_used_for_fit=True,worker_peak_RSS_bytes=100*2**30)
    gate.require_qualification(qualification)
    for key,value in [('schema','European_cohort8_full_parent_qualification_v1'),('passed',False),
        ('sampled_rows',1),('samples',951),('argmax_changes',1),('gradient_tensors',48),('parameter_tensors',48),
        ('Adam_tensors',146),('all_original_rows_used',False),('qualified_parent','different'),('prepared_parents',[]),
        ('tolerances',{}),('worker_peak_RSS_bytes',513*2**30),('TEST_truth_read',True),
        ('no_qualification_checkpoint_used_for_fit',False),('source_manifest_ref',{}),('scientific_condition','wrong')]:
        value0=copy.deepcopy(qualification);value0[key]=value
        rejected(lambda:gate.require_qualification(value0))
    checks.append('Full original200Mb qualification passes;17 concrete incomplete/wrong-binding cases reject')
    directory=gate.RUNS/'synthetic_fit_contract_no_files'
    qref=dict(path=str((directory/'data/qualification.json').relative_to(gate.ROOT)),sha256='a'*64)
    release=dict(schema='European_cohort8_TRAIN_release_v1',passed=True,source_manifest_ref=gate.SOURCE_MANIFEST_REF,
        source_review_ref=gate.SOURCE_REVIEW_REF,qualification_ref=qref,qualification_actual_wait_returncode=0,
        qualification_process_family_absent=True,live_process_family_host_GPU_storage_admission_required=True)
    gate.require_release(release,qref)
    for key,value in [('passed',False),('qualification_ref',{}),('source_manifest_ref',{}),('source_review_ref',{}),
        ('qualification_actual_wait_returncode',-15),('qualification_process_family_absent',False),
        ('live_process_family_host_GPU_storage_admission_required',False)]:
        item=copy.deepcopy(release);item[key]=value;rejected(lambda:gate.require_release(item,qref))
    checks.append('Exact release gate rejects failed wait,live family,source mismatch and removed live admission')
    command=['.runtime/env/bin/python3.11','-B','-m',gate.QUALIFICATION_MODULE,'--output',str(directory/'data'),
        '--source-pins-sha256',gate.SOURCE_MANIFEST_REF['sha256'],'--source-review-sha256',gate.SOURCE_REVIEW_REF['sha256']]
    request=dict(command=command,physical_gpu=5,gpu_uuid=gate.GPU_UUID)
    launch=dict(actual_argv=command,pid=123456,process_start_ticks='1234567')
    wait=dict(actual_wait=True,owned_group_absent=True,actual_returncode=0,pid=123456,process_start_ticks='1234567')
    def fake_ref(path):return dict(path=str(Path(path).relative_to(gate.ROOT)),sha256='a'*64)
    request_ref=fake_ref(directory/'request.json');launch_ref=fake_ref(directory/'launch.json')
    completion=dict(schema='European_cohort8_qualification_CLI_completion_v1',passed=True,qualification_ref=qref,
        source_manifest_ref=gate.SOURCE_MANIFEST_REF,actual_ROOT_request_ref=request_ref,actual_ROOT_launch_ref=launch_ref)
    release_path=gate.ROOT/'research/efficiency_20261027/root/synthetic_FIT_release_no_file.json'
    values={str(directory/'data/qualification.json'):qualification,str(release_path):release,
        str(directory/'request.json'):request,str(directory/'launch.json'):launch,str(directory/'actual_wait.json'):wait,
        str(directory/'data/qualification_wrapper_completed.json'):completion}
    def fake_read(path,digest=None):return copy.deepcopy(values[str(gate.contained(path))]),fake_ref(gate.contained(path))
    def run_gate():return gate.qualification_release(qref['path'],'a'*64,str(release_path),'a'*64)
    with patch.object(gate,'read',fake_read),patch.object(gate,'identity',return_value=None),patch.object(gate,'owned_group_members',return_value=[]):
        assert run_gate()['actual_returncode']==0
        with patch.object(gate,'identity',return_value='1234567'):rejected(run_gate)
        with patch.object(gate,'owned_group_members',return_value=[123456]):rejected(run_gate)
        wait['actual_returncode']=-15;rejected(run_gate);wait['actual_returncode']=0
        launch['process_start_ticks']='wrong';rejected(run_gate);launch['process_start_ticks']='1234567'
        completion['qualification_ref']={};rejected(run_gate);completion['qualification_ref']=qref
        request['command']=[];rejected(run_gate);request['command']=command
    checks.append('Complete in-memory qualification/request/launch/wait chain;live leader,descendant,failed/mismatched wait and receipts reject')
    args=SimpleNamespace(required_peer_directory=[],family_cap_bytes=512*2**30,device_soft_bytes=70*2**30,
        output_cap_bytes=2*2**30,max_seconds=864000,gpu_wait_seconds=86400)
    argv=['--fixture','metadata_only'];own_command=['.runtime/env/bin/python3.11','-B','-m',cli.MODULE,*argv]
    own_request=dict(command=own_command,peer_directories=[],supervisor_pid=222,supervisor_start_ticks='22',
        physical_gpu=5,gpu_uuid=gate.GPU_UUID,host_cap_bytes=args.family_cap_bytes,
        device_budget_bytes=args.device_soft_bytes,output_cap_bytes=args.output_cap_bytes)
    own_launch=dict(actual_argv=own_command,pid=333,process_start_ticks='33')
    def own():cli.validate_guardian_records(own_request,own_launch,args,argv,333,'33',222,'22')
    own()
    for key,value in [('physical_gpu',3),('host_cap_bytes',1),('output_cap_bytes',64*2**20),('peer_directories',['wrong']),
                      ('supervisor_start_ticks','other')]:
        old=own_request[key];own_request[key]=value;rejected(own);own_request[key]=old
    checks.append('Exact own ROOT guardian identity,visible argv,current peers and matching resource allowances')
    rows=[];checkpoints={}
    for index in range(91):
        if index:rows.append(dict(epoch=(index-1)//3+1,optimizer_updates=index,parent_id=plan['train_parent_ids'][(index-1)%3],
            one_complete_parent_optimizer_update=True))
        if index and index%15==0:checkpoints[index//3]=dict(path='synthetic',sha256='b'*64)
        result=progress.summarize(rows,checkpoints)
        assert result['durable_optimizer_updates']==index and result['resume_supported'] is False
    assert result['completed_epochs']==30 and result['latest_checkpoint_epoch']==30
    rejected(lambda:progress.summarize(rows[1:],checkpoints))
    bad=copy.deepcopy(rows);bad[-1]['epoch']=29;rejected(lambda:progress.summarize(bad,checkpoints))
    rejected(lambda:progress.summarize([], {5:{}}))
    checks.append('All91 progress positions retain90-update ceiling and six checkpoints;gaps/wrong epochs/early checkpoints reject')
    driver_ref=gate.reference(Path(driver.__file__))
    assert driver_ref in json.loads((Path(driver.__file__).parent/'source_pins_v2.json').read_text())['source_refs']
    assert not driver.torch.cuda.is_initialized()
    result=dict(schema='European_cohort8_FIT_CLI_CPU_contracts_v1',passed=True,checks=checks,source_refs=[gate.reference(p) for p in sorted(HERE.glob('*.py'))],
        driver_ref=driver_ref,scientific_arrays_opened=False,model_constructed=False,CUDA_initialized=False,
        child_process_launched=False,production_modified=False,resume_test_performed=False,fresh_fit_only=True)
    path=HERE/'CONTRACTS_CPU_v1.json';path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),evidence=gate.reference(path))))

if __name__=='__main__':main()

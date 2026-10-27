"""CPU-only supervision,interruption gate and file-backed progress contracts."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import fit_gate_v2 as gate,fit_cli_v2 as cli,progress_v2 as progress
from .test_contracts_v1 import rejected
from research.efficiency_20261027.architecture.european_cohort_train import driver_v3 as driver

HERE=Path(__file__).parent

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    return gate.reference(path)


def main():
    checks=[]
    for path in [*HERE.glob('*.py'),Path(driver.__file__)]:ast.parse(path.read_text(),filename=str(path))
    checks.append('All source parses;v2 numerical module reused without mutation')
    fixture=HERE/'cpu_fixture_v2';fixture.mkdir(exist_ok=False)
    prior=fixture/'prior';output=prior/'data';(output/'resume').mkdir(parents=True)
    checkpoint=output/'resume/update_017.pt';checkpoint.write_bytes(b'CPU gate fixture;not a Torch payload')
    checkpoint_ref=gate.reference(checkpoint)
    command=['.runtime/env/bin/python3.11','-B','-m',cli.MODULE,'--output',str(output)]
    request=dict(command=command);launch=dict(actual_argv=command,pid=99999999,process_start_ticks='1234')
    wait=dict(actual_wait=True,owned_group_absent=True,actual_returncode=-15,pid=99999999,process_start_ticks='1234')
    put(prior/'request.json',request);put(prior/'launch.json',launch);put(prior/'actual_wait.json',wait)
    with patch.object(gate,'RUNS',fixture),patch.object(gate,'identity',return_value=None),patch.object(gate,'owned_group_members',return_value=[]):
        def run():return gate.resume_release(str(checkpoint),checkpoint_ref['sha256'])
        assert run()['prior_actual_returncode']==-15
        with patch.object(gate,'identity',return_value='1234'):rejected(run)
        with patch.object(gate,'owned_group_members',return_value=[99999999]):rejected(run)
        wait['owned_group_absent']=False;put(prior/'actual_wait.json',wait);rejected(run)
        wait['owned_group_absent']=True;wait['process_start_ticks']='other';put(prior/'actual_wait.json',wait);rejected(run)
        wait['process_start_ticks']='1234';put(prior/'actual_wait.json',wait)
        rejected(lambda:gate.resume_release(str(checkpoint),'0'*64))
        assert gate.resume_release(None,None) is None
        rejected(lambda:gate.resume_release(None,'0'*64))
        rejected(lambda:gate.resume_release(str(checkpoint),None))
    checks.append('Actual reaped interruptedFIT may resume;live leader/session,wrong wait,wrong hash or half-specified source reject')
    # File-backed progress preserves inherited1..17 and original epoch5, then
    # appends18 and the next exact immutable resume artifact in a new output.
    events=[]
    for index in range(1,18):
        events.append(put(output/'updates'/('update_%03d.json'%index),dict(epoch=(index-1)//3+1,
            optimizer_updates=index,parent_id='TRAIN'+str(index%3),one_complete_parent_optimizer_update=True)))
    epoch=output/'checkpoints/epoch_05.pt';epoch.parent.mkdir();epoch.write_bytes(b'synthetic epoch5')
    current=fixture/'current/data';current.mkdir(parents=True)
    origin=dict(resume_checkpoint_ref=checkpoint_ref,start_update=17,update_event_refs=events,
        checkpoint_event_refs={'5':gate.reference(epoch)})
    put(current/'fit_resume_origin.json',origin)
    cache={};value=progress.snapshot(current,cache)
    assert value['durable_optimizer_updates']==17 and value['latest_resume_checkpoint_ref']==checkpoint_ref
    put(current/'updates/update_018.json',dict(epoch=6,optimizer_updates=18,parent_id='TRAIN0',one_complete_parent_optimizer_update=True))
    next_checkpoint=current/'resume/update_018.pt';next_checkpoint.parent.mkdir();next_checkpoint.write_bytes(b'synthetic resume18')
    value=progress.snapshot(current,cache)
    assert value['durable_optimizer_updates']==18 and value['completed_epochs']==6
    assert value['resume_supported'] is True and value['latest_resume_checkpoint_ref']==gate.reference(next_checkpoint)
    epoch.write_bytes(b'altered epoch5')
    rejected(lambda:progress.snapshot(current,cache))
    epoch.write_bytes(b'synthetic epoch5')
    checks.append('Actual tiny files:inherited17+new18 progression and checkpoint index work;immutable checkpoint mutation rejects')
    argv=['--fixture','metadata_only'];args=SimpleNamespace(required_peer_directory=[],family_cap_bytes=512*2**30,
        device_soft_bytes=70*2**30,output_cap_bytes=2*2**30,max_seconds=864000,gpu_wait_seconds=86400)
    command=['.runtime/env/bin/python3.11','-B','-m',cli.MODULE,*argv]
    request=dict(command=command,peer_directories=[],supervisor_pid=222,supervisor_start_ticks='22',physical_gpu=5,
        gpu_uuid=gate.GPU_UUID,host_cap_bytes=args.family_cap_bytes,device_budget_bytes=args.device_soft_bytes,output_cap_bytes=args.output_cap_bytes)
    launch=dict(actual_argv=command,pid=333,process_start_ticks='33')
    cli.validate_guardian_records(request,launch,args,argv,333,'33',222,'22')
    old=request['command'];request['command']=['wrong'];rejected(lambda:cli.validate_guardian_records(request,launch,args,argv,333,'33',222,'22'));request['command']=old
    checks.append('Resume-capable CLI still requires exact guardian argv,resources and PID generations')
    assert not driver.torch.cuda.is_initialized()
    result=dict(schema='European_cohort8_FIT_CLI_CPU_contracts_v2',passed=True,checks=checks,
        source_refs=[gate.reference(path) for path in sorted(HERE.glob('*.py'))]+[gate.reference(Path(driver.__file__))],
        fixture_directory=str(fixture.relative_to(gate.ROOT)),scientific_arrays_opened=False,model_constructed=False,
        CUDA_initialized=False,child_process_launched=False,production_modified=False)
    path=HERE/'CONTRACTS_CPU_v2.json';put(path,result)
    print(json.dumps(dict(passed=True,checks=len(checks),evidence=gate.reference(path))))

if __name__=='__main__':main()

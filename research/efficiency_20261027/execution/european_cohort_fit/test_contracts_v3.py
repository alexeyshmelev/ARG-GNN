"""Focused CPU tests for isolated preprocessing integration and unchanged FIT gates."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from . import fit_cli_v2 as old_cli,fit_cli_v3 as cli,fit_gate_v2 as old_gate,fit_gate_v3 as gate
from research.efficiency_20261027.fidelity.european_fit_preprocessing_v1 import bindings_v1 as bridge

HERE=Path(__file__).parent
CHECKS=[]


def check(name,ok=True):
    assert ok,name
    CHECKS.append(name)


def fails(name,function):
    try:function()
    except ValueError:check(name)
    else:raise AssertionError(name)


def node(source,name):
    return next(item for item in ast.walk(ast.parse(source)) if isinstance(item,ast.FunctionDef) and item.name==name)


def ast_same(left,right):return ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)


def main():
    old=Path(old_cli.__file__).read_text();new=Path(cli.__file__).read_text()
    check('guardian_command_resource_and_PID_validation_AST_unchanged',ast_same(node(old,'validate_guardian_records'),node(new,'validate_guardian_records')))
    for name in ('write','progress','check'):
        check(name+'_guard_AST_unchanged',ast_same(node(old,name),node(new,name)))
    acquire=node(new,'acquire_gpu0')
    for value in ast.walk(acquire):
        if isinstance(value,ast.Call):value.keywords=[kw for kw in value.keywords if kw.arg!='preprocessing_bindings_ref']
    check('late_GPU_acquisition_AST_unchanged_except_receipt_reference',ast_same(node(old,'acquire_gpu0'),acquire))
    check('qualification_gate_is_exact_original_callable',gate.qualification_release is old_gate.qualification_release)
    check('unchanged_numerical_driver_v3_still_imported',
          'from research.efficiency_20261027.architecture.european_cohort_train import driver_v3 as driver' in new)
    _,runtime=bridge.authenticate(cli.PREPROCESSING_SOURCE_REF)
    trace=[];receipt={'synthetic':'actual-call-order'};receipt_ref={'path':'synthetic_receipt','sha256':'0'*64}
    def resource_check(stage):trace.append(('guard',stage))
    def install(**kwargs):
        check('bridge_receives_exact_original_envelope_inputs_and_guard',
              kwargs['max_rss_bytes']==runtime['memory_envelope']['max_rss_bytes']
              and kwargs['preparation_headroom_bytes']==runtime['memory_envelope']['preparation_headroom_bytes']
              and kwargs['train_inputs'] is runtime['train_inputs']
              and kwargs['resource_check'] is resource_check
              and kwargs['source_manifest_ref']==cli.PREPROCESSING_SOURCE_REF)
        trace.append('install');return receipt
    def record(value):assert value is receipt;trace.append('record');return receipt_ref
    def verify(value):assert value is receipt;trace.append('verify');return True
    def acquire_gpu():trace.append('acquire');return 'lease'
    spec={'synthetic':'opaque_spec'}
    def run_fit(value,*,resource_check,acquire_gpu0):
        assert value is spec
        trace.append('run_fit');assert acquire_gpu0()=='lease'
        return {'synthetic_result':90}
    driver=SimpleNamespace(run_fit=run_fit)
    with patch.object(bridge,'install',side_effect=install),patch.object(bridge,'verify',side_effect=verify):
        result,actual_ref=cli.run_with_preprocessing(driver,spec,runtime,resource_check=resource_check,
            acquire_gpu0=acquire_gpu,record_bindings=record)
    check('install_record_verify_driver_lateGPU_verify_order_exact',trace==['install','record','verify','run_fit','verify','acquire','verify'])
    check('unchanged_driver_result_and_recorded_receipt_returned',result=={'synthetic_result':90} and actual_ref is receipt_ref)
    with patch.object(bridge,'install',return_value=receipt),patch.object(bridge,'verify',return_value=True),patch.object(driver,'run_fit') as run:
        def refuse(value):raise ValueError('synthetic output refusal')
        fails('record_failure_prevents_driver_run',lambda:cli.run_with_preprocessing(driver,spec,runtime,
            resource_check=resource_check,acquire_gpu0=acquire_gpu,record_bindings=refuse))
        check('driver_not_called_after_record_failure',not run.called)
    with patch.object(bridge,'install',return_value=receipt),patch.object(bridge,'verify',side_effect=ValueError('changed binding')),patch.object(driver,'run_fit') as run:
        fails('binding_failure_prevents_driver_run',lambda:cli.run_with_preprocessing(driver,spec,runtime,
            resource_check=resource_check,acquire_gpu0=acquire_gpu,record_bindings=record))
        check('driver_not_called_after_binding_failure',not run.called)
    trace.clear()
    with patch.object(bridge,'install',return_value=receipt),patch.object(bridge,'verify',side_effect=[True,ValueError('changed before GPU')]):
        fails('binding_failure_prevents_lateGPU_acquisition',lambda:cli.run_with_preprocessing(driver,spec,runtime,
            resource_check=resource_check,acquire_gpu0=acquire_gpu,record_bindings=record))
    check('lateGPU_not_called_after_binding_failure','acquire' not in trace)
    with tempfile.TemporaryDirectory(prefix='cpu_gate_v3_',dir=HERE) as temporary:
        fixture=Path(temporary);prior=fixture/'prior';output=prior/'data';(output/'resume').mkdir(parents=True)
        checkpoint=output/'resume/update_017.pt';checkpoint.write_bytes(b'Synthetic metadata fixture; no model or optimizer')
        checkpoint_ref=gate.reference(checkpoint)
        request=dict(command=['.runtime/env/bin/python3.11','-B','-m',cli.MODULE,'--output',str(output)])
        launch=dict(actual_argv=request['command'],pid=99999999,process_start_ticks='1234')
        wait=dict(actual_wait=True,owned_group_absent=True,actual_returncode=-15,pid=99999999,process_start_ticks='1234')
        def save():
            for name,value in (('request',request),('launch',launch),('actual_wait',wait)):
                (prior/(name+'.json')).write_text(json.dumps(value)+'\n')
        save()
        with patch.object(gate,'RUNS',fixture),patch.object(gate,'identity',return_value=None),patch.object(gate,'owned_group_members',return_value=[]):
            def resume():return gate.resume_release(str(checkpoint),checkpoint_ref['sha256'])
            check('reaped_v3_resume_is_accepted',resume()['prior_actual_returncode']==-15)
            request['command'][3]=old_cli.MODULE;save()
            check('reaped_v2_resume_is_still_accepted',resume()['prior_actual_returncode']==-15)
            request['command'][3]='unknown';save();fails('unknown_prior_worker_rejected',resume)
            request['command'][3]=cli.MODULE;save()
            with patch.object(gate,'identity',return_value='1234'):fails('live_prior_leader_rejected',resume)
            with patch.object(gate,'owned_group_members',return_value=[99999999]):fails('live_prior_family_rejected',resume)
            wait['owned_group_absent']=False;save();fails('unreaped_prior_wait_rejected',resume)
            wait['owned_group_absent']=True;wait['process_start_ticks']='changed';save();fails('wrong_prior_generation_rejected',resume)
            wait['process_start_ticks']='1234';save()
            fails('wrong_checkpoint_hash_rejected',lambda:gate.resume_release(str(checkpoint),'0'*64))
            check('fresh_fit_has_no_resume_gate',gate.resume_release(None,None) is None)
    argv=['--synthetic','metadata_only']
    args=SimpleNamespace(required_peer_directory=[],family_cap_bytes=512*2**30,device_soft_bytes=70*2**30,
        output_cap_bytes=2*2**30,max_seconds=864000,gpu_wait_seconds=86400)
    command=['.runtime/env/bin/python3.11','-B','-m',cli.MODULE,*argv]
    request=dict(command=command,peer_directories=[],supervisor_pid=222,supervisor_start_ticks='22',physical_gpu=5,
        gpu_uuid=gate.GPU_UUID,host_cap_bytes=args.family_cap_bytes,device_budget_bytes=args.device_soft_bytes,output_cap_bytes=args.output_cap_bytes)
    launch=dict(actual_argv=command,pid=333,process_start_ticks='33')
    cli.validate_guardian_records(request,launch,args,argv,333,'33',222,'22')
    check('actual_v3_guardian_record_contract_passes')
    request['command']=['wrong'];fails('changed_current_ROOT_command_rejected',lambda:cli.validate_guardian_records(request,launch,args,argv,333,'33',222,'22'))
    import torch
    check('no_CUDA_initialized',not torch.cuda.is_initialized())
    result=dict(schema='European_cohort8_FIT_CLI_CPU_contracts_v3',passed=True,checks=CHECKS,check_count=len(CHECKS),
        preprocessing_source_manifest_ref=cli.PREPROCESSING_SOURCE_REF,
        preprocessing_actual_route_CPU_ref=gate.reference(bridge.HERE/'CPU_TEST_RESULTS_v1.json'),
        source_refs=[gate.reference(HERE/name) for name in ('fit_cli_v3.py','fit_gate_v3.py','test_contracts_v3.py')],
        original_driver_unmodified=True,scientific_inputs_or_checkpoint_opened=False,
        model_constructed=False,CUDA_initialized=False,training_or_native_launches=0)
    path=HERE/'CONTRACTS_CPU_v3.json'
    with path.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=True,checks=len(CHECKS),evidence=gate.reference(path))))


if __name__=='__main__':main()

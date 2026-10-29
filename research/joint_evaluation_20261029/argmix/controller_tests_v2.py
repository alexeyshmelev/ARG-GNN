"""Bounded controller regression tests with mocked children; never launch a stage."""
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
from . import controller_v2 as c
from .runtime import BASE, ref, now


def main():
    scratch = BASE / 'controller_test_scratch_v2'
    scratch.mkdir(exist_ok=False)
    plan_ref, freeze_ref = ref(BASE/'task_plan_v2.json'), ref(BASE/'freeze_v1.json')
    command = [str(c.ROOT/'.runtime/env/bin/python'),'-B','-m','research.joint_evaluation_20261029.argmix.pipeline','barrier']
    passed = []
    def fixture(name):
        folder = scratch/name/'records'; folder.mkdir(parents=True)
        output = scratch/name/'output'
        spec = dict(id='joint_argmix.mock_'+name,output_directory=str(output),completion_file='completed.json',
                    max_seconds=100,max_rss_bytes=64*2**30,kind='cpu')
        return folder,output,spec
    class Child:
        pid=os.getpid()
        def __init__(self,code=0,timeout=False):self.code,self.timeout=code,timeout
        def wait(self,timeout=None):
            if self.timeout:
                self.timeout=False
                raise subprocess.TimeoutExpired(command,timeout)
            return self.code
    def run(name,code=0,emit=True,timeout=False,lock=None,rss=0):
        folder,output,spec=fixture(name)
        child=Child(code,timeout)
        def popen(*args,**kwargs):
            if emit:
                output.mkdir(parents=True)
                (output/'completed.json').write_text('{"mock_only": true}\n')
            return child
        with patch.object(c.subprocess,'Popen',side_effect=popen) as spawn, patch.object(c.processes,'drain',return_value=True) as drain, patch.object(c,'owned_rss',return_value=rss):
            failure=None
            try: result=c.stage(spec,command,{},folder,plan_ref,freeze_ref,24*2**30,lock)
            except ValueError as error:failure=str(error);result=None
        return folder,output,spec,result,failure,spawn,drain
    folder,output,spec,result,failure,spawn,drain=run('successful_wait')
    assert failure is None and spawn.call_count==1 and drain.call_count==1
    row=json.loads((output/'actual_wait.json').read_text())
    assert row['actual_wait'] and row['actual_returncode']==0 and row['status']=='succeeded' and set(row['result_ref'])=={'path','sha256'}
    passed.append('genuine_wait_fields_and_exact_product_reference')
    from arggnn.benchmarking.paper_campaign import runtime as strict_runtime
    from research.joint_evaluation_20261029.queue.successor_v2_support import future_result
    from arggnn.benchmarking.paper_argmix import paper_argmix_score as native_score
    from .runtime import require_wait, RECIPE_REF
    assert strict_runtime.read(row['result_ref']) == {'mock_only': True}
    native_ref, native_wait_ref = require_wait(output/'completed.json', spec['id'])
    assert native_ref['path'] == row['result_ref']['path'] and native_ref['sha256'] == row['result_ref']['sha256']
    assert native_ref['bytes'] == (output/'completed.json').stat().st_size
    alias = dict(future_wait_path=str(output/'actual_wait.json'),source_task_id=spec['id'],
                 producer_model_id=c.MODEL_ID,expected_result_path=str(output/'completed.json'),config_ref=RECIPE_REF)
    with patch.object(c.processes,'session_members',return_value=[]), patch.object(native_score,'verify_barrier',return_value={'mock_only':True}) as verifier:
        assert future_result(alias) == row['result_ref']
        assert verifier.call_count == 1
    passed.append('actual_successor_future_result_and_original_require_wait_accept_same_strict_reference')
    legacy = scratch/'legacy_extra_bytes'; legacy.mkdir()
    legacy_product = legacy/'completed.json'; legacy_product.write_text('{"mock_only": true}\n')
    legacy_row = dict(row, result_ref=dict(ref(legacy_product),bytes=legacy_product.stat().st_size))
    (legacy/'actual_wait.json').write_text(json.dumps(legacy_row))
    legacy_alias = dict(alias,future_wait_path=str(legacy/'actual_wait.json'),expected_result_path=str(legacy_product))
    with patch.object(c.processes,'session_members',return_value=[]), patch.object(native_score,'verify_barrier',side_effect=AssertionError('Must reject before verifier')):
        try:future_result(legacy_alias)
        except ValueError:passed.append('legacy_extra_bytes_reference_rejected_by_actual_successor_strict_reader')
        else:raise AssertionError('Legacy incompatible reference was accepted')

    with patch.object(c.subprocess,'Popen',side_effect=AssertionError('Replay forbidden')), patch.object(c.processes,'session_members',return_value=[]):
        assert c.stage(spec,command,{},folder,plan_ref,freeze_ref,24*2**30)==result
    passed.append('successful_exact_stage_reused_without_process_replay')
    for name,code,emit in [('nonzero_exit',7,False),('missing_product',0,False)]:
        f,o,s,r,error,spawn,drain=run(name,code,emit)
        row=json.loads((f/s['id']/'actual_wait.json').read_text())
        assert error and row['actual_returncode']==code and row['status']=='failed' and not (o/'actual_wait.json').exists()
        passed.append(name+'_stops_without_success_publication')
    f,o,s=fixture('partial_started');(f/s['id']).mkdir();(f/s['id']/'started.json').write_text('{}')
    with patch.object(c.subprocess,'Popen',side_effect=AssertionError('No silent replay')):
        try:c.stage(s,command,{},f,plan_ref,freeze_ref,24*2**30)
        except ValueError:passed.append('partial_started_stage_preserved_without_replay')
        else:raise AssertionError('Partial stage accepted')
    with (scratch/'mock_gpu.lock').open('a+b') as lock:
        f,o,s,r,error,spawn,drain=run('gpu_fd',lock=lock)
        assert error is None and spawn.call_args.kwargs['pass_fds']==(lock.fileno(),)
    passed.append('owned_GPU_child_inherits_mutex_descriptor')
    f,o,s,r,error,spawn,drain=run('resource_failure',code=-15,emit=False,timeout=True,rss=65*2**30)
    assert error and any(call.kwargs.get('terminate') is True for call in drain.call_args_list)
    passed.append('owned_resource_failure_drains_only_registered_child')
    with patch.object(c.subprocess,'check_output',side_effect=['3, GPU-mock, 100000\n','12345\n']):
        assert c.gpu_capacity(3,40*2**30)==('GPU-mock',False)
    with patch.object(c.subprocess,'check_output',side_effect=['3, GPU-mock, 100000\n','']):
        assert c.gpu_capacity(3,40*2**30)==('GPU-mock',True)
    passed.append('GPU_free_memory_never_overrides_active_compute_process')
    f,o,s=fixture('failed_dependency');o.mkdir();(o/'actual_wait.json').write_text(json.dumps(dict(actual_wait=True,actual_returncode=7,owned_process_group_absent=True,status='failed')))
    try:c.dependency_state(dict(dependencies=[s['id']]),{}, {s['id']:s})
    except ValueError:passed.append('failed_terminal_dependency_rejected_before_missing_product')
    else:raise AssertionError('Failed dependency waited forever')
    f,o,s=fixture('pending_dependency')
    assert c.dependency_state(dict(dependencies=[s['id']]),{}, {s['id']:s})==[s['id']]
    passed.append('all_pending_dependency_ids_reported')
    try:c.completion_path(dict(output_directory=str(scratch)))
    except ValueError:passed.append('missing_completion_filename_rejected_statically')
    else:raise AssertionError('Missing filename accepted')
    assert c.HOST_ADMISSION_BYTES==128*2**30
    passed.append('128GiB_host_admission_constant')
    receipt=dict(schema='argmix_joint_controller_regression_v2',passed=True,recorded_utc=now(),
                 source_ref=ref(BASE/'controller_v2.py'),test_source_ref=ref(__file__),
                 plan_ref=plan_ref,freeze_ref=freeze_ref,tests=passed,
                 all_children_mocked=True,production_launched=False,heldout_payloads_or_truth_opened=False,
                 bridge_consumer_ref=ref(BASE.parent/'queue/successor_v2_support.py'),
                 strict_runtime_ref=ref(c.ROOT/'src/arggnn/benchmarking/paper_campaign/runtime.py'),
                 original_require_wait_source_ref=ref(BASE/'runtime.py'))
    out=BASE/'controller_tests_result_v2.json'
    with out.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(result_ref=ref(out),tests=len(passed),passed=True),sort_keys=True))


if __name__=='__main__':main()

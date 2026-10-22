"""Metadata-only GPU target regression and fresh European qualification fragment."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import difflib
import json
import os
import torch
from research.repairs.benchmark_followup_20261022.treeloom import fit_v2
from research.repairs.benchmark_resume_20261021.gpu.queue_adapters import european_stage as previous_stage
from . import fit_v3, gpu_binding
from .queue_adapters import common, european_stage as stage

ROOT=fit_v3.ROOT
HERE=Path(__file__).parent
PACKAGE='research.repairs.benchmark_multigpu_20261022.scientific'
OLD_PACKAGE='research.repairs.benchmark_resume_20261021.gpu.queue_adapters'


def must_fail(call):
    try:call()
    except (ValueError,OSError):return
    raise AssertionError('Invalid GPU binding must fail')


def main():
    checks=[]
    with Path(__file__).open() as lease:
        for index,uuid in gpu_binding.GPU_UUIDS.items():
            target=gpu_binding.target(index,uuid)
            environment=dict(CUDA_VISIBLE_DEVICES=uuid,ARGGNN_PHYSICAL_GPU=str(index),
                ARGGNN_READABLE_GPU_FD=str(lease.fileno()))
            with patch.dict(os.environ,environment,clear=True):
                assert gpu_binding.validate_env(target)==target
                fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False))
                with patch.object(gpu_binding.subprocess,'run',return_value=SimpleNamespace(stdout=uuid+'\n')) as query:
                    assert gpu_binding.verify_before_cuda(fake,target)==target
                    assert query.call_args.args[0][2]==str(index)
                wrong=gpu_binding.target(3 if index!=3 else 2,gpu_binding.GPU_UUIDS[3 if index!=3 else 2])
                must_fail(lambda:gpu_binding.validate_env(wrong))
                must_fail(lambda:gpu_binding.verify_before_cuda(SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True)),target))
            with patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES=uuid,PYTORCH_NVML_BASED_CUDA_CHECK='1'),clear=True):
                assert gpu_binding.validate_env(target,acquired=False)==target
                must_fail(lambda:gpu_binding.validate_env(target))
    for index,uuid in ((0,gpu_binding.GPU_UUIDS[2]),('2',gpu_binding.GPU_UUIDS[2]),(True,gpu_binding.GPU_UUIDS[2]),(2,gpu_binding.GPU_UUIDS[5])):
        must_fail(lambda:gpu_binding.target(index,uuid))
    checks.append('All authorized targets, cross-target rejection and deferred/acquired guards')
    plan=json.loads((ROOT/'configs/benchmarks/executable_workflow_20261021_gpu5_v1.json').read_text())
    state=json.loads((ROOT/'.runtime/benchmarking/workflow_20261021_gpu5_v1/state.json').read_text())
    tasks={task['id']:task for task in plan['tasks']}
    def resolve(value):
        if isinstance(value,dict) and '$task' in value:
            row=state['tasks'][value['$task']]
            assert row['status']=='succeeded'
            result=fit_v3.read_ref(row['result_ref'])['value']
            if value.get('read',False):result=fit_v3.read_ref(result)
            for key in value.get('path',[]):result=result[key]
            return result
        if isinstance(value,dict):return {key:resolve(item) for key,item in value.items()}
        if isinstance(value,list):return [resolve(item) for item in value]
        return value
    arguments=resolve(tasks['AM_treeloom_fit']['factory_arguments'])
    arguments.update(output_directory='experiments/results/benchmark_multigpu_20261022/metadata_only/native',resume_checkpoint_ref=None)
    old=fit_v2.fit_factory(**arguments)
    all_refs={}
    for index,uuid in gpu_binding.GPU_UUIDS.items():
        call=fit_v3.fit_factory(**arguments,physical_gpu=index,gpu_uuid=uuid)
        expected=copy.deepcopy(old)
        expected['module']=fit_v3.MODULE
        expected['spec'].update(schema='treeloom_upfront_RAM_resume_fit_v3',physical_gpu=index,gpu_uuid=uuid)
        refs=[fit_v3.file_ref(fit_v3.__file__) if ref['path']==fit_v3.file_ref(fit_v2.__file__)['path'] else ref for ref in expected['source_refs']]
        refs.append(fit_v3.file_ref(gpu_binding.__file__))
        expected['source_refs']=expected['spec']['source_refs']=refs
        assert call==expected
        all_refs.update({ref['path']:ref for ref in call['source_refs']})
    checks.append('Three actual American metadata factories match frozen32GiB fitter except identity and target')
    prior=json.loads((ROOT/'research/repairs/benchmark_followup_20261022/european/qualification_plan_fragment_v1.json').read_text())
    qualifications=[];calls=[];rebindings=[]
    expected_sources=common.source_refs()
    assert len(expected_sources)==22
    for prior_task,prior_call,prior_binding in zip(prior['qualification_tasks'],prior['qualification_descriptors'],prior['fit_rebindings']):
        task=copy.deepcopy(prior_task)
        task['id']=task['id'].replace('_gpu5_20261022_v1','_multigpu_20261022_v1')
        task['factory_module']=common.PACKAGE+'.european_stage'
        task['output_directory']='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_multigpu_qualifications/'+task['id']
        call=stage.qualification_factory(output_directory=task['output_directory']+'/native',**resolve(task['factory_arguments']))
        assert call['spec']['candidate_source_refs']==expected_sources
        assert call['spec']['observed_inputs']==prior_call['descriptor']['spec']['observed_inputs']
        assert call['spec']['CPU_only'] and task['resource']=='cpu'
        fit=copy.deepcopy(prior_binding['replacement_task'])
        old_id=fit['factory_arguments']['qualification_ref']['$task']
        fit['dependencies']=[task['id'] if dep==old_id else dep for dep in fit['dependencies']]
        fit['factory_arguments']['qualification_ref']={'$task':task['id']}
        fit['factory_module']=common.PACKAGE+'.european_stage'
        index=3 if fit['id'].startswith('AP__') else 5
        target=gpu_binding.target(index,gpu_binding.GPU_UUIDS[index])
        fit.update(**target,max_cuda_reserved_bytes=32*2**30,minimum_free_gpu_bytes=40*2**30)
        fit['factory_arguments'].update(target)
        original_args=resolve(tasks[fit['id']]['factory_arguments'])
        historical_ref=original_args.pop('qualification_ref')
        original_args['output_directory']=fit['output_directory']+'/native'
        original=previous_stage.build_fit_spec(**original_args)
        spec=stage.build_fit_spec(**original_args,**target)
        expected=copy.deepcopy(original);expected.update(target)
        expected['requirements']['source_refs']=[
            common.c.file_ref(ref['path'].replace('research/repairs/benchmark_resume_20261021/gpu/queue_adapters/',common.DIRECTORY+'/'))
            if ref['path'].startswith('research/repairs/benchmark_resume_20261021/gpu/queue_adapters/') else ref
            for ref in expected['requirements']['source_refs']]
        assert spec==expected
        stage._validate(spec,fit=True)
        try:stage._require_pass(historical_ref,spec)
        except common.c.ContractError:pass
        else:raise AssertionError('Old qualification must not satisfy the new exact source contract')
        qualifications.append(task);calls.append(dict(task_id=task['id'],descriptor=call))
        rebindings.append(dict(task_id=fit['id'],replacement_task=fit))
        all_refs.update({ref['path']:ref for ref in call['source_refs']})
    checks.append('Both actual European specifications preserve science, exact22source fresh qualifiers and reject historical evidence')
    assert not torch.cuda.is_initialized()
    assert not (ROOT/arguments['output_directory']).exists()
    fragment=dict(schema='multiGPU_scientific_qualification_fragment_v1',qualification_tasks=qualifications,
        qualification_descriptors=calls,fit_rebindings=rebindings,exact_candidate_source_refs=expected_sources,
        required_source_refs=list(all_refs.values()),prepared_only=True,qualifications_executed=False)
    fit_v3.write_json(HERE/'qualification_plan_fragment_v1.json',fragment)
    modules=[PACKAGE+'.fit_v3',common.PACKAGE+'.american_prediction',common.PACKAGE+'.american_workflow_v2',
        common.PACKAGE+'.empirical',common.PACKAGE+'.european_stage']
    routes={key:dict(value,module=value['module'].replace(OLD_PACKAGE,common.PACKAGE))
        for key,value in plan['readable_factory_routes'].items() if value['module'].startswith(OLD_PACKAGE)}
    fit_v3.write_json(HERE/'routes_v1.json',dict(readable_factory_modules=modules,readable_factory_routes=routes,
        required_GPU_factory_arguments=['physical_gpu','gpu_uuid'],European_GPU_fields_location='spec.scientific_spec',
        eager_GPU_fields_location='spec',logical_device='cuda:0',European_fit_cap32GiB_supported=True))
    receipt=dict(schema='multiGPU_scientific_metadata_acceptance_v1',status='passed',checks=checks,
        targets=[2,3,5],CUDA_initialized=False,models_executed=False,scientific_arrays_loaded=False,
        qualifiers_executed=False,training_executed=False,original32GiB_cap_preserved=True,
        cap_sufficiency_proven=False,checkpoint_implementation_unchanged=True,
        exact_qualification_source_count=22)
    fit_v3.write_json(HERE/'metadata_acceptance_v1.result.json',receipt)
    files=list(HERE.glob('*.py'))+list((HERE/'queue_adapters').glob('*.py'))
    all_refs.update({str(p.relative_to(ROOT)):common.c.file_ref(p) for p in files})
    fit_v3.write_json(HERE/'source_manifest_v1.json',dict(schema='multiGPU_scientific_source_manifest_v1',
        source_refs=list(all_refs.values()),acceptance_ref=common.c.file_ref(HERE/'metadata_acceptance_v1.result.json'),
        qualification_fragment_ref=common.c.file_ref(HERE/'qualification_plan_fragment_v1.json')))
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__=='__main__':main()

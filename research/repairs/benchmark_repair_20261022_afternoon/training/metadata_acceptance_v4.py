"""Metadata-only exact resource successor comparison across all20 original parents."""
from pathlib import Path
import copy
import json
import torch
from research.repairs.benchmark_multigpu_20261022.scientific import fit_v3
from . import fit_v4


def main():
    root=fit_v4.ROOT
    here=Path(__file__).parent
    assert fit_v4.file_ref(fit_v3.__file__)['sha256']=='9891903b82aada6f110451502166320c007fefb849f015b33454f2157d4f11ea'
    expected=Path(fit_v3.__file__).read_text()
    changes=[('from . import gpu_binding','from research.repairs.benchmark_multigpu_20261022.scientific import gpu_binding',1),
        ('research.repairs.benchmark_multigpu_20261022.scientific.fit_v3','research.repairs.benchmark_repair_20261022_afternoon.training.fit_v4',1),
        ('treeloom_upfront_RAM_resume_fit_v3','treeloom_upfront_RAM_resume_fit_v4',2),
        ('32 GiB','64 GiB',2),('40 GiB','72 GiB',1),
        ('max_cuda_reserved_bytes=32*GIB','max_cuda_reserved_bytes=64*GIB',1),
        ("spec['max_cuda_reserved_bytes'] == 32*GIB","spec['max_cuda_reserved_bytes'] == 64*GIB",1)]
    for old,new,count in changes:
        assert expected.count(old)==count
        expected=expected.replace(old,new)
    assert Path(fit_v4.__file__).read_text()==expected
    assert fit_v3.gpu_binding is fit_v4.gpu_binding
    assert fit_v3.checkpoints_io is fit_v4.checkpoints_io
    plan=json.loads((root/'configs/benchmarks/executable_workflow_20261022_multigpu_v2.json').read_text())
    task=next(task for task in plan['tasks'] if task['id']=='AM_treeloom_fit')
    state=json.loads((root/'.runtime/benchmarking/workflow_20261022_multigpu_v2/state.json').read_text())
    def resolve(value):
        if isinstance(value,dict) and '$task' in value:
            row=state['tasks'][value['$task']]
            assert row['status']=='succeeded'
            result=fit_v4.read_ref(row['result_ref'])['value']
            if value.get('read',False):result=fit_v4.read_ref(result)
            for key in value.get('path',[]):result=result[key]
            return result
        if isinstance(value,dict):return {key:resolve(item) for key,item in value.items()}
        if isinstance(value,list):return [resolve(item) for item in value]
        return value
    arguments=resolve(task['factory_arguments'])
    arguments.update(output_directory='experiments/results/benchmark_repair_20261022_afternoon/metadata_only/native',resume_checkpoint_ref=None)
    assert arguments['physical_gpu']==2 and arguments['gpu_uuid']=='GPU-976a5cd3-df12-2535-fa51-e3ea063688ba'
    assert len(arguments['preparation_refs'])==len(arguments['generation_refs'])==20
    old_call=fit_v3.fit_factory(**arguments)
    new_call=fit_v4.fit_factory(**arguments)
    expected=copy.deepcopy(old_call)
    expected['module']=fit_v4.MODULE
    expected['spec'].update(schema='treeloom_upfront_RAM_resume_fit_v4',max_cuda_reserved_bytes=64*2**30)
    prior_ref=fit_v4.file_ref(fit_v3.__file__);new_ref=fit_v4.file_ref(fit_v4.__file__)
    references=[new_ref if ref==prior_ref else ref for ref in expected['source_refs']]
    expected['source_refs']=expected['spec']['source_refs']=references
    assert expected==new_call
    assert not torch.cuda.is_initialized()
    assert not (root/arguments['output_directory']).exists()
    patch=json.loads((here/'task_resource_patch_v4.json').read_text())
    assert patch['max_cuda_reserved_bytes']==new_call['spec']['max_cuda_reserved_bytes']==64*2**30
    assert patch['minimum_free_gpu_bytes']==72*2**30
    assert patch['max_rss_bytes']==new_call['spec']['max_rss_bytes']==512*2**30
    assert new_call['spec']['resume_checkpoint_ref'] is None
    receipt=dict(schema='treeloom_resource_only_metadata_acceptance_v4',status='passed',
        parents=20,physical_gpu=2,gpu_uuid=arguments['gpu_uuid'],
        exact_source_change_limited_to_resource_identity_and_same_helper_import=True,
        actual_metadata_factory_outputs_identical_except_cap_and_identity=True,
        original_fitter_and_24_source_references_verified=True,checkpoint_and_GPU_guard_modules_unchanged=True,
        original_600_update_training_body_unchanged=True,max_cuda_reserved_bytes=64*2**30,
        minimum_free_gpu_bytes=72*2**30,max_rss_bytes=512*2**30,
        CUDA_initialized=False,training_executed=False,scientific_arrays_loaded=False,truth_payloads_loaded=False,
        output_scope_created=False,resume_checkpoint_ref=None,cap_is_provisional=True,cap_sufficiency_proven=False,
        fit_source_ref=new_ref,source_ref_count=len(references),failure_receipt_ref=fit_v4.file_ref(here/'failed_attempt_v3.receipt.json'))
    fit_v4.write_json(here/'metadata_acceptance_v4.result.json',receipt)
    fit_v4.write_json(here/'source_manifest_v4.json',dict(schema='treeloom_resource_only_source_manifest_v4',
        source_refs=new_call['source_refs'],acceptance_ref=fit_v4.file_ref(here/'metadata_acceptance_v4.result.json'),
        test_source_ref=fit_v4.file_ref(__file__),task_resource_patch_ref=fit_v4.file_ref(here/'task_resource_patch_v4.json'),
        failure_receipt_ref=fit_v4.file_ref(here/'failed_attempt_v3.receipt.json')))
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__=='__main__':main()

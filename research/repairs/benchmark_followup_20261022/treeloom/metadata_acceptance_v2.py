"""Metadata-only comparison of the frozen fitter and resource-only successor."""
from pathlib import Path
import difflib
import json
import torch
from research.repairs.benchmark_resume_20261021.training import fit_v1
from . import fit_v2


def main():
    root=fit_v2.ROOT
    base=Path(__file__).parent
    old_source=Path(fit_v1.__file__).read_text()
    expected=old_source.replace(
        'Static original numerical modules remain unchanged. No gates or child launchers.',
        'Static original numerical modules remain unchanged. No gates or child launchers.\n'
        'The 32 GiB allocator cap is provisional until a genuine full backward succeeds;\n'
        'the queue must require at least 40 GiB free on physical GPU5 before admission.')
    expected=expected.replace('from . import checkpoint_v1 as checkpoints_io',
        'from research.repairs.benchmark_resume_20261021.training import checkpoint_v1 as checkpoints_io')
    expected=expected.replace(f"MODULE='{fit_v1.MODULE}'",f"MODULE='{fit_v2.MODULE}'")
    expected=expected.replace('treeloom_upfront_RAM_resume_fit_v1','treeloom_upfront_RAM_resume_fit_v2')
    expected=expected.replace('max_cuda_reserved_bytes=16*GIB','max_cuda_reserved_bytes=32*GIB')
    expected=expected.replace("spec['max_cuda_reserved_bytes'] == 16*GIB","spec['max_cuda_reserved_bytes'] == 32*GIB")
    expected=expected.replace('Queue and fit must share the 16 GiB allocator cap',
        'Queue and fit must share the provisional 32 GiB allocator cap')
    assert Path(fit_v2.__file__).read_text() == expected
    assert fit_v1.checkpoints_io is fit_v2.checkpoints_io
    assert fit_v1.file_ref(fit_v1.__file__)['sha256'] == '5491025abf491f5706dcbc1321ec444ee72d7b0c32feb1298607eea5124ca735'
    assert fit_v2.file_ref(fit_v2.checkpoints_io.__file__)['sha256'] == '68aa4ada6c34b62e167a48ff9b83e26eab275162d1364a779f1905f080a9720f'
    plan=json.loads((root/'configs/benchmarks/executable_workflow_20261021_gpu5_v1.json').read_text())
    task=next(row for row in plan['tasks'] if row['id'] == 'AM_treeloom_fit')
    state=json.loads((root/'.runtime/benchmarking/workflow_20261021_gpu5_v1/state.json').read_text())
    def resolve(value):
        if isinstance(value,dict) and '$task' in value:
            row=state['tasks'][value['$task']]
            assert row['status'] == 'succeeded'
            result=fit_v2.read_ref(row['result_ref'])['value']
            if value.get('read',False):result=fit_v2.read_ref(result)
            for key in value.get('path',[]):result=result[key]
            return result
        if isinstance(value,dict):return {key:resolve(item) for key,item in value.items()}
        if isinstance(value,list):return [resolve(item) for item in value]
        return value
    arguments=resolve(task['factory_arguments'])
    arguments['output_directory']='experiments/results/benchmark_followup_20261022/treeloom_metadata_acceptance_v2/native'
    arguments['resume_checkpoint_ref']=None
    assert not (root/arguments['output_directory']).exists()
    previous=fit_v1.fit_factory(**arguments)
    successor=fit_v2.fit_factory(**arguments)
    old_ref=fit_v2.file_ref(fit_v1.__file__)
    new_ref=fit_v2.file_ref(fit_v2.__file__)
    previous['module']=fit_v2.MODULE
    previous['spec']['schema']='treeloom_upfront_RAM_resume_fit_v2'
    previous['spec']['max_cuda_reserved_bytes']=32*fit_v2.GIB
    previous['spec']['source_refs']=[new_ref if ref == old_ref else ref for ref in previous['spec']['source_refs']]
    previous['source_refs']=previous['spec']['source_refs']
    assert previous == successor
    assert not torch.cuda.is_initialized()
    assert not (root/arguments['output_directory']).exists()
    patch=json.loads((base/'task_resource_patch_v2.json').read_text())
    assert patch['max_cuda_reserved_bytes'] == successor['spec']['max_cuda_reserved_bytes'] == 32*fit_v2.GIB
    assert patch['minimum_free_gpu_bytes'] == 40*fit_v2.GIB
    assert patch['max_rss_bytes'] == successor['spec']['max_rss_bytes'] == 512*fit_v2.GIB
    receipt=dict(schema='treeloom_resource_only_metadata_acceptance_v2',status='passed',
        exact_source_diff_limited_to_resource_and_identity=True,
        metadata_factories_match_except_resource_and_identity=True,
        original_fitter_unchanged=True,original_checkpoint_implementation_unchanged=True,
        original_numerical_and_RAM_pool_source_hashes_verified=True,
        parents=len(arguments['preparation_refs']),CUDA_initialized=False,
        scientific_data_arrays_loaded=False,truth_payloads_loaded=False,training_executed=False,
        output_scope_created=False,cap_is_provisional=True,cap_sufficiency_proven=False,
        max_cuda_reserved_bytes=32*fit_v2.GIB,minimum_free_gpu_bytes=40*fit_v2.GIB,
        max_rss_bytes=512*fit_v2.GIB,fit_source_ref=new_ref,
        checkpoint_source_ref=fit_v2.file_ref(fit_v2.checkpoints_io.__file__),
        source_ref_count=len(successor['source_refs']))
    fit_v2.write_json(base/'metadata_acceptance_v2.result.json',receipt)
    fit_v2.write_json(base/'source_manifest_v2.json',dict(schema='treeloom_resource_only_source_manifest_v2',
        source_refs=successor['source_refs'],acceptance_ref=fit_v2.file_ref(base/'metadata_acceptance_v2.result.json')))
    diff=''.join(difflib.unified_diff(old_source.splitlines(keepends=True),expected.splitlines(keepends=True),
        fromfile=str(Path(fit_v1.__file__).relative_to(root)),tofile=str(Path(fit_v2.__file__).relative_to(root))))
    with (base/'fit_v2.diff').open('x') as stream:stream.write(diff)
    print(json.dumps(receipt,sort_keys=True))


if __name__ == '__main__':
    main()

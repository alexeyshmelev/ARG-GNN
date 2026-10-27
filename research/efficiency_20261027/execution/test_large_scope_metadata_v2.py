"""CPU-only actual prefix scope serialization and metadata boundary contracts."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import os
import time
import numpy as np
import torch
from research.efficiency_20261027.execution import prepare_large_slice_v2 as repaired
from research.efficiency_20261027.execution.compare_large_blocks_deferred_v2 import source_names
from research.efficiency_20261027.root.compare_host_parallelism import (
    ROOT, CHECKPOINT, GIB, need, ref, rss, write)
from research.benchmark_rebuild_20261024.american.treeloom_speed import (
    runtime_v2 as fresh, training_v2 as validation)


def main():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be disabled for this test')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    output=ROOT/'research/efficiency_20261027/execution/large_scope_metadata_v2_cpu'
    output.mkdir(exist_ok=False)
    original_array=np.array([1, 2], dtype=np.int64)
    original_tensor=torch.tensor([1, 2])
    raw=dict(nested=[dict(integer=np.int64(7), floating=np.float32(1.25), boolean=np.bool_(True))],
             tuple_value=(np.uint64(9),), array=original_array, tensor=original_tensor)
    bundle,metadata,labels,frozen=object(),object(),object(),object()
    with patch.object(repaired, '_prepare_large_prefix', return_value=(bundle,metadata,labels,frozen,raw)):
        result=repaired.prepare_large_prefix(None,None,None,None)
    need(all(a is b for a,b in zip(result[:4], (bundle,metadata,labels,frozen))),
         'All numerical/metadata/model objects must preserve identity')
    normalized=result[4]
    need(normalized is not raw and normalized['nested'] is not raw['nested'], 'Scope receives a separate copy')
    need(type(raw['nested'][0]['integer']) is np.int64, 'Original receipt is not mutated')
    need(normalized['array'] is original_array and normalized['tensor'] is original_tensor,
         'Arrays and tensors are not converted or copied')
    need(type(normalized['nested'][0]['integer']) is int
         and type(normalized['nested'][0]['floating']) is float
         and type(normalized['nested'][0]['boolean']) is bool, 'Only scalar representation changed')
    try:
        json.dumps(normalized, allow_nan=False)
    except TypeError:
        pass
    else:
        raise AssertionError('Unsupported arrays/tensors must remain rejected by JSON')
    for bad in (np.float32(float('nan')), np.float64(float('inf'))):
        try:
            json.dumps(repaired.plain_scope_metadata(bad), allow_nan=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite scalar must fail strict JSON')
    def check(stage):
        need(rss() <= 24*GIB, 'CPU memory cap exceeded at '+stage)
        need(not torch.cuda.is_initialized(), 'No CUDA initialization allowed')
    sources=[ref(name) for name in source_names()]
    sources.append(ref(Path(__file__).resolve()))
    saved=torch.load(ROOT/CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    need(ref(CHECKPOINT)['sha256']=='12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24',
         'Exact checkpoint241')
    registry=fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref'])
    runtime=fresh.FreshTreeLoom(registry)
    validation.validate_saved_state(saved,runtime.build_model(),registry,expected_steps=241)
    started=time.perf_counter()
    bundle,metadata,labels,frozen,scope=repaired.prepare_large_prefix(
        SimpleNamespace(slice_holds=1153),runtime,saved,check)
    bundle.validate()
    write(output/'scope.json',scope)
    reread=json.loads((output/'scope.json').read_text())
    need(json.loads(json.dumps(scope, allow_nan=False)) == reread, 'Actual complete scope JSON roundtrip')
    need(len(labels)==scope['supervised_original_focals']>0 and bundle.native_holds==1153,
         'Actual observed prefix and original TRAIN targets')
    need([ref(item['path']) for item in sources] == sources, 'Every source remains frozen')
    check('actual_scope_serialization_complete')
    receipt=dict(status='passed',source_refs=sources,scope_ref=ref(output/'scope.json'),
        actual_prefix_holds=bundle.native_holds,actual_prefix_nodes=scope['native_nodes'],
        actual_sites=scope['sites'],original_TRAIN_focals=len(labels),CPU_threads=torch.get_num_threads(),
        cuda_initialized=torch.cuda.is_initialized(),max_RSS_bytes=24*GIB,current_RSS_bytes=rss(),
        elapsed_seconds=time.perf_counter()-started,strict_JSON_roundtrip=True,
        numerical_return_object_identity_preserved=True,original_scope_not_mutated=True,
        arrays_and_tensors_not_converted=True,nonfinite_values_rejected=True,
        GPU_work_executed=False,production_modified=False)
    write(output/'completed.json',receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False))


if __name__=='__main__':
    main()

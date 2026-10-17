"""Reproduce numeric blocker through the documented complete-bank API."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,importlib.util,traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('src').resolve()))
from arggnn import american_c3_gnn_observations_isolated_v2 as original
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
OWN=Path('research/reviews/C3_CPU_prepared_features_independent_20261017_v1')
SOURCE=Path('research/reviews/C3_CPU_prepared_features_candidate_20261017_v1/prepared_cpu_features_v2.py')
R={'schema':'C3_prepared_CPU_independent_full_bank_counterexample_v1','scientific_data_reads':0,'truth_reads':0,'guard_imports':0,'model_forwards':0,'operational_actions':0,'CPU_only':True}
try:
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='9daf798024a84aca7abb8d541c5abaa1772d868acfc53fd12cbfe67accd3f48b'
 spec=importlib.util.spec_from_file_location('independent_pinned_full_bank',SOURCE);c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
 torch.set_num_threads(1)
 tree=NativeObservedTreeV1(np.asarray([3,3,3,-1],dtype=np.int64),np.asarray([0,0,0,1000],dtype=np.float32),np.arange(3,dtype=np.int64),np.arange(3,dtype=np.int64))
 g=np.tile(np.asarray([0,1,2],dtype=np.int8),(4096,1));k=np.full(4096,3,dtype=np.int64)
 args=([tree],np.asarray([[0,4096]],dtype=np.float32),g,k,np.arange(4096,dtype=np.float32),np.asarray([0,1,-1],dtype=np.int64),np.zeros(3,dtype=np.float32))
 opts=dict(num_classes=3,age_split=350.,query_columns=np.asarray([2],dtype=np.int64))
 a,ar=original.encode_block(*args,**opts)
 layouts=c.ImmutableTreeLayoutCacheV1(max_bytes=2**20,max_entries=2)
 cache=c.ImmutableBlockFeatureCacheV1(layout_cache=layouts,max_tensor_bytes=2**21,max_entries=2)
 b,br=cache.get_or_prepare(('review','observation','plan',0),*args,tree_keys=[('review','tree',0)],site_tile=1,**opts)
 aa=a.branch_support_true;bb=b.branch_support_true
 R.update(status='BLOCKED',original=aa.tolist(),candidate=bb.tolist(),max_abs_error=float((aa-bb).abs().max()),declared_allclose=bool(torch.allclose(aa,bb,rtol=2e-6,atol=2e-6)),rtol=2e-6,atol=2e-6,original_receipt=ar,candidate_receipt=br,bank_tensor_bytes=cache.bytes,layout_bytes=layouts.bytes,cache_builds=cache.builds,fixture_input_bytes=sum(x.nbytes for x in args[1:])+sum(getattr(tree,n).nbytes for n in ('parent','node_time','sample_node','sample_id'))+opts['query_columns'].nbytes,all_input_float_arrays_FP32=all(x.dtype==np.float32 for x in (tree.node_time,args[1],args[4],args[6])))
 assert not R['declared_allclose'] and R['fixture_input_bytes']<16*2**20
except Exception:R.update(status='HARNESS_OR_COMPONENT_FAILURE',fatal_error=traceback.format_exc());print(R['fatal_error'])
(OWN/'full_bank_counterexample.v1.json').write_text(json.dumps(R,indent=2,sort_keys=True)+'\n')
print(json.dumps(R,indent=2))

"""Narrow sampled-ancestor and compensation accounting check for sealed v3."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,importlib.util,ast,traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('src').resolve()))
from arggnn import american_c3_gnn_observations_isolated_v2 as original
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1,branch_concordance_support_v1
D=Path('research/reviews/C3_CPU_prepared_features_candidate_20261017_v1');OWN=Path('research/reviews/C3_CPU_prepared_features_independent_20261017_v3')
R={'schema':'C3_prepared_CPU_v3_sampled_ancestor_and_memory_checks','checks':[],'negative_cases':[],'CPU_only':True,'actual_data_truth_guard_model_optimizer_GPU_probe_Git_actions':0}
def check(name,ok,**detail):
 R['checks'].append(dict(name=name,passed=bool(ok),**detail))
 if not ok:raise AssertionError(name)
def rejected(name,call):
 try:call()
 except ValueError as e:R['negative_cases'].append(dict(name=name,message=str(e)));return str(e)
 raise AssertionError('Accepted invalid '+name)
def dump(node):return ast.dump(node,include_attributes=False)
def run():
 old=(D/'prepared_cpu_features_v2.py').read_bytes();new=(D/'prepared_cpu_features_v3.py').read_bytes()
 check('sealed_v2_preserved',hashlib.sha256(old).hexdigest()=='9daf798024a84aca7abb8d541c5abaa1772d868acfc53fd12cbfe67accd3f48b')
 check('sealed_v3_pinned',hashlib.sha256(new).hexdigest()=='9a1614a89fd4eea54dc8c0213a0ae4aa1a6e585655e80c21ea73f4e93e220057')
 a={n.name:n for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};b={n.name:n for n in ast.parse(new).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 same=[n for n in a if dump(a[n])==dump(b[n])];changed=[n for n in a if n not in same]
 check('exact12other_function_class_ASTs_identical',len(same)==12 and changed==['branch_support_batched_v1'] and set(b)-set(a)=={'_compensated_add_'},same=same,changed=changed)
 check('new_FP32_only_compensation_dtype',b'correction = torch.zeros((n, 3), dtype=torch.float32)'in new and b'+ 96 * n'in new and b'float64'not in new and b'.double('not in new)
 spec=importlib.util.spec_from_file_location('independent_v3_sampled_ancestor',D/'prepared_cpu_features_v3.py');c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
 torch.set_num_threads(1)
 tree=NativeObservedTreeV1(np.asarray([2,2,3,-1],dtype=np.int64),np.asarray([0,0,100,200],dtype=np.float32),np.asarray([2,0,3,1],dtype=np.int64),np.asarray([71,11,99,23],dtype=np.int64))
 ids=np.asarray([23,99,11,71],dtype=np.int64);g=np.asarray([[0,1,2,-1],[1,0,1,0],[-1,-1,-1,-1],[0,0,0,0],[3,1,2,0]],dtype=np.int8);k=np.asarray([3,2,1,1,4],dtype=np.int64)
 layouts=c.ImmutableTreeLayoutCacheV1(max_bytes=2**20,max_entries=4)
 record=layouts.get(tree,('internal_samples',))
 check('sampled_internal_and_root_descendant_counts_include_own_observation',np.array_equal(record['layout'][7],np.asarray([1,1,3,4],dtype=np.int64)))
 n,h,tile=len(tree.parent),len(tree.sample_node),2
 allowance=512*(n+h+1)*tile+96*n;oldallowance=512*(n+h+1)*tile+64*n
 calls=[];base=c._compensated_add_
 def checked(output,correction,branches,channel,increment):
  check('compensation_call_FP32_'+str(len(calls)),output.dtype==correction.dtype==increment.dtype==torch.float32 and branches.dtype==torch.long)
  calls.append(dict(correction_bytes=correction.numel()*correction.element_size(),correction_shape=list(correction.shape)))
  return base(output,correction,branches,channel,increment)
 c._compensated_add_=checked
 expected=branch_concordance_support_v1(tree,g,k,ids)
 actual=c.branch_support_batched_v1(tree,g,k,ids,layout_cache=layouts,tree_key=('internal_samples',),site_tile=tile,max_working_bytes=allowance)
 check('sampled_internal_and_root_support_matches_original_with_permuted_identity_axis',actual.dtype==np.float32 and np.allclose(expected,actual,atol=2e-6,rtol=2e-6),max_abs_error=float(np.abs(expected-actual).max()),original=expected.tolist(),candidate=actual.tolist())
 check('compensation12bytes_per_vertex_and_allowance_add32bytes_per_vertex',calls and all(x['correction_bytes']==12*n and x['correction_shape']==[n,3]for x in calls)and allowance-oldallowance==32*n,correction_bytes=12*n,allowance=allowance,old_allowance=oldallowance,extra_allowance=32*n)
 rejected('one_byte_below_complete_v3_working_allowance',lambda:c.branch_support_batched_v1(tree,g,k,ids,layout_cache=layouts,tree_key=('internal_samples',),site_tile=tile,max_working_bytes=allowance-1))
 c._compensated_add_=base
 # The unchanged original C3 block encoder itself requires sampled leaves.
 bt=NativeObservedTreeV1(tree.parent,tree.node_time,tree.sample_node,np.arange(4,dtype=np.int64));ages=tree.node_time[tree.sample_node]
 args=([bt],np.asarray([[0,5]],dtype=np.float32),g,k,np.arange(5,dtype=np.float32),np.asarray([0,1,2,-1],dtype=np.int64),ages)
 left=rejected('original_C3_block_preserves_existing_sampled_leaf_requirement',lambda:original.encode_block(*args,num_classes=3))
 right=rejected('v3_C3_block_matches_original_sampled_leaf_requirement',lambda:c.encode_block_prepared_v1(*args,tree_keys=[('internal_block',)],layout_cache=layouts,num_classes=3))
 check('same_original_block_rejection_for_sampled_ancestors',left==right=='All active leaves must be original observed samples')
 R.update(status='SOURCE_PASS',fixture_input_bytes=sum(getattr(tree,x).nbytes for x in ('parent','node_time','sample_node','sample_id'))+ids.nbytes+g.nbytes+k.nbytes+sum(x.nbytes for x in args[1:]),atol=2e-6,rtol=2e-6,scope_note='Direct original branch helper supports sampled internal nodes/root; v3 matches it. Original complete C3 encode_block retains its existing sampled-leaf-only constraint and v3 rejects the same case. This inherited constraint is not resolved or adopted here.')
try:run()
except Exception:R.update(status='HARNESS_OR_COMPONENT_FAILURE',fatal_error=traceback.format_exc());print(R['fatal_error'])
(OWN/'sampled_ancestors_and_allowance.v3.json').write_text(json.dumps(R,indent=2,sort_keys=True)+'\n')
print(json.dumps(R,indent=2))

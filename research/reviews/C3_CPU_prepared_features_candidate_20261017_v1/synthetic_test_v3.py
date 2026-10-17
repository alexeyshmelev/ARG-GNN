"""Only self-created CPU fixtures. No original observations/truth or guards."""
from pathlib import Path
from dataclasses import fields
import json
import time
import numpy as np
import torch
import torch_geometric
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1, branch_concordance_support_v1
from arggnn import american_c3_gnn_observations_isolated_v2 as original
from prepared_cpu_features_v3 import (
    ImmutableTreeLayoutCacheV1, ImmutableBlockFeatureCacheV1,
    branch_support_batched_v1, encode_block_prepared_v1)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()

def tree(h, variant, ages):
    parent = [-1]*h
    times = ages.tolist()
    work = list(np.roll(np.arange(h), variant))
    if variant == 3:
        return NativeObservedTreeV1(np.array(parent), np.array(times,dtype=np.float32),
                                   np.arange(h), np.arange(h))
    if variant == 1:
        node = len(parent)
        parent.append(-1)
        times.append(times[work[0]]+50)
        parent[work[0]] = node
        work[0] = node
    width = 3 if variant == 2 else 2
    while len(work) > 1:
        following = []
        for start in range(0, len(work), width):
            children = work[start:start+width]
            if len(children) == 1:
                following.extend(children)
                continue
            node = len(parent)
            parent.append(-1)
            times.append(max(times[c] for c in children)+100)
            for child in children:
                parent[child] = node
            following.append(node)
        work = following
    return NativeObservedTreeV1(np.array(parent), np.array(times,dtype=np.float32),
                               np.arange(h), np.arange(h))

h, sites = 32, 129
ages = (np.arange(h)%2*400).astype(np.float32)
trees = [tree(h, i, ages) for i in range(4)]
rng = np.random.default_rng(20261017)
k = np.tile(np.array([2,3,4],dtype=np.int8), sites//3+1)[:sites*1]
g = np.stack([rng.integers(0,int(a),h,dtype=np.int8) for a in k])
g[0] = 0
g[1] = -1
g[2] = 0
g[2,-1] = 1
g[3,::3] = -1
classes = np.full(h,-1,dtype=np.int64)
classes[:18] = np.arange(18)%3
queries = np.flatnonzero(classes < 0)
layout = ImmutableTreeLayoutCacheV1(max_bytes=2**20,max_entries=8)
support_checks = []
for index, t in enumerate(trees[:3]):
    wanted = branch_concordance_support_v1(t,g,k,np.arange(h))
    for tile in [1,17,129]:
        got = branch_support_batched_v1(t,g,k,np.arange(h),layout_cache=layout,
            tree_key=("synthetic-fixture",index),site_tile=tile,max_working_bytes=8*2**20)
        error = float(np.max(np.abs(got-wanted)))
        assert got.dtype == np.float32
        assert np.allclose(got,wanted,rtol=2e-6,atol=2e-6), (index,tile,error)
        support_checks.append(dict(tree=index,site_tile=tile,max_abs_error=error))

intervals = np.array([[i*1000,(i+1)*1000] for i in range(4)],dtype=np.float32)
positions = np.concatenate([np.linspace(i*1000+1,(i+1)*1000-1,sites,dtype=np.float32) for i in range(4)])
genotypes = np.concatenate([g]*4)
alleles = np.concatenate([k]*4)
keys = [("synthetic-fixture",i) for i in range(4)]
cache = ImmutableBlockFeatureCacheV1(layout_cache=layout,max_tensor_bytes=2**20,max_entries=2)
arguments = (trees,intervals,genotypes,alleles,positions,classes,ages)
options = dict(tree_keys=keys,num_classes=3,age_split=350.,query_columns=queries,
               site_tile=17,max_working_bytes=8*2**20)
reference, reference_receipt = original.encode_block(*arguments,num_classes=3,
    age_split=350.,query_columns=queries)
prepared, receipt = cache.get_or_prepare(("synthetic-observed-sha","plan-sha",0),*arguments,**options)
assert receipt == reference_receipt
checks = []
for field in fields(reference):
    a,b = getattr(reference,field.name),getattr(prepared,field.name)
    pairs = list(zip(a,b)) if isinstance(a,tuple) else [(a,b)]
    assert len(a)==len(b) if isinstance(a,tuple) else a.shape == b.shape
    error = 0.
    for left,right in pairs:
        assert left.dtype == right.dtype and left.shape == right.shape
        if left.dtype.is_floating_point:
            assert left.dtype == torch.float32
            assert torch.allclose(left,right,rtol=2e-6,atol=2e-6), field.name
            if left.numel():error = max(error,float((left-right).abs().max()))
        else:
            assert torch.equal(left,right),field.name
    checks.append(dict(field=field.name,max_abs_error=error))
reused, reused_receipt = cache.get_or_prepare(("synthetic-observed-sha","plan-sha",0),*arguments,**options)
assert reused is prepared and reused_receipt == receipt and cache.builds == 1 and cache.hits == 1
assert layout.builds == 4
negative = []
changed = g.copy()
changed[4,0] = (changed[4,0]+1)%int(k[4])
changed_args = (trees,intervals,np.concatenate([changed,g,g,g]),alleles,positions,classes,ages)
try:
    cache.get_or_prepare(("synthetic-observed-sha","plan-sha",0),*changed_args,**options)
    raise AssertionError("Input drift accepted")
except ValueError:negative.append("same_key_input_drift_rejected")
prepared.parent[0] = prepared.parent[0]
try:
    # Actual value drift, including NumPy alias writes that do not bump _version.
    prepared.genotype.numpy()[5,0] = (int(prepared.genotype[5,0])+1)%int(alleles[5])
    cache.get_or_prepare(("synthetic-observed-sha","plan-sha",0),*arguments,**options)
    raise AssertionError("Feature drift accepted")
except ValueError:negative.append("cached_numpy_alias_value_drift_rejected")
changed_parent = trees[0].parent.copy()
changed_parent[0] = -1
changed_tree = NativeObservedTreeV1(changed_parent,trees[0].node_time,trees[0].sample_node,trees[0].sample_id)
try:
    layout.get(changed_tree,keys[0])
    raise AssertionError("Topology drift accepted")
except ValueError:negative.append("same_key_topology_drift_rejected")
try:
    branch_support_batched_v1(trees[0],g,k,np.arange(h),layout_cache=layout,
        tree_key=keys[0],site_tile=32,max_working_bytes=1)
    raise AssertionError("Scratch cap bypassed")
except ValueError:negative.append("prospective_tile_cap_rejected")
try:
    tiny = ImmutableBlockFeatureCacheV1(layout_cache=layout,max_tensor_bytes=1,max_entries=1)
    tiny.get_or_prepare(("tiny",),*arguments,**options)
    raise AssertionError("Bank cap bypassed")
except ValueError:negative.append("whole_bank_cap_rejected")

# Additional semantic cases for empty/degenerate sites and permuted joins.
for name, values, counts, ids in [
    ("no_sites", g[:0], k[:0], np.arange(h)),
    ("all_missing", np.full_like(g, -1), k, np.arange(h)),
    ("monomorphic", np.zeros_like(g), k, np.arange(h)),
    ("permuted_sample_axis", g[:, ::-1].copy(), k, np.arange(h)[::-1].copy()),
]:
    wanted = branch_concordance_support_v1(trees[0], values, counts, ids)
    got = branch_support_batched_v1(trees[0], values, counts, ids,
        layout_cache=layout, tree_key=keys[0], site_tile=17,
        max_working_bytes=8*2**20)
    assert np.allclose(got, wanted, rtol=2e-6, atol=2e-6), name
    support_checks.append(dict(tree=0, case=name, site_tile=17,
        max_abs_error=float(np.max(np.abs(got-wanted)))))
record = layout.get(trees[0], keys[0])
edge = record["propagation"][0][2]
old = int(edge[0,0])
edge[0,0] = old+1
try:
    layout.get(trees[0], keys[0])
    raise AssertionError("Propagation-index mutation accepted")
except ValueError:negative.append("cached_propagation_index_drift_rejected")
edge[0,0] = old
assert layout.get(trees[0], keys[0]) is record

# Exact independent full-bank counterexample input construction, preserved.
counter_tree=NativeObservedTreeV1(np.asarray([3,3,3,-1],dtype=np.int64),np.asarray([0,0,0,1000],dtype=np.float32),np.arange(3,dtype=np.int64),np.arange(3,dtype=np.int64))
counter_g=np.tile(np.asarray([0,1,2],dtype=np.int8),(4096,1))
counter_k=np.full(4096,3,dtype=np.int64)
counter_args=([counter_tree],np.asarray([[0,4096]],dtype=np.float32),counter_g,counter_k,np.arange(4096,dtype=np.float32),np.asarray([0,1,-1],dtype=np.int64),np.zeros(3,dtype=np.float32))
counter_options=dict(num_classes=3,age_split=350.,query_columns=np.asarray([2],dtype=np.int64))
wanted,wanted_receipt=original.encode_block(*counter_args,**counter_options)

def equal_bank(name,left,right):
    maximum=0.
    count=0
    for field in fields(left):
        a,b=getattr(left,field.name),getattr(right,field.name)
        assert isinstance(a,tuple)==isinstance(b,tuple)
        if isinstance(a,tuple):
            assert len(a)==len(b)
            pairs=zip(a,b)
        else:
            pairs=[(a,b)]
        for x,y in pairs:
            assert x.shape==y.shape and x.dtype==y.dtype
            if x.is_floating_point():
                assert x.dtype==torch.float32
                assert torch.allclose(x,y,rtol=2e-6,atol=2e-6),(name,field.name)
                if x.numel():maximum=max(maximum,float((x-y).abs().max()))
            else:assert torch.equal(x,y),(name,field.name)
        count+=1
    assert count==21
    return dict(case=name,complete_bank_fields=count,max_abs_error=maximum)

repaired=[]
for tile in [1,17,32,256]:
    layouts=ImmutableTreeLayoutCacheV1(max_bytes=2**20,max_entries=2)
    features=ImmutableBlockFeatureCacheV1(layout_cache=layouts,max_tensor_bytes=2**21,max_entries=2)
    candidate,candidate_receipt=features.get_or_prepare(('review','observation','plan',0),
        *counter_args,tree_keys=[('review','tree',0)],site_tile=tile,**counter_options)
    assert candidate_receipt==wanted_receipt
    check=equal_bank('independent_4096_triallelic_tile_'+str(tile),wanted,candidate)
    check.update(site_tile=tile,bank_tensor_bytes=features.bytes,
        original_mean_discrepancy=float(wanted.branch_support_true[0,2]),
        candidate_mean_discrepancy=float(candidate.branch_support_true[0,2]))
    repaired.append(check)

# Exact generator and1600sample fixture copied from the independent source.
R=dict(fixture_input_bytes=0)
def ownarray(x,dtype):
 a=np.asarray(x,dtype=dtype);R['fixture_input_bytes']+=a.nbytes;return a

def make_tree(style,ages,seed=0):
 h=len(ages);p=[-1]*h;clock=list(ages);active=list(range(h))
 if style=='isolated':pass
 elif style=='star':
  root=len(p);p.append(-1);clock.append(max(clock)+1000)
  for a in active:p[a]=root
 elif style=='unary':
  root=len(p);p.append(-1);clock.append(max(clock)+1000)
  for a in active:p[a]=root
  upper=len(p);p.append(-1);clock.append(max(clock)+1000);p[root]=upper
 else:
  while len(active)>1:
   size=min(3 if style=='nonbinary' else 2,len(active));take=active[:size];active=active[size:]
   root=len(p);p.append(-1);clock.append(max(clock[a] for a in take)+1000)
   for a in take:p[a]=root
   active.append(root)
 p=np.asarray(p,dtype=np.int64);clock=np.asarray(clock,dtype=np.float32)
 perm=np.random.default_rng(seed).permutation(len(p));inv=np.argsort(perm)
 return NativeObservedTreeV1(ownarray(np.where(p[perm]<0,-1,inv[np.maximum(p[perm],0)]),np.int64),ownarray(clock[perm],np.float32),ownarray(inv[:h],np.int64),ownarray(np.arange(h),np.int64))
ages=ownarray(np.zeros(1600,dtype=np.float32),np.float32)
large_tree=make_tree('binary',ages,75)
rng=np.random.default_rng(75)
large_g=ownarray(np.stack([rng.integers(-1,k,1600) for k in (2,3,4)]),np.int8)
large_k=ownarray([2,3,4],np.int64)
interval=ownarray([[0,10]],np.float32)
positions=ownarray([0,4,9],np.float32)
large_ref=ownarray(np.arange(1600)%3,np.int64)
large_ref[-1]=-1
large_data=([large_tree],interval,large_g,large_k,positions,large_ref,ages)
large_options=dict(num_classes=3,age_split=350.)
wanted,wanted_receipt=original.encode_block(*large_data,**large_options)
layouts=ImmutableTreeLayoutCacheV1(max_bytes=2**20,max_entries=2)
candidate,candidate_receipt=encode_block_prepared_v1(*large_data,
    tree_keys=[('realistic1600',)],layout_cache=layouts,**large_options)
assert candidate_receipt==wanted_receipt
realistic=equal_bank('independent_1600haplotypes_3199nodes_3SNPs',wanted,candidate)
assert len(large_tree.parent)==3199

out=Path(__file__).resolve().parent
result=dict(schema='inactive_CPU_feature_stable_FP32_repair_v3',
    status='AUTHOR_PASS_REPAIRED_SYNTHETIC_FP32_EQUIVALENCE',torch=torch.__version__,
    PyG=torch_geometric.__version__,CPU_threads=1,CUDA_initialized=torch.cuda.is_initialized(),
    support_checks=support_checks,prior_complete_bank_fields=checks,
    negative_checks=negative,independent_counterexample_repaired=repaired,
    independent_1600sample_fixture=realistic,independent_fixture_input_bytes=R['fixture_input_bytes'],
    numerical_change='Compensated FP32 summation across bounded SNP tile and allele partials',
    additional_compensation_bytes_per_vertex=12,named_working_allowance_margin_increase_bytes_per_vertex=32,
    tolerance=dict(rtol=2e-6,atol=2e-6),tolerance_widened=False,
    new_microtiming_performed=False,full_parent_speedup_claimed=False,
    original_payload_reads=0,truth_reads=0,guard_imports=0,GPU_jobs=0,model_forwards=0,optimizer_updates=0,
    v2_blocked_artifacts_preserved=True,live_adoption=False,execution_lineage_bridge=False)
(out/'synthetic_results.v3.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
size=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
assert size<16*2**20 and R['fixture_input_bytes']<16*2**20
print(json.dumps(dict(status=result['status'],repaired_counterexample=repaired,
    realistic=realistic,artifact_bytes=size,negative_cases=len(negative))))

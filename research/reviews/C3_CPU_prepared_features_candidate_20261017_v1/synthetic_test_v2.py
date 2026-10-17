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
from prepared_cpu_features_v2 import (
    ImmutableTreeLayoutCacheV1, ImmutableBlockFeatureCacheV1,
    branch_support_batched_v1)

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

timing = dict()
for name,call in [
    ("original_python_SNP_postorder",lambda: branch_concordance_support_v1(trees[0],g,k,np.arange(h))),
    ("candidate_tiled_PyG",lambda: branch_support_batched_v1(trees[0],g,k,np.arange(h),
        layout_cache=layout,tree_key=keys[0],site_tile=32,max_working_bytes=8*2**20))]:
    durations=[]
    for repeat in range(3):
        start=time.perf_counter()
        call()
        durations.append(time.perf_counter()-start)
    timing[name]=dict(seconds=durations,median_seconds=float(np.median(durations)))
out = Path(__file__).resolve().parent
result = dict(schema="inactive_CPU_feature_synthetic_equivalence_v2",
    status="PASS_SYNTHETIC_FP32_FEATURE_EQUIVALENCE",torch=torch.__version__,
    PyG=torch_geometric.__version__,CPU_threads=1,CUDA_initialized=torch.cuda.is_initialized(),
    observed_original_payload_reads=0,truth_reads=0,guard_imports=0,optimizer_updates=0,
    fixture=dict(haplotypes=h,sites_per_tree=sites,trees=4,total_sites=len(genotypes),
        cases=["binary","unary","nonbinary","isolated_forest","biallelic","triallelic",
               "four_alleles","missing_calls","all_missing_site","monomorphic_site","rare_allele"]),
    support_checks=support_checks,bank_fields=checks,negative_checks=negative,
    cache=dict(layout_builds=layout.builds,layout_hits=layout.hits,layout_bytes=layout.bytes,
        feature_builds=cache.builds,feature_hits=cache.hits,feature_tensor_bytes=cache.bytes),
    microtiming=timing,full_parent_speedup_claimed=False,
    numerical_adaptation="FP32 branch summary accumulator, original semantic counts and rules",
    limits=["synthetic CPU tensors only","no full-parent fit or throughput evidence",
            "no live adoption","no source lineage bridge","no model novelty claim"])
(out/"synthetic_results.v2.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
artifact_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
assert artifact_bytes < 16*2**20
print(json.dumps(dict(status=result["status"],artifact_bytes=artifact_bytes,
    max_support_abs_error=max(x["max_abs_error"] for x in support_checks),
    fields=len(checks),negative_checks=negative,microtiming=timing,
    cache=result["cache"])))

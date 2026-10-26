"""Native observed-tree slice parity and timings. No truth or model is opened."""
import dataclasses
import gc
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
import tskit
import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from research.repairs.benchmark_recovery_20261020.runtime import observations as current
from research.repairs.benchmark_recovery_20261020.runtime import episodes as old_episodes
from research.repairs.benchmark_recovery_20261020.runtime import cuts as old_cut
from research.preprocessing_optimization_20261026.tree import observations, parallel_observations, episodes, cut
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/preprocessing_optimization_20261026/tree'
manifest_path=ROOT/'experiments/results/benchmark_campaign_20261018/fresh_american/argformer_american_train_000/inference/observed/manifest.json'

def exact(a,b):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape and a.numpy().tobytes()==b.numpy().tobytes()
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    else:assert a==b

def bank_equal(a,b):
    for field in dataclasses.fields(a):exact(getattr(a,field.name),getattr(b,field.name))

def main():
    manifest=json.loads(manifest_path.read_text())
    ts=tskit.load(ROOT/manifest['tree_ref']['path'])
    roster=json.loads((ROOT/manifest['roster_ref']['path']).read_text())
    sample_nodes=np.asarray(roster['sample_index_to_inferred_node'],dtype=np.int64)
    workspace=current.native_workspace_v1(ts)
    trees=[];intervals=[]
    # First 256 unchanged native holds. Every original node and sample survives.
    for tr in itertools.islice(ts.trees(),256):
        trees.append(current.native_tree(tr,sample_nodes,workspace=workspace))
        intervals.append(tuple(tr.interval))
    all_positions=np.load(ROOT/manifest['positions_ref']['path'],mmap_mode='r')
    lo,hi=np.searchsorted(all_positions,[intervals[0][0],intervals[-1][1]])
    positions=np.asarray(all_positions[lo:hi])
    genotype=np.asarray(np.load(ROOT/manifest['genotypes_ref']['path'],mmap_mode='r')[lo:hi])
    allele=[]
    with (ROOT/manifest['alleles_ref']['path']).open() as stream:
        for i,line in enumerate(stream):
            if i>=hi:break
            if i>=lo:allele.append(len(json.loads(line)))
    k=np.asarray(allele,dtype=np.int64)
    ref=np.asarray([-1 if row['observed_reference_class'] is None else row['observed_reference_class'] for row in roster['samples']],dtype=np.int64)
    age=np.asarray([row['age_generations'] for row in roster['samples']],dtype=np.float64)
    args=(trees,intervals,genotype,k,positions,ref,age)
    variants=[('current',current.encode_block,{}),('compiled_serial',observations.encode_block,{})]
    variants += [(f'compiled_threads_{workers}',parallel_observations.encode_block,dict(preparation_workers=workers)) for workers in (1,2,4,8)]
    results=[];reference=None;reference_meta=None
    for name,fun,kwargs in variants:
        start=time.perf_counter()
        bank,metadata=fun(*args,num_classes=3,**kwargs)
        elapsed=time.perf_counter()-start
        if reference is None:reference,reference_meta=bank,metadata
        else:bank_equal(reference,bank);exact(reference_meta,metadata)
        results.append(dict(stage='encode',variant=name,seconds=elapsed,bitwise_equal=True))
        print(json.dumps(results[-1]),flush=True)
        if bank is not reference:del bank,metadata
        gc.collect()
    isolated=tuple(reference_meta.get('isolated_zero_site_holds',()))
    for stage,old,new,kwargs in (
        ('episodes',old_episodes.build_episodes,episodes.build_episodes,dict(max_payload_bytes=2**30,isolated_hold_indices=isolated)),
        ('cut',old_cut.prepare_cut_plan_v1,cut.prepare_cut_plan_v1,dict(expected_haplotypes=len(sample_nodes),isolated_hold_indices=isolated))):
        start=time.perf_counter();a,meta_a=old(reference,**kwargs);elapsed=time.perf_counter()-start
        results.append(dict(stage=stage,variant='current',seconds=elapsed));print(json.dumps(results[-1]),flush=True)
        start=time.perf_counter();b,meta_b=new(reference,**kwargs);elapsed=time.perf_counter()-start
        if stage=='cut':exact(a.tensors(),b.tensors())
        else:
            for f in dataclasses.fields(a):
                if f.name=='flags':exact(a.flags.node_flags,b.flags.node_flags)
                else:exact(getattr(a,f.name),getattr(b,f.name))
        exact(meta_a,meta_b)
        results.append(dict(stage=stage,variant='compiled_serial',seconds=elapsed,bitwise_equal=True));print(json.dumps(results[-1]),flush=True)
        del a,b,meta_a,meta_b;gc.collect()
    report=dict(status='pass',observation_manifest=str(manifest_path.relative_to(ROOT)),
        parent_id=manifest['parent_id'],native_hold_range=[0,len(trees)],nodes=reference.parent.numel(),
        sites=len(positions),samples=len(sample_nodes),input_support_bp=[intervals[0][0],intervals[-1][1]],
        results=results,torch_intraop_threads=1,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_truth_or_model_opened=True,no_reduction_inside_selected_native_holds=True)
    (OUT/'native_timing.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

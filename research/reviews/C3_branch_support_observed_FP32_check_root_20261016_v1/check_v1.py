from pathlib import Path
import hashlib,importlib.util,json,resource,time,datetime
import numpy as np
import torch
import tskit
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/C3_branch_support_observed_FP32_check_root_20261016_v1'
SOURCE=ROOT/'research/reviews/C3_vectorized_branch_support_source_20261016_v1/branch_concordance_support_torch_v2.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='bd9bdc32a1e48a4cd5ec0d6ebff33d1b7fabfcdec40cd7b6d88e6cf8389d9abe'
REF=ROOT/'src/arggnn/mutation_routed_tree_observations_v1.py'
assert hashlib.sha256(REF.read_bytes()).hexdigest()=='922df286e595370ade23595420a325014e4df64aac88060e8a1a846684eab068'
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1,branch_concordance_support_v1
spec=importlib.util.spec_from_file_location('_bounded_branch_support_candidate',SOURCE)
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
base=ROOT/'experiments/results/American_C3_short_train2mb_wave_v2/american_c3_short_train_000'
tree_path=base/'inference/observed/tsinfer_tsdate_sanitized.trees'
ts=tskit.load(tree_path)
positions=np.load(base/'simulation/observed/positions_bp0.npy',allow_pickle=False)
genotypes=np.load(base/'simulation/observed/genotypes_int8.npy',mmap_mode='r',allow_pickle=False)
alleles=np.asarray([len(json.loads(line)) for line in (base/'simulation/observed/site_alleles.jsonl').read_text().splitlines()],dtype=np.int64)
samples=ts.samples()
assert genotypes.shape==(len(positions),len(samples)) and np.array_equal(positions,ts.sites_position)
rows=[]
for fraction in [.25,.5,.75]:
    initial=int(fraction*ts.num_trees)
    selected=None
    for offset in range(min(256,ts.num_trees)):
        tree=ts.at_index((initial+offset)%ts.num_trees)
        lo,hi=np.searchsorted(positions,[tree.interval.left,tree.interval.right])
        if tree.num_roots==1 and hi>lo:
            selected=(tree,int(lo),int(hi),offset);break
    assert selected is not None
    tree,lo,hi,offset=selected
    nodes=np.asarray(list(tree.nodes()),dtype=np.int64)
    lookup={int(u):i for i,u in enumerate(nodes)}
    native=NativeObservedTreeV1(
        np.asarray([lookup.get(int(tree.parent(int(u))),-1) for u in nodes],dtype=np.int64),
        np.asarray(ts.nodes_time[nodes]),
        np.asarray([lookup[int(u)] for u in samples],dtype=np.int64),
        np.arange(len(samples),dtype=np.int64))
    g=np.asarray(genotypes[lo:hi]);k=alleles[lo:hi];ids=np.arange(len(samples),dtype=np.int64)
    start=time.perf_counter();old=branch_concordance_support_v1(native,g,k,ids);old_s=time.perf_counter()-start
    start=time.perf_counter();new=candidate.branch_concordance_support_torch_v2(native,g,k,ids);new_s=time.perf_counter()-start
    error=float(np.abs(old-new).max(initial=0))
    assert new.dtype==old.dtype==np.float32 and np.isfinite(new).all()
    assert np.allclose(old,new,rtol=2e-6,atol=2e-6),(tree.index,error)
    rows.append(dict(tree_index=int(tree.index),selection_quantile=fraction,forward_search_trees=offset,
        interval_bp=list(tree.interval),nodes=len(nodes),samples=len(samples),sites=hi-lo,
        original_seconds=old_s,candidate_seconds=new_s,ratio_original_over_candidate=old_s/new_s,
        max_abs_FP32_output_difference=error,within_declared_tolerance=True))
    print(json.dumps(rows[-1]),flush=True)
assert not torch.cuda.is_initialized()
result=dict(schema='C3_bounded_observed_branch_support_FP32_output_check_v1',
    completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='completed_three_observed_TRAIN_trees',parent='american_c3_short_train_000',split='TRAIN',
    candidate_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    reference_source_sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),
    exact_observed_tree_ref=dict(path=str(tree_path.relative_to(ROOT)),sha256=hashlib.sha256(tree_path.read_bytes()).hexdigest()),
    torch_threads=1,device='cpu',CUDA_context_created=False,model_training=False,query_ancestry_truth_opened=False,
    original_helper_internal_accumulation='NumPy float64, existing implementation',compared_outputs='NumPy float32',
    original_calls=3,candidate_calls=3,repetitions_per_case=1,rows=rows,
    peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
    general_correctness_or_speedup_claim=False,live_sources_or_run_changed=False)
dest=OUT/'completed.json'
assert not dest.exists()
dest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(completed_path=str(dest.relative_to(ROOT)),peak_RSS_bytes=result['peak_RSS_bytes'])),flush=True)

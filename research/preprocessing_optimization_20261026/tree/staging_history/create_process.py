from pathlib import Path
out=Path('/workspace-SR003.nfs2/ARG-GNN/research/preprocessing_optimization_20261026/tree')
s=(out/'observations.py').read_text()
s=s.replace('import math\n', 'import math\nimport multiprocessing\nfrom concurrent.futures import ProcessPoolExecutor\nfrom collections import deque\n')
s=s.replace('num_classes,age_split=350.,query_columns=None,resource_check=lambda stage:None):',
    'num_classes,age_split=350.,query_columns=None,resource_check=lambda stage:None,preparation_workers=1,tree_chunk_size=128):')
start=s.index('    for hold,tree in enumerate(trees):')
end=s.index('    parent=np.concatenate(parents);',start)
s=s[:start]+'''    require(type(preparation_workers) is int and 1<=preparation_workers<=8,"One to eight CPU preparation processes")
    require(type(tree_chunk_size) is int and 1<=tree_chunk_size<=256,"One to 256 whole native trees per transport chunk")
    def task(start):
        stop=min(start+tree_chunk_size,t)
        lo,hi=np.searchsorted(site_hold,[start,stop])
        offsets=np.searchsorted(site_hold,np.arange(start,stop+1))-lo
        return (trees[start:stop],np.asarray(g[lo:hi]),np.asarray(k[lo:hi]),offsets,identity,age)
    def ordered_results():
        if preparation_workers==1:
            for start in range(0,t,tree_chunk_size):
                yield from _prepare_chunk(task(start))
            return
        # Spawn is safe even after the parent's CUDA initialization. Every task
        # carries a bounded set of complete trees and their exact original sites.
        # Keep at most workers pending chunks and assemble only in native order.
        with ProcessPoolExecutor(max_workers=preparation_workers,
                mp_context=multiprocessing.get_context("spawn"),initializer=_initialize_worker) as executor:
            pending=deque();next_start=0
            for _ in range(min(preparation_workers,(t+tree_chunk_size-1)//tree_chunk_size)):
                pending.append(executor.submit(_prepare_chunk,task(next_start)));next_start+=tree_chunk_size
            try:
                while pending:
                    chunk=pending.popleft().result()
                    if next_start<t:
                        pending.append(executor.submit(_prepare_chunk,task(next_start)));next_start+=tree_chunk_size
                    yield from chunk
            except BaseException:
                for future in pending:future.cancel()
                raise
    resource_check("encode_full_observed_bank")
    for hold,result in enumerate(ordered_results()):
        if hold and hold%256==0:resource_check("encode_full_observed_bank")
        p,clock,sn,height,depth,descendant,duration,support,isolated=result
        if isolated:isolated_holds.append(hold)
        global_parent=p.copy();global_parent[p>=0]+=offset
        parents.append(global_parent);times.append(clock);samples.append(sn+offset)
        supports.append(support);heights.append(height);depths.append(depth)
        descendants.append(descendant);durations.append(duration);offset+=len(p)
'''+s[end:]
s += '''


def _initialize_worker():
    import os
    import torch
    import setproctitle
    require(not torch.cuda.is_initialized(), "Spawned CPU preparation worker has no CUDA state")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    setproctitle.setproctitle("AI Auto Research Pipeline Test")


def _prepare_chunk(task):
    trees,g,k,offsets,identity,age=task
    h=len(identity);results=[]
    for at,tree in enumerate(trees):
        lo,hi=offsets[at:at+2]
        raw_parent=np.asarray(tree.parent)
        isolated=raw_parent.ndim==1 and len(raw_parent)>1 and np.all(raw_parent==-1)
        if isolated:require(lo==hi,"H060 supports only zero-SNP all-sampled/all-isolated native holds")
        native_layout=_layout_with_isolated_samples(tree)
        p,clock,sn,sid,post,depth,height,descendant,duration=native_layout
        require(np.array_equal(sid,identity) and np.array_equal(clock[sn],age),'Every original sample/date retained in its exact column')
        child_count=np.bincount(p[p>=0],minlength=len(p))
        require(not child_count[sn].any() and np.count_nonzero(child_count==0)==h,'All active leaves must be original observed samples')
        if lo==hi:
            support=np.zeros((len(p),5),dtype=np.float32);support[:,4]=1.
        else:
            prepared=prepare_branch_level_layout_v1(native_layout)
            support=level_batched_branch_support_v1(prepared,g[lo:hi],k[lo:hi],identity,
                marker_chunk_size=32,max_count_elements=8388608).numpy()
        results.append((p,clock,sn,height,depth,descendant,duration,support,isolated))
    return results


def additional_memory_upper_bound(trees, genotype, *, preparation_workers, tree_chunk_size=128):
    """Conservative admission estimate, separate from scientific bank metadata.

    The fixed per-worker allowance covers imported Torch/PyG runtime and the
    unchanged 8,388,608-element int64 branch-count working tensor plus its
    bounded intermediate arrays. Input/output pickle copies are charged twice.
    Caller must add its existing complete-parent bank and global scratch bound.
    """
    require(1<=preparation_workers<=8 and 1<=tree_chunk_size<=256,"Bounded process-pool admission")
    max_nodes=max(len(tree.parent) for tree in trees)
    h=genotype.shape[1]
    chunk_native_bytes=tree_chunk_size*(max_nodes*80+h*24)
    # A single hold could own all markers, so charge the complete observed
    # genotype matrix as the conservative upper bound for one queued task.
    genotype_bytes=genotype.nbytes+len(genotype)*16
    worker_bytes=2**30
    return preparation_workers*(worker_bytes+2*chunk_native_bytes+2*genotype_bytes)
'''
(out/'process_observations.py').write_text(s)

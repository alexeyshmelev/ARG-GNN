"""Fresh explicit fast encoder for connected and isolated-compatible H060 holds.

Original validation, geometry, donor arrays, ordering and output construction
are retained. Only branch support preparation is replaced; all-connected holds
reach this same loop instead of escaping through the legacy connected encoder.
"""
from __future__ import annotations
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import numpy as np
from arggnn.american_c3_common_v1 import require
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1,_level_arrays,_validate_alleles
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.observations_v1 import (
    donor_frequencies)
from research.reviews.level_batched_observed_preparation_source_20261017_v1.level_batched_support_v1 import (
    prepare_branch_level_layout_v1,level_batched_branch_support_v1)


from .kernels import _layout_with_isolated_samples, _level_arrays, child_levels


def native_workspace_v1(ts):
    return dict(rank=np.empty(ts.num_nodes,dtype=np.int64),
        stamp=np.zeros(ts.num_nodes,dtype=np.uint64),epoch=0)


def native_tree(tree,sample_nodes,*,node_time=None,workspace=None):
    """Original local node order and exact FP64 metadata, vectorized C-array reads."""
    ts=tree.tree_sequence
    nodes=np.asarray(list(tree.nodes()),dtype=np.int64)
    if node_time is None:node_time=np.asarray(ts.nodes_time,dtype=np.float64)
    if workspace is None:workspace=native_workspace_v1(ts)
    require(workspace["rank"].shape==workspace["stamp"].shape==(ts.num_nodes,),
        "Same complete observed tree sequence workspace")
    epoch=workspace["epoch"]+1
    require(epoch<2**64,"Finite original native hold traversal")
    workspace["epoch"]=epoch
    rank,stamp=workspace["rank"],workspace["stamp"]
    rank[nodes]=np.arange(len(nodes),dtype=np.int64);stamp[nodes]=epoch
    raw_parent=tree.parent_array[nodes]
    nonroot=raw_parent>=0
    sample_nodes=np.asarray(sample_nodes,dtype=np.int64)
    require(np.all(stamp[sample_nodes]==epoch)
        and np.all(stamp[raw_parent[nonroot]]==epoch),"Every original sample/parent is active")
    parent=np.full(len(nodes),-1,dtype=np.int64)
    parent[nonroot]=rank[raw_parent[nonroot]]
    sample=rank[sample_nodes].copy()
    age=np.asarray(node_time[nodes],dtype=np.float64)
    return NativeObservedTreeV1(parent,age,sample,np.arange(len(sample),dtype=np.int64))


def encode_block(trees,intervals,genotype,allele_count,site_positions,reference_class,sample_age,*,
        num_classes,age_split=350.,query_columns=None,resource_check=lambda stage:None,preparation_workers=1,tree_chunk_size=128,
        max_parallel_bytes=8*2**30,max_chunk_bytes=64*2**20,preparation_observer=None):
    """Bounded complete forest, no hidden pruning or genotype/site sampling."""
    require(trees,'Nonempty whole-tree block required')
    g,k=_validate_alleles(genotype,allele_count);h=g.shape[1];t=len(trees)
    intervals=np.asarray(intervals,dtype=np.float64);x=np.asarray(site_positions,dtype=np.float64)
    ref=np.asarray(reference_class,dtype=np.int64);age=np.asarray(sample_age,dtype=np.float64)
    require(intervals.shape==(t,2) and np.array_equal(intervals[:-1,1],intervals[1:,0])
        and (intervals[:,1]>intervals[:,0]).all() and x.shape==(len(g),)
        and np.isfinite(x).all() and (np.diff(x)>0).all()
        and ((x>=intervals[0,0])&(x<intervals[-1,1])).all(),'Full original ordered block sites/intervals required')
    require(ref.shape==(h,) and age.shape==(h,) and ((ref>=-1)&(ref<num_classes)).all()
        and np.any(ref>=0) and np.any(ref<0),'Declared observed reference mask and unknown query required')
    query=np.flatnonzero(ref<0) if query_columns is None else np.asarray(query_columns,dtype=np.int64)
    require(query.ndim==1 and len(query)>0 and len(set(query.tolist()))==len(query)
        and ((query>=0)&(query<h)).all() and (ref[query]==-1).all(),'Exact actual-query columns must remain unknown input ancestry')
    site_hold=np.searchsorted(intervals[:,1],x,side='right').astype(np.int64)
    parents=[];times=[];samples=[];supports=[];heights=[];depths=[];descendants=[];durations=[];offset=0
    identity=np.arange(h,dtype=np.int64);isolated_holds=[]
    require(type(preparation_workers) is int and 1<=preparation_workers<=8,"One to eight CPU preparation processes")
    require(type(tree_chunk_size) is int and 1<=tree_chunk_size<=256,"One to 256 whole native trees per transport chunk")
    require(type(max_parallel_bytes) is int and max_parallel_bytes>0
        and type(max_chunk_bytes) is int and max_chunk_bytes>0,"Explicit positive parallel byte limits")
    admission=parallel_admission(preparation_workers,max_chunk_bytes)
    require(preparation_workers==1 or admission["additional_bytes"]<=max_parallel_bytes,
        "Child runtime and bounded IPC reserve must fit before process creation")
    def notify(event,detail):
        if preparation_observer is not None:preparation_observer(event,detail)
    notify("parallel_preparation_admitted",dict(admission,max_parallel_bytes=max_parallel_bytes,
        requested_workers=preparation_workers,max_chunk_bytes=max_chunk_bytes,tree_chunk_size=tree_chunk_size))
    def task(start,stop):
        lo,hi=np.searchsorted(site_hold,[start,stop])
        offsets=np.searchsorted(site_hold,np.arange(start,stop+1))-lo
        return (trees[start:stop],np.asarray(g[lo:hi]),np.asarray(k[lo:hi]),offsets,identity,age)
    descriptors=list(bounded_chunks(trees,site_hold,g,k,tree_chunk_size,max_chunk_bytes))
    def run_serial(descriptor):
        start,stop,charged=descriptor
        notify("parallel_preparation_serial_chunk",dict(first_hold=start,last_hold=stop,
            charged_bytes=charged,reason="serial_mode_or_single_tree_exceeds_IPC_bound"))
        result,_=_prepare_chunk(task(start,stop))
        yield from result
    def ordered_results():
        if preparation_workers==1:
            for descriptor in descriptors:yield from run_serial(descriptor)
            return
        with ProcessPoolExecutor(max_workers=preparation_workers,
                mp_context=multiprocessing.get_context("spawn"),initializer=_initialize_worker,
                initargs=(admission["worker_rss_limit_bytes"],)) as executor:
            pending=deque();cursor=0;charged_pending=0
            try:
                while cursor<len(descriptors) or pending:
                    while cursor<len(descriptors) and len(pending)<preparation_workers:
                        start,stop,charged=descriptors[cursor]
                        if charged>max_chunk_bytes:break
                        future=executor.submit(_prepare_chunk,task(start,stop))
                        pending.append((future,charged,start,stop));charged_pending+=charged;cursor+=1
                        require(charged_pending<=preparation_workers*max_chunk_bytes,"Bounded aggregate queued transport bytes")
                    if pending:
                        future,charged,start,stop=pending.popleft()
                        chunk,worker=future.result();charged_pending-=charged
                        resource_check("encode_full_observed_bank")
                        notify("parallel_preparation_chunk",dict(worker,first_hold=start,last_hold=stop,
                            charged_bytes=charged,pending_charged_bytes=charged_pending))
                        yield from chunk
                    else:
                        # Flush prior jobs before any oversized single tree.
                        # That original whole tree stays on the parent's existing
                        # admitted serial path, with no truncation or IPC copy.
                        yield from run_serial(descriptors[cursor]);cursor+=1
            except BaseException:
                for future,_,_,_ in pending:future.cancel()
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
    parent=np.concatenate(parents);clock=np.concatenate(times);height=np.concatenate(heights)
    depth=np.concatenate(depths);descendant=np.concatenate(descendants);duration=np.concatenate(durations)
    support=np.concatenate(supports);child=np.flatnonzero(parent>=0)
    geometry=np.zeros((len(parent),4),dtype=np.float32)
    geometry[child,:3]=np.stack((np.log1p(clock[child]),np.log1p(clock[parent[child]]),np.log1p(duration[child])),1)/math.log(10001.)
    geometry[child,3]=descendant[child]/h
    levels,children,ranks=child_levels(parent,height)
    donor,mask,panel,entropy,counts=donor_frequencies(g,k,ref,age,num_classes=num_classes,age_split=age_split)
    arrays=dict(parent=parent,node_time=clock,sample_index=np.stack(samples),upward_levels=levels,
        upward_children=children,upward_parent_rank=ranks,downward_levels=_level_arrays(depth)[1:],
        edge_geometry=geometry,branch_support_true=support,branch_support_sham=support.copy(),
        genotype=g.astype(np.int8,copy=True),log_donor_allele_frequency=donor,donor_call_mask=mask,
        panel_allele_frequency=panel,site_entropy=entropy,site_hold=site_hold,spans=intervals[:,1]-intervals[:,0],
        readout_hold=np.arange(t,dtype=np.int64),query_column=query,reference_class=ref,sample_age=age)
    bank=MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
    metadata=dict(nodes=len(parent),trees=t,haplotypes=h,sites=len(g),donor_group_counts=counts.tolist(),
        every_active_marginal_node_retained=True,unary_branches_preserved=True,targets_opened=False)
    if isolated_holds:
        metadata.update(isolated_zero_site_holds=isolated_holds,
            compatibility="all_sampled_all_isolated_zero_SNP_only",
            no_virtual_root_or_edge=True)
    metadata.update(observed_preparation_route="explicit_level_batched_global_v1",
        raw_layout_calls_per_tree=1,FP32_support_accumulation=True,
        historical_support_byte_identity_claimed=False)
    return bank,metadata



_WORKER_RSS_LIMIT = None
_WORKER_OBSERVED_PEAK_RSS = 0
_THREADPOOL_GUARD = None


def _initialize_worker(max_rss_bytes):
    import os
    import torch
    import setproctitle
    from threadpoolctl import threadpool_limits,threadpool_info
    global _WORKER_RSS_LIMIT,_THREADPOOL_GUARD
    require(not torch.cuda.is_initialized(), "Spawned CPU preparation worker has no CUDA state")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Apply to pools that NumPy/Torch may already have loaded during spawn.
    _THREADPOOL_GUARD=threadpool_limits(limits=1)
    require(all(pool["num_threads"]<=1 for pool in threadpool_info()),"Actual native worker pools must use one thread")
    _WORKER_RSS_LIMIT=max_rss_bytes
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    _worker_memory_check()


def _worker_memory_check():
    if _WORKER_RSS_LIMIT is not None:
        import os
        global _WORKER_OBSERVED_PEAK_RSS
        # Linux may retain the pre-exec parent's high-water ru_maxrss through
        # spawn. Admission must inspect this worker's current resident pages.
        with open('/proc/self/statm') as handle:
            current=int(handle.read().split()[1])*os.sysconf('SC_PAGE_SIZE')
        _WORKER_OBSERVED_PEAK_RSS=max(_WORKER_OBSERVED_PEAK_RSS,current)
        require(current<=_WORKER_RSS_LIMIT,
            "Preparation worker exceeded admitted current RSS, no reduced workload fallback")


def _prepare_chunk(task):
    trees,g,k,offsets,identity,age=task
    h=len(identity);results=[]
    for at,tree in enumerate(trees):
        _worker_memory_check()
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
    _worker_memory_check()
    import os,resource,torch
    from threadpoolctl import threadpool_info
    telemetry=dict(pid=os.getpid(),peak_rss_bytes=_WORKER_OBSERVED_PEAK_RSS,
        historical_rusage_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        torch_threads=torch.get_num_threads(),cuda_initialized=torch.cuda.is_initialized(),
        native_threads=[pool["num_threads"] for pool in threadpool_info()])
    return results,telemetry


def parallel_admission(workers,max_chunk_bytes):
    # A measured worker imports about0.5GiB. 1.5GiB also reserves the original
    # count tensor and FP32 support intermediates, then workers enforce this
    # current RSS ceiling after each complete native-tree preparation.
    worker_limit=1536*2**20
    ipc_bytes=2*(workers+1)*max_chunk_bytes
    return dict(worker_rss_limit_bytes=worker_limit,workers=workers,
        worker_bytes=workers*worker_limit,ipc_bytes=ipc_bytes,
        additional_bytes=workers*worker_limit+ipc_bytes)


def bounded_chunks(trees,site_hold,genotype,allele_count,max_trees,max_bytes):
    """Bound transported input AND output arrays, including pickle headroom.

    No data is changed. An oversized complete tree is explicitly marked for
    the parent's original serial path. Neighboring trees remain independent.
    """
    h=genotype.shape[1]
    start=0;charged=0
    for hold,tree in enumerate(trees):
        lo,hi=np.searchsorted(site_hold,[hold,hold+1])
        # Native arrays and returned geometry/support arrays together fit in
        # 96bytes per node plus32 per sampled leaf. Reserve32KiB per tree for
        # Python/NumPy pickle framing and twice the actual site payload.
        amount=(96*len(tree.parent)+32*h+32768
            +2*((hi-lo)*h*genotype.dtype.itemsize+(hi-lo)*allele_count.dtype.itemsize)
            +16*h+16)
        if hold>start and (hold-start>=max_trees or charged+amount>max_bytes):
            yield start,hold,charged
            start=hold;charged=0
        charged+=amount
        if charged>max_bytes:
            yield start,hold+1,charged
            start=hold+1;charged=0
    if start<len(trees):yield start,len(trees),charged


def additional_memory_upper_bound(trees,genotype,*,preparation_workers,tree_chunk_size=128,max_chunk_bytes=64*2**20):
    """Execution reserve to add to the existing whole-parent memory governor."""
    require(1<=preparation_workers<=8 and 1<=tree_chunk_size<=256,"Bounded process-pool admission")
    return parallel_admission(preparation_workers,max_chunk_bytes)["additional_bytes"]

from pathlib import Path
out=Path('/workspace-SR003.nfs2/ARG-GNN/research/preprocessing_optimization_20261026/tree')
p=out/'process_observations.py'
s=p.read_text()
s=s.replace('preparation_workers=1,tree_chunk_size=128):', '''preparation_workers=1,tree_chunk_size=128,
        max_parallel_bytes=8*2**30,max_chunk_bytes=64*2**20,preparation_observer=None):''')
start=s.index('    require(type(preparation_workers)')
end=s.index('    resource_check("encode_full_observed_bank")\n    for hold,result',start)
s=s[:start]+'''    require(type(preparation_workers) is int and 1<=preparation_workers<=8,"One to eight CPU preparation processes")
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
'''+s[end:]
start=s.index('def _initialize_worker(')
end=s.index('def _prepare_chunk(',start)
s=s[:start]+'''_WORKER_RSS_LIMIT = None
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
        import resource
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=_WORKER_RSS_LIMIT,
            "Preparation worker exceeded admitted RSS, no reduced workload fallback")


'''+s[end:]
s=s.replace('    for at,tree in enumerate(trees):\n        lo,hi=', '    for at,tree in enumerate(trees):\n        _worker_memory_check()\n        lo,hi=')
s=s.replace('    return results\n\n\ndef additional_memory_upper_bound', '''    _worker_memory_check()
    import os,resource,torch
    from threadpoolctl import threadpool_info
    telemetry=dict(pid=os.getpid(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        torch_threads=torch.get_num_threads(),cuda_initialized=torch.cuda.is_initialized(),
        native_threads=[pool["num_threads"] for pool in threadpool_info()])
    return results,telemetry


def additional_memory_upper_bound''')
start=s.index('def additional_memory_upper_bound')
s=s[:start]+'''def parallel_admission(workers,max_chunk_bytes):
    # A measured worker imports about0.5GiB. 1.5GiB also reserves the original
    # count tensor and FP32 support intermediates, then workers enforce this
    # RSS ceiling after each complete native-tree preparation.
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
'''
p.write_text(s)

from pathlib import Path
out=Path('/workspace-SR003.nfs2/ARG-GNN/research/preprocessing_optimization_20261026/tree')
s=(out/'observations.py').read_text()
s=s.replace('import math\n', 'import math\nfrom concurrent.futures import ThreadPoolExecutor\nfrom collections import deque\nimport threading\n')
s=s.replace('num_classes,age_split=350.,query_columns=None,resource_check=lambda stage:None):',
    'num_classes,age_split=350.,query_columns=None,resource_check=lambda stage:None,preparation_workers=1):')
start=s.index('    for hold,tree in enumerate(trees):')
end=s.index('    parent=np.concatenate(parents);',start)
replacement='''    require(type(preparation_workers) is int and 1<=preparation_workers<=8,"One to eight CPU tree workers")
    callback_lock=threading.Lock()
    def check():
        with callback_lock:resource_check("encode_full_observed_bank")
    def prepare_one(hold):
        tree=trees[hold]
        lo,hi=np.searchsorted(site_hold,[hold,hold+1])
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
        return p,clock,sn,height,depth,descendant,duration,support,isolated
    def ordered_results():
        if preparation_workers==1:
            for hold in range(t):yield prepare_one(hold)
            return
        # At most twice the worker count may be pending, including completed
        # results. Input order controls assembly, never task completion order.
        with ThreadPoolExecutor(max_workers=preparation_workers,thread_name_prefix="TreeLoom_exact_tree") as executor:
            pending=deque()
            next_hold=0
            for _ in range(min(t,2*preparation_workers)):
                pending.append(executor.submit(prepare_one,next_hold));next_hold+=1
            try:
                while pending:
                    result=pending.popleft().result()
                    if next_hold<t:
                        pending.append(executor.submit(prepare_one,next_hold));next_hold+=1
                    yield result
            except BaseException:
                for future in pending:future.cancel()
                raise
    for hold,result in enumerate(ordered_results()):
        if hold%256==0:check()
        p,clock,sn,height,depth,descendant,duration,support,isolated=result
        if isolated:isolated_holds.append(hold)
        global_parent=p.copy();global_parent[p>=0]+=offset
        parents.append(global_parent);times.append(clock);samples.append(sn+offset)
        supports.append(support);heights.append(height);depths.append(depth)
        descendants.append(descendant);durations.append(duration);offset+=len(p)
'''
s=s[:start]+replacement+s[end:]
s=s.replace('"""Fresh explicit fast encoder', '"""Bounded ordered threaded exact encoder')
(out/'parallel_observations.py').write_text(s)

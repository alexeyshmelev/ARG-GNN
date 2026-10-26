"""CPU spawn parity, uneven chunk boundaries and failure cleanup."""
import dataclasses
import json
import multiprocessing
import pickle
import random
from pathlib import Path
import numpy as np
import torch
import setproctitle
from research.preprocessing_optimization_20261026.tree import observations, process_observations
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1

def exact_bank(a,b):
    for f in dataclasses.fields(a):
        x,y=getattr(a,f.name),getattr(b,f.name)
        if isinstance(x,torch.Tensor):
            assert x.dtype==y.dtype and x.shape==y.shape and x.numpy().tobytes()==y.numpy().tobytes(),f.name
        else:
            assert len(x)==len(y),f.name
            for u,v in zip(x,y):
                assert u.dtype==v.dtype and u.shape==v.shape and u.numpy().tobytes()==v.numpy().tobytes(),f.name

def fixture():
    h=65;trees=[];positions=[]
    for at in range(37):
        p=np.full(h,-1,dtype=np.int64)
        ages=np.zeros(h)
        if at%5!=2:
            # A real proper clade plus a unary chain and multifurcating root.
            p=np.r_[np.full(32,h),np.full(33,h+2),h+1,h+2,-1]
            ages=np.r_[np.zeros(h),100.,200.,500.]
            positions.extend([10.*at+j for j in (0.,2.,5.,9.)])
        trees.append(NativeObservedTreeV1(p,ages,np.arange(h),np.arange(h)))
    positions=np.asarray(positions)
    rng=np.random.default_rng(812)
    k=rng.integers(1,5,len(positions))
    g=np.asarray([rng.integers(-1,int(width),h) for width in k])
    labels=np.arange(h)%3;labels[-5:]=-1
    return (trees,np.column_stack((np.arange(37)*10.,np.arange(1,38)*10.)),g,k,positions,labels,np.zeros(h))

def main():
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args=fixture()
    ref,meta=observations.encode_block(*args,num_classes=3)
    records=[]
    rng_before=(pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.get_rng_state().numpy().tobytes())
    for workers,chunk in [(1,1),(2,1),(4,8),(8,128),(8,1)]:
        result,new_meta=process_observations.encode_block(*args,num_classes=3,preparation_workers=workers,tree_chunk_size=chunk,max_parallel_bytes=16*2**30)
        exact_bank(ref,result);assert meta==new_meta
        assert not multiprocessing.active_children()
        records.append(dict(workers=workers,chunk=chunk,status='bitwise_identical'))
        print(records[-1],flush=True)
    # Prove byte-based splitting and oversized-tree fallback preserve all data.
    events=[]
    result,new_meta=process_observations.encode_block(*args,num_classes=3,preparation_workers=4,
        tree_chunk_size=128,max_chunk_bytes=40000,preparation_observer=lambda name,value:events.append((name,value)))
    exact_bank(ref,result);assert meta==new_meta
    assert any(name=='parallel_preparation_serial_chunk' for name,_ in events)
    assert not multiprocessing.active_children()
    # Input/output serialization stays below the declared transport charge.
    site_hold=np.searchsorted(np.asarray(args[1])[:,1],args[4],side='right')
    for start,stop,charge in process_observations.bounded_chunks(args[0],site_hold,args[2],args[3],8,2**20):
        lo,hi=np.searchsorted(site_hold,[start,stop])
        offsets=np.searchsorted(site_hold,np.arange(start,stop+1))-lo
        task=(args[0][start:stop],args[2][lo:hi],args[3][lo:hi],offsets,np.arange(65),args[6])
        result,_=process_observations._prepare_chunk(task)
        assert len(pickle.dumps(task))+len(pickle.dumps(result))<=charge
    try:process_observations.encode_block(*args,num_classes=3,preparation_workers=4,max_parallel_bytes=1)
    except ValueError:pass
    else:raise AssertionError('unadmitted pool started')
    assert not multiprocessing.active_children()
    bad=list(args)
    bad[0]=list(args[0]);bad[0][3]=dataclasses.replace(bad[0][3],sample_node=np.zeros(65,dtype=np.int64))
    try:process_observations.encode_block(*bad,num_classes=3,preparation_workers=4,tree_chunk_size=2)
    except ValueError:pass
    else:raise AssertionError('worker failure not forwarded')
    assert not multiprocessing.active_children()
    def fail(stage):raise RuntimeError('admission rejected')
    try:process_observations.encode_block(*args,num_classes=3,preparation_workers=4,resource_check=fail)
    except RuntimeError as error:assert str(error)=='admission rejected'
    else:raise AssertionError('admission failure not forwarded')
    assert not multiprocessing.active_children()
    for workers,chunk in [(0,1),(9,1),(True,1),(1,0),(1,257)]:
        try:process_observations.encode_block(*args,num_classes=3,preparation_workers=workers,tree_chunk_size=chunk,max_parallel_bytes=16*2**30)
        except ValueError:pass
        else:raise AssertionError('invalid worker/chunk limits accepted')
    rng_after=(pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.get_rng_state().numpy().tobytes())
    assert rng_before==rng_after
    output=dict(parent_rng_unchanged=True,transport_byte_bound_verified=True,oversized_serial_fallback_verified=True,admission_budget_enforced=True,status='pass',runs=records,worker_failure_cleanup=True,admission_before_start=True,invalid_bounds_rejected=5)
    Path('research/preprocessing_optimization_20261026/tree/process_parity.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()

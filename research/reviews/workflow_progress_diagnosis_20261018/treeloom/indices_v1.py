"""Exact integer preparation helpers; no model, truth, inference or launch API."""
from __future__ import annotations
import hashlib
import numpy as np


def need(ok,message):
    if not ok:
        raise ValueError(message)


def stable_levels(level):
    level=np.asarray(level)
    need(level.ndim==1 and level.dtype.kind in "iu" and (level>=0).all(),"Nonnegative integer levels")
    count=int(level.max(initial=0))+1
    order=np.argsort(level,kind="stable")
    return tuple(np.split(order,np.cumsum(np.bincount(level,minlength=count))[:-1]))


def fill_groups(groups,children,parent_levels):
    need(len(children)==len(parent_levels),"Exact child-to-level join")
    order=np.argsort(parent_levels,kind="stable")
    parts=np.split(order,np.cumsum(np.bincount(parent_levels,minlength=len(groups)))[:-1])
    groups[:]=[children[part] for part in parts]


def adjacency(parent,dtype=np.int64):
    """Original prepend order: head is largest child, sibling its predecessor."""
    p=np.asarray(parent)
    child=np.flatnonzero(p>=0)
    order=np.argsort(p[child],kind="stable")
    grouped=child[order]
    head=np.full(len(p),-1,dtype=dtype)
    sibling=np.full(len(p),-1,dtype=dtype)
    if len(grouped):
        start=np.r_[True,p[grouped[1:]]!=p[grouped[:-1]]]
        prior=np.r_[-1,grouped[:-1]]
        sibling[grouped]=np.where(start,-1,prior)
        end=np.r_[start[1:],True]
        head[p[grouped[end]]]=grouped[end]
    return head,sibling


def root_and_cut(bank,parent,times,roots,horizon):
    """One vector operation per existing depth level, preserving exact integers."""
    p=np.asarray(parent);clock=np.asarray(times)
    head,sibling=adjacency(p)
    owner=np.full(len(p),-1,dtype=np.int32)
    cut=np.full(len(p),-1,dtype=np.int64)
    owner[roots]=np.arange(len(roots),dtype=np.int32)
    valid=clock[roots]<=horizon
    cut[roots[valid]]=roots[valid]
    for tensor in bank.downward_levels:
        nodes=tensor.detach().numpy()
        need((p[nodes]>=0).all() and (owner[p[nodes]]>=0).all(),
             "Existing depth levels must retain parent-before-child order")
        owner[nodes]=owner[p[nodes]]
        low=nodes[clock[nodes]<=horizon]
        high_parent=clock[p[low]]>horizon
        cut[low]=np.where(high_parent,low,cut[p[low]])
        need((cut[low]>=0).all(),"Original highest real at-or-below-cut ancestor")
    need((owner>=0).all(),"All original nodes must be rooted and covered")
    return head,sibling,owner,cut


def forest_for_bank(bank,parent,samples,isolated_hold_indices=()):
    from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.structural_index_v1 import (
        classify_sample_roots,validate_root_partition)
    p=np.asarray(parent);roots=np.flatnonzero(p<0)
    head,sibling=adjacency(p,dtype=np.int32)
    owner=np.full(len(p),-1,dtype=np.int32)
    owner[roots]=np.arange(len(roots),dtype=np.int32)
    for tensor in bank.downward_levels:
        nodes=tensor.detach().numpy()
        need((p[nodes]>=0).all() and (owner[p[nodes]]>=0).all(),
             "Existing depth levels must retain parent-before-child order")
        owner[nodes]=owner[p[nodes]]
    need((owner>=0).all(),"All original nodes must be rooted and covered")
    root_of=roots[owner]
    ordered=[classify_sample_roots(p,head,root_of,row) for row in samples]
    validate_root_partition(ordered,roots,isolated_hold_indices)
    return head.astype(np.int32),sibling.astype(np.int32),ordered,int(np.count_nonzero(p>=0))


def first_occurrence(values):
    values=np.asarray(values)
    unique,first,inverse=np.unique(values,return_index=True,return_inverse=True)
    order=np.argsort(first,kind="stable")
    remap=np.empty(len(order),dtype=np.int64)
    remap[order]=np.arange(len(order),dtype=np.int64)
    return unique[order],remap[inverse]


def cut_groups(parent,times,samples,ages,cut,horizon):
    t,h=samples.shape
    local=np.empty((t,h),dtype=np.int64)
    global_groups=np.empty_like(local)
    offsets=[0];representatives=[];counts=[];root_flags=[]
    above=ages>horizon
    for at,row in enumerate(samples):
        selected=np.where(above,row,cut[row])
        reps,groups=first_occurrence(selected)
        need((reps>=0).all(),"Every original sample has its real cut representative")
        local[at]=groups
        global_groups[at]=groups+offsets[-1]
        representatives.append(reps)
        counts.append(np.bincount(groups,minlength=len(reps)))
        root_flags.append((parent[reps]<0)&(times[reps]<=horizon))
        offsets.append(offsets[-1]+len(reps))
    return (np.concatenate(representatives),np.concatenate(counts),
            np.concatenate(root_flags),offsets,local,global_groups,above)


def direction_layouts(groups,order,h):
    g=len(groups)
    cells=np.empty((g,h),dtype=np.int64)
    current=np.zeros((g,h),dtype=np.int64)
    old=np.zeros((g,h),dtype=np.int64)
    counts=np.zeros((g,h),dtype=np.float32)
    previous=None
    for e in order:
        now=groups[e]
        if previous is None:
            cells[e]=now
            k=int(now.max())+1
            current[e,:k]=np.arange(k)
            old[e,:k]=np.arange(k)
            counts[e,:k]=np.bincount(now,minlength=k)
        else:
            # Both class IDs are in [0,h), so this injective integer key is exact.
            keys=groups[previous]*h+now
            unique,inverse=first_occurrence(keys)
            k=len(unique)
            old[e,:k]=unique//h
            current[e,:k]=unique%h
            cells[e]=inverse
            counts[e,:k]=np.bincount(inverse,minlength=k)
        previous=e
    return cells,current,old,counts


def consecutive_members(holds,episodes):
    order=np.argsort(episodes,kind="stable")
    grouped=episodes[order];positions=holds[order]
    same=grouped[1:]==grouped[:-1]
    need(np.all(np.diff(positions)[same]==1),
         "Each episode must occur once per consecutive tree")


def fingerprint(tensor):
    """Same original dtype/shape/byte hash, with bounded hash scratch."""
    a=tensor.detach().contiguous().numpy()
    result=hashlib.sha256()
    result.update(str(a.dtype).encode());result.update(str(a.shape).encode())
    view=memoryview(a).cast("B")
    for first in range(0,len(view),8*2**20):
        result.update(view[first:first+8*2**20])
    return result.hexdigest()

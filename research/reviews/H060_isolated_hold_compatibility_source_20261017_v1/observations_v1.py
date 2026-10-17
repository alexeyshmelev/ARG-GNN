"""Generic cardinality observation encoding for the fresh American C3 GNN.

Connected-tree layout and mutation-concordance sweeps use the original functions.
Zero-edge observed sample forests retain every isolated sample without fake edges.
No target, ancestry annotation or mutation placement record is accepted.
"""
from __future__ import annotations
import math
import numpy as np
from arggnn.american_c3_common_v1 import require
from arggnn.american_c3_gnn_observations_v1 import encode_block as _connected_encode_block
from arggnn.mutation_routed_tree_observations_v1 import (
    NativeObservedTreeV1,_layout,_level_arrays,_validate_alleles,branch_concordance_support_v1,
)
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1


def donor_frequencies(genotype,allele_count,reference_class,sample_age,*,num_classes,age_split=350.,alpha=.5):
    """Original cohort-common-callability rule with C classes and 1/2 age cohorts."""
    g,k=_validate_alleles(genotype,allele_count);ref=np.asarray(reference_class);age=np.asarray(sample_age)
    require(type(num_classes) is int and num_classes>=2 and ref.shape==(g.shape[1],)
        and ref.dtype.kind in 'iu' and ((ref>=-1)&(ref<num_classes)).all()
        and age.shape==ref.shape and np.isfinite(age).all() and (age>=0).all() and alpha==.5,
        'Generic observed reference labels/ages and original alpha.5 required')
    cohorts=[np.ones(len(ref),dtype=bool)] if age_split is None else [age<age_split,age>=age_split]
    if age_split is not None:require(age_split>0 and math.isfinite(age_split),'Positive declared age threshold')
    s=len(g);a=int(k.max(initial=1));channels=num_classes*len(cohorts)
    logfreq=np.zeros((s,a,channels),dtype=np.float32);mask=np.zeros((s,channels),dtype=bool)
    groups=np.zeros(channels,dtype=np.int64);valid=np.arange(a)[None,:]<k[:,None]
    for cohort,age_group in enumerate(cohorts):
        donors={};calls={}
        for cls in range(num_classes):
            channel=cohort*num_classes+cls;columns=np.flatnonzero(age_group&(ref==cls));groups[channel]=len(columns)
            if len(columns):
                donors[channel]=g[:,columns];calls[channel]=np.count_nonzero(donors[channel]>=0,axis=1)
        common=np.ones(s,dtype=bool)
        for called in calls.values():common &= called>0
        for channel,donor in donors.items():
            count=np.stack([np.count_nonzero(donor==i,axis=1) for i in range(a)],axis=1)
            prob=(count+alpha)/(calls[channel]+k*alpha)[:,None]
            logfreq[:,:,channel]=np.where(common[:,None]&valid,np.log(prob),0.);mask[:,channel]=common
    called=np.count_nonzero(g>=0,axis=1)
    counts=np.stack([np.count_nonzero(g==i,axis=1) for i in range(a)],axis=1)
    panel=np.zeros((s,a),dtype=np.float32);np.divide(counts,called[:,None],out=panel,where=called[:,None]>0)
    entropy=-(panel*np.log(np.where(panel>0,panel,1.))).sum(1).astype(np.float32)
    return logfreq,mask,panel,entropy,groups


def native_tree(tree,sample_nodes):
    """Retain every active marginal node, including unary and nonbinary branches."""
    nodes=np.asarray(list(tree.nodes()),dtype=np.int64);lookup={int(u):i for i,u in enumerate(nodes)}
    parent=np.asarray([lookup.get(int(tree.parent(int(u))),-1) for u in nodes],dtype=np.int64)
    sample=np.asarray([lookup[int(u)] for u in sample_nodes],dtype=np.int64)
    age=np.asarray([tree.tree_sequence.node(int(u)).time for u in nodes],dtype=np.float64)
    return NativeObservedTreeV1(parent,age,sample,np.arange(len(sample),dtype=np.int64))


def _layout_with_isolated_samples(tree):
    """C3-only zero-edge sampled forest; delegate every other native topology."""
    p=np.asarray(tree.parent)
    if p.ndim!=1 or p.dtype.kind not in 'iu' or len(p)<=1 or not np.all(p==-1):
        return _layout(tree)
    clock=np.asarray(tree.node_time,dtype=np.float64)
    sn=np.asarray(tree.sample_node);sid=np.asarray(tree.sample_id);n=len(p)
    require(clock.shape==(n,) and np.isfinite(clock).all() and (clock>=0).all()
        and sn.ndim==sid.ndim==1 and sn.dtype.kind in 'iu' and sid.dtype.kind in 'iu'
        and sn.shape==sid.shape==(n,) and np.array_equal(np.sort(sn),np.arange(n))
        and (sid>=0).all() and len(set(sid.tolist()))==n,
        'An isolated observed forest must retain all and only unique sampled nodes/identities')
    return (p.astype(np.int64,copy=False),clock,sn.astype(np.int64,copy=False),
        sid.astype(np.int64,copy=False),np.arange(n,dtype=np.int64),
        np.zeros(n,dtype=np.int64),np.zeros(n,dtype=np.int64),
        np.ones(n,dtype=np.int64),np.zeros(n,dtype=np.float64))


def encode_block(trees,intervals,genotype,allele_count,site_positions,reference_class,sample_age,*,
        num_classes,age_split=350.,query_columns=None):
    """Bounded complete forest, no hidden pruning or genotype/site sampling."""
    if trees and all(np.count_nonzero(np.asarray(tree.parent)==-1)==1 for tree in trees):
        return _connected_encode_block(trees,intervals,genotype,allele_count,site_positions,reference_class,sample_age,
            num_classes=num_classes,age_split=age_split,query_columns=query_columns)
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
    for hold,tree in enumerate(trees):
        lo,hi=np.searchsorted(site_hold,[hold,hold+1])
        raw_parent=np.asarray(tree.parent)
        if raw_parent.ndim==1 and len(raw_parent)>1 and np.all(raw_parent==-1):
            require(lo==hi,"H060 supports only zero-SNP all-sampled/all-isolated native holds")
            isolated_holds.append(hold)
        p,clock,sn,sid,post,depth,height,descendant,duration=_layout_with_isolated_samples(tree)
        require(np.array_equal(sid,identity) and np.array_equal(clock[sn],age),'Every original sample/date retained in its exact column')
        children=np.bincount(p[p>=0],minlength=len(p))
        require(not children[sn].any() and np.count_nonzero(children==0)==h,'All active leaves must be original observed samples')
        global_parent=p.copy();global_parent[p>=0]+=offset
        parents.append(global_parent);times.append(clock);samples.append(sn+offset)
        if len(p)>1 and np.all(p==-1):
            support=np.zeros((len(p),5),dtype=np.float32);support[:,4]=1.
            supports.append(support)
        else:
            supports.append(branch_concordance_support_v1(tree,g[lo:hi],k[lo:hi],identity))
        heights.append(height);depths.append(depth);descendants.append(descendant);durations.append(duration);offset+=len(p)
    parent=np.concatenate(parents);clock=np.concatenate(times);height=np.concatenate(heights)
    depth=np.concatenate(depths);descendant=np.concatenate(descendants);duration=np.concatenate(durations)
    support=np.concatenate(supports);child=np.flatnonzero(parent>=0)
    geometry=np.zeros((len(parent),4),dtype=np.float32)
    geometry[child,:3]=np.stack((np.log1p(clock[child]),np.log1p(clock[parent[child]]),np.log1p(duration[child])),1)/math.log(10001.)
    geometry[child,3]=descendant[child]/h
    levels=_level_arrays(height);rank=np.empty(len(parent),dtype=np.int64)
    for level in levels:rank[level]=np.arange(len(level))
    groups=[[] for _ in levels]
    for node in child:groups[int(height[parent[node]])].append(int(node))
    children=tuple(np.asarray(v,dtype=np.int64) for v in groups)
    ranks=tuple(rank[parent[v]] for v in children)
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
    return bank,metadata

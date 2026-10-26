"""Fresh explicit fast encoder for connected and isolated-compatible H060 holds.

Original validation, geometry, donor arrays, ordering and output construction
are retained. Only branch support preparation is replaced; all-connected holds
reach this same loop instead of escaping through the legacy connected encoder.
"""
from __future__ import annotations
import math
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
        num_classes,age_split=350.,query_columns=None,resource_check=lambda stage:None):
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
    for hold,tree in enumerate(trees):
        if hold%256==0:resource_check("encode_full_observed_bank")
        lo,hi=np.searchsorted(site_hold,[hold,hold+1])
        raw_parent=np.asarray(tree.parent)
        if raw_parent.ndim==1 and len(raw_parent)>1 and np.all(raw_parent==-1):
            require(lo==hi,"H060 supports only zero-SNP all-sampled/all-isolated native holds")
            isolated_holds.append(hold)
        native_layout=_layout_with_isolated_samples(tree)
        p,clock,sn,sid,post,depth,height,descendant,duration=native_layout
        require(np.array_equal(sid,identity) and np.array_equal(clock[sn],age),'Every original sample/date retained in its exact column')
        children=np.bincount(p[p>=0],minlength=len(p))
        require(not children[sn].any() and np.count_nonzero(children==0)==h,'All active leaves must be original observed samples')
        global_parent=p.copy();global_parent[p>=0]+=offset
        parents.append(global_parent);times.append(clock);samples.append(sn+offset)
        if lo==hi:
            # Original support has no chunks: roots and every real branch get
            # empty=1, and all other four fields remain exact FP32 zero.
            support=np.zeros((len(p),5),dtype=np.float32);support[:,4]=1.
            supports.append(support)
        else:
            prepared=prepare_branch_level_layout_v1(native_layout)
            supports.append(level_batched_branch_support_v1(
                prepared,g[lo:hi],k[lo:hi],identity,
                marker_chunk_size=32,max_count_elements=8388608).numpy())
        heights.append(height);depths.append(depth);descendants.append(descendant);durations.append(duration);offset+=len(p)
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

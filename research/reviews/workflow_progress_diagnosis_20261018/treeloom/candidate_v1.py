"""Unpinned preparation candidate: exact integer changes, original float kernels.

No public/live module is mutated. This module cannot run training or inference.
Full-parent use needs separately approved finite CPU/storage admission.
"""
from __future__ import annotations
import ast
import copy
import inspect
import types
import textwrap
from . import indices_v1 as ix


def need(ok,message):
    if not ok:
        raise ValueError(message)


def compile_function(function, transform=None, **bindings):
    source=textwrap.dedent(inspect.getsource(function))
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    node.decorator_list=[]
    if transform is not None:
        transform(node)
    namespace=dict(function.__globals__,**bindings)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 function.__code__.co_filename+":exact_integer_candidate","exec"),namespace)
    return namespace[node.name]


def encode_block(*args,**kwargs):
    from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_observations_v1 as original
    def transform(node):
        changed=0
        for at,item in enumerate(node.body):
            if isinstance(item,ast.For) and isinstance(item.target,ast.Name) and item.target.id=="node" \
                    and isinstance(item.iter,ast.Name) and item.iter.id=="child":
                node.body[at]=ast.parse("_fill_groups(groups,child,height[parent[child]])").body[0]
                changed+=1
        need(changed==1,"Only original global child grouping loop may change")
    call=compile_function(original.encode_block,transform,
        _level_arrays=ix.stable_levels,_fill_groups=ix.fill_groups)
    return call(*args,**kwargs)


def build_episodes(bank,*,max_payload_bytes,isolated_hold_indices=()):
    from arggnn import ordered_clade_episode_gnn_v1 as original
    from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import episode_preparation_v1 as isolated
    def forest(parent,samples):
        return ix.forest_for_bank(bank,parent,samples,isolated_hold_indices)
    # Preserve all original clade masks, unary representatives, episode IDs,
    # member order and native FP32 leave-one-out span reductions.
    arrays,metadata=compile_function(original.build_clade_episode_arrays_v1,
        _forest_index=forest,_tree_clades=isolated._tree_clades)(
            bank.parent.numpy(),bank.sample_index.numpy(),bank.spans.numpy(),
            max_payload_bytes=max_payload_bytes)
    def transform(node):
        changed=0
        for at,item in enumerate(node.body):
            if isinstance(item,ast.For) and isinstance(item.target,ast.Tuple) \
                    and [getattr(n,'id',None) for n in item.target.elts]==["hold","episode"]:
                node.body[at]=ast.parse("_consecutive_members(holds,episodes)").body[0]
                changed+=1
        need(changed==1,"Only original integer consecutiveness loop may change")
    original_loader=original.OrderedCladeEpisodesV1.from_verified_arrays.__func__
    load=compile_function(original_loader,transform,_consecutive_members=ix.consecutive_members)
    episodes=load(original.OrderedCladeEpisodesV1,arrays,metadata)
    return episodes,metadata


def prepare_cut(bank,*,expected_haplotypes,max_nodes,max_holds,isolated_hold_indices=()):
    import numpy as np
    from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import cut_preparation_v1 as original
    def transform(node):
        # H060's compatibility body already supports the connected case; skip
        # only its routing shortcut so the same integer replacements cover both.
        need(isinstance(node.body[1],ast.If) or isinstance(node.body[0],ast.Expr),"Original cut source shape")
        routing=[i for i,n in enumerate(node.body) if isinstance(n,ast.If)
                 and "_original.prepare_cut_plan_v1" in ast.unparse(n)]
        need(len(routing)==1,"One original connected routing shortcut required")
        del node.body[routing[0]]
        starts=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="head" for t in n.targets)]
        ends=[i for i,n in enumerate(node.body) if isinstance(n,ast.For)
              and isinstance(n.target,ast.Tuple)
              and [getattr(e,'id',None) for e in n.target.elts]==["tree_id","root"]]
        need(len(starts)==len(ends)==1 and starts[0]<ends[0],"Exact adjacency/root traversal anchors")
        node.body[starts[0]:ends[0]+1]=ast.parse(
            "head,sibling,owner,cut=_root_and_cut(bank,p,times,roots,HORIZON)").body
        starts=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Tuple) and [getattr(e,'id',None) for e in t.elts]==
                        ["representatives","group_counts","root_flags"] for t in n.targets)]
        ends=[i for i,n in enumerate(node.body) if isinstance(n,ast.For)
              and isinstance(n.target,ast.Name) and n.target.id=="at"
              and "lookup" in ast.unparse(n)]
        need(len(starts)==len(ends)==1 and starts[0]<ends[0],"Exact original sample grouping anchors")
        node.body[starts[0]:ends[0]+1]=ast.parse(
            "representatives,group_counts,root_flags,offsets,local_groups,global_groups,above="
            "_cut_groups(p,times,samples,ages,cut,HORIZON)").body
        changed=0
        for item in ast.walk(node):
            if isinstance(item,ast.Compare) and "len(set(representatives))" in ast.unparse(item):
                item.left=ast.parse("np.unique(representatives).size",mode="eval").body
                changed+=1
        need(changed==1,"Only original representative uniqueness check may change")
    call=compile_function(original.prepare_cut_plan_v1,transform,
        _root_and_cut=ix.root_and_cut,_cut_groups=ix.cut_groups,
        _direction_layouts=ix.direction_layouts,_fingerprint=ix.fingerprint)
    return call(bank,expected_haplotypes=expected_haplotypes,max_nodes=max_nodes,
        max_holds=max_holds,isolated_hold_indices=isolated_hold_indices)


def prepare_synthetic_or_admitted(trees,intervals,genotype,allele_count,site_positions,
                                  reference_class,sample_age,*,num_classes,query_columns,
                                  max_nodes,max_holds,max_episode_payload_bytes,resource_check):
    """Complete CPU observations only; caller supplies all original observed data."""
    need(callable(resource_check),"Finite caller-owned CPU resource check required")
    need(len(trees)<=max_holds and sum(len(t.parent) for t in trees)<=max_nodes,
         "Complete parent must fit explicit bounds; no truncation")
    resource_check("candidate_encode_start")
    bank,metadata=encode_block(trees,intervals,genotype,allele_count,site_positions,
        reference_class,sample_age,num_classes=num_classes,query_columns=query_columns,
        age_split=350.,resource_check=resource_check)
    isolated=tuple(metadata.get("isolated_zero_site_holds",()))
    resource_check("candidate_episodes_start")
    episodes,episode_metadata=build_episodes(bank,max_payload_bytes=max_episode_payload_bytes,
                                            isolated_hold_indices=isolated)
    resource_check("candidate_cut_start")
    cut,cut_metadata=prepare_cut(bank,expected_haplotypes=len(reference_class),
        max_nodes=max_nodes,max_holds=max_holds,isolated_hold_indices=isolated)
    resource_check("candidate_complete")
    return bank,episodes,cut,dict(observation=metadata,episodes=episode_metadata,cut=cut_metadata)

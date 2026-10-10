"""Observed-tree canonicalization comparator, without ancestry annotations.

Removing anonymous unary vertices and merging exact duplicate physical holds
are preprocessing controls. Standard models after this preprocessing can share
the two representation contracts. Those contracts alone do not establish a
kernel model's predictive or scientific contribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from typing import Mapping

import torch
from torch import Tensor

from .data import GraphSequence
from .models import GraphSequenceInput, validate_model_input


@dataclass(frozen=True)
class CanonicalizedObservedTrees:
    sequence: GraphSequence
    provenance: dict
    source_graph_indices: tuple[tuple[int, ...], ...]
    source_node_indices: Tensor          # [Nout], first representative source rows
    source_graph_to_canonical: Tensor    # [Gin], target_graph remapping
    schema_version: int = 1

    def as_dict(self) -> dict:
        return {'sequence': self.sequence.as_dict(), 'provenance': copy.deepcopy(self.provenance),
                'source_graph_indices': self.source_graph_indices,
                'source_node_indices': self.source_node_indices,
                'source_graph_to_canonical': self.source_graph_to_canonical,
                'schema_version': self.schema_version}

    @classmethod
    def from_dict(cls, values: dict) -> CanonicalizedObservedTrees:
        expected = {'sequence','provenance','source_graph_indices','source_node_indices',
                    'source_graph_to_canonical','schema_version'}
        if not isinstance(values,dict) or set(values) != expected or values['schema_version'] != 1:
            raise ValueError('CanonicalizedObservedTrees schema/fields mismatch')
        result = cls(GraphSequence.from_dict(values['sequence']), copy.deepcopy(values['provenance']),
                     tuple(tuple(row) for row in values['source_graph_indices']),
                     values['source_node_indices'], values['source_graph_to_canonical'])
        validate_model_input(result.sequence)
        n,g = result.sequence.x.shape[0],result.sequence.num_graphs
        if result.source_node_indices.shape != (n,) or result.source_node_indices.dtype != torch.long or bool((result.source_node_indices < 0).any()):
            raise ValueError('Retained source-node correspondence is invalid')
        mapping = result.source_graph_to_canonical
        if mapping.ndim != 1 or mapping.dtype != torch.long or bool(((mapping < 0) | (mapping >= g)).any()):
            raise ValueError('Source graph correspondence is invalid')
        if len(result.source_graph_indices) != g or any(not row for row in result.source_graph_indices):
            raise ValueError('Source graph grouping is invalid')
        flat = [graph for row in result.source_graph_indices for graph in row]
        if sorted(flat) != list(range(mapping.numel())) or any(int(mapping[graph]) != index for index,row in enumerate(result.source_graph_indices) for graph in row):
            raise ValueError('Source graph grouping/mapping disagree')
        return result


@dataclass
class _CanonicalTree:
    x: Tensor
    node_ids: Tensor
    clades: Tensor
    edges: Tensor
    source_rows: Tensor
    interval: Tensor
    source_graphs: list[int]
    removed_unary_interior: int
    removed_unary_root: int


def _same_observed_tree(first: _CanonicalTree, second: _CanonicalTree) -> bool:
    # Equality includes every retained base field and descendant-landmark
    # indicator. Ages/topology/features are never averaged or tolerance-merged.
    return all(torch.equal(getattr(first,name),getattr(second,name))
               for name in ('x','node_ids','clades','edges'))


def _canonical_tree(seq: GraphSequence, graph: int, times: Tensor, *,
                    time_column: int, branch_column: int,
                    time_scale: float, packed: list[int],
                    edge_pairs: list[tuple[int,int]]) -> _CanonicalTree:
    nodes = set(packed)
    if any(u == v for u,v in edge_pairs):
        raise ValueError('Observed tree self edges are inadmissible')
    if len(edge_pairs) != len(packed)-1:
        raise ValueError('Require one connected acyclic observed tree')
    adjacency = {u:[] for u in packed}
    for u,v in edge_pairs:
        if u not in nodes or v not in nodes:
            raise ValueError('Observed tree edge crosses graph boundaries')
        adjacency[u].append(v); adjacency[v].append(u)
        if not bool(times[u] != times[v]):
            raise ValueError('Observed branches require strictly positive decoded age differences')
    seen,stack = set(),[packed[0]]
    while stack:
        u=stack.pop()
        if u not in seen:
            seen.add(u); stack.extend(adjacency[u])
    if len(seen) != len(packed):
        raise ValueError('Observed tree is disconnected')
    parent,children = {},{u:[] for u in packed}
    roots=[]
    for u in packed:
        older=[v for v in adjacency[u] if bool(times[v] > times[u])]
        if len(older)>1:
            raise ValueError('Observed ages require one older parent per nonroot node')
        parent[u]=older[0] if older else -1
        if older: children[older[0]].append(u)
        else: roots.append(u)
    if len(roots)!=1:
        raise ValueError('Observed ages require exactly one rooted tree')
    root=roots[0]
    for u in packed:
        sampled = int(seq.node_ids[u]) >= 0
        if sampled and children[u]:
            raise ValueError('Persistent sample IDs must identify terminal observed leaves')
        if not sampled and not children[u]:
            raise ValueError('Anonymous terminal tips are not sampled observed leaves')
    removed = {u for u in packed if int(seq.node_ids[u]) < 0 and len(children[u]) == 1}
    retained = nodes-removed
    if not retained:
        raise ValueError('Canonical observed tree must retain sampled leaves')
    new_parent,new_children = {},{u:[] for u in retained}
    for u in retained:
        p=parent[u]
        while p in removed: p=parent[p]
        new_parent[u]=p
        if p!=-1: new_children[p].append(u)
    # A removed unary root chain is dangling beyond all observed branching.
    # The first retained descendant becomes the root, without clamping a value.
    new_roots=[u for u in retained if new_parent[u] == -1]
    if len(new_roots)!=1:
        raise ValueError('Unary suppression did not preserve one rooted tree')
    new_root=new_roots[0]
    preorder,stack=[],[new_root]
    while stack:
        u=stack.pop(); preorder.append(u); stack.extend(new_children[u])
    signature,height = {},{}
    for u in reversed(preorder):
        if int(seq.node_ids[u]) >= 0:
            signature[u]=(0,int(seq.node_ids[u])); height[u]=0
        else:
            signature[u]=(1,tuple(sorted(signature[v] for v in new_children[u])))
            height[u]=1+max(height[v] for v in new_children[u])
    # Leaf identities and recursive topology signatures are matching keys only.
    # No key, rank, source row, or node identity is appended to x.
    ordered=sorted(retained,key=lambda u:(height[u],signature[u]))
    lookup={node:index for index,node in enumerate(ordered)}
    rows=torch.tensor(ordered,dtype=torch.long)
    x=seq.x[rows].detach().clone()
    encoded_scale=math.log1p(time_scale)
    for index,u in enumerate(ordered):
        p=new_parent[u]
        if p == -1: x[index,branch_column]=0
        else:
            duration=times[p]-times[u]
            if not bool(torch.isfinite(duration)) or not bool(duration > 0):
                raise ValueError('Canonical branch duration must be positive and finite')
            x[index,branch_column]=torch.log1p(duration)/encoded_scale
    pairs=sorted((min(lookup[u],lookup[p]),max(lookup[u],lookup[p]))
                 for u,p in new_parent.items() if p!=-1)
    directed=sorted([pair for u,v in pairs for pair in ((u,v),(v,u))])
    edges=torch.tensor(directed,dtype=torch.long).reshape(-1,2).T.contiguous()
    root_chain=0; u=root
    while u in removed:
        root_chain+=1; u=children[u][0]
    return _CanonicalTree(x,seq.node_ids[rows].clone(),seq.clades[rows].clone(),edges,rows,
        seq.intervals[graph].clone(),[graph],len(removed)-root_chain,root_chain)


@torch.no_grad()
def canonicalize_observed_trees(seq: GraphSequenceInput, *, time_column: int = 6,
        parent_branch_column: int = 7, base_feature_count: int = 8,
        time_scale: float = 10000., merge_identical_adjacent: bool = False,
        source_provenance: Mapping | None = None) -> CanonicalizedObservedTrees:
    """Canonicalize the observed base prefix on CPU, preserving physical holds.

    Recompute ReferenceFeatureAdapter after this operation. Already appended
    summaries are intentionally discarded. Source provenance is copied only to
    an audit field; it is never used in the output graph or model features.
    """
    if not isinstance(getattr(seq,'x',None),Tensor) or seq.x.device.type != 'cpu' or seq.x.ndim != 2:
        raise ValueError('Canonicalize observed trees on CPU')
    if type(base_feature_count) is not int or not 1 <= base_feature_count <= seq.x.shape[1]:
        raise ValueError('base_feature_count must select an existing observed prefix')
    if type(time_column) is not int or type(parent_branch_column) is not int or not 0 <= time_column < base_feature_count or not 0 <= parent_branch_column < base_feature_count or time_column == parent_branch_column:
        raise ValueError('Declare distinct time and parent-branch columns inside the base prefix')
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError('time_scale must be finite and positive')
    if type(merge_identical_adjacent) is not bool:
        raise ValueError('merge_identical_adjacent must be boolean')
    if source_provenance is not None:
        if not isinstance(source_provenance,Mapping):
            raise ValueError('source_provenance must be a JSON-compatible mapping')
        try:
            saved_provenance=json.loads(json.dumps(dict(source_provenance),allow_nan=False))
        except (ValueError,TypeError) as exc:
            raise ValueError('source_provenance must be finite JSON-compatible audit data') from exc
    else:
        saved_provenance={}
    # Explicitly read only GraphSequence observed fields, never extra attributes
    # such as labels, population annotations, or truth-tree handles.
    base=GraphSequence(x=seq.x[:,:base_feature_count].detach().clone(),
        edge_index=seq.edge_index,node_graph=seq.node_graph,node_ids=seq.node_ids,
        clades=seq.clades,intervals=seq.intervals,query_node_ids=seq.query_node_ids,
        positions=seq.positions,target_graph=seq.target_graph,
        window_bounds=getattr(seq,'window_bounds',None))
    validate_model_input(base)
    if bool((base.x[:,time_column]<0).any()) or bool((base.x[:,parent_branch_column]<0).any()):
        raise ValueError('Observed encoded ages and parent branches must be nonnegative')
    times=torch.expm1(base.x[:,time_column].double()*math.log1p(time_scale))
    if not bool(torch.isfinite(times).all()) or bool((times<0).any()):
        raise ValueError('Decoded observed ages must be finite and nonnegative')
    chronological=torch.argsort(base.intervals[:,0]).tolist()
    # Group once instead of rescanning all packed nodes/edges for every hold.
    groups=[[] for _ in range(base.num_graphs)]
    edge_groups=[set() for _ in range(base.num_graphs)]
    graph_ids=base.node_graph.tolist()
    for row,graph in enumerate(graph_ids): groups[graph].append(row)
    for u,v in base.edge_index.T.tolist():
        edge_groups[graph_ids[u]].add((min(u,v),max(u,v)))
    canonical=[]
    removed_interior=removed_root=0
    for graph in chronological:
        tree=_canonical_tree(base,graph,times,time_column=time_column,
                             branch_column=parent_branch_column,time_scale=time_scale,
                             packed=groups[graph],edge_pairs=sorted(edge_groups[graph]))
        removed_interior+=tree.removed_unary_interior; removed_root+=tree.removed_unary_root
        if merge_identical_adjacent and canonical and torch.equal(canonical[-1].interval[1:],tree.interval[:1]) and _same_observed_tree(canonical[-1],tree):
            canonical[-1].interval[1]=tree.interval[1]
            canonical[-1].source_graphs.extend(tree.source_graphs)
        else: canonical.append(tree)
    graph_map=torch.empty(base.num_graphs,dtype=torch.long)
    offset=0; all_edges=[]; all_graphs=[]
    for graph,tree in enumerate(canonical):
        graph_map[tree.source_graphs]=graph
        all_edges.append(tree.edges+offset)
        all_graphs.append(torch.full((tree.x.shape[0],),graph,dtype=torch.long))
        offset+=tree.x.shape[0]
    output=GraphSequence(x=torch.cat([tree.x for tree in canonical]),
        edge_index=torch.cat(all_edges,dim=1),node_graph=torch.cat(all_graphs),
        node_ids=torch.cat([tree.node_ids for tree in canonical]),
        clades=torch.cat([tree.clades for tree in canonical]),
        intervals=torch.stack([tree.interval for tree in canonical]),
        query_node_ids=base.query_node_ids.detach().clone(),positions=base.positions.detach().clone(),
        target_graph=graph_map[base.target_graph].clone(),
        window_bounds=None if base.window_bounds is None else base.window_bounds.detach().clone())
    validate_model_input(output)
    provenance={
        'schema_version':1,'kind':'observed_tree_exact_canonicalization_comparator',
        'operator_note':'Preprocessing comparator: ordinary GNN/GRU/Transformer models after exact unary suppression and adjacent duplicate-hold merging may satisfy both representation contracts. Kernel contribution requires evidence beyond invariance.',
        'time_column':time_column,'parent_branch_column':parent_branch_column,
        'base_feature_count':base_feature_count,'input_feature_count':seq.x.shape[1],
        'discarded_appended_feature_count':seq.x.shape[1]-base_feature_count,
        'time_scale':float(time_scale),'time_encoding':'log1p(generations)/log1p(time_scale)',
        'parent_branch_rule':'decode endpoint ages, subtract parent minus child, then encode; canonical root branch zero',
        'merge_identical_adjacent':merge_identical_adjacent,
        'merge_rule':'exact canonical topology, retained base features, sampled IDs and descendant-landmark indicators; exact physical adjacency',
        'canonical_row_rule':'sampled leaf identity and recursive subtree signatures, leaves before ancestors, matching only',
        'learned_numeric_id_features':False,'truth_fields_used':[],
        'source_graph_count':base.num_graphs,'canonical_graph_count':output.num_graphs,
        'source_node_count':base.x.shape[0],'canonical_node_count':output.x.shape[0],
        'removed_unary_interior_nodes':removed_interior,'removed_unary_root_nodes':removed_root,
        'merged_duplicate_graph_records':base.num_graphs-output.num_graphs,
        'downstream_requirement':'recompute observed ReferenceFeatureAdapter after canonicalization',
        'source_provenance':saved_provenance,
    }
    return CanonicalizedObservedTrees(output,provenance,
        tuple(tuple(tree.source_graphs) for tree in canonical),
        torch.cat([tree.source_rows for tree in canonical]),graph_map)

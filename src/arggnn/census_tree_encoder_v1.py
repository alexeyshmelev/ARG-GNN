"""Inherited whole-tree GNN backbone for H021 source-only qualification.

Only observed F47, retained native ages/topology/types and endpoint edge durations
enter the encoder. Actual sample IDs perform gathers and are never embeddings.
No data loader, scaler join, training entry point or new backbone novelty claim.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .census_cut_exchange_v1 import (
    CensusCutExchangeLayerV1, CensusExchangeBankV1, ObservedDatedTreeV1,
    WholeTreeEncoderInputsV1, gather_whole_tree_context_v1,
    prepare_whole_tree_encoder_inputs_v1, build_census_cut_partition_v1,
    )



@dataclass(frozen=True)
class ConnectedObservedTreeV1:
    tree: ObservedDatedTreeV1
    features: Tensor
    native_vertex_index: Tensor    # retained union index -> supplied marginal-node index
    excluded_non_sample_nodes: int # administrative count, never an encoder feature


def connected_observed_tree_v1(tree: ObservedDatedTreeV1, features: Tensor) -> ConnectedObservedTreeV1:
    """Extract only the actual sample-ancestor union from a marginal-tree table.

    A TS can store isolated non-sample nodes for other marginal trees. Those rows
    do not participate in this tree's graph, features, cut, or neural encoding.
    All actual samples must still belong to one complete connected dated tree.
    This tensor adapter is not a qualified tskit/native-bank/scaler loader.
    """
    parent, times = tree.parent, tree.node_times
    n = len(parent)
    if (parent.device.type != "cpu" or parent.dtype != torch.long or parent.ndim != 1
            or not n or times.device.type != "cpu" or times.dtype != torch.float64
            or times.requires_grad or times.shape != (n,)
            or features.device.type != "cpu" or features.shape != (n, 47)):
        raise ValueError("Marginal input requires fixed CPU native geometry and observed F47 rows")
    if (tree.sample_ids.device.type != "cpu" or tree.sample_ids.dtype != torch.long
            or tree.sample_ids.ndim != 1 or tree.sample_nodes.device.type != "cpu"
            or tree.sample_nodes.dtype != torch.long or tree.sample_nodes.shape != tree.sample_ids.shape
            or not tree.sample_ids.numel() or bool((tree.sample_ids < 0).any())
            or bool(((tree.sample_nodes < 0) | (tree.sample_nodes >= n)).any())
            or len(set(tree.sample_ids.tolist())) != len(tree.sample_ids)
            or len(set(tree.sample_nodes.tolist())) != len(tree.sample_nodes)):
        raise ValueError("Marginal sample identities/native-vertex join is invalid")
    union = set()
    parents = parent.tolist()
    for sample in tree.sample_nodes.tolist():
        node, path = sample, set()
        while node >= 0:
            if node >= n or node in path:
                raise ValueError("Sample-ancestor path has an invalid parent or cycle")
            path.add(node)
            union.add(node)
            node = parents[node]
            if node < -1:
                raise ValueError("Sample-ancestor path has an invalid parent")
    native = torch.tensor(sorted(union), dtype=torch.long)
    inverse = {node: rank for rank, node in enumerate(native.tolist())}
    selected = ObservedDatedTreeV1(
        torch.tensor([-1 if parents[node] < 0 else inverse[parents[node]] for node in native.tolist()], dtype=torch.long),
        times[native].clone(), tree.sample_ids.clone(),
        torch.tensor([inverse[node] for node in tree.sample_nodes.tolist()], dtype=torch.long))
    selected_features = features[native].clone()
    # Reuse the frozen connected/dating/leaf-only/F47 gate. This checks the union
    # before it is exposed to either the census core or the neural encoder.
    prepare_whole_tree_encoder_inputs_v1(selected, selected_features)
    return ConnectedObservedTreeV1(selected, selected_features, native, n-len(native))


def prepare_connected_whole_tree_inputs_v1(tree: ObservedDatedTreeV1, features: Tensor) -> WholeTreeEncoderInputsV1:
    """Select the sample-ancestor union, then suppress only anonymous unary nodes."""
    connected = connected_observed_tree_v1(tree, features)
    prepared = prepare_whole_tree_encoder_inputs_v1(connected.tree, connected.features)
    return replace(prepared, native_vertex_index=connected.native_vertex_index[prepared.native_vertex_index])


def _checked_inputs(inputs: WholeTreeEncoderInputsV1, time_scale: float):
    """Recheck the prepared representation without accepting sampled interiors."""
    n = len(inputs.parent)
    if (inputs.native_vertex_index.device.type != "cpu"
            or inputs.native_vertex_index.dtype != torch.long
            or inputs.native_vertex_index.shape != (n,)
            or bool((inputs.native_vertex_index < 0).any())
            or len(set(inputs.native_vertex_index.tolist())) != n):
        raise ValueError("Retained native vertex provenance must be unique CPU indexes")
    tree = ObservedDatedTreeV1(inputs.parent, inputs.node_times, inputs.sample_ids, inputs.sample_nodes)
    checked = prepare_whole_tree_encoder_inputs_v1(tree, inputs.features, time_scale=time_scale)
    if (len(checked.parent) != n or not torch.equal(checked.parent, inputs.parent)
            or not torch.equal(checked.sample_nodes, inputs.sample_nodes)):
        raise ValueError("Encoder requires the checked unary-suppressed retained representation")
    if (inputs.branch_duration.device.type != "cpu" or inputs.branch_duration.dtype != torch.float64
            or inputs.branch_duration.requires_grad or inputs.branch_duration.shape != (n,)
            or not torch.equal(inputs.branch_duration, checked.branch_duration)):
        raise ValueError("Retained branch durations must equal fixed native endpoint differences")
    children = [[] for _ in range(n)]
    for node, parent in enumerate(inputs.parent.tolist()):
        if parent >= 0:
            children[parent].append(node)
    if (sorted(inputs.upward_order) != list(range(n))
            or sorted(inputs.downward_order) != list(range(n))):
        raise ValueError("Each tree sweep must process every retained node exactly once")
    up_rank = {node: rank for rank, node in enumerate(inputs.upward_order)}
    down_rank = {node: rank for rank, node in enumerate(inputs.downward_order)}
    for child, parent in enumerate(inputs.parent.tolist()):
        if parent >= 0 and (up_rank[child] >= up_rank[parent] or down_rank[parent] >= down_rank[child]):
            raise ValueError("Tree sweeps must respect every retained parent-child edge")
    return children


@dataclass(frozen=True)
class RetainedObservedFeaturesV1:
    node: Tensor                   # [N,51] = stored observed F47 plus age/type4
    edge_time: Tensor              # [N,1] = encoded retained endpoint duration
    children: tuple[tuple[int, ...], ...]


def retained_observed_features_v1(inputs: WholeTreeEncoderInputsV1, *,
        time_scale: float = 10000.) -> RetainedObservedFeaturesV1:
    """Expose the exact observation-only node/edge features used by the encoder.

    The four extra node channels are encoded native age, root flag, leaf flag,
    and branching flag. No roster role, local/native index, sample identity or
    population metadata becomes a learned feature. F47 provenance remains a
    caller obligation; only its shape, finite values and encoded ages are checked.
    """
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("Observed time scale must be finite and positive")
    children = _checked_inputs(inputs, time_scale)
    age = torch.log1p(inputs.node_times) / math.log1p(time_scale)
    root = (inputs.parent == -1).to(torch.float64)
    leaf = torch.tensor([not values for values in children], dtype=torch.float64)
    branching = torch.tensor([len(values) > 1 for values in children], dtype=torch.float64)
    auxiliary = torch.stack((age, root, leaf, branching), -1).to(inputs.features.dtype)
    node = torch.cat((inputs.features, auxiliary), -1)
    edge = (torch.log1p(inputs.branch_duration) / math.log1p(time_scale))[:, None]
    return RetainedObservedFeaturesV1(node, edge, tuple(tuple(values) for values in children))


@dataclass(frozen=True)
class RetainedObservedForestV1:
    node: Tensor
    edge_time: Tensor
    parent: Tensor
    child_count: Tensor
    graph_offsets: tuple[int, ...]
    upward_levels: tuple[Tensor, ...]
    upward_children: tuple[Tensor, ...]
    upward_parent_rank: tuple[Tensor, ...]
    downward_levels: tuple[Tensor, ...]

    def to(self, device, dtype):
        return replace(self, node=self.node.to(device=device,dtype=dtype),
            edge_time=self.edge_time.to(device=device,dtype=dtype), parent=self.parent.to(device),
            child_count=self.child_count.to(device=device,dtype=dtype),
            upward_levels=tuple(x.to(device) for x in self.upward_levels),
            upward_children=tuple(x.to(device) for x in self.upward_children),
            upward_parent_rank=tuple(x.to(device) for x in self.upward_parent_rank),
            downward_levels=tuple(x.to(device) for x in self.downward_levels))


def prepare_retained_observed_forest_v1(inputs_by_graph: Sequence[WholeTreeEncoderInputsV1], *,
        time_scale: float = 10000.) -> RetainedObservedForestV1:
    """Checked CPU height/depth batches, including every edge in every whole tree.

    Forests batch independent trees at the same level. Administrative Python
    topology loops occur only in preparation, outside all neural vertex actions.
    The returned geometry can be cached for already qualified inputs.
    """
    if not inputs_by_graph:
        raise ValueError("Require at least one complete retained tree")
    nodes, edges, parent, child_count, children, height, depth = [], [], [], [], [], [], []
    offsets = [0]
    for inputs in inputs_by_graph:
        observed = retained_observed_features_v1(inputs, time_scale=time_scale)
        n, offset = len(inputs.parent), offsets[-1]
        local_height, local_depth = [0]*n, [0]*n
        for node in inputs.upward_order:
            values = observed.children[node]
            if values:
                local_height[node] = 1+max(local_height[child] for child in values)
        for node in inputs.downward_order:
            ancestor = int(inputs.parent[node])
            if ancestor >= 0:
                local_depth[node] = local_depth[ancestor]+1
        nodes.append(observed.node);edges.append(observed.edge_time)
        parent.extend(-1 if value<0 else value+offset for value in inputs.parent.tolist())
        child_count.extend(len(values) for values in observed.children)
        children.extend(tuple(child+offset for child in values) for values in observed.children)
        height.extend(local_height);depth.extend(local_depth);offsets.append(offset+n)
    up_levels, up_children, up_rank = [], [], []
    for level in range(max(height)+1):
        members = [node for node,value in enumerate(height) if value==level]
        level_children, ranks = [], []
        for rank,node in enumerate(members):
            level_children.extend(children[node]);ranks.extend([rank]*len(children[node]))
        up_levels.append(torch.tensor(members,dtype=torch.long))
        up_children.append(torch.tensor(level_children,dtype=torch.long))
        up_rank.append(torch.tensor(ranks,dtype=torch.long))
    down_levels = tuple(torch.tensor([node for node,value in enumerate(depth) if value==level],dtype=torch.long)
                        for level in range(max(depth)+1))
    return RetainedObservedForestV1(torch.cat(nodes),torch.cat(edges),torch.tensor(parent,dtype=torch.long),
        torch.tensor(child_count,dtype=torch.float64),tuple(offsets),tuple(up_levels),tuple(up_children),tuple(up_rank),down_levels)


class CensusWholeTreeEncoderV1(nn.Module):
    """Inherited mean-message tree GNN, batched by forest height/depth levels.

    Each retained node and edge contributes once per sweep. Neural operations
    are tensor batches for whole levels, never a separate Python neural call per
    vertex. Actual device efficiency and autograd memory are still unmeasured.
    """
    def __init__(self, hidden_dim: int = 32, *, time_scale: float = 10000.):
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim < 1 or not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Encoder dimensions/time scale must be positive")
        self.hidden_dim = hidden_dim
        self.time_scale = float(time_scale)
        self.input_projection = nn.Linear(51, hidden_dim)
        self.up_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_edge = nn.Linear(1, hidden_dim, bias=False)
        self.up_update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.down_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_edge = nn.Linear(1, hidden_dim, bias=False)
        self.down_update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward_forest(self, inputs: Sequence[WholeTreeEncoderInputsV1] | RetainedObservedForestV1) -> tuple[Tensor, ...]:
        forest = inputs if isinstance(inputs,RetainedObservedForestV1) else prepare_retained_observed_forest_v1(inputs,time_scale=self.time_scale)
        parameter = self.input_projection.weight
        forest = forest.to(parameter.device,parameter.dtype)
        base = self.input_projection(forest.node).tanh()
        upward = torch.zeros_like(base)
        for members,children,ranks in zip(forest.upward_levels,forest.upward_children,forest.upward_parent_rank):
            aggregate = base.new_zeros((len(members),self.hidden_dim))
            if children.numel():
                messages = self.up_message(upward[children])+self.up_edge(forest.edge_time[children])
                aggregate = aggregate.index_add(0,ranks,messages)
                aggregate = aggregate/forest.child_count[members,None]
            updated = self.up_update(torch.cat((base[members],aggregate),-1)).tanh()
            upward = upward.index_copy(0,members,updated)
        downward = torch.zeros_like(base)
        for level,members in enumerate(forest.downward_levels):
            if level:
                context = self.down_message(downward[forest.parent[members]])+self.down_edge(forest.edge_time[members])
            else:
                context = base.new_zeros((len(members),self.hidden_dim))
            updated = self.down_update(torch.cat((upward[members],context),-1)).tanh()
            downward = downward.index_copy(0,members,updated)
        encoded = self.output_projection(torch.cat((upward,downward),-1)).tanh()
        if not bool(torch.isfinite(encoded).all()):
            raise ValueError("Whole-tree encoded contexts are nonfinite")
        return tuple(encoded[left:right] for left,right in zip(forest.graph_offsets[:-1],forest.graph_offsets[1:]))

    def forward(self, inputs: WholeTreeEncoderInputsV1 | Sequence[WholeTreeEncoderInputsV1] | RetainedObservedForestV1):
        if isinstance(inputs,WholeTreeEncoderInputsV1):
            return self.forward_forest([inputs])[0]
        return self.forward_forest(inputs)


def encode_whole_tree_contexts_v1(encoder: CensusWholeTreeEncoderV1,
        inputs_by_graph: Sequence[WholeTreeEncoderInputsV1], query_ids: Tensor,
        reference_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Process complete retained trees and gather Q/R with actual identities only."""
    if not inputs_by_graph:
        raise ValueError("Require at least one complete retained tree")
    query_contexts, reference_contexts = [], []
    dates = None
    encoded_by_graph = encoder(inputs_by_graph)
    for inputs, encoded in zip(inputs_by_graph, encoded_by_graph):
        query, reference = gather_whole_tree_context_v1(inputs, encoded, query_ids, reference_ids)
        lookup = dict(zip(inputs.sample_ids.tolist(), inputs.sample_nodes.tolist()))
        observed_dates = inputs.node_times[torch.tensor([lookup[q] for q in query_ids.tolist() + reference_ids.tolist()])]
        if dates is not None and not torch.equal(observed_dates, dates):
            raise ValueError("Persistent observed sample dates changed across encoder inputs")
        dates = observed_dates
        query_contexts.append(query)
        reference_contexts.append(reference)
    return torch.stack(query_contexts), torch.stack(reference_contexts)


@dataclass(frozen=True)
class CensusTreeGNNOutputV1:
    logits: Tensor                  # [P, scoredQ, C], no labels are supplied
    scored_query_ids: Tensor        # actual IDs, identity join only
    anchor_hidden: Tensor          # [A,Q,D], dynamics roster preserved
    target_hidden: Tensor          # [P,Q,D]
    encoder_query_context: Tensor  # [G,Q,D]
    encoder_reference_context: Tensor  # [G,R,D]


class CensusTreeGNNV1(nn.Module):
    """One common inherited backbone/projections/gates/decoder for all H021 controls.

    Control selection changes only the declared exchange action or the explicit
    pointwise contrast. Whole-tree encoder, layers, source projections and scored
    decoder are literally the same parameter objects on every forward call.
    """
    def __init__(self, hidden_dim: int = 32, latent_dim: int = 8, num_classes: int = 4,
            num_layers: int = 2, num_branches: int = 3, *,
            initial_exchange=(5., 20., 80.), initial_damping=1., initial_eta=1.):
        super().__init__()
        if (not isinstance(num_classes, int) or num_classes < 1
                or not isinstance(num_layers, int) or num_layers < 1):
            raise ValueError("GNN class count and depth must be positive")
        self.encoder = CensusWholeTreeEncoderV1(hidden_dim)
        self.layers = nn.ModuleList([CensusCutExchangeLayerV1(hidden_dim, latent_dim, num_branches,
            initial_exchange=initial_exchange, initial_damping=initial_damping, initial_eta=initial_eta)
            for _ in range(num_layers)])
        self.decoder = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs_by_graph: Sequence[WholeTreeEncoderInputsV1],
            bank: CensusExchangeBankV1, *, control: str = "proposed",
            readout_chunk_size: int = 128) -> CensusTreeGNNOutputV1:
        if control not in {"proposed", "eta0", "permuted_groups", "raw_diagonal", "local"}:
            raise ValueError("Unknown H021 control")
        if len(inputs_by_graph) != bank.layout.num_graphs or len(bank.partitions) != len(inputs_by_graph):
            raise ValueError("Whole-tree inputs must match every serialized observed graph")
        queries = bank.query_ids.detach().cpu()
        references = bank.reference_ids.detach().cpu()
        if control in {"proposed", "eta0", "raw_diagonal"} and bank.membership_control != "observed_census_cut":
            raise ValueError("Observed H021 controls require observed census memberships")
        for graph, (inputs, partition) in enumerate(zip(inputs_by_graph, bank.partitions)):
            lookup = dict(zip(inputs.sample_ids.tolist(), inputs.sample_nodes.tolist()))
            if (not set(queries.tolist()) <= set(lookup)
                    or not torch.equal(inputs.node_times[torch.tensor([lookup[q] for q in queries.tolist()])],
                                       partition.query_ages)):
                raise ValueError("Encoder queries/native ages disagree with the checked census bank")
            tree = ObservedDatedTreeV1(inputs.parent, inputs.node_times, inputs.sample_ids, inputs.sample_nodes)
            observed = build_census_cut_partition_v1(tree, queries, references,
                queries[partition.scored_query_mask], census=partition.census,
                context_date_policy=partition.context_date_policy)
            if bank.membership_control == "observed_census_cut" and set(observed.group_members) != set(partition.group_members):
                raise ValueError("Encoder topology disagrees with observed census groups")
            # Donor banks must have been qualified against these observed trees
            # during preparation. No Python Q-by-R MRCA loop occurs in forward.
        query_context, reference_context = encode_whole_tree_contexts_v1(self.encoder, inputs_by_graph, queries, references)
        parameter = self.decoder.weight
        work_bank = bank.to(parameter.device, parameter.dtype)
        anchor = query_context[work_bank.layout.anchor_graph]
        target = query_context[work_bank.target_graph]
        anchor_reference = reference_context[work_bank.layout.anchor_graph]
        target_reference = reference_context[work_bank.target_graph]
        for layer in self.layers:
            anchor, target = layer(work_bank, anchor, anchor_reference, target, control=control,
                target_reference_context=target_reference, readout_chunk_size=readout_chunk_size)
        scored = bank.partitions[0].scored_query_mask
        if any(not torch.equal(part.scored_query_mask, scored) for part in bank.partitions):
            raise ValueError("Scored query roster changed across observed graphs")
        scored_device = scored.to(device=target.device)
        logits = self.decoder(target[:, scored_device])
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Whole-tree GNN logits are nonfinite")
        return CensusTreeGNNOutputV1(logits, queries[scored], anchor, target, query_context, reference_context)

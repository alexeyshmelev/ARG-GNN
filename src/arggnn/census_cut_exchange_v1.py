"""Observed census-cut exchange prototype, H021 source-only condition.

Block averaging/diffusion, fixed TMRCA donor pooling, finite impulses and gated
updates are inherited. Census-cut/soft-turnover biological utility is unvalidated.
Local group IDs are not globally persistent ancestral IDs. No ARGformer identity
mapping is fabricated. This module has no training/data/runtime entry point.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .anchor_kernel import AnchorLayerMessages
from .anchor_spectral_compact_v1 import CompactAnchorLayoutV1, prepare_compact_anchor_layout_v1


@dataclass(frozen=True)
class ObservedDatedTreeV1:
    parent: Tensor                 # [N], local native vertex indexes, root=-1
    node_times: Tensor             # [N], native observed generations, CPU float64
    sample_ids: Tensor             # [B], actual persistent sample identities
    sample_nodes: Tensor           # [B], local native vertex indexes


@dataclass(frozen=True)
class CensusCutPartitionV1:
    query_ids: Tensor
    group_index: Tensor            # [Q], local labels only, never cross-tree IDs
    group_size_by_query: Tensor    # [Q], exact group cardinalities
    group_members: tuple[frozenset[int], ...]  # actual query IDs, not node IDs
    query_ages: Tensor             # [Q], unchanged native float64 generations
    scored_query_mask: Tensor
    older_reference_ids: tuple[int, ...]
    census: float
    context_date_policy: str


def _ids(value: Tensor, name: str) -> list[int]:
    if value.device.type != "cpu" or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(name + " must be CPU long [B]")
    host = value.tolist()
    if len(host) != len(set(host)) or any(x < 0 for x in host):
        raise ValueError(name + " must be unique nonnegative actual sample identities")
    return host


def _checked_tree(tree: ObservedDatedTreeV1):
    parent, times = tree.parent, tree.node_times
    if (parent.device.type != "cpu" or parent.dtype != torch.long or parent.ndim != 1
            or not parent.numel() or times.device.type != "cpu"
            or times.dtype != torch.float64 or times.shape != parent.shape
            or times.requires_grad):
        raise ValueError("Require fixed CPU native parents and float64 observed times")
    if not bool(torch.isfinite(times).all() & (times >= 0).all()):
        raise ValueError("Native observed times must be finite and nonnegative")
    n = len(parent)
    if bool(((parent < -1) | (parent >= n)).any()):
        raise ValueError("Native parent vertex is out of range")
    roots = torch.nonzero(parent == -1, as_tuple=False).flatten().tolist()
    if len(roots) != 1:
        raise ValueError("Require a single complete observed tree")
    root = roots[0]
    children = [[] for _ in range(n)]
    for child, ancestor in enumerate(parent.tolist()):
        if ancestor >= 0:
            if ancestor == child or times[ancestor] < times[child]:
                raise ValueError("Native parent time must not precede child time")
            children[ancestor].append(child)
    preorder, postorder, seen = [], [], set()
    stack = [(root, False)]
    while stack:
        node, finish = stack.pop()
        if finish:
            postorder.append(node)
            continue
        if node in seen:
            raise ValueError("Observed tree contains a cycle")
        seen.add(node)
        preorder.append(node)
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(children[node]))
    if len(seen) != n:
        raise ValueError("Observed tree has disconnected vertices or a cycle")
    sample_ids = _ids(tree.sample_ids, "sample_ids")
    sample_nodes = _ids(tree.sample_nodes, "sample_nodes")
    if len(sample_ids) != len(sample_nodes) or any(x < 0 or x >= n for x in sample_nodes):
        raise ValueError("Sample identity/native-vertex join is invalid")
    leaves = {i for i, values in enumerate(children) if not values}
    if set(sample_nodes) != leaves:
        raise ValueError("Require all and only observed sampled leaves")
    lookup = dict(zip(sample_ids, sample_nodes))
    return root, children, preorder, postorder, lookup


def build_census_cut_partition_v1(tree: ObservedDatedTreeV1, query_ids: Tensor,
        reference_ids: Tensor, scored_query_ids: Tensor, *, census: float = 350.,
        context_date_policy: str = "require_younger") -> CensusCutPartitionV1:
    """Partition eligible Q by observed MRCA <= census, with explicit date gates.

    Every dynamics query, scored or context, must be strictly younger than the
    census. Older/equal context is rejected rather than given a fictitious hard
    census group. Pure references may be older and remain outside the cut.
    """
    if not math.isfinite(census) or census <= 0:
        raise ValueError("Census must be finite and positive")
    if context_date_policy != "require_younger":
        raise ValueError("Only checked require_younger context policy is implemented")
    _, _, _, _, lookup = _checked_tree(tree)
    queries = _ids(query_ids, "query_ids")
    references = _ids(reference_ids, "reference_ids")
    scored = _ids(scored_query_ids, "scored_query_ids")
    if (not queries or not references or set(queries) & set(references)
            or set(queries) | set(references) != set(lookup)
            or not scored or not set(scored) <= set(queries)):
        raise ValueError("Q/R/scored roles must form the declared observed sample partition")
    ages = tree.node_times[torch.tensor([lookup[q] for q in queries], dtype=torch.long)]
    scored_mask = torch.tensor([q in set(scored) for q in queries], dtype=torch.bool)
    if bool((ages[scored_mask] >= census).any()):
        raise ValueError("Scored queries must be strictly younger than the census")
    if bool((ages[~scored_mask] >= census).any()):
        raise ValueError("Context queries fail checked require_younger census policy")
    keys = []
    parents = tree.parent.tolist()
    times = tree.node_times.tolist()
    for query in queries:
        node = lookup[query]
        while parents[node] >= 0 and times[parents[node]] <= census:
            node = parents[node]
        keys.append(node)
    local_labels = {}
    index = []
    for key in keys:
        if key not in local_labels:
            local_labels[key] = len(local_labels)
        index.append(local_labels[key])
    group_index = torch.tensor(index, dtype=torch.long)
    counts = torch.bincount(group_index, minlength=len(local_labels))
    members = tuple(frozenset(q for q, group in zip(queries, index) if group == label)
                    for label in range(len(local_labels)))
    older = tuple(r for r in references if times[lookup[r]] >= census)
    return CensusCutPartitionV1(query_ids.clone(), group_index, counts[group_index].double(),
                                members, ages.clone(), scored_mask, older, census, context_date_policy)


def clade_turnover_rho_v1(left: CensusCutPartitionV1, right: CensusCutPartitionV1) -> Tensor:
    """Jaccard retention in right query-ID order, using actual sample sets only."""
    if (left.census != right.census or set(left.query_ids.tolist()) != set(right.query_ids.tolist())):
        raise ValueError("Retention requires the same census and persistent query samples")
    def groups(partition):
        result = {}
        for query, label in zip(partition.query_ids.tolist(), partition.group_index.tolist()):
            members = partition.group_members[label]
            if query not in members:
                raise ValueError("Query must belong to its own local clade")
            result[query] = members
        return result
    before, after = groups(left), groups(right)
    rho = torch.tensor([len(before[q] & after[q]) / len(before[q] | after[q])
                        for q in right.query_ids.tolist()], dtype=torch.float64)
    if not bool((rho > 0).all() & (rho <= 1).all()):
        raise ValueError("Sample-aligned retention must be strictly positive and <=1")
    return rho


def observed_tmrca_donor_weights_v1(tree: ObservedDatedTreeV1, query_ids: Tensor,
        reference_ids: Tensor, *, sigmas: Sequence[float] = (.05, .1, .2),
        time_scale: float = 10000.) -> Tensor:
    """Inherited fixed observed-TMRCA donor softmax, including older R proxies."""
    _, _, _, _, lookup = _checked_tree(tree)
    queries, references = _ids(query_ids, "query_ids"), _ids(reference_ids, "reference_ids")
    if (not queries or not references or not set(queries + references) <= set(lookup)
            or set(queries) & set(references)):
        raise ValueError("Donor roles must be disjoint observed samples")
    sigma = torch.tensor(tuple(sigmas), dtype=torch.float64)
    if (sigma.ndim != 1 or not sigma.numel() or not bool(torch.isfinite(sigma).all() & (sigma > 0).all())
            or not math.isfinite(time_scale) or time_scale <= 0):
        raise ValueError("Fixed donor scales must be finite and positive")
    parents = tree.parent.tolist()
    times = tree.node_times.tolist()
    tmrca = []
    for query in queries:
        ancestors = set()
        node = lookup[query]
        while node >= 0:
            ancestors.add(node)
            node = parents[node]
        row = []
        for reference in references:
            node = lookup[reference]
            while node not in ancestors:
                node = parents[node]
            row.append(times[node])
        tmrca.append(row)
    encoded = torch.log1p(torch.tensor(tmrca, dtype=torch.float64)) / math.log1p(time_scale)
    return torch.softmax(-encoded[None] / sigma[:, None, None], dim=-1)


@dataclass(frozen=True)
class WholeTreeEncoderInputsV1:
    parent: Tensor
    node_times: Tensor
    features: Tensor              # observed F47 at retained native vertices
    branch_duration: Tensor       # parent-child native endpoint difference
    sample_ids: Tensor
    sample_nodes: Tensor           # into retained vertices
    native_vertex_index: Tensor    # graph-local provenance, not persistent IDs
    upward_order: tuple[int, ...]
    downward_order: tuple[int, ...]


def prepare_whole_tree_encoder_inputs_v1(tree: ObservedDatedTreeV1, features: Tensor, *,
        encoded_age_column: int = 6, time_scale: float = 10000.) -> WholeTreeEncoderInputsV1:
    """Interface for ordinary whole-tree upward/downward encoding, no leaf-only replacement.

    Suppress anonymous nonroot unary vertices. Retain root, all sampled leaves
    and every branching interior. Preserve stored F47 and native endpoint ages.
    The original tree is not mutated; no hidden population/census labels enter.
    """
    root, children, preorder, postorder, _ = _checked_tree(tree)
    if (features.device.type != "cpu" or features.shape != (len(tree.parent), 47)
            or not features.is_floating_point() or not bool(torch.isfinite(features).all())
            or not 0 <= encoded_age_column < 47 or time_scale <= 0 or not math.isfinite(time_scale)):
        raise ValueError("Require observed CPU F47 with a declared stored encoded-age column")
    expected_age = (torch.log1p(tree.node_times) / math.log1p(time_scale)).to(features.dtype)
    if not torch.allclose(features[:, encoded_age_column], expected_age, rtol=0.,
                          atol=8 * torch.finfo(features.dtype).eps):
        raise ValueError("Stored encoded ages disagree with native observed times")
    sampled = set(tree.sample_nodes.tolist())
    keep = [node for node in range(len(tree.parent))
            if node == root or node in sampled or len(children[node]) != 1]
    inverse = {node: rank for rank, node in enumerate(keep)}
    parent = []
    duration = []
    native_parent = tree.parent.tolist()
    for node in keep:
        ancestor = native_parent[node]
        while ancestor >= 0 and ancestor not in inverse:
            ancestor = native_parent[ancestor]
        parent.append(-1 if ancestor < 0 else inverse[ancestor])
        duration.append(0. if ancestor < 0 else float(tree.node_times[ancestor] - tree.node_times[node]))
    ids = torch.tensor(keep, dtype=torch.long)
    return WholeTreeEncoderInputsV1(
        torch.tensor(parent, dtype=torch.long), tree.node_times[ids].clone(), features[ids].clone(),
        torch.tensor(duration, dtype=torch.float64), tree.sample_ids.clone(),
        torch.tensor([inverse[node] for node in tree.sample_nodes.tolist()], dtype=torch.long), ids,
        tuple(inverse[node] for node in postorder if node in inverse),
        tuple(inverse[node] for node in preorder if node in inverse))


def gather_whole_tree_context_v1(inputs: WholeTreeEncoderInputsV1, encoded_vertices: Tensor,
        query_ids: Tensor, reference_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Gather externally encoded whole-tree contexts by actual sample identity."""
    if (encoded_vertices.ndim != 2 or encoded_vertices.shape[0] != len(inputs.parent)
            or not bool(torch.isfinite(encoded_vertices).all())):
        raise ValueError("Whole-tree encoder output is malformed/nonfinite")
    lookup = dict(zip(inputs.sample_ids.tolist(), inputs.sample_nodes.tolist()))
    queries, references = _ids(query_ids, "query_ids"), _ids(reference_ids, "reference_ids")
    if set(queries) & set(references) or set(queries + references) != set(lookup):
        raise ValueError("Whole-tree context requires the checked Q/R roster")
    query_nodes = torch.tensor([lookup[q] for q in queries], device=encoded_vertices.device)
    reference_nodes = torch.tensor([lookup[r] for r in references], device=encoded_vertices.device)
    return encoded_vertices[query_nodes], encoded_vertices[reference_nodes]


@dataclass(frozen=True)
class CensusExchangeBankV1:
    layout: CompactAnchorLayoutV1
    partitions: tuple[CensusCutPartitionV1, ...]
    group_index: Tensor            # [G,Q], local group labels
    group_size_by_query: Tensor    # [G,Q], not a dense projection matrix
    donor_weights: Tensor          # [G,S,Q,R], fixed observed proxy map
    boundary_rho: Tensor           # [K,Q], ones at non-boundary compact events
    query_ids: Tensor
    reference_ids: Tensor
    target_graph: Tensor
    sigmas: tuple[float, ...]
    membership_control: str = "observed_census_cut"

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> CensusExchangeBankV1:
        return replace(self, layout=self.layout.to(device),
            group_index=self.group_index.to(device),
            group_size_by_query=self.group_size_by_query.to(device=device, dtype=dtype),
            donor_weights=self.donor_weights.to(device=device, dtype=dtype),
            boundary_rho=self.boundary_rho.to(device=device, dtype=dtype),
            query_ids=self.query_ids.to(device), reference_ids=self.reference_ids.to(device),
            target_graph=self.target_graph.to(device))


def _boundary_rho(layout, partitions):
    q = len(partitions[0].query_ids)
    rows = torch.ones((len(layout.source_at_event), q), dtype=torch.float64)
    for event in range(1, layout.num_segments):
        before, after = layout.graph_at_segment[event - 1:event + 1]
        if before != after:
            rows[event] = clade_turnover_rho_v1(partitions[before], partitions[after])
    return rows


def prepare_census_exchange_bank_v1(trees: Sequence[ObservedDatedTreeV1], intervals: Tensor,
        positions: Tensor, query_ids: Tensor, reference_ids: Tensor, scored_query_ids: Tensor, *,
        census: float = 350., context_date_policy: str = "require_younger", num_anchors: int = 32,
        sigmas: Sequence[float] = (.05, .1, .2)) -> CensusExchangeBankV1:
    """Observed-only CPU preparation, preserving serialized graph IDs and native dates."""
    if len(trees) != len(intervals) or not trees:
        raise ValueError("One complete observed tree is required per physical hold")
    partitions = tuple(build_census_cut_partition_v1(tree, query_ids, reference_ids, scored_query_ids,
        census=census, context_date_policy=context_date_policy) for tree in trees)
    sample_times = []
    for tree in trees:
        _, _, _, _, lookup = _checked_tree(tree)
        sample_times.append(tree.node_times[torch.tensor([lookup[x] for x in query_ids.tolist() + reference_ids.tolist()])])
    if any(not torch.equal(x, sample_times[0]) for x in sample_times[1:]):
        raise ValueError("Persistent observed sample dates changed across graphs")
    layout = prepare_compact_anchor_layout_v1(intervals, positions, num_anchors=num_anchors)
    order = torch.argsort(intervals[:, 0])
    target_graph = order[torch.searchsorted(intervals[order, 0].contiguous().double(),
                                           positions.double(), right=True) - 1]
    weights = torch.stack([observed_tmrca_donor_weights_v1(tree, query_ids, reference_ids, sigmas=sigmas)
                           for tree in trees])
    return CensusExchangeBankV1(layout, partitions,
        torch.stack([part.group_index for part in partitions]),
        torch.stack([part.group_size_by_query for part in partitions]), weights,
        _boundary_rho(layout, partitions), query_ids.clone(), reference_ids.clone(), target_graph,
        tuple(map(float, sigmas)))


def permuted_group_control_v1(bank: CensusExchangeBankV1, permutation: Tensor) -> CensusExchangeBankV1:
    """One fixed global reassignment of memberships, leaving samples/donors unchanged.

    This destroys biological assignment while preserving group-size and turnover
    distributions. It does not relabel actual source signals or reference IDs.
    """
    if bank.group_index.device.type != "cpu":
        raise ValueError("Prepare the matched global permutation on CPU")
    order = _ids(permutation, "permutation")
    q = len(bank.query_ids)
    if sorted(order) != list(range(q)):
        raise ValueError("Membership control requires one global Q permutation")
    groups = bank.group_index[:, permutation]
    queries = bank.query_ids.tolist()
    parts = []
    for graph, part in enumerate(bank.partitions):
        labels = groups[graph].tolist()
        members = tuple(frozenset(query for query, label in zip(queries, labels) if label == group)
                        for group in range(len(part.group_members)))
        parts.append(replace(part, group_index=groups[graph].clone(),
                             group_size_by_query=part.group_size_by_query[permutation].clone(),
                             group_members=members))
    parts = tuple(parts)
    return replace(bank, partitions=parts, group_index=groups.clone(),
                   group_size_by_query=bank.group_size_by_query[:, permutation].clone(),
                   boundary_rho=_boundary_rho(bank.layout, parts).to(
                       device=bank.boundary_rho.device, dtype=bank.boundary_rho.dtype),
                   membership_control="fixed_global_group_permutation")


def block_average_v1(state: Tensor, group_index: Tensor, group_size_by_query: Tensor) -> Tensor:
    """Orthogonal block projection in O(S Q H V), without any Q-by-Q matrix."""
    # Allocate Q rows to avoid a device scalar query for the number of groups.
    sums = torch.zeros_like(state).index_add(1, group_index, state)
    return sums[:, group_index] / group_size_by_query[None, :, None, None]


def _hold_action(state, group_index, sizes, exchange, damping, duration, raw_diagonal=False):
    if raw_diagonal:
        diagonal = damping[:, None, :] + exchange[:, None, :] * (1 - 1 / sizes)[None, :, None]
        return torch.exp(-duration * diagonal)[..., None] * state
    mean = block_average_v1(state, group_index, sizes)
    slow = torch.exp(-duration * damping)
    fast = torch.exp(-duration * (damping + exchange))
    # Positive coefficients avoid cancellation in a-b for short intervals.
    mean_gain = slow * (-torch.expm1(-duration * exchange))
    return fast[:, None, :, None] * state + mean_gain[:, None, :, None] * mean


def census_cut_action_v1(bank: CensusExchangeBankV1, sources: Tensor, exchange: Tensor,
        damping: Tensor, eta: Tensor, *, control: str = "proposed",
        readout_chunk_size: int = 128) -> Tensor:
    """Bundled finite-anchor E/J action, inclusive forward and exclusive reverse.

    Strictly ordered E/J products matter. The reverse is the transpose of the
    complete forward path, not an untransposed replay of forward boundaries.
    Sources [A,S,Q,H,V] are weighted by native physical anchor cell widths.
    """
    if control not in {"proposed", "eta0", "permuted_groups", "raw_diagonal"}:
        raise ValueError("Unknown finite-anchor census control")
    if control == "permuted_groups" and bank.membership_control != "fixed_global_group_permutation":
        raise ValueError("Permuted control requires a checked fixed global membership permutation")
    if not isinstance(readout_chunk_size, int) or readout_chunk_size < 1 or sources.ndim != 5:
        raise ValueError("Require [A,S,Q,H,V] sources and positive readout chunk size")
    a, s, q, h, v = sources.shape
    layout = bank.layout
    if (min(a, s, q, h, v) < 1 or a != len(layout.graph_at_anchor)
            or q != len(bank.query_ids) or s != len(bank.sigmas)
            or any(x.shape != (s, h) for x in (exchange, damping, eta))
            or bank.group_index.shape != (layout.num_graphs, q)
            or bank.group_size_by_query.shape != (layout.num_graphs, q)
            or bank.boundary_rho.shape != (len(layout.source_at_event), q)):
        raise ValueError("Finite-anchor census dimensions differ")
    fixed = (bank.group_size_by_query, bank.boundary_rho, bank.donor_weights)
    if any(x.requires_grad for x in fixed):
        raise ValueError("Observed groups/retention/donor geometry must remain fixed")
    if (any(x.dtype != sources.dtype or x.device != sources.device for x in
            (exchange, damping, eta, *fixed)) or layout.segment_duration.device != sources.device
            or bank.group_index.device != sources.device):
        raise ValueError("Transfer source/rates/fixed bank with matching device and dtype")
    flags = torch.stack((torch.isfinite(sources).all(), torch.isfinite(exchange).all(),
        torch.isfinite(damping).all(), torch.isfinite(eta).all(), (exchange > 0).all(),
        (damping > 0).all(), (eta >= 0).all(), torch.isfinite(bank.boundary_rho).all(),
        (bank.boundary_rho > 0).all(), (bank.boundary_rho <= 1).all(),
        torch.isfinite(bank.group_size_by_query).all(), (bank.group_size_by_query >= 1).all()))
    if not bool(flags.all()):
        raise ValueError("Sources, positive rates, nonnegative eta and fixed retention must be valid")
    exponent = torch.zeros_like(eta) if control == "eta0" else eta
    retention = torch.exp(bank.boundary_rho.log()[:, None, :, None] * exponent[None, :, None, :])
    if not bool(torch.isfinite(retention).all() & (retention > 0).all()):
        raise ValueError("Retention overflow/underflow is rejected without a floor")
    impulse = sources * layout.weights.to(sources.dtype)[:, None, None, None, None]
    zero = sources.new_zeros((s, q, h, v))
    forward, backward = [], [zero] * layout.num_segments
    state = zero
    raw_diagonal = control == "raw_diagonal"
    for segment, graph in enumerate(layout.graph_at_segment):
        if segment and graph != layout.graph_at_segment[segment - 1]:
            state = retention[segment, ..., None] * state
        anchor = layout.source_at_event[segment]
        if anchor >= 0:
            state = state + impulse[anchor]
        forward.append(state)
        state = _hold_action(state, bank.group_index[graph], bank.group_size_by_query[graph],
            exchange, damping, layout.segment_duration[segment].to(sources.dtype), raw_diagonal)
    state = zero
    for segment in range(layout.num_segments - 1, -1, -1):
        graph = layout.graph_at_segment[segment]
        event = segment + 1
        anchor = layout.source_at_event[event]
        if anchor >= 0:
            state = state + impulse[anchor]
        if event < layout.num_segments and graph != layout.graph_at_segment[event]:
            state = retention[event, ..., None] * state
        backward[segment] = state
        state = _hold_action(state, bank.group_index[graph], bank.group_size_by_query[graph],
            exchange, damping, layout.segment_duration[segment].to(sources.dtype), raw_diagonal)
    chunks = []
    for segment, (graph, outputs) in enumerate(zip(layout.graph_at_segment, layout.evaluation_at_segment)):
        if not outputs:
            continue
        sizes = bank.group_size_by_query[graph]
        if not raw_diagonal:
            mf = block_average_v1(forward[segment], bank.group_index[graph], sizes)
            mb = block_average_v1(backward[segment], bank.group_index[graph], sizes)
        for start in range(0, len(outputs), readout_chunk_size):
            indexes = list(outputs[start:start + readout_chunk_size])
            left = layout.evaluation_left[indexes].to(sources.dtype)
            right = layout.evaluation_right[indexes].to(sources.dtype)
            if raw_diagonal:
                diagonal = damping[:, None, :] + exchange[:, None, :] * (1 - 1 / sizes)[None, :, None]
                value = (torch.exp(-left[:, None, None, None] * diagonal)[..., None] * forward[segment][None]
                         + torch.exp(-right[:, None, None, None] * diagonal)[..., None] * backward[segment][None])
            else:
                def readout(state, mean, offset):
                    dt = offset[:, None, None]
                    slow = torch.exp(-dt * damping)
                    fast = torch.exp(-dt * (damping + exchange))
                    mean_gain = slow * (-torch.expm1(-dt * exchange))
                    return (fast[:, :, None, :, None] * state[None]
                            + mean_gain[:, :, None, :, None] * mean[None])
                value = readout(forward[segment], mf, left) + readout(backward[segment], mb, right)
            chunks.append(value)
    return torch.cat(chunks, dim=0)


def _sources_and_mass(query_signal, reference_signal, weights, exchange):
    source = query_signal[:, None] + (weights @ reference_signal[:, None]) * exchange[None, :, None, :]
    mass = torch.ones_like(source) + weights.sum(-1)[..., None] * exchange[None, :, None, :]
    return source, mass


def _normalize(raw: Tensor, mixture: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mixed = (raw * mixture[None, :, None, :, None]).sum(1)
    numerator, mass = mixed.unbind(-1)
    if not bool(torch.isfinite(mixed).all() & (mass > 0).all()):
        raise ValueError("Numerator/matching mass is nonfinite or mass is nonpositive")
    return numerator, mass, numerator / mass


def census_cut_messages_v1(bank: CensusExchangeBankV1, query_signal: Tensor,
        reference_signal: Tensor, exchange: Tensor, damping: Tensor, eta: Tensor,
        mixture: Tensor, *, control: str = "proposed", readout_chunk_size: int = 128) -> AnchorLayerMessages:
    a, s, q, r = len(bank.layout.graph_at_anchor), len(bank.sigmas), len(bank.query_ids), len(bank.reference_ids)
    h = exchange.shape[-1]
    if (query_signal.shape != (a, q, h) or reference_signal.shape != (a, r, h)
            or bank.donor_weights.shape != (bank.layout.num_graphs, s, q, r)
            or mixture.shape != (s, h)):
        raise ValueError("Census query/reference/mixture dimensions differ")
    if not bool(torch.isfinite(mixture).all() & (mixture >= 0).all()):
        raise ValueError("Scale mixture must be finite and nonnegative")
    weights = bank.donor_weights[bank.layout.anchor_graph]
    if not bool(torch.isfinite(weights).all() & (weights >= 0).all()):
        raise ValueError("Observed donor weights must be finite and nonnegative")
    source, mass_source = _sources_and_mass(query_signal, reference_signal, weights, exchange)
    raw = census_cut_action_v1(bank, torch.stack((source, mass_source), dim=-1), exchange, damping, eta,
                              control=control, readout_chunk_size=readout_chunk_size)
    numerator, mass, normalized = _normalize(raw, mixture)
    return AnchorLayerMessages(normalized[bank.layout.anchor_output], normalized[bank.layout.target_output],
                               numerator, mass, source, mass_source)


def pointwise_local_messages_v1(query_signal: Tensor, reference_signal: Tensor, donor_weights: Tensor,
        exchange: Tensor, mixture: Tensor) -> tuple[Tensor, Tensor]:
    """Matched local donor/source/mixture control with no genomic propagation.

    Coordinate-local contexts may be supplied at anchors or targets. This is a
    declared pointwise contrast, not the finite-anchor kernel or a new model.
    """
    if (donor_weights.requires_grad or not bool(torch.isfinite(donor_weights).all()
            & (donor_weights >= 0).all())):
        raise ValueError("Local donor geometry must be fixed, finite and nonnegative")
    if not bool(torch.isfinite(exchange).all() & (exchange > 0).all()):
        raise ValueError("Local exchange source rate must be finite and positive")
    if not bool(torch.isfinite(mixture).all() & (mixture >= 0).all()):
        raise ValueError("Scale mixture must be finite and nonnegative")
    source, mass = _sources_and_mass(query_signal, reference_signal, donor_weights, exchange)
    _, mass, normalized = _normalize(torch.stack((source, mass), -1), mixture)
    return normalized, mass


def _initial(values: float | Sequence[float], branches: int, name: str) -> Tensor:
    sequence = [values] * branches if isinstance(values, (float, int)) else list(values)
    if len(sequence) != branches or any(not math.isfinite(x) or x <= 0 for x in sequence):
        raise ValueError(name + " initial values must be finite and positive")
    return torch.tensor(sequence, dtype=torch.float32)[:, None]


class CensusCutExchangeLayerV1(nn.Module):
    """First executable constrained layer, with unchanged inherited gate pattern.

    Rates are positive init*exp(raw) per branch/channel. Proposed eta is positive
    init*exp(raw); the matched eta0 control sets eta exactly zero and makes its
    retention parameters inactive. Observed group/date/overlap geometry is fixed.
    """
    def __init__(self, hidden_dim: int = 32, latent_dim: int = 8, num_branches: int = 3, *,
                 initial_exchange: float | Sequence[float] = (5., 20., 80.),
                 initial_damping: float | Sequence[float] = 1., initial_eta: float | Sequence[float] = 1.):
        super().__init__()
        if min(hidden_dim, latent_dim, num_branches) < 1:
            raise ValueError("Layer dimensions must be positive")
        self.query_projection = nn.Linear(hidden_dim, latent_dim)
        self.reference_projection = nn.Linear(hidden_dim, latent_dim)
        self.message_projection = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("initial_exchange", _initial(initial_exchange, num_branches, "exchange"))
        self.register_buffer("initial_damping", _initial(initial_damping, num_branches, "damping"))
        self.register_buffer("initial_eta", _initial(initial_eta, num_branches, "eta"))
        self.raw_exchange = nn.Parameter(torch.zeros(num_branches, latent_dim))
        self.raw_damping = nn.Parameter(torch.zeros(num_branches, latent_dim))
        self.raw_eta = nn.Parameter(torch.zeros(num_branches, latent_dim))
        self.mixture_logits = nn.Parameter(torch.zeros(num_branches, latent_dim))

    @property
    def exchange(self):
        return self.initial_exchange * self.raw_exchange.exp()

    @property
    def damping(self):
        return self.initial_damping * self.raw_damping.exp()

    @property
    def eta(self):
        return self.initial_eta * self.raw_eta.exp()

    @property
    def mixture(self):
        return self.mixture_logits.softmax(0)

    def rate_and_mixture_parameters(self):
        yield from (self.raw_exchange, self.raw_damping, self.raw_eta, self.mixture_logits)

    def update(self, local: Tensor, message: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((local, message), -1)))
        return self.norm(local + gate * self.message_projection(message))

    def messages(self, bank, anchor_context, reference_context, *, control="proposed", readout_chunk_size=128):
        return census_cut_messages_v1(bank, self.query_projection(anchor_context).tanh(),
            self.reference_projection(reference_context).tanh(), self.exchange, self.damping,
            self.eta, self.mixture, control=control, readout_chunk_size=readout_chunk_size)

    def forward(self, bank, anchor_context, reference_context, target_context, *,
                control="proposed", target_reference_context=None, readout_chunk_size=128):
        if control == "local":
            if target_reference_context is None:
                raise ValueError("Pointwise local control requires coordinate-local target references")
            am, _ = pointwise_local_messages_v1(self.query_projection(anchor_context).tanh(),
                self.reference_projection(reference_context).tanh(), bank.donor_weights[bank.layout.anchor_graph],
                self.exchange, self.mixture)
            tm, _ = pointwise_local_messages_v1(self.query_projection(target_context).tanh(),
                self.reference_projection(target_reference_context).tanh(), bank.donor_weights[bank.target_graph],
                self.exchange, self.mixture)
        else:
            messages = self.messages(bank, anchor_context, reference_context,
                                      control=control, readout_chunk_size=readout_chunk_size)
            am, tm = messages.anchors, messages.targets
        return self.update(anchor_context, am), self.update(target_context, tm)

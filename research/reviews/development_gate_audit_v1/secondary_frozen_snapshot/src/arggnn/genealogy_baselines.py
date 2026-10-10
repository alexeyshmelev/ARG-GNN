"""Observed-tree ancestry controls without a finite-hop reference bottleneck.

These are mathematical feature controls, not novel GNNs or exact reproductions
of AncestralPaths/ARGMix. They read only topology, declared reference columns
and observed time fields. Truth annotations are never an argument.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from .models import GraphSequenceInput, validate_model_input


@dataclass
class ReferenceSummary:
    features: Tensor                  # [G,Q,K,6+S]
    nearest_votes: Tensor             # [G,Q,K]
    minimum_tmrca: Tensor             # [G,Q,K], observed reconstructed generations
    reference_counts: Tensor          # [G,K]


@dataclass
class _Tree:
    nodes: list[int]
    preorder: list[int]
    parent: dict[int, int]
    depth: dict[int, int]
    adjacency: dict[int, list[int]]
    query_nodes: list[int]
    reference_nodes: list[int]


def _observed_trees(seq: GraphSequenceInput, reference_columns: Sequence[int],
                    time_column: int, branch_column: int | None, validate: bool):
    if validate:
        validate_model_input(seq)
    columns = tuple(int(c) for c in reference_columns)
    if len(columns) < 2 or len(set(columns)) != len(columns):
        raise ValueError('Declare at least two unique reference feature columns')
    f = seq.x.shape[1]
    if any(c < 0 or c >= f for c in columns) or not 0 <= time_column < f:
        raise ValueError('Observed feature column outside input feature range')
    if time_column in columns or (branch_column is not None and (not 0 <= branch_column < f or branch_column in columns)):
        raise ValueError('Time/branch fields must be distinct from reference class columns')
    x = seq.x.detach().cpu().double().numpy()
    graph_ids = seq.node_graph.detach().cpu().numpy()
    leaf_ids = seq.node_ids.detach().cpu().numpy()
    query_ids = seq.query_node_ids.detach().cpu().tolist()
    reference = x[:, columns]
    if not np.isfinite(reference).all() or not np.isin(reference, (0.0, 1.0)).all() or (reference.sum(axis=1) > 1).any():
        raise ValueError('Reference columns must be observed one-hot panel labels or all zero')
    if ((reference.sum(axis=1) > 0) & (leaf_ids < 0)).any():
        raise ValueError('Internal-node reference labels are forbidden')
    encoded_times = x[:, time_column]
    if not np.isfinite(encoded_times).all() or (encoded_times < 0).any():
        raise ValueError('Observed encoded node times must be finite and nonnegative')
    edge_list = seq.edge_index.detach().cpu().T.tolist()
    adjacency = {u: set() for u in range(len(x))}
    for a, b in edge_list:
        if a == b:
            raise ValueError('Tree topology cannot contain self loops')
        adjacency[a].add(b)
        adjacency[b].add(a)
    trees = []
    for graph in range(seq.intervals.shape[0]):
        nodes = np.flatnonzero(graph_ids == graph).tolist()
        if sum(len(adjacency[u]) for u in nodes) != 2 * (len(nodes) - 1):
            raise ValueError('Each graph must be one connected acyclic coalescent tree')
        maximum = max(encoded_times[u] for u in nodes)
        candidates = [u for u in nodes if encoded_times[u] == maximum]
        if branch_column is not None:
            candidates = [u for u in candidates if x[u, branch_column] == 0]
        if len(candidates) != 1:
            raise ValueError('Root time is ambiguous. Supply an observed root-zero branch column')
        root = candidates[0]
        parent, depth, preorder = {root: -1}, {root: 0}, []
        queue = [root]
        cursor = 0
        while cursor < len(queue):
            u = queue[cursor]
            cursor += 1
            preorder.append(u)
            for v in sorted(adjacency[u]):
                if v == parent[u]:
                    continue
                if v in parent:
                    raise ValueError('A local genealogy contains a cycle')
                if encoded_times[v] > encoded_times[u] + 1e-7:
                    raise ValueError('Node times disagree with rooted coalescent topology')
                parent[v], depth[v] = u, depth[u] + 1
                queue.append(v)
        if len(preorder) != len(nodes):
            raise ValueError('Disconnected genealogy cannot provide between-root TMRCAs')
        queries = []
        for leaf in query_ids:
            matching = [u for u in nodes if leaf_ids[u] == leaf]
            if len(matching) != 1:
                raise ValueError('Every query leaf must occur once in every tree')
            u = matching[0]
            if reference[u].sum() != 0:
                raise ValueError('Query ancestry labels cannot appear in reference feature columns')
            if len(adjacency[u]) > 1:
                raise ValueError('Stable query IDs must identify terminal sample leaves')
            queries.append(u)
        refs = [u for u in nodes if reference[u].sum() > 0]
        if not refs or (reference[refs].sum(axis=0) == 0).any():
            raise ValueError('Each source population requires at least one labeled reference')
        if any(len(adjacency[u]) > 1 for u in refs):
            raise ValueError('Reference labels must identify terminal sample leaves')
        trees.append(_Tree(nodes, preorder, parent, depth,
                           {u: sorted(adjacency[u]) for u in nodes}, queries, refs))
    return x, reference, encoded_times, trees


class ReferenceTMRCAFeatures(nn.Module):
    """Exact source-conditioned summaries from upward/downward tree passes.

    Every labeled reference contributes through its unique sibling shell.
    The pass is O(N*K*(S+1)) after topology preparation, without a hop cutoff.
    This fixed extractor does not differentiate through observed input features.
    """
    def __init__(self, reference_columns: Sequence[int], time_column: int,
                 branch_column: int | None = None, time_scale: float = 10000.0,
                 kernel_scales: Sequence[float] = (0.05, 0.1, 0.2), validate: bool = True):
        super().__init__()
        self.reference_columns = tuple(int(c) for c in reference_columns)
        self.time_column = int(time_column)
        self.branch_column = branch_column
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('time_scale must be finite and positive')
        self.time_scale = float(time_scale)
        self.kernel_scales = tuple(float(s) for s in kernel_scales)
        if any(not math.isfinite(s) or s <= 0 for s in self.kernel_scales):
            raise ValueError('kernel scales must be finite and positive')
        self.validate_inputs = validate

    @property
    def num_classes(self) -> int:
        return len(self.reference_columns)

    @property
    def feature_dim(self) -> int:
        return self.num_classes * (6 + len(self.kernel_scales))

    def summarize(self, seq: GraphSequenceInput) -> ReferenceSummary:
        _, reference, times, trees = _observed_trees(seq, self.reference_columns,
                                                   self.time_column, self.branch_column,
                                                   self.validate_inputs)
        n, k = reference.shape
        s = len(self.kernel_scales)
        subtree = reference.copy()
        for tree in trees:
            for u in reversed(tree.preorder):
                p = tree.parent[u]
                if p != -1:
                    subtree[p] += subtree[u]
        outside = np.zeros((n, k), dtype=np.float64)
        first = np.zeros_like(outside)
        second = np.zeros_like(outside)
        kernels = np.zeros((n, k, s), dtype=np.float64)
        minimum = np.full((n, k), np.inf, dtype=np.float64)
        minimum_count = np.zeros_like(outside)
        scales = np.asarray(self.kernel_scales)
        features, votes, minimum_generations, reference_counts = [], [], [], []
        for tree in trees:
            for u in tree.preorder:
                p = tree.parent[u]
                if p == -1:
                    continue
                shell = subtree[p] - subtree[u]
                outside[u] = outside[p] + shell
                first[u] = first[p] + shell * times[p]
                second[u] = second[p] + shell * times[p] ** 2
                if s:
                    kernels[u] = kernels[p] + shell[:, None] * np.exp(-times[p] / scales)[None, :]
                candidate = np.where(shell > 0, times[p], np.inf)
                smaller = candidate < minimum[p]
                equal = candidate == minimum[p]
                minimum[u] = np.minimum(candidate, minimum[p])
                minimum_count[u] = np.where(smaller, shell,
                                             np.where(equal, shell + minimum_count[p], minimum_count[p]))
            query = np.asarray(tree.query_nodes)
            total = outside[query]
            if (total <= 0).any() or not np.isfinite(minimum[query]).all():
                raise ValueError('Missing reference source in a query tree')
            mean = first[query] / total
            std = np.sqrt(np.maximum(second[query] / total - mean ** 2, 0))
            global_minimum = minimum[query].min(axis=1, keepdims=True)
            nearest_counts = np.where(minimum[query] == global_minimum, minimum_count[query], 0)
            nearest_votes = nearest_counts / nearest_counts.sum(axis=1, keepdims=True)
            count_fraction = total / total.sum(axis=1, keepdims=True)
            nearest_fraction = minimum_count[query] / total
            components = [minimum[query], mean, std, count_fraction, nearest_fraction, nearest_votes]
            if s:
                density = kernels[query] / total[:, :, None]
                density /= np.maximum(density.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
                components.extend(density[:, :, i] for i in range(s))
            features.append(np.stack(components, axis=-1))
            votes.append(nearest_votes)
            reconstructed = np.expm1(minimum[query] * math.log1p(self.time_scale))
            if not np.isfinite(reconstructed).all():
                raise ValueError('Encoded observed times overflow reconstruction')
            minimum_generations.append(reconstructed)
            reference_counts.append(total[0])
        def tensor(array):
            return torch.as_tensor(np.asarray(array), dtype=seq.x.dtype, device=seq.x.device)
        return ReferenceSummary(tensor(features), tensor(votes), tensor(minimum_generations), tensor(reference_counts))

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        summary = self.summarize(seq)
        return summary.features[seq.target_graph].flatten(start_dim=-2)


class NearestTMRCAClassifier(nn.Module):
    """Deterministic observed-tree nearest-reference classifier with shared ties."""
    def __init__(self, input_dim: int, num_classes: int, *, reference_columns: Sequence[int],
                 time_column: int, branch_column: int | None = None,
                 time_scale: float = 10000.0, validate: bool = True):
        super().__init__()
        if len(reference_columns) != num_classes:
            raise ValueError('num_classes must match declared reference columns')
        self.input_dim = input_dim
        self.extractor = ReferenceTMRCAFeatures(reference_columns, time_column, branch_column,
                                               time_scale=time_scale, validate=validate)

    def probabilities(self, seq: GraphSequenceInput) -> Tensor:
        if seq.x.shape[1] != self.input_dim:
            raise ValueError('Input feature dimension does not match declared metadata')
        return self.extractor.summarize(seq).nearest_votes[seq.target_graph]

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        return self.probabilities(seq).clamp_min(1e-8).log()


class TMRCAFeatureMLP(nn.Module):
    """Trainable correction on whole-tree TMRCA summaries, with explicit prior.

    With nearest_init=True, initial outputs equal deterministic nearest-reference
    logits. This choice must be reported when comparing learning curves.
    """
    def __init__(self, input_dim: int, num_classes: int, *, reference_columns: Sequence[int],
                 time_column: int, branch_column: int | None = None,
                 time_scale: float = 10000.0, hidden_dim: int = 64, num_layers: int = 2,
                 nearest_init: bool = True, kernel_scales: Sequence[float] = (0.05, 0.1, 0.2),
                 validate: bool = True):
        super().__init__()
        if len(reference_columns) != num_classes or min(input_dim, hidden_dim, num_layers) < 1:
            raise ValueError('Invalid model dimensions or source-column declaration')
        self.input_dim, self.num_classes = input_dim, num_classes
        self.nearest_init = bool(nearest_init)
        self.extractor = ReferenceTMRCAFeatures(reference_columns, time_column, branch_column,
                                               time_scale=time_scale, kernel_scales=kernel_scales,
                                               validate=validate)
        layers = []
        width = self.extractor.feature_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_dim), nn.GELU()))
            width = hidden_dim
        layers.append(nn.Linear(width, num_classes))
        self.head = nn.Sequential(*layers)
        if nearest_init:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def features_and_prior(self, seq: GraphSequenceInput) -> tuple[Tensor, Tensor]:
        if seq.x.shape[1] != self.input_dim:
            raise ValueError('Input feature dimension does not match declared metadata')
        summary = self.extractor.summarize(seq)
        return (summary.features[seq.target_graph].flatten(start_dim=-2),
                summary.nearest_votes[seq.target_graph].clamp_min(1e-8).log())

    def forward_from_features(self, features: Tensor, prior_logits: Tensor | None = None) -> Tensor:
        result = self.head(features)
        if self.nearest_init:
            if prior_logits is None:
                raise ValueError('The nearest-initialized control requires declared prior_logits')
            result = result + prior_logits
        return result

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        features, prior = self.features_and_prior(seq)
        return self.forward_from_features(features, prior)


def pairwise_reference_tmrca(seq: GraphSequenceInput, *, reference_columns: Sequence[int],
                              time_column: int, branch_column: int | None = None,
                              time_scale: float = 10000.0, validate: bool = True) -> list[dict]:
    """Retain the full query-reference TMRCA relation for audits or richer heads.

    Independent ancestor-walk implementation, O(G*Q*R*h) for height h. The
    scalable summary extractor above avoids materializing these matrices.
    """
    _, reference, times, trees = _observed_trees(seq, reference_columns, time_column, branch_column, validate)
    ids = seq.node_ids.detach().cpu().numpy()
    records = []
    for graph, tree in enumerate(trees):
        refs = sorted(tree.reference_nodes, key=lambda u: int(ids[u]))
        encoded = np.empty((len(tree.query_nodes), len(refs)), dtype=np.float64)
        hops = np.empty_like(encoded, dtype=np.int64)
        for qi, query in enumerate(tree.query_nodes):
            ancestors = set()
            u = query
            while u != -1:
                ancestors.add(u)
                u = tree.parent[u]
            for ri, ref in enumerate(refs):
                lca = ref
                while lca not in ancestors:
                    lca = tree.parent[lca]
                encoded[qi, ri] = times[lca]
                hops[qi, ri] = tree.depth[query] + tree.depth[ref] - 2 * tree.depth[lca]
        records.append({'graph': graph,
                        'reference_ids': torch.tensor([int(ids[u]) for u in refs]),
                        'reference_classes': torch.tensor(reference[refs].argmax(axis=1)),
                        'tmrca': torch.from_numpy(np.expm1(encoded * math.log1p(time_scale))),
                        'encoded_tmrca': torch.from_numpy(encoded),
                        'hop_distances': torch.from_numpy(hops)})
    return records


def reference_hop_diagnostics(seq: GraphSequenceInput, *, reference_columns: Sequence[int],
                               time_column: int, branch_column: int | None = None,
                               num_layers: int = 2, validate: bool = True) -> dict:
    """Fixed-grid observed reachability only, without ancestry truth labels."""
    records = pairwise_reference_tmrca(seq, reference_columns=reference_columns,
                                       time_column=time_column, branch_column=branch_column,
                                       validate=validate)
    weights = torch.bincount(seq.target_graph.detach().cpu(), minlength=len(records))
    histogram, nearest_histogram = {}, {}
    within, nearest_within, total = 0, 0, 0
    for record, multiplicity in zip(records, weights.tolist()):
        if multiplicity == 0:
            continue
        hops, times = record['hop_distances'], record['encoded_tmrca']
        minimum_hops = hops.min(dim=1).values
        nearest = times == times.min(dim=1, keepdim=True).values
        nearest_hops = torch.where(nearest, hops, torch.full_like(hops, 10**9)).min(dim=1).values
        for h, nh in zip(minimum_hops.tolist(), nearest_hops.tolist()):
            histogram[str(h)] = histogram.get(str(h), 0) + multiplicity
            nearest_histogram[str(nh)] = nearest_histogram.get(str(nh), 0) + multiplicity
            within += multiplicity * (h <= num_layers)
            nearest_within += multiplicity * (nh <= num_layers)
            total += multiplicity
    return {'query_coordinate_pairs': total, 'message_passing_layers': num_layers,
            'fraction_with_any_reference_in_receptive_field': within / total,
            'fraction_with_a_nearest_tmrca_reference_in_receptive_field': nearest_within / total,
            'minimum_reference_hops_histogram': histogram,
            'nearest_tmrca_reference_hops_histogram': nearest_histogram}

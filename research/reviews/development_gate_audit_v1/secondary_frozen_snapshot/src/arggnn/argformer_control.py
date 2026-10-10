"""Paper-equation ARGMix controls on the common observed GraphSequence.

Independent adaptation of Shanks et al. (2026), DOI 10.64898/2026.04.05.714076v1,
Methods 4.1 Eqs. (1)-(6). This is not an exact ARGMix or ARGformer reproduction.
All observed nodes/features are retained by the primary full_tree control.
Integer IDs only locate query leaves. See docs/argformer_control.md.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import GraphSequenceInput, query_node_index, validate_model_input


@dataclass
class _ObservedTree:
    nodes: Tensor
    parent: Tensor
    mrca: Tensor
    hops: Tensor
    times: Tensor


def _observed_tree_geometry(seq: GraphSequenceInput, time_column: int,
                            branch_column: int | None, time_scale: float) -> list[_ObservedTree]:
    """CPU structural preprocessing, using observed edges and encoded times only.

    Pairwise LCA is filled by disjoint child-subtree products, using O(n^2)
    storage/work per tree. No reference labels, clade landmarks, or ID values
    determine the topology, root, or pairwise relations.
    """
    x = seq.x.detach().cpu().double().numpy()
    groups = seq.node_graph.detach().cpu().numpy()
    encoded = x[:, time_column]
    if not np.isfinite(encoded).all() or (encoded < 0).any():
        raise ValueError("Observed encoded times must be finite and nonnegative")
    with np.errstate(over="ignore"):
        times = np.expm1(encoded * math.log1p(time_scale))
    if not np.isfinite(times).all():
        raise ValueError("Observed times overflow the declared time encoding")
    adjacency = [set() for _ in range(len(x))]
    for a, b in seq.edge_index.detach().cpu().T.tolist():
        if a == b:
            raise ValueError("Observed marginal trees cannot contain self loops")
        if groups[a] != groups[b]:
            raise ValueError("Observed tree edges cannot cross graphs")
        adjacency[a].add(b)
        adjacency[b].add(a)
    result = []
    for graph in range(len(seq.intervals)):
        nodes = np.flatnonzero(groups == graph)
        if not len(nodes) or sum(len(adjacency[u]) for u in nodes) != 2 * (len(nodes) - 1):
            raise ValueError("Each observed marginal graph must be one connected acyclic tree")
        roots = nodes[encoded[nodes] == encoded[nodes].max()]
        if branch_column is not None:
            roots = roots[x[roots, branch_column] == 0]
        if len(roots) != 1:
            raise ValueError("Ambiguous observed root; declare the root-zero branch column")
        index = {int(u): i for i, u in enumerate(nodes)}
        root = index[int(roots[0])]
        parent = np.full(len(nodes), -2, dtype=np.int64)
        depth = np.zeros(len(nodes), dtype=np.int64)
        parent[root] = -1
        preorder = [root]
        children = [[] for _ in nodes]
        for u in preorder:
            for global_v in adjacency[nodes[u]]:
                v = index[global_v]
                if v == parent[u]:
                    continue
                if parent[v] != -2:
                    raise ValueError("Observed marginal tree contains a cycle")
                if encoded[nodes[v]] > encoded[nodes[u]] + 1e-7:
                    raise ValueError("Observed times disagree with rooted topology")
                parent[v] = u
                depth[v] = depth[u] + 1
                children[u].append(v)
                preorder.append(v)
        if len(preorder) != len(nodes):
            raise ValueError("Observed marginal tree is disconnected")
        descendants = [None for _ in nodes]
        mrca = np.full((len(nodes), len(nodes)), -1, dtype=np.int64)
        for u in reversed(preorder):
            seen = [u]
            mrca[u, u] = u
            for v in children[u]:
                other = descendants[v]
                mrca[np.ix_(seen, other)] = u
                mrca[np.ix_(other, seen)] = u
                seen.extend(other)
            descendants[u] = seen
        hops = depth[:, None] + depth[None, :] - 2 * depth[mrca]
        result.append(_ObservedTree(
            torch.from_numpy(nodes.copy()), torch.from_numpy(parent),
            torch.from_numpy(mrca), torch.from_numpy(hops),
            torch.from_numpy(times[nodes].copy()),
        ))
    return result


def relative_tmrca_bins(tree: _ObservedTree, max_rel_distance: int) -> Tensor:
    """Paper Eq. (2): floor(min(TMRCA(i,j)-time_i, cap)) in generations.

    floor is the disclosed lookup-table discretization for continuous inferred
    times. Structural bins are fixed observed inputs, not differentiable times.
    """
    elapsed = (tree.times[tree.mrca] - tree.times[:, None]).clamp_min(0)
    return elapsed.clamp_max(max_rel_distance).floor().long()


class RelativeTreeAttention(nn.Module):
    """Paper Eqs. (1), (3)-(6), with the published key-column term k_j.

    The relative Q/K/V tables are shared across layers by the calling model.
    Relational values are accumulated into bins to avoid O(n^2*d) storage.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if min(hidden_dim, num_heads) < 1:
            raise ValueError("hidden_dim and num_heads must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, relations: Tensor,
                relative_query: Tensor, relative_key: Tensor,
                relative_value: Tensor) -> Tensor:
        n, d = h.shape
        heads, width = self.num_heads, self.head_dim
        q, k, v = [projection(h).view(n, heads, width).transpose(0, 1)
                   for projection in (self.q, self.k, self.v)]
        tables = [table.view(-1, heads, width).permute(1, 0, 2)
                  for table in (relative_query, relative_key, relative_value)]
        qr, kr, vr = tables
        lookup = relations[None].expand(heads, -1, -1)
        query_terms = torch.einsum("hnd,hbd->hnb", q, qr).gather(2, lookup)
        # Evaluate k_j against relation(i,j), not k_i against that relation.
        key_terms = torch.einsum("hnd,hbd->hnb", k, kr).gather(
            2, lookup.transpose(1, 2)).transpose(1, 2)
        logits = (q @ k.transpose(1, 2) + query_terms + key_terms) / math.sqrt(width)
        weights = self.dropout(logits.softmax(-1))
        relative_mass = weights.new_zeros(heads, n, vr.shape[1])
        relative_mass = relative_mass.scatter_add(2, lookup, weights)
        output = weights @ v + torch.einsum("hnb,hbd->hnd", relative_mass, vr)
        return self.output(output.transpose(0, 1).reshape(n, d))


class _RelativeTreeLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_multiplier: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = RelativeTreeAttention(hidden_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_multiplier * hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, relations: Tensor, tables: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        h = h + self.dropout(self.attention(self.attention_norm(h), relations, *tables))
        return h + self.dropout(self.ffn(self.ffn_norm(h)))


class ARGMixTreeControl(nn.Module):
    """All-node, single-marginal-tree ARGMix equation adaptation.

    Primary scope full_tree consumes all supplied observed features and every
    node, including all query/reference/internal nodes. Optional
    nearest_reference uses one query, nearest observed references, and its
    immediate ancestors with a learned CLS readout. Neither scope performs
    smoothing or ordered/set aggregation across genomic trees.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 ffn_multiplier: int = 4, *, time_column: int = 6,
                 branch_column: int | None = 7, time_scale: float = 10000.0,
                 max_rel_distance: int = 1000, token_scope: str = "full_tree",
                 reference_columns: tuple[int, ...] = (0, 1, 2, 3),
                 num_references: int = 25, num_ancestors: int = 3,
                 validate: bool = True):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, num_heads, ffn_multiplier, max_rel_distance) < 1:
            raise ValueError("Dimensions, layers, heads and distance cap must be positive")
        if num_classes < 2 or not 0 <= dropout < 1:
            raise ValueError("Require num_classes >= 2 and dropout in [0,1)")
        if not 0 <= time_column < input_dim or (
                branch_column is not None and not 0 <= branch_column < input_dim):
            raise ValueError("Observed time/branch columns must be inside input_dim")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if token_scope not in {"full_tree", "nearest_reference"}:
            raise ValueError("token_scope must be full_tree or nearest_reference")
        if num_references < 1 or num_ancestors < 0:
            raise ValueError("Require positive reference count and nonnegative ancestor count")
        self.input_dim, self.num_classes, self.hidden_dim = input_dim, num_classes, hidden_dim
        self.num_layers, self.num_heads = num_layers, num_heads
        self.dropout_probability, self.ffn_multiplier = float(dropout), int(ffn_multiplier)
        self.time_column, self.branch_column, self.time_scale = time_column, branch_column, float(time_scale)
        self.max_rel_distance, self.token_scope = int(max_rel_distance), token_scope
        self.reference_columns = tuple(reference_columns)
        self.num_references, self.num_ancestors = num_references, num_ancestors
        if not self.reference_columns or any(c < 0 or c >= input_dim for c in self.reference_columns):
            raise ValueError("Observed reference columns must be inside input_dim")
        if len(set(self.reference_columns)) != len(self.reference_columns):
            raise ValueError("Observed reference columns must be unique")
        self.validate_inputs = validate
        self.relation_kind = "elapsed_tmrca"
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # Special CLS relation is separate from capped biological elapsed time.
        bins = self.max_rel_distance + 1 + int(token_scope == "nearest_reference")
        self.relative_query = nn.Embedding(bins, hidden_dim)
        self.relative_key = nn.Embedding(bins, hidden_dim)
        self.relative_value = nn.Embedding(bins, hidden_dim)
        self.layers = nn.ModuleList(_RelativeTreeLayer(hidden_dim, num_heads, dropout, ffn_multiplier)
                                    for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        if token_scope == "nearest_reference":
            self.cls_token = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self._geometry_cache: OrderedDict = OrderedDict()

    def clear_geometry_cache(self) -> None:
        self._geometry_cache.clear()

    def _geometry(self, seq: GraphSequenceInput) -> list[_ObservedTree]:
        tensors = (seq.x, seq.edge_index, seq.node_graph)
        key = tuple((id(t), t._version) for t in tensors) + (len(seq.intervals),)
        if key not in self._geometry_cache:
            trees = _observed_tree_geometry(seq, self.time_column, self.branch_column, self.time_scale)
            # Retain source tensors to prevent allocator/object-ID cache collisions.
            self._geometry_cache[key] = (tensors, trees)
            if len(self._geometry_cache) > 4:
                self._geometry_cache.popitem(last=False)
        self._geometry_cache.move_to_end(key)
        return self._geometry_cache[key][1]

    def _relations(self, tree: _ObservedTree) -> Tensor:
        if self.relation_kind == "hops":
            return tree.hops.clamp_max(self.max_rel_distance)
        return relative_tmrca_bins(tree, self.max_rel_distance)

    def _encode(self, h: Tensor, relations: Tensor) -> Tensor:
        # Only occupied relation bins are needed for this tree. This preserves
        # the exact lookup equation while avoiding a [heads,n,1001] work tensor.
        occupied, inverse = torch.unique(relations, sorted=True, return_inverse=True)
        relations = inverse.reshape_as(relations)
        tables = tuple(table.weight[occupied] for table in
                       (self.relative_query, self.relative_key, self.relative_value))
        for layer in self.layers:
            h = layer(h, relations, tables)
        return self.final_norm(h)

    def _subgraph(self, seq: GraphSequenceInput, tree: _ObservedTree, query: int) -> Tensor:
        panel = seq.x[tree.nodes.to(seq.x.device)][:, self.reference_columns].detach().cpu()
        if not bool(((panel == 0) | (panel == 1)).all()) or bool((panel.sum(-1) > 1).any()):
            raise ValueError("Reference columns must be observed one-hot panel labels")
        if float(panel[query].sum()) != 0:
            raise ValueError("Queried ancestry cannot be supplied as a reference label")
        refs = (panel.sum(-1) > 0).nonzero(as_tuple=False).flatten()
        if len(refs) == 0:
            raise ValueError("Nearest-reference scope requires observed reference leaves")
        # Include all cutoff ties. Never resolve an equal-TMRCA tie using ID values.
        distance = tree.times[tree.mrca[query, refs]]
        cutoff = distance.kthvalue(min(self.num_references, len(refs))).values
        refs = refs[distance <= cutoff]
        selected = [query, *refs.tolist()]
        parent = int(tree.parent[query])
        for _ in range(self.num_ancestors):
            if parent < 0:
                break
            if parent not in selected:
                selected.append(parent)
            parent = int(tree.parent[parent])
        return torch.tensor(selected, dtype=torch.long)

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        if self.validate_inputs:
            validate_model_input(seq)
        if seq.x.shape[1] != self.input_dim:
            raise ValueError("Input feature width differs from the declared common schema")
        indexes = query_node_index(seq)
        h = F.gelu(self.input_projection(seq.x))
        outputs = []
        for graph, tree in enumerate(self._geometry(seq)):
            nodes = tree.nodes.to(h.device)
            relation = self._relations(tree)
            # Map stable identities to local array positions, never to embeddings.
            local_map = {int(u): i for i, u in enumerate(tree.nodes.tolist())}
            queries = [local_map[int(u)] for u in indexes[graph].detach().cpu().tolist()]
            if self.token_scope == "full_tree":
                encoded = self._encode(h[nodes], relation.to(h.device))
                outputs.append(self.head(encoded[queries]))
            else:
                query_outputs = []
                for query in queries:
                    selected = self._subgraph(seq, tree, query)
                    subrelation = relation[selected[:, None], selected[None, :]]
                    with_cls = subrelation.new_full((len(selected) + 1, len(selected) + 1),
                                                   self.max_rel_distance + 1)
                    with_cls[1:, 1:] = subrelation
                    with_cls[0, 0] = 0
                    tokens = torch.cat((self.cls_token[None], h[nodes[selected.to(h.device)]]))
                    query_outputs.append(self.head(self._encode(tokens, with_cls.to(h.device))[0]))
                outputs.append(torch.stack(query_outputs))
        return torch.stack(outputs)[seq.target_graph]

    def configuration(self) -> dict:
        return {
            "scientific_status": "independent_paper_equation_common_input_adaptation",
            "paper": "10.64898/2026.04.05.714076v1",
            "input_dim": self.input_dim, "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
            "num_heads": self.num_heads, "token_scope": self.token_scope,
            "dropout": self.dropout_probability, "ffn_multiplier": self.ffn_multiplier,
            "input_projection": "linear_gelu", "normalization": "pre_layer_norm",
            "readout": "query_node_linear" if self.token_scope == "full_tree" else "learned_cls_linear",
            "distance_discretization": "integer_floor_observed_generations" if self.relation_kind == "elapsed_tmrca" else "integer_tree_hops",
            "relation_kind": self.relation_kind, "max_rel_distance": self.max_rel_distance,
            "time_column": self.time_column, "branch_column": self.branch_column,
            "time_scale": self.time_scale, "num_references": self.num_references,
            "num_ancestors": self.num_ancestors, "cutoff_ties": "include_all",
            "cross_tree_aggregation": "none", "learned_node_id_embeddings": False,
            "truth_features": [], "external_pretraining": False,
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }


class WholeTreeGraphTransformerControl(ARGMixTreeControl):
    """Generic full-tree GRPE control using observed unweighted tree-hop bins.

    This is a graph-transformer family control, not a published-model
    reproduction. It keeps the same dense Q/K/V relative attention and local
    query readout while replacing elapsed TMRCA with tree shortest-path length.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 *, max_rel_distance: int = 32, **options):
        if options.pop("token_scope", "full_tree") != "full_tree":
            raise ValueError("Generic whole-tree control always uses full_tree scope")
        super().__init__(input_dim, num_classes, hidden_dim, num_layers, num_heads, dropout,
                         max_rel_distance=max_rel_distance, token_scope="full_tree", **options)
        self.relation_kind = "hops"

    def configuration(self) -> dict:
        result = super().configuration()
        result["scientific_status"] = "independent_whole_tree_graph_transformer_family_control"
        result["paper"] = None
        return result

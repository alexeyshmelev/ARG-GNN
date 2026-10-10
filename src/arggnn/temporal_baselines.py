"""Independent temporal baseline families over the shared observed tensor input.

These controls are not exact reproductions of STAGIN, DySAT, GraphSSM, or any
published supra-graph method. Both return logits [positions, query leaves,
classes]. IDs define correspondence only and are never embedded as features.
Neither model promises exact invariance to subdividing an unchanged interval.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import (
    GraphSequenceInput,
    SharedGraphEncoder,
    query_node_index,
    validate_model_input,
)


def _graph_mean(h: Tensor, node_graph: Tensor, num_graphs: int) -> Tensor:
    aggregate = h.new_zeros((num_graphs, h.shape[-1]))
    aggregate.index_add_(0, node_graph, h)
    count = torch.bincount(node_graph, minlength=num_graphs).to(h.dtype)
    return aggregate / count.clamp_min(1)[:, None]


def _interval_features(intervals: Tensor, dtype: torch.dtype) -> Tensor:
    """Normalized endpoints, midpoint and span, independent of serialization."""
    left, right = intervals.to(dtype).unbind(dim=-1)
    return torch.stack((left, right, (left + right) / 2, right - left), dim=-1)


class LeafTemporalTransformer(nn.Module):
    """Whole-tree encoding followed by temporal attention for each query leaf.

    The final shared GNN bank is pooled over each entire tree and concatenated
    with the corresponding query-leaf representation. Interval coordinates
    condition the resulting tokens. A bidirectional temporal Transformer models
    each query leaf's sequence. Fixed-coordinate decoder queries use observed
    containing-tree leaf features and position encodings to attend to that
    temporal memory. Global sequence context is available, as in the proposed
    offline ancestry task. This is an independently implemented late-encoding
    family control, not a causal forecaster or a published-method reproduction.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, temporal_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1,
                 validate: bool = True, encoder: nn.Module | None = None):
        super().__init__()
        if num_classes < 2 or hidden_dim < 1 or num_heads < 1:
            raise ValueError("num_classes >= 2 and positive dimensions are required")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if temporal_layers < 1:
            raise ValueError("temporal_layers must be positive")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.validate_inputs = validate
        self.encoder = encoder if encoder is not None else SharedGraphEncoder(
            input_dim, hidden_dim, num_layers
        )
        self.tree_token = nn.Linear(2 * hidden_dim, hidden_dim)
        self.interval_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=4 * hidden_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            block, num_layers=temporal_layers, norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.query_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.query_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        if self.validate_inputs:
            validate_model_input(seq)
        h = self.encoder(seq)[-1]
        leaf_indexes = query_node_index(seq)
        leaf_h = h[leaf_indexes]  # [G,Q,D]
        graph_h = _graph_mean(h, seq.node_graph, seq.intervals.shape[0])
        whole_tree = graph_h[:, None, :].expand_as(leaf_h)
        tokens = self.tree_token(torch.cat((leaf_h, whole_tree), dim=-1))
        tokens = tokens + self.interval_encoder(
            _interval_features(seq.intervals, h.dtype)
        )[:, None, :]
        genomic_order = torch.argsort(seq.intervals[:, 0])
        memory = self.temporal_encoder(tokens[genomic_order].transpose(0, 1))
        # Each leaf is an attention batch item, with P fixed-coordinate queries.
        p = seq.positions.to(h.dtype)
        coordinate = torch.stack((p, torch.sin(2 * math.pi * p),
                                  torch.cos(2 * math.pi * p)), dim=-1)
        local = leaf_h[seq.target_graph].transpose(0, 1)  # [Q,P,D]
        position_h = self.position_encoder(coordinate)[None, :, :].expand_as(local)
        query = self.query_projection(torch.cat((local, position_h), dim=-1))
        context, _ = self.query_attention(query, memory, memory, need_weights=False)
        contextual = self.query_norm(query + context)
        logits = self.head(torch.cat((local, contextual), dim=-1))
        return logits.transpose(0, 1).contiguous()


def leaf_correspondence_edges(seq: GraphSequenceInput) -> Tensor:
    """Bidirectional edges between identical leaves in genomic neighbors.

    Every observed stable leaf can carry context, including reference leaves.
    Internal nodes (ID -1) are never matched. A missing leaf is not connected
    across an intervening graph. The sorting join uses linear-size storage,
    rather than a dense graph-count by leaf-count lookup table.
    """
    leaf_nodes = (seq.node_ids >= 0).nonzero(as_tuple=False).flatten()
    if seq.intervals.shape[0] < 2 or leaf_nodes.numel() < 2:
        return seq.edge_index.new_empty((2, 0))
    graph_order = torch.argsort(seq.intervals[:, 0])
    graph_rank = torch.empty_like(graph_order)
    graph_rank[graph_order] = torch.arange(graph_order.numel(), device=graph_order.device)
    # Stable lexicographic sort by (identity, genomic rank).
    nodes = leaf_nodes[torch.argsort(graph_rank[seq.node_graph[leaf_nodes]], stable=True)]
    nodes = nodes[torch.argsort(seq.node_ids[nodes], stable=True)]
    previous, following = nodes[:-1], nodes[1:]
    same_id = seq.node_ids[previous] == seq.node_ids[following]
    adjacent = graph_rank[seq.node_graph[following]] == graph_rank[seq.node_graph[previous]] + 1
    previous, following = previous[same_id & adjacent], following[same_id & adjacent]
    return torch.stack((torch.cat((previous, following)),
                        torch.cat((following, previous))), dim=0)


class _SupraMessageLayer(nn.Module):
    """Separate incoming spatial and correspondence means with gap features."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.spatial_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.temporal_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gap_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _incoming_mean(h: Tensor, edges: Tensor, messages: Tensor) -> Tensor:
        aggregate = torch.zeros_like(h)
        degree = h.new_zeros(h.shape[0])
        if edges.numel():
            aggregate.index_add_(0, edges[1], messages)
            degree.index_add_(0, edges[1], h.new_ones(edges.shape[1]))
        return aggregate / degree.clamp_min(1)[:, None]

    def forward(self, h: Tensor, spatial_edges: Tensor, temporal_edges: Tensor,
                interval_features: Tensor, node_graph: Tensor) -> Tensor:
        spatial = self._incoming_mean(
            h, spatial_edges, self.spatial_message(h[spatial_edges[0]])
        )
        source, target = temporal_edges
        source_interval = interval_features[node_graph[source]]
        target_interval = interval_features[node_graph[target]]
        gaps = torch.stack((target_interval[:, 2] - source_interval[:, 2],
                            source_interval[:, 3], target_interval[:, 3]), dim=-1)
        temporal_messages = self.temporal_message(h[source]) + self.gap_encoder(gaps)
        temporal = self._incoming_mean(h, temporal_edges, temporal_messages)
        return self.norm(h + self.update(torch.cat((h, spatial, temporal), dim=-1)))


class LeafCorrespondenceSupraMPNN(nn.Module):
    """Lift the sequence into a graph with spatial and stable-leaf edges.

    Independent whole-tree GNN encoding initializes the lifted graph. Additional
    relation-specific message passing uses native tree edges and correspondence
    edges between genomic neighbors. Interval endpoints, midpoint and span are
    supplied as observed coordinate features. Classification reads the query
    leaf in its containing graph and conditions on the fixed query position.
    This control has finite temporal receptive field set by supra_layers.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, supra_layers: int = 2,
                 dropout: float = 0.1, validate: bool = True,
                 encoder: nn.Module | None = None):
        super().__init__()
        if num_classes < 2 or hidden_dim < 1 or supra_layers < 1:
            raise ValueError("num_classes >= 2 and positive dimensions/layers are required")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.validate_inputs = validate
        self.encoder = encoder if encoder is not None else SharedGraphEncoder(
            input_dim, hidden_dim, num_layers
        )
        self.interval_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList(
            _SupraMessageLayer(hidden_dim, dropout) for _ in range(supra_layers)
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        if self.validate_inputs:
            validate_model_input(seq)
        h = self.encoder(seq)[-1]
        intervals = _interval_features(seq.intervals, h.dtype)
        h = h + self.interval_encoder(intervals)[seq.node_graph]
        temporal_edges = leaf_correspondence_edges(seq)
        for layer in self.layers:
            h = layer(h, seq.edge_index, temporal_edges, intervals, seq.node_graph)
        indexes = query_node_index(seq)[seq.target_graph]
        local = h[indexes]  # [P,Q,D]
        p = seq.positions.to(h.dtype)
        current = intervals[seq.target_graph]
        coordinate = torch.stack((p, p - current[:, 2], current[:, 3]), dim=-1)
        position_h = self.position_encoder(coordinate)[:, None, :].expand_as(local)
        return self.head(torch.cat((local, position_h), dim=-1))

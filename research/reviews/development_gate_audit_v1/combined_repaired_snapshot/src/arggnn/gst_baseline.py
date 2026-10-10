"""Independent GST graph-set adaptations of arXiv:2606.05116v1.

See docs/gst_baseline.md for equation mapping and the paper/source gate mismatch.
One GraphSequence is one set. No positional, ID or correspondence embeddings
are used. Record and span controls share parameters and differ only in key mass.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import GraphSequenceInput, query_node_index, validate_model_input

PAPER_URL = "https://arxiv.org/html/2606.05116v1"
INSPECTED_SOURCE_COMMIT = "6469d56aa131463325058b897b1e435c6d2edcd2"


def _graph_mean(h: Tensor, graph: Tensor, graphs: int) -> Tensor:
    total = h.new_zeros(graphs, h.shape[-1])
    total.index_add_(0, graph, h)
    count = torch.bincount(graph, minlength=graphs).to(h.dtype)
    return total / count[:, None]


class GSTGraphNorm(nn.Module):
    """Per-graph learned mean subtraction and affine normalization.

    mu_g=mean(h_i); r_i=h_i-alpha*mu_g; variance_g=mean(r_i**2).
    This is GraphNorm, including its learned mean scale, rather than LayerNorm.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, h: Tensor, graph: Tensor, graphs: int) -> Tensor:
        mean = _graph_mean(h, graph, graphs)
        centered = h - self.mean_scale * mean[graph]
        variance = _graph_mean(centered.square(), graph, graphs)
        return self.weight * centered / (variance[graph] + self.eps).sqrt() + self.bias


class GSTNormalizedGCN(nn.Module):
    """Native tensor GCN using supplied edges plus explicit self loops.

    For adjacency A_{dst,src}, propagate D^-1/2 (A+sI) D^-1/2 H W+b.
    The source requests GCNConv(improved=True), motivating default s=2.
    Directed edges are preserved. The project supplies both directions for trees.
    Supplied self edges are replaced by exactly one loop of the declared weight.
    """
    def __init__(self, hidden_dim: int, self_loop_weight: float = 2.):
        super().__init__()
        if not math.isfinite(self_loop_weight) or self_loop_weight <= 0:
            raise ValueError("self_loop_weight must be finite and positive")
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.self_loop_weight = self_loop_weight

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index[:, edge_index[0] != edge_index[1]]
        nodes = torch.arange(len(h), device=h.device)
        source = torch.cat((source, nodes))
        target = torch.cat((target, nodes))
        edge_weight = h.new_ones(len(source))
        edge_weight[-len(h):] = self.self_loop_weight
        degree = h.new_zeros(len(h))
        degree.index_add_(0, target, edge_weight)
        inverse = degree.rsqrt()
        weights = edge_weight * inverse[source] * inverse[target]
        projected = self.linear(h)
        result = torch.zeros_like(projected)
        result.index_add_(0, target, projected[source] * weights[:, None])
        return result + self.bias


class GSTAttentionalPool(nn.Module):
    """Equations 3-5: softmax scoring separately within each graph."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, h: Tensor, graph: Tensor, graphs: int) -> Tensor:
        score = self.score(h).squeeze(-1)
        maximum = score.new_full((graphs,), -torch.inf)
        maximum.scatter_reduce_(0, graph, score, reduce="amax", include_self=True)
        unnormalized = (score - maximum[graph]).exp()
        total = score.new_zeros(graphs)
        total.index_add_(0, graph, unnormalized)
        alpha = unnormalized / total[graph]
        pooled = h.new_zeros(graphs, h.shape[-1])
        pooled.index_add_(0, graph, alpha[:, None] * h)
        return pooled


class GSTBlock(nn.Module):
    """Local GCN -> graph pooling -> set Transformer -> gated node feedback.

    Paper x^(2) is undefined. The inspected source has no intervening node FFN,
    so x^(2) is explicitly interpreted as its post-GCN residual x^(1).
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.,
                 routing: str = "cross_attention", gate_variant: str = "paper",
                 ffn_multiplier: int = 2, self_loop_weight: float = 2.):
        super().__init__()
        if routing not in {"cross_attention", "broadcast"}:
            raise ValueError("routing must be cross_attention or broadcast")
        if gate_variant not in {"paper", "source"}:
            raise ValueError("gate_variant must be paper or source")
        self.routing, self.gate_variant = routing, gate_variant
        self.gcn = GSTNormalizedGCN(hidden_dim, self_loop_weight)
        self.graph_norm = GSTGraphNorm(hidden_dim)
        self.local_dropout = nn.Dropout(dropout)
        self.pool = GSTAttentionalPool(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.set_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                                   batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_multiplier*hidden_dim), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_multiplier*hidden_dim, hidden_dim),
                                 nn.Dropout(dropout))
        if routing == "cross_attention":
            self.node_norm = nn.LayerNorm(hidden_dim)
            self.node_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                                        batch_first=True)
        if gate_variant == "paper":
            # Equations 10-11. Both final weights and bias are zero initially.
            self.gate = nn.Sequential(nn.Linear(3*hidden_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, 2*hidden_dim))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        else:
            # Mathematical source-gate control: ungated home token plus set gate.
            self.gate = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim))
            nn.init.constant_(self.gate[-1].bias, -1.)

    @staticmethod
    def _attention_mass(log_mass: Tensor | None, queries: int) -> Tensor | None:
        return None if log_mass is None else log_mass[None, :].expand(queries, -1)

    def forward(self, h: Tensor, edge_index: Tensor, graph: Tensor, graphs: int,
                log_mass: Tensor | None = None) -> Tensor:
        # Equations 1-2. Feedback from the previous block is this GCN's input.
        local = h + self.local_dropout(F.silu(self.graph_norm(self.gcn(h, edge_index), graph, graphs)))
        tokens = self.pool(local, graph, graphs)
        normalized = self.token_norm(tokens)[None, :, :]
        attention, _ = self.set_attention(normalized, normalized, normalized,
            attn_mask=self._attention_mass(log_mass, graphs), need_weights=False)
        context_tokens = tokens + attention[0]
        context_tokens = context_tokens + self.ffn(self.ffn_norm(context_tokens))
        home = tokens[graph]
        if self.routing == "broadcast":
            context = context_tokens[graph]
        else:
            context, _ = self.node_attention(self.node_norm(local)[None, :, :],
                context_tokens[None, :, :], context_tokens[None, :, :],
                attn_mask=self._attention_mass(log_mass, len(local)), need_weights=False)
            context = context[0]
        if self.gate_variant == "paper":
            home_gate, set_gate = self.gate(torch.cat((local, home, context), dim=-1)).sigmoid().chunk(2, dim=-1)
            return local + home_gate*home + set_gate*context
        set_gate = self.gate(torch.cat((local, context), dim=-1)).sigmoid()
        return local + home + set_gate*context


class GSTAdaptation(nn.Module):
    """GST-style fixed-coordinate query-leaf classifier, returning [P,Q,K].

    ``token_weighting=record`` uses native counting measure over graph tokens.
    ``token_weighting=span`` adds log interval width to every graph-token key
    score in both set self-attention and node-to-set cross-attention.

    Endpoints, query coordinates and integer IDs are never learned features.
    IDs only select external query columns. There is one set per forward call.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.,
                 validate: bool = True, routing: str = "cross_attention",
                 token_weighting: str = "record", gate_variant: str = "paper",
                 ffn_multiplier: int = 2, self_loop_weight: float = 2.):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, num_heads, ffn_multiplier) < 1 or num_classes < 2:
            raise ValueError("Positive dimensions/layers and num_classes >= 2 are required")
        if hidden_dim % num_heads or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("hidden_dim must divide into heads, with dropout in [0,1)")
        if token_weighting not in {"record", "span"}:
            raise ValueError("token_weighting must be record or span")
        self.hidden_dim, self.num_classes = hidden_dim, num_classes
        self.validate_inputs, self.token_weighting = validate, token_weighting
        self.routing, self.gate_variant = routing, gate_variant
        self.num_heads, self.dropout = num_heads, dropout
        self.ffn_multiplier, self.self_loop_weight = ffn_multiplier, self_loop_weight
        self.input_dim = input_dim
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(GSTBlock(hidden_dim, num_heads, dropout, routing, gate_variant,
                                           ffn_multiplier, self_loop_weight) for _ in range(num_layers))
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, num_classes))

    @property
    def control_name(self) -> str:
        route = "ca" if self.routing == "cross_attention" else "bc"
        suffix = "" if self.gate_variant == "paper" else "_source_gate"
        return f"gst_{route}_{self.token_weighting}{suffix}"

    def configuration_provenance(self) -> dict[str, Any]:
        return {"control_name": self.control_name, "status": "independent baseline adaptation, not a reproduction",
                "paper": PAPER_URL, "inspected_source_commit": INSPECTED_SOURCE_COMMIT,
                "input_dim": self.input_dim, "hidden_dim": self.hidden_dim, "num_layers": len(self.blocks),
                "num_heads": self.num_heads, "num_classes": self.num_classes, "dropout": self.dropout,
                "ffn_multiplier": self.ffn_multiplier, "self_loop_weight": self.self_loop_weight,
                "routing": self.routing, "token_weighting": self.token_weighting,
                "gate_variant": self.gate_variant, "stochastic_depth": 0.,
                "input_projection": "learned linear", "position_encodings": False,
                "learned_numerical_id_embeddings": False,
                "readout": "MLP on query leaf in containing graph; fixed coordinates select graph only"}

    def encode_nodes(self, seq: GraphSequenceInput) -> Tensor:
        if self.validate_inputs:
            validate_model_input(seq)
        h = self.input_linear(seq.x)
        log_mass = None
        if self.token_weighting == "span":
            # Preserve physical width precision before passing additive bias to MHA.
            widths = seq.intervals[:, 1] - seq.intervals[:, 0]
            log_mass = (widths.log() - widths.sum().log()).to(h.dtype)
        for block in self.blocks:
            h = block(h, seq.edge_index, seq.node_graph, seq.intervals.shape[0], log_mass)
        return h

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        h = self.encode_nodes(seq)
        return self.head(h[query_node_index(seq)[seq.target_graph]])


class GSTRecordTokenAdaptation(GSTAdaptation):
    """Separately named native graph-record counting-measure control."""
    def __init__(self, *args, **kwargs):
        if "token_weighting" in kwargs:
            raise TypeError("GSTRecordTokenAdaptation fixes token_weighting=record")
        super().__init__(*args, token_weighting="record", **kwargs)


class GSTSpanTokenAdaptation(GSTAdaptation):
    """Separately named duration-measure adaptation, without coordinate order."""
    def __init__(self, *args, **kwargs):
        if "token_weighting" in kwargs:
            raise TypeError("GSTSpanTokenAdaptation fixes token_weighting=span")
        super().__init__(*args, token_weighting="span", **kwargs)

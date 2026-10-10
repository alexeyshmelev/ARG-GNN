"""Small CPU/GPU-native graph-sequence baselines and an experimental operator.

This is research code, not a reproduction of GST, ARGMix, or ARGformer.
No node-ID embeddings and no ground-truth ancestry annotations are used.
"""
from __future__ import annotations

from typing import Protocol
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphSequenceInput(Protocol):
    x: Tensor
    edge_index: Tensor
    node_graph: Tensor
    node_ids: Tensor
    clades: Tensor
    intervals: Tensor
    query_node_ids: Tensor
    positions: Tensor
    target_graph: Tensor


def validate_model_input(seq: GraphSequenceInput) -> None:
    """Validate one independent sequence, including a contiguous tile within global [0,1]."""
    fields = ('x', 'edge_index', 'node_graph', 'node_ids', 'clades',
              'intervals', 'query_node_ids', 'positions', 'target_graph')
    if any(not isinstance(getattr(seq, k, None), Tensor) for k in fields):
        raise ValueError('All graph-sequence fields must be tensors')
    if any(getattr(seq, k).device != seq.x.device for k in fields):
        raise ValueError('All graph-sequence tensors must share one device')
    if seq.x.ndim != 2 or min(seq.x.shape) < 1 or not seq.x.is_floating_point():
        raise ValueError('x must be a nonempty floating [N,F] tensor')
    n = seq.x.shape[0]
    if seq.intervals.ndim != 2 or seq.intervals.shape[1] != 2 or seq.intervals.shape[0] < 1:
        raise ValueError('intervals must have shape [G,2] with G >= 1')
    if not seq.intervals.is_floating_point():
        raise ValueError('intervals must be floating')
    g = seq.intervals.shape[0]
    for name in ('node_graph', 'node_ids'):
        t = getattr(seq, name)
        if t.shape != (n,) or t.dtype != torch.long:
            raise ValueError(f'{name} must be long [N]')
    if seq.edge_index.ndim != 2 or seq.edge_index.shape[0] != 2 or seq.edge_index.dtype != torch.long:
        raise ValueError('edge_index must be long [2,E]')
    if seq.clades.ndim != 2 or seq.clades.shape[0] != n:
        raise ValueError('clades must have shape [N,C]')
    if not bool(torch.isfinite(seq.clades).all()) or not bool(((seq.clades >= 0) & (seq.clades <= 1)).all()):
        raise ValueError('clades must contain finite descendant indicators in [0,1]')
    if seq.query_node_ids.ndim != 1 or seq.query_node_ids.numel() == 0 or seq.query_node_ids.dtype != torch.long:
        raise ValueError('query_node_ids must be nonempty long [Q]')
    if bool((seq.query_node_ids < 0).any()) or torch.unique(seq.query_node_ids).numel() != seq.query_node_ids.numel():
        raise ValueError('query_node_ids must be unique nonnegative identities')
    if seq.positions.ndim != 1 or seq.positions.numel() == 0 or not seq.positions.is_floating_point():
        raise ValueError('positions must be nonempty floating [P]')
    if seq.target_graph.shape != seq.positions.shape or seq.target_graph.dtype != torch.long:
        raise ValueError('target_graph must be long [P]')
    if not bool(torch.isfinite(seq.x).all()) or not bool(torch.isfinite(seq.intervals).all()) or not bool(torch.isfinite(seq.positions).all()):
        raise ValueError('Features, intervals and positions must be finite')
    if bool((seq.node_ids < -1).any()):
        raise ValueError('node_ids use nonnegative stable leaf IDs or -1 for internal nodes')
    if bool(((seq.node_graph < 0) | (seq.node_graph >= g)).any()):
        raise ValueError('node_graph outside [0,G)')
    if bool(((seq.target_graph < 0) | (seq.target_graph >= g)).any()):
        raise ValueError('target_graph outside [0,G)')
    counts = torch.bincount(seq.node_graph, minlength=g)
    if bool((counts == 0).any()):
        raise ValueError('Every graph must contain at least one node')
    if seq.edge_index.numel():
        if bool(((seq.edge_index < 0) | (seq.edge_index >= n)).any()):
            raise ValueError('edge_index outside [0,N)')
        if bool((seq.node_graph[seq.edge_index[0]] != seq.node_graph[seq.edge_index[1]]).any()):
            raise ValueError('Local graph edges must not cross graph boundaries')
    ordered = seq.intervals[torch.argsort(seq.intervals[:, 0])]
    if bool((ordered[:, 1] <= ordered[:, 0]).any()):
        raise ValueError('Tree intervals must have strictly positive width')
    if float(ordered[0, 0]) < 0.0 or float(ordered[-1, 1]) > 1.0:
        raise ValueError('Tree intervals must stay within the normalized global range [0,1]')
    bounds = getattr(seq, 'window_bounds', None)
    if bounds is not None:
        if not isinstance(bounds, Tensor) or bounds.shape != (2,) or bounds.device != seq.x.device:
            raise ValueError('window_bounds must be a [2] tensor on the input device')
        if not bool(torch.isfinite(bounds).all()) or not torch.equal(ordered[0, :1], bounds[:1].to(ordered.dtype)) or not torch.equal(ordered[-1, 1:], bounds[1:].to(ordered.dtype)):
            raise ValueError('Tree intervals must cover the complete declared window_bounds')
    elif float(ordered[0, 0]) != 0.0 or float(ordered[-1, 1]) != 1.0:
        raise ValueError('Without window_bounds, tree intervals must cover the complete [0,1] range')
    if ordered.shape[0] > 1 and not torch.equal(ordered[:-1, 1], ordered[1:, 0]):
        raise ValueError('Tree intervals must have no gaps or overlaps')
    if bool(((seq.positions < 0) | (seq.positions >= 1)).any()):
        raise ValueError('Query positions must be in [0,1)')
    target_intervals = seq.intervals[seq.target_graph]
    if bool(((seq.positions < target_intervals[:, 0]) | (seq.positions >= target_intervals[:, 1])).any()):
        raise ValueError('target_graph must contain each query position')
    for graph in range(g):
        ids = seq.node_ids[seq.node_graph == graph]
        stable = ids[ids >= 0]
        if torch.unique(stable).numel() != stable.numel():
            raise ValueError('Stable leaf identities must be unique within every graph')
        matches = ids[:, None] == seq.query_node_ids[None, :]
        if not bool((matches.sum(dim=0) == 1).all()):
            raise ValueError('Every query leaf must occur exactly once in every graph')


def log_interval_kernel_mass(positions: Tensor, intervals: Tensor, tau: Tensor) -> Tensor:
    """Stable log integral of exp(-abs(x-y)/tau) over each source interval.

    Returns [P,G]. Uses expm1 for narrow intervals and never exponentiates
    large negative source distances before log-softmax normalization.
    """
    if tau.numel() != 1 or not bool(torch.isfinite(tau).all()) or not bool((tau > 0).all()):
        raise ValueError('tau must be one finite positive scalar')
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not bool((intervals[:, 1] > intervals[:, 0]).all()):
        raise ValueError('Kernel intervals must have positive width and shape [G,2]')
    x = positions[:, None]
    a, b = intervals[:, 0][None, :], intervals[:, 1][None, :]
    width = b - a
    tiny = torch.finfo(intervals.dtype).tiny
    log_piece = torch.log((-torch.expm1(-width / tau)).clamp_min(tiny))
    left = torch.log(tau) + (b - x) / tau + log_piece
    right = torch.log(tau) + (x - a) / tau + log_piece
    left_length = torch.minimum((x - a).clamp_min(0), width)
    right_length = torch.minimum((b - x).clamp_min(0), width)
    middle_mass = -torch.expm1(-left_length / tau) - torch.expm1(-right_length / tau)
    middle = torch.log(tau) + torch.log(middle_mass.clamp_min(tiny))
    return torch.where(x >= b, left, torch.where(x <= a, right, middle))


class MeanMessageLayer(nn.Module):
    """Residual incoming-neighbor mean MPNN using supplied directed edges."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        aggregate = torch.zeros_like(h)
        degree = h.new_zeros(h.shape[0])
        if edge_index.numel():
            src, dst = edge_index
            aggregate.index_add_(0, dst, h[src])
            degree.index_add_(0, dst, h.new_ones(dst.numel()))
        aggregate = aggregate / degree.clamp_min(1)[:, None]
        update = F.gelu(self.self_linear(h) + self.neighbor_linear(aggregate))
        return self.norm(h + update)


class SharedGraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers) < 1:
            raise ValueError('input_dim, hidden_dim and num_layers must be positive')
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(MeanMessageLayer(hidden_dim) for _ in range(num_layers))

    def forward(self, seq: GraphSequenceInput) -> list[Tensor]:
        h = F.gelu(self.input_linear(seq.x))
        banks = [h]
        for layer in self.layers:
            h = layer(h, seq.edge_index)
            banks.append(h)
        return banks


def query_node_index(seq: GraphSequenceInput) -> Tensor:
    """Return global node indexes [G,Q], with matching by stable leaf identity."""
    rows = []
    for graph in range(seq.intervals.shape[0]):
        nodes = (seq.node_graph == graph).nonzero(as_tuple=False).flatten()
        match = seq.node_ids[nodes, None] == seq.query_node_ids[None, :]
        rows.append(nodes[match.to(torch.long).argmax(dim=0)])
    return torch.stack(rows)


class _BaseModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, validate: bool = True):
        super().__init__()
        if num_classes < 2:
            raise ValueError('num_classes must be at least two')
        self.encoder = SharedGraphEncoder(input_dim, hidden_dim, num_layers)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.validate_inputs = validate

    def prepare(self, seq: GraphSequenceInput) -> tuple[list[Tensor], Tensor]:
        if self.validate_inputs:
            validate_model_input(seq)
        return self.encoder(seq), query_node_index(seq)


class IndependentGNN(_BaseModel):
    """Independent local tree encoder and query-leaf classification."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        banks, indexes = self.prepare(seq)
        return self.head(banks[-1][indexes[seq.target_graph]])


class GNNSetPool(_BaseModel):
    """Late span-weighted graph-mean set context, shared across query leaves."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(2 * self.hidden_dim, self.num_classes)

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        banks, indexes = self.prepare(seq)
        h = banks[-1]
        g = seq.intervals.shape[0]
        pooled = h.new_zeros(g, self.hidden_dim)
        pooled.index_add_(0, seq.node_graph, h)
        counts = torch.bincount(seq.node_graph, minlength=g).to(h.dtype)
        pooled = pooled / counts[:, None]
        mass = (seq.intervals[:, 1] - seq.intervals[:, 0]).to(h.dtype)
        context = (mass[:, None] * pooled).sum(dim=0) / mass.sum()
        local = h[indexes[seq.target_graph]]
        return self.head(torch.cat((local, context.expand_as(local)), dim=-1))


class GNNGRU(_BaseModel):
    """Leaf-aligned bidirectional GRU with width and midpoint-gap features.

    This is an honest small graph-sequence baseline, not a paper reproduction.
    Its repeated per-record recurrence is not interval-refinement invariant.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gru = nn.GRU(self.hidden_dim + 2, self.hidden_dim,
                          batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * self.hidden_dim, self.num_classes)

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        banks, indexes = self.prepare(seq)
        order = torch.argsort(seq.intervals[:, 0])
        ordered_intervals = seq.intervals[order]
        midpoint = ordered_intervals.mean(dim=1)
        gaps = torch.cat((midpoint.new_zeros(1), midpoint[1:] - midpoint[:-1]))
        distances = torch.stack((ordered_intervals[:, 1] - ordered_intervals[:, 0], gaps), dim=-1)
        leaf = banks[-1][indexes[order]].transpose(0, 1)
        distances = distances.to(leaf.dtype)[None, :, :].expand(leaf.shape[0], -1, -1)
        encoded, _ = self.gru(torch.cat((leaf, distances), dim=-1))
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        query_encoded = encoded[:, inverse[seq.target_graph], :].transpose(0, 1)
        return self.head(query_encoded)


class LineageCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, correspondence: bool, initial_tau: float):
        super().__init__()
        if not math.isfinite(initial_tau) or initial_tau <= 1e-4:
            raise ValueError('initial_tau must be finite and greater than 1e-4')
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        adjusted_tau = initial_tau - 1e-4
        inverse_softplus = adjusted_tau + math.log(-math.expm1(-adjusted_tau))
        self.raw_tau = nn.Parameter(torch.tensor(inverse_softplus))
        self.correspondence = correspondence
        # Retained in all ablations for an identical parameter count.
        self.identity_strength = nn.Parameter(torch.tensor(1.0))
        self.clade_strength = nn.Parameter(torch.tensor(1.0))
        self.scale = hidden_dim ** -0.5

    @property
    def tau(self) -> Tensor:
        return F.softplus(self.raw_tau) + 1e-4

    def forward(self, state: Tensor, source: Tensor, target_nodes: Tensor,
                seq: GraphSequenceInput, position: Tensor, weighting: str,
                source_projections: tuple[Tensor, Tensor] | None = None) -> Tensor:
        # Preserve coordinate precision before reducing log masses to feature dtype.
        intervals = seq.intervals
        pos = position.reshape(1).to(intervals.dtype)
        if weighting == 'exact':
            log_mass = log_interval_kernel_mass(pos, intervals, self.tau.to(intervals.dtype))[0]
        else:
            midpoint = intervals.mean(dim=1)
            log_mass = -(pos[0] - midpoint).abs() / self.tau
            if weighting == 'midpoint':
                log_mass = log_mass + (intervals[:, 1] - intervals[:, 0]).log()
        counts = torch.bincount(seq.node_graph, minlength=intervals.shape[0]).to(intervals.dtype)
        node_log_mass = (log_mass - counts.log())[seq.node_graph].to(state.dtype)
        source_key, source_value = (self.key(source), self.value(source)) if source_projections is None else source_projections
        score = self.query(state) @ source_key.T * self.scale
        if self.correspondence:
            target_ids = seq.node_ids[target_nodes]
            identity = ((target_ids[:, None] == seq.node_ids[None, :]) & (target_ids[:, None] >= 0)).to(state.dtype)
            clades = seq.clades.to(state.dtype)
            target_clades = clades[target_nodes]
            intersection = target_clades @ clades.T
            union = target_clades.sum(dim=1)[:, None] + clades.sum(dim=1)[None, :] - intersection
            overlap = intersection / union.clamp_min(1)
            score = score + self.identity_strength * identity + self.clade_strength * overlap
        weights = torch.softmax(score + node_log_mass[None, :], dim=-1)
        message = weights @ source_value
        gate = torch.sigmoid(self.gate(torch.cat((state, message), dim=-1)))
        return self.norm(state + gate * message)


class IntervalLineageGNN(_BaseModel):
    """Experimental interval-integrated context-conditioned graph operator.

    Banks are local-only at every depth. Exact weights are additive under
    splitting an unchanged source interval. Queries live at fixed positions.
    """
    def __init__(self, *args, weighting: str = 'exact', correspondence: bool = True,
                 fusion: str = 'feedback', initial_tau: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        if weighting not in {'exact', 'midpoint', 'equal'}:
            raise ValueError('weighting must be exact, midpoint, or equal')
        if fusion not in {'feedback', 'late'}:
            raise ValueError('fusion must be feedback or late')
        self.weighting = weighting
        self.fusion = fusion
        self.cross_layers = nn.ModuleList(LineageCrossAttention(self.hidden_dim, correspondence, initial_tau)
                                          for _ in self.encoder.layers)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        banks, _ = self.prepare(seq)
        # Independent source projections are reused across fixed query coordinates.
        source_projections = [(cross.key(banks[l + 1]), cross.value(banks[l + 1]))
                              for l, cross in enumerate(self.cross_layers)]
        output = []
        for p in range(seq.positions.numel()):
            graph = seq.target_graph[p]
            target_nodes = (seq.node_graph == graph).nonzero(as_tuple=False).flatten()
            local_map = torch.full((seq.x.shape[0],), -1, dtype=torch.long, device=seq.x.device)
            local_map[target_nodes] = torch.arange(target_nodes.numel(), device=seq.x.device)
            edge_mask = seq.node_graph[seq.edge_index[0]] == graph
            local_edges = local_map[seq.edge_index[:, edge_mask]]
            state = banks[0 if self.fusion == 'feedback' else -1][target_nodes]
            for layer_index, cross in enumerate(self.cross_layers):
                if self.fusion == 'feedback':
                    state = self.encoder.layers[layer_index](state, local_edges)
                state = cross(state, banks[layer_index + 1], target_nodes,
                              seq, seq.positions[p], self.weighting,
                              source_projections[layer_index])
            match = seq.node_ids[target_nodes, None] == seq.query_node_ids[None, :]
            leaf_indexes = match.to(torch.long).argmax(dim=0)
            output.append(self.head(state[leaf_indexes]))
        return torch.stack(output)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def build_model(name: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
    """Stable names for experiment configuration, with explicit ablations."""
    baselines = {'independent': IndependentGNN, 'set_pool': GNNSetPool, 'gru': GNNGRU}
    if name in baselines:
        return baselines[name](input_dim, num_classes, **kwargs)
    variants = {
        'interval_lineage': {},
        'no_correspondence': {'correspondence': False},
        'midpoint_weights': {'weighting': 'midpoint'},
        'equal_tree_weights': {'weighting': 'equal'},
        'late_fusion': {'fusion': 'late'},
    }
    if name not in variants:
        raise ValueError(f'Unknown model {name!r}. Choose from {tuple(baselines) + tuple(variants)}')
    options = dict(kwargs)
    options.update(variants[name])
    return IntervalLineageGNN(input_dim, num_classes, **options)

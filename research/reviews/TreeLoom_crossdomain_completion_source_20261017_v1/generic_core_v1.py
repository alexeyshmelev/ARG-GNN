"""Generic fixed-entity port of the current TreeLoom stages.

Source only until ROOT separately reviews, registers and admits a consumer.
This is a disclosed graph-backbone adaptation, not the ancestry model class.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


@dataclass(frozen=True)
class ObservedHistoryV1:
    """Targets are external. Column identities are fixed throughout each history."""
    x: torch.Tensor                 # [B,T,N,F], observed features only
    edge_index: torch.Tensor        # [2,E], native non-self geographic edges
    durations: torch.Tensor         # [T], positive weeks
    entity_ids: tuple[str, ...]     # correspondence metadata, never numeric features
    target_indices: tuple[int, ...]
    content_indices: tuple[tuple[int, ...], ...]

    def validate(self):
        if self.x.ndim != 4 or self.x.dtype != torch.float32:
            raise ValueError("FP32 [batch,time,entity,feature] observations required")
        b, t, n, f = self.x.shape
        if min(b, t, n, f) < 1 or len(self.entity_ids) != n or len(set(self.entity_ids)) != n:
            raise ValueError("Nonempty unique persistent entity columns required")
        if len(self.target_indices) != b or len(self.content_indices) != b:
            raise ValueError("One external target index and history receipt per sample")
        if any(len(row) != t or len(set(row)) != t or
               any(type(i) is not int or i >= at for i in row)
               for at, row in zip(self.target_indices, self.content_indices)):
            raise ValueError("Every content week must precede its external forecast target")
        if (self.durations.shape != (t,) or self.durations.dtype != torch.float32
                or self.durations.device != self.x.device
                or not bool(torch.isfinite(self.durations).all())
                or not bool((self.durations > 0).all())):
            raise ValueError("Positive finite same-device FP32 observed dwell times")
        edge = self.edge_index
        if edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2 or edge.device != self.x.device:
            raise ValueError("Same-device long [2,E] local edge index")
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= n or
                           bool((edge[0] == edge[1]).any())):
            raise ValueError("Local entity edges without duplicate explicit self route")
        if not bool(torch.isfinite(self.x).all()):
            raise ValueError("No target/future-mask NaN may enter an observed history")
        return self


class _MeanNeighborsV1(MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean", flow="source_to_target", node_dim=0)

    def forward(self, value, edge_index):
        return self.propagate(edge_index, x=value, size=(len(value), len(value)))

    def message(self, x_j):
        return x_j


class SharedGraphPassV1(nn.Module):
    """Two local mean-message fields on a general graph, used twice with shared weights.

    The full ancestral traversals/cavity exclusion are deliberately replaced.
    No cyclic geography is represented as a fictitious rooted genealogy.
    """
    def __init__(self):
        super().__init__()
        self.aggregate = _MeanNeighborsV1()
        self.up_message = nn.Linear(32, 32, bias=False)
        self.up_update = nn.Linear(64, 32)
        self.up_norm = nn.LayerNorm(32)
        self.down_message = nn.Linear(32, 32, bias=False)
        self.down_update = nn.Linear(32, 32)
        self.down_norm = nn.LayerNorm(32)

    def forward(self, base, edges):
        neighbor = self.aggregate(self.up_message(base), edges)
        upward = self.up_norm(torch.tanh(self.up_update(torch.cat((base, neighbor), -1))))
        outside = self.down_norm(torch.tanh(
            self.down_update(self.aggregate(self.down_message(upward), edges))))
        return upward, outside


class OwnStateResidualV1(nn.Module):
    """H045 own-state rank-eight algebra; no foreign episode or clade is invented."""
    def __init__(self):
        super().__init__()
        self.U = nn.Linear(32, 8, bias=False)
        self.V = nn.Linear(32, 8, bias=False)
        self.W = nn.Linear(32, 8, bias=False)
        self.O = nn.Linear(8, 32, bias=False)
        nn.init.zeros_(self.O.weight)

    def forward(self, upward, repeated_identity):
        result = self.O(self.U(upward) * self.V(upward) + self.W(upward))
        return result if repeated_identity else torch.zeros_like(result)


class StableEntityMemoryV1(nn.Module):
    """Exactly 513 weights: identity carry, exponential relaxation, bidirectional centering.

    This ports CutMemoryV1's stable-leaf equations to singleton node groups.
    Clock: weeks, length bounds .1..100 and initial length 5. No rate interpretation.
    """
    def __init__(self):
        super().__init__()
        self.U = nn.Linear(32, 8, bias=False)
        self.O = nn.Linear(8, 32, bias=False)
        self.ell_theta = nn.Parameter(torch.tensor(
            math.log((5.0 - .1) / (100.0 - 5.0)), dtype=torch.float32))
        nn.init.zeros_(self.O.weight)

    @property
    def ell_weeks(self):
        return .1 + (100.0 - .1) * self.ell_theta.sigmoid()

    def _direction(self, forcing, duration, reverse):
        order = range(len(forcing) - 1, -1, -1) if reverse else range(len(forcing))
        result = [None] * len(forcing)
        state = None
        rho = torch.exp(-duration / self.ell_weeks)
        weight = -torch.expm1(-duration / self.ell_weeks)
        for at in order:
            state = forcing[at] if state is None else rho[at] * state + weight[at] * forcing[at]
            result[at] = state
        return torch.stack(result)

    def forward(self, first_upward, first_outside, duration):
        # Input layout [T,B*N,32]; each row-column is the same physical entity.
        forcing = self.U(first_upward + first_outside)
        history = .5 * (self._direction(forcing, duration, False)
                         + self._direction(forcing, duration, True))
        return self.O(history - forcing)


class TreeLoomGenericForecastV1(nn.Module):
    """Two shared graph passes + stable-node memory + complete-history BiGRU readout."""
    def __init__(self, *, with_memory=True, input_dim=1):
        super().__init__()
        if input_dim != 1 or type(with_memory) is not bool:
            raise ValueError("Fixed count-only one-feature completion protocol")
        self.with_memory = with_memory
        self.input_projection = nn.Sequential(nn.Linear(1, 32), nn.Tanh())
        self.graph_pass = SharedGraphPassV1()  # The same object is called for both stages.
        self.own_state = OwnStateResidualV1()
        self.leaf_context = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
        self.forward_gru = nn.GRU(35, 32, num_layers=1)
        self.reverse_gru = nn.GRU(35, 32, num_layers=1)
        self.regression_head = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))
        # Construct last so common-module initialization order is independent of the arm.
        self.memory = StableEntityMemoryV1() if with_memory else None
        self.float()

    @staticmethod
    def _edges(edge, graphs, nodes):
        offset = torch.arange(graphs, device=edge.device, dtype=torch.long) * nodes
        return (edge[:, None, :] + offset[None, :, None]).reshape(2, -1)

    def encode_history(self, observed: ObservedHistoryV1):
        observed.validate()
        x = observed.x
        b, t, n, _ = x.shape
        # Time-major flattening aligns identity memory and independent graph snapshots.
        base = self.input_projection(x.permute(1, 0, 2, 3).reshape(t*b*n, 1))
        edges = self._edges(observed.edge_index, t*b, n)
        up, outside = self.graph_pass(base, edges)
        injection = self.own_state(up, t >= 2)
        if self.memory is not None:
            delta = self.memory(up.reshape(t, b*n, 32),
                                outside.reshape(t, b*n, 32), observed.durations)
            injection = injection + delta.reshape(t*b*n, 32)
        second_up, second_out = self.graph_pass(base + injection, edges)
        return self.leaf_context(torch.cat((second_up, second_out), -1)).reshape(t, b*n, 32)

    def forward(self, observed: ObservedHistoryV1):
        context = self.encode_history(observed)
        t, bn, _ = context.shape
        duration = observed.durations
        prev = torch.cat((duration.new_zeros(1), .5*(duration[1:] + duration[:-1])))
        nxt = torch.cat((.5*(duration[1:] + duration[:-1]), duration.new_zeros(1)))
        # Domain clock substitution for the three inherited physical-geometry channels.
        geometry = torch.stack((duration.log1p(), prev.log1p(), nxt.log1p()), -1)
        geometry = geometry / math.log1p(t + 1)
        sequence = torch.cat((context, geometry[:, None, :].expand(t, bn, 3)), -1)
        forward, _ = self.forward_gru(sequence)
        reverse, _ = self.reverse_gru(sequence.flip(0))
        # Both terminal states encode only the complete observed history.
        # This is the external forecast-head adaptation, not a future graph query.
        result = self.regression_head(torch.cat((context[-1], forward[-1], reverse[-1]), -1))
        return result.reshape(observed.x.shape[0], observed.x.shape[2])

    def interface_contract(self):
        return dict(family="TreeLoom_generic_fixed_entity_forecast_v1",
                    hidden_dim=32, shared_graph_passes=2, graph_field_rounds_per_pass=2,
                    persistent_identity="caller-fixed observed columns; no numeric ID feature",
                    own_state_rank=8, clade_flags="absent; no clade metadata invented",
                    memory_rank=8 if self.memory is not None else 0,
                    memory_parameters=513 if self.memory is not None else 0,
                    memory_clock="weeks", memory_length_initial=5., memory_length_bounds=[.1, 100.],
                    recurrent="one layer per direction, complete known history",
                    output="one standardized scalar per entity, no clipping",
                    source_tree_class_reused=False, pretrained_ancestry_weights_reused=False)


def matched_pair_v1(seed: int):
    """Fresh common initialization; NoMem differs only by its absent 513-weight module."""
    if type(seed) is not int:
        raise TypeError("Explicit integer seed")
    torch.manual_seed(seed)
    full = TreeLoomGenericForecastV1(with_memory=True)
    no_memory = TreeLoomGenericForecastV1(with_memory=False)
    shared = {key: value for key, value in full.state_dict().items() if not key.startswith("memory.")}
    no_memory.load_state_dict(shared, strict=True)
    return {"full": full, "no_memory": no_memory}

"""Equation-based GraphSSM adaptation for observed marginal-tree intervals.

Reference: Li et al., NeurIPS 2024, arXiv:2406.00943v2, Eqs. (9), (10),
(21), (43), (44), Algorithm 1. Official MIT code was inspected at commit
8431e061b7e740b930b36e1ec3e07ed2ea1f57ea. This is an independent paper-based
implementation, not an exact reproduction. See docs/graphssm_adaptation.md.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import GraphSequenceInput, query_node_index, validate_model_input


OFFICIAL_GRAPHSSM_COMMIT = "8431e061b7e740b930b36e1ec3e07ed2ea1f57ea"


def align_node_memory(previous_ids: Tensor, current_ids: Tensor,
                      previous_memory: Tensor) -> tuple[Tensor, Tensor]:
    """Equation (21): retain persistent leaves, initialize new nodes to zero.

    Stable identities are nonnegative and unique within each snapshot. Every
    -1 denotes a graph-local anonymous internal vertex, so anonymous vertices
    never match one another. Returning a match mask also supports bypassing
    the representation mixer for newly appearing vertices, per Appendix B.2.
    Removed leaves are dropped. A leaf absent in the immediately previous graph
    receives a fresh state on reappearance, rather than resurrecting old memory.
    """
    if previous_ids.ndim != 1 or current_ids.ndim != 1:
        raise ValueError("node IDs must be one-dimensional")
    if previous_memory.shape[0] != previous_ids.numel():
        raise ValueError("previous_memory rows must match previous_ids")
    if previous_ids.dtype != torch.long or current_ids.dtype != torch.long:
        raise ValueError("node IDs must be torch.long")
    if previous_ids.device != current_ids.device or previous_memory.device != current_ids.device:
        raise ValueError("IDs and memory must share a device")
    aligned = previous_memory.new_zeros((current_ids.numel(), *previous_memory.shape[1:]))
    matched = torch.zeros(current_ids.numel(), dtype=torch.bool, device=current_ids.device)
    old_nodes = (previous_ids >= 0).nonzero(as_tuple=False).flatten()
    new_nodes = (current_ids >= 0).nonzero(as_tuple=False).flatten()
    if old_nodes.numel() == 0 or new_nodes.numel() == 0:
        return aligned, matched
    order = torch.argsort(previous_ids[old_nodes])
    old_nodes = old_nodes[order]
    old_ids = previous_ids[old_nodes]
    locations = torch.searchsorted(old_ids, current_ids[new_nodes])
    safe = locations.clamp_max(old_ids.numel() - 1)
    exists = (locations < old_ids.numel()) & (old_ids[safe] == current_ids[new_nodes])
    new_nodes, old_nodes = new_nodes[exists], old_nodes[safe[exists]]
    aligned.index_copy_(0, new_nodes, previous_memory[old_nodes])
    matched[new_nodes] = True
    return aligned, matched


class NativeGraphSAGEDiffusion(nn.Module):
    """Root-plus-incoming-neighbor mean affine map, GraphSAGE-style diffusion.

    This matches the form of default unnormalized mean SAGEConv without needing
    torch_geometric. No activation or normalization is hidden in this operator.
    Native edges are within-tree edges from the observed tensor contract.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.root = nn.Linear(input_dim, output_dim, bias=False)
        self.neighbor = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        aggregate = torch.zeros_like(x)
        degree = x.new_zeros(x.shape[0])
        if edge_index.numel():
            source, target = edge_index
            aggregate.index_add_(0, target, x[source])
            degree.index_add_(0, target, x.new_ones(target.numel()))
        aggregate = aggregate / degree.clamp_min(1)[:, None]
        return self.root(x) + self.neighbor(aggregate)


class RepresentationInterpolation(nn.Module):
    """Scaled interpolation from paper Eqs. (43), (44), previous then current."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Linear(2 * hidden_dim, hidden_dim)
        self.scale = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, previous: Tensor, current: Tensor, matched: Tensor) -> Tensor:
        pair = torch.cat((previous, current), dim=-1)
        xi = torch.sigmoid(self.weight(pair))
        rho = F.softplus(self.scale(pair))
        mixed = rho * (xi * previous + (1 - xi) * current)
        # New and anonymous vertices have no historical representation to mix.
        return torch.where(matched[:, None], mixed, current)


class DiagonalSISOStateSpace(nn.Module):
    """Per-channel real diagonal state, paper Algorithm 1 and Eq. (9).

    forward returns (raw updated U [V,D,S], projected Y [V,D]). Carry U, not Y,
    into the next interval. The default injection is Delta*B*x from Eq. (9).
    `discretization="zoh"` uses the exact single-hold factor from Eq. (8):
    (exp(Delta*A)-I)/A, evaluated with expm1 for narrow intervals.
    """

    def __init__(self, hidden_dim: int, state_dim: int = 16,
                 discretization: str = "paper"):
        super().__init__()
        if hidden_dim < 1 or state_dim < 1:
            raise ValueError("hidden_dim and state_dim must be positive")
        if discretization not in {"paper", "zoh"}:
            raise ValueError("discretization must be 'paper' or 'zoh'")
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.discretization = discretization
        # Paper/official HiPPO-real convention n=1,...,S: A_n=-(n+1).
        initial = torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32) + 1)
        self.log_nA = nn.Parameter(initial.repeat(hidden_dim, 1))
        self.B = nn.Parameter(torch.ones(hidden_dim, state_dim))
        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim))
        nn.init.xavier_uniform_(self.C)

    def transition(self, delta: Tensor | float, like: Tensor) -> tuple[Tensor, Tensor]:
        """Decay and input-integral factors [V,D,S] with finite negative poles."""
        dt = torch.as_tensor(delta, device=like.device, dtype=like.dtype)
        if dt.ndim == 0:
            dt = dt.expand(like.shape[0])
        if dt.shape != (like.shape[0],):
            raise ValueError("delta must be scalar or [V]")
        if not bool(torch.isfinite(dt).all()) or bool((dt < 0).any()):
            raise ValueError("delta must be finite and nonnegative")
        # Floating-point guards only: negative rates stay representable even
        # for extreme optimizer values. Ordinary parameter values are unchanged.
        limits = torch.finfo(like.dtype)
        log_rate = self.log_nA.to(like.dtype).clamp(
            min=math.log(limits.tiny), max=math.log(limits.max) - 2
        )
        rate = torch.exp(log_rate)
        # Form Delta*rate in log space so a stiff but representable pole does
        # not overflow before its decay has correctly rounded to zero.
        positive_dt = torch.where(dt == 0, torch.ones_like(dt), dt)
        log_product = positive_dt.log()[:, None, None] + log_rate[None]
        z = -torch.exp(log_product.clamp(max=math.log(limits.max) - 2))
        z = torch.where(dt[:, None, None] == 0, torch.zeros_like(z), z)
        decay = torch.exp(z)
        if self.discretization == "paper":
            integral = dt[:, None, None].expand_as(decay)
        else:
            # expm1 avoids cancellation. Near zero use the exprel series,
            # also avoiding an ill-conditioned derivative when rates underflow.
            near_zero = z.abs() < 1e-4
            safe_rate = torch.where(near_zero, torch.ones_like(z), rate[None, :, :])
            direct = -torch.expm1(z) / safe_rate
            small_z = torch.where(near_zero, z, torch.zeros_like(z))
            series = dt[:, None, None] * (
                1 + small_z / 2 + small_z.square() / 6 + small_z.pow(3) / 24
            )
            integral = torch.where(near_zero, series, direct)
        return decay, integral

    def forward(self, x: Tensor, memory: Tensor | None,
                delta: Tensor | float) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[1] != self.hidden_dim:
            raise ValueError("x must be [V,hidden_dim]")
        shape = (x.shape[0], self.hidden_dim, self.state_dim)
        if memory is None:
            memory = x.new_zeros(shape)
        elif memory.shape != shape:
            raise ValueError("memory must be [V,hidden_dim,state_dim]")
        decay, integral = self.transition(delta, x)
        updated = decay * memory + integral * self.B[None, :, :] * x[:, :, None]
        projected = torch.einsum("vds,ds->vd", updated, self.C)
        return updated, projected


class GraphSSMBlock(nn.Module):
    """Diffuse, optionally mix, align memory, scan, activate and add residual."""

    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int,
                 mix_representations: bool, discretization: str, dropout: float):
        super().__init__()
        self.diffusion = NativeGraphSAGEDiffusion(input_dim, hidden_dim)
        self.residual = nn.Linear(input_dim, hidden_dim)
        self.mixer = RepresentationInterpolation(hidden_dim) if mix_representations else None
        self.cell = DiagonalSISOStateSpace(hidden_dim, state_dim, discretization)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, seq: GraphSequenceInput, reverse: bool = False,
                return_trace: bool = False):
        z = self.diffusion(h, seq.edge_index)
        residual = self.residual(h)
        order = torch.argsort(seq.intervals[:, 0], descending=reverse)
        previous_ids = previous_memory = previous_z = None
        node_groups, outputs, trace = [], [], []
        for graph_tensor in order:
            graph = int(graph_tensor)
            nodes = (seq.node_graph == graph).nonzero(as_tuple=False).flatten()
            ids = seq.node_ids[nodes]
            current = z[nodes]
            if previous_memory is None:
                incoming = current.new_zeros((nodes.numel(), self.cell.hidden_dim,
                                               self.cell.state_dim))
                matched = torch.zeros(nodes.numel(), dtype=torch.bool, device=nodes.device)
                mixed = current
            else:
                incoming, matched = align_node_memory(previous_ids, ids, previous_memory)
                if self.mixer is not None:
                    aligned_z, _ = align_node_memory(previous_ids, ids, previous_z)
                    mixed = self.mixer(aligned_z, current, matched)
                else:
                    mixed = current
            delta = (seq.intervals[graph, 1] - seq.intervals[graph, 0]).to(current.dtype)
            memory, output = self.cell(mixed, incoming, delta)
            result = self.dropout(F.relu(output)) + residual[nodes]
            node_groups.append(nodes)
            outputs.append(result)
            if return_trace:
                trace.append({"graph": graph, "nodes": nodes, "matched": matched,
                              "incoming_memory": incoming, "updated_memory": memory,
                              "driving_representation": mixed, "delta": delta})
            previous_ids, previous_memory, previous_z = ids, memory, current
        packed_nodes = torch.cat(node_groups)
        packed_output = torch.cat(outputs)
        result = packed_output[torch.argsort(packed_nodes)]
        return (result, trace) if return_trace else result


class GraphSSMAdaptation(nn.Module):
    """GraphSSM-S4-style blocks adapted to full observed genomic intervals.

    Default: forward genomic scan, representation mixing in only the first
    block, supplied interval widths, paper Eq. (9) update and Eq. (10) residual.
    Anonymous internal-node memory resets each snapshot, persistent-leaf memory
    follows Appendix B.2. Readout repeats each containing-tree representation
    at the fixed scoring positions. These are snapshot-end representations,
    not a continuous-time decoder at fractional positions within an interval.

    Optional bidirectional=True uses the same blocks for a reverse scan and
    concatenates both outputs for noncausal offline inference. This is an
    explicitly documented task adaptation, not the published architecture.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, state_dim: int = 16,
                 discretization: str = "paper", bidirectional: bool = False,
                 dropout: float = 0.0, validate: bool = True):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, state_dim) < 1 or num_classes < 2:
            raise ValueError("positive dimensions/layers and num_classes >= 2 are required")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.validate_inputs = validate
        self.discretization = discretization
        self.blocks = nn.ModuleList([
            GraphSSMBlock(input_dim if layer == 0 else hidden_dim, hidden_dim,
                          state_dim, layer == 0, discretization, dropout)
            for layer in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def encode_nodes(self, seq: GraphSequenceInput, reverse: bool = False) -> Tensor:
        h = seq.x
        for block in self.blocks:
            h = block(h, seq, reverse=reverse)
        return h

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        if self.validate_inputs:
            validate_model_input(seq)
        h = self.encode_nodes(seq)
        if self.bidirectional:
            h = torch.cat((h, self.encode_nodes(seq, reverse=True)), dim=-1)
        nodes = query_node_index(seq)[seq.target_graph]
        return self.head(h[nodes])

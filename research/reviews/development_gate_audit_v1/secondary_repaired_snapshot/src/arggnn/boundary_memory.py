"""Experimental invariant-boundary GNN with exact coupled query memory.

Observed-only independent source banks condition a boundary diffusion model.
This is not a projection of full-tree dynamics or a strong-lumpability claim.
See docs/boundary_memory.md for the fixed interface and comparison controls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .kron_interface import LeafInterfaceBank


def _positive_raw(initial: float, name: str) -> float:
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return initial + math.log(-math.expm1(-initial))


def _decay(eigenvalues: Tensor, rates: Tensor, damping: Tensor) -> Tensor:
    if rates.ndim != 1 or damping.shape != rates.shape:
        raise ValueError("Channel coupling and damping must be matching vectors")
    if not bool(torch.isfinite(rates).all()) or not bool(torch.isfinite(damping).all()) or bool((rates <= 0).any()) or bool((damping <= 0).any()):
        raise ValueError("Channel coupling and damping must be finite and positive")
    decay = eigenvalues[..., :, None]*rates + damping
    if not bool(torch.isfinite(decay).all()) or bool((decay <= 0).any()):
        raise ValueError("Cached boundary dynamics are not numerically dissipative")
    return decay


def _factors(decay: Tensor, duration: Tensor | float) -> tuple[Tensor, Tensor, Tensor]:
    dt = torch.as_tensor(duration, dtype=decay.dtype, device=decay.device)
    if not bool(torch.isfinite(dt).all()) or bool((dt < 0).any()):
        raise ValueError("Flow duration must be finite and nonnegative")
    exponent = -dt[..., None, None]*decay
    return exponent.exp(), -torch.expm1(exponent)/decay, dt


def spectral_affine_action(eigenvalues: Tensor, eigenvectors: Tensor, state: Tensor,
        forcing: Tensor, rates: Tensor, damping: Tensor, duration: Tensor | float) -> Tensor:
    """Exact constant-forcing action for independent channels coupled across Q.

    Cached Cqq=U diag(c) U.T gives decay d_qh=lambda_h*c_q+delta_h.
    Returns U[exp(-dt*d)*U.T state + (-expm1(-dt*d)/d)*U.T forcing].
    No augmented matrix exponential or learned-matrix eigendecomposition occurs.
    """
    if state.shape != forcing.shape or state.ndim < 2 or eigenvalues.shape != state.shape[:-1] or eigenvectors.shape != (*state.shape[:-2], state.shape[-2], state.shape[-2]):
        raise ValueError("Require matched [...,Q,H] states/forcing and cached [...,Q] eigensystem")
    decay = _decay(eigenvalues, rates, damping)
    factor, integral, dt = _factors(decay, duration)
    transpose = eigenvectors.transpose(-1, -2)
    result = eigenvectors @ (factor*(transpose@state) + integral*(transpose@forcing))
    # A zero-time action is exactly identity, including finite-precision bases.
    return torch.where(dt[..., None, None] == 0, state, result)


def diagonal_affine_action(diagonal: Tensor, state: Tensor, forcing: Tensor,
        rates: Tensor, damping: Tensor, duration: Tensor | float) -> Tensor:
    """Exact no-cross-query action with the same learned forcing."""
    if diagonal.shape != state.shape[:-1] or forcing.shape != state.shape:
        raise ValueError("Require [...,Q] diagonal and matching [...,Q,H] states/forcing")
    factor, integral, dt = _factors(_decay(diagonal, rates, damping), duration)
    result = factor*state + integral*forcing
    return torch.where(dt[..., None, None] == 0, state, result)


def spectral_equilibrium(eigenvalues: Tensor, eigenvectors: Tensor, forcing: Tensor,
        rates: Tensor, damping: Tensor) -> Tensor:
    """Full static K^-1 F using cached Cqq eigenpairs, K=lambda*Cqq+delta*I."""
    if eigenvalues.shape != forcing.shape[:-1] or eigenvectors.shape != (*forcing.shape[:-2], forcing.shape[-2], forcing.shape[-2]):
        raise ValueError("Require [...,Q,H] forcing and matched cached eigensystem")
    decay = _decay(eigenvalues, rates, damping)
    result = eigenvectors@((eigenvectors.transpose(-1, -2)@forcing)/decay)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Static boundary equilibrium overflow")
    return result


def query_grounding_leakage(cqq: Tensor) -> Tensor:
    """Nonnegative reference-grounding row sums, with explicit roundoff policy.

    Reject negative rows beyond 4*Q*eps*sum(abs(row)). Only negatives within
    that floating-point bound are set to zero. No positive floor is added.
    """
    if cqq.ndim < 2 or cqq.shape[-1] != cqq.shape[-2] or not bool(torch.isfinite(cqq).all()):
        raise ValueError("Require finite [...,Q,Q] query response")
    leakage = cqq.sum(-1)
    bound = 4*cqq.shape[-1]*torch.finfo(cqq.dtype).eps*cqq.abs().sum(-1)
    if bool((leakage < -bound).any()):
        raise ValueError("Query grounding leakage has materially negative row sums")
    return torch.where(leakage < 0, torch.zeros_like(leakage), leakage)


def gain_matched_diagonal_forcing(diagonal: Tensor, eigenvalues: Tensor,
        eigenvectors: Tensor, forcing: Tensor, rates: Tensor, damping: Tensor) -> Tensor:
    """D e for independent dynamics, e=K^-1 F from full cached response.

    D=lambda*diagonal+delta. Pass diag(Cqq) for gain_matched_diagonal
    or query grounding row sums for gain_leak_diagonal.

    This retains instantaneous source mixing and full static equilibrium while
    eliminating off-diagonal initial-state memory transfer. Sources stay fixed.
    """
    if diagonal.shape != forcing.shape[:-1]:
        raise ValueError("Require [...,Q] diagonal and [...,Q,H] forcing")
    equilibrium = spectral_equilibrium(eigenvalues, eigenvectors, forcing, rates, damping)
    result = _decay(diagonal, rates, damping)*equilibrium
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Gain-matched diagonal forcing overflow")
    return result


class _BoundaryMessageLayer(nn.Module):
    """Weighted boundary-neighbor mean and residual node update."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, adjacency: Tensor) -> Tensor:
        degree = adjacency.sum(-1)
        denominator = torch.where(degree > 0, degree, torch.ones_like(degree))
        neighbor = (adjacency@h)/denominator[..., None]
        update = F.gelu(self.self_linear(h)+self.neighbor_linear(neighbor))
        return self.norm(h+update)


class BoundaryResponseGNN(nn.Module):
    """Graph-local encoder of invariant leaves and normalized Kron conductances.

    Off-diagonal conductance is -C_ij. Each graph is encoded independently.
    Neither genomic positions, interval widths, IDs nor temporal states enter.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers) < 1:
            raise ValueError("Positive input/hidden dimensions and layer count are required")
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(_BoundaryMessageLayer(hidden_dim) for _ in range(num_layers))

    def forward(self, features: Tensor, response: Tensor) -> Tensor:
        adjacency = -response + torch.diag_embed(response.diagonal(dim1=-2, dim2=-1))
        h = F.gelu(self.input_linear(features))
        for layer in self.layers:
            h = layer(h, adjacency)
        return h


@dataclass(frozen=True)
class BoundarySourceBank:
    local: Tensor                # [G,Q,node_hidden]
    query_signal: Tensor         # [G,Q,latent], bounded independent source
    reference_signal: Tensor     # [G,R,latent], bounded independent source
    forcing: Tensor              # [G,Q,latent], query_signal-lambda*Cqr*reference


@dataclass(frozen=True)
class BoundaryMemoryContext:
    local: Tensor                # [P,Q,node_hidden]
    forward: Tensor              # [P,Q,latent], or static equilibrium
    backward: Tensor | None      # [P,Q,latent], positive reversed flow


def _apply_prepared_coefficients(state: Tensor, vectors: Tensor | None,
        factor: Tensor, offset: Tensor) -> Tensor:
    """No tensor-to-Python validation inside the ordered recurrence."""
    spectral = state if vectors is None else vectors.transpose(-1, -2)@state
    updated = torch.addcmul(offset, factor, spectral)
    return updated if vectors is None else vectors@updated


@dataclass(frozen=True)
class _PreparedHoldBank:
    decay: Tensor
    vectors: Tensor | None
    projected_forcing: Tensor
    full_factor: Tensor
    full_offset: Tensor
    zero_full_duration: Tensor

    @classmethod
    def prepare(cls, decay: Tensor, vectors: Tensor | None, forcing: Tensor,
                widths: Tensor) -> _PreparedHoldBank:
        projected = forcing if vectors is None else vectors.transpose(-1, -2)@forcing
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("Prepared spectral forcing overflow")
        factor, integral, duration = _factors(decay, widths)
        return cls(decay, vectors, projected, factor, integral*projected, duration==0)

    def full_action(self, graph: int, state: Tensor) -> Tensor:
        vectors = None if self.vectors is None else self.vectors[graph]
        result = _apply_prepared_coefficients(state, vectors, self.full_factor[graph], self.full_offset[graph])
        return torch.where(self.zero_full_duration[graph], state, result)

    def partial_action(self, target: Tensor, state: Tensor, durations: Tensor) -> Tensor:
        factor, integral, duration = _factors(self.decay[target], durations)
        vectors = None if self.vectors is None else self.vectors[target]
        result = _apply_prepared_coefficients(state, vectors, factor, integral*self.projected_forcing[target])
        return torch.where(duration[..., None, None]==0, state, result)


class BoundaryMemoryModel(nn.Module):
    """Cached boundary-response GNN with coupled persistent-query memory.

    ``coupling=diagonal`` removes cross-query memory exchange but keeps forcing.
    ``coupling=gain_matched_diagonal`` independently relaxes to the full static
    equilibrium, retaining instantaneous source mixing while removing temporal
    initial-state exchange. It has the exact same learned parameter schema.
    ``aggregation=homogeneous_bag`` averages fixed Cqq and learned forcing by
    interval width, then uses one homogeneous generator on the same domain.
    ``aggregation=equilibrium`` reads each target graph's static equilibrium.
    Global channel rates make the fixed bag eigensystem mathematically valid.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32,
                 num_layers: int = 2, latent_dim: int = 8,
                 coupling: str = "full", aggregation: str = "ordered",
                 bidirectional: bool = True, symmetric_readout: bool = True,
                 initial_coupling: float = 20., initial_damping: float = 1.,
                 validate: bool = True):
        super().__init__()
        if min(input_dim, num_classes, hidden_dim, num_layers, latent_dim) < 1:
            raise ValueError("Positive dimensions/layers and output count are required")
        if coupling not in {"full", "diagonal", "gain_matched_diagonal", "gain_leak_diagonal"}:
            raise ValueError("coupling must be full, diagonal, gain_matched_diagonal or gain_leak_diagonal")
        if aggregation not in {"ordered", "homogeneous_bag", "equilibrium"}:
            raise ValueError("aggregation must be ordered, homogeneous_bag or equilibrium")
        self.input_dim, self.num_classes = input_dim, num_classes
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.latent_dim, self.coupling, self.aggregation = latent_dim, coupling, aggregation
        self.bidirectional = bool(bidirectional) and aggregation != "equilibrium"
        self.symmetric_readout, self.validate_inputs = bool(symmetric_readout), bool(validate)
        self.initial_coupling, self.initial_damping = initial_coupling, initial_damping
        self.encoder = BoundaryResponseGNN(input_dim, hidden_dim, num_layers)
        self.query_projection = nn.Linear(hidden_dim, latent_dim)
        self.reference_projection = nn.Linear(hidden_dim, latent_dim)
        self.raw_coupling = nn.Parameter(torch.full((latent_dim,), _positive_raw(initial_coupling, "initial_coupling")))
        self.raw_damping = nn.Parameter(torch.full((latent_dim,), _positive_raw(initial_damping, "initial_damping")))
        contexts = 2 if self.bidirectional else 1
        self.head = nn.Linear(hidden_dim+contexts*latent_dim, num_classes)

    @property
    def num_outputs(self) -> int:
        return self.head.out_features

    @property
    def rates(self) -> Tensor:
        return F.softplus(self.raw_coupling)

    @property
    def damping(self) -> Tensor:
        return F.softplus(self.raw_damping)

    def _validate_bank(self, bank: LeafInterfaceBank) -> None:
        cache = bank.query_cache
        if cache is None:
            raise ValueError("Prepare an exhaustive query/reference cache before forward")
        if bank.leaf_x.ndim != 3 or bank.leaf_x.shape[-1] != self.input_dim:
            raise ValueError("Cached invariant leaf feature dimension differs from model")
        weight = self.encoder.input_linear.weight
        if bank.leaf_x.device != weight.device or bank.leaf_x.dtype != weight.dtype:
            raise ValueError("Move/cast the cached bank once to the model device/dtype before forward")
        graphs, boundary, _ = bank.leaf_x.shape
        q, r = cache.query_indices.numel(), cache.reference_indices.numel()
        if cache.normalized_response.shape != (graphs, boundary, boundary) or q+r != boundary or cache.cqq.shape != (graphs, q, q) or cache.cqr.shape != (graphs, q, r):
            raise ValueError("Cached boundary response/partition shape mismatch")
        if cache.eigenvalues.shape != (graphs, q) or cache.eigenvectors.shape != (graphs, q, q) or cache.bag_cqq.shape != (q, q) or cache.bag_eigenvalues.shape != (q,) or cache.bag_eigenvectors.shape != (q, q):
            raise ValueError("Cached query eigensystem shape mismatch")
        if not torch.equal(cache.query_ids, bank.query_node_ids):
            raise ValueError("Cached query order differs from declared external target order")
        for name in ("leaf_x", "intervals", "positions"):
            if not bool(torch.isfinite(getattr(bank, name)).all()):
                raise ValueError(f"Nonfinite cached {name}")
        if bank.intervals.shape != (graphs, 2) or bank.target_graph.shape != bank.positions.shape:
            raise ValueError("Cached graph/coordinate shape mismatch")
        order = torch.argsort(bank.intervals[:, 0])
        intervals = bank.intervals[order]
        if bool((intervals[:, 1] <= intervals[:, 0]).any()) or (graphs > 1 and not torch.allclose(intervals[:-1, 1], intervals[1:, 0], rtol=0, atol=1e-10)):
            raise ValueError("Cached intervals must form a positive contiguous partition")
        if bool((bank.target_graph < 0).any()) or bool((bank.target_graph >= graphs).any()):
            raise ValueError("Cached target_graph out of range")
        selected = bank.intervals[bank.target_graph]
        if not bool(((bank.positions >= selected[:, 0]) & (bank.positions < selected[:, 1])).all()):
            raise ValueError("Cached target graph does not contain fixed coordinate")
        for value in (cache.normalized_response, cache.cqq, cache.cqr, cache.eigenvalues, cache.eigenvectors,
                      cache.bag_cqq, cache.bag_eigenvalues, cache.bag_eigenvectors):
            if value.device != weight.device or value.dtype != weight.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError("Cached response/eigensystem must be finite on model device/dtype")

    def source_bank(self, bank: LeafInterfaceBank) -> BoundarySourceBank:
        if self.validate_inputs:
            self._validate_bank(bank)
        elif bank.query_cache is None:
            raise ValueError("A precomputed query/reference cache is required")
        cache = bank.query_cache
        if cache.normalization != "mean_boundary_diagonal":
            raise ValueError("Boundary memory requires full-boundary mean-diagonal normalization")
        h = self.encoder(bank.leaf_x, cache.normalized_response)
        local = h[:, cache.query_indices]
        query = torch.tanh(self.query_projection(local))
        reference = torch.tanh(self.reference_projection(h[:, cache.reference_indices]))
        forcing = query - (cache.cqr@reference)*self.rates
        if not bool(torch.isfinite(forcing).all()):
            raise ValueError("Boundary forcing overflow")
        return BoundarySourceBank(local, query, reference, forcing)

    def _action(self, eigenvalues: Tensor, eigenvectors: Tensor | None, state: Tensor,
                forcing: Tensor, duration: Tensor | float) -> Tensor:
        if eigenvectors is None:
            return diagonal_affine_action(eigenvalues, state, forcing, self.rates, self.damping, duration)
        return spectral_affine_action(eigenvalues, eigenvectors, state, forcing, self.rates, self.damping, duration)

    def encode_context(self, bank: LeafInterfaceBank) -> BoundaryMemoryContext:
        sources = self.source_bank(bank)
        cache = bank.query_cache
        local = sources.local[bank.target_graph]
        zero = sources.forcing.new_zeros(sources.forcing.shape[1:])
        rates, damping = self.rates, self.damping
        matched_gain = self.coupling in {"gain_matched_diagonal", "gain_leak_diagonal"}
        if self.coupling == "full":
            values, vectors = cache.eigenvalues, cache.eigenvectors
            bag_values, bag_vectors = cache.bag_eigenvalues, cache.bag_eigenvectors
        else:
            values = query_grounding_leakage(cache.cqq) if self.coupling == "gain_leak_diagonal" else cache.cqq.diagonal(dim1=-2, dim2=-1)
            vectors = None
            bag_values = query_grounding_leakage(cache.bag_cqq) if self.coupling == "gain_leak_diagonal" else cache.bag_cqq.diagonal()
            bag_vectors = None
        if self.aggregation == "equilibrium":
            forcing = sources.forcing[bank.target_graph]
            if self.coupling == "full" or matched_gain:
                equilibrium = spectral_equilibrium(cache.eigenvalues[bank.target_graph],
                    cache.eigenvectors[bank.target_graph], forcing, rates, damping)
            else:
                equilibrium = forcing/_decay(values[bank.target_graph], rates, damping)
            return BoundaryMemoryContext(local, equilibrium, None)
        if self.aggregation == "homogeneous_bag":
            forcing = torch.einsum("g,gqh->qh", cache.interval_weights.to(sources.forcing.dtype), sources.forcing)
            decay = _decay(bag_values, rates, damping)
            if matched_gain:
                # Match the full averaged generator, not an average of local gains.
                forcing = decay*spectral_equilibrium(cache.bag_eigenvalues,
                    cache.bag_eigenvectors, forcing, rates, damping)
            projected = forcing if bag_vectors is None else bag_vectors.T@forcing
            if not bool(torch.isfinite(projected).all()):
                raise ValueError("Homogeneous spectral forcing overflow")
            p = bank.positions.numel()
            state = zero[None].expand(p, -1, -1)
            basis = None if bag_vectors is None else bag_vectors[None].expand(p, -1, -1)
            # Differences are formed before conversion to memory dtype.
            left, right = bank.intervals[:, 0].min(), bank.intervals[:, 1].max()
            def homogeneous(duration: Tensor) -> Tensor:
                factor, integral, dt = _factors(decay[None].expand(p, -1, -1), duration)
                result = _apply_prepared_coefficients(state, basis, factor, integral*projected)
                return torch.where(dt[..., None, None]==0, state, result)
            forward = homogeneous(bank.positions-left)
            backward = homogeneous(right-bank.positions) if self.bidirectional else None
            return BoundaryMemoryContext(local, forward, backward)
        decay = _decay(values, rates, damping)
        forcing = sources.forcing
        if matched_gain:
            forcing = decay*spectral_equilibrium(cache.eigenvalues, cache.eigenvectors,
                                                 forcing, rates, damping)
        widths = bank.intervals[:, 1]-bank.intervals[:, 0]
        holds = _PreparedHoldBank.prepare(decay, vectors, forcing, widths)
        order = torch.argsort(bank.intervals[:, 0])
        # One metadata transfer, rather than implicit per-hold CUDA scalar indexing.
        graph_order = order.detach().cpu().tolist()
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        rank = inverse[bank.target_graph]
        before_left = []
        state = zero
        for index, graph in enumerate(graph_order):
            before_left.append(state)
            if index+1 < len(graph_order):
                state = holds.full_action(graph, state)
        selected = bank.intervals[bank.target_graph]
        forward = holds.partial_action(bank.target_graph, torch.stack(before_left)[rank],
                                       bank.positions-selected[:, 0])
        backward = None
        if self.bidirectional:
            before_right = [zero]*len(graph_order)
            state = zero
            for index in range(len(graph_order)-1, -1, -1):
                before_right[index] = state
                if index > 0:
                    state = holds.full_action(graph_order[index], state)
            backward = holds.partial_action(bank.target_graph, torch.stack(before_right)[rank],
                                            selected[:, 1]-bank.positions)
        return BoundaryMemoryContext(local, forward, backward)

    def forward(self, bank: LeafInterfaceBank) -> Tensor:
        context = self.encode_context(bank)
        if context.backward is None:
            features = torch.cat((context.local, context.forward), dim=-1)
        elif self.symmetric_readout:
            features = torch.cat((context.local, context.forward+context.backward,
                                  (context.forward-context.backward).square()), dim=-1)
        else:
            features = torch.cat((context.local, context.forward, context.backward), dim=-1)
        return self.head(features)

    def configuration_manifest(self, bank: LeafInterfaceBank | None = None) -> dict[str, Any]:
        result = {"schema_version": 1, "kind": "experimental_coupled_boundary_query_memory",
            "scientific_status": "candidate, no novelty/performance/full-tree-projection claim",
            "input_dim": self.input_dim, "num_outputs": self.num_outputs,
            "hidden_dim": self.hidden_dim, "num_layers": self.num_layers, "latent_dim": self.latent_dim,
            "coupling": self.coupling, "aggregation": self.aggregation, "bidirectional": self.bidirectional,
            "initial_state_cross_query_exchange": self.coupling == "full",
            "forcing_transform": "D*K^-1F, cached full equilibrium; D=lambda*diag(Cqq)+delta" if self.coupling == "gain_matched_diagonal" else ("D*K^-1F, cached full equilibrium; D=lambda*rowsum(Cqq)+delta" if self.coupling == "gain_leak_diagonal" else "original learned query/reference forcing"),
            "leakage_roundoff_policy": "reject rows < -4*Q*eps*sum(abs(row)); set only accepted negative rows to0" if self.coupling == "gain_leak_diagonal" else None,
            "symmetric_readout": self.symmetric_readout, "initial_coupling": self.initial_coupling,
            "initial_damping": self.initial_damping, "channel_rates": "global learned softplus vectors",
            "source_bank": "independent boundary-response GNN; bounded learned query/reference signals",
            "normalization": "full-boundary mean diagonal per graph; exact zero response uses scale1",
            "discretization": "cached eigensystem exp/expm1 constant-forcing action",
            "online_eigensystem_or_augmented_matrix_exp": False,
            "scan_preparation": "batched decay/full factors/forcing projections; no per-hold bool validation",
            "sorted_geometry_order_transfer": "one CPU integer metadata transfer per forward",
            "boundary_states": "zero at supplied tile endpoints",
            "backward_scan": "positive-duration reverse flow, never inverse flow",
            "learned_numerical_id_embeddings": False, "raw_parent_branch_feature": "excluded by primitive",
            "parameters": sum(p.numel() for p in self.parameters()),
            "learned_parameter_dtype": str(self.raw_coupling.dtype), "torch": str(torch.__version__)}
        if bank is not None:
            cache = bank.query_cache
            result.update(feature_columns=list(bank.feature_columns), query_ids=cache.query_ids.tolist(),
                          reference_ids=cache.reference_ids.tolist(), time_scale=bank.time_scale,
                          resistance_scale=bank.resistance_scale,
                          normalization_scales=cache.normalization_scale.tolist(),
                          cached_model_dtype=str(bank.leaf_x.dtype), coordinate_dtype=str(bank.positions.dtype),
                          inactive_reference_parameters_if_no_references=(self.hidden_dim+1)*self.latent_dim if cache.reference_ids.numel()==0 else 0)
        canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        result["configuration_sha256"] = hashlib.sha256(canonical).hexdigest()
        result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["interface_dependency_sha256"] = hashlib.sha256(Path(__file__).with_name("kron_interface.py").read_bytes()).hexdigest()
        return result

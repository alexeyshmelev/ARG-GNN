"""Experimental exact affine-ZOH transport over independent graph banks.

Established semigroup mathematics, not a novelty claim. Full matrices test
cross-state mixing. Diagonal affine dynamics already preserve ordering.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import hashlib
import json
from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .models import SharedGraphEncoder, GraphSequenceInput, query_node_index, validate_model_input


def augmented_affine_generator(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or b.shape != a.shape[:-1]:
        raise ValueError('Require A[...,d,d] and b[...,d]')
    upper = torch.cat((a, b.unsqueeze(-1)), dim=-1)
    lower = a.new_zeros(*a.shape[:-2], 1, a.shape[-1] + 1)
    return torch.cat((upper, lower), dim=-2)


def affine_transition(generator: Tensor, duration: Tensor | float) -> Tensor:
    """Exact constant-generator flow. Duration broadcasts over batch axes."""
    if generator.ndim < 2 or generator.shape[-1] != generator.shape[-2]:
        raise ValueError('generator must be a batch of square affine matrices')
    dt = torch.as_tensor(duration, dtype=generator.dtype, device=generator.device)
    if not bool(torch.isfinite(dt).all()) or bool((dt < 0).any()):
        raise ValueError('Flow duration must be finite and nonnegative')
    while dt.ndim < generator.ndim - 2:
        dt = dt.unsqueeze(-1)
    return torch.matrix_exp(generator * dt[..., None, None])


def apply_affine_transition(transition: Tensor, state: Tensor) -> Tensor:
    """Use the upper affine block rather than repeatedly accumulating a 1."""
    d = transition.shape[-1] - 1
    if state.shape[-1] != d:
        raise ValueError('Affine transition and state dimensions differ')
    return (transition[..., :d, :d] @ state.unsqueeze(-1)).squeeze(-1) + transition[..., :d, d]


@dataclass
class GeneratorBank:
    local: Tensor       # [G,Q,H], original graph serialization order
    a: Tensor           # [G,Q,d,d]
    b: Tensor           # [G,Q,d]
    affine: Tensor      # [G,Q,d+1,d+1]


@dataclass
class SemigroupContext:
    local: Tensor       # [P,Q,H]
    forward: Tensor     # [P,Q,d]
    backward: Tensor | None  # [P,Q,d], positive-duration reversed scan


class DissipativeGenerator(nn.Module):
    """Sym(A) is strictly negative. The forced augmented matrix need not be."""
    def __init__(self, hidden_dim: int, latent_dim: int = 8, rank: int = 2,
                 matrix_kind: str = 'full', minimum_damping: float = 1e-3,
                 initial_rate: float = 20.0):
        super().__init__()
        if min(hidden_dim, latent_dim, rank) < 1 or rank > latent_dim:
            raise ValueError('Positive dimensions and 1 <= rank <= latent_dim required')
        if matrix_kind not in {'full', 'diagonal'}:
            raise ValueError('matrix_kind must be full or diagonal')
        if not math.isfinite(minimum_damping) or minimum_damping <= 0:
            raise ValueError('minimum_damping must be finite and positive')
        if not math.isfinite(initial_rate) or initial_rate <= 1e-4:
            raise ValueError('initial_rate must be finite and greater than 1e-4')
        self.latent_dim, self.rank = latent_dim, rank
        self.matrix_kind = matrix_kind
        self.minimum_damping = minimum_damping
        self.initial_rate = float(initial_rate)
        self.damping = nn.Linear(hidden_dim, latent_dim)
        self.forcing = nn.Linear(hidden_dim, latent_dim)
        if matrix_kind == 'full':
            self.u = nn.Linear(hidden_dim, latent_dim * rank)
            self.v = nn.Linear(hidden_dim, latent_dim * rank)
            self.w = nn.Linear(hidden_dim, latent_dim * rank)
            for layer in (self.u, self.v, self.w):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        else:
            self.u = self.v = self.w = None
        adjusted = initial_rate - 1e-4
        raw = adjusted + math.log(-math.expm1(-adjusted))
        self.raw_rate = nn.Parameter(torch.tensor(raw))

    @property
    def rate(self) -> Tensor:
        return F.softplus(self.raw_rate) + 1e-4

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        damping = F.softplus(self.damping(h)) + self.minimum_damping
        a = -torch.diag_embed(damping)
        if self.matrix_kind == 'full':
            shape = (*h.shape[:-1], self.latent_dim, self.rank)
            u = torch.tanh(self.u(h).reshape(shape)) / math.sqrt(self.rank)
            v = torch.tanh(self.v(h).reshape(shape)) / math.sqrt(self.rank)
            w = torch.tanh(self.w(h).reshape(shape)) / math.sqrt(self.rank)
            a = a - u @ u.transpose(-1, -2) + v @ w.transpose(-1, -2) - w @ v.transpose(-1, -2)
        b = torch.tanh(self.forcing(h))
        return self.rate * a, self.rate * b

    def inactive_parameter_count(self) -> int:
        # The diagonal schema contains no unused cross-state projection heads.
        return 0


class SemigroupGraphModel(nn.Module):
    """Exact piecewise-constant affine flow from local query-leaf graph banks.

    `aggregation='ordered'` scans in physical coordinate order. `span_set`
    exponentiates the span-weighted generator sum and discards arrangement.
    Source generators never depend on record count, midpoint or previous state.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, latent_dim: int = 8, generator_rank: int = 2,
                 matrix_kind: str = 'full', aggregation: str = 'ordered',
                 bidirectional: bool = True, symmetric_readout: bool = True,
                 initial_rate: float = 20.0, minimum_damping: float = 1e-3,
                 validate: bool = True):
        super().__init__()
        if num_classes < 1:
            raise ValueError('num_classes must be positive. One output supports scalar regression')
        if aggregation not in {'ordered', 'span_set', 'homogeneous_set'}:
            raise ValueError('aggregation must be ordered, span_set, or homogeneous_set')
        self.input_dim, self.num_outputs = input_dim, num_classes
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.encoder = SharedGraphEncoder(input_dim, hidden_dim, num_layers)
        self.generator = DissipativeGenerator(hidden_dim, latent_dim, generator_rank,
            matrix_kind, minimum_damping, initial_rate)
        self.latent_dim = latent_dim
        self.aggregation = aggregation
        self.bidirectional = bool(bidirectional)
        self.symmetric_readout = bool(symmetric_readout)
        self.validate_inputs = bool(validate)
        self.head = nn.Linear(hidden_dim + latent_dim * (2 if bidirectional else 1), num_classes)

    def inactive_parameter_count(self) -> int:
        count = self.generator.inactive_parameter_count()
        if self.aggregation == 'span_set' and self.bidirectional and self.symmetric_readout:
            count += self.head.out_features * self.latent_dim
        return count

    def configuration_manifest(self) -> dict:
        configuration = {
            'schema_version': 1,
            'kind': 'experimental_graph_conditioned_affine_semigroup',
            'scientific_status': 'candidate_no_novelty_or_performance_claim',
            'input_dim': self.input_dim, 'num_outputs': self.num_outputs,
            'hidden_dim': self.hidden_dim, 'num_layers': self.num_layers,
            'latent_dim': self.latent_dim, 'generator_rank': self.generator.rank,
            'matrix_kind': self.generator.matrix_kind,
            'aggregation': self.aggregation, 'bidirectional': self.bidirectional,
            'symmetric_readout': self.symmetric_readout,
            'initial_rate': self.generator.initial_rate,
            'minimum_damping': self.generator.minimum_damping,
            'source_bank': 'independent SharedGraphEncoder query-leaf states',
            'persistent_identity': 'matching only, no learned arbitrary ID embedding',
            'discretization': 'exact affine zero-order-hold matrix exponential',
            'boundary_states': 'zero at fixed supplied tile endpoints',
            'backward_scan': 'positive-duration reversed sequence, not inverse flow',
            'coordinate': 'supplied normalized GraphSequence coordinate',
            'parameters_allocated': sum(p.numel() for p in self.parameters()),
            'structurally_inactive_parameters': self.inactive_parameter_count(),
        }
        canonical = json.dumps(configuration, sort_keys=True, separators=(',', ':')).encode()
        configuration['configuration_sha256'] = hashlib.sha256(canonical).hexdigest()
        configuration['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        configuration['local_encoder_dependency_sha256'] = hashlib.sha256(Path(__file__).with_name('models.py').read_bytes()).hexdigest()
        return configuration

    def generator_bank(self, seq: GraphSequenceInput) -> GeneratorBank:
        if self.validate_inputs:
            validate_model_input(seq)
        local = self.encoder(seq)[-1][query_node_index(seq)]
        a, b = self.generator(local)
        return GeneratorBank(local, a, b, augmented_affine_generator(a, b))

    def encode_context(self, seq: GraphSequenceInput) -> SemigroupContext:
        bank = self.generator_bank(seq)
        local = bank.local[seq.target_graph]
        widths = seq.intervals[:, 1] - seq.intervals[:, 0]
        q = seq.query_node_ids.numel()
        zero = bank.local.new_zeros(q, self.latent_dim)
        if self.aggregation in {'span_set', 'homogeneous_set'}:
            weighted = bank.affine * widths.to(bank.affine.dtype)[:, None, None, None]
            combined = weighted.sum(dim=0)
            if self.aggregation == 'span_set':
                context = apply_affine_transition(affine_transition(combined, 1.0), zero)
                context = context.unsqueeze(0).expand(seq.positions.numel(), -1, -1)
                return SemigroupContext(local, context, context if self.bidirectional else None)
            left, right = seq.intervals[:, 0].min(), seq.intervals[:, 1].max()
            averaged = combined / (right - left).to(combined.dtype)
            averaged = averaged.unsqueeze(0).expand(seq.positions.numel(), -1, -1, -1)
            zero_at_positions = zero.unsqueeze(0).expand(seq.positions.numel(), -1, -1)
            forward = apply_affine_transition(affine_transition(averaged, seq.positions - left), zero_at_positions)
            backward = None
            if self.bidirectional:
                backward = apply_affine_transition(affine_transition(averaged, right - seq.positions), zero_at_positions)
            return SemigroupContext(local, forward, backward)
        order = torch.argsort(seq.intervals[:, 0])
        generators = bank.affine[order]
        intervals = seq.intervals[order]
        transitions = affine_transition(generators, widths[order])
        # All full-step exponentials are batched. Ordered state products below
        # are the first simple implementation, not a parallel-scan kernel.
        left_states = []
        state = zero
        for transition in transitions.unbind(dim=0):
            left_states.append(state)
            state = apply_affine_transition(transition, state)
        left_states = torch.stack(left_states)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        target = inverse[seq.target_graph]
        # Difference is formed in coordinate precision before the exp dtype cast.
        left_duration = seq.positions - intervals[target, 0]
        partial_left = affine_transition(generators[target], left_duration)
        forward = apply_affine_transition(partial_left, left_states[target])
        backward = None
        if self.bidirectional:
            right_states = [zero] * order.numel()
            state = zero
            for graph in range(order.numel() - 1, -1, -1):
                right_states[graph] = state
                state = apply_affine_transition(transitions[graph], state)
            right_states = torch.stack(right_states)
            right_duration = intervals[target, 1] - seq.positions
            partial_right = affine_transition(generators[target], right_duration)
            backward = apply_affine_transition(partial_right, right_states[target])
        return SemigroupContext(local, forward, backward)

    def forward(self, seq: GraphSequenceInput) -> Tensor:
        context = self.encode_context(seq)
        if not self.bidirectional:
            features = torch.cat((context.local, context.forward), dim=-1)
        elif self.symmetric_readout:
            # Reflection swaps the two scans. Sum and squared difference retain
            # that symmetry without imposing arbitrary chronology on chromosomes.
            features = torch.cat((context.local, context.forward + context.backward,
                                  (context.forward - context.backward).square()), dim=-1)
        else:
            features = torch.cat((context.local, context.forward, context.backward), dim=-1)
        return self.head(features)


class SpanSetSemigroupModel(SemigroupGraphModel):
    def __init__(self, *args, **kwargs):
        kwargs['aggregation'] = 'span_set'
        super().__init__(*args, **kwargs)

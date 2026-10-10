"""Fixed-anchor nonlinear GNN on cached conditional boundary kernels.

Exact finite anchor-kernel actions with Q-only states and independently encoded
clamped observed R sources. This is not nonlinear continuous-source integration,
an anchor-refinement contract, a novelty claim or a predictive result.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .boundary_memory import BoundaryResponseGNN
from .kron_interface import LeafInterfaceBank


@dataclass(frozen=True)
class AnchorOperators:
    cqq: Tensor                  # [G,S,Q,Q], fixed normalized query blocks
    cqr: Tensor                  # [G,S,Q,R], clamped reference source map
    eigenvalues: Tensor          # [G,S,Q]
    eigenvectors: Tensor         # [G,S,Q,Q]

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> AnchorOperators:
        return AnchorOperators(*(getattr(self, n).to(device=device, dtype=dtype)
            for n in ("cqq", "cqr", "eigenvalues", "eigenvectors")))


@dataclass(frozen=True)
class AnchorKernelLayout:
    anchors: Tensor              # [A], fixed physical cell midpoints
    weights: Tensor              # [A], physical cell widths
    anchor_graph: Tensor         # [A], half-open containing serialized graph
    coordinates: Tensor          # sorted union of endpoints, anchors, readouts
    segment_graph: Tensor        # [E-1], containing serialized graph
    segment_duration: Tensor     # [E-1], physical differences in float64
    anchor_output: Tensor        # [A], into unique evaluation coordinates
    target_output: Tensor        # [P], original target order incl duplicates
    source_at_event: tuple[int, ...]   # -1 or physical anchor index
    output_at_event: tuple[int, ...]   # -1 or unique evaluation index
    num_evaluations: int

    def to(self, device: torch.device | str) -> AnchorKernelLayout:
        values = {name: getattr(self, name).to(device=device) for name in (
            "anchors", "weights", "anchor_graph", "coordinates", "segment_graph",
            "segment_duration", "anchor_output", "target_output")}
        return replace(self, **values)


@dataclass(frozen=True)
class AnchorKernelBank:
    bank: LeafInterfaceBank
    layout: AnchorKernelLayout
    operators: dict[str, AnchorOperators]
    num_scales: int

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> AnchorKernelBank:
        return AnchorKernelBank(self.bank.to(device, dtype), self.layout.to(device),
            {name: value.to(device, dtype) for name, value in self.operators.items()}, self.num_scales)


def _layout(bank: LeafInterfaceBank, count: int) -> AnchorKernelLayout:
    if not isinstance(count, int) or count < 1:
        raise ValueError("num_anchors must be a positive integer")
    intervals, positions = bank.intervals.detach().double(), bank.positions.detach().double()
    if intervals.device.type != "cpu" or positions.device.type != "cpu":
        raise ValueError("Prepare anchor coordinates on CPU before transferring the bank")
    if intervals.ndim != 2 or intervals.shape[1] != 2 or intervals.shape[0] == 0 or positions.ndim != 1:
        raise ValueError("Require positive intervals and a fixed target-coordinate vector")
    if not bool(torch.isfinite(intervals).all()) or not bool(torch.isfinite(positions).all()):
        raise ValueError("Coordinates must be finite")
    order = torch.argsort(intervals[:, 0])
    sorted_intervals = intervals[order]
    if bool((sorted_intervals[:, 1] <= sorted_intervals[:, 0]).any()) or not torch.equal(sorted_intervals[:-1, 1], sorted_intervals[1:, 0]):
        raise ValueError("Intervals must form an exact positive contiguous tile")
    left, right = sorted_intervals[0, 0], sorted_intervals[-1, 1]
    if bank.window_bounds is not None and not torch.equal(bank.window_bounds.detach().cpu().double(), torch.stack((left, right))):
        raise ValueError("Supplied tile endpoints differ from exact interval coverage")
    if bool(((positions < left) | (positions >= right)).any()):
        raise ValueError("Readouts must lie within the half-open tile")
    anchor = left+(torch.arange(count, dtype=torch.float64)+.5)*((right-left)/count)
    if torch.unique(anchor).numel() != count or bool(((anchor <= left) | (anchor >= right)).any()):
        raise ValueError("Coordinate precision cannot represent the fixed midpoint anchors")
    width = ((right-left)/count).expand(count).clone()
    starts = sorted_intervals[:, 0].contiguous()
    anchor_rank = torch.searchsorted(starts, anchor, right=True)-1
    target_rank = torch.searchsorted(starts, positions, right=True)-1
    if bank.target_graph.shape != positions.shape or not torch.equal(bank.target_graph, order[target_rank]):
        raise ValueError("Target graph mapping differs from half-open containing intervals")
    evaluation = torch.unique(torch.cat((anchor, positions)), sorted=True)
    coordinates = torch.unique(torch.cat((sorted_intervals.flatten(), evaluation)), sorted=True)
    segment_rank = torch.searchsorted(starts, coordinates[:-1], right=True)-1
    segment_graph = order[segment_rank]
    durations = coordinates[1:]-coordinates[:-1]
    if bool((coordinates[1:] > sorted_intervals[segment_rank, 1]).any()):
        raise ValueError("A physical segment crosses an uncached graph hold")
    anchor_event = torch.searchsorted(coordinates, anchor)
    evaluation_event = torch.searchsorted(coordinates, evaluation)
    source_at_event = [-1]*len(coordinates)
    output_at_event = [-1]*len(coordinates)
    for p, event in enumerate(anchor_event.tolist()):
        source_at_event[event] = p
    for p, event in enumerate(evaluation_event.tolist()):
        output_at_event[event] = p
    return AnchorKernelLayout(anchor, width, order[anchor_rank], coordinates,
        segment_graph, durations, torch.searchsorted(evaluation, anchor),
        torch.searchsorted(evaluation, positions), tuple(source_at_event),
        tuple(output_at_event), len(evaluation))


def _check_operators(operators: AnchorOperators, graphs: int, scales: int, q: int, r: int) -> None:
    expected = {"cqq": (graphs,scales,q,q), "cqr": (graphs,scales,q,r),
        "eigenvalues": (graphs,scales,q), "eigenvectors": (graphs,scales,q,q)}
    for name, shape in expected.items():
        value = getattr(operators, name)
        if value.device.type != "cpu" or value.shape != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Cached {name} shape/CPU/finiteness mismatch")
    cqq = operators.cqq
    tolerance = 64*q*torch.finfo(cqq.dtype).eps*cqq.abs().amax().clamp_min(1)
    if bool((cqq-cqq.transpose(-1,-2)).abs().amax() > tolerance):
        raise ValueError("Cached query operators must be symmetric")
    offdiag = cqq-torch.diag_embed(cqq.diagonal(dim1=-2,dim2=-1))
    if bool((offdiag > tolerance).any()) or (r and bool((operators.cqr > tolerance).any())):
        raise ValueError("Cached response requires nonpositive off-diagonal/reference blocks")
    reconstructed = operators.eigenvectors@torch.diag_embed(operators.eigenvalues)@operators.eigenvectors.transpose(-1,-2)
    if not torch.allclose(reconstructed,cqq,rtol=1e-8,atol=float(tolerance)):
        raise ValueError("Cached eigensystem does not reconstruct its query operator")
    identity = torch.eye(q, dtype=cqq.dtype)
    if not torch.allclose(operators.eigenvectors.transpose(-1,-2)@operators.eigenvectors,
            identity.expand(graphs,scales,-1,-1),rtol=1e-8,atol=float(tolerance)) or bool((operators.eigenvalues < -tolerance).any()):
        raise ValueError("Cached query eigensystem must be orthonormal and PSD")


@torch.no_grad()
def prepare_anchor_kernel_bank(bank: LeafInterfaceBank, age_cache: Any = None, *,
        num_anchors: int = 32, num_scales: int | None = None) -> AnchorKernelBank:
    """Prepare fixed physical metadata, reusing all supplied cached eigenpairs.

    No eigenproblem is solved here. Without an age cache, only repeated plain
    geometry is available. Independent sources always use the plain bank.
    """
    cache = bank.query_cache
    if cache is None or cache.normalization != "mean_boundary_diagonal":
        raise ValueError("A plain full-boundary mean-diagonal Q/R cache is required")
    if not torch.equal(cache.query_ids, bank.query_node_ids):
        raise ValueError("Cache query identity order differs from bank")
    available_scales = None if age_cache is None else age_cache.cqq.shape[1]
    scales = (available_scales or 1) if num_scales is None else num_scales
    if not isinstance(scales,int) or scales < 1 or (available_scales is not None and scales != available_scales):
        raise ValueError("Positive branch count must match the fixed age cache")
    plain = AnchorOperators(cache.cqq[:,None].expand(-1,scales,-1,-1),
        cache.cqr[:,None].expand(-1,scales,-1,-1),
        cache.eigenvalues[:,None].expand(-1,scales,-1),
        cache.eigenvectors[:,None].expand(-1,scales,-1,-1))
    operators = {"plain":plain}
    if age_cache is not None:
        if age_cache.normalization != "shared_plain_mean_boundary_diagonal":
            raise ValueError("Age operators must use the shared plain full-boundary scale")
        for name in ("query_ids","reference_ids","query_indices","reference_indices"):
            if not torch.equal(getattr(age_cache,name),getattr(cache,name)):
                raise ValueError("Age cache partition order differs from plain input")
        if not torch.equal(age_cache.normalization_scale,cache.normalization_scale):
            raise ValueError("Age cache normalization scale differs from plain input")
        if not torch.equal(age_cache.cqq[:,0],cache.cqq) or not torch.equal(age_cache.cqr[:,0],cache.cqr):
            raise ValueError("First age branch must be the supplied plain operator")
        operators["age"] = AnchorOperators(age_cache.cqq,age_cache.cqr,
            age_cache.eigenvalues,age_cache.eigenvectors)
        operators["matched_scalar_age"] = AnchorOperators(age_cache.scalar_cqq,
            age_cache.scalar_cqr,age_cache.scalar_eigenvalues,age_cache.scalar_eigenvectors)
    graphs,boundary,_ = bank.leaf_x.shape
    q,r = cache.query_ids.numel(),cache.reference_ids.numel()
    if q+r != boundary:
        raise ValueError("The Q/R partition must exhaust observed boundary samples")
    for value in operators.values():
        _check_operators(value,graphs,scales,q,r)
    return AnchorKernelBank(bank,_layout(bank,num_anchors),operators,scales)


def _hold_action(state: Tensor, vectors: Tensor | None, factors: Tensor) -> Tensor:
    # [S,Q,H,V], where V can bundle numerator and matching ones-source mass.
    shape = state.shape
    flat = state.flatten(-2)
    if vectors is not None:
        flat = vectors.transpose(-1,-2)@flat
    evolved = (factors[...,None]*flat.reshape(shape)).flatten(-2)
    if vectors is not None:
        evolved = vectors@evolved
    return evolved.reshape(shape)


def gain_matched_anchor_sources(operators: AnchorOperators, anchor_graph: Tensor,
        sources: Tensor, rates: Tensor, damping: Tensor) -> Tensor:
    """D*K_full^-1 times each anchor source, including matching mass sources.

    The transform preserves homogeneous continuous-forcing equilibrium gain.
    It does not match pointwise finite-anchor messages or a global equilibrium
    across changing operators. It mixes instantaneous sources, not prior states.
    """
    values = operators.eigenvalues[anchor_graph]
    vectors = operators.eigenvectors[anchor_graph]
    decay = values[...,None]*rates[None,:,None,:]+damping[None,:,None,:]
    diagonal = operators.cqq[anchor_graph].diagonal(dim1=-2,dim2=-1)[...,None]*rates[None,:,None,:]+damping[None,:,None,:]
    if not bool(torch.isfinite(decay).all()) or not bool(torch.isfinite(diagonal).all()) or bool((decay<=0).any()) or bool((diagonal<=0).any()):
        raise ValueError("Gain-matched full/diagonal decay must be finite and positive")
    shape = sources.shape
    projected = (vectors.transpose(-1,-2)@sources.flatten(-2)).reshape(shape)
    equilibrium = (vectors@(projected/decay[...,None]).flatten(-2)).reshape(shape)
    transformed = diagonal[...,None]*equilibrium
    if not bool(torch.isfinite(transformed).all()):
        raise ValueError("Gain-matched anchor source overflow")
    return transformed


def anchor_kernel_action(layout: AnchorKernelLayout, operators: AnchorOperators,
        sources: Tensor, rates: Tensor, damping: Tensor, *, coupling: str = "full") -> Tensor:
    """Exact finite anchor sum at all unique anchor/readout coordinates.

    Sources are [A,S,Q,H,V]; V=1 for an ordinary source or V=2 when numerator
    and normalization mass are scanned together. Returns [M,S,Q,H,V]. Forward
    is inclusive and reverse exclusive, so coincident anchor self terms occur
    once. Actual physical subholds preserve noncommuting chronological order.
    The gain-matched control transforms every V component by D*K_full^-1
    at its anchor before diagonal propagation, including ones-source mass.
    """
    if coupling not in {"full","scalar","diagonal","gain_matched_diagonal"}:
        raise ValueError("Unknown kernel coupling control")
    a,s,q,h,v = sources.shape if sources.ndim == 5 else (0,0,0,0,0)
    if a != layout.anchors.numel() or rates.shape != (s,h) or damping.shape != (s,h) or operators.eigenvalues.shape[1:] != (s,q):
        raise ValueError("Require matching [A,S,Q,H,V] sources and global [S,H] rates")
    if not bool(torch.isfinite(sources).all()) or not bool(torch.isfinite(rates).all()) or not bool(torch.isfinite(damping).all()) or bool((rates <= 0).any()) or bool((damping <= 0).any()):
        raise ValueError("Sources and strictly positive rates must be finite")
    if coupling == "gain_matched_diagonal":
        sources = gain_matched_anchor_sources(operators,layout.anchor_graph,sources,rates,damping)
    if coupling == "full":
        values = operators.eigenvalues
    elif coupling == "scalar":
        values = torch.ones_like(operators.eigenvalues)
    else:
        values = operators.cqq.diagonal(dim1=-2,dim2=-1)
    decay = values[...,None]*rates[None,:,None,:]+damping[None,:,None,:]
    if not bool(torch.isfinite(decay).all()) or bool((decay <= 0).any()):
        raise ValueError("Kernel decay is not numerically positive")
    graph = layout.segment_graph
    dt = layout.segment_duration.to(sources.dtype)
    factors = torch.exp(-dt[:,None,None,None]*decay[graph])
    vectors = operators.eigenvectors[graph] if coupling == "full" else None
    impulse = sources*layout.weights.to(sources.dtype)[:,None,None,None,None]
    zero = sources.new_zeros((s,q,h,v))
    forward = [zero]*layout.num_evaluations
    reverse = [zero]*layout.num_evaluations
    state = zero
    for event,(source,output) in enumerate(zip(layout.source_at_event,layout.output_at_event)):
        if event:
            state = _hold_action(state,None if vectors is None else vectors[event-1],factors[event-1])
        if source >= 0:
            state = state+impulse[source]
        if output >= 0:
            forward[output] = state
    state = zero
    for event in range(len(layout.source_at_event)-1,-1,-1):
        source,output = layout.source_at_event[event],layout.output_at_event[event]
        if output >= 0:
            reverse[output] = state
        if source >= 0:
            state = state+impulse[source]
        if event:
            state = _hold_action(state,None if vectors is None else vectors[event-1],factors[event-1])
    return torch.stack(forward)+torch.stack(reverse)


def _initial_vector(initial: float | Sequence[float], scales: int, name: str) -> Tensor:
    values = [initial]*scales if isinstance(initial,(int,float)) else list(initial)
    if len(values) != scales or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError(f"{name} requires one finite positive initial value per branch")
    return torch.tensor(values,dtype=torch.float32)


@dataclass(frozen=True)
class AnchorLayerMessages:
    anchors: Tensor              # [A,Q,H], normalized nonlinear message input
    targets: Tensor              # [P,Q,H], exact coordinate normalized message
    numerator: Tensor            # [M,Q,H]
    mass: Tensor                 # [M,Q,H], no floor
    sources: Tensor              # [A,S,Q,H], evaluated only at fixed anchors
    mass_sources: Tensor         # [A,S,Q,H], 1Q-lambda*Cqr*1R


class AnchorKernelLayer(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, num_scales: int,
                 initial_coupling: float | Sequence[float], initial_damping: float | Sequence[float],
                 rate_parameterization: str = "log"):
        super().__init__()
        if rate_parameterization not in {"log","softplus"}:
            raise ValueError("rate_parameterization must be log or softplus")
        self.rate_parameterization = rate_parameterization
        self.query_projection = nn.Linear(hidden_dim,latent_dim)
        self.reference_projection = nn.Linear(hidden_dim,latent_dim)
        self.message_projection = nn.Linear(latent_dim,hidden_dim,bias=False)
        self.gate = nn.Linear(hidden_dim+latent_dim,hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("initial_coupling",_initial_vector(initial_coupling,num_scales,"initial_coupling")[:,None])
        self.register_buffer("initial_damping",_initial_vector(initial_damping,num_scales,"initial_damping")[:,None])
        self.register_buffer("softplus_coupling_offset",self.initial_coupling+torch.log(-torch.expm1(-self.initial_coupling)))
        self.register_buffer("softplus_damping_offset",self.initial_damping+torch.log(-torch.expm1(-self.initial_damping)))
        self.raw_coupling = nn.Parameter(torch.zeros(num_scales,latent_dim))
        self.raw_damping = nn.Parameter(torch.zeros(num_scales,latent_dim))
        self.mixture_logits = nn.Parameter(torch.zeros(num_scales,latent_dim))

    @property
    def rates(self) -> Tensor:
        if self.rate_parameterization == "log":
            return self.initial_coupling*torch.exp(self.raw_coupling)
        return self.initial_coupling*(F.softplus(self.softplus_coupling_offset+self.raw_coupling)/F.softplus(self.softplus_coupling_offset))

    @property
    def damping(self) -> Tensor:
        if self.rate_parameterization == "log":
            return self.initial_damping*torch.exp(self.raw_damping)
        return self.initial_damping*(F.softplus(self.softplus_damping_offset+self.raw_damping)/F.softplus(self.softplus_damping_offset))

    @property
    def mixture(self) -> Tensor:
        return self.mixture_logits.softmax(dim=0)

    def messages(self, prepared: AnchorKernelBank, query: Tensor, reference: Tensor,
                 geometry: str, coupling: str) -> AnchorLayerMessages:
        operators,layout = prepared.operators[geometry],prepared.layout
        rates,damping = self.rates,self.damping
        q_signal = self.query_projection(query).tanh()
        r_signal = self.reference_projection(reference).tanh()
        cqr = operators.cqr[layout.anchor_graph]
        sources = q_signal[:,None]-(cqr@r_signal[:,None])*rates[None,:,None,:]
        mass = torch.ones_like(sources)-cqr.sum(-1)[...,None]*rates[None,:,None,:]
        values = anchor_kernel_action(layout,operators,torch.stack((sources,mass),dim=-1),
            rates,damping,coupling=coupling)
        mixed = (values*self.mixture[None,:,None,:,None]).sum(dim=1)
        numerator,denominator = mixed.unbind(-1)
        if not bool(torch.isfinite(mixed).all()) or bool((denominator <= 0).any()):
            raise ValueError("Kernel row mass is nonpositive/nonfinite, including possible readout underflow")
        normalized = numerator/denominator
        return AnchorLayerMessages(normalized[layout.anchor_output],normalized[layout.target_output],
            numerator,denominator,sources,mass)

    def update(self, local: Tensor, message: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((local,message),dim=-1)))
        return self.norm(local+gate*self.message_projection(message))


@dataclass(frozen=True)
class AnchorKernelContext:
    anchors: Tensor
    targets: Tensor
    layers: tuple[AnchorLayerMessages,...]


class AnchorKernelModel(nn.Module):
    """Repeated nonlinear Q-context updates at fixed physical anchors only."""
    def __init__(self,input_dim: int = 47,num_classes: int = 4,hidden_dim: int = 32,
                 num_layers: int = 2,latent_dim: int = 8,kernel_depth: int = 2,
                 num_scales: int = 1,geometry: str = "plain",coupling: str = "full",
                 num_anchors: int = 32,initial_coupling: float | Sequence[float] = 20.,
                 initial_damping: float | Sequence[float] = 1.,rate_parameterization: str = "log"):
        super().__init__()
        if min(input_dim,num_classes,hidden_dim,num_layers,latent_dim,kernel_depth,num_scales,num_anchors) < 1:
            raise ValueError("All model dimensions, depths and anchor/branch counts must be positive")
        if geometry not in {"plain","age","matched_scalar_age"} or coupling not in {"full","scalar","diagonal","gain_matched_diagonal"}:
            raise ValueError("Unknown geometry or kernel coupling control")
        self.input_dim,self.num_classes,self.hidden_dim = input_dim,num_classes,hidden_dim
        self.num_layers,self.latent_dim,self.kernel_depth = num_layers,latent_dim,kernel_depth
        self.num_scales,self.geometry,self.coupling,self.num_anchors = num_scales,geometry,coupling,num_anchors
        self.rate_parameterization = rate_parameterization
        self.encoder = BoundaryResponseGNN(input_dim,hidden_dim,num_layers)
        self.layers = nn.ModuleList(AnchorKernelLayer(hidden_dim,latent_dim,num_scales,
            initial_coupling,initial_damping,rate_parameterization) for _ in range(kernel_depth))
        self.head = nn.Linear(hidden_dim,num_classes)

    def rate_and_mixture_parameters(self):
        """Parameters to exclude from weight decay in matched optimization controls."""
        for layer in self.layers:
            yield from (layer.raw_coupling,layer.raw_damping,layer.mixture_logits)

    def encode_context(self, prepared: AnchorKernelBank) -> AnchorKernelContext:
        bank,layout = prepared.bank,prepared.layout
        if prepared.num_scales != self.num_scales or layout.anchors.numel() != self.num_anchors or self.geometry not in prepared.operators:
            raise ValueError("Prepared branch/anchor count or geometry differs from model")
        weight = self.encoder.input_linear.weight
        if bank.leaf_x.shape[-1] != self.input_dim or bank.leaf_x.dtype != weight.dtype or bank.leaf_x.device != weight.device:
            raise ValueError("Prepare shared features with matching model input dimension/device/dtype")
        if not bool(torch.isfinite(bank.leaf_x).all()):
            raise ValueError("Observed feature bank must be finite")
        h = self.encoder(bank.leaf_x,bank.query_cache.normalized_response)
        query = h[:,bank.query_cache.query_indices]
        reference = h[:,bank.query_cache.reference_indices]
        anchors = query[layout.anchor_graph]
        targets = query[bank.target_graph]
        fixed_reference = reference[layout.anchor_graph]
        messages = []
        for layer in self.layers:
            message = layer.messages(prepared,anchors,fixed_reference,self.geometry,self.coupling)
            anchors,targets = layer.update(anchors,message.anchors),layer.update(targets,message.targets)
            messages.append(message)
        return AnchorKernelContext(anchors,targets,tuple(messages))

    def forward(self,prepared: AnchorKernelBank) -> Tensor:
        return self.head(self.encode_context(prepared).targets)

    def configuration_manifest(self,prepared: AnchorKernelBank | None = None) -> dict[str,Any]:
        result = {"kind":"experimental_fixed_anchor_boundary_kernel_GNN",
            "input_dim":self.input_dim,"num_outputs":self.num_classes,"hidden_dim":self.hidden_dim,
            "source_GNN_depth":self.num_layers,"latent_dim":self.latent_dim,"kernel_depth":self.kernel_depth,
            "num_scales":self.num_scales,"geometry":self.geometry,"coupling":self.coupling,
            "num_anchors":self.num_anchors,"anchor_grid":"equal physical cell midpoints with cell-width masses",
            "source_encoder":"same independent plain-boundary GNN and observed selected features",
            "memory_interface":"queries only; references clamped independently encoded observed sources",
            "rate_parameterization":self.rate_parameterization,
            "rates":"init*exp(raw)" if self.rate_parameterization=="log" else "init*softplus(invsoftplus(init)+raw)/softplus(invsoftplus(init))",
            "rate_scope":"global per layer/scale/channel",
            "scale_mixture":"softmax over scales, global per layer/channel, separate branch states",
            "normalization":"matching ones query/reference source scans; no floor",
            "source_maps":"bounded tanh q/r, shared across scales in each layer",
            "scalar_control":"ordinary scalar time-kernel family; K=(lambda+d)I, original Cqr source map retained" if self.coupling=="scalar" else None,
            "diagonal_control":"D=lambda*diag(Cqq)+d; original numerator/mass sources" if self.coupling=="diagonal" else None,
            "gain_matched_control":"D*K_full^-1 times numerator and mass anchor sources, then independent D propagation" if self.coupling=="gain_matched_diagonal" else None,
            "gain_matching_scope":"homogeneous continuous-forcing equilibrium / integrated infinite-domain impulse gain; not pointwise finite-anchor or changing-operator equilibrium" if self.coupling=="gain_matched_diagonal" else None,
            "initial_state_cross_query_exchange":self.coupling=="full",
            "instantaneous_anchor_source_mixing":self.coupling=="gain_matched_diagonal",
            "self_term":"inclusive forward plus exclusive reverse counts once",
            "decoder":"each layer exact readout-coordinate action from its fixed anchor sources and same gate",
            "row_normalized_operator_is_PSD_Gram":False,"underlying_symmetric_block_kernel_is_PSD":True,
            "online_eigensolve":False,"parameters":sum(p.numel() for p in self.parameters()),
            "initial_coupling":[layer.initial_coupling.detach().flatten().tolist() for layer in self.layers],
            "initial_damping":[layer.initial_damping.detach().flatten().tolist() for layer in self.layers],
            "current_rates":[layer.rates.detach().tolist() for layer in self.layers],
            "current_damping":[layer.damping.detach().tolist() for layer in self.layers],
            "current_mixture":[layer.mixture.detach().tolist() for layer in self.layers],
            "raw_coupling":[layer.raw_coupling.detach().tolist() for layer in self.layers],
            "raw_damping":[layer.raw_damping.detach().tolist() for layer in self.layers],
            "mixture_logits":[layer.mixture_logits.detach().tolist() for layer in self.layers],
            "limitations":"finite anchor action, no continuous nonlinear integral or anchor-refinement guarantee; no performance/novelty result"}
        if prepared is not None:
            result.update(feature_columns=list(prepared.bank.feature_columns),
                query_ids=prepared.bank.query_node_ids.tolist(),reference_ids=prepared.bank.query_cache.reference_ids.tolist(),
                anchors=prepared.layout.anchors.tolist(),weights=prepared.layout.weights.tolist(),
                inactive_reference_parameters=(self.hidden_dim+1)*self.latent_dim*self.kernel_depth if prepared.bank.query_cache.reference_ids.numel()==0 else 0)
        result["configuration_sha256"] = hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

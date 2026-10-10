"""Versioned faithful finite-anchor action with compact spectral hold scans.

This module does not change the frozen candidate or its execution admission.
Fixed geometry only. Readouts do not create impulses. Equivalent in real
arithmetic, with qualified floating-point rounding differences, not bitwise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .anchor_kernel import AnchorKernelLayout, AnchorLayerMessages, AnchorOperators


@dataclass(frozen=True)
class CompactAnchorLayoutV1:
    """Compact events are hold endpoints and fixed midpoint anchors only.

    Host tuples control a short sequential scan without per-event device item
    extraction. Evaluations are unique sorted anchors/readouts; target_output
    restores original target order, including duplicates. Coordinate tensors
    remain float64 when moving the execution metadata to another device.
    """
    anchors: Tensor
    weights: Tensor
    anchor_graph: Tensor
    evaluations: Tensor
    anchor_output: Tensor
    target_output: Tensor
    coordinates: Tensor
    segment_graph: Tensor
    segment_duration: Tensor
    evaluation_left: Tensor
    evaluation_right: Tensor
    source_at_event: tuple[int, ...]
    graph_at_segment: tuple[int, ...]
    graph_at_anchor: tuple[int, ...]
    evaluation_at_segment: tuple[tuple[int, ...], ...]
    num_graphs: int

    @property
    def num_evaluations(self) -> int:
        return self.evaluations.numel()

    @property
    def num_segments(self) -> int:
        return len(self.graph_at_segment)

    def to(self, device: torch.device | str) -> CompactAnchorLayoutV1:
        names = ("anchors", "weights", "anchor_graph", "evaluations",
                 "anchor_output", "target_output", "coordinates",
                 "segment_graph", "segment_duration", "evaluation_left",
                 "evaluation_right")
        return replace(self, **{name: getattr(self, name).to(device=device)
                               for name in names})


def prepare_compact_anchor_layout_v1(intervals: Tensor, positions: Tensor, *,
        num_anchors: int = 32, target_graph: Tensor | None = None,
        window_bounds: Tensor | None = None) -> CompactAnchorLayoutV1:
    """Prepare CPU metadata, preserving the native fixed-anchor source measure."""
    if not isinstance(num_anchors, int) or num_anchors < 1:
        raise ValueError("num_anchors must be a positive integer")
    if intervals.device.type != "cpu" or positions.device.type != "cpu":
        raise ValueError("Prepare compact coordinate metadata on CPU")
    intervals, positions = intervals.detach().double(), positions.detach().double()
    if (intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals)
            or positions.ndim != 1):
        raise ValueError("Require nonempty [G,2] intervals and [P] readouts")
    if not bool(torch.isfinite(intervals).all() & torch.isfinite(positions).all()):
        raise ValueError("Coordinates must be finite")
    order = torch.argsort(intervals[:, 0])
    holds = intervals[order]
    if (bool((holds[:, 1] <= holds[:, 0]).any())
            or not torch.equal(holds[:-1, 1], holds[1:, 0])):
        raise ValueError("Holds must be exactly positive and contiguous")
    left, right = holds[0, 0], holds[-1, 1]
    if window_bounds is not None:
        if (window_bounds.device.type != "cpu"
                or not torch.equal(window_bounds.detach().double(), torch.stack((left, right)))):
            raise ValueError("Window endpoints differ from exact hold coverage")
    if bool(((positions < left) | (positions >= right)).any()):
        raise ValueError("Readouts must be inside the half-open tile")
    anchors = left + (torch.arange(num_anchors, dtype=torch.float64) + .5) * (
        (right - left) / num_anchors)
    if (torch.unique(anchors).numel() != num_anchors
            or bool(((anchors <= left) | (anchors >= right)).any())):
        raise ValueError("Coordinate precision cannot represent midpoint anchors")
    weights = ((right - left) / num_anchors).expand(num_anchors).clone()
    starts = holds[:, 0].contiguous()
    anchor_graph = order[torch.searchsorted(starts, anchors, right=True) - 1]
    expected_target = order[torch.searchsorted(starts, positions, right=True) - 1]
    if target_graph is not None:
        if (target_graph.device.type != "cpu" or target_graph.dtype != torch.long
                or not torch.equal(target_graph, expected_target)):
            raise ValueError("Target graph differs from half-open physical mapping")
    evaluations = torch.unique(torch.cat((anchors, positions)), sorted=True)
    coordinates = torch.unique(torch.cat((holds.flatten(), anchors)), sorted=True)
    rank = torch.searchsorted(starts, coordinates[:-1], right=True) - 1
    segment_graph = order[rank]
    duration = coordinates[1:] - coordinates[:-1]
    if (bool((duration <= 0).any())
            or bool((coordinates[1:] > holds[rank, 1]).any())):
        raise ValueError("Compact segment crosses a supplied hold")
    evaluation_segment = torch.searchsorted(coordinates, evaluations, right=True) - 1
    evaluation_left = evaluations - coordinates[evaluation_segment]
    evaluation_right = coordinates[evaluation_segment + 1] - evaluations
    source_at_event = [-1] * len(coordinates)
    for anchor, event in enumerate(torch.searchsorted(coordinates, anchors).tolist()):
        source_at_event[event] = anchor
    groups = [[] for _ in range(len(duration))]
    for output, segment in enumerate(evaluation_segment.tolist()):
        groups[segment].append(output)
    if tuple(i for group in groups for i in group) != tuple(range(len(evaluations))):
        raise ValueError("Compact evaluation order is inconsistent")
    return CompactAnchorLayoutV1(
        anchors, weights, anchor_graph, evaluations,
        torch.searchsorted(evaluations, anchors),
        torch.searchsorted(evaluations, positions), coordinates,
        segment_graph, duration, evaluation_left, evaluation_right,
        tuple(source_at_event), tuple(segment_graph.tolist()),
        tuple(anchor_graph.tolist()), tuple(tuple(group) for group in groups),
        len(intervals))


def compact_layout_from_anchor_v1(layout: AnchorKernelLayout) -> CompactAnchorLayoutV1:
    """Losslessly derive compact CPU metadata from a frozen native layout.

    Call before the native layout is transferred. This adapter validates the
    same midpoint anchors, weights, graph mapping and evaluation/output maps.
    Original serialized banks remain unchanged.
    """
    if layout.coordinates.device.type != "cpu":
        raise ValueError("Convert the native layout before device transfer")
    coordinates = layout.coordinates.detach().double()
    graph = layout.segment_graph
    if (graph.device.type != "cpu" or graph.dtype != torch.long
            or coordinates.ndim != 1 or graph.shape != (len(coordinates) - 1,)
            or not len(graph) or bool((graph < 0).any())):
        raise ValueError("Invalid native segment metadata")
    if not torch.equal(layout.segment_duration.double(), coordinates[1:] - coordinates[:-1]):
        raise ValueError("Native durations differ from exact coordinate differences")
    runs = []
    start = 0
    host_graph = graph.tolist()
    for stop in range(1, len(host_graph) + 1):
        if stop == len(host_graph) or host_graph[stop] != host_graph[start]:
            runs.append((host_graph[start], start, stop))
            start = stop
    ids = [g for g, _, _ in runs]
    if sorted(ids) != list(range(len(runs))):
        raise ValueError("Native hold IDs must occur in one contiguous run each")
    intervals = coordinates.new_empty((len(runs), 2))
    for g, start, stop in runs:
        intervals[g] = torch.stack((coordinates[start], coordinates[stop]))
    native_evaluations = coordinates.new_empty((layout.num_evaluations,))
    seen = set()
    if len(layout.output_at_event) != len(coordinates):
        raise ValueError("Invalid native output-event mapping")
    for event, output in enumerate(layout.output_at_event):
        if output >= 0:
            if output in seen or output >= layout.num_evaluations:
                raise ValueError("Invalid native evaluation identity")
            native_evaluations[output] = coordinates[event]
            seen.add(output)
    if seen != set(range(layout.num_evaluations)):
        raise ValueError("Native evaluation mapping is incomplete")
    compact = prepare_compact_anchor_layout_v1(
        intervals, native_evaluations[layout.target_output],
        num_anchors=layout.anchors.numel())
    for name in ("anchors", "weights", "anchor_graph", "anchor_output", "target_output"):
        if not torch.equal(getattr(compact, name), getattr(layout, name)):
            raise ValueError("Native fixed source/output measure changed: " + name)
    if not torch.equal(compact.evaluations, native_evaluations):
        raise ValueError("Native evaluations changed")
    if len(layout.source_at_event) != len(coordinates):
        raise ValueError("Invalid native source-event mapping")
    actual_sources = [(coordinates[e].item(), a) for e, a in enumerate(layout.source_at_event) if a >= 0]
    expected_sources = [(x, a) for a, x in enumerate(layout.anchors.tolist())]
    if actual_sources != expected_sources:
        raise ValueError("Native source events differ from fixed anchors")
    return compact


@dataclass(frozen=True)
class CompactSpectralGeometryV1:
    """Minimal fixed per-hold spectral bank, without per-readout bases.

    cqr is needed only for finite anchor source construction. The diagonal is
    optional for full coupling and required by diagonal/gain-matched controls.
    These fields alias the source arrays at derivation time, with no eigensolve,
    quantization, truncation or replacement of the original sealed artifact.
    """
    eigenvalues: Tensor
    eigenvectors: Tensor
    cqr: Tensor
    diagonal: Tensor | None = None

    @classmethod
    def from_operators(cls, operators: AnchorOperators, *,
                       include_diagonal: bool = False) -> CompactSpectralGeometryV1:
        return cls(operators.eigenvalues, operators.eigenvectors, operators.cqr,
                   operators.cqq.diagonal(dim1=-2, dim2=-1) if include_diagonal else None)

    def to(self, device: torch.device | str,
           dtype: torch.dtype | None = None) -> CompactSpectralGeometryV1:
        return CompactSpectralGeometryV1(
            self.eigenvalues.to(device=device, dtype=dtype),
            self.eigenvectors.to(device=device, dtype=dtype),
            self.cqr.to(device=device, dtype=dtype),
            None if self.diagonal is None else self.diagonal.to(device=device, dtype=dtype))


def _state_project(state: Tensor, vectors: Tensor) -> Tensor:
    return (vectors.transpose(-1, -2) @ state.flatten(-2)).reshape_as(state)


def _state_reconstruct(state: Tensor, vectors: Tensor) -> Tensor:
    return (vectors @ state.flatten(-2)).reshape_as(state)


def _readout_reconstruct_chunk(spectral: Tensor, vectors: Tensor) -> Tensor:
    """One U multiplication with readouts packed as columns, no repeated U.

    [B,S,Q,H,V] -> [S,Q,B*H*V] permits a single per-scale matrix product.
    A broadcast matmul with a [B,S,Q,Q] logical expansion is deliberately
    avoided because its implementation may materialize repeated bases.
    """
    b, s, q, h, v = spectral.shape
    packed = spectral.permute(1, 2, 0, 3, 4).reshape(s, q, b * h * v)
    physical = vectors @ packed
    return physical.reshape(s, q, b, h, v).permute(2, 0, 1, 3, 4)


def anchor_spectral_compact_action_v1(layout: CompactAnchorLayoutV1,
        geometry: CompactSpectralGeometryV1, sources: Tensor, rates: Tensor,
        damping: Tensor, *, coupling: str = "full",
        readout_chunk_size: int = 128) -> Tensor:
    """Exact real-arithmetic finite-anchor sum at unique evaluation positions.

    Sources [A,S,Q,H,V] bundle arbitrary numerator/mass components. Return
    [M,S,Q,H,V]. Forward includes each anchor self term and reverse excludes it.
    Fixed per-hold eigenpairs remain compact. Rate/source gradients use ordinary
    autograd through the compact scan and batched same-hold readouts.
    """
    if coupling not in {"full", "scalar", "diagonal", "gain_matched_diagonal"}:
        raise ValueError("Unknown compact kernel coupling")
    if not isinstance(readout_chunk_size, int) or readout_chunk_size < 1:
        raise ValueError("readout_chunk_size must be positive")
    if sources.ndim != 5:
        raise ValueError("Sources must be [A,S,Q,H,V]")
    a, s, q, h, v = sources.shape
    if (min(a, s, q, h, v) < 1 or a != len(layout.graph_at_anchor)
            or rates.shape != (s, h) or damping.shape != (s, h)
            or geometry.eigenvalues.shape != (layout.num_graphs, s, q)
            or geometry.eigenvectors.shape != (layout.num_graphs, s, q, q)):
        raise ValueError("Compact source/rate/geometry dimensions differ")
    if not sources.is_floating_point():
        raise ValueError("Sources must have floating dtype")
    tensors = (rates, damping, geometry.eigenvalues, geometry.eigenvectors)
    if any(x.device != sources.device or x.dtype != sources.dtype for x in tensors):
        raise ValueError("Source/rate/geometry device and dtype must match")
    if (layout.segment_duration.device != sources.device
            or layout.anchor_output.device != sources.device):
        raise ValueError("Transfer compact layout to the source device")
    fixed_fields = (geometry.eigenvalues, geometry.eigenvectors, geometry.cqr)
    if (any(x.requires_grad for x in fixed_fields)
            or (geometry.diagonal is not None and geometry.diagonal.requires_grad)):
        raise ValueError("Compact geometry must be fixed, without geometry gradients")
    flags = torch.stack((torch.isfinite(sources).all(), torch.isfinite(rates).all(),
                         torch.isfinite(damping).all(), (rates > 0).all(),
                         (damping > 0).all(), torch.isfinite(geometry.eigenvalues).all(),
                         torch.isfinite(geometry.eigenvectors).all()))
    if not bool(flags.all()):
        raise ValueError("Sources/fixed geometry and positive rates/damping must be finite")
    vectors = geometry.eigenvectors if coupling == "full" else None
    if coupling in {"diagonal", "gain_matched_diagonal"}:
        if (geometry.diagonal is None
                or geometry.diagonal.shape != geometry.eigenvalues.shape
                or geometry.diagonal.device != sources.device
                or geometry.diagonal.dtype != sources.dtype):
            raise ValueError("Diagonal controls require the matching fixed diagonal")
        values = geometry.diagonal
    elif coupling == "scalar":
        values = torch.ones_like(geometry.eigenvalues)
    else:
        values = geometry.eigenvalues
    decay = values[..., None] * rates[None, :, None, :] + damping[None, :, None, :]
    if not bool(torch.isfinite(decay).all() & (decay > 0).all()):
        raise ValueError("Compact decay must be finite and numerically positive")
    if coupling == "gain_matched_diagonal":
        full_decay = geometry.eigenvalues[..., None] * rates[None, :, None, :] + damping[None, :, None, :]
        if not bool(torch.isfinite(full_decay).all() & (full_decay > 0).all()):
            raise ValueError("Gain-matched full decay must be finite and positive")
        transformed = []
        for anchor, graph in enumerate(layout.graph_at_anchor):
            projected = _state_project(sources[anchor], geometry.eigenvectors[graph])
            equilibrium = _state_reconstruct(projected / full_decay[graph, ..., None],
                                             geometry.eigenvectors[graph])
            transformed.append(decay[graph, ..., None] * equilibrium)
        sources = torch.stack(transformed)
        if not bool(torch.isfinite(sources).all()):
            raise ValueError("Gain-matched sources overflowed")
    impulse = sources * layout.weights.to(sources.dtype)[:, None, None, None, None]
    projected_impulse = (torch.stack([_state_project(impulse[anchor], vectors[graph])
                          for anchor, graph in enumerate(layout.graph_at_anchor)])
                         if vectors is not None else impulse)
    factors = torch.exp(-layout.segment_duration.to(sources.dtype)[:, None, None, None]
                        * decay[layout.segment_graph])
    zero = sources.new_zeros((s, q, h, v))
    forward = []
    state = zero
    previous_graph = layout.graph_at_segment[0]
    for segment, graph in enumerate(layout.graph_at_segment):
        if vectors is not None and graph != previous_graph:
            state = _state_project(_state_reconstruct(state, vectors[previous_graph]), vectors[graph])
        source = layout.source_at_event[segment]
        if source >= 0:
            state = state + projected_impulse[source]
        forward.append(state)
        state = state * factors[segment, ..., None]
        previous_graph = graph
    backward = [zero] * layout.num_segments
    state = zero
    previous_graph = layout.graph_at_segment[-1]
    for segment in range(layout.num_segments - 1, -1, -1):
        graph = layout.graph_at_segment[segment]
        source = layout.source_at_event[segment + 1]
        # At a boundary the anchor belongs to the right-hand containing hold.
        # Include it before expressing the physical state in the left basis.
        if source >= 0:
            state = state + projected_impulse[source]
        if vectors is not None and graph != previous_graph:
            state = _state_project(_state_reconstruct(state, vectors[previous_graph]), vectors[graph])
        backward[segment] = state
        state = state * factors[segment, ..., None]
        previous_graph = graph
    chunks = []
    for segment, (graph, outputs) in enumerate(zip(layout.graph_at_segment,
                                                  layout.evaluation_at_segment)):
        for start in range(0, len(outputs), readout_chunk_size):
            indexes = outputs[start:start + readout_chunk_size]
            left = layout.evaluation_left[list(indexes)].to(sources.dtype)
            right = layout.evaluation_right[list(indexes)].to(sources.dtype)
            left_factor = torch.exp(-left[:, None, None, None] * decay[graph])
            right_factor = torch.exp(-right[:, None, None, None] * decay[graph])
            spectral = (left_factor[..., None] * forward[segment][None]
                        + right_factor[..., None] * backward[segment][None])
            chunks.append(_readout_reconstruct_chunk(spectral, vectors[graph])
                          if vectors is not None else spectral)
    return torch.cat(chunks, dim=0)


def compact_anchor_messages_v1(layout: CompactAnchorLayoutV1,
        geometry: CompactSpectralGeometryV1, query_signal: Tensor,
        reference_signal: Tensor, rates: Tensor, damping: Tensor, mixture: Tensor,
        *, coupling: str = "full", readout_chunk_size: int = 128) -> AnchorLayerMessages:
    """Native bounded-source/matching-ones normalization, compatible messages.

    Signals are already tanh projected [A,Q,H]/[A,R,H]. Scale mixture is the
    native [S,H] softmax. Anchor signals and all Q rows remain present even when
    a future versioned adapter selects only Q120 target gates/head outputs.
    """
    a, s, q = len(layout.graph_at_anchor), geometry.eigenvalues.shape[1], geometry.eigenvalues.shape[2]
    h = rates.shape[-1]
    r = geometry.cqr.shape[-1]
    if (query_signal.shape != (a, q, h) or reference_signal.shape != (a, r, h)
            or geometry.cqr.shape != (layout.num_graphs, s, q, r)
            or mixture.shape != (s, h)):
        raise ValueError("Compact query/reference/mixture dimensions differ")
    tensors = (reference_signal, geometry.cqr, mixture)
    if any(x.device != query_signal.device or x.dtype != query_signal.dtype for x in tensors):
        raise ValueError("Signals/source map/mixture device and dtype must match")
    if not bool(torch.isfinite(mixture).all() & (mixture >= 0).all()):
        raise ValueError("Mixture must be finite and nonnegative")
    cqr = geometry.cqr[layout.anchor_graph]
    sources = query_signal[:, None] - (cqr @ reference_signal[:, None]) * rates[None, :, None, :]
    mass_sources = torch.ones_like(sources) - cqr.sum(-1)[..., None] * rates[None, :, None, :]
    raw = anchor_spectral_compact_action_v1(
        layout, geometry, torch.stack((sources, mass_sources), dim=-1), rates, damping,
        coupling=coupling, readout_chunk_size=readout_chunk_size)
    mixed = (raw * mixture[None, :, None, :, None]).sum(dim=1)
    numerator, mass = mixed.unbind(-1)
    if not bool(torch.isfinite(mixed).all() & (mass > 0).all()):
        raise ValueError("Compact mixed numerator/mass is nonfinite or mass is nonpositive")
    normalized = numerator / mass
    return AnchorLayerMessages(normalized[layout.anchor_output], normalized[layout.target_output],
                               numerator, mass, sources, mass_sources)

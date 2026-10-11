"""Experimental persistence-trace kernel on complete observed metric trees.

Finite hold-center approximation, not a continuum/subdivision-exact operator.
Qualified parent-wide identities are joins only. Geometry and correspondence are
fixed. Only first-order implicit derivatives of the screened solve are supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class MetricTreeHoldV1:
    parent: Tensor                  # [N], graph-local rows, -1 at the one root
    node_times: Tensor              # [N], observed generations, float64 CPU
    qualified_node_ids: Tensor      # [N], parent-wide native IDs, -1 if unknown
    identity_namespace: str | None  # producer-qualified namespace, never a feature
    sample_nodes: Tensor            # [B], graph-local observed sample rows
    sample_ids: Tensor              # [B], actual persistent sample identities


@dataclass(frozen=True)
class PersistenceTraceBankV1:
    num_states: int
    source_nodes: Tensor
    source_weights: Tensor
    readout_left: Tensor            # [P,Q], carrier rows at bracketing centers
    readout_right: Tensor
    readout_right_weight: Tensor
    spatial: Tensor                # normalized sparse COO Laplacians
    sample: Tensor
    genuine_ancestor: Tensor
    sham_ancestor: Tensor
    metadata: dict

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> PersistenceTraceBankV1:
        names = ("source_nodes", "source_weights", "readout_left", "readout_right",
                 "readout_right_weight", "spatial", "sample", "genuine_ancestor", "sham_ancestor")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Sparse solve geometry supports FP32 or optional FP64 only")
        integers = {"source_nodes", "readout_left", "readout_right"}
        return replace(self, **{name: getattr(self, name).to(device=device,
            dtype=torch.long if name in integers else dtype) for name in names})


def _fixed(value: Tensor, dtype: torch.dtype, name: str) -> None:
    if value.device.type != "cpu" or value.dtype != dtype or value.requires_grad:
        raise ValueError(f"{name} must be fixed CPU {dtype} geometry")


def _tree(tree: MetricTreeHoldV1):
    for name in ("parent", "qualified_node_ids", "sample_nodes", "sample_ids"):
        _fixed(getattr(tree, name), torch.long, name)
    _fixed(tree.node_times, torch.float64, "node_times")
    parent, times = tree.parent, tree.node_times
    n = parent.numel()
    if (parent.ndim != 1 or n < 1 or times.shape != (n,) or tree.qualified_node_ids.shape != (n,)
            or not bool(torch.isfinite(times).all()) or bool((times < 0).any())
            or bool(((parent < -1) | (parent >= n)).any()) or int((parent == -1).sum()) != 1):
        raise ValueError("Require one complete connected dated tree, without isolated table rows")
    ids = tree.qualified_node_ids
    known = ids[ids >= 0]
    if bool((ids < -1).any()) or torch.unique(known).numel() != known.numel():
        raise ValueError("Qualified native identities must be unique within a tree")
    if known.numel() and (not isinstance(tree.identity_namespace, str) or not tree.identity_namespace.strip()):
        raise ValueError("Known native identities require a producer-qualified parent-wide namespace")
    if (tree.sample_nodes.ndim != 1 or tree.sample_ids.shape != tree.sample_nodes.shape
            or not tree.sample_nodes.numel() or bool((tree.sample_ids < 0).any())
            or bool(((tree.sample_nodes < 0) | (tree.sample_nodes >= n)).any())
            or torch.unique(tree.sample_nodes).numel() != tree.sample_nodes.numel()
            or torch.unique(tree.sample_ids).numel() != tree.sample_ids.numel()):
        raise ValueError("Require unique actual observed sample identities and local rows")
    children = (parent >= 0).nonzero().flatten()
    durations = times[parent[children]] - times[children]
    if bool((durations <= 0).any()):
        raise ValueError("Metric branches require strictly positive observed endpoint durations")
    # Positive branch durations rule out cycles. Every retained row must be on
    # an observed sample-to-root path, not an unused tree-sequence table row.
    retained = set()
    parents = parent.tolist()
    for sample in tree.sample_nodes.tolist():
        node = sample
        while node >= 0 and node not in retained:
            retained.add(node)
            node = parents[node]
    if len(retained) != n:
        raise ValueError("Inputs must include the complete sample-ancestor union only")
    samples = dict(zip(tree.sample_ids.tolist(), tree.sample_nodes.tolist()))
    sample_rows = set(tree.sample_nodes.tolist())
    edges = []
    for rank, child in enumerate(children.tolist()):
        ancestor = parents[child]
        qualified = int(ids[child]) >= 0 and int(ids[ancestor]) >= 0
        key = (tree.identity_namespace, int(ids[child]), int(ids[ancestor])) if qualified else None
        edges.append((child, ancestor, n + rank, float(durations[rank]),
                      float((times[child] + times[ancestor]) / 2), key,
                      (child in sample_rows, ancestor in sample_rows)))
    return samples, edges, n + len(edges)


def _normalized_laplacian(n: int, edges: list[tuple[int, int, float]]) -> Tensor:
    """Mass-lumped symmetric degree normalization; isolated diagonal is zero."""
    if not edges:
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long),
                                       torch.empty(0, dtype=torch.float64), (n, n)).coalesce()
    index = torch.tensor([(u, v) for u, v, _ in edges], dtype=torch.long).T
    weight = torch.tensor([w for _, _, w in edges], dtype=torch.float64)
    if not bool(torch.isfinite(weight).all()) or bool((weight <= 0).any()):
        raise ValueError("Conductances must be finite and positive")
    u, v = index
    degree = torch.zeros(n, dtype=torch.float64)
    degree.index_add_(0, u, weight)
    degree.index_add_(0, v, weight)
    if not bool(torch.isfinite(degree).all()):
        raise ValueError("Conductance degree overflow")
    off = -weight / torch.sqrt(degree[u]) / torch.sqrt(degree[v])
    active = (degree > 0).nonzero().flatten()
    indices = torch.cat((torch.stack((u, v)), torch.stack((v, u)), torch.stack((active, active))), 1)
    values = torch.cat((off, off, torch.ones(active.numel(), dtype=torch.float64)))
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def prepare_persistence_trace_bank_v1(
        trees: Sequence[MetricTreeHoldV1], intervals_bp: Tensor,
        source_sample_ids: Sequence[Tensor], readout_x_bp: Tensor, readout_sample_ids: Tensor,
        *, sample_identity_namespace: str, genomic_scale_bp: float = 2_000_000.,
        age_scale_generations: float = 10_000., sham_seed: int = 101,
        sham_age_bin_width_generations: float = 500.,
        sham_log_duration_bin_width: float = math.log(2.)) -> PersistenceTraceBankV1:
    """Prepare all geometry, trace controls and physical readout interpolation.

    source_sample_ids explicitly declares which actual samples have frozen
    emitter outputs at each hold center. No missing query/context truth is
    synthesized. The namespace declarations are caller provenance obligations.
    """
    if not trees or len(source_sample_ids) != len(trees):
        raise ValueError("Require a nonempty ordered tree collection and source roster per tree")
    _fixed(intervals_bp, torch.float64, "intervals_bp")
    _fixed(readout_x_bp, torch.float64, "readout_x_bp")
    _fixed(readout_sample_ids, torch.long, "readout_sample_ids")
    if not isinstance(sample_identity_namespace, str) or not sample_identity_namespace.strip():
        raise ValueError("Actual sample identities require a parent-wide namespace")
    scales = (genomic_scale_bp, age_scale_generations, sham_age_bin_width_generations,
              sham_log_duration_bin_width)
    if any(not math.isfinite(x) or x <= 0 for x in scales) or not isinstance(sham_seed, int):
        raise ValueError("Physical scales, sham bin widths and seed must be valid")
    g = len(trees)
    if (intervals_bp.shape != (g, 2) or not bool(torch.isfinite(intervals_bp).all())
            or bool((intervals_bp[:, 1] <= intervals_bp[:, 0]).any())
            or not torch.equal(intervals_bp[:-1, 1], intervals_bp[1:, 0])
            or readout_x_bp.ndim != 1 or not readout_x_bp.numel()
            or not bool(torch.isfinite(readout_x_bp).all())
            or bool(((readout_x_bp < intervals_bp[0, 0]) | (readout_x_bp >= intervals_bp[-1, 1])).any())
            or readout_sample_ids.ndim != 1 or not readout_sample_ids.numel()
            or bool((readout_sample_ids < 0).any())
            or torch.unique(readout_sample_ids).numel() != readout_sample_ids.numel()):
        raise ValueError("Require contiguous positive ordered intervals and valid physical query readouts")
    widths = (intervals_bp[:, 1] - intervals_bp[:, 0]) / genomic_scale_bp
    centers = intervals_bp.mean(1)
    layouts, offsets, source_nodes, source_weights = [], [0], [], []
    spatial_edges, sample_edges, genuine_edges, sham_edges = [], [], [], []
    query_rows = torch.full((g, readout_sample_ids.numel()), -1, dtype=torch.long)
    known_ages, known_sample_ages, known_sample_native = {}, {}, {}
    for i, tree in enumerate(trees):
        samples, branches, count = _tree(tree)
        offset = offsets[-1]
        # A qualified node is one parent-wide observed object with one age.
        for row, identity in enumerate(tree.qualified_node_ids.tolist()):
            if identity >= 0:
                key = (tree.identity_namespace, identity)
                age = float(tree.node_times[row])
                if key in known_ages and known_ages[key] != age:
                    raise ValueError("A qualified persistent native node has inconsistent observed ages")
                known_ages[key] = age
        for identity, row in samples.items():
            age = float(tree.node_times[row])
            if identity in known_sample_ages and known_sample_ages[identity] != age:
                raise ValueError("A persistent actual sample has inconsistent observed ages")
            known_sample_ages[identity] = age
            native = int(tree.qualified_node_ids[row])
            if native >= 0:
                key = (identity, tree.identity_namespace)
                if key in known_sample_native and known_sample_native[key] != native:
                    raise ValueError("Actual sample and qualified native identity joins disagree")
                known_sample_native[key] = native
        source_ids = source_sample_ids[i]
        _fixed(source_ids, torch.long, "source_sample_ids")
        if (source_ids.ndim != 1 or not source_ids.numel()
                or torch.unique(source_ids).numel() != source_ids.numel()
                or any(identity not in samples for identity in source_ids.tolist())):
            raise ValueError("Each hold requires explicit unique frozen-emitter sample sources")
        source_set = set(source_ids.tolist())
        source_nodes.extend(offset + samples[identity] for identity in source_ids.tolist())
        source_weights.extend([float(widths[i])] * source_ids.numel())
        for q, identity in enumerate(readout_sample_ids.tolist()):
            if identity in samples:
                if identity not in source_set:
                    raise ValueError("A present readout query requires its own frozen local source")
                query_rows[i, q] = offset + samples[identity]
        for child, ancestor, midpoint, duration, _, _, _ in branches:
            conductance = float(widths[i]) * 2 * age_scale_generations / duration
            spatial_edges.extend(((offset + child, offset + midpoint, conductance),
                                  (offset + midpoint, offset + ancestor, conductance)))
        layouts.append((samples, branches, offset))
        offsets.append(offset + count)
    rng = torch.Generator(device="cpu").manual_seed(sham_seed)
    qualified_matches = changed_sham_matches = 0
    bins_total = singleton_bins = 0

    def add_trace(output, left, right, lo, ro, distance):
        child, parent, mid, duration, _, _, roles = left
        rchild, rparent, rmid = right[:3]
        # Lumped P1 trace quadrature at endpoints/midpoint. Samples have their
        # own continuity term, so never receive a duplicate ancestral trace.
        weight = (duration / age_scale_generations) / distance
        output.append((lo + mid, ro + rmid, weight * 4 / 6))
        if not roles[0] and not right[6][0]:
            output.append((lo + child, ro + rchild, weight / 6))
        if not roles[1] and not right[6][1]:
            output.append((lo + parent, ro + rparent, weight / 6))

    for i in range(g - 1):
        ls, le, lo = layouts[i]
        rs, re, ro = layouts[i + 1]
        distance = float((centers[i + 1] - centers[i]) / genomic_scale_bp)
        for identity in sorted(ls.keys() & rs.keys()):
            sample_edges.append((lo + ls[identity], ro + rs[identity], 1 / distance))
        right_by_key = {edge[5]: edge for edge in re if edge[5] is not None}
        matches = [(edge, right_by_key[edge[5]]) for edge in le
                   if edge[5] is not None and edge[5] in right_by_key]
        bins = {}
        for rank, (left, right) in enumerate(matches):
            if left[3:5] != right[3:5]:
                raise ValueError("A matched branch has inconsistent observed endpoint ages")
            add_trace(genuine_edges, left, right, lo, ro, distance)
            bin_key = (math.floor(left[4] / sham_age_bin_width_generations),
                       math.floor(math.log(left[3]) / sham_log_duration_bin_width), left[6], right[6])
            bins.setdefault(bin_key, []).append(rank)
        qualified_matches += len(matches)
        for members in bins.values():
            bins_total += 1
            singleton_bins += len(members) == 1
            order = [members[j] for j in torch.randperm(len(members), generator=rng).tolist()]
            for j, left_rank in enumerate(order):
                right_rank = order[(j + 1) % len(order)]
                add_trace(sham_edges, matches[left_rank][0], matches[right_rank][1], lo, ro, distance)
                changed_sham_matches += left_rank != right_rank
    n = offsets[-1]
    holding = torch.searchsorted(intervals_bp[:, 0].contiguous(), readout_x_bp, right=True) - 1
    left = (torch.searchsorted(centers, readout_x_bp, right=True) - 1).clamp(0, g - 1)
    right = (left + 1).clamp(0, g - 1)
    same = left == right
    distance = torch.where(same, torch.ones_like(readout_x_bp), centers[right] - centers[left])
    fraction = ((readout_x_bp - centers[left]) / distance).clamp(0, 1)
    lrow, rrow, local = query_rows[left], query_rows[right], query_rows[holding]
    both = (lrow >= 0) & (rrow >= 0)
    if bool((local < 0).any()):
        raise ValueError("Every requested query must be observed in its containing tree")
    # Partial sample correspondence: absent neighbors do not invent a trajectory.
    lrow = torch.where(both, lrow, local)
    rrow = torch.where(both, rrow, local)
    fraction = torch.where(both, fraction[:, None], torch.zeros_like(local, dtype=torch.float64))
    metadata = dict(kind="finite_persistence_trace_kernel_v1", holds=g, states=n,
                    original_vertices=sum(tree.parent.numel() for tree in trees),
                    branch_midpoints=sum(len(layout[1]) for layout in layouts),
                    sample_identity_namespace=sample_identity_namespace,
                    qualified_branch_matches=qualified_matches,
                    qualified_ancestral_namespaces=sorted({tree.identity_namespace for tree in trees
                        if bool((tree.qualified_node_ids >= 0).any())}),
                    sample_continuity_traces=len(sample_edges),
                    genuine_ancestor_traces=len(genuine_edges), sham_ancestor_traces=len(sham_edges),
                    sham_changed_branch_matches=changed_sham_matches,
                    sham_bins=bins_total, sham_singleton_bins=singleton_bins,
                    sham_seed=sham_seed, sham_age_bin_width_generations=sham_age_bin_width_generations,
                    sham_log_duration_bin_width=sham_log_duration_bin_width,
                    genomic_scale_bp=genomic_scale_bp, age_scale_generations=age_scale_generations,
                    screening="identity on finite carrier coefficients",
                    component_normalization="symmetric weighted degree, separately per component",
                    trace_quadrature="lumped endpoint/midpoint weights 1/6,4/6,1/6; samples separate",
                    default_solve_dtype="float32", default_solve_rtol=1e-4,
                    continuum_exact=False, subdivision_exact=False)
    return PersistenceTraceBankV1(n, torch.tensor(source_nodes, dtype=torch.long),
        torch.tensor(source_weights, dtype=torch.float64), lrow, rrow, fraction,
        _normalized_laplacian(n, spatial_edges), _normalized_laplacian(n, sample_edges),
        _normalized_laplacian(n, genuine_edges), _normalized_laplacian(n, sham_edges), metadata).to("cpu")


def _action(x: Tensor, spatial_rate: Tensor, temporal_rate: Tensor,
            spatial: Tensor, temporal: Tensor) -> Tensor:
    return x + torch.sparse.mm(spatial, x) * spatial_rate + torch.sparse.mm(temporal, x) * temporal_rate


def _diagonal(matrix: Tensor) -> Tensor:
    index, value = matrix.indices(), matrix.values()
    mask = index[0] == index[1]
    diagonal = value.new_zeros(matrix.shape[0])
    diagonal.index_add_(0, index[0, mask], value[mask])
    return diagonal


def _pcg(rhs: Tensor, spatial_rate: Tensor, temporal_rate: Tensor, spatial: Tensor,
         temporal: Tensor, max_iterations: int, rtol: float, atol: float) -> Tensor:
    diagonal = 1 + _diagonal(spatial)[:, None] * spatial_rate + _diagonal(temporal)[:, None] * temporal_rate
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    tolerance = atol + rtol * torch.linalg.vector_norm(rhs, dim=0)
    z = residual / diagonal
    direction = z.clone()
    rz = (residual * z).sum(0)
    for _ in range(max_iterations):
        active = torch.linalg.vector_norm(residual, dim=0) > tolerance
        if not bool(active.any()):
            # Verify the actual equation, rather than only recursively updated r.
            residual = rhs - _action(x, spatial_rate, temporal_rate, spatial, temporal)
            active = torch.linalg.vector_norm(residual, dim=0) > tolerance
            if not bool(active.any()):
                return x
            z = residual / diagonal
            direction = z.clone()
            rz = (residual * z).sum(0)
        direction = direction * active
        image = _action(direction, spatial_rate, temporal_rate, spatial, temporal)
        denominator = (direction * image).sum(0)
        if bool(((denominator <= 0) & active).any()) or not bool(torch.isfinite(denominator).all()):
            raise RuntimeError("Screened PCG lost positive curvature")
        step = torch.where(active, rz / torch.where(active, denominator, torch.ones_like(denominator)), 0)
        x = x + direction * step
        residual = residual - image * step
        z = residual / diagonal
        new_rz = (residual * z).sum(0)
        ratio = torch.where(active, new_rz / torch.where(active, rz, torch.ones_like(rz)), 0)
        direction = z + direction * ratio
        rz = new_rz
    residual = rhs - _action(x, spatial_rate, temporal_rate, spatial, temporal)
    if not bool(torch.isfinite(x).all()) or bool((torch.linalg.vector_norm(residual, dim=0) > tolerance).any()):
        raise RuntimeError("Screened PCG did not meet the declared residual tolerance")
    return x


class _ImplicitScreenedSolveV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs, spatial_rate, temporal_rate, spatial, temporal, iterations, rtol, atol):
        x = _pcg(rhs, spatial_rate, temporal_rate, spatial, temporal, iterations, rtol, atol)
        ctx.save_for_backward(x, spatial_rate, temporal_rate, spatial, temporal)
        ctx.settings = (iterations, rtol, atol)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, sr, tr, spatial, temporal = ctx.saved_tensors
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            adjoint = _pcg(gradient, sr, tr, spatial, temporal, *ctx.settings)
            ds = -(adjoint * torch.sparse.mm(spatial, x)).sum(0)
            dt = -(adjoint * torch.sparse.mm(temporal, x)).sum(0)
        return adjoint, ds, dt, None, None, None, None, None


def solve_screened_v1(rhs: Tensor, spatial_rate: Tensor, temporal_rate: Tensor,
        spatial: Tensor, temporal: Tensor, *, max_iterations: int = 128,
        rtol: float | None = None, atol: float = 0.) -> Tensor:
    """FP32 default/optional FP64 first-order solve of I + s*Ls + t*Lt."""
    rtol = (1e-4 if rhs.dtype == torch.float32 else 1e-10) if rtol is None else rtol
    if (rhs.ndim != 2 or rhs.dtype not in (torch.float32, torch.float64) or spatial_rate.shape != (rhs.shape[1],)
            or temporal_rate.shape != spatial_rate.shape or not isinstance(max_iterations, int)
            or max_iterations < 1 or not math.isfinite(rtol) or rtol <= 0
            or not math.isfinite(atol) or atol < 0):
        raise ValueError("Require FP32/FP64 matrix RHS, per-channel rates and valid solve tolerances")
    for matrix in (spatial, temporal):
        if (matrix.layout != torch.sparse_coo or not matrix.is_coalesced()
                or matrix.shape != (rhs.shape[0], rhs.shape[0]) or matrix.requires_grad
                or matrix.dtype != rhs.dtype or matrix.device != rhs.device
                or not bool(torch.isfinite(matrix.values()).all())):
            raise ValueError("Require fixed coalesced sparse Laplacians in the RHS dtype/device")
    for rate in (spatial_rate, temporal_rate):
        if rate.device != rhs.device or rate.dtype != rhs.dtype or not bool(torch.isfinite(rate).all()) or bool((rate <= 0).any()):
            raise ValueError("Per-channel rates must be finite and strictly positive")
    if not bool(torch.isfinite(rhs).all()):
        raise ValueError("RHS must be finite")
    # Projection/readout may use AMP, but do not autocast the sparse equation.
    with torch.autocast(device_type=rhs.device.type, enabled=False):
        return _ImplicitScreenedSolveV1.apply(rhs, spatial_rate, temporal_rate, spatial, temporal,
                                            max_iterations, rtol, atol)


class PersistenceTraceKernelV1(nn.Module):
    """Small residual kernel over frozen strong-emitter unaries, without a loader."""
    def __init__(self, num_classes: int = 4, channels: int = 8, *, max_rate: float = 20.,
                 initial_spatial_rate: float = 1., initial_temporal_rate: float = 1.):
        super().__init__()
        self.min_rate = 1e-4
        if num_classes < 2 or channels not in (4, 8) or not math.isfinite(max_rate) or max_rate <= self.min_rate:
            raise ValueError("Require at least two classes, 4/8 channels and a positive finite rate bound")
        if any(not math.isfinite(rate) or not self.min_rate < rate < max_rate
               for rate in (initial_spatial_rate, initial_temporal_rate)):
            raise ValueError("Initial rates must lie strictly inside the rate bound")
        self.num_classes, self.channels, self.max_rate = num_classes, channels, max_rate
        self.source_projection = nn.Linear(num_classes, channels, bias=False)
        self.raw_spatial = nn.Parameter(torch.full((channels,), math.log((initial_spatial_rate - self.min_rate) / (max_rate - initial_spatial_rate))))
        self.raw_temporal = nn.Parameter(torch.full((channels,), math.log((initial_temporal_rate - self.min_rate) / (max_rate - initial_temporal_rate))))
        self.residual_output = nn.Linear(num_classes + channels, num_classes)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)

    def forward(self, bank: PersistenceTraceBankV1, source_logits: Tensor,
                readout_logits: Tensor, *, control: str = "genuine_ancestor") -> Tensor:
        if control not in {"sample_only", "genuine_ancestor", "sham_ancestor"}:
            raise ValueError("Unknown persistence trace control")
        weight = self.source_projection.weight
        if (source_logits.shape != (bank.source_nodes.numel(), self.num_classes)
                or readout_logits.shape != (*bank.readout_left.shape, self.num_classes)
                or source_logits.device != weight.device or readout_logits.device != weight.device
                or source_logits.dtype != weight.dtype or readout_logits.dtype != weight.dtype
                or bank.source_nodes.device != weight.device
                or not bool(torch.isfinite(source_logits).all()) or not bool(torch.isfinite(readout_logits).all())):
            raise ValueError("Require aligned finite frozen-emitter source and direct-skip readout logits")
        signal = self.source_projection(source_logits.detach()).tanh().to(bank.spatial.dtype)
        strength = bank.source_weights[:, None]
        rhs = signal.new_zeros((bank.num_states, 2 * self.channels))
        rhs.index_add_(0, bank.source_nodes,
                       torch.cat((signal * strength, torch.ones_like(signal) * strength), 1))
        temporal = bank.sample
        if control != "sample_only":
            temporal = (temporal + getattr(bank, control)).coalesce()
        spatial_rate = (self.min_rate + (self.max_rate - self.min_rate) * self.raw_spatial.sigmoid()).to(bank.spatial.dtype).repeat(2)
        temporal_rate = (self.min_rate + (self.max_rate - self.min_rate) * self.raw_temporal.sigmoid()).to(bank.spatial.dtype).repeat(2)
        state = solve_screened_v1(rhs, spatial_rate, temporal_rate, bank.spatial, temporal)
        fraction = bank.readout_right_weight[..., None]
        readout = state[bank.readout_left] * (1 - fraction) + state[bank.readout_right] * fraction
        numerator, mass = readout.split(self.channels, dim=-1)
        if not bool(torch.isfinite(readout).all()) or bool((mass <= 0).any()):
            raise RuntimeError("Matching ones-source mass must be finite and positive; no numerical floor is used")
        message = (numerator / mass).to(readout_logits.dtype)
        local = readout_logits.detach()
        return local + self.residual_output(torch.cat((local, message), -1))


def optional_fp64_solver_diagnostic_v1() -> dict:
    """Optional explicit FP64 diagnostic only, never called by preparation/fit."""
    spatial = _normalized_laplacian(3, [(0, 1, 2.), (1, 2, 3.)])
    temporal = _normalized_laplacian(3, [(0, 2, 0.7)])
    rhs = torch.tensor([[1., -0.2], [0.3, 0.8], [-0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    sr = torch.tensor([0.8, 1.2], dtype=torch.float64, requires_grad=True)
    tr = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    solved = solve_screened_v1(rhs, sr, tr, spatial, temporal)
    dense = torch.stack([torch.linalg.solve(torch.eye(3, dtype=torch.float64) + sr[k] * spatial.to_dense()
                                           + tr[k] * temporal.to_dense(), rhs[:, k]) for k in range(2)], 1)
    loss = solved.square().sum()
    gradients = torch.autograd.grad(loss, (rhs, sr, tr))
    epsilon = 1e-5
    numeric = []
    for k in range(2):
        delta = torch.zeros_like(sr); delta[k] = epsilon
        plus = solve_screened_v1(rhs.detach(), sr.detach() + delta, tr.detach(), spatial, temporal).square().sum()
        minus = solve_screened_v1(rhs.detach(), sr.detach() - delta, tr.detach(), spatial, temporal).square().sum()
        numeric.append((plus - minus) / (2 * epsilon))
    difference = float((solved - dense).abs().max())
    gradient_difference = float((gradients[1] - torch.stack(numeric)).abs().max())
    if difference > 1e-9 or gradient_difference > 1e-7 or not all(bool(torch.isfinite(x).all()) for x in gradients):
        raise AssertionError("Tiny screened-solve arithmetic sanity failed")
    return dict(dense_sparse_max_abs=difference, spatial_rate_finite_difference_max_abs=gradient_difference,
                all_first_derivatives_finite=True, states=3, channels=2, device="cpu")

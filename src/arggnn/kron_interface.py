"""Fixed CPU Kron boundary interfaces and optional coupled affine query flow.

Kron/Schur reduction is established. This experimental primitive makes no
novelty, ancestry-performance or full-tree dynamical projection claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields, replace
import math
from typing import Sequence

import torch
from torch import Tensor
from .models import GraphSequenceInput, validate_model_input


def _cpu_double(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.device.type != 'cpu':
        raise ValueError(f'{name} must be a CPU tensor')
    value = value.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')
    return value


def _indices(value: Tensor, size: int, name: str, *, nonempty: bool = True) -> Tensor:
    if not isinstance(value, Tensor) or value.device.type != 'cpu' or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f'{name} must be a CPU long vector')
    if nonempty and value.numel() == 0:
        raise ValueError(f'{name} must be nonempty')
    if bool(((value < 0) | (value >= size)).any()) or torch.unique(value).numel() != value.numel():
        raise ValueError(f'{name} must contain unique valid indices')
    return value


def canonical_resistance_edges(num_nodes: int, edge_index: Tensor, resistance: Tensor) -> tuple[Tensor, Tensor]:
    """Deduplicate bidirected representations of a simple undirected graph.

    A repeated edge must carry the same resistance. Parallel physical edges
    must be combined into their equivalent conductance by the caller.
    """
    if not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError('num_nodes must be positive')
    if not isinstance(edge_index, Tensor) or edge_index.device.type != 'cpu' or edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index must be CPU long [2,E]')
    resistance = _cpu_double(resistance, 'resistance')
    if resistance.shape != (edge_index.shape[1],) or bool((resistance <= 0).any()):
        raise ValueError('Every edge requires one strictly positive finite resistance')
    if bool(((edge_index < 0) | (edge_index >= num_nodes)).any()):
        raise ValueError('edge endpoint out of range')
    records: dict[tuple[int, int], float] = {}
    for (u, v), r in zip(edge_index.T.tolist(), resistance.tolist()):
        if u == v:
            raise ValueError('Self edges are not accepted')
        pair = (min(u, v), max(u, v))
        if pair in records and not math.isclose(records[pair], r, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError('Duplicate edge resistances disagree')
        records[pair] = r
    pairs = sorted(records)
    edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
    return edges, torch.tensor([records[pair] for pair in pairs], dtype=torch.float64)


@torch.no_grad()
def resistance_laplacian(num_nodes: int, edge_index: Tensor, resistance: Tensor) -> Tensor:
    """L = incidence * diag(1/resistance) * incidence.T, in CPU float64."""
    edges, resistance = canonical_resistance_edges(num_nodes, edge_index, resistance)
    conductance = resistance.reciprocal()
    if not bool(torch.isfinite(conductance).all()):
        raise ValueError('Resistance reciprocal overflows float64')
    laplacian = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    if edges.numel():
        u, v = edges
        laplacian.index_put_((u, u), conductance, accumulate=True)
        laplacian.index_put_((v, v), conductance, accumulate=True)
        laplacian.index_put_((u, v), -conductance, accumulate=True)
        laplacian.index_put_((v, u), -conductance, accumulate=True)
    if not bool(torch.isfinite(laplacian).all()):
        raise ValueError('Accumulated conductance overflows float64')
    return laplacian


@torch.no_grad()
def observed_branch_laplacian(node_times: Tensor, edge_index: Tensor, *, resistance_scale: float = 1.0) -> Tensor:
    """Use a fixed positive scale times unencoded observed branch duration.

    This function does not add epsilon to zero branches. Exact zero branches
    require a separate explicit contraction policy, which is not implemented.
    """
    times = _cpu_double(node_times, 'node_times')
    if times.ndim != 1 or times.numel() == 0 or bool((times < 0).any()):
        raise ValueError('node_times must be a nonnegative CPU vector')
    if not math.isfinite(resistance_scale) or resistance_scale <= 0:
        raise ValueError('resistance_scale must be finite and positive')
    if not isinstance(edge_index, Tensor) or edge_index.dtype != torch.long or edge_index.device.type != 'cpu' or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index must be CPU long [2,E]')
    if bool(((edge_index < 0) | (edge_index >= times.numel())).any()):
        raise ValueError('edge endpoint out of range')
    duration = (times[edge_index[0]] - times[edge_index[1]]).abs()
    return resistance_laplacian(times.numel(), edge_index, resistance_scale * duration)


@dataclass(frozen=True)
class KronDiagnostics:
    interior_condition_number: float
    relative_solve_residual: float
    response_row_sum_relative_to_source: float
    response_row_sum_relative_to_response: float
    response_symmetry_relative_to_source: float
    minimum_response_eigenvalue: float
    maximum_response_offdiagonal: float


@dataclass(frozen=True)
class KronReduction:
    response: Tensor             # [B,B], in declared boundary order
    boundary_nodes: Tensor       # [B], original node indices
    interior_nodes: Tensor       # [I], original node indices
    interior_extension: Tensor   # [I,B], -solve(L_II,L_IB)
    num_nodes: int
    diagnostics: KronDiagnostics

    def boundary_current(self, boundary_values: Tensor) -> Tensor:
        values = _cpu_double(boundary_values, 'boundary_values')
        if values.ndim not in (1, 2) or values.shape[0] != self.boundary_nodes.numel():
            raise ValueError('boundary_values must be [B] or [B,D]')
        return self.response @ values

    def harmonic_lift(self, boundary_values: Tensor) -> Tensor:
        values = _cpu_double(boundary_values, 'boundary_values')
        if values.ndim not in (1, 2) or values.shape[0] != self.boundary_nodes.numel():
            raise ValueError('boundary_values must be [B] or [B,D]')
        lifted = values.new_zeros((self.num_nodes, *values.shape[1:]))
        lifted[self.boundary_nodes] = values
        lifted[self.interior_nodes] = self.interior_extension @ values
        return lifted


@torch.no_grad()
def kron_reduce(laplacian: Tensor, boundary_nodes: Tensor, *, max_condition: float = 1e12,
                relative_tolerance: float = 1e-9) -> KronReduction:
    """Exact static boundary response by Schur complement, without an inverse.

    Every connected component containing interior nodes must touch a boundary.
    Unanchored components and ill-conditioned interior solves are rejected.
    No diagonal jitter, spectral clipping or silent regularization is applied.
    """
    laplacian = _cpu_double(laplacian, 'laplacian')
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1] or laplacian.shape[0] == 0:
        raise ValueError('laplacian must be nonempty square')
    if not math.isfinite(max_condition) or max_condition < 1 or not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError('Require finite max_condition >= 1 and positive relative_tolerance')
    n = laplacian.shape[0]
    boundary = _indices(boundary_nodes, n, 'boundary_nodes').clone()
    eps = torch.finfo(torch.float64).eps
    source_scale = max(float(torch.linalg.matrix_norm(laplacian, ord=float('inf'))), torch.finfo(torch.float64).tiny)
    tolerance = relative_tolerance * source_scale
    if float((laplacian - laplacian.T).abs().max()) > tolerance or float(laplacian.sum(dim=1).abs().max()) > tolerance:
        raise ValueError('Require a symmetric zero-row-sum Laplacian')
    offdiagonal = laplacian - torch.diag_embed(laplacian.diagonal())
    if float(offdiagonal.max()) > tolerance or float(laplacian.diagonal().min()) < -tolerance:
        raise ValueError('Require nonnegative conductances')
    keep = torch.ones(n, dtype=torch.bool)
    keep[boundary] = False
    interior = torch.nonzero(keep, as_tuple=False).flatten()
    lbb = laplacian[boundary[:, None], boundary[None, :]]
    condition, residual = 1.0, 0.0
    if interior.numel():
        lii = laplacian[interior[:, None], interior[None, :]]
        lib = laplacian[interior[:, None], boundary[None, :]]
        eigenvalues = torch.linalg.eigvalsh(lii)
        if float(eigenvalues.min()) <= 0:
            raise ValueError('Interior Laplacian is not positive definite. Check unanchored components or numerical conditioning')
        condition = float(eigenvalues.max() / eigenvalues.min())
        if not math.isfinite(condition) or condition > max_condition:
            raise ValueError(f'Interior condition number {condition:.6g} exceeds max_condition={max_condition:.6g}')
        factor, info = torch.linalg.cholesky_ex(lii)
        if bool((info != 0).any()):
            raise ValueError('Interior Cholesky factorization failed')
        solution = torch.cholesky_solve(lib, factor)
        denominator = float(torch.linalg.matrix_norm(lii) * torch.linalg.matrix_norm(solution) + torch.linalg.matrix_norm(lib))
        residual = float(torch.linalg.matrix_norm(lii @ solution - lib)) / max(denominator, torch.finfo(torch.float64).tiny)
        extension = -solution
        response_raw = lbb - lib.T @ solution
        symmetry = float((response_raw - response_raw.T).abs().max()) / source_scale
        response = (response_raw + response_raw.T) * 0.5
    else:
        extension = laplacian.new_empty((0, boundary.numel()))
        response = lbb.clone()
        symmetry = 0.0
    row_error = float(response.sum(dim=1).abs().max())
    response_scale = max(float(torch.linalg.matrix_norm(response, ord=float('inf'))), eps * source_scale, torch.finfo(torch.float64).tiny)
    response_offdiagonal = response - torch.diag_embed(response.diagonal())
    minimum_eigenvalue = float(torch.linalg.eigvalsh(response).min())
    maximum_offdiagonal = float(response_offdiagonal.max())
    if residual > relative_tolerance or symmetry > relative_tolerance or row_error > tolerance or minimum_eigenvalue < -tolerance or maximum_offdiagonal > tolerance:
        raise ValueError('Kron response failed backward-error/PSD/conductance checks')
    diagnostics = KronDiagnostics(condition, residual, row_error / source_scale,
        row_error / response_scale, symmetry, minimum_eigenvalue, maximum_offdiagonal)
    return KronReduction(response, boundary, interior, extension, n, diagnostics)


@dataclass(frozen=True)
class QueryInterfaceCache:
    normalized_response: Tensor   # [G,B,B], normalization over full boundary
    normalization_scale: Tensor   # [G], raw C / scale = normalized C
    cqq: Tensor                   # [G,Q,Q], original query order
    cqr: Tensor                   # [G,Q,R], declared reference order
    eigenvalues: Tensor           # [G,Q], CPU float64 before optional transfer
    eigenvectors: Tensor          # [G,Q,Q]
    query_indices: Tensor         # [Q], into sorted sample_ids
    reference_indices: Tensor     # [R], into sorted sample_ids
    query_ids: Tensor             # [Q], must equal bank.query_node_ids
    reference_ids: Tensor         # [R], explicit observed boundary identities
    interval_weights: Tensor      # [G], widths / total covered width
    bag_cqq: Tensor               # [Q,Q], width-weighted normalized average
    bag_eigenvalues: Tensor       # [Q]
    bag_eigenvectors: Tensor      # [Q,Q]
    normalization: str = 'mean_boundary_diagonal'
    schema_version: int = 1

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> QueryInterfaceCache:
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device=device, dtype=dtype if value.is_floating_point() else None) if isinstance(value, Tensor) else value
        return QueryInterfaceCache(**values)

    def as_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: dict) -> QueryInterfaceCache:
        if values.get('schema_version') != 1 or set(values) != {field.name for field in fields(cls)}:
            raise ValueError('QueryInterfaceCache schema/fields mismatch')
        return cls(**values)


@dataclass(frozen=True)
class LeafInterfaceBank:
    response: Tensor              # [G,B,B], raw float64 Kron response
    sample_ids: Tensor            # [B], sorted persistent sample identities
    boundary_nodes: Tensor        # [G,B], packed original node indices
    leaf_x: Tensor                # [G,B,F_selected], branch excluded for trees
    feature_columns: tuple[int, ...]
    intervals: Tensor             # [G,2], original coordinate dtype/order
    query_node_ids: Tensor        # [Q], original requested query order
    positions: Tensor             # [P], original coordinate dtype/order
    target_graph: Tensor          # [P], original graph serialization mapping
    window_bounds: Tensor | None  # [2], original coordinate dtype
    diagnostics: tuple[KronDiagnostics, ...]
    resistance_scale: float | None
    time_scale: float | None
    geometry_kind: str = 'observed_tree_leaf'
    query_cache: QueryInterfaceCache | None = None
    schema_version: int = 1

    @property
    def features(self) -> Tensor:
        """Audit compatibility alias. Neural consumers should use leaf_x."""
        return self.leaf_x

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> LeafInterfaceBank:
        # Form genomic differences in original coordinate precision, then cast
        # durations in the consumer. Feature-dtype transfer never casts coords.
        coordinates = {'intervals', 'positions', 'window_bounds'}
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                selected_dtype = dtype if value.is_floating_point() and field.name not in coordinates else None
                values[field.name] = value.to(device=device, dtype=selected_dtype)
            elif field.name == 'query_cache' and value is not None:
                values[field.name] = value.to(device, dtype)
            else:
                values[field.name] = value
        return LeafInterfaceBank(**values)

    def as_dict(self) -> dict:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values['diagnostics'] = [asdict(diagnostic) for diagnostic in self.diagnostics]
        values['query_cache'] = None if self.query_cache is None else self.query_cache.as_dict()
        return values

    @classmethod
    def from_dict(cls, values: dict) -> LeafInterfaceBank:
        if values.get('schema_version') != 1 or set(values) != {field.name for field in fields(cls)}:
            raise ValueError('LeafInterfaceBank schema/fields mismatch')
        values = dict(values)
        values['feature_columns'] = tuple(values['feature_columns'])
        values['diagnostics'] = tuple(KronDiagnostics(**record) for record in values['diagnostics'])
        values['query_cache'] = None if values['query_cache'] is None else QueryInterfaceCache.from_dict(values['query_cache'])
        return cls(**values)


def _interface_bank(seq: GraphSequenceInput, responses: list[Tensor], sample_ids: Tensor,
        packed_boundaries: list[Tensor], leaf_features: list[Tensor], columns: tuple[int, ...],
        diagnostics: list[KronDiagnostics], *, resistance_scale: float | None,
        time_scale: float | None, geometry_kind: str) -> LeafInterfaceBank:
    bounds = getattr(seq, 'window_bounds', None)
    if not bool(torch.isin(seq.query_node_ids, sample_ids).all()):
        raise ValueError('Every requested query must be in the boundary sample set')
    return LeafInterfaceBank(response=torch.stack(responses), sample_ids=sample_ids,
        boundary_nodes=torch.stack(packed_boundaries), leaf_x=torch.stack(leaf_features),
        feature_columns=columns, intervals=seq.intervals.detach().clone(),
        query_node_ids=seq.query_node_ids.detach().clone(), positions=seq.positions.detach().clone(),
        target_graph=seq.target_graph.detach().clone(), window_bounds=None if bounds is None else bounds.detach().clone(),
        diagnostics=tuple(diagnostics), resistance_scale=resistance_scale,
        time_scale=time_scale, geometry_kind=geometry_kind)


@torch.no_grad()
def precompute_leaf_interfaces(seq: GraphSequenceInput, *, time_column: int,
        parent_branch_column: int, time_scale: float = 10000.0, resistance_scale: float = 1.0,
        feature_columns: Sequence[int] | None = None, max_condition: float = 1e12,
        validate: bool = True) -> LeafInterfaceBank:
    """Reduce each observed tree onto all persistent sample leaves on CPU.

    Times are decoded from log1p(generations)/log1p(time_scale), never from
    sums of log-encoded branch lengths. The declared parent-branch input is
    excluded. The caller must audit that every other selected feature itself
    is invariant to inserting non-sample unary ancestors.
    """
    if seq.x.device.type != 'cpu':
        raise ValueError('Precompute leaf interfaces on CPU')
    if validate:
        validate_model_input(seq)
    f = seq.x.shape[1]
    if not 0 <= time_column < f or not 0 <= parent_branch_column < f or time_column == parent_branch_column:
        raise ValueError('Declare distinct valid observed time and parent-branch columns')
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError('time_scale must be finite and positive')
    columns = tuple(c for c in range(f) if c != parent_branch_column) if feature_columns is None else tuple(int(c) for c in feature_columns)
    if not columns or len(set(columns)) != len(columns) or any(c < 0 or c >= f for c in columns) or parent_branch_column in columns:
        raise ValueError('Select unique valid leaf features and exclude the raw parent-branch column')
    encoded_times = _cpu_double(seq.x[:, time_column], 'encoded node times')
    if bool((encoded_times < 0).any()):
        raise ValueError('Encoded node times must be nonnegative')
    times = torch.expm1(encoded_times * math.log1p(time_scale))
    if not bool(torch.isfinite(times).all()):
        raise ValueError('Decoded observed ages overflow')
    g = seq.intervals.shape[0]
    groups = [[] for _ in range(g)]
    edge_groups = [[] for _ in range(g)]
    graph_ids = seq.node_graph.tolist()
    for node, graph in enumerate(graph_ids):
        groups[graph].append(node)
    unique_edges = {tuple(sorted(pair)) for pair in seq.edge_index.T.tolist()}
    for u, v in sorted(unique_edges):
        if graph_ids[u] != graph_ids[v]:
            raise ValueError('Tree edge crosses graph boundaries')
        edge_groups[graph_ids[u]].append((u, v))
    responses, features, packed_boundaries, diagnostics = [], [], [], []
    sample_ids = None
    for graph, packed_nodes in enumerate(groups):
        nodes = torch.tensor(packed_nodes, dtype=torch.long)
        index = {node: i for i, node in enumerate(packed_nodes)}
        edges = torch.tensor([(index[u], index[v]) for u, v in edge_groups[graph]], dtype=torch.long).reshape(-1, 2).T.contiguous()
        n = nodes.numel()
        if edges.shape[1] != n - 1:
            raise ValueError('Observed interfaces require simple acyclic connected trees')
        adjacency = [[] for _ in range(n)]
        for u, v in edges.T.tolist():
            if u == v:
                raise ValueError('Observed tree self edge')
            adjacency[u].append(v)
            adjacency[v].append(u)
        seen, stack = set(), [0]
        while stack:
            u = stack.pop()
            if u not in seen:
                seen.add(u)
                stack.extend(adjacency[u])
        if len(seen) != n:
            raise ValueError('Observed tree is disconnected')
        local_times = times[nodes]
        older_neighbors = [sum(bool(local_times[v] > local_times[u]) for v in adjacency[u]) for u in range(n)]
        if sum(count == 0 for count in older_neighbors) != 1 or any(count > 1 for count in older_neighbors):
            raise ValueError('Observed ages must orient a tree with one older parent per nonroot node')
        ids = seq.node_ids[nodes]
        boundary = torch.nonzero(ids >= 0, as_tuple=False).flatten()
        if boundary.numel() == 0 or any(len(adjacency[u]) > 1 for u in boundary.tolist()):
            raise ValueError('Persistent sample nodes must be observed tree leaves')
        if torch.unique(ids[boundary]).numel() != boundary.numel():
            raise ValueError('Duplicate persistent sample identity')
        boundary = boundary[torch.argsort(ids[boundary])]
        this_ids = ids[boundary]
        if sample_ids is None:
            sample_ids = this_ids.clone()
        elif not torch.equal(sample_ids, this_ids):
            raise ValueError('The persistent boundary sample set must match across trees')
        laplacian = observed_branch_laplacian(times[nodes], edges, resistance_scale=resistance_scale)
        reduction = kron_reduce(laplacian, boundary, max_condition=max_condition)
        packed_boundary = nodes[boundary]
        responses.append(reduction.response)
        features.append(seq.x[packed_boundary][:, columns].detach().double())
        packed_boundaries.append(packed_boundary)
        diagnostics.append(reduction.diagnostics)
    return _interface_bank(seq, responses, sample_ids, packed_boundaries, features, columns,
        diagnostics, resistance_scale=float(resistance_scale), time_scale=float(time_scale),
        geometry_kind='observed_tree_leaf')


@torch.no_grad()
def precompute_graph_interfaces(seq: GraphSequenceInput, edge_resistance: Tensor, *,
        boundary_ids: Tensor | None = None, feature_columns: Sequence[int] | None = None,
        max_condition: float = 1e12, validate: bool = True) -> LeafInterfaceBank:
    """Generic graph wrapper, with all persistent nodes as default boundary.

    This has no tree/time/branch-column assumption. Geography can declare every
    node persistent, obtaining C=L. Genealogy unary guarantees require the
    dedicated tree wrapper's observed resistance and feature exclusions.
    """
    if seq.x.device.type != 'cpu':
        raise ValueError('Precompute graph interfaces on CPU')
    if validate:
        validate_model_input(seq)
    resistance = _cpu_double(edge_resistance, 'edge_resistance')
    if resistance.shape != (seq.edge_index.shape[1],) or bool((resistance <= 0).any()):
        raise ValueError('Provide a positive resistance for every serialized edge')
    f = seq.x.shape[1]
    columns = tuple(range(f)) if feature_columns is None else tuple(int(c) for c in feature_columns)
    if not columns or len(set(columns)) != len(columns) or any(c < 0 or c >= f for c in columns):
        raise ValueError('Select unique valid observed boundary feature columns')
    if boundary_ids is not None:
        if boundary_ids.device.type != 'cpu' or boundary_ids.dtype != torch.long or boundary_ids.ndim != 1 or boundary_ids.numel() == 0 or bool((boundary_ids < 0).any()) or torch.unique(boundary_ids).numel() != boundary_ids.numel():
            raise ValueError('boundary_ids must be unique nonnegative CPU long identities')
        boundary_ids = boundary_ids.sort().values
    g = seq.intervals.shape[0]
    groups = [[] for _ in range(g)]
    edge_groups = [[] for _ in range(g)]
    graph_ids = seq.node_graph.tolist()
    for node, graph in enumerate(graph_ids):
        groups[graph].append(node)
    for edge, (u, v) in enumerate(seq.edge_index.T.tolist()):
        if graph_ids[u] != graph_ids[v]:
            raise ValueError('Graph edge crosses graph boundaries')
        edge_groups[graph_ids[u]].append((u, v, float(resistance[edge])))
    responses, features, packed_boundaries, diagnostics = [], [], [], []
    sample_ids = None
    for graph, packed_nodes in enumerate(groups):
        nodes = torch.tensor(packed_nodes, dtype=torch.long)
        lookup = {node: i for i, node in enumerate(packed_nodes)}
        records = edge_groups[graph]
        edges = torch.tensor([(lookup[u], lookup[v]) for u, v, _ in records], dtype=torch.long).reshape(-1, 2).T.contiguous()
        weights = torch.tensor([r for _, _, r in records], dtype=torch.float64)
        ids = seq.node_ids[nodes]
        boundary = torch.nonzero(ids >= 0 if boundary_ids is None else torch.isin(ids, boundary_ids), as_tuple=False).flatten()
        if boundary.numel() == 0 or torch.unique(ids[boundary]).numel() != boundary.numel():
            raise ValueError('Require unique persistent boundary nodes in every graph')
        boundary = boundary[torch.argsort(ids[boundary])]
        this_ids = ids[boundary]
        if boundary_ids is not None and not torch.equal(this_ids, boundary_ids):
            raise ValueError('A declared boundary identity is absent from a graph')
        if sample_ids is None:
            sample_ids = this_ids.clone()
        elif not torch.equal(sample_ids, this_ids):
            raise ValueError('The persistent boundary sample set must match across graphs')
        reduction = kron_reduce(resistance_laplacian(nodes.numel(), edges, weights), boundary, max_condition=max_condition)
        packed_boundary = nodes[boundary]
        responses.append(reduction.response)
        features.append(seq.x[packed_boundary][:, columns].detach().double())
        packed_boundaries.append(packed_boundary)
        diagnostics.append(reduction.diagnostics)
    return _interface_bank(seq, responses, sample_ids, packed_boundaries, features, columns,
        diagnostics, resistance_scale=None, time_scale=None, geometry_kind='generic_graph_boundary')


@dataclass(frozen=True)
class QueryReferencePartition:
    cqq: Tensor                   # [...,Q,Q]
    cqr: Tensor                   # [...,Q,R]
    query_indices: Tensor         # [Q], in requested ID order
    reference_indices: Tensor     # [R], in requested ID order


def query_reference_partition(response: Tensor, sample_ids: Tensor, query_ids: Tensor,
                              reference_ids: Tensor) -> QueryReferencePartition:
    """Require an exhaustive disjoint boundary partition. No silent omission."""
    response = _cpu_double(response, 'response')
    if response.ndim < 2 or response.shape[-1] != response.shape[-2]:
        raise ValueError('response must be [...,B,B]')
    b = response.shape[-1]
    for name, ids in [('sample_ids', sample_ids), ('query_ids', query_ids), ('reference_ids', reference_ids)]:
        if not isinstance(ids, Tensor) or ids.device.type != 'cpu' or ids.dtype != torch.long or ids.ndim != 1 or bool((ids < 0).any()) or torch.unique(ids).numel() != ids.numel():
            raise ValueError(f'{name} must contain unique nonnegative CPU long identities')
    if sample_ids.numel() != b or query_ids.numel() == 0:
        raise ValueError('Require a matching boundary ID vector and at least one query')
    selected = torch.cat((query_ids, reference_ids))
    if selected.numel() != b or torch.unique(selected).numel() != b or not torch.equal(selected.sort().values, sample_ids.sort().values):
        raise ValueError('Query/reference IDs must be an exhaustive disjoint boundary partition')
    lookup = {sample: i for i, sample in enumerate(sample_ids.tolist())}
    q = torch.tensor([lookup[s] for s in query_ids.tolist()], dtype=torch.long)
    r = torch.tensor([lookup[s] for s in reference_ids.tolist()], dtype=torch.long)
    return QueryReferencePartition(response[..., q[:, None], q[None, :]], response[..., q[:, None], r[None, :]], q, r)


@torch.no_grad()
def with_query_reference_cache(bank: LeafInterfaceBank, reference_ids: Tensor, *,
        query_ids: Tensor | None = None, normalization: str = 'mean_boundary_diagonal') -> LeafInterfaceBank:
    """Cache normalized Q/R blocks and exact fixed eigensystems on CPU.

    Full-boundary mean diagonal is a permutation/subdivision-invariant scalar.
    The homogeneous bag cache averages Cqq by width/total width. Its eigensystem
    assumes channel rates/damping remain global across the tile.
    """
    response = _cpu_double(bank.response, 'bank.response')
    query_ids = bank.query_node_ids if query_ids is None else query_ids
    if not torch.equal(query_ids, bank.query_node_ids):
        raise ValueError('Cached query order must match bank.query_node_ids')
    if normalization == 'mean_boundary_diagonal':
        scale = response.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        zero = response.abs().amax(dim=(-2, -1)) == 0
        scale = torch.where(zero, torch.ones_like(scale), scale)
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError('Nonzero response requires a positive finite mean boundary diagonal')
    elif normalization == 'none':
        scale = response.new_ones(response.shape[0])
    else:
        raise ValueError('normalization must be mean_boundary_diagonal or none')
    normalized = response / scale[:, None, None]
    partition = query_reference_partition(normalized, bank.sample_ids, query_ids, reference_ids)
    eigenvalues, eigenvectors = torch.linalg.eigh(partition.cqq)
    widths = (bank.intervals[:, 1] - bank.intervals[:, 0]).double()
    if not bool(torch.isfinite(widths).all()) or bool((widths <= 0).any()):
        raise ValueError('Cache intervals require positive finite widths')
    interval_weights = widths / widths.sum()
    bag_cqq = (partition.cqq * interval_weights[:, None, None]).sum(dim=0)
    bag_eigenvalues, bag_eigenvectors = torch.linalg.eigh(bag_cqq)
    cache = QueryInterfaceCache(normalized, scale, partition.cqq, partition.cqr,
        eigenvalues, eigenvectors, partition.query_indices, partition.reference_indices,
        query_ids.detach().clone(), reference_ids.detach().clone(), interval_weights,
        bag_cqq, bag_eigenvalues, bag_eigenvectors, normalization)
    return replace(bank, query_cache=cache)


def coupled_query_dynamics(cqq: Tensor, cqr: Tensor, reference_bank: Tensor, *,
        coupling_rate: float | Tensor = 1.0, damping: float | Tensor = 1e-3,
        learned_query_forcing: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """dZ/dx = A Z + B, with query-coupled state Z[...,Q,D].

    A = -lambda*Cqq-damping*I, B = -lambda*Cqr*reference_bank+forcing.
    Fixed Kron matrices are detached. Provided forcing/rate tensors may retain
    gradients, but this utility contains no encoder or trainable parameters.
    """
    cqq = _cpu_double(cqq, 'cqq')
    cqr = _cpu_double(cqr, 'cqr')
    if cqq.ndim < 2 or cqq.shape[-1] != cqq.shape[-2] or cqr.ndim < 2 or cqr.shape[-2] != cqq.shape[-1]:
        raise ValueError('Require cqq[...,Q,Q] and cqr[...,Q,R]')
    if not isinstance(reference_bank, Tensor) or reference_bank.device.type != 'cpu' or not reference_bank.is_floating_point() or reference_bank.ndim < 2 or reference_bank.shape[-2] != cqr.shape[-1] or reference_bank.shape[-1] < 1:
        raise ValueError('reference_bank must be CPU floating [...,R,D]')
    reference_bank = reference_bank.to(dtype=cqq.dtype)
    if not bool(torch.isfinite(reference_bank).all()):
        raise ValueError('reference_bank must be finite')
    rate = torch.as_tensor(coupling_rate, dtype=cqq.dtype, device='cpu')
    damp = torch.as_tensor(damping, dtype=cqq.dtype, device='cpu')
    if rate.numel() != 1 or damp.numel() != 1 or not bool(torch.isfinite(rate).all()) or not bool(torch.isfinite(damp).all()) or not bool((rate > 0).all()) or not bool((damp > 0).all()):
        raise ValueError('coupling_rate and damping must be finite positive scalars')
    rate, damp = rate.reshape(()), damp.reshape(())
    scale = max(float(cqq.abs().max()), torch.finfo(torch.float64).tiny)
    if float((cqq - cqq.transpose(-1, -2)).abs().max()) > 1e-9 * scale or float(torch.linalg.eigvalsh(cqq).min()) < -1e-9 * scale:
        raise ValueError('cqq must be symmetric positive semidefinite')
    a = -rate * cqq - damp * torch.eye(cqq.shape[-1], dtype=cqq.dtype)
    forcing = -rate * (cqr @ reference_bank)
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(forcing).all()):
        raise ValueError('Rate-scaled coupled dynamics overflow')
    if float(torch.linalg.eigvalsh(a.detach()).max()) >= 0:
        raise ValueError('Chosen rate/damping is not numerically dissipative. Check response precision and damping')
    if learned_query_forcing is not None:
        if not isinstance(learned_query_forcing, Tensor) or learned_query_forcing.device.type != 'cpu' or not learned_query_forcing.is_floating_point() or learned_query_forcing.shape != forcing.shape or not bool(torch.isfinite(learned_query_forcing).all()):
            raise ValueError('learned_query_forcing must be finite CPU floating with the resulting [...,Q,D] shape')
        forcing = forcing + learned_query_forcing.to(dtype=forcing.dtype)
        if not bool(torch.isfinite(forcing).all()):
            raise ValueError('Combined forcing overflows')
    return a, forcing


@dataclass(frozen=True)
class CoupledTransition:
    homogeneous: Tensor          # [...,Q,Q]
    offset: Tensor               # [...,Q,D]

    def apply(self, state: Tensor) -> Tensor:
        if state.device.type != 'cpu' or not state.is_floating_point() or state.shape[-2:] != self.offset.shape[-2:] or not bool(torch.isfinite(state).all()):
            raise ValueError('state must be finite CPU floating [...,Q,D] matching the forcing channels')
        return self.homogeneous @ state.to(dtype=self.homogeneous.dtype) + self.offset


def coupled_affine_transition(a: Tensor, forcing: Tensor, duration: Tensor | float) -> CoupledTransition:
    """Exact constant-generator affine flow, using a [Q+D,Q+D] block exp.

    For matrix state Z, augment [Z;I_D]. Each state channel has the same
    query-coupling matrix, with its own constant forcing column.
    """
    if a.device.type != 'cpu' or forcing.device.type != 'cpu' or a.ndim < 2 or forcing.ndim < 2 or a.shape[-1] != a.shape[-2] or forcing.shape[-2] != a.shape[-1] or forcing.shape[-1] < 1:
        raise ValueError('Require CPU a[...,Q,Q] and forcing[...,Q,D]')
    if not a.is_floating_point() or not forcing.is_floating_point() or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(forcing).all()):
        raise ValueError('a and forcing must be finite floating tensors')
    dt = torch.as_tensor(duration, dtype=a.dtype, device='cpu')
    if not bool(torch.isfinite(dt).all()) or bool((dt < 0).any()):
        raise ValueError('duration must be finite and nonnegative')
    batch = torch.broadcast_shapes(a.shape[:-2], forcing.shape[:-2], dt.shape)
    q, d = forcing.shape[-2:]
    a = a.expand(*batch, q, q)
    forcing = forcing.to(dtype=a.dtype).expand(*batch, q, d)
    upper = torch.cat((a, forcing), dim=-1)
    lower = a.new_zeros((*batch, d, q + d))
    generator = torch.cat((upper, lower), dim=-2)
    flow = torch.matrix_exp(generator * dt.expand(batch)[..., None, None])
    return CoupledTransition(flow[..., :q, :q], flow[..., :q, q:])

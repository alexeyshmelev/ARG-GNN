"""Offline observed age-resistance operators sharing the plain leaf interface.

This fixed CPU bank is a geometry experiment, not a predictive claim. All
eigensystems belong to observed fixed matrices. Neural consumers must keep the
ordinary plain local bank and may not cache learned matrices here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Sequence

import torch
from torch import Tensor

from .models import GraphSequenceInput
from .kron_interface import (
    KronDiagnostics, LeafInterfaceBank, _cpu_double, kron_reduce,
    precompute_leaf_interfaces, query_reference_partition,
    resistance_laplacian, with_query_reference_cache,
)


def _parameters(age_scale: float, strength: float) -> tuple[float, float]:
    scale, amount = float(age_scale), float(strength)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('age_scale must be finite and positive')
    if not math.isfinite(amount) or amount < 0 or not math.isfinite(1 + amount):
        raise ValueError('strength must be finite and nonnegative')
    return scale, amount


def _ages(value: Tensor, name: str) -> Tensor:
    value = _cpu_double(value, name)
    if bool((value < 0).any()):
        raise ValueError(f'{name} must be nonnegative')
    return value


@torch.no_grad()
def age_potential(node_times: Tensor, age_scale: float, *, strength: float = 3.) -> Tensor:
    """Phi_s(t)=(1+a)t-a*s*atan(t/s), on decoded physical observed ages."""
    scale, amount = _parameters(age_scale, strength)
    times = _ages(node_times, 'node_times')
    result = (1 + amount) * times - amount * scale * torch.atan(times / scale)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Age potential overflows float64')
    return result


@torch.no_grad()
def age_potential_difference(parent_times: Tensor, child_times: Tensor,
                             age_scale: float, *, strength: float = 3.) -> Tensor:
    """Stable endpoint-potential difference, without numerical quadrature.

    atan(p/s)-atan(c/s)=atan2(s*(p-c),s*s+p*c) for p>=c>=0.
    Scale both atan2 arguments before multiplication to avoid overflow. This
    evaluates the same telescoping potential difference in real arithmetic.
    """
    scale, amount = _parameters(age_scale, strength)
    parent = _ages(parent_times, 'parent_times')
    child = _ages(child_times, 'child_times')
    if parent.shape != child.shape or bool((parent < child).any()):
        raise ValueError('Require matching ages with parent_times >= child_times')
    duration = parent - child
    divisor = torch.maximum(parent, parent.new_full(parent.shape, scale))
    ps, cs, ss = parent / divisor, child / divisor, scale / divisor
    numerator = ss * (duration / divisor)
    denominator = ss.square() + ps * cs
    angle = torch.atan2(numerator, denominator)
    # Only possible with extreme underflow. The missing term is negligible
    # relative to duration unless c=0, where the limiting angle is pi/2.
    angle = torch.where((numerator == 0) & (denominator == 0) & (child == 0),
                        angle.new_full(angle.shape, math.pi / 2), angle)
    correction = (amount * angle) * scale
    result = (1 + amount) * duration - correction
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Age potential difference overflows float64')
    return result


@torch.no_grad()
def observed_age_laplacian(node_times: Tensor, edge_index: Tensor, *,
                           age_scale: float, strength: float = 3.,
                           resistance_scale: float = 1e-4) -> Tensor:
    """Positive integrated age resistance on observed edges, no epsilon floor."""
    times = _ages(node_times, 'node_times')
    if times.ndim != 1 or times.numel() == 0:
        raise ValueError('node_times must be a nonempty CPU vector')
    if not math.isfinite(resistance_scale) or resistance_scale <= 0:
        raise ValueError('resistance_scale must be finite and positive')
    if not isinstance(edge_index, Tensor) or edge_index.device.type != 'cpu' or edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index must be CPU long [2,E]')
    if bool(((edge_index < 0) | (edge_index >= times.numel())).any()):
        raise ValueError('edge endpoint out of range')
    endpoints = times[edge_index]
    difference = age_potential_difference(endpoints.max(dim=0).values,
        endpoints.min(dim=0).values, age_scale, strength=strength)
    return resistance_laplacian(times.numel(), edge_index, resistance_scale * difference)


@dataclass(frozen=True)
class AgeInterfaceCache:
    raw_response: Tensor             # [G,S,B,B], plain then declared age scales
    normalized_response: Tensor      # raw_response / one plain scale [G]
    normalization_scale: Tensor      # [G], plain full-boundary mean diagonal
    cqq: Tensor                      # [G,S,Q,Q], original query order
    cqr: Tensor                      # [G,S,Q,R], unchanged reference order
    eigenvalues: Tensor              # [G,S,Q]
    eigenvectors: Tensor             # [G,S,Q,Q]
    scalar_cqq: Tensor               # [G,S,Q,Q], mean-matched plain geometry
    scalar_cqr: Tensor               # [G,S,Q,R]
    scalar_eigenvalues: Tensor        # [G,S,Q], scaled plain eigenvalues
    scalar_eigenvectors: Tensor      # [G,S,Q,Q], repeated plain basis
    mean_diagonal_ratios: Tensor     # [G,S], age raw mean / plain raw mean
    sample_ids: Tensor               # [B], unchanged sorted boundary identities
    boundary_nodes: Tensor           # [G,B], original packed node indices
    query_indices: Tensor            # [Q]
    reference_indices: Tensor        # [R]
    query_ids: Tensor                # [Q]
    reference_ids: Tensor            # [R]
    interval_weights: Tensor         # [G], exact physical-width weights
    diagnostics: tuple[tuple[KronDiagnostics, ...], ...]  # [G][S], raw Schur audit
    spectral_reconstruction_relative: Tensor  # [G,S]
    spectral_orthogonality_max: Tensor         # [G,S]
    operator_scales: tuple[float, ...]         # (0,350,1400) by default
    operator_labels: tuple[str, ...]          # ('plain','age350','age1400')
    strength: float
    resistance_scale: float
    time_scale: float
    normalization: str = 'shared_plain_mean_boundary_diagonal'
    schema_version: int = 1

    @property
    def num_scales(self) -> int:
        return len(self.operator_scales)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> AgeInterfaceCache:
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device=device, dtype=dtype if value.is_floating_point() else None) if isinstance(value, Tensor) else value
        return AgeInterfaceCache(**values)

    def as_dict(self) -> dict:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values['diagnostics'] = [[asdict(item) for item in row] for row in self.diagnostics]
        return values

    @classmethod
    def from_dict(cls, values: dict) -> AgeInterfaceCache:
        if not isinstance(values, dict) or values.get('schema_version') != 1 or set(values) != {field.name for field in fields(cls)}:
            raise ValueError('AgeInterfaceCache schema/fields mismatch')
        values = dict(values)
        values['diagnostics'] = tuple(tuple(KronDiagnostics(**item) for item in row) for row in values['diagnostics'])
        values['operator_scales'] = tuple(values['operator_scales'])
        values['operator_labels'] = tuple(values['operator_labels'])
        cache = cls(**values)
        _validate_cache(cache)
        return cache


@dataclass(frozen=True)
class AgeInterfaceBank:
    plain_bank: LeafInterfaceBank
    operator_cache: AgeInterfaceCache
    schema_version: int = 1

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> AgeInterfaceBank:
        # Coordinate tensors remain in their original precision via plain.to().
        return AgeInterfaceBank(self.plain_bank.to(device, dtype),
                                self.operator_cache.to(device, dtype))

    def as_dict(self) -> dict:
        return {'plain_bank': self.plain_bank.as_dict(),
                'operator_cache': self.operator_cache.as_dict(), 'schema_version': 1}

    @classmethod
    def from_dict(cls, values: dict) -> AgeInterfaceBank:
        if not isinstance(values, dict) or values.get('schema_version') != 1 or set(values) != {'plain_bank', 'operator_cache', 'schema_version'}:
            raise ValueError('AgeInterfaceBank schema/fields mismatch')
        result = cls(LeafInterfaceBank.from_dict(values['plain_bank']),
                     AgeInterfaceCache.from_dict(values['operator_cache']))
        _validate_pair(result)
        return result


def _validate_cache(cache: AgeInterfaceCache) -> None:
    response = cache.raw_response
    if not isinstance(response, Tensor) or response.ndim != 4 or response.shape[-1] != response.shape[-2] or min(response.shape) < 1:
        raise ValueError('Age raw_response must be nonempty [G,S,B,B]')
    g, s, b, _ = response.shape
    q, r = cache.query_ids.numel(), cache.reference_ids.numel()
    shapes = {
        'normalized_response': (g,s,b,b), 'normalization_scale': (g,),
        'cqq': (g,s,q,q), 'cqr': (g,s,q,r), 'eigenvalues': (g,s,q),
        'eigenvectors': (g,s,q,q), 'scalar_cqq': (g,s,q,q),
        'scalar_cqr': (g,s,q,r), 'scalar_eigenvalues': (g,s,q),
        'scalar_eigenvectors': (g,s,q,q), 'mean_diagonal_ratios': (g,s),
        'interval_weights': (g,), 'spectral_reconstruction_relative': (g,s),
        'spectral_orthogonality_max': (g,s),
    }
    if not response.is_floating_point() or not bool(torch.isfinite(response).all()):
        raise ValueError('Age responses must be finite floating tensors')
    for name, shape in shapes.items():
        value = getattr(cache, name)
        if not isinstance(value, Tensor) or value.shape != shape or value.dtype != response.dtype or value.device != response.device or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Age cache {name} shape/dtype/device/finite mismatch')
    for name, shape in {'sample_ids': (b,), 'boundary_nodes': (g,b),
                        'query_ids': (q,), 'reference_ids': (r,),
                        'query_indices': (q,), 'reference_indices': (r,)}.items():
        value = getattr(cache, name)
        if not isinstance(value, Tensor) or value.shape != shape or value.dtype != torch.long or value.device != response.device:
            raise ValueError(f'Age cache {name} shape/dtype/device mismatch')
    if q == 0 or q + r != b or not torch.equal(torch.cat((cache.query_ids,cache.reference_ids)).sort().values,cache.sample_ids):
        raise ValueError('Age cache requires the exhaustive boundary partition')
    if torch.unique(cache.sample_ids).numel() != b or bool((cache.sample_ids < 0).any()) or bool((cache.boundary_nodes < 0).any()):
        raise ValueError('Age cache identities/packed boundary nodes are invalid')
    selected = torch.cat((cache.query_indices,cache.reference_indices))
    if torch.unique(selected).numel() != b or bool(((selected < 0) | (selected >= b)).any()) or not torch.equal(cache.sample_ids[cache.query_indices],cache.query_ids) or not torch.equal(cache.sample_ids[cache.reference_indices],cache.reference_ids):
        raise ValueError('Age cache boundary indices/order mismatch')
    if cache.normalization != 'shared_plain_mean_boundary_diagonal' or bool((cache.normalization_scale <= 0).any()) or bool((cache.mean_diagonal_ratios < 0).any()):
        raise ValueError('Age cache shared normalization is invalid')
    if bool((cache.interval_weights <= 0).any()) or not math.isclose(float(cache.interval_weights.sum()),1.,rel_tol=1e-6,abs_tol=1e-7):
        raise ValueError('Age cache physical interval weights are invalid')
    if len(cache.operator_scales) != s or len(cache.operator_labels) != s or cache.operator_scales[0] != 0 or cache.operator_labels[0] != 'plain' or len(set(cache.operator_scales)) != s or len(set(cache.operator_labels)) != s or any(not math.isfinite(x) or x <= 0 for x in cache.operator_scales[1:]):
        raise ValueError('Age cache operator scale schema is invalid')
    if len(cache.diagnostics) != g or any(len(row) != s for row in cache.diagnostics):
        raise ValueError('Age cache diagnostics shape mismatch')
    if not math.isfinite(cache.strength) or cache.strength < 0 or not math.isfinite(cache.resistance_scale) or cache.resistance_scale <= 0 or not math.isfinite(cache.time_scale) or cache.time_scale <= 0:
        raise ValueError('Age cache physical parameters are invalid')


def _validate_pair(bank: AgeInterfaceBank) -> None:
    plain, cache = bank.plain_bank, bank.operator_cache
    if plain.query_cache is None:
        raise ValueError('Age bank requires the ordinary plain query cache')
    for name, other in [('sample_ids',plain.sample_ids), ('boundary_nodes',plain.boundary_nodes),
                        ('query_ids',plain.query_node_ids), ('reference_ids',plain.query_cache.reference_ids)]:
        if not torch.equal(getattr(cache,name),other):
            raise ValueError(f'Age/plain {name} mismatch')
    if not torch.equal(cache.raw_response[:,0],plain.response) or not torch.equal(cache.normalization_scale,plain.query_cache.normalization_scale):
        raise ValueError('Age/plain response or shared scale mismatch')


def _verify_supplied_plain(actual: LeafInterfaceBank, expected: LeafInterfaceBank) -> None:
    if actual.geometry_kind != 'observed_tree_leaf' or actual.resistance_scale != expected.resistance_scale or actual.time_scale != expected.time_scale or actual.feature_columns != expected.feature_columns:
        raise ValueError('Supplied plain bank physical/feature schema mismatch')
    for name in ('sample_ids','boundary_nodes','leaf_x','intervals','query_node_ids',
                 'positions','target_graph','window_bounds'):
        left, right = getattr(actual,name),getattr(expected,name)
        if (left is None) != (right is None) or (left is not None and (left.device.type != 'cpu' or left.dtype != right.dtype or not torch.equal(left,right))):
            raise ValueError(f'Supplied plain bank {name} differs from observed source')
    if actual.response.device.type != 'cpu' or actual.response.dtype != torch.float64 or actual.response.shape != expected.response.shape or not torch.allclose(actual.response,expected.response,rtol=1e-11,atol=0):
        raise ValueError('Supplied plain bank response differs from observed source')
    if actual.query_cache is not None:
        pc = actual.query_cache
        if pc.normalization != 'mean_boundary_diagonal' or not torch.equal(pc.query_ids,actual.query_node_ids):
            raise ValueError('Supplied plain cache must use original query order and full-boundary mean diagonal')
        fresh = with_query_reference_cache(actual,pc.reference_ids).query_cache
        for name in ('normalized_response','normalization_scale','cqq','cqr','query_indices',
                     'reference_indices','query_ids','reference_ids','interval_weights','bag_cqq'):
            if not torch.equal(getattr(pc,name),getattr(fresh,name)):
                raise ValueError(f'Supplied plain query cache {name} differs from plain geometry')
        # Eigenvector signs and degenerate rotations may differ, so validate
        # reconstruction and orthogonality rather than requiring a fresh basis.
        _spectral_quality(pc.cqq,pc.eigenvalues,pc.eigenvectors)


def _spectral_quality(matrix: Tensor, values: Tensor, vectors: Tensor) -> tuple[Tensor,Tensor]:
    if values.shape != matrix.shape[:-1] or vectors.shape != matrix.shape or values.dtype != matrix.dtype or vectors.dtype != matrix.dtype or values.device != matrix.device or vectors.device != matrix.device:
        raise ValueError('Fixed eigensystem shape/dtype/device mismatch')
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(vectors).all()):
        raise ValueError('Fixed eigensystem must be finite')
    reconstructed = (vectors * values.unsqueeze(-2)) @ vectors.transpose(-1,-2)
    error = torch.linalg.matrix_norm(reconstructed-matrix)
    norm = torch.linalg.matrix_norm(matrix)
    relative = torch.where(norm == 0,error,error/torch.where(norm == 0,torch.ones_like(norm),norm))
    identity = torch.eye(matrix.shape[-1],dtype=matrix.dtype,device=matrix.device)
    orthogonality = (vectors.transpose(-1,-2) @ vectors - identity).abs().amax(dim=(-2,-1))
    tolerance = 1e-9 if matrix.dtype == torch.float64 else 1e-5
    if bool((relative > tolerance).any()) or bool((orthogonality > tolerance).any()):
        raise ValueError('Fixed eigensystem failed reconstruction/orthogonality checks')
    return relative,orthogonality


@torch.no_grad()
def precompute_age_interfaces(seq: GraphSequenceInput, *,
        plain_bank: LeafInterfaceBank | None = None, time_column: int = 6,
        parent_branch_column: int = 7, time_scale: float = 10000.,
        resistance_scale: float = 1e-4, age_scales: Sequence[float] = (350.,1400.),
        strength: float = 3., reference_ids: Tensor | None = None,
        feature_columns: Sequence[int] | None = None,
        max_condition: float = 1e12) -> AgeInterfaceBank:
    """Decode observed ages, then build fixed age operators on CPU float64.

    A supplied plain bank is checked against the exact observed sequence and
    retained. By default references are the sorted complement of query IDs.
    age_scales=() builds S1 plain and never evaluates an age operator.
    """
    scales = tuple(_parameters(x,strength)[0] for x in age_scales)
    if len(set(scales)) != len(scales):
        raise ValueError('age_scales must be unique')
    if not math.isfinite(float(strength)) or float(strength) < 0:
        raise ValueError('strength must be finite and nonnegative')
    columns = plain_bank.feature_columns if plain_bank is not None and feature_columns is None else feature_columns
    fresh = precompute_leaf_interfaces(seq,time_column=time_column,
        parent_branch_column=parent_branch_column,time_scale=time_scale,
        resistance_scale=resistance_scale,feature_columns=columns,max_condition=max_condition)
    if plain_bank is None:
        plain = fresh
    else:
        _verify_supplied_plain(plain_bank,fresh)
        plain = plain_bank
    if reference_ids is None:
        reference_ids = plain.query_cache.reference_ids if plain.query_cache is not None else plain.sample_ids[~torch.isin(plain.sample_ids,plain.query_node_ids)]
    if plain.query_cache is None:
        plain = with_query_reference_cache(plain,reference_ids)
    elif not torch.equal(reference_ids,plain.query_cache.reference_ids):
        raise ValueError('Age reference order must match the ordinary plain cache')
    pc = plain.query_cache
    times = torch.expm1(_cpu_double(seq.x[:,time_column],'encoded observed ages') * math.log1p(time_scale))
    if not bool(torch.isfinite(times).all()) or bool((times < 0).any()):
        raise ValueError('Decoded observed ages must be finite and nonnegative')
    g = plain.response.shape[0]
    groups = [[] for _ in range(g)]
    edge_groups = [[] for _ in range(g)]
    graph_ids = seq.node_graph.tolist()
    for node,graph in enumerate(graph_ids):
        groups[graph].append(node)
    for u,v in sorted({tuple(sorted(pair)) for pair in seq.edge_index.T.tolist()}):
        edge_groups[graph_ids[u]].append((u,v))
    graph_responses,diagnostics = [],[]
    for graph,packed in enumerate(groups):
        nodes = torch.tensor(packed,dtype=torch.long)
        lookup = {node:i for i,node in enumerate(packed)}
        edges = torch.tensor([(lookup[u],lookup[v]) for u,v in edge_groups[graph]],dtype=torch.long).reshape(-1,2).T.contiguous()
        boundary = torch.tensor([lookup[node] for node in plain.boundary_nodes[graph].tolist()],dtype=torch.long)
        responses,row_diagnostics = [plain.response[graph]],[plain.diagnostics[graph]]
        for scale in scales:
            laplacian = observed_age_laplacian(times[nodes],edges,age_scale=scale,
                strength=strength,resistance_scale=resistance_scale)
            reduced = kron_reduce(laplacian,boundary,max_condition=max_condition)
            responses.append(reduced.response)
            row_diagnostics.append(reduced.diagnostics)
        graph_responses.append(torch.stack(responses))
        diagnostics.append(tuple(row_diagnostics))
    raw = torch.stack(graph_responses)
    normalized = raw / pc.normalization_scale[:,None,None,None]
    partition = query_reference_partition(normalized,plain.sample_ids,plain.query_node_ids,reference_ids)
    # Reuse the plain fixed basis exactly. Every remaining matrix is fixed and
    # eigendecomposed here, never in a neural forward or learned-matrix cache.
    values,vectors = [pc.eigenvalues],[pc.eigenvectors]
    for index in range(1,len(scales)+1):
        eigenvalues,eigenvectors = torch.linalg.eigh(partition.cqq[:,index])
        values.append(eigenvalues); vectors.append(eigenvectors)
    eigenvalues,eigenvectors = torch.stack(values,dim=1),torch.stack(vectors,dim=1)
    spectral,orthogonality = _spectral_quality(partition.cqq,eigenvalues,eigenvectors)
    ratios = raw.diagonal(dim1=-2,dim2=-1).mean(-1) / pc.normalization_scale[:,None]
    # Exact zero C0 implies all age responses zero. Set a benign ratio of one
    # for that zero-geometry degeneracy, avoiding division by zero.
    zero = plain.response.abs().amax(dim=(-2,-1)) == 0
    if bool((raw[zero].abs().amax(dim=(-2,-1)) != 0).any()):
        raise ValueError('Age operator unexpectedly nonzero for analytically zero plain response')
    ratios = torch.where(zero[:,None],torch.ones_like(ratios),ratios)
    scalar_cqq = pc.cqq[:,None] * ratios[:,:,None,None]
    scalar_cqr = pc.cqr[:,None] * ratios[:,:,None,None]
    scalar_eigenvalues = pc.eigenvalues[:,None] * ratios[:,:,None]
    scalar_eigenvectors = pc.eigenvectors[:,None].expand(-1,len(scales)+1,-1,-1).clone()
    cache = AgeInterfaceCache(raw,normalized,pc.normalization_scale.clone(),
        partition.cqq,partition.cqr,eigenvalues,eigenvectors,scalar_cqq,scalar_cqr,
        scalar_eigenvalues,scalar_eigenvectors,ratios,plain.sample_ids.clone(),
        plain.boundary_nodes.clone(),partition.query_indices,partition.reference_indices,
        plain.query_node_ids.clone(),reference_ids.clone(),pc.interval_weights.clone(),
        tuple(diagnostics),spectral,orthogonality,(0.,*scales),
        ('plain',*(f'age{scale:g}' for scale in scales)),float(strength),
        float(resistance_scale),float(time_scale))
    _validate_cache(cache)
    result = AgeInterfaceBank(plain,cache)
    _validate_pair(result)
    return result

"""Fixed geographic interface adapter for observed-only Chickenpox forecasting.

Uses the unchanged BoundaryMemoryModel and its controls. Precomputation happens
once per geographic configuration, while histories share the fixed response,
partition and eigensystem. Forecast labels remain external to the bank.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .boundary_memory import BoundaryMemoryModel
from .chickenpox import ChickenpoxData, TrainingScaler
from .kron_interface import (LeafInterfaceBank, QueryInterfaceCache,
                             kron_reduce, resistance_laplacian)


def _tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _validate_county_graph(data: ChickenpoxData) -> Tensor:
    n = len(data.counties)
    if n < 1 or len(set(data.counties)) != n or data.counts.ndim != 2 or data.counts.shape[1] != n:
        raise ValueError("Require unique county names matching count columns")
    edge = np.asarray(data.edge_index)
    if edge.ndim != 2 or edge.shape[0] != 2 or not np.issubdtype(edge.dtype, np.integer):
        raise ValueError("County adjacency must be integer [2,E]")
    if np.any(edge < 0) or np.any(edge >= n):
        raise ValueError("County adjacency endpoint out of range")
    pairs = set(map(tuple, edge.T.tolist()))
    if not all((v, u) in pairs for u, v in pairs):
        raise ValueError("County adjacency must be symmetric")
    return torch.tensor(edge.copy(), dtype=torch.long)


@dataclass(frozen=True)
class ChickenpoxBoundaryGeometry:
    """Immutable configuration shared by all observed history feature banks.

    Call ``to`` once before binding histories on a new model dtype/device. Only
    the per-history [52,20,1] features change in ``bind_chickenpox_history``.
    Native adjacency, county order and source identity remain auditable.
    """
    template: LeafInterfaceBank
    counties: tuple[str, ...]
    native_edge_index: Tensor
    source_archive_sha256: str
    history: int
    no_geography: bool
    raw_laplacian_sha256: str

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> ChickenpoxBoundaryGeometry:
        return replace(self, template=self.template.to(device, dtype))

    def configuration_manifest(self) -> dict[str, Any]:
        cache = self.template.query_cache
        edge = self.native_edge_index
        nonself = edge[0] != edge[1]
        return {
            "kind": "fixed_geography_all_persistent_boundary",
            "source_archive_sha256": self.source_archive_sha256,
            "counties": list(self.counties), "history_weeks": self.history,
            "boundary_nodes": len(self.counties), "query_nodes": len(self.counties),
            "reference_nodes": 0, "native_adjacency_records": edge.shape[1],
            "native_self_loop_records": int((~nonself).sum()),
            "native_undirected_neighbor_links": len({tuple(sorted(p)) for p in edge[:, nonself].T.tolist()}),
            "no_geography": self.no_geography,
            "geometry": "C=L with all county nodes persistent, no interior reduction",
            "conductance": "one per unique native undirected non-self neighbor link",
            "self_loop_handling": "native loops cancel in D-A; source GNN has its own self/residual update",
            "raw_laplacian_sha256": self.raw_laplacian_sha256,
            "normalization": cache.normalization,
            "normalization_scale": float(cache.normalization_scale[0]),
            "features": "one observed count per county, standardized by training-only scaler",
            "coordinates": "52 uniform weekly holds on [0,1], final observed interval midpoint anchor" if self.history == 52 else f"{self.history} uniform weekly holds on [0,1], final observed interval midpoint anchor",
            "coordinate_dtype": str(self.template.positions.dtype),
            "cached_model_dtype": str(self.template.leaf_x.dtype),
            "eigensystem_reuse": "one fixed normalized L eigensystem shared across holds, bag and histories",
            "target_label_in_bank": False,
        }


@torch.no_grad()
def precompute_chickenpox_geometry(data: ChickenpoxData, history: int = 52, *,
                                    no_geography: bool = False) -> ChickenpoxBoundaryGeometry:
    """Prepare fixed CPU float64 C=L and one normalized eigensystem.

    Supplied self loops have zero Laplacian contribution, and bidirected links
    are deduplicated rather than doubled. ``no_geography`` removes the links,
    giving exact C=0 with scale one under the same source encoder and controls.
    Weekly coordinates intentionally match ``forecast_sequence`` float32.
    """
    if not isinstance(history, int) or history < 1:
        raise ValueError("history must be a positive integer")
    native = _validate_county_graph(data)
    n = len(data.counties)
    neighbor = native[:, native[0] != native[1]]
    if no_geography:
        neighbor = neighbor[:, :0]
    laplacian = resistance_laplacian(n, neighbor, torch.ones(neighbor.shape[1], dtype=torch.float64))
    ids = torch.arange(n, dtype=torch.long)
    reduction = kron_reduce(laplacian, ids)
    response = reduction.response
    scale = response.diagonal().mean() if bool(response.abs().any()) else response.new_tensor(1.)
    normalized = response / scale
    values, vectors = torch.linalg.eigh(normalized)
    bounds = torch.linspace(0., 1., history+1)
    intervals = torch.stack((bounds[:-1], bounds[1:]), dim=1)
    widths = (intervals[:, 1]-intervals[:, 0]).double()
    weights = widths/widths.sum()
    # Every hold has exactly the same generator, so its weighted mean is that
    # generator and the same fixed eigensystem is valid for the bag control.
    cache = QueryInterfaceCache(
        normalized_response=normalized.expand(history, -1, -1),
        normalization_scale=scale.expand(history),
        cqq=normalized.expand(history, -1, -1),
        cqr=response.new_empty((history, n, 0)),
        eigenvalues=values.expand(history, -1),
        eigenvectors=vectors.expand(history, -1, -1),
        query_indices=ids, reference_indices=ids[:0], query_ids=ids,
        reference_ids=ids[:0], interval_weights=weights,
        bag_cqq=normalized, bag_eigenvalues=values, bag_eigenvectors=vectors)
    template = LeafInterfaceBank(
        response=response.expand(history, -1, -1), sample_ids=ids,
        boundary_nodes=torch.arange(history*n).reshape(history, n),
        leaf_x=response.new_zeros((history, n, 1)), feature_columns=(0,),
        intervals=intervals, query_node_ids=ids,
        positions=intervals[-1].mean().reshape(1),
        target_graph=torch.tensor([history-1]), window_bounds=bounds[[0, -1]],
        diagnostics=(reduction.diagnostics,)*history,
        resistance_scale=None, time_scale=None,
        geometry_kind="generic_graph_boundary", query_cache=cache)
    return ChickenpoxBoundaryGeometry(template, tuple(data.counties), native,
        data.archive_sha256, history, bool(no_geography), _tensor_sha256(laplacian))


def bind_chickenpox_history(geometry: ChickenpoxBoundaryGeometry, data: ChickenpoxData,
                           scaler: TrainingScaler, target: int,
                           permutation: np.ndarray | None = None) -> LeafInterfaceBank:
    """Bind exactly [target-history:target] observed counts to shared geometry.

    No target values, labels, dates, identifiers as numeric features or future
    values enter the bank. Standardization and permutation match the existing
    protocol. For its chronology intervention, use history_permutation with
    fix_last=True. All geometry/cache tensors are shared without online eigh.
    """
    if (tuple(data.counties) != geometry.counties or data.archive_sha256 != geometry.source_archive_sha256
            or not np.array_equal(data.edge_index, geometry.native_edge_index.numpy())):
        raise ValueError("History source/counties/topology differs from prepared geometry")
    history, n = geometry.history, len(geometry.counties)
    if data.counts.ndim != 2 or data.counts.shape[1] != n:
        raise ValueError("Count columns differ from prepared county order")
    if not isinstance(target, (int, np.integer)) or not history <= target < data.counts.shape[0]:
        raise ValueError("Target must follow a complete observed history")
    if (scaler.mean.shape != (n,) or scaler.scale.shape != (n,)
            or not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all()
            or (scaler.scale <= 0).any() or not history < scaler.fit_end <= data.counts.shape[0]):
        raise ValueError("Require a valid training-only county scaler")
    values = scaler.transform(data.counts[target-history:target])
    if permutation is not None:
        permutation = np.asarray(permutation)
        if (permutation.shape != (history,) or not np.issubdtype(permutation.dtype, np.integer)
                or not np.array_equal(np.sort(permutation), np.arange(history))):
            raise ValueError("permutation must reorder each history graph once")
        values = values[permutation]
    if not np.isfinite(values).all():
        raise ValueError("Observed standardized history must be finite")
    template = geometry.template
    # The existing forecast_sequence first casts standardized observations to
    # float32. Preserve those exact common-input values even for double models.
    features = torch.as_tensor(np.asarray(values, dtype=np.float32).copy()).reshape(history, n, 1)
    features = features.to(device=template.leaf_x.device, dtype=template.leaf_x.dtype)
    return replace(template, leaf_x=features)


class ChickenpoxBoundaryModel(BoundaryMemoryModel):
    """Same learned memory and controls, one-sided scalar forecast constructor."""
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2, latent_dim: int = 8,
                 coupling: str = "full", aggregation: str = "ordered",
                 initial_coupling: float = 20., initial_damping: float = 1.,
                 validate: bool = True):
        super().__init__(input_dim=1, num_classes=1, hidden_dim=hidden_dim,
            num_layers=num_layers, latent_dim=latent_dim, coupling=coupling,
            aggregation=aggregation, bidirectional=False, symmetric_readout=False,
            initial_coupling=initial_coupling, initial_damping=initial_damping,
            validate=validate)

    def configuration_manifest(self, bank: LeafInterfaceBank | None = None) -> dict[str, Any]:
        result = super().configuration_manifest(bank)
        result["memory_configuration_sha256"] = result.pop("configuration_sha256")
        result.update(task="one-week-ahead standardized county count regression",
            input_history="52 observed graphs under frozen Chickenpox protocol",
            target="external following-week labels, outside observed historical domain",
            prediction_clipping="none, matching frozen pilot protocol",
            inactive_reference_parameters=(self.hidden_dim+1)*self.latent_dim,
            raw_parent_branch_feature="not applicable; geographic inputs have no branch feature",
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        result["configuration_sha256"] = hashlib.sha256(canonical).hexdigest()
        return result

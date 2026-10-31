"""Observed census-edge retention for the unchanged whole-tree Census model.

Only fixed observed boundary_rho changes. No loader, trainer, model, optimizer,
loss, resource tooling, source compilation or monkeypatch is added.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import Tensor

from . import census_biological_bank_v1 as core
from .census_tsinfer_observation_v1 import (
    CertifiedCensusTsinferObservationV1, _outer_content_v1,
)

ARMS = ("current_jaccard", "genuine_edge", "sham_edge")
CENSUS = 350.
SPAN_BIN_UPPER_BP = (100., 1000., 10000.)
ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/census_lineage_memory_v1.py"
CONFIG = "configs/census_lineage_memory_v1.json"


def native_census_edges_v1(tree, native_ids: Tensor, query_ids: Tensor):
    """Actual parent-wide (lower,upper) IDs, or explicit absence.

    Cut convention: lower_time <=350 <upper_time. A dynamics leaf at/older
    than350, or a root below350, supplies no invented crossing identity.
    """
    parent, times = tree.parent, tree.node_times
    n = parent.numel()
    arrays = (parent, times, native_ids, tree.sample_ids, tree.sample_nodes, query_ids)
    if any(x.device.type != "cpu" or x.requires_grad or x.ndim != 1 for x in arrays):
        raise ValueError("Fixed one-dimensional CPU observed geometry required")
    if (not n or times.shape != parent.shape or native_ids.shape != parent.shape
            or tree.sample_ids.shape != tree.sample_nodes.shape
            or any(x.dtype != torch.long for x in
                   (parent, native_ids, tree.sample_ids, tree.sample_nodes, query_ids))
            or bool(((parent < -1) | (parent >= n)).any())
            or not bool(torch.isfinite(times).all())
            or bool((native_ids < 0).any())
            or len(set(native_ids.tolist())) != n):
        raise ValueError("Complete unique actual native IDs and matching tree dimensions required")
    lookup = dict(zip(tree.sample_ids.tolist(), tree.sample_nodes.tolist()))
    if (len(lookup) != tree.sample_ids.numel() or not len(query_ids)
            or len(set(query_ids.tolist())) != len(query_ids)):
        raise ValueError("Unique observed samples and nonempty unique query order required")
    keys, status = [], []
    for query in query_ids.tolist():
        if query not in lookup:
            raise ValueError("Missing persistent query sample join")
        node = lookup[query]
        if not 0 <= node < n:
            raise ValueError("Invalid observed query node")
        if float(times[node]) >= CENSUS:
            keys.append(None); status.append("at_or_older_than_census"); continue
        steps = 0
        while int(parent[node]) >= 0 and float(times[int(parent[node])]) <= CENSUS:
            node = int(parent[node]); steps += 1
            if steps > n:
                raise ValueError("Observed parent cycle")
        upper = int(parent[node])
        if upper < 0:
            keys.append(None); status.append("no_crossing_edge"); continue
        if not float(times[node]) <= CENSUS < float(times[upper]):
            raise ValueError("Missing genuine native census-crossing edge")
        keys.append((int(native_ids[node]), int(native_ids[upper])))
        status.append("crossing_edge")
    return tuple(keys), tuple(status)


def sham_changes_v1(changed: Tensor, eligible: Tensor, span_bins: Tensor,
                    query_ids: Tensor, *, parent_id: str, namespace: str,
                    tile_id: int, seed: int) -> Tensor:
    """Preserve counts per parent/query/tile/right-span bin using local RNG.

    Unknown edges are never permutation destinations. The extra tile stratum
    retains the original trainer's independent within-tile memory scope.
    """
    if (changed.dtype != torch.bool or eligible.dtype != torch.bool
            or span_bins.dtype != torch.long or query_ids.dtype != torch.long
            or changed.ndim != 2 or changed.shape != eligible.shape
            or span_bins.shape != (changed.shape[0],)
            or query_ids.shape != (changed.shape[1],)
            or any(x.device.type != "cpu" or x.requires_grad
                   for x in (changed, eligible, span_bins, query_ids))
            or type(seed) is not int or type(tile_id) is not int
            or not parent_id or not namespace
            or bool((changed & ~eligible).any())
            or bool(((span_bins < 0) | (span_bins > len(SPAN_BIN_UPPER_BP))).any())):
        raise ValueError("Fixed matching change/eligibility/query/span dimensions required")
    result = changed.clone()
    for q, query in enumerate(query_ids.tolist()):
        for bin_id in range(len(SPAN_BIN_UPPER_BP) + 1):
            rows = torch.nonzero(eligible[:, q] & (span_bins == bin_id)).flatten().tolist()
            values = changed[rows, q].tolist()
            identity = [parent_id, namespace, tile_id, int(query), seed, bin_id]
            local_seed = int.from_bytes(hashlib.sha256(
                json.dumps(identity, separators=(",", ":")).encode()).digest()[:8], "big")
            random.Random(local_seed).shuffle(values)
            if rows:
                result[rows, q] = torch.tensor(values, dtype=torch.bool)
            if int(result[rows, q].sum()) != int(changed[rows, q].sum()):
                raise ValueError("Sham changed parent/query/span-bin event count")
    if bool((result & ~eligible).any()):
        raise ValueError("Sham invented an unknown-edge change")
    return result


@dataclass(frozen=True)
class LineageMemoryInputsV1:
    parent_id: str
    namespace: str
    tile_id: int
    edge_keys: tuple
    edge_status: tuple
    boundary_events: tuple[int, ...]
    span_bins: Tensor
    eligible: Tensor
    genuine_changed: Tensor
    sham_changed: Tensor
    current_jaccard_certificate: CertifiedCensusTsinferObservationV1
    genuine_edge_certificate: CertifiedCensusTsinferObservationV1
    sham_edge_certificate: CertifiedCensusTsinferObservationV1

    def certificate_for_arm(self, arm: str):
        if arm not in ARMS:
            raise ValueError("Unknown matched lineage-memory arm")
        return getattr(self, arm + "_certificate")

    def bank_for_arm(self, arm: str):
        return self.certificate_for_arm(arm).native_certificate.bank

    def summary(self):
        return {
            "parent_id": self.parent_id, "namespace": self.namespace,
            "tile_id": self.tile_id, "boundary_events": len(self.boundary_events),
            "query_count": self.eligible.shape[1],
            "eligible_event_queries": int(self.eligible.sum()),
            "unknown_event_queries": int((~self.eligible).sum()),
            "genuine_changes": int(self.genuine_changed.sum()),
            "sham_changes": int(self.sham_changed.sum()),
            "sham_reassigned_indicators": int((self.genuine_changed != self.sham_changed).sum()),
            "sham_strata": "parent/query/tile/fixed_right_physical_span_bin",
            "native_identity_is_observed_continuity_proxy": True,
            "arm_certificate_sha256": {
                arm: self.certificate_for_arm(arm).certificate_sha256 for arm in ARMS
            },
        }


def prepare_lineage_memory_v1(observation, expected_certificate_sha256: str, *,
                              seed: int, experiment_config_sha256: str):
    """Freshly certified arms derived from an existing same-tsinfer observation.

    Masks use only parent links/dates/native maps/physical spans. No target is
    accepted or accessed. The original certificate and bank remain unchanged.
    The caller pins the exact experiment config bytes independently.
    """
    if not isinstance(observation, CertifiedCensusTsinferObservationV1):
        raise ValueError("Existing same-tsinfer observation certificate required")
    config_bytes = (ROOT / CONFIG).read_bytes()
    if (not isinstance(experiment_config_sha256, str)
            or hashlib.sha256(config_bytes).hexdigest() != experiment_config_sha256):
        raise ValueError("Pinned lineage-memory experiment config bytes changed")
    experiment_config = json.loads(config_bytes)
    if (experiment_config.get("kind") != "census_lineage_memory_v1"
            or experiment_config.get("arms") != list(ARMS)
            or experiment_config.get("sham", {}).get("span_bin_upper_bp") != list(SPAN_BIN_UPPER_BP)
            or type(seed) is not int or seed not in experiment_config.get("seeds", ())):
        raise ValueError("Matched lineage-memory arms, span bins and configured seed required")
    source_binding = {"path": SOURCE, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_binding = {"path": CONFIG, "sha256": experiment_config_sha256}
    inner, lineage = observation.native_certificate, observation.lineage
    core.validate_census_biological_certificate_v1(inner, inner.certificate_sha256)
    content = _outer_content_v1(inner, lineage)
    if (observation.certificate_sha256 != expected_certificate_sha256
            or core.fingerprint_sha256_v1(content) != expected_certificate_sha256
            or lineage.get("no_copied_Relate_features") is not True):
        raise ValueError("Original same-tsinfer observed certificate changed")
    parent_id, namespace = lineage.get("parent_id"), lineage.get("namespace")
    tile_id = inner.provenance.get("tile_id")
    if (not isinstance(parent_id, str) or not parent_id
            or not isinstance(namespace, str) or not namespace
            or parent_id != inner.provenance.get("parent_id") or type(tile_id) is not int):
        raise ValueError("Actual parent-wide namespace and tile identity required")
    bank = inner.bank
    maps = inner.provenance.get("native_vertex_by_source_row")
    physical = inner.provenance.get("physical_intervals_bp")
    if maps is None or physical is None:
        raise ValueError("Missing actual native maps or physical holds")
    spans = torch.as_tensor(physical, dtype=torch.float64)
    if (len(maps) != len(inner.native_trees)
            or len(bank.partitions) != len(inner.native_trees)
            or spans.shape != (len(inner.native_trees), 2)
            or not bool(torch.isfinite(spans).all() & (spans[:, 1] > spans[:, 0]).all())
            or any(part.census != CENSUS for part in bank.partitions)
            or bank.boundary_rho.shape != (len(bank.layout.source_at_event), len(bank.query_ids))):
        raise ValueError("Complete observed maps/spans and fixed350 bank dimensions required")
    edges, statuses = [], []
    for tree, native_ids in zip(inner.native_trees, maps):
        keys, status = native_census_edges_v1(tree, native_ids, bank.query_ids)
        if "at_or_older_than_census" in status:
            raise ValueError("Dynamics leaves at/older than350 violate existing require_younger policy")
        edges.append(keys); statuses.append(status)
    events, bins, changes, eligible = [], [], [], []
    for event in range(1, bank.layout.num_segments):
        before, after = bank.layout.graph_at_segment[event - 1:event + 1]
        if before == after:
            continue
        if not float(spans[before, 1]) == float(spans[after, 0]):
            raise ValueError("Observed adjacent tree holds must be physically contiguous")
        known = [a is not None and b is not None for a, b in zip(edges[before], edges[after])]
        changes.append([ok and a != b for ok, a, b in zip(known, edges[before], edges[after])])
        eligible.append(known); events.append(event)
        width = float(spans[after, 1] - spans[after, 0])
        bins.append(bisect_right(SPAN_BIN_UPPER_BP, width))
    q = len(bank.query_ids)
    changed = torch.tensor(changes, dtype=torch.bool).reshape(-1, q)
    known = torch.tensor(eligible, dtype=torch.bool).reshape(-1, q)
    span_bins = torch.tensor(bins, dtype=torch.long)
    sham = sham_changes_v1(changed, known, span_bins, bank.query_ids,
                           parent_id=parent_id, namespace=namespace, tile_id=tile_id, seed=seed)
    def derived(mask):
        rho = torch.ones_like(bank.boundary_rho)
        if events:
            rho[list(events)] = torch.exp(-mask.to(dtype=rho.dtype))
        return replace(bank, boundary_rho=rho)
    def certified(arm, arm_bank):
        binding = {
            "schema": "census_lineage_memory_derivation_v1",
            "arm": arm, "seed": seed, "census": CENSUS,
            "implementation": source_binding, "experiment_config": config_binding,
            "source_observation_certificate_sha256": observation.certificate_sha256,
            "source_native_certificate_sha256": inner.certificate_sha256,
            "boundary_retention": experiment_config["boundary_retention"][arm],
            "edge_definition": experiment_config["edge_definition"],
            "missing_keys": experiment_config["missing_keys"],
            "sham": experiment_config["sham"],
        }
        provenance = dict(inner.provenance, census_lineage_memory_v1=binding)
        fresh_inner = replace(inner, bank=arm_bank, provenance=provenance,
                              certificate_sha256="")
        inner_hash = core.fingerprint_sha256_v1(core._content(fresh_inner))
        fresh_inner = replace(fresh_inner, certificate_sha256=inner_hash)
        core.validate_census_biological_certificate_v1(fresh_inner, inner_hash)
        fresh_lineage = dict(lineage, census_lineage_memory_v1=binding)
        outer_hash = core.fingerprint_sha256_v1(_outer_content_v1(fresh_inner, fresh_lineage))
        return CertifiedCensusTsinferObservationV1(fresh_inner, fresh_lineage, outer_hash)
    return LineageMemoryInputsV1(parent_id, namespace, tile_id, tuple(edges), tuple(statuses),
                                 tuple(events), span_bins, known, changed, sham,
                                 certified("current_jaccard", bank),
                                 certified("genuine_edge", derived(changed)),
                                 certified("sham_edge", derived(sham)))

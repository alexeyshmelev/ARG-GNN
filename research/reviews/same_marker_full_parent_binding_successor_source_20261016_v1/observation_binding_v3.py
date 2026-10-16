"""SOURCE-ONLY exact bank/roster/site binding for future root-owned integration.

No file reads or scientific execution at authoring. Chunked observation
comparisons are future root actions, not a training/benchmark or self-issued
resource admission. External refs must be authenticated by the existing root
reader/guard. A path/hash supplied to this module is not self-authenticating.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping
import hashlib
import json
import math

import torch
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.reviews.biological_operator_next_fresh_20261016_v1.packed_csr_v2.same_marker_packed_csr_v2 import PackedParentCPUV2, require, stamp

CORE_INPUTS = ("parent", "sample_index", "site_hold", "reference_class",
               "sample_age", "query_column", "genotype")
CLASSES = ("Ana", "CHG", "WHG", "EHG")


def bank_stamps(bank):
    require(type(bank) is MutationRoutedWholeTreeInputsV1, "Exact observation-only bank type required")
    out = []
    for field in fields(bank):
        value = getattr(bank, field.name)
        if isinstance(value, tuple):
            out.extend((field.name+":"+str(i), stamp(x)) for i, x in enumerate(value))
        else:
            out.append((field.name, stamp(value)))
    return tuple(out)


def compare_observation_tensor_chunks(cpu, device, *, chunk_elements):
    require(isinstance(cpu, torch.Tensor) and isinstance(device, torch.Tensor)
            and cpu.device.type == "cpu" and cpu.dtype == device.dtype
            and cpu.shape == device.shape and cpu.is_contiguous() and device.is_contiguous()
            and not cpu.requires_grad and not device.requires_grad,
            "Original contiguous observation shape/dtype/gradient contract differs")
    a, b = cpu.view(-1), device.view(-1)
    for start in range(0, a.numel(), chunk_elements):
        # At most chunk_elements*element_size additional CPU comparison bytes.
        # No extra full GPU bank or full comparison mirror is constructed.
        require(torch.equal(a[start:start+chunk_elements], b[start:start+chunk_elements].to("cpu")),
                "Device observation bytes differ from authenticated CPU bank")


def compare_original_bank_v2(cpu_bank, device_bank, *, chunk_elements):
    require(type(cpu_bank) is type(device_bank) is MutationRoutedWholeTreeInputsV1,
            "Exact original observation bank type required")
    for field in fields(cpu_bank):
        left, right = getattr(cpu_bank, field.name), getattr(device_bank, field.name)
        if isinstance(left, tuple):
            require(isinstance(right, tuple) and len(left) == len(right), "Original level tuple differs")
            for a, b in zip(left, right):
                compare_observation_tensor_chunks(a, b, chunk_elements=chunk_elements)
        else:
            compare_observation_tensor_chunks(left, right, chunk_elements=chunk_elements)


def validate_observed_roster_v2(roster, bank, permutation):
    """Bind column/source/date/query and complete diploid identities to metadata.

    This checks the producer-declared roster. Original genotype-column/Relate
    provenance still requires the external authenticated observed entry/manifest.
    """
    h = bank.sample_index.shape[1]
    require(isinstance(roster, (list, tuple)) and len(roster) == h,
            "Complete authenticated observed roster required")
    keys, queries, pairs = [], [], {}
    ref, ages = bank.reference_class.tolist(), bank.sample_age.tolist()
    for col, row in enumerate(roster):
        require(isinstance(row, Mapping) and row.get("sample_index") == col,
                "Exact observed roster column index required")
        sid, phase = row.get("sample_id"), row.get("haplotype")
        age, role, cls = row.get("age_generations"), row.get("role"), row.get("observed_reference_class")
        require(isinstance(sid, str) and sid and type(phase) is int and phase in (0, 1)
                and type(age) in (int, float) and math.isfinite(age) and age >= 0
                and role in ("query", "pure_source_reference", "mixed_reference_only"),
                "Observed sample/phase/date/role metadata invalid")
        rounded_age = float(torch.tensor(age, dtype=bank.sample_age.dtype))
        require(ages[col] == rounded_age, "Bank sample date differs from observed roster")
        if role == "pure_source_reference":
            require(type(cls) is int and 0 <= cls < 4 and row.get("observed_cohort") == CLASSES[cls]
                    and ref[col] == cls, "Only actual observed pure-reference classes may seed evidence")
        else:
            require(cls is None and ref[col] == -1, "Query/context ancestry MUST stay unknown")
        if role == "query":
            queries.append(col)
        require(phase not in pairs.setdefault(sid, {}), "Duplicate observed diploid phase")
        pairs[sid][phase] = col
        keys.append((sid, phase))
    require(len(set(keys)) == h and queries == bank.query_column.tolist(),
            "Original query order and sampled-haplotype identities differ")
    p = permutation.long().tolist()
    for sid, pair in pairs.items():
        require(set(pair) == {0, 1}, "Both original phases required for every diploid")
        i, j = pair[0], pair[1]
        a, b = roster[i], roster[j]
        require(all(a[k] == b[k] for k in ("age_generations", "role", "observed_reference_class", "observed_cohort")),
                "Original diploid metadata disagree")
        source0, source1 = keys[p[i]], keys[p[j]]
        require(source0[0] == source1[0] and source0[1] == 0 and source1[1] == 1,
                "Null must move complete phased diploid sequences")
    raw = json.dumps(list(roster), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return tuple(keys), hashlib.sha256(raw).hexdigest()


def reference_tuple(name, value):
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and value["path"]
            and isinstance(value["sha256"], str) and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"]),
            "Externally authenticated exact source/metadata ref required: "+name)
    return name, value["path"], value["sha256"]


@dataclass(frozen=True)
class PackedObservationBindingV3:
    route_identity: int
    route_global_query_stamp: tuple
    device_bank_stamps: tuple
    device_episode_stamps: tuple
    observed_roster_keys: tuple
    observed_roster_sha256: str
    external_refs: tuple
    chunk_elements: int

    def validate(self, bank, episode, route):
        require(id(route) == self.route_identity and bank_stamps(bank) == self.device_bank_stamps,
                "Payload no longer matches the EXACT bound current observation bank")
        require(episode_stamps(episode) == self.device_episode_stamps,
                "Original H045 episode payload changed")
        require(stamp(route.global_query_node) == self.route_global_query_stamp,
                "Packed concatenated original-query scatter rows changed")
        # O(T) structural metadata/versions only. No GPU value scans per step.
        for hold in route.holds:
            hold.verify_immutable()


def episode_fields(episode):
    return (("node_flags", episode.flags.node_flags),) + tuple((name, getattr(episode, name)) for name in
        ("member_node", "member_hold", "member_episode", "episode_size", "hold_span"))


def episode_stamps(episode):
    return tuple((name, stamp(value)) for name, value in episode_fields(episode))


def bind_original_query_scatter_rows_v3(route, cpu_bank, *, chunk_elements):
    """Bind hold-major query scatter rows to the original observed CPU bank.

    No full T*Q expected join is materialized. Each expected comparison tile
    and its int64-compatible actual tile contain at most chunk_elements items.
    These setup-only tiles are discarded. No extra persistent tensor is added.
    """
    rows = route.global_query_node
    t, _h = cpu_bank.sample_index.shape
    q = cpu_bank.query_column.numel()
    require(isinstance(rows, torch.Tensor) and rows.device.type == "cpu"
            and rows.dtype == torch.int32 and rows.ndim == 1
            and rows.is_contiguous() and not rows.requires_grad
            and t > 0 and q > 0 and rows.numel() == t*q,
            "Complete contiguous int32 CPU hold-major query scatter join required")
    before = stamp(rows)
    matrix = rows.view(t, q)
    query = cpu_bank.query_column.long()
    for qstart in range(0, q, chunk_elements):
        qstop = min(q, qstart+chunk_elements)
        hold_chunk = max(1, chunk_elements // (qstop-qstart))
        for start in range(0, t, hold_chunk):
            stop = min(t, start+hold_chunk)
            expected = cpu_bank.sample_index[start:stop].index_select(1, query[qstart:qstop])
            actual = matrix[start:stop, qstart:qstop].to(dtype=expected.dtype)
            require(torch.equal(actual, expected),
                    "Packed concatenated scatter rows differ from original observed hold/query order")
    require(stamp(rows) == before, "Packed query scatter rows mutated during initial binding")
    return before


def bind_current_observations_v3(*, route, route_inputs_cpu, cpu_bank, device_bank,
                                 cpu_episode, device_episode, observed_roster,
                                 observed_entry_ref, observed_cache_ref, site_sidecar_ref,
                                 root_full_parent_layout_resource_proof_ref,
                                 chunk_elements=1048576):
    """Future root-only setup after original bank/episode transfer and proof.

    Exact byte comparisons bind topology, original site/genotype order, every
    branch support and feature field, roles/classes/dates/query row order, and
    original episode tensors. No full duplicate bank is transferred to GPU.
    The resource proof ref is mandatory, but its authentication/execution power
    belongs to the external root guard, never this metadata helper.
    """
    require(isinstance(route, PackedParentCPUV2) and isinstance(route_inputs_cpu, Mapping)
            and type(chunk_elements) is int and 1 <= chunk_elements <= 1048576,
            "Exact supplied CPU route and bounded comparison chunks required")
    route.verify_for_supplied_cpu_bank(**route_inputs_cpu)
    require(type(cpu_bank) is type(device_bank) is MutationRoutedWholeTreeInputsV1,
            "Exact original observation bank types required")
    for key in CORE_INPUTS:
        require(stamp(route_inputs_cpu[key]) == stamp(getattr(cpu_bank, key)),
                "Packed layout derives from a DIFFERENT original CPU field: "+key)
    require(len(route.holds) == cpu_bank.sample_index.shape[0], "All original holds required")
    cpu_bank_before, device_bank_before = bank_stamps(cpu_bank), bank_stamps(device_bank)
    cpu_episode_before, device_episode_before = episode_stamps(cpu_episode), episode_stamps(device_episode)
    query_scatter_stamp = bind_original_query_scatter_rows_v3(
        route, cpu_bank, chunk_elements=chunk_elements)
    permutation = route_inputs_cpu.get("route_permutation", torch.arange(cpu_bank.sample_index.shape[1]))
    keys, roster_hash = validate_observed_roster_v2(observed_roster, cpu_bank, permutation)
    compare_original_bank_v2(cpu_bank, device_bank, chunk_elements=chunk_elements)
    left, right = episode_fields(cpu_episode), episode_fields(device_episode)
    require([name for name, _ in left] == [name for name, _ in right], "Original episode field contract differs")
    for (_, a), (_, b) in zip(left, right):
        compare_observation_tensor_chunks(a, b, chunk_elements=chunk_elements)
    refs = tuple(reference_tuple(k, v) for k, v in (
        ("observed_entry", observed_entry_ref), ("observed_cache", observed_cache_ref),
        ("site_sidecar", site_sidecar_ref), ("root_full_parent_layout_resource_proof", root_full_parent_layout_resource_proof_ref)))
    # Preserve all original version guards across setup, including the new
    # concatenated scatter buffer. These are metadata checks, not value scans.
    require(bank_stamps(cpu_bank) == cpu_bank_before and bank_stamps(device_bank) == device_bank_before,
            "Original CPU/device observation bank mutated during binding")
    require(episode_stamps(cpu_episode) == cpu_episode_before
            and episode_stamps(device_episode) == device_episode_before,
            "Original CPU/device episode payload mutated during binding")
    route.verify_for_supplied_cpu_bank(**route_inputs_cpu)
    require(stamp(route.global_query_node) == query_scatter_stamp,
            "Packed query scatter rows changed during observation binding")
    return PackedObservationBindingV3(id(route), query_scatter_stamp, device_bank_before,
                                     device_episode_before, keys, roster_hash, refs, chunk_elements)

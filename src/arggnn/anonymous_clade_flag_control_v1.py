"""Exact observation-only changed-clade feature control. Source-only API.

Proper sampled-leaf clades are compared by ephemeral exact integer bitsets.
No ancestral-node correspondence, target reader, disk bitset cache or launcher
is provided. Root owns observation lineage, registration and future execution.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .anonymous_clade_gnn_v1 import AnonymousCladeTensorGNNV1, NEW_KEYS
from .mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
from .mutation_routed_whole_tree_gnn_v1 import (
    MutationRoutedWholeTreeInputsV1, _FLOAT_FIELDS, _LEVEL_FIELDS,
)

SOURCE = "src/arggnn/anonymous_clade_flag_control_v1.py"
SCHEMA = "anonymous_clade_node_flags_v1"
CHANNELS = ("gained_from_left", "lost_to_right")
ARM = "exact_clade_flags"


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _cpu_index_array(value):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.requires_grad or value.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Read-only CPU observed integer indexes required")
    return value.detach().numpy()


def _tree_clades(root, row, parent, head, sibling):
    """Only one tree's exact masks exist here; duplicate unary clades collapse."""
    sampled = {int(node): 1 << column for column, node in enumerate(row)}
    if len(sampled) != len(row):
        raise ValueError("Distinct sampled leaves in every native tree required")
    preorder, stack = [], [root]
    while stack:
        node = stack.pop()
        preorder.append(node)
        child = int(head[node])
        while child >= 0:
            stack.append(child)
            child = int(sibling[child])
    masks, clades = {}, {}
    unary_duplicates = 0
    leaves = 0
    full_mask = (1 << len(row)) - 1
    for node in reversed(preorder):
        child = int(head[node])
        if child < 0:
            if node not in sampled:
                raise ValueError("Every native terminal node must be in the sampled panel")
            mask = sampled[node]
            leaves += 1
            child_count = 0
        else:
            if node in sampled:
                raise ValueError("Sampled-interior cases are unsupported")
            mask, child_count = 0, 0
            while child >= 0:
                mask |= masks[child]
                child_count += 1
                child = int(sibling[child])
        masks[node] = mask
        if 1 < mask.bit_count() < len(row):
            if mask in clades:
                # Postorder already encountered the deepest representative.
                if child_count != 1 or masks[int(head[node])] != mask:
                    raise ValueError("Equal proper clades must form a unary chain")
                unary_duplicates += 1
            else:
                clades[mask] = node
    if leaves != len(row) or masks[root] != full_mask:
        raise ValueError("Each native tree must contain exactly the full sampled panel")
    return clades, len(preorder), unary_duplicates


def build_clade_flag_arrays_v1(parent, sample_index):
    """Pure structural CPU preparation, also permitting a tiny synthetic panel.

    Returns only [N,2] uint8 flags and compact metadata. Exact masks are local
    to the previous/current tree and are never retained in a returned payload.
    Stable sample columns are an input contract, certified by root's reader.
    """
    parent, samples = np.asarray(parent), np.asarray(sample_index)
    if (parent.ndim != 1 or not parent.size or parent.size >= 2**31
            or parent.dtype.kind != "i" or np.any(parent < -1)
            or np.any(parent >= parent.size)):
        raise ValueError("Bounded signed-integer observed parent forest required")
    if (samples.ndim != 2 or samples.dtype.kind != "i"
            or samples.shape[0] < 1 or samples.shape[1] < 3
            or np.any(samples < 0) or np.any(samples >= parent.size)):
        raise ValueError("Complete native sampled-leaf index rows required")
    n, (t, h) = parent.size, samples.shape
    roots = np.flatnonzero(parent < 0)
    if len(roots) != t:
        raise ValueError("Exactly one disjoint native tree per hold required")
    # Two compact native integer arrays, no global array of exact clade masks.
    head = np.full(n, -1, dtype=np.int32)
    sibling = np.full(n, -1, dtype=np.int32)
    for node, ancestor in enumerate(parent):
        if ancestor >= 0:
            sibling[node] = head[ancestor]
            head[ancestor] = node
    ordered_roots, used_roots = [], set()
    for row in samples:
        node, chain = int(row[0]), set()
        while parent[node] >= 0:
            if node in chain:
                raise ValueError("Acyclic observed parent forest required")
            chain.add(node)
            node = int(parent[node])
        if node in used_roots:
            raise ValueError("Every hold must contain a distinct native tree")
        used_roots.add(node)
        ordered_roots.append(node)

    flags = np.zeros((n, 2), dtype=np.uint8)
    previous = None
    visited_nodes = 0
    clade_counts, node_counts, unary_counts = [], [], []
    for root, row in zip(ordered_roots, samples):
        current, count, unary = _tree_clades(root, row, parent, head, sibling)
        visited_nodes += count
        node_counts.append(count)
        clade_counts.append(len(current))
        unary_counts.append(unary)
        if previous is not None:
            for mask, representative in previous.items():
                if mask not in current:
                    flags[representative, 1] = 1
            for mask, representative in current.items():
                if mask not in previous:
                    flags[representative, 0] = 1
        previous = current
    if visited_nodes != n:
        raise ValueError("No cyclic, omitted or unused native forest vertices allowed")
    # No exact masks, clade sets or inferred endpoint pairs escape this function.
    del previous
    meta = dict(
        schema=SCHEMA, channels=list(CHANNELS), native_holds=int(t),
        haplotypes=int(h), native_nodes=int(n), node_flag_shape=[int(n), 2],
        node_flag_dtype="uint8", node_flag_bytes=int(flags.nbytes),
        topology_sha256=_sha(parent), sample_index_sha256=_sha(samples),
        node_flags_sha256=_sha(flags), proper_clades_per_tree=clade_counts,
        native_nodes_per_tree=node_counts, unary_equivalences_per_tree=unary_counts,
        gained_from_left_count=int(flags[:, 0].sum()),
        lost_to_right_count=int(flags[:, 1].sum()),
        representative_policy="deepest_structural_node_in_equal_clade_unary_chain",
        exact_bitsets_retained=False, internal_ancestor_correspondence_claimed=False,
        observation_only=True, labels_times_targets_or_outcomes_used=False,
    )
    return flags, meta


@dataclass(frozen=True)
class AnonymousCladeFlagsV1:
    node_flags: torch.Tensor
    native_holds: int
    haplotypes: int
    native_nodes: int

    @classmethod
    def from_verified_arrays(cls, flags, meta):
        n, t, h = meta["native_nodes"], meta["native_holds"], meta["haplotypes"]
        if (meta.get("schema") != SCHEMA or meta.get("channels") != list(CHANNELS)
                or type(n) is not int or type(t) is not int or type(h) is not int
                or n < 1 or t < 1 or h < 3
                or not isinstance(flags, np.ndarray) or flags.dtype != np.uint8
                or flags.shape != (n, 2) or np.any(flags > 1)
                or _sha(flags) != meta.get("node_flags_sha256")
                or meta.get("exact_bitsets_retained") is not False
                or meta.get("internal_ancestor_correspondence_claimed") is not False
                or meta.get("labels_times_targets_or_outcomes_used") is not False):
            raise ValueError("Verified compact observation-only exact-clade flags required")
        return cls(torch.from_numpy(flags), t, h, n)

    def to(self, device, *, non_blocking=False):
        return type(self)(
            self.node_flags.to(device, non_blocking=non_blocking),
            self.native_holds, self.haplotypes, self.native_nodes,
        )

    def tensor_bytes(self):
        return self.node_flags.numel() * self.node_flags.element_size()


def build_clade_flags_v1(bank):
    """Actual source-only integration API. Root supplies the original CPU bank."""
    if type(bank) is not MutationRoutedWholeTreeInputsV1:
        raise TypeError("Exact original observation-only bank required")
    parent = _cpu_index_array(bank.parent)
    samples = _cpu_index_array(bank.sample_index)
    if samples.ndim != 2 or samples.shape[1] != 952 or bank.query_column.shape != (686,):
        raise ValueError("Original full952/all686query native bank required")
    arrays, meta = build_clade_flag_arrays_v1(parent, samples)
    return AnonymousCladeFlagsV1.from_verified_arrays(arrays, meta), meta


def validate_clade_flags_v1(flags, upward):
    if not isinstance(flags, AnonymousCladeFlagsV1):
        raise TypeError("Verified explicit clade-flag payload required")
    if (upward.dtype != torch.float32 or upward.ndim != 2
            or upward.shape != (flags.native_nodes, 32)
            or flags.node_flags.shape != (flags.native_nodes, 2)
            or flags.node_flags.dtype != torch.uint8
            or flags.node_flags.device != upward.device
            or flags.node_flags.requires_grad):
        raise ValueError("FP32 node states and compact same-device flag payload required")
    # Values/content and topology lineage are preparation-time proofs, not GPU scans.


class RankEightCladeFlagAdapterV1(nn.Module):
    """Local 1,024-weight feature adapter. It never gathers another tree's state."""
    def __init__(self):
        super().__init__()
        self.U = nn.Linear(32, 8, bias=False)
        self.V = nn.Linear(32, 8, bias=False)
        self.W = nn.Linear(32, 8, bias=False)
        self.O = nn.Linear(8, 32, bias=False)
        nn.init.zeros_(self.O.weight)

    def rank_residual(self, upward, flags):
        validate_clade_flags_v1(flags, upward)
        if any(p.dtype != torch.float32 or p.device != upward.device
               for p in self.parameters()):
            raise ValueError("FP32 same-device local adapter parameters required")
        channels = flags.node_flags.to(dtype=upward.dtype)
        common = self.U(upward)
        return (channels[:, 0:1] * common * self.V(upward)
                - channels[:, 1:2] * common * self.W(upward))

    def forward(self, upward, flags):
        return self.O(self.rank_residual(upward, flags))


class AnonymousCladeFlagControlGNNV1(MutationRoutedFullPanelPyGGNNV1):
    """Same29,764-weight body plus local1,024-weight adapter, two full PyG sweeps."""
    # Reuse frozen scientific operations without altering their source/state keys.
    prepare_tree_base_v1 = AnonymousCladeTensorGNNV1.prepare_tree_base_v1
    sweep_tree_state_v1 = AnonymousCladeTensorGNNV1.sweep_tree_state_v1
    leaf_context_v1 = AnonymousCladeTensorGNNV1.leaf_context_v1

    def __init__(self, hidden_dim=32, *, expected_haplotypes=952,
                 site_chunk_size=256, checkpoint_sites=True):
        if hidden_dim != 32 or expected_haplotypes != 952:
            raise ValueError("Frozen width32/all952 H038 body required")
        super().__init__(
            hidden_dim=hidden_dim, expected_haplotypes=expected_haplotypes,
            site_chunk_size=site_chunk_size, checkpoint_sites=checkpoint_sites,
        )
        self.clade_coupling = RankEightCladeFlagAdapterV1()

    def encode_trees(self, bank, *, flags, control="aligned"):
        base, ug, dg, degree = self.prepare_tree_base_v1(bank, control=control)
        upward, _outside = self.sweep_tree_state_v1(bank, base, ug, dg, degree)
        injection = self.clade_coupling(upward, flags)
        second_upward, second_outside = self.sweep_tree_state_v1(
            bank, base + injection, ug, dg, degree,
        )
        return self.leaf_context_v1(bank, second_upward, second_outside)

    def forward_holds(self, bank, *, flags, control="aligned"):
        parameter = next(self.parameters())
        t, h = bank.sample_index.shape
        if (parameter.dtype != torch.float32 or h != 952
                or bank.query_column.numel() != 686 or not t
                or bank.parent.device != parameter.device
                or flags.native_holds != t or flags.haplotypes != h):
            raise ValueError("Complete original FP32 native bank and matching flags required")
        for name in _FLOAT_FIELDS:
            value = getattr(bank, name)
            if value.dtype != torch.float32 or value.device != parameter.device:
                raise ValueError("Original floating observations/device differ")
        for f in fields(bank):
            value = getattr(bank, f.name)
            values = value if f.name in _LEVEL_FIELDS else (value,)
            if any(v.device != parameter.device for v in values):
                raise ValueError("Complete observation bank device differs")
        context = self.encode_trees(bank, flags=flags, control=control)
        span = bank.spans.log1p() / math.log(2000001.)
        prev = torch.cat((
            bank.spans.new_zeros(1), .5 * (bank.spans[1:] + bank.spans[:-1]),
        ))
        nxt = torch.cat((
            .5 * (bank.spans[1:] + bank.spans[:-1]), bank.spans.new_zeros(1),
        ))
        geometry = torch.stack((
            span, prev.log1p() / math.log(2000001.),
            nxt.log1p() / math.log(2000001.),
        ), -1)
        sequence = torch.cat((context, geometry[:, None].expand(t, h, 3)), -1)
        history, _ = self.genomic_gru(sequence)
        return self.classifier(torch.cat((context, history), -1))

    def forward_train_holds(self, bank, *, flags, control="aligned"):
        return self.forward_holds(bank, flags=flags, control=control).index_select(
            1, bank.query_column,
        )

    def forward(self, bank, *, flags, control="aligned"):
        return self.forward_train_holds(bank, flags=flags, control=control).index_select(
            0, bank.readout_hold,
        )


def load_h038_flag_warmstart_v1(model, payload):
    """Same selected aligned/seed101/teacher-free H038 payload schema. No reader."""
    if (not isinstance(model, AnonymousCladeFlagControlGNNV1)
            or payload.get("arm") != "aligned" or payload.get("seed") != 101
            or payload.get("teacher_used") is not False):
        raise ValueError("Original selected alignedH038/seed101/teacher-free schema required")
    state = payload["model_state"]
    if any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32
           for v in state.values()):
        raise ValueError("Original FP32 learned backbone weights required")
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != NEW_KEYS or result.unexpected_keys:
        raise ValueError("Only the same four bias-free rank8 projection keys may be missing")
    if sum(p.numel() for p in model.parameters()) != 30788:
        raise ValueError("29,764+1,024 learned parameters required")
    if bool(model.clade_coupling.O.weight.detach().count_nonzero()):
        raise ValueError("Zero output projection initialization required")
    return dict(
        arm=ARM, base_parameter_count=29764, added_parameter_count=1024,
        total_parameter_count=30788, output_projection_initialized_zero=True,
        first_update_new_core_projection_gradients_zero_until_O_moves=True,
        optimizer_policy="root_must_reset_moments_and_step_to_match_registered_controls",
        original_h038_payload_schema=True,
    )

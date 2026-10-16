"""SOURCE-ONLY H045 integration for packed same-marker evidence.

No future full-parent use before independent review and root layout-resource
proof/admission. This file contains no launcher, reader or resource authority.
Standalone bounded one-hold qualification can use the evidence module directly.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import torch

from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodeGNNV1, OrderedCladeEpisodesV1
from .same_marker_packed_csr_v2 import PackedParentCPUV2, PackedSameMarkerAdapterV2
from .observation_binding_v2 import PackedObservationBindingV2


@dataclass(frozen=True)
class PackedSameMarkerPayloadV2:
    episodes: OrderedCladeEpisodesV1
    route_cpu: PackedParentCPUV2
    binding: PackedObservationBindingV2

    @property
    def native_holds(self):
        return self.episodes.native_holds

    @property
    def haplotypes(self):
        return self.episodes.haplotypes

    @property
    def native_nodes(self):
        return self.episodes.native_nodes

    def to(self, device):
        raise RuntimeError("Transfer original bank/episodes first, then bind this exact device payload; new route stays CPU")


class PackedSameMarkerSupportGNNV2(OrderedCladeEpisodeGNNV1):
    """Known H045 own-state body plus same714 additional parameters asv1."""
    def __init__(self, hidden_dim=32, *, expected_haplotypes=952,
                 site_chunk_size=256, checkpoint_sites=True, marker_block_size=16):
        super().__init__(hidden_dim=hidden_dim, expected_haplotypes=expected_haplotypes,
                         site_chunk_size=site_chunk_size, checkpoint_sites=checkpoint_sites,
                         arm="own_state")
        self.same_marker = PackedSameMarkerAdapterV2(marker_block_size)
        self.float()

    def encode_trees(self, bank, *, flags, control="aligned"):
        if not isinstance(flags, PackedSameMarkerPayloadV2) or control != "aligned":
            raise ValueError("Original aligned backbone and packed CPU observation route required")
        episodes, route = flags.episodes, flags.route_cpu
        if not isinstance(flags.binding, PackedObservationBindingV2):
            raise ValueError("Exact current-bank/roster/sites binding required")
        flags.binding.validate(bank, episodes, route)
        if (len(route.holds) != bank.sample_index.shape[0]
                or bank.spans.shape != episodes.hold_span.shape
                or not torch.equal(bank.spans, episodes.hold_span)
                or route.global_query_node.device.type != "cpu"):
            raise ValueError("Original complete hold joins and CPU-only packed layout required")
        base, ug, dg, degree = self.prepare_tree_base_v1(bank, control=control)
        upward, _outside = self.sweep_tree_state_v1(bank, base, ug, dg, degree)
        injection = (self.clade_coupling(upward, episodes.flags)
                     + self.episode_coupling(upward, episodes, arm="own_state"))
        residuals = []
        for cpu_hold in route.holds:
            residuals.append(self.same_marker.forward_native_hold_v2(
                cpu_hold, upward, bank.edge_geometry, bank.branch_support_true, bank.genotype))
        # Only one packed query-row vector is copied outside checkpoint, not the
        # whole-treebank CSR. Count this O(T*Q) vector and both residual copies.
        rows = route.global_query_node.to(device=upward.device, dtype=torch.long)
        injection = injection.index_add(0, rows, torch.cat(residuals, 0))
        second_upward, second_outside = self.sweep_tree_state_v1(
            bank, base+injection, ug, dg, degree)
        return self.leaf_context_v1(bank, second_upward, second_outside)

    def forward_holds(self, bank, *, episodes, control="aligned"):
        if not isinstance(episodes, PackedSameMarkerPayloadV2):
            raise ValueError("Exact packed route payload required")
        return super().forward_holds(bank, episodes=episodes, control=control)


def initialize_supplied_h045_state_v2(model, state):
    """Source-only future helper: no file read and no provenance certification."""
    if not isinstance(model, PackedSameMarkerSupportGNNV2) or not isinstance(state, Mapping):
        raise ValueError("Exact new class and externally authenticated H045 state required")
    expected = model.state_dict()
    added = {key for key in expected if key.startswith("same_marker.")}
    if set(state) != set(expected)-added:
        raise ValueError("Complete original H045 state keys required")
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.float32 or value.shape != expected[key].shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Original state differs: "+key)
    if (sum(p.numel() for p in model.parameters()) != 32526
            or bool(model.same_marker.output.weight.count_nonzero())
            or not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())):
        raise ValueError("Expected32526 trainable FP32 weights and zero new output")
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != added or result.unexpected_keys:
        raise ValueError("Only the same714 new parameters may be absent")
    return dict(additional_parameters=714, allocated_parameters=32526,
                initial_prediction_equality_numerically_unverified=True,
                optimizer_constructed=False, checkpoint_provenance_verified=False)

"""SOURCE-ONLY H045 integration. No loaders, launchers, targets or training.

Place this review folder on a future root-controlled module path/package only
following independent review and source pinning. Authoring did not import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from arggnn.ordered_clade_episode_gnn_v1 import (
    OrderedCladeEpisodeGNNV1, OrderedCladeEpisodesV1,
)
from .same_marker_allele_support_v1 import (
    HoldLayoutV1, SameMarkerAlleleSupportAdapterV1,
)


@dataclass(frozen=True)
class SameMarkerPayloadV1:
    episodes: OrderedCladeEpisodesV1
    layouts: tuple[HoldLayoutV1, ...]

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
        return type(self)(self.episodes.to(device), tuple(x.to(device) for x in self.layouts))


class SameMarkerSupportGNNV1(OrderedCladeEpisodeGNNV1):
    """Same H045 own-state body with one additional pre-second-sweep residual.

    Aligned and null use the exact same class/weights. ONLY the observation-only
    payload route_permutation differs. The full native-hold decoder is inherited.
    """
    def __init__(self, hidden_dim=32, *, expected_haplotypes=952,
                 site_chunk_size=256, checkpoint_sites=True, marker_block_size=16):
        super().__init__(hidden_dim=hidden_dim, expected_haplotypes=expected_haplotypes,
                         site_chunk_size=site_chunk_size, checkpoint_sites=checkpoint_sites,
                         arm="own_state")
        self.same_marker = SameMarkerAlleleSupportAdapterV1(marker_block_size)
        self.float()

    def encode_trees(self, bank, *, flags, control="aligned"):
        if not isinstance(flags, SameMarkerPayloadV1) or control != "aligned":
            raise ValueError("Exact new observation payload and unchanged aligned backbone required")
        episodes, layouts = flags.episodes, flags.layouts
        if (len(layouts) != bank.sample_index.shape[0]
                or bank.spans.shape != episodes.hold_span.shape
                or not torch.equal(bank.spans, episodes.hold_span)):
            raise ValueError("Complete original hold/layout/episode span joins required")
        # Original H045 source computation, with the full learned flag and
        # own-state episode adapters retained and all original weights trainable.
        base, ug, dg, degree = self.prepare_tree_base_v1(bank, control=control)
        upward, _outside = self.sweep_tree_state_v1(bank, base, ug, dg, degree)
        injection = (self.clade_coupling(upward, episodes.flags)
                     + self.episode_coupling(upward, episodes, arm="own_state"))
        rows, values = [], []
        for layout in layouts:
            nodes = layout.native_node
            edge9 = torch.cat((bank.edge_geometry.index_select(0, nodes),
                               bank.branch_support_true.index_select(0, nodes)), -1)
            values.append(self.same_marker(
                layout, bank.genotype[layout.site_start:layout.site_stop],
                edge9, upward.index_select(0, nodes)))
            rows.append(nodes.index_select(0, layout.query_node))
        # One global scatter, rather than copying the entire forest per hold.
        injection = injection.index_add(0, torch.cat(rows), torch.cat(values))
        second_upward, second_outside = self.sweep_tree_state_v1(
            bank, base + injection, ug, dg, degree)
        return self.leaf_context_v1(bank, second_upward, second_outside)

    def forward_holds(self, bank, *, episodes, control="aligned"):
        """episodes is SameMarkerPayloadV1, matching inherited runner signature."""
        if not isinstance(episodes, SameMarkerPayloadV1):
            raise ValueError("Verified same-marker payload required")
        # Parent forward_holds supplies flags to encode_trees above and retains
        # all original geometry, full-parent BiGRU, classifier and output mapping.
        return super().forward_holds(bank, episodes=episodes, control=control)


def initialize_supplied_h045_state_v1(model, state):
    """Future root supplies verified CPU model_state; this function reads no file.

    Does not certify checkpoint identity, completion, optimizer schedule or input
    provenance. Those are separate immutable source/receipt bindings owned by root.
    """
    if not isinstance(model, SameMarkerSupportGNNV1) or not isinstance(state, Mapping):
        raise ValueError("Exact new class and supplied original H045 state mapping required")
    expected = model.state_dict()
    added = {key for key in expected if key.startswith("same_marker.")}
    if set(state) != set(expected) - added:
        raise ValueError("Complete original H045 body/flag/episode state required")
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.float32 or value.shape != expected[key].shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Original finite CPU FP32 state/shape differs: " + key)
    if (sum(p.numel() for p in model.parameters()) != 32526
            or bool(model.same_marker.output.weight.count_nonzero())
            or not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())):
        raise ValueError("Expected31812+714 trainable FP32 weights and zero new output")
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != added or result.unexpected_keys:
        raise ValueError("Only additional same-marker weights may be missing")
    return dict(source_only=True, original_h045_weights_preserved=True,
                initial_prediction_equality_not_numerically_tested=True,
                new_output_zero=True, allocated_parameters=32526,
                additional_parameters=714, optimizer_constructed=False)

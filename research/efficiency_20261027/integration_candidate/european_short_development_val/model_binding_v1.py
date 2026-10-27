"""Inactive H050 stable-leaf TreeLoom with explicit European panel/length binding.

Retains H050's trunk state keys, one-layer bidirectional GRU, direct classifier,
two shared full-tree sweeps, local flag/own-state adapters and513-scalar memory.
No file, checkpoint, target, guard, admission, launcher or CLI is provided.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
import torch
from torch import nn
from arggnn.mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1, _FLOAT_FIELDS, _LEVEL_FIELDS
from arggnn.anonymous_clade_gnn_v1 import AnonymousCladeTensorGNNV1
from arggnn.anonymous_clade_flag_control_v1 import RankEightCladeFlagAdapterV1
from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1, RankEightCladeEpisodeAdapterV1
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as memory_core

CONDITIONS = {
    "AP_short10Mb_development_native6": dict(haplotypes=952, queries=686,
        length_bp=10000000, output_classes=("Ana", "CHG", "WHG", "EHG", "Ana_via_BAA", "CHG_via_BAA"),
        parameters=32391),
}
CLASS_ORDER = ("Ana", "CHG", "WHG", "EHG")
PARAMETERS = 32325


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _bank_tensors(bank):
    values = []
    for f in fields(bank):
        v = getattr(bank, f.name)
        values.extend(v if isinstance(v, tuple) else (v,))
    return tuple(values)


def _episode_tensors(episodes):
    return (episodes.flags.node_flags,) + tuple(getattr(episodes, name) for name in
        ("member_node", "member_hold", "member_episode", "episode_size", "hold_span"))


@dataclass(frozen=True)
class ShortDevelopmentBundleV1:
    bank: MutationRoutedWholeTreeInputsV1
    episodes: OrderedCladeEpisodesV1
    cut: memory_core.CutPlanV1
    condition: str
    parent_id: str
    split: str
    source_refs: dict
    bank_stamps: tuple
    episode_stamps: tuple

    def validate(self, bank=None):
        need(self.condition in CONDITIONS, "Exact declared European condition")
        c = CONDITIONS[self.condition]
        need(type(self.bank) is MutationRoutedWholeTreeInputsV1
             and (bank is None or bank is self.bank)
             and isinstance(self.episodes, OrderedCladeEpisodesV1),
             "Exact complete observation bank and structural recipes")
        need(self.bank_stamps == memory_core._stamps(_bank_tensors(self.bank))
             and self.episode_stamps == memory_core._stamps(_episode_tensors(self.episodes)),
             "Immutable complete observation and episode tensors")
        t, h = self.bank.sample_index.shape
        need(h == c["haplotypes"] and t > 0
             and self.bank.query_column.shape == (c["queries"],)
             and self.bank.log_donor_allele_frequency.shape[-1] == 8
             and self.episodes.native_holds == t
             and self.episodes.haplotypes == h
             and self.episodes.native_nodes == self.bank.parent.numel()
             and torch.equal(self.episodes.hold_span, self.bank.spans),
             "Full panel, query and native structural support")
        # Native Relate support can end before requestedL. Never extend its tail.
        length = float(self.bank.spans.double().sum())
        need(0 < length <= c["length_bp"],
             "Declared genuine native physical support; no invented coverage")
        need(self.split == "val_prediction"
             and isinstance(self.source_refs, dict)
             and self.source_refs.get("query_truth_in_inputs") is False
             and self.source_refs.get("all_observed_trees_nodes_sites_samples_retained") is True
             and self.source_refs.get("parent_id") == self.parent_id
             and self.source_refs.get("supported_length_bp") == length
             and self.source_refs.get("requested_length_bp") == c["length_bp"],
             "ROOT-authenticated observation-only complete parent identity/support")
        self.cut.validate(self.bank)
        return self.bank

    def to(self, device):
        self.validate()
        bank = self.bank.to(device)
        episodes = self.episodes.to(device)
        cut = self.cut._to_for_bank(device, self.bank, bank)
        result = type(self)(bank, episodes, cut, self.condition, self.parent_id,
            self.split, dict(self.source_refs), memory_core._stamps(_bank_tensors(bank)),
            memory_core._stamps(_episode_tensors(episodes)))
        result.validate()
        return result


class ShortDevelopmentTreeLoomV1(MutationRoutedFullPanelPyGGNNV1):
    prepare_tree_base_v1 = AnonymousCladeTensorGNNV1.prepare_tree_base_v1
    sweep_tree_state_v1 = AnonymousCladeTensorGNNV1.sweep_tree_state_v1
    leaf_context_v1 = AnonymousCladeTensorGNNV1.leaf_context_v1

    def __init__(self, *, condition, checkpoint_memory=True):
        need(condition in CONDITIONS, "Declared data binding only; no model search")
        super().__init__(hidden_dim=32,
            expected_haplotypes=CONDITIONS[condition]["haplotypes"],
            site_chunk_size=256, checkpoint_sites=True)
        self.condition = condition
        self.output_classes = tuple(CONDITIONS[condition]["output_classes"])
        # Observation donors/one-hot stay four founder populations. Native AP
        # predicts six census350 paths; only the final emission width changes.
        if len(self.output_classes) == 6:
            self.classifier[2] = nn.Linear(32, 6)
        self.arm = "own_state"
        self.clade_coupling = RankEightCladeFlagAdapterV1()
        self.episode_coupling = RankEightCladeEpisodeAdapterV1()
        self.memory = memory_core.CutMemoryV1(
            arm="stable_leaf_memory", checkpoint_memory=checkpoint_memory)
        self.float()
        need(sum(p.numel() for p in self.parameters()) == CONDITIONS[condition]["parameters"],
             "Exact frozen trunk plus task-required output width; no panel-size parameters")

    def encode_trees(self, bank, *, flags, control="aligned"):
        bundle = flags
        need(type(bundle) is ShortDevelopmentBundleV1 and bundle.condition == self.condition,
             "Exact bound European parent")
        bundle.validate(bank)
        episodes = bundle.episodes
        base, ug, dg, degree = self.prepare_tree_base_v1(bank, control=control)
        upward, outside = self.sweep_tree_state_v1(bank, base, ug, dg, degree)
        injection = (self.clade_coupling(upward, episodes.flags)
            + self.episode_coupling(upward, episodes, arm="own_state"))
        delta = self.memory(upward, outside, bank.spans, bundle.cut)
        injection = injection.index_add(0, bundle.cut.representatives, delta)
        second_up, second_out = self.sweep_tree_state_v1(
            bank, base + injection, ug, dg, degree)
        return self.leaf_context_v1(bank, second_up, second_out)

    def forward_holds(self, bank, *, episodes, control="aligned"):
        parameter = next(self.parameters())
        t, h = bank.sample_index.shape
        need(parameter.dtype == torch.float32 and bank.parent.device == parameter.device
             and h == CONDITIONS[self.condition]["haplotypes"],
             "Complete FP32 panel on one ROOT-admitted device")
        for name in _FLOAT_FIELDS:
            value = getattr(bank, name)
            need(value.dtype == torch.float32 and value.device == parameter.device,
                 "Original floating observation dtype/device")
        for f in fields(bank):
            value = getattr(bank, f.name)
            values = value if f.name in _LEVEL_FIELDS else (value,)
            need(all(v.device == parameter.device for v in values),
                 "Complete same-device observation tensors")
        context = self.encode_trees(bank, flags=episodes, control=control)
        span = bank.spans.log1p() / math.log(2000001.)
        prev = torch.cat((bank.spans.new_zeros(1), .5 * (bank.spans[1:] + bank.spans[:-1])))
        nxt = torch.cat((.5 * (bank.spans[1:] + bank.spans[:-1]), bank.spans.new_zeros(1)))
        geometry = torch.stack((span, prev.log1p() / math.log(2000001.),
            nxt.log1p() / math.log(2000001.)), -1)
        sequence = torch.cat((context, geometry[:, None].expand(t, h, 3)), -1)
        history, _ = self.genomic_gru(sequence)
        return self.classifier(torch.cat((context, history), -1))

    def forward_train_holds(self, bank, *, episodes, control="aligned"):
        return self.forward_holds(bank, episodes=episodes, control=control).index_select(
            1, bank.query_column)

    def forward(self, bank, *, episodes, control="aligned"):
        return self.forward_train_holds(bank, episodes=episodes, control=control).index_select(
            0, bank.readout_hold)

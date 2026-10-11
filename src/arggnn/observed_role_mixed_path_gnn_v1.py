"""FP32 prototype of parallel observed-role mixed paths on complete H030 trees.

A_new T - T A_old and its sum control are established sparse path operations.
Observed role matches are not true ancestral identity. No retained recurrence,
new-operator priority, training entry point, data loader or GPU qualification.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor, nn

from .persistence_trace_kernel_v1 import MetricTreeHoldV1, _tree
from .reference_logcavity_gnn_v1 import (
    ReferenceLogCavityBankV1, ReferenceLogCavityGNNV1,
    prepare_reference_logcavity_bank_v1,
)

ARMS = ("signed_difference", "path_sum")


@dataclass(frozen=True)
class ObservedRoleMixedPathBankV1:
    reference: ReferenceLogCavityBankV1
    edge_children: Tensor
    edge_parents: Tensor
    left_source: Tensor
    left_target: Tensor
    right_source: Tensor
    right_target: Tensor
    metadata: dict

    @property
    def forest(self):
        return self.reference.forest

    def to(self, device):
        names = ("edge_children", "edge_parents", "left_source", "left_target",
                 "right_source", "right_target")
        return replace(self, reference=self.reference.to(device),
            **{name: getattr(self, name).to(device) for name in names})


def _native_code_v1(tree, intern):
    """Exact bottom-up structural interning, without persistent ancestor IDs."""
    samples, _, _ = _tree(tree)
    p, t = tree.parent.tolist(), tree.node_times.tolist()
    sample_at = {node: identity for identity, node in samples.items()}
    children = [[] for _ in p]
    for node, parent in enumerate(p):
        if parent >= 0:
            children[parent].append(node)
    code = [None] * len(p)
    for node in sorted(range(len(p)), key=lambda n: t[n]):
        key = (float(t[node]).hex(), sample_at.get(node, -1),
               tuple(sorted(code[c] for c in children[node])))
        if key not in intern:
            intern[key] = len(intern)
        code[node] = intern[key]
    return code[p.index(-1)], samples


def _augmented_roles_v1(tree, census, sample_identity_namespace):
    """Mirror H030's cut insertion order and derive predictor-only signatures."""
    samples, _, _ = _tree(tree)
    p, t = tree.parent.tolist(), tree.node_times.tolist()
    native_p, native_t = p.copy(), t.copy()
    for child, parent in enumerate(native_p):
        if parent >= 0 and native_t[child] < census < native_t[parent]:
            cut = len(p)
            p.append(parent); t.append(census); p[child] = cut
    sample_at = {node: identity for identity, node in samples.items()}
    # The sorted actual roster is identical across holds. Python integers are
    # exact compact bitsets; their numeric bit values never enter neural inputs.
    descendants = [0 for _ in p]
    for bit, identity in enumerate(sorted(samples)):
        descendants[samples[identity]] |= 1 << bit
    for node in sorted(range(len(p)), key=lambda n: t[n]):
        if p[node] >= 0:
            descendants[p[node]] |= descendants[node]
    keys = [("sample", sample_identity_namespace, sample_at[node]) if node in sample_at else
            ("role", sample_identity_namespace, descendants[node], float(t[node]).hex())
            for node in range(len(p))]
    return keys


def _unique_role_pairs_v1(source_keys, target_keys, source_offset=0, target_offset=0):
    """Only one-to-one exact keys join; ambiguous support is reset."""
    source, target = defaultdict(list), defaultdict(list)
    for row, key in enumerate(source_keys):
        source[key].append(row)
    for row, key in enumerate(target_keys):
        target[key].append(row)
    pairs, ambiguous = [], 0
    for key in source.keys() & target.keys():
        if len(source[key]) == len(target[key]) == 1:
            pairs.append((source_offset + source[key][0], target_offset + target[key][0]))
        else:
            ambiguous += len(source[key]) + len(target[key])
    return sorted(pairs, key=lambda pair: pair[1]), ambiguous


def prepare_observed_role_mixed_path_bank_v1(
        trees: Sequence[MetricTreeHoldV1], intervals_bp: Tensor, readout_x_bp: Tensor,
        query_ids: Tensor, reference_ids: Tensor, reference_classes: Tensor, *,
        census=350., sample_identity_namespace: str, **geometry):
    """Target-free CPU preparation; coalesce exact adjacent labelled metric trees."""
    if (not trees or intervals_bp.device.type != "cpu" or intervals_bp.requires_grad
            or intervals_bp.dtype != torch.float64 or intervals_bp.shape != (len(trees), 2)
            or not bool(torch.isfinite(intervals_bp).all())
            or bool((intervals_bp[:, 1] <= intervals_bp[:, 0]).any())
            or not torch.equal(intervals_bp[:-1, 1], intervals_bp[1:, 0])):
        raise ValueError("Fixed contiguous complete observed intervals required")
    intern, kept, spans, codes, roster = {}, [], [], [], None
    for hold, tree in enumerate(trees):
        code, samples = _native_code_v1(tree, intern)
        dated = {identity: float(tree.node_times[row]) for identity, row in samples.items()}
        if roster is None:
            roster = dated
        elif dated != roster:
            raise ValueError("Actual sampled identity/date roster changed")
        if codes and code == codes[-1]:
            spans[-1][1] = float(intervals_bp[hold, 1])
        else:
            kept.append(tree)
            spans.append(intervals_bp[hold].tolist())
            codes.append(code)
    spans = torch.tensor(spans, dtype=torch.float64)
    reference = prepare_reference_logcavity_bank_v1(
        kept, spans, readout_x_bp, query_ids, reference_ids, reference_classes,
        census=census, sample_identity_namespace=sample_identity_namespace, **geometry)
    # H030 preparation validates the complete policy. Carrier frames are never
    # used by this prototype and are discarded rather than retained as memory.
    reference = replace(reference, forest=replace(reference.forest,
        forward_frames=(), reverse_frames=()))
    layouts = [_augmented_roles_v1(tree, census, sample_identity_namespace) for tree in kept]
    offsets = [0]
    for keys in layouts:
        offsets.append(offsets[-1] + len(keys))
    if offsets[-1] != len(reference.forest.parent):
        raise ValueError("Role rows differ from complete H030 augmented forest")
    left, ambiguous = [], 0
    for hold in range(1, len(kept)):
        pairs, count = _unique_role_pairs_v1(layouts[hold-1], layouts[hold],
                                           offsets[hold-1], offsets[hold])
        left.extend(pairs); ambiguous += count
    ls = torch.tensor([s for s, _ in left], dtype=torch.long)
    lt = torch.tensor([t for _, t in left], dtype=torch.long)
    children = (reference.forest.parent >= 0).nonzero().flatten()
    metadata = dict(original_holds=len(trees), coalesced_holds=len(kept),
        coalesced_duplicate_holds=len(trees)-len(kept), vertices=offsets[-1],
        edges=len(children), adjacent_matched_pairs=len(left),
        ambiguous_rows_reset=ambiguous, retained_recurrence=False,
        native_identity_used_for_alignment=False, true_ancestry_asserted=False,
        sample_identity_namespace=sample_identity_namespace, descendant_encoding="exact_roster_bitset",
        whole_observed_trees_and_analytic_cuts=True,
        older_proxy_policy="unchanged_H030_actual_sampling_nodes",
        source_policy="unchanged_H030_class_balanced_proxy_values",
        node_alignment="actual_samples_or_unique_descendant_set_and_exact_observed_age",
        interval_policy="coalesce_identical_adjacent_labelled_metric_trees",
        nonlinear_unary_subdivision_invariant=False, arms=list(ARMS))
    return ObservedRoleMixedPathBankV1(reference, children,
        reference.forest.parent[children], ls, lt, lt.clone(), ls.clone(), metadata)


class ObservedRoleMixedPathGNNV1(ReferenceLogCavityGNNV1):
    """Shared sparse mixed paths, static H030 gates and linear three-stream decode."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path_edge_gate = nn.Linear(8, 1)
        self.path_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # Inherited allocation is disclosed; no unused memory block is trained.
        unused = ("value.", "key.", "retention.", "write_gate.", "carrier_gru.",
                  "decoder_edge.", "kernel_head.")
        for name, parameter in self.named_parameters():
            if name.startswith(unused):
                parameter.requires_grad_(False)
        self.last_diagnostics = {}

    def parameter_counts(self, control):
        if control not in ARMS:
            raise ValueError("Unknown matched mixed-path arm")
        return dict(total=sum(p.numel() for p in self.parameters()),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            frozen_unused=sum(p.numel() for p in self.parameters() if not p.requires_grad),
            per_vertex_path_streams=2, retained_recurrent_state_scalars=0)

    def path_action(self, bank, value):
        """Shared undirected weighted tree mean with unit self loops."""
        forest = bank.forest
        child, parent = bank.edge_children, bank.edge_parents
        weight = self.path_edge_gate(forest.geometry[child]).sigmoid().flatten()
        denominator = value.new_ones(len(value))
        denominator = denominator.index_add(0, child, weight).index_add(0, parent, weight)
        result = value.index_add(0, child, value[parent] * weight[:, None])
        result = result.index_add(0, parent, value[child] * weight[:, None])
        return result / denominator[:, None]

    @staticmethod
    def align(value, source, target):
        return value.new_zeros(value.shape).index_add(0, target, value[source])

    def mixed_paths(self, bank, value, *, control):
        if control not in ARMS:
            raise ValueError("Unknown matched mixed-path arm")
        propagated = self.path_action(bank, value)
        result = []
        sign = -1. if control == "signed_difference" else 1.
        for source, target in ((bank.left_source, bank.left_target),
                               (bank.right_source, bank.right_target)):
            aligned = self.align(value, source, target)
            after = self.path_action(bank, aligned)
            before = self.align(propagated, source, target)
            result.append((after + sign * before) * .5)
        return tuple(result)

    def decode_paths(self, forest, up, outside, left, right):
        """H030 static maps/gates, with node-level path values as extra streams."""
        n, d = len(up), self.hidden_dim
        base = torch.stack((self.decoder_base(up+outside),
                            self.path_value(left), self.path_value(right)), 1)
        condition = torch.cat((up, outside, forest.geometry), -1)
        ug = self.decoder_up_gate(condition).sigmoid()[:, None]
        dg = self.decoder_down_gate(condition).sigmoid()[:, None]
        nodes_value = up.new_zeros(n, 3, d)
        total = up.new_zeros(n, 3, d); messages = up.new_zeros(n, 3, d)
        for nodes, children, ranks in zip(forest.upward_levels,
                forest.upward_children, forest.upward_ranks):
            acc = up.new_zeros(len(nodes), 3, d)
            if children.numel():
                value = self.decoder_up(nodes_value[children]) * ug[children]
                value = value / forest.child_count[forest.parent[children]][:, None, None]
                messages.index_copy_(0, children, value)
                acc.index_add_(0, ranks, value)
            value = base[nodes] + acc
            total.index_copy_(0, nodes, value); nodes_value.index_copy_(0, nodes, value)
        down = up.new_zeros(n, 3, d)
        for nodes in forest.downward_levels:
            parents = forest.parent[nodes]
            cavity = down[parents] + total[parents] - messages[nodes]
            down.index_copy_(0, nodes, self.decoder_down(cavity) * dg[nodes])
        return nodes_value + down

    def forward(self, bank, frozen_baseline_logits, *, control):
        forest = bank.forest
        if (control not in ARMS or bank.reference.log_source.dtype != torch.float32
                or frozen_baseline_logits.dtype != torch.float32
                or frozen_baseline_logits.shape !=
                    (len(forest.readout_hold), forest.query_nodes.shape[1], 4)
                or bank.reference.log_source.device != self.raw_cost.device
                or frozen_baseline_logits.device != self.raw_cost.device
                or not bool(torch.isfinite(frozen_baseline_logits).all())):
            raise ValueError("Matched FP32 H030 bank and complete finite frozen support required")
        log_up, log_out = self.log_cavities(bank.reference)
        up, outside = self.encode_units(bank.reference, log_up, log_out)
        left, right = self.mixed_paths(bank, up+outside, control=control)
        decoded = self.decode_paths(forest, up, outside, left, right)
        context = decoded[forest.query_nodes[forest.readout_hold]]
        distance = torch.stack((torch.ones_like(forest.forward_distance),
            torch.exp(-forest.forward_distance/self.ell_bp()),
            torch.exp(-forest.reverse_distance/self.ell_bp())), -1)
        context = (context * distance[:, None, :, None]).flatten(2)
        centered = context.clone()
        for group in forest.age_groups:
            centered[:, group] = context[:, group] - context[:, group].mean(1, keepdim=True)
        result = frozen_baseline_logits.detach() + self.context_head(centered)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite mixed-path residual")
        self.last_diagnostics = dict(left_path_l2=float(left.detach().norm()),
            right_path_l2=float(right.detach().norm()),
            centered_context_l2=float(centered.detach().norm()),
            ell_bp=float(self.ell_bp().detach()), retained_recurrent_state_scalars=0)
        return result

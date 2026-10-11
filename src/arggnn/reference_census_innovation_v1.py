"""H026 fixed census-conditioned reference-kernel refinement.

Observed pure-reference classes are proxy inputs, not ancestral truth. Complete
metric trees and analytic cut350 attachments are retained. This discovery model
has no ordered cross-tree transport and is not claimed as a new GNN.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch
from torch import Tensor, nn

from .persistence_trace_kernel_v1 import MetricTreeHoldV1, _tree

ARMS = ("native_reference", "age_free_reference", "census_reference", "census_sham")


@dataclass(frozen=True)
class ReferenceCensusContextV1:
    fields: dict[str, Tensor]
    native_trees: tuple
    census_attachments: tuple
    query_ids: Tensor
    query_ages: Tensor
    reference_ids: Tensor
    reference_classes: Tensor
    reference_ages: Tensor
    sham_permutation: Tensor
    metadata: dict

    def field_for_arm(self, arm: str) -> Tensor:
        if arm not in ARMS:
            raise ValueError("Unknown reference-kernel control")
        return self.fields[arm]


def _fixed(value: Tensor, dtype, shape, name):
    if (not isinstance(value, Tensor) or value.device.type != "cpu" or value.requires_grad
            or value.dtype != dtype or value.shape != shape):
        raise ValueError(name + " requires fixed matching CPU geometry")


def _diploid_sham(query_ages, subject_ids, haplotypes, namespace, seed):
    q = len(query_ages)
    if len(subject_ids) != q or len(haplotypes) != q or not namespace or type(seed) is not int:
        raise ValueError("Explicit actual diploid identities and parent-wide namespace required")
    subjects = {}
    for row, subject in enumerate(subject_ids):
        subjects.setdefault(str(subject), []).append(row)
    by_age = {}
    for subject, rows in subjects.items():
        if (len(rows) != 2 or len({str(haplotypes[row]) for row in rows}) != 2
                or float(query_ages[rows[0]]) != float(query_ages[rows[1]])):
            raise ValueError("Sham requires complete observed diploids of one exact age")
        rows.sort(key=lambda row: str(haplotypes[row]))
        by_age.setdefault(float(query_ages[rows[0]]), []).append((subject, rows))
    local_seed = int.from_bytes(hashlib.sha256((namespace + ":" + str(seed)).encode()).digest()[:8], "big")
    rng = torch.Generator(device="cpu").manual_seed(local_seed)
    permutation = torch.arange(q)
    singleton_groups = 0
    for age, blocks in sorted(by_age.items()):
        if len(blocks) == 1:
            singleton_groups += 1
            continue
        order = torch.randperm(len(blocks), generator=rng).tolist()
        for index, source in enumerate(order):
            destination = order[(index + 1) % len(order)]
            left, right = blocks[source][1], blocks[destination][1]
            if [str(haplotypes[row]) for row in left] != [str(haplotypes[row]) for row in right]:
                raise ValueError("Actual haplotype order must agree across diploid sham blocks")
            permutation[left] = torch.tensor(right)
    return permutation, {"diploid_blocks": len(subjects), "exact_age_groups": len(by_age),
        "singleton_age_groups": singleton_groups,
        "deranged_query_rows": int((permutation != torch.arange(q)).sum()),
        "query_ids": list(subject_ids), "query_haplotypes": list(haplotypes),
        "policy": "parent-wide_fixed_diploid_block_derangement_within_exact_observed_query_age"}


def _affinity(distance: Tensor, labels: Tensor, scale: float) -> Tensor:
    """Stable per-class and total log mass; absent classes contribute zero."""
    count = torch.bincount(labels, minlength=4)
    columns = []
    for category in range(4):
        selected = labels == category
        if int(count[category]) == 0:
            columns.append(torch.full((len(distance),), -torch.inf, dtype=torch.float32))
        else:
            scores = -distance[:, selected] / scale - count[category].float().log()
            columns.append(torch.logsumexp(scores, dim=1))
    log_mass = torch.stack(columns, 1)
    peak = log_mass.amax(dim=1, keepdim=True)
    if not bool(torch.isfinite(peak).all()):
        raise ValueError("A reference cohort needs a finite observed source mass")
    shifted = log_mass - peak
    total = torch.logsumexp(shifted, dim=1, keepdim=True)
    return torch.exp(shifted - total)


def prepare_reference_census_context_v1(
        trees: Sequence[MetricTreeHoldV1], intervals_bp: Tensor, readout_x_bp: Tensor,
        query_ids: Tensor, reference_ids: Tensor, reference_classes: Tensor, *,
        query_diploid_ids: Sequence, query_haplotypes: Sequence, parent_namespace: str,
        seed: int = 101, census: float = 350., younger_scale_generations: float = 350.,
        older_scale_generations: float = 1400.) -> ReferenceCensusContextV1:
    """Fixed FP32 eight-channel fields on each containing original tree.

    Younger and older observed reference cohorts are always both exposed.
    Labels are class-balanced separately within cohorts. No query truth is
    accepted. Age-group centering uses every actual Q in each exact age group.
    """
    if not trees or any(scale <= 0 for scale in (census, younger_scale_generations, older_scale_generations)):
        raise ValueError("Nonempty complete trees and positive fixed scales required")
    _fixed(intervals_bp, torch.float64, (len(trees), 2), "intervals")
    _fixed(readout_x_bp, torch.float64, (len(readout_x_bp),), "readouts")
    _fixed(query_ids, torch.long, (len(query_ids),), "queries")
    _fixed(reference_ids, torch.long, (len(reference_ids),), "references")
    _fixed(reference_classes, torch.long, reference_ids.shape, "reference classes")
    if (not len(query_ids) or not len(reference_ids) or not len(readout_x_bp)
            or query_ids.unique().numel() != len(query_ids)
            or reference_ids.unique().numel() != len(reference_ids)
            or bool(torch.isin(query_ids, reference_ids).any())
            or bool(((reference_classes < 0) | (reference_classes > 3)).any())
            or not bool(torch.isfinite(intervals_bp).all() & torch.isfinite(readout_x_bp).all())
            or bool((intervals_bp[:, 1] <= intervals_bp[:, 0]).any())
            or not torch.equal(intervals_bp[:-1, 1], intervals_bp[1:, 0])
            or bool(((readout_x_bp < intervals_bp[0, 0]) | (readout_x_bp >= intervals_bp[-1, 1])).any())):
        raise ValueError("Exact contiguous physical support and distinct observed Q/R joins required")
    held = {arm: [] for arm in ARMS[:-1]}
    attachments, query_ages, reference_ages = [], None, None
    clamp_pairs = 0
    opportunity_query_bp = 0.
    for tree, interval in zip(trees, intervals_bp):
        samples, _, _ = _tree(tree)
        if any(identity not in samples for identity in (*query_ids.tolist(), *reference_ids.tolist())):
            raise ValueError("Every original hold must contain the complete observed Q/R roster")
        qnodes = [samples[identity] for identity in query_ids.tolist()]
        rnodes = [samples[identity] for identity in reference_ids.tolist()]
        qa, ra = tree.node_times[qnodes], tree.node_times[rnodes]
        if query_ages is None:
            query_ages, reference_ages = qa.clone(), ra.clone()
        elif not torch.equal(query_ages, qa) or not torch.equal(reference_ages, ra):
            raise ValueError("Persistent observed sample dates changed across holds")
        if bool((qa >= census).any()):
            raise ValueError("Actual queries must be strictly younger than census")
        young, old = ra < census, ra >= census
        if not bool(young.any()) or not bool(old.any()):
            raise ValueError("Both genuine observed reference age cohorts are required")
        parents = tree.parent.tolist()
        paths = {}
        for node in qnodes + rnodes:
            path, at = [], node
            while at >= 0:
                path.append(at); at = parents[at]
            paths[node] = path
        mrca = []
        for node in qnodes:
            ancestors = set(paths[node])
            mrca.append([float(tree.node_times[next(at for at in paths[ref] if at in ancestors)])
                         for ref in rnodes])
        mrca = torch.tensor(mrca, dtype=torch.float32)
        native_distance = 2 * mrca - qa.float()[:, None] - ra.float()[None]
        if bool((native_distance < 0).any()):
            raise ValueError("Observed metric path distances must be nonnegative")
        old_field = _affinity(native_distance[:, old], reference_classes[old], older_scale_generations)
        # Normalization cancels the native-Q to cut350 row constant for old R:
        # the old proxy field is exactly shared, not made more census-informed.
        native_young = _affinity(native_distance[:, young], reference_classes[young], younger_scale_generations)
        age_free = _affinity(2 * mrca[:, young], reference_classes[young], younger_scale_generations)
        cut = _affinity(2 * mrca[:, young].clamp_min(census), reference_classes[young], younger_scale_generations)
        for arm, value in zip(ARMS[:-1], (native_young, age_free, cut)):
            held[arm].append(torch.cat((value, old_field), 1))
        active = mrca[:, young] < census
        clamp_pairs += int(active.sum())
        opportunity_query_bp += float(interval[1] - interval[0]) * int(active.any(1).sum())
        cut_points = {}
        for node in qnodes + [rnodes[i] for i in young.nonzero().flatten().tolist()]:
            lower = node
            while parents[lower] >= 0 and float(tree.node_times[parents[lower]]) <= census:
                lower = parents[lower]
            upper = parents[lower]
            if upper < 0:
                raise ValueError("No invented cut point when native ancestry does not cross census")
            fraction = (census - float(tree.node_times[lower])) / float(tree.node_times[upper] - tree.node_times[lower])
            cut_points[node] = (int(tree.qualified_node_ids[lower]), int(tree.qualified_node_ids[upper]), fraction)
        attachments.append(cut_points)
    permutation, sham = _diploid_sham(query_ages, query_diploid_ids, query_haplotypes, parent_namespace, seed)
    holding = torch.searchsorted(intervals_bp[:, 1].contiguous(), readout_x_bp, right=True)
    fields = {}
    for arm in ARMS[:-1]:
        value = torch.stack(held[arm])[holding].clone()
        for age in torch.unique(query_ages):
            columns = query_ages == age
            value[:, columns] -= value[:, columns].mean(1, keepdim=True)
        fields[arm] = value
    fields[ARMS[-1]] = fields["census_reference"][:, permutation].clone()
    span = float(intervals_bp[-1, 1] - intervals_bp[0, 0])
    young = reference_ages < census
    metadata = {
        "kind": "fixed_census_conditioned_observed_reference_kernel_refinement",
        "holds": len(trees), "complete_native_vertices": sum(t.parent.numel() for t in trees),
        "analytic_cut_points": sum(len(points) for points in attachments),
        "census": census, "channels": 8, "query_count": len(query_ids),
        "younger_reference_count": int(young.sum()), "older_reference_count": int((~young).sum()),
        "younger_class_counts": torch.bincount(reference_classes[young], minlength=4).tolist(),
        "older_class_counts": torch.bincount(reference_classes[~young], minlength=4).tolist(),
        "query_age_groups": {str(float(age)): int((query_ages == age).sum()) for age in torch.unique(query_ages)},
        "centering": "each_original_readout_over_full_actual_Q_of_same_exact_observed_age;unlabeled_transduction",
        "reference_class_balancing": "inverse_observed_class_count_separately_young_and_old;missing_classes_zero",
        "older_proxy_field_shared_exactly_across_native_age_free_census": True,
        "young_MRCA_below_census_pairs": clamp_pairs,
        "span_weighted_census_clamp_opportunity_fraction": opportunity_query_bp / (span * len(query_ids)),
        "census_vs_age_free_max_context_difference": float((fields["census_reference"] - fields["age_free_reference"]).abs().max()),
        "sham_permutation_actual_query_ids": query_ids[permutation].tolist(), "sham": sham,
        "readout": "containing_original_tree_no_cross_tree_interpolation",
        "older_reference_labels_are_proxies_not_census_truth": True,
        "ordered_cross_tree_mechanism_tested": False, "novel_GNN_claim": False,
    }
    return ReferenceCensusContextV1(fields, tuple(trees), tuple(attachments), query_ids.clone(),
        query_ages, reference_ids.clone(), reference_classes.clone(), reference_ages,
        permutation, metadata)


class ReferenceCensusInnovationV1(nn.Module):
    """Only a graph-context correction; no local logits input, intercept or bias."""
    def __init__(self):
        super().__init__()
        self.context_head = nn.Linear(8, 4, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.context_head.weight)

    def forward(self, frozen_baseline_logits: Tensor, context: Tensor) -> Tensor:
        if (frozen_baseline_logits.ndim != 3 or frozen_baseline_logits.shape[-1] != 4
                or context.shape != (*frozen_baseline_logits.shape[:2], 8)
                or frozen_baseline_logits.dtype != torch.float32 or context.dtype != torch.float32
                or frozen_baseline_logits.device != self.context_head.weight.device
                or context.device != frozen_baseline_logits.device
                or not bool(torch.isfinite(frozen_baseline_logits).all() & torch.isfinite(context).all())):
            raise ValueError("Aligned finite FP32 frozen baseline and fixed observed context required")
        return frozen_baseline_logits.detach() + self.context_head(context.detach())

"""H028 fixed observed SNP evidence and matched context-only residual heads.

Original phased allele observations and pureR96 proxy labels only. Genotype
composite scores are features, not calibrated posterior probabilities. No
ancestry targets, fitted ancestry labels, baseline logits or tree populations
are accepted by feature preparation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

ARMS = ("genotype_only", "graph_only", "genotype_graph_fusion")


def validate_observed_roster_v1(rows: Sequence[dict]) -> dict:
    if len(rows) != 280 or [r["sample_index"] for r in rows] != list(range(280)):
        raise ValueError("Original ordered280 observed haplotypes required")
    keys = [(r["sample_id"], r["haplotype"]) for r in rows]
    if len(set(keys)) != 280:
        raise ValueError("Unique sample ID/haplotype identity joins required")
    for i in range(0, 280, 2):
        a, b = rows[i:i + 2]
        if (a["sample_id"] != b["sample_id"] or (a["haplotype"], b["haplotype"]) != (0, 1)
                or any(a[k] != b[k] for k in ("role", "observed_cohort",
                    "age_generations", "observed_reference_class"))):
            raise ValueError("Original phased diploid order, role, date or class differs")
    roles = {role: sum(r["role"] == role for r in rows)
             for role in ("query", "pure_source_reference", "mixed_reference_only")}
    if roles != {"query": 120, "pure_source_reference": 96, "mixed_reference_only": 64}:
        raise ValueError("Exact observedQ120/pureR96/mixed64 roster required")
    donors = [r for r in rows if r["role"] == "pure_source_reference"]
    if (any(type(r["observed_reference_class"]) is not int or
            not 0 <= r["observed_reference_class"] < 4 for r in donors)
            or any(sum(r["observed_reference_class"] == c for r in donors) != 24 for c in range(4))
            or any(r["observed_reference_class"] is not None for r in rows
                   if r["role"] != "pure_source_reference")
            or any(not math.isfinite(float(r["age_generations"])) or
                   r["age_generations"] < 0 for r in rows)):
        raise ValueError("Only observed pureR96 may carry C4 labels; dates must be observed")
    return {"samples": 280, "diploids": 140, "role_counts": roles,
            "sample_names": [r["sample_id"] for r in rows[::2]],
            "query_indexes": [r["sample_index"] for r in rows if r["role"] == "query"],
            "reference_indexes": [r["sample_index"] for r in donors]}


@dataclass(frozen=True)
class ObservedAllelesV1:
    positions_bp0: np.ndarray
    allele_counts: np.ndarray
    genotypes: np.ndarray
    metadata: dict


def parse_observed_vcf_v1(text: str, roster: Sequence[dict], coordinates: dict) -> ObservedAllelesV1:
    joined = validate_observed_roster_v1(roster)
    if (coordinates.get("build") != "synthetic" or coordinates.get("contig_id") != "22"
            or coordinates.get("length_bp") != 2000000
            or coordinates.get("empirical_GRCh37_coordinate_claimed") is not False):
        raise ValueError("Original synthetic2Mb contig22 coordinate metadata required")
    rows = coordinates["rows"]
    if not rows:
        raise ValueError("No observed site coordinates")
    for i, row in enumerate(rows):
        if (row["site_id"] != i or type(row["tree_position_0based"]) is not int
                or row["VCF_position_1based"] != row["tree_position_0based"] + 1
                or not 0 <= row["tree_position_0based"] < 2000000
                or (i and row["tree_position_0based"] <= rows[i - 1]["tree_position_0based"])):
            raise ValueError("Original site/physical/VCF+1 join differs")
    genotypes = np.empty((len(rows), 280), dtype=np.int8)
    allele_counts = np.empty(len(rows), dtype=np.int8)
    seen_header, at = False, 0
    for line in text.splitlines():
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            columns = line.split("\t")
            if seen_header or columns[:9] != ["#CHROM", "POS", "ID", "REF", "ALT",
                    "QUAL", "FILTER", "INFO", "FORMAT"] or columns[9:] != joined["sample_names"]:
                raise ValueError("Original VCF sample order/header differs")
            seen_header = True
            continue
        if line.startswith("#") or not seen_header:
            raise ValueError("Unsupported or absent original VCF header")
        fields = line.split("\t")
        if at >= len(rows) or len(fields) != 149:
            raise ValueError("Original VCF site/sample count differs")
        expected = rows[at]
        if fields[0] != "22" or int(fields[1]) != expected["VCF_position_1based"] or fields[2] != str(expected["site_id"]):
            raise ValueError("Original VCF site identity/position differs")
        alleles = [fields[3]] + ([] if fields[4] == "." else fields[4].split(","))
        if not 1 <= len(alleles) <= 4 or len(set(alleles)) != len(alleles) or any(a not in ("A", "C", "G", "T") for a in alleles):
            raise ValueError("Original categorical nucleotide alleles required")
        fmt = fields[8].split(":")
        if fmt.count("GT") != 1:
            raise ValueError("Exactly one phasedGT field required")
        gt_index = fmt.index("GT")
        values = []
        for cell in fields[9:]:
            pieces = cell.split(":")
            if len(pieces) <= gt_index:
                raise ValueError("MissingGT field")
            pair = pieces[gt_index].split("|")
            if len(pair) != 2:
                raise ValueError("Unphased input cannot be silently phased")
            for allele in pair:
                if allele == ".":
                    values.append(-1)
                elif allele.isdecimal() and 0 <= int(allele) < len(alleles):
                    values.append(int(allele))
                else:
                    raise ValueError("Invalid categorical genotype index")
        genotypes[at] = values
        allele_counts[at] = len(alleles)
        at += 1
    if not seen_header or at != len(rows):
        raise ValueError("All original observed sites must be retained, including multiallelic")
    positions = np.asarray([r["tree_position_0based"] for r in rows], dtype=np.float64)
    return ObservedAllelesV1(positions, allele_counts, genotypes, {
        "original_sites": at, "multiallelic_sites": int((allele_counts > 2).sum()),
        "missing_haplotype_calls": int((genotypes < 0).sum()),
        "phase_policy": "original phase retained; no inference/imputation",
        "site_filtering": "none", "ancestral_polarization_used": False})


def categorical_log_evidence_v1(genotypes, allele_counts, query_indexes, reference_indexes,
        reference_classes, reference_ages, *, alpha: float = 0.5, census: float = 350.):
    """Per-site log evidence with common callability within each donor cohort.

    A globally absent class stays absent. A site missing all donors for a
    globally present class is unavailable to EVERY class in that cohort.
    """
    g = np.asarray(genotypes)
    k = np.asarray(allele_counts)
    qi, ri = np.asarray(query_indexes, dtype=np.int64), np.asarray(reference_indexes, dtype=np.int64)
    labels, ages = np.asarray(reference_classes), np.asarray(reference_ages, dtype=np.float64)
    if (g.ndim != 2 or g.dtype.kind not in "iu" or k.shape != (len(g),)
            or k.dtype.kind not in "iu" or np.any((k < 1) | (k > 4))
            or np.any(g < -1) or np.any(g >= k[:, None])
            or qi.ndim != 1 or ri.ndim != 1 or not len(qi) or not len(ri)
            or len(set(qi.tolist())) != len(qi) or len(set(ri.tolist())) != len(ri)
            or np.intersect1d(qi, ri).size or np.any(qi < 0) or np.any(ri < 0)
            or np.any(qi >= g.shape[1]) or np.any(ri >= g.shape[1])
            or labels.shape != ri.shape or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels > 3)) or ages.shape != ri.shape
            or not np.isfinite(ages).all() or np.any(ages < 0)
            or not math.isfinite(alpha) or alpha <= 0
            or not math.isfinite(census) or census <= 0):
        raise ValueError("Fixed valid observed categorical Q/R arrays and positive pseudocount required")
    q = g[:, qi]
    scores = np.zeros((len(g), len(qi), 8), dtype=np.float64)
    available = np.zeros_like(scores, dtype=bool)
    class_counts = np.zeros((2, 4), dtype=np.int64)
    for cohort, group in enumerate((ages < census, ages >= census)):
        present, logp = [], {}
        cohort_call = q >= 0
        for category in range(4):
            selected = ri[group & (labels == category)]
            class_counts[cohort, category] = len(selected)
            if not len(selected):
                continue
            present.append(category)
            donor = g[:, selected]
            called = (donor >= 0).sum(1)
            cohort_call &= called[:, None] > 0
            counts = np.stack([(donor == a).sum(1) for a in range(4)], 1)
            probabilities = (counts + alpha) / (called + k * alpha)[:, None]
            chosen = probabilities[np.arange(len(g))[:, None], np.maximum(q, 0)]
            logp[category] = np.log(chosen)
        for category in present:
            channel = cohort * 4 + category
            scores[:, :, channel] = np.where(cohort_call, logp[category], 0.)
            available[:, :, channel] = cohort_call
    return scores, available, class_counts


def composite_window_fields_v1(positions_bp0, site_log_evidence, available, centers_bp0,
        coverage_bp0, *, window_bp: int = 10000, ld_block_bp: int = 1000):
    """Parent-coordinate windows, common cohort callability, fixed density weights.

    Each original1kb block has total observed-site weight1. This is an explicit
    composite-evidence heuristic, not a learned LD model or exact likelihood.
    """
    x, centers = np.asarray(positions_bp0, dtype=np.float64), np.asarray(centers_bp0, dtype=np.float64)
    score = np.asarray(site_log_evidence, dtype=np.float64)
    valid = np.asarray(available)
    coverage = np.asarray(coverage_bp0, dtype=np.float64)
    if (x.ndim != 1 or not len(x) or np.any(np.diff(x) <= 0) or centers.ndim != 1
            or not len(centers) or np.any(np.diff(centers) <= 0)
            or score.ndim != 3 or score.shape[0] != len(x) or score.shape[2] != 8
            or valid.shape != score.shape or valid.dtype != np.bool_
            or not np.isfinite(x).all() or not np.isfinite(centers).all()
            or not np.isfinite(score).all() or coverage.shape != (2,)
            or not np.isfinite(coverage).all() or not 0 <= coverage[0] < coverage[1] <= 2000000
            or np.any(centers < coverage[0]) or np.any(centers >= coverage[1])
            or type(window_bp) is not int or window_bp != 10000
            or type(ld_block_bp) is not int or ld_block_bp != 1000):
        raise ValueError("Fixed10kb original physical windows and1kb density weights required")
    inside = (x >= coverage[0]) & (x < coverage[1])
    x, score, valid = x[inside], score[inside], valid[inside]
    blocks = (x // ld_block_bp).astype(np.int64)
    _, inverse, density = np.unique(blocks, return_inverse=True, return_counts=True)
    weights = 1. / density[inverse]
    prefix_score = np.concatenate((np.zeros((1, *score.shape[1:])),
        np.cumsum(score * weights[:, None, None], axis=0)), axis=0)
    prefix_count = np.concatenate((np.zeros((1, *score.shape[1:]), dtype=np.int32),
        np.cumsum(valid, axis=0, dtype=np.int32)), axis=0)
    lo = np.searchsorted(x, np.maximum(centers - window_bp / 2., coverage[0]), side="left")
    hi = np.searchsorted(x, np.minimum(centers + window_bp / 2., coverage[1]), side="left")
    sums = prefix_score[hi] - prefix_score[lo]
    counts = prefix_count[hi] - prefix_count[lo]
    fields = np.zeros_like(sums)
    for cohort in range(2):
        columns = slice(cohort * 4, cohort * 4 + 4)
        usable = counts[:, :, columns] > 0
        values = np.where(usable, sums[:, :, columns], -np.inf)
        peak = np.max(values, axis=-1, keepdims=True)
        active = np.any(usable, axis=-1, keepdims=True)
        shifted = values - np.where(active, peak, 0.)
        numerator = np.exp(shifted)
        denominator = numerator.sum(-1, keepdims=True)
        np.divide(numerator, denominator, out=fields[:, :, columns], where=active)
    if not np.isfinite(fields).all():
        raise ValueError("Nonfinite composite field; no probability floor is allowed")
    return fields, counts, {
        "window_bp": window_bp, "ld_block_bp": ld_block_bp,
        "window_coordinates": "original_parent_bp0_center_plusminus5000;half_open;parent_coverage_clip_only",
        "crosses_artificial250kb_tiles": True,
        "scored_center_count": len(centers), "original_sites_in_coverage": len(x),
        "original_1kb_blocks": len(density),
        "empty_query_cohort_windows": [int((counts[:, :, h * 4:h * 4 + 4].sum(-1) == 0).sum()) for h in range(2)],
        "called_site_counts_min_by_channel": counts.min((0, 1)).tolist(),
        "called_site_counts_max_by_channel": counts.max((0, 1)).tolist(),
        "normalization": "stable_softmax_weighted_log_evidence_over_only_available_classes_per_cohort",
        "globally_absent_classes": "exact_zero_no_synthetic_donor;excluded_from_softmax_denominator"}


def center_observed_query_fields_v1(fields, counts, query_ages):
    value = np.asarray(fields, dtype=np.float64).copy()
    count = np.asarray(counts)
    ages = np.asarray(query_ages, dtype=np.float64)
    if count.shape != value.shape or ages.shape != (value.shape[1],) or not np.isfinite(ages).all():
        raise ValueError("Matching observed query dates/callability required")
    for age in np.unique(ages):
        group = np.flatnonzero(ages == age)
        for cohort in range(2):
            cols = slice(4 * cohort, 4 * cohort + 4)
            valid = count[:, group, cols].sum(-1) > 0
            current = value[:, group, cols]
            n = valid.sum(1, keepdims=True)
            mean = np.zeros((len(value), 1, 4), dtype=np.float64)
            np.divide((current * valid[:, :, None]).sum(1, keepdims=True),
                      n[:, :, None], out=mean, where=n[:, :, None] > 0)
            value[:, group, cols] = np.where(valid[:, :, None], current - mean, 0.)
    return torch.from_numpy(value.astype(np.float32))


def prepare_observed_variant_fields_v1(alleles: ObservedAllelesV1, roster, centers_bp0, coverage_bp0,
        *, alpha=0.5, census=350., window_bp=10000, ld_block_bp=1000):
    joined = validate_observed_roster_v1(roster)
    qi, ri = joined["query_indexes"], joined["reference_indexes"]
    labels = np.asarray([roster[i]["observed_reference_class"] for i in ri], dtype=np.int64)
    ages = np.asarray([roster[i]["age_generations"] for i in ri], dtype=np.float64)
    scores, available, class_counts = categorical_log_evidence_v1(alleles.genotypes,
        alleles.allele_counts, qi, ri, labels, ages, alpha=alpha, census=census)
    fields, counts, meta = composite_window_fields_v1(alleles.positions_bp0, scores, available,
        centers_bp0, coverage_bp0, window_bp=window_bp, ld_block_bp=ld_block_bp)
    tensor = center_observed_query_fields_v1(fields, counts, [roster[i]["age_generations"] for i in qi])
    meta.update(alleles.metadata, alpha=alpha, census=census,
        younger_class_counts=class_counts[0].tolist(), older_class_counts=class_counts[1].tolist(),
        younger_reference_count=int(class_counts[0].sum()), older_reference_count=int(class_counts[1].sum()),
        centering="full_actual_Q_same_exact_observed_age_at_each_original_readout;only_called_cohort_rows",
        target_or_fitted_ancestry_used=False, reference_labels="observed_pureR96_proxies_only")
    return tensor, meta


@dataclass(frozen=True)
class ObservedVariantContextV1:
    genotype: Tensor
    graph: Tensor
    metadata: dict

    def field_for_arm(self, arm):
        if arm == "genotype_only":
            return self.genotype
        if arm == "graph_only":
            return self.graph
        if arm == "genotype_graph_fusion":
            return (self.genotype + self.graph) * 0.5
        raise ValueError("Unknown fixed observed-information control")


class ObservedVariantResidualV1(nn.Module):
    """Matched32param correction; baseline is an output skip, never a head input."""
    def __init__(self):
        super().__init__()
        self.context_head = nn.Linear(8, 4, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.context_head.weight)

    def forward(self, frozen_baseline_logits: Tensor, context: Tensor):
        if (frozen_baseline_logits.ndim != 3 or frozen_baseline_logits.shape[-1] != 4
                or context.shape != (*frozen_baseline_logits.shape[:2], 8)
                or context.dtype != torch.float32 or frozen_baseline_logits.dtype != torch.float32
                or context.device != self.context_head.weight.device
                or frozen_baseline_logits.device != context.device
                or not bool(torch.isfinite(context).all() & torch.isfinite(frozen_baseline_logits).all())):
            raise ValueError("Matching finiteFP32 frozen skip and observation-only context required")
        return frozen_baseline_logits.detach() + self.context_head(context.detach())

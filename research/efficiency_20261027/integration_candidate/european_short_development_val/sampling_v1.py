"""Prepared explicit VAL sampler, preserving the original observed-SNP algorithm.

Only the input plan schema, validation split guard and fixed VAL quota checks
change. The output keeps the canonical registry schema consumed unchanged by
the original observation adapter. No TEST reads or label-dependent choices.
"""
import hashlib
import math
import random
from arggnn.benchmarking.european_sampling import COHORTS, QUERY_COUNTS, NATIVE_PATHS, _need
from .short_plan_v1 import LENGTH, plan_ref, validate_sampling_plan

def build_validation_registry(plan, parent_id, observed_rows, inference_metadata, *,
                           observation_ref):
    """Return position-sorted observed records with stable canonical example IDs."""
    validate_sampling_plan(plan)
    _need(plan.get("schema") == "ap_short_development_validation_sampled_plan_v1"
          and parent_id in plan["parents"], "Frozen paper sampling plan parent required")
    definition = plan["parents"][parent_id]
    split = definition["split"]
    _need(definition["cohort_quotas"] == {cohort: 2000 for cohort in COHORTS}
          and plan["site_sampling_uses_truth"] is False
          and plan["duplicate_focal_filter"] is False,
          "Independent VAL retains10000 public-metadata sampled examples")
    _need(split == "validation" and plan["sequence_length_bp"] == LENGTH,
          "Complete AP parent and declared independent split required")
    _need(isinstance(observation_ref, dict) and set(observation_ref) == {"path", "sha256"},
          "Explicit immutable observed inference reference required")
    _need(len(observed_rows) == 952
          and len({r["sample_index"] for r in observed_rows}) == 952
          and len({r["observed_node_id"] for r in observed_rows}) == 952
          and len({(r["sample_id"], r["haplotype"]) for r in observed_rows}) == 952,
          "All 952 observed AP haplotypes and semantic identities required")
    queries = [r for r in observed_rows if r["role"] == "query"]
    _need(len(queries) == 686
          and tuple(sum(r["observed_cohort"] == c for r in queries) for c in COHORTS) == QUERY_COUNTS
          and len({r["observed_cohort"] for r in observed_rows}) == 9,
          "All five admixed and four reference AP cohorts required")
    mapping = inference_metadata["sample_index_to_inferred_tree_node_id"]
    _need(len(mapping) == 952 and all(
        row["sample_index"] == mapped["sample_index"]
        and row["observed_node_id"] == mapped["inferred_tree_node_id"]
        for row, mapped in zip(observed_rows, mapping)),
        "Observed rows must join actual inferred sample identities")
    sites = inference_metadata["site_join"]
    left, right = inference_metadata["coverage_original_bp0"]
    offset = inference_metadata["local_to_original_offset_bp"]
    _need(0 <= left < right <= LENGTH and offset == left and len(sites) > 0,
          "Actual inferred coverage without extrapolation required")
    previous = -math.inf
    for index, site in enumerate(sites):
        local = site["local_position_0based"]
        original = site["original_position_0based"]
        _need(site["native_site_id"] == index
              and isinstance(local, (int, float)) and math.isfinite(local)
              and previous < local < right - left and local >= 0
              and local + offset == original and left <= original < right,
              "Exact ordered retained observed SNP coordinates required")
        previous = local
    query_identity = [dict(sample_index=r["sample_index"], sample_id=r["sample_id"],
        haplotype=r["haplotype"], query_node_id=r["observed_node_id"],
        cohort=r["observed_cohort"]) for r in queries]
    records = []
    for cohort in COHORTS:
        quota = definition["cohort_quotas"][cohort]
        members = sorted((r for r in queries if r["observed_cohort"] == cohort),
                         key=lambda r: r["sample_index"])
        stream = hashlib.sha256(
            (str(plan["seed"]) + "|" + parent_id + "|" + cohort).encode()).digest()
        random.Random(int.from_bytes(stream[:8], "big")).shuffle(members)
        n_positions = (quota + len(members) - 1) // len(members)
        _need(len(sites) >= n_positions,
              "Too few native observed SNPs for declared evenly spaced registry, no redraw/crop")
        ranks = [((2 * j + 1) * len(sites)) // (2 * n_positions)
                 for j in range(n_positions)]
        remaining = quota
        for rank in ranks:
            site = sites[rank]
            for row in members[:min(remaining, len(members))]:
                records.append(dict(
                    example_index=len(records), native_site_id=rank,
                    position_local_bp=site["local_position_0based"],
                    position_original_bp=site["original_position_0based"],
                    query_node_id=row["observed_node_id"], sample_index=row["sample_index"],
                    sample_id=row["sample_id"], haplotype=row["haplotype"], cohort=cohort))
                remaining -= 1
            if not remaining:
                break
        _need(remaining == 0, "Every declared sampled example must be retained")
    records.sort(key=lambda row: (row["position_local_bp"], row["example_index"]))
    expected = sum(definition["cohort_quotas"].values())
    _need(len(records) == expected and {r["example_index"] for r in records} == set(range(expected)),
          "Complete stable canonical example identity required")
    return dict(schema="ap_paper_sampled_registry_v1", parent_id=parent_id, split=split,
        sequence_length_bp=LENGTH, observation_ref=dict(observation_ref),
        short_development_plan_ref=plan_ref(), development_model_selection_only=True,
        cohort_quotas=dict(definition["cohort_quotas"]), query_identities=query_identity,
        records=records, examples=expected, native_path_names=list(NATIVE_PATHS),
        sorted_by="position_local_bp_then_example_index",
        labels_present=False, query_truth_read=False,
        sampling_policy={k: plan[k] for k in (
            "seed", "parent_integer_quota_policy", "site_policy", "query_order_policy",
            "remaining_sampling_choices_are_disclosed_reconstruction")})

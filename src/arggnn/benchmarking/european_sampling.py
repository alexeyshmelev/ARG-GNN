"""Paper-sized AP sampling on frozen observed SNPs, separate from true labels.

The supplement specifies counts and evenly spaced sites, but not integer
allocation across parents, a site-grid phase or haplotype tie order. This module
freezes those reconstruction choices explicitly. It does not read truth or
choose examples using ancestry labels. Native six-path labels are joined later.
"""
from __future__ import annotations

import hashlib
import math
import random

COHORTS = ("present_day", "Bronze_Age", "BAA", "Neo", "Yam")
QUERY_COUNTS = (182, 162, 14, 302, 26)
NATIVE_PATHS = ("Ana_nonBAA", "CHG_nonBAA", "WHG", "EHG", "Ana_viaBAA", "CHG_viaBAA")
LENGTH = 200_000_000


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def paper_sampling_plan(train_parent_ids, test_parent_ids, *, seed):
    """Three independent TRAIN parents, five test parents, exact paper totals."""
    train, test = tuple(train_parent_ids), tuple(test_parent_ids)
    _need(len(train) == 3 and len(test) == 5
          and all(isinstance(p, str) and p for p in train + test)
          and len(set(train + test)) == 8
          and type(seed) is int and 0 <= seed < 2**32,
          "Explicit independent three-TRAIN/five-test identities and sampling seed required")
    parents = {}
    for index, parent in enumerate(train):
        quotas = {cohort: 3333 + int(index == c % 3)
                  for c, cohort in enumerate(COHORTS)}
        parents[parent] = dict(split="train", cohort_quotas=quotas)
    for parent in test:
        parents[parent] = dict(split="test", cohort_quotas={c: 2000 for c in COHORTS})
    return dict(schema="ap_paper_sampled_plan_v1", parents=parents,
        train_parent_ids=list(train), test_parent_ids=list(test), seed=seed,
        sequence_length_bp=LENGTH, native_path_names=list(NATIVE_PATHS),
        training_total=50000, training_per_cohort=10000, testing_per_parent=10000,
        parent_integer_quota_policy="3333 plus one rotated across three TRAIN parents per cohort",
        site_policy="midpoints of equal bins in retained observed SNP rank, not bp",
        query_order_policy="source sample order with deterministic per-parent/cohort permutation",
        site_sampling_uses_truth=False, duplicate_focal_filter=False,
        paper_explicit_counts_and_even_spacing=True,
        remaining_sampling_choices_are_disclosed_reconstruction=True,
        released_class_cap_sampler_reproduced=False)


def build_sampled_registry(plan, parent_id, observed_rows, inference_metadata, *,
                           observation_ref):
    """Return position-sorted observed records with stable canonical example IDs."""
    _need(plan.get("schema") == "ap_paper_sampled_plan_v1"
          and parent_id in plan["parents"], "Frozen paper sampling plan parent required")
    definition = plan["parents"][parent_id]
    split = definition["split"]
    _need(split in ("train", "test") and plan["sequence_length_bp"] == LENGTH,
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
        cohort_quotas=dict(definition["cohort_quotas"]), query_identities=query_identity,
        records=records, examples=expected, native_path_names=list(NATIVE_PATHS),
        sorted_by="position_local_bp_then_example_index",
        labels_present=False, query_truth_read=False,
        sampling_policy={k: plan[k] for k in (
            "seed", "parent_integer_quota_policy", "site_policy", "query_order_policy",
            "remaining_sampling_choices_are_disclosed_reconstruction")})


def native_truth_labels(registry, target_schema, truth):
    """Pure same-site native-six join. No file I/O or sampling changes.

    A TRAIN caller may pass its separately authenticated supervised targets.
    An external evaluator may call this only after its all-method prediction
    barrier. This pure join provides no capability to resolve private files.
    """
    import numpy as np
    _need(registry.get("schema") == "ap_paper_sampled_registry_v1"
          and target_schema["parent_id"] == registry["parent_id"]
          and target_schema["split"] == registry["split"]
          and target_schema["native_paths_1based"] == [1, 2, 3, 4, 5, 6],
          "Same-parent native-six target schema required")
    intervals = np.asarray(truth["intervals_bp"])
    paths = np.asarray(truth["native_paths_1based"])
    raw = np.asarray(truth["query_raw_node_ids"])
    _need(intervals.ndim == 2 and intervals.shape[1] == 2 and len(intervals) > 0
          and np.isfinite(intervals).all() and intervals[0, 0] == 0
          and intervals[-1, 1] == LENGTH and (intervals[:, 1] > intervals[:, 0]).all()
          and np.array_equal(intervals[:-1, 1], intervals[1:, 0])
          and raw.shape == (686,) and raw.dtype.kind in "iu"
          and len(set(raw.tolist())) == 686
          and paths.shape == (len(intervals), 686) and paths.dtype.kind in "iu"
          and ((paths >= 1) & (paths <= 6)).all(),
          "Complete native-six truth domain and actual query columns required")
    _need(np.array_equal(raw, target_schema["query_sample_indices"])
          and len(target_schema["query_keys"]) == 686,
          "Explicit raw sample and semantic query-key target mapping required")
    by_key = {tuple(key): (int(sample), column)
              for column, (key, sample) in enumerate(zip(target_schema["query_keys"], raw))}
    _need(len(by_key) == 686 and len(registry["query_identities"]) == 686
          and all(by_key.get((r["sample_id"], r["haplotype"]), (None,))[0] == r["sample_index"]
                  for r in registry["query_identities"]),
          "Actual biological query identities must agree across observed/truth namespaces")
    result = np.empty(registry["examples"], dtype=np.int64)
    for record in registry["records"]:
        position = record["position_original_bp"]
        _need(0 <= position < LENGTH, "No same-site truth extrapolation")
        column = by_key[(record["sample_id"], record["haplotype"])][1]
        row = int(np.searchsorted(intervals[:, 1], position, side="right"))
        path = int(paths[row, column])
        _need(path in ((5, 6) if record["cohort"] == "BAA" else (1, 2, 3, 4)),
              "Native BAA and non-BAA path meanings must remain distinct")
        result[record["example_index"]] = path - 1
    return result


def registry_arrays(registry):
    """Canonical example order shared by all models, retaining every sample/site."""
    import numpy as np
    _need(registry.get("schema") == "ap_paper_sampled_registry_v1"
          and registry.get("native_path_names") == list(NATIVE_PATHS),
          "Canonical AP sampled registry and native class order required")
    records = sorted(registry["records"], key=lambda row: row["example_index"])
    _need(len(records) == registry["examples"]
          and [r["example_index"] for r in records] == list(range(len(records))),
          "Every original canonical example must occur exactly once")
    return dict(
        record_ids=np.asarray([r["example_index"] for r in records], dtype=np.int64),
        sample_nodes=np.asarray([r["query_node_id"] for r in records], dtype=np.int64),
        sample_indices=np.asarray([r["sample_index"] for r in records], dtype=np.int64),
        positions_bp0=np.asarray([r["position_original_bp"] for r in records], dtype=np.float64),
        parent_id=np.asarray(registry["parent_id"]),
        class_order=np.asarray(NATIVE_PATHS))

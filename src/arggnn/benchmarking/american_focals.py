"""Exact compact American query focals, without native training corpora.

The original full raw-path record IDs, NumPy without-replacement selection and
first-occurrence midpoint are preserved. Because a path includes its sample node,
deduplication sets can be discarded between samples. All samples/trees are still
traversed to recover exact IDs. Selected query samples are traversed again to
recover first occurrences. No vocabulary, reference corpus, MLM shard, path/train
split, occurrence mapping, private truth or model is created or opened.

Import and prepare are inert with respect to numerical payloads and execution.
ROOT must provide actual CPU admission and an outer wall/RSS/output supervisor.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import re
import resource
import time

from . import american_observation as observation
from . import american_preparation as generation

ROOT = generation.ROOT
SCHEMA = "american_fresh_compact_focal_stage_v1"
RESULT_SCHEMA = "american_fresh_compact_focal_preparation_v1"
CAP_KEYS = ("max_unique_paths", "max_path_tokens", "max_path_occurrences",
            "max_sample_unique_paths", "max_sample_path_tokens", "max_raw_path_nodes",
            "max_output_bytes", "max_seconds", "max_rss_bytes")
CONVENTIONS = dict(unique_raw_paths_before_truncation=True, count=10000,
                   query_selection="label_blind_without_replacement",
                   coordinate="midpoint_of_first_observed_occurrence",
                   record_ids="unchanged_full_original_raw_path_order")
SOURCE_REFS = (
    {"path": "src/arggnn/benchmarking/american_preparation.py",
     "sha256": "a3778cd8b11cd7064233c4bf55f14e38815c95ffb6341e36208917b9e3989b02"},
    {"path": "src/arggnn/benchmarking/american_observation.py",
     "sha256": "edeae32bf6dcc740030a595e562484f43cdabca3318e1456b6f7e9f7f74b2c59"},
    {"path": "src/arggnn/american_c3_common_v1.py",
     "sha256": "615d3892f1df7e9de39e57991eab883b2eca1af9d13e424661541aac4b2d2abf"},
    {"path": "src/arggnn/american_c3_paths_v1.py",
     "sha256": "3b32f16e0a81160852e5b93fe9033abf9bc2eeb56c3b15ca552d73f4aacf374c"},
    {"path": ".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py",
     "sha256": "e7e66a029694b21a2ba1f9c9a6c4fbc193d81b8831282580ba9639a9e224726c"},
)


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _public_ref(ref):
    _need(type(ref) is dict and set(ref) == {"path", "sha256"},
          "Exact public observed reference required")
    _need(not any(part.lower() in {"private", "truth", "oracle", "census"}
                  for part in Path(ref["path"]).parts), "No private/truth/oracle input")
    return generation._path(ref["path"])


def _read(ref):
    _public_ref(ref)
    return generation._read(ref)


def requirements():
    for ref in SOURCE_REFS:
        _need(generation._reference(ref["path"]) == ref,
              "Pinned compact focal dependency changed: " + ref["path"])
    return dict(source_refs=[*map(dict, SOURCE_REFS), generation._reference(__file__)],
                conventions=copy.deepcopy(CONVENTIONS), numerical_payloads_opened=False,
                full_first_pass_required=True, second_pass_selected_query_samples=True,
                native_training_corpus_produced=False, jobs_started=0)


def output_bound():
    # 10,000 rows: three int64 vectors, float64 coordinate and interval pair.
    # Unicode scalars/headers, ZIP expansion and all metadata fit this margin.
    return 10000 * 48 + 2 * 2**20


def build_spec(*, input_ref, output_directory, **caps):
    _need(set(caps) == set(CAP_KEYS), "All finite whole-parent caps must be explicit")
    spec = dict(schema=SCHEMA, input_ref=copy.deepcopy(input_ref),
                output_directory=str(generation._path(output_directory).relative_to(ROOT)),
                declared_focal_conventions=copy.deepcopy(CONVENTIONS),
                source_refs=requirements()["source_refs"], **caps)
    prepare(spec)
    return spec


def prepare(spec):
    """Validate public metadata and sources without opening a tree or array."""
    _need(spec.get("schema") == SCHEMA and spec.get("declared_focal_conventions") == CONVENTIONS,
          "Explicit exact compact focal stage/conventions required")
    _need(all(type(spec.get(k)) is int and spec[k] > 0 for k in CAP_KEYS),
          "Finite positive whole-parent/per-sample count, wall, RSS and output caps required")
    _need(spec["max_output_bytes"] >= output_bound(), "Complete focal serialization bound exceeds cap")
    _need(spec["source_refs"] == requirements()["source_refs"], "Exact compact source closure required")
    source = _read(spec["input_ref"])
    _need(source.get("schema") == "american_fresh_inference_ready_v1"
          and source.get("requires_tsinfer_tsdate") is False
          and source.get("query_truth_in_training_inputs") is False
          and source.get("truth_scoring_authorized") is False
          and source.get("exact_paper_reproduction") is False
          and source.get("split") in ("train", "val", "test")
          and len(source.get("parent_id", "")) <= 128
          and re.fullmatch(r"argformer_american_[A-Za-z0-9_.-]+", source.get("parent_id", "")) is not None
          and source.get("declared_conventions") == generation.CONVENTIONS
          and source.get("declared_inference_conventions") == observation.INFERENCE_CONVENTIONS,
          "Actual complete fresh shared inference lineage required")
    seed = source.get("seeds", {}).get("sampling")
    _need(type(seed) is int and seed >= 0, "Original same-parent sampling seed required")
    observed = _read(source["observation_ref"])
    completion = _read(source["stage_completion_ref"])
    _need(completion.get("status") == "completed_current_C3_inference"
          and completion.get("observation_ref") == source["observation_ref"],
          "Actual inferred observation must bind its genuine completion")
    _need(observed.get("schema") == "american_c3_observed_v1"
          and observed.get("parent_id") == source["parent_id"]
          and observed.get("split") == source["split"]
          and observed.get("class_order") == ["EUR", "EAS", "AFR"]
          and observed.get("sequence_length_bp") == 30000000
          and observed.get("genealogy_kind") == observation.GENEALOGY
          and observed.get("query_ancestry_truth_in_inputs") is False
          and observed.get("sites_and_sample_identities_verified") is True
          and observed.get("migrations_and_internal_population_labels_removed") is True,
          "Full shared sanitized 30Mb inferred observation required")
    for key in ("tree_ref", "roster_ref", "genotypes_ref", "positions_ref", "alleles_ref"):
        _public_ref(observed[key])
    roster = _read(observed["roster_ref"])
    rows = roster["samples"]
    _need(roster.get("parent_id") == source["parent_id"]
          and roster.get("class_order") == observed["class_order"]
          and len(rows) == 1600
          and [r["sample_index"] for r in rows] == list(range(1600))
          and len({r["observed_node_id"] for r in rows}) == 1600
          and [r["role"] for r in rows] == ["pure_source_reference"] * 600 + ["query"] * 1000,
          "All original 600 references and 1000 query columns must remain ordered")
    _need(all(r["observed_reference_class"] is None for r in rows[600:])
          and all(type(r["observed_reference_class"]) is int and 0 <= r["observed_reference_class"] < 3
                  for r in rows[:600])
          and [sum(r["observed_reference_class"] == c for r in rows[:600]) for c in range(3)] == [200]*3
          and len({(r["sample_id"], r["haplotype"]) for r in rows}) == 1600
          and all(math.isfinite(r["age_generations"]) and r["age_generations"] >= 0 for r in rows),
          "Known reference classes and full label-blind query/sample/date support required")
    out = generation._path(spec["output_directory"])
    input_root = generation._path(spec["input_ref"]["path"]).parent
    _need(out.is_relative_to(ROOT / "experiments/results") and source["parent_id"] in out.parts
          and not out.exists() and not out.is_relative_to(input_root),
          "Fresh isolated same-parent compact focal scope required")
    return dict(source=source, observed=observed, rows=rows,
                output_directory=str(out), sampling_seed=seed, execution_authorized=False)


def _exact_focals(ts, rows, cfg, *, path_key, resource_check):
    """Two exact passes with per-sample raw-key deduplication; tiny fakes test this."""
    import numpy as np
    samples = list(map(int, ts.samples()))
    _need(samples == [r["observed_node_id"] for r in rows]
          and [r["sample_index"] for r in rows] == list(range(len(rows))),
          "Original sample order required before exact raw-path deduplication")
    roles = [r["role"] for r in rows]
    references = roles.count("pure_source_reference")
    _need(roles == ["pure_source_reference"]*references + ["query"]*(len(rows)-references),
          "Original reference prefix/query suffix required for exact global IDs")
    occurrences = len(samples) * ts.num_trees
    _need(occurrences <= cfg["max_path_occurrences"], "Complete raw-path occurrence cap exceeded")
    counts, total, tokens, reference_count, traversed, maximum = [], 0, 0, 0, 0, 0

    def unique_keys(sample):
        nonlocal traversed, maximum
        seen, sample_tokens = set(), 0
        tree_count = 0
        for tree in ts.trees():
            _need(tree.index == tree_count, "Every original tree must remain in chronological order")
            tree_count += 1
            traversed += 1
            if traversed % 1024 == 0:
                resource_check("compact_focal_raw_path_traversal")
            key = path_key(tree, sample)
            _need(type(key) is tuple and key and key[0] == sample
                  and len(key) <= cfg["max_raw_path_nodes"],
                  "Complete original raw path required; no truncated deduplication")
            maximum = max(maximum, len(key))
            if key in seen:
                continue
            _need(len(seen) < cfg["max_sample_unique_paths"], "Per-sample unique-path cap exceeded")
            sample_tokens += len(key)
            _need(sample_tokens <= cfg["max_sample_path_tokens"], "Per-sample raw-token cap exceeded")
            seen.add(key)
            yield key, tree
        _need(tree_count == ts.num_trees, "Every original tree must be traversed for each selected sample")

    for row, sample in zip(rows, samples):
        start, before_tokens = total, tokens
        for key, tree in unique_keys(sample):
            total += 1
            tokens += len(key)
            _need(total <= cfg["max_unique_paths"] and tokens <= cfg["max_path_tokens"],
                  "Whole-parent exact unique-path/token cap exceeded; no subset fallback")
        counts.append(dict(sample_index=row["sample_index"], sample_node=sample,
                           first_record_id=start, unique_paths=total-start,
                           path_tokens=tokens-before_tokens, role=row["role"]))
        if row["role"] == "pure_source_reference":
            reference_count = total
        resource_check("compact_focal_first_pass_sample_complete")
    _need(traversed == occurrences, "Complete first-pass sample/tree support required")
    query_count = total-reference_count
    _need(query_count >= cfg["query_focal_count"], "Insufficient complete unique query paths")
    # This array is exactly original query_ids, including the reference offset.
    # Keeping it explicit preserves the identical NumPy choice call and RNG use.
    population = np.arange(reference_count, total, dtype=np.int64)
    chosen = np.random.default_rng(cfg["path_sampling_seed"]).choice(
        population, size=cfg["query_focal_count"], replace=False)
    del population
    chosen.sort()
    n = len(chosen)
    sample_nodes, sample_indices = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    intervals = np.empty((n, 2), dtype=np.float64)
    found, selected_samples = 0, 0
    for record in counts[references:]:
        start = record["first_record_id"]
        stop = start+record["unique_paths"]
        lo, hi = np.searchsorted(chosen, [start, stop])
        if lo == hi:
            continue
        selected_samples += 1
        wanted = {int(chosen[i]): i for i in range(int(lo), int(hi))}
        next_id = start
        for key, tree in unique_keys(record["sample_node"]):
            index = wanted.get(next_id)
            if index is not None:
                sample_nodes[index] = key[0]
                sample_indices[index] = record["sample_index"]
                intervals[index] = [tree.interval.left, tree.interval.right]
                found += 1
            next_id += 1
        _need(next_id == stop, "Exact second-pass path identity/count drift")
        resource_check("compact_focal_second_pass_sample_complete")
    _need(found == n and traversed <= 2*occurrences, "Every selected first occurrence must be recovered")
    arrays = dict(record_ids=chosen, sample_nodes=sample_nodes, sample_indices=sample_indices,
                  positions_bp0=(intervals[:, 0]+intervals[:, 1])/2,
                  first_occurrence_intervals_bp=intervals)
    return arrays, dict(unique_paths=total, query_unique_paths=query_count,
                        reference_unique_paths=reference_count, path_tokens=tokens,
                        full_native_path_occurrences=occurrences,
                        actual_path_occurrences_traversed=traversed,
                        selected_query_samples=selected_samples, maximum_raw_path_nodes=maximum,
                        per_sample_counts=counts)


def run(spec, *, resource_check, execution_check):
    """Produce exact shared focals inside ROOT's admitted CPU-only supervisor."""
    _need(callable(resource_check) and callable(execution_check),
          "Actual external admission and resource checks required")
    plan = prepare(spec)
    execution_check()
    _need(Path.cwd().resolve() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
          "Explicit CPU-only repository worker required")
    deadline = time.monotonic()+spec["max_seconds"]
    last_approval = float("-inf")

    def check(boundary):
        nonlocal last_approval
        now = time.monotonic()
        _need(now < deadline, "Compact focal whole-parent wall cap exceeded")
        _need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= spec["max_rss_bytes"],
              "Compact focal process peak RSS cap exceeded")
        if now-last_approval >= 2:
            execution_check()
            last_approval = now
        resource_check(boundary)

    check("compact_focal_before_scientific_import")
    import numpy as np
    import tskit
    from arggnn import american_c3_paths_v1 as original
    tree_path = _public_ref(plan["observed"]["tree_ref"])
    _need(generation._reference(tree_path) == plan["observed"]["tree_ref"], "Actual shared tree SHA required")
    check("compact_focal_before_tree_load")
    ts = tskit.load(str(tree_path))
    rows = plan["rows"]
    tables = ts.tables
    _need(ts.num_samples == 1600 and ts.sequence_length == 30000000 and ts.num_nodes < 2**31
          and ts.num_migrations == ts.num_populations == ts.num_individuals == ts.num_provenances == 0
          and not tables.metadata and (tables.nodes.population == -1).all()
          and (tables.nodes.individual == -1).all()
          and all(len(table.metadata) == 0 for table in
                  (tables.nodes, tables.edges, tables.sites, tables.mutations))
          and np.array_equal(ts.nodes_time[ts.samples()], [r["age_generations"] for r in rows]),
          "Full sanitized shared inferred tree/sample dates required")
    check("compact_focal_tree_loaded")
    cfg = dict(spec, query_focal_count=10000, path_sampling_seed=plan["sampling_seed"])
    arrays, counts = _exact_focals(ts, rows, cfg,
                                  path_key=original.author_functions()["get_path_key"], resource_check=check)
    source = plan["source"]
    arrays.update(parent_id=np.asarray(source["parent_id"]), class_order=np.asarray(["EUR", "EAS", "AFR"]))
    check("compact_focal_selection_complete_before_output")
    writer = generation._Writer(generation._path(plan["output_directory"]), spec["max_output_bytes"], check)
    focal_ref = writer.npz("focals.npz", **arrays)
    count_ref = writer.json("exact_path_counts.json", counts)
    result = dict(schema=RESULT_SCHEMA, status="completed_compact_query_focals",
                  parent_id=source["parent_id"], split=source["split"],
                  observation_ref=source["observation_ref"], input_ref=spec["input_ref"],
                  focals_ref=focal_ref, class_order=["EUR", "EAS", "AFR"],
                  genealogy_kind=observation.GENEALOGY, sequence_length_bp=30000000,
                  query_focal_count=10000, path_sampling_seed=plan["sampling_seed"],
                  declared_focal_conventions=copy.deepcopy(CONVENTIONS), exact_path_counts_ref=count_ref,
                  all_original_samples_and_trees_traversed=True, original_record_ids_preserved=True,
                  query_ancestry_truth_opened=False, query_truth_in_training_inputs=False,
                  truth_scoring_authorized=False, exact_paper_reproduction=False,
                  native_training_corpus_produced=False, native_path_train_split_produced=False,
                  source_refs=copy.deepcopy(spec["source_refs"]),
                  campaign_ref=copy.deepcopy(source["campaign_ref"]),
                  comparison_ref=copy.deepcopy(source["comparison_ref"]),
                  recipe_ref=copy.deepcopy(source["recipe_ref"]), automatic_retry=False)
    result_ref = writer.json("compact_focal_preparation.json", result)
    _need(not writer.pending, "Every compact artifact must actually finish")
    check("compact_focal_complete")
    return result_ref

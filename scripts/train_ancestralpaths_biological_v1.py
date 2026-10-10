#!/usr/bin/env python
"""H020 source-native AncestralPaths biological CPU fit; no paper-CNN claim."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
DEFAULT_CONFIG = "configs/ancestralpaths_biological_pilot_v1.json"
from arggnn.ancestralpaths_native_bridge import (
    COHORT_FEATURE_ORDER, PATH_TO_SOURCE_CLASS, ObservedPanel, extract_native_features,
    build_native_keras_model,
    collapse_native_probabilities, inspect_author_source, native_accuracy_confusion,
    native_model_spec,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_path(value):
    p = (ROOT / value).resolve()
    p.relative_to(ROOT)
    return p


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def check_file(record):
    p = local_path(record["path"])
    if not p.is_file() or digest(p) != record["sha256"]:
        raise ValueError("Missing or changed bound file: " + str(p))
    return p


def protocol(config_path):
    cfg = json.loads(config_path.read_text())
    if cfg["hypothesis_id"] != "H020" or cfg["training"]["seeds"] != [101, 202, 303]:
        raise ValueError("Protocol identity/seeds differ from the frozen pilot")
    if (cfg["training"]["epochs"], cfg["training"]["batch_size"],
        cfg["training"]["shuffle"]) != (5, 32, True):
        raise ValueError("Source-default fitting settings differ")
    if digest(Path(__file__)) != cfg["implementation_sha256"]:
        raise ValueError("Runner differs from the frozen protocol hash")
    for name, expected in cfg["frozen_files_sha256"].items():
        check_file({"path": name, "sha256": expected})
    source = inspect_author_source(local_path(cfg["author_source"]))
    if source["commit"] != cfg["author_commit"] or native_model_spec() != cfg["native_model"]:
        raise ValueError("Author source or native model specification drift")
    dataset = json.loads(local_path(cfg["dataset_manifest"]).read_text())
    rows = dataset["replicates"]
    expected = cfg["parent_partition"]
    actual = {name: [r["parent_id"] for r in rows if r["split"] == name]
              for name in ("train", "val", "test")}
    if actual != expected or len(set(sum(actual.values(), []))) != 4:
        raise ValueError("Require the frozen two TRAIN / two VAL / zero TEST parents")
    if not any(x.get("id") == "H020" for x in
               json.loads((ROOT / "research/experiment_decisions.json").read_text()).get("decisions", [])):
        # The repository's ledger format may nest its entries differently.
        if '"H020"' not in (ROOT / "research/experiment_decisions.json").read_text():
            raise ValueError("H020 has not been registered")
    parents = []
    for row in rows:
        directory = local_path(row["directory"])
        bound = {}
        # Observed schema/source preflight never opens truth targets.
        for rel in ("observed/sample_manifest.json", "observed/scoring_coordinates.npz",
                    "observed/population_labels.txt", "observed/sample_order.txt"):
            bound[rel] = check_file({"path": str((directory / rel).relative_to(ROOT)),
                                    "sha256": row["files"][rel]["sha256"]})
        samples = json.loads(bound["observed/sample_manifest.json"].read_text())
        if len(samples) != 280 or [r["sample_index"] for r in samples] != list(range(280)):
            raise ValueError("Require the frozen ordered 280-haplotype manifest")
        fields = ("sample_id", "haplotype", "observed_cohort", "age_generations", "role")
        observed = [{k: r[k] for k in fields} for r in samples]
        if any(r["observed_cohort"] not in set(COHORT_FEATURE_ORDER) | {"present_day", "GBR"}
               for r in observed):
            raise ValueError("Undeclared observed cohort")
        for i in range(0, 280, 2):
            a, b = observed[i:i + 2]
            if a["sample_id"] != b["sample_id"] or (a["haplotype"], b["haplotype"]) != (0, 1):
                raise ValueError("Explicit paired diploid order is required")
            if any(a[k] != b[k] for k in ("observed_cohort", "age_generations", "role")):
                raise ValueError("Diploid observed attributes disagree")
        query_slots = np.array([i for i, r in enumerate(observed) if r["role"] == "query"],
                               dtype=np.int32)
        cohorts = np.array([observed[i]["observed_cohort"] for i in query_slots])
        if len(query_slots) != 120 or int(np.sum(cohorts == "present_day")) != 64:
            raise ValueError("Require 120 explicit queries, including 64 present-day")
        with np.load(bound["observed/scoring_coordinates.npz"], allow_pickle=False) as z:
            positions, edges = z["positions_bp"].copy(), z["bin_edges_bp"].copy()
        if not np.array_equal(edges, np.arange(0, 2000001, 100, dtype=float)):
            raise ValueError("Frozen complete physical 100-bp bins are required")
        if not np.array_equal(positions, (edges[:-1] + edges[1:]) / 2):
            raise ValueError("Fixed midpoint coordinates differ")
        parents.append(dict(parent_id=row["parent_id"], split=row["split"], row=row,
                            samples=observed, raw_query_nodes=np.array(
                                [samples[i]["observed_node_id"] for i in query_slots]),
                            query_slots=query_slots, cohorts=cohorts,
                            positions=positions, edges=edges, directory=directory))
    return cfg, parents, source


def observations(manifest_path, parents, cfg):
    """Consume the finalized observed-only Relate producer, retaining physical joins."""
    spec = json.loads(manifest_path.read_text())
    if (spec.get("kind") != "biological_paths_relate_v1"
            or spec.get("status") != "four_development_parents_inferred_pre_and_post_five_iterations"
            or spec.get("primary_condition") != "post_five_size_iterations_primary"
            or spec.get("truth_or_oracle_read") is not False):
        raise ValueError("Require the completed biological Relate primary-condition manifest")
    producer = cfg["relate_producer"]
    if (spec.get("implementation_sha256") != producer["implementation_sha256"]
            or spec.get("config_sha256") != producer["config_sha256"]):
        raise ValueError("Relate producer source/config differ from the bound contract")
    entries = spec["replicates"]
    by_id = {r["parent_id"]: r for r in entries}
    if len(entries) != 4 or len(by_id) != 4 or set(by_id) != {p["parent_id"] for p in parents}:
        raise ValueError("Observed manifest must contain exactly the four frozen parents")
    result = {}
    for parent in parents:
        r = by_id[parent["parent_id"]]
        if (r.get("split") != parent["split"] or r.get("status") != "inferred_both_conditions"
                or r.get("truth_or_oracle_read") is not False):
            raise ValueError("Observed parent split/completion/provenance mismatch")
        tree_record = r["inferred_tree"]
        if tree_record.get("condition") != "post_five_size_iterations_primary":
            raise ValueError("Require actual post-five-size-iteration primary inferred trees")
        tree_path = check_file(tree_record)
        tree_parts = [part.lower() for part in tree_path.relative_to(ROOT).parts]
        if any("oracle" in part or "truth" in part or "census" in part for part in tree_parts):
            raise ValueError("Oracle/truth trees cannot be primary inferred input")
        derived_record = r["observed_sample_manifest"]
        derived_path = check_file(derived_record)
        source_record = derived_record["source_manifest"]
        source_path = check_file(source_record)
        expected_hash = parent["row"]["files"]["observed/sample_manifest.json"]["sha256"]
        expected_path = parent["directory"] / "observed/sample_manifest.json"
        if (source_record["sha256"] != expected_hash or source_path != expected_path
                or r["original_observed_sample_manifest"] != source_record):
            raise ValueError("Derived roster must bind the separate frozen source observed roster")
        source_rows = json.loads(source_path.read_text())
        derived_rows = json.loads(derived_path.read_text())
        mapping = r["sample_index_to_inferred_tree_node_id"]
        if (derived_record.get("rows") != 280 or len(derived_rows) != 280
                or len(mapping) != 280
                or [m["sample_index"] for m in mapping] != list(range(280))):
            raise ValueError("Require the complete ordered inferred sample-index mapping")
        nodes = np.asarray([m["inferred_tree_node_id"] for m in mapping])
        if (nodes.shape != (280,) or not np.issubdtype(nodes.dtype, np.integer)
                or len(set(nodes.tolist())) != 280 or np.any(nodes < 0)):
            raise ValueError("Actual inferred sample node IDs must be distinct nonnegative integers")
        for source, derived, joined in zip(source_rows, derived_rows, mapping):
            expected = dict(source)
            expected["source_observed_node_id"] = expected.pop("observed_node_id", None)
            expected["inferred_tree_node_id"] = joined["inferred_tree_node_id"]
            expected["observed_node_id"] = joined["inferred_tree_node_id"]
            if derived != expected:
                raise ValueError("Derived roster changed observed attributes or inferred identity")
        completion_path = check_file(r["completion_provenance"])
        completion = json.loads(completion_path.read_text())
        if completion != {k: v for k, v in r.items() if k != "completion_provenance"}:
            raise ValueError("Hashed parent completion receipt differs from its manifest record")
        check_file(r["commands_provenance"])
        coordinate = r["coordinate_contract"]
        if (coordinate["original_parent_interval_bp0"] != [0, 2000000]
                or coordinate["normalization_length_bp"] != 2000000
                or coordinate["physical_coordinate_transform"] !=
                   "physical_original = local_tree_position + coverage_left"
                or coordinate["uncovered_tails"] != "excluded; no extrapolation"):
            raise ValueError("Physical domain/normalization/transform differs from the producer")
        coverage = np.asarray(coordinate["coverage_original_bp0"], dtype=np.float64)
        if (coverage.shape != (2,) or not np.isfinite(coverage).all()
                or not 0 <= coverage[0] < coverage[1] <= 2000000
                or coordinate["local_to_original_offset_bp"] != coverage[0]):
            raise ValueError("Invalid clipped physical coverage/local offset")
        grid_record = coordinate["fixed_grid"]
        grid_path = check_file(grid_record)
        expected_grid = parent["directory"] / "observed/scoring_coordinates.npz"
        if (grid_path != expected_grid or grid_record["sha256"] !=
                parent["row"]["files"]["observed/scoring_coordinates.npz"]["sha256"]
                or grid_record["total"] != 20000):
            raise ValueError("Retained grid must bind the original frozen physical grid")
        retained = np.asarray(coordinate["retained_fixed100bp_indexes"])
        expected_retained = np.flatnonzero((parent["positions"] >= coverage[0]) &
                                          (parent["positions"] < coverage[1]))
        if (retained.ndim != 1 or not np.issubdtype(retained.dtype, np.integer)
                or not np.array_equal(retained, expected_retained) or not len(retained)):
            raise ValueError("Retain exactly declared original centers within observed coverage")
        if np.any(np.diff(retained) != 1):
            raise ValueError("This producer's single physical span requires contiguous retained indexes")
        post = r["post_five_iterations"]
        if (post["clean_tree"] != tree_record["path"]
                or post["clean_tree_sha256"] != tree_record["sha256"]
                or post["sample_index_to_inferred_tree_node_id"] != mapping
                or post["observed_sample_manifest"] != derived_record
                or post["coverage_original_bp0"] != coordinate["coverage_original_bp0"]
                or post["local_to_original_offset_bp"] != coverage[0]
                or post["sequence_length_local_bp"] != coverage[1] - coverage[0]
                or post["retained_fixed100bp_indexes"] != retained.tolist()):
            raise ValueError("Top-level primary join disagrees with its condition-specific receipt")
        # Include only bins whose original fixed center is retained. Clip their
        # outer physical edges to coverage; do not fill partial bins lacking centers.
        edges = parent["edges"][np.r_[retained, retained[-1] + 1]].copy()
        edges[0] = max(edges[0], coverage[0])
        edges[-1] = min(edges[-1], coverage[1])
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Retained clipped physical bins have nonpositive spans")
        result[parent["parent_id"]] = dict(
            tree_path=tree_path, nodes=nodes.astype(np.int64), entry=r,
            receipt_path=completion_path, derived_roster_path=derived_path,
            source_roster_path=source_path, coverage=coverage,
            offset=float(coverage[0]), retained_indexes=retained.astype(np.int32),
            positions=parent["positions"][retained].copy(), bin_edges=edges,
            site_join=post["site_join"])
    return result

def feature_row(signature):
    """Preserve native float64 repeated-add normalization, then Keras float32."""
    x = np.zeros((5, 9), dtype=np.float64)
    for depth, descriptor in enumerate(signature):
        age, counts = descriptor[0], descriptor[1:]
        add = 1.0 / sum(counts)
        for c, count in enumerate(counts):
            for _ in range(count):
                x[depth, c] += add
        x[depth, 8] = age / 1500
    x[len(signature):] = -15
    return x.astype(np.float32)


def cache_features(parent, observation, output):
    """Observed-only descendant counts and unique five-shell paths; no truth reader."""
    import tskit
    ts = tskit.load(observation["tree_path"])
    nodes = observation["nodes"]
    if (ts.sequence_length != observation["coverage"][1] - observation["coverage"][0]
            or set(ts.samples()) != set(nodes.tolist())):
        raise ValueError("Inferred physical domain/sample join differs")
    queries = nodes[parent["query_slots"]]
    for i, node in enumerate(nodes):
        if not np.isclose(ts.node(int(node)).time, parent["samples"][i]["age_generations"],
                          atol=1e-6, rtol=0):
            raise ValueError("Inferred sample ages do not match the declared observed ages")
    panel = ObservedPanel(
        tuple(parent["samples"][i]["sample_id"] for i in range(0, 280, 2)),
        tuple(map(int, nodes)),
        {int(u): r["observed_cohort"] for u, r in zip(nodes, parent["samples"])})
    positions = observation["positions"]
    bin_edges = observation["bin_edges"]
    sites = np.array([s.position + observation["offset"] for s in ts.sites()],
                     dtype=np.float64)
    join = observation["site_join"]
    if (len(join) != len(sites) or any(
            row["native_site_id"] != i or row["original_position_0based"] != sites[i]
            or row["local_position_0based"] != sites[i] - observation["offset"]
            for i, row in enumerate(join))):
        raise ValueError("Actual inferred SNP positions differ from the declared physical site join")
    if parent["split"] == "train" and not len(sites):
        raise ValueError("Native TRAIN sampler requires actual inferred-tree observed sites")
    coordinates = np.unique(np.concatenate((positions, sites)))
    all_index = np.empty((len(coordinates), len(queries)), dtype=np.int32)
    counts = np.zeros((ts.num_nodes, 8), dtype=np.int32)
    group = {int(u): (COHORT_FEATURE_ORDER.index(r["observed_cohort"])
                     if r["observed_cohort"] in COHORT_FEATURE_ORDER else 8)
             for u, r in zip(nodes, parent["samples"])}
    path_to_index, bank = {}, []
    visited_trees, evaluated_starts = 0, 0
    for tree in ts.trees():
        lo, hi = np.searchsorted(coordinates,
                                 (tree.interval.left + observation["offset"],
                                  tree.interval.right + observation["offset"]))
        if lo == hi:
            continue
        visited_trees += 1
        # Only actual leaves count, as in native tree.leaves(parent); no populations,
        # census flags, internal metadata, or truth tree properties are read.
        for u in tree.nodes(order="postorder"):
            counts[u] = 0
            children = tuple(tree.children(u))
            if children:
                for child in children:
                    counts[u] += counts[child]
            elif group.get(u, 8) < 8:
                counts[u, group[u]] = 1
        starts = {}
        tree_indices = []
        for q in queries:
            start = tree.parent(int(q))
            if start not in starts:
                evaluated_starts += 1
                u, previous, signature = start, np.zeros(8, dtype=np.int32), []
                while u != tskit.NULL and len(signature) < 5:
                    current = counts[u]
                    shell = current - previous
                    if np.any(shell < 0):
                        raise ValueError("Inferred descendant shells are not nested")
                    if shell.sum():
                        age = tree.time(u)
                        if not np.isfinite(age) or age < 0:
                            raise ValueError("Invalid observed ancestor age")
                        signature.append((int(age), *map(int, shell)))
                        previous = current.copy()
                    u = tree.parent(u)
                signature = tuple(signature)
                if signature not in path_to_index:
                    path_to_index[signature] = len(bank)
                    bank.append(feature_row(signature))
                starts[start] = path_to_index[signature]
            tree_indices.append(starts[start])
        if visited_trees == 1:
            reference = extract_native_features(tree, tuple(map(int, queries)), panel,
                                                age_mode="training_int").astype(np.float32)
            cached = np.asarray([bank[i] for i in tree_indices])
            if not np.array_equal(reference, cached):
                raise ValueError("Optimized observed features differ from the unchanged native bridge")
        all_index[lo:hi] = tree_indices
        if visited_trees % 500 == 0:
            print(json.dumps(dict(stage="features", parent_id=parent["parent_id"],
                                  observed_trees=visited_trees, unique_paths=len(bank))), flush=True)
    features = np.asarray(bank, dtype=np.float32)
    grid_index = all_index[np.searchsorted(coordinates, positions)]
    site_index = all_index[np.searchsorted(coordinates, sites)]
    cache = output / (parent["parent_id"] + ".npz")
    with cache.open("xb") as stream:
        np.savez_compressed(stream, features=features, grid_feature_index=grid_index,
                            site_feature_index=site_index, site_positions_bp=sites,
                            positions_bp=positions, bin_edges_bp=bin_edges,
                            retained_fixed100bp_indexes=observation["retained_indexes"],
                            coverage_original_bp0=observation["coverage"],
                            local_to_original_offset_bp=np.asarray(observation["offset"]),
                            query_node_ids=queries, query_sample_indices=parent["query_slots"])
    metadata = dict(parent_id=parent["parent_id"], cache_sha256=digest(cache),
                    cache_path=str(cache.relative_to(ROOT)),
                    observed_tree_sha256=digest(observation["tree_path"]),
                    observed_manifest_entry=observation["entry"],
                    feature_source="observed parent/children/leaves counts and node ages only",
                    feature_age_mode="training_int", ancient_focal_self_inclusion=True,
                    observed_trees_evaluated=visited_trees, unique_first_ancestors=evaluated_starts,
                    unique_native_paths=len(bank), grid_rows=len(positions),
                    original_fixed_grid_total=20000,
                    retained_fixed100bp_indexes=observation["retained_indexes"],
                    coverage_original_bp0=observation["coverage"],
                    normalization_length_bp=2000000,
                    local_to_original_offset_bp=observation["offset"],
                    represented_bin_support_bp=[bin_edges[0], bin_edges[-1]],
                    represented_span_bp=float(bin_edges[-1] - bin_edges[0]),
                    partial_bins_with_unretained_centers_filled=False,
                    observed_site_rows=len(sites), truth_accessed=False,
                    source_bridge_float32_equivalence_first_observed_tree=True)
    write_json(cache.with_suffix(".json"), metadata)
    return cache, metadata


class TruthReader:
    """External labels only, instantiated for TRAIN supervision or sealed VAL scoring."""
    def __init__(self, parent, purpose):
        if purpose not in ("train_supervision", "sealed_val_score"):
            raise ValueError("Unknown truth capability")
        required = "train" if purpose == "train_supervision" else "val"
        if parent["split"] != required:
            raise ValueError("Truth capability does not match the parent split")
        record = parent["row"]["files"]["truth/full_truth_intervals.npz"]
        path = check_file(dict(path=str((parent["directory"] /
                                        "truth/full_truth_intervals.npz").relative_to(ROOT)),
                               sha256=record["sha256"]))
        with np.load(path, allow_pickle=False) as z:
            intervals, raw_nodes = z["intervals_bp"].copy(), z["query_raw_node_ids"].copy()
            paths, c4 = z["native_paths_1based"].copy(), z["source_classes_0based"].copy()
        if (intervals.ndim != 2 or intervals.shape[1] != 2 or intervals[0, 0] != 0
                or intervals[-1, 1] != 2000000
                or np.any(intervals[:, 1] <= intervals[:, 0])
                or not np.array_equal(intervals[:-1, 1], intervals[1:, 0])):
            raise ValueError("External truth must cover the exact physical domain")
        if not np.array_equal(raw_nodes, parent["raw_query_nodes"]):
            raise ValueError("External truth query correspondence differs")
        if (paths.shape != (len(intervals), 120) or paths.dtype.kind not in "iu"
                or np.any((paths < 1) | (paths > 6))
                or not np.array_equal(c4, np.array(PATH_TO_SOURCE_CLASS)[paths - 1])):
            raise ValueError("Invalid six-path/C4 external truth")
        self.intervals, self.paths, self.c4 = intervals, paths, c4
        self.provenance = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), purpose=purpose)

    def labels_at(self, positions):
        indices = np.searchsorted(self.intervals[:, 1], positions, side="right")
        return self.paths[indices]


def sample_native_train(parent, cache, cfg, output):
    """Native site-ID stride, path caps and repeated-pass replacement on TRAIN only."""
    truth = TruthReader(parent, "train_supervision")
    with np.load(cache, allow_pickle=False) as z:
        sites = z["site_positions_bp"]
        index = z["site_feature_index"]
        features = z["features"]
        labels = truth.labels_at(sites)
        cohort = parent["cohorts"]
        selected, reports = [], []
        for name, budget in cfg["sampler"]["per_parent_cohort_budgets"].items():
            columns = np.flatnonzero(cohort == name)
            if not len(columns):
                raise ValueError("Missing declared TRAIN cohort: " + name)
            target, cap = budget["target"], budget["per_path_cap"]
            step = math.floor(len(sites) / ((budget["candidate_scan_budget"] +
                                            10000 / 3) / len(columns)))
            if step < 1:
                raise ValueError("Native source initial site stride is zero for " + name)
            counts, accepted, pass_reports = np.zeros(6, dtype=np.int64), [], []
            for pass_number in range(cfg["sampler"]["maximum_source_passes"]):
                before = len(accepted)
                # Same site.id % step test, ordered sites then ordered query samples.
                for site in range(len(sites)):
                    if site % step != 0:
                        continue
                    for query in columns:
                        label = int(labels[site, query]) - 1
                        if counts[label] >= cap:
                            continue
                        accepted.append((site, int(query)))
                        counts[label] += 1
                        if len(accepted) == target:
                            break
                    if len(accepted) == target:
                        break
                pass_reports.append(dict(pass_number=pass_number, step=step,
                                         added=len(accepted) - before,
                                         realized_native_path_counts=counts.copy()))
                if len(accepted) == target:
                    break
                if len(accepted) == before:
                    break
                step = step + step / 2
            pairs = np.asarray(accepted, dtype=np.int32).reshape(-1, 2)
            report = dict(cohort=name, observed_queries=int(len(columns)), target=target,
                          per_path_cap=cap, realized_native_path_counts=counts,
                          selected_examples=len(accepted),
                          unique_site_query_examples=len(set(accepted)),
                          replacement_duplicates=len(accepted) - len(set(accepted)),
                          passes=pass_reports, quota_filled=len(accepted) == target)
            reports.append(report)
            # Preserve actual failure and partial sampler evidence instead of relaxing caps.
            if len(accepted) != target:
                write_json(output / (parent["parent_id"] + "_sampler_failure.json"),
                           dict(reports=reports, truth_provenance=truth.provenance))
                raise ValueError("Native quota unfilled for " + parent["parent_id"] + "/" + name)
            selected.extend(accepted)
        pairs = np.asarray(selected, dtype=np.int32)
        y = labels[pairs[:, 0], pairs[:, 1]].astype(np.int64) - 1
        x = features[index[pairs[:, 0], pairs[:, 1]]]
        with (output / (parent["parent_id"] + "_training_selection.npz")).open("xb") as stream:
            np.savez_compressed(stream, site_row=pairs[:, 0], query_column=pairs[:, 1],
                                positions_bp=sites[pairs[:, 0]],
                                query_sample_indices=parent["query_slots"][pairs[:, 1]],
                                native_labels_0based=y)
    write_json(output / (parent["parent_id"] + "_sampler.json"),
               dict(parent_id=parent["parent_id"], reports=reports,
                    truth_provenance=truth.provenance, train_only_truth_selection=True))
    return x, y


def intersections(edges, intervals):
    bins, rows, spans = [], [], []
    i = j = 0
    while i < len(edges) - 1 and j < len(intervals):
        left = max(edges[i], intervals[j, 0])
        right = min(edges[i + 1], intervals[j, 1])
        if right > left:
            bins.append(i); rows.append(j); spans.append(right - left)
        if edges[i + 1] <= intervals[j, 1]:
            i += 1
        else:
            j += 1
    spans = np.asarray(spans, dtype=np.float64)
    if not np.isclose(spans.sum(), edges[-1] - edges[0], atol=1e-8, rtol=0):
        raise ValueError("Full truth/prediction intersections do not cover the domain")
    return np.asarray(bins), np.asarray(rows), spans


def boundary_match(actual, predicted, tolerance):
    i = j = 0
    matches = []
    while i < len(actual) and j < len(predicted):
        if abs(actual[i] - predicted[j]) <= tolerance:
            matches.append(dict(true_bp=actual[i], predicted_bp=predicted[j],
                                absolute_error_bp=abs(actual[i] - predicted[j])))
            i += 1; j += 1
        elif predicted[j] < actual[i] - tolerance:
            j += 1
        else:
            i += 1
    n = len(matches)
    return dict(true_boundaries=len(actual), predicted_boundaries=len(predicted), matched=n,
                missed=len(actual) - n, extra=len(predicted) - n, matches=matches,
                precision=n / len(predicted) if len(predicted) else None,
                recall=n / len(actual) if len(actual) else None,
                f1=2 * n / (len(actual) + len(predicted)) if len(actual) else None)


def tracts(bounds, labels):
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    stops = np.r_[starts[1:], len(labels)]
    return [dict(left_bp=float(bounds[a]), right_bp=float(bounds[b]),
                 class_0based=int(labels[a])) for a, b in zip(starts, stops)]


def length_summary(records):
    x = np.array([r["right_bp"] - r["left_bp"] for r in records])
    return dict(count=len(x), mean_bp=float(x.mean()), median_bp=float(np.median(x)),
                minimum_bp=float(x.min()), maximum_bp=float(x.max()))


def full_span_score(p4, parent, truth, columns, metrics_cfg):
    """Exact integration of constant prediction bins against every true interval."""
    p = p4[:, columns].astype(np.float64)
    labels = truth.c4[:, columns]
    bins, rows, spans = intersections(parent["edges"], truth.intervals)
    confusion = np.zeros((4, 4), dtype=float)  # truth rows, predicted columns, query-bp
    reliability = np.zeros((metrics_cfg["calibration_bins"], 3), dtype=float)
    nll_sum = brier_sum = infinite_nll_bp = 0.0
    dosage_squared = dosage_absolute = 0.0
    if len(columns) % 2 or any(
            parent["samples"][parent["query_slots"][columns[i]]]["sample_id"] !=
            parent["samples"][parent["query_slots"][columns[i + 1]]]["sample_id"]
            for i in range(0, len(columns), 2)):
        raise ValueError("Dosage scoring requires complete ordered query diploids")
    for start in range(0, len(bins), 1024):
        stop = start + 1024
        probs = p[bins[start:stop]]
        y = labels[rows[start:stop]]
        weights = spans[start:stop, None]
        predicted = probs.argmax(axis=-1)
        selected = np.take_along_axis(probs, y[..., None], axis=-1)[..., 0]
        positive = selected > 0
        infinite_nll_bp += float(np.sum(weights * ~positive))
        nll_sum += float(np.sum(weights * -np.log(np.where(positive, selected, 1))))
        brier = np.sum(probs * probs, axis=-1) + 1 - 2 * selected
        brier_sum += float(np.sum(weights * brier))
        np.add.at(confusion, (y.ravel(), predicted.ravel()),
                  np.broadcast_to(weights, y.shape).ravel())
        confidence = probs.max(axis=-1)
        slots = np.minimum((confidence * len(reliability)).astype(int), len(reliability) - 1)
        for slot in range(len(reliability)):
            mask = slots == slot
            reliability[slot, 0] += np.sum(weights * mask)
            reliability[slot, 1] += np.sum(weights * confidence * mask)
            reliability[slot, 2] += np.sum(weights * (predicted == y) * mask)
        target = np.eye(4)[y[:, 0::2]] + np.eye(4)[y[:, 1::2]]
        dose = probs[:, 0::2] + probs[:, 1::2]
        dosage_squared += float(np.sum(weights[..., None] * (dose - target) ** 2))
        dosage_absolute += float(np.sum(weights[..., None] * np.abs(dose - target)))
    total = float(spans.sum() * len(columns))
    actual_span, predicted_span = confusion.sum(1), confusion.sum(0)
    diagonal = np.diag(confusion)
    per_class = []
    for c in range(4):
        precision = diagonal[c] / predicted_span[c] if predicted_span[c] else None
        recall = diagonal[c] / actual_span[c] if actual_span[c] else None
        f1 = 2 * diagonal[c] / (actual_span[c] + predicted_span[c]) if actual_span[c] else None
        union = actual_span[c] + predicted_span[c] - diagonal[c]
        per_class.append(dict(class_0based=c, true_query_bp=actual_span[c],
                              predicted_query_bp=predicted_span[c], precision=precision,
                              recall=recall, f1=f1, span_iou=diagonal[c] / union if union else None))
    query_reports = []
    for local, query in enumerate(columns):
        complete_true = tracts(np.r_[truth.intervals[:, 0], truth.intervals[-1, 1]],
                               truth.c4[:, query])
        support_left, support_right = parent["edges"][0], parent["edges"][-1]
        true = []
        for record in complete_true:
            left, right = max(record["left_bp"], support_left), min(record["right_bp"], support_right)
            if right > left:
                true.append(dict(record, left_bp=float(left), right_bp=float(right),
                                 original_left_bp=record["left_bp"],
                                 original_right_bp=record["right_bp"],
                                 original_span_bp=record["right_bp"] - record["left_bp"],
                                 support_left_censored=left == support_left,
                                 support_right_censored=right == support_right))
        predicted = tracts(parent["edges"], p[:, local].argmax(axis=-1))
        match = boundary_match([r["right_bp"] for r in true[:-1]],
                               [r["right_bp"] for r in predicted[:-1]],
                               metrics_cfg["boundary_tolerance_bp"])
        short = []
        for tract in true:
            width = tract["right_bp"] - tract["left_bp"]
            if tract["original_span_bp"] > metrics_cfg["short_tract_cutoff_bp"]:
                continue
            selected_bins = np.flatnonzero(
                (parent["edges"][1:] > tract["left_bp"]) &
                (parent["edges"][:-1] < tract["right_bp"]))
            w = (np.minimum(parent["edges"][selected_bins + 1], tract["right_bp"]) -
                 np.maximum(parent["edges"][selected_bins], tract["left_bp"]))
            good = p[selected_bins, local].argmax(-1) == tract["class_0based"]
            short.append(dict(**tract, correct_overlap_fraction=float(np.sum(w * good) / width),
                              support_endpoint_censored=tract["support_left_censored"] or
                              tract["support_right_censored"],
                              original_domain_endpoint=tract["original_left_bp"] == 0 or
                              tract["original_right_bp"] == 2000000))
        query_reports.append(dict(query_sample_index=int(parent["query_slots"][query]),
                                  observed_cohort=str(parent["cohorts"][query]),
                                  true_tract_lengths=length_summary(true),
                                  predicted_tract_lengths=length_summary(predicted),
                                  boundaries=match, short_true_tracts=short,
                                  true_tracts=true, predicted_tracts=predicted))
    boundary_totals = {k: sum(r["boundaries"][k] for r in query_reports)
                       for k in ("true_boundaries", "predicted_boundaries", "matched", "missed", "extra")}
    nt, npred, matched = (boundary_totals[k] for k in
                          ("true_boundaries", "predicted_boundaries", "matched"))
    boundary_totals.update(precision=matched / npred if npred else None,
                           recall=matched / nt if nt else None,
                           f1=2 * matched / (nt + npred) if nt else None)
    represented = float(parent["edges"][-1] - parent["edges"][0])
    coverage = parent["coverage_original"]
    return dict(query_count=len(columns), query_bp=total, denominator_query_bp=total,
                original_parent_interval_bp0=[0, 2000000], normalization_length_bp=2000000,
                original_parent_query_bp=2000000 * len(columns),
                coverage_original_bp0=coverage,
                physical_coverage_fraction=float((coverage[1] - coverage[0]) / 2000000),
                represented_bin_support_bp=[parent["edges"][0], parent["edges"][-1]],
                represented_span_bp=represented,
                represented_fraction_original_parent=represented / 2000000,
                covered_span_without_retained_center_bp=float(coverage[1] - coverage[0] - represented),
                retained_original_grid_indexes=parent["retained_indexes"],
                score_scope="conditional on exact retained-center clipped-bin support; uncovered tails excluded",
                prediction_definition="constant raw C4 probabilities over frozen physical 100-bp bins",
                truth_definition="exact intersections with complete external truth intervals",
                accuracy=float(diagonal.sum() / total),
                nll=None if infinite_nll_bp else nll_sum / total,
                infinite_nll_query_bp=infinite_nll_bp, probability_floor_used=False,
                brier=brier_sum / total,
                macro_f1=float(np.mean([r["f1"] for r in per_class if r["f1"] is not None])),
                balanced_accuracy=float(np.mean([r["recall"] for r in per_class
                                                  if r["recall"] is not None])),
                confusion_truth_rows_prediction_columns_query_bp=confusion,
                per_class=per_class,
                calibration_ece=float(np.abs(reliability[:, 2] - reliability[:, 1]).sum() / total),
                calibration_reliability_bins=[
                    dict(bin_index=i, query_bp=r[0], mean_confidence=r[1] / r[0] if r[0] else None,
                         accuracy=r[2] / r[0] if r[0] else None)
                    for i, r in enumerate(reliability)],
                dosage_mse_per_diploid_bp_class=dosage_squared / (total / 2 * 4),
                dosage_mae_per_diploid_bp_class=dosage_absolute / (total / 2 * 4),
                boundaries=boundary_totals, per_query=query_reports)


def predict_and_score(model, parent, cache, seed_dir, cfg):
    with np.load(cache, allow_pickle=False) as z:
        features, index, queries = z["features"], z["grid_feature_index"], z["query_node_ids"]
        parent = dict(parent, positions=z["positions_bp"].copy(), edges=z["bin_edges_bp"].copy(),
                      retained_indexes=z["retained_fixed100bp_indexes"].copy(),
                      coverage_original=z["coverage_original_bp0"].copy(),
                      local_offset=float(z["local_to_original_offset_bp"]))
        bank_probs = model.predict(features, batch_size=cfg["training"]["prediction_batch_size"], verbose=0)
        if (bank_probs.shape != (len(features), 6) or not np.isfinite(bank_probs).all()
                or np.any(bank_probs < 0) or
                not np.allclose(bank_probs.sum(-1), 1, atol=1e-6, rtol=1e-5)):
            raise ValueError("Invalid native six-class softmax output")
        p6 = bank_probs[index]
    p4 = collapse_native_probabilities(p6)
    prediction_path = seed_dir / (parent["parent_id"] + "_raw_probabilities.npz")
    # Seal observed predictions before instantiating the VAL truth capability.
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, native_probabilities=p6, C4_probabilities=p4,
                            native_path_1based=p6.argmax(-1) + 1,
                            C4_class_0based=p4.argmax(-1), positions_bp=parent["positions"],
                            bin_edges_bp=parent["edges"], query_node_ids=queries,
                            retained_fixed100bp_indexes=parent["retained_indexes"],
                            coverage_original_bp0=parent["coverage_original"],
                            local_to_original_offset_bp=np.asarray(parent["local_offset"]),
                            normalization_length_bp=np.asarray(2000000),
                            query_sample_indices=parent["query_slots"])
    prediction_hash = digest(prediction_path)
    truth = TruthReader(parent, "sealed_val_score")
    labels = truth.labels_at(parent["positions"])
    cohorts = parent["cohorts"]
    groups = {"primary_present_day_64": np.flatnonzero(cohorts == "present_day"),
              "secondary_ancient_56": np.flatnonzero(cohorts != "present_day"),
              "all_queries_120": np.arange(120)}
    native = {name: native_accuracy_confusion(p6[:, columns], labels[:, columns])
              for name, columns in groups.items()}
    native["by_observed_cohort"] = {
        name: native_accuracy_confusion(p6[:, cohorts == name], labels[:, cohorts == name])
        for name in sorted(set(cohorts.tolist()))}
    c4 = {name: full_span_score(p4, parent, truth, columns, cfg["metrics"])
          for name, columns in groups.items()}
    result = dict(parent_id=parent["parent_id"], split="val",
                  prediction_path=str(prediction_path.relative_to(ROOT)),
                  prediction_sha256=prediction_hash, truth_provenance=truth.provenance,
                  native_unweighted_midpoint_metrics=native, C4_exact_full_span_metrics=c4,
                  C4_collapse="p0+p4,p1+p5,p2,p3; sum probabilities before argmax",
                  calibration="descriptive raw-probability calibration; no VAL calibration fit",
                  cohort_role="present64 primary; ancient56 secondary, native observed self-inclusion retained",
                  truth_used_for_eligibility_or_site_filtering=False,
                  claim="source-native Dense bridge biological pilot, not published CNN reproduction")
    write_json(seed_dir / (parent["parent_id"] + "_metrics.json"), result)
    return result


def run(cfg, parents, observed_spec, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This runner requires explicitly empty CUDA visibility")
    if Path(sys.prefix).resolve() != local_path(cfg["tensorflow_cpu_prefix"]):
        raise RuntimeError("Use the already-qualified isolated TensorFlow CPU prefix")
    import tensorflow as tf
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Unexpected GPU in the isolated CPU execution")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    if importlib.metadata.version("tensorflow-cpu") != "2.15.1":
        raise RuntimeError("Qualified TensorFlow version differs")
    output.mkdir(exist_ok=False)
    cache_dir = output / "feature_cache"; cache_dir.mkdir()
    selection_dir = output / "training_sampler"; selection_dir.mkdir()
    receipt = dict(hypothesis_id="H020", kind="ancestralpaths_biological_pilot_v1",
                   started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   config_path=cfg["_config_path"],
                   config_sha256=digest(local_path(cfg["_config_path"])),
                   implementation_sha256=digest(Path(__file__)),
                   observed_manifest=cfg["_observed_manifest"],
                   observed_manifest_sha256=digest(local_path(cfg["_observed_manifest"])),
                   source_native_dense=True, published_CNN_reproduced=False,
                   GPU_used=False, status="running", feature_caches={}, failures=[], seeds={},
                   runtime=dict(executable=sys.executable, packages={
                       p: importlib.metadata.version(p) for p in
                       ("numpy", "tensorflow-cpu", "keras", "tskit", "msprime", "demes", "pandas")}))
    started = time.monotonic()
    def failure(stage, exc, **identity):
        item = dict(stage=stage, **identity, error=traceback.format_exc(),
                    exception_type=type(exc).__name__)
        receipt["failures"].append(item)
        with (output / "failures.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        write_json(output / "receipt.json", receipt)
    caches = {}
    for parent in parents:
        try:
            cache, metadata = cache_features(parent, observed_spec[parent["parent_id"]], cache_dir)
            caches[parent["parent_id"]] = cache
            receipt["feature_caches"][parent["parent_id"]] = metadata
        except Exception as exc:
            failure("observed_feature_cache", exc, parent_id=parent["parent_id"])
    if receipt["failures"]:
        receipt.update(status="failed", elapsed_seconds=time.monotonic() - started)
        write_json(output / "receipt.json", receipt)
        return 1
    train_x, train_y = [], []
    for parent in parents:
        if parent["split"] == "train":
            try:
                x, y = sample_native_train(parent, caches[parent["parent_id"]], cfg, selection_dir)
                train_x.append(x); train_y.append(y)
            except Exception as exc:
                failure("native_TRAIN_sampler", exc, parent_id=parent["parent_id"])
    if not receipt["failures"]:
        x, y = np.concatenate(train_x), np.concatenate(train_y)
        targets = tf.keras.utils.to_categorical(y, num_classes=6)
        receipt["training_examples"] = len(y)
        receipt["realized_native_path_counts"] = np.bincount(y, minlength=6).tolist()
        for seed in cfg["training"]["seeds"]:
            seed_dir = output / ("seed_" + str(seed)); seed_dir.mkdir()
            state = dict(status="running", parents={})
            receipt["seeds"][str(seed)] = state
            try:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = build_native_keras_model(tf.keras)
                if model.count_params() != 26630 or model.output_shape != (None, 6):
                    raise ValueError("Unchanged native Dense model contract differs")
                before = [w.numpy().copy() for w in model.trainable_weights]
                history = model.fit(x, targets, epochs=5, batch_size=32, shuffle=True, verbose=2)
                delta = max(float(np.max(np.abs(a - b.numpy())))
                            for a, b in zip(before, model.trainable_weights))
                if not delta > 0 or not all(np.isfinite(v).all() for v in history.history.values()):
                    raise ValueError("Actual optimizer fit did not produce finite changed weights")
                checkpoint = seed_dir / "native_dense_epoch5.keras"
                model.save(checkpoint)
                state.update(optimizer_fit_completed=True, fit_history=history.history,
                             optimizer=model.optimizer.get_config(),
                             checkpoint=str(checkpoint.relative_to(ROOT)),
                             checkpoint_sha256=digest(checkpoint), maximum_weight_delta=delta,
                             epochs=5, batch_size=32, no_VAL_early_stopping=True)
                write_json(seed_dir / "fit.json", dict(state, status="fit_completed"))
                for parent in parents:
                    if parent["split"] == "val":
                        try:
                            result = predict_and_score(model, parent, caches[parent["parent_id"]],
                                                       seed_dir, cfg)
                            state["parents"][parent["parent_id"]] = dict(
                                status="passed", prediction_sha256=result["prediction_sha256"],
                                metrics=str((seed_dir / (parent["parent_id"] +
                                                        "_metrics.json")).relative_to(ROOT)))
                        except Exception as exc:
                            state["parents"][parent["parent_id"]] = dict(status="failed")
                            failure("sealed_VAL_prediction_score", exc, seed=seed,
                                    parent_id=parent["parent_id"])
                state["status"] = "passed" if all(v["status"] == "passed" for v in
                                                 state["parents"].values()) else "failed"
            except Exception as exc:
                state["status"] = "failed"
                failure("actual_native_fit", exc, seed=seed)
            write_json(output / "receipt.json", receipt)
    receipt.update(status="passed" if not receipt["failures"] else "failed",
                   elapsed_seconds=time.monotonic() - started,
                   TEST_parents=0, biological_fit_ran=any(
                       s.get("optimizer_fit_completed", False) for s in receipt["seeds"].values()),
                   parent_independence="two distinct TRAIN, two distinct VAL; no final holdout inference")
    write_json(output / "receipt.json", receipt)
    return 0 if receipt["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--observed-manifest",
                        default="data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json")
    parser.add_argument("--output", default="results/ancestralpaths_biological_pilot_v1")
    parser.add_argument("--check-only", action="store_true",
                        help="Small frozen-source/observed-schema check; no truth, TensorFlow fit, cache or output")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from the dedicated anogena repository root")
    cfg, parents, source = protocol(local_path(args.config))
    manifest = local_path(args.observed_manifest)
    if args.check_only:
        observed = observations(manifest, parents, cfg) if manifest.exists() else None
        print(json.dumps(dict(status="passed", source_commit=source["commit"],
                              parents={p["parent_id"]: p["split"] for p in parents},
                              samples_per_parent=280, queries_per_parent=120,
                              primary_present_queries=64, observed_manifest_exists=manifest.exists(),
                              completed_Relate_manifest_admitted=observed is not None,
                              truth_opened=False, TensorFlow_imported=False, training_run=False)))
        return 0
    spec = observations(manifest, parents, cfg)
    cfg["_config_path"] = str(local_path(args.config).relative_to(ROOT))
    cfg["_observed_manifest"] = str(manifest.relative_to(ROOT))
    return run(cfg, parents, spec, local_path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())

"""Observed-only full-panel AP/ARGMix preparation adapters; no work at import.

Root owns execution/storage/runtime admission. Mutating APIs require a root guard
callback and explicitly empty CUDA visibility. This module never fits, selects,
scores, opens census trees, or authorizes outer truth. Old pilot guards are frozen.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np
from arggnn import full_panel_native_baseline_contracts_v1 as contracts

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SCHEMA = "prepare_full_panel_native_supervised_v1"
CONFIG = "configs/full_panel_native_baselines_v2.json"
SPLIT = "research/reviews/full_panel_native_comparison_split_v1/split.json"
DATA = "data/ancestralpaths_full_panel_discovery_v1"
AP_CACHE_HELPER = {
    "path": "scripts/train_ancestralpaths_biological_v1.py",
    "sha256": "826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056",
}
MODULE_CACHE: dict[str, Any] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise contracts.ContractError(message)


def local(value: str | Path, *, observed: bool = False) -> Path:
    path = Path(value)
    path = ROOT / path if not path.is_absolute() else path
    require(not path.is_symlink(), "Symlink artifact refused")
    path = path.resolve()
    require(path != ROOT and path.is_relative_to(ROOT), "Artifact must be inside repository")
    if observed:
        require(not any(("truth" in part.lower() or "oracle" in part.lower()
                         or "census" in part.lower()) for part in path.relative_to(ROOT).parts),
                "Observed reader refuses truth/oracle/census paths")
    return path


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(record: Mapping[str, Any], *, observed: bool = False) -> Path:
    require(isinstance(record, Mapping) and set(record) >= {"path", "sha256"},
            "Explicit artifact path/hash required")
    contracts._sha(record["sha256"])
    path = local(record["path"], observed=observed)
    require(path.is_file() and sha(path) == record["sha256"], "Missing/changed bound artifact")
    return path


def record(path: str | Path) -> dict[str, str]:
    path = local(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def _guard(guard: Callable[[int], Any], needed: int) -> None:
    require(callable(guard), "Root-owned write admission callback required")
    require(Path.cwd().resolve() == ROOT, "Run mutations from the dedicated repository")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Preparation must be explicitly CPU-only")
    guard(int(needed))


def new_directory(output: str | Path, guard: Callable[[int], Any]) -> Path:
    path = local(output)
    _guard(guard, 16384)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: Any, guard: Callable[[int], Any]) -> dict[str, str]:
    payload = (json.dumps(jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _guard(guard, len(payload) + 4096)
    with path.open("xb") as stream:
        stream.write(payload)
    return record(path)


def write_text(path: Path, text: str, guard: Callable[[int], Any]) -> dict[str, str]:
    payload = text.encode()
    _guard(guard, len(payload) + 4096)
    with path.open("xb") as stream:
        stream.write(payload)
    return record(path)


def write_npz(path: Path, arrays: Mapping[str, Any], guard: Callable[[int], Any]) -> dict[str, str]:
    size = sum(np.asarray(v).nbytes for v in arrays.values())
    _guard(guard, int(size * 1.02) + 65536)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return record(path)


def protocol(config_path: str | Path = CONFIG, split_path: str | Path = SPLIT) -> dict[str, Any]:
    cfg_path, split_path = local(config_path, observed=True), local(split_path, observed=True)
    cfg = contracts.validate_policy_bytes(cfg_path.read_bytes(), split_path.read_bytes())
    split = contracts.validate_split_bytes(split_path.read_bytes())
    for ref in [cfg["generation_config"], cfg["generation_source"],
                *cfg["helper_algorithm_refs"], *cfg["native_author_source_refs"], AP_CACHE_HELPER]:
        checked(ref, observed=True)
    return {"policy": cfg, "split": split, "config_record": record(cfg_path),
            "split_record": record(split_path)}


def pinned_module(ref: Mapping[str, Any], name: str) -> Any:
    path = checked(ref, observed=True)
    key = str(path) + ":" + ref["sha256"]
    if key not in MODULE_CACHE:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        MODULE_CACHE[key] = module
    return MODULE_CACHE[key]


def helper_ref(observation: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    return next(r for r in observation["protocol"]["policy"]["helper_algorithm_refs"]
                if r["path"] == path)


def read_completed_parent(parent_id: str, inference_record: Mapping[str, Any],
                          completion_record: Mapping[str, Any], *,
                          config_path: str | Path = CONFIG, split_path: str | Path = SPLIT,
                          require_argmix_native_ids: bool = False) -> dict[str, Any]:
    """Strict completed post-size observation reader. No target bytes are opened.

    Supplied completion/inference bindings must come from root admission. Inner
    observation bindings are verified independently, including the original nine
    whitelisted files, template, SNP conversion join, actual flagged IDs and ages.
    Generic actual sample IDs are retained for AP/ARGformer. ARGMix requests its
    explicit stock contiguous-ID boundary. This does not grant execution admission.
    """
    import tskit
    proto = protocol(config_path, split_path)
    require(parent_id in contracts.PARENT_ROLES, "Unknown frozen parent")
    directory = local(DATA + "/" + parent_id, observed=True)
    inference_path = checked(inference_record, observed=True)
    completion_path = checked(completion_record, observed=True)
    require(inference_path == directory / "relate/inference.json", "Parent inference binding differs")
    require(completion_path in {directory / "all_completion.json", directory / "relate_completion.json"},
            "Completed generation/Relate receipt required")
    completion = json.loads(completion_path.read_text())
    expected = next(p for p in proto["split"]["parents"] if p["parent_id"] == parent_id)
    nominal = {k: expected[k] for k in (
        "parent_id", "split", "replicate", "seeds", "seed_domain_sha256",
        "relate_size_iteration_seeds", "relate_final_reestimate_seed")}
    require(completion.get("status") == "complete" and completion.get("stage") in {"all", "relate"}
            and completion.get("parent_id") == parent_id and completion.get("split") == nominal["split"]
            and completion.get("models_fitted") is False
            and completion.get("config_sha256") == proto["policy"]["generation_config"]["sha256"]
            and completion.get("source_sha256") == proto["policy"]["generation_source"]["sha256"],
            "Completion/provenance differs")
    freeze_path = directory / "freeze.json"
    require(freeze_path.is_file() and not freeze_path.is_symlink(), "Original generation freeze missing")
    frozen = json.loads(freeze_path.read_text())
    require(frozen.get("selected_parent") == nominal
            and frozen.get("config_sha256") == completion["config_sha256"]
            and frozen.get("source_sha256") == completion["source_sha256"], "Frozen original stream differs")
    inference = json.loads(inference_path.read_text())
    require(inference.get("parent_id") == parent_id and inference.get("split") == nominal["split"]
            and inference.get("status") == "post_five_iterations_inferred"
            and inference.get("truth_or_oracle_read") is False
            and inference.get("common_C4_and_native_six_path_consumers_use_same_post_size_tree") is True
            and inference.get("outer_command_returncodes") == [0] * 5,
            "Completed primary inference differs")
    post = inference["post_five_iterations"]
    tree_path = checked(inference["inferred_tree"], observed=True)
    roster_path = checked(inference["observed_sample_manifest"], observed=True)
    require(tree_path == directory / "relate/post_size.observed.trees"
            and roster_path == directory / "relate/observed_sample_manifest.json"
            and local(post["clean_tree"], observed=True) == tree_path
            and post["clean_tree_sha256"] == inference["inferred_tree"]["sha256"]
            and post["observed_sample_manifest"] == inference["observed_sample_manifest"]
            and post["sample_age_order_preserved"] is True
            and post["node_ids_preserved_during_clipping"] is True, "Primary observed bindings differ")
    rows = json.loads(roster_path.read_text())
    roster = contracts.validate_observed_roster(rows, require_argmix_native_ids=require_argmix_native_ids)
    require(post["sample_index_to_inferred_tree_node_id"] == [
        {"sample_index": r["sample_index"], "inferred_tree_node_id": r["inferred_tree_node_id"]} for r in rows],
        "Actual sample flag mapping differs")
    queries = [r for r in rows if r["role"] == "query"]
    require(post["query_keys"] == [[r["sample_id"], r["haplotype"]] for r in queries],
            "Complete ordered query identity differs")
    support = contracts.validate_physical_support(post)
    site_join = contracts.validate_site_coordinate_join(
        post["site_join"], coverage_original_bp0=support["coverage_original_bp0"],
        declared_site_count=post["sites"])
    producer = pinned_module(proto["policy"]["generation_source"], "full_native_frozen_producer")
    generation_cfg = json.loads(checked(proto["policy"]["generation_config"], observed=True).read_text())
    require(set(inference["input_observed_sha256"]) == set(producer.OBSERVED_FILES),
            "Exact nine observed whitelist files required")
    input_records = {}
    for name, digest in inference["input_observed_sha256"].items():
        ref = {"path": str((directory / "observed" / name).relative_to(ROOT)), "sha256": digest}
        checked(ref, observed=True)
        input_records[name] = ref
    # Reuse the producer's exact template/date/cohort/map and full-grid validation.
    assets = producer.observed_assets(generation_cfg, nominal, directory)
    require(assets["source_sha256"] == inference["input_observed_sha256"], "Original input bindings differ")
    for inferred, original in zip(rows, assets["rows"]):
        require(all(inferred[k] == original[k] for k in (
            "sample_index", "sample_id", "haplotype", "observed_cohort",
            "age_generations", "role", "observed_reference_class"))
            and inferred["source_observed_node_id"] == original["observed_node_id"],
            "Inferred/source observed roster identity differs")
    original_sites = json.loads(checked(input_records["site_coordinate_manifest.json"], observed=True).read_text())["rows"]
    for join in post["site_join"]:
        original = original_sites[join["original_site_id"]]
        require(original["site_id"] == join["original_site_id"]
                and original["tree_position_0based"] == join["original_position_0based"]
                and original["VCF_position_1based"] == join["VCF_position_1based"],
                "Native SNP join does not use exact original conversion map")
    ts = tskit.load(tree_path)
    tables = ts.tables
    require(bool(np.isin(tables.nodes.flags, [0, 1]).all())
            and not tables.metadata and not ts.num_populations and not ts.num_individuals
            and not ts.num_migrations and bool(np.all(tables.nodes.population == -1))
            and bool(np.all(tables.nodes.individual == -1))
            and all(len(t.metadata) == 0 for t in (
                tables.nodes, tables.edges, tables.sites, tables.mutations,
                tables.individuals, tables.populations, tables.migrations)),
            "Observed tree must exclude census/ancestry/population/individual annotations")
    require(set(map(int, ts.samples())) == set(roster.all_sample_nodes)
            and ts.sequence_length == post["sequence_length_local_bp"]
            and ts.num_sites == post["sites"] and ts.num_trees == post["trees"],
            "Actual inferred samples/coverage/site count differs")
    require(all(abs(ts.node(r["inferred_tree_node_id"]).time - r["age_generations"]) <= 1e-6 for r in rows),
            "Actual flagged sample age differs")
    local_sites = np.asarray(ts.tables.sites.position, dtype=np.float64)
    require(np.array_equal(local_sites, [r["local_position_0based"] for r in post["site_join"]]),
            "Actual native sites differ from complete exact join")
    for cohort in contracts.SCAN_BUDGET:
        contracts.native_ap_stride(ts.num_sites, cohort, traversal="TRAIN" if
                                   contracts.PARENT_ROLES[parent_id] == "TRAIN" else "HOLDOUT")
    return {
        "parent_id": parent_id, "downstream_role": contracts.PARENT_ROLES[parent_id],
        "nominal_split": nominal["split"], "ts": ts, "rows": rows, "roster": roster,
        "tree_record": inference["inferred_tree"], "inference_record": dict(inference_record),
        "completion_record": dict(completion_record), "freeze_record": record(freeze_path),
        "observed_input_records": input_records, "support": support, "site_join": post["site_join"],
        "local_site_positions_bp": local_sites.astype(np.int64),
        "original_site_positions_bp": np.asarray([j[2] for j in site_join], dtype=np.int64),
        "protocol": proto, "truth_opened": False,
    }


def observed_summary(observation: Mapping[str, Any]) -> dict[str, Any]:
    roster = observation["roster"]
    return {"schema": SCHEMA, "parent_id": observation["parent_id"],
            "downstream_role": observation["downstream_role"], "nominal_split": observation["nominal_split"],
            "tree": observation["tree_record"], "inference": observation["inference_record"],
            "completion": observation["completion_record"], "freeze": observation["freeze_record"],
            "observed_inputs": observation["observed_input_records"],
            "config": observation["protocol"]["config_record"], "split_manifest": observation["protocol"]["split_record"],
            "samples": 952, "diploids": 476, "queries": 686, "pure_refs": 266,
            "older_reference_counts": dict(roster.older_reference_counts),
            "stock_native_ID_correspondence_verified": roster.stock_argmix_id_correspondence,
            "truth_opened": False, "execution_admission_created": False,
            "original_paper_reproduction": False}


def argmix_reference_text(rows: list[dict[str, Any]]) -> str:
    contracts.validate_observed_roster(rows, require_argmix_native_ids=True)
    return "".join(f"{r['inferred_tree_node_id']} "
                   f"{contracts.CLASS_ORDER[r['observed_reference_class']] if r['role'] == 'pure_source_reference' else 'QUERY'} "
                   f"{r['age_generations']}\n" for r in rows)


def native_sample_text(rows: list[dict[str, Any]]) -> str:
    contracts.validate_observed_roster(rows, require_argmix_native_ids=True)
    return "ID_1 ID_2 missing\n0 0 0\n" + "".join(
        f"{r['sample_id']} {r['sample_id']} 0\n" for r in rows[::2])


def prepare_observed_frame(observation: Mapping[str, Any], output_dir: str | Path, *,
                           guard: Callable[[int], Any]) -> dict[str, Any]:
    """Stock ARGMix metadata and common coordinates; all 266 donors retained."""
    contracts.validate_observed_roster(observation["rows"], require_argmix_native_ids=True)
    output = new_directory(output_dir, guard)
    rows, support = observation["rows"], observation["support"]
    result = observed_summary(observation)
    result.update(kind="full_panel_native_observed_frame_v1",
        references=write_text(output / "references.tsv", argmix_reference_text(rows), guard),
        native_sample=write_text(output / "native_sample.sample", native_sample_text(rows), guard),
        local_map=write_text(output / "local_uniform_map.tsv", "position Genetic_Map(cM)\n0 0\n2000000 2\n", guard),
        roster=write_json(output / "observed_roster.json", rows, guard),
        coordinates=write_npz(output / "observed_coordinates.npz", {
            "grid_indexes": np.asarray(support["grid_indexes"], dtype=np.int64),
            "positions_bp": np.asarray(support["original_centers_bp"], dtype=np.float64),
            "local_positions_bp": np.asarray(support["local_centers_bp"], dtype=np.float64),
            "bin_edges_bp": np.asarray(support["clipped_bin_edges_bp"], dtype=np.float64),
            "local_site_positions_bp": observation["local_site_positions_bp"],
            "original_site_positions_bp": observation["original_site_positions_bp"],
            "query_nodes": np.asarray(observation["roster"].query_nodes, dtype=np.int64),
            "query_sample_indices": np.asarray([r["sample_index"] for r in rows if r["role"] == "query"], dtype=np.int64),
            "query_raw_node_ids": np.asarray([r["source_observed_node_id"] for r in rows if r["role"] == "query"], dtype=np.int64),
        }, guard),
        class_order=list(contracts.CLASS_ORDER), source_aliases={"ANA": "Ana"},
        local_offset_bp=support["coverage_original_bp0"][0],
        coverage_original_bp0=list(support["coverage_original_bp0"]),
        normalization_length_bp=2000000,
        older_references_are_source_deme_prototypes=True,
        census350_reference_descendants_fabricated=False)
    result["manifest"] = write_json(output / "frame.json", result, guard)
    return result


def _full_roster_cache_function() -> tuple[Any, Any]:
    """Reuse the pinned v1 cache algorithm with one cardinality literal adapter.

    The function's only 280 literal is ObservedPanel's diploid-order range. AST
    replacement keeps all other bytecode source logic, shell arithmetic, padding,
    identity deduplication and observed site/grid traversal unchanged.
    """
    module = pinned_module(AP_CACHE_HELPER, "full_native_frozen_ap_cache")
    tree = ast.parse(inspect.getsource(module.cache_features))
    literals = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == 280]
    require(len(literals) == 1, "Pinned cache cardinality adapter target changed")
    literals[0].value = 952
    namespace = dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(tree), str(checked(AP_CACHE_HELPER)), "exec"), namespace)
    return namespace["cache_features"], module


def prepare_ap_features(observation: Mapping[str, Any], output_dir: str | Path, *,
                        guard: Callable[[int], Any]) -> dict[str, Any]:
    """Native observed-only float32 bank with every query × site/grid index."""
    output = new_directory(output_dir, guard)
    ts, rows, support = observation["ts"], observation["rows"], observation["support"]
    queries = [r for r in rows if r["role"] == "query"]
    require(len(queries) == 686, "Full AP query panel required")
    require(not bool(np.isin(ts.tables.edges.parent, observation["roster"].all_sample_nodes).any()),
            "Pinned optimized AP cache requires observed sample nodes to be terminal; no silent feature substitution")
    # Conservative uncompressed bound: every physical tree/query has a unique
    # 5x9 float32 signature, plus all site/grid int32 indexes and metadata.
    bound = ts.num_trees * 686 * 45 * 4 + (len(support["grid_indexes"]) + ts.num_sites) * 686 * 4
    _guard(guard, int(bound * 1.02) + 16 * 1024 * 1024)
    cache_function, _ = _full_roster_cache_function()
    parent = {"parent_id": observation["parent_id"], "split": "train" if
              observation["downstream_role"] == "TRAIN" else "val", "samples": rows,
              "query_slots": np.asarray([r["sample_index"] for r in queries], dtype=np.int32)}
    original = {"tree_path": checked(observation["tree_record"], observed=True),
        "nodes": np.asarray(observation["roster"].all_sample_nodes, dtype=np.int64),
        "coverage": np.asarray(support["coverage_original_bp0"]), "offset": support["coverage_original_bp0"][0],
        "positions": np.asarray(support["original_centers_bp"]),
        "bin_edges": np.asarray(support["clipped_bin_edges_bp"]),
        "retained_indexes": list(support["grid_indexes"]),
        "site_join": observation["site_join"],
        "entry": observed_summary(observation)}
    path, metadata = cache_function(parent, original, output)
    result = dict(observed_summary(observation), kind="full_panel_native_ap_features_v1",
                  feature_cache=record(path), feature_metadata=record(path.with_suffix(".json")),
                  helper=AP_CACHE_HELPER, feature_bridge=helper_ref(observation,
                      "src/arggnn/ancestralpaths_native_bridge.py"),
                  mechanical_adapter="only ObservedPanel diploid-order range literal280_to952",
                  native_shells=5, native_columns=9, head_classes=6,
                  truth_accessed=False, metadata=metadata)
    result["manifest"] = write_json(output / "features.json", result, guard)
    _guard(guard, 0)
    return result


class QueryTruth:
    """Only root-bound external query targets at original coordinates; no trees."""
    def __init__(self, observation: Mapping[str, Any], truth_record: Mapping[str, Any],
                 purpose: str, method: str):
        self.observation = observation
        self.capability = contracts.authorize_truth_use(
            observation["parent_id"], purpose, method=method,
            policy=observation["protocol"]["policy"])
        require(purpose != "SEALED_OUTER_SCORE", "This preparation bridge has no outer truth capability")
        directory = local(DATA + "/" + observation["parent_id"])
        require(local(truth_record["path"]) == directory / "truth/full_truth_intervals.npz",
                "Exact external query target path required")
        # Generation metadata is read only after the downstream truth capability.
        generation = json.loads((directory / "generation.json").read_text())
        require(generation["parent_id"] == observation["parent_id"] and
                generation["files"]["truth/full_truth_intervals.npz"]["sha256"] == truth_record["sha256"],
                "External target binding differs from producer receipt")
        path = checked(truth_record)
        with np.load(path, allow_pickle=False) as z:
            self.intervals = z["intervals_bp"].copy()
            self.paths = z["native_paths_1based"].copy()
            self.c4 = z["source_classes_0based"].copy()
            raw = z["query_raw_node_ids"].copy()
        expected_raw = [r["source_observed_node_id"] for r in observation["rows"] if r["role"] == "query"]
        require(self.intervals.ndim == 2 and self.intervals.shape[1:] == (2,) and len(self.intervals) > 0
                and bool(np.isfinite(self.intervals).all()) and self.intervals[0, 0] == 0
                and self.intervals[-1, 1] == 2000000
                and bool(np.all(self.intervals[:, 1] > self.intervals[:, 0]))
                and np.array_equal(self.intervals[:-1, 1], self.intervals[1:, 0]),
                "External intervals must cover the exact full original domain")
        require(self.paths.shape == self.c4.shape == (len(self.intervals), 686)
                and self.paths.dtype.kind in "iu" and self.c4.dtype.kind in "iu"
                and bool(np.all((self.paths >= 1) & (self.paths <= 6)))
                and np.array_equal(self.c4, np.asarray([0, 1, 2, 3, 0, 1])[self.paths - 1])
                and np.array_equal(raw, expected_raw), "Native query six-path/C4 target identity differs")
        self.provenance = dict(truth_record, purpose=purpose, method=method,
            downstream_role=observation["downstream_role"], query_columns=686,
            labels_enter_observed_features=False, donor_census_targets_used=False)

    def labels_at(self, positions: Any) -> np.ndarray:
        positions = np.asarray(positions)
        require(bool(np.isfinite(positions).all()) and bool(np.all((positions >= 0) & (positions < 2000000))),
                "Query target join outside original domain")
        return self.paths[np.searchsorted(self.intervals[:, 1], positions, side="right")]

    def c4_at(self, positions: Any) -> np.ndarray:
        return np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int16)[self.labels_at(positions) - 1]


def ap_sampler_config(policy: Mapping[str, Any]) -> dict[str, Any]:
    ap = policy["native_method_contracts"]["ancestralpaths"]
    return {"sampler": {
        "maximum_source_passes": ap["maximum_passes"],
        "unfilled_quota_policy": "retain_realized_accepted_rows_and_continue_cohorts",
        "per_parent_cohort_budgets": {cohort: {
            "target": ap["requested_rows_per_train_parent"][cohort],
            "per_path_cap": ap["BAA_path_cap"] if cohort == "BAA" else ap["non_BAA_path_cap"],
            "candidate_scan_budget": ap["candidate_scan_budget_per_parent"][cohort],
        } for cohort in contracts.SCAN_BUDGET}}}


def prepare_ap_training(observation: Mapping[str, Any], feature_record: Mapping[str, Any],
                        truth_record: Mapping[str, Any], output_dir: str | Path, *,
                        guard: Callable[[int], Any]) -> dict[str, Any]:
    """Reuse exact v2 realized sampler, including /3 overhead and float pass stride."""
    require(observation["downstream_role"] == "TRAIN", "AP training never consumes internal VAL or outer truth")
    features = json.loads(checked(feature_record, observed=True).read_text())
    require(features["kind"] == "full_panel_native_ap_features_v1"
            and features["parent_id"] == observation["parent_id"]
            and features["tree"] == observation["tree_record"]
            and features["inference"] == observation["inference_record"]
            and features["truth_accessed"] is False, "Bound feature-manifest lineage differs")
    cache = checked(features["feature_cache"], observed=True)
    cache_meta = json.loads(checked(features["feature_metadata"], observed=True).read_text())
    require(cache_meta["cache_sha256"] == features["feature_cache"]["sha256"]
            and cache_meta["observed_tree_sha256"] == observation["tree_record"]["sha256"]
            and cache_meta["truth_accessed"] is False, "Feature cache native metadata differs")
    with np.load(cache, allow_pickle=False) as z:
        require(np.array_equal(z["query_node_ids"], observation["roster"].query_nodes)
                and np.array_equal(z["site_positions_bp"], observation["original_site_positions_bp"])
                and np.array_equal(z["positions_bp"], observation["support"]["original_centers_bp"])
                and np.array_equal(z["coverage_original_bp0"], observation["support"]["coverage_original_bp0"])
                and np.array_equal(z["bin_edges_bp"], observation["support"]["clipped_bin_edges_bp"])
                and np.array_equal(z["retained_fixed100bp_indexes"], observation["support"]["grid_indexes"])
                and float(z["local_to_original_offset_bp"]) == observation["support"]["coverage_original_bp0"][0]
                and np.array_equal(z["query_sample_indices"], [r["sample_index"] for r in observation["rows"] if r["role"] == "query"])
                and z["site_feature_index"].shape == (observation["ts"].num_sites, 686)
                and z["grid_feature_index"].shape == (len(observation["support"]["grid_indexes"]), 686)
                and z["features"].ndim == 3 and z["features"].shape[1:] == (5, 9)
                and z["features"].dtype == np.float32 and bool(np.isfinite(z["features"]).all())
                and bool(np.all(z["site_feature_index"] >= 0))
                and bool(np.all(z["site_feature_index"] < len(z["features"])))
                and bool(np.all(z["grid_feature_index"] >= 0))
                and bool(np.all(z["grid_feature_index"] < len(z["features"]))),
                "Full observed feature-cache correspondence differs")
    truth = QueryTruth(observation, truth_record, "TRAIN_SUPERVISED_ROW_SELECTION", "ancestralpaths_dense6")
    output = new_directory(output_dir, guard)
    cfg = ap_sampler_config(observation["protocol"]["policy"])
    maximum = sum(b["target"] for b in cfg["sampler"]["per_parent_cohort_budgets"].values())
    _guard(guard, maximum * 256 + 8 * 1024 * 1024)
    sampler_ref = helper_ref(observation, "scripts/train_ancestralpaths_biological_v2.py")
    sampler = pinned_module(sampler_ref, "full_native_frozen_ap_sampler_v2")
    cache_module = pinned_module(AP_CACHE_HELPER, "full_native_frozen_ap_cache")
    def reader(parent: Any, purpose: str) -> QueryTruth:
        require(purpose == "train_supervision" and parent["parent_id"] == observation["parent_id"],
                "Frozen sampler requested wrong target capability")
        return truth
    definition = SimpleNamespace(TruthReader=reader, write_json=cache_module.write_json)
    queries = [r for r in observation["rows"] if r["role"] == "query"]
    parent = {"parent_id": observation["parent_id"],
        "query_slots": np.asarray([r["sample_index"] for r in queries], dtype=np.int32),
        "cohorts": np.asarray([r["observed_cohort"] for r in queries])}
    x, y, reports = sampler.sample_realized_source_rows(definition, parent, cache, cfg, output)
    training = write_npz(output / "training_rows.npz", {
        "features": x, "native_labels_0based": y,
        "native_targets_onehot6": np.eye(6, dtype=np.float32)[y],
    }, guard)
    result = dict(observed_summary(observation), kind="full_panel_native_ap_training_v1",
        features=feature_record, training_rows=training, sampler=sampler_ref,
        selection=record(output / (observation["parent_id"] + "_training_selection.npz")),
        sampler_report=record(output / (observation["parent_id"] + "_sampler.json")),
        truth_provenance=truth.provenance, reports=reports, realized_rows=len(y),
        requested_rows=maximum, shortfall_rows=maximum - len(y), source_num_seq_stride=3,
        source_overhead=10000 / 3, phantom_rows=0, final_checkpoint_epoch=5,
        internal_VAL_selection=False, native_head_classes=6)
    result["manifest"] = write_json(output / "training.json", result, guard)
    _guard(guard, 0)
    return result


def prepare_argmix_targets(observation: Mapping[str, Any], truth_record: Mapping[str, Any],
                           output_dir: str | Path, *, guard: Callable[[int], Any]) -> dict[str, Any]:
    """External stock SNP NPZ, never donor targets or outer supervision."""
    contracts.validate_observed_roster(observation["rows"], require_argmix_native_ids=True)
    role = observation["downstream_role"]
    require(role in {"TRAIN", "INTERNAL_VAL"}, "ARGMix preparation refuses outer targets")
    purpose = "TRAIN_GRADIENTS" if role == "TRAIN" else "NATIVE_INTERNAL_VAL_SELECTION"
    truth = QueryTruth(observation, truth_record, purpose, "argmix_current_joint_c4")
    output = new_directory(output_dir, guard)
    queries = [r for r in observation["rows"] if r["role"] == "query"]
    native_truth = write_npz(output / (observation["parent_id"] + ".native_truth.npz"), {
        "variant_pos": observation["local_site_positions_bp"].astype(np.int64),
        "truth__hap_labels": truth.c4_at(observation["original_site_positions_bp"]).astype(np.int16),
        "truth__sample_ids": np.asarray([r["sample_id"] for r in queries[::2]], dtype=str),
        "ancestry_names": np.asarray(contracts.CLASS_ORDER, dtype=str),
    }, guard)
    result = dict(observed_summary(observation), kind="full_panel_native_argmix_targets_v1",
        native_truth=native_truth, truth_provenance=truth.provenance,
        label_frame="local integer SNP bp0; exact original SNP census350 join",
        gradients_allowed=role == "TRAIN", native_selection_only=role == "INTERNAL_VAL",
        observed_features_changed=False, donor_truth_labels=0, ancestry_names=list(contracts.CLASS_ORDER))
    result["manifest"] = write_json(output / "targets.json", result, guard)
    return result


def prepare_argmix_archives(frames: Mapping[str, Mapping[str, Any]],
                            targets: Mapping[str, Mapping[str, Any]],
                            output_dir: str | Path, *,
                            guard: Callable[[int], Any], admitted_cli_record: Mapping[str, Any],
                            native_modules: tuple[Any, Any, Any, Any, Any],
                            archive_write_upper_bound_bytes: Mapping[str, int]) -> dict[str, Any]:
    """Unchanged native Rust build_training and stock archive validation.

    Root supplies independently qualified Torch/ancpath/joint/trainer/loader modules
    and bound CLI after runtime admission. This API performs no fitting or CUDA use.
    It requires the exact two TRAIN and one INTERNAL_VAL biological parent.
    """
    import yaml
    expected = {p for p, role in contracts.PARENT_ROLES.items() if role in {"TRAIN", "INTERNAL_VAL"}}
    require(set(frames) == set(targets) == expected, "Exact downstream TRAIN000/001 and VAL002 required")
    require(set(archive_write_upper_bound_bytes) == {"train", "val"}
            and all(type(v) is int and v > 0 for v in archive_write_upper_bound_bytes.values()),
            "Root native archive write upper bounds required")
    for bundle, kind in [(f, "full_panel_native_observed_frame_v1") for f in frames.values()] + [
            (t, "full_panel_native_argmix_targets_v1") for t in targets.values()]:
        frozen = json.loads(checked(bundle["manifest"]).read_text())
        require(frozen == {k: jsonable(v) for k, v in bundle.items() if k != "manifest"}
                and frozen["kind"] == kind, "Changed/unbound frame or target manifest")
    first = frames["ap_full_train_000"]
    policy = protocol()["policy"]
    bridge_ref = next(r for r in policy["helper_algorithm_refs"] if
                      r["path"] == "src/arggnn/official_argmix_biological.py")
    bridge = pinned_module(bridge_ref, "full_native_frozen_argmix_bridge")
    bridge.source_inventory(require_copy=True)
    require(checked(admitted_cli_record, observed=True) == bridge.CLI, "Root-admitted stock Rust CLI required")
    require(len(native_modules) == 5, "Qualified native modules required")
    torch, ancpath, joint, trainer, loader = native_modules
    require(Path(sys.prefix).resolve() == bridge.PREFIX and sys.version_info[:2] == (3, 12)
            and torch.__version__ == "2.10.0+cu129" and torch.version.cuda == "12.9",
            "Root-qualified native ABI210/cu129 interpreter required")
    require(not torch.cuda.is_initialized(), "Archive CPU stage must not create CUDA contexts")
    for module in (joint, trainer, loader):
        require(Path(module.__file__).resolve().is_relative_to(bridge.SOURCE / "src"),
                "Qualified unchanged execution-copy module required")
    output = new_directory(output_dir, guard)
    pairs: dict[str, list[Any]] = {"train": [], "val": []}
    for parent in contracts.PARENT_ROLES:
        if parent not in expected:
            continue
        frame, target = frames[parent], targets[parent]
        require(frame["parent_id"] == target["parent_id"] == parent
                and frame["downstream_role"] == target["downstream_role"] == contracts.PARENT_ROLES[parent]
                and frame["class_order"] == list(contracts.CLASS_ORDER)
                and target["ancestry_names"] == list(contracts.CLASS_ORDER)
                and frame["stock_native_ID_correspondence_verified"] is True
                and frame["tree"] == target["tree"]
                and frame["inference"] == target["inference"], "Frame/target lineage differs")
        rows = json.loads(checked(frame["roster"], observed=True).read_text())
        require(checked(frame["references"], observed=True).read_text() == argmix_reference_text(rows)
                and checked(frame["native_sample"], observed=True).read_text() == native_sample_text(rows),
                "Stock metadata changed")
        split = "train" if frame["downstream_role"] == "TRAIN" else "val"
        pairs[split].append({"rel": str(checked(frame["tree"], observed=True)),
            "references": str(checked(frame["references"], observed=True)),
            "relate_sample": str(checked(frame["native_sample"], observed=True)),
            "truth_npz": str(checked(target["native_truth"]))})
    pair_records = {s: write_text(output / (s + "_pairs.yaml"), yaml.safe_dump(pairs[s], sort_keys=False), guard)
                    for s in ("train", "val")}
    map_text = checked(first["local_map"], observed=True).read_text()
    require(map_text == "position Genetic_Map(cM)\n0 0\n2000000 2\n"
            and all(checked(f["local_map"], observed=True).read_text() == map_text for f in frames.values()),
            "Local native uniform map differs")
    map_record = write_text(output / "local_uniform_map.tsv", map_text, guard)
    backend = bridge.backend_config({"class_order": list(contracts.CLASS_ORDER),
        "native_model": copy.deepcopy(bridge.NATIVE_MODEL), "joint_model": copy.deepcopy(bridge.JOINT_MODEL)}, output)
    # Stock training startup counting needs one exact observed metadata pointer.
    # All parent role/order/class identities have just been independently checked.
    backend["paths"]["references"] = str(checked(first["references"], observed=True))
    loader.validate_config(backend)
    token_to_id = joint.build_token_to_id_from_config(backend)
    token_to_class = joint.build_token_to_class_from_config(backend)
    backend_record = write_text(output / "backend.yaml", yaml.safe_dump(backend, sort_keys=False), guard)
    archive_info, commands = {}, []
    for split, code in (("train", 0), ("val", 1)):
        archive = output / (split + ".pt")
        # Runtime/storage admission determines the approved native archive upper
        # bound. Root guard must account for bulk subprocess writes before launch.
        _guard(guard, archive_write_upper_bound_bytes[split])
        argv = [str(bridge.CLI), "--config", str(checked(backend_record)),
                "--split", split, "--output", str(archive)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        started = time.monotonic()
        with (output / (split + ".Rust.log")).open("xb") as stream:
            result = subprocess.run(argv, cwd=bridge.SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        command = {"argv": argv, "exit_code": result.returncode,
            "elapsed_seconds": time.monotonic() - started, "stock_binary": dict(admitted_cli_record),
            "log": record(output / (split + ".Rust.log"))}
        commands.append(command)
        write_json(output / (split + ".command.json"), command, guard)
        require(result.returncode == 0 and archive.is_file(), "Unchanged native Rust archive stage failed")
        require(archive.stat().st_size + (output / (split + ".Rust.log")).stat().st_size
                <= archive_write_upper_bound_bytes[split], "Native archive/log exceeded root write admission")
        payload = trainer._load_gt_gru_split_pt(str(archive), expected_split_code=code)
        require(payload["input_ids"].shape[1:] == (1024, 16) and bool(payload["row_mask"].any())
                and payload["base_encoder"] == "tmrca_transformer" and payload["format_version"] == 2
                and payload["pad_id"] == 0 and payload["vocab_size"] == len(token_to_id)
                and payload["cm_features"].shape[-1] == 2, "Stock native archive row/chunk/vocab contract differs")
        cm = payload["cm_features"].numpy()
        valid = payload["labels"][payload["row_mask"]]
        require(bool(np.isfinite(cm).all()) and bool(np.all(cm[..., 0] >= 0))
                and not bool((valid < 0).any()) and not bool((valid >= 4).any()),
                "Stock archive map/target values invalid")
        archive_info[split] = dict(record(archive), chunks=payload["num_chunks"],
            valid_rows_including_overlap=int(payload["row_mask"].sum()), metadata=payload["metadata"].tolist())
    result = {"schema": SCHEMA, "kind": "full_panel_native_argmix_archives_v1",
        "frames": {p: f["manifest"] for p, f in frames.items()},
        "targets": {p: t["manifest"] for p, t in targets.items()},
        "backend": backend_record, "archives": archive_info, "commands": commands,
        "train_pairs": pair_records["train"], "val_pairs": pair_records["val"], "local_map": map_record,
        "token_to_id": token_to_id, "token_to_class": token_to_class,
        "class_order": list(contracts.CLASS_ORDER), "source_aliases": {"ANA": "Ana"},
        "native_VAL_selection": "source smoothed unweighted mean of batch masked losses",
        "checkpoint_selection_changed": False, "stock_source_unchanged": True,
        "archive_write_upper_bound_bytes": dict(archive_write_upper_bound_bytes),
        "stock_helper": bridge_ref, "source_tree_sha256": bridge.SOURCE_TREE_SHA,
        "query_ancestry_in_observed_features": False, "older_reference_count": 34,
        "donor_pool": "all266_source_deme_prototypes", "models_fitted": False}
    result["manifest"] = write_json(output / "native_prepared.json", result, guard)
    _guard(guard, 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--parent-id", choices=list(contracts.PARENT_ROLES))
    parser.add_argument("--inference-record", help="JSON path/hash object, supplied by root")
    parser.add_argument("--completion-record", help="JSON path/hash object, supplied by root")
    parser.add_argument("--require-argmix-native-ids", action="store_true")
    args = parser.parse_args()
    # CLI is read-only. Bulk mutations are importable root-guarded APIs only.
    proto = protocol(args.config, args.split)
    if args.parent_id:
        require(args.inference_record and args.completion_record, "Root-bound receipt records required")
        observation = read_completed_parent(args.parent_id, json.loads(args.inference_record),
            json.loads(args.completion_record), config_path=args.config, split_path=args.split,
            require_argmix_native_ids=args.require_argmix_native_ids)
        print(json.dumps(jsonable(observed_summary(observation)), sort_keys=True))
    else:
        print(json.dumps({"schema": SCHEMA, "status": "source_policy_validated",
            "config": proto["config_record"], "split": proto["split_record"],
            "parent_files_opened": False, "truth_opened": False, "models_fitted": False,
            "execution_admission_created": False}, sort_keys=True))


if __name__ == "__main__":
    main()

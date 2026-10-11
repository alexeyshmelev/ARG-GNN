"""H038 expanded released AncestralPaths Dense adapter; no execution at import.

Root owns run, runtime and storage admission. Only guarded callable stages mutate.
Observed features and predictions never receive truth. TRAIN query labels are
separate capabilities, and native six-path focal metrics are separate from C4.
Frozen author sources, old pilot policies and native algorithms stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

from . import ancestralpaths_native_bridge as bridge
from . import full_panel_native_baseline_contracts_v1 as contracts

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/expanded_native_ancestralpaths_v1.py"
AUTHOR = ".runtime/author_sources/ancestralpaths_77db3cfae100"
DRAFT = "research/reviews/expanded_native_ancestralpaths_v1_author/config.draft.json"
HELPERS = {
    "bridge": "src/arggnn/ancestralpaths_native_bridge.py",
    "contracts": "src/arggnn/full_panel_native_baseline_contracts_v1.py",
    "full_cache_adapter": "src/arggnn/prepare_full_panel_native_supervised_v1.py",
    "cache": "scripts/train_ancestralpaths_biological_v1.py",
    "sampler": "scripts/train_ancestralpaths_biological_v2.py",
    "producer": "src/arggnn/prepare_expanded_full_panel_v1.py",
    "original_producer": "src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py",
}
TARGETS = dict(present_day=50000, Bronze_Age=50000, BAA=12500, Neo=25000, Yam=25000)
SCAN = dict(present_day=100000, Bronze_Age=100000, BAA=25000, Neo=50000, Yam=25000)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _local(value, *, observed=False):
    path = Path(value)
    _require(".." not in path.parts, "Path traversal forbidden")
    if path.is_absolute():
        path = path.relative_to(ROOT)
    _require(bool(path.parts), "Dedicated project artifact required")
    for n in range(1, len(path.parts) + 1):
        _require(not ROOT.joinpath(*path.parts[:n]).is_symlink(), "Artifact symlink forbidden")
    if observed:
        _require(not any(word in part.lower() for part in path.parts
                         for word in ("truth", "oracle", "census")), "Observed capability cannot read truth")
    result = ROOT / path
    result.resolve().relative_to(ROOT)
    return result


def _ref(value):
    path = _local(value)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=h.hexdigest())


def _checked(ref, *, observed=False):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact artifact reference required")
    path = _local(ref["path"], observed=observed)
    _require(_ref(path) == ref, "Bound artifact bytes differ")
    return path


def _read(ref, *, observed=False):
    return json.loads(_checked(ref, observed=observed).read_text())


def _jsonable(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _guard(guard, planned=0):
    _require(Path.cwd().resolve() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
             "Dedicated repository and explicit CPU visibility required")
    _require(callable(guard) and guard(int(planned)) is not False, "Root admitted write guard required")


def _directory(value, guard):
    path = _local(value)
    _guard(guard, 16384)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save(path, value, guard, *, npz=False):
    if npz:
        stream = io.BytesIO()
        np.savez_compressed(stream, **value)
        payload = stream.getvalue()
    else:
        payload = (json.dumps(_jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _guard(guard, len(payload) + 4096)
    with path.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return _ref(path)


def _module(ref, name):
    path = _checked(ref, observed=True)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def default_config_v1():
    return dict(schema="expanded_native_ancestralpaths_v1_config", hypothesis_id="H038",
        status="source_only_draft_root_admission_required", source_ref=_ref(SOURCE),
        generation_config_ref=_ref("configs/expanded_full_panel_v1.json"),
        registration_ref=_ref("research/reviews/expanded_full_panel_v1_root/registration.json"),
        helper_refs={name: _ref(path) for name, path in HELPERS.items()},
        author_root=AUTHOR, author_commit=bridge.AUTHOR_COMMIT,
        author_archive_ref=_ref(".runtime/author_sources/ancestralpaths_77db3cfae100.tar.gz"),
        published_archive_ref=_ref(".runtime/author_sources/ancestralpaths_published_zenodo8319452.zip"),
        native_model=bridge.native_model_spec(), tensorflow_cpu_prefix=".runtime/conda/envs/ancestralpaths_cpu_v1",
        training=dict(seed=101, epochs=5, batch_size=32, shuffle=True, checkpoint="final_epoch5",
                      VAL_selection=False, prediction_batch_size=128),
        sampler=dict(source_num_seq=3, maximum_source_passes=128,
            unfilled_quota_policy="retain_realized_accepted_rows_and_continue_cohorts",
            per_parent_cohort_budgets={name: dict(target=TARGETS[name], candidate_scan_budget=SCAN[name],
                per_path_cap=6250 if name == "BAA" else 12500) for name in TARGETS}),
        split_counts=dict(TRAIN=32, INTERNAL_VAL=8, outer_prepared=0),
        C4_class_order=["Ana", "CHG", "WHG", "EHG"], native_path_axis_1based=list(range(1, 7)),
        published_CNN_reproduction=False, original_paper_numerical_reproduction=False)


def _protocol(config):
    _require(config == default_config_v1(), "Frozen native method/config/source pins differ")
    bridge.inspect_author_source(_local(config["author_root"]),
        archive=_checked(config["author_archive_ref"], observed=True),
        published_archive=_checked(config["published_archive_ref"], observed=True))
    generation = _read(config["generation_config_ref"], observed=True)
    registration = _read(config["registration_ref"], observed=True)
    registry = registration["parent_registry"]
    _require(registry == generation["parent_registry"] and registration["registry_sha256"] == _digest(registry)
        and registration["config_ref"] == config["generation_config_ref"]
        and registration["source_ref"] == config["helper_refs"]["producer"]
        and registration["outer_evaluation_authorized"] is False
        and generation["output_root"] == "data/expanded_full_panel_v1", "Expanded registration lineage differs")
    roles = {row["parent_id"]: row["split"] for row in registry}
    _require(len(roles) == 48 and [list(roles.values()).count(s) for s in ("train", "val", "outer")] == [32, 8, 8],
             "Registered 32TRAIN/8VAL/8 reserved outer split required")
    return generation, registry, roles


def read_completed_parent_v1(parent_id, completion_ref, *, config):
    """Root-bound completed post-size observations, full actual IDs and ages."""
    import tskit
    generation, registry, roles = _protocol(config)
    _require(roles.get(parent_id) in ("train", "val"), "Only registered inner parents may be prepared")
    row = next(r for r in registry if r["parent_id"] == parent_id)
    base = generation["output_root"] + "/" + parent_id
    _require(completion_ref["path"] == base + "/expanded_parent_completed.json", "Exact completion path required")
    completed = _read(completion_ref, observed=True)
    _require(completed["parent_id"] == parent_id and completed["split"] == row["split"]
        and completed["generation_completed"] is True and completed["Relate_completed"] is True
        and completed["source_ref"] == config["helper_refs"]["producer"]
        and completed["original_producer_ref"] == config["helper_refs"]["original_producer"]
        and completed["config_content_sha256"] == _digest(generation)
        and completed["parent_registry_sha256"] == _digest(registry)
        and completed["outer_truth_evaluated"] is False, "Completed expanded lineage differs")
    inference_ref = completed["inference_ref"]
    _require(inference_ref["path"] == base + "/relate/inference.json", "Native inference receipt path differs")
    inference = _read(inference_ref, observed=True)
    _require(inference["parent_id"] == parent_id and inference["split"] == row["split"]
        and inference["status"] == "post_five_iterations_inferred"
        and inference["truth_or_oracle_read"] is False and inference["outer_command_returncodes"] == [0] * 5,
        "Completed native post-size inference required")
    post = inference["post_five_iterations"]
    _require(inference["inferred_tree"]["path"] == base + "/relate/post_size.observed.trees"
        and inference["observed_sample_manifest"]["path"] == base + "/relate/observed_sample_manifest.json"
        and post["clean_tree_sha256"] == inference["inferred_tree"]["sha256"]
        and post["observed_sample_manifest"] == inference["observed_sample_manifest"]
        and post["sample_age_order_preserved"] is True and post["node_ids_preserved_during_clipping"] is True,
        "Actual primary observed tree/roster bindings differ")
    tree_path = _checked(inference["inferred_tree"], observed=True)
    rows = _read(inference["observed_sample_manifest"], observed=True)
    roster = contracts.validate_observed_roster(rows)
    _require(post["sample_index_to_inferred_tree_node_id"] == [
        dict(sample_index=r["sample_index"], inferred_tree_node_id=r["inferred_tree_node_id"]) for r in rows],
        "Actual inferred sample index mapping differs")
    support = contracts.validate_physical_support(post)
    joins = contracts.validate_site_coordinate_join(post["site_join"],
        coverage_original_bp0=support["coverage_original_bp0"], declared_site_count=post["sites"])
    producer = _module(config["helper_refs"]["original_producer"], "arggnn._expanded_ap_original_producer")
    _require(set(inference["input_observed_sha256"]) == set(producer.OBSERVED_FILES), "Native observed whitelist differs")
    for name, sha in inference["input_observed_sha256"].items():
        _checked(dict(path=base + "/observed/" + name, sha256=sha), observed=True)
    assets = producer.observed_assets(generation, row, _local(base, observed=True))
    _require(assets["source_sha256"] == inference["input_observed_sha256"], "Original observed inputs differ")
    for actual, source in zip(rows, assets["rows"]):
        _require(all(actual[k] == source[k] for k in ("sample_index", "sample_id", "haplotype",
            "observed_cohort", "age_generations", "role", "observed_reference_class"))
            and actual["source_observed_node_id"] == source["observed_node_id"], "Original sample/date/role join differs")
    original_sites = json.loads(assets["paths"]["site_coordinate_manifest.json"].read_text())["rows"]
    for joined in post["site_join"]:
        source = original_sites[joined["original_site_id"]]
        _require(source["site_id"] == joined["original_site_id"]
            and source["tree_position_0based"] == joined["original_position_0based"]
            and source["VCF_position_1based"] == joined["VCF_position_1based"], "Exact original SNP conversion join differs")
    ts = tskit.load(tree_path)
    tables = ts.tables
    _require(not tables.metadata and ts.num_populations == ts.num_individuals == ts.num_migrations == 0
        and np.isin(tables.nodes.flags, [0, 1]).all()
        and np.all(tables.nodes.population == -1) and np.all(tables.nodes.individual == -1)
        and all(len(t.metadata) == 0 for t in (tables.nodes, tables.edges, tables.sites, tables.mutations,
            tables.individuals, tables.populations, tables.migrations)), "Observed trees cannot contain oracle annotations")
    _require(set(map(int, ts.samples())) == set(roster.all_sample_nodes)
        and ts.sequence_length == post["sequence_length_local_bp"]
        and ts.num_sites == post["sites"] and ts.num_trees == post["trees"], "Native tree counts/physical domain differ")
    _require(all(abs(ts.node(r["inferred_tree_node_id"]).time - r["age_generations"]) <= 1e-6 for r in rows),
             "Actual sampled ages differ")
    _require(np.array_equal(ts.tables.sites.position, [j["local_position_0based"] for j in post["site_join"]]),
             "Complete native/local SNP order differs")
    for cohort in TARGETS:
        contracts.native_ap_stride(ts.num_sites, cohort, traversal="TRAIN" if row["split"] == "train" else "HOLDOUT")
    return dict(parent_id=parent_id, split=row["split"], ts=ts, rows=rows, roster=roster,
        support=support, site_join=post["site_join"], completion_ref=completion_ref,
        generation_ref=completed["generation_ref"], inference_ref=inference_ref,
        tree_ref=inference["inferred_tree"], config_digest=_digest(config), targets_read=False)


def _lineage(observation, config):
    return dict(parent_id=observation["parent_id"], split=observation["split"],
        source_ref=config["source_ref"], config_digest=_digest(config),
        completion_ref=observation["completion_ref"], inference_ref=observation["inference_ref"],
        tree_ref=observation["tree_ref"], input_samples=952, query_haplotypes=686,
        pure_reference_haplotypes=266, source_class_counts=[50, 28, 102, 86],
        source_num_seq_stride=3, outer_included=False, paper_reproduction=False)


def prepare_parent_v1(observation, output_directory, *, config, guard):
    """Observed-only native features for every SNP and retained original center."""
    _protocol(config)
    _require(observation["config_digest"] == _digest(config) and observation["targets_read"] is False,
             "Bound observed parent required")
    ts, rows, support = observation["ts"], observation["rows"], observation["support"]
    _require(not np.isin(ts.tables.edges.parent, observation["roster"].all_sample_nodes).any(),
             "Optimized native cache requires terminal observed sample nodes")
    output = _directory(output_directory, guard)
    bound = ts.num_trees * 686 * 45 * 4 + (len(support["grid_indexes"]) + ts.num_sites) * 686 * 4
    _guard(guard, int(bound * 1.02) + 16 * 1024 * 1024)
    adapter = _module(config["helper_refs"]["full_cache_adapter"], "arggnn._expanded_ap_cache_adapter")
    cache_function, _ = adapter._full_roster_cache_function()
    queries = [r for r in rows if r["role"] == "query"]
    parent = dict(parent_id=observation["parent_id"], split=observation["split"], samples=rows,
                  query_slots=np.asarray([r["sample_index"] for r in queries], dtype=np.int32))
    payload = dict(tree_path=_checked(observation["tree_ref"], observed=True),
        nodes=np.asarray(observation["roster"].all_sample_nodes, dtype=np.int64),
        coverage=np.asarray(support["coverage_original_bp0"]), offset=support["coverage_original_bp0"][0],
        positions=np.asarray(support["original_centers_bp"]), bin_edges=np.asarray(support["clipped_bin_edges_bp"]),
        retained_indexes=list(support["grid_indexes"]), site_join=observation["site_join"],
        entry=_lineage(observation, config))
    cache, metadata = cache_function(parent, payload, output)
    result = dict(_lineage(observation, config), schema="expanded_native_ap_features_v1",
        feature_cache_ref=_ref(cache), feature_metadata_ref=_ref(cache.with_suffix(".json")),
        feature_age_mode="training_int", native_shells=5, native_columns=9, targets_read=False,
        query_rows=queries, helper_refs=config["helper_refs"],
        optimized_native_first_tree_bridge_parity=metadata["source_bridge_float32_equivalence_first_observed_tree"])
    result["manifest_ref"] = _save(output / "features.json", result, guard)
    _guard(guard)
    return result


def _features(ref, config):
    value = _read(ref, observed=True)
    _require(value["schema"] == "expanded_native_ap_features_v1" and value["source_ref"] == config["source_ref"]
        and value["config_digest"] == _digest(config) and value["targets_read"] is False,
        "Exact observed feature manifest required")
    with np.load(_checked(value["feature_cache_ref"], observed=True), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    _require(arrays["features"].ndim == 3 and arrays["features"].shape[1:] == (5, 9)
        and arrays["features"].dtype == np.float32 and np.isfinite(arrays["features"]).all()
        and len(value["query_rows"]) == 686 and arrays["site_feature_index"].shape[1] == 686
        and arrays["grid_feature_index"].shape == (len(arrays["positions_bp"]), 686)
        and np.array_equal(arrays["query_sample_indices"], [r["sample_index"] for r in value["query_rows"]])
        and np.array_equal(arrays["query_node_ids"], [r["inferred_tree_node_id"] for r in value["query_rows"]]),
        "Full native feature/query dimensions differ")
    for name in ("site_feature_index", "grid_feature_index"):
        index = arrays[name]
        _require(index.dtype.kind in "iu" and np.all(index >= 0) and np.all(index < len(arrays["features"])),
                 "Native feature index outside complete bank")
    return value, arrays


def _labels_at(intervals, paths, positions):
    positions = np.asarray(positions)
    _require(np.isfinite(positions).all() and np.all((positions >= 0) & (positions < 2000000)),
             "Original-coordinate truth join outside parent")
    return paths[np.searchsorted(intervals[:, 1], positions, side="right")]


class _TrainTruth:
    def __init__(self, observation, truth_ref, config):
        _, _, roles = _protocol(config)
        _require(roles.get(observation["parent_id"]) == observation["split"] == "train",
                 "External TRAIN query supervision only")
        base = "data/expanded_full_panel_v1/" + observation["parent_id"]
        _require(truth_ref["path"] == base + "/truth/full_truth_intervals.npz"
            and observation["generation_ref"]["path"] == base + "/generation.json", "Exact external target path required")
        generated = _read(observation["generation_ref"])
        _require(generated["parent_id"] == observation["parent_id"]
            and generated["files"]["truth/full_truth_intervals.npz"]["sha256"] == truth_ref["sha256"],
            "Producer external target binding differs")
        with np.load(_checked(truth_ref), allow_pickle=False) as archive:
            intervals = archive["intervals_bp"].copy()
            paths = archive["native_paths_1based"].copy()
            c4 = archive["source_classes_0based"].copy()
            raw = archive["query_raw_node_ids"].copy()
        _require(intervals.ndim == 2 and intervals.shape[1] == 2 and len(intervals) > 0
            and np.isfinite(intervals).all() and intervals[0, 0] == 0 and intervals[-1, 1] == 2000000
            and np.all(intervals[:, 1] > intervals[:, 0]) and np.array_equal(intervals[:-1, 1], intervals[1:, 0]),
            "External truth must cover the complete original domain")
        _require(paths.shape == c4.shape == (len(intervals), 686) and paths.dtype.kind in "iu"
            and np.all((paths >= 1) & (paths <= 6))
            and np.array_equal(c4, np.asarray(bridge.PATH_TO_SOURCE_CLASS)[paths - 1])
            and np.array_equal(raw, [r["source_observed_node_id"] for r in observation["rows"] if r["role"] == "query"]),
            "External six-path/C4/query identity differs")
        self.intervals, self.paths = intervals, paths
        self.provenance = dict(truth_ref, purpose="TRAIN_supervised_row_selection_and_gradients",
            query_haplotypes=686, labels_enter_features=False, donor_truth_used=False)

    def labels_at(self, positions):
        return _labels_at(self.intervals, self.paths, positions)


def prepare_train_rows_v1(observation, features_ref, truth_ref, output_directory, *, config, guard):
    """Unchanged v2 realized sampler: original /3, float stride, caps, duplicates."""
    value, arrays = _features(features_ref, config)
    _require(value["parent_id"] == observation["parent_id"] and value["tree_ref"] == observation["tree_ref"],
             "TRAIN feature lineage differs")
    truth = _TrainTruth(observation, truth_ref, config)
    output = _directory(output_directory, guard)
    maximum = sum(TARGETS.values())
    _guard(guard, maximum * 512 + 8 * 1024 * 1024)
    sampler = _module(config["helper_refs"]["sampler"], "arggnn._expanded_ap_realized_sampler")
    definition = SimpleNamespace(TruthReader=lambda parent, purpose: truth,
        write_json=lambda path, item: _save(Path(path), item, guard))
    parent = dict(parent_id=observation["parent_id"],
        query_slots=arrays["query_sample_indices"],
        cohorts=np.asarray([r["observed_cohort"] for r in value["query_rows"]]))
    x, y, reports = sampler.sample_realized_source_rows(definition, parent,
        _checked(value["feature_cache_ref"], observed=True), config, output)
    rows_ref = _save(output / "training_rows.npz", dict(features=x, native_labels_0based=y), guard, npz=True)
    result = dict(_lineage(observation, config), schema="expanded_native_ap_training_rows_v1",
        features_ref=features_ref, rows_ref=rows_ref, truth_provenance=truth.provenance,
        sampler_ref=config["helper_refs"]["sampler"], reports=reports,
        selection_ref=_ref(output / (observation["parent_id"] + "_training_selection.npz")),
        sampler_report_ref=_ref(output / (observation["parent_id"] + "_sampler.json")),
        realized_rows=len(y), requested_rows=maximum, shortfall_rows=maximum - len(y),
        phantom_rows=0, targets_read=True, source_overhead=10000 / 3,
        bounded_realized_sampler_departure_from_unlimited_author_loops=True, VAL_selection=False)
    result["manifest_ref"] = _save(output / "training.json", result, guard)
    return result


def _tensorflow(config):
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and Path(sys.prefix).resolve() == _local(config["tensorflow_cpu_prefix"]),
        "Qualified isolated TensorFlow CPU environment required")
    _require(importlib.metadata.version("tensorflow-cpu") == "2.15.1"
        and importlib.metadata.version("keras") == "2.15.0", "Qualified TF2.15.1/Keras2.15 versions required")
    import tensorflow as tf
    _require(not tf.config.list_physical_devices("GPU"), "CPU runtime must expose no GPU")
    return tf


def fit_all_train_v1(training_refs, output_directory, *, config, guard):
    """All32 TRAIN, native five-epoch Dense fit, no validation selection."""
    _, registry, _ = _protocol(config)
    expected = [r["parent_id"] for r in registry if r["split"] == "train"]
    _require(set(training_refs) == set(expected), "Exactly all32 registered TRAIN preparations required")
    output = _directory(output_directory, guard)
    xs, ys, reports = [], [], {}
    for parent_id in expected:
        value = _read(training_refs[parent_id])
        _require(value["schema"] == "expanded_native_ap_training_rows_v1"
            and value["parent_id"] == parent_id and value["split"] == "train"
            and value["source_ref"] == config["source_ref"] and value["config_digest"] == _digest(config)
            and value["requested_rows"] == sum(TARGETS.values()) and value["phantom_rows"] == 0,
            "Frozen registered TRAIN rows required")
        with np.load(_checked(value["rows_ref"]), allow_pickle=False) as archive:
            x, y = archive["features"].copy(), archive["native_labels_0based"].copy()
        _require(x.shape == (len(y), 5, 9) and len(y) == value["realized_rows"] > 0
            and np.isfinite(x).all() and y.dtype.kind in "iu" and np.all((y >= 0) & (y < 6)),
            "Genuine finite native TRAIN rows and six labels required")
        xs.append(x); ys.append(y); reports[parent_id] = value["reports"]
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    tf = _tensorflow(config)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(101)
    model = bridge.build_native_keras_model(tf.keras)
    _require(model.count_params() == 26630 and model.output_shape == (None, 6), "Released native Dense6 model required")
    started = time.monotonic()
    history = model.fit(x, tf.keras.utils.to_categorical(y, num_classes=6),
                       epochs=5, batch_size=32, shuffle=True, verbose=2)
    _require(len(history.history["loss"]) == 5
        and all(np.isfinite(v).all() for v in history.history.values()), "Five finite native fit epochs required")
    checkpoint = output / "native_dense_epoch5.keras"
    _guard(guard, model.count_params() * 32 + 2 * 1024 * 1024)
    _require(not checkpoint.exists(), "Checkpoint overwrite forbidden")
    model.save(checkpoint)
    result = dict(schema="expanded_native_ap_fit_v1", source_ref=config["source_ref"],
        config_digest=_digest(config), training_refs=training_refs, TRAIN_parent_ids=expected,
        checkpoint_ref=_ref(checkpoint), training_rows=len(y), native_path_counts=np.bincount(y, minlength=6),
        requested_rows=32 * sum(TARGETS.values()), shortfall_rows=32 * sum(TARGETS.values()) - len(y),
        sampler_reports=reports, epochs=5, batch_size=32, shuffle=True, seed=101,
        checkpoint_selection="final_epoch5", VAL_selection=False, outer_included=False,
        history=history.history, optimizer=model.optimizer.get_config(),
        fit_elapsed_seconds=time.monotonic() - started, GPU_used=False,
        native_model=config["native_model"], published_CNN_reproduction=False)
    result["manifest_ref"] = _save(output / "fit.json", result, guard)
    return model, result


def _native_focal_pairs(arrays, cohorts):
    """Author testing traversal, observation-defined; caps cannot bind below10k."""
    sites = len(arrays["site_positions_bp"])
    selected, names, reports = [], [], []
    for name in TARGETS:
        columns = np.flatnonzero(cohorts == name)
        step = math.floor(sites / (20000 / len(columns)))
        _require(step >= 1, "Native test initial stride is zero")
        initial, accepted, passes = step, [], []
        reason = "maximum_source_passes_reached"
        for number in range(128):
            before = len(accepted)
            for site in range(sites):
                if site % step != 0:
                    continue
                for query in columns:
                    accepted.append((site, int(query)))
                    if len(accepted) == 10000:
                        break
                if len(accepted) == 10000:
                    break
            passes.append(dict(pass_number=number, step=step, added=len(accepted) - before))
            if len(accepted) == 10000:
                reason = "source_target_filled"; break
            if len(accepted) == before:
                reason = "source_stride_pass_added_no_rows"; break
            step = step + step / 2
        selected.extend(accepted); names.extend([name] * len(accepted))
        reports.append(dict(cohort=name, initial_step=initial, requested=10000, realized=len(accepted),
            shortfall=10000 - len(accepted), duplicates=len(accepted) - len(set(accepted)), passes=passes,
            stop_reason=reason, non_binding_native_test_path_caps=25000 if name == "BAA" else 50000))
    return np.asarray(selected, dtype=np.int32), np.asarray(names), reports


def seal_parent_predictions_v1(fit_ref, features_ref, output_directory, *, config, guard):
    """Complete original100bp C4 plus separate native6 focal seal, no truth."""
    _, _, roles = _protocol(config)
    features, arrays = _features(features_ref, config)
    _require(roles.get(features["parent_id"]) == features["split"] == "val", "Registered internal VAL raw seal only")
    fit = _read(fit_ref, observed=True)
    _require(fit["schema"] == "expanded_native_ap_fit_v1" and fit["source_ref"] == config["source_ref"]
        and fit["config_digest"] == _digest(config) and fit["epochs"] == 5 and fit["VAL_selection"] is False,
        "Frozen final epoch5 checkpoint required")
    tf = _tensorflow(config)
    checkpoint = _checked(fit["checkpoint_ref"], observed=True)
    model = tf.keras.models.load_model(checkpoint)
    output = _directory(output_directory, guard)
    started = time.monotonic()
    p6 = bridge.predict_native_probabilities(model, arrays["features"], batch_size=128)
    collapsed = bridge.collapse_native_probabilities(p6)
    common = collapsed[arrays["grid_feature_index"]]
    queries = features["query_rows"]
    common_ref = _save(output / "common_C4_original100bp.npz", dict(source_probabilities=common,
        positions_bp=arrays["positions_bp"], bin_edges_bp=arrays["bin_edges_bp"],
        query_node_ids=arrays["query_node_ids"], query_sample_indices=arrays["query_sample_indices"],
        retained_fixed100bp_indexes=arrays["retained_fixed100bp_indexes"],
        coverage_original_bp0=arrays["coverage_original_bp0"],
        local_to_original_offset_bp=arrays["local_to_original_offset_bp"],
        raw_query_nodes=np.asarray([r["source_observed_node_id"] for r in queries])), guard, npz=True)
    cohorts = np.asarray([r["observed_cohort"] for r in queries])
    pairs, focal_cohorts, reports = _native_focal_pairs(arrays, cohorts)
    focal = p6[arrays["site_feature_index"][pairs[:, 0], pairs[:, 1]]]
    focal_ref = _save(output / "native6_SNP_focal.npz", dict(native_probabilities=focal,
        site_row=pairs[:, 0], query_column=pairs[:, 1], cohort=focal_cohorts,
        positions_bp=arrays["site_positions_bp"][pairs[:, 0]],
        query_sample_indices=arrays["query_sample_indices"][pairs[:, 1]],
        query_node_ids=arrays["query_node_ids"][pairs[:, 1]]), guard, npz=True)
    result = dict(schema="expanded_native_ap_prediction_seal_v1", parent_id=features["parent_id"],
        split="val", source_ref=config["source_ref"], config_digest=_digest(config),
        fit_ref=fit_ref, checkpoint_ref=fit["checkpoint_ref"], features_ref=features_ref,
        common_C4_ref=common_ref, native6_focal_ref=focal_ref, native_focal_sampler_reports=reports,
        C4_class_order=config["C4_class_order"], native_path_axis_1based=list(range(1, 7)),
        query_keys=[[r["sample_id"], r["haplotype"]] for r in queries],
        query_haplotypes=686, BAA_queries=14, native_six_argmax_unmasked=True,
        C4_collapse_before_argmax="p1+p5,p2+p6,p3,p4", no_tail_extrapolation=True,
        targets_read=False, VAL_selection=False, outer_included=False, calibration_fitted=False,
        prediction_elapsed_seconds=time.monotonic() - started)
    result["manifest_ref"] = _save(output / "prediction_seal.json", result, guard)
    return result


def native_focal_metrics_v1(probabilities, truth_paths_1based, *, cohorts):
    """Pure external scorer endpoint after seals; no target file reader here."""
    result = bridge.native_accuracy_confusion(probabilities, truth_paths_1based, cohorts=cohorts)
    confusion = result["confusion_predicted_rows_truth_columns"]
    predicted = confusion.sum(axis=1)
    result["precision_by_native_path"] = [
        float(confusion[i, i] / predicted[i]) if predicted[i] > 0 else None for i in range(6)]
    result["undefined_precision_policy"] = "null_when_no_predicted_rows"
    result["common_physical_C4_metric"] = False
    return result


def focused_join_check_v1():
    intervals = np.array([[0, 100], [100, 2000000]], dtype=np.float64)
    paths = np.array([[1, 2], [5, 6]], dtype=np.int8)
    assert np.array_equal(_labels_at(intervals, paths, [99, 100]), paths)
    p6 = np.array([[.30, .34, .02, .01, .30, .03]])
    np.testing.assert_allclose(bridge.collapse_native_probabilities(p6), [[.60, .37, .02, .01]])
    assert int(p6.argmax()) == 1 and int(bridge.collapse_native_probabilities(p6).argmax()) == 0
    return dict(original_bp0_boundary_join=True, source6_probability_collapse_before_argmax=True,
                actual_parent_or_target_arrays_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(default_config_v1(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

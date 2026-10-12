"""Root-only AP scoring draft. No filesystem/data read or scoring at import.

Not executed or admitted by the assistance reviewer. Root supplies verified H038
scoring parents, VAL-only target bindings, reader, metrics helper and write guard.
Native AP inference/training and existing sealed products remain unchanged.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import numpy as np

VAL_IDS = tuple(f"ap_expanded_val_{i:03d}" for i in range(8))
COHORT_COUNTS = {"present_day": 182, "Bronze_Age": 162,
                 "BAA": 14, "Neo": 302, "Yam": 26}
COMMON_FIELDS = {
    "source_probabilities", "positions_bp", "bin_edges_bp", "query_node_ids",
    "query_sample_indices", "retained_fixed100bp_indexes", "coverage_original_bp0",
    "local_to_original_offset_bp", "raw_query_nodes",
}
FOCAL_FIELDS = {
    "native_probabilities", "site_row", "query_column", "cohort",
    "positions_bp", "query_sample_indices", "query_node_ids",
}
TRUTH_FIELDS = {"intervals_bp", "query_raw_node_ids", "native_paths_1based",
                "source_classes_0based"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _json(reader, ref):
    return json.loads(reader.read(ref))


def _npz(reader, ref, expected):
    with np.load(io.BytesIO(reader.read(ref)), allow_pickle=False) as archive:
        _require(set(archive.files) == expected, "Sealed archive schema differs")
        return {name: archive[name].copy() for name in expected}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def jsonable_v1(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return jsonable_v1(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable_v1(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable_v1(v) for v in value]
    return value


def read_h038_scoring_only_v1(index, parent_id, reader):
    """Convenience for a root-verified cache index. Loads no model tensors."""
    _require(parent_id in VAL_IDS, "Only the registered eight VAL parents")
    record = index["parents"][parent_id]
    payload = reader.read(record["array_ref"])
    _require(len(payload) == record["compressed_bytes"],
             "H038 observed cache byte count differs")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        parent = {
            name: archive[value["array_key"]].copy()
            if "array_key" in value else value["json"]
            for name, value in record["scoring"].items()
        }
    _require(parent["parent_id"] == parent_id and parent["split"] == "val",
             "H038 scoring metadata parent/split differs")
    return parent


def _check_parent(parent, binding, reader):
    pid = parent["parent_id"]
    _require(pid in VAL_IDS and parent["split"] == binding["split"] == "val",
             "VAL-only parent and external binding required")
    q = np.asarray(parent["query_slots"])
    _require(q.shape == (686,) and q.dtype.kind in "iu",
             "Complete original686 sampled query columns required")
    _require(dict(Counter(np.asarray(parent["cohorts"]).tolist())) == COHORT_COUNTS,
             "Exact H038 full-panel query strata required")
    _require(binding["truth_npz"]["path"] ==
             f"data/expanded_full_panel_v1/{pid}/truth/full_truth_intervals.npz",
             "Exact registered same-parent external target path required")
    _require(binding["generation_receipt"]["path"] ==
             f"data/expanded_full_panel_v1/{pid}/generation.json",
             "Exact registered generation receipt required")
    generated = _json(reader, binding["generation_receipt"])
    _require(generated["parent_id"] == pid and
             generated["files"]["truth/full_truth_intervals.npz"]["sha256"] ==
             binding["truth_npz"]["sha256"],
             "External target hash must match the original generation receipt")
    return [[parent["samples"][int(i)]["sample_id"],
             parent["samples"][int(i)]["haplotype"]] for i in q]


def _common(reader, manifest, parent):
    """Return unmodified AP C4 and a small same-AP identity/coverage map."""
    arrays = _npz(reader, manifest["common_C4_ref"], COMMON_FIELDS)
    pairs = {
        "positions_bp": "positions", "bin_edges_bp": "edges",
        "query_sample_indices": "query_slots",
        "retained_fixed100bp_indexes": "retained_indexes",
        "raw_query_nodes": "raw_query_nodes",
        "coverage_original_bp0": "coverage_original",
    }
    for actual, expected in pairs.items():
        _require(np.array_equal(arrays[actual], parent[expected]),
                 "AP/H038 common support or sampled identity differs: " + actual)
    _require(np.asarray(arrays["local_to_original_offset_bp"]).shape == () and
             float(arrays["local_to_original_offset_bp"]) ==
             float(np.asarray(arrays["coverage_original_bp0"])[0]),
             "Original-coordinate AP offset differs")
    p = arrays["source_probabilities"]
    _require(p.dtype == np.float32 and
             p.shape == (len(parent["positions"]), 686, 4) and
             np.isfinite(p).all() and (p >= 0).all() and
             np.allclose(p.sum(-1), 1, atol=1e-6, rtol=1e-5),
             "Unmodified finite raw FP32 common C4 probabilities required")
    ids = arrays["query_node_ids"]
    _require(ids.shape == (686,) and ids.dtype.kind in "iu",
             "Native Relate sampled-node identity map required")
    # These IDs stay in the AP Relate namespace. Do not compare them with
    # H038 inferred internal/query-node IDs from another genealogy producer.
    meta = {"AP_query_node_ids": ids.copy(),
            "coverage_original": arrays["coverage_original_bp0"].copy()}
    return p, meta


def _focal(reader, manifest, parent, meta):
    from arggnn import ancestralpaths_native_bridge as bridge
    arrays = _npz(reader, manifest["native6_focal_ref"], FOCAL_FIELDS)
    p = bridge._probabilities(arrays["native_probabilities"])
    n = len(p)
    _require(p.shape == (n, 6) and n > 0, "Complete sealed native six-path focals")
    columns = arrays["query_column"]
    _require(columns.shape == (n,) and columns.dtype.kind in "iu" and
             ((columns >= 0) & (columns < 686)).all(),
             "Native focal query columns outside original full686")
    for name in ("site_row", "positions_bp", "cohort",
                 "query_sample_indices", "query_node_ids"):
        _require(arrays[name].shape == (n,), "Native focal axis differs: " + name)
    _require(arrays["site_row"].dtype.kind in "iu" and
             (arrays["site_row"] >= 0).all(), "Native SNP row indexes required")
    _require(np.array_equal(arrays["query_sample_indices"],
                           np.asarray(parent["query_slots"])[columns]) and
             np.array_equal(arrays["query_node_ids"],
                            meta["AP_query_node_ids"][columns]) and
             np.array_equal(arrays["cohort"],
                            np.asarray(parent["cohorts"])[columns]),
             "Native focal sampled identity/cohort correspondence differs")
    positions = arrays["positions_bp"]
    left, right = meta["coverage_original"]
    _require(np.isfinite(positions).all() and
             ((positions >= left) & (positions < right)).all() and
             ((positions >= 0) & (positions < 2_000_000)).all(),
             "Native focal original SNP coordinates outside observed coverage")
    reports = manifest["native_focal_sampler_reports"]
    _require(len(reports) == 5 and
             {r["cohort"] for r in reports} == set(COHORT_COUNTS),
             "All five native focal sampler reports required")
    counts = Counter(arrays["cohort"].tolist())
    _require(all(counts.get(r["cohort"], 0) == r["realized"] for r in reports),
             "Sealed native focal counts differ from sampler reports")
    # Repeated source focal pairs are retained exactly. No deduplication.
    return arrays


def _native6_labels(reader, binding, parent, target, focal):
    """Small full686 six-path adapter, callable only after the all-eight gate."""
    from arggnn.ancestralpaths_native_bridge import PATH_TO_SOURCE_CLASS
    arrays = _npz(reader, binding["truth_npz"], TRUTH_FIELDS)
    paths = arrays["native_paths_1based"]
    _require(np.array_equal(arrays["intervals_bp"], target.intervals) and
             np.array_equal(arrays["query_raw_node_ids"], parent["raw_query_nodes"]) and
             np.array_equal(arrays["source_classes_0based"], target.c4) and
             paths.shape == target.c4.shape and paths.dtype.kind in "iu" and
             ((paths >= 1) & (paths <= 6)).all(),
             "Complete external six-path/C4 target join differs")
    _require(np.array_equal(np.asarray(PATH_TO_SOURCE_CLASS)[paths - 1], target.c4),
             "External native6/C4 ontology differs")
    # Right-sided membership preserves the producer's half-open intervals.
    rows = np.searchsorted(target.intervals[:, 1],
                           focal["positions_bp"], side="right")
    return paths[rows, focal["query_column"]]


def score_root_v1(*, completed_ref, parents, bindings, reader, helpers,
                  on_parent=None, on_stage=None):
    """Root entry: primary H038 C4 + separate native6 focals, no model execution.

    reader must SHA-verify exact project-local records. parents are root-verified
    H038 scoring objects; bindings are VAL-only original external target refs.
    on_parent, if supplied by root, must be its guarded compact-JSON writer.
    """
    from arggnn.mutation_routed_full_panel_primary_v1 import (
        FullPanelExternalTargetV1, PRIMARY, means_v1,
    )
    from arggnn.mutation_routed_fast_primary_v1 import (
        mass_cache_v1, score_primary_v1,
    )
    from arggnn.expanded_native_ancestralpaths_v1 import native_focal_metrics_v1

    _require(completed_ref["path"] ==
             "data/expanded_native_ancestralpaths_v2/completed.json",
             "Exact completed AP run required")
    completed = _json(reader, completed_ref)
    _require(completed["schema"] == "expanded_native_ap_driver_completed_v2" and
             completed["TRAIN_count"] == 32 and completed["VAL_count"] == 8 and
             completed["epochs"] == 5 and
             completed["checkpoint_selection"] == "final_epoch5" and
             completed["VAL_selection"] is False and
             completed["VAL_targets_read"] is False and
             completed["outer_included"] is False,
             "Completed fixed-epoch native AP run required")
    _require(set(parents) == set(bindings) == set(VAL_IDS) ==
             set(completed["VAL_prediction_seals"]),
             "Exactly all eight registered internal VAL parents required")
    cfg = _json(reader, completed["config_ref"])
    config_digest = _digest(cfg)
    fit = _json(reader, completed["fit_ref"])
    _require(fit["schema"] == "expanded_native_ap_fit_v1" and
             fit["source_ref"] == cfg["source_ref"] and
             fit["config_digest"] == config_digest and fit["epochs"] == 5 and
             fit["VAL_selection"] is False and
             fit["checkpoint_selection"] == "final_epoch5" and
             fit["training_rows"] == completed["realized_training_rows"],
             "Completed native fit/checkpoint lineage differs")

    manifests, focal_arrays = {}, {}
    all_manifest_refs = [completed["VAL_prediction_seals"][p] for p in VAL_IDS]
    # Phase 1: every C4 AND native6 product is verified before any target opens.
    # Keep only small native focal arrays, not all eight dense C4 grids.
    for pid in VAL_IDS:
        keys = _check_parent(parents[pid], bindings[pid], reader)
        manifest = _json(reader, completed["VAL_prediction_seals"][pid])
        _require(manifest["schema"] == "expanded_native_ap_prediction_seal_v1" and
                 manifest["parent_id"] == pid and manifest["split"] == "val" and
                 manifest["source_ref"] == cfg["source_ref"] and
                 manifest["config_digest"] == config_digest and
                 manifest["fit_ref"] == completed["fit_ref"] and
                 manifest["checkpoint_ref"] == fit["checkpoint_ref"] and
                 manifest["query_keys"] == keys and
                 manifest["query_haplotypes"] == 686 and manifest["BAA_queries"] == 14 and
                 manifest["C4_class_order"] == ["Ana", "CHG", "WHG", "EHG"] and
                 manifest["native_path_axis_1based"] == list(range(1, 7)) and
                 manifest["C4_collapse_before_argmax"] == "p1+p5,p2+p6,p3,p4" and
                 manifest["native_six_argmax_unmasked"] is True and
                 manifest["targets_read"] is False and manifest["VAL_selection"] is False and
                 manifest["outer_included"] is False and
                 manifest["calibration_fitted"] is False,
                 "AP sealed prediction method/query/class lineage differs")
        p4, meta = _common(reader, manifest, parents[pid])
        focal_arrays[pid] = _focal(reader, manifest, parents[pid], meta)
        manifests[pid] = manifest
        del p4, meta
        if on_stage is not None:
            on_stage("AP_seal_preflight_parent_verified", pid)
    if on_stage is not None:
        on_stage("all_eight_AP_C4_native6_seals_verified_before_truth")

    cells = []
    for pid in VAL_IDS:
        parent, binding = parents[pid], bindings[pid]
        p4, _ = _common(reader, manifests[pid], parent)
        target = FullPanelExternalTargetV1(
            parent, binding, "VAL_after_sealed_predictions", reader,
            sealed_records=all_manifest_refs,
        )
        # This is precisely H038's FP64 mass construction -> FP32 primary cache.
        cache = mass_cache_v1(
            parent, target.target_mass(parent, helpers),
            target_provenance=target.provenance, expected_present_queries=182,
        )
        group = score_primary_v1(p4, cache, backend="numpy")
        focal = focal_arrays[pid]
        labels = _native6_labels(reader, binding, parent, target, focal)
        native6 = native_focal_metrics_v1(
            focal["native_probabilities"], labels, cohorts=focal["cohort"],
        )
        cell = jsonable_v1({
            "parent_id": pid, "split": "val",
            "prediction_seal": completed["VAL_prediction_seals"][pid],
            "common_C4_ref": manifests[pid]["common_C4_ref"],
            "native6_focal_ref": manifests[pid]["native6_focal_ref"],
            "target_provenance": target.provenance,
            "groups": {PRIMARY: group},
            "native6_SNP_focal": native6,
            "native_focal_sampler_reports": manifests[pid]["native_focal_sampler_reports"],
            "checkpoint_selection": "fixed_final_epoch5",
            "development_only": True, "outer_parents_accessed": 0,
        })
        cells.append(cell)
        if on_parent is not None:
            on_parent(pid, cell)
        del p4, target, cache, labels

    c4_means = means_v1(cells)
    n6 = [cell["native6_SNP_focal"] for cell in cells]
    native6_mean = {
        "accuracy_equal_parent": float(np.mean(
            [row["accuracy_fraction"] for row in n6], dtype=np.float64)),
        "aggregation": "unweighted_mean_of_eight_parent_native_focal_accuracies",
        "common_physical_C4_metric": False,
        "by_cohort": {},
    }
    for cohort in COHORT_COUNTS:
        present = [row["by_cohort"][cohort] for row in n6
                   if cohort in row.get("by_cohort", {})]
        native6_mean["by_cohort"][cohort] = {
            "parents_with_examples": len(present),
            "accuracy_equal_parent": float(np.mean(
                [row["accuracy_fraction"] for row in present], dtype=np.float64))
                if len(present) == 8 else None,
        }
    return jsonable_v1({
        "schema": "expanded_native_AP_root_score_draft_v1",
        "completed_ref": completed_ref,
        "VAL_parents": 8, "TEST_or_outer_parents": 0,
        "training_rows": completed["realized_training_rows"],
        "sampler_shortfall_rows": completed["sampler_shortfall_rows"],
        "checkpoint_selection": "fixed_final_epoch5",
        "C4_primary_equal_parent": c4_means,
        "native6_SNP_focal_equal_parent": native6_mean,
        "per_parent": cells,
        "all_eight_C4_and_native6_seals_verified_before_truth": True,
        "C4_arithmetic": "same_H038_FP32_primary_scorer",
        "C4_probability_collapse": "existing_sealed_p1+p5,p2+p6,p3,p4",
        "native6_scope": "sealed_author_SNP_focals; unweighted; no_grid_reconstruction",
        "retraining_or_new_prediction": False, "calibration": False,
        "probability_floor": False, "posthoc_renormalization": False,
        "helper_validation_before_root_run": "syntax_only; no reviewer data execution",
    })

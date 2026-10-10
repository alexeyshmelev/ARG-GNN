#!/usr/bin/env python
"""External exact-support VAL scoring of frozen native ARGformer C4 readouts."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SCRIPT = ROOT / "scripts/score_official_argformer_biological_v1.py"
CONFIG = ROOT / "configs/official_argformer_biological_readout_v1.json"
PARENTS = ("bio_paths_pilot_train_000", "bio_paths_pilot_train_001",
           "bio_paths_pilot_val_000", "bio_paths_pilot_val_001")
VAL = PARENTS[2:]
NATIVE_CLASSES = ["ANA", "CHG", "WHG", "EHG"]
OBSERVED_CLASSES = ["Ana", "CHG", "WHG", "EHG"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_string(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def local(value, observed=False):
    p = Path(value)
    if ".." in p.parts or (observed and any(
            "truth" in part.lower() or "oracle" in part.lower() for part in p.parts)):
        raise ValueError("Traversal or truth/oracle input path forbidden")
    p = p if p.is_absolute() else ROOT / p
    if p.resolve() != p or not p.is_relative_to(ROOT):
        raise ValueError("Canonical project-local path required")
    return p


def bound(record, observed=False):
    if not isinstance(record, dict) or not digest_string(record.get("sha256")):
        raise ValueError("Explicit path and full SHA256 binding required")
    p = local(record["path"], observed)
    if sha(p) != record["sha256"]:
        raise ValueError("Bound artifact changed: " + str(p))
    return p


def jsonable(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Unrepresented nonfinite JSON value")
    return value


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(jsonable(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def source_policy(expected):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Explicitly empty CUDA visibility required")
    if Path.cwd().resolve() != ROOT or Path(__file__).resolve() != SCRIPT:
        raise RuntimeError("Dedicated root and fixed scoring source required")
    if not digest_string(expected) or sha(CONFIG) != expected:
        raise ValueError("Root-reviewed readout policy SHA256 required")
    cfg = json.loads(CONFIG.read_text())
    if (cfg["kind"] != "official_argformer_biological_readout_v1"
            or cfg["implementation_sha256"] != sha(SCRIPT)
            or cfg["parent_ids"] != list(PARENTS) or cfg["VAL_parent_ids"] != list(VAL)
            or cfg["native_class_order"] != NATIVE_CLASSES
            or cfg["query_count"] != 120 or cfg["primary_query_count"] != 64
            or cfg["secondary_query_count"] != 56
            or cfg["normalization_length_bp"] != 2000000
            or cfg["probability_max_absolute_error"] != 1e-6
            or cfg["probability_floor"] is not False or cfg["posthoc_renormalization"] is not False
            or cfg["calibration_fit"] is not False or cfg["inference_or_training"] is not False
            or cfg["all_two_VAL_exports_sealed_before_truth"] is not True):
        raise ValueError("Frozen C4 scoring policy differs")
    for name, expected_sha in cfg["frozen_files_sha256"].items():
        bound({"path": name, "sha256": expected_sha}, observed=True)
    candidate = json.loads(local(cfg["common_protocol_source_config"], True).read_text())
    if any(cfg[key] != candidate[key] for key in
           ("metrics", "relate_producer", "observed_roster_sha256")):
        raise ValueError("Common exact-support metrics/observation contract changed")
    if (cfg["external_truth_manifest"]["path"] != candidate["external_truth_manifest"]["path"]
            or cfg["external_truth_manifest"]["sha256"] != candidate["external_truth_manifest"]["sha256"]):
        raise ValueError("Frozen external target manifest binding changed")
    native = json.loads(local(cfg["native_fit_config"], True).read_text())
    if (native["class_order"] != NATIVE_CLASSES or native["source_unchanged"] is not True
            or native["query_ancestry_targets_read"] is not False
            or native["readout"]["temperature"] != .3 or native["readout"]["calibration_fitted"] is not False
            or native["readout"]["tail_extrapolation"] is not False):
        raise ValueError("Native C4 class/readout/source identity differs")
    sys.path.insert(0, str(ROOT / "src"))
    path = local(cfg["metric_helper"], True)
    spec = importlib.util.spec_from_file_location("_frozen_ARGformer_external_AP_metrics", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)  # Frozen numpy/observed bridge; no protocol or truth call.
    from arggnn.ancestralpaths_native_bridge import SOURCE_CLASS_NAMES
    if list(SOURCE_CLASS_NAMES) != OBSERVED_CLASSES or [c.upper() for c in SOURCE_CLASS_NAMES] != NATIVE_CLASSES:
        raise ValueError("Frozen external C4 source-class index order differs")
    return cfg, helper


def observed_parents(cfg, manifest):
    import numpy as np
    spec = json.loads(Path(manifest).read_text())
    entries = {r["parent_id"]: r for r in spec["replicates"]}
    if len(spec["replicates"]) != 4 or set(entries) != set(PARENTS):
        raise ValueError("Exactly four common observed manifest entries required")
    rows = []
    for name in PARENTS:
        directory = local("data/biological_paths_execution_pilot_v1/" + name, True)
        roster = bound({"path": str((directory / "observed/sample_manifest.json").relative_to(ROOT)),
                        "sha256": cfg["observed_roster_sha256"][name]}, True)
        samples = json.loads(roster.read_text())
        if (len(samples) != 280 or [r["sample_index"] for r in samples] != list(range(280))):
            raise ValueError("Ordered observed280 roster required")
        for i in range(0, 280, 2):
            a, b = samples[i:i + 2]
            if (a["sample_id"] != b["sample_id"] or (a["haplotype"], b["haplotype"]) != (0, 1)
                    or any(a[k] != b[k] for k in ("role", "observed_cohort", "age_generations"))):
                raise ValueError("Observed diploid subject/haplotype identity differs")
        if len({samples[i]["sample_id"] for i in range(0, 280, 2)}) != 140:
            raise ValueError("Explicit unique observed diploid subject IDs required")
        query = np.array([i for i, r in enumerate(samples) if r["role"] == "query"], dtype=np.int64)
        refs = np.array([i for i, r in enumerate(samples) if r["role"] == "pure_source_reference"])
        mixed = [r for r in samples if r["role"] == "mixed_reference_only"]
        cohorts = np.array([samples[i]["observed_cohort"] for i in query])
        if (len(query), len(refs), len(mixed), int(np.sum(cohorts == "present_day"))) != (120, 96, 64, 64):
            raise ValueError("Require Q120/R96/mixed64 and present64/ancient56")
        for sample in samples:
            c = sample["observed_reference_class"]
            if sample["role"] == "pure_source_reference":
                if type(c) is not int or c not in range(4) or sample["observed_cohort"] != OBSERVED_CLASSES[c]:
                    raise ValueError("Observed source class/cohort order differs")
            elif c is not None:
                raise ValueError("Query/mixed ancestry class must remain null")
        if [sum(samples[i]["observed_reference_class"] == c for i in refs) for c in range(4)] != [24] * 4:
            raise ValueError("Exactly24 pure observed references per C4 class required")
        grid = entries[name]["coordinate_contract"]["fixed_grid"]
        grid_path = bound(grid, True)
        if grid_path != directory / "observed/scoring_coordinates.npz":
            raise ValueError("Original observed coordinate-grid path differs")
        with np.load(grid_path, allow_pickle=False) as z:
            if (not np.array_equal(z["positions_bp"], np.arange(50, 2000000, 100, dtype=np.float64))
                    or not np.array_equal(z["bin_edges_bp"], np.arange(0, 2000001, 100, dtype=np.float64))):
                raise ValueError("Original complete physical100bp grid differs")
        rows.append(dict(parent_id=name, split="val" if name in VAL else "train",
                         directory=directory, samples=samples, query_slots=query, reference_slots=refs,
                         cohorts=cohorts, raw_query_nodes=np.array([samples[i]["observed_node_id"] for i in query]),
                         positions=np.arange(50, 2000000, 100, dtype=np.float64),
                         edges=np.arange(0, 2000001, 100, dtype=np.float64),
                         row={"files": {"observed/sample_manifest.json": {"sha256": sha(roster)},
                                        "observed/scoring_coordinates.npz": {"sha256": grid["sha256"]}}}))
    return rows


def fit_and_preparation(name, records, parent, obs, cfg):
    import numpy as np
    if set(records) != {"prepared_manifest", "fit_receipt"}:
        raise ValueError("Each parent requires exactly one observed manifest and completed-fit receipt")
    prepared_path = bound(records["prepared_manifest"], True)
    prepared_dir = prepared_path.parent
    if prepared_path.name != "prepared_manifest.json":
        raise ValueError("Explicit native prepared-manifest filename required")
    prepared = json.loads(prepared_path.read_text())
    entry = obs["entry"]
    if (prepared["schema"] != "official_argformer_prepared_observed_arg_v1"
            or prepared["status"] != "prepared_observed_paths_only" or prepared["parent_id"] != name
            or prepared["parent_split"] != parent["split"] or prepared["policy_sha256"] != cfg["native_fit_config_sha256"]
            or prepared["input_manifest_sha256"] != cfg["_observed_manifest_sha256"]
            or prepared["true_query_ancestry_read"] is not False or prepared["within_ARG_transductive"] is not True
            or prepared["class_order"] != NATIVE_CLASSES or prepared["coordinate_contract"] != entry["coordinate_contract"]
            or prepared["namespace"] != name + ":" + entry["inferred_tree"]["sha256"]
            or prepared["inferred_tree_sha256"] != entry["inferred_tree"]["sha256"]
            or prepared["derived_inferred_roster_sha256"] != entry["observed_sample_manifest"]["sha256"]
            or prepared["source_observed_roster_sha256"] != cfg["observed_roster_sha256"][name]
            or prepared["completion_provenance_sha256"] != entry["completion_provenance"]["sha256"]
            or prepared["grid_sha256"] != entry["coordinate_contract"]["fixed_grid"]["sha256"]):
        raise ValueError("Native prepared manifest differs from exact observed parent")
    paths = {}
    for relative in ("observed_roster.json", "observed_parent_entry.json", "coordinate_path_joins.npz"):
        paths[relative] = bound({"path": str(prepared_dir / relative),
                               "sha256": prepared["files_sha256"][relative]}, True)
    if (json.loads(paths["observed_roster.json"].read_text()) != json.loads(obs["derived_roster_path"].read_text())
            or json.loads(paths["observed_parent_entry.json"].read_text()) != entry):
        raise ValueError("Prepared observed subject roster/parent provenance differs")
    with np.load(paths["coordinate_path_joins.npz"], allow_pickle=False) as z:
        joins = {key: z[key].copy() for key in
                 ("grid_indexes", "original_positions_bp", "local_positions_bp", "query_nodes",
                  "reference_nodes", "reference_classes", "query_path_ids", "reference_path_ids")}
    checks = (("grid_indexes", parent["retained_indexes"]), ("original_positions_bp", parent["positions"]),
              ("local_positions_bp", parent["positions"] - obs["offset"]),
              ("query_nodes", obs["nodes"][parent["query_slots"]]),
              ("reference_nodes", obs["nodes"][parent["reference_slots"]]),
              ("reference_classes", np.array([parent["samples"][i]["observed_reference_class"]
                                              for i in parent["reference_slots"]])))
    if any(joins[key].dtype.kind not in "iu" for key in
           ("grid_indexes", "query_nodes", "reference_nodes", "reference_classes")):
        raise ValueError("Typed integer grid/query/reference/class joins required")
    if any(not np.array_equal(joins[key], expected) for key, expected in checks):
        raise ValueError("Native prepared exact grid/query/reference/class join differs")
    for field, width in (("query_path_ids", 120), ("reference_path_ids", 96)):
        values = joins[field]
        if (values.shape != (len(parent["positions"]), width) or values.dtype.kind not in "iu"
                or np.any(values < 0) or np.any(values >= prepared["unique_full_paths"])):
            raise ValueError("Native observed path support differs")
    receipt_path = bound(records["fit_receipt"], True)
    if receipt_path.name != "receipt.json":
        raise ValueError("Completed native fit receipt required")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"] != "passed" or receipt["exit_code"] != 0 or receipt["action"] != "fit"
            or receipt["root_actual_execution"] is not True or receipt["true_query_ancestry_read"] is not False
            or receipt["parent_id"] != name or receipt["namespace"] != prepared["namespace"]
            or receipt["implementation_sha256"] != cfg["native_fit_implementation_sha256"]
            or receipt["config_sha256"] != cfg["native_fit_config_sha256"]
            or receipt["prepared_manifest_sha256"] != records["prepared_manifest"]["sha256"]
            or receipt["fitting_regime"] != "within_ARG_transductive"
            or receipt["runtime"].get("actual_cuda") is not True
            or receipt["published_native_task_reproduction"] is not False
            or receipt["classifier_unchanged_unused"] is not True):
        raise ValueError("Completed genuine native within-ARG fit identity required")
    artifacts = receipt["artifacts_sha256"]
    for phase, epochs in (("pretraining", 5), ("contrastive", 1)):
        record = receipt[phase]
        if (record["mode"] != ("mlm" if phase == "pretraining" else "contrastive")
                or record["execution"] != "full_batch" or record["completed_epochs"] != epochs
                or type(record["actual_optimizer_updates"]) is not int or record["actual_optimizer_updates"] <= 0
                or not digest_string(record["initial_model_state_sha256"])
                or not digest_string(record["final_model_state_sha256"])
                or record["initial_model_state_sha256"] == record["final_model_state_sha256"]
                or record["true_query_ancestry_read"] is not False):
            raise ValueError("Unchanged source MLM5/contrastive1 actual optimizer fits required")
        directory = "mlm" if phase == "pretraining" else "contrastive"
        if artifacts[directory + "/composer_final.pt"] != record["checkpoint_sha256"]:
            raise ValueError("Phase checkpoint provenance differs")
        phase_path = bound({"path": str(receipt_path.parent / directory / "phase_receipt.json"),
                            "sha256": artifacts[directory + "/phase_receipt.json"]}, True)
        if json.loads(phase_path.read_text()) != record:
            raise ValueError("Native phase receipt differs")
    initialized = receipt["encoder_initialization"]
    if (initialized["missing_encoder"] or initialized["unequal_encoder"]
            or not initialized["all_encoder_keys_loaded_equal"]
            or initialized["checkpoint_sha256"] != receipt["pretraining"]["checkpoint_sha256"]
            or artifacts["contrastive_source_compatible.pt"] != receipt["source_compatible_checkpoint_sha256"]):
        raise ValueError("Genuine trained-encoder initializer/checkpoint identity differs")
    readout = receipt["readout"]
    if (readout["class_order"] != NATIVE_CLASSES or readout["readout_temperature"] != .3
            or readout["native_classifier_used"] is not False or readout["downstream_adapter"] is not True
            or readout["no_tail_extrapolation"] is not True or readout["normalization_length_bp"] != 2000000):
        raise ValueError("Native uncalibrated direct C4 readout identity differs")
    for field in ("downstream_logits_sha256", "uncalibrated_probabilities_sha256"):
        if not digest_string(readout[field]):
            raise ValueError("Native readout tensor hashes required")
    prediction = {"path": str(receipt_path.parent / "downstream_C4_local_readout.npz"),
                  "sha256": artifacts["downstream_C4_local_readout.npz"]}
    if not digest_string(prediction["sha256"]):
        raise ValueError("Completed native prediction bytes must be receipt-bound")
    if name in VAL:
        bound(prediction, True)  # Hash only during preflight, no score-array parsing.
    return dict(records=records, prepared=prepared, receipt=receipt, joins=joins, prediction=prediction)


def validate_inputs(cfg, helper, binding_record):
    import numpy as np
    binding_path = bound(binding_record, True)
    bindings = json.loads(binding_path.read_text())
    if (bindings["kind"] != "official_argformer_biological_readout_inputs_v1"
            or set(bindings["parents"]) != set(PARENTS)
            or set(bindings) != {"kind", "observed_manifest", "parents"}):
        raise ValueError("Exactly four explicit prepared/completed-fit bindings required")
    observed = bound(bindings["observed_manifest"], True)
    if observed != local(cfg["observed_manifest_path"], True):
        raise ValueError("Completed common observed manifest path differs")
    cfg = dict(cfg, _observed_manifest_sha256=bindings["observed_manifest"]["sha256"])
    parents = observed_parents(cfg, observed)
    # Frozen observations validates all four exact producer/roster/grid/physical joins.
    observations = helper.observations(observed, parents, cfg)
    states = {}
    for parent in parents:
        obs = observations[parent["parent_id"]]
        parent.update(positions=obs["positions"], edges=obs["bin_edges"],
                      retained_indexes=obs["retained_indexes"], coverage_original=obs["coverage"],
                      observation=obs)
        if (not np.array_equal(parent["positions"], 50 + 100 * parent["retained_indexes"])
                or len(parent["positions"]) != len(parent["edges"]) - 1):
            raise ValueError("Exact original100bp centers/clipped bins differ")
        states[parent["parent_id"]] = fit_and_preparation(
            parent["parent_id"], bindings["parents"][parent["parent_id"]], parent, obs, cfg)
    return cfg, {p["parent_id"]: p for p in parents}, states


def seal_native(parent, state, cfg, output):
    import numpy as np
    path = bound(state["prediction"], True)
    with np.load(path, allow_pickle=False) as z:
        logits, probabilities = z["logits"].copy(), z["probabilities"].copy()
        mapping = {"grid_indexes": "grid_indexes", "positions_bp": "original_positions_bp",
                   "local_positions_bp": "local_positions_bp", "query_nodes": "query_nodes",
                   "query_path_ids": "query_path_ids", "reference_nodes": "reference_nodes",
                   "reference_path_ids": "reference_path_ids"}
        for raw, prepared in mapping.items():
            if not np.array_equal(z[raw], state["joins"][prepared]):
                raise ValueError("Native readout differs from observed support/subject join: " + raw)
    shape = (len(parent["positions"]), 120, 4)
    if (logits.shape != shape or probabilities.shape != shape
            or logits.dtype != np.float32 or probabilities.dtype != np.float32
            or not np.isfinite(logits).all() or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(-1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Finite raw native FP32 C4 probabilities/logits required")
    for array, field in ((logits, "downstream_logits_sha256"),
                         (probabilities, "uncalibrated_probabilities_sha256")):
        if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != state["receipt"]["readout"][field]:
            raise ValueError("Native tensor bytes differ from fit receipt")
    value = logits.astype(np.float64)
    shifted = value - value.max(-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    stable = np.exp(logp)
    error = float(np.max(np.abs(stable - probabilities.astype(np.float64))))
    if (not np.isfinite(logp).all() or error > cfg["probability_max_absolute_error"]
            or not np.allclose(stable.sum(-1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Auxiliary stable C4 log probability parity/normalization failed")
    destination = output / (parent["parent_id"] + "_sealed_native_C4.npz")
    query = parent["query_slots"]
    with destination.open("xb") as stream:
        np.savez_compressed(stream, C4_probabilities_raw_float32=probabilities, C4_logits_float32=logits,
            C4_log_probabilities_auxiliary_float64=logp, positions_bp=parent["positions"],
            bin_edges_bp=parent["edges"], retained_fixed100bp_indexes=parent["retained_indexes"],
            coverage_original_bp0=parent["coverage_original"], normalization_length_bp=np.asarray(2000000),
            local_to_original_offset_bp=np.asarray(parent["observation"]["offset"]),
            query_sample_indexes=query, query_node_ids=state["joins"]["query_nodes"],
            query_raw_node_ids=parent["raw_query_nodes"],
            query_subject_ids=np.asarray([parent["samples"][i]["sample_id"] for i in query]),
            query_haplotypes=np.asarray([parent["samples"][i]["haplotype"] for i in query]),
            class_order=np.asarray(NATIVE_CLASSES), **{
                field: state["joins"][field] for field in
                ("query_path_ids", "reference_nodes", "reference_classes", "reference_path_ids")})
    record = {"parent_id": parent["parent_id"], "split": "val",
              "path": str(destination.relative_to(ROOT)), "sha256": sha(destination),
              "source_prediction": state["prediction"], "completed_fit_receipt": state["records"]["fit_receipt"],
              "prepared_observed_manifest": state["records"]["prepared_manifest"],
              "probabilities_preserved_without_floor_or_renormalization": True,
              "sealed_before_external_truth": True, "native_class_order": NATIVE_CLASSES,
              "stable_probability_max_absolute_error": error, "raw_probability_zero_count": int((probabilities == 0).sum()),
              "auxiliary_exp_logp_zero_count": int((stable == 0).sum()),
              "auxiliary_argmax_disagreements": int((stable.argmax(-1) != probabilities.argmax(-1)).sum())}
    write_new(destination.with_suffix(".seal.json"), record)
    return record


class ExternalVALC4Truth:
    """Full-domain external C4 labels admitted only after both VAL exports are sealed."""
    def __init__(self, parent, cfg, aggregate):
        import numpy as np
        seals = json.loads(bound(aggregate).read_text())
        if (parent["split"] != "val" or seals["status"] != "both_VAL_predictions_sealed_before_truth"
                or [r["parent_id"] for r in seals["predictions"]] != list(VAL)):
            raise ValueError("Both exact VAL prediction seals required before truth")
        for record in seals["predictions"]:
            bound(record)
        # The external manifest itself is first opened after the two prediction seals.
        metadata = json.loads(bound(cfg["external_truth_manifest"]).read_text())
        rows = [r for r in metadata["replicates"] if r["parent_id"] == parent["parent_id"]]
        if len(rows) != 1 or rows[0]["split"] != "val":
            raise ValueError("External truth parent/split differs")
        relative = "truth/full_truth_intervals.npz"
        path = parent["directory"] / relative
        record = {"path": str(path.relative_to(ROOT)), "sha256": rows[0]["files"][relative]["sha256"]}
        with np.load(bound(record), allow_pickle=False) as z:
            intervals = z["intervals_bp"].copy()
            classes = z["source_classes_0based"].copy()
            query_ids = z["query_raw_node_ids"].copy()
        if (intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals)
                or not np.isfinite(intervals).all() or intervals[0, 0] != 0 or intervals[-1, 1] != 2000000
                or np.any(intervals[:, 1] <= intervals[:, 0])
                or not np.array_equal(intervals[:-1, 1], intervals[1:, 0])
                or classes.shape != (len(intervals), 120) or classes.dtype.kind not in "iu"
                or np.any((classes < 0) | (classes > 3))
                or not np.array_equal(query_ids, parent["raw_query_nodes"])):
            raise ValueError("Complete external C4 truth/query identity invalid")
        self.intervals, self.c4 = intervals, classes
        self.provenance = dict(record, purpose="external_VAL_after_both_sealed_native_predictions",
                               target_manifest=cfg["external_truth_manifest"])


def auxiliary_log_nll(logp, parent, truth, columns, helper):
    import numpy as np
    bins, rows, spans = helper.intersections(parent["edges"], truth.intervals)
    total = float(spans.sum() * len(columns))
    value = 0.
    for start in range(0, len(bins), 1024):
        cut = slice(start, start + 1024)
        labels = truth.c4[rows[cut]][:, columns]
        selected = np.take_along_axis(logp[bins[cut]][:, columns], labels[..., None], -1)[..., 0]
        value += float(np.sum(spans[cut, None] * -selected))
    if total <= 0 or not math.isfinite(value):
        raise ValueError("Auxiliary direct-log NLL is nonfinite/unsupported")
    return {"C4_nll_from_direct_logp": value / total, "denominator_query_bp": total,
            "condition": "auxiliary float64 logsumexp of unchanged native C4 logits",
            "probability_floor_used": False, "calibration_fitted": False,
            "raw_probability_metrics_retained": True}


def score_one(parent, seal, cfg, helper, aggregate):
    import numpy as np
    with np.load(bound(seal), allow_pickle=False) as z:
        probabilities = z["C4_probabilities_raw_float32"].copy()
        logp = z["C4_log_probabilities_auxiliary_float64"].copy()
        for key, expected in (("positions_bp", parent["positions"]), ("bin_edges_bp", parent["edges"]),
                              ("query_sample_indexes", parent["query_slots"]),
                              ("query_raw_node_ids", parent["raw_query_nodes"])):
            if not np.array_equal(z[key], expected):
                raise ValueError("Sealed scoring support/query identity differs")
    truth = ExternalVALC4Truth(parent, cfg, aggregate)
    groups = {"primary_present_day_64": np.flatnonzero(parent["cohorts"] == "present_day"),
              "secondary_ancient_56": np.flatnonzero(parent["cohorts"] != "present_day"),
              "all_queries_120": np.arange(120)}
    raw, auxiliary = {}, {}
    for name, columns in groups.items():
        item = helper.full_span_score(probabilities, parent, truth, columns, cfg["metrics"])
        item["nll_is_infinite"] = item["infinite_nll_query_bp"] > 0
        item["nll_null_representation"] = "positive exact truth mass at raw probability zero" if item["nll_is_infinite"] else None
        log_item = auxiliary_log_nll(logp, parent, truth, columns, helper)
        if item["denominator_query_bp"] != log_item["denominator_query_bp"]:
            raise ValueError("Raw/auxiliary exact score denominator differs")
        raw[name], auxiliary[name] = item, log_item
    return {"parent_id": parent["parent_id"], "split": "val", "native_class_order": NATIVE_CLASSES,
            "sealed_prediction": seal, "external_truth_provenance": truth.provenance,
            "native_C4_raw_exact_support_metrics": raw, "auxiliary_direct_log_C4_NLL": auxiliary,
            "fitting_regime": "within_ARG_transductive; native MLM and pure-reference contrastive labels",
            "same_common_scoring_support": True, "identical_supervision_claim": False,
            "published_native_task_reproduction": False, "final_holdout_evaluation": False,
            "calibration_fitted": False, "native_classifier_used": False, "no_tail_extrapolation": True}


def run(cfg, helper, binding, parents, states, output, config_sha):
    output.mkdir(exist_ok=False)
    receipt = {"kind": "official_argformer_biological_external_VAL_readout_v1", "status": "running",
               "implementation_sha256": sha(SCRIPT), "config_sha256": config_sha,
               "root_inputs": binding, "predictions": [], "metrics": [], "external_truth_inputs": [],
               "all_four_completed_native_fits_required": True, "training": False, "inference": False,
               "calibration_fit": False, "original_native_outputs_modified": False,
               "published_native_task_reproduction": False, "identical_supervision_claim": False,
               "final_holdout_evaluation": False}
    start, code = time.monotonic(), 1
    try:
        for name in VAL:
            receipt["predictions"].append(seal_native(parents[name], states[name], cfg, output))
        aggregate_path = output / "both_VAL_predictions_sealed_before_truth.json"
        write_new(aggregate_path, {"status": "both_VAL_predictions_sealed_before_truth",
                                  "predictions": receipt["predictions"], "root_inputs": binding})
        aggregate = {"path": str(aggregate_path.relative_to(ROOT)), "sha256": sha(aggregate_path)}
        receipt["aggregate_prediction_seal"] = aggregate
        for seal in receipt["predictions"]:
            result = score_one(parents[seal["parent_id"]], seal, cfg, helper, aggregate)
            path = output / (seal["parent_id"] + "_external_VAL_metrics.json")
            write_new(path, result)
            receipt["external_truth_inputs"].append(result["external_truth_provenance"])
            receipt["metrics"].append({"parent_id": seal["parent_id"], "path": str(path.relative_to(ROOT)),
                                       "sha256": sha(path)})
        final_cfg, final_helper = source_policy(config_sha)
        validate_inputs(final_cfg, final_helper, binding)  # Rebind original observations, fits and VAL files.
        bound(cfg["external_truth_manifest"])
        for record in receipt["external_truth_inputs"]:
            bound(record)
        receipt.update(status="passed", exit_code=0)
        code = 0
    except BaseException as error:
        receipt.update(status="failed", exit_code=1,
                       failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - start
        write_new(output / "receipt.json", receipt)
        with (output / "exit_status").open("x") as stream:
            stream.write(str(code) + "\n")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--expected-bindings-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", default="experiments/results/official_argformer_biological_readout_v1")
    args = parser.parse_args()
    cfg, helper = source_policy(args.expected_config_sha256)
    binding = {"path": args.bindings, "sha256": args.expected_bindings_sha256}
    cfg, parents, states = validate_inputs(cfg, helper, binding)
    if not args.execute:
        print(json.dumps({"status": "all_four_completed_fit_and_observed_bindings_checked_no_scoring",
                          "config_sha256": args.expected_config_sha256,
                          "truth_or_prediction_arrays_read": False, "models_loaded": False}))
        return 0
    output = local(args.output, True)
    protected = [local(r[key]["path"], True).parent for state in states.values()
                 for r in [state["records"]] for key in ("fit_receipt", "prepared_manifest")]
    if output.exists() or not output.parent.is_dir() or any(output.is_relative_to(p) for p in protected):
        raise ValueError("Fresh separate scoring output required")
    return run(cfg, helper, binding, parents, states, output, args.expected_config_sha256)


if __name__ == "__main__":
    raise SystemExit(main())

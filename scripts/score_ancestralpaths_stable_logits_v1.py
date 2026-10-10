#!/usr/bin/env python
"""Auxiliary stable-logit readout of frozen APv2 checkpoints; never trains."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
CONFIG = ROOT / "configs/ancestralpaths_stable_readout_v1.json"
SCRIPT = ROOT / "scripts/score_ancestralpaths_stable_logits_v1.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local(value):
    lexical = Path(value)
    if ".." in lexical.parts:
        raise ValueError("Traversal forbidden")
    path = lexical if lexical.is_absolute() else ROOT / lexical
    if path.resolve() != path or not path.is_relative_to(ROOT):
        raise ValueError("Canonical project-local path required")
    return path


def bound(record):
    path = local(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError("Integrity-bound artifact changed: " + str(path))
    return path


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def validate(expected_config=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Explicitly empty CUDA visibility required")
    if Path.cwd().resolve() != ROOT or Path(__file__).resolve() != SCRIPT:
        raise RuntimeError("Fixed source and dedicated repository root required")
    if expected_config and sha(CONFIG) != expected_config:
        raise ValueError("Root-reviewed readout policy changed")
    cfg = json.loads(CONFIG.read_text())
    if (cfg["kind"] != "ancestralpaths_stable_readout_v1" or
            cfg["implementation_sha256"] != sha(SCRIPT) or
            cfg["seeds"] != [101, 202, 303] or cfg["prediction_batch_size"] != 1024 or
            cfg["C4_native_groups_0based"] != [[0, 4], [1, 5], [2], [3]] or
            cfg["native_probability_max_absolute_error"] != 1e-6 or
            cfg["logit_dtype"] != "float32" or cfg["log_probability_dtype"] != "float64" or
            cfg["retraining"] is not False or cfg["calibration_fit"] is not False or
            cfg["probability_floor"] is not False):
        raise ValueError("Frozen numerical readout policy differs")
    for name, digest in cfg["frozen_files_sha256"].items():
        bound({"path": name, "sha256": digest})
    v2_path = local(cfg["v2_definition"])
    spec = importlib.util.spec_from_file_location("_frozen_APv2_stable_readout", v2_path)
    v2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v2
    spec.loader.exec_module(v2)
    base, definition, parents, caches, metadata, source = v2.validate(local(cfg["v2_config"]))
    if cfg["metrics"] != base["metrics"] or cfg["tensorflow_cpu_prefix"] != base["tensorflow_cpu_prefix"]:
        raise ValueError("Metric definitions/native prefix changed")
    val = {parent["parent_id"]: parent for parent in parents if parent["split"] == "val"}
    if list(val) != cfg["VAL_parent_ids"] or len(val) != 2:
        raise ValueError("Exactly the unchanged two VAL parents required")
    receipt = json.loads(bound(cfg["completed_v2_receipt"]).read_text())
    if receipt["status"] != "passed" or receipt["biological_fit_ran"] is not True:
        raise ValueError("Completed genuine APv2 optimizer fits required")
    if (set(receipt["seeds"]) != {str(seed) for seed in cfg["seeds"]} or
            [model["seed"] for model in cfg["models"]] != cfg["seeds"]):
        raise ValueError("Complete original seed identities required")
    for model in cfg["models"]:
        seed = str(model["seed"])
        state = receipt["seeds"][seed]
        if (state["status"] != "passed" or state["optimizer_fit_completed"] is not True or
                state["checkpoint"] != model["checkpoint"]["path"] or
                state["checkpoint_sha256"] != model["checkpoint"]["sha256"]):
            raise ValueError("Completed checkpoint provenance differs")
        bound(model["checkpoint"])
        if list(model["parents"]) != cfg["VAL_parent_ids"]:
            raise ValueError("Every checkpoint requires both unchanged VAL parents")
        for name, records in model["parents"].items():
            raw = records["raw_prediction"]
            if raw["sha256"] != state["parents"][name]["prediction_sha256"]:
                raise ValueError("Original sealed raw p6 hash differs")
            bound(raw)
            if records["original_metrics"]["path"] != state["parents"][name]["metrics"]:
                raise ValueError("Original metric provenance differs")
            bound(records["original_metrics"])  # bytes only; no VAL score parsing in preflight
    return cfg, definition, val, caches, metadata, source


def stable_probabilities(logits):
    """Finite float64 log probabilities, with no clipping or probability floor."""
    import numpy as np
    if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1] != 6:
        raise ValueError("Native six FP32 logits required")
    if not np.isfinite(logits).all():
        raise ValueError("Native logits are nonfinite")
    value = logits.astype(np.float64)
    shifted = value - value.max(axis=-1, keepdims=True)
    logp6 = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    logp4 = np.stack((np.logaddexp(logp6[:, 0], logp6[:, 4]),
                     np.logaddexp(logp6[:, 1], logp6[:, 5]),
                     logp6[:, 2], logp6[:, 3]), axis=-1)
    p6, p4 = np.exp(logp6), np.exp(logp4)
    if (not np.isfinite(logp6).all() or not np.isfinite(logp4).all() or
            not np.allclose(p6.sum(-1), 1, atol=1e-12, rtol=0) or
            not np.allclose(p4.sum(-1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Stable readout is nonfinite or unnormalized")
    if not np.allclose(p4, np.stack((p6[:, 0] + p6[:, 4], p6[:, 1] + p6[:, 5],
                                    p6[:, 2], p6[:, 3]), axis=-1), atol=1e-15, rtol=1e-12):
        raise ValueError("Stable C4 probabilities do not sum native groups")
    return logp6, logp4, p6, p4


def weight_manifest(model):
    records = []
    for weight in model.weights:
        value = weight.numpy()
        records.append({"name": weight.name, "shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()})
    return records


def source_logits(model, features, batch_size, tf):
    """Exact installed Keras Dense rank2 matmul+bias before its softmax."""
    import numpy as np
    layers = model.layers
    if (len(layers) != 4 or model.count_params() != 26630 or model.input_shape != (None, 5, 9) or
            model.output_shape != (None, 6) or
            type(layers[0]) is not tf.keras.layers.Flatten or
            type(layers[1]) is not tf.keras.layers.Dense or layers[1].units != 512 or
            tf.keras.activations.serialize(layers[1].activation) != "relu" or
            type(layers[2]) is not tf.keras.layers.Dropout or layers[2].rate != .2 or
            type(layers[3]) is not tf.keras.layers.Dense or layers[3].units != 6 or
            tf.keras.activations.serialize(layers[3].activation) != "softmax"):
        raise ValueError("Actual frozen source-native Dense architecture differs")
    final = layers[-1]
    if (not final.use_bias or final.kernel.dtype.name != "float32" or
            final.bias.dtype.name != "float32" or features.dtype != np.float32 or
            features.shape[1:] != (5, 9)):
        raise ValueError("Actual native FP32 Dense/features required")
    before = weight_manifest(model)
    hidden_model = tf.keras.Model(inputs=model.inputs, outputs=final.input)
    logits, reconstructed = [], []
    # Keep the native model.predict path and batch size used for the original p6.
    native_p6 = model.predict(features, batch_size=batch_size, verbose=0)
    for start in range(0, len(features), batch_size):
        hidden = hidden_model(features[start:start + batch_size], training=False)
        if hidden.dtype.name != "float32" or hidden.shape.rank != 2:
            raise ValueError("Deterministic hidden activation differs")
        value = tf.nn.bias_add(tf.matmul(a=hidden, b=final.kernel), final.bias)
        logits.append(value.numpy())
        reconstructed.append(final.activation(value).numpy())
    logits = np.concatenate(logits)
    reconstructed = np.concatenate(reconstructed)
    if not np.isfinite(native_p6).all() or native_p6.dtype != np.float32:
        raise ValueError("Actual native p6 is nonfinite/not FP32")
    error = float(np.max(np.abs(reconstructed.astype(np.float64) - native_p6)))
    if error > 1e-6:
        raise ValueError("FP32 Dense logit reconstruction disagrees with native p6")
    if weight_manifest(model) != before:
        raise ValueError("Readout changed checkpoint weights")
    return logits, native_p6, {"native_reconstructed_softmax_max_error": error,
                              "weights_before_and_after": before,
                              "hidden_activation_training": False,
                              "Dense_math": "FP32 tf.matmul then tf.nn.bias_add; original kernel/bias"}


def seal_predictions(model, definition, parent, cache, records, seed_dir, cfg, tf):
    import numpy as np
    raw_path = bound(records["raw_prediction"])
    with np.load(cache, allow_pickle=False) as z:
        features = z["features"]
        index = z["grid_feature_index"]
        support = {key: z[key].copy() for key in
            ("positions_bp", "bin_edges_bp", "query_node_ids", "query_sample_indices",
             "retained_fixed100bp_indexes", "coverage_original_bp0", "local_to_original_offset_bp")}
    with np.load(raw_path, allow_pickle=False) as raw:
        original_p6 = raw["native_probabilities"].copy()
        original_p4 = raw["C4_probabilities"].copy()
        for key, value in support.items():
            if not np.array_equal(raw[key], value):
                raise ValueError("Original raw prediction/cache support differs: " + key)
    if original_p6.dtype != np.float32 or original_p6.shape != (*index.shape, 6):
        raise ValueError("Preserved actual source FP32 p6 shape/dtype differs")
    bank_logits, bank_p6, numerical = source_logits(
        model, features, cfg["prediction_batch_size"], tf)
    logp6, logp4, p6, p4 = stable_probabilities(bank_logits)
    stable_error = float(np.max(np.abs(p6 - bank_p6.astype(np.float64))))
    original_error = float(np.max(np.abs(bank_p6[index].astype(np.float64) - original_p6)))
    stable_original_error = float(np.max(np.abs(p6[index] - original_p6)))
    if max(stable_error, original_error, stable_original_error) > cfg["native_probability_max_absolute_error"]:
        raise ValueError("Stable/native/reloaded p6 probability agreement exceeded 1e-6")
    path = seed_dir / (parent["parent_id"] + "_stable_logits_and_probabilities.npz")
    with path.open("xb") as stream:
        np.savez_compressed(stream, native_logits_fp32=bank_logits[index],
            native_log_probabilities_float64=logp6[index],
            C4_log_probabilities_float64=logp4[index],
            native_probabilities_stable_float64=p6[index],
            C4_probabilities_stable_float64=p4[index],
            native_bank_logits_fp32=bank_logits, grid_feature_index=index,
            normalization_length_bp=np.asarray(2000000), **support)
    numerical.update(stable_vs_native_p6_max_error=stable_error,
                     stable_vs_preserved_raw_p6_max_error=stable_original_error,
                     recomputed_vs_preserved_raw_p6_max_error=original_error,
                     raw_native_zero_count=int(np.sum(original_p6 == 0)),
                     raw_C4_zero_count=int(np.sum(original_p4 == 0)),
                     stable_native_zero_count=int(np.sum(p6[index] == 0)),
                     stable_C4_zero_count=int(np.sum(p4[index] == 0)),
                     native_argmax_disagreements=int(np.sum(p6[index].argmax(-1) != original_p6.argmax(-1))),
                     C4_argmax_disagreements=int(np.sum(p4[index].argmax(-1) != original_p4.argmax(-1))))
    seal = {"parent_id": parent["parent_id"], "prediction_path": str(path.relative_to(ROOT)),
            "prediction_sha256": sha(path), "original_raw_prediction": records["raw_prediction"],
            "original_metrics": records["original_metrics"], "numerical_checks": numerical,
            "sealed_before_any_VAL_truth": True, "probability_floor_used": False}
    write_new(path.with_suffix(".seal.json"), seal)
    return seal


def auxiliary_score(logp6, logp4, p4, parent, truth, columns, definition, metric_cfg):
    """Same exact support/Brier/ECE algebra; NLL uses logp directly."""
    import numpy as np
    bins, rows, spans = definition.intersections(parent["edges"], truth.intervals)
    reliability = np.zeros((metric_cfg["calibration_bins"], 3), dtype=np.float64)
    nll4 = nll6 = brier_sum = correct_bp = 0.
    for start in range(0, len(bins), 1024):
        cut = slice(start, start + 1024)
        y4 = truth.c4[rows[cut]][:, columns]
        y6 = truth.paths[rows[cut]][:, columns].astype(np.int64) - 1
        probs = p4[bins[cut]][:, columns]
        selected_log4 = np.take_along_axis(logp4[bins[cut]][:, columns], y4[..., None], -1)[..., 0]
        selected_log6 = np.take_along_axis(logp6[bins[cut]][:, columns], y6[..., None], -1)[..., 0]
        weights = spans[cut, None]
        nll4 += float(np.sum(weights * -selected_log4))
        nll6 += float(np.sum(weights * -selected_log6))
        selected = np.take_along_axis(probs, y4[..., None], -1)[..., 0]
        brier_sum += float(np.sum(weights * (np.sum(probs * probs, axis=-1) + 1 - 2 * selected)))
        predicted = probs.argmax(-1)
        correct_bp += float(np.sum(weights * (predicted == y4)))
        confidence = probs.max(-1)
        slots = np.minimum((confidence * len(reliability)).astype(int), len(reliability) - 1)
        for slot in range(len(reliability)):
            mask = slots == slot
            reliability[slot, 0] += np.sum(weights * mask)
            reliability[slot, 1] += np.sum(weights * confidence * mask)
            reliability[slot, 2] += np.sum(weights * (predicted == y4) * mask)
    total = float(spans.sum() * len(columns))
    if total <= 0 or not all(np.isfinite(value) for value in (nll4, nll6, brier_sum)):
        raise ValueError("Direct log-probability NLL is nonfinite or unsupported")
    return {"query_count": len(columns), "denominator_query_bp": total,
            "C4_nll_from_finite_logp": nll4 / total,
            "native6_nll_from_finite_logp": nll6 / total,
            "C4_brier_stable_probabilities": brier_sum / total,
            "C4_calibration_ece_stable_probabilities":
                float(np.abs(reliability[:, 2] - reliability[:, 1]).sum() / total),
            "auxiliary_C4_argmax_accuracy": correct_bp / total,
            "calibration_reliability_bins": [
                {"bin_index": i, "query_bp": float(r[0]),
                 "mean_confidence": float(r[1] / r[0]) if r[0] else None,
                 "accuracy": float(r[2] / r[0]) if r[0] else None}
                for i, r in enumerate(reliability)],
            "probability_floor_used": False, "probability_or_logit_clipping": False,
            "normalization_length_bp": 2000000,
            "score_scope": "unchanged exact retained-center clipped-bin/complete-truth intersections"}


def score_sealed(definition, parent, seal, cfg):
    import numpy as np
    artifact = bound({"path": seal["prediction_path"], "sha256": seal["prediction_sha256"]})
    with np.load(artifact, allow_pickle=False) as z:
        logp6 = z["native_log_probabilities_float64"]
        logp4 = z["C4_log_probabilities_float64"]
        p4 = z["C4_probabilities_stable_float64"]
        parent = dict(parent, positions=z["positions_bp"].copy(), edges=z["bin_edges_bp"].copy(),
                      retained_indexes=z["retained_fixed100bp_indexes"].copy(),
                      coverage_original=z["coverage_original_bp0"].copy())
    # All six prediction files are already sealed by run() before this capability.
    original = json.loads(bound(seal["original_metrics"]).read_text())
    if (original["parent_id"] != parent["parent_id"] or original["split"] != "val" or
            original["prediction_sha256"] != seal["original_raw_prediction"]["sha256"]):
        raise ValueError("Original metric/prediction identity differs")
    truth = definition.TruthReader(parent, "sealed_val_score")
    cohort = parent["cohorts"]
    groups = {"primary_present_day_64": np.flatnonzero(cohort == "present_day"),
              "secondary_ancient_56": np.flatnonzero(cohort != "present_day"),
              "all_queries_120": np.arange(120)}
    results = {}
    for name, columns in groups.items():
        item = auxiliary_score(logp6, logp4, p4, parent, truth, columns, definition, cfg["metrics"])
        raw = original["C4_exact_full_span_metrics"][name]
        if raw["denominator_query_bp"] != item["denominator_query_bp"]:
            raise ValueError("Original and auxiliary scored support differ")
        item["preserved_raw_float32_metrics"] = {
            "nll": raw["nll"], "nll_is_infinite": raw["infinite_nll_query_bp"] > 0,
            "infinite_nll_query_bp": raw["infinite_nll_query_bp"],
            "brier": raw["brier"], "calibration_ece": raw["calibration_ece"],
            "accuracy": raw["accuracy"], "original_metric_file_unchanged": True}
        item["Brier_numerical_delta"] = item["C4_brier_stable_probabilities"] - raw["brier"]
        item["ECE_numerical_delta"] = item["C4_calibration_ece_stable_probabilities"] - raw["calibration_ece"]
        results[name] = item
    return {"parent_id": parent["parent_id"], "split": "val", "groups": results,
            "prediction_seal": seal, "truth_provenance": truth.provenance,
            "readout_condition": "auxiliary logit-derived float64 log probabilities; original raw benchmark retained",
            "no_retraining_or_calibration": True, "no_paper_CNN_or_holdout_claim": True}


def run(cfg, definition, parents, caches, output):
    if Path(sys.prefix).resolve() != local(cfg["tensorflow_cpu_prefix"]):
        raise RuntimeError("Qualified existing TensorFlow CPU interpreter required")
    import tensorflow as tf
    if (importlib.metadata.version("tensorflow-cpu") != "2.15.1" or
            importlib.metadata.version("keras") != "2.15.0" or tf.config.list_physical_devices("GPU")):
        raise RuntimeError("Qualified CPU-only native runtime required")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    output.mkdir(exist_ok=False)
    receipt = {"status": "running", "implementation_sha256": sha(SCRIPT), "config_sha256": sha(CONFIG),
               "prediction_seals": [], "scores": [], "training": False, "GPU": False,
               "calibration_fit": False, "original_raw_predictions_modified": False}
    started = time.monotonic()
    code = 1
    try:
        for binding in cfg["models"]:
            seed_dir = output / ("seed_" + str(binding["seed"])); seed_dir.mkdir()
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(binding["seed"])
            checkpoint = bound(binding["checkpoint"])
            model = tf.keras.models.load_model(checkpoint, compile=False)
            for name in cfg["VAL_parent_ids"]:
                seal = seal_predictions(model, definition, parents[name], caches[name],
                                        binding["parents"][name], seed_dir, cfg, tf)
                seal["seed"] = binding["seed"]
                seal["checkpoint"] = binding["checkpoint"]
                receipt["prediction_seals"].append(seal)
            if sha(checkpoint) != binding["checkpoint"]["sha256"]:
                raise ValueError("Checkpoint bytes changed during inference")
            del model
        if len(receipt["prediction_seals"]) != 6:
            raise RuntimeError("Seal all three-seed/two-parent exports before VAL scoring")
        write_new(output / "all_predictions_sealed_before_truth.json", receipt)
        for seal in receipt["prediction_seals"]:
            result = score_sealed(definition, parents[seal["parent_id"]], seal, cfg)
            path = output / ("seed_" + str(seal["seed"])) / (seal["parent_id"] + "_stable_auxiliary_metrics.json")
            write_new(path, result)
            receipt["scores"].append({"seed": seal["seed"], "parent_id": seal["parent_id"],
                                     "path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        validate(receipt["config_sha256"])  # Verify every immutable input again after scoring.
        receipt["status"] = "passed"
        code = 0
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error),
                              "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        receipt.update(exit_code=code, elapsed_seconds=time.monotonic() - started)
        write_new(output / "receipt.json", receipt)
        (output / "exit_status").write_text(str(code) + "\n")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--output", default="experiments/results/ancestralpaths_stable_readout_v1")
    args = parser.parse_args()
    if args.execute and not args.expected_config_sha256:
        parser.error("Actual readout requires root-reviewed config hash")
    cfg, definition, parents, caches, metadata, source = validate(args.expected_config_sha256)
    if not args.execute:
        print(json.dumps({"status": "frozen_readout_bindings_checked_no_inference_or_truth",
                          "config_sha256": sha(CONFIG), "model_loaded": False,
                          "TensorFlow_imported": False, "truth_read": False}))
        return 0
    output = local(args.output)
    if output.exists() or not output.parent.is_dir() or any(output.is_relative_to(local(name)) for name in
            ("checkpoints/ancestralpaths_biological_pilot_v1", "checkpoints/ancestralpaths_biological_pilot_v2")):
        raise ValueError("Fresh separate readout output required; original outputs are immutable")
    return run(cfg, definition, parents, caches, output)


if __name__ == "__main__":
    raise SystemExit(main())

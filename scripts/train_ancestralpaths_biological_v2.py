#!/usr/bin/env python
"""H020 v2: fit realized native source-quota rows using frozen qualified v1 features."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
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
DEFAULT_CONFIG = "configs/ancestralpaths_biological_pilot_v2.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local(value):
    p = (ROOT / value).resolve()
    p.relative_to(ROOT)
    return p


def load_definition(path):
    spec = importlib.util.spec_from_file_location("frozen_ancestralpaths_biological_v1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(config_path):
    cfg = json.loads(config_path.read_text())
    if cfg["kind"] != "ancestralpaths_biological_pilot_v2" or cfg["hypothesis_id"] != "H020":
        raise ValueError("Wrong biological v2 protocol identity")
    if Path.cwd().resolve() != ROOT or sha(Path(__file__)) != cfg["implementation_sha256"]:
        raise ValueError("Dedicated root and frozen v2 implementation are required")
    for name, expected in cfg["frozen_files_sha256"].items():
        if sha(local(name)) != expected:
            raise ValueError("Frozen source/evidence drift: " + name)
    definition = load_definition(local(cfg["v1_definition"]))
    base, parents, source = definition.protocol(local(cfg["v1_config"]))
    if (cfg["training"] != base["training"] or cfg["native_model"] != base["native_model"]
            or cfg["parent_partition"] != base["parent_partition"]):
        raise ValueError("Only the sampler's unfilled-quota failure policy may change")
    if cfg["metrics"] != base["metrics"]:
        raise ValueError("Frozen native/C4 metrics must remain identical")
    if cfg["sampler"]["unfilled_quota_policy"] != "retain_realized_accepted_rows_and_continue_cohorts":
        raise ValueError("Require the explicitly approved realized-row policy")
    if list(cfg["sampler"]["per_parent_cohort_budgets"]) != list(
            base["sampler"]["per_parent_cohort_budgets"]):
        raise ValueError("Source cohort sampling order must remain identical")
    sampler = dict(cfg["sampler"])
    sampler["unfilled_quota_policy"] = base["sampler"]["unfilled_quota_policy"]
    if sampler != base["sampler"]:
        raise ValueError("Native candidate budgets/caps/strides/pass bound must remain unchanged")
    manifest = local(cfg["observed_manifest"])
    if sha(manifest) != cfg["observed_manifest_sha256"]:
        raise ValueError("Completed observed manifest differs from qualified v1")
    observed = definition.observations(manifest, parents, base)
    prior = json.loads(local(cfg["v1_execution_receipt"]).read_text())
    if (prior["implementation_sha256"] != cfg["frozen_files_sha256"][cfg["v1_definition"]]
            or prior["config_sha256"] != cfg["frozen_files_sha256"][cfg["v1_config"]]
            or prior["observed_manifest_sha256"] != cfg["observed_manifest_sha256"]
            or prior.get("seeds") != {} or prior.get("biological_fit_ran") is not False
            or prior["status"] != "failed"
            or not all(r["stage"] == "native_TRAIN_sampler" for r in prior["failures"])):
        raise ValueError("Require the preserved source-sampler failure before any v1 fit")
    caches, metadata = {}, {}
    for parent in parents:
        name = parent["parent_id"]
        record = prior["feature_caches"][name]
        binding = cfg["reused_feature_caches"][name]
        path, meta_path = local(binding["path"]), local(binding["metadata_path"])
        if (path != local(record["cache_path"]) or sha(path) != record["cache_sha256"]
                or binding["sha256"] != record["cache_sha256"]
                or sha(meta_path) != binding["metadata_sha256"]):
            raise ValueError("Qualified v1 feature cache bytes/metadata changed: " + name)
        actual_meta = json.loads(meta_path.read_text())
        if (actual_meta != record or record["observed_manifest_entry"] != observed[name]["entry"]
                or record["source_bridge_float32_equivalence_first_observed_tree"] is not True
                or record["truth_accessed"] is not False
                or record["observed_tree_sha256"] != observed[name]["entry"]["inferred_tree"]["sha256"]):
            raise ValueError("Cache qualification/observed source identity mismatch: " + name)
        # Only observed arrays/feature banks: no external targets, no extraction rerun.
        with np.load(path, allow_pickle=False) as z:
            if (not np.array_equal(z["query_sample_indices"], parent["query_slots"])
                    or not np.array_equal(z["query_node_ids"],
                                          observed[name]["nodes"][parent["query_slots"]])
                    or not np.array_equal(z["positions_bp"], observed[name]["positions"])
                    or not np.array_equal(z["bin_edges_bp"], observed[name]["bin_edges"])
                    or not np.array_equal(z["retained_fixed100bp_indexes"],
                                          observed[name]["retained_indexes"])
                    or not np.array_equal(z["coverage_original_bp0"], observed[name]["coverage"])
                    or float(z["local_to_original_offset_bp"]) != observed[name]["offset"]
                    or z["grid_feature_index"].shape != (len(observed[name]["positions"]), 120)):
                raise ValueError("Cache physical grid/query correspondence mismatch: " + name)
            site_positions = z["site_positions_bp"]
            if not np.array_equal(site_positions, np.array(
                    [r["original_position_0based"] for r in observed[name]["site_join"]])):
                raise ValueError("Cache TRAIN site physical join mismatch: " + name)
        caches[name], metadata[name] = path, record
    return cfg, definition, parents, caches, metadata, source


def sample_realized_source_rows(definition, parent, cache, cfg, output):
    """Keep the v1/native order, caps and replacement, accepting actual shortfall rows."""
    truth = definition.TruthReader(parent, "train_supervision")
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
                raise ValueError("Native source initial stride is zero for " + name)
            initial_step = step
            counts, accepted, passes = np.zeros(6, dtype=np.int64), [], []
            reason = "maximum_source_passes_reached"
            for pass_number in range(cfg["sampler"]["maximum_source_passes"]):
                before = len(accepted)
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
                passes.append(dict(pass_number=pass_number, step=step,
                                   added=len(accepted) - before,
                                   realized_native_path_counts=counts.copy()))
                if len(accepted) == target:
                    reason = "source_target_filled"
                    break
                if len(accepted) == before:
                    reason = "source_stride_pass_added_no_uncapped_rows"
                    break
                step = step + step / 2
            report = dict(cohort=name, observed_queries=int(len(columns)),
                          observed_sites=len(sites), target=target, per_path_cap=cap,
                          candidate_scan_budget=budget["candidate_scan_budget"],
                          initial_step=initial_step, source_overhead=10000 / 3,
                          realized_native_path_counts=counts,
                          selected_examples=len(accepted), shortfall_examples=target - len(accepted),
                          unique_site_query_examples=len(set(accepted)),
                          replacement_duplicates=len(accepted) - len(set(accepted)),
                          passes=passes, stop_reason=reason, quota_filled=len(accepted) == target,
                          unfilled_quota_policy=cfg["sampler"]["unfilled_quota_policy"])
            reports.append(report)
            # The sole v2 change: include these genuine accepted rows despite
            # shortfall, then proceed to the next cohort (including Yam).
            selected.extend(accepted)
        if not selected:
            raise ValueError("No genuine supervised rows were accepted")
        pairs = np.asarray(selected, dtype=np.int32)
        y = labels[pairs[:, 0], pairs[:, 1]].astype(np.int64) - 1
        x = features[index[pairs[:, 0], pairs[:, 1]]]
        with (output / (parent["parent_id"] + "_training_selection.npz")).open("xb") as stream:
            np.savez_compressed(stream, site_row=pairs[:, 0], query_column=pairs[:, 1],
                                positions_bp=sites[pairs[:, 0]],
                                query_sample_indices=parent["query_slots"][pairs[:, 1]],
                                native_labels_0based=y)
    definition.write_json(output / (parent["parent_id"] + "_sampler.json"), dict(
        parent_id=parent["parent_id"], reports=reports, selected_examples=len(y),
        total_source_target=sum(r["target"] for r in reports),
        total_shortfall=sum(r["shortfall_examples"] for r in reports),
        truth_provenance=truth.provenance, train_only_truth_selection=True,
        padded_or_phantom_rows=0, class_caps_increased=False,
        actual_rows_fitted_even_if_source_quota_unfilled=True))
    return x, y, reports


def run(cfg, definition, parents, caches, metadata, output, config_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Require explicitly empty CUDA visibility")
    if Path(sys.prefix).resolve() != local(cfg["tensorflow_cpu_prefix"]):
        raise RuntimeError("Use the qualified isolated TensorFlow CPU prefix")
    import tensorflow as tf
    if importlib.metadata.version("tensorflow-cpu") != "2.15.1":
        raise RuntimeError("Qualified TensorFlow version differs")
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Unexpected GPU in native CPU fit")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    output.mkdir(parents=True, exist_ok=False)
    sampler_dir = output / "training_sampler"; sampler_dir.mkdir()
    receipt = dict(hypothesis_id="H020", kind="ancestralpaths_biological_pilot_v2",
                   status="running", started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   implementation_sha256=sha(Path(__file__)), config_sha256=sha(config_path),
                   repeat_reason=cfg["repeat_reason"], feature_extraction_rerun=False,
                   reused_v1_feature_caches=metadata,
                   v1_execution_receipt=dict(path=cfg["v1_execution_receipt"],
                                            sha256=sha(local(cfg["v1_execution_receipt"]))),
                   unchanged_definition=dict(path=cfg["v1_definition"],
                                             sha256=sha(local(cfg["v1_definition"]))),
                   observed_manifest=dict(path=cfg["observed_manifest"],
                                          sha256=cfg["observed_manifest_sha256"]),
                   runtime=dict(executable=sys.executable, packages={
                       name: importlib.metadata.version(name) for name in
                       ("numpy", "tensorflow-cpu", "keras", "tskit", "msprime", "demes", "pandas")}),
                   GPU_used=False, biological_fit_ran=False, TEST_parents=0,
                   sampler_reports={}, failures=[], seeds={},
                   v1_outputs_modified=False, phantom_rows=0,
                   changed_policy="retain realized source-quota rows and continue cohorts",
                   native_Dense_CNN_distinction="source-native Dense; no published CNN claim")
    start = time.monotonic()
    def save():
        definition.write_json(output / "receipt.json", receipt)
    def failure(stage, exc, **identity):
        item = dict(stage=stage, **identity, exception_type=type(exc).__name__,
                    error=traceback.format_exc())
        receipt["failures"].append(item)
        with (output / "failures.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        save()
    save()
    train_x, train_y = [], []
    for parent in parents:
        if parent["split"] == "train":
            try:
                x, y, reports = sample_realized_source_rows(
                    definition, parent, caches[parent["parent_id"]], cfg, sampler_dir)
                train_x.append(x); train_y.append(y)
                receipt["sampler_reports"][parent["parent_id"]] = reports
                save()
            except Exception as exc:
                failure("realized_native_TRAIN_sampler", exc, parent_id=parent["parent_id"])
    if not receipt["failures"]:
        x, y = np.concatenate(train_x), np.concatenate(train_y)
        targets = tf.keras.utils.to_categorical(y, num_classes=6)
        receipt["training_examples"] = len(y)
        receipt["realized_native_path_counts"] = np.bincount(y, minlength=6).tolist()
        receipt["source_target_total"] = sum(
            r["target"] for rows in receipt["sampler_reports"].values() for r in rows)
        receipt["shortfall_total"] = receipt["source_target_total"] - len(y)
        for seed in cfg["training"]["seeds"]:
            seed_dir = output / ("seed_" + str(seed)); seed_dir.mkdir()
            state = dict(status="running", parents={})
            receipt["seeds"][str(seed)] = state
            try:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = definition.build_native_keras_model(tf.keras)
                if model.count_params() != 26630 or model.output_shape != (None, 6):
                    raise ValueError("Unchanged native model differs")
                before = [w.numpy().copy() for w in model.trainable_weights]
                history = model.fit(x, targets, epochs=5, batch_size=32, shuffle=True, verbose=2)
                delta = max(float(np.max(np.abs(a - b.numpy())))
                            for a, b in zip(before, model.trainable_weights))
                if not delta > 0 or not all(np.isfinite(v).all() for v in history.history.values()):
                    raise ValueError("Actual native optimizer fit lacks finite changed weights")
                checkpoint = seed_dir / "native_dense_epoch5.keras"
                model.save(checkpoint)
                state.update(optimizer_fit_completed=True, fit_history=history.history,
                             optimizer=model.optimizer.get_config(), maximum_weight_delta=delta,
                             checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                             epochs=5, batch_size=32, no_VAL_early_stopping=True)
                receipt["biological_fit_ran"] = True
                definition.write_json(seed_dir / "fit.json", dict(state, status="fit_completed"))
                for parent in parents:
                    if parent["split"] == "val":
                        try:
                            result = definition.predict_and_score(
                                model, parent, caches[parent["parent_id"]], seed_dir, cfg)
                            state["parents"][parent["parent_id"]] = dict(
                                status="passed", prediction_sha256=result["prediction_sha256"],
                                metrics=str((seed_dir / (parent["parent_id"] +
                                                        "_metrics.json")).relative_to(ROOT)))
                        except Exception as exc:
                            state["parents"][parent["parent_id"]] = dict(status="failed")
                            failure("unchanged_sealed_VAL_prediction_score", exc,
                                    seed=seed, parent_id=parent["parent_id"])
                state["status"] = "passed" if all(
                    r["status"] == "passed" for r in state["parents"].values()) else "failed"
            except Exception as exc:
                state["status"] = "failed"
                failure("actual_source_native_fit", exc, seed=seed)
            save()
    receipt.update(status="passed" if not receipt["failures"] else "failed",
                   elapsed_seconds=time.monotonic() - start,
                   sampler_shortfall_is_disclosed_method_condition=True,
                   final_holdout_claim=False, exact_paper_reproduction=False)
    save()
    return 0 if receipt["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default="checkpoints/ancestralpaths_biological_pilot_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path = local(args.config)
    cfg, definition, parents, caches, metadata, source = validate(config_path)
    if args.check_only:
        print(json.dumps(dict(status="source_observed_cache_bindings_verified",
                              cache_reuse=4, extraction_rerun=False, truth_opened=False,
                              TensorFlow_imported=False, fit_run=False,
                              v1_outputs_modified=False, source_commit=source["commit"])))
        return 0
    output = local(args.output)
    if output.is_relative_to(local("checkpoints/ancestralpaths_biological_pilot_v1")):
        raise ValueError("V2 must never write inside preserved v1 outputs")
    return run(cfg, definition, parents, caches, metadata, output, config_path)


if __name__ == "__main__":
    raise SystemExit(main())

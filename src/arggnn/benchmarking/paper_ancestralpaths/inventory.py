"""Targeted metadata/array inventory from the known queue; never reads TEST truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import CONDITION, data
from .model import model_spec
from .. import european_native_models as io


def inventory(config_path, output_path):
    import numpy as np
    config_ref = io._ref(io._path(config_path))
    config = io._read(config_ref)
    by_id = {t["id"]: t for t in config["tasks"]}
    plan_ref = io._ref(io.ROOT / "configs/benchmarks/recipes/ap_paper_sampled_plan_v1.json")
    plan = data.checked_plan(plan_ref)

    def result(identity):
        task = by_id[identity]
        directory = task.get("output_directory") or task["workflow_predecessor"]["output_directory"]
        record = json.loads((io._path(directory) / "result.json").read_text())
        return record["value"]

    dense_fit_ref = result("AP__all_train__ap_fit")
    dense_fit = io._read(dense_fit_ref)
    training_products = dense_fit["training_products"]
    records, test_features, dense_predictions, targets, paired = {}, {}, {}, {}, {}
    for parent in plan["train_parent_ids"] + plan["test_parent_ids"]:
        split = plan["parents"][parent]["split"]
        generated = result("AP__" + parent + "__ap_generate")
        io._need(generated["parent_id"] == parent and generated["split"] == split
            and generated["sequence_length_bp"] == 200000000,
            "Exact existing paper-length generation result required")
        if split == "train":
            feature_ref = training_products[parent]["features_ref"]
        else:
            dense_predictions[parent] = result("AP__" + parent + "__ap_predict")
            prediction = io._read(dense_predictions[parent], observed=True)
            io._checked(prediction["predictions_ref"], observed=True)
            feature_ref = prediction["features_ref"]
            test_features[parent] = feature_ref
            # Opaque producer references: no target schema, target NPZ or generation JSON is opened.
            targets[parent] = {key: generated[key] for key in
                               ("generation_ref", "target_schema_ref", "truth_ref")}
        f, r, x = data.load_feature(feature_ref, plan, parent=parent, split=split)
        inference_path = io._path(f["inference_ref"]["path"], observed=True)
        io._need(inference_path.is_file() and not inference_path.is_symlink(),
                 "Frozen inference manifest must still exist")
        paired[parent] = dict(inference_ref=f["inference_ref"],
            observed_vcf_ref=generated["observed_input_refs"]["genotypes.vcf"],
            sampled_registry_ref=f["registry_ref"])
        if split == "train":
            paired[parent]["training_targets_ref"] = training_products[parent]["targets_ref"]
        records[parent] = dict(split=split, rows=len(x), feature_ref=feature_ref,
            feature_arrays_ref=f["arrays_ref"], registry_ref=f["registry_ref"],
            inference_ref=f["inference_ref"], cohort_quotas=r["cohort_quotas"],
            features_shape=list(x.shape), features_dtype=str(x.dtype),
            feature_arrays_and_registry_sha256_reverified=True,
            feature_min=float(x.min()), feature_max=float(x.max()),
            inferred_manifest_and_large_observed_inputs_rehashed_in_inventory=False,
            generation_result=generated,
            observed_only=True)
    x, y, ids, cohort_counts = data.load_training(plan_ref, training_products)
    io._checked(dense_fit["checkpoint_ref"], observed=True)
    output = dict(schema="paper_ancestralpaths_frozen476_reuse_v1", condition=CONDITION,
        queue_config_ref=config_ref, sampling_plan_ref=plan_ref,
        parent_registry_ref=config["ap_parent_registry_ref"], parents=records,
        training_products=training_products, test_features=test_features,
        test_target_references=targets, test_truth_opened=False,
        training_shape=list(x.shape), training_label_shape=list(y.shape),
        rows_per_cohort=cohort_counts, global_training_ids_unique=len(set(ids)) == 50000,
        dense_control=dict(fit_ref=dense_fit_ref, checkpoint_ref=dense_fit["checkpoint_ref"],
            test_predictions=dense_predictions, reusable_as_historical_cnn_checkpoint=False,
            reason="Dense26,630 parameters differs from CNN19,078; retain separate control"),
        treeloom_observation_focal_interfaces=paired,
        reuse_assessment=dict(generation="reuse all eight existing 476-person 200Mb parents",
            inference="reuse the identical frozen post-five-size Relate products through feature provenance",
            registries="reuse exact observation-only 50k TRAIN and 10k per TEST rows",
            features="reuse exact Float32 [N,5,9]; add singleton input channel only",
            train_targets="reuse exact 0-based native-six TRAIN labels",
            dense_weights="incompatible; fresh CNN initialization required"),
        limits="Large VCF/tree contents not rehashed here; execution must verify exact pinned refs before use",
        model_spec=model_spec())
    with io._path(output_path).open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = inventory(args.queue_config, args.output)
    print(json.dumps(dict(parents=len(out["parents"]), training_rows=50000,
        test_parents=5, test_truth_opened=False, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()

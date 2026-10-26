"""Build an additive four-path task DAG without changing a running campaign."""
from __future__ import annotations

from . import CONDITION, PATH_NAMES, QUERY_COHORTS
from .model import model_spec
from .. import european_native_models as io
from ..paper_ancestralpaths.data import checked_plan

MODULE = "arggnn.benchmarking.paper_ancestralpaths_four.workflow"


def build(source_config_ref, *, output_root):
    source = io._read(source_config_ref, observed=True)
    original_fit = next(t for t in source["tasks"] if t["factory_arguments"]["stage"] == "fit")
    original_score = next(t for t in source["tasks"] if t["factory_arguments"]["stage"] == "score")
    fit_args = original_fit["factory_arguments"]["arguments"]
    score_args = original_score["factory_arguments"]["arguments"]
    plan_ref = fit_args["plan_ref"]
    plan = checked_plan(plan_ref)
    io._need(score_args["plan_ref"] == plan_ref, "One original sampling plan required")
    root = io._path(output_root)
    io._need(root.is_relative_to(io.ROOT / "experiments/results") and not root.exists(),
             "Fresh output namespace required")
    runtime = ".runtime/conda/envs/ancestralpaths_cpu_v1/bin/python3.11"
    prepare_runtime = ".runtime/env/bin/python3.11"
    tasks, prepared = [], {}

    def task(identity, stage, args, deps, *, rss=4*2**30, seconds=7200, size=32*2**20, python=runtime):
        return dict(id=identity, resource="cpu", dependencies=deps,
            factory_module=MODULE, factory_function="make_call",
            factory_arguments=dict(stage=stage, arguments=dict(args, write_headroom_bytes=4*2**20), python=python),
            output_directory=str(root / identity), max_seconds=seconds, max_rss_bytes=rss, max_output_bytes=size)

    interfaces = source["treeloom_observation_focal_interfaces"]
    for parent in plan["train_parent_ids"] + plan["test_parent_ids"]:
        refs = interfaces[parent]
        identity = "paper_ap_four__" + parent + "__prepare"
        tasks.append(task(identity, "prepare", dict(source_plan_ref=plan_ref,
            registry_ref=refs["sampled_registry_ref"], inference_ref=refs["inference_ref"]),
            [], rss=64*2**30, seconds=14400, python=prepare_runtime))
        prepared[parent] = {"$task": identity}
    fit_id = "paper_ap_four__all_train__fit"
    training = {parent: dict(features_ref=prepared[parent],
        source_targets_ref=fit_args["training_products"][parent]["targets_ref"])
        for parent in plan["train_parent_ids"]}
    tasks.append(task(fit_id, "fit", dict(source_plan_ref=plan_ref, training_products=training,
        seed=20261027, layout="portable_nhwc", device="cpu", threads=4, workers=1),
        [prepared[p]["$task"] for p in plan["train_parent_ids"]]))
    predictions = {}
    for parent in plan["test_parent_ids"]:
        identity = "paper_ap_four__" + parent + "__predict"
        tasks.append(task(identity, "predict", dict(fit_ref={"$task":fit_id},
            features_ref=prepared[parent], device="cpu", threads=4),
            [fit_id, prepared[parent]["$task"]], seconds=900, size=16*2**20))
        predictions[parent] = {"$task":identity}
    tasks.append(task("paper_ap_four__all_test__score", "score", dict(source_plan_ref=plan_ref,
        predictions={CONDITION:predictions}, expected_methods=[CONDITION],
        test_targets=score_args["test_targets"], max_truth_uncompressed_bytes=16*2**30),
        [p["$task"] for p in predictions.values()], rss=32*2**30, seconds=1800, size=16*2**20,
        python=prepare_runtime))
    return dict(schema="paper_ancestralpaths_four_plan_v1", plan_only=True, admitted=False,
        condition=CONDITION, source_config_ref=source_config_ref, source_plan_ref=plan_ref,
        model_spec=model_spec(), class_names=list(PATH_NAMES), tasks=tasks,
        training_rows=40000, training_rows_per_cohort=10000, test_rows_per_parent=8000,
        present_day_test_rows_per_parent=2000, query_cohorts=list(QUERY_COHORTS),
        train_parents=3, test_parents=5, full_fit_optimizer_updates=6670,
        fresh_training=True, probability_merging=False, six_class_results_preserved=True,
        new_simulation_required=False, new_genealogy_inference_required=False,
        feature_reextraction_required=True, primary_cohort="present_day",
        sampling_note="Non-BAA subset of existing observed focal registries. Note3.1 does not specify counts.",
        published_four_class_counts_reproduced=False, exact_published_numbers_reproduced=False,
        treeloom_four_path_training_included=False,
        comparison_note="AP four-class baseline only. TreeLoom needs its own matched four-output fit before a comparison claim.",
        max_new_output_bytes=sum(t["max_output_bytes"] for t in tasks),
        resources="CPU tasks using existing runtimes. Limits are conservative execution bounds, not measured peaks.")

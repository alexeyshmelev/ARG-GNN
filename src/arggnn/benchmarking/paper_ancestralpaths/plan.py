"""Inert DAG construction for ROOT admission; never schedules or starts a job."""
from __future__ import annotations
from . import CONDITION
from .model import model_spec
from .. import european_native_models as io

MODULE = "arggnn.benchmarking.paper_ancestralpaths.workflow"


def build(reuse_ref, *, output_root, treeloom_prediction_task):
    reuse = io._read(reuse_ref, observed=True)
    io._need(reuse["schema"] == "paper_ancestralpaths_frozen476_reuse_v1",
             "Frozen476 reuse manifest required")
    cpu_python = ".runtime/conda/envs/ancestralpaths_cpu_v1/bin/python3.11"
    common = dict(write_headroom_bytes=4 * 2**20)
    fit_id = "paper_ap_cnn__all_train__fit"
    root = io._path(output_root)
    io._need(root.is_relative_to(io.ROOT / "experiments/results"),
             "Fresh proposed benchmark output root required")

    def task(identity, stage, arguments, dependencies, limits, python=cpu_python):
        return dict(id=identity, resource="cpu", dependencies=dependencies,
            output_directory=str(root / identity), factory_module=MODULE,
            factory_function="make_call", factory_arguments=dict(stage=stage,
                arguments=dict(arguments, **common), python=python), **limits)
    fit_limits = dict(max_seconds=7200, max_rss_bytes=4 * 2**30, max_output_bytes=32 * 2**20)
    fit_task = task(fit_id, "fit", dict(plan_ref=reuse["sampling_plan_ref"],
        training_products=reuse["training_products"], seed=20261024, layout="portable_nhwc",
        device="cpu", threads=4, workers=1), [], fit_limits)
    tasks = [fit_task]
    predictions = {CONDITION: {}, "ancestralpaths_dense_control": reuse["dense_control"]["test_predictions"],
                   "treeloom": {}}
    for parent, feature in reuse["test_features"].items():
        identity = "paper_ap_cnn__" + parent + "__predict"
        tasks.append(task(identity, "predict", dict(fit_ref={"$task":fit_id},
            features_ref=feature, device="cpu", threads=4), [fit_id],
            dict(max_seconds=900, max_rss_bytes=4 * 2**30, max_output_bytes=16 * 2**20)))
        predictions[CONDITION][parent] = {"$task":identity}
        predictions["treeloom"][parent] = {"$task":treeloom_prediction_task,
                                          "path":["predictions", parent]}
    tasks.append(task("paper_ap_cnn__all_test__score", "score",
        dict(plan_ref=reuse["sampling_plan_ref"], predictions=predictions,
            test_targets=reuse["test_target_references"], expected_methods=list(predictions),
            max_truth_uncompressed_bytes=16 * 2**30),
        [t["id"] for t in tasks[1:]] + [treeloom_prediction_task],
        dict(max_seconds=1800, max_rss_bytes=32 * 2**30, max_output_bytes=16 * 2**20)))
    return dict(schema="paper_ancestralpaths_plan_v1", plan_only=True, admitted=False,
        condition=CONDITION, reuse_ref=reuse_ref, model_spec=model_spec(), tasks=tasks,
        full_fit_updates=8335, train_rows=50000, test_parents=5,
        new_generation_required=False, new_inference_required=False,
        dense_control_preserved=True,
        source_runtime="qualified existing TFCPU2.15.1/Keras2.15.0",
        resources="Operational caps, not measured full-fit peaks; no GPU dependency",
        treeloom_observation_focal_interfaces=reuse["treeloom_observation_focal_interfaces"])

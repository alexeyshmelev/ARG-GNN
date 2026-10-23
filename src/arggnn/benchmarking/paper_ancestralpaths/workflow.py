"""Admitted fit/predict/score entrypoints and inert factory for the historical CNN."""
from __future__ import annotations

import math
from pathlib import Path

from . import CONDITION, data, metrics
from .model import AUTHOR, FLATTEN_SOURCE, build_cnn, load_runtime, model_spec
from .. import european_native_models as io
from .. import european_sampling as sampling

MODULE = "arggnn.benchmarking.paper_ancestralpaths.workflow"
STAGES = ("fit", "predict", "score")


def fit(plan_ref, training_products, output_directory, *, tensorflow, seed, layout,
        workers, guard, execution_check, write_headroom_bytes):
    import numpy as np
    io._ready(guard, execution_check, write_headroom_bytes)
    io._need(type(seed) is int and 0 <= seed < 2**32 and type(workers) is int and workers > 0,
             "Fresh declared seed and finite worker count required")
    x, y, ids, cohorts = data.load_training(plan_ref, training_products)
    tensorflow.keras.backend.clear_session()
    tensorflow.keras.utils.set_random_seed(seed)
    network = build_cnn(tensorflow, layout=layout)

    class Check(tensorflow.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None):
            execution_check(); guard(0)
        def on_epoch_end(self, epoch, logs=None):
            execution_check(); guard(0)

    history = network.fit(x, tensorflow.keras.utils.to_categorical(y, 6),
        epochs=5, batch_size=30, shuffle=True, verbose=2,
        workers=workers, use_multiprocessing=False, max_queue_size=2,
        callbacks=[Check()])
    updates = int(network.optimizer.iterations.numpy())
    io._need(updates == 5 * math.ceil(50000 / 30) == 8335
        and len(history.history["loss"]) == 5
        and all(np.isfinite(v).all() for v in history.history.values()),
        "All five finite epochs and 8335 fresh Adam updates required")
    output = io._new_output(output_directory)
    guard(4 * 2**20)
    checkpoint = output / "historical_cnn_epoch5.h5"
    network.save(checkpoint)
    identity_ref = io._npz(output / "training_example_ids.npz",
                           dict(example_ids=np.asarray(ids)), guard)
    return io._json(output / "fit.json", dict(
        schema="paper_ancestralpaths_historical_cnn_fit_v1", condition=CONDITION,
        plan_ref=plan_ref, training_products=training_products,
        training_parent_ids=data.checked_plan(plan_ref)["train_parent_ids"],
        checkpoint_ref=io._ref(checkpoint), training_example_ids_ref=identity_ref,
        native_path_names=list(sampling.NATIVE_PATHS), training_rows=50000,
        rows_per_cohort=cohorts, seed=seed, layout=layout, workers=workers,
        parameters=network.count_params(), optimizer_updates=updates,
        epochs=5, batch_size=30, shuffle=True, checkpoint_selection="final_epoch5",
        validation_selection=False, history=history.history, model_spec=model_spec(),
        reconstruction_choices="paper counts and frozen observation-only sampling",
        exact_published_numerical_reproduction_claimed=False,
        dense_control_replaced=False), guard)


def predict(fit_ref, features_ref, output_directory, *, tensorflow,
            guard, execution_check, write_headroom_bytes):
    import numpy as np
    io._ready(guard, execution_check, write_headroom_bytes)
    fitted = io._read(fit_ref, observed=True)
    io._need(fitted["schema"] == "paper_ancestralpaths_historical_cnn_fit_v1"
        and fitted["condition"] == CONDITION and fitted["parameters"] == 19078
        and fitted["epochs"] == 5 and fitted["batch_size"] == 30
        and fitted["optimizer_updates"] == 8335
        and fitted["checkpoint_selection"] == "final_epoch5",
        "Fresh full historical-CNN final-epoch fit required")
    plan = data.checked_plan(fitted["plan_ref"])
    feature = io._read(features_ref, observed=True)
    parent = feature["parent_id"]
    io._need(parent in plan["test_parent_ids"], "Independent frozen TEST parent required")
    feature, registry, x = data.load_feature(features_ref, plan, parent=parent, split="test")
    network = tensorflow.keras.models.load_model(io._checked(fitted["checkpoint_ref"], observed=True))
    io._need(network.count_params() == 19078 and network.input_shape == (None, 1, 5, 9),
             "Saved historical CNN topology required")
    probabilities = np.empty((len(x), 6), dtype=np.float32)
    for start in range(0, len(x), 128):
        execution_check(); guard(0)
        probabilities[start:start+128] = network(x[start:start+128, None, :, :], training=False).numpy()
    # Synthetic labels only validate the probability domain; no TEST truth access.
    metrics.metrics(probabilities, np.zeros(len(x), dtype=np.int64), sampling.NATIVE_PATHS)
    output = io._new_output(output_directory)
    arrays = dict(probabilities=probabilities, example_indices=np.arange(len(x), dtype=np.int64),
        example_ids=np.asarray([parent + ":" + str(i) for i in range(len(x))]))
    pred_ref = io._npz(output / "predictions.npz", arrays, guard)
    return io._json(output / "predictions.json", dict(
        schema="ap_paper_native_predictions_v1", condition=CONDITION,
        parent_id=parent, split="test", rows=len(x), registry_ref=feature["registry_ref"],
        features_ref=features_ref, fit_ref=fit_ref, predictions_ref=pred_ref,
        native_path_names=list(sampling.NATIVE_PATHS), query_truth_read=False,
        calibrated_probabilities_claimed=False, denominator_reduction=False), guard)


def score(plan_ref, predictions, test_targets, expected_methods, output_directory, *,
          guard, execution_check, write_headroom_bytes, max_truth_uncompressed_bytes):
    """Strict all-method/five-parent barrier; truth paths cannot enter prediction loops."""
    import numpy as np
    io._ready(guard, execution_check, write_headroom_bytes)
    plan = data.checked_plan(plan_ref)
    parents = plan["test_parent_ids"]
    io._need(isinstance(expected_methods, list) and len(expected_methods) >= 2
        and len(set(expected_methods)) == len(expected_methods)
        and CONDITION in expected_methods and "treeloom" in expected_methods
        and set(predictions) == set(expected_methods) and set(test_targets) == set(parents)
        and all(set(predictions[m]) == set(parents) for m in expected_methods),
        "All predeclared competing conditions and all five TEST parents required")
    sealed, registries = {}, {}
    # Phase 1 reads only observed predictions/registry. It does not inspect test_targets.
    for parent in parents:
        sealed[parent], registry_ref = {}, None
        for method in expected_methods:
            product = io._read(predictions[method][parent], observed=True)
            io._need(product["parent_id"] == parent and product["split"] == "test"
                and product["rows"] == 10000
                and product.get("query_truth_read", product.get("query_ancestry_truth_opened")) is False
                and product["native_path_names"] == list(sampling.NATIVE_PATHS),
                "Complete truth-free native-six predictions required")
            if method == CONDITION:
                io._need(product.get("condition") == CONDITION, "CNN identity cannot alias Dense control")
            current_ref = product["registry_ref"]
            io._need(registry_ref is None or current_ref == registry_ref,
                     "Identical immutable focal support across methods required")
            registry_ref = current_ref
            a = io._arrays(product["predictions_ref"], cap=16 * 2**20, observed=True)
            io._need(np.array_equal(a["example_indices"], np.arange(10000)),
                     "All canonical prediction identities required")
            if "example_ids" in a:
                io._need(np.array_equal(a["example_ids"],
                    np.asarray([parent + ":" + str(i) for i in range(10000)])),
                    "Global parent/example IDs must remain exact")
            metrics.metrics(a["probabilities"], np.zeros(10000, dtype=np.int64), sampling.NATIVE_PATHS)
            sealed[parent][method] = a["probabilities"]
        registry = io._read(registry_ref, observed=True)
        io._need(registry["parent_id"] == parent and registry["split"] == "test"
            and registry["labels_present"] is False and registry["examples"] == 10000
            and registry["cohort_quotas"] == plan["parents"][parent]["cohort_quotas"],
            "Frozen complete TEST support without denominator revision required")
        sampling.registry_arrays(registry)
        registries[parent] = registry
    # Phase 2 begins only after every required probability file has passed Phase 1.
    rows = {m: {} for m in expected_methods}
    for parent in parents:
        execution_check(); guard(0)
        refs = test_targets[parent]
        generated, schema = io._read(refs["generation_ref"]), io._read(refs["target_schema_ref"])
        io._need(generated["parent_id"] == schema["parent_id"] == parent
            and generated["split"] == schema["split"] == "test"
            and generated["files"]["truth/full_truth_intervals.npz"]["sha256"] == refs["truth_ref"]["sha256"]
            and generated["files"]["truth/target_schema.json"]["sha256"] == refs["target_schema_ref"]["sha256"],
            "Same-parent producer-authenticated external TEST truth required")
        base = io._checked(refs["generation_ref"]).parent
        io._need(io._path(refs["truth_ref"]["path"]) == base / "truth/full_truth_intervals.npz"
            and io._path(refs["target_schema_ref"]["path"]) == base / "truth/target_schema.json",
            "Exact same-parent truth namespace required")
        truth = io._arrays(refs["truth_ref"], cap=max_truth_uncompressed_bytes)
        y = sampling.native_truth_labels(registries[parent], schema, truth)
        del truth
        records = sorted(registries[parent]["records"], key=lambda r: r["example_index"])
        cohort = np.asarray([r["cohort"] for r in records])
        for method in expected_methods:
            p = sealed[parent][method]
            rows[method][parent] = dict(overall=metrics.endpoints(p, y), cohorts={
                c: metrics.endpoints(p[cohort == c], y[cohort == c]) for c in sampling.COHORTS})
    summary = {}
    for method in expected_methods:
        summary[method] = {}
        for group in ("overall", *sampling.COHORTS):
            summary[method][group] = {}
            for endpoint in ("native6", "derived_C4"):
                subset = [rows[method][p]["overall"] if group == "overall"
                          else rows[method][p]["cohorts"][group] for p in parents]
                summary[method][group][endpoint] = {name: metrics.parent_summary([
                    v[endpoint][name] for v in subset]) for name in
                    ("accuracy", "macro_F1", "Brier", "raw_NLL", "mean_predictive_entropy_nats",
                     "mean_max_probability", "ECE_10_equal_width")}
    paired = {}
    for method in expected_methods:
        if method == "treeloom":
            continue
        paired[method] = {}
        for group in ("overall", *sampling.COHORTS):
            paired[method][group] = {}
            for endpoint in ("native6", "derived_C4"):
                delta = []
                for parent in parents:
                    a = rows["treeloom"][parent]
                    b = rows[method][parent]
                    if group != "overall":
                        a, b = a["cohorts"][group], b["cohorts"][group]
                    else:
                        a, b = a["overall"], b["overall"]
                    delta.append(a[endpoint]["accuracy"] - b[endpoint]["accuracy"])
                paired[method][group][endpoint] = dict(
                    per_parent=delta, direction="TreeLoom minus " + method,
                    unit="fraction", **metrics.parent_summary(delta))
    output = io._new_output(output_directory)
    return io._json(output / "comparison.json", dict(
        schema="paper_ancestralpaths_native6_and_derived_C4_comparison_v1",
        plan_ref=plan_ref, expected_methods=expected_methods, predictions=predictions,
        test_targets=test_targets, rows=rows, summary=summary,
        paired_accuracy_differences=paired, primary_endpoint="native6",
        derived_endpoint="C4 probability sum; native-six model retained",
        units="accuracy, F1, ECE and differences are fractions; NLL and entropy are nats",
        uncertainty="equal independent parent mean and sample SD, ddof=1",
        raw_probability_uncertainty_is_not_a_calibration_claim=True,
        all_method_all_parent_prediction_barrier_before_truth=True,
        test_parents=5, records_per_parent=10000, records_per_cohort_per_parent=2000,
        exact_published_numerical_reproduction_claimed=False), guard)


def execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    args = dict(spec["arguments"])
    stage = spec["stage"]
    io._need(stage in STAGES, "Explicit historical CNN stage required")
    common = dict(guard=resource_check, execution_check=execution_check)
    io._ready(resource_check, execution_check, args["write_headroom_bytes"])
    if stage == "score":
        return score(output_directory=spec["output_directory"], **args, **common)
    tf = load_runtime(spec["runtime_prefix"], device=args.pop("device"), threads=args.pop("threads"))
    fn = fit if stage == "fit" else predict
    return fn(output_directory=spec["output_directory"], tensorflow=tf, **args, **common)


def make_call(stage, arguments, output_directory, *, python):
    io._need(stage in STAGES and isinstance(arguments, dict), "Explicit inert stage arguments required")
    executable = io._path(python)
    io._need(executable.is_file() and executable.is_relative_to(io.ROOT / ".runtime"),
             "Provisioned project runtime required")
    names = ["__init__.py", "data.py", "metrics.py", "model.py", "workflow.py"]
    return dict(module=MODULE, function="execute_stage", python=str(executable),
        spec=dict(stage=stage, arguments=arguments, output_directory=str(io._path(output_directory) / "native"),
                  runtime_prefix=str(executable.parent.parent)),
        source_refs=[io._ref(Path(__file__).parent / name) for name in names] +
                    [AUTHOR, io._ref(io.ROOT / FLATTEN_SOURCE),
                     io._ref(io.ROOT / "src/arggnn/benchmarking/european_native_models.py"),
                     io._ref(io.ROOT / "src/arggnn/benchmarking/european_sampling.py"),
                     io._ref(io.ROOT / "src/arggnn/benchmarking/european_execution.py"),
                     io._ref(io.ROOT / "src/arggnn/benchmarking/european_recipes.py")])

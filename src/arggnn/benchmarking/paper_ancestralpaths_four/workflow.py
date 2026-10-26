"""Separate four-path preparation, fresh training, prediction and scoring.

The six-path benchmark and author source remain untouched. Every stage uses
original focal IDs. BAA removal is decided from observed cohorts before labels
are opened. Full tasks use the existing bounded campaign runtime.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import CONDITION, PATH_NAMES, QUERY_COHORTS, data, features, model
from .. import european_native_models as io, european_sampling as sampling
from ..paper_ancestralpaths import data as original_data, metrics
from ...ancestralpaths_native_bridge import ObservedPanel

MODULE = "arggnn.benchmarking.paper_ancestralpaths_four.workflow"
STAGES = ("prepare", "fit", "predict", "score")


def selected_records(registry, plan):
    """Authenticate the original registry, then retain non-BAA rows by cohort."""
    sampling.registry_arrays(registry)
    parent = registry["parent_id"]
    expected = plan["parents"][parent]
    io._need(registry["split"] == expected["split"]
             and registry["cohort_quotas"] == expected["cohort_quotas"]
             and registry["examples"] == sum(expected["cohort_quotas"].values())
             and registry["labels_present"] is False
             and registry["query_truth_read"] is False,
             "Original observed-only frozen registry required")
    records = sorted(registry["records"], key=lambda row: row["example_index"])
    io._need(dict(Counter(r["cohort"] for r in records)) == expected["cohort_quotas"],
             "Every original cohort quota must be represented")
    indices = data.non_baa_row_indices([r["cohort"] for r in records])
    return [records[i] for i in indices], indices


def extract_rows(tree_sequence, panel, records, *, check):
    """Group sampled queries by marginal tree and preserve canonical row order."""
    import numpy as np
    io._need(bool(records), "Nonempty four-path focal support required")
    x = np.empty((len(records), 5, 8), dtype=np.float32)
    ordered = sorted(enumerate(records), key=lambda item: item[1]["position_local_bp"])
    tree = tree_sequence.first()
    cursor = 0
    while cursor < len(ordered):
        check()
        position = float(ordered[cursor][1]["position_local_bp"])
        io._need(0 <= position < tree_sequence.sequence_length, "Focal position outside observed tree")
        tree.seek(position)
        end = cursor + 1
        while end < len(ordered) and ordered[end][1]["position_local_bp"] < tree.interval.right:
            end += 1
        group = ordered[cursor:end]
        queries = list(dict.fromkeys(int(r["query_node_id"]) for _, r in group))
        values = features.extract_four_path_features(tree, queries, panel, dtype=np.float64)
        by_node = {node: i for i, node in enumerate(queries)}
        for row, record in group:
            x[row] = values[by_node[int(record["query_node_id"])]]
        cursor = end
    io._need(np.isfinite(x).all(), "Every recomputed feature must be finite")
    return x


def prepare(source_plan_ref, registry_ref, inference_ref, output_directory, *,
            guard, execution_check, write_headroom_bytes):
    import numpy as np
    import tskit
    io._ready(guard, execution_check, write_headroom_bytes)
    plan = original_data.checked_plan(source_plan_ref)
    registry = io._read(registry_ref, observed=True)
    records, indices = selected_records(registry, plan)
    inferred = io._read(inference_ref, observed=True)
    io._need(registry["observation_ref"] == inference_ref
             and inferred["parent_id"] == registry["parent_id"]
             and inferred["split"] == registry["split"]
             and inferred["truth_or_oracle_read"] is False,
             "Original same-parent observed genealogy required")
    rows = io._read(inferred["observed_sample_manifest"], observed=True)
    panel = ObservedPanel(tuple(r["sample_id"] for r in rows[::2]),
                          tuple(r["observed_node_id"] for r in rows),
                          {r["observed_node_id"]: r["observed_cohort"] for r in rows})
    by_node = {r["observed_node_id"]: r for r in rows}
    for record in records:
        observed = by_node[record["query_node_id"]]
        io._need(observed["sample_id"] == record["sample_id"]
                 and observed["observed_cohort"] == record["cohort"]
                 and observed["sample_index"] == record["sample_index"]
                 and observed["haplotype"] == record["haplotype"]
                 and observed["inferred_tree_node_id"] == record["query_node_id"],
                 "Focal sample, haplotype and inferred node must match the observation manifest")
    ts = tskit.load(io._checked(inferred["inferred_tree"], observed=True))
    io._need(np.array_equal(ts.samples(), panel.haplotype_node_ids),
             "Observed sample order must match the original inferred tree")
    offsets = {r["position_original_bp"] - r["position_local_bp"] for r in records}
    io._need(len(offsets) == 1, "One original-to-local coordinate offset required")
    offset = next(iter(offsets))
    io._need(0 <= offset and offset + ts.sequence_length <= plan["sequence_length_bp"],
             "Inferred coverage must stay inside the original simulated chromosome")
    io._need(all(0 <= r["native_site_id"] < ts.num_sites
                 and ts.site(r["native_site_id"]).position == r["position_local_bp"] for r in records),
             "Every focal coordinate must match its original observed SNP")
    x = extract_rows(ts, panel, records, check=lambda: (execution_check(), guard(0)))
    output = io._new_output(output_directory)
    arrays = io._npz(output / "features.npz", dict(features=x, example_indices=indices), guard)
    return io._json(output / "features.json", dict(
        schema="ap_four_features_v1", condition=CONDITION,
        source_plan_ref=source_plan_ref, registry_ref=registry_ref, inference_ref=inference_ref,
        parent_id=registry["parent_id"], split=registry["split"], rows=len(records),
        rows_per_cohort=dict(Counter(r["cohort"] for r in records)),
        arrays_ref=arrays, feature_shape=[5, 8], feature_spec=features.feature_spec(),
        query_truth_read=False, row_selection="original non-BAA observed cohort rows",
        source_example_indices_preserved=True), guard)


def load_features(feature_ref, source_plan_ref, *, split):
    import numpy as np
    plan = original_data.checked_plan(source_plan_ref)
    product = io._read(feature_ref, observed=True)
    io._need(product["schema"] == "ap_four_features_v1"
             and product["condition"] == CONDITION and product["source_plan_ref"] == source_plan_ref
             and product["split"] == split and product["feature_shape"] == [5, 8]
             and product["feature_spec"] == features.feature_spec()
             and product["query_truth_read"] is False,
             "Recomputed four-path features with exact experiment definition required")
    registry = io._read(product["registry_ref"], observed=True)
    records, indices = selected_records(registry, plan)
    io._need(registry["parent_id"] == product["parent_id"]
             and registry["split"] == split
             and registry["observation_ref"] == product["inference_ref"]
             and product["rows"] == len(records)
             and product["rows_per_cohort"] == dict(Counter(r["cohort"] for r in records)),
             "Complete original same-parent support required")
    arrays = io._arrays(product["arrays_ref"], cap=32 * 2**20, observed=True)
    x = arrays["features"]
    io._need(x.shape == (len(records), 5, 8) and x.dtype == np.float32
             and np.isfinite(x).all() and np.array_equal(arrays["example_indices"], indices),
             "All finite four-path features in original focal order required")
    return product, registry, records, indices, x


def fit(source_plan_ref, training_products, seed, layout, workers, output_directory, *,
        tensorflow, guard, execution_check, write_headroom_bytes):
    import numpy as np
    io._ready(guard, execution_check, write_headroom_bytes)
    plan = original_data.checked_plan(source_plan_ref)
    io._need(set(training_products) == set(plan["train_parent_ids"])
             and type(seed) is int and 0 <= seed < 2**32
             and type(workers) is int and workers > 0,
             "All three independent TRAIN parents, explicit seed and workers required")
    xs, ys, identities, cohorts = [], [], [], Counter()
    for parent in plan["train_parent_ids"]:
        execution_check()
        refs = training_products[parent]
        feature, registry, records, indices, x = load_features(refs["features_ref"], source_plan_ref, split="train")
        io._need(feature["parent_id"] == parent, "TRAIN mapping must match parent")
        targets = io._read(refs["source_targets_ref"])
        original_feature = io._read(targets["features_ref"], observed=True)
        io._need(targets["schema"] == "ap_paper_TRAIN_native_targets_v1"
                 and targets["parent_id"] == parent and targets["split"] == "train"
                 and targets["registry_ref"] == feature["registry_ref"]
                 and original_feature["registry_ref"] == feature["registry_ref"]
                 and original_feature["inference_ref"] == feature["inference_ref"]
                 and targets["native_path_names"] == list(sampling.NATIVE_PATHS)
                 and targets["no_label_dependent_selection"] is True,
                 "Original TRAIN-only supervision on identical observed rows required")
        full_y = io._arrays(targets["arrays_ref"], cap=2 * 2**20)["native_labels_0based"]
        io._need(full_y.shape == (registry["examples"],), "Complete original TRAIN labels required")
        y = data.four_path_targets(full_y[indices], [r["cohort"] for r in records])
        xs.append(x[:, None]); ys.append(y)
        identities.extend(parent + ":" + str(i) for i in indices)
        cohorts.update(r["cohort"] for r in records)
    io._need(len(identities) == len(set(identities)) == 40000
             and dict(cohorts) == {c: 10000 for c in QUERY_COHORTS},
             "All 40000 original non-BAA TRAIN rows required")
    x, y = np.concatenate(xs), np.concatenate(ys)
    tensorflow.keras.backend.clear_session()
    tensorflow.keras.utils.set_random_seed(seed)
    network = model.build_cnn(tensorflow, layout=layout)

    class Check(tensorflow.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None):
            execution_check(); guard(0)
        def on_epoch_end(self, epoch, logs=None):
            execution_check(); guard(0)

    history = network.fit(x, tensorflow.keras.utils.to_categorical(y, 4), epochs=5,
                          batch_size=30, shuffle=True, verbose=2, workers=workers,
                          use_multiprocessing=False, max_queue_size=2, callbacks=[Check()])
    updates = int(network.optimizer.iterations.numpy())
    io._need(updates == model.expected_optimizer_updates(len(x)) == 6670
             and len(history.history["loss"]) == 5
             and all(np.isfinite(v).all() for v in history.history.values()),
             "Five complete finite epochs and 6670 fresh optimizer updates required")
    output = io._new_output(output_directory)
    guard(4 * 2**20)
    checkpoint = output / "four_path_cnn_epoch5.h5"
    network.save(checkpoint)
    ids_ref = io._npz(output / "training_example_ids.npz", dict(example_ids=np.asarray(identities)), guard)
    return io._json(output / "fit.json", dict(schema="ap_four_fit_v1", condition=CONDITION,
        source_plan_ref=source_plan_ref, training_products=training_products,
        checkpoint_ref=io._ref(checkpoint), training_example_ids_ref=ids_ref,
        seed=seed, layout=layout, workers=workers, rows_per_cohort=dict(cohorts), training_rows=len(x),
        optimizer_updates=updates, epochs=5, batch_size=30, checkpoint_selection="final_epoch5",
        history=history.history, model_spec=model.model_spec(), class_names=list(PATH_NAMES),
        six_class_checkpoint_read=False, published_four_class_counts_reproduced=False), guard)


def predict(fit_ref, features_ref, output_directory, *, tensorflow,
            guard, execution_check, write_headroom_bytes):
    import numpy as np
    io._ready(guard, execution_check, write_headroom_bytes)
    fitted = io._read(fit_ref, observed=True)
    io._need(fitted["schema"] == "ap_four_fit_v1" and fitted["condition"] == CONDITION
             and fitted["model_spec"] == model.model_spec() and fitted["optimizer_updates"] == 6670
             and fitted["epochs"] == 5 and fitted["training_rows"] == 40000
             and fitted["checkpoint_selection"] == "final_epoch5",
             "Fresh complete four-output model required, never a merged six-output fit")
    product, registry, records, indices, x = load_features(features_ref, fitted["source_plan_ref"], split="test")
    network = tensorflow.keras.models.load_model(io._checked(fitted["checkpoint_ref"], observed=True))
    io._need(network.input_shape == (None, 1, 5, 8) and network.output_shape == (None, 4)
             and network.count_params() == model.PARAMETERS,
             "Actual checkpoint must have four-output topology")
    probabilities = np.empty((len(x), 4), dtype=np.float32)
    for start in range(0, len(x), 128):
        execution_check(); guard(0)
        probabilities[start:start+128] = network(x[start:start+128, None], training=False).numpy()
    metrics.metrics(probabilities, np.zeros(len(x), dtype=np.int64), PATH_NAMES)
    parent = product["parent_id"]
    output = io._new_output(output_directory)
    arrays = io._npz(output / "predictions.npz", dict(probabilities=probabilities,
        example_indices=indices, example_ids=np.asarray([parent + ":" + str(i) for i in indices])), guard)
    return io._json(output / "predictions.json", dict(schema="ap_four_predictions_v1",
        condition=CONDITION, parent_id=parent, split="test", rows=len(x), class_names=list(PATH_NAMES),
        source_plan_ref=fitted["source_plan_ref"], registry_ref=product["registry_ref"],
        features_ref=features_ref, fit_ref=fit_ref, predictions_ref=arrays,
        query_truth_read=False, probability_merging=False), guard)


def score(source_plan_ref, predictions, test_targets, expected_methods, output_directory, *,
          guard, execution_check, write_headroom_bytes, max_truth_uncompressed_bytes):
    import numpy as np
    io._ready(guard, execution_check, write_headroom_bytes)
    plan = original_data.checked_plan(source_plan_ref)
    parents = plan["test_parent_ids"]
    io._need(bool(expected_methods) and len(set(expected_methods)) == len(expected_methods)
             and set(predictions) == set(expected_methods) and set(test_targets) == set(parents),
             "All declared methods and all five independent TEST parents required")
    sealed, registries, selected = {}, {}, {}
    # Validate complete predictions before any TEST truth file is read.
    for method in expected_methods:
        io._need(set(predictions[method]) == set(parents), "No partial TEST scoring")
        sealed[method] = {}
        for parent in parents:
            p = io._read(predictions[method][parent], observed=True)
            io._need(p["schema"] == "ap_four_predictions_v1" and p["condition"] == method
                     and p["parent_id"] == parent and p["split"] == "test"
                     and p["source_plan_ref"] == source_plan_ref and p["class_names"] == list(PATH_NAMES)
                     and p["query_truth_read"] is False and p["probability_merging"] is False,
                     "Direct four-path predictions on the common experiment required")
            registry = io._read(p["registry_ref"], observed=True)
            records, indices = selected_records(registry, plan)
            io._need(registry["parent_id"] == parent and registry["split"] == "test"
                     and len(records) == p["rows"] == 8000,
                     "All 8000 observed non-BAA TEST rows required")
            if parent in registries:
                io._need(registries[parent] == p["registry_ref"], "Methods must use identical focal registries")
            arrays = io._arrays(p["predictions_ref"], cap=16 * 2**20, observed=True)
            io._need(np.array_equal(arrays["example_indices"], indices)
                     and np.array_equal(arrays["example_ids"], [parent + ":" + str(i) for i in indices]),
                     "Exact original focal identities required")
            metrics.metrics(arrays["probabilities"], np.zeros(8000, dtype=np.int64), PATH_NAMES)
            sealed[method][parent] = arrays["probabilities"]
            registries[parent], selected[parent] = p["registry_ref"], (records, indices)
    results = {method: {} for method in expected_methods}
    for parent in parents:
        execution_check(); guard(0)
        refs = test_targets[parent]
        generated, schema = io._read(refs["generation_ref"]), io._read(refs["target_schema_ref"])
        base = io._checked(refs["generation_ref"]).parent
        io._need(generated["parent_id"] == schema["parent_id"] == parent
                 and generated["split"] == schema["split"] == "test"
                 and generated["files"]["truth/full_truth_intervals.npz"]["sha256"] == refs["truth_ref"]["sha256"]
                 and generated["files"]["truth/target_schema.json"]["sha256"] == refs["target_schema_ref"]["sha256"]
                 and io._path(refs["truth_ref"]["path"]) == base / "truth/full_truth_intervals.npz"
                 and io._path(refs["target_schema_ref"]["path"]) == base / "truth/target_schema.json",
                 "Producer-bound same-parent TEST truth required")
        registry = io._read(registries[parent], observed=True)
        truth = io._arrays(refs["truth_ref"], cap=max_truth_uncompressed_bytes)
        all_y = sampling.native_truth_labels(registry, schema, truth)
        del truth
        records, indices = selected[parent]
        cohort = np.asarray([r["cohort"] for r in records])
        y = data.four_path_targets(all_y[indices], cohort)
        for method in expected_methods:
            p = sealed[method][parent]
            results[method][parent] = {"overall": metrics.metrics(p, y, PATH_NAMES)}
            for c in QUERY_COHORTS:
                mask = cohort == c
                results[method][parent][c] = metrics.metrics(p[mask], y[mask], PATH_NAMES)
    summaries, pooled = {}, {}
    for method in expected_methods:
        summaries[method], pooled[method] = {}, {}
        for group in ("overall", *QUERY_COHORTS):
            rows = [results[method][parent][group] for parent in parents]
            summaries[method][group] = {name: metrics.parent_summary([r[name] for r in rows])
                for name in ("accuracy", "macro_F1", "Brier", "raw_NLL")}
            counts = np.sum([r["confusion"] for r in rows], axis=0)
            denominators = counts.sum(axis=0)
            precision = np.divide(counts.T, denominators[:, None], out=np.zeros((4, 4)),
                                  where=denominators[:, None] != 0)
            pooled[method][group] = dict(confusion_truth_rows=counts.tolist(),
                precision_predicted_rows=precision.tolist(), class_names=list(PATH_NAMES),
                predicted_class_counts=denominators.tolist())
    output = io._new_output(output_directory)
    return io._json(output / "comparison.json", dict(schema="ap_four_comparison_v1",
        source_plan_ref=source_plan_ref, expected_methods=expected_methods, predictions=predictions,
        test_targets=test_targets, per_parent=results, summaries=summaries,
        pooled_confusion=pooled, primary_cohort="present_day", primary_metric="accuracy",
        uncertainty="sample SD across five independent simulated test parents, ddof=1",
        training_seed_uncertainty_measured=False, probability_merging=False,
        exact_published_numbers_reproduced=False,
        benchmark_scope="separately trained four-path classifier excluding BAA distributions and labels"), guard)


def execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    args = dict(spec["arguments"])
    stage = spec["stage"]
    io._need(stage in STAGES, "Known four-path stage required")
    common = dict(guard=resource_check, execution_check=execution_check)
    io._ready(resource_check, execution_check, args["write_headroom_bytes"])
    if stage in ("fit", "predict"):
        tf = model.load_runtime(spec["runtime_prefix"], device=args.pop("device"), threads=args.pop("threads"))
        common["tensorflow"] = tf
    function = {"prepare": prepare, "fit": fit, "predict": predict, "score": score}[stage]
    return function(output_directory=spec["output_directory"], **args, **common)


def make_call(stage, arguments, output_directory, *, python):
    io._need(stage in STAGES and isinstance(arguments, dict), "Known stage and explicit arguments required")
    executable = io._path(python)
    io._need(executable.is_file() and executable.is_relative_to(io.ROOT / ".runtime"), "Project runtime required")
    names = ["__init__.py", "data.py", "features.py", "model.py", "workflow.py"]
    reused = ["src/arggnn/ancestralpaths_native_bridge.py", "src/arggnn/benchmarking/european_native_models.py",
              "src/arggnn/benchmarking/european_sampling.py", "src/arggnn/benchmarking/european_execution.py",
              "src/arggnn/benchmarking/european_recipes.py", "src/arggnn/benchmarking/paper_ancestralpaths/data.py",
              "src/arggnn/benchmarking/paper_ancestralpaths/model.py", "src/arggnn/benchmarking/paper_ancestralpaths/metrics.py"]
    return dict(module=MODULE, function="execute_stage", python=str(executable),
        spec=dict(stage=stage, arguments=arguments, output_directory=str(io._path(output_directory) / "native"),
                  runtime_prefix=str(executable.parent.parent)),
        source_refs=[io._ref(Path(__file__).parent / name) for name in names]
                    + [io._ref(io.ROOT / name) for name in reused]
                    + [model.AUTHOR, io._ref(io.ROOT / model.FLATTEN_SOURCE)])

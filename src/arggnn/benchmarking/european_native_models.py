"""Executable paper-sized native AP feature, target, fit and prediction stages.

Imports are inert. ROOT supplies its admitted bounded supervisor and current
execution/resource callbacks. The author Dense/compile expressions are executed
unchanged from the pinned training.py AST. Model selection is final epoch five.
No TEST truth resolver or scientific scoring is present.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import zipfile

from . import european_execution as stages
from . import european_sampling as sampling

ROOT = stages.recipes.ROOT
AUTHOR_TRAIN = {
    "path": ".runtime/author_sources/ancestralpaths_77db3cfae100/training.py",
    "sha256": "13b4f30ab0df75fc461b764ae2cdbd1fdc071ff45200e6401e84faa565508e4f"}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _path(value, *, observed=False):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    _need(".." not in path.parts and path.is_relative_to(ROOT)
          and path.resolve() == path, "Canonical repository path required")
    _need(not observed or not any("truth" in p.lower() or "oracle" in p.lower()
                                  for p in path.parts), "Observed path cannot resolve truth")
    return path


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ref(path):
    path = _path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=_sha(path))


def _checked(record, *, observed=False):
    _need(isinstance(record, dict) and set(record) == {"path", "sha256"},
          "Exact immutable file reference required")
    path = _path(record["path"], observed=observed)
    _need(path.is_file() and not path.is_symlink() and _sha(path) == record["sha256"],
          "Source-bound file changed")
    return path


def _read(record, *, observed=False, cap=64 * 2**20):
    path = _checked(record, observed=observed)
    _need(path.stat().st_size <= cap, "Explicit bounded JSON metadata required")
    return json.loads(path.read_text())


def _ready(guard, execution_check, headroom):
    _need(callable(guard) and callable(execution_check)
          and type(headroom) is int and headroom > 0,
          "Current ROOT execution/resource callbacks and write headroom required")
    execution_check()
    guard(headroom)


def _new_output(value):
    path = _path(value)
    path.mkdir(exist_ok=False)
    return path


def _json(path, value, guard):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    guard(len(payload) + 8192)
    with path.open("xb") as stream:
        stream.write(payload)
    guard(0)
    return _ref(path)


def _npz(path, values, guard):
    import numpy as np
    guard(sum(value.nbytes for value in values.values()) + 1048576)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    guard(0)
    return _ref(path)


def _arrays(record, *, cap, observed=False):
    import numpy as np
    path = _checked(record, observed=observed)
    _need(type(cap) is int and cap > 0, "Explicit NPZ uncompressed cap required")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        _need(len({m.filename for m in members}) == len(members)
              and sum(m.file_size for m in members) <= cap,
              "Bounded NPZ without duplicate members required")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def freeze_registry(registry, output_file, *, guard, execution_check,
                    write_headroom_bytes):
    _ready(guard, execution_check, write_headroom_bytes)
    _need(registry.get("schema") == "ap_paper_sampled_registry_v1"
          and registry.get("labels_present") is False,
          "Observation-only frozen sampled registry required")
    return _json(_path(output_file, observed=True), registry, guard)


def prepare_features(recipe, registry_ref, inference_ref, output_directory, *,
                     guard, execution_check, write_headroom_bytes,
                     max_feature_bytes, max_metadata_bytes):
    _ready(guard, execution_check, write_headroom_bytes)
    registry = _read(registry_ref, observed=True)
    inference = _read(inference_ref, observed=True, cap=max_metadata_bytes)
    _need(registry.get("schema") == "ap_paper_sampled_registry_v1"
          and registry["observation_ref"] == inference_ref
          and inference["parent_id"] == registry["parent_id"]
          and inference["split"] == registry["split"]
          and inference["truth_or_oracle_read"] is False
          and registry.get("labels_present") is False,
          "Same frozen observation-only parent/registry required")
    n = registry["examples"]
    _need(type(n) is int and n > 0 and n == len(registry["records"])
          and type(max_feature_bytes) is int and n * 45 * 4 <= max_feature_bytes,
          "Complete native feature rows must fit their declared bound")
    import numpy as np
    import tskit
    bridge = importlib.import_module("arggnn.ancestralpaths_native_bridge")
    rows = _read(inference["observed_sample_manifest"], observed=True)
    panel = bridge.ObservedPanel(
        tuple(r["sample_id"] for r in rows[::2]),
        tuple(r["observed_node_id"] for r in rows),
        {r["observed_node_id"]: r["observed_cohort"] for r in rows})
    tree = tskit.load(_checked(inference["inferred_tree"], observed=True))
    queries = [r["observed_node_id"] for r in rows if r["role"] == "query"]
    x = np.empty((n, 5, 9), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    for index, feature in stages.iter_native_sampled_features(
            recipe, tree, panel, queries, registry["records"],
            guard=guard, execution_check=execution_check,
            write_headroom_bytes=write_headroom_bytes):
        _need(type(index) is int and 0 <= index < n and not seen[index]
              and feature.shape == (5, 9) and feature.dtype == np.float32
              and np.isfinite(feature).all(), "Complete unique original AP feature rows required")
        x[index], seen[index] = feature, True
    _need(bool(seen.all()), "No missing sampled native feature rows")
    output = _new_output(output_directory)
    array_ref = _npz(output / "features.npz",
                     dict(features=x, example_indices=np.arange(n, dtype=np.int64)), guard)
    product = dict(schema="ap_paper_native_features_v1",
        parent_id=registry["parent_id"], split=registry["split"], rows=n,
        registry_ref=registry_ref, inference_ref=inference_ref, arrays_ref=array_ref,
        feature_shape=[5, 9], original_native_feature_arithmetic=True,
        source_repeated_add_float64_final_input_float32=True,
        features_contain_no_query_truth=True, all_sampled_rows_retained=True)
    return _json(output / "features.json", product, guard)


def prepare_train_labels(registry_ref, features_ref, generation_ref,
                         target_schema_ref, truth_ref, output_directory, *,
                         guard, execution_check, write_headroom_bytes,
                         max_truth_uncompressed_bytes):
    _ready(guard, execution_check, write_headroom_bytes)
    registry = _read(registry_ref, observed=True)
    features = _read(features_ref, observed=True)
    _need(registry.get("split") == features.get("split") == "train"
          and registry["parent_id"] == features["parent_id"]
          and features["registry_ref"] == registry_ref
          and features["features_contain_no_query_truth"] is True,
          "Only frozen registered TRAIN supervision may resolve private targets")
    # This is the first private-target metadata access.
    generated = _read(generation_ref)
    schema = _read(target_schema_ref)
    _need(generated["parent_id"] == registry["parent_id"]
          and generated["split"] == "train" and schema["parent_id"] == registry["parent_id"]
          and schema["split"] == "train"
          and generated["files"]["truth/full_truth_intervals.npz"]["sha256"] == truth_ref["sha256"]
          and generated["files"]["truth/target_schema.json"]["sha256"] == target_schema_ref["sha256"],
          "Same-parent producer-authenticated TRAIN truth required")
    base = _checked(generation_ref).parent
    _need(_path(truth_ref["path"]) == base / "truth/full_truth_intervals.npz"
          and _path(target_schema_ref["path"]) == base / "truth/target_schema.json",
          "Exact same-parent private target namespace required")
    truth = _arrays(truth_ref, cap=max_truth_uncompressed_bytes)
    labels = sampling.native_truth_labels(registry, schema, truth)
    output = _new_output(output_directory)
    array_ref = _npz(output / "native_targets.npz", dict(native_labels_0based=labels), guard)
    return _json(output / "targets.json", dict(
        schema="ap_paper_TRAIN_native_targets_v1", parent_id=registry["parent_id"],
        split="train", registry_ref=registry_ref, features_ref=features_ref,
        generation_ref=generation_ref, target_schema_ref=target_schema_ref,
        truth_ref=truth_ref, arrays_ref=array_ref, rows=len(labels),
        no_label_dependent_selection=True, native_path_names=list(sampling.NATIVE_PATHS)), guard)


def load_tensorflow_cpu(prefix, *, threads):
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
          and Path(sys.prefix).resolve() == _path(prefix)
          and type(threads) is int and threads > 0,
          "Explicit isolated native TensorFlow CPU runtime and thread count required")
    _need(importlib.metadata.version("tensorflow-cpu") == "2.15.1"
          and importlib.metadata.version("keras") == "2.15.0",
          "Qualified TF2.15.1/Keras2.15 runtime required")
    tf = importlib.import_module("tensorflow")
    _need(not tf.config.list_physical_devices("GPU"), "Native AP CPU runtime must expose no GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    return tf


def build_author_dense(tf):
    """Execute only the original final model and compile statements."""
    source = _checked(AUTHOR_TRAIN).read_text()
    tree = ast.parse(source)
    models = [node for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "model" for t in node.targets)]
    compiles = [node for node in tree.body if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "model" and node.value.func.attr == "compile"]
    _need(len(models) == 1 and len(compiles) == 2,
          "Exactly one original Dense model and two original compile statements required")
    namespace = {"tf": tf}
    exec(compile(ast.Module(body=[models[0], compiles[-1]], type_ignores=[]),
                 str(_checked(AUTHOR_TRAIN)), "exec"), namespace)
    model = namespace["model"]
    _need(model.count_params() == 26630 and model.output_shape == (None, 6),
          "Original native Dense6 shape and parameter count required")
    return model


def fit_author_arrays(tf, features, labels, *, seed, guard):
    """One fresh source Dense fit, exactly five native epochs."""
    import numpy as np
    _need(features.shape == (50000, 5, 9) and features.dtype == np.float32
          and labels.shape == (50000,) and labels.dtype.kind in "iu"
          and np.isfinite(features).all() and ((labels >= 0) & (labels < 6)).all()
          and type(seed) is int and 0 <= seed < 2**32 and callable(guard),
          "Complete finite paper-sized native TRAIN arrays and fresh seed required")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_author_dense(tf)
    _need(int(model.optimizer.iterations.numpy()) == 0, "Fresh native Adam state required")
    guard(0)
    history = model.fit(features, tf.keras.utils.to_categorical(labels, num_classes=6),
                        epochs=5, batch_size=32, shuffle=True, verbose=2)
    expected = 5 * math.ceil(50000 / 32)
    _need(len(history.history["loss"]) == 5
          and all(np.isfinite(values).all() for values in history.history.values())
          and int(model.optimizer.iterations.numpy()) == expected,
          "Five finite epochs and all 7815 genuine Adam updates required")
    guard(0)
    return model, history, expected


def fit_native_dense(plan_ref, training_products, output_directory, *, tensorflow,
                     seed, guard, execution_check, write_headroom_bytes):
    _ready(guard, execution_check, write_headroom_bytes)
    plan = _read(plan_ref, observed=True)
    expected_plan = sampling.paper_sampling_plan(
        plan["train_parent_ids"], plan["test_parent_ids"], seed=plan["seed"])
    _need(plan == expected_plan and set(training_products) == set(plan["train_parent_ids"]),
          "Exactly all three frozen paper-sized TRAIN products required")
    import numpy as np
    xs, ys, total_cohorts = [], [], {c: 0 for c in sampling.COHORTS}
    for parent in plan["train_parent_ids"]:
        refs = training_products[parent]
        feature = _read(refs["features_ref"], observed=True)
        target = _read(refs["targets_ref"])
        registry = _read(feature["registry_ref"], observed=True)
        _need(feature["schema"] == "ap_paper_native_features_v1"
              and target["schema"] == "ap_paper_TRAIN_native_targets_v1"
              and feature["parent_id"] == target["parent_id"] == registry["parent_id"] == parent
              and feature["split"] == target["split"] == registry["split"] == "train"
              and target["features_ref"] == refs["features_ref"]
              and target["registry_ref"] == feature["registry_ref"]
              and target["no_label_dependent_selection"] is True
              and registry["cohort_quotas"] == plan["parents"][parent]["cohort_quotas"],
              "Same-parent native TRAIN feature/target/registry products required")
        arrays = _arrays(feature["arrays_ref"], cap=32 * 2**20, observed=True)
        labels = _arrays(target["arrays_ref"], cap=2 * 2**20)["native_labels_0based"]
        x = arrays["features"]
        n = registry["examples"]
        _need(x.shape == (n, 5, 9) and x.dtype == np.float32
              and labels.shape == (n,) and feature["rows"] == target["rows"] == n
              and np.array_equal(arrays["example_indices"], np.arange(n)),
              "All exact canonical native rows required")
        for cohort, count in registry["cohort_quotas"].items():
            total_cohorts[cohort] += count
        xs.append(x)
        ys.append(labels)
    _need(set(total_cohorts.values()) == {10000},
          "Exactly 10000 TRAIN examples per admixed cohort required")
    model, history, updates = fit_author_arrays(
        tensorflow, np.concatenate(xs), np.concatenate(ys), seed=seed, guard=guard)
    output = _new_output(output_directory)
    checkpoint = output / "native_dense_epoch5.keras"
    guard(model.count_params() * 32 + 2 * 2**20)
    model.save(checkpoint)
    _need(checkpoint.is_file(), "Genuine saved native final checkpoint required")
    return _json(output / "fit.json", dict(
        schema="ap_paper_native_dense_fit_v1", plan_ref=plan_ref,
        training_products=training_products, training_parent_ids=plan["train_parent_ids"],
        training_rows=50000, rows_per_cohort=total_cohorts,
        checkpoint_ref=_ref(checkpoint), author_training_ref=AUTHOR_TRAIN,
        native_path_names=list(sampling.NATIVE_PATHS), parameters=26630,
        epochs=5, batch_size=32, shuffle=True, optimizer_updates=updates, seed=seed,
        checkpoint_selection="final_epoch5", validation_selection=False,
        history=history.history, published_CNN_reproduction=False,
        original_released_Dense_model_statements_executed=True,
        paper_counts_with_disclosed_observation_only_sampling=True), guard)


def predict_native_dense(fit_ref, features_ref, output_directory, *, tensorflow,
                         guard, execution_check, write_headroom_bytes):
    _ready(guard, execution_check, write_headroom_bytes)
    fit = _read(fit_ref)
    feature = _read(features_ref, observed=True)
    registry = _read(feature["registry_ref"], observed=True)
    plan = _read(fit["plan_ref"], observed=True)
    _need(fit["schema"] == "ap_paper_native_dense_fit_v1"
          and fit["epochs"] == 5 and fit["optimizer_updates"] == 7815
          and fit["checkpoint_selection"] == "final_epoch5"
          and feature["parent_id"] == registry["parent_id"] in plan["test_parent_ids"]
          and feature["split"] == registry["split"] == "test"
          and registry["cohort_quotas"] == plan["parents"][registry["parent_id"]]["cohort_quotas"]
          and registry["examples"] == 10000,
          "Frozen independent native AP test support and final TRAIN checkpoint required")
    import numpy as np
    bridge = importlib.import_module("arggnn.ancestralpaths_native_bridge")
    arrays = _arrays(feature["arrays_ref"], cap=32 * 2**20, observed=True)
    _need(np.array_equal(arrays["example_indices"], np.arange(10000)),
          "All prespecified test example identities required")
    model = tensorflow.keras.models.load_model(_checked(fit["checkpoint_ref"]))
    probabilities = bridge.predict_native_probabilities(
        model, arrays["features"], batch_size=128).astype(np.float32)
    output = _new_output(output_directory)
    predictions = _npz(output / "predictions.npz",
        dict(example_indices=arrays["example_indices"], probabilities=probabilities), guard)
    return _json(output / "predictions.json", dict(
        schema="ap_paper_native_predictions_v1", parent_id=registry["parent_id"],
        split="test", registry_ref=feature["registry_ref"], features_ref=features_ref,
        fit_ref=fit_ref, predictions_ref=predictions, rows=10000,
        native_path_names=list(sampling.NATIVE_PATHS), query_truth_read=False,
        calibrated_probabilities_claimed=False, denominator_reduction=False), guard)


def _native_metrics(probabilities, labels):
    """Raw six-class metrics with truth rows and prediction columns."""
    import numpy as np
    p, y = np.asarray(probabilities), np.asarray(labels)
    _need(p.ndim == 2 and p.shape == (len(y), 6) and len(y) > 0
          and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
          and np.allclose(p.sum(1), 1., atol=1e-6, rtol=1e-5)
          and y.dtype.kind in "iu" and ((y >= 0) & (y < 6)).all(),
          "Complete raw native probabilities and actual labels required")
    calls = p.argmax(1)
    confusion = np.bincount(y * 6 + calls, minlength=36).reshape(6, 6)
    per_class = {}
    for i, name in enumerate(sampling.NATIVE_PATHS):
        tp, actual, predicted = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        per_class[name] = dict(
            precision=tp / predicted if predicted else 0.,
            recall=tp / actual if actual else 0.,
            F1=2 * tp / (actual + predicted) if actual + predicted else 0.,
            actual_count=actual, predicted_count=predicted)
    true_p = p[np.arange(len(y)), y]
    zeros = int((true_p == 0).sum())
    return dict(accuracy=float((calls == y).mean()),
        macro_F1=float(np.mean([v["F1"] for v in per_class.values()])),
        per_class=per_class, confusion=confusion.tolist(),
        confusion_convention="truth rows, prediction columns", records=len(y),
        raw_NLL=None if zeros else float(-np.log(true_p).mean()),
        raw_NLL_is_infinite=bool(zeros), zero_probability_truth_count=zeros,
        Brier=float(((p - np.eye(6)[y]) ** 2).sum(1).mean()),
        probabilities_clipped_or_renormalized=False)


def score_native_comparison(plan_ref, predictions, test_targets, output_directory, *,
                            guard, execution_check, write_headroom_bytes,
                            max_truth_uncompressed_bytes):
    """All-five/all-method barrier, then external common native-six scoring."""
    _ready(guard, execution_check, write_headroom_bytes)
    plan = _read(plan_ref, observed=True)
    parents = plan["test_parent_ids"]
    methods = ("ancestralpaths", "treeloom")
    _need(len(parents) == 5 and set(predictions) == set(methods)
          and set(test_targets) == set(parents)
          and all(set(predictions[m]) == set(parents) for m in methods),
          "Every method and all five independent evaluation parents required")
    import numpy as np
    sealed, registries = {}, {}
    # Read and validate EVERY prediction before any private target metadata.
    for parent in parents:
        sealed[parent] = {}
        registry_ref = None
        for method in methods:
            product = _read(predictions[method][parent], observed=True)
            _need(product["parent_id"] == parent and product["split"] == "test"
                  and product["rows"] == 10000
                  and product["native_path_names"] == list(sampling.NATIVE_PATHS),
                  "Complete declared native test predictions required")
            current_ref = product["registry_ref"]
            _need(registry_ref is None or current_ref == registry_ref,
                  "Identical immutable evaluation support across methods required")
            registry_ref = current_ref
            arrays = _arrays(product["predictions_ref"], cap=8 * 2**20, observed=True)
            p = arrays["probabilities"]
            _need(np.array_equal(arrays["example_indices"], np.arange(10000))
                  and p.shape == (10000, 6) and np.isfinite(p).all()
                  and ((p >= 0) & (p <= 1)).all()
                  and np.allclose(p.sum(1), 1., atol=1e-6, rtol=1e-5),
                  "All canonical prediction identities and raw six probabilities required")
            sealed[parent][method] = p
        registry = _read(registry_ref, observed=True)
        _need(registry["parent_id"] == parent and registry["split"] == "test"
              and registry["examples"] == 10000
              and registry["cohort_quotas"] == plan["parents"][parent]["cohort_quotas"],
              "Frozen paper-sized support without denominator revision required")
        registries[parent] = registry
    rows = {m: {} for m in methods}
    for parent in parents:
        execution_check()
        guard(0)
        refs = test_targets[parent]
        generated = _read(refs["generation_ref"])
        schema = _read(refs["target_schema_ref"])
        _need(generated["parent_id"] == parent and generated["split"] == "test"
              and generated["files"]["truth/full_truth_intervals.npz"]["sha256"] == refs["truth_ref"]["sha256"]
              and generated["files"]["truth/target_schema.json"]["sha256"] == refs["target_schema_ref"]["sha256"],
              "Same-parent producer-authenticated external evaluation targets required")
        base = _checked(refs["generation_ref"]).parent
        _need(_path(refs["truth_ref"]["path"]) == base / "truth/full_truth_intervals.npz"
              and _path(refs["target_schema_ref"]["path"]) == base / "truth/target_schema.json",
              "Exact independent evaluation target namespace required")
        truth = _arrays(refs["truth_ref"], cap=max_truth_uncompressed_bytes)
        y = sampling.native_truth_labels(registries[parent], schema, truth)
        del truth
        canonical = sorted(registries[parent]["records"], key=lambda r: r["example_index"])
        cohort = np.asarray([r["cohort"] for r in canonical])
        for method in methods:
            p = sealed[parent][method]
            rows[method][parent] = dict(overall=_native_metrics(p, y), cohorts={
                c: _native_metrics(p[cohort == c], y[cohort == c]) for c in sampling.COHORTS})
    summary, differences = {}, {}
    for method in methods:
        summary[method] = {}
        for group in ("overall", *sampling.COHORTS):
            values = [rows[method][p]["overall"] if group == "overall"
                      else rows[method][p]["cohorts"][group] for p in parents]
            summary[method][group] = {}
            for metric in ("accuracy", "macro_F1", "Brier", "raw_NLL"):
                actual = [v[metric] for v in values]
                summary[method][group][metric] = dict(
                    mean=None if any(v is None for v in actual) else float(np.mean(actual)),
                    sample_SD=None if any(v is None for v in actual) else float(np.std(actual, ddof=1)),
                    independent_parents=5, unit="fraction" if metric in ("accuracy", "macro_F1") else metric)
    for group in ("overall", *sampling.COHORTS):
        delta = [((rows["treeloom"][p]["overall"] if group == "overall"
                   else rows["treeloom"][p]["cohorts"][group])["accuracy"] -
                  (rows["ancestralpaths"][p]["overall"] if group == "overall"
                   else rows["ancestralpaths"][p]["cohorts"][group])["accuracy"])
                 for p in parents]
        differences[group] = dict(per_parent=delta, mean=float(np.mean(delta)),
                                  sample_SD=float(np.std(delta, ddof=1)),
                                  direction="TreeLoom minus AncestralPaths", unit="fraction")
    output = _new_output(output_directory)
    return _json(output / "comparison.json", dict(
        schema="ap_paper_native6_paired_comparison_v1", plan_ref=plan_ref,
        predictions=predictions, test_targets=test_targets,
        all_method_all_parent_prediction_barrier_before_truth=True,
        records_per_parent=10000, rows=rows, summary=summary,
        paired_accuracy_differences=differences,
        uncertainty="equal-parent mean and sample SD, ddof=1",
        endpoint="native six census-350 paths at the same observed SNP coordinates",
        published_CNN_reproduction=False,
        comparison_scope="paper quantities and released Dense classifier, with disclosed common sampling"), guard)

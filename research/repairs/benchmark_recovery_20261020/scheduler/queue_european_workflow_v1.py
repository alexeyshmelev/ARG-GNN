"""Plan-only factories and single-spec European scientific workflow stages.

Only execute_stage performs scientific work, inside ROOT's bounded supervisor.
Factories import no scientific runtime. All callbacks are mandatory at execution.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from . import queue_european_models_v1 as models
from arggnn.benchmarking import european_native_relate as native
from . import queue_european_recipes_v1 as recipes
from arggnn.benchmarking import european_sampling as sampling

MODULE = "research.repairs.benchmark_recovery_20261020.scheduler.queue_european_workflow_v1"
STAGES = ("ap_generate", "ap_infer", "ap_registry", "ap_features",
          "ap_train_targets", "ap_fit", "ap_predict", "ap_score")


def make_call(stage, arguments, output_directory, *, python):
    """Return a concrete inert descriptor for workflow.materialize."""
    models._need(stage in STAGES and isinstance(arguments, dict),
                 "Known European stage and explicit arguments required")
    executable = models._path(python)
    models._need(executable.is_file() and executable.is_relative_to(models.ROOT / ".runtime"),
                 "Existing dedicated project interpreter required")
    output = models._path(output_directory)
    return dict(module=MODULE, function="execute_stage", python=str(executable),
        spec=dict(stage=stage, arguments=dict(arguments), output_directory=str(output),
                  runtime_prefix=str(executable.parent.parent)),
        source_refs=[models._ref(models.ROOT / name) for name in (
            "src/arggnn/benchmarking/european_workflow.py",
            "src/arggnn/benchmarking/european_sampling.py",
            "src/arggnn/benchmarking/european_native_models.py",
            "src/arggnn/benchmarking/european_native_relate.py",
            "src/arggnn/benchmarking/european_execution.py",
            "src/arggnn/benchmarking/european_recipes.py",
            "src/arggnn/ancestralpaths_native_bridge.py",
            "src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py",
            "scripts/infer_biological_paths_relate_v1.py",
            models.AUTHOR_TRAIN["path"])])


def native_step_bridge(bounded_runner, *, execution_check):
    """Adapt the real supervised runner to the unchanged native journal API."""
    models._need(callable(bounded_runner) and callable(execution_check),
                 "ROOT-owned bounded native runner and current execution check required")
    def step(argv, work, journal, label, config, guard):
        execution_check()
        guard(config["native_step_headroom_bytes"])
        log = work / f"{len(journal):02d}_{label}.log"
        record = dict(label=label, argv=argv, cwd=str(work),
                      log=str(log.relative_to(models.ROOT)), status="started")
        journal.append(record)
        started = time.monotonic()
        try:
            result = bounded_runner(
                argv, cwd=work,
                env=dict(CUDA_VISIBLE_DEVICES="", TMPDIR=str(work / "tmp"),
                         PYTHONDONTWRITEBYTECODE="1"),
                limits={"max_seconds": config["relate"]["timeout_seconds_per_step"]},
                log_path=log)
            models._need(result.get("actual_direct_wait") is True
                         and result.get("native_process_absent") is True,
                         "Native runner must supply genuine direct wait and absence")
            record.update(returncode=result["actual_direct_returncode"],
                          actual_wait=result,
                          status="complete" if result["actual_direct_returncode"] == 0 else "failed")
            if record["returncode"]:
                raise RuntimeError("Native AP command failed: " + label)
        except BaseException as error:
            record.update(status="exception", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            if log.is_file():
                record["log_sha256"] = models._sha(log)
            payload = json.dumps(dict(steps=journal,
                copyright_notice="Copyright © 2018, University of Oxford"),
                indent=2, allow_nan=False).encode()
            guard(len(payload) + 8192)
            (work / "commands.json").write_bytes(payload)
            guard(0)
    return step


def execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    """ROOT worker entrypoint. Returns an immutable artifact ref or small map."""
    models._need(spec["stage"] in STAGES, "Explicit European stage required")
    args = dict(spec["arguments"])
    headroom = args.pop("write_headroom_bytes")
    models._ready(resource_check, execution_check, headroom)
    common = dict(guard=resource_check, execution_check=execution_check,
                  write_headroom_bytes=headroom)
    output = spec["output_directory"]
    stage = spec["stage"]
    if stage == "ap_generate":
        recipe = recipes.load_recipe("ancestralpaths_european")
        parent = dict(args.pop("parent"))
        parent["seeds"] = {k: parent["seeds"][k] for k in ("ancestry", "mutations")}
        recipes.verify_sources(recipe)
        directory = models._new_output(output)
        result = registered_generator()(
            recipe, parent=parent, output=directory, **args, **common)
        directory = models._path(output)
        return dict(
            generation_ref=models._ref(directory / "generation.json"),
            observed_directory=str(directory / "observed"),
            observed_input_refs={
                name: dict(path=str((directory / "observed" / name).relative_to(models.ROOT)),
                           sha256=result["files"]["observed/" + name]["sha256"])
                for name in native.OBSERVED_FILES},
            target_schema_ref=dict(path=str((directory / "truth/target_schema.json").relative_to(models.ROOT)),
                sha256=result["files"]["truth/target_schema.json"]["sha256"]),
            truth_ref=dict(path=str((directory / "truth/full_truth_intervals.npz").relative_to(models.ROOT)),
                sha256=result["files"]["truth/full_truth_intervals.npz"]["sha256"]),
            parent_id=result["parent_id"], split=result["split"],
            sequence_length_bp=result["sequence_length_bp"])
    if stage == "ap_infer":
        recipe = recipes.load_recipe("ancestralpaths_european")
        config = models._read(args.pop("native_config_ref"))
        native.infer_parent(
            recipe, destination=output, native_config=config,
            native_step=native_step_bridge(bounded_runner, execution_check=execution_check),
            **args, **common)
        return models._ref(models._path(output) / "inference.json")
    if stage == "ap_registry":
        plan = models._read(args["plan_ref"], observed=True)
        inference = models._read(args["inference_ref"], observed=True,
                                 cap=args["max_metadata_bytes"])
        rows = models._read(inference["observed_sample_manifest"], observed=True)
        registry = sampling.build_sampled_registry(
            plan, inference["parent_id"], rows, inference["post_five_iterations"],
            observation_ref=args["inference_ref"])
        output = models._new_output(output)
        registry_ref = models.freeze_registry(
            registry, output / "registry.json", **common)
        arrays_ref = models._npz(output / "registry_arrays.npz",
                                sampling.registry_arrays(registry), resource_check)
        return dict(registry_ref=registry_ref, arrays_ref=arrays_ref,
                    parent_id=registry["parent_id"], examples=registry["examples"])
    if stage == "ap_features":
        return models.prepare_features(
            recipes.load_recipe("ancestralpaths_european"),
            output_directory=output, **args, **common)
    if stage == "ap_train_targets":
        return models.prepare_train_labels(output_directory=output, **args, **common)
    if stage == "ap_score":
        return models.score_native_comparison(output_directory=output, **args, **common)
    tf = models.load_tensorflow_cpu(spec["runtime_prefix"], threads=args.pop("threads"))
    if stage == "ap_fit":
        return models.fit_native_dense(
            output_directory=output, tensorflow=tf, **args, **common)
    if stage == "ap_predict":
        return models.predict_native_dense(
            output_directory=output, tensorflow=tf, **args, **common)
    raise ValueError("Unknown European scientific stage")


def build_tasks(*, prefix, output_root, parent_records, sampling_plan_ref,
                native_config_ref, python_data, python_tensorflow,
                stage_limits, stage_parameters, treeloom_parameters):
    """Build the full concrete AP DAG in memory, with caller-chosen finite caps.

    ARGMix's available C4 release is a separate condition and is not silently
    substituted into this native-six paper task. All AP stages below are real
    maintained callables, including source-native training and external scoring.
    """
    plan = models._read(sampling_plan_ref, observed=True)
    expected = sampling.paper_sampling_plan(
        plan["train_parent_ids"], plan["test_parent_ids"], seed=plan["seed"])
    models._need(plan == expected, "Exact frozen sampling plan required")
    by_parent = {p["parent_id"]: p for p in parent_records}
    models._need(len(by_parent) == len(parent_records) == 8
                 and set(by_parent) == set(plan["parents"])
                 and all(by_parent[p]["split"] == plan["parents"][p]["split"]
                         for p in by_parent),
                 "Exactly the independent three-TRAIN/five-test simulation parents required")
    streams = [p["seeds"][key] for p in parent_records for key in
               ("ancestry", "mutations", "relate_initial", "relate_size_base")]
    models._need(len(streams) == len(set(streams)),
                 "All fresh simulation, mutation and native seed bases must be distinct")
    root = models._path(output_root)
    models._need(root.is_relative_to(models.ROOT / "experiments/results"),
                 "Dedicated admitted results root required")
    headroom = stage_parameters["write_headroom_bytes"]
    metadata_cap = stage_parameters["max_metadata_bytes"]
    tasks, identifiers = [], {}

    def link(task, *path, read=False):
        return dict({"$task": task}, **({"read": True} if read else {}),
                    **({"path": list(path)} if path else {}))

    def add(stage, identity, args, dependencies, *, python=python_data,
            resource="cpu", factory_module=MODULE, factory_function="make_call",
            factory_extra=None):
        task_id = prefix + "__" + identity + "__" + stage
        limits = dict(stage_limits[stage])
        for key in ("max_seconds", "max_output_bytes", "max_rss_bytes"):
            models._need(type(limits.get(key)) is int and limits[key] > 0,
                         "Explicit finite limits for every AP stage required")
        if resource == "gpu0":
            models._need(all(type(limits.get(k)) is int and limits[k] > 0
                             for k in ("minimum_free_gpu_bytes", "max_cuda_reserved_bytes")),
                         "Explicit whole-parent GPU caps required")
        arguments = (dict(stage=stage, arguments=dict(args, write_headroom_bytes=headroom),
                          python=python)
                     if factory_module == MODULE else dict(args))
        if factory_extra:
            arguments.update(factory_extra)
        tasks.append(dict(id=task_id, resource=resource, dependencies=list(dependencies),
            output_directory=str(root / identity / stage),
            factory_module=factory_module, factory_function=factory_function,
            factory_arguments=arguments, **limits))
        return task_id

    train_features = []
    for parent in plan["train_parent_ids"] + plan["test_parent_ids"]:
        generation = add("ap_generate", parent, dict(
            parent=by_parent[parent], require_paper_match=True), [])
        inference = add("ap_infer", parent, dict(
            parent=by_parent[parent], native_config_ref=native_config_ref,
            observed_directory=link(generation, "observed_directory"),
            expected_input_refs=link(generation, "observed_input_refs"),
            max_metadata_bytes=metadata_cap), [generation])
        registry = add("ap_registry", parent, dict(
            plan_ref=sampling_plan_ref, inference_ref=link(inference),
            max_metadata_bytes=metadata_cap), [inference])
        features = add("ap_features", parent, dict(
            registry_ref=link(registry, "registry_ref"),
            inference_ref=link(inference), max_metadata_bytes=metadata_cap,
            max_feature_bytes=stage_parameters["max_feature_bytes"]), [registry])
        identifiers[parent] = dict(generation=generation, inference=inference,
                                   registry=registry, features=features)
        if by_parent[parent]["split"] == "train":
            train_features.append(features)
    train_products, tree_inputs, heldout_inputs, test_targets = {}, {}, {}, {}
    for parent in plan["train_parent_ids"] + plan["test_parent_ids"]:
        ids = identifiers[parent]
        common = dict(
            inference_ref=link(ids["inference"]),
            observed_vcf_ref=link(ids["generation"], "observed_input_refs", "genotypes.vcf"),
            sampled_registry_ref=link(ids["registry"], "registry_ref"))
        if by_parent[parent]["split"] == "train":
            targets = add("ap_train_targets", parent, dict(
                registry_ref=link(ids["registry"], "registry_ref"),
                features_ref=link(ids["features"]),
                generation_ref=link(ids["generation"], "generation_ref"),
                target_schema_ref=link(ids["generation"], "target_schema_ref"),
                truth_ref=link(ids["generation"], "truth_ref"),
                max_truth_uncompressed_bytes=stage_parameters["max_truth_uncompressed_bytes"]),
                train_features)
            ids["targets"] = targets
            train_products[parent] = dict(features_ref=link(ids["features"]), targets_ref=link(targets))
            tree_inputs[parent] = dict(common, training_targets_ref=link(targets))
        else:
            heldout_inputs[parent] = common
            test_targets[parent] = {
                name: link(ids["generation"], name)
                for name in ("generation_ref", "target_schema_ref", "truth_ref")}
    native_fit = add("ap_fit", "all_train", dict(
        plan_ref=sampling_plan_ref, training_products=train_products,
        seed=stage_parameters["native_training_seed"],
        threads=stage_parameters["native_tensorflow_threads"]),
        [identifiers[p]["targets"] for p in plan["train_parent_ids"]], python=python_tensorflow)
    native_predictions = {}
    for parent in plan["test_parent_ids"]:
        ids = identifiers[parent]
        native_predictions[parent] = add("ap_predict", parent, dict(
            fit_ref=link(native_fit), features_ref=link(ids["features"]),
            threads=stage_parameters["native_tensorflow_threads"]),
            [native_fit, ids["features"]], python=python_tensorflow)
    tree_fit = add("treeloom_fit", "all_train", dict(
        protocol_id="ancestralpaths_paper200mb_native6",
        sampling_plan_ref=sampling_plan_ref, train_inputs=tree_inputs,
        **treeloom_parameters["fit"]),
        [identifiers[p]["targets"] for p in plan["train_parent_ids"]],
        resource="gpu0", factory_module="arggnn.benchmarking.european_training",
        factory_function="fit_factory")
    tree_predictions = add("treeloom_predict", "all_test", dict(
        protocol_id="ancestralpaths_paper200mb_native6",
        sampling_plan_ref=sampling_plan_ref,
        fit_completion_ref=link(tree_fit, "completion_ref"),
        heldout_inputs=heldout_inputs, **treeloom_parameters["predict"]),
        [tree_fit] + [identifiers[p]["registry"] for p in plan["test_parent_ids"]],
        resource="gpu0", factory_module="arggnn.benchmarking.european_training",
        factory_function="prediction_factory")
    add("ap_score", "all_test", dict(
        plan_ref=sampling_plan_ref, test_targets=test_targets,
        predictions={
            "ancestralpaths": {p: link(native_predictions[p]) for p in plan["test_parent_ids"]},
            "treeloom": {p: link(tree_predictions, "predictions", p)
                         for p in plan["test_parent_ids"]}},
        max_truth_uncompressed_bytes=stage_parameters["max_truth_uncompressed_bytes"]),
        list(native_predictions.values()) + [tree_predictions])
    return tasks


def registered_generator():
    """Plain static generator with the accepted four bookkeeping split names."""
    return recipes.generate_ancestralpaths_parent

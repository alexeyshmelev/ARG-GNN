"""Concrete extension DAG for available-release ARGMix and fresh C4 TreeLoom.

The AP native-six benchmark remains separate. Both scorers wait every registered
prediction on every independent TEST parent, regardless of endpoint.
"""
from __future__ import annotations

import copy
from . import european_argmix_release as release
from . import european_native_models as m
from . import european_workflow as ap

MODULE = "arggnn.benchmarking.european_argmix_workflow"
STAGES = ("release_frame", "release_c4_registry", "release_c4_targets",
    "release_native_targets", "release_archive", "release_fit",
    "release_infer", "release_seal", "release_c4_score", "release_native6_score")


def make_call(stage, arguments, output_directory, *, python):
    m._need(stage in STAGES and isinstance(arguments,dict),
            "Explicit new available-release stage required")
    executable = m._path(python)
    m._need(executable.is_file() and executable.is_relative_to(m.ROOT/".runtime"),
            "Existing project-local dedicated runtime required")
    return dict(module=MODULE, function="execute_stage", python=str(executable),
        spec=dict(stage=stage, arguments=dict(arguments),
                  output_directory=str(m._path(output_directory))),
        source_refs=[m._ref(m.ROOT/path) for path in (
            "src/arggnn/benchmarking/european_argmix_workflow.py",
            "src/arggnn/benchmarking/european_argmix_release.py",
            "src/arggnn/benchmarking/european_argmix_shards.py",
            "src/arggnn/benchmarking/european_c4_training.py",
            "src/arggnn/benchmarking/european_native_models.py",
            "src/arggnn/benchmarking/european_native_relate.py",
            "src/arggnn/benchmarking/european_workflow.py",
            "src/arggnn/benchmarking/european_sampling.py",
            "src/arggnn/expanded_native_argmix_v1.py",
            "src/arggnn/official_argmix_biological.py",
            "src/arggnn/full_panel_native_baseline_contracts_v1.py")])


def execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    m._need(spec["stage"] in STAGES, "Known concrete C4 workflow stage required")
    arguments = dict(spec["arguments"])
    headroom = arguments.pop("write_headroom_bytes")
    common = dict(guard=resource_check,execution_check=execution_check,
                  write_headroom_bytes=headroom)
    stage, output = spec["stage"], spec["output_directory"]
    functions = dict(release_frame=release.prepare_frame,
        release_c4_registry=release.prepare_c4_registry,
        release_c4_targets=release.prepare_sampled_c4_targets,
        release_native_targets=release.prepare_native_targets,
        release_archive=release.build_archive,
        release_fit=release.fit_release, release_infer=release.infer_release,
        release_seal=release.seal_sampled_predictions, release_c4_score=release.score_c4)
    if stage in ("release_archive", "release_infer"):
        m._need(callable(bounded_runner),"Actual ROOT-owned bounded native runner required")
        arguments["bounded_runner"] = bounded_runner
    if stage == "release_native6_score":
        plan = m._read(arguments["plan_ref"],observed=True)
        release.verify_cross_endpoint_barrier(
            plan, arguments["predictions"], arguments.pop("c4_predictions"))
        return m.score_native_comparison(output_directory=output,**arguments,**common)
    return functions[stage](output_directory=output,**arguments,**common)


def build_tasks(*, ap_tasks, ap_prefix, prefix, output_root, config_ref,
                sampling_plan_ref, native_config_ref, python_data, python_native,
                stage_limits, stage_parameters, treeloom_parameters):
    """Return existing AP tasks plus the executable C4 extension, no launch.

    The returned AP score task is replaced by a new adapter which verifies the
    cross-endpoint barrier before delegating to the unchanged native-six scorer.
    Callers must not append the original AP task list again.
    """
    cfg = release.config(config_ref)
    scope = cfg["data_condition"]
    plan = m._read(sampling_plan_ref,observed=True)
    m._need(plan["train_parent_ids"] == scope["train_parent_ids"]
        and plan["test_parent_ids"] == scope["test_parent_ids"],
        "Same original independent three TRAIN/five TEST parent identities required")
    tasks = copy.deepcopy(ap_tasks)
    known = {t["id"]:t for t in tasks}
    parents = scope["train_parent_ids"]+scope["test_parent_ids"]
    val = scope["validation_parent"]
    m._need(val["split"] == "validation" and val["parent_id"] not in parents,
            "Fresh independent full-length validation parent required")
    existing_parents = [t["factory_arguments"]["arguments"]["parent"]
        for t in tasks if t["factory_module"] == ap.MODULE
        and t["factory_arguments"].get("stage") == "ap_generate"]
    m._need(len(existing_parents) == 8,
            "Complete concrete existing AP observation DAG required")
    streams = [p["seeds"][k] for p in existing_parents+[val]
        for k in ("ancestry","mutations","relate_initial","relate_size_base")]
    m._need(len(streams) == len(set(streams)),
            "Independent validation simulation/mutation/Relate seed streams required")
    root = m._path(output_root)
    m._need(root.is_relative_to(m.ROOT/"experiments/results"),
            "Dedicated current project result scope required")
    headroom = stage_parameters["write_headroom_bytes"]
    metadata = stage_parameters["max_metadata_bytes"]
    truth_cap = stage_parameters["max_truth_uncompressed_bytes"]

    def link(task,*path):
        return dict({"$task":task},**({"path":list(path)} if path else {}))
    def original(parent,stage):
        key = ap_prefix+"__"+parent+"__"+stage
        m._need(key in known,"Actual AP task missing: "+key)
        return key
    def add(stage,identity,args,deps,*,resource="cpu",python=python_data,
            factory_module=MODULE,factory_function="make_call",native_stage=None):
        identifier = prefix+"__"+identity+"__"+stage
        m._need(identifier not in known,"Duplicate new C4 task identity forbidden")
        limits = dict(stage_limits[stage])
        m._need(all(type(limits.get(k)) is int and limits[k]>0 for k in
            ("max_seconds","max_output_bytes","max_rss_bytes")),
            "Explicit finite whole-stage bounds required")
        if resource == "gpu0":
            m._need(all(type(limits.get(k)) is int and limits[k]>0 for k in
                ("minimum_free_gpu_bytes","max_cuda_reserved_bytes")),
                "Explicit physical GPU0 bounds required")
        arguments = (dict(stage=native_stage or stage,
            arguments=dict(args,write_headroom_bytes=headroom),python=python)
            if factory_function == "make_call" else dict(args))
        task = dict(id=identifier,resource=resource,dependencies=list(dict.fromkeys(deps)),
            output_directory=str(root/identity/stage),factory_module=factory_module,
            factory_function=factory_function,factory_arguments=arguments,**limits)
        tasks.append(task)
        known[identifier] = task
        return identifier

    vparent = val["parent_id"]
    vgen = add("validation_generate",vparent,dict(parent=val,require_paper_match=True),[],
        factory_module=ap.MODULE,native_stage="ap_generate")
    vinfer = add("validation_infer",vparent,dict(parent=val,native_config_ref=native_config_ref,
        observed_directory=link(vgen,"observed_directory"),
        expected_input_refs=link(vgen,"observed_input_refs"),
        max_metadata_bytes=metadata),[vgen],
        factory_module=ap.MODULE,native_stage="ap_infer")
    frames, registries = {},{}
    for parent in parents+[vparent]:
        inference = vinfer if parent == vparent else original(parent,"ap_infer")
        frames[parent] = add("release_frame",parent,dict(
            inference_ref=link(inference),max_metadata_bytes=metadata),[inference])
        if parent != vparent:
            registries[parent] = add("release_c4_registry",parent,dict(
                plan_ref=sampling_plan_ref,inference_ref=link(inference),
                max_metadata_bytes=metadata),[inference])
    native_targets, native_archives, c4_targets = {},{},{}
    train_observed = [frames[p] for p in scope["train_parent_ids"]]
    train_registries = [registries[p] for p in scope["train_parent_ids"]]
    for parent in scope["train_parent_ids"]+[vparent]:
        generation = vgen if parent == vparent else original(parent,"ap_generate")
        binding = {k:link(generation,k) for k in
                   ("generation_ref","target_schema_ref","truth_ref")}
        native_targets[parent] = add("release_native_targets",parent,dict(
            frame_ref=link(frames[parent]),**binding,max_metadata_bytes=metadata,
            max_truth_uncompressed_bytes=truth_cap,
            max_snp_truth_bytes=stage_parameters["max_snp_truth_bytes"]),
            [frames[parent]]+train_observed)
        native_archives[parent] = add("release_archive",parent,dict(
            config_ref=config_ref,frame_ref=link(frames[parent]),
            targets_ref=link(native_targets[parent]),
            build_ref=stage_parameters["native_build_ref"],
            cpu_ref=stage_parameters["native_cpu_ref"],
            command_limits=stage_parameters["native_archive_command_limits"],
            max_archive_bytes=stage_parameters["max_native_archive_bytes"],
            max_metadata_bytes=metadata),
            [native_targets[parent]],python=python_native)
        if parent != vparent:
            c4_targets[parent] = add("release_c4_targets",parent,dict(
                registry_ref=link(registries[parent],"registry_ref"),**binding,
                max_truth_uncompressed_bytes=truth_cap),train_registries)
    native_fit = add("release_fit","all_train",dict(config_ref=config_ref,
        archive_products={p:link(t) for p,t in native_archives.items()},
        build_ref=stage_parameters["native_build_ref"],
        cpu_ref=stage_parameters["native_cpu_ref"],
        max_metadata_bytes=metadata),list(native_archives.values()),
        resource="gpu0",python=python_native)
    native_infer, native_predictions = {},{}
    for parent in scope["test_parent_ids"]:
        native_infer[parent] = add("release_infer",parent,dict(
            config_ref=config_ref,fit_ref=link(native_fit),frame_ref=link(frames[parent]),
            build_ref=stage_parameters["native_build_ref"],
            cpu_ref=stage_parameters["native_cpu_ref"],max_metadata_bytes=metadata),
            [native_fit,frames[parent]],resource="gpu0",python=python_native)
        native_predictions[parent] = add("release_seal",parent,dict(
            interval_predictions_ref=link(native_infer[parent]),
            registry_ref=link(registries[parent],"registry_ref"),
            max_h5_chunk_rows=stage_parameters["max_h5_chunk_rows"]),
            [native_infer[parent],registries[parent]],python=python_native)
    training_inputs, heldout_inputs = {},{}
    for parent in parents:
        item = dict(inference_ref=link(original(parent,"ap_infer")),
            observed_vcf_ref=link(original(parent,"ap_generate"),
                                  "observed_input_refs","genotypes.vcf"),
            sampled_registry_ref=link(registries[parent],"registry_ref"))
        if parent in scope["train_parent_ids"]:
            training_inputs[parent] = dict(item,training_targets_ref=link(c4_targets[parent]))
        else:
            heldout_inputs[parent] = item
    tree_fit = add("c4_treeloom_fit","all_train",dict(
        protocol_id="ancestralpaths_paper200mb_c4",sampling_plan_ref=sampling_plan_ref,
        train_inputs=training_inputs,**treeloom_parameters["fit"]),
        list(c4_targets.values()),resource="gpu0",
        factory_module="arggnn.benchmarking.european_c4_training",
        factory_function="fit_factory")
    tree_predictions = add("c4_treeloom_predict","all_test",dict(
        protocol_id="ancestralpaths_paper200mb_c4",sampling_plan_ref=sampling_plan_ref,
        fit_completion_ref=link(tree_fit,"completion_ref"),heldout_inputs=heldout_inputs,
        **treeloom_parameters["predict"]),
        [tree_fit]+[registries[p] for p in scope["test_parent_ids"]],
        resource="gpu0",factory_module="arggnn.benchmarking.european_c4_training",
        factory_function="prediction_factory")
    score_id = original("all_test","ap_score")
    old_score = known[score_id]
    native6_predictions = old_score["factory_arguments"]["arguments"]["predictions"]
    test_targets = old_score["factory_arguments"]["arguments"]["test_targets"]
    c4_predictions = dict(
        argmix={p:link(t) for p,t in native_predictions.items()},
        treeloom={p:link(tree_predictions,"predictions",p) for p in scope["test_parent_ids"]})
    complete_barrier = old_score["dependencies"]+list(native_predictions.values())+[tree_predictions]
    add("release_c4_score","all_test",dict(
        plan_ref=sampling_plan_ref,predictions=c4_predictions,
        native6_predictions=native6_predictions,test_targets=test_targets,
        max_truth_uncompressed_bytes=truth_cap),complete_barrier)
    old_score["factory_module"] = MODULE
    old_score["factory_function"] = "make_call"
    old_score["factory_arguments"]["stage"] = "release_native6_score"
    old_score["factory_arguments"]["arguments"]["c4_predictions"] = c4_predictions
    old_score["dependencies"] = list(dict.fromkeys(complete_barrier))
    return tasks

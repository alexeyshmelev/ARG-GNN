"""Source-only H040 frozen paired evaluation with exact interrupted provenance.

Scientific evaluation is the unchanged original H040 implementation. Only
checkpoint provenance and root GPU4/5 execution capability validation differ.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json
from pathlib import Path
from . import H040_latent_clade_execution_v1 as base
from . import H040_GPU45_resume_v1 as resume_worker
from .H040_latent_clade_execution_v1 import (
    ROOT, ARMS, TRAIN, VAL, PASSES, MODEL, SELECTION, WARM_CHECKPOINT, MECHANISM_REFS,
    require, bound, metadata, PhaseReader, WitnessCache, RAMMatrix, load_banks,
    validate_protocol)
SOURCE = "src/arggnn/H040_GPU45_frozen_evaluation_v1.py"
CONTEXT = "research/reviews/H040_GPU45_frozen_evaluation_source_v1/context_factory_v1.py"
ORIGINAL_SHA = "28973fe0a9dae2c950c07dd463a23edcb6b45ab6659813f5ef31b9dc766bf84d"
RESUME_SHA = "3fb89e4efc7a7fc6e50d3da5e2e30517357a7ac78cbf1279f42b7138b1e4e39d"
RESUME_CONTEXT = "research/reviews/H040_GPU45_resume_source_v1/context_factory_v2.py"
RESUME_CONTEXT_SHA = "71ed68a36ae7dc5d8614f23a711e753ccf8a632f4fb1a3ad13c840fee502aee4"
LAUNCHER = "scripts/launch_reserved_gpu_authorized_GPU45_v1.py"
GPU_AUTH = "research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json"
RESUME_MANIFESTS = {
    "tensor": dict(path="research/reviews/H040_GPU45_resume_source_v1/resume_tensor_inactive.json",
                   sha256="9ebb757af8189d04b6ab7c08f61d9803c6d7965abeb5c0e73bbde627832574a6"),
    "event_mlp": dict(path="research/reviews/H040_GPU45_resume_source_v1/resume_event_mlp_inactive.json",
                      sha256="cf21f3fe3614db09a0ff71fd69f1ebfe21f5f63d61fc9ba0cccca27bf34c0c6a"),
}


def actual_ref(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())


def science():
    base.science()
    globals().update(np=base.np, torch=base.torch, h038=base.h038, fit=base.fit)


def current_launcher_config(cfg):
    value = dict(cfg)
    value["helper_refs"] = dict(cfg["helper_refs"], actual_full_storage_launcher=actual_ref(LAUNCHER))
    return value


def validate_evaluation_contract_v1(spec, cfg, reader):
    """Validate only metadata/capabilities before the root creates CUDA state."""
    require(spec.get("schema") == "H040_GPU45_frozen_evaluation_input_v1"
            and spec.get("phase") == "evaluate" and spec.get("arm") == "joint_pair"
            and spec.get("main_source_ref") == actual_ref(SOURCE)
            and spec.get("root_context_module_ref") == actual_ref(CONTEXT),
            "Exact paired evaluation successor capability required")
    closure = spec["source_sha256"]
    needed = {base.SOURCE:ORIGINAL_SHA, resume_worker.SOURCE:RESUME_SHA,
              RESUME_CONTEXT:RESUME_CONTEXT_SHA, SOURCE:actual_ref(SOURCE)["sha256"],
              CONTEXT:actual_ref(CONTEXT)["sha256"]}
    require(all(closure.get(path) == sha for path,sha in needed.items()),
            "Original, resume and new evaluation source closure required")
    require(spec.get("GPU45_authorization_ref") == actual_ref(GPU_AUTH),
            "Exact root GPU4/5 authorization required")
    auth = reader.decode_json(spec["GPU45_authorization_ref"])
    require(auth["physical_GPUs"] == [4,5] and auth["preserve_external_processes"] is True,
            "Only user-authorized GPUs4/5")
    require(set(spec["train_completion_refs"]) == set(ARMS)
            and set(spec["train_actual_wait_refs"]) == set(ARMS),
            "Both actual resumed TRAIN completion and direct launcher wait refs required")
    for arm in ARMS:
        manifest = reader.decode_json(RESUME_MANIFESTS[arm])
        old = reader.decode_json(manifest["original_input_ref"])
        old_cfg = reader.decode_json(manifest["original_config_ref"])
        require(cfg == current_launcher_config(old_cfg), "Evaluation config may change only actual launcher helper")
        require(all(closure.get(path) == sha for path,sha in old["source_sha256"].items()),
                "Every original science/source pin must remain bound")
        for key in ("registration","generation_config","observation_cache"):
            require(spec["refs"][key] == old["refs"][key], "Original observation/generation source required")
        require(spec["observation_entry_refs"] == old["observation_entry_refs"]
                and spec["observation_cache_source_refs"] == old["observation_cache_source_refs"],
                "All original observed entries/cache source refs required")
    return True


def expected_checkpoint_provenance_v1(train, manifest, train_input, at, arm):
    """A unique expected provenance tuple, never a relaxed accepted-membership set."""
    require(type(at) is int and at in PASSES and manifest["arm"] == arm
            and arm in ARMS, "Registered checkpoint cell required")
    if at <= manifest["resume_pass_index"]:
        return dict(checkpoint_ref=manifest["retained_checkpoint_groups"][str(at)][arm],
                    input_ref=manifest["original_input_ref"], config_ref=manifest["original_config_ref"])
    record = train["checkpoint_groups"][str(at)][arm]
    require(record["path"] == train_input["output_directory"] + "/" + arm + ".pass_%02d.pt"%at,
            "Resumed checkpoint must be its exact original per-pass worker output")
    return dict(checkpoint_ref=record, input_ref=train["input_ref"], config_ref=train["config_ref"])


def validate_training_lineages_v1(ctx, trains):
    reader, cfg = ctx["reader"], ctx["cfg"]
    require(set(trains) == set(ARMS) and set(ctx["train_actual_wait_refs"]) == set(ARMS),
            "Both exact actual TRAIN lineages required")
    lineages = {}
    for arm in ARMS:
        train = trains[arm]
        manifest_ref = RESUME_MANIFESTS[arm]
        manifest = reader.decode_json(manifest_ref)
        require(train.get("schema") == "H040_train_completion_v1"
                and train.get("status") == "completed_fixed_30_pass_single_arm_TRAIN"
                and train.get("arm") == arm and train.get("hypothesis_id") == "H040"
                and train.get("steps") == {arm:960}
                and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
                and train.get("resume_manifest_ref") == manifest_ref
                and train.get("resumed_from_pass") == manifest["resume_pass_index"]
                and train.get("resumed_from_checkpoint_ref") == manifest["resume_checkpoint_ref"]
                and train.get("original_input_ref") == manifest["original_input_ref"]
                and train.get("original_config_ref") == manifest["original_config_ref"]
                and train.get("resumed_optimizer_steps_performed") == manifest["remaining_updates"]
                and train.get("optimizer_state_restored") is True
                and train.get("optimizer_state_reset_on_resume") is False,
                "Exact complete saved-plus-resumed TRAIN lineage required")
        require(set(train["checkpoint_groups"]) == set(map(str,PASSES))
                and set(train["checkpoint_provenance"]) == set(map(str,PASSES))
                and len(train["history"]) == 30
                and [row["pass_index"] for row in train["history"]] == list(range(1,31))
                and all(row["steps"] == {arm:32*row["pass_index"]} and row["full_parent_updates"] == 32
                        for row in train["history"]), "All six checkpoints and 30 complete passes required")
        train_input = reader.decode_json(train["input_ref"])
        train_cfg = reader.decode_json(train["config_ref"])
        require(train_input["refs"]["config"] == train["config_ref"]
                and train_input["resume_manifest_ref"] == manifest_ref
                and train_cfg == cfg, "Actual resumed input/config must match evaluation science exactly")
        resume_worker.validate_original_contract(train_input, train_cfg, reader)
        wait = reader.decode_json(ctx["train_actual_wait_refs"][arm])
        require(wait.get("returncode") == 0 and wait.get("actual_direct_launcher_wait") is True
                and wait.get("completed_ref") == ctx["train_completion_refs"][arm]
                and ctx["train_completion_refs"][arm]["path"] == train_input["output_directory"]+"/completed.json",
                "Actual direct exit0 launcher wait for this exact resumed completion required")
        initial = reader.decode_json(manifest["original_initialization_ref"])
        require(train["initialization"] == initial["initialization"]
                and train["initial_state_sha256"] == initial["initial_state_sha256"][arm],
                "Original matched initialization must be retained")
        for at in PASSES:
            require(set(train["checkpoint_groups"][str(at)]) == {arm}
                    and set(train["checkpoint_provenance"][str(at)]) == {arm},
                    "Exact single-arm per-pass checkpoint mapping required")
            expected = expected_checkpoint_provenance_v1(train,manifest,train_input,at,arm)
            require(train["checkpoint_provenance"][str(at)][arm] == expected
                    and train["checkpoint_groups"][str(at)][arm] == expected["checkpoint_ref"],
                    "Checkpoint provenance must equal its unique original/resumed input/config tuple")
        lineages[arm] = dict(manifest=manifest,train_input=train_input)
    return lineages


def validate_checkpoint_provenance_v1(cp, train, lineage, at, arm):
    expected = expected_checkpoint_provenance_v1(train,lineage["manifest"],lineage["train_input"],at,arm)
    require(cp.get("input_ref") == expected["input_ref"] and cp.get("config_ref") == expected["config_ref"],
            "Checkpoint payload must match its exact per-pass input/config provenance")
    if at > lineage["manifest"]["resume_pass_index"]:
        require(cp.get("optimizer_state_restored_from_checkpoint") is True
                and cp.get("optimizer_state_reset") is False
                and cp.get("resume_manifest_ref") == train["resume_manifest_ref"]
                and cp.get("resumed_from_checkpoint_ref") == lineage["manifest"]["resume_checkpoint_ref"]
                and cp.get("resumed_from_pass") == lineage["manifest"]["resume_pass_index"],
                "Actual resumed checkpoint restore provenance required")
    else:
        require(cp.get("optimizer_state_reset") is True,
                "Original checkpoint must preserve initial optimizer reset provenance")
    return True


def run_evaluate(ctx,input_ref):
    from .anonymous_clade_gnn_v1 import AnonymousCladeTensorGNNV1
    from .mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    cfg,output,guard,device=ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader=PhaseReader(ctx["reader"],"evaluate",ctx["bindings"])
    require(set(ctx["train_completion_refs"])==set(ARMS), "Both actual single-arm completion refs")
    trains={a:reader.decode_json(ctx["train_completion_refs"][a]) for a in ARMS}
    for arm, train in trains.items():
        require(train.get("status")=="completed_fixed_30_pass_single_arm_TRAIN"
            and train.get("hypothesis_id")=="H040" and train.get("arm")==arm
            and train.get("steps")=={arm:960} and train.get("outer_truth_read") is False,
            "Actual complete30-pass/960-update single-arm TRAIN required")
    require(trains[ARMS[0]]["initial_state_sha256"]==trains[ARMS[1]]["initial_state_sha256"]
        and trains[ARMS[0]]["initialization"]==trains[ARMS[1]]["initialization"],
        "Both arms must share the original matched initializer")
    lineages=validate_training_lineages_v1(ctx,trains)
    states,checkpoints={},{}
    for at in PASSES:
        for arm in ARMS:
            guard()
            train=trains[arm]
            ref=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(ref)),map_location="cpu",weights_only=False)
            require(cp.get("schema")=="H040_frozen_checkpoint_v1" and cp.get("arm")==arm
                and cp.get("pass_index")==at and cp.get("additional_optimizer_steps")==32*at
                and cp.get("parameter_count")==30788 and cp.get("model_kwargs")==MODEL
                and cp.get("teacher_used") is False and cp.get("mechanism_refs")==list(MECHANISM_REFS)
                and cp.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT
                and validate_checkpoint_provenance_v1(cp,train,lineages[arm],at,arm),
                "Exact actual frozen H040 checkpoint/provenance required")
            require(all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 for v in cp["model_state"].values()),
                "Raw FP32 learned states required")
            states[at,arm],checkpoints[at,arm]=cp["model_state"],ref
            del cp
    cache=WitnessCache(cfg,guard)
    banks,parents,keys,observation_seal=load_banks(ctx,reader,VAL,cache)
    ram=RAMMatrix(output,reader,parents,banks,cfg)
    reader.ram=ram
    residency=h038.FullParentResidencyV1(banks,{},device,cfg["GPU_residency_cap_bytes"])
    model=AnonymousCladeTensorGNNV1(**MODEL).to(device=device,dtype=torch.float32).eval().requires_grad_(False)
    for pid in VAL:
        guard()
        events_cpu=cache.get(pid,banks[pid],keys[pid])
        require(events_cpu.tensor_bytes()<=cfg["GPU_event_cap_bytes"],"Complete GPU event cap")
        bank,_=residency.get(pid)
        events=events_cpu.to(device)
        for at in PASSES:
            for arm in ARMS:
                guard()
                model.load_state_dict(states[at,arm],strict=True)
                with torch.no_grad(),torch.autocast(device_type="cuda",enabled=False):
                    probability=model.forward_train_holds(bank,boundary=events,interaction=arm).softmax(-1)
                ram.seal(at,arm,pid,probability.cpu().numpy(),checkpoints[at,arm])
                del probability
        del bank,events,events_cpu
    del model,residency,states
    ram.verify()
    reader.truth_gate=True
    helpers=fit.metric_helpers_v1()
    selected,history,full={}, {}, {}
    for arm in ARMS:
        validation=FullPanelPrimaryValidationV1(parents,ctx["bindings"],reader,helpers,cfg)
        rows=[]
        for at in PASSES:
            guard()
            seals=[(p,ram.records[at,arm,p]) for p in VAL]
            cells,means=validation.score(seals)
            row=dict(pass_index=at,cells=cells,equal_parent_metrics=means,
                checkpoint=checkpoints[at,arm],selection_rule=SELECTION)
            rows.append(row)
            output.save("%s.pass_%02d.primary_metrics.json"%(arm,at),row)
        best=min(rows,key=lambda row:h038.selection_key_v1(row["equal_parent_metrics"],row["pass_index"]))
        selected[arm],history[arm]=best,rows
        chosen=[(p,ram.records[best["pass_index"],arm,p]) for p in VAL]
        cells,means=validation.full_final(chosen)
        full[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        output.save(arm+".selected_full_diagnostics.json",full[arm])
    result=dict(schema="H040_frozen_VAL_completion_v1",status="completed_original_H038_selection",
        hypothesis_id="H040",input_ref=input_ref,train_completion_refs=ctx["train_completion_refs"],
        observations_seal=observation_seal,selection_rule=SELECTION,selected=selected,history=history,
        selected_full_diagnostics=full,optimization_performed=False,
        all96_RAM_seals_before_any_VAL_truth=True,RAM_native_probability_bytes=ram.bytes,
        dense_prediction_arrays_saved=False,outer_truth_read=False)
    output.save("completed.json",result)
    return result


def run_context(ctx, input_ref):
    require(ctx.get("phase") == "evaluate" and ctx.get("arm") == "joint_pair"
            and ctx.get("train_actual_wait_refs"), "Frozen paired evaluation context required")
    science()
    previous = base.run_evaluate
    base.run_evaluate = run_evaluate
    try:
        return base.run_context(ctx,input_ref)
    finally:
        base.run_evaluate = previous


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_frozen_evaluation_not_executed",source=SOURCE,
            checkpoint_passes=list(PASSES),arms=list(ARMS),VAL=list(VAL),
            native_RAM_prediction_count=96,physical_GPUs=[4,5],optimization_performed=False),indent=2))
        return 0
    require(args.input and args.expected_input_sha256,"Exact fresh root evaluation input required")
    record=dict(path=args.input,sha256=args.expected_input_sha256)
    spec=metadata(record)
    require(spec.get("schema")=="H040_GPU45_frozen_evaluation_input_v1"
            and spec.get("root_authorizer")=="root_agent" and spec.get("status")=="active"
            and spec.get("hypothesis_id")=="H040" and spec.get("phase")=="evaluate"
            and spec.get("arm")=="joint_pair","Active root frozen paired evaluation capability required")
    context_ref=spec["root_context_module_ref"]
    require(context_ref==actual_ref(CONTEXT),"Exact separate evaluation context required")
    bound(context_ref)
    module_spec=importlib.util.spec_from_file_location("H040_GPU45_evaluation_context",ROOT/CONTEXT)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    ctx=module.build_context_v1(record,spec)
    require(ctx["phase"]==spec["phase"] and ctx["arm"]==spec["arm"],"Root factory phase/arm differs")
    run_context(ctx,record)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

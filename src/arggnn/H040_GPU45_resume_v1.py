"""Source-only GPU4/5 continuation of frozen H040 checkpoints and AdamW state."""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json, random, time
from pathlib import Path
from . import H040_latent_clade_execution_v1 as base
from .H040_latent_clade_execution_v1 import (
    ROOT, ARMS, TRAIN, VAL, PASSES, MODEL, SELECTION, WARM_CHECKPOINT, MECHANISM_REFS,
    require, bound, metadata, PhaseReader, WitnessCache, load_banks, state_identity,
    train_one, validate_protocol)
SOURCE = "src/arggnn/H040_GPU45_resume_v1.py"
CONTEXT = "research/reviews/H040_GPU45_resume_source_v1/context_factory_v2.py"
ORIGINAL_SHA = "28973fe0a9dae2c950c07dd463a23edcb6b45ab6659813f5ef31b9dc766bf84d"
LAUNCHER = "scripts/launch_reserved_gpu_authorized_GPU45_v1.py"
GPU_AUTH = "research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json"


def actual_ref(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())


def science():
    base.science()
    globals().update(np=base.np, torch=base.torch, h038=base.h038, fit=base.fit)


def validate_original_contract(spec, cfg, reader):
    """Metadata-only contract; root context calls this before creating CUDA state."""
    require(spec.get("schema") == "H040_GPU45_resume_input_v1" and spec.get("phase") == "train"
            and spec.get("arm") in ARMS, "Exact root resume input required")
    require(spec.get("main_source_ref") == actual_ref(SOURCE)
            and spec.get("root_context_module_ref") == actual_ref(CONTEXT),
            "Exact new main/context source required")
    closure = spec["source_sha256"]
    require(closure.get(base.SOURCE) == ORIGINAL_SHA
            and closure.get(SOURCE) == actual_ref(SOURCE)["sha256"]
            and closure.get(CONTEXT) == actual_ref(CONTEXT)["sha256"],
            "Original plus new execution source closure required")
    manifest = reader.decode_json(spec["resume_manifest_ref"])
    arm = spec["arm"]
    expected_pass = 15 if arm == "tensor" else 5
    require(manifest["schema"] == "H040_GPU45_resume_manifest_v1"
            and manifest["arm"] == arm and manifest["resume_pass_index"] == expected_pass
            and manifest["resume_optimizer_steps"] == 32*expected_pass,
            "Exact saved checkpoint boundary required")
    old = reader.decode_json(manifest["original_input_ref"])
    old_cfg = reader.decode_json(manifest["original_config_ref"])
    expected_cfg = dict(old_cfg)
    expected_cfg["helper_refs"] = dict(old_cfg["helper_refs"],
        actual_full_storage_launcher=actual_ref(LAUNCHER))
    require(cfg == expected_cfg, "Only actual GPU45 launcher helper identity may differ from checkpoint config")
    require(old["arm"] == arm and old["phase"] == "train"
            and old["refs"]["config"] == manifest["original_config_ref"],
            "Original checkpoint input/config pair required")
    for key in ("registration", "generation_config", "observation_cache", "target_bindings"):
        require(spec["refs"][key] == old["refs"][key], "Original input data/target binding changed: " + key)
    require(spec["observation_entry_refs"] == old["observation_entry_refs"]
            and spec["observation_cache_source_refs"] == old["observation_cache_source_refs"],
            "All original observed data/source bindings required")
    require(all(closure.get(path) == sha for path, sha in old["source_sha256"].items()),
            "Every original execution source pin must be preserved")
    require(spec.get("GPU45_authorization_ref") == actual_ref(GPU_AUTH),
            "Exact new human GPU45 authority required")
    authority = reader.decode_json(spec["GPU45_authorization_ref"])
    require(authority["physical_GPUs"] == [4, 5] and authority["preserve_external_processes"] is True,
            "Only GPU4/5 authorized")
    return manifest


def validate_resume(ctx, input_ref):
    reader = ctx["reader"]
    manifest = reader.decode_json(ctx["resume_manifest_ref"])
    start, arm = manifest["resume_pass_index"], ctx["arm"]
    require(manifest["arm"] == arm and start == (15 if arm == "tensor" else 5)
            and manifest["resume_optimizer_steps"] == start*32,
            "Exact original saved resume boundary required")
    cp = torch.load(io.BytesIO(reader.read(manifest["resume_checkpoint_ref"])),
                    map_location="cpu", weights_only=False)
    require(cp.get("schema") == "H040_frozen_checkpoint_v1"
            and cp.get("arm") == arm and cp.get("seed") == 101
            and cp.get("pass_index") == start and cp.get("additional_optimizer_steps") == start*32
            and cp.get("model_kwargs") == MODEL and cp.get("parameter_count") == 30788
            and cp.get("teacher_used") is False and cp.get("mechanism_refs") == list(MECHANISM_REFS)
            and cp.get("warmstart_checkpoint_ref") == WARM_CHECKPOINT
            and cp.get("input_ref") == manifest["original_input_ref"]
            and cp.get("config_ref") == manifest["original_config_ref"]
            and cp.get("observation_seal") == manifest["original_observation_seal_ref"],
            "Exact original frozen checkpoint and provenance required")
    init = reader.decode_json(manifest["original_initialization_ref"])
    require(init["arm"] == arm and init["initial_state_sha256"][arm] == cp["initial_state_sha256"]
            and len(set(init["initial_state_sha256"].values())) == 1,
            "Original matched initialization identity required")
    expected = [at for at in PASSES if at <= start]
    require(set(manifest["retained_checkpoint_groups"]) == set(map(str, expected)),
            "Every earlier registered checkpoint group required")
    require(manifest["retained_checkpoint_groups"][str(start)][arm] == manifest["resume_checkpoint_ref"],
            "Resume boundary must be the retained original checkpoint")
    history = [reader.decode_json(r) for r in manifest["retained_TRAIN_history_refs"]]
    require(len(history) == start and [r["pass_index"] for r in history] == list(range(1, start+1))
            and all(r["steps"] == {arm:32*r["pass_index"]} and r["full_parent_updates"] == 32
                    for r in history), "Complete TRAIN history through saved checkpoint required")
    return manifest, cp, init, history


def restore_state_v1(model, optimizer, payload, expected_steps):
    """Restore learned state and AdamW moments/steps; never reset on migration."""
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    require(optimizer.state, "Populated saved AdamW state required")
    for parameter, state in optimizer.state.items():
        require(float(state["step"]) == expected_steps, "Saved AdamW per-parameter step differs")
        for key in ("exp_avg", "exp_avg_sq"):
            require(state[key].shape == parameter.shape, "Saved AdamW moment shape differs")
            state[key] = state[key].to(device=parameter.device, dtype=parameter.dtype)
    for group in optimizer.param_groups:
        require(group["lr"] == .001 and tuple(group["betas"]) == (.9, .999)
                and group["eps"] == 1e-8 and group["weight_decay"] == .0001
                and group["amsgrad"] is False and group["capturable"] is False
                and group["maximize"] is False and group["differentiable"] is False
                and group["foreach"] is None and group["fused"] is None,
                "Original AdamW hyperparameters/options required")


def run_train(ctx, input_ref):
    from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    cfg, output, guard, device = ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader = PhaseReader(ctx["reader"],"train",ctx["bindings"])
    cache = WitnessCache(cfg,guard)
    arm = ctx["arm"]
    require(arm in ARMS, "Exactly one registered arm per TRAIN invocation")
    active_arms = (arm,)
    resume, saved, old_init, history = validate_resume(ctx, input_ref)
    start_pass = resume["resume_pass_index"]
    checkpoints = {k:dict(v) for k,v in resume["retained_checkpoint_groups"].items()}
    steps = {arm:resume["resume_optimizer_steps"]}
    checkpoint_provenance = {k:{arm:dict(checkpoint_ref=v[arm],
        input_ref=resume["original_input_ref"], config_ref=resume["original_config_ref"])}
        for k,v in checkpoints.items()}
    history_records = list(resume["retained_TRAIN_history_refs"])
    banks,parents,keys,observation_seal = load_banks(ctx,reader,TRAIN,cache)
    require(set(cache.proofs)==set(TRAIN), "All32 observed recipes before any TRAIN truth")
    original_seal = reader.decode_json(resume["original_observation_seal_ref"])
    require(original_seal["recipes"] == cache.proofs
        and original_seal["entry_refs"] == {p:ctx["entry_refs"][p] for p in TRAIN}
        and original_seal["observation_cache_ref"] == ctx["observation_cache_ref"],
        "Rebuilt observed witness recipes must equal original pre-truth seal")
    reader.truth_gate = True
    targets = {}
    helpers = fit.metric_helpers_v1()
    target_bytes = 0
    for pid in TRAIN:
        guard()
        target = FullPanelExternalTargetV1(parents[pid],ctx["bindings"][pid],"TRAIN_loss",reader)
        mass = h038.native_hold_train_mass_v1(parents[pid],banks[pid],target,helpers)
        target_bytes += int(mass.nbytes)
        require(target_bytes <= cfg["CPU_target_cap_bytes"], "External TRAIN mass exceeds cap")
        targets[pid] = torch.from_numpy(mass)
        del target,mass
    from .anonymous_clade_gnn_v1 import AnonymousCladeTensorGNNV1
    guard()
    model = AnonymousCladeTensorGNNV1(**MODEL).to(device=device,dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
        betas=(.9,.999),eps=1e-8)
    restore_state_v1(model, optimizer, saved, steps[arm])
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all())
        for v in model.state_dict().values()), "Saved learned FP32 state must remain finite")
    family, optimizers = {arm:model}, {arm:optimizer}
    initialization = old_init["initialization"]
    initial_hashes = old_init["initial_state_sha256"]
    restored_state_sha256 = state_identity(model.state_dict())
    require(restored_state_sha256 == state_identity(saved["model_state"]),
        "Exact learned state restoration required")
    del saved
    residency = h038.FullParentResidencyV1(banks,targets,device,cfg["GPU_residency_cap_bytes"])
    output.save("initialization.json",dict(initialization=initialization,initial_state_sha256=initial_hashes,
        observation_seal=observation_seal,external_TRAIN_mass_bytes=target_bytes,
        witness_storage_bytes=0,checkpoint_passes=list(PASSES),VAL_influences_optimization=False,
        arm=arm,matched_initial_states_verified_before_selecting_one_arm=True,
        resume_manifest_ref=ctx["resume_manifest_ref"], resumed_from_pass=start_pass,
        restored_optimizer_steps=steps[arm], optimizer_state_reset_on_resume=False,
        restored_state_sha256=restored_state_sha256))
    try:
        for at in range(start_pass+1,31):
            order = list(TRAIN)
            random.Random(101+at).shuffle(order)
            sums = {arm:0.}
            started = time.monotonic()
            for number,pid in enumerate(order,1):
                guard()
                events_cpu = cache.get(pid,banks[pid],keys[pid])
                require(events_cpu.tensor_bytes()<=cfg["GPU_event_cap_bytes"], "One complete GPU event payload exceeds cap")
                bank,mass = residency.get(pid)
                events = events_cpu.to(device)
                for arm in active_arms:
                    row = train_one(family[arm],optimizers[arm],bank,mass,events,arm,guard)
                    steps[arm] += 1
                    sums[arm] += row["loss"]/32
                del events,bank,mass,events_cpu
                if number%8==0:
                    print(json.dumps(dict(stage="H040_TRAIN_progress",pass_index=at,
                        complete_parents=number,steps=steps.copy(),elapsed_seconds=time.monotonic()-started)),flush=True)
            row = dict(pass_index=at,steps=steps.copy(),mean_TRAIN_NLL=sums,
                full_parent_updates=32,seconds=time.monotonic()-started)
            history.append(row)
            history_records.append(output.save("TRAIN_pass_%02d.json"%at,row))
            if at in PASSES:
                groups = {}
                for arm in active_arms:
                    guard()
                    groups[arm] = output.save("%s.pass_%02d.pt"%(arm,at),dict(
                        schema="H040_frozen_checkpoint_v1",hypothesis_id="H040",phase="train",
                        model_state={k:v.detach().cpu() for k,v in family[arm].state_dict().items()},
                        optimizer_state=optimizers[arm].state_dict(),pass_index=at,
                        additional_optimizer_steps=steps[arm],arm=arm,seed=101,model_kwargs=MODEL,
                        parameter_count=30788,input_ref=input_ref,config_ref=ctx["config_ref"],
                        observation_seal=observation_seal,warmstart_checkpoint_ref=WARM_CHECKPOINT,
                        initial_state_sha256=initial_hashes[arm],teacher_used=False,
                        optimizer_state_reset=False,mechanism_refs=list(MECHANISM_REFS),
                        optimizer_state_restored_from_checkpoint=True,
                        resume_manifest_ref=ctx["resume_manifest_ref"],
                        resumed_from_checkpoint_ref=resume["resume_checkpoint_ref"],
                        resumed_from_pass=start_pass),"torch")
                checkpoints[str(at)] = groups
                checkpoint_provenance[str(at)] = {arm:dict(checkpoint_ref=groups[arm],
                    input_ref=input_ref,config_ref=ctx["config_ref"])}
                output.save("pass_%02d_checkpoints.json"%at,groups)
        require(steps=={arm:960} and len(checkpoints)==6 and len(history)==30,
            "Complete retained-plus-resumed30-pass schedule required")
        result = dict(schema="H040_train_completion_v1",status="completed_fixed_30_pass_single_arm_TRAIN",arm=arm,
            hypothesis_id="H040",input_ref=input_ref,config_ref=ctx["config_ref"],
            checkpoint_groups=checkpoints,steps=steps,history=history,initialization=initialization,
            initial_state_sha256=initial_hashes[arm],warmstart_checkpoint_ref=WARM_CHECKPOINT,
            observations_seal=observation_seal,VAL_truth_read=False,outer_truth_read=False,
            witness_arrays_written=False,witness_preparation_builds=cache.builds,
            witness_preparation_seconds=cache.build_seconds,
            peak_cached_witness_tensor_bytes=cache.peak_runtime_bytes,selection_deferred=True,
            checkpoint_provenance=checkpoint_provenance, history_record_refs=history_records,
            resume_manifest_ref=ctx["resume_manifest_ref"], resumed_from_pass=start_pass,
            resumed_from_checkpoint_ref=resume["resume_checkpoint_ref"],
            original_input_ref=resume["original_input_ref"],original_config_ref=resume["original_config_ref"],
            resumed_optimizer_steps_performed=960-resume["resume_optimizer_steps"],
            optimizer_state_reset_on_resume=False, optimizer_state_restored=True,
            preserved_uncheckpointed_history_refs=resume["preserved_uncheckpointed_history_refs"])
        output.save("completed.json",result)
        return result
    except Exception as error:
        output.save("failed.json",dict(status="failed_partial_preserved",error_type=type(error).__name__,
            error=str(error),logical_steps=steps,completed_checkpoint_groups=checkpoints,
            automatic_replay=False,VAL_truth_read=False,outer_truth_read=False))
        raise


def run_context(ctx, input_ref):
    require(ctx.get("phase") == "train" and ctx.get("resume_manifest_ref"),
            "Resume TRAIN capability required")
    science()
    previous = base.run_train
    base.run_train = run_train
    try:
        return base.run_context(ctx, input_ref)
    finally:
        base.run_train = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_resume_not_executed", source=SOURCE,
            physical_GPUs=[4, 5], total_passes=30, total_updates=960,
            resume_passes=dict(tensor=15, event_mlp=5), optimizer_state_restore_required=True), indent=2))
        return 0
    require(args.input and args.expected_input_sha256, "Pinned fresh root resume capability required")
    record = dict(path=args.input, sha256=args.expected_input_sha256)
    spec = metadata(record)
    require(spec.get("schema") == "H040_GPU45_resume_input_v1"
            and spec.get("root_authorizer") == "root_agent" and spec.get("status") == "active"
            and spec.get("hypothesis_id") == "H040" and spec.get("phase") == "train",
            "Active root H040 GPU45 resume capability required")
    context_ref = spec["root_context_module_ref"]
    require(context_ref == actual_ref(CONTEXT), "Exact separate resume root context required")
    bound(context_ref)
    module_spec = importlib.util.spec_from_file_location("H040_GPU45_root_context", ROOT/CONTEXT)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    ctx = module.build_context_v1(record, spec)
    require(ctx["phase"] == spec["phase"] and ctx["arm"] == spec["arm"], "Root phase/arm differs")
    run_context(ctx, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

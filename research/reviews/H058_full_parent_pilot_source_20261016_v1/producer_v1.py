"""Concrete H058 full TRAIN/VAL engine. Default inert; ROOT provides existing guards.

No scheduler, GPU lease, storage ledger, admission or process controller is added.
Scientific imports occur only after explicit authenticated ROOT context validation.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,os,random,statistics,time
from .protocol_v1 import *

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H058_full_parent_pilot_source_20261016_v1"
SOURCE=HERE+"/producer_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
CAP_NAMES=("CPU_bank_cap_bytes","CPU_target_cap_bytes","episode_payload_cap_bytes",
    "episode_cache_cap_bytes","GPU_episode_cap_bytes","GPU_residency_cap_bytes",
    "packed_layout_cap_bytes","packed_prepare_named_cap_bytes","RAM_prediction_cap_bytes",
    "checkpoint_file_cap_bytes","output_cap_bytes","max_cpu_rss_bytes","max_cuda_reserved_bytes",
    "binding_chunk_elements","packed_max_hold_nodes")


def actual_ref(path):
    p=ROOT/path
    require(not Path(path).is_absolute() and ".." not in Path(path).parts
        and p.resolve()==p and p.is_file() and not p.is_symlink(), "Canonical project source/metadata ref")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def science():
    global torch,np,h45,shared,model,observations,matrix,diagnostics
    import numpy as np
    import torch
    from arggnn import train_ordered_clade_episode_v1 as h45
    from arggnn import H040_latent_clade_execution_v1 as shared
    from . import model_v1 as model,observations_v1 as observations,matrix_v1 as matrix,diagnostics_v1 as diagnostics
    h45.science()


def runtime_flags_v1():
    if not torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return dict(backend="cpu",intraop_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),default_dtype=str(torch.get_default_dtype()),
            CUDA_initialized=False,torch_version=torch.__version__,
            numpy_version=np.__version__,PyG_version=__import__("torch_geometric").__version__)
    return dict(matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_TF32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def validate_runtime_flags_v1(ctx):
    if ctx["device"].type=="cpu":
        expected=dict(backend="cpu",intraop_threads=32,interop_threads=1,
            default_dtype="torch.float32",CUDA_initialized=False,
            torch_version="2.6.0+cu124",numpy_version="2.4.6",PyG_version="2.7.0")
        require(runtime_flags_v1()==expected, "Original CPU FP32 inference runtime/32 threads/interop1/noCUDA")
        proof=ctx["reader"].decode_json(ctx["cfg"]["runtime_equivalence_ref"])
        require(proof.get("root_authorizer")=="root_agent" and proof.get("flags")==expected
            and proof.get("original_H050_runtime_ref")==ctx["cfg"]["H050_CPU_runtime_ref"],
            "Actual ROOT same CPU inference runtime evidence")
        return
    expected=dict(matmul_precision="highest",cuda_matmul_TF32=False,cudnn_TF32=False,
        cudnn_benchmark=False,cudnn_deterministic=True,deterministic_algorithms=True,
        deterministic_warn_only=False,CUBLAS_WORKSPACE_CONFIG=":4096:8")
    require(runtime_flags_v1()==expected, "Exact actual H050 FP32/determinism settings; ROOT configures before CUDA")
    proof=ctx["reader"].decode_json(ctx["cfg"]["runtime_equivalence_ref"])
    require(proof.get("root_authorizer")=="root_agent"
        and proof.get("configured_before_CUDA") is True and proof.get("flags")==expected
        and proof.get("original_H050_runtime_ref")==ctx["cfg"]["H050_runtime_qualification_ref"],
        "Actual ROOT runtime identity and pre-CUDA configuration evidence")


def validate_context_v1(ctx):
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("hypothesis_id")=="H058" and ctx.get("phase") in
        ("train","evaluate","train_continuation","evaluate_continuation")
        and all(callable(ctx.get(k)) for k in ("guard","before_episode_build")),
        "Explicit actual ROOT context using the existing resource framework")
    ctx["guard"]()  # Existing authority/lease/storage boundary before scientific imports.
    cfg=ctx["cfg"];phase=ctx["phase"]
    require(cfg.get("schema")=="H058_full_parent_execution_config_v1"
        and cfg.get("execution_enabled") is True
        and cfg.get("scientific_protocol")==fixed_protocol_v1()
        and cfg.get("producer_ref")==actual_ref(SOURCE)
        and cfg.get("observation_cache_ref")==OBSERVATION
        and cfg.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT_REF
        and cfg.get("warmstart_completion_ref")==WARM_COMPLETION_REF
        and cfg.get("root_resource_caps_reviewed") is True
        and all(type(cfg.get(k)) is int and cfg[k]>0 for k in CAP_NAMES)
        and cfg["binding_chunk_elements"]<=1048576
        and cfg.get("checkpoint_backbone") is True
        and cfg.get("evaluation_backend") in ("cpu","cuda")
        and cfg.get("continuation_mode") in ("pending_identity_audit","reuse_original_CPU_selected_metrics",
            "replay_frozen_H050_checkpoints")
        and (cfg.get("continuation_mode")!="reuse_original_CPU_selected_metrics"
            or cfg["evaluation_backend"]=="cpu"), "Exact full-budget science and reviewed resource caps")
    require(ctx["reader"].decode_json(ctx["config_ref"])==cfg
        and ctx["reader"].root.resolve()==ROOT
        and set(ctx["entry_refs"])==set(TRAIN+VAL)
        and set(ctx["bindings"])==set(TRAIN if phase.startswith("train") else VAL),
        "Exact original corpus and phase-only target capability")
    require((phase.startswith("train") and ctx.get("physical_gpu")==0 and ctx["device"].type=="cuda")
        or (phase.startswith("evaluate") and ctx["device"].type==cfg["evaluation_backend"]
            and (ctx["device"].type=="cpu" or ctx.get("physical_gpu")==0)),
        "Only actual leased physical GPU0; explicit frozen evaluation backend")
    old=ctx["reader"].decode_json(H050_INPUT_REF)
    old_cfg=ctx["reader"].decode_json(old["refs"]["config"])
    require(cfg["metrics"]==old_cfg["metrics"]
        and cfg["H050_runtime_qualification_ref"]==H050_GPU_QUALIFICATION_REF
        and cfg["H050_CPU_runtime_ref"]==H050_CPU_RUNTIME_REF
        and cfg["H050_CPU_eval_completion_ref"]==H050_CPU_EVAL_REF
        and cfg["H050_train_completion_ref"]==H050_TRAIN_COMPLETION_REF,
        "Exact original scoring/support and historical control/runtime records")
    target_record=H050_TRAIN_BINDINGS_REF if phase.startswith("train") else H050_VAL_BINDINGS_REF
    original_targets=ctx["reader"].decode_json(target_record)["parents"]
    require(ctx["bindings"]==original_targets
        and ctx["entry_refs"]==old["observation_entry_refs"]
        and ctx["observation_source_refs"]==old["observation_cache_source_refs"],
        "Exact original phase truth bindings and all40 observed entry/source dictionaries")
    manifest=ctx["reader"].decode_json(cfg["source_manifest_ref"])
    require(cfg["source_manifest_ref"]==actual_ref(MANIFEST), "Exact source closure manifest")
    for record in manifest["refs"]:
        ctx["reader"].read(record)
    registration=ctx["reader"].decode_json(cfg["registration_snapshot_ref"])
    require(registration.get("id")=="H058"
        and registration.get("protocol_ref")==PROTOCOL_REF
        and registration.get("F1_binder_independent_review_ref")==F1_REVIEW_REF,
        "Actual registered H058 snapshot and passed F1 successor")
    if phase.startswith("train"):
        directive=ctx["reader"].decode_json(cfg["full_training_directive_ref"])
        require(directive.get("root_authorizer")=="root_agent"
            and directive.get("full30pass_training_authorized") is True
            and directive.get("scientific_protocol")==fixed_protocol_v1(),
            "Separate actual full-budget ROOT training authority")
    # Separate operational authority is necessary even after source review.
    completed=ctx["reader"].decode_json(cfg["one_hold_completion_ref"])
    wait=ctx["reader"].decode_json(cfg["one_hold_direct_wait_ref"])
    terminal=ctx["reader"].decode_json(cfg["one_hold_outer_terminal_ref"])
    receipt=ctx["reader"].decode_json(completed["qualification_ref"])
    require(completed.get("status")=="completed_H058_one_observed_hold_FP32_qualification"
        and receipt.get("status")=="PASS" and receipt.get("optimizer_updates")==0
        and receipt.get("schema")=="same_marker_one_observed_TRAIN000_FP32_qualification_v1"
        and receipt.get("parent_id")=="ap_expanded_train_000"
        and completed.get("actual_wait_ref")==cfg["one_hold_direct_wait_ref"]
        and wait.get("completed_ref")==completed["qualification_ref"]
        and wait.get("returncode")==0 and wait.get("actual_direct_worker_wait") is True
        and wait.get("all_owned_processes_and_groups_drained") is True
        and wait.get("operational_failure") is None
        and terminal.get("actual_exit_code")==0 and terminal.get("actual_terminal_consumed") is True
        and terminal.get("completed_ref")==cfg["one_hold_completion_ref"]
        and terminal.get("direct_wait_ref")==cfg["one_hold_direct_wait_ref"],
        "Genuine successful one-hold arithmetic qualification and consumed direct/outer waits")
    layout=ctx["reader"].decode_json(cfg["layout_resource_proof_ref"])
    require(layout.get("root_authorizer")=="root_agent"
        and layout.get("status")=="admitted_first_scheduled_full_parent_step"
        and layout.get("source_count_ref")==cfg["source_count_ref"]
        and layout.get("physical_gpu")==0 and layout.get("reduced_data") is False
        and layout.get("counted_in_960_updates") is True,
        "ROOT full-coverage source-bound admission; no disposable qualification update")
    if phase.endswith("_continuation"):
        directive=ctx["reader"].decode_json(cfg["separate_continuation_directive_ref"])
        require(directive.get("root_authorizer")=="root_agent"
            and directive.get("separately_authorized_full960_continuation") is True
            and directive.get("scientific_protocol")==fixed_protocol_v1()
            and directive.get("registration_amendment_ref") is not None,
            "No silent replay: separate full960 matched-control authority and registration amendment")
    return cfg


def state_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scheduled_step_v1(ctx,arm,value,optimizer,bank,mass,payload,steps,first):
    captures={};checks={};handles=[]
    before={k:p.detach().cpu().clone() for k,p in value.named_parameters()} if first else {}
    if first:
        def capture(_module,_inputs,out):
            captures["logits"]=out.detach().index_select(1,bank.query_column).cpu()
        handles.append(value.classifier.register_forward_hook(capture))
        for name,parameter in value.named_parameters():
            def observe(gradient,key=name):
                checks[key]=dict(dtype=str(gradient.dtype),finite=bool(torch.isfinite(gradient).all()))
                return gradient
            handles.append(parameter.register_hook(observe))
    def counted():
        steps[arm]+=1
    try:
        # Exact existing Torch/PyG full-parent objective/backward/clip/AdamW step.
        row=h45.train_one(value,optimizer,bank,mass,payload,ctx["guard"],
            qualify=first,after_optimizer_step=counted)
    finally:
        for handle in handles:handle.remove()
    if first:
        named=dict(value.named_parameters())
        require(set(checks)==set(named) and all(v["dtype"]=="torch.float32" and v["finite"] for v in checks.values()),
            "Every allocated parameter has a finite FP32 first scheduled gradient")
        require(len(optimizer.state)==len(named)
            and all(int(v["step"].item())==1 for v in optimizer.state.values()), "Fresh actual full-model AdamW step1")
        changes={k:not torch.equal(p.detach().cpu(),before[k]) for k,p in named.items()}
        require(any(changed for k,changed in changes.items() if not k.startswith("same_marker.")),
            "Actual original H045 body updated")
        row["first_scheduled_step"]=dict(gradient_checks=checks,updated_parameters=changes,
            optimizer_steps=steps[arm],no_extra_updates=True,retained_in_full960=True,
            parameter_count=PARAMETERS[arm],runtime_flags=runtime_flags_v1())
        require("logits" in captures, "Logits captured from the actual scheduled forward")
    return row,captures.get("logits")


def save_checkpoint_v1(ctx,input_ref,arm,at,models,optimizers,steps,hashes,seal):
    require(len(optimizers[arm].state)==len(list(models[arm].parameters()))
        and all(int(value["step"].item())==steps[arm] for value in optimizers[arm].state.values()),
        "Actual saved AdamW counters match all scheduled updates")
    checkpoint=dict(schema="H058_full_parent_checkpoint_v1",hypothesis_id="H058",
        phase="train",arm=arm,pass_index=at,additional_optimizer_steps=steps[arm],
        seed=101,parameter_count=PARAMETERS[arm],optimizer_state_reset=True,teacher_used=False,
        model_state={k:v.detach().cpu() for k,v in models[arm].state_dict().items()},
        optimizer_state=optimizers[arm].state_dict(),initial_state_sha256=hashes[arm],
        input_ref=input_ref,config_ref=ctx["config_ref"],observation_seal=seal,
        scientific_protocol=fixed_protocol_v1(),producer_ref=actual_ref(SOURCE),
        warmstart_checkpoint_ref=WARM_CHECKPOINT_REF,warmstart_completion_ref=WARM_COMPLETION_REF)
    buffer=io.BytesIO();torch.save(checkpoint,buffer)
    require(buffer.tell()<=ctx["cfg"]["checkpoint_file_cap_bytes"], "Bounded real checkpoint bytes")
    return ctx["output"].save("%s.pass_%02d.pt"%(arm,at),checkpoint,"torch")


def run_train_v1(ctx,input_ref):
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    arms=phase_arms(ctx["phase"]);reader=h45.PhaseReader(ctx["reader"],"train",ctx["bindings"])
    output=ctx["output"];steps={a:0 for a in arms};groups={};history=[];qualified={}
    try:
        observed=observations.load_observations_v1(ctx,reader,TRAIN)
        completion=reader.decode_json(WARM_COMPLETION_REF)
        warm=torch.load(io.BytesIO(reader.read(WARM_CHECKPOINT_REF)),map_location="cpu",weights_only=False)
        models,optimizers,initialization=model.build_family_v1(warm,completion,
            checkpoint_ref=WARM_CHECKPOINT_REF,completion_ref=WARM_COMPLETION_REF,
            device=ctx["device"],arms=arms,checkpoint_backbone=True)
        del warm
        hashes={a:state_hash(m.state_dict()) for a,m in models.items()}
        if arms==ARMS:require(len(set(hashes.values()))==1, "Identical aligned/null initial learned states")
        reader.truth_gate=True;targets={};total=0;helpers=h45.fit.metric_helpers_v1()
        for pid in TRAIN:
            ctx["guard"]()
            target=FullPanelExternalTargetV1(observed["parents"][pid],ctx["bindings"][pid],"TRAIN_loss",reader)
            mass=h45.h038.native_hold_train_mass_v1(observed["parents"][pid],observed["banks"][pid],target,helpers)
            total+=mass.nbytes
            require(total<=ctx["cfg"]["CPU_target_cap_bytes"], "Complete exact native-hold target cap")
            targets[pid]=torch.from_numpy(mass)
            del mass,target
        output.save("initialization.json",dict(hypothesis_id="H058",initialization=initialization,
            initial_state_sha256=hashes,observations_seal=observed["seal"],
            continuation_not_automatically_retrained=True,VAL_influences_optimization=False))
        for at in range(1,31):
            order=list(TRAIN);random.Random(101+at).shuffle(order)
            means={a:0. for a in arms};started=time.monotonic()
            for number,pid in enumerate(order,1):
                parent_start=time.monotonic()
                bank,episode,mass=observations.device_parent_v1(ctx,observed,pid,targets[pid])
                first=at==1 and number==1;reference=None
                for arm in arms:
                    require(all(p.grad is None for m in models.values() for p in m.parameters()), "No cross-arm live gradients")
                    payload,packing=observations.payload_v1(ctx,observed,pid,arm,bank,episode)
                    row,initial=scheduled_step_v1(ctx,arm,models[arm],optimizers[arm],bank,mass,payload,steps,first)
                    means[arm]+=row["loss"]/32.
                    if first:
                        if reference is None:reference=initial
                        require(initial.shape==reference.shape and torch.allclose(initial,reference,rtol=1e-5,atol=1e-6),
                            "Actual initial aligned/null scheduled predictors agree")
                        qualified[arm]=dict(row,parent_id=pid,packed_preparation=packing,
                            parent_seconds_to_receipt=time.monotonic()-parent_start,
                            initial_pair_max_abs_difference=float((initial-reference).abs().max()))
                        # Commit each successful real update's evidence immediately.
                        output.save("%s.first_scheduled_step.json"%arm,qualified[arm])
                    del payload,packing,initial
                del reference,bank,episode,mass
                if number%8==0:print(json.dumps(dict(stage="H058_TRAIN",pass_index=at,parents=number,steps=steps)),flush=True)
            require(steps=={a:32*at for a in arms}, "All32 complete parent updates per arm/pass")
            row=dict(pass_index=at,steps=steps.copy(),parent_order=order,mean_TRAIN_NLL=means,
                seconds=time.monotonic()-started,full_parent_updates_per_arm=32)
            history.append(row);output.save("TRAIN_pass_%02d.json"%at,row)
            if at in PASSES:
                groups[str(at)]={a:save_checkpoint_v1(ctx,input_ref,a,at,models,optimizers,steps,hashes,observed["seal"]) for a in arms}
        require(steps=={a:960 for a in arms} and sum(map(len,groups.values()))==6*len(arms), "Full30pass budget and all six checkpoints")
        result=dict(schema="H058_full_parent_train_completion_v1",hypothesis_id="H058",
            status="completed_fixed30pass_full_TRAIN",input_ref=input_ref,config_ref=ctx["config_ref"],
            arms=list(arms),steps=steps,history=history,checkpoint_groups=groups,
            first_scheduled_step=qualified,initial_state_sha256=hashes,observations_seal=observed["seal"],
            parameters_per_arm={a:PARAMETERS[a] for a in arms},scientific_protocol=fixed_protocol_v1(),
            warmstart_checkpoint_ref=WARM_CHECKPOINT_REF,warmstart_completion_ref=WARM_COMPLETION_REF,
            VAL_truth_read=False,outer_truth_read=False,disposable_updates=0)
        output.save("completed.json",result);return result
    except BaseException as error:
        output.save("failed.json",dict(hypothesis_id="H058",phase=ctx["phase"],status="failed_partial_preserved",
            logical_steps=steps,completed_checkpoint_groups=groups,error_type=type(error).__name__,
            error=str(error),automatic_replay=False,VAL_truth_read=False,outer_truth_read=False))
        raise


def load_H058_checkpoints_v1(ctx,reader):
    train=reader.decode_json(ctx["train_completion_ref"]);arms=phase_arms(ctx["phase"])
    require(train.get("schema")=="H058_full_parent_train_completion_v1"
        and train.get("status")=="completed_fixed30pass_full_TRAIN"
        and train.get("arms")==list(arms) and train.get("steps")=={a:960 for a in arms}
        and train.get("scientific_protocol")==fixed_protocol_v1()
        and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
        and len(train.get("history",[]))==30, "Actual complete full-budget TRAIN")
    for at,row in enumerate(train["history"],1):
        order=list(TRAIN);random.Random(101+at).shuffle(order)
        require(row["pass_index"]==at and row["steps"]=={a:32*at for a in arms}
            and row["parent_order"]==order, "Exact full actual update schedule")
    require(set(train["checkpoint_groups"])=={str(at) for at in PASSES}, "Exactly six scheduled groups")
    states,refs={},{}
    for at in PASSES:
        require(set(train["checkpoint_groups"][str(at)])==set(arms), "Exact scheduled checkpoint arms")
        for arm in arms:
            record=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(record)),map_location="cpu",weights_only=False)
            require(cp.get("schema")=="H058_full_parent_checkpoint_v1"
                and cp.get("hypothesis_id")=="H058" and cp.get("phase")=="train"
                and cp.get("producer_ref")==actual_ref(SOURCE)
                and cp.get("arm")==arm and cp.get("pass_index")==at
                and cp.get("additional_optimizer_steps")==32*at and cp.get("seed")==101
                and cp.get("parameter_count")==PARAMETERS[arm] and cp.get("optimizer_state_reset") is True
                and cp.get("teacher_used") is False and cp.get("initial_state_sha256")==train["initial_state_sha256"][arm]
                and cp.get("observation_seal")==train["observations_seal"]
                and cp.get("input_ref")==train["input_ref"] and cp.get("config_ref")==train["config_ref"]
                and cp.get("scientific_protocol")==fixed_protocol_v1()
                and cp.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT_REF
                and cp.get("warmstart_completion_ref")==WARM_COMPLETION_REF, "Exact checkpoint lineage")
            require(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in cp["model_state"].values()),
                "Finite original raw FP32 checkpoint state")
            require(cp["optimizer_state"]["state"]
                and all(int(v["step"].item())==32*at for v in cp["optimizer_state"]["state"].values()),
                "Genuine scheduled checkpoint optimizer counters")
            states[arm,at],refs[arm,at]=cp["model_state"],record
    return train,states,refs


def load_reused_continuation_v1(ctx,reader):
    cfg=ctx["cfg"];authority=reader.decode_json(cfg["continuation_replay_directive_ref"])
    require(authority.get("root_authorizer")=="root_agent"
        and authority.get("checkpoint_only_prediction_authorized") is True
        and authority.get("optimizer_updates")==0 and authority.get("required_products")==48
        and authority.get("inference_backend")==cfg["evaluation_backend"],
        "Separate actual ROOT authority for 48 fixed-checkpoint predictions, never control training")
    attestation=reader.decode_json(cfg["continuation_reuse_attestation_ref"])
    require(attestation.get("root_authorizer")=="root_agent"
        and attestation.get("frozen_TRAIN_checkpoint_series_reuse_approved") is True
        and all(attestation.get(k) is True for k in ("initializer_identity","corpus_and_episode_identity",
            "original_backbone_objective_identity","optimizer_order_budget_identity",
            "runtime_determinism_identity","all_six_checkpoint_lineages_identity"))
        and attestation.get("candidate_scientific_protocol")==fixed_protocol_v1(),
        "Authenticated actual control reuse identity; equal update counts are insufficient")
    train=reader.decode_json(cfg["H050_train_completion_ref"])
    require(train.get("schema")=="H050_four_arm_train_completion_v1"
        and train.get("steps",{}).get(CONTINUATION)==960
        and train.get("parameters_per_arm",{}).get(CONTINUATION)==31812
        and train.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT_REF
        and train.get("warmstart_completion_ref")==WARM_COMPLETION_REF,
        "Frozen actual completed H050 continuation series")
    states,refs={},{}
    for at in PASSES:
        record=train["checkpoint_groups"][str(at)][CONTINUATION]
        cp=torch.load(io.BytesIO(reader.read(record)),map_location="cpu",weights_only=False)
        require(cp.get("schema")=="H050_direct_readout_checkpoint_v1"
            and cp.get("arm")==CONTINUATION and cp.get("pass_index")==at
            and cp.get("additional_optimizer_steps")==32*at and cp.get("parameter_count")==31812
            and cp.get("seed")==101 and cp.get("optimizer_state_reset") is True
            and cp.get("initial_state_sha256")==train["initial_state_sha256"][CONTINUATION]
            and cp.get("observation_seal")==train["observations_seal"]
            and all(cp.get(k)==train[k] for k in ("input_ref","config_ref","registration_ref","producer_ref",
                "scientific_protocol","warmstart_checkpoint_ref","warmstart_completion_ref")),
            "All exact unmodified H050 scheduled control checkpoint provenance")
        require(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in cp["model_state"].values()),
            "Finite raw FP32 control state")
        states[CONTINUATION,at],refs[CONTINUATION,at]=cp["model_state"],record
    return states,refs


def paired_criteria_v1(full):
    require(set(full)=={*ARMS,CONTINUATION}, "Both matched controls and aligned result required")
    comparisons={}
    for control in (ARMS[1],CONTINUATION):
        rows=[]
        for pid in VAL:
            cells={a:next(c for c in full[a]["cells"] if c["parent_id"]==pid)["groups"]["primary_present_day_182"]
                for a in (ARMS[0],control)}
            left,right=cells[ARMS[0]],cells[control]
            require(left["boundaries"]["f1"] is not None and right["boundaries"]["f1"] is not None,
                "Defined fixed1kb boundary metrics")
            rows.append(dict(parent_id=pid,accuracy_delta=left["accuracy"]-right["accuracy"],
                boundary_F1_delta=left["boundaries"]["f1"]-right["boundaries"]["f1"],
                nll_delta=None if left.get("nll") is None or right.get("nll") is None else left["nll"]-right["nll"],
                short_tract_recall_delta=(left["H058_tract_diagnostics"]["uncensored_true_short_recall"]
                    -right["H058_tract_diagnostics"]["uncensored_true_short_recall"])
                    if all("H058_tract_diagnostics" in cell
                        and cell["H058_tract_diagnostics"]["uncensored_true_short_recall"] is not None
                        for cell in (left,right)) else None))
        accuracy=statistics.mean(r["accuracy_delta"] for r in rows)
        boundary=statistics.mean(r["boundary_F1_delta"] for r in rows)
        nll=None if any(r["nll_delta"] is None for r in rows) else statistics.mean(r["nll_delta"] for r in rows)
        short_delta=None if any(r["short_tract_recall_delta"] is None for r in rows) else statistics.mean(r["short_tract_recall_delta"] for r in rows)
        threshold=(boundary>=.01 and accuracy>=-.0005 and nll is not None and nll<=0)
        comparisons[control]=dict(per_parent=rows,mean_accuracy_delta=accuracy,mean_boundary_F1_delta=boundary,
            mean_uncensored_true_short_recall_delta=short_delta,
            registered_thresholds_met=threshold,tract_preservation_checked=short_delta is not None,
            mean_nll_delta=nll,paired_accuracy_sample_SD_pp=100*statistics.stdev(r["accuracy_delta"] for r in rows),
            paired_boundary_F1_sample_SD_pp=100*statistics.stdev(r["boundary_F1_delta"] for r in rows),
            registered_utility=(False if not threshold else None if short_delta is None else short_delta>=0))
    verdict=False if any(v["registered_utility"] is False for v in comparisons.values()) else (
        True if all(v["registered_utility"] is True for v in comparisons.values()) else None)
    return dict(comparisons=comparisons,registered_utility=verdict,
        pending_tract_diagnostics=(verdict is None),
        old_CPU_control_extra_tract_diagnostics_require_separate_prediction_authority=(verdict is None),
        one_seed_reused8VAL_development_only=True,ancestry_improvement_beyond_this_development_sample_established=False,
        no_generic_operator_priority_claim=True)


def reuse_original_CPU_metrics_v1(ctx,reader):
    cfg=ctx["cfg"]
    require(cfg["evaluation_backend"]=="cpu", "Original H050 CPU inference must match both new arms")
    attestation=reader.decode_json(cfg["continuation_reuse_attestation_ref"])
    require(attestation.get("root_authorizer")=="root_agent"
        and attestation.get("frozen_TRAIN_checkpoint_series_reuse_approved") is True
        and attestation.get("original_CPU_evaluation_reuse_approved") is True
        and attestation.get("candidate_scientific_protocol")==fixed_protocol_v1()
        and all(attestation.get(k) is True for k in ("initializer_identity","corpus_and_episode_identity",
            "original_backbone_objective_identity","optimizer_order_budget_identity",
            "runtime_determinism_identity","all_six_checkpoint_lineages_identity",
            "CPU_inference_and_selection_scoring_identity")),
        "Actual complete ROOT identity audit before reusing old CPU-selected metrics")
    original=reader.decode_json(cfg["H050_CPU_eval_completion_ref"])
    require(original.get("optimization_performed") is False
        and original.get("outer_truth_read") is False
        and original.get("schema")=="H050_four_arm_VAL_completion_v1"
        and original.get("status")=="completed_original8VAL_four_arm_development_selection"
        and original.get("all192_RAM_seals_before_any_VAL_truth") is True
        and original.get("train_completion_ref")==cfg["H050_train_completion_ref"]
        and CONTINUATION in original.get("selected",{}),
        "Actual successful complete H050 CPU evaluation with original full pretruth barrier")
    require(original["selected"][CONTINUATION]["pass_index"] in PASSES
        and {r["pass_index"] for r in original["history"][CONTINUATION]}==set(PASSES),
        "All original six-control-checkpoint selection candidates")
    return original


def run_evaluate_v1(ctx,input_ref):
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodeGNNV1
    reader=h45.PhaseReader(ctx["reader"],"evaluate",ctx["bindings"])
    train,states,checkpoints=load_H058_checkpoints_v1(ctx,reader)
    arms=phase_arms(ctx["phase"])
    if ctx["phase"]=="evaluate" and ctx["cfg"]["continuation_mode"]=="replay_frozen_H050_checkpoints":
        control_states,control_refs=load_reused_continuation_v1(ctx,reader)
        states.update(control_states);checkpoints.update(control_refs);arms=(*ARMS,CONTINUATION)
    observed=observations.load_observations_v1(ctx,reader,VAL)
    groups={str(at):{a:checkpoints[a,at] for a in arms} for at in PASSES}
    ram=matrix.H058RAMMatrixV1(ctx["output"],reader,observed["parents"],observed["banks"],
        ctx["cfg"],checkpoint_groups=groups,arms=arms);reader.ram=ram
    models={a:(OrderedCladeEpisodeGNNV1(**MODEL,arm="own_state") if a==CONTINUATION
        else model.H058FullParentGNNV1(checkpoint_backbone=True)).to(ctx["device"]).eval().requires_grad_(False) for a in arms}
    for pid in VAL:
        bank,episode,_mass=observations.device_parent_v1(ctx,observed,pid)
        for arm in arms:
            payload,_packing=observations.payload_v1(ctx,observed,pid,arm,bank,episode)
            for at in PASSES:
                ctx["guard"]();models[arm].load_state_dict(states[arm,at],strict=True)
                with torch.no_grad(),torch.autocast(device_type=ctx["device"].type,enabled=False):
                    probability=models[arm].forward_train_holds(bank,episodes=payload,control="aligned").softmax(-1)
                ram.seal(at,arm,pid,probability.cpu().numpy(),checkpoints[arm,at])
                del probability
            del payload
        del bank,episode,_mass
    del models,states
    ram.verify()
    reader.truth_gate=True;ctx["VAL_truth_gate_opened"]=True
    selected,histories,diagnostics,receipts={},{},{},{}
    for arm in arms:
        validation=diagnostics.H058ValidationV1(observed["parents"],ctx["bindings"],reader,h45.fit.metric_helpers_v1(),ctx["cfg"])
        rows=[]
        for at in PASSES:
            ctx["guard"]()
            cells,means=validation.score([(p,ram.records[at,arm,p]) for p in VAL])
            rows.append(dict(pass_index=at,cells=cells,equal_parent_metrics=means,checkpoint=checkpoints[arm,at]))
        best=min(rows,key=lambda r:h45.h038.selection_key_v1(r["equal_parent_metrics"],r["pass_index"]))
        cells,means=validation.full_final([(p,ram.records[best["pass_index"],arm,p]) for p in VAL])
        selected[arm],histories[arm]=best,rows
        diagnostics[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        receipts[arm]=validation.receipt()
    old_CPU_reused=False
    if ctx["phase"]=="evaluate" and ctx["cfg"]["continuation_mode"]=="reuse_original_CPU_selected_metrics":
        original=reuse_original_CPU_metrics_v1(ctx,reader)
        selected[CONTINUATION]=original["selected"][CONTINUATION]
        histories[CONTINUATION]=original["history"][CONTINUATION]
        diagnostics[CONTINUATION]=original["selected_full_diagnostics"][CONTINUATION]
        receipts[CONTINUATION]=original["metric_receipts"][CONTINUATION]
        old_CPU_reused=True
    result=dict(schema="H058_full_parent_VAL_completion_v1",hypothesis_id="H058",
        status="completed_original8VAL_development_selection",input_ref=input_ref,
        train_completion_ref=ctx["train_completion_ref"],selected=selected,history=histories,
        selected_full_diagnostics=diagnostics,metric_receipts=receipts,
        complete_probability_matrix_manifest_ref=ram.barrier_ref,pretruth_product_count=len(ram.expected),
        mandatory96_new_seals_before_truth=set(ARMS).issubset(arms),
        parameters_per_arm={a:PARAMETERS[a] for a in selected},scientific_protocol=fixed_protocol_v1(),
        inference_backend=ctx["cfg"]["evaluation_backend"],old_CPU_selected_metrics_directly_reused=old_CPU_reused,
        optimization_performed=False,outer_truth_read=False,ancestry_accuracy_not_inferred_from_training_loss=True,
        registered_practical_criteria=paired_criteria_v1(diagnostics) if set(diagnostics)=={*ARMS,CONTINUATION}
            else dict(status="PENDING_MATCHED_CONTINUATION_COMPARISON",utility_verdict=None))
    ctx["output"].save("completed.json",result)
    return result


def run_context_v1(ctx,input_ref):
    validate_context_v1(ctx)
    science();validate_runtime_flags_v1(ctx);ctx["guard"]()
    if ctx["phase"].startswith("train"):return run_train_v1(ctx,input_ref)
    try:return run_evaluate_v1(ctx,input_ref)
    except BaseException as error:
        ctx["output"].save("failed.json",dict(hypothesis_id="H058",phase=ctx["phase"],
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),outer_truth_read=False,automatic_replay=False))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input-ref")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",execution=False,protocol=fixed_protocol_v1())))
        return 0
    require(args.input_ref is not None, "Actual pinned ROOT capability")
    record=json.loads(args.input_ref)
    require(set(record)=={"path","sha256"} and actual_ref(record["path"])==record, "Exact actual ROOT input bytes")
    spec=json.loads((ROOT/record["path"]).read_bytes())
    require(spec.get("schema")=="H058_full_parent_ROOT_input_v1" and spec.get("root_authorizer")=="root_agent"
        and spec.get("status")=="active" and spec.get("producer_ref")==actual_ref(SOURCE)
        and spec.get("execution_authorized") is True, "Actual separate H058 ROOT authority")
    for path,sha in spec["source_sha256"].items():
        require(actual_ref(path)==dict(path=path,sha256=sha), "Every actual root source pin")
    context=spec["root_context_module_ref"]
    require(spec["source_sha256"].get(SOURCE)==actual_ref(SOURCE)["sha256"]
        and spec["source_sha256"].get(context["path"])==context["sha256"]
        and actual_ref(context["path"])==context, "Producer and existing-framework context in actual closure")
    loader=importlib.util.spec_from_file_location("_H058_existing_ROOT_context",ROOT/context["path"])
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    ctx=module.build_context_v1(record,spec)
    require(ctx["phase"]==spec["phase"], "Actual unchanged input phase")
    run_context_v1(ctx,record)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

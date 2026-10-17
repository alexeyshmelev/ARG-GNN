"""Inactive source-only scientific TRAIN/VAL producer for the placement pair.

ROOT must separately supply an authenticated admitted context, real reader,
bounded writer and existing guard callbacks. No CLI, wrapper, process controller,
resource guard, reservation or execution authority is created by this package.
No scientific import/model operation happened during source-only authoring.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,io,json,os,random,statistics,time
from .protocol_v2 import *

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/ordered_marker_learning_integration_source_20261017_v2"
SOURCE=HERE+"/producer_v2.py"
MANIFEST=HERE+"/MANIFEST.v2.json"
REGISTRATION=HERE+"/SOURCE_REGISTRATION_SNAPSHOT.v2.json"
CAP_NAMES=("CPU_bank_cap_bytes","CPU_target_cap_bytes","episode_payload_cap_bytes",
    "episode_cache_cap_bytes","CPU_parent_episode_cap_bytes","CPU_parent_residency_cap_bytes",
    "packed_layout_cap_bytes","packed_prepare_named_cap_bytes","RAM_prediction_cap_bytes",
    "checkpoint_file_cap_bytes","output_cap_bytes","max_cpu_rss_bytes","CPU_prepared_cache_cap_bytes",
    "CPU_packed_cache_cap_bytes","CPU_packed_cache_max_tensor_objects","CPU_packed_python_allowance_bytes",
    "binding_chunk_elements","packed_max_hold_nodes","CPU_positive_plan_cap_bytes")


def actual_ref(path):
    p=ROOT/path
    require(not Path(path).is_absolute() and ".." not in Path(path).parts
        and p.resolve()==p and p.is_file() and not p.is_symlink(),"Canonical source/metadata ref")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def science():
    global torch,np,h45,shared,model,observations,matrix,diagnostics
    import numpy as np
    import torch
    from arggnn import train_ordered_clade_episode_v1 as h45
    from arggnn import H040_latent_clade_execution_v1 as shared
    from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1 import diagnostics_v1 as diagnostics
    from . import ordered_learning_gnn_v2 as model,observations_v2 as observations,matrix_v2 as matrix
    h45.science()


def validate_context_v2(ctx):
    """Scientific/context join only; existing ROOT framework owns real authority."""
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("hypothesis_id")==HYPOTHESIS and ctx.get("phase") in ("train","evaluate")
        and all(callable(ctx.get(k)) for k in ("guard","before_episode_build")),
        "Separately admitted ROOT context required; this module creates no authority")
    ctx["guard"]() # Existing authority boundary occurs before scientific imports.
    cfg=ctx["cfg"];phase=ctx["phase"]
    require(cfg.get("schema")=="ordered_marker_placement_scientific_config_v2"
        and cfg.get("execution_enabled") is True and cfg.get("scientific_protocol")==fixed_protocol_v2()
        and cfg.get("producer_ref")==actual_ref(SOURCE)
        and cfg.get("source_manifest_ref")==actual_ref(MANIFEST)
        and cfg.get("registration_snapshot_ref")==actual_ref(REGISTRATION)
        and cfg.get("observation_cache_ref")==OBSERVATION
        and cfg.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT_REF
        and cfg.get("warmstart_completion_ref")==WARM_COMPLETION_REF
        and cfg.get("root_resource_caps_reviewed") is True
        and all(type(cfg.get(k)) is int and cfg[k]>0 for k in CAP_NAMES)
        and cfg["binding_chunk_elements"]<=1048576
        and cfg.get("checkpoint_backbone") is True and cfg.get("checkpoint_added") is True
        and cfg.get("evaluation_backend")=="cpu" and cfg.get("max_cuda_reserved_bytes")==0
        and cfg.get("CPU_training_threads")==32 and cfg.get("CPU_training_interop_threads")==1
        and cfg.get("CPU_prepared_cache_cap_bytes")==32*2**30
        and cfg.get("max_cpu_rss_bytes")==128*2**30,
        "Exact complete protocol, new source refs, FP32/CPU condition and reviewed unchanged aggregate caps")
    require(ctx["reader"].decode_json(ctx["config_ref"])==cfg
        and ctx["reader"].root.resolve()==ROOT and set(ctx["entry_refs"])==set(TRAIN+VAL)
        and set(ctx["bindings"])==set(TRAIN if phase=="train" else VAL)
        and ctx.get("observation_cache_ref")==OBSERVATION,
        "Exact original corpus and phase-only target capability")
    require(ctx["device"].type=="cpu" and ctx.get("physical_gpu") is None
        and os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only original condition")
    old=ctx["reader"].decode_json(H050_INPUT_REF)
    old_cfg=ctx["reader"].decode_json(old["refs"]["config"])
    target_record=H050_TRAIN_BINDINGS_REF if phase=="train" else H050_VAL_BINDINGS_REF
    require(cfg["metrics"]==old_cfg["metrics"]
        and ctx["bindings"]==ctx["reader"].decode_json(target_record)["parents"]
        and ctx["entry_refs"]==old["observation_entry_refs"]
        and ctx["observation_source_refs"]==old["observation_cache_source_refs"],
        "Unchanged original targets, all40 observed entries, cohort/scoring support")
    snapshot=ctx["reader"].decode_json(cfg["registration_snapshot_ref"])
    require(snapshot.get("id")==HYPOTHESIS and snapshot.get("scientific_protocol")==fixed_protocol_v2()
        and snapshot.get("execution_authority") is False,
        "Exact immutable source-registration snapshot; it grants no execution authority")
    # A source candidate is not an experiment registration. ROOT must supply the
    # real immutable experiment record through the admitted configuration.
    registration=ctx["reader"].decode_json(cfg["experiment_registration_ref"])
    require(registration.get("id")==HYPOTHESIS
        and registration.get("scientific_protocol")==fixed_protocol_v2()
        and registration.get("source_registration_snapshot_ref")==cfg["registration_snapshot_ref"]
        and registration.get("root_authorizer")=="root_agent",
        "Actual separately supplied experiment registration, never the mutable ledger")
    for record in ctx["reader"].decode_json(cfg["source_manifest_ref"])["refs"]:
        ctx["reader"].read(record)
    return cfg


def runtime_flags_v2():
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not torch.cuda.is_initialized(),
        "Explicit CPU-only environment")
    return dict(backend="cpu",intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),default_dtype=str(torch.get_default_dtype()),
        CUDA_initialized=False,torch_version=torch.__version__,numpy_version=np.__version__,
        PyG_version=__import__("torch_geometric").__version__)


def validate_runtime_flags_v2(ctx):
    expected=dict(backend="cpu",intraop_threads=32,interop_threads=1,
        default_dtype="torch.float32",CUDA_initialized=False,torch_version="2.6.0+cu124",
        numpy_version="2.4.6",PyG_version="2.7.0")
    require(runtime_flags_v2()==expected,"Exact original CPU FP32/32threads/interop1 runtime")
    declaration=ctx["reader"].decode_json(ctx["cfg"]["runtime_equivalence_ref"])
    require(declaration.get("root_authorizer")=="root_agent" and declaration.get("flags")==expected
        and declaration.get("CPU_entry_ref")==actual_ref(SOURCE)
        and declaration.get("extra_qualification_or_probe_authorized") is False,
        "Future source-bound ROOT runtime declaration; no inherited model fit claimed")


def state_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scheduled_step_v2(ctx,arm,value,optimizer,bank,mass,payload,steps,first):
    model.validate_trainable_fp32_v2(value)
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
        observations.validate_scheduled_v2(ctx,bank,payload)
        row=h45.train_one(value,optimizer,bank,mass,payload,ctx["guard"],
            qualify=first,after_optimizer_step=counted)
        observations.validate_scheduled_v2(ctx,bank,payload)
    finally:
        for handle in handles:handle.remove()
    if first:
        named=dict(value.named_parameters())
        require(set(checks)==set(named) and all(v["dtype"]=="torch.float32" and v["finite"] for v in checks.values()),
            "Every allocated parameter has a finite FP32 first scheduled gradient")
        require(len(optimizer.state)==len(named)
            and all(int(v["step"].item())==1 for v in optimizer.state.values()), "Fresh actual full-model AdamW step1")
        changes={k:not torch.equal(p.detach().cpu(),before[k]) for k,p in named.items()}
        require(any(changed for k,changed in changes.items() if not k.startswith(("same_marker.","ordered_marker."))),
            "Actual original H045 body updated")
        row["first_scheduled_step"]=dict(gradient_checks=checks,updated_parameters=changes,
            optimizer_steps=steps[arm],no_extra_updates=True,retained_in_full960=True,
            parameter_count=PARAMETERS[arm],runtime_flags=runtime_flags_v2())
        row["first_scheduled_step"]["CPU_runtime_cache"]=observations.scheduled_CPU_receipt_v2(ctx)
        require("logits" in captures, "Logits captured from the actual scheduled forward")
    return row,captures.get("logits")


def save_checkpoint_v2(ctx,input_ref,arm,at,models,optimizers,steps,hashes,seal):
    require(len(optimizers[arm].state)==len(list(models[arm].parameters()))
        and all(int(value["step"].item())==steps[arm] for value in optimizers[arm].state.values()),
        "Actual saved AdamW counters match all scheduled updates")
    checkpoint=dict(schema="ordered_marker_placement_checkpoint_v2",hypothesis_id=HYPOTHESIS,
        phase="train",arm=arm,pass_index=at,additional_optimizer_steps=steps[arm],
        seed=101,parameter_count=PARAMETERS[arm],optimizer_state_reset=True,teacher_used=False,
        model_state={k:v.detach().cpu() for k,v in models[arm].state_dict().items()},
        optimizer_state=optimizers[arm].state_dict(),initial_state_sha256=hashes[arm],
        input_ref=input_ref,config_ref=ctx["config_ref"],observation_seal=seal,
        scientific_protocol=fixed_protocol_v2(),producer_ref=actual_ref(SOURCE),
        source_registration_snapshot_ref=actual_ref(REGISTRATION),source_manifest_ref=actual_ref(MANIFEST),
        warmstart_checkpoint_ref=WARM_CHECKPOINT_REF,warmstart_completion_ref=WARM_COMPLETION_REF)
    buffer=io.BytesIO();torch.save(checkpoint,buffer)
    require(buffer.tell()<=ctx["cfg"]["checkpoint_file_cap_bytes"], "Bounded real checkpoint bytes")
    return ctx["output"].save("%s.pass_%02d.pt"%(arm,at),checkpoint,"torch")


def run_train_v2(ctx,input_ref):
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    arms=ARMS;reader=h45.PhaseReader(ctx["reader"],"train",ctx["bindings"])
    output=ctx["output"];steps={a:0 for a in arms};groups={};history=[];qualified={}
    try:
        observed=observations.load_observations_v2(ctx,reader,TRAIN)
        completion=reader.decode_json(WARM_COMPLETION_REF)
        warm=torch.load(io.BytesIO(reader.read(WARM_CHECKPOINT_REF)),map_location="cpu",weights_only=False)
        models,optimizers,initialization=model.build_family_v2(warm,completion,
            checkpoint_ref=WARM_CHECKPOINT_REF,completion_ref=WARM_COMPLETION_REF,
            device=ctx["device"],arms=arms,checkpoint_backbone=True)
        del warm
        hashes={a:state_hash(m.state_dict()) for a,m in models.items()}
        if arms==ARMS:require(len(set(hashes.values()))==1, "Identical placement-arm initial learned states")
        reader.truth_gate=True;targets={};total=0;helpers=h45.fit.metric_helpers_v1()
        for pid in TRAIN:
            ctx["guard"]()
            target=FullPanelExternalTargetV1(observed["parents"][pid],ctx["bindings"][pid],"TRAIN_loss",reader)
            mass=h45.h038.native_hold_train_mass_v1(observed["parents"][pid],observed["banks"][pid],target,helpers)
            total+=mass.nbytes
            require(total<=ctx["cfg"]["CPU_target_cap_bytes"], "Complete exact native-hold target cap")
            targets[pid]=torch.from_numpy(mass)
            del mass,target
        output.save("initialization.json",dict(hypothesis_id=HYPOTHESIS,initialization=initialization,
            initial_state_sha256=hashes,observations_seal=observed["seal"],
            comparison="ordered_residual_placement_only",VAL_influences_optimization=False))
        for at in range(1,31):
            order=list(TRAIN);random.Random(101+at).shuffle(order)
            means={a:0. for a in arms};started=time.monotonic()
            for number,pid in enumerate(order,1):
                parent_start=time.monotonic()
                bank,episode,mass=observations.device_parent_v1(ctx,observed,pid,targets[pid])
                first=at==1 and number==1;reference=None
                for arm in arms:
                    require(all(p.grad is None for m in models.values() for p in m.parameters()), "No cross-arm live gradients")
                    payload,packing=observations.payload_v2(ctx,observed,pid,arm,bank,episode)
                    row,initial=scheduled_step_v2(ctx,arm,models[arm],optimizers[arm],bank,mass,payload,steps,first)
                    means[arm]+=row["loss"]/32.
                    if first:
                        if reference is None:reference=initial
                        require(initial.shape==reference.shape and torch.allclose(initial,reference,rtol=1e-5,atol=1e-6),
                            "Actual initial placement-arm scheduled predictors agree")
                        qualified[arm]=dict(row,parent_id=pid,packed_preparation=packing,
                            parent_seconds_to_receipt=time.monotonic()-parent_start,
                            initial_pair_max_abs_difference=float((initial-reference).abs().max()))
                        # Commit each successful real update's evidence immediately.
                        output.save("%s.first_scheduled_step.json"%arm,qualified[arm])
                    del payload,packing,initial
                del reference,bank,episode,mass
                if number%8==0:print(json.dumps(dict(stage="ordered_marker_placement_TRAIN",pass_index=at,parents=number,steps=steps)),flush=True)
            require(steps=={a:32*at for a in arms}, "All32 complete parent updates per arm/pass")
            row=dict(pass_index=at,steps=steps.copy(),parent_order=order,mean_TRAIN_NLL=means,
                seconds=time.monotonic()-started,full_parent_updates_per_arm=32)
            history.append(row);output.save("TRAIN_pass_%02d.json"%at,row)
            if at in PASSES:
                groups[str(at)]={a:save_checkpoint_v2(ctx,input_ref,a,at,models,optimizers,steps,hashes,observed["seal"]) for a in arms}
        require(steps=={a:960 for a in arms} and sum(map(len,groups.values()))==6*len(arms), "Full30pass budget and all six checkpoints")
        result=dict(schema="ordered_marker_placement_train_completion_v2",hypothesis_id=HYPOTHESIS,
            status="completed_fixed30pass_full_TRAIN",input_ref=input_ref,config_ref=ctx["config_ref"],
            arms=list(arms),steps=steps,history=history,checkpoint_groups=groups,
            first_scheduled_step=qualified,initial_state_sha256=hashes,observations_seal=observed["seal"],
            parameters_per_arm={a:PARAMETERS[a] for a in arms},scientific_protocol=fixed_protocol_v2(),
            producer_ref=actual_ref(SOURCE),source_registration_snapshot_ref=actual_ref(REGISTRATION),source_manifest_ref=actual_ref(MANIFEST),
            warmstart_checkpoint_ref=WARM_CHECKPOINT_REF,warmstart_completion_ref=WARM_COMPLETION_REF,
            VAL_truth_read=False,outer_truth_read=False,disposable_updates=0)
        output.save("completed.json",result);return result
    except BaseException as error:
        output.save("failed.json",dict(hypothesis_id=HYPOTHESIS,phase=ctx["phase"],status="failed_partial_preserved",
            logical_steps=steps,completed_checkpoint_groups=groups,error_type=type(error).__name__,
            error=str(error),automatic_replay=False,VAL_truth_read=False,outer_truth_read=False))
        raise


def load_placement_checkpoints_v2(ctx,reader):
    train=reader.decode_json(ctx["train_completion_ref"]);arms=ARMS
    require(train.get("schema")=="ordered_marker_placement_train_completion_v2"
        and train.get("status")=="completed_fixed30pass_full_TRAIN"
        and train.get("arms")==list(arms) and train.get("steps")=={a:960 for a in arms}
        and train.get("scientific_protocol")==fixed_protocol_v2()
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
            require(cp.get("schema")=="ordered_marker_placement_checkpoint_v2"
                and cp.get("hypothesis_id")==HYPOTHESIS and cp.get("phase")=="train"
                and cp.get("producer_ref")==actual_ref(SOURCE)
                and cp.get("source_registration_snapshot_ref")==actual_ref(REGISTRATION)
                and cp.get("source_manifest_ref")==actual_ref(MANIFEST)
                and cp.get("arm")==arm and cp.get("pass_index")==at
                and cp.get("additional_optimizer_steps")==32*at and cp.get("seed")==101
                and cp.get("parameter_count")==PARAMETERS[arm] and cp.get("optimizer_state_reset") is True
                and cp.get("teacher_used") is False and cp.get("initial_state_sha256")==train["initial_state_sha256"][arm]
                and cp.get("observation_seal")==train["observations_seal"]
                and cp.get("input_ref")==train["input_ref"] and cp.get("config_ref")==train["config_ref"]
                and cp.get("scientific_protocol")==fixed_protocol_v2()
                and cp.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT_REF
                and cp.get("warmstart_completion_ref")==WARM_COMPLETION_REF, "Exact checkpoint lineage")
            require(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in cp["model_state"].values()),
                "Finite original raw FP32 checkpoint state")
            require(cp["optimizer_state"]["state"]
                and all(int(v["step"].item())==32*at for v in cp["optimizer_state"]["state"].values()),
                "Genuine scheduled checkpoint optimizer counters")
            states[arm,at],refs[arm,at]=cp["model_state"],record
    return train,states,refs


def paired_criteria_v2(full):
    require(set(full)==set(ARMS),"Both placement-arm selected full diagnostics required")
    rows=[]
    for pid in VAL:
        cells={a:next(c for c in full[a]["cells"] if c["parent_id"]==pid)["groups"]["primary_present_day_182"]
            for a in ARMS}
        left,right=cells[ARMS[0]],cells[ARMS[1]]
        require(left["boundaries"]["f1"] is not None and right["boundaries"]["f1"] is not None,
            "Original fixed1kb boundary scores required")
        short=[c.get("H058_tract_diagnostics",{}).get("uncensored_true_short_recall") for c in (left,right)]
        rows.append(dict(parent_id=pid,accuracy_delta=left["accuracy"]-right["accuracy"],
            boundary_F1_delta=left["boundaries"]["f1"]-right["boundaries"]["f1"],
            nll_delta=None if left.get("nll") is None or right.get("nll") is None else left["nll"]-right["nll"],
            short_tract_recall_delta=None if any(x is None for x in short) else short[0]-short[1]))
    accuracy=statistics.mean(r["accuracy_delta"] for r in rows)
    boundary=statistics.mean(r["boundary_F1_delta"] for r in rows)
    nll=None if any(r["nll_delta"] is None for r in rows) else statistics.mean(r["nll_delta"] for r in rows)
    short=None if any(r["short_tract_recall_delta"] is None for r in rows) else statistics.mean(r["short_tract_recall_delta"] for r in rows)
    threshold=boundary>=.01 and accuracy>=-.0005 and nll is not None and nll<=0
    return dict(comparison=ARMS[0]+"_minus_"+ARMS[1],per_parent=rows,
        mean_accuracy_delta=accuracy,mean_boundary_F1_delta=boundary,mean_nll_delta=nll,
        mean_uncensored_true_short_recall_delta=short,
        paired_accuracy_sample_SD_pp=100*statistics.stdev(r["accuracy_delta"] for r in rows),
        paired_boundary_F1_sample_SD_pp=100*statistics.stdev(r["boundary_F1_delta"] for r in rows),
        prospective_utility=False if not threshold else None if short is None else short>=0,
        one_seed_reused8VAL_development_only=True,placement_not_general_order_contrast=True,
        speed_efficacy_or_novelty_established=False)


def run_evaluate_v2(ctx,input_ref):
    reader=h45.PhaseReader(ctx["reader"],"evaluate",ctx["bindings"])
    train,states,checkpoints=load_placement_checkpoints_v2(ctx,reader)
    arms=ARMS
    observed=observations.load_observations_v2(ctx,reader,VAL)
    groups={str(at):{a:checkpoints[a,at] for a in arms} for at in PASSES}
    ram=matrix.OrderedMarkerRAMMatrixV2(ctx["output"],reader,observed["parents"],observed["banks"],
        ctx["cfg"],checkpoint_groups=groups,arms=arms);reader.ram=ram
    models={a:model.OrderedMarkerLearningGNNV2(arm=a).to(ctx["device"]).eval().requires_grad_(False) for a in arms}
    for pid in VAL:
        bank,episode,_mass=observations.device_parent_v1(ctx,observed,pid)
        for arm in arms:
            payload,_packing=observations.payload_v2(ctx,observed,pid,arm,bank,episode)
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
    selected,histories,selected_diagnostics,receipts={},{},{},{}
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
        selected_diagnostics[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        receipts[arm]=validation.receipt()
    result=dict(schema="ordered_marker_placement_VAL_completion_v2",hypothesis_id=HYPOTHESIS,
        status="completed_original8VAL_development_selection",input_ref=input_ref,
        train_completion_ref=ctx["train_completion_ref"],selected=selected,history=histories,
        selected_full_diagnostics=selected_diagnostics,metric_receipts=receipts,
        complete_probability_matrix_manifest_ref=ram.barrier_ref,pretruth_product_count=len(ram.expected),
        all96_new_products_before_either_VAL_truth=True,parameters_per_arm=PARAMETERS,
        scientific_protocol=fixed_protocol_v2(),producer_ref=actual_ref(SOURCE),
        source_registration_snapshot_ref=actual_ref(REGISTRATION),source_manifest_ref=actual_ref(MANIFEST),
        inference_backend="cpu",optimization_performed=False,outer_truth_read=False,
        ancestry_accuracy_not_inferred_from_training_loss=True,
        prospective_practical_criteria=paired_criteria_v2(selected_diagnostics))
    ctx["output"].save("completed.json",result)
    return result


def run_context_v2(ctx,input_ref):
    """Real scientific caller; ROOT-only process/admission integration is absent."""
    validate_context_v2(ctx)
    science();validate_runtime_flags_v2(ctx);ctx["guard"]()
    if ctx["phase"]=="train":return run_train_v2(ctx,input_ref)
    try:return run_evaluate_v2(ctx,input_ref)
    except BaseException as error:
        ctx["output"].save("failed.json",dict(hypothesis_id=HYPOTHESIS,phase=ctx["phase"],
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),
            outer_truth_read=False,automatic_replay=False))
        raise

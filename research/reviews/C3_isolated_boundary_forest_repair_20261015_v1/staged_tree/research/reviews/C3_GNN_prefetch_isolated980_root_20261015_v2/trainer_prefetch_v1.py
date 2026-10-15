"""Scientific callable for the fixed fresh-C3 20-parent/600-update condition.

No CLI, admission, launcher, storage policy or installation. Root sets title and
checks its actual lease before importing Torch/PyG or calling this module, and
supplies an admitted output writer plus a fast resource callback. This source
does not grant execution authority.
"""
from __future__ import annotations
import copy,random
import numpy as np
import torch
from arggnn.american_c3_common_v1 import require,ref,bound,CLASSES
from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from arggnn.american_c3_gnn_run_isolated_v2 import checkpoint_payload
from arggnn.american_c3_gnn_targets_v1 import REGISTRY_REF,TRAIN,TrainTargetJoinV1,authoritative_registry_v1
from arggnn.american_c3_gnn_stream_train_isolated_v2 import (
    BASELINE,json_sha,SealedObservationProviderV1,checkpointed_complete_parent_step_v1)

CONDITION="fresh_C3_supervised_short_independent_complete_parents"
SCHEMA="american_c3_checkpointed_complete_parent_training_v1"
CHECKPOINT_EPOCHS=(5,10,15,20,25,30)
SOURCES=("src/arggnn/american_c3_common_v1.py","src/arggnn/american_c3_gnn_targets_v1.py",
    "src/arggnn/american_c3_gnn_stream_train_isolated_v2.py","research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/trainer_prefetch_v1.py",
    "src/arggnn/american_c3_gnn_model_v1.py","src/arggnn/american_c3_gnn_run_isolated_v2.py",
    "src/arggnn/american_c3_gnn_dataset_isolated_v2.py","src/arggnn/american_c3_gnn_observations_isolated_v2.py",
    "src/arggnn/american_c3_evaluate_v1.py","src/arggnn/mutation_routed_whole_tree_gnn_v1.py",
    "src/arggnn/mutation_routed_tree_observations_v1.py","src/arggnn/mutation_routed_pyg_gnn_v1.py")
DEFAULTS=dict(schema=SCHEMA,execution_enabled=False,registry_ref=REGISTRY_REF,
    backbone=BASELINE,H043_extension=None,condition=CONDITION,class_order=list(CLASSES),
    model_seed=70001,parent_order_seed=70002,epochs=30,parents_per_epoch=20,total_updates=600,
    training_length_bp=2000000,focals_per_parent=10000,focal_exposures=6000000,precision="fp32",
    model_kwargs=dict(hidden_dim=32,class_order=list(CLASSES),expected_haplotypes=1600,
        donor_cohorts=2,site_chunk_size=256,checkpoint_sites=True),
    optimizer=dict(name="torch.optim.AdamW",lr=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8),
    gradient_clip=1.,scheduler="constant_learning_rate",checkpoint_epochs=list(CHECKPOINT_EPOCHS),
    early_stopping=False,purity_filter=False,CPU_observation_cache_bytes=536870912,
    focal_chunk_size=1024,max_truth_uncompressed_bytes=None)
# Cache/chunk/truth caps are prospective root-configured bounds, never fit claims.

def validate_config_v1(cfg):
    require(isinstance(cfg,dict) and set(cfg)==set(DEFAULTS),"Exact scientific configuration keys")
    adjustable={"execution_enabled","CPU_observation_cache_bytes","focal_chunk_size","max_truth_uncompressed_bytes"}
    require(all(cfg[k]==v for k,v in DEFAULTS.items() if k not in adjustable)
        and cfg["execution_enabled"] is True,"Fixed inactive protocol needs explicit root activation")
    require(type(cfg["CPU_observation_cache_bytes"]) is int and cfg["CPU_observation_cache_bytes"]>=0
        and type(cfg["focal_chunk_size"]) is int and 0<cfg["focal_chunk_size"]<=10000
        and type(cfg["max_truth_uncompressed_bytes"]) is int and cfg["max_truth_uncompressed_bytes"]>0,
        "Root must supply bounded prospective cache/chunk/private-read caps")
    authoritative_registry_v1(cfg["registry_ref"])

def epoch_order_v1(epoch):
    require(type(epoch) is int and 1<=epoch<=30,"Fixed one-based training epoch")
    ids=list(TRAIN);random.Random(70002+epoch).shuffle(ids);return ids

def _cpu_snapshot(value):
    if torch.is_tensor(value):return value.detach().to("cpu").clone()
    if isinstance(value,dict):return {k:_cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_snapshot(v) for v in value)
    require(value is None or isinstance(value,(str,int,float,bool)),"weights_only-safe checkpoint state")
    return value

def _rng_state(device):
    nr=np.random.get_state()
    return dict(python=random.getstate(),
        numpy=dict(algorithm=nr[0],keys=nr[1].tolist(),position=int(nr[2]),
            has_gauss=int(nr[3]),cached_gaussian=float(nr[4])),
        torch_CPU=torch.random.get_rng_state().clone(),
        torch_device=torch.cuda.get_rng_state(device).cpu().clone() if device.type=="cuda" else None,
        device_type=device.type)

def _restore_rng(state,device):
    require(state["device_type"]==device.type,"Resume on same CPU/CUDA runtime kind")
    random.setstate(state["python"]);nr=state["numpy"]
    np.random.set_state((nr["algorithm"],np.asarray(nr["keys"],dtype=np.uint32),
        nr["position"],nr["has_gauss"],nr["cached_gaussian"]))
    torch.random.set_rng_state(state["torch_CPU"])
    if device.type=="cuda":torch.cuda.set_rng_state(state["torch_device"],device)

def _seed_fresh(device):
    random.seed(70001);np.random.seed(70001)
    # Seed CPU initialization without requesting states/contexts on other GPUs.
    torch.random.default_generator.manual_seed(70001)
    if device.type=="cuda":
        with torch.cuda.device(device):torch.cuda.manual_seed(70001)

def _signature(cfg,parent_plan_refs):
    scientific=dict(cfg);scientific.pop("execution_enabled")
    value=dict(schema=SCHEMA,config=scientific,parent_plan_refs=parent_plan_refs,
        source_refs=[ref(p) for p in SOURCES],parent_order="random.Random(70002+one_based_epoch).shuffle(registry_order)",
        loss="unweighted_mean_CE_all10000_common_query_focals_per_complete_parent",
        exact_gradient="checkpoint_encoder_and_directional_GRUs_CPU_boundaries_without_detach",
        no_C4_initialization=True,H043_extension=None)
    return value,json_sha(value)

def _validate_progress(state):
    step=state["global_step"]
    require(type(step) is int and 0<step<=600 and len(state["history"])==step
        and state["focal_exposures"]==step*10000,"Complete-parent cursor/history/exposure state")
    expected=[]
    for epoch in range(1,31):
        for index,parent in enumerate(epoch_order_v1(epoch)):
            if len(expected)<step:expected.append((epoch,index,parent))
    for i,(row,(epoch,index,parent)) in enumerate(zip(state["history"],expected)):
        require(row["global_step"]==i+1 and row["epoch"]==epoch and row["parent_index"]==index
            and row["parent_id"]==parent and row["records"]==10000
            and row["whole_parent_optimizer_updates"]==1
            and row["block_reset_or_truncated_BPTT"] is False,"Exact completed update trace")
    next_epoch=step//20+1;next_index=step%20
    require(state["next_epoch"]==next_epoch and state["next_parent_index"]==next_index
        and state["next_parent_order"]==(epoch_order_v1(next_epoch) if next_epoch<=30 else [])
        and state["scheduler"]==dict(name="constant_learning_rate",completed_parent_updates=step,lr=.001)
        and len(state["scheduled_checkpoints"]) in
            (sum(e*20<step for e in CHECKPOINT_EPOCHS),sum(e*20<=step for e in CHECKPOINT_EPOCHS)),
        "Exact resume order and constant schedule state")
    scheduled=state["scheduled_checkpoints"]
    expected_epochs=[e for e in CHECKPOINT_EPOCHS if e*20<step]
    if len(scheduled)>len(expected_epochs):expected_epochs.append(step//20)
    require([r["epoch"] for r in scheduled]==expected_epochs
        and all(r["global_step"]==r["epoch"]*20 and isinstance(r["checkpoint_ref"],dict)
            and set(r["checkpoint_ref"])=={"path","sha256"} for r in scheduled),
        "Every prior scheduled epoch retained in chronological order")
    return step

def _make_snapshot(model,optimizer,*,signature,signature_sha,observation_sha,target_sha,
        observation_manifest_ref,target_manifest_ref,history,scheduled,device):
    step=len(history);trained=[p for p in TRAIN if any(r["parent_id"]==p for r in history)]
    require(step>0,"A resume checkpoint requires at least one actual completed parent")
    value=checkpoint_payload(model,model_seed=70001,training_condition=CONDITION,
        trained_parent_ids=trained,train_sequence_lengths_bp=[2000000]*len(trained),optimizer=optimizer)
    value["trainer_state"]=dict(schema=SCHEMA,protocol=signature,protocol_sha256=signature_sha,
        observation_seal_sha256=observation_sha,target_join_sha256=target_sha,
        observation_manifest_ref=observation_manifest_ref,target_manifest_ref=target_manifest_ref,
        global_step=step,next_epoch=step//20+1,next_parent_index=step%20,
        next_parent_order=epoch_order_v1(step//20+1) if step<600 else [],
        focal_exposures=step*10000,history=history,
        scheduler=dict(name="constant_learning_rate",completed_parent_updates=step,lr=.001),
        rng_state=_rng_state(device),scheduled_checkpoints=scheduled)
    value["candidate"]=dict(eligible_for_VAL_selection=False,kind="complete_parent_resume_boundary",
        completed_epoch=step//20,global_step=step)
    return _cpu_snapshot(value)

def scheduled_checkpoint_metadata_v1(value,*,protocol_sha256=None):
    state=value.get("trainer_state",{});step=_validate_progress(state)
    require(value.get("schema")=="american_c3_fresh_whole_tree_GNN_checkpoint_v1"
        and value.get("C4_initialization_or_head_slicing") is False and value.get("model_seed")==70001
        and value.get("training_condition")==CONDITION and state.get("schema")==SCHEMA
        and json_sha(state["protocol"])==state["protocol_sha256"]
        and (protocol_sha256 is None or protocol_sha256==state["protocol_sha256"])
        and value.get("candidate")==dict(eligible_for_VAL_selection=True,kind="scheduled_epoch",
            completed_epoch=step//20,global_step=step)
        and step%20==0 and step//20 in CHECKPOINT_EPOCHS
        and len(state["scheduled_checkpoints"])==sum(e*20<step for e in CHECKPOINT_EPOCHS)
        and value.get("trained_parent_ids")==list(TRAIN) and value.get("train_sequence_lengths_bp")==[2000000]*20,
        "Only six frozen complete scheduled-epoch fresh C3 checkpoints are VAL candidates")
    return dict(epoch=step//20,global_step=step,protocol_sha256=state["protocol_sha256"],
        observation_seal_sha256=state["observation_seal_sha256"],target_join_sha256=state["target_join_sha256"])

def _resume(record,model,optimizer,*,signature,signature_sha,observation_sha,target_sha,device):
    value=torch.load(bound(record),map_location="cpu",weights_only=True,mmap=True)
    state=value.get("trainer_state",{});step=_validate_progress(state)
    require(value.get("schema")=="american_c3_fresh_whole_tree_GNN_checkpoint_v1"
        and value.get("contract")==model.checkpoint_contract() and value.get("model_seed")==70001
        and value.get("training_condition")==CONDITION and value.get("C4_initialization_or_head_slicing") is False
        and state.get("schema")==SCHEMA and state.get("protocol")==signature
        and state.get("protocol_sha256")==signature_sha
        and state.get("observation_seal_sha256")==observation_sha
        and state.get("target_join_sha256")==target_sha and "optimizer_state" in value,
        "Resume requires same new-family protocol/source/data seals plus full optimizer/cursor/RNG state")
    trained=[p for p in TRAIN if any(r["parent_id"]==p for r in state["history"])]
    require(value["trained_parent_ids"]==trained and value["train_sequence_lengths_bp"]==[2000000]*len(trained),
        "Checkpoint records actual completed TRAIN membership")
    model.load_state_dict(value["model_state"],strict=True);optimizer.load_state_dict(value["optimizer_state"])
    for group in optimizer.param_groups:
        require(group["lr"]==.001 and group["weight_decay"]==.0001 and group["betas"]==(.9,.999)
            and group["eps"]==1e-8 and group.get("amsgrad",False) is False
            and all(group.get(k,False) is False for k in
                ("maximize","foreach","capturable","differentiable","fused")),
            "Exact resumed AdamW settings; no optimizer reset or hidden schedule")
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"Finite resumed model")
    require(len(optimizer.param_groups)==1 and len(optimizer.state)==sum(1 for p in model.parameters())
        and all(bool(torch.isfinite(t).all()) for item in optimizer.state.values()
        for t in item.values() if torch.is_tensor(t))
        and all(int(item["step"].item())==step for item in optimizer.state.values()),
        "Finite full AdamW moments and exact complete-parent step counts")
    _restore_rng(state["rng_state"],device)
    return value,state,step

def run_train_v1(cfg,*,parent_plan_refs,writer,device,worker_tmp_directory,
        resolve_train_truth,runtime_guard,resume_ref=None,train_target_joiner_factory=None,
        observation_provider_factory=None):
    """Run primary30 epochs; real first-update checks occur inside this run.

    All20 supplied plans, every observation block and all external TRAIN targets
    must validate before the fresh model can optimize. A failure preserves a
    CPU snapshot of the last COMPLETED parent boundary, never partial-step
    tensors. Root owns output admission, process supervision and continuation.
    Writer implements save(name,value,kind='json'|'torch'|'npz').
    """
    validate_config_v1(cfg)
    require(isinstance(parent_plan_refs,dict) and set(parent_plan_refs)==set(TRAIN)
        and all(isinstance(parent_plan_refs[p],dict) and set(parent_plan_refs[p])=={"path","sha256"} for p in TRAIN)
        and callable(getattr(writer,"save",None)) and callable(runtime_guard) and callable(resolve_train_truth),
        "All20 actual SHA-bound plans and root-owned writer/resource/target capabilities required")
    actual=torch.device(device)
    require(actual.type in ("cpu","cuda") and (actual.type!="cuda" or actual.index is not None),
        "One explicit root-admitted device required")
    runtime_guard("before_model_or_payload")
    providers={};safe=None;scheduled=[];history=[];step=0
    try:
        # Constructors read plan metadata only. No resolver is reachable yet.
        provider_factory=(SealedObservationProviderV1 if observation_provider_factory is None
            else observation_provider_factory)
        require(callable(provider_factory),"Explicit optional observation provider factory")
        for parent in TRAIN:
            providers[parent]=provider_factory(parent_plan_refs[parent],parent_id=parent,
                worker_tmp_directory=worker_tmp_directory,cache_bytes=cfg["CPU_observation_cache_bytes"])
        seals={p:providers[p].seal(resource_check=runtime_guard) for p in TRAIN}
        require(len(seals)==20 and all(s["all_blocks_sealed_before_truth"] for s in seals.values()),
            "Global all20 observation-ready barrier before any private TRAIN truth resolution")
        observation=dict(schema="american_c3_all20_TRAIN_observation_barrier_v1",
            registry_ref=REGISTRY_REF,parent_ids=list(TRAIN),parents=seals,all20_ready=True,
            observation_only=True,truth_resolver_calls_before_barrier=0)
        observation_sha=json_sha(observation)
        observation_ref=writer.save("TRAIN_observation_barrier.json",observation)
        target_factory=TrainTargetJoinV1 if train_target_joiner_factory is None else train_target_joiner_factory
        require(callable(target_factory),"Root-supplied audited TRAIN target joiner")
        joiner=target_factory(resolve_train_truth,max_truth_uncompressed_bytes=cfg["max_truth_uncompressed_bytes"])
        targets={}
        for parent in TRAIN:
            runtime_guard("external_TRAIN_target_join")
            targets[parent]=joiner.join(parent,split="train",plan=providers[parent].plan,observation_seal=seals[parent])
        target_manifest=dict(schema="american_c3_all20_external_TRAIN_target_joins_v1",
            registry_ref=REGISTRY_REF,observation_seal_sha256=observation_sha,
            parents={p:targets[p].receipt for p in TRAIN},all20_targets_ready=True,
            labels_in_observation_bank=False,purity_filter=False)
        target_sha=json_sha(target_manifest)
        target_ref=writer.save("TRAIN_external_target_joins.json",target_manifest)
        signature,signature_sha=_signature(cfg,parent_plan_refs)
        if train_target_joiner_factory is not None:
            binding=copy.deepcopy(getattr(train_target_joiner_factory,"execution_binding",None))
            require(isinstance(binding,dict) and binding.get("schema")=="C3_actual_TRAIN_simulation_binding_operational_v1"
                and binding.get("registry_ref")==REGISTRY_REF and set(binding.get("parents",{}))==set(TRAIN)
                and binding.get("target_equations_unchanged") is True and binding.get("source_refs"),
                "Actual TRAIN path successor must bind unchanged registry/targets and its execution source")
            require(all(ref(r["path"])==r for r in binding["source_refs"]),"Exact operational successor sources")
            signature["actual_TRAIN_execution_binding"]=binding
            signature["source_refs"].extend(binding["source_refs"])
            signature_sha=json_sha(signature)
        if observation_provider_factory is not None:
            provider_binding=copy.deepcopy(getattr(observation_provider_factory,"execution_binding",None))
            require(isinstance(provider_binding,dict)
                and provider_binding.get("schema")=="C3_observation_provider_operational_v1"
                and provider_binding.get("observation_equations_unchanged") is True
                and provider_binding.get("checkpoint_parent_gradient_unchanged") is True
                and provider_binding.get("cached_features_rehashed") is True
                and provider_binding.get("source_refs"),"Audited optional observation provider binding")
            require(all(ref(r["path"])==r for r in provider_binding["source_refs"]),
                "Exact optional provider source closure")
            signature["observation_provider_execution_binding"]=provider_binding
            signature["source_refs"].extend(provider_binding["source_refs"])
            signature_sha=json_sha(signature)
        runtime_guard("before_fresh_model")
        require(torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled(),
            "Fresh FP32 initialization and no ambient mixed-precision training")
        _seed_fresh(actual)
        with torch.device("cpu"):
            model=AmericanC3WholeTreeGNNV1(**cfg["model_kwargs"])
        model=model.to(actual)
        opts=cfg["optimizer"]
        optimizer=torch.optim.AdamW(model.parameters(),lr=opts["lr"],weight_decay=opts["weight_decay"],
            betas=tuple(opts["betas"]),eps=opts["eps"],foreach=False,fused=False)
        if resume_ref is not None:
            value,state,step=_resume(resume_ref,model,optimizer,signature=signature,signature_sha=signature_sha,
                observation_sha=observation_sha,target_sha=target_sha,device=actual)
            history=copy.deepcopy(state["history"]);scheduled=copy.deepcopy(state["scheduled_checkpoints"])
            safe=_cpu_snapshot(value)
            # A scheduled checkpoint excludes its own ref to avoid a hash cycle.
            if value["candidate"]["eligible_for_VAL_selection"]:
                meta=scheduled_checkpoint_metadata_v1(value,protocol_sha256=signature_sha)
                scheduled.append(dict(epoch=meta["epoch"],global_step=step,checkpoint_ref=resume_ref))
            elif step%20==0 and step//20 in CHECKPOINT_EPOCHS and len(scheduled)==sum(e*20<step for e in CHECKPOINT_EPOCHS):
                # Failure after the actual update but before its scheduled save:
                # publish that frozen boundary before making another update.
                recovered=_cpu_snapshot(value)
                recovered["candidate"]=dict(eligible_for_VAL_selection=True,kind="scheduled_epoch",
                    completed_epoch=step//20,global_step=step)
                scheduled_checkpoint_metadata_v1(recovered,protocol_sha256=signature_sha)
                recovered_ref=writer.save("checkpoint_epoch_%03d.pt"%(step//20),recovered,kind="torch")
                scheduled.append(dict(epoch=step//20,global_step=step,checkpoint_ref=recovered_ref))
            require(len(scheduled)==sum(e*20<=step for e in CHECKPOINT_EPOCHS),"Resume retains every scheduled checkpoint")
            for item in scheduled:
                pinned=torch.load(bound(item["checkpoint_ref"]),map_location="cpu",weights_only=True,mmap=True)
                prior=scheduled_checkpoint_metadata_v1(pinned,protocol_sha256=signature_sha)
                require(prior["epoch"]==item["epoch"]
                    and prior["observation_seal_sha256"]==observation_sha and prior["target_join_sha256"]==target_sha,
                    "Resume retains SHA-bound scheduled candidates from this exact training condition")
                del pinned
        for global_index in range(step,600):
            epoch=global_index//20+1;index=global_index%20;parent=epoch_order_v1(epoch)[index]
            result=checkpointed_complete_parent_step_v1(model,providers[parent],targets[parent],optimizer,
                device=actual,resource_check=runtime_guard,gradient_clip=1.,
                focal_chunk_size=cfg["focal_chunk_size"],first_update=(global_index==0))
            if observation_provider_factory is not None:
                result["observation_provider"]=providers[parent].diagnostics()
            result.update(epoch=epoch,parent_index=index,global_step=global_index+1);history.append(result)
            # Snapshot before any potentially failing serialization/next update.
            candidate=_make_snapshot(model,optimizer,signature=signature,signature_sha=signature_sha,
                observation_sha=observation_sha,target_sha=target_sha,observation_manifest_ref=observation_ref,
                target_manifest_ref=target_ref,history=history,scheduled=scheduled,device=actual)
            safe=candidate;step=global_index+1
            if step%20==0 and epoch in CHECKPOINT_EPOCHS:
                candidate["candidate"]=dict(eligible_for_VAL_selection=True,kind="scheduled_epoch",
                    completed_epoch=epoch,global_step=step)
                scheduled_checkpoint_metadata_v1(candidate,protocol_sha256=signature_sha)
                checkpoint_ref=writer.save("checkpoint_epoch_%03d.pt"%epoch,candidate,kind="torch")
                scheduled.append(dict(epoch=epoch,global_step=step,checkpoint_ref=checkpoint_ref))
                # Keep a boundary resume snapshot with this scheduled ref included.
                safe=_make_snapshot(model,optimizer,signature=signature,signature_sha=signature_sha,
                    observation_sha=observation_sha,target_sha=target_sha,observation_manifest_ref=observation_ref,
                    target_manifest_ref=target_ref,history=history,scheduled=scheduled,device=actual)
        require(step==600 and len(history)==600 and len(scheduled)==6
            and [x["epoch"] for x in scheduled]==list(CHECKPOINT_EPOCHS),"Full prespecified budget, no early exit")
        return writer.save("completed_training.json",dict(status="completed_C3_checkpointed_complete_parent_training",
            schema=SCHEMA,protocol_sha256=signature_sha,registry_ref=REGISTRY_REF,
            trained_parent_ids=list(TRAIN),train_sequence_lengths_bp=[2000000]*20,
            epochs=30,complete_parent_updates=600,focal_exposures=6000000,
            observation_manifest_ref=observation_ref,target_manifest_ref=target_ref,
            observation_seal_sha256=observation_sha,target_join_sha256=target_sha,
            checkpoint_candidates=scheduled,history=history,resume_ref=resume_ref,
            selected_heldout_checkpoint=None,VAL_used_for_optimization=False,TEST_deferred=True,
            first_real_update_checks=history[0]["first_real_update_checks"],
            H043_extension=None,purity_filter=False,training_length_transfer_to_30Mb=True,
            scientific_accuracy_or_runtime_fit_claim=False))
    except BaseException as error:
        # Do not bless a partially applied optimizer step. Re-raise original.
        try:
            boundary_ref=None
            if safe is not None:
                safe["candidate"]=dict(eligible_for_VAL_selection=False,kind="complete_parent_resume_boundary",
                    completed_epoch=safe["trainer_state"]["global_step"]//20,
                    global_step=safe["trainer_state"]["global_step"])
                boundary_ref=writer.save("failure_last_complete_parent.pt",safe,kind="torch")
            failure_ref=writer.save("failed_training.json",dict(status="failed_C3_full_training_preserved",
                exception_type=type(error).__name__,message=str(error)[:2000],
                last_complete_parent_checkpoint_ref=boundary_ref,partial_update_checkpointed=False,
                full_training_budget_complete=False))
            if hasattr(error,"add_note"):error.add_note("Preserved failure receipt: "+str(failure_ref))
        except BaseException as seal_error:
            if hasattr(error,"add_note"):error.add_note("Failure sealing also failed: "+type(seal_error).__name__+": "+str(seal_error)[:500])
        raise
    finally:
        for provider in providers.values():provider.release()

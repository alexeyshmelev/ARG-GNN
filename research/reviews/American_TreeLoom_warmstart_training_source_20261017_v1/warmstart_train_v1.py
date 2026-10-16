"""Inactive matched TreeLoom transfer runner: accepted saved600 weights +600 new updates.

Reuses original20-parent C3 plans/loaders, observation seals, TRAIN target joins,
epoch order, CPU snapshot and RNG helpers. No CLI/admission/lease/guard policy.
ROOT supplies an admitted writer, one device and a fast current resource callback.
"""
from __future__ import annotations
from pathlib import Path
import copy, hashlib, json, time
import numpy as np
import torch
from arggnn.american_c3_common_v1 import require, read, bound, ref, CLASSES
from arggnn.american_c3_gnn_dataset_v1 import load_observation
from arggnn.american_c3_gnn_targets_v1 import (
    TRAIN, VAL, REGISTRY_REF, TrainTargetJoinV1, authoritative_registry_v1)
from arggnn.american_c3_gnn_stream_train_v1 import (
    SealedObservationProviderV1, json_sha, validate_plan_v1)
from arggnn.american_c3_gnn_train_v1 import (
    epoch_order_v1, _cpu_snapshot, _rng_state, _restore_rng, _seed_fresh)
from .warmstart_models_v1 import (
    ARMS, FAMILY, CONDITION, PARAMETERS, TENSORS, full,
    make_matched_family_v1, validate_arm_model_v1, train_complete_parent_step_v1)

HERE = Path(__file__).resolve().parent
SOURCE = str(Path(__file__).resolve().relative_to(
    Path("/workspace-SR003.nfs2/ARG-GNN")))
CHECKPOINT_EPOCHS = (5,10,15,20,25,30)
SCHEMA = FAMILY+"_training_v1"
ACCEPTANCE = dict(
    path="research/reviews/C3_saved600_validator_ROOT_actual_20261016_v1/artifact_acceptance.root_active_v1.json",
    sha256="5065ce083306a30a7cbf75d42f77b44db5a3d1c121b0b2e5406ac76c22f474ce")
ENGINE = dict(
    path="research/reviews/American_TreeLoom_full_mechanism_source_20261017_v2/american_treeloom_full_c3_v2.py",
    sha256="f6f4b5ec49231268ea2ae8607739ee52c03953a1b8cc6ac34729d08a3fe24f66")
ENGINE_REVIEW = dict(
    path="research/reviews/American_TreeLoom_full_mechanism_independent_review_20261017_v2/review.v2.json",
    sha256="85e8596f1ae4c005477fbf49805e690f706cd4492b21c82c4f5ee631382d0222")
DEFAULTS = dict(schema=SCHEMA, execution_enabled=False, family=FAMILY,
    condition=CONDITION, registry_ref=REGISTRY_REF, arms=list(ARMS),
    model_seed=70001, parent_order_seed=70002, epochs=30, parents_per_epoch=20,
    additional_updates_per_arm=600, inherited_saved_logical_updates=600,
    focal_exposures_per_arm=6000000, training_length_bp=2000000,
    class_order=list(CLASSES), checkpoint_epochs=list(CHECKPOINT_EPOCHS),
    optimizer=dict(name="AdamW",lr=.001,weight_decay=.0001,
        betas=[.9,.999],eps=1e-8,maximize=False,amsgrad=False),
    precision="fp32", gradient_clip=1., early_stopping=False, purity_filter=False,
    max_truth_uncompressed_bytes=None, max_native_nodes=None, max_native_holds=None,
    max_episode_payload_bytes=None, CPU_observation_cache_bytes=0)


def validate_config_v1(cfg):
    require(isinstance(cfg,dict) and set(cfg)==set(DEFAULTS), "Exact disclosed transfer config")
    operational={"execution_enabled","max_truth_uncompressed_bytes","max_native_nodes",
        "max_native_holds","max_episode_payload_bytes","CPU_observation_cache_bytes"}
    require(all(cfg[k]==v for k,v in DEFAULTS.items() if k not in operational)
        and cfg["execution_enabled"] is True,
        "Inactive by default; no science/budget/seed/objective change at execution")
    require(all(type(cfg[k]) is int and cfg[k]>0 for k in
        ("max_truth_uncompressed_bytes","max_native_nodes",
         "max_native_holds","max_episode_payload_bytes"))
        and type(cfg["CPU_observation_cache_bytes"]) is int
        and cfg["CPU_observation_cache_bytes"]>=0,
        "Explicit ROOT full-parent/target failstop bounds; no fit guarantee")
    require(ref(ENGINE["path"])==ENGINE and read(ENGINE_REVIEW)["status"]=="SOURCE_PASS",
        "Independently reviewed unchanged full-mechanism engine")


def load_accepted_backbone_v1(acceptance_ref, *, runtime_guard):
    """Real future admitted CPU checkpoint load; no old optimizer/RNG restoration."""
    require(acceptance_ref==ACCEPTANCE and callable(runtime_guard),
        "Exact explicit ROOT accepted-saved-artifact input")
    runtime_guard("before_accepted_saved600_metadata")
    acceptance=read(acceptance_ref)
    require(acceptance["schema"]=="C3_saved600_ROOT_artifact_acceptance_v1"
        and acceptance["root_authorizer"]=="root_agent"
        and acceptance["accepted_saved_artifacts_only"] is True
        and acceptance["original_invocation_success_claimed"] is False
        and acceptance["original_history"]["producer_cli_exit_code"]==1
        and acceptance["future_original_C3_training_writes_cooperatively_revoked"] is True,
        "Accepted scientific artifacts preserve original failed producer history")
    result=read(acceptance["validation_result_ref"])
    training=read(acceptance["training_completion_ref"])
    require(result["status"]=="validated_C3_saved600_artifacts_after_failed_producer"
        and result["all_six_verified"] is True
        and result["saved_effective_logical_updates"]==600
        and result["model_parameters"]==29571
        and result["original_producer_success_claimed"] is False
        and result["physical_work_exactly600_claimed"] is False
        and training["complete_parent_updates"]==600 and training["epochs"]==30
        and training["trained_parent_ids"]==list(TRAIN)
        and training["train_sequence_lengths_bp"]==[2000000]*20
        and training["VAL_used_for_optimization"] is False,
        "Exact accepted final backbone lineage, not a full TreeLoom checkpoint")
    rows=[r for r in result["checkpoints"] if r["epoch"]==30]
    candidates=[r for r in training["checkpoint_candidates"] if r["epoch"]==30]
    require(len(rows)==len(candidates)==1
        and rows[0]["global_step"]==candidates[0]["global_step"]==600
        and rows[0]["all_checks_passed"] is True
        and rows[0]["checkpoint_ref"]==candidates[0]["checkpoint_ref"],
        "Fixed final epoch30 warmstart, no validation-dependent choice")
    checkpoint_ref=candidates[0]["checkpoint_ref"]
    runtime_guard("before_accepted_saved600_CPU_payload")
    path=bound(checkpoint_ref)
    require(path.stat().st_size<=2*2**20, "Original bounded accepted backbone checkpoint")
    value=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    state=value["trainer_state"]
    require(value["schema"]=="american_c3_fresh_whole_tree_GNN_checkpoint_v1"
        and value["model_seed"]==70001
        and value["training_condition"]=="fresh_C3_supervised_short_independent_complete_parents"
        and value["contract"]["class_order"]==list(CLASSES)
        and value["contract"]["expected_haplotypes"]==1600
        and value["contract"]["hidden_dim"]==32
        and value["contract"]["site_feature_width"]==15
        and value["contract"]["observed_leaf_width"]==5
        and state["global_step"]==600 and len(state["history"])==600
        and state["protocol_sha256"]==training["protocol_sha256"]
        and state["observation_seal_sha256"]==training["observation_seal_sha256"]
        and state["target_join_sha256"]==training["target_join_sha256"],
        "Exact separately accepted C3 payload/contract/source-data lineage")
    shared={k:v.detach().cpu().clone() for k,v in value["model_state"].items()}
    require(len(shared)==38 and sum(v.numel() for v in shared.values())==29571,
        "Exact accepted backbone cardinality")
    original_plans=copy.deepcopy(state["protocol"]["parent_plan_refs"])
    original_target_binding=copy.deepcopy(state["protocol"].get("actual_TRAIN_execution_binding"))
    binding=dict(schema="TreeLoom_accepted_saved600_warmstart_binding_v1",
        acceptance_ref=copy.deepcopy(acceptance_ref),
        validation_result_ref=acceptance["validation_result_ref"],
        original_training_metadata_ref=acceptance["training_completion_ref"],
        checkpoint_ref=checkpoint_ref,epoch=30,saved_logical_updates=600,
        model_parameters=29571,accepted_saved_artifacts_only=True,
        original_producer_success_claimed=False,original_producer_cli_exit_code=1,
        original_outer_74136_unknown_in_pinned_acceptance=True,
        old_optimizer_or_rng_restored=False,prior_physical_work_exactly600_claimed=False)
    del value,state,result,training
    require(ref(checkpoint_ref["path"])==checkpoint_ref, "Accepted checkpoint unchanged after CPU load")
    runtime_guard("after_accepted_saved600_CPU_payload")
    return shared,binding,original_plans,original_target_binding


def _global_bundle_v1(provider,cfg,runtime_guard):
    plan=provider.plan
    validate_plan_v1(plan,parent_id=plan["parent_id"],length_bp=2000000)
    runtime_guard("before_global_TreeLoom_observation_preparation")
    state=load_observation(plan["observation_ref"],plan["focal_registry_ref"],plan["config"])
    bundle,metadata=full.prepare_from_loaded_observation_v1(state,split="train",
        observation_ref=plan["observation_ref"],focal_registry_ref=plan["focal_registry_ref"],
        native_preparation_ref=plan["config"]["native_preparation_ref"],
        max_nodes=cfg["max_native_nodes"],max_holds=cfg["max_native_holds"],
        max_episode_payload_bytes=cfg["max_episode_payload_bytes"])
    require(bundle.native_holds==plan["trees"] and bundle.original_sites==plan["sites"]
        and bundle.focal_identity_sha256==provider.seal_record["focal_identity_sha256"],
        "Same original plan/SNP/focal support in global preparation")
    del state
    runtime_guard("after_global_TreeLoom_observation_preparation")
    return bundle,metadata


def _global_identity_v1(bundle):
    # CPU observation/topology tensors only; no ancestry target enters this digest.
    tensors=(full._bank_tensors(bundle.bank)+full._episode_tensors(bundle.episodes)
        +bundle.cut.tensors()+(bundle.focal_hold,bundle.focal_sample_column))
    digest=hashlib.sha256();total=0
    for i,tensor in enumerate(tensors):
        require(tensor.device.type=="cpu" and not tensor.requires_grad,
            "CPU observation-only global feature seal")
        value=tensor.detach().contiguous().numpy()
        digest.update(str(i).encode());digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode());digest.update(value.tobytes())
        total+=value.nbytes
    digest.update(bundle.focal_identity_sha256.encode())
    return dict(tensor_sha256=digest.hexdigest(),tensor_bytes=total,
        focal_identity_sha256=bundle.focal_identity_sha256,
        trees=bundle.native_holds,sites=bundle.original_sites,
        all_global_structural_recipes_before_truth=True)


def _signature_v1(cfg,plans,warmstart_binding,target_binding):
    frozen=dict(cfg);frozen.pop("execution_enabled")
    source_paths=[SOURCE,
        str((HERE/"warmstart_models_v1.py").relative_to(Path("/workspace-SR003.nfs2/ARG-GNN"))),
        "src/arggnn/american_c3_gnn_train_v1.py",
        "src/arggnn/american_c3_gnn_stream_train_v1.py",
        "src/arggnn/american_c3_gnn_dataset_v1.py",
        "src/arggnn/american_c3_gnn_targets_v1.py"]
    signature=dict(schema=SCHEMA,config=frozen,parent_plan_refs=plans,
        source_refs=[ref(p) for p in source_paths]+[ENGINE,ENGINE_REVIEW],
        accepted_warmstart=warmstart_binding,actual_TRAIN_execution_binding=target_binding,
        parent_order="random.Random(70002+one_based_epoch).shuffle(original20order)",
        loss="one_unweighted_mean_CE_all10000_native_focals_per_complete_parent",
        final_mechanisms=True,additional_updates_per_arm=600,
        old_backbone_not_retrained_from_scratch=True,whole_parent_no_detached_carry=True)
    return signature,json_sha(signature)


def _snapshot_v1(family,optimizers,*,step,history,scheduled,signature,signature_sha,
        observation_ref,observation_sha,target_ref,target_sha,initialization,device):
    return _cpu_snapshot(dict(schema=FAMILY+"_paired_checkpoint_v1",
        family=FAMILY,condition=CONDITION,arms=list(ARMS),
        model_state={a:m.state_dict() for a,m in family.items()},
        contracts={a:m.checkpoint_contract() for a,m in family.items()},
        optimizer_state={a:o.state_dict() for a,o in optimizers.items()},
        initialization=initialization,
        trainer_state=dict(schema=SCHEMA,protocol=signature,protocol_sha256=signature_sha,
            additional_updates_per_arm=step,inherited_saved_logical_updates=600,
            next_epoch=step//20+1,next_parent_index=step%20,
            next_parent_order=epoch_order_v1(step//20+1) if step<600 else [],
            focal_exposures_per_arm=step*10000,history=history,
            scheduled_checkpoints=scheduled,observation_manifest_ref=observation_ref,
            observation_seal_sha256=observation_sha,target_manifest_ref=target_ref,
            target_join_sha256=target_sha,
            scheduler=dict(name="constant_learning_rate",completed_parent_pairs=step,lr=.001),
            rng_state=_rng_state(device)),
        candidate=dict(kind="paired_complete_parent_boundary",
            completed_epoch=step//20,additional_updates_per_arm=step,
            VAL_selection_authorized=False)))


def _progress_v1(state):
    step=state["additional_updates_per_arm"]
    require(type(step) is int and 0<step<=600
        and state["inherited_saved_logical_updates"]==600
        and state["focal_exposures_per_arm"]==step*10000
        and set(state["history"])==set(ARMS)
        and all(len(state["history"][a])==step for a in ARMS),
        "Only paired complete-parent updates, distinct from inherited600")
    for index in range(step):
        epoch=index//20+1;at=index%20;parent=epoch_order_v1(epoch)[at]
        for arm in ARMS:
            row=state["history"][arm][index]
            require(row["arm"]==arm and row["parent_id"]==parent
                and row["epoch"]==epoch and row["parent_index"]==at
                and row["global_step"]==index+1 and row["records"]==10000
                and row["whole_parent_optimizer_updates"]==1
                and row["block_reset_or_truncated_BPTT"] is False,
                "Same frozen whole-parent order/exposure for both arms")
    require(state["next_epoch"]==step//20+1 and state["next_parent_index"]==step%20
        and state["next_parent_order"]==(epoch_order_v1(step//20+1) if step<600 else [])
        and state["scheduler"]==dict(name="constant_learning_rate",completed_parent_pairs=step,lr=.001),
        "Exact paired cursor and constant optimizer schedule")
    scheduled=state["scheduled_checkpoints"]
    expected=[e for e in CHECKPOINT_EPOCHS if e*20<step]
    require([x["epoch"] for x in scheduled] in
        (expected,expected+[step//20] if step%20==0 and step//20 in CHECKPOINT_EPOCHS else expected)
        and all(x["additional_updates_per_arm"]==20*x["epoch"] for x in scheduled),
        "Preserved six prescribed checkpoint boundaries")
    return step


def _resume_v1(record,family,optimizers,*,signature,signature_sha,
        observation_sha,target_sha,initialization,device):
    value=torch.load(bound(record),map_location="cpu",weights_only=True,mmap=True)
    require(value["schema"]==FAMILY+"_paired_checkpoint_v1"
        and value["family"]==FAMILY and value["condition"]==CONDITION
        and value["arms"]==list(ARMS) and value["initialization"]==initialization,
        "Distinct full matched transfer checkpoint family and identical warmstart decision")
    state=value["trainer_state"];step=_progress_v1(state)
    require(state["protocol"]==signature and state["protocol_sha256"]==signature_sha
        and state["observation_seal_sha256"]==observation_sha
        and state["target_join_sha256"]==target_sha,
        "Exact new protocol, warmstart, source and complete observation/target lineage")
    for arm in ARMS:
        model=family[arm];require(value["contracts"][arm]==model.checkpoint_contract(),
            "Exact final mechanism/NoMem checkpoint contract")
        model.load_state_dict(value["model_state"][arm],strict=True)
        optimizers[arm].load_state_dict(value["optimizer_state"][arm])
        validate_arm_model_v1(arm,model)
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            "Finite complete resumed model state")
        opt=optimizers[arm]
        require(len(opt.param_groups)==1, "One exact resumed AdamW group")
        group=opt.param_groups[0]
        require(group["lr"]==.001 and group["weight_decay"]==.0001
            and tuple(group["betas"])==(.9,.999) and group["eps"]==1e-8
            and group["maximize"] is False and group["amsgrad"] is False
            and len(group["params"])==TENSORS[arm]
            and {id(p) for p in group["params"]}=={id(p) for p in model.parameters()},
            "Resumed original optimizer minimization and every model parameter")
        require(len(opt.state)==TENSORS[arm],
            "Every new optimizer moment present, not original backbone moments")
        for parameter in model.parameters():
            item=opt.state[parameter]
            require(set(item)=={"step","exp_avg","exp_avg_sq"}
                and float(item["step"])==step
                and item["exp_avg"].shape==item["exp_avg_sq"].shape==parameter.shape
                and bool(torch.isfinite(item["exp_avg"]).all())
                and bool(torch.isfinite(item["exp_avg_sq"]).all()),
                "Exact new per-arm step count and finite complete moments")
    _restore_rng(state["rng_state"],device)
    return value,state,step


def run_train_v1(cfg,*,parent_plan_refs,artifact_acceptance_ref,writer,device,
        worker_tmp_directory,resolve_train_truth,runtime_guard,
        train_target_joiner_factory=None,resume_ref=None):
    """Real matched600-additional-update runner. No automatic launch/retry.

    ROOT must qualify actual full-parent fit and admit this new condition.
    The last fully completed PAIR is preserved if either arm fails mid-parent.
    """
    validate_config_v1(cfg)
    require(isinstance(parent_plan_refs,dict) and set(parent_plan_refs)==set(TRAIN)
        and callable(getattr(writer,"save",None)) and callable(runtime_guard)
        and callable(resolve_train_truth), "Actual20plans and admitted writer/resource/target capabilities")
    authoritative_registry_v1()
    actual=torch.device(device)
    require(actual.type in ("cpu","cuda") and (actual.type!="cuda" or actual.index is not None),
        "One explicit ROOT-admitted device; physical GPU policy remains external")
    runtime_guard("before_warmstart_transfer")
    shared,warmstart,old_plans,old_target_binding=load_accepted_backbone_v1(
        artifact_acceptance_ref,runtime_guard=runtime_guard)
    require(parent_plan_refs==old_plans,
        "Same frozen original20TRAIN observations/focals as accepted saved600")
    providers={};safe=None;scheduled=[];history={a:[] for a in ARMS};step=0
    try:
        for parent in TRAIN:
            providers[parent]=SealedObservationProviderV1(parent_plan_refs[parent],
                parent_id=parent,worker_tmp_directory=worker_tmp_directory,
                cache_bytes=cfg["CPU_observation_cache_bytes"])
        seals={p:providers[p].seal(resource_check=runtime_guard) for p in TRAIN}
        global_seals={}
        for parent in TRAIN:
            bundle,meta=_global_bundle_v1(providers[parent],cfg,runtime_guard)
            global_seals[parent]=dict(identity=_global_identity_v1(bundle),
                preparation_metadata=meta)
            del bundle
            providers[parent].release()
        observation=dict(schema=FAMILY+"_all20_global_TRAIN_observation_barrier_v1",
            registry_ref=REGISTRY_REF,parent_ids=list(TRAIN),parents=seals,
            global_structural_preparations=global_seals,all20_ready=True,
            observation_only=True,all_global_recipes_before_TRAIN_truth=True,
            truth_resolver_calls_before_barrier=0)
        observation_sha=json_sha(observation)
        observation_ref=writer.save("TRAIN_observation_barrier.json",observation)
        target_factory=TrainTargetJoinV1 if train_target_joiner_factory is None else train_target_joiner_factory
        target_binding=copy.deepcopy(getattr(target_factory,"execution_binding",None))
        require(target_binding==old_target_binding,
            "Same actual native TRAIN target-path adapter/equations as accepted backbone")
        if target_binding is not None:
            require(target_binding.get("target_equations_unchanged") is True
                and target_binding.get("registry_ref")==REGISTRY_REF
                and all(ref(r["path"])==r for r in target_binding["source_refs"]),
                "Authenticated existing TRAIN-only operational target adapter")
        joiner=target_factory(resolve_train_truth,
            max_truth_uncompressed_bytes=cfg["max_truth_uncompressed_bytes"])
        targets={}
        for parent in TRAIN:
            runtime_guard("TreeLoom_external_TRAIN_target_join")
            targets[parent]=joiner.join(parent,split="train",
                plan=providers[parent].plan,observation_seal=seals[parent])
        target_manifest=dict(schema=FAMILY+"_all20_native_focal_target_joins_v1",
            registry_ref=REGISTRY_REF,observation_seal_sha256=observation_sha,
            parents={p:targets[p].receipt for p in TRAIN},all20_targets_ready=True,
            labels_in_observation_bank=False,purity_filter=False)
        target_sha=json_sha(target_manifest)
        target_ref=writer.save("TRAIN_external_target_joins.json",target_manifest)
        signature,signature_sha=_signature_v1(cfg,parent_plan_refs,warmstart,target_binding)
        require(torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled(),
            "Original full FP32 model/optimizer condition")
        runtime_guard("before_first_CUDA_seed_or_model_transfer")
        _seed_fresh(actual)
        family,optimizers,initialization=make_matched_family_v1(shared,warmstart,device=actual)
        del shared
        writer.save("matched_initialization.json",initialization)
        if resume_ref is not None:
            runtime_guard("before_explicit_new_family_resume")
            value,state,step=_resume_v1(resume_ref,family,optimizers,
                signature=signature,signature_sha=signature_sha,
                observation_sha=observation_sha,target_sha=target_sha,
                initialization=initialization,device=actual)
            history=copy.deepcopy(state["history"]);scheduled=copy.deepcopy(state["scheduled_checkpoints"])
            safe=_cpu_snapshot(value);del value,state
            if step%20==0 and step//20 in CHECKPOINT_EPOCHS and len(scheduled)<sum(e*20<=step for e in CHECKPOINT_EPOCHS):
                recovered=copy.deepcopy(safe)
                recovered["candidate"]=dict(kind="scheduled_epoch_pair",
                    completed_epoch=step//20,additional_updates_per_arm=step,
                    VAL_selection_authorized=False)
                recovered_ref=writer.save("checkpoint_pair_epoch_%03d.pt"%(step//20),recovered,kind="torch")
                scheduled.append(dict(epoch=step//20,additional_updates_per_arm=step,
                    checkpoint_ref=recovered_ref))
        for index in range(step,600):
            epoch=index//20+1;at=index%20;parent=epoch_order_v1(epoch)[at]
            pair_started=time.monotonic()
            cpu_bundle,_=_global_bundle_v1(providers[parent],cfg,runtime_guard)
            require(_global_identity_v1(cpu_bundle)==global_seals[parent]["identity"],
                "Every global observation/structural tensor unchanged after truth access")
            bundle=cpu_bundle.to(actual);del cpu_bundle
            preparation_wall=time.monotonic()-pair_started
            for arm in ARMS:
                result=train_complete_parent_step_v1(arm,family[arm],bundle,
                    targets[parent],optimizers[arm],resource_check=runtime_guard,
                    first_update=(index==0))
                result.update(epoch=epoch,parent_index=at,global_step=index+1)
                history[arm].append(result)
            paired_wall=time.monotonic()-pair_started
            for arm in ARMS:
                history[arm][-1].update(
                    paired_parent_wall_seconds_including_preparation=paired_wall,
                    global_observation_preparation_wall_seconds=preparation_wall)
            del bundle;providers[parent].release()
            step=index+1
            safe=_snapshot_v1(family,optimizers,step=step,history=history,
                scheduled=scheduled,signature=signature,signature_sha=signature_sha,
                observation_ref=observation_ref,observation_sha=observation_sha,
                target_ref=target_ref,target_sha=target_sha,
                initialization=initialization,device=actual)
            if step%20==0 and epoch in CHECKPOINT_EPOCHS:
                candidate=copy.deepcopy(safe)
                candidate["candidate"]=dict(kind="scheduled_epoch_pair",
                    completed_epoch=epoch,additional_updates_per_arm=step,
                    VAL_selection_authorized=False)
                checkpoint_ref=writer.save("checkpoint_pair_epoch_%03d.pt"%epoch,candidate,kind="torch")
                scheduled.append(dict(epoch=epoch,additional_updates_per_arm=step,
                    checkpoint_ref=checkpoint_ref))
                safe=_snapshot_v1(family,optimizers,step=step,history=history,
                    scheduled=scheduled,signature=signature,signature_sha=signature_sha,
                    observation_ref=observation_ref,observation_sha=observation_sha,
                    target_ref=target_ref,target_sha=target_sha,
                    initialization=initialization,device=actual)
        require(step==600 and all(len(history[a])==600 for a in ARMS)
            and [r["epoch"] for r in scheduled]==list(CHECKPOINT_EPOCHS),
            "Both arms complete all600new whole-parent updates and all6paired checkpoints")
        return writer.save("completed_training.json",dict(schema=SCHEMA,
            status="completed_TreeLoom_saved600_warmstart_matched_training",
            family=FAMILY,condition=CONDITION,arms=list(ARMS),parameters_per_arm=PARAMETERS,
            protocol_sha256=signature_sha,initialization=initialization,
            inherited_saved_logical_updates=600,additional_updates_per_arm=600,
            new_saved_logical_optimizer_updates_total=1200,
            exact_physical_update_total_not_inferred=True,
            focal_exposures_per_arm=6000000,
            epochs=30,trained_parent_ids=list(TRAIN),train_sequence_lengths_bp=[2000000]*20,
            observation_manifest_ref=observation_ref,target_manifest_ref=target_ref,
            observation_seal_sha256=observation_sha,target_join_sha256=target_sha,
            checkpoint_candidates={a:[dict(r,arm=a) for r in scheduled] for a in ARMS},
            history=history,resume_ref=resume_ref,selected_heldout_checkpoint=None,
            VAL_used_for_optimization=False,TEST_deferred=True,
            new_prediction_seals_required_per_arm=30,
            all60_matched_prediction_seals_required_before_any_VAL_truth=True,
            total_new_prediction_seals_required=60,
            either_arm_validation_truth_or_checkpoint_choice_before60_seals=False,
            no_current_validation_scoring_capability=True,
            first_step_timing_inside_full_budget=True,
            first_real_update_checks={a:history[a][0]["first_real_update_checks"] for a in ARMS},
            first_scheduled_update_wall_seconds={
                a:history[a][0]["complete_parent_step_wall_seconds_including_checks"] for a in ARMS},
            standalone_token_update_probe=False,
            existing_C3_all30_barrier_unchanged=True,
            exact_European_or_fourstage_reproduction_claimed=False,
            prior_failed_producer_success_claimed=False,ancestry_accuracy=None))
    except BaseException as error:
        try:
            boundary_ref=None
            if safe is not None:
                boundary_ref=writer.save("failure_last_complete_pair.pt",safe,kind="torch")
            failure_ref=writer.save("failed_training.json",dict(
                status="failed_TreeLoom_warmstart_training_preserved",
                family=FAMILY,condition=CONDITION,
                last_complete_paired_updates_per_arm=step,
                last_complete_pair_checkpoint_ref=boundary_ref,
                last_safe_checkpoint_updates_per_arm=(
                    safe["trainer_state"]["additional_updates_per_arm"]
                    if safe is not None else None),
                attempted_arm_counts={a:len(history[a]) for a in ARMS},
                partial_pair_or_update_not_checkpointed=True,
                exception_type=type(error).__name__,message=str(error)[:2000]))
            if hasattr(error,"add_note"):error.add_note("Preserved failure receipt: "+str(failure_ref))
        except BaseException as sealing:
            if hasattr(error,"add_note"):error.add_note("Failure sealing failed: "+type(sealing).__name__)
        raise
    finally:
        for provider in providers.values():provider.release()

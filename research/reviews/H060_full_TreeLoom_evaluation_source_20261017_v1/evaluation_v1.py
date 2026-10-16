"""Source-only H060 full TreeLoom evaluation. No launcher, TEST or automatic scoring.

ROOT supplies current admitted writer/resource capabilities and explicit phase authority.
One complete original30Mb parent is prepared once and reused for12 full inferences.
All60 fresh matched seals must validate before any VAL resolver or selection.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,math,stat,statistics,zipfile

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent
SOURCE=str(Path(__file__).resolve().relative_to(ROOT))
PROTOCOL=dict(path="research/reviews/H060_American_TreeLoom_matched_registration_20261017_v1/protocol.root_frozen_v1.json",
 sha256="0a68acb3e3e6054f8edf1de591252b1ad0cc9904cf2a677db54d1d809e488f6d")
TRAIN_SOURCE_DIR="research/reviews/American_TreeLoom_warmstart_training_source_20261017_v2"
RUNNER_REF=dict(path=TRAIN_SOURCE_DIR+"/warmstart_train_v2.py",sha256="c26f3cb2c949648ccea97570007e08bc49fff12608906be9d3084d5919d1ea07")
MODEL_REF=dict(path=TRAIN_SOURCE_DIR+"/warmstart_models_v1.py",sha256="4036ff51dcfe94a130c9023894c748e86f9a4f37688eaceb1894439ed0c79571")
TRAIN_PROTOCOL=dict(path=TRAIN_SOURCE_DIR+"/protocol.inactive.v2.json",sha256="e25d626ae3d0f49767416e4697fbae646d622396ed978fbdeacb8a0cfbbfc595")
TARGET_HOOK=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/execution_hooks_prefetch_v1.py",
 sha256="b7418267d329b0d83df45c5f5d053ffd034db0e88c794d5a59c7b59ca52900d1")
ARMS=("stable_leaf_memory","no_added_memory")
EPOCHS=(5,10,15,20,25,30)
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))
CLASSES=("EUR","EAS","AFR")
IDENTITY=("record_ids","sample_nodes","sample_indices","positions_bp0","parent_id","class_order")
DEFAULTS=dict(schema="H060_full_TreeLoom_evaluation_config_v1",execution_enabled=False,
 max_native_nodes=None,max_native_holds=None,max_episode_payload_bytes=None,
 max_cached_parent_bytes=None,max_checkpoint_file_bytes=None,
 max_prediction_uncompressed_bytes=None,max_truth_uncompressed_bytes=None)

def require(ok,message):
    if not ok:raise ValueError(message)

def _metadata():
    from arggnn.american_c3_common_v1 import read,bound,ref
    return read,bound,ref

def _scientific():
    from research.reviews.American_TreeLoom_warmstart_training_source_20261017_v2 import warmstart_train_v2 as train
    from research.reviews.American_TreeLoom_warmstart_training_source_20261017_v2 import warmstart_models_v1 as models
    return train,models

def _config(cfg):
    require(isinstance(cfg,dict) and set(cfg)==set(DEFAULTS)
        and cfg["schema"]==DEFAULTS["schema"] and cfg["execution_enabled"] is True
        and all(type(cfg[k]) is int and cfg[k]>0 for k in DEFAULTS if k.startswith("max_")),
        "Explicit ROOT full-parent/cache/checkpoint/prediction/truth failstop bounds; no fit or reduction")

def _manifest_ref():
    return _metadata()[2](str((HERE/"source_manifest.v1.json").relative_to(ROOT)))

def _source_snapshot():
    read,bound,ref=_metadata();rows=json.loads((HERE/"source_pins.v1.json").read_text())["refs"]
    snapshot=[]
    for record in rows:
        path=ROOT/record["path"];before=path.lstat()
        require(stat.S_ISREG(before.st_mode) and path.resolve()==path,"Canonical scientific source")
        require(ref(record["path"])==record,"Exact reviewed evaluation dependency")
        after=path.lstat()
        stamp=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        require(stamp(before)==stamp(after),"Source changed during authentication")
        snapshot.append((record["path"],)+stamp(after))
    return tuple(snapshot)

def _source_check(snapshot):
    for row in snapshot:
        info=(ROOT/row[0]).lstat()
        require(stat.S_ISREG(info.st_mode) and
            (row[0],info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)==row,
            "Authenticated evaluation source identity changed; no reload/fallback")

def _authority(record,*,phase,training_context_ref,evaluation_context_ref,cfg,parent_id=None,matrix_ref=None):
    read,_,_=_metadata();value=read(record)
    now=datetime.now(timezone.utc)
    created=datetime.fromisoformat(value["created_at_utc"]);expires=datetime.fromisoformat(value["expires_at_utc"])
    require(created.tzinfo and expires.tzinfo and created<=now<expires
        and value.get("schema")=="H060_evaluation_ROOT_authority_v1"
        and value.get("root_authorizer")=="root_agent" and value.get("active") is True
        and value.get("phase")==phase and value.get("source_manifest_ref")==_manifest_ref()
        and value.get("protocol_ref")==PROTOCOL and value.get("training_context_ref")==training_context_ref
        and value.get("evaluation_context_ref")==evaluation_context_ref and value.get("config")==cfg
        and value.get("parent_id")==parent_id and value.get("validation_matrix_ref")==matrix_ref
        and value.get("TEST_authorized") is False
        and value.get("VAL_truth_authorized") is (phase=="score_and_select"),
        "Separate current ROOT phase authority bound to exact source/protocol/complete workload")
    if phase=="score_and_select":
        require(value.get("all60_seals_frozen_before_authorization") is True,"All60pretruth seals before score authority")
    return value

def _completed_training(record):
    """Metadata only: current CPU producer must genuinely finish and be waited/drained."""
    read,_,_= _metadata();ctx=read(record)
    required_refs=("training_input_ref","completed_envelope_ref","actual_wait_ref","outer_terminal_ref",
        "producer_source_ref","producer_manifest_ref","producer_entry_ref","producer_review_ref")
    require(ctx.get("schema")=="H060_completed_matched_training_context_v1" and ctx.get("completed") is True
        and ctx.get("root_authorizer")=="root_agent" and ctx.get("protocol_ref")==PROTOCOL
        and all(isinstance(ctx.get(k),dict) for k in required_refs),
        "Genuine completed H060 context, unavailable while training/admission is pending")
    train,models=_scientific()
    inp=read(ctx["training_input_ref"]);req=read(inp["root_stage_request_ref"])
    envelope=read(ctx["completed_envelope_ref"]);wait=read(ctx["actual_wait_ref"]);outer=read(ctx["outer_terminal_ref"])
    producer=ctx["producer_source_ref"];producer_manifest=ctx["producer_manifest_ref"]
    review=read(ctx["producer_review_ref"]);source_map=read(producer_manifest)["source_sha256"]
    require(review.get("independent") is True and review.get("verdict") in ("SOURCE_PASS","PASS")
        and review.get("blockers")==[] and review.get("source_ref")==producer
        and review.get("source_manifest_ref")==producer_manifest and review.get("runner_source_ref")==RUNNER_REF
        and review.get("protocol_ref")==TRAIN_PROTOCOL
        and all(source_map.get(r["path"])==r["sha256"] for r in (RUNNER_REF,MODEL_REF,train.ENGINE)),
        "ROOT-certified actual successful producer and independent wrapper review bind exact numerical sources")
    for path,sha in source_map.items():
        require(_metadata()[2](path)==dict(path=path,sha256=sha),"Actual producer source closure remains exact")
    require(inp.get("root_driver_ref")==producer and inp.get("main_source_ref")==ctx["producer_entry_ref"]
        and inp.get("source_sha256")==source_map
        and req.get("source_manifest_ref")==producer_manifest and req.get("source_review_ref")==ctx["producer_review_ref"]
        and req.get("registration_ref")==PROTOCOL
        and req.get("phase")=="train_matched" and req.get("GPU_authorized") is False
        and req.get("physical_gpu") is None and req.get("CUDA_authorized") is False,
        "Exact reviewed actual matched CPU source, frozen H060 and no GPU condition")
    require(envelope.get("status")=="completed_TreeLoom_saved600_matched_CPU_training"
        and envelope.get("input_ref")==ctx["training_input_ref"] and envelope.get("root_driver_ref")==producer
        and wait.get("input_ref")==ctx["training_input_ref"] and wait.get("completed_ref")==ctx["completed_envelope_ref"]
        and wait.get("actual_direct_worker_wait") is True and wait.get("returncode")==0
        and wait.get("all_owned_processes_and_groups_drained") is True and wait.get("operational_failure") is None
        and outer.get("root_authorizer")=="root_agent" and outer.get("actual_terminal_consumed") is True
        and outer.get("actual_exit_code")==0 and outer.get("direct_wait_ref")==ctx["actual_wait_ref"]
        and outer.get("completed_ref")==ctx["completed_envelope_ref"],
        "Genuine successful direct/outer waits and complete owned drain, never a pending/failed producer")
    result=read(envelope["result_ref"]);tc=read(req["training_context_ref"]);cfg=req["scientific_config"]
    original_ref=dict(path="research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/root_resume_actual_v1/worker_request.json",
        sha256="3c3d7f6c113167a8084ba91726bd460e3f35030c2f97d1ed6b5805fd5a55063f")
    original=read(original_ref)["bindings"]
    require(tc.get("original_training_request_ref")==original_ref
        and tc.get("accepted_artifact_ref")==train.ACCEPTANCE
        and tc["parent_plan_refs"]==original["parent_plan_refs"]
        and tc["actual_train_bindings"]==original["actual_train_bindings"],
        "Exact original20TRAIN native plans and operational targets")
    require(result.get("status")=="completed_TreeLoom_saved600_warmstart_matched_training"
        and result.get("family")==models.FAMILY and result.get("condition")==models.CONDITION
        and result.get("arms")==list(ARMS) and result.get("additional_updates_per_arm")==600
        and result.get("new_saved_logical_optimizer_updates_total")==1200
        and result.get("inherited_saved_logical_updates")==600 and result.get("epochs")==30
        and result.get("trained_parent_ids")==list(TRAIN) and result.get("train_sequence_lengths_bp")==[2000000]*20
        and result.get("focal_exposures_per_arm")==6000000 and result.get("VAL_used_for_optimization") is False
        and result.get("selected_heldout_checkpoint") is None and result.get("TEST_deferred") is True
        and result.get("total_new_prediction_seals_required")==60
        and result.get("all60_matched_prediction_seals_required_before_any_VAL_truth") is True,
        "Completed full600additional EACHarm/1200total, all60new seals, no preselected checkpoint")
    train.validate_config_v1(cfg)
    init=result["initialization"];warm=init["warmstart_binding"]
    require(warm.get("acceptance_ref")==train.ACCEPTANCE and warm.get("epoch")==30
        and warm.get("saved_logical_updates")==600 and warm.get("model_parameters")==29571
        and warm.get("original_producer_success_claimed") is False
        and init.get("old_optimizer_and_rng_restored") is False
        and init.get("parameters_per_arm")==models.PARAMETERS and init.get("tensors_per_arm")==models.TENSORS,
        "Accepted saved600 warmstart remains distinct from new matched training")
    target_binding=dict(schema="C3_actual_TRAIN_simulation_binding_operational_v1",registry_ref=train.REGISTRY_REF,
        source_refs=[TARGET_HOOK],parents=copy.deepcopy(tc["actual_train_bindings"]),target_equations_unchanged=True,
        change="validated_actual_wave_output_paths_only_no_truth_copy_or_node_join_change")
    signature,sha=train._signature_v1(cfg,tc["parent_plan_refs"],warm,target_binding)
    require(result["protocol_sha256"]==sha,"Exact original TRAIN order/config/source/target/warmstart signature")
    require(set(result["history"])==set(ARMS) and all(len(result["history"][a])==600 for a in ARMS),
        "Complete histories for both arms")
    for index in range(600):
        parent=train.epoch_order_v1(index//20+1)[index%20]
        for arm in ARMS:
            row=result["history"][arm][index]
            require(row["arm"]==arm and row["parent_id"]==parent and row["global_step"]==index+1
                and row["epoch"]==index//20+1 and row["parent_index"]==index%20
                and row["records"]==10000 and row["whole_parent_optimizer_updates"]==1
                and row["block_reset_or_truncated_BPTT"] is False,"Same complete original600updates in each arm")
    candidates={}
    require(set(result["checkpoint_candidates"])==set(ARMS),"Both arms' prescribed candidates")
    for arm in ARMS:
        rows=result["checkpoint_candidates"][arm]
        require(len(rows)==6 and [r["epoch"] for r in rows]==list(EPOCHS),"Exactly6prescribed candidates per arm")
        for row in rows:
            require(row["arm"]==arm and row["additional_updates_per_arm"]==20*row["epoch"],"Exact scheduled candidate boundary")
            if row["epoch"] in candidates:require(candidates[row["epoch"]]==row["checkpoint_ref"],"One paired file contains both arms")
            candidates[row["epoch"]]=row["checkpoint_ref"]
    require(len({r["path"] for r in candidates.values()})==6,"Six distinct paired checkpoint files")
    return dict(context=ctx,input=inp,request=req,result=result,result_ref=envelope["result_ref"],
        signature=signature,signature_sha256=sha,candidates=candidates)

def _evaluation_context(record,*,parent_id=None):
    read,_,_= _metadata();ctx=read(record)
    require(ctx.get("schema")=="H060_original_VAL_context_v1" and ctx.get("root_authorizer")=="root_agent"
        and ctx.get("protocol_ref")==PROTOCOL and set(ctx["original_C3_request_refs"])==set(VAL)
        and set(ctx["simulation_completion_refs"])==set(VAL)
        and all(isinstance(r,dict) for r in ctx["original_C3_request_refs"].values())
        and all(isinstance(r,dict) for r in ctx["simulation_completion_refs"].values()),
        "All5genuinely completed originalVAL plans/context required; VAL004 unbound is not readiness")
    known=json.loads((HERE/"original_VAL_request_pins.v1.json").read_text())["known_requests"]
    parents=VAL if parent_id is None else (parent_id,)
    require(all(p in VAL for p in parents),"Only original5VAL")
    plans={}
    for parent in parents:
        r=ctx["original_C3_request_refs"][parent]
        require(isinstance(r,dict) and (known[parent] is None or r==known[parent]),
            "Exact original canonical C3 request; VAL004 must genuinely complete before binding")
        request=read(r);ev=request["plan_evidence"]
        require(request.get("schema")=="C3_full30Mb_evaluation_ROOT_request_v1"
            and request.get("root_authorizer")=="root_agent" and request.get("phase")=="predict_val_"+parent[-3:]
            and ev.get("parent_id")==parent and isinstance(ev.get("plan_ref"),dict)
            and isinstance(ev.get("actual_wait_ref"),dict) and isinstance(ev.get("close_receipt_ref"),dict),
            "Original completed-plan evidence, no invented/missing plan")
        complete=read(ev["completion_ref"])
        for _ in range(2):
            if "block_index_ref" in complete:break
            require("result_ref" in complete,"Real plan completion must reference its actual plan result")
            complete=read(complete["result_ref"])
        require(complete.get("status")=="completed_C3_observation_only_block_plan"
            and complete.get("parent_id")==parent and complete.get("block_index_ref")==ev["plan_ref"],
            "Exact genuinely completed original block_index plan")
        # Known request refs already freeze original lifecycle evidence. Future004 must bind its actual successful evidence.
        if known[parent] is None:
            wait=read(ev["actual_wait_ref"]);close=read(ev["close_receipt_ref"]);outer=read(ev["outer_terminal_ref"])
            require(wait.get("returncode")==0 and wait.get("actual_direct_worker_wait") is True
                and wait.get("operational_failure") is None and wait.get("all_owned_processes_and_groups_drained") is True
                and wait.get("completed_ref")==ev["completion_ref"]
                and outer.get("root_authorizer")=="root_agent" and outer.get("actual_terminal_consumed") is True
                and outer.get("actual_exit_code")==0 and outer.get("direct_wait_ref")==ev["actual_wait_ref"]
                and outer.get("completed_ref")==ev["completion_ref"]
                and close.get("kind")=="concurrent_completed_scope_close_receipt_v1"
                and close["value"].get("future_writes_and_relaunch_revoked") is True
                and close["value"]["request"].get("completed_ref")==ev["completion_ref"]
                and close["value"]["request"].get("direct_wait_ref")==ev["actual_wait_ref"],
                "Future originalVAL004 genuine completion/wait/close/outer evidence")
        plans[parent]=ev["plan_ref"]
    return ctx,plans

def _plan(plan_ref,parent):
    read,_,_=_metadata()
    from arggnn.american_c3_gnn_shared30mb_v1 import validate_shared_plan_v1,validate_shared_metadata_v1
    plan=read(plan_ref);validate_shared_plan_v1(plan,parent_id=parent);validate_shared_metadata_v1(plan["config"],parent_id=parent)
    return plan

def _checkpoint(record,epoch,training,models_by_arm,cfg):
    import torch
    _,bound,_=_metadata();train,models=_scientific()
    path=bound(record)
    require(path.stat().st_size<=cfg["max_checkpoint_file_bytes"],"Complete paired checkpoint file bound")
    value=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    require(value.get("schema")==models.FAMILY+"_paired_checkpoint_v1" and value.get("family")==models.FAMILY
        and value.get("condition")==models.CONDITION and value.get("arms")==list(ARMS)
        and value.get("initialization")==training["result"]["initialization"]
        and all(set(value[k])==set(ARMS) for k in ("model_state","contracts","optimizer_state")),
        "Exact matched paired checkpoint family")
    state=value["trainer_state"];step=train._progress_v1(state)
    require(step==epoch*20 and state["protocol"]==training["signature"] and state["protocol_sha256"]==training["signature_sha256"]
        and state["observation_seal_sha256"]==training["result"]["observation_seal_sha256"]
        and state["target_join_sha256"]==training["result"]["target_join_sha256"]
        and state["observation_manifest_ref"]==training["result"]["observation_manifest_ref"]
        and state["target_manifest_ref"]==training["result"]["target_manifest_ref"]
        and all(state["history"][a]==training["result"]["history"][a][:step] for a in ARMS)
        and value["candidate"]==dict(kind="scheduled_epoch_pair",completed_epoch=epoch,
            additional_updates_per_arm=epoch*20,VAL_selection_authorized=False),"Exact prescribed complete pair and TRAIN seals")
    for arm,model in models_by_arm.items():
        models.validate_arm_model_v1(arm,model)
        require(value["contracts"][arm]==model.checkpoint_contract(),"Exact full mechanism/matched NoMem contract")
        weights=value["model_state"][arm];expected=model.state_dict()
        require(set(weights)==set(expected) and all(isinstance(v,torch.Tensor) and v.device.type=="cpu"
            and v.dtype==torch.float32 and v.shape==expected[k].shape and bool(torch.isfinite(v).all())
            for k,v in weights.items()),"Every finite FP32 checkpoint tensor, no dropped/renamed/sliced weights")
        opt=value["optimizer_state"][arm];groups=opt["param_groups"];parameters=list(model.parameters())
        require(len(groups)==1 and len(groups[0]["params"])==len(set(groups[0]["params"]))==len(parameters)
            and set(opt["state"])==set(groups[0]["params"]),"Complete original optimizer metadata, not partial weights")
        group=groups[0]
        require(group["lr"]==.001 and group["weight_decay"]==.0001 and tuple(group["betas"])==(.9,.999)
            and group["eps"]==1e-8 and group["maximize"] is False and group["amsgrad"] is False,
            "Exact minimizing AdamW metadata")
        for key,parameter in zip(group["params"],parameters):
            item=opt["state"][key]
            require(set(item)=={"step","exp_avg","exp_avg_sq"} and float(item["step"])==step
                and all(item[k].shape==parameter.shape and item[k].dtype==torch.float32
                    and item[k].device.type=="cpu" and bool(torch.isfinite(item[k]).all())
                    for k in ("exp_avg","exp_avg_sq")),"Complete finite per-parameter matched optimizer state")
        model.load_state_dict(weights,strict=True)
    return value

def _load_npz(record,cap):
    import numpy as np
    _,bound,_=_metadata();path=bound(record)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        require(len(names)==len(set(names)) and sum(i.file_size for i in z.infolist())<=cap
            and all(n.endswith(".npy") and "/" not in n for n in names),"Bounded unique flat NPZ members")
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def _predictions(record,registry_ref,cfg):
    import numpy as np
    from arggnn.american_c3_evaluate_v1 import registry,verify_prediction
    from arggnn.american_c3_gnn_targets_v1 import identity_sha
    records=registry(registry_ref);pred=_load_npz(record,cfg["max_prediction_uncompressed_bytes"])
    calls=verify_prediction(pred,records)
    require((pred["probabilities"]<=1).all(),"Raw class probabilities within unit interval; no correction")
    require(len(calls)==10000 and np.isin(records["sample_indices"],np.arange(600,1600)).all()
        and ((records["positions_bp0"]>=0)&(records["positions_bp0"]<30000000)).all(),
        "All original10000native query focals, no support/mask/coordinate change")
    return pred,records,calls,identity_sha(records)

def predict_parent_v1(cfg,*,parent_id,training_context_ref,evaluation_context_ref,
        authority_ref,writer,device,runtime_guard):
    """No truth path or resolver argument. Run all12 complete candidates for one VAL parent."""
    _config(cfg);require(parent_id in VAL and callable(runtime_guard) and callable(getattr(writer,"save",None)),"Original VAL and ROOT capabilities")
    _authority(authority_ref,phase="predict",training_context_ref=training_context_ref,
        evaluation_context_ref=evaluation_context_ref,cfg=cfg,parent_id=parent_id)
    snapshot=_source_snapshot();runtime_guard("before_H060_prediction_scientific_import")
    import numpy as np,torch
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from arggnn.american_c3_evaluate_v1 import verify_prediction
    train,models=_scientific();training=_completed_training(training_context_ref)
    _,plans=_evaluation_context(evaluation_context_ref,parent_id=parent_id);plan=_plan(plans[parent_id],parent_id)
    actual=torch.device(device)
    require(actual.type in ("cpu","cuda") and (actual.type!="cuda" or actual.index is not None)
        and torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled(),"One ROOT-admitted FP32 device")
    runtime_guard("before_H060_complete_parent_preparation")
    state=load_observation(plan["observation_ref"],plan["focal_registry_ref"],plan["config"])
    records={k:np.array(state["focals"][k],copy=True) for k in IDENTITY}
    bundle,preparation=models.full.prepare_from_loaded_observation_v1(state,split="val_prediction",
        observation_ref=plan["observation_ref"],focal_registry_ref=plan["focal_registry_ref"],
        native_preparation_ref=plan["config"]["native_preparation_ref"],max_nodes=cfg["max_native_nodes"],
        max_holds=cfg["max_native_holds"],max_episode_payload_bytes=cfg["max_episode_payload_bytes"])
    del state
    require(bundle.native_holds==plan["trees"] and bundle.original_sites==plan["sites"],"Every original plan tree/site retained")
    identity=train._global_identity_v1(bundle);cache={}
    charged=train._cache_admit_v2(cache,parent_id,bundle,current_bytes=0,
        maximum_bytes=cfg["max_cached_parent_bytes"],runtime_guard=runtime_guard)
    device_bundle=None;family={};seals=[]
    try:
        with torch.random.fork_rng(devices=[]),torch.device("cpu"):
            family={ARMS[0]:models.WarmstartTreeLoomFullV1(),ARMS[1]:models.WarmstartTreeLoomNoMemV1()}
        runtime_guard("before_H060_single_parent_device_transfer")
        device_bundle=bundle.to(actual)
        for epoch in EPOCHS:
            _source_check(snapshot);train._cache_get_v2(cache,parent_id)
            runtime_guard("before_H060_complete_paired_checkpoint_load")
            value=_checkpoint(training["candidates"][epoch],epoch,training,family,cfg)
            del value
            for arm in ARMS:
                _source_check(snapshot);train._cache_get_v2(cache,parent_id)
                runtime_guard("before_H060_full_parent_candidate_inference")
                model=family[arm].to(actual);model.eval();models.validate_arm_model_v1(arm,model)
                with torch.no_grad(),torch.autocast(device_type=actual.type,enabled=False):
                    probability=model(device_bundle).softmax(-1).cpu()
                require(probability.dtype==torch.float32 and probability.shape==(10000,3),"Full native-focal FP32 predictions")
                predictions=dict(records,probabilities=probability.numpy())
                verify_prediction(predictions,records)
                runtime_guard("before_H060_new_prediction_seal")
                stem="H060_%s_%s_pass%03d"%(arm,parent_id,epoch)
                prediction_ref=writer.save(stem+"_predictions.npz",predictions,kind="npz")
                seals.append(writer.save(stem+"_prediction_seal.json",dict(
                    status="sealed_H060_full_TreeLoom_VAL_prediction_v1",hypothesis_id="H060",
                    arm=arm,parent_id=parent_id,epoch=epoch,additional_updates_per_arm=20*epoch,
                    checkpoint_ref=training["candidates"][epoch],training_context_ref=training_context_ref,
                    training_result_ref=training["result_ref"],evaluation_context_ref=evaluation_context_ref,
                    frozen_protocol_ref=PROTOCOL,evaluation_source_manifest_ref=_manifest_ref(),
                    prediction_authority_ref=authority_ref,
                    training_protocol_sha256=training["signature_sha256"],prediction_ref=prediction_ref,
                    plan_ref=plans[parent_id],observation_ref=plan["observation_ref"],
                    focal_registry_ref=plan["focal_registry_ref"],focal_identity_sha256=bundle.focal_identity_sha256,
                    records=10000,class_order=list(CLASSES),length_bp=30000000,
                    complete_native_parent=True,full_parent_bidirectional_GRU=True,
                    private_truth_opened=False,probabilities_modified=False,probability_floor=False,
                    purity_filter=False,TEST_deferred=True)))
                model.to("cpu");del predictions,probability
                train._cache_get_v2(cache,parent_id);_source_check(snapshot)
        return writer.save("H060_"+parent_id+"_all12_prediction_seals.json",dict(
            status="sealed_H060_one_VAL_all12_candidates_v1",parent_id=parent_id,
            training_context_ref=training_context_ref,evaluation_context_ref=evaluation_context_ref,
            prediction_seal_refs=seals,complete_CPU_parent_preparations=1,CPU_cache_accounted_bytes=charged,
            max_cached_parent_bytes=cfg["max_cached_parent_bytes"],preparation_metadata=preparation,
            initial_tensor_identity=identity,all12_same_immutable_parent=True,
            full_preparation_or_hash_repeated=False,private_truth_opened=False,
            all60_barrier_still_required=True,TEST_deferred=True))
    finally:
        device_bundle=None;family.clear();cache.clear();del bundle

def _cell_keys(rows):
    require(isinstance(rows,list) and len(rows)==60,"All60fresh seals required before either arm VAL truth/selection")
    cells={}
    expected={(a,e,p) for a in ARMS for e in EPOCHS for p in VAL}
    for row in rows:
        key=(row["arm"],row["epoch"],row["parent_id"])
        require(key in expected and key not in cells,"Unique complete2x6x5 matched cells")
        cells[key]=row
    require(set(cells)==expected,"No missing/extra matched cells")
    return cells

def _validate_seals(rows,*,training_context_ref,evaluation_context_ref,cfg,runtime_guard):
    require(isinstance(rows,list) and len(rows)==60,"All60fresh seal refs required")
    read,_,_=_metadata();training=_completed_training(training_context_ref)
    _,plan_refs=_evaluation_context(evaluation_context_ref);plans={p:_plan(plan_refs[p],p) for p in VAL}
    seen_prediction_paths=set();identities={};items=[]
    for record in rows:
        runtime_guard("H060_all60_pretruth_payload_authentication")
        seal=read(record);arm,epoch,parent=seal["arm"],seal["epoch"],seal["parent_id"]
        require(arm in ARMS and epoch in EPOCHS and parent in VAL,"Matched VAL cell only")
        require(seal.get("status")=="sealed_H060_full_TreeLoom_VAL_prediction_v1"
            and seal.get("hypothesis_id")=="H060" and seal.get("training_context_ref")==training_context_ref
            and seal.get("training_result_ref")==training["result_ref"] and seal.get("evaluation_context_ref")==evaluation_context_ref
            and seal.get("frozen_protocol_ref")==PROTOCOL and seal.get("evaluation_source_manifest_ref")==_manifest_ref()
            and seal.get("training_protocol_sha256")==training["signature_sha256"]
            and seal.get("checkpoint_ref")==training["candidates"][epoch]
            and seal.get("additional_updates_per_arm")==20*epoch
            and seal.get("plan_ref")==plan_refs[parent]
            and seal.get("observation_ref")==plans[parent]["observation_ref"]
            and seal.get("focal_registry_ref")==plans[parent]["focal_registry_ref"]
            and seal.get("records")==10000 and seal.get("length_bp")==30000000 and seal.get("class_order")==list(CLASSES)
            and seal.get("complete_native_parent") is True and seal.get("full_parent_bidirectional_GRU") is True
            and seal.get("private_truth_opened") is False and seal.get("probabilities_modified") is False
            and seal.get("probability_floor") is False and seal.get("purity_filter") is False,
            "Fresh full matched H060 seal; simplerC3all30 is not accepted or changed")
        authority=read(seal["prediction_authority_ref"])
        require(authority.get("schema")=="H060_evaluation_ROOT_authority_v1"
            and authority.get("root_authorizer")=="root_agent" and authority.get("phase")=="predict"
            and authority.get("source_manifest_ref")==_manifest_ref() and authority.get("protocol_ref")==PROTOCOL
            and authority.get("training_context_ref")==training_context_ref
            and authority.get("evaluation_context_ref")==evaluation_context_ref
            and authority.get("parent_id")==parent and authority.get("config")==cfg
            and authority.get("VAL_truth_authorized") is False and authority.get("TEST_authorized") is False,
            "Original separate prediction authority remains authenticated; no truth capability in prediction")
        require(seal["prediction_ref"]["path"] not in seen_prediction_paths,"60distinct fresh prediction products")
        seen_prediction_paths.add(seal["prediction_ref"]["path"])
        _,records,_,identity=_predictions(seal["prediction_ref"],seal["focal_registry_ref"],cfg)
        require(records["parent_id"].item()==parent and identity==seal["focal_identity_sha256"],"Exact same original native focal identity")
        if parent in identities:require(identities[parent]==identity,"Both arms/all6passes identical parent focal support")
        identities[parent]=identity
        items.append(dict(arm=arm,epoch=epoch,parent_id=parent,prediction_seal_ref=record,
            prediction_ref=seal["prediction_ref"],focal_registry_ref=seal["focal_registry_ref"],
            focal_identity_sha256=identity,checkpoint_ref=seal["checkpoint_ref"]))
    return training,_cell_keys(items)

def seal_all60_v1(cfg,prediction_seal_refs,*,training_context_ref,evaluation_context_ref,writer,runtime_guard):
    """No truth resolver: authenticate every payload/identity before writing the joint barrier."""
    _config(cfg);require(callable(runtime_guard) and callable(getattr(writer,"save",None)),"ROOT capabilities")
    snapshot=_source_snapshot()
    training,cells=_validate_seals(prediction_seal_refs,training_context_ref=training_context_ref,
        evaluation_context_ref=evaluation_context_ref,cfg=cfg,runtime_guard=runtime_guard)
    _source_check(snapshot)
    return writer.save("H060_VAL_all60_predictions_sealed.json",dict(
        status="sealed_H060_all60_original_VAL_predictions_v1",frozen_protocol_ref=PROTOCOL,
        evaluation_source_manifest_ref=_manifest_ref(),training_context_ref=training_context_ref,
        training_result_ref=training["result_ref"],evaluation_context_ref=evaluation_context_ref,
        arms=list(ARMS),epochs=list(EPOCHS),validation_parent_ids=list(VAL),
        cells=[cells[k] for k in sorted(cells)],all60_sealed_before_either_arm_VAL_truth_or_selection=True,
        original_simpler_C3_all30_unchanged=True,TEST_deferred=True))

def _truth(record,records,cap,parent):
    import numpy as np
    value=_load_npz(record,cap)
    require(set(value)=={"intervals_bp","query_sample_indices","source_classes_0based","parent_id","class_order"}
        and np.array_equal(value["parent_id"],records["parent_id"])
        and np.array_equal(value["class_order"],records["class_order"]),"Same-parent external VAL truth schema/classes")
    intervals,samples,labels=value["intervals_bp"],value["query_sample_indices"],value["source_classes_0based"]
    require(intervals.dtype==np.float64 and intervals.ndim==2 and intervals.shape[1:]==(2,) and len(intervals)>0
        and np.isfinite(intervals).all() and intervals[0,0]==0 and intervals[-1,1]==30000000
        and (intervals[:,1]>intervals[:,0]).all() and np.array_equal(intervals[:-1,1],intervals[1:,0])
        and samples.dtype==np.int64 and np.array_equal(samples,np.arange(600,1600,dtype=np.int64))
        and labels.dtype==np.uint8 and labels.shape==(len(intervals),1000) and (labels<3).all(),
        "Full30Mb contiguous half-open truth, all1000original query columns; no fallback")
    rows=np.searchsorted(intervals[:,1],records["positions_bp0"],side="right")
    columns=records["sample_indices"]-600
    return np.asarray(labels[rows,columns],dtype=np.int64)

def _score(prediction,truth):
    import numpy as np
    from arggnn.american_c3_evaluate_v1 import metrics
    probabilities=prediction["probabilities"];calls=probabilities.argmax(-1)
    confusion=np.zeros((3,3),dtype=np.int64);np.add.at(confusion,(truth,calls),1)
    result=metrics(confusion);raw=probabilities[np.arange(10000),truth].astype(np.float64)
    zero=int(np.count_nonzero(raw==0))
    result["raw_NLL"]=dict(mean_nats=None if zero else float(-np.log(raw).mean()),
        positive_infinity=bool(zero),zero_true_class_probabilities=zero,
        probability_floor=False,probabilities_renormalized=False)
    return result

def _summary(values):
    require(len(values)==5 and all(math.isfinite(x) for x in values),"Exactly5finite parent rates")
    return dict(per_parent=dict(zip(VAL,values)),equal_parent_mean=statistics.mean(values),
        sample_SD=statistics.stdev(values),parents=5,SD_ddof=1)

def _rate_vector(result):
    rates=dict(macro_F1=100*result["macro_F1"],focal_accuracy=100*result["accuracy"])
    for name in CLASSES:
        for metric in ("precision","recall","F1"):rates[name+"_"+metric]=100*result["per_class"][name][metric]
    return rates

def _nll_summary(results):
    per_parent={p:r["raw_NLL"] for p,r in zip(VAL,results)}
    if any(r["raw_NLL"]["positive_infinity"] for r in results):
        return dict(per_parent=per_parent,equal_parent_mean_nats=None,positive_infinity=True,sample_SD_nats=None,
            sample_SD_status="undefined_for_nonfinite_parent_NLL",parents=5,SD_ddof=1)
    summary=_summary([r["raw_NLL"]["mean_nats"] for r in results])
    return dict(per_parent=per_parent,equal_parent_mean_nats=summary["equal_parent_mean"],positive_infinity=False,
        sample_SD_nats=summary["sample_SD"],parents=5,SD_ddof=1)

def _nll_difference(a,b):
    if a["positive_infinity"] or b["positive_infinity"]:
        both=a["positive_infinity"] and b["positive_infinity"]
        return dict(value_nats=None,infinity_sign=None if both else (1 if a["positive_infinity"] else -1),
            undefined=both,reason="infinity_minus_infinity" if both else "raw_infinite_difference")
    return dict(value_nats=a["mean_nats"]-b["mean_nats"],infinity_sign=0,undefined=False,reason=None)

def score_and_select_v1(cfg,matrix_ref,*,authority_ref,writer,resolve_val_truth,runtime_guard):
    """Separate VAL-only capability. Never resolve truth before full60payload authentication."""
    _config(cfg);require(callable(resolve_val_truth) and callable(runtime_guard) and callable(getattr(writer,"save",None)),"Separate ROOT VAL-only resolver/writer/resource callbacks")
    read,_,_=_metadata();matrix=read(matrix_ref)
    require(matrix.get("status")=="sealed_H060_all60_original_VAL_predictions_v1"
        and matrix.get("frozen_protocol_ref")==PROTOCOL and matrix.get("evaluation_source_manifest_ref")==_manifest_ref()
        and matrix.get("arms")==list(ARMS) and matrix.get("epochs")==list(EPOCHS)
        and matrix.get("validation_parent_ids")==list(VAL)
        and matrix.get("all60_sealed_before_either_arm_VAL_truth_or_selection") is True
        and matrix.get("original_simpler_C3_all30_unchanged") is True,"Separate complete joint60seal barrier")
    _cell_keys(matrix["cells"])
    _authority(authority_ref,phase="score_and_select",training_context_ref=matrix["training_context_ref"],
        evaluation_context_ref=matrix["evaluation_context_ref"],cfg=cfg,matrix_ref=matrix_ref)
    snapshot=_source_snapshot()
    training,cells=_validate_seals([row["prediction_seal_ref"] for row in matrix["cells"]],
        training_context_ref=matrix["training_context_ref"],evaluation_context_ref=matrix["evaluation_context_ref"],
        cfg=cfg,runtime_guard=runtime_guard)
    require(matrix["cells"]==[cells[k] for k in sorted(cells)] and matrix["training_result_ref"]==training["result_ref"],
        "Unchanged complete60cell matrix; no substitution before resolver")
    context,plans=_evaluation_context(matrix["evaluation_context_ref"])
    results={};score_refs={}
    # This is the first reachable VAL resolver invocation in this module.
    for parent in VAL:
        _source_check(snapshot);runtime_guard("before_H060_VAL_only_truth_resolution_after60seals")
        manifest=read(_plan(plans[parent],parent)["observation_ref"])
        simulation=read(context["simulation_completion_refs"][parent])
        require(simulation.get("status")=="completed_C3_simulation_reconstruction"
            and simulation.get("parent_id")==parent and simulation.get("class_order")==list(CLASSES)
            and manifest.get("source_observation_ref")==simulation.get("observation_ref"),
            "Actual originalVALsimulation truth lineage is the exact inferred-observation ancestor")
        truth_ref=resolve_val_truth(parent)
        require(truth_ref==simulation["private_truth_ref"],"Resolver returns exact genuine same-parent private truth")
        records=None;truth=None
        for arm in ARMS:
            for epoch in EPOCHS:
                key=(arm,epoch,parent);cell=cells[key]
                runtime_guard("H060_external_raw_native_focal_score")
                prediction,current,_,identity=_predictions(cell["prediction_ref"],cell["focal_registry_ref"],cfg)
                if records is None:records=current;truth=_truth(truth_ref,records,cfg["max_truth_uncompressed_bytes"],parent)
                require(identity==cell["focal_identity_sha256"],"No focal support change after barrier")
                result=_score(prediction,truth);results[key]=result
                score_refs[key]=writer.save("H060_%s_%s_pass%03d_score.json"%(arm,parent,epoch),dict(
                    status="completed_H060_external_VAL_cell_v1",arm=arm,epoch=epoch,parent_id=parent,
                    validation_matrix_ref=matrix_ref,prediction_seal_ref=cell["prediction_seal_ref"],
                    prediction_ref=cell["prediction_ref"],checkpoint_ref=cell["checkpoint_ref"],
                    focal_registry_ref=cell["focal_registry_ref"],focal_identity_sha256=identity,
                    external_truth_ref=truth_ref,records=10000,class_order=list(CLASSES),result=result,
                    probabilities_modified=False,probability_floor=False,VAL_used_for_optimization=False,TEST_deferred=True))
                del prediction,current
        del records,truth
    require(set(results)==set(cells),"All60scores complete before independent per-arm selection")
    tables={};selected={}
    for arm in ARMS:
        table=[]
        for epoch in EPOCHS:
            parent_results=[results[(arm,epoch,p)] for p in VAL]
            vectors=[_rate_vector(r) for r in parent_results]
            table.append(dict(epoch=epoch,checkpoint_ref=training["candidates"][epoch],
                mean_macro_F1_fraction_for_selection=sum(r["macro_F1"] for r in parent_results)/5,
                rates_percent={key:_summary([v[key] for v in vectors]) for key in vectors[0]},
                raw_NLL=_nll_summary(parent_results),score_refs={p:score_refs[(arm,epoch,p)] for p in VAL}))
        tables[arm]=table
        selected[arm]=min(table,key=lambda row:(-row["mean_macro_F1_fraction_for_selection"],row["epoch"]))
    def paired(epoch_full,epoch_nomem):
        vectors=[{key:_rate_vector(results[(ARMS[0],epoch_full,p)])[key]-_rate_vector(results[(ARMS[1],epoch_nomem,p)])[key]
            for key in _rate_vector(results[(ARMS[0],epoch_full,p)] )} for p in VAL]
        nll=[]
        for p in VAL:
            a=results[(ARMS[0],epoch_full,p)]["raw_NLL"];b=results[(ARMS[1],epoch_nomem,p)]["raw_NLL"]
            nll.append(_nll_difference(a,b))
        return dict(full_epoch=epoch_full,no_memory_epoch=epoch_nomem,
            rate_differences_percentage_points={k:_summary([v[k] for v in vectors]) for k in vectors[0]},
            raw_NLL_difference_nats=(dict(status="extended_real_parent_differences_sampleSD_undefined",per_parent=dict(zip(VAL,nll)))
                if any(x["undefined"] or x["infinity_sign"]!=0 for x in nll)
                else _summary([x["value_nats"] for x in nll])),
            direction="TreeLoom_full_minus_matched_NoMem",uncertainty_unit="same5simulationparents")
    _source_check(snapshot);runtime_guard("before_H060_final_independent_selection_receipt")
    return writer.save("H060_VAL_scores_and_independent_selections.json",dict(
        status="selected_H060_original5VAL_full_TreeLoom_checkpoints_v1",frozen_protocol_ref=PROTOCOL,
        evaluation_source_manifest_ref=_manifest_ref(),validation_matrix_ref=matrix_ref,score_authority_ref=authority_ref,
        training_context_ref=matrix["training_context_ref"],training_result_ref=training["result_ref"],
        criterion="greatest_equal_parent_mean_macro_F1_ties_earlier_exact_pass",
        all60_new_predictions_before_either_VAL_truth=True,all60_scores_before_either_arm_selection=True,
        candidate_tables=tables,selected=selected,paired_at_same_pass={str(e):paired(e,e) for e in EPOCHS},
        paired_independently_selected=paired(selected[ARMS[0]]["epoch"],selected[ARMS[1]]["epoch"]),
        metric_rate_unit="percent; parent_SD_and_paired_differences_in_percentage_points",
        raw_NLL_unit="nats_per_native_focal; positive_infinity_explicitly_tagged_without_floor",
        uncertainty="sampleSDacross5parents_ddof1;oneoptimizerseed;no_trainingseed_uncertainty",
        validation_development_only=True,independent_TEST_confirmation=False,TEST_deferred=True,
        original_simpler_C3_all30_unchanged=True,scientific_comparison="shared_data_pipeline_not_architecture_only_or_exact_paper_reproduction"))

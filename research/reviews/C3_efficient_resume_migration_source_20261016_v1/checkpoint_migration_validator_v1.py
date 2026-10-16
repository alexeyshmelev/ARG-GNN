"""Deferred strict CPU checkpoint migration. Import is stdlib-only and inert.
No admission, launcher, truth reader, model fit, GPU action or fresh-weight fallback.
Root alone owns actual stopped-boundary evidence, guarded reads and persistence.
"""
from pathlib import Path
import copy,hashlib,importlib.util,io,json,math,os,random,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_efficient_resume_migration_source_20261016_v1"
SOURCE=HERE+"/checkpoint_migration_validator_v1.py"
INTEGRATION_PATH=HERE+"/resume_integration_v1.py"
TRAINER_REF=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/trainer_prefetch_v1.py",
 sha256="b2f08a3e84bd5fb65c5f3d95566ab082fb71a2dd0a826180a10aa518aa35025c")
STEP_REF=dict(path="research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/efficient_step_v1.py",
 sha256="0df05b1957affc222ca1c11d9f34cc3277f5a3098834fb36fedee64bb242d2e8")
CACHE_REF=dict(path="research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/single_parent_device_cache_v1.py",
 sha256="d19879458474883e125b7efc7cbd649839f25c3db22ecb876ba3f7f5ab9b49b6")
_RAN=False

def require(ok,message):
    if not ok:raise ValueError(message)

def exact_ref(r):
    require(isinstance(r,dict) and set(r)=={"path","sha256"}
            and isinstance(r["path"],str) and not Path(r["path"]).is_absolute()
            and ".." not in Path(r["path"]).parts and isinstance(r["sha256"],str)
            and len(r["sha256"])==64 and all(x in "0123456789abcdef" for x in r["sha256"]),
            "Exact relative path/SHA reference")
    return r

def own_ref(path):
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
            and p.stat().st_size<=1048576,"Canonical bounded source metadata")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read_exact(r,cap,check):
    exact_ref(r);check("before_actual_migration_read")
    p=ROOT/r["path"]
    require(p.resolve()==p and p.is_relative_to(ROOT) and not p.is_symlink(),"Canonical exact project checkpoint/metadata")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,"Explicit bounded actual source/checkpoint")
        parts=[];h=hashlib.sha256();count=0
        while True:
            chunk=os.read(fd,min(1048576,cap-count+1))
            if not chunk:break
            count+=len(chunk);require(count<=cap,"Prospective checkpoint read cap")
            h.update(chunk);parts.append(chunk)
        after=os.fstat(fd)
    finally:os.close(fd)
    identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    require(identity(before)==identity(after)==identity(p.lstat()) and count==before.st_size
            and h.hexdigest()==r["sha256"],"Actual stable byte-bound original/migrated checkpoint")
    check("after_actual_migration_read")
    return b"".join(parts)

def read_json(r,check):
    raw=read_exact(r,1048576,check)
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,"Unique metadata JSON keys");out[k]=v
        return out
    def invalid(value):raise ValueError("Nonfinite metadata JSON constant "+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def load_exact(r,label):
    require(own_ref(r["path"])==r,"Exact deferred migration source")
    spec=importlib.util.spec_from_file_location(label,ROOT/r["path"])
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def validate_intent_v1(record,check):
    value=read_json(record,check)
    fields={"schema","active","root_authorizer","old_boundary_ref","old_protocol_sha256",
            "old_job_id","old_input_ref","old_terminal_observation_ref","old_scheduled_checkpoints",
            "new_integration_ref","new_runtime_ref","bounds_ref","qualification_ref",
            "migration_output_directory","max_checkpoint_bytes","expected_actual_global_step",
            "continuation_mode","rollback_authorization_ref"}
    require(set(value)==fields and value["schema"]=="C3_actual_complete_boundary_migration_intent_v1"
            and value["active"] is True and value["root_authorizer"]=="root_agent",
            "Separate explicit actual complete-boundary migration intent")
    for key in ("old_boundary_ref","old_input_ref","old_terminal_observation_ref","new_integration_ref",
                "new_runtime_ref","bounds_ref","qualification_ref"):exact_ref(value[key])
    require(value["new_integration_ref"]["path"]==INTEGRATION_PATH
            and value["new_runtime_ref"]["path"]==HERE+"/root_resume_runtime_v1.py"
            and own_ref(value["new_runtime_ref"]["path"])==value["new_runtime_ref"]
            and type(value["max_checkpoint_bytes"])is int and 0<value["max_checkpoint_bytes"]<=2*2**30
            and type(value["expected_actual_global_step"])is int and 0<value["expected_actual_global_step"]<600,
            "Explicit bounded actual unfinished progress, never inferred/default zero")
    original_input=read_json(value["old_input_ref"],check)
    require(original_input["job_id"]==value["old_job_id"] and original_input["phase"]=="train"
            and original_input["physical_gpu"]==0,"Actual original GPU0 C3 job identity")
    terminal=read_json(value["old_terminal_observation_ref"],check)
    require(terminal.get("schema")=="C3_actual_stopped_boundary_transition_evidence_v1"
            and terminal.get("root_authorizer")=="root_agent" and terminal.get("old_job_id")==value["old_job_id"]
            and terminal.get("old_input_ref")==value["old_input_ref"]
            and terminal.get("actual_source_job_terminal_consumed") is True
            and type(terminal.get("actual_source_exit_code"))is int
            and terminal.get("all_old_owned_processes_drained") is True
            and terminal.get("all_old_future_write_relaunch_capabilities_revoked") is True
            and terminal.get("actual_boundary_ref")==value["old_boundary_ref"]
            and type(terminal.get("actual_worker_pid"))is int and terminal["actual_worker_pid"]>1
            and type(terminal.get("actual_worker_start_ticks"))is int and terminal["actual_worker_start_ticks"]>0
            and terminal.get("actual_boundary_update_count")==value["expected_actual_global_step"],
            "Genuine root terminal/drain/revocation and exact actual boundary required, no live migration")
    mode=value["continuation_mode"]
    require(mode in ("exact_latest_boundary","root_authorized_durable_scheduled_checkpoint_rollback"),
            "Explicit exact continuation or separately root-approved transparent rollback")
    elapsed=terminal.get("original_attempt_elapsed_seconds")
    elapsed_refs=terminal.get("original_attempt_elapsed_provenance_refs")
    require(type(elapsed)in (int,float) and math.isfinite(elapsed) and elapsed>=0
            and isinstance(elapsed_refs,list) and elapsed_refs,"Original attempt time recorded separately with provenance")
    for evidence_ref in elapsed_refs:read_json(evidence_ref,check)
    if mode=="exact_latest_boundary":
        require(value["rollback_authorization_ref"] is None
                and terminal.get("actual_original_optimizer_step")==value["expected_actual_global_step"]
                and terminal.get("latest_completed_boundary_verified") is True
                and terminal.get("no_completed_parent_updates_after_boundary") is True
                and isinstance(terminal.get("actual_cursor_provenance_refs"),list)
                and terminal["actual_cursor_provenance_refs"],
                "Exact-latest mode needs genuine no-later-update proof, never infer it from checkpoint")
        for evidence_ref in terminal["actual_cursor_provenance_refs"]:read_json(evidence_ref,check)
    else:
        approval=read_json(value["rollback_authorization_ref"],check)
        require(approval.get("schema")=="C3_root_explicit_durable_checkpoint_rollback_authorization_v1"
                and approval.get("active") is True and approval.get("root_authorizer")=="root_agent"
                and approval.get("old_job_id")==value["old_job_id"] and approval.get("old_input_ref")==value["old_input_ref"]
                and approval.get("selected_checkpoint_ref")==value["old_boundary_ref"]
                and approval.get("selected_checkpoint_global_step")==value["expected_actual_global_step"]
                and approval.get("latest_genuine_durable_scheduled_checkpoint_selected") is True
                and approval.get("rollback_of_later_uncheckpointed_work_explicitly_authorized") is True
                and approval.get("discarded_or_repeated_TRAIN_work_disclosed") is True
                and approval.get("no_work_lost_claimed") is False
                and approval.get("preserve_all_old_failed_and_checkpoint_artifacts") is True
                and approval.get("original_attempt_elapsed_seconds")==elapsed
                and approval.get("fresh_weights_or_optimizer_reset_authorized") is False,
                "Narrow actual root-approved rollback, complete checkpoint state preserved and work loss disclosed")
        later=approval.get("later_completed_update_count")
        require(later is None or type(later)is int and later>=0,"Later discarded update count may honestly remain unknown")
        if later is not None:
            proofs=approval.get("later_update_count_provenance_refs")
            require(isinstance(proofs,list) and proofs,"Known later count needs actual provenance")
            for evidence_ref in proofs:read_json(evidence_ref,check)
        refs=approval.get("durable_checkpoint_selection_provenance_refs")
        require(isinstance(refs,list) and refs,"Actual scheduled-checkpoint selection provenance")
        for evidence_ref in refs:read_json(evidence_ref,check)
    # A checkpoint proves its own full state. Transparent rollback does not require
    # an unknowable latest live counter and never claims that no work was lost.
    output=Path(value["migration_output_directory"])
    require(not output.is_absolute() and ".." not in output.parts and output.parts
            and str(output).startswith("experiments/results/") and output.as_posix()==str(output),
            "New root-charged migration output scope")
    require(isinstance(value["old_scheduled_checkpoints"],list) and len(value["old_scheduled_checkpoints"])<=5,
            "At most five actual candidates before an unfinished600-update boundary")
    old_scope=Path(original_input["output_directory"])
    require(output!=old_scope and output not in old_scope.parents and old_scope not in output.parents,
            "Separate migration scope outside the entire original training output")
    for old in [value["old_boundary_ref"],*[x["checkpoint_ref"] for x in value["old_scheduled_checkpoints"]]]:
        require(not (ROOT/old["path"]).is_relative_to(ROOT/output),"Original evidence outside new migration scope")
    return value,original_input

def expected_parameter_shapes_v1():
    out={}
    def linear(prefix,a,b,bias=True):
        out[prefix+".weight"]=(a,b)
        if bias:out[prefix+".bias"]=(a,)
    linear("site_mlp.0",32,15);linear("site_mlp.2",32,32)
    linear("site_pool_projection.0",32,67);linear("leaf_projection.0",32,37);linear("node_projection.0",32,3)
    linear("up_gate",32,9);linear("down_gate",32,9);linear("up_message",32,32,False)
    linear("up_update",32,64);out["up_norm.weight"]=(32,);out["up_norm.bias"]=(32,)
    linear("down_message",32,32,False);linear("down_update",32,32);out["down_norm.weight"]=(32,);out["down_norm.bias"]=(32,)
    linear("leaf_context.0",32,64)
    for prefix in ("forward_gru","reverse_gru"):
        out[prefix+".weight_ih_l0"]=(96,35);out[prefix+".weight_hh_l0"]=(96,32)
        out[prefix+".bias_ih_l0"]=(96,);out[prefix+".bias_hh_l0"]=(96,)
    linear("classifier.0",32,96);linear("classifier.2",3,32)
    return out

def digest_value_v1(value,torch):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            require(x.device.type=="cpu" and x.layout==torch.strided,"CPU tensor state only")
            y=x.detach().contiguous()
            h.update(b"T"+str(y.dtype).encode()+str(tuple(y.shape)).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b"{")
            for k,v in x.items():visit(k);visit(v)
            if hasattr(x,"_metadata"):h.update(b"state_dict_metadata");visit(x._metadata)
            h.update(b"}")
        elif isinstance(x,(list,tuple)):
            h.update(b"L" if isinstance(x,list) else b"U")
            for v in x:visit(v)
            h.update(b";")
        else:
            require(x is None or type(x)in (str,int,float,bool),"Bounded weights-only checkpoint primitives")
            h.update(type(x).__name__.encode()+json.dumps(x,allow_nan=False,separators=(",",":")).encode()+b"\0")
    visit(value);return h.hexdigest()

def protected_state_v1(value):
    out={k:v for k,v in value.items() if k!="execution_migration"}
    state=dict(value["trainer_state"])
    state.pop("protocol");state.pop("protocol_sha256")
    state["scheduled_checkpoints"]=[dict(epoch=x["epoch"],global_step=x["global_step"])
                                     for x in state["scheduled_checkpoints"]]
    out["trainer_state"]=state
    return out

def validate_full_checkpoint_v1(value,old_protocol,trainer,torch):
    state=value.get("trainer_state",{});step=trainer._validate_progress(state)
    require(value.get("schema")=="american_c3_fresh_whole_tree_GNN_checkpoint_v1"
            and value.get("model_seed")==70001 and value.get("training_condition")==trainer.CONDITION
            and value.get("C4_initialization_or_head_slicing") is False
            and "execution_migration" not in value and state.get("schema")==trainer.SCHEMA
            and state.get("protocol")==old_protocol and state.get("protocol_sha256")==trainer.json_sha(old_protocol),
            "Original full C3 protocol and checkpoint, not model-only or an implicit remigration")
    cfg=dict(old_protocol["config"],execution_enabled=True);trainer.validate_config_v1(cfg)
    expected_contract=dict(schema="american_c3_fresh_whole_tree_GNN_checkpoint_v1",class_order=["EUR","EAS","AFR"],
        hidden_dim=32,donor_cohorts=2,donor_channels=6,expected_haplotypes=1600,initialization="fresh_no_C4_loading",
        site_feature_width=15,observed_leaf_width=5,recurrent_contract="one_layer_bidirectional_equations_as_two_directional_GRUs",
        tree_backbone="unchanged_PyG_gated_upward_recipient_excluded_downward_cavity",whole_parent_training_required=True)
    require(value.get("contract")==expected_contract,"Exact full1600 C3 model contract")
    shapes=expected_parameter_shapes_v1();model=value.get("model_state",{})
    require(list(model)==list(shapes),"Exact source-declared complete parameter order")
    for name,shape in shapes.items():
        x=model[name];require(torch.is_tensor(x) and x.device.type=="cpu" and x.dtype==torch.float32
            and tuple(x.shape)==shape and bool(torch.isfinite(x).all()),"Finite full original FP32 parameter: "+name)
    trained=[p for p in trainer.TRAIN if any(r["parent_id"]==p for r in state["history"])]
    require(value["trained_parent_ids"]==trained and value["train_sequence_lengths_bp"]==[2000000]*len(trained),
            "Exact actual completed TRAIN membership")
    opt=value.get("optimizer_state",{});groups=opt.get("param_groups",[]);moments=opt.get("state",{})
    require(len(groups)==1 and groups[0]["params"]==list(range(len(shapes)))
            and set(moments)==set(range(len(shapes))),"Complete original AdamW parameter map and moments")
    group=groups[0]
    require(group["lr"]==.001 and group["weight_decay"]==.0001 and tuple(group["betas"])==(.9,.999)
            and group["eps"]==1e-8 and all(group.get(k,False) is False for k in
            ("amsgrad","maximize","foreach","capturable","differentiable","fused")),"Unchanged exact original AdamW settings")
    for index,(name,shape) in enumerate(shapes.items()):
        item=moments[index];require(set(item)=={"step","exp_avg","exp_avg_sq"},"Complete non-AMSGrad AdamW state")
        counter=item["step"]
        require(torch.is_tensor(counter) and counter.device.type=="cpu" and counter.numel()==1
                and bool(torch.isfinite(counter).all()) and float(counter.item())==step,
                "Every actual optimizer counter equals complete-parent global cursor")
        for key in ("exp_avg","exp_avg_sq"):
            x=item[key];require(torch.is_tensor(x) and x.device.type=="cpu" and x.dtype==torch.float32
                and tuple(x.shape)==shape and bool(torch.isfinite(x).all()),"Full finite unchanged AdamW moment: "+name+key)
    rng=state["rng_state"];require(set(rng)=={"python","numpy","torch_CPU","torch_device","device_type"}
                                  and rng["device_type"]=="cuda","Original CUDA training RNG state, no CPU replacement")
    local_random=random.Random();local_random.setstate(rng["python"])
    nr=rng["numpy"]
    require(set(nr)=={"algorithm","keys","position","has_gauss","cached_gaussian"} and nr["algorithm"]=="MT19937"
            and isinstance(nr["keys"],list) and len(nr["keys"])==624
            and all(type(x)is int and 0<=x<2**32 for x in nr["keys"])
            and type(nr["position"])is int and 0<=nr["position"]<=624 and nr["has_gauss"]in (0,1)
            and isinstance(nr["cached_gaussian"],(int,float)) and math.isfinite(nr["cached_gaussian"]),
            "Complete original NumPy RNG state")
    for key in ("torch_CPU","torch_device"):
        x=rng[key];require(torch.is_tensor(x) and x.device.type=="cpu" and x.dtype==torch.uint8
                and x.ndim==1 and x.numel()>0,"Original complete Torch CPU/CUDA RNG bytes")
    for key in ("observation_seal_sha256","target_join_sha256"):
        require(isinstance(state[key],str) and len(state[key])==64,"Exact original data/target seals")
    for key in ("observation_manifest_ref","target_manifest_ref"):exact_ref(state[key])
    candidate=value["candidate"]
    if candidate["eligible_for_VAL_selection"] is True:
        trainer.scheduled_checkpoint_metadata_v1(value,protocol_sha256=trainer.json_sha(old_protocol))
    else:
        require(candidate==dict(eligible_for_VAL_selection=False,kind="complete_parent_resume_boundary",
                completed_epoch=step//20,global_step=step),"Actual complete-parent boundary, no partial optimizer state")
    return step

def validate_protocol_transition_v1(old,new,intent_ref,intent,trainer):
    fixed={"schema","config","parent_plan_refs","parent_order","loss","no_C4_initialization","H043_extension",
           "actual_TRAIN_execution_binding","observation_provider_execution_binding"}
    require(set(old)==fixed|{"source_refs","exact_gradient"}
            and set(new)==set(old)|{"efficient_same_objective_execution_binding"},
            "Only declared execution metadata may differ, no unreviewed scientific protocol keys")
    require(all(k in old and k in new and old[k]==new[k] for k in fixed),
            "Unchanged exact model/config/data/labels/order/objective and observed provider/target execution bindings")
    require(old["exact_gradient"]=="checkpoint_encoder_and_directional_GRUs_CPU_boundaries_without_detach"
            and "efficient_same_objective_execution_binding" not in old,
            "Exact original live gradient family only, never arbitrary source migration")
    old_refs={r["path"]:r for r in old["source_refs"]};new_refs={r["path"]:r for r in new["source_refs"]}
    require(all(new_refs.get(p)==r for p,r in old_refs.items()),"Every original source reference retained unchanged")
    for r in new["source_refs"]:require(own_ref(r["path"])==r,"Exact actual old/successor source closure")
    bind=new["efficient_same_objective_execution_binding"]
    require(bind["resume_migration_intent_ref"]==intent_ref and bind["migration_validator_ref"]==own_ref(SOURCE)
            and bind["integration_source_ref"]==intent["new_integration_ref"] and bind["bounds_ref"]==intent["bounds_ref"]
            and bind["step_source_ref"]==STEP_REF and bind["cache_source_ref"]==CACHE_REF
            and bind["no_progress_reset"] is True,"Explicit same-objective source migration binding")
    require(trainer.json_sha(old)==intent["old_protocol_sha256"],"Actual old protocol signature")

def migrate_complete_boundary_v1(intent_ref,*,resource_check,persist_checkpoint):
    global _RAN
    require(not _RAN and callable(resource_check) and callable(persist_checkpoint),"One root-owned actual migration, no default writer")
    resource_check("before_actual_checkpoint_migration_metadata")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only actual migration preparation, no GPU access")
    intent,old_input=validate_intent_v1(intent_ref,resource_check)
    _RAN=True
    resource_check("before_actual_migration_numerical_checkpoint_import")
    import torch
    trainer=load_exact(TRAINER_REF,"_C3_original_checkpoint_schema_for_migration")
    integration=load_exact(intent["new_integration_ref"],"_C3_resume_expected_signature_only")
    require(integration.MIGRATION_VALIDATOR_REF==own_ref(SOURCE),"Reciprocal exact migration/integration source binding")
    prior_rng=torch.random.get_rng_state().clone()
    cap=intent["max_checkpoint_bytes"]
    def read_checkpoint(record):
        raw=read_exact(record,cap,resource_check)
        value=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
        require(isinstance(value,dict),"Actual complete checkpoint dictionary")
        return value
    try:
        boundary=read_checkpoint(intent["old_boundary_ref"]);old=boundary["trainer_state"]["protocol"]
        step=validate_full_checkpoint_v1(boundary,old,trainer,torch)
        if intent["continuation_mode"]=="root_authorized_durable_scheduled_checkpoint_rollback":
            trainer.scheduled_checkpoint_metadata_v1(boundary,protocol_sha256=trainer.json_sha(old))
        require(step==intent["expected_actual_global_step"] and 0<step<600,
                "Actual optimizer/cursor progress proven, only remaining600-step suffix may run")
        require(trainer.json_sha(old)==intent["old_protocol_sha256"]
                and all(old_input["source_sha256"].get(r["path"])==r["sha256"] for r in old["source_refs"]),
                "Actual admitted original source family and checkpoint signature")
        actual_bounds=integration.read_bounds_v1(intent["bounds_ref"])
        require(actual_bounds["qualification_ref"]==intent["qualification_ref"],"Exact already passed ordinaryFP32 step qualification")
        make_expected=integration.make_training_callable_v1(intent["bounds_ref"],
            migration_intent_ref=intent_ref,metadata_only=True)
        new,new_sha=make_expected(old)
        validate_protocol_transition_v1(old,new,intent_ref,intent,trainer)
        original_entries=copy.deepcopy(boundary["trainer_state"]["scheduled_checkpoints"])
        if boundary["candidate"]["eligible_for_VAL_selection"]:
            original_entries.append(dict(epoch=step//20,global_step=step,checkpoint_ref=intent["old_boundary_ref"]))
        require(original_entries==intent["old_scheduled_checkpoints"],"Every and only actual scheduled original candidate, no missing or invented epoch")
        converted={};receipts=[]
        def migrate_one(record,value,label):
            validate_full_checkpoint_v1(value,old,trainer,torch)
            original_digest=digest_value_v1(value,torch);protected=digest_value_v1(protected_state_v1(value),torch)
            changed=dict(value);state=dict(value["trainer_state"])
            state["protocol"]=copy.deepcopy(new);state["protocol_sha256"]=new_sha
            state["scheduled_checkpoints"]=[dict(epoch=x["epoch"],global_step=x["global_step"],
                checkpoint_ref=converted[(x["checkpoint_ref"]["path"],x["checkpoint_ref"]["sha256"])])
                for x in state["scheduled_checkpoints"]]
            changed["trainer_state"]=state
            changed["execution_migration"]=dict(schema="C3_full_state_same_objective_execution_migration_v1",
                migration_intent_ref=intent_ref,original_checkpoint_ref=record,
                continuation_mode=intent["continuation_mode"],rollback_authorization_ref=intent["rollback_authorization_ref"],
                original_protocol_sha256=intent["old_protocol_sha256"],successor_protocol_sha256=new_sha,
                original_global_step=value["trainer_state"]["global_step"],protected_state_sha256=protected,
                original_implementation_trained_this_prefix=True,new_implementation_training_prefix_not_claimed=True,
                model_optimizer_cursor_history_RNG_data_order_unchanged=True,optimizer_updates_performed=0)
            require(digest_value_v1(protected_state_v1(changed),torch)==protected,"Exact protected complete state retained before write")
            out=ROOT/intent["migration_output_directory"];target=out/label
            require(out.resolve()==out and target.parent==out and not target.exists() and not target.is_symlink(),
                    "Fresh canonical separate checkpoint path before root exclusive write")
            resource_check("before_root_guarded_migrated_checkpoint_write")
            result=persist_checkpoint(target.relative_to(ROOT).as_posix(),changed)
            exact_ref(result)
            require(ROOT/result["path"]==target and result!=record,
                    "Separate actual root-charged migrated copy, never original overwrite")
            saved=read_checkpoint(result)
            require(digest_value_v1(saved,torch)==digest_value_v1(changed,torch)
                    and digest_value_v1(protected_state_v1(saved),torch)==protected
                    and digest_value_v1(value,torch)==original_digest,
                    "Actual serialized readback preserves all state and untouched original after root persistence")
            trainer._validate_progress(saved["trainer_state"])
            if saved["candidate"]["eligible_for_VAL_selection"]:
                trainer.scheduled_checkpoint_metadata_v1(saved,protocol_sha256=new_sha)
            receipts.append(dict(original_ref=record,migrated_ref=result,global_step=state["global_step"],
                protected_state_sha256=protected,serialized_readback_verified=True))
            converted[(record["path"],record["sha256"])]=result
            return result
        for item in original_entries:
            record=item["checkpoint_ref"]
            value=boundary if record==intent["old_boundary_ref"] else read_checkpoint(record)
            actual_step=validate_full_checkpoint_v1(value,old,trainer,torch)
            meta=trainer.scheduled_checkpoint_metadata_v1(value,protocol_sha256=intent["old_protocol_sha256"])
            require(meta["epoch"]==item["epoch"] and actual_step==item["global_step"] and actual_step<=step
                    and value["trainer_state"]["history"]==boundary["trainer_state"]["history"][:actual_step]
                    and value["trainer_state"]["observation_seal_sha256"]==boundary["trainer_state"]["observation_seal_sha256"]
                    and value["trainer_state"]["target_join_sha256"]==boundary["trainer_state"]["target_join_sha256"],
                    "Exact chronological original candidates from this completed prefix")
            migrate_one(record,value,"checkpoint_epoch_%03d.pt"%item["epoch"])
        key=(intent["old_boundary_ref"]["path"],intent["old_boundary_ref"]["sha256"])
        boundary_ref=converted.get(key)
        if boundary_ref is None:boundary_ref=migrate_one(intent["old_boundary_ref"],boundary,"resume_complete_parent_boundary.pt")
        for original in [intent["old_boundary_ref"],*[x["checkpoint_ref"] for x in original_entries]]:
            read_exact(original,cap,resource_check)
        require(torch.equal(prior_rng,torch.random.get_rng_state()),"Migration never consumes/restores/reseeds caller Torch RNG")
        resource_check("after_actual_complete_state_migration")
        terminal=read_json(intent["old_terminal_observation_ref"],resource_check)
        rollback=(intent["continuation_mode"]=="root_authorized_durable_scheduled_checkpoint_rollback")
        approval=(read_json(intent["rollback_authorization_ref"],resource_check) if rollback else None)
        return dict(schema="C3_complete_state_execution_migration_manifest_v1",root_authorizer="root_agent",passed=True,
            actual_migration_tool_terminal_consumed=False,actual_migration_tool_exit_code=None,
            migration_intent_ref=intent_ref,migration_validator_ref=own_ref(SOURCE),integration_source_ref=intent["new_integration_ref"],
            runtime_source_ref=intent["new_runtime_ref"],bounds_ref=intent["bounds_ref"],qualification_ref=intent["qualification_ref"],
            old_boundary_ref=intent["old_boundary_ref"],migrated_boundary_ref=boundary_ref,
            old_protocol_sha256=intent["old_protocol_sha256"],successor_protocol_sha256=new_sha,
            actual_global_step=step,actual_AdamW_step=step,remaining_updates=600-step,total_planned_updates=600,
            continuation_mode=intent["continuation_mode"],rollback_authorization_ref=intent["rollback_authorization_ref"],
            later_completed_update_count=(approval.get("later_completed_update_count") if rollback else 0),
            discarded_or_repeated_work_possible=rollback,no_work_lost_claimed=False,
            selected_checkpoint_complete_state_preserved=True,
            exact_completed_update_progress_preserved=(not rollback),
            original_attempt_elapsed_seconds=terminal["original_attempt_elapsed_seconds"],
            original_attempt_elapsed_provenance_refs=terminal["original_attempt_elapsed_provenance_refs"],
            effective_trajectory="retained_checkpoint_prefix_then_only_remaining_suffix_to600",
            physical_training_work_may_exceed600_updates_due_to_discarded_or_repeated_work=rollback,
            observation_seal_sha256=boundary["trainer_state"]["observation_seal_sha256"],
            target_join_sha256=boundary["trainer_state"]["target_join_sha256"],
            original_scheduled_checkpoints=original_entries,migrated_checkpoint_receipts=receipts,
            original_model_optimizer_cursor_RNG_order_history_unchanged=True,optimizer_updates_performed=0,
            original_sources_and_checkpoint_files_untouched=True,execution_or_stop_or_relaunch_authorized=False,
            bitwise_future_training_trajectory_claimed=False,fullparent_fit_speedup_or_accuracy_claimed=False)
    finally:
        require(torch.equal(prior_rng,torch.random.get_rng_state()),"No hidden Torch RNG modification during migration, including failure")

"""Future guarded C3 training callable. Imports stdlib only until root invokes.
Private original trainer, with exact operational progress hook after safe cursor.
"""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_efficient_resume_migration_source_20261016_v1"
SOURCE=HERE+"/resume_integration_v1.py"
TRAINER_REF=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/trainer_prefetch_v1.py",
 sha256="b2f08a3e84bd5fb65c5f3d95566ab082fb71a2dd0a826180a10aa518aa35025c")
STEP_REF={'path': 'research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/efficient_step_v1.py', 'sha256': '0df05b1957affc222ca1c11d9f34cc3277f5a3098834fb36fedee64bb242d2e8'}
CACHE_REF={'path': 'research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/single_parent_device_cache_v1.py', 'sha256': 'd19879458474883e125b7efc7cbd649839f25c3db22ecb876ba3f7f5ab9b49b6'}

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    p=ROOT/path
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=1048576,
            "Exact bounded canonical source/metadata")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def load(record,label):
    require(ref(record["path"])==record,"Exact future source before runtime import")
    spec=importlib.util.spec_from_file_location(label,ROOT/record["path"])
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def read_bounds_v1(record,*,allow_unqualified=False):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,
            "Exact actual root bounds reference")
    value=json.loads((ROOT/record["path"]).read_text())
    require(set(value)=={"schema","active","root_authorizer","phase","step_source_ref","integration_source_ref",
                        "cache_source_ref","precision","autocast","TF32","memory_caps","progress",
                        "qualification_ref","execution_authority_granted"}
            and value["schema"]=="C3_efficient_query_cache_bounds_v1" and value["active"] is True
            and value["root_authorizer"]=="root_agent" and value["phase"]=="train"
            and value["step_source_ref"]==STEP_REF and value["cache_source_ref"]==CACHE_REF
            and value["integration_source_ref"]==ref(SOURCE)
            and value["precision"]=="fp32" and value["autocast"] is False and value["TF32"] is False
            and value["execution_authority_granted"] is False,"Prospective bounds only, separate original admission required")
    require(value["progress"]==dict(per_completed_optimizer_update=True,stdout_record_cap_bytes=2048,
            durable_record_cap_bytes=8192,max_records=600,writer="original_PilotOutputV9_save",
            original_storage_boundary_per_receipt_preserved=True),
            "Exact bounded per-update progress and original writer guard")
    if not allow_unqualified:
        qref=value["qualification_ref"]
        require(isinstance(qref,dict) and ref(qref["path"])==qref,"Root actual new qualification required")
        q=json.loads((ROOT/qref["path"]).read_text())
        require(q.get("schema")=="C3_efficient_query_cache_root_qualification_v1"
                and q.get("root_authorizer")=="root_agent" and q.get("passed") is True
                and q.get("actual_tool_terminal_consumed") is True and q.get("actual_tool_exit_code")==0
                and q.get("step_source_ref")==STEP_REF and q.get("cache_source_ref")==CACHE_REF
                and q.get("precision")=="fp32" and q.get("optimizer_steps")==1
                and q.get("full_training_fit_or_accuracy_claimed") is False,
                "Exact actual ordinaryFP32 query/cache gradient+AdamW qualification, no manufactured success")
    return value

MIGRATION_VALIDATOR_REF={'path': 'research/reviews/C3_efficient_resume_migration_source_20261016_v1/checkpoint_migration_validator_v1.py', 'sha256': '31414598d0f69533864cf600a9a6f0c24ef5f0d81f9ca131fb5f9612f761fafd'}

def read_resume_manifest_v1(record,bounds_ref):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,
            "Actual separately root-sealed migration manifest")
    m=json.loads((ROOT/record["path"]).read_text())
    require(m.get("schema")=="C3_complete_state_execution_migration_manifest_v1"
            and m.get("root_authorizer")=="root_agent" and m.get("passed") is True
            and m.get("actual_migration_tool_terminal_consumed") is True
            and m.get("actual_migration_tool_exit_code")==0
            and m.get("migration_validator_ref")==MIGRATION_VALIDATOR_REF
            and m.get("integration_source_ref")==ref(SOURCE) and m.get("bounds_ref")==bounds_ref
            and m.get("original_model_optimizer_cursor_RNG_order_history_unchanged") is True
            and m.get("original_sources_and_checkpoint_files_untouched") is True
            and m.get("optimizer_updates_performed")==0
            and m.get("execution_or_stop_or_relaunch_authorized") is False
            and type(m.get("actual_global_step"))is int and 0<m["actual_global_step"]<600
            and m.get("actual_AdamW_step")==m["actual_global_step"]
            and m.get("remaining_updates")==600-m["actual_global_step"] and m.get("total_planned_updates")==600,
            "Genuine actual full-state migration terminal, no fresh or guessed progress")
    intent_ref=m["migration_intent_ref"]
    require(ref(intent_ref["path"])==intent_ref,"Exact original-boundary migration intent")
    intent=json.loads((ROOT/intent_ref["path"]).read_text())
    bounds=read_bounds_v1(bounds_ref)
    require(intent.get("schema")=="C3_actual_complete_boundary_migration_intent_v1"
            and intent.get("active") is True and intent.get("root_authorizer")=="root_agent"
            and intent["new_integration_ref"]==ref(SOURCE) and intent["bounds_ref"]==bounds_ref
            and intent["qualification_ref"]==m["qualification_ref"]==bounds["qualification_ref"]
            and intent["old_boundary_ref"]==m["old_boundary_ref"]
            and intent["new_runtime_ref"]==m["runtime_source_ref"]
            and intent["expected_actual_global_step"]==m["actual_global_step"],
            "Exact original/successor source/cursor/qualified-bounds lineage")
    require(m.get("continuation_mode")==intent["continuation_mode"]
            and m.get("rollback_authorization_ref")==intent["rollback_authorization_ref"]
            and type(m.get("original_attempt_elapsed_seconds"))in (int,float),
            "Explicit continuation/rollback mode and separately disclosed original elapsed time")
    if intent["continuation_mode"]=="root_authorized_durable_scheduled_checkpoint_rollback":
        require(m.get("discarded_or_repeated_work_possible") is True and m.get("no_work_lost_claimed") is False
                and (m.get("later_completed_update_count") is None or type(m["later_completed_update_count"])is int),
                "Honest known-or-unknown discarded work, no exact-latest claim in rollback mode")
        require(ref(intent["rollback_authorization_ref"]["path"])==intent["rollback_authorization_ref"],
                "Actual explicit root rollback authorization")
    else:require(intent["continuation_mode"]=="exact_latest_boundary","Declared source continuation mode")
    boundary=m["migrated_boundary_ref"]
    require(isinstance(boundary,dict) and set(boundary)=={"path","sha256"}
            and isinstance(boundary["sha256"],str) and len(boundary["sha256"])==64
            and Path(boundary["path"]).suffix==".pt"
            and not Path(boundary["path"]).is_absolute() and ".." not in Path(boundary["path"]).parts
            and (ROOT/boundary["path"]).is_relative_to(ROOT/intent["migration_output_directory"]),
            "Exact separately materialized full-state boundary, hashed by unchanged trainer before load")
    return m,intent

def make_training_callable_v1(bounds_ref,resume_manifest_ref=None,*,migration_intent_ref=None,metadata_only=False):
    bounds=read_bounds_v1(bounds_ref)
    if metadata_only:
        require(resume_manifest_ref is None and migration_intent_ref is not None,
                "Explicit migration signature preparation only, no training")
        require(ref(migration_intent_ref["path"])==migration_intent_ref,"Exact actual migration intent")
        intent=json.loads((ROOT/migration_intent_ref["path"]).read_text())
        require(intent["new_integration_ref"]==ref(SOURCE) and intent["bounds_ref"]==bounds_ref,
                "Expected successor source/bounds for migration")
        manifest=None
    else:
        require(migration_intent_ref is None and resume_manifest_ref is not None,
                "Actual sealed migration manifest mandatory, no fresh fallback")
        manifest,intent=read_resume_manifest_v1(resume_manifest_ref,bounds_ref)
        migration_intent_ref=manifest["migration_intent_ref"]
    require(ref(MIGRATION_VALIDATOR_REF["path"])==MIGRATION_VALIDATOR_REF,"Exact full-state validator source")
    step=load(STEP_REF,"_C3_efficient_query_device_step_private")
    trainer=load(TRAINER_REF,"_C3_efficient_frozen_trainer_private")
    original_signature=trainer._signature
    binding=dict(schema="C3_efficient_query_cache_same_objective_binding_v1",
        original_trainer_ref=TRAINER_REF,original_shared_context_ref=step.BASE_REF,
        step_source_ref=STEP_REF,cache_source_ref=CACHE_REF,integration_source_ref=ref(SOURCE),
        bounds_ref=copy.deepcopy(bounds_ref),full_graph_haplotypes=1600,recurrent_query_columns=1000,
        all_focal_labels=10000,complete_parent_mean_CE=True,one_global_clip_AdamW_per_parent=True,
        full_bidirectional_state_carry=True,encoder_gradient="originalencoder_queryselector_VJP",
        seed_budget_model_and_candidate_selection_unchanged=True,no_learned_cache_across_updates=True,
        old_family_resume_rejected=True,current_live_run_untouched=True,
        resume_migration_intent_ref=copy.deepcopy(migration_intent_ref),migration_validator_ref=MIGRATION_VALIDATOR_REF,
        no_progress_reset=True,original_implementation_training_prefix_retained=True)
    trainer.SOURCES=tuple(dict.fromkeys((*trainer.SOURCES,STEP_REF["path"],CACHE_REF["path"],SOURCE,bounds_ref["path"],
        MIGRATION_VALIDATOR_REF["path"],migration_intent_ref["path"])))
    def signature(cfg,plans):
        value,_=original_signature(cfg,plans)
        value["exact_gradient"]="shared_device_query_context_adjoints_then_original_fullgraph_encoder_VJP"
        value["efficient_same_objective_execution_binding"]=copy.deepcopy(binding)
        return value,trainer.json_sha(value)
    trainer._signature=signature
    if metadata_only:
        def expected_protocol_v1(old_protocol):
            cfg=dict(old_protocol["config"],execution_enabled=True)
            value,_=signature(cfg,copy.deepcopy(old_protocol["parent_plan_refs"]))
            for key in ("actual_TRAIN_execution_binding","observation_provider_execution_binding"):
                require(key in old_protocol and isinstance(old_protocol[key],dict),"Actual original execution binding")
                value[key]=copy.deepcopy(old_protocol[key])
                value["source_refs"].extend(copy.deepcopy(old_protocol[key]["source_refs"]))
            return value,trainer.json_sha(value)
        return expected_protocol_v1
    def bound_step(model,provider,targets,optimizer,**kwargs):
        return step.complete_parent_step_v1(model,provider,targets,optimizer,caps=copy.deepcopy(bounds["memory_caps"]),**kwargs)
    trainer.checkpointed_complete_parent_step_v1=bound_step
    progress_state=dict(last_step=None,receipts=0,start=None,last_write_seconds=0.)
    def progress_v1(candidate,result,writer):
        state=candidate["trainer_state"];count=state["global_step"]
        require(type(count)is int and 0<count<=600 and len(state["history"])==count
                and state["history"][-1]["global_step"]==count
                and result["whole_parent_optimizer_updates"]==1,"Completed fullparent update before telemetry")
        if progress_state["last_step"] is not None:
            require(count==progress_state["last_step"]+1,"Consecutive actual newly completed steps")
        values=candidate["optimizer_state"]["state"].values()
        steps=[int(v["step"].item()) for v in values]
        require(steps and all(x==count for x in steps),"Actual saved AdamW counters equal fullparent cursor")
        now=time.monotonic()
        row=dict(schema="C3_completed_optimizer_update_progress_v1",global_step=count,
            epoch=result["epoch"],parent_index=result["parent_index"],parent_id=result["parent_id"],
            actual_AdamW_step_min=min(steps),actual_AdamW_step_max=max(steps),records=10000,
            loss=result["loss"],gradient_norm=result["gradient_norm"],timings=result["timings"],
            elapsed_since_training_callable_seconds=now-progress_state["start"],
            previous_progress_write_seconds=progress_state["last_write_seconds"],
            full_graph_haplotypes=1600,recurrent_query_columns=1000,
            optimizer_update_completed=True,checkpoint_snapshot_held_in_RAM=True,
            durable_optimizer_checkpoint_asserted=False,source_binding=STEP_REF)
        require(len((json.dumps(row,sort_keys=True,allow_nan=False)+"\n").encode())<=2048,"Capped live stdout update")
        print(json.dumps(dict(C3_completed_update=row),sort_keys=True,allow_nan=False),flush=True)
        require(len((json.dumps(row,indent=2,sort_keys=True,allow_nan=False)+"\n").encode())<=8192,
                "Capped durable progress receipt")
        t0=time.monotonic();receipt=writer.save("progress_update_%06d.json"%count,row)
        progress_state["last_write_seconds"]=time.monotonic()-t0
        progress_state["last_step"]=count;progress_state["receipts"]+=1
        require(progress_state["receipts"]<=600,"At most600 new update receipts")
        print(json.dumps(dict(C3_progress_receipt=receipt,global_step=count,
                              original_guarded_write_seconds=progress_state["last_write_seconds"]),sort_keys=True),flush=True)
    trainer._efficient_progress_v1=progress_v1
    source=(ROOT/TRAINER_REF["path"]).read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="run_train_v1")
    body=ast.get_source_segment(source,node)
    needle="safe=candidate;step=global_index+1"
    require(body.count(needle)==1,"One operational after-safe-cursor progress hook")
    body=body.replace(needle,needle+"\n            _efficient_progress_v1(candidate,result,writer)")
    exec(compile(body,"<original_C3_full600_trainer_with_after_update_progress_hook>","exec"),trainer.__dict__)
    def run_train_v1(*args,**kwargs):
        require(read_bounds_v1(bounds_ref)==bounds and ref(STEP_REF["path"])==STEP_REF
                and ref(CACHE_REF["path"])==CACHE_REF and ref(TRAINER_REF["path"])==TRAINER_REF,
                "Exact separately admitted future source/caps family")
        require(progress_state["start"] is None,"One exact remaining-budget continuation invocation")
        require(kwargs.get("resume_ref")==manifest["migrated_boundary_ref"],
                "Only exact full-state migrated boundary may enter unchanged original resume validator")
        progress_state["start"]=time.monotonic()
        return trainer.run_train_v1(*args,**kwargs)
    run_train_v1.execution_binding=copy.deepcopy(binding)
    return run_train_v1

class HooksProxyV1:
    def __init__(self,original,bounds_ref,resume_manifest_ref):
        self.original,self.bounds_ref=original,copy.deepcopy(bounds_ref)
        self.resume_manifest_ref=copy.deepcopy(resume_manifest_ref)
        self.manifest,_=read_resume_manifest_v1(self.resume_manifest_ref,self.bounds_ref)
        self.callable=None
    def __getattr__(self,name):return getattr(self.original,name)
    def run_train_stage_v1(self,cfg,*,actual_train_bindings,**kwargs):
        require(self.callable is None,"One prospective full TRAIN stage")
        require(kwargs.get("resume_ref") is None,"Only the original explicitNone resume call is adapted")
        kwargs["resume_ref"]=copy.deepcopy(self.manifest["migrated_boundary_ref"])
        self.callable=make_training_callable_v1(self.bounds_ref,self.resume_manifest_ref)
        factory=self.original.make_actual_target_joiner_factory_v1(actual_train_bindings)
        return self.callable(cfg,train_target_joiner_factory=factory,**kwargs)

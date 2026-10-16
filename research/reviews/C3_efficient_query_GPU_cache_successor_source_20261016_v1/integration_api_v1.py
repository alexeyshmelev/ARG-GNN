"""Future guarded C3 training callable. Imports stdlib only until root invokes.
Private original trainer, with exact operational progress hook after safe cursor.
"""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1"
SOURCE=HERE+"/integration_api_v1.py"
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

def make_training_callable_v1(bounds_ref):
    bounds=read_bounds_v1(bounds_ref)
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
        old_family_resume_rejected=True,current_live_run_untouched=True)
    trainer.SOURCES=tuple(dict.fromkeys((*trainer.SOURCES,STEP_REF["path"],CACHE_REF["path"],SOURCE,bounds_ref["path"])))
    def signature(cfg,plans):
        value,_=original_signature(cfg,plans)
        value["exact_gradient"]="shared_device_query_context_adjoints_then_original_fullgraph_encoder_VJP"
        value["efficient_same_objective_execution_binding"]=copy.deepcopy(binding)
        return value,trainer.json_sha(value)
    trainer._signature=signature
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
        require(progress_state["start"] is None,"One fresh complete training invocation")
        progress_state["start"]=time.monotonic()
        return trainer.run_train_v1(*args,**kwargs)
    run_train_v1.execution_binding=copy.deepcopy(binding)
    return run_train_v1

class HooksProxyV1:
    def __init__(self,original,bounds_ref):
        self.original,self.bounds_ref=original,copy.deepcopy(bounds_ref)
        self.callable=None
    def __getattr__(self,name):return getattr(self.original,name)
    def run_train_stage_v1(self,cfg,*,actual_train_bindings,**kwargs):
        require(self.callable is None,"One prospective full TRAIN stage")
        self.callable=make_training_callable_v1(self.bounds_ref)
        factory=self.original.make_actual_target_joiner_factory_v1(actual_train_bindings)
        return self.callable(cfg,train_target_joiner_factory=factory,**kwargs)

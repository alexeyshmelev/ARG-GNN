"""One bounded CPU-only saved-state audit. No model, optimizer, data loader or project import."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, random, resource, stat, sys, time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/C3_completion600_readiness_audit_20261016_v1"
COMPLETION={"path":"experiments/results/C3_GNN_efficient_resume_GPU0_train_v1/completed_training.json","sha256":"a58c13c50d4358a189771c896afbdbd7b99aa56a195b3deffc3d116bb9235f61"}
MIGRATION={"path":"research/reviews/C3_actual_full_state_handover_root_20261016_v1/migration_v2.root_sealed_manifest_v1.json","sha256":"7cbd2387ea536bf49adb7af93477c229c6473b7a2f9e4f681583e8d97b466830"}
AUDIT={"path":"research/reviews/native_v9_controller_entry_failure_audit_20261016_v1/audit.v1.json","sha256":"b25487fa5480ac92faf33fd51d5a33fcc451c51a19e41a1e217eb5d0a6b6178d"}
EPOCHS=[5,10,15,20,25,30]
TRAIN=["american_c3_short_train_%03d"%i for i in range(20)]
CONDITION="fresh_C3_supervised_short_independent_complete_parents"
def js(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def identity(p):
    s=p.lstat()
    return {"device":s.st_dev,"inode":s.st_ino,"size":s.st_size,"mtime_ns":s.st_mtime_ns,"ctime_ns":s.st_ctime_ns}
def exact(r,cap):
    p=ROOT/r["path"]
    assert p.resolve()==p and p.is_relative_to(ROOT) and stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink()
    raw=p.read_bytes()
    assert len(raw)<=cap and sha(raw)==r["sha256"]
    return p,raw
def read(r):return json.loads(exact(r,4*2**20)[1])
def check(out,name,value):out[name]=bool(value)
def order(epoch):
    a=list(TRAIN);random.Random(70002+epoch).shuffle(a);return a
assert Path.cwd()==ROOT and sys.prefix==str(ROOT/".runtime/env") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
started=time.monotonic()
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
training=read(COMPLETION);migration=read(MIGRATION);prior=read(AUDIT)
source_ref={"path":str(Path(__file__).relative_to(ROOT)),"sha256":sha(Path(__file__).read_bytes())}
summary_checks={}
check(summary_checks,"completed600_scientific_metadata",training["status"]=="completed_C3_checkpointed_complete_parent_training" and training["schema"]=="american_c3_checkpointed_complete_parent_training_v1" and training["complete_parent_updates"]==600 and training["epochs"]==30 and training["focal_exposures"]==6000000 and training["trained_parent_ids"]==TRAIN and training["train_sequence_lengths_bp"]==[2000000]*20 and training["VAL_used_for_optimization"] is False and training["TEST_deferred"] is True and training["H043_extension"] is None and training["purity_filter"] is False)
check(summary_checks,"migration_lineage",training["resume_ref"]==migration["migrated_boundary_ref"] and migration["actual_global_step"]==100 and migration["actual_AdamW_step"]==100 and migration["remaining_updates"]==500 and migration["passed"] is True)
for key,mkey in [("protocol_sha256","successor_protocol_sha256"),("observation_seal_sha256","observation_seal_sha256"),("target_join_sha256","target_join_sha256")]:
    check(summary_checks,key+"_matches_migration",training[key]==migration[mkey])
candidates=training["checkpoint_candidates"]
assert [x["epoch"] for x in candidates]==EPOCHS and [x["global_step"] for x in candidates]==[20*e for e in EPOCHS]
assert len({x["checkpoint_ref"]["path"] for x in candidates})==6
contract={"schema":"american_c3_fresh_whole_tree_GNN_checkpoint_v1","class_order":["EUR","EAS","AFR"],"hidden_dim":32,"donor_cohorts":2,"donor_channels":6,"expected_haplotypes":1600,"initialization":"fresh_no_C4_loading","site_feature_width":15,"observed_leaf_width":5,"recurrent_contract":"one_layer_bidirectional_equations_as_two_directional_GRUs","tree_backbone":"unchanged_PyG_gated_upward_recipient_excluded_downward_cavity","whole_parent_training_required":True}
rows=[];signature=None
for i,c in enumerate(candidates):
    p,raw=exact(c["checkpoint_ref"],2*2**20);before=identity(p)
    v=torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    s=v["trainer_state"];step=c["global_step"];epoch=c["epoch"];checks={}
    check(checks,"checkpoint_family",v["schema"]=="american_c3_fresh_whole_tree_GNN_checkpoint_v1" and v["contract"]==contract and v["model_seed"]==70001 and v["training_condition"]==CONDITION and v["C4_initialization_or_head_slicing"] is False and v["trained_parent_ids"]==TRAIN and v["train_sequence_lengths_bp"]==[2000000]*20)
    check(checks,"scheduled_candidate",v["candidate"]==dict(eligible_for_VAL_selection=True,kind="scheduled_epoch",completed_epoch=epoch,global_step=step))
    check(checks,"cursor_exposure_scheduler",s["schema"]==training["schema"] and s["global_step"]==step and s["focal_exposures"]==step*10000 and s["next_epoch"]==step//20+1 and s["next_parent_index"]==0 and s["next_parent_order"]==(order(step//20+1) if step<600 else []) and s["scheduler"]==dict(name="constant_learning_rate",completed_parent_updates=step,lr=.001))
    expected=[(e,j,pid) for e in range(1,31) for j,pid in enumerate(order(e))][:step]
    history=s["history"]
    check(checks,"exact_complete_parent_history",len(history)==step and all(r["global_step"]==n+1 and (r["epoch"],r["parent_index"],r["parent_id"])==e and r["records"]==10000 and r["whole_parent_optimizer_updates"]==1 and r["block_reset_or_truncated_BPTT"] is False for n,(r,e) in enumerate(zip(history,expected))))
    check(checks,"history_is_prefix_of_completion",history==training["history"][:step])
    check(checks,"prior_scheduled_chain",s["scheduled_checkpoints"]==candidates[:i])
    check(checks,"protocol_self_hash_and_lineage",sha(js(s["protocol"]))==s["protocol_sha256"]==training["protocol_sha256"] and s["observation_seal_sha256"]==training["observation_seal_sha256"] and s["target_join_sha256"]==training["target_join_sha256"])
    signature=s["protocol"] if signature is None else signature
    check(checks,"same_protocol_all_candidates",s["protocol"]==signature)
    if i==0:check(checks,"epoch5_exact_migrated_boundary",c["checkpoint_ref"]==migration["migrated_boundary_ref"] and c["checkpoint_ref"]==migration["migrated_checkpoint_receipts"][0]["migrated_ref"])
    else:check(checks,"exact_continuation_namespace",c["checkpoint_ref"]["path"]=="experiments/results/C3_GNN_efficient_resume_GPU0_train_v1/checkpoint_epoch_%03d.pt"%epoch)
    models=v["model_state"];optim=v["optimizer_state"];states=optim["state"];groups=optim["param_groups"]
    model_rows=[{"name":k,"shape":list(t.shape),"dtype":str(t.dtype),"numel":t.numel(),"sha256":sha(t.contiguous().numpy().tobytes())} for k,t in models.items()]
    model_count=sum(t.numel() for t in models.values())
    check(checks,"model_finite_fp32_cpu",bool(models) and all(t.device.type=="cpu" and t.dtype==torch.float32 and bool(torch.isfinite(t).all()) for t in models.values()) and model_count==29571)
    check(checks,"full_optimizer_mapping",len(groups)==1 and len(states)==len(models)==len(groups[0]["params"]) and set(states)==set(groups[0]["params"]))
    steps=[int(x["step"].item()) for x in states.values()]
    check(checks,"actual_saved_AdamW_steps",bool(steps) and all(x==step for x in steps))
    g=groups[0]
    check(checks,"AdamW_hyperparameters",g["lr"]==.001 and g["weight_decay"]==.0001 and tuple(g["betas"])==(.9,.999) and g["eps"]==1e-8 and all(g.get(k,False) is False for k in ("amsgrad","maximize","foreach","capturable","differentiable","fused")))
    check(checks,"finite_full_optimizer_moments",all(set(x)=={"step","exp_avg","exp_avg_sq"} and x["exp_avg"].shape==t.shape and x["exp_avg_sq"].shape==t.shape and all(z.device.type=="cpu" and bool(torch.isfinite(z).all()) for z in x.values()) for x,t in zip((states[k] for k in g["params"]),models.values())))
    rng=s["rng_state"]
    check(checks,"RNG_state_saved_without_restore",set(rng)=={"python","numpy","torch_CPU","torch_device","device_type"} and rng["device_type"]=="cuda" and torch.is_tensor(rng["torch_CPU"]) and torch.is_tensor(rng["torch_device"]) and rng["torch_CPU"].device.type=="cpu" and rng["torch_device"].device.type=="cpu" and bool(rng["python"]) and rng["numpy"]["algorithm"]=="MT19937" and len(rng["numpy"]["keys"])==624)
    check(checks,"stable_exact_file",identity(p)==before and sha(p.read_bytes())==c["checkpoint_ref"]["sha256"])
    rows.append(dict(checkpoint_ref=c["checkpoint_ref"],epoch=epoch,global_step=step,bytes=len(raw),identity=before,all_checks_passed=all(checks.values()),checks=checks,actual_AdamW_step_min=min(steps),actual_AdamW_step_max=max(steps),optimizer_parameter_states=len(states),model_tensor_count=len(models),model_parameter_count=model_count,model_tensor_manifest_sha256=sha(js(model_rows)),history_rows=len(history),history_sha256=sha(js(history)),protocol_sha256=s["protocol_sha256"],observation_seal_sha256=s["observation_seal_sha256"],target_join_sha256=s["target_join_sha256"],prior_scheduled_checkpoint_count=len(s["scheduled_checkpoints"])))
    del v,s,models,optim,states,groups,model_rows
missing={}
for name in ("completed.json","failed.json","actual_wait.json","observation_prefetch_all_worker_drains.json"):
    missing[name]={"exists_at_audit":(ROOT/"experiments/results/C3_GNN_efficient_resume_GPU0_train_v1"/name).exists()}
check(summary_checks,"all_six_saved_candidates_pass",len(rows)==6 and all(r["all_checks_passed"] for r in rows))
check(summary_checks,"CUDA_uninitialized_after_audit",not torch.cuda.is_initialized())
leader=prior["C3_recorded_terminal_lineage"]
result=dict(schema="C3_completion600_bounded_saved_tensor_readiness_audit_v1",status="VERIFIED_SAVED600_ARTIFACTS_ORIGINAL_PRODUCER_FAILED" if all(summary_checks.values()) else "SAVED_STATE_AUDIT_CHECK_FAILURE",sealed_utc=datetime.now(timezone.utc).isoformat(),audit_source_ref=source_ref,training_completion_ref=COMPLETION,migration_manifest_ref=MIGRATION,original_failure_audit_ref=AUDIT,summary_checks=summary_checks,checkpoints=rows,checkpoint_bytes_read=sum(x["bytes"] for x in rows),original_terminal_lineage={"controller_row_state":leader["state"],"leader_exit_code":leader["exit_code"],"leader_wait_status_known":leader["leader_wait_status_known"],"controller_owned_drain_observed":leader["owned_drain_observed"],"outer_74136_terminal_consumed":None,"outer_74136_actual_exit_code":None,"outer_unknown_is_parent_report_not_tool_observed_here":True},original_terminal_files=missing,source_mismatch={"expected_releases_resumed_source_derived":520,"original_required_releases":620,"observed_release_count":None,"actual_terminal_exception_text":None,"observed_cause_asserted":False},scope={"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch_version":str(torch.__version__),"checkpoint_load":"weights_only=True,map_location=cpu,mmap=True","Torch_CPU_threads":1,"model_instantiated":False,"optimizer_instantiated_or_updated":False,"RNG_restored":False,"training_or_prediction":False,"VAL_or_TEST_truth_or_predictions_opened":False,"project_source_imported":False,"live_sources_guards_authority_registry_receipts_modified":False,"acceptance_or_execution_authority":False,"storage_mutation_or_credit":False,"physical_work_exactly600_claimed":False},elapsed_seconds=time.monotonic()-started,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
out=HERE/"checkpoint_audit.v1.json"
with out.open("x") as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status=result["status"],audit_ref=dict(path=str(out.relative_to(ROOT)),sha256=sha(out.read_bytes())),all_summary_checks_pass=all(summary_checks.values()),checkpoint_count=len(rows),checkpoint_bytes_read=result["checkpoint_bytes_read"],saved_steps=[r["actual_AdamW_step_min"] for r in rows],optimizer_parameter_states=[r["optimizer_parameter_states"] for r in rows],failed_checks={str(r["epoch"]):[k for k,v in r["checks"].items() if not v] for r in rows if not r["all_checks_passed"]},CUDA_initialized=torch.cuda.is_initialized(),elapsed_seconds=result["elapsed_seconds"])))
raise SystemExit(0 if all(summary_checks.values()) else 1)

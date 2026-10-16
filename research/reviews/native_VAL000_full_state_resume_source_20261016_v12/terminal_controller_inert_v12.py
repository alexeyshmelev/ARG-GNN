"""No real resources: source extraction and explicit fake process/lease/GPU/authority/checkpoint metadata only."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,stat,types,traceback,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OLD="research/reviews/native_VAL000_full_state_resume_source_20261016_v10"
NEW="research/reviews/native_VAL000_full_state_resume_source_20261016_v12"
results=[]
def require(v,m):
    if not v:raise ValueError(m)
def good(n,v=True):
    assert v,n
    results.append(dict(name=n,status="PASS"))
def rejects(n,fn):
    try:fn()
    except (ValueError,RuntimeError,FileNotFoundError,OSError,KeyError):results.append(dict(name=n,status="PASS_REJECTED"));return
    raise AssertionError("Invalid case accepted: "+n)
def read(path):return (ROOT/path).read_text()
def j(path):return json.loads(read(path))
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def fx(label):return dict(path="IN_MEMORY_FIXTURE/"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
def extract(path,names,ns):
    nodes=[n for n in ast.parse(read(path)).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<native_v11_source_only_fakes>","exec"),ns)
def fmap(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(read(path)).body if isinstance(n,ast.FunctionDef)}
def constants(path):
    ns={}
    for n in ast.parse(read(path)).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:ns[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    return ns
def run():
    names=("__init__.py","amendment_contract_v1.py","direct_gpu0_v1.py","execution_only_v1.py","corpus_metadata_bridge_v1.py","resume_scientific_bridge_v5.py","alias_namespace_v7.py","fresh_entry_v9.py")
    for n in names:ast.parse(read(NEW+"/"+n))
    good("all8_executed_sources_AST_parse")
    for n in names:
        if n not in ("amendment_contract_v1.py","direct_gpu0_v1.py"):
            good(n+"_namespace_only",read(NEW+"/"+n)==read(OLD+"/"+n).replace(OLD,NEW).replace(OLD.replace("/","."),NEW.replace("/",".")))
    good("resume_policy_byte_identical",(ROOT/NEW/"resume_policy.INACTIVE.v5.json").read_bytes()==(ROOT/OLD/"resume_policy.INACTIVE.v5.json").read_bytes())
    old=fmap(OLD+"/direct_gpu0_v1.py");new=fmap(NEW+"/direct_gpu0_v1.py")
    good("capacity_math_and_every_other_direct_body_unchanged",all(new[k]==old[k] for k in old if k not in ("controller_v1","_registry_v1")))
    ot=ast.parse(read(OLD+"/amendment_contract_v1.py"));nt=ast.parse(read(NEW+"/amendment_contract_v1.py"))
    oldhist=copy.deepcopy(next(n for n in ot.body if isinstance(n,ast.FunctionDef) and n.name=="validate_resume_prerequisites_v5"))
    newhist=next(n for n in nt.body if isinstance(n,ast.FunctionDef) and n.name=="validate_historical6500_resume_prerequisites_v11")
    oldhist.name=newhist.name
    good("historical6500_prerequisite_body_unchanged",ast.dump(oldhist,include_attributes=False)==ast.dump(newhist,include_attributes=False))
    cn=constants(NEW+"/amendment_contract_v1.py")
    dn=constants(NEW+"/direct_gpu0_v1.py")
    cap=j("research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json")
    audit=j(cn["V9_CONTROLLER_AUDIT_REF"]["path"])
    saved=j(cn["V10_ATTACHMENT_REF"]["path"])
    registry=copy.deepcopy(j(NEW+"/diagnosis.v11.json")["contemporaneous_registry_metadata_snapshot_not_exact_failure_RPC_payload"])
    now=1792177000.
    registry["updated_unix"]=now-7200.
    metadata=saved["physical_lease_metadata"]
    data=(json.dumps(metadata,sort_keys=True,separators=(",",":"))+"\n").encode()
    saved["physical_lease_sha256"]=hashlib.sha256(data).hexdigest()
    registry["lease_sha256"]=saved["physical_lease_sha256"]
    fixed_ref=dn["CURRENT_CAP"]
    q=dict(job_id=saved["job_id"],caps=dict(max_cuda_reserved_bytes=32*2**30),expires_at_utc="2026-10-18T23:41:32.476768+00:00",direct_attachment_authority_ref=fx("direct_authority"))
    authority=dict(schema="native32_fixed_GPU0_direct_attachment_authority_v1",root_authorizer="root_agent",status="active",borrowed_physical_inode_attachment_authorized=True,additional_old_controller_submissions_revoked_while_native32_attached=True,old_controller_preserved_until_native32_drained=True,controller_pid=3349405,controller_start_ticks=481281638,controller_capability_ref=fixed_ref,native32_job_id=q["job_id"],C3_fixed_envelope_bytes=26*2**30,native32_max_cuda_reserved_bytes=32*2**30,native32_non_Torch_margin_bytes=2*2**30,headroom_bytes=4*2**30)
    docs={fixed_ref["path"]:cap,cap["controller_storage_ref"]["path"]:dict(output_directory="FAKE_REGISTRY_OUT"),cn["V9_CONTROLLER_AUDIT_REF"]["path"]:audit,cn["V10_ATTACHMENT_REF"]["path"]:saved,q["direct_attachment_authority_ref"]["path"]:authority}
    reg_baseline=copy.deepcopy(registry);saved_baseline=copy.deepcopy(saved)
    meter=dict(held=True,lease=data,inode=saved["physical_inode"],dev=saved["physical_device"],nlink=1,controller_start=481281638,controller_state="R",calls=0,opens=0,closes=0,inventory=0)
    class File:
        def __enter__(self):return self
        def __exit__(self,*a):return False
    class P:
        def __init__(self,key):self.key=key
        def __str__(self):return self.key
        def is_file(self):return True
        def is_symlink(self):return False
        def stat(self):return types.SimpleNamespace(st_size=len(json.dumps(registry).encode()))
        def read_bytes(self):return json.dumps(registry).encode()
        def open(self,*a,**k):return File()
        def lstat(self):return types.SimpleNamespace(st_mode=stat.S_IFREG,st_nlink=meter["nlink"],st_dev=meter["dev"],st_ino=meter["inode"],st_size=len(meter["lease"]))
    def fake_open(path,flags):meter["opens"]+=1;return 17
    def fake_close(fd):meter["closes"]+=1
    def flock(file,flags):
        if flags==3 and meter["held"]:raise BlockingIOError("fake held")
    fakeos=types.SimpleNamespace(getuid=lambda:1000,open=fake_open,close=fake_close,O_RDWR=2,O_NOFOLLOW=4,fstat=lambda fd:P("fake").lstat(),pread=lambda fd,n,o:meter["lease"])
    lease_path=str(ROOT/".runtime/locks/gpu_0.lock")
    def local(path):
        return P(lease_path if path==".runtime/locks/gpu_0.lock" else "FAKE_REGISTRY")
    controller_identity=lambda:dict(pid=3349405,uid=1000,start_ticks=meter["controller_start"],state=meter["controller_state"],pgid=3349405,sid=3349405)
    def proc(pid):
        meter["calls"]+=1
        if pid==3349405:return controller_identity()
        if pid==3703734:return dict(pid=pid,uid=1000,start_ticks=484791257,state="R",pgid=pid,sid=pid)
        raise FileNotFoundError("No synthetic stale C3 identity")
    gpu=dict(index=0,uuid="GPU-ff06ca1f-403a-f2d0-b584-71e177452667",used_mib=24959,total_mib=81920,utilization=93,compute=[dict(pid=3703734,name="AI Auto Research Pipeline Test",used_mib=24948)])
    base=types.SimpleNamespace(proc_identity=proc,real_inventory=lambda:[copy.deepcopy(gpu)])
    ns=dict(cn);ns.update(dn);ns.update(require=require,read=lambda r:copy.deepcopy(docs[r["path"]]),active=lambda a:None,active_amendment_v1=lambda:None,local=local,ROOT=ROOT,datetime=datetime,stat=stat,json=json,hashlib=hashlib,os=fakeos,fcntl=types.SimpleNamespace(flock=flock,LOCK_EX=1,LOCK_NB=2,LOCK_UN=4),time=types.SimpleNamespace(time=lambda:now),GPU0_UUID=gpu["uuid"],CAPACITY_EVIDENCE_SINK=lambda sample:None,LAST_CONTROLLER_LIVENESS=None)
    extract(NEW+"/direct_gpu0_v1.py",{"_registry_v1","held_writable_probe_v1","live_terminal_lease_v11","controller_v1","capacity_v1","bounded_capacity_sample_v5"},ns)
    ns["_fresh_status_v1"]=lambda *a:(_ for _ in ()).throw(AssertionError("No status RPC permitted"))
    got=ns["controller_v1"](q,base)
    proof=ns["LAST_CONTROLLER_LIVENESS"]
    good("two_hour_old_terminal_registry_keeps_literal_timestamp_without_RPC",got[1]["updated_unix"]==now-7200 and proof["registry_updated_unix"]==now-7200 and proof["registry_age_seconds"]==7200 and proof["registry_freshness_claimed"] is False and proof["controller_reply_requested"] is False and proof["current_kernel_and_lease"]["held_writable_probe_passed"] is True)
    good("every_kernel_lease_descriptor_closed",meter["opens"]==meter["closes"])
    for field,value in [("lifecycle","draining"),("safety_error","error"),("ambiguous_descendants_or_lease_holders",[1]),("capability_ref",fx("wrongcap"))]:
        registry=copy.deepcopy(reg_baseline);registry[field]=value
        rejects("snapshot_"+field,lambda:ns["controller_v1"](q,base))
    registry=copy.deepcopy(reg_baseline);registry["updated_unix"]=now+6
    rejects("future_snapshot_timestamp",lambda:ns["controller_v1"](q,base))
    c3_index=next(i for i,row in enumerate(reg_baseline["jobs"]) if row["logical_job"]==dn["C3_JOB"])
    for field,value in [("state","running"),("state","succeeded"),("exit_code",0),("owned_drain_observed",False),("leader_wait_status_known",False),("finished_unix",0)]:
        registry=copy.deepcopy(reg_baseline);registry["jobs"][c3_index][field]=value
        rejects("terminal_C3_"+field+"_"+str(len(results)),lambda:ns["controller_v1"](q,base))
    for state in ("queued","launching","running","draining","failed"):
        registry=copy.deepcopy(reg_baseline);registry["jobs"].append(dict(logical_job="new_work",state=state))
        rejects("any_new_controller_work_"+state,lambda:ns["controller_v1"](q,base))
    for field,value in [("inode",1),("dev",1),("nlink",2),("held",False),("lease",b"changed"),("controller_start",1),("controller_state","Z")]:
        registry=copy.deepcopy(reg_baseline);before=meter[field];meter[field]=value
        rejects("current_kernel_or_exact_lease_"+field,lambda:ns["controller_v1"](q,base))
        meter[field]=before
    good("all_error_lease_descriptors_closed",meter["opens"]==meter["closes"])
    registry=copy.deepcopy(reg_baseline)
    sample=ns["capacity_v1"](q,base,worker_pid=3703734)
    good("fresh_capacity_remains_true_at93percentGPU",sample["capacity_fit"] is True and sample["GPU0"]["utilization"]==93 and sample["native32_observed_mib"]==24948 and sample["C3_fixed_envelope_mib"]==26624 and sample["headroom_mib"]==4096)
    good("capacity_records_literal_terminal_snapshot",sample["registry_updated_unix"]==now-7200 and sample["actual_controller_liveness_evidence"]["registry_freshness_claimed"] is False)
    oldgpu=copy.deepcopy(gpu)
    gpu["compute"][0]["used_mib"]=34817;gpu["used_mib"]=34828
    rejects("native32plus2_actual_memory_overrun",lambda:ns["capacity_v1"](q,base,worker_pid=3703734))
    gpu=copy.deepcopy(oldgpu);gpu["compute"][0]["name"]="wrong_title"
    rejects("native_exact_process_title",lambda:ns["capacity_v1"](q,base,worker_pid=3703734))
    gpu=copy.deepcopy(oldgpu);gpu["compute"].append(dict(pid=999,name="external",used_mib=20000));gpu["used_mib"]+=20000
    rejects("external_and_unfilled_envelopes_headroom",lambda:ns["capacity_v1"](q,base,worker_pid=3703734))
    gpu=copy.deepcopy(oldgpu)
    # Actual compact preserved metadata is read as source only, never a payload.
    mpath=NEW+"/amendment_contract_v1.py"
    old_amend=j(cn["V10_AMENDMENT_REF"]["path"]);actualwait=j(cn["V10_WAIT_REF"]["path"]);inner=j(cn["V10_SUBPROCESS_TERMINAL_REF"]["path"]);outer=j(cn["V10_OUTER_TERMINAL_REF"]["path"])
    directive_ref=fx("future_ROOT_directive_test_only");manifest_ref=fx("candidate_manifest_test_only")
    directive=dict(root_authorizer="root_agent",template_only=False,source_manifest_ref=manifest_ref,controller_capability_ref=cn["CONTROLLER_CAPABILITY_REF"],controller_pid=3349405,controller_start_ticks=481281638,terminal_failed_C3_historical_snapshot_authorized=True,all_old_controller_submissions_permanently_revoked=True,v10_future_launch_write_capabilities_permanently_revoked=True)
    authority_args=dict(terminal_failed_C3_historical_snapshot_authorized=True,all_old_controller_submissions_permanently_revoked=True,v10_future_launch_write_capabilities_permanently_revoked=True,preserved_actual_v10_amendment_ref=cn["V10_AMENDMENT_REF"],genuine_v10_direct_wait_ref=cn["V10_WAIT_REF"],genuine_v10_subprocess_terminal_ref=cn["V10_SUBPROCESS_TERMINAL_REF"],genuine_v10_consumed_outer_terminal_ref=cn["V10_OUTER_TERMINAL_REF"],ROOT_directive_ref=directive_ref,additional_known_old_ROOT_launcher_source_refs=[cn["V10_WAIT_REF"],cn["V10_LAUNCH_REF"]])
    adocs={cn["V10_AMENDMENT_REF"]["path"]:old_amend,cn["V10_WAIT_REF"]["path"]:actualwait,cn["V10_SUBPROCESS_TERMINAL_REF"]["path"]:inner,cn["V10_OUTER_TERMINAL_REF"]["path"]:outer,directive_ref["path"]:directive}
    baseline_adocs=copy.deepcopy(adocs)
    an=dict(cn,require=require,read=lambda r:copy.deepcopy(adocs[r["path"]]),ref=lambda p:manifest_ref,MANIFEST="candidate")
    extract(mpath,{"terminal_snapshot_authority_v11"},an)
    good("actual_v10_direct_minus15_inner241_outer10747_241_are_distinct",an["terminal_snapshot_authority_v11"](authority_args))
    for key in ("terminal_failed_C3_historical_snapshot_authorized","all_old_controller_submissions_permanently_revoked","v10_future_launch_write_capabilities_permanently_revoked"):
        a=copy.deepcopy(authority_args);a[key]=False
        rejects("explicit_ROOT_amendment_"+key,lambda:an["terminal_snapshot_authority_v11"](a))
        adocs=copy.deepcopy(baseline_adocs);adocs[directive_ref["path"]][key]=False
        rejects("explicit_ROOT_directive_"+key,lambda:an["terminal_snapshot_authority_v11"](authority_args))
    for field,value in [("actual_terminal_consumed",False),("actual_tool_session_id",1),("actual_direct_worker_returncode",0),("native_v10_attempt_status","SUCCESS"),("completed_ref",fx("invented_completion"))]:
        adocs=copy.deepcopy(baseline_adocs);adocs[cn["V10_OUTER_TERMINAL_REF"]["path"]][field]=value
        rejects("genuine_separate_outer_"+field,lambda:an["terminal_snapshot_authority_v11"](authority_args))
    adocs=copy.deepcopy(baseline_adocs);a=copy.deepcopy(authority_args);a["additional_known_old_ROOT_launcher_source_refs"]=[]
    rejects("v10_known_owner_roster_mandatory_for_fresh_entry",lambda:an["terminal_snapshot_authority_v11"](a))
    # Current22000 state selection uses the genuine ROOT audit; generic strict FD/
    # full model/optimizer/scheduler/RNG restore implementation is byte-identical.
    actual_progress=j(cn["V11_SAVED22000_AUDIT_REF"]["path"])
    pdocs={cn["V11_SAVED22000_AUDIT_REF"]["path"]:actual_progress}
    checkpoint_stat=types.SimpleNamespace(st_mode=stat.S_IFREG,st_nlink=1,st_size=2017943366,st_dev=67,st_ino=123,st_mtime_ns=1,st_ctime_ns=1)
    fakepath=types.SimpleNamespace(lstat=lambda:checkpoint_stat)
    current_args=dict(ROOT_resume_qualification_ref=cn["V11_SAVED22000_AUDIT_REF"],resume_checkpoint_ref=cn["V11_RESUME_CHECKPOINT_REF"],resume_exact_verified22000_full_state_authorized=True)
    historical_calls=[]
    def historical(a):
        historical_calls.append(a)
        assert a["resume_checkpoint_ref"]==cn["RESUME_CHECKPOINT_REF"] and a["ROOT_resume_qualification_ref"]==cn["V8_QUALIFICATION_REF"]
        return dict(checkpoint_ref=cn["RESUME_CHECKPOINT_REF"])
    pn=dict(cn,copy=copy,require=require,read=lambda r:copy.deepcopy(pdocs[r["path"]]),raw=lambda r:None,local=lambda p:fakepath,stat=stat,validate_historical6500_resume_prerequisites_v11=historical,RESUME_STATE=None)
    extract(mpath,{"validate_resume_prerequisites_v5"},pn)
    selected=pn["validate_resume_prerequisites_v5"](current_args)
    good("exact_current22000_selected_without6500_replay",selected["checkpoint_ref"]==cn["V11_RESUME_CHECKPOINT_REF"] and selected["actual_saved_progress"]==dict(epoch=0,batch=22000,batch_in_epoch=22000,optimizer_unique_steps=[22000]) and selected["earned_updates_beyond6500"]==15500 and selected["post_save_extra_updates"]=="UNKNOWN")
    a=copy.deepcopy(current_args);a["resume_checkpoint_ref"]=cn["RESUME_CHECKPOINT_REF"]
    rejects("6500_current_load_rejected",lambda:pn["validate_resume_prerequisites_v5"](a))
    a=copy.deepcopy(current_args);a["resume_exact_verified22000_full_state_authorized"]=False
    rejects("unauthorized22000_load_rejected",lambda:pn["validate_resume_prerequisites_v5"](a))
    for field,value in [("actual_saved_optimizer_updates",21999),("scheduler_callbacks_RNG_present",False),("full_byte_hash_and_FD_mmap",False),("pre_post_named_and_FD_identity_equal",False),("full_budget_complete",True),("live_updates_after_saved_checkpoint_unknown",False),("checkpoint_ref",fx("other_checkpoint"))]:
        pdocs={cn["V11_SAVED22000_AUDIT_REF"]["path"]:copy.deepcopy(actual_progress)};pdocs[cn["V11_SAVED22000_AUDIT_REF"]["path"]][field]=value
        rejects("sealed_saved22000_"+field,lambda:pn["validate_resume_prerequisites_v5"](current_args))
    for field,value in [("parameter_state_count",76),("unique_step_values",[22000,22001]),("minimum_step",21999),("maximum_step",22001)]:
        pdocs={cn["V11_SAVED22000_AUDIT_REF"]["path"]:copy.deepcopy(actual_progress)};pdocs[cn["V11_SAVED22000_AUDIT_REF"]["path"]]["actual_saved_optimizer_counters"][field]=value
        rejects("saved22000_77AdamW_"+field,lambda:pn["validate_resume_prerequisites_v5"](current_args))
    for field,value in [("epoch",1),("batch",22001),("batch_in_epoch",21999)]:
        pdocs={cn["V11_SAVED22000_AUDIT_REF"]["path"]:copy.deepcopy(actual_progress)};pdocs[cn["V11_SAVED22000_AUDIT_REF"]["path"]]["Composer_timestamp"][field]=value
        rejects("saved22000_timestamp_"+field,lambda:pn["validate_resume_prerequisites_v5"](current_args))
    good("no_actual_scientific_modules_imported",not any(n in sys.modules for n in ("torch","torch_geometric","composer","numpy","tskit")))
    return dict(schema="native_v11_terminal_controller_and_verified22000_inert_checks_v1",status="PASS_SOFTWARE_ONLY",checks=results,check_count=len(results),exact_original6500_preserved=True,current_saved22000_checkpoint_ref=cn["V11_RESUME_CHECKPOINT_REF"],current_saved22000_audit_ref=cn["V11_SAVED22000_AUDIT_REF"],actual_payload_read_or_hash=False,actual_resource_process_lease_GPU_or_controller_RPC_probe=False,actual_guard_accounting_authority_admission_execution=False,source_only=True,finished_utc=datetime.now(timezone.utc).isoformat())
def save(name,obj):
    with (ROOT/NEW/name).open("x") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n")
if __name__=="__main__":
    try:
        out=run();save("terminal_controller_inert.v11.json",out);print(json.dumps(dict(status=out["status"],checks=out["check_count"]),indent=2))
    except BaseException as e:save("terminal_controller_inert.attempt1.FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc(),completed_checks=results));raise

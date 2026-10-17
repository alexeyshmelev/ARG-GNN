"""Inactive post-drain two-slot contract. No runtime work on import."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,stat,time
from . import common_v1 as h
ROOT=h.ROOT;HERE=h.HERE;SOURCE=HERE+"/pool_contract_v1.py";POOL=HERE+"/pool_v1.py"
UUID=h.GPU0_UUID
GPU_POLICY={"path":"research/reviews/GPU0_only_user_transition_root_20261016_v1/GPU0_only_policy.root_active_v1.json","sha256":"bba84cecf416b2d284126bd31b72cff26ef9c46ee24dae375721d3f57240ed24"}
SHARED={"path":"research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_shared_pool_v1.py","sha256":"28bb690f33c2384d04f88dbacff1f910694c398c81874f9b9751336f4fb6429c"}
OLD={"path":"src/arggnn/gpu45_c3cache_h050_pool_v1.py","sha256":"d3279c6208737371031330f4bef9325199ba20c86f0f7d8e930c134b128d29fb"}
DOWN={"path":"research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py","sha256":"07d930aba589c16be8bd8c97a7948e519e108def97bc7538bb061bf634c6a5ce"}
OLD_CAP={"path":"research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json","sha256":"4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870"}
require=h.require;read=h.read;ref=h.ref;raw=h.raw;local=h.local
SLOTS={"H058":dict(cuda_cap_bytes=28*2**30,margin_mib=2048,peak_mib=30720,cpu_rss_cap_bytes=128*2**30,operational_deadline_seconds=86400),"native":dict(cuda_cap_bytes=32*2**30,margin_mib=2048,peak_mib=34816,cpu_rss_cap_bytes=384*2**30,operational_deadline_seconds=172800)}

def reviewed_v1(record):
    r=read(record)
    require(r.get("independent") is True and r.get("verdict")=="SOURCE_PASS" and r.get("blockers")==[]
        and r.get("source_ref")==ref(h.DRIVER) and r.get("source_manifest_ref")==ref(h.MANIFEST)
        and r.get("pool_source_ref")==ref(POOL) and r.get("pool_contract_ref")==ref(SOURCE),"Fresh independent whole pool/H058 join review")
    return r

def validate_handover_v1(record,H058_job,*,fresh=False):
    v=read(record)
    require(v.get("schema")=="GPU0_complete_old_borrowers_closed_handover_v1" and v.get("root_authorizer")=="root_agent"
        and v.get("old_controller_capability_ref")==OLD_CAP and v.get("old_controller_identity")==dict(pid=3349405,start_ticks=481281638,uid=os.getuid())
        and v.get("complete_old_borrower_authority_inventory") is True and v.get("all_old_borrowers_drained") is True
        and v.get("borrower_inventory_scope")=="all_current_or_still_authorized_attachments_at_handover"
        and v.get("historical_unknown_descendants_are_not_certified_as_drained") is True
        and v.get("all_old_controller_and_borrower_launch_resume_write_routes_permanently_revoked") is True
        and v.get("old_files_charges_clocks_failures_unknowns_preserved") is True
        and v.get("old_C3_compute_reservation_ended_only_after_owned_absence") is True
        and v.get("historical_C3_failed1_and_outer74136UNKNOWN_preserved") is True
        and v.get("no_live_ownership_adoption_or_migration") is True,"Honest complete post-drain ROOT handover")
    final=read(v["old_controller_final_registry_ref"]);terminal=read(v["old_controller_actual_outer_terminal_ref"])
    require(final.get("controller",{}).get("pid")==3349405 and final["controller"]["start_ticks"]==481281638
        and final.get("capability_ref")==OLD_CAP and final.get("lifecycle")=="closed"
        and final.get("ambiguous_descendants_or_lease_holders")==[] and final.get("safety_error") is None
        and all(j.get("state")in("failed","succeeded","cancelled") for j in final["jobs"])
        and all(j.get("pid") is None or(j.get("leader_wait_status_known") is True and j.get("owned_drain_observed") is True)for j in final["jobs"]),"Actual closed old registry with every started logical wait/drain")
    require(terminal.get("root_authorizer")=="root_agent" and terminal.get("actual_terminal_consumed") is True
        and terminal.get("actual_exit_code")==0 and terminal.get("controller_identity")==v["old_controller_identity"]
        and terminal.get("final_registry_ref")==v["old_controller_final_registry_ref"],"Genuine consumed controller outer0; no historical scientific wait relabel")
    rows=v.get("borrowers");require(isinstance(rows,list)and rows,"Complete separate borrower dispositions")
    identities=[v["old_controller_identity"]]
    for row in rows:
        raw(row["authority_ref"]);w=read(row["actual_wait_ref"]);o=read(row["actual_outer_terminal_ref"])
        require(w.get("actual_direct_worker_wait") is True and type(w.get("returncode")) is int
            and w.get("all_owned_processes_and_groups_drained") is True and w.get("complete_owned_roster_known",True) is True
            and o.get("actual_terminal_consumed") is True and type(o.get("actual_exit_code")) is int
            and o.get("actual_wait_ref")==row["actual_wait_ref"] and row.get("future_routes_revoked") is True
            and row.get("logical_lock_released") is True,"Preserved genuine original borrower wait/outcome/drain/revocation")
        require(isinstance(row.get("process_identities"),list)and row["process_identities"],"Complete exact original borrower identities")
        identities.extend(row["process_identities"])
    native=h.validate_native_noncoexistence_v3(dict(job_id=H058_job,native_noncoexistence_qualification_ref=v["completed_current_native_success_ref"]),fresh=fresh)
    expected={(x["pid"],x["start_ticks"],x["uid"])for x in identities}
    require({(x["pid"],x["start_ticks"],x["uid"])for x in native["native_process_identities"]}<=expected,"Current completed native included among drained borrowers")
    absence=read(v["targeted_old_owner_absence_ref"])
    require(absence.get("root_authorizer")=="root_agent"and absence.get("all_listed_owned_roots_groups_contexts_absent") is True
        and absence.get("old_controller_identity")==v["old_controller_identity"]
        and absence.get("complete_identity_list")==identities,"Targeted complete old owner/borrower absence, never external absence")
    lease=v["old_physical_lease"]
    require(lease.get("metadata")==final.get("lease_metadata") and lease.get("sha256")==final.get("lease_sha256")
        and lease.get("path")==".runtime/locks/gpu_0.lock" and lease["metadata"]["uuid"]==UUID
        and lease["metadata"]["launcher_pid"]==3349405,"Preserved exact old physical inode generation")
    stamp=datetime.fromisoformat(v["observed_at_utc"]);now=datetime.now(timezone.utc)
    require(stamp.tzinfo and stamp<=now,"Actual nonfuture closed handover")
    if fresh:require((now-stamp).total_seconds()<=300,"Fresh complete handover before new exclusive ownership")
    return v,identities

def validate_cap_v1(record,*,allow_drain=False):
    cap=read(record);q=read(cap["root_request_ref"]);h.active(q);h.source_refs_v1();reviewed_v1(q["source_review_ref"]);preflight_capability_v1(cap,q)
    require(cap.get("schema")=="GPU0_postdrain_two_slot_capability_v1" and cap.get("status")=="active"
        and cap.get("root_authorizer")=="root_agent" and cap.get("adapter_manifest_ref")==ref(h.MANIFEST)
        and cap.get("storage_policy_selector_ref")==h.SELECTOR_REF and cap.get("physical_gpu")==0 and cap.get("gpu_uuid")==UUID
        and cap.get("headroom_mib")==4096 and cap.get("maximum_concurrent_jobs")==2
        and cap.get("C3_compute_slots")==0 and cap.get("logical_job_sharing_authorized") is True
        and cap.get("external_job_sharing_authorized") is True and cap.get("owned_termination_escalation_authorized") is True
        and cap.get("owned_drain_after_expiry_authorized") is True,"Exact new two-slot GPU0 authority")
    require({k:cap[k]for k in q["capability"]}==q["capability"] and cap["source_review_ref"]==q["source_review_ref"] and cap["plan_ref"]==cap["root_request_ref"],"Immutable exact fresh ROOT capability request")
    require(cap["pool"] and len(cap["pool"])<=32 and all(c in "abcdefghijklmnopqrstuvwxyz0123456789_-"for c in cap["pool"]),"Safe new pool name")
    require(time.time()<(cap["drain_expires_unix"]if allow_drain else cap["expires_unix"])
        and 0<cap["termination_grace_seconds"]<=120 and cap["expires_unix"]+cap["termination_grace_seconds"]+5<=cap["drain_expires_unix"]
        and cap["drain_expires_unix"]<=datetime.fromisoformat(q["expires_at_utc"]).timestamp(),"Finite work/drain authority, no old deadline reset")
    jobs=cap["jobs"];require(set(j["worker_kind"]for j in jobs.values())==set(SLOTS)and 2<=len(jobs)<=40,"Exactly H058 and native declarations; no C3")
    validate_native_roster_v1(jobs)
    for name,j in jobs.items():
        kind=j["worker_kind"];require(j.get("module")==POOL[:-3].replace("/","."),"Only exact new pool dispatcher module")
        require(j["job_id"]==name and all(j.get(k)==v for k,v in slot_spec_v1(j).items()),"Exact per-slot allocator/margin/RSS/deadline")
        require(name and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"for c in name),"Safe finite logical identity")
        if kind=="H058":require(j.get("execution_enabled") is True and j.get("implementation_status")=="RUNNABLE_H058_REQUIRES_WHOLE_REVIEW","Exact implemented H058 slot")
        if kind=="native":require(j.get("implementation_status")=="COMPLETE_NATIVE_PARENT_MLM_REQUIRES_WHOLE_REVIEW"and j.get("execution_enabled") is True,"Exact implemented native parent runner, fresh whole review required")
    storage=read(cap["controller_storage_ref"])
    require(storage["job_id"]==cap["controller_job_id"] and storage["output_directory"]==q["controller_output_directory"]
        and storage["reserved_new_bytes"]==128*2**20 and storage["reservation"]["auxiliary_scopes"]==[],"Fresh isolated128MiB controller scope")
    validate_handover_v1(cap["handover_ref"],next(n for n,j in jobs.items()if j["worker_kind"]=="H058"))
    return cap

def guard_v1():return h.guard_v1()
def guarded_check_v1(g,storage,required=0):return g.budget_check_v9(storage,boundary=True,required_write_bytes=required)

def registry_v1(cap_ref,*,fresh=True):
    cap=validate_cap_v1(cap_ref);storage=read(cap["controller_storage_ref"]);path=storage["output_directory"]+"/registry.json"
    p=local(path);require(p.is_file()and not p.is_symlink()and p.stat().st_size<=32*2**20,"Bounded exact new controller registry")
    value=json.loads(p.read_bytes());who=value["controller"]
    require(value.get("capability_ref")==cap_ref and value.get("physical_gpu")==0 and value.get("pool")==cap["pool"]
        and value.get("lifecycle")=="running"and value.get("safety_error") is None
        and value.get("ambiguous_descendants_or_lease_holders")==[],"Same live new pool registry")
    proof=value.get("new_exclusive_acquisition",{})
    require(proof.get("acquired_new_exclusive_flock") is True and proof.get("writable_independent_exclusion_observed") is True and proof.get("inherited_or_adopted_ownership") is False
        and proof.get("pid")==who["pid"] and value["lease_metadata"]["launcher_pid"]==who["pid"] and value["lease_metadata"]["uuid"]==UUID,"Genuine newly acquired exclusive owner proof, never adoption")
    base=h.load(h.BASE_POOL,"_postdrain_identity_primitives")
    now=base.proc_identity(who["pid"])
    require(now["uid"]==who["uid"]==os.getuid()and now["start_ticks"]==who["start_ticks"]and now["state"]not in("Z","X"),"Exact actual new controller identity")
    require(0<=time.time()-value["updated_unix"]<=30,"Fresh live registry; stale failure stops owned attachment")
    return cap,value

def validate_H058_request_v1(q):
    cap=validate_cap_v1(q["postdrain_pool_capability_ref"])
    require(q["job_id"]in cap["jobs"] and cap["jobs"][q["job_id"]]["worker_kind"]=="H058"
        and datetime.fromisoformat(q["expires_at_utc"]).timestamp()<=cap["expires_unix"],"Exact new finite H058 slot/authority")
    proof,_=validate_handover_v1(cap["handover_ref"],q["job_id"])
    require(q["native_noncoexistence_qualification_ref"]==proof["completed_current_native_success_ref"],"Same genuine completed original native gate")


def validate_submission_v1(cap_ref,record):
    cap=validate_cap_v1(cap_ref);v=read(record);h.active(v)
    require(v.get("root_authorizer")=="root_agent" and v.get("capability_ref")==cap_ref and v.get("job_id")in cap["jobs"],"Exact declared fresh pool submission")
    kind=cap["jobs"][v["job_id"]]["worker_kind"]
    if kind=="H058":
        require(v.get("schema")=="GPU0_postdrain_H058_submission_v1","Exact H058 submission")
        s,q,cfg,storage,_=h.validate_input_v1(v["input_ref"])
    else:
        phase=cap["jobs"][v["job_id"]].get("native_phase","mlm")
        if phase=="mlm":
            from . import native_common_v1 as native
            require(v.get("schema")=="GPU0_postdrain_native_parent_submission_v1","Exact original-parent MLM submission")
            s,q,cfg,storage,_=native.validate_input_v1(v["input_ref"])
        else:
            from . import downstream_contract_v1 as downstream
            require(v.get("schema")=="GPU0_postdrain_native_phase_submission_v1","Exact full downstream phase submission")
            s,q,cfg,storage,_=downstream.inspect_input_v1(v["input_ref"])
        require(q["parent_id"]==cap["jobs"][v["job_id"]]["parent_id"] and q["phase"]==phase,"Exact original parent/phase")
    require(q["postdrain_pool_capability_ref"]==cap_ref and s["job_id"]==v["job_id"]
        and (kind=="native"and cap["jobs"][v["job_id"]].get("native_phase","mlm")!="mlm" or s["worker_auxiliary_scope"]==q["packet_directory"]+"/worker_aux") and not local(s["output_directory"]).exists(),"Fresh admitted original source/input/output")
    return v,s,q,cfg,storage

def expires(value):return datetime.fromisoformat(value["expires_at_utc"]).timestamp()

def preflight_capability_v1(cap,q):
    """No reservation mutation until the complete concrete capsule shape passes."""
    policy=read(GPU_POLICY);renewal=read(h.CERT)
    require(policy.get("status")=="active"and policy.get("root_authorizer")=="root_agent"and policy.get("allowed_physical_GPUs")==[0]and policy.get("GPU0_uuid")==UUID
        and policy.get("parallel_jobs_on_GPU0_allowed") is True and policy.get("preserve_external_jobs") is True,"Pinned actual human GPU0/external preservation policy")
    require(cap.get("human_GPU0_policy_ref")==GPU_POLICY and type(cap.get("drain_expires_unix"))in(int,float)and cap["drain_expires_unix"]<datetime.fromisoformat(renewal["renewed_expires_at_utc"]).timestamp(),"Complete drain window inside preserved marker renewal authority")
    require(q.get("schema")=="GPU0_postdrain_controller_ROOT_request_v1"and q.get("root_authorizer")=="root_agent"
        and q.get("status")=="active"and q.get("execution_authorized") is True
        and q.get("adapter_manifest_ref")==ref(h.MANIFEST)and q.get("storage_policy_selector_ref")==h.SELECTOR_REF,"Exact fresh ROOT controller request")
    require(cap.get("schema")=="GPU0_postdrain_two_slot_capability_v1"and cap.get("status")=="active"and cap.get("root_authorizer")=="root_agent"
        and cap.get("adapter_manifest_ref")==ref(h.MANIFEST)and cap.get("storage_policy_selector_ref")==h.SELECTOR_REF
        and cap.get("physical_gpu")==0 and cap.get("gpu_uuid")==UUID and cap.get("maximum_distinct_project_GPUs")==1
        and cap.get("maximum_concurrent_jobs")==2 and cap.get("headroom_mib")==4096 and cap.get("C3_compute_slots")==0
        and cap.get("logical_job_sharing_authorized") is True and cap.get("external_job_sharing_authorized") is True
        and cap.get("owned_termination_escalation_authorized") is True and cap.get("owned_drain_after_expiry_authorized") is True,"Complete exact GPU0 capability shape before reserve")
    require(type(cap.get("expires_unix"))in(int,float)and type(cap.get("drain_expires_unix"))in(int,float)
        and type(cap.get("termination_grace_seconds"))is int and 0<cap["termination_grace_seconds"]<=120
        and time.time()<cap["expires_unix"]and cap["expires_unix"]+cap["termination_grace_seconds"]+5<=cap["drain_expires_unix"]<=expires(q),"Finite exact work and drain interval")
    require(isinstance(cap.get("pool"),str)and 1<=len(cap["pool"])<=32 and all(ch in "abcdefghijklmnopqrstuvwxyz0123456789_-"for ch in cap["pool"]),"Safe fresh pool name")
    jobs=cap.get("jobs");require(isinstance(jobs,dict)and 2<=len(jobs)<=40 and{j.get("worker_kind")for j in jobs.values()}==set(SLOTS),"Exactly two typed slots")
    validate_native_roster_v1(jobs)
    for name,j in jobs.items():
        kind=j["worker_kind"]
        require(j.get("job_id")==name and isinstance(name,str)and name and all(ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"for ch in name)
            and j.get("module")==POOL[:-3].replace("/",".")and all(j.get(k)==v for k,v in slot_spec_v1(j).items()),"Exact finite slot identity and original resource caps")
        if kind=="H058":require(j.get("execution_enabled") is True and j.get("implementation_status")=="RUNNABLE_H058_REQUIRES_WHOLE_REVIEW","Implemented H058 source only")
        else:require(j.get("execution_enabled") is True and j.get("implementation_status")=="COMPLETE_NATIVE_PARENT_MLM_REQUIRES_WHOLE_REVIEW","Concrete native slot requires exact new review and ROOT capability")
    return cap


def slot_spec_v1(job):
    if job["worker_kind"]=="H058":return SLOTS["H058"]
    phase=job.get("native_phase","mlm");value=dict(SLOTS["native"])
    if phase=="finalize":
        value.update(cuda_cap_bytes=0,margin_mib=0,peak_mib=0)
    elif phase!="mlm":
        require(job.get("resource_profile")in("original_downstream_74GiB_proposal","explicit_unqualified_full_stage_32GiB"),"Explicit downstream resource choice, never implied fit")
        allocator=74 if job["resource_profile"]=="original_downstream_74GiB_proposal" else 32
        value.update(cuda_cap_bytes=allocator*2**30,peak_mib=(allocator+2)*1024)
    return value

def validate_native_roster_v1(jobs):
    registry=read(ref(h.HERE+"/parent_registry.v1.json"));allowed=set(registry["parents"])
    H058=[j for j in jobs.values()if j.get("worker_kind")=="H058"];native=[j for j in jobs.values()if j.get("worker_kind")=="native"]
    phases={"mlm","contrastive","embedding_contrastive_train","embedding_contrastive_val","embedding_query_paths_train","embedding_query_paths_val","RAM_retrieve_and_seal","finalize"}
    identities={(j.get("parent_id"),j.get("native_phase","mlm"))for j in native}
    require(len(H058)==1 and 1<=len(native)<=39 and len(H058)+len(native)==len(jobs) and len(identities)==len(native)
        and all(p in allowed and phase in phases and (phase!="mlm"or p in registry["remaining_fresh_fit_parents"])for p,phase in identities),
        "One H058 and finite unique original parent/phase declarations, no replay of current VAL000 MLM")
    return jobs

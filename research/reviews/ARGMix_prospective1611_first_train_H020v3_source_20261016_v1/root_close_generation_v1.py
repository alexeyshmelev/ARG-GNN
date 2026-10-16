"""Narrow root-only close of one genuinely successful prospective1611 generation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1"
SOURCE=REL+"/root_close_generation_v1.py"
EXEC="research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1"
WRAPPER={'path': 'research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/root_generate_v1.py', 'sha256': '302c4216e2f95272a83903ec476342e6d0b8999b6bbacb8a01a339d18c2693ed'}
HANDOFF={'path': 'research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/execution_handoff.v1.json', 'sha256': '812eb477f9af961b68d1aadea54a0fb60ea2d566e8d940e7f1996ddb0461b157'}
STATUS="completed_prospective1611_full_parent_generation"

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def helper():
    p=ROOT/WRAPPER["path"];need(sha(p)==WRAPPER["sha256"],"Exact sealed generation wrapper utilities")
    sp=importlib.util.spec_from_file_location("_prospective1611_frozen_generation_utilities",p)
    c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c);return c

def metadata_charge_gate(c,generation_packet,close_packet,records,allowance):
    # Only these exact root metadata directories/refs; never a host/project scan.
    paths=set()
    for directory in (generation_packet,close_packet):
        for p in directory.iterdir():
            if p.name=="worker_aux":continue
            need(p.is_file() and not p.is_symlink(),"Immutable flat owned root metadata")
            paths.add(p)
    paths.update(c.checked(r) for r in records)
    total=0
    for p in paths:
        st=p.lstat();need(stat.S_ISREG(st.st_mode) and st.st_nlink==1,"Regular singly linked root metadata")
        total+=max(st.st_size,st.st_blocks*512)
    # Leave64KiB each for close authority, close result and the appended event.
    need(total+3*65536<=allowance,"Actual root metadata plus planned close metadata fit unchanged per-event charge")
    return total

def validate(c,record):
    r=c.read(record)
    need(r.get("schema")=="prospective1611_generation_success_close_root_request_v1" and r.get("active") is True
        and r.get("root_authorizer")=="root_agent" and r.get("close_source_ref")==c.ref(SOURCE)
        and r.get("execution_handoff_ref")==HANDOFF,"Explicit exact root success-close request")
    now=datetime.now(timezone.utc);end=c.stamp(r["expires_at_utc"])
    need(c.stamp(r["created_at_utc"])<=now<end and end-now<=timedelta(hours=1),"Fresh root close authority <=1hour")
    need(r.get("preserve_all_files") is True and r.get("failed_scope_credit_authorized") is False
        and r.get("historical_mutation_authorized") is False,"Only successful scope, no deletion/history/failed credit")
    c.read(HANDOFF)
    completed=c.read(r["completed_ref"])
    need(completed.get("schema")=="prospective1611_generation_root_completed_v1" and completed.get("status")==STATUS
        and completed.get("length_bp")==35178458 and completed.get("haplotypes")==3222
        and completed.get("wrapper_ref")==WRAPPER and completed.get("inference_performed") is False,"Exact actual full wrapper completion")
    need(completed["actual_wait_ref"]==r["worker_actual_wait_ref"],"Actual completed worker wait ref")
    er=c.read(completed["root_request_ref"]);scope=er["scope_id"];parent=er["parent_id"]
    need(er["active"] is True and er["root_authorizer"]=="root_agent" and er["registered_before_execution"] is True
        and parent in c.PARENTS and c.re.fullmatch("[A-Za-z0-9_-]{1,80}",scope)
        and er["wrapper_ref"]==WRAPPER and r["parent_id"]==parent
        and r["scope_id"]==scope and er["automatic_retry"] is False and er["downscale_or_seed_fallback"] is False,"Exact registered successful parent/scope")
    packet=c.local(EXEC+"/admitted_"+scope);output=c.local("data/ARGMix_prospective1611_first_train_H020v3_generated_"+scope)
    need(c.local(r["completed_ref"]["path"])==output/"completed.json" and c.local(r["worker_actual_wait_ref"]["path"])==packet/"actual_wait.json","Actual scope completion/wait paths")
    storage=c.read(completed["storage_ref"]);own=storage["reservation"]
    need(c.local(completed["storage_ref"]["path"])==packet/"storage.json"
        and own["key"]=="ARGMix/prospective1611/first_train_H020v3/generate/"+scope and own["output"]==str(output)
        and own["auxiliary_scopes"]==[str(packet/"worker_aux")]
        and own["reserved_bytes"]==storage["reserved_new_bytes"]==16*1024**3
        and WRAPPER in own["consumer_bindings"] and completed["root_request_ref"] in own["consumer_bindings"],"Exact16GiB original generation allocation and source")
    gr=c.read(completed["generation_request_ref"]);resource_record=dict(path=str(c.local(gr["resource_binding"]).relative_to(ROOT)),sha256=gr["resource_binding_sha256"])
    resource=c.read(resource_record)
    need(gr["schema"]=="prospective1611_generation_request_v1" and gr["root_approved"] is True and gr["parent_id"]==parent
        and gr["generator_source_sha256"]==c.GEN_SHA and gr["science_plan_sha256"]==c.PLAN_SHA
        and gr["map_derivation_receipt_sha256"]==c.MAP_SHA and resource["wrapper_source_sha256"]==WRAPPER["sha256"]
        and resource["admission_receipt_sha256"]==c.ref(packet/"reserve_receipt.json")["sha256"],"Actual unchanged generator/science/map/resource/admission")
    observed=c.read(completed["observed_generation_receipt_ref"]);private=c.read(completed["private_generation_receipt_ref"])
    need(observed["complete"] is True and private["complete"] is True and observed["parent_id"]==private["parent_id"]==parent
        and observed["diploids"]==1611 and observed["haplotypes"]==3222 and observed["map_binding"]["length_bp"]==35178458
        and observed["generator_source_sha256"]==private["generator_source_sha256"]==c.GEN_SHA
        and observed["map_derivation_receipt_sha256"]==c.MAP_SHA
        and private["request"]["sha256"]==completed["generation_request_ref"]["sha256"]
        and private["resource_binding_sha256"]==resource_record["sha256"],"Actual complete common observations/private generation metadata; no payload opened")
    worker=c.read(r["worker_actual_wait_ref"])
    need(worker["schema"]=="prospective1611_generation_actual_direct_wait_v1" and worker["actual_direct_worker_wait"] is True
        and type(worker["actual_OS_wait_status"]) is int and os.waitstatus_to_exitcode(worker["actual_OS_wait_status"])==worker["returncode"]==0
        and worker["all_owned_processes_and_groups_drained"] is True and worker["operational_failure"] is None
        and worker["drain_method"]=="subreaper_owned_identity_roster_absence_no_global_scan"
        and worker["adopted_child_actual_waits"]==[],"Genuine direct worker OS wait0 and own subreaper drain")
    leader=worker["root_pid"];child=worker["child_pid"];e=worker["owned_lifecycle_evidence"]
    need(type(leader) is int and type(child) is int and leader>=2 and child>=2 and leader!=child
        and worker["process_group"]==child and e["signals_use_pidfd_only"] is True and e["no_numeric_group_signals"] is True
        and e["reused_numeric_PIDs_excluded"]==[] and len(e["owned_identities"])==1
        and e["owned_identities"][0]["pid"]==child and type(e["owned_identities"][0]["start"]) is int,"Exact owned child pidfd identity evidence")
    outer=c.read(r["outer_actual_wait_ref"])
    need(outer["schema"]=="prospective1611_generation_actual_outer_wait_root_v1" and outer["root_authorizer"]=="root_agent"
        and outer["actual_direct_outer_tool_wait"] is True and type(outer["actual_exit_code"]) is int and outer["actual_exit_code"]==0
        and type(outer["actual_exec_session_id"]) is int and outer["actual_exec_session_id"]>=0
        and isinstance(outer["actual_tool_chunk_id"],str) and bool(outer["actual_tool_chunk_id"])
        and outer["wrapper_pid"]==leader and outer["wrapper_ref"]==WRAPPER
        and outer["completed_ref"]==r["completed_ref"] and outer["worker_actual_wait_ref"]==r["worker_actual_wait_ref"],"Genuine consumed outer tool terminal0; never infer it from worker wait")
    revoked=c.read(r["root_revocation_ref"])
    need(revoked["schema"]=="prospective1611_generation_success_scope_root_revocation_v1" and revoked["root_authorizer"]=="root_agent"
        and revoked["key"]==own["key"] and revoked["root_generation_request_ref"]==completed["root_request_ref"]
        and revoked["completed_ref"]==r["completed_ref"] and revoked["worker_actual_wait_ref"]==r["worker_actual_wait_ref"]
        and revoked["outer_actual_wait_ref"]==r["outer_actual_wait_ref"] and revoked["close_source_ref"]==r["close_source_ref"]
        and revoked["all_old_scope_launch_resume_write_capabilities_revoked"] is True
        and revoked["no_future_writes_to_closed_output_or_auxiliary_scopes"] is True and revoked["preserve_all_files"] is True
        and revoked["historical_mutation_authorized"] is False and revoked["failed_scope_credit_authorized"] is False
        and revoked["all_remote_allocation_pids"]==[leader,child],"Actual permanent root future-write revocation for this exact successful scope")
    need(c.stamp(completed["completed_utc"])<=c.stamp(outer["observed_utc"])<=c.stamp(revoked["issued_at_utc"])<=now,"Root actually consumes terminal completion before revoking/closing")
    close_packet=c.local(REL+"/closed_"+scope);need(not close_packet.exists(),"Fresh single-use close packet; no blind retry")
    return r,completed,storage,own,worker,outer,revoked,packet,close_packet

def execute(record):
    c=helper();r,done,storage,own,worker,outer,revoked,packet,close_packet=validate(c,record)
    sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    renewal=c.load(c.RENEWAL,"_prospective1611_close_fresh_renewal");renewal.install_renewal_v1(c.CERT)
    need(c.stamp(r["expires_at_utc"])+timedelta(seconds=600)<c.stamp(c.read(c.CERT)["renewed_expires_at_utc"]),"Current renewal covers close and drain")
    guard=c.load(c.GUARD,"_prospective1611_first_train_H020v3_success_close_guard");guard.configure_v2(c.REPAIR,policy_ref=c.POLICY)
    need(guard.META_CHARGE==524288,"Unchanged separate per-event metadata charge")
    reservation_sha=guard.budget.digest(own)
    need(revoked["reservation_sha256"]==reservation_sha,"Root revoked exact original reservation")
    view=guard.inspect_v1(expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    close_packet.mkdir()
    pids=[worker["root_pid"],worker["child_pid"]]
    observations=[guard._absent(pid) for pid in pids]  # ROOT-only exact original targeted signal0 ESRCH checks.
    observation=c.save(close_packet/"root_observation.json",dict(schema="concurrent_completed_scope_targeted_root_observation_v1",
        root_authorizer="root_agent",key=own["key"],reservation_sha256=reservation_sha,storage_ref=done["storage_ref"],
        launch_ref=r["worker_actual_wait_ref"],direct_wait_ref=r["outer_actual_wait_ref"],completed_ref=r["completed_ref"],
        launcher_pid=worker["root_pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
        completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,
        pids=pids,observations=observations,root_close_request_ref=record,root_revocation_ref=r["root_revocation_ref"],
        worker_pidfd_subreaper_wait_ref=r["worker_actual_wait_ref"],outer_actual_tool_wait_ref=r["outer_actual_wait_ref"],close_source_ref=r["close_source_ref"]))
    request=dict(key=own["key"],reservation_sha256=reservation_sha,storage_ref=done["storage_ref"],storage_locator=[],
        launch_ref=r["worker_actual_wait_ref"],launch_pid_locator=["root_pid"],
        launch_identity=dict(schema=worker["schema"],root_pid=worker["root_pid"],child_pid=worker["child_pid"],
            actual_direct_worker_wait=True,returncode=0,operational_failure=None,all_owned_processes_and_groups_drained=True),
        direct_wait_ref=r["outer_actual_wait_ref"],wait_pid_locator=["wrapper_pid"],wait_exit_locator=["actual_exit_code"],
        wait_identity=dict(schema=outer["schema"],root_authorizer="root_agent",actual_direct_outer_tool_wait=True,
            wrapper_pid=worker["root_pid"],actual_exit_code=0,completed_ref=r["completed_ref"],worker_actual_wait_ref=r["worker_actual_wait_ref"]),
        completed_ref=r["completed_ref"],completed_identity=dict(schema=done["schema"],status=STATUS,parent_id=done["parent_id"],
            length_bp=35178458,haplotypes=3222,wrapper_ref=WRAPPER,actual_wait_ref=r["worker_actual_wait_ref"],completed_utc=done["completed_utc"]),
        observation_ref=observation)
    # status=completed_* uses the original guard branch; no schema/policy/function mutation.
    auth=guard.operation_draft_v1("close_completed",request,expected_before=guard._pins(view),bindings=view["bindings"])
    auth.update(active=True,expires_at_utc=r["expires_at_utc"])
    metadata_charge_gate(c,packet,close_packet,[record,r["outer_actual_wait_ref"],r["root_revocation_ref"]],guard.META_CHARGE)
    authority=c.save(close_packet/"close_authority.json",auth)
    result=guard.apply_close_completed_v1(authority,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    need(result["retained_bytes"]==result["value"]["measurement"]["conservative_scope_bytes"]
        and result["credit_bytes"]==own["reserved_bytes"]-result["retained_bytes"]
        and result["scope_measurements"]==1 and result["old_journal_mutated"] is False and result["artifacts_deleted"]==0
        and result["value"]["future_writes_and_relaunch_revoked"] is True,"Exact original stat-only retained-byte charge and append-only successful close")
    receipt=c.save(close_packet/"close_receipt.json",result)
    print(json.dumps(dict(close_receipt_ref=receipt,retained_bytes=result["retained_bytes"],credit_bytes=result["credit_bytes"],
        projected_bytes=result["projected_bytes"],all_files_preserved=True,failed_scope_relaxation=False)),flush=True)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--request");p.add_argument("--request-sha256");a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status="source_only_no_import_guard_probe_or_close",success_only=True)));return 0
    need(Path.cwd()==ROOT and a.request and a.request_sha256,"Exact root cwd and explicit actual request/hash")
    return execute(dict(path=a.request,sha256=a.request_sha256))

if __name__=="__main__":raise SystemExit(main())

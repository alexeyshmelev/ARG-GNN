"""Root-only exact close of four completed C3 short-parent stages. Inert default."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/American_C3_short2mb_completed_close_source_v1"
SOURCE="research/reviews/American_C3_short2mb_completed_close_source_v1/root_close_completed_stages.py"
EVIDENCE_REF={'path': 'research/reviews/American_C3_short2mb_completed_close_source_v1/stage_evidence_refs.json', 'sha256': 'bd925c5c254c40ebccc50b284ad6bdbea6ee31442b3d935ec98c9e617def16de'}
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CHOICES=("000-simulate","000-infer","001-simulate","001-infer")
def require(ok,message):
    if not ok:raise ValueError(message)
def utc():return datetime.now(timezone.utc).isoformat()
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(p!=ROOT and ROOT in p.parents and ".." not in p.parts and p.resolve()==p,"Canonical project metadata path")
    return p
def ref(path):
    p=local(path);return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact metadata SHA reference required")
    p=local(record["path"]);require(p.suffix==".json","Only compact JSON evidence")
    raw=p.read_bytes();require(len(raw)<=1048576 and hashlib.sha256(raw).hexdigest()==record["sha256"],"Unchanged bounded JSON evidence required")
    return json.loads(raw)
def save(folder,name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Compact charged root operation metadata required")
    with (folder/name).open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(folder/name)
def overlay():
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    require(ref(GUARD_REF["path"])==GUARD_REF,"Frozen current credit-v3 source")
    spec=importlib.util.spec_from_file_location("_root_C3_completed_close_original_v3",ROOT/GUARD_REF["path"])
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # Close mutations must use the original functions, never a consumer memo.
    for name in ("canonical","file_bytes","mutex"):
        require(getattr(m.budget.v4,name).__module__==m.budget.v4.__name__,"Memo/wrapped v4 functions forbidden during root close")
    m.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    return m
def absent(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=utc())
    except PermissionError:raise ValueError("Root cannot certify target PID absence") from None
    raise ValueError("Owned PID remains present, zombie or reused; no credit: "+str(pid))
def group_absent(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return dict(pgid=pgid,absent=True,method="targeted os.killpg(pgid,0) ESRCH",checked_utc=utc())
    except PermissionError:raise ValueError("Root cannot certify owned group absence") from None
    raise ValueError("Owned process group remains present or reused; no credit: "+str(pgid))
def evidence_for(which):
    evidence=read(EVIDENCE_REF);row=evidence["stages"][which]
    ad=read(row["admission_ref"]);launch=read(row["launch_ref"]);wait=read(row["wait_ref"])
    completed=read(row["completed_ref"]);preparation=read(row["preparation_ref"])
    own=ad["storage"]["reservation"]
    require(own["key"]==row["key"] and own["output"]==row["output"]
        and own["auxiliary_scopes"]==row["auxiliary_scopes"] and own["reserved_bytes"]==row["reserved_bytes"],
        "Exact original reservation key and all owned scopes/cap required")
    require(ad["parent_id"]==launch["parent_id"]==wait["parent_id"]==preparation["parent_id"]==row["parent_id"]
        and ad["stage"]==launch["stage"]==wait["stage"]==row["stage"]
        and launch["admission_ref"]==wait["admission_ref"]==row["admission_ref"]
        and wait["pid"]==launch["pid"]==launch["worker_pid"]==wait["worker_pid"]
        and launch["supervisor_pid"]==wait["supervisor_pid"]==preparation["root_supervisor_pid"],
        "Exact original parent/stage/direct worker/supervisor identity required")
    require(wait["returncode"]==0 and type(wait["returncode"]) is int and wait["actual_direct_worker_wait"] is True
        and wait["worker_process_group_drained"] is True
        and wait["root_must_observe_supervisor_exit_before_credit"] is True
        and wait["operational_failure"] is None and wait["supervisor_capture_error"] is None
        and wait["terminal_log_materialization_error"] is None and wait["completed_ref"]==row["completed_ref"],
        "Actual clean direct wait0 and historical group drain required")
    require(preparation["status"]=="completed_C3_2Mb_TRAIN_parent_preparation"
        and preparation["split"]=="train" and preparation["sequence_length_bp"]==2000000
        and preparation["model_training_performed"] is False
        and preparation["stages"][row["stage"]]==dict(completion_ref=row["completed_ref"],wait_ref=row["wait_ref"]),
        "Original complete independent TRAIN preparation and stage references required")
    require(all(completed.get(k)==v for k,v in row["completed_identity"].items()),"Completed producer identity changed")
    return evidence,row,ad,launch,wait,own
def approval(record,evidence):
    cert=read(record)
    require(cert.get("schema")=="American_C3_short_wave_root_close_approval_v1"
        and cert.get("root_authorizer")=="root_agent" and cert.get("active") is True
        and cert.get("close_helper_ref")==ref(SOURCE) and cert.get("stage_evidence_ref")==EVIDENCE_REF
        and cert.get("stage_evidence")==evidence["stages"]
        and cert.get("all_genuinely_owned_processes_and_groups_listed") is True
        and cert.get("all_prior_input_config_storage_source_launch_resume_write_capabilities_revoked") is True
        and cert.get("preserve_all_files_and_historical_journals") is True,
        "Explicit root complete-roster/revocation approval required")
    pids=cert.get("root_owned_pids");groups=cert.get("root_owned_process_groups")
    require(isinstance(pids,list) and 1<=len(pids)<=256 and len(set(pids))==len(pids)
        and all(type(p) is int and p>=2 for p in pids) and set(evidence["required_owned_pids"])<=set(pids),
        "Root must include all genuinely owned supervisor/worker/descendant PIDs")
    require(isinstance(groups,list) and 1<=len(groups)<=256 and len(set(groups))==len(groups)
        and all(type(g) is int and g>=2 for g in groups)
        and set(evidence["required_owned_worker_groups"])<=set(groups),"Complete genuinely owned group roster required")
    return cert,pids,groups
def close(which,approval_ref):
    evidence,row,ad,launch,wait,own=evidence_for(which)
    cert,pids,groups=approval(approval_ref,evidence)
    observations=[absent(pid) for pid in pids];group_observations=[group_absent(g) for g in groups]
    m=overlay()
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    require(state[0].get(row["key"])==own and row["key"] not in state[1],"Original reservation must still be open and unchanged")
    packet=HERE/("closed_"+which);require(not packet.exists(),"No close replay or metadata overwrite");packet.mkdir()
    obs=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",
        key=row["key"],reservation_sha256=m.budget.digest(own),storage_ref=row["admission_ref"],
        launch_ref=row["launch_ref"],direct_wait_ref=row["wait_ref"],completed_ref=row["completed_ref"],
        launcher_pid=launch["pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
        completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
        external_NFS_writers_outside_observation=True,pids=pids,observations=observations,
        root_owned_process_groups=groups,process_group_observations=group_observations,
        root_approval_ref=approval_ref,preparation_ref=row["preparation_ref"],source_ref=ref(SOURCE),recorded_utc=utc(),
        revocation_applies_to_all_prior_input_config_storage_source_launch_and_resume_capabilities=True)
    observation_ref=save(packet,"root_observation.json",obs)
    request=dict(key=row["key"],reservation_sha256=m.budget.digest(own),storage_ref=row["admission_ref"],
        storage_locator=["storage"],launch_ref=row["launch_ref"],launch_pid_locator=["pid"],
        launch_identity=dict(pid=launch["pid"],parent_id=row["parent_id"],stage=row["stage"],supervisor_pid=launch["supervisor_pid"]),
        direct_wait_ref=row["wait_ref"],wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
        wait_identity=dict(returncode=0,actual_direct_worker_wait=True,worker_process_group_drained=True,
                           parent_id=row["parent_id"],stage=row["stage"],supervisor_pid=wait["supervisor_pid"]),
        completed_ref=row["completed_ref"],completed_identity=row["completed_identity"],observation_ref=observation_ref)
    operation=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
    operation.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    authority_ref=save(packet,"close_authority.json",operation)
    for pgid in groups:group_absent(pgid)
    # Sole scope inventory is inside this unchanged close primitive. No memo.
    receipt=m.apply_close_completed_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(packet,"close_receipt.json",receipt)
    print(json.dumps(dict(closed=which,key=row["key"],retained_bytes=receipt["retained_bytes"],
        credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"],files_deleted=0)),flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--close",choices=CHOICES)
    p.add_argument("--root-approval-ref");p.add_argument("--root-approval-sha256");args=p.parse_args()
    if args.close:
        require(args.root_approval_ref and args.root_approval_sha256,"Explicit bound root close approval required")
        close(args.close,dict(path=args.root_approval_ref,sha256=args.root_approval_sha256));return 0
    print(json.dumps(dict(status="source_only_no_guard_or_process_checks",stages=list(CHOICES),
        root_approval_required=True,scope_inventories=0,closes_applied=0,files_deleted=0,GPU_authorized=False),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

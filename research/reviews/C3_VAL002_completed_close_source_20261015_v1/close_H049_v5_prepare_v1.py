"""Single original completed H049v5 preparation close; inherited executor with original launcher_pid wait locator. Default metadata only."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE='research/reviews/C3_VAL002_completed_close_source_20261015_v1'
SOURCE=HERE+"/close_H049_v5_prepare_v1.py"
REFS_REF={'path': 'research/reviews/C3_VAL002_completed_close_source_20261015_v1/H049_v5_actual_refs.json', 'sha256': 'bc0fbbb3b65c78a02df6abb596936776a8e931fe22c280b704af47b0eee97566'}
SCOPES=("H049_v5_prepare",)
def need(ok,msg):
    if not ok:raise ValueError(msg)

def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project path")
    return p

def ref(value):
    p=local(value);need(p.suffix in (".py",".json") and p.stat().st_size<=262144,"Source/JSON only; no payload hashing")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(r):
    p=local(r["path"]);need(p.suffix==".json" and p.stat().st_size<=131072 and ref(p)==r,"Exact bounded actual JSON")
    return json.loads(p.read_text())

def save(folder,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(raw)<=65536 and Path(name).name==name,"Compact flat root close certificate")
    fd=os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(folder/name)

def absence(pid,group=False):
    need(type(pid) is int and pid>=2,"Exact known original PID/group")
    try:
        if group:os.killpg(pid,0)
        else:os.kill(pid,0)
    except ProcessLookupError:
        return dict(**({"process_group":pid} if group else {"pid":pid}),absent=True,
            method="targeted os.killpg(group,0) ESRCH" if group else "targeted os.kill(pid,0) ESRCH",
            checked_utc=datetime.now(timezone.utc).isoformat())
    raise ValueError("Original PID/group present, reused, zombie or inaccessible")

def collect(scope):
    need(scope=="H049_v5_prepare","Only fixed genuinely completed bounded v5 scope")
    manifest=read(REFS_REF);components=manifest["components"];row=manifest["targets"][scope]
    need(all(ref(r["path"])==r for r in components.values()),"Actual original980 guard/policy/repair/core/launcher pins")
    need(ref(manifest["original_close_helper_ref"]["path"])==manifest["original_close_helper_ref"],"Inherited original close source")
    for name,r in row.items():
        if name.endswith("_ref"):need(ref(r["path"])==r,"Exact original boundedv5 source/evidence pin")
    spec=read(row["admission_ref"]);storage=read(row["storage_ref"]);own=storage["reservation"]
    launch=read(row["launch_ref"]);wait=read(row["direct_wait_ref"]);completed=read(row["completed_ref"])
    lease=read(row["phase_lease_ref"]);accepted=read(row["root_acceptance_ref"])
    need(storage["accounting_mode"]=="prospective_980gb_v1" and storage["completed_overlay_policy_ref"]==components["policy"]
        and storage["budget_components"]["consumer_adapter"]==components["guard"]
        and storage["reserved_new_bytes"]==own["reserved_bytes"]==row["cap"]
        and row["completed_ref"]["path"]==storage["output_directory"]+"/completed.json"
        and spec["completed_overlay_policy_ref"]==components["policy"]
        and spec["refs"]["storage"]==row["storage_ref"] and spec["root_driver_ref"]==row["IO_source_ref"],
        "Exact original successful boundedv5 storage/cap/source")
    need(launch["input_ref"]==completed["input_ref"]==row["admission_ref"]
        and launch["phase"]==wait["phase"]==spec["phase"]=="prepare"
        and wait["actual_direct_launcher_wait"] is True and wait["returncode"]==0 and wait["error"] is None
        and wait["owned_group_drained"] is True and wait["independent_phase_lease_exclusion_observed"] is True
        and wait["full_terminal_guard_returned"] is True
        and wait["launch_ref"]==row["launch_ref"] and wait["completed_ref"]==row["completed_ref"]
        and launch["launcher_pid"]==wait["launcher_pid"] and launch["launcher_start"]==wait["launcher_start"]
        and launch["preparation_execution_version"]==wait["preparation_execution_version"]==completed["preparation_execution_version"]==2
        and launch["bounded_writer_execution_version"]==wait["bounded_writer_execution_version"]==5,
        "Original genuine preparev2/writerv5 direct wait0/drain/terminal guard lineage")
    need(launch["phase_lease_ref"]==wait["phase_lease_ref"]==spec["bounded_phase_lease_ref"]==row["phase_lease_ref"]
        and lease["schema"]=="bounded_phase_root_lifetime_lease_v1"
        and lease["job_id"]==storage["job_id"]==spec["job_id"] and lease["phase"]=="prepare"
        and type(lease["root_pid"]) is int and type(lease["root_start_ticks"]) is int,
        "Exact original outer root supervisor from lifetime lease; no invented outer PID")
    need(accepted["root_authorizer"]=="root_agent" and accepted["actual_tool_session_id"]==88211
        and accepted["actual_consumed_tool_terminal_exit_code"]==0
        and accepted["actual_wait_ref"]==row["direct_wait_ref"] and accepted["completed_ref"]==row["completed_ref"]
        and accepted["all40_parent_ids_verified"] is True and accepted["all80_compact_archive_receipt_hashes_verified"] is True
        and accepted["no_failed_scope_credit_or_artifact_deletion"] is True,
        "Preserve genuine root consumed outer terminal0 and existing all40 metadata acceptance")
    need(completed["hypothesis_id"]=="H049" and completed["status"]=="completed_all40_compact_cut_recipes"
        and completed["observations_only"] is True and completed["targets_read"] is False
        and completed["checkpoint_payloads_read"] is False and completed["CUDA_initialized"] is False
        and len(completed["parents"])==40,"Actual observation-only successful all40 completion")
    pids={launch["launcher_pid"],lease["root_pid"],*[r["pid"] for r in wait["owned_process_identities"]]}
    groups={*wait["historical_owned_groups"],*[r["group"] for r in wait["owned_process_identities"]]}
    need(all(type(x) is int and x>=2 for x in pids|groups) and len(pids)<=256,"Exact bounded original root/worker roster")
    launch_pid="launcher_pid"
    launch_identity={k:launch[k] for k in ("launcher_pid","launcher_start","phase","input_ref","preparation_execution_version","bounded_writer_execution_version","phase_lease_ref")}
    wait_identity={k:wait[k] for k in ("launcher_pid","launcher_start","returncode","actual_direct_launcher_wait","owned_group_drained","error","launch_ref","completed_ref","preparation_execution_version","bounded_writer_execution_version","phase_lease_ref","independent_phase_lease_exclusion_observed","full_terminal_guard_returned")}
    completion_identity={k:completed[k] for k in ("schema","hypothesis_id","status","input_ref","producer_ref","preparation_execution_version","observations_only","targets_read","checkpoint_payloads_read","CUDA_initialized")}
    return manifest,row,spec,storage,own,launch,wait,completed,sorted(pids),sorted(groups),launch_pid,launch_identity,wait_identity,completion_identity

def execute(args):
    need(Path.cwd().resolve()==ROOT,"Root repository execution")
    manifest,row,spec,storage,own,launch,wait,completed,pids,groups,launch_pid,li,wi,ci=collect(args.scope)
    # Exact known source-owned roots and groups must be absent, including the outer supervisor.
    observations=[absence(x) for x in pids];group_observations=[absence(x,True) for x in groups]
    components=manifest["components"];sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    ms=importlib.util.spec_from_file_location("_VAL_FLARE_root980_close",ROOT/components["guard"]["path"])
    m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m)
    m.configure_v2(components["repair"],policy_ref=components["policy"])
    need(m.MODE=="prospective_980gb_v1" and all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Exact actual980 original unmemoized close API")
    folder=local(args.output_packet or HERE+"/closed_"+args.scope+"_v1");need(not folder.exists(),"Fresh close packet; preserve prior attempts")
    # Cooperate with the live wave admission lock without touching its source/configs.
    lock=ROOT/"research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock"
    with lock.open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        with m.budget.v4.mutex(ROOT):
            state=m._state_locked(components["budget"]["sha256"],components["launcher"]["sha256"],fresh=True)
            m.assert_open(own["key"],state[0],state[1]);need(state[0][own["key"]]==own,"Exact still-open original reservation")
        folder.mkdir()
        revoke=save(folder,"root_scope_revocation.json",dict(root_authorizer="root_agent",key=own["key"],
            all_original_input_config_storage_source_launch_resume_retry_write_capabilities_revoked=True,
            read_only_reuse_preserved=True,preserve_all_files=True,all_known_source_owned_root_worker_pids=pids,
            PID_observations=observations,group_observations=group_observations,
            actual_evidence_refs=row,root_close_helper_ref=ref(SOURCE),no_outer_remote_direct_Popen_wait_claim=True,
            simulation_unseeded_RSS_zero_is_invalid=bool(row.get("RSS_correction_ref"))))
        observation=save(folder,"root_observation.json",dict(schema="concurrent_completed_scope_targeted_root_observation_v1",
            root_authorizer="root_agent",key=own["key"],reservation_sha256=m.budget.digest(own),
            storage_ref=row["storage_ref"],launch_ref=row["launch_ref"],direct_wait_ref=row["direct_wait_ref"],
            completed_ref=row["completed_ref"],launcher_pid=launch[launch_pid],
            all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
            completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
            external_NFS_writers_outside_observation=True,pids=pids,observations=[absence(x) for x in pids],
            worker_group_observations=[absence(x,True) for x in groups],scope_revocation_ref=revoke,
            actual_evidence_refs=row,root_close_helper_ref=ref(SOURCE)))
        request=dict(key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=row["storage_ref"],
            storage_locator=row["storage_locator"],launch_ref=row["launch_ref"],launch_pid_locator=[launch_pid],
            launch_identity=li,direct_wait_ref=row["direct_wait_ref"],wait_pid_locator=[launch_pid],wait_exit_locator=["returncode"],
            wait_identity=wi,completed_ref=row["completed_ref"],completed_identity=ci,observation_ref=observation)
        with m.budget.v4.mutex(ROOT):
            state=m._state_locked(components["budget"]["sha256"],components["launcher"]["sha256"],fresh=True)
            m.assert_open(own["key"],state[0],state[1]);need(state[0][own["key"]]==own,"Reservation unchanged before close")
            authority=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
        authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
        receipt=m.apply_close_completed_v1(save(folder,"close_authority.json",authority),
            expected_source_sha256=components["budget"]["sha256"],expected_launcher_sha256=components["launcher"]["sha256"])
        rr=save(folder,"close_receipt.json",receipt)
        print(json.dumps(dict(closed_key=own["key"],receipt_ref=rr,retained_bytes=receipt["retained_bytes"],
            credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"],files_deleted=0)),flush=True)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--scope",choices=SCOPES);p.add_argument("--output-packet");a=p.parse_args()
    if a.execute:return execute(a)
    scopes=[a.scope] if a.scope else SCOPES
    rows=[dict(scope=s,key=collect(s)[4]["key"],actual_completed_metadata_validated=True) for s in scopes]
    print(json.dumps(dict(status="source_only_metadata_review_no_guard_PID_probe_or_close",scopes=rows),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

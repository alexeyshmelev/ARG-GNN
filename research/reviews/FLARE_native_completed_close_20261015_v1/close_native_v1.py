"""Root-only close of one actual completed FLARE native980 scope; default reads metadata only."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_native_completed_close_20261015_v1"
SOURCE=HERE+"/close_native_v1.py"
REFS_REF={'path': 'research/reviews/FLARE_native_completed_close_20261015_v1/actual_refs.json', 'sha256': '483dae80464a3d6a3e63e744d7cbe0c24c5a92734ae46432aed222445906a8ae'}
SCOPES=("FLARE_native",)
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
    need(scope in SCOPES,"Only the actual completed FLARE native reservation")
    manifest=read(REFS_REF);components=manifest["components"];row=manifest["target"]
    need(all(ref(r["path"])==r for r in components.values()),"Exact current980 guard/policy/repair/core/launcher")
    need(ref(manifest["basis_helper_ref"]["path"])==manifest["basis_helper_ref"],"Frozen existing close pattern")
    for name,r in row.items():
        if name.endswith("_ref"):need(ref(r["path"])==r,"Every actual source/JSON evidence pin")
    spec=read(row["admission_ref"]);storage=read(row["storage_ref"]);own=storage["reservation"]
    cfg=read(row["config_ref"]);launch=read(row["launch_ref"]);wait=read(row["direct_wait_ref"])
    outer=read(row["outer_wait_ref"]);completed=read(row["completed_ref"]);native=read(row["native_wait_ref"])
    need(spec["stage"]=="native" and spec["GPU_authorized"] is False
        and spec["storage_ref"]==row["storage_ref"] and spec["config_ref"]==row["config_ref"]
        and spec["stage_source_ref"]==row["stage_source_ref"]
        and spec["job_id"]==storage["job_id"]==own["key"]=="FLARE/TRAIN000-native-980-root-v1"
        and storage["accounting_mode"]=="prospective_980gb_v1"
        and spec["completed_overlay_policy_ref"]==storage["completed_overlay_policy_ref"]==components["policy"]
        and storage["budget_components"]["consumer_adapter"]==components["guard"]
        and storage["reserved_new_bytes"]==own["reserved_bytes"]==row["cap"]==9000000000
        and storage["output_directory"]==spec["output_directory"]=="data/FLARE_train000_native_980_root_v1"
        and row["completed_ref"]["path"]==storage["output_directory"]+"/completed.json",
        "Exact original current980 native reservation and completed output")
    need(wait["actual_direct_wait"] is True and wait["returncode"]==outer["returncode"]==0
        and outer["error"] is None and outer["actual_wait_ref"]==row["direct_wait_ref"]
        and outer["observed_owned_groups_ESRCH"] is True and outer["child_compute_released"] is True
        and outer["launch_monitor_seed_ref"]==row["launch_ref"]
        and wait["admission_ref"]==outer["admission_ref"]==launch["admission_ref"]==row["admission_ref"]
        and wait["pid"]==wait["worker_pgid"]==launch["worker_pid"]==launch["worker_pgid"]==launch["worker_session"]
        and launch["actual_owned_roster_seeded"] is True
        and wait["root_stage_driver_ref"]==spec["root_stage_driver_ref"]==row["IO_source_ref"],
        "Actual seeded worker direct wait0, original supervisor and owned-group drain")
    need(native["actual_direct_native_wait"] is True and native["returncode"]==0
        and native["operational_failure"] is None and native["owned_process_group_ESRCH"] is True
        and completed["native_actual_wait_ref"]==row["native_wait_ref"]
        and native["native_argv_ref"]==completed["native_argv_ref"],
        "Actual successful native FLARE direct wait0 and group drain")
    need(completed["status"]=="completed_C3_native_FLARE" and completed["parent_id"]=="american_c3_train_000"
        and completed["query_ancestry_truth_opened"] is False and completed["native_outputs_modified"] is False
        and completed["author_Table1_exact_reproduction"] is False
        and completed["conversion_ref"]==cfg["conversion_ref"]
        and completed["focal_registry_ref"]==cfg["focal_registry_ref"]
        and completed["jar_ref"]==cfg["jar_ref"] and completed["native_options"]==cfg["native_options"]
        and cfg["scientific_role"]=="TRAIN_full30Mb_inference_canary_development_not_heldout",
        "Actual native TRAIN000 development completion with unchanged products/options; no ancestry score")
    pids={launch["supervisor_pid"],wait["pid"],*[int(x) for x in outer["observed_process_identities"]]}
    groups={launch["worker_pgid"],*outer["observed_owned_groups"]};launch_pid="worker_pid"
    need(len(pids)<=256 and all(type(x) is int and x>=2 for x in pids|groups),
        "Bounded exact known source-owned supervisor/worker/native roster")
    li={k:launch[k] for k in ("worker_pid","worker_start_ticks","worker_pgid","worker_session","supervisor_pid","admission_ref","actual_owned_roster_seeded")}
    wi={k:wait[k] for k in ("schema","actual_direct_wait","returncode","pid","worker_pgid","admission_ref","root_stage_driver_ref")}
    ci={k:completed[k] for k in ("status","parent_id","query_ancestry_truth_opened","native_outputs_modified","native_actual_wait_ref","conversion_ref")}
    return manifest,row,spec,storage,own,launch,wait,completed,sorted(pids),sorted(groups),launch_pid,li,wi,ci

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
            launch_identity=li,direct_wait_ref=row["direct_wait_ref"],wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
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

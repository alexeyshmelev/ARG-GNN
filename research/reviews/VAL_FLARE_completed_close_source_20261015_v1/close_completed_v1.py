"""Root close glue for three actual completed980 CPU scopes. Default reads metadata only."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/VAL_FLARE_completed_close_source_20261015_v1"
SOURCE=HERE+"/close_completed_v1.py"
REFS_REF={'path': 'research/reviews/VAL_FLARE_completed_close_source_20261015_v1/actual_refs.json', 'sha256': 'f6ef5ecabf450d94effce18e5665941a92cd9a9e414002d838ea036434daaadc'}
SCOPES=("VAL_simulate","VAL_infer","FLARE_convert")
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
    need(scope in SCOPES,"Only three actual completed scopes")
    manifest=read(REFS_REF);components=manifest["components"];row=manifest["targets"][scope]
    need(all(ref(r["path"])==r for r in components.values()),"Actual980 guard/policy/repair/core/launcher source pins")
    for name,r in row.items():
        if name.endswith("_ref"):need(ref(r["path"])==r,"Every actual source/evidence pin")
    spec=read(row["admission_ref"]);storage=read(row["storage_ref"])
    for key in row["storage_locator"]:storage=storage[key]
    own=storage["reservation"];launch=read(row["launch_ref"]);wait=read(row["direct_wait_ref"]);completed=read(row["completed_ref"])
    need(storage["accounting_mode"]=="prospective_980gb_v1" and storage["completed_overlay_policy_ref"]==components["policy"]
        and storage["budget_components"]["consumer_adapter"]==components["guard"]
        and storage["reserved_new_bytes"]==own["reserved_bytes"]==row["cap"]
        and row["completed_ref"]["path"]==storage["output_directory"]+"/completed.json"
        and spec["completed_overlay_policy_ref"]==components["policy"],"Exact original current980 storage/output/cap")
    if row["kind"]=="VAL_CPU":
        need(wait["actual_direct_worker_wait"] is True and wait["returncode"]==0
            and wait["owned_worker_group_ESRCH"] is True and wait["operational_failure"] is None
            and wait["parent_id"]=="american_c3_val_000" and wait["stage"]==row["stage"]
            and wait["admission_ref"]==launch["admission_ref"]==row["admission_ref"]
            and wait["completed_ref"]==row["completed_ref"] and wait["launch_ref"]==row["launch_ref"]
            and wait["minimal_wait_ref"]==row["minimal_wait_ref"],"Actual VAL direct wait0/drain and unchanged terminal lineage")
        minimal=read(row["minimal_wait_ref"])
        need(minimal["actual_direct_worker_wait"] is True and minimal["returncode"]==0
            and minimal["pid"]==wait["pid"]==launch["pid"]==launch["worker_pid"]==launch["process_group"]
            and minimal["admission_ref"]==row["admission_ref"]
            and launch["IO_supervisor_ref"]==wait["root_IO_source_ref"]==row["IO_source_ref"],
            "Actual direct Popen worker/minimal wait and pinned supervisor")
        pids={launch["supervisor_pid"],wait["pid"],*[int(x) for x in wait["observed_owned_pid_identities"]]}
        groups={launch["process_group"]};launch_pid="pid"
        launch_identity={k:launch[k] for k in ("pid","worker_pid","process_group","supervisor_pid","admission_ref","IO_supervisor_ref")}
        wait_identity={k:wait[k] for k in ("actual_direct_worker_wait","returncode","owned_worker_group_ESRCH","operational_failure","parent_id","stage","pid","admission_ref","completed_ref","minimal_wait_ref","root_IO_source_ref")}
        if scope=="VAL_simulate":
            need(completed["status"]=="completed_C3_simulation_reconstruction" and completed["parent_id"]=="american_c3_val_000"
                and completed["sample_unit"]=="haploid","Actual independent VAL simulation completion")
            correction=read(row["RSS_correction_ref"])
            need(correction["root_authorizer"]=="root_agent" and correction["pid"]==wait["pid"]
                and correction["original_v1_peak_RSS_field_is_invalid_not_measured_zero"] is True
                and correction["current_observation_is_not_full_run_peak"] is True
                and correction["original_direct_wait_deadline_and_private_work_guard_unaffected"] is True,
                "Preserve original unseeded-RSS correction; never treat zero as measured peak")
            pids.update(int(x) for x in correction["observed_identities"])
            completion_identity={k:completed[k] for k in ("status","parent_id","sample_unit","observation_ref")}
        else:
            need(completed["status"]=="completed_current_C3_inference" and completed["tsinfer_num_threads"]==8,
                "Actual completed unchanged VAL inference")
            completion_identity={k:completed[k] for k in ("status","tsinfer_num_threads","unary_preprocessing_policy","observation_ref")}
        obs=read(completed["observation_ref"])
        need(obs["parent_id"]=="american_c3_val_000" and obs["split"]=="val" and obs["sequence_length_bp"]==30000000
            and obs["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
            and obs["query_ancestry_truth_in_inputs"] is False,"Original full30Mb1600-haplotype pre-truth observation")
    else:
        outer=read(row["outer_wait_ref"])
        need(wait["actual_direct_wait"] is True and wait["returncode"]==outer["returncode"]==0
            and outer["error"] is None and outer["actual_wait_ref"]==row["direct_wait_ref"]
            and outer["observed_owned_groups_ESRCH"] is True and outer["child_compute_released"] is True
            and outer["launch_monitor_seed_ref"]==row["launch_ref"]
            and wait["admission_ref"]==outer["admission_ref"]==launch["admission_ref"]==row["admission_ref"]
            and wait["pid"]==wait["worker_pgid"]==launch["worker_pid"]==launch["worker_pgid"]==launch["worker_session"]
            and launch["actual_owned_roster_seeded"] is True
            and wait["root_stage_driver_ref"]==spec["root_stage_driver_ref"]==row["IO_source_ref"],
            "Actual FLARE seeded Popen/direct wait0 and outer group drain")
        need(completed["status"]=="completed_C3_FLARE_observation_conversion"
            and completed["parent_id"]=="american_c3_train_000" and completed["query_ancestry_truth_opened"] is False
            and completed["reference_haplotypes"]==600 and completed["query_haplotypes"]==1000
            and completed["common_focal_records"]==10000 and completed["common_focal_registry_unchanged"] is True,
            "Actual successful original TRAIN conversion; no scoring claim")
        pids={launch["supervisor_pid"],wait["pid"],*[int(x) for x in outer["observed_process_identities"]]}
        groups={launch["worker_pgid"],*outer["observed_owned_groups"]};launch_pid="worker_pid"
        launch_identity={k:launch[k] for k in ("worker_pid","worker_start_ticks","worker_pgid","worker_session","supervisor_pid","admission_ref","actual_owned_roster_seeded")}
        wait_identity={k:wait[k] for k in ("schema","actual_direct_wait","returncode","pid","worker_pgid","admission_ref","root_stage_driver_ref")}
        completion_identity={k:completed[k] for k in ("status","parent_id","observation_ref","focal_registry_ref","input_sites","reference_haplotypes","query_haplotypes","common_focal_records","query_ancestry_truth_opened")}
    need(len(pids)<=256 and all(type(x) is int and x>=2 for x in pids|groups),"Complete bounded known root/worker roster")
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

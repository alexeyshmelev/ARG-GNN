"""Root-only close IO for one genuine future980 short-wave stage. Default is inert."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/American_C3_short_train2mb_wave_980_close_source_v1"
SOURCE=HERE+"/close_stage_v1.py"
PINS_REF={'path': 'research/reviews/American_C3_short_train2mb_wave_980_close_source_v1/frozen_refs.json', 'sha256': 'a543183ee664e3217d437bf3c037bd494b7e26ad813822515220e7f6f837931a'}
WAVE="research/reviews/American_C3_short_train2mb_wave_980_source_v1"
BASE=ROOT/WAVE/"admitted"
PARENTS=tuple("american_c3_short_train_%03d"%i for i in range(14,20))
STAGES=("simulate","infer")
def need(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project-only path")
    return p
def ref(value):
    p=local(value)
    need(p.suffix in (".py",".json") and p.stat().st_size<=262144,"Bounded source/JSON evidence only")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    p=local(record["path"])
    need(set(record)=={"path","sha256"} and p.suffix==".json" and p.stat().st_size<=131072 and ref(p)==record,
        "Exact bounded JSON evidence")
    return json.loads(p.read_text())
def save(folder,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(raw)<=65536 and Path(name).name==name,"Compact flat root evidence")
    fd=os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return ref(folder/name)
def absence(pid,group=False):
    need(type(pid) is int and pid>=2,"Exact known root PID/group")
    try:
        if group:os.killpg(pid,0)
        else:os.kill(pid,0)
    except ProcessLookupError:
        return dict(**({"process_group":pid} if group else {"pid":pid}),absent=True,
            method="targeted os.killpg(group,0) ESRCH" if group else "targeted os.kill(pid,0) ESRCH",
            checked_utc=datetime.now(timezone.utc).isoformat())
    raise ValueError("Original PID/group is present, reused, zombie or inaccessible")
def pinned():
    pins=read(PINS_REF)["refs"]
    need(all(ref(r["path"])==r for r in pins.values()),"Every frozen source/guard/policy/repair pin")
    return pins
def input_ref(args,name):
    value=getattr(args,name);digest=getattr(args,"expected_"+name+"_sha256")
    need(value and digest,"Actual pinned "+name+" JSON required")
    r=dict(path=local(value).relative_to(ROOT).as_posix(),sha256=digest);read(r);return r
def collect_pair_v1(parent,launch_ref,terminal_ref,pins):
    index=int(parent[-3:]);first=index-index%2
    pair=["american_c3_short_train_%03d"%i for i in (first,first+1)]
    launch=read(launch_ref);terminal=read(terminal_ref)
    need(launch["kind"]=="actual_exec_session_launch_identity" and launch["root_authorizer"]=="root_agent"
        and launch["driver_ref"]==pins["wave_driver"] and launch["parent_ids"]==pair
        and type(launch["exec_session_id"]) is int and launch["exec_session_id"]>0
        and str(launch["remote_supervisor_start_ticks"]).isdigit()
        and int(launch["remote_supervisor_start_ticks"])>0
        and launch["all_additional_root_owned_remote_processes_listed"] is True,
        "Authenticated actual remote launch PID/start/group and complete additional root roster")
    need(terminal["kind"]=="actual_exec_session_terminal" and terminal["root_authorizer"]=="root_agent"
        and terminal["launch_identity_ref"]==launch_ref and terminal["driver_ref"]==pins["wave_driver"]
        and terminal["parent_ids"]==pair and terminal["exec_session_id"]==launch["exec_session_id"]
        and terminal["actual_exec_session_terminal_observed"] is True
        and terminal["remote_cli_status_propagated_by_ssh"] is True
        and terminal["remote_direct_Popen_wait_observed"] is False
        and terminal["tool_result"]["exit_code"]==0 and isinstance(terminal["tool_result"]["output"],str),
        "Genuine SSH-propagated exec-session terminal0; never claim remote direct Popen.wait")
    supervisor=launch["remote_supervisor_pid"]
    extra=launch["additional_root_owned_remote_pids"]
    need(type(supervisor) is int and supervisor>=2 and isinstance(extra,list)
        and all(type(x) is int and x>=2 for x in extra),"Exact complete known root PID list")
    pids={supervisor,*extra};groups={launch["remote_supervisor_process_group"]};rows=[]
    for p in pair:
        parent_done_ref=ref(BASE/p/"parent_preparation_completed.json");parent_done=read(parent_done_ref)
        need(parent_done["status"]=="completed_C3_2Mb_TRAIN_parent_preparation" and parent_done["parent_id"]==p
            and parent_done["truth_kept_out_of_model_inputs"] is True,"Both actual complete unchanged parents required")
        for stage in STAGES:
            folder=BASE/p/stage
            ar,lr,wr=ref(folder/"admission.json"),ref(folder/"launch.json"),ref(folder/"actual_wait.json")
            spec,child,wait=read(ar),read(lr),read(wr);own=spec["storage"]["reservation"]
            completed_ref=wait["completed_ref"];completed=read(completed_ref)
            expected_output="experiments/results/American_C3_short_train2mb_wave_v2/"+p+("/simulation" if stage=="simulate" else "/inference")
            expected_key="H041/American-C3-short2mb-"+p+"-"+stage+"-v2"
            need(spec["driver_ref"]==pins["wave_driver"] and spec["storage_adapter_ref"]==pins["guard"]
                and spec["completed_overlay_policy_ref"]==pins["policy"] and spec["parent_id"]==p and spec["stage"]==stage
                and spec["output_directory"]==expected_output and own["key"]==expected_key
                and spec["storage"]["accounting_mode"]=="prospective_980gb_v1"
                and own["reserved_bytes"]==(1000000000 if stage=="simulate" else 2000000000)
                and child["admission_ref"]==wait["admission_ref"]==ar and child["no_helper_launcher"] is True,
                "Exact unchanged future980 stage source/reservation/output/cap")
            need(child["pid"]==wait["pid"]==wait["worker_pid"]==wait["worker_process_group"]==child["worker_process_group"]
                and child["supervisor_pid"]==wait["supervisor_pid"]==supervisor
                and wait["driver_ref"]==pins["wave_driver"] and wait["parent_id"]==p and wait["stage"]==stage
                and wait["returncode"]==0 and wait["actual_direct_worker_wait"] is True
                and wait["worker_process_group_drained"] is True and wait["operational_failure"] is None
                and wait["supervisor_capture_error"] is None and wait["terminal_log_materialization_error"] is None,
                "Genuine per-stage direct worker wait0/drain and same remote supervisor")
            need(parent_done["stages"][stage]==dict(wait_ref=wr,completion_ref=completed_ref)
                and completed_ref["path"]==expected_output+"/completed.json"
                and child["owned_process_roster"]==wait["owned_process_roster"]
                and {r["pid"] for r in wait["owned_process_roster"]}=={supervisor,wait["pid"]},
                "Exact same complete stage lineage and original source-owned roster")
            obs=read(completed["observation_ref"])
            need(obs["parent_id"]==p and obs["split"]=="train" and obs["sequence_length_bp"]==2000000
                and obs["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
                and obs["query_ancestry_truth_in_inputs"] is False,"Original2Mb1600-haplotype TRAIN observation only")
            if stage=="simulate":
                need(completed["status"]=="completed_C3_simulation_reconstruction" and completed["parent_id"]==p
                    and completed["sample_unit"]=="haploid","Actual unchanged simulation completion")
            else:
                need(completed["status"]=="completed_current_C3_inference" and completed["tsinfer_num_threads"]==8
                    and obs["sites_and_sample_identities_verified"] is True,"Actual unchanged verified inference completion")
            pids.update(r["pid"] for r in wait["owned_process_roster"]);groups.add(wait["worker_process_group"])
            rows.append((p,stage,ar,lr,wr,completed_ref,spec,child,wait,completed,own))
    need(len(pids)<=256 and all(type(x) is int and x>=2 for x in groups),"Bounded complete root PID/group roster")
    return rows,sorted(pids),sorted(groups)
def run(args):
    need(Path.cwd().resolve()==ROOT and args.parent in PARENTS and args.stage in STAGES,"Exact future980 scope selected by root")
    pins=pinned();launch_ref=input_ref(args,"exec_launch");terminal_ref=input_ref(args,"exec_terminal")
    rows,pids,groups=collect_pair_v1(args.parent,launch_ref,terminal_ref,pins)
    selected=next(r for r in rows if r[0]==args.parent and r[1]==args.stage)
    parent,stage,ar,lr,wr,cr,spec,child,wait,completed,own=selected
    # Both host/root transcript exit0 and exact local root/worker absence are mandatory.
    observed=[absence(x) for x in pids];group_observed=[absence(x,True) for x in groups]
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    ms=importlib.util.spec_from_file_location("_C3_wave980_root_close_guard",ROOT/pins["guard"]["path"])
    m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m)
    m.configure_v2(pins["repair"],policy_ref=pins["policy"])
    need(m.MODE=="prospective_980gb_v1" and all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Actual980 original unmemoized close semantics")
    directory=local(args.output_packet or HERE+"/closed_"+parent[-3:]+"_"+stage+"_v1")
    need(not directory.exists(),"Fresh per-scope close evidence; never overwrite partial attempts")
    lock=ROOT/"research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock"
    with lock.open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        with m.budget.v4.mutex(ROOT):
            state=m._state_locked(pins["budget"]["sha256"],pins["base_launcher"]["sha256"],fresh=True)
            m.assert_open(own["key"],state[0],state[1]);need(state[0][own["key"]]==own,"Exact original fully charged reservation")
        directory.mkdir()
        revoke=save(directory,"scope_write_resume_revocation.json",dict(root_authorizer="root_agent",key=own["key"],
            old_input_config_storage_source_launch_resume_retry_write_capabilities_revoked=True,read_only_reuse_preserved=True,
            preserve_all_files=True,exec_launch_ref=launch_ref,actual_exec_session_terminal_ref=terminal_ref,
            remote_direct_Popen_wait_observed=False,complete_original_roster_from_pinned_source=True,
            pids=pids,observations=observed,worker_and_supervisor_group_observations=group_observed))
        observation=save(directory,"root_observation.json",dict(schema="concurrent_completed_scope_targeted_root_observation_v1",
            root_authorizer="root_agent",key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=ar,
            launch_ref=lr,direct_wait_ref=wr,completed_ref=cr,launcher_pid=child["pid"],
            all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
            completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
            external_NFS_writers_outside_observation=True,pids=pids,observations=[absence(x) for x in pids],
            worker_group_observations=[absence(x,True) for x in groups],scope_revocation_ref=revoke,
            actual_exec_session_terminal_ref=terminal_ref,remote_direct_Popen_wait_observed=False,root_close_helper_ref=ref(SOURCE)))
        launch_identity={k:child[k] for k in ("parent_id","stage","pid","worker_pid","worker_process_group","supervisor_pid","owned_process_roster","admission_ref","no_helper_launcher")}
        wait_identity={k:wait[k] for k in ("parent_id","stage","pid","worker_pid","worker_process_group","supervisor_pid","owned_process_roster","admission_ref","driver_ref","returncode","operational_failure","worker_process_group_drained","group_drain_check","actual_direct_worker_wait","supervisor_capture_error","terminal_log_materialization_error","root_must_observe_supervisor_exit_before_credit")}
        ci={k:completed[k] for k in (("status","parent_id","sample_unit") if stage=="simulate" else ("status","tsinfer_num_threads","unary_preprocessing_policy","observation_ref"))}
        request=dict(key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=ar,storage_locator=["storage"],
            launch_ref=lr,launch_pid_locator=["pid"],launch_identity=launch_identity,direct_wait_ref=wr,
            wait_pid_locator=["pid"],wait_exit_locator=["returncode"],wait_identity=wait_identity,
            completed_ref=cr,completed_identity=ci,observation_ref=observation)
        with m.budget.v4.mutex(ROOT):
            state=m._state_locked(pins["budget"]["sha256"],pins["base_launcher"]["sha256"],fresh=True)
            m.assert_open(own["key"],state[0],state[1]);need(state[0][own["key"]]==own,"Unchanged actual reservation")
            authority=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
        authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
        receipt=m.apply_close_completed_v1(save(directory,"close_authority.json",authority),
            expected_source_sha256=pins["budget"]["sha256"],expected_launcher_sha256=pins["base_launcher"]["sha256"])
        saved=save(directory,"close_receipt.json",receipt)
        print(json.dumps(dict(closed_key=own["key"],receipt_ref=saved,retained_bytes=receipt["retained_bytes"],
            credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"],files_deleted=0)),flush=True)
    return 0
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--parent",choices=PARENTS);p.add_argument("--stage",choices=STAGES);p.add_argument("--output-packet")
    for name in ("exec-launch","exec-terminal"):
        p.add_argument("--"+name);p.add_argument("--expected-"+name+"-sha256")
    args=p.parse_args()
    if args.execute:return run(args)
    print(json.dumps(dict(status="source_only_no_guard_observation_or_close",eligible_parent_ids=list(PARENTS),
        actual_exec_terminal_and_genuine_stage_direct_waits_required=True,remote_Popen_wait_claim_forbidden=True)));return 0
if __name__=="__main__":raise SystemExit(main())

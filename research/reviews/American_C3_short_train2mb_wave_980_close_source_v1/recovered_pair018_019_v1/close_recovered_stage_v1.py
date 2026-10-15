"""Root-only close of recovered018/019 stages; preserves old1 plus actual recovery0. Default inert."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/American_C3_short_train2mb_wave_980_close_source_v1/recovered_pair018_019_v1"
SOURCE=HERE+"/close_recovered_stage_v1.py"
RECOVERY_SOURCE_REF={'path': 'research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/root_train019_infer_recovery_v1.py', 'sha256': '5581b5f01f26558359dee354348abb3ad000dbb78d03622990b1378e926488cd'}
ACTUAL_RECOVERY_LAUNCH_REF={'path': 'research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/root_finish_exec_launch_identity_actual_v1.json', 'sha256': '68a197a78e1afd683bec5ae9d12ad8365db5a704e5245842c906c30fed478f39'}
ACTUAL_RECOVERY_TOOL_LAUNCH_REF={'path': 'research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/root_finish_exec_launch_identity_with_tool_session_actual_v1.json', 'sha256': 'ac8b4b07bfabf0dce3d8575f2794eb5d399e7115111c14aefc563037ab3a26a1'}
ACTUAL_RECOVERY_TRANSPORT_REF={'path': 'research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/root_finish_SSH_transport_failure_actual_v1.json', 'sha256': 'fa96ca9b0e4dd9dc55fb2a5186291c63a63aa6eb2e610643de1a626a13f19dd1'}
RECOVERY_PINS_REF={'path': 'research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/pins.json', 'sha256': '86948186cc804a552467a5b8f1480bd4a07ee3e823ef3da90387b5b258f41a08'}
PINS_REF={'path': 'research/reviews/American_C3_short_train2mb_wave_980_close_source_v1/frozen_refs.json', 'sha256': 'a543183ee664e3217d437bf3c037bd494b7e26ad813822515220e7f6f837931a'}
WAVE="research/reviews/American_C3_short_train2mb_wave_980_source_v1"
BASE=ROOT/WAVE/"admitted"
PARENTS=("american_c3_short_train_018","american_c3_short_train_019")
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
def collect_pair_v1(parent,launch_ref,terminal_ref,pins,recovery_launch_ref,recovery_terminal_ref,recovery_transport_ref):
    need(parent in PARENTS,"Only recovered018/019 pair")
    pair=["american_c3_short_train_018","american_c3_short_train_019"]
    recovery_pins=read(RECOVERY_PINS_REF)
    need(launch_ref==recovery_pins["original_pair_launch"]
        and terminal_ref==recovery_pins["original_pair_terminal"],"Exact original failed pair evidence preserved")
    launch=read(launch_ref);terminal=read(terminal_ref)
    def launch_valid(value,driver,parents,direct=False):
        need(value["kind"]=="actual_exec_session_launch_identity" and value["root_authorizer"]=="root_agent"
            and value["driver_ref"]==driver and value["parent_ids"]==parents
            and ((type(value.get("exec_session_id")) is int and value["exec_session_id"]>0)
                or (direct and value.get("exec_session_id_pending") is True and value.get("exec_session_id") is None))
            and str(value["remote_supervisor_start_ticks"]).isdigit()
            and int(value["remote_supervisor_start_ticks"])>0
            and value["all_additional_root_owned_remote_processes_listed"] is True,
            "Actual launch PID/start/group and complete additional owned roster")
    def terminal_valid(value,record,launch_value,driver,parents,code,direct=False):
        need(value["kind"]=="actual_exec_session_terminal" and value["root_authorizer"]=="root_agent"
            and value["launch_identity_ref"]==record and value["driver_ref"]==driver
            and value["parent_ids"]==parents and value["exec_session_id"]==launch_value["exec_session_id"]
            and value["actual_exec_session_terminal_observed"] is True
            and value["remote_cli_status_propagated_by_ssh"] is True
            and value["remote_direct_Popen_wait_observed"] is direct
            and value["tool_result"]["exit_code"]==code and isinstance(value["tool_result"]["output"],str),
            "Genuine SSH terminal code must match its own launch; no invented direct Popen.wait")
    launch_valid(launch,pins["wave_driver"],pair)
    terminal_valid(terminal,launch_ref,launch,pins["wave_driver"],pair,1)
    need(terminal.get("whole_pair_completion_claimed") is False,"Original terminal1 is not a complete pair")
    need(recovery_launch_ref==ACTUAL_RECOVERY_LAUNCH_REF
        and recovery_transport_ref==ACTUAL_RECOVERY_TRANSPORT_REF,"Actual existing immutable recovery launch/transport refs")
    recovery_launch=read(recovery_launch_ref);recovery_terminal=read(recovery_terminal_ref)
    need(ref(RECOVERY_SOURCE_REF["path"])==RECOVERY_SOURCE_REF,"Sealed inference recovery source")
    launch_valid(recovery_launch,RECOVERY_SOURCE_REF,[pair[1]],direct=True)
    need(recovery_terminal["launch_identity_ref"]==recovery_launch_ref
        and recovery_terminal["driver_ref"]==RECOVERY_SOURCE_REF
        and recovery_terminal["actual_remote_direct_Popen_wait_observed"] is True
        and recovery_terminal["remote_direct_Popen_wait_observed"] is True
        and recovery_terminal["actual_supervisor_pid"]==recovery_launch["remote_supervisor_pid"]
        and type(recovery_terminal["actual_remote_direct_Popen_returncode"]) is int
        and recovery_terminal["actual_remote_direct_Popen_returncode"]==0
        and recovery_terminal["root_wrapper_pid"] in recovery_launch["additional_root_owned_remote_pids"],
        "Genuine existing outer direct Popen.wait0 schema bound to helper and wrapper PIDs")
    if "parent_ids" in recovery_terminal:
        need(recovery_terminal["parent_ids"]==[pair[1]],"Actual direct-wait parent IDs when recorded")
    tool_launch=read(ACTUAL_RECOVERY_TOOL_LAUNCH_REF)
    need(tool_launch["exec_session_id_pending"] is False
        and type(tool_launch["exec_session_id"]) is int and tool_launch["exec_session_id"]>0,
        "Actual attached transport toolsession identity")
    for key,value in recovery_launch.items():
        if key!="exec_session_id_pending":need(tool_launch[key]==value,"Immutable launch identity and toolsession attachment agree")
    transport=read(recovery_transport_ref)
    need(transport["kind"]=="actual_SSH_transport_terminal"
        and transport["exec_session_id"]==tool_launch["exec_session_id"]
        and transport["actual_tool_exit_code"]==255 and transport["remote_helper_returncode"] is None
        and isinstance(transport["transport_error"],str) and bool(transport["transport_error"])
        and transport["remote_direct_wait_observed_by_root"] is False and transport["no_restart"] is True,
        "Actual SSH255/unknown remote status at transport loss preserved")
    helper_observed=transport["existing_remote_helper_observed_alive"]
    wrapper_observed=transport["existing_remote_wrapper_observed_alive"]
    need(helper_observed["pid"]==recovery_launch["remote_supervisor_pid"]
        and helper_observed["start_ticks"]==recovery_launch["remote_supervisor_start_ticks"],
        "Actual helper identity alive at transport-loss observation")
    wrapper=next(r for r in recovery_launch["additional_root_owned_remote_process_identity_evidence"]
        if r["pid"]==recovery_terminal["root_wrapper_pid"])
    need(wrapper_observed["pid"]==wrapper["pid"] and wrapper_observed["start_ticks"]==wrapper["start_ticks"],
        "Actual wrapper identity alive at transport-loss observation")
    supervisor=launch["remote_supervisor_pid"];recovery_supervisor=recovery_launch["remote_supervisor_pid"]
    need(supervisor!=recovery_supervisor,"Distinct failed and recovery supervisors")
    call_ref=ref(ROOT/WAVE/"train019_infer_recovery_v1/root_recovery_finish_actual_terminal_v1.json")
    call=read(call_ref)
    need(call["schema"]=="TRAIN019_infer_recovery_actual_root_CLI_terminal_v1"
        and call["mode"]=="finish" and call["source_ref"]==RECOVERY_SOURCE_REF
        and call["root_pid"]==recovery_supervisor and call["returncode"]==0 and call["error"] is None
        and call["actual_recovery_call_completed"] is True
        and call["original_pair_terminal1_preserved"] is True
        and call["original_exception_cause"] is None
        and call["no_original_scientific_completion_replayed"] is True,
        "Actual recovery finish0 under its authenticated remote supervisor")
    parent_done=read(ref(BASE/pair[1]/"parent_preparation_completed.json"))
    need(call["result"]==parent_done,"Recovery result matches genuine completed019parent")
    revoke_ref=ref(ROOT/WAVE/"train019_infer_recovery_v1/root_old_authority_cooperative_revocation_v1.json")
    revoked=read(revoke_ref)
    need(revoked["active"] is True and revoked["old_authority_ref"]==recovery_pins["original_authority"]
        and revoked["old_authority_must_never_be_applied"] is True
        and revoked["original_returncode"]==1 and revoked["root_cooperative_revocation_only"] is True
        and revoked["recovery_source_ref"]==RECOVERY_SOURCE_REF,"Original failed authority remains cooperatively revoked")
    need(ref(recovery_pins["original_authority"]["path"])==recovery_pins["original_authority"],
        "Original authority bytes preserved")
    for record in recovery_pins["preserved_successful_waits"].values():
        need(ref(record["path"])==record,"Three original successful stage waits byte-identical")
    pids=set();groups=set();rows=[]
    for item in [launch,recovery_launch]:
        extra=item["additional_root_owned_remote_pids"]
        identities=item["additional_root_owned_remote_process_identity_evidence"]
        need(isinstance(extra,list) and len(set(extra))==len(extra)
            and {r["pid"] for r in identities}==set(extra),"Complete additional PID/group identity list")
        pids.update([item["remote_supervisor_pid"],*extra])
        groups.add(item["remote_supervisor_process_group"])
        for r in identities:
            need(type(r["pid"]) is int and r["pid"]>=2 and int(r["start_ticks"])>0
                and type(r["process_group"]) is int and r["process_group"]>=2,"Actual additional PID/start/group")
            groups.add(r["process_group"])
    need(all(type(x) is int and x>=2 for x in pids),"Exact complete known root PID list")
    context=dict(original_pair_launch_ref=launch_ref,original_pair_terminal_ref=terminal_ref,
        original_pair_returncode=1,original_remote_direct_Popen_wait_observed=False,recovery_exec_launch_ref=recovery_launch_ref,
        recovery_direct_Popen_terminal_ref=recovery_terminal_ref,recovery_returncode=0,recovery_remote_direct_Popen_wait_observed=True,
        recovery_SSH_transport_terminal_ref=recovery_transport_ref,recovery_SSH_transport_returncode=255,remote_exit_known_from_transport=False,
        recovery_tool_session_launch_attachment_ref=ACTUAL_RECOVERY_TOOL_LAUNCH_REF,
        actual_recovery_finish_call_terminal_ref=call_ref,old_authority_cooperative_revocation_ref=revoke_ref,
        original_exception_cause=None,recovery_source_ref=RECOVERY_SOURCE_REF,
        three_original_waits_unchanged=True,no_completed_science_rerun=True)
    for p in pair:
        parent_done_ref=ref(BASE/p/"parent_preparation_completed.json");parent_done=read(parent_done_ref)
        need(parent_done["status"]=="completed_C3_2Mb_TRAIN_parent_preparation" and parent_done["parent_id"]==p
            and parent_done["truth_kept_out_of_model_inputs"] is True,"Both actual complete unchanged parents required")
        expected_parent_supervisor=recovery_supervisor if p.endswith("019") else supervisor
        need(parent_done["root_supervisor_pid"]==expected_parent_supervisor,"Exact genuine parent completion supervisor")
        for stage in STAGES:
            expected_supervisor=recovery_supervisor if p.endswith("019") and stage=="infer" else supervisor
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
                and child["supervisor_pid"]==wait["supervisor_pid"]==expected_supervisor
                and wait["driver_ref"]==pins["wave_driver"] and wait["parent_id"]==p and wait["stage"]==stage
                and wait["returncode"]==0 and wait["actual_direct_worker_wait"] is True
                and wait["worker_process_group_drained"] is True and wait["operational_failure"] is None
                and wait["supervisor_capture_error"] is None and wait["terminal_log_materialization_error"] is None,
                "Genuine per-stage direct worker wait0/drain and same remote supervisor")
            need(parent_done["stages"][stage]==dict(wait_ref=wr,completion_ref=completed_ref)
                and completed_ref["path"]==expected_output+"/completed.json"
                and child["owned_process_roster"]==wait["owned_process_roster"]
                and {r["pid"] for r in wait["owned_process_roster"]}=={expected_supervisor,wait["pid"]},
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
    return rows,sorted(pids),sorted(groups),context

def run(args):
    need(Path.cwd().resolve()==ROOT and args.parent in PARENTS and args.stage in STAGES,"Exact future980 scope selected by root")
    pins=pinned();launch_ref=input_ref(args,"exec_launch");terminal_ref=input_ref(args,"exec_terminal")
    recovery_launch_ref=input_ref(args,"recovery_exec_launch");recovery_terminal_ref=input_ref(args,"recovery_exec_terminal")
    recovery_transport_ref=input_ref(args,"recovery_transport_terminal")
    rows,pids,groups,context=collect_pair_v1(args.parent,launch_ref,terminal_ref,pins,recovery_launch_ref,recovery_terminal_ref,recovery_transport_ref)
    selected=next(r for r in rows if r[0]==args.parent and r[1]==args.stage)
    parent,stage,ar,lr,wr,cr,spec,child,wait,completed,own=selected
    recovery_direct=parent.endswith("019") and stage=="infer"
    selected_root_launch_ref=recovery_launch_ref if recovery_direct else launch_ref
    selected_root_terminal_ref=recovery_terminal_ref if recovery_direct else terminal_ref
    # Original terminal1 plus actual recovery finish0 and both complete owned rosters are mandatory.
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
            preserve_all_files=True,exec_launch_ref=selected_root_launch_ref,actual_exec_session_terminal_ref=selected_root_terminal_ref,
            remote_direct_Popen_wait_observed=recovery_direct,complete_original_roster_from_pinned_source=True,
            pids=pids,observations=observed,worker_and_supervisor_group_observations=group_observed,recovered_pair_context=context))
        observation=save(directory,"root_observation.json",dict(schema="concurrent_completed_scope_targeted_root_observation_v1",
            root_authorizer="root_agent",key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=ar,
            launch_ref=lr,direct_wait_ref=wr,completed_ref=cr,launcher_pid=child["pid"],
            all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
            completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
            external_NFS_writers_outside_observation=True,pids=pids,observations=[absence(x) for x in pids],
            worker_group_observations=[absence(x,True) for x in groups],scope_revocation_ref=revoke,
            actual_exec_session_terminal_ref=selected_root_terminal_ref,remote_direct_Popen_wait_observed=recovery_direct,root_close_helper_ref=ref(SOURCE),recovered_pair_context=context))
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
    for name in ("exec-launch","exec-terminal","recovery-exec-launch","recovery-exec-terminal","recovery-transport-terminal"):
        p.add_argument("--"+name);p.add_argument("--expected-"+name+"-sha256")
    args=p.parse_args()
    if args.execute:return run(args)
    print(json.dumps(dict(status="source_only_no_guard_observation_or_close",eligible_parent_ids=list(PARENTS),
        original_terminal1_plus_direct_recovery_wait0_and_transport255_required=True,actual_exec_terminal_and_genuine_stage_direct_waits_required=True,recovery_genuine_direct_wait0_required=True)));return 0
if __name__=="__main__":raise SystemExit(main())

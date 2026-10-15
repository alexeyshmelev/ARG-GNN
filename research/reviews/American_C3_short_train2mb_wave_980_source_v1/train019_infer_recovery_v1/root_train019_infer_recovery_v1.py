"""Root-only resume of never-admitted TRAIN019 infer; default is source report only.

Preserves original terminal1, partial config/authority and three successful stages.
No guard import/configuration, accounting or process probe occurs in default mode.
Only explicit root modes mutate admission metadata or launch unfinished inference.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, contextlib, copy, hashlib, importlib.util, io, json, os, sys, traceback

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_short_train2mb_wave_980_source_v1"
HERE=ROOT/REL/"train019_infer_recovery_v1"
RECOVERY_SOURCE=REL+"/train019_infer_recovery_v1/root_train019_infer_recovery_v1.py"
SOURCE=REL+"/root_wave_980_driver.py"
PIN_SHA="86948186cc804a552467a5b8f1480bd4a07ee3e823ef3da90387b5b258f41a08"
PARENT="american_c3_short_train_019"
STAGE="infer"
CONDITION="C3_independent20x2Mb_TRAIN_1600haps_inferred_unary_v1"
_PHASE="source_report"

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(".." not in p.parts and p.resolve()==p,"Exact nonsymlink local path")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record):
    require(ref(record["path"])==record,"Exact pinned metadata/source required")
    raw=(ROOT/record["path"]).read_bytes()
    require(len(raw)<=1048576,"Bounded source metadata")
    return json.loads(raw)

def save(folder,name,value):
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    require(len(data)<=65536,"Bounded root metadata")
    with (folder/name).open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    return ref(folder/name)

def read_pins():
    return read(dict(path=REL+"/train019_infer_recovery_v1/pins.json",sha256=PIN_SHA))

def original_driver(p):
    require(ref(p["original_driver"]["path"])==p["original_driver"],"Original driver frozen")
    spec=importlib.util.spec_from_file_location("_train019_original_wave",ROOT/p["original_driver"]["path"])
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def preserved_waits(p):
    values={name:read(record) for name,record in p["preserved_successful_waits"].items()}
    require(set(values)=={"018_simulate","018_infer","019_simulate"},"Exactly three genuine prior stages")
    for name,w in values.items():
        require(w["returncode"]==0 and w["actual_direct_worker_wait"] is True
            and w["worker_process_group_drained"] is True and w["operational_failure"] is None,
            "Preserve genuine direct wait0 and drain; never rerun completed science")
    return values

def require_original_root_absence(p,waits):
    launch=read(p["original_pair_launch"])
    terminal=read(p["original_pair_terminal"])
    require(terminal["actual_exec_session_terminal_observed"] is True
        and terminal["tool_result"]["exit_code"]==1
        and terminal["launch_identity_ref"]==p["original_pair_launch"],
        "Genuine original terminal1 must stay1")
    ids={launch["remote_supervisor_pid"],*launch["additional_root_owned_remote_pids"]}
    groups={launch["remote_supervisor_process_group"],
        *(r["process_group"] for r in launch["additional_root_owned_remote_process_identity_evidence"])}
    for w in waits.values():
        for row in w["owned_process_roster"]:
            ids.add(row["pid"])
            if "process_group" in row:groups.add(row["process_group"])
    observed=[]
    for pid in sorted(ids):
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError("Original owned PID still exists; root must audit/drain, no recovery")
        observed.append(dict(pid=pid,absent=True,method="kill0_ESRCH",checked_utc=datetime.now(timezone.utc).isoformat()))
    for group in sorted(groups):
        try:os.killpg(group,0)
        except ProcessLookupError:pass
        else:raise ValueError("Original owned group still exists; root must audit/drain, no recovery")
    return dict(PID_observations=observed,absent_process_groups=sorted(groups))

def admit_recovery():
    global _PHASE
    _PHASE="validate_original_partial_admission"
    require(Path.cwd().resolve()==ROOT,"Root must run in repository")
    parent,stage=PARENT,STAGE
    p=read_pins();m=original_driver(p);waits=preserved_waits(p)
    absent=require_original_root_absence(p,waits)
    old=read(p["original_authority"]);old_config=read(p["original_config"])
    observed=read(p["accounting_source_observation"])
    frozen_ref,frozen_pins=m.frozen()
    require(frozen_ref==p["original_frozen_refs"],"Original frozen source closure")
    row=m.registry_parent(PARENT,frozen_pins)
    policy_ref=frozen_pins["policy"]
    evidence=m.qualifier(frozen_pins)
    cfg_template,cfg,dependencies=m.activate_config(PARENT,STAGE,row,frozen_pins)
    packet,output,job,key=m.stage_identity(PARENT,STAGE,row)
    require(old_config==cfg and cfg["output_cap_bytes"]==2000000000,"Unchanged original inference recipe/cap")
    require(sorted(x.name for x in packet.iterdir())==["config.json","reserve_authority.json"],
        "Only original two-file partial packet may be resumed; no implicit retry")
    require(not m.local(output).exists() and not (packet/"worker_aux").exists(),"No prior inference output/worker")
    require(old["operation"]=="reserve" and old["request"]["key"]==key
        and old["request"]["output"]==str(ROOT/output)
        and old["request"]["auxiliary_scopes"]==[str(packet/"worker_aux")]
        and old["request"]["reserved_bytes"]==2000000000,"Exact old request identity")
    _PHASE="cooperative_revocation_and_fresh_admission"
    with m.root_lock("admission.lock"):
        guard=m.active980_guard_v1(policy_ref,frozen_pins)
        with guard.budget.v4.mutex(ROOT):
            state=guard._state_locked(frozen_pins["budget"]["sha256"],frozen_pins["launcher"]["sha256"],fresh=True)
        snapshot=observed["observed_prefix"]
        require(len(state[5])>=snapshot["events"] and state[5][snapshot["events"]-1]==snapshot["tail_sha256"],
            "Original observed append-only journal prefix must be retained")
        require(key not in state[0] and key not in state[1],"No original or duplicate reservation may exist")
        require(guard._pins(state[2])!=old["expected_before"],"Old failed authority is permanently stale")
        m.require_pair_predecessors_closed_v1(PARENT,frozen_pins,state)
        open_parents=set()
        for registered in m.read(frozen_pins["registry"])["parents"]:
            for which in m.STAGES:
                for suffix in ["v1","v2"]:
                    registered_key="H041/American-C3-short2mb-"+registered["parent_id"]+"-"+which+"-"+suffix
                    if registered_key in state[0] and registered_key not in state[1]:open_parents.add(registered["parent_id"])
                    require(registered["parent_id"]!=PARENT or suffix!="v1" or registered_key not in state[0],
                        "Never duplicate an earlier v1 admission")
        require(PARENT in open_parents or len(open_parents)<2,"Original at-most-two-parent accounting limit")
        revocation=m.save(HERE,"root_old_authority_cooperative_revocation_v1.json",dict(
            schema="root_failed_authority_cooperative_revocation_v1",root_authorizer="root_agent",active=True,
            old_authority_ref=p["original_authority"],old_bytes_preserved=True,old_authority_must_never_be_applied=True,
            original_pair_terminal_ref=p["original_pair_terminal"],original_returncode=1,
            original_exception_cause=None,no_old_key_reservation_in_fresh_state=True,
            fresh_prefix=guard._pins(state[2]),original_owned_absence=absent,
            recovery_source_ref=ref(RECOVERY_SOURCE),original_wave_admit_and_run_parent_disabled_until_recovery_admitted=True,
            root_cooperative_revocation_only=True,guard_policy_not_modified=True,
            created_utc=datetime.now(timezone.utc).isoformat()))
        request=copy.deepcopy(old["request"])
        consumers=request["consumer_bindings"]+[ref(RECOVERY_SOURCE),ref(HERE/"pins.json"),
            p["accounting_source_observation"],p["original_authority"],p["original_pair_terminal"],revocation]
        request["consumer_bindings"]=list({(r["path"],r["sha256"]):r for r in consumers}.values())
        require(len(request["consumer_bindings"])<=64,"Original direct-source binding limit")
        require(all(ref(r["path"])==r for r in request["consumer_bindings"]),"Every direct binding remains pinned")
        require(all(r in request["consumer_bindings"] for r in state[3].values()),"All actual guard bindings retained")
        expiry=(datetime.now(timezone.utc)+timedelta(hours=22)).isoformat()
        authority=guard.operation_draft_v1("reserve",request,expected_before=guard._pins(state[2]),bindings=state[3])
        authority.update(active=True,expires_at_utc=expiry)
        authority_ref=m.save(packet,"reserve_authority_recovery1.json",authority)
        _PHASE="apply_fresh_original_reserve"
        receipt=guard.apply_reserve_v1(authority_ref,expected_source_sha256=frozen_pins["budget"]["sha256"],
            expected_launcher_sha256=frozen_pins["launcher"]["sha256"])
        m.save(packet,"reserve_receipt.json",receipt)
        _PHASE="materialize_original_admission"
        pins=frozen_pins
        config_ref=p["original_config"]
        worker=pins["inference_worker"];runtime=pins["inference_runtime"]
        closure=[pins["common"],worker,pins["simulation_worker"]]
        storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
            accounting_mode=guard.MODE,job_id=job,output_directory=output,reserved_new_bytes=cfg["output_cap_bytes"],
            reservation=receipt["value"],expires_at_utc=expiry,completed_overlay_policy_ref=policy_ref,
            initial_ledger_pins=receipt["initial_ledger_pins"],
            budget_components=dict(implementation=pins["budget"],launcher=pins["launcher"],consumer_adapter=pins["overlay"]))
        spec=dict(root_authorizer="root_agent",hypothesis_id="H041",scientific_condition_id=CONDITION,
            job_id=job,execution_authorized=True,GPU_authorized=False,source_ref=worker,source_closure_refs=closure,
            config_ref=config_ref,runtime_ref=runtime,storage_adapter_ref=pins["overlay"],
            marker_repair_authority_ref=pins["repair_authority"],completed_overlay_policy_ref=policy_ref,
            qualification_evidence_refs=evidence,stage_dependencies=dependencies,driver_ref=ref(SOURCE),
            parent_id=parent,stage=stage,storage=storage,output_directory=output,
            created_at_utc=datetime.now(timezone.utc).isoformat(),expires_at_utc=expiry)
        admission_ref=m.save(packet,"admission.json",spec,guard,storage)
        admitted=dict(parent_id=parent,stage=stage,admission_ref=admission_ref,driver_ref=ref(SOURCE),policy_ref=policy_ref)
        m.save(packet,"admitted.json",admitted,guard,storage)
        return admitted

def finish_original_parent():
    global _PHASE
    _PHASE="launch_only_unfinished_original_infer"
    p=read_pins();m=original_driver(p);preserved_waits(p)
    revoked=read(ref(HERE/"root_old_authority_cooperative_revocation_v1.json"))
    require(revoked["active"] is True and revoked["old_authority_ref"]==p["original_authority"],"Old failed authority revoked")
    packet=ROOT/REL/"admitted"/PARENT/STAGE
    require((packet/"admitted.json").is_file() and not (packet/"launch.json").exists()
        and not (packet/"actual_wait.json").exists() and not (packet/"worker_aux").exists(),"One fresh inference launch only")
    _,frozen_pins=m.frozen()
    result=m.prepare_parent_v1(PARENT,frozen_pins["policy"])
    preserved_waits(p)
    return result

class BoundedCapture:
    def __init__(self,target):self.target=target;self.parts=[];self.retained=0;self.dropped=0
    def write(self,value):
        value=str(value);self.target.write(value);self.target.flush()
        raw=value.encode("utf-8",errors="replace");keep=raw[:max(0,16384-self.retained)]
        self.parts.append(keep);self.retained+=len(keep);self.dropped+=len(raw)-len(keep)
        return len(value)
    def flush(self):self.target.flush()
    def text(self):return b"".join(self.parts).decode("utf-8",errors="replace")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--admit-recovery",action="store_true")
    mode.add_argument("--finish-original-parent",action="store_true")
    mode.add_argument("--source-report",action="store_true")
    args=parser.parse_args()
    if not args.admit_recovery and not args.finish_original_parent:
        print(json.dumps(dict(status="inert_source_only",recovery_source_ref=ref(RECOVERY_SOURCE),
            pins_ref=ref(HERE/"pins.json"),parent=PARENT,stage=STAGE,execution=False,
            old_terminal_returncode=1,original_exception_unknown=True),indent=2,sort_keys=True))
        return 0
    mode_name="admit" if args.admit_recovery else "finish"
    terminal_name="root_recovery_"+mode_name+"_actual_terminal_v1.json"
    require(not (HERE/terminal_name).exists(),"Existing recovery terminal evidence; root must audit, never blind retry")
    out=BoundedCapture(sys.stdout);err=BoundedCapture(sys.stderr);rc=0;result=None;error=None
    with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
        try:result=admit_recovery() if args.admit_recovery else finish_original_parent()
        except BaseException as exc:
            rc=1;traceback.print_exc()
            error=dict(type=type(exc).__name__,message=str(exc)[:4096],traceback=traceback.format_exc()[-16384:])
    record=dict(schema="TRAIN019_infer_recovery_actual_root_CLI_terminal_v1",mode=mode_name,
        source_ref=ref(RECOVERY_SOURCE),root_pid=os.getpid(),actual_recovery_call_completed=True,
        returncode=rc,phase=_PHASE,error=error,result=result,
        stdout_prefix=out.text(),stderr_prefix=err.text(),stdout_dropped_bytes=out.dropped,stderr_dropped_bytes=err.dropped,
        original_pair_terminal1_preserved=True,original_exception_cause=None,
        no_original_scientific_completion_replayed=True,finished_utc=datetime.now(timezone.utc).isoformat())
    terminal_ref=save(HERE,terminal_name,record)
    print(json.dumps(dict(returncode=rc,phase=_PHASE,durable_terminal_ref=terminal_ref),sort_keys=True))
    return rc

if __name__=="__main__":raise SystemExit(main())

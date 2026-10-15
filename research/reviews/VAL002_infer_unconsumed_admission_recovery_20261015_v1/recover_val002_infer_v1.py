"""Fixed unconsumed VAL002 inference admission recovery. Default validates metadata only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,copy,fcntl,hashlib,importlib.util,json,os,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
RECOVERY=ROOT/'research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v1'
RECOVERY_SOURCE='research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v1/recover_val002_infer_v1.py'
PINS_REF=dict(path='research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v1/pins.json',sha256='a61429069472553d677d500ee0c52ec9c5c59ba3b8bac2448cc6db1cdd8c4bc5')
KEY="H041/American-C3-val002-full30mb-infer-shared-980-v1"
PHASE="source_metadata"
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and p.is_relative_to(ROOT),"Exact canonical project metadata")
    require(p.suffix in (".json",".py") and p.stat().st_size<=262144,"Bounded JSON/source only")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(ref(record["path"])==record,"Exact frozen source/metadata pin")
    raw=(ROOT/record["path"]).read_bytes();require(len(raw)<=131072,"Bounded JSON")
    return json.loads(raw)
def inspect():
    p=read(PINS_REF)
    require(ref(p["original_driver"]["path"])==p["original_driver"],"Original packet driver unchanged")
    ms=importlib.util.spec_from_file_location("_VAL002_original_packet_recovery",ROOT/p["original_driver"]["path"])
    m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m)
    m.select_parent("american_c3_val_002");pins=m.frozen()
    cfg=m.configuration("infer",completion_ref=p["simulation_completion"],wait_ref=p["simulation_wait"],active=True)
    require(read(p["original_config"])==cfg and cfg["output_cap_bytes"]==12500000000
        and read(p["original_contract"])==m.contract(),"Exact frozen science/contract/cap/seed/protocol")
    packet,output,job=m.identity("infer");old=read(p["original_authority"])
    require(sorted(x.name for x in packet.iterdir())==["config.json","reserve_authority.json","shared_contract.json"],
        "Only untouched three-file never-admitted packet; no implicit repeat")
    require(not m.local(output).exists() and not (packet/"worker_aux").exists(),"No inference output or worker scope")
    require(old["operation"]=="reserve" and old["request"]["key"]==KEY and old["request"]["output"]==str(ROOT/output)
        and old["request"]["auxiliary_scopes"]==[str(packet/"worker_aux")]
        and old["request"]["reserved_bytes"]==12500000000,"Exact original request")
    terminal=read(p["original_parent_terminal"]);wait=read(p["simulation_wait"]);closed=read(p["simulation_close"])
    require(terminal["actual_exec_session_id"]==92429 and terminal["actual_exec_session_terminal"] is True
        and terminal["actual_exec_session_exit_code"]==1 and terminal["next_inference_worker_not_started"] is True
        and terminal["simulate_close_ref"]==p["simulation_close"],"Genuine original parent terminal1 preserved")
    require(wait["actual_direct_worker_wait"] is True and wait["returncode"]==0 and wait["operational_failure"] is None
        and wait["owned_worker_group_ESRCH"] is True and wait["completed_ref"]==p["simulation_completion"]
        and closed["key"]==m.predecessor_key("infer"),"Successful closed simulation reused without replay")
    observed=read(p["accounting_source_observation"])
    require(observed["key_events"]==observed["frozen_base_key_events"]==[]
        and observed["whole_chain_contiguous_and_hashed"] is True and observed["original_prefix_verified"] is True
        and observed["journal_key"]==KEY and observed["original_authority_ref"]==p["original_authority"],
        "Plain exact journal metadata establishes never charged through observed prefix")
    return p,m,pins,cfg,packet,output,job,old,observed
def admit_recovery():
    global PHASE
    require(Path.cwd().resolve()==ROOT,"Exact repository cwd")
    PHASE="validate_unconsumed_partial"
    p,m,pins,cfg,packet,output,job,old,observed=inspect()
    with m.ADMISSION_LOCK.open("r+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        PHASE="fresh_original_guard_state"
        guard=m.load_guard_v1(consumer=False)
        with guard.budget.v4.mutex(ROOT):
            state=guard._state_locked(pins[m.BUDGET]["sha256"],pins[m.LAUNCHER]["sha256"],fresh=True)
        snapshot=observed["observed_prefix"]
        require(len(state[5])>=snapshot["events"] and state[5][snapshot["events"]-1]==snapshot["tail_sha256"],
            "Observed original append-only prefix retained")
        require(KEY not in state[0] and KEY not in state[1],"Never duplicate any current or closed reservation")
        require(m.predecessor_key("infer") in state[1],"Original successful simulation genuinely closed")
        require(guard._pins(state[2])!=old["expected_before"],"Old unconsumed draft permanently stale")
        PHASE="cooperative_old_authority_revocation"
        revocation=m.save(RECOVERY,"root_old_authority_cooperative_revocation_v1.json",dict(
            schema="root_failed_authority_cooperative_revocation_v1",root_authorizer="root_agent",active=True,
            old_authority_ref=p["original_authority"],old_bytes_preserved=True,old_authority_must_never_be_applied=True,
            original_parent_terminal_ref=p["original_parent_terminal"],original_returncode=1,original_exception_cause=None,
            no_old_key_reservation_in_fresh_state=True,fresh_prefix=guard._pins(state[2]),
            no_inference_worker_was_started=True,simulation_completion_and_charge_closure_preserved=True,
            recovery_source_ref=ref(RECOVERY_SOURCE),original_admit_infer_and_run_parent_disabled_until_recovery_admitted=True,
            root_cooperative_revocation_only=True,guard_policy_not_modified=True,no_remote_direct_Popen_wait_claim=True,
            created_utc=datetime.now(timezone.utc).isoformat()))
        request=copy.deepcopy(old["request"])
        additions=[ref(RECOVERY_SOURCE),PINS_REF,p["accounting_source_observation"],p["original_authority"],p["original_parent_terminal"],revocation]
        request["consumer_bindings"]=list({(r["path"],r["sha256"]):r for r in request["consumer_bindings"]+additions}.values())
        require(len(request["consumer_bindings"])<=64 and all(ref(r["path"])==r for r in request["consumer_bindings"]),
            "Existing direct binding bound; all original bindings unchanged")
        require(all(r in request["consumer_bindings"] for r in state[3].values()),"Actual current guard bindings retained")
        now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=22)).isoformat()
        authority=guard.operation_draft_v1("reserve",request,expected_before=guard._pins(state[2]),bindings=state[3])
        authority.update(active=True,expires_at_utc=expiry)
        authority_ref=m.save(packet,"reserve_authority_recovery1.json",authority)
        PHASE="apply_fresh_original_reserve_once"
        receipt=guard.apply_reserve_v1(authority_ref,expected_source_sha256=pins[m.BUDGET]["sha256"],expected_launcher_sha256=pins[m.LAUNCHER]["sha256"])
        m.save(packet,"reserve_receipt.json",receipt)
        PHASE="materialize_exact_original_admission"
        stage="infer";policy_ref=pins[m.POLICY];config_ref=p["original_config"];contract_ref=p["original_contract"]
        completion_ref=p["simulation_completion"];wait_ref=p["simulation_wait"]
        closure=m.source_closure(stage,pins);runtime=pins[m.INFER_RUNTIME]
        SOURCE=m.SOURCE;HERE=m.HERE;PARENT=m.PARENT;SEED_ROW=m.SEED_ROW
        PRODUCERS=m.PRODUCERS;OVERLAY=m.OVERLAY;REPAIR=m.REPAIR;BUDGET=m.BUDGET;LAUNCHER=m.LAUNCHER
        MEMO_REF=m.MEMO_REF;NAMESPACE_MEMO_REF=m.NAMESPACE_MEMO_REF;V4_REF=m.V4_REF;REGISTRY_REF=m.REGISTRY_REF
        save=m.save
        storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,
            job_id=job,output_directory=output,reserved_new_bytes=cfg["output_cap_bytes"],reservation=receipt["value"],
            expires_at_utc=expiry,completed_overlay_policy_ref=policy_ref,initial_ledger_pins=receipt["initial_ledger_pins"],
            budget_components=dict(implementation=pins[BUDGET],launcher=pins[LAUNCHER],consumer_adapter=pins[OVERLAY]))
        spec=dict(root_authorizer="root_agent",hypothesis_id="H041",job_id=job,parent_id=PARENT,split="val",stage=stage,
            execution_authorized=True,GPU_authorized=False,source_ref=pins[PRODUCERS[stage]],source_closure_refs=closure,
            config_ref=config_ref,runtime_ref=runtime,storage_adapter_ref=pins[OVERLAY],marker_repair_authority_ref=pins[REPAIR],
            completed_overlay_policy_ref=policy_ref,shared_evaluation_contract_ref=contract_ref,
            driver_ref=ref(SOURCE),storage=storage,output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry,
            upstream_completion_ref=completion_ref,upstream_actual_wait_ref=wait_ref,
            query_ancestry_truth_payload_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False,
            consumer_memo_ref=MEMO_REF,consumer_namespace_memo_ref=NAMESPACE_MEMO_REF,original_v4_ref=V4_REF,registry_ref=REGISTRY_REF,parent_seed_row=dict(SEED_ROW),
            worker_bootstrap_ref=ref(HERE/"worker_bootstrap_v1.py"),IO_supervisor_ref=ref(HERE/"root_execution_v1.py"),
            operational_source_manifest_ref=ref(HERE/"source_manifest.json"))
        admission_ref=save(packet,"admission.json",spec)
        admitted_ref=save(packet,"admitted.json",dict(stage=stage,parent_id=PARENT,admission_ref=admission_ref,driver_ref=ref(SOURCE)))
        PHASE="admitted_without_launch"
        return dict(admission_ref=admission_ref,admitted_ref=admitted_ref,reserve_authority_ref=authority_ref,
            projected_bytes=receipt["projected_bytes"],scientific_worker_launched=False)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--admit-recovery",action="store_true");args=parser.parse_args()
    if not args.admit_recovery:
        p,m,pins,cfg,packet,output,job,old,observed=inspect()
        print(json.dumps(dict(status="source_only_metadata_no_guard_probe_reserve_or_job",key=KEY,
            observed_prefix=observed["observed_prefix"],original_partial_bytes_preserved=True),sort_keys=True));return 0
    terminal=RECOVERY/"root_recovery_admit_actual_terminal_v1.json"
    require(not terminal.exists(),"Recovery already attempted; inspect exact evidence, no blind retry")
    rc=0;result=None;error=None
    try:result=admit_recovery()
    except BaseException as exc:
        rc=1;traceback.print_exc();error=dict(type=type(exc).__name__,message=str(exc)[:4096],traceback=traceback.format_exc()[-8192:])
    p=read(PINS_REF);m_spec=importlib.util.spec_from_file_location("_VAL002_terminal_save",ROOT/p["original_driver"]["path"])
    m=importlib.util.module_from_spec(m_spec);m_spec.loader.exec_module(m)
    rr=m.save(RECOVERY,terminal.name,dict(schema="VAL002_infer_recovery_actual_root_CLI_terminal_v1",
        source_ref=ref(RECOVERY_SOURCE),root_pid=os.getpid(),actual_recovery_call_completed=True,returncode=rc,phase=PHASE,
        error=error,result=result,original_parent_terminal1_preserved=True,original_exception_cause=None,
        no_original_scientific_completion_replayed=True,finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(returncode=rc,phase=PHASE,durable_terminal_ref=rr),sort_keys=True));return rc
if __name__=="__main__":raise SystemExit(main())

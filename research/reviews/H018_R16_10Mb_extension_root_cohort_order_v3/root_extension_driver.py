"""Root-only registered 10Mb real-observation extraction with a 2400s group deadline."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,selectors,signal,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/H018_R16_10Mb_extension_root_cohort_order_v3"
HERE=ROOT/REL
SOURCE=REL+"/root_extension_driver.py"
PLAN="research/reviews/H018_R16_10Mb_extension_source_v1"
WORKER="src/arggnn/prepare_h018_r16_10mb_extension_v2.py"
POLICY={"path":"research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json","sha256":"a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85"}
GUARD={"path":"research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py","sha256":"ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef"}
REPAIR={"path":"research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json","sha256":"f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"}
WAIT_HELPER="research/reviews/American_C3_observation_supervisor_repair_source_v1/root_observation_driver.py"
WAIT_SHA="8e282edad3253a0e1d32807a94104129a7593fab3010b8a00b9f5526f8a8fe93"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
OLD_REL="research/reviews/H018_R16_10Mb_extension_root_v1"
OLD_OUTPUT="data/H018_R16_20_30Mb_extension_v1"
OUTPUT="data/H018_R16_20_30Mb_extension_cohort_order_v3"
OLD_DRIVER_REF={'path': 'research/reviews/H018_R16_10Mb_extension_root_v1/root_extension_driver.py', 'sha256': '5f3721ba37115d5cf966c5420d49c9404232523af010ff6ce1862120c56ff2a8'}
OLD_WORKER_REF={'path': 'src/arggnn/prepare_h018_r16_10mb_extension_v1.py', 'sha256': '4c97c6297b12c8e3bed5a3bb44bcdd912f6ee4bd00d8b88fb962c99f11330f49'}
WORKER_REF={'path': 'src/arggnn/prepare_h018_r16_10mb_extension_v2.py', 'sha256': 'e152bd3cdd4c50ac31ca40f90286d21b50744d2ddc803245d497f4ce9255193d'}
MEMO_REF={'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}
V4_REF={'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}
FAILED_REL='research/reviews/H018_R16_10Mb_extension_root_memo_v2'
FAILED_OUTPUT="data/H018_R16_20_30Mb_extension_memo_v2"
FAILED_EVIDENCE={'driver_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/root_extension_driver.py', 'sha256': '13a9c22b6f801fcb7e2d6550bc860c45d4a4ee531304f38d9802502979d91ee3'}, 'source_ref': {'path': 'src/arggnn/prepare_h018_r16_10mb_extension_v1.py', 'sha256': '4c97c6297b12c8e3bed5a3bb44bcdd912f6ee4bd00d8b88fb962c99f11330f49'}, 'admitted_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/admitted.json', 'sha256': 'f3921b29436818c1e3141eea30d9c797aa79fc78a255d7cc75ec3e50ea27e2d3'}, 'admission_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/admission.json', 'sha256': '2891ebc829d8482de07c1a2a79a922026b8fb2ba15a57f15386365463a9cb8a3'}, 'launch_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/launch.json', 'sha256': 'db89f35457fb9a7d2e4857f05a4927c21ea9d579ae81b0e56887625d51669d15'}, 'actual_wait_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/actual_wait.json', 'sha256': '722cec476267cedda0e9ccbfc31ea9e1d1629fa19948a0825d68f0a0023aec10'}, 'log_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/worker_aux/launch.log', 'sha256': 'be295b54ca31d08fd23f19bdeb7f69a5049ac417f83c154da37ed4c18ab8f357'}, 'storage_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/storage.json', 'sha256': '047e1d8e5d17950a2579e5e3bb9b41aece3f2e0296231ee74e2ff2d807d6fbe3'}, 'registration_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/registration.json', 'sha256': '731f296491bfe736373981a2ae84369d63f6001a8c7593f9b082650e5653e5c0'}, 'original_unused_revocation_ref': {'path': 'research/reviews/H018_R16_10Mb_extension_root_memo_v2/old_never_launched_revocation_root.json', 'sha256': '60177d1ecddbb9255f9312b1d1c40fd2eda87cb89f51c9151baa97272e84aea4'}}
CORRECTION_REF={'path': 'research/reviews/H018_10Mb_cohort_order_correction_source_v1/source_handoff.json', 'sha256': '120066d9f4dbc5280e535731c801b72f0943cfc93d0ef34b270467149afe238f'}
CAP=200000000
LOG_CAP=16777216
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
def require(ok,message):
    if not ok:raise ValueError(message)
def local(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(p!=ROOT and ROOT in p.parents and p.resolve()==p and ".." not in p.parts,"Project canonical path")
    return p
def ref(path):
    p=local(path);return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(ref(record["path"])==record,"Exact unchanged metadata")
    raw=local(record["path"]).read_bytes();require(len(raw)<15000000,"Bounded metadata")
    return json.loads(raw)
def save(name,value):
    p=HERE/name;raw=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Compact root receipt")
    with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(p)
def load(path,name):
    s=importlib.util.spec_from_file_location(name,local(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def guard(*,memo_consumer=False):
    for r in [POLICY,GUARD,REPAIR]:require(ref(r["path"])==r,"Frozen resource component")
    m=load(GUARD["path"],"H018_extension_current_guard")
    if memo_consumer:
        for r in (MEMO_REF,V4_REF):require(ref(r["path"])==r,"Frozen consumer memo and original v4 source")
        memo=load(MEMO_REF["path"],"H018_extension_transaction_memo_v2")
        m._memo_handle=memo.install(m.budget.v4,expected_source_path=ROOT/V4_REF["path"],expected_source_sha256=V4_REF["sha256"])
    m.configure_v2(REPAIR,policy_ref=POLICY);return m

def failed_predecessor_v3(certificate_ref):
    """Root-only future barrier; both prior allocations retain their full charge."""
    certificate=read(certificate_ref)
    require(certificate.get("schema")=="H018_R16_failed_memo_v2_root_replacement_certificate_v1"
        and certificate.get("root_authorizer")=="root_agent" and certificate.get("active") is True
        and certificate.get("failed_evidence")==FAILED_EVIDENCE
        and certificate.get("corrected_worker_ref")==WORKER_REF
        and certificate.get("correction_source_handoff_ref")==CORRECTION_REF
        and certificate.get("successor_output_directory")==OUTPUT
        and certificate.get("successor_job_id")=="H018_R16_20_30Mb_extension_root_cohort_order_v3"
        and certificate.get("condition_id")=="H018_R16_20_30Mb_extension_v1"
        and certificate.get("complete_owned_pid_and_group_roster") is True
        and certificate.get("root_targeted_pid_and_group_absence_checked") is True
        and certificate.get("all_prior_launch_resume_write_capabilities_revoked") is True
        and certificate.get("failed_memo_v2_capabilities_revoked") is True
        and certificate.get("original_unused_capabilities_revoked") is True
        and certificate.get("failed_memo_v2_reservation_remains_fully_charged") is True
        and certificate.get("original_unused_reservation_remains_fully_charged") is True
        and certificate.get("all_prior_artifacts_preserved") is True
        and certificate.get("credit_or_release_applied") is False
        and certificate.get("deletion_applied") is False
        and certificate.get("same_scientific_condition") is True,
        "Root complete-roster/revocation certificate required; both prior allocations stay fully charged")
    recorded=datetime.fromisoformat(certificate["recorded_utc"].replace("Z","+00:00"))
    require(recorded.tzinfo is not None and recorded<=datetime.now(timezone.utc),"Dated root replacement certificate")
    for r in (FAILED_EVIDENCE["driver_ref"],OLD_WORKER_REF,FAILED_EVIDENCE["log_ref"],CORRECTION_REF):
        require(ref(r["path"])==r,"Preserved exact failed-attempt source/log and correction evidence")
    failed=read(FAILED_EVIDENCE["admitted_ref"]);ad=read(FAILED_EVIDENCE["admission_ref"])
    launched=read(FAILED_EVIDENCE["launch_ref"]);wait=read(FAILED_EVIDENCE["actual_wait_ref"])
    require(failed["driver_ref"]==FAILED_EVIDENCE["driver_ref"]
        and failed["admission_ref"]==FAILED_EVIDENCE["admission_ref"]
        and ad["source_ref"]==OLD_WORKER_REF and ad["output_directory"]==FAILED_OUTPUT
        and ad["storage_ref"]==FAILED_EVIDENCE["storage_ref"]
        and ad["registration_ref"]==FAILED_EVIDENCE["registration_ref"]
        and ad["operational_predecessor_revocation_ref"]==FAILED_EVIDENCE["original_unused_revocation_ref"]
        and ad["memo_helper_ref"]==MEMO_REF and ad["condition_id"]=="H018_R16_20_30Mb_extension_v1"
        and ad["output_cap_bytes"]==CAP and ad["outer_process_group_timeout_seconds"]==2400,
        "Exact launched failed memo_v2 admission/storage/registration lineage required")
    require(launched["admission_ref"]==FAILED_EVIDENCE["admission_ref"]
        and launched["worker_pid"]==launched["worker_process_group"]==2563001
        and launched["supervisor_pid"]==2562979 and launched["group_deadline_seconds"]==2400
        and launched["memo_helper_ref"]==MEMO_REF
        and launched["command"]==[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/FAILED_EVIDENCE["driver_ref"]["path"]),"--worker"]
        and wait["actual_direct_worker_wait"] is True
        and wait["worker_pid"]==wait["process_group"]==launched["worker_pid"]
        and wait["returncode"]==1 and wait["owned_worker_group_ESRCH"] is True
        and wait["completed_ref"] is None and wait["log_ref"]==FAILED_EVIDENCE["log_ref"]
        and wait["operational_failure"] is None and wait["post_wait_error"] is None
        and wait["cleanup_errors"]==[] and wait["memo_helper_ref"]==MEMO_REF,
        "Genuine failed memo_v2 direct wait-1/log required; historical drain is not fresh absence")
    unused=read(FAILED_EVIDENCE["original_unused_revocation_ref"])
    require(unused["schema"]=="H018_R16_extension_never_launched_revocation_v1"
        and unused["active"] is True and unused["root_authorizer"]=="root_agent"
        and unused["old_driver_ref"]==OLD_DRIVER_REF
        and unused["old_launch_json_absent"] is True and unused["old_output_directory_absent"] is True
        and unused["all_old_launch_resume_write_capabilities_revoked"] is True
        and unused["old_reservation_remains_fully_charged"] is True
        and unused["credit_or_release_applied"] is False,
        "Original unused allocation remains distinct, revoked and fully charged")
    original=read(unused["old_admitted_ref"]);original_ad=read(unused["old_admission_ref"])
    require(unused["old_admitted_ref"]["path"]==OLD_REL+"/admitted.json"
        and unused["old_admission_ref"]["path"]==OLD_REL+"/admission.json"
        and original["driver_ref"]==OLD_DRIVER_REF and original["admission_ref"]==unused["old_admission_ref"]
        and original_ad["source_ref"]==OLD_WORKER_REF and original_ad["output_directory"]==OLD_OUTPUT
        and original_ad["storage_ref"]==unused["old_storage_ref"],
        "Original unused admission/storage lineage must remain preserved")
    require(not os.path.lexists(ROOT/OLD_REL/"launch.json") and not os.path.lexists(ROOT/OLD_OUTPUT)
        and not os.path.lexists(ROOT/FAILED_OUTPUT/"completed.json"),"Prior unused scope and failed completion stay absent")
    pids=certificate["owned_pids"];groups=certificate["owned_process_groups"]
    require(type(pids) is list and type(groups) is list
        and all(type(x) is int and x>1 for x in pids+groups)
        and len(pids)==len(set(pids)) and len(groups)==len(set(groups))
        and {2562979,2563001}.issubset(pids) and 2563001 in groups,
        "Root must supply the complete owned roster including the known supervisor, worker and worker group")
    for pid in pids:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise ValueError("Prior owned PID is still present")
    for pgid in groups:
        try:os.killpg(pgid,0)
        except ProcessLookupError:continue
        raise ValueError("Prior owned process group is still present")
    return certificate

def validate_operational_v2(ad):
    failed_predecessor_v3(ad["operational_predecessor_revocation_ref"])
    require(ref(WORKER)==WORKER_REF and ad["source_ref"]==WORKER_REF
        and ad["memo_helper_ref"]==MEMO_REF and ad["original_v4_ref"]==V4_REF
        and ad["output_directory"]==OUTPUT and ad["condition_id"]=="H018_R16_20_30Mb_extension_v1"
        and ad["output_cap_bytes"]==CAP and ad["outer_process_group_timeout_seconds"]==2400,
        "Operational successor must retain frozen worker, condition,200MB cap and2400s deadline")
    storage=read(ad["storage_ref"])
    require(all(r in storage["reservation"]["consumer_bindings"] for r in
        (MEMO_REF,V4_REF,ad["operational_predecessor_revocation_ref"],ref(SOURCE))),
        "Actual memo/v4/certificate/operational driver must be direct reserved source bindings")

def admit(certificate_ref):
    require(not (HERE/"admitted.json").exists(),"Fresh root admission")
    failed_predecessor_v3(certificate_ref)
    spec=read(ref(PLAN+"/admission.draft.json"))
    require(spec["source_ref"]==OLD_WORKER_REF and ref(WORKER)==WORKER_REF,"Original draft and exact corrected frozen acquisition worker required")
    spec.update(output_directory=OUTPUT,job_id="H018_R16_20_30Mb_extension_root_cohort_order_v3",source_ref=WORKER_REF)
    require(not local(spec["output_directory"]).exists(),"Fresh operational extension scope")
    reg=read(ref(PLAN+"/registration.draft.json"))
    reg.update(experiment_decisions_ref=ref(HERE/"experiment_decisions_snapshot.json"),registered_at_utc=datetime.now(timezone.utc).isoformat(),registered_before_execution=True,implementation_correction_source_handoff_ref=CORRECTION_REF,corrected_worker_ref=WORKER_REF)
    registration_ref=save("registration.json",reg)
    manifest=read(ref(PLAN+"/consumer_manifest.draft.json"));manifest.update(source_ref=WORKER_REF,implementation_correction_source_handoff_ref=CORRECTION_REF,registration_ref=registration_ref,status="root_registered_frozen_extension",memo_helper_ref=MEMO_REF,original_v4_ref=V4_REF,operational_predecessor_revocation_ref=certificate_ref,operational_driver_ref=ref(SOURCE))
    manifest_ref=save("consumer_manifest.json",manifest)
    m=guard();view=m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    consumers=[spec["source_ref"],spec["config_ref"],spec["runtime_ref"],manifest_ref,registration_ref,ref(SOURCE),ref(WAIT_HELPER),POLICY,GUARD,REPAIR,MEMO_REF,V4_REF,certificate_ref,*view["bindings"].values()]
    consumers=list({r["path"]:r for r in consumers}.values())
    require(len(consumers)<=64,"Unchanged64-source consumer binding limit")
    expiry=(datetime.now(timezone.utc)+timedelta(hours=4)).isoformat()
    request=dict(key="H018/real_20_30Mb_extension_root_cohort_order_v3",output=str(local(spec["output_directory"])),auxiliary_scopes=[str(HERE/"worker_aux")],reserved_bytes=CAP+LOG_CAP+65536,consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=view["bindings"]);authority.update(active=True,expires_at_utc=expiry)
    receipt=m.apply_reserve_v1(save("reserve_authority.json",authority),expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save("reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,job_id=spec["job_id"],output_directory=spec["output_directory"],reserved_new_bytes=CAP+LOG_CAP+65536,reservation=receipt["value"],expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],completed_overlay_policy_ref=POLICY,budget_components=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD))
    spec.update(execution_authorized=True,expires_at_utc=expiry,registration_ref=registration_ref,consumer_manifest_ref=manifest_ref,storage_ref=save("storage.json",storage),prospective_condition_registered=True,outer_process_group_timeout_enforced=True,memo_helper_ref=MEMO_REF,original_v4_ref=V4_REF,operational_predecessor_revocation_ref=certificate_ref,operational_successor_same_scientific_condition=True)
    save("admitted.json",dict(admission_ref=save("admission.json",spec),driver_ref=ref(SOURCE)))
    print(json.dumps(dict(admitted=True,reserved_bytes=storage["reserved_new_bytes"])),flush=True)
def worker():
    admitted=read(ref(HERE/"admitted.json"));require(admitted["driver_ref"]==ref(SOURCE),"Frozen root worker adapter")
    ad=read(admitted["admission_ref"]);validate_operational_v2(ad);storage=read(ad["storage_ref"]);m=guard(memo_consumer=True)
    m.budget_check_v9(storage,first=True,boundary=True)
    w=load(WORKER,"H018_extension_scientific_worker")
    result=w.execute_v1(admitted["admission_ref"],guard=lambda planned:m.budget_check_v9(storage,boundary=True,required_write_bytes=planned))
    print(json.dumps(dict(completed_ref=result,memo_helper_ref=MEMO_REF,memo_operation_counts=m._memo_handle.last_operation_counts())),flush=True)
def launch():
    admitted=read(ref(HERE/"admitted.json"));require(admitted["driver_ref"]==ref(SOURCE),"Frozen root driver")
    require(ref(WAIT_HELPER)["sha256"]==WAIT_SHA,"Owned-group wait helper")
    helper=load(WAIT_HELPER,"H018_owned_group_wait")
    ad=read(admitted["admission_ref"]);validate_operational_v2(ad);storage=read(ad["storage_ref"]);m=guard(memo_consumer=True)
    auxiliary=HERE/"worker_aux";require(not auxiliary.exists(),"Fresh retained log scope")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+8192);auxiliary.mkdir()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(ROOT/"src"),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(POLICY))
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/SOURCE),"--worker"]
    started=time.monotonic();captured=bytearray();received=0;failure=None;proc=None;selector=None
    try:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        save("launch.json",dict(worker_pid=proc.pid,worker_process_group=proc.pid,supervisor_pid=os.getpid(),command=cmd,admission_ref=admitted["admission_ref"],actual_wait_pending=True,group_deadline_seconds=2400,memo_helper_ref=MEMO_REF,memo_installation=m._memo_handle.receipt))
        print(json.dumps(dict(launched=True,pid=proc.pid,group_deadline_seconds=2400)),flush=True)
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        while selector.get_map() or proc.poll() is None:
            if time.monotonic()-started>2400:raise TimeoutError("Root 2400-second whole-process-group deadline")
            for key,_ in selector.select(timeout=1):
                block=os.read(key.fileobj.fileno(),65536)
                if not block:selector.unregister(key.fileobj);continue
                received+=len(block);captured.extend(block[:max(0,LOG_CAP-len(captured))])
                require(received<=LOG_CAP,"Bounded retained stdout")
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc)[:4096])
        if isinstance(exc,TimeoutError) and proc is not None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    finally:
        if proc is not None:rc,drained,errors=helper.wait_owned_worker_v1(proc,failure is not None)
        else:rc=None;drained=False;errors=[]
        if selector is not None:selector.close()
        if proc is not None:proc.stdout.close()
    require(proc is not None,"No actual worker created")
    log_ref=None;post_error=None
    if drained:
        try:
            m.budget_check_v9(storage,boundary=True,required_write_bytes=len(captured)+8192)
            with (auxiliary/"launch.log").open("xb") as f:f.write(captured);f.flush();os.fsync(f.fileno())
            m.budget_check_v9(storage,boundary=True);log_ref=ref(auxiliary/"launch.log")
        except BaseException as exc:post_error=dict(type=type(exc).__name__,message=str(exc)[:4096])
    completed=local(ad["output_directory"])/"completed.json"
    wait=dict(actual_direct_worker_wait=True,worker_pid=proc.pid,process_group=proc.pid,returncode=rc,owned_worker_group_ESRCH=drained,operational_failure=failure,post_wait_error=post_error,cleanup_errors=errors,completed_ref=ref(completed) if drained and completed.exists() else None,log_ref=log_ref,received_stdout_bytes=received,retained_stdout_bytes=len(captured),elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat(),memo_helper_ref=MEMO_REF,memo_operation_counts=m._memo_handle.last_operation_counts())
    print(json.dumps(dict(actual_wait=wait)),flush=True);save("actual_wait.json",wait)
    return 0 if rc==0 and drained and failure is None and post_error is None and not errors and wait["completed_ref"] else 1
def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument("--admit",action="store_true");g.add_argument("--launch",action="store_true");g.add_argument("--worker",action="store_true");p.add_argument("--failed-predecessor-ref");p.add_argument("--failed-predecessor-sha256");a=p.parse_args()
    if a.admit:
        require(a.failed_predecessor_ref and a.failed_predecessor_sha256,"Explicit root failed-predecessor replacement certificate path/SHA required")
        admit(dict(path=a.failed_predecessor_ref,sha256=a.failed_predecessor_sha256));return 0
    if a.launch:return launch()
    if a.worker:worker();return 0
    print(json.dumps(dict(source_only=True,GPU_used=False,condition="H018_R16_20_30Mb_extension_v1",output_cap_bytes=CAP,outer_deadline_seconds=2400)));return 0
if __name__=="__main__":raise SystemExit(main())

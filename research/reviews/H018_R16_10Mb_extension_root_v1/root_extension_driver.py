"""Root-only registered 10Mb real-observation extraction with a 2400s group deadline."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,selectors,signal,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/H018_R16_10Mb_extension_root_v1"
HERE=ROOT/REL
SOURCE=REL+"/root_extension_driver.py"
PLAN="research/reviews/H018_R16_10Mb_extension_source_v1"
WORKER="src/arggnn/prepare_h018_r16_10mb_extension_v1.py"
POLICY={"path":"research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json","sha256":"a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85"}
GUARD={"path":"research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py","sha256":"ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef"}
REPAIR={"path":"research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json","sha256":"f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"}
WAIT_HELPER="research/reviews/American_C3_observation_supervisor_repair_source_v1/root_observation_driver.py"
WAIT_SHA="8e282edad3253a0e1d32807a94104129a7593fab3010b8a00b9f5526f8a8fe93"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
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
def guard():
    for r in [POLICY,GUARD,REPAIR]:require(ref(r["path"])==r,"Frozen resource component")
    m=load(GUARD["path"],"H018_extension_current_guard");m.configure_v2(REPAIR,policy_ref=POLICY);return m
def admit():
    require(not (HERE/"admitted.json").exists(),"Fresh root admission")
    spec=read(ref(PLAN+"/admission.draft.json"))
    require(not local(spec["output_directory"]).exists(),"Fresh extension scope")
    reg=read(ref(PLAN+"/registration.draft.json"))
    reg.update(experiment_decisions_ref=ref(HERE/"experiment_decisions_snapshot.json"),registered_at_utc=datetime.now(timezone.utc).isoformat(),registered_before_execution=True)
    registration_ref=save("registration.json",reg)
    manifest=read(ref(PLAN+"/consumer_manifest.draft.json"));manifest.update(registration_ref=registration_ref,status="root_registered_frozen_extension")
    manifest_ref=save("consumer_manifest.json",manifest)
    m=guard();view=m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    consumers=[spec["source_ref"],spec["config_ref"],spec["runtime_ref"],manifest_ref,registration_ref,ref(SOURCE),ref(WAIT_HELPER),POLICY,GUARD,REPAIR,*view["bindings"].values()]
    consumers=list({r["path"]:r for r in consumers}.values())
    expiry=(datetime.now(timezone.utc)+timedelta(hours=4)).isoformat()
    request=dict(key="H018/real_20_30Mb_extension_root_v1",output=str(local(spec["output_directory"])),auxiliary_scopes=[str(HERE/"worker_aux")],reserved_bytes=CAP+LOG_CAP+65536,consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=view["bindings"]);authority.update(active=True,expires_at_utc=expiry)
    receipt=m.apply_reserve_v1(save("reserve_authority.json",authority),expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save("reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,job_id=spec["job_id"],output_directory=spec["output_directory"],reserved_new_bytes=CAP+LOG_CAP+65536,reservation=receipt["value"],expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],completed_overlay_policy_ref=POLICY,budget_components=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD))
    spec.update(execution_authorized=True,expires_at_utc=expiry,registration_ref=registration_ref,consumer_manifest_ref=manifest_ref,storage_ref=save("storage.json",storage),prospective_condition_registered=True,outer_process_group_timeout_enforced=True)
    save("admitted.json",dict(admission_ref=save("admission.json",spec),driver_ref=ref(SOURCE)))
    print(json.dumps(dict(admitted=True,reserved_bytes=storage["reserved_new_bytes"])),flush=True)
def worker():
    admitted=read(ref(HERE/"admitted.json"));require(admitted["driver_ref"]==ref(SOURCE),"Frozen root worker adapter")
    ad=read(admitted["admission_ref"]);storage=read(ad["storage_ref"]);m=guard()
    m.budget_check_v9(storage,first=True,boundary=True)
    w=load(WORKER,"H018_extension_scientific_worker")
    result=w.execute_v1(admitted["admission_ref"],guard=lambda planned:m.budget_check_v9(storage,boundary=True,required_write_bytes=planned))
    print(json.dumps(dict(completed_ref=result)),flush=True)
def launch():
    admitted=read(ref(HERE/"admitted.json"));require(admitted["driver_ref"]==ref(SOURCE),"Frozen root driver")
    require(ref(WAIT_HELPER)["sha256"]==WAIT_SHA,"Owned-group wait helper")
    helper=load(WAIT_HELPER,"H018_owned_group_wait")
    ad=read(admitted["admission_ref"]);storage=read(ad["storage_ref"]);m=guard()
    auxiliary=HERE/"worker_aux";require(not auxiliary.exists(),"Fresh retained log scope")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+8192);auxiliary.mkdir()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(ROOT/"src"),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(POLICY))
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/SOURCE),"--worker"]
    started=time.monotonic();captured=bytearray();received=0;failure=None;proc=None;selector=None
    try:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        save("launch.json",dict(worker_pid=proc.pid,worker_process_group=proc.pid,supervisor_pid=os.getpid(),command=cmd,admission_ref=admitted["admission_ref"],actual_wait_pending=True,group_deadline_seconds=2400))
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
    wait=dict(actual_direct_worker_wait=True,worker_pid=proc.pid,process_group=proc.pid,returncode=rc,owned_worker_group_ESRCH=drained,operational_failure=failure,post_wait_error=post_error,cleanup_errors=errors,completed_ref=ref(completed) if drained and completed.exists() else None,log_ref=log_ref,received_stdout_bytes=received,retained_stdout_bytes=len(captured),elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    print(json.dumps(dict(actual_wait=wait)),flush=True);save("actual_wait.json",wait)
    return 0 if rc==0 and drained and failure is None and post_error is None and not errors and wait["completed_ref"] else 1
def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument("--admit",action="store_true");g.add_argument("--launch",action="store_true");g.add_argument("--worker",action="store_true");a=p.parse_args()
    if a.admit:admit();return 0
    if a.launch:return launch()
    if a.worker:worker();return 0
    print(json.dumps(dict(source_only=True,GPU_used=False,condition="H018_R16_20_30Mb_extension_v1",output_cap_bytes=CAP,outer_deadline_seconds=2400)));return 0
if __name__=="__main__":raise SystemExit(main())

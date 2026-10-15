"""Specific root CPU admission/supervisor for unchanged complete C3 TRAIN paths."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,json,os,selectors,signal,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
from research.reviews.American_C3_native_paths_root_v2 import native_paths_contract_v2 as c
HERE=ROOT/c.REL
PACKET=HERE/"admitted_native_paths_train000_full30mb_v2"
SOURCE=c.DRIVER
PYTHON=ROOT/".runtime/env/bin/python"
MODULE="arggnn.american_c3_native_paths_consumer_v2"

def preflight(close_receipt_ref):
    cfg,bound=c.configuration_v1(active=True);counts=c.validate_counts_v1()
    for record in [*c.NATIVE_CLOSURE,c.MEMO_REF,c.V4_REF,c.GUARD_REF,c.POLICY_REF,c.REPAIR_REF,c.RUNTIME_REF]:
        c.require(c.ref(record["path"])==record,"Exact original source/helper/runtime/authority required")
    runtime=c.read(c.RUNTIME_REF)
    c.require(runtime["prefix"]==str(ROOT/".runtime/env") and runtime["CUDA_VISIBLE_DEVICES"]=="",
              "Exact qualified shared CPU runtime")
    c.validate_close_receipt_v1(close_receipt_ref)
    return cfg,bound

def admit(close_receipt_ref):
    c.require(Path.cwd().resolve()==ROOT,"Repository execution only")
    cfg,bound=preflight(close_receipt_ref)
    c.require(not PACKET.exists() and not (ROOT/c.OUTPUT).exists(),"Fresh exclusive packet/output; no replay")
    PACKET.mkdir()
    cfg_ref=c.save(PACKET,"config.json",cfg)
    m=c.original_guard_v1()  # Root mutations remain unmemoized.
    with m.budget.v4.mutex(ROOT):state=m._state_locked(c.CORE,c.BASE,fresh=True)
    view,bindings=state[2],state[3]
    extra=[c.ref(p) for p in (
        "research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py",
        "research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py",
        "research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py",
        "src/arggnn/project_budget_v9_consumer.py")]
    consumers=[*c.closure_v1(),cfg_ref,c.RUNTIME_REF,c.COUNTS_REF,c.EXACT_COUNT_REF,c.POLICY_REF,c.REPAIR_REF,
        close_receipt_ref,*extra,*bindings.values()]
    consumers=list({(r["path"],r["sha256"]):r for r in consumers}.values())
    c.require(len(consumers)<=64,"Existing64 direct source/JSON binding limit")
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=3)).isoformat()
    job="American-C3-native-paths-train000-full30mb-92001-v2"
    request=dict(key="H041/"+job,output=str(ROOT/c.OUTPUT),
        auxiliary_scopes=[str(PACKET/"worker_aux")],reserved_bytes=c.CAP,consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry)
    authority_ref=c.save(PACKET,"reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    c.save(PACKET,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=job,output_directory=c.OUTPUT,reserved_new_bytes=c.CAP,
        reservation=receipt["value"],expires_at_utc=expiry,completed_overlay_policy_ref=c.POLICY_REF,
        initial_ledger_pins=receipt["initial_ledger_pins"],
        budget_components=dict(implementation=c.ref(m.budget.SOURCE_REL),
            launcher=c.ref(m.budget.LAUNCHER_REL),consumer_adapter=c.GUARD_REF))
    spec=dict(root_authorizer="root_agent",hypothesis_id="H041",execution_authorized=True,GPU_authorized=False,
        parent_id="american_c3_train_000",split="train",job_id=job,source_ref=c.PATHS_REF,
        source_closure_refs=c.NATIVE_CLOSURE,integration_source_closure_refs=c.closure_v1(),
        consumer_wrapper_ref=c.ref(c.WRAPPER),driver_ref=c.ref(SOURCE),config_ref=cfg_ref,
        runtime_ref=c.RUNTIME_REF,count_ref=c.COUNTS_REF,exact_native_path_count_ref=c.EXACT_COUNT_REF,
        storage_adapter_ref=c.GUARD_REF,
        marker_repair_authority_ref=c.REPAIR_REF,completed_overlay_policy_ref=c.POLICY_REF,
        completed_inference_close_receipt_ref=close_receipt_ref,storage=storage,output_directory=c.OUTPUT,
        operational_deadline_seconds=c.DEADLINE_SECONDS,max_owned_group_RSS_bytes=c.RSS_CAP,
        created_at_utc=now.isoformat(),expires_at_utc=expiry)
    admission_ref=c.save(PACKET,"admission.json",spec,m,storage)
    result=dict(admission_ref=admission_ref,driver_ref=c.ref(SOURCE),consumer_wrapper_ref=c.ref(c.WRAPPER),
        policy_ref=c.POLICY_REF,reserved_bytes=c.CAP,unchanged_native_storage_bound_bytes=bound,
        bounded_log_bytes=c.LOG_CAP,metadata_allowance_bytes=c.METADATA_ALLOWANCE,execution_started=False)
    c.save(PACKET,"admitted.json",result,m,storage)
    print(json.dumps(result),flush=True)

def owned_group_RSS_v1(root_pid,known):
    """Only owned PID/task/child records; no global process or filesystem scan."""
    pending=list(known);visited=set();rss=0;page=os.sysconf("SC_PAGE_SIZE")
    while pending:
        pid=pending.pop()
        if pid in visited:continue
        visited.add(pid)
        try:
            parts=Path("/proc/%d/stat"%pid).read_text().rsplit(")",1)[1].split()
        except FileNotFoundError:continue
        c.require(int(parts[2])==root_pid and int(parts[3])==root_pid,
                  "Known owned PID escaped its new worker process group/session")
        start=int(parts[19])
        c.require(pid not in known or known[pid] in (None,start),"Owned PID identity changed")
        known[pid]=start;rss+=int(parts[21])*page
        try:
            tasks=list(Path("/proc/%d/task"%pid).iterdir())
        except FileNotFoundError:continue
        for task in tasks:
            try:children=(task/"children").read_text().split()
            except FileNotFoundError:continue
            for child in children:
                child=int(child)
                if child not in known:known[child]=None
                pending.append(child)
    return rss

def launch():
    admitted=c.read(c.ref(PACKET/"admitted.json"));spec=c.read(admitted["admission_ref"])
    cfg,bound=preflight(spec["completed_inference_close_receipt_ref"])
    c.require(admitted["driver_ref"]==spec["driver_ref"]==c.ref(SOURCE)
        and spec["consumer_wrapper_ref"]==c.ref(c.WRAPPER)
        and spec["exact_native_path_count_ref"]==c.EXACT_COUNT_REF
        and spec["integration_source_closure_refs"]==c.closure_v1()
        and c.read(spec["config_ref"])==cfg and spec["source_closure_refs"]==c.NATIVE_CLOSURE,
        "Actual frozen CPU driver/wrapper/native source/config closure")
    storage=spec["storage"];reserve=c.read(c.ref(PACKET/"reserve_receipt.json"))
    c.require(storage["initial_ledger_pins"]==reserve["initial_ledger_pins"]
        and storage["reservation"]==reserve["value"],"Preserve original immutable reservation prefix; no rebasing")
    c.require(spec["operational_deadline_seconds"]==7200 and spec["max_owned_group_RSS_bytes"]==512000000000,
              "Exact real CPU operational deadline/RSS caps")
    m=c.original_guard_v1()
    m.validate_storage_v9(spec,cfg,storage,source_ref=c.PATHS_REF,config_ref=spec["config_ref"],runtime_ref=c.RUNTIME_REF)
    aux=PACKET/"worker_aux"
    c.require(not aux.exists() and not (ROOT/c.OUTPUT).exists()
        and storage["reservation"]["auxiliary_scopes"]==[str(aux)],"Fresh exact owned product/auxiliary scopes")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=c.LOG_CAP+c.METADATA_ALLOWANCE)
    aux.mkdir();work=aux/"work";(work/"tmp").mkdir(parents=True);(work/"cache").mkdir()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="",PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]),
        PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",TMPDIR=str(work/"tmp"),TMP=str(work/"tmp"),TEMP=str(work/"tmp"),
        XDG_CACHE_HOME=str(work/"cache"),NUMBA_CACHE_DIR=str(work/"cache"),
        TORCH_HOME=str(work/"cache"),TORCH_EXTENSIONS_DIR=str(work/"cache"),
        ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(c.POLICY_REF,separators=(",",":")))
    command=[str(PYTHON),"-B","-m",MODULE,"--execute","--admission",admitted["admission_ref"]["path"],
             "--expected-admission-sha256",admitted["admission_ref"]["sha256"]]
    started=time.monotonic();captured=bytearray();received=0;failure=None;proc=None;selector=None
    supervisor_errors=[];post_wait_errors=[];peak_RSS=0;known={};last_RSS=started-5;terminate_at=None;roster_verified=True
    try:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        known[proc.pid]=None
        c.save(PACKET,"launch.json",dict(pid=proc.pid,worker_pid=proc.pid,process_group=proc.pid,
            supervisor_pid=os.getpid(),stage="native_paths_full30mb",parent_id="american_c3_train_000",
            admission_ref=admitted["admission_ref"],driver_ref=c.ref(SOURCE),consumer_wrapper_ref=c.ref(c.WRAPPER),
            command=command,policy_ref=c.POLICY_REF,actual_wait_pending=True,
            stdout_policy="bounded_RAM_until_direct_wait_and_owned_group_ESRCH"))
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        while selector.get_map() or proc.poll() is None:
            now=time.monotonic()
            if failure is None:
                if now-started>=c.DEADLINE_SECONDS:failure="Actual7200second CPU worker deadline exceeded"
                elif datetime.now(timezone.utc)>=datetime.fromisoformat(spec["expires_at_utc"]):failure="Root admission expired"
                if now-last_RSS>=5:
                    try:peak_RSS=max(peak_RSS,owned_group_RSS_v1(proc.pid,known))
                    except BaseException:
                        roster_verified=False;raise
                    last_RSS=now
                    if peak_RSS>c.RSS_CAP:failure="Owned process-group512GB RSS cap exceeded"
                if failure is not None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    terminate_at=now
            if terminate_at is not None and proc.poll() is None and now-terminate_at>30:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            events=selector.select(timeout=1)
            for key,_ in events:
                block=os.read(key.fileobj.fileno(),65536)
                if not block:selector.unregister(key.fileobj);continue
                received+=len(block);remaining=max(0,c.LOG_CAP-len(captured));captured.extend(block[:remaining])
                if failure is None and len(block)>remaining:
                    failure="Bounded stdout RAM cap exceeded";terminate_at=time.monotonic()
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            if not events and proc.poll() is not None:break
    except BaseException as error:
        failure=failure or "Supervisor exception: "+type(error).__name__
        supervisor_errors.append(type(error).__name__+": "+str(error)[:2048])
    finally:
        if proc is not None:rc,group_absent,cleanup_errors=c.previous.wait_owned_worker_v1(proc,failure is not None)
        else:rc=None;group_absent=False;cleanup_errors=[]
        if selector is not None:
            try:selector.close()
            except BaseException as error:post_wait_errors.append(str(error)[:1024])
        if proc is not None and proc.stdout is not None:
            try:proc.stdout.close()
            except BaseException as error:post_wait_errors.append(str(error)[:1024])
    if proc is None:
        c.save(PACKET,"launch_failure.json",dict(actual_direct_worker_wait=False,supervisor_errors=supervisor_errors))
        raise RuntimeError("No worker was created; no actual wait claimed")
    log_ref=completed_ref=None
    if not group_absent or not roster_verified:failure=failure or "Owned group/roster not proven terminal; no scope/log write"
    if group_absent and roster_verified:
        try:
            m.budget_check_v9(storage,boundary=True,required_write_bytes=len(captured)+8192)
            with (aux/"launch.log").open("xb") as log:log.write(captured);log.flush();os.fsync(log.fileno())
            log_ref=c.ref(aux/"launch.log")
            completed=ROOT/c.OUTPUT/"completed.json"
            if completed.exists():
                completed_ref=c.ref(completed);value=c.read(completed_ref)
                c.require(value["status"]=="completed_native_C3_path_preparation"
                    and value["parent_id"]=="american_c3_train_000" and value["observation_ref"]==c.OBSERVATION_REF
                    and value["query_ancestry_truth_opened"] is False
                    and value["author_Table1_exact_reproduction"] is False,"Actual complete unchanged native corpus metadata")
                c.require([value[k] for k in ("unique_paths","path_tokens","path_occurrences","vocabulary_nodes")]
                    ==[cfg[k] for k in ("max_unique_paths","max_path_tokens","max_path_occurrences","max_vocabulary_nodes")]
                    and value["native_split_seed"]==42 and value["native_train_fraction"]==.9
                    and value["query_focal_count"]==10000
                    and value["contrastive_labels_reference_only"] is True
                    and value["query_embedding_view_prespecified_focals_only"] is True,
                    "Exact counted full corpus and original seed42 split/reference-only/focal views required")
        except BaseException as error:
            failure=failure or "Terminal guarded log/completion audit failed"
            post_wait_errors.append(type(error).__name__+": "+str(error)[:2048])
    if rc==0 and completed_ref is None:failure=failure or "Exit0 without native completion"
    if cleanup_errors:failure=failure or "Owned group wait/cleanup needs root follow-up"
    wait=dict(stage="native_paths_full30mb",parent_id="american_c3_train_000",pid=proc.pid,worker_pid=proc.pid,
        supervisor_pid=os.getpid(),process_group=proc.pid,returncode=rc,actual_direct_worker_wait=True,
        owned_worker_group_ESRCH=group_absent,owned_RSS_roster_verified=roster_verified,
        operational_failure=failure,supervisor_errors=supervisor_errors,post_wait_errors=post_wait_errors,
        cleanup_errors=cleanup_errors,admission_ref=admitted["admission_ref"],driver_ref=c.ref(SOURCE),
        policy_ref=c.POLICY_REF,completed_ref=completed_ref,log_ref=log_ref,stdout_bytes_received=received,
        retained_log_bytes=len(captured) if log_ref else 0,captured_log_bytes=len(captured),
        captured_log_sha256=hashlib.sha256(captured).hexdigest(),elapsed_seconds=time.monotonic()-started,
        actual_deadline_seconds=c.DEADLINE_SECONDS,peak_owned_group_RSS_bytes=peak_RSS,
        owned_PID_roster_observed=sorted(known),unmodified_native_scientific_source_ref=c.PATHS_REF,
        no_scientific_subsampling_or_cap_fallback=True,finished_utc=datetime.now(timezone.utc).isoformat())
    # Compact root packet evidence is outside mutable product/auxiliary scopes.
    print(json.dumps(dict(actual_worker_wait_evidence=wait),sort_keys=True),flush=True)
    try:c.save(PACKET,"actual_wait.json",wait)
    except BaseException as error:
        c.save(PACKET,"actual_wait_fallback.json",dict(wait=wait,evidence_write_error=str(error)[:2048]))
    return rc if rc else (1 if failure else 0)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument("--admit",action="store_true");mode.add_argument("--launch",action="store_true")
    parser.add_argument("--completed-close-receipt");parser.add_argument("--expected-close-receipt-sha256")
    args=parser.parse_args(argv)
    if args.admit:
        c.require(args.completed_close_receipt and args.expected_close_receipt_sha256,"Actual completed full30Mb credit receipt required")
        admit(dict(path=args.completed_close_receipt,sha256=args.expected_close_receipt_sha256));return 0
    if args.launch:return launch()
    print(json.dumps(dict(status="source_only_inactive_no_guard_or_process_checks",
        native_complete_corpus=True,CPU_only=True,proposed_cap_bytes=c.CAP,deadline_seconds=c.DEADLINE_SECONDS,
        max_owned_group_RSS_bytes=c.RSS_CAP)));return 0
if __name__=="__main__":raise SystemExit(main())

"""One fixed root-admitted CPU FP32 fixture, launch-order repair v2.
Import is inert and stdlib only. Original v1 sources remain unchanged.
Root explicitly invokes --execute once. No training, GPU lease or data payload.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,importlib.metadata,json,os,selectors,signal,subprocess,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CONTRACT_REF={'path': 'research/reviews/C3_one_new_FP32_CPU_root_source_20261016_v2/contract.source_only.v2.json', 'sha256': 'aaa6006809b7d2a73b95aeb19cf4bc21c3be646b2813682d6dcb75a16f8385b6'}
HERE=ROOT/CONTRACT_REF["path"].rsplit("/",1)[0]
ADMIN=HERE/"actual_admission_v1"
CAP=1048576
META_CAP=131072
THREADS=4
RSS=2*2**30
WALL=300
LOG_CAP=16384
RESULT_CAP=4096
_RAN=False

def need(ok,message):
    if not ok:raise ValueError(message)

def utc():
    return datetime.now(timezone.utc).isoformat()

def canonical(name):
    p=Path(name)
    p=p if p.is_absolute() else ROOT/p
    need(p!=ROOT and ROOT in p.parents and p.resolve()==p,"Canonical contained project pathname")
    return p

def ref(name):
    p=canonical(name)
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record,limit=65536):
    p=canonical(record["path"])
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit and ref(p)==record,"Exact bounded source/metadata")
    return json.loads(p.read_text())

def encoded(value,limit):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(data)<=limit,"Finite predeclared serialized output cap")
    return data

def exclusive(path,data):
    p=canonical(path)
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    return ref(p)

def load(record,label):
    p=canonical(record["path"])
    need(ref(p)==record and p.stat().st_size<=1048576,"Exact authenticated source before import")
    spec=importlib.util.spec_from_file_location(label,p)
    module=importlib.util.module_from_spec(spec)
    sys.modules[label]=module
    exec(compile(p.read_bytes(),str(p),"exec"),module.__dict__)
    return module

def unique(records):
    pins={}
    for row in records:
        need(set(row)=={"path","sha256"} and (row["path"] not in pins or pins[row["path"]]==row["sha256"]),"Unambiguous exact bindings")
        need(Path(row["path"]).suffix in {".py",".json",".sh"} and ref(row["path"])==row,"Actual allowed source binding")
        pins[row["path"]]=row["sha256"]
    result=[dict(path=p,sha256=h) for p,h in sorted(pins.items())]
    need(len(result)<=64,"Original at-most64 direct source bindings")
    return result

def contract():
    need(Path.cwd().resolve()==ROOT and Path(__file__).resolve()==HERE/"ordinary_root_cpu_v2.py","Exact fixed source and project cwd")
    c=read(CONTRACT_REF)
    need(c["schema"]=="C3_one_new_CPU_fixture_fixed_root_contract_v1" and c["execution_authorized"] is False,"Source contract is not admission")
    for row in c["source_binding_basis_refs"]:need(ref(row["path"])==row,"Every exact transitive science/guard source")
    h=read(c["minimal_handoff_ref"])
    need(h["callable"]["source_ref"]==c["wrapper_ref"] and h["fixture"]["device"]=="cpu" and h["fixture"]["optimizer_steps"]==1,"One already reviewed fixture")
    return c

def runtime():
    executable=str(Path(sys.executable).resolve())
    prefix=str(Path(sys.prefix).resolve())
    need(prefix==str(ROOT/".runtime/env") and executable==str((ROOT/".runtime/env/bin/python").resolve()),"Actual canonical existing project interpreter")
    need(sys.version.split()[0]=="3.11.17","Existing pinned Python version")
    packages={n:importlib.metadata.version(n) for n in ("torch","torch_geometric","numpy")}
    need(packages=={"torch":"2.6.0+cu124","torch_geometric":"2.7.0","numpy":"2.4.6"},"Existing pinned numerical library versions")
    return dict(schema="C3_one_new_CPU_fixture_runtime_v1",executable=executable,prefix=prefix,python=sys.version.split()[0],
                packages=packages,CUDA_VISIBLE_DEVICES="",CPU_threads=THREADS)

def guard(c):
    renewal=load(c["fixed_refs"]["renewal"],"_C3_one_CPU_original_renewal")
    renewal.install_renewal_v1(c["fixed_refs"]["renewal_certificate"])
    m=load(c["fixed_refs"]["guard"],"_C3_one_CPU_original980")
    m.configure_v2(c["fixed_refs"]["repair_authority"],policy_ref=c["fixed_refs"]["policy"])
    return m

def bounds(storage,m):
    need(storage["reserved_new_bytes"]==CAP and storage["accounting_mode"]=="prospective_980gb_v1","Exact1MiB original980 allocation")
    paths=[canonical(storage["output_directory"]),*(canonical(p) for p in storage["reservation"]["auxiliary_scopes"])]
    measured=m.budget.v4.measure_scopes(ROOT,paths)
    need(measured["conservative_scope_bytes"]<=CAP,"Complete owned execution scope exceeds1MiB")
    return measured

def boundary(storage,m,growth=0):
    return m.budget_check_v9(storage,boundary=True,required_write_bytes=growth)

def child_env(aux):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",
        TMPDIR=str(aux/"tmp"),TMP=str(aux/"tmp"),TEMP=str(aux/"tmp"),
        XDG_CACHE_HOME=str(aux/"cache"),TORCH_HOME=str(aux/"torch"),
        TORCHINDUCTOR_CACHE_DIR=str(aux/"inductor"),TORCH_EXTENSIONS_DIR=str(aux/"extensions"))
    env.pop("PYTHONPYCACHEPREFIX",None)
    return env

def worker(args):
    global _RAN
    need(not _RAN,"One fresh worker invocation")
    _RAN=True
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Explicit CPU-only worker")
    c=contract()
    ir=dict(path=args.input,sha256=args.expected_sha256)
    spec=read(ir)
    need(spec["schema"]=="C3_one_new_CPU_fixture_actual_input_v1" and spec["root_authorizer"]=="root_agent"
         and spec["execution_authorized"] is True and spec["GPU_authorized"] is False
         and spec["source_ref"]==ref(__file__) and spec["contract_ref"]==CONTRACT_REF,"Actual fixed root authorization")
    need(args.monitor_parent_pid==os.getppid(),"Actual owned outer monitor parent")
    with os.fdopen(args.monitor_read_fd,"rb") as f:ticket=f.read(66)
    need(ticket==(ir["sha256"]+"\n").encode(),"Inherited single-launch root handshake")
    actual=runtime()
    need(read(spec["runtime_ref"])==actual,"Exact canonical worker runtime")
    cfg=read(spec["config_ref"])
    need(cfg==spec["fixed_config"] and cfg["output_cap_bytes"]==CAP and cfg["wrapper_ref"]==c["wrapper_ref"],"Exact fixture/caps/config")
    closure=read(spec["closure_ref"])
    direct=unique([*closure["bound_refs"],spec["closure_ref"]])
    storage=read(spec["storage_ref"])
    need(storage["reservation"]["consumer_bindings"]==direct and storage["job_id"]==spec["job_id"]
         and storage["output_directory"]==c["output_directory"],"Complete actual direct source/storage binding")
    out=canonical(c["output_directory"]);aux=canonical(c["auxiliary_directory"])
    expected_env=child_env(aux)
    for key in ("CUDA_VISIBLE_DEVICES","TMPDIR","TMP","TEMP","XDG_CACHE_HOME","TORCH_HOME",
                "TORCHINDUCTOR_CACHE_DIR","TORCH_EXTENSIONS_DIR","PYTHONDONTWRITEBYTECODE",
                "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        need(os.environ.get(key)==expected_env[key],"Dedicated charged temporary/cache/thread environment")
    need(not any(n in sys.modules for n in ("torch","torch_geometric","numpy")),"Fresh worker before numerical imports")
    m=guard(c);boundary(storage,m,16384)
    base=load(c["monitor_helper_ref"],"_C3_one_CPU_owned_helpers_worker")
    started=spec["root_execute_monotonic"]
    need(type(started) is float and 0<started<=time.monotonic(),"Actual original root monotonic start")
    def resource_check(stage):
        need(time.time()<datetime.fromisoformat(spec["expires_at_utc"]).timestamp()
             and time.monotonic()-started<WALL,"Actual root CPU wall/expiry bound")
        row=base.process_info(os.getpid())
        need(row is not None and row["rss_bytes"]<=RSS,"Actual CPU worker RSS soft bound")
    resource_check("before_fixture_source")
    exclusive(aux/"worker_started.json",encoded(dict(schema="C3_one_CPU_actual_worker_started_v1",
        pid=os.getpid(),parent_pid=os.getppid(),input_ref=ir,started_utc=utc()),2048))
    module=load(c["wrapper_ref"],"_C3_one_new_CPU_qualification_wrapper")
    result=module.run_single_CPU_qualification_v1(resource_check=resource_check)
    resource_check("after_fixture")
    data=encoded(result,RESULT_CAP)
    boundary(storage,m,len(data)+8192)
    result_ref=exclusive(out/"result.json",data)
    bounds(storage,m)
    print(json.dumps(dict(result_ref=result_ref,fixture_passed=result["fixture_result"]["passed"])),flush=True)
    return 0

def execute():
    # One lifetime includes source/runtime qualification and the entire admission.
    started=time.monotonic()
    wall_started=datetime.now(timezone.utc)
    expires=(wall_started+timedelta(seconds=WALL)).isoformat()
    c=contract()
    need(not ADMIN.exists(),"No admission replay or failed-attempt reuse")
    out=canonical(c["output_directory"]);aux=canonical(c["auxiliary_directory"])
    need(not out.exists() and not aux.exists(),"Fresh fixed output and auxiliary scope")
    own_runtime=runtime()
    ADMIN.mkdir()
    metadata_used=0
    def meta(name,value,limit=65536):
        nonlocal metadata_used
        data=encoded(value,limit);need(metadata_used+len(data)+4096<=META_CAP,"Original per-event metadata allowance")
        metadata_used+=len(data)+4096
        return exclusive(ADMIN/name,data)
    runtime_ref=meta("runtime.actual_v1.json",own_runtime)
    cfg=dict(schema="C3_one_new_CPU_fixture_actual_config_v1",wrapper_ref=c["wrapper_ref"],device="cpu",
        CPU_threads=THREADS,max_aggregate_RSS_soft_bytes=RSS,wall_seconds=WALL,output_cap_bytes=CAP,
        result_JSON_cap_bytes=RESULT_CAP,combined_log_retention_bytes=LOG_CAP,
        temporary_and_cache_allowance_bytes=262144,source_ref=ref(__file__),contract_ref=CONTRACT_REF)
    config_ref=meta("config.actual_v1.json",cfg)
    direct=unique([*c["source_binding_basis_refs"],c["minimal_handoff_ref"],CONTRACT_REF,ref(__file__),runtime_ref,config_ref])
    closure_ref=meta("source_closure.actual_v1.json",dict(schema="C3_one_CPU_actual_source_closure_v1",bound_refs=direct))
    direct=unique([*direct,closure_ref])
    m=guard(c);b=c["original_bindings"]
    kw=dict(expected_source_sha256=b["implementation"]["sha256"],expected_launcher_sha256=b["launcher"]["sha256"])
    with m.budget.v4.mutex(ROOT):
        state=m._state_locked(kw["expected_source_sha256"],kw["expected_launcher_sha256"],fresh=True)
        need(state[3]==b,"Actual unchanged original guard binding map")
        pins=m._pins(state[2])
    request=dict(key=c["reservation_key"],output=str(out),auxiliary_scopes=[str(aux)],reserved_bytes=CAP,consumer_bindings=direct)
    auth=m.operation_draft_v1("reserve",request,expected_before=pins,bindings=b)
    auth.update(active=True,expires_at_utc=expires)
    authority_ref=meta("reserve_authority.root_active_v1.json",auth)
    receipt=m.apply_reserve_v1(authority_ref,**kw)
    meta("reserve_receipt.actual_v1.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode="prospective_980gb_v1",job_id=c["job_id"],output_directory=c["output_directory"],
        expires_at_utc=expires,reserved_new_bytes=CAP,reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],completed_overlay_policy_ref=c["fixed_refs"]["policy"],
        budget_components=dict(consumer_adapter=c["fixed_refs"]["guard"],implementation=b["implementation"],launcher=b["launcher"]))
    storage_ref=meta("storage.actual_v1.json",storage)
    spec=dict(schema="C3_one_new_CPU_fixture_actual_input_v1",root_authorizer="root_agent",
        execution_authorized=True,GPU_authorized=False,source_ref=ref(__file__),contract_ref=CONTRACT_REF,
        job_id=c["job_id"],expires_at_utc=expires,root_execute_started_utc=wall_started.isoformat(),
        root_execute_monotonic=started,runtime_ref=runtime_ref,config_ref=config_ref,
        closure_ref=closure_ref,storage_ref=storage_ref,fixed_config=cfg)
    input_ref=meta("input.root_active_v1.json",spec)
    boundary(storage,m,131072)
    out.mkdir();aux.mkdir()
    for name in ("tmp","cache","torch","inductor","extensions"):(aux/name).mkdir()
    env=child_env(aux)
    base=load(c["monitor_helper_ref"],"_C3_one_CPU_owned_helpers_monitor")
    rfd,wfd=os.pipe()
    argv=[own_runtime["executable"],"-B",str(Path(__file__).resolve()),"--worker","--input",input_ref["path"],
          "--expected-sha256",input_ref["sha256"],"--monitor-parent-pid",str(os.getpid()),"--monitor-read-fd",str(rfd)]
    proc=None;selector=None;owned={};groups={};log=bytearray();discarded=0;peak=0;races=0;error=None;rc=None
    try:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(rfd,))
        os.close(rfd);rfd=None
        # The child remains blocked before numerical imports until all launch
        # identities, initial aggregate RSS, durable receipt and selector exist.
        worker_identity=base.process_info(proc.pid)
        root_identity=base.process_info(os.getpid())
        need(worker_identity is not None and root_identity is not None
             and worker_identity["pid"]==proc.pid and root_identity["pid"]==os.getpid()
             and worker_identity["pgid"]==worker_identity["session"]==proc.pid,
             "Exact owned fresh worker/root identities before handshake")
        owned[proc.pid]=worker_identity["start_ticks"]
        groups[worker_identity["pgid"]]={proc.pid:worker_identity["start_ticks"]}
        initial_RSS=worker_identity["rss_bytes"]+root_identity["rss_bytes"]
        peak=max(peak,initial_RSS)
        need(initial_RSS<=RSS,"Initial worker plus root-monitor RSS soft cap")
        # The original pre-Popen boundary reserved this bounded launch receipt.
        # No blocking guard call may occur after release and before monitoring.
        exclusive(aux/"actual_launch.json",encoded(dict(schema="C3_one_CPU_actual_owned_launch_v2",
            worker_pid=proc.pid,worker_identity=worker_identity,root_monitor_identity=root_identity,
            initial_aggregate_RSS_bytes=initial_RSS,RSS_limit_bytes=RSS,
            aggregate_includes_root_monitor=True,handshake_not_yet_released=True,
            original_observer_parent_pid=os.getppid(),input_ref=input_ref,argv=argv,
            root_execute_started_utc=wall_started.isoformat(),
            elapsed_seconds_including_admission=time.monotonic()-started,started_utc=utc()),4096))
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        last_scope=time.monotonic()
        need(last_scope-started<WALL and time.time()<datetime.fromisoformat(expires).timestamp(),
             "Unextended300s lifetime before numerical handshake")
        os.write(wfd,(input_ref["sha256"]+"\n").encode());os.close(wfd);wfd=None
        while proc.poll() is None:
            now=time.monotonic()
            try:
                rss,misses=base.observe(owned,groups);peak=max(peak,rss);races+=misses
                need(rss<=RSS and now-started<WALL and time.time()<datetime.fromisoformat(expires).timestamp(),
                     "Original owned aggregate CPU RSS/wall bound")
                if now-last_scope>=1.:
                    bounds(storage,m)
                    temporary=m.budget.v4.measure_scopes(ROOT,[aux/n for n in ("tmp","cache","torch","inductor","extensions")])
                    need(temporary["conservative_scope_bytes"]<=262144,"Dedicated temporary/cache allowance exceeded")
                    last_scope=now
            except BaseException as exc:
                error=dict(type=type(exc).__name__,message=str(exc)[:2048]);break
            for key,_ in selector.select(.25):
                chunk=os.read(key.fd,65536)
                if chunk:
                    keep=min(len(chunk),LOG_CAP-len(log));log.extend(chunk[:keep]);discarded+=len(chunk)-keep
                else:selector.unregister(key.fileobj)
    except BaseException as exc:
        error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
    finally:
        for fd in (rfd,wfd):
            if fd is not None:os.close(fd)
        if proc is not None:
            if error is not None and proc.poll() is None:proc.terminate()
            try:rc=proc.wait(timeout=40 if error else 5)
            except subprocess.TimeoutExpired:
                base.signal_owned(groups,signal.SIGKILL)
                proc.kill();rc=proc.wait()
            # One precharged genuine minimal terminal survives failure/expiry.
            exclusive(aux/"actual_wait_minimal.json",encoded(dict(schema="C3_one_CPU_actual_direct_wait_minimal_v1",
                actual_direct_wait=True,pid=proc.pid,returncode=rc,worker_pgid=proc.pid,input_ref=input_ref,
                waited_utc=utc()),4096))
    need(proc is not None,"No child spawned, no actual wait claimed")
    live=base.absent_groups(groups)
    if live:
        base.signal_owned(groups,signal.SIGKILL)
        until=time.monotonic()+10
        while live and time.monotonic()<until:time.sleep(.1);live=base.absent_groups(groups)
    if selector is not None:
        for _ in range(16):
            ready=selector.select(0)
            if not ready:break
            for key,_ in ready:
                chunk=os.read(key.fd,65536)
                if chunk:
                    keep=min(len(chunk),LOG_CAP-len(log));log.extend(chunk[:keep]);discarded+=len(chunk)-keep
                else:selector.unregister(key.fileobj)
        selector.close()
    proc.stdout.close()
    need(not live,"Owned group unresolved, preserve actual minimal wait")
    boundary(storage,m,len(log)+32768)
    log_ref=exclusive(aux/"worker_stdout.txt",bytes(log))
    wait_ref=exclusive(aux/"outer_wait.json",encoded(dict(schema="C3_one_CPU_actual_outer_wait_v1",
        returncode=rc,error=error,actual_direct_wait=True,worker_pid=proc.pid,input_ref=input_ref,
        owned_process_identities=owned,owned_groups=sorted(groups),owned_groups_ESRCH=True,
        root_monitor_identity=base.process_info(os.getpid()),original_observer_parent_pid=os.getppid(),
        peak_aggregate_RSS_bytes=peak,RSS_soft_limit_bytes=RSS,RSS_is_polling_soft_bound=True,
        poll_seconds=.25,proc_exit_or_task_read_races=races,stdout_ref=log_ref,
        discarded_stdout_bytes=discarded,elapsed_seconds=time.monotonic()-started),8192))
    need(rc==0 and error is None,"Actual one-shot fixture failed, preserve artifacts and do not retry")
    result_ref=ref(out/"result.json")
    result=read(result_ref,RESULT_CAP)
    need(result["fixture_result"]["passed"] is True and result["fixture_result"]["precision"]=="fp32"
         and result["fixture_result"]["optimizer_steps"]==1,"Genuine one FP32 fixture result")
    boundary(storage,m,16384)
    completed=dict(schema="C3_one_new_FP32_CPU_root_completed_v1",status="completed_single_FP32_CPU_fixture",
        input_ref=input_ref,result_ref=result_ref,actual_direct_wait_ref=ref(aux/"actual_wait_minimal.json"),
        actual_outer_wait_ref=wait_ref,physical_GPU_used=None,actual_fixture_passed=True,
        full_training_fit_or_ancestry_accuracy_claimed=False,root_tool_terminal_consumption_still_required=True)
    completed_ref=exclusive(out/"completed.json",encoded(completed,8192))
    boundary(storage,m)
    print(json.dumps(dict(completed_ref=completed_ref,result_ref=result_ref,actual_outer_wait_ref=wait_ref,
        direct_binding_count=len(direct),reserved_bytes=CAP,ancestry_accuracy_result=False)),flush=True)
    return 0

def main():
    os.chdir(ROOT)
    sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute",action="store_true");modes.add_argument("--worker",action="store_true")
    parser.add_argument("--input");parser.add_argument("--expected-sha256")
    parser.add_argument("--monitor-parent-pid",type=int);parser.add_argument("--monitor-read-fd",type=int)
    args=parser.parse_args()
    return worker(args) if args.worker else execute()

if __name__=="__main__":
    try:sys.exit(main())
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        print(json.dumps(dict(error=type(exc).__name__,message=str(exc)[:2048],actual_pass_not_asserted=True)),flush=True)
        sys.exit(1)

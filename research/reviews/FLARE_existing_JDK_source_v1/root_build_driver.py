"""Root-only FLARE build admission/launch. Import is inert; no science stage is activated.

The outer RSS/deadline monitor is a polling soft bound, not a kernel quota.
The frozen builder separately supervises and drains each native tool group.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib,json,os,selectors,signal,subprocess,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/FLARE_existing_JDK_source_v1/root_build_driver.py"
PACKET="research/reviews/FLARE_existing_JDK_source_v1"
CONTEXT_REF=dict(path=PACKET+"/root_context_v1.py",sha256="46d4ddfa1c4b6aa20ea0f51be5810c63b9aa44ec8d3adb1e544587c0a066e620")
MANIFEST_REF=dict(path=PACKET+"/source_manifest.json",sha256="60c5619282dd568a10bd43de971cb8283cb0838e5548f6802ef94853c10a1859")
CONFIG_REF=dict(path=PACKET+"/build_inactive.json",sha256="00dd2b1d27f737b9ed5bdd2a26fcc0dfba70423a5d5b41920b7aaed3a64b236f")
RUNTIME_BASIS_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json",sha256="e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602")
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CAP,RSS_LIMIT,DEADLINE=19000000,2000000000,3600
AUX_ALLOWANCE,LOG_LIMIT=524288,262144
ENV=dict(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",PYTHONDONTWRITEBYTECODE="1")

def need(ok,message):
    if not ok:raise ValueError(message)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project-only path")
    return p
def ref(value):
    p=local(value)
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    p=local(record["path"]);need(ref(p)==record and p.stat().st_size<=65536,"Bound compact metadata")
    return json.loads(p.read_text())
def put(path,value,limit=65536):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(data)<=limit,"Precharged compact metadata")
    fd=os.open(local(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    return ref(path)
def unique(records):
    result=[];by_path={}
    for r in records:
        need(set(r)=={"path","sha256"},"Exact reference")
        need(r["path"] not in by_path or by_path[r["path"]]==r,"No contradictory source binding")
        if r not in result:result.append(r);by_path[r["path"]]=r
    need(len(result)<=64,"At most64 actual direct consumer bindings")
    return result
def expiry(value):
    d=datetime.datetime.fromisoformat(value.replace("Z","+00:00"))
    need(d.tzinfo is not None and d>datetime.datetime.now(datetime.timezone.utc),"Unexpired explicit root authority")
    return d
def modules():
    need(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Project cwd and explicit CPU-only visibility")
    for p in (str(ROOT),str(ROOT/"src")):
        if p not in sys.path:sys.path.insert(0,p)
    from arggnn import american_c3_common_v1 as common
    return common
def adapter():
    modules()
    need(ref(GUARD_REF["path"])==GUARD_REF,"Frozen current-credit-v3 source")
    m=importlib.import_module("research.reviews.concurrent_completed_scope_credits_v3.completed_scope_credits_v3")
    m.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    need(all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__ for n in ("canonical","file_bytes","mutex")),
         "Original unmemoized v4 admission and consumer route")
    return m
def actual_runtime():
    c=modules();basis=read(RUNTIME_BASIS_REF);actual=c.runtime_report()
    need(actual["prefix"]==str(ROOT/".runtime/env") and actual["prefix"]==basis["prefix"]
        and actual["CUDA_VISIBLE_DEVICES"]=="" and actual["packages"]==basis["packages"]
        and actual["executable"]==basis["executable"] and actual["python"]==basis["python"],"Exact existing CPU runtime metadata")
    return actual

def admit(args):
    """Only explicit root --admit calls this original, unmemoized reserve path."""
    need(args.packet and args.output and args.job_id and args.expires_at_utc,"Explicit fresh packet/output/job/expiry required")
    expiry(args.expires_at_utc);actual=actual_runtime()
    packet,out=local(args.packet),local(args.output);aux=packet/"worker_aux"
    need(not packet.exists() and not out.exists() and not out.is_relative_to(packet)
         and not packet.is_relative_to(out),"Fresh disjoint output and root metadata packet")
    manifest=read(MANIFEST_REF);cfg=read(CONFIG_REF)
    need(manifest["source_sha256"][CONTEXT_REF["path"]]==CONTEXT_REF["sha256"] and ref(CONTEXT_REF["path"])==CONTEXT_REF,
         "Actual46d4 context; never trust stale inactive request")
    for p,h in manifest["source_sha256"].items():need(ref(p)==dict(path=p,sha256=h),"Every frozen actual source")
    for r in manifest["required_metadata_refs"]:read(r)
    own=ref(SOURCE)
    manifest=dict(manifest,source_sha256=dict(sorted(dict(manifest["source_sha256"],**{SOURCE:own["sha256"]}).items())),
        required_metadata_refs=unique([*manifest["required_metadata_refs"],MANIFEST_REF,CONFIG_REF,RUNTIME_BASIS_REF]),
        root_build_driver_ref=own)
    cfg.update(schema="FLARE_existing_JDK_root_build_active_v1",execution_enabled=True,
        root_host_resource_enforcement_active=True,CPU_threads=4,output_cap_bytes=CAP,
        max_host_RSS_bytes=RSS_LIMIT,outer_process_group_timeout_seconds=DEADLINE,
        RSS_enforcement="outer_owned_process_tree_polling_soft_bound_not_kernel_quota",
        precharged_outer_auxiliary_allowance_bytes=AUX_ALLOWANCE)
    packet.mkdir(parents=True)
    cfg_ref=put(packet/"build_active.json",cfg);runtime_ref=put(packet/"runtime.json",actual)
    manifest_ref=put(packet/"build_source_manifest.json",manifest)
    closure=[dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()]
    helpers=[r for r in closure if any(s in r["path"] for s in ("project_budget","failed_scope_capacity","marker_repair","completed_scope_credits"))]
    direct=unique([own,CONTEXT_REF,cfg_ref,runtime_ref,manifest_ref,GUARD_REF,POLICY_REF,REPAIR_REF,
        *helpers,*closure,*manifest["required_metadata_refs"],cfg["installed_JDK_qualification_ref"]])
    m=adapter()
    # No memo installation precedes, wraps or replaces this reserve.
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    consumers=unique([*direct,*state[3].values()])
    request=dict(key=args.job_id,output=str(out),auxiliary_scopes=[str(aux)],reserved_bytes=CAP,consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=args.expires_at_utc)
    authority_ref=put(packet/"reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=args.job_id,output_directory=out.relative_to(ROOT).as_posix(),
        reserved_new_bytes=CAP,reservation=receipt["value"],expires_at_utc=args.expires_at_utc,
        initial_ledger_pins=receipt["initial_ledger_pins"],completed_overlay_policy_ref=POLICY_REF,
        budget_components=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD_REF))
    storage_ref=put(packet/"storage.json",storage)
    spec=dict(schema="FLARE_existing_JDK_current_v3_root_admission_v1",root_authorizer="root_agent",
        execution_authorized=True,GPU_authorized=False,stage="build",source_ref=CONTEXT_REF,
        stage_source_ref=ref(PACKET+"/build_v1.py"),root_build_driver_ref=own,
        config_ref=cfg_ref,runtime_ref=runtime_ref,source_manifest_ref=manifest_ref,source_closure_refs=closure,
        storage_ref=storage_ref,job_id=args.job_id,output_directory=storage["output_directory"],
        auxiliary_directory=aux.relative_to(ROOT).as_posix(),expires_at_utc=args.expires_at_utc,
        storage_adapter_ref=GUARD_REF,completed_overlay_policy_ref=POLICY_REF,marker_repair_authority_ref=REPAIR_REF,
        mandatory_helper_refs=helpers,installed_JDK_qualification_ref=cfg["installed_JDK_qualification_ref"],
        root_host_resource_enforcement_active=True)
    result=put(packet/"admission.json",spec)
    put(packet/"reserve_receipt.json",receipt)
    return dict(status="admitted_build_only_not_launched",admission_ref=result,consumer_binding_count=len(consumers),reserved_bytes=CAP)

def load_admission(args):
    need(args.admission and args.expected_admission_sha256,"Explicit admission path and actual SHA required")
    record=dict(path=local(args.admission).relative_to(ROOT).as_posix(),sha256=args.expected_admission_sha256)
    spec=read(record);cfg=read(spec["config_ref"]);storage=read(spec["storage_ref"]);manifest=read(spec["source_manifest_ref"])
    need(spec["schema"]=="FLARE_existing_JDK_current_v3_root_admission_v1" and spec["root_authorizer"]=="root_agent"
         and spec["execution_authorized"] is True and spec["GPU_authorized"] is False and spec["stage"]=="build"
         and spec["source_ref"]==CONTEXT_REF and spec["root_build_driver_ref"]==ref(SOURCE)
         and spec["root_host_resource_enforcement_active"] is True,"Exact root build-only driver/context authority")
    expiry(spec["expires_at_utc"])
    need(cfg["execution_enabled"] is True and cfg["CPU_threads"]==4 and cfg["output_cap_bytes"]==CAP
         and cfg["max_host_RSS_bytes"]==RSS_LIMIT and cfg["outer_process_group_timeout_seconds"]==DEADLINE
         and cfg["precharged_outer_auxiliary_allowance_bytes"]==AUX_ALLOWANCE
         and storage["reserved_new_bytes"]==CAP,"Exact build-only resource policy")
    need(storage["reservation"]["auxiliary_scopes"]==[str(local(spec["auxiliary_directory"]))]
         and spec["source_closure_refs"]==[dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()],"Exact reserved auxiliary scope and ordered closure")
    for r in unique([spec["root_build_driver_ref"],CONTEXT_REF,spec["stage_source_ref"],spec["config_ref"],spec["runtime_ref"],
            spec["source_manifest_ref"],GUARD_REF,POLICY_REF,REPAIR_REF,*spec["mandatory_helper_refs"],
            *spec["source_closure_refs"],*manifest["required_metadata_refs"],spec["installed_JDK_qualification_ref"]]):
        need(ref(r["path"])==r and r in storage["reservation"]["consumer_bindings"],"Actual direct complete source/runtime/metadata binding")
    actual_runtime()
    return record,spec,cfg,storage

class WorkerStop(BaseException):pass
def worker(args):
    record,spec,_,_=load_admission(args)
    need(args.monitor_parent_pid==os.getppid() and args.monitor_read_fd is not None,"Worker requires actual outer Popen monitor parent")
    with os.fdopen(args.monitor_read_fd,"rb") as f:ticket=f.read(65)
    need(ticket==(record["sha256"]+"\n").encode(),"Inherited launch-only monitor handshake")
    def stopped(signum,frame):raise WorkerStop("Outer monitor requested worker termination")
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    m=adapter()
    context=importlib.import_module("research.reviews.FLARE_existing_JDK_source_v1.root_context_v1")
    result=context.execute_v1(record,adapter=m)
    print(json.dumps(dict(build_completion_ref=result)),flush=True)
    return 0

def process_info(pid):
    """Read only a known owned PID. Linux stat gives stable start identity and RSS."""
    try:
        fields=(Path("/proc")/str(pid)/"stat").read_text().rsplit(") ",1)[1].split()
        return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]),session=int(fields[3]),
            rss_bytes=max(0,int(fields[21]))*os.sysconf("SC_PAGE_SIZE"))
    except FileNotFoundError:return None

def observe(owned,groups):
    pending=list(owned);seen=set();total=0;races=0
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid);need(len(seen)<=128,"Bounded owned process roster")
        info=process_info(pid)
        if info is None:races+=1;continue
        need(owned[pid] is None or owned[pid]==info["start_ticks"],"Owned PID identity changed")
        owned[pid]=info["start_ticks"];groups.setdefault(info["pgid"],{} )[pid]=info["start_ticks"]
        total+=info["rss_bytes"]
        taskdir=Path("/proc")/str(pid)/"task"
        try:tasks=list(taskdir.iterdir())
        except FileNotFoundError:races+=1;continue
        need(len(tasks)<=2048,"Bounded owned task roster")
        for task in tasks:
            try:children=(task/"children").read_text().split()
            except FileNotFoundError:races+=1;continue
            for child in children:
                child=int(child)
                if child not in owned:owned[child]=None
                pending.append(child)
    # Exited identities remain recorded, but never adopt a reused PID.
    launcher=process_info(os.getpid())
    need(launcher is not None,"Actual outer monitor RSS")
    return total+launcher["rss_bytes"],races

def signal_owned(groups,sig):
    unresolved=[]
    for pgid,members in groups.items():
        try:os.killpg(pgid,0)
        except ProcessLookupError:continue
        proven=False
        for pid,start in members.items():
            row=process_info(pid)
            if row is not None and row["start_ticks"]==start and row["pgid"]==pgid:proven=True;break
        if proven:
            try:os.killpg(pgid,sig)
            except ProcessLookupError:pass
        else:unresolved.append(pgid)  # No signal to an unproven or reused group.
    return unresolved

def absent_groups(groups):
    live=[]
    for pgid in groups:
        try:os.killpg(pgid,0);live.append(pgid)
        except ProcessLookupError:pass
    return live

def launch(args):
    record,spec,_,storage=load_admission(args);m=adapter()
    m.validate_storage_v9(spec,read(spec["config_ref"]),storage,source_ref=CONTEXT_REF,
        config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    aux=local(spec["auxiliary_directory"])
    need(not aux.exists() and not local(spec["output_directory"]).exists(),"Single launch into fresh exclusive scopes")
    # Precharge every outer receipt, retained log and directory before any Popen.
    m.budget_check_v9(storage,boundary=True,required_write_bytes=AUX_ALLOWANCE);aux.mkdir()
    env=os.environ.copy();env.update(ENV)
    runtime=read(spec["runtime_ref"]);rfd,wfd=os.pipe()
    argv=[runtime["executable"],"-B",str(ROOT/SOURCE),"--worker","--admission",record["path"],
        "--expected-admission-sha256",record["sha256"],"--monitor-parent-pid",str(os.getpid()),"--monitor-read-fd",str(rfd)]
    started=utc();clock=time.monotonic();owned={};groups={};peak=races=discarded=0;log=bytearray();error=None
    proc=None;selector=None;minimal_ref=None;rc=None;term_at=None;killed=False
    def retain(chunk):
        nonlocal discarded
        keep=min(len(chunk),LOG_LIMIT-len(log));log.extend(chunk[:keep]);discarded+=len(chunk)-keep
    try:
        os.write(wfd,(record["sha256"]+"\n").encode())
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(rfd,))
        owned[proc.pid]=None;groups[proc.pid]={}
        os.close(rfd);rfd=None;os.close(wfd);wfd=None
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        while proc.poll() is None:
            now=time.monotonic()
            try:
                rss,misses=observe(owned,groups);peak=max(peak,rss);races+=misses
                need(rss<=RSS_LIMIT,"Aggregate owned-tree plus monitor RSS soft bound exceeded")
                need(now-clock<=DEADLINE,"Outer3600s wall deadline exceeded")
                expiry(spec["expires_at_utc"])
            except BaseException as exc:
                if error is None:error=dict(type=type(exc).__name__,message=str(exc)[:2048])
            if error is not None and term_at is None:
                # Worker handler raises; frozen builder finally drains its native group.
                proc.terminate();term_at=now
            if term_at is not None and now-term_at>=40 and not killed:
                signal_owned(groups,signal.SIGKILL);proc.kill();killed=True
            for key,_ in selector.select(.25):
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
    except BaseException as exc:
        error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:
            try:proc.terminate()
            except ProcessLookupError:pass
    finally:
        for fd in (rfd,wfd):
            if fd is not None:
                try:os.close(fd)
                except OSError as exc:
                    error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:
            try:rc=proc.wait(timeout=40 if error else 5)
            except subprocess.TimeoutExpired:
                try:signal_owned(groups,signal.SIGKILL)
                except BaseException as exc:
                    error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
                try:proc.kill()
                except ProcessLookupError:pass
                rc=proc.wait()
            # First post-wait disk write: real wait evidence, before guard, group checks or log writes.
            minimal_ref=put(aux/"actual_wait_minimal.json",dict(schema="FLARE_root_build_actual_direct_wait_minimal_v1",
                actual_direct_wait=True,returncode=rc,pid=proc.pid,worker_pgid=proc.pid,
                admission_ref=record,root_build_driver_ref=spec["root_build_driver_ref"],
                started_utc=started,waited_utc=utc(),elapsed_seconds=time.monotonic()-clock),limit=8192)
    if proc is None:
        put(aux/"spawn_failed.json",dict(actual_direct_wait=False,error=error))
        raise RuntimeError("Popen failed; no actual wait exists")
    # Minimal wait is durable even if any following drain, guard, log or receipt step fails.
    live=absent_groups(groups)
    if live:
        signal_owned(groups,signal.SIGKILL)
        until=time.monotonic()+10
        while live and time.monotonic()<until:time.sleep(.1);live=absent_groups(groups)
    if selector is not None:
        # Bounded terminal drain. Never block on an inherited stdout held by a survivor.
        for _ in range(16):
            ready=selector.select(0)
            if not ready:break
            for key,_ in ready:
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
        selector.close()
    proc.stdout.close()
    need(not live,"Observed owned group unresolved; preserve minimal wait and perform no guard/log writes")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=len(log)+65536)
    logpath=aux/"worker_stdout.txt"
    fd=os.open(logpath,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(log);f.flush();os.fsync(f.fileno())
    terminal=dict(schema="FLARE_root_build_outer_monitor_wait_v1",actual_wait_ref=minimal_ref,
        returncode=rc,error=error,admission_ref=record,argv=argv,observed_owned_groups_ESRCH=True,
        observed_owned_groups=sorted(groups),observed_process_identities=owned,observed_peak_aggregate_RSS_bytes=peak,
        RSS_limit_bytes=RSS_LIMIT,RSS_bound_is_polling_soft_bound_not_kernel_quota=True,
        aggregate_includes_outer_monitor=True,poll_seconds=.25,proc_exit_or_task_read_races=races,
        unobserved_between_poll_processes_possible=True,native_tool_groups_separately_supervised_by_frozen_builder=True,
        wall_deadline_seconds=DEADLINE,termination_grace_seconds=40,elapsed_seconds=time.monotonic()-clock,
        retained_stdout_ref=ref(logpath),discarded_stdout_bytes=discarded)
    terminal_ref=put(aux/"outer_wait.json",terminal)
    m.budget_check_v9(storage,boundary=True)
    need(rc==0 and error is None,"Actual build worker failed; retain all partial artifacts and wait evidence")
    completion=local(spec["output_directory"])/"completed.json"
    need(completion.exists(),"Actual frozen builder completion required")
    return dict(status="completed_actual_build_worker",completion_ref=ref(completion),outer_wait_ref=terminal_ref,
        actual_wait_ref=minimal_ref,automatic_completed_scope_credit=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    for mode in ("admit","launch","worker"):modes.add_argument("--"+mode,action="store_true")
    for name in ("packet","output","job-id","expires-at-utc","admission","expected-admission-sha256"):p.add_argument("--"+name)
    p.add_argument("--monitor-parent-pid",type=int);p.add_argument("--monitor-read-fd",type=int)
    args=p.parse_args()
    result=admit(args) if args.admit else launch(args) if args.launch else worker(args)
    if not args.worker:print(json.dumps(result,sort_keys=True),flush=True)
    return result if args.worker else 0

if __name__=="__main__":raise SystemExit(main())

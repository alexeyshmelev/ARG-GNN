"""H044 root-only minimal admission/launch driver. Default is inert.

Uses current completed-credit guard without memo wrappers, preserves reserve pins,
and buffers logs in RAM through original direct wait and owned-group drain.
"""
from pathlib import Path
import argparse,collections,ctypes,hashlib,importlib.util,json,os,selectors,signal,subprocess,time
from datetime import datetime,timedelta,timezone
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H044_marker_resolution_root_v1/root_driver_v2.py"
CONTEXT="research/reviews/H044_marker_resolution_root_v1/context_factory_v2.py"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"

def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    require(path.resolve()==path and path.is_file(),"Canonical existing file")
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(record):
    require(ref(record["path"])==record,"Pinned root metadata changed")
    return json.loads((ROOT/record["path"]).read_bytes())
def save(folder,name,value):
    path=folder/name;data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as out:out.write(data);out.flush();os.fsync(out.fileno())
    return ref(path)
def unique(records):
    return list({(r["path"],r["sha256"]):r for r in records}.values())
def context():
    s=importlib.util.spec_from_file_location("_H044_root_constants",ROOT/CONTEXT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load_guard(c):
    require(ref(c.GUARD_REF["path"])==c.GUARD_REF,"Current guard changed")
    s=importlib.util.spec_from_file_location("_H044_unwrapped_root_guard",ROOT/c.GUARD_REF["path"])
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.configure_v2(c.REPAIR_REF,policy_ref=c.POLICY_REF);return m
def admit(request_ref):
    r=read(request_ref);c=context();phase=r["phase"]
    require(r["schema"]=="H044_marker_root_admission_request_v1" and r["status"]=="active"
        and r["root_authorizer"]=="root_agent" and phase in ("prepare","train","evaluate"),"Active exact root request")
    require(r["physical_gpu"] is None if phase=="prepare" else r["physical_gpu"] in (4,5),"CPU or physicalGPU4/5")
    packet=ROOT/r["packet_directory"];output=r["output_directory"];aux=r["worker_auxiliary_scope"]
    require(not packet.exists() and not (ROOT/output).exists() and not (ROOT/aux).exists(),"Fresh packet/output/aux, no replay")
    require(packet.resolve()==packet and (ROOT/output).resolve()==ROOT/output and (ROOT/aux).resolve()==ROOT/aux,"Canonical root-owned paths")
    cfg=read(r["config_ref"]);sources=read(cfg["source_manifest_ref"])["source_sha256"]
    require(cfg["schema"]=="H044_marker_execution_config_v1" and cfg["hypothesis_id"]=="H044","H044 config")
    for path,digest in sources.items():require(ref(path)["sha256"]==digest,"Frozen source differs")
    require(sources[CONTEXT]==ref(CONTEXT)["sha256"] and sources[SOURCE]==ref(SOURCE)["sha256"],"Actual context/driver closure")
    metadata=read(r["observation_template_ref"])
    entries=metadata["observation_entry_refs"];cache_sources=metadata["observation_cache_source_refs"]
    require(len(entries)==40 and cache_sources==cfg["observation_cache_source_refs"],"Original40 observed entries")
    source=c.PREP_SOURCE if phase=="prepare" else c.WORKER_SOURCE
    require(type(r["reserved_bytes"]) is int and 0<r["reserved_bytes"]<=cfg["output_cap_bytes"],"Reviewed reserve cap")
    packet.mkdir()
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=24)).isoformat()
    runtime=save(packet,"runtime.json",dict(status="active",device="cpu" if phase=="prepare" else "cuda",
        physical_gpu=r["physical_gpu"],job_id=r["job_id"],output_directory=output,
        process_title=None if phase=="prepare" else "AI Auto Research Pipeline Test",
        created_at_utc=now.isoformat(),expires_at_utc=expiry,completed_overlay_policy_ref=c.POLICY_REF))
    refs={k:metadata["refs"][k] for k in ("registration","generation_config","observation_cache")}
    refs.update(config=r["config_ref"],runtime=runtime)
    terminal={}
    if phase!="prepare":
        for k in ("preparation_completion_ref","preparation_actual_wait_ref"):terminal[k]=r[k];read(r[k])
        require(r.get("phase_target_bindings_ref") is not None,"Explicit phase-only target metadata")
        bindings=read(r["phase_target_bindings_ref"])
        require(bindings["outer_included"] is False and len(bindings["parents"])==(32 if phase=="train" else 8),"Exact phase-only truth metadata")
        refs["target_bindings"]=r["phase_target_bindings_ref"]
    if phase=="evaluate":
        for k in ("train_completion_ref","train_actual_wait_ref"):terminal[k]=r[k];read(r[k])
    closure=dict(schema="H044_marker_exact_consumer_closure_v1",hypothesis_id="H044",phase=phase,
        source_manifest_ref=cfg["source_manifest_ref"],source_sha256=sources,refs=dict(refs),
        observation_entry_refs=entries,observation_cache_source_refs=cache_sources,**terminal)
    closure_ref=save(packet,"consumer_manifest.json",closure)
    m=load_guard(c)
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    view,budget_bindings=state[2],state[3]
    mandatory=[ref(source),r["config_ref"],runtime,ref(CONTEXT),ref(SOURCE),cfg["source_manifest_ref"],closure_ref,
        c.GUARD_REF,c.POLICY_REF,c.REPAIR_REF,c.MEMO_REF,c.V4_REF,
        ref("src/arggnn/marker_resolution_readout_v1.py"),
        ref("src/arggnn/project_budget_v9_consumer.py"),
        ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py")]
    if phase!="prepare":mandatory += [c.LAUNCHER_REF,c.AUTH_REF,cfg["base_fit_config"]]
    consumers=unique(mandatory+list(terminal.values())+list(budget_bindings.values()))
    require(len(consumers)<=64,"Guard direct-ref bound")
    request=dict(key="H044/"+r["job_id"],output=str(ROOT/output),auxiliary_scopes=[str(ROOT/aux)],
        reserved_bytes=r["reserved_bytes"],consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=budget_bindings)
    authority.update(active=True,expires_at_utc=expiry)
    receipt=m.apply_reserve_v1(save(packet,"reserve_authority.json",authority),
        expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=r["job_id"],output_directory=output,reserved_new_bytes=r["reserved_bytes"],
        reservation=receipt["value"],expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],
        completed_overlay_policy_ref=c.POLICY_REF,budget_components=dict(
            implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=c.GUARD_REF))
    refs["storage"]=save(packet,"storage.json",storage)
    spec=dict(schema="H044_marker_root_execution_input_v1",hypothesis_id="H044",root_authorizer="root_agent",
        status="active",phase=phase,seed=101,device="cpu" if phase=="prepare" else "cuda",physical_gpu=r["physical_gpu"],
        fit_authorized=phase!="prepare",pyg_authorized=True,job_id=r["job_id"],output_directory=output,
        worker_auxiliary_scope=aux,created_at_utc=now.isoformat(),expires_at_utc=expiry,refs=refs,source_sha256=sources,
        root_context_module_ref=ref(CONTEXT),root_driver_ref=ref(SOURCE),main_source_ref=ref(source),
        consumer_manifest_ref=closure_ref,observation_entry_refs=entries,observation_cache_source_refs=cache_sources,
        max_cpu_rss_bytes=r["max_cpu_rss_bytes"],max_cuda_reserved_bytes=r.get("max_cuda_reserved_bytes",0),
        GPU45_authorization_ref=c.AUTH_REF,completed_overlay_policy_ref=c.POLICY_REF,**terminal)
    input_ref=save(packet,"input.json",spec)
    admitted=dict(schema="H044_marker_root_admitted_v1",input_ref=input_ref,driver_ref=ref(SOURCE),
        request_ref=request_ref,consumer_binding_count=len(consumers),phase=phase,physical_gpu=r["physical_gpu"],
        auxiliary_scope=aux,scientific_execution_started=False)
    save(packet,"admitted.json",admitted)
    print(json.dumps(admitted),flush=True)
    return admitted


def _stat(pid):
    try:
        values=Path("/proc/%d/stat"%pid).read_text().rsplit(")",1)[1].split()
        return dict(pid=pid,state=values[0],ppid=int(values[1]),group=int(values[2]),session=int(values[3]),
            start=int(values[19]),rss=int(values[21])*os.sysconf("SC_PAGE_SIZE"))
    except FileNotFoundError:return None

def _tree(known,groups):
    queue=[(pid,None) for pid in known]
    # This subreaper starts without unrelated children. A newly seen identity
    # must still be a direct child of the exact parent that enumerated it.
    try:queue.extend((int(p),os.getpid()) for p in Path("/proc/self/task/%d/children"%os.getpid()).read_text().split())
    except FileNotFoundError:pass
    visited=set();rss=0;reused=[]
    while queue:
        pid,parent=queue.pop()
        if pid in visited:continue
        visited.add(pid);row=_stat(pid)
        if row is None:continue
        actual_child=parent is not None and row["ppid"]==parent
        old=known.get(pid)
        if old is not None and old["start"]!=row["start"] and not actual_child:
            reused.append(pid)
            continue  # never traverse or signal an unrelated PID replacement
        if old is None and not actual_child:continue
        if old is not None:groups.add(old["group"])
        known[pid]=row;groups.add(row["group"]);rss+=row["rss"]
        try:
            for task in Path("/proc/%d/task"%pid).iterdir():
                try:queue.extend((int(p),pid) for p in (task/"children").read_text().split())
                except FileNotFoundError:pass
        except FileNotFoundError:pass
    if reused:raise ValueError("Historical owned PID reused outside owned tree; replacement excluded after adopted children registered")
    own=_stat(os.getpid())
    return rss+(own["rss"] if own is not None else 0)

def _signal_owned(known,signum):
    for pid,old in list(known.items()):
        now=_stat(pid)
        if now is not None and old is not None and old["start"]==now["start"]:
            try:os.kill(pid,signum)
            except ProcessLookupError:pass

def _absent_group(group):
    try:os.killpg(group,0);return False
    except ProcessLookupError:return True

class _RAMLog:
    def __init__(self,cap):self.cap=cap;self.parts=collections.deque();self.kept=0;self.total=0
    def add(self,data):
        self.total+=len(data);self.parts.append(data);self.kept+=len(data)
        while self.parts and self.kept>self.cap:
            first=self.parts.popleft();over=self.kept-self.cap
            if len(first)>over:self.parts.appendleft(first[over:]);self.kept-=over
            else:self.kept-=len(first)
    def record(self):
        return dict(tail=b"".join(self.parts).decode("utf-8",errors="replace"),
            total_pipe_bytes=self.total,retained_raw_bytes=self.kept,cap_bytes=self.cap,
            discarded_earlier_bytes=self.total-self.kept,buffered_in_RAM_until_actual_drain=True)

def _subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,"Linux subreaper needed to adopt/drain native DataLoader descendants")

def _children():
    raw=Path("/proc/self/task/%d/children"%os.getpid()).read_text().strip()
    return [int(p) for p in raw.split()] if raw else []

def launch(admitted_ref):
    admitted=read(admitted_ref);spec=read(admitted["input_ref"]);c=context()
    require(admitted["driver_ref"]==ref(SOURCE) and spec["root_driver_ref"]==ref(SOURCE),"Exact admitted root driver")
    require(not (ROOT/spec["output_directory"]).exists(),"No execution replay")
    m=load_guard(c);storage=read(spec["refs"]["storage"])
    # Use the exact original reserve value and initial_ledger_pins. Never rebase.
    m.budget_check_v9(storage,boundary=True,required_write_bytes=17000000)
    auxiliary=ROOT/spec["worker_auxiliary_scope"];auxiliary.mkdir()
    env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
    env["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(c.POLICY_REF,sort_keys=True)
    phase=spec["phase"];worker=c.PREP_SOURCE if phase=="prepare" else c.WORKER_SOURCE
    args=["--execute","--input",admitted["input_ref"]["path"],"--expected-input-sha256",admitted["input_ref"]["sha256"]]
    if phase=="prepare":
        env["CUDA_VISIBLE_DEVICES"]="";env["OMP_NUM_THREADS"]="1";env["MKL_NUM_THREADS"]="1"
        cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m",worker[:-3].replace("src/","").replace("/",".")]+args
    else:
        cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m",c.LAUNCHER_REF["path"][:-3].replace("/","."),
            "--gpu",str(spec["physical_gpu"]),"--job",spec["job_id"],
            "--storage-record",spec["refs"]["storage"]["path"],"--expected-storage-sha256",spec["refs"]["storage"]["sha256"],
            "--expected-budget-source-sha256",CORE,"--expected-launcher-sha256",BASE,
            worker[:-3].replace("src/","").replace("/",".")]+args
    _subreaper();require(not _children(),"Fresh root supervisor without unrelated child processes")
    log=_RAMLog(2000000);start=time.monotonic()
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    first=_stat(proc.pid);require(first is not None,"Actual launcher identity")
    known={proc.pid:first};groups={first["group"]}
    launch_record=dict(schema="H044_marker_original_launch_v1",launcher_pid=proc.pid,launcher_start=first["start"],
        direct_child=True,phase=phase,physical_gpu=spec["physical_gpu"],input_ref=admitted["input_ref"],
        command=cmd,actual_wait_pending=True,created_at_utc=datetime.now(timezone.utc).isoformat())
    # Small root metadata only; no active output/log/storage scan.
    launch_path=ROOT/admitted_ref["path"];launch_ref=save(launch_path.parent,"launch.json",launch_record)
    selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ);eof=False;peak=0;error=None
    try:
        while not eof or proc.poll() is None:
            peak=max(peak,_tree(known,groups))
            require(peak<=spec["max_cpu_rss_bytes"],"Owned process-tree RSS cap")
            require(time.monotonic()-start<86400,"Root admission time bound")
            for key,_ in selector.select(timeout=.25):
                data=os.read(key.fileobj.fileno(),65536)
                if data:log.add(data)
                else:selector.unregister(key.fileobj);eof=True
        rc=proc.wait()
    except BaseException as exc:
        error=type(exc).__name__+": "+str(exc)
        _tree(known,groups);_signal_owned(known,signal.SIGTERM)
        deadline=time.monotonic()+10
        while proc.poll() is None and time.monotonic()<deadline:time.sleep(.1)
        _signal_owned(known,signal.SIGKILL);rc=proc.wait()
    finally:selector.close();proc.stdout.close()
    deadline=time.monotonic()+30
    while True:
        _tree(known,groups)
        while True:
            try:
                pid,_=os.waitpid(-1,os.WNOHANG)
                if pid==0:break
            except ChildProcessError:break
        drained=not _children() and all(_absent_group(g) for g in groups)
        if drained:break
        if time.monotonic()>=deadline:
            _signal_owned(known,signal.SIGKILL)
            error=error or "Owned descendants did not drain within30seconds"
            break
        time.sleep(.1)
    require(drained,"No durable log/terminal certificate before complete owned-group drain")
    # Direct wait/drain have actually happened. Only now perform storage boundary and durable logging.
    m.budget_check_v9(storage,boundary=True,required_write_bytes=17000000)
    log_ref=save(auxiliary,"launch_log.json",log.record())
    completion=ROOT/spec["output_directory"]/"completed.json"
    waited=dict(schema="H044_marker_original_direct_wait_v1",launcher_pid=proc.pid,launcher_start=first["start"],
        launch_ref=launch_ref,returncode=rc,actual_direct_launcher_wait=True,owned_group_drained=True,
        owned_process_identities=[dict(pid=p,start=x["start"],group=x["group"]) for p,x in sorted(known.items())],
        historical_owned_groups=sorted(groups),completed_ref=ref(completion) if completion.is_file() else None,
        phase=phase,physical_gpu=spec["physical_gpu"],log_ref=log_ref,peak_owned_RSS_bytes=peak,
        elapsed_seconds=time.monotonic()-start,error=error,waited_utc=datetime.now(timezone.utc).isoformat())
    wait_ref=save(launch_path.parent,"actual_wait.json",waited)
    print(json.dumps(dict(actual_wait_ref=wait_ref,returncode=rc,owned_group_drained=True,error=error)),flush=True)
    return rc or (1 if error else 0)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("admit","launch"));parser.add_argument("--execute",action="store_true")
    parser.add_argument("--record");parser.add_argument("--sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed",source=SOURCE,root_owns_execution=True)));return 0
    require(args.record and args.sha256,"Exact active root request/admitted ref")
    record=dict(path=args.record,sha256=args.sha256)
    if args.action=="admit":admit(record);return 0
    return launch(record)
if __name__=="__main__":raise SystemExit(main())

"""Root-only980GB admission/IO/CPU launch for unchanged H018 inference_v2.

Import is inert. --admit reserves only20MB; --launch is separate. Worker blocks
on an inherited pipe until parent seeds real owned PID/RSS observations. RSS
polling is a soft bound, not a kernel quota. No scientific source is modified.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib,json,os,resource,selectors,signal,subprocess,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
SOURCE="research/reviews/rootH018_IO_v1/root_io_v1.py"
SCIENCE_REF=dict(path="research/reviews/H018_H045_empirical_application_source_v1/inference_v2.py",sha256="a8223979ee775d0aae83f576d48d3cbec482eb03022397acf1d684a08d50f227")
CFG_REF=dict(path="research/reviews/H018_H045_empirical_application_source_v1/config.active_template_v2.json",sha256="97ab5f019a672147acf15a84e77bf4dd8cb690300991d88313cb105caee7d8a0")
SCI_MANIFEST_REF=dict(path="research/reviews/H018_H045_empirical_application_source_v1/source_manifest_v2.json",sha256="0dc5a040f018afe43b1e34b5ec6c985d2c07994398c6fdfd010b4440497aad65")
MONITOR_REF=dict(path="research/reviews/FLARE_existing_JDK_source_v1/root_build_driver.py",sha256="d7afaf5c6c6976a5c4804fe41d320f8609fd346d3f4ed0bca29235001d7fe561")
COMMON_REF=dict(path="src/arggnn/american_c3_common_v1.py",sha256="615d3892f1df7e9de39e57991eab883b2eca1af9d13e424661541aac4b2d2abf")
GUARD_REF=dict(path="research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py",sha256="fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56")
POLICY_REF=dict(path="research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json",sha256="eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CAP,RSS,DEADLINE=20000000,2000000000,1800
AUX,LOG_CAP=524288,262144
ENV=dict(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")

def need(ok,message):
    if not ok:raise ValueError(message)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project-only path")
    return p
def ref(value):
    p=local(value);return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def immutable_bytes(record,cap):
    need(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact path/SHA reference")
    p=local(record["path"]);before=p.lstat()
    signature=lambda st:(st.st_dev,st.st_ino,st.st_mode,st.st_nlink,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    need(p.is_file() and not p.is_symlink() and before.st_nlink==1 and before.st_size<=cap,"Bounded singly linked regular original file")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,"rb") as stream:
        need(signature(os.fstat(stream.fileno()))==signature(before),"Original read identity")
        raw=stream.read(cap+1);after=os.fstat(stream.fileno())
    need(len(raw)<=cap and signature(before)==signature(after)==signature(p.lstat())
        and hashlib.sha256(raw).hexdigest()==record["sha256"],"Exact immutable read")
    return raw
def read(record):
    need(Path(record["path"]).suffix==".json","Bounded JSON metadata")
    return json.loads(immutable_bytes(record,1048576))
def put(path,value,cap=65536):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(raw)<=cap,"Compact precharged root metadata")
    fd=os.open(local(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(path)
def unique(records):
    values={}
    for r in records:
        need(set(r)=={"path","sha256"} and Path(r["path"]).suffix in (".py",".sh",".json"),"Only directly admissible metadata/source refs")
        need(r["path"] not in values or values[r["path"]]==r,"No conflicting path hash")
        values[r["path"]]=r
    result=list(values.values());need(len(result)<=64,"At most64 complete actual consumer bindings");return result
def expiry(value):
    stamp=datetime.datetime.fromisoformat(value.replace("Z","+00:00"))
    need(stamp.tzinfo is not None and stamp>datetime.datetime.now(datetime.timezone.utc),"Unexpired root authority")
def cpu():
    need(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Project cwd and explicit empty CUDA visibility")
def monitor_helpers():
    need(ref(MONITOR_REF["path"])==MONITOR_REF and ref(COMMON_REF["path"])==COMMON_REF,"Frozen reused monitor/runtime helpers")
    return importlib.import_module("research.reviews.FLARE_existing_JDK_source_v1.root_build_driver")
def configured_guard():
    need(all(ref(r["path"])==r for r in (GUARD_REF,POLICY_REF,REPAIR_REF,V4_REF)),"Actual active980GB policy/guard/repair/original-v4 sources")
    m=importlib.import_module("research.reviews.prospective_980GB_capacity_source_v1.prospective_980gb_v1")
    m.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    need(m.MODE=="prospective_980gb_v1" and all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Original unmemoized v4 under actual980GB guard")
    return m

def admit(args):
    cpu();need(args.packet and args.output and args.job_id and args.expires_at_utc,"Explicit fresh packet/output/job/expiry")
    expiry(args.expires_at_utc);packet,out=local(args.packet),local(args.output);aux=packet/"worker_aux"
    need(not packet.exists() and not out.exists() and not out.is_relative_to(packet) and not packet.is_relative_to(out),"Fresh disjoint scopes")
    cfg=read(CFG_REF);science=read(SCI_MANIFEST_REF)
    need(cfg["source_manifest_ref"]==SCI_MANIFEST_REF and ref(SCIENCE_REF["path"])==SCIENCE_REF
        and cfg["output_cap_bytes"]==CAP and cfg["max_host_RSS_bytes"]==RSS and cfg["wall_deadline_seconds"]==DEADLINE,
        "Unchanged final scientific config/producer")
    for p,h in science["source_sha256"].items():need(ref(p)==dict(path=p,sha256=h),"Every unchanged scientific source")
    for r in science["required_metadata_refs"]:immutable_bytes(r,33554432)
    b=monitor_helpers();runtime=b.actual_runtime();m=configured_guard()
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    resource_refs=unique(list(state[3].values()))
    own=ref(SOURCE);code=dict(science["source_sha256"])
    for r in unique([own,MONITOR_REF,COMMON_REF,GUARD_REF,V4_REF,*resource_refs]):
        if Path(r["path"]).suffix==".py":code[r["path"]]=r["sha256"]
    metadata=unique([*science["required_metadata_refs"],CFG_REF,SCI_MANIFEST_REF,POLICY_REF,REPAIR_REF,
        *[r for r in resource_refs if Path(r["path"]).suffix!=".py"]])
    packet.mkdir(parents=True);runtime_ref=put(packet/"runtime.json",runtime)
    io_manifest=put(packet/"IO_source_manifest.json",dict(schema="rootH018_actual980_IO_source_closure_v1",
        source_sha256=dict(sorted(code.items())),required_metadata_refs=metadata,scientific_manifest_ref=SCI_MANIFEST_REF,
        actual_guard_ref=GUARD_REF,actual_policy_ref=POLICY_REF,CPU_only=True))
    consumers=unique([*[dict(path=p,sha256=h) for p,h in code.items()],*metadata,runtime_ref,io_manifest,*resource_refs])
    request=dict(key=args.job_id,output=str(out),auxiliary_scopes=[str(aux)],reserved_bytes=CAP,consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=args.expires_at_utc)
    authority_ref=put(packet/"reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,
        job_id=args.job_id,output_directory=out.relative_to(ROOT).as_posix(),reserved_new_bytes=CAP,
        reservation=receipt["value"],expires_at_utc=args.expires_at_utc,initial_ledger_pins=receipt["initial_ledger_pins"],
        completed_overlay_policy_ref=POLICY_REF,budget_components=dict(implementation=ref(m.budget.SOURCE_REL),
            launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD_REF))
    storage_ref=put(packet/"storage.json",storage)
    spec=dict(schema="H018_H045_empirical_CPU_application_root_v1",root_authorizer="root_agent",execution_authorized=True,
        GPU_authorized=False,optimization_authorized=False,root_host_resource_enforcement_active=True,
        producer_ref=SCIENCE_REF,config_ref=CFG_REF,runtime_ref=runtime_ref,storage_ref=storage_ref,storage_adapter_ref=GUARD_REF,
        io_context_ref=own,io_source_manifest_ref=io_manifest,resource_binding_refs=unique([*resource_refs,V4_REF,MONITOR_REF,COMMON_REF,io_manifest]),
        source_manifest_ref=SCI_MANIFEST_REF,job_id=args.job_id,output_directory=storage["output_directory"],
        auxiliary_directory=aux.relative_to(ROOT).as_posix(),expires_at_utc=args.expires_at_utc)
    input_ref=put(packet/"input.json",spec);put(packet/"reserve_receipt.json",receipt)
    return dict(status="admitted20MB_not_launched",input_ref=input_ref,consumer_binding_count=len(consumers),
        guard_ref=GUARD_REF,policy_ref=POLICY_REF,reserved_bytes=CAP)

def load_input(args):
    cpu();need(args.input and args.expected_sha256,"Exact input path/SHA")
    record=dict(path=local(args.input).relative_to(ROOT).as_posix(),sha256=args.expected_sha256)
    spec=read(record);cfg=read(spec["config_ref"]);storage=read(spec["storage_ref"]);expiry(spec["expires_at_utc"])
    need(spec["schema"]=="H018_H045_empirical_CPU_application_root_v1" and spec["root_authorizer"]=="root_agent"
        and spec["execution_authorized"] is True and spec["GPU_authorized"] is False and spec["optimization_authorized"] is False
        and spec["root_host_resource_enforcement_active"] is True and spec["io_context_ref"]==ref(SOURCE)
        and spec["producer_ref"]==SCIENCE_REF and spec["config_ref"]==CFG_REF and spec["source_manifest_ref"]==SCI_MANIFEST_REF
        and spec["storage_adapter_ref"]==GUARD_REF and storage["reserved_new_bytes"]==CAP
        and storage["accounting_mode"]=="prospective_980gb_v1" and storage["completed_overlay_policy_ref"]==POLICY_REF,
        "Exact root980GB CPU IO/scientific authority")
    need(storage["reservation"]["auxiliary_scopes"]==[str(local(spec["auxiliary_directory"]))],"Exact charged auxiliary scope")
    manifest=read(spec["io_source_manifest_ref"]);bindings=storage["reservation"]["consumer_bindings"]
    required=unique([spec["io_context_ref"],SCIENCE_REF,CFG_REF,SCI_MANIFEST_REF,spec["runtime_ref"],spec["io_source_manifest_ref"],
        *spec["resource_binding_refs"],*[dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()],*manifest["required_metadata_refs"]])
    need(len(bindings)<=64 and all(r in bindings and ref(r["path"])==r for r in required),"All actual source/runtime/helper/metadata bindings")
    return record,spec,cfg,storage

class Reader:
    def __init__(self,records,bank_ref,checkpoint_ref):
        self.allowed={r["path"]:r for r in records};self.bank_ref=bank_ref;self.checkpoint_ref=checkpoint_ref
        for r in (bank_ref,checkpoint_ref):self.allowed[r["path"]]=r
    def read(self,record):
        need(record==self.allowed.get(record.get("path")),"Explicit actual admitted metadata or selected payload only")
        if record==self.bank_ref:cap=33554432
        elif record==self.checkpoint_ref:cap=1048576
        else:
            need(Path(record["path"]).suffix in (".py",".sh",".json"),"No other payload capability")
            cap=33554432
        return immutable_bytes(record,cap)

def worker(args):
    # This blocks before configuration, tensor imports or any scientific payload.
    need(args.monitor_parent_pid==os.getppid() and args.monitor_read_fd is not None,"Actual outer monitor parent")
    with os.fdopen(args.monitor_read_fd,"rb") as stream:ticket=stream.read(65)
    need(ticket==(args.expected_sha256+"\n").encode(),"Parent seeded own PID/RSS before release")
    b=monitor_helpers()
    def stop(signum,frame):raise b.WorkerStop("Root outer monitor requested termination")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic();record,spec,cfg,storage=load_input(args);m=configured_guard()
    m.validate_storage_v9(spec,cfg,storage,source_ref=SCIENCE_REF,config_ref=CFG_REF,runtime_ref=spec["runtime_ref"])
    b.actual_runtime()
    reader=Reader([*storage["reservation"]["consumer_bindings"],record,spec["storage_ref"]],cfg["bank_ref"],cfg["checkpoint_ref"])
    checks=[];out=local(spec["output_directory"]);need(not out.exists(),"Fresh exclusive actual prediction directory")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=8192);out.mkdir()
    def before(stage,plan):
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        need(peak<=RSS and time.monotonic()-started<=DEADLINE,"Worker actual peak RSS/deadline")
        expiry(spec["expires_at_utc"]);growth=plan.get("required_write_bytes",0)
        need(type(growth) is int and growth>=0,"Actual planned growth")
        checked=m.budget_check_v9(storage,boundary=True,required_write_bytes=growth)
        need(checked["reservation"]==storage["reservation"],"Actual guard returned exact charged reservation")
        need(len(checks)<64,"Bounded root resource receipt")
        checks.append(dict(stage=stage,peak_worker_RSS_bytes=peak,elapsed_seconds=time.monotonic()-started,planned_growth_bytes=growth))
    used=0
    def writer(name,raw):
        nonlocal used
        need(isinstance(name,str) and Path(name).name==name and name not in (".","..")
            and isinstance(raw,bytes) and used+len(raw)<=6000000,"Compact basename-only complete raw outputs")
        before("root_writer_boundary",dict(required_write_bytes=len(raw)+8192))
        fd=os.open(out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        used+=len(raw);actual=ref(out/name)
        need(actual["sha256"]==hashlib.sha256(raw).hexdigest(),"Actual fsynced root output SHA");return actual
    context=dict(root_authorizer="root_agent",admitted=True,input_ref=record,source_ref=SCIENCE_REF,config_ref=CFG_REF,
        runtime_ref=spec["runtime_ref"],cfg=cfg,consumer_bindings=storage["reservation"]["consumer_bindings"],
        io_context_ref=spec["io_context_ref"],resource_binding_refs=spec["resource_binding_refs"],reader=reader,writer=writer,before_stage=before)
    science=importlib.import_module("research.reviews.H018_H045_empirical_application_source_v1.inference_v2")
    result=science.execute_v1(context)
    need(result==ref(out/"completed.json"),"Actual original scientific completion return")
    raw=(json.dumps(dict(schema="rootH018_actual980_IO_worker_receipt_v1",scientific_completion_ref=result,
        io_context_ref=spec["io_context_ref"],actual_guard_ref=GUARD_REF,actual_policy_ref=POLICY_REF,
        resource_checks=checks,elapsed_seconds=time.monotonic()-started),sort_keys=True,indent=2)+"\n").encode()
    receipt=writer("root_IO_worker_receipt.json",raw)
    print(json.dumps(dict(scientific_completion_ref=result,IO_worker_receipt_ref=receipt)),flush=True);return 0

def launch(args):
    record,spec,cfg,storage=load_input(args);b=monitor_helpers();m=configured_guard()
    m.validate_storage_v9(spec,cfg,storage,source_ref=SCIENCE_REF,config_ref=CFG_REF,runtime_ref=spec["runtime_ref"])
    aux=local(spec["auxiliary_directory"])
    need(not aux.exists() and not local(spec["output_directory"]).exists(),"Single launch; fresh output and auxiliary")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=AUX);aux.mkdir()
    runtime=read(spec["runtime_ref"]);env=os.environ.copy();env.update(ENV)
    rfd,wfd=os.pipe()
    argv=[runtime["executable"],"-B","-m","research.reviews.rootH018_IO_v1.root_io_v1","--worker",
        "--input",record["path"],"--expected-sha256",record["sha256"],"--monitor-parent-pid",str(os.getpid()),"--monitor-read-fd",str(rfd)]
    proc=None;selector=None;owned={};groups={};peak=races=discarded=0;log=bytearray();error=None
    started=utc();clock=time.monotonic();term_at=None;killed=False;rc=None;minimal_ref=None;seed_ref=None;released=False
    def retain(chunk):
        nonlocal discarded
        keep=min(len(chunk),LOG_CAP-len(log));log.extend(chunk[:keep]);discarded+=len(chunk)-keep
    try:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(rfd,))
        row=b.process_info(proc.pid)
        need(row is not None and row["pgid"]==proc.pid and row["session"]==proc.pid,"Seed actual owned PID identity before child release")
        owned[proc.pid]=row["start_ticks"];groups[proc.pid]={proc.pid:row["start_ticks"]}
        initial,misses=b.observe(owned,groups);peak=initial;races+=misses
        need(initial<=RSS,"Initial aggregate owned+monitor RSS before any child compute")
        seed_ref=put(aux/"launch_monitor_seed.json",dict(schema="rootH018_precompute_owned_monitor_seed_v1",worker_pid=proc.pid,
            worker_start_ticks=row["start_ticks"],worker_pgid=row["pgid"],worker_session=row["session"],supervisor_pid=os.getpid(),
            initial_aggregate_RSS_bytes=initial,actual_owned_roster_seeded=True,child_compute_released=False,input_ref=record,checked_utc=utc()))
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        # Only now may worker configure guards/import tensor libraries/read payloads.
        os.write(wfd,(record["sha256"]+"\n").encode());released=True
        os.close(wfd);wfd=None;os.close(rfd);rfd=None
        while proc.poll() is None:
            now=time.monotonic()
            try:
                rss,misses=b.observe(owned,groups);peak=max(peak,rss);races+=misses
                need(rss<=RSS and now-clock<=DEADLINE,"Aggregate2GB RSS soft bound or1800s deadline exceeded")
                expiry(spec["expires_at_utc"])
            except BaseException as exc:
                if error is None:error=dict(type=type(exc).__name__,message=str(exc)[:2048])
            if error is not None and term_at is None:proc.terminate();term_at=now
            if term_at is not None and now-term_at>=30 and not killed:
                b.signal_owned(groups,signal.SIGKILL);proc.kill();killed=True
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
                except OSError as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:
            try:rc=proc.wait(timeout=30 if error else 5)
            except subprocess.TimeoutExpired:
                try:b.signal_owned(groups,signal.SIGKILL)
                except BaseException as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
                try:proc.kill()
                except ProcessLookupError:pass
                rc=proc.wait()
            # First postwait write; no guard, drain certificate or log precedes this.
            minimal_ref=put(aux/"actual_wait_minimal.json",dict(schema="rootH018_actual_direct_wait_minimal_v1",actual_direct_wait=True,
                returncode=rc,pid=proc.pid,worker_pgid=proc.pid,input_ref=record,io_context_ref=spec["io_context_ref"],
                started_utc=started,waited_utc=utc(),elapsed_seconds=time.monotonic()-clock),cap=8192)
    if proc is None:
        put(aux/"spawn_failed.json",dict(actual_direct_wait=False,error=error));raise RuntimeError("Popen failed; no actual wait exists")
    live=b.absent_groups(groups)
    if live:
        b.signal_owned(groups,signal.SIGKILL);until=time.monotonic()+10
        while live and time.monotonic()<until:time.sleep(.1);live=b.absent_groups(groups)
    if selector is not None:
        for _ in range(16):
            ready=selector.select(0)
            if not ready:break
            for key,_ in ready:
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
        selector.close()
    proc.stdout.close();need(not live,"Observed owned group unresolved; minimal wait retained, no final guard/log")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=len(log)+65536)
    fd=os.open(aux/"worker_stdout.txt",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(log);f.flush();os.fsync(f.fileno())
    terminal=put(aux/"outer_wait.json",dict(schema="rootH018_actual980_outer_monitor_wait_v1",actual_wait_ref=minimal_ref,
        input_ref=record,returncode=rc,error=error,argv=argv,launch_monitor_seed_ref=seed_ref,child_compute_released=released,
        observed_owned_groups_ESRCH=True,observed_owned_groups=sorted(groups),observed_process_identities=owned,
        observed_peak_aggregate_RSS_bytes=peak,RSS_limit_bytes=RSS,RSS_is_polling_soft_bound_not_kernel_quota=True,
        aggregate_includes_outer_monitor=True,poll_seconds=.25,proc_exit_or_task_read_races=races,
        unobserved_between_poll_processes_possible=True,wall_deadline_seconds=DEADLINE,termination_grace_seconds=30,
        elapsed_seconds=time.monotonic()-clock,retained_stdout_ref=ref(aux/"worker_stdout.txt"),discarded_stdout_bytes=discarded))
    m.budget_check_v9(storage,boundary=True)
    need(rc==0 and error is None,"Actual empirical CPU worker failed; preserve partials/direct wait, no fallback")
    return dict(status="completed_actual_empirical_CPU_application",scientific_completion_ref=ref(local(spec["output_directory"])/"completed.json"),
        actual_wait_ref=minimal_ref,outer_wait_ref=terminal,accuracy_available=False,automatic_completed_scope_credit=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for n in ("admit","launch","worker"):g.add_argument("--"+n,action="store_true")
    for n in ("packet","output","job-id","expires-at-utc","input","expected-sha256"):p.add_argument("--"+n)
    p.add_argument("--monitor-parent-pid",type=int);p.add_argument("--monitor-read-fd",type=int)
    args=p.parse_args();result=admit(args) if args.admit else launch(args) if args.launch else worker(args)
    if not args.worker:print(json.dumps(result,sort_keys=True),flush=True)
    return result if args.worker else 0

if __name__=="__main__":raise SystemExit(main())

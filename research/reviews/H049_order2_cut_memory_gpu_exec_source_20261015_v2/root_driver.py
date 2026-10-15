"""Additive H049 canonical GPU successor with exact root-declared preparation provenance."""
from pathlib import Path
import argparse,collections,ctypes,hashlib,importlib.util,json,os,selectors,signal,subprocess,time,sys
from datetime import datetime,timedelta,timezone
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1.root_common_v1 import (
    HERE,require,ref,read,save,unique,load_guard_v1,count_plan_v1,
    policy_env,GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF,CORE,BASE,
    validate_eval_train_io_v1)
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import producer_v1 as worker
SOURCE="research/reviews/H049_order2_cut_memory_gpu_exec_source_20261015_v2/root_driver.py"
CONTEXT="research/reviews/H049_order2_cut_memory_gpu_exec_source_20261015_v2/context_factory_v2.py"
PREP_SOURCE=None  # This driver cannot prepare; actual provenance comes from the active config.
WORKER_SOURCE=HERE+"/producer_v1.py"
CANONICAL_DISPATCHER_REF={'path': 'src/arggnn/H049_order2_cut_memory_exec_v2.py', 'sha256': 'fed50df6fba72699a87089a3d76984dd56e5d8f9897d820871fb4734a1ae785b'}


def context():
    from research.reviews.H049_order2_cut_memory_gpu_exec_source_20261015_v2 import context_factory_v2
    return context_factory_v2


def bound_source_bytes_v2(record):
    require(ref(record["path"])==record,"Exact declared preparation source bytes")
    return (ROOT/record["path"]).read_bytes()


def load_guard(c):
    return load_guard_v1(consumer=False)[0]


def plan_request_v1(r):
    require(r["phase"] in ("train","evaluate"),"GPU successor phases only")
    cfg=read(r["config_ref"])
    counts=read(cfg["count_manifest_ref"])
    marker=read(cfg["preparation_completion_ref"])
    return count_plan_v1(counts,r["phase"],marker,cfg)


def admit(request_ref):
    r=read(request_ref);phase=r["phase"]
    require(r.get("schema")=="H049_order2_root_admission_request_v1"
        and r.get("status")=="active" and r.get("root_authorizer")=="root_agent"
        and phase in ("train","evaluate"),"Explicit root H049 admission request")
    cfg=read(r["config_ref"]);worker.validate_protocol(cfg,phase=phase)
    prep_refs=context().declared_cut_preparation_source_refs_v2(cfg)
    require(cfg.get("canonical_dispatcher_ref")==CANONICAL_DISPATCHER_REF
        and cfg.get("GPU_execution_successor_driver_ref")==ref(SOURCE),
        "Exact additive canonical dispatcher and GPU driver config pins")
    worker.bind_registration_v1(cfg,type("MetadataReader",(),{"decode_json":staticmethod(read)})())
    require(cfg["producer_ref"]==ref(WORKER_SOURCE),"Exact frozen scientific producer")
    gpu=r.get("physical_gpu")
    require(gpu is None if phase=="prepare" else gpu in (4,5),"CPU prepare or root-selected GPU4/5")
    packet=ROOT/HERE/("admitted_"+phase)
    output=("experiments/results/H049_CPU_order2_cut_memory_prepare_v1" if phase=="prepare" else
        "experiments/results/H049_GPU%d_order2_cut_memory_%s_seed101_v1"%(gpu,phase))
    auxiliary=str(packet.relative_to(ROOT)/"worker_aux")
    require(not packet.exists() and not (ROOT/output).exists() and not (ROOT/auxiliary).exists(),
        "Fresh exclusive H049 packet/output/auxiliary")
    plan=plan_request_v1(r)
    require(plan["total_output_scope_upper_bound_bytes"] is not None
        and plan["total_output_scope_upper_bound_bytes"]<=r["reserved_bytes"]<=cfg["output_cap_bytes"],
        "Genuine complete count-derived output forecast within reviewed reservation")
    if phase=="prepare":
        require(all(p["compact_cut_recipe"]["compressed_NPZ_upper_bytes"]<=cfg["cut_recipe_archive_cap_bytes"]
            for p in plan["parents"].values()),"Each compact recipe within reviewed cap")
    else:
        require(plan["maximum_parent_grid_plus_marker_cache_upper_bytes"]<=cfg["grid_cache_cap_bytes"]
            and plan["maximum_parent_marker_tensor_bytes"]<=cfg["marker_cache_cap_bytes"]
            and plan["all144_original_grid_probability_bytes"]<=cfg["RAM_prediction_cap_bytes"],
            "Whole-parent LRU minima and all144 RAM fit reviewed caps")
        require(all(p["GPU_persistent_grid_upper_bytes"]<=cfg["GPU_grid_cap_bytes"]
            and p["named_preparation_allowance_bytes"]<=cfg["grid_preparation_cap_bytes"]
            for p in plan["parents"].values()),"Cut-inclusive per-parent GPU/preparation payloads")
    sources=read(cfg["source_manifest_ref"])["source_sha256"]
    for path,digest in sources.items():require(ref(path)["sha256"]==digest,"Frozen source closure")
    require(sources[SOURCE]==ref(SOURCE)["sha256"] and sources[CONTEXT]==ref(CONTEXT)["sha256"],
        "Actual driver/context bound")
    context().validate_declared_cut_preparation_sources_v2(
        type("MetadataReader",(),{"read":staticmethod(bound_source_bytes_v2),
            "decode_json":staticmethod(read)})(),cfg,sources)
    old=read(cfg["original_H038_input_ref"])
    require(set(old["observation_entry_refs"])==set(worker.TRAIN+worker.VAL)
        and old["observation_cache_source_refs"]==cfg["observation_cache_source_refs"],
        "Original40 observed entries and sources")
    terminal={k:cfg[k] for k in ("preparation_completion_ref","preparation_actual_wait_ref")}
    for value in terminal.values():read(value)
    if phase!="prepare":
        terminal.update({k:cfg[k] for k in context().PREPARATION_METADATA_FIELDS_V2})
        for name in context().PREPARATION_METADATA_FIELDS_V2:read(terminal[name])
    context().completed_cut_preparation_v1(
        type("MetadataReader",(),{"read":staticmethod(bound_source_bytes_v2),"decode_json":staticmethod(read)})(),
        dict(source_sha256=sources,observation_entry_refs=old["observation_entry_refs"],
            observation_cache_source_refs=old["observation_cache_source_refs"],
            refs=dict(observation_cache=old["refs"]["observation_cache"]),**prep_refs,**terminal),cfg)
    train_io_lineage=None
    if phase=="evaluate":
        terminal.update({k:r[k] for k in ("train_completion_ref","train_launch_ref","train_actual_wait_ref")})
        train_io_lineage=validate_eval_train_io_v1(
            type("MetadataReader",(),{"decode_json":staticmethod(read)})(),dict(phase=phase,**terminal))
    else:
        require(all(r.get(k) is None for k in ("train_completion_ref","train_launch_ref","train_actual_wait_ref")),
            "No previous TRAIN completion in prepare/train")
    packet.mkdir()
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=24)).isoformat()
    job="H049-order2-%s-seed101-v1-%s"%(phase,"CPU" if phase=="prepare" else "GPU%d"%gpu)
    runtime=save(packet,"runtime.json",dict(status="active",device="cpu" if phase=="prepare" else "cuda",
        physical_gpu=gpu,job_id=job,output_directory=output,
        process_title=None if phase=="prepare" else "AI Auto Research Pipeline Test",
        created_at_utc=now.isoformat(),expires_at_utc=expiry,completed_overlay_policy_ref=POLICY_REF))
    refs={k:old["refs"][k] for k in ("registration","generation_config","observation_cache")}
    refs.update(config=r["config_ref"],runtime=runtime)
    if phase!="prepare":
        targets=read(old["refs"]["target_bindings"]);ids=worker.TRAIN if phase=="train" else worker.VAL
        binding=dict(targets,parents={p:targets["parents"][p] for p in ids},
            TRAIN_included=phase=="train",VAL_included=phase=="evaluate",outer_included=False,phase=phase)
        refs["target_bindings"]=save(packet,"phase_target_bindings.json",binding)
    manifest=save(packet,"consumer_manifest.json",dict(schema="H049_exact_transitive_consumer_closure_v1",
        hypothesis_id="H049",phase=phase,source_sha256=sources,refs=dict(refs),
        observation_entry_refs=old["observation_entry_refs"],
        observation_cache_source_refs=old["observation_cache_source_refs"],
        train_io_lineage=train_io_lineage,**prep_refs,**terminal))
    source=PREP_SOURCE if phase=="prepare" else WORKER_SOURCE
    m,_=load_guard_v1(consumer=False)
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    view,budget_bindings=state[2],state[3]
    mandatory=[CANONICAL_DISPATCHER_REF]+list(prep_refs.values())+[ref(source),r["config_ref"],runtime,ref(CONTEXT),ref(SOURCE),cfg["source_manifest_ref"],
        cfg["registration_ref"],cfg["count_manifest_ref"],manifest,ref(HERE+"/root_common_v1.py"),
        GUARD_REF,POLICY_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF,
        ref("src/arggnn/project_budget_v9_consumer.py"),
        ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py")]
    if phase!="prepare":mandatory += [LAUNCHER_REF,AUTH_REF,cfg["base_fit_config"],
        ref("scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py")]
    consumers=unique(mandatory+list(terminal.values())+list(budget_bindings.values())
        +([] if train_io_lineage is None else [v for k,v in train_io_lineage.items() if k!="train_checkpoint_groups"]))
    require(len(consumers)<=64,"Exact transitive closure below64 direct bindings")
    reserve=dict(key="H049/"+job,output=str(ROOT/output),auxiliary_scopes=[str(ROOT/auxiliary)],
        reserved_bytes=r["reserved_bytes"],consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",reserve,expected_before=m._pins(view),bindings=budget_bindings)
    authority.update(active=True,expires_at_utc=expiry)
    receipt=m.apply_reserve_v1(save(packet,"reserve_authority.json",authority),
        expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=job,output_directory=output,reserved_new_bytes=r["reserved_bytes"],
        reservation=receipt["value"],expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],
        completed_overlay_policy_ref=POLICY_REF,budget_components=dict(
            implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD_REF))
    refs["storage"]=save(packet,"storage.json",storage)
    spec=dict(schema="H049_marker_grid_root_input_v1",hypothesis_id="H049",arm="joint_three_arm",
        root_authorizer="root_agent",status="active",phase=phase,seed=101,
        device="cpu" if phase=="prepare" else "cuda",physical_gpu=gpu,
        fit_authorized=phase!="prepare",pyg_authorized=True,job_id=job,output_directory=output,
        worker_auxiliary_scope=auxiliary,created_at_utc=now.isoformat(),expires_at_utc=expiry,
        refs=refs,source_sha256=sources,root_context_module_ref=ref(CONTEXT),root_driver_ref=ref(SOURCE),
        canonical_dispatcher_ref=CANONICAL_DISPATCHER_REF,
        producer_ref=ref(source),main_source_ref=ref(source),worker_cli_ref=ref(source),
        consumer_manifest_ref=manifest,observation_entry_refs=old["observation_entry_refs"],
        observation_cache_source_refs=old["observation_cache_source_refs"],
        max_cpu_rss_bytes=cfg["max_cpu_rss_bytes"],max_cuda_reserved_bytes=0 if phase=="prepare" else cfg["max_cuda_reserved_bytes"],
        GPU45_authorization_ref=AUTH_REF,completed_overlay_policy_ref=POLICY_REF,
        storage_adapter_ref=GUARD_REF,storage_launcher_ref=LAUNCHER_REF,
        marker_repair_authority_ref=REPAIR_REF,io_source_manifest_ref=cfg["source_manifest_ref"],
        train_io_lineage=train_io_lineage,**prep_refs,**terminal)
    input_ref=save(packet,"input.json",spec)
    admitted=dict(schema="H049_order2_root_admitted_v1",input_ref=input_ref,driver_ref=ref(SOURCE),
        request_ref=request_ref,consumer_binding_count=len(consumers),phase=phase,physical_gpu=gpu,
        auxiliary_scope=auxiliary,scientific_execution_started=False,output_forecast=plan,
        passes=30,updates_per_arm=960,updates_total=2880,checkpoint_files=18,VAL_products=144)
    save(packet,"admitted.json",admitted)
    print(json.dumps(admitted),flush=True);return admitted

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
    require(spec["phase"] in ("train","evaluate"),"GPU successor cannot launch preparation")
    require(spec.get("canonical_dispatcher_ref")==CANONICAL_DISPATCHER_REF
        and read(spec["refs"]["config"]).get("canonical_dispatcher_ref")==CANONICAL_DISPATCHER_REF,
        "Exact actual launched canonical dispatcher")
    require(admitted["driver_ref"]==ref(SOURCE) and spec["root_driver_ref"]==ref(SOURCE),"Exact admitted root driver")
    require(not (ROOT/spec["output_directory"]).exists(),"No execution replay")
    m=load_guard(c);storage=read(spec["refs"]["storage"])
    packet=(ROOT/admitted_ref["path"]).parent
    launch_input=save(packet,"launch_input.json",spec)
    admitted=dict(admitted,input_ref=launch_input)
    # Use the exact original reserve value and initial_ledger_pins. Never rebase.
    m.budget_check_v9(storage,boundary=True,required_write_bytes=6*read(spec["refs"]["config"])["log_cap_bytes"]+65536+16384)
    auxiliary=ROOT/spec["worker_auxiliary_scope"];auxiliary.mkdir()
    env=policy_env();env["PYTHONDONTWRITEBYTECODE"]="1"
    # Original pinned launcher imports its scripts-local budget modules.
    env["PYTHONPATH"]=str(ROOT/"scripts")+os.pathsep+env.get("PYTHONPATH","")
    env["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(c.POLICY_REF,sort_keys=True)
    phase=spec["phase"];worker=CANONICAL_DISPATCHER_REF["path"]
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
    log=_RAMLog(read(spec["refs"]["config"])["log_cap_bytes"]);start=time.monotonic()
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    first=_stat(proc.pid);require(first is not None,"Actual launcher identity")
    known={proc.pid:first};groups={first["group"]}
    launch_record=dict(schema="H049_marker_original_launch_v1",launcher_pid=proc.pid,launcher_start=first["start"],
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
    m.budget_check_v9(storage,boundary=True,required_write_bytes=6*read(spec["refs"]["config"])["log_cap_bytes"]+65536+16384)
    log_ref=save(auxiliary,"launch_log.json",log.record())
    completion=ROOT/spec["output_directory"]/"completed.json"
    waited=dict(schema="H049_marker_original_direct_wait_v1",launcher_pid=proc.pid,launcher_start=first["start"],
        launch_ref=launch_ref,returncode=rc,actual_direct_launcher_wait=True,owned_group_drained=True,
        owned_process_identities=[dict(pid=p,start=x["start"],group=x["group"]) for p,x in sorted(known.items())],
        historical_owned_groups=sorted(groups),completed_ref=ref(completion) if completion.is_file() else None,
        phase=phase,physical_gpu=spec["physical_gpu"],log_ref=log_ref,peak_owned_RSS_bytes=peak,
        elapsed_seconds=time.monotonic()-start,error=error,waited_utc=datetime.now(timezone.utc).isoformat())
    wait_ref=save(launch_path.parent,"actual_wait.json",waited)
    print(json.dumps(dict(actual_wait_ref=wait_ref,returncode=rc,owned_group_drained=True,error=error)),flush=True)
    return rc or (1 if error else 0)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("describe","plan","admit","launch"),nargs="?",default="describe")
    p.add_argument("--execute",action="store_true");p.add_argument("--record");p.add_argument("--sha256")
    args=p.parse_args(argv)
    if args.action=="describe":
        print(json.dumps(dict(status="source_only_no_admission_or_execution",hypothesis_id="H049",
            phases=["train","evaluate"],canonical_module="arggnn.H049_order2_cut_memory_exec_v2",passes=30,updates_per_arm=960,
            arms=3,checkpoints=18,VAL_seals=144,genuine_resource_caps_pending=True)));return 0
    require(args.record and args.sha256,"Exact root metadata record")
    record=dict(path=args.record,sha256=args.sha256)
    if args.action=="plan":
        print(json.dumps(plan_request_v1(read(record)),indent=2));return 0
    require(args.execute,"Explicit root execution required for admit or launch")
    if args.action=="admit":admit(record);return 0
    return launch(record)

if __name__=="__main__":raise SystemExit(main())

"""Executable fresh H049 bounded CPU prepare root. Default inert; original guards retained."""
from pathlib import Path
import argparse,collections,contextlib,ctypes,hashlib,importlib.util,json,os,selectors,signal,subprocess,time,sys,traceback,uuid
from datetime import datetime,timedelta,timezone
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1.root_common_v1 import (
    HERE,require,ref,read,save,unique,load_guard_v1,count_plan_v1,
    policy_env,GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF,CORE,BASE,
    validate_eval_train_io_v1)
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import producer_v1 as worker
PACKET_HERE="research/reviews/H049_bounded_directory_metadata_repair_20261015_v1"
SOURCE=PACKET_HERE+"/root_driver.py"
BOUNDED="research/reviews/H049_bounded_directory_metadata_repair_20261015_v1"
CONTEXT=BOUNDED+"/H049_prepare_context_v1.py"
WRITER=BOUNDED+"/bounded_phase_writer_v1.py"
LAYOUT=BOUNDED+"/H049_prepare_layout_v1.py"
LIFECYCLE=BOUNDED+"/root_phase_lifecycle_v1.py"
sys.path.insert(0,str(ROOT/BOUNDED))
from H049_prepare_layout_v1 import proposed_plan_v1,validate_count_layout_v1
from root_phase_lifecycle_v1 import establish_after_original_full_guard_v1,gate_after_actual_wait_and_drain_v1
PREP_SOURCE="research/reviews/H049_order2_cut_memory_prepare_source_20261015_v2/prepare_v2.py"
CODEC_REF={'path': 'research/reviews/H049_order2_cut_memory_prepare_source_20261015_v2/cut_recipe_codec_v2.py', 'sha256': 'd9fa0e511d34420bd294b61c0a2266fd4a58a96b1ab255ab191b108c05578a45'}
WORKER_SOURCE=HERE+"/producer_v1.py"


def context():
    from research.reviews.H049_bounded_directory_metadata_repair_20261015_v1 import H049_prepare_context_v1
    return H049_prepare_context_v1


def load_guard(c):
    return load_guard_v1(consumer=False)[0]


def plan_request_v1(r):
    require(r["phase"]=="prepare","Prep-v2 plan only")
    cfg=read(r["config_ref"])
    counts=read(cfg["count_manifest_ref"])
    marker=read(cfg["preparation_completion_ref"])
    return count_plan_v1(counts,r["phase"],marker,cfg)


def admit(request_ref):
    r=read(request_ref);phase=r["phase"]
    require(r.get("schema")=="H049_order2_root_admission_request_v1"
        and r.get("status")=="active" and r.get("root_authorizer")=="root_agent"
        and phase=="prepare","Explicit root H049 bounded prepare-v5 admission request")
    cfg=read(r["config_ref"]);worker.validate_protocol(cfg,phase=phase)
    require(r["reserved_bytes"]==cfg["output_cap_bytes"]==268435456,"Exact reviewed fresh 256MiB full charge")
    require(cfg.get("preparation_execution_version")==2,"Exact sealed repaired scientific producer version2")
    require(cfg.get("cut_recipe_codec_ref")==CODEC_REF,"Exact one-bug repaired codec config pin")
    worker.bind_registration_v1(cfg,type("MetadataReader",(),{"decode_json":staticmethod(read)})())
    require(cfg["producer_ref"]==ref(WORKER_SOURCE)
        and cfg["preparation_producer_ref"]==ref(PREP_SOURCE),"Actual frozen fresh producers")
    gpu=r.get("physical_gpu")
    require(gpu is None if phase=="prepare" else gpu in (4,5),"CPU prepare or root-selected GPU4/5")
    packet=ROOT/PACKET_HERE/("admitted_"+phase)
    output=("experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v5" if phase=="prepare" else
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
    old=read(cfg["original_H038_input_ref"])
    require(set(old["observation_entry_refs"])==set(worker.TRAIN+worker.VAL)
        and old["observation_cache_source_refs"]==cfg["observation_cache_source_refs"],
        "Original40 observed entries and sources")
    terminal={k:cfg[k] for k in ("preparation_completion_ref","preparation_actual_wait_ref")}
    for value in terminal.values():read(value)
    if phase!="prepare":
        terminal.update({k:cfg[k] for k in ("cut_preparation_completion_ref","cut_preparation_actual_wait_ref")})
        for name in ("cut_preparation_completion_ref","cut_preparation_actual_wait_ref"):read(terminal[name])
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
    job="H049-order2-%s-seed101-bounded-v5-%s"%(phase,"CPU" if phase=="prepare" else "GPU%d"%gpu)
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
        train_io_lineage=train_io_lineage,**terminal))
    required_local=[ref(WRITER),ref(LAYOUT),ref(CONTEXT),ref(LIFECYCLE)]
    local_plan=proposed_plan_v1(count_plan=plan,cfg=cfg,parents=worker.TRAIN+worker.VAL,
        job_id=job,output=output,auxiliary=auxiliary,reservation_key="H049/"+job,
        reservation_bytes=r["reserved_bytes"],nonce=uuid.uuid4().hex,
        expires_unix=datetime.fromisoformat(expiry).timestamp(),source_sha256=sources,
        required_reserved_refs=required_local)
    local_plan["status"]="active"
    local_plan_ref=save(packet,"bounded_phase_plan.json",local_plan)
    validate_count_layout_v1(local_plan,plan,cfg,worker.TRAIN+worker.VAL)
    source=PREP_SOURCE if phase=="prepare" else WORKER_SOURCE
    m,_=load_guard_v1(consumer=False)
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    view,budget_bindings=state[2],state[3]
    mandatory=[local_plan_ref,*required_local,CODEC_REF,ref(source),r["config_ref"],runtime,ref(CONTEXT),ref(SOURCE),cfg["source_manifest_ref"],
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
        producer_ref=ref(source),main_source_ref=ref(source),worker_cli_ref=ref(source),
        consumer_manifest_ref=manifest,observation_entry_refs=old["observation_entry_refs"],
        observation_cache_source_refs=old["observation_cache_source_refs"],
        max_cpu_rss_bytes=cfg["max_cpu_rss_bytes"],max_cuda_reserved_bytes=0 if phase=="prepare" else cfg["max_cuda_reserved_bytes"],
        GPU45_authorization_ref=AUTH_REF,completed_overlay_policy_ref=POLICY_REF,
        storage_adapter_ref=GUARD_REF,storage_launcher_ref=LAUNCHER_REF,
        marker_repair_authority_ref=REPAIR_REF,io_source_manifest_ref=cfg["source_manifest_ref"],
        train_io_lineage=train_io_lineage,bounded_phase_plan_ref=local_plan_ref,
        bounded_phase_writer_ref=ref(WRITER),preparation_execution_version=2,**terminal)
    input_ref=save(packet,"input.json",spec)
    admitted=dict(schema="H049_order2_root_admitted_v1",input_ref=input_ref,driver_ref=ref(SOURCE),
        request_ref=request_ref,consumer_binding_count=len(consumers),phase=phase,physical_gpu=gpu,
        auxiliary_scope=auxiliary,scientific_execution_started=False,output_forecast=plan,
        bounded_phase_plan_ref=local_plan_ref,bounded_phase_peak_bytes=local_plan["peak_phase_charge_bound_bytes"],
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

class _RAMStderr:
    """Capture root stderr in RAM and tee it into the caller's combined log."""
    def __init__(self,log):self.log=log
    def write(self,text):
        self.log.add(text.encode("utf-8",errors="replace"))
        sys.stdout.write(text);sys.stdout.flush();return len(text)
    def flush(self):sys.stdout.flush()

def launch(admitted_ref,root_stderr):
    admitted=read(admitted_ref);spec=read(admitted["input_ref"]);c=context()
    require(spec.get("phase")=="prepare" and spec.get("output_directory")=="experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v5",
        "Bounded-v4 launch never reuses revoked failure scopes")
    require(admitted["driver_ref"]==ref(SOURCE) and spec["root_driver_ref"]==ref(SOURCE),"Actual fresh root driver")
    require(spec["producer_ref"]==ref(PREP_SOURCE) and spec["preparation_execution_version"]==2,"Exact sealed corrected producer")
    require(not (ROOT/spec["output_directory"]).exists(),"No execution replay")
    m=load_guard(c);storage=read(spec["refs"]["storage"]);cfg=read(spec["refs"]["config"])
    local_plan=read(spec["bounded_phase_plan_ref"])
    validate_count_layout_v1(local_plan,admitted["output_forecast"],cfg,worker.TRAIN+worker.VAL)
    packet=(ROOT/admitted_ref["path"]).parent
    # Same original unwrapped full guard, before every root-side aux mutation.
    full=m.budget_check_v9(storage,boundary=True,
        required_write_bytes=local_plan["reserved_terminal_tail_bytes"])
    phase_fd,lease_ref,private_env=establish_after_original_full_guard_v1(root=ROOT,
        plan_ref=spec["bounded_phase_plan_ref"],spec=spec,storage=storage,full_prewrite_result=full)
    env=policy_env();env.update(private_env)
    env["PYTHONPATH"]=str(ROOT/"scripts")+os.pathsep+str(ROOT/BOUNDED)+os.pathsep+env.get("PYTHONPATH","")
    env["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(POLICY_REF,sort_keys=True)
    env["OMP_NUM_THREADS"]="1";env["MKL_NUM_THREADS"]="1"
    spec=dict(spec,bounded_phase_lease_ref=lease_ref)
    launch_input=save(packet,"launch_input.json",spec)
    args=["--execute","--input",launch_input["path"],"--expected-input-sha256",launch_input["sha256"]]
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m",PREP_SOURCE[:-3].replace("/",".")]+args
    _subreaper();require(not _children(),"Original fresh subreaper/root-child condition")
    log=_RAMLog(cfg["log_cap_bytes"]);start=time.monotonic();proc=None
    try:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(phase_fd,))
    except BaseException:
        log.add(traceback.format_exc().encode("utf-8",errors="replace"))
        os.close(phase_fd)
        print(json.dumps(dict(event="bounded_prepare_spawn_failed_no_child",stderr=log.record(),
            original_storage_charge_retained=True)),flush=True)
        raise
    first=_stat(proc.pid);require(first is not None,"Actual producer child identity")
    known={proc.pid:first};groups={first["group"]}
    launch_record=dict(schema="H049_marker_original_launch_v1",launcher_pid=proc.pid,launcher_start=first["start"],
        direct_child=True,phase="prepare",physical_gpu=None,input_ref=launch_input,command=cmd,
        preparation_execution_version=2,bounded_writer_execution_version=5,bounded_phase_plan_ref=spec["bounded_phase_plan_ref"],
        phase_lease_ref=lease_ref,phase_lease_fd_inherited=True,actual_wait_pending=True,
        created_at_utc=datetime.now(timezone.utc).isoformat())
    launch_ref=save(packet,"launch.json",launch_record)
    selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ);eof=False;peak=0;error=None
    try:
        while not eof or proc.poll() is None:
            peak=max(peak,_tree(known,groups))
            require(peak<=spec["max_cpu_rss_bytes"],"Original owned process-tree RSS cap")
            require(time.monotonic()-start<86400,"Original root admission time bound")
            for key,_ in selector.select(timeout=.25):
                data=os.read(key.fileobj.fileno(),65536)
                if data:log.add(data)
                else:selector.unregister(key.fileobj);eof=True
        rc=proc.wait()
    except BaseException:
        error=traceback.format_exc();log.add(error.encode("utf-8",errors="replace"))
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
    require(drained,"No terminal receipt before original complete owned-group drain")
    # No local counter substitutes for either the lifetime exclusion or full guard.
    gate=gate_after_actual_wait_and_drain_v1(root=ROOT,phase_lease_ref=lease_ref,lifetime_fd=phase_fd,
        actual_wait=dict(actual_direct_launcher_wait=True,owned_group_drained=True,returncode=rc))
    try:
        full_terminal=m.budget_check_v9(storage,boundary=True,
            required_write_bytes=local_plan["reserved_terminal_tail_bytes"])
        if root_stderr.kept:log.add(b"\n[ROOT STDERR]\n"+b"".join(root_stderr.parts))
        auxiliary=ROOT/spec["worker_auxiliary_scope"]
        log_ref=save(auxiliary,"launch_log.json",log.record())
        completion=ROOT/spec["output_directory"]/"completed.json"
        completed_ref=ref(completion) if completion.is_file() else None
        require(rc!=0 or completed_ref is not None,"Zero exit requires actual all40 completion receipt")
        if rc==0 and completed_ref is not None:
            result=read(completed_ref)
            require(result["input_ref"]==launch_input and result["producer_ref"]==ref(PREP_SOURCE)
                and result["codec_ref"]==CODEC_REF and result["preparation_execution_version"]==2
                and result["status"]=="completed_all40_compact_cut_recipes"
                and set(result["parents"])==set(worker.TRAIN+worker.VAL)
                and all(p["lossless_roundtrip_verified"] is True for p in result["parents"].values()),"Genuine repaired40parent completion lineage")
        waited=dict(schema="H049_marker_original_direct_wait_v1",launcher_pid=proc.pid,launcher_start=first["start"],
            launch_ref=launch_ref,returncode=rc,actual_direct_launcher_wait=True,owned_group_drained=True,
            owned_process_identities=[dict(pid=p,start=x["start"],group=x["group"]) for p,x in sorted(known.items())],
            historical_owned_groups=sorted(groups),completed_ref=completed_ref,phase="prepare",physical_gpu=None,
            log_ref=log_ref,peak_owned_RSS_bytes=peak,elapsed_seconds=time.monotonic()-start,error=error,
            preparation_execution_version=2,bounded_writer_execution_version=5,phase_lease_ref=lease_ref,
            independent_phase_lease_exclusion_observed=True,full_terminal_guard_returned=True,
            original_full_reservation_retained_bytes=268435456,waited_utc=datetime.now(timezone.utc).isoformat())
        wait_ref=save(packet,"actual_wait.json",waited)
    finally:os.close(gate)
    print(json.dumps(dict(actual_wait_ref=wait_ref,returncode=rc,owned_group_drained=True,error=error,
        phase_lease_ref=lease_ref,full_terminal_guard_returned=True)),flush=True)
    return rc or (1 if error else 0)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--admit",choices=["prepare"]);group.add_argument("--launch",choices=["prepare"])
    group.add_argument("--plan",choices=["prepare"])
    p.add_argument("--execute",action="store_true");p.add_argument("--record");p.add_argument("--sha256")
    args=p.parse_args(argv)
    if not (args.admit or args.launch or args.plan):
        print(json.dumps(dict(status="source_only_no_guard_admission_or_execution",
            phase="prepare",science_version=2,bounded_writer_version=5,reserved_bytes=268435456,
            actual_repaired_producer_ref=ref(PREP_SOURCE),actual_codec_ref=CODEC_REF)));return 0
    require(args.record and args.sha256,"Exact root metadata record")
    record=dict(path=args.record,sha256=args.sha256)
    root_stderr=_RAMLog(262144)
    try:
        with contextlib.redirect_stderr(_RAMStderr(root_stderr)):
            if args.plan:print(json.dumps(plan_request_v1(read(record)),indent=2));return 0
            require(args.execute,"Explicit root execution")
            if args.admit:admit(record);return 0
            return launch(record,root_stderr)
    except BaseException:
        detail=traceback.format_exc();root_stderr.add(detail.encode("utf-8",errors="replace"))
        print(json.dumps(dict(event="root_prepare_v5_error",stderr=root_stderr.record(),
            accounting_not_closed_or_credited=True)),flush=True)
        return 1

if __name__=="__main__":raise SystemExit(main())

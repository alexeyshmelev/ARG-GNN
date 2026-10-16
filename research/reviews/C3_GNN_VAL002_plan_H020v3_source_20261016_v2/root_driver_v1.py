"""Root-only current980 executable C3 GNN stages. Default inert."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,collections,hashlib,importlib.util,json,os,selectors,signal,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_GNN_VAL002_plan_H020v3_source_20261016_v2"
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/REL),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12")]
import root_contract_v1 as c
SOURCE=c.DRIVER
STATUSES=c.STATUSES


def unique(rows):
    indexed={}
    for row in rows:
        c.require(Path(row["path"]).suffix in (".py",".sh",".json"),"Only source/JSON direct consumer bindings")
        c.require(row["path"] not in indexed or indexed[row["path"]]==row,"No conflicting direct source pins")
        indexed[row["path"]]=row
    result=list(indexed.values());c.require(len(result)<=64,"64directref cap; use source/evidence transitive manifest")
    return result

def admit(request_ref):
    req,caps,runtime=c.validate_request(request_ref);phase=req["phase"];packet,output=c.phase_paths(phase)
    c.require(not packet.exists() and not c.local(output).exists(),"Fresh exclusive GNN packet/output")
    config=c.make_config(req,caps);tmp=c.local(config["short_tmp_directory"])
    c.require(not tmp.exists(),"Fresh separately reserved TMP")
    sources={p:c.ref(p)["sha256"] for p in c.source_paths()}
    packet.mkdir();config_ref=c.save(packet,"config.json",config)
    closure_ref=c.save(packet,"source_and_input_closure.json",dict(schema="C3_GNN_current980_transitive_closure_v1",
        source_sha256=sources,root_stage_request_ref=request_ref,bindings=req["bindings"],
        scientific_train_source_ref=c.TRAIN_SOURCE_REF,hook_ref=c.HOOK_REF,native_MLM_dependency=False))
    guard=c.original_guard()
    with guard.budget.v4.mutex(ROOT):state=guard._state_locked(c.CORE,c.BASE,fresh=True)
    view,bindings=state[2],state[3]
    main=c.verify_shim() if phase in c.GPU_PHASES else c.ref(c.ENTRY)
    mandatory=[main,c.ref(c.SOURCE),c.ref(c.DRIVER),c.ref(c.ENTRY),request_ref,req["caps_ref"],
        config_ref,c.RUNTIME_REF,c.BASE_CONFIG_REF,closure_ref,c.GUARD_REF,c.POLICY_REF,c.REPAIR_REF,
        c.MEMO_REF,c.V4_REF,c.HOOK_REF,c.RENEWAL_REF,c.CERT_REF,
        c.SUPERVISOR_HELPER_REF,c.PIDFD_REF,c.OWNED_REF,c.BASE_HELPER,
        c.PREP_REF,c.WAIT_REF,c.TERMINAL_REF,c.ref(c.SOURCE_PINS),c.ref(c.FIXED_CONFIG),
        c.ref("scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"),
        c.ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        c.ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        c.ref("research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py"),
        c.ref("src/arggnn/project_budget_v9_consumer.py")]
    consumers=unique([*bindings.values(),*mandatory])
    now=datetime.now(timezone.utc);expiry=now+timedelta(seconds=caps["operational_deadline_seconds"])
    job="American-C3-GNN-VAL002-plan-H020v3-v2";aux=packet/"worker_aux"
    reserve_request=dict(key="H041/"+job,output=str(ROOT/output),auxiliary_scopes=[str(aux),str(tmp)],
        reserved_bytes=caps["reservation_cap_bytes"],consumer_bindings=consumers)
    authority=guard.operation_draft_v1("reserve",reserve_request,expected_before=guard._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry.isoformat())
    authority_ref=c.save(packet,"reserve_authority.json",authority)
    reserved=guard.apply_reserve_v1(authority_ref,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    reserve_ref=c.save(packet,"reserve_receipt.json",reserved)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,
        job_id=job,output_directory=output,reserved_new_bytes=caps["reservation_cap_bytes"],reservation=reserved["value"],
        initial_ledger_pins=reserved["initial_ledger_pins"],expires_at_utc=expiry.isoformat(),completed_overlay_policy_ref=c.POLICY_REF,
        budget_components=dict(implementation=c.ref(guard.budget.SOURCE_REL),launcher=c.ref(guard.budget.LAUNCHER_REL),consumer_adapter=c.GUARD_REF))
    refs=dict(config=config_ref,runtime=c.RUNTIME_REF,storage=c.save(packet,"storage.json",storage))
    spec=dict(schema="C3_GNN_current980_active_input_v1",status="active",root_authorizer="root_agent",
        execution_authorized=True,GPU_authorized=(phase in c.GPU_PHASES),parent_id=c.PARENT,phase=phase,job_id=job,
        output_directory=output,root_stage_request_ref=request_ref,caps_ref=req["caps_ref"],root_driver_ref=c.ref(SOURCE),
        main_source_ref=main,actual_launcher_ref=c.LAUNCHER_REF if phase in c.GPU_PHASES else None,
        source_sha256=sources,transitive_source_closure_ref=closure_ref,refs=refs,physical_gpu=req["physical_gpu"],
        execution_gpu_uuid=req["GPU_UUID"],root_auxiliary_directory=str(aux.relative_to(ROOT)),
        created_at_utc=now.isoformat(),expires_at_utc=expiry.isoformat(),max_cpu_rss_bytes=caps["max_cpu_rss_bytes"],
        max_cuda_reserved_bytes=caps["max_cuda_reserved_bytes"],completed_overlay_policy_ref=c.POLICY_REF)
    input_ref=c.save(packet,"input.json",spec)
    c.save(packet,"admitted.json",dict(input_ref=input_ref,driver_ref=c.ref(SOURCE),phase=phase,parent_id=c.PARENT,
        reserve_receipt_ref=reserve_ref,storage=storage,config_ref=config_ref,runtime_ref=c.RUNTIME_REF,caps_ref=req["caps_ref"],
        consumer_binding_count=len(consumers),execution_started=False))
    print(json.dumps(dict(input_ref=input_ref,phase=phase,consumer_binding_count=len(consumers))),flush=True)


def _stat(pid):
    try:
        values=Path("/proc/%d/stat"%pid).read_text().rsplit(")",1)[1].split()
        return dict(pid=pid,state=values[0],ppid=int(values[1]),group=int(values[2]),session=int(values[3]),
            start=int(values[19]),rss=int(values[21])*os.sysconf("SC_PAGE_SIZE"))
    except (FileNotFoundError,ProcessLookupError):return None

def _tree(known,groups):
    queue=[(pid,None) for pid in known]
    # This subreaper starts without unrelated children. A newly seen identity
    # must still be a direct child of the exact parent that enumerated it.
    try:queue.extend((int(p),os.getpid()) for p in Path("/proc/self/task/%d/children"%os.getpid()).read_text().split())
    except (FileNotFoundError,ProcessLookupError):pass
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
                except (FileNotFoundError,ProcessLookupError):pass
        except (FileNotFoundError,ProcessLookupError):pass
    if reused:raise ValueError("Historical owned PID reused outside owned tree; replacement excluded after adopted children registered")
    own=_stat(os.getpid())
    return rss+(own["rss"] if own is not None else 0)

def _absent_group(group):
    try:os.killpg(group,0);return False
    except ProcessLookupError:return True

def _harvest_owned_roster(phase,output,completion,known,groups):
    # GNN loaders use num_workers0; the outer root subreaper owns the launcher tree.
    return sorted(known),sorted(groups),[]

def launch(phase):
    c.require(os.getpgrp()==os.getpid() and os.getsid(0)==os.getpid(),
        "Root launch supervisor must begin its own dedicated session via setsid")
    packet,output=c.phase_paths(phase);admitted=c.read(c.ref(packet/"admitted.json"));spec=c.read(admitted["input_ref"],cap=33554432)
    req,caps,runtime=c.validate_request(spec["root_stage_request_ref"]);c.active(spec)
    c.require(admitted["driver_ref"]==c.ref(SOURCE) and not c.local(output).exists(),"Exact original source and fresh output")
    for path,sha in spec["source_sha256"].items():c.require(c.ref(path)["sha256"]==sha,"Admitted source changed")
    m,_launch_memo=c.consumer_guard();storage=c.read(spec["refs"]["storage"])
    reserve=c.read(admitted["reserve_receipt_ref"])
    c.require(storage["reservation"]==reserve["value"] and storage["initial_ledger_pins"]==reserve["initial_ledger_pins"],
        "Preserve actual original immutable reservation prefix; no rebasing")
    m.validate_storage_v9(spec,dict(output_cap_bytes=caps["reservation_cap_bytes"]),storage,
        source_ref=spec["main_source_ref"],config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
    aux=packet/"worker_aux";c.require(not aux.exists(),"Fresh root launcher auxiliary scope")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=caps["outer_auxiliary_cap_bytes"])
    aux.mkdir()
    tmp=c.local(c.make_config(req,caps)["short_tmp_directory"])
    c.require(not tmp.exists(),"Fresh separately reserved CPU TMP scope")
    tmp.mkdir(parents=True)
    env=os.environ.copy();env.update(TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]),
        ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(c.POLICY_REF,separators=(",",":")))
    # All cache/config writes stay in the separately reserved stage TMP scope.
    for name,key in (("cache","XDG_CACHE_HOME"),("config","XDG_CONFIG_HOME"),("data","XDG_DATA_HOME"),
        ("torch","TORCH_HOME"),("cuda","CUDA_CACHE_PATH"),("inductor","TORCHINDUCTOR_CACHE_DIR"),
        ("triton","TRITON_CACHE_DIR"),("matplotlib","MPLCONFIGDIR"),("huggingface","HF_HOME"),
        ("numba","NUMBA_CACHE_DIR")):
        folder=tmp/name;folder.mkdir();env[key]=str(folder)
    env.update(CUDA_CACHE_MAXSIZE=str(256*2**20),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",WANDB_MODE="disabled",PYTHONNOUSERSITE="1")
    module=c.GPU_MODULE if phase in c.GPU_PHASES else c.CPU_MAIN
    command=[str(ROOT/".runtime/env/bin/python"),"-B"]
    if phase in c.GPU_PHASES:
        command+=[str(ROOT/c.launcher_ref_v1()["path"]),"--gpu",str(req["physical_gpu"]),"--job",spec["job_id"],
            "--storage-record",spec["refs"]["storage"]["path"],"--expected-storage-sha256",spec["refs"]["storage"]["sha256"],
            "--expected-budget-source-sha256",c.CORE,"--expected-launcher-sha256",c.BASE,module]
    else:env["CUDA_VISIBLE_DEVICES"]="";command+=[str(ROOT/module)]
    command+=["--execute","--input",admitted["input_ref"]["path"],"--expected-input-sha256",admitted["input_ref"]["sha256"]]
    from arggnn.american_c3_native_run_v1 import _RAMLog,_subreaper,_children
    from pidfd_backend_v1 import self_probe_v1
    self_probe_v1()  # Actual root only, before Popen; source preparation never calls it.
    c.require(not _children(),"Root supervisor has no unrelated children")
    _subreaper();logs=_RAMLog(caps["outer_log_cap_bytes"]);known={};observed_groups=set();proc=None;failure=None;peak=0;interrupted=None
    selector=selectors.DefaultSelector();started=time.monotonic();stop_at=None;original_handlers={};reaped=[]
    def stop(signum,_frame):
        nonlocal interrupted
        interrupted=signum
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):original_handlers[sig]=signal.signal(sig,stop)
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            bufsize=0,start_new_session=True)
        known[proc.pid]=_stat(proc.pid);os.set_blocking(proc.stdout.fileno(),False)
        selector.register(proc.stdout,selectors.EVENT_READ)
        while proc.poll() is None or selector.get_map():
            try:peak=max(peak,_tree(known,observed_groups))
            except BaseException as error:failure=failure or str(error)
            now=time.monotonic()
            if now-started>caps["operational_deadline_seconds"]:failure=failure or "Root actual operational deadline exceeded"
            if peak>caps["max_cpu_rss_bytes"]:failure=failure or "Root actual owned RSS cap exceeded"
            if interrupted is not None:failure=failure or "Root supervisor interrupted"
            if failure:
                if stop_at is None:stop_at=now
                _signal_owned(known,signal.SIGKILL if now-stop_at>20 else signal.SIGTERM)
            for key,_ in selector.select(timeout=.5):
                data=os.read(key.fileobj.fileno(),65536)
                if data:logs.add(data)
                else:selector.unregister(key.fileobj)
            if proc.poll() is not None:
                proc.wait()
                while True:
                    try:pid,status=os.waitpid(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if pid==0:break
                    if len(reaped)<256:reaped.append(dict(pid=pid,status=status))
                if _children():failure=failure or "Direct launcher exited with adopted descendants"
    except BaseException as error:
        failure=failure or type(error).__name__+": "+str(error)[:1000]
    finally:
        if proc is not None:
            if proc.poll() is None:
                try:_tree(known,observed_groups)
                except BaseException:pass
                _signal_owned(known,signal.SIGKILL)
            rc=proc.wait()
            minimal_ref=None
            try:
                minimal_ref=c.save(packet,"actual_wait_minimal.json",dict(pid=proc.pid,returncode=rc,
                    actual_direct_launcher_wait=True,phase=phase,parent_id=c.PARENT,
                    input_ref=admitted["input_ref"],driver_ref=c.ref(SOURCE),
                    drain_state="not_yet_certified",finished_utc=datetime.now(timezone.utc).isoformat()))
                print(json.dumps(dict(event="genuine_C3_GNN_current980_direct_wait",pid=proc.pid,returncode=rc,
                    actual_wait_minimal_ref=minimal_ref)),flush=True)
            except BaseException as error:
                failure=failure or "Genuine wait capture: "+type(error).__name__+": "+str(error)[:1000]
            # A failed evidence write must never prevent actual adopted drain.
            while _children():
                try:_tree(known,observed_groups)
                except BaseException:pass
                _signal_owned(known,signal.SIGKILL)
                while True:
                    try:pid,status=os.waitpid(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if pid==0:break
                    if len(reaped)<256:reaped.append(dict(pid=pid,status=status))
                if _children():time.sleep(.1)
        else:rc=None
        selector.close()
        if proc is not None and proc.stdout is not None:proc.stdout.close()
        for sig,handler in original_handlers.items():signal.signal(sig,handler)
    c.require(proc is not None,"No root worker created; no wait fabricated")
    groups=sorted(observed_groups|{r["group"] for r in known.values() if r is not None})
    absent=all(_absent_group(g) for g in groups)
    c.require(absent and not _children(),"No scope/log access until every actually owned group is ESRCH")
    complete_path=ROOT/output/"completed.json";completed_ref=None;completion=None
    if complete_path.is_file():
        completed_ref=c.ref(complete_path);completion=c.read(completed_ref)
        c.require(completion["status"]==STATUSES[phase] and completion["parent_id"]==c.PARENT,"Exact actual native stage completion")
    pids,groups,owned_wait_refs=_harvest_owned_roster(phase,output,completion,known,groups)
    absent=all(_absent_group(g) for g in groups)
    c.require(absent and not _children(),"Every recorded native group must also be terminal before writes")
    # These writes occur only after terminal direct wait and all owned drains.
    m.budget_check_v9(storage,boundary=True,required_write_bytes=6*caps["outer_log_cap_bytes"]+65536)
    log_path=aux/"launcher_log.json"
    with log_path.open("xb") as stream:stream.write((json.dumps(logs.record(),sort_keys=True)+"\n").encode())
    if rc!=0 or completed_ref is None:failure=failure or "Native root stage failed or no completion"
    launch_ref=c.save(packet,"launch.json",dict(pid=proc.pid,process_group=proc.pid,supervisor_pid=os.getpid(),supervisor_process_group=os.getpgrp(),
        input_ref=admitted["input_ref"],driver_ref=c.ref(SOURCE),phase=phase,command=command,
        known_owned_process_groups=groups,known_owned_pids=pids,owned_terminal_wait_refs=owned_wait_refs,actual_launch=True))
    wait=dict(pid=proc.pid,returncode=rc,actual_direct_launcher_wait=True,phase=phase,parent_id=c.PARENT,
        input_ref=admitted["input_ref"],launch_ref=launch_ref,completed_ref=completed_ref,driver_ref=c.ref(SOURCE),
        all_owned_processes_and_groups_drained=absent,operational_failure=failure,
        known_owned_pids=pids,known_owned_process_groups=groups,owned_terminal_wait_refs=owned_wait_refs,supervisor_pid=os.getpid(),supervisor_process_group=os.getpgrp(),
        peak_owned_RSS_bytes=peak,adopted_waits=reaped,log_ref=c.ref(log_path),actual_wait_minimal_ref=minimal_ref,finished_utc=datetime.now(timezone.utc).isoformat())
    wait_ref=c.save(packet,"actual_wait.json",wait)
    print(json.dumps(dict(phase=phase,returncode=rc,completed_ref=completed_ref,actual_wait_ref=wait_ref)),flush=True)
    return rc if rc else (1 if failure else 0)

def close(phase,approval_ref):
    packet,_=c.phase_paths(phase);admitted=c.read(c.ref(packet/"admitted.json"));spec=c.read(admitted["input_ref"])
    launch_ref=c.ref(packet/"launch.json");wait_ref=c.ref(packet/"actual_wait.json")
    launch=c.read(launch_ref);wait=c.read(wait_ref);approval=c.read(approval_ref)
    c.require(approval.get("schema")=="C3_GNN_current980_root_close_approval_v1"
        and approval.get("root_authorizer")=="root_agent" and approval.get("active") is True
        and approval.get("driver_ref")==c.ref(SOURCE) and approval.get("input_ref")==admitted["input_ref"]
        and approval.get("actual_wait_ref")==wait_ref and approval.get("root_actually_consumed_supervisor_terminal_wait0") is True
        and approval.get("all_prior_launch_resume_write_capabilities_revoked") is True
        and approval.get("all_genuinely_owned_PIDs_and_groups_listed") is True,
        "Root must supply genuine complete supervisor/writer revocation certificate")
    c.require(wait["returncode"]==0 and wait["operational_failure"] is None and wait["all_owned_processes_and_groups_drained"],
        "Only genuinely completed wait0 stage can close")
    pids=approval["root_owned_pids"];groups=approval["root_owned_process_groups"]
    c.require(set(wait["known_owned_pids"])|{wait["supervisor_pid"]}<=set(pids)
        and set(wait["known_owned_process_groups"])|{wait["supervisor_process_group"]}<=set(groups)
        and 1<=len(set(pids))==len(pids)<=256 and 1<=len(set(groups))==len(groups)<=256,
        "Complete actual worker/launcher/supervisor/descendant roster")
    observations=[]
    for pid in pids:
        try:os.kill(pid,0)
        except ProcessLookupError:observations.append(dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat()))
        else:raise ValueError("Owned process remains alive; no completed credit")
    c.require(all(_absent_group(g) for g in groups),"Every root-owned group must remain ESRCH")
    m=c.original_guard();storage=admitted["storage"];own=storage["reservation"]
    with m.budget.v4.mutex(ROOT):state=m._state_locked(c.CORE,c.BASE,fresh=True)
    c.require(state[0].get(own["key"])==own and own["key"] not in state[1],"Exact original open reservation, no replay/rebase")
    folder=packet/"closed";c.require(not folder.exists(),"No close replay");folder.mkdir()
    completion=c.read(wait["completed_ref"])
    observation=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",
        key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=c.ref(packet/"admitted.json"),
        launch_ref=launch_ref,direct_wait_ref=wait_ref,completed_ref=wait["completed_ref"],launcher_pid=launch["pid"],
        all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
        completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
        external_NFS_writers_outside_observation=True,pids=pids,observations=observations,
        root_owned_process_groups=groups,root_approval_ref=approval_ref)
    obs_ref=c.save(folder,"root_observation.json",observation)
    request=dict(key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=c.ref(packet/"admitted.json"),
        storage_locator=["storage"],launch_ref=launch_ref,launch_pid_locator=["pid"],
        launch_identity=dict(pid=launch["pid"],phase=phase,input_ref=admitted["input_ref"],driver_ref=c.ref(SOURCE)),
        direct_wait_ref=wait_ref,wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
        wait_identity=dict(pid=wait["pid"],returncode=0,actual_direct_launcher_wait=True,phase=phase,
            all_owned_processes_and_groups_drained=True,completed_ref=wait["completed_ref"],operational_failure=None),
        completed_ref=wait["completed_ref"],completed_identity=dict(status=STATUSES[phase],parent_id=c.PARENT),
        observation_ref=obs_ref)
    authority=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    auth_ref=c.save(folder,"close_authority.json",authority)
    result=m.apply_close_completed_v1(auth_ref,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    receipt_ref=c.save(folder,"close_receipt.json",result);print(json.dumps(dict(close_receipt_ref=receipt_ref)),flush=True)

def _signal_owned(known,signum):
    from owned_lifecycle_v2 import signal_identity
    for old in list(known.values()):
        if old is not None:signal_identity(old,signum,read_stat=_stat)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
 mode.add_argument("--bind",action="store_true");mode.add_argument("--admit",choices=c.PHASES)
 mode.add_argument("--launch",choices=c.PHASES);mode.add_argument("--close",choices=c.PHASES)
 p.add_argument("--inactive-request");p.add_argument("--expected-inactive-request-sha256")
 p.add_argument("--root-approval");p.add_argument("--expected-root-approval-sha256")
 a=p.parse_args(argv)
 if a.bind:
  c.require(a.inactive_request and a.expected_inactive_request_sha256,"Actual filled metadata-only request path/SHA")
  print(json.dumps(c.bind(dict(path=a.inactive_request,sha256=a.expected_inactive_request_sha256))));return 0
 if a.admit:admit(c.ref(ROOT/REL/("request_"+a.admit+".root_active_v1.json")));return 0
 if a.launch:return launch(a.launch)
 if a.close:
  c.require(a.root_approval and a.expected_root_approval_sha256,"Actual root terminal/revocation certificate")
  close(a.close,dict(path=a.root_approval,sha256=a.expected_root_approval_sha256));return 0
 print(json.dumps(dict(status="source_only_C3_GNN_current980",phases=list(c.PHASES),
  actual_execution=False,actual_admission=False,native_MLM_dependency=False)));return 0
if __name__=="__main__":raise SystemExit(main())

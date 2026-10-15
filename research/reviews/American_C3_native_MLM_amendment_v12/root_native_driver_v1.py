"""Root-only same-charge v12 execution amendment; legacy close revoked."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,collections,hashlib,importlib.util,json,os,selectors,signal,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_amendment_v12"
sys.path[:0]=[str(ROOT/REL),str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
_ms=importlib.util.spec_from_file_location("_C3_native_specific_root_contract",ROOT/REL/"root_native_contract_v1.py")
c=importlib.util.module_from_spec(_ms);_ms.loader.exec_module(c)
from owned_lifecycle_v2 import signal_identity
from pidfd_backend_v1 import self_probe_v1
import amendment_v12 as amendment
SOURCE=c.DRIVER
STATUSES=dict(mlm="completed_C3_native_full_MLM_v1")
def unique(rows):
    indexed={}
    for row in rows:
        c.require(Path(row["path"]).suffix in (".py",".sh",".json"),"Only source/JSON direct consumer bindings")
        c.require(row["path"] not in indexed or indexed[row["path"]]==row,"No conflicting direct source pins")
        indexed[row["path"]]=row
    result=list(indexed.values());c.require(len(result)<=64,"64directref cap; use source/evidence transitive manifest")
    return result

def _stat(pid):
    try:
        values=Path("/proc/%d/stat"%pid).read_text().rsplit(")",1)[1].split()
        return dict(pid=pid,state=values[0],ppid=int(values[1]),group=int(values[2]),session=int(values[3]),
            start=int(values[19]),rss=int(values[21])*os.sysconf("SC_PAGE_SIZE"))
    except (FileNotFoundError, ProcessLookupError):return None
def _tree(known,groups):
    queue=[(pid,None) for pid in known]
    # This subreaper starts without unrelated children. A newly seen identity
    # must still be a direct child of the exact parent that enumerated it.
    try:queue.extend((int(p),os.getpid()) for p in Path("/proc/self/task/%d/children"%os.getpid()).read_text().split())
    except (FileNotFoundError, ProcessLookupError):pass
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
                except (FileNotFoundError, ProcessLookupError):pass
        except (FileNotFoundError, ProcessLookupError):pass
    if reused:raise ValueError("Historical owned PID reused outside owned tree; replacement excluded after adopted children registered")
    own=_stat(os.getpid())
    return rss+(own["rss"] if own is not None else 0)
def _signal_owned(known,signum):
    # Same pidfd helper as inner cleanup: no numeric PID signal can race reuse.
    for old in list(known.values()):
        if old is not None:signal_identity(old,signum)
def _absent_group(group):
    try:os.killpg(group,0);return False
    except ProcessLookupError:return True

def _harvest_native_waits(phase,output,completion,known,groups):
    path=ROOT/output/(phase+"_actual_wait.json");pids=set(known);all_groups=set(groups);receipts=[]
    if not path.is_file():
        c.require(completion is None,"Completed stage lacks native direct wait");return sorted(pids),sorted(all_groups),receipts
    record=c.ref(path);value=c.read(record)
    if completion is not None:c.require(completion["actual_wait_ref"]==record,"Completion-bound actual native wait")
    c.require(value.get("schema")=="american_c3_native_direct_wait_and_drain_v1"
        and value.get("actual_direct_child_wait") is True and value.get("process_group_ESRCH") is True
        and value.get("no_adopted_live_writer") is True and value.get("adopted_wait_records_may_be_truncated") is False,
        "Actual direct wait and complete adopted drain")
    pid=value["native_child_pid"];group=value["process_group"];c.require(type(pid) is int and pid>1 and group==pid,"Native new-session identity")
    pids.add(pid);all_groups.add(group)
    for row in value["adopted_descendant_waits"]:pids.add(row["pid"])
    for row in value["owned_identity_drain_proof"]["owned_identities"]:pids.add(row["pid"])
    c.require(len(pids)<=256 and len(all_groups)<=256,"Complete bounded root close roster")
    return sorted(pids),sorted(all_groups),[record]

def launch(phase):
    c.require(os.getpgrp()==os.getpid() and os.getsid(0)==os.getpid(),
        "Root launch supervisor must begin its own dedicated session via setsid")
    packet,output=c.phase_paths(phase);admitted=c.read(c.ref(packet/"admitted.json"));spec=c.read(admitted["input_ref"])
    req,caps,runtime=c.validate_request(spec["root_stage_request_ref"]);c.active(spec)
    c.require(admitted["driver_ref"]==c.ref(SOURCE) and not c.local(output).exists(),"Exact original source and fresh output")
    for path,sha in spec["source_sha256"].items():c.require(c.ref(path)["sha256"]==sha,"Admitted source changed")
    m=amendment.AllocationBridge(c.original_guard());storage=c.read(spec["refs"]["storage"])
    reserve=c.read(admitted["reserve_receipt_ref"])
    c.require(storage["reservation"]==reserve["value"] and storage["initial_ledger_pins"]==reserve["initial_ledger_pins"],
        "Preserve actual original immutable reservation prefix; no rebasing")
    amendment.validate_execution(admitted["input_ref"],spec,spec["main_source_ref"]["path"])
    amendment.allocation_boundary(m,storage,required_write_bytes=0)
    aux=packet/"worker_aux";c.require(not aux.exists(),"Fresh root launcher auxiliary scope")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=caps["outer_auxiliary_cap_bytes"])
    aux.mkdir()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]),
        ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(c.POLICY_REF,separators=(",",":")))
    module=amendment.ENTRY
    command=[str(ROOT/".runtime/env/bin/python"),"-B"]
    if phase in c.GPU_PHASES:
        command+=[str(ROOT/amendment.LAUNCHER),"--gpu",str(spec["physical_gpu"]),"--job",spec["job_id"],
            "--storage-record",spec["refs"]["storage"]["path"],"--expected-storage-sha256",spec["refs"]["storage"]["sha256"],
            "--expected-budget-source-sha256",c.CORE,"--expected-launcher-sha256",c.BASE,module]
    else:env["CUDA_VISIBLE_DEVICES"]="";command+=["-m",module]
    command+=["--execute","--input",admitted["input_ref"]["path"],"--expected-input-sha256",admitted["input_ref"]["sha256"]]
    from supervisor_v1 import _RAMLog,_subreaper,_children
    self_probe_v1()  # exact own-pidfd signal0/CLOEXEC backend probe before Popen
    c.require(not _children(),"Root supervisor has no unrelated children")
    terminal=amendment.capture("outer");direct_captured=False
    _subreaper();logs=_RAMLog(caps["outer_log_cap_bytes"]);known={};observed_groups=set();proc=None;failure=None;peak=0;interrupted=None
    from progress_v8 import RootRelay
    progress=RootRelay()
    from terminal_capture_v12 import WaitRelay
    wait_relay=WaitRelay()
    selector=selectors.DefaultSelector();started=time.monotonic();stop_at=None;original_handlers={};reaped=[]
    def capture_direct(rc):
        nonlocal direct_captured,failure
        if not direct_captured:
            try:
                print(json.dumps(dict(event="actual_direct_launcher_wait_v9",pid=proc.pid,returncode=rc,
                    actual_direct_launcher_wait=True,drain_state="not_yet_certified"),separators=(",",":")),flush=True)
            except BaseException as error:
                failure=failure or "wait stdout: "+type(error).__name__
        try:
            if not direct_captured:
                terminal.event(0,dict(schema="native_MLM_genuine_launcher_direct_wait_v9",pid=proc.pid,returncode=rc,
                    actual_direct_launcher_wait=True,drain_state="not_yet_certified",finished_unix=time.time()))
                direct_captured=True
            terminal.tail(logs)
        except BaseException as error:
            failure=failure or "terminal capture: "+type(error).__name__+": "+str(error)[:1000]
    def stop(signum,_frame):
        nonlocal interrupted
        interrupted=signum
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):original_handlers[sig]=signal.signal(sig,stop)
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            bufsize=0,start_new_session=True,pass_fds=(amendment.lifetime_fd(),))
        known[proc.pid]=_stat(proc.pid);os.set_blocking(proc.stdout.fileno(),False)
        selector.register(proc.stdout,selectors.EVENT_READ)
        while proc.poll() is None or selector.get_map():
            try:peak=max(peak,_tree(known,observed_groups))
            except BaseException as error:failure=failure or str(error)
            now=time.monotonic()
            try:amendment.check_lifetime()
            except BaseException as error:failure=failure or str(error)
            if now-started>caps["operational_deadline_seconds"]:failure=failure or "Root actual operational deadline exceeded"
            if peak>caps["max_cpu_rss_bytes"]:failure=failure or "Root actual owned RSS cap exceeded"
            if interrupted is not None:failure=failure or "Root supervisor interrupted"
            if failure:
                if stop_at is None:stop_at=now
                _signal_owned(known,signal.SIGKILL if now-stop_at>20 else signal.SIGTERM)
            for key,_ in selector.select(timeout=.5):
                data=os.read(key.fileobj.fileno(),65536)
                if data:wait_relay.feed(data);logs.add(data);progress.feed(data)
                else:selector.unregister(key.fileobj)
            if proc.poll() is not None:
                rc=proc.wait();capture_direct(rc)
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
            rc=proc.wait();capture_direct(rc)
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
    capture_direct(rc)
    tail_evidence=terminal.tail(logs)
    terminal.event(1,dict(schema="native_MLM_actual_outer_drain_v9",pid=proc.pid,returncode=rc,
        all_owned_processes_and_groups_drained=absent,no_adopted_live_writer=not _children(),
        known_owned_pids=sorted(known),known_owned_process_groups=groups,raw_stdout_tail=tail_evidence))
    terminal.close()
    complete_path=ROOT/output/"completed.json";completed_ref=None;completion=None
    if complete_path.is_file():
        completed_ref=c.ref(complete_path);completion=c.read(completed_ref)
        c.require(completion["status"]==STATUSES[phase] and completion["parent_id"]==c.PARENT,"Exact actual native stage completion")
    pids,groups,native_wait_refs=_harvest_native_waits(phase,output,completion,known,groups)
    absent=all(_absent_group(g) for g in groups)
    c.require(absent and not _children(),"Every recorded native group must also be terminal before writes")
    # These writes occur only after terminal direct wait and all owned drains.
    m.budget_check_v9(storage,boundary=True,required_write_bytes=6*caps["outer_log_cap_bytes"]+65536)
    log_path=aux/"launcher_log.json"
    with log_path.open("xb") as stream:stream.write((json.dumps(logs.record(),sort_keys=True)+"\n").encode())
    if rc!=0 or completed_ref is None:failure=failure or "Native root stage failed or no completion"
    if rc==0 and completed_ref is not None and failure is None:
        completed_ref=amendment.publish_parent_completion(m,storage,completed_ref)
    launch_ref=c.save(packet,"launch.json",dict(pid=proc.pid,process_group=proc.pid,supervisor_pid=os.getpid(),supervisor_process_group=os.getpgrp(),
        input_ref=admitted["input_ref"],driver_ref=c.ref(SOURCE),phase=phase,command=command,execution_amendment_ref=amendment.authority_ref(),
        known_owned_process_groups=groups,known_owned_pids=pids,native_actual_wait_refs=native_wait_refs,actual_launch=True))
    wait=dict(pid=proc.pid,returncode=rc,actual_direct_launcher_wait=True,phase=phase,parent_id=c.PARENT,
        input_ref=admitted["input_ref"],launch_ref=launch_ref,completed_ref=completed_ref,driver_ref=c.ref(SOURCE),execution_amendment_ref=amendment.authority_ref(),
        all_owned_processes_and_groups_drained=absent,operational_failure=failure,
        known_owned_pids=pids,known_owned_process_groups=groups,native_actual_wait_refs=native_wait_refs,supervisor_pid=os.getpid(),supervisor_process_group=os.getpgrp(),
        peak_owned_RSS_bytes=peak,adopted_waits=reaped,log_ref=c.ref(log_path),finished_utc=datetime.now(timezone.utc).isoformat())
    wait_ref=c.save(packet,"actual_wait.json",wait)
    print(json.dumps(dict(phase=phase,returncode=rc,completed_ref=completed_ref,actual_wait_ref=wait_ref)),flush=True)
    return rc if rc else (1 if failure else 0)
def close(phase,approval_ref):
    raise ValueError("Legacy old-v3 reserve/close mutations are revoked; root must use the separately authorized current accounting path")
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority");parser.add_argument("--expected-authority-sha256")
    parser.add_argument("--execute-amendment",action="store_true")
    parser.add_argument("--close",choices=("mlm",));parser.add_argument("--root-approval");parser.add_argument("--expected-root-approval-sha256")
    args=parser.parse_args(argv)
    if args.close:return close("mlm",None)
    if not args.execute_amendment and not args.close:
        print(json.dumps(dict(status="source_only_inert_v12_no_new_reservation",actual_execution=False)));return 0
    c.require(args.authority and args.expected_authority_sha256,"Exact root execution-only authority")
    record=dict(path=args.authority,sha256=args.expected_authority_sha256)
    os.environ[amendment.ENV_REF]=json.dumps(record,separators=(",",":"))
    self_probe_v1()
    fd=amendment.activate(record,c.original_guard())
    try:return launch("mlm")
    finally:os.close(fd)
if __name__=="__main__":raise SystemExit(main())

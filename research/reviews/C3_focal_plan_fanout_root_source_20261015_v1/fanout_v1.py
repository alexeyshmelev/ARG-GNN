"""Root CPU fanout for exact focal/plan parents001–013. Default does not admit or execute."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import argparse,fcntl,hashlib,json,os,select,signal,subprocess,threading,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/C3_focal_plan_fanout_root_source_20261015_v1"
SOURCE=HERE/"fanout_v1.py"
BRIDGE_REF=dict(path="research/reviews/C3_next_plan_root_handoff_20261015_v1/focal_plan_bridge_v1.py",
    sha256="b2e7fcfa1d301b01cda62a99408986e89cd75fe14509dc4188bd04d0cf84317e")
MANIFEST_REF=dict(path="research/reviews/C3_next_plan_root_handoff_20261015_v1/source_manifest.json",
    sha256="48b058e8655a99cbf09e509362d46c4f16fb9fc4b029df2fed331b7d584a7756")
PARENTS=tuple("american_c3_short_train_%03d"%i for i in range(1,14))
PACKET=ROOT/"research/reviews/C3_next_plan_root_handoff_20261015_v1"
OUTPUT=ROOT/"experiments/results/C3_short_TRAIN_focal_plans_v1"
ADMISSION_LOCK=ROOT/"research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock"
PYTHON=ROOT/".runtime/env/bin/python"
MAX_WORKERS=4;LOG_CAP=65536
STOP=threading.Event()
def utc():return datetime.now(timezone.utc).isoformat()
def need(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT),"Canonical repository path")
    return p
def ref(value):
    p=local(value);need(p.suffix in (".json",".py",".log") and p.stat().st_size<=262144,"Bounded source/metadata/log only")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
    p=local(r["path"]);need(p.suffix==".json" and ref(p)==r,"Exact source/lineage JSON")
    return json.loads(p.read_text())
def save(folder,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(raw)<=131072 and Path(name).name==name,"Compact flat evidence")
    fd=os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(folder/name)
def identity(pid):
    raw=Path("/proc")/str(pid)/"stat"
    fields=raw.read_text().rsplit(") ",1)[1].split()
    need(int(fields[2])==int(fields[3])==pid,"Actual fresh child PGID/session")
    return dict(pid=pid,start_ticks=int(fields[19]),process_group=int(fields[2]),session=int(fields[3]))
def group_absent(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return True
    return False
def child_env():
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ("src","scripts",".")))
    return env
def parent_paths(parent):
    return PACKET/("admitted_"+parent),OUTPUT/parent
def append_error(errors,error):
    STOP.set();errors.append((type(error).__name__+": "+str(error))[:2048])
def run_child(parent,phase,folder,command,admission_ref=None):
    folder.mkdir();captured=bytearray();received=0;errors=[];launch_ref=None;who=None;proc=None
    if phase=="execute" and STOP.is_set():
        record=dict(parent_id=parent,phase=phase,spawned=False,admission_ref=admission_ref,
            skipped_due_to_prior_error=True,utc=utc())
        save(folder,"not_launched.json",record);return dict(ok=False,**record)
    try:
        proc=subprocess.Popen(command,cwd=ROOT,env=child_env(),stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    except BaseException as error:
        append_error(errors,error)
        record=dict(parent_id=parent,phase=phase,spawned=False,admission_ref=admission_ref,errors=errors,utc=utc())
        save(folder,"spawn_failure.json",record);return dict(ok=False,**record)
    try:
        with proc:
            try:
                who=identity(proc.pid)
                launch_ref=save(folder,"launch.json",dict(parent_id=parent,phase=phase,**who,
                    supervisor_pid=os.getpid(),argv=command,admission_ref=admission_ref,
                    bridge_ref=BRIDGE_REF,CPU_only=True,actual_direct_wait_pending=True,utc=utc()))
            except BaseException as error:append_error(errors,error)
            pipe=proc.stdout;os.set_blocking(pipe.fileno(),False)
            def drain():
                nonlocal received
                for _ in range(4):
                    try:data=os.read(pipe.fileno(),65536)
                    except BlockingIOError:return
                    if not data:return
                    received+=len(data);captured.extend(data[:max(0,LOG_CAP-len(captured))])
            try:
                while True:
                    drain()
                    if proc.poll() is not None:
                        rc=proc.wait()
                        # Genuine direct wait is persisted before any log, drain proof or completion inspection.
                        minimal_ref=save(folder,"actual_wait_minimal.json",dict(parent_id=parent,phase=phase,
                            pid=proc.pid,returncode=rc,actual_direct_subprocess_wait=True,
                            actual_direct_worker_wait=phase=="execute",actual_direct_admission_wait=phase=="admit",
                            launch_ref=launch_ref,utc=utc()))
                        if rc:STOP.set()
                        break
                    select.select([pipe],[],[],.05)
                drain()
            finally:
                pipe.close()
                # Operational stream/write errors still leave the child genuinely waited.
                if proc.returncode is None:
                    rc=proc.wait()
                    STOP.set()
                    if not (folder/"actual_wait_minimal.json").exists():
                        save(folder,"actual_wait_minimal.json",dict(parent_id=parent,phase=phase,pid=proc.pid,
                            returncode=rc,actual_direct_subprocess_wait=True,actual_direct_worker_wait=phase=="execute",
                            actual_direct_admission_wait=phase=="admit",launch_ref=launch_ref,
                            operational_stream_failure=True,utc=utc()))
            absent=False;group_checked_utc=None
            try:
                # Only the exact owned group is probed. Never signal or inspect unrelated processes.
                deadline=time.monotonic()+30
                while not (absent:=group_absent(proc.pid)) and time.monotonic()<deadline:time.sleep(.05)
                group_checked_utc=utc()
                need(absent,"Owned process group remains present after direct child wait")
            except BaseException as error:append_error(errors,error)
            log_path=folder/"stdout.log"
            with log_path.open("xb") as log:log.write(captured);log.flush();os.fsync(log.fileno())
            completed_ref=None
            try:
                need(rc==0 and not errors,"Actual child wait/metadata/group failure")
                packet,out=parent_paths(parent)
                if phase=="admit":
                    admission_ref=ref(packet/"admission.json");spec=read(admission_ref)
                    announced=[]
                    for line in bytes(captured).decode(errors="replace").splitlines():
                        try:value=json.loads(line)
                        except json.JSONDecodeError:continue
                        if isinstance(value,dict) and "admission_ref" in value:announced.append(value)
                    need(len(announced)==1 and announced[0]["admission_ref"]==admission_ref
                        and announced[0]["reserved_bytes"]==33554432
                        and announced[0]["scientific_execution_started"] is False,"Actual successful admission stdout/path/hash")
                    need(spec["parent_id"]==parent and spec["source_ref"]==BRIDGE_REF
                        and spec["GPU_authorized"] is False and spec["execution_authorized"] is True
                        and spec["output_directory"]==str(out.relative_to(ROOT)),"Actual unchanged parent CPU admission")
                else:
                    completed_ref=ref(out/"completed.json");done=read(completed_ref)
                    need(done["status"]=="completed_C3_short_TRAIN_native_focal_and_PyG_plan_v1"
                        and done["parent_id"]==parent and done["source_ref"]==BRIDGE_REF
                        and done["focal_records"]==10000 and done["query_ancestry_truth_opened"] is False
                        and done["all_trees_and_samples_retained"] is True
                        and done["native_MLM_or_training_dependency"] is False,"Actual complete unchanged focal/plan metadata")
            except BaseException as error:append_error(errors,error)
            minimal_ref=ref(folder/"actual_wait_minimal.json")
            record=dict(parent_id=parent,phase=phase,pid=proc.pid,identity=who,returncode=rc,
                actual_direct_subprocess_wait=True,actual_direct_worker_wait=phase=="execute",
                actual_direct_admission_wait=phase=="admit",minimal_wait_ref=minimal_ref,launch_ref=launch_ref,
                owned_process_group_ESRCH=absent,group_checked_utc=group_checked_utc,admission_ref=admission_ref,
                completed_ref=completed_ref,stdout_ref=ref(log_path),stdout_bytes_received=received,
                stdout_bytes_retained=len(captured),stdout_truncated=received>len(captured),errors=errors,utc=utc())
            rr=save(folder,"actual_wait.json",record)
            return dict(ok=not errors and rc==0 and absent,wait_ref=rr,**record)
    finally:
        # Even an operational exception is genuinely waited by Popen's context manager.
        if not (folder/"actual_wait_minimal.json").exists():
            STOP.set();rc=proc.wait()
            save(folder,"actual_wait_minimal.json",dict(parent_id=parent,phase=phase,pid=proc.pid,
                returncode=rc,actual_direct_subprocess_wait=True,actual_direct_worker_wait=phase=="execute",
                actual_direct_admission_wait=phase=="admit",launch_ref=launch_ref,
                operational_monitor_failure=True,utc=utc()))
def monitored(*args):
    try:return run_child(*args)
    except BaseException:
        STOP.set();raise
def execute(args):
    need(Path.cwd().resolve()==ROOT,"Root repository invocation")
    need(ref(BRIDGE_REF["path"])==BRIDGE_REF and ref(MANIFEST_REF["path"])==MANIFEST_REF,"Unchanged bridge/manifest")
    manifest=read(MANIFEST_REF)
    need(all(ref(r["path"])==r for r in manifest["source_refs"]),"Existing exact source closure")
    need(OUTPUT.is_dir() and local(OUTPUT)==OUTPUT,"Existing output collection parent from successful000")
    for parent in PARENTS:
        packet,out=parent_paths(parent);need(not packet.exists() and not out.exists(),"No replay or partial parent: "+parent)
    need(args.run_name and len(args.run_name)<=64 and all(c.isalnum() or c in "_-" for c in args.run_name),
        "Flat fresh run name")
    run=HERE/args.run_name;need(not run.exists(),"Fresh coordinator packet only");run.mkdir()
    errors=[];results={};admitted_without_launch={};live={}
    def stopped(signum,frame):
        STOP.set();errors.append("Root requested stop signal "+str(signum)+"; drain owned running children")
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,stopped)
    save(run,"started.json",dict(source_ref=ref(SOURCE),bridge_ref=BRIDGE_REF,
        parents=list(PARENTS),max_concurrent_preparations=MAX_WORKERS,admissions_sequential=True,
        per_subprocess_log_cap_bytes=LOG_CAP,supervisor_pid=os.getpid(),CPU_only=True,utc=utc()))
    def collect_finished(block=False):
        if block and live:wait(list(live),return_when=FIRST_COMPLETED)
        for future in list(live):
            if future.done():
                parent=live.pop(future)
                try:
                    results[parent]=future.result()
                    if results[parent].get("spawned") is False and results[parent].get("admission_ref"):
                        admitted_without_launch[parent]=results[parent]["admission_ref"]
                except BaseException as error:append_error(errors,error)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        try:
            for parent in PARENTS:
                collect_finished()
                while len(live)>=MAX_WORKERS and not STOP.is_set():collect_finished(True)
                if STOP.is_set():break
                packet,out=parent_paths(parent)
                need(not packet.exists() and not out.exists(),"Parent changed before admission; preserve it")
                folder=run/parent;folder.mkdir()
                # Existing shared wave lock serializes admissions, while monitor threads retain waits promptly.
                with ADMISSION_LOCK.open("r+") as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    if STOP.is_set():break
                    admitted=monitored(parent,"admit",folder/"admit",
                        [str(PYTHON),"-B",str(ROOT/BRIDGE_REF["path"]),"--admit",parent])
                save(folder,"admission_result.json",dict(ok=admitted["ok"],wait_ref=admitted.get("wait_ref"),
                    admission_ref=admitted.get("admission_ref"),utc=utc()))
                if not admitted["ok"]:break
                admission_ref=admitted["admission_ref"]
                if STOP.is_set():
                    admitted_without_launch[parent]=admission_ref;break
                command=[str(PYTHON),"-B",str(ROOT/BRIDGE_REF["path"]),"--execute",
                    "--admission",admission_ref["path"],"--expected-admission-sha256",admission_ref["sha256"]]
                admitted_without_launch[parent]=admission_ref
                future=pool.submit(monitored,parent,"execute",folder/"execute",command,admission_ref)
                live[future]=parent
                admitted_without_launch.pop(parent)
        except BaseException as error:append_error(errors,error)
        while live:collect_finished(True)
    success=not STOP.is_set() and not errors and len(results)==len(PARENTS) and all(r["ok"] for r in results.values())
    final=dict(status="completed_all13_exact_focal_plan_preparations" if success else "failed_or_stopped_partial_preserved",
        source_ref=ref(SOURCE),bridge_ref=BRIDGE_REF,all_direct_children_waited=True,
        max_concurrent_preparations=MAX_WORKERS,admissions_sequential=True,
        parent_results={p:dict(ok=r["ok"],wait_ref=r.get("wait_ref"),completed_ref=r.get("completed_ref"),admission_ref=r.get("admission_ref")) for p,r in results.items()},
        admitted_without_launch=admitted_without_launch,errors=errors,
        parents_without_completed_preparation=[p for p in PARENTS if p not in results or not results[p]["ok"]],
        files_deleted=0,automatic_replay=False,CPU_only=True,utc=utc())
    rr=save(run,"fanout_result.json",final)
    print(json.dumps(dict(status=final["status"],result_ref=rr,completed_parent_count=sum(r["ok"] for r in results.values()))),flush=True)
    return 0 if success else 1
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--run-name",default="run_v1");args=p.parse_args()
    if args.execute:return execute(args)
    print(json.dumps(dict(status="source_only_no_guard_admission_execution_or_process_probe",
        parents=list(PARENTS),max_concurrent_preparations=MAX_WORKERS,admissions_sequential=True,
        bridge_ref=BRIDGE_REF,per_subprocess_log_cap_bytes=LOG_CAP)));return 0
if __name__=="__main__":raise SystemExit(main())

"""Plaintext sequential CPU archive supervisor. ROOT alone approves execution."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v1 as recovery

ROOT = Path(__file__).resolve().parents[4]
SOURCE = "research/repairs/benchmark_recovery_20261020/storage/archive_supervisor_v1.py"
MODULE = "research.repairs.benchmark_recovery_20261020.storage.archive_supervisor_v1"
LOCK = ".runtime/locks/benchmark_recovery_capacity_20261020.lock"
CONTROLLER_LOCK = ".runtime/locks/closed_storage_recovery_controller_20261020_v1.lock"
_STOP_REQUESTED = False

def request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True

def load(value):
    with Path(value).open() as handle:
        return json.load(handle)

def save(value, data):
    value = Path(value)
    value.parent.mkdir(parents=True, exist_ok=True)
    recovery.codec._save(value, data)

def read_config(config_path, approved=False):
    config_path = recovery.path(ROOT, config_path)
    config = load(config_path)
    config_ref = recovery.reference(ROOT, config_path)
    recovery.need(config["schema"] == "sequential_closed_storage_recovery_20261020_v1"
                  and config["storage_ceiling_bytes"] == 980_000_000_000,
                  "Exact sequential recovery and unchanged 980GB ceiling required")
    recovery.need(datetime.now(timezone.utc) < datetime.fromisoformat(config["expires_at_utc"]),
                  "Fresh finite recovery authority required")
    recovery.need(tuple(row["path"] for row in config["source_refs"]) == recovery.NEW_SOURCES+(SOURCE,),
                  "Complete exact reviewed source closure required")
    for reference in config["source_refs"]:
        item = recovery.path(ROOT, reference["path"])
        recovery.need(item.stat().st_size <= 32*2**20 and recovery.reference(ROOT,item) == reference,
                      "Reviewed source changed")
    policy = load(ROOT/"configs/benchmarks/execution_policy.json")
    if approved:
        recovery.need(policy.get("benchmark_execution_approved") is True
                      and policy.get("physical_gpus") == [0]
                      and config_ref in policy.get("approved_closed_storage_recovery_refs",[])
                      and policy.get("resource_continuation_drain_ref") == config["source_plan_ref"],
                      "Explicit current ROOT recovery approval required")
        inventory = recovery.read(ROOT,config["inventory_ref"])
        revoked = policy.get("closed_storage_recovery_writer_revocations",[])
        for target in inventory["targets"]:
            expected = dict(scope=target["scope"],original_job_ref=target["producer"]["job_ref"],
                            source_plan_ref=target["producer"]["source_plan_ref"])
            recovery.need(expected in revoked, "Exact old scope writer revocation required")
    return config,config_ref,policy

def base_capacity(config):
    """Original named-scope accounting without importing a compiled scheduler."""
    plan = recovery.read(ROOT,config["source_plan_ref"])
    state = load(recovery.path(ROOT,plan["records_directory"])/"state.json")
    recovery.need(state["plan_ref"] == config["source_plan_ref"],"Original workflow state changed plan")
    baseline = recovery.read(ROOT,plan["storage_baseline_ref"])
    total = baseline["measured_project_bytes"] + plan["metadata_margin_bytes"]
    rss = 0
    charges = {}
    live_caps = {}
    for external in plan["external_storage"]:
        inactive = bool(external.get("completion_evidence"))
        for proof in external.get("completion_evidence",[]):
            queue = load(recovery.path(ROOT,proof["queue_path"]))
            entry = next(row for row in queue["jobs"] if row["admission"]["job_id"] == proof["job_id"])
            wait = load(recovery.path(ROOT,proof["wait_path"]))
            inactive = (inactive and entry["status"] == "succeeded" and entry.get("actual_returncode") == 0
                        and wait.get("actual_direct_wait") is True and wait.get("actual_direct_returncode") == 0
                        and wait.get("owned_process_group_absent") is True)
        if inactive:
            # These six completed external scopes have no symbolic links.
            total += sum(recovery.scope_bytes(recovery.path(ROOT,scope)) for scope in external["scopes"])
        else:
            total += external["cap_bytes"]
    tasks = {task["id"]:task for task in plan["tasks"]}
    for identifier,entry in state["tasks"].items():
        if entry.get("external") or "reservation_bytes" not in entry:
            continue
        charge = entry.get("retained_bytes",entry["reservation_bytes"])
        total += charge
        task = tasks[identifier]
        scope = str(recovery.path(ROOT,task["output_directory"]))
        charges[(scope,identifier)] = charges.get((scope,identifier),0)+charge
        if entry["status"] == "running" or entry["status"] == "failed" and entry.get("owned_process_group_absent") is not True:
            rss += task["max_rss_bytes"]
            live_caps[(scope,identifier)] = live_caps.get((scope,identifier),0)+task["max_rss_bytes"]
    for binding in plan["workflow_external_predecessors"]:
        old = load(recovery.path(ROOT,binding["source_state_path"]))["tasks"][binding["source_task_id"]]
        drained = (old["status"] in ("succeeded","failed")
                   and type(old.get("actual_controller_wait")) is int
                   and old.get("owned_process_group_absent") is True)
        if drained:
            wait = load(recovery.path(ROOT,binding["output_directory"])/"actual_wait.json")
            drained = (wait.get("job_ref") == binding["original_job_ref"]
                       and wait.get("actual_direct_wait") is True
                       and type(wait.get("actual_direct_returncode")) is int
                       and wait.get("owned_process_group_absent") is True)
        charge = old["retained_bytes"] if drained else binding["max_output_bytes"]
        total += charge
        scope = str(recovery.path(ROOT,binding["output_directory"]))
        original_key=(scope,binding["source_task_id"])
        charges[original_key] = charges.get(original_key,0)+charge
        if not drained:
            rss += binding["max_rss_bytes"]
            live_caps[original_key] = live_caps.get(original_key,0)+binding["max_rss_bytes"]
    for attempt in config.get("closed_native_attempts",[]):
        old_request = recovery.read(ROOT,attempt["request_ref"])
        old_job = recovery.read(ROOT,attempt["job_ref"])
        old_wait = recovery.read(ROOT,attempt["wait_ref"])
        old_task = old_request["task"]
        old_scope = recovery.path(ROOT,attempt["scope"])
        recovery.need(old_job["task"] == old_task
                      and recovery.path(ROOT,old_task["output_directory"]) == old_scope
                      and old_job.get("shared_native_request_ref") == attempt["request_ref"]
                      and old_wait.get("shared_native_request_ref") == attempt["request_ref"]
                      and old_wait.get("job_ref") == attempt["job_ref"]
                      and old_wait.get("actual_direct_wait") is True
                      and old_wait.get("actual_direct_returncode") == 1
                      and old_wait.get("owned_process_group_absent") is True,
                      "Exact genuinely failed native attempt required")
        retained = recovery.scope_bytes(old_scope)
        recovery.need(retained == attempt["retained_bytes"],"Preserved failed native scope changed")
        old_key = (str(old_scope),old_task["id"])
        total += max(0,retained-charges.get(old_key,0))
    native = recovery.read(ROOT,config["native_request_ref"])
    native_task = native["task"]
    native_scope = str(recovery.path(ROOT,native_task["output_directory"]))
    native_key=(native_scope,native_task["id"])
    # A single exact namespace is charged at least the outer native cap. An old
    # reserved task name alone is never sufficient for this deduplication.
    total += max(0,native_task["max_output_bytes"]-charges.get(native_key,0))
    rss += max(0,native_task["max_rss_bytes"]-live_caps.get(native_key,0))
    recovery.need(plan.get("external_native_rss_bytes")==0 and plan.get("prior_unknown_rss_bytes")==0,
                  "Use only the approved current-host RAM accounting source")
    for external in plan.get("external_resources",[]):
        queue=load(recovery.path(ROOT,external["queue_path"]))
        old=next(row for row in queue["jobs"] if row["admission"]["job_id"]==external["job_id"])
        wait=load(recovery.path(ROOT,external["wait_path"]))
        drained=(old["status"] in ("failed","succeeded") and wait.get("actual_direct_wait") is True
                 and wait.get("owned_process_group_absent") is True)
        if not drained:
            rss+=external["max_rss_bytes"]
    total += config["concurrent_ROOT_preparation"]["max_output_bytes"]
    rss += config["concurrent_ROOT_preparation"]["max_rss_bytes"]
    return total,rss

def capacity(config, state, extra_task=None):
    total,rss = base_capacity(config)
    for task in config["recovery_plan"]["tasks"]:
        entry = state["tasks"][task["id"]]
        if "reservation_bytes" in entry:
            total += entry.get("retained_bytes",entry["reservation_bytes"])
        if entry["status"] == "running" or entry["status"] == "failed" and entry.get("owned_process_group_absent") is not True:
            rss += task["max_rss_bytes"]
    total -= recovery.completed_storage_reduction(config["recovery_plan"],state)
    if extra_task is not None:
        total += extra_task["max_output_bytes"]
        rss += extra_task["max_rss_bytes"]
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    return dict(projected_bytes=total,reserved_rss_bytes=rss,kernel_MemAvailable_bytes=available,
                storage_allowed=total<config["storage_ceiling_bytes"],
                ram_allowed=(rss<=config["max_parallel_rss_bytes"]
                             and available-rss>=config["host_ram_margin_bytes"]))

def group_absent(pgid):
    try:
        os.killpg(pgid,0)
        return False
    except ProcessLookupError:
        return True

def stop_owned(process):
    """Signal only this unreaped direct Popen child, then consume its real wait."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            return process.wait()
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    return process.wait()

@contextmanager
def locked(relative):
    lock_path = recovery.path(ROOT,relative)
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    descriptor = os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,"r+") as lock:
        mode = fcntl.LOCK_EX if relative == LOCK else fcntl.LOCK_EX|fcntl.LOCK_NB
        fcntl.flock(lock,mode)
        yield

def monitor_owned(process,output,task,entry,check_authority):
    """Every failure after Popen consumes the directly owned child's real wait.

    The archive worker uses Python streaming gzip in this one process and does
    not spawn scientific subprocesses. No pidfd API is required by this runtime.
    """
    error = None
    try:
        entry["pid"] = process.pid
        save(output/"owned_process_identity.json",dict(pid=process.pid,pgid=process.pid,
             session=process.pid,proc_stat=Path(f"/proc/{process.pid}/stat").read_text(),
             signal_method="direct_unreaped_Popen_child"))
        while process.poll() is None:
            if _STOP_REQUESTED:
                raise InterruptedError("Owned archive shutdown requested")
            if time.time()-entry["started_at"]>task["max_seconds"]:
                raise TimeoutError("Owned archive wall cap exceeded")
            status = Path(f"/proc/{process.pid}/status").read_text()
            rss = next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith("VmRSS:")),0)
            if rss>task["max_rss_bytes"]:
                raise RuntimeError("Owned archive RSS cap exceeded")
            check_authority()
            if recovery.scope_bytes(output)>task["max_output_bytes"]:
                raise RuntimeError("Owned archive output cap exceeded")
            time.sleep(2)
        returncode = process.wait()
    except BaseException as exc:
        error = repr(exc)
        returncode = stop_owned(process)
    return returncode,error

def child(config_path, task_key):
    config,config_ref,_ = read_config(config_path,approved=True)
    task = next(task for task in config["recovery_plan"]["tasks"] if task["closed_storage_recovery_target"] == task_key)
    resource.setrlimit(resource.RLIMIT_AS,(task["max_rss_bytes"],task["max_rss_bytes"]))
    output = recovery.path(ROOT,task["output_directory"])
    job = recovery.read(ROOT,recovery.reference(ROOT,output/"job.json"))
    expected_spec=dict(schema=recovery.SCHEMA,inventory_ref=config["inventory_ref"],
                       target_key=task_key,output_directory=str(output/"native"))
    recovery.need(job["plan_ref"]==config_ref and job["task"]==task
                  and job["call"]["module"]==recovery.MODULE and job["call"]["function"]=="run"
                  and job["call"]["spec"]==expected_spec,"Exact approved child job required")
    last_check = 0.0
    def check(label):
        nonlocal last_check
        now = time.monotonic()
        if now-last_check >= 5:
            read_config(config_path,approved=True)
            recovery.need(recovery.scope_bytes(output)<=task["max_output_bytes"],"Archive output cap exceeded")
            last_check = now
    result = recovery.run(job["call"]["spec"],resource_check=check,
                          execution_check=lambda:read_config(config_path,approved=True))
    save(output/"child_value.json",dict(value=result))

def supervise(config_path,max_tasks,continuation=False):
    config,config_ref,policy = read_config(config_path,approved=True)
    records = recovery.path(ROOT,config["records_directory"])
    records.mkdir(parents=True,exist_ok=True)
    # This lock excludes another archive controller only. The shared capacity
    # lock is held solely for reservation and settlement transactions below.
    with locked(CONTROLLER_LOCK):
        state_path = records/"state.json"
        if state_path.exists():
            recovery.need(continuation,"Existing delivery requires explicit --continue; never resubmit blindly")
            state = load(state_path)
            recovery.need(state["plan_ref"]==config_ref and all(row["status"] in ("pending","succeeded") for row in state["tasks"].values()),
                          "Failed, live or unknown attempts require separate ROOT reconciliation")
        else:
            state = dict(plan_ref=config_ref,tasks={task["id"]:dict(status="pending",external=False)
                         for task in config["recovery_plan"]["tasks"]},run_status="prepared")
        count = 0
        for task in config["recovery_plan"]["tasks"]:
            entry = state["tasks"][task["id"]]
            if entry["status"]=="succeeded":
                continue
            if count>=max_tasks:
                break
            recovery.need(not _STOP_REQUESTED,"Sequential supervisor stop requested before admission")
            with locked(LOCK):
                config,config_ref,policy = read_config(config_path,approved=True)
                snapshot = capacity(config,state,extra_task=task)
                recovery.need(snapshot["storage_allowed"] and snapshot["ram_allowed"],"Fresh full-cap storage/RAM admission denied")
                output = recovery.path(ROOT,task["output_directory"])
                recovery.need(not output.exists(),"Never reuse an archive attempt output")
                call = recovery.factory(**task["factory_arguments"],output_directory=str(output/"native"))
                output.mkdir(parents=True,exist_ok=False)
                job = dict(schema="static_closed_archive_job_v1",plan_ref=config_ref,task=task,call=call,
                           sources=config["source_refs"],admission=snapshot)
                save(output/"job.json",job)
                job_ref = recovery.reference(ROOT,output/"job.json")
                save(output/"execution_policy_snapshot.json",policy)
                entry.update(status="running",reservation_bytes=task["max_output_bytes"],
                             job_ref=job_ref,started_at=time.time())
                state["run_status"]="running"
                save(state_path,state)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable,"-s","-B","-m",MODULE,"--config",str(recovery.path(ROOT,config_path)),
                       "--child",task["closed_storage_recovery_target"]]
            with (output/"supervisor.log").open("xb") as log:
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                returncode,error = monitor_owned(process,output,task,entry,
                                                lambda:read_config(config_path,approved=True))
            with locked(LOCK):
                absent=group_absent(process.pid)
                save(output/"actual_wait.json",dict(job_ref=job_ref,actual_direct_wait=True,
                     actual_direct_returncode=returncode,owned_process_group_absent=absent,supervisor_error=error))
                entry.update(actual_controller_wait=returncode,owned_process_group_absent=absent,
                             finished_at=time.time())
                if returncode==0 and absent and error is None:
                    value=load(output/"child_value.json")["value"]
                    save(output/"result.json",dict(job_ref=job_ref,value=value))
                    entry.update(status="succeeded",result_ref=recovery.reference(ROOT,output/"result.json"))
                else:
                    entry.update(status="failed",error=error)
                entry["retained_bytes"]=recovery.scope_bytes(output) if absent else task["max_output_bytes"]
                state["updated_utc"]=datetime.now(timezone.utc).isoformat()
                save(state_path,state)
                if entry["status"]!="succeeded":
                    raise RuntimeError("Archive stopped with a genuine failed or unknown owned attempt")
                recovery.completed_storage_reduction(config["recovery_plan"],state)
            count+=1
        with locked(LOCK):
            state["run_status"]="complete" if all(row["status"]=="succeeded" for row in state["tasks"].values()) else "batch_complete_pending"
            state["capacity_after_batch"]=capacity(config,state)
            save(state_path,state)
            save(records/"successor_handoff.json",dict(schema="closed_storage_recovery_handoff_v1",config_ref=config_ref,
                 state_ref=recovery.reference(ROOT,state_path),completed=[identifier for identifier,row in state["tasks"].items() if row["status"]=="succeeded"],
                 source_plan_ref=config["source_plan_ref"],no_history_relabelled=True))
        print(json.dumps(dict(run_status=state["run_status"],completed_this_run=count,
                             capacity=state["capacity_after_batch"]),sort_keys=True),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--continue",dest="continuation",action="store_true")
    parser.add_argument("--max-tasks",type=int,default=1)
    parser.add_argument("--child")
    args=parser.parse_args()
    if args.child:
        child(args.config,args.child)
    elif args.execute:
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        recovery.need(1<=args.max_tasks<=12,"Explicit bounded sequential batch required")
        supervise(args.config,args.max_tasks,args.continuation)
    else:
        config,_,_=read_config(args.config,approved=False)
        state=dict(tasks={task["id"]:dict(status="pending") for task in config["recovery_plan"]["tasks"]})
        print(json.dumps(dict(tasks=len(state["tasks"]),first_admission=capacity(config,state,config["recovery_plan"]["tasks"][0])),sort_keys=True))

if __name__=="__main__":
    main()

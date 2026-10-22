"""Dependency-driven benchmark execution with one GPU and bounded CPU stages.

A frozen ROOT-approved plan is the authority. Source/data references are resolved
only after genuine predecessor completion. Owned workers supervise the scientific
process group and record real waits. Failed/unknown stages never retry themselves.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from . import contracts_recovery_v1 as c
from arggnn.benchmarking.execution import TITLE
from arggnn.benchmarking.workflow_v4 import scope_bytes
from research.repairs.benchmark_multigpu_20261022.runtime import queue_support_v1 as support
from . import queue_gpu_v1 as shared_gpu
from . import queue_execute_v1 as runtime
from . import overlay_predecessor_v1
overlay_predecessor_v1.install()

MODULE = "research.repairs.benchmark_repair_20261022_afternoon.runtime_v2.readable_workflow_v1"
SOURCE = "research/repairs/benchmark_repair_20261022_afternoon/runtime_v2/readable_workflow_v1.py"
SCHEMA = "arggnn_executable_workflow_v1"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def authenticated_adopted_completed(plan):
    """Bind each historical exemption to its immutable succeeded source state."""
    mapping=plan.get('adopted_completed_task_fingerprints',{})
    need(isinstance(mapping,dict),'Explicit adopted completion fingerprint mapping required')
    if not mapping:return {}
    snapshot=c.read_ref(plan['adopted_state_snapshot_ref'])
    original=c.read_ref(snapshot['plan_ref'])
    current={task['id']:task for task in plan['tasks']}
    previous={task['id']:task for task in original['tasks']}
    authenticated={}
    for identifier,fingerprint in mapping.items():
        need(identifier in current and identifier in previous
             and c.fingerprint(current[identifier])==fingerprint==c.fingerprint(previous[identifier]),
             'Historical exemption requires exact original/current task fingerprint')
        entry=snapshot['tasks'][identifier]
        need(entry.get('status')=='succeeded' and isinstance(entry.get('result_ref'),dict),
             'Historical exemption requires a genuinely recorded succeeded snapshot entry')
        authenticated[identifier]=entry
    return authenticated


def validate_adopted_state(plan,state):
    for identifier,original in authenticated_adopted_completed(plan).items():
        entry=state['tasks'][identifier]
        need(entry.get('status')=='succeeded' and entry.get('result_ref')==original['result_ref'],
             'An adopted completed task cannot become pending or change its result')


def initial_state(plan,reference):
    authenticated_adopted_completed(plan)
    state=support.initial_state(plan,reference)
    validate_adopted_state(plan,state)
    return state


def save(path, value, *, exclusive=False):
    path = c.contained_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if exclusive:
        with path.open("x") as f:
            f.write(encoded); f.flush(); os.fsync(f.fileno())
    else:
        temp = path.with_name(path.name+".tmp."+str(os.getpid()))
        with temp.open("x") as f:
            f.write(encoded); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    return c.file_ref(path)


def source_check(refs):
    for ref in refs:
        p = c.contained_path(ref["path"])
        need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 32*2**20
             and c.file_ref(p) == ref, "Workflow source changed: "+ref["path"])


def current_policy():
    canonical=c.check_execution_release(c.file_ref(c.POLICY_PATH))
    approved=canonical.get("approved_recovery_continuation_workflow_ref")
    need(isinstance(approved,dict) and set(approved)=={"path","sha256"},"Exact readable workflow ROOT approval required")
    policy=dict(canonical,approved_workflow_ref=approved)
    policy.pop("workflow_drain_ref",None)
    if "recovery_continuation_workflow_drain_ref" in canonical:
        policy["workflow_drain_ref"]=canonical["recovery_continuation_workflow_drain_ref"]
    return policy


def load_plan(ref, *, approved=False):
    plan = c.read_ref(ref)
    need(plan.get("schema") == SCHEMA and plan.get("physical_gpu") == 5 and plan.get("physical_gpus") == [2,3,5],
         "Frozen explicit GPU2/3/5 parallel workflow required")
    need(type(plan.get("cpu_parallelism")) is int and 1 <= plan["cpu_parallelism"] <= 8,
         "Explicit bounded CPU parallelism required")
    need(plan.get("sources") and plan.get("tasks"), "Sources and concrete tasks required")
    source_check(plan["sources"])
    ids = [t["id"] for t in plan["tasks"]]
    need(len(ids) == len(set(ids)), "Unique workflow task IDs required")
    adopted=authenticated_adopted_completed(plan)
    holds=plan.get("dispatch_hold_task_ids",[])
    need(isinstance(holds,list) and len(holds)==len(set(holds)) and set(holds)<=set(ids),
         "Dispatch holds must name distinct existing tasks")
    need(not holds or isinstance(plan.get("dispatch_hold_reason"),str) and bool(plan["dispatch_hold_reason"]),
         "Held work requires a visible operational reason")
    done = set()
    while len(done) < len(ids):
        available = {t["id"] for t in plan["tasks"] if set(t["dependencies"]) <= done}
        need(available-done, "Missing or cyclic workflow dependency")
        done |= available
    need(type(plan.get("max_parallel_rss_bytes")) is int and plan["max_parallel_rss_bytes"]>0
         and type(plan.get("host_ram_margin_bytes")) is int and plan["host_ram_margin_bytes"]>=0,
         "Explicit aggregate and host RAM bounds required")
    total_ram=int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")))*1024
    need(plan["host_ram_margin_bytes"]>=64*2**30 and plan["max_parallel_rss_bytes"]<=1792*2**30
         and plan["max_parallel_rss_bytes"]+plan["host_ram_margin_bytes"]<=total_ram,"Physical RAM and original64GiB margin bound required")
    by_id={t["id"]:t for t in plan["tasks"]}
    def references(value):
        if isinstance(value,dict):
            if "$task" in value:return {value["$task"]}
            return set().union(*(references(v) for v in value.values()))
        if isinstance(value,list):return set().union(*(references(v) for v in value))
        return set()
    def ancestors(task):
        found=set(task["dependencies"]);todo=list(found)
        while todo:
            for dep in by_id[todo.pop()]["dependencies"]:
                if dep not in found:found.add(dep);todo.append(dep)
        return found
    for task in plan["tasks"]:
        need(references(task.get("factory_arguments",{}))<=ancestors(task),
             "Every artifact dependency must be a declared predecessor")
        need(task["resource"] in ("cpu", "gpu0", "external"), "Unknown resource")
        if task["resource"] != "external":
            need((task["factory_module"].startswith("arggnn.benchmarking.") or task["factory_module"] in plan.get("readable_factory_modules",[])) and
                 not task["factory_function"].startswith("_"), "Maintained factory required")
            for key in ("max_seconds", "max_output_bytes", "max_rss_bytes"):
                need(type(task[key]) is int and task[key] > 0, "Finite task caps required")
            if task["resource"]=="gpu0" or task.get("gpu_after_preparation") is True:
                if task["id"] not in adopted:shared_gpu.target(task)
                need(all(type(task.get(k)) is int and 0<task[k]<=80*2**30
                         for k in ("minimum_free_gpu_bytes","max_cuda_reserved_bytes"))
                     and task["max_cuda_reserved_bytes"]<=74*2**30
                     and task["minimum_free_gpu_bytes"]>=task["max_cuda_reserved_bytes"],
                     "Finite physicalGPU2/3/5 free-memory/allocator caps required")
            need(c.contained_path(task["output_directory"]).is_relative_to(c.ROOT/"experiments/results"),
                 "Project result scope required")
    scopes=[c.contained_path(t["output_directory"]).resolve() for t in plan["tasks"] if t["resource"]!="external"]
    need(all(a!=b and not a.is_relative_to(b) and not b.is_relative_to(a)
             for i,a in enumerate(scopes) for b in scopes[i+1:]),
         "Scientific output scopes must be distinct and nonoverlapping")
    support.validate_plan_routes(plan)
    if approved:
        support.validate_settlements(plan)
        policy = current_policy()
        need(policy.get("approved_workflow_ref") == ref, "This exact workflow needs ROOT release")
    return plan


def resolve(value, state):
    """References come only from succeeded dependencies, never a guessed filename."""
    if isinstance(value, dict):
        if "$task" in value:
            need(set(value) <= {"$task", "read", "path"}, "Unknown dependency expression")
            entry = state["tasks"][value["$task"]]
            need(entry["status"] == "succeeded", "Unfinished artifact dependency")
            result = c.read_ref(entry["result_ref"])["value"]
            if value.get("read", False):
                result = c.read_ref(result)
            for part in value.get("path", []):
                result = result[part]
            return result
        return {k:resolve(v,state) for k,v in value.items()}
    if isinstance(value, list):
        return [resolve(v,state) for v in value]
    return value


def external_result(task):
    return support.external_result(task)


def _base_storage_projection(plan, state, new_cap=0):
    """Initial actual baseline plus retained/active/new scopes, no project rescan."""
    base = c.read_ref(plan["storage_baseline_ref"])
    total = base["measured_project_bytes"] + plan["metadata_margin_bytes"]
    for external in plan["external_storage"]:
        # External groups are kept fully reserved until every named actual job
        # has succeeded and every owned group is explicitly drained.
        inactive = True
        for evidence in external.get("completion_evidence", []):
            try:
                q = json.loads(c.contained_path(evidence["queue_path"]).read_text())
                j = next(j for j in q["jobs"] if j["admission"]["job_id"] == evidence["job_id"])
                w = json.loads(c.contained_path(evidence["wait_path"]).read_text())
                inactive &= j["status"] == "succeeded" and j["actual_returncode"] == 0
                inactive &= w.get("actual_direct_wait") is True and w.get("actual_direct_returncode") == 0
                inactive &= w.get("owned_process_group_absent") is True
            except (OSError, KeyError, StopIteration, ValueError):
                inactive = False
        if inactive and external.get("completion_evidence"):
            retained = sum(scope_bytes(c.contained_path(p)) for p in external["scopes"])
            need(retained <= external["cap_bytes"], "External retained scope exceeded its reserve")
            total += retained
        else:
            total += external["cap_bytes"]
    for entry in state["tasks"].values():
        if entry.get("external",False):
            continue
        if "reservation_bytes" in entry:
            total += entry.get("retained_bytes",entry["reservation_bytes"])
    return total+new_cap


def storage_projection(plan,state,new_cap=0):
    return support.storage_projection(plan,state,new_cap,_base_storage_projection)


def external_resources(plan):
    return support.external_resources(plan)


def materialize(plan_ref, plan, task, state):
    output = c.contained_path(task["output_directory"])
    need(not output.exists(), "Never reuse a scientific output directory")
    kwargs = resolve(task.get("factory_arguments",{}), state)
    kwargs["output_directory"] = str(output/"native")
    factory_module,factory_function=support.factory_route(plan,task)
    factory = getattr(importlib.import_module(factory_module),factory_function)
    call = factory(**kwargs)
    need(set(call) <= {"module","function","spec","python","source_refs"}
         and {"module","function","spec","python"} <= set(call), "Concrete scientific callable required")
    need((call["module"].startswith("arggnn.benchmarking.") or call["module"] in plan.get("readable_factory_modules",[])) and
         not call["function"].startswith("_"), "Maintained scientific stage required")
    python = c.contained_path(call["python"])
    need(python.is_file() and str(python).startswith(str(c.ROOT/".runtime")), "Project interpreter required")
    refs = list({r["path"]:r for r in [*plan["sources"],*call.get("source_refs",[])]}.values())
    source_check(refs)
    job = dict(schema="arggnn_workflow_job_v1", plan_ref=plan_ref, task=task,
               task_fingerprint=c.fingerprint(task), call=call, source_refs=refs)
    return save(c.contained_path(plan["records_directory"])/"jobs"/(task["id"]+".json"),job,exclusive=True)


def job_spec(ref):
    job = c.read_ref(ref)
    need(job.get("schema") == "arggnn_workflow_job_v1", "Named current workflow worker required")
    plan = load_plan(job["plan_ref"],approved=True)
    task = job["task"]
    need(task["id"] not in authenticated_adopted_completed(plan), "Historical completed tasks cannot execute again")
    need(task in plan["tasks"] and c.fingerprint(task) == job["task_fingerprint"], "Exact planned task required")
    source_check(job["source_refs"])
    return job


def child(ref):
    return runtime.child(ref,sys.modules[__name__])


def execute(ref):
    return runtime.execute(ref,sys.modules[__name__])


class ProgressReporter:
    """Plain metadata snapshots. No scientific imports, truth reads or hidden publishers."""
    def __init__(self,plan,ref,state_path):
        self.plan,self.ref,self.state_path=plan,ref,state_path
        self.last=0
    def submit(self,state,final=False):
        if not final and time.monotonic()-self.last<60:return
        from .queue_reporting_v1 import publish
        publish(self.plan,self.ref,state,save)
        self.last=time.monotonic()
    def close(self,state):self.submit(state,final=True)


def dispatch_block_reason(plan, policy, ref, task_id):
    """Pause new work without cancelling scientific processes already running."""
    if policy.get("workflow_drain_ref") == ref:
        return "ROOT requested drain before operational update"
    if support.archive_controller_live(plan):
        return "Standalone archive v3 owns capacity accounting; fresh successor dispatch waits for its exact controller exit"
    if task_id in plan.get("dispatch_hold_task_ids", []):
        return plan["dispatch_hold_reason"]
    return None


def run(ref, *, execute_jobs=False, poll_seconds=5):
    plan=load_plan(ref,approved=execute_jobs)
    directory=c.contained_path(plan["records_directory"]);directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/"state.json"
    if state_path.exists():
        state=json.loads(state_path.read_text());need(state["plan_ref"]==ref,"Workflow state/plan differs")
    else:
        state=initial_state(plan,ref)
    validate_adopted_state(plan,state)
    if not execute_jobs:return state
    with (directory/"controller.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not any(e["status"] in ("running","unknown") and not e.get("standalone_archive_dispatch_owner") for e in state["tasks"].values()),
             "Interrupted active workflow requires actual-worker reconciliation, never automatic replay")
        running={}
        reporter=ProgressReporter(plan,ref,state_path)
        try:
            while True:
                policy=current_policy()
                need(policy.get("approved_workflow_ref")==ref,"Workflow release withdrawn")
                changed=False
                support.adoption.refresh_archives(plan,state)
                for task in plan["tasks"]:
                    entry=state["tasks"][task["id"]]
                    if task["id"] in running:
                        process,log=running[task["id"]]
                        if process.poll() is not None:
                            code=process.wait();log.close();del running[task["id"]]
                            if code==75 and not c.contained_path(task["output_directory"]).exists():
                                entry.update(status="pending",waiting_reason="Assigned GPU lease or memory is occupied",not_before=time.time()+30)
                                entry.pop("reservation_bytes",None)
                                changed=True
                                continue
                            entry.update(status="succeeded" if code==0 else "failed",actual_controller_wait=code,finished_at=time.time())
                            out=c.contained_path(task["output_directory"])
                            wait=json.loads((out/"actual_wait.json").read_text()) if (out/"actual_wait.json").exists() else {}
                            if code==0:
                                need(wait.get("actual_direct_returncode")==0 and wait.get("owned_process_group_absent") is True,
                                     "Successful actual scientific wait required")
                                entry["result_ref"]=c.file_ref(out/"result.json")
                            entry["owned_process_group_absent"]=wait.get("owned_process_group_absent") is True
                            if entry["owned_process_group_absent"]:
                                entry["retained_bytes"]=scope_bytes(out)
                            if code==0 and task.get("remeasure_succeeded_tasks"):
                                retirement=c.read_ref(c.read_ref(entry["result_ref"])["value"])
                                support.validate_retirement(task,entry,retirement,plan,state)
                                for owner_id in task["remeasure_succeeded_tasks"]:
                                    owner=next(t for t in plan["tasks"] if t["id"]==owner_id)
                                    owner_entry=state["tasks"][owner_id]
                                    need(owner_entry["status"]=="succeeded" and not owner_entry.get("external"),
                                         "Only succeeded current-workflow scopes may be remeasured")
                                    owner_entry["retained_bytes"]=scope_bytes(c.contained_path(owner["output_directory"]))
                                    owner_entry["retirement_receipt_ref"]=c.read_ref(entry["result_ref"])["value"]
                            changed=True
                    if entry["status"]=="pending" and task["resource"]=="external":
                        result=external_result(task)
                        if result:
                            entry.update(status=result["status"])
                            if result["status"]=="succeeded":
                                entry["result_ref"]=save(directory/"external"/(task["id"]+".json"),result,exclusive=True)
                            else:entry["error"]=result["error"]
                            changed=True
                if changed:save(state_path,state)
                active=[t for t in plan["tasks"] if t["id"] in running]
                external_cpu,external_rss=external_resources(plan)
                cpu=sum(t["resource"]=="cpu" for t in active)+external_cpu
                rss_reserved=sum(t["max_rss_bytes"] for t in active)+external_rss
                gpu=any(t["resource"]=="gpu0" for t in active)
                projection=storage_projection(plan,state)
                undrained=[k for k,e in state["tasks"].items() if e.get("reservation_bytes")
                           and e["status"] in ("failed","unknown") and not e.get("owned_process_group_absent")]
                state["undrained_failed_jobs"]=undrained
                with support.admission_lane() as admitted:
                    if not admitted:
                        for task in plan['tasks']:
                            entry=state['tasks'][task['id']]
                            if entry['status']=='pending' and task['resource']!='external':
                                entry['waiting_reason']='Shared recovery capacity admission is occupied'
                    else:
                        support.adoption.refresh_archives(plan,state)
                        external_cpu,external_rss=external_resources(plan)
                        cpu=sum(t["resource"]=="cpu" for t in active)+external_cpu
                        rss_reserved=sum(t["max_rss_bytes"] for t in active)+external_rss
                        projection=storage_projection(plan,state)
                        for task in plan["tasks"]:
                            entry=state["tasks"][task["id"]]
                            if entry["status"]!="pending" or task["resource"]=="external":continue
                            entry.pop("waiting_reason",None)
                            deps=[state["tasks"][i]["status"] for i in task["dependencies"]]
                            if any(s in ("failed","blocked") for s in deps):
                                entry.update(status="blocked",error="A required predecessor failed");save(state_path,state);continue
                            if not all(s=="succeeded" for s in deps):
                                entry["waiting_reason"]="Required predecessors have not completed";continue
                            dispatch_reason=dispatch_block_reason(plan,policy,ref,task["id"])
                            if dispatch_reason:
                                entry["waiting_reason"]=dispatch_reason;continue
                            if undrained:
                                entry["waiting_reason"]="Failed worker lacks confirmed group cleanup";continue
                            if entry.get("not_before",0)>time.time():continue
                            if task["resource"]=="cpu" and cpu>=plan["cpu_parallelism"]:
                                entry["waiting_reason"]="CPU lanes occupied";continue
                            if rss_reserved+task["max_rss_bytes"]>plan["max_parallel_rss_bytes"]:
                                entry["waiting_reason"]="Aggregate RAM reservation";continue
                            if not support.ram_allows(plan,state,task):
                                entry["waiting_reason"]="Host RAM remaining growth reservation";continue
                            route_reason=support.route_block_reason(plan,task)
                            if route_reason:
                                entry["waiting_reason"]=route_reason;continue
                            if support.gpu_dispatch_needed(task):
                                reason=shared_gpu.admission_reason(plan,task,locked=True)
                                if reason:
                                    entry["waiting_reason"]=reason;continue
                            projected=projection+task["max_output_bytes"]
                            if projected>=min(plan["storage_ceiling_bytes"],980000000000):
                                entry["waiting_reason"]="Storage reservation would exceed ceiling";continue
                            try:
                                job_ref=entry.get("job_ref") or materialize(ref,plan,task,state)
                            except Exception as exc:
                                entry.update(status="failed",error="Materialization: "+type(exc).__name__+": "+str(exc));save(state_path,state);continue
                            entry.update(status="running",job_ref=job_ref,reservation_bytes=task["max_output_bytes"],started_at=time.time(),projected_storage_bytes=projected)
                            entry.pop("waiting_reason",None);save(state_path,state)
                            log=(directory/(task["id"]+".supervisor.log")).open("ab")
                            command=[str(c.ROOT/".runtime/env/bin/python"),"-B","-s","-m",MODULE,"--execute-job","--job-ref",json.dumps(job_ref)]
                            try:
                                process=subprocess.Popen(command,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
                            except BaseException as exc:
                                log.close();entry.update(status="failed",error="Spawn: "+str(exc));save(state_path,state);continue
                            running[task["id"]]=(process,log);entry["pid"]=process.pid;save(state_path,state)
                            projection=projected
                            rss_reserved+=task["max_rss_bytes"]
                            if task["resource"]=="cpu":cpu+=1
                            else:gpu=True
                statuses=[e["status"] for e in state["tasks"].values()]
                state["updated_utc"]=datetime.now(timezone.utc).isoformat()
                state["complete"]=all(s=="succeeded" for s in statuses)
                waiting={k:v["waiting_reason"] for k,v in state["tasks"].items() if v.get("status")=="pending" and "waiting_reason" in v}
                state["waiting_reasons"]=waiting
                state["run_status"]="running" if running or any(e["status"]=="running" for e in state["tasks"].values()) else ("complete" if state["complete"] else "waiting_for_dependencies_or_capacity")
                state["active_cpu_lanes_including_external"]=cpu
                state["reserved_rss_bytes_including_external"]=rss_reserved
                save(state_path,state)
                reporter.submit(state)
                if policy.get("workflow_drain_ref")==ref and not running:
                    state["run_status"]="drained_for_operational_update";save(state_path,state);return state
                if all(s in ("succeeded","failed","blocked") for s in statuses):return state
                if not running and undrained:
                    state["run_status"]="blocked_on_failed_worker_cleanup";save(state_path,state);return state
                if not running and not support.archive_controller_live(plan) and not any(t["resource"]=="external" and state["tasks"][t["id"]]["status"]=="pending" for t in plan["tasks"]):
                    ready_waits=list(waiting.values())
                    if ready_waits and all(reason=="Storage reservation would exceed ceiling" for reason in ready_waits):
                        state["run_status"]="blocked_on_storage_capacity";save(state_path,state);return state
                time.sleep(poll_seconds)
        finally:
            # This controller owns its supervisor children. On cancellation or
            # unexpected error, ask each supervisor to stop and actually wait
            # its own scientific group before releasing controller ownership.
            for process,log in running.values():
                if process.poll() is None:process.terminate()
            for task_id,(process,log) in list(running.items()):
                try:
                    code=process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    state["tasks"][task_id].update(status="unknown",error="Supervisor cleanup did not finish")
                else:
                    task=next(t for t in plan["tasks"] if t["id"]==task_id)
                    output=c.contained_path(task["output_directory"])
                    wait_path=output/"actual_wait.json"
                    wait=json.loads(wait_path.read_text()) if wait_path.exists() else {}
                    state["tasks"][task_id].update(status="failed",actual_controller_wait=code,
                        owned_process_group_absent=wait.get("owned_process_group_absent") is True,
                        error="Controller interrupted, genuine producer wait preserved",finished_at=time.time())
                log.close()
            if running:save(state_path,state)
            reporter.close(state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-ref");parser.add_argument("--job-ref")
    parser.add_argument("--execute",action="store_true");parser.add_argument("--execute-job",action="store_true")
    parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    if args.child:result=child(json.loads(args.job_ref))
    elif args.execute_job:result=execute(json.loads(args.job_ref))
    else:result=run(json.loads(args.plan_ref),execute_jobs=args.execute)
    print(json.dumps(result,sort_keys=True),flush=True)
    if result.get("deferred_before_scientific_start"):sys.exit(75)
    if args.execute and not result.get("complete",False):sys.exit(1)


if __name__=="__main__":main()

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

from . import contracts as c
from .execution import GPU0_UUID, TITLE, gpu0_lease, scope_bytes, _own_group_rss, _stop_group

MODULE = "arggnn.benchmarking.workflow"
SOURCE = "src/arggnn/benchmarking/workflow.py"
SCHEMA = "arggnn_executable_workflow_v1"


def need(ok, message):
    if not ok:
        raise ValueError(message)


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
    for attempt in range(3):
        try:return c.check_execution_release(c.file_ref(c.POLICY_PATH))
        except c.ContractError as exc:
            if str(exc) not in ("Use the current canonical execution policy", "Metadata changed after its reference was frozen") or attempt==2:
                raise
    raise RuntimeError("Canonical policy did not stabilize")


def load_plan(ref, *, approved=False):
    plan = c.read_ref(ref)
    need(plan.get("schema") == SCHEMA and plan.get("physical_gpu") == 0,
         "Frozen one-GPU workflow required")
    need(type(plan.get("cpu_parallelism")) is int and 1 <= plan["cpu_parallelism"] <= 8,
         "Explicit bounded CPU parallelism required")
    need(plan.get("sources") and plan.get("tasks"), "Sources and concrete tasks required")
    source_check(plan["sources"])
    ids = [t["id"] for t in plan["tasks"]]
    need(len(ids) == len(set(ids)), "Unique workflow task IDs required")
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
            need(task["factory_module"].startswith("arggnn.benchmarking.") and
                 not task["factory_function"].startswith("_"), "Maintained factory required")
            for key in ("max_seconds", "max_output_bytes", "max_rss_bytes"):
                need(type(task[key]) is int and task[key] > 0, "Finite task caps required")
            if task["resource"]=="gpu0":
                need(all(type(task.get(k)) is int and 0<task[k]<=80*2**30
                         for k in ("minimum_free_gpu_bytes","max_cuda_reserved_bytes"))
                     and task["max_cuda_reserved_bytes"]<=74*2**30
                     and task["minimum_free_gpu_bytes"]>=task["max_cuda_reserved_bytes"],
                     "Finite physicalGPU0 free-memory/allocator caps required")
            need(c.contained_path(task["output_directory"]).is_relative_to(c.ROOT/"experiments/results"),
                 "Project result scope required")
    scopes=[c.contained_path(t["output_directory"]).resolve() for t in plan["tasks"] if t["resource"]!="external"]
    need(all(a!=b and not a.is_relative_to(b) and not b.is_relative_to(a)
             for i,a in enumerate(scopes) for b in scopes[i+1:]),
         "Scientific output scopes must be distinct and nonoverlapping")
    if approved:
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
    """Adopt an explicitly named prior live pipeline only after both real waits."""
    queue = json.loads(c.contained_path(task["queue_path"]).read_text())
    rows = [j for j in queue["jobs"] if j["admission"]["job_id"] == task["job_id"]]
    need(len(rows) == 1, "Expected one named external job")
    entry = rows[0]
    if entry["status"] in ("failed", "rejected", "unknown"):
        return {"status":"failed", "error":"External predecessor "+entry["status"]}
    if entry["status"] != "succeeded":
        return None
    need(entry["actual_returncode"] == 0, "External controller must record actual success")
    wait = c.read_ref(c.file_ref(task["actual_wait_path"]))
    need(wait.get("actual_direct_wait") is True and wait.get("actual_direct_returncode") == 0
         and wait.get("owned_process_group_absent") is True and wait.get("supervisor_error") is None,
         "External scientific process must genuinely succeed and drain")
    result = c.file_ref(task["result_path"])
    c.read_ref(result)
    return {"status":"succeeded", "value":result, "external_actual_wait_ref":c.file_ref(task["actual_wait_path"])}


def storage_projection(plan, state, new_cap=0):
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


def available_ram_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):return int(line.split()[1])*1024
    raise RuntimeError("Host available RAM is unknown")


def external_resources(plan):
    cpu,rss=0,0
    for row in plan.get("external_resources",[]):
        queue=json.loads(c.contained_path(row["queue_path"]).read_text())
        entries=[j for j in queue["jobs"] if j["admission"]["job_id"]==row["job_id"]]
        need(len(entries)==1,"Named external resource identity missing")
        drained=False
        if entries[0]["status"] in ("succeeded","failed"):
            try:
                wait=json.loads(c.contained_path(row["wait_path"]).read_text())
                drained=wait.get("actual_direct_wait") is True and wait.get("owned_process_group_absent") is True
            except (OSError,KeyError,ValueError):pass
        if not drained:
            cpu+=int(row["resource"]=="cpu");rss+=row["max_rss_bytes"]
    # Native continuation is independently active. Its fixed conservative RSS
    # remains reserved until this workflow is complete.
    rss+=plan.get("external_native_rss_bytes",0)
    return cpu,rss


def materialize(plan_ref, plan, task, state):
    output = c.contained_path(task["output_directory"])
    need(not output.exists(), "Never reuse a scientific output directory")
    kwargs = resolve(task.get("factory_arguments",{}), state)
    kwargs["output_directory"] = str(output/"native")
    factory = getattr(importlib.import_module(task["factory_module"]),task["factory_function"])
    call = factory(**kwargs)
    need(set(call) <= {"module","function","spec","python","source_refs"}
         and {"module","function","spec","python"} <= set(call), "Concrete scientific callable required")
    need(call["module"].startswith("arggnn.benchmarking.") and
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
    need(task in plan["tasks"] and c.fingerprint(task) == job["task_fingerprint"], "Exact planned task required")
    source_check(job["source_refs"])
    return job


def child(ref):
    job=job_spec(ref); task,call=job["task"],job["call"]
    output=c.contained_path(task["output_directory"])
    need(Path(sys.prefix).resolve() == c.contained_path(call["python"]).parent.parent,
         "Declared stage interpreter required")
    try:
        import setproctitle
    except ImportError:
        need(task["resource"] != "gpu0", "GPU process title package is required")
    else:
        setproctitle.setproctitle(TITLE if task["resource"]=="gpu0" else "AI Research Data Preparation")
    if task["resource"]=="gpu0":
        need(os.environ.get("CUDA_VISIBLE_DEVICES")==GPU0_UUID and os.environ.get("ARGGNN_PHYSICAL_GPU")=="0",
             "Physical GPU0 identity required")
        fd=int(os.environ["ARGGNN_GPU_LEASE_FD"])
        owned,named=os.fstat(fd),(c.ROOT/".runtime/locks/gpu_0.lock").stat()
        record=json.loads(os.pread(fd,65536,0))
        need((owned.st_dev,owned.st_ino)==(named.st_dev,named.st_ino)
             and record["launcher_pid"]==os.getppid() and record["job"]==task["id"],
             "Actual inherited GPU0 lease required")
    else:
        need(os.environ.get("CUDA_VISIBLE_DEVICES")=="", "CPU stages cannot use CUDA")
    resource.setrlimit(resource.RLIMIT_FSIZE,(task["max_output_bytes"],task["max_output_bytes"]))
    deadline=float(os.environ["ARGGNN_WORKFLOW_DEADLINE"])
    last=[0.0]
    def check(*_args,**_kwargs):
        now=time.monotonic()
        need(now<deadline,"Scientific wall limit exceeded")
        if now-last[0]<5:return
        last[0]=now
        current_policy()
        need(_own_group_rss(os.getpgrp())<=task["max_rss_bytes"],"Whole worker RSS limit exceeded")
        need(scope_bytes(output)<=task["max_output_bytes"],"Whole worker output limit exceeded")
        torch=sys.modules.get("torch")
        if task["resource"]=="gpu0" and torch is not None and torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved(0)<=task["max_cuda_reserved_bytes"],"CUDA allocation cap exceeded")
    def bounded_runner(argv,cwd,env,limits,log_path):
        need(Path(cwd).resolve().is_relative_to(c.ROOT),"Native working directory escaped project")
        log=c.contained_path(log_path)
        need(log.is_relative_to(output),"Native log must stay in this output")
        log.parent.mkdir(parents=True,exist_ok=True)
        # Native commands retain the admitted device, lease and project caches.
        child_env={**os.environ,**env}
        protected=("CUDA_VISIBLE_DEVICES","ARGGNN_PHYSICAL_GPU","ARGGNN_GPU_LEASE_FD",
                   "ARGGNN_MAX_CUDA_RESERVED_BYTES","ARGGNN_WORKFLOW_DEADLINE")
        need(all(child_env.get(k)==os.environ.get(k) for k in protected),
             "Native command cannot override the worker resource admission")
        for k in ("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","TORCH_HOME","HF_HOME",
                  "MPLCONFIGDIR","NUMBA_CACHE_DIR","CUDA_CACHE_PATH","TRITON_CACHE_DIR"):
            if child_env.get(k):
                need(c.contained_path(child_env[k]).is_relative_to(output),"Native cache escaped output scope")
        inherited=(() if task["resource"]!="gpu0" else (int(os.environ["ARGGNN_GPU_LEASE_FD"]),))
        with log.open("xb") as stream:
            process=subprocess.Popen(argv,cwd=cwd,env=child_env,stdout=stream,stderr=subprocess.STDOUT,
                                     pass_fds=inherited)
            try:
                end=min(deadline,time.monotonic()+limits.get("max_seconds",task["max_seconds"]))
                while process.poll() is None:
                    check()
                    if time.monotonic()>=end:raise TimeoutError("Native command wall limit exceeded")
                    time.sleep(1)
                code=process.wait()
            except BaseException:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        return dict(actual_direct_wait=True,actual_direct_returncode=code,
                    native_process_absent=process.poll() is not None,
                    outer_owned_group_cleanup_pending=True,pid=process.pid)
    module=importlib.import_module(call["module"]); fn=getattr(module,call["function"])
    callbacks=dict(resource_check=check,execution_check=lambda:job_spec(ref),bounded_runner=bounded_runner)
    accepted=inspect.signature(fn).parameters
    kwargs={k:v for k,v in callbacks.items() if k in accepted}
    value=fn(call["spec"],**kwargs)
    check()
    return save(output/"result.json",dict(value=value,job_ref=ref),exclusive=True)


def execute(ref):
    job=job_spec(ref); task,call=job["task"],job["call"]
    output=c.contained_path(task["output_directory"])
    need(not output.exists(),"No automatic retry or output reuse")
    lease_spec=dict(job_id=task["id"],minimum_free_gpu_bytes=task.get("minimum_free_gpu_bytes",74*2**30))
    context=gpu0_lease(lease_spec) if task["resource"]=="gpu0" else nullcontext(None)
    with ExitStack() as stack:
        try:
            fd=stack.enter_context(context)
        except (BlockingIOError, RuntimeError) as exc:
            if task["resource"] != "gpu0":raise
            # The lease may be taken between dispatch and supervisor acquisition.
            # No scientific child or output exists, so this is safe deferral.
            return dict(deferred_before_scientific_start=True,reason=str(exc))
        output.mkdir(parents=True,exist_ok=False)
        temporary=output/"tmp";temporary.mkdir()
        cache=output/"cache";cache.mkdir()
        env={k:v for k,v in os.environ.items() if not k.startswith("ARGGNN_GPU") and k!="ARGGNN_PHYSICAL_GPU"}
        env.update(CUDA_VISIBLE_DEVICES=GPU0_UUID if fd is not None else "",TMPDIR=str(temporary),TMP=str(temporary),TEMP=str(temporary),
            XDG_CACHE_HOME=str(cache),NUMBA_CACHE_DIR=str(cache/"numba"),MPLCONFIGDIR=str(cache/"matplotlib"),
            TORCH_HOME=str(cache/"torch"),HF_HOME=str(cache/"huggingface"),STDPOPSIM_CACHE=str(cache/"stdpopsim"),
            TORCHINDUCTOR_CACHE_DIR=str(cache/"inductor"),TRITON_CACHE_DIR=str(cache/"triton"),CUDA_CACHE_PATH=str(cache/"cuda"),
            PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(c.ROOT)+":"+str(c.ROOT/"src")+":"+str(c.ROOT/"scripts"),
            OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8",NUMBA_NUM_THREADS="8")
        if fd is not None:
            env.update(ARGGNN_GPU_LEASE_FD=str(fd),ARGGNN_PHYSICAL_GPU="0",
                       ARGGNN_MAX_CUDA_RESERVED_BYTES=str(task["max_cuda_reserved_bytes"]))
        deadline=time.monotonic()+task["max_seconds"]
        env["ARGGNN_WORKFLOW_DEADLINE"]=str(deadline)
        command=[str(c.contained_path(call["python"])) ,"-B","-s","-m",MODULE,"--child","--job-ref",json.dumps(ref)]
        cancelled=[];handlers={s:signal.signal(s,lambda n,f:cancelled.append(n)) for s in (signal.SIGTERM,signal.SIGINT)}
        error=None
        with (output/"stage.log").open("xb") as log:
            process=subprocess.Popen(command,cwd=c.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True,pass_fds=(() if fd is None else (fd,)))
            try:
                while process.poll() is None:
                    policy=current_policy()
                    need(policy.get("approved_workflow_ref")==job["plan_ref"],"Workflow release withdrawn")
                    if cancelled or time.monotonic()>=deadline:raise TimeoutError("Worker cancelled or timed out")
                    need(_own_group_rss(process.pid)<=task["max_rss_bytes"],"Worker RSS limit exceeded")
                    need(scope_bytes(output)<=task["max_output_bytes"],"Worker output limit exceeded")
                    time.sleep(2)
            except BaseException as exc:error=type(exc).__name__+": "+str(exc)
            finally:
                _stop_group(process);code=process.wait()
                for s,h in handlers.items():signal.signal(s,h)
        try:os.killpg(process.pid,0);drained=False
        except ProcessLookupError:drained=True
        retained=scope_bytes(output)
        if not drained or retained>task["max_output_bytes"]:error=error or "Owned group did not drain or output cap exceeded"
        result=dict(actual_direct_wait=True,actual_direct_returncode=code,supervisor_error=error,
                    owned_process_group_absent=drained,output_bytes=retained,job_ref=ref,automatic_retry=False)
        save(output/"actual_wait.json",result,exclusive=True)
        need(code==0 and error is None and (output/"result.json").exists(),"Scientific worker failed: "+str(result))
        return result


class ProgressReporter:
    """One background publication task, never blocking scientific supervision."""
    def __init__(self, plan, ref, state_path):
        self.enabled = plan.get("publish_progress") is True
        self.ref, self.state_path = ref, state_path
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark_report") if self.enabled else None
        self.future = None
        self.last_submit = 0.0
        self.last_event = None
        self.score_ids = {t["id"] for t in plan["tasks"] if
            str(t.get("factory_arguments",{}).get("stage", "")).endswith("score")
            or str(t.get("factory_module", "")).endswith("_scoring")}

    def submit(self, state, *, final=False):
        if not self.enabled:
            return
        try:
            if self.future is not None and not self.future.done():
                return
            if self.future is not None:
                self.future.result()
            event = (state.get("complete", False), tuple(sorted(
                (key, row["status"]) for key,row in state["tasks"].items()
                if row["status"] in ("failed","unknown")
                or (key in self.score_ids and row["status"] == "succeeded"))))
            urgent = event != self.last_event
            now = time.monotonic()
            if not final and not urgent and now-self.last_submit < 60:
                return
            from .workflow_reporting import publish
            self.future = self.executor.submit(publish, self.ref, str(self.state_path))
            self.last_event, self.last_submit = event, now
        except Exception as exc:
            # Reporting cannot change a scientific result or stop its workers.
            try:
                save(self.state_path.parent/"reporting_callback_error.json",
                     dict(error=type(exc).__name__+": "+str(exc), at=time.time()))
            except Exception:
                pass

    def close(self, state):
        if self.executor is not None:
            # Queue one final metadata snapshot behind any current publication.
            # No wait occurs in the controller or its owned-worker cleanup.
            try:
                from .workflow_reporting import publish
                self.executor.submit(publish, self.ref, str(self.state_path))
            except Exception:
                pass
            self.executor.shutdown(wait=False)


def dispatch_block_reason(plan, policy, ref, task_id):
    """Pause new work without cancelling scientific processes already running."""
    if policy.get("workflow_drain_ref") == ref:
        return "ROOT requested drain before operational update"
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
        state=dict(plan_ref=ref,tasks={t["id"]:dict(status="pending",external=t["resource"]=="external") for t in plan["tasks"]})
    if not execute_jobs:return state
    with (directory/"controller.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not any(e["status"] in ("running","unknown") for e in state["tasks"].values()),
             "Interrupted active workflow requires actual-worker reconciliation, never automatic replay")
        running={}
        reporter=ProgressReporter(plan,ref,state_path)
        try:
            while True:
                policy=current_policy()
                need(policy.get("approved_workflow_ref")==ref,"Workflow release withdrawn")
                changed=False
                for task in plan["tasks"]:
                    entry=state["tasks"][task["id"]]
                    if task["id"] in running:
                        process,log=running[task["id"]]
                        if process.poll() is not None:
                            code=process.wait();log.close();del running[task["id"]]
                            if code==75 and not c.contained_path(task["output_directory"]).exists():
                                entry.update(status="pending",waiting_reason="GPU0 lease or memory is occupied",not_before=time.time()+30)
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
                                need(retirement.get("successful_new_scratch_only") is True,"Actual retirement completion required")
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
                for task in plan["tasks"]:
                    entry=state["tasks"][task["id"]]
                    if entry["status"]!="pending" or task["resource"]=="external":continue
                    deps=[state["tasks"][i]["status"] for i in task["dependencies"]]
                    if any(s in ("failed","blocked") for s in deps):
                        entry.update(status="blocked",error="A required predecessor failed");save(state_path,state);continue
                    if not all(s=="succeeded" for s in deps):continue
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
                    # Conservatively deduct full reservations even before workers
                    # allocate them. This also covers host activity between polls.
                    if available_ram_bytes()-rss_reserved<task["max_rss_bytes"]+plan["host_ram_margin_bytes"]:
                        entry["waiting_reason"]="Host available RAM";continue
                    if task["resource"]=="gpu0" and gpu:continue
                    # Do not claim a new GPU before the exact shared lease is free.
                    if task["resource"]=="gpu0":
                        path=c.ROOT/".runtime/locks/gpu_0.lock"
                        with path.open("a+") as probe:
                            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            except BlockingIOError:
                                entry["waiting_reason"]="GPU0 lease occupied";continue
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
                state["run_status"]="running" if running else ("complete" if state["complete"] else "waiting_for_dependencies_or_capacity")
                state["active_cpu_lanes_including_external"]=cpu
                state["reserved_rss_bytes_including_external"]=rss_reserved
                save(state_path,state)
                reporter.submit(state)
                if policy.get("workflow_drain_ref")==ref and not running:
                    state["run_status"]="drained_for_operational_update";save(state_path,state);return state
                if all(s in ("succeeded","failed","blocked") for s in statuses):return state
                if not running and undrained:
                    state["run_status"]="blocked_on_failed_worker_cleanup";save(state_path,state);return state
                if not running and not any(t["resource"]=="external" and state["tasks"][t["id"]]["status"]=="pending" for t in plan["tasks"]):
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

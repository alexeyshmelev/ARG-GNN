"""Read-only completion gate before an isolated profile initializes CUDA.

Only ROOT's explicitly named profile directory is observed. This module never
signals a process, changes a reservation, launches a job or retries an update.
"""
import json
import os
from pathlib import Path
import time
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, GIB, need, ref
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity, gpu_free


def owned_group_members(pid):
    members=[]
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields=(path/"stat").read_text().rsplit(")",1)[1].split()
        except (FileNotFoundError,ProcessLookupError):
            continue
        if int(fields[2]) == pid or int(fields[3]) == pid:
            members.append(int(path.name))
    return sorted(members)


def validate_completion(launch,wait,current_identity,members):
    need(wait.get("actual_wait") is True and wait.get("owned_group_absent") is True,
         "Exact predecessor wait and owned-group completion receipt required")
    need(wait.get("actual_returncode") == 0,"Predecessor did not complete successfully")
    need(wait.get("pid") == launch["pid"]
         and wait.get("process_start_ticks") == launch["process_start_ticks"],
         "Predecessor completion must match exact launched process identity")
    need(current_identity != launch["process_start_ticks"],
         "Predecessor original PID/start identity remains alive")
    need(not members,"Predecessor process group/session still has members")


def _status(path,row):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def wait_before_cuda(torch,check,*,predecessor_directory,output,gpu_index,
                     gpu_budget_bytes,timeout_seconds=14400,poll_seconds=5):
    need(not torch.cuda.is_initialized(),"CPU preparation must not initialize CUDA")
    need(0 < poll_seconds <= 30 and timeout_seconds > 0,"Bounded readable gate polling")
    predecessor=launch=request=launch_ref=request_ref=None
    if predecessor_directory is not None:
        predecessor=(ROOT/predecessor_directory).resolve()
        need(predecessor.is_relative_to(ROOT/"research/efficiency_20261027/root/runs"),
             "Explicit ROOT-owned predecessor profile required")
        launch_ref=ref(predecessor/"launch.json")
        request_ref=ref(predecessor/"request.json")
        launch=json.loads((predecessor/"launch.json").read_text())
        request=json.loads((predecessor/"request.json").read_text())
        need(type(launch["pid"]) is int and launch["pid"]>1
             and str(launch["process_start_ticks"]).isdigit(),
             "Exact predecessor PID and start ticks required")
        need(launch["actual_argv"] == request["command"],
             "Predecessor launch must match its own admitted visible command")
    started=time.monotonic()
    last_print=0.
    wait_ref=None
    while True:
        need(not torch.cuda.is_initialized(),"No CUDA initialization while awaiting predecessor")
        check("waiting_for_predecessor_before_CUDA")
        reason=None
        if predecessor is not None:
            need(ref(predecessor/"launch.json")==launch_ref
                 and ref(predecessor/"request.json")==request_ref,
                 "Predecessor identity files changed while waiting")
            receipt=predecessor/"actual_wait.json"
            if not receipt.exists():
                reason="Waiting for exact predecessor actual_wait receipt"
            else:
                wait=json.loads(receipt.read_text())
                current=identity(launch["pid"])
                members=owned_group_members(launch["pid"])
                validate_completion(launch,wait,current,members)
                wait_ref=ref(receipt)
        free=None
        if reason is None:
            free=gpu_free(gpu_index)
            if free < gpu_budget_bytes+2*GIB:
                reason="Waiting for declared GPU budget plus 2 GiB context margin"
        elapsed=time.monotonic()-started
        row=dict(status="waiting" if reason else "ready",reason=reason,
            observed_unix=time.time(),elapsed_seconds=elapsed,
            predecessor_directory=None if predecessor is None else str(predecessor.relative_to(ROOT)),
            predecessor_launch_ref=launch_ref,predecessor_request_ref=request_ref,
            predecessor_actual_wait_ref=wait_ref,
            physical_gpu=gpu_index,current_free_GPU_bytes=free,
            required_free_GPU_bytes=gpu_budget_bytes+2*GIB,
            CUDA_initialized=False,owned_processes_signaled=False)
        _status(output/"gpu_predecessor_status.json",row)
        if not reason:
            print(json.dumps(dict(event="GPU_predecessor_gate_ready",**row)),flush=True)
            return row
        if elapsed-last_print>=30 or last_print==0:
            print(json.dumps(dict(event="GPU_predecessor_gate_waiting",**row)),flush=True)
            last_print=elapsed
        need(elapsed<timeout_seconds,"Predecessor/GPU admission wait expired")
        time.sleep(poll_seconds)


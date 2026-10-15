#!/usr/bin/env python3
"""Launch with full reservation accounting, preserving canonical GPU leases and runtime guards."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v6"),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
from arggnn.resources import ALLOWED_GPUS,gpu_inventory
import amendment_v6 as amendment
import root_native_contract_v1 as contract
from owned_lifecycle_v2 import OwnedTree
from pidfd_backend_v1 import self_probe_v1
from supervisor_v1 import _subreaper,_children
_STORAGE_GUARD=None
def eligible_free(record):
    available=(record["memory_total_mib"]-record["memory_used_mib"])*1048576
    return record["index"]==4 and record["uuid"]==amendment.GPU_UUID and available>=54000000000

import project_budget_v9 as budget

SCHEDULABLE_GPUS = (4,)
SOURCE = "research/reviews/American_C3_native_MLM_amendment_v6/launcher_v6.py"
BASE_LAUNCHER = "scripts/launch_reserved_gpu_v6.py"


def full_storage_check_locked(record,source_sha,launcher_sha,*,required_write_bytes=0):
    amendment.check_lifetime()
    if source_sha!=contract.CORE or launcher_sha!=contract.BASE:raise ValueError("Original core/base source pins")
    if record!=amendment.read(amendment.ORIGINAL_STORAGE_REF):raise ValueError("Exact immutable allocation object")
    account=_STORAGE_GUARD.check_storage_locked_v1(record,boundary=True,required_write_bytes=required_write_bytes)
    return dict(account,actual_launcher_ref=amendment.ref(SOURCE),
        execution_amendment_ref=amendment.authority_ref())


def admitted_storage_locked(path, expected_sha, job, source_sha, launcher_sha):
    record = budget.v4.bound(ROOT, {"path": path, "sha256": expected_sha}, True)
    if (record.get("kind") != "project_budget_v9_storage_record"
            or record.get("job_id") != job or record.get("active") is not True
            or record.get("consumer_uses_v9_guard") is not True):
        raise ValueError("Exact active root-issued v9 consumer storage authority required")
    account = full_storage_check_locked(record, source_sha, launcher_sha)
    return record, account

def save_wait_evidence_locked(record, account, child, returncode, source_sha, launcher_sha):
    output = budget.v4.canonical(ROOT, record["reservation"]["output"])
    if not output.is_dir():
        # A failed-before-output run remains charged; root may record terminal evidence separately.
        return
    path = output / "launcher_exit_full_storage_v1.json"
    data = budget.v4.encoded({"kind": "native_MLM_amended_GPU_launcher_actual_wait_v6",
        "actual_launcher_ref": amendment.ref(SOURCE), "execution_amendment_ref": amendment.authority_ref(),
        "job_id": record["job_id"], "launcher_pid": os.getpid(), "worker_pid": child.pid,
        "returncode": returncode, "wait_status_known": True,
        "completed_unix": time.time(), "reservation_sha256": budget.digest(record["reservation"]),
        "descendant_drain_or_no_live_writer_claim": False,
        "storage_accounting": "concurrent_completed_scope_credits_v3"}) + b"\n"
    full_storage_check_locked(record, source_sha, launcher_sha,
        required_write_bytes=len(data) + 8192)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

def main():
    global _STORAGE_GUARD
    auth,spec,original_storage=amendment.validate_authority(amendment.authority_ref())
    amendment.check_lifetime();self_probe_v1()
    _STORAGE_GUARD=contract.original_guard()
    sharing=contract.read(contract.GPU_AUTH_REF)
    if (sharing.get("schema")!="project_external_GPU_sharing_authorization_v1" or sharing.get("status")!="authorized"
        or sharing.get("root_authorizer")!="root_agent" or sharing.get("physical_GPUs")!=[4,5]
        or sharing.get("maximum_distinct_project_GPUs")!=2 or sharing.get("preserve_external_processes") is not True
        or sharing.get("max_project_cuda_reserved_bytes")!=50000000000 or sharing.get("minimum_additional_free_bytes")!=4000000000):
        raise ValueError("Original exact GPU sharing authorization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=SCHEDULABLE_GPUS)
    parser.add_argument("--job", required=True)
    parser.add_argument("--storage-record", required=True)
    parser.add_argument("--expected-storage-sha256", required=True)
    parser.add_argument("--expected-budget-source-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("module")
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.module!=amendment.ENTRY or args.gpu!=4 or args.job!=amendment.JOB:
        parser.error("Only exact amended qualifier entry/job/GPU4")
    amendment.validate_execution(auth["input_ref"],spec,args.module)
    # Never grant scientific/runtime authority: the bound consumer still validates those original guards.
    lease = None
    child = None
    record = account = None
    selected = None
    lock_dir = ROOT / ".runtime/locks"
    try:
        with budget.v4.mutex(ROOT):
            record, account = admitted_storage_locked(
                args.storage_record, args.expected_storage_sha256, args.job,
                args.expected_budget_source_sha256, args.expected_launcher_sha256)
            if spec["main_source_ref"]!=amendment.ref(args.module):
                raise ValueError("Actual producer must be amendment-bound, not a forged original binding")
            amendment.check_lifetime()
            candidates = {r["index"]: r for r in gpu_inventory()}
            held, free_handles = [], {}
            try:
                for idx in ALLOWED_GPUS:
                    path = budget.v4.canonical(ROOT, lock_dir / f"gpu_{idx}.lock")
                    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                    handle = os.fdopen(fd, "r+")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        free_handles[idx] = handle
                    except BlockingIOError:
                        held.append(idx)
                        handle.close()
                if len(held) >= 2:
                    raise RuntimeError("Two project GPU jobs are already active")
                requested = [args.gpu] if args.gpu is not None else list(SCHEDULABLE_GPUS)
                selected = next((idx for idx in requested if idx in free_handles
                                 and eligible_free(candidates[idx])), None)
                if selected is None:
                    raise RuntimeError("No requested eligible GPU is free")
                lease = free_handles.pop(selected)
                lease.seek(0)
                lease.truncate()
                lease.write(json.dumps({"launcher_pid": os.getpid(), "gpu": selected,
                    "uuid": candidates[selected]["uuid"], "job": args.job,
                    "started_unix": time.time(), "storage_bytes": account["projected_bytes"],
                    "storage_accounting": "concurrent_completed_scope_credits_v3",
                    "storage_journal_tail_sha256": account["journal_tail_sha256"]}))
                lease.flush()
            finally:
                for handle in free_handles.values():
                    handle.close()
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": candidates[selected]["uuid"],
            "ARGGNN_PHYSICAL_GPU": str(selected), "ARGGNN_JOB_NAME": args.job,
            "ARGGNN_GPU_LEASE": str(lock_dir / f"gpu_{selected}.lock"),
            "ARGGNN_GPU_LEASE_FD": str(lease.fileno()),
            "ARGGNN_BUDGET_V9_SOURCE_SHA256": args.expected_budget_source_sha256,
            "ARGGNN_BUDGET_V9_LAUNCHER_SHA256": args.expected_launcher_sha256})
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
        env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
        command = [str(ROOT / ".runtime/env/bin/python"), "-B", str(ROOT/args.module), *args.module_args]
        print(json.dumps({"event": "launch", "gpu": selected, "uuid": candidates[selected]["uuid"],
                          "job": args.job, "command_module": args.module}), flush=True)
        if _children():raise ValueError("No unrelated launcher children")
        _subreaper();amendment.check_lifetime()
        child = subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,
            pass_fds=(lease.fileno(),amendment.lifetime_fd()))
        ownership=OwnedTree(child.pid);interrupted=[]
        def stop(signum,_frame):
            interrupted.append(signum)
            ownership.signal_all(signum)
        original_handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
        control_error=None
        try:
            while child.poll() is None:
                ownership.refresh()
                try:amendment.check_lifetime()
                except BaseException as error:control_error=error;break
                if interrupted:control_error=ValueError("Launcher interrupted");break
                time.sleep(.2)
        finally:
            if child.poll() is None:ownership.signal_all(signal.SIGTERM)
            until=time.monotonic()+20.
            while child.poll() is None and time.monotonic()<until:
                ownership.refresh();time.sleep(.1)
            if child.poll() is None:ownership.signal_all(signal.SIGKILL)
            returncode=child.wait()  # actual direct owned wait, never reconstructed
            while ownership.live() or _children():
                ownership.signal_all(signal.SIGKILL)
                while True:
                    try:pid,status=os.waitpid(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if pid==0:break
                if ownership.live() or _children():time.sleep(.1)
            if not ownership.identities_absent():raise ValueError("Owned launcher descendants not drained")
            for sig,old in original_handlers.items():signal.signal(sig,old)
        if control_error is not None:raise control_error
        print(json.dumps({"event": "supervisor_wait", "worker_pid": child.pid,
                          "returncode": returncode, "wait_status_known": True}), flush=True)
        with budget.v4.mutex(ROOT):
            save_wait_evidence_locked(record, account, child, returncode,
                args.expected_budget_source_sha256, args.expected_launcher_sha256)
        return returncode
    finally:
        if lease is not None:
            lease.close()

if __name__ == "__main__":
    raise SystemExit(main())

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

from arggnn.resources import ALLOWED_GPUS, ROOT, eligible_free, gpu_inventory
import project_budget_v9 as budget

SCHEDULABLE_GPUS = (2, 3)
SOURCE = "scripts/launch_reserved_gpu_full_storage_v1.py"
BASE_LAUNCHER = "scripts/launch_reserved_gpu_v6.py"


def full_storage_check_locked(record, source_sha, launcher_sha, *, required_write_bytes=0):
    """Caller holds the original mutex. Charge full allowances; inspect own scope."""
    if record.get("accounting_mode") != "full_reserved_no_closed_scope_credit":
        raise ValueError("Root full-reservation accounting mode required")
    own = record["reservation"]
    actual_launcher = budget.v4.reference(ROOT, SOURCE)
    base_launcher = budget.v4.reference(ROOT, BASE_LAUNCHER)
    if (base_launcher["sha256"] != launcher_sha
            or record["budget_components"]["launcher"] != base_launcher
            or actual_launcher not in own["consumer_bindings"]
            or record["reserved_new_bytes"] != own["reserved_bytes"]
            or own["output"] != str(budget.v4.canonical(ROOT, record["output_directory"]))):
        raise ValueError("Exact actual launcher binding, base dependency and own output/cap required")
    budget.v4.integer(required_write_bytes, "planned own-scope growth")
    # Keep original immutable/journal/active-reservation checks. No closed-scope
    # credits are used and no global process inspection is requested.
    account = budget.check_reservation_locked(ROOT, own,
        expected_source_sha256=source_sha, expected_launcher_sha256=launcher_sha,
        boundary=False, required_write_bytes=0)
    full_charge = (budget.v4.integer(account["v4_projected_bytes"], "v4 full charge")
                   + budget.v4.integer(account["v9_reserved_bytes"], "v9 full charge"))
    hard_limit = budget.v4.integer(account["hard_limit_bytes"], "human storage limit", 1)
    if hard_limit != 1000000000000 or full_charge >= hard_limit:
        raise ValueError("Full original reservation charge reaches human 1TB limit")
    own_paths = budget.successor_paths(ROOT, own, True)
    measured = budget.v4.measure_scopes(ROOT, own_paths)
    if measured["conservative_scope_bytes"] + required_write_bytes > own["reserved_bytes"]:
        raise ValueError("Own output and planned growth exceed original reservation")
    return dict(account, credited_v9_projection_bytes=account["projected_bytes"],
        projected_bytes=full_charge, limit_bytes=hard_limit,
        accounting_mode="full_reserved_no_closed_scope_credit",
        closed_allowance_credit_applied_to_launcher_bytes=0,
        global_process_inspection_performed=False, actual_launcher_ref=actual_launcher,
        scope_measurement=measured,
        remaining_scope_bytes=own["reserved_bytes"]-measured["conservative_scope_bytes"],
        required_write_bytes=required_write_bytes)


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
    data = budget.v4.encoded({"kind": "reserved_gpu_full_storage_v1_supervisor_wait",
        "job_id": record["job_id"], "launcher_pid": os.getpid(), "worker_pid": child.pid,
        "returncode": returncode, "wait_status_known": True,
        "completed_unix": time.time(), "reservation_sha256": budget.digest(record["reservation"]),
        "descendant_drain_or_no_live_writer_claim": False,
        "storage_accounting": "full_reserved_no_closed_scope_credit"}) + b"\n"
    full_storage_check_locked(record, source_sha, launcher_sha,
        required_write_bytes=len(data) + 8192)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

def main():
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
    if not args.module.startswith("arggnn.") or any(not part.isidentifier() for part in args.module.split(".")):
        parser.error("Only canonical project arggnn modules may be launched")
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
            module_path = "src/" + args.module.replace(".", "/") + ".py"
            expected = budget.v4.reference(ROOT, module_path)
            if expected not in record["reservation"]["consumer_bindings"]:
                raise ValueError("Launched module is not an exact reserved source binding")
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
                    "storage_accounting": "full_reserved_no_closed_scope_credit",
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
        command = [str(ROOT / ".runtime/env/bin/python"), "-B", "-m", args.module, *args.module_args]
        print(json.dumps({"event": "launch", "gpu": selected, "uuid": candidates[selected]["uuid"],
                          "job": args.job, "command_module": args.module}), flush=True)
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                 pass_fds=(lease.fileno(),))
        def stop(signum, _frame):
            if child.poll() is None:
                os.killpg(child.pid, signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        returncode = child.wait()
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

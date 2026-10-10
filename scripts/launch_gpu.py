#!/usr/bin/env python3
"""Reserve one free eligible GPU and launch a project Python module."""
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

from arggnn.resources import ALLOWED_GPUS, ROOT, eligible_free, gpu_inventory, require_storage

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=ALLOWED_GPUS)
    parser.add_argument("--job", required=True)
    parser.add_argument("module", help="Project module, e.g. arggnn.train")
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.module.startswith("arggnn."):
        parser.error("Only project arggnn modules may be launched")
    used = require_storage(headroom_bytes=2_000_000_000)
    lock_dir = ROOT / ".runtime" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    scheduler = open(lock_dir / "scheduler.lock", "a+")
    lease = None
    with scheduler:
        fcntl.flock(scheduler, fcntl.LOCK_EX)
        candidates = {r["index"]: r for r in gpu_inventory()}
        held = []
        free_handles = {}
        try:
            for idx in ALLOWED_GPUS:
                handle = open(lock_dir / f"gpu_{idx}.lock", "a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    free_handles[idx] = handle
                except BlockingIOError:
                    held.append(idx)
                    handle.close()
            if len(held) >= 2:
                raise RuntimeError("Two project GPU jobs are already active")
            requested = [args.gpu] if args.gpu is not None else list(ALLOWED_GPUS)
            selected = next((idx for idx in requested
                             if idx in free_handles and eligible_free(candidates[idx])), None)
            if selected is None:
                raise RuntimeError("No requested eligible GPU is free")
            lease = free_handles.pop(selected)
            lease.seek(0)
            lease.truncate()
            lease.write(json.dumps({
                "launcher_pid": os.getpid(), "gpu": selected,
                "uuid": candidates[selected]["uuid"], "job": args.job,
                "started_unix": time.time(), "storage_bytes": used,
            }))
            lease.flush()
        finally:
            for handle in free_handles.values():
                handle.close()
        fcntl.flock(scheduler, fcntl.LOCK_UN)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = candidates[selected]["uuid"]
    env["ARGGNN_PHYSICAL_GPU"] = str(selected)
    env["ARGGNN_JOB_NAME"] = args.job
    env["ARGGNN_GPU_LEASE"] = str(lock_dir / f"gpu_{selected}.lock")
    env["ARGGNN_GPU_LEASE_FD"] = str(lease.fileno())
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
    child = None
    try:
        command = [str(ROOT / ".runtime/env/bin/python"), "-m", args.module, *args.module_args]
        print(json.dumps({"event": "launch", "gpu": selected, "uuid": candidates[selected]["uuid"],
                          "job": args.job, "command_module": args.module}), flush=True)
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                 pass_fds=(lease.fileno(),))
        def stop(signum, _frame):
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        return child.wait()
    finally:
        if lease is not None:
            lease.close()

if __name__ == "__main__":
    raise SystemExit(main())

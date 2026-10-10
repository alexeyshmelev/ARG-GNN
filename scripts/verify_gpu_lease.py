"""Verify that a child preserves the reservation after its launcher exits."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from arggnn.resources import ROOT, project_path, require_storage


def main():
    require_storage(headroom_bytes=100_000_000)
    log = project_path(".runtime/logs/lease_inheritance_probe_v2.log")
    result_path = project_path("experiments/results/gpu_lease_inheritance_test.json")
    if result_path.exists():
        raise RuntimeError("Probe result exists")
    record = None
    child_pid = None
    with log.open("w") as stream:
        launcher = subprocess.Popen(
            [str(ROOT / ".runtime/env/bin/python"), "scripts/launch_gpu.py",
             "--gpu", "2", "--job", "lease-inheritance-probe-v2", "arggnn.gpu_probe"],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                for line in log.read_text().splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("event") == "probe_ready":
                        record = item
                if record:
                    break
                if launcher.poll() is not None:
                    raise RuntimeError("Probe launcher exited before readiness")
                time.sleep(.2)
            if not record:
                raise RuntimeError("Probe readiness timeout")
            child_pid = int(record["pid"])
            lease_path = project_path(record["lease_path"])
            fd_path = Path(f"/proc/{child_pid}/fd/{record['lease_fd']}")
            if fd_path.resolve() != lease_path:
                raise RuntimeError("Child descriptor does not match project reservation")
            title = Path(f"/proc/{child_pid}/cmdline").read_bytes().split(b"\0")[0].decode()
            if title != "AI Auto Research Pipeline Test":
                raise RuntimeError("Unexpected worker title")
            os.kill(launcher.pid, signal.SIGKILL)
            launcher.wait(timeout=5)
            time.sleep(.3)
            # An inherited descriptor refers to the same open file description.
            os.kill(child_pid, 0)
            lock_held = False
            with lease_path.open("a+") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock_held = True
            if not lock_held:
                raise RuntimeError("Reservation released while child still alive")
            result = {"kind": "infrastructure_test", "launcher_pid": launcher.pid,
                      "worker_pid": child_pid, "physical_gpu": int(record["physical_gpu"]),
                      "observed_title": title, "child_descriptor_path": str(fd_path),
                      "reservation_survived_launcher_sigkill": True,
                      "claim": "Project reservation stays held by inherited worker descriptor"}
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            if child_pid is not None and record is not None:
                cmdline = Path(f"/proc/{child_pid}/cmdline")
                if cmdline.exists() and cmdline.read_bytes().startswith(b"AI Auto Research Pipeline Test\0"):
                    os.kill(child_pid, signal.SIGTERM)
            if launcher.poll() is None:
                launcher.terminate()
                launcher.wait(timeout=5)


if __name__ == "__main__":
    main()

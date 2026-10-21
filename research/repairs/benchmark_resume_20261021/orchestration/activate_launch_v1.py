"""Activate and launch the reviewed GPU5 successor once, preserving all old evidence."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
from research.repairs.benchmark_resume_20261021.gpu import readable_workflow_v1 as queue

c = queue.c
BASE = c.ROOT / "research/repairs/benchmark_resume_20261021/orchestration"
PLAN = "configs/benchmarks/executable_workflow_20261021_gpu5_v1.json"
MODULE = "research.repairs.benchmark_resume_20261021.gpu.readable_workflow_v1"


def main():
    receipt_path = BASE / "launch_receipt_v1.json"
    queue.need(not receipt_path.exists(), "One launch only, inspect the existing receipt")
    ref = c.file_ref(c.contained_path(PLAN))
    plan = queue.load_plan(ref)
    directory = c.contained_path(plan["records_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    queue.need(not (directory / "state.json").exists(), "Fresh controller state required")
    queue.need(not Path("/proc/4171358").exists(), "Retired ARG-GNN controller must remain absent")
    with (c.ROOT / ".runtime/locks/benchmark_recovery_capacity_20261020.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = c.read_json(c.POLICY_PATH)
        queue.need(policy["physical_gpus"] == [5], "Current user assignment is only GPU5")
        previous = queue.save(BASE / "policy_before_launch_v1.json", policy, exclusive=True)
        policy.update(approved_readable_workflow_ref=ref,
            human_instruction="Continue benchmark on physical GPU5 only. Control only verified ARG-GNN task processes.",
            gpu5_successor_released_at_utc=datetime.now(timezone.utc).isoformat(),
            gpu5_successor_release_reason="Preserve 183 completed stages and unchanged scientific budgets. Correct TreeLoom RAM, ARGformer paths and static ARGMix GPU5 execution.",
            gpu5_successor_previous_policy_ref=previous)
        queue.save(c.POLICY_PATH, policy)
        queue.load_plan(ref, approved=True)
        command = [str(c.ROOT / ".runtime/env/bin/python"), "-B", "-s", "-m", MODULE,
            "--plan-ref", json.dumps(ref, sort_keys=True), "--execute"]
        with (directory / "controller.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=c.ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
        stat = Path(f"/proc/{process.pid}/stat").read_text()
        start_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
        receipt = dict(plan_ref=ref, command=command, controller_pid=process.pid,
            controller_start_ticks=start_ticks, physical_gpu=5,
            gpu_uuid="GPU-3db85505-d061-e45d-d154-e7e8c1cad497",
            launched_at_utc=datetime.now(timezone.utc).isoformat(),
            policy_ref=c.file_ref(c.POLICY_PATH), source_ref=c.file_ref(Path(__file__)),
            unrelated_processes_signalled=False)
        queue.save(receipt_path, receipt, exclusive=True)
        print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

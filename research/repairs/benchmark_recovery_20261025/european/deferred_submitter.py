"""ROOT-reviewed deferred repair submission after one exact profile child exits.

Default is read-only. --submit-after-profile records a single bounded wait, then
runs the frozen apply helper. This script never signals processes, starts GPU
work, alters the profile, or bypasses the existing campaign controller.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
BASE = ROOT / "research/repairs/benchmark_recovery_20261025/european"
PROFILE = ROOT / "research/scalability_20261024/profiling/runs/full_gpu5_v2"
STATE = ROOT / ".runtime/benchmarking/paper_campaign_20261024_v2/state.json"
FROZEN = {
    "research/repairs/benchmark_recovery_20261025/european/apply_repair.py":
        "dc5477299ed955ca117b09bd9c982773b7c5f3435340f42a1ae96f75f704c7d1",
    "research/repairs/benchmark_recovery_20261025/european/patch_manifest.json":
        "8f18bf0993b2eb484d702fa35657742d932cbaa5772feaaabdef2f7a1d891070",
    "research/scalability_20261024/profiling/runs/full_gpu5_v2/launch.json":
        "75748a0028fcb279cb9334418c49d72de5c356968cf918a1ef0911e505e5ffb7",
    "research/scalability_20261024/orchestration/run_guarded_profile.py":
        "c332fa235ff90f2fc4868e826ead4955211ae3d6f1e47cea73100b2b7279fa75",
    "src/arggnn/benchmarking/paper_campaign/processes.py":
        "9860e688fe204196b72d5a868edd4a0df9287f349dce1beb39b48d6a94604f84",
}
MODULE = "research.repairs.benchmark_recovery_20261025.european.apply_repair"


def reference(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_frozen():
    for relative, digest in FROZEN.items():
        path = ROOT / relative
        if path.is_symlink() or reference(path)["sha256"] != digest:
            raise ValueError("Frozen deferred-submission input changed: " + relative)


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def profile_exit():
    """Read actual wait evidence; independently inspect the exact owned session."""
    verify_frozen()
    launch = json.loads((PROFILE / "launch.json").read_text())
    path = PROFILE / "actual_wait.json"
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_size > 65536:
        raise ValueError("Ordinary bounded profile actual-wait evidence required")
    waited = json.loads(path.read_text())
    if (waited.get("actual_wait") is not True
            or waited.get("owned_group_absent") is not True
            or waited.get("pid") != launch["pid"]
            or waited.get("process_start_ticks") != launch["process_start_ticks"]
            or type(waited.get("actual_returncode")) is not int
            or waited.get("finished_unix", 0) < launch["started_unix"]):
        raise ValueError("Profile actual-wait evidence does not match frozen launch")
    from arggnn.benchmarking.paper_campaign.processes import session_members
    if session_members(launch["pid"]):
        return None
    return dict(actual_wait_ref=reference(path), actual_wait=waited,
                fresh_owned_session_absent=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit-after-profile", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=18000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 18000 or not 1 <= args.poll_seconds <= 60:
        raise ValueError("Wait must be at most five hours; poll interval at most60seconds")
    os.chdir(ROOT)
    ready = profile_exit()
    revision = json.loads(STATE.read_text())["plan_revision"]
    if not args.submit_after_profile:
        print(json.dumps(dict(status="read_only", profile_release_ready=ready is not None,
            current_plan_revision=revision, maximum_wait_seconds=args.wait_seconds,
            default_performed_no_writes=True, frozen_inputs_verified=len(FROZEN)), indent=2))
        return
    directory = BASE / "deferred_submission"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    write(directory / "request.json", dict(status="waiting_for_exact_profile_exit",
        created_unix=time.time(), maximum_wait_seconds=args.wait_seconds,
        profile_launch_ref=reference(PROFILE / "launch.json"),
        submitter_ref=reference(Path(__file__)), frozen_inputs=FROZEN))
    try:
        while ready is None:
            elapsed = time.monotonic() - started
            write(directory / "status.json", dict(status="waiting_for_exact_profile_exit",
                observed_unix=time.time(), elapsed_seconds=elapsed))
            if elapsed >= args.wait_seconds:
                raise TimeoutError("Exact profile exit was not established within bounded wait")
            time.sleep(min(args.poll_seconds, args.wait_seconds - elapsed))
            ready = profile_exit()
        verify_frozen()
        revision = json.loads(STATE.read_text())["plan_revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("Explicit current integer campaign revision required")
        command = [str(ROOT / ".runtime/env/bin/python3.11"), "-B", "-u", "-m",
                   MODULE, "--expected-plan-revision", str(revision), "--apply"]
        write(directory / "submission.json", dict(status="profile_exited_apply_requested",
            observed_unix=time.time(), command=command, expected_plan_revision=revision,
            profile_exit=ready, profile_success_required=False))
        with (directory / "apply.stdout.log").open("xb") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        result = dict(status="apply_helper_completed" if completed.returncode == 0 else
                      "apply_helper_failed", actual_returncode=completed.returncode,
                      finished_unix=time.time(), direct_GPU_launches=0,
                      process_signals_sent=0, existing_controller_retains_admission=True)
        write(directory / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
        if completed.returncode:
            raise RuntimeError("Frozen apply helper failed; inspect apply.stdout.log")
    except Exception as error:
        if not (directory / "result.json").exists():
            write(directory / "result.json", dict(status="rejected", finished_unix=time.time(),
                error_type=type(error).__name__, error=str(error),
                direct_GPU_launches=0, process_signals_sent=0))
        raise


if __name__ == "__main__":
    main()

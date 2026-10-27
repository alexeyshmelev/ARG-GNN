"""Visible guarded learned-timeline launch plan. Default prints argv; only ROOT may choose --launch."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, need
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity
from research.efficiency_20261027.execution.deferred_gpu_gate import validate_completion, owned_group_members
from research.efficiency_20261027.execution.compare_boundary_timeline_v1 import verify_pins

PYTHON = ".runtime/env/bin/python3.11"
GUARDIAN = "research/efficiency_20261027/root/run_guarded_profile_with_peer.py"


def argv_for(args):
    directory = (ROOT / args.directory).resolve()
    need(directory.is_relative_to(ROOT / "research/efficiency_20261027/root/runs") and not directory.exists(), "Fresh ROOT run directory required")
    need(args.gpu in (3, 4, 5), "Explicit admitted GPU3/4/5")
    predecessor = (ROOT / args.predecessor_directory).resolve()
    need(predecessor.is_relative_to(ROOT / "research/efficiency_20261027/root/runs"), "ROOT predecessor required")
    peers = list(dict.fromkeys(args.peer_directory))
    need(bool(peers), "Explicit ROOT-reviewed peer reservation list required; include predecessor")
    need(str(predecessor.relative_to(ROOT)) in peers, "Preserve predecessor peer reservation accounting")
    for name in peers:
        need((ROOT / name).resolve().is_relative_to(ROOT / "research/efficiency_20261027/root/runs"), "ROOT peer profile directory required")
    command = [PYTHON, "-B", GUARDIAN, "--directory", args.directory, "--gpu", str(args.gpu),
        "--host-gib", "32", "--device-gib", "24", "--wait-seconds", "86400", "--run-seconds", "1800"]
    for peer in peers:
        command.extend(["--peer-directory", peer])
    command.extend(["--", PYTHON, "-B", "-m", "research.efficiency_20261027.execution.compare_boundary_timeline_v1",
        "--output", str(directory / "data"), "--source-pins-sha256", args.source_pins_sha256,
        "--gpu-predecessor-directory", str(predecessor.relative_to(ROOT))])
    return command


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--gpu", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--predecessor-directory", required=True)
    parser.add_argument("--peer-directory", action="append", required=True)
    parser.add_argument("--source-pins-sha256", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    need(Path.cwd() == ROOT, "Run within the authorized project root")
    pins = verify_pins(args.source_pins_sha256)
    command = argv_for(args)
    print(json.dumps(dict(mode="launch" if args.launch else "review_only", argv=command,
        visible_command=shlex.join(command), source_manifest_ref=pins["source_manifest_ref"],
        environment=dict(PYTHONPATH="src:.", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")), indent=2), flush=True)
    if not args.launch:
        return
    predecessor = ROOT / args.predecessor_directory
    launch = json.loads((predecessor / "launch.json").read_text())
    wait = json.loads((predecessor / "actual_wait.json").read_text())
    validate_completion(launch, wait, identity(launch["pid"]), owned_group_members(launch["pid"]))
    env = dict(os.environ, PYTHONPATH="src:.", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    # The reviewed guardian alone owns its new worker session and exact actual_wait receipt.
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

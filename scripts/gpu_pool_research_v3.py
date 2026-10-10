#!/usr/bin/env python3
"""Root-owned source-only allowlist extension; scheduling stays in gpu_pool_v1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

from arggnn import gpu_pool_v1 as pool

CONFIG = "configs/gpu_pool_research_v3.json"
CONFIG_SHA256 = "e67fce8bbd59042f3deba8bbf434c9d8759f1f9d26781df74eccf84b432e29c7"
WORKER = "arggnn.anchor_compact_qualification_runner_v1"
BOOTSTRAP = "arggnn.anchor_compact_cuda_bootstrap_v2"
MODULES = frozenset({
    WORKER, BOOTSTRAP,
    "arggnn.biological_candidate",
    "arggnn.biological_gst_control_v1",
    "arggnn.biological_mechanism_controls_v1",
    "arggnn.official_argmix_biological",
    "arggnn.train_official_argformer_biological",
    "arggnn.native_argformer_exec",
})


def verify_bindings(root=pool.ROOT):
    """Read small source/receipt files only. Root injection is for CPU fixtures."""
    root = Path(root)
    if root != root.resolve():
        raise ValueError("Canonical project root required")

    def bound(record):
        path = pool.canonical_path(root, record["path"])
        if not path.is_file():
            raise ValueError("Regular bound source or receipt required")
        with path.open("rb") as handle:
            data = handle.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024 or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Reviewed source/config/receipt SHA256 differs: " + record["path"])
        return data

    config = json.loads(bound(dict(path=CONFIG, sha256=CONFIG_SHA256)))
    if (config["kind"] != "gpu_pool_research_v3_source_extension"
            or config["status"] != "source_only_actual_execution_not_admitted"
            or config["project_root"] != str(pool.ROOT)
            or config["actual_execution_admitted"] is not False
            or config["root_bootstrap_and_fresh_admission_required"] is not True
            or config["allowed_modules"] != sorted(MODULES)
            or MODULES != pool.MODULES | {WORKER, BOOTSTRAP}):
        raise ValueError("Exact reviewed source-only allowlist required")
    bound(config["base_controller"])
    for path, digest in config["unchanged_entrypoint_and_guards"].items():
        bound(dict(path=path, sha256=digest))
    worker = config["extension_worker"]
    if worker["module"] != WORKER:
        raise ValueError("Only the exact compact qualification runner may be added")
    bound(worker["source"])
    bound(worker["config"])
    review = json.loads(bound(worker["independent_review"]))
    if (review.get("verdict") != "PASS"
            or review.get("runner_sha256") != worker["source"]["sha256"]
            or review.get("protocol_sha256") != worker["config"]["sha256"]
            or review.get("actual_execution_admitted") is not False):
        raise ValueError("Exact source-only independent runner PASS required")
    bootstrap = config["bootstrap_worker"]
    if (bootstrap["module"] != BOOTSTRAP
            or bootstrap["independent_review_required_before_execution"] is not True
            or config["direct_runner_submission_allowed"] is not False):
        raise ValueError("Fixed reviewed bootstrap and no direct runner submission required")
    bound(bootstrap["source"])
    bound(bootstrap["config"])
    return config


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep the original client, fallback, drain, terminate and wait semantics.
    if argv and argv[0] in {"status", "wait", "shutdown"}:
        return pool.main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--pool", required=True)
    start.add_argument("--gpu", type=int, choices=pool.ALLOWED_GPUS, required=True)
    start.add_argument("--headroom-mib", type=int, required=True)
    start.add_argument("--storage-snapshot", required=True)
    start.add_argument("--storage-snapshot-sha256", required=True)
    start.add_argument("--external-storage-reserve-bytes", type=int, required=True)
    start.add_argument("--controller-storage-reserve-bytes", type=int, default=pool.MIN_CONTROLLER_RESERVE)
    start.add_argument("--storage-limit-bytes", type=int, default=450_000_000_000)
    submit = sub.add_parser("submit")
    submit.add_argument("--pool", required=True)
    submit.add_argument("--job", required=True)
    submit.add_argument("--model", required=True)
    submit.add_argument("--seed", type=int, required=True)
    submit.add_argument("--output", required=True)
    submit.add_argument("--peak-mib", type=int, required=True)
    submit.add_argument("--storage-bytes", type=int, required=True)
    submit.add_argument("--policy", choices=("queue", "reject"), default="queue")
    submit.add_argument("--log-limit-bytes", type=int, default=1024 * 1024)
    submit.add_argument("module", choices=sorted(MODULES))
    submit.add_argument("module_args", nargs=argparse.REMAINDER)
    for command in ("status", "wait", "shutdown"):
        sub.add_parser(command, help="Delegates unchanged to gpu_pool_v1")
    args = parser.parse_args(argv)
    if not pool.NAME.fullmatch(args.pool) or len(args.pool) > 32:
        parser.error("Safe pool name of at most 32 characters required")
    if (Path(__file__).resolve() != pool.ROOT / "scripts/gpu_pool_research_v3.py"
            or Path(pool.__file__).resolve() != pool.ROOT / "src/arggnn/gpu_pool_v1.py"):
        raise ValueError("Only canonical reviewed extension and base source admitted")
    verify_bindings()
    if args.command == "start":
        controller = pool.Controller(
            root=pool.ROOT, pool=args.pool, gpu=args.gpu, headroom_mib=args.headroom_mib,
            storage_snapshot=args.storage_snapshot,
            storage_snapshot_sha256=args.storage_snapshot_sha256,
            external_storage_reserve_bytes=args.external_storage_reserve_bytes,
            controller_storage_reserve_bytes=args.controller_storage_reserve_bytes,
            storage_limit_bytes=args.storage_limit_bytes, modules=MODULES).start()

        def stop(_signum, _frame):
            controller.shutdown_requested = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(json.dumps(controller.status(), sort_keys=True), flush=True)
        controller.run()
        return 0
    if args.module == WORKER:
        raise ValueError("Direct runner submission is forbidden; use its reviewed CUDA bootstrap")
    spec = dict(job=args.job, model=args.model, seed=args.seed, output=args.output,
                peak_mib=args.peak_mib, storage_bytes=args.storage_bytes,
                policy=args.policy, log_limit_bytes=args.log_limit_bytes,
                module=args.module, args=args.module_args)
    socket_path = pool.ROOT / f".runtime/gpup/{args.pool}.sock"
    print(json.dumps(pool.request(socket_path, dict(v=1, op="submit", spec=spec)), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)

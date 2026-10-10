#!/usr/bin/env python
"""Source-only preflight by default or root-only, lease-qualified biological GST fitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-source-only", action="store_true")
    modes.add_argument("--train", action="store_true")
    parser.add_argument("--config", default="configs/biological_gst_control_v1.json")
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--observed-manifest",
        default="data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json")
    parser.add_argument("--expected-observed-manifest-sha256")
    parser.add_argument("--prepared-manifest", default="data/biological_candidate_pilot_v1/manifest.json")
    parser.add_argument("--expected-prepared-manifest-sha256")
    parser.add_argument("--seed", type=int, choices=(101, 202, 303), default=101)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--storage-snapshot")
    parser.add_argument("--expected-storage-snapshot-sha256")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the dedicated anogena project root")
    # Standard-library lease/title admission precedes Torch and scientific imports.
    if args.train and args.device == "cuda":
        from arggnn.native_argformer_exec import validate_lease, apply_and_validate_title
        validate_lease()
        apply_and_validate_title()
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Source preflight or CPU fit requires explicitly empty CUDA visibility")
    from arggnn.biological_gst_control_v1 import path, load_protocol, source_only_receipt, train
    config_path = path(args.config)
    cfg = load_protocol(config_path, args.expected_config_sha256)
    if not args.train:
        print(json.dumps(source_only_receipt(cfg, args.expected_config_sha256)))
        return 0
    if not all((args.expected_observed_manifest_sha256, args.expected_prepared_manifest_sha256,
                args.storage_snapshot, args.expected_storage_snapshot_sha256, args.output)):
        parser.error("Fit requires frozen observed/prepared hashes, complete root storage snapshot/hash and fresh output")
    output = path(args.output)
    existed_before = output.exists()
    try:
        result = train(cfg, config_path, args.expected_config_sha256,
            path(args.observed_manifest), args.expected_observed_manifest_sha256,
            path(args.prepared_manifest), args.expected_prepared_manifest_sha256,
            args.seed, output, path(args.storage_snapshot), args.expected_storage_snapshot_sha256,
            device_name=args.device)
        print(json.dumps(dict(status=result["status"], output=str(output), execution_mode="one_seed_two_GST_conditions")))
        return 0
    except Exception as error:
        if output.is_dir() and not existed_before:
            from arggnn.biological_candidate import write_json
            failure = dict(status="failed_partial_preserved", mode="train",
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            write_json(output / "runner_failure.json", failure)
            receipt_path = output / "receipt.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                receipt.update(status="failed_partial_preserved", runner_failure=failure)
                write_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Root-only source check or seven-condition fit reusing frozen H020 observed banks."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-source-only", action="store_true")
    modes.add_argument("--train", action="store_true")
    parser.add_argument("--config", default="configs/biological_mechanism_controls_v1.json")
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-base-config-sha256", required=True)
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
    if args.train and args.device == "cuda":
        from arggnn.native_argformer_exec import validate_lease, apply_and_validate_title
        validate_lease()
        apply_and_validate_title()
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Source check or CPU fit requires explicitly empty CUDA visibility")
    from arggnn import biological_mechanism_controls_v1 as controls
    from arggnn.biological_candidate import path, write_json
    config_path = path(args.config)
    cfg, base_cfg, helpers = controls.load_protocol(config_path, args.expected_config_sha256,
        args.expected_base_config_sha256)
    if args.check_source_only:
        print(json.dumps({"status": "extension_frozen_source_contract_checked_only",
            "config_sha256": args.expected_config_sha256,
            "base_config_sha256": args.expected_base_config_sha256,
            "condition_parameter_counts": {name: sum(p.numel() for p in
                controls.model_factory(name, cfg).parameters()) for name in cfg["models"]},
            "truth_opened": False, "observed_tree_or_bank_opened": False,
            "preparation_or_training_executed": False, "GPU_accessed": False}))
        return 0
    if (not args.expected_observed_manifest_sha256 or not args.expected_prepared_manifest_sha256
            or not args.storage_snapshot or not args.expected_storage_snapshot_sha256):
        parser.error("Fit requires observed/prepared manifests and root storage snapshot SHA256 bindings")
    output = path(args.output or f"results/biological_mechanism_controls_v1_seed{args.seed}")
    controls.require_output_namespace(output)
    try:
        result = controls.train(cfg, base_cfg, helpers, config_path, args.expected_config_sha256,
            path(args.observed_manifest), args.expected_observed_manifest_sha256,
            path(args.prepared_manifest), args.expected_prepared_manifest_sha256, args.seed,
            output, path(args.storage_snapshot), args.expected_storage_snapshot_sha256,
            device_name=args.device)
        print(json.dumps({"status": result["status"], "output": str(output),
            "execution_mode": "one_seed_seven_existing_or_qualified_controls"}))
        return 0
    except Exception as error:
        if output.is_dir():
            failure = {"status": "failed_partial_preserved", "mode": "train",
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc()}
            write_json(output / "runner_failure.json", failure)
            receipt_file = output / "receipt.json"
            if receipt_file.exists():
                receipt = json.loads(receipt_file.read_text())
                receipt["status"] = "failed_partial_preserved"
                receipt["runner_failure"] = failure
                write_json(receipt_file, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

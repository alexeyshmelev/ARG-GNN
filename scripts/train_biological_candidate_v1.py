#!/usr/bin/env python
"""Root-only CPU biological bank preparation or lease-qualified one-seed candidate fitting."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-source-only",action="store_true")
    modes.add_argument("--prepare-observed-cpu","--prepare",dest="prepare",action="store_true")
    modes.add_argument("--train",action="store_true")
    parser.add_argument("--config",default="configs/biological_candidate_pilot_v1.json")
    parser.add_argument("--expected-config-sha256",required=True)
    parser.add_argument("--observed-manifest",
        default="data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json")
    parser.add_argument("--expected-observed-manifest-sha256")
    parser.add_argument("--prepared-manifest",default="data/biological_candidate_pilot_v1/manifest.json")
    parser.add_argument("--expected-prepared-manifest-sha256")
    parser.add_argument("--seed",type=int,choices=(101,202,303),default=101)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--storage-snapshot")
    parser.add_argument("--expected-storage-snapshot-sha256")
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:
        parser.error("Run from the dedicated anogena project root")
    # Standard-library guard precedes Torch/model imports and any CUDA context.
    if args.train and args.device=="cuda":
        from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
        validate_lease();apply_and_validate_title()
    elif os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        parser.error("CPU preparation/check/fit requires explicitly empty CUDA visibility")
    from arggnn.biological_candidate import path,write_json,load_protocol,prepare,train
    config_path=path(args.config)
    cfg,helpers=load_protocol(config_path,args.expected_config_sha256)
    if args.check_source_only:
        print(json.dumps({"status":"frozen_source_contract_checked_only",
                          "config_sha256":args.expected_config_sha256,
                          "truth_opened":False,"observed_tree_or_bank_opened":False,
                          "preparation_or_training_executed":False,"GPU_accessed":False}))
        return 0
    if not args.expected_observed_manifest_sha256 or not args.storage_snapshot or not args.expected_storage_snapshot_sha256:
        parser.error("Execution requires frozen observed manifest and complete root storage snapshot hashes")
    output=path(args.output or ("data/biological_candidate_pilot_v1" if args.prepare else
                               f"results/biological_candidate_pilot_v1_seed{args.seed}"))
    existed_before=output.exists()
    try:
        if args.prepare:
            result=prepare(cfg,helpers,config_path,args.expected_config_sha256,
                           path(args.observed_manifest),args.expected_observed_manifest_sha256,
                           output,path(args.storage_snapshot),args.expected_storage_snapshot_sha256)
        else:
            if not args.expected_prepared_manifest_sha256:
                parser.error("Training requires the root-reviewed prepared manifest SHA256")
            result=train(cfg,helpers,config_path,args.expected_config_sha256,
                         path(args.observed_manifest),args.expected_observed_manifest_sha256,
                         path(args.prepared_manifest),args.expected_prepared_manifest_sha256,
                         args.seed,output,path(args.storage_snapshot),args.expected_storage_snapshot_sha256,
                         device_name=args.device)
        print(json.dumps({"status":result["status"],"output":str(output),
                          "execution_mode":"observed_prepare" if args.prepare else "one_seed_three_conditions"}))
        return 0
    except Exception as error:
        if output.is_dir() and not existed_before:
            failure={"status":"failed_partial_preserved","mode":"prepare" if args.prepare else "train",
                     "error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()}
            write_json(output/"runner_failure.json",failure)
            receipt_file=output/("manifest.json" if args.prepare else "receipt.json")
            if receipt_file.exists():
                receipt=json.loads(receipt_file.read_text())
                receipt["status"]="failed_partial_preserved";receipt["runner_failure"]=failure
                write_json(receipt_file,receipt)
        raise


if __name__=="__main__":
    raise SystemExit(main())

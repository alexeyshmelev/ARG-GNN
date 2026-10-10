#!/usr/bin/env python
"""Root-only current-author ARGMix biological stages; default policy/source only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from arggnn import official_argmix_biological as bridge


def record(path, digest, label):
    if not path or not digest:
        raise ValueError("Explicit factual path/hash required: " + label)
    return {"path": path, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("policy", "qualify-gpu", "prepare-observed",
                        "prepare-native", "train", "infer-and-score"), default="policy")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--output")
    parser.add_argument("--build-receipt")
    parser.add_argument("--expected-build-sha256")
    parser.add_argument("--cpu-admission")
    parser.add_argument("--expected-cpu-admission-sha256")
    parser.add_argument("--gpu-admission")
    parser.add_argument("--expected-gpu-admission-sha256")
    parser.add_argument("--input-receipt")
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--observed-manifest",
        default="data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json")
    parser.add_argument("--expected-observed-manifest-sha256")
    parser.add_argument("--dataset-manifest", default="data/biological_paths_execution_pilot_v1/manifest.json")
    parser.add_argument("--expected-dataset-manifest-sha256")
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = bridge.policy(args.expected_config_sha256)
    if not args.execute:
        print(json.dumps({"status": "current_author_source_policy_only_no_biology_or_models",
            "phase_requested": args.phase, "source_tree_sha256": bridge.SOURCE_TREE_SHA,
            "config_sha256": bridge.sha(bridge.CONFIG), "truth_opened": False,
            "native_code_executed": False, "model_instantiated": False, "GPU": False}))
        return 0
    if args.phase == "policy" or not args.expected_config_sha256 or not args.output:
        parser.error("Actual stages require phase, fresh output and root-reviewed policy hash")
    build = record(args.build_receipt, args.expected_build_sha256, "completed native build")
    cpu = record(args.cpu_admission, args.expected_cpu_admission_sha256, "actual CPU FFI admission")
    gpu = args.phase in {"qualify-gpu", "train", "infer-and-score"}
    ownership = {"created": False}
    if not args.native:
        try:
            bridge.bootstrap(args, gpu, ownership)
        except BaseException as error:
            if ownership["created"]:
                output = bridge.local(args.output)
                bridge.new_json(output / "bootstrap_failure.json", {"status": "failed",
                    "error": str(error), "type": type(error).__name__, "exit_code": 1})
                (output / "exit_status").write_text("1\n")
            raise
    output = bridge.local(args.output)
    if not (output / "exec_admission.json").is_file() or (output / "receipt.json").exists():
        raise RuntimeError("Fresh admitted native output required")
    admission = json.loads((output / "exec_admission.json").read_text())
    if (admission["module_sha256"] != bridge.sha(bridge.MODULE) or
            admission["script_sha256"] != bridge.sha(bridge.SCRIPT) or
            os.environ.get("ARGGNN_ARGMIX_BIOLOGICAL_CODE_SHA256") != bridge.sha(bridge.MODULE)):
        raise RuntimeError("Native exec source admission changed")
    started = time.monotonic()
    result = {"status": "failed", "phase": args.phase, "config_sha256": bridge.sha(bridge.CONFIG),
              "owned_code_sha256": cfg["owned_code_sha256"], "root_execution": True,
              "paper_April_reproduction": False, "exit_code": 1}
    anchor = None
    try:
        result["build_CPU_admission"] = bridge.runtime_admission(build, cpu, cfg)
        if args.phase != "qualify-gpu":
            bridge.gpu_receipt(record(args.gpu_admission, args.expected_gpu_admission_sha256,
                                     "same-code actual GPU admission"), cfg)
        if gpu:
            runtime, anchor, modules = bridge.gpu_runtime(output)
            result["runtime"] = runtime
            if admission["lease"] != runtime["lease"]:
                raise RuntimeError("Native actual scheduler lease differs")
            bridge.new_json(output / "runtime_start.json", runtime)
        else:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise RuntimeError("CPU preparation requested CUDA visibility")
            bridge.source_imports()
        if args.phase == "qualify-gpu":
            result["result"] = bridge.qualify_gpu(cfg, output, runtime, modules)
        elif args.phase == "prepare-observed":
            manifest = record(args.observed_manifest, args.expected_observed_manifest_sha256, "completed Relate manifest")
            dataset = record(args.dataset_manifest, args.expected_dataset_manifest_sha256, "observed parent partition")
            result["result"] = bridge.frame_prepare(manifest, dataset, output, cfg)
        else:
            input_record = record(args.input_receipt, args.expected_input_sha256, "prior fresh-stage receipt")
            if args.phase == "prepare-native":
                result["result"] = bridge.native_archives(input_record, output, cfg)
            elif args.phase == "train":
                result["result"] = bridge.stock_train(input_record, output, cfg, modules)
                bridge.new_json(output / "stock_fit.json", result["result"])
            elif args.phase == "infer-and-score":
                seals = bridge.infer_and_seal(input_record, output, cfg, modules)
                result["prediction_seals"] = seals
                result["external_VAL_scores"] = [bridge.external_score(seal, output, cfg) for seal in seals]
        bridge.policy(result["config_sha256"])
        bridge.source_inventory()
        bridge.runtime_admission(build, cpu, cfg)
        if gpu:
            guard = bridge.load_frozen(bridge.GUARD, bridge.GUARD_SHA, "_argmix_final_lease")
            if guard.validate_lease() != runtime["lease"]:
                raise RuntimeError("Actual GPU lease changed during native phase")
        result.update(status="passed", exit_code=0)
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error),
                             "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["artifacts_sha256"] = {str(p.relative_to(output)): bridge.sha(p)
            for p in sorted(output.rglob("*")) if p.is_file() and
            not {"cache", "tmp"} & set(p.relative_to(output).parts) and
            p.name not in {"receipt.json", "exit_status"}}
        bridge.new_json(output / "receipt.json", result)
        (output / "exit_status").write_text(str(result["exit_code"]) + "\n")
        print(json.dumps({"phase": args.phase, "status": result["status"],
                          "exit_code": result["exit_code"], "output": str(output)}), flush=True)
    del anchor
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

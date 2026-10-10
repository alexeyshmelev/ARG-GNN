#!/usr/bin/env python
"""Root-only observed ARG corpus preparation; default checks policy without trees."""
import argparse
import json
import os
import traceback
from arggnn.train_official_argformer_biological import CONFIG, observed_path, policy, prepare_paths, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--parent-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CPU preparation requires empty CUDA_VISIBLE_DEVICES")
    if not args.execute:
        config = policy(args.expected_config_sha256)
        print(json.dumps({"status": "source_policy_only_no_tree_read", "config_sha256": sha(CONFIG),
                          "parent_ids": config["parent_ids"], "actual_tree_read": False,
                          "model_training": False, "query_ancestry_read": False}))
        return 0
    if not all((args.manifest, args.parent_id, args.output_dir, args.expected_manifest_sha256,
                args.expected_config_sha256)):
        parser.error("Execution requires explicit completed observed manifest, parent, output and reviewed hashes")
    candidate = observed_path(args.output_dir)
    if candidate.exists() or not candidate.parent.is_dir():
        parser.error("Fresh canonical prepared output with existing parent required")
    code = 1
    try:
        prepared = prepare_paths(args.manifest, args.parent_id, args.output_dir,
                                 args.expected_manifest_sha256, args.expected_config_sha256)
        print(json.dumps({key: prepared[key] for key in
                         ("status", "parent_id", "namespace", "path_counts", "vocab_size",
                          "unique_full_paths", "paths_without_ancestor")}))
        code = 0
    except BaseException as error:
        traceback.print_exc()
        if candidate.is_dir():
            (candidate / "preparation_failure.json").write_text(json.dumps({
                "status": "failed_observed_preparation", "error_type": type(error).__name__,
                "error": str(error), "exit_code": 1, "query_ancestry_read": False}) + "\n")
    finally:
        if candidate.is_dir():
            (candidate / "preparation.exit_status").write_text(str(code) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

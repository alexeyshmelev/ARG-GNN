#!/usr/bin/env python3
"""Validate completed development caches. Write new bound configs only explicitly."""
import argparse
import json
from pathlib import Path
import torch
from arggnn.resources import ROOT,project_path
from arggnn.development_training_bindings import (
    sha,verify_completion,make_configs,checked_protocol)
from arggnn.train_anchor_development import checked_development_protocol

DEFAULT="experiments/results/fresh_development_preparation_v1/completion.json"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion",default=DEFAULT)
    parser.add_argument("--expected-completion-sha256")
    parser.add_argument("--write-configs",action="store_true")
    parser.add_argument("--output-directory",default="configs/development_training_bound_v1")
    args=parser.parse_args(argv)
    torch.set_num_threads(4)
    receipt=project_path(args.completion)
    if not receipt.is_file():raise RuntimeError("No successful preparation completion exists; no configs written")
    expected=args.expected_completion_sha256 or sha(receipt)
    if args.write_configs and not args.expected_completion_sha256:
        raise RuntimeError("Writing requires an explicit reviewed completion SHA256")
    completion,dataset,ages=verify_completion(receipt,expected)
    del dataset,ages
    if not args.write_configs:
        print(json.dumps({"event":"development_completion_verified","completion_sha256":expected,
            "train_parents":64,"val_parents":16,"test_parents":0,"configs_written":False}),flush=True)
        return
    output=project_path(args.output_directory)
    if output.exists():raise RuntimeError("Use a fresh dedicated config directory")
    if ROOT/"configs" not in output.parents:raise RuntimeError("Bound configs must stay under repository configs")
    configs=make_configs(completion,receipt,expected)
    output.mkdir(parents=True)
    facts={}
    for family,settings in configs.items():
        path=output/(family+".json")
        with path.open("x") as handle:handle.write(json.dumps(settings,indent=2)+"\n")
        value=sha(path)
        if family=="candidate":checked_development_protocol(path)
        else:checked_protocol(path,value)
        facts[family]={"path":str(path),"sha256":value,"configurations":8}
    with (output/"binding_receipt.json").open("x") as handle:
        handle.write(json.dumps({"completion_receipt":str(receipt),"completion_receipt_sha256":expected,
            "families":facts,"test_access":False,"phase":"fresh_development_validation_only"},indent=2)+"\n")
    print(json.dumps({"event":"development_training_configs_bound","families":facts}),flush=True)


if __name__=="__main__":main()

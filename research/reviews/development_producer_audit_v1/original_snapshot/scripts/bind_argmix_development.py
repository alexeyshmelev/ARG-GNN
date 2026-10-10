#!/usr/bin/env python3
"""Verify completed preparation and optionally write one new bound ARGMix config."""
import argparse
import json
import torch
from arggnn.argmix_development_bindings import bind_completed_development,COMPLETION,PLAN


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion",default=COMPLETION)
    parser.add_argument("--expected-completion-sha256",required=True)
    parser.add_argument("--plan",default=PLAN)
    parser.add_argument("--expected-plan-sha256",required=True)
    parser.add_argument("--write-config",help="Optional fresh config path; omission verifies only")
    args=parser.parse_args(argv)
    torch.set_num_threads(4)
    settings,_,config_sha=bind_completed_development(args.completion,args.expected_completion_sha256,
        args.plan,args.expected_plan_sha256,args.write_config)
    print(json.dumps({"event":"argmix_development_binding_verified","config_written":args.write_config,
        "config_sha256":config_sha,"dataset_manifest_sha256":settings["dataset_manifest_sha256"],
        "dataset_protocol_sha256":settings["dataset_protocol_sha256"],"test_access":False}),flush=True)


if __name__=="__main__":main()

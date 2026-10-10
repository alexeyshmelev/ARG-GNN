#!/usr/bin/env python3
"""Bind both reviewed prior-art grids to actual completed fresh preparation."""
import argparse
import json
from arggnn.priorart_development_bindings import bind_completed_development


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--expected-plan-sha256',required=True)
    parser.add_argument('--completion',required=True);parser.add_argument('--expected-completion-sha256',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    _,_,hashes=bind_completed_development(args.completion,args.expected_completion_sha256,
        args.plan,args.expected_plan_sha256,output_dir=args.output_dir)
    print(json.dumps({'event':'completed_priorart_configs_bound','sha256':hashes,'test_access':False}),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Validate completed ten-family development facts; write a plan only explicitly."""
import argparse
import json
from arggnn.development_selection_bindings import (
    DEFAULT_COMPLETION,SCOPE_AMENDMENT,bind_development_selection,guard_development_path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion",default=DEFAULT_COMPLETION)
    parser.add_argument("--expected-completion-sha256")
    parser.add_argument("--scope-amendment",default=SCOPE_AMENDMENT)
    parser.add_argument("--expected-scope-amendment-sha256")
    parser.add_argument("--family-config",nargs=3,action="append",default=[],metavar=("FAMILY","PATH","SHA256"))
    parser.add_argument("--write-plan",action="store_true")
    parser.add_argument("--output-directory",default="configs/development_selection_bound_v2")
    args=parser.parse_args(argv)
    completion=guard_development_path(args.completion)
    guard_development_path(args.scope_amendment)
    guard_development_path(args.output_directory)
    for _,path,_ in args.family_config:guard_development_path(path)
    if not completion.is_file():raise RuntimeError("No successful preparation completion exists; no plan written")
    if args.expected_completion_sha256 is None or args.expected_scope_amendment_sha256 is None:
        raise RuntimeError("Explicit reviewed completion and scope-amendment SHA256 values are required")
    refs=[{"family_id":family,"path":path,"sha256":digest} for family,path,digest in args.family_config]
    result=bind_development_selection(completion,args.expected_completion_sha256,
        scope_amendment_path=args.scope_amendment,expected_scope_amendment_sha256=args.expected_scope_amendment_sha256,
        family_configs=refs,output_directory=args.output_directory if args.write_plan else None)
    print(json.dumps({"event":"fresh_development_selection_plan_bound" if args.write_plan else "fresh_development_selection_plan_valid",
        "plan_sha256":result.sha256,"binding_receipt_sha256":result.receipt_sha256,
        "plan_path":result.path,"binding_receipt_path":result.receipt_path,
        "families":10,"required_baselines":9,"plans_written":args.write_plan,
        "selection_performed":False,"final_test_access":False}),flush=True)


if __name__=="__main__":main()

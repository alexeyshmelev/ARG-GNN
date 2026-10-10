#!/usr/bin/env python3
"""Conditional secondary admission or independent two-LR selection synthesis."""
import argparse
import json
from arggnn import secondary_control_bindings as binding
from arggnn.development_selection_bindings import guard_development_path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal",default=binding.PROPOSAL);parser.add_argument("--expected-proposal-sha256")
    parser.add_argument("--main-plan");parser.add_argument("--expected-main-plan-sha256")
    parser.add_argument("--main-selection",default="experiments/results/development_selection_v2/selection.json")
    parser.add_argument("--expected-main-selection-sha256");parser.add_argument("--write-config",action="store_true")
    parser.add_argument("--output-config",default="configs/secondary_development_bound_v1.json")
    parser.add_argument("--synthesize",action="store_true");parser.add_argument("--config");parser.add_argument("--expected-config-sha256")
    parser.add_argument("--run-receipt",action="append",nargs=2,default=[]);parser.add_argument("--failure-ledger",action="append",nargs=2,default=[])
    parser.add_argument("--output-selection")
    args=parser.parse_args(argv)
    if args.synthesize:
        if not args.config or not args.expected_config_sha256:raise RuntimeError("Actual captured secondary config/hash required")
        result=binding.synthesize_secondary_selection(args.config,args.expected_config_sha256,
            run_receipts=[{"path":path,"sha256":digest} for path,digest in args.run_receipt],
            failure_ledgers=[{"path":path,"sha256":digest} for path,digest in args.failure_ledger],output_path=args.output_selection)
        print(json.dumps({"event":"secondary_attribution_selection","sha256":result["sha256"],"matrix_resolved":result["values"]["matrix_resolved"],"primary_selection_changed":False}),flush=True)
        return
    for path in (args.proposal,args.main_selection,args.output_config):guard_development_path(path)
    if args.main_plan:guard_development_path(args.main_plan)
    if not guard_development_path(args.main_selection).is_file():raise RuntimeError("No genuine complete main-study selection exists; no secondary config written")
    if not all((args.expected_proposal_sha256,args.main_plan,args.expected_main_plan_sha256,args.expected_main_selection_sha256)):
        raise RuntimeError("Explicit reviewed proposal, main plan and main receipt hashes required")
    result=binding.bind_secondary_development({"path":args.proposal,"sha256":args.expected_proposal_sha256},
        {"path":args.main_plan,"sha256":args.expected_main_plan_sha256},
        {"path":args.main_selection,"sha256":args.expected_main_selection_sha256},output_path=args.output_config if args.write_config else None)
    print(json.dumps({"event":"secondary_development_bound" if args.write_config else "secondary_development_valid",
        "config_sha256":result.sha256,"controls":result.settings["controls"],"optimizer_starts":result.settings["optimizer_starts"],
        "config_written":args.write_config,"final_test_access":False}),flush=True)


if __name__=="__main__":main()

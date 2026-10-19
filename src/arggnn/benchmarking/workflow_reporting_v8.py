"""V8 current reports in a private v6 formatter/publisher namespace."""
from __future__ import annotations
import argparse
import importlib.util
import json
import time
from . import contracts as c

_spec=importlib.util.spec_from_file_location(
    "arggnn.benchmarking._reporting_v8_isolated",
    c.ROOT/"src/arggnn/benchmarking/workflow_reporting_v6.py")
_private=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_private)
_core=_private._core
BEGIN="<!-- BEGIN CURRENT BENCHMARK WORKFLOW -->"
END="<!-- END CURRENT BENCHMARK WORKFLOW -->"
STATUS_PATH="research/benchmark_active_current.json"
SCORE_PATH="research/active_matched_benchmark_results.json"
RUNTIME_PATH=".runtime/benchmarking/runtime_reporting_v8"
for owner in (_private,_core):
    owner.BEGIN,owner.END=BEGIN,END
    owner.STATUS_PATH,owner.SCORE_PATH=STATUS_PATH,SCORE_PATH
_core.REPORT_PATHS=("README.md",STATUS_PATH,SCORE_PATH)
_core.RUNTIME_PATH=RUNTIME_PATH
_snapshot=_private._snapshot
_markdown=_private._markdown
publish=_private.publish
ProgressReporter=_private.ProgressReporter


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",required=True)
    parser.add_argument("--state",required=True)
    parser.add_argument("--loop",action="store_true")
    parser.add_argument("--exit-on-complete",action="store_true")
    args=parser.parse_args(argv)
    plan_ref=c.file_ref(args.plan)
    while True:
        approved=c.read_json("configs/benchmarks/execution_policy.json").get("approved_cohort_continuation_ref")
        if approved!=plan_ref:
            print(json.dumps(dict(status="superseded",plan_ref=plan_ref),sort_keys=True),flush=True)
            return 0
        result=publish(plan_ref,args.state)
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
        if not args.loop:
            return 1 if result.get("status")=="error" else 0
        if args.exit_on_complete and result.get("status") not in ("error","busy") \
                and c.read_json(args.state).get("complete") is True:
            return 0
        time.sleep(60)


if __name__=="__main__":raise SystemExit(main())

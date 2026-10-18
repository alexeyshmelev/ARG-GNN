"""Isolated repaired reporter: legacy queue lookups exclude workflow predecessors."""
from __future__ import annotations
import argparse
import importlib.util
import json
import time
from . import contracts as c

_spec=importlib.util.spec_from_file_location(
    "arggnn.benchmarking._reporting_v6_isolated",
    c.ROOT/"src/arggnn/benchmarking/workflow_reporting_v5.py")
_private=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_private)
_core=_private._private._core
_legacy_external=_core._external


def _external(plan):
    # Filter a temporary view for legacy queue lookup only. V5 snapshot still
    # sees the original plan and reports inherited workflow jobs separately.
    queue_plan=dict(plan)
    queue_plan["tasks"]=[task for task in plan.get("tasks",[])
                         if not (task.get("resource")=="external" and "workflow_predecessor" in task)]
    return _legacy_external(queue_plan)


_core._external=_external
BEGIN,END=_private.BEGIN,_private.END
STATUS_PATH,SCORE_PATH=_private.STATUS_PATH,_private.SCORE_PATH
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
        approved=c.read_json("configs/benchmarks/execution_policy.json").get("approved_repair_continuation_ref")
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

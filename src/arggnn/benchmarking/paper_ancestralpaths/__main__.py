"""Readable CLI for inventory, synthetic checks and inert admission descriptors."""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path
from .. import european_native_models as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", required=True)
    scoring = commands.add_parser("scoring-smoke")
    scoring.add_argument("--output", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--queue-config", required=True); inv.add_argument("--output", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--reuse", required=True); plan.add_argument("--output", required=True)
    plan.add_argument("--output-root", required=True); plan.add_argument("--treeloom-prediction-task", required=True)
    describe = commands.add_parser("describe")
    call = commands.add_parser("factory")
    call.add_argument("--plan", required=True); call.add_argument("--task", required=True)
    call.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        from .smoke import run
        result = run(args.output)
        print(json.dumps(dict(status=result["status"], receipt=args.output + "/receipt.json")))
    elif args.command == "scoring-smoke":
        from .scoring_smoke import run
        result = run(args.output)
        print(json.dumps(dict(status=result["status"], receipt=args.output + "/receipt.json")))
    elif args.command == "inventory":
        from .inventory import inventory
        inventory(args.queue_config, args.output)
        print(json.dumps(dict(output=args.output, test_truth_opened=False)))
    elif args.command == "plan":
        from .plan import build
        result = build(io._ref(io._path(args.reuse)), output_root=args.output_root,
                       treeloom_prediction_task=args.treeloom_prediction_task)
        with io._path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        print(json.dumps(dict(output=args.output, plan_only=True, tasks=len(result["tasks"]))))
    elif args.command == "factory":
        from .workflow import make_call
        plan = json.loads(io._path(args.plan).read_text())
        matching = [t for t in plan["tasks"] if t["id"] == args.task]
        io._need(len(matching) == 1, "Exact plan task identity required")
        task = matching[0]
        # Task links are intentionally unresolved until ROOT's DAG worker resolves dependencies.
        result = make_call(output_directory=task["output_directory"], **task["factory_arguments"])
        with io._path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        print(json.dumps(dict(output=args.output, executed=False)))
    else:
        from .model import model_spec
        print(json.dumps(model_spec(), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise

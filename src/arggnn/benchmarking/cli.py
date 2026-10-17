"""Read-only benchmark planning and explicitly approval-gated execution."""
from __future__ import annotations

import argparse
import json
import sys

from .contracts import (ROOT, POLICY_PATH, ContractError, file_ref, preflight,
                        read_json, validate_admission, validate_execution_policy, check_execution_release)
from .scheduler import PersistentScheduler

STATE = ROOT / ".runtime/benchmarking/queue.json"
LOCK = ROOT / ".runtime/benchmarking/controller.lock"


def _scheduler():
    return PersistentScheduler(STATE, LOCK, validate_admission=validate_admission,
                               validate_execution_policy=validate_execution_policy,
                               cpu_parallelism=2)


def _protocol(name):
    if "/" in name or "\\" in name or name.startswith("."):
        raise ContractError("Use a registered protocol ID")
    return read_json("configs/benchmarks/protocols/" + name + ".json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("methods")
    commands.add_parser("protocols")
    commands.add_parser("status")
    for name in ("plan", "validate"):
        p = commands.add_parser(name)
        p.add_argument("protocol")
    p = commands.add_parser("enqueue")
    p.add_argument("--admission-ref", required=True, help="JSON path/SHA256 of ROOT job receipt")
    p = commands.add_parser("run")
    p.add_argument("--execute", action="store_true", help="Requires a separately approved canonical policy")
    p.add_argument("--execution-policy-ref", help="Exact JSON path/SHA256 of the approved policy")
    args = parser.parse_args(argv)
    try:
        if args.command == "methods":
            from .methods import list_methods
            result = list_methods()
        elif args.command == "protocols":
            result = [{"id": p.stem, "comparison_kind": read_json(p)["comparison_kind"]}
                      for p in sorted((ROOT / "configs/benchmarks/protocols").glob("*.json"))]
        elif args.command in ("plan", "validate"):
            protocol = _protocol(args.protocol)
            result = preflight(protocol)
            result["stages"] = protocol.get("stage_order", [])
            result["execution_policy"] = read_json(POLICY_PATH)
        elif args.command == "status":
            result = {"execution_policy": read_json(POLICY_PATH),
                      "queue_present": STATE.exists(),
                      "queue": read_json(STATE) if STATE.exists() else None}
        elif args.command == "enqueue":
            result = _scheduler().enqueue(json.loads(args.admission_ref))
        else:
            if not args.execute and not STATE.exists():
                result = {"plan": [], "execute": False, "jobs_started": 0}
            else:
                policy = json.loads(args.execution_policy_ref) if args.execution_policy_ref else None
                if args.execute:
                    check_execution_release(policy)
                result = _scheduler().run(execute=args.execute, execution_policy_ref=policy)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False, default=str))
        return 0
    except (ContractError, ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "blocked", "error": str(error), "jobs_started": 0}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

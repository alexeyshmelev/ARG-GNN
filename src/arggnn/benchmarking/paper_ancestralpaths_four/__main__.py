"""Inspect the four-path condition or build its independent task plan."""
from __future__ import annotations

import argparse
import json
from .. import european_native_models as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe")
    plan = commands.add_parser("plan")
    plan.add_argument("--source-config", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "describe":
        from .features import feature_spec
        from .model import model_spec
        print(json.dumps(dict(features=feature_spec(), model=model_spec()), indent=2))
    else:
        from .plan import build
        result = build(io._ref(io._path(args.source_config)), output_root=args.output_root)
        with io._path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(output=args.output, tasks=len(result["tasks"]),
                              admitted=False, executed=False)))


if __name__ == "__main__":
    main()

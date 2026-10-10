#!/usr/bin/env python
"""Validate fixed biological pilot; root-only --execute requires reviewed SHA."""
import argparse
import json
from pathlib import Path

from arggnn.biological_paths_simulation import (
    ROOT, canonical_digest, narrow_counterfactual_check, prepare_pilot,
    project_path, sha256, validate_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/biological_paths_execution_pilot_v1.json")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--check-counterfactual", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = project_path(args.config)
    actual_hash = sha256(path)
    config = json.loads(path.read_text())
    if args.expected_config_sha256 is not None and args.expected_config_sha256 != actual_hash:
        parser.error("reviewed configuration bytes changed")
    if args.execute and not args.expected_config_sha256:
        parser.error("--execute requires root-reviewed --expected-config-sha256")
    if args.execute and args.check_counterfactual:
        parser.error("validation counterfactual and biological execution are separate")
    result = validate_config(config)
    result["config_sha256"] = actual_hash
    result["implementation_sha256"] = {
        str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT / "src/arggnn/biological_paths_simulation.py",
            ROOT / "scripts/prepare_biological_paths_pilot.py")
    }
    if args.check_counterfactual:
        result["counterfactual"] = narrow_counterfactual_check(config)
    if args.execute:
        result = prepare_pilot(config, config_sha256=actual_hash)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

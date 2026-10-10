#!/usr/bin/env python
"""Lock-first custody preparation or operational prediction-roster sealing."""
import argparse
import json
from arggnn.confirmatory_custodian import (
    custodian_schema_values, prepare_confirmatory_release, seal_confirmatory_prediction_roster)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", action="store_true", help="Print prospective schema only")
    parser.add_argument("--seal-roster", action="store_true", help="Seal terminal prediction receipts; never open targets")
    for name in ("lock", "expected-lock-sha256", "request", "expected-request-sha256", "output",
                 "admission-ledger", "expected-admission-ledger-sha256",
                 "initial-roster", "expected-initial-roster-sha256"):
        parser.add_argument("--"+name)
    parser.add_argument("--terminal-receipt", nargs=2, action="append", metavar=("PATH", "SHA256"), default=[])
    args = parser.parse_args(argv)
    if args.schema:
        if args.seal_roster or args.terminal_receipt or any(value is not None
                for name, value in vars(args).items() if name not in ("schema", "seal_roster", "terminal_receipt")):
            parser.error("--schema cannot be combined with an actual operation")
        print(json.dumps(custodian_schema_values(), sort_keys=True, indent=2)); return
    common = ("lock", "expected_lock_sha256", "output")
    if any(getattr(args, name) is None for name in common):
        parser.error("Operation requires explicit lock path/SHA256 and fresh output directory")
    if args.seal_roster:
        fields = ("admission_ledger", "expected_admission_ledger_sha256", "initial_roster", "expected_initial_roster_sha256")
        if any(getattr(args, name) is None for name in fields) or args.request is not None or args.expected_request_sha256 is not None:
            parser.error("Roster sealing requires explicit ledger/initial-roster paths and SHA256s")
        result = seal_confirmatory_prediction_roster(args.lock, args.expected_lock_sha256,
            args.admission_ledger, args.expected_admission_ledger_sha256,
            args.initial_roster, args.expected_initial_roster_sha256,
            terminal_receipts=[{"path": path, "sha256": digest} for path, digest in args.terminal_receipt],
            output_directory=args.output)
        print(json.dumps({"path": result.path, "sha256": result.sha256,
            "cell_counts": result.receipt["cell_counts"], "positive_confirmation_authorized": False}, sort_keys=True))
    else:
        if args.request is None or args.expected_request_sha256 is None or args.terminal_receipt or any(
                getattr(args, name) is not None for name in ("admission_ledger", "expected_admission_ledger_sha256",
                    "initial_roster", "expected_initial_roster_sha256")):
            parser.error("Preparation requires explicit request path/SHA256 only")
        result = prepare_confirmatory_release(args.lock, args.expected_lock_sha256, args.request,
            args.expected_request_sha256, output_directory=args.output)
        print(json.dumps({"path": result.path, "sha256": result.sha256,
            "admitted_parents": result.receipt["admitted_parents"],
            "expected_prediction_cells": result.receipt["expected_prediction_cells"],
            "positive_confirmation_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()


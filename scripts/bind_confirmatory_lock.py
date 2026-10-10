#!/usr/bin/env python3
"""Validate a factual confirmatory lock request; emit only with --write-lock."""
import argparse
import json
from arggnn.confirmatory_lock import bind_confirmatory_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--expected-request-sha256', required=True)
    parser.add_argument('--write-lock', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.write_lock and args.output is None:
        parser.error('--write-lock requires a new --output path')
    if args.output is not None and not args.write_lock:
        parser.error('--output requires explicit --write-lock')
    result = bind_confirmatory_lock(args.request, args.expected_request_sha256,
                                    output_path=args.output if args.write_lock else None)
    print(json.dumps({'event': 'confirmatory_provenance_lock_written' if args.write_lock else 'confirmatory_provenance_lock_valid',
                      'sha256': result.sha256, 'path': result.path, 'test_preparation_performed': False,
                      'final_test_access': False, 'workflow_permissions_enforced': False}), flush=True)


if __name__ == '__main__':
    main()

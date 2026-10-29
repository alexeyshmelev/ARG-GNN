"""Execution-only claim and timeout adapter for the unchanged native Relate code."""
import argparse
import json
import subprocess
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import claims


class UnlimitedNativeSubprocess:
    """Forward visible argv and actual subprocess waits; remove only wall timeout."""
    DEVNULL = subprocess.DEVNULL
    STDOUT = subprocess.STDOUT

    def run(self, *args, **kwargs):
        forwarded = dict(kwargs)
        forwarded['timeout'] = None
        return subprocess.run(*args, **forwarded)


def claimed_fresh_directory(original, output, marker_value):
    target = r.path(output)
    used = False

    def fresh(path, guard):
        nonlocal used
        candidate = r.path(path)
        if candidate != target:
            return original(path, guard)
        claims.need(not used, 'Claimed native output may be opened only once')
        claims.need(target.is_dir() and not target.is_symlink(), 'Ordinary claimed output required')
        claims.need(sorted(p.name for p in target.iterdir()) == [claims.MARKER], 'Claim must be empty except marker')
        claims.need(json.loads((target / claims.MARKER).read_text()) == marker_value, 'Exact native claim ownership')
        guard(0)
        used = True
        return target
    return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec = json.loads(r.path(args.spec).read_text())
    for source in spec['source_refs']:
        claims.need(r.ref(source['path']) == source, 'Pinned native source changed: ' + source['path'])
    r.read(spec['config_ref'])
    r.read(spec['commands_ref'])
    from arggnn.benchmarking.paper_argmix import paper_argmix_relate as native
    from arggnn.benchmarking.paper_argmix.paper_argmix_features import ScopeGuard
    output_guard = ScopeGuard(spec['output_directory'], spec['payload_max_output_bytes'], 9223372036854775807)
    native_guard = ScopeGuard(spec['native_work'], spec['native_work_max_bytes'], 9223372036854775807)
    def guard(required_write_bytes=0):
        output_guard(required_write_bytes)
        native_guard(0)
    original_fresh = native.g._fresh_directory
    original_subprocess = native.subprocess
    native.g._fresh_directory = claimed_fresh_directory(original_fresh, spec['output_directory'], spec['claim_marker'])
    native.subprocess = UnlimitedNativeSubprocess()
    try:
        result = native.run_parallel(str(r.path(spec['config_ref']['path'])), spec['native_work'], spec['output_directory'], guard=guard)
        receipt = dict(schema='native_execution_amendment_v1', spec_ref=r.ref(args.spec),
                       unchanged_native_source_ref=r.ref(native.__file__),
                       changes=['Accept exact owned empty claim once', 'Native subprocess timeout=None; genuine subprocess.run wait retained',
                                'ScopeGuard wall deadline INT64_MAX; byte caps unchanged'],
                       native_terminal_ref=r.ref(r.path(spec['output_directory']) / 'terminal.json'))
        r.save(r.path(spec['output_directory']) / 'execution_amendment.json', receipt)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        native.g._fresh_directory = original_fresh
        native.subprocess = original_subprocess


if __name__ == '__main__':
    main()

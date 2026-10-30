"""Preserve the empty failed output and launch the reviewed recovery exactly once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.argmix.recovery_20261030_queue import controller_v1 as controller

BASE = Path('research/joint_evaluation_20261029')
MANIFEST = BASE/'argmix/recovery_20261030_queue/controller_manifest_v1.json'
EXPECTED_MANIFEST_SHA = '1e0a206fff1ec45c391708ca4cbfe6a8dd3bbc27e813c4db4d44c7316c3fcb26'
RECORDS = BASE/'argmix/recovery_20261030_queue/records_v1'


def save(path, value):
    with r.path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    review_ref = r.ref(args.review)
    review = r.read(review_ref)
    assert review['passed'] is True
    manifest_ref = r.ref(MANIFEST)
    assert manifest_ref['sha256'] == EXPECTED_MANIFEST_SHA
    manifest = r.read(manifest_ref)
    assert manifest_ref == review['manifest_ref']
    for reference in manifest['source_refs']:
        assert r.ref(reference['path']) == reference
    plan = r.read(manifest['plan_ref'])
    controller.validate_recovery(manifest, manifest_ref, plan)
    assert controller.host_available_bytes() >= manifest['minimum_host_available_bytes']
    source = r.path('experiments/results/joint_argmix_20261029/cache_build')
    destination = source.with_name('cache_build_failed_20261030_000743')
    assert source.is_dir() and not source.is_symlink() and not list(source.iterdir())
    assert not destination.exists()
    folder = r.path(RECORDS)
    assert not folder.exists(), 'A previous recovery launch exists; inspect it instead of retrying'
    command = [str(r.ROOT/'.runtime/env/bin/python3.11'), '-B', '-s', '-u', '-m',
               'research.joint_evaluation_20261029.argmix.recovery_20261030_queue.controller_v1',
               '--manifest', str(MANIFEST), '--gpu', '3']
    if not args.apply:
        print(json.dumps({'valid': True, 'launch': command, 'manifest_ref': manifest_ref,
                          'empty_directory_to_preserve': str(source), 'preserved_as': str(destination)}))
        return
    folder.mkdir()
    source.rename(destination)
    save(RECORDS/'failed_empty_directory_preserved.json', {
        'source': str(source.relative_to(r.ROOT)), 'destination': str(destination.relative_to(r.ROOT)),
        'verified_empty_before_rename': True, 'files_deleted': 0,
        'failed_wait_ref': r.read(manifest['resource_amendment_ref'])['failed_cache_wait_ref'],
        'recorded_utc': datetime.now(timezone.utc).isoformat()})
    with (folder/'controller.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=r.ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    receipt = {'pid': child.pid, 'process_start_ticks': ticks, 'command': command,
               'manifest_ref': manifest_ref, 'review_ref': review_ref, 'launcher_ref': r.ref(__file__),
               'started_utc': datetime.now(timezone.utc).isoformat(), 'workers_signalled': 0}
    save(RECORDS/'root_launch.json', receipt)
    print(json.dumps(receipt), flush=True)
    time.sleep(2)
    assert child.poll() is None, 'Recovery exited at startup; inspect the saved log'


if __name__ == '__main__':
    main()

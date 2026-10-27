"""Launch the independently reviewed CPU-only validation successor once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .contracts import ROOT, AGGREGATE, ref, read, verify_sources, validate_release, retained_attempts
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity

MANIFEST_SHA = '012ac6d970d85845aec119ed74ec12b89d630c7dca7ee1d5bf8c7352e0069d0e'
OTHER_RESERVED = 21072183296
HERE = ROOT / 'research/resume_no_timeouts_20261027/fullval'


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--review-path', required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    review_ref = dict(path=args.review_path, sha256=args.review_sha256)
    manifest_ref, _ = verify_sources(MANIFEST_SHA, review_ref)
    retained_attempts()
    assert not AGGREGATE.exists(), 'Never duplicate an existing run'
    launch_path = HERE / 'ROOT_GUARDIAN_LAUNCH.json'
    assert not launch_path.exists()
    profiles = []
    for directory in sorted(AGGREGATE.parent.iterdir()):
        request_path = directory / 'request.json'
        if not request_path.is_file():
            continue
        request = json.loads(request_path.read_text())
        terminal = json.loads((directory / 'actual_wait.json').read_text())
        assert terminal['actual_wait'] and terminal['owned_group_absent'], str(directory)
        if request.get('supervisor_pid'):
            assert identity(request['supervisor_pid']) != str(request['supervisor_start_ticks']), str(directory)
        profiles.append(str(directory.relative_to(ROOT)))
    state = json.loads((ROOT / '.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json').read_text())
    assert time.time() - state['updated_unix'] < 90
    release = json.loads((HERE / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    release.update(passed=True, execution_authorized=True,
        independent_source_review_ref=review_ref, other_project_reserved_bytes=OTHER_RESERVED,
        peer_directories=[], prepared_unix=time.time(), terminal_profiles_checked=profiles,
        initial_campaign_reserved_bytes=state['project_reserved_bytes'])
    assert state['project_reserved_bytes'] + OTHER_RESERVED + release['reservation_bytes'] <= 1195000000000
    validate_release(release, manifest_ref, review_ref, [])
    release_path = HERE / 'ROOT_EXECUTION_RELEASE.json'
    write_new(release_path, release)
    release_ref = ref(release_path)
    command = ['.runtime/env/bin/python3.11', '-B', '-m',
        'research.resume_no_timeouts_20261027.fullval.guardian',
        '--source-pins-sha256', MANIFEST_SHA,
        '--source-review-path', review_ref['path'], '--source-review-sha256', review_ref['sha256'],
        '--execution-release-path', release_ref['path'], '--execution-release-sha256', release_ref['sha256']]
    environment = dict(os.environ, PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    for key in ('ARGGNN_PHYSICAL_GPU', 'ARGGNN_READABLE_GPU_FD', 'ARGGNN_MAX_CUDA_RESERVED_BYTES', 'ARGGNN_GPU_MEMORY_POLICY'):
        environment.pop(key, None)
    log_path = HERE / 'guardian.stdout.log'
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record = dict(pid=process.pid, process_start_ticks=identity(process.pid), actual_argv=command,
        started_unix=time.time(), source_manifest_ref=manifest_ref, source_review_ref=review_ref,
        release_ref=release_ref, launcher_ref=ref(__file__), log_path=str(log_path.relative_to(ROOT)))
    write_new(launch_path, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

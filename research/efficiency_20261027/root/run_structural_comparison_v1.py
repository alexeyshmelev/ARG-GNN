"""One CPU-only structural comparison, admitted alongside the live campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil
from research.efficiency_20261027.root.run_guarded_profile_with_peer import (
    ROOT, campaign_growth, peer_growth, current_memory, identity, write)
from arggnn.benchmarking.paper_campaign.processes import drain
from research.efficiency_20261027.execution.structural_episode_comparison.comparator_v1 import file_ref, need

BASE = ROOT / 'research/efficiency_20261027'
CONTRACT = BASE / 'execution/structural_episode_comparison/ROOT_LAUNCH_CONTRACT_v1.json'
RUN = BASE / 'root/runs/structural_episode_train001_v1'
GIB = 2**30
HOST_CAP = 32*GIB
MARGIN = 64*GIB
OUTPUT_CAP = 16*2**20
# These existing aggregate allowances include this small diagnostic.
OTHER_COMMITMENTS = 16*GIB + 2*GIB + 512*2**20


def admission(peers, own_used=0):
    state, growth, rows = campaign_growth()
    for directory in peers:
        amount, detail = peer_growth(directory)
        growth += amount
        rows.append(detail)
    available = current_memory()
    required = HOST_CAP + growth + MARGIN
    total = state['project_reserved_bytes'] + OTHER_COMMITMENTS
    need(total <= 995_000_000_000, 'Existing project reservations exceed admission ceiling')
    need(available + own_used >= required, 'Current host capacity does not cover committed growth')
    return dict(observed_unix=time.time(), host_available_bytes=available,
        required_host_bytes=required, live_growth_bytes=growth, peers=rows,
        total_committed_storage_bytes=total, output_in_existing_512MiB_allowance=True)


def verify_review(path, digest, manifest, contract_ref):
    need(file_ref(path)['sha256'] == digest, 'Exact independent source review')
    review = json.loads((ROOT / path).read_text())
    need(review.get('passed') is True and review.get('source_manifest_ref') == manifest
         and review.get('launch_contract_ref') == contract_ref,
         'Independent review must bind this exact comparator manifest and launch contract')
    return file_ref(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--review-path', required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    need(Path.cwd() == ROOT, 'Explicit project directory required')
    raw_contract = CONTRACT.read_bytes()
    contract_ref = dict(path=str(CONTRACT.relative_to(ROOT)), sha256=hashlib.sha256(raw_contract).hexdigest())
    contract = json.loads(raw_contract)
    manifest = contract['source_manifest_ref']
    need(file_ref(manifest['path']) == manifest, 'Sealed comparator sources')
    review = verify_review(args.review_path, args.review_sha256, manifest, contract_ref)
    peers = contract['ROOT_admission_fields']['peer_directories']
    snapshot = admission(peers)
    need(not RUN.exists(), 'One fresh diagnostic only, never retry over historical output')
    RUN.mkdir()
    admission_path = RUN / 'admission.json'
    admitted = dict(contract['ROOT_admission_fields'],
        schema=contract['ROOT_admission_schema'], observed_unix=time.time(),
        independent_review_ref=review, resource_snapshot=snapshot)
    write(admission_path, admitted)
    command = ['.runtime/env/bin/python3.11', '-B', '-m', contract['module'],
        '--source-sha256', manifest['sha256'], '--admission', str(admission_path.relative_to(ROOT)),
        '--output', str((RUN/'data').relative_to(ROOT))]
    write(RUN/'request.json', dict(command=command, host_cap_bytes=HOST_CAP,
        output_cap_bytes=OUTPUT_CAP, peer_directories=peers,
        supervisor_pid=os.getpid(), supervisor_start_ticks=identity(os.getpid()),
        CUDA_VISIBLE_DEVICES='', created_unix=time.time(),
        guardian_source_ref=file_ref(Path(__file__)), contract_ref=contract_ref,
        comparator_review_ref=review, output_in_existing_512MiB_allowance=True))
    env = dict(os.environ, **contract['environment'])
    for key in ('ARGGNN_PHYSICAL_GPU', 'ARGGNN_READABLE_GPU_FD', 'ARGGNN_MAX_CUDA_RESERVED_BYTES', 'ARGGNN_GPU_MEMORY_POLICY'):
        env.pop(key, None)
    child = None
    ticks = None
    peak = 0
    started = time.monotonic()
    next_growth = started + 60
    def cancelled(signum, frame):
        raise KeyboardInterrupt('Own diagnostic guardian interrupted')
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    with (RUN/'stdout.log').open('xb') as log:
        try:
            need(file_ref(CONTRACT) == contract_ref, 'Launch contract unchanged at dispatch')
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            ticks = identity(child.pid)
            need(ticks is not None, 'Own newly created child identity')
            write(RUN/'launch.json', dict(pid=child.pid, process_start_ticks=ticks,
                actual_argv=command, started_unix=time.time(), admitted_snapshot=snapshot))
            while child.poll() is None:
                need(identity(child.pid) == ticks, 'Own child identity remains fixed')
                process = psutil.Process(child.pid)
                need(not process.children(recursive=True), 'Reviewed comparator creates no subprocesses')
                used = process.memory_info().rss + psutil.Process().memory_info().rss
                peak = max(peak, used)
                elapsed = time.monotonic() - started
                need(used < HOST_CAP and current_memory() >= MARGIN, 'Diagnostic host memory bound')
                need(elapsed < contract['maximum_seconds'] + 60, 'Diagnostic finite wall time')
                output = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
                need(output < OUTPUT_CAP, 'Diagnostic output bound')
                if time.monotonic() >= next_growth:
                    snapshot = admission(peers, used)
                    next_growth = time.monotonic() + 60
                write(RUN/'status.json', dict(status='running', pid=child.pid,
                    process_start_ticks=ticks, observed_unix=time.time(), elapsed_seconds=elapsed,
                    recursive_family_RSS_bytes=used, peak_sampled_family_RSS_bytes=peak,
                    last_admission_snapshot=snapshot))
                time.sleep(2)
            code = child.wait()
            absent = drain(child)
            need(absent, 'No child session survives completed comparator')
            write(RUN/'actual_wait.json', dict(actual_wait=True, actual_returncode=code,
                pid=child.pid, process_start_ticks=ticks, owned_group_absent=True,
                finished_unix=time.time(), peak_sampled_family_RSS_bytes=peak))
            need(code == 0, 'Comparator exited unsuccessfully')
            result = json.loads((RUN/'data/completed.json').read_text())
            need(result.get('status') == 'passed', 'Complete scientific comparison must pass')
            write(RUN/'status.json', dict(status='finished', observed_unix=time.time(),
                actual_returncode=0, completed_ref=file_ref(RUN/'data/completed.json')))
        except BaseException as error:
            if child is not None:
                drain(child, terminate=True)
                code = child.wait()
                if not (RUN/'actual_wait.json').exists():
                    write(RUN/'actual_wait.json', dict(actual_wait=True, actual_returncode=code,
                        pid=child.pid, process_start_ticks=ticks, owned_group_absent=True,
                        finished_unix=time.time(), peak_sampled_family_RSS_bytes=peak))
            write(RUN/'failure.json', dict(error=repr(error), observed_unix=time.time(),
                historical_evidence_retained=True))
            raise


if __name__ == '__main__':
    main()

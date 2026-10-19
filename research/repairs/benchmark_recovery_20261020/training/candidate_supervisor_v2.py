"""Versioned direct-child owner, genuine waits and shared-GPU free-memory floor.

Only the Popen object created here is signalled. The scientific worker does not
spawn scientific subprocesses. Historical pidfd supervisor sources stay frozen.
"""
from pathlib import Path
import argparse
import os
import signal
import subprocess
import sys
import time
from .candidate_common import (
    ROOT, GIB, need, load_request, identity, rss_bytes, scope_bytes, write_json,
    continuation_allocation, gpu_inventory, file_ref,
)


def terminate_owned(child, timeout=15):
    """Terminate and reap this exact direct child through its Popen owner."""
    if child.poll() is None:
        child.terminate()
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
    return child.wait(timeout=timeout)


def owned_child_absent(observed):
    if observed is None:
        return None
    try:
        return identity(observed['pid']) != observed
    except (FileNotFoundError, ProcessLookupError):
        return True


def run_owned(command, log_path, check, poll_seconds=1.):
    """Own one child until an actual wait; convert monitor errors to failure."""
    child, observed, violation = None, None, None
    started = time.monotonic()
    with log_path.open('xb') as log:
        try:
            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            observed = identity(child.pid)
            while child.poll() is None:
                try:
                    current = identity(child.pid)
                except (FileNotFoundError, ProcessLookupError):
                    if child.poll() is not None:
                        break
                    raise
                need(current == observed, 'Owned direct-child identity changed')
                check(observed, time.monotonic()-started)
                time.sleep(poll_seconds)
            status = child.wait()
        except BaseException as error:
            violation = type(error).__name__+': '+str(error)
            if child is None:
                raise
            # Further termination signals must not interrupt direct-child cleanup.
            previous = {number:signal.signal(number,signal.SIG_IGN)
                        for number in (signal.SIGTERM,signal.SIGINT)}
            try:
                status = terminate_owned(child)
            finally:
                for number,handler in previous.items():
                    signal.signal(number,handler)
        finally:
            if child is not None and child.poll() is None:
                terminate_owned(child)
    return dict(worker_identity=observed,actual_exit_code=status,
        resource_violation=violation,owned_child_absent=owned_child_absent(observed),
        elapsed_seconds=time.monotonic()-started,lifecycle='direct_owned_Popen_wait',
        automatic_retry=False)


class SharedGPUFloor:
    """Three consecutive low samples, at least two seconds apart."""
    def __init__(self, gpu_uuid, minimum_bytes):
        self.gpu_uuid, self.minimum_bytes = gpu_uuid, minimum_bytes
        self.last_sample, self.low_samples = float('-inf'), 0
        self.samples, self.minimum_observed = 0, None

    def check(self):
        now = time.monotonic()
        if now-self.last_sample < 2.:
            return
        self.last_sample = now
        free = gpu_inventory(self.gpu_uuid)['free_bytes']
        self.samples += 1
        self.minimum_observed = free if self.minimum_observed is None else min(free,self.minimum_observed)
        self.low_samples = self.low_samples+1 if free < self.minimum_bytes else 0
        need(self.low_samples < 3,
             'Shared GPU free memory below floor for three consecutive samples')

    def receipt(self):
        return dict(floor_bytes=self.minimum_bytes,sampling_seconds=2.,
            consecutive_low_samples=self.low_samples,samples=self.samples,
            minimum_observed_free_bytes=self.minimum_observed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    request,request_ref,job,barrier,output = load_request(args.request)
    need(Path.cwd() == ROOT, 'Project working directory required')
    need(Path(sys.executable).resolve() == (ROOT/'.runtime/env/bin/python').resolve(),
         'Pinned project Python required')
    need(file_ref(Path(__file__).resolve()) in request['source_refs'],
         'Exact version 2 supervisor source reference required')
    floor = request.get('shared_gpu_free_floor_bytes')
    need(type(floor) is int and floor == 8*GIB, 'Explicit shared GPU 8 GiB running floor required')
    need(request['minimum_free_gpu_bytes'] >= 26*GIB, 'Initial shared GPU admission floor is 26 GiB')
    need(not output.exists(), 'Fresh output directory required; no automatic retry')
    if args.validate_only:
        print('Validated version 2 candidate request; no child launched and no CUDA used.',flush=True)
        return
    output.mkdir(parents=True)
    (output/'checkpoints').mkdir()
    owner = identity()
    command = [str(ROOT/'.runtime/env/bin/python'),'-B','-m',
        'research.repairs.benchmark_recovery_20261020.training.candidate_worker',
        '--request',str(ROOT/request_ref['path']),'--request-sha256',request_ref['sha256'],
        '--supervisor-pid',str(owner['pid']),'--supervisor-start-ticks',str(owner['start_ticks'])]
    write_json(output/'launch.json',dict(request_ref=request_ref,supervisor_identity=owner,
        command=command,source_refs=request['source_refs'],automatic_retries=False,
        lifecycle='direct_owned_Popen_wait',pidfd_open_available=hasattr(os,'pidfd_open'),
        pidfd_send_signal_available=hasattr(signal,'pidfd_send_signal')))
    def interrupted(signum, frame):
        raise RuntimeError('Supervisor received signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    active = dict(rss_cap=request['max_rss_bytes'],continuation_ref=None,identity_written=False)
    gpu_floor = SharedGPUFloor(request['gpu_uuid'],floor)
    def check(observed, elapsed):
        if not active['identity_written']:
            write_json(output/'child_identity.json',dict(worker_identity=observed,supervisor_identity=owner))
            active['identity_written'] = True
        need(elapsed <= request['max_seconds'], 'Candidate wall cap exceeded')
        if active['continuation_ref'] is None and (output/'continue_gate.json').exists():
            allocation = continuation_allocation(output,request_ref,observed)
            active['rss_cap'] = allocation['max_rss_bytes']
            active['continuation_ref'] = allocation['cpu_admission_ref']
        need(rss_bytes(observed['pid']) <= active['rss_cap'], 'Candidate RSS cap exceeded')
        need(scope_bytes(output) <= request['max_output_bytes'], 'Candidate output cap exceeded')
        if (output/'gpu_gate.json').exists():
            gpu_floor.check()
    result = run_owned(command,output/'stage.log',check)
    success_file = 'preflight_passed.json' if request['phase'] == 'preflight' else 'training_completed.json'
    success = (result['actual_exit_code'] == 0 and result['resource_violation'] is None
               and result['owned_child_absent'] is True and (output/success_file).is_file())
    write_json(output/'supervisor_result.json',dict(result,
        schema='treeloom_candidate_actual_wait_v1',request_ref=request_ref,supervisor_identity=owner,
        success=success,output_scope_bytes=scope_bytes(output),active_max_rss_bytes=active['rss_cap'],
        continuation_CPU_admission_ref=active['continuation_ref'],shared_gpu_floor=gpu_floor.receipt()))
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()

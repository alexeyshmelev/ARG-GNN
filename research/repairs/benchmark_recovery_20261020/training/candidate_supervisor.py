"""Direct child owner with finite RSS/output/wall caps and genuine OS wait."""
from pathlib import Path
import argparse
import os
import signal
import subprocess
import sys
import time
from .candidate_common import ROOT, need, load_request, identity, rss_bytes, scope_bytes, write_json, continuation_allocation


def terminate_owned(child, descriptor):
    if child.poll() is not None:
        return
    signal.pidfd_send_signal(descriptor,signal.SIGTERM,None,0)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        signal.pidfd_send_signal(descriptor,signal.SIGKILL,None,0)
        child.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    request,request_ref,job,barrier,output = load_request(args.request)
    need(Path.cwd() == ROOT, "Project working directory required")
    need(Path(sys.executable).resolve() == (ROOT/'.runtime/env/bin/python').resolve(),
         "Pinned project Python required")
    if args.validate_only:
        need(not output.exists(), "Output must be fresh for a future launch")
        print('Validated candidate request; no process launched and no CUDA used.',flush=True)
        return
    need(not output.exists(), "Fresh output directory required; no automatic retry")
    output.mkdir(parents=True)
    (output/'checkpoints').mkdir()
    owner = identity()
    command = [str(ROOT/'.runtime/env/bin/python'),'-B','-m',
        'research.repairs.benchmark_recovery_20261020.training.candidate_worker',
        '--request',str(ROOT/request_ref['path']),'--request-sha256',request_ref['sha256'],
        '--supervisor-pid',str(owner['pid']),'--supervisor-start-ticks',str(owner['start_ticks'])]
    write_json(output/'launch.json',dict(request_ref=request_ref,supervisor_identity=owner,
        command=command,source_refs=request['source_refs'],automatic_retries=False))
    def interrupted(signum, frame):
        raise RuntimeError('Supervisor received signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    started = time.monotonic()
    child = None
    descriptor = None
    violation = None
    try:
        with (output/'stage.log').open('xb') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            descriptor = os.pidfd_open(child.pid,0)
            observed = identity(child.pid)
            write_json(output/'child_identity.json',dict(worker_identity=observed,supervisor_identity=owner))
            active_rss_cap = request['max_rss_bytes']
            continuation_ref = None
            while child.poll() is None:
                try:
                    need(identity(child.pid) == observed, "Owned child identity changed")
                    need(time.monotonic()-started <= request['max_seconds'], "Candidate wall cap exceeded")
                    if continuation_ref is None and (output/'continue_gate.json').exists():
                        allocation = continuation_allocation(output,request_ref,observed)
                        active_rss_cap = allocation['max_rss_bytes']
                        continuation_ref = allocation['cpu_admission_ref']
                    need(rss_bytes(child.pid) <= active_rss_cap, "Candidate RSS cap exceeded")
                    need(scope_bytes(output) <= request['max_output_bytes'], "Candidate output cap exceeded")
                except Exception as error:
                    violation = type(error).__name__+': '+str(error)
                    terminate_owned(child,descriptor)
                    break
                time.sleep(1.)
            status = child.wait()
        success_file = 'preflight_passed.json' if request['phase'] == 'preflight' else 'training_completed.json'
        success = status == 0 and violation is None and (output/success_file).is_file()
        write_json(output/'supervisor_result.json',dict(schema='treeloom_candidate_actual_wait_v1',
            request_ref=request_ref,supervisor_identity=owner,worker_identity=observed,
            actual_exit_code=status,resource_violation=violation,success=success,
            elapsed_seconds=time.monotonic()-started,output_scope_bytes=scope_bytes(output),
            active_max_rss_bytes=active_rss_cap,continuation_CPU_admission_ref=continuation_ref,
            automatic_retry=False))
        raise SystemExit(0 if success else 1)
    except Exception as error:
        if child is not None and descriptor is not None:
            terminate_owned(child,descriptor)
            status = child.wait()
            if not (output/'supervisor_result.json').exists():
                write_json(output/'supervisor_result.json',dict(schema='treeloom_candidate_actual_wait_v1',
                    request_ref=request_ref,supervisor_identity=owner,worker_identity=observed,
                    actual_exit_code=status,success=False,supervisor_error=type(error).__name__+': '+str(error),
                    elapsed_seconds=time.monotonic()-started,automatic_retry=False))
        raise
    finally:
        if child is not None and descriptor is not None and child.poll() is None:
            terminate_owned(child,descriptor)
        if descriptor is not None:
            os.close(descriptor)


if __name__ == '__main__':
    main()

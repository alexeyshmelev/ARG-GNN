"""ROOT-owned one-shot CUDA compatibility launch with a retained GPU0 reservation."""
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import time
from .candidate_common import ROOT, load_request, write_json, read_json, identity
from ..scheduler.shared_gpu0_v1 import gpu_sample

REQUEST = 'research/repairs/benchmark_recovery_20261020/training/preflight_request_v1.json'


def main():
    request, request_ref, job, barrier, output = load_request(REQUEST)
    lease_path = ROOT/'.runtime/locks/gpu_0.lock'
    with lease_path.open('r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sample = gpu_sample()
        if sample['free_bytes'] < request['minimum_free_gpu_bytes']:
            raise RuntimeError('Actual free GPU0 memory cannot admit the16GiB preflight')
        command = [str(ROOT/'.runtime/env/bin/python'), '-B', '-s', '-m',
                   'research.repairs.benchmark_recovery_20261020.training.candidate_supervisor', '--request', REQUEST]
        log_dir = ROOT/'.runtime/benchmarking/recovery_20261020_treeloom_preflight'
        log_dir.mkdir(exist_ok=False)
        with (log_dir/'supervisor.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            descriptor = os.pidfd_open(child.pid)
            def stop(number, frame):
                signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
                child.wait(timeout=30)
                raise SystemExit(1)
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            write_json(log_dir/'launch.json', dict(request_ref=request_ref, command=command,
                       supervisor_identity=identity(child.pid), reservation_owner=identity(), gpu_sample=sample))
            try:
                deadline = time.monotonic()+120
                ready_path = output/'ready_for_gpu.json'
                while child.poll() is None and not ready_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.5)
                if not ready_path.exists():
                    raise RuntimeError('Candidate did not reach its explicit GPU gate')
                ready, _ = read_json(ready_path)
                admission = dict(request_ref=request_ref, worker_identity=ready['worker_identity'],
                                 physical_gpu=0, gpu_uuid=request['gpu_uuid'],
                                 max_cuda_reserved_bytes=request['max_cuda_reserved_bytes'],
                                 reservation_owner=identity(), actual_free_sample=gpu_sample(),
                                 canonical_lease_held_until_supervisor_exit=True)
                admission_ref = write_json(output/'root_gpu_admission.json', admission)
                write_json(output/'gpu_gate.json', dict(schema='treeloom_candidate_gate_v1', gate='gpu_gate.json',
                           allow=True, request_ref=request_ref, worker_identity=ready['worker_identity'],
                           gpu_uuid=request['gpu_uuid'], max_cuda_reserved_bytes=request['max_cuda_reserved_bytes'],
                           expires_unix=time.time()+120, admission_ref=admission_ref))
                code = child.wait()
                write_json(log_dir/'launcher_wait.json', dict(actual_wait=True, actual_returncode=code,
                           request_ref=request_ref, supervisor_identity=identity() if child.pid == os.getpid() else None))
                raise SystemExit(code)
            finally:
                if child.poll() is None:
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
                    child.wait(timeout=30)
                os.close(descriptor)


if __name__ == '__main__':
    main()

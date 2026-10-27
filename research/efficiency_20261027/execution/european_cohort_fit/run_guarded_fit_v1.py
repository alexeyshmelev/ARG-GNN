"""Guard one isolated fixed90 FIT with an explicit2GiB whole-run allowance.

This is a cooperative opportunistic admission, not a new campaign reservation.
Only the profile child belongs to this supervisor. Existing jobs are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
STATE = ROOT / '.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json'
UUIDS = {
    3: 'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',
    4: 'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',
    5: 'GPU-3db85505-d061-e45d-d154-e7e8c1cad497',
}
GIB = 2**30


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def identity(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def current_memory():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def campaign_growth():
    """Fresh PSS and verified PID identity, no duplicate legacy sessions."""
    from arggnn.benchmarking.paper_campaign.processes import pss
    state = json.loads(STATE.read_text())
    if time.time() - state['updated_unix'] > 90:
        raise RuntimeError('Campaign snapshot is stale')
    plan = json.loads((ROOT / state['plan_ref']['path']).read_text())
    growth, seen, details = 0, set(), []
    for task_id, row in state['tasks'].items():
        if row['status'] != 'running':
            continue
        pid = row['pid']
        if identity(pid) != row['process_start_ticks']:
            raise RuntimeError('Campaign worker boundary changed, retry admission')
        suffix = '' if row.get('attempt', 1) == 1 else '.attempt' + str(row['attempt'])
        spec_path = ROOT / plan['records_directory'] / 'jobs' / (task_id + suffix + '.json')
        if not spec_path.resolve().is_relative_to(ROOT):
            raise RuntimeError('Campaign job spec must be contained')
        spec = json.loads(spec_path.read_text())
        if spec['id'] != task_id:
            raise RuntimeError('Campaign job identity changed')
        used = pss(pid)
        amount = max(0, spec['max_rss_bytes'] - used)
        growth += amount
        seen.add(pid)
        details.append(dict(task=task_id, pid=pid, observed_pss_bytes=used,
                            max_rss_bytes=spec['max_rss_bytes'], remaining_growth_bytes=amount))
    for row in plan.get('existing_workers', []):
        pid = row['pid']
        if pid in seen or identity(pid) != row['start_ticks']:
            continue
        used = pss(pid)
        amount = max(0, row['max_rss_bytes'] - used)
        growth += amount
        details.append(dict(task=row['task_id'], pid=pid, observed_pss_bytes=used,
                            max_rss_bytes=row['max_rss_bytes'], remaining_growth_bytes=amount))
    return state, growth, details


def peer_growth(directory):
    """Reserve remaining host growth of another explicitly named own profile."""
    from arggnn.benchmarking.paper_campaign.processes import pss
    directory = (ROOT / directory).resolve()
    if not directory.is_relative_to(ROOT / 'research/efficiency_20261027/root/runs'):
        raise ValueError('Peer must be an explicitly named own efficiency profile')
    request = json.loads((directory / 'request.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    launch_path = directory / 'launch.json'
    if (directory / 'actual_wait.json').exists():
        wait = json.loads((directory / 'actual_wait.json').read_text())
        if wait.get('actual_wait') is not True or wait.get('owned_group_absent') is not True:
            raise RuntimeError('Peer has no confirmed process-session completion')
        return 0, dict(peer_directory=str(directory.relative_to(ROOT)), status='finished', remaining_growth_bytes=0)
    if (directory / 'failure.json').exists() and not launch_path.exists():
        return 0, dict(peer_directory=str(directory.relative_to(ROOT)), status='failed_before_launch', remaining_growth_bytes=0)
    cap = request['host_cap_bytes']
    if type(cap) is not int or cap <= 0:
        raise RuntimeError('Peer must have a positive declared host reservation')
    if not launch_path.exists():
        return cap, dict(peer_directory=str(directory.relative_to(ROOT)), status='waiting', remaining_growth_bytes=cap)
    launch = json.loads(launch_path.read_text())
    pid = launch['pid']
    if identity(pid) != launch['process_start_ticks']:
        raise RuntimeError('Peer identity changed before its exact completion receipt')
    if time.time() - status['observed_unix'] > 120:
        raise RuntimeError('Peer supervisor status is stale')
    used = pss(pid)
    remaining = max(0, cap - used)
    return remaining, dict(peer_directory=str(directory.relative_to(ROOT)), status='running',
        pid=pid, host_cap_bytes=cap, observed_pss_bytes=used, remaining_growth_bytes=remaining)


def gpu_free(index):
    raw = subprocess.check_output([
        'nvidia-smi', '--id=' + str(index), '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    uuid, mib = (x.strip() for x in raw.strip().split(','))
    if uuid != UUIDS[index]:
        raise RuntimeError('Physical GPU UUID changed')
    return int(mib) * 2**20


def child_gpu_bytes(pid, index):
    raw = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    total = 0
    for line in raw.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 3 and fields[0] == str(pid) and fields[1] == UUIDS[index]:
            total += int(fields[2]) * 2**20
    return total


def scope_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def stop_child(process, start_ticks):
    """Drain only this freshly created process session, never a benchmark worker."""
    from arggnn.benchmarking.paper_campaign.processes import drain
    if process.poll() is None and identity(process.pid) != start_ticks:
        raise RuntimeError('Owned profile identity changed')
    drain(process, terminate=True)
    return process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--peer-directory', action='append', default=[])
    parser.add_argument('--gpu', type=int, choices=(3, 4, 5), default=5)
    parser.add_argument('--host-gib', type=float, required=True)
    parser.add_argument('--device-gib', type=float, required=True)
    parser.add_argument('--wait-seconds', type=int, default=86400)
    parser.add_argument('--run-seconds', type=int, default=14400)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or Path(command[0]).resolve() != (ROOT / '.runtime/env/bin/python').resolve():
        raise ValueError('Use the visible project Python argv')
    directory = (ROOT / args.directory).resolve()
    if directory.parent != ROOT / 'research/efficiency_20261027/root/runs':
        raise ValueError('FIT output must be one fresh ROOT run directory')
    directory.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    cap = int(args.host_gib * GIB)
    device_cap = int(args.device_gib * GIB)
    output_cap = 2 * GIB
    guard = 64 * GIB
    start = time.monotonic()
    request = dict(command=command, host_cap_bytes=cap, device_budget_bytes=device_cap,
                   output_cap_bytes=output_cap, physical_gpu=args.gpu, gpu_uuid=UUIDS[args.gpu],
                   peer_directories=args.peer_directory,
                   supervisor_pid=os.getpid(), supervisor_start_ticks=identity(os.getpid()),
                   admitted_as='opportunistic_profile_not_campaign_reservation',
                   created_unix=time.time())
    write(directory / 'request.json', request)
    def cancelled(signum, frame):
        raise KeyboardInterrupt('Profile supervisor received signal ' + str(signum))
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    child = None
    child_ticks = None
    log = None
    try:
        while True:
            reason = None
            try:
                state, growth, details = campaign_growth()
                for peer in args.peer_directory:
                    amount, detail = peer_growth(peer)
                    growth += amount
                    details.append(detail)
                available = current_memory()
                free_gpu = gpu_free(args.gpu)
                if available < cap + growth + guard:
                    reason = 'Waiting for host RAM after existing worker growth allowances'
                elif free_gpu < device_cap + 2*GIB:
                    reason = 'Waiting for GPU memory including context margin'
                elif state['project_reserved_bytes'] + output_cap > 995_000_000_000:
                    reason = 'Waiting for project output allowance'
                snapshot = dict(observed_unix=time.time(), host_available_bytes=available,
                                live_growth_bytes=growth, workers=details, gpu_free_bytes=free_gpu,
                                required_host_bytes=cap+growth+guard)
            except (RuntimeError, FileNotFoundError) as exc:
                reason = str(exc)
                snapshot = dict(observed_unix=time.time())
            write(directory / 'status.json', dict(status='waiting' if reason else 'admitted',
                                                  reason=reason, **snapshot))
            if reason is None:
                break
            if time.monotonic() - start > args.wait_seconds:
                raise TimeoutError('Profile admission wait expired: ' + reason)
            time.sleep(30)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUIDS[args.gpu],
                   ARGGNN_PHYSICAL_GPU=str(args.gpu), ARGGNN_GPU_MEMORY_POLICY='full_device',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        log = (directory / 'stdout.log').open('xb')
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        child_ticks = identity(child.pid)
        run_start = time.monotonic()
        write(directory / 'launch.json', dict(pid=child.pid, process_start_ticks=child_ticks,
                                             actual_argv=command, admitted_snapshot=snapshot,
                                             started_unix=time.time()))
        from arggnn.benchmarking.paper_campaign.processes import pss
        peak = 0
        growth_errors = 0
        while child.poll() is None:
            used = pss(child.pid)
            peak = max(peak, used)
            reason = None
            gpu_used = child_gpu_bytes(child.pid, args.gpu)
            try:
                _, growth, _ = campaign_growth()
                for peer in args.peer_directory:
                    amount, detail = peer_growth(peer)
                    growth += amount
                growth_errors = 0
                if current_memory() + used < cap + growth + guard:
                    reason = 'New campaign growth needs capacity, yielding own profile'
            except (RuntimeError, FileNotFoundError) as exc:
                growth_errors += 1
                if growth_errors >= 3:
                    reason = 'Cannot verify campaign growth after three samples: ' + str(exc)
            if used > cap:
                reason = 'Profile host cap exceeded'
            elif gpu_used > device_cap + GIB:
                reason = 'Profile GPU budget plus CUDA context margin exceeded'
            elif gpu_free(args.gpu) < 2*GIB:
                reason = 'GPU pressure, yielding profile to benchmark jobs'
            elif current_memory() < guard:
                reason = 'Host pressure, yielding profile to benchmark jobs'
            elif time.monotonic() - run_start > args.run_seconds:
                reason = 'Profile wall limit exceeded'
            elif scope_bytes(directory) > output_cap:
                reason = 'Profile output cap exceeded'
            if reason:
                code = stop_child(child, child_ticks)
                raise RuntimeError(reason + ', own child return code ' + str(code))
            write(directory / 'status.json', dict(status='running', pid=child.pid,
                  process_start_ticks=child_ticks, observed_unix=time.time(),
                  elapsed_seconds=time.monotonic()-run_start, pss_bytes=used, peak_sampled_pss_bytes=peak,
                  sampled_gpu_process_bytes=gpu_used))
            time.sleep(5)
        code = child.wait()
        from arggnn.benchmarking.paper_campaign.processes import drain
        group_absent = drain(child)
        if not group_absent:
            stop_child(child, child_ticks)
        write(directory / 'actual_wait.json', dict(actual_returncode=code, actual_wait=True,
              pid=child.pid, process_start_ticks=child_ticks, finished_unix=time.time(),
              peak_sampled_pss_bytes=peak, owned_group_absent=True,
              surviving_descendants_required_drain=not group_absent))
        if not group_absent:
            raise RuntimeError('Profile leader exited with surviving descendants, own session drained')
        write(directory / 'status.json', dict(status='finished' if code == 0 else 'failed',
                                             actual_returncode=code, observed_unix=time.time()))
        if code:
            raise RuntimeError('Profile exited with code ' + str(code))
    except BaseException as exc:
        if child is not None:
            code = stop_child(child, child_ticks)
            if not (directory / 'actual_wait.json').exists():
                write(directory / 'actual_wait.json', dict(actual_returncode=code, actual_wait=True,
                      pid=child.pid, process_start_ticks=child_ticks, finished_unix=time.time(),
                      owned_group_absent=True))
        write(directory / 'failure.json', dict(error=str(exc), observed_unix=time.time()))
        raise
    finally:
        if log is not None:
            log.close()


if __name__ == '__main__':
    main()

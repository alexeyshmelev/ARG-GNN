"""Root-owned finite queue for two reviewed concurrent biological fits."""
from pathlib import Path
import argparse, datetime as dt, hashlib, json, os, signal, subprocess, time
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
PYTHON = str(ROOT / '.runtime/env/bin/python')
AUTH = ROOT / 'research/reviews/gpu3_parallel_fits_v2_root/pool_admission.json'
OUTPUT = ROOT / 'experiments/results/gpu3_parallel_fits_v2_root'
def write(name, value):
    p = OUTPUT / name
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, p)
def identity(pid):
    try:
        raw = Path('/proc') / str(pid) / 'stat'
        parts = raw.read_text().rsplit(') ', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(parts[19]), 'state': parts[0]}
    except FileNotFoundError:
        return None
def live(saved):
    current = identity(saved['pid'])
    return current is not None and current['start_ticks'] == saved['start_ticks'] and current['state'] != 'Z'
def command(action, bound, job=None):
    args = [PYTHON, '-B', '-m', 'arggnn.budgeted_shared_pool_v3', action]
    if job is not None:
        args.append(job)
    return args + ['--admission', str(AUTH.relative_to(ROOT)), '--expected-admission-sha256', bound,
                   '--expected-source-sha256', '2dd01da79462e0c469a1671bd509f0f0ce0b7ac700e63dcd4922db9fc9e9ad13',
                   '--expected-config-sha256', 'd0e3dba441ecad299ff61b53a262e38b767e1d78118c7d351c0f96888b077c28']
def call(args):
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(json.dumps({'argv': args, 'returncode': completed.returncode,
                                      'stdout': completed.stdout[-16000:], 'stderr': completed.stderr[-16000:]}))
    return json.loads(completed.stdout)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authority-sha256')
    parser.add_argument('--supervisor-sha256')
    parser.add_argument('--old-identities', required=True)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'source_only': True, 'GPU_or_lease_used': False}))
        return
    own = Path(__file__)
    if hashlib.sha256(own.read_bytes()).hexdigest() != args.supervisor_sha256:
        raise ValueError('Exact independently reviewed supervisor bytes required')
    if hashlib.sha256(AUTH.read_bytes()).hexdigest() != args.authority_sha256:
        raise ValueError('Exact activated independently reviewed pool authority required')
    authority = json.loads(AUTH.read_text())
    saved = json.loads(args.old_identities)
    expires = dt.datetime.fromisoformat(authority['expires_utc'])
    wait_deadline = min(time.monotonic() + 21600, time.monotonic() + max(0, (expires - dt.datetime.now(dt.timezone.utc)).total_seconds() - 14500))
    OUTPUT.mkdir(exist_ok=False)
    write('started.json', {'status': 'waiting_for_existing_GPU3_workers_to_finish',
                          'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                          'supervisor': identity(os.getpid()), 'old_identities': saved,
                          'actual_GPU_execution': False})
    controller = None
    try:
        while any(live(x) for x in saved):
            if time.monotonic() >= wait_deadline:
                raise TimeoutError('Existing GPU3 lease did not naturally release before bounded wait deadline')
            time.sleep(30)
        # The original controller separately acquires the canonical free lease and enforces physical cap2.
        # This observation is read-only. No old FD, lease or controller is adopted.
        from arggnn import gpu_pool_v1 as gpu
        inv = gpu.checked_inventory(gpu.real_inventory)
        write('prestart_inventory.json', {'device': inv[3], 'definitive_acquisition': 'original_controller'})
        valid = call(command('--validate-admission', args.authority_sha256))
        write('authority_validation.json', valid)
        with (OUTPUT / 'controller.stdout').open('w') as stdout, (OUTPUT / 'controller.stderr').open('w') as stderr:
            controller = subprocess.Popen(command('--start', args.authority_sha256), cwd=ROOT,
                                          stdout=stdout, stderr=stderr, start_new_session=True)
            write('controller_started.json', {'identity': identity(controller.pid),
                                            'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
            deadline = min(time.monotonic() + 82800, time.monotonic() + max(0, (expires - dt.datetime.now(dt.timezone.utc)).total_seconds() - 60))
            socket_path = ROOT / '.runtime/gpup/gpu3_fits_v2/control.sock'
            while not socket_path.exists():
                if controller.poll() is not None:
                    raise RuntimeError('Fresh controller failed acquisition, see bounded controller logs')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Fresh controller startup deadline exceeded')
                time.sleep(1)
            for job in authority['plan']['jobs']:
                result = call(command('--submit-job', args.authority_sha256, job['spec']['job']))
                write(job['spec']['job'] + '.submit.json', result)
            # Preserve queued jobs until all planned submissions really run.
            last = None
            while True:
                status = gpu.request(socket_path, {'v': 1, 'op': 'status'})
                concise = {'lifecycle': status['lifecycle'], 'safety_error': status.get('safety_error'),
                           'jobs': {k: {'state': v['state'], 'pid': v.get('pid'), 'exit_code': v.get('exit_code')}
                                    for v in status['jobs'] for k in [v['logical_job']]}}
                if concise != last:
                    write('progress.json', {**concise, 'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
                    last = concise
                # Bound observed memory for the exact controller-recorded worker identities.
                limits = {'mechanisms303-cache-v2': (49152, 128000000000),
                          'census-relate101-fit2-v2': (16384, 64000000000)}
                observed = status.get('memory_observation') or {}
                apps = observed.get('observed', {}).get('compute', [])
                bounds = []
                for job in status['jobs']:
                    if job['state'] not in ('launching', 'running', 'draining'):
                        continue
                    records = list(job.get('process_identities', []))
                    if job.get('leader_start_ticks') is not None and job.get('pid') is not None:
                        records.append({'pid': job['pid'], 'start_ticks': job['leader_start_ticks'], 'uid': os.getuid()})
                    unique = {p['pid']: p for p in records}
                    verified = []
                    rss = 0
                    for pid, saved_worker in unique.items():
                        try:
                            current = gpu.proc_identity(pid)
                            if (current['start_ticks'], current['uid']) != (saved_worker['start_ticks'], saved_worker['uid']):
                                continue
                            verified.append(saved_worker)
                            for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
                                if line.startswith('VmRSS:'):
                                    rss += int(line.split()[1]) * 1024
                                    break
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                    used = sum(app['used_mib'] for app in apps if app['pid'] in {p['pid'] for p in verified})
                    gpu_bound, cpu_bound = limits[job['logical_job']]
                    bound = {'job': job['logical_job'], 'GPU_mib': used, 'CPU_RSS_bytes': rss,
                             'GPU_limit_mib': gpu_bound, 'CPU_limit_bytes': cpu_bound}
                    bounds.append(bound)
                    if used > gpu_bound or rss > cpu_bound:
                        # pidfd plus UID/start-tick recheck signals only the offending registered worker.
                        sent = []
                        for worker in verified:
                            try:
                                fd = gpu.open_pidfd(worker['pid'])
                                try:
                                    current = gpu.proc_identity(worker['pid'])
                                    if (current['start_ticks'], current['uid']) == (worker['start_ticks'], worker['uid']):
                                        gpu.send_pidfd(fd, signal.SIGTERM)
                                        sent.append(worker['pid'])
                                finally:
                                    os.close(fd)
                            except (FileNotFoundError, ProcessLookupError):
                                continue
                        write(job['logical_job'] + '.memory_bound_failure.json', {**bound, 'signalled_owned_PIDs': sent})
                write('memory_bounds.json', {'observations': bounds, 'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                            'sampling_interval_seconds': 15, 'instantaneous_OS_quota_claimed': False})
                for log in (OUTPUT / 'controller.stdout', OUTPUT / 'controller.stderr'):
                    if log.stat().st_size > 1048576:
                        raise RuntimeError('Controller log exceeded root one-MiB bound')
                states = [v['state'] for v in status['jobs']]
                if len(states) == 2 and all(s in ('succeeded', 'failed', 'cancelled') for s in states):
                    write('shutdown.json', gpu.request(socket_path, {'v': 1, 'op': 'shutdown', 'terminate': False}))
                    break
                if time.monotonic() >= deadline:
                    write('watchdog_shutdown.json', gpu.request(socket_path, {'v': 1, 'op': 'shutdown', 'terminate': True}))
                    raise TimeoutError('Biological-fit watchdog requested termination of controller-owned workers')
                if controller.poll() is not None:
                    raise RuntimeError('Fresh controller exited unexpectedly')
                time.sleep(15)
            rc = controller.wait(timeout=120)
        write('exit.json', {'status': 'completed_biological_fit_queue', 'controller_returncode': rc,
                           'scientific_fits': True, 'query_truth_accessed': True, 'scope': 'TRAIN fit and development VAL only',
                           'jobs': last['jobs'], 'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    except BaseException as error:
        if controller is not None and controller.poll() is None:
            try:
                from arggnn import gpu_pool_v1 as gpu
                write('failure_shutdown.json', gpu.request(ROOT / '.runtime/gpup/gpu3_fits_v2/control.sock',
                                                         {'v': 1, 'op': 'shutdown', 'terminate': True}))
            except BaseException as shutdown_error:
                write('failure_shutdown_error.json', {'error': repr(shutdown_error)})
        write('exit.json', {'status': 'failed_partial_preserved', 'error': repr(error),
                           'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        raise
if __name__ == '__main__':
    main()

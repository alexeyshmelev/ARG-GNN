"""Root-owned ARGMix continuation with explicit admission, argv, and actual waits.

This controller launches only its registered new-model stages. It preserves
native producers, never adopts a prior running process, and publishes GPU-chain
completion before waiting for the native paired scorer.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import psutil
from arggnn.benchmarking.paper_campaign import processes
from .runtime import (ROOT, BASE, RESULTS, CACHE, MODEL_ID, ref, read, verify, require,
                      now, tree_bytes, frozen, cache_admission, require_wait)

RECORD_CAP = 32 * 2**20
HOST_ADMISSION_BYTES = 128 * 2**30


def atomic_record(filename, value, folder, *, immutable=False):
    filename = Path(filename)
    require(filename.is_relative_to(folder), 'Controller records remain in their registered scope')
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    require(tree_bytes(folder) + len(content) <= RECORD_CAP, 'Bounded32MiB controller record scope')
    filename.parent.mkdir(parents=True, exist_ok=True)
    if immutable:
        with filename.open('xb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
    else:
        temporary = filename.with_name(filename.name + '.tmp.' + str(os.getpid()))
        with temporary.open('xb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, filename)
    return ref(filename)


def owned_rss(pid):
    try:
        leader = psutil.Process(pid)
        family = [leader] + leader.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in family:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def stage(spec, command, environment, folder, plan_ref, freeze_ref, cache_limit, gpu_lock=None):
    directory = folder / spec['id']
    started = directory / 'started.json'
    receipt_path = directory / 'actual_wait.json'
    product_path = Path(spec['output_directory']) / spec['completion_file']
    if receipt_path.exists():
        receipt = read(ref(receipt_path))
        require(receipt['actual_wait'] is True and receipt['actual_returncode'] == 0 and
                receipt['owned_process_group_absent'] is True and receipt['status'] == 'succeeded' and
                receipt['command'] == command and receipt['plan_ref'] == plan_ref and
                receipt['freeze_ref'] == freeze_ref and not processes.session_members(receipt['pid']),
                'Only exact successful previously waited stage may be reused')
        verify(receipt['result_ref'])
        require_wait(product_path, spec['id'])
        return ref(receipt_path)
    require(not started.exists() and not product_path.exists(), 'Never silently replay or adopt a prior stage')
    frozen(freeze_ref['path'])
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update(environment)
    require(command[0] == str(ROOT / '.runtime/env/bin/python') and
            command[1:4] == ['-B', '-m', 'research.joint_evaluation_20261029.argmix.pipeline'] and
            all(isinstance(value, str) for value in command), 'Visible registered project Python module argv')
    begin = time.monotonic()
    failure = None
    with (directory / 'worker.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True,
                                 pass_fds=() if gpu_lock is None else (gpu_lock.fileno(),))
        ticks = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        receipt = dict(task_id=spec['id'], model_id=MODEL_ID, plan_ref=plan_ref, freeze_ref=freeze_ref,
                       pid=child.pid, process_start_ticks=ticks, command=command, started_utc=now())
        atomic_record(started, receipt, folder, immutable=True)
        while True:
            try:
                code = child.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - begin > spec['max_seconds']:
                    failure = 'Registered wall time exceeded'
                elif owned_rss(child.pid) + psutil.Process().memory_info().rss > spec['max_rss_bytes']:
                    failure = 'Registered owned64GiB RAM limit exceeded'
                elif tree_bytes(RESULTS) > 2**30 or tree_bytes(CACHE) > cache_limit:
                    failure = 'Registered output or cache scope limit exceeded'
                elif tree_bytes(folder) > RECORD_CAP:
                    failure = 'Registered controller metadata limit exceeded'
                if failure:
                    processes.drain(child, terminate=True)
                    code = child.wait()
                    break
        while not processes.drain(child):
            time.sleep(1)
    result_ref = None
    if code == 0 and failure is None:
        try:
            product = ref(product_path)
            result_ref = dict(product, bytes=product_path.stat().st_size)
        except (FileNotFoundError, ValueError) as error:
            failure = 'Missing expected stage product: ' + str(error)
    successful = code == 0 and failure is None
    receipt.update(actual_wait=True, actual_returncode=code, owned_process_group_absent=True,
                   status='succeeded' if successful else 'failed', products_complete=successful,
                   result_ref=result_ref, finished_utc=now(), elapsed_seconds=time.monotonic()-begin)
    if failure:
        receipt['failure'] = failure
    actual_ref = atomic_record(receipt_path, receipt, folder, immutable=True)
    # Native consumers authenticate this exact colocated wait, including result hash and bytes.
    if successful:
        product_path.parent.mkdir(parents=True, exist_ok=True)
        target = product_path.parent / 'actual_wait.json'
        require(not target.exists(), 'No overwrite of product actual wait')
        payload = (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode()
        require(tree_bytes(RESULTS) + len(payload) <= 2**30, 'Results cap before actual-wait publication')
        with target.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    require(successful, 'Owned ARGMix stage failed: ' + spec['id'])
    return actual_ref


def completion_path(spec):
    require(isinstance(spec.get("completion_file"), str) and spec["completion_file"] and
            "/" not in spec["completion_file"], "Every dependency declares an exact completion filename")
    output = Path(spec["output_directory"])
    require(output.is_absolute() and output.is_relative_to(ROOT), "Project-contained dependency output")
    return output / spec["completion_file"]


def dependency_state(spec, internal, external):
    missing = []
    for identifier in spec['dependencies']:
        product = completion_path(internal.get(identifier, external.get(identifier)))
        wait_path = product.parent / 'actual_wait.json'
        if wait_path.exists():
            # Inspect terminal failure even when no successful product was emitted.
            wait = read(ref(wait_path))
            require(wait['actual_wait'] is True and wait['actual_returncode'] == 0 and
                    wait['owned_process_group_absent'] is True and
                    wait.get('status', 'succeeded') == 'succeeded', 'Failed prerequisite: ' + identifier)
            require_wait(product, identifier)
        else:
            missing.append(identifier)
    return missing


def host_available_bytes():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))) * 1024


def gpu_capacity(index, minimum_bytes):
    listing = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = {int(row.split(",")[0]): row.split(",")[1:] for row in listing.splitlines()}
    uuid, free = [value.strip() for value in rows[index]]
    apps = subprocess.check_output(["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    return uuid, not apps.strip() and int(free) * 2**20 >= minimum_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(BASE / 'controller_manifest_v1.json'))
    parser.add_argument('--gpu', type=int, choices=[3, 4, 5], default=3)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest_ref = ref(args.manifest)
    manifest = read(manifest_ref)
    for source in manifest['source_refs']:
        verify(source)
    plan_ref, freeze_ref = manifest['plan_ref'], manifest['freeze_ref']
    plan = read(plan_ref)
    _, protocol = frozen(freeze_ref['path'])
    verify(freeze_ref)
    require(plan['protocol_ref'] == ref(BASE / 'protocol_v1.json') and plan['model_id'] == MODEL_ID,
            'Exact root-owned task plan and scientific protocol')
    require(len(plan['tasks']) == 15 and len({t['id'] for t in plan['tasks']}) == 15,
            'All15 successor stages exactly once')
    original = read(manifest['original_effective_plan_ref'])
    external = {t['id']: t for t in original['tasks']}
    internal = {t['id']: t for t in plan['tasks']}
    seen = set()
    for spec in plan['tasks']:
        require(all(dep in external or dep in seen for dep in spec['dependencies']), 'Forward-executable exact task DAG')
        completion_path(spec)
        for identifier in spec['dependencies']:
            completion_path(internal.get(identifier, external.get(identifier)))
        seen.add(spec['id'])
    if args.check_only:
        print(json.dumps(dict(passed=True, tasks=len(internal), manifest_ref=manifest_ref,
                              production_launched=False, heldout_payloads_opened=False), sort_keys=True), flush=True)
        return
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    folder = BASE / 'records_v1'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'controller.lock').open('a+b') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (folder / 'completed.json').exists(), 'ARGMix controller already complete')
        previous = None
        def status(state, reason, missing_dependencies=None):
            nonlocal previous
            key = (state, reason, tuple(missing_dependencies or []))
            if key != previous:
                atomic_record(folder/'state.json', dict(status=state, reason=reason, manifest_ref=manifest_ref,
                    plan_ref=plan_ref, model_id=MODEL_ID, controller_pid=os.getpid(),
                    missing_dependencies=missing_dependencies or [], updated_utc=now()), folder)
                print(state + ': ' + reason, flush=True)
                previous = key
        admission_path = ROOT / protocol['production_admission_path']
        while not admission_path.exists():
            status('waiting', 'Fresh root-confirmed25GiB production storage admission is pending')
            time.sleep(20)
        admission = cache_admission(admission_path, protocol)
        admission_ref = ref(admission_path)
        def dependencies(spec):
            while True:
                missing = dependency_state(spec, internal, external)
                if not missing:
                    return
                status('waiting', 'Successful actual producer waits pending', missing)
                time.sleep(20)
        gpu_lock = None
        gpu_waits, all_waits = [], []
        gpu_specs = [t for t in plan['tasks'] if t['kind'] == 'gpu']
        last_gpu = gpu_specs[-1]['id']
        try:
            for spec in plan['tasks']:
                dependencies(spec)
                verify(admission_ref)
                while host_available_bytes() < HOST_ADMISSION_BYTES:
                    status('waiting', 'Host admission requires64GiB family allowance plus64GiB free headroom before ' + spec['id'])
                    time.sleep(20)
                if spec['kind'] == 'gpu':
                    if gpu_lock is None:
                        gpu_lock = Path(plan['GPU_mutex_path']).open('a+b')
                        while True:
                            try:
                                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                status('waiting', 'Another authorized root GPU chain holds the shared mutex')
                                time.sleep(20)
                    while True:
                        uuid, available = gpu_capacity(args.gpu, spec['minimum_free_gpu_bytes'])
                        if available and host_available_bytes() >= HOST_ADMISSION_BYTES:
                            break
                        status('waiting', 'Empty assigned GPU and128GiB host capacity pending; existing jobs preserved')
                        time.sleep(20)
                    command = spec['argv_by_physical_gpu'][str(args.gpu)]
                    environment = dict(CUDA_VISIBLE_DEVICES=uuid, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
                else:
                    command, environment = spec['argv'], spec['environment']
                status('running', spec['id'])
                wait_ref = stage(spec, command, environment, folder, plan_ref, freeze_ref,
                                 admission['cache_max_bytes'], gpu_lock)
                all_waits.append(wait_ref)
                if spec['kind'] == 'gpu':
                    # The genuine child wait and complete session drain precede each mutex release.
                    gpu_lock.close(); gpu_lock = None
                    gpu_waits.append(wait_ref)
                    if spec['stage'] == 'qualify-gpu':
                        qualification = read(ref(Path(spec['output_directory']) / spec['completion_file']))
                        require(qualification['passed'] and qualification['physical_gpu'] == args.gpu and
                                qualification['maximum_peak_reserved_bytes'] <= spec['minimum_free_gpu_bytes'],
                                'Global full16-window GPU qualification fits registered capacity')
                    if spec['id'] == last_gpu:
                        last = read(wait_ref)
                        completed = folder / 'gpu_chain_completed.json'
                        if not completed.exists():
                            atomic_record(completed, dict(schema='joint_argmix_gpu_chain_completed_v1',
                                chain_id='argmix_joint', model_id=MODEL_ID, status='succeeded',
                                manifest_ref=manifest_ref, plan_ref=plan_ref, freeze_ref=freeze_ref,
                                stage_actual_wait_refs=gpu_waits, actual_wait=True, actual_returncode=0,
                                owned_process_group_absent=True, products_complete=True, gpu_mutex_released=True,
                                pid=last['pid'], process_start_ticks=last['process_start_ticks'],
                                physical_gpu=args.gpu, completed_utc=now()), folder, immutable=True)
            atomic_record(folder/'completed.json', dict(schema='joint_argmix_completed_chain_v1',
                model_id=MODEL_ID, plan_ref=plan_ref, freeze_ref=freeze_ref, stage_actual_wait_refs=all_waits,
                actual_wait=True, actual_returncode=0, owned_process_group_absent=True,
                products_complete=True, completed_utc=now()), folder, immutable=True)
            status('completed', 'Fixed-final fit, all held-out products, native paired score and validation diagnostic complete')
        except BaseException as error:
            status('failed', str(error))
            raise
        finally:
            if gpu_lock is not None:
                gpu_lock.close()


if __name__ == '__main__':
    main()

"""Original owned stage execution with atomic immutable receipt publication only."""
import os
from pathlib import Path
import subprocess
import time
import psutil
from arggnn.benchmarking.paper_campaign import processes
from research.joint_evaluation_20261029.argmix.recovery_20261030_queue.controller_v1 import (
    ROOT, RESULTS, CACHE, MODEL_ID, WORKER_MODULE, RECORD_CAP, ref, read, verify, require,
    now, tree_bytes, frozen, require_wait, owned_rss)
from research.joint_evaluation_20261029.argmix.recovery_20261030_atomic_publication_v1.atomic_v1 import atomic_record, publish_bytes


def stage(spec, command, environment, folder, plan_ref, freeze_ref, cache_limit, gpu_lock=None, recovery_binding=None):
    directory = folder / spec['id']
    started = directory / 'started.json'
    receipt_path = directory / 'actual_wait.json'
    product_path = Path(spec['output_directory']) / spec['completion_file']
    if receipt_path.exists():
        receipt = read(ref(receipt_path))
        require(receipt['actual_wait'] is True and receipt['actual_returncode'] == 0 and
                receipt['owned_process_group_absent'] is True and receipt['status'] == 'succeeded' and
                receipt['command'] == command and receipt['plan_ref'] == plan_ref and
                receipt['freeze_ref'] == freeze_ref and receipt.get('recovery_binding') == recovery_binding and
                not processes.session_members(receipt['pid']),
                'Only exact successful previously waited stage may be reused')
        verify(receipt['result_ref'])
        require(receipt['result_ref'] == ref(product_path), 'Reused wait binds the exact stage product')
        require_wait(product_path, spec['id'])
        return ref(receipt_path)
    require(not started.exists() and not product_path.exists(), 'Never silently replay or adopt a prior stage')
    frozen(freeze_ref['path'])
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update(environment)
    require(command[0] == str(ROOT / '.runtime/env/bin/python') and
            command[1:4] == ['-B', '-m', WORKER_MODULE] and
            all(isinstance(value, str) for value in command), 'Visible registered recovery worker module argv')
    require(command == spec['argv_by_physical_gpu']['3'] if spec['kind'] == 'gpu' else command == spec['argv'],
            'Exact reviewed stage argv, without hidden command substitutions')
    require(recovery_binding is not None, 'Explicit recovery authority on every actual stage wait')
    begin = time.monotonic()
    failure = None
    with (directory / 'worker.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True,
                                 pass_fds=() if gpu_lock is None else (gpu_lock.fileno(),))
        ticks = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        receipt = dict(task_id=spec['id'], model_id=MODEL_ID, plan_ref=plan_ref, freeze_ref=freeze_ref,
                       pid=child.pid, process_start_ticks=ticks, command=command, started_utc=now(),
                       recovery_binding=recovery_binding)
        atomic_record(started, receipt, folder, immutable=True)
        while True:
            try:
                code = child.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - begin > spec['max_seconds']:
                    failure = 'Registered wall time exceeded'
                elif owned_rss(child.pid) + psutil.Process().memory_info().rss > spec['max_rss_bytes']:
                    failure = 'Amended aggregate owned RSS limit exceeded: ' + str(spec['max_rss_bytes']) + ' bytes'
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
            result_ref = ref(product_path)
        except (FileNotFoundError, ValueError) as error:
            failure = 'Missing expected stage product: ' + str(error)
    successful = code == 0 and failure is None
    receipt.update(actual_wait=True, actual_returncode=code, owned_process_group_absent=True,
                   status='succeeded' if successful else 'failed', products_complete=successful,
                   result_ref=result_ref, finished_utc=now(), elapsed_seconds=time.monotonic()-begin)
    if failure:
        receipt['failure'] = failure
    actual_ref = atomic_record(receipt_path, receipt, folder, immutable=True)
    # Native consumers authenticate this exact colocated wait, using the strict path-and-SHA256 reference schema.
    if successful:
        product_path.parent.mkdir(parents=True, exist_ok=True)
        target = product_path.parent / 'actual_wait.json'
        require(not target.exists(), 'No overwrite of product actual wait')
        payload = (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode()
        require(tree_bytes(RESULTS) + len(payload) <= 2**30, 'Results cap before actual-wait publication')
        publish_bytes(target, payload, RESULTS, immutable=True, max_bytes=2**30)
    require(successful, 'Owned ARGMix stage failed: ' + spec['id'])
    return actual_ref


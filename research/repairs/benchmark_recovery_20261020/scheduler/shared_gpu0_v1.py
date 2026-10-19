"""Explicit borrowed GPU0 identity, one additional native reservation, live headroom."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from arggnn.benchmarking import contracts as c

GPU0_UUID = "GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
CUDA_CAP = 40 * 2**30
DRIVER_MARGIN = 2 * 2**30
START_FREE = 50 * 2**30
LIVE_FREE = 8 * 2**30
POOL = ".runtime/locks/shared_native_gpu0.lock"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), start_ticks=int(fields[19]))


def authority(reference):
    request = c.read_ref(reference)
    policy = c.read_json(c.POLICY_PATH)
    need(policy.get('benchmark_execution_approved') is True and policy.get('physical_gpus') == [0],
         'Current approval must remain physical GPU0 only')
    need(reference in policy.get('approved_shared_native_request_refs', []),
         'Exact shared native request has not been released by ROOT')
    need(policy.get('resource_continuation_drain_ref') == request['plan_ref'],
         'Prior v9 future dispatch must be cooperatively drained')
    need(request['cuda_cap_bytes'] == CUDA_CAP and request['driver_margin_bytes'] == DRIVER_MARGIN
         and request['start_free_bytes'] == START_FREE and request['live_free_bytes'] == LIVE_FREE,
         'Exact ROOT-approved bounded sharing envelope required')
    return request


def gpu_sample(run=subprocess.run):
    result = run(['nvidia-smi', '--id=0',
        '--query-gpu=index,uuid,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    rows = [line.split(',') for line in result.stdout.splitlines() if line.strip()]
    need(len(rows) == 1 and len(rows[0]) == 5, 'One exact physical GPU0 inventory row required')
    index, uuid, total, used, free = [x.strip() for x in rows[0]]
    need(index == '0' and uuid == GPU0_UUID, 'Unexpected physical device identity')
    total, used, free = [int(x) * 2**20 for x in (total, used, free)]
    reserved=total-used-free
    need(0 <= used <= total and 0 <= free <= total and -2*2**20 <= reserved <= total,
         'Incomplete or inconsistent GPU0 memory inventory')
    return dict(physical_gpu=0, uuid=uuid, total_bytes=total, used_bytes=used,
                free_bytes=free, driver_reserved_bytes=max(0,reserved),
                driver_reserved_bytes_derived_from_inventory=True,monotonic=time.monotonic())


def check_start(sample):
    need(sample['free_bytes'] >= START_FREE, 'Shared native first CUDA requires at least50GiB actually free')
    need(CUDA_CAP + DRIVER_MARGIN + LIVE_FREE <= sample['free_bytes'],
         'Native reservation, driver margin and live headroom do not fit')


class HeadroomMonitor:
    def __init__(self):
        self.consecutive_breaches = 0
        self.minimum_free_bytes = None
        self.samples = 0
    def observe(self, sample):
        free = sample['free_bytes']
        self.samples += 1
        self.minimum_free_bytes = free if self.minimum_free_bytes is None else min(self.minimum_free_bytes, free)
        self.consecutive_breaches = self.consecutive_breaches + 1 if free < LIVE_FREE else 0
        need(self.consecutive_breaches < 3,
             'GPU0 free memory stayed below8GiB for three consecutive samples; stop only the new native job')


def borrow_snapshot(worker_pid):
    path = c.contained_path('.runtime/locks/gpu_0.lock')
    info = path.stat()
    record = json.loads(path.read_text())
    values = dict(x.split('=', 1) for x in (Path('/proc') / str(worker_pid) / 'environ').read_bytes().decode().split('\0') if '=' in x)
    descriptor = int(values['ARGGNN_GPU_LEASE_FD'])
    return dict(device=info.st_dev, inode=info.st_ino, record=record,
                launcher_identity=identity(record['launcher_pid']),
                worker_identity=identity(worker_pid), worker_lease_fd=descriptor)


def validate_borrow(expected):
    path = c.contained_path('.runtime/locks/gpu_0.lock')
    info = path.stat()
    need((info.st_dev, info.st_ino) == (expected['device'], expected['inode'])
         and json.loads(path.read_text()) == expected['record'], 'Borrowed physical lease changed')
    need(expected['record'].get('gpu') == 0 and expected['record'].get('uuid') == GPU0_UUID,
         'Borrowed lease must refer to physical GPU0')
    for name in ('launcher_identity', 'worker_identity'):
        need(identity(expected[name]['pid']) == expected[name], 'Original TreeLoom owner generation changed')
    descriptor_path = Path('/proc') / str(expected['worker_identity']['pid']) / 'fd' / str(expected['worker_lease_fd'])
    held = descriptor_path.stat()
    need((held.st_dev, held.st_ino) == (expected['device'], expected['inode']),
         'Original worker no longer retains the borrowed lease inode')
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        raise RuntimeError('Original physical GPU0 exclusive lease is no longer held')
    finally:
        os.close(descriptor)


def validate_reservation(request, descriptor, supervisor_pid):
    named=c.contained_path(POOL).stat()
    held=os.fstat(descriptor)
    record=json.loads(os.pread(descriptor,65536,0))
    need((named.st_dev,named.st_ino)==(held.st_dev,held.st_ino)
         and record['supervisor_identity']==identity(supervisor_pid)
         and record['task_id']==request['task']['id'], 'Exact logical native reservation changed')
    if request['borrowed_lease'] is not None:
        validate_borrow(request['borrowed_lease'])
    else:
        physical=int(record['owned_physical_lease_fd'])
        path=c.contained_path('.runtime/locks/gpu_0.lock')
        owned,named=os.fstat(physical),path.stat()
        value=json.loads(os.pread(physical,65536,0))
        need((owned.st_dev,owned.st_ino)==(named.st_dev,named.st_ino)
             and value.get('launcher_pid')==supervisor_pid and value.get('gpu')==0
             and value.get('uuid')==GPU0_UUID and value.get('job')==request['task']['id'],
             'Actual new canonical GPU0 lease changed')
    return record


@contextmanager
def reserve_native(reference, request):
    path=c.contained_path(POOL)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    physical=None
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if request['borrowed_lease'] is not None:
            validate_borrow(request['borrowed_lease'])
        else:
            physical=os.open(c.contained_path('.runtime/locks/gpu_0.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
            fcntl.flock(physical,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sample=gpu_sample()
        check_start(sample)
        if physical is not None:
            lease=dict(gpu=0,uuid=GPU0_UUID,launcher_pid=os.getpid(),job=request['task']['id'],started_unix=time.time())
            os.ftruncate(physical,0)
            os.write(physical,json.dumps(lease,sort_keys=True).encode())
            os.fsync(physical)
        record=dict(schema='shared_native_gpu0_reservation_v1',request_ref=reference,
            supervisor_identity=identity(os.getpid()),task_id=request['task']['id'],
            cuda_cap_bytes=CUDA_CAP,driver_margin_bytes=DRIVER_MARGIN,
            borrowed_physical_lease=physical is None,inherited_physical_lease=False,
            owned_physical_lease_fd=physical)
        os.ftruncate(descriptor,0)
        os.write(descriptor,(json.dumps(record,sort_keys=True)+'\n').encode())
        os.fsync(descriptor)
        yield descriptor,sample
    finally:
        if physical is not None:os.close(physical)
        os.close(descriptor)


def validate_worker_lease(first_cuda=False):
    reference = json.loads(os.environ['ARGGNN_SHARED_NATIVE_REQUEST_REF'])
    request = authority(reference)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU0_UUID
         and os.environ.get('ARGGNN_PHYSICAL_GPU') == '0'
         and int(os.environ.get('ARGGNN_MAX_CUDA_RESERVED_BYTES', '0')) == CUDA_CAP
         and 'ARGGNN_GPU_LEASE_FD' not in os.environ,
         'Explicit borrowed GPU0 visibility and40GiB allocator cap required')
    descriptor = int(os.environ['ARGGNN_SHARED_NATIVE_FD'])
    info, named = os.fstat(descriptor), c.contained_path(POOL).stat()
    record = json.loads(os.pread(descriptor, 65536, 0))
    need((info.st_dev, info.st_ino) == (named.st_dev, named.st_ino)
         and record.get('request_ref') == reference
         and record.get('task_id') == request['task']['id']
         and record.get('cuda_cap_bytes') == CUDA_CAP
         and record.get('supervisor_identity') == identity(os.getppid()),
         'Exact inherited logical reservation must belong to this native supervisor')
    validate_reservation(request,descriptor,os.getppid())
    if first_cuda:
        sample = gpu_sample()
        check_start(sample)
        path = c.contained_path(request['call']['spec']['output_directory']) / 'before_first_cuda.json'
        path.write_text(json.dumps(sample, sort_keys=True)+'\n')
    return request


def await_startup():
    descriptor = int(os.environ.pop('ARGGNN_SHARED_STARTUP_FD'))
    try:
        need(os.read(descriptor, 1) == b'1', 'Supervisor did not release the recorded native identity')
    finally:
        os.close(descriptor)


def native_driver_bytes(pids, run=subprocess.run):
    result = run(['nvidia-smi', '--id=0', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    total = 0
    for line in result.stdout.splitlines():
        if not line.strip():continue
        fields = [value.strip() for value in line.split(',')]
        need(len(fields) == 3 and fields[0] == GPU0_UUID, 'Unexpected GPU process inventory')
        if int(fields[1]) in pids:
            total += int(fields[2]) * 2**20
    need(total <= CUDA_CAP+DRIVER_MARGIN, 'New owned native GPU memory exceeded40GiB plus2GiB driver margin')
    return total

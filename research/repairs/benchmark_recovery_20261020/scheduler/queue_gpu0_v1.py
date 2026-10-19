"""Shared GPU0 reservations based on free memory and unallocated owned growth."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .shared_gpu0_v1 import GPU0_UUID, gpu_sample, identity
from arggnn.benchmarking import contracts as c

POOL = c.ROOT/'.runtime/locks/readable_gpu0'
CAPACITY = c.ROOT/'.runtime/locks/benchmark_recovery_capacity_20261020.lock'
MARGIN = 2 * 2**30
FREE_FLOOR = 8 * 2**30

class CapacityUnavailable(RuntimeError):
    pass


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


@contextmanager
def capacity_lock(blocking=True):
    CAPACITY.parent.mkdir(parents=True,exist_ok=True)
    with CAPACITY.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def record_path(task):
    return POOL/(hashlib.sha256(task['id'].encode()).hexdigest()+'.json')


def process_bytes():
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    rows = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        device, pid, memory = [v.strip() for v in line.split(',')]
        need(device == GPU0_UUID, 'Unexpected GPU inventory identity')
        rows[int(pid)] = int(memory) * 2**20
    return rows


def _outstanding_growth_locked(plan,exclude_ref=None):
    used = process_bytes()
    total = standalone_outstanding_growth(plan,used)
    if not POOL.exists():
        return total
    for path in POOL.glob('*.json'):
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                row = json.loads(os.pread(fd, 65536, 0))
                if row['job_ref']==exclude_ref:continue
                resident = 0
                for item in row.get('members', []):
                    try:
                        if identity(item['pid']) == item:
                            resident += used.get(item['pid'], 0)
                    except FileNotFoundError:
                        pass
                total += max(0, row['cap_bytes'] + MARGIN - resident)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    return total


def _admission_reason_locked(plan, task, exclude_ref=None):
    required=max(task.get('minimum_free_gpu_bytes',0),task['max_cuda_reserved_bytes']+MARGIN+FREE_FLOOR)
    available=gpu_sample()['free_bytes']-_outstanding_growth_locked(plan,exclude_ref)
    return None if available>=required else 'GPU0 free memory minus owned reservation growth is insufficient'


def admission_reason(plan, task, locked=False):
    if locked:return _admission_reason_locked(plan,task)
    try:
        with capacity_lock(blocking=False):return _admission_reason_locked(plan,task)
    except BlockingIOError:
        return 'Shared recovery capacity admission is occupied'


@contextmanager
def reserve(job, ref):
    task = job['task']
    POOL.mkdir(parents=True,exist_ok=True)
    physical = os.open(c.ROOT/'.runtime/locks/gpu_0.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    descriptor = None
    try:
        # Legacy native work retains an exclusive lease. New readable workers
        # share this lease and account for their own memory separately.
        fcntl.flock(physical, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with capacity_lock(blocking=False):
            reason = _admission_reason_locked(c.read_ref(job['plan_ref']), task)
            if reason:
                raise CapacityUnavailable(reason)
            descriptor = os.open(record_path(task), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            row = dict(job_ref=ref, task_id=task['id'], supervisor_identity=identity(os.getpid()),
                       cap_bytes=task['max_cuda_reserved_bytes'], members=[], physical_fd=physical)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, json.dumps(row).encode())
            os.fsync(descriptor)
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(physical)


def inherited(descriptor):
    with capacity_lock():
        row=json.loads(os.pread(descriptor,65536,0))
        return descriptor,row['physical_fd']


def set_root(descriptor,root):
    with capacity_lock():
        row=json.loads(os.pread(descriptor,65536,0))
        row['scientific_root']=root
        os.ftruncate(descriptor,0)
        os.pwrite(descriptor,json.dumps(row).encode(),0)
        os.fsync(descriptor)


def validate_child(job,ref,first_cuda=False):
    from arggnn.benchmarking import workflow_v4 as lifecycle
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU0_UUID and os.environ.get('ARGGNN_PHYSICAL_GPU')=='0',
         'Only physical GPU0 may be visible')
    need(int(os.environ['ARGGNN_MAX_CUDA_RESERVED_BYTES'])==job['task']['max_cuda_reserved_bytes'],
         'Exact admitted CUDA allocator cap required')
    fd=int(os.environ['ARGGNN_READABLE_GPU_FD'])
    with capacity_lock():
        row=json.loads(os.pread(fd,65536,0))
        held,named=os.fstat(fd),record_path(job['task']).stat()
        need((held.st_dev,held.st_ino)==(named.st_dev,named.st_ino) and row['job_ref']==ref,
             'Exact inherited logical GPU reservation required')
        need(row['supervisor_identity']==identity(row['supervisor_identity']['pid']),
             'Actual supervisor generation changed')
        root=row['scientific_root']
        current=lifecycle.identity(os.getpid())
        need(current['pgid']==root['pgid']==root['pid'] and current['session']==root['session']==root['pid']
             and current['start_ticks']>=root['start_ticks'], 'Worker left its exact admitted scientific session')
        physical,named=os.fstat(row['physical_fd']),(c.ROOT/'.runtime/locks/gpu_0.lock').stat()
        need((physical.st_dev,physical.st_ino)==(named.st_dev,named.st_ino),'Physical shared lease inode changed')
        if first_cuda:
            reason=_admission_reason_locked(c.read_ref(job['plan_ref']),job['task'],exclude_ref=ref)
            if reason:raise CapacityUnavailable(reason)
    return job


def monitor(job,ref,fd,tracker,expected_owner=None):
    members=[]
    for pid in tracker.capture():
        try:members.append(identity(pid))
        except (FileNotFoundError,ProcessLookupError):pass
    used=process_bytes()
    owned=sum(used.get(item['pid'],0) for item in members)
    sample=gpu_sample()
    with capacity_lock():
        row=json.loads(os.pread(fd,65536,0))
        need(row['job_ref']==ref and row['supervisor_identity']==(expected_owner or identity(os.getpid())),'GPU reservation changed')
        row['members']=members
        row['low_memory_samples']=row.get('low_memory_samples',0)+1 if sample['free_bytes']<FREE_FLOOR else 0
        row['last_sample']=dict(sample,owned_driver_bytes=owned)
        os.ftruncate(fd,0)
        os.pwrite(fd,json.dumps(row).encode(),0)
        os.fsync(fd)
    need(owned<=row['cap_bytes']+MARGIN,'Owned GPU memory exceeds cap and driver margin')
    need(row['low_memory_samples']<3,'GPU free memory below8GiB for three samples')


def standalone_outstanding_growth(plan,used):
    """Reserve unallocated CUDA growth of separately leased standalone jobs."""
    from . import queue_adoption_v1 as adoption
    from arggnn.benchmarking import workflow_v4 as lifecycle
    from research.reviews.benchmark_repairs_20261019.supervision.finite_pss_v1 import repaired_owned_group_class
    total=0
    for task in plan['tasks']:
        binding=task.get('standalone_predecessor')
        if not binding or not binding.get('max_cuda_reserved_bytes') or adoption.drained(binding):continue
        resident=0
        try:
            root=adoption.scientific_root(binding)
            current=lifecycle.identity(root['pid'])
            need(all(current[key]==root[key] for key in ('pid','start_ticks','pgid','session')),
                 'Standalone scientific generation changed')
            tracker=repaired_owned_group_class(lifecycle)(root['pid'])
            for pid in tracker.capture():resident+=used.get(pid,0)
        except (OSError,ValueError,RuntimeError,KeyError):pass
        total+=max(0,binding['max_cuda_reserved_bytes']+MARGIN-resident)
    return total


def monitor_deferred(job,ref,tracker):
    """Observe a scientific child's late lease without claiming its ownership."""
    path=record_path(job['task'])
    if not path.exists():return
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            owner=identity(tracker.root['pid'])
            monitor(job,ref,fd,tracker,expected_owner=owner)
        else:fcntl.flock(fd,fcntl.LOCK_UN)
    finally:os.close(fd)

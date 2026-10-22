"""Exclusive owned-device leases without artificial CUDA allocator limits.

The canonical release, immutable task, exact hardware UUID, inherited physical
lease and generation-aware scientific session remain mandatory. Actual free
memory is checked before CUDA; low free memory is telemetry, not cancellation.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from .shared_gpu_v1 import GPU_UUIDS, target, validate_visible_target, gpu_sample, identity, process_bytes
from . import contracts_recovery_v1 as c

CAPACITY=c.ROOT/'.runtime/locks/benchmark_recovery_capacity_20261020.lock'


class CapacityUnavailable(RuntimeError):
    pass


def need(value,message):
    if not value:raise RuntimeError(message)


def pool_directory(physical_gpu):
    need(type(physical_gpu) is int and physical_gpu in GPU_UUIDS,'Authorized pool device required')
    return c.ROOT/'.runtime/locks'/('readable_gpu345_full_vram_gpu'+str(physical_gpu)+'_20261022')


def physical_path(physical_gpu):
    need(type(physical_gpu) is int and physical_gpu in GPU_UUIDS,'Authorized physical lease required')
    # Reuse the physical inode held shared by healthy legacy workers. An
    # exclusive new lease waits for them without changing their source or jobs.
    return c.ROOT/'.runtime/locks'/('gpu_'+str(physical_gpu)+'.lock')


def validate_job_target(job):
    index,uuid=target(job['task'])
    plan=c.read_ref(job['plan_ref'])
    need(plan.get('physical_gpus')==[3,4,5] and index in plan['physical_gpus'],'Exact parallel GPU3/4/5 plan required')
    need(job['task'].get('gpu_memory_policy')=='full_device','Explicit full-device task permission required')
    policy=c.check_execution_release(c.file_ref(c.POLICY_PATH))
    need(policy.get('approved_gpu345_workflow_ref')==job['plan_ref'],'Exact full-VRAM workflow must retain current approval')
    return plan


def validate_policy_environment(task=None):
    validate_visible_target(task)
    need(os.environ.get('ARGGNN_GPU_MEMORY_POLICY')=='full_device','Authenticated full-device GPU memory environment required')
    need(os.environ.get('ARGGNN_MAX_CUDA_RESERVED_BYTES') in (None,'','0'),'Artificial CUDA allocator cap must be absent')
    if task is not None:need(task.get('gpu_memory_policy')=='full_device','Full-device task permission required')


def validate_record_target(row,physical_gpu):
    need(row.get('physical_gpu')==physical_gpu and row.get('gpu_uuid')==GPU_UUIDS[physical_gpu],
         'Pool record must name exact assigned physical GPU and UUID')
    need(row.get('gpu_memory_policy')=='full_device' and row.get('physical_lease')=='exclusive',
         'An exclusive full-device GPU lease is required')


@contextmanager
def capacity_lock(blocking=True):
    CAPACITY.parent.mkdir(parents=True,exist_ok=True)
    with CAPACITY.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
        yield


def record_path(task):
    index,_=target(task)
    return pool_directory(index)/(hashlib.sha256(task['id'].encode()).hexdigest()+'.json')


def standalone_writer_live(plan,physical_gpu):
    """Do not overlap an authenticated old standalone lacking this lease."""
    from research.repairs.benchmark_recovery_20261020.scheduler import queue_adoption_v1 as adoption
    for task in plan['tasks']:
        binding=task.get('standalone_predecessor')
        if not binding or binding.get('physical_gpu',5)!=physical_gpu:continue
        if binding.get('max_cuda_reserved_bytes') and not adoption.drained(binding):return True
    return False


def _admission_reason_locked(plan,task,exclude_ref=None,physical_held=False):
    index,_=target(task)
    need(plan.get('physical_gpus')==[3,4,5],'Only the explicit parallel plan may enter this pool')
    need(task.get('gpu_memory_policy')=='full_device','Explicit full-device GPU permission required')
    required=task.get('minimum_free_gpu_bytes')
    need(type(required) is int and required>=0,'Evidence-based actual free-memory threshold required')
    if standalone_writer_live(plan,index):return 'A preserved standalone GPU writer has not drained'
    if not physical_held:
        descriptor=os.open(physical_path(index),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return 'Assigned physical GPU has a preserved or active owned lease'
        finally:os.close(descriptor)
    sample=gpu_sample(index)
    return None if sample['free_bytes']>=required else 'Assigned GPU actual free memory is below the evidenced admission threshold'


def admission_reason(plan,task,locked=False):
    if locked:return _admission_reason_locked(plan,task)
    try:
        with capacity_lock(blocking=False):return _admission_reason_locked(plan,task)
    except BlockingIOError:return 'Shared recovery capacity admission is occupied'


@contextmanager
def reserve(job,reference):
    task=job['task'];index,uuid=target(task)
    plan=validate_job_target(job)
    pool_directory(index).mkdir(parents=True,exist_ok=True)
    physical=os.open(physical_path(index),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    descriptor=None
    try:
        fcntl.flock(physical,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with capacity_lock(blocking=False):
            reason=_admission_reason_locked(plan,task,physical_held=True)
            if reason:raise CapacityUnavailable(reason)
            descriptor=os.open(record_path(task),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            sample=gpu_sample(index)
            row=dict(job_ref=reference,task_id=task['id'],supervisor_identity=identity(os.getpid()),
                cap_bytes=sample['total_bytes'],members=[],physical_fd=physical,
                physical_gpu=index,gpu_uuid=uuid,gpu_memory_policy='full_device',
                physical_lease='exclusive',artificial_allocator_limit=False,
                admission_sample=sample)
            os.ftruncate(descriptor,0);os.write(descriptor,json.dumps(row).encode());os.fsync(descriptor)
        yield descriptor
    finally:
        if descriptor is not None:os.close(descriptor)
        os.close(physical)


def inherited(descriptor):
    with capacity_lock():
        row=json.loads(os.pread(descriptor,65536,0))
        validate_record_target(row,row['physical_gpu'])
        return descriptor,row['physical_fd']


def set_root(descriptor,root):
    with capacity_lock():
        row=json.loads(os.pread(descriptor,65536,0))
        validate_record_target(row,row['physical_gpu'])
        row['scientific_root']=root
        os.ftruncate(descriptor,0);os.pwrite(descriptor,json.dumps(row).encode(),0);os.fsync(descriptor)


def validate_child(job,reference,first_cuda=False):
    from arggnn.benchmarking import workflow_v4 as lifecycle
    validate_job_target(job)
    index,_=validate_visible_target(job['task'])
    validate_policy_environment(job['task'])
    descriptor=int(os.environ['ARGGNN_READABLE_GPU_FD'])
    with capacity_lock():
        row=json.loads(os.pread(descriptor,65536,0))
        validate_record_target(row,index)
        held,named=os.fstat(descriptor),record_path(job['task']).stat()
        need((held.st_dev,held.st_ino)==(named.st_dev,named.st_ino) and row['job_ref']==reference,
             'Exact inherited logical GPU reservation required')
        need(row['supervisor_identity']==identity(row['supervisor_identity']['pid']),'Actual supervisor generation changed')
        root=row['scientific_root'];current=lifecycle.identity(os.getpid())
        need(current['pgid']==root['pgid']==root['pid'] and current['session']==root['session']==root['pid']
             and current['start_ticks']>=root['start_ticks'],'Worker left its exact admitted scientific session')
        physical,named=os.fstat(row['physical_fd']),physical_path(index).stat()
        need((physical.st_dev,physical.st_ino)==(named.st_dev,named.st_ino),'Physical exclusive lease inode changed')
        if first_cuda:
            reason=_admission_reason_locked(c.read_ref(job['plan_ref']),job['task'],exclude_ref=reference,physical_held=True)
            if reason:raise CapacityUnavailable(reason)
    return job


def monitor(job,reference,descriptor,tracker,expected_owner=None):
    index,_=target(job['task']);members=[]
    for pid in tracker.capture():
        try:members.append(identity(pid))
        except (FileNotFoundError,ProcessLookupError):pass
    # Preserve observation and generation-aware identity checks. Actual CUDA
    # failures remain genuine failures; VRAM use never triggers an invented cap.
    used=process_bytes(index);owned=sum(used.get(member['pid'],0) for member in members)
    sample=gpu_sample(index)
    with capacity_lock():
        row=json.loads(os.pread(descriptor,65536,0));validate_record_target(row,index)
        need(row['job_ref']==reference and row['supervisor_identity']==(expected_owner or identity(os.getpid())),
             'GPU reservation changed')
        row['members']=members
        row['last_sample']=dict(sample,owned_driver_bytes=owned,gpu_memory_policy='full_device',artificial_allocator_limit=False)
        os.ftruncate(descriptor,0);os.pwrite(descriptor,json.dumps(row).encode(),0);os.fsync(descriptor)


def monitor_deferred(job,reference,tracker):
    path=record_path(job['task'])
    if not path.exists():return
    descriptor=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            owner=identity(tracker.root['pid'])
            monitor(job,reference,descriptor,tracker,expected_owner=owner)
        else:fcntl.flock(descriptor,fcntl.LOCK_UN)
    finally:os.close(descriptor)

"""Per-device shared reservations with one preserved global capacity mutex."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .shared_gpu_v1 import GPU_UUIDS, target, validate_visible_target, gpu_sample, identity
from . import contracts_parallel_v1 as c

CAPACITY=c.ROOT/'.runtime/locks/benchmark_recovery_capacity_20261020.lock'
MARGIN=2*2**30
FREE_FLOOR=8*2**30


class CapacityUnavailable(RuntimeError):
    pass


def need(value,message):
    if not value:raise RuntimeError(message)


def pool_directory(physical_gpu):
    need(type(physical_gpu) is int and physical_gpu in GPU_UUIDS,'Authorized pool device required')
    return c.ROOT/'.runtime/locks'/('readable_parallel_gpu'+str(physical_gpu)+'_20261022')


def physical_path(physical_gpu):
    need(type(physical_gpu) is int and physical_gpu in GPU_UUIDS,'Authorized physical lease required')
    return c.ROOT/'.runtime/locks'/('gpu_'+str(physical_gpu)+'.lock')


def accounting_pool_directories(physical_gpu):
    directories=[pool_directory(physical_gpu)]
    if physical_gpu==5:directories.append(c.ROOT/'.runtime/locks/readable_gpu5_resume_20261021')
    return directories


def validate_job_target(job):
    index,uuid=target(job['task'])
    plan=c.read_ref(job['plan_ref'])
    need(plan.get('physical_gpus')==[2,3,5] and index in plan['physical_gpus'],'Exact parallel GPU2/3/5 plan required')
    policy=c.check_execution_release(c.file_ref(c.POLICY_PATH))
    need(policy.get('approved_parallel_workflow_ref')==job['plan_ref'],'Exact parallel workflow must retain current approval')
    return plan


def validate_record_target(row,physical_gpu):
    need(row.get('physical_gpu')==physical_gpu and row.get('gpu_uuid')==GPU_UUIDS[physical_gpu],
         'Pool record must name exact assigned physical GPU and UUID')


@contextmanager
def capacity_lock(blocking=True):
    CAPACITY.parent.mkdir(parents=True,exist_ok=True)
    with CAPACITY.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
        yield


def record_path(task):
    index,_=target(task)
    return pool_directory(index)/(hashlib.sha256(task['id'].encode()).hexdigest()+'.json')


def process_bytes(physical_gpu):
    need(type(physical_gpu) is int and physical_gpu in GPU_UUIDS,'Authorized compute inventory required')
    result=subprocess.run(['nvidia-smi','--id='+str(physical_gpu),
        '--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=10)
    rows={}
    for line in result.stdout.splitlines():
        if not line.strip():continue
        uuid,pid,memory=[value.strip() for value in line.split(',')]
        need(uuid==GPU_UUIDS[physical_gpu],'Unexpected GPU inventory identity')
        rows[int(pid)]=int(memory)*2**20
    return rows


def _outstanding_growth_locked(plan,physical_gpu,exclude_ref=None):
    used=process_bytes(physical_gpu)
    total=standalone_outstanding_growth(plan,used,physical_gpu)
    for directory in accounting_pool_directories(physical_gpu):
        if not directory.exists():continue
        for path in directory.glob('*.json'):
            descriptor=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
            try:
                try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    row=json.loads(os.pread(descriptor,65536,0))
                    validate_record_target(row,physical_gpu)
                    if row['job_ref']==exclude_ref:continue
                    resident=0
                    for member in row.get('members',[]):
                        try:
                            if identity(member['pid'])==member:resident+=used.get(member['pid'],0)
                        except (FileNotFoundError,ProcessLookupError):pass
                    total+=max(0,row['cap_bytes']+MARGIN-resident)
                else:fcntl.flock(descriptor,fcntl.LOCK_UN)
            finally:os.close(descriptor)
    return total


def _admission_reason_locked(plan,task,exclude_ref=None):
    index,_=target(task)
    need(plan.get('physical_gpus')==[2,3,5],'Only the explicit parallel plan may enter this pool')
    required=max(task.get('minimum_free_gpu_bytes',0),task['max_cuda_reserved_bytes']+MARGIN+FREE_FLOOR)
    available=gpu_sample(index)['free_bytes']-_outstanding_growth_locked(plan,index,exclude_ref)
    return None if available>=required else 'Assigned GPU free memory minus owned reservation growth is insufficient'


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
        fcntl.flock(physical,fcntl.LOCK_SH|fcntl.LOCK_NB)
        with capacity_lock(blocking=False):
            reason=_admission_reason_locked(plan,task)
            if reason:raise CapacityUnavailable(reason)
            descriptor=os.open(record_path(task),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            row=dict(job_ref=reference,task_id=task['id'],supervisor_identity=identity(os.getpid()),
                cap_bytes=task['max_cuda_reserved_bytes'],members=[],physical_fd=physical,
                physical_gpu=index,gpu_uuid=uuid)
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
    need(int(os.environ['ARGGNN_MAX_CUDA_RESERVED_BYTES'])==job['task']['max_cuda_reserved_bytes'],
         'Exact admitted CUDA allocator cap required')
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
        need((physical.st_dev,physical.st_ino)==(named.st_dev,named.st_ino),'Physical shared lease inode changed')
        if first_cuda:
            reason=_admission_reason_locked(c.read_ref(job['plan_ref']),job['task'],exclude_ref=reference)
            if reason:raise CapacityUnavailable(reason)
    return job


def monitor(job,reference,descriptor,tracker,expected_owner=None):
    index,_=target(job['task']);members=[]
    for pid in tracker.capture():
        try:members.append(identity(pid))
        except (FileNotFoundError,ProcessLookupError):pass
    used=process_bytes(index);owned=sum(used.get(member['pid'],0) for member in members)
    sample=gpu_sample(index)
    with capacity_lock():
        row=json.loads(os.pread(descriptor,65536,0));validate_record_target(row,index)
        need(row['job_ref']==reference and row['supervisor_identity']==(expected_owner or identity(os.getpid())),
             'GPU reservation changed')
        row['members']=members
        row['low_memory_samples']=row.get('low_memory_samples',0)+1 if sample['free_bytes']<FREE_FLOOR else 0
        row['last_sample']=dict(sample,owned_driver_bytes=owned)
        os.ftruncate(descriptor,0);os.pwrite(descriptor,json.dumps(row).encode(),0);os.fsync(descriptor)
    need(owned<=row['cap_bytes']+MARGIN,'Owned GPU memory exceeds cap and driver margin')
    need(row['low_memory_samples']<3,'GPU free memory below8GiB for three samples')


def standalone_outstanding_growth(plan,used,physical_gpu):
    from research.repairs.benchmark_recovery_20261020.scheduler import queue_adoption_v1 as adoption
    from arggnn.benchmarking import workflow_v4 as lifecycle
    from research.reviews.benchmark_repairs_20261019.supervision.finite_pss_v1 import repaired_owned_group_class
    total=0
    for task in plan['tasks']:
        binding=task.get('standalone_predecessor')
        if not binding or binding.get('physical_gpu',5)!=physical_gpu:continue
        if not binding.get('max_cuda_reserved_bytes') or adoption.drained(binding):continue
        resident=0
        try:
            root=adoption.scientific_root(binding);current=lifecycle.identity(root['pid'])
            need(all(current[key]==root[key] for key in ('pid','start_ticks','pgid','session')),'Standalone scientific generation changed')
            tracker=repaired_owned_group_class(lifecycle)(root['pid'])
            for pid in tracker.capture():resident+=used.get(pid,0)
        except (OSError,ValueError,RuntimeError,KeyError):pass
        total+=max(0,binding['max_cuda_reserved_bytes']+MARGIN-resident)
    return total


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

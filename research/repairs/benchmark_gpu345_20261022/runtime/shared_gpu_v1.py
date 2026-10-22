"""Explicit task device selection and read-only GPU3/4/5 hardware inventory."""
from pathlib import Path
import os
import subprocess
import time

GPU_UUIDS={
    4:'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',
    3:'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',
    5:'GPU-3db85505-d061-e45d-d154-e7e8c1cad497',
}


def need(value,message):
    if not value:raise RuntimeError(message)


def target(task):
    index=task.get('physical_gpu')
    need(type(index) is int and index in GPU_UUIDS,'Explicit physical GPU3/4/5 task assignment required')
    need(task.get('gpu_uuid')==GPU_UUIDS[index],'Task index and exact GPU UUID must match')
    return index,GPU_UUIDS[index]


def validate_visible_target(task=None):
    value=os.environ.get('ARGGNN_PHYSICAL_GPU')
    need(value in ('3','4','5'),'An admitted physical GPU3/4/5 environment is required')
    index=int(value)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUIDS[index],'Exact single admitted GPU UUID visibility required')
    if task is not None:need(target(task)==(index,GPU_UUIDS[index]),'Environment differs from exact assigned task device')
    return index,GPU_UUIDS[index]


def identity(pid):
    fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(pid),start_ticks=int(fields[19]))


def gpu_sample(physical_gpu,run=subprocess.run):
    need(type(physical_gpu) is int and physical_gpu in GPU_UUIDS,'Authorized inventory device required')
    result=run(['nvidia-smi','--id='+str(physical_gpu),
        '--query-gpu=index,uuid,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
    rows=[line.split(',') for line in result.stdout.splitlines() if line.strip()]
    need(len(rows)==1 and len(rows[0])==5,'One exact assigned physical GPU inventory row required')
    index,uuid,total,used,free=[x.strip() for x in rows[0]]
    need(index==str(physical_gpu) and uuid==GPU_UUIDS[physical_gpu],'Unexpected physical device identity')
    total,used,free=[int(x)*2**20 for x in (total,used,free)]
    reserved=total-used-free
    need(0<=used<=total and 0<=free<=total and -2*2**20<=reserved<=total,'Inconsistent GPU memory inventory')
    return dict(physical_gpu=physical_gpu,uuid=uuid,total_bytes=total,used_bytes=used,free_bytes=free,
        driver_reserved_bytes=max(0,reserved),driver_reserved_bytes_derived_from_inventory=True,
        monotonic=time.monotonic())


def process_bytes(physical_gpu):
    """Read driver residency for the currently authorized exact device UUID."""
    need(type(physical_gpu) is int and physical_gpu in GPU_UUIDS,'Authorized compute inventory required')
    result=subprocess.run(['nvidia-smi','--id='+str(physical_gpu),
        '--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=10)
    used={}
    for line in result.stdout.splitlines():
        if not line.strip():continue
        uuid,pid,memory=[value.strip() for value in line.split(',')]
        need(uuid==GPU_UUIDS[physical_gpu],'Unexpected GPU inventory identity')
        need(pid.isdigit() and memory.isdigit(),'Finite exact GPU process inventory required')
        used[int(pid)]=int(memory)*2**20
    return used

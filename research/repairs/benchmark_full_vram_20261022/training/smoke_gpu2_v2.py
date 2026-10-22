"""Review-only preparation for a tiny actual-device GPU2 setup smoke.

This is a direct smoke with its own exclusive project physical-GPU lock. It does
not create or claim a queue job reservation. The callback checks canonical
execution authority, the exact held lock inode and the approved AM task's actual
free-memory threshold before the shared helper initializes CUDA. At most one
1024-byte FP32 tensor is allocated; no data, model, optimizer or training is run.
Run only after root review, with the normal project interpreter and PYTHONPATH.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
from . import full_vram_v2 as setup
from research.repairs.benchmark_full_vram_20261022.runtime import contracts_recovery_v1 as authority
from research.repairs.benchmark_full_vram_20261022.runtime import queue_gpu_v1 as leases

HERE=Path(__file__).resolve().parent
PHYSICAL_GPU=2
GPU_UUID=setup.gpu_binding.GPU_UUIDS[PHYSICAL_GPU]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',required=True,
        help='Fresh smoke receipt directory inside this training folder')
    args=parser.parse_args()
    output=Path(args.output_directory)
    output=output if output.is_absolute() else authority.ROOT/output
    setup.need(output.resolve()==output and output.is_relative_to(HERE) and not output.exists(),
        'Fresh smoke output scope inside the new training folder required')
    setup.need('torch' not in sys.modules,'A fresh interpreter before importing torch is required')
    setup.need('ARGGNN_READABLE_JOB_REF' not in os.environ,
        'This direct smoke must not claim an inherited queue reservation')
    policy_ref=authority.file_ref(authority.POLICY_PATH)
    policy=authority.check_execution_release(policy_ref)
    plan_ref=policy['approved_full_vram_workflow_ref']
    plan=authority.read_ref(plan_ref)
    am_task=next(task for task in plan['tasks'] if task['id']=='AM_treeloom_fit')
    setup.need(plan['physical_gpus']==[2,3,5]
        and am_task['physical_gpu']==PHYSICAL_GPU and am_task['gpu_uuid']==GPU_UUID
        and am_task['gpu_memory_policy']=='full_device',
        'Exact current approved AM physical GPU2 full-device assignment required')
    minimum_free=am_task['minimum_free_gpu_bytes']
    setup.need(type(minimum_free) is int and minimum_free>0,
        'Actual approved AM free-memory threshold required')
    lock_path=leases.physical_path(PHYSICAL_GPU)
    descriptor=os.open(lock_path,os.O_RDWR|os.O_NOFOLLOW)
    samples=[]
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        held=os.fstat(descriptor)
        setup.need(stat.S_ISREG(held.st_mode),'Ordinary canonical physical lock inode required')
        os.environ.update(CUDA_VISIBLE_DEVICES=GPU_UUID,ARGGNN_PHYSICAL_GPU=str(PHYSICAL_GPU),
            ARGGNN_GPU_MEMORY_POLICY='full_device',ARGGNN_READABLE_GPU_FD=str(descriptor))
        os.environ.pop('ARGGNN_MAX_CUDA_RESERVED_BYTES',None)
        import setproctitle
        setproctitle.setproctitle(setup.PROCESS_TITLE)
        import torch
        setup.need(not torch.cuda.is_initialized(),'Actual CUDA must start uninitialized')
        def resource_check(label):
            setup.need(authority.file_ref(authority.POLICY_PATH)==policy_ref,
                'Canonical authority must remain unchanged during the smoke')
            current=authority.check_execution_release(policy_ref)
            setup.need(current['approved_full_vram_workflow_ref']==plan_ref,
                'Exact current approved full-device workflow authority required')
            named=lock_path.stat()
            actual=os.fstat(descriptor)
            setup.need((held.st_dev,held.st_ino)==(actual.st_dev,actual.st_ino)==(named.st_dev,named.st_ino),
                'Held descriptor must retain the canonical physical GPU2 lock inode')
            probe=os.open(lock_path,os.O_RDWR|os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise RuntimeError('Direct smoke physical lease is not exclusively held')
            finally:
                os.close(probe)
            leases.validate_policy_environment()
            setup.need(setproctitle.getproctitle()==setup.PROCESS_TITLE,'Exact prescribed process title required')
            sample=leases.gpu_sample(PHYSICAL_GPU)
            setup.need(sample['free_bytes']>=minimum_free,
                'Actual GPU2 free memory is below the exact approved AM admission threshold')
            samples.append(dict(stage=label,**sample))
        resource_check('direct_smoke_before_CUDA_setup')
        output.mkdir()
        spec=dict(physical_gpu=PHYSICAL_GPU,gpu_uuid=GPU_UUID,gpu_memory_policy='full_device',
            full_device_vram_permission=setup.permission(PHYSICAL_GPU,GPU_UUID))
        setup.enable_gpu(torch,spec,output,resource_check)
        setup.need(torch.cuda.is_initialized(),'Actual admitted CUDA must initialize before peak reset')
        value=torch.arange(256,dtype=torch.float32,device='cuda:0')
        setup.need(value.numel()*value.element_size()==1024,'Only a 1024-byte smoke tensor is permitted')
        torch.cuda.synchronize(0)
        setup.need(value.sum().item()==32640.0,'Actual tiny CUDA tensor result required')
        del value
        torch.cuda.empty_cache()
        resource_check('direct_smoke_after_tiny_tensor')
        receipt=dict(schema='treeloom_full_device_GPU2_setup_smoke_v2',status='passed',
            created_utc=datetime.now(timezone.utc).isoformat(),physical_gpu=PHYSICAL_GPU,gpu_uuid=GPU_UUID,
            canonical_policy_ref=policy_ref,approved_workflow_ref=plan_ref,
            actual_queue_reservation=False,lease_kind='direct_exclusive_project_physical_GPU2_lock',
            lock_path=str(lock_path),lock_device=held.st_dev,lock_inode=held.st_ino,
            minimum_free_gpu_bytes=minimum_free,driver_samples=samples,
            CUDA_initially_uninitialized=True,CUDA_initialized_before_peak_reset=True,
            tiny_tensor_bytes=1024,tiny_tensor_sum=32640.0,
            models_instantiated=False,optimizer_updates=0,training_launched=False,
            helper_source_refs=setup.source_refs(),process_title=setproctitle.getproctitle())
        ref=setup.write_json(output/'smoke_completed.json',receipt)
        print(json.dumps(dict(status='passed',receipt_ref=ref),sort_keys=True))
    finally:
        os.close(descriptor)


if __name__=='__main__':
    main()

"""Explicit full-VRAM setup for unchanged scientific adapters in a fresh worker.

Only CUDA resource admission is rebound. Models, batches, optimizers, data,
training schedules, observations, final counters and native command arguments
remain in their pinned original functions. No running process is modified.
"""
from pathlib import Path
import hashlib
from research.repairs.benchmark_multigpu_20261022.scientific import gpu_binding
from .queue_native_admission_v1 import validate_worker_lease
from .queue_gpu_v1 import validate_policy_environment


def finish_gpu_setup(torch,spec):
    gpu_binding.verify_before_cuda(torch,spec)
    validate_worker_lease(first_cuda=True)
    validate_policy_environment()
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one admitted physical GPU may be visible')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_properties(0).total_memory<=0:
        raise RuntimeError('Actual admitted physical CUDA memory required')


def admit_native(build_ref,cpu_ref,*,require_gpu=False):
    from research.repairs.benchmark_multigpu_20261022.native import c4_runtime_bindings_v1 as bindings
    if not require_gpu:
        return bindings.original.ORIGINAL_ADMIT(build_ref,cpu_ref,require_gpu=False)
    bindings.require_job_device(require_title=True)
    validate_worker_lease(first_cuda=True)
    validate_policy_environment()
    bindings.native.runtime_admission(build_ref,cpu_ref,{})
    result=bindings.native.source_imports()
    bindings.m._need(result[0].cuda.device_count()==1,
        'ROOT-owned single assigned physical GPU admission required')
    bindings.m._need(result[0].cuda.get_device_properties(0).total_memory>0,
        'Actual admitted physical CUDA memory required')
    return result


def install(scientific_module):
    if scientific_module.startswith('research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.'):
        from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import common
        common.finish_gpu_setup=finish_gpu_setup
    if scientific_module=='research.repairs.benchmark_multigpu_20261022.native.queue_european_argmix_v1':
        from research.repairs.benchmark_multigpu_20261022.native import c4_runtime_bindings_v1 as bindings
        bindings.admit_native=admit_native
        bindings.STARTUP=Path(__file__).with_name('sitecustomize_full_vram_v1.py')
        bindings.STARTUP_SHA256=STARTUP_SHA256


STARTUP_SHA256='233eeb179c1bfb4b2ab320bce3312c07c9d97d40d57eb4203fc83e35dfc8c32b'

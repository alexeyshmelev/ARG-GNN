"""Full-device permission with admitted CUDA initialization before peak reset."""
import os
from research.repairs.benchmark_multigpu_20261022.scientific import gpu_binding
from research.repairs.benchmark_recovery_20261020.training.candidate_common import (
    need, file_ref, write_json,
)

GPU_BINDING_REF = dict(
    path="research/repairs/benchmark_multigpu_20261022/scientific/gpu_binding.py",
    sha256="1c989c35621c3996856cc1f1c657f6eb50ff6376b8669be2e07fb0556349623b",
)
PROCESS_TITLE = "AI Auto Research Pipeline Test"
POLICY = "full_device"


def permission(physical_gpu, gpu_uuid):
    """Bind the owner's full-VRAM permission to one authorized physical device."""
    return dict(schema="authorized_full_device_vram_v1",
        **gpu_binding.target(physical_gpu, gpu_uuid), artificial_allocator_limit=False)


def validate_permission(spec):
    need(spec.get("gpu_memory_policy") == POLICY
         and spec.get("full_device_vram_permission") == permission(spec["physical_gpu"], spec["gpu_uuid"])
         and "max_cuda_reserved_bytes" not in spec,
         "Explicit full-device permission without an artificial CUDA allocator limit required")
    need(file_ref(gpu_binding.__file__) == GPU_BINDING_REF,
         "Exact authorized physical GPU identity source required")


def source_refs():
    need(file_ref(gpu_binding.__file__) == GPU_BINDING_REF,
         "Exact authorized GPU identity source required")
    return [GPU_BINDING_REF, file_ref(__file__)]


def enable_gpu(torch, spec, output, resource_check):
    """Verify admission before CUDA; leave the device's allocator unrestricted."""
    import setproctitle
    validate_permission(spec)
    need(callable(resource_check), "Actual host RAM and storage resource guard required")
    need(os.environ.get("ARGGNN_GPU_MEMORY_POLICY") == POLICY
         and "ARGGNN_MAX_CUDA_RESERVED_BYTES" not in os.environ,
         "The worker must carry explicit full-device permission and no legacy CUDA cap")
    gpu_binding.verify_before_cuda(torch, spec)
    need(setproctitle.getproctitle() == PROCESS_TITLE,
         "Prescribed supervised title required before CUDA")
    resource_check("TreeLoom_before_full_device_CUDA")
    need(torch.cuda.device_count() == 1, "Exactly one admitted visible physical device required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # This read initializes the already admitted device. Peak reset requires it.
    need(torch.cuda.get_device_properties(0).total_memory > 0,
         "Actual admitted physical CUDA memory required before peak reset")
    torch.cuda.reset_peak_memory_stats(0)
    write_json(output / "gpu_binding.json", dict(
        **gpu_binding.target(spec["physical_gpu"], spec["gpu_uuid"]),
        logical_device="cuda:0", process_title=setproctitle.getproctitle(),
        gpu_memory_policy=POLICY, full_device_vram_permission=spec["full_device_vram_permission"]))

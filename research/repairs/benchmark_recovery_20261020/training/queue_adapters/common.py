"""Ordinary-source stage metadata and truthful late shared GPU setup."""
from pathlib import Path
import os
import tempfile
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import american_preparation as generation
from arggnn.benchmarking import treeloom_training as memory
from arggnn.benchmarking.execution import GPU0_UUID

PACKAGE = "research.repairs.benchmark_recovery_20261020.training.queue_adapters"
DIRECTORY = "research/repairs/benchmark_recovery_20261020/training/queue_adapters"
SOURCE_NAMES = ("common.py", "european_observations.py", "european_execution.py",
                "european_gru.py", "european_stage.py", "qualification.py",
                "american_prediction.py", "empirical.py")


def need(ok, message):
    if not ok:
        raise c.ContractError(message)


def source_refs():
    names = [DIRECTORY + "/" + name for name in SOURCE_NAMES]
    names += ["research/repairs/benchmark_recovery_20261020/runtime/" + name
              for name in ("binding.py", "preparation.py", "observations.py", "episodes.py", "cuts.py")]
    names += ["research/repairs/benchmark_recovery_20261020/training/" + name
              for name in ("execution.py", "decoder.py", "layout.py", "progress.py", "candidate_gru.py")]
    names += ["research/reviews/workflow_progress_diagnosis_20261018/treeloom/" + name
              for name in ("cache_v1.py", "resident_pool_v2.py", "indices_v1.py")]
    return [c.file_ref(name) for name in names]


def descriptor(module, function, spec, inherited=()):
    refs = [*inherited, *source_refs()]
    return dict(module=PACKAGE + "." + module, function=function, spec=spec,
                python=".runtime/env/bin/python",
                source_refs=list({ref["path"]: ref for ref in refs}.values()))


def _cpu_setup(spec, resource_check):
    need(callable(resource_check), "The actual worker resource guard is required")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU0_UUID
         and os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"
         and "ARGGNN_PHYSICAL_GPU" not in os.environ
         and "ARGGNN_READABLE_GPU_FD" not in os.environ,
         "Observed preparation starts without an acquired GPU reservation")
    def check(*args, **kwargs):
        resource_check(*args, **kwargs)
        need(memory._current_rss_bytes() <= spec["memory_envelope"]["max_rss_bytes"],
             "Complete preparation process RSS cap exceeded")
    check("readable_observed_preparation_setup")
    out = c.contained_path(spec["output_directory"])
    writer = generation._Writer(out, spec["max_output_bytes"], check)
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_HOME",
                 "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        path = out / (".tmp" if name in ("TMPDIR", "TMP", "TEMP") else ".cache/" + name.lower())
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(out / ".tmp")
    os.environ["CUDA_CACHE_MAXSIZE"] = str(16 * 2**20)
    import torch
    need(not torch.cuda.is_initialized(), "Observed preparation cannot inherit initialized CUDA")
    torch.set_num_threads(8)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return torch, check, writer


def finish_gpu_setup(torch):
    need(os.environ.get("ARGGNN_PHYSICAL_GPU") == "0"
         and os.environ.get("CUDA_VISIBLE_DEVICES") == GPU0_UUID,
         "The callback must acquire the actual shared physical GPU0 reservation")
    need("ARGGNN_READABLE_GPU_FD" in os.environ,
         "A genuine current readable shared reservation descriptor is required")
    need(torch.cuda.device_count() == 1, "Only physical GPU0 may be visible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
    total = torch.cuda.get_device_properties(0).total_memory
    need(0 < cap <= min(74 * 2**30, total), "The actual finite shared CUDA allocator cap is required")
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)


def _setup_gpu(spec, resource_check):
    need(callable(resource_check), "The actual prediction resource guard is required")
    need(os.environ.get("ARGGNN_PHYSICAL_GPU") == "0", "An admitted physical GPU0 worker is required")
    def check(*args, **kwargs):
        resource_check(*args, **kwargs)
        need(memory._current_rss_bytes() <= spec["memory_envelope"]["max_rss_bytes"],
             "Prediction process RSS cap exceeded")
    check("readable_prediction_setup")
    out = c.contained_path(spec["output_directory"])
    writer = generation._Writer(out, spec["max_output_bytes"], check)
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_HOME",
                 "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        path = out / (".tmp" if name in ("TMPDIR", "TMP", "TEMP") else ".cache/" + name.lower())
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(out / ".tmp")
    os.environ["CUDA_CACHE_MAXSIZE"] = str(16 * 2**20)
    import torch
    torch.set_num_threads(8)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    finish_gpu_setup(torch)
    return torch, check, writer

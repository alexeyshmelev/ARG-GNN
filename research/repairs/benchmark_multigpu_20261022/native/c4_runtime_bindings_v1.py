"""Per-job device admission around the existing unchanged static ARGMix bindings."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

from research.repairs.benchmark_resume_20261021.native.c4 import runtime_bindings_v1 as original
from .device_binding_v1 import require_job_device

release, shards, m, native = original.release, original.shards, original.m, original.native
STARTUP = Path(__file__).with_name("sitecustomize_multigpu_v1.py")
STARTUP_SHA256 = "e0fcf00e82fcd2dee4fe3724e2a78735803dd68b84a77f5b5d156ec8aa732fa2"


def admit_native(build_ref, cpu_ref, *, require_gpu=False):
    if not require_gpu:
        return original.ORIGINAL_ADMIT(build_ref, cpu_ref, require_gpu=False)
    require_job_device(require_title=True)
    native.runtime_admission(build_ref, cpu_ref, {})
    result = native.source_imports()
    m._need(result[0].cuda.device_count() == 1,
        "ROOT-owned single assigned physical GPU admission required")
    cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
    total = result[0].cuda.get_device_properties(0).total_memory
    m._need(0 < cap <= min(74 * 2**30, total),
        "Finite ROOT-admitted CUDA allocator cap required")
    result[0].cuda.set_per_process_memory_fraction(cap / total, 0)
    return result


@contextmanager
def native_python_startup(output, guard):
    """Copy the fixed plaintext startup file only into the current fresh stage."""
    payload = STARTUP.read_bytes()
    m._need(hashlib.sha256(payload).hexdigest() == STARTUP_SHA256,
        "Pinned static per-job GPU startup source required")
    directory = output / "python_startup"
    directory.mkdir()
    ref = release._write(directory / "sitecustomize.py", payload, guard)
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(directory) + (os.pathsep + previous if previous else "")
    try:
        yield ref
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous


def fit_release(**arguments):
    m._need(release.admit_native is original.ORIGINAL_ADMIT
        and shards.run_original_main is original.ORIGINAL_TRAIN_RUNNER,
        "Fresh single-call trainer I/O binding required")
    release.admit_native = admit_native
    shards.run_original_main = original.run_original_main
    try:
        return release.fit_release(**arguments)
    finally:
        release.admit_native = original.ORIGINAL_ADMIT
        shards.run_original_main = original.ORIGINAL_TRAIN_RUNNER


def infer_release(**arguments):
    m._need(release.admit_native is original.ORIGINAL_ADMIT
        and release._native_python_startup is original.ORIGINAL_STARTUP
        and release._native_local_runner is original.ORIGINAL_LOCAL_RUNNER
        and native.run_author is original.ORIGINAL_AUTHOR_RUNNER,
        "Fresh single-call inference I/O bindings required")
    release.admit_native = admit_native
    release._native_python_startup = native_python_startup
    release._native_local_runner = original.native_local_runner
    native.run_author = original.run_author
    try:
        return release.infer_release(**arguments)
    finally:
        release.admit_native = original.ORIGINAL_ADMIT
        release._native_python_startup = original.ORIGINAL_STARTUP
        release._native_local_runner = original.ORIGINAL_LOCAL_RUNNER
        native.run_author = original.ORIGINAL_AUTHOR_RUNNER

"""Exec the pinned native ARGformer qualifier without releasing its GPU lease.

This module has only standard-library imports until process-title admission.
The project scheduler supplies the reservation. No GPU is queried here.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
PREFIX = ROOT / ".runtime/conda/envs/argformer_native_v1"
PYTHON = PREFIX / "bin/python"
SCRIPT = ROOT / "scripts/qualify_official_argformer_runtime.py"
PROCESS_TITLE = "AI Auto Research Pipeline Test"
ALLOWED_GPUS = (2, 3, 4, 5)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_path(value):
    lexical = Path(value)
    if ".." in lexical.parts:
        raise ValueError("Output traversal is forbidden")
    path = lexical if lexical.is_absolute() else ROOT / lexical
    if path != path.resolve() or not path.is_relative_to(ROOT):
        raise ValueError("Canonical in-project path without symlink traversal required")
    return path


def canonical_uuid(value):
    if not isinstance(value, str):
        raise RuntimeError("GPU UUID must be a string")
    bare = value[4:] if value.startswith("GPU-") else value
    parsed = str(uuid.UUID(bare))
    if bare != parsed:
        raise RuntimeError("Canonical lowercase GPU UUID required")
    return parsed


def validate_lease(environ=None):
    """Validate the actual held inherited descriptor, without CUDA inspection."""
    environ = os.environ if environ is None else environ
    physical = environ.get("ARGGNN_PHYSICAL_GPU")
    if physical not in {str(gpu) for gpu in ALLOWED_GPUS}:
        raise RuntimeError("Eligible physical GPU 2, 3, 4 or 5 required")
    expected = ROOT / ".runtime/locks" / ("gpu_" + physical + ".lock")
    try:
        lease = Path(environ["ARGGNN_GPU_LEASE"])
        raw_fd = environ["ARGGNN_GPU_LEASE_FD"]
        if not raw_fd.isdecimal() or str(int(raw_fd)) != raw_fd:
            raise ValueError("Canonical descriptor number required")
        fd = int(raw_fd)
        descriptor = os.fstat(fd)
        disk = lease.lstat()
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError("Valid inherited GPU lease required") from error
    if (lease != expected or lease.resolve() != expected or
            not stat.S_ISREG(disk.st_mode) or not stat.S_ISREG(descriptor.st_mode) or
            (disk.st_dev, disk.st_ino) != (descriptor.st_dev, descriptor.st_ino) or
            Path("/proc/self/fd/" + str(fd)).resolve() != expected):
        raise RuntimeError("Lease descriptor, inode or canonical path differs")
    contents = os.pread(fd, 65536, 0)
    if len(contents) != descriptor.st_size:
        raise RuntimeError("Lease metadata is oversized or changed")
    metadata = json.loads(contents)
    gpu_uuid = metadata.get("uuid")
    canonical_uuid(gpu_uuid)
    if (not gpu_uuid.startswith("GPU-") or metadata.get("gpu") != int(physical) or
            environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or
            not environ.get("ARGGNN_JOB_NAME") or
            metadata.get("job") != environ.get("ARGGNN_JOB_NAME") or
            type(metadata.get("launcher_pid")) is not int or metadata["launcher_pid"] <= 0):
        raise RuntimeError("Lease UUID, physical index or job metadata differs")
    with expected.open("r+") as independent:
        try:
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(independent, fcntl.LOCK_UN)
            raise RuntimeError("Scheduler reservation is not held")
    return {"physical_gpu": int(physical), "visible_gpu_uuid": gpu_uuid,
            "canonical_uuid": canonical_uuid(gpu_uuid), "lease_path": str(expected),
            "lease_fd": fd, "lease_device": descriptor.st_dev,
            "lease_inode": descriptor.st_ino, "lease_metadata_sha256":
            hashlib.sha256(contents).hexdigest(), "launcher_metadata": metadata}


def apply_and_validate_title():
    import setproctitle
    setproctitle.setproctitle(PROCESS_TITLE)
    observed = Path("/proc/self/cmdline").read_bytes().split(b"\0")[0].decode().strip()
    if setproctitle.getproctitle() != PROCESS_TITLE or observed != PROCESS_TITLE:
        raise RuntimeError("Required actual process title missing")
    return observed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mlm", "contrastive"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args(argv)
    if Path(__file__).resolve() != ROOT / "src/arggnn/native_argformer_exec.py":
        raise RuntimeError("Only assigned project runner is admitted")
    output = project_path(args.output_dir)
    if output.exists() or not output.parent.is_dir():
        raise RuntimeError("Fresh output directory with existing project parent required")
    output.mkdir()
    try:
        lease = validate_lease()
        title = apply_and_validate_title()
        if (PREFIX.resolve() != PREFIX or not PYTHON.is_file() or
                not PYTHON.resolve().is_relative_to(PREFIX / "bin") or
                not os.access(PYTHON, os.X_OK) or SCRIPT.is_symlink() or
                SCRIPT.resolve() != SCRIPT or not SCRIPT.is_file()):
            raise RuntimeError("Fixed isolated interpreter and qualification script required")
        script_hash = sha256_file(SCRIPT)
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.update(PYTHONPATH="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
                   CONDA_PREFIX=str(PREFIX), ARGGNN_NATIVE_OUTPUT_DIR=str(output),
                   ARGGNN_NATIVE_SCRIPT_SHA256=script_hash,
                   ARGGNN_NATIVE_RUNNER_SHA256=sha256_file(__file__))
        env["PATH"] = str(PREFIX / "bin") + os.pathsep + env.get("PATH", "")
        for name, relative in {
                "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache",
                "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
                "HF_HOME": "cache/huggingface", "TORCH_HOME": "cache/torch",
                "TRITON_CACHE_DIR": "cache/triton", "CUDA_CACHE_PATH": "cache/cuda",
                "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
                "NUMBA_CACHE_DIR": "cache/numba", "MPLCONFIGDIR": "config/matplotlib",
                "WANDB_DIR": "logs/wandb", "WANDB_CACHE_DIR": "cache/wandb",
                "WANDB_CONFIG_DIR": "config/wandb"}.items():
            directory = output / relative
            directory.mkdir(parents=True, exist_ok=True)
            env[name] = str(directory)
        os.set_inheritable(lease["lease_fd"], True)
        if not os.get_inheritable(lease["lease_fd"]):
            raise RuntimeError("Lease cannot be inherited across exec")
        if validate_lease() != lease:
            raise RuntimeError("Lease changed before exec")
        command = [str(PYTHON), "-B", str(SCRIPT), "--mode", args.mode,
                   "--output-dir", str(output)]
        if args.backward:
            command.append("--backward")
        record = {"kind": "native_argformer_exec_admission", "pid": os.getpid(),
                  "lease": lease, "title_before_exec": title,
                  "interpreter": str(PYTHON), "script": str(SCRIPT),
                  "script_sha256": script_hash, "runner_sha256": sha256_file(__file__),
                  "command": command, "lease_inheritable": True,
                  "native_forward_completed": False}
        (output / "exec_admission.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"event": "native_argformer_exec", "mode": args.mode,
                          "output_dir": str(output), "lease_fd": lease["lease_fd"]}), flush=True)
        os.chdir(ROOT)
        os.execve(PYTHON, command, env)
    except BaseException as error:
        (output / "exec_failure.json").write_text(json.dumps({
            "status": "exec_admission_failed", "error_type": type(error).__name__,
            "error": str(error), "native_forward_completed": False,
            "exit_code": 1}, indent=2) + "\n")
        (output / "exit_status").write_text("1\n")
        raise


if __name__ == "__main__":
    main()

"""Fresh GPU0 worker family with deferred real admission and actual child waits."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import contracts as c

GPU0_UUID = "GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
TITLE = "AI Auto Research Pipeline Test"
SCHEMA = "benchmark_guarded_worker_spec_v1"
MAX_OUTPUT = 64 * 2**30
MODULE = "arggnn.benchmarking.execution"
CALLABLES = {
    ("arggnn.benchmarking.argformer_execution", "run_contrastive"),
    ("arggnn.benchmarking.argformer_prediction", "run_embedding"),
    ("arggnn.benchmarking.argformer_prediction", "run_retrieval"),
}


def validate_spec(ref):
    spec = c.read_ref(ref)
    if (spec.get("schema") != SCHEMA or spec.get("resource") != "gpu0"
            or (type(spec.get("physical_gpu")) is not int or spec["physical_gpu"] != 0) or spec.get("gpu_uuid") != GPU0_UUID
            or not isinstance(spec.get("job_id"), str) or not spec["job_id"]
            or (spec.get("callable_module"), spec.get("callable_name")) not in CALLABLES):
        raise c.ContractError("A registered fresh GPU0 scientific worker is required")
    for key in ("max_output_bytes", "max_seconds", "max_rss_bytes", "minimum_free_gpu_bytes", "max_cuda_reserved_bytes"):
        if type(spec.get(key)) is not int or spec[key] <= 0:
            raise c.ContractError("Finite positive worker caps are required")
    if spec["max_cuda_reserved_bytes"] > 74 * 2**30:
        raise c.ContractError("Native allocator cap must not exceed74GiB")
    if spec["max_output_bytes"] > MAX_OUTPUT:
        raise c.ContractError("Scientific output cap cannot exceed64GiB")
    python = c.contained_path(spec["python"])
    if not python.is_file() or "python" not in python.name:
        raise c.ContractError("A native project Python interpreter is required")
    output = c.contained_path(spec["output_directory"])
    if (output == c.ROOT or not output.is_relative_to(c.ROOT / "experiments/results")
            or output.resolve() != output):
        raise c.ContractError("A dedicated canonical experiments/results scope is required")
    if not isinstance(spec.get("scientific_spec"), dict) or not spec.get("source_refs"):
        raise c.ContractError("Source-pinned native scientific arguments are required")
    native_output = c.contained_path(spec["scientific_spec"].get("output_directory", ""))
    if native_output != output / "native":
        raise c.ContractError("Native artifacts must stay in the assigned output/native directory")
    for record in spec["source_refs"]:
        path = c.contained_path(record["path"])
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 2**20
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise c.ContractError("Scientific worker source changed")
    return spec


def worker_argv(ref, spec=None):
    spec = validate_spec(ref) if spec is None else spec
    return (str(c.contained_path(spec["python"])), "-B", "-s", "-m", MODULE,
            "--execute", "--job-ref", json.dumps(ref, sort_keys=True, separators=(",", ":")))


def _approved(ref, spec):
    policy_ref = c.file_ref(c.POLICY_PATH)
    policy = c.check_execution_release(policy_ref)
    matches = [item for item in policy.get("approved_jobs", [])
               if item.get("job_id") == spec["job_id"] and item.get("input_ref") == ref]
    if len(matches) != 1:
        raise c.ContractError("The exact current worker input needs one canonical approval")
    job = c.validate_admission(matches[0]["admission_receipt_ref"])
    if (dict(job.input_ref) != ref or job.job_id != spec["job_id"]
            or job.argv != worker_argv(ref, spec) or job.resource != "gpu0"
            or job.cwd != str(c.ROOT) or job.max_seconds != spec["max_seconds"]):
        raise c.ContractError("Approved command, input, resource and wall cap differ")
    c.validate_execution_policy(policy_ref, job)
    return policy


def scope_bytes(output):
    """Inspect only this worker scope, allowing confined native latest symlinks."""
    output = Path(output)
    total = 0
    for directory, dirs, files in os.walk(output, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            if path.is_symlink() and not path.resolve().is_relative_to(output):
                raise RuntimeError("Native output symlink escaped the assigned scope")
            total += max(info.st_size, info.st_blocks * 512) + 4096
    return total


def _own_group_rss(group):
    total = 0
    page = os.sysconf("SC_PAGE_SIZE")
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                total += int((path / "statm").read_text().split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue
    return total


def _inventory_check(minimum):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=20)
    records = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    zero = [row for row in records if int(row[0].strip()) == 0]
    if (len(zero) != 1 or zero[0][1].strip() != GPU0_UUID
            or int(zero[0][2].strip()) * 2**20 < minimum):
        raise RuntimeError("Physical GPU0 identity or free memory does not satisfy this job")


@contextmanager
def gpu0_lease(spec):
    locks = c.ROOT / ".runtime/locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / "scheduler.lock").open("a+") as scheduler:
        fcntl.flock(scheduler, fcntl.LOCK_EX)
        fd = os.open(locks / "gpu_0.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _inventory_check(spec["minimum_free_gpu_bytes"])
            record = dict(gpu=0, uuid=GPU0_UUID, launcher_pid=os.getpid(),
                          job=spec["job_id"], started_unix=time.time())
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(record, sort_keys=True).encode())
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            raise
    try:
        yield fd
    finally:
        os.close(fd)


def _stop_group(process):
    """Signal only the new session/group created for this exact native child."""
    for sig, delay in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        end = time.monotonic() + delay
        while time.monotonic() < end:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.1)


def _child(ref):
    spec = validate_spec(ref)
    _approved(ref, spec)
    import setproctitle
    setproctitle.setproctitle(TITLE)
    fd = int(os.environ["ARGGNN_GPU_LEASE_FD"])
    lease = c.ROOT / ".runtime/locks/gpu_0.lock"
    info, named = os.fstat(fd), lease.stat()
    if ((info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU0_UUID
            or os.environ.get("ARGGNN_PHYSICAL_GPU") != "0"):
        raise RuntimeError("Native worker must inherit the actual fresh GPU0 lease")
    metadata = json.loads(os.pread(fd, 65536, 0))
    if metadata.get("launcher_pid") != os.getppid() or metadata.get("job") != spec["job_id"]:
        raise RuntimeError("Native child must belong to this fresh lease-owning supervisor")
    probe = os.open(lease, os.O_RDWR | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError("Actual GPU0 lease is not held")
    finally:
        os.close(probe)
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (spec["max_output_bytes"], spec["max_output_bytes"]))
    output = c.contained_path(spec["output_directory"])
    deadline = float(os.environ["ARGGNN_BENCHMARK_DEADLINE"])
    last_measurement = [-float("inf")]
    def tick(*_args, **_kwargs):
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("Scientific wall cap reached")
        if now - last_measurement[0] < 5:
            return
        last_measurement[0] = now
        if scope_bytes(output) > spec["max_output_bytes"]:
            raise RuntimeError("Scientific own-output cap exceeded")
        if _own_group_rss(os.getpgrp()) > spec["max_rss_bytes"]:
            raise RuntimeError("Scientific RSS cap exceeded")
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            if torch.cuda.memory_reserved(0) > spec["max_cuda_reserved_bytes"]:
                raise RuntimeError("Native CUDA allocator cap exceeded")
    tick()
    native = importlib.import_module(spec["callable_module"])
    return getattr(native, spec["callable_name"])(spec["scientific_spec"], resource_check=tick)


def execute(ref):
    spec = validate_spec(ref)
    policy = _approved(ref, spec)
    output = c.contained_path(spec["output_directory"])
    if output.exists():
        raise RuntimeError("One-shot scientific output already exists, no automatic retry")
    from arggnn.resources import storage_bytes
    used = storage_bytes()
    ceiling = min(policy.get("storage_ceiling_bytes", 980000000000),
                  policy.get("human_storage_limit_bytes", 1000000000000) - 20000000000)
    approved_caps = policy.get("approved_new_storage_cap_bytes", spec["max_output_bytes"])
    if (type(approved_caps) is not int or approved_caps < spec["max_output_bytes"]
            or (len(policy["approved_jobs"]) > 1 and "approved_new_storage_cap_bytes" not in policy)):
        raise RuntimeError("Current policy must bind the total approved new output caps")
    if used + approved_caps >= ceiling:
        raise RuntimeError("Actual project usage plus full approved caps reaches storage ceiling")
    with gpu0_lease(spec) as fd:
        _approved(ref, spec)
        output.mkdir(parents=True, exist_ok=False)
        (output / "storage_admission.json").write_text(json.dumps(dict(
            measured_project_bytes=used, approved_new_output_caps_bytes=approved_caps,
            projected_upper_bytes=used + approved_caps, ceiling_bytes=ceiling,
            measurement_unix=time.time(), job_id=spec["job_id"],
            policy_ref=c.file_ref(c.POLICY_PATH)), indent=2) + "\n")
        temporary = output / "tmp"
        temporary.mkdir()
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=GPU0_UUID, ARGGNN_PHYSICAL_GPU="0",
                   ARGGNN_GPU_LEASE=str(c.ROOT / ".runtime/locks/gpu_0.lock"),
                   ARGGNN_GPU_LEASE_FD=str(fd), ARGGNN_JOB_NAME=spec["job_id"],
                   ARGGNN_MAX_CUDA_RESERVED_BYTES=str(spec["max_cuda_reserved_bytes"]),
                   TMPDIR=str(temporary), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   PYTHONPATH=str(c.ROOT)+":"+str(c.ROOT/"src")+":"+str(c.ROOT/"scripts"))
        deadline = time.monotonic() + spec["max_seconds"]
        env["ARGGNN_BENCHMARK_DEADLINE"] = repr(deadline)
        command = [str(c.contained_path(spec["python"])), "-B", "-s", "-m", MODULE,
                   "--child", "--job-ref", json.dumps(ref, sort_keys=True)]
        error = None
        cancelled = []
        handlers = {sig: signal.signal(sig, lambda number, _frame: cancelled.append(number))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        with (output / "native.log").open("xb") as log:
            try:
                process = subprocess.Popen(command, cwd=c.ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
            except BaseException:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
                raise
            last_measurement = -float("inf")
            try:
                while process.poll() is None:
                    if cancelled:
                        raise InterruptedError(f"Supervisor cancelled by signal{cancelled[0]}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Scientific wall cap reached")
                    now = time.monotonic()
                    if now - last_measurement >= 5:
                        last_measurement = now
                        if _own_group_rss(process.pid) > spec["max_rss_bytes"]:
                            raise RuntimeError("Scientific RSS cap exceeded")
                        if scope_bytes(output) > spec["max_output_bytes"]:
                            raise RuntimeError("Scientific own-output cap exceeded")
                    time.sleep(1)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if process.poll() is not None:
                    status = process.wait()
                    _stop_group(process)
                else:
                    _stop_group(process)
                    status = process.wait()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        result = dict(job_id=spec["job_id"], actual_direct_wait=True,
                      actual_direct_returncode=status, supervisor_error=error,
                      scientific_completion_claimed=False, automatic_retry=False,
                      output_bytes=scope_bytes(output))
        (output / "actual_wait.json").write_text(json.dumps(result, indent=2)+"\n")
        if error is not None or status != 0:
            raise RuntimeError(f"Native worker failed: {result}")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job-ref", required=True)
    args = parser.parse_args(argv)
    if args.execute and args.child:
        parser.error("Choose one worker mode")
    ref = json.loads(args.job_ref)
    if args.child:
        result = _child(ref)
    elif args.execute:
        result = execute(ref)
    else:
        result = dict(job_id=validate_spec(ref)["job_id"], execution_authorized=False, jobs_started=0)
    if result is not None:
        print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

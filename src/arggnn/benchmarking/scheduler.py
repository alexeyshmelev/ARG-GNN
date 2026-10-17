"""Durable scheduling of already admitted guarded commands.

This module has no scientific or resource-admission code. Receipt and execution
policy validators belong to the ROOT contract layer. Planning is the default;
execution always needs an explicit policy reference and an approval validator.
A zero subprocess return code records only an actual OS wait, not scientific
completion, owned drain, storage close, or permission to retry.

The controller lock is a logical queue lock, never a physical GPU lease. An
unknown prior start blocks its resource lane until ROOT resolves it outside
this scheduler; files named completed.json are never inspected.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "arggnn.admitted_benchmark_queue.v1"
STATUSES = {"queued", "started", "succeeded", "failed", "rejected", "unknown"}


class SchedulerBusy(RuntimeError):
    """Another controller or short metadata transaction owns the logical lock."""


def _queue_path(value: str | Path) -> Path:
    """Canonical queue-owned metadata path, without creating directories."""
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if (".." in path.parts or path.name == "gpu_0.lock"
            or not path.is_relative_to(ROOT)):
        raise ValueError("Queue metadata and logical locks must remain inside the repository")
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Queue metadata cannot escape the repository through a symlink")
    return path


def _reference(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("An exact admission/input/policy path and SHA256 are required")
    path, digest = value["path"], value["sha256"]
    if (not isinstance(path, str) or not path or "\0" in path
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Invalid immutable reference")
    return {"path": path, "sha256": digest}


@dataclass(frozen=True)
class AdmittedJob:
    """Exact command returned by the ROOT admission-receipt validator."""

    job_id: str
    resource: str
    argv: tuple[str, ...]
    cwd: str
    input_ref: Mapping[str, str]
    admission_receipt_ref: Mapping[str, str]
    max_seconds: int

    def __post_init__(self):
        if (not isinstance(self.job_id, str) or not self.job_id or len(self.job_id) > 256
                or self.resource not in {"gpu0", "cpu"}):
            raise ValueError("A bounded job id and resource gpu0 or cpu are required")
        if (not isinstance(self.argv, (tuple, list)) or not self.argv
                or any(not isinstance(a, str) or "\0" in a for a in self.argv)
                or not self.argv[0] or sum(len(a) for a in self.argv) > 32768):
            raise ValueError("Exact bounded admitted argv required; shell commands are unsupported")
        if not isinstance(self.cwd, str) or not Path(self.cwd).is_absolute():
            raise ValueError("The admitted working directory must be absolute")
        if type(self.max_seconds) is not int or self.max_seconds <= 0:
            raise ValueError("The admitted worker must have a positive finite wall cap")
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "input_ref", _reference(self.input_ref))
        object.__setattr__(self, "admission_receipt_ref", _reference(self.admission_receipt_ref))

    def as_dict(self) -> dict:
        return dict(job_id=self.job_id, resource=self.resource, argv=list(self.argv),
                    cwd=self.cwd, input_ref=dict(self.input_ref),
                    admission_receipt_ref=dict(self.admission_receipt_ref),
                    max_seconds=self.max_seconds)


AdmissionValidator = Callable[[Mapping[str, str]], AdmittedJob]
PolicyValidator = Callable[[Mapping[str, str], AdmittedJob], None]


class PersistentScheduler:
    """One GPU0 lane plus a bounded number of independent CPU lanes.

    The supplied runner must have subprocess.Popen semantics (poll and wait).
    stdout/stderr and the environment are inherited; exact guarded workers keep
    their own GPU lease, title, deadlines, accounting, drain and result checks.
    No background worker, monitor, auto-retry, or automatic execution is created.
    """

    def __init__(self, state_path: str | Path, controller_lock_path: str | Path, *,
                 validate_admission: AdmissionValidator,
                 validate_execution_policy: PolicyValidator | None = None,
                 cpu_parallelism: int = 1, runner: Callable = subprocess.Popen,
                 poll_seconds: float = .2, sleep: Callable = time.sleep,
                 clock: Callable = time.time, max_jobs: int = 1024,
                 max_state_bytes: int = 8 * 2**20):
        self.state_path = _queue_path(state_path)
        self.controller_lock_path = _queue_path(controller_lock_path)
        self.state_lock_path = self.state_path.with_name(self.state_path.name + ".lock")
        if (len({self.state_path, self.controller_lock_path, self.state_lock_path}) != 3
                or any(p.name == "gpu_0.lock" for p in
                       (self.state_path, self.controller_lock_path, self.state_lock_path))):
            raise ValueError("Queue metadata/locks must be distinct from the physical GPU0 lease")
        if (type(cpu_parallelism) is not int or cpu_parallelism < 1
                or type(max_jobs) is not int or max_jobs < 1
                or type(max_state_bytes) is not int or max_state_bytes < 1024
                or not isinstance(poll_seconds, (int, float))
                or not math.isfinite(poll_seconds) or poll_seconds < 0):
            raise ValueError("Finite queue, CPU and polling bounds are required")
        if not callable(validate_admission) or (validate_execution_policy is not None
                                               and not callable(validate_execution_policy)):
            raise TypeError("ROOT admission/policy validators must be callable")
        self.validate_admission = validate_admission
        self.validate_execution_policy = validate_execution_policy
        self.cpu_parallelism = cpu_parallelism
        self.runner, self.sleep, self.clock = runner, sleep, clock
        self.poll_seconds, self.max_jobs, self.max_state_bytes = poll_seconds, max_jobs, max_state_bytes

    @contextmanager
    def _lock(self, path: Path, *, wait: bool = False):
        # Only queue-owned empty regular lock files are opened. Physical leases
        # are exclusively the responsibility of the existing admitted workers.
        # First enqueue/approved execution may start from a fresh checkout.
        # Planning never reaches this write path or creates directories.
        path = _queue_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.parent.resolve().is_relative_to(ROOT):
            raise ValueError("Queue lock parent escaped the repository")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != 0:
                raise ValueError("Queue lock must be an empty singly linked regular file")
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if not wait or time.monotonic() >= deadline:
                        raise SchedulerBusy(str(path)) from exc
                    time.sleep(.01)
            yield
        finally:
            os.close(fd)

    def _read(self) -> dict:
        try:
            fd = os.open(self.state_path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return {"schema": SCHEMA, "controller_lock_path": str(self.controller_lock_path), "jobs": []}
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_size > self.max_state_bytes):
                raise ValueError("Invalid or oversized queue metadata")
            state = json.loads(stream.read())
        if (state.get("schema") != SCHEMA
                or state.get("controller_lock_path") != str(self.controller_lock_path)
                or not isinstance(state.get("jobs"), list)
                or len(state["jobs"]) > self.max_jobs):
            raise ValueError("Invalid queue schema or job bound")
        ids, inputs, receipts = set(), set(), set()
        for entry in state["jobs"]:
            job = AdmittedJob(**entry["admission"])
            key = json.dumps(dict(job.input_ref), sort_keys=True)
            receipt = json.dumps(dict(job.admission_receipt_ref), sort_keys=True)
            if (entry["status"] not in STATUSES or job.job_id in ids
                    or key in inputs or receipt in receipts):
                raise ValueError("Invalid or duplicate persisted queue entry")
            ids.add(job.job_id); inputs.add(key); receipts.add(receipt)
        return state

    def _write(self, state: dict):
        encoded = (json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if len(encoded) > self.max_state_bytes:
            raise ValueError("Queue metadata limit exceeded")
        fd, name = tempfile.mkstemp(prefix=self.state_path.name + ".", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
            os.replace(name, self.state_path)
            parent = os.open(self.state_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    def _update(self, callback: Callable[[dict], object]):
        with self._lock(self.state_lock_path, wait=True):
            state = self._read()
            result = callback(state)
            self._write(state)
            return result

    def _admitted(self, receipt_ref: Mapping[str, str]) -> AdmittedJob:
        receipt_ref = _reference(receipt_ref)
        job = self.validate_admission(dict(receipt_ref))
        if not isinstance(job, AdmittedJob) or dict(job.admission_receipt_ref) != receipt_ref:
            raise ValueError("The admission validator must return this exact receipt's AdmittedJob")
        return AdmittedJob(**job.as_dict())  # Defensive snapshot of validator-owned mappings.

    def enqueue(self, admission_receipt_ref: Mapping[str, str]) -> dict:
        """Validate a receipt and persist a never-started job; never launch it."""
        job = self._admitted(admission_receipt_ref)
        payload = job.as_dict()

        def add(state):
            for entry in state["jobs"]:
                old = entry["admission"]
                if old["job_id"] == job.job_id:
                    if old != payload:
                        raise ValueError("A job id cannot be rebound to another admission")
                    return dict(entry)
                if (old["input_ref"] == payload["input_ref"]
                        or old["admission_receipt_ref"] == payload["admission_receipt_ref"]):
                    raise ValueError("An admitted input/receipt cannot be queued under another id")
            if len(state["jobs"]) >= self.max_jobs:
                raise ValueError("Queue job bound exceeded")
            entry = dict(admission=payload, status="queued", queued_at=self.clock(),
                         started_at=None, finished_at=None, pid=None,
                         actual_returncode=None, error=None)
            state["jobs"].append(entry)
            return dict(entry)

        return self._update(add)

    def jobs(self) -> list[dict]:
        """Atomic metadata snapshot, with no completion-file or process probes."""
        return self._read()["jobs"]

    def plan(self) -> dict:
        jobs = self.jobs()
        return {"mode": "plan", "execution_authorized": False, "jobs": jobs,
                "unknown_jobs": [j["admission"]["job_id"] for j in jobs
                                 if j["status"] in {"started", "unknown"}]}

    def _set(self, job_id: str, **values):
        def change(state):
            entry = next(j for j in state["jobs"] if j["admission"]["job_id"] == job_id)
            entry.update(values)
        self._update(change)

    def _recover_unknown_starts(self):
        def recover(state):
            for entry in state["jobs"]:
                if entry["status"] == "started":
                    entry.update(status="unknown",
                                 error="Controller interrupted before an actual subprocess wait was recorded")
        self._update(recover)

    def run(self, *, execute: bool = False,
            execution_policy_ref: Mapping[str, str] | None = None,
            stop_requested: Callable[[], bool] = lambda: False) -> dict:
        """Plan by default; explicitly approved execution waits on actual children.

        Recovery never relaunches a started/failed/rejected job. An unknown GPU
        start blocks GPU0, and unknown CPU starts consume CPU lane capacity.
        ROOT must resolve unknowns using genuine external wait/drain evidence.
        """
        if type(execute) is not bool:
            raise TypeError("Execution requires an explicit boolean flag")
        if not execute:
            return self.plan()
        if execution_policy_ref is None or self.validate_execution_policy is None:
            raise PermissionError("Explicit approved ROOT policy and execution-policy validator required")
        policy_ref = _reference(execution_policy_ref)
        running: dict[str, object] = {}
        launched, approval_error = [], None
        stop = False
        with self._lock(self.controller_lock_path):
            self._recover_unknown_starts()
            while True:
                for job_id, process in list(running.items()):
                    if process.poll() is not None:
                        # wait(), never an output file, is the terminal authority.
                        returncode = process.wait()
                        if type(returncode) is not int:
                            raise TypeError("Runner wait must return an actual integer subprocess status")
                        self._set(job_id, status="succeeded" if returncode == 0 else "failed",
                                  actual_returncode=returncode, finished_at=self.clock())
                        del running[job_id]
                stop = stop or bool(stop_requested())
                snapshot = self.jobs()
                occupied = [j for j in snapshot if j["status"] in {"started", "unknown"}]
                gpu_busy = any(j["admission"]["resource"] == "gpu0" for j in occupied)
                cpu_busy = sum(j["admission"]["resource"] == "cpu" for j in occupied)
                dispatched = False
                for entry in snapshot:
                    if stop or approval_error is not None or entry["status"] != "queued":
                        continue
                    old = entry["admission"]
                    if old["resource"] == "gpu0" and gpu_busy:
                        continue
                    if old["resource"] == "cpu" and cpu_busy >= self.cpu_parallelism:
                        continue
                    job_id = old["job_id"]
                    try:
                        job = self._admitted(old["admission_receipt_ref"])
                        if job.as_dict() != old:
                            raise ValueError("Admitted command/input changed since enqueue")
                    except Exception as exc:
                        self._set(job_id, status="rejected", error=f"{type(exc).__name__}: {exc}")
                        continue
                    try:
                        approved = self.validate_execution_policy(dict(policy_ref), job)
                        if approved is not None and approved is not True:
                            raise PermissionError("ROOT execution policy was not approved")
                    except Exception as exc:
                        approval_error = exc
                        break
                    # Persist the start before spawning: a crash in this window
                    # conservatively becomes unknown and can never cause replay.
                    self._set(job_id, status="started", started_at=self.clock())
                    try:
                        process = self.runner(list(job.argv), cwd=job.cwd)
                    except Exception as exc:
                        self._set(job_id, status="failed", finished_at=self.clock(),
                                  error=f"Launch failed: {type(exc).__name__}: {exc}")
                        continue
                    running[job_id] = process
                    self._set(job_id, pid=process.pid)
                    launched.append(job_id); dispatched = True
                    if job.resource == "gpu0":
                        gpu_busy = True
                    else:
                        cpu_busy += 1
                if not running:
                    # No background wait: blocked/unknown jobs remain visible.
                    if approval_error is not None:
                        raise PermissionError("ROOT execution policy rejected dispatch") from approval_error
                    if not dispatched:
                        break
                if running:
                    self.sleep(self.poll_seconds)
        return {"mode": "executed", "launched_jobs": launched, "jobs": self.jobs(),
                "scientific_completion_claimed": False}

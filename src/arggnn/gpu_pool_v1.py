"""Shared inherited-descriptor GPU pool. Standard library only; no CUDA imports.

The production CLI has a fixed project root, real inventory and audited modules.
Constructor injection exists solely for contained CPU/fake-inventory tests.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
ALLOWED_GPUS = (2, 3, 4, 5)
MODULES = frozenset({
    "arggnn.biological_candidate",
    "arggnn.biological_gst_control_v1",
    "arggnn.biological_mechanism_controls_v1",
    "arggnn.official_argmix_biological",
    "arggnn.train_official_argformer_biological",
    "arggnn.native_argformer_exec",
})
MAX_REQUEST = 65536
MAX_JOBS = 128
MAX_REPLY = 32 * 1024 * 1024
MIN_CONTROLLER_RESERVE = 64 * 1024 * 1024
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def canonical_path(root, value):
    p = Path(value)
    if ".." in p.parts:
        raise ValueError("Path traversal is forbidden")
    p = p if p.is_absolute() else root / p
    if p != p.resolve() or not p.is_relative_to(root):
        raise ValueError("Canonical contained path without symlinks required")
    return p


def read_json(path, maximum=MAX_REPLY):
    with Path(path).open("rb") as f:
        data = f.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("JSON artifact exceeds its size limit")
    return json.loads(data)


def atomic_json(path, value):
    data = encoded(value) + b"\n"
    if len(data) > MAX_REPLY:
        raise ValueError("Registry exceeds its size limit")
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def lock_file(root, name):
    directory = canonical_path(root, root / ".runtime/locks")
    directory.mkdir(parents=True, exist_ok=True)
    p = canonical_path(root, directory / name)
    fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("Regular canonical lock file required")
    return os.fdopen(fd, "r+")


def canonical_gpu_uuid(value):
    if not isinstance(value, str) or not value.startswith("GPU-"):
        raise ValueError("GPU UUID prefix required")
    if str(uuid.UUID(value[4:])) != value[4:]:
        raise ValueError("Canonical lowercase GPU UUID required")
    return value


def real_inventory():
    """Two driver queries, no CUDA context; retain individual compute PIDs."""
    def query(fields, option):
        result = subprocess.run(
            ["nvidia-smi", option + "=" + fields, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15, shell=False)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    devices = query("index,uuid,memory.used,memory.total,utilization.gpu", "--query-gpu")
    apps = query("gpu_uuid,pid,process_name,used_memory", "--query-compute-apps")
    by_uuid = {}
    result = []
    for row in devices:
        if len(row) != 5:
            raise RuntimeError("Incomplete GPU inventory")
        idx, device_uuid, used, total, util = [v.strip() for v in row]
        device_uuid = canonical_gpu_uuid(device_uuid)
        record = dict(index=int(idx), uuid=device_uuid, used_mib=int(used),
                      total_mib=int(total), utilization=int(util), compute=[])
        if device_uuid in by_uuid:
            raise RuntimeError("Duplicate GPU UUID")
        by_uuid[device_uuid] = record
        result.append(record)
    for row in apps:
        if len(row) != 4:
            raise RuntimeError("Incomplete compute PID inventory")
        device_uuid, pid, name, used = [v.strip() for v in row]
        if device_uuid not in by_uuid:
            raise RuntimeError("Compute UUID missing from inventory")
        by_uuid[device_uuid]["compute"].append(
            dict(pid=int(pid), name=name, used_mib=int(used)))
    return result


def checked_inventory(provider):
    rows = provider()
    if not isinstance(rows, list):
        raise ValueError("Inventory must be a list")
    result = {}
    seen_uuid = set()
    for raw in rows:
        row = dict(raw)
        idx = integer(row["index"], "physical GPU")
        canonical_gpu_uuid(row["uuid"])
        if idx in result or row["uuid"] in seen_uuid:
            raise ValueError("Duplicate physical device or UUID")
        integer(row["used_mib"], "observed GPU memory")
        integer(row["total_mib"], "GPU total memory", 1)
        integer(row["utilization"], "GPU utilization")
        if row["used_mib"] > row["total_mib"] or row["utilization"] > 100:
            raise ValueError("Impossible GPU inventory")
        apps = row["compute"]
        if not isinstance(apps, list):
            raise ValueError("Compute PID inventory must be complete")
        seen_pids = set()
        for app in apps:
            integer(app["pid"], "compute PID", 1)
            integer(app["used_mib"], "compute memory")
            if app["pid"] in seen_pids:
                raise ValueError("Duplicate compute PID")
            seen_pids.add(app["pid"])
        result[idx] = row
        seen_uuid.add(row["uuid"])
    if any(idx not in result for idx in ALLOWED_GPUS):
        raise ValueError("Complete eligible-device inventory required")
    return result


def proc_identity(pid):
    base = Path("/proc") / str(pid)
    raw = (base / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return dict(pid=pid, uid=base.stat().st_uid, state=fields[0],
                ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]),
                start_ticks=int(fields[19]))


def enable_child_subreaper():
    """Keep orphaned descendants attached to this Linux controller, including setsid."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Linux child subreaper admission failed")
    observed = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(observed), 0, 0, 0) != 0 or observed.value != 1:
        raise RuntimeError("Linux child subreaper verification failed")


def _pidfd_syscall(number, *arguments):
    if sys.platform != "linux" or os.uname().machine not in {"x86_64", "aarch64"}:
        raise RuntimeError("Safe pidfd syscall wrapper requires Linux x86_64/aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def send_pidfd(fd, signum):
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(fd, signum, None, 0)
    else:
        _pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(signum),
                      ctypes.c_void_p(), ctypes.c_uint(0))


class ProcReader:
    """Strictly inspect launched/recorded descendants; never trust process titles.

    Accessible unknown inode holders conservatively block drain/admission.
    Inaccessible unrelated CPU processes are reported, not adopted or signalled.
    Subreaping preserves the parent relationship for escaped orphaned descendants.
    """
    def __init__(self):
        self.uninspected_unrelated = []
        self.controller_children = []

    def __call__(self, lease, controller_pid, known=None):
        known = known or {}
        inode = os.fstat(lease.fileno())
        result = {}
        self.uninspected_unrelated = []
        with os.scandir("/proc") as entries:
            for count, entry in enumerate(entries):
                if count > 100000:
                    raise RuntimeError("Process inventory limit exceeded")
                if not entry.name.isdecimal():
                    continue
                pid = int(entry.name)
                try:
                    item = proc_identity(pid)
                    item["lease_fd_present"] = False
                    result[pid] = item
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except PermissionError:
                    if pid in known:
                        raise RuntimeError(f"Cannot inspect known descendant PID {pid}")
                    self.uninspected_unrelated.append(pid)
        # /proc enumeration and per-PID stat reads are not an atomic snapshot.
        # Subreaped orphans created during enumeration must be admitted to the
        # table before an exited leader can release its reservation.
        def children_now():
            children = set()
            with os.scandir(Path("/proc") / str(controller_pid) / "task") as tasks:
                for task in tasks:
                    try:
                        contents = (Path(task.path) / "children").read_text()
                    except FileNotFoundError:
                        continue
                    for raw_pid in contents.split():
                        pid = int(raw_pid)
                        if pid <= 0:
                            raise RuntimeError("Invalid controller child identity")
                        children.add(pid)
            return children
        previous_children = None
        for _ in range(8):
            direct_children = children_now()
            for pid in direct_children:
                try:
                    item = proc_identity(pid)
                    item["lease_fd_present"] = False
                    result[pid] = item
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except PermissionError:
                    raise RuntimeError(f"Cannot inspect own controller child PID {pid}")
            # A child can fork/exit while its stat is read. Recheck AFTER the
            # identity merge, so a newly subreaped grandchild forces another pass.
            after_identity = children_now()
            if direct_children == after_identity == previous_children:
                break
            previous_children = after_identity
        else:
            raise RuntimeError("Controller child list did not stabilise")
        self.controller_children = sorted(after_identity)
        strict = {controller_pid}
        sessions = set()
        for pid, saved in known.items():
            item = result.get(pid)
            if item is None:
                continue
            if saved["start_ticks"] is None or item["start_ticks"] == saved["start_ticks"]:
                if item["uid"] != os.getuid():
                    raise RuntimeError(f"Known descendant PID {pid} changed UID")
                strict.add(pid)
                if saved.get("leader"):
                    sessions.add(pid)
        changed = True
        while changed:
            changed = False
            for pid, item in result.items():
                related = (item["ppid"] in strict or
                           (item["sid"] in sessions and item["pgid"] == item["sid"]))
                if related and pid not in strict:
                    if item["uid"] != os.getuid():
                        raise RuntimeError(f"Own descendant PID {pid} changed UID")
                    strict.add(pid)
                    changed = True
        for pid, item in result.items():
            item["controller_descendant"] = pid in strict and pid != controller_pid
            if pid == controller_pid or item["uid"] != os.getuid() or item["state"] in {"Z", "X"}:
                continue
            try:
                with os.scandir(Path("/proc") / str(pid) / "fd") as descriptors:
                    for count, entry in enumerate(descriptors):
                        if count > 65536:
                            raise RuntimeError("Process FD inventory limit exceeded")
                        try:
                            observed = os.stat(entry.path)
                        except FileNotFoundError:
                            continue
                        if (observed.st_dev, observed.st_ino) == (inode.st_dev, inode.st_ino):
                            item["lease_fd_present"] = True
                            break
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                if pid in strict:
                    raise RuntimeError(f"Cannot inspect own descendant PID {pid} descriptors")
                self.uninspected_unrelated.append(pid)
        self.uninspected_unrelated = sorted(set(self.uninspected_unrelated))
        return result

def snapshot_record(root, path, expected_sha):
    path = canonical_path(root, path)
    data = path.read_bytes()
    if len(data) > 1024 * 1024 or not re.fullmatch("[0-9a-f]{64}", expected_sha):
        raise ValueError("Bounded hashed storage snapshot required")
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Storage snapshot SHA256 differs")
    r = json.loads(data)
    if (r.get("schema") != "biological_candidate_storage_snapshot_v1"
            or r.get("status") != "complete" or type(r.get("returncode")) is not int
            or r["returncode"] != 0 or r.get("project_root") != str(root)
            or r.get("command") != ["du", "-s", "-B1", str(root)]):
        raise ValueError("Complete successful project-root storage snapshot required")
    integer(r.get("used_bytes"), "snapshot byte count")
    completed = dt.datetime.fromisoformat(r["completed_utc"])
    if completed.tzinfo is None or completed > dt.datetime.now(dt.timezone.utc):
        raise ValueError("Timezone-aware past snapshot timestamp required")
    return dict(path=str(path), sha256=expected_sha, receipt=r)


class StorageLedger:
    """Persistent, shared accounting under scheduler.lock. Never reclaims outputs."""
    def __init__(self, root, snapshot, external_reserve, limit, available):
        self.root = root
        self.snapshot = snapshot
        self.external = integer(external_reserve, "external write reserve")
        self.limit = integer(limit, "project storage ceiling", 1)
        if self.limit > 450_000_000_000:
            raise ValueError("Pool storage ceiling must be <=450 GB")
        self.available = available
        self.path = root / ".runtime/gpup/storage_reservations.json"

    def _load(self):
        canonical_path(self.root, self.path)
        if self.path.exists():
            r = read_json(self.path)
            if (r.get("schema") != "gpu_pool_storage_ledger_v1"
                    or r.get("root") != str(self.root)
                    or r.get("snapshot_sha256") != self.snapshot["sha256"]
                    or r.get("baseline_bytes") != self.snapshot["receipt"]["used_bytes"]
                    or r.get("external_reserve_bytes") != self.external
                    or r.get("limit_bytes") != self.limit):
                raise ValueError("Existing persistent storage ledger binding differs; audited rebase required")
            if not isinstance(r.get("controllers"), dict) or not isinstance(r.get("jobs"), dict):
                raise ValueError("Malformed storage ledger")
            return r
        return dict(schema="gpu_pool_storage_ledger_v1", root=str(self.root),
                    snapshot_sha256=self.snapshot["sha256"],
                    baseline_bytes=self.snapshot["receipt"]["used_bytes"],
                    external_reserve_bytes=self.external, limit_bytes=self.limit,
                    controllers={}, jobs={})

    def _check(self, r):
        reserves = self.external
        for collection in (r["controllers"], r["jobs"]):
            for entry in collection.values():
                reserves += integer(entry["reserved_bytes"], "persistent storage reserve")
        projected = r["baseline_bytes"] + reserves
        if projected >= self.limit:
            raise ValueError("Accounted project storage ceiling exceeded")
        if reserves >= integer(self.available(), "filesystem free bytes"):
            raise ValueError("Filesystem free-space headroom exceeded")
        return dict(snapshot_sha256=self.snapshot["sha256"],
                    baseline_bytes=r["baseline_bytes"], reserve_bytes=reserves,
                    projected_bytes=projected, limit_bytes=self.limit,
                    persistent_reservations=len(r["jobs"]))

    def controller(self, pool, reserve):
        r = self._load()
        if pool in r["controllers"]:
            raise ValueError("Pool names cannot be reused")
        r["controllers"][pool] = dict(reserved_bytes=reserve, pid=os.getpid())
        account = self._check(r)
        atomic_json(self.path, r)
        return account

    def job(self, pool, spec):
        with lock_file(self.root, "scheduler.lock") as scheduler:
            fcntl.flock(scheduler, fcntl.LOCK_EX)
            r = self._load()
            if len(r["jobs"]) >= 4096:
                raise ValueError("Persistent storage job limit reached")
            key = pool + "/" + spec["job"]
            if key in r["jobs"] or any(e["output"] == spec["output"] for e in r["jobs"].values()):
                raise ValueError("Logical job or output already reserved")
            r["jobs"][key] = dict(reserved_bytes=spec["storage_bytes"], output=spec["output"])
            account = self._check(r)
            atomic_json(self.path, r)
            return account


def acquire_device(root, requested, pool_job, inventory, storage_admission, storage_bytes):
    """Interoperate atomically with launch_gpu.py through the identical locks."""
    if requested not in ALLOWED_GPUS:
        raise ValueError("Eligible physical GPU 2, 3, 4 or 5 required")
    lease = None
    with lock_file(root, "scheduler.lock") as scheduler:
        fcntl.flock(scheduler, fcntl.LOCK_EX)
        free = {}
        held = []
        try:
            for idx in ALLOWED_GPUS:
                h = lock_file(root, f"gpu_{idx}.lock")
                try:
                    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    free[idx] = h
                except BlockingIOError:
                    h.close()
                    held.append(idx)
            if len(held) >= 2:
                raise RuntimeError("Two distinct project physical GPUs are already reserved")
            rows = checked_inventory(inventory)
            r = rows[requested]
            if (requested not in free or r["compute"] or r["used_mib"] > 100 or r["utilization"]):
                raise RuntimeError("Requested eligible physical GPU is not free")
            account = storage_admission()
            lease = free.pop(requested)
            metadata = dict(launcher_pid=os.getpid(), gpu=requested, uuid=r["uuid"],
                            job=pool_job, started_unix=time.time(), storage_bytes=storage_bytes)
            # One immutable write, before any worker exists. No LOCK_UN on this FD.
            data = encoded(metadata)
            os.ftruncate(lease.fileno(), len(data))
            os.pwrite(lease.fileno(), data, 0)
            os.fsync(lease.fileno())
            return lease, r, metadata, account
        except BaseException:
            if lease is not None:
                lease.close()
            raise
        finally:
            for h in free.values():
                h.close()


class Controller:
    def __init__(self, *, root=ROOT, pool, gpu, headroom_mib, storage_snapshot,
                 storage_snapshot_sha256, external_storage_reserve_bytes,
                 controller_storage_reserve_bytes=MIN_CONTROLLER_RESERVE,
                 storage_limit_bytes=450_000_000_000, inventory=real_inventory,
                 proc_reader=None, available_bytes=None, python=None, modules=MODULES,
                 socket_relative=None):
        self.root = Path(root)
        if self.root != self.root.resolve():
            raise ValueError("Canonical project root required")
        if not NAME.fullmatch(pool) or len(pool) > 32:
            raise ValueError("Pool name must be 1-32 safe characters")
        self.pool, self.gpu = pool, gpu
        self.headroom = integer(headroom_mib, "device headroom MiB", 1)
        self.controller_reserve = integer(controller_storage_reserve_bytes, "controller reserve",
                                          MIN_CONTROLLER_RESERVE)
        self.pool_job = f"gpu_pool_v1:{pool}:gpu{gpu}"
        self.inventory = inventory
        self.proc_reader = proc_reader or ProcReader()
        self.python = str(python or self.root / ".runtime/env/bin/python")
        self.modules = frozenset(modules)
        self.base = canonical_path(self.root, self.root / ".runtime/gpup")
        self.directory = canonical_path(self.root, self.base / pool)
        self.registry = self.directory / "registry.json"
        self.socket_path = canonical_path(self.root, self.root / (socket_relative or
                                          f".runtime/gpup/{pool}.sock"))
        if len(os.fsencode(self.socket_path)) > 107:
            raise ValueError("Contained Unix socket exceeds Linux pathname budget")
        snapshot = snapshot_record(self.root, storage_snapshot, storage_snapshot_sha256)
        def free_bytes():
            v = os.statvfs(self.root)
            return v.f_bavail * v.f_frsize
        self.storage = StorageLedger(self.root, snapshot, external_storage_reserve_bytes,
                                     storage_limit_bytes, available_bytes or free_bytes)
        self.lock = threading.RLock()
        self.lease = None
        self.listener = None
        self.jobs = {}
        self.children = {}
        self.logs = {}
        self.processes = {}
        self.pid_jobs = {}
        self.ambiguous = []
        self.lifecycle = "new"
        self.terminate_requested = False
        self.safety_error = None
        self.memory_observation = None
        self.shutdown_requested = False
        self.socket_inode = None
        self.device = None
        self.storage_account = None
        self.controller_identity = proc_identity(os.getpid())
        self.next_poll = 0.0

    def start(self):
        with self.lock:
            if self.lifecycle != "new":
                raise ValueError("Controller already started")
            enable_child_subreaper()
            self.base.mkdir(parents=True, exist_ok=True)
            self.directory.mkdir(mode=0o700, exist_ok=False)
            (self.directory / "jobs").mkdir(mode=0o700)
            try:
                if self.socket_path.exists() or self.socket_path.is_symlink():
                    raise ValueError("Existing socket path is never removed or adopted")
                self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.listener.bind(str(self.socket_path))
                socket_stat = self.socket_path.lstat()
                self.socket_inode = (socket_stat.st_dev, socket_stat.st_ino)
                os.chmod(self.socket_path, 0o600)
                self.listener.listen(32)
                self.listener.setblocking(False)
                self.lease, self.device, self.metadata, self.storage_account = acquire_device(
                    self.root, self.gpu, self.pool_job, self.inventory,
                    lambda: self.storage.controller(self.pool, self.controller_reserve),
                    self.storage.snapshot["receipt"]["used_bytes"])
                if self.headroom + self.device["used_mib"] >= self.device["total_mib"]:
                    raise ValueError("Device headroom exhausts total memory")
                self.lease_sha = hashlib.sha256(os.pread(self.lease.fileno(), 65536, 0)).hexdigest()
                self.lifecycle = "running"
                self._persist()
            except BaseException:
                self.lifecycle = "startup_failed"
                self._release_handles()
                raise
        return self

    def _lease_check(self):
        p = self.root / f".runtime/locks/gpu_{self.gpu}.lock"
        disk, fd = p.lstat(), os.fstat(self.lease.fileno())
        if (not stat.S_ISREG(disk.st_mode) or p.resolve() != p
                or (disk.st_dev, disk.st_ino) != (fd.st_dev, fd.st_ino)
                or hashlib.sha256(os.pread(self.lease.fileno(), 65536, 0)).hexdigest() != self.lease_sha):
            raise RuntimeError("Immutable pool lease changed")

    def _persist(self):
        canonical_path(self.root, self.registry)
        atomic_json(self.registry, self.status())

    def status(self):
        with self.lock:
            return dict(schema="gpu_pool_registry_v1", pool=self.pool, pool_job=self.pool_job,
                        lifecycle=self.lifecycle, controller=self.controller_identity,
                        socket=str(self.socket_path), physical_gpu=self.gpu,
                        lease_path=str(self.root / f".runtime/locks/gpu_{self.gpu}.lock"),
                        lease_sha256=getattr(self, "lease_sha", None),
                        lease_metadata=getattr(self, "metadata", None),
                        device=self.device, headroom_mib=self.headroom,
                        storage=self.storage_account, safety_error=self.safety_error,
                        memory_observation=self.memory_observation,
                        ambiguous_descendants_or_lease_holders=self.ambiguous,
                        uninspected_unrelated_cpu_pids=getattr(self.proc_reader, "uninspected_unrelated", []),
                        updated_unix=time.time(), jobs=list(self.jobs.values()))

    def _validate_spec(self, value):
        if not isinstance(value, dict) or len(encoded(value)) > MAX_REQUEST:
            raise ValueError("Bounded job object required")
        required = {"job", "model", "seed", "module", "args", "output", "peak_mib", "storage_bytes"}
        optional = {"policy", "log_limit_bytes"}
        if not required.issubset(value) or set(value) - required - optional:
            raise ValueError("Exact required job fields and supported options required")
        s = dict(value)
        if not isinstance(s["job"], str) or not NAME.fullmatch(s["job"]):
            raise ValueError("Safe distinct logical job name required")
        if s["job"] in self.jobs or len(self.jobs) >= MAX_JOBS:
            raise ValueError("Duplicate logical job or pool job limit reached")
        if not isinstance(s["model"], str) or not 1 <= len(s["model"]) <= 256:
            raise ValueError("Requested model provenance required")
        if s["seed"] is not None:
            integer(s["seed"], "requested seed")
        if s["module"] not in self.modules:
            raise ValueError("Only admitted GPU worker modules may be launched")
        if (not isinstance(s["args"], list) or len(s["args"]) > 1024
                or any(not isinstance(a, str) or "\0" in a for a in s["args"])):
            raise ValueError("Literal argument vector required")
        p = canonical_path(self.root, s["output"])
        if p.exists() or not p.parent.is_dir():
            raise ValueError("Fresh contained scientific output with existing parent required")
        s["output"] = str(p)
        integer(s["peak_mib"], "peak memory reservation MiB", 1)
        integer(s["storage_bytes"], "job write reservation", 1)
        s.setdefault("policy", "queue")
        s.setdefault("log_limit_bytes", 1024 * 1024)
        if s["policy"] not in {"queue", "reject"}:
            raise ValueError("Admission policy must be queue or reject")
        integer(s["log_limit_bytes"], "bounded worker log bytes", 1)
        if s["log_limit_bytes"] > s["storage_bytes"]:
            raise ValueError("Job storage reservation must include its bounded log")
        if s["peak_mib"] + self.headroom + self.device["used_mib"] > self.device["total_mib"]:
            # Initial free-device overhead is a permanent conservative floor.
            raise ValueError("Job peak and headroom cannot fit the physical device")
        return s

    def _process_scan(self):
        known_processes = {}
        for name, child in self.children.items():
            job = self.jobs[name]
            for item in job["process_identities"]:
                known_processes[item["pid"]] = dict(item)
            known_processes[child.pid] = dict(start_ticks=job["leader_start_ticks"], leader=True)
        table = self.proc_reader(self.lease, os.getpid(), known_processes)
        if not isinstance(table, dict):
            raise RuntimeError("Complete process table required")
        assignments = {}
        for name, child in self.children.items():
            job = self.jobs[name]
            leader = table.get(child.pid)
            if job["leader_start_ticks"] is None and child.returncode is None and leader is not None:
                if leader["uid"] != os.getuid():
                    raise RuntimeError("Worker UID changed before provenance capture")
                job["leader_start_ticks"] = leader["start_ticks"]
            start = job["leader_start_ticks"]
            known = {p["pid"]: p for p in job["process_identities"]}
            if start is not None:
                current_leader = table.get(child.pid)
                leader_valid = bool(current_leader and
                                    current_leader["uid"] == os.getuid() and
                                    current_leader["start_ticks"] == start)
                leader_reused = bool(current_leader and not leader_valid)
                verified_session_member = any(
                    p["pid"] in table
                    and table[p["pid"]]["uid"] == os.getuid()
                    and table[p["pid"]]["start_ticks"] == p["start_ticks"]
                    and table[p["pid"]]["state"] not in {"Z", "X"}
                    and table[p["pid"]]["pgid"] == child.pid
                    and table[p["pid"]]["sid"] == child.pid
                    for p in known.values())
                session_valid = not leader_reused and (leader_valid or verified_session_member)
                changed = True
                while changed:
                    changed = False
                    for pid, item in table.items():
                        if item["uid"] != os.getuid() or item["start_ticks"] < start:
                            continue
                        prior = known.get(pid)
                        identity_matches = prior and prior["start_ticks"] == item["start_ticks"]
                        parent = table.get(item["ppid"])
                        parent_known = known.get(item["ppid"])
                        ancestry = (parent is not None and parent_known is not None
                                    and parent["uid"] == os.getuid()
                                    and parent_known["start_ticks"] == parent["start_ticks"])
                        session = (session_valid and item["pgid"] == child.pid
                                   and item["sid"] == child.pid)
                        leader_matches = pid == child.pid and item["start_ticks"] == start
                        if identity_matches or ancestry or session or leader_matches:
                            if not identity_matches:
                                known[pid] = dict(item)
                                changed = True
                if len(known) > 512:
                    raise RuntimeError("Worker process identity limit exceeded")
                job["process_identities"] = [known[p] for p in sorted(known)]
                for pid, prior in known.items():
                    current = table.get(pid)
                    if (current and current["uid"] == os.getuid()
                            and current["start_ticks"] == prior["start_ticks"]
                            and current["state"] not in {"Z", "X"}):
                        if pid in assignments and assignments[pid] != name:
                            raise RuntimeError("Ambiguous worker process ancestry")
                        assignments[pid] = name
        if sum(len(j["process_identities"]) for j in self.jobs.values()) > 8192:
            raise RuntimeError("Pool process identity limit exceeded")
        self.processes, self.pid_jobs = table, assignments
        self.ambiguous = sorted(pid for pid, p in table.items()
                                if (p.get("lease_fd_present") or p.get("controller_descendant"))
                                and p["uid"] == os.getuid()
                                and pid != os.getpid() and p["state"] not in {"Z", "X"}
                                and pid not in assignments)

    def _memory(self, incoming=0):
        rows = checked_inventory(self.inventory)
        r = rows[self.gpu]
        if r["uuid"] != self.metadata["uuid"] or r["total_mib"] != self.device["total_mib"]:
            raise RuntimeError("Physical GPU UUID or total memory changed")
        active = [j for j in self.jobs.values() if j["state"] in {"launching", "running", "draining"}]
        owned_usage = {}
        foreign = []
        for app in r["compute"]:
            name = self.pid_jobs.get(app["pid"])
            if name is None:
                foreign.append(app["pid"])
            else:
                owned_usage[name] = owned_usage.get(name, 0) + app["used_mib"]
        known_used = sum(owned_usage.values())
        # Includes driver overhead and all unattributed observed usage, never subtracts
        # pending workers' reservations merely because CUDA is not initialised yet.
        unattributed = max(self.device["used_mib"], r["used_mib"] - known_used, 0)
        peaks = sum(j["requested"]["peak_mib"] for j in active)
        overruns = sorted(name for name, used in owned_usage.items()
                          if used > self.jobs[name]["requested"]["peak_mib"])
        projected = peaks + incoming + unattributed + self.headroom
        reasons = []
        if foreign:
            reasons.append("foreign_compute_pids")
        if self.ambiguous:
            reasons.append("unattributed_descendants_or_lease_holders")
        if overruns:
            reasons.append("worker_peak_overrun")
        if projected > r["total_mib"] or r["used_mib"] + self.headroom > r["total_mib"]:
            reasons.append("memory_headroom")
        self.memory_observation = dict(observed=r, active_peak_mib=peaks, incoming_peak_mib=incoming,
                    unattributed_mib=unattributed, projected_mib=projected,
                    foreign_compute_pids=sorted(foreign), overrun_jobs=overruns,
                    blocked_by=reasons)
        return self.memory_observation

    def submit(self, value):
        with self.lock:
            if self.lifecycle != "running":
                raise ValueError("Pool is not accepting jobs")
            self.tick(launch_queued=False)
            if self.safety_error:
                raise RuntimeError(self.safety_error)
            s = self._validate_spec(value)
            m = self._memory(s["peak_mib"])
            queued_ahead = any(j["state"] == "queued" for j in self.jobs.values())
            if s["policy"] == "reject" and (m["blocked_by"] or queued_ahead):
                raise ValueError("Unsafe admission: " + ",".join(m["blocked_by"] or ["FIFO_queue"]))
            argv = [self.python, "-m", s["module"], *s["args"]]
            source_path = canonical_path(self.root, self.root / "src" /
                                         (s["module"].replace(".", "/") + ".py"))
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            self.storage_account = self.storage.job(self.pool, s)
            self.jobs[s["job"]] = dict(logical_job=s["job"], requested=s, argv=argv,
                                      module_source_sha256=source_sha, state="queued",
                                      submitted_unix=time.time(), admission=m,
                                      pid=None, leader_start_ticks=None, process_identities=[],
                                      exit_code=None, log_bytes=0, log_dropped_bytes=0)
            self._persist()
            self._launch_queue()
            self._persist()
            return dict(self.jobs[s["job"]])

    def _launch_queue(self):
        if self.lifecycle != "running" or self.safety_error:
            return
        for name, job in self.jobs.items():
            if job["state"] != "queued":
                continue
            self._process_scan()
            m = self._memory(job["requested"]["peak_mib"])
            job["admission"] = m
            if m["blocked_by"]:
                break  # Strict FIFO, no fitting-job bypass.
            output = canonical_path(self.root, job["requested"]["output"])
            source_path = canonical_path(self.root, self.root / "src" /
                                         (job["requested"]["module"].replace(".", "/") + ".py"))
            if output.exists() or hashlib.sha256(source_path.read_bytes()).hexdigest() != job["module_source_sha256"]:
                job.update(state="failed", error="Output or submitted worker source changed",
                           finished_unix=time.time())
                continue
            self._lease_check()
            job["state"] = "launching"
            job["started_unix"] = time.time()
            self._persist()  # Reservation is active before Popen or CUDA initialisation.
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=self.metadata["uuid"],
                       ARGGNN_PHYSICAL_GPU=str(self.gpu), ARGGNN_JOB_NAME=self.pool_job,
                       ARGGNN_LOGICAL_JOB_NAME=name,
                       ARGGNN_GPU_LEASE=str(self.root / f".runtime/locks/gpu_{self.gpu}.lock"),
                       ARGGNN_GPU_LEASE_FD=str(self.lease.fileno()),
                       PYTHONPATH=str(self.root / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1")
            env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
            env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
            log_path = canonical_path(self.root, self.directory / "jobs" / (name + ".log"))
            log = None
            child = None
            try:
                log = log_path.open("xb")
                child = subprocess.Popen(job["argv"], cwd=self.root, env=env, shell=False,
                                         start_new_session=True, close_fds=True,
                                         pass_fds=(self.lease.fileno(),),
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
                os.set_blocking(child.stdout.fileno(), False)
                self.children[name] = child
                self.logs[name] = log
                job.update(state="running", pid=child.pid, log_path=str(log_path))
                self._process_scan()
            except Exception as error:
                if child is not None:
                    # A spawned process always retains its reservation and supervision.
                    self.children[name] = child
                    self.logs[name] = log
                    job.update(state="draining", pid=child.pid, error=str(error))
                    self.safety_error = "Post-spawn supervision failed: " + str(error)
                    break
                if log is not None:
                    log.close()
                job.update(state="failed", error=str(error), finished_unix=time.time())

    def _drain_logs(self):
        for name, child in list(self.children.items()):
            stream = child.stdout
            if stream is None or stream.closed:
                continue
            job = self.jobs[name]
            for _ in range(16):
                try:
                    block = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    break
                if not block:
                    stream.close()
                    break
                remaining = job["requested"]["log_limit_bytes"] - job["log_bytes"]
                kept = block[:max(0, remaining)]
                if kept:
                    self.logs[name].write(kept)
                    self.logs[name].flush()
                    job["log_bytes"] += len(kept)
                job["log_dropped_bytes"] += len(block) - len(kept)

    def _signal_own(self):
        for pid, name in list(self.pid_jobs.items()):
            saved = self.processes[pid]
            if self.jobs[name].get("last_signalled", {}).get(str(pid)) == saved["start_ticks"]:
                continue
            try:
                fd = open_pidfd(pid)
                try:
                    current = proc_identity(pid)
                    if (current["uid"], current["start_ticks"]) != (saved["uid"], saved["start_ticks"]):
                        continue
                    send_pidfd(fd, signal.SIGTERM)
                    self.jobs[name].setdefault("last_signalled", {})[str(pid)] = saved["start_ticks"]
                finally:
                    os.close(fd)
            except (ProcessLookupError, FileNotFoundError):
                continue

    def tick(self, launch_queued=True):
        with self.lock:
            if self.lifecycle not in {"running", "draining"}:
                return
            try:
                self._lease_check()
                self._drain_logs()
                self._process_scan()
                memory = self._memory()
                self.safety_error = None
                for name, child in list(self.children.items()):
                    job = self.jobs[name]
                    exit_code = child.poll()
                    if exit_code is not None:
                        job.update(exit_code=exit_code, state="draining")
                        # poll() may observe an exit after the earlier process
                        # snapshot. Refresh after that exit before finalising.
                        self._process_scan()
                        memory = self._memory()
                        own_live = name in self.pid_jobs.values()
                        own_compute = any(self.pid_jobs.get(a["pid"]) == name
                                          for a in memory["observed"]["compute"])
                        # Foreign or unreadable compute activity is a conservative
                        # drain blocker, never a reason to kill a foreign process.
                        if (not own_live and not own_compute and not self.ambiguous
                                and not memory["foreign_compute_pids"]):
                            self._drain_logs()
                            job.update(state="succeeded" if exit_code == 0 else "failed",
                                       finished_unix=time.time())
                            if child.stdout is not None:
                                child.stdout.close()
                            self.logs.pop(name).close()
                            del self.children[name]
                if self.terminate_requested:
                    self._signal_own()
                if launch_queued:
                    self._launch_queue()
                if self.lifecycle == "draining" and not self.children and not self.ambiguous:
                    # Refresh after leader exits: no live compute PID may be ignored.
                    final_memory = self._memory()
                    if not final_memory["observed"]["compute"]:
                        self.lifecycle = "closed"
                self._persist()
            except Exception as error:
                self.safety_error = f"{type(error).__name__}: {error}"
                self._persist()

    def shutdown(self, terminate=False):
        with self.lock:
            if type(terminate) is not bool:
                raise ValueError("terminate must be a boolean")
            if self.lifecycle == "running":
                self.lifecycle = "draining"
                for job in self.jobs.values():
                    if job["state"] == "queued":
                        job.update(state="cancelled", finished_unix=time.time(),
                                   error="Pool shutdown cancelled queued work; storage reserve retained")
            self.terminate_requested |= terminate
            self.tick(launch_queued=False)
            return self.status()

    def _release_handles(self):
        if self.listener is not None:
            # Only unlink the socket this controller actually bound.
            try:
                bound = self.listener.getsockname()
            except OSError:
                bound = ""
            self.listener.close()
            self.listener = None
            if bound == str(self.socket_path):
                try:
                    current = self.socket_path.lstat()
                    if (current.st_dev, current.st_ino) == self.socket_inode:
                        self.socket_path.unlink()
                except FileNotFoundError:
                    pass
        if self.lease is not None:
            self.lease.close()  # Never LOCK_UN: inherited workers can still hold the OFD.
            self.lease = None

    def close(self):
        with self.lock:
            if self.children or self.ambiguous or self.lifecycle != "closed":
                raise RuntimeError("Graceful close requires fully drained shutdown")
            self._release_handles()

    def dispatch(self, request):
        if not isinstance(request, dict) or request.get("v") != 1 or type(request.get("v")) is not int:
            raise ValueError("Protocol version 1 required")
        op = request.get("op")
        fields = {"submit": {"v", "op", "spec"}, "status": {"v", "op"},
                  "shutdown": {"v", "op", "terminate"}}
        if op not in fields or set(request) != fields[op]:
            raise ValueError("Exact supported protocol fields required")
        if op == "submit":
            return self.submit(request["spec"])
        if op == "shutdown":
            return self.shutdown(request["terminate"])
        return self.status()

    def run(self, poll_seconds=1.0):
        """Bounded one-request connections; waiting clients never stop supervision."""
        selector = selectors.DefaultSelector()
        selector.register(self.listener, selectors.EVENT_READ, None)
        clients = {}
        normal = False
        def drop(conn):
            clients.pop(conn, None)
            try:
                selector.unregister(conn)
            except KeyError:
                pass
            conn.close()
        try:
            while self.lifecycle != "closed" or clients:
                now = time.monotonic()
                if self.shutdown_requested:
                    self.shutdown_requested = False
                    self.shutdown(terminate=False)
                if now >= self.next_poll:
                    self.tick()
                    self.next_poll = time.monotonic() + poll_seconds
                for conn, state in list(clients.items()):
                    if now > state["deadline"]:
                        drop(conn)
                for key, events in selector.select(timeout=min(poll_seconds, .25)):
                    if key.fileobj is self.listener:
                        conn, _ = self.listener.accept()
                        try:
                            peer = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                            if peer[1] != os.getuid() or len(clients) >= 32:
                                conn.close()
                                continue
                        except Exception:
                            conn.close()
                            continue
                        conn.setblocking(False)
                        clients[conn] = dict(input=bytearray(), output=b"", deadline=time.monotonic() + 30)
                        selector.register(conn, selectors.EVENT_READ, clients[conn])
                        continue
                    conn, state = key.fileobj, key.data
                    try:
                        if events & selectors.EVENT_READ:
                            chunk = conn.recv(8192)
                            if not chunk:
                                drop(conn)
                                continue
                            state["input"].extend(chunk)
                            if len(state["input"]) > MAX_REQUEST:
                                raise ValueError("Request exceeds size limit")
                            if b"\n" in state["input"]:
                                line, remainder = bytes(state["input"]).split(b"\n", 1)
                                if remainder:
                                    raise ValueError("One request per connection required")
                                try:
                                    answer = dict(ok=True, result=self.dispatch(json.loads(line)))
                                except Exception as error:
                                    answer = dict(ok=False, error=f"{type(error).__name__}: {error}")
                                state["output"] = encoded(answer) + b"\n"
                                if len(state["output"]) > MAX_REPLY:
                                    raise ValueError("Response exceeds size limit")
                                selector.modify(conn, selectors.EVENT_WRITE, state)
                        if events & selectors.EVENT_WRITE:
                            sent = conn.send(state["output"])
                            state["output"] = state["output"][sent:]
                            if not state["output"]:
                                drop(conn)
                    except (OSError, ValueError):
                        drop(conn)
            normal = True
        finally:
            for conn in list(clients):
                drop(conn)
            selector.close()
            if normal:
                self.close()
            else:
                self.lifecycle = "controller_error"
                self._persist()
                # Crash/error handling does not unlock inherited leases or kill users.
                self._release_handles()


def request(socket_path, value):
    data = encoded(value) + b"\n"
    if len(data) > MAX_REQUEST:
        raise ValueError("Request exceeds size limit")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(35)
        client.connect(str(socket_path))
        client.sendall(data)
        reply = bytearray()
        while b"\n" not in reply:
            block = client.recv(65536)
            if not block:
                raise RuntimeError("Controller closed without a complete response")
            reply.extend(block)
            if len(reply) > MAX_REPLY:
                raise RuntimeError("Controller response exceeds size limit")
    r = json.loads(bytes(reply).split(b"\n", 1)[0])
    if not r.get("ok"):
        raise RuntimeError(r.get("error", "Controller rejected request"))
    return r["result"]


def disk_status(root, pool):
    r = read_json(root / ".runtime/gpup" / pool / "registry.json")
    saved = r["controller"]
    try:
        current = proc_identity(saved["pid"])
        live = (current["uid"], current["start_ticks"]) == (saved["uid"], saved["start_ticks"])
    except (FileNotFoundError, ProcessLookupError):
        live = False
    r.update(status_from_disk=True, controller_live=live, snapshot_stale=r["lifecycle"] != "closed")
    return r


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--pool", required=True)
    start.add_argument("--gpu", type=int, choices=ALLOWED_GPUS, required=True)
    start.add_argument("--headroom-mib", type=int, required=True)
    start.add_argument("--storage-snapshot", required=True)
    start.add_argument("--storage-snapshot-sha256", required=True)
    start.add_argument("--external-storage-reserve-bytes", type=int, required=True)
    start.add_argument("--controller-storage-reserve-bytes", type=int, default=MIN_CONTROLLER_RESERVE)
    start.add_argument("--storage-limit-bytes", type=int, default=450_000_000_000)
    submit = sub.add_parser("submit")
    submit.add_argument("--pool", required=True)
    submit.add_argument("--job", required=True)
    submit.add_argument("--model", required=True)
    submit.add_argument("--seed", type=int, required=True)
    submit.add_argument("--output", required=True)
    submit.add_argument("--peak-mib", type=int, required=True)
    submit.add_argument("--storage-bytes", type=int, required=True)
    submit.add_argument("--policy", choices=("queue", "reject"), default="queue")
    submit.add_argument("--log-limit-bytes", type=int, default=1024 * 1024)
    submit.add_argument("module", choices=sorted(MODULES))
    submit.add_argument("module_args", nargs=argparse.REMAINDER)
    for command in ("status", "wait", "shutdown"):
        p = sub.add_parser(command)
        p.add_argument("--pool", required=True)
        if command == "wait":
            p.add_argument("--job")
            p.add_argument("--timeout-seconds", type=float, default=0,
                           help="0 waits indefinitely; CLI polling does not block the server")
        if command == "shutdown":
            p.add_argument("--terminate", action="store_true",
                           help="SIGTERM only proven own PID identities; default drains naturally")
    args = parser.parse_args(argv)
    if not NAME.fullmatch(args.pool) or len(args.pool) > 32:
        parser.error("Safe pool name of at most 32 characters required")
    socket_path = ROOT / f".runtime/gpup/{args.pool}.sock"
    if args.command == "start":
        controller = Controller(root=ROOT, pool=args.pool, gpu=args.gpu, headroom_mib=args.headroom_mib,
                                storage_snapshot=args.storage_snapshot,
                                storage_snapshot_sha256=args.storage_snapshot_sha256,
                                external_storage_reserve_bytes=args.external_storage_reserve_bytes,
                                controller_storage_reserve_bytes=args.controller_storage_reserve_bytes,
                                storage_limit_bytes=args.storage_limit_bytes).start()
        def stop(_signum, _frame):
            controller.shutdown_requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(json.dumps(controller.status(), sort_keys=True), flush=True)
        controller.run()
        return 0
    if args.command == "submit":
        spec = dict(job=args.job, model=args.model, seed=args.seed, output=args.output,
                    peak_mib=args.peak_mib, storage_bytes=args.storage_bytes,
                    policy=args.policy, log_limit_bytes=args.log_limit_bytes,
                    module=args.module, args=args.module_args)
        print(json.dumps(request(socket_path, dict(v=1, op="submit", spec=spec)), sort_keys=True))
        return 0
    if args.command == "shutdown":
        print(json.dumps(request(socket_path, dict(v=1, op="shutdown", terminate=args.terminate)), sort_keys=True))
        return 0
    def status():
        try:
            return request(socket_path, dict(v=1, op="status"))
        except (FileNotFoundError, ConnectionRefusedError):
            return disk_status(ROOT, args.pool)
    if args.command == "status":
        print(json.dumps(status(), sort_keys=True))
        return 0
    if args.timeout_seconds < 0:
        parser.error("Timeout must be nonnegative")
    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds else None
    while True:
        r = status()
        if args.job:
            matches = [j for j in r["jobs"] if j["logical_job"] == args.job]
            if not matches:
                raise ValueError("Logical job does not exist")
            j = matches[0]
            if j["state"] in TERMINAL:
                print(json.dumps(j, sort_keys=True))
                return 0 if j["state"] == "succeeded" else 1
        elif all(j["state"] in TERMINAL for j in r["jobs"]):
            print(json.dumps(r, sort_keys=True))
            return 0 if all(j["state"] == "succeeded" for j in r["jobs"]) else 1
        if r.get("snapshot_stale") and not r.get("controller_live"):
            raise RuntimeError("Controller unavailable with unfinished workers; lease adoption is forbidden")
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError("Wait timeout; workers remain supervised")
        time.sleep(.5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)

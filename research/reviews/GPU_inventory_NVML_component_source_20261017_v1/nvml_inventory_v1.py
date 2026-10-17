"""Inactive, interpreter-local NVML inventory candidate. No calls on import.

Only make_real_inventory_v1 explicitly loads NVML. The callable output has the
original gpu_pool_v1.real_inventory schema. This file grants no execution or
integration authority and never changes a base module, eligibility, or a guard.
See handoff.v1.json for the hard-timeout limitation and required ROOT qualification.
"""
from __future__ import annotations

import ctypes as C
import sys
import threading
import time
import uuid

MIB = 2 ** 20
UINT_MAX = 2 ** 32 - 1
ULL_MAX = 2 ** 64 - 1
GPU0_UUID = "GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
ALLOWED_PHYSICAL_GPUS = (0,)  # Informational invariant, never a selector.
MAX_DEVICES = 64
MAX_PROCESSES_PER_DEVICE = 8192
MAX_TOTAL_PROCESSES = 8192
MAX_TOTAL_NAME_UTF8_BYTES = 4 * 2 ** 20
MAX_PROCESS_ARRAY_ATTEMPTS = 4
NAME_BUFFER_BYTES = 65536
CALL_BUDGET_SECONDS = 30.0


class InventoryError(RuntimeError):
    """Incomplete, unsupported, ambiguous or late inventory: reject admission."""


class NvmlCallError(InventoryError):
    def __init__(self, operation, code):
        self.operation, self.code = operation, code
        super().__init__(f"{operation} failed with NVML status {code}")


class MemoryInfoV1(C.Structure):
    _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong),
                ("used", C.c_ulonglong)]


class UtilizationV1(C.Structure):
    _fields_ = [("gpu", C.c_uint), ("memory", C.c_uint)]


class ProcessInfoV3(C.Structure):
    # The API suffix is v3, its public nvmlProcessInfo_t layout is the v2 layout.
    _fields_ = [("pid", C.c_uint), ("usedGpuMemory", C.c_ulonglong),
                ("gpuInstanceId", C.c_uint), ("computeInstanceId", C.c_uint)]


def _integer(value, label, low=0, high=ULL_MAX - 1):
    if type(value) is not int or not low <= value <= high:
        raise InventoryError(f"Invalid or unavailable {label}")
    return value


def _text(value, label, maximum):
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise InventoryError(f"Undecodable {label}") from exc
    if not isinstance(value, str):
        raise InventoryError(f"Unknown {label}")
    value = value.strip()  # Matches each original CSV field's strip operation.
    if (not value or len(value.encode("utf-8")) >= maximum - 1
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
            or value.lower() in {"n/a", "[n/a]", "unknown", "[unknown]", "[not found]"}):
        raise InventoryError(f"Empty, unavailable, truncated or invalid {label}")
    return value


def _gpu_uuid(value):
    value = _text(value, "physical GPU UUID", 96)
    if not value.startswith("GPU-"):
        raise InventoryError("Physical canonical GPU UUID required, no MIG UUID")
    try:
        if str(uuid.UUID(value[4:])) != value[4:]:
            raise ValueError("noncanonical")
    except ValueError as exc:
        raise InventoryError("Canonical lowercase GPU UUID required") from exc
    return value


class CtypesNvmlV1:
    """Small source binding for explicit public symbols, no CUDA or subprocess.

    A supplied library is for RAM ABI tests only. No function/version fallback is
    allowed after missing-symbol, permission, process, memory or driver errors.
    The default path is the system NVML SONAME, never libcuda or CUDA Runtime.
    """
    def __init__(self, *, library=None):
        if sys.platform != "linux" or C.sizeof(C.c_void_p) != 8:
            raise InventoryError("This reviewed binding requires 64-bit Linux")
        if (C.sizeof(MemoryInfoV1), C.sizeof(UtilizationV1), C.sizeof(ProcessInfoV3)) != (24, 8, 24):
            raise InventoryError("Unexpected NVML public structure ABI")
        self._library = C.CDLL("libnvidia-ml.so.1") if library is None else library
        ptr_u = C.POINTER(C.c_uint)
        self._functions = {}
        signatures = {
            "nvmlInit_v2": [], "nvmlShutdown": [],
            "nvmlDeviceGetCount_v2": [ptr_u],
            "nvmlDeviceGetHandleByIndex_v2": [C.c_uint, C.POINTER(C.c_void_p)],
            "nvmlDeviceGetIndex": [C.c_void_p, ptr_u],
            "nvmlDeviceGetUUID": [C.c_void_p, C.POINTER(C.c_char), C.c_uint],
            "nvmlDeviceGetMemoryInfo": [C.c_void_p, C.POINTER(MemoryInfoV1)],
            "nvmlDeviceGetUtilizationRates": [C.c_void_p, C.POINTER(UtilizationV1)],
            "nvmlDeviceGetMigMode": [C.c_void_p, ptr_u, ptr_u],
            "nvmlDeviceGetComputeRunningProcesses_v3": [C.c_void_p, ptr_u, C.POINTER(ProcessInfoV3)],
            "nvmlSystemGetProcessName": [C.c_uint, C.POINTER(C.c_char), C.c_uint],
        }
        for name, args in signatures.items():
            try:
                fn = getattr(self._library, name)
            except AttributeError as exc:
                raise InventoryError(f"Required NVML symbol missing: {name}") from exc
            fn.argtypes, fn.restype = args, C.c_int
            self._functions[name] = fn

    def _raw(self, operation, *args):
        code = self._functions[operation](*args)
        if type(code) is not int:
            raise InventoryError("Invalid NVML return status")
        return code

    def _call(self, operation, *args):
        code = self._raw(operation, *args)
        if code != 0:
            raise NvmlCallError(operation, code)

    def init(self):
        self._call("nvmlInit_v2")

    def shutdown(self):
        self._call("nvmlShutdown")

    def device_count(self):
        count = C.c_uint()
        self._call("nvmlDeviceGetCount_v2", C.byref(count))
        return count.value

    def device_handle(self, index):
        handle = C.c_void_p()
        self._call("nvmlDeviceGetHandleByIndex_v2", index, C.byref(handle))
        if not handle.value:
            raise InventoryError("Null physical-device handle")
        return handle

    def device_index(self, handle):
        index = C.c_uint()
        self._call("nvmlDeviceGetIndex", handle, C.byref(index))
        return index.value

    def _string(self, operation, size, *prefix):
        value = C.create_string_buffer(size)
        self._call(operation, *prefix, value, size)
        end = value.raw.find(b"\0")
        if end < 0 or end >= size - 1:
            raise InventoryError(f"Possibly truncated {operation}")
        return value.raw[:end]

    def device_uuid(self, handle):
        return self._string("nvmlDeviceGetUUID", 96, handle)

    def memory_info(self, handle):
        value = MemoryInfoV1()
        self._call("nvmlDeviceGetMemoryInfo", handle, C.byref(value))
        return value.total, value.free, value.used

    def utilization(self, handle):
        value = UtilizationV1()
        self._call("nvmlDeviceGetUtilizationRates", handle, C.byref(value))
        return value.gpu

    def mig_mode(self, handle):
        current, pending = C.c_uint(), C.c_uint()
        code = self._raw("nvmlDeviceGetMigMode", handle, C.byref(current), C.byref(pending))
        if code == 3:  # Public NVML_ERROR_NOT_SUPPORTED means this device lacks MIG.
            return None
        if code != 0:
            raise NvmlCallError("nvmlDeviceGetMigMode", code)
        return current.value, pending.value

    def compute_processes(self, handle):
        operation = "nvmlDeviceGetComputeRunningProcesses_v3"
        count = C.c_uint(0)
        code = self._raw(operation, handle, C.byref(count), None)
        if code == 0:
            if count.value != 0:
                raise InventoryError("Successful size-only process call has nonzero count")
            return []
        if code != 7:
            raise NvmlCallError(operation, code)
        needed = _integer(count.value, "compute process count", 1, MAX_PROCESSES_PER_DEVICE)
        capacity = min(MAX_PROCESSES_PER_DEVICE, max(8, needed * 2 + 8))
        for _ in range(MAX_PROCESS_ARRAY_ATTEMPTS):
            array = (ProcessInfoV3 * capacity)()
            count = C.c_uint(capacity)
            code = self._raw(operation, handle, C.byref(count), array)
            if code == 0:
                _integer(count.value, "returned process count", 0, capacity)
                return [dict(pid=p.pid, used_bytes=p.usedGpuMemory,
                             gpu_instance_id=p.gpuInstanceId,
                             compute_instance_id=p.computeInstanceId)
                        for p in array[:count.value]]
            if code != 7:
                raise NvmlCallError(operation, code)
            needed = _integer(count.value, "growing process count", capacity + 1,
                              MAX_PROCESSES_PER_DEVICE)
            capacity = min(MAX_PROCESSES_PER_DEVICE, max(capacity * 2, needed + 8))
        raise InventoryError("Compute list changed beyond bounded complete-array attempts")

    def process_name(self, pid):
        return self._string("nvmlSystemGetProcessName", NAME_BUFFER_BYTES, pid)


class NvmlInventoryV1:
    """Fresh full physical inventory, with no inventory/device/process cache.

    Each invocation performs init, full queries and shutdown. A lock serializes
    callers on this instance. A driver C call cannot be forcibly timed out here.
    The elapsed checks reject late returns but do not promise timely return.
    """
    def __init__(self, backend, *, monotonic=time.monotonic):
        self._backend = backend
        self._clock = monotonic
        self._lock = threading.Lock()

    def _within(self, deadline):
        if self._clock() >= deadline:
            raise InventoryError("Inventory exceeded cooperative 30-second elapsed budget")

    def _step(self, deadline, method, *args):
        self._within(deadline)
        value = method(*args)
        self._within(deadline)
        return value

    def _mig(self, deadline, handle):
        value = self._step(deadline, self._backend.mig_mode, handle)
        if value is not None and (type(value) is not tuple or len(value) != 2
                or any(type(v) is not int or v != 0 for v in value)):
            raise InventoryError("MIG enabled, pending, or unknown: physical inventory rejected")

    def __call__(self):
        deadline = self._clock() + CALL_BUDGET_SECONDS
        if not self._lock.acquire(timeout=CALL_BUDGET_SECONDS):
            raise InventoryError("Concurrent inventory did not release its bounded lock")
        initialized = False
        try:
            self._within(deadline)
            self._backend.init()
            initialized = True
            self._within(deadline)
            result = self._collect(deadline)
        finally:
            try:
                if initialized:
                    self._backend.shutdown()  # Always balance successful init, even on error.
                    self._within(deadline)
            finally:
                self._lock.release()
        return result

    def _collect(self, deadline):
        step = lambda method, *args: self._step(deadline, method, *args)
        backend = self._backend
        count = _integer(step(backend.device_count), "physical device count", 1, MAX_DEVICES)
        handles, records, uuids = [], [], set()
        for expected_index in range(count):
            handle = step(backend.device_handle, expected_index)
            index = _integer(step(backend.device_index, handle), "physical index", 0, MAX_DEVICES - 1)
            if index != expected_index:
                raise InventoryError("Physical NVML enumeration index changed or duplicated")
            device_uuid = _gpu_uuid(step(backend.device_uuid, handle))
            if device_uuid in uuids:
                raise InventoryError("Duplicate physical GPU UUID")
            uuids.add(device_uuid)
            self._mig(deadline, handle)
            memory = step(backend.memory_info, handle)
            if type(memory) is not tuple or len(memory) != 3:
                raise InventoryError("Complete v1 total/free/used memory required")
            total, free, used = (_integer(v, "device memory bytes") for v in memory)
            if total < MIB or used > total or free + used != total:
                raise InventoryError("Inconsistent or unavailable v1 GPU memory")
            total_mib, used_mib = total // MIB, (used + MIB - 1) // MIB
            if used_mib > total_mib:
                raise InventoryError("Conservative MiB bounds leave impossible capacity")
            util = _integer(step(backend.utilization, handle), "GPU utilization", 0, 100)
            records.append(dict(index=index, uuid=device_uuid, used_mib=used_mib,
                                total_mib=total_mib, utilization=util, compute=[]))
            handles.append(handle)
        if records[0]["uuid"] != GPU0_UUID:
            raise InventoryError("Physical GPU0 does not match the immutable project UUID")
        total_processes = total_name_bytes = 0
        for handle, record in zip(handles, records):
            processes = step(backend.compute_processes, handle)
            if type(processes) is not list or len(processes) > MAX_PROCESSES_PER_DEVICE:
                raise InventoryError("Complete bounded compute process list required")
            total_processes += len(processes)
            if total_processes > MAX_TOTAL_PROCESSES:
                raise InventoryError("Whole-inventory compute count exceeds complete metadata bound")
            seen = set()
            for process in processes:
                if type(process) is not dict or set(process) != {
                    "pid", "used_bytes", "gpu_instance_id", "compute_instance_id"
                }:
                    raise InventoryError("Complete public process fields required")
                pid = _integer(process["pid"], "compute PID", 1, 2 ** 31 - 1)
                used = _integer(process["used_bytes"], "compute memory bytes")
                if pid in seen:
                    raise InventoryError("Duplicate per-device compute PID")
                seen.add(pid)
                if (type(process["gpu_instance_id"]) is not int
                        or type(process["compute_instance_id"]) is not int
                        or process["gpu_instance_id"] != UINT_MAX
                        or process["compute_instance_id"] != UINT_MAX):
                    raise InventoryError("MIG or unknown process-instance fields")
                # A single integer field is used as BOTH a cap upper bound and
                # a subtraction from device occupancy by the pinned consumers.
                # Neither rounding direction is safe for all those predicates.
                if used % MIB:
                    raise InventoryError("Nonintegral compute MiB cannot preserve both capacity bounds")
                if used // MIB > record["total_mib"]:
                    raise InventoryError("Impossible per-process GPU memory")
                name = _text(step(backend.process_name, pid), "compute process name", NAME_BUFFER_BYTES)
                total_name_bytes += len(name.encode("utf-8"))
                if total_name_bytes > MAX_TOTAL_NAME_UTF8_BYTES:
                    raise InventoryError("Whole-inventory process names exceed complete metadata bound")
                record["compute"].append(dict(pid=pid, name=name, used_mib=used // MIB))
        # Metadata stability, not a claim of an atomic memory/process snapshot.
        if step(backend.device_count) != count:
            raise InventoryError("Physical inventory changed during collection")
        for handle, record in zip(handles, records):
            if (step(backend.device_index, handle) != record["index"]
                    or _gpu_uuid(step(backend.device_uuid, handle)) != record["uuid"]):
                raise InventoryError("Physical identity changed during collection")
            self._mig(deadline, handle)
        return records


def make_real_inventory_v1():
    """Explicit ROOT-only future seam. Loading/querying is never done on import."""
    return NvmlInventoryV1(CtypesNvmlV1())

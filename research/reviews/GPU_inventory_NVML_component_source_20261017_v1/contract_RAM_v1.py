"""RAM contract checks. Never load NVML, launch children, or inspect processes.

Executes authenticated ORIGINAL inventory/capacity ASTs against only tiny mocked
metadata. The candidate's ctypes ABI is exercised with Python fake functions.
"""
from pathlib import Path
import ast
import copy
import csv
import ctypes as C
import hashlib
import io
import json
import sys
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/GPU_inventory_NVML_component_source_20261017_v1"
PINS = {
    "src/arggnn/gpu_pool_v1.py": "ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1",
    "research/reviews/H058_GPU0_full_training_successor_source_20261017_v9/direct_gpu0_v1.py": "615465f93b76d00b84b663a92d4ca072ff4208a527ceccf3a45983a6856e0080",
    "research/reviews/C3_remaining_cells_GPU0_source_20261017_v5/direct_gpu0_v1.py": "ae0b51362ce61ab1054df10ea92025a1be5de4642a5030a2f8c87bbbe2103fa2",
}
SOURCES = {}
for path, expected in PINS.items():
    data = (ROOT / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("Original source pin changed: " + path)
    SOURCES[path] = data.decode()

candidate = types.ModuleType("inactive_nvml_inventory_v1")
source = (HERE / "nvml_inventory_v1.py").read_text()
# Imported source defines classes/functions only. No factory is invoked.
with patch.object(C, "CDLL", side_effect=AssertionError("Real NVML loading forbidden in RAM tests")):
    exec(compile(source, str(HERE / "nvml_inventory_v1.py"), "exec"), candidate.__dict__)
MIB = candidate.MIB
TITLE = "AI Auto Research Pipeline Test"
UINT_MAX = candidate.UINT_MAX
BASE = next(path for path in PINS if path.startswith("src/"))
H058 = next(path for path in PINS if "H058_" in path)
C3 = next(path for path in PINS if "C3_remaining" in path)


AST_CODE_CACHE = {}


def exact_functions(path, names, namespace):
    key = path, tuple(sorted(names))
    if key not in AST_CODE_CACHE:
        tree = ast.parse(SOURCES[path], filename=path)
        selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if {node.name for node in selected} != set(names):
            raise AssertionError("Missing exact original AST function")
        AST_CODE_CACHE[key] = compile(ast.Module(body=selected, type_ignores=[]), path, "exec")
    exec(AST_CODE_CACHE[key], namespace)
    return namespace


def require(value, message):
    if not value:
        raise ValueError(message)


def proc(pid, mib):
    return dict(pid=pid, used_bytes=mib * MIB, gpu_instance_id=UINT_MAX, compute_instance_id=UINT_MAX)


def fixture():
    rows = []
    for index in range(6):
        identifier = candidate.GPU0_UUID if index == 0 else "GPU-00000000-0000-0000-0000-%012d" % index
        rows.append(dict(index=index, uuid=identifier, total=80 * 1024 * MIB,
                         used=0, utilization=12, mig=(0, 0), processes=[]))
    rows[0]["processes"] = [proc(100, 6144), proc(200, 8192), proc(300, 2048), proc(400, 1024)]
    rows[0]["used"] = (6144 + 8192 + 2048 + 1024 + 512) * MIB
    return rows


class FakeBackend:
    def __init__(self, rows=None):
        self.rows = fixture() if rows is None else rows
        self.names = {100: TITLE, 200: TITLE, 300: TITLE, 400: "/other/user program"}
        self.calls = []
        self.failure = None
    def _call(self, name, result, *args):
        self.calls.append((name, args))
        if self.failure == name:
            raise candidate.NvmlCallError(name, 4)
        return copy.deepcopy(result)
    def init(self): return self._call("init", None)
    def shutdown(self): return self._call("shutdown", None)
    def device_count(self): return self._call("device_count", len(self.rows))
    def device_handle(self, index): return self._call("device_handle", index, index)
    def device_index(self, handle): return self._call("device_index", self.rows[handle]["index"], handle)
    def device_uuid(self, handle): return self._call("device_uuid", self.rows[handle]["uuid"], handle)
    def mig_mode(self, handle): return self._call("mig_mode", self.rows[handle]["mig"], handle)
    def memory_info(self, handle):
        row = self.rows[handle]
        return self._call("memory_info", (row["total"], row["total"] - row["used"], row["used"]), handle)
    def utilization(self, handle): return self._call("utilization", self.rows[handle]["utilization"], handle)
    def compute_processes(self, handle): return self._call("compute_processes", self.rows[handle]["processes"], handle)
    def process_name(self, pid): return self._call("process_name", self.names[pid], pid)


def original_cli_inventory(backend):
    calls = []
    def csv_text(rows):
        stream = io.StringIO(); csv.writer(stream).writerows(rows); return stream.getvalue()
    def run(argv, **kwargs):
        calls.append(dict(argv=argv, kwargs=kwargs))
        if argv[1].startswith("--query-gpu="):
            rows = [[row["index"], row["uuid"], row["used"] // MIB, row["total"] // MIB, row["utilization"]]
                    for row in backend.rows]
        else:
            rows = [[row["uuid"], app["pid"], backend.names[app["pid"]], app["used_bytes"] // MIB]
                    for row in backend.rows for app in row["processes"]]
        return types.SimpleNamespace(stdout=csv_text(rows))
    namespace = dict(csv=csv, subprocess=types.SimpleNamespace(run=run), uuid=__import__("uuid"))
    exact_functions(BASE, ["canonical_gpu_uuid", "real_inventory"], namespace)
    result = namespace["real_inventory"]()
    assert len(calls) == 2
    assert calls[0]["argv"] == ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    assert calls[1]["argv"] == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    assert all(row["kwargs"] == dict(check=True, capture_output=True, text=True, timeout=15, shell=False) for row in calls)
    return result


def consumers(inventory):
    identities = {pid: dict(pid=pid, uid=7, start_ticks=pid + 1000, state="S") for pid in (100, 200, 300)}
    base = types.SimpleNamespace(real_inventory=lambda: copy.deepcopy(inventory), proc_identity=lambda pid: copy.deepcopy(identities[pid]))
    cap, registry, c3, authority = {}, dict(updated_unix=123), dict(process_identities=[identities[100]]), {}
    hq = dict(caps=dict(max_cuda_reserved_bytes=28 * 2 ** 30))
    common = dict(require=require, time=types.SimpleNamespace(time=lambda: 456), os=types.SimpleNamespace(getuid=lambda: 7),
                  GPU0_UUID=candidate.GPU0_UUID, CURRENT_CAP={"path": "synthetic", "sha256": "0" * 64}, json=json)
    peer = exact_functions(H058, ["GPUCapacityFailureV3", "bounded_capacity_failure_v3", "capacity_v1"], dict(common,
        controller_v1=lambda q, base: (cap, registry, c3, authority),
        validate_native_resource_disposition_v1=lambda q, fresh: dict(native_process_identities=[])))
    peer_module = types.SimpleNamespace(capacity_v1=peer["capacity_v1"])
    namespace = exact_functions(C3, ["capacity_v1"], dict(common,
        controller_v1=lambda q, base: None,
        c=types.SimpleNamespace(peer_values_v1=lambda ref: (dict(worker_identity=dict(pid=200, start=1200)), hq)),
        peer_module_v1=lambda: peer_module))
    q = dict(peer_launch_capsule_ref={})
    return peer["capacity_v1"](hq, base, worker_pid=200), namespace["capacity_v1"](q, base, worker_pid=300)


def passes(inventory):
    try: consumers(inventory)
    except ValueError: return False
    return True


class FakeFunction:
    def __init__(self, callback): self.callback = callback
    def __call__(self, *args): return self.callback(*args)


class FakeLibrary:
    def __init__(self, process_steps=None, process_name=b"AI Auto Research Pipeline Test"):
        self.process_steps = list(process_steps or [(0, 0, [])])
        self.process_name = process_name
        self.process_calls = 0
        self.codes = {}
        names = ["nvmlInit_v2", "nvmlShutdown", "nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2",
                 "nvmlDeviceGetIndex", "nvmlDeviceGetUUID", "nvmlDeviceGetMemoryInfo", "nvmlDeviceGetUtilizationRates",
                 "nvmlDeviceGetMigMode", "nvmlDeviceGetComputeRunningProcesses_v3", "nvmlSystemGetProcessName"]
        for name in names:
            setattr(self, name, FakeFunction(lambda *args, name=name: self.invoke(name, *args)))
    @staticmethod
    def uint(pointer, value): C.cast(pointer, C.POINTER(C.c_uint))[0] = value
    def invoke(self, name, *args):
        if self.codes.get(name, 0): return self.codes[name]
        if name in ("nvmlInit_v2", "nvmlShutdown"): return 0
        if name == "nvmlDeviceGetCount_v2": self.uint(args[0], 1)
        elif name == "nvmlDeviceGetHandleByIndex_v2": C.cast(args[1], C.POINTER(C.c_void_p))[0] = 123
        elif name == "nvmlDeviceGetIndex": self.uint(args[1], 0)
        elif name == "nvmlDeviceGetUUID": args[1].value = candidate.GPU0_UUID.encode()
        elif name == "nvmlDeviceGetMemoryInfo":
            value = C.cast(args[1], C.POINTER(candidate.MemoryInfoV1)).contents
            value.total, value.used, value.free = 80 * 1024 * MIB, 100 * MIB, (80 * 1024 - 100) * MIB
        elif name == "nvmlDeviceGetUtilizationRates":
            C.cast(args[1], C.POINTER(candidate.UtilizationV1)).contents.gpu = 17
        elif name == "nvmlDeviceGetMigMode": self.uint(args[1], 0); self.uint(args[2], 0)
        elif name == "nvmlSystemGetProcessName":
            if len(self.process_name) >= args[2]: C.memmove(args[1], self.process_name, args[2])
            else: args[1].value = self.process_name
        elif name == "nvmlDeviceGetComputeRunningProcesses_v3":
            self.process_calls += 1
            code, count, rows = self.process_steps.pop(0)
            self.uint(args[1], count)
            if rows:
                for i, row in enumerate(rows):
                    args[2][i].pid = row["pid"]; args[2][i].usedGpuMemory = row["used_bytes"]
                    args[2][i].gpuInstanceId = row["gpu_instance_id"]; args[2][i].computeInstanceId = row["compute_instance_id"]
            return code
        return 0


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch.object(C, "CDLL", side_effect=AssertionError("Actual NVML loading forbidden"))
        self.guard.start(); self.addCleanup(self.guard.stop)
    def rejected(self, change):
        backend = FakeBackend(); change(backend)
        with self.assertRaises((candidate.InventoryError, KeyError)):
            candidate.NvmlInventoryV1(backend)()
        self.assertEqual(backend.calls[-1][0], "shutdown")
    def test_exact_original_schema_and_two_original_consumers(self):
        backend = FakeBackend(); actual = candidate.NvmlInventoryV1(backend)()
        expected = original_cli_inventory(backend)
        self.assertEqual(actual, expected)
        self.assertEqual(consumers(actual), consumers(expected))
        self.assertEqual(set(actual[0]), {"index", "uuid", "used_mib", "total_mib", "utilization", "compute"})
        self.assertEqual(consumers(actual)[1]["projected_mib"], 73216)
        namespace = exact_functions(BASE, ["integer", "canonical_gpu_uuid", "checked_inventory"], dict(uuid=__import__("uuid"), ALLOWED_GPUS=(2, 3, 4, 5)))
        # This tests the unchanged old validator, not authorization for old GPUs.
        self.assertEqual(len(namespace["checked_inventory"](lambda: actual)), 6)
        self.assertEqual(candidate.ALLOWED_PHYSICAL_GPUS, (0,))
    def test_no_cache_every_call_queries_and_returns_detached_complete_rows(self):
        backend = FakeBackend(); provider = candidate.NvmlInventoryV1(backend)
        first = provider(); first[0]["compute"].clear()
        backend.rows[0]["used"] += MIB
        second = provider()
        self.assertEqual(len(second[0]["compute"]), 4)
        self.assertEqual(second[0]["used_mib"], 17921)
        self.assertEqual(sum(name == "init" for name, args in backend.calls), 2)
        self.assertEqual(sum(name == "shutdown" for name, args in backend.calls), 2)
        self.assertEqual(sum(name == "compute_processes" for name, args in backend.calls), 12)
        self.assertEqual(sum(name == "memory_info" for name, args in backend.calls), 12)
    def test_device_fractional_bytes_are_conservative_in_original_predicates(self):
        checked = 0
        for external in (0, 1, 1024, 9000):
            for driver in (0, 512, 9000):
                for total_mib in (65536, 71680, 73215, 73216, 73728, 81920):
                    for remainder in (0, 1, MIB // 2, MIB - 1):
                        backend = FakeBackend()
                        backend.rows[0]["processes"][-1] = proc(400, external)
                        backend.rows[0]["used"] = (6144 + 8192 + 2048 + external + driver) * MIB + remainder
                        backend.rows[0]["total"] = total_mib * MIB + MIB // 3
                        actual = candidate.NvmlInventoryV1(backend)()
                        prior_lower_rounding = original_cli_inventory(backend)
                        self.assertGreaterEqual(actual[0]["used_mib"], prior_lower_rounding[0]["used_mib"])
                        self.assertLessEqual(actual[0]["total_mib"], prior_lower_rounding[0]["total_mib"])
                        self.assertFalse(passes(actual) and not passes(prior_lower_rounding))
                        checked += 1
        self.assertEqual(checked, 288)
    def test_original_caps_names_external_residual_and_headroom_still_reject(self):
        for pid, over in ((100, 26625), (200, 30721), (300, 10241), (400, 20000)):
            backend = FakeBackend()
            for row in backend.rows[0]["processes"]:
                if row["pid"] == pid: row["used_bytes"] = over * MIB
            backend.rows[0]["used"] = sum(p["used_bytes"] for p in backend.rows[0]["processes"]) + 512 * MIB
            self.assertFalse(passes(candidate.NvmlInventoryV1(backend)()))
        for pid in (100, 200, 300):
            backend = FakeBackend(); backend.names[pid] = "wrong-title"
            self.assertFalse(passes(candidate.NvmlInventoryV1(backend)()))
        backend = FakeBackend(); backend.rows[0]["used"] = 70000 * MIB
        self.assertFalse(passes(candidate.NvmlInventoryV1(backend)()))
        # Exact original C3 projected boundary, no reduced headroom.
        for total, expected in ((73215, False), (73216, True)):
            backend = FakeBackend(); backend.rows[0]["total"] = total * MIB
            self.assertEqual(passes(candidate.NvmlInventoryV1(backend)()), expected)
    def test_nonintegral_process_bytes_fail_instead_of_unsafe_rounding(self):
        for pid in (100, 200, 300, 400):
            for remainder in (1, MIB // 2, MIB - 1):
                def change(backend, pid=pid, remainder=remainder):
                    next(p for p in backend.rows[0]["processes"] if p["pid"] == pid)["used_bytes"] += remainder
                self.rejected(change)
    def test_duplicates_unknown_memory_mig_and_missing_fields_fail_closed(self):
        changes = [
            lambda b: b.rows[1].update(index=0),
            lambda b: b.rows[1].update(uuid=b.rows[0]["uuid"]),
            lambda b: b.rows[0].update(uuid="MIG-unknown"),
            lambda b: b.rows[0].update(uuid="GPU-00000000-0000-0000-0000-000000000000"),
            lambda b: b.rows[0].update(total=0),
            lambda b: b.rows[0].update(used=candidate.ULL_MAX),
            lambda b: b.rows[0].update(utilization=101),
            lambda b: b.rows[0].update(mig=(1, 1)),
            lambda b: b.rows[0].update(mig=(0, 1)),
            lambda b: b.rows[0].update(mig="unknown"),
            lambda b: b.rows[0]["processes"].append(copy.deepcopy(b.rows[0]["processes"][0])),
            lambda b: b.rows[0]["processes"][0].update(pid=0),
            lambda b: b.rows[0]["processes"][0].update(used_bytes=None),
            lambda b: b.rows[0]["processes"][0].update(used_bytes=candidate.ULL_MAX),
            lambda b: b.rows[0]["processes"][0].update(gpu_instance_id=0),
            lambda b: b.rows[0]["processes"][0].pop("compute_instance_id"),
            lambda b: b.names.update({400: None}),
            lambda b: b.names.update({400: "[Not Found]"}),
            lambda b: b.names.update({400: b"\xff"}),
            lambda b: b.names.update({400: "x" * (candidate.NAME_BUFFER_BYTES - 1)}),
        ]
        for change in changes: self.rejected(change)
        backend = FakeBackend(); backend.rows[1]["mig"] = None
        self.assertEqual(len(candidate.NvmlInventoryV1(backend)()), 6)
    def test_any_query_error_propagates_and_unknown_process_is_never_skipped(self):
        for operation in ("device_count", "device_handle", "device_index", "device_uuid", "mig_mode", "memory_info", "utilization", "compute_processes", "process_name"):
            backend = FakeBackend(); backend.failure = operation
            with self.assertRaises(candidate.NvmlCallError): candidate.NvmlInventoryV1(backend)()
            self.assertEqual(backend.calls[-1][0], "shutdown")
        for operation in ("init", "shutdown"):
            backend = FakeBackend(); backend.failure = operation
            with self.assertRaises(candidate.NvmlCallError): candidate.NvmlInventoryV1(backend)()
        backend = FakeBackend(); backend.rows = []
        with self.assertRaises(candidate.InventoryError): candidate.NvmlInventoryV1(backend)()
    def test_whole_inventory_metadata_bounds_reject_without_truncation(self):
        backend = FakeBackend()
        with patch.object(candidate, "MAX_TOTAL_PROCESSES", 3):
            with self.assertRaises(candidate.InventoryError): candidate.NvmlInventoryV1(backend)()
        self.assertEqual(backend.calls[-1][0], "shutdown")
        backend = FakeBackend()
        with patch.object(candidate, "MAX_TOTAL_NAME_UTF8_BYTES", 5):
            with self.assertRaises(candidate.InventoryError): candidate.NvmlInventoryV1(backend)()
        self.assertEqual(backend.calls[-1][0], "shutdown")
    def test_changed_topology_fails_closed(self):
        backend = FakeBackend(); original = backend.device_count; count = 0
        def device_count():
            nonlocal count
            count += 1
            return original() if count == 1 else 5
        backend.device_count = device_count
        with self.assertRaises(candidate.InventoryError): candidate.NvmlInventoryV1(backend)()
    def test_late_return_rejected_and_init_balanced(self):
        clock = [0.0]; backend = FakeBackend(); original = backend.memory_info
        def memory_info(handle): clock[0] = 31.0; return original(handle)
        backend.memory_info = memory_info
        with self.assertRaises(candidate.InventoryError): candidate.NvmlInventoryV1(backend, monotonic=lambda: clock[0])()
        self.assertEqual(backend.calls[-1][0], "shutdown")
    def test_ctypes_public_abi_and_complete_bounded_growth(self):
        lib = FakeLibrary([(7, 1, []), (7, 12, []), (0, 1, [proc(100, 418)])])
        backend = candidate.CtypesNvmlV1(library=lib)
        backend.init(); handle = backend.device_handle(0)
        self.assertEqual(backend.device_count(), 1)
        self.assertEqual(backend.device_index(handle), 0)
        self.assertEqual(backend.device_uuid(handle).decode(), candidate.GPU0_UUID)
        self.assertEqual(backend.memory_info(handle)[2], 100 * MIB)
        self.assertEqual(backend.utilization(handle), 17)
        self.assertEqual(backend.mig_mode(handle), (0, 0))
        self.assertEqual(backend.compute_processes(handle), [proc(100, 418)])
        self.assertEqual(backend.process_name(100).decode(), TITLE)
        backend.shutdown(); self.assertEqual(lib.process_calls, 3)
        self.assertEqual(C.sizeof(candidate.ProcessInfoV3), 24)
    def test_ctypes_process_bounds_errors_and_no_partial_list(self):
        bad = [
            [(4, 0, [])], [(3, 0, [])], [(7, 0, [])], [(0, 1, [])],
            [(7, candidate.MAX_PROCESSES_PER_DEVICE + 1, [])],
            [(7, 1, []), (0, 11, [])],
            [(7, 1, []), (7, 10, [])],
            [(7, 1, []), (7, 11, []), (7, 21, []), (7, 41, []), (7, 81, [])],
        ]
        for steps in bad:
            backend = candidate.CtypesNvmlV1(library=FakeLibrary(steps))
            with self.assertRaises(candidate.InventoryError): backend.compute_processes(C.c_void_p(123))
        backend = candidate.CtypesNvmlV1(library=FakeLibrary([(0, 0, [])]))
        self.assertEqual(backend.compute_processes(C.c_void_p(123)), [])
    def test_ctypes_missing_symbols_mig_errors_and_truncation(self):
        lib = FakeLibrary(); del lib.nvmlDeviceGetComputeRunningProcesses_v3
        with self.assertRaises(candidate.InventoryError): candidate.CtypesNvmlV1(library=lib)
        for code in (4, 999):
            lib = FakeLibrary(); lib.codes["nvmlDeviceGetMigMode"] = code
            with self.assertRaises(candidate.NvmlCallError): candidate.CtypesNvmlV1(library=lib).mig_mode(C.c_void_p(123))
        lib = FakeLibrary(); lib.codes["nvmlDeviceGetMigMode"] = 3
        self.assertIsNone(candidate.CtypesNvmlV1(library=lib).mig_mode(C.c_void_p(123)))
        for size in (candidate.NAME_BUFFER_BYTES - 1, candidate.NAME_BUFFER_BYTES):
            backend = candidate.CtypesNvmlV1(library=FakeLibrary(process_name=b"x" * size))
            with self.assertRaises(candidate.InventoryError): backend.process_name(100)
    def test_ctypes_end_to_end_output_reaches_exact_original_validator(self):
        lib = FakeLibrary([(7, 1, []), (0, 1, [proc(100, 100)])])
        rows = candidate.NvmlInventoryV1(candidate.CtypesNvmlV1(library=lib))()
        self.assertEqual(rows, [dict(index=0, uuid=candidate.GPU0_UUID, used_mib=100,
                                    total_mib=81920, utilization=17,
                                    compute=[dict(pid=100, name=TITLE, used_mib=100)])])
    def test_source_import_and_factory_never_implicitly_execute(self):
        forbidden = {"subprocess", "torch", "torch_geometric", "pynvml", "cuda"}
        imports = {node.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.alias)}
        self.assertFalse(imports & forbidden)
        self.assertNotIn("make_real_inventory_v1()", source.split("def make_real_inventory_v1():")[0])


if __name__ == "__main__":
    started = time.monotonic()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    report = dict(schema="GPU_inventory_NVML_RAM_contract_result_v1", source_only=True, independent=False,
                  test_count=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  passed=result.wasSuccessful(), elapsed_RAM_test_seconds=time.monotonic() - started,
                  original_exact_AST_refs=[dict(path=p, sha256=h) for p, h in PINS.items()],
                  candidate_source_ref=dict(path=str((HERE / "nvml_inventory_v1.py").relative_to(ROOT)), sha256=hashlib.sha256(source.encode()).hexdigest()),
                  behavioral_cases=dict(conservative_original_capacity_cases=288, nonintegral_process_rejections=12,
                                        malformed_inventory_rejections=20, public_C_ABI_mock=True),
                  actual_calls=dict(NVML=0, GPU=0, CUDA=0, nvidia_smi=0, subprocesses=0,
                                    process_or_guard_probes=0, scientific_data_or_arrays=0,
                                    package_installs=0, Git=0),
                  hard_driver_call_timeout_proved=False, speedup_or_runtime_fit_claimed=False,
                  execution_authority=False, log=stream.getvalue())
    (HERE / "contract_RAM.result.v1.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "log"}, indent=2, sort_keys=True))
    if not result.wasSuccessful(): print(stream.getvalue()); sys.exit(1)

"""Acquire GPU 0 only after a new worker finishes its CPU observation phase.

The same supervised process retains its RAM pool. Existing running workers and
leases are untouched. The worker owns the late lease, rather than claiming an
inherited supervisor lease. All CUDA access remains on physical GPU 0.
"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import types
from . import contracts as c
from .execution import GPU0_UUID, TITLE, gpu0_lease


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_task(task):
    if not task.get("deferred_gpu0"):
        return
    need(task["deferred_gpu0"] is True and task["resource"] == "cpu",
         "A deferred GPU worker starts as an accounted CPU task")
    need(task["factory_module"] in ("arggnn.benchmarking.european_preparation_v9",)
         and task["factory_function"] in ("fit_factory", "c4_fit_factory"),
         "Deferred CUDA is restricted to the reviewed European fit wrappers")
    low, cap = task.get("minimum_free_gpu_bytes"), task.get("max_cuda_reserved_bytes")
    need(type(low) is int and type(cap) is int and 0 < cap <= 74*2**30
         and cap <= low <= 80*2**30, "Original finite GPU memory limits required")


class DeferredGPU0:
    def __init__(self, task, output, check, *, lease_factory=gpu0_lease, sleep=time.sleep):
        validate_task(task)
        self.task, self.output, self.check = task, Path(output), check
        self.lease_factory, self.sleep = lease_factory, sleep
        self.stack = ExitStack()
        self.fd = None
        self.original_environment = None

    def _record(self, phase, **values):
        value = dict(schema="deferred_gpu0_phase_v1", phase=phase,
                     worker_pid=os.getpid(), task_id=self.task["id"],
                     physical_gpu=0, uuid=GPU0_UUID, unix_time=time.time(), **values)
        path = self.output / "gpu_phase.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def __enter__(self):
        need(self.task.get("deferred_gpu0") is True, "Explicit deferred GPU task required")
        need(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU0_UUID
             and "ARGGNN_GPU_LEASE_FD" not in os.environ,
             "CPU preparation retains fixed GPU0 visibility without a GPU lease")
        self._record("cpu_preparation", GPU_lease_held=False)
        return self

    def acquire(self):
        need(self.fd is None, "GPU promotion occurs exactly once")
        torch = sys.modules.get("torch")
        need(torch is None or not torch.cuda.is_initialized(),
             "CPU preparation must finish before a PyTorch CUDA context is initialized")
        self.check("before_deferred_GPU0_acquisition")
        self._record("waiting_for_gpu0", GPU_lease_held=False)
        spec = dict(job_id=self.task["id"],
                    minimum_free_gpu_bytes=self.task["minimum_free_gpu_bytes"])
        while True:
            self.check("waiting_for_deferred_GPU0")
            try:
                self.fd = self.stack.enter_context(self.lease_factory(spec))
                break
            except BlockingIOError:
                self.sleep(2)
            except RuntimeError as exc:
                # Recheck device identity separately. Memory pressure is a wait,
                # whereas an unexpected physical GPU identity is an error.
                if str(exc) != "Physical GPU0 identity or free memory does not satisfy this job":
                    raise
                from .execution import _inventory_check
                _inventory_check(0)
                self.sleep(2)
        record = json.loads(os.pread(self.fd, 65536, 0))
        owned, named = os.fstat(self.fd), (c.ROOT / ".runtime/locks/gpu_0.lock").stat()
        need((owned.st_dev, owned.st_ino) == (named.st_dev, named.st_ino)
             and record.get("launcher_pid") == os.getpid()
             and record.get("job") == self.task["id"]
             and record.get("uuid") == GPU0_UUID,
             "Actual late lease must belong to this exact scientific worker")
        import setproctitle
        setproctitle.setproctitle(TITLE)
        keys = ("CUDA_VISIBLE_DEVICES", "ARGGNN_PHYSICAL_GPU", "ARGGNN_GPU_LEASE_FD",
                "ARGGNN_MAX_CUDA_RESERVED_BYTES")
        self.original_environment = {key: os.environ.get(key) for key in keys}
        os.environ.update(CUDA_VISIBLE_DEVICES=GPU0_UUID, ARGGNN_PHYSICAL_GPU="0",
                          ARGGNN_GPU_LEASE_FD=str(self.fd),
                          ARGGNN_MAX_CUDA_RESERVED_BYTES=str(self.task["max_cuda_reserved_bytes"]))
        self._record("gpu0_acquired", GPU_lease_held=True,
                     lease_owner="scientific_worker", max_cuda_reserved_bytes=self.task["max_cuda_reserved_bytes"])
        self.check("after_deferred_GPU0_acquisition")
        return self.fd

    def __exit__(self, kind, value, trace):
        # CUDA context lifetime belongs to this process. Keep the lease FD alive
        # until process exit when CUDA has initialized, including failure exits.
        # There are no further scientific commands after this callable returns.
        torch = sys.modules.get("torch")
        initialized = torch is not None and torch.cuda.is_initialized()
        if self.fd is not None and initialized:
            try:
                self._record("worker_finishing", GPU_lease_held=True,
                             release="kernel_closes_lease_at_process_exit", callable_failed=kind is not None)
            finally:
                _PROCESS_LIFETIME_LEASES.append(self.stack.pop_all())
        else:
            self.stack.close()
        return False


_PROCESS_LIFETIME_LEASES = []


def compile_child(original):
    """Clone frozen worker lifecycle with a bounded, explicit phase callback."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    node = tree.body[0]
    source = ast.unparse(node)
    old_title = "TITLE if task['resource'] == 'gpu0' else 'AI Research Data Preparation'"
    need(source.count(old_title) == 1, "One process-title boundary required")
    source = source.replace(old_title, "TITLE if task['resource'] == 'gpu0' or task.get('deferred_gpu0') else 'AI Research Data Preparation'")
    old_title_guard = "need(task['resource'] != 'gpu0', 'GPU process title package is required')"
    need(source.count(old_title_guard) == 1, "One required title-package guard")
    source = source.replace(old_title_guard,
        "need(task['resource'] != 'gpu0' and not task.get('deferred_gpu0'), 'GPU process title package is required')")
    old_visibility = "need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU stages cannot use CUDA')"
    need(source.count(old_visibility) == 1, "One CPU visibility guard required")
    source = source.replace(old_visibility,
        "need(os.environ.get('CUDA_VISIBLE_DEVICES') == (GPU0_UUID if task.get('deferred_gpu0') else ''), 'Exact CPU or deferred GPU0 visibility required')")
    marker = "torch = sys.modules.get('torch')"
    need(source.count(marker) == 1, "One runtime CUDA-check boundary required")
    source = source.replace(marker, marker + "\n        if task.get('deferred_gpu0') and 'ARGGNN_GPU_LEASE_FD' not in os.environ and torch is not None:\n            need(not torch.cuda.is_initialized(), 'Deferred preparation created a CUDA context before its lease')")
    old = "task['resource'] == 'gpu0' and torch is not None and torch.cuda.is_initialized()"
    need(source.count(old) == 1, "One original CUDA allocation check required")
    source = source.replace(old,
        "(task['resource'] == 'gpu0' or task.get('deferred_gpu0', False)) and torch is not None and torch.cuda.is_initialized()")
    old = "inherited = () if task['resource'] != 'gpu0' else (int(os.environ['ARGGNN_GPU_LEASE_FD']),)"
    need(source.count(old) == 1, "One native lease propagation boundary required")
    source = source.replace(old,
        "inherited = (int(os.environ['ARGGNN_GPU_LEASE_FD']),) if 'ARGGNN_GPU_LEASE_FD' in os.environ else ()")
    tree = ast.parse(source)
    node = tree.body[0]
    at = next(i for i, item in enumerate(node.body)
              if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "module" for t in item.targets))
    node.body.insert(at, ast.parse(
        "phase = DeferredGPU0(task, output, check) if task.get('deferred_gpu0') else None").body[0])
    callback_at = next(i for i, item in enumerate(node.body)
                      if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "callbacks" for t in item.targets))
    node.body.insert(callback_at + 1, ast.parse(
        "if phase is not None:\n    callbacks['acquire_gpu0'] = phase.acquire").body[0])
    call_at = next(i for i, item in enumerate(node.body)
                  if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "value" for t in item.targets))
    original_call = node.body[call_at]
    node.body[call_at:call_at+1] = [ast.If(
        test=ast.Compare(left=ast.Name(id="phase", ctx=ast.Load()), ops=[ast.IsNot()], comparators=[ast.Constant(None)]),
        body=[ast.With(items=[ast.withitem(context_expr=ast.Name(id="phase",ctx=ast.Load()))],body=[original_call])],
        orelse=[original_call])]
    namespace = dict(original.__globals__, DeferredGPU0=DeferredGPU0)
    exec(compile(ast.fix_missing_locations(tree), __file__ + ":worker", "exec"), namespace)
    compiled = namespace[node.name]
    def invoke(*args, **kwargs):
        # install_bindings runs before execution. Resolve current policy/PSS/
        # output callbacks then, rather than freezing import-time callbacks.
        actual = types.FunctionType(compiled.__code__,
            dict(original.__globals__, DeferredGPU0=DeferredGPU0),
            compiled.__name__, compiled.__defaults__, compiled.__closure__)
        actual.__kwdefaults__ = compiled.__kwdefaults__
        return actual(*args, **kwargs)
    return invoke


def compile_execute(original):
    """Keep one physical-device identity visible throughout a deferred worker."""
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    text = ast.unparse(tree)
    old = "CUDA_VISIBLE_DEVICES=GPU0_UUID if fd is not None else ''"
    need(text.count(old) == 1, "One original device environment assignment required")
    text = text.replace(old,
        "CUDA_VISIBLE_DEVICES=GPU0_UUID if fd is not None or task.get('deferred_gpu0') else ''")
    marker = "env['ARGGNN_WORKFLOW_DEADLINE'] = str(deadline)"
    need(text.count(marker) == 1, "One exact worker deadline assignment required")
    text = text.replace(marker, marker + "\n        if task.get('deferred_gpu0'):\n            env['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'")
    namespace = dict(original.__globals__)
    exec(compile(text, __file__ + ":supervisor", "exec"), namespace)
    compiled = namespace[original.__name__]
    return types.FunctionType(compiled.__code__, original.__globals__,
        original.__name__, original.__defaults__, original.__closure__)

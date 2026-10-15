"""Inactive optional C3 CPU Dataset worker provider. No launcher or model fit."""
from __future__ import annotations
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import tempfile
import time

from research.reviews.C3_GNN_prefetch_provider_20261015_v1.abstract_ipc_bootstrap_v1 import ensure_abstract_ipc_before_torch_v1

# Spawn imports this module while unpickling the CPU opener. Install here before
# Torch/PyG imports. The GPU parent must already have bootstrapped at its entry.
IPC_BOOTSTRAP = ensure_abstract_ipc_before_torch_v1()
if mp.current_process().name != "MainProcess":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from arggnn.american_c3_common_v1 import require, ref, local
from arggnn.american_c3_gnn_dataset_isolated_v2 import BANK_FIELDS, BLOCK_TENSORS
from arggnn.american_c3_gnn_stream_train_isolated_v2 import SealedObservationProviderV1, _tensors
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.bounded_workers_v1 import BoundedOrderedWorkersV1, _start_ticks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PINS={'src/arggnn/american_c3_gnn_stream_train_isolated_v2.py': '09394390199ceee020e4123d25a4cfcdcee38229450abc3ddd24dde55773b912', 'src/arggnn/american_c3_gnn_dataset_isolated_v2.py': '6e4f32fcec9e6af0c6aa2c81cdbcc4ef1bf7ff83da04b233f0064b6a65c33cb2', 'src/arggnn/american_c3_gnn_observations_isolated_v2.py': 'adf1be0aab14e54b0d48c8cb577b3df541bed2ee88ac846f16bab8a75d56f781'}



def _buffer_feature_identity(block):
    """Same canonical bytes as original digest, without whole-array bytes copies."""
    h = hashlib.sha256()
    size = 0
    for name in BANK_FIELDS + BLOCK_TENSORS:
        h.update(name.encode())
        for tensor in _tensors(block[name]):
            require(tensor.device.type == "cpu" and tensor.layout == torch.strided,
                    "CPU complete-tree tensors required")
            array = tensor.detach().contiguous().numpy()
            size += array.nbytes
            h.update(str(array.dtype).encode())
            h.update(json.dumps(list(array.shape)).encode())
            h.update(memoryview(array.reshape(-1)).cast("B"))
    h.update(json.dumps({key:getattr(block,key) for key in
        ("parent_id","block_index","start_tree","stop_tree","sequence_length_bp","complete_parent")},
        sort_keys=True).encode())
    return h.hexdigest(), size


def _check_pins():
    for name, expected in PINS.items():
        require(ref(name)["sha256"] == expected, "Original observation/checkpoint source pin")


def _rss(pid, start_ticks):
    try:
        require(_start_ticks(pid) == start_ticks, "Exact owned worker PID/start identity")
    except (FileNotFoundError, ProcessLookupError):
        return 0
    values = {}
    try:
        stream = open("/proc/%d/status" % pid)
    except (FileNotFoundError, ProcessLookupError):
        return 0
    with stream:
        for line in stream:
            if line.startswith("State:") and line.split()[1] == "Z":
                return 0
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, value, unit = line.split()
                require(unit == "kB", "Linux CPU RSS unit")
                values[name] = int(value) * 1024
    try:
        require(_start_ticks(pid) == start_ticks, "Stable owned worker PID/start identity")
    except (FileNotFoundError, ProcessLookupError):
        return 0
    require("VmRSS:" in values, "Exact owned CPU worker RSS required")
    return max(values.values())


class _CPUReader:
    def __init__(self, *, plan_ref, parent_id, worker_tmp_directory,
                 seal_record, max_block_bytes, worker_rss_cap_bytes):
        require(not torch.cuda.is_initialized(), "Spawned observation reader must have no CUDA context")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)
        ensure_abstract_ipc_before_torch_v1()
        _check_pins()
        directory = local(Path(worker_tmp_directory) / ("prefetch_worker_%d" % os.getpid()))
        directory.mkdir(parents=True, exist_ok=False)
        os.environ["TMPDIR"] = str(directory)
        tempfile.tempdir = str(directory)
        self.provider = SealedObservationProviderV1(plan_ref, parent_id=parent_id,
            worker_tmp_directory=str(directory), cache_bytes=0)
        self.provider.seal_record = seal_record
        self.max_block_bytes, self.rss_cap = max_block_bytes, worker_rss_cap_bytes

    def __call__(self, index):
        # Exact original Dataset, source joins, observation checks and digest.
        started = time.perf_counter()
        block, digest, size = self.provider._read(index)
        require(size <= self.max_block_bytes, "Declared feature block byte cap")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= self.rss_cap,
                "Actual owned CPU worker peak RSS cap")
        return block, digest, size, dict(read_and_original_hash_seconds=time.perf_counter()-started)

    def close(self):
        self.provider.release()


def _open_cpu_reader(**kwargs):
    return _CPUReader(**kwargs)


class PrefetchSealedObservationProviderV1(SealedObservationProviderV1):
    """Exact original get/seal API, with lazy bounded spawn workers.

    All cached CPU blocks remain private by caller contract exactly as in the
    original provider. Every cache hit is fully rehashed, with no trusted cache
    bypass. Original checkpoint consumers still clone before device transfer.
    """
    def __init__(self, plan_ref, *, parent_id, worker_tmp_directory, cache_bytes,
                 num_workers, max_block_bytes, worker_rss_cap_bytes,
                 aggregate_rss_cap_bytes, resource_check,
                 task_timeout_seconds=300., drain_timeout_seconds=10.):
        _check_pins()
        super().__init__(plan_ref, parent_id=parent_id,
                         worker_tmp_directory=worker_tmp_directory, cache_bytes=cache_bytes)
        require(type(num_workers) is int and 1 <= num_workers <= 4
                and type(max_block_bytes) is int and max_block_bytes > 0
                and cache_bytes >= max_block_bytes
                and type(worker_rss_cap_bytes) is int and worker_rss_cap_bytes > max_block_bytes
                and type(aggregate_rss_cap_bytes) is int
                and aggregate_rss_cap_bytes >= cache_bytes + (num_workers + 2) * max_block_bytes
                    + num_workers * worker_rss_cap_bytes
                and callable(resource_check), "Explicit worker/cache/transport/RSS budgets")
        require(max_block_bytes >= self.plan["feature_tensor_byte_planning_bound_per_block"],
                "Feature cap covers original declared planning bound")
        self.options = dict(num_workers=num_workers, max_block_bytes=max_block_bytes,
            worker_rss_cap_bytes=worker_rss_cap_bytes, aggregate_rss_cap_bytes=aggregate_rss_cap_bytes,
            task_timeout_seconds=task_timeout_seconds, drain_timeout_seconds=drain_timeout_seconds)
        self.resource_check = resource_check
        self.worker_tmp_directory = str(worker_tmp_directory)
        self.engine = None
        self.direction, self.last_index = 1, None
        self.drains = []
        self.counters = dict(cache_hits=0, cache_misses=0, delivered_worker_blocks=0,
            delivered_worker_read_and_hash_seconds=0., demand_wait_seconds=0.,
            transport_hash_seconds=0., cache_hash_seconds=0.)

    def _check_resources(self, event, pids):
        self.resource_check(event)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        for pid, start_ticks in pids:
            owned = _rss(pid, start_ticks)
            require(owned <= self.options["worker_rss_cap_bytes"], "Per-owned-worker RSS cap")
            rss += owned
        require(rss <= self.options["aggregate_rss_cap_bytes"],
                "Parent plus exact owned workers RSS cap; no unrelated PID inspection")

    def _pool(self):
        if self.engine is None:
            self.engine = BoundedOrderedWorkersV1(count=len(self), workers=self.options["num_workers"],
                opener=_open_cpu_reader,
                opener_kwargs=dict(plan_ref=self.plan["plan_ref"], parent_id=self.plan["parent_id"],
                    worker_tmp_directory=self.worker_tmp_directory, seal_record=self.seal_record,
                    max_block_bytes=self.options["max_block_bytes"],
                    worker_rss_cap_bytes=self.options["worker_rss_cap_bytes"]),
                check=self._check_resources,
                task_timeout_seconds=self.options["task_timeout_seconds"],
                drain_timeout_seconds=self.options["drain_timeout_seconds"])
        return self.engine

    def _read(self, index):
        started = time.perf_counter()
        block, digest, size, worker_metrics = self._pool().take(index, self.direction, lambda i: i in self.cache)
        self.counters["demand_wait_seconds"] += time.perf_counter()-started
        self.counters["delivered_worker_blocks"] += 1
        self.counters["delivered_worker_read_and_hash_seconds"] += worker_metrics["read_and_original_hash_seconds"]
        started = time.perf_counter()
        # Verify transport content against worker's original validator output.
        actual, actual_size = _buffer_feature_identity(block)
        self.counters["transport_hash_seconds"] += time.perf_counter()-started
        spec = self.plan["blocks"][index]
        require(actual == digest and actual_size == size and size <= self.options["max_block_bytes"]
                and block.plan_ref == self.plan["plan_ref"] and block.parent_id == self.plan["parent_id"]
                and block.block_index == index and block.start_tree == spec["start_tree"]
                and block.stop_tree == spec["stop_tree"], "Exact ordered checked CPU transport")
        if self.seal_record is not None:
            require(digest == self.seal_record["block_feature_sha256"][index],
                    "Unchanged sealed features on every worker recomputation")
        return block, digest, size

    def get(self, index):
        require(type(index) is int and 0 <= index < len(self), "Physical block index")
        if self.last_index is not None and abs(index-self.last_index) == 1:
            self.direction = 1 if index > self.last_index else -1
        self.last_index = index
        try:
            if index in self.cache:
                self.counters["cache_hits"] += 1
                started = time.perf_counter()
                block, digest, size = self.cache.pop(index)
                actual, actual_size = _buffer_feature_identity(block)
                self.counters["cache_hash_seconds"] += time.perf_counter()-started
                require(actual == digest and actual_size == size, "CPU feature cache mutated")
                self.cache[index] = (block, digest, size)
            else:
                self.counters["cache_misses"] += 1
                block, digest, size = self._read(index)
                # Constructor guarantees that one entire block fits. Therefore
                # the row-only get and immediate checkpoint get cannot rebuild it.
                while self.cache and self.used + size > self.cache_bytes:
                    _, (_, _, old_size) = self.cache.popitem(last=False)
                    self.used -= old_size
                self.cache[index] = (block, digest, size)
                self.used += size
            self._pool().prefetch(index+self.direction, self.direction, lambda i: i in self.cache)
            return block
        except BaseException:
            self._release(failed=True)
            raise

    def seal(self, *, resource_check):
        # Inherited implementation covers every focal exactly once, then invokes
        # release. All worker copies drain before returning the original seal.
        try:
            return super().seal(resource_check=resource_check)
        except BaseException:
            self._release(failed=True)
            raise

    def _release(self, *, failed):
        try:
            if self.engine is not None:
                result = self.engine.close(failed=failed)
                self.drains.append(result)
                require(result["drained"] and (failed or not result["failed"]),
                        "Owned CPU prefetch workers drained without hidden failures")
        finally:
            self.engine = None
            self.last_index, self.direction = None, 1
            super().release()

    def release(self):
        self._release(failed=False)

    def diagnostics(self):
        return dict(schema="C3_optional_observation_prefetch_diagnostics_v1",
            counters=dict(self.counters), bounds=dict(self.options), cache_bytes=self.cache_bytes,
            latest_drain=dict(self.drains[-1]) if self.drains else None,
            worker_time_excludes_discarded_prefetch=True, speedup_claimed=False)


class ObservationProviderFactoryV1:
    """Root constructs one factory after reviewing its explicit operational caps."""
    def __init__(self, *, num_workers, max_block_bytes, worker_rss_cap_bytes,
                 aggregate_rss_cap_bytes, resource_check,
                 task_timeout_seconds=300., drain_timeout_seconds=10.):
        _check_pins()
        self.options = dict(num_workers=num_workers, max_block_bytes=max_block_bytes,
            worker_rss_cap_bytes=worker_rss_cap_bytes, aggregate_rss_cap_bytes=aggregate_rss_cap_bytes,
            resource_check=resource_check, task_timeout_seconds=task_timeout_seconds,
            drain_timeout_seconds=drain_timeout_seconds)
        paths = [str(HERE.relative_to(ROOT) / name)
                 for name in ("prefetch_provider_v1.py", "bounded_workers_v1.py", "abstract_ipc_bootstrap_v1.py")]
        self.execution_binding = dict(schema="C3_observation_provider_operational_v1",
            source_refs=[ref(p) for p in paths] + [ref(p) for p in PINS],
            IPC_bootstrap=dict(IPC_BOOTSTRAP), num_workers=num_workers, max_inflight=num_workers,
            cache_requires_full_block=True, cached_features_rehashed=True,
            observation_equations_unchanged=True, checkpoint_parent_gradient_unchanged=True,
            bounds={k:v for k,v in self.options.items() if k != "resource_check"})

    def __call__(self, plan_ref, *, parent_id, worker_tmp_directory, cache_bytes):
        return PrefetchSealedObservationProviderV1(plan_ref, parent_id=parent_id,
            worker_tmp_directory=worker_tmp_directory, cache_bytes=cache_bytes, **self.options)

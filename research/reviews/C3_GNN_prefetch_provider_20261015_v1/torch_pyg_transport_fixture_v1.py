"""One tiny actual Torch/PyG CPU pipe/shared-storage transport scenario.

No project Dataset, model, optimizer, scientific data, GPU or resource guard.
"""
from __future__ import annotations
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import stat
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = HERE / "torch_transport_tmp_v1"
if mp.current_process().name == "MainProcess":
    BASE.mkdir(exist_ok=False)
else:
    assert BASE.is_dir()
PROCESS_TMP = BASE / ("process_%d" % os.getpid())
PROCESS_TMP.mkdir(exist_ok=False)
os.environ["TMPDIR"] = str(PROCESS_TMP)
tempfile.tempdir = str(PROCESS_TMP)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.abstract_ipc_bootstrap_v1 import ensure_abstract_ipc_before_torch_v1
IPC = ensure_abstract_ipc_before_torch_v1()
import torch
from torch_geometric.data import Data
import torch_geometric
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.bounded_workers_v1 import BoundedOrderedWorkersV1

torch.set_num_threads(1)
assert not torch.cuda.is_initialized()


def tensor_fields(block):
    for name in ("x", "edge_index", "focal_rows", "empty", "scalar"):
        yield name, block[name]
    for i, tensor in enumerate(block["levels"]):
        yield "levels_%d" % i, tensor


def make_block(index):
    return Data(x=torch.tensor([[index, 1., 2.], [3., 4., 5.]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        focal_rows=torch.tensor([index], dtype=torch.long),
        empty=torch.empty((0, 2), dtype=torch.float32),
        scalar=torch.tensor(float(index), dtype=torch.float32),
        levels=(torch.tensor([0, 1], dtype=torch.long), torch.empty(0, dtype=torch.long)),
        num_nodes=2, physical_index=index)


class ActualTensorReader:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at

    def __call__(self, index):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and not torch.cuda.is_initialized()
        if index == self.fail_at:
            raise ValueError("single injected actual-tensor transport failure")
        import multiprocessing.resource_sharer as sharer
        listener = sharer._resource_sharer._listener
        address = None if listener is None else listener.address
        if address is not None:
            assert isinstance(address, bytes) and address.startswith(b"\0arggnn-c3-gnn-prefetch-v1-")
        return dict(block=make_block(index), worker_pid=os.getpid(),
                    prior_real_resource_sharer_address=address,
                    bootstrap_namespace=IPC["namespace"])


def opener(**kwargs):
    return ActualTensorReader(**kwargs)


def verify_payload(value, index):
    assert isinstance(value["block"], Data) and value["block"].physical_index == index
    expected = make_block(index)
    got = list(tensor_fields(value["block"]))
    wanted = list(tensor_fields(expected))
    assert [n for n, _ in got] == [n for n, _ in wanted]
    shared = 0
    total_bytes = 0
    for (name, actual), (_, original) in zip(got, wanted):
        assert actual.device.type == "cpu" and not actual.requires_grad
        assert actual.dtype == original.dtype and actual.shape == original.shape
        assert torch.equal(actual, original), name
        total_bytes += actual.numel() * actual.element_size()
        if actual.numel():
            assert actual.is_shared(), "Actual ForkingPickler CPU shared storage: " + name
            shared += 1
    return dict(index=index, worker_pid=value["worker_pid"], pyg_type=type(value["block"]).__name__,
                tensor_bytes=total_bytes, nonempty_shared_tensors=shared,
                resource_sharer_address_repr=repr(value["prior_real_resource_sharer_address"]))


def no_filesystem_sockets():
    nodes = [str(p.relative_to(BASE)) for p in BASE.rglob("*") if stat.S_ISSOCK(p.lstat().st_mode)]
    assert not nodes, nodes
    return nodes


def main():
    maximum = [0]
    engines = []
    def make(count, fail_at=None):
        def bounded_transport_check(event, identities):
            # Fixture assertion only; no project resource guard or admission.
            assert len(identities) <= 2
            if engines:
                maximum[0] = max(maximum[0], len(engines[-1].pending))
        engine = BoundedOrderedWorkersV1(count=count, workers=2, opener=opener,
            opener_kwargs=dict(fail_at=fail_at), check=bounded_transport_check,
            task_timeout_seconds=30., drain_timeout_seconds=10.)
        engines.append(engine)
        return engine
    delivered = []
    actual_sharer_workers = set()
    success_pool = make(4)
    try:
        for index in range(4):
            value = success_pool.take(index, 1)
            delivered.append(verify_payload(value, index))
            address = value["prior_real_resource_sharer_address"]
            if address is not None:
                assert address.startswith(b"\0")
                actual_sharer_workers.add(value["worker_pid"])
            success_pool.prefetch(index+1, 1)
    finally:
        success = success_pool.close()
    assert success["drained"] and not success["failed"] and success["exitcodes"] == [0, 0]
    assert len(actual_sharer_workers) == 2
    no_filesystem_sockets()
    failure_pool = make(2, fail_at=1)
    failure_prefix = None
    rejected = False
    try:
        failure_prefix = verify_payload(failure_pool.take(0, 1), 0)
        failure_pool.take(1, 1)
    except RuntimeError as error:
        rejected = "single injected actual-tensor transport failure" in str(error)
    finally:
        failure = failure_pool.close(failed=True)
    assert rejected and failure["drained"] and failure["failed"]
    assert all(code is not None for code in failure["exitcodes"])
    assert maximum[0] <= 2 and not torch.cuda.is_initialized()
    no_filesystem_sockets()
    receipt = dict(schema="C3_actual_Torch_PyG_shared_transport_fixture_v1",status="passed",
        runtime=dict(executable=sys.executable,python=sys.version.split()[0],
            Torch=torch.__version__,PyG=torch_geometric.__version__,
            CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],CUDA_initialized=False),
        scope="one tiny transport scenario; normal and one injected-failure phases",
        pools=2,workers_per_pool=2,max_simultaneous_workers=2,
        max_observed_inflight=maximum[0],normal_payloads=delivered,
        failure_successful_prefix=failure_prefix,actual_resource_sharer_workers=len(actual_sharer_workers),
        success_drain=success,injected_failure_drain=failure,
        abstract_IPC_bootstrap=IPC,filesystem_socket_nodes_in_fixture_TMP=[],
        actual_model_optimizer_science_data_GPU_guard_or_Git=False,
        throughput_training_or_numerical_equivalence_claim=False)
    output = HERE / "torch_pyg_transport_receipt.v1.json"
    with output.open("x") as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt,indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One handmade transport/coverage check; no Torch, data, model or fit imports."""
from __future__ import annotations
import ast
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.bounded_workers_v1 import BoundedOrderedWorkersV1
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.abstract_ipc_bootstrap_v1 import ensure_abstract_ipc_before_torch_v1

BLOCKS = (
    {"trees": (0, 1), "focal_rows": (0, 3)},
    {"trees": (2,), "focal_rows": (1,)},
    {"trees": (3, 4), "focal_rows": ()},
    {"trees": (5,), "focal_rows": (2, 4, 5)},
)


class HandmadeReader:
    def __init__(self, fail_index=None):
        ipc = ensure_abstract_ipc_before_torch_v1()
        import multiprocessing.connection as connection
        address = connection.arbitrary_address("AF_UNIX")
        assert ipc["leading_NUL_abstract"] and address.startswith(b"\0arggnn-c3-gnn-prefetch-v1-")
        # Bind only the actual abstract address; no filesystem socket or fallback.
        listener = connection.Listener(address=address, family="AF_UNIX", authkey=b"handmade-c3-only")
        listener.close()
        self.fail_index = fail_index

    def __call__(self, index):
        if index == self.fail_index:
            raise ValueError("handmade injected worker failure")
        if index % 2 == 0:
            time.sleep(.01)  # completion order can differ from demand order
        return dict(index=index, **BLOCKS[index])


def opener(**kwargs):
    return HandmadeReader(**kwargs)



def check_handmade_feature_digest():
    # Execute only the two pure digest function ASTs, with CPU array stubs.
    # No Torch/PyG, model, observation Dataset or scientific payload is imported.
    import numpy as np
    class Tensor:
        device = SimpleNamespace(type="cpu")
        layout = "strided"
        def __init__(self, value): self.array = value
        def detach(self): return self
        def contiguous(self):
            return self if self.array.flags.c_contiguous else Tensor(np.ascontiguousarray(self.array))
        def numpy(self): return self.array
    class Block(dict):
        parent_id="handmade"; block_index=0; start_tree=0; stop_tree=2
        sequence_length_bp=6; complete_parent=False
    block = Block(bank=(Tensor(np.asarray(2., dtype=np.float32)),
        Tensor(np.empty((0, 3), dtype=np.int64)),
        Tensor(np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2])),
        geometry=Tensor(np.asarray([[1., 2.]], dtype=np.float32)))
    def tensors(value):
        if isinstance(value, (tuple, list)):
            for item in value: yield item
        else: yield value
    def require(ok, message):
        if not ok: raise AssertionError(message)
    scope=dict(hashlib=hashlib,json=json,torch=SimpleNamespace(strided="strided"),
        require=require,_tensors=tensors,BANK_FIELDS=("bank",),BLOCK_TENSORS=("geometry",))
    for path, name in (
        (ROOT/"src/arggnn/american_c3_gnn_stream_train_v1.py","_feature_identity"),
        (Path(__file__).parent/"prefetch_provider_v1.py","_buffer_feature_identity")):
        nodes=ast.parse(path.read_text()).body
        function=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),"exec"),scope)
    assert scope["_feature_identity"](block)==scope["_buffer_feature_identity"](block)
    original=scope["_buffer_feature_identity"](block)
    block["geometry"].array[0,0]+=1
    assert scope["_buffer_feature_identity"](block)!=original


def main():
    ensure_abstract_ipc_before_torch_v1()
    check_handmade_feature_digest()
    maximum = [0]
    engines = []
    def make(fail_index=None):
        def check(event, identities):
            assert len(identities) <= 2
            if engines:
                maximum[0] = max(maximum[0], len(engines[-1].pending))
        engine = BoundedOrderedWorkersV1(count=4, workers=2, opener=opener,
            opener_kwargs={"fail_index": fail_index}, check=check,
            task_timeout_seconds=10., drain_timeout_seconds=5.)
        engines.append(engine)
        return engine
    engine = make()
    try:
        for direction, order in ((1, list(range(4))), (-1, list(range(3, -1, -1)))):
            got = []
            for physical in order:
                block = engine.take(physical, direction)
                assert block["index"] == physical
                got.append(block)
                engine.prefetch(physical+direction, direction)
            assert sorted(t for b in got for t in b["trees"]) == list(range(6))
            assert sorted(r for b in got for r in b["focal_rows"]) == list(range(6))
            assert [b["index"] for b in got] == order
        # Exact repeated/nonmonotone checkpoint demands remain exact physical IDs.
        demands = [0, 0, 3, 2, 2, 1]
        assert [engine.take(i, -1)["index"] for i in demands] == demands
        assert engine.take(1, 1, skip=lambda candidate: True)["index"] == 1
    finally:
        success = engine.close()
    assert success["drained"] and not success["failed"] and success["exitcodes"] == [0, 0]
    bad = make(fail_index=2)
    rejected = False
    try:
        for index in range(4):
            bad.take(index, 1)
    except RuntimeError as error:
        rejected = "handmade injected worker failure" in str(error)
    finally:
        failure = bad.close(failed=True)
    assert rejected and failure["drained"] and failure["failed"]
    assert all(code is not None for code in failure["exitcodes"])
    assert maximum[0] <= 2
    print(json.dumps(dict(schema="C3_prefetch_handmade_coverage_check_v1", status="passed",
        handmade_trees=6, handmade_focals=6, empty_focal_block_retained=True,
        canonical_feature_digest_identical_for_handmade_scalar_empty_noncontiguous_arrays=True,
        exact_forward_reverse_and_repeated_demands=True, workers=2, abstract_IPC_bound_in_each_handmade_spawn=True,
        max_observed_inflight=maximum[0], success_drain=success, injected_failure_drain=failure,
        original_data_models_fits_GPU_or_guards_used=False,
        throughput_or_gradient_equivalence_claim=False), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

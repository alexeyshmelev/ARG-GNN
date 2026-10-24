"""Small CPU cache/lease and real checkpoint-closure wiring smoke only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import copy
import gc
import hashlib
import json
import weakref
import numpy as np
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed import (
    staged_execution_v2 as original,static_model_v2)
from .cache import LayoutCache
from .executors import IntegratedStagedExecution

OUT = Path("research/scalability_20261024/integration")
PACKET = Path("research/scalability_20261024/execution/representative_slice.npz")


@dataclass
class TopologyBank:
    parent: object
    sample_index: object
    upward_levels: tuple
    downward_levels: tuple


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    meta = json.loads(PACKET.with_suffix(".json").read_text())
    assert hashlib.sha256(PACKET.read_bytes()).hexdigest()==meta["packet_ref"]["sha256"]
    z = np.load(PACKET,allow_pickle=False)
    counts = np.diff(z["node_offsets"])[:4].tolist()
    n = sum(counts)
    def make_bundle(scope="smoke"):
        bank = TopologyBank(torch.from_numpy(z["parent"][:n].copy()),
            torch.from_numpy(z["sample_index"][:4].copy()),
            tuple(torch.from_numpy(z["upward_levels_"+str(i)][
                z["upward_levels_"+str(i)]<n].copy())
                for i in range(meta["level_tuple_lengths"]["upward_levels"])),
            tuple(torch.from_numpy(z["downward_levels_"+str(i)][
                z["downward_levels_"+str(i)]<n].copy())
                for i in range(meta["level_tuple_lengths"]["downward_levels"])))
        bundle = SimpleNamespace(bank=bank,parent_id=meta["parent_id"],native_holds=4,
            observation_ref=meta["source_manifest_ref"],
            episodes=SimpleNamespace(member_hold=torch.tensor([0,1,1,3],dtype=torch.long)))
        metadata = dict(episode_metadata=dict(native_nodes_per_tree=counts),
                        layout_cache_scope=scope)
        return bundle,metadata
    b,m = make_bundle()
    admits = []
    cache = LayoutCache(max_bytes=4*2**20,
        admit=lambda amount,stage:admits.append((amount,stage)))
    lease = cache.acquire(b,m,holds_per_block=4)
    assert cache.snapshot()["active_executor_leases"]==1
    lease.release()
    lease = cache.acquire(b,m,holds_per_block=4)
    assert cache.snapshot()["builds"]==1 and cache.snapshot()["fast_hits"]==1
    lease.release()
    weak_bank = weakref.ref(b.bank)
    del b
    gc.collect()
    assert weak_bank() is None
    assert cache.snapshot()["live_bank_bindings"]==0
    b,m = make_bundle()
    lease = cache.acquire(b,m,holds_per_block=4)
    assert cache.snapshot()["restored_content_binds"]==1
    lease.release()
    b.bank.parent.add_(0)
    try:
        cache.acquire(b,m,holds_per_block=4)
    except ValueError:
        pass
    else:
        raise AssertionError("A changed tensor version must reject fast reuse")
    cache_snapshot = cache.snapshot()
    cache.clear()
    b1,m1 = make_bundle("lease_A")
    estimate,_ = LayoutCache.estimate(b1.bank,counts,4)
    limited = LayoutCache(max_bytes=estimate+24576)
    active = limited.acquire(b1,m1,holds_per_block=4)
    b2,m2 = make_bundle("lease_B")
    try:
        limited.acquire(b2,m2,holds_per_block=4)
    except MemoryError:
        pass
    else:
        raise AssertionError("A live leased plan cannot be evicted")
    try:
        limited.clear()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Cache clear must reject an active lease")
    assert limited.snapshot()["active_executor_leases"]==1
    assert limited.snapshot()["evictions"]==0
    active.release()
    replacement = limited.acquire(b2,m2,holds_per_block=4)
    replacement.release()
    assert limited.snapshot()["evictions"]==1
    assert limited.snapshot()["retained_bytes"]<=limited.max_bytes
    limited_snapshot = limited.snapshot()
    limited.clear()
    # Real _sites method through the unchanged non-reentrant _cp closure.
    # Invoke the methods on a CPU diagnostic context rather than weakening
    # either production constructor's admitted-CUDA guard.
    names = ("genotype","log_donor_allele_frequency","donor_call_mask",
             "panel_allele_frequency","site_entropy","site_hold","spans")
    bank = SimpleNamespace(**{name:torch.from_numpy(z[name].copy()) for name in names})
    saved = torch.load("research/repairs/TreeLoom_resume94_RAM_headroom_v1/checkpoint.pt",
                      map_location="cpu",weights_only=True,mmap=True)
    model = static_model_v2.AmericanTreeLoomC3FullV1()
    model.load_state_dict(saved["model_state"],strict=True)
    model.eval()
    candidate = copy.deepcopy(model)
    assert len(list(model.parameters()))==49
    def run(which,selected_model):
        context = SimpleNamespace(bank=bank,model=selected_model,device=torch.device("cpu"))
        calls = [0]
        def closure():
            calls[0] += 1
            return which(context,0,4)
        selected_model.zero_grad(set_to_none=True)
        value = original._cp(closure)
        loss = (value*torch.linspace(.1,1.,32)).mean()
        loss.backward()
        gradients = {name:p.grad.clone() for name,p in selected_model.named_parameters()
                     if p.grad is not None}
        assert calls[0]>=2
        return value.detach(),gradients,calls[0]
    old,old_grad,old_calls = run(original.StagedExecutionV1._sites,model)
    new,new_grad,new_calls = run(IntegratedStagedExecution._sites,candidate)
    torch.testing.assert_close(old,new,rtol=1e-4,atol=1e-4)
    assert old_grad.keys()==new_grad.keys()
    for key in old_grad:
        torch.testing.assert_close(old_grad[key],new_grad[key],rtol=1e-4,atol=1e-4)
    assert not torch.cuda.is_initialized()
    result = dict(schema="TreeLoom_integrated_CPU_wiring_smoke_v1",passed=True,
        cache_build_fast_restore_mutation=cache_snapshot,
        active_lease_eviction_rejected=True,active_clear_rejected=True,
        idle_eviction_and_budget=limited_snapshot,
        weak_bank_ownership=True,admission_callback_used=bool(admits),
        checkpoint_closure=dict(original_calls=old_calls,integrated_calls=new_calls,
            held_native_trees=4,source_chunk_rows=256,haplotypes=1600,
            output_max_abs_error=float((old-new).abs().max()),
            parameter_gradient_max_abs_error=max(float((old_grad[k]-new_grad[k]).abs().max())
                for k in old_grad),connected_parameter_tensors=len(old_grad)),
        original_model_parameter_tensors=49,CUDA_initialized=False,
        full_history_or49_gradient_GPU_validation=False,
        source_refs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(OUT.glob("*.py"))])
    (OUT/"cpu_smoke.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(passed=False,error_type=type(error).__name__,error=str(error))),flush=True)
        raise

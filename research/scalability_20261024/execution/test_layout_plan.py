"""CPU tests of immutable layout reuse against actual frozen sweep operations."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from research.scalability_20261024.execution.layout_plan import LayoutPlan
from research.scalability_20261024.execution.measure_representative import (
    assert_same, baseline_layout)
from research.benchmark_rebuild_20261024.american.treeloom_speed import static_model_v2


OUT = Path("research/scalability_20261024/execution")


def load_bank():
    meta = json.loads((OUT/"representative_slice.json").read_text())
    packet_path = OUT/"representative_slice.npz"
    assert hashlib.sha256(packet_path.read_bytes()).hexdigest() == meta["packet_ref"]["sha256"]
    with np.load(packet_path, allow_pickle=False) as data:
        fields = {}
        for name in ("parent","sample_index"):
            fields[name] = torch.from_numpy(data[name].copy())
        for name in ("upward_levels","upward_children","upward_parent_rank","downward_levels"):
            fields[name] = tuple(torch.from_numpy(data[name+"_"+str(j)].copy())
                for j in range(meta["level_tuple_lengths"][name]))
        offsets = data["node_offsets"].copy()
    return SimpleNamespace(**fields), np.diff(offsets).tolist()


def sweep_bank(z):
    return SimpleNamespace(parent=torch.from_numpy(z[4].astype(np.int64)),
        upward_levels=tuple(torch.from_numpy(x.astype(np.int64)) for x in z[5]),
        upward_children=tuple(torch.from_numpy(x.astype(np.int64)) for x in z[6]),
        upward_parent_rank=tuple(torch.from_numpy(x.astype(np.int64)) for x in z[7]),
        downward_levels=tuple(torch.from_numpy(x.astype(np.int64)) for x in z[8]))


def run_sweep(model, bank, raw):
    base, up_gate, down_gate = [x.detach().clone().requires_grad_(True) for x in raw]
    degree = torch.bincount(bank.parent[bank.parent>=0],
                            minlength=len(bank.parent)).float().clamp_min(1.)
    up, outside = model.sweep_tree_state_v1(bank,base,up_gate,down_gate,degree)
    weights = torch.linspace(.1,1.,up.numel()).reshape_as(up)
    loss = (up*weights).sum()+(outside*weights.flip(0)).sum()
    loss.backward()
    return (up.detach(),outside.detach()), [
        base.grad,up_gate.grad,down_gate.grad], {
        k: None if p.grad is None else p.grad.detach().clone()
        for k,p in model.named_parameters()}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bank,counts = load_bank()
    members = torch.tensor([0,0,1,3,3,4,9,63,63,64,99,127],dtype=torch.long)
    cases = []
    for size in (1,3,4,17,64,128,256):
        ref = baseline_layout(bank,counts,members,holds_per_block=size)
        plan = LayoutPlan.build(bank,counts,members,holds_per_block=size)
        assert_same(ref,plan.executor_layout())
        binding = plan.bind(bank,counts,members)
        assert_same(ref,binding.assert_binding(bank,counts,members))
        cases.append({"holds_per_block":size,"blocks":len(ref),"exact":True})
    # Actual frozen sweep uses identical arrays/order with independent CPU inputs.
    reference = baseline_layout(bank,counts,members,holds_per_block=4)[0]
    plan = LayoutPlan.build(bank,counts,members,holds_per_block=4)
    candidate = plan.executor_layout()[0]
    original_bank,candidate_bank = sweep_bank(reference),sweep_bank(candidate)
    torch.manual_seed(70001)
    model = static_model_v2.AmericanTreeLoomC3FullV1()
    copy_model = copy.deepcopy(model)
    n = len(original_bank.parent)
    generator = torch.Generator().manual_seed(20261024)
    raw = [torch.randn(n,32,generator=generator),
           torch.rand(n,32,generator=generator),
           torch.rand(n,32,generator=generator)]
    original_outputs,original_inputs,original_parameters = run_sweep(model,original_bank,raw)
    new_outputs,new_inputs,new_parameters = run_sweep(copy_model,candidate_bank,raw)
    output_error = max(float((a-b).abs().max()) for a,b in zip(original_outputs,new_outputs))
    input_error = max(float((a-b).abs().max()) for a,b in zip(original_inputs,new_inputs))
    parameter_error = 0.
    connected = 0
    for key in original_parameters:
        a,b=original_parameters[key],new_parameters[key]
        assert (a is None)==(b is None),key
        if a is not None:
            connected += 1
            parameter_error=max(parameter_error,float((a-b).abs().max()))
    assert output_error==input_error==parameter_error==0.
    result={"schema":"TreeLoom_reusable_layout_cpu_tests_v1",
        "all_passed":True,"real_slice_layout_cases":cases,
        "frozen_first_sweep":{"actual_native_holds":4,"native_nodes":n,
            "hidden_width":32,"output_max_abs_error":output_error,
            "base_and_gate_input_gradient_max_abs_error":input_error,
            "parameter_gradient_max_abs_error":parameter_error,
            "connected_parameter_tensors":connected,
            "all_disconnected_parameters_match":True},
        "scope":"Exact layout-only parity. Uses real observed topology and seeded CPU base/gate probes. No site encoder/history/whole-parent numerical or GPU-speed claim.",
        "packet_ref":json.loads((OUT/"representative_slice.json").read_text())["packet_ref"],
        "source_refs":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in (OUT/"layout_plan.py",OUT/"test_layout_plan.py",
                       Path("src/arggnn/anonymous_clade_gnn_v1.py"),
                       Path("research/benchmark_rebuild_20261024/american/treeloom_speed/static_model_v2.py"))]}
    (OUT/"test_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

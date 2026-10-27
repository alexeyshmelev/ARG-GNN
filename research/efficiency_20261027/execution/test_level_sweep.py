"""CPU parity/gradient tests and bounded real-topology kernel measurements."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import numpy as np
import torch
from research.efficiency_20261027.execution.level_sweep import LevelSweepPlan, sweep_tree_state
from research.scalability_20261024.execution.test_layout_plan import load_bank, sweep_bank
from research.scalability_20261024.execution.measure_representative import baseline_layout
from research.benchmark_rebuild_20261024.american.treeloom_speed import static_model_v2
from research.benchmark_rebuild_20261024.american.treeloom_speed.layout import block_indices

OUT = Path("research/efficiency_20261027/execution")


def synthetic_bank(parent):
    parent = np.asarray(parent, np.int64)
    children = [[] for _ in parent]
    for child, p in enumerate(parent):
        if p >= 0:
            children[p].append(child)
    height = np.full(len(parent), -1, np.int64)
    depth = np.full(len(parent), -1, np.int64)
    def h(node):
        if height[node] < 0:
            height[node] = 1 + max((h(child) for child in children[node]), default=-1)
        return height[node]
    def d(node):
        if depth[node] < 0:
            depth[node] = 0 if parent[node] < 0 else 1+d(parent[node])
        return depth[node]
    for node in range(len(parent)):
        h(node)
        d(node)
    levels, children, ranks, downward = block_indices(parent, height, depth)
    return SimpleNamespace(parent=torch.from_numpy(parent),
        upward_levels=tuple(torch.from_numpy(x) for x in levels),
        upward_children=tuple(torch.from_numpy(x) for x in children),
        upward_parent_rank=tuple(torch.from_numpy(x) for x in ranks),
        downward_levels=tuple(torch.from_numpy(x) for x in downward))


def run(model, bank, raw, plan=None, double_sweep=False, checkpointed=False):
    model.zero_grad(set_to_none=True)
    inputs = [x.detach().clone().requires_grad_(True) for x in raw]
    base, ug, dg = inputs
    degree = torch.bincount(bank.parent[bank.parent >= 0], minlength=len(bank.parent)).float().clamp_min(1.)
    def evaluate(b, u, d):
        if plan is None:
            up, out = model.sweep_tree_state_v1(bank, b, u, d, degree)
        else:
            up, out = sweep_tree_state(model, bank, b, u, d, degree, plan=plan)
        if double_sweep:
            # Nonlinear cross-sweep dependency exercises repeated shared parameters.
            injection = 0.1*torch.tanh(up + out)
            if plan is None:
                up, out = model.sweep_tree_state_v1(bank, b+injection, u, d, degree)
            else:
                up, out = sweep_tree_state(model, bank, b+injection, u, d, degree, plan=plan)
        return up, out
    begin = time.perf_counter()
    if checkpointed:
        from torch.utils.checkpoint import checkpoint
        up, out = checkpoint(evaluate, base, ug, dg, use_reentrant=False, preserve_rng_state=True)
    else:
        up, out = evaluate(base, ug, dg)
    middle = time.perf_counter()
    weight = torch.linspace(.1, 1., up.numel()).reshape_as(up)
    loss = (up*weight).mean() + (out*weight.flip(0)).mean()
    loss.backward()
    end = time.perf_counter()
    return dict(output=(up.detach(), out.detach()), inputs=[x.grad for x in inputs],
        parameters={name: None if p.grad is None else p.grad.detach().clone() for name, p in model.named_parameters()},
        seconds=dict(forward=middle-begin, backward=end-middle, total=end-begin))


def compare(a, b):
    errors = dict(output=0., input_gradient=0., parameter_gradient=0.)
    connected = 0
    for label, pairs in (("output", zip(a["output"], b["output"])),
                          ("input_gradient", zip(a["inputs"], b["inputs"])),
                          ("parameter_gradient", ((a["parameters"][key], b["parameters"][key]) for key in a["parameters"]))):
        for x, y in pairs:
            assert (x is None) == (y is None), (label, "disconnection differs")
            if x is None:
                continue
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)
            errors[label] = max(errors[label], float((x-y).abs().max()))
            connected += int(label == "parameter_gradient")
    return dict(max_absolute_errors=errors, connected_parameter_tensors=connected,
                disconnected_parameters_equal=True, tolerance=dict(rtol=1e-4, atol=1e-6))


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(70001)
    model = static_model_v2.AmericanTreeLoomC3FullV1()
    candidate = copy.deepcopy(model)
    cases = [("single_root", synthetic_bank([-1])),
             ("unary_chain", synthetic_bank([-1, 0, 1, 2, 3, 4])),
             ("multifurcation_and_forest", synthetic_bank([-1, 0, 0, 0, 2, 2, -1, 6, 6, 8]))]
    original = np.array([-1, 0, 0, 1, 1, 2, 2, 6, 6], np.int64)
    order = np.random.default_rng(11).permutation(len(original))
    inverse = np.argsort(order)
    shuffled = original[order]
    shuffled = np.where(shuffled >= 0, inverse[shuffled.clip(0)], -1)
    cases.append(("permuted_node_ids", synthetic_bank(shuffled)))
    real_bank, counts = load_bank()
    for holds in (4, 32, 128):
        block = baseline_layout(real_bank, counts, torch.empty(0, dtype=torch.long), holds_per_block=holds)[0]
        cases.append(("real_train000_"+str(holds)+"_holds", sweep_bank(block)))
    rows = []
    for name, bank in cases:
        start = time.perf_counter()
        plan = LevelSweepPlan.build(bank)
        build_seconds = time.perf_counter()-start
        generator = torch.Generator().manual_seed(20261027)
        raw = [torch.randn(len(bank.parent), 32, generator=generator),
               torch.rand(len(bank.parent), 32, generator=generator),
               torch.rand(len(bank.parent), 32, generator=generator)]
        reference = run(model, bank, raw)
        result = run(candidate, bank, raw, plan)
        parity = compare(reference, result)
        timings = {"original": [reference["seconds"]], "candidate": [result["seconds"]]}
        # Warm paired order alternates to reduce systematic ordering bias.
        if name.endswith("32_holds") or name.endswith("128_holds"):
            for arm_order in (("candidate", "original"), ("original", "candidate")):
                for arm in arm_order:
                    timed = run(model if arm == "original" else candidate, bank, raw,
                                None if arm == "original" else plan)
                    compare(reference, timed)
                    timings[arm].append(timed["seconds"])
        checkpoint_parity = None
        if name.endswith("4_holds") or name == "permuted_node_ids":
            checkpoint_parity = compare(run(model, bank, raw, double_sweep=True, checkpointed=True),
                run(candidate, bank, raw, plan, double_sweep=True, checkpointed=True))
        rows.append(dict(case=name, nodes=len(bank.parent), upward_levels=len(bank.upward_levels),
            downward_levels=len(bank.downward_levels), plan_bytes=plan.tensor_bytes(),
            plan_build_seconds=build_seconds, parity=parity, two_sweep_checkpoint_parity=checkpoint_parity,
            timings=timings, median_total_speedup=statistics.median(x["total"] for x in timings["original"])
                / statistics.median(x["total"] for x in timings["candidate"])))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(schema="treeloom_level_local_sweep_cpu_v1", all_passed=True, CPU_threads=1,
        CUDA_context_created=False, hypothesis="treeloom_efficiency_redesign_20261027", cases=rows,
        scope="Exact sweep component. Real topology uses seeded CPU base/gate probes, not a complete learned-site/history/decoder training step. No GPU speed or ancestry result.",
        sources=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in
                 (OUT/"level_sweep.py", OUT/"test_level_sweep.py", Path("src/arggnn/anonymous_clade_gnn_v1.py"),
                  Path("src/arggnn/mutation_routed_pyg_gnn_v1.py"))])
    (OUT/"cpu_results.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()

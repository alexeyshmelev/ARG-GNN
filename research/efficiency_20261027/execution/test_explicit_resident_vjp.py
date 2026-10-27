"""CPU full-history composed derivative with a strided external context VJP."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import ast
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.scalability_20261024.execution.measure_representative import baseline_layout
from research.scalability_20261024.integration.executors import IntegratedStagedExecution
from research.efficiency_20261027.execution.explicit_resident_vjp import ExplicitResidentVJPExecution
from research.benchmark_rebuild_20261024.american.treeloom_speed import static_model_v2 as frozen
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import NullProfileV1
from research.repairs.benchmark_recovery_20261020.runtime.episodes import build_episodes
from research.repairs.benchmark_recovery_20261020.runtime.cuts import prepare_cut_plan_v1
from research.efficiency_20261027.root.compare_execution import ref, write, CHECKPOINT, ROOT


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path("research/efficiency_20261027/execution")
    ast.parse((out/"explicit_resident_vjp.py").read_text())
    packet = Path("research/scalability_20261024/execution/representative_slice.npz")
    meta = json.loads(packet.with_suffix(".json").read_text())
    assert ref(packet) == meta["packet_ref"]
    with np.load(packet, allow_pickle=False) as data:
        arrays = {}
        for field in fields(MutationRoutedWholeTreeInputsV1):
            name = field.name
            arrays[name] = (tuple(data[name+"_"+str(i)] for i in range(meta["level_tuple_lengths"][name]))
                if name in meta["level_tuple_lengths"] else data[name])
        offsets = data["node_offsets"]
        counts = np.diff(offsets).tolist()
        bank = MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
    isolated = tuple(i for i, (a, b) in enumerate(zip(offsets, offsets[1:]))
        if b-a == 1600 and bool((bank.parent[int(a):int(b)] < 0).all()))
    episodes, _ = build_episodes(bank, max_payload_bytes=2**30, isolated_hold_indices=isolated)
    cut, _ = prepare_cut_plan_v1(bank, expected_haplotypes=1600, max_nodes=400000,
        max_holds=128, isolated_hold_indices=isolated)
    bundle = SimpleNamespace(bank=bank, episodes=episodes, cut=cut)
    saved = torch.load(ROOT/CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    model = frozen.AmericanTreeLoomC3FullV1()
    model.load_state_dict(saved["model_state"], strict=True)
    layout = baseline_layout(bank, counts, episodes.member_hold, holds_per_block=32)
    assert len(layout) == 4
    results = []
    def run(cls):
        selected = copy.deepcopy(model)
        for name, parameter in selected.named_parameters():
            parameter.grad = (torch.full_like(parameter, .125) if name.startswith(
                ("forward_gru.", "reverse_gru.", "classifier.")) else None)
        executor = object.__new__(cls)
        executor.model, executor.bank, executor.bundle = selected, bank, bundle
        executor.device = torch.device("cpu")
        executor.tick = lambda label: None
        executor.profile = NullProfileV1()
        executor.layout = layout
        executor.history_window = 17
        executor.hap_tile = 32
        tick = time.perf_counter()
        context = executor.context_forward()
        forward_seconds = time.perf_counter()-tick
        generator = torch.Generator().manual_seed(427)
        backing = torch.randn((*context.shape[:-1], context.shape[-1]*2), generator=generator)
        gradient = backing[..., ::2]/context.numel()
        # Division creates a contiguous result, so copy it into a strided view.
        backing[..., ::2].copy_(gradient)
        gradient = backing[..., ::2]
        assert not gradient.is_contiguous()
        loss = (context.detach()*gradient).sum()
        context.backward(gradient)
        gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in selected.named_parameters()}
        receipt = dict(kind=cls.__name__, seconds=time.perf_counter()-tick,
            forward_seconds=forward_seconds, loss=float(loss.detach()),
            explicit_vjp=executor.explicit_vjp_snapshot() if hasattr(executor, "explicit_vjp_snapshot") else None)
        print(json.dumps(receipt), flush=True)
        return context.detach(), gradients, receipt
    original = run(IntegratedStagedExecution)
    candidate = run(ExplicitResidentVJPExecution)
    torch.testing.assert_close(original[0], candidate[0], atol=1e-6, rtol=1e-5)
    errors, connected = {}, 0
    for name, value in original[1].items():
        other = candidate[1][name]
        assert (value is None) == (other is None), name
        if value is None:
            continue
        torch.testing.assert_close(value, other, atol=1e-6, rtol=1e-4, msg=name)
        errors[name] = float((value-other).abs().max())
        connected += 1
        if name.startswith(("forward_gru.", "reverse_gru.", "classifier.")):
            assert bool((other == .125).all()), name
    proof = candidate[2]["explicit_vjp"]
    assert proof["second_backward_blocks"] == proof["forcing_backward_blocks"] == proof["first_backward_blocks"] == 4
    assert proof["full_history_backward_calls"] == 1
    assert not torch.cuda.is_initialized()
    result = dict(schema="TreeLoom_explicit_resident_VJP_CPU_parity_v1", status="pass",
        CUDA_initialized=False, actual_native_holds=128, all_haplotypes=1600,
        holds_per_block=32, blocks=4, history_holds_per_window=17,
        complete_bidirectional_slice_history=True, source_refs=[ref(out/"explicit_resident_vjp.py"), ref(__file__), ref(CHECKPOINT)],
        output_max_abs_error=float((original[0]-candidate[0]).abs().max()),
        parameter_gradient_max_abs_errors=errors, gradient_tensors_including_preexisting_decoder=connected,
        preexisting_decoder_gradients_preserved=True, noncontiguous_external_context_VJP=True, runs=[original[2], candidate[2]],
        scope="Actual128hold observed packet/all1600haps and checkpoint241 trunk weights. Full restricted-slice bidirectional history across8windows and4treeblocks. Noncontiguous external context VJP; no GPU residency, actual focal decoder, optimizer or full-parent performance claim.")
    write(out/"explicit_resident_vjp_cpu_results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

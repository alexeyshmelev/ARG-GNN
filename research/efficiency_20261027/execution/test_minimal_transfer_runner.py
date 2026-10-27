"""CPU-only runner/accounting checks; no CUDA, labels, optimizer or disk cache."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import ast
from collections import Counter
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.scalability_20261024.execution.measure_representative import baseline_layout
from research.efficiency_20261027.execution.compare_minimal_transfer import (
    KINDS, BlockAudit, compare, schedule, ref, write)
from research.efficiency_20261027.execution.executors_v2 import MinimalTransferLevelExecution
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import StagedExecutionV1
from research.scalability_20261024.integration.executors import IntegratedStagedExecution
from research.efficiency_20261027.root.context_residency import ResidentContextExecution


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    here = Path("research/efficiency_20261027/execution")
    for path in (here/"compare_minimal_transfer.py", here/"test_minimal_transfer_runner.py"):
        ast.parse(path.read_text(), filename=str(path))
    positions = {kind: Counter() for kind in KINDS}
    for repeat, order in schedule():
        assert len(order) == len(KINDS) and set(order) == set(KINDS)
        for position, kind in enumerate(order):
            positions[kind][position] += 1
    assert all(set(counts.values()) == {2} and len(counts) == 4 for counts in positions.values())
    for invalid in (0, 5, 6, 7, 9, 24):
        try:
            schedule(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Unbalanced schedule accepted")
    probe = dict(logits=torch.tensor([[.1, .2, .3]]), loss=torch.tensor(.7),
        gradients={"p": torch.tensor([.1])}, parameters={"p": torch.tensor([.2])},
        moments={"0": dict(step=torch.tensor(242.), exp_avg=torch.tensor([.01]), exp_avg_sq=torch.tensor([.001]))})
    assert compare(probe, probe)["argmax_disagreements"] == 0
    bad = dict(probe, gradients={"p": torch.tensor([.1002])})
    try:
        compare(probe, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Gradient mismatch accepted")
    packet = Path("research/scalability_20261024/execution/representative_slice.npz")
    meta = json.loads(packet.with_suffix(".json").read_text())
    assert ref(packet) == meta["packet_ref"]
    with np.load(packet, allow_pickle=False) as data:
        arrays = {}
        for field in fields(MutationRoutedWholeTreeInputsV1):
            name = field.name
            if name in meta["level_tuple_lengths"]:
                arrays[name] = tuple(data[name+"_"+str(i)] for i in range(meta["level_tuple_lengths"][name]))
            else:
                arrays[name] = data[name]
        counts = np.diff(data["node_offsets"]).tolist()
        bank = MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
    layout = baseline_layout(bank, counts, torch.empty(0, dtype=torch.long), holds_per_block=128)
    assert len(layout) == 1
    z = layout[0]
    results, outputs = {}, {}
    for name, base in (("original", StagedExecutionV1), ("integrated", IntegratedStagedExecution),
                       ("level_local_v2", MinimalTransferLevelExecution), ("context_gpu_block_recipe", ResidentContextExecution)):
        class Audited(BlockAudit, base):
            pass
        executor = object.__new__(Audited)
        executor.bank, executor.device, executor.layout = bank, torch.device("cpu"), ()
        for visit in range(4):
            outputs[name] = executor._block(z)
        results[name] = executor.transfer_snapshot()
        assert results[name]["observed_block_invocations"] == 4
        assert results[name]["CUDA_event_interval_ms"] == 0
    original, minimal = outputs["original"], outputs["level_local_v2"]
    for key, value in vars(minimal).items():
        torch.testing.assert_close(value, getattr(original, key), atol=0, rtol=0)
    excluded = set(vars(original)) - set(vars(minimal))
    assert excluded == {"upward_levels", "upward_children", "upward_parent_rank", "downward_levels"}
    diff = {key: sum(results["original"]["fields"][name][key] for name in excluded)
            for key in ("tensor_calls", "nonempty_tensor_calls", "payload_bytes")}
    for key in diff:
        assert diff[key] == results["original"][key]-results["level_local_v2"][key]
    assert not torch.cuda.is_initialized()
    result = dict(schema="minimal_transfer_runner_CPU_semantics_v1", status="pass",
        CUDA_initialized=False, source_refs=[ref(here/"compare_minimal_transfer.py"), ref(__file__)],
        schedule_position_counts={kind: dict(counts) for kind, counts in positions.items()},
        strict_comparator_rejects_gradient_mismatch=True,
        actual_128hold_block_recipe_CPU_tests=results,
        four_explicit_CPU_visits_removed_topology=diff,
        per_visit_removed_topology={key: value//4 for key, value in diff.items()},
        shared_consumed_fields_bitwise_equal=True,
        scope="CPU recipe/instrumentation semantics only. Four explicit test visits are not observed GPU checkpoint counts. Runtime diagnostic will count actual calls and H2D payload separately. No labels or checkpoint loaded; no disk caches created.")
    write(here/"runner_cpu_semantics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

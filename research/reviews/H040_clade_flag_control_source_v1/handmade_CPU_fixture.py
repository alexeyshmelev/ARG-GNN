"""One bounded handmade CPU property fixture. No project bank or checkpoint reads."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from arggnn.anonymous_clade_flag_control_v1 import (
    AnonymousCladeFlagsV1, AnonymousCladeFlagControlGNNV1,
    RankEightCladeFlagAdapterV1, build_clade_flag_arrays_v1,
    load_h038_flag_warmstart_v1,
)
from arggnn.anonymous_clade_gnn_v1 import (
    AnonymousCladeTensorGNNV1, load_h038_warmstart_v1,
)
from arggnn.mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1


def forest(trees):
    parents, rows, representatives = [], [], []
    def visit(tree, row, record):
        if isinstance(tree, int):
            node = len(parents)
            parents.append(-1)
            row[tree] = node
            return node, 1 << tree
        parts = [visit(child, row, record) for child in tree]
        node = len(parents)
        parents.append(-1)
        mask = 0
        for child, child_mask in parts:
            parents[child] = node
            mask |= child_mask
        record.setdefault(mask, node)  # Children precede parents: deepest representative.
        return node, mask
    for tree in trees:
        row, record = [-1] * 7, {}
        visit(tree, row, record)
        rows.append(row)
        representatives.append(record)
    return (np.asarray(parents, dtype=np.int64),
            np.asarray(rows, dtype=np.int64), representatives)


def run():
    torch.set_num_threads(1)
    torch.manual_seed(101)
    landmarks = ((0, 1), (2, 3))
    left = (landmarks, ((4, 5), 6))
    right = (landmarks, (4, (5, 6)))
    parent, samples, reps = forest((left, right))
    flags, meta = build_clade_flag_arrays_v1(parent, samples)
    expected = np.zeros_like(flags)
    expected[reps[0][(1 << 4) | (1 << 5)], 1] = 1
    expected[reps[1][(1 << 5) | (1 << 6)], 0] = 1
    assert np.array_equal(flags, expected)
    assert meta["proper_clades_per_tree"] == [5, 5]
    assert meta["gained_from_left_count"] == meta["lost_to_right_count"] == 1

    # Internal/global node reindexing changes indexes, never feature semantics.
    order = np.random.default_rng(101).permutation(len(parent))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    reindexed_parent = parent[order].copy()
    nonroot = reindexed_parent >= 0
    reindexed_parent[nonroot] = inverse[reindexed_parent[nonroot]]
    reindexed, reindexed_meta = build_clade_flag_arrays_v1(
        reindexed_parent, inverse[samples],
    )
    assert np.array_equal(reindexed, flags[order])
    column_order = np.asarray([6, 2, 4, 0, 5, 1, 3])
    column_flags, _ = build_clade_flag_arrays_v1(parent, samples[:, column_order])
    assert np.array_equal(column_flags, flags)
    reversed_flags, _ = build_clade_flag_arrays_v1(parent, samples[::-1])
    assert np.array_equal(reversed_flags, flags[:, ::-1])

    unary_left = (landmarks, (((4, 5),), 6))
    unary_parent, unary_samples, unary_reps = forest((unary_left, right))
    unary_flags, unary_meta = build_clade_flag_arrays_v1(unary_parent, unary_samples)
    unary_expected = np.zeros_like(unary_flags)
    unary_expected[unary_reps[0][48], 1] = 1
    unary_expected[unary_reps[1][96], 0] = 1
    assert np.array_equal(unary_flags, unary_expected)
    assert unary_meta["unary_equivalences_per_tree"] == [1, 0]
    identical_parent, identical_samples, _ = forest((left, unary_left))
    identical_flags, _ = build_clade_flag_arrays_v1(identical_parent, identical_samples)
    assert not identical_flags.any()

    payload = AnonymousCladeFlagsV1.from_verified_arrays(flags, meta)
    adapter = RankEightCladeFlagAdapterV1()
    upward = torch.linspace(-.3, .7, len(parent) * 32, dtype=torch.float32)
    upward = upward.reshape(len(parent), 32).requires_grad_()
    output = adapter(upward, payload)
    assert not output.detach().count_nonzero()
    output.sum().backward()
    initial_norms = {
        name: float(parameter.grad.norm()) for name, parameter in adapter.named_parameters()
    }
    assert initial_norms["O.weight"] > 0
    assert all(initial_norms[name + ".weight"] == 0 for name in ("U", "V", "W"))
    assert not upward.grad.count_nonzero()

    # Move only O by a deterministic assignment, not by an optimizer update.
    with torch.no_grad():
        adapter.O.weight.copy_(torch.linspace(-.1, .2, 256).reshape(32, 8))
    adapter.zero_grad(set_to_none=True)
    upward.grad = None
    output = adapter(upward, payload)
    output.square().sum().backward()
    later_norms = {
        name: float(parameter.grad.norm()) for name, parameter in adapter.named_parameters()
    }
    assert all(value > 0 for value in later_norms.values())
    assert torch.isfinite(upward.grad).all()
    assert upward.grad.norm() > 0

    shuffled_payload = AnonymousCladeFlagsV1.from_verified_arrays(
        reindexed, reindexed_meta,
    )
    shuffled_output = adapter(upward.detach()[order], shuffled_payload)
    assert torch.allclose(shuffled_output, output.detach()[order], atol=1e-7, rtol=1e-6)

    # Toy random backbone states verify schema/init only, never real checkpoint weights.
    torch.manual_seed(101)
    base = MutationRoutedFullPanelPyGGNNV1(hidden_dim=32, expected_haplotypes=952)
    toy_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    toy_payload = dict(arm="aligned", seed=101, teacher_used=False, model_state=toy_state)
    torch.manual_seed(101)
    tensor_model = AnonymousCladeTensorGNNV1()
    load_h038_warmstart_v1(tensor_model, toy_payload)
    torch.manual_seed(101)
    flag_model = AnonymousCladeFlagControlGNNV1()
    proof = load_h038_flag_warmstart_v1(flag_model, toy_payload)
    assert sum(p.numel() for p in base.parameters()) == 29764
    assert sum(p.numel() for p in adapter.parameters()) == 1024
    assert sum(p.numel() for p in flag_model.parameters()) == 30788
    assert tensor_model.state_dict().keys() == flag_model.state_dict().keys()
    assert all(torch.equal(value, flag_model.state_dict()[key])
               for key, value in tensor_model.state_dict().items())
    assert flag_model.sweep_tree_state_v1.__func__ is tensor_model.sweep_tree_state_v1.__func__
    source = Path("src/arggnn/anonymous_clade_flag_control_v1.py")
    return dict(
        schema="exact_clade_flag_tiny_CPU_fixture_v1", passed=True,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cpu_only=True, floating_dtype="float32", handmade_haplotypes=7,
        handmade_holds=2, handmade_native_nodes=len(parent),
        gained_flags=1, lost_flags=1, flags_dtype=str(flags.dtype),
        node_and_sample_column_covariance=True, reversal_swaps_flag_channels=True,
        unary_representative_and_equivalence=True, identical_topology_zero=True,
        exact_nonlandmark_rearrangement_detected=True,
        adapter_covariance=True, zero_O_output_and_gradient_gate=True,
        initial_gradient_norms=initial_norms, nonzero_O_gradient_norms=later_norms,
        same_initial_state_keys_and_values_as_H040_with_toy_backbone=True,
        frozen_tree_sweep_function_reused=True, parameter_proof=proof,
        project_data_or_checkpoint_read=False, optimizer_updates=0,
        model_full_forward_or_training_run=False,
        torch_version=torch.__version__, numpy_version=np.__version__,
    )


if __name__ == "__main__":
    receipt = run()
    target = Path(__file__).with_name("source_CPU_check.json")
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))

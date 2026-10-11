"""Tiny CPU FP32 equivalence probe, with handmade observations only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from arggnn.mutation_routed_whole_tree_gnn_v1 import (
    MutationRoutedWholeTreeGNNV1, MutationRoutedWholeTreeInputsV1,
    validate_inputs_v1,
)
from arggnn.mutation_routed_pyg_gnn_v1 import MutationRoutedPyGGNNV1


def handmade_bank():
    sample_order = [2, 0, 5, 1, 7, 3, 6, 4]
    ages = torch.tensor([0., 0., 50., 130., 200., 350., 420., 700.])
    local_parents = [
        [8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, -1],
        [9, 8, 8, 8, 8, 8, 8, 8, -1, 10, 8],
        [8, 8, 9, 10, 11, 12, 13, 14, 9, 10, 11, 12, 13, 14, -1],
    ]
    parents, clocks, samples = [], [], []
    offset = 0
    for hold, local in enumerate(local_parents):
        times = torch.zeros(len(local), dtype=torch.float32)
        times[torch.tensor(sample_order)] = ages
        if hold == 1:
            times[8:] = torch.tensor([2000., 1100., 1200.])
        else:
            times[8:] = torch.arange(1100., 1800., 100.)
        parents.extend(p + offset if p >= 0 else -1 for p in local)
        clocks.append(times)
        samples.append(torch.tensor(sample_order) + offset)
        offset += len(local)
    parent = torch.tensor(parents, dtype=torch.long)
    node_time = torch.cat(clocks)
    children = [[] for _ in parents]
    for child, ancestor in enumerate(parents):
        if ancestor >= 0:
            children[ancestor].append(child)
    pending = [len(ch) for ch in children]
    current = [v for v, count in enumerate(pending) if not count]
    levels, gathered, ranks = [], [], []
    while current:
        levels.append(torch.tensor(current, dtype=torch.long))
        gathered.append(torch.tensor(
            [c for v in current for c in children[v]], dtype=torch.long))
        ranks.append(torch.tensor(
            [rank for rank, v in enumerate(current) for _ in children[v]],
            dtype=torch.long))
        nxt = []
        for v in current:
            p = parents[v]
            if p >= 0:
                pending[p] -= 1
                if pending[p] == 0:
                    nxt.append(p)
        current = nxt
    current = [v for v, p in enumerate(parents) if p < 0]
    down = []
    while current:
        current = [c for v in current for c in children[v]]
        if current:
            down.append(torch.tensor(current, dtype=torch.long))
    n = len(parents)
    geometry = torch.zeros(n, 4, dtype=torch.float32)
    descendants = torch.ones(n, dtype=torch.float32)
    for level in levels:
        for v in level.tolist():
            if children[v]:
                descendants[v] = sum(descendants[c] for c in children[v])
    active = parent >= 0
    geometry[active, 0] = node_time[active].log1p()/math.log(10001.)
    geometry[active, 1] = node_time[parent[active]].log1p()/math.log(10001.)
    geometry[active, 2] = (
        node_time[parent[active]]-node_time[active]).log1p()/math.log(10001.)
    geometry[active, 3] = descendants[active]/8
    generator = torch.Generator().manual_seed(701)
    support = torch.rand(n, 5, generator=generator)
    support[:, 3] = torch.arange(n, dtype=torch.float32) % 7 + 1.
    support[:, 4] = 0
    support[~active] = torch.tensor([0., 0., 0., 0., 1.])
    sham = support.clone()
    for start, local in zip([0, 15, 26], local_parents):
        rows = torch.tensor([start+i for i, p in enumerate(local) if p >= 0])
        sham[rows] = support[rows.flip(0)]
    genotype = torch.tensor([
        [0, 1, 2, 0, 1, 2, 0, 1],
        [1, -1, 0, 2, 1, 1, 0, 2],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [2, 1, 0, 0, 1, 2, 2, 0],
        [0, 0, 1, 1, 2, 2, -1, 0],
        [1, 2, 1, 0, 2, 0, 1, 2],
        [2, 0, 2, 1, 0, 1, 2, 1],
    ], dtype=torch.int8)
    donor = torch.rand(7, 3, 8, generator=generator).softmax(1).log()
    mask = torch.ones(7, 8, dtype=torch.bool)
    mask[:, 7] = False
    mask[2] = False
    donor[:, :, 7] = -torch.inf
    panel = torch.rand(7, 3, generator=generator).softmax(1)
    return MutationRoutedWholeTreeInputsV1(
        parent=parent, node_time=node_time, sample_index=torch.stack(samples),
        upward_levels=tuple(levels), upward_children=tuple(gathered),
        upward_parent_rank=tuple(ranks), downward_levels=tuple(down),
        edge_geometry=geometry, branch_support_true=support,
        branch_support_sham=sham, genotype=genotype,
        log_donor_allele_frequency=donor, donor_call_mask=mask,
        panel_allele_frequency=panel, site_entropy=torch.ones(7),
        site_hold=torch.tensor([0, 0, 0, 2, 2, 2, 2], dtype=torch.long),
        spans=torch.tensor([1000., 700., 1300.]),
        readout_hold=torch.tensor([0, 0, 1, 2, 2], dtype=torch.long),
        query_column=torch.tensor([0, 2, 1, 3], dtype=torch.long),
        reference_class=torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3]),
        sample_age=ages,
    )


def main():
    import torch_geometric
    torch.set_num_threads(1)
    torch.manual_seed(101)
    bank = handmade_bank()
    structure = validate_inputs_v1(bank, expected_haplotypes=8)
    old = MutationRoutedWholeTreeGNNV1(
        expected_haplotypes=8, site_chunk_size=2, checkpoint_sites=True)
    new = MutationRoutedPyGGNNV1(
        expected_haplotypes=8, site_chunk_size=2, checkpoint_sites=True)
    old_keys, new_keys = list(old.state_dict()), list(new.state_dict())
    if old_keys != new_keys:
        raise AssertionError("Legacy checkpoint keys/order changed")
    if [n for n, _ in old.named_parameters()] != [
            n for n, _ in new.named_parameters()]:
        raise AssertionError("Legacy parameter order changed")
    loaded = new.load_state_dict(old.state_dict(), strict=True)
    count = sum(p.numel() for p in new.parameters())
    if count != 29764 or count != sum(p.numel() for p in old.parameters()):
        raise AssertionError("Width32 learned parameter count changed")
    old.train()
    new.train()
    rows = []
    for control in ("aligned", "shuffled"):
        old.zero_grad(set_to_none=True)
        new.zero_grad(set_to_none=True)
        left = old(bank, control=control)
        right = new(bank, control=control)
        probe = torch.linspace(-.9, 1.2, left.numel()).reshape_as(left)
        loss_old = (left*probe).sum() + .13*left.square().sum()
        loss_new = (right*probe).sum() + .13*right.square().sum()
        loss_old.backward()
        loss_new.backward()
        forward_max_abs = float((left-right).abs().max())
        gradients = []
        for (name, a), (other, b) in zip(
                old.named_parameters(), new.named_parameters()):
            if name != other or a.grad is None or b.grad is None:
                raise AssertionError("Parameter gradient is absent/misaligned")
            if not bool(torch.isfinite(a.grad).all() & torch.isfinite(b.grad).all()):
                raise AssertionError("Nonfinite parameter gradient")
            diff = float((a.grad-b.grad).abs().max())
            scale = max(1., float(a.grad.abs().max()))
            gradients.append({"name": name, "max_abs": diff, "scaled_max_abs": diff/scale})
        max_grad_abs = max(r["max_abs"] for r in gradients)
        max_grad_scaled = max(r["scaled_max_abs"] for r in gradients)
        if forward_max_abs > 2e-6 or max_grad_scaled > 1e-5:
            raise AssertionError("FP32 handmade model/gradient mismatch")
        if new.pyg_upward._edge_messages is not None:
            raise AssertionError("An ephemeral propagation graph was retained")
        rows.append({
            "control": control, "forward_shape": list(left.shape),
            "forward_max_abs": forward_max_abs,
            "loss_max_abs": float((loss_old-loss_new).abs()),
            "parameter_gradient_max_abs": max_grad_abs,
            "parameter_gradient_scaled_max_abs": max_grad_scaled,
            "all_parameter_gradients_finite_and_present": True,
            "parameter_gradients": gradients,
        })
    directory = Path("research/reviews/mutation_routed_pyg_gnn_v1_author")
    source = Path("src/arggnn/mutation_routed_pyg_gnn_v1.py")
    original = Path("src/arggnn/mutation_routed_whole_tree_gnn_v1.py")
    receipt = {
        "schema": "H036_PyG_handmade_FP32_equivalence_v1",
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "device": "cpu", "dtype": "float32", "optimizer_or_actual_data_run": False,
        "handmade_structure": structure,
        "covers": ["binary_tree", "unary_branches", "polytomy", "unbalanced_tree",
                   "nonidentity_sample_node_join", "heterochronous_leaves",
                   "missing_alleles", "inactive_donor_channel",
                   "empty_site_hold", "site_checkpoint_recomputation",
                   "aligned_and_shuffled_support"],
        "strict_state_load_missing": list(loaded.missing_keys),
        "strict_state_load_unexpected": list(loaded.unexpected_keys),
        "state_keys_and_parameter_order_equal": True,
        "state_dict_key_count": len(old_keys),
        "learned_parameter_count": count,
        "new_message_parameters": sum(p.numel() for module in [
            new.pyg_upward, new.pyg_downward] for p in module.parameters()),
        "forward_tolerance": 2e-6, "gradient_scaled_tolerance": 1e-5,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "rows": rows,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"toy_check.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="rows"}, indent=2))
    print(json.dumps([{
        k:v for k,v in row.items() if k!="parameter_gradients"} for row in rows], indent=2))


if __name__ == "__main__":
    main()

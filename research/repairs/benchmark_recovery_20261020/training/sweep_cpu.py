"""CPU phase check: unchanged PyG cavity sweeps in 16 versus 128 hold blocks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from research.reviews.frozen_benchmark_training_ready_20261017_v1.american_full30mb_frozen_v1 import AmericanTreeLoomC3FullV1
from .layout import block_indices


def main():
    torch.set_num_threads(4)
    torch.manual_seed(70019)
    holds, leaves, width = 128, 32, 32
    nodes = 2*leaves-1
    parent = np.full(nodes, -1, dtype=np.int64)
    height = np.zeros(nodes, dtype=np.int16)
    for index in range(leaves-1):
        current = leaves+index
        parent[0 if index == 0 else current-1] = current
        parent[index+1] = current
        height[current] = index+1
    depth = np.zeros(nodes, dtype=np.int16)
    for node in range(nodes-2, -1, -1):
        depth[node] = depth[parent[node]]+1
    full_parent = np.concatenate([np.where(parent < 0, -1, parent+at*nodes) for at in range(holds)])
    full_height, full_depth = np.tile(height, holds), np.tile(depth, holds)
    base = torch.randn(holds*nodes, width)
    gate_up, gate_down = torch.rand_like(base), torch.rand_like(base)
    degree = torch.from_numpy(np.bincount(full_parent[full_parent >= 0], minlength=len(full_parent))).float().clamp_min(1)
    reference = AmericanTreeLoomC3FullV1()
    candidate = copy.deepcopy(reference)

    def run(model, block_holds):
        start = time.perf_counter()
        value = base.clone().requires_grad_(True)
        outputs = []
        for first in range(0, holds, block_holds):
            left, right = first*nodes, min(first+block_holds, holds)*nodes
            p = full_parent[left:right].copy()
            p[p >= 0] -= left
            levels, children, ranks, downward = block_indices(p, full_height[left:right], full_depth[left:right])
            bank = SimpleNamespace(parent=torch.from_numpy(p),
                upward_levels=tuple(torch.from_numpy(row) for row in levels),
                upward_children=tuple(torch.from_numpy(row) for row in children),
                upward_parent_rank=tuple(torch.from_numpy(row) for row in ranks),
                downward_levels=tuple(torch.from_numpy(row) for row in downward))
            def sweep(b, u, d, deg, bank=bank):
                return model.sweep_tree_state_v1(bank, b, u, d, deg)
            outputs.append(checkpoint(sweep, value[left:right], gate_up[left:right],
                gate_down[left:right], degree[left:right], use_reentrant=False))
        upward = torch.cat([item[0] for item in outputs])
        outside = torch.cat([item[1] for item in outputs])
        forward_seconds = time.perf_counter()-start
        loss = upward.square().mean() + outside.square().mean()
        start = time.perf_counter()
        loss.backward()
        return upward.detach(), outside.detach(), value.grad, forward_seconds, time.perf_counter()-start

    run(reference, 16)
    reference.zero_grad(set_to_none=True)
    expected, actual = run(reference, 16), run(candidate, 128)
    max_value = max(float((a-b).abs().max()) for a,b in zip(expected[:2],actual[:2]))
    max_input_gradient = float((expected[2]-actual[2]).abs().max())
    assert all(torch.allclose(a,b,rtol=5e-4,atol=2e-6) for a,b in zip(expected[:3],actual[:3]))
    errors = []
    for (name, parameter), (other_name, other) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name == other_name
        if parameter.grad is not None:
            assert other.grad is not None
            error = float((parameter.grad-other.grad).abs().max())
            assert torch.allclose(parameter.grad,other.grad,rtol=5e-4,atol=2e-6),(name,error)
            errors.append(error)
        else:
            assert other.grad is None
    report = dict(schema="TreeLoom_sweep_CPU_phase_v1", passed=True, CUDA_used=False,
        observed_data_reads=0, synthetic_holds=holds, leaves_per_hold=leaves,
        levels=leaves, original_block_holds=16, candidate_block_holds=128,
        original_forward_seconds=expected[3], candidate_forward_seconds=actual[3],
        original_backward_seconds=expected[4], candidate_backward_seconds=actual[4],
        max_forward_absolute_error=max_value, max_input_gradient_absolute_error=max_input_gradient,
        max_parameter_gradient_absolute_error=max(errors),
        checkpoint_path_warmed=True, production_GPU_peak_or_speed_claimed=False)
    (Path(__file__).parent/'sweep_cpu.result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True))


if __name__ == '__main__':
    main()

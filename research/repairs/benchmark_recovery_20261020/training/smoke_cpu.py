"""Synthetic CPU VJP/update equivalence and phase benchmarks, with zero CUDA."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .decoder import decode_parent_backward, geometry_from_spans
from .layout import block_indices
from .progress import StageProgress


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(11, 32)
        self.forward_gru = nn.GRU(35, 32)
        self.reverse_gru = nn.GRU(35, 32)
        self.classifier = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 3))

    def scan_context(self, context, geometry, *, direction):
        sequence = torch.cat((context, geometry[:, None, :].expand(-1, context.shape[1], -1)), -1)
        if direction == "reverse":
            history, state = self.reverse_gru(sequence.flip(0))
            return history.flip(0), state
        return self.forward_gru(sequence)


def original_backward(model, inputs, geometry, holds, columns, labels):
    context = model.trunk(inputs).tanh()
    result = context.new_empty((len(labels), 3))
    zero = context.new_zeros(())
    def decode(tile, focal_hold, focal_column):
        forward, _ = model.scan_context(tile, geometry, direction="forward")
        reverse, _ = model.scan_context(tile, geometry, direction="reverse")
        logits = model.classifier(torch.cat((tile[focal_hold, focal_column],
            forward[focal_hold, focal_column], reverse[focal_hold, focal_column]), -1))
        return logits, forward.sum()*0 + reverse.sum()*0
    for first in range(0, context.shape[1], 8):
        rows = torch.nonzero((columns >= first) & (columns < first+8), as_tuple=False).flatten()
        logits, dummy = checkpoint(decode, context[:, first:first+8], holds[rows],
            columns[rows]-first, use_reentrant=False, preserve_rng_state=True)
        result = result.index_copy(0, rows, logits)
        zero = zero + dummy
    loss = F.cross_entropy(result+zero, labels)
    loss.backward()
    return float(loss.detach())


def decoder_check():
    torch.manual_seed(70019)
    torch.set_num_threads(4)
    inputs = torch.randn(96, 96, 11)
    holds = torch.randint(0, len(inputs), (1000,))
    columns = torch.randint(32, 96, (1000,))
    labels = torch.randint(0, 3, (1000,))
    geometry = geometry_from_spans(torch.rand(96)*1000+1)
    baseline = Decoder()
    candidate = copy.deepcopy(baseline)
    original_optimizer = torch.optim.AdamW(baseline.parameters(), lr=.001, weight_decay=.0001)
    candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=.001, weight_decay=.0001)
    warmup_model = copy.deepcopy(baseline)
    original_backward(warmup_model, inputs, geometry, holds, columns, labels)
    del warmup_model
    start = time.perf_counter()
    expected_loss = original_backward(baseline, inputs, geometry, holds, columns, labels)
    original_seconds = time.perf_counter()-start
    start = time.perf_counter()
    context = candidate.trunk(inputs).tanh()
    loss, gradient, receipt = decode_parent_backward(candidate, context, geometry, holds,
        columns, labels, tile_haplotypes=32)
    context.backward(gradient)
    staged_seconds = time.perf_counter()-start
    errors = {}
    for (name, expected), (actual_name, actual) in zip(baseline.named_parameters(), candidate.named_parameters()):
        assert name == actual_name and actual.grad is not None
        error = float((expected.grad-actual.grad).abs().max())
        assert torch.allclose(expected.grad, actual.grad, rtol=5e-4, atol=2e-7), (name, error)
        errors[name] = error
    torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
    original_optimizer.step()
    candidate_optimizer.step()
    update_error = max(float((a-b).abs().max()) for a,b in zip(baseline.parameters(),candidate.parameters()))
    assert update_error < 2e-6, update_error
    assert abs(float(loss)-expected_loss) < 1e-6
    assert receipt["zero_focal_tiles"] == 1 and receipt["native_focals"] == 1000
    assert all(float(value["step"]) == 1 for value in candidate_optimizer.state.values())
    return dict(passed=True, original_seconds=original_seconds, staged_seconds=staged_seconds,
        CPU_speed_ratio=original_seconds/staged_seconds, shape=list(inputs.shape),
        loss_absolute_error=abs(float(loss)-expected_loss), max_parameter_gradient_absolute_error=max(errors.values()),
        max_parameter_after_AdamW_absolute_error=update_error,
        complete_trunk_gradient_checked=True, checkpoint_path_warmed=True, optimizer_updates=1, **receipt)


def old_indices(parent, height, depth):
    levels = tuple(np.flatnonzero(height == k) for k in range(int(height.max())+1))
    rank = np.empty(len(parent), dtype=np.int64)
    for rows in levels:
        rank[rows] = np.arange(len(rows))
    child = np.flatnonzero(parent >= 0)
    children = tuple(child[height[parent[child]] == k] for k in range(len(levels)))
    ranks = tuple(rank[parent[rows]] for rows in children)
    downward = tuple(np.flatnonzero(depth == k) for k in range(1, int(depth.max())+1))
    return levels, children, ranks, downward


def layout_check():
    generator = np.random.default_rng(70019)
    size = 1600*256
    parent = generator.integers(0, size, size=size, dtype=np.int64)
    parent[::1600] = -1
    height = generator.integers(0, 128, size=size, dtype=np.int16)
    depth = generator.integers(0, 128, size=size, dtype=np.int16)
    start = time.perf_counter()
    expected = old_indices(parent, height, depth)
    original_seconds = time.perf_counter()-start
    start = time.perf_counter()
    actual = block_indices(parent, height, depth)
    staged_seconds = time.perf_counter()-start
    for expected_family, actual_family in zip(expected, actual):
        assert len(expected_family) == len(actual_family)
        assert all(np.array_equal(a,b) for a,b in zip(expected_family, actual_family))
    return dict(passed=True, nodes=size, levels=128, original_seconds=original_seconds,
        stable_bucket_seconds=staged_seconds, CPU_speed_ratio=original_seconds/staged_seconds,
        all_four_index_families_bitwise_equal=True)


def main():
    root = Path(__file__).resolve().parent
    profile = StageProgress(root/"smoke_status.json", parent_id="synthetic_CPU_only")
    try:
        with profile.stage("layout_benchmark"):
            layout = layout_check()
        profile.mark("layout_checked")
        with profile.stage("decoder_benchmark"):
            decoder = decoder_check()
        profile.mark("smoke_complete")
    finally:
        profile.close()
    report = dict(schema="treeloom_recovery_CPU_smoke_v1", scientific_data_reads=0,
        CUDA_used=False, layout=layout, decoder=decoder,
        runtime_qualification=False, full_parent_speedup_claimed=False)
    (root/"smoke_cpu.result.json").write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

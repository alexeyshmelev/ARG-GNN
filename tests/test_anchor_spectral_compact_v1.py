"""Handmade CPU qualification, never actual project data or holdout arrays."""
from dataclasses import replace
from types import SimpleNamespace
import os

import pytest
import torch
from torch import nn

from arggnn.anchor_kernel import AnchorKernelLayer, AnchorOperators, _layout, anchor_kernel_action
from arggnn.anchor_spectral_compact_v1 import (
    CompactSpectralGeometryV1, anchor_spectral_compact_action_v1,
    compact_anchor_messages_v1, compact_layout_from_anchor_v1,
    prepare_compact_anchor_layout_v1)
import arggnn.anchor_spectral_compact_v1 as compact_module


@pytest.fixture(autouse=True)
def cpu_only_one_thread():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Run with explicitly empty CUDA visibility"
    torch.set_num_threads(1)


def close(actual, expected, dtype, gradient=False):
    if dtype == torch.float64:
        rtol, atol = (1e-8, 1e-10) if gradient else (1e-10, 1e-12)
    else:
        rtol, atol = (1e-3, 1e-5) if gradient else (2e-4, 2e-5)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def fixture(dtype=torch.float64, *, intervals=None, positions=None, anchors=3, reorder=None):
    if intervals is None:
        intervals = [[0., .22], [.22, .5], [.5, 1.]]
    if positions is None:
        positions = [.5, 0., .12, .999, .22, .5, 1 / 6, .82]
    intervals = torch.tensor(intervals, dtype=torch.float64)
    base = torch.tensor([
        [[2., -1., 0.], [-1., 3., -1.], [0., -1., 2.]],
        [[3., 0., -1.], [0., 2., -.5], [-1., -.5, 2.5]],
        [[2.5, -.4, -.3], [-.4, 1.8, -.2], [-.3, -.2, 2.2]]], dtype=torch.float64)
    base = base[:len(intervals)]
    cqq = torch.stack((base, 1.3 * base + .2 * torch.eye(3, dtype=torch.float64)), dim=1)
    ref_fractions = torch.tensor([[.6, .4], [.3, .7], [.8, .2]], dtype=torch.float64)
    cqr = -cqq.sum(-1)[..., None] * ref_fractions
    eigenvalues, eigenvectors = torch.linalg.eigh(cqq)
    if reorder is not None:
        order = torch.tensor(reorder, dtype=torch.long)
        intervals, cqq, cqr = intervals[order], cqq[order], cqr[order]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[order]
    operators = AnchorOperators(*(x.to(dtype) for x in (cqq, cqr, eigenvalues, eigenvectors)))
    positions = torch.tensor(positions, dtype=torch.float64)
    order = torch.argsort(intervals[:, 0])
    graph = order[torch.searchsorted(intervals[order, 0].contiguous(), positions, right=True) - 1]
    bank = SimpleNamespace(intervals=intervals, positions=positions, target_graph=graph,
                           window_bounds=torch.stack((intervals[order[0], 0], intervals[order[-1], 1])))
    native = _layout(bank, anchors)
    compact = compact_layout_from_anchor_v1(native)
    geometry = CompactSpectralGeometryV1.from_operators(operators, include_diagonal=True)
    return bank, native, compact, operators, geometry


def inputs(layout, dtype, *, requires_grad=True):
    generator = torch.Generator().manual_seed(417)
    a = len(layout.anchors)
    query = (.6 * torch.randn(a, 3, 2, dtype=torch.float64, generator=generator)).tanh().to(dtype)
    reference = (.7 * torch.randn(a, 2, 2, dtype=torch.float64, generator=generator)).tanh().to(dtype)
    raw_rate = torch.tensor([[.1, -.2], [.3, -.1]], dtype=dtype)
    raw_damping = torch.tensor([[.2, -.1], [-.2, .1]], dtype=dtype)
    logits = torch.tensor([[.4, -.3], [-.2, .5]], dtype=dtype)
    return tuple(x.requires_grad_(requires_grad) for x in (query, reference, raw_rate, raw_damping, logits))


def parameters(values):
    query, reference, raw_rate, raw_damping, logits = values
    rates = raw_rate.new_tensor([[.7, 1.3], [1.1, .6]]) * raw_rate.exp()
    damping = raw_damping.new_tensor([[.5, .8], [.3, .6]]) * raw_damping.exp()
    return query, reference, rates, damping, logits.softmax(0)


def source_bundle(layout, operators, query, reference, rates):
    cqr = operators.cqr[layout.anchor_graph]
    sources = query[:, None] - (cqr @ reference[:, None]) * rates[None, :, None, :]
    mass = torch.ones_like(sources) - cqr.sum(-1)[..., None] * rates[None, :, None, :]
    return torch.stack((sources, mass), dim=-1)


def oracle_matrix_exp(matrix):
    """Independent dense exponential, numerically qualified against closed/scipy.

    A constant unrelated 1x1 block forces a stable norm regime in the local
    Torch 2.6 CPU matrix_exp. Its top-left block is mathematically exp(matrix).
    This avoids small-norm approximation errors diagnosed independently in
    oracle_diagnostic.json, without loosening the acceptance tolerances.
    """
    q = matrix.shape[-1]
    padded = torch.block_diag(matrix, matrix.new_ones((1, 1)))
    return torch.matrix_exp(padded)[:q, :q]


def dense_path(intervals, operators, rate, damping, x, y, scale, coupling):
    """Independent differentiable physical path, dense matrix_exp, no scan/eigh."""
    q = operators.cqq.shape[-1]
    identity = torch.eye(q, dtype=rate.dtype)
    product = identity
    lower, upper = min(x, y), max(x, y)
    for graph in torch.argsort(intervals[:, 0]).tolist():
        left, right = intervals[graph].tolist()
        duration = max(0., min(upper, right) - max(lower, left))
        if duration == 0:
            continue
        if coupling == "full":
            matrix = operators.cqq[graph, scale]
        elif coupling == "scalar":
            matrix = identity
        else:
            matrix = torch.diag(operators.cqq[graph, scale].diagonal())
        transition = oracle_matrix_exp(-duration * (rate * matrix + damping * identity))
        product = transition @ product
    return product if x >= y else product.T


def dense_action(bank, layout, operators, sources, rates, damping, coupling="full"):
    """Direct finite anchor sum and optional dense solve, independent of recurrence."""
    a, s, q, h, v = sources.shape
    if coupling == "gain_matched_diagonal":
        transformed = []
        for anchor, graph in enumerate(layout.anchor_graph.tolist()):
            branches = []
            for scale in range(s):
                channels = []
                for channel in range(h):
                    generator = (rates[scale, channel] * operators.cqq[graph, scale]
                                 + damping[scale, channel] * torch.eye(q, dtype=sources.dtype))
                    equilibrium = torch.linalg.solve(generator, sources[anchor, scale, :, channel])
                    channels.append(generator.diagonal()[:, None] * equilibrium)
                branches.append(torch.stack(channels, dim=1))
            transformed.append(torch.stack(branches))
        sources = torch.stack(transformed)
    output = []
    for x in layout.evaluations.tolist():
        branches = []
        for scale in range(s):
            channels = []
            for channel in range(h):
                terms = [layout.weights[anchor].to(sources.dtype) * dense_path(
                    bank.intervals, operators, rates[scale, channel], damping[scale, channel],
                    x, y, scale, coupling) @ sources[anchor, scale, :, channel]
                    for anchor, y in enumerate(layout.anchors.tolist())]
                channels.append(torch.stack(terms).sum(0))
            branches.append(torch.stack(channels, dim=1))
        output.append(torch.stack(branches))
    return torch.stack(output)


def mixed_messages(raw, mixture):
    mixed = (raw * mixture[None, :, None, :, None]).sum(1)
    numerator, mass = mixed.unbind(-1)
    return numerator, mass, numerator / mass


def objective(normalized, mass):
    cotangent = torch.linspace(-.7, .9, normalized.numel(), dtype=normalized.dtype).reshape_as(normalized)
    return (normalized * cotangent).sum() + .013 * mass.log().sum()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("coupling", ["full", "scalar", "diagonal", "gain_matched_diagonal"])
def test_raw_mass_messages_and_vjp_against_dense_and_original(dtype, coupling):
    bank, native, layout, operators, geometry = fixture(dtype, reorder=[2, 0, 1])
    values = inputs(layout, dtype)
    query, reference, rates, damping, mixture = parameters(values)
    bundle = source_bundle(layout, operators, query, reference, rates)
    actual = anchor_spectral_compact_action_v1(layout, geometry, bundle, rates, damping,
                                               coupling=coupling, readout_chunk_size=2)
    original = anchor_kernel_action(native, operators, bundle, rates, damping, coupling=coupling)
    oracle = dense_action(bank, layout, operators, bundle, rates, damping, coupling)
    close(actual, original, dtype)
    close(actual, oracle, dtype)
    actual_parts = mixed_messages(actual, mixture)
    original_parts = mixed_messages(original, mixture)
    oracle_parts = mixed_messages(oracle, mixture)
    for x, y, z in zip(actual_parts, original_parts, oracle_parts):
        close(x, y, dtype)
        close(x, z, dtype)
    assert (actual_parts[1] > 0).all()
    assert actual_parts[2].abs().max() <= 1 + (1e-10 if dtype == torch.float64 else 2e-5)
    wrapped = compact_anchor_messages_v1(layout, geometry, query, reference, rates, damping,
                                         mixture, coupling=coupling, readout_chunk_size=2)
    close(wrapped.numerator, actual_parts[0], dtype)
    close(wrapped.mass, actual_parts[1], dtype)
    close(wrapped.anchors, actual_parts[2][layout.anchor_output], dtype)
    close(wrapped.targets, actual_parts[2][layout.target_output], dtype)
    gradients = [torch.autograd.grad(objective(parts[2], parts[1]), values, retain_graph=True)
                 for parts in (actual_parts, original_parts, oracle_parts)]
    for x, y, z in zip(*gradients):
        assert torch.isfinite(x).all()
        close(x, y, dtype, gradient=True)
        close(x, z, dtype, gradient=True)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_boundary_anchor_collision_duplicate_outputs_and_self_term(dtype):
    _, native, layout, operators, geometry = fixture(dtype)
    assert layout.anchors[1] == .5
    assert layout.anchor_graph[1] == 2  # Right-half-open hold, not the left graph.
    assert layout.target_output[0] == layout.target_output[5] == layout.anchor_output[1]
    assert layout.num_segments == 5
    rates = torch.tensor([[.7], [1.4]], dtype=dtype)
    damping = torch.tensor([[.3], [.6]], dtype=dtype)
    sources = torch.zeros(3, 2, 3, 1, 2, dtype=dtype)
    sources[1] = torch.tensor([1., 2., 3.], dtype=dtype)[None, :, None, None]
    actual = anchor_spectral_compact_action_v1(layout, geometry, sources, rates, damping)
    close(actual[layout.anchor_output[1]], layout.weights[1].to(dtype) * sources[1], dtype)
    close(actual, anchor_kernel_action(native, operators, sources, rates, damping), dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_holds_with_no_anchors_and_many_anchors_in_middle(dtype):
    bank, native, layout, operators, geometry = fixture(
        dtype, intervals=[[0., .1], [.1, .9], [.9, 1.]])
    assert layout.graph_at_anchor == (1, 1, 1)
    values = inputs(layout, dtype)
    query, reference, rates, damping, _ = parameters(values)
    sources = source_bundle(layout, operators, query, reference, rates)
    actual = anchor_spectral_compact_action_v1(layout, geometry, sources, rates, damping)
    close(actual, anchor_kernel_action(native, operators, sources, rates, damping), dtype)
    close(actual, dense_action(bank, layout, operators, sources, rates, damping), dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_zero_operator_coupling_is_querywise_damping_with_zero_rate_gradient(dtype):
    _, native, layout, operators, _ = fixture(dtype)
    generator = torch.Generator().manual_seed(901)
    rotations = torch.linalg.qr(torch.randn(3, 2, 3, 3, dtype=torch.float64, generator=generator)).Q.to(dtype)
    operators = AnchorOperators(torch.zeros_like(operators.cqq), torch.zeros_like(operators.cqr),
                                torch.zeros_like(operators.eigenvalues), rotations)
    geometry = CompactSpectralGeometryV1.from_operators(operators)
    values = inputs(layout, dtype)
    query, reference, rates, damping, mixture = parameters(values)
    sources = source_bundle(layout, operators, query, reference, rates)
    actual = anchor_spectral_compact_action_v1(layout, geometry, sources, rates, damping)
    direct = torch.stack([sum(layout.weights[a].to(dtype) * torch.exp(-abs(x - y) * damping)[:, None, :, None]
                              * sources[a] for a, y in enumerate(layout.anchors.tolist()))
                          for x in layout.evaluations.tolist()])
    close(actual, direct, dtype)
    close(actual, anchor_kernel_action(native, operators, sources, rates, damping), dtype)
    gradients = torch.autograd.grad(actual.square().sum(), (values[2], values[3]))
    close(gradients[0], torch.zeros_like(gradients[0]), dtype, gradient=True)
    assert gradients[1].abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sample_permutation_preserves_physical_rows_and_rate_gradients(dtype):
    _, _, layout, operators, geometry = fixture(dtype)
    values = inputs(layout, dtype)
    query, reference, rates, damping, mixture = parameters(values)
    base = compact_anchor_messages_v1(layout, geometry, query, reference, rates, damping, mixture)
    qp = torch.tensor([2, 0, 1])
    rp = torch.tensor([1, 0])
    permuted_ops = AnchorOperators(operators.cqq[:, :, qp][:, :, :, qp],
        operators.cqr[:, :, qp][:, :, :, rp], operators.eigenvalues,
        operators.eigenvectors[:, :, qp])
    permuted = compact_anchor_messages_v1(layout, CompactSpectralGeometryV1.from_operators(permuted_ops),
        query[:, qp], reference[:, rp], rates, damping, mixture)
    close(permuted.targets, base.targets[:, qp], dtype)
    close(permuted.mass, base.mass[:, qp], dtype)
    g1 = torch.autograd.grad(base.targets.square().sum(), values[2:4], retain_graph=True)
    g2 = torch.autograd.grad(permuted.targets.square().sum(), values[2:4])
    for x, y in zip(g1, g2):
        close(x, y, dtype, gradient=True)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_noncommuting_physical_order_and_serialization(dtype):
    bank, _, layout, operators, geometry = fixture(dtype)
    assert torch.linalg.matrix_norm(operators.cqq[0, 0] @ operators.cqq[1, 0]
                                   - operators.cqq[1, 0] @ operators.cqq[0, 0]) > .1
    values = inputs(layout, dtype)
    query, reference, rates, damping, _ = parameters(values)
    sources = source_bundle(layout, operators, query, reference, rates)
    base = anchor_spectral_compact_action_v1(layout, geometry, sources, rates, damping)
    _, _, serial_layout, serial_ops, serial_geometry = fixture(dtype, reorder=[2, 0, 1])
    serialized = anchor_spectral_compact_action_v1(serial_layout, serial_geometry, sources, rates, damping)
    close(base, serialized, dtype)
    reversed_ops = AnchorOperators(*(x.flip(0) for x in (operators.cqq, operators.cqr,
                                                       operators.eigenvalues, operators.eigenvectors)))
    reversed_result = anchor_spectral_compact_action_v1(layout,
        CompactSpectralGeometryV1.from_operators(reversed_ops), sources, rates, damping)
    assert (base - reversed_result).abs().max() > .005


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_identical_hold_subdivision_and_dense_readouts_are_not_impulses(dtype, monkeypatch):
    _, native, layout, operators, geometry = fixture(dtype, intervals=[[0., 1.]], positions=[.0, .13, .5, .999])
    values = inputs(layout, dtype)
    query, reference, rates, damping, _ = parameters(values)
    sources = source_bundle(layout, operators, query, reference, rates)
    base = anchor_spectral_compact_action_v1(layout, geometry, sources, rates, damping)
    split = prepare_compact_anchor_layout_v1(torch.tensor([[0., .27], [.27, 1.]], dtype=torch.float64),
        torch.tensor([.0, .13, .5, .999], dtype=torch.float64), num_anchors=3)
    split_geometry = CompactSpectralGeometryV1(*(x.expand(2, *x.shape[1:]) for x in
        (geometry.eigenvalues, geometry.eigenvectors, geometry.cqr)))
    split_result = anchor_spectral_compact_action_v1(split, split_geometry, sources, rates, damping)
    close(base, split_result, dtype)
    dense_positions = torch.cat((torch.linspace(0., .999, 1025, dtype=torch.float64),
                                 torch.tensor([.0, .13, .5, .999], dtype=torch.float64)))
    _, dense_native, dense_layout, dense_ops, dense_geometry = fixture(
        dtype, intervals=[[0., 1.]], positions=dense_positions.tolist())
    assert dense_layout.num_segments == layout.num_segments == 4
    assert dense_layout.source_at_event == layout.source_at_event
    seen = []
    original_reconstruct = compact_module._readout_reconstruct_chunk
    def tracked(spectral, vectors):
        seen.append((spectral.shape, vectors.shape))
        return original_reconstruct(spectral, vectors)
    monkeypatch.setattr(compact_module, "_readout_reconstruct_chunk", tracked)
    dense_result = anchor_spectral_compact_action_v1(dense_layout, dense_geometry, sources, rates, damping,
                                                    readout_chunk_size=17)
    assert sum(shape[0] for shape, _ in seen) == dense_layout.num_evaluations
    assert all(shape[0] <= 17 and basis == (2, 3, 3) for shape, basis in seen)
    common = torch.searchsorted(dense_layout.evaluations, layout.evaluations)
    close(dense_result[common], base, dtype)
    original_dense = anchor_kernel_action(dense_native, dense_ops, sources, rates, damping)
    close(dense_result, original_dense, dtype)
    g1 = torch.autograd.grad(base.square().sum(), values[2:4], retain_graph=True)
    g2 = torch.autograd.grad(dense_result[common].square().sum(), values[2:4], retain_graph=True)
    g3 = torch.autograd.grad(original_dense[common].square().sum(), values[2:4])
    for x, y, z in zip(g1, g2, g3):
        close(x, y, dtype, gradient=True)
        close(x, z, dtype, gradient=True)


class ToyDepth(nn.Module):
    def __init__(self, use_compact):
        super().__init__()
        self.use_compact = use_compact
        self.layers = nn.ModuleList([AnchorKernelLayer(4, 2, 2, [.7, 1.3], [.5, .8]) for _ in range(2)])
        self.head = nn.Linear(4, 2)

    def forward(self, bank, native, layout, operators, geometry, context, references):
        anchors = context[native.anchor_graph]
        targets = context[bank.target_graph]
        fixed_reference = references[native.anchor_graph]
        for layer in self.layers:
            if self.use_compact:
                message = compact_anchor_messages_v1(layout, geometry,
                    layer.query_projection(anchors).tanh(), layer.reference_projection(fixed_reference).tanh(),
                    layer.rates, layer.damping, layer.mixture, readout_chunk_size=2)
            else:
                message = layer.messages(SimpleNamespace(layout=native, operators={"fixture": operators}),
                                          anchors, fixed_reference, "fixture", "full")
            anchors = layer.update(anchors, message.anchors)
            targets = layer.update(targets, message.targets)
        return self.head(targets)


def optimizer(model):
    rate = [x for layer in model.layers for x in (layer.raw_coupling, layer.raw_damping, layer.mixture_logits)]
    ids = {id(x) for x in rate}
    return torch.optim.AdamW([{"params": [x for x in model.parameters() if id(x) not in ids], "weight_decay": .01},
                             {"params": rate, "weight_decay": 0.}], lr=.003)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_two_layer_gate_head_clipped_adamw_step_parity(dtype):
    bank, native, layout, operators, geometry = fixture(dtype)
    torch.manual_seed(730)
    original, compact = ToyDepth(False).to(dtype), ToyDepth(True).to(dtype)
    compact.load_state_dict(original.state_dict())
    context = torch.randn(3, 3, 4, dtype=dtype)
    references = torch.randn(3, 2, 4, dtype=dtype)
    predictions = [m(bank, native, layout, operators, geometry, context, references) for m in (original, compact)]
    close(predictions[0], predictions[1], dtype)
    weights = torch.arange(1, predictions[0].numel() + 1, dtype=dtype).reshape_as(predictions[0])
    weights = weights / weights.sum()
    opt = [optimizer(m) for m in (original, compact)]
    for prediction in predictions:
        (-(prediction.log_softmax(-1) * weights).sum()).backward()
    for x, y in zip(original.parameters(), compact.parameters()):
        assert x.grad is not None and y.grad is not None
        close(x.grad, y.grad, dtype, gradient=True)
    norms = [torch.nn.utils.clip_grad_norm_(m.parameters(), .1) for m in (original, compact)]
    assert norms[0] > .1  # Exercise clipping rather than its inactive branch.
    close(norms[0], norms[1], dtype, gradient=True)
    for o in opt:
        o.step()
    for x, y in zip(original.parameters(), compact.parameters()):
        close(x, y, dtype, gradient=True)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            close(opt[0].state[x][name], opt[1].state[y][name], dtype, gradient=True)


def test_five_point_directional_rate_damping_and_mixture_derivatives():
    bank, _, layout, operators, geometry = fixture()
    values = inputs(layout, torch.float64)
    def evaluate(current):
        query, reference, rates, damping, mixture = parameters(current)
        messages = compact_anchor_messages_v1(layout, geometry, query, reference, rates, damping, mixture)
        return objective(messages.targets, messages.mass)
    value = evaluate(values)
    gradients = torch.autograd.grad(value, values)
    generator = torch.Generator().manual_seed(72)
    direction = tuple(torch.randn(x.shape, dtype=x.dtype, generator=generator) for x in values)
    expected = sum((g * d).sum() for g, d in zip(gradients, direction))
    step = 2e-4
    points = {k: evaluate(tuple(x.detach() + k * step * d for x, d in zip(values, direction)))
              for k in (-2, -1, 1, 2)}
    difference = (points[-2] - 8 * points[-1] + 8 * points[1] - points[2]) / (12 * step)
    torch.testing.assert_close(expected, difference, rtol=2e-8, atol=1e-9)


def test_minimal_geometry_view_and_layout_adapter_preserve_fields():
    _, native, layout, operators, geometry = fixture()
    assert geometry.eigenvectors.data_ptr() == operators.eigenvectors.data_ptr()
    assert geometry.eigenvalues.data_ptr() == operators.eigenvalues.data_ptr()
    full_only = CompactSpectralGeometryV1.from_operators(operators)
    assert full_only.diagonal is None
    assert not hasattr(full_only, "cqq")
    transferred = layout.to("cpu")
    assert transferred.coordinates.dtype == transferred.segment_duration.dtype == torch.float64
    assert transferred.evaluation_at_segment == layout.evaluation_at_segment
    close(transferred.evaluations, layout.evaluations, torch.float64)
    with pytest.raises(ValueError, match="fixed source/output measure"):
        compact_layout_from_anchor_v1(replace(native, anchors=native.anchors + 1.))
    bad_sources = tuple(-1 for _ in native.source_at_event)
    with pytest.raises(ValueError, match="source events"):
        compact_layout_from_anchor_v1(replace(native, source_at_event=bad_sources))


@pytest.mark.parametrize("intervals,positions,count", [
    ([[0., .4], [.5, 1.]], [.2], 3),
    ([[0., .6], [.4, 1.]], [.2], 3),
    ([[0., 0.]], [0.], 3),
    ([[0., 1.]], [1.], 3),
    ([[0., 1.]], [-.1], 3),
    ([[0., 1.]], [float("nan")], 3),
    ([[0., 1.]], [.2], 0),
])
def test_invalid_coordinate_contract_fails(intervals, positions, count):
    with pytest.raises(ValueError):
        prepare_compact_anchor_layout_v1(torch.tensor(intervals, dtype=torch.float64),
            torch.tensor(positions, dtype=torch.float64), num_anchors=count)


def test_shifted_clipped_tile_anchor_measure_and_reject_wrong_graph_mapping():
    layout = prepare_compact_anchor_layout_v1(torch.tensor([[4., 4.4], [4.4, 5.]], dtype=torch.float64),
        torch.tensor([4., 4.4, 4.999], dtype=torch.float64), num_anchors=4)
    assert torch.equal(layout.anchors, torch.tensor([4.125, 4.375, 4.625, 4.875], dtype=torch.float64))
    assert torch.equal(layout.weights, torch.full((4,), .25, dtype=torch.float64))
    assert ((layout.anchors > 4) & (layout.anchors < 5)).all()
    with pytest.raises(ValueError, match="Target graph"):
        prepare_compact_anchor_layout_v1(torch.tensor([[0., .5], [.5, 1.]], dtype=torch.float64),
            torch.tensor([.5], dtype=torch.float64), num_anchors=1, target_graph=torch.tensor([0]))


def test_invalid_numeric_contract_fails_before_normalization():
    _, _, layout, _, geometry = fixture()
    values = inputs(layout, torch.float64)
    query, reference, rates, damping, mixture = parameters(values)
    for bad_rate in (torch.zeros_like(rates), -rates, rates * float("inf")):
        with pytest.raises(ValueError, match="positive rates/damping"):
            compact_anchor_messages_v1(layout, geometry, query, reference, bad_rate, damping, mixture)
    with pytest.raises(ValueError, match="positive rates/damping"):
        compact_anchor_messages_v1(layout, geometry, query * float("nan"), reference, rates, damping, mixture)
    with pytest.raises(ValueError, match="mass is nonpositive"):
        compact_anchor_messages_v1(layout, geometry, query, reference, rates, damping, torch.zeros_like(mixture))
    with pytest.raises(ValueError, match="mass is nonpositive"):
        compact_anchor_messages_v1(layout, geometry, query, reference, rates,
                                  torch.full_like(damping, 1e9), mixture)
    with pytest.raises(ValueError, match="geometry must be fixed"):
        bad = replace(geometry, eigenvalues=geometry.eigenvalues.clone().requires_grad_())
        compact_anchor_messages_v1(layout, bad, query, reference, rates, damping, mixture)
    with pytest.raises(ValueError, match="readout_chunk_size"):
        compact_anchor_messages_v1(layout, geometry, query, reference, rates, damping, mixture,
                                  readout_chunk_size=0)


def test_independent_dense_exponential_oracle_accuracy_and_derivative():
    from scipy.linalg import expm
    diagonal = torch.tensor([2.544750485712468, 2.003997623343231, 2.312674918002],
                            dtype=torch.float64, requires_grad=True)
    duration = .013333333333333308
    matrix = -duration * torch.diag(diagonal)
    actual = oracle_matrix_exp(matrix)
    expected = torch.diag((-duration * diagonal).exp())
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-15)
    cotangent = torch.tensor([[.3, -.1, .5], [.1, .8, .2], [-.4, .7, -.2]], dtype=torch.float64)
    actual_grad = torch.autograd.grad((actual * cotangent).sum(), diagonal, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * cotangent).sum(), diagonal)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-14)
    _, _, _, operators, _ = fixture()
    for cqq in operators.cqq.flatten(0, 1):
        for dt in (.001, .013333333333333308, .2, .7):
            matrix = -dt * (1.4 * cqq + .3 * torch.eye(3, dtype=torch.float64))
            scipy = torch.as_tensor(expm(matrix.numpy()))
            torch.testing.assert_close(oracle_matrix_exp(matrix), scipy, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("field", ["cqr", "diagonal"])
def test_source_map_and_control_diagonal_are_fixed_geometry(field):
    _, _, layout, _, geometry = fixture()
    query, reference, rates, damping, mixture = parameters(inputs(layout, torch.float64))
    bad = replace(geometry, **{field: getattr(geometry, field).clone().requires_grad_()})
    with pytest.raises(ValueError, match="geometry must be fixed"):
        compact_anchor_messages_v1(layout, bad, query, reference, rates, damping, mixture,
                                  coupling="diagonal" if field == "diagonal" else "full")

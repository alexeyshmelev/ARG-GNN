"""Independent handmade CPU tests. No project observed arrays, no CUDA."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from arggnn.anchor_kernel import AnchorOperators, _layout, anchor_kernel_action
from arggnn.anchor_spectral_compact_v1 import (
    CompactSpectralGeometryV1, prepare_compact_anchor_layout_v1,
    compact_layout_from_anchor_v1, anchor_spectral_compact_action_v1,
    compact_anchor_messages_v1)

DIRECTORY = Path(__file__).resolve().parent
METRICS = []


@pytest.fixture(autouse=True, scope="session")
def explicit_cpu():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    yield
    (DIRECTORY / "independent_metrics.json").write_text(json.dumps(METRICS, indent=2) + "\n")


def tolerance(dtype, gradient=False):
    return ((1e-8, 1e-10) if gradient else (1e-10, 1e-12)) if dtype == torch.float64 else (
        (1e-3, 1e-5) if gradient else (2e-4, 2e-5))


def check(actual, expected, dtype, name, gradient=False):
    rtol, atol = tolerance(dtype, gradient)
    error = (actual.detach() - expected.detach()).abs()
    METRICS.append(dict(name=name, dtype=str(dtype), gradient=gradient,
                        max_absolute=float(error.max()),
                        max_acceptance_units=float((error / (atol + rtol * expected.detach().abs())).max())))
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def fixture(dtype, offset=0.):
    intervals = torch.tensor([[-1., -.5], [-.5, .1], [.1, .5], [.5, 1.5],
                              [1.5, 2.5], [2.5, 3.]], dtype=torch.float64) + offset
    positions = torch.tensor([2.5, -1., .5, -.5, 2.875, .1, 1.5, -.5, -.875],
                             dtype=torch.float64) + offset
    bases = torch.tensor([[[1.4, -.2], [-.2, 2.1]],
                          [[2.3, -.7], [-.7, 1.1]],
                          [[1.6, -.3], [-.3, 1.8]],
                          [[3.1, -.8], [-.8, 1.7]],
                          [[1.3, -.15], [-.15, 2.5]],
                          [[2.7, -.4], [-.4, 1.4]]], dtype=torch.float64)
    matrices = torch.stack([bases, .65 * bases + .35 * torch.eye(2, dtype=torch.float64),
                             1.6 * bases + .12 * torch.eye(2, dtype=torch.float64)], dim=1)
    fractions = torch.tensor([[.2, .5, .3], [.4, .1, .5]], dtype=torch.float64)
    cqr = -matrices.sum(-1)[..., None] * fractions
    eigenvalues, eigenvectors = torch.linalg.eigh(matrices)
    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    intervals = intervals[permutation]
    operators = AnchorOperators(*(x[permutation].to(dtype) for x in
                                  (matrices, cqr, eigenvalues, eigenvectors)))
    order = torch.argsort(intervals[:, 0])
    graph = order[torch.searchsorted(intervals[order, 0].contiguous(), positions, right=True) - 1]
    bank = SimpleNamespace(intervals=intervals, positions=positions, target_graph=graph,
                           window_bounds=torch.tensor([-1. + offset, 3. + offset], dtype=torch.float64))
    native = _layout(bank, 4)
    layout = compact_layout_from_anchor_v1(native)
    geometry = CompactSpectralGeometryV1.from_operators(operators, include_diagonal=True)
    return bank, native, layout, operators, geometry


def raw_parameters(dtype, shape):
    src = torch.sin(torch.arange(torch.tensor(shape).prod(), dtype=torch.float64)).reshape(shape).to(dtype)
    lr = torch.tensor([[-.3, .2, .0], [.5, -.2, .1], [.25, -.4, .3]], dtype=dtype)
    ld = torch.tensor([[.1, -.1, .4], [-.3, .25, .2], [.4, .3, -.2]], dtype=dtype)
    return tuple(x.requires_grad_() for x in (src, lr, ld))


def closed_exponential(matrix, rate, damping, duration, control):
    """Closed exp for symmetric 2x2 matrices, without matrix_exp or eigenpairs."""
    eye = torch.eye(2, dtype=rate.dtype)
    if control == "scalar":
        return (-duration * (rate + damping)).exp() * eye
    if control != "full":
        return torch.diag((-duration * (rate * matrix.diagonal() + damping)).exp())
    k = rate * matrix + damping * eye
    mean = (k[0, 0] + k[1, 1]) / 2
    centered = k - mean * eye
    radius = (((k[0, 0] - k[1, 1]) / 2).square() + k[0, 1].square()).sqrt()
    return (-duration * mean).exp() * (
        (duration * radius).cosh() * eye - (duration * radius).sinh() / radius * centered)


def closed_inverse(matrix, rate, damping, source):
    k = rate * matrix + damping * torch.eye(2, dtype=rate.dtype)
    inverse = torch.stack((torch.stack((k[1, 1], -k[0, 1])),
                           torch.stack((-k[1, 0], k[0, 0])))) / torch.linalg.det(k)
    return k.diagonal()[:, None] * (inverse @ source)


def closed_action(bank, layout, operators, source, rates, damping, control):
    """Enumerate anchors and traverse the physical path in travel direction."""
    s, _, h, _ = source.shape[1:]
    order = torch.argsort(bank.intervals[:, 0]).tolist()
    output = []
    for x in layout.evaluations.tolist():
        branches = []
        for scale in range(s):
            channels = []
            for channel in range(h):
                terms = []
                for anchor, y in enumerate(layout.anchors.tolist()):
                    z = source[anchor, scale, :, channel]
                    if control == "gain_matched_diagonal":
                        g = int(layout.anchor_graph[anchor])
                        z = closed_inverse(operators.cqq[g, scale], rates[scale, channel],
                                           damping[scale, channel], z)
                    for g in order if x >= y else order[::-1]:
                        left, right = bank.intervals[g].tolist()
                        duration = max(0., min(max(x, y), right) - max(min(x, y), left))
                        if duration:
                            z = closed_exponential(operators.cqq[g, scale], rates[scale, channel],
                                                   damping[scale, channel], duration, control) @ z
                    terms.append(z * layout.weights[anchor].to(source.dtype))
                channels.append(torch.stack(terms).sum(0))
            branches.append(torch.stack(channels, dim=1))
        output.append(torch.stack(branches))
    return torch.stack(output)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("control", ["full", "scalar", "diagonal", "gain_matched_diagonal"])
def test_closed_path_all_anchor_boundary_collisions_and_general_bundle(dtype, control):
    bank, native, layout, ops, geometry = fixture(dtype)
    assert set(layout.anchors.tolist()).issubset(set(bank.intervals.flatten().tolist()))
    assert layout.num_segments == 6  # readouts do not split holds
    source, lr, ld = raw_parameters(dtype, (4, 3, 2, 3, 3))
    rate, damping = lr.exp(), .4 * ld.exp()
    compact = anchor_spectral_compact_action_v1(layout, geometry, source, rate, damping,
                                               coupling=control, readout_chunk_size=1)
    reference = closed_action(bank, layout, ops, source, rate, damping, control)
    native_output = anchor_kernel_action(native, ops, source, rate, damping, coupling=control)
    check(compact, reference, dtype, control + "_closed_raw")
    check(compact, native_output, dtype, control + "_native_raw")
    cotangent = torch.cos(torch.arange(compact.numel(), dtype=dtype)).reshape_as(compact)
    gradients = [torch.autograd.grad((value * cotangent).sum() + .017 * value.square().sum(),
                                    (source, lr, ld), retain_graph=True)
                 for value in (compact, reference, native_output)]
    for family, (a, b, c) in zip(("source", "raw_rate", "raw_damping"), zip(*gradients)):
        check(a, b, dtype, control + "_closed_" + family, gradient=True)
        check(a, c, dtype, control + "_native_" + family, gradient=True)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("control", ["full", "scalar", "diagonal", "gain_matched_diagonal"])
def test_matching_mass_and_normalized_signal_gradients(dtype, control):
    bank, _, layout, ops, geometry = fixture(dtype)
    _, lr, ld = raw_parameters(dtype, (4, 3, 2, 3, 3))
    query = (.7 * torch.cos(torch.arange(24, dtype=dtype))).reshape(4, 2, 3).requires_grad_()
    reference = (.6 * torch.sin(torch.arange(36, dtype=dtype))).reshape(4, 3, 3).requires_grad_()
    logits = torch.tensor([[.3, -.3, .2], [-.1, .2, .3], [.5, .1, -.4]], dtype=dtype, requires_grad=True)
    rates, damping, mixture = lr.exp(), .4 * ld.exp(), logits.softmax(0)
    action = compact_anchor_messages_v1(layout, geometry, query, reference, rates, damping,
                                        mixture, coupling=control, readout_chunk_size=2)
    cqr = ops.cqr[layout.anchor_graph]
    signal = query[:, None] - rates[None, :, None, :] * (cqr @ reference[:, None])
    mass_source = torch.ones_like(signal) - rates[None, :, None, :] * cqr.sum(-1)[..., None]
    raw = closed_action(bank, layout, ops, torch.stack((signal, mass_source), dim=-1),
                        rates, damping, control)
    numerator, mass = (raw * mixture[None, :, None, :, None]).sum(1).unbind(-1)
    normalized = numerator / mass
    check(action.numerator, numerator, dtype, control + "_numerator")
    check(action.mass, mass, dtype, control + "_mass")
    check(action.targets, normalized[layout.target_output], dtype, control + "_targets")
    assert bool((action.mass > 0).all())
    assert float(action.targets.abs().max()) < 1.
    def loss(target, mass):
        c = torch.linspace(-.4, .8, target.numel(), dtype=dtype).reshape_as(target)
        return (c * target).sum() + .041 * mass.log().sum()
    values = (query, reference, lr, ld, logits)
    a = torch.autograd.grad(loss(action.targets, action.mass), values, retain_graph=True)
    b = torch.autograd.grad(loss(normalized[layout.target_output], mass), values)
    for family, x, y in zip(("query", "reference", "raw_rate", "raw_damping", "logits"), a, b):
        check(x, y, dtype, control + "_normalized_" + family, gradient=True)
    ones = compact_anchor_messages_v1(layout, geometry, torch.ones_like(query),
                                      torch.ones_like(reference), rates, damping, mixture, coupling=control)
    check(ones.targets, torch.ones_like(ones.targets), dtype, control + "_matching_ones")
    check(action.targets[3], action.targets[7], dtype, control + "_duplicate")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_large_shift_preserves_float64_physical_offsets_and_gradients(dtype):
    _, _, layout, _, geometry = fixture(dtype)
    # .1 cannot be represented identically after the large shift, so use dyadic endpoints/readouts.
    intervals = torch.tensor([[0., .25], [.25, .5], [.5, .75], [.75, 1.]], dtype=torch.float64)
    positions = torch.tensor([.125, .25, .375, .5, .625, .75, .875, 0.], dtype=torch.float64)
    layout = prepare_compact_anchor_layout_v1(intervals, positions, num_anchors=4)
    shifted = prepare_compact_anchor_layout_v1(intervals + 2. ** 25, positions + 2. ** 25, num_anchors=4)
    geometry = replace(geometry, eigenvalues=geometry.eigenvalues[:4], eigenvectors=geometry.eigenvectors[:4],
                       cqr=geometry.cqr[:4], diagonal=geometry.diagonal[:4])
    assert layout.coordinates.dtype == shifted.coordinates.dtype == torch.float64
    source, lr, ld = raw_parameters(dtype, (4, 3, 2, 3, 3))
    rate, damping = lr.exp(), ld.exp()
    base = anchor_spectral_compact_action_v1(layout, geometry, source, rate, damping)
    moved = anchor_spectral_compact_action_v1(shifted, geometry, source, rate, damping)
    check(base, moved, dtype, "large_shift")
    a = torch.autograd.grad(base.square().sum(), (source, lr, ld), retain_graph=True)
    b = torch.autograd.grad(moved.square().sum(), (source, lr, ld))
    for x, y in zip(a, b):
        check(x, y, dtype, "large_shift_gradient", gradient=True)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_degenerate_equal_geometry_subdivision_with_changing_orthogonal_bases(dtype):
    generators = torch.Generator().manual_seed(951)
    rotations = torch.linalg.qr(torch.randn(4, 1, 4, 4, dtype=torch.float64, generator=generators)).Q.to(dtype)
    c = torch.full((4, 1, 4), 1.7, dtype=dtype)
    geometry = CompactSpectralGeometryV1(c, rotations, torch.empty(4, 1, 4, 0, dtype=dtype))
    layout = prepare_compact_anchor_layout_v1(
        torch.tensor([[0., .125], [.125, .5], [.5, .875], [.875, 1.]], dtype=torch.float64),
        torch.tensor([0., .125, .5, .875, .999], dtype=torch.float64), num_anchors=4)
    source = torch.randn(4, 1, 4, 2, 3, dtype=dtype, generator=generators).requires_grad_()
    lr = torch.tensor([[.2, -.3]], dtype=dtype, requires_grad=True)
    ld = torch.tensor([[-.1, .3]], dtype=dtype, requires_grad=True)
    rate, damping = lr.exp(), ld.exp()
    actual = anchor_spectral_compact_action_v1(layout, geometry, source, rate, damping, readout_chunk_size=2)
    expected = torch.stack([sum(layout.weights[a].to(dtype) *
        (-abs(x - y) * (1.7 * rate + damping)).exp()[:, None, :, None] * source[a]
        for a, y in enumerate(layout.anchors.tolist())) for x in layout.evaluations.tolist()])
    check(actual, expected, dtype, "degenerate_changing_bases")
    a = torch.autograd.grad(actual.square().sum(), (source, lr, ld), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (source, lr, ld))
    for x, y in zip(a, b):
        check(x, y, dtype, "degenerate_gradient", gradient=True)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_eigenpair_gauge_and_query_reference_permutations(dtype):
    _, _, layout, _, geometry = fixture(dtype)
    source, lr, ld = raw_parameters(dtype, (4, 3, 2, 3, 3))
    rate, damping = lr.exp(), ld.exp()
    original = anchor_spectral_compact_action_v1(layout, geometry, source, rate, damping)
    # Each hold/scale chooses independent column order and signs.
    values = geometry.eigenvalues.clone()
    vectors = geometry.eigenvectors.clone()
    for g in range(6):
        for s in range(3):
            order = [1, 0] if (g + s) % 2 else [0, 1]
            values[g, s] = values[g, s, order].clone()
            vectors[g, s] = vectors[g, s, :, order].clone() * torch.tensor([(-1.) ** g, (-1.) ** s], dtype=dtype)
    gauge = replace(geometry, eigenvalues=values, eigenvectors=vectors)
    altered = anchor_spectral_compact_action_v1(layout, gauge, source, rate, damping)
    check(altered, original, dtype, "eigenpair_gauge")
    permuted = replace(gauge, eigenvectors=gauge.eigenvectors[:, :, [1, 0]],
                       cqr=gauge.cqr[:, :, [1, 0]][:, :, :, [2, 0, 1]],
                       diagonal=gauge.diagonal[:, :, [1, 0]])
    changed = anchor_spectral_compact_action_v1(layout, permuted, source[:, :, [1, 0]], rate, damping)
    check(changed, original[:, :, [1, 0]], dtype, "query_permutation")

    query = torch.cos(torch.arange(24, dtype=dtype)).reshape(4, 2, 3)
    reference = torch.sin(torch.arange(36, dtype=dtype)).reshape(4, 3, 3)
    mixture = torch.ones(3, 3, dtype=dtype) / 3
    base_message = compact_anchor_messages_v1(layout, gauge, query, reference, rate, damping, mixture)
    changed_message = compact_anchor_messages_v1(layout, permuted, query[:, [1, 0]],
                                                  reference[:, [2, 0, 1]], rate, damping, mixture)
    check(changed_message.targets, base_message.targets[:, [1, 0]], dtype, "query_reference_message_permutation")
    check(changed_message.mass, base_message.mass[:, [1, 0]], dtype, "query_reference_mass_permutation")


def test_empty_targets_zero_references_and_singleton_packing():
    layout = prepare_compact_anchor_layout_v1(torch.tensor([[4., 6.]], dtype=torch.float64),
                                              torch.empty(0, dtype=torch.float64), num_anchors=1)
    geometry = CompactSpectralGeometryV1(torch.tensor([[[2.]]], dtype=torch.float64),
        torch.ones(1, 1, 1, 1, dtype=torch.float64), torch.empty(1, 1, 1, 0, dtype=torch.float64))
    query = torch.tensor([[[.23]]], dtype=torch.float64, requires_grad=True)
    rate = torch.tensor([[.7]], dtype=torch.float64, requires_grad=True)
    damping = torch.tensor([[.4]], dtype=torch.float64, requires_grad=True)
    message = compact_anchor_messages_v1(layout, geometry, query, torch.empty(1, 0, 1, dtype=torch.float64),
                                          rate, damping, torch.ones(1, 1, dtype=torch.float64))
    check(message.anchors, query, torch.float64, "singleton_self")
    check(message.mass, torch.tensor([[[2.]]], dtype=torch.float64), torch.float64, "singleton_mass")
    assert message.targets.shape == (0, 1, 1)
    qgrad, rgrad, dgrad = torch.autograd.grad(message.anchors.sum(), (query, rate, damping))
    assert torch.equal(qgrad, torch.ones_like(query))
    assert torch.equal(rgrad, torch.zeros_like(rate))
    assert torch.equal(dgrad, torch.zeros_like(damping))


def test_author_ordinary_dense_oracle_failures_reproduce_without_tolerance_change(monkeypatch):
    spec = importlib.util.spec_from_file_location("author_handmade", "tests/test_anchor_spectral_compact_v1.py")
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    # First prove the padding's block identity with a closed diagonal value and derivative.
    diagonal = torch.tensor([2.544750485712468, 3.384744881932209, 2.53611600617982],
                             dtype=torch.float64, requires_grad=True)
    dt = .013333333333333308
    matrix = -dt * torch.diag(diagonal)
    closed = torch.diag((-dt * diagonal).exp())
    ordinary = torch.matrix_exp(matrix)
    padded = author.oracle_matrix_exp(matrix)
    check(padded, closed, torch.float64, "padded_diagonal_closed")
    assert float((ordinary - closed).abs().max()) > 1e-11
    METRICS.append(dict(name="ordinary_diagonal_error", max_absolute=float((ordinary - closed).abs().max())))
    a = torch.autograd.grad(padded.diagonal().sum(), diagonal, retain_graph=True)[0]
    b = torch.autograd.grad(closed.diagonal().sum(), diagonal)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
    # The preserved failing log prints these exact constants at its close helper.
    log = Path("research/reviews/anchor_spectral_compact_v1_author/pytest_attempt_01.txt").read_text()
    for line in ("(1e-8, 1e-10)", "(1e-10, 1e-12)", "(1e-3, 1e-5)", "(2e-4, 2e-5)"):
        assert line in log
    monkeypatch.setattr(author, "oracle_matrix_exp", torch.matrix_exp)
    for control in ("full", "diagonal", "gain_matched_diagonal"):
        with pytest.raises(AssertionError):
            author.test_raw_mass_messages_and_vjp_against_dense_and_original(torch.float64, control)
    with pytest.raises(AssertionError):
        author.test_holds_with_no_anchors_and_many_anchors_in_middle(torch.float64)
    METRICS.append(dict(name="reproduced_original_four_oracle_failures", count=4))


def test_geometry_view_is_alias_and_does_not_replace_bank_admission():
    _, _, _, operators, _ = fixture(torch.float64)
    full = CompactSpectralGeometryV1.from_operators(operators)
    diagonal = CompactSpectralGeometryV1.from_operators(operators, include_diagonal=True)
    assert full.cqr.data_ptr() == operators.cqr.data_ptr()
    assert full.eigenvalues.data_ptr() == operators.eigenvalues.data_ptr()
    assert full.eigenvectors.data_ptr() == operators.eigenvectors.data_ptr()
    assert not hasattr(full, "cqq") and full.diagonal is None
    # A CPU diagonal view retains the original full-Cqq storage unless copied.
    assert diagonal.diagonal.untyped_storage().data_ptr() == operators.cqq.untyped_storage().data_ptr()
    assert diagonal.diagonal.untyped_storage().nbytes() == operators.cqq.untyped_storage().nbytes()
    copied = diagonal.to("cpu", dtype=torch.float32)
    assert copied.diagonal.untyped_storage().nbytes() == copied.diagonal.numel() * copied.diagonal.element_size()
    # from_operators only extracts arrays. Original offline bank admission is required.
    corrupted = replace(operators, eigenvectors=torch.zeros_like(operators.eigenvectors))
    view = CompactSpectralGeometryV1.from_operators(corrupted)
    assert torch.equal(view.eigenvectors, corrupted.eigenvectors)

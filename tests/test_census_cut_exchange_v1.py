"""H021 handmade CPU qualification. No actual project arrays or model fits."""
from dataclasses import replace
import os
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import arggnn.census_cut_exchange_v1 as module
from arggnn.census_cut_exchange_v1 import (
    ObservedDatedTreeV1, CensusCutExchangeLayerV1, build_census_cut_partition_v1,
    clade_turnover_rho_v1, observed_tmrca_donor_weights_v1,
    prepare_whole_tree_encoder_inputs_v1, gather_whole_tree_context_v1,
    prepare_census_exchange_bank_v1, permuted_group_control_v1,
    block_average_v1, census_cut_action_v1, census_cut_messages_v1,
    pointwise_local_messages_v1)

Q = torch.tensor([101, 103, 107, 109])
R = torch.tensor([201, 203])
SCORED = torch.tensor([101, 103])
INTERVALS = torch.tensor([[0., .5], [.5, 1.]], dtype=torch.float64)
POSITIONS = torch.tensor([.5, .0, .12, .5, .49, .82, 1 / 6, .999], dtype=torch.float64)


@pytest.fixture(autouse=True)
def cpu_handmade_only():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)


def close(actual, expected, dtype=torch.float64, gradient=False):
    if dtype == torch.float64:
        rt, at = (1e-8, 1e-10) if gradient else (1e-10, 1e-12)
    else:
        rt, at = (1e-3, 1e-5) if gradient else (2e-4, 2e-5)
    torch.testing.assert_close(actual, expected, rtol=rt, atol=at)


def tree(which=0):
    # Actual identities are disjoint from graph-local native indexes 0..9.
    # Cut 0 is {101,103},{107},{109}; cut 1 is {101},{103,107},{109}.
    parent = [6, 6, 7, 7, 8, 9, 8, 9, 9, -1]
    times = [0., 100., 200., 300., 400., 600., 300., 450., 550., 800.]
    if which == 1:
        parent[:4] = [7, 6, 6, 7]
        times[6:8] = [330., 500.]
    return ObservedDatedTreeV1(torch.tensor(parent), torch.tensor(times, dtype=torch.float64),
                               torch.cat((Q, R)), torch.arange(6))


def bank(dtype=torch.float64, *, trees=None, intervals=INTERVALS, positions=POSITIONS,
         queries=Q, references=R, scored=SCORED, anchors=3):
    trees = [tree(0), tree(1)] if trees is None else trees
    result = prepare_census_exchange_bank_v1(trees, intervals, positions, queries, references,
                                             scored, num_anchors=anchors)
    return result.to("cpu", dtype=dtype)


def leaves(b, dtype=torch.float64):
    generator = torch.Generator().manual_seed(883)
    a = len(b.layout.anchors)
    query = (.6 * torch.randn(a, 4, 2, dtype=torch.float64, generator=generator)).tanh().to(dtype)
    reference = (.5 * torch.randn(a, 2, 2, dtype=torch.float64, generator=generator)).tanh().to(dtype)
    raw_exchange = torch.tensor([[.1, -.2], [.3, -.1], [-.2, .2]], dtype=dtype)
    raw_damping = torch.tensor([[.2, -.1], [-.2, .1], [.1, .3]], dtype=dtype)
    raw_eta = torch.tensor([[.2, -.1], [-.1, .3], [.1, .2]], dtype=dtype)
    logits = torch.tensor([[.4, -.3], [-.2, .5], [.1, -.2]], dtype=dtype)
    return tuple(x.requires_grad_() for x in (query, reference, raw_exchange, raw_damping, raw_eta, logits))


def params(values):
    query, reference, raw_exchange, raw_damping, raw_eta, logits = values
    exchange = raw_exchange.new_tensor([[.7, 1.3], [1.1, .6], [.9, 1.4]]) * raw_exchange.exp()
    damping = raw_damping.new_tensor([[.5, .8], [.3, .6], [.7, .4]]) * raw_damping.exp()
    eta = raw_eta.new_tensor([[.7, 1.2], [1.1, .8], [.6, 1.3]]) * raw_eta.exp()
    return query, reference, exchange, damping, eta, logits.softmax(0)


def bundle(b, query, reference, exchange):
    w = b.donor_weights[b.layout.anchor_graph]
    source = query[:, None] + (w @ reference[:, None]) * exchange[None, :, None, :]
    mass = torch.ones_like(source) + w.sum(-1)[..., None] * exchange[None, :, None, :]
    return torch.stack((source, mass), -1)


def dense_projection(partition, dtype):
    k = len(partition.group_members)
    p = F.one_hot(partition.group_index, num_classes=k).to(dtype)
    return (p / p.sum(0)[None]) @ p.T


def padded_dense_exponential(matrix):
    # Qualified independent Torch dense exp oracle, as in the earlier H020 audit.
    q = len(matrix)
    return torch.matrix_exp(torch.block_diag(matrix, matrix.new_ones((1, 1))))[:q, :q]


def oracle_path(b, intervals, exchange, damping, eta, x, y, branch, channel, control):
    q = len(b.query_ids)
    identity = torch.eye(q, dtype=exchange.dtype)
    result = identity
    lower, upper = min(x, y), max(x, y)
    order = torch.argsort(intervals[:, 0]).tolist()
    for rank, graph in enumerate(order):
        left, right = intervals[graph].tolist()
        if rank and lower < left <= upper:
            before = order[rank - 1]
            rho = clade_turnover_rho_v1(b.partitions[before], b.partitions[graph]).to(exchange.dtype)
            exponent = torch.zeros_like(eta[branch, channel]) if control == "eta0" else eta[branch, channel]
            jump = torch.diag(torch.exp(rho.log() * exponent))
            result = jump @ result
        duration = max(0., min(upper, right) - max(lower, left))
        if duration:
            projection = dense_projection(b.partitions[graph], exchange.dtype)
            generator = damping[branch, channel] * identity + exchange[branch, channel] * (identity - projection)
            if control == "raw_diagonal":
                generator = torch.diag(generator.diagonal())
            result = padded_dense_exponential(-duration * generator) @ result
    return result if x >= y else result.T


def oracle_action(b, intervals, sources, exchange, damping, eta, control="proposed"):
    a, s, q, h, v = sources.shape
    outputs = []
    for x in b.layout.evaluations.tolist():
        branches = []
        for branch in range(s):
            channels = []
            for channel in range(h):
                contributions = [b.layout.weights[anchor].to(sources.dtype) * oracle_path(
                    b, intervals, exchange, damping, eta, x, y, branch, channel, control)
                    @ sources[anchor, branch, :, channel]
                    for anchor, y in enumerate(b.layout.anchors.tolist())]
                channels.append(torch.stack(contributions).sum(0))
            branches.append(torch.stack(channels, 1))
        outputs.append(torch.stack(branches))
    return torch.stack(outputs)


def parts(raw, mixture):
    numerator, mass = (raw * mixture[None, :, None, :, None]).sum(1).unbind(-1)
    return numerator, mass, numerator / mass


def objective(normalized, mass):
    weights = torch.linspace(-.8, .9, normalized.numel(), dtype=normalized.dtype).reshape_as(normalized)
    return (normalized * weights).sum() + .017 * mass.log().sum()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("control", ["proposed", "eta0", "permuted_groups", "raw_diagonal"])
def test_finite_anchor_output_mass_and_vjp_against_independent_dense_oracle(dtype, control):
    b = bank(dtype)
    if control == "permuted_groups":
        b = permuted_group_control_v1(b, torch.tensor([2, 3, 1, 0]))
    values = leaves(b, dtype)
    query, reference, exchange, damping, eta, mixture = params(values)
    sources = bundle(b, query, reference, exchange)
    actual = census_cut_action_v1(b, sources, exchange, damping, eta, control=control, readout_chunk_size=2)
    expected = oracle_action(b, INTERVALS, sources, exchange, damping, eta, control)
    close(actual, expected, dtype)
    ap, ep = parts(actual, mixture), parts(expected, mixture)
    for x, y in zip(ap, ep):
        close(x, y, dtype)
    assert (ap[1] > 0).all()
    assert ap[2].abs().max() <= 1 + (1e-10 if dtype == torch.float64 else 2e-5)
    wrapped = census_cut_messages_v1(b, query, reference, exchange, damping, eta, mixture,
                                      control=control, readout_chunk_size=2)
    close(wrapped.mass, ap[1], dtype)
    close(wrapped.anchors, ap[2][b.layout.anchor_output], dtype)
    close(wrapped.targets, ap[2][b.layout.target_output], dtype)
    gradients = [torch.autograd.grad(objective(x[2], x[1]), values, retain_graph=True, allow_unused=True)
                 for x in (ap, ep)]
    for name, x, y in zip(("query", "reference", "exchange", "damping", "eta", "mixture"), *gradients):
        if control == "eta0" and name == "eta":
            assert x is y is None
        else:
            assert x is not None and y is not None and torch.isfinite(x).all()
            close(x, y, dtype, gradient=True)
    if control != "eta0":
        assert gradients[0][4].abs().sum() > 1e-5


def test_cut_identity_relation_and_exact_census_collision():
    t = tree(0)
    p = build_census_cut_partition_v1(t, Q, R, SCORED)
    assert p.group_members == (frozenset([101, 103]), frozenset([107]), frozenset([109]))
    assert p.older_reference_ids == (201, 203)
    assert all(not set(R.tolist()) & group for group in p.group_members)
    collision_times = t.node_times.clone()
    collision_times[6] = 350.
    collision = build_census_cut_partition_v1(replace(t, node_times=collision_times), Q, R, SCORED)
    assert collision.group_members == p.group_members  # MRCA == census is included.
    above_times = collision_times.clone()
    above_times[6] = 350.0001
    above = build_census_cut_partition_v1(replace(t, node_times=above_times), Q, R, SCORED)
    assert len(above.group_members) == 4
    # Native vertex IDs differ from actual sample IDs and are never exported as ancestors.
    assert not hasattr(p, "persistent_ancestor_ids")
    assert set.union(*(set(x) for x in p.group_members)) == set(Q.tolist())


@pytest.mark.parametrize("query_slot,age,error", [(0, 350., "Scored"), (2, 350., "Context"), (2, 400., "Context")])
def test_checked_query_context_dates_reject_equal_or_older(query_slot, age, error):
    t = tree(0)
    times = t.node_times.clone()
    times[query_slot] = age
    times[6] = max(float(times[6]), age)
    with pytest.raises(ValueError, match=error):
        build_census_cut_partition_v1(replace(t, node_times=times), Q, R, SCORED)


def test_policy_is_explicit_older_references_are_proxies_not_targets():
    t = tree(0)
    with pytest.raises(ValueError, match="checked require_younger"):
        build_census_cut_partition_v1(t, Q, R, SCORED, context_date_policy="assume_eligible")
    weights = observed_tmrca_donor_weights_v1(t, Q, R)
    close(weights.sum(-1), torch.ones(3, 4, dtype=weights.dtype))
    assert (weights[..., 0] > 0).all() and (weights[..., 1] > 0).all()
    expected = torch.softmax(-torch.log1p(torch.tensor([[550., 800.], [550., 800.], [800., 800.], [800., 800.]],
        dtype=torch.float64))[None] / math.log1p(10000.) / torch.tensor([.05, .1, .2], dtype=torch.float64)[:, None, None], -1)
    close(weights, expected)
    changed = t.node_times.clone()
    changed[4] = 350.
    p = build_census_cut_partition_v1(replace(t, node_times=changed), Q, R, SCORED)
    assert 201 in p.older_reference_ids  # Reference == census remains outside hard cut.
    with pytest.raises(ValueError, match="Scored"):
        build_census_cut_partition_v1(t, torch.cat((Q, R[:1])), R[1:], torch.cat((SCORED, R[:1])))


def test_sample_aligned_jaccard_and_local_labels_do_not_imply_ancestral_identity():
    left = build_census_cut_partition_v1(tree(0), Q, R, SCORED)
    right = build_census_cut_partition_v1(tree(1), Q, R, SCORED)
    close(clade_turnover_rho_v1(left, right), torch.tensor([.5, 1 / 3, .5, 1.], dtype=torch.float64))
    # Reverse local group labels, preserving memberships. rho must be unaffected.
    labels = (len(right.group_members) - 1) - right.group_index
    relabeled = replace(right, group_index=labels, group_members=tuple(reversed(right.group_members)))
    close(clade_turnover_rho_v1(left, relabeled), clade_turnover_rho_v1(left, right))
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_right = build_census_cut_partition_v1(tree(1), Q[permutation], R, SCORED)
    close(clade_turnover_rho_v1(left, permuted_right), clade_turnover_rho_v1(left, right)[permutation])


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_block_projection_and_closed_form_are_dense_equivalent_and_positive(dtype, monkeypatch):
    b = bank(dtype)
    state = torch.rand(3, 4, 2, 2, dtype=dtype, requires_grad=True)
    values = leaves(b, dtype)
    _, _, exchange, damping, _, _ = params(values)
    projection = dense_projection(b.partitions[0], dtype)
    actual = block_average_v1(state, b.group_index[0], b.group_size_by_query[0])
    expected = (projection @ state.flatten(-2)).reshape_as(state)
    close(actual, expected, dtype)
    close(block_average_v1(actual, b.group_index[0], b.group_size_by_query[0]), actual, dtype)
    duration = torch.tensor(.13, dtype=dtype)
    closed = module._hold_action(state, b.group_index[0], b.group_size_by_query[0], exchange, damping, duration)
    dense = torch.stack([torch.stack([padded_dense_exponential(-duration * (damping[s, h] * torch.eye(4, dtype=dtype)
        + exchange[s, h] * (torch.eye(4, dtype=dtype) - projection))) @ state[s, :, h]
        for h in range(2)], 1) for s in range(3)])
    close(closed, dense, dtype)
    assert (closed >= 0).all()
    assert closed.max() <= state.max()
    # Production actions must not solve eigenproblems or call a matrix exponential.
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("eigh")))
    monkeypatch.setattr(torch, "matrix_exp", lambda *a, **k: (_ for _ in ()).throw(AssertionError("matrix_exp")))
    positive_sources = torch.rand(3, 3, 4, 2, 2, dtype=dtype)
    eta = torch.ones(3, 2, dtype=dtype)
    assert (census_cut_action_v1(b, positive_sources, exchange, damping, eta) >= 0).all()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_boundary_anchor_readout_collision_self_once_duplicates_and_order(dtype):
    b = bank(dtype)
    assert b.layout.anchor_graph[1] == 1
    assert b.layout.target_output[0] == b.layout.target_output[3] == b.layout.anchor_output[1]
    values = leaves(b, dtype)
    _, _, exchange, damping, eta, _ = params(values)
    sources = torch.zeros(3, 3, 4, 2, 2, dtype=dtype)
    sources[1] = torch.arange(1, 49, dtype=dtype).reshape(3, 4, 2, 2)
    actual = census_cut_action_v1(b, sources, exchange, damping, eta)
    torch.testing.assert_close(actual[b.layout.anchor_output[1]], b.layout.weights[1].to(dtype) * sources[1], rtol=0, atol=0)
    close(actual, oracle_action(b, INTERVALS, sources, exchange, damping, eta), dtype)
    p0, p1 = (dense_projection(x, dtype) for x in b.partitions)
    assert torch.linalg.matrix_norm(p0 @ p1 - p1 @ p0) > .1
    reverse = bank(dtype, trees=[tree(1), tree(0)])
    reversed_value = census_cut_action_v1(reverse, sources, exchange, damping, eta)
    assert (actual - reversed_value).abs().max() > .01
    serialized = bank(dtype, trees=[tree(1), tree(0)], intervals=INTERVALS.flip(0))
    close(actual, census_cut_action_v1(serialized, sources, exchange, damping, eta), dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_eta0_static_partition_and_raw_diagonal_controls(dtype):
    b = bank(dtype)
    values = leaves(b, dtype)
    query, reference, exchange, damping, eta, mixture = params(values)
    sources = bundle(b, query, reference, exchange)
    eta0 = census_cut_action_v1(b, sources, exchange, damping, eta, control="eta0")
    zero_eta = census_cut_action_v1(b, sources, exchange, damping, torch.zeros_like(eta))
    close(eta0, zero_eta, dtype)
    proposed = census_cut_action_v1(b, sources, exchange, damping, eta)
    assert (proposed - eta0).abs().max() > .01
    static = bank(dtype, trees=[tree(0), tree(0)])
    static_sources = bundle(static, query, reference, exchange)
    active = census_cut_action_v1(static, static_sources, exchange, damping, eta)
    inactive = census_cut_action_v1(static, static_sources, exchange, damping, eta, control="eta0")
    close(active, inactive, dtype)
    eta_gradient = torch.autograd.grad(active.square().sum(), values[4], retain_graph=True)[0]
    close(eta_gradient, torch.zeros_like(eta_gradient), dtype, True)
    diagonal = census_cut_action_v1(b, sources, exchange, damping, eta, control="raw_diagonal")
    assert (diagonal - proposed).abs().max() > .001
    close(diagonal, oracle_action(b, INTERVALS, sources, exchange, damping, eta, "raw_diagonal"), dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_global_membership_permutation_control_preserves_statistics_not_signals(dtype):
    b = bank(dtype)
    permutation = torch.tensor([2, 3, 1, 0])
    shuffled = permuted_group_control_v1(b, permutation)
    assert shuffled.query_ids is b.query_ids and shuffled.donor_weights is b.donor_weights
    close(shuffled.boundary_rho, b.boundary_rho[:, permutation], dtype)
    close(shuffled.group_size_by_query, b.group_size_by_query[:, permutation], dtype)
    values = leaves(b, dtype)
    query, reference, exchange, damping, eta, mixture = params(values)
    original = census_cut_messages_v1(b, query, reference, exchange, damping, eta, mixture)
    control = census_cut_messages_v1(shuffled, query, reference, exchange, damping, eta, mixture,
                                     control="permuted_groups")
    assert (original.targets - control.targets).abs().max() > .001
    close(original.sources, control.sources, dtype)
    close(original.mass_sources, control.mass_sources, dtype)
    with pytest.raises(ValueError, match="checked fixed global"):
        census_cut_messages_v1(b, query, reference, exchange, damping, eta, mixture, control="permuted_groups")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_actual_sample_permutation_and_hold_subdivision(dtype):
    b = bank(dtype)
    values = leaves(b, dtype)
    query, reference, exchange, damping, eta, mixture = params(values)
    base = census_cut_messages_v1(b, query, reference, exchange, damping, eta, mixture)
    qp, rp = torch.tensor([2, 0, 3, 1]), torch.tensor([1, 0])
    sample_permuted = bank(dtype, queries=Q[qp], references=R[rp])
    changed = census_cut_messages_v1(sample_permuted, query[:, qp], reference[:, rp], exchange, damping, eta, mixture)
    close(changed.targets, base.targets[:, qp], dtype)
    close(changed.mass, base.mass[:, qp], dtype)
    gradients = [torch.autograd.grad(x.targets.square().sum(), values[2:5], retain_graph=True) for x in (base, changed)]
    for x, y in zip(*gradients):
        close(x, y, dtype, True)
    split_intervals = torch.tensor([[0., .13], [.13, .5], [.5, 1.]], dtype=torch.float64)
    split = bank(dtype, trees=[tree(0), tree(0), tree(1)], intervals=split_intervals)
    inserted_event = int(torch.searchsorted(split.layout.coordinates, torch.tensor(.13, dtype=torch.float64)))
    close(split.boundary_rho[inserted_event], torch.ones(4, dtype=dtype), dtype)
    split_result = census_cut_messages_v1(split, query, reference, exchange, damping, eta, mixture)
    close(split_result.targets, base.targets, dtype)
    close(split_result.mass, base.mass, dtype)


def test_serial_age_subdivision_and_whole_tree_encoder_interface():
    original = tree(0)
    parent = torch.cat((original.parent.clone(), torch.tensor([8])))
    parent[6] = 10
    times = torch.cat((original.node_times, torch.tensor([350.], dtype=torch.float64)))
    serial = replace(original, parent=parent, node_times=times)
    before = build_census_cut_partition_v1(original, Q, R, SCORED)
    after = build_census_cut_partition_v1(serial, Q, R, SCORED)
    assert before.group_members == after.group_members
    close(clade_turnover_rho_v1(before, after), torch.ones(4, dtype=torch.float64))
    close(observed_tmrca_donor_weights_v1(original, Q, R), observed_tmrca_donor_weights_v1(serial, Q, R))
    def features(t):
        value = torch.zeros(len(t.parent), 47, dtype=torch.float64)
        value[:, 6] = torch.log1p(t.node_times) / math.log1p(10000.)
        value[t.sample_nodes, 5] = 1.
        value[4, 0] = 1.
        value[5, 1] = 1.
        return value
    original_inputs = prepare_whole_tree_encoder_inputs_v1(original, features(original))
    serial_inputs = prepare_whole_tree_encoder_inputs_v1(serial, features(serial))
    for name in ("parent", "node_times", "features", "branch_duration", "sample_ids", "sample_nodes"):
        torch.testing.assert_close(getattr(original_inputs, name), getattr(serial_inputs, name), rtol=0, atol=0)
    assert len(original_inputs.parent) == 10  # Every branching interior remains.
    assert len(serial_inputs.parent) == 10 and 10 not in serial_inputs.native_vertex_index.tolist()
    assert original_inputs.upward_order == serial_inputs.upward_order
    assert original_inputs.downward_order == serial_inputs.downward_order
    encoded = torch.randn(10, 5, dtype=torch.float64, requires_grad=True)
    query, reference = gather_whole_tree_context_v1(serial_inputs, encoded, Q, R)
    close(query, encoded[:4])
    close(reference, encoded[4:6])
    (query.square().sum() + reference.sum()).backward()
    assert encoded.grad[:6].abs().sum() > 0
    bad_features = features(original)
    bad_features[:, 6] = 0.
    with pytest.raises(ValueError, match="Stored encoded ages"):
        prepare_whole_tree_encoder_inputs_v1(original, bad_features)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_singleton_exchange_and_pointwise_local_contrast(dtype):
    t = tree(0)
    times = t.node_times.clone()
    times[6] = 400.
    b = bank(dtype, trees=[replace(t, node_times=times)] * 2)
    assert all(len(x) == 1 for x in b.partitions[0].group_members)
    values = leaves(b, dtype)
    query, reference, exchange, damping, eta, mixture = params(values)
    sources = bundle(b, query, reference, exchange)
    actual = census_cut_action_v1(b, sources, exchange, damping, eta)
    direct = torch.stack([sum(b.layout.weights[a].to(dtype) * torch.exp(-abs(x - y) * damping)[:, None, :, None]
                              * sources[a] for a, y in enumerate(b.layout.anchors.tolist()))
                          for x in b.layout.evaluations.tolist()])
    close(actual, direct, dtype)
    local, mass = pointwise_local_messages_v1(query, reference, b.donor_weights[b.layout.anchor_graph], exchange, mixture)
    expected_raw = bundle(b, query, reference, exchange)
    ep = parts(expected_raw, mixture)
    close(local, ep[2], dtype)
    close(mass, ep[1], dtype)
    changed_query = query.clone()
    changed_query[2] = -changed_query[2]
    changed, _ = pointwise_local_messages_v1(changed_query, reference, b.donor_weights[b.layout.anchor_graph], exchange, mixture)
    torch.testing.assert_close(local[:2], changed[:2], rtol=0, atol=0)
    assert (changed[2] - local[2]).abs().sum() > .01


def test_completed_dense_kernel_is_symmetric_psd_and_jumps_do_not_commute():
    b = bank()
    _, _, exchange, damping, eta, _ = params(leaves(b))
    points = [0., .2, .5, .7, .999]
    gram = torch.cat([torch.cat([oracle_path(b, INTERVALS, exchange, damping, eta, x, y, 0, 0, "proposed")
                               for y in points], 1) for x in points], 0)
    close(gram, gram.T)
    assert torch.linalg.eigvalsh(gram).min() > -1e-11
    projection = dense_projection(b.partitions[0], torch.float64)
    rho = clade_turnover_rho_v1(*b.partitions)
    jump = torch.diag(rho ** eta[0, 0])
    assert torch.linalg.matrix_norm(projection @ jump - jump @ projection) > .01


def test_five_point_source_rate_retention_and_mixture_directional_derivatives():
    b = bank()
    values = leaves(b)
    def evaluate(inputs):
        query, reference, exchange, damping, eta, mixture = params(inputs)
        msg = census_cut_messages_v1(b, query, reference, exchange, damping, eta, mixture)
        return objective(msg.targets, msg.mass)
    gradients = torch.autograd.grad(evaluate(values), values)
    generator = torch.Generator().manual_seed(951)
    direction = tuple(torch.randn(x.shape, dtype=x.dtype, generator=generator) for x in values)
    expected = sum((g * d).sum() for g, d in zip(gradients, direction))
    step = 2e-4
    points = {k: evaluate(tuple(x.detach() + k * step * d for x, d in zip(values, direction)))
              for k in (-2, -1, 1, 2)}
    difference = (points[-2] - 8 * points[-1] + 8 * points[1] - points[2]) / (12 * step)
    torch.testing.assert_close(expected, difference, rtol=2e-8, atol=1e-9)


@pytest.mark.parametrize("control", ["proposed", "eta0", "raw_diagonal", "permuted_groups", "local"])
def test_executable_layer_two_depth_gates_have_finite_gradients(control):
    b = bank()
    if control == "permuted_groups":
        b = permuted_group_control_v1(b, torch.tensor([2, 3, 1, 0]))
    torch.manual_seed(771)
    layers = nn.ModuleList([CensusCutExchangeLayerV1(5, 2, 3, initial_exchange=(.7, 1.1, 1.8), initial_damping=.5)
                           for _ in range(2)]).double()
    contexts = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    references = torch.randn(2, 2, 5, dtype=torch.float64, requires_grad=True)
    anchors = contexts[b.layout.anchor_graph]
    target = contexts[b.target_graph]
    ar = references[b.layout.anchor_graph]
    tr = references[b.target_graph]
    for layer in layers:
        anchors, target = layer(b, anchors, ar, target, control=control, target_reference_context=tr, readout_chunk_size=2)
    (target.square() * torch.linspace(.2, 1., target.numel()).reshape_as(target)).sum().backward()
    assert contexts.grad is not None and references.grad is not None
    for layer in layers:
        for name, parameter in layer.named_parameters():
            inactive = (name == "raw_eta" and control in {"eta0", "local"}) or (name == "raw_damping" and control == "local")
            if inactive:
                assert parameter.grad is None
            else:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert (layer.exchange > 0).all() and (layer.damping > 0).all() and (layer.eta > 0).all()
    # No optimizer or model fit is performed in this architecture unit fixture.


def test_native_dates_roles_tree_shapes_and_numeric_failure_contracts():
    b = bank()
    values = leaves(b)
    query, reference, exchange, damping, eta, mixture = params(values)
    for bad_exchange, bad_eta in ((torch.zeros_like(exchange), eta), (exchange, -eta), (exchange, eta * float("inf"))):
        with pytest.raises(ValueError, match="positive rates"):
            census_cut_messages_v1(b, query, reference, bad_exchange, damping, bad_eta, mixture)
    with pytest.raises(ValueError, match="mass is nonpositive"):
        census_cut_messages_v1(b, query, reference, exchange, torch.full_like(damping, 1e9), eta, mixture)
    with pytest.raises(ValueError, match="mass is nonpositive"):
        census_cut_messages_v1(b, query, reference, exchange, damping, eta, torch.zeros_like(mixture))
    with pytest.raises(ValueError, match="Retention overflow/underflow"):
        census_cut_messages_v1(b, query, reference, exchange, damping, torch.full_like(eta, 1e9), mixture)
    bad_geometry = replace(b, donor_weights=b.donor_weights.clone().requires_grad_())
    with pytest.raises(ValueError, match="geometry must remain fixed"):
        census_cut_messages_v1(bad_geometry, query, reference, exchange, damping, eta, mixture)
    changed_dates = tree(1).node_times.clone()
    changed_dates[2] += 1.
    with pytest.raises(ValueError, match="sample dates changed"):
        bank(trees=[tree(0), replace(tree(1), node_times=changed_dates)])
    reversed_age = tree(0).node_times.clone()
    reversed_age[6] = 50.
    with pytest.raises(ValueError, match="parent time"):
        build_census_cut_partition_v1(replace(tree(0), node_times=reversed_age), Q, R, SCORED)
    with pytest.raises(ValueError, match="half-open"):
        bank(positions=torch.tensor([1.], dtype=torch.float64))
    with pytest.raises(ValueError, match="global Q permutation"):
        permuted_group_control_v1(b, torch.tensor([0, 1, 2, 8]))


def test_invalid_local_mixture_and_negative_sample_identity_reject():
    b = bank()
    query, reference, exchange, _, _, mixture = params(leaves(b))
    weights = b.donor_weights[b.layout.anchor_graph]
    for invalid in (-mixture, mixture * float("inf"), mixture * float("nan")):
        with pytest.raises(ValueError, match="Scale mixture"):
            pointwise_local_messages_v1(query, reference, weights, exchange, invalid)
    with pytest.raises(ValueError, match="mass is nonpositive"):
        pointwise_local_messages_v1(query, reference, weights, exchange, torch.zeros_like(mixture))
    t = tree()
    invalid_ids = t.sample_ids.clone()
    invalid_ids[0] = -1
    with pytest.raises(ValueError, match="nonnegative actual sample"):
        build_census_cut_partition_v1(replace(t, sample_ids=invalid_ids), Q, R, SCORED)

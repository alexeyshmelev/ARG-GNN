"""Tiny CPU equation and correspondence checks for the GraphSSM adaptation."""
from dataclasses import replace
import math

import pytest
import torch

from arggnn.data import GraphSequence
from arggnn.graphssm_baseline import (
    DiagonalSISOStateSpace,
    GraphSSMAdaptation,
    RepresentationInterpolation,
    align_node_memory,
)


def variable_node_sequence() -> GraphSequence:
    xs, edges, ids, graph_ids, clades = [], [], [], [], []
    offset = 0
    for graph, internals in enumerate([1, 3, 2]):
        count = 3 + internals
        features = torch.zeros(count, 5)
        features[:3, 0] = torch.tensor([1., 0., 1.])
        features[:3, 1] = torch.tensor([0., 1., 0.])
        features[:, 2] = torch.arange(count) * .2 + graph * .7
        features[:, 3] = .4 + graph * .1
        features[:, 4] = torch.arange(count) * .1
        xs.append(features)
        # First graph is a polytomy. Additional graphs have binary/unary nodes.
        if internals == 1:
            spatial = torch.tensor([[3, 3, 3], [0, 1, 2]])
        else:
            spatial = torch.tensor([[3, 3, 4, 4], [0, 1, 3, 2]])
            if internals == 3:
                spatial = torch.cat((spatial, torch.tensor([[5], [4]])), dim=1)
        spatial = torch.cat((spatial, spatial.flip(0)), dim=1)
        edges.append(spatial + offset)
        ids.append(torch.tensor([10, 20, 30] + [-1] * internals))
        graph_ids.append(torch.full((count,), graph, dtype=torch.long))
        clades.append(torch.tensor([[0.], [1.], [0.]] + [[1.]] * internals))
        offset += count
    seq = GraphSequence(
        x=torch.cat(xs), edge_index=torch.cat(edges, dim=1),
        node_graph=torch.cat(graph_ids), node_ids=torch.cat(ids),
        clades=torch.cat(clades),
        intervals=torch.tensor([[0., .17], [.17, .63], [.63, 1.]]),
        query_node_ids=torch.tensor([10, 30]),
        positions=torch.tensor([.03, .10, .25, .49, .81, .96]),
        target_graph=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    seq.validate()
    return seq


def make_model(**kwargs):
    torch.manual_seed(841)
    return GraphSSMAdaptation(5, 3, hidden_dim=8, num_layers=2,
                             state_dim=4, dropout=0., **kwargs).eval()


def test_paper_cell_matches_equation_nine_and_carries_raw_memory():
    cell = DiagonalSISOStateSpace(2, 3, "paper").double()
    with torch.no_grad():
        cell.B.copy_(torch.tensor([[1., 2., 3.], [.2, .4, .6]]))
        cell.C.copy_(torch.tensor([[.1, -.2, .3], [.5, .2, -.1]]))
    x = torch.tensor([[.4, -.3], [.2, .8]], dtype=torch.float64)
    memory = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3) * .1
    dt = torch.tensor([.13, .47], dtype=torch.float64)
    expected = torch.exp(-dt[:, None, None] * cell.log_nA.exp()[None]) * memory
    expected += dt[:, None, None] * cell.B[None] * x[:, :, None]
    updated, projected = cell(x, memory, dt)
    torch.testing.assert_close(updated, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(projected, (expected * cell.C[None]).sum(-1))
    assert updated.shape == (2, 2, 3)
    assert projected.shape == (2, 2)
    later, _ = cell(x, updated, dt)
    assert later.shape == memory.shape


def test_exact_zoh_factor_and_constant_hold_semigroup():
    cell = DiagonalSISOStateSpace(2, 3, "zoh").double()
    x = torch.tensor([[.4, -.3]], dtype=torch.float64)
    initial = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3) * .1
    first, _ = cell(x, initial, .19)
    split, _ = cell(x, first, .42)
    whole, _ = cell(x, initial, .61)
    torch.testing.assert_close(split, whole, atol=2e-14, rtol=2e-14)
    decay, integral = cell.transition(.61, x)
    rate = cell.log_nA.exp()[None]
    torch.testing.assert_close(decay, torch.exp(-.61 * rate))
    torch.testing.assert_close(integral, -torch.expm1(-.61 * rate) / rate)


@pytest.mark.parametrize("discretization", ["paper", "zoh"])
def test_discretization_zero_narrow_stiff_and_extreme_rate_stability(discretization):
    cell = DiagonalSISOStateSpace(2, 3, discretization).double()
    with torch.no_grad():
        cell.log_nA[0] = torch.tensor([-1000., -20., 0.])
        cell.log_nA[1] = torch.tensor([2., 20., 1000.])
    x = torch.tensor([[.4, -.3], [.2, .8], [.1, .3]],
                     dtype=torch.float64, requires_grad=True)
    state = torch.ones(3, 2, 3, dtype=torch.float64)
    dt = torch.tensor([0., 1e-12, 1e4], dtype=torch.float64)
    updated, output = cell(x, state, dt)
    decay, integral = cell.transition(dt, x)
    assert torch.isfinite(updated).all() and torch.isfinite(output).all()
    assert torch.isfinite(integral).all()
    assert ((decay >= 0) & (decay <= 1)).all()
    torch.testing.assert_close(updated[0], state[0])
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(cell.log_nA.grad).all()
    assert torch.isfinite(cell.B.grad).all()
    assert torch.isfinite(cell.C.grad).all()
    with pytest.raises(ValueError, match="nonnegative"):
        cell(x, state, -1.)


def test_eq_nine_input_approximation_is_not_exact_zoh():
    paper = DiagonalSISOStateSpace(1, 1, "paper").double()
    exact = DiagonalSISOStateSpace(1, 1, "zoh").double()
    exact.load_state_dict(paper.state_dict())
    x = torch.ones(1, 1, dtype=torch.float64)
    approximate, _ = paper(x, None, 1.)
    integrated, _ = exact(x, None, 1.)
    assert not torch.allclose(approximate, integrated)


def test_memory_alignment_preserves_only_matching_stable_ids_and_gradients():
    previous_ids = torch.tensor([8, -1, 3, -1])
    current_ids = torch.tensor([-1, 3, 9, 8, -1])
    previous = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3).requires_grad_()
    aligned, matched = align_node_memory(previous_ids, current_ids, previous)
    assert torch.equal(matched, torch.tensor([False, True, False, True, False]))
    torch.testing.assert_close(aligned[1], previous[2])
    torch.testing.assert_close(aligned[3], previous[0])
    assert (aligned[~matched] == 0).all()
    aligned.sum().backward()
    assert (previous.grad[[0, 2]] == 1).all()
    assert (previous.grad[[1, 3]] == 0).all()


def test_removed_leaf_does_not_recover_old_memory_on_reappearance():
    memory = torch.tensor([[7.], [9.]])
    middle, _ = align_node_memory(torch.tensor([3, 4]), torch.tensor([4, -1]), memory)
    later, matched = align_node_memory(torch.tensor([4, -1]), torch.tensor([3, 4]), middle)
    assert torch.equal(matched, torch.tensor([False, True]))
    assert later[0].item() == 0
    assert later[1].item() == 9


def test_first_layer_mixer_matches_equations_43_44_and_bypasses_new_nodes():
    mixer = RepresentationInterpolation(2).double()
    with torch.no_grad():
        mixer.weight.weight.zero_()
        mixer.weight.bias.zero_()
        mixer.scale.weight.zero_()
        mixer.scale.bias.fill_(math.log(math.expm1(1.)))
    previous = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    current = torch.tensor([[5., 6.], [7., 8.]], dtype=torch.float64)
    output = mixer(previous, current, torch.tensor([True, False]))
    torch.testing.assert_close(output[0], (previous[0] + current[0]) / 2)
    torch.testing.assert_close(output[1], current[1])


@pytest.mark.parametrize("discretization", ["paper", "zoh"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_variable_internal_nodes_output_shapes_and_training(discretization, bidirectional):
    seq = variable_node_sequence()
    model = make_model(discretization=discretization, bidirectional=bidirectional)
    output = model(seq)
    assert output.shape == (6, 2, 3)
    assert torch.isfinite(output).all()
    target = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 2], [1, 2], [2, 0]])
    torch.nn.functional.cross_entropy(output.reshape(-1, 3), target.reshape(-1)).backward()
    assert model.blocks[0].diffusion.root.weight.grad.abs().sum() > 0
    assert model.blocks[0].cell.log_nA.grad.abs().sum() > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_scan_aligns_all_persistent_leaves_and_resets_all_anonymous_nodes():
    seq = variable_node_sequence()
    model = make_model()
    _, traces = model.blocks[0](seq.x, seq, return_trace=True)
    assert [t["graph"] for t in traces] == [0, 1, 2]
    assert not traces[0]["matched"].any()
    for previous, current in zip(traces, traces[1:]):
        ids = seq.node_ids[current["nodes"]]
        assert torch.equal(current["matched"], ids >= 0)
        assert (current["incoming_memory"][ids < 0] == 0).all()
        for index, identity in enumerate(ids):
            if identity >= 0:
                old = (seq.node_ids[previous["nodes"]] == identity).nonzero().item()
                torch.testing.assert_close(current["incoming_memory"][index],
                                           previous["updated_memory"][old])
        torch.testing.assert_close(current["delta"],
                                   seq.intervals[current["graph"]].diff().squeeze())
    assert model.blocks[0].mixer is not None
    assert model.blocks[1].mixer is None


@pytest.mark.parametrize("bidirectional", [False, True])
def test_node_index_and_identity_relabeling(bidirectional):
    seq = variable_node_sequence()
    model = make_model(bidirectional=bidirectional)
    order = torch.randperm(seq.x.shape[0], generator=torch.Generator().manual_seed(44))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    changed = replace(seq, x=seq.x[order], edge_index=inverse[seq.edge_index],
                      node_graph=seq.node_graph[order], node_ids=seq.node_ids[order],
                      clades=seq.clades[order])
    ids, queries = changed.node_ids.clone(), changed.query_node_ids.clone()
    for old, new in [(10, 200), (20, 3), (30, 91)]:
        ids[changed.node_ids == old] = new
        queries[changed.query_node_ids == old] = new
    changed = replace(changed, node_ids=ids, query_node_ids=queries)
    with torch.no_grad():
        torch.testing.assert_close(model(changed), model(seq), atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_serialization_invariance_and_actual_coordinate_order_sensitivity(bidirectional):
    seq = variable_node_sequence()
    model = make_model(bidirectional=bidirectional)
    order = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(3)
    serialized = replace(seq, intervals=seq.intervals[order],
                         node_graph=inverse[seq.node_graph],
                         target_graph=inverse[seq.target_graph])
    serialized.validate()
    with torch.no_grad():
        expected = model(seq)
        torch.testing.assert_close(model(serialized), expected, atol=3e-6, rtol=3e-6)
        actual_change = replace(seq, node_graph=inverse[seq.node_graph])
        assert (model(actual_change) - expected).abs().max() > 1e-5


@pytest.mark.parametrize("bidirectional", [False, True])
def test_default_scan_causality_and_optional_reverse_context(bidirectional):
    seq = variable_node_sequence()
    model = make_model(bidirectional=bidirectional)
    x = seq.x.clone().requires_grad_()
    model(replace(seq, x=x))[0, 0, 0].backward()
    remote = x.grad[seq.node_graph == 2].abs().max()
    if bidirectional:
        assert remote > 1e-10
    else:
        assert remote == 0


def test_fixed_grid_and_query_ordering():
    seq = variable_node_sequence()
    model = make_model()
    positions = torch.tensor([5, 1, 3, 0, 2, 4])
    queries = torch.tensor([1, 0])
    changed = replace(seq, positions=seq.positions[positions],
                      target_graph=seq.target_graph[positions],
                      query_node_ids=seq.query_node_ids[queries])
    with torch.no_grad():
        expected = model(seq)
        torch.testing.assert_close(model(changed), expected[positions][:, queries])
        # The documented readout is piecewise constant within each tree.
        torch.testing.assert_close(expected[0], expected[1])


@pytest.mark.parametrize("discretization", ["paper", "zoh"])
def test_stiff_product_overflow_is_safe_before_parameter_guard(discretization):
    cell = DiagonalSISOStateSpace(1, 1, discretization).double()
    with torch.no_grad():
        cell.log_nA.fill_(705.)  # rate finite, rate * delta overflows float64
    x = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    updated, output = cell(x, torch.ones(1, 1, 1, dtype=torch.float64), 1e4)
    assert torch.isfinite(updated).all() and torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(cell.log_nA.grad).all()



def test_subnormal_interval_is_not_rounded_up_before_stiff_decay():
    cell = DiagonalSISOStateSpace(1, 1, "zoh").double()
    with torch.no_grad():
        cell.log_nA.fill_(700.)
    x = torch.ones(1, 1, dtype=torch.float64)
    dt = 3e-310
    decay, integral = cell.transition(dt, x)
    rate = cell.log_nA.exp()[None]
    torch.testing.assert_close(decay, torch.exp(-dt * rate), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(integral, -torch.expm1(-dt * rate) / rate,
                               atol=0., rtol=1e-12)

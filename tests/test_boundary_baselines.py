"""Mechanism controls share complete invariant inputs and identity contracts."""
from dataclasses import replace
import pytest
import torch
from test_boundary_memory import sequence, cached
from arggnn.boundary_baselines import make_boundary_model

NAMES = ["boundary_feature", "boundary_local", "boundary_kernel", "boundary_gru",
         "boundary_transformer", "boundary_full", "boundary_diagonal",
         "boundary_gain_matched", "boundary_gain_leak", "boundary_bag", "boundary_equilibrium", "boundary_diagonal_equilibrium"]
SETTINGS = dict(hidden_dim=8, num_layers=2, latent_dim=3, bidirectional=True)


@pytest.mark.parametrize("name", NAMES)
def test_controls_shapes_gradients_and_query_identity(name):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    net = make_boundary_model(name, 4, 3, SETTINGS).double()
    bank = cached(sequence())
    output = net(bank)
    assert output.shape == (4, 2, 3) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    swapped = cached(replace(sequence(), query_node_ids=torch.tensor([20, 10])))
    torch.testing.assert_close(net(swapped), output.flip(1), rtol=1e-10, atol=1e-10)
    assert not any(isinstance(layer, torch.nn.Embedding) for layer in net.modules())


@pytest.mark.parametrize("name", NAMES)
def test_graph_record_serialization_does_not_change_controls(name):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    seq = sequence()
    net = make_boundary_model(name, 4, 3, SETTINGS).double()
    order = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(3)
    permuted = replace(seq, node_graph=inverse[seq.node_graph],
                       intervals=seq.intervals[order], target_graph=inverse[seq.target_graph])
    torch.testing.assert_close(net(cached(seq)), net(cached(permuted)), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name", ["boundary_feature", "boundary_local", "boundary_kernel"])
def test_nonrecurrent_controls_are_exactly_refinement_invariant(name):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    net = make_boundary_model(name, 4, 3, SETTINGS).double()
    torch.testing.assert_close(net(cached(sequence())), net(cached(sequence(unary=True, cap=True, refine=True))),
                               rtol=1e-10, atol=1e-10)

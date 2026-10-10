"""GST equations, graph-set symmetries, weighting and real feedback contracts."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from arggnn.data import GraphSequence
from arggnn.gst_baseline import (GSTAdaptation, GSTGraphNorm, GSTNormalizedGCN,
                                GSTRecordTokenAdaptation, GSTSpanTokenAdaptation)


@pytest.fixture
def sequence():
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(631)
    edges = [(0, 2), (1, 2), (3, 6), (4, 6), (5, 6), (7, 9), (8, 9)]
    edge_index = torch.tensor(edges + [(b, a) for a, b in edges]).T.contiguous()
    result = GraphSequence(
        x=torch.randn(10, 5, generator=gen, dtype=torch.float64), edge_index=edge_index,
        node_graph=torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
        node_ids=torch.tensor([0, 1, -1, 0, 1, 2, -1, 0, 1, -1]),
        clades=torch.zeros(10, 2, dtype=torch.float64),
        intervals=torch.tensor([[0., .4], [.4, .7], [.7, 1.]], dtype=torch.float64),
        query_node_ids=torch.tensor([0, 1]),
        positions=torch.tensor([.1, .25, .5, .8], dtype=torch.float64),
        target_graph=torch.tensor([0, 0, 1, 2]),
        window_bounds=torch.tensor([0., 1.], dtype=torch.float64))
    result.validate()
    return result


def model(weighting="record", routing="cross_attention", gate="paper", layers=2):
    torch.manual_seed(33)
    return GSTAdaptation(5, 3, hidden_dim=8, num_layers=layers, num_heads=2,
                         token_weighting=weighting, routing=routing,
                         gate_variant=gate).double().eval()


def refined_first_graph(seq):
    original = seq.edge_index[:, seq.node_graph[seq.edge_index[0]] == 0]
    remainder = seq.edge_index[:, seq.node_graph[seq.edge_index[0]] != 0] + 3
    return replace(seq,
        x=torch.cat((seq.x[:3], seq.x[:3], seq.x[3:])),
        node_graph=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
        node_ids=torch.cat((seq.node_ids[:3], seq.node_ids[:3], seq.node_ids[3:])),
        clades=torch.cat((seq.clades[:3], seq.clades[:3], seq.clades[3:])),
        edge_index=torch.cat((original, original+3, remainder), dim=1),
        intervals=torch.tensor([[0., .15], [.15, .4], [.4, .7], [.7, 1.]], dtype=torch.float64),
        target_graph=torch.tensor([0, 1, 2, 3]))


@pytest.mark.parametrize("routing", ["broadcast", "cross_attention"])
@pytest.mark.parametrize("gate", ["paper", "source"])
@pytest.mark.parametrize("weighting", ["record", "span"])
def test_output_and_training_gradients(sequence, weighting, routing, gate):
    net = model(weighting, routing, gate).train()
    output = net(sequence)
    assert output.shape == (4, 2, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("weighting", ["record", "span"])
@pytest.mark.parametrize("routing", ["broadcast", "cross_attention"])
def test_node_indices_leaf_ids_queries_and_storage_are_equivariant(sequence, weighting, routing):
    net = model(weighting, routing)
    baseline = net(sequence)
    permutation = torch.tensor([6, 0, 8, 3, 1, 9, 2, 7, 5, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    nodes = replace(sequence, x=sequence.x[permutation], edge_index=inverse[sequence.edge_index],
                    node_graph=sequence.node_graph[permutation], node_ids=sequence.node_ids[permutation],
                    clades=sequence.clades[permutation])
    rename = replace(sequence, node_ids=torch.where(sequence.node_ids >= 0, sequence.node_ids+1_000_000, sequence.node_ids),
                     query_node_ids=sequence.query_node_ids+1_000_000)
    order = torch.tensor([2, 0, 1]); remap = torch.empty_like(order); remap[order] = torch.arange(3)
    graphs = replace(sequence, intervals=sequence.intervals[order], node_graph=remap[sequence.node_graph],
                     target_graph=remap[sequence.target_graph])
    queries = replace(sequence, query_node_ids=sequence.query_node_ids.flip(0))
    for altered in (nodes, rename, graphs):
        torch.testing.assert_close(baseline, net(altered), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(baseline.flip(1), net(queries), rtol=1e-10, atol=1e-10)
    assert not any(isinstance(layer, nn.Embedding) for layer in net.modules())


@pytest.mark.parametrize("weighting", ["record", "span"])
def test_source_arrangement_is_order_blind_at_unchanged_target(sequence, weighting):
    # Graphs 1/2 have equal widths. Move their actual contents while leaving
    # target graph0 and its positions untouched, not merely their storage order.
    graph = sequence.node_graph.clone()
    graph[sequence.node_graph == 1] = 2
    graph[sequence.node_graph == 2] = 1
    moved = replace(sequence, node_graph=graph)
    moved.validate()
    net = model(weighting)
    torch.testing.assert_close(net(sequence)[:2], net(moved)[:2], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("routing", ["broadcast", "cross_attention"])
@pytest.mark.parametrize("gate", ["paper", "source"])
def test_span_measure_preserves_unchanged_interval_refinement(sequence, routing, gate):
    refined = refined_first_graph(sequence)
    refined.validate()
    span = model("span", routing, gate)
    torch.testing.assert_close(span(sequence), span(refined), rtol=1e-10, atol=1e-10)
    record = model("record", routing, gate)
    assert (record(sequence)-record(refined)).abs().max() > 1e-7


@pytest.mark.parametrize("weighting", ["record", "span"])
def test_context_reaches_next_local_gnn_before_readout(sequence, weighting):
    net = model(weighting)
    captured = []
    handle = net.blocks[1].gcn.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
    perturbed = sequence.x.clone(); perturbed[sequence.node_graph == 2] += torch.tensor([2., -1., .5, 3., -.7])
    baseline = net(sequence)
    changed = net(replace(sequence, x=perturbed))
    handle.remove()
    local = sequence.node_graph == 0
    assert (captured[0][local]-captured[1][local]).abs().max() > 1e-7
    assert (baseline[:2]-changed[:2]).abs().max() > 1e-7
    x = sequence.x.clone().requires_grad_(True)
    net(replace(sequence, x=x))[0, 0].square().sum().backward()
    assert x.grad[sequence.node_graph == 2].abs().sum() > 1e-8


@pytest.mark.parametrize("weighting", ["record", "span"])
def test_coordinates_do_not_create_order_or_distance_features(sequence, weighting):
    net = model(weighting)
    reflected = replace(sequence, intervals=torch.stack((1-sequence.intervals[:, 1], 1-sequence.intervals[:, 0]), -1),
                        positions=1-sequence.positions)
    torch.testing.assert_close(net(sequence), net(reflected), rtol=1e-10, atol=1e-10)
    intervals = .2*sequence.intervals + .3
    translated = replace(sequence, intervals=intervals, positions=.2*sequence.positions+.3,
                         window_bounds=torch.stack((intervals[:, 0].min(), intervals[:, 1].max())))
    torch.testing.assert_close(net(sequence), net(translated), rtol=1e-10, atol=1e-10)
    clades = replace(sequence, clades=torch.ones_like(sequence.clades))
    torch.testing.assert_close(net(sequence), net(clades), rtol=1e-10, atol=1e-10)


def test_declared_gcn_normalization_and_graphnorm():
    gcn = GSTNormalizedGCN(1, self_loop_weight=2.).double()
    with torch.no_grad(): gcn.linear.weight.fill_(1.)
    h = torch.tensor([[1.], [3.]], dtype=torch.float64)
    edges = torch.tensor([[0, 1], [1, 0]])
    torch.testing.assert_close(gcn(h, edges), torch.tensor([[5/3], [7/3]], dtype=torch.float64))
    loops = torch.tensor([[0, 1], [0, 1]])
    torch.testing.assert_close(gcn(h, edges), gcn(h, torch.cat((edges, loops), 1)))
    norm = GSTGraphNorm(1).double()
    graph = torch.tensor([0, 0, 1, 1])
    normalized = norm(torch.tensor([[1.], [3.], [21.], [23.]], dtype=torch.float64), graph, 2)
    torch.testing.assert_close(normalized[:2], normalized[2:])
    torch.testing.assert_close(normalized.reshape(2, 2).mean(1), torch.zeros(2, dtype=torch.float64))


def test_named_controls_and_paper_gate_initialization():
    record = GSTRecordTokenAdaptation(5, 3, hidden_dim=8, num_heads=2)
    span = GSTSpanTokenAdaptation(5, 3, hidden_dim=8, num_heads=2)
    assert record.control_name == "gst_ca_record"
    assert span.control_name == "gst_ca_span"
    assert list(record.state_dict()) == list(span.state_dict())
    assert sum(p.numel() for p in record.parameters()) == sum(p.numel() for p in span.parameters())
    for block in record.blocks:
        assert torch.count_nonzero(block.gate[-1].weight) == 0
        assert torch.count_nonzero(block.gate[-1].bias) == 0
    assert record.configuration_provenance()["status"] == "independent baseline adaptation, not a reproduction"
    with pytest.raises(TypeError, match="fixes"):
        GSTSpanTokenAdaptation(5, 3, token_weighting="record")


@pytest.mark.parametrize("routing", ["broadcast", "cross_attention"])
def test_one_graph_with_no_edges_is_supported(sequence, routing):
    one = replace(sequence, x=sequence.x[:3], edge_index=torch.empty(2, 0, dtype=torch.long),
                  node_graph=torch.zeros(3, dtype=torch.long), node_ids=sequence.node_ids[:3],
                  clades=sequence.clades[:3], intervals=torch.tensor([[0., 1.]], dtype=torch.float64),
                  target_graph=torch.zeros(4, dtype=torch.long))
    output = model("span", routing)(one)
    assert output.shape == (4, 2, 3) and torch.isfinite(output).all()


def test_invalid_configuration_and_cross_graph_edges_rejected(sequence):
    with pytest.raises(ValueError, match="divide"):
        GSTAdaptation(5, 3, hidden_dim=7, num_heads=2)
    with pytest.raises(ValueError, match="weighting"):
        GSTAdaptation(5, 3, token_weighting="midpoint")
    with pytest.raises(ValueError, match="routing"):
        GSTAdaptation(5, 3, routing="late")
    with pytest.raises(ValueError, match="boundaries"):
        model()(replace(sequence, edge_index=torch.tensor([[0], [3]])))

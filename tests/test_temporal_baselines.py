"""Behavioral checks for independent temporal baseline families, CPU only."""
from dataclasses import replace

import pytest
import torch

from arggnn.data import GraphSequence
from arggnn.temporal_baselines import (
    LeafCorrespondenceSupraMPNN,
    LeafTemporalTransformer,
    leaf_correspondence_edges,
)


def example_sequence() -> GraphSequence:
    intervals = torch.tensor([[0.0, 0.35], [0.35, 0.70], [0.70, 1.0]])
    node_ids = torch.tensor([10, 20, 30, -1, -1] * 3)
    node_graph = torch.arange(3).repeat_interleave(5)
    # Nontrivial whole-tree observed features, distinct across graph contents.
    base = torch.tensor([[1.0, 0.0, 0.2, 0.0, 0.4],
                         [0.0, 1.0, 0.1, 0.0, 0.6],
                         [1.0, 0.0, 0.3, 0.0, 0.8],
                         [0.0, 0.0, 0.7, 0.5, 0.1],
                         [0.0, 0.0, 1.0, 0.8, 0.2]])
    x = torch.cat([base + i * torch.tensor([0., 0., .8, -.3, .5]) for i in range(3)])
    edge = torch.tensor([[3, 3, 4, 4], [0, 1, 3, 2]])
    edge = torch.cat((edge, edge.flip(0)), dim=1)
    edges = torch.cat([edge + 5 * i for i in range(3)], dim=1)
    seq = GraphSequence(
        x=x, edge_index=edges, node_graph=node_graph, node_ids=node_ids,
        clades=torch.tensor([[0.], [1.], [0.], [1.], [1.]] * 3),
        intervals=intervals, query_node_ids=torch.tensor([10, 30]),
        positions=torch.tensor([0.08, 0.25, 0.54, 0.88]),
        target_graph=torch.tensor([0, 0, 1, 2]),
    )
    seq.validate()
    return seq


def model_for(family):
    torch.manual_seed(1729)
    return family(input_dim=5, num_classes=3, hidden_dim=16,
                  num_layers=2, dropout=0.0).eval()


def permute_nodes(seq):
    generator = torch.Generator().manual_seed(203)
    order = torch.randperm(seq.x.shape[0], generator=generator)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    return replace(seq, x=seq.x[order], edge_index=inverse[seq.edge_index],
                   node_graph=seq.node_graph[order], node_ids=seq.node_ids[order],
                   clades=seq.clades[order])


@pytest.mark.parametrize("family", [LeafTemporalTransformer, LeafCorrespondenceSupraMPNN])
def test_output_shape_and_training_gradients(family):
    seq = example_sequence()
    model = model_for(family)
    output = model(seq)
    assert output.shape == (4, 2, 3)
    assert torch.isfinite(output).all()
    target = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1]])
    torch.nn.functional.cross_entropy(output.reshape(-1, 3), target.reshape(-1)).backward()
    assert model.encoder.input_linear.weight.grad is not None
    assert torch.isfinite(model.encoder.input_linear.weight.grad).all()
    assert model.encoder.input_linear.weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("family", [LeafTemporalTransformer, LeafCorrespondenceSupraMPNN])
def test_node_index_and_stable_identity_relabeling(family):
    seq = example_sequence()
    model = model_for(family)
    with torch.no_grad():
        expected = model(seq)
        reordered = permute_nodes(seq)
        torch.testing.assert_close(model(reordered), expected, atol=2e-6, rtol=2e-6)
        renamed_ids = reordered.node_ids.clone()
        renamed_queries = reordered.query_node_ids.clone()
        for old, new in [(10, 95), (20, 7), (30, 31)]:
            renamed_ids[reordered.node_ids == old] = new
            renamed_queries[reordered.query_node_ids == old] = new
        renamed = replace(reordered, node_ids=renamed_ids, query_node_ids=renamed_queries)
        torch.testing.assert_close(model(renamed), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("family", [LeafTemporalTransformer, LeafCorrespondenceSupraMPNN])
def test_graph_serialization_is_not_genomic_order(family):
    seq = example_sequence()
    model = model_for(family)
    order = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(3)
    serialized = replace(seq, intervals=seq.intervals[order],
                         node_graph=inverse[seq.node_graph],
                         target_graph=inverse[seq.target_graph])
    serialized.validate()
    with torch.no_grad():
        expected = model(seq)
        torch.testing.assert_close(model(serialized), expected, atol=2e-6, rtol=2e-6)
        # Moving content to other fixed intervals changes the coordinate signal.
        changed_order = replace(seq, node_graph=inverse[seq.node_graph])
        changed_order.validate()
        assert (model(changed_order) - expected).abs().max() > 1e-5


@pytest.mark.parametrize("family", [LeafTemporalTransformer, LeafCorrespondenceSupraMPNN])
def test_fixed_position_and_query_ordering(family):
    seq = example_sequence()
    model = model_for(family)
    positions = torch.tensor([3, 1, 0, 2])
    queries = torch.tensor([1, 0])
    reordered = replace(seq, positions=seq.positions[positions],
                        target_graph=seq.target_graph[positions],
                        query_node_ids=seq.query_node_ids[queries])
    with torch.no_grad():
        expected = model(seq)[positions][:, queries]
        torch.testing.assert_close(model(reordered), expected, atol=2e-6, rtol=2e-6)


def test_correspondence_uses_all_stable_leaves_and_only_neighboring_graphs():
    seq = example_sequence()
    edges = leaf_correspondence_edges(seq)
    assert edges.shape == (2, 12)  # 3 leaves, 2 boundaries, both directions
    assert torch.equal(seq.node_ids[edges[0]], seq.node_ids[edges[1]])
    assert (seq.node_ids[edges] >= 0).all()
    assert ((seq.node_graph[edges[0]] - seq.node_graph[edges[1]]).abs() == 1).all()
    assert (seq.node_ids[edges] == 20).any()  # non-query reference leaf included


@pytest.mark.parametrize("family", [LeafTemporalTransformer, LeafCorrespondenceSupraMPNN])
def test_single_graph_without_temporal_edges(family):
    seq = example_sequence()
    one = replace(seq, x=seq.x[:5], edge_index=seq.edge_index[:, :8],
                  node_graph=seq.node_graph[:5], node_ids=seq.node_ids[:5],
                  clades=seq.clades[:5], intervals=torch.tensor([[0., 1.]]),
                  target_graph=torch.zeros_like(seq.target_graph))
    one.validate()
    assert leaf_correspondence_edges(one).shape == (2, 0)
    output = model_for(family)(one)
    assert output.shape == (4, 2, 3)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("family", [LeafTemporalTransformer, LeafCorrespondenceSupraMPNN])
def test_nonlocal_graph_context_reaches_fixed_local_query(family):
    seq = example_sequence()
    model = model_for(family)
    x = seq.x.clone().requires_grad_(True)
    observed = replace(seq, x=x)
    # Position zero belongs to graph zero. Graph two supplies remote context.
    model(observed)[0, 0, 0].backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad[seq.node_graph == 2].abs().max() > 1e-10


def test_missing_identity_is_not_bridged_across_an_intervening_graph():
    seq = example_sequence()
    ids = seq.node_ids.clone()
    ids[6] = -1  # identity 20 absent in the middle graph
    edges = leaf_correspondence_edges(replace(seq, node_ids=ids))
    assert not (ids[edges] == 20).any()
    assert edges.shape == (2, 8)  # remaining two leaves, two boundaries, both ways

"""CPU checks for ARGMix equation fidelity and the shared-input symmetries."""
from __future__ import annotations
import io
import math
import pytest
import torch

from arggnn.data import GraphSequence
from arggnn.argformer_control import (
    ARGMixTreeControl, WholeTreeGraphTransformerControl, RelativeTreeAttention,
    _ObservedTree, relative_tmrca_bins,
)


def example_sequence():
    generator = torch.Generator().manual_seed(99)
    n = 11
    x = torch.randn(2 * n, 48, dtype=torch.float64, generator=generator) * .1
    times = torch.tensor([0., 0., 20., 30., 40., 50., 100., 200., 300., 800., 1200.],
                         dtype=torch.float64)
    edges, clades = [], []
    topologies = [
        [(6, 0), (6, 2), (7, 1), (7, 3), (8, 4), (8, 5), (9, 6), (9, 7), (10, 9), (10, 8)],
        [(6, 0), (6, 3), (7, 1), (7, 2), (8, 4), (8, 5), (9, 6), (9, 7), (10, 9), (10, 8)],
    ]
    for graph, pairs in enumerate(topologies):
        offset = graph * n
        x[offset:offset+n, :8] = 0
        x[offset+2:offset+6, :4] = torch.eye(4, dtype=x.dtype)
        x[offset:offset+2, 4] = 1
        x[offset:offset+6, 5] = 1
        x[offset:offset+n, 6] = times.log1p() / math.log1p(10000.)
        desc = torch.zeros(n, 4, dtype=x.dtype)
        desc[2:6] = torch.eye(4, dtype=x.dtype)
        for parent, child in pairs:
            x[offset+child, 7] = torch.log1p(times[parent] - times[child]) / math.log1p(10000.)
            edges.extend([(offset+parent, offset+child), (offset+child, offset+parent)])
        for u in (6, 7, 8, 9, 10):
            for parent, child in pairs:
                if parent == u:
                    desc[u] += desc[child]
        clades.append(desc)
    return GraphSequence(
        x=x, edge_index=torch.tensor(edges).T.contiguous(),
        node_graph=torch.arange(2).repeat_interleave(n),
        node_ids=torch.tensor([100, 200, 300, 400, 500, 600, -1, -1, -1, -1, -1]).repeat(2),
        clades=torch.cat(clades), intervals=torch.tensor([[0., .3], [.3, 1.]], dtype=x.dtype),
        query_node_ids=torch.tensor([100, 200]), positions=torch.tensor([.1, .2, .4, .9], dtype=x.dtype),
        target_graph=torch.tensor([0, 0, 1, 1]), window_bounds=torch.tensor([0., 1.], dtype=x.dtype),
    )


def clone(seq, **updates):
    values = {k: v.clone() if v is not None else None for k, v in seq.as_dict().items()}
    values.update(updates)
    return GraphSequence(**values)


def model_for(kind, **updates):
    settings = dict(input_dim=48, num_classes=4, hidden_dim=8, num_layers=2,
                    num_heads=2, dropout=0., max_rel_distance=24)
    settings.update(updates)
    if kind == "hops":
        return WholeTreeGraphTransformerControl(**settings).double().eval()
    if kind == "subgraph":
        settings["token_scope"] = "nearest_reference"
        settings["num_references"] = 25
    return ARGMixTreeControl(**settings).double().eval()


@pytest.mark.parametrize("kind", ["tmrca", "hops", "subgraph"])
def test_shape_all_observed_features_and_gradient(kind):
    torch.manual_seed(18)
    torch.set_num_threads(1)
    seq = example_sequence()
    seq.validate()
    seq.x.requires_grad_(True)
    model = model_for(kind)
    out = model(seq)
    assert out.shape == (4, 2, 4)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head.weight.grad.abs().sum() > 0
    assert seq.x.grad is not None and torch.isfinite(seq.x.grad).all()
    assert bool((seq.x.grad.abs().sum(0) > 0).all())
    # Internal nodes and labeled references both contribute to the classifier.
    assert seq.x.grad[seq.node_ids < 0].abs().sum() > 0
    assert seq.x.grad[seq.node_ids >= 300].abs().sum() > 0


@pytest.mark.parametrize("kind", ["tmrca", "hops", "subgraph"])
def test_node_relabeling_queries_serialization_and_clades(kind):
    torch.manual_seed(21)
    seq = example_sequence()
    model = model_for(kind)
    expected = model(seq)
    permutation = torch.randperm(len(seq.x))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    permuted = clone(seq, x=seq.x[permutation], edge_index=inverse[seq.edge_index],
                     node_graph=seq.node_graph[permutation], node_ids=seq.node_ids[permutation],
                     clades=seq.clades[permutation])
    # Arbitrary, nonmonotone ID renaming leaves biological observations unchanged.
    old, new = [100, 200, 300, 400, 500, 600], [999, 17, 32, 880, 27, 66]
    ids = seq.node_ids.clone()
    for a, b in zip(old, new):
        ids[seq.node_ids == a] = b
    renamed = clone(seq, node_ids=ids, query_node_ids=torch.tensor(new[:2]))
    graph_swapped = clone(seq, node_graph=1-seq.node_graph,
                          intervals=seq.intervals.flip(0), target_graph=1-seq.target_graph)
    for alternative in (permuted, renamed, graph_swapped, clone(seq, clades=seq.clades.flip(1))):
        torch.testing.assert_close(model(alternative), expected, atol=1e-10, rtol=1e-10)
    reordered_queries = clone(seq, query_node_ids=seq.query_node_ids.flip(0))
    torch.testing.assert_close(model(reordered_queries), expected.flip(1), atol=1e-10, rtol=1e-10)
    stream = io.BytesIO()
    torch.save(seq.as_dict(), stream)
    stream.seek(0)
    restored = GraphSequence.from_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(model(restored), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("kind", ["tmrca", "hops", "subgraph"])
def test_local_tree_classification_has_no_genomic_aggregation(kind):
    torch.manual_seed(31)
    seq = example_sequence()
    model = model_for(kind)
    shifted = clone(seq, positions=torch.tensor([.05, .29, .31, .8], dtype=seq.x.dtype))
    torch.testing.assert_close(model(seq), model(shifted), atol=1e-10, rtol=1e-10)
    out = model(seq)
    torch.testing.assert_close(out[0], out[1])
    torch.testing.assert_close(out[2], out[3])


def test_attention_matches_paper_column_key_and_relative_values():
    torch.manual_seed(81)
    attention = RelativeTreeAttention(8, 2).double()
    h = torch.randn(5, 8, dtype=torch.float64)
    relations = torch.tensor([[0, 1, 2, 1, 2], [2, 0, 1, 2, 1], [1, 2, 0, 1, 2],
                              [2, 1, 2, 0, 1], [1, 2, 1, 2, 0]])
    tables = [torch.randn(3, 8, dtype=h.dtype, requires_grad=True) for _ in range(3)]
    q, k, v = [p(h).reshape(5, 2, 4).transpose(0, 1)
               for p in (attention.q, attention.k, attention.v)]
    qr, kr, vr = [t[relations].reshape(5, 5, 2, 4).permute(2, 0, 1, 3) for t in tables]
    score = ((q[:, :, None] * k[:, None]).sum(-1)
             + (q[:, :, None] * qr).sum(-1) + (k[:, None] * kr).sum(-1)) / 2
    weight = score.softmax(-1)
    expected = (weight[..., None] * (v[:, None] + vr)).sum(2)
    expected = attention.output(expected.transpose(0, 1).reshape(5, 8))
    torch.testing.assert_close(attention(h, relations, *tables), expected, atol=1e-12, rtol=1e-12)
    # The official current implementation's row-key term differs on asymmetric relations.
    row_score = ((q[:, :, None] * k[:, None]).sum(-1)
                 + (q[:, :, None] * qr).sum(-1) + (k[:, :, None] * kr).sum(-1)) / 2
    assert (score - row_score).abs().max() > .1


def test_relative_tmrca_is_asymmetric_self_zero_and_clamped():
    tree = _ObservedTree(torch.arange(3), torch.tensor([2, 2, -1]),
                         torch.tensor([[0, 2, 2], [2, 1, 2], [2, 2, 2]]),
                         torch.tensor([[0, 2, 1], [2, 0, 1], [1, 1, 0]]),
                         torch.tensor([0., 20., 120.], dtype=torch.float64))
    torch.testing.assert_close(relative_tmrca_bins(tree, 100),
                               torch.tensor([[0, 100, 100], [100, 0, 100], [0, 0, 0]]))
    relation = relative_tmrca_bins(tree, 200)
    assert relation[0, 1] == 120 and relation[1, 0] == 100


def test_whole_tree_control_uses_structure_and_every_node_without_id_embedding():
    torch.manual_seed(42)
    seq = example_sequence()
    model = model_for("hops")
    changed = seq.edge_index.clone()
    # Exchange two leaf attachments in only the first tree, preserving node features/times.
    for a, b in [(0, 1), (1, 0)]:
        changed[(seq.edge_index == a)] = b
    different = clone(seq, edge_index=changed)
    assert (model(seq) - model(different)).abs().max() > 1e-7
    assert model.configuration()["learned_node_id_embeddings"] is False
    assert model.configuration()["cross_tree_aggregation"] == "none"


def test_nearest_reference_ties_and_actual_subgraph_size():
    seq = example_sequence()
    model = model_for("subgraph")
    tree = model._geometry(seq)[0]
    selected = model._subgraph(seq, tree, 0)
    # Query + four available references + three ancestral vertices.
    assert len(selected) == 8
    assert 1 not in selected  # the other query is not a reference
    assert set(selected.tolist()) == {0, 2, 3, 4, 5, 6, 9, 10}
    model.num_references = 1
    # In this tree refs4,5 tie at the root for query0, while ref2 is nearer.
    assert set(model._subgraph(seq, tree, 0).tolist()) == {0, 2, 6, 9, 10}
    tied = _ObservedTree(tree.nodes, tree.parent, tree.mrca, tree.hops, tree.times.clone())
    tied.times[6] = tied.times[9]
    assert set(model._subgraph(seq, tied, 0).tolist()) == {0, 2, 3, 6, 9, 10}
    leakage = clone(seq)
    leakage.x[0, 0] = 1
    with pytest.raises(ValueError, match="Queried ancestry"):
        model(leakage)


def test_geometry_cache_tracks_inplace_mutation_and_checkpoint_roundtrip():
    torch.manual_seed(58)
    seq = example_sequence()
    model = model_for("tmrca")
    original = model._geometry(seq)
    seq.x[:, 6] *= .9
    assert model._geometry(seq) is not original
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = model_for("tmrca")
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(model(seq), restored(seq))
    model.clear_geometry_cache()
    assert len(model._geometry_cache) == 0


@pytest.mark.parametrize("change", ["cycle", "time", "width"])
def test_invalid_observed_input_fails(change):
    seq = example_sequence()
    model = model_for("tmrca")
    if change == "cycle":
        seq.edge_index = torch.cat((seq.edge_index, torch.tensor([[0, 1], [1, 0]])), 1)
    elif change == "time":
        seq.x[0, 6] = 1.9
    else:
        seq.x = seq.x[:, :-1]
    with pytest.raises(ValueError):
        model(seq)

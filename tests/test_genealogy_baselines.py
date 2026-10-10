"""Exact observed-tree reference summaries, independent pairwise cross-checks."""
import unittest
import math
import torch
from arggnn.data import GraphSequence
from arggnn.genealogy_baselines import (ReferenceTMRCAFeatures, NearestTMRCAClassifier,
    TMRCAFeatureMLP, pairwise_reference_tmrca, reference_hop_diagnostics)


def example():
    x = torch.zeros(7, 6, dtype=torch.float64)
    x[2, 0], x[3, 1] = 1, 1
    x[:2, 2], x[:4, 3] = 1, 1
    times = torch.tensor([0, 0, 0, 0, 10, 20, 50], dtype=torch.float64)
    branches = torch.tensor([10, 10, 20, 50, 10, 30, 0], dtype=torch.float64)
    x[:, 4] = times.log1p() / math.log1p(10000)
    x[:, 5] = branches.log1p() / math.log1p(10000)
    edges = torch.tensor([[4, 0], [4, 1], [5, 4], [5, 2], [6, 5], [6, 3]]).T
    return GraphSequence(x, torch.cat((edges, edges.flip(0)), dim=1), torch.zeros(7, dtype=torch.long),
                         torch.tensor([0, 1, 2, 3, -1, -1, -1]), torch.zeros(7, 1, dtype=torch.float64),
                         torch.tensor([[0., 1.]], dtype=torch.float64), torch.tensor([0, 1]),
                         torch.tensor([.15, .45, .8], dtype=torch.float64), torch.zeros(3, dtype=torch.long))


def changed(seq, **updates):
    values = seq.as_dict()
    values = {k: None if v is None else v.clone() for k, v in values.items()}
    values.update(updates)
    return GraphSequence(**values)


class GenealogyBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.seq = example()
        self.kwargs = dict(reference_columns=(0, 1), time_column=4, branch_column=5)

    def test_exact_tmrca_and_deep_reference_reachability(self):
        summary = ReferenceTMRCAFeatures(**self.kwargs).summarize(self.seq)
        expected = torch.tensor([[[20., 50.], [20., 50.]]], dtype=torch.float64)
        torch.testing.assert_close(summary.minimum_tmrca, expected)
        torch.testing.assert_close(summary.nearest_votes, torch.tensor([[[1., 0.], [1., 0.]]], dtype=torch.float64))
        diagnostic = reference_hop_diagnostics(self.seq, **self.kwargs, num_layers=2)
        self.assertEqual(diagnostic['fraction_with_any_reference_in_receptive_field'], 0)
        self.assertEqual(diagnostic['minimum_reference_hops_histogram'], {'3': 6})

    def test_pairwise_cross_check_all_reference_information(self):
        record = pairwise_reference_tmrca(self.seq, **self.kwargs)[0]
        torch.testing.assert_close(record['tmrca'], torch.tensor([[20., 50.], [20., 50.]], dtype=torch.float64))
        torch.testing.assert_close(record['hop_distances'], torch.tensor([[3, 4], [3, 4]]))
        self.assertEqual(record['reference_classes'].tolist(), [0, 1])
        features = ReferenceTMRCAFeatures(**self.kwargs).summarize(self.seq).features
        # One reference per source makes mean/min logged time identical.
        torch.testing.assert_close(features[..., 0], features[..., 1])
        self.assertEqual(features.shape, (1, 2, 2, 9))

    def test_shared_nearest_ties(self):
        edges = torch.tensor([[4, 0], [4, 1], [5, 4], [5, 2], [6, 5], [5, 3]]).T
        tied = changed(self.seq, edge_index=torch.cat((edges, edges.flip(0)), dim=1))
        classifier = NearestTMRCAClassifier(6, 2, **self.kwargs)
        torch.testing.assert_close(classifier.probabilities(tied), torch.full((3, 2, 2), .5, dtype=torch.float64))

    def test_node_identity_graph_and_query_permutations(self):
        permutation = torch.tensor([6, 0, 4, 2, 5, 1, 3])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(7)
        seq = self.seq
        permuted = changed(seq, x=seq.x[permutation], edge_index=inverse[seq.edge_index],
                           node_graph=seq.node_graph[permutation], node_ids=seq.node_ids[permutation],
                           clades=seq.clades[permutation])
        renamed = changed(seq, node_ids=torch.where(seq.node_ids >= 0, 100-seq.node_ids, seq.node_ids),
                          query_node_ids=100-seq.query_node_ids)
        classifier = NearestTMRCAClassifier(6, 2, **self.kwargs)
        extractor = ReferenceTMRCAFeatures(**self.kwargs)
        for case in (permuted, renamed):
            torch.testing.assert_close(extractor(seq), extractor(case))
            torch.testing.assert_close(classifier(seq), classifier(case))
        swapped = changed(seq, query_node_ids=seq.query_node_ids.flip(0))
        torch.testing.assert_close(extractor(seq).flip(1), extractor(swapped))

    def test_refinement_invariance(self):
        seq = self.seq
        split = changed(seq, x=seq.x.repeat(2, 1), edge_index=torch.cat((seq.edge_index, seq.edge_index + 7), dim=1),
                        node_graph=torch.tensor([0]*7+[1]*7), node_ids=seq.node_ids.repeat(2), clades=seq.clades.repeat(2, 1),
                        intervals=torch.tensor([[0., .3], [.3, 1.]], dtype=torch.float64),
                        target_graph=torch.tensor([0, 1, 1]))
        model = TMRCAFeatureMLP(6, 2, **self.kwargs, hidden_dim=8).double()
        torch.testing.assert_close(model(seq), model(split))
        reverse_graphs = changed(split, node_graph=1-split.node_graph, intervals=split.intervals.flip(0),
                                 target_graph=1-split.target_graph)
        torch.testing.assert_close(model(split), model(reverse_graphs))

    def test_mlp_shapes_prior_and_gradients(self):
        model = TMRCAFeatureMLP(6, 2, **self.kwargs, hidden_dim=8).double()
        logits = model(self.seq)
        self.assertEqual(logits.shape, (3, 2, 2))
        prior = NearestTMRCAClassifier(6, 2, **self.kwargs)(self.seq)
        torch.testing.assert_close(logits, prior)
        labels = torch.tensor([[0, 1], [1, 0], [0, 1]])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), labels.flatten())
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.head.parameters()))
        features, base = model.features_and_prior(self.seq)
        torch.testing.assert_close(model.forward_from_features(features, base), logits)
        no_prior = TMRCAFeatureMLP(6, 2, **self.kwargs, hidden_dim=8, nearest_init=False).double()
        self.assertTrue(bool(torch.isfinite(no_prior(self.seq)).all()))

    def test_query_label_leakage_and_bad_topology_rejected(self):
        labeled = self.seq.x.clone()
        labeled[0, 0] = 1
        internal = self.seq.x.clone()
        internal[5, 0] = 1
        model = ReferenceTMRCAFeatures(**self.kwargs)
        cases = [changed(self.seq, x=labeled), changed(self.seq, x=internal),
                 changed(self.seq, edge_index=self.seq.edge_index[:, ~(((self.seq.edge_index[0] == 4) & (self.seq.edge_index[1] == 0)) | ((self.seq.edge_index[0] == 0) & (self.seq.edge_index[1] == 4)))]),
                 changed(self.seq, edge_index=torch.cat((self.seq.edge_index, torch.tensor([[0], [3]])), dim=1))]
        for case in cases:
            with self.assertRaises(ValueError):
                model(case)

    def test_two_hop_encoder_cannot_observe_deeper_reference_label_changes(self):
        from arggnn.models import IndependentGNN
        swapped_features = self.seq.x.clone()
        swapped_features[:, :2] = swapped_features[:, :2].flip(1)
        swapped = changed(self.seq, x=swapped_features)
        torch.manual_seed(21)
        shallow = IndependentGNN(6, 2, hidden_dim=8, num_layers=2).double().eval()
        torch.testing.assert_close(shallow(self.seq), shallow(swapped), rtol=0, atol=0)
        exact = NearestTMRCAClassifier(6, 2, **self.kwargs)
        torch.testing.assert_close(exact.probabilities(self.seq).flip(-1), exact.probabilities(swapped))

    def test_one_direction_edges_are_equivalent(self):
        directed = changed(self.seq, edge_index=self.seq.edge_index[:, :6])
        extractor = ReferenceTMRCAFeatures(**self.kwargs)
        torch.testing.assert_close(extractor(self.seq), extractor(directed))


if __name__ == '__main__':
    unittest.main()

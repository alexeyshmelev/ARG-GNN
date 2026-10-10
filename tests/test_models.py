"""Meaningful CPU contracts and representation-symmetry checks."""
import unittest
from types import SimpleNamespace
import torch
from arggnn.models import (build_model, count_parameters,
                           log_interval_kernel_mass, validate_model_input)


def example_sequence():
    generator = torch.Generator().manual_seed(81)
    return SimpleNamespace(
        x=torch.randn(7, 3, generator=generator, dtype=torch.float64),
        edge_index=torch.tensor([[0, 2, 1, 2, 3, 6, 4, 6, 5, 6],
                                 [2, 0, 2, 1, 6, 3, 6, 4, 6, 5]]),
        node_graph=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        node_ids=torch.tensor([0, 1, -1, 0, 1, 2, -1]),
        clades=torch.tensor([[0, 0], [1, 0], [1, 0], [0, 0],
                             [1, 0], [0, 1], [1, 1]], dtype=torch.float64),
        intervals=torch.tensor([[0.0, 0.4], [0.4, 1.0]], dtype=torch.float64),
        query_node_ids=torch.tensor([0, 1]),
        positions=torch.tensor([0.1, 0.25, 0.65, 0.9], dtype=torch.float64),
        target_graph=torch.tensor([0, 0, 1, 1]),
        window_bounds=torch.tensor([0.0, 1.0], dtype=torch.float64),
    )


def clone(seq, **updates):
    fields = {k: v.clone() for k, v in vars(seq).items()}
    fields.update(updates)
    return SimpleNamespace(**fields)


def split_first_interval(seq):
    first_edges = seq.edge_index[:, seq.node_graph[seq.edge_index[0]] == 0]
    second_edges = seq.edge_index[:, seq.node_graph[seq.edge_index[0]] == 1] + 3
    return clone(seq,
        x=torch.cat((seq.x[:3], seq.x[:3], seq.x[3:])),
        edge_index=torch.cat((first_edges, first_edges + 3, second_edges), dim=1),
        node_graph=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
        node_ids=torch.cat((seq.node_ids[:3], seq.node_ids[:3], seq.node_ids[3:])),
        clades=torch.cat((seq.clades[:3], seq.clades[:3], seq.clades[3:])),
        intervals=torch.tensor([[0.0, 0.15], [0.15, 0.4], [0.4, 1.0]], dtype=torch.float64),
        target_graph=torch.tensor([0, 1, 2, 2]),
    )


class ModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.seq = example_sequence()

    def model(self, name):
        return build_model(name, 3, 3, hidden_dim=8, num_layers=2).double().eval()

    def test_all_shapes_finite_and_gradient(self):
        names = ('independent', 'set_pool', 'gru', 'interval_lineage',
                 'no_correspondence', 'midpoint_weights', 'equal_tree_weights', 'late_fusion')
        for name in names:
            with self.subTest(name=name):
                model = self.model(name)
                logits = model(self.seq)
                self.assertEqual(logits.shape, (4, 2, 3))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                logits.square().mean().backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertGreater(count_parameters(model), 0)

    def test_node_permutation_and_identity_renaming(self):
        permutation = torch.tensor([6, 0, 3, 1, 5, 2, 4])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        permuted = clone(self.seq, x=self.seq.x[permutation],
                         edge_index=inverse[self.seq.edge_index],
                         node_graph=self.seq.node_graph[permutation],
                         node_ids=self.seq.node_ids[permutation], clades=self.seq.clades[permutation])
        renamed = clone(self.seq,
                        node_ids=torch.where(self.seq.node_ids >= 0, self.seq.node_ids + 100, self.seq.node_ids),
                        query_node_ids=self.seq.query_node_ids + 100)
        for name in ('independent', 'set_pool', 'gru', 'interval_lineage'):
            model = self.model(name)
            torch.testing.assert_close(model(self.seq), model(permuted), rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(model(self.seq), model(renamed), rtol=1e-10, atol=1e-10)

    def test_graph_serialization_and_query_permutation(self):
        swapped = clone(self.seq, node_graph=1 - self.seq.node_graph,
                        intervals=self.seq.intervals.flip(0), target_graph=1 - self.seq.target_graph)
        query_swapped = clone(self.seq, query_node_ids=self.seq.query_node_ids.flip(0))
        for name in ('independent', 'set_pool', 'gru', 'interval_lineage'):
            model = self.model(name)
            torch.testing.assert_close(model(self.seq), model(swapped), rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(model(self.seq).flip(1), model(query_swapped), rtol=1e-10, atol=1e-10)

    def test_refinement_invariance_at_common_positions(self):
        split = split_first_interval(self.seq)
        for name in ('independent', 'set_pool', 'interval_lineage', 'no_correspondence', 'late_fusion'):
            model = self.model(name)
            torch.testing.assert_close(model(self.seq), model(split), rtol=1e-10, atol=1e-10)
        for name in ('midpoint_weights', 'equal_tree_weights'):
            model = self.model(name)
            self.assertGreater(float((model(self.seq) - model(split)).abs().max()), 1e-7)

    def test_reversal_landmark_permutation_and_translation(self):
        model = self.model('interval_lineage')
        reflected = clone(self.seq,
                          intervals=torch.stack((1 - self.seq.intervals[:, 1], 1 - self.seq.intervals[:, 0]), dim=1),
                          positions=1 - self.seq.positions)
        landmarks = clone(self.seq, clades=self.seq.clades.flip(1))
        torch.testing.assert_close(model(self.seq), model(reflected), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(model(self.seq), model(landmarks), rtol=1e-10, atol=1e-10)
        tile = clone(self.seq, intervals=self.seq.intervals * 0.2,
                     positions=self.seq.positions * 0.2,
                     window_bounds=torch.tensor([0.0, 0.2], dtype=torch.float64))
        translated = clone(tile, intervals=tile.intervals + 0.3,
                           positions=tile.positions + 0.3,
                           window_bounds=torch.tensor([0.3, 0.5], dtype=torch.float64))
        torch.testing.assert_close(model(tile), model(translated), rtol=1e-10, atol=1e-10)

    def test_kernel_partition_numerics_and_gradient(self):
        positions = torch.tensor([0.0, 0.2, 0.7, 0.999], dtype=torch.float64)
        tau = torch.tensor(0.003, dtype=torch.float64, requires_grad=True)
        whole = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        pieces = torch.tensor([[0.0, 0.2], [0.2, 0.200000000001], [0.200000000001, 1.0]], dtype=torch.float64)
        a = log_interval_kernel_mass(positions, whole, tau)
        b = log_interval_kernel_mass(positions, pieces, tau)
        torch.testing.assert_close(a[:, 0], torch.logsumexp(b, dim=1), rtol=1e-12, atol=1e-12)
        self.assertTrue(bool(torch.isfinite(b).all()))
        b.sum().backward()
        self.assertTrue(bool(torch.isfinite(tau.grad)))
        with self.assertRaises(ValueError):
            log_interval_kernel_mass(positions, whole, torch.tensor(0.0))

    def test_invalid_inputs_fail(self):
        failures = [
            {'intervals': torch.tensor([[0.1, 0.4], [0.4, 1.0]], dtype=torch.float64)},
            {'intervals': torch.tensor([[0.0, 0.4], [0.5, 1.0]], dtype=torch.float64)},
            {'intervals': torch.tensor([[0.0, 0.5], [0.4, 1.0]], dtype=torch.float64)},
            {'intervals': torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)},
            {'query_node_ids': torch.tensor([9])},
            {'positions': torch.tensor([0.1, 0.25, 0.65, 1.0], dtype=torch.float64)},
            {'target_graph': torch.tensor([1, 0, 1, 1])},
            {'edge_index': torch.tensor([[0], [3]])},
        ]
        for updates in failures:
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    validate_model_input(clone(self.seq, **updates))

    def test_fixed_global_coordinate_tile(self):
        tile = clone(self.seq, intervals=self.seq.intervals * 0.2 + 0.3,
                     positions=self.seq.positions * 0.2 + 0.3,
                     window_bounds=torch.tensor([0.3, 0.5], dtype=torch.float64))
        validate_model_input(tile)
        self.assertEqual(self.model('interval_lineage')(tile).shape, (4, 2, 3))

    def test_empty_edges_allowed(self):
        empty = clone(self.seq, edge_index=torch.empty(2, 0, dtype=torch.long))
        self.assertEqual(self.model('interval_lineage')(empty).shape, (4, 2, 3))


if __name__ == '__main__':
    unittest.main()

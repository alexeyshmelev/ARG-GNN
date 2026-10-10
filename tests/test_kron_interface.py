"""Static Kron identities, observed subdivision contracts and coupled memory."""
import math
import io
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
import pytest
import torch

from arggnn.kron_interface import (
    canonical_resistance_edges, resistance_laplacian, observed_branch_laplacian,
    kron_reduce, precompute_leaf_interfaces, query_reference_partition,
    coupled_query_dynamics, coupled_affine_transition, LeafInterfaceBank,
    precompute_graph_interfaces, with_query_reference_cache,
)


def star(resistance=(1., 2., 3.)):
    edges = torch.tensor([[0, 1, 2], [3, 3, 3]])
    r = torch.tensor(resistance, dtype=torch.float64)
    return resistance_laplacian(4, edges, r)


def observed_example(subdivided=False):
    times = torch.tensor([0., .2, .4, 2.] + ([.7] if subdivided else []), dtype=torch.float64)
    parent_branch = torch.tensor([2., 1.8, 1.6, 0.] + ([1.3] if subdivided else []), dtype=torch.float64)
    if subdivided:
        parent_branch[0] = .7
        edges = torch.tensor([[0, 4, 1, 2], [4, 3, 3, 3]])
    else:
        edges = torch.tensor([[0, 1, 2], [3, 3, 3]])
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    x = torch.stack((torch.tensor([0., 0., 1., 0.] + ([0.] if subdivided else []), dtype=torch.float64),
        torch.tensor([1., 1., 1., 0.] + ([0.] if subdivided else []), dtype=torch.float64),
        torch.log1p(times) / math.log1p(10000.), torch.log1p(parent_branch) / math.log1p(10000.)), dim=1)
    n = times.numel()
    return SimpleNamespace(x=x, edge_index=edges, node_graph=torch.zeros(n, dtype=torch.long),
        node_ids=torch.tensor([10, 20, 30, -1] + ([-1] if subdivided else [])),
        clades=torch.zeros(n, 1, dtype=torch.float64), intervals=torch.tensor([[0., 1.]], dtype=torch.float64),
        query_node_ids=torch.tensor([10, 20]), positions=torch.tensor([.2, .7], dtype=torch.float64),
        target_graph=torch.zeros(2, dtype=torch.long), window_bounds=torch.tensor([0., 1.], dtype=torch.float64))


class KronInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)

    def test_bidirected_deduplication_and_analytic_star_response(self):
        edges = torch.tensor([[0, 1, 2], [3, 3, 3]])
        r = torch.tensor([1., 2., 3.], dtype=torch.float64)
        doubled = torch.cat((edges, edges.flip(0)), dim=1)
        laplacian = resistance_laplacian(4, doubled, r.repeat(2))
        torch.testing.assert_close(laplacian, star())
        reduction = kron_reduce(laplacian, torch.tensor([0, 1, 2]))
        k = r.reciprocal()
        expected = torch.diag(k) - k[:, None] * k[None, :] / k.sum()
        torch.testing.assert_close(reduction.response, expected, rtol=1e-13, atol=1e-13)
        self.assertLess(reduction.diagnostics.relative_solve_residual, 1e-14)

    def test_series_unary_subdivision(self):
        original = kron_reduce(star(), torch.tensor([0, 1, 2]))
        edges = torch.tensor([[0, 4, 1, 2], [4, 3, 3, 3]])
        subdivided = resistance_laplacian(5, edges, torch.tensor([.25, .75, 2., 3.], dtype=torch.float64))
        after = kron_reduce(subdivided, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(original.response, after.response, rtol=1e-13, atol=1e-13)
        for scale in (.1, 7.):
            times = torch.tensor([0., 0., 0., 1., .25], dtype=torch.float64)
            direct = observed_branch_laplacian(times[:4], torch.tensor([[0, 1, 2], [3, 3, 3]]), resistance_scale=scale)
            split = observed_branch_laplacian(times, edges, resistance_scale=scale)
            torch.testing.assert_close(kron_reduce(direct, torch.tensor([0, 1, 2])).response,
                kron_reduce(split, torch.tensor([0, 1, 2])).response, rtol=1e-12, atol=1e-12)

    def test_node_relabeling_and_boundary_order(self):
        laplacian = star()
        original = kron_reduce(laplacian, torch.tensor([2, 0, 1]))
        permutation = torch.tensor([3, 1, 0, 2])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(4)
        changed = kron_reduce(laplacian[permutation[:, None], permutation[None, :]], inverse[original.boundary_nodes])
        torch.testing.assert_close(changed.response, original.response, rtol=1e-13, atol=1e-13)
        values = torch.randn(3, 2, dtype=torch.float64)
        torch.testing.assert_close(changed.harmonic_lift(values)[inverse], original.harmonic_lift(values))

    def test_boundary_current_energy_and_harmonic_minimizer(self):
        edges = torch.tensor([[0, 1, 3, 2], [3, 3, 4, 4]])
        laplacian = resistance_laplacian(5, edges, torch.tensor([.3, .7, 1.2, .5], dtype=torch.float64))
        reduction = kron_reduce(laplacian, torch.tensor([0, 1, 2]))
        values = torch.randn(3, 4, dtype=torch.float64)
        lifted = reduction.harmonic_lift(values)
        currents = laplacian @ lifted
        torch.testing.assert_close(currents[reduction.interior_nodes], torch.zeros(2, 4, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(currents[reduction.boundary_nodes], reduction.boundary_current(values), atol=1e-12, rtol=1e-12)
        original_energy = (lifted * (laplacian @ lifted)).sum()
        boundary_energy = (values * reduction.boundary_current(values)).sum()
        torch.testing.assert_close(original_energy, boundary_energy, atol=1e-12, rtol=1e-12)
        disturbed = lifted.clone()
        disturbed[reduction.interior_nodes] += torch.randn(2, 4, dtype=torch.float64)
        self.assertGreater(float((disturbed * (laplacian @ disturbed)).sum()), float(original_energy))

    def test_all_nodes_boundary_is_original_geographic_graph(self):
        edges = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
        laplacian = resistance_laplacian(4, edges, torch.tensor([1., 2., 3., 4., 2.], dtype=torch.float64))
        boundary = torch.tensor([3, 1, 2, 0])
        reduction = kron_reduce(laplacian, boundary)
        torch.testing.assert_close(reduction.response, laplacian[boundary[:, None], boundary[None, :]], rtol=0, atol=0)
        self.assertEqual(reduction.interior_nodes.numel(), 0)
        singleton = kron_reduce(torch.zeros(1, 1, dtype=torch.float64), torch.tensor([0]))
        self.assertEqual(float(singleton.response), 0.)

    def test_positive_resistance_and_simple_edge_assumptions(self):
        edges = torch.tensor([[0], [1]])
        for value in (0., -1., float('inf'), 1e-320):
            with self.assertRaises(ValueError):
                resistance_laplacian(2, edges, torch.tensor([value], dtype=torch.float64))
        with self.assertRaises(ValueError):
            canonical_resistance_edges(2, torch.tensor([[0, 1], [1, 0]]), torch.tensor([1., 2.]))
        with self.assertRaises(ValueError):
            observed_branch_laplacian(torch.zeros(2), edges)
        with self.assertRaises(ValueError):
            resistance_laplacian(1, torch.tensor([[0], [0]]), torch.ones(1))

    def test_conditioning_and_unanchored_components(self):
        disconnected = resistance_laplacian(4, torch.tensor([[0, 2], [1, 3]]), torch.ones(2))
        with self.assertRaisesRegex(ValueError, 'not positive definite'):
            kron_reduce(disconnected, torch.tensor([0]))
        ill = resistance_laplacian(4, torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([1., 1e7, 1e7], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, 'condition number'):
            kron_reduce(ill, torch.tensor([0, 3]), max_condition=1e5)
        accepted = kron_reduce(ill, torch.tensor([0, 3]), max_condition=1e8)
        self.assertGreater(accepted.diagnostics.interior_condition_number, 1e6)
        self.assertLess(accepted.diagnostics.relative_solve_residual, 1e-12)
        nonlaplacian = torch.eye(2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'zero-row-sum'):
            kron_reduce(nonlaplacian, torch.tensor([0, 1]))

    def test_leaf_feature_branch_exclusion_and_unary_consistency(self):
        original, split = observed_example(), observed_example(True)
        self.assertNotEqual(float(original.x[0, 3]), float(split.x[0, 3]))
        before = precompute_leaf_interfaces(original, time_column=2, parent_branch_column=3)
        after = precompute_leaf_interfaces(split, time_column=2, parent_branch_column=3)
        self.assertEqual(before.feature_columns, (0, 1, 2))
        torch.testing.assert_close(before.features, after.features, rtol=0, atol=0)
        torch.testing.assert_close(before.response, after.response, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(before.sample_ids, torch.tensor([10, 20, 30]))
        permutation = torch.tensor([4, 1, 3, 0, 2])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(5)
        permuted = SimpleNamespace(**{name: value.clone() for name, value in vars(split).items()})
        for name in ('x', 'node_ids', 'node_graph', 'clades'):
            setattr(permuted, name, getattr(split, name)[permutation])
        permuted.edge_index = inverse[split.edge_index]
        relabeled = precompute_leaf_interfaces(permuted, time_column=2, parent_branch_column=3)
        torch.testing.assert_close(after.response, relabeled.response, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(after.features, relabeled.features, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'exclude'):
            precompute_leaf_interfaces(original, time_column=2, parent_branch_column=3, feature_columns=[0, 3])

    def test_variable_tree_sizes_in_a_persistent_leaf_bank(self):
        original, split = observed_example(), observed_example(True)
        joined = SimpleNamespace(x=torch.cat((original.x, split.x)),
            edge_index=torch.cat((original.edge_index, split.edge_index+4), dim=1),
            node_graph=torch.tensor([0]*4+[1]*5), node_ids=torch.cat((original.node_ids, split.node_ids)),
            clades=torch.zeros(9, 1, dtype=torch.float64), intervals=torch.tensor([[0., .4], [.4, 1.]], dtype=torch.float64),
            query_node_ids=torch.tensor([10, 20]), positions=torch.tensor([.2, .7], dtype=torch.float64),
            target_graph=torch.tensor([0, 1]), window_bounds=torch.tensor([0., 1.], dtype=torch.float64))
        bank = precompute_leaf_interfaces(joined, time_column=2, parent_branch_column=3)
        self.assertEqual(bank.response.shape, (2, 3, 3))
        torch.testing.assert_close(bank.response[0], bank.response[1], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(bank.features[0], bank.features[1], rtol=0, atol=0)
        joined.node_ids[6] = 40
        with self.assertRaisesRegex(ValueError, 'sample set'):
            precompute_leaf_interfaces(joined, time_column=2, parent_branch_column=3)

    def test_query_cache_normalization_eigensystems_and_unary_invariance(self):
        original = precompute_leaf_interfaces(observed_example(), time_column=2, parent_branch_column=3)
        split = precompute_leaf_interfaces(observed_example(True), time_column=2, parent_branch_column=3)
        before = with_query_reference_cache(original, torch.tensor([30]))
        after = with_query_reference_cache(split, torch.tensor([30]))
        cache = before.query_cache
        torch.testing.assert_close(cache.normalized_response, after.query_cache.normalized_response, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cache.cqq, after.query_cache.cqq, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cache.cqr, after.query_cache.cqr, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cache.normalized_response.diagonal(dim1=-2, dim2=-1).mean(dim=-1), torch.ones(1, dtype=torch.float64))
        reconstructed = cache.eigenvectors @ torch.diag_embed(cache.eigenvalues) @ cache.eigenvectors.transpose(-1, -2)
        torch.testing.assert_close(reconstructed, cache.cqq, rtol=1e-12, atol=1e-12)
        bag_reconstructed = cache.bag_eigenvectors @ torch.diag(cache.bag_eigenvalues) @ cache.bag_eigenvectors.T
        torch.testing.assert_close(bag_reconstructed, cache.bag_cqq, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cache.bag_cqq, (cache.cqq*cache.interval_weights[:, None, None]).sum(dim=0))
        rescaled = precompute_leaf_interfaces(observed_example(), time_column=2, parent_branch_column=3, resistance_scale=7.)
        rescaled = with_query_reference_cache(rescaled, torch.tensor([30]))
        torch.testing.assert_close(rescaled.response*7, before.response, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(rescaled.query_cache.normalized_response, cache.normalized_response, rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'query order'):
            with_query_reference_cache(original, torch.tensor([30]), query_ids=torch.tensor([20, 10]))
        with self.assertRaises(FrozenInstanceError):
            before.query_cache = None

    def test_generic_geographic_bank_metadata_transfer_and_safe_roundtrip(self):
        seq = observed_example()
        seq.x = torch.randn(4, 2, dtype=torch.float32)
        seq.node_ids = torch.tensor([10, 20, 30, 40])
        seq.query_node_ids = torch.tensor([30, 10])
        base_edges = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
        r0 = torch.tensor([1., 2., 3., 4., 2.], dtype=torch.float64)
        r1 = torch.tensor([3., 1., 2., 1., 4.], dtype=torch.float64)
        seq.x = seq.x.repeat(2, 1)
        seq.node_ids = seq.node_ids.repeat(2)
        seq.node_graph = torch.tensor([0]*4+[1]*4)
        seq.clades = torch.zeros(8, 1, dtype=torch.float64)
        seq.edge_index = torch.cat((base_edges, base_edges+4), dim=1)
        seq.intervals = torch.tensor([[0., .2], [.2, 1.]], dtype=torch.float64)
        seq.positions = torch.tensor([.1, .7], dtype=torch.float64)
        seq.target_graph = torch.tensor([0, 1])
        bank = precompute_graph_interfaces(seq, torch.cat((r0, r1)))
        expected = resistance_laplacian(4, base_edges, r0)
        torch.testing.assert_close(bank.response[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(bank.response[1], resistance_laplacian(4, base_edges, r1), rtol=0, atol=0)
        self.assertEqual(bank.geometry_kind, 'generic_graph_boundary')
        bank = with_query_reference_cache(bank, torch.tensor([40, 20]))
        torch.testing.assert_close(bank.query_cache.query_indices, torch.tensor([2, 0]))
        torch.testing.assert_close(bank.query_cache.reference_indices, torch.tensor([3, 1]))
        torch.testing.assert_close(bank.query_cache.interval_weights, torch.tensor([.2, .8], dtype=torch.float64))
        torch.testing.assert_close(bank.query_cache.bag_cqq, .2*bank.query_cache.cqq[0]+.8*bank.query_cache.cqq[1])
        reconstructed = bank.query_cache.bag_eigenvectors @ torch.diag(bank.query_cache.bag_eigenvalues) @ bank.query_cache.bag_eigenvectors.T
        torch.testing.assert_close(reconstructed, bank.query_cache.bag_cqq, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(bank.query_node_ids, seq.query_node_ids)
        torch.testing.assert_close(bank.positions, seq.positions)
        torch.testing.assert_close(bank.target_graph, seq.target_graph)
        transferred = bank.to('cpu', dtype=torch.float32)
        self.assertEqual(transferred.leaf_x.dtype, torch.float32)
        self.assertEqual(transferred.response.dtype, torch.float32)
        self.assertEqual(transferred.query_cache.eigenvectors.dtype, torch.float32)
        for field in ('intervals', 'positions', 'window_bounds'):
            self.assertEqual(getattr(transferred, field).dtype, getattr(seq, field).dtype)
            torch.testing.assert_close(getattr(transferred, field), getattr(seq, field), rtol=0, atol=0)
        encoded = io.BytesIO()
        torch.save(bank.as_dict(), encoded)
        encoded.seek(0)
        restored = LeafInterfaceBank.from_dict(torch.load(encoded, weights_only=True))
        torch.testing.assert_close(restored.leaf_x, bank.leaf_x, rtol=0, atol=0)
        torch.testing.assert_close(restored.query_cache.cqr, bank.query_cache.cqr, rtol=0, atol=0)
        torch.testing.assert_close(restored.query_cache.bag_cqq, bank.query_cache.bag_cqq, rtol=0, atol=0)
        self.assertEqual(restored.diagnostics, bank.diagnostics)
        self.assertEqual(restored.schema_version, 1)

    def test_zero_response_normalizes_to_identity_scale(self):
        seq = observed_example()
        seq.node_ids = torch.tensor([10, 20, 30, 40])
        seq.edge_index = torch.empty(2, 0, dtype=torch.long)
        bank = precompute_graph_interfaces(seq, torch.empty(0))
        bank = with_query_reference_cache(bank, torch.tensor([30, 40]))
        torch.testing.assert_close(bank.query_cache.normalization_scale, torch.ones(1, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(bank.query_cache.eigenvalues, torch.zeros(1, 2, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(bank.query_cache.normalized_response, bank.response, rtol=0, atol=0)

    def test_query_reference_partition_and_cross_query_memory(self):
        response = kron_reduce(star((1., 1., 1.)), torch.tensor([0, 1, 2])).response
        partition = query_reference_partition(response, torch.tensor([10, 20, 30]), torch.tensor([20, 10]), torch.tensor([30]))
        with self.assertRaisesRegex(ValueError, 'exhaustive'):
            query_reference_partition(response, torch.tensor([10, 20, 30]), torch.tensor([10]), torch.tensor([30]))
        a, forcing = coupled_query_dynamics(partition.cqq, partition.cqr, torch.zeros(1, 2, dtype=torch.float64), coupling_rate=2., damping=.1)
        initial = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
        transition = coupled_affine_transition(a, forcing, .3)
        evolved = transition.apply(initial)
        self.assertGreater(float(evolved[1, 0]), 0.)
        self.assertLess(float(torch.linalg.eigvalsh(a).max()), 0.)
        self.assertLessEqual(float(torch.linalg.matrix_norm(transition.homogeneous, ord=2)), math.exp(-.1*.3)+1e-12)
        # A response negative only at the PSD input tolerance can still defeat
        # extremely small damping. Reject actual numerical instability.
        with self.assertRaisesRegex(ValueError, 'numerically dissipative'):
            coupled_query_dynamics(torch.diag(torch.tensor([-1e-12, 1.], dtype=torch.float64)),
                torch.zeros(2, 0, dtype=torch.float64), torch.zeros(0, 1, dtype=torch.float64), damping=1e-15)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            coupled_query_dynamics(100.*torch.eye(2, dtype=torch.float64),
                torch.ones(2, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64), coupling_rate=1e308)

    def test_exact_coupled_semigroup_zero_time_and_gradients(self):
        response = kron_reduce(star(), torch.tensor([0, 1, 2])).response.repeat(2, 1, 1)
        partition = query_reference_partition(response, torch.tensor([10, 20, 30]), torch.tensor([10, 20]), torch.tensor([30]))
        reference = torch.randn(2, 1, 3, dtype=torch.float64, requires_grad=True)
        query_force = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
        rate = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
        damping = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
        a, forcing = coupled_query_dynamics(partition.cqq, partition.cqr, reference,
            coupling_rate=rate, damping=damping, learned_query_forcing=query_force)
        initial = torch.randn(2, 2, 3, dtype=torch.float64)
        whole = coupled_affine_transition(a, forcing, torch.tensor([.4, .6], dtype=torch.float64)).apply(initial)
        first = coupled_affine_transition(a, forcing, torch.tensor([.1, .2], dtype=torch.float64)).apply(initial)
        split = coupled_affine_transition(a, forcing, torch.tensor([.3, .4], dtype=torch.float64)).apply(first)
        torch.testing.assert_close(whole, split, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(coupled_affine_transition(a, forcing, 0.).apply(initial), initial, rtol=0, atol=0)
        whole.square().mean().backward()
        for parameter in (reference, query_force, rate, damping):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        with self.assertRaises(ValueError):
            coupled_affine_transition(a, forcing, -.1)

    def test_cached_eigenbasis_formula_with_channel_scalar_rates(self):
        response = kron_reduce(star(), torch.tensor([0, 1, 2])).response.repeat(2, 1, 1)
        partition = query_reference_partition(response, torch.tensor([10, 20, 30]), torch.tensor([10, 20]), torch.tensor([30]))
        eigenvalues, eigenvectors = torch.linalg.eigh(partition.cqq)
        rates = torch.tensor([.3, 1., 3.], dtype=torch.float64)
        damping = torch.tensor([.1, .2, .3], dtype=torch.float64)
        state = torch.randn(2, 2, 3, dtype=torch.float64)
        reference = torch.randn(2, 1, 3, dtype=torch.float64)
        forcing = -(partition.cqr @ reference) * rates
        duration = torch.tensor([.3, .5], dtype=torch.float64)
        decay = eigenvalues[:, :, None] * rates + damping
        factor = torch.exp(-duration[:, None, None] * decay)
        integral = -torch.expm1(-duration[:, None, None] * decay) / decay
        spectral = eigenvectors @ (factor * (eigenvectors.transpose(-1, -2) @ state)
            + integral * (eigenvectors.transpose(-1, -2) @ forcing))
        a = -partition.cqq[:, None] * rates[None, :, None, None]
        a = a - damping[None, :, None, None] * torch.eye(2, dtype=torch.float64)
        exact = coupled_affine_transition(a, forcing.transpose(1, 2).unsqueeze(-1),
            duration[:, None]).apply(state.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        torch.testing.assert_close(spectral, exact, rtol=1e-12, atol=1e-12)

    def test_kron_heat_flow_is_not_projected_full_tree_heat_flow(self):
        laplacian = resistance_laplacian(3, torch.tensor([[0, 1], [1, 2]]), torch.tensor([1., 2.]))
        reduction = kron_reduce(laplacian, torch.tensor([0, 2]))
        boundary_initial = torch.tensor([1., 0.], dtype=torch.float64)
        initial = reduction.harmonic_lift(boundary_initial)
        full_boundary = (torch.matrix_exp(-laplacian) @ initial)[reduction.boundary_nodes]
        reduced = torch.matrix_exp(-reduction.response) @ boundary_initial
        self.assertGreater(float((full_boundary-reduced).abs().max()), .001)


if __name__ == '__main__':
    unittest.main()


@pytest.mark.parametrize("resistances", [[.2, .3], [.7], [.4, .6, .7]])
def test_single_boundary_response_is_analytically_zero_before_normalization(resistances):
    from arggnn.kron_interface import resistance_laplacian, kron_reduce
    n = len(resistances) + 1
    edge = torch.tensor([list(range(n-1)), list(range(1, n))])
    result = kron_reduce(resistance_laplacian(n, edge, torch.tensor(resistances, dtype=torch.float64)),
                         torch.tensor([0]))
    torch.testing.assert_close(result.response, torch.zeros(1, 1, dtype=torch.float64), rtol=0, atol=0)
    assert torch.isfinite(result.interior_extension).all()
    torch.testing.assert_close(result.harmonic_lift(torch.tensor([.4], dtype=torch.float64)),
                               torch.full((n,), .4, dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_single_boundary_components_in_disconnected_graph_are_exact_zero():
    edge = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])
    laplacian = resistance_laplacian(6, edge, torch.tensor([.2, .3, .2, .3], dtype=torch.float64))
    result = kron_reduce(laplacian, torch.tensor([0, 3]))
    torch.testing.assert_close(result.response, torch.zeros(2, 2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(result.harmonic_lift(torch.tensor([.4, .7], dtype=torch.float64)),
                               torch.tensor([.4, .4, .4, .7, .7, .7], dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)


def test_mixed_boundary_components_preserve_genuine_response():
    edge = torch.tensor([[0, 1, 3], [1, 2, 4]])
    laplacian = resistance_laplacian(5, edge, torch.tensor([.2, .3, .7], dtype=torch.float64))
    result = kron_reduce(laplacian, torch.tensor([0, 3, 4]))
    torch.testing.assert_close(result.response[0], torch.zeros(3, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(result.response[:,0], torch.zeros(3, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(result.response[1:,1:], torch.tensor([[1., -1.],[-1., 1.]], dtype=torch.float64)/.7)

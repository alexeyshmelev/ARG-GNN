"""Independent algebra and isolation tests, CPU only, with fixed toy observations."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from research.redesign_20261028.architecture.window_model import (
    Cut, WindowTreeLoom, prepare_observations, collate_observations,
    mean_to_cut, interval_kernel, coordinate_observations, context_kernel, temporal_mean)
from research.redesign_20261028.architecture.test_window_model import toy

HERE = Path('research/redesign_20261028/review')


class IndependentModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261028)

    def test_unequal_partition_mass_constant_and_contraction(self):
        groups = torch.tensor([[0, 0, 0, 1, 1], [2, 2, 3, 3, 3]])
        counts = torch.bincount(groups.flatten()).double()
        cut = Cut(torch.arange(4), groups, counts, torch.ones(4), torch.zeros(4))
        source = torch.tensor([[2., -1.], [8., 4.]], dtype=torch.float64)
        field = source[groups[0]]
        observed = mean_to_cut(field[None].repeat(2, 1, 1), cut)[2:]
        expected = torch.tensor([[2., -1.], [6., 7./3]], dtype=torch.float64)
        torch.testing.assert_close(observed, expected)
        torch.testing.assert_close((observed * counts[2:, None]).sum(0), (source * counts[:2, None]).sum(0))
        before = (source.square().sum(1) * counts[:2]).sum()
        after = (observed.square().sum(1) * counts[2:]).sum()
        self.assertLessEqual(float(after), float(before))
        constant = mean_to_cut(torch.ones(2, 5, 2, dtype=torch.float64), cut)
        torch.testing.assert_close(constant, torch.ones_like(constant))

    def test_kernel_refinement_limits_are_explicit_counterexamples(self):
        coarse = torch.tensor([[0., 1.], [1., 4.]], dtype=torch.float64)
        refined = torch.tensor([[0., .2], [.2, 1.], [1., 4.]], dtype=torch.float64)
        ell = torch.tensor(.7, dtype=torch.float64)
        k = interval_kernel(coarse, torch.zeros(2, dtype=torch.long), ell)
        r = interval_kernel(refined, torch.zeros(3, dtype=torch.long), ell)
        merge = torch.tensor([[1., 1., 0.], [0., 0., 1.]], dtype=torch.float64)
        torch.testing.assert_close(merge @ r @ merge.T, k, atol=1e-12, rtol=1e-12)
        field = torch.tensor([1., 0.], dtype=torch.float64)
        mean = (k @ field) / k.sum(1)
        physical_mass_before = float((torch.tensor([1., 3.]) * field).sum())
        physical_mass_after = float((torch.tensor([1., 3.]) * mean).sum())
        self.assertGreater(abs(physical_mass_before - physical_mass_after), 1e-3)
        fine_field = torch.tensor([1., 1., 0.], dtype=torch.float64)
        fine_mean = (r @ fine_field) / r.sum(1)
        merged_nonlinear = ((fine_mean[:2].square() * r.sum(1)[:2]).sum() / r.sum(1)[:2].sum())
        nonlinear_coarse = mean[0].square()
        self.assertGreater(float((merged_nonlinear - nonlinear_coarse).abs()), 1e-4)
        receipt = dict(raw_integral_additivity=True,
            physical_mass_before=physical_mass_before, physical_mass_after=physical_mass_after,
            coarse_nonlinear=float(nonlinear_coarse), refined_then_merged_nonlinear=float(merged_nonlinear),
            interpretation='Row-normalized temporal mixing does not conserve original genomic mass; nonlinear target refinement is not invariant.')
        (HERE/'algebra_counterexamples.json').write_text(json.dumps(receipt, indent=2)+'\n')

    def test_loss_on_one_window_has_zero_input_gradient_to_other_window(self):
        a = toy()
        b = copy.deepcopy(a)
        b['site_genotypes'] = (b['site_genotypes'] + 1) % 2
        batch = prepare_observations(collate_observations([a, b]))
        metadata = batch.metadata.detach().clone().requires_grad_(True)
        batch = replace(batch, metadata=metadata)
        model = WindowTreeLoom(width=16)
        logits = model(batch)
        gradient = torch.autograd.grad(logits[:3].square().sum(), metadata)[0]
        self.assertGreater(float(gradient[0].abs().max()), 0.)
        self.assertEqual(float(gradient[1].abs().max()), 0.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_coordinate_duplicate_measure_forced_atom_and_whitelist(self):
        a = toy()
        for key in ['focal_tree_index', 'focal_sample_index', 'focal_position_local_bp', 'focal_genotypes', 'focal_allele_count']:
            a[key] = a[key][1:2]
        context = dict(context_coordinate_local_bp=np.array([120., 20., 20., 250.]),
            context_tree_index=np.array([1, 0, 0, 2]), context_window_index=np.zeros(4, dtype=np.int64),
            context_quadrature_weight=np.array([0., 1/3, 1/3, 1/3]),
            context_proposal_density_per_bp=np.array([0., 1/400, 1/400, 1/400]),
            context_forced_focal=np.array([True, False, False, False]),
            window_context_domain_local_bp=np.array([[0., 400.]]), window_context_kind=np.array([1]))
        obs = coordinate_observations(a, context)
        batch = prepare_observations(obs)
        self.assertEqual(len(batch.leaves), 4)
        self.assertEqual(batch.focal_tree.tolist(), [0])
        self.assertEqual(batch.core.tolist(), [True, False, False, False])
        values = torch.tensor([100000., 10., 10., 0.], dtype=torch.float64)[:, None, None]
        got = temporal_mean(values, context_kernel(batch, torch.tensor(100.)), batch.tree_window)[0, 0, 0]
        expected = 20 * np.exp(-1) / (2 * np.exp(-1) + np.exp(-1.3))
        collapsed_wrong = 10 * np.exp(-1) / (np.exp(-1) + np.exp(-1.3))
        self.assertAlmostEqual(float(got), expected, places=10)
        self.assertGreater(abs(float(got) - collapsed_wrong), 1.)
        with self.assertRaisesRegex(ValueError, 'whitelist'):
            coordinate_observations(dict(a, labels=np.array([4])), context)
        with self.assertRaisesRegex(ValueError, 'whitelist'):
            prepare_observations(dict(obs, labels=np.array([4])))
        joined = prepare_observations(collate_observations([obs, obs]))
        metadata = joined.metadata.detach().clone().requires_grad_(True)
        joined = replace(joined, metadata=metadata)
        model = WindowTreeLoom(width=16)
        logits = model(joined)
        grad = torch.autograd.grad(logits[:1].square().sum(), metadata)[0]
        self.assertEqual(float(grad[1].abs().max()), 0.)

    def test_optimized_no_cross_matches_explicit_two_sweeps_and_gradients(self):
        batch = prepare_observations(toy())
        model = WindowTreeLoom(width=16)
        reference = copy.deepcopy(model)
        actual = model(batch, mode='no_cross')
        base, gates, degree = reference.bases(batch)
        reference.sweep(batch, base, gates, degree)
        _, _, second = reference.sweep(batch, base + torch.zeros_like(base), gates, degree)
        focal = reference.encode_snps(batch.focal_genotype, batch.focal_alleles,
            batch.tree_window[batch.focal_tree], batch)
        row = torch.arange(len(batch.focal_tree))
        features = torch.cat((second[batch.focal_tree, batch.focal_sample], focal[row, batch.focal_sample],
            batch.metadata[batch.tree_window[batch.focal_tree], batch.focal_sample]), -1)
        expected = reference.classifier(features)
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.grad is None or other.grad is None:
                self.assertIsNone(parameter.grad)
                self.assertIsNone(other.grad)
            else:
                torch.testing.assert_close(parameter.grad, other.grad, atol=0., rtol=0.)

    def test_shared_parent_metadata_expands_over_all_windows(self):
        a = toy()
        joined = collate_observations([a, a])
        for key in ['leaf_features', 'reference_class', 'sample_age', 'observed_cohort', 'query_mask']:
            joined[key] = joined[key][0]
        batch = prepare_observations(joined)
        self.assertEqual(tuple(batch.metadata.shape), (2, 4, 17))
        composed = prepare_observations(collate_observations([joined, a]))
        self.assertEqual(tuple(composed.metadata.shape), (3, 4, 17))


if __name__ == '__main__':
    unittest.main(verbosity=2)

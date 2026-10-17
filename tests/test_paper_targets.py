import math
import unittest
from arggnn.benchmarking.contracts import fingerprint
from arggnn.benchmarking.targets import SampledFocalTargets, sampled_focal_loss



def targets(rows, columns, labels, classes):
    return SampledFocalTargets(rows, columns, labels, classes, "parent", "train",
        {"path": "fixtures/train_registry.json", "sha256": fingerprint("registry")})


class SampledTargetTests(unittest.TestCase):
    def test_sample_multiplicity_and_gradients_without_bp_weights(self):
        import torch
        torch.set_num_threads(1)
        logits = torch.tensor([[[math.log(.8), math.log(.2)]],
                               [[math.log(.25), math.log(.75)]]], requires_grad=True)
        sample = targets((0, 1, 0), (0, 0, 0), (0, 0, 0), ("A", "B"))
        loss = sampled_focal_loss(logits, sample)
        self.assertAlmostEqual(float(loss.detach()), -(2*math.log(.8)+math.log(.25))/3, places=6)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertAlmostEqual(float(logits.grad[0, 0, 0]), 2*(.8-1)/3, places=6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_endpoint_or_outside_readout_rejected(self):
        import torch
        logits = torch.zeros((2, 1, 6))
        for sample in (targets((0,), (0,), (1,), ("A", "B")),
                        targets((2,), (0,), (0,), tuple("ABCDEF"))):
            with self.assertRaises(ValueError):
                sampled_focal_loss(logits, sample)
        self.assertFalse(torch.cuda.is_initialized())

    def test_target_indices_and_labels_are_explicit(self):
        with self.assertRaises(ValueError):
            targets((True,), (0,), (0,), ("A", "B"))
        with self.assertRaises(ValueError):
            targets((0,), (0,), (6,), tuple("ABCDEF"))

    def test_target_parent_and_sampling_registry_are_bound(self):
        sample = targets((0,), (0,), (0,), ("A", "B"))
        sample.validate_binding(parent_id="parent", split="train",
                                sampled_registry_ref=sample.sampled_registry_ref)
        for override in ({"parent_id": "other"}, {"split": "test"},
                         {"sampled_registry_ref": {"path": "fixtures/other.json",
                                                   "sha256": fingerprint("other")}}):
            args = dict(parent_id="parent", split="train",
                        sampled_registry_ref=sample.sampled_registry_ref)
            args.update(override)
            with self.assertRaises(ValueError):
                sample.validate_binding(**args)

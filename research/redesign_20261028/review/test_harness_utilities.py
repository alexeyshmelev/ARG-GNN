"""Independent deterministic harness checks; never invoke run or an optimizer."""
import copy
import math
import unittest
import numpy as np
import torch
from research.redesign_20261028.root import learning_diagnostic as harness
from research.redesign_20261028.architecture.test_window_model import toy


class HarnessUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metrics_match_hand_computed_confusion_and_six_class_f1(self):
        labels = torch.tensor([0, 1, 1, 2])
        predictions = torch.tensor([0, 1, 2, 2])
        logits = torch.nn.functional.one_hot(predictions, 6).float()
        score = harness.score(logits, labels)
        self.assertEqual(score['accuracy'], .75)
        self.assertAlmostEqual(score['macro_f1_six_classes'], 7/18)
        self.assertAlmostEqual(score['cross_entropy'], math.log(math.e+5)-.75, places=6)
        self.assertEqual(score['true_class_counts'], [1, 2, 1, 0, 0, 0])
        self.assertEqual(score['predicted_class_counts'], [1, 1, 2, 0, 0, 0])
        self.assertEqual(score['confusion_true_rows_predicted_columns'][1], [0, 1, 1, 0, 0, 0])

    def test_exact_order_has_twelve_full_epochs_and_half_epoch(self):
        order = np.array(harness.make_order(128, 400, 4, 20261031))
        np.testing.assert_array_equal(order, harness.make_order(128, 400, 4, 20261031))
        self.assertEqual(order.shape, (400, 4))
        for epoch in range(12):
            np.testing.assert_array_equal(np.sort(order[epoch*32:(epoch+1)*32].ravel()), np.arange(128))
        self.assertEqual(len(np.unique(order[384:])), 64)
        counts = np.bincount(order.ravel(), minlength=128)
        self.assertEqual(np.unique(counts).tolist(), [12, 13])
        self.assertEqual(int(counts.sum()), 1600)

    def test_collation_retains_repeated_target_draws_and_separate_labels(self):
        a = toy()
        for key in ['focal_tree_index', 'focal_sample_index', 'focal_position_local_bp', 'focal_genotypes', 'focal_allele_count']:
            a[key] = a[key][1:2]
        batch, labels, indices, stats = harness.collate([(a, 2, 7, .1), (copy.deepcopy(a), 2, 7, .2)])
        self.assertEqual(labels.tolist(), [2, 2])
        self.assertEqual(indices, [7, 7])
        self.assertEqual(len(batch.focal_tree), 2)
        self.assertTrue(bool(batch.core_focal_mask.all()))
        self.assertFalse(hasattr(batch, 'labels'))
        self.assertAlmostEqual(stats['packet_read_worker_seconds'], .3)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Bounded synthetic CPU checks for fresh four-output CNN construction."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from arggnn.benchmarking.paper_ancestralpaths_four import CONDITION
from arggnn.benchmarking.paper_ancestralpaths_four import model


class FourModelSpecTests(unittest.TestCase):
    def test_actual_four_output_shapes_and_fresh_training_spec(self):
        spec = model.model_spec()
        self.assertEqual(spec["input_shape"], [1, 5, 8])
        self.assertEqual(spec["parameters"], 16900)
        self.assertEqual(spec["weight_shapes"][-2:], [[64, 4], [4]])
        self.assertFalse(spec["probability_collapsing"])
        self.assertFalse(spec["six_class_weights_reused"])
        self.assertFalse(spec["old_checkpoint_initialization"])
        self.assertFalse(spec["published_four_path_training_counts_specified"])
        self.assertEqual(model.expected_optimizer_updates(40000), 6670)
        self.assertEqual(model.expected_optimizer_updates(1), 5)
        self.assertEqual(model.expected_optimizer_updates(30), 5)
        self.assertEqual(model.expected_optimizer_updates(31), 10)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                model.expected_optimizer_updates(invalid)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "qualified TensorFlow CPU runtime required")
class FourCNNCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("This smoke test requires explicit zero-CUDA visibility")
        cls.tf = model.load_runtime(Path(sys.prefix), device="cpu", threads=1)

    def tearDown(self):
        self.tf.keras.backend.clear_session()

    def test_zero_input_uniform_four_probs_and_separate_native_six_unchanged(self):
        from arggnn.benchmarking.paper_ancestralpaths.model import build_cnn as build_six
        four = model.build_cnn(self.tf, layout="portable_nhwc")
        six = build_six(self.tf, layout="portable_nhwc")
        self.assertEqual(four.name, CONDITION)
        self.assertEqual(four.count_params(), 16900)
        self.assertEqual(six.count_params(), 19078)
        self.assertEqual(six.input_shape, (None, 1, 5, 9))
        self.assertEqual(six.output_shape, (None, 6))
        p = four(np.zeros((2, 1, 5, 8), dtype=np.float32), training=False).numpy()
        np.testing.assert_array_equal(p, np.full((2, 4), .25, dtype=np.float32))
        self.assertIsNot(four.optimizer, six.optimizer)
        self.assertEqual(int(four.optimizer.iterations.numpy()), 0)
        self.assertEqual(int(six.optimizer.iterations.numpy()), 0)
        self.assertEqual(self.tf.config.list_physical_devices("GPU"), [])

    def test_one_synthetic_padded_cpu_update_is_finite_and_fresh_rebuild(self):
        self.tf.keras.utils.set_random_seed(7)
        network = model.build_cnn(self.tf, layout="portable_nhwc")
        x = np.random.default_rng(7).normal(size=(4, 1, 5, 8)).astype(np.float32)
        x[1, :, 2:, :] = -15
        y = np.eye(4, dtype=np.float32)
        loss = network.train_on_batch(x, y)
        self.assertTrue(np.isfinite(loss).all())
        self.assertEqual(int(network.optimizer.iterations.numpy()), 1)
        p = network(x, training=False).numpy()
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-7)
        fresh = model.build_cnn(self.tf, layout="portable_nhwc")
        self.assertEqual(int(fresh.optimizer.iterations.numpy()), 0)

    def test_checkpoint_preserves_four_predictions_and_optimizer(self):
        from arggnn.benchmarking import european_native_models as io
        self.tf.keras.utils.set_random_seed(11)
        network = model.build_cnn(self.tf, layout="portable_nhwc")
        x = np.random.default_rng(11).normal(size=(4, 1, 5, 8)).astype(np.float32)
        network.train_on_batch(x, np.eye(4, dtype=np.float32))
        expected = network(x, training=False).numpy()
        with tempfile.TemporaryDirectory(prefix="ap_four_test_", dir=io.ROOT / ".runtime/tmp") as directory:
            path = Path(directory) / "synthetic_four.h5"
            network.save(path)
            restored = self.tf.keras.models.load_model(path)
            self.assertEqual(restored.input_shape, (None, 1, 5, 8))
            self.assertEqual(restored.output_shape, (None, 4))
            self.assertEqual(int(restored.optimizer.iterations.numpy()), 1)
            np.testing.assert_array_equal(restored(x, training=False).numpy(), expected)
            for before, after in zip(network.optimizer.get_weights(), restored.optimizer.get_weights()):
                np.testing.assert_array_equal(before, after)

    def test_nchw_declares_same_kernel_shapes_without_cpu_execution(self):
        network = model.build_cnn(self.tf, layout="native_nchw")
        self.assertEqual([list(w.shape) for w in network.weights], model.WEIGHT_SHAPES)
        self.assertEqual(network.get_layer("author_flatten").data_format, "channels_first")
        with self.assertRaises(ValueError):
            model.build_cnn(self.tf, layout="unknown")


if __name__ == "__main__":
    unittest.main()

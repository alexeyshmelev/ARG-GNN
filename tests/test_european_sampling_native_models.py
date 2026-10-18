"""Synthetic AP sampling, author-AST and fake-fit tests. No model is trained."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import numpy as np

from arggnn.benchmarking import european_sampling as sampling
from arggnn.benchmarking import european_native_models as models


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.plan = sampling.paper_sampling_plan(
            ["train_" + str(i) for i in range(3)],
            ["test_" + str(i) for i in range(5)], seed=8123)
        self.rows = []
        for cohort, count in zip(
                (*sampling.COHORTS, "WHG", "EHG", "Ana", "CHG"),
                (182, 162, 14, 302, 26, 102, 86, 50, 28)):
            for _ in range(count):
                i = len(self.rows)
                self.rows.append(dict(
                    sample_index=i, sample_id="s" + str(i // 2), haplotype=i % 2,
                    observed_node_id=1000 + i, observed_cohort=cohort,
                    role="query" if i < 686 else "pure_source_reference"))
        self.inferred = dict(
            sample_index_to_inferred_tree_node_id=[
                dict(sample_index=r["sample_index"], inferred_tree_node_id=r["observed_node_id"])
                for r in self.rows],
            coverage_original_bp0=[0., 200000000.], local_to_original_offset_bp=0.,
            site_join=[dict(native_site_id=i, local_position_0based=float(i * 1000),
                            original_position_0based=float(i * 1000)) for i in range(1000)])
        self.observed_ref = dict(path="experiments/results/synthetic/inference.json", sha256="a" * 64)

    def registry(self, parent="train_0"):
        return sampling.build_sampled_registry(
            self.plan, parent, self.rows, self.inferred, observation_ref=self.observed_ref)

    def test_exact_paper_counts_and_disjoint_parent_allocation(self):
        self.assertEqual([sum(self.plan["parents"][p]["cohort_quotas"].values())
                          for p in self.plan["train_parent_ids"]], [16667, 16667, 16666])
        for c in sampling.COHORTS:
            self.assertEqual(sum(self.plan["parents"][p]["cohort_quotas"][c]
                                 for p in self.plan["train_parent_ids"]), 10000)
        self.assertTrue(all(sum(self.plan["parents"][p]["cohort_quotas"].values()) == 10000
                            for p in self.plan["test_parent_ids"]))
        with self.assertRaises(ValueError):
            sampling.paper_sampling_plan(["a", "b", "c"], ["a", "d", "e", "f", "g"], seed=1)

    def test_deterministic_observation_only_registry_and_canonical_array_order(self):
        registry = self.registry()
        self.assertEqual(registry, self.registry())
        self.assertEqual(registry["examples"], 16667)
        positions = [r["position_local_bp"] for r in registry["records"]]
        self.assertEqual(positions, sorted(positions))
        self.assertFalse(registry["query_truth_read"])
        arrays = sampling.registry_arrays(registry)
        canonical = sorted(registry["records"], key=lambda r: r["example_index"])
        self.assertTrue(np.array_equal(arrays["record_ids"], np.arange(16667)))
        self.assertTrue(np.array_equal(arrays["sample_nodes"],
                                       [r["query_node_id"] for r in canonical]))
        self.assertEqual(arrays["parent_id"].shape, ())
        self.assertEqual(arrays["class_order"].tolist(), list(sampling.NATIVE_PATHS))

    def test_native_six_labels_join_actual_same_sites_and_query_keys(self):
        registry = self.registry()
        identities = registry["query_identities"]
        schema = dict(parent_id=registry["parent_id"], split="train",
                      native_paths_1based=[1, 2, 3, 4, 5, 6],
                      query_sample_indices=[r["sample_index"] for r in identities],
                      query_keys=[[r["sample_id"], r["haplotype"]] for r in identities])
        first = np.array([5 if r["cohort"] == "BAA" else 1 for r in identities], dtype=np.uint8)
        truth = dict(intervals_bp=np.array([[0., 500000.], [500000., 200000000.]]),
                     query_raw_node_ids=np.array(schema["query_sample_indices"], dtype=np.int32),
                     native_paths_1based=np.stack((first, first + 1)))
        labels = sampling.native_truth_labels(registry, schema, truth)
        for row in registry["records"]:
            expected = (4 if row["cohort"] == "BAA" else 0) + int(row["position_original_bp"] >= 500000)
            self.assertEqual(int(labels[row["example_index"]]), expected)
        bad = dict(schema, parent_id="another_parent")
        with self.assertRaisesRegex(ValueError, "Same-parent"):
            sampling.native_truth_labels(registry, bad, truth)
        bad = dict(schema, query_keys=list(reversed(schema["query_keys"])))
        with self.assertRaisesRegex(ValueError, "identities"):
            sampling.native_truth_labels(registry, bad, truth)

    def test_too_few_observed_sites_is_failure_without_redraw(self):
        small = dict(self.inferred, site_join=self.inferred["site_join"][:1])
        with self.assertRaisesRegex(ValueError, "Too few"):
            sampling.build_sampled_registry(
                self.plan, "train_0", self.rows, small, observation_ref=self.observed_ref)


class FakeIterations:
    def __init__(self):
        self.value = 0
    def numpy(self):
        return self.value


class FakeModel:
    def __init__(self, layers):
        self.layers = layers
        self.output_shape = (None, 6)
        self.optimizer = SimpleNamespace(iterations=FakeIterations())
    def compile(self, **kwargs):
        self.compiled = kwargs
    def count_params(self):
        return 26630
    def fit(self, x, y, **kwargs):
        self.fit_kwargs = kwargs
        self.optimizer.iterations.value = 7815
        return SimpleNamespace(history={"loss": [1., .9, .8, .7, .6]})


def fake_tf():
    def layer(name):
        return lambda *a, **kw: (name, a, kw)
    created = []
    def sequential(layers):
        model = FakeModel(layers)
        created.append(model)
        return model
    keras = SimpleNamespace(
        models=SimpleNamespace(Sequential=sequential),
        layers=SimpleNamespace(Flatten=layer("Flatten"), Dense=layer("Dense"),
                               Dropout=layer("Dropout")),
        backend=SimpleNamespace(clear_session=lambda: None),
        utils=SimpleNamespace(set_random_seed=lambda seed: None,
            to_categorical=lambda y, num_classes: np.eye(num_classes)[y]))
    return SimpleNamespace(keras=keras, nn=SimpleNamespace(relu=object(), softmax=object())), created


class NativeModelTests(unittest.TestCase):
    def test_original_author_model_and_final_compile_statements_execute(self):
        tf, created = fake_tf()
        model = models.build_author_dense(tf)
        self.assertIs(model, created[0])
        self.assertEqual([layer[0] for layer in model.layers],
                         ["Flatten", "Dense", "Dropout", "Dense"])
        self.assertEqual(model.layers[1][1], (512,))
        self.assertEqual(model.layers[2][1], (0.2,))
        self.assertEqual(model.layers[3][1], (6,))
        self.assertEqual(model.compiled,
                         dict(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]))

    def test_fake_fit_receives_full_five_epoch_budget_and_exact_update_count(self):
        tf, _ = fake_tf()
        x = np.zeros((50000, 5, 9), dtype=np.float32)
        y = np.arange(50000, dtype=np.int64) % 6
        model, _, updates = models.fit_author_arrays(tf, x, y, seed=111, guard=lambda _: None)
        self.assertEqual(updates, 7815)
        self.assertEqual(model.fit_kwargs,
                         dict(epochs=5, batch_size=32, shuffle=True, verbose=2))

    def test_test_registry_denied_before_private_target_metadata(self):
        records = {
            "registry": dict(split="test", parent_id="heldout"),
            "features": dict(split="test", parent_id="heldout", registry_ref="registry",
                             features_contain_no_query_truth=True)}
        seen = []
        def read(record, **kwargs):
            seen.append(record)
            return records[record]
        with patch.object(models, "_read", side_effect=read):
            with self.assertRaisesRegex(ValueError, "Only frozen registered TRAIN"):
                models.prepare_train_labels(
                    "registry", "features", "PRIVATE_GENERATION", "PRIVATE_SCHEMA", "PRIVATE_TRUTH",
                    "unused", guard=lambda _: None, execution_check=lambda: None,
                    write_headroom_bytes=1, max_truth_uncompressed_bytes=1)
        self.assertEqual(seen, ["registry", "features"])


if __name__ == "__main__":
    unittest.main()

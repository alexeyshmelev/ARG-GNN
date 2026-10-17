import dataclasses
import unittest
import numpy as np

from arggnn.benchmarking.contracts import ContractError, fingerprint
from arggnn.benchmarking.metrics import (Prediction, score_parent, collapse_probabilities,
                                        summarize_parents, assert_paired_support as _paired)


def prediction(method="a", parent="p"):
    return Prediction(method, "fixture", parent, ("A", "B"),
                      ((parent, "h0", 1.0), (parent, "h1", 2.0)),
                      np.array([1, 0]), np.array([[.9, .1], [.1, .9]]),
                      "observed", "panels")


def score_parent(prediction, truth_keys, truth, **kwargs):
    from arggnn.benchmarking.metrics import score_parent as score
    kwargs.setdefault("expected_support_fingerprint", fingerprint(sorted(prediction.keys)))
    return score(prediction, truth_keys, truth, **kwargs)


def assert_paired_support(results):
    from test_benchmark_contracts import protocol
    frozen = protocol()
    frozen.update(class_order=["A", "B"], methods=["a", "b"])
    frozen["split_parents"]["validation"] = ["p"]
    return _paired(results, protocol=frozen, split="validation",
                   expected_support_fingerprints={"p": fingerprint(sorted(prediction().keys))})


class MetricTests(unittest.TestCase):
    def test_native_calls_are_not_replaced_by_argmax(self):
        p = prediction()
        r = score_parent(p, p.keys, np.array([1, 0]))
        self.assertEqual(r["accuracy"], 1)
        self.assertGreater(r["nll"], 2)
        self.assertFalse(r["hard_calls_replaced_by_posterior_argmax"])

    def test_same_support_can_arrive_in_different_order(self):
        p = prediction()
        r = score_parent(p, p.keys[::-1], np.array([0, 1]))
        self.assertEqual(r["accuracy"], 1)

    def test_missing_targets_rejected(self):
        p = prediction()
        with self.assertRaises(ContractError):
            score_parent(p, p.keys[:1], np.array([1]))

    def test_duplicate_predictions_rejected(self):
        p = dataclasses.replace(prediction(), keys=(("p", "h0", 1.0),) * 2)
        with self.assertRaises(ContractError):
            score_parent(p, p.keys, np.array([1, 0]))

    def test_raw_native_zero_gives_infinite_nll(self):
        p = dataclasses.replace(prediction(), probabilities=np.array([[1., 0.], [1., 0.]]))
        r = score_parent(p, p.keys, np.array([1, 0]))
        self.assertIsNone(r["nll"])
        self.assertEqual(r["nll_status"], "infinite_zero_true_class_probability")

    def test_weights_are_explicit(self):
        p = prediction()
        r = score_parent(p, p.keys, np.array([1, 1]), weights=np.array([2., 1.]))
        self.assertAlmostEqual(r["accuracy"], 2 / 3)

    def test_different_observations_do_not_form_a_paired_comparison(self):
        p = prediction()
        a = score_parent(p, p.keys, np.array([1, 0]))
        q = dataclasses.replace(p, method_id="b", observation_fingerprint="other")
        b = score_parent(q, q.keys, np.array([1, 0]))
        with self.assertRaises(ContractError):
            assert_paired_support({"a": [a], "b": [b]})

    def test_same_support_is_required_for_every_parent(self):
        p = prediction()
        a = score_parent(p, p.keys, np.array([1, 0]))
        b = dict(a, method_id="b")
        assert_paired_support({"a": [a], "b": [b]})
        with self.assertRaises(ContractError):
            assert_paired_support({"a": [a], "b": []})

    def test_common_parent_omission_rejected(self):
        from test_benchmark_contracts import protocol
        frozen = protocol()
        frozen.update(methods=["a", "b"])
        frozen["split_parents"]["validation"] = ["p", "omitted"]
        p = prediction()
        a = score_parent(p, p.keys, np.array([1, 0]))
        with self.assertRaisesRegex(ContractError, "omitted"):
            _paired({"a": [a], "b": [dict(a, method_id="b")]},
                    protocol=frozen, split="validation",
                    expected_support_fingerprints={"p": a["support_fingerprint"], "omitted": "x"})

    def test_joint_prediction_truth_truncation_rejected(self):
        p = prediction()
        q = dataclasses.replace(p, keys=p.keys[:1], hard_calls=p.hard_calls[:1],
                                probabilities=p.probabilities[:1])
        with self.assertRaisesRegex(ContractError, "pre-frozen"):
            score_parent(q, q.keys, np.array([1]),
                         expected_support_fingerprint=fingerprint(sorted(p.keys)))

    def test_row_order_does_not_change_paired_identity(self):
        p = prediction()
        a = score_parent(p, p.keys, np.array([1, 0]), weights=[2., 1.])
        b = score_parent(dataclasses.replace(p, method_id="b"), p.keys[::-1],
                         np.array([0, 1]), weights=[1., 2.])
        assert_paired_support({"a": [a], "b": [b]})

    def test_class_sd_is_between_parents_not_between_classes(self):
        rows = [dict(method_id="a", protocol_id="fixture", parent_id=str(i),
                     class_order=["A", "B"], precision=[1., 0.]) for i in range(2)]
        r = summarize_parents(rows, "precision")
        self.assertEqual(r["mean"], [1., 0.])
        self.assertEqual(r["sample_sd"], [0., 0.])

    def test_invalid_native_probabilities_cannot_be_hidden_by_collapse(self):
        for p in (np.array([[-.2, .4, .3, .3, .1, .1]]),
                  np.array([[.1, .1, .1, .1, .1, .1]])):
            with self.subTest(p=p), self.assertRaises(ContractError):
                collapse_probabilities(p, [[0, 1], [2, 3], [4], [5]])

    def test_sample_sd_is_across_independent_parents(self):
        rows = [dict(method_id="a", protocol_id="x", parent_id=str(i),
                     class_order=["A", "B"], accuracy=x)
                for i, x in enumerate([.5, 1.])]
        result = summarize_parents(rows)
        self.assertAlmostEqual(result["mean"], .75)
        self.assertAlmostEqual(result["sample_sd"], np.std([.5, 1.], ddof=1))
        self.assertEqual(result["unit"], "fraction")
        with self.assertRaises(ContractError):
            summarize_parents([rows[0], rows[0]])

    def test_loss_units_are_not_labelled_as_fractions(self):
        rows = [dict(method_id="a", protocol_id="x", parent_id="p",
                     class_order=["A", "B"], nll=2., brier=1.5)]
        self.assertEqual(summarize_parents(rows, "nll")["unit"], "nats")
        self.assertEqual(summarize_parents(rows, "brier")["unit"], "squared_probability_sum")

    def test_ancestralpath_endpoint_conversion_is_explicit(self):
        p = np.array([[.1, .2, .1, .2, .15, .25]])
        out = collapse_probabilities(p, [[0, 4], [1, 5], [2], [3]])
        np.testing.assert_allclose(out, [[.25, .45, .1, .2]])
        with self.assertRaises(ContractError):
            collapse_probabilities(p, [[0, 0], [1], [2, 3, 4, 5]])

    def test_no_supervised_empirical_accuracy(self):
        p = prediction()
        with self.assertRaises(ContractError):
            score_parent(p, p.keys, np.array([1, 0]), empirical=True)


if __name__ == "__main__":
    unittest.main()

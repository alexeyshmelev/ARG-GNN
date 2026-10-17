"""Independent RAM-only regressions: no admission, resource, process or payload I/O."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import metrics as m

class AdmissionRegressionTests(unittest.TestCase):
    def attempt(self, *, argv=None, status=0):
        def ref(name):
            return {"path": name+".json", "sha256": "a"*64}
        product, scientific_input, actual_wait, receipt = map(ref, (
            "RAM_product", "RAM_input", "RAM_wait", "RAM_receipt"))
        argv = argv if argv is not None else [".runtime/env/bin/python", "-m", "arggnn.example"]
        packet = {"schema": "benchmark_admitted_job_v1", "guarded_admission_ref": product,
                  "input_ref": scientific_input, "admission_actual_wait_ref": actual_wait,
                  "resource": "cpu", "argv": argv, "cwd": str(c.ROOT),
                  "max_seconds": 1, "job_id": "RAM_only_never_dispatched"}
        objects = {
            product["path"]: {"execute_argv": argv, "input_ref": scientific_input},
            scientific_input["path"]: {},
            actual_wait["path"]: {"actual_direct_wait": True, "actual_direct_returncode": status,
                                  "admitted_ref": product},
            receipt["path"]: packet,
        }
        def fake_path(value):
            if value == str(c.ROOT):
                return c.ROOT
            return SimpleNamespace(is_file=lambda: True, name="python")
        with patch.object(c, "read_ref", side_effect=lambda value: objects[value["path"]]),              patch.object(c, "contained_path", side_effect=fake_path):
            return c.validate_admission(receipt)

    def test_generic_dictionary_is_not_a_genuine_guarded_admission(self):
        with self.assertRaises(c.ContractError):
            self.attempt()

    def test_boolean_zero_is_not_an_actual_wait_integer(self):
        with self.assertRaises(c.ContractError):
            self.attempt(status=False)

    def test_sets_id_without_command_has_a_contract_error(self):
        with self.assertRaises(c.ContractError):
            self.attempt(argv=["setsid"])

    def test_module_flag_without_module_has_a_contract_error(self):
        with self.assertRaises(c.ContractError):
            self.attempt(argv=[".runtime/env/bin/python", "-m"])

class MetricRegressionTests(unittest.TestCase):
    def test_native_simplex_is_validated_before_endpoint_collapse(self):
        with self.assertRaises(c.ContractError):
            m.collapse_probabilities(
                np.array([[-.1, .3, .2, .2, .3, .1]]), [[0,4], [1,5], [2], [3]])

    def test_class_values_are_not_flattened_into_parent_standard_deviation(self):
        rows = [dict(method_id="a", protocol_id="fixture", parent_id=str(index),
                     class_order=["A", "B"], precision=[1., 0.]) for index in range(2)]
        try:
            result = m.summarize_parents(rows, "precision")
        except c.ContractError:
            return
        # A vector-aware implementation is fine, provided both class SDs are zero.
        sd = np.asarray(result["sample_sd"])
        self.assertEqual(sd.shape, (2,))
        np.testing.assert_array_equal(sd, [0., 0.])

if __name__ == "__main__":
    unittest.main()

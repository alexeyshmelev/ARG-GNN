"""Small complete CPU forests: all cut bytes/metadata and actual gather counts."""
import ast
import copy
from dataclasses import fields
import hashlib
from pathlib import Path
import types
from types import SimpleNamespace
import unittest
import torch
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import (
    candidate_v1 as frozen, prepare_integration_v2 as frozen_integration,
    qualify_parent_v2 as frozen_qualifier)
from research.reviews.workflow_progress_diagnosis_20261018.treeloom.test_candidate_v1 import (
    fixture, prepare_original)
from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_preparation_v1 as original
from . import cut_hoist_v1 as repaired
from . import integration_v1 as integration


def instrument_original(counter):
    def compiling(function, transform=None, **bindings):
        def instrument(node):
            if transform is not None:
                transform(node)
            class CountRootGather(ast.NodeTransformer):
                count = 0
                def visit_Subscript(self, item):
                    if ast.dump(item, include_attributes=False) == ast.dump(
                            ast.parse("roots[owner]", mode="eval").body, include_attributes=False):
                        self.count += 1
                        return ast.parse("_root_gather(roots,owner)", mode="eval").body
                    return self.generic_visit(item)
            visitor = CountRootGather()
            visitor.visit(node)
            frozen.need(visitor.count == 1, "One original root gather required")
        return frozen.compile_function(function, instrument, _root_gather=counter, **bindings)
    function = frozen.prepare_cut
    result = types.FunctionType(function.__code__, dict(function.__globals__, compile_function=compiling),
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


class RootGather:
    def __init__(self):
        self.calls = 0
        self.elements = 0
    def __call__(self, roots, owner):
        self.calls += 1
        self.elements += owner.size
        return roots[owner]


class RootHoistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_equal(self, expected, actual):
        left, left_meta = expected
        right, right_meta = actual
        self.assertEqual(left_meta, right_meta)
        for field in fields(left):
            if field.name == "tensor_stamps":
                continue  # Each independently built plan owns different tensor pointers.
            a, b = getattr(left, field.name), getattr(right, field.name)
            if torch.is_tensor(a):
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.numpy().tobytes(), b.numpy().tobytes())
            else:
                self.assertEqual(a, b)

    def test_connected_unary_and_isolated_complete_cut_byte_and_metadata_parity(self):
        for isolated in (False, True):
            with self.subTest(isolated=isolated):
                bank, episodes, native_cut, metadata = prepare_original(fixture(isolated))
                kwargs = dict(expected_haplotypes=8, max_nodes=1000, max_holds=10,
                              isolated_hold_indices=(2,) if isolated else ())
                expected = frozen.prepare_cut(bank, **kwargs)
                actual = repaired.prepare_cut(bank, **kwargs)
                self.assert_equal(expected, actual)
                self.assert_equal((native_cut, metadata["cut"]), actual)
                actual[0].validate(bank)

    def test_actual_full_root_gather_changes_from_T_to_one_with_identical_outputs(self):
        for isolated in (False, True):
            with self.subTest(isolated=isolated):
                bank, _, _, _ = prepare_original(fixture(isolated))
                old_count, new_count = RootGather(), RootGather()
                kwargs = dict(expected_haplotypes=8, max_nodes=1000, max_holds=10,
                              isolated_hold_indices=(2,) if isolated else ())
                expected = instrument_original(old_count)(bank, **kwargs)
                actual = repaired.make_prepare_cut(root_gather=new_count)(bank, **kwargs)
                self.assert_equal(expected, actual)
                self.assertEqual(old_count.calls, bank.sample_index.shape[0])
                self.assertEqual(new_count.calls, 1)
                self.assertEqual(old_count.elements, bank.parent.numel() * bank.sample_index.shape[0])
                self.assertEqual(new_count.elements, bank.parent.numel())

    def test_private_runtime_and_both_qualification_fit_bindings_preserve_frozen_sources(self):
        paths = [Path(module.__file__) for module in (frozen, frozen_integration, frozen_qualifier)]
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        function = original.prepare_from_loaded_observation_v1
        class PrivateBundle:
            pass
        private = types.FunctionType(function.__code__, dict(function.__globals__,
            TRAIN=("fresh_train",), VAL=("fresh_test",), AmericanTreeLoomWholeParentV1=PrivateBundle),
            function.__name__, function.__defaults__, function.__closure__)
        runtime = SimpleNamespace(preparation=SimpleNamespace(prepare_from_loaded_observation_v1=private))
        integration.install_runtime_preparation(runtime)
        prepared = runtime.preparation.prepare_from_loaded_observation_v1
        self.assertEqual(prepared.__globals__["TRAIN"], ("fresh_train",))
        self.assertIs(prepared.__globals__["AmericanTreeLoomWholeParentV1"], PrivateBundle)
        self.assertIs(prepared.__globals__["prepare_cut_plan_v1"], repaired.prepare_cut)
        self.assertIs(integration.run_training.__globals__["install_runtime_preparation"],
                      integration.install_runtime_preparation)
        self.assertIs(integration.qualify.__globals__["install_runtime_preparation"],
                      integration.install_runtime_preparation)
        self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths], before)
        self.assertIs(frozen_integration.compiled_complete().__globals__["prepare_cut_plan_v1"],
                      frozen.prepare_cut)


if __name__ == "__main__":
    unittest.main()

"""Focused RAM/source regressions only. No Torch/checkpoint/training/device work."""
from collections import OrderedDict
from pathlib import Path
from types import ModuleType,SimpleNamespace
import unittest
from arggnn.benchmarking import argformer_ordereddict_bridge_v1 as b

class Tensor: pass
TORCH=SimpleNamespace(is_tensor=lambda value:isinstance(value,Tensor))

class CompatibilityTests(unittest.TestCase):
    def test_actual_container_type_is_accepted_without_copying(self):
        tensor=Tensor()
        value=OrderedDict([("bert.weight",tensor)])
        value._metadata={"version":"preserved"}
        self.assertTrue(b.supported_model_state(value,TORCH))
        self.assertIs(value["bert.weight"],tensor)
        self.assertEqual(value._metadata,{"version":"preserved"})
        self.assertTrue(b.supported_model_state(dict(value),TORCH))

    def test_malformed_model_states_still_rejected(self):
        for value in (None,{},OrderedDict(),[],{"x":object()},{1:Tensor()}):
            with self.subTest(value=type(value).__name__):
                self.assertFalse(b.supported_model_state(value,TORCH))
        class OtherMapping(dict):pass
        self.assertFalse(b.supported_model_state(OtherMapping(x=Tensor()),TORCH))

    def test_only_private_namespace_and_original_type_check_change(self):
        root=Path(__file__).resolve().parents[1]
        original=ModuleType("RAM_original")
        original.__file__=str(root/b.ORIGINAL_PATH)
        marker=lambda:None
        original._instrument_training=marker
        private=b.compatible_bridge_v1(original)
        self.assertIs(original._instrument_training,marker)
        self.assertIsNot(private._instrument_training,marker)
        self.assertIs(private._instrument_training.__globals__["_supported_model_state"],
                      b.supported_model_state)
        self.assertNotIn("_supported_model_state",original.__dict__)

if __name__=="__main__":
    unittest.main()

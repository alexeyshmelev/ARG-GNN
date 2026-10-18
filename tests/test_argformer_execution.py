"""Tiny metadata/mock checks; never import Torch or open scientific payloads."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from arggnn.benchmarking import argformer_execution as a


class NativeContrastiveExecutionTests(unittest.TestCase):
    def test_spec_has_full_native_budget_and_exact_sources(self):
        spec = a.build_spec("experiments/results/benchmark_argformer_mock_check/native")
        self.assertEqual(spec["requirements"]["settings"]["updates"], 2135)
        self.assertEqual(spec["requirements"]["settings"]["epochs"], 1)
        self.assertEqual(spec["requirements"]["settings"]["batch"], 4096)
        self.assertEqual(spec["requirements"]["settings"]["temperature"], .30)
        self.assertTrue(spec["requirements"]["settings"]["uniform_class_weight_fallback"])
        self.assertEqual(len(spec["requirements"]["input_refs"]), 5)
        self.assertEqual(a._validate_spec(spec), a.ROOT / spec["output_directory"])
        self.assertFalse(spec["requirements"]["exact_paper_reproduction"])
        self.assertEqual(spec["requirements"]["phase_scope"], "contrastive_only_no_embedding_retrieval_or_scoring")

    def test_changed_budget_temperature_or_checkpoint_reject_before_runtime(self):
        spec = a.build_spec("experiments/results/benchmark_argformer_mock_check/native")
        for field, value in (("updates", 1), ("temperature", .05), ("batch", 512)):
            bad = copy.deepcopy(spec)
            bad["requirements"]["settings"][field] = value
            with self.assertRaises(a.ContractError):
                a._validate_spec(bad)
        bad = copy.deepcopy(spec)
        bad["requirements"]["initializer_checkpoint_ref"]["sha256"] = "0" * 64
        with self.assertRaises(a.ContractError):
            a._validate_spec(bad)

    def test_source_drift_rejects_metadata_plan(self):
        bad = copy.deepcopy(a.PINS)
        bad["integration"]["sha256"] = "0" * 64
        with patch.object(a, "PINS", bad), self.assertRaises(a.ContractError):
            a.requirements()

    def test_existing_or_escaped_output_cannot_be_replayed(self):
        with self.assertRaises(a.ContractError):
            a.build_spec("experiments/results")
        with self.assertRaises(a.ContractError):
            a.build_spec("../elsewhere")
        with self.assertRaises(a.ContractError):
            a.build_spec("experiments/results/American_C3_val000_full30mb_shared_v1")

    @staticmethod
    def state():
        return dict(model={"weight": object()},
                    timestamp={"Timestamp": {"epoch": 1, "batch": 2135}},
                    optimizers={"DecoupledAdamW": {"state": {
                        0: dict(step=2135, exp_avg=object(), exp_avg_sq=object()),
                        1: dict(step=2135., exp_avg=object(), exp_avg_sq=object())}}})

    def test_actual_timestamp_and_all_optimizer_states_are_verified(self):
        state = self.state()
        result = a._checkpoint_counters(state)
        self.assertEqual(result["actual_optimizer_updates"], 2135)
        self.assertEqual(result["optimizer_parameter_state_count"], 2)
        self.assertEqual(result["optimizer_unique_step_values"], [2135])
        state["optimizers"]["DecoupledAdamW"]["state"][1]["step"] = 2134
        with self.assertRaises(a.ContractError):
            a._checkpoint_counters(state)

    def test_filename_cannot_substitute_for_timestamp_or_moments(self):
        for defect in ("timestamp", "moments", "model"):
            state = self.state()
            if defect == "timestamp":
                state["timestamp"]["Timestamp"]["batch"] = 1
            elif defect == "moments":
                del state["optimizers"]["DecoupledAdamW"]["state"][0]["exp_avg"]
            else:
                state["model"] = {}
            with self.assertRaises(a.ContractError):
                a._checkpoint_counters(state)

    def test_partial_checkpoint_with_final_filename_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT / ".runtime/tmp") as tmp:
            directory = Path(tmp)
            (directory / "ep1-ba2135-rank0.pt").write_bytes(b"mock")
            state = self.state()
            state["timestamp"]["Timestamp"]["batch"] = 20
            class MockTorch:
                @staticmethod
                def load(*args, **kwargs):
                    return {"state": state}
            with self.assertRaises(a.ContractError):
                a._final_checkpoint(directory, MockTorch(), lambda *args: None)

    def test_planning_imports_no_scientific_runtime(self):
        code = (
            "import sys;sys.path.insert(0,'src');"
            "from arggnn.benchmarking.argformer_execution import requirements;"
            "requirements();"
            "assert not any(x in sys.modules for x in "
            "('torch','torch_geometric','numpy','composer','transformers'));print('metadata_only')"
        )
        result = subprocess.run([sys.executable, "-B", "-s", "-c", code],
                                cwd=a.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("metadata_only", result.stdout)


if __name__ == "__main__":
    unittest.main()

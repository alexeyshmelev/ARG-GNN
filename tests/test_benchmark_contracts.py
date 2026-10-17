import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from arggnn.benchmarking import contracts as c
from arggnn.benchmarking.cli import main


def protocol():
    return dict(schema="benchmark_protocol_v1", id="fixture",
                comparison_kind="common_data_reconstruction",
                class_order=["A", "B"], split_unit="independent_simulation_parent_before_sampling",
                split_parents=dict(train=["train"], validation=["val"], test=["test"]),
                observations=dict(query_truth_in_features=False),
                evaluation=dict(probability_floor=False,
                                parent_aggregation="equal_parent_mean_sample_sd"),
                unresolved_details=[], methods=[], ready_for_execution=False)


class ProtocolTests(unittest.TestCase):
    def test_parent_leakage_rejected(self):
        p = protocol()
        p["split_parents"]["test"] = ["train"]
        with self.assertRaises(c.ContractError):
            c.validate_protocol(p)

    def test_query_truth_cannot_be_input(self):
        p = protocol()
        p["observations"]["query_truth_in_features"] = True
        with self.assertRaises(c.ContractError):
            c.validate_protocol(p)

    def test_reconstruction_is_not_paper_reproduction(self):
        p = protocol()
        p.update(comparison_kind="paper_reproduction", unresolved_details=["ploidy unknown"])
        with self.assertRaises(c.ContractError):
            c.validate_protocol(p)

    def test_empirical_accuracy_rejected(self):
        p = protocol()
        p.update(comparison_kind="empirical_application")
        p["evaluation"]["supervised_accuracy"] = True
        with self.assertRaises(c.ContractError):
            c.validate_protocol(p)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        temp_root = c.ROOT / ".runtime/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp.name).resolve()
        self.patch = patch.object(c, "ROOT", self.root)
        self.patch.start()
        p = self.root / c.POLICY_PATH
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(benchmark_execution_approved=False)))

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_changed_metadata_rejected(self):
        p = self.root / "data.json"
        p.write_text("{}")
        ref = c.file_ref(p)
        p.write_text('{"changed":true}')
        with self.assertRaises(c.ContractError):
            c.read_ref(ref)

    def test_paths_cannot_escape(self):
        with self.assertRaises(c.ContractError):
            c.contained_path("../outside.json")

    def test_read_ref_hashes_and_parses_one_snapshot(self):
        p = self.root / "data.json"
        p.write_text('{"value":1}')
        ref = c.file_ref(p)
        with patch.object(c, "read_json", side_effect=AssertionError("second read forbidden")):
            self.assertEqual(c.read_ref(ref), {"value": 1})

    def test_malformed_module_argv_rejected_without_index_error(self):
        interpreter = self.root / "python"
        interpreter.write_text("")
        for argv in (["setsid"], [str(interpreter), "-m"], []):
            with self.subTest(argv=argv), self.assertRaises(c.ContractError):
                c.validate_module_argv(argv)

    def test_generic_normalization_is_not_execution_authority(self):
        interpreter = self.root / "python"
        interpreter.write_text("")
        packet = dict(schema="benchmark_admitted_job_v1",
                      argv=[str(interpreter), "-m", "arggnn.benchmarking"])
        # No fabricated returncode, including False==0, can release a real family.
        for code in (0, False, 1):
            packet["actual_direct_returncode"] = code
            with patch.object(c, "read_ref", return_value=packet), self.assertRaisesRegex(
                    c.ContractError, "admission-family adapters are not released"):
                c.validate_admission({"path": "fake", "sha256": "a" * 64})

    def test_hold_blocks_even_forged_approved_policy(self):
        p = self.root / "forged.json"
        p.write_text('{"benchmark_execution_approved":true}')
        with self.assertRaises(c.ContractError):
            c.check_execution_release(c.file_ref(p))

    def test_explicit_release_can_validate_exact_job_without_launching(self):
        from types import SimpleNamespace
        job = SimpleNamespace(job_id="fixture", admission_receipt_ref={"path": "a", "sha256": "b"},
                              input_ref={"path": "c", "sha256": "d"})
        value = dict(benchmark_execution_approved=True, status="approved",
                     physical_gpus=[0], approval_origin="explicit_user_go_ahead",
                     smoke_tests_only=False,
                     expires_at_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                     approved_jobs=[dict(job_id=job.job_id,
                        admission_receipt_ref=job.admission_receipt_ref, input_ref=job.input_ref)])
        p = self.root / c.POLICY_PATH
        p.write_text(json.dumps(value))
        self.assertTrue(c.validate_execution_policy(c.file_ref(p), job))
        job.job_id = "different"
        with self.assertRaises(c.ContractError):
            c.validate_execution_policy(c.file_ref(p), job)

    def test_execute_hold_precedes_queue_creation_or_process(self):
        with patch("arggnn.benchmarking.cli._scheduler",
                   side_effect=AssertionError("must not create a run queue")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", "--execute"]), 2)


if __name__ == "__main__":
    unittest.main()

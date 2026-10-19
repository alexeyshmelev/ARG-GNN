import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("european_archive_v1", HERE / "european_archive_v1.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
ROOT = HERE.parents[3]

class ArchiveTests(unittest.TestCase):
    def test_lossless_replacement_and_durable_manifest(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value)
            source = root / "native.txt"
            payload = b"ancestral genealogy\n" * 100000
            source.write_bytes(payload)
            out = root / "archive"
            out.mkdir()
            entry = a.archive_one(source, out / "file_00.json", 2**20,
                                  lambda event: None, lambda: None)
            self.assertFalse(source.exists())
            self.assertEqual(entry["state"], "archive_committed")
            self.assertEqual(Path(entry["archive_path"]).parent, out)
            self.assertEqual(gzip.open(entry["archive_path"], "rb").read(), payload)
            self.assertEqual(entry["original_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(json.loads((out / "file_00.json").read_text()), entry)
            self.assertGreater(entry["credited_bytes"], 0)

    def test_incompressible_keeps_raw_and_no_credit(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value)
            source = root / "random.bin"
            payload = os.urandom(128 * 1024)
            source.write_bytes(payload)
            entry = a.archive_one(source, root / "file_00.json", 2**20,
                                  lambda event: None, lambda: None)
            self.assertEqual(source.read_bytes(), payload)
            self.assertEqual(entry["credited_bytes"], 0)
            self.assertEqual(entry["state"], "original_kept_no_storage_gain")
            self.assertFalse(Path(entry["archive_path"]).exists())

    def test_cap_failure_preserves_raw_partial_is_in_task_scope(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value)
            source = root / "random.bin"
            payload = os.urandom(2 * 2**20)
            source.write_bytes(payload)
            out = root / "task"
            out.mkdir()
            with self.assertRaises(ValueError):
                a.archive_one(source, out / "file_00.json", 2**20,
                              lambda event: None, lambda: None)
            self.assertEqual(source.read_bytes(), payload)
            self.assertTrue(list(out.glob("*.pending")))
            self.assertLessEqual(sum(p.stat().st_size for p in out.iterdir()), 2**20)

    def test_failure_before_unlink_has_both_raw_and_verified_archive(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value)
            source = root / "source"
            payload = b"constant" * 100000
            source.write_bytes(payload)
            def reject():
                raise RuntimeError("injected post-verification failure")
            with self.assertRaises(RuntimeError):
                a.archive_one(source, root / "file.json", 2**20,
                              lambda event: None, reject)
            manifest = json.loads((root / "file.json").read_text())
            self.assertEqual(source.read_bytes(), payload)
            self.assertEqual(gzip.open(manifest["archive_path"], "rb").read(), payload)
            self.assertEqual(manifest["state"], "verified_archive_original_preserved")

    def test_frozen_original_hash_mismatch_preserves_raw(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value)
            source = root / "source"
            source.write_bytes(b"constant" * 100000)
            with self.assertRaises(ValueError):
                a.archive_one(source, root / "file.json", 2**20,
                              lambda event: None, lambda: None, "0" * 64)
            self.assertTrue(source.exists())
            self.assertFalse((root / "file.json").exists())

    def test_symlink_and_hardlink_not_candidates(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value)
            source = root / "source"
            source.write_bytes(b"one")
            (root / "link").symlink_to(source)
            with self.assertRaises(ValueError):
                a._identity(root / "link")
            os.link(source, root / "hard")
            with self.assertRaises(ValueError):
                a._identity(source)

    def test_task_insertion_preserves_science_and_skips_external_c4(self):
        original = json.loads((ROOT / "configs/benchmarks/executable_workflow_20261019_v8.json").read_text())
        before = copy.deepcopy(original)
        changed, evidence = a.add_archive_tasks(original, "experiments/results/source_only_v9_fixture")
        self.assertEqual(original, before)
        additions = changed["tasks"][:-len(original["tasks"])]
        self.assertEqual(changed["tasks"][len(additions):], original["tasks"])
        expected = sum(t["id"].startswith("AP__") and t["id"].endswith("__ap_infer")
                       and t.get("resource") != "external" and bool(t.get("output_directory"))
                       for t in original["tasks"])
        self.assertEqual(len(additions), expected)
        self.assertEqual(expected, 7)  # V9 builder requeues failed external APtest004, yielding8.
        self.assertTrue(evidence["skipped_external_owner_task_ids"])
        self.assertEqual(evidence["prospective_savings_credited_bytes"], 0)
        for task in additions:
            self.assertEqual(task["max_output_bytes"], a.MAX_OUTPUT)
            self.assertEqual(task["remeasure_succeeded_tasks"], task["dependencies"])
            self.assertTrue(task["id"].startswith("AP__"))

    def test_wait_authenticates_job_ref_instead_of_absent_job_id(self):
        # Actual original workflow metadata only; never runs compression.
        plan = json.loads((ROOT / "configs/benchmarks/executable_workflow_20261019_v8.json").read_text())
        task = next(x for x in plan["tasks"] if x["id"] == "AP__ancestralpaths_european_train_000__ap_infer")
        owner = a._path(ROOT, task["output_directory"])
        spec = dict(schema="closed_european_lossless_archive_v1",
                    owner_scope=str(owner), owner_task_id=task["id"],
                    inference_ref=a._ref(ROOT, owner / "native/inference.json"),
                    actual_wait_ref=a._ref(ROOT, owner / "actual_wait.json"),
                    max_temporary_bytes=a.MAX_TEMP, max_output_bytes=a.MAX_OUTPUT,
                    consumer_source_refs=[a._ref(ROOT, p) for p in a.CONSUMER_SOURCES],
                    root_current_scope_writer_revocation_required=True)
        self.assertNotIn("job_id", a._read(ROOT, spec["actual_wait_ref"]))
        inference, actual_owner = a.validate(spec, ROOT)
        self.assertEqual(actual_owner, owner)
        self.assertFalse(inference["truth_or_oracle_read"])
        spec["owner_task_id"] = "wrong_task"
        with self.assertRaises(ValueError):
            a.validate(spec, ROOT)
        spec["owner_task_id"] = task["id"]
        spec["consumer_source_refs"] = []
        with self.assertRaises(ValueError):
            a.validate(spec, ROOT)

if __name__ == "__main__":
    unittest.main()

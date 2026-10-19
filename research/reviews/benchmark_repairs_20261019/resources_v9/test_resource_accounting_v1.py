import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("resource_accounting_v1", HERE / "resource_accounting_v1.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
ROOT = HERE.parents[3]

class ResourceAccountingTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "configs/benchmarks/executable_workflow_20261019_v8.json").read_text())

    def test_copy_and_exact_settlement(self):
        before = copy.deepcopy(self.plan)
        changed, report = r.apply_resource_corrections(self.plan, ROOT)
        self.assertEqual(self.plan, before)
        self.assertEqual(report["native_storage_after_bytes"], 52308926488)
        self.assertEqual(report["released_native_storage_bytes"], 46475321320)
        self.assertFalse(report["activated"])
        self.assertEqual(changed["storage_baseline_ref"], before["storage_baseline_ref"])
        self.assertEqual(changed["external_storage"][:6], before["external_storage"][:6])
        self.assertEqual(changed["external_storage"][7:], before["external_storage"][7:])
        self.assertEqual(changed["historical_unknown_acceptance_ref"], before["historical_unknown_acceptance_ref"])
        self.assertFalse(changed["resource_accounting_v9"]["historical_waits_and_group_absence_relabelled"])
        self.assertEqual(r.validate_static_settlements(changed, ROOT), 52308926488)

    def test_rejects_weakened_storage_ceiling(self):
        self.plan["storage_ceiling_bytes"] = 1_000_000_000_000
        with self.assertRaises(ValueError):
            r.apply_resource_corrections(self.plan, ROOT)

    def test_rejects_unexpected_legacy_or_duplicate_scope(self):
        self.plan["prior_unknown_rss_bytes"] = 0
        with self.assertRaises(ValueError):
            r.apply_resource_corrections(self.plan, ROOT)
        self.plan["prior_unknown_rss_bytes"] = r.OLD_FIXED_RSS
        self.plan["external_storage"].append(copy.deepcopy(self.plan["external_storage"][6]))
        with self.assertRaises(ValueError):
            r.apply_resource_corrections(self.plan, ROOT)

    def test_retained_charge_cannot_be_changed(self):
        changed, _ = r.apply_resource_corrections(self.plan, ROOT)
        changed["external_storage"][6]["cap_bytes"] -= 1
        with self.assertRaises(ValueError):
            r.validate_static_settlements(changed, ROOT)

    def test_scope_growth_changes_manifest_and_overlap_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value)
            (root / "scope").mkdir()
            (root / "scope/a").write_bytes(b"one")
            first = r.measure_scopes(root, ["scope"])
            (root / "scope/a").write_bytes(b"changed")
            self.assertNotEqual(first, r.measure_scopes(root, ["scope"]))
            with self.assertRaises(ValueError):
                r.measure_scopes(root, ["scope", "scope"])

    def test_metadata_ref_tamper_fails(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value)
            (root / "proof.json").write_text('{"status":"succeeded"}')
            reference = r.file_ref(root, "proof.json")
            (root / "proof.json").write_text('{"status":"failed"}')
            with self.assertRaises(ValueError):
                r._read_ref(root, reference)

    def test_ram_full_cap_and_margin_boundary_no_pss(self):
        cap = 128 * 2**30
        yes = r.ram_admission(cap + cap + r.MARGIN, [cap], cap)
        no = r.ram_admission(cap + cap + r.MARGIN - 1, [cap], cap)
        self.assertTrue(yes["allowed"])
        self.assertFalse(no["allowed"])
        self.assertEqual(yes["pss_credit_bytes"], 0)
        self.assertFalse(r.ram_admission(10**15, [1280 * 2**30], 1)["allowed"])

    def test_memavailable_parser_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            path = Path(value) / "meminfo"
            path.write_text("MemAvailable: 100 kB\n")
            self.assertEqual(r.kernel_available_ram_bytes(path), 102400)
            path.write_text("MemAvailable: 100 kB\nMemAvailable: 200 kB\n")
            with self.assertRaises(ValueError):
                r.kernel_available_ram_bytes(path)

if __name__ == "__main__":
    unittest.main()

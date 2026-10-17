"""Narrow CPU/stdlib checks; no datasets, author numerics, devices or real jobs."""
import ast
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from arggnn.benchmarking import methods, treeloom


def artifact(name):
    # A deliberately nonexistent artifact: planning must not read payloads.
    return {"path": ".runtime/tmp/benchmark_fake_" + name + ".json", "sha256": "a" * 64}


class MethodRegistryTests(unittest.TestCase):
    def test_registry_is_lazy_in_a_fresh_interpreter(self):
        code = (
            "import sys; sys.path.insert(0, " + repr(str(methods.ROOT / "src")) + "); "
            "from arggnn.benchmarking import methods,treeloom; "
            "assert len(methods.list_methods())==6; "
            "assert 'torch' not in sys.modules; "
            "assert 'tensorflow' not in sys.modules; "
            "assert 'torch_geometric' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-s", "-c", code],
                                cwd=methods.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_descriptors_have_real_source_pinned_entrypoints(self):
        self.assertEqual([m["id"] for m in methods.list_methods()],
                         ["treeloom", "argformer", "argmix", "flare", "ancestralpaths", "gst"])
        for method in methods.list_methods():
            self.assertFalse(method["fidelity"]["exact_paper_reproduction"])
            for protocol in method["protocol_ids"]:
                pipeline = method.get("pipeline_by_protocol", {}).get(protocol, method["pipeline"])
                for name in pipeline:
                    self.assertIn(protocol, method["stages"][name].get(
                        "compatible_protocols", method["protocol_ids"]))
            methods.verify_method_sources(method["id"])
            for stage in method["stages"].values():
                entry = stage["entrypoint"]
                if entry["kind"] == "callable":
                    tree = ast.parse((methods.ROOT / entry["source_ref"]["path"]).read_text())
                    self.assertIn(entry["callable"],
                                  {node.name for node in tree.body if isinstance(
                                      node, (ast.FunctionDef, ast.ClassDef))})

    def test_callers_cannot_mutate_registry_state(self):
        descriptor = methods.get_method("argmix")
        descriptor["pipeline"].clear()
        self.assertTrue(methods.get_method("argmix")["pipeline"])

    def test_duplicate_method_ids_and_unsubstantiated_paper_claim_reject(self):
        registry = methods.load_registry()
        duplicate = copy.deepcopy(registry)
        duplicate["methods"].append(copy.deepcopy(duplicate["methods"][0]))
        with patch.object(Path, "read_text", return_value=json.dumps(duplicate)):
            with self.assertRaisesRegex(ValueError, "unique"):
                methods.load_registry()
        unsupported = copy.deepcopy(registry)
        unsupported["methods"][0]["fidelity"]["exact_paper_reproduction"] = True
        with patch.object(Path, "read_text", return_value=json.dumps(unsupported)):
            with self.assertRaisesRegex(ValueError, "verified evidence"):
                methods.load_registry()

    def test_unknown_and_incompatible_protocols_reject(self):
        with self.assertRaises(KeyError):
            methods.get_method("invented_method")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            methods.build_job_spec("ancestralpaths", "fit", protocol_id="american_c3_full30mb",
                                   artifacts={}, output_directory=".runtime/tmp/benchmark_fake_output")
        with self.assertRaisesRegex(ValueError, "no implementation"):
            methods.build_job_spec("argmix", "prepare_observation",
                                   protocol_id="european_c4_expanded", artifacts={},
                                   output_directory=".runtime/tmp/benchmark_fake_output")

    def test_argmix_plan_uses_unchanged_author_training_and_gpu0_policy(self):
        method = methods.get_method("argmix")
        refs = {key: artifact(key) for key in method["stages"]["fit"]["required_artifacts"]}
        plan = methods.build_job_spec("argmix", "fit", protocol_id="european_c4_prospective1611",
                                     artifacts=refs, output_directory=".runtime/tmp/benchmark_fake_output")
        self.assertIn("/.runtime/author_sources/argmix_c2481248eb0d/src/train_joint_model.py",
                      plan["argv"][3])
        self.assertIn("--train-pt", plan["argv"])
        self.assertIn("--val-pt", plan["argv"])
        self.assertNotIn("expanded_native_argmix_gpu_fit_root", " ".join(plan["argv"]))
        self.assertEqual(plan["required_gpu_policy"]["physical_gpu"], 0)
        self.assertFalse(plan["execution_authorized"])
        self.assertTrue(plan["requires_fresh_admission"])
        self.assertTrue(plan["requires_later_explicit_user_approval"])
        self.assertFalse(plan["missing_artifacts"])
        self.assertTrue(plan["blockers"])

    def test_missing_artifacts_do_not_produce_executable_argv(self):
        plan = methods.build_job_spec("argformer", "mlm", protocol_id="american_c3_full30mb",
                                     artifacts={}, output_directory=".runtime/tmp/benchmark_fake_output")
        self.assertIsNone(plan["argv"])
        self.assertIn("native_MLM_config_ref", plan["missing_artifacts"])
        self.assertFalse(plan["execution_authorized"])

    def test_artifact_paths_cannot_override_command_fields_or_escape_repository(self):
        for refs in ({"python": artifact("python")},
                     {"bad_ref": {"path": "../outside.json", "sha256": "a" * 64}}):
            with self.assertRaises(ValueError):
                methods.build_job_spec("argmix", "fit", protocol_id="european_c4_prospective1611",
                                       artifacts=refs, output_directory=".runtime/tmp/benchmark_fake_output")

    def test_source_drift_fails_before_import(self):
        descriptor = methods.get_method("treeloom")
        with patch.object(Path, "read_bytes", return_value=b"modified source"):
            with self.assertRaisesRegex(ValueError, "source changed"):
                methods.verify_method_sources(descriptor["id"])

    def test_treeloom_uses_exact_reviewed_manifest(self):
        self.assertEqual(treeloom.MANIFEST_REF["sha256"],
                         "0b0013f0907ef1b9a2b7b37dda1e59230111761d19a55821e9c6489adde5c4eb")
        self.assertEqual(treeloom.verify_implementation()["history_holds_per_window"], 128)

    def test_treeloom_training_delegates_without_reimplementing_equations(self):
        executor = Mock()
        executor.train_step_v1.return_value = {"fake": "test-only"}
        guard = Mock()
        with patch.object(treeloom, "verify_implementation"), patch.object(
                treeloom.importlib, "import_module", return_value=executor) as load:
            result = treeloom.train_step("model", "bundle", "targets", "optimizer", "metadata",
                                        resource_check=guard)
        load.assert_called_once_with(treeloom.EXECUTION_MODULE)
        executor.train_step_v1.assert_called_once_with(
            "model", "bundle", "targets", "optimizer", "metadata",
            resource_check=guard, history_holds_per_window=128, profile=None)
        self.assertEqual(result, {"fake": "test-only"})

    def test_european_treeloom_plan_binds_real_c4_math_without_c3_authority(self):
        plan = methods.build_job_spec(
            "treeloom", "train_european_step",
            protocol_id="european_c4_prospective1611", artifacts={},
            output_directory=".runtime/tmp/benchmark_fake_output")
        self.assertEqual(plan["entrypoint"]["callable"], "train_european_step")
        self.assertEqual(plan["stage_configuration"]["output_classes"], 4)
        self.assertEqual(plan["stage_configuration"]["execution_mode"], "bounded_history")
        self.assertIn("full_parent_GPU0_fit_qualification_ref", plan["missing_artifacts"])
        self.assertNotIn("bounded_execution_v1", plan["entrypoint"]["module"])
        self.assertFalse(plan["execution_authorized"])
        self.assertEqual(treeloom.EUROPEAN_CONDITIONS["european_c4_expanded"],
                         "benchmark_european_c4_expanded")

    def test_gst_is_an_architecture_control_with_no_fake_expanded_fit(self):
        control = methods.get_method("gst")
        self.assertEqual(control["comparison_role"], "architecture_control")
        self.assertFalse(control["historical_fitting_entrypoint"]["portable_to_expanded_protocol"])
        self.assertNotIn("fit", control["stages"])
        self.assertFalse(control["fidelity"]["exact_paper_reproduction"])

    def test_european_protocols_are_truthful_and_blocked(self):
        for identifier in ("european_c4_prospective1611", "european_c4_expanded"):
            protocol = json.loads((methods.ROOT / "configs/benchmarks/protocols" /
                                   (identifier + ".json")).read_text())
            self.assertFalse(protocol["ready_for_execution"])
            self.assertIn("treeloom", protocol["methods"])
            self.assertEqual(protocol["split_unit"], "independent_simulation_parent_before_sampling")
            self.assertFalse(protocol["observations"]["query_truth_in_features"])
            self.assertFalse(protocol["evaluation"]["probability_floor"])
            groups = [set(rows) for rows in protocol["split_parents"].values()]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            for identifier in protocol["methods"]:
                self.assertIn(protocol["id"], methods.get_method(identifier)["protocol_ids"])


if __name__ == "__main__":
    unittest.main()

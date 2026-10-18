"""Workflow planning and fake native waits, without simulations or training."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import inspect
import tempfile
import unittest
import numpy as np

from arggnn.benchmarking import european_workflow as workflow
from arggnn.benchmarking import european_sampling as sampling
from arggnn.benchmarking import european_native_models as models


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".european_workflow_tests_", dir=models.ROOT / "tests")
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_native6_DAG_has_real_stages_and_complete_prediction_barrier(self):
        from arggnn.benchmarking import european_training
        parents = []
        for split, n in (("train", 3), ("test", 5)):
            for i in range(n):
                index = len(parents)
                parents.append(dict(parent_id=split + str(i), split=split, seeds={
                    k: 1000 + index * 100 + j * 10
                    for j, k in enumerate(("ancestry", "mutations", "relate_initial", "relate_size_base"))}))
        plan = sampling.paper_sampling_plan(
            [p["parent_id"] for p in parents if p["split"] == "train"],
            [p["parent_id"] for p in parents if p["split"] == "test"], seed=1)
        stages = (*workflow.STAGES, "treeloom_fit", "treeloom_predict")
        limits = {s: dict(max_seconds=10, max_output_bytes=1024**2,
                          max_rss_bytes=1024**3,
                          minimum_free_gpu_bytes=1024**3,
                          max_cuda_reserved_bytes=1024**3) for s in stages}
        shared = dict(max_rss_bytes=1024**3, preparation_headroom_bytes=100,
                      observation_cache_bytes=100, max_sites=1000,
                      max_metadata_bytes=1024**2, preparation_limits=dict(
                          max_nodes=1000, max_holds=1000,
                          max_episode_payload_bytes=1024**2))
        parameters = dict(
            fit=dict(shared, training_headroom_bytes=100),
            predict=dict(shared, prediction_headroom_bytes=100))
        with patch.object(models, "_read", return_value=plan):
            tasks = workflow.build_tasks(
                prefix="syntheticAP", output_root="experiments/results/synthetic_AP_workflow",
                parent_records=parents, sampling_plan_ref={"path": "plan", "sha256": "a"*64},
                native_config_ref={"path": "native", "sha256": "b"*64},
                python_data=".runtime/env/bin/python",
                python_tensorflow=".runtime/conda/envs/ancestralpaths_cpu_v1/bin/python",
                stage_limits=limits, stage_parameters=dict(
                    write_headroom_bytes=1, max_metadata_bytes=1024**2,
                    max_feature_bytes=32*1024**2, max_truth_uncompressed_bytes=1024**3,
                    native_training_seed=101, native_tensorflow_threads=2),
                treeloom_parameters=parameters)
        self.assertEqual(len(tasks), 44)
        ids = {t["id"] for t in tasks}
        self.assertEqual(len(ids), 44)
        self.assertTrue(all(set(t["dependencies"]) <= ids for t in tasks))
        score = next(t for t in tasks if t["factory_arguments"].get("stage") == "ap_score")
        self.assertEqual(len(score["dependencies"]), 6)  # five Dense plus full TreeLoom batch
        for t in tasks:
            if t["factory_module"].endswith("european_training"):
                fn = getattr(european_training, t["factory_function"])
                inspect.signature(fn).bind(
                    **t["factory_arguments"], output_directory=t["output_directory"] + "/native")
        labels = [t for t in tasks if t["factory_arguments"].get("stage") == "ap_train_targets"]
        self.assertEqual(len(labels), 3)
        self.assertTrue(all(len(t["dependencies"]) == 3 for t in labels))

    def test_native_step_bridge_records_real_wait_fields_without_starting_process(self):
        log = self.base / "log"
        work = self.base
        (work / "tmp").mkdir()
        calls = []
        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            Path(kwargs["log_path"]).write_text("fake runner log")
            return dict(actual_direct_wait=True, actual_direct_returncode=0,
                        native_process_absent=True, pid=123)
        step = workflow.native_step_bridge(runner, execution_check=lambda: None)
        journal = []
        step(["FAKE_NATIVE"], work, journal, "synthetic", dict(
            native_step_headroom_bytes=1, relate=dict(timeout_seconds_per_step=4)),
            lambda _: None)
        self.assertEqual(journal[0]["status"], "complete")
        self.assertTrue(journal[0]["actual_wait"]["actual_direct_wait"])
        self.assertEqual(calls[0][1]["limits"]["max_seconds"], 4)
        self.assertEqual(calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue((work / "commands.json").is_file())

    def test_common_metric_zero_probability_is_not_clipped(self):
        p = np.array([[1.,0,0,0,0,0], [1.,0,0,0,0,0]], dtype=np.float32)
        y = np.array([0,1], dtype=np.int64)
        result = models._native_metrics(p, y)
        self.assertEqual(result["accuracy"], .5)
        self.assertIsNone(result["raw_NLL"])
        self.assertTrue(result["raw_NLL_is_infinite"])
        self.assertFalse(result["probabilities_clipped_or_renormalized"])
        self.assertEqual(result["confusion"][1][0], 1)

    def test_missing_method_predictions_block_before_private_truth(self):
        plan = dict(test_parent_ids=["test" + str(i) for i in range(5)])
        seen = []
        def read(record, **kwargs):
            seen.append(record)
            return plan
        with patch.object(models, "_read", side_effect=read):
            with self.assertRaisesRegex(ValueError, "Every method"):
                models.score_native_comparison(
                    "plan", {"treeloom": {}}, {}, "unused",
                    guard=lambda _: None, execution_check=lambda: None,
                    write_headroom_bytes=1, max_truth_uncompressed_bytes=1)
        self.assertEqual(seen, ["plan"])

    def test_registered_generator_changes_only_split_membership_constant(self):
        original = workflow.recipes.generate_ancestralpaths_parent
        private = workflow.registered_generator()
        self.assertIsNot(private, original)
        # Branch offsets can differ when a larger literal changes compiler
        # layout. Compare the compiled AST after normalizing only that guard.
        import ast
        import builtins
        import copy
        import textwrap
        captured = []
        actual_compile = builtins.compile
        def capture(source, *args, **kwargs):
            if isinstance(source, ast.Module):
                captured.append(copy.deepcopy(source))
            return actual_compile(source, *args, **kwargs)
        with patch("builtins.compile", side_effect=capture):
            workflow.registered_generator()
        self.assertEqual(len(captured), 1)
        changed = ast.parse('parent.get("split") not in {"train", "evaluation", "test", "validation"}',
                            mode="eval").body
        restored = ast.parse('parent.get("split") not in {"train", "evaluation"}',
                             mode="eval").body
        count = []
        class RestoreGuard(ast.NodeTransformer):
            def visit_Compare(self, node):
                if ast.dump(node, include_attributes=False) == ast.dump(changed, include_attributes=False):
                    count.append(1)
                    return copy.deepcopy(restored)
                return self.generic_visit(node)
        normalized = RestoreGuard().visit(captured[0])
        expected = ast.parse(textwrap.dedent(inspect.getsource(original)))
        self.assertEqual(count, [1])
        self.assertEqual(ast.dump(normalized, include_attributes=False),
                         ast.dump(expected, include_attributes=False))
        old_sets = [c for c in original.__code__.co_consts if isinstance(c, frozenset)]
        new_sets = [c for c in private.__code__.co_consts if isinstance(c, frozenset)]
        self.assertIn(frozenset(("train", "evaluation")), old_sets)
        self.assertIn(frozenset(("train", "evaluation", "test", "validation")), new_sets)
        self.assertNotIn(frozenset(("train", "evaluation", "test", "validation")), old_sets)

    def test_generate_dispatch_owns_empty_directory_and_keeps_fresh_test_identity(self):
        output = self.base / "generation"
        parent = dict(parent_id="new_test_parent", split="test", seeds=dict(
            ancestry=1, mutations=2, relate_initial=3, relate_size_base=4))
        calls = []
        def generate(recipe, *, parent, output, **kwargs):
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(parent["split"], "test")
            self.assertEqual(parent["seeds"], {"ancestry": 1, "mutations": 2})
            calls.append(parent)
            (output / "generation.json").write_text("{}")
            return dict(parent_id=parent["parent_id"], split=parent["split"],
                sequence_length_bp=200000000,
                files={name: {"sha256": "a"*64}
                       for name in [*("observed/" + n for n in workflow.native.OBSERVED_FILES),
                                    "truth/target_schema.json", "truth/full_truth_intervals.npz"]})
        with patch.object(workflow.recipes, "load_recipe", return_value={}), \
             patch.object(workflow.recipes, "verify_sources"), \
             patch.object(workflow, "registered_generator", return_value=generate):
            result = workflow.execute_stage(dict(
                stage="ap_generate", output_directory=str(output),
                arguments=dict(parent=parent, write_headroom_bytes=1)),
                resource_check=lambda _: None, execution_check=lambda: None)
        self.assertEqual(result["split"], "test")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

"""One focused synthetic campaign-to-producer test; no scientific imports."""
import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from arggnn.benchmarking import american_preparation as preparation


class AmericanFreshPreparationTest(unittest.TestCase):
    def test_fresh_shared_splits_caps_and_private_truth_use_real_writer_interface(self):
        source_root = preparation.ROOT
        def reference(path):
            return preparation._reference(path)
        base = dict(
            schema="american_fresh_preparation_v1",
            campaign_ref=reference("configs/benchmarks/campaign.json"),
            comparison_ref=reference("configs/benchmarks/paper_comparisons/argformer_american.json"),
            recipe_ref=reference("configs/benchmarks/recipes/argformer_american.json"),
            declared_conventions=copy.deepcopy(preparation.CONVENTIONS),
            max_sites=1000000, max_tables_bytes=2000000000, max_truth_intervals=100000,
            max_output_bytes=6500000000, max_seconds=3600, max_rss_bytes=16 * 2**30)
        campaign = preparation._read(base["campaign_ref"])
        tasks = [t for t in campaign["tasks"] if t.get("recipe") == "argformer_american"
                 and t.get("stage") == "simulate_once"]
        # Source-only planning authenticates the actual 20/5/10 registry and all
        # fresh streams. No numerical import or old parent reuse is permitted.
        with patch.object(preparation.importlib, "import_module", side_effect=AssertionError("No numerical import")):
            for task in tasks:
                spec = dict(base, task_id=task["id"],
                    output_directory=str(source_root / "experiments/results/american_preparation_test_inert" /
                                         task["shared_parent_id"]))
                plan = preparation.prepare(spec)
                cfg = plan["config"]
                self.assertEqual(cfg["parent_id"], task["shared_parent_id"])
                self.assertEqual(cfg["split"], preparation.SPLITS[task["split"]])
                self.assertEqual(cfg["ancestry_seed"], task["seeds"]["ancestry"])
                self.assertEqual(cfg["mutation_seed"], task["seeds"]["mutations"])
                self.assertEqual(cfg["sequence_length_bp"], 30000000)
                self.assertEqual(cfg["sample_counts"], {"AFR": 200, "EUR": 200, "ASIA": 200, "ADMIX": 1000})
                self.assertEqual((cfg["mutation_rate"], cfg["recombination_rate"]), (1.25e-8, 1e-8))
                self.assertFalse(plan["exact_paper_reproduction"])
                self.assertFalse(plan["execution_authorized"])
        selected = next(t for t in tasks if t["split"] == "test")
        spec = dict(base, task_id=selected["id"], output_directory=str(
            source_root / "experiments/results/american_preparation_test_inert" / selected["shared_parent_id"]))
        spec_bad = copy.deepcopy(spec)
        spec_bad["declared_conventions"]["sample_unit"] = "diploid"
        with self.assertRaisesRegex(ValueError, "conventions"):
            preparation.prepare(spec_bad)

        # Only tiny fake metadata are written, in a repository temporary
        # directory. The actual I/O adapter is exercised by a fake execute(cfg,w).
        with tempfile.TemporaryDirectory(dir=source_root / "tests", prefix="american_prep_") as directory:
            root = Path(directory)
            output = root / "experiments/results/fresh" / selected["shared_parent_id"]
            plan = dict(
                config=dict(parent_id=selected["shared_parent_id"], split="test",
                            sample_counts={"AFR": 200, "EUR": 200, "ASIA": 200, "ADMIX": 1000}),
                task=selected, output_directory=str(output), paper_known_parameters={"length": 30000000},
                declared_conventions=copy.deepcopy(preparation.CONVENTIONS))
            events, attempts = [], []
            def fake_execute(config, writer):
                attempts.append(config)
                secret = writer.json("truth/secret.json", {"query_ancestry": "FAKE_PRIVATE"})
                observation = writer.json("observed/manifest.json", {"query_ancestry_truth_in_inputs": False})
                return writer.json("completed.json", dict(
                    status="completed_C3_simulation_reconstruction", parent_id=config["parent_id"],
                    observation_ref=observation, private_truth_ref=secret, sites=2))
            fake = types.SimpleNamespace(execute=fake_execute, storage_bound=lambda cfg, counts: 1000)
            local_spec = dict(spec, output_directory=str(output))
            with patch.object(preparation, "ROOT", root), \
                 patch.object(preparation, "prepare", return_value=plan), \
                 patch.object(preparation.importlib, "import_module", return_value=fake), \
                 patch.object(Path, "cwd", return_value=root), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                result_ref = preparation.run(local_spec,
                    resource_check=lambda stage: events.append(stage),
                    execution_check=lambda: events.append("approved"))
                public = preparation._read(result_ref)
                self.assertEqual(len(attempts), 1)
                self.assertEqual(events[0], "approved")
                self.assertTrue((output / "private/truth/secret.json").is_file())
                self.assertTrue((output / "private/generation_completed.json").is_file())
                self.assertNotIn("private_truth_ref", public)
                self.assertNotIn("private_true_tree_ref", public)
                self.assertNotIn("private/", json.dumps(public))
                self.assertTrue(public["requires_tsinfer_tsdate"])
                self.assertFalse(public["query_truth_in_training_inputs"])
                self.assertFalse(public["truth_scoring_authorized"])
                self.assertEqual(public["split"], "test")
                # No automatic retry of an existing parent.
                with self.assertRaises(FileExistsError):
                    preparation.run(local_spec, resource_check=Mock(), execution_check=Mock())
                self.assertEqual(len(attempts), 1)
                # A failed count/output bound prevents any scientific attempt.
                rejected = dict(local_spec, max_output_bytes=999)
                with self.assertRaisesRegex(ValueError, "production bounds"):
                    preparation.run(rejected, resource_check=Mock(), execution_check=Mock())
                self.assertEqual(len(attempts), 1)


if __name__ == "__main__":
    unittest.main()

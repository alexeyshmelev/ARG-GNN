"""Synthetic public generation -> inference -> native format/focal chain."""
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from arggnn.benchmarking import american_observation as observation
from arggnn.benchmarking import american_preparation as generation


class FreshObservationTest(unittest.TestCase):
    def test_same_parent_truth_blind_capped_direct_chain(self):
        # Authenticate real source bytes only. No data or scientific runtime.
        generation._verify_sources()
        for ref in observation.SOURCE_REFS:
            self.assertEqual(generation._reference(ref["path"]), ref)
        original_root = generation.ROOT
        with tempfile.TemporaryDirectory(dir=original_root / "tests", prefix="american_observe_") as directory:
            root = Path(directory)
            parent = "argformer_american_test_000"
            scope = root / "experiments/results/fresh" / parent
            source = scope / "generation"
            source.mkdir(parents=True)
            with patch.object(generation, "ROOT", root), patch.object(observation, "ROOT", root):
                def write(path, value):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value))
                    return generation._reference(path)

                manifest = dict(
                    schema="american_c3_observed_v1", parent_id=parent, split="test",
                    class_order=["EUR", "EAS", "AFR"], sequence_length_bp=30000000,
                    genealogy_kind="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG",
                    query_ancestry_truth_in_inputs=False,
                    migrations_and_internal_population_labels_removed=True)
                manifest_ref = write(source / "observed/manifest.json", manifest)
                public = dict(
                    schema="american_fresh_shared_observation_ready_v1", parent_id=parent, split="test",
                    observation_ref=manifest_ref, requires_tsinfer_tsdate=True,
                    query_truth_in_training_inputs=False, truth_scoring_authorized=False,
                    exact_paper_reproduction=False, declared_conventions=copy.deepcopy(generation.CONVENTIONS),
                    seeds={"sampling": 12345}, paper_known_parameters={"length": 30000000},
                    campaign_ref={"path": "FROZEN_CAMPAIGN", "sha256": "a" * 64},
                    comparison_ref={"path": "FROZEN_DESIGN", "sha256": "b" * 64},
                    recipe_ref={"path": "FROZEN_RECIPE", "sha256": "c" * 64})
                public_ref = write(source / "observation_ready.json", public)
                base = dict(schema="american_fresh_observation_stage_v1",
                    max_output_bytes=64 * 2**20, max_seconds=3600, max_rss_bytes=16 * 2**30)
                inference_spec = dict(base, stage="shared_inference", input_ref=public_ref,
                    output_directory=str(scope / "inference"), max_sites=1000,
                    max_tables_bytes=100000, max_node_map_bytes=100000, max_sample_store_bytes=100000,
                    declared_inference_conventions=copy.deepcopy(observation.INFERENCE_CONVENTIONS))
                calls, reads = [], []
                original_read = generation._read
                def recorded_read(ref):
                    reads.append(ref["path"])
                    return original_read(ref)

                def infer(config, writer):
                    calls.append(("infer", copy.deepcopy(config)))
                    self.assertEqual(config["observation_ref"], manifest_ref)
                    self.assertEqual(config["num_threads"], 8)
                    self.assertEqual(config["mutation_rate"], 1.25e-8)
                    result = dict(manifest, genealogy_kind=observation.GENEALOGY,
                                  sites_and_sample_identities_verified=True)
                    ref = writer.json("observed/manifest.json", result)
                    return writer.json("completed.json", dict(
                        status="completed_current_C3_inference", observation_ref=ref))

                def native(config, writer):
                    calls.append(("native", copy.deepcopy(config)))
                    self.assertEqual(config["genealogy_kind"], observation.GENEALOGY)
                    self.assertEqual(config["query_focal_count"], 10000)
                    self.assertEqual(config["path_sampling_seed"], 12345)
                    vocab = writer.pickle("vocab.pkl", {"synthetic": True}, 65536)
                    dataset = writer.pickle("mlm/train/raw_train_sequences_and_vocab.pkl",
                                            {"synthetic": True}, 65536)
                    focals = writer.json("focals_metadata_synthetic.json", {"no_scientific_data": True})
                    return writer.json("completed.json", dict(
                        status="completed_native_C3_path_preparation", parent_id=parent,
                        observation_ref=config["observation_ref"], genealogy_kind=observation.GENEALOGY,
                        query_ancestry_truth_opened=False, query_focal_count=10000,
                        contrastive_labels_reference_only=True, native_dataset_refs={"mlm/train": dataset},
                        focals_ref=focals, vocab_ref=vocab))

                modules = {
                    "arggnn.american_c3_observe_v3": types.SimpleNamespace(execute=infer),
                    "arggnn.american_c3_paths_v1": types.SimpleNamespace(execute=native, storage_bound=lambda cfg: 1000)}
                with patch.object(generation, "_verify_sources"), \
                     patch.object(observation, "SOURCE_REFS", ()), \
                     patch.object(generation, "_read", side_effect=recorded_read), \
                     patch.object(observation.importlib, "import_module", side_effect=lambda name: modules[name]), \
                     patch.object(Path, "cwd", return_value=root), \
                     patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                    approved = Mock()
                    inferred_ref = observation.run_inference(
                        inference_spec, resource_check=Mock(), execution_check=approved)
                    approved.assert_called_once()
                    inferred = original_read(inferred_ref)
                    self.assertFalse(inferred["requires_tsinfer_tsdate"])
                    self.assertEqual(inferred["parent_id"], parent)
                    native_spec = dict(base, stage="native_preparation", input_ref=inferred_ref,
                        output_directory=str(scope / "native"), max_unique_paths=1000,
                        max_path_tokens=10000, max_path_occurrences=100000, max_vocabulary_nodes=10000,
                        max_raw_path_nodes=1024,
                        declared_focal_conventions=copy.deepcopy(observation.FOCAL_CONVENTIONS))
                    native_ref = observation.run_native_preparation(
                        native_spec, resource_check=Mock(), execution_check=Mock())
                    ready = original_read(native_ref)
                    self.assertEqual(ready["schema"], "american_fresh_native_preparation_ready_v1")
                    self.assertEqual(ready["observation_ref"], inferred["observation_ref"])
                    self.assertFalse(ready["query_truth_in_training_inputs"])
                    self.assertFalse(ready["truth_scoring_authorized"])
                    self.assertIn("focals_ref", ready)
                    self.assertIn("native_dataset_refs", ready)
                    self.assertTrue((scope / "native/vocab.pkl").is_file())
                    self.assertEqual([item[0] for item in calls], ["infer", "native"])
                    with self.assertRaisesRegex(ValueError, "Fresh isolated"):
                        observation.run_inference(inference_spec, resource_check=Mock(), execution_check=Mock())
                    wrong = dict(inference_spec, input_ref={"path": "private/truth/secret.json", "sha256": "d" * 64})
                    with self.assertRaisesRegex(ValueError, "public observed"):
                        observation.prepare(wrong)
                    self.assertEqual([item[0] for item in calls], ["infer", "native"])
                    self.assertFalse(any("/truth/" in path or "/private/" in path for path in reads))


if __name__ == "__main__":
    unittest.main()

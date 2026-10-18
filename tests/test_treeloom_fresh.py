"""Fresh parent plumbing with synthetic CPU data. No training or CUDA calls."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib
import json
import random
import tempfile
import unittest

import numpy as np
from arggnn.american_c3_common_v1 import ref
from arggnn.benchmarking import treeloom_fresh as fresh


class FreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix=".treeloom_fresh_tests_", dir=fresh.ROOT / "tests")
        cls.base = Path(cls.temporary.name)
        cls.payload = dict(
            schema=fresh.SCHEMA, split_unit="independent_simulation_parent",
            class_order=list(fresh.CLASSES), sequence_length_bp=fresh.LENGTH,
            parents=[dict(parent_id="argformer_american_" + split + "_000",
                          split=split, simulation_stream="independent_" + split)
                     for split in ("train", "validation", "test")])
        cls.registry_ref = cls.write_json("registry.json", cls.payload)
        cls.registry = fresh.FreshParentRegistry.from_ref(cls.registry_ref)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def write_json(cls, name, value):
        path = cls.base / name
        path.write_text(json.dumps(value))
        return ref(path)

    def make_join(self, truth_parent=None):
        parent = self.registry.ids("train")[0]
        focal = dict(
            record_ids=np.arange(10000, dtype=np.int64),
            sample_nodes=np.tile(np.arange(1699, 699, -1, dtype=np.int64), 10),
            sample_indices=np.tile(np.arange(1599, 599, -1, dtype=np.int64), 10),
            positions_bp0=np.repeat(np.array([0., 15000000.], dtype=np.float64), 5000),
            parent_id=np.array(parent), class_order=np.array(fresh.CLASSES))
        focal_path = self.base / "focals.npz"
        np.savez(focal_path, **focal)
        labels = np.stack((np.arange(1000) % 3, (np.arange(1000) + 1) % 3))
        truth_path = self.base / "truth.npz"
        np.savez(truth_path,
                 intervals_bp=np.array([[0., 15000000.], [15000000., 30000000.]]),
                 query_sample_indices=np.arange(600, 1600, dtype=np.int64),
                 source_classes_0based=labels.astype(np.uint8),
                 parent_id=np.array(truth_parent or parent),
                 class_order=np.array(fresh.CLASSES))
        values = dict(
            observation_ref=self.write_json("observation.json", {"parent_id": parent}),
            focal_registry_ref=ref(focal_path),
            native_preparation_ref=self.write_json("native.json", {"parent_id": parent}),
            truth_ref=ref(truth_path))
        called = []
        def resolve(parent_id):
            called.append(parent_id)
            return dict(values["truth_ref"])
        join = fresh.FreshTrainTargetJoin(
            self.registry, resolve, train_inputs={parent: values},
            max_truth_uncompressed_bytes=100000)
        plan = dict(parent_id=parent, sequence_length_bp=fresh.LENGTH,
                    **{k: values[k] for k in fresh.INPUT_REFS})
        plan["plan_ref"] = self.write_json("plan.json", plan)
        seal = dict(plan, observation_only=True, all_blocks_sealed_before_truth=True,
                    parent_registry_ref=self.registry.reference,
                    focal_identity_sha256=fresh._identity_sha(focal))
        return parent, join, called, plan, seal, focal

    def test_registry_retains_new_ids_and_deterministic_epoch_order(self):
        self.assertTrue(all(p.startswith("argformer_american_")
                            for p, _, _ in self.registry.parents))
        expected = list(self.registry.ids("train"))
        random.Random(70003).shuffle(expected)
        self.assertEqual(self.registry.epoch_order(1), expected)
        self.assertEqual(self.registry.reference, self.registry_ref)
        for epoch in (0, 31, True):
            with self.assertRaises(ValueError):
                self.registry.epoch_order(epoch)

    def test_registry_rejects_overlapping_simulation_streams(self):
        value = json.loads(json.dumps(self.payload))
        value["parents"][2]["simulation_stream"] = value["parents"][0]["simulation_stream"]
        record = self.write_json("bad_streams.json", value)
        with self.assertRaisesRegex(ValueError, "Disjoint"):
            fresh.FreshParentRegistry.from_ref(record)

    def test_registry_ref_is_authenticated(self):
        bad = dict(self.registry_ref, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA"):
            fresh.FreshParentRegistry.from_ref(bad)

    def test_target_join_preserves_sample_mapping_and_half_open_boundary(self):
        parent, join, calls, plan, seal, focals = self.make_join()
        targets = join.join(parent, split="train", plan=plan, observation_seal=seal)
        columns = focals["sample_indices"] - 600
        expected = (columns + (focals["positions_bp0"] >= 15000000.)) % 3
        self.assertTrue(np.array_equal(targets.labels_cpu.numpy(), expected))
        self.assertEqual(calls, [parent])
        self.assertEqual(targets.parent_id, parent)
        self.assertEqual(targets.receipt["parent_registry_ref"], self.registry_ref)
        self.assertFalse(targets.receipt["purity_filter"])
        self.assertFalse(targets.receipt["labels_in_observation_bank"])

    def test_heldout_and_unregistered_parents_denied_before_truth_resolver(self):
        _, join, calls, plan, seal, _ = self.make_join()
        for parent in (*self.registry.ids("validation"), *self.registry.ids("test"),
                       "unregistered"):
            with self.subTest(parent=parent), self.assertRaises(ValueError):
                join.join(parent, split="train", plan=plan, observation_seal=seal)
        self.assertEqual(calls, [])

    def test_observed_binding_and_identity_mismatch_denied_before_truth(self):
        parent, join, calls, plan, seal, _ = self.make_join()
        bad = dict(seal, focal_identity_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "focal identities"):
            join.join(parent, split="train", plan=plan, observation_seal=bad)
        bad_plan = dict(plan, observation_ref=dict(plan["observation_ref"], sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "observed inputs"):
            join.join(parent, split="train", plan=bad_plan, observation_seal=seal)
        self.assertEqual(calls, [])

    def test_same_parent_truth_is_verified_after_resolution(self):
        parent, join, calls, plan, seal, _ = self.make_join(truth_parent="another_parent")
        with self.assertRaisesRegex(ValueError, "parent/class"):
            join.join(parent, split="train", plan=plan, observation_seal=seal)
        self.assertEqual(calls, [parent])

    def test_private_namespaces_cover_all_historical_membership_and_type_barriers(self):
        original = importlib.import_module(fresh.treeloom.MODEL_MODULE)
        prep = importlib.import_module(fresh.treeloom.PREPARATION_MODULE)
        execution = importlib.import_module(fresh.treeloom.EXECUTION_MODULE)
        original_train, original_val = original.TRAIN, original.VAL
        original_bundle = original.AmericanTreeLoomWholeParentV1
        runtime = fresh.FreshTreeLoom(self.registry)
        self.assertEqual(runtime.frozen.TRAIN, self.registry.ids("train"))
        self.assertEqual(runtime.frozen.VAL,
                         self.registry.ids("validation") + self.registry.ids("test"))
        self.assertIsNot(runtime.frozen.AmericanTreeLoomWholeParentV1, original_bundle)
        self.assertIs(runtime.preparation.AmericanTreeLoomWholeParentV1,
                      runtime.frozen.AmericanTreeLoomWholeParentV1)
        self.assertIs(runtime.execution.frozen, runtime.frozen)
        self.assertEqual(runtime.preparation.TRAIN, self.registry.ids("train"))
        self.assertEqual(runtime.preparation.VAL, runtime.frozen.VAL)
        # Function bytecode is unchanged. Only the private global bindings differ.
        self.assertEqual(runtime.frozen.AmericanTreeLoomWholeParentV1.validate.__code__.co_code,
                         original_bundle.validate.__code__.co_code)
        self.assertEqual(runtime.preparation.prepare_from_loaded_observation_v1.__code__.co_code,
                         prep.prepare_from_loaded_observation_v1.__code__.co_code)
        self.assertEqual(runtime.execution.train_step_v1.__code__.co_code,
                         execution.train_step_v1.__code__.co_code)
        self.assertEqual(original.TRAIN, original_train)
        self.assertEqual(original.VAL, original_val)
        self.assertIs(original.AmericanTreeLoomWholeParentV1, original_bundle)
        self.assertIs(execution.frozen, original)

    def test_prediction_phase_alias_keeps_true_split_and_parent_identity(self):
        runtime = object.__new__(fresh.FreshTreeLoom)
        runtime.registry = self.registry
        captured = []
        def prepare(state, **kwargs):
            captured.append((state["manifest"]["parent_id"], kwargs["split"]))
            return object(), dict(parent_id=state["manifest"]["parent_id"])
        runtime.preparation = SimpleNamespace(prepare_from_loaded_observation_v1=prepare)
        for split in ("train", "validation", "test"):
            parent = self.registry.ids(split)[0]
            manifest = {"parent_id": parent}
            record = self.write_json("phase_" + split + ".json", manifest)
            _, metadata = runtime.prepare_observation(
                {"manifest": manifest}, split=split, observation_ref=record,
                resource_check=lambda _: None)
            self.assertEqual(metadata["declared_split"], split)
            self.assertEqual(captured[-1][0], parent)
            self.assertEqual(captured[-1][1],
                             "train" if split == "train" else "val_prediction")

    def test_direct_train_api_delegates_without_changing_arguments(self):
        runtime = object.__new__(fresh.FreshTreeLoom)
        runtime.registry = self.registry
        class Model:
            pass
        class Bundle:
            def validate(self):
                return None
        runtime.frozen = SimpleNamespace(
            AmericanTreeLoomC3FullV1=Model, AmericanTreeLoomWholeParentV1=Bundle)
        captured = []
        runtime.execution = SimpleNamespace(
            train_step_v1=lambda *args, **kwargs: captured.append((args, kwargs)) or "updated")
        model, bundle = Model(), Bundle()
        bundle.parent_id = self.registry.ids("train")[0]
        bundle.split = "train"
        targets = SimpleNamespace(receipt={
            "parent_registry_ref": self.registry.reference, "split": "train"})
        metadata = dict(parent_id=bundle.parent_id, declared_split="train",
                        parent_registry_ref=self.registry.reference)
        optimizer = object()
        callback = lambda _: None
        result = runtime.train_step(model, bundle, targets, optimizer, metadata,
                                    resource_check=callback)
        self.assertEqual(result, "updated")
        self.assertEqual(captured[0][0], (model, bundle, targets, optimizer, metadata))
        self.assertEqual(captured[0][1]["history_holds_per_window"], 128)
        self.assertIs(captured[0][1]["resource_check"], callback)
        bundle.parent_id = self.registry.ids("test")[0]
        bundle.split = "val_prediction"
        with self.assertRaises(ValueError):
            runtime.train_step(model, bundle, targets, optimizer, metadata,
                               resource_check=callback)
        self.assertEqual(len(captured), 1)


if __name__ == "__main__":
    unittest.main()

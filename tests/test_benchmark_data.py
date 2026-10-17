"""Metadata/fake-loader smoke tests: no data, Torch runtime or worker processes."""
from dataclasses import FrozenInstanceError, replace
import json
import os
import pickle
import sys
import types
import unittest
from unittest.mock import patch

from arggnn.benchmarking.data import (
    ImmutableRef, MAX_DESCRIPTOR_BYTES, PreparedParentDataset,
    PreparedParentDescriptor, _cpu_worker_init, make_descriptor_loader,
)


def reference(name):
    return {"path": "RAM_ONLY/" + name, "sha256": "a" * 64}


def descriptor(parent="parent_0", **changes):
    options = dict(parent_id=parent, split="train",
                   protocol_ref=reference("protocol.json"),
                   preparation_source_ref=reference("source_manifest.json"),
                   observation_ref=reference(parent + "/observed.json"),
                   focal_registry_ref=reference(parent + "/focals.npz"),
                   native_preparation_ref=reference(parent + "/preparation.json"))
    options.update(changes)
    return PreparedParentDescriptor(**options)


class FakeGenerator:
    def __init__(self, *, device):
        self.device, self.seed = device, None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeLoader:
    def __init__(self, dataset, **options):
        self.dataset, self.options = dataset, options

    def __iter__(self):
        return (self.options["collate_fn"](self.dataset[i])
                for i in range(len(self.dataset)))


def fake_torch_modules(thread_calls=None):
    torch = types.ModuleType("torch")
    torch.Generator = FakeGenerator
    torch.set_num_threads = lambda count: thread_calls.append(("threads", count))
    torch.set_num_interop_threads = lambda count: thread_calls.append(("interop", count))

    def forbidden_seed(*_):
        raise AssertionError("Scientific main-process Torch RNG must not be reset")
    torch.manual_seed = forbidden_seed
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.DataLoader = FakeLoader
    return {"torch": torch, "torch.utils": utils, "torch.utils.data": data}


class DescriptorTests(unittest.TestCase):
    def test_descriptor_is_small_immutable_metadata(self):
        value = descriptor()
        self.assertLess(len(json.dumps(value.as_dict()).encode()), MAX_DESCRIPTOR_BYTES)
        self.assertLess(len(pickle.dumps(value)), 4096)
        with self.assertRaises(FrozenInstanceError):
            value.parent_id = "other"
        with self.assertRaises(FrozenInstanceError):
            value.observation_ref.path = "other"
        exported = value.as_dict()
        exported["observation_ref"]["path"] = "changed"
        self.assertEqual(value.observation_ref.path, "RAM_ONLY/parent_0/observed.json")

    def test_exact_mapping_roundtrip_and_identity(self):
        value = descriptor()
        self.assertEqual(PreparedParentDescriptor.from_mapping(value.as_dict()), value)
        self.assertEqual(len(value.identity_key), 64)

    def test_source_and_protocol_and_artifact_versions_key_identity(self):
        original = descriptor()
        for change in [
            {"protocol_ref": reference("other_protocol.json")},
            {"preparation_source_ref": reference("other_source.json")},
            {"observation_ref": reference("other_observed.json")},
            {"prepared_bundle_ref": reference("existing_guarded_scope/observed_preparation.cache")},
        ]:
            self.assertNotEqual(original.identity_key, descriptor(**change).identity_key)

    def test_stale_serialized_identity_is_rejected(self):
        value = descriptor().as_dict()
        value["protocol_ref"]["sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            PreparedParentDescriptor.from_mapping(value)

    def test_query_truth_and_array_fields_are_rejected(self):
        for field in ["truth_ref", "query_ancestry", "targets", "genotype", "tensors"]:
            value = descriptor().as_dict()
            value[field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                PreparedParentDescriptor.from_mapping(value)

    def test_observed_only_flag_is_required(self):
        value = descriptor().as_dict()
        value["observed_only"] = False
        with self.assertRaises(ValueError):
            PreparedParentDescriptor.from_mapping(value)

    def test_explicit_query_truth_paths_are_rejected(self):
        for path in ["truth/parent.npz", "observed/query_ancestry_truth.npz",
                     "observed/query_truth.npz", "targets/parent.npz"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                descriptor(observation_ref={"path": path, "sha256": "a" * 64})

    def test_reference_population_labels_are_explicitly_allowed(self):
        labels = reference("observed/reference_labels.npz")
        value = descriptor(reference_labels_ref=labels)
        self.assertEqual(value.reference_labels_ref.as_dict(), labels)
        self.assertNotEqual(value.identity_key, descriptor().identity_key)

    def test_existing_prepared_artifact_is_only_a_handle(self):
        value = descriptor(prepared_bundle_ref=reference("closed_scope/observed_preparation.cache"))
        self.assertEqual(value.prepared_bundle_ref.path, "RAM_ONLY/closed_scope/observed_preparation.cache")
        self.assertFalse(hasattr(value, "decode"))
        self.assertFalse(hasattr(value, "tensors"))

    def test_references_reject_payload_and_noncanonical_paths(self):
        for path in ["/outside/data", "../data", "data/../other", "./data", "data//file", "data\\file"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                ImmutableRef(path, "a" * 64)
        with self.assertRaises(ValueError):
            descriptor(observation_ref={"path": "data", "sha256": "a" * 64, "payload": []})

    def test_dataset_preserves_declared_nonalphabetical_order_and_repeated_visits(self):
        values = [descriptor("parent_2"), descriptor("parent_1"), descriptor("parent_2")]
        dataset = PreparedParentDataset(values, parent_order=["parent_2", "parent_1", "parent_2"])
        self.assertEqual([dataset[i].parent_id for i in range(len(dataset))],
                         ["parent_2", "parent_1", "parent_2"])
        self.assertEqual(dataset[0], dataset[2])
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(restored.parent_order, dataset.parent_order)
        self.assertEqual(restored[0].identity_key, dataset[0].identity_key)

    def test_dataset_does_not_silently_sort_or_shuffle(self):
        with self.assertRaises(ValueError):
            PreparedParentDataset([descriptor("parent_2"), descriptor("parent_1")],
                                  parent_order=["parent_1", "parent_2"])

    def test_dataset_does_not_mix_versions_or_splits(self):
        variants = [
            descriptor("parent_1", split="validation"),
            descriptor("parent_1", protocol_ref=reference("other_protocol")),
            descriptor("parent_1", preparation_source_ref=reference("other_source")),
        ]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PreparedParentDataset([descriptor(), value], parent_order=["parent_0", "parent_1"])

    def test_repeated_parent_cannot_change_identity(self):
        with self.assertRaises(ValueError):
            PreparedParentDataset([descriptor(), descriptor(observation_ref=reference("other"))],
                                  parent_order=["parent_0", "parent_0"])


class LoaderTests(unittest.TestCase):
    def dataset(self):
        return PreparedParentDataset([descriptor("parent_1"), descriptor("parent_0")],
                                     parent_order=["parent_1", "parent_0"])

    def test_zero_worker_loader_is_ordered_metadata_only_and_leaves_main_environment(self):
        with patch.dict(sys.modules, fake_torch_modules([])), patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "MAIN_GPU0"}, clear=False):
            loader = make_descriptor_loader(self.dataset(), num_workers=0)
            options = loader.options
            self.assertFalse(options["shuffle"])
            self.assertFalse(options["pin_memory"])
            self.assertFalse(options["persistent_workers"])
            self.assertIsNone(options["batch_size"])
            self.assertNotIn("prefetch_factor", options)
            self.assertNotIn("worker_init_fn", options)
            self.assertEqual([value.parent_id for value in loader], ["parent_1", "parent_0"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "MAIN_GPU0")

    def test_workers_use_spawn_one_prefetch_and_persistence(self):
        with patch.dict(sys.modules, fake_torch_modules([])):
            loader = make_descriptor_loader(self.dataset(), num_workers=3)
        self.assertEqual(loader.options["num_workers"], 3)
        self.assertEqual(loader.options["prefetch_factor"], 1)
        self.assertTrue(loader.options["persistent_workers"])
        self.assertEqual(loader.options["multiprocessing_context"], "spawn")
        self.assertIs(loader.options["worker_init_fn"], _cpu_worker_init)
        self.assertFalse(loader.options["pin_memory"])

    def test_private_CPU_generator_does_not_change_scientific_RNG(self):
        with patch.dict(sys.modules, fake_torch_modules([])):
            generator = make_descriptor_loader(self.dataset()).options["generator"]
        self.assertEqual((generator.device, generator.seed), ("cpu", 0))

    def test_worker_init_has_one_CPU_thread_and_no_CUDA_visibility(self):
        calls = []
        with patch.dict(sys.modules, fake_torch_modules(calls)), patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU0"}, clear=False):
            _cpu_worker_init(0)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                self.assertEqual(os.environ[key], "1")
        self.assertEqual(calls, [("threads", 1), ("interop", 1)])

    def test_bad_worker_counts_and_non_descriptor_dataset_fail_before_torch(self):
        for count in [-1, True, 1.5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                make_descriptor_loader(self.dataset(), num_workers=count)
        with self.assertRaises(TypeError):
            make_descriptor_loader([descriptor()])


if __name__ == "__main__":
    unittest.main()

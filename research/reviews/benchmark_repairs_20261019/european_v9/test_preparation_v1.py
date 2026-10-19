"""Tiny complete four-donor forests, namespace parity and GPU-free phase mocks."""
import ast
from dataclasses import fields
import os
from pathlib import Path
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from arggnn.benchmarking import european_training as original
from arggnn.benchmarking import european_c4_training as c4
from arggnn.benchmarking import treeloom_prediction as gpu
from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_observations_v1 as old_observations
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import (
    episode_preparation_v1 as old_episodes, cut_preparation_v1 as old_cut)
from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1
from research.reviews.workflow_progress_diagnosis_20261018.treeloom.test_candidate_v1 import fixture
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import candidate_v1 as candidate
from research.reviews.benchmark_repairs_20261019.treeloom.cut_hoist_v1 import prepare_cut
from . import integration_v1 as integration


class PreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def same_fields(self, a, b, *, skip=()):
        for field in fields(a):
            if field.name in skip:
                continue
            left, right = getattr(a, field.name), getattr(b, field.name)
            if torch.is_tensor(left):
                self.assertEqual(left.dtype, right.dtype)
                self.assertEqual(left.shape, right.shape)
                self.assertEqual(left.numpy().tobytes(), right.numpy().tobytes())
            elif isinstance(left, tuple) and left and torch.is_tensor(left[0]):
                self.assertEqual(len(left), len(right))
                for x, y in zip(left, right):
                    self.assertEqual(x.dtype, y.dtype)
                    self.assertEqual(x.numpy().tobytes(), y.numpy().tobytes())
            else:
                self.assertEqual(left, right)

    def test_four_donor_connected_unary_isolated_bank_episode_cut_and_metadata_parity(self):
        for isolated in (False, True):
            with self.subTest(isolated=isolated):
                values = list(fixture(isolated))
                values[5] = np.array([0, 1, 2, 3, -1, -1, -1, -1])
                old_bank, old_meta = old_observations.encode_block(*values, num_classes=4,
                    query_columns=np.arange(4, 8))
                bank, meta = candidate.encode_block(*values, num_classes=4, query_columns=np.arange(4, 8))
                self.same_fields(old_bank, bank);self.assertEqual(old_meta, meta)
                holds = tuple(meta.get("isolated_zero_site_holds", ()))
                arrays, episode_meta = old_episodes.build_clade_episode_arrays_v1(
                    old_bank.parent.numpy(), old_bank.sample_index.numpy(), old_bank.spans.numpy(),
                    max_payload_bytes=2**20, isolated_hold_indices=holds)
                expected_episodes = OrderedCladeEpisodesV1.from_verified_arrays(arrays, episode_meta)
                episodes, new_episode_meta = candidate.build_episodes(bank, max_payload_bytes=2**20,
                                                                       isolated_hold_indices=holds)
                self.same_fields(expected_episodes.flags, episodes.flags)
                self.same_fields(expected_episodes, episodes, skip=("flags",))
                self.assertEqual(episode_meta, new_episode_meta)
                kwargs = dict(expected_haplotypes=8, max_nodes=1000, max_holds=10,
                              isolated_hold_indices=holds)
                expected_cut, expected_meta = old_cut.prepare_cut_plan_v1(old_bank, **kwargs)
                cut, cut_meta = prepare_cut(bank, **kwargs)
                self.same_fields(expected_cut, cut, skip=("source_stamps", "tensor_stamps"))
                self.assertEqual(expected_meta, cut_meta)
                self.assertEqual(bank.log_donor_allele_frequency.shape[-1], 8)

    def test_both_protocol_namespaces_keep_models_classes_targets_and_schedules(self):
        for protocol in integration.PROTOCOLS:
            bound = integration.backend(protocol)
            core = original if protocol == integration.AP else c4._bound()
            prepared = bound.european.prepare_observation
            self.assertIs(prepared.__globals__["EuropeanWholeParent"], core.european.EuropeanWholeParent)
            self.assertIs(prepared.__globals__["_optimized_observations"].encode_block, candidate.encode_block)
            self.assertIs(prepared.__globals__["_optimized_episodes"], candidate.build_episodes)
            self.assertIs(prepared.__globals__["_optimized_cut_preparation"].prepare_cut_plan_v1, prepare_cut)
            self.assertIs(bound.run_fit.__globals__["european"], core.european)
            self.assertIs(bound.run_fit.__globals__["_training_targets"], core._training_targets)
            self.assertEqual(bound.run_fit.__globals__["SETTINGS"], original.SETTINGS)
            self.assertIs(bound.run_predictions.__globals__["gpu"], core.gpu)
            if protocol == integration.C4:
                self.assertIs(bound.parent_plan, c4._parent_plan)
                self.assertIs(prepared.__globals__["_sampled_records"], c4._sampled_records)
                self.assertIs(bound.run_fit.__globals__["_training_targets"], c4._targets)

    def test_CPU_setup_has_no_CUDA_access_and_preserves_writer_threads_and_RAM_check(self):
        events = []
        class FakeCuda:
            def is_initialized(self):return False
            def __getattr__(self, name):raise AssertionError("CUDA runtime touched during CPU preparation")
        class FakeTorch:
            cuda=FakeCuda()
            def set_num_threads(self, count):events.append(("threads", count))
            def set_num_interop_threads(self, count):events.append(("interop", count))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            setup = integration._cpu_setup_function(gpu._setup_gpu)
            setup = integration.clone(setup, c=SimpleNamespace(contained_path=lambda value:Path(folder)),
                training=SimpleNamespace(_current_rss_bytes=lambda:1),
                generation=SimpleNamespace(_Writer=lambda out, cap, check: events.append(("writer",cap)) or "writer"),
                tempfile=SimpleNamespace(tempdir=None))
            with patch.dict("sys.modules", torch=FakeTorch()), patch.dict(os.environ,
                    CUDA_VISIBLE_DEVICES=integration.GPU0_UUID,PYTORCH_NVML_BASED_CUDA_CHECK="1"):
                os.environ.pop("ARGGNN_GPU_LEASE_FD",None);os.environ.pop("ARGGNN_PHYSICAL_GPU",None)
                actual, check, writer = setup(dict(output_directory=folder,max_output_bytes=1024,
                    memory_envelope=dict(max_rss_bytes=2)), lambda stage:events.append(("check",stage)))
            self.assertEqual(writer, "writer")
            self.assertIn(("threads",8),events);self.assertIn(("interop",1),events)
            self.assertIn(("writer",1024),events)
            self.assertEqual(sum(event[0]=="check" for event in events),1)

    def test_all_observations_and_barrier_precede_acquisition_and_first_CUDA_setup(self):
        class Finished(BaseException):pass
        events=[];parents=("p0","p1","p2")
        def load(parent):
            events.append(parent)
            return SimpleNamespace(focal_identity_sha256="a"*64,native_holds=4),{}
        def writer_json(name, value):
            self.assertTrue(value["all_original_TRAIN_observations_sealed_before_targets"])
            events.append("barrier")
            return {}
        def finish(torch):events.append("CUDA_setup");raise Finished()
        fit = integration._fit_function(original.run_fit, loader=lambda *args, **kwargs:load,
            gpu=SimpleNamespace(_setup_gpu=lambda *args:(events.append("CPU_setup") or "torch",
                lambda stage:None,SimpleNamespace(json=writer_json))))
        fit = integration.clone(fit, _validate=lambda *args,**kwargs:dict(train_parent_ids=parents),
                                _finish_gpu_setup=finish)
        spec=dict(protocol_id=integration.AP,sampling_plan_ref={},
            train_inputs={parent:{key:{} for key in original.INPUTS} for parent in parents})
        with self.assertRaises(Finished):
            fit(spec,resource_check=lambda stage:None,acquire_gpu0=lambda:events.append("acquire"))
        self.assertEqual(events,["CPU_setup",*parents,"barrier","acquire","CUDA_setup"])

    def test_GPU_tail_preserves_original_FP32_and_allocator_settings_without_real_device(self):
        calls=[]
        fake=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,
            get_device_properties=lambda index:SimpleNamespace(total_memory=80*2**30),
            set_per_process_memory_fraction=lambda fraction,index:calls.append((fraction,index))),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)))
        with patch.dict(os.environ,ARGGNN_PHYSICAL_GPU="0",ARGGNN_MAX_CUDA_RESERVED_BYTES=str(74*2**30)):
            integration.finish_gpu_setup(fake)
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake.backends.cudnn.allow_tf32)
        self.assertEqual(calls,[(74/80,0)])


if __name__=="__main__":unittest.main()

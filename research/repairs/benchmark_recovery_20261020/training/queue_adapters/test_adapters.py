"""Small synthetic CPU checks. No real observations, training or GPU allocation."""
import ast
import copy
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import common, european_stage as stage, european_observations as observed
from . import american_prediction as american, empirical
from .european_gru import install_chunked_gru
from .european_execution import decode_parent_backward


class MiniDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.genomic_gru = nn.GRU(35, 32, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 4))


class AdapterTests(unittest.TestCase):
    def test_plain_source_closure_and_routes(self):
        banned = {"exec", "eval", "compile", "FunctionType", "_private_module", "compile_function"}
        for name in common.SOURCE_NAMES:
            source = (Path(common.DIRECTORY)/name).read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    function = node.func
                    label = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
                    # model.eval() sets evaluation mode, it does not execute source.
                    if label != "eval" or isinstance(function, ast.Name):
                        self.assertNotIn(label, banned, (name, node.lineno, label))
        self.assertGreaterEqual(len(common.source_refs()), 21)
        self.assertIn("acquire_gpu0", inspect.signature(stage.run_fit).parameters)
        self.assertIn("resource_check", inspect.signature(american.run_predictions).parameters)
        self.assertEqual(empirical.CLASSES, ("Ana", "CHG", "WHG", "EHG"))

    def test_late_fit_requires_fixed_UUID_and_NVML_before_torch(self):
        import os
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":"", "PYTORCH_NVML_BASED_CUDA_CHECK":"1"}):
            with self.assertRaises(ValueError):
                common._cpu_setup({}, lambda label:None)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":common.GPU0_UUID, "PYTORCH_NVML_BASED_CUDA_CHECK":"0"}):
            with self.assertRaises(ValueError):
                common._cpu_setup({}, lambda label:None)

    def test_chunked_bidirectional_gru_full_BPTT(self):
        torch.manual_seed(9)
        reference = MiniDecoder()
        adapted = copy.deepcopy(reference)
        before = [(name, id(value)) for name, value in adapted.named_parameters()]
        state_before = {key: value.clone() for key, value in adapted.state_dict().items()}
        rng = torch.get_rng_state().clone()
        install_chunked_gru(adapted, chunk_holds=4)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, [(name, id(value)) for name, value in adapted.named_parameters()])
        self.assertEqual(tuple(state_before), tuple(adapted.state_dict()))
        self.assertTrue(all(torch.equal(value, adapted.state_dict()[key]) for key, value in state_before.items()))
        values = torch.randn(17, 5, 35, requires_grad=True)
        other = values.detach().clone().requires_grad_(True)
        hidden = torch.randn(2, 5, 32, requires_grad=True)
        hidden_other = hidden.detach().clone().requires_grad_(True)
        output, final = reference.genomic_gru(values, hidden)
        got, got_final = adapted.genomic_gru(other, hidden_other)
        torch.testing.assert_close(got, output, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(got_final, final, rtol=2e-5, atol=2e-6)
        (output.square().sum()+final.square().sum()).backward()
        (got.square().sum()+got_final.square().sum()).backward()
        torch.testing.assert_close(other.grad, values.grad, rtol=3e-5, atol=1e-5)
        torch.testing.assert_close(hidden_other.grad, hidden.grad, rtol=3e-5, atol=1e-5)
        for (name, parameter), (got_name, actual) in zip(reference.genomic_gru.named_parameters(), adapted.genomic_gru.named_parameters()):
            self.assertEqual(name, got_name)
            torch.testing.assert_close(actual.grad, parameter.grad, rtol=3e-5, atol=1e-5)
        reference.eval();adapted.eval()
        with torch.no_grad():
            torch.testing.assert_close(adapted.genomic_gru(values)[0], reference.genomic_gru(values)[0], rtol=2e-5, atol=2e-6)

    def test_streamed_decoder_repeated_focals_and_empty_tiles(self):
        torch.manual_seed(13)
        reference = MiniDecoder()
        adapted = install_chunked_gru(copy.deepcopy(reference), chunk_holds=4)
        context = torch.randn(17, 7, 32, requires_grad=True)
        got_context = context.detach().clone().requires_grad_(True)
        geometry = torch.randn(17, 3)
        holds = torch.tensor([0, 16, 8, 8, 3, 4, 2, 16])
        columns = torch.tensor([0, 1, 2, 2, 6, 0, 6, 1])
        labels = torch.tensor([0, 1, 2, 2, 3, 1, 0, 2])
        sequence = torch.cat((context, geometry[:, None].expand(-1, 7, -1)), -1)
        history, _ = reference.genomic_gru(sequence)
        logits = reference.classifier(torch.cat((context[holds, columns], history[holds, columns]), -1))
        expected_loss = F.cross_entropy(logits, labels)
        expected_loss.backward()
        actual_loss, gradient = decode_parent_backward(adapted, got_context, geometry, holds, columns,
            labels, tile_haplotypes=2)
        got_context.backward(gradient)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(got_context.grad, context.grad, rtol=3e-5, atol=2e-6)
        for (name, parameter), (got_name, actual) in zip(reference.named_parameters(), adapted.named_parameters()):
            self.assertEqual(name, got_name)
            torch.testing.assert_close(actual.grad, parameter.grad, rtol=4e-5, atol=2e-6)
        self.assertTrue(torch.equal(gradient[:, 4:6], torch.zeros_like(gradient[:, 4:6])))
        # Earlier recurrent inputs contribute to labels at later positions.
        self.assertGreater(float(gradient[0, 1].abs().sum()), 0)

    def test_full_synthetic_European_trunk_logits_and_gradients(self):
        from research.repairs.benchmark_recovery_20261020.runtime import observations, episodes, cuts
        from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
        from .european_execution import EuropeanStagedExecutionV1, _tile_logits
        from ..decoder import geometry_from_spans
        implementation = observed.implementation
        h = 952
        trees = []
        for at in range(3):
            parent = np.full(h+3, h, dtype=np.int64)
            parent[h//2:h] = h+1
            if at == 2:
                parent[0], parent[h//2] = h+1, h
            parent[h:h+2] = h+2
            parent[-1] = -1
            age = np.zeros(h+3, dtype=np.float64)
            age[h:] = [100, 150, 1000]
            trees.append(NativeObservedTreeV1(parent, age, np.arange(h), np.arange(h)))
        labels = np.full(h, -1, dtype=np.int64)
        labels[:266] = np.arange(266)%4
        positions = np.asarray([10, 30, 120, 250], dtype=np.float64)
        calls = np.random.default_rng(7).integers(0, 2, (len(positions), h), dtype=np.int8)
        bank, _ = observations.encode_block(trees, [(0, 100), (100, 200), (200, 300)], calls,
            np.full(len(positions), 2, dtype=np.int8), positions, labels, np.zeros(h), num_classes=4)
        episode, metadata = episodes.build_episodes(bank, max_payload_bytes=2**20)
        cut, _ = cuts.prepare_cut_plan_v1(bank, expected_haplotypes=h, max_nodes=10000, max_holds=10)
        condition = observed.PROTOCOLS["ancestralpaths_paper200mb_c4"]
        refs = dict(parent_id="synthetic", query_truth_in_inputs=False,
            all_observed_trees_nodes_sites_samples_retained=True, supported_length_bp=300., requested_length_bp=200000000)
        original = implementation.FrozenEuropeanBundleV1(bank, episode, cut, condition, "synthetic", "train", refs,
            implementation.memory_core._stamps(implementation._bank_tensors(bank)),
            implementation.memory_core._stamps(implementation._episode_tensors(episode)))
        holds = torch.tensor([0, 2, 1, 1])
        columns = torch.tensor([266, 267, 268, 268])
        bundle = observed.EuropeanWholeParent(original, holds, columns, {}, "synthetic",
            implementation.memory_core._stamps((holds, columns)))
        torch.manual_seed(101)
        reference = implementation.FrozenEuropeanTreeLoomV1(condition=condition)
        adapted = install_chunked_gru(copy.deepcopy(reference), chunk_holds=2)
        expected = reference.forward_holds(bank, episodes=original)[holds, columns]
        executor = EuropeanStagedExecutionV1(adapted, bundle, dict(episode_metadata=metadata), holds_per_block=1)
        context = executor.context_forward()
        actual = _tile_logits(adapted, context, geometry_from_spans(bank.spans), holds, columns)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        target = torch.tensor([0, 1, 2, 2])
        F.cross_entropy(expected, target).backward()
        F.cross_entropy(actual, target).backward()
        for (name, parameter), (got_name, got) in zip(reference.named_parameters(), adapted.named_parameters()):
            self.assertEqual(name, got_name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(got.grad, name)
            torch.testing.assert_close(got.grad, parameter.grad, rtol=8e-4, atol=3e-6, msg=name)

    def test_all_observation_seals_precede_late_gpu_callback(self):
        class Acquired(Exception):
            pass
        calls = []
        spec = dict(protocol_id="ancestralpaths_paper200mb_c4", sampling_plan_ref={},
            train_inputs={parent:{key:{} for key in stage.INPUTS} for parent in ("a", "b", "c")})
        def load(parent):
            calls.append(("observed", parent))
            return SimpleNamespace(focal_identity_sha256=parent, native_holds=2), {}
        def write(name, value):
            calls.append(("barrier", tuple(value["parents"])))
            return {}
        def acquire():
            calls.append(("acquire",))
            raise Acquired()
        with patch.object(stage, "_validate", return_value=dict(train_parent_ids=["a", "b", "c"])), \
             patch.object(common, "_cpu_setup", return_value=(object(), lambda label:None, SimpleNamespace(json=write))), \
             patch.object(stage, "_observed_loader", return_value=load), \
             patch.object(stage, "_training_targets") as native_targets, \
             patch.object(stage, "_c4_training_targets") as c4_targets:
            with self.assertRaises(Acquired):
                stage._run_scientific_fit(spec, resource_check=lambda label:None, acquire_gpu0=acquire)
            native_targets.assert_not_called()
            c4_targets.assert_not_called()
        self.assertEqual(calls, [("observed", "a"), ("observed", "b"), ("observed", "c"),
                                 ("barrier", ("a", "b", "c")), ("acquire",)])

    def test_full_checkpoint_validators_reject_incomplete_counters(self):
        implementation = observed.implementation
        for protocol in ("ancestralpaths_paper200mb_c4", "ancestralpaths_paper200mb_native6"):
            model = implementation.FrozenEuropeanTreeLoomV1(condition=observed.PROTOCOLS[protocol])
            if protocol.endswith("native6"):
                model.output_classes = observed.NATIVE_SIX_CLASSES
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
            saved_optimizer = optimizer.state_dict()
            parameters = list(model.parameters())
            self.assertEqual(len(parameters), 49)
            for index, parameter in zip(saved_optimizer["param_groups"][0]["params"], parameters):
                saved_optimizer["state"][index] = dict(step=torch.tensor(90.),
                    exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
            saved = dict(schema="european_treeloom_full_checkpoint_v1", protocol_id=protocol,
                sampling_plan_ref={}, observation_barrier_ref={}, epoch=30, optimizer_updates=90,
                settings=copy.deepcopy(stage.SETTINGS), class_order=list(model.output_classes),
                model_state=model.state_dict(), optimizer_state=saved_optimizer,
                checkpoint_selection="fixed_final_epoch30", TEST_truth_read=False)
            stage._validate_saved(saved, model, protocol_id=protocol, sampling_plan_ref={}, updates=90, barrier_ref={})
            next(iter(saved_optimizer["state"].values()))["step"] = torch.tensor(89.)
            with self.assertRaises(ValueError):
                stage._validate_saved(saved, model, protocol_id=protocol, sampling_plan_ref={}, updates=90, barrier_ref={})
        from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as frozen
        model = frozen.AmericanTreeLoomC3FullV1()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
            betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
        saved_optimizer = optimizer.state_dict()
        for index, parameter in zip(saved_optimizer["param_groups"][0]["params"], model.parameters()):
            saved_optimizer["state"][index] = dict(step=torch.tensor(600.),
                exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
        registry = SimpleNamespace(reference={})
        complete = dict(observation_barrier_ref={})
        saved = dict(schema="treeloom_fresh_full_checkpoint_v1", epoch=30, complete_parent_updates=600,
            model_seed=70001, parent_registry_ref={}, observation_barrier_ref={}, settings=american.training.SETTINGS,
            contract=model.checkpoint_contract(), checkpoint_selection="fixed_final_epoch30",
            VAL_truth_read=False, TEST_truth_read=False, model_state=model.state_dict(), optimizer_state=saved_optimizer)
        american.validate_saved_checkpoint(saved, model, registry, complete)
        next(iter(saved_optimizer["state"].values()))["step"] = torch.tensor(599.)
        with self.assertRaises(ValueError):
            american.validate_saved_checkpoint(saved, model, registry, complete)

    def test_exact_ninety_update_schedule(self):
        parents = ("train_0", "train_1", "train_2")
        counter = [0]
        events, saved = [], []
        def step(bundle, labels, metadata):
            counter[0] += 1
            return dict(parent_id=bundle, one_complete_parent_optimizer_update=True)
        result = stage.fixed_training_loop(parents=parents,
            load_parent=lambda parent:(parent, {}), join_targets=lambda *args:None, step=step,
            optimizer_counts=lambda:[counter[0]]*49,
            checkpoint=lambda epoch, updates:saved.append((epoch, updates)) or {"epoch":epoch},
            check=lambda label:None, event=events.append)
        self.assertEqual(result["optimizer_updates"], 90)
        self.assertEqual(result["selected_epoch"], 30)
        self.assertEqual(saved, [(5, 15), (10, 30), (15, 45), (20, 60), (25, 75), (30, 90)])
        for first in range(0, 90, 3):
            self.assertEqual({row["parent_id"] for row in events[first:first+3]}, set(parents))
        with self.assertRaises(ValueError):
            stage.fixed_training_loop(parents=parents, load_parent=lambda parent:(parent, {}),
                join_targets=lambda *args:None, step=step, optimizer_counts=lambda:[0]*49,
                checkpoint=lambda *args:None, check=lambda label:None, event=lambda row:None)

    def test_C4_and_native6_targets_are_separate(self):
        registry = {"path":"fixture", "sha256":"0"*64}
        bundle = SimpleNamespace(sampled_registry_ref=registry, focal_hold=torch.tensor([0, 1, 1]),
            focal_sample_column=torch.tensor([2, 3, 3]), bank=SimpleNamespace(query_column=torch.tensor([2, 3])))
        model = SimpleNamespace(output_classes=stage.C4_CLASSES)
        metadata = dict(schema="european_release_C4_sampled_targets_v1", parent_id="parent", split="train",
            registry_ref=registry, source_class_order=list(stage.C4_CLASSES), arrays_ref=registry, rows=3)
        with tempfile.TemporaryDirectory(dir=common.DIRECTORY, prefix="synthetic_targets_") as directory:
            path = Path(directory)/"labels.npz"
            np.savez(path, source_labels_0based=np.asarray([0, 1, 1], dtype=np.int64))
            with patch.object(observed, "_json_ref", return_value=metadata), patch.object(observed, "_bound", return_value=path):
                target = stage._c4_training_targets(registry, "parent", bundle, model, max_metadata_bytes=2**20)
                self.assertEqual(target.labels, (0, 1, 1))
                self.assertEqual(target.query_columns, (0, 1, 1))
                self.assertEqual(target.readout_rows, (0, 1, 1))
                with self.assertRaises(ValueError):
                    stage._training_targets(registry, "parent", bundle, model, max_metadata_bytes=2**20)

    def test_empirical_pooling_keeps_diploid_and_physical_weights(self):
        result = {}
        for index, chrom in enumerate(empirical.CHROMOSOMES, 1):
            probabilities = np.zeros((20, 4), dtype=np.float64)
            probabilities[:, index-1] = 1
            result[chrom] = dict(covered_bp=index, haplotype_proportions=probabilities)
        summary = empirical.descriptive_summary(result, ["subject_"+str(i) for i in range(10)])
        np.testing.assert_allclose(summary["pooled_person_probabilities"], np.tile([1/6, 2/6, 3/6, 0], (10, 1)))
        self.assertFalse(summary["ancestry_accuracy_available"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)

"""CPU-only native Composer DataSpec smoke test of the exact unapplied patch.

Extract only capture_trainer from the baseline/candidate AST. The real native
DataSpec/DataLoader are used. Trainer construction is stubbed with Composer's
actual ensure_data_spec normalization; no model, batch, fit, optimizer or GPU.
"""
from pathlib import Path
import os
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
RUNTIME = BASE / "smoke_runtime"
RUNTIME.mkdir(exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME",
             "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
    directory = RUNTIME / name.lower()
    directory.mkdir(exist_ok=True)
    os.environ[name] = str(directory)

import ast
import difflib
import hashlib
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from composer.core.data_spec import DataSpec, ensure_data_spec

BASELINE_SHA = "559af12cae9e4c135c224816af756eca53619850caceaf24e15db92f5f721ee3"
BASELINE = BASE / "american_competitor_worker.baseline.py"
PATCH = BASE / "dataspec_loader_length.patch"
SOURCE_PATH = "src/arggnn/benchmarking/american_competitor_worker.py"
OLD = 'len(kwargs["train_dataloader"])'
NEW = 'len(t.state.train_dataloader)'
source = BASELINE.read_text()
assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA
assert source.count(OLD) == 1
candidate = source.replace(OLD, NEW)
expected_patch = "".join(difflib.unified_diff(source.splitlines(keepends=True),
    candidate.splitlines(keepends=True), fromfile="a/" + SOURCE_PATH, tofile="b/" + SOURCE_PATH))
assert PATCH.read_text() == expected_patch, "Standalone patch must equal the tested one-expression candidate"
assert Path(sys.executable).resolve() == (ROOT / ".runtime/conda/envs/argformer_native_v1/bin/python").resolve()
assert sys.dont_write_bytecode and sys.flags.no_user_site
assert not torch.cuda.is_initialized()


def need(condition, message):
    if not condition:
        raise ValueError(message)


class NeverReadDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 23
    def __getitem__(self, index):
        raise AssertionError("No synthetic item may be read by a loader-length check")


def extraction(text, phase, expected_updates):
    module = ast.parse(text)
    train = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_train")
    capture = next(node for node in train.body if isinstance(node, ast.FunctionDef) and node.name == "capture_trainer")
    seen = []
    def trainer(*args, **kwargs):
        # This matches installed Trainer.__init__ on the admitted CUDA backend,
        # without constructing its model/optimizer/device or calling fit.
        passed = kwargs["train_dataloader"]
        seen.append(passed)
        underlying = ensure_data_spec(passed).dataloader
        return SimpleNamespace(state=SimpleNamespace(train_dataloader=underlying))
    namespace = dict(actual_trainer=trainer, trainer_box={}, phase=phase,
                     spec=dict(expected_updates=expected_updates), api=SimpleNamespace(need=need))
    exec(compile(ast.Module(body=[capture], type_ignores=[]), "<extracted-capture-trainer>", "exec"), namespace)
    return namespace["capture_trainer"], seen, namespace["trainer_box"]


class LoaderLengthSmokeTest(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(192)
        self.loader = torch.utils.data.DataLoader(NeverReadDataset(), batch_size=5,
            shuffle=True, drop_last=True, num_workers=0, generator=self.generator)
        self.dataspec = DataSpec(self.loader)
        self.assertEqual(len(self.loader), 4)

    def without_iteration_or_rng(self, callback):
        before = (torch.get_rng_state().clone(), self.generator.get_state().clone(),
                  random.getstate(), np.random.get_state())
        self.assertIsNone(self.loader._iterator)
        with patch.object(torch.utils.data.DataLoader, "__iter__",
                          side_effect=AssertionError("Loader iteration is forbidden")):
            try:
                return callback()
            finally:
                self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
                self.assertTrue(torch.equal(before[1], self.generator.get_state()))
                self.assertEqual(before[2], random.getstate())
                after_np = np.random.get_state()
                self.assertEqual(before[3][0], after_np[0])
                self.assertTrue(np.array_equal(before[3][1], after_np[1]))
                self.assertEqual(before[3][2:], after_np[2:])
                self.assertIsNone(self.loader._iterator)
                self.assertFalse(torch.cuda.is_initialized())

    def test_actual_native_DataSpec_reproduces_baseline_type_error(self):
        capture, seen, _ = extraction(source, "mlm", 20)
        with self.assertRaisesRegex(TypeError, "DataSpec.*has no len"):
            self.without_iteration_or_rng(lambda: capture(train_dataloader=self.dataspec))
        self.assertIs(seen[0], self.dataspec)

    def test_candidate_accepts_actual_native_DataSpec_without_consumption(self):
        capture, seen, box = extraction(candidate, "mlm", 20)
        trainer = self.without_iteration_or_rng(lambda: capture(train_dataloader=self.dataspec))
        self.assertIs(seen[0], self.dataspec)
        self.assertIs(trainer.state.train_dataloader, self.loader)
        self.assertIs(box["trainer"], trainer)

    def test_candidate_accepts_plain_native_DataLoader_contrastive(self):
        capture, seen, _ = extraction(candidate, "contrastive", 4)
        trainer = self.without_iteration_or_rng(lambda: capture(train_dataloader=self.loader))
        self.assertIs(seen[0], self.loader)
        self.assertIs(trainer.state.train_dataloader, self.loader)

    def test_candidate_rejects_incorrect_MLM_budget_without_consumption(self):
        capture, _, _ = extraction(candidate, "mlm", 25)
        with self.assertRaisesRegex(ValueError, "complete native loader length"):
            self.without_iteration_or_rng(lambda: capture(train_dataloader=self.dataspec))

    def test_candidate_rejects_incorrect_contrastive_budget_without_consumption(self):
        capture, _, _ = extraction(candidate, "contrastive", 5)
        with self.assertRaisesRegex(ValueError, "complete native loader length"):
            self.without_iteration_or_rng(lambda: capture(train_dataloader=self.loader))


if __name__ == "__main__":
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(LoaderLengthSmokeTest))
    import composer
    outcome = dict(schema="argformer_DataSpec_loader_length_CPU_smoke_v1",
        successful=result.wasSuccessful(), tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors),
        baseline_sha256=BASELINE_SHA, candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        patch_sha256=hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        native_python=str(Path(sys.executable).resolve()), torch_version=torch.__version__,
        composer_version=composer.__version__, actual_native_DataSpec_and_DataLoader=True,
        Trainer_construction_mocked_with_actual_Composer_ensure_data_spec=True,
        extracted_only="_train.capture_trainer", no_model_or_fit_or_optimizer=True,
        no_iterator_or_item_consumption=True, RNG_states_unchanged=True,
        CUDA_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], CUDA_initialized=torch.cuda.is_initialized())
    (BASE / "smoke_result.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outcome, sort_keys=True), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)

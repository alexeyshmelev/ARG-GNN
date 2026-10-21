"""Synthetic CPU checks. No model/data load, native subprocess or CUDA context."""
import ast
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import runtime_bindings_v1 as b
from . import queue_european_argmix_v1 as q

BASE = Path(__file__).parent
THIS = sys.modules[__name__]
FAIL = False
SEEN = []


def old_loader(*args, **kwargs):
    raise AssertionError("unbound archive loader")


_load_gt_gru_split_pt = old_loader


def trainer_main():
    SEEN.append((_load_gt_gru_split_pt("logical.json", expected_split_code=0), sys.argv[:]))
    if FAIL:
        raise RuntimeError("synthetic trainer failure")


def old_local(*args, **kwargs):
    raise AssertionError("unbound local runner")


_run_local = old_local


def cmd_infer():
    _run_local(["fixed-author-child", "--device", "cuda"])


def cli_main():
    SEEN.append(sys.argv[:])
    return 0 if not FAIL else 7


class BindingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="synthetic_", dir=BASE)
        self.output = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.title = SimpleNamespace(getproctitle=lambda: b.TITLE, setproctitle=lambda value: None)
        self.env = dict(CUDA_VISIBLE_DEVICES=b.GPU_UUID, ARGGNN_PHYSICAL_GPU="5",
            ARGGNN_MAX_CUDA_RESERVED_BYTES=str(4 * 2**30))
        SEEN.clear()

    def test_gpu5_admission_rejects_other_uuid_before_runtime(self):
        with patch.dict(os.environ, self.env), patch.dict(sys.modules, setproctitle=self.title), \
                patch.object(b.native, "runtime_admission") as admission:
            os.environ["ARGGNN_PHYSICAL_GPU"] = "0"
            with self.assertRaises(ValueError):
                b.admit_native({}, {}, require_gpu=True)
            admission.assert_not_called()

    def test_gpu5_finite_allocator(self):
        fractions = []
        cuda = SimpleNamespace(device_count=lambda: 1,
            get_device_properties=lambda i: SimpleNamespace(total_memory=80 * 2**30),
            set_per_process_memory_fraction=lambda fraction, device: fractions.append((fraction, device)))
        modules = (SimpleNamespace(cuda=cuda),)
        with patch.dict(os.environ, self.env), patch.dict(sys.modules, setproctitle=self.title), \
                patch.object(b.native, "runtime_admission"), \
                patch.object(b.native, "source_imports", return_value=modules):
            self.assertIs(b.admit_native({}, {}, require_gpu=True), modules)
        self.assertEqual(fractions, [(0.05, 0)])

    def test_trainer_callable_and_archive_binding_restore_on_both_paths(self):
        original_argv = sys.argv[:]
        original_handlers = logging.getLogger().handlers[:]
        for failure in (False, True):
            output = self.output / str(failure)
            output.mkdir()
            with patch.object(THIS, "__file__", str(b.native.SOURCE / "src/train_joint_model.py")), \
                    patch.object(THIS, "main", trainer_main, create=True), \
                    patch.object(THIS, "FAIL", failure), \
                    patch.object(b.shards, "load_sharded_split", return_value="mapped"):
                code = trainer_main.__code__
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "synthetic trainer failure"):
                        b.run_original_main(THIS, ["--config", "unchanged"], output,
                            guard=lambda value: None, execution_check=lambda: None)
                else:
                    b.run_original_main(THIS, ["--config", "unchanged"], output,
                        guard=lambda value: None, execution_check=lambda: None)
                self.assertIs(THIS.main, trainer_main)
                self.assertIs(THIS.main.__code__, code)
                self.assertIs(_load_gt_gru_split_pt, old_loader)
            self.assertEqual(sys.argv, original_argv)
            self.assertEqual(logging.getLogger().handlers, original_handlers)
        self.assertEqual([value[0] for value in SEEN], ["mapped", "mapped"])

    def test_inference_runner_identity_wait_and_restoration(self):
        wait = dict(actual_direct_wait=True, actual_direct_returncode=0,
            native_process_absent=True, outer_owned_group_cleanup_pending=True)
        for failure in (False, True):
            with patch.object(THIS, "__file__", str(b.native.SOURCE / "src/commands/infer.py")), \
                    patch.object(b.importlib, "import_module", return_value=THIS):
                try:
                    with b.native_local_runner(self.output, lambda *a, **kw: wait, lambda: None) as waits:
                        self.assertIs(THIS.cmd_infer, cmd_infer)
                        cmd_infer()
                        self.assertEqual(waits[0]["argv"], ["fixed-author-child", "--device", "cuda"])
                        if failure:
                            raise RuntimeError("synthetic outer failure")
                except RuntimeError:
                    self.assertTrue(failure)
                self.assertIs(THIS.cmd_infer, cmd_infer)
                self.assertIs(_run_local, old_local)
        with patch.object(THIS, "__file__", str(b.native.SOURCE / "src/commands/infer.py")), \
                patch.object(b.importlib, "import_module", return_value=THIS):
            with self.assertRaises(ValueError):
                with b.native_local_runner(self.output, lambda *a, **kw: {}, lambda: None):
                    cmd_infer()
        self.assertIs(_run_local, old_local)

    def test_fixed_startup_preserves_environment_and_restores_pythonpath(self):
        before = dict(os.environ)
        with patch.dict(os.environ, PYTHONPATH="previous-path"):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                with b.native_python_startup(self.output, lambda value: None):
                    self.assertEqual((self.output / "python_startup/sitecustomize.py").read_bytes(), b.STARTUP.read_bytes())
                    self.assertTrue(os.environ["PYTHONPATH"].endswith(os.pathsep + "previous-path"))
                    raise RuntimeError("synthetic")
            self.assertEqual(os.environ["PYTHONPATH"], "previous-path")
        self.assertEqual(dict(os.environ), before)

    def test_child_cpu_export_and_gpu_guard_without_real_torch(self):
        from . import sitecustomize_gpu5_v1 as module
        fractions = []
        cuda = SimpleNamespace(device_count=lambda: 1,
            get_device_properties=lambda i: SimpleNamespace(total_memory=80 * 2**30),
            set_per_process_memory_fraction=lambda fraction, device: fractions.append((fraction, device)))
        with patch.dict(os.environ, self.env), patch.dict(sys.modules,
                setproctitle=self.title, torch=SimpleNamespace(cuda=cuda)):
            with patch.object(sys, "argv", [module.SOURCE_ROOT + "/export_torchscript.py", "--device", "cpu"]):
                module.admit_child()
                self.assertEqual(fractions, [])
            with patch.object(sys, "argv", [module.SOURCE_ROOT + "/run_inference.py", "--device", "cuda"]):
                module.admit_child()
                self.assertEqual(fractions, [(0.05, 0)])
                os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-other"
                with self.assertRaises(RuntimeError):
                    module.admit_child()
            self.assertEqual(fractions, [(0.05, 0)])

    def test_imported_cli_argv_return_and_restoration(self):
        original_argv = sys.argv[:]
        for failure in (False, True):
            output = self.output / str(failure)
            output.mkdir()
            with patch.object(THIS, "__file__", str(b.native.SOURCE / "src/argmix_cli.py")), \
                    patch.object(THIS, "main", cli_main, create=True), \
                    patch.object(THIS, "FAIL", failure), \
                    patch.object(b.importlib, "import_module", return_value=THIS):
                if failure:
                    with self.assertRaises(RuntimeError):
                        b.run_author("argmix_cli.py", ["infer", "--device", "cuda"], output)
                else:
                    b.run_author("argmix_cli.py", ["infer", "--device", "cuda"], output)
                self.assertIs(THIS.main, cli_main)
            self.assertEqual(sys.argv, original_argv)
        self.assertTrue(all(argv[1:] == ["infer", "--device", "cuda"] for argv in SEEN))

    def test_original_release_calls_and_hooks_restore_on_failure(self):
        for name, wrapper in (("fit_release", b.fit_release), ("infer_release", b.infer_release)):
            def fail(**arguments):
                self.assertIs(b.release.admit_native, b.admit_native)
                if name == "fit_release":
                    self.assertIs(b.shards.run_original_main, b.run_original_main)
                else:
                    self.assertIs(b.release._native_python_startup, b.native_python_startup)
                    self.assertIs(b.release._native_local_runner, b.native_local_runner)
                    self.assertIs(b.native.run_author, b.run_author)
                raise RuntimeError("synthetic release failure")
            with patch.object(b.release, name, fail):
                with self.assertRaises(RuntimeError):
                    wrapper()
            self.assertIs(b.release.admit_native, b.ORIGINAL_ADMIT)
            self.assertIs(b.shards.run_original_main, b.ORIGINAL_TRAIN_RUNNER)
            self.assertIs(b.release._native_python_startup, b.ORIGINAL_STARTUP)
            self.assertIs(b.release._native_local_runner, b.ORIGINAL_LOCAL_RUNNER)
            self.assertIs(b.native.run_author, b.ORIGINAL_AUTHOR_RUNNER)

    def test_readable_static_sources_and_pinned_call(self):
        for filename in ("runtime_bindings_v1.py", "queue_european_argmix_v1.py", "sitecustomize_gpu5_v1.py"):
            tree = ast.parse((BASE / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                    self.assertNotIn(name, ("exec", "eval", "compile", "FunctionType", "run_path"))
        original_call = dict(module="old", spec=dict(arguments={"unchanged": "arguments"}), source_refs=[])
        with patch.object(q.original, "make_call", return_value=original_call):
            call = q.make_call("release_fit", {"unchanged": "arguments"},
                b.native.ROOT / "experiments/results/synthetic_c4_call_only", python="metadata-only")
        self.assertEqual(call["module"], q.MODULE)
        self.assertEqual(call["spec"]["arguments"], {"unchanged": "arguments"})
        self.assertTrue(all(any(ref["path"] == path for ref in call["source_refs"]) for path in q.SOURCE_PATHS))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Synthetic per-job GPU checks and metadata factories; no scientific launches."""
import ast
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import device_binding_v1 as device
from . import c4_runtime_bindings_v1 as c4
from . import native_scientific_v1 as argformer
from . import queue_european_argmix_v1 as queue
from . import sitecustomize_multigpu_v1 as startup
from research.repairs.benchmark_followup_20261022.argmix.test_archive_path_v1 import ArchivePathTests

BASE = Path(__file__).parent


class StopBeforeNative(RuntimeError):
    pass


class MultigpuTests(unittest.TestCase):
    def setUp(self):
        self.title = SimpleNamespace(getproctitle=lambda: device.TITLE, setproctitle=lambda value: None)

    def environment(self, physical):
        return dict(ARGGNN_PHYSICAL_GPU=physical, CUDA_VISIBLE_DEVICES=device.UUID_BY_DEVICE[physical],
            ARGGNN_MAX_CUDA_RESERVED_BYTES=str(4 * 2**30))

    def test_device_uuid_pairs_accept_exact_jobs_and_reject_mismatch(self):
        for physical, uuid in device.UUID_BY_DEVICE.items():
            with patch.dict(os.environ, self.environment(physical)), patch.dict(sys.modules, setproctitle=self.title):
                self.assertEqual(device.require_job_device(require_title=True), (int(physical), uuid))
                for wrong in ("", "0", "02", "4", "GPU2"):
                    with patch.dict(os.environ, ARGGNN_PHYSICAL_GPU=wrong):
                        with self.assertRaises(ValueError):
                            device.require_job_device()
                for wrong_uuid in ("", "GPU-other", uuid + ",GPU-other"):
                    with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=wrong_uuid):
                        with self.assertRaises(ValueError):
                            device.require_job_device()
        for physical in ("2", "3", "5"):
            other = "3" if physical == "2" else "2"
            with patch.dict(os.environ, ARGGNN_PHYSICAL_GPU=physical, CUDA_VISIBLE_DEVICES=device.UUID_BY_DEVICE[other]):
                with self.assertRaises(ValueError):
                    device.require_job_device()

    def test_argformer_each_job_reaches_original_stage_boundary(self):
        spec = dict(method="argformer", stage="mlm")
        original_spec = copy.deepcopy(spec)
        for physical in device.UUID_BY_DEVICE:
            with patch.dict(os.environ, self.environment(physical)), \
                    patch.object(argformer.api, "validate_spec", return_value=Path("never-created")):
                def boundary(marker):
                    self.assertEqual(marker, "before_native_stage_scope")
                    raise StopBeforeNative("before output and native execution")
                with self.assertRaises(StopBeforeNative):
                    argformer.run_argformer(spec, bounded_runner=lambda *a, **kw: None, resource_check=boundary)
        self.assertEqual(spec, original_spec)

    def test_argformer_wrapper_science_and_argv_ast_unchanged(self):
        original_path = c4.native.ROOT / "research/repairs/benchmark_resume_20261021/gpu/native_scientific_gpu5_v1.py"
        original_function = next(node for node in ast.parse(original_path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_argformer")
        new_function = next(node for node in ast.parse((BASE / "native_scientific_v1.py").read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_argformer")
        original_function.body = [node for node in original_function.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and any(isinstance(arg, ast.Constant) and arg.value == "Native stages must inherit ROOT-admitted physical GPU5 visibility"
                    for arg in node.value.args))]
        new_function.body = [node for node in new_function.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "require_job_device")]
        self.assertEqual(ast.dump(original_function, include_attributes=False), ast.dump(new_function, include_attributes=False))

    def test_c4_admission_each_device_and_rejects_before_runtime(self):
        fractions = []
        cuda = SimpleNamespace(device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(total_memory=80 * 2**30),
            set_per_process_memory_fraction=lambda fraction, index: fractions.append((fraction, index)))
        result = (SimpleNamespace(cuda=cuda),)
        for physical in device.UUID_BY_DEVICE:
            with patch.dict(os.environ, self.environment(physical)), patch.dict(sys.modules, setproctitle=self.title), \
                    patch.object(c4.native, "runtime_admission") as admission, \
                    patch.object(c4.native, "source_imports", return_value=result):
                self.assertIs(c4.admit_native({}, {}, require_gpu=True), result)
                admission.assert_called_once()
                admission.reset_mock()
                os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-mismatch"
                with self.assertRaises(ValueError):
                    c4.admit_native({}, {}, require_gpu=True)
                admission.assert_not_called()
        self.assertEqual(fractions, [(0.05, 0)] * 3)

    def test_static_child_cpu_export_has_no_cuda_and_gpu_matches_each_job(self):
        fractions = []
        cuda = SimpleNamespace(device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(total_memory=80 * 2**30),
            set_per_process_memory_fraction=lambda fraction, index: fractions.append((fraction, index)))
        for physical in device.UUID_BY_DEVICE:
            with patch.dict(os.environ, self.environment(physical)), \
                    patch.dict(sys.modules, setproctitle=self.title, torch=SimpleNamespace(cuda=cuda)):
                before = len(fractions)
                with patch.object(sys, "argv", [startup.SOURCE_ROOT + "/export_torchscript.py", "--device", "cpu"]):
                    startup.admit_child()
                    self.assertEqual(len(fractions), before)
                for script in ("run_inference.py", "run_joint_inference.py"):
                    with patch.object(sys, "argv", [startup.SOURCE_ROOT + "/" + script, "--device", "cuda"]):
                        startup.admit_child()
                        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-wrong"):
                            with self.assertRaises(RuntimeError):
                                startup.admit_child()
        self.assertEqual(fractions, [(0.05, 0)] * 6)

    def test_stage_local_static_file_and_environment_restore(self):
        before = dict(os.environ)
        with tempfile.TemporaryDirectory(prefix="synthetic_", dir=BASE) as directory:
            with self.assertRaises(StopBeforeNative):
                with c4.native_python_startup(Path(directory), lambda value: None):
                    self.assertEqual((Path(directory) / "python_startup/sitecustomize.py").read_bytes(), c4.STARTUP.read_bytes())
                    raise StopBeforeNative()
        self.assertEqual(dict(os.environ), before)

    def test_c4_author_hooks_restore_on_failure(self):
        for name, wrapper in (("fit_release", c4.fit_release), ("infer_release", c4.infer_release)):
            def boundary(**arguments):
                self.assertIs(c4.release.admit_native, c4.admit_native)
                if name == "fit_release":
                    self.assertIs(c4.shards.run_original_main, c4.original.run_original_main)
                else:
                    self.assertIs(c4.release._native_python_startup, c4.native_python_startup)
                    self.assertIs(c4.release._native_local_runner, c4.original.native_local_runner)
                    self.assertIs(c4.native.run_author, c4.original.run_author)
                raise StopBeforeNative()
            with patch.object(c4.release, name, boundary):
                with self.assertRaises(StopBeforeNative):
                    wrapper()
            self.assertIs(c4.release.admit_native, c4.original.ORIGINAL_ADMIT)
            self.assertIs(c4.shards.run_original_main, c4.original.ORIGINAL_TRAIN_RUNNER)
            self.assertIs(c4.release._native_python_startup, c4.original.ORIGINAL_STARTUP)
            self.assertIs(c4.release._native_local_runner, c4.original.ORIGINAL_LOCAL_RUNNER)
            self.assertIs(c4.native.run_author, c4.original.ORIGINAL_AUTHOR_RUNNER)

    def test_factory_metadata_preserves_spec_python_output_and_source_pins(self):
        arguments = dict(write_headroom_bytes=1073741824, scientific_setting="unchanged")
        output = c4.native.ROOT / "experiments/results/metadata_only_multigpu_c4"
        python = c4.native.PREFIX / "bin/python3.12"
        old = queue.original.make_call("release_fit", arguments, output, python=python)
        new = queue.make_call("release_fit", arguments, output, python=python)
        self.assertEqual(new["spec"], old["spec"])
        self.assertEqual(new["python"], old["python"])
        self.assertEqual(new["function"], old["function"])
        self.assertEqual(new["module"], queue.MODULE)
        self.assertEqual(new["source_refs"][:len(old["source_refs"])], old["source_refs"])
        self.assertTrue(all(any(ref["path"] == path for ref in new["source_refs"]) for path in queue.SOURCE_PATHS))
        self.assertFalse(output.exists())

    def test_queue_uses_new_fit_infer_and_archive_path_only(self):
        callback = lambda *args, **kwargs: None
        for stage in ("release_fit", "release_infer"):
            with patch.object(c4, stage.replace("release_", "") + "_release", return_value="same") as function:
                spec = dict(stage=stage, arguments=dict(write_headroom_bytes=1, setting="same"), output_directory="same-output")
                before = copy.deepcopy(spec)
                self.assertEqual(queue.execute_stage(spec, resource_check=callback, execution_check=callback, bounded_runner=callback), "same")
                self.assertEqual(spec, before)
                self.assertEqual(function.call_args.kwargs["setting"], "same")
        with patch.object(queue.original, "execute_stage", return_value="same") as execute:
            queue.execute_stage(dict(stage="release_archive"), resource_check=callback,
                execution_check=callback, bounded_runner=callback)
            self.assertIsNot(execute.call_args.kwargs["bounded_runner"], callback)
        with patch.object(queue.original, "execute_stage", return_value="same") as execute:
            queue.execute_stage(dict(stage="release_frame"), resource_check=callback,
                execution_check=callback, bounded_runner=callback)
            self.assertIs(execute.call_args.kwargs["bounded_runner"], callback)


if __name__ == "__main__":
    unittest.main(verbosity=2)

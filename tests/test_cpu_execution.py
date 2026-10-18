"""Focused CPU pipeline handoff/supervisor test with fake numerical/process work."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from arggnn.benchmarking import cpu_execution as cpu


class CPUExecutionTest(unittest.TestCase):
    def test_actual_handoff_cpu_only_wait_cleanup_and_one_storage_measurement(self):
        source_root = cpu.ROOT
        with tempfile.TemporaryDirectory(dir=source_root / "tests", prefix="cpu_execution_") as directory:
            root = Path(directory)
            with patch.object(cpu, "ROOT", root), patch.object(cpu.c, "ROOT", root):
                output = root / "experiments/results/argformer_american_train_000/pipeline"
                output.mkdir(parents=True)
                spec = dict(output_directory=str(output), generation_spec={"parent": "fresh"},
                            inference_template={"stage": "shared_inference"})
                calls = []
                def first(config, **callbacks):
                    calls.append("generation")
                    return cpu._json(output / "generation_ready.synthetic.json", {"public": True})
                def second(config, **callbacks):
                    calls.append("inference")
                    self.assertEqual(config["input_ref"], cpu.c.file_ref(output / "generation_ready.synthetic.json"))
                    return cpu._json(output / "inference_ready.synthetic.json", {"public": True})
                def stage_runner(ref, frozen, kind, config):
                    stage_ref = cpu._json(output / (kind + "_spec.actual.json"), config)
                    actual = first(config) if kind == "generation" else second(config)
                    return actual, stage_ref
                with patch.object(cpu, "_run_stage", side_effect=stage_runner):
                    completion = cpu.run_pipeline(spec, job_ref={"path": "FAKE", "sha256": "f" * 64},
                                                  resource_check=Mock(), execution_check=Mock())
                result = cpu.c.read_ref(completion)
                bound = cpu.c.read_ref(result["inference_spec_ref"])
                self.assertEqual(calls, ["generation", "inference"])
                self.assertEqual(bound["input_ref"], result["generation_ready_ref"])
                self.assertNotIn("input_ref", spec["inference_template"])
                self.assertEqual(result["native_path_preparation"], "pending_separate_compact_focal_stage")
                self.assertFalse(result["test_labels_opened"])

                # A single actual project du is a prestart operation. The
                # supervisor loop below never calls it or scans project storage.
                capacity = {"max_output_bytes": 64 * 2**30}
                policy = dict(approved_new_storage_cap_bytes=64 * 2**30,
                              storage_ceiling_bytes=980000000000, human_storage_limit_bytes=1000000000000)
                with patch.object(cpu.subprocess, "run", return_value=types.SimpleNamespace(stdout="1000000 project\n")) as du:
                    storage = cpu._storage_admission(capacity, policy)
                self.assertEqual(storage["measured_bytes"], 1000000)
                du.assert_called_once()
                self.assertEqual(du.call_args.args[0][:2], ["du", "-sb"])

                supervised = root / "experiments/results/argformer_american_train_000/supervised"
                job = dict(job_id="cpu_synthetic", output_directory=str(supervised),
                           python=".runtime/env/bin/python", max_seconds=60,
                           max_rss_bytes=128 * 2**30, max_output_bytes=64 * 2**30)
                process = types.SimpleNamespace(pid=1234567, poll=Mock(side_effect=[None, 0]), wait=Mock(return_value=0))
                with patch.object(cpu, "validate_spec", return_value=job), \
                     patch.object(cpu, "_approved", return_value=policy), \
                     patch.object(cpu, "_storage_admission", return_value=storage) as admission, \
                     patch.object(cpu, "_own_group_rss", return_value=0), \
                     patch.object(cpu, "scope_bytes", return_value=0), \
                     patch.object(cpu, "_stop_group") as stop, \
                     patch.object(cpu.os, "killpg", side_effect=ProcessLookupError), \
                     patch.object(cpu.subprocess, "Popen", return_value=process) as spawn, \
                     patch.object(cpu.time, "sleep"):
                    terminal = cpu.execute({"path": "synthetic_spec.json", "sha256": "a" * 64})
                admission.assert_called_once()
                process.wait.assert_called_once()
                stop.assert_called_once_with(process)
                self.assertTrue(terminal["actual_direct_wait"])
                self.assertTrue(terminal["owned_process_group_absent"])
                self.assertEqual(terminal["actual_direct_returncode"], 0)
                self.assertFalse(terminal["GPU_used"])
                self.assertFalse(terminal["automatic_retry"])
                command = spawn.call_args
                self.assertTrue(command.kwargs["start_new_session"])
                self.assertEqual(command.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(command.kwargs["cwd"], root)
                for key in ("TMPDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "STDPOPSIM_CACHE", "MPLCONFIGDIR", "HF_HOME", "TORCH_HOME"):
                    self.assertTrue(Path(command.kwargs["env"][key]).is_relative_to(supervised), key)
                self.assertEqual(command.args[0][-3:-1], ["--child", "--job-ref"])
                self.assertTrue((supervised / "actual_wait.json").is_file())


if __name__ == "__main__":
    unittest.main()

"""Focused synthetic regression for the archive PATH correction; no launches."""
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from . import queue_european_argmix_v2 as q


class ArchivePathTests(unittest.TestCase):
    def test_runtime_path_only_change_and_actual_wait_identity(self):
        argv = [str(q.release.native.CLI), "--config", "unchanged.yaml", "--split", "train", "--output", "unchanged.pt"]
        env = dict(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8")
        limits = dict(max_seconds=604800)
        cwd = q.release.native.SOURCE
        log_path = Path("unchanged.log")
        actual_wait = dict(actual_direct_wait=True, actual_direct_returncode=0,
            native_process_absent=True, outer_owned_group_cleanup_pending=True)
        seen = []

        def bounded(command, **arguments):
            self.assertIs(command, argv)
            self.assertIs(arguments["cwd"], cwd)
            self.assertIs(arguments["limits"], limits)
            self.assertIs(arguments["log_path"], log_path)
            self.assertEqual({key: value for key, value in arguments["env"].items() if key != "PATH"}, env)
            seen.append(arguments["env"]["PATH"])
            return actual_wait

        with patch.dict(os.environ, PATH="/usr/bin:/bin"):
            result = q.archive_runner(bounded)(argv, cwd=cwd, env=env, limits=limits, log_path=log_path)
            self.assertEqual(os.environ["PATH"], "/usr/bin:/bin")
        self.assertIs(result, actual_wait)
        self.assertEqual(seen, [str(q.release.native.PREFIX / "bin") + ":/usr/bin:/bin"])
        self.assertNotIn("PATH", env)
        self.assertEqual(Path(shutil.which("python", path=seen[0])).resolve(),
            (q.release.native.PREFIX / "bin/python3.12").resolve())

    def test_explicit_path_is_preserved_after_runtime_prefix(self):
        env = dict(PATH="/explicit/bin", CUDA_VISIBLE_DEVICES="")
        captured = []
        sentinel = object()

        def bounded(argv, **arguments):
            captured.append(arguments["env"])
            return sentinel

        with patch.dict(os.environ, PATH="/different/inherited/path"):
            result = q.archive_runner(bounded)([], cwd="same", env=env, limits={}, log_path="same.log")
        self.assertIs(result, sentinel)
        self.assertEqual(captured, [dict(PATH=str(q.release.native.PREFIX / "bin") + ":/explicit/bin", CUDA_VISIBLE_DEVICES="")])
        self.assertEqual(env, dict(PATH="/explicit/bin", CUDA_VISIBLE_DEVICES=""))

    def test_only_archive_dispatch_wraps_runner(self):
        resource_check, execution_check, bounded_runner = object(), object(), lambda *a, **kw: None
        result = object()
        for stage in ("release_archive", "release_fit", "release_infer", "release_frame"):
            spec = dict(stage=stage, arguments={"unchanged": True}, output_directory="unchanged")
            with patch.object(q.original, "execute_stage", return_value=result) as execute:
                self.assertIs(q.execute_stage(spec, resource_check=resource_check,
                    execution_check=execution_check, bounded_runner=bounded_runner), result)
                arguments = execute.call_args
                self.assertIs(arguments.args[0], spec)
                self.assertIs(arguments.kwargs["resource_check"], resource_check)
                self.assertIs(arguments.kwargs["execution_check"], execution_check)
                runner = arguments.kwargs["bounded_runner"]
                if stage == "release_archive":
                    self.assertIsNot(runner, bounded_runner)
                else:
                    self.assertIs(runner, bounded_runner)


if __name__ == "__main__":
    unittest.main(verbosity=2)

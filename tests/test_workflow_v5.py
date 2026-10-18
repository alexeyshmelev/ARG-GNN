"""Tiny repair-binding checks; production leases and scientific work untouched."""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from arggnn.benchmarking import workflow_v4 as old
from arggnn.benchmarking import workflow_v5 as v5


class DummyReporter:
    def __init__(self, *args):
        pass
    def submit(self, *args, **kwargs):
        pass
    def close(self, *args):
        pass


class WorkflowV5Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=v5.c.ROOT / "tests")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        names = ("MODULE", "current_policy", "_own_group_rss", "execute", "child", "run",
                 "external_result", "external_resources", "storage_projection", "available_ram_bytes",
                 "ProgressReporter", "scope_bytes")
        saved = {name: getattr(v5.base, name) for name in names}
        self.addCleanup(lambda: [setattr(v5.base, key, value) for key, value in saved.items()])

    def test_repair_approval_and_drain_are_separate_from_both_live_predecessors(self):
        parent = dict(path="v3", sha256="3" * 64)
        continuation = dict(path="v4", sha256="4" * 64)
        repair = dict(path="v5", sha256="5" * 64)
        canonical = dict(approved_workflow_ref=parent, approved_continuation_ref=continuation,
                         approved_repair_continuation_ref=repair, workflow_drain_ref=parent,
                         continuation_drain_ref=continuation)
        untouched = copy.deepcopy(canonical)
        with patch.object(v5._core, "_BASE_POLICY", return_value=canonical):
            observed = v5.current_policy()
        self.assertEqual(observed["approved_workflow_ref"], repair)
        self.assertEqual(observed["canonical_parent_workflow_ref"], parent)
        self.assertEqual(observed["canonical_continuation_workflow_ref"], continuation)
        self.assertNotIn("workflow_drain_ref", observed)
        self.assertEqual(canonical, untouched)
        canonical["repair_continuation_drain_ref"] = repair
        with patch.object(v5._core, "_BASE_POLICY", return_value=canonical):
            self.assertEqual(v5.current_policy()["workflow_drain_ref"], repair)
        for invalid in (None, {}, dict(path="v5")):
            with patch.object(v5._core, "_BASE_POLICY", return_value=dict(
                    approved_workflow_ref=parent, approved_continuation_ref=continuation,
                    approved_repair_continuation_ref=invalid)):
                with self.assertRaisesRegex(RuntimeError, "approved_repair_continuation_ref"):
                    v5.current_policy()

    def test_isolated_core_preserves_public_v4_source_and_namespace(self):
        path = Path(old.__file__)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        old_callback = old.current_policy
        v5.install_bindings()
        self.assertIsNot(v5._core, old)
        self.assertEqual(old.MODULE, "arggnn.benchmarking.workflow_v4")
        self.assertIs(old.current_policy, old_callback)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertEqual(v5.base.MODULE, "arggnn.benchmarking.workflow_v5")
        for name in ("current_policy", "execute", "child", "run", "external_result",
                     "external_resources", "storage_projection", "available_ram_bytes"):
            self.assertIs(getattr(v5.base, name), getattr(v5, name))
        self.assertIs(v5.base._own_group_rss, v5.group_pss)
        self.assertIs(v5.execute.__globals__, v5._core.__dict__)
        self.assertEqual(v5.execute.__globals__["MODULE"], v5.MODULE)

    def test_public_v4_reporter_endpoints_stay_unchanged(self):
        from arggnn.benchmarking import workflow_reporting_v4 as previous
        before = (previous.BEGIN, previous.END, previous.STATUS_PATH, previous.SCORE_PATH,
                  previous._core.RUNTIME_PATH)
        from arggnn.benchmarking import workflow_reporting_v5 as repaired
        self.assertEqual(before, (previous.BEGIN, previous.END, previous.STATUS_PATH,
                                  previous.SCORE_PATH, previous._core.RUNTIME_PATH))
        self.assertEqual(previous.STATUS_PATH, "research/benchmark_continuation_current.json")
        self.assertEqual(previous._core.RUNTIME_PATH, ".runtime/benchmarking/workflow_reporting_v4")
        self.assertNotEqual(previous.STATUS_PATH, repaired.STATUS_PATH)
        self.assertNotEqual(previous.BEGIN, repaired.BEGIN)
        self.assertEqual(repaired._private._core.RUNTIME_PATH,
                         ".runtime/benchmarking/workflow_reporting_v5")

    def test_all_external_accounting_uses_new_helper_once_and_preserves_unknown_caps(self):
        from arggnn.benchmarking import workflow_predecessor_v5 as helper
        plan = dict(workflow_external_predecessors=[dict(resource="gpu0")], prior_unknown_rss_bytes=256)
        with patch.object(v5._core, "_BASE_EXTERNAL_RESOURCES", return_value=(2, 100)), \
                patch.object(helper, "external_resources", return_value=(1, 512)) as resources:
            self.assertEqual(v5.external_resources(plan), (3, 868))
            resources.assert_called_once_with(plan)
        with patch.object(v5._core, "_BASE_STORAGE", return_value=1000) as storage, \
                patch.object(helper, "storage_reservation_bytes", return_value=50) as predecessor_storage:
            self.assertEqual(v5.storage_projection(plan, {}, 30), 1050)
            storage.assert_called_once_with(plan, {}, 30)
            predecessor_storage.assert_called_once_with(plan)
        task = dict(workflow_predecessor={"synthetic": True})
        with patch.object(helper, "external_result", return_value={"status": "succeeded"}) as result, \
                patch.object(v5._core, "_BASE_EXTERNAL_RESULT", side_effect=AssertionError("old helper")):
            self.assertEqual(v5.external_result(task), {"status": "succeeded"})
            result.assert_called_once_with(task)

    def test_external_pss_credit_uses_new_helper_exact_identity_and_no_fallback_credit(self):
        from arggnn.benchmarking import workflow_predecessor_v5 as helper
        binding = dict(pid=123, start_ticks=45, pgid=123, session=123, max_rss_bytes=256)
        root = dict(binding, state="S")
        tracker = SimpleNamespace(root=root, latest=dict(members=[
            dict(accounted_bytes=300, method="smaps_rollup_PSS"),
            dict(accounted_bytes=1000, method="conservative_RSS_fallback")]), measure=lambda: 1300)
        plan = dict(workflow_external_predecessors=[{}])
        with patch.object(helper, "live_group_bindings", return_value=[binding]) as groups, \
                patch.object(v5._core, "identity", return_value=root), \
                patch.object(v5._core, "OwnedGroup", return_value=tracker):
            self.assertEqual(v5._known_credit(plan, self.directory / "absent.json"), 256)
            groups.assert_called_once_with(plan)
        with patch.object(helper, "live_group_bindings", return_value=[binding]), \
                patch.object(v5._core, "identity", return_value=dict(root, start_ticks=46)):
            self.assertEqual(v5._known_credit(plan, self.directory / "absent.json"), 0)

    def test_run_binds_only_repair_reporter_and_own_context(self):
        marker = object()
        reporter = SimpleNamespace(ProgressReporter=marker)
        ref = dict(path="synthetic", sha256="a" * 64)
        plan = dict(records_directory=str(self.directory))
        with patch.dict(sys.modules, {"arggnn.benchmarking.workflow_reporting_v5": reporter}), \
                patch.object(v5.c, "read_ref", return_value=plan), \
                patch.object(v5._core, "_BASE_RUN", return_value={"synthetic": True}) as run:
            self.assertEqual(v5.run(ref, execute_jobs=True), {"synthetic": True})
        self.assertIs(v5.base.ProgressReporter, marker)
        self.assertEqual(v5._core._RUN_CONTEXT, (plan, self.directory / "state.json"))
        self.assertIsNone(old._RUN_CONTEXT)
        run.assert_called_once_with(ref, execute_jobs=True)

    def test_occupied_predecessor_gpu_lease_blocks_before_materialization_or_spawn(self):
        class StopAfterPoll(Exception):
            pass
        ref = dict(path="synthetic", sha256="a" * 64)
        target = dict(id="new_gpu", resource="gpu0", dependencies=[], max_rss_bytes=128,
                      max_output_bytes=1024, output_directory="new_scientific_output")
        predecessor = dict(id="old_gpu", resource="external", dependencies=[])
        plan = dict(records_directory="records", tasks=[predecessor, target],
                    cpu_parallelism=2, max_parallel_rss_bytes=10000,
                    host_ram_margin_bytes=64, storage_ceiling_bytes=980000000000)
        fake_root = self.directory
        lock_path = fake_root / ".runtime/locks/gpu_0.lock"
        lock_path.parent.mkdir(parents=True)
        v5.install_bindings()
        with lock_path.open("a+") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(v5.c, "ROOT", fake_root), \
                    patch.object(v5.base, "load_plan", return_value=plan), \
                    patch.object(v5.base, "current_policy", return_value=dict(approved_workflow_ref=ref)), \
                    patch.object(v5.base, "external_result", return_value=None), \
                    patch.object(v5.base, "external_resources", return_value=(0, 256)), \
                    patch.object(v5.base, "available_ram_bytes", return_value=100000), \
                    patch.object(v5.base, "storage_projection", return_value=0), \
                    patch.object(v5.base, "ProgressReporter", DummyReporter), \
                    patch.object(v5.base, "materialize", side_effect=AssertionError("must not materialize")) as materialize, \
                    patch.object(v5.base.subprocess, "Popen", side_effect=AssertionError("must not spawn")) as spawn, \
                    patch.object(v5.base.time, "sleep", side_effect=StopAfterPoll):
                with self.assertRaises(StopAfterPoll):
                    v5._core._BASE_RUN(ref, execute_jobs=True, poll_seconds=0)
        state = json.loads((fake_root / "records/state.json").read_text())
        self.assertEqual(state["tasks"]["new_gpu"]["status"], "pending")
        self.assertEqual(state["tasks"]["new_gpu"]["waiting_reason"], "GPU0 lease occupied")
        self.assertEqual(state["reserved_rss_bytes_including_external"], 256)
        self.assertFalse((fake_root / "new_scientific_output").exists())
        materialize.assert_not_called()
        spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()

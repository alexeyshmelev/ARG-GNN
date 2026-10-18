"""Tiny scheduler polls against a temporary lease; no science or live lock access."""
import ast
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arggnn.benchmarking import workflow
from .guard_order_v1 import compile_scheduler_run


class StopPoll(BaseException):
    pass


class DummyReporter:
    def __init__(self, *args):
        pass
    def submit(self, *args, **kwargs):
        pass
    def close(self, *args):
        pass


class SchedulerGuardOrderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=workflow.c.ROOT / "tests")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.run = compile_scheduler_run(workflow.run)

    def poll(self, *, resource="cpu", occupied=False, storage=0, available=100000,
             expect_materialize=False):
        ref = dict(path="synthetic", sha256="a" * 64)
        task = dict(id="candidate", resource=resource, dependencies=[], max_rss_bytes=128,
                    max_output_bytes=1024, output_directory="science_never_created")
        plan = dict(records_directory="records", tasks=[task], cpu_parallelism=2,
                    max_parallel_rss_bytes=10000, host_ram_margin_bytes=64,
                    storage_ceiling_bytes=10000)
        lock_path = self.root / ".runtime/locks/gpu_0.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        records = self.root / "records"
        records.mkdir(exist_ok=True)
        (records / "state.json").write_text(json.dumps(dict(
            plan_ref=ref, tasks={"candidate": dict(status="pending", external=False)})))
        events = []
        def ram():
            events.append("fresh_RAM")
            return available
        def materialize(*args):
            events.append("materialize")
            raise StopPoll()
        with lock_path.open("a+") as owner:
            if occupied:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(workflow.c, "ROOT", self.root), \
                    patch.object(workflow, "load_plan", return_value=plan), \
                    patch.object(workflow, "current_policy", return_value=dict(approved_workflow_ref=ref)), \
                    patch.object(workflow, "external_resources", return_value=(0, 256)), \
                    patch.object(workflow, "available_ram_bytes", side_effect=ram) as memory, \
                    patch.object(workflow, "storage_projection", return_value=storage), \
                    patch.object(workflow, "ProgressReporter", DummyReporter), \
                    patch.object(workflow, "materialize", side_effect=materialize) as factory, \
                    patch.object(workflow.subprocess, "Popen", side_effect=AssertionError("science forbidden")) as spawn, \
                    patch.object(workflow.time, "sleep", side_effect=StopPoll):
                try:
                    self.run(ref, execute_jobs=True, poll_seconds=0)
                except StopPoll:
                    pass
        spawn.assert_not_called()
        self.assertFalse((self.root / "science_never_created").exists())
        self.assertEqual(factory.call_count, int(expect_materialize))
        state = json.loads((self.root / "records/state.json").read_text())
        return state["tasks"]["candidate"], events, memory.call_count

    def test_blocked_gpu_lease_never_samples_host_ram(self):
        entry, events, count = self.poll(resource="gpu0", occupied=True)
        self.assertEqual(entry["waiting_reason"], "GPU0 lease occupied")
        self.assertEqual(events, [])
        self.assertEqual(count, 0)

    def test_blocked_storage_never_samples_host_ram(self):
        entry, events, count = self.poll(storage=9999)
        self.assertEqual(entry["waiting_reason"], "Storage reservation would exceed ceiling")
        self.assertEqual(events, [])
        self.assertEqual(count, 0)

    def test_eligible_candidate_samples_fresh_ram_before_materialize(self):
        for resource in ("cpu", "gpu0"):
            with self.subTest(resource=resource):
                entry, events, count = self.poll(resource=resource, expect_materialize=True)
                self.assertEqual(events, ["fresh_RAM", "materialize"])
                self.assertEqual(count, 1)

    def test_same_ram_limit_still_blocks_eligible_candidate(self):
        entry, events, count = self.poll(available=447)
        # Full256 predecessor reserve +128 candidate +64 host margin requires448.
        self.assertEqual(entry["waiting_reason"], "Host available RAM")
        self.assertEqual(events, ["fresh_RAM"])
        self.assertEqual(count, 1)

    def test_original_source_globals_and_keyword_defaults_preserved(self):
        path = Path(workflow.__file__)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        original_code = workflow.run.__code__
        cloned = compile_scheduler_run(workflow.run)
        self.assertIs(cloned.__globals__, workflow.run.__globals__)
        self.assertEqual(cloned.__kwdefaults__, workflow.run.__kwdefaults__)
        self.assertEqual(inspect.signature(cloned), inspect.signature(workflow.run))
        self.assertIs(workflow.run.__code__, original_code)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()

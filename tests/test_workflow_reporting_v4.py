import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import workflow_reporting as original
from arggnn.benchmarking import workflow_reporting_v4 as continuation


class ReportingCoexistence(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=c.ROOT / ".runtime")
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(c, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.directory.cleanup()

    def test_private_core_does_not_change_original(self):
        self.assertIsNot(original, continuation._core)
        self.assertNotEqual(original.STATUS_PATH, continuation._core.STATUS_PATH)
        self.assertNotEqual(original.SCORE_PATH, continuation._core.SCORE_PATH)
        self.assertNotEqual(original.RUNTIME_PATH, continuation._core.RUNTIME_PATH)
        self.assertEqual(original.BEGIN, "<!-- BEGIN EXECUTABLE BENCHMARK WORKFLOW -->")

    def test_alternating_publishers_preserve_other_readme_blocks(self):
        path = self.root / "README.md"
        path.write_text("Introduction\n")
        old = original.BEGIN + "\nold status\n" + original.END
        new = continuation.BEGIN + "\nnew status\n" + continuation.END
        path.write_text(original._readme(old))
        path.write_text(continuation._core._readme(new))
        path.write_text(original._readme(old.replace("old status", "old finished")))
        path.write_text(continuation._core._readme(new.replace("new status", "new running")))
        actual = path.read_text()
        self.assertIn("old finished", actual)
        self.assertIn("new running", actual)
        self.assertTrue(actual.startswith("Introduction\n"))
        self.assertEqual(actual.count(original.BEGIN), 1)
        self.assertEqual(actual.count(continuation.BEGIN), 1)

    def test_shared_lock_blocks_second_writer_and_releases(self):
        path = self.root / ".runtime/benchmarking/workflow_reporting/reporting.lock"
        path.parent.mkdir(parents=True)
        with path.open("a+") as held, patch.object(continuation._core, "publish", return_value={"status": "updated"}) as writer:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(continuation.publish({}, "unused"), {"status": "busy"})
            writer.assert_not_called()
            fcntl.flock(held, fcntl.LOCK_UN)
            self.assertEqual(continuation.publish({}, "unused"), {"status": "updated"})
            writer.assert_called_once()


if __name__ == "__main__":
    unittest.main()

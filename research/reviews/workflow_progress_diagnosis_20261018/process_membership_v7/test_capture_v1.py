"""Deterministic membership snapshot races, plus one actual tiny Linux fork."""
import copy
import json
import select
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from arggnn.benchmarking import workflow_v4 as frozen
from .capture_v1 import capture_members, repaired_owned_group_class


def row(pid=100, start=10, pgid=100, session=100):
    return dict(pid=pid, start_ticks=start, pgid=pgid, session=session, ppid=1, state="S")


def tracker():
    root = row()
    return SimpleNamespace(root=root, history={100: copy.deepcopy(root)})


def capture(value, rows, pids=None):
    return capture_members(value, group_pids=lambda group: set(rows) if pids is None else pids,
                           identity=lambda pid: rows[pid], same=frozen.same, need=frozen.need)


class CaptureRaceTest(unittest.TestCase):
    def test_original_bug_and_never_tracked_mismatch_is_not_owned(self):
        root = row()
        stale = row(pid=200, start=20, pgid=200, session=200)
        old = frozen.OwnedGroup.__new__(frozen.OwnedGroup)
        old.root = root
        old.history = {100: root}
        with patch.object(frozen, "group_pids", return_value={100, 200}), \
                patch.object(frozen, "identity", side_effect=lambda pid: {100: root, 200: stale}[pid]):
            with self.assertRaisesRegex(RuntimeError, "Owned session/group changed"):
                old.capture()
        fixed = tracker()
        self.assertEqual(capture(fixed, {100: root, 200: stale}), {100: root})
        self.assertNotIn(200, fixed.history)

    def test_current_new_member_commits_exact_identity_once(self):
        fixed = tracker()
        child = row(pid=200, start=20)
        self.assertEqual(capture(fixed, {200: child}), {200: child})
        self.assertEqual(fixed.history[200], child)
        self.assertEqual(capture(fixed, {200: dict(child, state="R")}), {200: dict(child, state="R")})

    def test_known_start_group_and_session_changes_still_fail(self):
        for changed in (dict(start_ticks=21), dict(pgid=200), dict(session=200),
                        dict(start_ticks=21, pgid=200, session=200)):
            with self.subTest(changed=changed):
                fixed = tracker()
                child = row(pid=200, start=20)
                fixed.history[200] = child
                with self.assertRaisesRegex(RuntimeError, "identity changed or was reused"):
                    capture(fixed, {200: dict(child, **changed)})
                self.assertEqual(fixed.history[200], child)

    def test_vanished_is_skipped_but_permission_and_predating_member_errors_propagate(self):
        for error in (FileNotFoundError(), ProcessLookupError()):
            self.assertEqual(capture_members(tracker(), group_pids=lambda group: {200},
                identity=lambda pid: (_ for _ in ()).throw(error), same=frozen.same, need=frozen.need), {})
        with self.assertRaises(PermissionError):
            capture_members(tracker(), group_pids=lambda group: {200},
                identity=lambda pid: (_ for _ in ()).throw(PermissionError("unreadable live PID")),
                same=frozen.same, need=frozen.need)
        with self.assertRaisesRegex(RuntimeError, "predates"):
            capture(tracker(), {200: row(pid=200, start=9)})

    def test_actual_fork_leaves_group_between_enumeration_and_identity_then_actually_waits(self):
        program = r"""
import json,os,sys
read,write=os.pipe();ack_read,ack_write=os.pipe()
child=os.fork()
if child==0:
 os.close(write);os.close(ack_read)
 command=os.read(read,1)
 if command==b'M':
  os.setpgid(0,0);os.write(ack_write,b'M')
 os.read(read,1)
 os._exit(0)
os.close(read);os.close(ack_write)
print(json.dumps(dict(child=child)),flush=True)
if sys.stdin.readline().strip()=='MOVE':
 os.write(write,b'M');os.read(ack_read,1);print('MOVED',flush=True)
sys.stdin.readline();os.write(write,b'S')
waited,status=os.waitpid(child,0)
print(json.dumps(dict(waited=waited,child_returncode=os.waitstatus_to_exitcode(status))),flush=True)
"""
        process = subprocess.Popen([sys.executable, "-B", "-s", "-c", program],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        def line():
            self.assertTrue(select.select([process.stdout], [], [], 5)[0], "Toy CPU response timeout")
            return process.stdout.readline().strip()
        try:
            child = json.loads(line())["child"]
            actual_root = frozen.identity(process.pid)
            observed_child = frozen.identity(child)
            self.assertEqual(observed_child["pgid"], actual_root["pgid"])
            def snapshot_then_move(group):
                candidates = frozen.group_pids(group)
                self.assertIn(child, candidates)
                process.stdin.write("MOVE\n");process.stdin.flush()
                self.assertEqual(line(), "MOVED")
                return candidates
            core = SimpleNamespace(OwnedGroup=frozen.OwnedGroup, group_pids=snapshot_then_move,
                                   identity=frozen.identity, same=frozen.same, need=frozen.need)
            repaired = repaired_owned_group_class(core)(process.pid)
            got = repaired.capture()
            self.assertIn(process.pid, got)
            self.assertNotIn(child, got)
            self.assertNotIn(child, repaired.history)
            self.assertEqual(frozen.identity(child)["pgid"], child)
            process.stdin.write("STOP\n");process.stdin.flush()
            waited = json.loads(line())
            self.assertEqual(waited, dict(waited=child, child_returncode=0))
            self.assertEqual(process.wait(timeout=5), 0)
        finally:
            if process.poll() is None:
                process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()


if __name__ == "__main__":
    unittest.main()

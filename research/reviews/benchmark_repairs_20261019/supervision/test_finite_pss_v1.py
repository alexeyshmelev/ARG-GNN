"""Bounded CPU fixtures for the isolated finite PSS replacement."""
from __future__ import annotations
from pathlib import Path
import copy
import json
import os
import select
import signal
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch

from arggnn.benchmarking import workflow_v4 as v4
from research.reviews.benchmark_repairs_20261019.supervision import finite_pss_v1 as repair


def row(pid=100, start=10, pgid=100, session=100, ppid=1, state="S"):
    return dict(pid=pid, start_ticks=start, pgid=pgid, session=session, ppid=ppid, state=state)


class FakeBase:
    def __init__(self, pid):
        self.root = fake.identity(pid)
        self.history = {pid: self.root}
        self.latest = None
        self.peak_accounted_bytes = self.peak_summed_RSS_bytes = 0
        self.samples = self.fallback_samples = 0
        self.adopted_waits = []
        self.direct_wait = None
        self.cleanup_errors = []


fake = types.SimpleNamespace(OwnedGroup=FakeBase, identity=None, group_pids=None,
                             _read=None, need=v4.need, same=v4.same)


def init_fake(identities=None, snapshots=None, reader=None):
    identities = identities or {100: row(), 101: row(101, 20)}
    fake.identity = lambda pid: dict(identities[pid])
    fake.group_pids = lambda pgid: set(identities)
    fake._read = reader or (lambda path: "Pss: 20 kB\nRss: 40 kB\n")
    return repair.repaired_owned_group_class(fake)(100), identities


MEMORY_PROGRAM = """
import json,mmap,os,sys
shared=mmap.mmap(-1,16*2**20,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
for offset in range(0,16*2**20,4096):shared[offset]=1
read,write=os.pipe();ar,aw=os.pipe();pid=os.fork()
if pid==0:
 os.close(write);os.close(ar)
 while True:
  command=os.read(read,1)
  if command==b'A':
   private=mmap.mmap(-1,8*2**20,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
   for offset in range(0,8*2**20,4096):private[offset]=2
   os.write(aw,b'A')
  else:break
 os._exit(0)
os.close(read);os.close(aw)
print(json.dumps(dict(child=pid)),flush=True)
if sys.stdin.readline().strip()=='ALLOCATE':
 os.write(write,b'A');os.read(ar,1);print('ALLOCATED',flush=True)
sys.stdin.readline();os.write(write,b'S');os.waitpid(pid,0)
"""

CHURN_PROGRAM = """
import os,signal,time
live=set()
def finish(sig,frame):
 for kid in live:
  try:os.kill(kid,signal.SIGTERM)
  except ProcessLookupError:pass
 for kid in list(live):
  try:os.waitpid(kid,0)
  except ChildProcessError:pass
 raise SystemExit(0)
signal.signal(signal.SIGTERM,finish)
print('READY',flush=True)
end=time.monotonic()+8
while time.monotonic()<end:
 kid=os.fork()
 if kid==0:
  signal.signal(signal.SIGTERM,signal.SIG_DFL)
  time.sleep(.012);os._exit(0)
 live.add(kid)
 for old in list(live):
  done,status=os.waitpid(old,os.WNOHANG)
  if done:live.remove(old)
 time.sleep(.001)
finish(0,None)
"""

ORPHAN_PROGRAM = """
import os,signal,time
pid=os.fork()
if pid==0:
 signal.signal(signal.SIGTERM,lambda sig,frame:os._exit(9))
 print('CHILD '+str(os.getpid()),flush=True)
 while True:time.sleep(.01)
time.sleep(.08)
os._exit(7)
"""


class SyntheticTest(unittest.TestCase):
    def test_pid_reuse_accepts_owned_new_generation_and_keeps_root_exact(self):
        tracker, identities = init_fake()
        original = dict(tracker.root)
        tracker.measure()
        identities[101] = row(101, 30)
        self.assertEqual(tracker.measure(), 40 * 1024)
        self.assertIn((101, 20), tracker.history)
        self.assertIn((101, 30), tracker.history)
        self.assertEqual(tracker.root, original)
        self.assertTrue(tracker.latest["complete"])

    def test_foreign_reused_numeric_pid_is_excluded_from_accounting_and_signal(self):
        tracker, identities = init_fake()
        tracker.measure()
        identities[101] = row(101, 30, pgid=900, session=900)
        sent = []
        with patch.object(repair, "_pidfd_open", side_effect=lambda pid: sent.append(pid) or 123), \
             patch.object(repair, "_pidfd_signal"), patch.object(repair.os, "close"):
            tracker._signal(signal.SIGTERM)
        self.assertEqual(sent, [100])
        self.assertEqual(tracker.measure(), 20 * 1024)
        self.assertNotIn(101, [member["pid"] for member in tracker.latest["members"]])

    def test_numeric_root_reuse_is_never_admitted_and_root_is_unchanged(self):
        tracker, identities = init_fake()
        original = dict(tracker.root)
        identities[100] = row(100, 99)
        self.assertNotIn(100, tracker.capture())
        self.assertEqual(tracker.root, original)

    def test_recycled_root_and_its_foreign_children_are_all_excluded(self):
        tracker, identities = init_fake()
        original = dict(tracker.root)
        tracker.measure()
        identities[100] = row(100, 1000)
        identities[101] = row(101, 1001)
        identities[102] = row(102, 1002)
        self.assertEqual(tracker.measure(),0)
        self.assertEqual(tracker.capture(),{})
        with patch.object(repair,"_pidfd_open") as opener:
            tracker._signal(signal.SIGTERM)
        opener.assert_not_called()
        self.assertEqual(tracker.root,original)
        self.assertTrue(tracker.receipt()["original_root_identity_invalidated"])
        # Exclusion survives the unrelated replacement leader's later exit.
        del identities[100]
        old_identity=fake.identity
        def missing_root(pid):
            try:return old_identity(pid)
            except KeyError:raise ProcessLookupError(pid)
        fake.identity=missing_root
        self.assertEqual(tracker.capture(),{})

    def test_root_replacement_during_enumeration_excludes_whole_new_group(self):
        tracker, identities = init_fake()
        original = dict(tracker.root)
        def replace_during_enumeration(pgid):
            identities[100] = row(100, 1000)
            identities[101] = row(101, 1001)
            # The numeric root can be absent from the separate enumeration.
            return [101]
        fake.group_pids = replace_during_enumeration
        self.assertEqual(tracker.capture(),{})
        self.assertNotIn((101,1001),tracker.history)
        self.assertEqual(tracker.root,original)
        with patch.object(repair,"_pidfd_open") as opener:
            tracker._signal(signal.SIGTERM)
        opener.assert_not_called()
        self.assertTrue(tracker.receipt()["original_root_identity_invalidated"])

    def test_births_and_exits_during_read_are_incomplete_without_credit(self):
        tracker, identities = init_fake()
        root_read = False
        def reading(path):
            nonlocal root_read
            if "/100/" in str(path) and not root_read:
                root_read = True
                identities[102] = row(102, 40)
                del identities[101]
            return "Pss: 20 kB\nRss: 40 kB\n"
        fake._read = reading
        old_identity = fake.identity
        def identity(pid):
            try:return old_identity(pid)
            except KeyError:raise ProcessLookupError(pid)
        fake.identity = identity
        self.assertEqual(tracker.measure(), 20*1024)
        self.assertFalse(tracker.latest["complete"])
        self.assertEqual(tracker.latest["headroom_credit_bytes"], 0)
        self.assertEqual(tracker.latest["reservation_policy"], "retain_full_task_cap")
        self.assertEqual(tracker.latest["new_generation_count"], 1)
        self.assertEqual(tracker.latest["retired_generation_count"], 1)
        self.assertEqual(tracker.latest["unmeasured_new_generations"][0]["pid"], 102)

    def test_reuse_during_read_discards_old_measurement_and_measures_next_poll(self):
        tracker, identities = init_fake()
        changed = False
        def reading(path):
            nonlocal changed
            if "/101/" in str(path) and not changed:
                identities[101] = row(101, 30)
                changed = True
            return "Pss: 20 kB\nRss: 40 kB\n"
        fake._read = reading
        self.assertEqual(tracker.measure(), 20*1024)
        self.assertFalse(tracker.latest["complete"])
        self.assertEqual(tracker.latest["headroom_credit_bytes"], 0)
        self.assertEqual(tracker.latest["invalidated_generations"][0]["reason"],
                         "generation_changed_after_memory_read")
        self.assertEqual(tracker.measure(), 40*1024)
        self.assertTrue(tracker.latest["complete"])

    def test_RSS_only_excess_is_diagnostic_and_bounded_PSS_retry_can_confirm(self):
        attempts = []
        def deny(path):
            attempts.append(str(path))
            if str(path).endswith("smaps_rollup"):raise PermissionError("test")
            return "10000 10000 0\n"
        tracker, _ = init_fake(reader=deny)
        self.assertEqual(tracker.measure(), 0)
        self.assertGreater(tracker.latest["summed_RSS_bytes"], 50*2**20)
        self.assertFalse(tracker.latest["complete"])
        self.assertEqual(tracker.latest["headroom_credit_bytes"], 0)
        self.assertEqual(sum(path.endswith("smaps_rollup") for path in attempts), 4)
        counts = {}
        def retry(path):
            pid = Path(path).parent.name
            counts[pid] = counts.get(pid,0)+1
            if counts[pid]==1:raise PermissionError("transient")
            return "Pss: 20 kB\nRss: 40000 kB\n"
        fake._read = retry
        self.assertEqual(tracker.measure(), 40*1024)
        self.assertTrue(tracker.latest["complete"])
        self.assertTrue(all(member["PSS_confirmation_after_retry"]
                            for member in tracker.latest["members"]))

    def test_signal_open_then_identity_check_does_not_signal_reused_process(self):
        tracker, identities = init_fake()
        def open_pid(pid):
            if pid == 101:identities[pid] = row(101, 31)
            return pid
        with patch.object(repair, "_pidfd_open", side_effect=open_pid), \
             patch.object(repair, "_pidfd_signal") as send, \
             patch.object(repair.os, "close"):
            tracker._signal(signal.SIGTERM)
        self.assertEqual(send.call_args_list, [((100, signal.SIGTERM),)])

    def test_history_and_events_are_bounded_under_many_generations(self):
        tracker, identities = init_fake()
        for start in range(20, 400):
            identities[101] = row(101, start)
            tracker.capture()
        self.assertLessEqual(len(tracker.history), repair.HISTORY_LIMIT)
        self.assertLessEqual(len(tracker.events), repair.EVENT_LIMIT)
        self.assertIn((100,10),tracker.history)
        self.assertGreater(tracker.history_evictions,0)


class ActualCPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v4.enable_subreaper()

    def spawn(self, program):
        p=subprocess.Popen([sys.executable, "-B", "-s", "-c", program],
            start_new_session=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        tracker=repair.repaired_owned_group_class(v4)(p.pid)
        self.addCleanup(self.finish, p, tracker)
        return p, tracker

    @staticmethod
    def line(p, timeout=3):
        ready,_,_=select.select([p.stdout],[],[],timeout)
        if not ready:raise AssertionError("CPU fixture did not become ready")
        return p.stdout.readline().strip()

    @staticmethod
    def finish(p, tracker):
        tracker.stop_and_wait(p)
        for stream in (p.stdin,p.stdout,p.stderr):
            stream.close()
        if v4.group_pids(p.pid):
            raise AssertionError("Owned CPU fixture group survived actual cleanup")

    def test_real_shared_pages_private_growth_and_RSS_only_fallback(self):
        p,tracker=self.spawn(MEMORY_PROGRAM)
        child=json.loads(self.line(p))["child"]
        before=tracker.measure()
        self.assertTrue(tracker.latest["complete"])
        self.assertIn((child,v4.identity(child)["start_ticks"]),tracker.history)
        self.assertGreater(tracker.latest["summed_RSS_bytes"]-before,12*2**20)
        p.stdin.write("ALLOCATE\n");p.stdin.flush()
        self.assertEqual(self.line(p),"ALLOCATED")
        after=tracker.measure()
        self.assertGreater(after-before,7*2**20)
        self.assertLess(after-before,12*2**20)
        read=v4._read
        def deny(path):
            if str(path).endswith("smaps_rollup"):raise PermissionError("fixture")
            return read(path)
        with patch.object(v4,"_read",side_effect=deny):
            self.assertEqual(tracker.measure(),0)
        self.assertFalse(tracker.latest["complete"])
        self.assertEqual(tracker.latest["headroom_credit_bytes"],0)
        self.assertGreater(tracker.latest["summed_RSS_bytes"],after)
        p.stdin.write("STOP\n");p.stdin.flush()
        self.assertEqual(p.wait(timeout=3),0)

    def test_real_continuous_child_churn_returns_with_bounded_capture_count(self):
        p,tracker=self.spawn(CHURN_PROGRAM)
        self.assertEqual(self.line(p),"READY")
        durations=[]
        for _ in range(30):
            started=time.monotonic()
            with patch.object(tracker,"capture",wraps=tracker.capture) as capture:
                tracker.measure()
            durations.append(time.monotonic()-started)
            self.assertEqual(capture.call_count,2)
            self.assertEqual(tracker.latest["sampling_policy"],
                             "one_finite_initial_cohort_one_end_snapshot")
            if not tracker.latest["complete"]:
                self.assertEqual(tracker.latest["headroom_credit_bytes"],0)
        self.assertTrue(p.poll() is None)
        self.assertLess(max(durations),3)
        self.assertGreater(tracker.generation_admissions,3)
        print(json.dumps(dict(fixture="real_finite_churn",samples=30,
              max_sample_seconds=max(durations),incomplete=tracker.incomplete_samples,
              generations=tracker.generation_admissions)),flush=True)

    def test_actual_nonzero_Popen_wait_and_adopted_child_wait_preserved(self):
        p,tracker=self.spawn(ORPHAN_PROGRAM)
        self.assertTrue(self.line(p).startswith("CHILD "))
        self.assertEqual(p.wait(timeout=3),7)
        tracker.stop_and_wait(p)
        receipt=tracker.receipt()
        self.assertEqual(receipt["direct_Popen_wait"]["actual_returncode"],7)
        self.assertEqual([entry["actual_returncode"]
                          for entry in receipt["actual_adopted_child_waits"]],[9])
        self.assertGreater(receipt["pidfd_signals"],0)
        self.assertFalse(v4.group_pids(p.pid))
        self.assertEqual(receipt["cleanup_errors"],[])


if __name__=="__main__":
    unittest.main(verbosity=2)

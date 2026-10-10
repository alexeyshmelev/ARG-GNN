"""CPU/fake-only pool tests. No real driver query, CUDA import or science inputs."""
from __future__ import annotations
import contextlib
import ast
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from arggnn import gpu_pool_v1 as pool

AUDIT = PROJECT / "research/reviews/gpu_pool_v1_source_audit"
GUARD = PROJECT / "src/arggnn/native_argformer_exec.py"
FAKE_MODULE = "arggnn.pool_fake_worker"
UUIDS = {i: f"GPU-00000000-0000-0000-0000-{i:012d}" for i in (2, 3, 4, 5)}

FAKE_WORKER = r'''
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--output",required=True)
p.add_argument("--gate",required=True)
p.add_argument("--guard",required=True)
p.add_argument("--seed",type=int,required=True)
p.add_argument("--model",required=True)
p.add_argument("--mode",default="normal")
p.add_argument("--spam",type=int,default=0)
a=p.parse_args()
s=importlib.util.spec_from_file_location("contained_guard",a.guard)
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
g.ROOT=Path.cwd()
lease=g.validate_lease()
output=Path(a.output);output.mkdir()
def write(name,value):
 (output/name).write_text(json.dumps(value,sort_keys=True))
write("observed.json",{"pid":os.getpid(),"seed":a.seed,"model":a.model,
 "logical":os.environ["ARGGNN_LOGICAL_JOB_NAME"],
 "pool_job":os.environ["ARGGNN_JOB_NAME"],"lease":lease})
if a.mode in ("fork","escaped","fork-grandchild"):
 pid=os.fork()
 if pid:
  write("child.json",{"pid":pid});os._exit(0)
 if a.mode=="escaped":
  os.setsid()
 write("child-ready.json",{"pid":os.getpid(),"sid":os.getsid(0)})
 if a.mode=="fork-grandchild":
  while not Path(str(a.gate)+".advance").exists():
   time.sleep(.01)
  grandchild=os.fork()
  if grandchild:
   write("grandchild.json",{"pid":grandchild});os._exit(0)
  write("grandchild-ready.json",{"pid":os.getpid()})
if a.spam:
 sys.stdout.write("x"*a.spam);sys.stdout.flush()
while not Path(a.gate).exists():
 time.sleep(.01)
write("final.json",{"offset":os.lseek(lease["lease_fd"],0,os.SEEK_CUR),
 "lease":g.validate_lease(),"pid":os.getpid()})
os._exit(0)
'''

def inventory():
    return [dict(index=i, uuid=UUIDS[i], used_mib=0, total_mib=80*1024,
                 utilization=0, compute=[]) for i in (2, 3, 4, 5)]


def race_acquire(root, index, barrier, release, answers):
    barrier.wait()
    lease = None
    try:
        lease, _, _, _ = pool.acquire_device(Path(root), index, f"race_{index}",
                                             inventory, lambda: {}, 0)
        answers.put((index, True))
        release.wait(10)
    except Exception as error:
        answers.put((index, False, str(error)))
    finally:
        if lease is not None:
            lease.close()


class PoolTests(unittest.TestCase):
    def setUp(self):
        AUDIT.mkdir(parents=True, exist_ok=True)
        test_parent = AUDIT / "t"
        test_parent.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="", dir=test_parent)
        self.root = Path(self.tmp.name)
        self.controllers = []
        self.gates = []
        self.rows = inventory()
        (self.root / "src/arggnn").mkdir(parents=True)
        (self.root / "src/arggnn/__init__.py").write_text("")
        (self.root / "src/arggnn/pool_fake_worker.py").write_text(FAKE_WORKER)
        (self.root / "outputs").mkdir()
        self.snapshot_path = self.root / "snapshot.json"
        self.receipt = dict(schema="biological_candidate_storage_snapshot_v1",
                            status="complete", returncode=0, project_root=str(self.root),
                            command=["du", "-s", "-B1", str(self.root)], used_bytes=1_000_000_000,
                            completed_utc=(dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=1)).isoformat())
        self.save_snapshot()

    def save_snapshot(self):
        self.snapshot_path.write_text(json.dumps(self.receipt))
        self.snapshot_sha = hashlib.sha256(self.snapshot_path.read_bytes()).hexdigest()

    def controller(self, name="p", gpu=2, **options):
        kwargs = dict(root=self.root, pool=name, gpu=gpu, headroom_mib=8192,
                      storage_snapshot=self.snapshot_path,
                      storage_snapshot_sha256=self.snapshot_sha,
                      external_storage_reserve_bytes=50_000_000_000,
                      inventory=lambda: self.rows, available_bytes=lambda: 1_000_000_000_000,
                      python=sys.executable, modules={FAKE_MODULE}, socket_relative="s"+name)
        kwargs.update(options)
        c = pool.Controller(**kwargs).start()
        self.controllers.append(c)
        return c

    def spec(self, name, peak=12288, seed=202, mode="normal", **options):
        gate = self.root / ("g"+name)
        self.gates.append(gate)
        output = self.root / "outputs" / name
        s = dict(job=name, model="candidate_age_d2", seed=seed, module=FAKE_MODULE,
                 args=["--output",str(output),"--gate",str(gate),"--guard",str(GUARD),
                       "--seed",str(seed),"--model","candidate_age_d2","--mode",mode],
                 output=str(output), peak_mib=peak, storage_bytes=8*1024*1024,
                 policy="queue", log_limit_bytes=1024*1024)
        s.update(options)
        return s

    def until(self, predicate, controller=None, timeout=8):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if controller:
                controller.tick()
            if predicate():
                return
            time.sleep(.02)
        self.fail("Timed out waiting for contained test condition" +
                  (": "+json.dumps(controller.status(),sort_keys=True) if controller else ""))

    def observed(self, job):
        return self.root / "outputs" / job / "observed.json"

    def tearDown(self):
        for gate in self.gates:
            gate.touch(exist_ok=True)
        for row in self.rows:
            row["compute"]=[]
            row["used_mib"]=0
        for c in self.controllers:
            if c.listener is None:
                continue
            c.shutdown()
            deadline=time.monotonic()+8
            while c.lifecycle!="closed" and time.monotonic()<deadline:
                c.tick(launch_queued=False)
                time.sleep(.02)
            if c.lifecycle!="closed":
                c.shutdown(terminate=True)
                deadline=time.monotonic()+5
                while c.lifecycle!="closed" and time.monotonic()<deadline:
                    c.tick(launch_queued=False)
                    time.sleep(.02)
            if c.lifecycle=="closed" and c.listener is not None:
                c.close()
        for c in self.controllers:
            if not c.children:
                c._release_handles()  # Failed fake-test cleanup only, never production leases.
        self.tmp.cleanup()

    def test_two_device_race_is_atomic(self):
        ctx=mp.get_context("fork")
        barrier,release,answers=ctx.Barrier(4),ctx.Event(),ctx.Queue()
        processes=[ctx.Process(target=race_acquire,args=(str(self.root),i,barrier,release,answers))
                   for i in (2,3,4,5)]
        try:
            for p in processes:p.start()
            records=[answers.get(timeout=10) for _ in processes]
            self.assertEqual(sum(r[1] for r in records),2)
            self.assertTrue(all("Two distinct" in r[2] for r in records if not r[1]))
        finally:
            release.set()
            for p in processes:p.join(10)
            self.assertTrue(all(p.exitcode==0 for p in processes))
            answers.close()

    def test_existing_exclusive_lease_counts_as_one_device(self):
        with pool.lock_file(self.root,"gpu_3.lock") as exclusive:
            fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
            exclusive.write(json.dumps(dict(launcher_pid=os.getpid(),gpu=3,uuid=UUIDS[3],
                                           job="exclusive",started_unix=time.time(),storage_bytes=0)))
            exclusive.flush()
            c=self.controller()
            with self.assertRaisesRegex(RuntimeError,"Two distinct"):
                pool.acquire_device(self.root,4,"third",lambda:self.rows,lambda:{},0)
            self.assertEqual(c.gpu,2)
            self.assertEqual(json.loads((self.root/".runtime/locks/gpu_3.lock").read_text())["job"],
                             "exclusive")

    def test_busy_and_excluded_devices_rejected(self):
        self.rows[0]["compute"]=[dict(pid=999999,name="foreign",used_mib=1)]
        self.rows[0]["used_mib"]=1
        with self.assertRaisesRegex(RuntimeError,"not free"):
            self.controller("busy")
        for index in (0,1,6):
            with self.assertRaisesRegex(ValueError,"Eligible"):
                pool.acquire_device(self.root,index,"excluded",lambda:self.rows,lambda:{},0)

    def test_actual_shared_open_description_and_unchanged_guard(self):
        c=self.controller()
        original=(self.root/".runtime/locks/gpu_2.lock").read_bytes()
        a=c.submit(self.spec("a",seed=202))
        b=c.submit(self.spec("b",seed=303))
        self.until(lambda:self.observed("a").exists() and self.observed("b").exists(),c)
        obs=[json.loads(self.observed(n).read_text()) for n in ("a","b")]
        self.assertNotEqual(a["pid"],b["pid"])
        self.assertEqual([v["logical"] for v in obs],["a","b"])
        self.assertEqual([v["seed"] for v in obs],[202,303])
        self.assertTrue(all(v["pool_job"]==c.pool_job for v in obs))
        self.assertTrue(all(v["lease"]["lease_fd"]==c.lease.fileno() for v in obs))
        self.assertTrue(all(v["lease"]["lease_inode"]==os.fstat(c.lease.fileno()).st_ino for v in obs))
        self.assertTrue(all(v["lease"]["lease_metadata_sha256"]==c.lease_sha for v in obs))
        # An offset written by the controller must be visible to BOTH independent
        # children. Same fd number/inode alone would not prove shared OFD identity.
        os.lseek(c.lease.fileno(),7001,os.SEEK_SET)
        for gate in self.gates:gate.touch()
        self.until(lambda:all(c.jobs[n]["state"]=="succeeded" for n in ("a","b")),c)
        for n in ("a","b"):
            final=json.loads((self.root/"outputs"/n/"final.json").read_text())
            self.assertEqual(final["offset"],7001)
            self.assertEqual(final["lease"]["lease_metadata_sha256"],c.lease_sha)
        self.assertEqual((self.root/".runtime/locks/gpu_2.lock").read_bytes(),original)
        self.assertEqual([c.jobs[n]["requested"]["seed"] for n in ("a","b")],[202,303])
        self.assertFalse(any((self.root/".runtime/gpup/p/jobs"/n).is_dir() for n in ("a","b")))

    def test_pending_cuda_memory_reservations_precede_observed_allocations(self):
        c=self.controller()
        for name,peak,seed in (("a",12288,202),("b",12288,303),("mix",40960,101)):
            c.submit(self.spec(name,peak,seed))
        self.assertEqual(c._memory()["active_peak_mib"],65536)
        fourth=c.submit(self.spec("fourth",12288))
        self.assertEqual(fourth["state"],"queued")
        self.assertIn("memory_headroom",fourth["admission"]["blocked_by"])
        self.assertIsNone(c.jobs["fourth"]["pid"])
        self.assertFalse((self.root/"outputs/fourth").exists())
        self.gates[0].touch()
        self.until(lambda:c.jobs["a"]["state"]=="succeeded" and c.jobs["fourth"]["state"]=="running",c)

    def test_foreign_compute_and_peak_overrun_block_new_work(self):
        c=self.controller()
        first=c.submit(self.spec("a"))
        self.until(lambda:self.observed("a").exists(),c)
        self.rows[0].update(used_mib=1,compute=[dict(pid=999999,name="foreign",used_mib=1)])
        with self.assertRaisesRegex(ValueError,"foreign_compute"):
            c.submit(self.spec("reject",policy="reject"))
        queued=c.submit(self.spec("queue"))
        self.assertEqual(queued["state"],"queued")
        self.rows[0].update(used_mib=13312,compute=[dict(pid=first["pid"],name="title",used_mib=13312)])
        c.tick()
        self.assertIn("worker_peak_overrun",c.jobs["queue"]["admission"]["blocked_by"])
        self.assertIsNone(c.jobs["queue"]["pid"])
        self.rows[0].update(used_mib=12288,compute=[dict(pid=first["pid"],name="title",used_mib=12288)])
        self.until(lambda:c.jobs["queue"]["state"]=="running",c)
        self.assertEqual(c._memory()["unattributed_mib"],0)

    def test_observed_unattributed_memory_and_impossible_peak_reject(self):
        c=self.controller()
        c.submit(self.spec("a",32768))
        self.rows[0]["used_mib"]=10240
        queued=c.submit(self.spec("b",40960))
        self.assertEqual(queued["state"],"queued")
        self.assertEqual(queued["admission"]["unattributed_mib"],10240)
        with self.assertRaisesRegex(ValueError,"cannot fit"):
            c.submit(self.spec("impossible",81920))

    def test_descendant_reservation_survives_leader_exit(self):
        c=self.controller()
        c.submit(self.spec("fork",65536,mode="fork"))
        ready=self.root/"outputs/fork/child-ready.json"
        self.until(lambda:ready.exists() and c.jobs["fork"]["exit_code"]==0,c)
        self.assertEqual(c.jobs["fork"]["state"],"draining")
        self.assertEqual(c._memory()["active_peak_mib"],65536)
        queued=c.submit(self.spec("waiting",16384))
        self.assertEqual(queued["state"],"queued")
        with pool.lock_file(self.root,"gpu_2.lock") as h:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.gates[0].touch()
        self.until(lambda:c.jobs["fork"]["state"]=="succeeded" and c.jobs["waiting"]["state"]=="running",c)

    def test_stale_snapshot_is_refreshed_after_leader_exit_before_reservation_release(self):
        c=self.controller()
        c.submit(self.spec("fork",65536,mode="fork"))
        ready=self.root/"outputs/fork/child-ready.json"
        self.until(lambda:ready.exists())
        child_pid=json.loads(ready.read_text())["pid"]
        leader=c.children["fork"]
        self.until(lambda:pool.proc_identity(leader.pid)["state"]=="Z")
        self.assertIsNone(leader.returncode)
        real_reader=pool.ProcReader()
        observations=[]
        def stale_once(lease,controller_pid,known):
            table=real_reader(lease,controller_pid,known)
            observations.append(1)
            if len(observations)==1:
                table.pop(child_pid,None)
                if leader.pid in table:table[leader.pid]["state"]="Z"
            return table
        c.proc_reader=stale_once
        c.tick(launch_queued=False)
        self.assertGreaterEqual(len(observations),2)
        self.assertEqual(c.jobs["fork"]["state"],"draining")
        self.assertEqual(c._memory()["active_peak_mib"],65536)
        pending=c.submit(self.spec("pending",16384))
        self.assertEqual(pending["state"],"queued")
        self.gates[0].touch()
        self.until(lambda:c.jobs["fork"]["state"]=="succeeded"
                   and c.jobs["pending"]["state"]=="running",c)

    def test_post_identity_kernel_child_recheck_captures_new_grandchild(self):
        c=self.controller()
        c.submit(self.spec("grand",65536,mode="fork-grandchild"))
        output=self.root/"outputs/grand"
        self.until(lambda:(output/"child-ready.json").exists())
        child_pid=json.loads((output/"child-ready.json").read_text())["pid"]
        leader=c.children["grand"]
        self.until(lambda:pool.proc_identity(leader.pid)["state"]=="Z")
        advance=Path(str(self.gates[0])+".advance")
        self.gates.append(advance)
        real_identity=pool.proc_identity
        reads=[]
        def identity_with_fork_at_final_merge(pid):
            if pid==child_pid:
                reads.append(pid)
                if len(reads)==3:
                    advance.touch()
                    deadline=time.monotonic()+5
                    while time.monotonic()<deadline:
                        if ((output/"grandchild-ready.json").exists()
                                and real_identity(child_pid)["state"]=="Z"):
                            break
                        time.sleep(.01)
                    else:
                        self.fail("Contained child-to-grandchild transition timed out")
            return real_identity(pid)
        with mock.patch.object(pool,"proc_identity",side_effect=identity_with_fork_at_final_merge):
            c.tick(launch_queued=False)
        self.assertGreaterEqual(len(reads),3)
        self.assertTrue((output/"grandchild-ready.json").exists())
        self.assertEqual(c.jobs["grand"]["state"],"draining")
        self.assertEqual(c._memory()["active_peak_mib"],65536)
        pending=c.submit(self.spec("pending",16384))
        self.assertEqual(pending["state"],"queued")
        self.gates[0].touch()
        self.until(lambda:c.jobs["grand"]["state"]=="succeeded"
                   and c.jobs["pending"]["state"]=="running",c)

    def test_reused_leader_pid_does_not_adopt_unrelated_same_uid_session(self):
        c=self.controller()
        c.submit(self.spec("escaped",65536,mode="escaped"))
        output=self.root/"outputs/escaped"
        self.until(lambda:(output/"child-ready.json").exists()
                   and c.jobs["escaped"]["exit_code"]==0,c)
        old_pid=c.jobs["escaped"]["pid"]
        old_start=c.jobs["escaped"]["leader_start_ticks"]
        self.assertIsNotNone(old_start)
        real_reader=pool.ProcReader()
        def reused_leader(lease,controller_pid,known):
            table=real_reader(lease,controller_pid,known)
            table[old_pid]=dict(pid=old_pid,uid=os.getuid(),start_ticks=old_start+10000,
                               state="S",ppid=1,pgid=old_pid,sid=old_pid,
                               lease_fd_present=False,controller_descendant=False)
            return table
        c.proc_reader=reused_leader
        c._process_scan()
        self.assertNotIn(old_pid,c.pid_jobs)
        self.rows[0].update(used_mib=1,compute=[
            dict(pid=old_pid,name="AI Auto Research Pipeline Test",used_mib=1)])
        self.assertEqual(c._memory()["foreign_compute_pids"],[old_pid])
        self.rows[0].update(used_mib=0,compute=[])
        c.proc_reader=real_reader
        self.gates[0].touch()
        self.until(lambda:c.jobs["escaped"]["state"]=="succeeded",c)

    def test_escaped_inherited_descriptor_blocks_drain_and_admission(self):
        c=self.controller()
        c.submit(self.spec("escaped",65536,mode="escaped"))
        ready=self.root/"outputs/escaped/child-ready.json"
        self.until(lambda:ready.exists() and c.jobs["escaped"]["exit_code"]==0,c)
        self.assertEqual(c.jobs["escaped"]["state"],"draining")
        pending=c.submit(self.spec("later",16384))
        self.assertEqual(pending["state"],"queued")
        c.shutdown()
        self.assertEqual(c.lifecycle,"draining")
        self.assertEqual(c.jobs["later"]["state"],"cancelled")
        self.gates[0].touch()
        self.until(lambda:c.lifecycle=="closed",c)

    def test_unknown_same_inode_holder_is_never_signalled(self):
        c=self.controller()
        foreign=subprocess.Popen([sys.executable,"-c",
            "import sys,time;f=open(sys.argv[1]);time.sleep(30)",
            str(self.root/".runtime/locks/gpu_2.lock")],
            start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            self.until(lambda:foreign.pid in c.ambiguous,c)
            self.assertEqual(c.submit(self.spec("queue"))["state"],"queued")
            c.shutdown(terminate=True)
            self.assertIsNone(foreign.poll())
            self.assertEqual(c.lifecycle,"draining")
        finally:
            # This subprocess belongs to the CPU test, not the pool.
            foreign.terminate();foreign.wait(timeout=5)
        self.until(lambda:c.lifecycle=="closed",c)

    def test_controller_loss_preserves_identical_worker_lease(self):
        specs=[self.spec("a"),self.spec("b")]
        script=self.root/"crash.py"
        script.write_text("\n".join([
            "import json,os,sys,time",
            f"sys.path.insert(0,{str(PROJECT/'src')!r})",
            "from arggnn.gpu_pool_v1 import Controller",
            f"root={str(self.root)!r}",
            "rows="+repr(self.rows),
            "c=Controller(root=root,pool='crash',gpu=2,headroom_mib=8192,"
            f"storage_snapshot={str(self.snapshot_path)!r},storage_snapshot_sha256={self.snapshot_sha!r},"
            "external_storage_reserve_bytes=50000000000,inventory=lambda:rows,"
            "available_bytes=lambda:1000000000000,python=sys.executable,"
            f"modules={{{FAKE_MODULE!r}}},socket_relative='sc').start()",
            "specs="+repr(specs),
            "for s in specs:c.submit(s)",
            "from pathlib import Path",
            "deadline=time.monotonic()+8",
            "while not all((Path(s['output'])/'observed.json').exists() for s in specs):",
            " c.tick();time.sleep(.02)",
            " if time.monotonic()>deadline:raise RuntimeError('CPU worker startup timeout')",
            "os._exit(0)",
        ]))
        with (self.root/"crash.log").open("wb") as log:
            parent=subprocess.Popen([sys.executable,str(script)],stdout=log,stderr=log)
            self.assertEqual(parent.wait(timeout=12),0,(self.root/"crash.log").read_text())
        original=(self.root/".runtime/locks/gpu_2.lock").read_bytes()
        with pool.lock_file(self.root,"gpu_2.lock") as h:
            with self.assertRaises(BlockingIOError):fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A new pool must not adopt/rewrite this held lease.
        with self.assertRaisesRegex(RuntimeError,"not free"):
            self.controller("replacement")
        self.assertEqual((self.root/".runtime/locks/gpu_2.lock").read_bytes(),original)
        for gate in self.gates:gate.touch()
        def free():
            with pool.lock_file(self.root,"gpu_2.lock") as h:
                try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return False
                return True
        self.until(free)
        self.assertEqual((self.root/".runtime/locks/gpu_2.lock").read_bytes(),original)

    def test_bounded_logs_drain_without_unbounded_writes(self):
        c=self.controller()
        s=self.spec("spam",log_limit_bytes=1024)
        s["args"]+=["--spam","200000"]
        c.submit(s)
        self.until(lambda:c.jobs["spam"]["log_dropped_bytes"]>=198976,c)
        self.assertEqual(Path(c.jobs["spam"]["log_path"]).stat().st_size,1024)
        self.assertEqual(c.jobs["spam"]["log_bytes"],1024)

    def test_snapshot_and_persistent_storage_admission(self):
        self.receipt["status"]="partial";self.save_snapshot()
        with self.assertRaisesRegex(ValueError,"Complete"):self.controller("bad")
        self.receipt["status"]="complete";self.save_snapshot()
        with self.assertRaisesRegex(ValueError,"SHA256"):
            self.controller("badsha",storage_snapshot_sha256="0"*64)
        c=self.controller()
        c.submit(self.spec("a",65536))
        c.submit(self.spec("queued",16384))
        ledger_path=self.root/".runtime/gpup/storage_reservations.json"
        before=json.loads(ledger_path.read_text())
        c.shutdown()
        self.assertEqual(c.jobs["queued"]["state"],"cancelled")
        self.assertEqual(json.loads(ledger_path.read_text())["jobs"],before["jobs"])
        self.gates[0].touch()
        self.until(lambda:c.lifecycle=="closed",c)
        c.close()
        c2=self.controller("p2")
        with self.assertRaisesRegex(ValueError,"already reserved"):
            c2.submit(self.spec("collision",output=str(self.root/"outputs/queued")))
        self.assertEqual(c2.storage_account["persistent_reservations"],2)

    def test_storage_budget_and_shared_ledger_binding_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"ceiling"):
            self.controller("budget",external_storage_reserve_bytes=449_000_000_000)
        c=self.controller()
        with self.assertRaisesRegex(ValueError,"ceiling"):
            c.submit(self.spec("huge",storage_bytes=449_000_000_000))
        self.assertFalse(c.jobs)
        self.receipt["used_bytes"]+=1;self.save_snapshot()
        with self.assertRaisesRegex(ValueError,"binding differs"):
            self.controller("rebase",gpu=3)

    def test_argv_provenance_fresh_output_and_protected_environment(self):
        c=self.controller()
        s=self.spec("a")
        bad=dict(s,environment={"ARGGNN_JOB_NAME":"forged"})
        with self.assertRaisesRegex(ValueError,"supported"):c.submit(bad)
        self.assertFalse((self.root/"outputs/a").exists())
        admitted=c.submit(s)
        self.assertEqual(admitted["argv"],[sys.executable,"-m",FAKE_MODULE,*s["args"]])
        self.assertEqual(admitted["requested"]["model"],s["model"])
        with self.assertRaisesRegex(ValueError,"Duplicate"):c.submit(s)
        (self.root/"outputs/preexisting").mkdir()
        with self.assertRaisesRegex(ValueError,"Fresh"):c.submit(self.spec("preexisting"))

    def test_socket_permissions_peer_protocol_and_slow_client_supervision(self):
        c=self.controller()
        server=threading.Thread(target=c.run,kwargs={"poll_seconds":.05})
        server.start()
        slow=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            self.assertEqual(c.socket_path.stat().st_mode&0o777,0o600)
            with self.assertRaisesRegex(RuntimeError,"protocol"):
                pool.request(c.socket_path,dict(v=1,op="status",extra=True))
            s=self.spec("a")
            admitted=pool.request(c.socket_path,dict(v=1,op="submit",spec=s))
            self.assertEqual(admitted["state"],"running")
            slow.connect(str(c.socket_path));slow.sendall(b"{")
            self.until(lambda:self.observed("a").exists())
            self.gates[0].touch()
            self.until(lambda:pool.request(c.socket_path,dict(v=1,op="status"))["jobs"][0]["state"]=="succeeded")
            self.assertTrue(server.is_alive())
            slow.close()
            status=pool.request(c.socket_path,dict(v=1,op="shutdown",terminate=False))
            self.assertEqual(status["lifecycle"],"closed")
            server.join(5)
            self.assertFalse(server.is_alive())
            disk=pool.disk_status(self.root,c.pool)
            self.assertFalse(disk["snapshot_stale"])
            self.assertFalse(c.socket_path.exists())
        finally:
            slow.close()
            if server.is_alive():
                c.shutdown()
                server.join(5)

    def test_inaccessible_known_descendant_fails_closed_without_release(self):
        c=self.controller()
        first=c.submit(self.spec("known"))
        self.until(lambda:self.observed("known").exists(),c)
        real_scandir=pool.os.scandir
        forbidden=f"/proc/{first['pid']}/fd"
        def limited(path):
            if str(path)==forbidden:
                raise PermissionError("contained simulated descendant restriction")
            return real_scandir(path)
        with mock.patch.object(pool.os,"scandir",side_effect=limited):
            c.tick()
            self.assertIn("own descendant",c.safety_error)
            self.assertEqual(c.jobs["known"]["state"],"running")
            with self.assertRaisesRegex(RuntimeError,"own descendant"):
                c.submit(self.spec("blocked"))
        self.assertEqual(c._memory()["active_peak_mib"],12288)
        self.gates[0].touch()
        self.until(lambda:c.jobs["known"]["state"]=="succeeded",c)

    def test_terminate_signals_only_proven_worker_pid(self):
        c=self.controller()
        c.submit(self.spec("own"))
        self.until(lambda:self.observed("own").exists(),c)
        unrelated=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],
                                   start_new_session=True,
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            c.shutdown(terminate=True)
            self.until(lambda:c.jobs["own"]["exit_code"]==-signal.SIGTERM,c)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate();unrelated.wait(timeout=5)
        self.until(lambda:c.lifecycle=="closed",c)

    def test_real_inventory_parser_is_tested_with_fake_subprocess_outputs_only(self):
        gpu_csv="\n".join(f"{i}, {UUIDS[i]}, 0, 81920, 0" for i in (2,3,4,5))+"\n"
        app_csv=f"{UUIDS[2]}, 123, AI Auto Research Pipeline Test, 1024\n"
        with mock.patch.object(pool.subprocess,"run",side_effect=[
                subprocess.CompletedProcess([],0,gpu_csv,""),
                subprocess.CompletedProcess([],0,app_csv,"")]) as run:
            rows=pool.checked_inventory(pool.real_inventory)
        self.assertEqual(rows[2]["compute"][0]["pid"],123)
        self.assertEqual(run.call_count,2)
        self.assertTrue(all(call.kwargs["shell"] is False for call in run.call_args_list))

    def test_gst_source_bootstrap_matches_shared_guard_contract(self):
        self.assertIn("arggnn.biological_gst_control_v1",pool.MODULES)
        gst=PROJECT/"src/arggnn/biological_gst_control_v1.py"
        runner=PROJECT/"scripts/train_biological_gst_control_v1.py"
        self.assertEqual(hashlib.sha256(gst.read_bytes()).hexdigest(),
                         "66788b41cabcb852c739505200a641bdc93fea3c9fbb642c291d52cefc77350f")
        self.assertEqual(hashlib.sha256(runner.read_bytes()).hexdigest(),
                         "e8496eb6a92cc2f9edb5f107a89d23cdbc945a82ae0e00bf2a4e0f00ed9ab72b")
        tree=ast.parse(gst.read_text())
        top_imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any("torch" in ast.dump(n) or "numpy" in ast.dump(n) for n in top_imports))
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train")
        inherited=next(n for n in train.body if isinstance(n,ast.ImportFrom)
                       and n.module=="biological_candidate")
        self.assertTrue({"configure_fit","check_fit_lease"}.issubset({a.name for a in inherited.names}))
        fit_line=min(n.lineno for n in ast.walk(train) if isinstance(n,ast.Call)
                     and isinstance(n.func,ast.Name) and n.func.id=="configure_fit")
        cuda_lines=[n.lineno for n in ast.walk(train) if isinstance(n,ast.Attribute)
                    and n.attr=="cuda"]
        model_lines=[n.lineno for n in ast.walk(train) if isinstance(n,ast.Call)
                     and isinstance(n.func,ast.Name) and n.func.id=="model_factory"]
        self.assertTrue(all(n>fit_line for n in cuda_lines+model_lines))
        cli=ast.parse(runner.read_text())
        main=next(n for n in cli.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        title_line=min(n.lineno for n in ast.walk(main) if isinstance(n,ast.Call)
                       and isinstance(n.func,ast.Name) and n.func.id=="apply_and_validate_title")
        scientific_line=min(n.lineno for n in ast.walk(main) if isinstance(n,ast.ImportFrom)
                            and n.module=="arggnn.biological_gst_control_v1")
        self.assertLess(title_line,scientific_line)

    def test_mechanism_source_bootstrap_matches_shared_guard_contract(self):
        self.assertIn("arggnn.biological_mechanism_controls_v1",pool.MODULES)
        module=PROJECT/"src/arggnn/biological_mechanism_controls_v1.py"
        runner=PROJECT/"scripts/train_biological_mechanism_controls_v1.py"
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         "28f3e7d3f28a48eeee0944b2e4ad55e3ea6ebb314d8b0342bd785facb08dcbeb")
        self.assertEqual(hashlib.sha256(runner.read_bytes()).hexdigest(),
                         "f19bdead17a135a42ad70513f1522cef2a9fd58dc159d04f189adbe08de67dec")
        tree=ast.parse(module.read_text())
        guard=next(n for n in tree.body if isinstance(n,ast.If)
                   and "__name__" in ast.unparse(n.test) and "--train" in ast.unparse(n.test))
        title_line=min(n.lineno for n in ast.walk(guard) if isinstance(n,ast.Call)
                       and isinstance(n.func,ast.Name) and n.func.id=="apply_and_validate_title")
        torch_import=min(n.lineno for n in tree.body if isinstance(n,ast.Import)
                         and any(a.name=="torch" for a in n.names))
        self.assertLess(title_line,torch_import)
        inherited=next(n for n in tree.body if isinstance(n,ast.ImportFrom)
                       and n.module=="biological_candidate")
        self.assertTrue({"configure_fit","check_fit_lease"}.issubset({a.name for a in inherited.names}))
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train")
        fit_line=min(n.lineno for n in ast.walk(train) if isinstance(n,ast.Call)
                     and isinstance(n.func,ast.Name) and n.func.id=="configure_fit")
        cuda_lines=[n.lineno for n in ast.walk(train) if isinstance(n,ast.Attribute)
                    and n.attr=="cuda"]
        self.assertTrue(all(n>fit_line for n in cuda_lines))
        cli=ast.parse(runner.read_text())
        main=next(n for n in cli.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        title_line=min(n.lineno for n in ast.walk(main) if isinstance(n,ast.Call)
                       and isinstance(n.func,ast.Name) and n.func.id=="apply_and_validate_title")
        scientific_line=min(n.lineno for n in ast.walk(main) if isinstance(n,ast.ImportFrom)
                            and n.module=="arggnn")
        self.assertLess(title_line,scientific_line)

    def test_owned_sources_have_no_cuda_import_or_shell_launch(self):
        source=(PROJECT/"src/arggnn/gpu_pool_v1.py").read_text()
        self.assertNotIn("import torch",source)
        self.assertNotIn("import numpy",source)
        self.assertNotIn('["du"',source.split("def snapshot_record")[0])
        self.assertNotIn("shell=True",source)
        self.assertNotIn("LOCK_UN",source.split("def _release_handles")[1].split("def close")[0].replace(
            "# Never LOCK_UN: inherited workers can still hold the OFD.",""))
        self.assertEqual(hashlib.sha256(GUARD.read_bytes()).hexdigest(),
                         "84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0")


if __name__=="__main__":
    unittest.main(verbosity=2)

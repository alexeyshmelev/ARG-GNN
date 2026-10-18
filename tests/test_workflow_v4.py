"""Actual tiny CPU forks/waits; policy/GPU/scientific factory calls mocked."""
from pathlib import Path
import copy
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from arggnn.benchmarking import workflow_v4 as v4

MEMORY_PROGRAM="""
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


class WorkflowV4Test(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=v4.c.ROOT/"tests")
        self.directory=Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        # Bindings are permitted only in a new v4 process; restore this test's
        # imported base module so unrelated legacy tests keep their namespace.
        names=("MODULE","current_policy","_own_group_rss","execute","child","run",
               "external_result","external_resources","storage_projection","available_ram_bytes","ProgressReporter","job_spec","scope_bytes")
        saved={name:getattr(v4.base,name) for name in names}
        self.addCleanup(lambda:[setattr(v4.base,name,value) for name,value in saved.items()])

    def test_policy_projects_only_exact_continuation_and_preserves_canonical_parent(self):
        parent=dict(path="v3",sha256="3"*64);continuation=dict(path="v4",sha256="4"*64)
        original=dict(approved_workflow_ref=parent,approved_continuation_ref=continuation,
                      workflow_drain_ref=parent,continuation_drain_ref=continuation)
        with patch.object(v4,"_BASE_POLICY",return_value=original):
            observed=v4.current_policy()
        self.assertEqual(observed["approved_workflow_ref"],continuation)
        self.assertEqual(observed["canonical_parent_workflow_ref"],parent)
        self.assertEqual(observed["workflow_drain_ref"],continuation)
        self.assertEqual(original["approved_workflow_ref"],parent)
        with patch.object(v4,"_BASE_POLICY",return_value=dict(approved_workflow_ref=parent)):
            with self.assertRaisesRegex(RuntimeError,"approved_continuation_ref"):v4.current_policy()

    def test_spawn_namespace_and_all_inherited_boundary_bindings(self):
        v4.install_bindings()
        self.assertEqual(v4.base.MODULE,"arggnn.benchmarking.workflow_v4")
        self.assertIs(v4.base.current_policy,v4.current_policy)
        self.assertIs(v4.base.execute,v4.execute)
        self.assertIs(v4.base.child,v4.child)
        self.assertIs(v4.base.run,v4.run)
        self.assertIs(v4.base._own_group_rss,v4.group_pss)
        self.assertIs(v4.base.external_resources,v4.external_resources)
        self.assertIs(v4.base.storage_projection,v4.storage_projection)

    def test_actual_fork_PSS_private_growth_RSS_fallback_and_identity_rejection(self):
        v4.enable_subreaper()
        process=subprocess.Popen([sys.executable,"-B","-s","-c",MEMORY_PROGRAM],
            start_new_session=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
        tracker=v4.OwnedGroup(process.pid)
        try:
            fork=json.loads(process.stdout.readline());before=tracker.measure()
            self.assertIn(fork["child"],tracker.history)
            self.assertGreater(tracker.latest["summed_RSS_bytes"]-before,12*2**20)
            process.stdin.write("ALLOCATE\n");process.stdin.flush()
            self.assertEqual(process.stdout.readline().strip(),"ALLOCATED")
            after=tracker.measure();self.assertGreater(after-before,7*2**20)
            self.assertLess(after-before,12*2**20)
            reader=v4._read
            def deny_pss(path):
                if str(path).endswith("smaps_rollup"):raise PermissionError("synthetic denial")
                return reader(path)
            with patch.object(v4,"_read",side_effect=deny_pss):
                fallback=tracker.measure()
            self.assertTrue(tracker.latest["conservative_RSS_fallback"])
            self.assertEqual(fallback,tracker.latest["summed_RSS_bytes"])
            self.assertGreaterEqual(fallback,after)
            remembered=tracker.history[process.pid]["start_ticks"]
            tracker.history[process.pid]=dict(tracker.history[process.pid],start_ticks=remembered+1)
            with self.assertRaisesRegex(RuntimeError,"reused"):tracker.measure()
            tracker.history[process.pid]["start_ticks"]=remembered
            process.stdin.write("STOP\n");process.stdin.flush();self.assertEqual(process.wait(timeout=5),0)
        finally:
            tracker.stop_and_wait(process)
            self.assertFalse(v4.group_pids(process.pid))
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()

    def _actual_execute(self,root_code,child_code):
        output=self.directory/("execute_"+str(root_code))
        task=dict(id="tiny_cpu",resource="cpu",output_directory=str(output),
                  max_seconds=10,max_output_bytes=2**20,max_rss_bytes=256*2**20)
        plan_ref=dict(path="synthetic_plan",sha256="a"*64)
        job=dict(task=task,call=dict(python=str(Path(sys.executable))),plan_ref=plan_ref)
        job_path=self.directory/("job_"+str(root_code)+".json");job_path.write_text(json.dumps(job))
        job_ref=v4.c.file_ref(job_path)
        code="""
import json,os,time
from pathlib import Path
from arggnn.benchmarking.workflow_v4 import await_startup
await_startup()
kid=os.fork()
if kid==0:
 time.sleep(.03);os._exit(CHILD)
time.sleep(.12)
Path(OUTPUT,'result.json').write_text(json.dumps(dict(synthetic=True)))
os._exit(ROOT)
""".replace("CHILD",str(child_code)).replace("ROOT",str(root_code)).replace("OUTPUT",repr(str(output)))
        original_popen=subprocess.Popen
        def actual_toy(argv,**kwargs):
            self.assertEqual(argv[4],v4.MODULE)
            return original_popen([sys.executable,"-B","-s","-c",code],**kwargs)
        with patch.object(v4.base,"job_spec",return_value=job),              patch.object(v4,"current_policy",return_value=dict(approved_workflow_ref=plan_ref)),              patch.object(v4.subprocess,"Popen",side_effect=actual_toy):
            if root_code:
                with self.assertRaises(ValueError):v4.execute(job_ref)
            else:v4.execute(job_ref)
        wait=json.loads((output/"actual_wait.json").read_text())
        immediate=json.loads((output/"supervision_identity.json").read_text())
        self.assertEqual(wait["actual_direct_returncode"],root_code)
        self.assertTrue(wait["actual_direct_wait"])
        self.assertTrue(wait["owned_process_group_absent"])
        supervision=wait["supervision"]
        self.assertTrue(supervision["linux_subreaper_enabled"])
        self.assertEqual(supervision["direct_Popen_wait"]["actual_returncode"],root_code)
        self.assertEqual(supervision["root_identity"],immediate["child_identity"])
        self.assertGreater(supervision["memory_samples"],0)
        self.assertGreater(supervision["peak_accounted_bytes"],0)
        self.assertEqual([row["actual_returncode"] for row in supervision["actual_adopted_child_waits"]],[child_code])
        self.assertTrue(all(row["actual_waitpid"] for row in supervision["actual_adopted_child_waits"]))
        self.assertFalse(v4.group_pids(supervision["root_identity"]["pgid"]))
        return wait

    def test_actual_zero_exit_wait_and_orphan_fork_worker_reaped(self):
        self._actual_execute(0,0)

    def test_actual_nonzero_exit_preserved_and_nonzero_orphan_reaped(self):
        self._actual_execute(7,9)

    def test_startup_receipt_failure_blocks_science_and_actually_waits_child(self):
        output=self.directory/"startup_failure";marker=self.directory/"SCIENCE_MUST_NOT_START"
        task=dict(id="tiny_cpu",resource="cpu",output_directory=str(output),
                  max_seconds=10,max_output_bytes=2**20,max_rss_bytes=256*2**20)
        plan_ref=dict(path="synthetic",sha256="a"*64)
        job=dict(task=task,call=dict(python=str(Path(sys.executable))),plan_ref=plan_ref)
        path=self.directory/"startup_job.json";path.write_text(json.dumps(job));ref=v4.c.file_ref(path)
        original_popen=subprocess.Popen;actual_pid=[]
        program="from arggnn.benchmarking.workflow_v4 import await_startup; from pathlib import Path; await_startup(); Path(%r).write_text('bad')"%str(marker)
        def toy(argv,**kwargs):
            process=original_popen([sys.executable,"-B","-s","-c",program],**kwargs)
            actual_pid.append(process.pid);return process
        saver=v4._BASE_SAVE
        def denied(path,value,**kwargs):
            if Path(path).name=="supervision_identity.json":raise OSError("synthetic identity disk failure")
            return saver(path,value,**kwargs)
        with patch.object(v4.base,"job_spec",return_value=job),              patch.object(v4,"current_policy",return_value=dict(approved_workflow_ref=plan_ref)),              patch.object(v4.subprocess,"Popen",side_effect=toy),patch.object(v4,"_BASE_SAVE",side_effect=denied):
            with self.assertRaisesRegex(OSError,"identity disk failure"):v4.execute(ref)
        self.assertFalse(marker.exists())
        wait=json.loads((output/"actual_wait.json").read_text())
        self.assertTrue(wait["actual_direct_wait"])
        self.assertTrue(wait["owned_process_group_absent"])
        self.assertTrue(wait["scientific_entry_barrier_not_released"])
        self.assertNotEqual(wait["actual_direct_returncode"],0)
        self.assertFalse(v4.group_pids(actual_pid[0]))

    def test_ESRCH_is_only_a_vanished_owned_PID_not_an_unreadable_zero(self):
        tracker=v4.OwnedGroup.__new__(v4.OwnedGroup)
        tracker.root=dict(pid=123,start_ticks=1,pgid=123,session=123)
        tracker.history={123:tracker.root}
        with patch.object(v4,"group_pids",return_value={123}),              patch.object(v4,"identity",side_effect=ProcessLookupError(3,"gone")):
            self.assertEqual(tracker.capture(),{})
        with patch.object(v4,"group_pids",return_value={123}),              patch.object(v4,"identity",side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):tracker.capture()

    def test_scope_actual_writer_remove_race_and_escape_failclosed(self):
        output=self.directory/"scope";output.mkdir();stable=output/"stable.log";stable.write_text("keep")
        stop=threading.Event();failures=[]
        def writer():
            index=0
            try:
                while not stop.is_set():
                    path=output/("sec"+str(index%4)+".log")
                    path.write_text("transient");path.unlink();index+=1
            except BaseException as exc:failures.append(exc)
        thread=threading.Thread(target=writer);thread.start()
        try:
            minimum=max(stable.stat().st_size,stable.stat().st_blocks*512)+4096
            for _ in range(100):self.assertGreaterEqual(v4.scope_bytes(output),minimum)
        finally:stop.set();thread.join(timeout=5)
        self.assertFalse(thread.is_alive());self.assertEqual(failures,[])
        vanished=output/"sec87.log";vanished.write_text("gone")
        original=Path.lstat
        def remove_before_stat(path,*args,**kwargs):
            if path==vanished and path.exists():path.unlink()
            return original(path,*args,**kwargs)
        with patch.object(Path,"lstat",remove_before_stat):
            self.assertGreaterEqual(v4.scope_bytes(output),minimum)
        outside=self.directory/"outside";outside.mkdir();(output/"escape").symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError,"symlink escaped"):v4.scope_bytes(output)
        (output/"escape").unlink()
        def denied(path,*args,**kwargs):
            if path==stable:raise PermissionError("real unreadable entry")
            return original(path,*args,**kwargs)
        with patch.object(Path,"lstat",denied):
            with self.assertRaises(PermissionError):v4.scope_bytes(output)

    def test_headroom_credits_only_minimum_measured_allocated_growth(self):
        state_path=self.directory/"state.json"
        with patch.object(v4,"_RUN_CONTEXT",({},state_path)),              patch.object(v4,"_known_credit",side_effect=[165,150]),              patch.object(v4,"_BASE_AVAILABLE",return_value=1130):
            self.assertEqual(v4.available_ram_bytes(),1280)
        receipt=json.loads((self.directory/"host_headroom_observation.json").read_text())
        self.assertEqual(receipt["known_live_PSS_credit_bytes"],150)
        self.assertTrue(receipt["unknown_groups_receive_zero_credit"])
        self.assertTrue(receipt["full_aggregate_reservations_unchanged"])

    def test_external_fullcaps_and_storage_are_not_relaxed_by_hostcredit(self):
        plan=dict(prior_unknown_rss_bytes=256)
        with patch.object(v4,"_BASE_EXTERNAL_RESOURCES",return_value=(3,640)):
            self.assertEqual(v4.external_resources(plan),(3,896))
        with patch.object(v4,"_BASE_STORAGE",return_value=1000):
            self.assertEqual(v4.storage_projection(plan,{},100),1000)


if __name__=="__main__":unittest.main()

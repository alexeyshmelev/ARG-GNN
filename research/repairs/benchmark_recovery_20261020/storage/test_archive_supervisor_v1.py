"""Pure admission and fake-process lifecycle checks; no production worker launch."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile
import unittest
from unittest.mock import Mock, patch

from research.repairs.benchmark_recovery_20261020.storage import archive_supervisor_v1 as supervisor
from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v1 as recovery

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

class SupervisorTests(unittest.TestCase):
    def test_strict_980GB_boundary(self):
        task=dict(id="fixture",max_output_bytes=1024,max_rss_bytes=1024)
        config=dict(recovery_plan=dict(tasks=[task]),storage_ceiling_bytes=980_000_000_000,
                    host_ram_margin_bytes=64*2**30,max_parallel_rss_bytes=1280*2**30)
        state=dict(tasks={"fixture":dict(status="pending")})
        with patch.object(supervisor,"base_capacity",return_value=(980_000_000_000-1024,0)):
            self.assertFalse(supervisor.capacity(config,state,task)["storage_allowed"])
        with patch.object(supervisor,"base_capacity",return_value=(980_000_000_000-1025,0)):
            self.assertTrue(supervisor.capacity(config,state,task)["storage_allowed"])

    def test_running_archive_is_charged_once_and_partial_failure_stays_charged(self):
        task=dict(id="fixture",max_output_bytes=1024,max_rss_bytes=4096)
        config=dict(recovery_plan=dict(tasks=[task]),storage_ceiling_bytes=980_000_000_000,
                    host_ram_margin_bytes=0,max_parallel_rss_bytes=1280*2**30)
        state=dict(tasks={"fixture":dict(status="running",reservation_bytes=1024)})
        with patch.object(supervisor,"base_capacity",return_value=(1000,500)):
            result=supervisor.capacity(config,state)
            self.assertEqual(result["projected_bytes"],2024)
            self.assertEqual(result["reserved_rss_bytes"],4596)
            state["tasks"]["fixture"].update(status="failed",retained_bytes=200)
            self.assertEqual(supervisor.capacity(config,state)["projected_bytes"],1200)

    def test_direct_exit_race_still_consumes_real_wait(self):
        process=Mock()
        process.poll.return_value=None
        process.terminate.side_effect=ProcessLookupError
        process.wait.return_value=0
        self.assertEqual(supervisor.stop_owned(process),0)
        process.wait.assert_called_once_with()

    def test_direct_timeout_escalates_only_owned_Popen_and_waits(self):
        process=Mock()
        process.poll.return_value=None
        process.wait.side_effect=[subprocess.TimeoutExpired("fixture",30),-9]
        self.assertEqual(supervisor.stop_owned(process),-9)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count,2)

    def test_actual_runtime_synthetic_child_success(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix="supervisor_smoke_") as directory:
            output=Path(directory)
            task=dict(max_seconds=30,max_rss_bytes=2**30,max_output_bytes=2**20)
            entry=dict(started_at=time.time())
            with (output/"synthetic.log").open("wb") as log:
                process=subprocess.Popen([sys.executable,"-s","-B","-c","print('synthetic archive lifecycle smoke')"],
                                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code,error=supervisor.monitor_owned(process,output,task,entry,lambda:None)
            self.assertEqual(code,0)
            self.assertIsNone(error)
            self.assertTrue(supervisor.group_absent(process.pid))
            identity=json.loads((output/"owned_process_identity.json").read_text())
            self.assertEqual(identity["signal_method"],"direct_unreaped_Popen_child")
            self.assertIn("synthetic archive lifecycle smoke",(output/"synthetic.log").read_text())

    def test_actual_setup_failure_terminates_and_waits_synthetic_child(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix="supervisor_failure_smoke_") as directory:
            output=Path(directory)
            task=dict(max_seconds=30,max_rss_bytes=2**30,max_output_bytes=2**20)
            entry=dict(started_at=time.time())
            process=subprocess.Popen([sys.executable,"-s","-B","-c","import time; time.sleep(30)"],
                                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            with patch.object(supervisor,"save",side_effect=RuntimeError("synthetic identity write failure")):
                code,error=supervisor.monitor_owned(process,output,task,entry,lambda:None)
            self.assertEqual(code,-signal.SIGTERM)
            self.assertIn("synthetic identity write failure",error)
            self.assertEqual(process.returncode,-signal.SIGTERM)
            self.assertTrue(supervisor.group_absent(process.pid))

    def test_shared_capacity_lock_blocks_and_controller_is_nonblocking(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix="supervisor_lock_fixture_") as directory:
            with patch.object(supervisor,"ROOT",Path(directory)):
                with patch.object(supervisor.fcntl,"flock") as flock:
                    with supervisor.locked(supervisor.LOCK):
                        pass
                    self.assertEqual(flock.call_args.args[1],supervisor.fcntl.LOCK_EX)
                    with supervisor.locked(supervisor.CONTROLLER_LOCK):
                        pass
                    self.assertEqual(flock.call_args.args[1],supervisor.fcntl.LOCK_EX|supervisor.fcntl.LOCK_NB)

    def test_group_absence_does_not_hide_permission_errors(self):
        with patch.object(supervisor.os,"killpg",side_effect=ProcessLookupError):
            self.assertTrue(supervisor.group_absent(123))
        with patch.object(supervisor.os,"killpg",return_value=None):
            self.assertFalse(supervisor.group_absent(123))
        with patch.object(supervisor.os,"killpg",side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                supervisor.group_absent(123)

    def test_current_native_and_ROOT_reservations_fit_first_archive(self):
        source_ref=recovery.reference(ROOT,"configs/benchmarks/executable_workflow_20261019_v9.json")
        source=recovery.read(ROOT,source_ref)
        changed,_=recovery.add_recovery_tasks(source,"experiments/results/source_only_supervisor_fixture")
        tasks=[task for task in changed["tasks"] if task.get("factory_module")==recovery.MODULE]
        failed_ref=recovery.reference(ROOT,"research/repairs/benchmark_recovery_20261020/scheduler/validation001_request.draft_v2.json")
        failed_scope=recovery.path(ROOT,recovery.read(ROOT,failed_ref)["task"]["output_directory"])
        closed_attempt=dict(request_ref=failed_ref,scope=str(failed_scope),
                            job_ref=recovery.reference(ROOT,failed_scope/"job.json"),
                            wait_ref=recovery.reference(ROOT,failed_scope/"actual_wait.json"),
                            retained_bytes=recovery.scope_bytes(failed_scope))
        config=dict(source_plan_ref=source_ref,closed_native_attempts=[closed_attempt],
                    native_request_ref=recovery.reference(ROOT,"research/repairs/benchmark_recovery_20261020/scheduler/validation001_request.draft_v3.json"),
                    concurrent_ROOT_preparation=dict(max_output_bytes=64*2**20,max_rss_bytes=128*2**30),
                    storage_ceiling_bytes=980_000_000_000,host_ram_margin_bytes=64*2**30,
                    max_parallel_rss_bytes=1280*2**30,
                    recovery_plan=dict(tasks=tasks,external_storage=source["external_storage"],
                         workflow_external_predecessors=source["workflow_external_predecessors"],
                         closed_storage_recovery=changed["closed_storage_recovery"]))
        state=dict(tasks={task["id"]:dict(status="pending") for task in tasks})
        decision=supervisor.capacity(config,state,tasks[0])
        self.assertTrue(decision["storage_allowed"])
        self.assertTrue(decision["ram_allowed"])
        self.assertGreaterEqual(decision["reserved_rss_bytes"],(256+128+1)*2**30)

if __name__=="__main__":
    unittest.main()

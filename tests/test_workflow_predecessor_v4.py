"""Real tiny CPU producers plus bounded metadata, never scientific launches."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import workflow_predecessor_v4 as helper

spec = importlib.util.spec_from_file_location("v4_builder", c.ROOT / "scripts/build_benchmark_workflow_v4.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=c.ROOT / "tests")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        patcher = patch.object(c, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.old = "experiments/results/old"
        self.new = "experiments/results/new"
        def task(key, resource, dependencies=()):
            return dict(id=key, resource=resource, dependencies=list(dependencies),
                factory_module="arggnn.benchmarking.never_called", factory_function="never_called",
                factory_arguments={"source": "unchanged"}, output_directory=self.old + "/" + key,
                max_output_bytes=100000, max_rss_bytes=100000000, max_seconds=60)
        self.done = task("complete", "cpu")
        self.live = task("live", "cpu", ("complete",))
        self.retry = task(builder.FIRST_GPU, "gpu0", ("complete",))
        self.plan = dict(schema="arggnn_executable_workflow_v1", physical_gpu=0,
            records_directory="oldrecords", tasks=[self.retry, self.done, self.live],
            sources=[], external_storage=[], dispatch_hold_task_ids=[], external_native_rss_bytes=1000)
        self.plan_ref = self.write("old_plan.json", self.plan)
        self.entries = {}
        for task in [self.done, self.live, self.retry]:
            job = self.write("oldrecords/jobs/" + task["id"] + ".json",
                dict(plan_ref=self.plan_ref, task=task, task_fingerprint=c.fingerprint(task)))
            self.entries[task["id"]] = dict(status="pending", external=False, job_ref=job)
        self.succeed(self.done)
        self.ready = self.root / "ready"
        self.gate = self.root / "gate"
        code = ("from pathlib import Path;import sys,time;"
                "Path(sys.argv[1]).write_text('ready');"
                "\nwhile not Path(sys.argv[2]).exists():time.sleep(.01)\n")
        self.process = subprocess.Popen([sys.executable, "-B", "-s", "-c", code,
            str(self.ready), str(self.gate)], start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        self.addCleanup(self.clean_process)
        deadline = time.monotonic() + 5
        while not self.ready.exists():
            if time.monotonic() >= deadline:
                self.fail("Tiny real CPU fixture did not become ready")
            time.sleep(.01)
        self.entries["live"].update(status="running", pid=os.getpid(), reservation_bytes=100000)
        self.entries[self.retry["id"]].update(status="failed", actual_controller_wait=1,
            owned_process_group_absent=False, reservation_bytes=100000)
        wait_ref = self.write(self.retry["output_directory"] + "/actual_wait.json",
            dict(job_ref=self.entries[self.retry["id"]]["job_ref"], actual_direct_wait=True,
                 actual_direct_returncode=-15, owned_process_group_absent=False, output_bytes=200))
        self.observation_ref = self.write("observation.json",
            dict(schema="workflow_v4_fresh_namespace_root_observation_v1",
                failed_original_wait_ref=wait_ref, historical_owned_group_absence="unproven, preserve original false value",
                root_decision="Authorize only a fresh namespace, preserve historical unknown"))
        self.state_ref = self.save_state()
        self.binding = builder.live_binding(self.live, self.entries["live"], self.plan_ref,
            self.state_ref, "oldrecords/state.json")
        self.external = dict(id="live", resource="external", dependencies=["complete"],
                             workflow_predecessor=self.binding)
        self.external_plan = dict(workflow_external_predecessors=[self.binding])

    def clean_process(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=5)

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return c.file_ref(path)

    def save_state(self):
        state = dict(plan_ref=self.plan_ref, tasks=self.entries)
        self.write("oldrecords/state.json", state)
        return self.write("snapshot.json", state)

    def succeed(self, task, value=None):
        entry = self.entries[task["id"]]
        result = self.write(task["output_directory"] + "/result.json",
            dict(job_ref=entry["job_ref"], value=value or {"actual_result": "synthetic fixture"}))
        self.write(task["output_directory"] + "/actual_wait.json",
            dict(job_ref=entry["job_ref"], actual_direct_wait=True, actual_direct_returncode=0,
                 owned_process_group_absent=True, supervisor_error=None, output_bytes=100))
        entry.update(status="succeeded", actual_controller_wait=0, owned_process_group_absent=True,
                     reservation_bytes=100000, retained_bytes=1000, result_ref=result)

    def finish_real_producer(self, value=None):
        self.gate.write_text("continue")
        code = self.process.wait(timeout=5)
        self.assertEqual(code, 0)
        with self.assertRaises(ProcessLookupError):
            os.killpg(self.process.pid, 0)
        self.succeed(self.live, value)
        self.save_state()

    def prepare(self):
        return builder.prepare_continuation(self.plan_ref, self.state_ref, self.observation_ref,
            records="newrecords", prior_output=self.old, output=self.new,
            state_path="oldrecords/state.json", expected_successes=1, expected_live_ids=("live",))

    def test_real_live_producer_waits_and_counts_full_resources_once(self):
        self.assertIsNone(helper.external_result(self.external))
        self.assertEqual(helper.external_resources(self.external_plan), (1, 100000000))
        self.assertEqual(helper.storage_reservation_bytes(self.external_plan), 100000)
        groups = helper.live_group_bindings(self.external_plan)
        self.assertEqual(groups[0]["pid"], self.process.pid)
        self.assertEqual(groups[0]["pgid"], self.process.pid)
        self.assertEqual(groups[0]["session"], self.process.pid)

    def test_real_success_unwraps_original_value_and_releases_only_real_retained_bytes(self):
        value = {"path": "giant_payload.json", "sha256": "a" * 64}
        # A sparse oversized non-metadata file must not be opened by adoption.
        with (self.root / "giant_payload.json").open("wb") as stream:
            stream.truncate(c.MAX_METADATA_BYTES + 1)
        self.finish_real_producer(value)
        with self.assertRaises(c.ContractError):
            c.read_ref(value)
        result = helper.external_result(self.external)
        self.assertEqual(result["value"], value)
        self.assertEqual(result["external_source_result_ref"], self.entries["live"]["result_ref"])
        self.assertEqual(helper.external_resources(self.external_plan), (0, 0))
        self.assertEqual(helper.storage_reservation_bytes(self.external_plan), 1000)
        self.assertEqual(helper.live_group_bindings(self.external_plan), [])

    def test_claimed_success_without_group_absence_never_releases(self):
        self.succeed(self.live)
        self.entries["live"]["owned_process_group_absent"] = False
        self.save_state()
        self.assertIsNone(helper.external_result(self.external))
        self.assertEqual(helper.external_resources(self.external_plan), (1, 100000000))
        self.assertEqual(helper.storage_reservation_bytes(self.external_plan), 100000)

    def test_wrong_original_job_or_supervisor_is_rejected(self):
        self.entries["live"]["pid"] += 1000
        self.save_state()
        with self.assertRaisesRegex(ValueError, "supervisor identities"):
            helper.external_result(self.external)

    def test_builder_preserves_success_live_scope_unknown_failure_and_exact_science(self):
        old_plan = copy.deepcopy(self.plan)
        plan, entries, evidence = self.prepare()
        tasks = {t["id"]: t for t in plan["tasks"]}
        self.assertEqual(tasks["complete"], self.done)
        self.assertEqual(entries["complete"]["result_ref"], self.entries["complete"]["result_ref"])
        self.assertEqual(tasks["live"]["resource"], "external")
        self.assertEqual(tasks["live"]["workflow_predecessor"]["output_directory"], self.live["output_directory"])
        self.assertNotIn("reservation_bytes", entries["live"])
        self.assertEqual(tasks[self.retry["id"]]["factory_arguments"], self.retry["factory_arguments"])
        self.assertEqual(tasks[self.retry["id"]]["output_directory"], self.new + "/" + self.retry["id"])
        self.assertEqual(plan["prior_unknown_rss_bytes"], self.retry["max_rss_bytes"])
        self.assertEqual(plan["external_storage"][-1]["cap_bytes"], 100000)
        self.assertEqual(len(plan["external_storage"]), 1)
        self.assertEqual(evidence["retained_failed_attempts"][0]["original_entry"]["owned_process_group_absent"], False)
        self.assertEqual(c.read_ref(self.plan_ref), old_plan)
        self.assertFalse((self.root / self.new).exists())

    def test_original_candidate_that_really_failed_drained_gets_fresh_scope_and_actual_charge(self):
        self.process.terminate()
        code = self.process.wait(timeout=5)
        self.assertEqual(code, -15)
        with self.assertRaises(ProcessLookupError):
            os.killpg(self.process.pid, 0)
        self.write(self.live["output_directory"] + "/actual_wait.json",
            dict(job_ref=self.entries["live"]["job_ref"], actual_direct_wait=True,
                 actual_direct_returncode=code, owned_process_group_absent=True, output_bytes=100))
        self.entries["live"].update(status="failed", actual_controller_wait=1,
            owned_process_group_absent=True, retained_bytes=500)
        self.state_ref = self.save_state()
        plan, entries, evidence = self.prepare()
        tasks = {t["id"]: t for t in plan["tasks"]}
        self.assertEqual(plan["workflow_external_predecessors"], [])
        self.assertEqual(tasks["live"]["output_directory"], self.new + "/live")
        self.assertEqual(entries["live"]["status"], "pending")
        self.assertEqual(evidence["total_failed_charge_bytes"], 100500)
        self.assertEqual(plan["prior_unknown_rss_bytes"], self.retry["max_rss_bytes"])

    def test_new_scope_and_wrong_unknown_acceptance_are_rejected(self):
        (self.root / self.new).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "fresh and disjoint"):
            self.prepare()
        (self.root / self.new).rmdir()
        observation = c.read_ref(self.observation_ref)
        observation["failed_original_wait_ref"] = self.entries["complete"]["result_ref"]
        self.observation_ref = self.write("observation.json", observation)
        with self.assertRaisesRegex(ValueError, "ROOT must explicitly accept"):
            self.prepare()

    def test_source_failure_propagates_and_still_reserves_undrained_group(self):
        self.entries["live"].update(status="failed", actual_controller_wait=1,
                                    owned_process_group_absent=False)
        self.save_state()
        self.assertEqual(helper.external_result(self.external)["status"], "failed")
        self.assertEqual(helper.external_resources(self.external_plan), (1, 100000000))
        self.assertEqual(helper.storage_reservation_bytes(self.external_plan), 100000)

if __name__ == "__main__":
    unittest.main()

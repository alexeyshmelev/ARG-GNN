"""Small fake-process tests; no benchmark, guard, GPU, data or download is used."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from arggnn.benchmarking.idle_preflight import idle_preflight
from arggnn.benchmarking.scheduler import AdmittedJob, PersistentScheduler, SchedulerBusy


def reference(name):
    return {"path": "RAM_ONLY/" + name, "sha256": "a" * 64}


class FakeProcess:
    def __init__(self, owner, job, ticks, returncode):
        self.owner, self.job, self.ticks, self.returncode = owner, job, ticks, returncode
        self.pid = 1000 + len(owner.calls)
        self.waited = False

    def poll(self):
        if self.ticks:
            self.ticks -= 1
            return None
        return self.returncode

    def wait(self):
        self.waited = True
        self.owner.active[self.job.resource] -= 1
        self.owner.events.append(("wait", self.job.job_id))
        return self.returncode


class FakeRunner:
    def __init__(self, admissions, *, ticks=2, returncodes=None):
        self.admissions, self.ticks = admissions, ticks
        self.returncodes = returncodes or {}
        self.calls, self.events = [], []
        self.active = {"gpu0": 0, "cpu": 0}
        self.maximum = dict(self.active)

    def __call__(self, argv, *, cwd):
        job = next(j for j in self.admissions.values() if list(j.argv) == argv)
        self.events.append(("spawn", job.job_id))
        self.calls.append((argv, cwd))
        self.active[job.resource] += 1
        self.maximum[job.resource] = max(self.maximum[job.resource], self.active[job.resource])
        return FakeProcess(self, job, self.ticks, self.returncodes.get(job.job_id, 0))


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        # Repository instructions keep even smoke-test temporaries inside it.
        self.temporary = tempfile.TemporaryDirectory(prefix=".scheduler-test-", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).absolute()
        self.admissions = {}
        self.runner = FakeRunner(self.admissions)
        self.policy_calls = []

    def approve(self, policy, job):
        self.assertEqual(policy, reference("approved_policy"))
        self.policy_calls.append(job.job_id)
        self.runner.events.append(("approve", job.job_id))

    def scheduler(self, **kwargs):
        options = dict(validate_admission=lambda receipt: self.admissions[receipt["path"]],
                       validate_execution_policy=self.approve, runner=self.runner,
                       cpu_parallelism=2, poll_seconds=0, sleep=lambda _: None)
        options.update(kwargs)
        return PersistentScheduler(self.root / "queue.json", self.root / "controller.lock", **options)

    def job(self, name, resource="gpu0", **changes):
        job = AdmittedJob(name, resource, ("/already/admitted/guarded_worker", "--job", name),
                          str(self.root), reference(name + "_input"), reference(name + "_receipt"), 600)
        job = replace(job, **changes)
        self.admissions[job.admission_receipt_ref["path"]] = job
        return job

    def enqueue(self, scheduler, name, resource="gpu0"):
        job = self.job(name, resource)
        scheduler.enqueue(job.admission_receipt_ref)
        return job

    def execute(self, scheduler):
        return scheduler.run(execute=True, execution_policy_ref=reference("approved_policy"))

    def test_first_enqueue_creates_fresh_nested_metadata_parents_only(self):
        fresh = self.root / "fresh"
        state = fresh / "queue" / "state.json"
        controller = fresh / "control" / "controller.lock"
        scheduler = PersistentScheduler(
            state, controller,
            validate_admission=lambda receipt: self.admissions[receipt["path"]],
            validate_execution_policy=self.approve, runner=self.runner,
            poll_seconds=0, sleep=lambda _: None)
        self.assertEqual(scheduler.run()["jobs"], [])
        self.assertFalse(fresh.exists())
        self.enqueue(scheduler, "one")
        self.assertTrue(state.is_file())
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(scheduler.jobs()[0]["status"], "queued")
        self.assertFalse(controller.parent.exists())
        self.execute(scheduler)  # Fake runner only; exercises the other nested lock parent.
        self.assertTrue(controller.is_file())
        self.assertEqual(scheduler.jobs()[0]["status"], "succeeded")

    def test_queue_metadata_paths_cannot_leave_repository(self):
        with self.assertRaises(ValueError):
            PersistentScheduler("/outside-repository/queue.json", self.root / "controller.lock",
                                validate_admission=lambda _: None)

    def test_default_is_plan_only_even_with_an_admitted_receipt(self):
        scheduler = self.scheduler()
        self.enqueue(scheduler, "one")
        self.assertEqual(scheduler.run()["mode"], "plan")
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(scheduler.jobs()[0]["status"], "queued")

    def test_execution_flag_does_not_accept_truthy_strings(self):
        scheduler = self.scheduler()
        self.enqueue(scheduler, "one")
        with self.assertRaises(TypeError):
            scheduler.run(execute="false", execution_policy_ref=reference("approved_policy"))
        self.assertEqual(self.runner.calls, [])

    def test_execution_requires_both_explicit_policy_and_validator(self):
        scheduler = self.scheduler(validate_execution_policy=None)
        self.enqueue(scheduler, "one")
        with self.assertRaises(PermissionError):
            scheduler.run(execute=True, execution_policy_ref=reference("approved_policy"))
        with self.assertRaises(PermissionError):
            self.scheduler().run(execute=True)
        self.assertEqual(self.runner.calls, [])

    def test_raw_commands_cannot_be_enqueued(self):
        with self.assertRaises(ValueError):
            self.scheduler().enqueue(["python", "raw_gpu_work.py"])
        self.assertFalse((self.root / "queue.json").exists())

    def test_validator_cannot_substitute_another_receipt(self):
        job = self.job("one")
        scheduler = self.scheduler(validate_admission=lambda _: job)
        with self.assertRaises(ValueError):
            scheduler.enqueue(reference("different_receipt"))

    def test_gpu_is_serial_and_cpu_parallelism_is_bounded(self):
        scheduler = self.scheduler()
        for name, resource in [("g1", "gpu0"), ("c1", "cpu"), ("c2", "cpu"),
                               ("g2", "gpu0"), ("c3", "cpu"), ("c4", "cpu")]:
            self.enqueue(scheduler, name, resource)
        result = self.execute(scheduler)
        self.assertEqual(len(result["launched_jobs"]), 6)
        self.assertEqual(self.runner.maximum, {"gpu0": 1, "cpu": 2})
        self.assertTrue(all(j["status"] == "succeeded" and j["actual_returncode"] == 0
                            for j in scheduler.jobs()))
        self.assertFalse(result["scientific_completion_claimed"])

    def test_policy_is_checked_immediately_before_each_spawn(self):
        scheduler = self.scheduler()
        for name in ("one", "two"):
            self.enqueue(scheduler, name)
        self.execute(scheduler)
        for name in ("one", "two"):
            at = self.runner.events.index(("spawn", name))
            self.assertEqual(self.runner.events[at - 1], ("approve", name))

    def test_rejected_policy_keeps_job_unstarted(self):
        scheduler = self.scheduler(validate_execution_policy=lambda *_: False)
        self.enqueue(scheduler, "one")
        with self.assertRaises(PermissionError):
            self.execute(scheduler)
        self.assertEqual(scheduler.jobs()[0]["status"], "queued")
        self.assertIsNone(scheduler.jobs()[0]["started_at"])
        self.assertEqual(self.runner.calls, [])

    def test_policy_expiry_between_serial_jobs_does_not_launch_second(self):
        def approve(policy, job):
            if job.job_id == "two":
                raise PermissionError("Expired ROOT policy")
            self.approve(policy, job)
        scheduler = self.scheduler(validate_execution_policy=approve)
        self.enqueue(scheduler, "one")
        self.enqueue(scheduler, "two")
        with self.assertRaises(PermissionError):
            self.execute(scheduler)
        self.assertEqual([j["status"] for j in scheduler.jobs()], ["succeeded", "queued"])
        self.assertEqual(len(self.runner.calls), 1)

    def test_actual_nonzero_wait_is_preserved_without_retry(self):
        scheduler = self.scheduler()
        self.runner.returncodes["one"] = -15
        self.enqueue(scheduler, "one")
        self.execute(scheduler)
        self.execute(self.scheduler())
        state = scheduler.jobs()[0]
        self.assertEqual((state["status"], state["actual_returncode"]), ("failed", -15))
        self.assertEqual(len(self.runner.calls), 1)

    def test_completion_files_do_not_imply_success(self):
        scheduler = self.scheduler()
        self.enqueue(scheduler, "one")
        (self.root / "completed.json").write_text('{"status":"completed"}')
        self.runner.returncodes["one"] = 7
        self.execute(scheduler)
        self.assertEqual(scheduler.jobs()[0]["actual_returncode"], 7)
        self.assertEqual(scheduler.jobs()[0]["status"], "failed")

    def test_never_started_jobs_survive_restart(self):
        self.enqueue(self.scheduler(), "one")
        result = self.execute(self.scheduler())
        self.assertEqual(result["launched_jobs"], ["one"])

    def test_unknown_gpu_start_blocks_gpu_lane_but_not_cpu(self):
        scheduler = self.scheduler()
        self.enqueue(scheduler, "old")
        self.enqueue(scheduler, "next")
        self.enqueue(scheduler, "cpu", "cpu")
        scheduler._set("old", status="started", started_at=1, pid=999)
        result = self.execute(self.scheduler())
        self.assertEqual(result["launched_jobs"], ["cpu"])
        statuses = {j["admission"]["job_id"]: j["status"] for j in scheduler.jobs()}
        self.assertEqual(statuses, {"old": "unknown", "next": "queued", "cpu": "succeeded"})
        self.assertIsNone(scheduler.jobs()[0]["actual_returncode"])

    def test_unknown_cpu_starts_consume_bounded_cpu_lanes(self):
        scheduler = self.scheduler(cpu_parallelism=1)
        self.enqueue(scheduler, "old", "cpu")
        self.enqueue(scheduler, "next", "cpu")
        scheduler._set("old", status="started", started_at=1)
        self.assertEqual(self.execute(self.scheduler(cpu_parallelism=1))["launched_jobs"], [])

    def test_changed_admission_is_rejected_before_spawn(self):
        scheduler = self.scheduler()
        job = self.enqueue(scheduler, "one")
        self.admissions[job.admission_receipt_ref["path"]] = replace(job, argv=("/different/worker",))
        self.execute(scheduler)
        self.assertEqual(scheduler.jobs()[0]["status"], "rejected")
        self.assertEqual(self.runner.calls, [])

    def test_failure_to_launch_is_preserved_without_retry(self):
        def fail(*_, **__):
            raise OSError("fixture exec failed")
        scheduler = self.scheduler(runner=fail)
        self.enqueue(scheduler, "one")
        self.execute(scheduler)
        self.assertEqual(scheduler.jobs()[0]["status"], "failed")
        self.assertIn("fixture exec failed", scheduler.jobs()[0]["error"])
        self.assertEqual(self.execute(self.scheduler())["launched_jobs"], [])

    def test_same_input_cannot_be_requeued_under_another_id(self):
        scheduler = self.scheduler()
        job = self.enqueue(scheduler, "one")
        duplicate = self.job("two", input_ref=job.input_ref)
        with self.assertRaises(ValueError):
            scheduler.enqueue(duplicate.admission_receipt_ref)

    def test_same_job_cannot_be_rebound(self):
        scheduler = self.scheduler()
        job = self.enqueue(scheduler, "one")
        changed = replace(job, argv=("/different/worker",))
        self.admissions[job.admission_receipt_ref["path"]] = changed
        with self.assertRaises(ValueError):
            scheduler.enqueue(job.admission_receipt_ref)

    def test_queue_job_bound(self):
        scheduler = self.scheduler(max_jobs=1)
        self.enqueue(scheduler, "one")
        with self.assertRaises(ValueError):
            self.enqueue(scheduler, "two")
        self.assertEqual(len(scheduler.jobs()), 1)

    def test_only_one_logical_controller(self):
        scheduler = self.scheduler()
        self.enqueue(scheduler, "one")
        with scheduler._lock(scheduler.controller_lock_path):
            with self.assertRaises(SchedulerBusy):
                self.execute(scheduler)
        self.assertEqual(self.runner.calls, [])

    def test_controller_lock_cannot_be_changed_for_persisted_queue(self):
        scheduler = self.scheduler()
        self.enqueue(scheduler, "one")
        other = PersistentScheduler(self.root / "queue.json", self.root / "other.lock",
                                    validate_admission=lambda r: self.admissions[r["path"]])
        with self.assertRaises(ValueError):
            other.plan()

    def test_physical_gpu_lease_is_never_a_queue_lock(self):
        with self.assertRaises(ValueError):
            PersistentScheduler(self.root / "queue.json", self.root / "gpu_0.lock",
                                validate_admission=lambda _: None)
        self.assertFalse((self.root / "gpu_0.lock").exists())

    def test_no_other_physical_gpu_resource(self):
        with self.assertRaises(ValueError):
            self.job("wrong_gpu", "gpu1")


class PreflightTests(unittest.TestCase):
    def test_no_slot_skips_rss_and_expensive_accounting(self):
        def forbidden():
            self.fail("Blocked queue must not call RSS/accounting")
        result = idle_preflight(available_slots=lambda: 0, available_rss_bytes=forbidden,
                                required_rss_bytes=10, capture_accounting=forbidden)
        self.assertFalse(result.ready)
        self.assertEqual(result.reason, "no_existing_slot")

    def test_low_rss_skips_expensive_accounting(self):
        def forbidden():
            self.fail("Blocked queue must not capture fresh accounting state")
        result = idle_preflight(available_slots=lambda: 1, available_rss_bytes=lambda: 9,
                                required_rss_bytes=10, capture_accounting=forbidden)
        self.assertFalse(result.ready)
        self.assertEqual(result.reason, "insufficient_rss")

    def test_positive_preflight_calls_accounting_once_without_caching(self):
        calls = []
        marker = object()
        def capture():
            calls.append(True)
            return marker
        for _ in range(2):
            result = idle_preflight(available_slots=lambda: 1, available_rss_bytes=lambda: 10,
                                    required_rss_bytes=10, capture_accounting=capture)
            self.assertTrue(result.ready)
            self.assertIs(result.accounting, marker)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

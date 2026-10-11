"""Bounded handmade fixture using original Controller methods, fake resources only.

All writes/locks/socket paths stay in a fresh contained author fixture directory.
No real inventory, proc scan, subprocess, CUDA/model import, installation or GPU.
"""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import types
from unittest import mock

PROJECT = Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path.insert(0, str(PROJECT / "src"))
from arggnn import budgeted_shared_pool_v1 as shared
from arggnn import gpu_pool_v1 as gpu
from arggnn import project_budget_v4 as original_budget

AUTHOR = PROJECT / "research/reviews/budgeted_shared_pool_v1_author"
FAKE = AUTHOR / "f"
CHECKS = []
MIB = 1024 * 1024


def passed(name):
    CHECKS.append(name)


def expect(label, exception, message, function):
    try:
        function()
    except exception as error:
        assert message in str(error), (label, str(error))
        passed(label)
        return str(error)
    raise AssertionError(label + ": expected rejection absent")


class FakeChild:
    children = []
    def __init__(self, argv, **kwargs):
        assert kwargs["shell"] is False
        assert kwargs["start_new_session"] is True
        assert kwargs["close_fds"] is True
        assert len(kwargs["pass_fds"]) == 1
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
        assert kwargs["env"]["ARGGNN_PHYSICAL_GPU"] == "2"
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        self.pid = 800000 + len(self.children)
        self.argv, self.returncode = list(argv), None
        rfd, wfd = os.pipe()
        os.close(wfd)
        self.stdout = os.fdopen(rfd, "rb", buffering=0)
        self.children.append(self)
    def poll(self):
        return self.returncode


def fake_proc(_lease, _controller, _known=None):
    return {child.pid: {
        "pid": child.pid, "uid": os.getuid(), "state": "S",
        "ppid": os.getpid(), "pgid": child.pid, "sid": child.pid,
        "start_ticks": child.pid, "lease_fd_present": True,
        "controller_descendant": True,
    } for child in FakeChild.children if child.returncode is None}


def main():
    assert not FAKE.exists(), "fresh owned fixture directory required"
    immutable = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                 for name in ("src/arggnn/gpu_pool_v1.py", "src/arggnn/project_budget_v4.py")}
    assert immutable["src/arggnn/gpu_pool_v1.py"] == shared.FIXED_POLICY["legacy_pool_source"]["sha256"]
    assert immutable["src/arggnn/project_budget_v4.py"] == shared.FIXED_POLICY["budget_source"]["sha256"]
    FAKE.mkdir()
    controllers = []
    started = time.monotonic()
    old_alarm = signal.getsignal(signal.SIGALRM)
    def deadline(_signum, _frame):
        raise TimeoutError("bounded fake Controller flow exceeded 25 seconds")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(25)
    try:
        def text(path, value):
            p = FAKE / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(value)
            return original_budget.reference(FAKE, p)
        def data(path, value):
            return text(path, json.dumps(value, indent=2) + "\n")
        def bound(path):
            return original_budget.reference(FAKE, path)
        for path in ("src/arggnn/gpu_pool_v1.py", "src/arggnn/project_budget_v4.py",
                     "src/arggnn/budgeted_shared_pool_v1.py"):
            text(path, (PROJECT / path).read_text())
        cache_source = text("src/arggnn/biological_bank_cache_v1.py", "# fake inert worker bytes\n")
        cache_config = data("configs/biological_bank_cache_v1.json", {"GPU_budget_bytes": 8 * MIB})
        runtime = text("scripts/project_env.sh", "# fake inert runtime binding\n")
        human = data("meta/human.json", {
            "kind": "human_project_storage_authorization", "project_root": str(FAKE),
            "authorized_hard_limit_bytes": 1000000000000,
            "future_scheduling_ceiling_bytes": 950000000000,
        })
        now = dt.datetime.now(dt.timezone.utc)
        snapshot = data("meta/snapshot.json", {
            "schema": "biological_candidate_storage_snapshot_v1", "status": "complete",
            "returncode": 0, "project_root": str(FAKE),
            "command": ["du", "-s", "-B1", str(FAKE)],
            "used_bytes": 1000000, "completed_utc": (now - dt.timedelta(seconds=1)).isoformat(),
        })
        private = types.ModuleType("contained_budget_fixture")
        raw_budget = (PROJECT / "src/arggnn/project_budget_v4.py").read_text()
        exec(compile(raw_budget, "<unchanged-bound-budget-source>", "exec"), private.__dict__)
        private.ROOT = FAKE
        private.SOURCE = FAKE / "src/arggnn/project_budget_v4.py"
        private.CONFIG = FAKE / "configs/project_budget_v4.json"
        private.FIXED_POLICY = copy.deepcopy(original_budget.FIXED_POLICY)
        private.FIXED_POLICY.update(root=str(FAKE), human_authorization=human)
        private.FIXED_POLICY["legacy_invariants"].update(
            snapshot_sha256=snapshot["sha256"], baseline_bytes=1000000, external_reserve_bytes=1000)
        budget_config = data("configs/project_budget_v4.json", private.FIXED_POLICY)
        policy = copy.deepcopy(shared.FIXED_POLICY)
        policy.update(root=str(FAKE), human_authorization=human, budget_config=budget_config,
                      cache_source=cache_source, cache_config=cache_config,
                      socket_relative_template=".runtime/gpup/{pool}/s")
        config = data("configs/budgeted_shared_pool_v1.json", policy)
        implementation = bound("src/arggnn/budgeted_shared_pool_v1.py")
        common = {"implementation": bound("src/arggnn/project_budget_v4.py"),
                  "config": budget_config, "human_authorization": human}
        consumer_base = [implementation, config]
        requests = []
        pool_requests, job_requests = {}, {}
        for pool in ("a", "b", "c"):
            pool_requests[pool] = {
                "key": "budgeted_shared_pool_v1/" + pool + "/controller",
                "output": str(FAKE / ".runtime/gpup" / pool), "auxiliary_scopes": [],
                "reserved_bytes": gpu.MIN_CONTROLLER_RESERVE + 3000,
                "consumer_bindings": consumer_base,
            }
            requests.append(pool_requests[pool])
            job_requests[pool] = []
            for job in ("one", "two", "three"):
                request = {
                    "key": "budgeted_shared_pool_v1/" + pool + "/" + job,
                    "output": str(FAKE / "outputs" / (pool + "_" + job)),
                    "auxiliary_scopes": [], "reserved_bytes": 5000,
                    "consumer_bindings": [*consumer_base, cache_source, cache_config],
                }
                job_requests[pool].append(request)
                requests.append(request)
        budget_review = data("meta/budget_review.json", {
            "kind": "project_budget_v4_independent_review", "review_role": "independent_reviewer",
            "verdict": "PASS", **common, "actual_apply_performed": False,
            "fixture_only_not_real_review": True,
        })
        def reserve_admission(path, incoming):
            return data(path, {
                "kind": "project_budget_v4_root_reserve_admission", "root_authorizer": "root_agent",
                "active": True, **common, "all_budget_writers_use_scheduler_mutex": True,
                "no_legacy_mutation_authorized": True, "GPU_authorized": False,
                "scientific_execution_authorized": False, "global_rebase_authorized": False,
                "reservations": incoming, "independent_review": budget_review,
                "fixture_only_not_real_authority": True,
            })
        root_reserve = reserve_admission("meta/budget_admission.json", requests)
        stored = {request["key"]: {**request, "root_admission": root_reserve} for request in requests}
        legacy = {**private.FIXED_POLICY["legacy_invariants"], "root": str(FAKE),
                  "controllers": {}, "jobs": {}}
        legacy_ref = data(private.LEGACY, legacy)
        bindings = common
        supplemental = private.empty_ledger(FAKE, private.FIXED_POLICY, bindings)
        supplemental["reservations"] = stored
        supplemental_ref = data(private.SUPPLEMENTAL, supplemental)
        legacy_before = (FAKE / private.LEGACY).read_bytes()
        supplemental_before = (FAKE / private.SUPPLEMENTAL).read_bytes()
        (FAKE / "outputs").mkdir()
        (FAKE / ".runtime/locks").mkdir(parents=True)
        (FAKE / private.LOCK).touch()
        reviewed = {}
        for index, pool in enumerate(("a", "b", "c")):
            admission_path = "meta/pool_" + pool + ".json"
            recipe = [str(FAKE / ".runtime/env/bin/python"), "-m", "arggnn.budgeted_shared_pool_v1", "--start",
                      "--admission", admission_path, "--expected-admission-sha256", "<ROOT_ADMISSION_SHA256>",
                      "--expected-source-sha256", implementation["sha256"],
                      "--expected-config-sha256", config["sha256"]]
            jobs = []
            for job_index, request in enumerate(job_requests[pool]):
                memory = {
                    "model_optimizer_baseline": 5, "cached_bank": 10,
                    "uncached_bank_and_retained_references": 5,
                    "autograd": (10, 5, 20)[job_index], "runtime_allocator": 10,
                }
                spec = {
                    "job": ("one", "two", "three")[job_index], "model": "inert cache fixture",
                    "seed": 1, "module": shared.CACHE_MODULE, "args": ["--fake-only", str(job_index)],
                    "output": request["output"], "peak_mib": sum(memory.values()),
                    "storage_bytes": request["reserved_bytes"], "policy": "queue", "log_limit_bytes": 1000,
                }
                jobs.append({
                    "spec": spec, "reservation": stored[request["key"]], "module_source": cache_source,
                    "worker_bindings": [cache_config], "memory_components_mib": memory,
                    "argv_sha256": shared.digest([str(FAKE / ".runtime/env/bin/python"), "-m", spec["module"], *spec["args"]]),
                })
            plan = {
                "pool": {"name": pool, "physical_gpu": index + 2, "headroom_mib": 10,
                         "controller_reservation": stored[pool_requests[pool]["key"]],
                         "storage_snapshot": snapshot, "python": str(FAKE / ".runtime/env/bin/python"),
                         "start_recipe_sha256": shared.digest(recipe)},
                "jobs": jobs, "initial_ledger_pins": {"legacy_ledger_sha256": legacy_ref["sha256"],
                "supplemental_ledger_sha256": supplemental_ref["sha256"]}, "runtime_bindings": [runtime],
            }
            review = data("meta/review_" + pool + ".json", {
                "kind": "budgeted_shared_pool_v1_independent_review", "review_role": "independent_reviewer",
                "verdict": "PASS", "implementation": implementation, "config": config,
                "legacy_pool_source": policy["legacy_pool_source"], "budget_source": policy["budget_source"],
                "budget_config": policy["budget_config"], "human_authorization": human,
                "reviewed_plan_sha256": shared.digest(plan),
                "actual_live_lock_or_gpu_execution_performed": False, "original_sources_unchanged": True,
                "fixture_only_not_real_review": True,
            })
            authority = data(admission_path, {
                "kind": "budgeted_shared_pool_v1_root_admission", "root_authorizer": "root_agent",
                "active": True, "root": str(FAKE), "implementation": implementation, "config": config,
                "legacy_pool_source": policy["legacy_pool_source"], "budget_source": policy["budget_source"],
                "budget_config": budget_config, "human_authorization": human,
                "future_pools_only": True, "no_live_pool_or_worker_patch": True,
                "no_legacy_ledger_mutation": True, "no_duplicate_legacy_reservations": True,
                "scientific_execution_requires_worker_authority": True,
                "not_before_utc": (now - dt.timedelta(minutes=1)).isoformat(),
                "expires_utc": (now + dt.timedelta(hours=1)).isoformat(), "plan": plan,
                "independent_review": review,
            })
            reviewed[pool] = shared.ReviewedPlan(FAKE, policy, implementation, config, authority, api=private)
        authority = copy.deepcopy(reviewed["a"].authority)
        authority["plan"]["jobs"][0]["spec"]["peak_mib"] = 8
        review_value = original_budget.bound(FAKE, authority["independent_review"], True)
        review_value["reviewed_plan_sha256"] = shared.digest(authority["plan"])
        authority["independent_review"] = data("meta/small_peak_review.json", review_value)
        small_authority = data("meta/small_peak_authority.json", authority)
        expect("cache_cap_cannot_be_total_worker_peak", ValueError, "include all",
               lambda: shared.ReviewedPlan(FAKE, policy, implementation, config, small_authority, api=private))
        authority = copy.deepcopy(reviewed["a"].authority)
        memory = authority["plan"]["jobs"][0]["memory_components_mib"]
        memory["cached_bank"] = 1
        authority["plan"]["jobs"][0]["spec"]["peak_mib"] = sum(memory.values())
        review_value = original_budget.bound(FAKE, authority["independent_review"], True)
        review_value["reviewed_plan_sha256"] = shared.digest(authority["plan"])
        authority["independent_review"] = data("meta/small_cache_review.json", review_value)
        small_authority = data("meta/small_cache_authority.json", authority)
        expect("full_configured_cache_cap_required", ValueError, "full cache cap",
               lambda: shared.ReviewedPlan(FAKE, policy, implementation, config, small_authority, api=private))
        for method in ("start", "_memory", "_process_scan", "submit", "_launch_queue",
                       "_drain_logs", "_signal_own", "tick", "shutdown", "close", "dispatch", "run"):
            assert getattr(shared.BudgetedController, method) is getattr(gpu.Controller, method), method
        passed("original_controller_lease_memory_proc_FIFO_subprocess_methods_inherited")
        rows = [{
            "index": index, "uuid": "GPU-00000000-0000-0000-0000-" + ("%012d" % index),
            "used_mib": 0, "total_mib": 100, "utilization": 0, "compute": [],
        } for index in gpu.ALLOWED_GPUS]
        available = [1000000000000]
        inventory = lambda: copy.deepcopy(rows)
        pool_a = shared.BudgetedController(reviewed["a"], inventory=inventory,
                    proc_reader=fake_proc, available_bytes=lambda: available[0])
        pool_b = shared.BudgetedController(reviewed["b"], inventory=inventory,
                    proc_reader=lambda *_: {}, available_bytes=lambda: available[0])
        pool_c = shared.BudgetedController(reviewed["c"], inventory=inventory,
                    proc_reader=lambda *_: {}, available_bytes=lambda: available[0])
        expect("owned_mutex_requires_existing_proven_EX_FLOCK", RuntimeError,
               "not proved", lambda: pool_a.storage._check(reviewed["a"].reservations()[0], owned=True))
        with mock.patch.object(gpu, "enable_child_subreaper", lambda: None), \
                mock.patch.object(gpu.subprocess, "Popen", FakeChild):
            pool_a.storage.preflight()
            pool_a.start()
            controllers.append(pool_a)
            passed("original_start_acquire_device_owned_checker_no_nested_deadlock")
            pool_b.storage.preflight()
            pool_b.start()
            controllers.append(pool_b)
            expect("original_two_physical_device_limit", RuntimeError,
                   "Two distinct", pool_c.start)
            first = copy.deepcopy(reviewed["a"].jobs["one"]["spec"])
            second = copy.deepcopy(reviewed["a"].jobs["two"]["spec"])
            third = copy.deepcopy(reviewed["a"].jobs["three"]["spec"])
            assert pool_a.submit(first)["state"] == "running"
            assert pool_a.submit(second)["state"] == "running"
            assert len(FakeChild.children) == 2
            assert pool_a._memory()["active_peak_mib"] == 75
            passed("original_multiple_fitting_workers_share_same_GPU")
            assert pool_a.submit(third)["state"] == "queued"
            assert len(FakeChild.children) == 2
            passed("original_FIFO_and_pending_CUDA_memory_reservation")
            changed = copy.deepcopy(first)
            changed["job"] = "unregistered"
            expect("unregistered_job_rejected", ValueError, "unregistered", lambda: pool_a.submit(changed))
            changed = copy.deepcopy(second)
            changed["args"].append("--alter-science")
            expect("unregistered_argv_rejected", ValueError, "root job/output/args", lambda: pool_a.submit(changed))
            changed = copy.deepcopy(second)
            changed["output"] = str(FAKE / "outputs/altered")
            expect("unregistered_output_rejected", ValueError, "root job/output/args", lambda: pool_a.submit(changed))
            source_path = FAKE / cache_source["path"]
            original = source_path.read_bytes()
            source_path.write_bytes(original + b"# drift\n")
            expect("stale_worker_source_blocks_launch_boundary", ValueError,
                   "bytes changed", pool_a._lease_check)
            source_path.write_bytes(original)
            config_path = FAKE / config["path"]
            original = config_path.read_bytes()
            config_path.write_bytes(original + b"\n")
            expect("stale_companion_config_blocks_launch_boundary", ValueError,
                   "bytes changed", pool_a._lease_check)
            config_path.write_bytes(original)
            changed = copy.deepcopy(supplemental)
            del changed["reservations"][reviewed["a"].jobs["one"]["reservation"]["key"]]
            data(private.SUPPLEMENTAL, changed)
            expect("missing_own_reservation_rejected", ValueError, "absent", pool_a.storage.revalidate)
            (FAKE / private.SUPPLEMENTAL).write_bytes(supplemental_before)
            extra = {
                "key": "unrelated_future_scope", "output": str(FAKE / "outputs/unrelated"),
                "auxiliary_scopes": [], "reserved_bytes": 123456, "consumer_bindings": consumer_base,
            }
            extra_admission = reserve_admission("meta/extra_admission.json", [extra])
            changed = copy.deepcopy(supplemental)
            changed["reservations"][extra["key"]] = {**extra, "root_admission": extra_admission}
            data(private.SUPPLEMENTAL, changed)
            view = pool_a.storage.revalidate()
            assert view["supplemental_reserved_bytes"] == sum(x["reserved_bytes"] for x in stored.values()) + 123456
            passed("legitimate_unrelated_supplemental_change_reaccounted_dynamically")
            expect("stale_initial_ledger_pin_rejected", ValueError, "first-read", pool_a.storage.preflight)
            (FAKE / private.SUPPLEMENTAL).write_bytes(supplemental_before)
            huge = {**extra, "key": "huge_future_scope", "reserved_bytes": 950000000000}
            huge_admission = reserve_admission("meta/huge_admission.json", [huge])
            changed = copy.deepcopy(supplemental)
            changed["reservations"][huge["key"]] = {**huge, "root_admission": huge_admission}
            data(private.SUPPLEMENTAL, changed)
            expect("dynamic_combined_950GB_ceiling_rejected", ValueError, "ceiling", pool_a.storage.revalidate)
            (FAKE / private.SUPPLEMENTAL).write_bytes(supplemental_before)
            available[0] = 1
            expect("current_free_space_under_owned_scheduler_mutex", ValueError,
                   "free-space", pool_a.storage.revalidate)
            available[0] = 1000000000000
            rows[0].update(used_mib=41, compute=[{
                "pid": FakeChild.children[0].pid, "name": "fake",
                "used_mib": 41,
            }])
            pool_a._process_scan()
            assert "worker_peak_overrun" in pool_a._memory()["blocked_by"]
            passed("original_worker_memory_overrun_detection")
            rows[0].update(used_mib=0, compute=[])
            lease_sha = pool_a.lease_sha
            FakeChild.children[0].returncode = 0
            pool_a.tick()
            assert pool_a.jobs["one"]["state"] == "succeeded"
            assert pool_a.jobs["two"]["state"] == "running"
            assert pool_a.jobs["three"]["state"] == "running"
            assert len(FakeChild.children) == 3 and pool_a.lease_sha == lease_sha
            passed("natural_worker_drain_reuses_same_original_device_lease")
            FakeChild.children[1].returncode = 0
            FakeChild.children[2].returncode = 0
            pool_a.tick()
            assert pool_a.jobs["two"]["state"] == "succeeded"
            assert pool_a.jobs["three"]["state"] == "succeeded"
            for controller in controllers:
                assert controller.shutdown()["lifecycle"] == "closed"
                controller.close()
            controllers.clear()
            repeated = shared.BudgetedController(reviewed["a"], inventory=inventory,
                        proc_reader=lambda *_: {}, available_bytes=lambda: available[0])
            expect("existing_pool_never_adopted", FileExistsError, "File exists", repeated.start)
        assert (FAKE / private.LEGACY).read_bytes() == legacy_before
        assert (FAKE / private.SUPPLEMENTAL).read_bytes() == supplemental_before
        passed("no_duplicate_reservations_no_legacy_or_supplemental_scheduler_write")
        for path, sha in immutable.items():
            assert hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() == sha
        passed("immutable_original_sources_preserved")
        return {
            "kind": "budgeted_shared_pool_v1_bounded_fake_checks",
            "status": "PASS", "checks": CHECKS, "check_count": len(CHECKS),
            "seconds": time.monotonic() - started,
            "fake_registered_pools": 3, "fake_jobs_per_pool": 3,
            "fake_subprocess_objects_total": len(FakeChild.children),
            "actual_subprocess_worker_launches": 0, "actual_GPU_queries_or_jobs": 0,
            "actual_live_ledger_reads_or_writes": 0, "actual_live_locks_acquired": 0,
            "fake_only_flock_socket_and_file_operations": True,
            "original_source_sha256": immutable,
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for controller in controllers:
            for child in FakeChild.children:
                child.returncode = 0
            controller._release_handles()
        for child in FakeChild.children:
            if not child.stdout.closed:
                child.stdout.close()
        shutil.rmtree(FAKE)


if __name__ == "__main__":
    result = main()
    receipt = AUTHOR / "bounded_fake_checks.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

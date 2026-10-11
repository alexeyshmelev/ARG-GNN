"""Exact root-bound supplemental storage companion for future shared GPU pools.

Default CLI reads source/config only. Original Controller lease, inventory,
memory, process supervision, FIFO and subprocess implementations are inherited.
Never adopts an existing pool, worker or device lease. No CUDA/model imports.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import types

from . import gpu_pool_v1 as gpu
from . import project_budget_v4 as budget

ROOT = gpu.ROOT
SOURCE = ROOT / "src/arggnn/budgeted_shared_pool_v3.py"
CONFIG = ROOT / "configs/budgeted_shared_pool_v3.json"
CACHE_MODULE = "arggnn.biological_bank_cache_v1"
PROFILE_MODULE = "arggnn.census_biological_device_profile_v1"
ADDITIONAL_WORKERS = {'arggnn.native_argformer_tsinfer_fit_v5': {'source': {'path': 'src/arggnn/native_argformer_tsinfer_fit_v5.py', 'sha256': '1f05220ddfe7ba2fd3e55b079088eb361c6901323a9520145b0f037f6cfd22e0'}, 'config': {'path': 'configs/native_argformer_tsinfer_fit_v5.json', 'sha256': '6d58f126a9fa16f9dfc04956215e85569f9696cb3b12c06f63fa458d8bb0986f'}, 'shared_launch_allowed': False, 'compatibility_limit': 'sealed_native_v5_requires_existing_exclusive_launcher_contract_shared_route_successor_required'}, 'arggnn.census_biological_fit_v1': {'source': {'path': 'src/arggnn/census_biological_fit_v1.py', 'sha256': '84f7f39c94003256cb22c905f610d5ce4a965d1ca733f483e22367f91dbc90ce'}, 'config': {'path': 'configs/census_biological_fit_v1.json', 'sha256': 'b6f70c6d76421662713028b23deedaef071fb6f0d82a0ef6a9aedaa43f5fcd9e'}, 'shared_launch_allowed': True, 'compatibility_limit': 'eligibility_only_worker_scientific_review_GPU_profile_launcher_watchdog_and_fit_capabilities_required_no_profile_only_CLI'}, 'arggnn.census_biological_fit_v2': {'source': {'path': 'src/arggnn/census_biological_fit_v2.py', 'sha256': 'c21884ad90a2c940b49968353545edc52d345c0c7bc10fd87c492c54d4b38c34'}, 'config': {'path': 'configs/census_biological_fit_v2.json', 'sha256': 'a512816bbab61e80052e7466613d2de4c861b77ef61d0d35cd50fbf81905c621'}, 'shared_launch_allowed': True, 'compatibility_limit': 'eligibility_only_worker_scientific_review_GPU_profile_launcher_watchdog_and_fit_capabilities_required_no_profile_only_CLI'}}
FIXED_POLICY = {
    "kind": "budgeted_shared_pool_v3_policy",
    "root": str(ROOT),
    "legacy_pool_source": {
        "path": "src/arggnn/gpu_pool_v1.py",
        "sha256": "ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1",
    },
    "budget_source": {
        "path": "src/arggnn/project_budget_v4.py",
        "sha256": "a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47",
    },
    "budget_config": {
        "path": "configs/project_budget_v4.json",
        "sha256": "5788f87d11825cf4fbe94a25abe056132107b155848111354e70637334e91d63",
    },
    "human_authorization": budget.FIXED_POLICY["human_authorization"],
    "cache_source": {
        "path": "src/arggnn/biological_bank_cache_v1.py",
        "sha256": "1582c493c54e9b75dfa65c910fd6cec47fa11622ca4ae8bbb30564391a82caec",
    },
    "cache_config": {
        "path": "configs/biological_bank_cache_v1.json",
        "sha256": "24a74c529fc154fe8059c7ec6c3523fc688b443529ccdd00889d89a9509bfb45",
    },
    "allowed_modules": sorted(gpu.MODULES | {CACHE_MODULE, PROFILE_MODULE} | set(ADDITIONAL_WORKERS)),
    "additional_worker_contracts": ADDITIONAL_WORKERS,
    "eligible_physical_gpus": list(gpu.ALLOWED_GPUS),
    "maximum_physical_gpus": 2,
    "scheduling_ceiling_bytes": 950000000000,
    "hard_limit_bytes": 1000000000000,
    "socket_relative_template": ".runtime/gpup/{pool}/control.sock",
    "source_only_by_default": True,
    "existing_pool_adoption_authorized": False,
    "original_controller_methods_overridden": ["_validate_spec", "_lease_check"],
    "memory_components_mib": [
        "model_optimizer_baseline", "cached_bank", "uncached_bank_and_retained_references",
        "autograd", "runtime_allocator"
    ],
}


def digest(value):
    return hashlib.sha256(budget.encoded(value)).hexdigest()


def exact(value, expected, label):
    if budget.encoded(value) != budget.encoded(expected):
        raise ValueError("exact reviewed " + label + " required")


def when(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware root authority dates required")
    return result.astimezone(dt.timezone.utc)


def _fd_has_exclusive_scheduler_lock(fd, identity):
    """Linux fdinfo proves ownership of the original flock open description."""
    for line in Path("/proc/self/fdinfo", str(fd)).read_text().splitlines():
        fields = line.split()
        if (len(fields) == 9 and fields[0] == "lock:" and
                fields[2:5] == ["FLOCK", "ADVISORY", "WRITE"] and
                fields[5] == str(os.getpid()) and fields[7:] == ["0", "EOF"]):
            major, minor, inode = fields[6].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == (
                    os.major(identity.st_dev), os.minor(identity.st_dev), identity.st_ino):
                return True
    return False


@contextlib.contextmanager
def _owned_scheduler_mutex(root):
    """Borrow a duplicate of an already owned EX-flock; never lock/unlock it.

    Used solely by storage.controller inside unchanged acquire_device. Every
    ordinary boundary uses budget.mutex. No globals, stack inspection, monkey-
    patches or original source changes. A duplicate preserves the open-file-
    description lock until original acquire_device closes its own descriptor.
    """
    path = budget.canonical(root, budget.LOCK)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("existing canonical singly linked scheduler mutex required")
    with os.scandir("/proc/self/fd") as entries:
        names = [entry.name for entry in entries]
    if len(names) > 4096:
        raise RuntimeError("bounded self descriptor inventory required")
    borrowed = None
    for name in names:
        if not name.isdecimal():
            continue
        candidate = None
        try:
            candidate = os.dup(int(name))
            current = os.fstat(candidate)
            if ((current.st_dev, current.st_ino) == (before.st_dev, before.st_ino)
                    and _fd_has_exclusive_scheduler_lock(candidate, current)):
                borrowed, candidate = candidate, None
                break
        except (FileNotFoundError, OSError):
            # Descriptor enumeration races are allowed; lack of proof fails below.
            continue
        finally:
            if candidate is not None:
                os.close(candidate)
    if borrowed is None:
        raise RuntimeError("original owned exclusive scheduler mutex not proved")
    try:
        current = os.fstat(borrowed)
        after = path.lstat()
        if ((after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
                or not _fd_has_exclusive_scheduler_lock(borrowed, current)):
            raise RuntimeError("owned scheduler mutex identity/ownership changed")
        yield
        after = path.lstat()
        if ((after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
                or not _fd_has_exclusive_scheduler_lock(borrowed, current)):
            raise RuntimeError("owned scheduler mutex changed during boundary")
    finally:
        os.close(borrowed)  # no LOCK_UN; original owns the descriptor lifetime


def _owned_checker(api):
    """Same reviewed checker code; only its mutex binding is privately composed."""
    namespace = dict(api.check_reservation.__globals__)
    namespace["mutex"] = _owned_scheduler_mutex
    function = types.FunctionType(
        api.check_reservation.__code__, namespace, api.check_reservation.__name__,
        api.check_reservation.__defaults__, api.check_reservation.__closure__)
    function.__kwdefaults__ = dict(api.check_reservation.__kwdefaults__ or {})
    return function


class ReviewedPlan:
    """Internal injected root/metadata seam for bounded contained CPU tests."""
    def __init__(self, root, policy, implementation, config, authority_ref, *, api=budget):
        self.root, self.policy, self.api = Path(root), copy.deepcopy(policy), api
        self.implementation, self.config = copy.deepcopy(implementation), copy.deepcopy(config)
        self.authority_ref = copy.deepcopy(authority_ref)
        self.authority = None
        self.plan = None
        self.jobs = None
        self.validate()

    def validate(self):
        api, root, c = self.api, self.root, self.policy
        api.bound(root, self.implementation)
        exact(api.bound(root, self.config, True), c, "companion config")
        for field in ("legacy_pool_source", "budget_source", "budget_config",
                      "human_authorization", "cache_source", "cache_config"):
            api.bound(root, c[field])
        for module, contract in c["additional_worker_contracts"].items():
            if (module not in c["allowed_modules"] or set(contract)!={"source","config","shared_launch_allowed","compatibility_limit"}
                    or type(contract["shared_launch_allowed"]) is not bool):
                raise ValueError("exact additional worker eligibility/compatibility contract required")
            exact(contract["source"]["path"], "src/"+module.replace(".","/")+".py", "additional worker source path")
            api.bound(root, contract["source"])
            api.bound(root, contract["config"])
        authority = api.bound(root, self.authority_ref, True)
        expected = {
            "kind": "budgeted_shared_pool_v3_root_admission",
            "root_authorizer": "root_agent", "active": True, "root": str(root),
            "implementation": self.implementation, "config": self.config,
            "legacy_pool_source": c["legacy_pool_source"],
            "budget_source": c["budget_source"], "budget_config": c["budget_config"],
            "human_authorization": c["human_authorization"],
            "future_pools_only": True, "no_live_pool_or_worker_patch": True,
            "no_legacy_ledger_mutation": True, "no_duplicate_legacy_reservations": True,
            "scientific_execution_requires_worker_authority": True,
        }
        if set(authority) != set(expected) | {"not_before_utc", "expires_utc", "plan", "independent_review"}:
            raise ValueError("exact root companion admission fields required")
        for field, value in expected.items():
            exact(authority[field], value, "root authority " + field)
        now = dt.datetime.now(dt.timezone.utc)
        if not when(authority["not_before_utc"]) <= now < when(authority["expires_utc"]):
            raise ValueError("root companion authority is outside its active time window")
        plan = authority["plan"]
        if not isinstance(plan, dict) or set(plan) != {"pool", "jobs", "initial_ledger_pins", "runtime_bindings"}:
            raise ValueError("exact reviewed pool/job plan required")
        review = api.bound(root, authority["independent_review"], True)
        review_expected = {
            "kind": "budgeted_shared_pool_v3_independent_review",
            "review_role": "independent_reviewer", "verdict": "PASS",
            "implementation": self.implementation, "config": self.config,
            "legacy_pool_source": c["legacy_pool_source"],
            "budget_source": c["budget_source"], "budget_config": c["budget_config"],
            "human_authorization": c["human_authorization"],
            "reviewed_plan_sha256": digest(plan),
            "actual_live_lock_or_gpu_execution_performed": False,
            "original_sources_unchanged": True,
        }
        if any(budget.encoded(review.get(k)) != budget.encoded(v)
               for k, v in review_expected.items()):
            raise ValueError("fresh independent exact companion/plan PASS required")
        pins = plan["initial_ledger_pins"]
        if not isinstance(pins, dict) or set(pins) != {"legacy_ledger_sha256", "supplemental_ledger_sha256"}:
            raise ValueError("exact first-read ledger pins required")
        for value in pins.values():
            if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
                raise ValueError("pre-registered exact first-read ledger SHA256 required")
        refs = plan["runtime_bindings"]
        if not isinstance(refs, list) or not 1 <= len(refs) <= 64:
            raise ValueError("bounded immutable runtime bindings required")
        for ref in refs:
            if Path(ref.get("path", "")).suffix not in {".py", ".sh", ".json"}:
                raise ValueError("runtime metadata/source bindings required")
            api.bound(root, ref)
        pool = plan["pool"]
        if not isinstance(pool, dict) or set(pool) != {
                "name", "physical_gpu", "headroom_mib", "controller_reservation",
                "storage_snapshot", "python", "start_recipe_sha256"}:
            raise ValueError("exact future pool descriptor required")
        name = pool["name"]
        if not isinstance(name, str) or not gpu.NAME.fullmatch(name) or len(name) > 32:
            raise ValueError("safe future pool name required")
        if type(pool["physical_gpu"]) is not int or pool["physical_gpu"] not in gpu.ALLOWED_GPUS:
            raise ValueError("physical GPU 2-5 required")
        gpu.integer(pool["headroom_mib"], "full device headroom MiB", 1)
        if pool["python"] != str(root / ".runtime/env/bin/python"):
            raise ValueError("unchanged project environment Python required")
        api.bound(root, pool["storage_snapshot"], True)
        snapshot = gpu.snapshot_record(root, pool["storage_snapshot"]["path"], pool["storage_snapshot"]["sha256"])
        if (snapshot["sha256"] != api.FIXED_POLICY["legacy_invariants"]["snapshot_sha256"]
                or snapshot["receipt"]["used_bytes"] != api.FIXED_POLICY["legacy_invariants"]["baseline_bytes"]):
            raise ValueError("unchanged historical legacy storage snapshot required")
        jobs = plan["jobs"]
        if not isinstance(jobs, list) or not 1 <= len(jobs) <= gpu.MAX_JOBS:
            raise ValueError("bounded explicitly reviewed job registration required")
        by_name = {}
        total_logs = 0
        for item in jobs:
            if not isinstance(item, dict) or set(item) != {"spec", "reservation", "module_source", "worker_bindings", "memory_components_mib", "argv_sha256"}:
                raise ValueError("exact reviewed job record required")
            spec = item["spec"]
            fields = {"job", "model", "seed", "module", "args", "output", "peak_mib", "storage_bytes", "policy", "log_limit_bytes"}
            if not isinstance(spec, dict) or set(spec) != fields or len(gpu.encoded(spec)) > gpu.MAX_REQUEST:
                raise ValueError("exact normalized original Controller job spec required")
            if not isinstance(spec["job"], str) or not gpu.NAME.fullmatch(spec["job"]) or spec["job"] in by_name:
                raise ValueError("unique exact logical job required")
            if not isinstance(spec["model"], str) or not 1 <= len(spec["model"]) <= 256:
                raise ValueError("model provenance required")
            if spec["seed"] is not None:
                gpu.integer(spec["seed"], "seed")
            if spec["module"] not in c["allowed_modules"]:
                raise ValueError("finite reviewed worker module allowlist required")
            if (not isinstance(spec["args"], list) or len(spec["args"]) > 1024
                    or any(not isinstance(a, str) or "\0" in a for a in spec["args"])):
                raise ValueError("exact literal worker argv required")
            if spec["policy"] not in {"queue", "reject"}:
                raise ValueError("original queue/reject policy required")
            for field in ("peak_mib", "storage_bytes", "log_limit_bytes"):
                gpu.integer(spec[field], field, 1)
            if spec["log_limit_bytes"] > spec["storage_bytes"]:
                raise ValueError("original job log bound exceeds reservation")
            reservation = item["reservation"]
            api.scope_spec(root, reservation, True)
            if (reservation["key"] != "budgeted_shared_pool_v3/" + name + "/" + spec["job"]
                    or reservation["output"] != spec["output"]
                    or reservation["reserved_bytes"] != spec["storage_bytes"]):
                raise ValueError("exact per-job supplemental key/output/charge required")
            exact(item["module_source"]["path"], "src/" + spec["module"].replace(".", "/") + ".py", "worker module source path")
            api.bound(root, item["module_source"])
            bindings = item["worker_bindings"]
            if not isinstance(bindings, list) or not 1 <= len(bindings) <= 64:
                raise ValueError("explicit worker configuration/capability bindings required")
            for ref in bindings:
                if Path(ref.get("path", "")).suffix not in {".py", ".sh", ".json"}:
                    raise ValueError("worker source/configuration authority binding required")
                api.bound(root, ref)
            if spec["module"] in c["additional_worker_contracts"]:
                contract=c["additional_worker_contracts"][spec["module"]]
                exact(item["module_source"],contract["source"],"pinned additional worker source")
                if contract["config"] not in bindings:
                    raise ValueError("pinned additional worker config binding required")
                if not contract["shared_launch_allowed"]:
                    raise ValueError("native v5 exclusive launcher provenance does not authorize shared launch")
            required_consumers = [self.implementation, self.config, item["module_source"], *bindings]
            for ref in required_consumers:
                if ref not in reservation["consumer_bindings"]:
                    raise ValueError("supplemental reservation omits a reviewed consumer binding")
            memory = item["memory_components_mib"]
            if not isinstance(memory, dict) or set(memory) != set(c["memory_components_mib"]):
                raise ValueError("baseline/cache/autograd/runtime memory decomposition required")
            for field, amount in memory.items():
                gpu.integer(amount, field, 1 if field in {"model_optimizer_baseline", "runtime_allocator"} else 0)
            if sum(memory.values()) != spec["peak_mib"]:
                raise ValueError("total worker peak must include all reviewed memory components")
            if spec["module"] == CACHE_MODULE:
                exact(item["module_source"], c["cache_source"], "cache source")
                cache_config = api.bound(root, c["cache_config"], True)
                cache_bytes = gpu.integer(cache_config["GPU_budget_bytes"], "cache GPU byte cap", 1)
                cache_mib = (cache_bytes + 1024 * 1024 - 1) // (1024 * 1024)
                if (c["cache_config"] not in bindings or memory["cached_bank"] < cache_mib
                        or memory["uncached_bank_and_retained_references"] <= 0 or memory["autograd"] <= 0):
                    raise ValueError("full cache cap plus uncached-bank/autograd/headroom reservations required")
            exact(item["argv_sha256"], digest([pool["python"], "-m", spec["module"], *spec["args"]]), "worker command hash")
            by_name[spec["job"]] = item
            total_logs += spec["log_limit_bytes"]
        controller = pool["controller_reservation"]
        api.scope_spec(root, controller, True)
        if (controller["key"] != "budgeted_shared_pool_v3/" + name + "/controller"
                or controller["output"] != str(root / ".runtime/gpup" / name)
                or controller["auxiliary_scopes"]
                or controller["reserved_bytes"] < gpu.MIN_CONTROLLER_RESERVE + total_logs):
            raise ValueError("exclusive controller directory and bounded log/runtime headroom required")
        for ref in (self.implementation, self.config):
            if ref not in controller["consumer_bindings"]:
                raise ValueError("controller supplemental reservation omits companion binding")
        all_paths = []
        for reservation in [controller, *(item["reservation"] for item in jobs)]:
            paths = api.scope_spec(root, reservation, True)
            if any(api.paths_overlap(a, b) for a in paths for b in all_paths):
                raise ValueError("controller/job registered scopes overlap")
            all_paths.extend(paths)
        recipe = [
            pool["python"], "-m", "arggnn.budgeted_shared_pool_v3", "--start",
            "--admission", self.authority_ref["path"],
            "--expected-admission-sha256", "<ROOT_ADMISSION_SHA256>",
            "--expected-source-sha256", self.implementation["sha256"],
            "--expected-config-sha256", self.config["sha256"],
        ]
        exact(pool["start_recipe_sha256"], digest(recipe), "future start command recipe")
        if self.plan is not None:
            exact(plan, self.plan, "unchanged registered plan")
        self.authority, self.plan, self.jobs = authority, copy.deepcopy(plan), copy.deepcopy(by_name)
        return self

    def reservations(self):
        return [self.plan["pool"]["controller_reservation"],
                *(item["reservation"] for item in self.plan["jobs"])]

    def start_command(self):
        pool = self.plan["pool"]
        return [pool["python"], "-m", "arggnn.budgeted_shared_pool_v3", "--start",
                "--admission", self.authority_ref["path"],
                "--expected-admission-sha256", self.authority_ref["sha256"],
                "--expected-source-sha256", self.implementation["sha256"],
                "--expected-config-sha256", self.config["sha256"]]


class SupplementalStorage:
    """Read-only duck-type; every reservation already exists in budget_v4."""
    def __init__(self, reviewed, snapshot, available):
        self.reviewed, self.snapshot, self.available = reviewed, snapshot, available
        self._inside_owned = _owned_checker(reviewed.api)

    def _check(self, reservation, *, owned=False, boundary=False, initial=None):
        r, c = self.reviewed, self.reviewed.policy
        def checked():
            view = self._inside_owned(
                r.root, reservation,
                expected_source_sha256=c["budget_source"]["sha256"],
                expected_config_sha256=c["budget_config"]["sha256"],
                initial_ledger_pins=initial, boundary=boundary)
            # The original checker validates the complete dynamic ledger.
            # Bind a same-mutex reread to that checked hash, then compare every
            # own registration without repeating global validation per job.
            ledger, sha = r.api.read_json(r.root, r.api.SUPPLEMENTAL)
            if sha != view["supplemental_ledger_sha256"]:
                raise ValueError("supplemental ledger changed under scheduler mutex")
            for own in r.reservations():
                actual = ledger["reservations"].get(own["key"])
                if actual is None or budget.encoded(actual) != budget.encoded(own):
                    raise ValueError("own registered supplemental reservation changed or absent")
            outstanding = view["external_reserve_bytes"] + view["legacy_reserved_bytes"] + view["supplemental_reserved_bytes"]
            free = gpu.integer(self.available(), "current filesystem free bytes")
            if outstanding >= free:
                raise ValueError("combined legacy+supplemental filesystem free-space headroom exceeded")
            return {**view, "reserve_bytes": outstanding, "filesystem_free_bytes": free,
                    "snapshot_sha256": self.snapshot["sha256"],
                    "legacy_or_supplemental_write_performed": False}
        if owned:
            return checked()
        with r.api.mutex(r.root):
            return checked()

    def preflight(self):
        self.reviewed.validate()
        return self._check(self.reviewed.reservations()[0],
                           initial=self.reviewed.plan["initial_ledger_pins"])

    def controller(self, pool, reserve):
        self.reviewed.validate()
        planned = self.reviewed.plan["pool"]
        if (pool != planned["name"] or reserve != planned["controller_reservation"]["reserved_bytes"]):
            raise ValueError("root-bound controller reservation differs")
        # acquire_device owns scheduler.lock here. The private checker borrows
        # only an FD proven by Linux fdinfo to share the original EX-flock.
        return self._check(planned["controller_reservation"], owned=True)

    def job(self, pool, spec):
        self.reviewed.validate()
        planned = self.reviewed.jobs.get(spec.get("job"))
        if planned is None or pool != self.reviewed.plan["pool"]["name"]:
            raise ValueError("unregistered root job")
        exact(spec, planned["spec"], "root job/output/args")
        return self._check(planned["reservation"], boundary=True)

    def revalidate(self):
        self.reviewed.validate()
        return self._check(self.reviewed.reservations()[0])


class BudgetedController(gpu.Controller):
    """Only extends exact spec binding and pre-launch storage/source guards."""
    def __init__(self, reviewed, *, inventory=gpu.real_inventory, proc_reader=None, available_bytes=None):
        self.reviewed = reviewed
        p = reviewed.plan["pool"]
        c = reviewed.policy
        invariant = reviewed.api.FIXED_POLICY["legacy_invariants"]
        super().__init__(
            root=reviewed.root, pool=p["name"], gpu=p["physical_gpu"],
            headroom_mib=p["headroom_mib"],
            storage_snapshot=p["storage_snapshot"]["path"],
            storage_snapshot_sha256=p["storage_snapshot"]["sha256"],
            external_storage_reserve_bytes=invariant["external_reserve_bytes"],
            controller_storage_reserve_bytes=p["controller_reservation"]["reserved_bytes"],
            storage_limit_bytes=invariant["limit_bytes"],
            inventory=inventory, proc_reader=proc_reader,
            available_bytes=available_bytes, python=p["python"],
            modules={item["spec"]["module"] for item in reviewed.plan["jobs"]},
            socket_relative=c["socket_relative_template"].format(pool=p["name"]))
        def free_bytes():
            result = os.statvfs(reviewed.root)
            return result.f_bavail * result.f_frsize
        self.storage = SupplementalStorage(reviewed, self.storage.snapshot, available_bytes or free_bytes)

    def _validate_spec(self, value):
        self.reviewed.validate()
        item = self.reviewed.jobs.get(value.get("job")) if isinstance(value, dict) else None
        if item is None:
            raise ValueError("unregistered root logical job")
        exact(value, item["spec"], "root job/output/args")
        return super()._validate_spec(value)

    def _lease_check(self):
        super()._lease_check()
        self.storage_account = self.storage.revalidate()


def load_authority(admission, admission_sha, source_sha, config_sha):
    config, actual = budget.read_json(ROOT, CONFIG)
    exact(config, FIXED_POLICY, "fixed production policy")
    if actual != config_sha:
        raise ValueError("reviewed companion config hash differs")
    implementation = {"path": str(SOURCE.relative_to(ROOT)), "sha256": source_sha}
    config_ref = {"path": str(CONFIG.relative_to(ROOT)), "sha256": config_sha}
    return ReviewedPlan(ROOT, config, implementation, config_ref,
                        {"path": admission, "sha256": admission_sha})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate-admission", action="store_true")
    action.add_argument("--start", action="store_true")
    action.add_argument("--submit-job")
    action.add_argument("--status", action="store_true")
    action.add_argument("--shutdown", action="store_true")
    parser.add_argument("--admission")
    parser.add_argument("--expected-admission-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args(argv)
    if not any((args.validate_admission, args.start, args.submit_job, args.status, args.shutdown)):
        print(json.dumps({
            "kind": "budgeted_shared_pool_v3_source_report",
            "implementation": budget.reference(ROOT, SOURCE),
            "config": budget.reference(ROOT, CONFIG),
            "legacy_pool_source": FIXED_POLICY["legacy_pool_source"],
            "budget_source": FIXED_POLICY["budget_source"],
            "live_ledger_device_scope_or_lock_access": False,
            "actual_start_submit_or_gpu_execution_performed": False,
        }, indent=2))
        return 0
    if not all((args.admission, args.expected_admission_sha256,
                args.expected_source_sha256, args.expected_config_sha256)):
        raise ValueError("exact reviewed source/config/root authority opt-in required")
    reviewed = load_authority(args.admission, args.expected_admission_sha256,
                              args.expected_source_sha256, args.expected_config_sha256)
    if args.validate_admission:
        print(json.dumps({
            "kind": "budgeted_shared_pool_v3_authority_source_report",
            "admission": reviewed.authority_ref, "reviewed_plan_sha256": digest(reviewed.plan),
            "start_command": reviewed.start_command(),
            "start_command_sha256": digest(reviewed.start_command()),
            "live_ledger_device_scope_or_lock_access": False,
            "actual_start_submit_or_gpu_execution_performed": False,
        }, indent=2))
        return 0
    if args.start:
        controller = BudgetedController(reviewed)
        controller.storage.preflight()
        controller.start()  # unchanged original start/acquire_device
        def stop(_signum, _frame):
            controller.shutdown_requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(json.dumps(controller.status(), sort_keys=True), flush=True)
        controller.run()
        return 0
    path = ROOT / FIXED_POLICY["socket_relative_template"].format(pool=reviewed.plan["pool"]["name"])
    if args.submit_job:
        item = reviewed.jobs.get(args.submit_job)
        if item is None:
            raise ValueError("only an exact root-registered job may be submitted")
        request = {"v": 1, "op": "submit", "spec": item["spec"]}
    elif args.shutdown:
        request = {"v": 1, "op": "shutdown", "terminate": False}
    else:
        request = {"v": 1, "op": "status"}
    print(json.dumps(gpu.request(path, request), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

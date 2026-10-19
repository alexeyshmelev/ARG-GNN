"""Repair continuation using an isolated copy of the frozen v4 runtime.

Only this new process receives the repair approval, predecessor and reporting
bindings. Existing v3/v4 files and running controller namespaces stay untouched.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

from . import contracts as c

MODULE = "arggnn.benchmarking.workflow_v8"
_spec = importlib.util.spec_from_file_location(
    "arggnn.benchmarking._workflow_v8_core",
    Path(__file__).with_name("workflow_v4.py"))
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
from research.reviews.benchmark_repairs_20261019.supervision.finite_pss_v1 import repaired_owned_group_class
_core.OwnedGroup = repaired_owned_group_class(_core)
from research.reviews.workflow_progress_diagnosis_20261018.scheduler_v6.guard_order_v1 import compile_scheduler_run
_core._BASE_RUN = compile_scheduler_run(_core._BASE_RUN)
base = _core.base
need = _core.need


def current_policy():
    canonical = _core._BASE_POLICY()
    approved = canonical.get("approved_cohort_continuation_ref")
    need(isinstance(approved, dict) and set(approved) == {"path", "sha256"},
         "Exact canonical approved_cohort_continuation_ref required for v8")
    policy = dict(canonical)
    policy["canonical_parent_workflow_ref"] = canonical.get("approved_workflow_ref")
    policy["canonical_continuation_workflow_ref"] = canonical.get("approved_continuation_ref")
    policy["approved_workflow_ref"] = approved
    policy.pop("workflow_drain_ref", None)
    if "cohort_continuation_drain_ref" in canonical:
        policy["workflow_drain_ref"] = canonical["cohort_continuation_drain_ref"]
    return policy


def external_result(task):
    if "workflow_predecessor" in task:
        from . import workflow_predecessor_v5 as predecessor
        return predecessor.external_result(task)
    return _core._BASE_EXTERNAL_RESULT(task)


def external_resources(plan):
    cpu, rss = _core._BASE_EXTERNAL_RESOURCES(plan)
    if plan.get("workflow_external_predecessors"):
        from . import workflow_predecessor_v5 as predecessor
        extra_cpu, extra_rss = predecessor.external_resources(plan)
        cpu += extra_cpu
        rss += extra_rss
    return cpu, rss + plan.get("prior_unknown_rss_bytes", 0)


def storage_projection(plan, state, new_cap=0):
    total = _core._BASE_STORAGE(plan, state, new_cap)
    if plan.get("workflow_external_predecessors"):
        from . import workflow_predecessor_v5 as predecessor
        total += predecessor.storage_reservation_bytes(plan)
    return total


def _known_credit(plan, state_path):
    # Incomplete finite PSS cohorts retain the full RAM reservation.
    bindings = []
    if plan.get("workflow_external_predecessors"):
        from . import workflow_predecessor_v5 as predecessor
        bindings.extend(predecessor.live_group_bindings(plan))
    if state_path.exists():
        state = c.read_json(state_path)
        for task in plan["tasks"]:
            entry = state["tasks"][task["id"]]
            if task["resource"] == "external" or entry["status"] != "running":
                continue
            path = c.contained_path(task["output_directory"]) / "supervision_identity.json"
            if not path.exists():
                continue
            receipt = c.read_json(path)
            need(receipt.get("job_ref") == entry.get("job_ref"),
                 "Headroom identity must match actual current job")
            bindings.append(dict(receipt["child_identity"], max_rss_bytes=task["max_rss_bytes"]))
    credit = 0
    seen = set()
    for binding in bindings:
        try:
            pid = binding["pid"]
            need(pid not in seen, "Never credit one live group twice")
            seen.add(pid)
            root = _core.identity(pid)
            need(_core.same(root, binding) and root["state"] != "Z", "Live headroom identity changed")
            tracker = _core.OwnedGroup(pid)
            need(_core.same(tracker.root, binding), "Live leader changed before PSS admission sample")
            tracker.measure()
            known = tracker.latest["headroom_credit_bytes"]
            credit += min(binding["max_rss_bytes"], known)
        except (OSError, RuntimeError, KeyError, ValueError):
            continue
    return credit


def run(ref, **kwargs):
    install_bindings()
    plan = c.read_ref(ref)
    if kwargs.get("execute_jobs", False):
        from .workflow_reporting_v8 import ProgressReporter
        base.ProgressReporter = ProgressReporter
    _core._RUN_CONTEXT = (plan, c.contained_path(plan["records_directory"]) / "state.json")
    return _core._BASE_RUN(ref, **kwargs)


# The private core retains all v4 supervision, startup barrier and shared GPU
# lease checks. Its own globals resolve these repair-only callbacks at execution.
_core.MODULE = MODULE
_core.current_policy = current_policy
_core.external_result = external_result
_core.external_resources = external_resources
_core.storage_projection = storage_projection
_core._known_credit = _known_credit
_core.run = run
execute = _core.execute
child = _core.child
group_pss = _core.group_pss
scope_bytes = _core.scope_bytes
available_ram_bytes = _core.available_ram_bytes
await_startup = _core.await_startup
OwnedGroup = _core.OwnedGroup


def install_bindings():
    _core.install_bindings()


def main():
    await_startup()
    install_bindings()
    base.main()


if __name__ == "__main__":
    main()

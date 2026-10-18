"""Freeze a drained-v2 operational continuation; never release or launch work.

Completed output/ref bindings remain intact. Only unfinished output namespaces,
European metadata read caps, ROOT dispatch holds and current source pins change.
All scientific datasets, dependencies and numerical settings are inherited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from arggnn.benchmarking import contracts as c

PRIOR_PLAN = "configs/benchmarks/executable_workflow_20261018_v2.json"
PRIOR_STATE = ".runtime/benchmarking/workflow_20261018_v2/state.json"
PLAN = "configs/benchmarks/executable_workflow_20261018_v3.json"
RECORDS = ".runtime/benchmarking/workflow_20261018_v3"
PRIOR_OUTPUT = "experiments/results/benchmark_campaign_20261018_workflow/release_v2"
OUTPUT = "experiments/results/benchmark_campaign_20261018_workflow/release_v3"
SELF = "scripts/build_benchmark_workflow_v3.py"
METADATA_CAP = 512 * 2**20
HOLDS = ("AM_treeloom_fit", "AP__all_train__treeloom_fit", "C4__all_train__c4_treeloom_fit")
HOLD_REASON = "TreeLoom preprocessing repair awaits validation; native comparator work may proceed"
FIRST_GPU = "AM_argformer_american_validation_000_argformer_mlm"
PATCHED_NATIVE_SHA = "149fad77bfc54fd14bd1a7ef906fff1d97e4b2f98df25b6c1913c1f99732011a"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    target = c.contained_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return c.file_ref(target)


def _ref_object(value):
    return isinstance(value, dict) and set(value) == {"path", "sha256"}


def _wait(ref, *, successful):
    wait = c.read_ref(ref)
    need(wait.get("actual_direct_wait") is True
         and type(wait.get("actual_direct_returncode")) is int
         and wait.get("owned_process_group_absent") is True,
         "Every actual attempt requires its genuine direct wait and absent owned group")
    if successful:
        need(wait["actual_direct_returncode"] == 0 and wait.get("supervisor_error") is None,
             "Adoption requires genuinely successful scientific completion")
    return wait


def _job(task, entry, prior_ref):
    need(_ref_object(entry.get("job_ref")), "Attempt must identify its exact frozen job")
    job = c.read_ref(entry["job_ref"])
    need(job.get("plan_ref") == prior_ref and job.get("task") == task
         and job.get("task_fingerprint") == c.fingerprint(task),
         "Attempt job must match the exact original v2 plan/task")


def _completion_metadata(value):
    # Authenticate only bounded completion JSON. Never follow payload refs inside it.
    if _ref_object(value):
        need(Path(value["path"]).suffix == ".json", "Completion reference must be bounded JSON metadata")
        c.read_ref(value)
    else:
        need(isinstance(value, dict), "Scientific completion must return metadata or a metadata FileRef")


def verify_success(task, entry, prior_ref):
    need(_ref_object(entry.get("result_ref")), "Succeeded task needs its exact result envelope")
    envelope = c.read_ref(entry["result_ref"])
    need("value" in envelope, "Succeeded result envelope must contain its actual return value")
    _completion_metadata(envelope["value"])
    if task["resource"] == "external":
        need(entry.get("external") is True and envelope.get("status") == "succeeded",
             "Exact externally completed input adoption required")
        wait_ref = envelope.get("external_actual_wait_ref")
        need(wait_ref == c.file_ref(task["actual_wait_path"]), "Original named external wait required")
        _wait(wait_ref, successful=True)
        queue_ref = c.file_ref(task["queue_path"])
        queue = c.read_ref(queue_ref)
        matches = [job for job in queue["jobs"] if job["admission"]["job_id"] == task["job_id"]]
        need(len(matches) == 1 and matches[0]["status"] == "succeeded"
             and matches[0].get("actual_returncode") == 0, "Original named external controller must succeed")
        return dict(result_ref=entry["result_ref"], actual_wait_ref=wait_ref, queue_ref=queue_ref)
    _job(task, entry, prior_ref)
    need(entry.get("actual_controller_wait") == 0 and entry.get("owned_process_group_absent") is True,
         "Succeeded controller must record actual wait0 and group absence")
    need(envelope.get("job_ref") == entry["job_ref"], "Result must belong to the exact succeeded job")
    output = c.contained_path(task["output_directory"])
    need(c.contained_path(entry["result_ref"]["path"]) == output / "result.json",
         "Succeeded envelope must remain in its original output scope")
    wait_ref = c.file_ref(output / "actual_wait.json")
    wait = _wait(wait_ref, successful=True)
    need(wait.get("job_ref") == entry["job_ref"], "Actual wait must bind the same producer job")
    need(type(entry.get("reservation_bytes")) is int and entry["reservation_bytes"] > 0
         and type(entry.get("retained_bytes")) is int and 0 <= entry["retained_bytes"] <= entry["reservation_bytes"],
         "Preserve the succeeded task's actual retained-byte accounting")
    return dict(result_ref=entry["result_ref"], job_ref=entry["job_ref"], actual_wait_ref=wait_ref)


def _map_paths(value, replacements):
    if isinstance(value, dict):
        return {key: _map_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_paths(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            for before, after in ((str(old), str(new)),
                                  (str(old.relative_to(c.ROOT)), str(new.relative_to(c.ROOT)))):
                if value == before or value.startswith(before + "/"):
                    return after + value[len(before):]
    return value


def _metadata_cap(value):
    if isinstance(value, dict):
        return {key: METADATA_CAP if key == "max_metadata_bytes" else _metadata_cap(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata_cap(item) for item in value]
    return value


def prepare_continuation(prior_ref, state_ref, *, records=RECORDS,
                         prior_output=PRIOR_OUTPUT, output=OUTPUT):
    """Read-only preparation. No factory, payload hash, source refresh or launch."""
    prior, state = c.read_ref(prior_ref), c.read_ref(state_ref)
    need(prior.get("schema") == "arggnn_executable_workflow_v1" and prior.get("physical_gpu") == 0,
         "Exact one-GPU executable predecessor required")
    need(state.get("plan_ref") == prior_ref, "Frozen state must identify the exact v2 plan")
    tasks = prior["tasks"]
    by_id = {task["id"]: task for task in tasks}
    need(len(by_id) == len(tasks) and set(by_id) == set(state["tasks"]), "Exact predecessor task set required")
    statuses = {key: entry["status"] for key, entry in state["tasks"].items()}
    need(set(statuses.values()) <= {"succeeded", "failed", "blocked", "pending"},
         "V2 must have no running/unknown task; actual reconciliation comes first")
    need(set(HOLDS) <= set(by_id) and FIRST_GPU in by_id, "Expected TreeLoom holds and native MLM task required")
    old_root, new_root = c.contained_path(prior_output), c.contained_path(output)
    need(old_root != new_root and not new_root.exists(), "Entire v3 release root must be fresh")
    need(not old_root.is_relative_to(new_root) and not new_root.is_relative_to(old_root),
         "New release must not contain or nest inside old scopes")
    adopted, failed, replacements = {}, [], []
    for task in tasks:
        identity, status = task["id"], statuses[task["id"]]
        entry = state["tasks"][identity]
        if status == "succeeded":
            need(all(statuses[dep] == "succeeded" for dep in task["dependencies"]),
                 "Succeeded task cannot depend on an unfinished predecessor")
            evidence = verify_success(task, entry, prior_ref)
            adopted[identity] = dict(evidence, output_directory=task.get("output_directory"))
            continue
        need(task["resource"] != "external", "All original external inputs must already genuinely succeed")
        old_scope = c.contained_path(task["output_directory"])
        need(old_scope.is_relative_to(old_root), "Unfinished scope must belong to original release_v2")
        new_scope = new_root / old_scope.relative_to(old_root)
        need(not new_scope.exists(), "Never reuse a v3 scientific output scope")
        replacements.append((old_scope, new_scope))
        attempted = any(key in entry for key in ("job_ref", "pid", "started_at", "actual_controller_wait", "reservation_bytes"))
        if attempted or old_scope.exists():
            need(status == "failed" and type(entry.get("actual_controller_wait")) is int
                 and entry.get("owned_process_group_absent") is True,
                 "Every previous attempt must be failed, actually waited and drained")
            _job(task, entry, prior_ref)
            wait_ref = c.file_ref(old_scope / "actual_wait.json")
            wait = _wait(wait_ref, successful=False)
            need(wait.get("job_ref") == entry["job_ref"], "Failed wait must bind the original job")
            reservation = entry.get("reservation_bytes")
            need(type(reservation) is int and reservation > 0, "Original failed reservation must remain charged")
            charge = entry.get("retained_bytes", reservation)
            need(type(charge) is int and 0 < charge <= reservation
                 and type(wait.get("output_bytes")) is int and wait["output_bytes"] <= charge,
                 "Recorded retained charge or original full reservation must cover failed output")
            failed.append(dict(task_id=identity, scope=str(old_scope.relative_to(c.ROOT)),
                charged_bytes=charge, original_reservation_bytes=reservation,
                charge_basis="recorded_retained_bytes" if "retained_bytes" in entry else "full_original_reservation_no_new_credit",
                original_entry=copy.deepcopy(entry), actual_wait_ref=wait_ref))
        else:
            need(not any(key in entry for key in ("result_ref", "retained_bytes")),
                 "Unattempted task may not claim completion or retained bytes")
    new_plan = copy.deepcopy(prior)
    new_entries = {}
    for task in new_plan["tasks"]:
        identity = task["id"]
        if statuses[identity] == "succeeded":
            new_entries[identity] = copy.deepcopy(state["tasks"][identity])
            new_entries[identity]["adopted_from_v2"] = dict(plan_ref=prior_ref, state_ref=state_ref,
                                                         **adopted[identity])
        else:
            task["output_directory"] = _map_paths(task["output_directory"], replacements)
            task["factory_arguments"] = _map_paths(task.get("factory_arguments", {}), replacements)
            new_entries[identity] = dict(status="pending", external=False, prior_status=statuses[identity],
                                         prior_state_ref=state_ref)
            if identity.startswith(("AP__", "C4__")):
                task["factory_arguments"] = _metadata_cap(task.get("factory_arguments", {}))
    # Stable priority changes only dispatch order, never graph or datasets.
    def priority(task):
        if task["id"] == FIRST_GPU:
            return 0
        if task.get("factory_arguments", {}).get("stage") == "mlm" and task["resource"] == "gpu0":
            return 1
        return 2
    new_plan["tasks"].sort(key=priority)
    new_plan.update(records_directory=records, operational_successor_of=prior_ref,
        adopted_prior_state_ref=state_ref, dispatch_hold_task_ids=list(HOLDS), dispatch_hold_reason=HOLD_REASON,
        task_family_dispatch_order="native MLM first, then inherited stable task order",
        continuation_metadata_policy="European max_metadata_bytes=512MiB; operational read cap only",
        metadata_factory_smoke_scope="Inherited v2 factory smoke; v3 verifies bounded metadata/waits only, no factories called")
    # No state-scoped reservations for reset work: its retained failures are
    # separately charged below. Adopted successes keep their old reservations.
    ready_gpu = [task["id"] for task in new_plan["tasks"] if task["resource"] == "gpu0"
                 and new_entries[task["id"]]["status"] == "pending" and task["id"] not in HOLDS
                 and all(new_entries[dep]["status"] == "succeeded" for dep in task["dependencies"])]
    need(ready_gpu and ready_gpu[0] == FIRST_GPU, "Native MLM must be the first ready unheld GPU task")
    evidence = dict(schema="arggnn_v3_preserved_v2_attempts_v1", prior_plan_ref=prior_ref,
        prior_state_ref=state_ref, adopted_succeeded=len(adopted), adopted=adopted,
        retained_failed_attempts=failed, total_failed_charge_bytes=sum(row["charged_bytes"] for row in failed),
        first_ready_gpu_task_id=ready_gpu[0], all_original_scientific_payloads_unmodified=True,
        scientific_functions_called=0, scientific_payloads_rehashed=0)
    need(c.file_ref(state_ref["path"]) == state_ref and c.file_ref(prior_ref["path"]) == prior_ref,
         "Predecessor plan/state changed during preparation")
    return new_plan, new_entries, evidence


def build():
    """Create exclusive frozen metadata only; ROOT must separately release it."""
    for path in (PLAN, RECORDS + "/state.json", RECORDS + "/initial_state.json", RECORDS + "/freeze_result.json",
                 RECORDS + "/preserved_v2_attempts.json"):
        need(not c.contained_path(path).exists(), "Never overwrite an existing v3 plan/state/freeze result")
    need(c.file_ref("src/arggnn/benchmarking/american_competitor_worker.py")["sha256"] == PATCHED_NATIVE_SHA,
         "ROOT must first apply the exact reviewed DataSpec operational patch")
    prior_ref, state_ref = c.file_ref(PRIOR_PLAN), c.file_ref(PRIOR_STATE)
    plan, entries, evidence = prepare_continuation(prior_ref, state_ref)
    evidence_ref = write_new(RECORDS + "/preserved_v2_attempts.json", evidence)
    if evidence["retained_failed_attempts"]:
        plan["external_storage"].append(dict(
            cap_bytes=evidence["total_failed_charge_bytes"],
            scopes=[row["scope"] for row in evidence["retained_failed_attempts"]],
            completion_evidence=[], evidence_ref=evidence_ref,
            retention="Permanent preserved v2 failed/interrupted attempts, no deletion or new credit"))
    # Deliberately refreshed only now, after ROOT's native/scheduler source fixes.
    source_paths = {ref["path"] for ref in plan["sources"]} | {SELF}
    plan["sources"] = [c.file_ref(path) for path in sorted(source_paths)]
    plan["continuation_evidence_ref"] = evidence_ref
    plan_ref = write_new(PLAN, plan)
    from arggnn.benchmarking import workflow
    need(callable(getattr(workflow, "dispatch_block_reason", None)), "ROOT dispatch hold/drain scheduler is required")
    workflow.load_plan(plan_ref)
    initial = dict(plan_ref=plan_ref, tasks=entries, complete=False, run_status="prepared_not_released",
        prior_plan_ref=prior_ref, prior_state_ref=state_ref,
        updated_utc=datetime.now(timezone.utc).isoformat(), waiting_reasons={}, undrained_failed_jobs=[])
    initial_ref = write_new(RECORDS + "/initial_state.json", initial)
    state_ref_new = write_new(RECORDS + "/state.json", initial)
    result = dict(schema="arggnn_v3_frozen_continuation_v1", plan_ref=plan_ref,
        initial_state_ref=initial_ref, state_ref=state_ref_new, continuation_evidence_ref=evidence_ref,
        predecessor_plan_ref=prior_ref, predecessor_state_ref=state_ref,
        task_counts=dict(Counter(entry["status"] for entry in entries.values())),
        retained_failed_charge_bytes=evidence["total_failed_charge_bytes"],
        dispatch_hold_task_ids=list(HOLDS), first_ready_gpu_task_id=evidence["first_ready_gpu_task_id"],
        scientific_functions_called=0, scientific_payloads_rehashed=0, execution_released=False,
        original_v2_plan_and_state_unmodified=True)
    freeze_ref = write_new(RECORDS + "/freeze_result.json", result)
    print(json.dumps(dict(result, freeze_result_ref=freeze_ref), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Write a new plan and initial state only")
    args = parser.parse_args()
    if args.freeze:
        build()
    else:
        parser.error("Explicit --freeze is required; this script never releases or launches work")

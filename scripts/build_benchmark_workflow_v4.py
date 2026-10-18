"""Freeze fresh v4 work while the exact drained v3 Relate workers continue."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import workflow_predecessor_v4 as predecessor

PRIOR_PLAN = "configs/benchmarks/executable_workflow_20261018_v3.json"
PRIOR_STATE = ".runtime/benchmarking/workflow_20261018_v3/state.json"
PLAN = "configs/benchmarks/executable_workflow_20261018_v4.json"
RECORDS = ".runtime/benchmarking/workflow_20261018_v4"
PRIOR_OUTPUT = "experiments/results/benchmark_campaign_20261018_workflow/release_v3"
OUTPUT = "experiments/results/benchmark_campaign_20261018_workflow/release_v4"
OBSERVATION = "research/reviews/workflow_progress_diagnosis_20261018/argformer_ram/root_fresh_namespace_observation_v1.json"
OBSERVATION_SHA = "312666df1597b51714bc8cb61f07a3e848ff4598bea8b646b0d29f4fa7d5ad00"
SELF = "scripts/build_benchmark_workflow_v4.py"
NEW_SOURCES = (SELF, "src/arggnn/benchmarking/workflow_v4.py",
               "src/arggnn/benchmarking/workflow_predecessor_v4.py",
               "src/arggnn/benchmarking/workflow_reporting_v4.py",
               "src/arggnn/benchmarking/treeloom_preparation_v4.py") + tuple(
    "research/reviews/workflow_progress_diagnosis_20261018/treeloom/" + name for name in (
        "indices_v1.py", "candidate_v1.py", "cache_v1.py", "resident_pool_v2.py",
        "prepare_integration_v2.py", "qualify_parent_v2.py", "storage_bound_v2.json"))
LIVE_IDS = ("AP__ancestralpaths_european_train_000__ap_infer",
            "C4__argmix_release_ap200mb_validation_000__validation_infer",
            "AP__ancestralpaths_european_train_001__ap_infer")
FIRST_GPU = "AM_argformer_american_validation_000_argformer_mlm"
QUALIFICATION = "AM_treeloom_full_parent_qualification"
TRELOOM_MODULE = "arggnn.benchmarking.treeloom_preparation_v4"

def need(ok, message):
    if not ok:
        raise ValueError(message)

def write_new(path, value):
    path = c.contained_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return c.file_ref(path)

def _job(task, entry, prior_ref, *, allow_adopted=False):
    job = c.read_ref(entry["job_ref"])
    expected_plan_ref = prior_ref
    if allow_adopted and job.get("plan_ref") != prior_ref:
        prior = c.read_ref(prior_ref)
        evidence = c.read_ref(prior["continuation_evidence_ref"])
        adopted = evidence["adopted"][task["id"]]
        need(adopted.get("job_ref") == entry["job_ref"]
             and adopted.get("result_ref") == entry["result_ref"]
             and adopted.get("actual_wait_ref") == c.file_ref(
                 c.contained_path(task["output_directory"]) / "actual_wait.json"),
             "Inherited success must match immutable v3 adoption evidence")
        original = c.read_ref(job["plan_ref"])
        need(next(t for t in original["tasks"] if t["id"] == task["id"]) == task,
             "Inherited original descriptor must remain exact")
        expected_plan_ref = job["plan_ref"]
    need(job.get("plan_ref") == expected_plan_ref and job.get("task") == task
         and job.get("task_fingerprint") == c.fingerprint(task),
         "Exact original v3 attempt required")
    return job

def _wait(task, entry):
    ref = c.file_ref(c.contained_path(task["output_directory"]) / "actual_wait.json")
    wait = c.read_ref(ref)
    need(wait.get("job_ref") == entry["job_ref"] and wait.get("actual_direct_wait") is True
         and type(wait.get("actual_direct_returncode")) is int,
         "Preserve the original genuine wait without inventing group absence")
    return ref, wait

def verify_success(task, entry, prior_ref):
    envelope = c.read_ref(entry["result_ref"])
    need("value" in envelope and isinstance(envelope["value"], dict),
         "Exact original small result envelope required")
    # Do not follow payload FileRefs. A native inference payload may exceed 32 MiB.
    if task["resource"] == "external":
        need(entry.get("external") is True and envelope.get("status") == "succeeded",
             "Exact original external completion required")
        wait_ref = c.file_ref(task["actual_wait_path"])
        need(envelope.get("external_actual_wait_ref") == wait_ref,
             "Original external wait binding required")
        wait = c.read_ref(wait_ref)
        queue = c.read_json(task["queue_path"])
        jobs = [j for j in queue["jobs"] if j["admission"]["job_id"] == task["job_id"]]
        need(len(jobs) == 1 and jobs[0]["status"] == "succeeded"
             and jobs[0].get("actual_returncode") == 0, "Original external controller must succeed")
    else:
        _job(task, entry, prior_ref, allow_adopted=True)
        need(entry.get("actual_controller_wait") == 0
             and entry.get("owned_process_group_absent") is True
             and envelope.get("job_ref") == entry["job_ref"]
             and c.contained_path(entry["result_ref"]["path"]) ==
             c.contained_path(task["output_directory"]) / "result.json",
             "Exact genuinely successful controller and result required")
        wait_ref, wait = _wait(task, entry)
        need(type(entry.get("reservation_bytes")) is int
             and type(entry.get("retained_bytes")) is int
             and 0 <= entry["retained_bytes"] <= entry["reservation_bytes"],
             "Preserve successful task storage accounting")
    need(wait.get("actual_direct_wait") is True and wait.get("actual_direct_returncode") == 0
         and wait.get("owned_process_group_absent") is True and wait.get("supervisor_error") is None,
         "Original successful direct wait and group absence required")
    return dict(result_ref=entry["result_ref"], actual_wait_ref=wait_ref)

def live_binding(task, entry, prior_ref, snapshot_ref, state_path):
    _job(task, entry, prior_ref)
    supervisor = predecessor.proc_identity(entry["pid"])
    need(supervisor is not None and supervisor["state"] != "Z",
         "Named original supervisor must still be live")
    children_path = Path("/proc") / str(supervisor["pid"]) / "task" / str(supervisor["pid"]) / "children"
    children = [predecessor.proc_identity(int(v)) for v in children_path.read_text().split()]
    leaders = [v for v in children if v is not None and v["ppid"] == supervisor["pid"]
               and v["pid"] == v["pgid"] == v["session"] and v["state"] != "Z"]
    need(len(leaders) == 1, "Exactly one genuine original scientific group leader required")
    group = leaders[0]
    need(entry.get("reservation_bytes") == task["max_output_bytes"],
         "Original live output reservation must remain unchanged")
    return dict(source_plan_ref=prior_ref, source_state_snapshot_ref=snapshot_ref,
        source_state_path=state_path, source_task_id=task["id"],
        source_task_fingerprint=c.fingerprint(task), original_job_ref=entry["job_ref"],
        original_supervisor_pid=supervisor["pid"],
        original_supervisor_start_ticks=supervisor["start_ticks"],
        owned_process_group_id=group["pid"], owned_group_leader_start_ticks=group["start_ticks"],
        owned_group_leader_session=group["session"], output_directory=task["output_directory"],
        max_output_bytes=task["max_output_bytes"], max_rss_bytes=task["max_rss_bytes"])

def _map_paths(value, replacements):
    if isinstance(value, dict):
        return {k: _map_paths(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_paths(v, replacements) for v in value]
    if isinstance(value, str):
        for old, new in replacements:
            for before, after in ((str(old), str(new)),
                    (str(old.relative_to(c.ROOT)), str(new.relative_to(c.ROOT)))):
                if value == before or value.startswith(before + "/"):
                    return after + value[len(before):]
    return value

def prepare_continuation(prior_ref, snapshot_ref, observation_ref, *,
                         records=RECORDS, prior_output=PRIOR_OUTPUT, output=OUTPUT,
                         state_path=PRIOR_STATE, expected_successes=139, expected_live_ids=LIVE_IDS,
                         include_qualification=False):
    """Bounded read-only metadata preparation. No factories, payload hashes or launch."""
    prior, state, observation = c.read_ref(prior_ref), c.read_ref(snapshot_ref), c.read_ref(observation_ref)
    need(prior.get("schema") == "arggnn_executable_workflow_v1" and prior.get("physical_gpu") == 0,
         "Exact one-GPU v3 predecessor required")
    need(state["plan_ref"] == prior_ref, "Immutable source state must identify original plan")
    tasks = prior["tasks"]
    by_id = {t["id"]: t for t in tasks}
    need(len(by_id) == len(tasks) and set(by_id) == set(state["tasks"]),
         "Exact original task set required")
    statuses = {k: e["status"] for k, e in state["tasks"].items()}
    need(set(statuses.values()) <= {"succeeded", "running", "failed", "blocked", "pending"},
         "Do not reinterpret unknown task states")
    live_ids = {k for k, s in statuses.items() if s == "running"}
    need(live_ids <= set(expected_live_ids)
         and sum(s == "succeeded" and k not in set(expected_live_ids)
                 for k, s in statuses.items()) == expected_successes,
         "Preserve original completed stages and only genuine remaining live predecessors")
    old_root, new_root = c.contained_path(prior_output), c.contained_path(output)
    need(not new_root.exists() and old_root != new_root
         and not new_root.is_relative_to(old_root) and not old_root.is_relative_to(new_root),
         "Entire v4 release namespace must be fresh and disjoint")
    adopted, live, failed, replacements = {}, [], [], []
    for task in tasks:
        key = task["id"]
        entry = state["tasks"][key]
        if entry["status"] == "succeeded":
            need(all(statuses[d] == "succeeded" for d in task["dependencies"]),
                 "Succeeded task cannot have an unfinished predecessor")
            adopted[key] = verify_success(task, entry, prior_ref)
            continue
        need(task["resource"] != "external", "Original external inputs must already have succeeded")
        if key in live_ids:
            need(task["resource"] == "cpu", "Only exact existing CPU jobs may continue externally")
            live.append(live_binding(task, entry, prior_ref, snapshot_ref, state_path))
            continue
        old_scope = c.contained_path(task["output_directory"])
        need(old_scope.is_relative_to(old_root), "Unfinished stage must belong to v3 namespace")
        replacements.append((old_scope, new_root / old_scope.relative_to(old_root)))
        attempted = any(k in entry for k in ("job_ref", "pid", "started_at", "reservation_bytes"))
        if attempted or old_scope.exists():
            need(entry["status"] == "failed" and type(entry.get("actual_controller_wait")) is int,
                 "Previous attempt must preserve actual controller wait")
            _job(task, entry, prior_ref)
            wait_ref, wait = _wait(task, entry)
            reserve = entry.get("reservation_bytes")
            need(type(reserve) is int and reserve > 0
                 and type(wait.get("output_bytes")) is int and wait["output_bytes"] <= reserve,
                 "Preserve full original failed output reservation")
            unknown = wait.get("owned_process_group_absent") is not True
            if unknown:
                need(observation.get("schema") == "workflow_v4_fresh_namespace_root_observation_v1"
                     and observation.get("failed_original_wait_ref") == wait_ref
                     and observation.get("historical_owned_group_absence") ==
                         "unproven, preserve original false value"
                     and isinstance(observation.get("root_decision"), str)
                     and bool(observation["root_decision"]),
                     "ROOT must explicitly accept historical unknown and reserve it conservatively")
            charge = reserve
            if not unknown:
                need(entry.get("owned_process_group_absent") is True
                     and type(entry.get("retained_bytes")) is int
                     and 0 < entry["retained_bytes"] <= reserve
                     and wait["output_bytes"] <= entry["retained_bytes"],
                     "Only genuinely drained original retained accounting may release failed reserve")
                charge = entry["retained_bytes"]
            failed.append(dict(task_id=key, original_entry=copy.deepcopy(entry), actual_wait_ref=wait_ref,
                scope=str(old_scope.relative_to(c.ROOT)), charged_bytes=charge,
                charge_basis="full_original_reservation" if unknown else "original_genuinely_drained_retained_bytes",
                unknown_cleanup=unknown, conservative_rss_bytes=task["max_rss_bytes"] if unknown else 0))
        else:
            need(not any(k in entry for k in ("result_ref", "retained_bytes")),
                 "Unattempted stage cannot claim completion")
    new_plan = copy.deepcopy(prior)
    new_entries = {}
    live_by_id = {b["source_task_id"]: b for b in live}
    for index, task in enumerate(new_plan["tasks"]):
        key = task["id"]
        if statuses[key] == "succeeded":
            new_entries[key] = copy.deepcopy(state["tasks"][key])
            new_entries[key]["adopted_from_v3"] = dict(plan_ref=prior_ref, state_snapshot_ref=snapshot_ref,
                                                      **adopted[key])
        elif key in live_by_id:
            new_plan["tasks"][index] = dict(id=key, resource="external",
                dependencies=task["dependencies"], workflow_predecessor=live_by_id[key])
            new_entries[key] = dict(status="pending", external=True,
                preserved_original_job_ref=state["tasks"][key]["job_ref"])
        else:
            task["output_directory"] = _map_paths(task["output_directory"], replacements)
            task["factory_arguments"] = _map_paths(task.get("factory_arguments", {}), replacements)
            new_entries[key] = dict(status="pending", external=False, prior_status=statuses[key],
                                    prior_state_snapshot_ref=snapshot_ref)
    failed_charge = sum(v["charged_bytes"] for v in failed)
    if failed:
        new_plan["external_storage"].append(dict(cap_bytes=failed_charge,
            scopes=[v["scope"] for v in failed], completion_evidence=[],
            retention="Permanent preserved v3 failed bytes, full caps for historical unknown, actual retained only after genuine drain"))
    new_plan.update(records_directory=records, operational_successor_of=prior_ref,
        adopted_prior_state_snapshot_ref=snapshot_ref, workflow_external_predecessors=live,
        prior_unknown_rss_bytes=sum(v["conservative_rss_bytes"] for v in failed),
        historical_unknown_acceptance_ref=observation_ref,
        continuation_approval_key="approved_continuation_ref",
        scientific_payloads_rehashed=0, scientific_factories_called=0)
    if include_qualification:
        fit = next(t for t in new_plan["tasks"] if t["id"] == "AM_treeloom_fit")
        args = fit["factory_arguments"]
        parent = "argformer_american_train_000"
        preparation = copy.deepcopy(args["preparation_refs"][parent])
        generation = copy.deepcopy(args["generation_refs"][parent])
        dependencies = sorted({preparation["$task"], generation["$task"]})
        qualification = dict(id=QUALIFICATION, resource="cpu", dependencies=dependencies,
            factory_module=TRELOOM_MODULE, factory_function="qualification_factory",
            factory_arguments=dict(preparation_ref=preparation, generation_ref=generation,
                max_rss_bytes=128*2**30, max_seconds=7200, max_output_bytes=16*2**20),
            output_directory=str(new_root / "american/shared" / QUALIFICATION),
            max_rss_bytes=128*2**30, max_seconds=7200, max_output_bytes=16*2**20)
        need(QUALIFICATION not in new_entries, "Never replace an existing qualification")
        new_plan["tasks"].insert(0, qualification)
        new_entries[QUALIFICATION] = dict(status="pending", external=False)
        fit["dependencies"].append(QUALIFICATION)
        fit["factory_module"] = TRELOOM_MODULE
        fit["factory_arguments"]["qualification_ref"] = {"$task": QUALIFICATION}
        new_plan["dispatch_hold_task_ids"] = [v for v in new_plan.get("dispatch_hold_task_ids", [])
                                               if v != "AM_treeloom_fit"]
        # Native GPU dispatch first, then CPU qualification before fresh Relate
        # retries. This changes operational readiness, never training settings.
        new_plan["tasks"].sort(key=lambda t: 0 if t["id"] == FIRST_GPU else
                               1 if t["id"] == QUALIFICATION else 2)
        new_plan["treeloom_qualification_task_id"] = QUALIFICATION
    # Preserve v3's native-MLM-first order and exact holds. Three original live
    # tasks are external, without state reservations or static duplicate caps.
    ready = [t["id"] for t in new_plan["tasks"] if t["resource"] == "gpu0"
             and new_entries[t["id"]]["status"] == "pending"
             and t["id"] not in new_plan.get("dispatch_hold_task_ids", [])
             and all(new_entries[d]["status"] == "succeeded" for d in t["dependencies"])]
    need(ready and ready[0] == FIRST_GPU, "Original validation MLM must remain first ready GPU stage")
    evidence = dict(schema="arggnn_v4_preserved_v3_work_v1", source_plan_ref=prior_ref,
        source_state_snapshot_ref=snapshot_ref, adopted_succeeded=adopted,
        live_predecessors=live, retained_failed_attempts=failed,
        total_failed_charge_bytes=failed_charge,
        prior_unknown_rss_bytes=new_plan["prior_unknown_rss_bytes"],
        first_ready_gpu_task_id=ready[0], scientific_factories_called=0,
        scientific_payloads_rehashed=0, old_scopes_unmodified=True)
    return new_plan, new_entries, evidence

def build():
    targets = [PLAN, RECORDS + "/source_v3_state_snapshot.json", RECORDS + "/preserved_v3_work.json",
               RECORDS + "/initial_state.json", RECORDS + "/state.json", RECORDS + "/freeze_result.json"]
    need(all(not c.contained_path(p).exists() for p in targets), "Never overwrite v4 frozen metadata")
    prior_ref = c.file_ref(PRIOR_PLAN)
    policy = c.read_json(c.POLICY_PATH)
    need(policy.get("approved_workflow_ref") == prior_ref and policy.get("workflow_drain_ref") == prior_ref,
         "Keep v3 approved and drain its dispatch before preparing v4")
    observation_ref = c.file_ref(OBSERVATION)
    need(observation_ref["sha256"] == OBSERVATION_SHA, "Exact ROOT unknown-cleanup acceptance required")
    # One atomic JSON snapshot. Never present a mutable live-state fingerprint
    # as replayable evidence.
    snapshot_ref = write_new(RECORDS + "/source_v3_state_snapshot.json", c.read_json(PRIOR_STATE))
    plan, entries, evidence = prepare_continuation(prior_ref, snapshot_ref, observation_ref, include_qualification=True)
    evidence_ref = write_new(RECORDS + "/preserved_v3_work.json", evidence)
    paths = {r["path"] for r in plan["sources"]} | set(NEW_SOURCES)
    # Source checks must prove every old v3 pin remains untouched, including
    # files unrelated to the ongoing CPU scientific stage.
    from arggnn.benchmarking import workflow
    workflow.source_check(plan["sources"])
    plan["sources"] = [c.file_ref(p) for p in sorted(paths)]
    plan["continuation_evidence_ref"] = evidence_ref
    plan_ref = write_new(PLAN, plan)
    workflow.load_plan(plan_ref, approved=False)
    initial = dict(plan_ref=plan_ref, tasks=entries, complete=False, run_status="prepared_not_released",
        updated_utc=datetime.now(timezone.utc).isoformat(), waiting_reasons={}, undrained_failed_jobs=[])
    initial_ref = write_new(RECORDS + "/initial_state.json", initial)
    state_ref = write_new(RECORDS + "/state.json", initial)
    result = dict(schema="arggnn_v4_frozen_continuation_v1", plan_ref=plan_ref,
        initial_state_ref=initial_ref, state_ref=state_ref, continuation_evidence_ref=evidence_ref,
        source_state_snapshot_ref=snapshot_ref, task_counts=dict(Counter(e["status"] for e in entries.values())),
        live_predecessor_count=len(plan["workflow_external_predecessors"]),
        failed_storage_charge_bytes=evidence["total_failed_charge_bytes"],
        prior_unknown_rss_bytes=plan["prior_unknown_rss_bytes"], old_approval_unchanged=True,
        scientific_factories_called=0, scientific_payloads_rehashed=0, execution_released=False)
    freeze_ref = write_new(RECORDS + "/freeze_result.json", result)
    print(json.dumps(dict(result, freeze_result_ref=freeze_ref), indent=2, sort_keys=True))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        build()
    else:
        parser.error("Explicit --freeze required; builder never releases or launches work")

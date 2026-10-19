"""Read only named v9 accounting scopes and exact closed European producer metadata."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PLAN = ROOT / "configs/benchmarks/executable_workflow_20261019_v9.json"
STATE = ROOT / ".runtime/benchmarking/workflow_20261019_v9/state.json"

def load(path):
    return json.loads(Path(path).read_text())

def reference(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def measure(relative):
    scope = ROOT / relative
    total = 0
    largest = []
    count = 0
    for base, dirs, files in os.walk(scope, followlinks=False):
        for name in dirs + files:
            path = Path(base) / name
            info = path.lstat()
            if path.is_symlink() and not path.resolve().is_relative_to(scope.resolve()):
                raise ValueError("Scope symlink escaped")
            total += max(info.st_size, info.st_blocks * 512) + 4096
            count += 1
            if name in files and not path.is_symlink():
                largest.append((max(info.st_size, info.st_blocks * 512) + 4096, str(path.relative_to(scope))))
    return {"bytes": total, "entries": count, "largest": sorted(largest, reverse=True)[:14]}

def producer(relative):
    scope = ROOT / relative
    wait_path = scope / "actual_wait.json"
    if not wait_path.is_file():
        return {"scope": relative, "closed_proof": False, "reason": "No direct wait"}
    wait = load(wait_path)
    job_ref = wait.get("job_ref")
    if not job_ref:
        return {"scope": relative, "closed_proof": False, "reason": "No exact workflow job reference"}
    job = load(ROOT / job_ref["path"])
    if reference(ROOT / job_ref["path"]) != job_ref:
        raise ValueError("Original job changed")
    source_plan = load(ROOT / job["plan_ref"]["path"])
    source_state = load(ROOT / source_plan["records_directory"] / "state.json")
    entry = source_state["tasks"][job["task"]["id"]]
    closed = (entry["status"] in ("failed", "succeeded")
              and type(entry.get("actual_controller_wait")) is int
              and entry.get("owned_process_group_absent") is True
              and wait.get("actual_direct_wait") is True
              and type(wait.get("actual_direct_returncode")) is int
              and wait.get("owned_process_group_absent") is True
              and entry.get("job_ref") == job_ref
              and (ROOT / job["task"]["output_directory"]).resolve() == scope.resolve())
    return {"scope": relative, "closed_proof": closed, "job_ref": job_ref,
            "wait_ref": reference(wait_path), "task_id": job["task"]["id"],
            "status": entry["status"], "controller_wait": entry.get("actual_controller_wait"),
            "direct_wait": wait.get("actual_direct_returncode"),
            "retained_bytes": entry.get("retained_bytes"), "measurement": measure(relative)}

def main():
    plan, state = load(PLAN), load(STATE)
    baseline = load(ROOT / plan["storage_baseline_ref"]["path"])
    external = []
    for index, row in enumerate(plan["external_storage"]):
        inactive = bool(row.get("completion_evidence"))
        for proof in row.get("completion_evidence", []):
            queue = load(ROOT / proof["queue_path"])
            entry = next(job for job in queue["jobs"] if job["admission"]["job_id"] == proof["job_id"])
            wait = load(ROOT / proof["wait_path"])
            inactive = (inactive and entry["status"] == "succeeded" and entry.get("actual_returncode") == 0
                        and wait.get("actual_direct_wait") is True and wait.get("actual_direct_returncode") == 0
                        and wait.get("owned_process_group_absent") is True)
        charge = sum(measure(scope)["bytes"] for scope in row["scopes"]) if inactive else row["cap_bytes"]
        external.append({"index": index, "charged_bytes": charge, "cap_bytes": row["cap_bytes"]})
    local_charges = {}
    for entry in state["tasks"].values():
        if not entry.get("external") and "reservation_bytes" in entry:
            local_charges[entry["status"]] = local_charges.get(entry["status"], 0) + entry.get("retained_bytes", entry["reservation_bytes"])
    predecessor_charges = []
    for binding in plan["workflow_external_predecessors"]:
        entry = load(ROOT / binding["source_state_path"])["tasks"][binding["source_task_id"]]
        drained = (entry["status"] in ("succeeded", "failed")
                   and type(entry.get("actual_controller_wait")) is int
                   and entry.get("owned_process_group_absent") is True)
        charge = entry.get("retained_bytes") if drained else binding["max_output_bytes"]
        predecessor_charges.append({"task_id": binding["source_task_id"], "charged_bytes": charge})
    total = (baseline["measured_project_bytes"] + plan["metadata_margin_bytes"]
             + sum(row["charged_bytes"] for row in external) + sum(local_charges.values())
             + sum(row["charged_bytes"] for row in predecessor_charges))
    targets = []
    for index in (9, 11, 12, 13):
        for scope in plan["external_storage"][index]["scopes"]:
            if "/european/" in scope or "/european_c4/" in scope or "/inherited/AP__" in scope:
                targets.append(producer(scope))
    c4 = next(binding for binding in plan["workflow_external_predecessors"] if binding["source_task_id"].endswith("__validation_infer"))
    c4_scope = str(Path(c4["output_directory"]).relative_to(ROOT)) if Path(c4["output_directory"]).is_absolute() else c4["output_directory"]
    targets.append(producer(c4_scope))
    report = {"schema": "v9_exact_scope_storage_audit_v1", "plan_ref": reference(PLAN),
              "state_updated_utc": state.get("updated_utc"), "baseline_bytes": baseline["measured_project_bytes"],
              "metadata_margin_bytes": plan["metadata_margin_bytes"], "external_charges": external,
              "local_charges_by_status": local_charges, "predecessor_charges": predecessor_charges,
              "current_projection_bytes": total, "strict_headroom_bytes": plan["storage_ceiling_bytes"] - total - 1,
              "next_AP_cap_bytes": 96 * 2**30, "next_AP_deficit_bytes": max(0, total + 96 * 2**30 - plan["storage_ceiling_bytes"] + 1),
              "closed_European_targets": targets}
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

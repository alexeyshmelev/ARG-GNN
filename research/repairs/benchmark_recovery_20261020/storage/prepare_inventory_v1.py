"""Prepare an immutable exact-scope recovery inventory without reading payload bytes."""
import hashlib
import json
import math
import os
from pathlib import Path
from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v1 as recovery

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = ROOT / "research/repairs/benchmark_recovery_20261020/storage"

def main():
    audit = json.loads((DIRECTORY / "audit_storage_v1.json").read_text())
    plan = json.loads((ROOT / audit["plan_ref"]["path"]).read_text())
    targets = []
    for row in audit["closed_European_targets"]:
        if not row["closed_proof"]:
            continue
        scope = ROOT / row["scope"]
        job = recovery.read(ROOT, row["job_ref"])
        source_plan = recovery.read(ROOT, job["plan_ref"])
        entry = json.loads((ROOT / source_plan["records_directory"] / "state.json").read_text())["tasks"][row["task_id"]]
        candidates, protected = [], []
        for base, dirs, files in os.walk(scope, followlinks=False):
            dirs.sort()
            files.sort()
            for name in files:
                item = Path(base) / name
                relative = item.relative_to(scope)
                info = dict(path=str(item.relative_to(ROOT)), **recovery.record(item))
                if recovery.candidate(relative, entry["status"] == "failed"):
                    candidates.append(info)
                else:
                    protected.append(info)
        if not candidates:
            continue
        candidates.sort(key=lambda value: value["path"])
        candidate_bytes = sum(value["charge_bytes"] for value in candidates)
        largest = max(value["charge_bytes"] for value in candidates)
        if largest + 16384 > recovery.MAX_TEMP:
            raise ValueError("Candidate needs a separately reviewed temporary bound")
        output_bound = math.ceil((candidate_bytes + largest + 4 * recovery.METADATA) / 2**30) * 2**30
        if output_bound > 32 * 2**30:
            raise ValueError("Closed recovery scope needs more than 32GiB output")
        groups = [group for group in plan["external_storage"] if row["scope"] in group["scopes"]]
        if groups:
            if len(groups) != 1:
                raise ValueError("Ambiguous original storage charge")
            group = groups[0]
            accounting = dict(kind="external_storage", group_scopes=group["scopes"],
                              group_cap_bytes=group["cap_bytes"])
        else:
            bindings = [binding for binding in plan["workflow_external_predecessors"]
                        if (ROOT / binding["output_directory"]).resolve() == scope.resolve()]
            if len(bindings) != 1:
                raise ValueError("No unique unchanged predecessor charge")
            accounting = dict(kind="workflow_predecessor", source_task_id=bindings[0]["source_task_id"])
        key = "closed_" + hashlib.sha256(row["scope"].encode()).hexdigest()[:16]
        proof = dict(job_ref=row["job_ref"], wait_ref=row["wait_ref"],
                     source_plan_ref=job["plan_ref"], task_id=row["task_id"],
                     entry={name:entry.get(name) for name in
                            ("status", "actual_controller_wait", "owned_process_group_absent", "job_ref", "retained_bytes")})
        target = dict(key=key, scope=row["scope"], producer=proof, accounting=accounting,
                      before_scope_bytes=recovery.scope_bytes(scope),
                      candidates=candidates, protected_files=protected,
                      candidate_charge_bytes=candidate_bytes,
                      largest_candidate_charge_bytes=largest,
                      max_output_bytes=output_bound,
                      future_old_scope_writes_require_root_revocation=True)
        recovery.closed_proof(target, ROOT)
        targets.append(target)
    targets.sort(key=lambda target: (target["producer"]["entry"]["status"] != "failed",
                                     -target["candidate_charge_bytes"]))
    inventory = dict(schema=recovery.SCHEMA + "_inventory", source_plan_ref=audit["plan_ref"],
                     audit_ref=recovery.reference(ROOT, DIRECTORY / "audit_storage_v1.json"),
                     targets=targets, initial_credit_bytes=0,
                     historical_unknown_scopes_untouched=True,
                     old_sources_waits_and_failures_unchanged=True,
                     candidate_bytes_are_not_predicted_compression_savings=True,
                     codec_ref=recovery.reference(ROOT, recovery.CODEC))
    destination = DIRECTORY / "closed_archive_inventory_v1.json"
    with destination.open("x") as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("INVENTORY_REF", recovery.reference(ROOT, destination))
    print("TARGETS", len(targets), "FAILED", sum(target["producer"]["entry"]["status"] == "failed" for target in targets))
    print("ORIGINAL_SCOPES_BYTES", sum(target["before_scope_bytes"] for target in targets))
    print("CANDIDATE_BYTES", sum(target["candidate_charge_bytes"] for target in targets))
    for target in targets:
        print("TARGET", target["key"], target["producer"]["entry"]["status"],
              target["before_scope_bytes"], target["candidate_charge_bytes"],
              len(target["candidates"]), target["max_output_bytes"], target["scope"])

if __name__ == "__main__":
    main()

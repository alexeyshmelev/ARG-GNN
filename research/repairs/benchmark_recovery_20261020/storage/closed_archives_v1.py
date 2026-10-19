"""Source-only closed-European recovery; ROOT owns reviewed activation.

Old jobs, waits, source plans and failures remain immutable. Each selected raw
file is replaced only after verified gzip and a durable restoration sidecar.
New archives and partial files remain in the new worker's accounted scope.
"""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from research.reviews.benchmark_repairs_20261019.resources_v9 import european_archive_v1 as codec

ROOT = Path(__file__).resolve().parents[4]
SOURCE = "research/repairs/benchmark_recovery_20261020/storage/closed_archives_v1.py"
INVENTORY = "research/repairs/benchmark_recovery_20261020/storage/closed_archive_inventory_v1.json"
SHIM = "src/arggnn/benchmarking/storage_recovery_20261020.py"
MODULE = "arggnn.benchmarking.storage_recovery_20261020"
CODEC = "research/reviews/benchmark_repairs_20261019/resources_v9/european_archive_v1.py"
NEW_SOURCES = (SOURCE, INVENTORY, SHIM, CODEC)
MAX_TEMP = 8 * 2**30
METADATA = 32 * 2**20
SCHEMA = "exact_closed_european_archive_20261020_v1"

def need(ok, message):
    if not ok:
        raise ValueError(message)

def path(repo, value):
    return codec._path(Path(repo).resolve(), value)

def reference(repo, value):
    return codec._ref(Path(repo).resolve(), path(repo, value))

def read(repo, value, cap=32 * 2**20):
    return codec._read(Path(repo).resolve(), value, cap)

def record(value):
    value = Path(value)
    info = value.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Single-link regular payload required")
    return dict(identity=[info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                          info.st_ctime_ns, info.st_mode, info.st_nlink],
                charge_bytes=max(info.st_size, info.st_blocks * 512) + 4096)

def scope_bytes(value):
    total = 0
    for base, dirs, files in os.walk(value, followlinks=False):
        for name in dirs + files:
            item = Path(base) / name
            info = item.lstat()
            need(not item.is_symlink(), "Closed recovery scope cannot contain symlinks")
            total += max(info.st_size, info.st_blocks * 512) + 4096
    return total

def candidate(relative, failed):
    relative = Path(relative)
    if not relative.parts or relative.parts[0] != "native":
        return False
    within = relative.relative_to("native")
    if str(within) in ("inputs/input.vcf", "inputs/input.haps", "inputs/input.annot",
                        "pre_size.anc", "pre_size.mut", "post_size.trees", "post_size.pairwise.bin"):
        return True
    return (failed and within.parts[0] in ("pre_size", "post_size")
            and within.suffix in (".bin", ".hap", ".anc", ".mut"))

def closed_proof(target, repo=ROOT):
    owner = path(repo, target["scope"])
    proof = target["producer"]
    wait = read(repo, proof["wait_ref"], 2**20)
    job = read(repo, proof["job_ref"])
    source_plan = read(repo, proof["source_plan_ref"])
    need(wait.get("job_ref") == proof["job_ref"] and job.get("plan_ref") == proof["source_plan_ref"],
         "Exact original producer lineage required")
    need(job["task"]["id"] == proof["task_id"]
         and path(repo, job["task"]["output_directory"]) == owner,
         "Original producer scope changed")
    source_state = json.loads((path(repo, source_plan["records_directory"]) / "state.json").read_text())
    need(source_state.get("plan_ref") == proof["source_plan_ref"], "Original source state plan changed")
    original_tasks = [task for task in source_plan["tasks"] if task["id"] == proof["task_id"]]
    need(len(original_tasks) == 1 and original_tasks[0] == job["task"], "Exact original scientific descriptor required")
    need("benchmark_campaign_20261018_workflow" in owner.parts
         and (proof["task_id"].startswith("AP__") and proof["task_id"].endswith("__ap_infer")
              or proof["task_id"].startswith("C4__") and proof["task_id"].endswith("__validation_infer")),
         "Only exact completed European inference scopes are eligible")
    entry = source_state["tasks"][proof["task_id"]]
    for key in ("status", "actual_controller_wait", "owned_process_group_absent", "job_ref", "retained_bytes"):
        need(entry.get(key) == proof["entry"].get(key), "Original terminal entry changed")
    need(entry["status"] in ("failed", "succeeded")
         and type(entry.get("actual_controller_wait")) is int
         and entry.get("owned_process_group_absent") is True
         and wait.get("actual_direct_wait") is True
         and type(wait.get("actual_direct_returncode")) is int
         and wait.get("owned_process_group_absent") is True,
         "Genuine closed producer and owned-group drain required")
    need(target["future_old_scope_writes_require_root_revocation"] is True,
         "Explicit ROOT old-scope write revocation required")
    return owner

def load_target(inventory_ref, target_key, repo=ROOT):
    inventory = read(repo, inventory_ref)
    need(inventory["schema"] == SCHEMA + "_inventory", "Wrong closed recovery inventory")
    matches = [row for row in inventory["targets"] if row["key"] == target_key]
    need(len(matches) == 1, "One exact inventoried closed scope required")
    target = matches[0]
    owner = closed_proof(target, repo)
    need(inventory.get("initial_credit_bytes") == 0
         and inventory.get("historical_unknown_scopes_untouched") is True
         and reference(repo, CODEC) == inventory["codec_ref"], "Exact zero-credit reviewed codec inventory required")
    originals = set()
    for row in target["candidates"]:
        original = path(repo, row["path"])
        need(original.is_relative_to(owner) and candidate(original.relative_to(owner), target["producer"]["entry"]["status"] == "failed")
             and original not in originals, "One allowlisted old native intermediate required")
        originals.add(original)
    need(target["candidates"] and not originals.intersection(path(repo, row["path"]) for row in target["protected_files"]),
         "Science and evidence protection cannot overlap archival candidates")
    required = sum(row["charge_bytes"] for row in target["candidates"]) + max(row["charge_bytes"] for row in target["candidates"]) + 4 * METADATA
    need(type(target["max_output_bytes"]) is int and required <= target["max_output_bytes"] <= 32 * 2**30,
         "Complete final-plus-temporary output envelope required")
    return target

def factory(*, inventory_ref, target_key, output_directory):
    target = load_target(inventory_ref, target_key)
    owner = path(ROOT, target["scope"])
    need(scope_bytes(owner) == target["before_scope_bytes"], "Closed scope changed before dispatch")
    for row in target["candidates"]:
        need(record(path(ROOT, row["path"]))["identity"] == row["identity"], "Candidate identity changed")
    spec = dict(schema=SCHEMA, inventory_ref=inventory_ref, target_key=target_key,
                output_directory=output_directory)
    return dict(module=MODULE, function="run", python=".runtime/env/bin/python", spec=spec,
                source_refs=[reference(ROOT, name) for name in NEW_SOURCES])

def run(spec, *, resource_check, execution_check=None, bounded_runner=None):
    need(spec.get("schema") == SCHEMA, "Exact closed archival specification required")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Recovery is CPU-only")
    target = load_target(spec["inventory_ref"], spec["target_key"])
    owner = path(ROOT, target["scope"])
    need(scope_bytes(owner) == target["before_scope_bytes"], "Closed scope changed")
    output = path(ROOT, spec["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    need(not output.is_relative_to(owner) and not owner.is_relative_to(output),
         "New accounted output must be separate from old scope")
    lock_directory = path(ROOT, ".runtime/locks")
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(owner).encode()).hexdigest()
    descriptor = os.open(lock_directory / ("closed_recovery_" + lock_key + ".lock"),
                         os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    check_execution = execution_check or (lambda: None)
    entries = []
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        codec._save(output / "original_inventory.json", target)
        finalized_bytes = scope_bytes(output)
        for index, row in enumerate(target["candidates"]):
            source = path(ROOT, row["path"])
            need(record(source)["identity"] == row["identity"], "Original candidate changed")
            check_execution()
            def before_unlink():
                check_execution()
                wait = read(ROOT, target["producer"]["wait_ref"], 2**20)
                need(wait.get("actual_direct_wait") is True and wait.get("owned_process_group_absent") is True,
                     "Original genuine drain evidence changed")
            allowance = min(MAX_TEMP, target["max_output_bytes"] - finalized_bytes - METADATA)
            need(allowance >= 2**20, "No admitted temporary output allowance remains")
            entry = codec.archive_one(source, output / ("file_%05d.json" % index),
                                      allowance, resource_check, before_unlink)
            if entry["state"] == "archive_committed":
                archive_record = record(entry["archive_path"])
                entry["archive_identity"] = archive_record["identity"]
                finalized_bytes += archive_record["charge_bytes"] + record(output / ("file_%05d.json" % index))["charge_bytes"]
            entries.append(entry)
        for row in target["protected_files"]:
            need(record(path(ROOT, row["path"]))["identity"] == row["identity"],
                 "Protected original science or evidence changed")
        after = scope_bytes(owner)
        need(after <= target["before_scope_bytes"], "Old scope grew during closed recovery")
        manifest = dict(schema=SCHEMA, inventory_ref=spec["inventory_ref"],
                        target_key=target["key"], owner_scope=target["scope"],
                        archive_output_directory=str(output),
                        original_wait_ref=target["producer"]["wait_ref"],
                        original_status_preserved=target["producer"]["entry"]["status"],
                        entries=entries, before_scope_bytes=target["before_scope_bytes"],
                        after_scope_bytes=after,
                        original_scope_reduction_bytes=target["before_scope_bytes"]-after,
                        raw_replacement_after_verified_durable_archive=True,
                        every_original_byte_restorable=True,
                        protected_science_and_evidence_unchanged=True,
                        archives_and_partial_files_inside_new_accounted_scope=True)
        codec._save(output / "restore_manifest.json", manifest)
    return reference(ROOT, output / "restore_manifest.json")

def add_recovery_tasks(plan, output_root, repo=ROOT):
    """No credit at preparation: prepend only newly inventoried closed workers."""
    need(plan["storage_ceiling_bytes"] == 980_000_000_000, "Keep 980GB ceiling")
    inventory_ref = reference(repo, INVENTORY)
    inventory = read(repo, inventory_ref)
    tasks = []
    for target in inventory["targets"]:
        load_target(inventory_ref, target["key"], repo)
        identifier = "STORAGE_RECOVERY__" + target["key"]
        tasks.append(dict(id=identifier, dependencies=[], resource="cpu",
                          max_rss_bytes=2**30, max_output_bytes=target["max_output_bytes"],
                          max_seconds=86400,
                          output_directory=str(Path(output_root) / "closed_storage" / identifier),
                          factory_module=MODULE, factory_function="factory",
                          factory_arguments=dict(inventory_ref=inventory_ref, target_key=target["key"]),
                          closed_storage_recovery_target=target["key"]))
    result = copy.deepcopy(plan)
    need(not {task["id"] for task in tasks} & {task["id"] for task in result["tasks"]},
         "No automatic replay of recovery tasks")
    result["tasks"] = tasks + result["tasks"]
    result["closed_storage_recovery"] = dict(schema=SCHEMA, inventory_ref=inventory_ref,
                                           initial_credit_bytes=0,
                                           old_scope_writer_revocation_required=True)
    return result, dict(tasks_added=len(tasks), initial_credit_bytes=0,
                        exact_old_scopes=[target["scope"] for target in inventory["targets"]])

def completed_storage_reduction(plan, state, repo=ROOT):
    """Deduct only verified successful CLOSED external-scope changes from base projection.

    Archive output is already counted as a normal task by the unchanged base.
    Local AP archive remeasurements are outside this correction, preventing
    duplicate credit. Original external caps and source entries are not edited.
    """
    policy = plan.get("closed_storage_recovery")
    if policy is None:
        return 0
    need(policy["schema"] == SCHEMA and policy["initial_credit_bytes"] == 0, "Exact recovery policy required")
    total = 0
    seen = set()
    for task in plan["tasks"]:
        if task.get("factory_module") != MODULE:
            continue
        entry = state["tasks"][task["id"]]
        if entry["status"] != "succeeded":
            continue
        target = load_target(policy["inventory_ref"], task["closed_storage_recovery_target"], repo)
        need(target["scope"] not in seen, "Closed scope cannot receive duplicate credit")
        seen.add(target["scope"])
        need(task["factory_arguments"] == dict(inventory_ref=policy["inventory_ref"], target_key=target["key"]),
             "Recovery task inventory changed")
        need(entry.get("actual_controller_wait") == 0 and entry.get("owned_process_group_absent") is True,
             "Actual successful recovery controller wait and drain required")
        job = read(repo, entry["job_ref"])
        need(job["task"] == task and job["call"]["module"] == MODULE and job["call"]["function"] == "run"
             and job["call"]["spec"] == dict(schema=SCHEMA, inventory_ref=policy["inventory_ref"],
                                            target_key=target["key"], output_directory=str(path(repo, task["output_directory"]) / "native")),
             "Recovery job must authenticate exact plan task")
        wait = read(repo, reference(repo, path(repo, task["output_directory"]) / "actual_wait.json"), 2**20)
        need(wait.get("job_ref") == entry["job_ref"] and wait.get("actual_direct_wait") is True
             and wait.get("actual_direct_returncode") == 0 and wait.get("owned_process_group_absent") is True,
             "Actual successful recovery scientific wait and drain required")
        envelope = read(repo, entry["result_ref"])
        need(envelope.get("job_ref") == entry["job_ref"], "Recovery result changed")
        manifest_ref = envelope["value"]
        output = path(repo, task["output_directory"]) / "native"
        need(path(repo, manifest_ref["path"]) == output / "restore_manifest.json", "Exact restoration manifest required")
        manifest = read(repo, manifest_ref)
        need(manifest["schema"] == SCHEMA and manifest["inventory_ref"] == policy["inventory_ref"]
             and manifest["target_key"] == target["key"] and manifest["owner_scope"] == target["scope"]
             and manifest["original_wait_ref"] == target["producer"]["wait_ref"]
             and manifest["every_original_byte_restorable"] is True
             and manifest["protected_science_and_evidence_unchanged"] is True
             and manifest["raw_replacement_after_verified_durable_archive"] is True
             and manifest["archives_and_partial_files_inside_new_accounted_scope"] is True,
             "Complete lossless closed-scope proof required")
        originals = {row["path"]: row for row in target["candidates"]}
        need(len(manifest["entries"]) == len(originals), "Every inventoried candidate must have a terminal entry")
        seen_originals = set()
        for row in manifest["entries"]:
            original = path(repo, row["original_path"])
            original_key = str(original.relative_to(Path(repo).resolve()))
            need(original_key in originals and original_key not in seen_originals,
                 "One exact original candidate per restoration entry")
            seen_originals.add(original_key)
            need(row["decompressed_hash_and_size_verified"] is True, "Verified decompressed content required")
            if row["state"] == "archive_committed":
                archive = path(repo, row["archive_path"])
                need(archive.is_relative_to(output) and not original.exists()
                     and record(archive)["identity"] == row["archive_identity"]
                     and archive.stat().st_size == row["archive_bytes"],
                     "Committed archive must remain in its accounted scope")
            else:
                need(row["state"] == "original_kept_no_storage_gain" and row["credited_bytes"] == 0
                     and record(original)["identity"] == originals[original_key]["identity"],
                     "Uncompressed original receives no speculative credit")
        binding = target["accounting"]
        if binding["kind"] == "external_storage":
            matches = [row for row in plan["external_storage"] if row["scopes"] == binding["group_scopes"]]
            need(len(matches) == 1 and matches[0]["cap_bytes"] == binding["group_cap_bytes"],
                 "Original external reservation changed")
        else:
            need(binding["kind"] == "workflow_predecessor", "Recognized immutable original charge required")
            matches = [row for row in plan["workflow_external_predecessors"]
                       if row["source_task_id"] == binding["source_task_id"]]
            need(len(matches) == 1 and matches[0]["original_job_ref"] == target["producer"]["job_ref"],
                 "Exact preserved workflow predecessor charge required")
        owner = path(repo, target["scope"])
        current = scope_bytes(owner)
        need(current == manifest["after_scope_bytes"] and manifest["before_scope_bytes"] == target["before_scope_bytes"]
             and 0 <= current <= target["before_scope_bytes"], "Current measured closed scope changed")
        for row in target["protected_files"]:
            need(record(path(repo, row["path"]))["identity"] == row["identity"], "Protected original changed")
        delta = target["before_scope_bytes"] - current
        need(delta == manifest["original_scope_reduction_bytes"], "No invented storage reduction")
        total += delta
    return total

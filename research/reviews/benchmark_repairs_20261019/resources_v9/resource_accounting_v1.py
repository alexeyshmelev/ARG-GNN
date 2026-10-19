"""Source-only v9 resource correction; activation remains ROOT-owned.

Historical unknown waits remain unknown. Storage is never deleted here. The
native settlement is a current-scope operational closure, not a historical
owned-group absence claim. Fresh RAM uses kernel MemAvailable and full live
managed caps, with no PSS credit or guessed historical RSS.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path

SOURCE = "research/reviews/benchmark_repairs_20261019/resources_v9/resource_accounting_v1.py"
SETTLEMENT = "research/reviews/benchmark_repairs_20261019/resources_v9/native_chain_settlement_v1.json"
NEW_SOURCES = (SOURCE, SETTLEMENT)
NATIVE_SCOPES = (
    "experiments/results/benchmark_campaign_20261018/native_argformer_val000_contrastive_ordered_state",
    "experiments/results/benchmark_campaign_20261018/native_argformer_predictions",
)
OLD_NATIVE_CAP = 92 * 2**30
OLD_FIXED_RSS = 256 * 2**30
CEILING = 980_000_000_000
MARGIN = 64 * 2**30

def need(condition, message):
    if not condition:
        raise ValueError(message)

def _root(repo_root):
    return Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[4]

def _path(repo, value):
    path = Path(value)
    path = path if path.is_absolute() else repo / path
    need(path.resolve().is_relative_to(repo), "Path escaped repository")
    return path

def _json(path):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 2**20,
         "Small regular metadata file required")
    return json.loads(path.read_text())

def file_ref(repo, path):
    path = _path(repo, path)
    _json(path)
    return dict(path=str(path.relative_to(repo)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def _read_ref(repo, ref):
    need(type(ref) is dict and set(ref) == {"path", "sha256"}, "Exact metadata reference required")
    path = _path(repo, ref["path"])
    need(file_ref(repo, path) == ref, "Resource evidence changed: " + ref["path"])
    return _json(path)

def measure_scopes(repo_root, scopes):
    """Same conservative byte convention as execution.scope_bytes; no payload reads."""
    repo = _root(repo_root)
    roots = [_path(repo, value) for value in scopes]
    resolved = [path.resolve() for path in roots]
    need(len(set(resolved)) == len(resolved), "Duplicate scope")
    for index, path in enumerate(roots):
        need(path.is_dir() and not path.is_symlink(), "Existing regular scope required")
        need(not any(resolved[index].is_relative_to(other)
                     for other in resolved if other != resolved[index]), "Overlapping scopes")
    records, measurements = [], []
    for scope, path in zip(scopes, roots):
        total = count = 0
        for directory, dirs, files in os.walk(path, followlinks=False):
            dirs.sort()
            files.sort()
            for name in sorted(dirs + files):
                item = Path(directory) / name
                info = item.lstat()
                target = os.readlink(item) if item.is_symlink() else None
                if target is not None:
                    need(item.resolve().is_relative_to(path.resolve()), "Scope symlink escaped")
                charge = max(info.st_size, info.st_blocks * 512) + 4096
                total += charge
                count += 1
                records.append([str(item.relative_to(repo)), info.st_mode, info.st_dev,
                                info.st_ino, info.st_size, info.st_blocks,
                                info.st_mtime_ns, info.st_ctime_ns, target])
        measurements.append(dict(scope=str(scope), bytes=total, entries=count))
    encoded = json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode()
    return dict(total_bytes=sum(row["bytes"] for row in measurements),
                scopes=measurements, metadata_manifest_sha256=hashlib.sha256(encoded).hexdigest())

def _verify_settlement(repo, evidence):
    need(evidence.get("schema") == "native_chain_current_scope_settlement_v1", "Wrong settlement schema")
    need(evidence.get("historical_owned_group_absence_asserted") is False,
         "Historical absence must remain unasserted")
    need(evidence.get("root_activation_and_future_writer_revocation_required") is True,
         "ROOT operational acceptance required")
    source_plan = _read_ref(repo, evidence["source_plan_ref"])
    need(source_plan["storage_baseline_ref"] == evidence["storage_baseline_ref"], "Baseline lineage changed")
    _read_ref(repo, evidence["storage_baseline_ref"])
    for row in evidence["terminal_evidence"]:
        value = _read_ref(repo, row["ref"])
        if row["kind"] == "actual_wait":
            need(value.get("job_id") == row["job_id"] and value.get("actual_direct_wait") is True
                 and type(value.get("actual_direct_returncode")) is int
                 and value["actual_direct_returncode"] == 0
                 and value.get("supervisor_error") is None, "Native chain direct wait is not successful")
        elif row["kind"] == "queue_result":
            need(value.get("admission", {}).get("job_id") == row["job_id"]
                 and value.get("status") == "succeeded" and value.get("actual_returncode") == 0
                 and value.get("error") is None, "Native queue is not successful")
        else:
            raise ValueError("Unrecognized terminal evidence")
    waits = [row["job_id"] for row in evidence["terminal_evidence"] if row["kind"] == "actual_wait"]
    prefix = "native_argformer_val000_contrastive_ordered_state_20261018"
    need(sorted(waits) == sorted([prefix] + [prefix + "/" + side for side in
         ("query_train", "query_val", "reference_train", "reference_val", "retrieval")]),
         "All six exact native chain waits required")
    need(evidence["measurement"] == measure_scopes(repo, NATIVE_SCOPES),
         "Settled native scopes changed after observation")
    need(0 <= evidence["measurement"]["total_bytes"] <= OLD_NATIVE_CAP, "Native retained bytes exceed cap")

def apply_resource_corrections(plan, repo_root=None):
    """Return a copied source-only plan and evidence. Never write or activate."""
    repo = _root(repo_root)
    need("resource_accounting_v9" not in plan, "Resource correction already applied")
    need(plan.get("storage_ceiling_bytes") == CEILING, "Preserve the 980 GB ceiling")
    need(plan.get("host_ram_margin_bytes") == MARGIN, "Preserve 64 GiB host margin")
    need(plan.get("max_parallel_rss_bytes") == 1280 * 2**30, "Preserve managed aggregate cap")
    need(plan.get("external_native_rss_bytes") == OLD_FIXED_RSS
         and plan.get("prior_unknown_rss_bytes") == OLD_FIXED_RSS, "Unexpected legacy RSS fields")
    evidence_ref = file_ref(repo, SETTLEMENT)
    evidence = _read_ref(repo, evidence_ref)
    _verify_settlement(repo, evidence)
    need(plan["storage_baseline_ref"] == evidence["storage_baseline_ref"], "Target baseline changed")
    matches = [index for index, row in enumerate(plan["external_storage"])
               if tuple(row["scopes"]) == NATIVE_SCOPES]
    need(len(matches) == 1, "One exact external native storage scope required")
    original = plan["external_storage"][matches[0]]
    need(original["cap_bytes"] == OLD_NATIVE_CAP and original.get("completion_evidence") == [],
         "Original native allowance changed")
    corrected = copy.deepcopy(plan)
    changed = corrected["external_storage"][matches[0]]
    changed["cap_bytes"] = evidence["measurement"]["total_bytes"]
    changed["settled_from_cap_bytes"] = OLD_NATIVE_CAP
    changed["retained_measurement_ref"] = evidence_ref
    changed["retention"] = "All files preserved; current completed native-chain scope retention"
    corrected["external_native_rss_bytes"] = 0
    corrected["prior_unknown_rss_bytes"] = 0
    corrected["resource_accounting_v9"] = dict(
        schema="workflow_v9_current_host_resource_accounting_v1",
        source_only=True,
        activation_owner="ROOT",
        source=SOURCE,
        settlement_ref=evidence_ref,
        baseline_unchanged=True,
        historical_waits_and_group_absence_relabelled=False,
        original_fixed_rss_bytes=dict(external_native_rss_bytes=OLD_FIXED_RSS,
                                      prior_unknown_rss_bytes=OLD_FIXED_RSS),
        ram_admission="kernel_MemAvailable_minus_full_live_managed_caps",
        unknown_group_pss_credit_bytes=0,
        fixed_legacy_estimates_removed_from_fresh_aggregate_only=True,
        static_scope_validation_required_before_each_admission=True,
        future_native_chain_writer_revocation_required=True,
    )
    return corrected, dict(native_storage_before_bytes=OLD_NATIVE_CAP,
                           native_storage_after_bytes=changed["cap_bytes"],
                           released_native_storage_bytes=OLD_NATIVE_CAP - changed["cap_bytes"],
                           removed_fixed_ram_estimates_bytes=2 * OLD_FIXED_RSS,
                           settlement_ref=evidence_ref,
                           historical_waits_relabelled=False,
                           files_removed=0, activated=False)

def validate_static_settlements(plan, repo_root=None):
    """Call before each new storage admission; fail on any settled-scope change."""
    repo = _root(repo_root)
    policy = plan.get("resource_accounting_v9")
    need(policy is not None and policy.get("static_scope_validation_required_before_each_admission") is True,
         "V9 resource policy required")
    evidence = _read_ref(repo, policy["settlement_ref"])
    _verify_settlement(repo, evidence)
    rows = [row for row in plan["external_storage"] if tuple(row["scopes"]) == NATIVE_SCOPES]
    need(len(rows) == 1 and rows[0]["cap_bytes"] == evidence["measurement"]["total_bytes"]
         and rows[0].get("retained_measurement_ref") == policy["settlement_ref"],
         "Native settled charge changed")
    return evidence["measurement"]["total_bytes"]

def kernel_available_ram_bytes(meminfo_path="/proc/meminfo"):
    """No PSS credit: every current process is already reflected by the kernel."""
    rows = [line.split() for line in Path(meminfo_path).read_text().splitlines()
            if line.startswith("MemAvailable:")]
    need(len(rows) == 1 and len(rows[0]) == 3 and rows[0][2] == "kB",
         "Unambiguous kernel MemAvailable required")
    value = int(rows[0][1]) * 1024
    need(value > 0, "Positive MemAvailable required")
    return value

def ram_admission(mem_available_bytes, live_managed_caps, new_cap_bytes,
                  host_margin_bytes=MARGIN, aggregate_cap_bytes=1280 * 2**30):
    """Pure conservative decision. Unknown/historical groups earn no credit."""
    values = [mem_available_bytes, new_cap_bytes, host_margin_bytes, aggregate_cap_bytes,
              *live_managed_caps]
    need(all(type(value) is int and value >= 0 for value in values), "Nonnegative integer bytes required")
    live = sum(live_managed_caps)
    return dict(allowed=(live + new_cap_bytes <= aggregate_cap_bytes
                        and mem_available_bytes - live >= new_cap_bytes + host_margin_bytes),
                kernel_MemAvailable_bytes=mem_available_bytes,
                live_managed_reservations_bytes=live,
                new_cap_bytes=new_cap_bytes, host_margin_bytes=host_margin_bytes,
                aggregate_cap_bytes=aggregate_cap_bytes, pss_credit_bytes=0)

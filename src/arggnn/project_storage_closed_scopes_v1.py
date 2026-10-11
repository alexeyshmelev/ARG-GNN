"""H038 storage adapter for existing certified V9 completed-scope credits.

Preserves the original journal, reservations and source guards. Closed job
allowances become their recorded retained-byte charge. No global disk or
process-descriptor scan occurs here. Scientific computation is unchanged.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

from scripts import project_budget_v9 as budget

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/project_storage_closed_scopes_v1.py"


def _source_ref(ref, expected_path):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or ref["path"] != expected_path:
        raise ValueError("Exact root-pinned budget component reference required")
    if budget.v4.file_bytes(ROOT, ref["path"])[1] != ref["sha256"]:
        raise ValueError("Budget component source differs from root pin")
    return ref



_CHECKED_RESERVATIONS_V1 = {}

def _reservation_fingerprint_v1(storage):
    """Cheap coherence checks for immutable journal plus owned consumer pins."""
    paths = [budget.v4.LEGACY, budget.v4.SUPPLEMENTAL, budget.JOURNAL_REL,
             budget.SOURCE_REL, budget.LAUNCHER_REL,
             "configs/project_budget_v4.json"]
    paths.extend(ref["path"] for ref in storage["reservation"]["consumer_bindings"])
    rows = []
    for relative in dict.fromkeys(paths):
        path = budget.v4.canonical(ROOT, relative)
        st = path.lstat()
        rows.append((str(path), st.st_dev, st.st_ino, st.st_mode, st.st_nlink,
                     st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    return tuple(rows)

def _checked_reservation_v1(storage, implementation, launcher):
    """Replay unchanged immutable history once, then recheck on metadata change."""
    key = budget.digest(storage["reservation"])
    before = _reservation_fingerprint_v1(storage)
    cached = _CHECKED_RESERVATIONS_V1.get(key)
    if cached is not None and cached[0] == before:
        return dict(cached[1], cached_journal_validation=True)
    value = budget.check_reservation_locked(ROOT, storage["reservation"],
        expected_source_sha256=implementation["sha256"],
        expected_launcher_sha256=launcher["sha256"],
        boundary=False, required_write_bytes=0)
    after = _reservation_fingerprint_v1(storage)
    if before != after:
        raise ValueError("Authorization namespace changed during journal validation")
    _CHECKED_RESERVATIONS_V1[key] = (after, dict(value))
    return dict(value, cached_journal_validation=False)

def budget_check_v9(storage, *, first=False, boundary=False, required_write_bytes=0):
    """Use the V9 journal's existing completed-scope credit. Measure only this admitted job."""
    if (storage.get("kind") != "project_budget_v9_storage_record"
            or storage.get("active") is not True
            or storage.get("consumer_uses_v9_guard") is not True
            or storage.get("accounting_mode") != "journal_closed_scopes_credit_v1"):
        raise ValueError("Active root full-reservation storage authority required")
    if storage.get("expires_at_utc"):
        expiry = datetime.fromisoformat(storage["expires_at_utc"].replace("Z", "+00:00"))
        if expiry.tzinfo is None or datetime.now(timezone.utc) >= expiry:
            raise ValueError("Root consumer authority expired")
    refs = storage["budget_components"]
    implementation = _source_ref(refs["implementation"], "scripts/project_budget_v9.py")
    launcher = _source_ref(refs["launcher"], "scripts/launch_reserved_gpu_v6.py")
    adapter = _source_ref(refs["consumer_adapter"], SOURCE)
    own = storage["reservation"]
    if (not isinstance(storage.get("job_id"), str)
            or storage["reserved_new_bytes"] != own["reserved_bytes"]
            or own["output"] != str(budget.v4.canonical(ROOT, storage["output_directory"]))
            or any(ref not in own["consumer_bindings"] for ref in (implementation, launcher, adapter))):
        raise ValueError("Root job/output/cap/component bindings differ")
    budget.v4.integer(required_write_bytes, "planned scope growth")
    if required_write_bytes and not boundary:
        raise ValueError("Planned writes require own-scope boundary measurement")
    with budget.v4.mutex(ROOT):
        # Existing certified closures are credited by the unchanged journal core.
        # Only the active owned write scope is measured at boundaries.
        result = _checked_reservation_v1(storage, implementation, launcher)
        full_charge = budget.v4.integer(result["projected_bytes"], "certified journal project charge")
        human_limit = budget.v4.integer(result["hard_limit_bytes"], "human storage limit", 1)
        if human_limit != 1000000000000 or full_charge >= human_limit:
            raise ValueError("Certified project charge reaches human 1TB limit")
        measurement = None
        if boundary:
            paths = budget.successor_paths(ROOT, own, True)
            measurement = budget.v4.measure_scopes(ROOT, paths)
            if measurement["conservative_scope_bytes"] + required_write_bytes > own["reserved_bytes"]:
                raise ValueError("Consumer output exceeds original allowance")
        if first:
            pins = storage["initial_ledger_pins"]
            keys = {"legacy_ledger_sha256", "supplemental_ledger_sha256", "journal_tail_sha256"}
            if set(pins) != keys or any(result[k] != pins[k] for k in keys):
                raise ValueError("Initial exact historical/successor ledger pins changed")
        return dict(result, credited_v9_projection_bytes=result["projected_bytes"],
            projected_bytes=full_charge, limit_bytes=human_limit,
            closed_allowance_credit_applied_to_consumer_bytes=result["closed_allowance_credit_bytes"],
            accounting_mode="journal_closed_scopes_credit_v1",
            global_process_inspection_performed=False,
            scope_measurement=measurement,
            remaining_scope_bytes=own["reserved_bytes"] - measurement["conservative_scope_bytes"] if measurement else None,
            required_write_bytes=required_write_bytes)


def validate_storage_v9(spec, cfg, storage, *, source_ref, config_ref, runtime_ref):
    """Verify the real source call site has an admitted output, then first-read pins."""
    if (storage["job_id"] != spec["job_id"]
            or storage["output_directory"] != spec["output_directory"]
            or not 0 < storage["reserved_new_bytes"] <= cfg["output_cap_bytes"]
            or any(ref not in storage["reservation"]["consumer_bindings"]
                   for ref in (source_ref, config_ref, runtime_ref))):
        raise ValueError("Exact admitted consumer/config/runtime/output required")
    return budget_check_v9(storage, first=True, boundary=True)


class PilotOutputV9:
    """Exclusive compact output writer with an actual V9 guard call per write.

    Serialization matches the existing output format. Only files made by this
    instance may be pruned. Never alters another scope or a closed reservation.
    """
    def __init__(self, directory, storage, reader):
        self.storage = storage
        self.reader = reader
        self.path = budget.v4.canonical(ROOT, directory)
        if reader.root.resolve() != ROOT or self.path != ROOT / storage["output_directory"]:
            raise ValueError("Dedicated project output and reader required")
        if self.path.exists():
            raise ValueError("Exclusive fresh output required")
        budget_check_v9(storage, boundary=True, required_write_bytes=8192)
        self.path.mkdir(parents=False)
        self.limit = storage["reserved_new_bytes"]
        self.used = 0
        self.owned = {}

    def save(self, name, value, kind="json"):
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Flat named project output required")
        buffer = io.BytesIO()
        if kind == "json":
            from .census_tsinfer_fit_v1 import _jsonable_v1
            data = (json.dumps(_jsonable_v1(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        elif kind == "npz":
            import numpy as np
            np.savez_compressed(buffer, **value)
            data = buffer.getvalue()
        elif kind == "torch":
            import torch
            torch.save(value, buffer)
            data = buffer.getvalue()
        else:
            raise ValueError("Unknown compact output type")
        planned = len(data) + 8192
        if self.used + planned > self.limit:
            raise ValueError("Owned output counter exceeds reserved cap")
        result = budget_check_v9(self.storage, boundary=True, required_write_bytes=planned)
        available = os.statvfs(self.path)
        if planned > available.f_bavail * available.f_frsize:
            raise ValueError("Insufficient filesystem free bytes")
        path = self.path / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            identity = os.fstat(stream.fileno())
        named_identity = path.lstat()
        if (identity.st_dev, identity.st_ino) != (named_identity.st_dev, named_identity.st_ino):
            raise ValueError("Created output pathname changed during save")
        self.used += planned
        record = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()}
        self.owned[record["path"]] = (dict(record), planned, identity.st_dev, identity.st_ino)
        return record

    def discard(self, record):
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("Exact owned output reference required")
        known = self.owned.get(record["path"])
        path = budget.v4.canonical(ROOT, record["path"])
        def unchanged():
            identity = path.lstat()
            return (known is not None and known[0] == record and path.parent == self.path
                and stat.S_ISREG(identity.st_mode) and identity.st_nlink == 1
                and (identity.st_dev, identity.st_ino) == known[2:]
                and hashlib.sha256(path.read_bytes()).hexdigest() == known[0]["sha256"])
        if not unchanged():
            raise ValueError("Prune only original unchanged files created in this new output")
        budget_check_v9(self.storage, boundary=True)
        # Revalidate after the budget boundary. Cooperative scope writers remain
        # bound by exclusive ownership and closed-scope write revocation.
        if not unchanged():
            raise ValueError("Owned output identity changed before pruning")
        path.unlink()
        self.used -= known[1]
        del self.owned[record["path"]]


def source_report_v9():
    return {"kind": "project_budget_v9_consumer_source_report",
        "source": {"path": SOURCE, "sha256": budget.v4.file_bytes(ROOT, SOURCE)[1]},
        "storage_kind": "project_budget_v9_storage_record",
        "component_refs_required": ["implementation", "launcher", "consumer_adapter"],
        "scientific_model_or_data_changed": False, "live_ledger_or_scope_access": False}


if __name__ == "__main__":
    print(json.dumps(source_report_v9(), indent=2))

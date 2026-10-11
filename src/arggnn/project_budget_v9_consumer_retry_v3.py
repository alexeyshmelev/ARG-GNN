"""Reviewed-budget consumer boundary for new discovery jobs.

This is storage/output plumbing only. It changes no scientific loss, prediction
or scoring. Historical v4 consumers and completed outputs remain unchanged.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import time
import re
from datetime import datetime, timezone
from pathlib import Path

from scripts import project_budget_v9 as budget

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/project_budget_v9_consumer_retry_v3.py"


def _source_ref(ref, expected_path):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or ref["path"] != expected_path:
        raise ValueError("Exact root-pinned budget component reference required")
    if budget.v4.file_bytes(ROOT, ref["path"])[1] != ref["sha256"]:
        raise ValueError("Budget component source differs from root pin")
    return ref


def transient_process_observation(error):
    if str(error) in {
        "Process inventory changed during writer inspection",
        "Process descriptors changed during writer inspection",
    }:
        return True
    cause = error.__cause__
    return (str(error) == "Cannot inspect an own-user or recorded project process"
        and isinstance(cause, PermissionError)
        and isinstance(cause.filename, str)
        and re.fullmatch(r"/proc/[1-9][0-9]*/(?:cwd|maps|fd|fd/[0-9]+|fdinfo/[0-9]+)", cause.filename) is not None)

def budget_check_v9(storage, *, first=False, boundary=False, required_write_bytes=0):
    """Full reservation check at prep, epoch and planned writes, never every arm."""
    if (storage.get("kind") != "project_budget_v9_storage_record"
            or storage.get("active") is not True or storage.get("consumer_uses_v9_guard") is not True):
        raise ValueError("Active root V9 consumer storage authority required")
    if storage.get("expires_at_utc"):
        expiry = datetime.fromisoformat(storage["expires_at_utc"].replace("Z", "+00:00"))
        if expiry.tzinfo is None or datetime.now(timezone.utc) >= expiry:
            raise ValueError("Root V9 consumer authority expired")
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
    # A benign process exit/descriptor close can invalidate an observation.
    # Retry the entire original guard. No action follows a failed observation.
    for attempt in range(16):
        try:
            result = budget.check_reservation(ROOT, own,
                expected_source_sha256=implementation["sha256"],
                expected_launcher_sha256=launcher["sha256"],
                boundary=boundary, required_write_bytes=required_write_bytes)
            break
        except ValueError as error:
            if not transient_process_observation(error) or attempt == 15:
                raise
            # A fresh complete observation is required after every delay.
            # Total backoff is at most 27.5 seconds, no stale success is reused.
            time.sleep(min(0.25 * (2 ** attempt), 2.0))
    if attempt:
        result = dict(result, transient_process_snapshot_retries=attempt)

    if first:
        pins = storage["initial_ledger_pins"]
        keys = {"legacy_ledger_sha256", "supplemental_ledger_sha256", "journal_tail_sha256"}
        if set(pins) != keys or any(result[k] != pins[k] for k in keys):
            raise ValueError("Initial exact historical/successor ledger pins changed")
    return result


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

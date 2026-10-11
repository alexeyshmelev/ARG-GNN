#!/usr/bin/env python3
"""Append-only successor accounting for proven closed supplemental output scopes."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import copy
import hashlib
import json
import os
import stat
from pathlib import Path

from arggnn import project_budget_v4 as v4

ROOT = v4.ROOT
SOURCE_REL = "scripts/project_budget_v5.py"
LAUNCHER_REL = "scripts/launch_reserved_gpu_v2.py"
JOURNAL_REL = ".runtime/storage/project_budget_v5"
V4_SOURCE_SHA = "a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47"
V4_CONFIG_SHA = "5788f87d11825cf4fbe94a25abe056132107b155848111354e70637334e91d63"
MAX_EVENTS = 4096
MAX_JOURNAL_BYTES = 32 * 1024 * 1024

def digest(value):
    return hashlib.sha256(v4.encoded(value)).hexdigest()

def reviewed(root, source_sha, launcher_sha):
    if (v4.file_bytes(root, SOURCE_REL)[1] != source_sha
            or v4.file_bytes(root, LAUNCHER_REL)[1] != launcher_sha):
        raise ValueError("Exact reviewed v5 implementation/launcher bytes required")
    c, bindings = v4.policy(root, V4_SOURCE_SHA, V4_CONFIG_SHA)
    return c, {"implementation": v4.reference(root, SOURCE_REL),
               "launcher": v4.reference(root, LAUNCHER_REL),
               "v4_implementation": bindings["implementation"],
               "v4_config": bindings["config"],
               "human_authorization": bindings["human_authorization"]}

def successor_paths(root, value, stored=False):
    fields = {"key", "output", "auxiliary_scopes", "reserved_bytes", "consumer_bindings"}
    if stored:
        fields.add("root_admission")
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Exact v5 reservation schema required")
    request = {k: val for k, val in value.items() if k != "root_admission"}
    paths = v4.scope_spec(root, request)
    journal = root / JOURNAL_REL
    if any(v4.paths_overlap(p, journal) for p in paths):
        raise ValueError("Output overlaps successor journal")
    return paths

def check_authority(root, ref, operation, bindings):
    authority = v4.bound(root, ref, True)
    required = {"kind": "project_budget_v5_root_admission", "root_authorizer": "root_agent",
                "active": True, "operation": operation, **bindings,
                "all_budget_writers_use_scheduler_mutex": True,
                "all_new_consumers_use_v5_guard": True,
                "legacy_and_v4_new_admissions_frozen": True,
                "no_historical_record_mutation_authorized": True,
                "global_rebase_authorized": False, "GPU_authorized": False,
                "scientific_execution_authorized": False}
    if any(v4.encoded(authority.get(k)) != v4.encoded(val) for k, val in required.items()):
        raise ValueError("Exact activated v5 root authority required")
    review = v4.bound(root, authority["independent_review"], True)
    expected = {"kind": "project_budget_v5_independent_review",
                "review_role": "independent_reviewer", "verdict": "PASS",
                **bindings, "actual_apply_performed": False}
    if any(v4.encoded(review.get(k)) != v4.encoded(val) for k, val in expected.items()):
        raise ValueError("Exact v5 independent review required")
    return authority

def terminal_binding(root, closure, reservation, paths):
    required = {"key", "reservation_sha256", "completed", "completed_identity",
                "terminal_receipt", "scope_paths", "expected_scope_measurement"}
    if set(closure) != required or closure["key"] != reservation["key"]:
        raise ValueError("Exact closure request required")
    if closure["reservation_sha256"] != digest(reservation):
        raise ValueError("Closure reservation changed")
    if closure["scope_paths"] != [str(p) for p in paths]:
        raise ValueError("Closure must include every original declared scope")
    completed_path = v4.canonical(root, closure["completed"]["path"])
    if completed_path != paths[0] / "completed.json":
        raise ValueError("Completion evidence must be the output root's completed.json")
    completed = v4.bound(root, closure["completed"], True)
    identity = closure["completed_identity"]
    if not isinstance(identity, dict) or not identity or "status" not in identity:
        raise ValueError("Nonempty exact completion identity including status required")
    if not isinstance(identity["status"], str) or not identity["status"].startswith("completed"):
        raise ValueError("Successful completed status required")
    if any(v4.encoded(completed.get(k)) != v4.encoded(val) for k, val in identity.items()):
        raise ValueError("Completion identity differs")
    terminal = v4.bound(root, closure["terminal_receipt"], True)
    if (terminal.get("kind") != "project_budget_v5_terminal_receipt"
            or terminal.get("root_authorizer") != "root_agent"
            or terminal.get("reservation_sha256") != closure["reservation_sha256"]
            or terminal.get("completed") != closure["completed"]):
        raise ValueError("Terminal evidence is not bound to the completed reservation")
    pids = terminal.get("pids")
    if not isinstance(pids, list) or not 1 <= len(pids) <= 256 or len(set(pids)) != len(pids):
        raise ValueError("Bounded unique terminal process identities required")
    for pid in pids:
        v4.integer(pid, "terminal PID", 1)
    method = terminal.get("evidence_method")
    if method == "supervisor_wait":
        v4.integer(terminal.get("returncode"), "observed successful terminal returncode")
        if terminal["returncode"] != 0:
            raise ValueError("Successful supervisor wait required")
    elif method == "root_observed_absence":
        if (terminal.get("status") != "driver_completed_launcher_exit_unknown"
                or terminal.get("returncode") is not None or terminal.get("exit_status_known") is not False):
            raise ValueError("PID absence must explicitly preserve unknown launcher exit status")
    else:
        raise ValueError("Actual wait or explicit root terminal absence evidence required")
    if not isinstance(terminal.get("observed_utc"), str) or not terminal["observed_utc"]:
        raise ValueError("Terminal evidence timestamp required")
    runtime = completed.get("runtime", {})
    known = [runtime.get("pid"), runtime.get("lease", {}).get("launcher_metadata", {}).get("launcher_pid")]
    if any(pid is not None and pid not in pids for pid in known):
        raise ValueError("Completed worker/launcher missing from terminal evidence")
    if (terminal.get("recorded_pids_absent") is not True
            or terminal.get("original_lease_exclusive_acquire_release_verified") is not True
            or terminal.get("no_recorded_gpu_context") is not True
            or terminal.get("external_NFS_writers_outside_observation") is not True):
        raise ValueError("Truthful local terminal/lease/GPU attestation required")
    for field in ["process_observation", "lease_observation", "gpu_observation"]:
        v4.bound(root, terminal[field], True)
    lease_data = runtime.get("lease", {})
    lease_path = lease_data.get("lease_path")
    if (not isinstance(lease_path, str) or terminal.get("lease_path") != lease_path
            or v4.canonical(root, lease_path).parent != root / ".runtime/locks"
            or Path(lease_path).name not in {"gpu_2.lock", "gpu_3.lock", "gpu_4.lock", "gpu_5.lock"}):
        raise ValueError("Original completed GPU lease identity required")
    if terminal.get("original_lease_identity") != {
            "device": lease_data.get("lease_device"), "inode": lease_data.get("lease_inode")}:
        raise ValueError("Terminal receipt must preserve original completed lease device/inode")
    v4.integer(lease_data.get("lease_device"), "completed lease device")
    v4.integer(lease_data.get("lease_inode"), "completed lease inode", 1)
    return pids

@contextlib.contextmanager
def terminal_lease(root, closure):
    terminal = v4.bound(root, closure["terminal_receipt"], True)
    p = v4.canonical(root, terminal["lease_path"])
    before = p.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or terminal["original_lease_identity"] != {"device": before.st_dev, "inode": before.st_ino}):
        raise ValueError("Canonical original GPU mutex identity required")
    fd = os.open(p, os.O_RDWR | os.O_NOFOLLOW)
    try:
        if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Original GPU lease replaced")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)

def proc_identity(proc_root, pid):
    raw = (proc_root / str(pid) / "stat").read_text()
    end = raw.rfind(")")
    fields = raw[end + 2:].split()
    if end < 0 or len(fields) < 20:
        raise ValueError("Malformed process identity")
    return fields[0], fields[19]

def inside_scope(raw, paths):
    raw = raw.removesuffix(" (deleted)")
    p = Path(raw)
    return p.is_absolute() and any(p == scope or scope in p.parents for scope in paths)

def no_live_writers(paths, terminal_pids, *, proc_root=Path("/proc")):
    """Fail closed on unreadable processes; only local, cooperative writers are covered."""
    proc_root = Path(proc_root)
    def inventory():
        with os.scandir(proc_root) as entries:
            pids = sorted(int(e.name) for e in entries if e.name.isdecimal())
        if len(pids) > 100000:
            raise ValueError("Process inventory too large")
        return pids
    before = inventory()
    identities = {}
    uninspected_external = []
    for pid in before:
        p = proc_root / str(pid)
        try:
            state, start = proc_identity(proc_root, pid)
            identities[pid] = (state, start)
            if pid in terminal_pids and state not in {"Z", "X"}:
                raise ValueError("Recorded terminal PID is still live (or has been reused)")
            if state in {"Z", "X"}:
                continue
            if inside_scope(os.readlink(p / "cwd"), paths):
                raise ValueError("Live process cwd lies in closed output scope")
            # Keep scandir open while inspecting: its own fd appears in this process's inventory.
            with os.scandir(p / "fd") as fds:
                names = sorted(e.name for e in fds)
                for name in names:
                    fd = p / "fd" / name
                    try:
                        target = os.readlink(fd)
                        if not inside_scope(target, paths):
                            continue
                        flags = None
                        for line in (p / "fdinfo" / name).read_text().splitlines():
                            if line.startswith("flags:"):
                                flags = int(line.split()[1], 8)
                        if flags is None:
                            raise ValueError("Open scope descriptor has no flags")
                        if (flags & os.O_ACCMODE) != os.O_RDONLY or stat.S_ISDIR(fd.stat().st_mode):
                            raise ValueError("Live writable descriptor or directory handle in closed scope")
                    except FileNotFoundError:
                        raise ValueError("Process descriptors changed during writer inspection")
            for line in (p / "maps").read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and "w" in parts[1] and inside_scope(parts[5], paths):
                    raise ValueError("Live writable mapping in closed scope")
            if proc_identity(proc_root, pid)[1] != start:
                raise ValueError("Process identity changed during writer inspection")
        except (FileNotFoundError, ProcessLookupError) as exc:
            raise ValueError("Process inventory changed during writer inspection") from exc
        except PermissionError as exc:
            if pid in terminal_pids or p.stat().st_uid == os.getuid():
                raise ValueError("Cannot inspect an own-user or recorded project process") from exc
            uninspected_external.append(pid)
    if inventory() != before:
        raise ValueError("Process inventory changed during writer inspection")
    return {"local_proc_inspected": True, "processes": len(before),
            "terminal_pids": terminal_pids, "terminal_pids_live": False,
            "uninspected_other_user_pids": uninspected_external,
            "external_processes_outside_observation": True,
            "remote_writers_verified": False}

def measure(root, paths):
    """A deterministic stat manifest is compared on subsequent boundaries, with no contents decoded."""
    result = v4.measure_scopes(root, paths)
    rows = []
    def scan(p):
        before = p.lstat()
        rows.append([str(p), before.st_dev, before.st_ino, before.st_mode,
                     before.st_nlink, before.st_size, before.st_blocks,
                     before.st_mtime_ns, before.st_ctime_ns])
        if stat.S_ISDIR(before.st_mode):
            with os.scandir(p) as entries:
                names = sorted(e.name for e in entries)
            for name in names:
                scan(v4.canonical(root, p / name))
    for p in paths:
        if p.exists():
            scan(p)
        else:
            rows.append([str(p), "absent"])
    # The guarded v4 scan validates the complete namespace before and after inventory.
    after = v4.measure_scopes(root, paths)
    if result != after:
        raise ValueError("Scope changed during closure measurement")
    fields = lambda s: [s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size,
                        s.st_blocks, s.st_mtime_ns, s.st_ctime_ns]
    for row in rows:
        p = Path(row[0])
        if row[1] == "absent":
            if p.exists():
                raise ValueError("Absent closure scope appeared")
        elif fields(p.lstat()) != row[1:]:
            raise ValueError("Scope stat manifest changed")
    return {**result, "stat_manifest_sha256": digest(rows)}

def reconcile(view, old_reservations, new_reservations, closed):
    """No baseline/legacy subtraction; a unique closed key changes R to A exactly once."""
    all_reservations = {**old_reservations, **new_reservations}
    if set(old_reservations) & set(new_reservations):
        raise ValueError("Historical reservation key reused")
    if not set(closed) <= set(all_reservations):
        raise ValueError("Closed key is absent")
    credits = 0
    for key, closure in closed.items():
        reservation = all_reservations[key]
        if closure["reservation_sha256"] != digest(reservation):
            raise ValueError("Closed reservation drift")
        retained = v4.integer(closure["measurement"]["conservative_scope_bytes"], "closed retained bytes")
        allowance = v4.integer(reservation["reserved_bytes"], "closed allowance", 1)
        if retained > allowance:
            raise ValueError("Closed scope exceeds historical allowance")
        credits += allowance - retained
    added = sum(v4.integer(r["reserved_bytes"], "new allowance", 1) for r in new_reservations.values())
    projected = view["projected_bytes"] + added - credits
    if projected >= view["limit_bytes"] or projected >= view["hard_limit_bytes"]:
        raise ValueError("Reconciled project scheduling ceiling exceeded")
    return {**view, "v4_projected_bytes": view["projected_bytes"], "v5_reserved_bytes": added,
            "closed_allowance_credit_bytes": credits, "projected_bytes": projected,
            "accounting_is_conservative_not_global_physical_measurement": True,
            "valid_under_cooperative_project_write_authority_contract": True,
            "external_NFS_writers_outside_observation": True}

def load_journal(root, bindings):
    directory = v4.canonical(root, JOURNAL_REL)
    if not directory.exists():
        return [], None
    before = directory.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("Journal must be a canonical directory")
    with os.scandir(directory) as entries:
        names = sorted(e.name for e in entries)
    if len(names) > MAX_EVENTS:
        raise ValueError("Journal event count exceeds cap")
    total, events, tail = 0, [], None
    for seq, name in enumerate(names, 1):
        if name != f"{seq:06d}.json":
            raise ValueError("Journal has gaps or unknown entries")
        raw, sha = v4.file_bytes(root, directory / name)
        total += len(raw)
        if total > MAX_JOURNAL_BYTES:
            raise ValueError("Journal byte cap exceeded")
        event = v4.parse(raw)
        required = {"schema", "root", "sequence", "previous_sha256", "bindings",
                    "operation", "authority", "value"}
        if (not isinstance(event, dict) or set(event) != required
                or event["schema"] != "project_budget_v5_event" or event["root"] != str(root)
                or type(event["sequence"]) is not int or event["sequence"] != seq
                or event["previous_sha256"] != tail or event["bindings"] != bindings
                or event["operation"] not in {"close", "reserve"}):
            raise ValueError("Append-only event chain/header drift")
        events.append(event)
        tail = sha
    after = directory.lstat()
    if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Journal directory changed during read")
    return events, tail

def state(root, c, bindings, *, verify_closed=True):
    legacy, old, view = v4.current(root, c, {
        "implementation": bindings["v4_implementation"], "config": bindings["v4_config"],
        "human_authorization": bindings["human_authorization"]})
    events, tail = load_journal(root, bindings)
    if events:
        activation = v4.bound(root, events[0]["authority"], True).get("expected_before", {})
        if any(activation.get(k) != view[k] for k in
               ["legacy_ledger_sha256", "supplemental_ledger_sha256"]):
            raise ValueError("Historical ledgers changed after successor admission barrier")
    new, closed = {}, {}
    occupied = [v4.canonical(root, r["output"]) for r in legacy["jobs"].values()]
    for reservation in old["reservations"].values():
        occupied += v4.scope_spec(root, reservation, True)
    for event in events:
        authority = check_authority(root, event["authority"], event["operation"], bindings)
        if authority.get("expected_before", {}).get("journal_tail_sha256") != event["previous_sha256"]:
            raise ValueError("Event predecessor not bound by immutable root authority")
        value = event["value"]
        if event["operation"] == "reserve":
            if len(new) + len(old["reservations"]) >= 4096:
                raise ValueError("Combined reservation cap exceeded")
            paths = successor_paths(root, value, True)
            request = {k: val for k, val in value.items() if k != "root_admission"}
            if (value["root_admission"] != event["authority"]
                    or authority.get("reservation") != request
                    or value["key"] in old["reservations"] or value["key"] in new
                    or value["key"] in legacy["jobs"]
                    or any(v4.paths_overlap(a, b) for a in paths for b in occupied)):
                raise ValueError("Successor reservation authority/key/scope drift")
            new[value["key"]] = value
            occupied += paths
        else:
            if set(value) != {"request", "reservation_sha256", "measurement", "writer_check"}:
                raise ValueError("Exact closure event value required")
            request = value["request"]
            key = request["key"]
            own = old["reservations"].get(key) or new.get(key)
            if own is None or key in closed or authority.get("closure") != request:
                raise ValueError("Duplicate/unknown closure or changed authority")
            if any(authority.get(k) is not True for k in [
                    "scope_write_authority_revoked_for_all_cooperative_project_jobs",
                    "closed_scope_future_writes_and_relaunch_revoked",
                    "external_NFS_writers_outside_observation"]):
                raise ValueError("Closed scope future write prohibition required")
            paths = successor_paths(root, own, True) if key in new else v4.scope_spec(root, own, True)
            pids = terminal_binding(root, request, own, paths)
            if (value["reservation_sha256"] != request["reservation_sha256"]
                    or value["measurement"] != request["expected_scope_measurement"]):
                raise ValueError("Closure digest/measurement differs from immutable root authority")
            if verify_closed:
                no_live_writers(paths, pids)
                if measure(root, paths) != value["measurement"]:
                    raise ValueError("Closed output changed; full allowance must be restored before scheduling")
            closed[key] = value
    account = reconcile(view, old["reservations"], new, closed)
    return legacy, old, new, closed, {**account, "journal_tail_sha256": tail,
                                    "journal_events": len(events)}, events

def pins(view):
    return {k: view[k] for k in ["legacy_ledger_sha256", "supplemental_ledger_sha256",
                                 "journal_tail_sha256"]}

def append_event(root, event, before, c, bindings):
    if before["journal_events"] >= MAX_EVENTS:
        raise ValueError("Journal event cap exceeded before write")
    data = v4.encoded(event) + b"\n"
    existing, tail = load_journal(root, bindings)
    total = sum(len(v4.file_bytes(root, root / JOURNAL_REL / f"{i:06d}.json")[0])
                for i in range(1, len(existing) + 1))
    if total + len(data) > MAX_JOURNAL_BYTES:
        raise ValueError("Journal byte cap exceeded before write")
    if tail != before["journal_tail_sha256"] or len(existing) != before["journal_events"]:
        raise ValueError("Journal changed under mutex")
    if v4.read_json(root, v4.LEGACY)[1] != before["legacy_ledger_sha256"]:
        raise ValueError("Legacy changed under mutex")
    if v4.read_json(root, v4.SUPPLEMENTAL)[1] != before["supplemental_ledger_sha256"]:
        raise ValueError("Historical supplemental ledger changed under mutex")
    directory = v4.canonical(root, JOURNAL_REL)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    p = v4.canonical(root, directory / f"{event['sequence']:06d}.json")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    # Any interruption leaves an immutable partial final event: fail closed, no rewrite/rollback.
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    written, sha = v4.file_bytes(root, p)
    if written != data:
        raise ValueError("Append uncertainty; preserve evidence, do not rewrite journal")
    if (v4.read_json(root, v4.LEGACY)[1] != before["legacy_ledger_sha256"]
            or v4.read_json(root, v4.SUPPLEMENTAL)[1] != before["supplemental_ledger_sha256"]):
        raise ValueError("Post-append historical ledger uncertainty; preserve evidence")
    return sha

def apply(root, authority_ref, *, expected_source_sha256, expected_launcher_sha256):
    root = Path(root)
    with v4.mutex(root):
        c, bindings = reviewed(root, expected_source_sha256, expected_launcher_sha256)
        operation = v4.bound(root, authority_ref, True).get("operation")
        if operation not in {"close", "reserve"}:
            raise ValueError("Exact close/reserve operation required")
        authority = check_authority(root, authority_ref, operation, bindings)
        legacy, old, new, closed, before, events = state(root, c, bindings)
        if authority.get("expected_before") != pins(before):
            raise ValueError("Exact fresh historical ledgers/journal pins required")
        if operation == "reserve":
            request = authority["reservation"]
            paths = successor_paths(root, request)
            occupied = [v4.canonical(root, r["output"]) for r in legacy["jobs"].values()]
            for reservation in [*old["reservations"].values(), *new.values()]:
                occupied += successor_paths(root, reservation, True) if reservation["key"] in new else v4.scope_spec(root, reservation, True)
            if (len(old["reservations"]) + len(new) >= 4096
                    or request["key"] in old["reservations"] or request["key"] in new
                    or request["key"] in legacy["jobs"]
                    or any(p.exists() for p in paths)
                    or any(v4.paths_overlap(a, b) for a in paths for b in occupied)):
                raise ValueError("New successor key/scopes must be fresh and exclusive")
            value = {**copy.deepcopy(request), "root_admission": copy.deepcopy(authority_ref)}
            projected_new = {**new, request["key"]: value}
            base = {**before, "projected_bytes": before["v4_projected_bytes"]}
            after = reconcile(base, old["reservations"], projected_new, closed)
        else:
            if any(authority.get(k) is not True for k in [
                    "scope_write_authority_revoked_for_all_cooperative_project_jobs",
                    "closed_scope_future_writes_and_relaunch_revoked",
                    "external_NFS_writers_outside_observation"]):
                raise ValueError("Closure requires exclusive local ownership and future write revocation")
            request = authority["closure"]
            own = old["reservations"].get(request["key"]) or new.get(request["key"])
            if own is None or request["key"] in closed:
                raise ValueError("Unknown/already closed reservation")
            paths = successor_paths(root, own, True) if request["key"] in new else v4.scope_spec(root, own, True)
            pids = terminal_binding(root, request, own, paths)
            with terminal_lease(root, request):
                no_live_writers(paths, pids)
                measurement = measure(root, paths)
                writer_check = no_live_writers(paths, pids)
                if measure(root, paths) != measurement:
                    raise ValueError("Closed scope changed across terminal/writer checks")
            if measurement != request["expected_scope_measurement"]:
                raise ValueError("Root-pinned scoped measurement changed before closure")
            value = {"request": copy.deepcopy(request), "reservation_sha256": digest(own),
                     "measurement": measurement, "writer_check": writer_check}
            base = {**before, "projected_bytes": before["v4_projected_bytes"]}
            after = reconcile(base, old["reservations"], new, {**closed, request["key"]: value})
        reviewed(root, expected_source_sha256, expected_launcher_sha256)
        check_authority(root, authority_ref, operation, bindings)
        event = {"schema": "project_budget_v5_event", "root": str(root),
                 "sequence": before["journal_events"] + 1,
                 "previous_sha256": before["journal_tail_sha256"], "bindings": bindings,
                 "operation": operation, "authority": copy.deepcopy(authority_ref), "value": value}
        sha = append_event(root, event, before, c, bindings)
        return {"kind": "project_budget_v5_apply_receipt", "operation": operation,
                "before": before, "projected_bytes": after["projected_bytes"],
                "journal_tail_sha256": sha, "value": value,
                "historical_records_mutated": False, "global_scan_performed": False,
                "GPU_used": False, "scientific_execution_performed": False}

def inspect(root, *, expected_source_sha256, expected_launcher_sha256):
    with v4.mutex(Path(root)):
        c, bindings = reviewed(Path(root), expected_source_sha256, expected_launcher_sha256)
        return state(Path(root), c, bindings)[4]

def check_reservation_locked(root, reservation, *, expected_source_sha256,
                             expected_launcher_sha256, boundary=False, required_write_bytes=0):
    """Caller MUST hold the canonical scheduler mutex; launcher uses this before GPU allocation."""
    root = Path(root)
    c, bindings = reviewed(root, expected_source_sha256, expected_launcher_sha256)
    _, _, new, closed, view, _ = state(root, c, bindings, verify_closed=boundary)
    own = new.get(reservation.get("key"))
    if own is None or v4.encoded(own) != v4.encoded(reservation) or own["key"] in closed:
        raise ValueError("Exact active successor reservation required; closed scopes cannot relaunch")
    paths = successor_paths(root, own, True)
    v4.integer(required_write_bytes, "planned scope growth")
    if required_write_bytes and not boundary:
        raise ValueError("Planned writes require boundary measurement")
    measurement = v4.measure_scopes(root, paths) if boundary else None
    if measurement and measurement["conservative_scope_bytes"] + required_write_bytes > own["reserved_bytes"]:
        raise ValueError("Consumer output exceeds original allowance")
    return {**view, "reservation": copy.deepcopy(own), "scope_measurement": measurement,
            "remaining_scope_bytes": own["reserved_bytes"] - measurement["conservative_scope_bytes"] if measurement else None,
            "required_write_bytes": required_write_bytes}

def check_reservation(root, reservation, **kwargs):
    with v4.mutex(Path(root)):
        return check_reservation_locked(root, reservation, **kwargs)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true")
    group.add_argument("--inspect", action="store_true")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-launcher-sha256")
    parser.add_argument("--admission")
    parser.add_argument("--expected-admission-sha256")
    args = parser.parse_args(argv)
    if not (args.apply or args.inspect):
        print(json.dumps({"kind": "project_budget_v5_source_report",
                          "implementation": v4.reference(ROOT, SOURCE_REL),
                          "launcher": v4.reference(ROOT, LAUNCHER_REL),
                          "historical_v4_source_sha256": V4_SOURCE_SHA,
                          "historical_v4_config_sha256": V4_CONFIG_SHA,
                          "live_ledger_scope_process_access": False,
                          "actual_apply_performed": False}, indent=2))
        return 0
    if not args.expected_source_sha256 or not args.expected_launcher_sha256:
        raise ValueError("Reviewed source/launcher pins required")
    kwargs = {"expected_source_sha256": args.expected_source_sha256,
              "expected_launcher_sha256": args.expected_launcher_sha256}
    if args.inspect:
        result = inspect(ROOT, **kwargs)
    else:
        if not args.admission or not args.expected_admission_sha256:
            raise ValueError("Exact activated root admission required")
        result = apply(ROOT, {"path": args.admission, "sha256": args.expected_admission_sha256}, **kwargs)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

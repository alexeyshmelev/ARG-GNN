"""One fake-core metadata fixture: no journal, scope, lock or process access."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arggnn import project_storage_closed_scopes_v1 as new
from arggnn import project_storage_full_reservation_v1 as old

OUT = Path("research/reviews/project_storage_closed_scopes_v1_review_author")
ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
IMPL = "scripts/project_budget_v9.py"
LAUNCH = "scripts/launch_reserved_gpu_v6.py"
pins = {IMPL: "1"*64, LAUNCH: "2"*64, new.SOURCE: "3"*64, old.SOURCE: "4"*64}
component_refs = {
    "implementation": {"path": IMPL, "sha256": pins[IMPL]},
    "launcher": {"path": LAUNCH, "sha256": pins[LAUNCH]},
    "consumer_adapter": {"path": new.SOURCE, "sha256": pins[new.SOURCE]},
}
callsite = [{"path": "fixture/source.py", "sha256": "5"*64},
            {"path": "fixture/config.json", "sha256": "6"*64},
            {"path": "fixture/runtime.json", "sha256": "7"*64}]
scope = ROOT/"data/expanded_full_panel_v1/ap_expanded_train_000"
reservation = {
    "key": "fixture/H038/train000", "output": str(scope),
    "auxiliary_scopes": [], "reserved_bytes": 4_000_000_000,
    "consumer_bindings": list(component_refs.values()) + callsite,
    "root_admission": {"path": "fixture/admission.json", "sha256": "8"*64},
}
ledger_pins = {"legacy_ledger_sha256": "9"*64,
               "supplemental_ledger_sha256": "a"*64,
               "journal_tail_sha256": "b"*64}
storage = {
    "kind": "project_budget_v9_storage_record", "active": True,
    "consumer_uses_v9_guard": True, "accounting_mode": "journal_closed_scopes_credit_v1",
    "expires_at_utc": "2030-01-01T00:00:00+00:00",
    "budget_components": component_refs, "reservation": reservation,
    "job_id": "fixture/H038/train000",
    "reserved_new_bytes": 4_000_000_000,
    "output_directory": str(scope.relative_to(ROOT)),
    "initial_ledger_pins": ledger_pins,
}
fixture = {
    "v4_projected_bytes": 960_000_000_000, "v9_reserved_bytes": 80_000_000_000,
    "closed_allowance_credit_bytes": 140_000_000_000,
    "projected_bytes": 900_000_000_000, "hard_limit_bytes": 1_000_000_000_000,
    **ledger_pins,
}
calls = []
measurement = {"conservative_scope_bytes": 3_000_000_000,
               "scope_paths": [str(scope)], "stat_only": True}
def integer(v, name, minimum=0):
    if type(v) is not int or v < minimum:
        raise ValueError("Invalid fake integer: " + name)
    return v
def file_bytes(root, path):
    calls.append({"operation": "fake_source_pin", "path": path})
    return b"", pins[path]
def successor_paths(root, own, stored):
    assert root == ROOT and own == reservation and stored is True
    return [scope]
def measure_scopes(root, paths):
    assert root == ROOT and paths == [scope]
    calls.append({"operation": "fake_owned_scope_measurement", "paths": [str(p) for p in paths]})
    return copy.deepcopy(measurement)
def check_locked(root, own, **kwargs):
    assert root == ROOT and own == reservation
    assert kwargs == {"expected_source_sha256": pins[IMPL],
                      "expected_launcher_sha256": pins[LAUNCH],
                      "boundary": False, "required_write_bytes": 0}
    calls.append({"operation": "fake_existing_core", **kwargs})
    return copy.deepcopy(fixture)
fake = SimpleNamespace(
    v4=SimpleNamespace(
        file_bytes=file_bytes,
        canonical=lambda root, path: (root/Path(path)).resolve(),
        integer=integer, mutex=lambda root: nullcontext(),
        measure_scopes=measure_scopes,
    ),
    check_reservation_locked=check_locked, successor_paths=successor_paths,
)
def rejected(action, contains):
    try:
        action()
    except ValueError as error:
        assert contains in str(error), str(error)
        return str(error)
    raise AssertionError("Expected rejection was absent")

before = copy.deepcopy(storage)
with patch.object(new, "budget", fake), patch.object(old, "budget", fake):
    accepted = new.validate_storage_v9(
        {"job_id": storage["job_id"], "output_directory": storage["output_directory"]},
        {"output_cap_bytes": 4_000_000_000}, storage,
        source_ref=callsite[0], config_ref=callsite[1], runtime_ref=callsite[2],
    )
    assert accepted["projected_bytes"] == 900_000_000_000
    assert accepted["closed_allowance_credit_applied_to_consumer_bytes"] == 140_000_000_000
    assert accepted["remaining_scope_bytes"] == 1_000_000_000
    assert accepted["scope_measurement"]["scope_paths"] == [str(scope)]
    assert storage == before

    old_storage = copy.deepcopy(storage)
    old_storage["accounting_mode"] = "full_reserved_no_closed_scope_credit"
    old_storage["budget_components"]["consumer_adapter"] = {
        "path": old.SOURCE, "sha256": pins[old.SOURCE]}
    old_storage["reservation"]["consumer_bindings"] = [
        ref for ref in old_storage["reservation"]["consumer_bindings"] if ref["path"] != new.SOURCE
    ] + [old_storage["budget_components"]["consumer_adapter"]]
    # The exact reservation expected by the fake original core is rebound here.
    saved_reservation = reservation
    reservation = old_storage["reservation"]
    old_error = rejected(lambda: old.budget_check_v9(old_storage), "Full original reservation charge")
    reservation = saved_reservation

    cap_error = rejected(
        lambda: new.budget_check_v9(storage, boundary=True, required_write_bytes=1_000_000_001),
        "Consumer output exceeds original allowance",
    )
    missing = copy.deepcopy(storage)
    missing["reservation"]["consumer_bindings"].remove(component_refs["launcher"])
    missing_error = rejected(lambda: new.budget_check_v9(missing), "component bindings differ")
    changed = copy.deepcopy(storage)
    changed["budget_components"]["consumer_adapter"]["sha256"] = "c"*64
    changed_error = rejected(lambda: new.budget_check_v9(changed), "source differs from root pin")

# Existing writer/callsite/source guards are byte-equivalent in their AST.
def definitions(path):
    tree = ast.parse(path.read_text())
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
old_path = ROOT/old.SOURCE
new_path = ROOT/new.SOURCE
a,b = definitions(old_path),definitions(new_path)
preserved = {name: a[name] == b[name]
             for name in ["_source_ref", "validate_storage_v9", "PilotOutputV9", "source_report_v9"]}
assert all(preserved.values())
receipt = {
    "schema": "closed_scope_storage_adapter_fake_core_metadata_check_v1",
    "fixture_is_not_actual_project_accounting": True,
    "fake_core_return": fixture,
    "accepted_charge": {
        k: accepted[k] for k in ["projected_bytes", "closed_allowance_credit_applied_to_consumer_bytes",
                                "remaining_scope_bytes", "scope_measurement", "accounting_mode"]},
    "original_full_charge_from_same_fixture": 1_040_000_000_000,
    "rejections": {"unchanged_original_full_adapter": old_error,
                   "original_per_parent_cap": cap_error,
                   "missing_component_binding": missing_error,
                   "changed_component_source_pin": changed_error},
    "storage_metadata_and_component_bindings_unmodified": storage == before,
    "same_AST_definitions": preserved,
    "fake_operations": calls,
    "real_journal_mutation_or_read": False,
    "real_scope_or_disk_or_process_scan": False,
    "real_scheduler_mutex_used": False,
    "GPU_or_actual_data_run": False,
    "source_refs": [{"path": p, "sha256": hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}
                    for p in [new.SOURCE, old.SOURCE, "scripts/project_budget_v9.py"]],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"fake_metadata_check.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="fake_operations"}, indent=2))

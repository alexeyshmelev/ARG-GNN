"""Cached-adapter hit/miss probe using fake stat/core/scope operations only."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arggnn import project_storage_closed_scopes_v1 as adapter
from arggnn import project_storage_full_reservation_v1 as original

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OUT = Path("research/reviews/project_storage_closed_scopes_v1_review_author")
EXPECTED = "974bcf342965abb7e7e062547c68123de9bb9414bf9b1f97fd962d1488d3fdd7"
assert hashlib.sha256((ROOT/adapter.SOURCE).read_bytes()).hexdigest() == EXPECTED
production_budget = adapter.budget
impl, launch = production_budget.SOURCE_REL, production_budget.LAUNCHER_REL
legacy, supplemental, journal = (
    production_budget.v4.LEGACY, production_budget.v4.SUPPLEMENTAL,
    production_budget.JOURNAL_REL,
)
pins = {impl: "1"*64, launch: "2"*64, adapter.SOURCE: "3"*64}
refs = {"implementation": {"path": impl, "sha256": pins[impl]},
        "launcher": {"path": launch, "sha256": pins[launch]},
        "consumer_adapter": {"path": adapter.SOURCE, "sha256": pins[adapter.SOURCE]}}
callsite = [{"path": "fixture/source.py", "sha256": "4"*64},
            {"path": "fixture/config.json", "sha256": "5"*64},
            {"path": "fixture/runtime.json", "sha256": "6"*64}]
scope = ROOT/"data/expanded_full_panel_v1/ap_expanded_train_000"
own = {"key": "fixture/H038/parent", "output": str(scope),
       "auxiliary_scopes": [], "reserved_bytes": 4_000_000_000,
       "consumer_bindings": list(refs.values())+callsite,
       "root_admission": {"path": "fixture/admission.json", "sha256": "7"*64}}
ledger_pins = {"legacy_ledger_sha256": "8"*64,
               "supplemental_ledger_sha256": "9"*64,
               "journal_tail_sha256": "a"*64}
storage = {"kind": "project_budget_v9_storage_record", "active": True,
           "consumer_uses_v9_guard": True,
           "accounting_mode": "journal_closed_scopes_credit_v1",
           "expires_at_utc": "2030-01-01T00:00:00+00:00",
           "budget_components": refs, "reservation": own, "job_id": "fixture/H038/parent",
           "reserved_new_bytes": own["reserved_bytes"],
           "output_directory": str(scope.relative_to(ROOT)),
           "initial_ledger_pins": ledger_pins}
snapshot = copy.deepcopy(storage)
core_value = {"v4_projected_bytes": 960_000_000_000,
              "v9_reserved_bytes": 80_000_000_000,
              "closed_allowance_credit_bytes": 140_000_000_000,
              "projected_bytes": 900_000_000_000,
              "hard_limit_bytes": 1_000_000_000_000, **ledger_pins}
metadata_paths = list(dict.fromkeys(
    [legacy, supplemental, journal, impl, launch, "configs/project_budget_v4.json"]
    +[r["path"]for r in own["consumer_bindings"]]))
stats = {p: {"st_dev": 1, "st_ino": i+1, "st_mode": 0o100600,
             "st_nlink": 1, "st_size": 100, "st_mtime_ns": 10, "st_ctime_ns": 10}
         for i,p in enumerate(metadata_paths)}
stats[journal]["st_mode"] = 0o40700
counts = {"core": 0, "scope": 0, "mutex_entries": 0, "source_pin": 0}
state = {"mutex_depth": 0, "core_error": None, "race_path": None}
log = []
class FakePath:
    def __init__(self, relative):
        self.relative = str(relative)
    def __str__(self):
        return str(ROOT/self.relative)
    def lstat(self):
        assert state["mutex_depth"] == 1
        assert self.relative in stats, "Unexpected real/global path"
        log.append({"operation": "fake_lstat", "path": self.relative})
        return SimpleNamespace(**stats[self.relative])
def canonical(root, relative):
    assert root == ROOT
    p = Path(relative)
    return FakePath(str(p.relative_to(ROOT)) if p.is_absolute() else str(p))
def integer(v, name, minimum=0):
    if type(v) is not int or v < minimum:
        raise ValueError(name)
    return v
@contextmanager
def mutex(root):
    assert root == ROOT and state["mutex_depth"] == 0
    counts["mutex_entries"] += 1
    state["mutex_depth"] = 1
    try:
        yield
    finally:
        state["mutex_depth"] = 0
def file_bytes(root, relative):
    counts["source_pin"] += 1
    return b"", pins[relative]
def core(root, reservation, **kwargs):
    assert state["mutex_depth"] == 1 and root == ROOT and reservation == own
    assert kwargs == {"expected_source_sha256": pins[impl],
                      "expected_launcher_sha256": pins[launch],
                      "boundary": False, "required_write_bytes": 0}
    counts["core"] += 1
    if state["core_error"]:
        raise ValueError(state["core_error"])
    if state["race_path"]:
        stats[state["race_path"]]["st_ctime_ns"] += 1
        state["race_path"] = None
    return copy.deepcopy(core_value)
def successor_paths(root, reservation, stored):
    assert root == ROOT and reservation == own and stored is True
    return [scope]
def measure_scopes(root, paths):
    assert state["mutex_depth"] == 1 and root == ROOT and paths == [scope]
    counts["scope"] += 1
    return {"conservative_scope_bytes": 3_000_000_000, "scope_paths": [str(scope)],
            "stat_only": True}
fake = SimpleNamespace(
    SOURCE_REL=impl, LAUNCHER_REL=launch, JOURNAL_REL=journal,
    digest=lambda value: hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),
    v4=SimpleNamespace(LEGACY=legacy, SUPPLEMENTAL=supplemental,
                       canonical=canonical, integer=integer, mutex=mutex,
                       file_bytes=file_bytes, measure_scopes=measure_scopes),
    check_reservation_locked=core, successor_paths=successor_paths,
)
def reject(action, expected):
    try:
        action()
    except ValueError as error:
        assert expected in str(error), str(error)
        return str(error)
    raise AssertionError("Expected rejection absent")

observations=[]
with patch.object(adapter, "budget", fake), patch.object(adapter, "_CHECKED_RESERVATIONS_V1", {}):
    first=adapter.validate_storage_v9(
        {"job_id":storage["job_id"],"output_directory":storage["output_directory"]},
        {"output_cap_bytes":4_000_000_000}, storage,
        source_ref=callsite[0],config_ref=callsite[1],runtime_ref=callsite[2])
    assert first["cached_journal_validation"] is False and counts["core"]==1 and counts["scope"]==1
    second=adapter.budget_check_v9(storage,boundary=True)
    assert second["cached_journal_validation"] is True and counts["core"]==1 and counts["scope"]==2
    observations.append({"case":"unchanged_metadata_cache_hit","core_replays":1,
                         "own_scope_measurements":2,"charge":second["projected_bytes"]})

    stats["fixture/source.py"]["st_mtime_ns"] += 1
    third=adapter.budget_check_v9(storage,boundary=True)
    assert third["cached_journal_validation"] is False and counts["core"]==2
    observations.append({"case":"consumer_metadata_change_miss","core_replays":2})

    stats[journal]["st_ctime_ns"] += 1
    core_value["projected_bytes"] = 910_000_000_000
    core_value["closed_allowance_credit_bytes"] = 130_000_000_000
    fourth=adapter.budget_check_v9(storage,boundary=True)
    assert fourth["cached_journal_validation"] is False and counts["core"]==3
    assert fourth["projected_bytes"]==910_000_000_000
    observations.append({"case":"journal_directory_change_miss_refreshes_charge",
                         "core_replays":3,"charge":fourth["projected_bytes"]})

    stats[legacy]["st_mtime_ns"] += 1
    fifth=adapter.budget_check_v9(storage,boundary=True)
    assert fifth["cached_journal_validation"] is False and counts["core"]==4
    observations.append({"case":"fixed_ledger_metadata_change_miss","core_replays":4})

    rejections={}
    before_core=counts["core"];before_scope=counts["scope"]
    rejections["own_cap_on_cache_hit"]=reject(
        lambda:adapter.budget_check_v9(storage,boundary=True,required_write_bytes=1_000_000_001),
        "Consumer output exceeds original allowance")
    assert counts["core"]==before_core and counts["scope"]==before_scope+1

    missing=copy.deepcopy(storage)
    missing["reservation"]["consumer_bindings"].remove(refs["launcher"])
    rejections["component_binding_on_warm_cache"]=reject(
        lambda:adapter.budget_check_v9(missing),"component bindings differ")
    changed=copy.deepcopy(storage)
    changed["budget_components"]["consumer_adapter"]["sha256"]="b"*64
    rejections["source_pin_on_warm_cache"]=reject(
        lambda:adapter.budget_check_v9(changed),"source differs from root pin")

    stats[journal]["st_mtime_ns"] += 1
    state["core_error"]="Exact active successor reservation required; closed scopes cannot relaunch"
    rejections["core_rejection_on_cache_miss"]=reject(
        lambda:adapter.budget_check_v9(storage,boundary=True),"closed scopes cannot relaunch")
    assert counts["core"]==5
    state["core_error"]=None

    state["race_path"]="fixture/runtime.json"
    rejections["metadata_changes_during_replay"]=reject(
        lambda:adapter.budget_check_v9(storage,boundary=True),
        "Authorization namespace changed during journal validation")
    assert counts["core"]==6

assert storage==snapshot
def definitions(path):
    tree=ast.parse(path.read_text())
    return {n.name:ast.dump(n,include_attributes=False)for n in tree.body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a=definitions(ROOT/original.SOURCE);b=definitions(ROOT/adapter.SOURCE)
preserved={name:a[name]==b[name]for name in [
    "_source_ref","validate_storage_v9","PilotOutputV9","source_report_v9"]}
assert all(preserved.values())
receipt={"schema":"closed_scopes_cache_fake_metadata_fixture_v1",
         "source_sha256":EXPECTED,"all_checks_passed":True,
         "fixture_values_not_actual_project_accounting":True,
         "observations":observations,"rejections":rejections,
         "fake_counts":counts,"fingerprinted_fake_paths":metadata_paths,
         "stat_fingerprints_only_under_fake_mutex":True,
         "source_pins_and_component_guards_checked_before_each_cache_lookup":True,
         "active_scope_measurement_still_per_write":True,
         "storage_metadata_unmodified":storage==snapshot,
         "original_API_writer_guard_AST_preserved":preserved,
         "real_journal_read_or_mutation":False,
         "real_scope_global_disk_FD_or_process_scan":False,
         "real_scheduler_mutex":False,"GPU_or_actual_data":False,
         "qualifications":["append-only journal and authority/dependency immutability",
                           "metadata fingerprint does not hash journal event contents"]}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"fake_cache_metadata_check.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))

"""Independent storage-only v3 helper/AST checks; fake ledgers only."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
AUDIT = ROOT / "research/reviews/biological_paths_tsinfer_v1_source_audit/storage_revision_v3"
spec = importlib.util.spec_from_file_location("v3_author_handmade_helpers", ROOT / "tests/test_biological_paths_tsinfer_v1.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
m = helpers.m


def rewrite_charge(receipt, path, charge):
    path.write_text(json.dumps(charge))
    receipt["installation_charge"]["sha256"] = m.sha(path)


def test_scientific_contract_and_native_pipeline_body_are_unchanged_from_v2():
    old = ast.parse((AUDIT / "previous_script_v2.py").read_text())
    new = ast.parse((ROOT / "scripts/infer_biological_paths_tsinfer_v1.py").read_text())
    before = {n.name:n for n in old.body if isinstance(n, ast.FunctionDef)}
    after = {n.name:n for n in new.body if isinstance(n, ast.FunctionDef)}
    for name in ["source_only", "validate_roster", "parse_vcf", "grid_indexes", "load_assets",
                 "check_samples", "check_genotypes", "strict_times", "connected_identity",
                 "require_identity", "sanitize_clip", "infer_parent", "handmade_assets", "qualification_summary"]:
        assert ast.dump(before[name]) == ast.dump(after[name]), name
    # Only the nested storage journaling wrapper differs within the pipeline.
    for tree in [before["pipeline"], after["pipeline"]]:
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == "stage")]
    assert ast.dump(before["pipeline"]) == ast.dump(after["pipeline"])
    prior = json.loads((AUDIT / "previous_biological_paths_tsinfer_v1.json").read_text())
    current = json.loads((ROOT / "configs/biological_paths_tsinfer_v1.json").read_text())
    prior.pop("status"); prior.pop("max_added_bytes")
    current.pop("status"); current.pop("storage_accounting")
    assert prior == current


def test_later_jobs_and_controllers_are_accounted_without_ledger_mutation(tmp_path, monkeypatch):
    c, receipt, runtime, path, state, *_ = helpers.storage_fixture(tmp_path, monkeypatch)
    accounting = m.StorageAccounting(c, receipt, runtime)
    baseline = accounting.check()["projected_bytes"]
    state["controllers"]["another_controller"] = {"reserved_bytes": 2000}
    state["jobs"]["another_job"] = {"reserved_bytes": 1000, "output": str(tmp_path / "another_job")}
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    result = accounting.check()
    assert result["projected_bytes"] == baseline + 3000
    assert result["persistent_reservations"] == 6
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["missing_job", "different_output"])
def test_reservation_identity_is_required_not_only_an8GB_numeric_budget(tmp_path, monkeypatch, change):
    c, receipt, runtime, path, state, *_ = helpers.storage_fixture(tmp_path, monkeypatch)
    key = c["storage_accounting"]["qualify_reservation_key"]
    if change == "missing_job":
        state["jobs"].pop(key)
    else:
        state["jobs"][key]["output"] = str(tmp_path / "different_runtime")
    path.write_text(json.dumps(state))
    receipt["ledger_sha256_at_admission"] = m.sha(path)
    with pytest.raises(ValueError, match="exact existing"):
        m.StorageAccounting(c, receipt, runtime)


def test_unlocked_or_noncooperating_ledger_change_during_read_is_rejected(tmp_path, monkeypatch):
    c, receipt, runtime, path, state, *_ = helpers.storage_fixture(tmp_path, monkeypatch)
    pool = m.storage_module(c)
    original = pool.StorageLedger._load
    def changed(self):
        result = original(self)
        newer = copy.deepcopy(state)
        newer["controllers"]["unexpected_change"] = {"reserved_bytes": 1}
        path.write_text(json.dumps(newer))
        return result
    monkeypatch.setattr(pool.StorageLedger, "_load", changed)
    with pytest.raises(ValueError, match="changed during atomic read"):
        m.StorageAccounting(c, receipt, runtime)


@pytest.mark.parametrize("shortfall", [0, 1])
def test_available_filesystem_space_must_exceed_every_reserved_writer(tmp_path, monkeypatch, shortfall):
    c, receipt, runtime, path, state, *_ = helpers.storage_fixture(tmp_path, monkeypatch)
    reserve = state["external_reserve_bytes"] + sum(v["reserved_bytes"] for v in state["controllers"].values()) + sum(v["reserved_bytes"] for v in state["jobs"].values())
    monkeypatch.setattr(m.os, "statvfs", lambda path: SimpleNamespace(f_bavail=reserve-shortfall, f_frsize=1))
    with pytest.raises(ValueError, match="free-space headroom"):
        m.StorageAccounting(c, receipt, runtime)


@pytest.mark.parametrize("scope", ["qualification_tree", "launch_log_parent"])
def test_readonly_installation_cannot_hide_future_qualification_writes(tmp_path, monkeypatch, scope):
    c, receipt, runtime, _, _, path, charge = helpers.storage_fixture(tmp_path, monkeypatch)
    overlap = c["qualification_root"] if scope == "qualification_tree" else str(Path(c["storage_accounting"]["qualification_launch_log"]).parent)
    charge["scope_paths"].append(overlap)
    rewrite_charge(receipt, path, charge)
    with pytest.raises(ValueError, match="readonly installation"):
        m.StorageAccounting(c, receipt, runtime)


def test_local_scope_counts_sparse_apparent_bytes_and_rejects_nested_escape(tmp_path):
    folder = tmp_path / "local"; folder.mkdir()
    sparse = folder / "handmade_sparse.bin"
    with sparse.open("wb") as stream:
        stream.truncate(16385)
    info = sparse.stat()
    assert m.local_scope_bytes(sparse) == max(info.st_size, info.st_blocks * 512) >= 16385
    assert m.local_scope_bytes(folder) >= m.local_scope_bytes(sparse)
    outside = tmp_path / "outside"; outside.mkdir()
    (folder / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        m.local_scope_bytes(folder)


def test_global_walk_removed_and_runtime_inventory_not_in_stage_accounting():
    tree = ast.parse((ROOT / "scripts/infer_biological_paths_tsinfer_v1.py").read_text())
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "project_bytes" not in functions
    assert all(not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "walk"
                   and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == "ROOT")
               for n in ast.walk(tree))
    accounting = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StorageAccounting")
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "runtime_inventory"
                   for n in ast.walk(accounting))
    stage = next(n for n in functions["pipeline"].body if isinstance(n, ast.FunctionDef) and n.name == "stage")
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "runtime_inventory"
                   for n in ast.walk(stage))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "runtime_inventory"]
    assert len(calls) == 2

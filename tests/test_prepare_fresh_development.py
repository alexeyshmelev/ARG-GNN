"""Development preparation safety checks. No simulation, inference or GPU work."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
spec = importlib.util.spec_from_file_location("fresh_development", ROOT/"scripts/prepare_fresh_development.py")
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def registry():
    return [{"split": "train", "replicate": 0,
             "seeds": {"ancestry": 42, "mutations": 43}, "inference_seed": 44}]


def test_actual_seed_registry_has80_independent_parent_identities_and240_unique_streams():
    rows = fresh.planned_registry()
    assert [(r["split"], r["replicate"]) for r in rows] == fresh.expected_ids()
    assert len(rows) == 80
    assert fresh.reject_collisions(rows, []) == 0
    assert rows == fresh.planned_registry()
    assert all(1 <= r["inference_seed"] <= 2**31-1 for r in rows)


@pytest.mark.parametrize("bad", [
    [*fresh.expected_ids()[:-1]],
    [*fresh.expected_ids(), ("test", 0)],
    [*fresh.expected_ids()[:-1], fresh.expected_ids()[0]],
])
def test_parent_split_identity_checks_reject_missing_test_or_duplicate(bad):
    with pytest.raises(RuntimeError, match="parents|TEST"):
        fresh.validate_parent_records([{"split": s, "replicate": i} for s, i in bad])


def test_all_manifest_partial_fixture_and_orphan_metadata_sources_are_scanned(tmp_path):
    write(tmp_path/"data/past/manifest.json", {"replicates": [{"seeds": {"ancestry": 1}}]})
    write(tmp_path/"data/failed/manifest.partial.json", {"seeds": {"mutations": 2}})
    write(tmp_path/".runtime/tmp/fixture/source_manifest.json", {"inference_seed": 3})
    write(tmp_path/"data/orphan/train/metadata.json", {"seeds": {"ancestry": 4}})
    excluded = tmp_path/"data/fresh"
    write(excluded/"manifest.json", {"seeds": {"ancestry": 42}})
    rows = fresh.manifest_inventory(tmp_path, exclude_roots=[excluded])
    assert len(rows) == 4
    assert {u["value"] for r in rows for u in r["seed_uses"]} == {1,2,3,4}


def test_malformed_prior_manifest_fails_closed_and_fixture_is_removed(tmp_path):
    path = tmp_path/"data/past/manifest.partial.json"
    path.parent.mkdir(parents=True)
    path.write_text("{")
    try:
        with pytest.raises(RuntimeError, match="malformed prior manifest"):
            fresh.manifest_inventory(tmp_path)
    finally:
        path.unlink()  # This test's own malformed fixture must not poison future repository audits.


def test_final_confirmatory_manifest_is_blocked_before_reading(tmp_path, monkeypatch):
    path = write(tmp_path/"data/ancestry_confirmatory_v1/manifest.json", {"seeds": {"ancestry": 9}})
    def forbidden(_):
        pytest.fail("Final held-out file was opened")
    monkeypatch.setattr(fresh, "read_json", forbidden)
    with pytest.raises(RuntimeError, match="refusing access"):
        fresh.manifest_inventory(tmp_path)
    path.unlink()  # Own synthetic held-out path, never actual data.


def test_prior_collision_and_proposed_cross_stream_collision_rejected():
    history = [{"path": "past/manifest.json", "seed_uses": [{"value": 42, "location": "$.seeds.ancestry"}]}]
    with pytest.raises(RuntimeError, match="Fresh development seed collision"):
        fresh.reject_collisions(registry(), history)
    bad = registry()
    bad[0]["inference_seed"] = 42
    with pytest.raises(RuntimeError, match="across streams"):
        fresh.reject_collisions(bad, [])


def test_seed_collection_ignores_training_seed_and_finds_nested_inference_seed():
    uses = fresh.seed_uses({"training_seed": 42, "inference_configuration": {"seed": 44},
                           "source_parent": {"seeds": {"ancestry": 43}}, "ancestry": True})
    assert {r["value"] for r in uses} == {43,44}


def test_storage_guard_uses_conservative_full_cohort_reserve_and_stops_at450gb():
    plan, _ = fresh.load_plan()
    result = fresh.storage_guard(plan, used=24_000_000_000, free=1_000_000_000_000)
    assert result["development_payload_estimate_bytes"] == 48_316_828_784
    assert result["projected_bytes_with_reserves"] < 450_000_000_000
    with pytest.raises(RuntimeError, match="reaches450GB"):
        fresh.storage_guard(plan, used=400_000_000_000, free=1_000_000_000_000)
    with pytest.raises(RuntimeError, match="free space"):
        fresh.storage_guard(plan, used=0, free=50_000_000_000)


@pytest.mark.parametrize("key,value", [
    ("physical_coverage_fraction", 0.94), ("physical_coverage_fraction", float("nan")),
    ("retained_grid_points",119), ("retained_grid_points",129), ("packed_nodes",0),
    ("packed_nodes",1_000_001), ("native_returncodes",[0,1]),
])
def test_admission_rejects_whole_parent_by_observations_only(key, value):
    plan, _ = fresh.load_plan()
    row = {"physical_coverage_fraction": 0.99, "retained_grid_points": 128,
           "packed_nodes": 20_000, "native_returncodes":[0,0]}
    fresh.coverage_admission(row, plan)
    row[key] = value
    with pytest.raises(RuntimeError, match="Whole development parent"):
        fresh.coverage_admission(row, plan)


def test_development_plan_rejects_final_parent_counts_and_masters(tmp_path):
    plan, oracle = fresh.load_plan()
    oracle_path = write(tmp_path/"oracle_config.json", oracle)
    plan = {**plan, "oracle_config": str(oracle_path)}
    path = write(tmp_path/"plan.json", plan)
    fresh.load_plan(path)
    for change in [{"train":64,"val":16,"test":1}, {"train":160,"val":16,"test":0}]:
        bad = deepcopy(plan)
        bad["expected_splits"] = change
        write(path, bad)
        with pytest.raises(ValueError, match="development64/16"):
            fresh.load_plan(path)
    bad = deepcopy(oracle)
    bad["master_seed"] = 20261111
    write(oracle_path, bad)
    write(path, plan)
    with pytest.raises(ValueError, match="master seeds"):
        fresh.load_plan(path)


def test_confined_paths_reject_outside_and_symlink_escape(tmp_path):
    root = tmp_path/"root"
    root.mkdir()
    (root/"escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        fresh.confined(root/"escape/file.json", root=root)
    with pytest.raises(ValueError, match="escapes"):
        fresh.confined(tmp_path/"outside.json", root=root)


def test_explicit_freeze_sha_and_code_pins_are_required(tmp_path, monkeypatch):
    frozen = {"kind":"fresh_development_preparation_execution_freeze",
              "confirmatory_access":False,"not_confirmatory_test_freeze":True,
              "config_files":{},"source_code_sha256":{"core":"original"},"parent_registry":registry()}
    path = write(tmp_path/"frozen.json", frozen)
    monkeypatch.setattr(fresh, "code_pins", lambda: {"core":"original"})
    monkeypatch.setattr(fresh, "planned_registry", registry)
    assert fresh.verify_freeze(path, fresh.sha(path)) == frozen
    with pytest.raises(RuntimeError, match="SHA256"):
        fresh.verify_freeze(path, "")
    monkeypatch.setattr(fresh, "code_pins", lambda: {"core":"changed"})
    with pytest.raises(RuntimeError, match="implementation drift"):
        fresh.verify_freeze(path, fresh.sha(path))


def test_no_argument_main_is_read_only_preflight_without_execute_or_freeze(monkeypatch):
    monkeypatch.setattr(fresh, "configure_cpu", lambda: None)
    result = {"parent_registry":fresh.expected_ids(), "prior_inventory":[],
              "prior_unique_seed_values":0,"prior_inventory_sha256":"a",
              "storage":{},"cpu_available_bytes":40_000_000_000}
    monkeypatch.setattr(fresh, "preflight", lambda *_: result)
    monkeypatch.setattr(fresh, "execute", lambda *_: pytest.fail("Unexpected data execution"))
    monkeypatch.setattr(fresh, "write_new", lambda *_: pytest.fail("Unexpected write in preflight"))
    monkeypatch.setattr(fresh, "report_event", lambda value: None)
    fresh.main([])


def test_empty_existing_output_fails_before_any_simulation(tmp_path, monkeypatch):
    import arggnn.ancient_simulate as ancient
    target = tmp_path/"empty_oracle"
    target.mkdir()
    monkeypatch.setattr(ancient, "simulate_ancient_replicate",
                        lambda *_a, **_k: pytest.fail("Unexpected simulation"))
    with pytest.raises(RuntimeError, match="exists; no overwrite"):
        fresh.stage_oracle({"paths":{"oracle":str(target)}}, {}, {})


def test_failed_stage_retains_parent_reason_and_immutable_attempt_ledger(tmp_path, monkeypatch):
    directory = tmp_path/"receipts"
    plan = {"paths":{"receipts":str(directory)}}
    monkeypatch.setattr(fresh, "assert_live_pins", lambda *_: {})
    def fail(*_):
        fresh.parent_started("fake", "train", 7)
        raise RuntimeError("Synthetic observation admission failure")
    monkeypatch.setattr(fresh, "STAGES", {"fake":fail})
    with pytest.raises(RuntimeError, match="Synthetic observation"):
        fresh.execute("fake", plan, {}, {}, "f"*64)
    receipts = list(directory.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "failed"
    assert receipt["failed_or_last_active_parent"] == {"stage":"fake","split":"train","replicate":7}
    ledger = [json.loads(line) for line in Path(receipt["parent_attempt_ledger"]).read_text().splitlines()]
    assert any(r["event"] == "development_parent_attempt_started" for r in ledger)
    assert any(r["event"] == "development_stage_failed" and r["no_parent_removed_or_replaced"] for r in ledger)
    assert fresh._EVENT_LOG is None and fresh._CURRENT_PARENT is None

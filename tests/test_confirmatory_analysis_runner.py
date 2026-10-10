"""Handmade NPZ/census fixtures and an explicit mocked frozen-lock admission.

No real confirmatory lock, cohort, target, model prediction, CUDA or scientific
workflow evidence is produced. Production frozen-lock/cohort guards are external.
"""
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arggnn import confirmatory_analysis_runner as analysis
from arggnn.dense_tract_metrics import coverage_clipped_midpoint_grid

ROOT = Path(__file__).resolve().parents[1]
REAL_FROZEN_VALIDATOR = analysis.lock_api.validate_confirmatory_lock


def write_json(root, path, values):
    p = root/path; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(analysis._canonical(values))
    return {"path": path, "sha256": analysis._sha(p.read_bytes())}


def write_npz(root, path, metadata, **arrays):
    p = root/path; p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
    return {"path": path, "sha256": analysis._sha(p.read_bytes())}


def remake(study, name):
    ref = write_json(study["root"], study[name+"_ref"]["path"], study[name])
    study[name+"_ref"] = ref
    if name == "ledger":
        study["roster"]["admission_ledger_sha256"] = ref["sha256"]
        remake(study, "roster")
    return ref


def fixture_study(root, monkeypatch):
    # These copied code bytes and actual protocol prose are evidence about this
    # test fixture only. All development/cohort/frozen-lock admission is mocked.
    for path in analysis.CODE_PATHS.values():
        dst = root/path; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes((ROOT/path).read_bytes())
    schema = analysis.analysis_schema_values()
    schema_ref = write_json(root, "metadata/schema.json", schema)
    protocol = json.loads((ROOT/"research/protocols/confirmatory_design_v1.json").read_text())
    protocol_ref = write_json(root, "metadata/protocol.json", protocol)
    namespace = protocol["confirmation"]["namespace"]
    registry = {"training_seeds": analysis.SEEDS, "analysis_seed": analysis.ANALYSIS_SEED,
        "cohorts": {"id_confirmation": {"namespace": namespace, "target_admitted_parents": 160,
            "parents": [{"parent_id": i, "ancestry_seed": 10000+i, "mutation_seed": 20000+i,
                         "inference_seed": 30000+i} for i in range(192)]}}}
    registry_ref = write_json(root, "metadata/registry.json", registry)
    contract_ref = write_json(root, "metadata/contract.json",
        {"confirmation": protocol["confirmation"], "separate_prespecified_analyses": protocol["separate_prespecified_analyses"]})
    methods = {}
    for method, family in (("candidate", "candidate"), ("strongest_required_baseline", "canonical_gru"),
                           ("gain_matched_diagonal", "gain_matched_diagonal"), ("static_local", "static_local")):
        methods[method] = {"family_id": family, "configuration_id": method+"_toy",
            "protocol": {"path": "synthetic_development/"+method+".json", "sha256": "1"*64},
            "choice": {"family_id": family, "configuration_id": method+"_toy", "synthetic_fixture": True},
            "seed_bundles": [{"seed": seed, "checkpoint": {"path": "synthetic_development/"+method+str(seed)+".pt", "sha256": "2"*64},
                              "development_prediction": {"path": "synthetic_development/"+method+str(seed)+"_dev.pt", "sha256": "3"*64}}
                              for seed in analysis.SEEDS]}
    temperatures = {"schema_version": 1, "kind": "prespecified_validation_temperature_manifest",
        "status": "frozen_before_test_preparation", "training_seeds": analysis.SEEDS, "final_test_access": False,
        "records": [{"method_id": method, "seed": seed, "temperature": 1.5,
            "checkpoint": analysis._seed_bundle(bundle, seed)["checkpoint"],
            "development_prediction": analysis._seed_bundle(bundle, seed)["development_prediction"],
            "fit_scope": "fresh_development_validation_only"} for method, bundle in methods.items() for seed in analysis.SEEDS]}
    temperature_ref = write_json(root, "metadata/temperatures.json", temperatures)
    payload = {"kind": "confirmatory_provenance_lock", "training_seeds": analysis.SEEDS,
        "lock_execution_scope": "id_confirmation_only", "candidate": methods["candidate"],
        "strongest_required_baseline": methods["strongest_required_baseline"],
        "secondary_controls": {key: value for key, value in methods.items() if key not in ("candidate", "strongest_required_baseline")},
        "analysis_environment_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")},
        "bound_references": {"analysis_schema": schema_ref, "protocol_json": protocol_ref,
            "analysis_contract": contract_ref, "parent_seed_registry": registry_ref, "temperature_manifest": temperature_ref},
        "code_artifacts": {"joined_analysis": schema["code_artifacts"]["joined_analysis"] | {"entry_point": "main"}}}
    lock_ref = write_json(root, "metadata/lock.json", payload)
    frozen = SimpleNamespace(payload=payload, content=(root/lock_ref["path"]).read_bytes(), sha256=lock_ref["sha256"], check=lambda: None)
    calls = []
    def mock_frozen_admission(path, digest, *, project_root):
        # Mock deliberately replaces expensive actual development/custody proof.
        assert Path(project_root) == root
        assert path == root/lock_ref["path"] and digest == lock_ref["sha256"]
        calls.append("mocked_frozen_lock_admission")
        return frozen
    monkeypatch.setattr(analysis.lock_api, "validate_confirmatory_lock", mock_frozen_admission, raising=False)
    primary = coverage_clipped_midpoint_grid(domain_bp=(0., 1e6), coverage_bp=(0., 1e6),
        grid_id="synthetic_primary", grid_points=128).identity_record()
    dense = coverage_clipped_midpoint_grid(domain_bp=(0., 1e6), coverage_bp=(0., 1e6),
        grid_id="synthetic_dense", grid_points=4096).identity_record()
    queries = list(range(100, 164)); pairs = np.arange(64).reshape(32, 2).tolist()
    attempts = []; entries = []; targets = set(); observed_refs = {}; label_headers = {}
    for parent in range(192):
        row = {"parent_id": parent, "status": "rejected", "reason": "Explicit synthetic label-blind rejected parent",
            "label_blind": True, "observed_identity": None, "primary_labels": None, "census_tracts": None}
        if parent < 3:
            # Handmade source arrays are never passed through genuine topology/cohort admission.
            source_header={"schema_version":1,"kind":"sealed_confirmatory_observed_graph_sequence",
                "lock_sha256":frozen.sha256,"namespace":namespace,"parent_id":parent,
                "feature_names":schema["observed_npz"]["feature_names"],"time_scale":10000.,
                "truth_free_observed_inputs":True}
            source_ref=write_npz(root,f"synthetic_observed/source{parent}.npz",source_header,
                x=np.zeros((64,8),dtype=np.float32),edge_index=np.empty((2,0),dtype=np.int64),
                node_graph=np.zeros(64,dtype=np.int64),node_ids=np.asarray(queries,dtype=np.int64),
                clades=np.zeros((64,16),dtype=np.float32),intervals=np.asarray([[0.,1.]],dtype=np.float64),
                query_node_ids=np.asarray(queries,dtype=np.int64),window_bounds=np.asarray([0.,1.],dtype=np.float64))
            identity = {"schema_version": 1, "kind": "sealed_confirmatory_observed_parent_identity",
                "lock_sha256": frozen.sha256, "namespace": namespace, "parent_id": parent,
                "parent_seeds": {key: registry["cohorts"]["id_confirmation"]["parents"][parent][key]
                                 for key in ("ancestry_seed", "mutation_seed", "inference_seed")},
                "source_artifacts": [source_ref], "query_ids": queries, "diploid_pairs": pairs,
                "primary_grid": primary, "dense_grid": dense,
                "artificial_tile_edges_bp": np.arange(1, 8).tolist(), "truth_free_observed_inputs": True}
            identity_ref = write_json(root, f"synthetic_observed/identity{parent}.json", identity)
            observed_refs[parent] = identity_ref
            header = {"schema_version": 1, "kind": "sealed_external_primary_labels", "lock_sha256": frozen.sha256,
                "namespace": namespace, "parent_id": parent, "observed_identity_sha256": identity_ref["sha256"],
                "primary_grid_sha256": primary["sha256"], "class_order": ["ANA", "CHG", "WHG", "EHG"]}
            label_headers[parent] = header
            labels_ref = write_npz(root, f"synthetic_targets/labels{parent}.npz", header,
                labels=np.zeros((128, 64), dtype=np.int64), positions_bp=np.asarray(primary["positions_bp"]),
                query_ids=np.asarray(queries, dtype=np.int64))
            census = {key: value for key, value in header.items() if key != "primary_grid_sha256"}
            census.update(kind="sealed_external_census_tract_partitions", query_ids=queries,
                          domain_bp=[0., 1e6], tracts=[[[0., 1e6, 0]] for _ in queries])
            census_ref = write_json(root, f"synthetic_targets/census{parent}.json", census)
            targets.update((labels_ref["path"], census_ref["path"]))
            row.update(status="admitted", reason="Explicit synthetic label-blind accepted parent",
                       observed_identity=identity_ref, primary_labels=labels_ref, census_tracts=census_ref)
            for method_index, (method, bundle) in enumerate(methods.items()):
                for seed_index, seed in enumerate(analysis.SEEDS):
                    score = 1.0+method_index*.04+parent*.003+seed_index*.002
                    z = -math.log(math.expm1(score)/3)
                    small = np.zeros((128, 64, 4), dtype=np.float64); small[..., 0] = z
                    large = np.zeros((4096, 64, 4), dtype=np.float32); large[..., 0] = z
                    checkpoint = analysis._seed_bundle(bundle, seed)["checkpoint"]
                    prediction_header = {"schema_version": 1, "kind": "sealed_confirmatory_raw_prediction",
                        "lock_sha256": frozen.sha256, "namespace": namespace, "parent_id": parent,
                        "method_id": method, "training_seed": seed, "configuration_id": bundle["configuration_id"],
                        "family_id": bundle["family_id"], "protocol": bundle["protocol"], "checkpoint": checkpoint,
                        "observed_identity_sha256": identity_ref["sha256"], "primary_grid_sha256": primary["sha256"],
                        "dense_grid_sha256": dense["sha256"], "class_order": ["ANA", "CHG", "WHG", "EHG"],
                        "no_gradients_or_optimization": True}
                    pred = write_npz(root, f"synthetic_predictions/{method}_s{seed}_p{parent}.npz", prediction_header,
                        primary_logits=small, dense_logits=large, primary_positions_bp=np.asarray(primary["positions_bp"]),
                        dense_positions_bp=np.asarray(dense["positions_bp"]), query_ids=np.asarray(queries, dtype=np.int64))
                    entries.append({"method_id": method, "seed": seed, "parent_id": parent,
                                    "status": "success", "prediction": pred, "checkpoint": checkpoint, "reason": ""})
        attempts.append(row)
    ledger = {"schema_version": 1, "kind": "sealed_confirmatory_admission_ledger", "status": "sealed_before_model_scoring",
        "lock_sha256": frozen.sha256, "parent_seed_registry_sha256": registry_ref["sha256"],
        "cohort": "id_confirmation", "namespace": namespace, "target_admitted_parents": 160,
        "maximum_attempted_parents": 192, "attempts": attempts}
    ledger_ref = write_json(root, "metadata/ledger.json", ledger)
    roster = {"schema_version": 1, "kind": "sealed_confirmatory_prediction_roster", "status": "sealed_before_target_join",
        "lock_sha256": frozen.sha256, "parent_seed_registry_sha256": registry_ref["sha256"],
        "admission_ledger_sha256": ledger_ref["sha256"], "cohort": "id_confirmation", "namespace": namespace, "entries": entries}
    roster_ref = write_json(root, "metadata/roster.json", roster)
    return {"root": root, "lock_ref": lock_ref, "frozen": frozen, "ledger": ledger, "ledger_ref": ledger_ref,
        "roster": roster, "roster_ref": roster_ref, "targets": targets, "identity_refs": observed_refs,
        "primary": primary, "dense": dense, "queries": queries, "label_headers": label_headers,
        "temperatures": temperatures, "calls": calls}


@pytest.fixture
def study(tmp_path, monkeypatch):
    return fixture_study(tmp_path, monkeypatch)


def admit(study):
    return analysis.admit_analysis(study["root"]/study["lock_ref"]["path"], study["lock_ref"]["sha256"],
        study["root"]/study["roster_ref"]["path"], study["roster_ref"]["sha256"],
        study["root"]/study["ledger_ref"]["path"], study["ledger_ref"]["sha256"], project_root=study["root"])


def watch_targets(monkeypatch, study):
    reads = []; original = analysis._Capture.read
    def watch(self, ref, **kwargs):
        if self.ref(ref)["path"] in study["targets"]:
            reads.append(ref["path"])
        return original(self, ref, **kwargs)
    monkeypatch.setattr(analysis._Capture, "read", watch)
    return reads


def test_admission_is_metadata_only_and_requires_public_frozen_lock(study, monkeypatch):
    reads = watch_targets(monkeypatch, study)
    admitted = admit(study)
    assert reads == [] and len(admitted.parents) == 3 and len(admitted.entries) == 36
    assert study["calls"] == ["mocked_frozen_lock_admission"]
    def reject(*args, **kwargs):
        raise ValueError("Actual production lock refused")
    monkeypatch.setattr(analysis.lock_api, "validate_confirmatory_lock", reject)
    with pytest.raises(ValueError, match="production lock refused"):
        admit(study)
    assert reads == []


@pytest.mark.parametrize("mutation", ["missing_cell", "duplicate_cell", "extra_seed", "switch_checkpoint", "unsealed",
                                     "wrong_lock", "wrong_ledger", "unresolved_with_prediction"])
def test_bad_roster_blocks_before_target_open(study, monkeypatch, mutation):
    reads = watch_targets(monkeypatch, study)
    roster = study["roster"]
    if mutation == "missing_cell": roster["entries"].pop()
    elif mutation == "duplicate_cell": roster["entries"].append(deepcopy(roster["entries"][0]))
    elif mutation == "extra_seed": roster["entries"][0]["seed"] = 404
    elif mutation == "switch_checkpoint": roster["entries"][0]["checkpoint"]["sha256"] = "a"*64
    elif mutation == "unsealed": roster["status"] = "draft"
    elif mutation == "wrong_lock": roster["lock_sha256"] = "b"*64
    elif mutation == "wrong_ledger": roster["admission_ledger_sha256"] = "c"*64
    else: roster["entries"][0]["status"] = "unresolved"; roster["entries"][0]["reason"] = "Actual fixture unresolved"
    remake(study, "roster")
    with pytest.raises(ValueError):
        admit(study)
    assert reads == []


@pytest.mark.parametrize("mutation", ["early_stop", "reorder", "not_label_blind", "target_traversal", "rej_target",
                                     "source_drift", "query_order", "diploid_overlap", "grid_drift"])
def test_bad_admission_source_or_identity_blocks_before_target_open(study, monkeypatch, mutation):
    reads = watch_targets(monkeypatch, study)
    if mutation == "early_stop": study["ledger"]["attempts"].pop()
    elif mutation == "reorder": study["ledger"]["attempts"][0]["parent_id"] = 1
    elif mutation == "not_label_blind": study["ledger"]["attempts"][0]["label_blind"] = False
    elif mutation == "target_traversal": study["ledger"]["attempts"][0]["primary_labels"]["path"] = "../target.npz"
    elif mutation == "rej_target": study["ledger"]["attempts"][3]["primary_labels"] = deepcopy(study["ledger"]["attempts"][0]["primary_labels"])
    elif mutation == "source_drift": (study["root"]/"synthetic_observed/source0.npz").write_bytes(b"Changed synthetic source")
    else:
        ref = study["identity_refs"][0]; identity = json.loads((study["root"]/ref["path"]).read_text())
        if mutation == "query_order": identity["query_ids"][0] = identity["query_ids"][1]
        elif mutation == "diploid_overlap": identity["diploid_pairs"][0][0] = identity["diploid_pairs"][0][1]
        else: identity["primary_grid"]["positions_bp"][0] += 1
        new = write_json(study["root"], ref["path"], identity)
        study["ledger"]["attempts"][0]["observed_identity"] = new
    remake(study, "ledger")
    with pytest.raises(ValueError):
        admit(study)
    assert reads == []


def test_all_predictions_preflight_before_any_target_join(study, monkeypatch):
    reads = watch_targets(monkeypatch, study)
    admitted = admit(study)
    last = study["roster"]["entries"][-1]["prediction"]
    (study["root"]/last["path"]).write_bytes(b"Corrupt last synthetic prediction")
    with pytest.raises(ValueError, match="hash"):
        analysis.analyze_confirmatory(admitted)
    assert reads == []


def test_in_memory_parent_or_policy_drift_is_rejected(study):
    admitted = admit(study)
    admitted.parents.pop()
    with pytest.raises(ValueError, match="In-memory"):
        analysis.analyze_confirmatory(admitted)


def test_complete_synthetic_join_seed_vectors_calibration_and_dense_na(study, monkeypatch):
    reads = watch_targets(monkeypatch, study)
    admitted = admit(study)
    output=study["root"]/"synthetic_id_report.json"
    emitted=analysis.analyze_confirmatory(admitted,output_path=output)
    assert output.read_bytes()==emitted["bytes"] and analysis._sha(output.read_bytes())==emitted["sha256"]
    result = emitted["values"]
    with pytest.raises(ValueError,match="Fresh exclusive"):analysis.analyze_confirmatory(admitted,output_path=output)
    assert len(reads) == 6
    assert result["all_predictions_successful"] and result["admission_target_shortfall"]
    assert not result["full_protocol_readiness"]
    assert not result["primary"]["interpretation"]["robust_positive_confirmation"]
    primary = result["primary"]["report"]
    assert primary["bootstrap_seed"] == 20261115 and primary["bootstrap_resamples"] == 10000
    assert primary["independent_parents"] == 3 and primary["fixed_training_models"] == 3
    assert np.asarray(primary["parent_seed_differences"]).shape == (3, 3)
    assert len(primary["seed_specific_mean_differences"]) == 3
    assert len(result["method_parent_seed_nll"]) == 12
    assert result["conditional_holm_family_size"] == 2
    assert all(row["holm_two_sided_pvalue"] <= 1 for row in result["conditional_secondary_holm"])
    row = result["descriptive_parent_seed_diagnostics"][0]
    assert row["raw_primary"]["nll"] != row["val_temperature_primary"]["nll"]
    assert len(row["raw_primary"]["calibration"]) == 15
    assert row["raw_primary"]["per_class"][1]["recall"] is None
    assert row["raw_dense_tracts"]["boundaries"]["recall"] is None
    assert row["raw_dense_tracts"]["boundaries"]["f1"] is None
    assert row["raw_primary"]["diploid_pairs"] == np.arange(64).reshape(32, 2).tolist()
    assert result["primary_selection_changed"] is False
    assert result["roster_sha256"] == study["roster_ref"]["sha256"]


def test_terminal_failed_primary_cells_visible_with_conservative_conditional_family(study):
    row = next(row for row in study["roster"]["entries"] if row["method_id"] == "strongest_required_baseline" and row["seed"] == 202 and row["parent_id"] == 1)
    row.update(status="failed", prediction=None, reason="Explicit preserved synthetic terminal failed prediction")
    remake(study, "roster")
    result = analysis.analyze_confirmatory(admit(study))["values"]
    assert not result["all_predictions_successful"]
    assert len(result["prediction_failures_or_unresolved"]) == 1
    assert result["primary"]["omitted_from_complete_case_sensitivity"] == [1]
    assert result["primary"]["complete_case_parent_ids"] == [0, 2]
    assert not result["primary"]["interpretation"]["robust_positive_confirmation"]
    vector = next(row for row in result["method_parent_seed_nll"] if row["method_id"] == "strongest_required_baseline" and row["parent_id"] == 1)
    assert vector["nll"][1] is None and vector["mean_three_seed_nll"] is None


@pytest.mark.parametrize("mutation", ["nan", "temperature_missing", "temperature_test_fit"])
def test_bad_pretest_temperature_blocks_before_targets(study, monkeypatch, mutation):
    reads = watch_targets(monkeypatch, study)
    temperatures = study["temperatures"]
    if mutation == "nan": temperatures["records"][0]["temperature"] = 0.
    elif mutation == "temperature_missing": temperatures["records"].pop()
    else: temperatures["records"][0]["fit_scope"] = "test_fit"
    ref = study["frozen"].payload["bound_references"]["temperature_manifest"]
    # Explicit mocked lock permits this forged temperature; the analysis layer
    # still independently rejects its matrix/scope/value before targets.
    study["frozen"].payload["bound_references"]["temperature_manifest"] = write_json(study["root"], ref["path"], temperatures)
    with pytest.raises(ValueError):
        admit(study)
    assert reads == []


def test_primary_uncapped_raw_nll_and_whole_parent_aggregation():
    z = np.zeros((2, 2, 4)); z[..., 0] = -1000.
    metrics = analysis.logit_metrics(z, np.zeros((2, 2), dtype=np.int64))
    assert metrics["nll"] > 1000 and metrics["historical_floor_1e12_nll"] < 28
    c = np.asarray([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]])
    b = c+np.asarray([[.01], [.02], [.03]])
    report = analysis.paired_parent_report(c, b, parent_ids=[11, 22, 33], training_seeds=analysis.SEEDS)
    np.testing.assert_allclose(report["paired_parent_differences"], [-.01, -.02, -.03])
    assert report["independent_parents"] == 3 and report["degrees_of_freedom"] == 2


@pytest.mark.parametrize("failure", ["write", "check", "race"])
def test_report_publication_preserves_existing_files_and_cleans_owned_pending(tmp_path, monkeypatch, failure):
    target = tmp_path/"synthetic_non_executable_report.json"
    original = analysis.tempfile.NamedTemporaryFile
    if failure == "write":
        class Broken:
            def __init__(self, handle): self.handle = handle; self.name = handle.name
            def __enter__(self): return self
            def fileno(self): return self.handle.fileno()
            def __exit__(self, *args): return self.handle.__exit__(*args)
            def write(self, value):
                self.handle.write(value[:7]); raise OSError("Synthetic interrupted report")
        monkeypatch.setattr(analysis.tempfile, "NamedTemporaryFile", lambda **kwargs: Broken(original(**kwargs)))
    def check():
        if failure == "check": raise ValueError("Synthetic admission drift")
        if failure == "race": target.write_bytes(b"Existing external raced bytes")
    with pytest.raises((OSError, ValueError)):
        analysis._publish(target, b"Explicit synthetic non-executable receipt", check)
    assert not list(tmp_path.glob(".analysis_pending_*"))
    if failure == "race": assert target.read_bytes() == b"Existing external raced bytes"
    else: assert not target.exists()


def test_no_admission_object_cannot_claim_a_result(tmp_path):
    with pytest.raises(ValueError, match="Actual frozen-lock"):
        analysis.analyze_confirmatory(SimpleNamespace(), output_path=tmp_path/"report.json")
    assert not (tmp_path/"report.json").exists()


@pytest.mark.parametrize("mutation",["header","query_order","nan","shape","object_metadata"])
def test_prediction_array_contract_fails_before_targets(study,monkeypatch,mutation):
    reads=watch_targets(monkeypatch,study)
    row=study["roster"]["entries"][-1];path=study["root"]/row["prediction"]["path"]
    with np.load(path,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    header=json.loads(arrays.pop("metadata_json").item())
    if mutation=="header":header["no_gradients_or_optimization"]=False
    elif mutation=="query_order":arrays["query_ids"]=arrays["query_ids"][::-1]
    elif mutation=="nan":arrays["primary_logits"][0,0,0]=np.nan
    elif mutation=="shape":arrays["primary_logits"]=arrays["primary_logits"][:-1]
    row["prediction"]=write_npz(study["root"],row["prediction"]["path"],header,**arrays)
    if mutation=="object_metadata":
        with np.load(path,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
        arrays["metadata_json"]=np.asarray({"unsafe_fixture":True},dtype=object)
        np.savez_compressed(path,**arrays)
        row["prediction"]["sha256"]=analysis._sha(path.read_bytes())
    remake(study,"roster")
    with pytest.raises(ValueError):analysis.analyze_confirmatory(admit(study))
    assert reads==[]


def test_external_labels_and_census_must_join_consistently(study):
    row=study["ledger"]["attempts"][0];ref=row["census_tracts"]
    census=json.loads((study["root"]/ref["path"]).read_text());census["tracts"][0][0][2]=1
    row["census_tracts"]=write_json(study["root"],ref["path"],census);remake(study,"ledger")
    output=study["root"]/"never_report.json"
    with pytest.raises(ValueError,match="disagree"):analysis.analyze_confirmatory(admit(study),output_path=output)
    assert not output.exists()


def test_imported_analysis_source_drift_blocks_metadata_helper(monkeypatch):
    pins=analysis._pins()
    pins["joined_analysis"]={**pins["joined_analysis"],"sha256":"f"*64}
    monkeypatch.setattr(analysis,"_pins",lambda:pins)
    with pytest.raises(ValueError,match="Imported analysis source"):analysis.analysis_schema_values()


def test_noisy_shortfall_does_not_become_positive_confirmation():
    report=analysis.paired_parent_report(np.zeros((3,3)),np.asarray([[.04]*3,[.05]*3,[.06]*3]),
        parent_ids=[1,2,3],training_seeds=analysis.SEEDS)
    interpretation=analysis._interpret(report,complete=True,target=160)
    assert interpretation["precision_shortfall"] and not interpretation["robust_positive_confirmation"]


def test_real_frozen_lock_reader_refuses_handmade_lock_before_targets(study,monkeypatch):
    reads=watch_targets(monkeypatch,study)
    monkeypatch.setattr(analysis.lock_api,"validate_confirmatory_lock",REAL_FROZEN_VALIDATOR)
    with pytest.raises((ValueError,KeyError)):
        admit(study)
    assert reads==[]


def test_unresolved_roster_cannot_open_any_target(study,monkeypatch):
    reads=watch_targets(monkeypatch,study)
    row=study["roster"]["entries"][-1]
    row.update(status="unresolved",prediction=None,reason="Explicit initial unresolved synthetic prediction")
    remake(study,"roster")
    admitted=admit(study)
    assert any(row["status"]=="unresolved" for row in admitted.entries.values())
    with pytest.raises(ValueError,match="Complete terminal prediction roster"):
        analysis.analyze_confirmatory(admitted,output_path=study["root"]/"never_unresolved_report.json")
    assert reads==[] and not (study["root"]/"never_unresolved_report.json").exists()

"""Custody CPU fixtures. Positive published-lock admission is explicitly mocked.

No actual confirmation input, targets, generation, native inference or GPU use.
The observed star tree, electrical/canonical QA and target joins are real tiny code paths.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from arggnn import confirmatory_custodian as custody
from arggnn.dense_tract_metrics import coverage_clipped_midpoint_grid

REAL_LOCK_VALIDATOR = custody.lock_api.validate_confirmatory_lock
torch.set_num_threads(1)


def write_json(root, name, values):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(custody._canonical(values))
    return {"path": name, "sha256": custody._sha(path.read_bytes())}


def write_npz(root, name, header, arrays):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(header, sort_keys=True)), **arrays)
    return {"path": name, "sha256": custody._sha(path.read_bytes())}


def rewrite_request(study):
    study["request_ref"] = write_json(study["root"], "metadata/import.json", study["request"])


def source_arrays():
    n = 89; x = np.zeros((n, 8), dtype=np.float64)
    ages = np.zeros(n); ages[88] = 1000.
    for category, age in enumerate((260., 300., 250., 250.)):
        x[6*category:6*(category+1), category] = 1.
        ages[6*category:6*(category+1)] = age
    x[:88, 5] = 1.; x[24:88, 4] = 1.
    x[:, 6] = np.log1p(ages)/math.log1p(10000.)
    x[:88, 7] = np.log1p(1000.-ages[:88])/math.log1p(10000.)
    clades = np.zeros((n, 16), dtype=np.float64)
    landmarks = [category*6+index for category in range(4) for index in (0, 1, 3, 5)]
    clades[landmarks, np.arange(16)] = 1.; clades[88] = 1.
    return {"x": x, "edge_index": np.stack((np.full(88, 88), np.arange(88))).astype(np.int64),
        "node_graph": np.zeros(n, dtype=np.int64), "node_ids": np.asarray(list(range(24))+list(range(100, 164))+[-1], dtype=np.int64),
        "clades": clades, "intervals": np.asarray([[0., 1.]], dtype=np.float64),
        "query_node_ids": np.arange(100, 164, dtype=np.int64), "window_bounds": np.asarray([0., 1.], dtype=np.float64)}


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path
    pins = custody.custodian_schema_values()["code_artifacts"]
    for ref in pins.values():
        p = root/ref["path"]; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((custody.ROOT/ref["path"]).read_bytes())
    schema = custody.analysis.analysis_schema_values()
    schema_ref = write_json(root, "metadata/schema.json", schema)
    protocol = {"confirmation": {"namespace": "ARG-GNN/ancestry-confirmatory-v1",
        "target_admitted_parents": 160, "max_attempted_parents": 192},
        "common_generation": {"sequence_length_bp": 1000000}}
    protocol_ref = write_json(root, "metadata/protocol.json", protocol)
    namespace = protocol["confirmation"]["namespace"]
    registry = {"training_seeds": custody.SEEDS, "cohorts": {"id_confirmation": {
        "namespace": namespace, "target_admitted_parents": 160,
        "parents": [{"parent_id": i, "ancestry_seed": 10000+i, "mutation_seed": 20000+i,
            "inference_seed": 30000+i} for i in range(192)]}}}
    registry_ref = write_json(root, "metadata/registry.json", registry)
    methods = {name: {"seed_bundles": [{"seed": seed, "checkpoint": {
        "path": f"synthetic_development/{name}_{seed}.pt", "sha256": "4"*64}} for seed in custody.SEEDS]}
        for name in ("candidate", "strongest_required_baseline", "gain_matched_diagonal", "static_local")}
    payload = {"kind": "confirmatory_provenance_lock", "lock_execution_scope": "id_confirmation_only",
        "training_seeds": custody.SEEDS, "candidate": methods["candidate"],
        "strongest_required_baseline": methods["strongest_required_baseline"],
        "secondary_controls": {key: value for key, value in methods.items()
            if key not in ("candidate", "strongest_required_baseline")},
        "bound_references": {"analysis_schema": schema_ref, "protocol_json": protocol_ref, "parent_seed_registry": registry_ref},
        "code_artifacts": {"custodian": pins["custodian"] | {"entry_point": "prepare_confirmatory_release"}},
        "captured_artifacts": [{"path": ref["path"], "sha256": ref["sha256"],
            "bytes": (root/ref["path"]).stat().st_size} for ref in pins.values()]}
    lock_ref = write_json(root, "metadata/synthetic_lock.json", payload)
    lock_checks = []
    frozen = SimpleNamespace(payload=payload, sha256=lock_ref["sha256"],
        check=lambda full=True: lock_checks.append(full))
    monkeypatch.setattr(custody.lock_api, "validate_confirmatory_lock", lambda *args, **kwargs: frozen)
    primary = coverage_clipped_midpoint_grid(domain_bp=(0., 1e6), coverage_bp=(0., 1e6),
        grid_id="synthetic_primary", grid_points=128).identity_record()
    dense = coverage_clipped_midpoint_grid(domain_bp=(0., 1e6), coverage_bp=(0., 1e6),
        grid_id="synthetic_dense", grid_points=4096).identity_record()
    attempts = []; sources = {}; identities = {}; target_headers = {}; target_arrays = {}; targets = set()
    for parent in range(192):
        row = {"parent_id": parent, "status": "rejected", "reason": "Synthetic observed-only rejection",
            "label_blind": True, "native_inference_returncode": None,
            "observed_identity": None, "primary_labels": None, "census_tracts": None}
        if parent < 2:
            header = {"schema_version": 1, "kind": "sealed_confirmatory_observed_graph_sequence",
                "lock_sha256": frozen.sha256, "namespace": namespace, "parent_id": parent,
                "feature_names": schema["observed_npz"]["feature_names"], "time_scale": 10000.,
                "truth_free_observed_inputs": True}
            arrays = source_arrays()
            source_ref = write_npz(root, f"synthetic_observed/source{parent}.npz", header, arrays)
            sources[parent] = (source_ref, header, arrays)
            identity = {"schema_version": 1, "kind": "sealed_confirmatory_observed_parent_identity",
                "lock_sha256": frozen.sha256, "namespace": namespace, "parent_id": parent,
                "parent_seeds": {key: registry["cohorts"]["id_confirmation"]["parents"][parent][key]
                    for key in ("ancestry_seed", "mutation_seed", "inference_seed")},
                "source_artifacts": [source_ref], "query_ids": list(range(100, 164)),
                "diploid_pairs": np.arange(64).reshape(32, 2).tolist(), "primary_grid": primary,
                "dense_grid": dense, "artificial_tile_edges_bp": [float(i*125000) for i in range(1, 8)],
                "truth_free_observed_inputs": True}
            identity_ref = write_json(root, f"synthetic_observed/identity{parent}.json", identity)
            identities[parent] = identity
            label_header = {"schema_version": 1, "kind": "sealed_external_primary_labels",
                "lock_sha256": frozen.sha256, "namespace": namespace, "parent_id": parent,
                "observed_identity_sha256": identity_ref["sha256"], "primary_grid_sha256": primary["sha256"],
                "class_order": custody.CLASS_ORDER}
            label_arrays = {"labels": np.zeros((128, 64), dtype=np.int64),
                "positions_bp": np.asarray(primary["positions_bp"], dtype=np.float64),
                "query_ids": np.arange(100, 164, dtype=np.int64)}
            labels_ref = write_npz(root, f"synthetic_targets/labels{parent}.npz", label_header, label_arrays)
            census = {key: value for key, value in label_header.items() if key != "primary_grid_sha256"}
            census.update(kind="sealed_external_census_tract_partitions", query_ids=list(range(100, 164)),
                domain_bp=[0., 1e6], tracts=[[[0., 1e6, 0]] for _ in range(64)])
            census_ref = write_json(root, f"synthetic_targets/census{parent}.json", census)
            target_headers[parent] = label_header; target_arrays[parent] = label_arrays
            targets.update((labels_ref["path"], census_ref["path"]))
            row.update(status="admitted", reason="Synthetic label-blind candidate", native_inference_returncode=0,
                observed_identity=identity_ref, primary_labels=labels_ref, census_tracts=census_ref)
        attempts.append(row)
    request = {"schema_version": 1, "kind": "confirmatory_custodian_import_request",
        "status": "awaiting_label_blind_release", "lock_sha256": frozen.sha256,
        "parent_seed_registry_sha256": registry_ref["sha256"], "cohort": "id_confirmation",
        "namespace": namespace, "attempts": attempts}
    request_ref = write_json(root, "metadata/import.json", request)
    return {"root": root, "frozen": frozen, "lock_ref": lock_ref, "request": request, "request_ref": request_ref,
        "identities": identities, "sources": sources, "targets": targets,
        "target_headers": target_headers, "target_arrays": target_arrays, "lock_checks": lock_checks}


def release(study, output="release"):
    return custody.prepare_confirmatory_release(study["root"]/study["lock_ref"]["path"], study["lock_ref"]["sha256"],
        study["root"]/study["request_ref"]["path"], study["request_ref"]["sha256"],
        output_directory=study["root"]/output, project_root=study["root"])


def watch_targets(monkeypatch, study):
    reads = []; original = custody.analysis._Capture.read
    def watch(self, ref, **kwargs):
        if self.ref(ref)["path"] in study["targets"]: reads.append(self.ref(ref)["path"])
        return original(self, ref, **kwargs)
    monkeypatch.setattr(custody.analysis._Capture, "read", watch)
    return reads


def test_real_observed_qa_all_decisions_before_targets_and_exact_roster(study, monkeypatch):
    observed = []; original = custody._observed_qa; target = custody._target_integrity
    def qa(identity, *args):
        value = original(identity, *args); observed.append(identity["parent_id"]); return value
    def integrity(*args):
        assert observed == [0, 1]
        return target(*args)
    monkeypatch.setattr(custody, "_observed_qa", qa); monkeypatch.setattr(custody, "_target_integrity", integrity)
    result = release(study)
    assert result.receipt["admitted_parents"] == 2 and result.receipt["attempted_parents"] == 192
    assert result.receipt["precision_shortfall"] and not result.receipt["positive_confirmation_authorized"]
    assert len(result.roster["entries"]) == 24
    assert all(row["status"] == "unresolved" and row["prediction"] is None for row in result.roster["entries"])
    assert result.roster["admission_ledger_sha256"] == result.receipt["admission_ledger"]["sha256"]
    assert all(row["full_parent_electrical_admissible"] for row in result.receipt["observed_qa"])
    assert set(p.name for p in (study["root"]/"release").iterdir()) == set(custody.custodian_schema_values()["outputs"])
    assert custody._sha(Path(result.path).read_bytes()) == result.sha256
    assert not torch.cuda.is_initialized()


def test_real_lock_refusal_before_any_request_observed_or_target_read(study, monkeypatch):
    monkeypatch.setattr(custody.lock_api, "validate_confirmatory_lock", REAL_LOCK_VALIDATOR)
    reads = watch_targets(monkeypatch, study)
    with pytest.raises(ValueError, match="published complete ID"):
        release(study)
    assert reads == [] and not (study["root"]/"release").exists()


@pytest.mark.parametrize("fault", ["shortfall", "order", "blind", "wrong_lock", "target_alias", "source_hash", "seed", "diploids", "tile_edges"])
def test_bad_import_identity_or_hash_aborts_before_target_bytes(study, monkeypatch, fault):
    reads = watch_targets(monkeypatch, study)
    row = study["request"]["attempts"][0]
    if fault == "shortfall": study["request"]["attempts"].pop()
    elif fault == "order": row["parent_id"] = 1
    elif fault == "blind": row["label_blind"] = False
    elif fault == "wrong_lock": study["request"]["lock_sha256"] = "7"*64
    elif fault == "target_alias": row["observed_identity"] = deepcopy(row["primary_labels"])
    elif fault == "source_hash":
        ref = study["sources"][0][0]; (study["root"]/ref["path"]).write_bytes(b"synthetic source drift")
    else:
        identity = study["identities"][0]
        if fault == "seed": identity["parent_seeds"]["ancestry_seed"] += 1
        elif fault == "diploids": identity["diploid_pairs"][0] = [0, 0]
        else: identity["artificial_tile_edges_bp"][0] = 1.
        row["observed_identity"] = write_json(study["root"], row["observed_identity"]["path"], identity)
    rewrite_request(study)
    with pytest.raises((ValueError, FileNotFoundError)): release(study)
    assert reads == [] and not (study["root"]/"release").exists()


@pytest.mark.parametrize("fault", ["native_return", "clades", "target_channel", "electrical"])
def test_observed_only_failure_remains_rejected_without_target_read(study, monkeypatch, fault):
    reads = watch_targets(monkeypatch, study); row = study["request"]["attempts"][0]
    if fault == "native_return": row["native_inference_returncode"] = 2
    elif fault == "electrical":
        original = custody.precompute_leaf_interfaces
        def reject(seq, **kwargs):
            if seq.query_node_ids.tolist() == list(range(100, 164)):
                raise ValueError("Synthetic full-parent conditioning failure")
            return original(seq, **kwargs)
        monkeypatch.setattr(custody, "precompute_leaf_interfaces", reject)
    else:
        ref, header, arrays = study["sources"][0]
        if fault == "clades": arrays["clades"][88, 0] = 0.
        else: arrays["labels"] = np.zeros((128, 64), dtype=np.int64)
        source = write_npz(study["root"], ref["path"], header, arrays)
        identity = study["identities"][0]; identity["source_artifacts"] = [source]
        row["observed_identity"] = write_json(study["root"], row["observed_identity"]["path"], identity)
    rewrite_request(study)
    result = release(study)
    assert result.ledger["attempts"][0]["status"] == "rejected"
    assert result.ledger["attempts"][0]["observed_identity"] is None
    assert all(not name.endswith(("labels0.npz", "census0.json")) for name in reads)
    assert result.receipt["positive_confirmation_authorized"] is False


@pytest.mark.parametrize("fault", ["label_census_disagree", "query_order", "position", "dtype", "header", "census_gap"])
def test_target_integrity_failure_aborts_entire_release(study, fault):
    row = study["request"]["attempts"][0]; arrays = study["target_arrays"][0]; header = study["target_headers"][0]
    if fault == "label_census_disagree": arrays["labels"][0, 0] = 3
    elif fault == "query_order": arrays["query_ids"] = arrays["query_ids"][::-1]
    elif fault == "position": arrays["positions_bp"][0] += 1
    elif fault == "dtype": arrays["labels"] = arrays["labels"].astype(np.uint8)
    elif fault == "header": header["observed_identity_sha256"] = "8"*64
    else:
        census = json.loads((study["root"]/row["census_tracts"]["path"]).read_text())
        census["tracts"][0][0][0] = 1.
        row["census_tracts"] = write_json(study["root"], row["census_tracts"]["path"], census)
    row["primary_labels"] = write_npz(study["root"], row["primary_labels"]["path"], header, arrays)
    rewrite_request(study)
    with pytest.raises(ValueError): release(study)
    assert not (study["root"]/"release").exists()
    assert not list(study["root"].glob(".custodian_pending_*"))


def test_label_counterfactual_does_not_change_admission(study):
    before = release(study, "release_before")
    for parent in (0, 1):
        row = study["request"]["attempts"][parent]
        study["target_arrays"][parent]["labels"].fill(3)
        row["primary_labels"] = write_npz(study["root"], row["primary_labels"]["path"],
            study["target_headers"][parent], study["target_arrays"][parent])
        census = json.loads((study["root"]/row["census_tracts"]["path"]).read_text())
        for tracts in census["tracts"]: tracts[0][2] = 3
        row["census_tracts"] = write_json(study["root"], row["census_tracts"]["path"], census)
    rewrite_request(study); after = release(study, "release_after")
    assert [(r["parent_id"], r["status"], r["reason"]) for r in before.ledger["attempts"]] == [
        (r["parent_id"], r["status"], r["reason"]) for r in after.ledger["attempts"]]
    assert before.receipt["observed_qa"] == after.receipt["observed_qa"]


@pytest.mark.parametrize("fault", ["write", "source_drift", "raced_output", "publish_link"])
def test_release_fault_cleanup_and_raced_output_preservation(study, monkeypatch, fault):
    original_write = custody._write_file
    if fault == "write":
        def broken(handle, body):
            handle.write(body[:17]); raise OSError("Synthetic staged partial write")
        monkeypatch.setattr(custody, "_write_file", broken)
    elif fault == "source_drift":
        def drift(handle, body):
            original_write(handle, body)
            if Path(handle.name).name == "release_receipt.json":
                ref = study["sources"][0][0]
                (study["root"]/ref["path"]).write_bytes(b"synthetic source drift")
        monkeypatch.setattr(custody, "_write_file", drift)
    elif fault == "publish_link":
        original_link = custody.os.link; calls = []
        def fail_link(*args, **kwargs):
            calls.append(None)
            if len(calls) == 2: raise OSError("Synthetic partial directory publication")
            return original_link(*args, **kwargs)
        monkeypatch.setattr(custody.os, "link", fail_link)
    else:
        def race(handle, body):
            original_write(handle, body)
            if Path(handle.name).name == "release_receipt.json":
                output = study["root"]/"release"; output.mkdir()
                (output/"unrelated.txt").write_bytes(b"preserve unrelated raced file")
        monkeypatch.setattr(custody, "_write_file", race)
    with pytest.raises((OSError, ValueError)): release(study)
    assert not list(study["root"].glob(".custodian_pending_*"))
    if fault == "raced_output":
        assert (study["root"]/"release"/"unrelated.txt").read_bytes() == b"preserve unrelated raced file"
    else: assert not (study["root"]/"release").exists()


def test_in_memory_admission_and_captured_support_drift(study):
    admitted = custody.admit_custodian(study["root"]/study["lock_ref"]["path"], study["lock_ref"]["sha256"],
        project_root=study["root"])
    admitted.methods["candidate"]["seed_bundles"][0]["checkpoint"]["sha256"] = "a"*64
    with pytest.raises(ValueError, match="In-memory"): admitted.check()
    admitted.methods = deepcopy(custody.analysis._methods(study["frozen"].payload))
    ref = custody.custodian_schema_values()["code_artifacts"]["observed_electrical"]
    (study["root"]/ref["path"]).write_bytes(b"synthetic support drift")
    with pytest.raises(ValueError): admitted.check(full=True)



def test_target_integrity_cannot_mutate_already_sealed_admission(study, monkeypatch):
    original = custody._target_integrity
    def mutate(admission, row, identity):
        original(admission, row, identity)
        identity["query_ids"][0] = 999
    monkeypatch.setattr(custody, "_target_integrity", mutate)
    with pytest.raises(ValueError, match="Sealed admission"):
        release(study)
    assert not (study["root"]/"release").exists()


@pytest.mark.parametrize("count", [160, 161])
def test_fixed_first160_admission_policy_without_extra_target_reads(study, monkeypatch, count):
    # Count-policy fixture only: observed identity/QA and target integrity are
    # explicitly mocked here, not a fabricated160-parent actual admission.
    template = deepcopy(study["request"]["attempts"][0])
    study["request"]["attempts"] = [dict(deepcopy(template), parent_id=i) for i in range(count)]
    calls = []
    monkeypatch.setattr(custody, "_identity", lambda admission, row, forbidden:
        (row["observed_identity"], {"parent_id": row["parent_id"]}, b"synthetic policy-only bytes"))
    monkeypatch.setattr(custody, "_observed_qa", lambda *args:
        {"max_graph_nodes": 1, "graphs": 1, "maximum_interior_condition": 1.,
         "full_parent_electrical_admissible": True, "full_parent_canonicalization_admissible": True})
    monkeypatch.setattr(custody, "_target_integrity", lambda admission, row, identity: calls.append(row["parent_id"]))
    rewrite_request(study)
    if count == 161:
        with pytest.raises(ValueError, match="beyond first160"): release(study)
        assert calls == [] and not (study["root"]/"release").exists()
    else:
        result = release(study)
        assert result.receipt["admitted_parents"] == 160 and not result.receipt["precision_shortfall"]
        assert len(result.roster["entries"]) == 1920 and calls == list(range(160))
        assert not result.receipt["positive_confirmation_authorized"]


def test_missing_published_support_provenance_refused_before_targets(study, monkeypatch):
    reads = watch_targets(monkeypatch, study)
    study["frozen"].payload["captured_artifacts"].pop()
    with pytest.raises(ValueError, match="support code lacks"):
        release(study)
    assert reads == [] and not (study["root"]/"release").exists()

    
def test_insufficient_observed_coverage_is_explicit_label_blind_rejection(study, monkeypatch):
    reads = watch_targets(monkeypatch, study)
    identity = study["identities"][0]
    for prefix, points in (("primary", 128), ("dense", 4096)):
        identity[prefix+"_grid"] = coverage_clipped_midpoint_grid(domain_bp=(0., 1e6),
            coverage_bp=(0., 900000.), grid_id="synthetic_"+prefix, grid_points=points).identity_record()
    ref, header, arrays = study["sources"][0]
    arrays["window_bounds"] = np.asarray([0., .9], dtype=np.float64)
    arrays["intervals"] = np.asarray([[0., .9]], dtype=np.float64)
    identity["source_artifacts"] = [write_npz(study["root"], ref["path"], header, arrays)]
    row = study["request"]["attempts"][0]
    row["observed_identity"] = write_json(study["root"], row["observed_identity"]["path"], identity)
    rewrite_request(study)
    result = release(study)
    assert result.ledger["attempts"][0]["status"] == "rejected"
    assert "coverage" in result.ledger["attempts"][0]["reason"]
    assert all(not name.endswith(("labels0.npz", "census0.json")) for name in reads)

    
@pytest.mark.parametrize("fault", ["same_inode_bytes", "replacement_inode"])
def test_staged_output_drift_is_rejected_and_foreign_replacement_preserved(tmp_path, fault):
    output = tmp_path/"release"
    fired = []
    def mutate(**kwargs):
        pending = list(tmp_path.glob(".custodian_pending_*/release_receipt.json"))
        if pending and not fired:
            path = pending[0]; fired.append(path)
            if fault == "replacement_inode":path.unlink()
            path.write_bytes(b"foreign staged mutation")
    with pytest.raises(ValueError, match="Staged"):
        custody._publish(output, {"release_receipt.json": b"expected immutable receipt"}, mutate)
    assert not output.exists()
    if fault == "replacement_inode":
        assert fired[0].read_bytes() == b"foreign staged mutation"
    else:
        assert not list(tmp_path.glob(".custodian_pending_*"))


@pytest.fixture
def terminal_study(tmp_path, monkeypatch):
    # Shared handmade NPZ fixture; ALL positive published-lock and GPU metadata
    # claims are explicit mocks. No model execution or CUDA evidence is produced.
    from test_confirmatory_analysis_runner import fixture_study
    study = fixture_study(tmp_path, monkeypatch)
    study["frozen"].check = lambda full=True: None
    pins = custody.custodian_schema_values()["code_artifacts"]
    for ref in pins.values():
        path = tmp_path/ref["path"]; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((custody.ROOT/ref["path"]).read_bytes())
    study["frozen"].payload["code_artifacts"]["custodian"] = pins["custodian"] | {"entry_point": "prepare_confirmatory_release"}
    study["frozen"].payload["captured_artifacts"] = [
        {**ref, "bytes": (tmp_path/ref["path"]).stat().st_size} for ref in pins.values()]
    study["terminal_refs"] = [deepcopy(row["terminal_receipt"]) for row in study["roster"]["entries"]]
    for row in study["roster"]["entries"]:
        row.update(status="unresolved", prediction=None, terminal_receipt=None, reason="Awaiting frozen operational prediction")
    study["initial_ref"] = write_json(tmp_path, "metadata/initial_roster.json", study["roster"])
    return study


def seal_roster(study, refs=None, output="sealed_roster"):
    return custody.seal_confirmatory_prediction_roster(
        study["root"]/study["lock_ref"]["path"], study["lock_ref"]["sha256"],
        study["root"]/study["ledger_ref"]["path"], study["ledger_ref"]["sha256"],
        study["root"]/study["initial_ref"]["path"], study["initial_ref"]["sha256"],
        terminal_receipts=study["terminal_refs"] if refs is None else refs,
        output_directory=study["root"]/output, project_root=study["root"])


def test_terminal_roster_qualification_is_prediction_only_and_preserves_full_matrix(terminal_study, monkeypatch):
    reads = watch_targets(monkeypatch, terminal_study)
    result = seal_roster(terminal_study)
    assert result.receipt["cell_counts"] == {"success": 36, "failed": 0, "unresolved": 0}
    assert result.receipt["matrix_terminal"] and reads == []
    assert all(row["terminal_receipt"] is not None for row in result.roster["entries"])
    assert not result.receipt["positive_confirmation_authorized"]
    assert not torch.cuda.is_initialized()


def test_missing_terminal_cells_remain_unresolved_and_cannot_join_targets(terminal_study, monkeypatch):
    reads = watch_targets(monkeypatch, terminal_study)
    result = seal_roster(terminal_study, terminal_study["terminal_refs"][:1])
    assert result.receipt["cell_counts"] == {"success": 1, "failed": 0, "unresolved": 35}
    assert not result.receipt["matrix_terminal"] and reads == []
    admitted = custody.analysis.admit_analysis(
        terminal_study["root"]/terminal_study["lock_ref"]["path"], terminal_study["lock_ref"]["sha256"],
        result.path, result.sha256, terminal_study["root"]/terminal_study["ledger_ref"]["path"],
        terminal_study["ledger_ref"]["sha256"], project_root=terminal_study["root"])
    with pytest.raises(ValueError, match="Complete terminal prediction roster"):
        custody.analysis.analyze_confirmatory(admitted)
    assert reads == []


def test_actual_failed_terminal_receipt_is_preserved_with_null_prediction(terminal_study, monkeypatch):
    reads = watch_targets(monkeypatch, terminal_study)
    ref = terminal_study["terminal_refs"][0]
    value = json.loads((terminal_study["root"]/ref["path"]).read_text())
    for name in ("captured_artifacts", "cuda_execution_performed", "cuda_allocator_peak_bytes"):value.pop(name)
    value.update(status="failed", prediction=None, orphan_prediction=None, error_type="SyntheticOperationalFailure",
        reason="Explicit mocked operational runtime failure", phase="runtime_initialization",
        runtime={"device_kind": "cuda", "initialization_stage": "failed"}, cpu_validation_only=False)
    failed = write_json(terminal_study["root"], ref["path"], value)
    result = seal_roster(terminal_study, [failed])
    row = result.roster["entries"][0]
    assert row["status"] == "failed" and row["prediction"] is None and row["terminal_receipt"] == failed
    assert result.receipt["cell_counts"] == {"success": 0, "failed": 1, "unresolved": 35}
    assert reads == [] and not result.receipt["positive_confirmation_authorized"]


@pytest.mark.parametrize("fault", ["cpu", "source", "stack", "gpu", "process", "checkpoint",
                                  "prediction_header", "duplicate", "target_alias"])
def test_unqualified_or_mismatched_terminal_cannot_seal_before_targets(terminal_study, monkeypatch, fault):
    reads = watch_targets(monkeypatch, terminal_study)
    ref = terminal_study["terminal_refs"][0]
    value = json.loads((terminal_study["root"]/ref["path"]).read_text())
    if fault == "target_alias": refs = [terminal_study["ledger"]["attempts"][0]["census_tracts"]]
    elif fault == "duplicate": refs = [ref, deepcopy(ref)]
    else:
        if fault == "cpu": value.update(cpu_validation_only=True, cuda_execution_performed=False)
        elif fault == "source":
            name = next(iter(value["imported_source_sha256"])); value["imported_source_sha256"][name] = "f"*64
        elif fault == "stack": value["runtime"]["software"]["torch_build"] = "different mocked CUDA build"
        elif fault == "gpu": value["runtime"]["physical_gpu"] = 0
        elif fault == "process": value["runtime"]["process_title"] = "wrong process title"
        elif fault == "checkpoint": value["cell"]["checkpoint"]["sha256"] = "f"*64
        else:
            prediction = value["prediction"]; path = terminal_study["root"]/prediction["path"]
            with np.load(path, allow_pickle=False) as data:arrays = {name: data[name] for name in data.files}
            header = json.loads(arrays.pop("metadata_json").item()); header["observed_identity_sha256"] = "f"*64
            value["prediction"] = write_npz(terminal_study["root"], prediction["path"], header, arrays)
        refs = [write_json(terminal_study["root"], ref["path"], value)]
    with pytest.raises(ValueError):
        seal_roster(terminal_study, refs)
    assert reads == [] and not (terminal_study["root"]/"sealed_roster").exists()


def test_terminal_receipt_drift_during_publication_aborts_whole_roster(terminal_study, monkeypatch):
    reads = watch_targets(monkeypatch, terminal_study); original = custody._write_file
    def drift(handle, body):
        original(handle, body)
        if Path(handle.name).name == "roster_seal_receipt.json":
            ref = terminal_study["terminal_refs"][0]
            (terminal_study["root"]/ref["path"]).write_bytes(b"Explicit synthetic terminal source drift")
    monkeypatch.setattr(custody, "_write_file", drift)
    with pytest.raises(ValueError):
        seal_roster(terminal_study, terminal_study["terminal_refs"][:1])
    assert reads == [] and not (terminal_study["root"]/"sealed_roster").exists()


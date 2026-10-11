"""Fresh handmade native TS fixtures only. No actual inputs, models or root admission."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
import tskit
from arggnn import census_tsinfer_observation_v1 as bridge

torch.set_num_threads(1)
ROOT = bridge.ROOT
AUTHOR = ROOT / "research/reviews/census_tsinfer_observation_v1_author"
CFG = bridge.load_config_v1()
BINDINGS = {k: {"path": f"research/reviews/census_tsinfer_observation_v1_author/handmade/{k}",
                 "sha256": "a" * 64}
            for k in ("native_observed_ts", "serialized_whole_observed", "observed_roster",
                      "observed_grid", "producer_receipt")}
LINEAGE = {"parent_id": "bio_paths_pilot_train_000", "split": "train",
           "namespace": "handmade:not_a_real_ARG", "fixture_mode": True}

@pytest.fixture(autouse=True)
def forbid_models_and_cuda(monkeypatch):
    from arggnn import census_native_qualification_runner_v1 as old
    def forbidden(*args, **kwargs):
        raise AssertionError("No model or CUDA operation belongs to author fixtures")
    monkeypatch.setattr(old, "_model", forbidden)
    for name in ("is_available", "device_count", "current_device", "init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)

def handmade(full=False, *, unary=False, population_flip=False, group_time=300.):
    t = tskit.TableCollection(6400.)
    for _ in range(5):
        t.populations.add_row()
    if full:
        cohorts = ["present_day"] * 64 + ["Bronze_Age"] * 16 + ["BAA"] * 8 + ["Neo"] * 16 + ["Yam"] * 16
        roles = ["query"] * 120 + ["mixed_reference_only"] * 64 + ["pure_source_reference"] * 96
        classes = [None] * 184 + [c for c in range(4) for _ in range(24)]
        ages = [{"present_day": 0., "Bronze_Age": 135., "BAA": 155., "Neo": 200., "Yam": 170.}[c] for c in cohorts] + [150.] * 64 + [176.] * 24 + [473.] * 24 + [200.] * 24 + [300.] * 24
        cohorts += ["UNLABELED_CONTEXT"] * 64 + [CFG["classes"][c] for c in range(4) for _ in range(24)]
    else:
        roles = ["query"] * 2 + ["mixed_reference_only"] * 2 + ["pure_source_reference"] * 4
        classes = [None] * 4 + list(range(4))
        ages = [0., 100., 150., 175., 180., 220., 400., 473.]
        cohorts = ["present_day", "Neo", "UNKNOWN_CONTEXT", "UNKNOWN_CONTEXT"] + CFG["classes"]
    nodes = []
    for i, (age, cls) in enumerate(zip(ages, classes)):
        t.nodes.add_row(time=1000., population=4)  # globally unused anonymous row
        nodes.append(t.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=age,
                                    population=(i % 5 + (1 if population_flip else 0)) % 5))
    roots = [t.nodes.add_row(time=900., population=4) for _ in range(2)]
    for graph, (left, right) in enumerate(((0., 3200.), (3200., 6400.))):
        if full:
            for node in nodes:
                t.edges.add_row(left, right, roots[graph], node)
        else:
            pair_sets = ((0, 1), (2, 3)) if graph == 0 else ((0, 2), (1, 3))
            for pair in pair_sets:
                ancestor = t.nodes.add_row(time=group_time, population=4)
                t.edges.add_row(left, right, roots[graph], ancestor)
                for j in pair:
                    parent = ancestor
                    if unary and graph == 0 and j == pair[0]:
                        parent = t.nodes.add_row(time=(ages[j] + group_time) / 2., population=4)
                        t.edges.add_row(left, right, ancestor, parent)
                    t.edges.add_row(left, right, parent, nodes[j])
            for node in nodes[4:]:
                t.edges.add_row(left, right, roots[graph], node)
    t.sort()
    ts = t.tree_sequence()
    original, derived, mapping = [], [], []
    for i, (node, role, age, cls, cohort) in enumerate(zip(nodes, roles, ages, classes, cohorts)):
        raw = {"sample_index": i, "sample_id": f"handmade_subject_{i // 2}", "haplotype": i % 2,
               "observed_node_id": 10000 + i, "observed_cohort": cohort, "age_generations": age,
               "role": role, "observed_reference_class": cls}
        row = dict(raw)
        row["source_observed_node_id"] = row.pop("observed_node_id")
        row["observed_node_id"] = row["inferred_tree_node_id"] = node
        original.append(raw)
        derived.append(row)
        mapping.append({"sample_index": i, "inferred_tree_node_id": node})
    roster = bridge.role_join_v1(ts, original, derived, mapping, CFG, fixture=not full)
    return ts, roster, original, derived, mapping

def positions():
    return np.arange(50., 2_000_000., 100., dtype=np.float64)

def certificate(full=False, **kwargs):
    ts, roles, *_ = handmade(full, **kwargs)
    return bridge.certify_native_tile_v1(ts, roles, positions(), [0., 6400.], 0,
                                        BINDINGS, LINEAGE, CFG, fixture=not full)[0]

def test_standard_library_default():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; from arggnn.census_tsinfer_observation_v1 import source_report_v1; "
        "r=source_report_v1(); assert 'torch' not in sys.modules and 'numpy' not in sys.modules "
        "and 'tskit' not in sys.modules; print(r['status'])"],
        cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "source_only_no_actual_admission" in result.stdout

def test_extractor_AST_exact_and_dependency_drift_rejected():
    report = bridge.source_report_v1(CFG)
    assert report["extractor_AST_sha256"] == CFG["extractor"]["AST_sha256"]
    broken = deepcopy(CFG)
    broken["source_sha256"]["src/arggnn/reference_encoder.py"] = "0" * 64
    with pytest.raises(ValueError, match="Pinned dependency"):
        bridge.source_report_v1(broken)
    broken = deepcopy(CFG)
    broken["extractor"]["AST_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="AST changed"):
        bridge.source_report_v1(broken)

def test_full280_exact_role_ID_dates_and_query_strata():
    ts, roles, original, derived, mapping = handmade(full=True)
    assert len(roles) == 280 and len(ts.samples()) == 280
    assert [r.actual_id for r in roles] == list(range(280))
    assert [r.native_sample_node for r in roles] != list(range(280))
    assert sum(r.role == "query" for r in roles) == 120
    assert sum(r.role == "mixed_reference_only" for r in roles) == 64
    assert sum(r.role == "pure_source_reference" for r in roles) == 96
    changed = deepcopy(original)
    changed[0]["observed_cohort"] = "Neo"
    altered = deepcopy(derived)
    altered[0]["observed_cohort"] = "Neo"
    with pytest.raises(ValueError, match="cohort contract"):
        bridge.role_join_v1(ts, changed, altered, mapping, CFG)

@pytest.mark.parametrize("fault", ["extra_population", "query_class", "date", "native_ID", "missing_leaf", "context_350", "diploid"])
def test_role_and_label_leakage_rejections(fault):
    ts, roles, original, derived, mapping = handmade(full=True)
    if fault == "extra_population":
        original[0]["sampling_population"] = 0
    elif fault == "query_class":
        original[0]["observed_reference_class"] = derived[0]["observed_reference_class"] = 0
    elif fault == "date":
        original[0]["age_generations"] = derived[0]["age_generations"] = 1.
    elif fault == "native_ID":
        mapping[0]["inferred_tree_node_id"] = mapping[1]["inferred_tree_node_id"]
    elif fault == "missing_leaf":
        original.pop()
        derived.pop()
        mapping.pop()
    elif fault == "context_350":
        original[120]["age_generations"] = derived[120]["age_generations"] = 350.
    elif fault == "diploid":
        original[1]["sample_id"] = derived[1]["sample_id"] = "different"
    with pytest.raises(ValueError):
        bridge.role_join_v1(ts, original, derived, mapping, CFG)

def test_leaf_proxy_mask_query120_and_context64_zero():
    ts, roles, *_ = handmade(full=True)
    raw, _, trees, landmarks, *_ = bridge.extract_native_tsinfer_tile_v1(ts, roles, positions(), [0., 6400.], 0, CFG)
    seq = raw["sequence"]
    assert seq["x"].shape[1] == 48 and len(trees) == 2 and len(landmarks) == 16
    assert seq["query_node_ids"].numel() == 184
    assert int(seq["x"][:, :4].sum()) == 192
    assert int(seq["x"][:, 4].sum()) == 240
    context = {r.native_sample_node for r in roles if r.role == "mixed_reference_only"}
    queries = {r.native_sample_node for r in roles if r.role == "query"}
    for node, x in zip(seq["node_ids"].tolist(), seq["x"]):
        if node in context:
            assert x[4] == 0 and x[:4].sum() == 0
        if node in queries:
            assert x[:4].sum() == 0
        if node == -1:
            assert x[:4].sum() == 0
    assert raw["labels_in_input"] is False

def test_population_annotation_is_never_feature_or_cut_input():
    a, b = certificate(), certificate(population_flip=True)
    assert torch.equal(a.native_certificate.stored_sequence.x, b.native_certificate.stored_sequence.x)
    assert torch.equal(a.native_certificate.bank.group_index, b.native_certificate.bank.group_index)
    assert a.certificate_sha256 == b.certificate_sha256

def test_every_full_native_tree_and_source_dtype():
    c = certificate(full=True).native_certificate
    assert c.provenance["original_tree_ids"] == [0, 1]
    assert [len(t.parent) for t in c.native_trees] == [281, 281]
    assert c.stored_sequence.x.dtype == torch.float32
    assert c.stored_sequence.intervals.dtype == torch.float64
    assert all(t.node_times.dtype == torch.float64 for t in c.native_trees)
    assert c.provenance["feature_columns"] == tuple(CFG["feature_columns"])
    assert all((x >= 0).all() for x in c.provenance["native_vertex_by_source_row"])

def test_original_readouts_right_boundary_no_tile_normalization():
    c = certificate().native_certificate
    assert torch.equal(c.stored_sequence.positions, torch.arange(50., 6400., 100., dtype=torch.float64) / 2_000_000.)
    assert c.stored_sequence.window_bounds.tolist() == [0., 6400. / 2_000_000.]
    assert c.stored_sequence.target_graph[31] == 0
    assert c.stored_sequence.target_graph[32] == 1
    assert c.provenance["original_grid_indexes"].tolist() == list(range(64))

@pytest.mark.parametrize("coverage,support,denom", [
    ([273., 1999681.], [300., 1999681.], 127960384.),
    ([116., 1999724.], [116., 1999700.], 127973376.)])
def test_exact_common_score_bin_edges(coverage, support, denom):
    pos, edges = positions(), np.arange(0., 2_000_001., 100., dtype=np.float64)
    kept = np.flatnonzero((pos >= coverage[0]) & (pos < coverage[1]))
    helper = {"coordinate_contract": {"coverage_original_bp0": coverage, "local_to_original_offset_bp": coverage[0],
        "original_parent_interval_bp0": [0, 2_000_000], "normalization_length_bp": 2_000_000,
        "physical_coordinate_transform": "physical_original = local_tree_position + coverage_left",
        "retained_fixed100bp_indexes": kept.tolist()}}
    actual, clipped = bridge.physical_contract_v1(helper, pos, edges, CFG)
    assert actual.tolist() == kept.tolist()
    assert [clipped[0], clipped[-1]] == support and (clipped[-1] - clipped[0]) * 64 == denom
    wrong = pos.copy()
    wrong[0] += 1
    with pytest.raises(ValueError, match="coordinate contract"):
        bridge.physical_contract_v1(helper, wrong, edges, CFG)

def test_unary_suppression_only_after_full_native_feature_parity():
    plain, serial = certificate(), certificate(unary=True)
    a, b = plain.native_certificate, serial.native_certificate
    assert len(b.native_trees[0].parent) > len(a.native_trees[0].parent)
    assert len(b.inputs[0].parent) == len(a.inputs[0].parent)
    assert torch.equal(a.bank.group_index, b.bank.group_index)
    assert torch.equal(a.bank.donor_weights, b.bank.donor_weights)
    for before, after in zip(a.inputs, b.inputs):
        for actual_id in before.sample_ids.tolist():
            row_a = int(before.sample_nodes[before.sample_ids == actual_id])
            row_b = int(after.sample_nodes[after.sample_ids == actual_id])
            assert torch.equal(before.features[row_a], after.features[row_b])

def test_direct_float64_census_collision_not_encoded_age_inversion():
    equal, older = certificate(group_time=350.), certificate(group_time=350.0000001)
    assert torch.equal(equal.native_certificate.stored_sequence.x, older.native_certificate.stored_sequence.x)
    assert not torch.equal(equal.native_certificate.bank.group_index, older.native_certificate.bank.group_index)
    assert equal.certificate_sha256 != older.certificate_sha256

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_CPU32_64_roundtrip_and_unchanged_integer_geometry_contract(dtype):
    c = certificate()
    bridge.validate_tsinfer_certificate_v1(c, c.certificate_sha256, CFG)
    ts, roles, *_ = handmade()
    fresh = bridge.extract_native_tsinfer_tile_v1(ts, roles, positions(), [0., 6400.], 0, CFG, fixture=True)[0]
    restored = torch.load(io.BytesIO(bridge.raw_sequence_bytes_v1(fresh, CFG)), map_location="cpu", weights_only=True)
    assert all(torch.equal(fresh["sequence"][k], restored["sequence"][k]) if isinstance(fresh["sequence"][k], torch.Tensor)
               else fresh["sequence"][k] is restored["sequence"][k] for k in fresh["sequence"])
    bank, forest = c.native_certificate.bank.to("cpu", dtype), c.native_certificate.forest.to("cpu", dtype)
    assert bank.donor_weights.dtype == dtype and forest.node.dtype == dtype
    assert torch.equal(bank.query_ids, c.native_certificate.bank.query_ids)
    assert torch.equal(bank.target_graph, c.native_certificate.bank.target_graph)
    assert bank.layout.coordinates.dtype == torch.float64 and bank.layout.anchors.dtype == torch.float64
    assert torch.equal(forest.parent, c.native_certificate.forest.parent)

@pytest.mark.parametrize("fault", ["feature", "mask", "labels", "extra_field", "edge", "position"])
def test_source_and_cache_contamination_rejected(fault):
    ts, roles, *_ = handmade()
    raw = bridge.extract_native_tsinfer_tile_v1(ts, roles, positions(), [0., 6400.], 0, CFG, fixture=True)[0]
    changed = deepcopy(raw)
    if fault == "feature":
        changed["sequence"]["x"][0, 8] += 0.01
    elif fault == "mask":
        row = int(torch.nonzero(changed["sequence"]["node_ids"] == roles[0].native_sample_node, as_tuple=False)[0, 0])
        changed["sequence"]["x"][row, 0] = 1.
    elif fault == "labels":
        changed["labels_in_input"] = True
    elif fault == "extra_field":
        changed["sequence"]["query_population"] = torch.zeros(4)
    elif fault == "edge":
        changed["sequence"]["edge_index"] = changed["sequence"]["edge_index"][:, 2:].clone()
    elif fault == "position":
        changed["sequence"]["positions"][0] += 1e-8
    with pytest.raises(ValueError):
        bridge.certify_native_tile_v1(ts, roles, positions(), [0., 6400.], 0,
                                     BINDINGS, LINEAGE, CFG, fixture=True, cached_raw=changed)

def test_changed_native_sample_roster_or_gap_rejected():
    ts, roles, *_ = handmade()
    bad = list(roles)
    bad[0] = replace(bad[0], native_sample_node=roles[1].native_sample_node)
    with pytest.raises(ValueError):
        bridge.extract_native_tsinfer_tile_v1(ts, bad, positions(), [0., 6400.], 0, CFG, fixture=True)
    with pytest.raises(ValueError, match="span"):
        bridge.extract_native_tsinfer_tile_v1(ts, roles, positions(), [0., 6300.], 0, CFG, fixture=True)

def test_stale_outer_native_or_lineage_certificate_rejected():
    cert = certificate()
    changed = replace(cert, lineage=dict(cert.lineage, namespace="other_ARG"))
    with pytest.raises(ValueError, match="same-tsinfer"):
        bridge.validate_tsinfer_certificate_v1(changed, cert.certificate_sha256, CFG)
    cert.native_certificate.stored_sequence.x[0, 8] += 0.01
    with pytest.raises(ValueError, match="Changed/stale"):
        bridge.validate_tsinfer_certificate_v1(cert, cert.certificate_sha256, CFG)

def test_no_actual_admission_from_source_or_handmade_review(tmp_path):
    bad = {"schema": "census_tsinfer_observation_admission_v1", "status": "inactive"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    ref = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="root observation-only"):
        bridge.admit_phase_v1(ref, "inspect")

def test_reader_rejects_byte_drift_symlink_and_external_scope(tmp_path):
    reader = bridge.VerifiedObservationReaderV1()
    p = tmp_path / "bytes"
    p.write_bytes(b"handmade")
    with pytest.raises(ValueError, match="bytes changed"):
        reader.read({"path": str(p), "sha256": "0" * 64})
    q = tmp_path / "linked"
    q.symlink_to(p)
    with pytest.raises(ValueError, match="symlink"):
        reader.read({"path": str(q), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="External"):
        reader.record({"path": "/outside-project/file", "sha256": "0" * 64})


@pytest.mark.parametrize("fault", ["sampled_interior", "disconnected_sample"])
def test_sampled_interior_and_sample_bearing_second_root_fail(fault):
    ts, roles, *_ = handmade()
    tables = ts.dump_tables()
    # Rebuild edges by hand, changing only observed topology of one sample path.
    rows = [(e.left, e.right, e.parent, e.child) for e in tables.edges]
    sample = roles[0].native_sample_node
    if fault == "disconnected_sample":
        rows = [r for r in rows if not (r[0] == 0. and r[3] == sample)]
    else:
        ancestor_sample = roles[-1].native_sample_node  # age473 > focal age0
        rows = [(l, r, ancestor_sample if l == 0. and c == sample else p, c) for l, r, p, c in rows]
    tables.edges.clear()
    for l, r, p, c in rows:
        tables.edges.add_row(l, r, p, c)
    tables.sort()
    changed = tables.tree_sequence()
    with pytest.raises(ValueError, match="complete connected trees"):
        bridge.extract_native_tsinfer_tile_v1(changed, roles, positions(), [0., 6400.], 0, CFG, fixture=True)


@pytest.mark.parametrize("fault", [False, True])
def test_budget_v4_readonly_call_contract_without_live_ledger(fault):
    class Stub:
        def __init__(self): self.calls = []
        def check_reservation(self, root, **kwargs):
            self.calls.append(kwargs)
            return {"reservation": kwargs["reservation"], "limit_bytes": 950000000000,
                    "hard_limit_bytes": 1000000000000, "projected_bytes": 949999999999,
                    "legacy_ledger_mutated": False, "supplemental_ledger_mutated": bool(fault)}
    stub = Stub()
    storage = {"reservation": {"handmade_only": True},
               "initial_ledger_pins": {"legacy_ledger_sha256": "a"*64, "supplemental_ledger_sha256": "b"*64}}
    if fault:
        with pytest.raises(ValueError, match="readonly950GB"):
            bridge.budget_call_v1(stub, storage, CFG, first=True, required_write_bytes=123)
    else:
        bridge.budget_call_v1(stub, storage, CFG, first=True, required_write_bytes=123)
        bridge.budget_call_v1(stub, storage, CFG, required_write_bytes=0)
        assert stub.calls[0]["initial_ledger_pins"] == storage["initial_ledger_pins"]
        assert stub.calls[1]["initial_ledger_pins"] is None
        assert stub.calls[0]["required_write_bytes"] == 123 and stub.calls[0]["boundary"] is True
        assert stub.calls[0]["expected_source_sha256"] == CFG["storage_companion"]["implementation"]["sha256"]

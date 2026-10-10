"""Independent narrow support-repair regressions; no inference/date execution."""
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import tskit

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
AUDIT = ROOT / "research/reviews/biological_paths_tsinfer_v1_source_audit"
spec = importlib.util.spec_from_file_location("independent_support_v2", ROOT / "scripts/infer_biological_paths_tsinfer_v1.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def qualification_native_style(*, gap=None, edge_left=10, edge_right=61):
    # Fresh handwritten tables with noncontiguous IDs, all six fixed input sites,
    # nonzero ages, two marginal trees, and empty native-style unscored tails.
    case = m.handmade_assets()
    t = tskit.TableCollection(2000000)
    inds = [t.individuals.add_row() for _ in range(4)]
    samples = list(range(0, 16, 2))
    for node in range(16):
        if node % 2:
            t.nodes.add_row(time=500 + node)
        else:
            i = node // 2
            t.nodes.add_row(time=case["ages"][i], flags=tskit.NODE_IS_SAMPLE,
                            individual=inds[i//2])
    a, b, root = [t.nodes.add_row(time=x) for x in [70, 80, 200]]
    for lo, hi, pair, internal in [(10, 35, samples[:2], a), (35, 61, samples[4:6], b)]:
        lo, hi = max(lo, edge_left), min(hi, edge_right)
        pieces = [(lo, hi)] if gap is None else [(lo, min(hi, gap[0])), (max(lo, gap[1]), hi)]
        for left, right in pieces:
            if left >= right:
                continue
            for sample in samples:
                t.edges.add_row(left, right, internal if sample in pair else root, sample)
            t.edges.add_row(left, right, root, internal)
    for site in case["sites"]:
        site_id = t.sites.add_row(site["position"], "A")
        for i, value in enumerate(site["genotypes"]):
            if value:
                t.mutations.add_row(site_id, samples[i], "G", time=case["ages"][i] + 1)
    t.sort()
    return t.tree_sequence(), case


def test_support_qualified_handmade_output_preserves_every_input_and_native_table():
    ts, case = qualification_native_style()
    before = ts.dump_tables()
    assert m.check_samples(ts, case["ages"]) == list(range(0, 16, 2))
    m.check_genotypes(ts, case["sites"])
    m.strict_times(ts, case["ages"])
    result = m.qualification_summary({"minimum_connected_internal_reuse": 1}, case, ts)
    assert case["identity_support_original_bp0"] == [10, 61]
    assert result["identity_support_original_bp0"] == [10, 61]
    assert result["status"] == "passed"
    assert result["identity"]["native_trees_total"] == 4
    assert result["identity"]["trees"] == 2
    assert result["identity"]["internal_reuse_occurrences"] == 1
    assert ts.num_sites == 6 and ts.sequence_length == 2000000
    assert ts.tables.equals(before)
    # Default full-length inspection remains strict, rather than silently shrinking.
    with pytest.raises(ValueError, match="inside identity support"):
        m.require_identity(ts, 1)


@pytest.mark.parametrize("left,right", [(9.999, 61), (10, 61.001)])
def test_requested_support_cannot_shrink_to_available_native_topology(left, right):
    ts, case = qualification_native_style()
    before = ts.dump_tables()
    with pytest.raises(ValueError, match="inside identity support"):
        m.require_identity(ts, 1, support=[left, right])
    assert ts.tables.equals(before)


@pytest.mark.parametrize("gap", [(34.999, 35.001), (44.125, 44.25)])
def test_gap_between_scoring_centers_fails_identity_and_qualification(gap):
    ts, case = qualification_native_style(gap=gap)
    before = ts.dump_tables()
    # The only fixed100bp center here is 50, but coverage is continuous.
    assert m.grid_indexes(10, 61) == [0]
    with pytest.raises(ValueError, match="inside identity support"):
        m.require_identity(ts, 1, support=[10, 61])
    with pytest.raises(ValueError, match="inside identity support"):
        m.qualification_summary({"minimum_connected_internal_reuse": 1}, case, ts)
    assert ts.tables.equals(before)


def test_missing_identity_support_does_not_fall_back_to_sites_or_centers():
    ts, case = qualification_native_style()
    case.pop("identity_support_original_bp0")
    with pytest.raises(KeyError):
        m.qualification_summary({"minimum_connected_internal_reuse": 1}, case, ts)


def test_scored_clip_metrics_equal_original_interval_metrics_without_node_changes():
    ts, case = qualification_native_style()
    before = ts.dump_tables()
    native = m.require_identity(ts, 1, support=[19, 55])
    clean = m.sanitize_clip(ts, case["ages"], 19, 55)
    local = m.require_identity(clean, 1)
    assert ts.tables.equals(before)
    assert clean.num_nodes == ts.num_nodes and np.array_equal(clean.nodes_time, ts.nodes_time)
    assert clean.samples().tolist() == ts.samples().tolist()
    assert clean.sequence_length == 36
    m.check_genotypes(clean, [{**s, "position": s["position"]-19} for s in case["sites"] if 19 <= s["position"] < 55])
    for key in ["trees", "connected_internal_occurrences", "unique_connected_internal_ids",
                "internal_reuse_occurrences", "adjacent_shared_internal_min", "adjacent_shared_internal_max"]:
        assert local[key] == native[key]
    assert native["evaluated_support_bp0"] == [19, 55]
    assert local["evaluated_support_bp0"] == [0, 36]


def test_scientific_operations_and_fixed_contract_are_unchanged_from_blocked_bytes():
    old = ast.parse((AUDIT / "blocked_revision_v1/infer_biological_paths_tsinfer_v1.py").read_text())
    new = ast.parse((ROOT / "scripts/infer_biological_paths_tsinfer_v1.py").read_text())
    old_functions = {n.name:n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name:n for n in new.body if isinstance(n, ast.FunctionDef)}
    for name in ["validate_config", "validate_roster", "parse_vcf", "grid_indexes",
                 "check_samples", "check_genotypes", "strict_times", "sanitize_clip",
                 "admission", "infer_parent", "source_only"]:
        assert ast.dump(old_functions[name]) == ast.dump(new_functions[name]), name
    def native_calls(tree):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id in {"tsinfer", "tsdate"}]
    assert native_calls(old_functions["pipeline"]) == native_calls(new_functions["pipeline"])
    config = json.loads((ROOT / "configs/biological_paths_tsinfer_v1.json").read_text())
    lock = json.loads((ROOT / config["observed_asset_pins"]["path"]).read_text())
    metadata = json.loads((ROOT / lock["basis"]["path"]).read_text())
    for parent in metadata["replicates"]:
        pin = lock["parents"][parent["parent_id"]]
        assert pin["comparison_coverage_original_bp0"] == parent["coordinate_contract"]["coverage_original_bp0"]
        assert pin["comparison_retained_fixed100bp_indexes"] == parent["coordinate_contract"]["retained_fixed100bp_indexes"]
        assert pin["observed_files_sha256"] == parent["input_observed_sha256"]


def test_both_actual_modes_use_explicit_support_and_do_not_modify_native_topology():
    tree = ast.parse((ROOT / "scripts/infer_biological_paths_tsinfer_v1.py").read_text())
    funcs = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    pipeline_checks = [n for n in ast.walk(funcs["pipeline"]) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == "require_identity"]
    assert len(pipeline_checks) == 1
    support = next(k.value for k in pipeline_checks[0].keywords if k.arg == "support")
    assert ast.unparse(support) == "assets['identity_support_original_bp0']"
    load = funcs["load_assets"]
    returned = next(n.value for n in ast.walk(load) if isinstance(n, ast.Return))
    support_value = returned.values[next(i for i, key in enumerate(returned.keys)
                                         if isinstance(key, ast.Constant) and key.value == "identity_support_original_bp0")]
    assert ast.unparse(support_value) == "[left, right]"
    assignment = next(n for n in ast.walk(load) if isinstance(n, ast.Assign)
                      and any(ast.unparse(t) == "(left, right)" for t in n.targets))
    assert ast.unparse(assignment.value) == "pin['comparison_coverage_original_bp0']"
    qchecks = [n for n in ast.walk(funcs["qualification_summary"]) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == "require_identity"]
    assert len(qchecks) == 1 and any(k.arg == "support" for k in qchecks[0].keywords)
    # The main qualification branch now invokes that same support-aware helper.
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "qualification_summary"
               for n in ast.walk(funcs["main"]))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "connected_identity"
                   for n in ast.walk(funcs["main"]))

"""Independent handmade/helper regressions. No native inference or dating."""
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
spec = importlib.util.spec_from_file_location("independent_tsinfer_helpers", ROOT / "scripts/infer_biological_paths_tsinfer_v1.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def hand_tree(*, disconnected=False, single_tree=False):
    t = tskit.TableCollection(100)
    t.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    t.edges.metadata_schema = tskit.MetadataSchema.permissive_json()
    t.sites.metadata_schema = tskit.MetadataSchema.permissive_json()
    t.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
    t.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
    t.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
    pop = t.populations.add_row(metadata={"handmade": True})
    individuals = [t.individuals.add_row(metadata={"handmade": True}) for _ in range(2)]
    samples = [1, 3, 4, 6]
    for n, age in enumerate([200, 0, 150, 0, 10, 175, 10]):
        i = samples.index(n) if n in samples else None
        t.nodes.add_row(time=age, flags=tskit.NODE_IS_SAMPLE | 1024 if i is not None else 4096,
                        individual=individuals[i//2] if i is not None else tskit.NULL,
                        population=pop, metadata={"handmade": n})
    a = t.nodes.add_row(time=50, metadata={"handmade": "a"})
    b = t.nodes.add_row(time=60, metadata={"handmade": "b"})
    root = t.nodes.add_row(time=100, metadata={"handmade": "root"})
    segments = [(0, 100 if single_tree else 50, [(1, a), (3, a), (a, root), (4, root), (6, root)])]
    if not single_tree:
        segments.append((50, 100, [(4, b), (6, b), (b, root), (1, root), (3, root)]))
    for left, right, edges in segments:
        for child, parent in edges:
            if disconnected and child == 6:
                continue
            t.edges.add_row(left, right, parent, child, metadata={"handmade": True})
    for pos in [5, 10, 89, 90, 95]:
        site = t.sites.add_row(pos, "A", metadata={"handmade": pos})
        first = t.mutations.add_row(site, 1, "G", time=2, metadata={"handmade": True})
        t.mutations.add_row(site, 1, "T", parent=first, time=1, metadata={"handmade": True})
        t.mutations.add_row(site, 4, "G", time=11, metadata={"handmade": True})
    t.metadata_schema = tskit.MetadataSchema.permissive_json()
    t.metadata = {"handmade": True}
    t.provenances.add_row(record='{"handmade":true}')
    t.sort()
    return t.tree_sequence()


def hand_sites():
    # Deliberately use an allele order different from the tree's native encoding.
    return [{"site_id": i, "position": pos, "vcf_position_1based": pos + 1,
             "alleles": ["A", "T", "G"], "genotypes": [1, 0, 2, 0],
             "ancestral_allele": 0} for i, pos in enumerate([5, 10, 89, 90, 95])]


def test_recurrent_multiallelic_decoding_compares_strings_and_phase():
    ts = hand_tree()
    m.check_genotypes(ts, hand_sites())
    changed = hand_sites()
    changed[0]["genotypes"] = [0, 1, 2, 0]
    with pytest.raises(ValueError, match="phased"):
        m.check_genotypes(ts, changed)


def test_noncontiguous_sample_ids_dates_and_boundary_sanitation():
    native = hand_tree()
    clean = m.sanitize_clip(native, [0, 0, 10, 10], 10, 90)
    assert clean.samples().tolist() == [1, 3, 4, 6]
    assert clean.num_nodes == native.num_nodes
    assert np.array_equal(clean.nodes_time, native.nodes_time)
    assert np.all(clean.nodes_flags[[0, 2, 5]] == 0)
    assert np.all(clean.nodes_flags[[1, 3, 4, 6]] == tskit.NODE_IS_SAMPLE)
    assert clean.sites_position.tolist() == [0, 79]
    assert clean.num_mutations == 6
    assert clean.num_populations == clean.num_individuals == clean.num_migrations == clean.num_provenances == 0
    for table in [clean.tables.nodes, clean.tables.edges, clean.tables.sites, clean.tables.mutations]:
        assert len(table.metadata) == 0 and table.metadata_schema.schema is None
    m.check_genotypes(clean, [{**x, "position": x["position"]-10} for x in hand_sites() if 10 <= x["position"] < 90])
    assert m.require_identity(clean, 1)["internal_reuse_occurrences"] == 1


def test_disconnected_actual_sample_is_rejected_despite_shared_root_elsewhere():
    with pytest.raises(ValueError, match="one inferred ancestral root"):
        m.connected_identity(hand_tree(disconnected=True))


def test_single_marginal_tree_cannot_qualify_cross_tree_identity():
    result = m.connected_identity(hand_tree(single_tree=True))
    assert result["trees"] == 1 and result["internal_reuse_occurrences"] == 0
    with pytest.raises(ValueError, match="no ID repair"):
        m.require_identity(hand_tree(single_tree=True), 1)


def test_parent_sidecar_preserves_full_site_join_and_actual_sample_ids(tmp_path, monkeypatch):
    native = hand_tree()
    assets = {
        "rows": [{"sample_index": i, "observed_node_id": 100+i, "sample_id": "d"+str(i//2)} for i in range(4)],
        "ages": [0, 0, 10, 10], "sites": hand_sites(), "paths": {},
    }
    # The only two producer entry points that could load observations or run
    # scientific inference are replaced with handwritten objects.
    monkeypatch.setattr(m, "load_assets", lambda *args: assets)
    monkeypatch.setattr(m, "pipeline", lambda *args: native)
    assets["paths"]["sample_manifest.json"] = tmp_path / "handmade_roster.json"
    assets["paths"]["scoring_coordinates.npz"] = tmp_path / "handmade_grid_metadata.txt"
    for p in assets["paths"].values():
        p.write_text("handmade")
    pin = {"comparison_coverage_original_bp0": [10, 90],
           "observed_files_sha256": {name: m.sha(p) for name, p in assets["paths"].items()}}
    c = {"length_bp": 100, "minimum_connected_internal_reuse": 1}
    result = m.infer_parent(c, "bio_paths_pilot_train_000", tmp_path, pin, 0)
    assert [x["original_site_id"] for x in result["site_join"]] == list(range(5))
    assert [x["clean_site_id"] for x in result["site_join"]] == [None, 0, 1, None, None]
    assert [x["local_position_0based"] for x in result["site_join"]] == [None, 0, 79, None, None]
    assert result["coordinate_contract"]["retained_fixed100bp_indexes"] == [0]
    assert result["coordinate_contract"]["normalization_length_bp"] == 100
    assert [x["inferred_tree_node_id"] for x in result["sample_index_to_inferred_tree_node_id"]] == [1, 3, 4, 6]
    roster = json.loads((tmp_path / "bio_paths_pilot_train_000/observed_sample_manifest.json").read_text())
    assert [x["observed_node_id"] for x in roster] == [1, 3, 4, 6]
    assert [x["source_observed_node_id"] for x in roster] == [100, 101, 102, 103]


def test_explicit_staged_native_kwargs_exist_in_pinned_upstream_signatures():
    source = ast.parse((ROOT / "scripts/infer_biological_paths_tsinfer_v1.py").read_text())
    modules = {"tsinfer": ast.parse((AUDIT / "tsinfer_inference.py").read_text()),
               "tsdate": ast.parse((AUDIT / "tsdate_core.py").read_text())}
    util = ast.parse((AUDIT / "tsdate_util.py").read_text())
    modules["tsdate"].body.extend(util.body)
    signatures = {module: {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for module, tree in modules.items()}
    calls = []
    for node in ast.walk(source):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module, name = node.func.value.id, node.func.attr
            if module in signatures and name in signatures[module]:
                native = signatures[module][name]
                permitted = {a.arg for a in native.args.posonlyargs + native.args.args + native.args.kwonlyargs}
                explicit = {k.arg for k in node.keywords if k.arg is not None}
                assert explicit <= permitted or native.args.kwarg is not None
                calls.append((module, name))
    assert ("tsinfer", "match_samples") in calls and ("tsdate", "add_sampledata_times") in calls
    # Convenience infer must not receive the historical-only keyword.
    for node in ast.walk(source):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "tsinfer" and node.func.attr == "infer":
            assert "force_sample_times" not in {k.arg for k in node.keywords}


def test_default_cli_path_cannot_reach_actual_modes_without_admission():
    source = ast.parse((ROOT / "scripts/infer_biological_paths_tsinfer_v1.py").read_text())
    main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    branch = next(n for n in ast.walk(main) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                  and ast.unparse(n.test) == "args.mode == 'source-only'")
    assert isinstance(branch.body[-1], ast.Return)
    admission_line = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "admission")
    pipeline_line = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "pipeline")
    assert branch.end_lineno < admission_line < pipeline_line
    target_mkdir = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and isinstance(n.func.value, ast.Name) and n.func.value.id == "target" and n.func.attr == "mkdir")
    assert any(k.arg == "exist_ok" and isinstance(k.value, ast.Constant) and k.value.value is False for k in target_mkdir.keywords)


def test_native_style_empty_flanks_reproduce_full_length_gate_blocker():
    full = hand_tree()
    tables = full.dump_tables()
    e = tables.edges
    lo, hi = full.sites_position[0], full.sites_position[-1] + 1
    left, right = np.maximum(e.left, lo), np.minimum(e.right, hi)
    keep = right > left
    tables.edges.set_columns(left=left[keep], right=right[keep],
                            parent=e.parent[keep], child=e.child[keep])
    tables.sort()
    native_style = tables.tree_sequence()
    assert native_style.sequence_length == 100
    m.check_genotypes(native_style, hand_sites())
    m.strict_times(native_style, [0, 0, 10, 10])
    # These are the native expected unscored tails, not disconnected scored samples.
    assert list(native_style.first().interval) == [0, 5]
    with pytest.raises(ValueError, match="one inferred ancestral root"):
        m.require_identity(native_style, 1)
    qualified_support = m.sanitize_clip(native_style, [0, 0, 10, 10], 10, 90)
    assert m.require_identity(qualified_support, 1)["internal_reuse_occurrences"] == 1

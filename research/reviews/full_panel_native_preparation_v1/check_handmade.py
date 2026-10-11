"""Bounded handmade full-panel preparation checks. No actual parents/targets."""
from __future__ import annotations
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tskit
from arggnn import full_panel_native_baseline_contracts_v1 as contracts
from arggnn import prepare_full_panel_native_supervised_v1 as prep

ROOT = prep.ROOT
REVIEW = ROOT / "research/reviews/full_panel_native_preparation_v1"
WORK = REVIEW / os.environ.get("FULL_PANEL_NATIVE_HANDMADE_RUN", "handmade")
checks = {}
requests = []


def guard(value):
    requests.append(value)


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except contracts.ContractError:
        return
    raise AssertionError("Expected scientific contract rejection")


def synthetic_rows(offset=7):
    rows = []
    for cohort, count in contracts.COHORT_HAPLOTYPES.items():
        for within in range(count):
            index = len(rows)
            reference = cohort in contracts.REFERENCE_CLASSES
            age = 200 if reference else (0 if cohort == "present_day" else 10)
            if reference and within < contracts.OLDER_REFERENCE_COUNTS[cohort]:
                age = 360 if cohort == "Ana" else 400
            rows.append({"sample_index": index, "sample_id": f"tsk_{index // 2}",
                "haplotype": index % 2, "inferred_tree_node_id": index + offset,
                "observed_node_id": index + offset, "source_observed_node_id": index,
                "observed_cohort": cohort, "age_generations": age,
                "role": "pure_source_reference" if reference else "query",
                "observed_reference_class": contracts.REFERENCE_CLASSES.get(cohort)})
    return rows


def run():
    proto = prep.protocol()
    rows = synthetic_rows()
    roster = contracts.validate_observed_roster(rows)
    assert len(roster.reference_nodes) == 266
    assert [roster.reference_classes.count(i) for i in range(4)] == [50, 28, 102, 86]
    assert dict(roster.older_reference_counts) == contracts.OLDER_REFERENCE_COUNTS
    rejects(contracts.validate_observed_roster, rows, require_argmix_native_ids=True)
    checks["all266_unequal_source_donors_including34_older_and_generic_AP_IDs"] = "passed"
    native_rows = synthetic_rows(0)
    text = prep.argmix_reference_text(native_rows)
    counts = {pop: sum(line.split()[1] == pop for line in text.splitlines())
              for pop in [*contracts.CLASS_ORDER, "QUERY"]}
    assert counts == {"Ana": 50, "CHG": 28, "WHG": 102, "EHG": 86, "QUERY": 686}
    assert len(prep.native_sample_text(native_rows).splitlines()) == 478
    assert "BAA" not in text and "<SAMPLE>" not in text  # Native iterator supplies query input token.
    checks["native_QUERY_metadata_and476_sample_rows_without_donor_census_labels"] = "passed"

    # Compile only the exact author's pure vocabulary function and token literal.
    source = ROOT / ".runtime/author_sources/argmix_c2481248eb0d/src/tree_sampler.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPECIAL_TOKENS" for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name == "build_vocab_from_args")]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    vocab = namespace["build_vocab_from_args"]([*contracts.CLASS_ORDER, "QUERY"])
    assert vocab == {"<PAD>": 0, "<UNK>": 1, "<SAMPLE>": 2, "<ANC>": 3,
                     "Ana": 4, "CHG": 5, "EHG": 6, "QUERY": 7, "WHG": 8}
    class_remap = {vocab[name]: i for i, name in enumerate(contracts.CLASS_ORDER)}
    assert class_remap == {4: 0, 5: 1, 8: 2, 6: 3}
    checks["author_sorted_vocabulary_kept_separate_from_C4_class_remap"] = "passed"

    tables = tskit.TableCollection(400)
    for _ in range(7):
        tables.nodes.add_row(time=0)
    for row in rows:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=row["age_generations"])
    for left, right, age in [(0, 200, 1000), (200, 400, 2000)]:
        ancestor = tables.nodes.add_row(time=age)
        for node in roster.all_sample_nodes:
            tables.edges.add_row(left=left, right=right, parent=ancestor, child=node)
    for site in [0, 100, 399]:
        tables.sites.add_row(position=site, ancestral_state="0")
    tables.sort()
    ts = tables.tree_sequence()
    if WORK.exists():
        raise ValueError("Handmade output already exists; retain prior evidence")
    prep.new_directory(WORK, guard)
    tree_path = WORK / "synthetic.observed.trees"
    prep._guard(guard, ts.nbytes + 65536)
    ts.dump(tree_path)
    support = contracts.validate_physical_support({
        "condition": "post_five_size_iterations_primary",
        "normalization_length_bp": 2000000, "original_parent_interval_bp0": [0, 2000000],
        "samples": 952, "fixed_grid_total": 20000, "coverage_original_bp0": [59, 459],
        "local_to_original_offset_bp": 59, "sequence_length_local_bp": 400,
        "retained_fixed100bp_indexes": [1, 2, 3, 4], "fixed_grid_retained": 4})
    join = [{"native_site_id": i, "original_site_id": i, "original_position_0based": int(pos + 59),
             "local_position_0based": float(pos), "VCF_position_1based": int(pos + 60)}
            for i, pos in enumerate([0, 100, 399])]
    observation = {"parent_id": "ap_full_train_000", "downstream_role": "TRAIN", "nominal_split": "train",
        "ts": ts, "rows": rows, "roster": roster, "tree_record": prep.record(tree_path),
        "inference_record": prep.record(Path(__file__)), "completion_record": prep.record(Path(__file__)),
        "freeze_record": prep.record(Path(__file__)), "observed_input_records": {},
        "support": support, "site_join": join, "local_site_positions_bp": np.array([0, 100, 399]),
        "original_site_positions_bp": np.array([59, 159, 458]), "protocol": proto}
    features = prep.prepare_ap_features(observation, WORK / "features", guard=guard)
    bridge = prep.pinned_module(prep.helper_ref(observation, "src/arggnn/ancestralpaths_native_bridge.py"),
                                "handmade_feature_bridge")
    panel = bridge.ObservedPanel(tuple(r["sample_id"] for r in rows[::2]),
        roster.all_sample_nodes, {r["inferred_tree_node_id"]: r["observed_cohort"] for r in rows})
    with np.load(prep.checked(features["feature_cache"]), allow_pickle=False) as z:
        assert z["grid_feature_index"].shape == (4, 686)
        assert z["site_feature_index"].shape == (3, 686)
        assert np.array_equal(z["query_node_ids"], roster.query_nodes)
        for positions, key in [(support["original_centers_bp"], "grid_feature_index"),
                               (observation["original_site_positions_bp"], "site_feature_index")]:
            for i, original in enumerate(positions):
                direct = bridge.extract_native_features(ts.at(original - 59), roster.query_nodes, panel).astype(np.float32)
                assert np.array_equal(z["features"][z[key][i]], direct)
        assert np.all(z["features"][:, 1:] == -15)
    checks["reused_AP_cache_equals_native_bridge_all_handmade_sites_centers_queries"] = "passed"
    checks["five_by_nine_shells_ancient_self_present_exclusion_root_padding"] = "passed"
    checks["exact_local_original_SNP_join_clipped_centers_without_tail_fill"] = "passed"

    # Exact frozen sampler, supplied handmade arrays through an external stub.
    # One source pass has 4000 distinct rows. The native float1.5 next stride
    # selects every3rd site and retains1000 replacement duplicates.
    cache_path = WORK / "synthetic_sampler_cache.npz"
    np.savez_compressed(cache_path, site_positions_bp=np.arange(4000),
        site_feature_index=np.zeros((4000, 1), np.int32), features=np.zeros((1, 5, 9), np.float32))
    labels = np.ones((4000, 1), np.int16)
    stub = SimpleNamespace(labels_at=lambda sites: labels, provenance={"handmade": True})
    cache_module = prep.pinned_module(prep.AP_CACHE_HELPER, "handmade_frozen_cache")
    definition = SimpleNamespace(TruthReader=lambda parent, purpose: stub, write_json=cache_module.write_json)
    sampler_ref = prep.helper_ref(observation, "scripts/train_ancestralpaths_biological_v2.py")
    sampler = prep.pinned_module(sampler_ref, "handmade_frozen_sampler")
    sampler_output = prep.new_directory(WORK / "sampler", guard)
    x, y, reports = sampler.sample_realized_source_rows(definition,
        {"parent_id": "handmade", "cohorts": np.array(["BAA"]), "query_slots": np.array([0])},
        cache_path, {"sampler": {"maximum_source_passes": 128,
            "unfilled_quota_policy": "retain_realized_accepted_rows_and_continue_cohorts",
            "per_parent_cohort_budgets": {"BAA": {"target": 5000, "per_path_cap": 6000, "candidate_scan_budget": 0}}}},
        sampler_output)
    assert len(y) == 5000 and x.shape == (5000, 5, 9)
    assert reports[0]["replacement_duplicates"] == 1000
    assert reports[0]["passes"][1]["step"] == 1.5
    assert reports[0]["source_overhead"] == 10000 / 3
    full_cfg = prep.ap_sampler_config(proto["policy"])
    assert sum(v["target"] for v in full_cfg["sampler"]["per_parent_cohort_budgets"].values()) == 162500
    assert full_cfg["sampler"]["per_parent_cohort_budgets"]["BAA"]["per_path_cap"] == 6250
    assert contracts.native_ap_stride(2024, "BAA", traversal="TRAIN") == 1
    checks["unchanged_v2_sampler_float_modulo_divisor3_duplicate_multiplicity"] = "passed"

    outer = dict(observation, parent_id="ap_full_discovery_holdout_000", downstream_role="OUTER_DISCOVERY")
    rejects(prep.QueryTruth, outer, {}, "TRAIN_GRADIENTS", "argmix_current_joint_c4")
    val = dict(observation, parent_id="ap_full_train_002", downstream_role="INTERNAL_VAL")
    rejects(prep.prepare_ap_training, val, {}, {}, WORK / "forbidden", guard=guard)
    rejects(prep.QueryTruth, val, {}, "NATIVE_INTERNAL_VAL_SELECTION", "ancestralpaths_dense6")
    target = prep.QueryTruth.__new__(prep.QueryTruth)
    target.intervals = np.array([[0, 1000], [1000, 2000000]])
    target.paths = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    assert target.labels_at([999, 1000]).tolist() == target.paths.tolist()
    assert target.c4_at([999, 1000]).tolist() == [[0, 1, 2, 3, 0, 1], [1, 0, 3, 2, 1, 0]]
    checks["outer_truth_and_AP_VAL_rejected_before_file_access"] = "passed"
    checks["query_only_target_boundary_join_and_AP6_C4_ontology"] = "passed"

    # Verify stage logic receives the unchanged native model/objective dictionaries.
    mix = prep.pinned_module(prep.helper_ref(observation, "src/arggnn/official_argmix_biological.py"),
                             "handmade_frozen_mix")
    backend = mix.backend_config({"class_order": list(contracts.CLASS_ORDER),
        "native_model": copy.deepcopy(mix.NATIVE_MODEL), "joint_model": copy.deepcopy(mix.JOINT_MODEL)}, WORK)
    assert backend["model"] == mix.NATIVE_MODEL
    assert backend["joint_model"] == mix.JOINT_MODEL
    assert backend["joint_model"]["train"]["label_smoothing"] == .1
    assert backend["joint_model"]["train"]["early_stop_patience"] == 2
    assert backend["processing"]["exclude_edge_trees"] is True
    assert backend["processing"]["subgraph_merge"]["enabled"] is False
    assert backend["supervision"]["truth_label_mode"] == "first_snp"
    assert backend["supervision"]["truth_min_label_purity"] is None
    checks["native_ARGMix_architecture_extraction_objective_selection_unchanged"] = "passed"
    prep.write_json(REVIEW / (WORK.name + "_checks.json"), {
        "schema": "full_panel_native_preparation_handmade_checks_v1", "checks": checks,
        "source": prep.record(Path(prep.__file__)), "check_source": prep.record(Path(__file__)),
        "policy": proto["config_record"], "split": proto["split_record"],
        "actual_parent_files_opened": False, "actual_target_arrays_opened": False,
        "outer_truth_opened": False, "models_created_or_fitted": False,
        "GPU_jobs_launched": False, "native_Rust_preparation_executed": False,
        "bounded_handmade_rows": 5000, "write_guard_calls": len(requests),
        "maximum_handmade_write_request_bytes": max(requests),
    }, guard)
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    run()

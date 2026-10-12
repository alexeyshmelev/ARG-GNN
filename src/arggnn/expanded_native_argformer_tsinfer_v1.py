"""H038 native tsinfer/tsdate inference and released ARGformer; source-only default.

The frozen Relate ARGformer adapter remains a separately disclosed diagnostic.
Root owns CPU/storage/GPU admission. No inference, tensor, tree or array at import.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/expanded_native_argformer_tsinfer_v1.py"
FROZEN = "src/arggnn/expanded_native_argformer_v1.py"
FROZEN_SHA = "fcd2c80e216d2cf1cbb84e477747bf95186a7a496d723a773aee172be1ee4d64"
PIPELINE = "scripts/infer_biological_paths_tsinfer_v1.py"
PIPELINE_SHA = "b1f1c9a50b77daa9a416e7fed04d06a1bbbc247644d63103b11a2fa80a4fb2e0"
RECIPE = "configs/biological_paths_tsinfer_v1.json"
RECIPE_SHA = "85d8dc47f771826f3d23c42c55b847a2527f124540def1c91e5f51b200121e6c"
RUNTIME = "experiments/results/biological_tsinfer_runtime_install_v1/runtime_receipt.json"
RUNTIME_SHA = "04876226b6f90fc803741b4257f17b9595374c58e1681e4c04b7d6d9ba4e6e8f"
GENERATION = "configs/expanded_full_panel_v1.json"
REGISTRATION = "research/reviews/expanded_full_panel_v1_root/registration.json"
PRODUCER = "src/arggnn/prepare_expanded_full_panel_v1.py"
ORIGINAL = "src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py"
CONTRACTS = "src/arggnn/full_panel_native_baseline_contracts_v1.py"
CONDITION = "H038_native_ARGformer_current_tsinfer_tsdate_all952_actual_ages_v1"
OBSERVATION = "H038_current_tsinfer_tsdate_observed_heterochronous_full952_v1"
TS_PREFIX = ".runtime/envs/biological_tsinfer_v1"
PREP_PREFIX = ".runtime/conda/envs/argformer_native_v1"
PARENTS = tuple("ap_expanded_%s_%03d" % (split, i)
                for split, n in (("train", 32), ("val", 8)) for i in range(n))
FILES = ("genotypes.vcf", "sample_manifest.json", "sample_order.txt",
         "sample_ages.generations", "population_labels.txt", "site_coordinate_manifest.json",
         "uniform_recombination_map.txt", "scoring_coordinates.npz", "observation_manifest.json")
_NATIVE = None
VAL_PARENTS = PARENTS[32:]


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _path(value):
    p = Path(value)
    if p.is_absolute():
        p = p.relative_to(ROOT)
    _require(bool(p.parts) and ".." not in p.parts, "Dedicated project path required")
    for n in range(1, len(p.parts) + 1):
        _require(not ROOT.joinpath(*p.parts[:n]).is_symlink(), "Project symlink forbidden")
    out = ROOT / p
    out.resolve().relative_to(ROOT)
    return out


def _ref(value):
    p = _path(value)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=p.relative_to(ROOT).as_posix(), sha256=h.hexdigest())


def _checked(ref):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"},
             "Exact path/SHA reference required")
    _require(_ref(ref["path"]) == ref, "Bound artifact bytes changed: " + ref["path"])
    return _path(ref["path"])


def _read(ref):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
        and isinstance(ref["path"], str), "Exact nontruth JSON metadata reference required")
    _require(ref["path"].endswith(".json") and not any(
        word in part.lower() for part in Path(ref["path"]).parts for word in ("truth", "oracle")),
        "Nontruth JSON metadata required")
    return json.loads(_checked(ref).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _save(path, value, guard, *, new=True):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _guard(guard, len(data) + 4096)
    with path.open("xb" if new else "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return _ref(path)


def _guard(guard, planned=0):
    _require(Path.cwd().resolve() == ROOT and callable(guard) and guard(int(planned)) is not False,
             "Root admitted project write guard required")


def _frozen():
    _require(_ref(FROZEN)["sha256"] == FROZEN_SHA, "Frozen Relate diagnostic changed")
    return importlib.import_module("arggnn.expanded_native_argformer_v1")


def default_config_v1():
    cfg = copy.deepcopy(_frozen().default_config_v1())
    for path, sha in ((PIPELINE, PIPELINE_SHA), (RECIPE, RECIPE_SHA), (RUNTIME, RUNTIME_SHA)):
        _require(_ref(path)["sha256"] == sha, "Qualified native recipe/runtime pin changed")
    recipe = _read(_ref(RECIPE))
    keys = ("length_bp", "score_bin_bp", "mutation_rate", "num_threads", "recombination_rate",
            "mismatch_ratio", "path_compression", "post_process", "preprocess", "dating",
            "site_age", "proxy", "force_sample_times", "minimum_connected_internal_reuse")
    observation = {key: copy.deepcopy(recipe[key]) for key in keys}
    observation.update(condition=OBSERVATION, runtime_prefix=TS_PREFIX,
        runtime_ref=_ref(RUNTIME), runtime_versions=recipe["runtime_versions"],
        pipeline_ref=_ref(PIPELINE), recipe_ref=_ref(RECIPE),
        generation_config_ref=_ref(GENERATION), registration_ref=_ref(REGISTRATION),
        producer_ref=_ref(PRODUCER), original_producer_ref=_ref(ORIGINAL),
        contracts_ref=_ref(CONTRACTS), parent_order=list(PARENTS),
        all952_actual_dates=True, query_truth_read=False,
        support_policy="frozen_Relate_observations_only_support_and_common_SNP_join",
        REF_ancestral_rule="pinned_tskit_1.0.3_writer_ancestral_allele_first",
        no_input_variant_filtering=True)
    cfg.update(schema="expanded_native_argformer_tsinfer_v1_config", source_ref=_ref(SOURCE),
        condition=CONDITION, shared_task="H038_full952_2Mb_4sources_actual_ages_native_tsinfer_tsdate",
        observation=observation, frozen_Relate_diagnostic_ref=_ref(FROZEN),
        comparison_claim="end_to_end_inference_plus_method_vs_Relate_baselines",
        primary_fit_parent_ids=list(VAL_PARENTS), primary_fitting_scope="eight_independent_transductive_VAL_ARGs",
        TRAIN_ARGformer_role="registry_metadata_not_checkpoint_training_for_VAL",
        all40_CPU_inference_role="optional_future_protocol",
        source_version_claim="author_family_current_pinned_heterochronous_transfer",
        resource_planning=dict(measured=False, workers=1, CPU_threads=1,
            per_parent_output_cap_bytes=4000000000, total_preparation_cap_bytes=160000000000,
            CPU_RSS_cap_bytes=51539607552, per_stage_wall_seconds=14400,
            maximum_native_nodes=2000000, maximum_native_trees=20000,
            maximum_native_edges=20000000, maximum_native_sites=200000,
            maximum_full_path_occurrences=10000000, maximum_unique_full_paths=1000000,
            fit_output_cap_policy="root_measured_atomic_per_ARG_scopes_after_full952_canary",
            unmeasured_fit_storage_reservation_authorized=False,
            project_warning_bytes=950000000000, project_limit_bytes=1000000000000,
            fit_memory_requires_root_corpus_qualification=True,
            cap_exceeded_policy="stop_preserve_disclose_no_semantic_reduction"))
    return cfg


def _registry(config):
    _require(config == default_config_v1(), "Frozen native inference/method config required")
    obs = config["observation"]
    generation = _read(obs["generation_config_ref"])
    registration = _read(obs["registration_ref"])
    rows = registration["parent_registry"]
    _require(rows == generation["parent_registry"]
        and registration["config_ref"] == obs["generation_config_ref"]
        and registration["source_ref"] == obs["producer_ref"]
        and registration["registry_sha256"] == _digest(rows)
        and registration["outer_evaluation_authorized"] is False,
        "Exact H038 registration required")
    roles = {r["parent_id"]: r["split"] for r in rows}
    _require([k for k, v in roles.items() if v in ("train", "val")] == list(PARENTS),
             "Exactly original 32 TRAIN then 8 VAL; no outer")
    return generation, rows, roles


def runtime_report_v1(stage):
    _require(stage in ("observe", "prepare"), "CPU stage required")
    packages = ("numpy", "tskit", "tsinfer", "tsdate") if stage == "observe" else ("numpy", "tskit", "torch")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(schema="expanded_argformer_tsinfer_CPU_runtime_v1", stage=stage,
        executable=sys.executable, prefix=str(Path(sys.prefix).resolve()),
        python=list(sys.version_info[:3]), packages=versions, source_ref=_ref(SOURCE),
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"), GPU_used=False)


def _CPU_runtime(stage, config):
    report = runtime_report_v1(stage)
    expected = config["observation"]["runtime_versions"] if stage == "observe" else {
        "numpy": "1.26.4", "tskit": "0.6.4", "torch": "2.4.0"}
    prefix = TS_PREFIX if stage == "observe" else PREP_PREFIX
    py = [3, 11, 17] if stage == "observe" else [3, 11, 9]
    _require(report["prefix"] == str(_path(prefix)) and report["python"] == py
        and {k: (v.split("+")[0] if v else v) for k, v in report["packages"].items()} == expected
        and report["CUDA_VISIBLE_DEVICES"] == "", "Qualified CPU runtime/prefix required")
    if stage == "observe":
        receipt = _read(config["observation"]["runtime_ref"])
        _require(receipt["prefix"] == prefix and receipt["versions"] == expected
            and _ref(Path(sys.executable).resolve())["sha256"] == receipt["python_executable_sha256"],
            "Qualified native inference executable binding required")
    return report


def _limits(config, resource_caps=None):
    limits = dict(config["resource_planning"])
    if resource_caps is not None:
        allowed = {"per_parent_output_cap_bytes", "CPU_RSS_cap_bytes", "per_stage_wall_seconds",
            "maximum_native_nodes", "maximum_native_trees", "maximum_native_edges", "maximum_native_sites",
            "maximum_full_path_occurrences", "maximum_unique_full_paths"}
        _require(isinstance(resource_caps, dict) and set(resource_caps) == allowed
            and all(type(value) is int and 0 < value <= limits[key]
                    for key, value in resource_caps.items()),
            "Positive root per-parent resource bounds within provisional ceilings required")
        limits.update(resource_caps)
    return limits


def _recipe_namespace(config, guard, resource_caps=None):
    """Only pinned scientific functions; old reduced-parent/storage admission is excluded."""
    ref = config["observation"]["pipeline_ref"]
    source = _checked(ref).read_text()
    names = {"parse_vcf", "grid_indexes", "check_samples", "check_genotypes", "strict_times",
             "connected_identity", "require_identity", "sanitize_clip", "pipeline"}
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    _require({n.name for n in selected} == names, "Exact qualified recipe function closure required")
    g = dict(ROOT=ROOT, math=math, json=json, os=os, Path=Path, sys=sys, time=time,
             confined=_path, ref=_ref)
    def boundary(c, initial):
        _guard(guard)
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        _require(peak <= _limits(config, resource_caps)["CPU_RSS_cap_bytes"], "Native CPU RSS cap exceeded")
        return dict(root_storage_guard=True, peak_cpu_rss_bytes=peak)
    g["storage_check"] = boundary
    g["write_json"] = lambda p, v, new=False: _save(p, v, guard, new=new)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(_checked(ref)), "exec"), g)
    return g


def _completed(parent_id, completion_ref, config):
    generation, registry, roles = _registry(config)
    _require(roles.get(parent_id) in ("train", "val"), "Registered inner parent required")
    base = generation["output_root"] + "/" + parent_id
    _require(completion_ref["path"] == base + "/expanded_parent_completed.json", "Exact completion path required")
    completed = _read(completion_ref)
    obs = config["observation"]
    _require(completed.get("schema") == "expanded_full_panel_v1_parent_completion"
        and completed["parent_id"] == parent_id and completed["split"] == roles[parent_id]
        and completed["generation_completed"] is True and completed["Relate_completed"] is True
        and completed["source_ref"] == obs["producer_ref"]
        and completed["original_producer_ref"] == obs["original_producer_ref"]
        and completed["config_content_sha256"] == _digest(generation)
        and completed["parent_registry_sha256"] == _digest(registry)
        and completed["outer_truth_evaluated"] is False,
        "Frozen completed H038 observations required")
    _require(completed["inference_ref"]["path"] == base + "/relate/inference.json", "Frozen support metadata path required")
    inference = _read(completed["inference_ref"])
    _require(inference["parent_id"] == parent_id and inference["split"] == roles[parent_id]
        and inference["status"] == "post_five_iterations_inferred"
        and inference["truth_or_oracle_read"] is False and inference["outer_command_returncodes"] == [0] * 5
        and set(inference["input_observed_sha256"]) == set(FILES), "Frozen observation whitelist/support lineage required")
    contracts = importlib.import_module("arggnn.full_panel_native_baseline_contracts_v1")
    post = inference["post_five_iterations"]
    support = contracts.validate_physical_support(post)
    contracts.validate_site_coordinate_join(post["site_join"],
        coverage_original_bp0=support["coverage_original_bp0"], declared_site_count=post["sites"])
    return generation, next(r for r in registry if r["parent_id"] == parent_id), completed, inference, support


def _validate_tree_shape(ts, config, resource_caps=None):
    limits = _limits(config, resource_caps)
    for actual, key in ((ts.num_nodes, "maximum_native_nodes"), (ts.num_trees, "maximum_native_trees"),
                        (ts.num_edges, "maximum_native_edges"), (ts.num_sites, "maximum_native_sites")):
        _require(actual <= limits[key], "Native shape cap exceeded: " + key)
    _require(ts.num_trees * 952 <= limits["maximum_full_path_occurrences"],
             "Full-path occurrence cap exceeded; retain native trees or stop")


def _validate_corpus_shape(corpus, config, resource_caps=None):
    limits = _limits(config, resource_caps)
    _require(len(corpus.paths) <= limits["maximum_unique_full_paths"]
        and len(corpus.occurrences) <= limits["maximum_full_path_occurrences"],
        "Complete native path corpus exceeds root planning caps; no subsampling")


def observe_parent_v1(parent_id, completion_ref, output_directory, *, config, guard, resource_caps=None):
    """All original VCF sites/date slots, native shared IDs; metadata-only Relate support."""
    _CPU_runtime("observe", config)
    generation, parent, completed, original_inference, support = _completed(parent_id, completion_ref, config)
    producer = importlib.import_module("arggnn.prepare_ancestralpaths_full_panel_discovery_v1")
    assets = producer.observed_assets(generation, parent, _path(generation["output_root"] + "/" + parent_id))
    _require(assets["source_sha256"] == original_inference["input_observed_sha256"], "Identical H038 VCF/roster/site/grid bytes required")
    native = _recipe_namespace(config, guard, resource_caps)
    coords = json.loads(assets["paths"]["site_coordinate_manifest.json"].read_text())
    _require(coords["build"] == "synthetic" and coords["contig_id"] == "22" and coords["length_bp"] == 2000000,
             "Original VCF physical coordinate contract required")
    with assets["paths"]["genotypes.vcf"].open() as f:
        sites = native["parse_vcf"](f, assets["names"], coords["rows"], 2000000,
                                    expected_writer_source="tskit 1.0.3")
    # The full-panel reader returns an ndarray; the qualified recipe's JSON
    # provenance expects the identical primitive ordered age list, not new dates.
    assets.update(sites=sites, ages=assets["ages"].tolist(),
                  identity_support_original_bp0=list(support["coverage_original_bp0"]))
    left, right = support["coverage_original_bp0"]
    _require(sites[0]["position"] <= left < right <= sites[-1]["position"] + 1,
             "Frozen common comparison support exceeds original observed site support")
    output = _path(output_directory); _guard(guard, 32768); output.mkdir(parents=True, exist_ok=False)
    c = config["observation"]
    dated = native["pipeline"](c, assets, output, None)
    _validate_tree_shape(dated, config, resource_caps)
    clean = native["sanitize_clip"](dated, assets["ages"], left, right)
    clean_sites = [{**s, "position": s["position"] - left} for s in sites if left <= s["position"] < right]
    native["check_genotypes"](clean, clean_sites)
    identity = native["require_identity"](clean, c["minimum_connected_internal_reuse"])
    _guard(guard, 32768); clean.dump(output / "observed.trees")
    rows = []
    for source, node in zip(assets["rows"], clean.samples().tolist()):
        joined = dict(source)
        joined["source_observed_node_id"] = joined.pop("observed_node_id")
        joined.update(observed_node_id=node, inferred_tree_node_id=node, genotype_slot=source["sample_index"])
        rows.append(joined)
    contracts = importlib.import_module("arggnn.full_panel_native_baseline_contracts_v1")
    contracts.validate_observed_roster(rows)
    native["check_samples"](clean, assets["ages"], diploid=False)
    position_to_site = {s.position + left: s.id for s in clean.sites()}
    common_join = [{**j, "tsinfer_site_id": position_to_site[j["original_position_0based"]]}
                   for j in original_inference["post_five_iterations"]["site_join"]]
    full_join = [dict(original_site_id=s["site_id"], original_position_0based=s["position"],
        VCF_position_1based=s["vcf_position_1based"], tsinfer_site_id=position_to_site.get(s["position"]),
        local_position_0based=s["position"] - left if left <= s["position"] < right else None,
        in_scored_support=left <= s["position"] < right) for s in sites]
    roster_ref = _save(output / "observed_roster.json", rows, guard)
    post = original_inference["post_five_iterations"]
    keys = ("condition", "normalization_length_bp", "original_parent_interval_bp0", "samples",
            "fixed_grid_total", "coverage_original_bp0", "local_to_original_offset_bp",
            "sequence_length_local_bp", "retained_fixed100bp_indexes", "fixed_grid_retained")
    support_metadata = {k: post[k] for k in keys}
    for name, sha in assets["source_sha256"].items():
        _checked(dict(path=generation["output_root"] + "/" + parent_id + "/observed/" + name, sha256=sha))
    result = dict(schema="expanded_native_argformer_tsinfer_observation_v1", condition=OBSERVATION,
        parent_id=parent_id, split=parent["split"], source_ref=config["source_ref"], config_digest=_digest(config),
        completion_ref=completion_ref, frozen_Relate_support_metadata_ref=completed["inference_ref"],
        input_observed_sha256=assets["source_sha256"], inferred_tree_ref=_ref(output / "observed.trees"),
        roster_ref=roster_ref, sample_index_to_inferred_tree_node_id=[
            dict(sample_index=r["sample_index"], inferred_tree_node_id=r["inferred_tree_node_id"]) for r in rows],
        comparison_support_metadata=support_metadata, common_site_join=common_join,
        full_native_site_join=full_join, connected_identity=identity,
        native_nodes=clean.num_nodes, native_trees=clean.num_trees, native_sites_in_support=clean.num_sites,
        native_full_original_sites=len(sites), common_comparison_sites=len(common_join),
        all952_actual_dates=True, all_original_genotypes_and_sites_verified=True,
        sample_dates_encoded_as_tokens=False, node_ids_and_times_preserved_during_sanitation=True,
        internal_ids_synthesized=False, input_variants_filtered=False,
        truth_or_oracle_read=False, source_labels_passed_to_inference=False,
        outer_included=False, paper_original_numerical_reproduction=False,
        inference_recipe_ref=c["recipe_ref"], inference_pipeline_ref=c["pipeline_ref"],
        runtime_report=runtime_report_v1("observe"))
    result["manifest_ref"] = _save(output / "observation_manifest.json", result, guard)
    return result


def read_observed_parent_v1(parent_id, observation_ref, *, config):
    full = default_config_v1()
    _require(config == full["observation"], "Exact native observation config required")
    _, _, roles = _registry(full)
    value = _read(observation_ref)
    _require(value["schema"] == "expanded_native_argformer_tsinfer_observation_v1"
        and value["condition"] == OBSERVATION and value["parent_id"] == parent_id
        and value["split"] == roles[parent_id] and roles[parent_id] in ("train", "val")
        and value["source_ref"] == full["source_ref"] and value["config_digest"] == _digest(full)
        and value["truth_or_oracle_read"] is False and value["all952_actual_dates"] is True
        and value["all_original_genotypes_and_sites_verified"] is True
        and value["internal_ids_synthesized"] is False and value["input_variants_filtered"] is False,
        "Exact separately prepared native observation required")
    generation, parent, completed, frozen, support = _completed(parent_id, value["completion_ref"], full)
    _require(value["frozen_Relate_support_metadata_ref"] == completed["inference_ref"]
        and value["input_observed_sha256"] == frozen["input_observed_sha256"], "Original observation/support pins differ")
    original_post = frozen["post_five_iterations"]
    for name, sha in value["input_observed_sha256"].items():
        _checked(dict(path=generation["output_root"] + "/" + parent_id + "/observed/" + name, sha256=sha))
    _require(value["comparison_support_metadata"] == {k: original_post[k] for k in value["comparison_support_metadata"]}
        and [{k: v for k, v in j.items() if k != "tsinfer_site_id"} for j in value["common_site_join"]] == original_post["site_join"],
        "Exact frozen support and common comparison SNP join required")
    directory = _checked(observation_ref).parent
    _require(_checked(value["inferred_tree_ref"]) == directory / "observed.trees"
        and _checked(value["roster_ref"]) == directory / "observed_roster.json", "Canonical native observation paths required")
    rows = _read(value["roster_ref"])
    contracts = importlib.import_module("arggnn.full_panel_native_baseline_contracts_v1")
    roster = contracts.validate_observed_roster(rows)
    original = _read(dict(path=generation["output_root"] + "/" + parent_id + "/observed/sample_manifest.json",
                          sha256=value["input_observed_sha256"]["sample_manifest.json"]))
    for row, source in zip(rows, original):
        _require(all(row[k] == source[k] for k in ("sample_index", "sample_id", "haplotype", "observed_cohort",
            "age_generations", "role", "observed_reference_class"))
            and row["source_observed_node_id"] == source["observed_node_id"]
            and row["genotype_slot"] == source["sample_index"], "Actual date/genotype slot/source-node mapping differs")
    _require(value["sample_index_to_inferred_tree_node_id"] == [
        dict(sample_index=r["sample_index"], inferred_tree_node_id=r["inferred_tree_node_id"]) for r in rows],
        "Actual flagged native sample ID mapping differs")
    import numpy as np
    import tskit
    ts = tskit.load(_checked(value["inferred_tree_ref"]))
    _validate_tree_shape(ts, full)
    _require(ts.samples().tolist() == [r["inferred_tree_node_id"] for r in rows]
        and np.array_equal(ts.tables.nodes.time[ts.samples()], [r["age_generations"] for r in rows])
        and ts.sequence_length == support["coverage_original_bp0"][1] - support["coverage_original_bp0"][0]
        and ts.num_nodes == value["native_nodes"] and ts.num_trees == value["native_trees"]
        and ts.num_sites == value["native_sites_in_support"], "Native sample times/order/shape changed")
    tables = ts.tables
    _require(not tables.metadata and ts.num_populations == ts.num_individuals == ts.num_migrations == ts.num_provenances == 0
        and np.isin(tables.nodes.flags, [0, 1]).all()
        and np.all(tables.nodes.population == -1) and np.all(tables.nodes.individual == -1)
        and all(len(t.metadata) == 0 for t in (tables.nodes, tables.edges, tables.sites, tables.mutations)),
        "Sanitized observed tree cannot contain annotations")
    for joined in value["common_site_join"]:
        _require(ts.site(joined["tsinfer_site_id"]).position == joined["local_position_0based"],
                 "Common SNP to actual tsinfer site ID join changed")
    return dict(parent_id=parent_id, split=parent["split"], ts=ts, rows=rows, roster=roster,
        support=support, site_join=value["common_site_join"], tree_ref=value["inferred_tree_ref"],
        completion_ref=observation_ref, inference_ref=observation_ref, targets_read=False)


def _native_namespace(resource_caps=None):
    """Fresh globals, pinned release functions; only schemas/readers are adapted."""
    global _NATIVE
    if resource_caps is None and _NATIVE is not None:
        return _NATIVE
    frozen = _frozen()
    source = _checked(_ref(FROZEN)).read_text()
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.If) and
        isinstance(n.test, ast.Compare) and "__name__" in ast.unparse(n.test))]
    class Bindings(ast.NodeTransformer):
        def visit_Constant(self, node):
            mapping = {s: s.replace("expanded_native_argformer_", "expanded_native_argformer_tsinfer_")
                for s in ("expanded_native_argformer_prepared_v1", "expanded_native_argformer_fit_v1",
                          "expanded_native_argformer_prediction_seal_v1")}
            if isinstance(node.value, str) and node.value in mapping:
                return ast.copy_location(ast.Constant(mapping[node.value]), node)
            return node
        def visit_FunctionDef(self, node):
            if node.name == "prepare_parent_v1":
                body = []
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "reader" for t in stmt.targets):
                        continue
                    if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "observed" for t in stmt.targets):
                        stmt.value = ast.parse("_read_native_observation(parent_id, completion_ref, config=reader_cfg)", mode="eval").body
                    body.append(stmt)
                    if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "observed" for t in stmt.targets):
                        body.append(ast.parse("_validate_tree_shape(observed['ts'], config)").body[0])
                    if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "corpus" for t in stmt.targets):
                        body.append(ast.parse("_validate_corpus_shape(corpus, config)").body[0])
                node.body = body
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(Bindings().visit(tree))
    g = dict(__name__="arggnn._expanded_tsinfer_release_isolated", __package__="arggnn")
    exec(compile(tree, FROZEN, "exec"), g)
    g["SOURCE"] = SOURCE
    g["default_config_v1"] = default_config_v1
    base_policy = g["_policy"]
    def policy(config):
        _, roles = base_policy(config)
        _registry(config)
        return config["observation"], roles
    g["_policy"] = policy
    g["_read_native_observation"] = read_observed_parent_v1
    g["_validate_tree_shape"] = lambda ts, config: _validate_tree_shape(ts, config, resource_caps)
    g["_validate_corpus_shape"] = lambda corpus, config: _validate_corpus_shape(corpus, config, resource_caps)
    if resource_caps is None:
        _NATIVE = g
    return g


def prepare_parent_v1(parent_id, observation_ref, output_directory, *, config, guard, resource_caps=None):
    _CPU_runtime("prepare", config)
    return _native_namespace(resource_caps)["prepare_parent_v1"](parent_id, observation_ref, output_directory, config=config, guard=guard)


def fit_parent_v1(prepared_ref, output_directory, *, config, guard, gpu_authorizer):
    value = _read(prepared_ref)
    _require(value["parent_id"] in VAL_PARENTS and value["split"] == "val",
             "Primary native benchmark fits exactly independent VAL ARGs; TRAIN supplies no checkpoint")
    return _native_namespace()["fit_parent_v1"](prepared_ref, output_directory,
        config=config, guard=guard, gpu_authorizer=gpu_authorizer)


def infer_parent_v1(fit_ref, prepared_ref, output_directory, *, config, guard, gpu_authorizer):
    value = _read(prepared_ref)
    _require(value["parent_id"] in VAL_PARENTS and value["split"] == "val",
             "Primary native prediction scope is the eight independent VAL ARGs")
    return _native_namespace()["infer_parent_v1"](fit_ref, prepared_ref, output_directory,
        config=config, guard=guard, gpu_authorizer=gpu_authorizer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-report", choices=("observe", "prepare"))
    args = parser.parse_args(argv)
    print(json.dumps(runtime_report_v1(args.runtime_report) if args.runtime_report else default_config_v1(),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

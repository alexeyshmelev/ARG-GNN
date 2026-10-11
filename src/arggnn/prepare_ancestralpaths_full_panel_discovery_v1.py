"""Full 476-diploid AncestralPaths discovery preparation; no work at import.

Default CLI is read-only. Execution handles exactly one frozen parent and needs
root-owned input/storage admission. Model fitting and GPU use are excluded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from fractions import Fraction
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
KIND = "ancestralpaths_full_panel_discovery_v1"
CONFIG = "configs/ancestralpaths_full_panel_discovery_v1.json"
SOURCE = "src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py"
AUTHOR = ".runtime/author_sources/ancestralpaths_77db3cfae100"
COHORTS = ("present_day", "Bronze_Age", "BAA", "Neo", "Yam", "WHG", "EHG", "Ana", "CHG")
COUNTS = (91, 81, 7, 151, 13, 51, 43, 25, 14)
CLASSES = {"Ana": 0, "CHG": 1, "WHG": 2, "EHG": 3}
OBSERVED_FILES = ("genotypes.vcf", "sample_manifest.json", "sample_order.txt",
                  "sample_ages.generations", "population_labels.txt",
                  "site_coordinate_manifest.json", "uniform_recombination_map.txt",
                  "scoring_coordinates.npz", "observation_manifest.json")


def confined(value):
    path = Path(value)
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    if ROOT not in path.parents:
        raise ValueError("All paths must be strictly inside the project")
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def ref(path):
    return {"path": path, "sha256": sha(confined(path))}


def roster():
    """Author order and original within-cohort age order; GBR input rows ignored."""
    ages = {}
    source = confined(AUTHOR + "/example/mesoneo_ages_pop.txt")
    if sha(source) != "3cadb16abbc0a1913eadb183d324eadc26c035d61e5116cd58675f52a942b7de":
        raise ValueError("Author age input changed")
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        cohort, age = line.split()
        ages.setdefault(cohort, []).append((int(age), line_number))
    rows = []
    diploid = 0
    for cohort, count in zip(COHORTS, COUNTS):
        dates = [(0, None)] * 91 if cohort == "present_day" else ages[cohort]
        if len(dates) != count:
            raise ValueError("Full author cohort count changed: " + cohort)
        population = "present_bronze" if cohort in ("present_day", "Bronze_Age") else cohort
        for age, line_number in dates:
            role = "pure_source_reference" if cohort in CLASSES else "query"
            for haplotype in (0, 1):
                rows.append({"sample_index": len(rows), "diploid_index": diploid,
                             "sample_id": f"tsk_{diploid}", "haplotype": haplotype,
                             "observed_cohort": cohort, "sampling_population": population,
                             "age_generations": age, "role": role,
                             "observed_reference_class": CLASSES.get(cohort),
                             "author_age_line_1based": line_number})
            diploid += 1
    if len(rows) != 952 or sum(r["role"] == "query" for r in rows) != 686:
        raise ValueError("Native full-roster arithmetic differs")
    return rows


def parents():
    result = []
    domain = "arggnn|ancestralpaths_full_panel_discovery_v1|root=101"
    for split, count in (("train", 3), ("discovery_holdout", 2)):
        for replicate in range(count):
            seeds, domains = {}, {}
            for purpose in ("ancestry", "mutations", "relate_initial", "relate_size_base"):
                text = f"{domain}|split={split}|parent={replicate:03d}|purpose={purpose}"
                raw = hashlib.sha256(text.encode("ascii")).digest()
                modulus = 2**31 - 100 if purpose.startswith("relate") else 2**32 - 1
                seeds[purpose] = int.from_bytes(raw[:8], "big") % modulus + 1
                domains[purpose] = raw.hex()
            result.append({"parent_id": f"ap_full_{split}_{replicate:03d}", "split": split,
                           "replicate": replicate, "seeds": seeds, "seed_domain_sha256": domains,
                           "relate_size_iteration_seeds": [seeds["relate_size_base"] + i for i in range(1, 6)],
                           "relate_final_reestimate_seed": seeds["relate_size_base"] + 5})
    values = [value for parent in result for value in parent["seeds"].values()]
    expanded = values + [value for parent in result for value in parent["relate_size_iteration_seeds"]]
    if len(set(expanded)) != len(expanded):
        raise ValueError("Frozen purpose/iteration seed collision")
    return result



def sampler_geometry(site_count=None, split=None):
    """Exact author stride geometry, using post-size observations only."""
    scan = dict(zip(COHORTS[:5], (100000, 100000, 25000, 50000, 25000)))
    rows = []
    for cohort, diploids in zip(COHORTS[:5], COUNTS[:5]):
        haplotypes = 2 * diploids
        train = Fraction(scan[cohort] * 3 + 10000, 3 * haplotypes)
        holdout = Fraction(20000, haplotypes)
        item = {"cohort": cohort, "haplotypes": haplotypes,
                "candidate_scan_budget_per_parent": scan[cohort],
                "TRAIN_stride_denominator_exact": str(train),
                "HOLDOUT_stride_denominator_exact": str(holdout),
                "minimum_TRAIN_inferred_sites": math.ceil(train),
                "minimum_HOLDOUT_inferred_sites": math.ceil(holdout)}
        if site_count is not None:
            denominator = train if split == "train" else holdout
            item["actual_initial_stride"] = math.floor(Fraction(int(site_count), 1) / denominator)
        rows.append(item)
    result = {"source_num_seq": 3,
              "TRAIN_formula": "floor(S / ((candidate_scan_budget + 10000/3) / cohort_haplotypes))",
              "HOLDOUT_formula": "floor(S / (20000 / cohort_haplotypes))",
              "minimum_all_cohort_TRAIN_inferred_sites": max(r["minimum_TRAIN_inferred_sites"] for r in rows),
              "minimum_all_cohort_HOLDOUT_inferred_sites": max(r["minimum_HOLDOUT_inferred_sites"] for r in rows),
              "rows": rows, "truth_or_labels_read": False}
    if site_count is not None:
        result.update(post_size_inferred_site_count=int(site_count), split=split,
                      all_initial_strides_positive=all(r["actual_initial_stride"] >= 1 for r in rows))
    return result


def dependencies():
    return [ref(path) for path in (
        "src/arggnn/biological_paths_simulation.py",
        "scripts/infer_biological_paths_relate_v1.py",
        "configs/biological_paths_execution_pilot_v1.json",
        "configs/biological_paths_relate_v1.json",
        "src/arggnn/project_storage_full_reservation_v1.py")]


def helpers():
    from arggnn import biological_paths_simulation as simulation
    name = "_ap_full_pinned_relate_helpers_v1"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, confined("scripts/infer_biological_paths_relate_v1.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return simulation, sys.modules[name]


def default_config():
    import msprime
    import numpy as np
    import tskit
    simulation, _ = helpers()
    sample_rows = roster()
    native = json.loads(confined("configs/biological_paths_relate_v1.json").read_text())
    source_files = ("simulate.py", "relate.sh", "training.py", "testing_bygroup.py",
                    "paint.py", "genetic_map_simulated.txt", "example/mesoneo_ages_pop.txt")
    return {
        "kind": KIND, "status": "source_prepared_no_generation",
        "fidelity": "released_author_workflow_with_declared_split_runtime_and_observation_adapters",
        "source": {"commit": simulation.AUTHOR_COMMIT, "root": AUTHOR,
                   "files": [ref(AUTHOR + "/" + name) for name in source_files],
                   "code_license": "no_named_license_in_cited_archive"},
        "dependencies": dependencies(),
        "runtime": {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "msprime": msprime.__version__, "tskit": tskit.__version__, "numpy": np.__version__},
        "output_root": "data/ancestralpaths_full_panel_discovery_v1",
        "output_cap_bytes": 4000000000, "native_step_headroom_bytes": 1000000000,
        "maximum_parents_per_invocation": 1, "default_parent_id": "ap_full_train_000",
        "root_seed": 101, "splits": {"train": 3, "discovery_holdout": 2, "val": 0, "test": 0},
        "parent_registry": parents(),
        "seed_scheme": "SHA256 ASCII arggnn|ancestralpaths_full_panel_discovery_v1|root=101|split={split}|parent={000..}|purpose={purpose};first8bytes big%(2**32-1)+1 for simulation;%(2**31-100)+1 for Relate;size uses base+1..5 and final reestimate reuses base+5",
        "roster": {"cohort_order": list(COHORTS), "diploid_counts": dict(zip(COHORTS, COUNTS)),
                   "diploids": 476, "haplotypes": 952, "query_haplotypes": 686,
                   "pure_reference_haplotypes": 266, "sample_template_sha256": digest(sample_rows),
                   "within_cohort_age_order": "original_age_file_order",
                   "present_day": "hardcoded91_diploids_age0_author_GBR_rows_ignored",
                   "sample_identity": ["parent_id", "sample_id", "haplotype"]},
        "demography": simulation.build_demography().asdict(),
        "simulation": {"length_bp": 2000000, "recombination_rate": 1e-8, "mutation_rate": 1.25e-8,
                       "ancestry_model": "hudson", "mutation_model": "JC69",
                       "discrete_genome": True, "record_migrations": False, "census_time": 350,
                       "mutation_on_census_stripped_genealogy": True, "score_bin_bp": 100,
                       "cleaned_oracle_exported": False},
        "relate": {"native_tool_release": "1.2.4",
                   "installation": ref(".runtime/tools/relate/installation.json"),
                   "source_coordinate_reference": native["source_coordinate_reference"],
                   "tool_sha256": native["tool_sha256"], "initial_effective_haploid_size": 30000,
                   "mutation_rate": 1.25e-8, "initial_threads": 8, "size_threads": 8,
                   "chunk_memory_gb": 2, "population_size_iterations": 5,
                   "size_tree_mutation_threshold": 0.5, "years_per_generation": 28,
                   "population_size_plotting": False, "mutation_rate_normalization": False,
                   "final_branch_length_reestimate": True, "timeout_seconds_per_step": 7200,
                   "primary_condition": "post_five_size_iterations_primary"},
        "observation_contract": {"Relate_input_whitelist": list(OBSERVED_FILES),
                                 "truth_or_oracle_reads_in_Relate": False,
                                 "ancestry_labels_in_observed_files": False,
                                 "native_sample_ages_preserved_in_generations": True,
                                 "reference_classes": CLASSES, "query_reference_class": None,
                                 "VCF_position": "integer_zero_based_mutation_position_plus1",
                                 "contig": "22", "build": "synthetic_no_empirical_GRCh37_claim",
                                 "native_inferred_sample_IDs": "actual_sample_flags_in_verified_haps_order",
                                 "truth_breakpoints_define_model_windows": False,
                                 "heldout_truth_for_model_selection": False,
                                 "feature_extraction_and_model_fitting_in_this_producer": False},
        "scientific_endpoints": {
            "native_AncestralPaths": {"ontology": "six_census350_paths", "sampling": "native_SNP_focal_rows",
                                     "metric": "categorical_sample_accuracy", "confusion": "predicted_rows_true_columns"},
            "common_comparison": {"ontology": ["Ana", "CHG", "WHG", "EHG"],
                                  "label": "explicitly_derived_C4", "native_probability_collapse": [[1, 5], [2, 6], [3], [4]],
                                  "metric": "equal_parent_query_bp_accuracy_and_NLL_on_matched_observed_support",
                                  "queries": "all686_admixed_haplotypes_BAA_included",
                                  "existing_project_aliases": {"ANA": "Ana"},
                                  "coordinates": "fixed100bp_centers_and_exact_clipped_bin_intersections",
                                  "primary_cohort_report": "present_day182", "additional_report": "each_ancient_cohort_and_all686",
                                  "uncovered_tails": "exclude_and_report_fraction_no_extrapolation"},
            "competitors": {"AncestralPaths": simulation.AUTHOR_COMMIT,
                            "ARGMix_released": "c2481248eb0de5fda659d68f59c51e0e27becd20",
                            "ARGformer": "f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596"},
            "original_paper_CNN_reproduced": False, "superiority_claims": False},
        "sampler_handoff": {"initial_stride_geometry": sampler_geometry(),
                            "zero_initial_stride_policy": "fail_cell_preserve_outputs_no_stride_change_or_parent_replacement",
                            "source_class_caps": {"non_BAA_per_path": 12500, "BAA_per_path": 6250},
                            "training_policy": "v2_realized_source_schedule_with_disclosed_shortfalls",
                            "requested_per_parent": {"present_day": 50000, "Bronze_Age": 50000,
                                                     "BAA": 12500, "Neo": 25000, "Yam": 25000},
                            "holdout_requested_per_cohort_per_parent": 10000,
                            "sampling_or_training_executed_here": False}
    }


def validate_config(config):
    if config != default_config():
        raise ValueError("Frozen full-panel recipe/runtime/dependency/config differs")
    rows = roster()
    simulation, _ = helpers()
    old = json.loads(confined("configs/biological_paths_execution_pilot_v1.json").read_text())
    old_native = json.loads(confined("configs/biological_paths_relate_v1.json").read_text())
    known = {v for parent in old["parent_registry"] for v in parent["seeds"].values()}
    for values in old_native["parent_seeds"].values():
        known.add(values["initial"])
        known.update(values["size_base"] + i for i in range(6))
    fresh = {v for parent in parents() for v in parent["seeds"].values()}
    fresh.update(v for parent in parents() for v in parent["relate_size_iteration_seeds"])
    if known & fresh:
        raise ValueError("New stream collides with the pinned earlier pilot streams")
    retirement = {"present_bronze": 166, "BAA": 169, "Yam": 177, "Neo": 259,
                  "WHG": 600, "EHG": 600, "Ana": 800, "CHG": 800}
    for row in rows:
        if not 0 <= row["age_generations"] < retirement[row["sampling_population"]]:
            raise ValueError("Observed date exceeds population lifetime")
        if row["role"] == "query" and row["age_generations"] >= 350:
            raise ValueError("Scored query is older than native census")
    native_tools(config)
    return {"kind": KIND, "diploids": 476, "haplotypes": 952, "queries": 686,
            "pure_references": 266, "frozen_parents": 5, "new_purpose_seeds": 20,
            "previous_pilot_seed_collisions": 0, "simulation_executed": False,
            "Relate_executed": False, "output_written": False,
            "sample_template_sha256": digest(rows),
            "demography_sha256": digest(simulation.build_demography().asdict())}


def native_tools(config):
    pinned = config["relate"]
    installation = json.loads(confined(pinned["installation"]["path"]).read_text())
    if (sha(confined(pinned["installation"]["path"])) != pinned["installation"]["sha256"]
            or installation["version"] != "1.2.4"
            or installation["archive_sha256"] != "9d6fe0f25846da9a846099b0d7c8d1f9f529aee0042e92803aef72b82e771438"
            or sha(confined(installation["license_file"])) != installation["license_sha256"]):
        raise ValueError("Existing authorized Relate installation/provenance changed")
    release = confined(installation["release_root"])
    result = {}
    for relative, expected in pinned["tool_sha256"].items():
        path = release / relative
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("Pinned native tool changed: " + relative)
        result[relative] = str(path)
    return result


def json_write(path, value, guard):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    guard(len(data) + 8192)
    with Path(path).open("xb") as stream:
        stream.write(data)
    guard(0)


def text_write(path, value, guard):
    data = value.encode()
    guard(len(data) + 8192)
    with Path(path).open("xb") as stream:
        stream.write(data)
    guard(0)


def npz_write(path, value, guard):
    import numpy as np
    guard(sum(array.nbytes for array in value.values()) + 65536)
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **value)
    guard(0)


def generate_parent(config, parent, output, guard):
    import msprime
    import numpy as np
    import tskit
    simulation, _ = helpers()
    rows = roster()
    # Same source grouping: one91-diploid present SampleSet, then ancient diploids.
    samples = [msprime.SampleSet(91, ploidy=2, population="present_bronze", time=0)]
    samples += [msprime.SampleSet(1, ploidy=2, population=rows[i]["sampling_population"],
                                 time=rows[i]["age_generations"]) for i in range(182, 952, 2)]
    guard(config["native_step_headroom_bytes"])
    truth = msprime.sim_ancestry(samples=samples, demography=simulation.build_demography(),
                                sequence_length=2000000, recombination_rate=1e-8,
                                model="hudson", discrete_genome=True, record_migrations=False,
                                random_seed=parent["seeds"]["ancestry"])
    if list(truth.samples()) != list(range(952)):
        raise ValueError("Native952 sample order changed")
    for row in rows:
        node = truth.node(row["sample_index"])
        if node.time != row["age_generations"] or simulation.POPULATIONS[node.population] != row["sampling_population"]:
            raise ValueError("Simulated date/population order differs")
    for i in range(0, 952, 2):
        if truth.node(i).individual == tskit.NULL or truth.node(i).individual != truth.node(i + 1).individual:
            raise ValueError("Diploid pairing changed")
    targets = simulation.external_truth_intervals(truth, rows)
    if targets["native_paths_1based"].shape != (truth.num_trees, 686):
        raise ValueError("Full native truth shape differs")
    clean, mapping = simulation.strip_truth_annotations(truth)
    observed = msprime.sim_mutations(clean, rate=1.25e-8, model=msprime.JC69(),
                                    discrete_genome=True, record_provenance=False,
                                    random_seed=parent["seeds"]["mutations"])
    simulation.assert_observed_only(observed)
    manifest = []
    for row in rows:
        item = {k: row[k] for k in ("sample_index", "sample_id", "haplotype", "observed_cohort",
                                    "age_generations", "role", "observed_reference_class")}
        item["observed_node_id"] = int(mapping[row["sample_index"]])
        manifest.append(item)
    if [r["observed_node_id"] for r in manifest] != list(observed.samples()):
        raise ValueError("Clean sample correspondence differs")
    truth_dir, obs = output / "truth", output / "observed"
    guard(16384)
    truth_dir.mkdir()
    obs.mkdir()
    guard(int(truth.nbytes) + 1048576)
    truth.dump(truth_dir / "census_ancestry.trees")
    guard(0)
    npz_write(truth_dir / "full_truth_intervals.npz", targets, guard)
    json_write(truth_dir / "target_schema.json", {
        "parent_id": parent["parent_id"], "split": parent["split"],
        "native_paths_1based": [1, 2, 3, 4, 5, 6], "C4_0based": ["Ana", "CHG", "WHG", "EHG"],
        "native_to_C4": [0, 1, 2, 3, 0, 1],
        "query_sample_indices": targets["query_raw_node_ids"].tolist(),
        "query_keys": [[r["sample_id"], r["haplotype"]] for r in manifest if r["role"] == "query"],
        "sample_identity_includes_parent": True, "inferred_node_IDs_require_external_join": True,
        "truth_interval_boundaries_not_permitted_as_inference_windows": True}, guard)
    edges = np.arange(0, 2000001, 100, dtype=np.float64)
    npz_write(obs / "scoring_coordinates.npz",
              {"bin_edges_bp": edges, "positions_bp": (edges[:-1] + edges[1:]) / 2}, guard)
    json_write(obs / "sample_manifest.json", manifest, guard)
    names = [r["sample_id"] for r in manifest[::2]]
    text_write(obs / "sample_order.txt", "\n".join(names) + "\n", guard)
    text_write(obs / "sample_ages.generations", "\n".join(str(r["age_generations"]) for r in manifest) + "\n", guard)
    text_write(obs / "population_labels.txt", "sample population group sex\n" + "".join(
        f'{r["sample_id"]} {r["observed_cohort"]} {r["observed_cohort"]} NA\n' for r in manifest[::2]), guard)
    guard(config["native_step_headroom_bytes"])
    with (obs / "genotypes.vcf").open("x") as stream:
        observed.write_vcf(stream, ploidy=2, individual_names=names, contig_id="22",
                           position_transform=lambda p: np.asarray(p, dtype=np.int64) + 1)
    guard(0)
    positions = observed.tables.sites.position
    if not np.array_equal(positions, positions.astype(np.int64)):
        raise ValueError("Mutation SNP coordinates are not discrete")
    json_write(obs / "site_coordinate_manifest.json", {
        "build": "synthetic", "contig_id": "22", "length_bp": 2000000,
        "rows": [{"site_id": i, "tree_position_0based": int(p), "VCF_position_1based": int(p) + 1}
                 for i, p in enumerate(positions)]}, guard)
    text_write(obs / "uniform_recombination_map.txt",
               "Position(bp)\tRate(cM/Mb)\tMap(cM)\n1\t1\t0\n2000001\t1\t2\n", guard)
    json_write(obs / "observation_manifest.json", {
        "parent_id": parent["parent_id"], "split": parent["split"], "config_kind": KIND,
        "input_kind": "simulated_phased_genotypes_for_Relate", "samples": 952, "diploids": 476,
        "query_haplotypes": 686, "pure_reference_haplotypes": 266, "sites": observed.num_sites,
        "census_or_ancestry_annotations_present": False, "cleaned_oracle_exported": False,
        "scoring_bin_bp": 100, "Relate_executed": False,
        "sample_manifest": "sample_manifest.json", "sample_template_sha256": digest(rows)}, guard)
    # Hashing target outputs is confined to the generation stage, never Relate.
    files = {str(p.relative_to(output)): {"sha256": sha(p), "bytes": p.stat().st_size}
             for folder in (truth_dir, obs) for p in sorted(folder.iterdir()) if p.is_file()}
    receipt = {"parent_id": parent["parent_id"], "split": parent["split"],
               "status": "observations_and_external_targets_prepared",
               "files": files, "sites": observed.num_sites, "query_haplotypes": 686,
               "truth_intervals": truth.num_trees, "Relate_executed": False}
    json_write(output / "generation.json", receipt, guard)
    return receipt


def observed_assets(config, parent, output):
    import numpy as np
    obs = output / "observed"
    paths = {name: obs / name for name in OBSERVED_FILES}
    if any(p.is_symlink() or not p.is_file() for p in paths.values()):
        raise ValueError("Observed whitelist input missing/nonregular")
    meta = json.loads(paths["observation_manifest.json"].read_text())
    if (meta["parent_id"] != parent["parent_id"] or meta["split"] != parent["split"]
            or meta["samples"] != 952 or meta["diploids"] != 476
            or meta["query_haplotypes"] != 686 or meta["pure_reference_haplotypes"] != 266
            or meta["census_or_ancestry_annotations_present"] is not False
            or meta["sample_template_sha256"] != config["roster"]["sample_template_sha256"]):
        raise ValueError("Full observed panel provenance differs")
    rows = json.loads(paths["sample_manifest.json"].read_text())
    expected = roster()
    if len(rows) != 952:
        raise ValueError("Observed roster lost haplotypes")
    for actual, original in zip(rows, expected):
        for key in ("sample_index", "sample_id", "haplotype", "observed_cohort",
                    "age_generations", "role", "observed_reference_class"):
            if actual[key] != original[key]:
                raise ValueError("Observed sample/date/role order differs: " + key)
    names = [r["sample_id"] for r in rows[::2]]
    if paths["sample_order.txt"].read_text().splitlines() != names:
        raise ValueError("Diploid order differs")
    ages = np.asarray([r["age_generations"] for r in rows], dtype=np.float64)
    if not np.array_equal(np.loadtxt(paths["sample_ages.generations"]), ages):
        raise ValueError("Haplotype ages differ")
    labels = [line.split() for line in paths["population_labels.txt"].read_text().splitlines()]
    if labels != [["sample", "population", "group", "sex"]] + [
            [r["sample_id"], r["observed_cohort"], r["observed_cohort"], "NA"] for r in rows[::2]]:
        raise ValueError("Observed cohort labels differ")
    mapping = json.loads(paths["site_coordinate_manifest.json"].read_text())
    positions = {}
    for record in mapping["rows"]:
        pos = record["VCF_position_1based"]
        if pos != record["tree_position_0based"] + 1 or not 1 <= pos <= 2000000 or pos in positions:
            raise ValueError("Observed SNP coordinate join differs")
        positions[pos] = record
    if paths["uniform_recombination_map.txt"].read_text().split() != [
            "Position(bp)", "Rate(cM/Mb)", "Map(cM)", "1", "1", "0", "2000001", "1", "2"]:
        raise ValueError("Constant map differs")
    with np.load(paths["scoring_coordinates.npz"], allow_pickle=False) as grid:
        if (not np.array_equal(grid["positions_bp"], np.arange(50, 2000000, 100))
                or not np.array_equal(grid["bin_edges_bp"], np.arange(0, 2000001, 100))):
            raise ValueError("Observation-defined fixed grid differs")
    return {"paths": paths, "rows": rows, "names": names, "ages": ages,
            "position_join": positions, "source_sha256": {name: sha(path) for name, path in paths.items()}}


def haps_check(path, eligible):
    expected = {r["pos1"]: r for r in eligible}
    retained, previous = [], 0
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            pos = int(fields[2])
            if len(fields) != 957 or pos <= previous or pos not in expected:
                raise ValueError("Native952 haps row/SNP order differs")
            row = expected[pos]
            if fields[3:5] != row["fields"][3:5] or list(map(int, fields[5:])) != row["genotypes"]:
                raise ValueError("Native phased haplotype columns differ")
            retained.append(pos)
            previous = pos
    if not retained:
        raise ValueError("No native sites retained")
    return retained


def native_step(argv, work, journal, label, config, guard):
    """Existing native tools, own TMPDIR, phase-boundary budget checks."""
    guard(config["native_step_headroom_bytes"])
    log = work / f"{len(journal):02d}_{label}.log"
    record = {"label": label, "argv": argv, "command_sha256": digest(argv),
              "cwd": str(work), "log": str(log.relative_to(ROOT)), "status": "started"}
    journal.append(record)
    started = time.monotonic()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(work / "tmp"),
                       PYTHONDONTWRITEBYTECODE="1")
    try:
        with log.open("x") as stream:
            process = subprocess.Popen(argv, cwd=work, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=config["relate"]["timeout_seconds_per_step"])
            except BaseException:
                # Stop only this newly created native command group on timeout/interruption.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
                raise
        record.update(returncode=returncode, status="complete" if returncode == 0 else "failed")
    except Exception as error:
        record.update(status="exception", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["log_sha256"] = sha(log)
        guard(0)
        # Owned journal only. Earlier native command records stay in this file.
        data = (json.dumps({"steps": journal, "copyright_notice": "Copyright © 2018, University of Oxford"},
                           indent=2, allow_nan=False) + "\n").encode()
        guard(len(data) + 8192)
        (work / "commands.json").write_bytes(data)
        guard(0)
    if record["returncode"]:
        raise RuntimeError("Native step failed; partial owned outputs preserved: " + label)


def infer_parent(config, parent, output, guard, expected_input_refs):
    _, native_helper = helpers()
    assets = observed_assets(config, parent, output)
    actual_refs = {name: {"path": str(path.relative_to(ROOT)), "sha256": assets["source_sha256"][name]}
                   for name, path in assets["paths"].items()}
    if expected_input_refs != actual_refs:
        raise ValueError("Frozen observed-only input hashes differ from admitted/generated bytes")
    native = native_tools(config)
    work = output / "relate"
    guard(16384)
    work.mkdir()
    (work / "tmp").mkdir()
    text_write(work / "NOTICE.txt",
               "Relate1.2.4\nCopyright © 2018, University of Oxford\nPreviously authorized internal academic research use.\n", guard)
    guard(2 * assets["paths"]["genotypes.vcf"].stat().st_size + 1048576)
    eligible, counts = native_helper.prepare_inputs(assets, work / "inputs")
    guard(0)
    inputs, journal = work / "inputs", []
    f = native["bin/RelateFileFormats"]
    def run(argv, label):
        native_step(argv, work, journal, label, config, guard)
    run([f, "--mode", "ConvertFromVcf", "--haps", str(inputs / "input.haps"),
         "--sample", str(inputs / "converted.sample"), "-i", str(inputs / "input")], "convert_vcf")
    text_write(inputs / "input.sample",
               native_helper.canonical_sample_text((inputs / "converted.sample").read_text(), assets["names"]), guard)
    retained = haps_check(inputs / "input.haps", eligible)
    run([f, "--mode", "GenerateSNPAnnotations", "--haps", str(inputs / "input.haps"),
         "--sample", str(inputs / "input.sample"), "--poplabels", str(inputs / "input.poplabels"),
         "-o", str(inputs / "input")], "annotate_observed_cohorts")
    run(["bash", "-x", native["scripts/RelateParallel/RelateParallel.sh"], "-m", "1.25e-8",
         "-N", "30000", "--threads", "8", "--memory", "2",
         "--seed", str(parent["seeds"]["relate_initial"]), "--haps", str(inputs / "input.haps"),
         "--sample", str(inputs / "input.sample"), "--sample_ages", str(inputs / "sample_ages.txt"),
         "--map", str(inputs / "input.map"), "--annot", str(inputs / "input.annot"),
         "-o", "pre_size"], "relate_initial")
    pre_refs = {name: ref(str((work / name).relative_to(ROOT))) for name in ("pre_size.anc", "pre_size.mut")}
    run(["bash", "-x", native["scripts/EstimatePopulationSize/EstimatePopulationSize.sh"],
         "-i", "pre_size", "-o", "post_size", "-m", "1.25e-8", "--threads", "8",
         "--num_iter", "5", "--threshold", "0.5", "--years_per_gen", "28",
         "--seed", str(parent["seeds"]["relate_size_base"]), "--poplabels", str(inputs / "input.poplabels"),
         "--noplot"], "five_population_size_iterations")
    run([f, "--mode", "ConvertToTreeSequence", "-i", "post_size", "-o", "post_size"], "convert_post_size")
    guard(2 * (work / "post_size.trees").stat().st_size + 1048576)
    post = native_helper.clean_converted_tree(work / "post_size.trees", assets, retained,
                                              work / "post_size.observed.trees", "post_five_size_iterations_primary")
    post["samples"] = 952  # Shared helper's old280 receipt constant, no transformation change.
    inferred_rows = []
    for row, mapped in zip(assets["rows"], post["sample_index_to_inferred_tree_node_id"]):
        if row["sample_index"] != mapped["sample_index"]:
            raise ValueError("Native inferred sample mapping differs")
        item = dict(row)
        item["source_observed_node_id"] = item.pop("observed_node_id")
        item["inferred_tree_node_id"] = mapped["inferred_tree_node_id"]
        item["observed_node_id"] = mapped["inferred_tree_node_id"]
        inferred_rows.append(item)
    json_write(work / "observed_sample_manifest.json", inferred_rows, guard)
    guard(0)
    if assets["source_sha256"] != {name: sha(path) for name, path in assets["paths"].items()}:
        raise ValueError("Original observed assets changed")
    if any(record["sha256"] != sha(confined(record["path"])) for record in pre_refs.values()):
        raise ValueError("Population-size helper changed initial native files")
    post["observed_sample_manifest"] = ref(str((work / "observed_sample_manifest.json").relative_to(ROOT)))
    receipt = {"parent_id": parent["parent_id"], "split": parent["split"], "status": "post_five_iterations_inferred",
               "truth_or_oracle_read": False, "inferred_tree": ref(str((work / "post_size.observed.trees").relative_to(ROOT))),
               "observed_sample_manifest": post["observed_sample_manifest"],
               "input_observed_sha256": assets["source_sha256"], "filter_counts": counts,
               "native_converter_sites": len(retained), "converter_additional_attrition": len(eligible) - len(retained),
               "post_five_iterations": post, "coordinate_source_reference": config["relate"]["source_coordinate_reference"],
               "native_tools": {key: {"path": path, "sha256": sha(path)} for key, path in native.items()},
               "commands": ref(str((work / "commands.json").relative_to(ROOT))),
               "pre_size_native_files_unchanged": pre_refs,
               "outer_command_returncodes": [item["returncode"] for item in journal],
               "native_status_scope": "outer helper returncodes; bash-x trace retained; child statuses not independently asserted",
               "common_C4_and_native_six_path_consumers_use_same_post_size_tree": True,
               "native_sampler_geometry": sampler_geometry(post["sites"], parent["split"])}
    if not receipt["native_sampler_geometry"]["all_initial_strides_positive"]:
        receipt["status"] = "inference_prepared_native_sampler_zero_stride_cell_failed"
    json_write(work / "inference.json", receipt, guard)
    if not receipt["native_sampler_geometry"]["all_initial_strides_positive"]:
        raise ValueError("Native initial site stride is zero; preserve failed cell, no adaptive stride/resampling")
    return receipt


def handmade_check():
    """One tiny counterfactual, not stochastic simulation or native execution."""
    import msprime
    import numpy as np
    import tskit
    simulation, native_helper = helpers()
    rows = [{"sample_index": i, "role": "query", "observed_cohort": "BAA" if i >= 2 else "present_day"}
            for i in range(4)]
    tables = tskit.TableCollection(100)
    for _ in range(11):
        tables.populations.add_row()
    for _ in rows:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=0)
    root = tables.nodes.add_row(time=1000, population=10)
    for i in range(4):
        census = tables.nodes.add_row(time=350, flags=msprime.NODE_IS_CEN_EVENT,
                                     population=8 if i % 2 == 0 else 7)
        tables.edges.add_row(0, 100, census, i)
        tables.edges.add_row(0, 100, root, census)
    tables.sort()
    truth = tables.tree_sequence()
    targets = simulation.external_truth_intervals(truth, rows)
    if targets["native_paths_1based"].tolist() != [[1, 2, 5, 6]] or targets["source_classes_0based"].tolist() != [[0, 1, 0, 1]]:
        raise AssertionError("Native/C4 target mapping differs")
    changed = truth.dump_tables()
    pop = changed.nodes.population.copy()
    pop[5] = 7
    changed.nodes.population = pop
    alternate_truth = changed.tree_sequence()
    alternate_targets = simulation.external_truth_intervals(alternate_truth, rows)
    if alternate_targets["native_paths_1based"][0, 0] != 2:
        raise AssertionError("Hidden-truth counterfactual did not change the target")
    clean, _ = simulation.strip_truth_annotations(truth)
    alternate, _ = simulation.strip_truth_annotations(alternate_truth)
    clean.tables.assert_equals(alternate.tables)
    mapped = tskit.TableCollection(1)
    mapped.nodes.add_row(time=500)
    mapped.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    mapped.nodes.add_row(time=600)
    mapped.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=80)
    joined = native_helper.inferred_sample_mapping(mapped.tree_sequence(), {
        "rows": [{"sample_index": 0}, {"sample_index": 1}], "ages": np.asarray([0., 80.])})
    if [r["inferred_tree_node_id"] for r in joined] != [1, 3]:
        raise AssertionError("Actual noncontiguous sample IDs not preserved")
    for count, split, expected in ((2023, "train", False), (2024, "train", True),
                                   (1428, "discovery_holdout", False), (1429, "discovery_holdout", True)):
        if sampler_geometry(count, split)["all_initial_strides_positive"] is not expected:
            raise AssertionError("Native stride threshold differs")
    return {"hidden_truth_changed_clean_observation_unchanged": True,
            "native_stride_boundaries_checked": True,
            "native_six_path_and_C4_mapping_checked": True, "noncontiguous_inferred_ID_join_checked": True,
            "simulation_executed": False, "Relate_executed": False, "files_written": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--parent", default="ap_full_train_000")
    parser.add_argument("--stage", choices=("simulate", "relate", "all"), default="simulate")
    parser.add_argument("--admission", help="Root-owned job/output/runtime/storage record")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-handmade", action="store_true")
    args = parser.parse_args()
    config_path = confined(args.config)
    config = json.loads(config_path.read_text())
    actual_config, actual_source = sha(config_path), sha(confined(SOURCE))
    if args.expected_config_sha256 and args.expected_config_sha256 != actual_config:
        parser.error("Frozen config bytes changed")
    if args.expected_source_sha256 and args.expected_source_sha256 != actual_source:
        parser.error("Frozen producer bytes changed")
    result = validate_config(config)
    registry = {row["parent_id"]: row for row in config["parent_registry"]}
    if args.parent not in registry:
        parser.error("Parent must be one of the five frozen streams")
    parent = registry[args.parent]
    result.update(config_sha256=actual_config, source_sha256=actual_source, selected_parent=args.parent, stage=args.stage)
    if not args.execute:
        if args.check_handmade:
            result["handmade"] = handmade_check()
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if (not args.expected_config_sha256 or not args.expected_source_sha256 or not args.admission or args.check_handmade):
        parser.error("--execute needs root-reviewed source/config hashes and --admission; handmade check is separate")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CPU-only execution requires CUDA_VISIBLE_DEVICES=''")
    admission = json.loads(confined(args.admission).read_text())
    output_relative = config["output_root"] + "/" + args.parent
    output = confined(output_relative)
    if (admission["parent_id"] != args.parent or args.stage not in admission["allowed_stages"]
            or admission["output_directory"] != output_relative
            or admission["source_ref"] != {"path": SOURCE, "sha256": actual_source}
            or admission["config_ref"] != {"path": str(config_path.relative_to(ROOT)), "sha256": actual_config}
            or ref(admission["runtime_ref"]["path"]) != admission["runtime_ref"]):
        raise ValueError("Exact root job/parent/stage/source/config/runtime binding required")
    from arggnn.project_storage_full_reservation_v1 import budget_check_v9, validate_storage_v9
    storage = admission["storage"]
    validate_storage_v9(admission, config, storage, source_ref=admission["source_ref"],
                        config_ref=admission["config_ref"], runtime_ref=admission["runtime_ref"])
    def guard(planned=0):
        return budget_check_v9(storage, boundary=True, required_write_bytes=int(planned))
    frozen = {"kind": KIND, "config_sha256": actual_config, "source_sha256": actual_source,
              "selected_parent": parent, "all_parent_streams_frozen_before_truth": config["parent_registry"],
              "sample_template_sha256": config["roster"]["sample_template_sha256"],
              "dependencies": config["dependencies"], "scientific_endpoints": config["scientific_endpoints"]}
    if args.stage in ("simulate", "all"):
        guard(16384)
        output.parent.mkdir(exist_ok=True)
        output.mkdir(exist_ok=False)
        json_write(output / "freeze.json", frozen, guard)
    else:
        freeze = output / "freeze.json"
        if freeze.is_symlink() or not freeze.is_file() or json.loads(freeze.read_text()) != frozen:
            raise ValueError("Existing parent freeze differs; no automatic retry/overwrite")
    started = time.monotonic()
    try:
        generated = None
        if args.stage in ("simulate", "all"):
            generated = generate_parent(config, parent, output, guard)
        if args.stage in ("relate", "all"):
            if generated is not None:
                input_refs = {name: {"path": str((output / "observed" / name).relative_to(ROOT)),
                                     "sha256": generated["files"]["observed/" + name]["sha256"]}
                              for name in OBSERVED_FILES}
            else:
                input_refs = admission["observed_input_refs"]
                if (set(input_refs) != set(OBSERVED_FILES)
                        or any(value not in storage["reservation"]["consumer_bindings"]
                               for value in input_refs.values())):
                    raise ValueError("Relate-only admission must pin the nine observed whitelist inputs")
            infer_parent(config, parent, output, guard, input_refs)
        completion = {"kind": KIND, "parent_id": args.parent, "split": parent["split"],
                      "stage": args.stage, "status": "complete", "config_sha256": actual_config,
                      "source_sha256": actual_source, "elapsed_seconds": time.monotonic() - started,
                      "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "scope_budget": guard(0), "models_fitted": False}
        json_write(output / (args.stage + "_completion.json"), completion, guard)
        print(json.dumps({k: completion[k] for k in ("parent_id", "stage", "status", "elapsed_seconds")}), flush=True)
    except Exception as error:
        failure = {"kind": KIND, "parent_id": args.parent, "stage": args.stage,
                   "status": "failed_partial_preserved", "error_type": type(error).__name__,
                   "error": str(error), "elapsed_seconds": time.monotonic() - started}
        try:
            json_write(output / (args.stage + "_failure.json"), failure, guard)
        except Exception as receipt_error:
            print(json.dumps({"failure": failure, "failure_receipt_error": str(receipt_error)}),
                  file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()

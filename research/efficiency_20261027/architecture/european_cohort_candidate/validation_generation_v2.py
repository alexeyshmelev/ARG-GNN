"""Prepared plaintext validation-only copy of the canonical AP generator.

The scientific generator body is preserved. Changes: public function name,
validation-only split guard, matching guard message and adapter provenance.
No AST/source execution, source monkeypatch, launcher or permissive guard.
Caller must verify the frozen validation plan and provide ROOT supervision.
This module has no command-line entrypoint and does not launch on import.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping
from arggnn.benchmarking.european_recipes import (
    ROOT, load_recipe, material_blockers, _seeds, verify_sources, _load_ap_runtime, _demography)

def generate_validation_parent(recipe: Mapping, parent: Mapping, output, *,
                                   guard: Callable[[int], None],
                                   execution_check: Callable[[], None],
                                   write_headroom_bytes: int,
                                   require_paper_match: bool = True) -> dict:
    """Generate one complete configurable-length AP parent using real msprime.

    The caller must first authorize execution, enforce the current hold, admit
    the entire output scope, and supervise finite wall/RSS limits. There are no
    permissive default guard/approval callbacks, fallback, shortening or redraw.
    Output schemas and guarded writers are reused from the pinned full-panel
    producer. All length-dependent coordinates and map endpoints are new values.

    Strict paper mode rejects unresolved demographic/source conflicts. An
    explicitly declared common-data reconstruction may opt out, with those
    conflicts retained in its receipt; that is not a paper-reproduction claim.
    """
    if (not callable(guard) or not callable(execution_check)
            or type(write_headroom_bytes) is not int or write_headroom_bytes <= 0
            or type(require_paper_match) is not bool):
        raise ValueError("Explicit approval/hold check, existing guard and write bound required")
    if recipe.get("schema") != "paper_recipe_v1" or recipe.get("id") != "ancestralpaths_european":
        raise ValueError("This writer supports only the complete AP European parent")
    if require_paper_match:
        registered = load_recipe(recipe["id"])
        fields = ("parameters", "generator", "evidence", "source_refs")
        try:
            declared = json.dumps({key: recipe[key] for key in fields},
                                  sort_keys=True, separators=(",", ":"), allow_nan=False)
            canonical = json.dumps({key: registered[key] for key in fields},
                                   sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Strict paper generation requires canonical registered recipe metadata") from error
        if declared != canonical:
            raise ValueError("Strict paper generation requires canonical registered parameters, generator, evidence and source refs")
    blockers = material_blockers(recipe)
    if require_paper_match and blockers:
        raise ValueError("Paper-matched generation blocked: " + "; ".join(blockers))
    parameters = recipe["parameters"]
    length = parameters["length_bp"]
    bin_bp = parameters["auxiliary_scoring_grid_bp"]
    if (type(length) is not int or length <= 0 or type(bin_bp) is not int
            or bin_bp <= 0 or length % bin_bp
            or (require_paper_match and length != 200000000)
            or parameters["diploids"] != 476 or parameters["ploidy"] != 2
            or parameters["haplotypes"] != 952):
        raise ValueError("Complete declared AP length, exact 476 diploids and uncropped grid required")
    if (parameters["mutation_model"] != "JC69" or parameters["genotype_error_rate"] != 0
            or parameters["discrete_genome"] is not True
            or parameters["ancestry_model"] != "hudson"
            or parameters["query_haplotypes"] != 686
            or parameters["pure_reference_haplotypes"] != 266
            or parameters["demography"]["census_time"] != 350
            or parameters["demography"]["growth_rates"] != 0
            or parameters["demography"]["migration_rates"] != 0):
        raise ValueError("Only the declared census-350, discrete Hudson/JC69, no-error AP condition is supported")
    for rate in ("recombination_rate", "mutation_rate"):
        value = parameters[rate]
        if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError("Finite positive AP mutation/recombination rates required")
    seeds = _seeds(parent["seeds"])
    if (parent.get("split") != "validation"
            or not isinstance(parent.get("parent_id"), str) or not parent["parent_id"]):
        raise ValueError("An independently registered validation parent is required")
    execution_check()
    guard(write_headroom_bytes)
    verify_sources(recipe)
    msprime, np, tskit, full = _load_ap_runtime()
    output = full.confined(output)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("ROOT must supply a fresh empty already-admitted parent directory")
    simulation, _ = full.helpers()
    rows = full.roster()
    if len(rows) != 952:
        raise ValueError("The pinned complete 476-person simulation age roster is required")
    names = [item["name"] for item in parameters["demography"]["populations"]]
    if names != list(simulation.POPULATIONS):
        raise ValueError("Original demographic population order must be preserved")
    samples = [msprime.SampleSet(91, ploidy=2, population="present_bronze", time=0)]
    samples += [msprime.SampleSet(1, ploidy=2, population=rows[i]["sampling_population"],
                                 time=rows[i]["age_generations"]) for i in range(182, 952, 2)]
    truth = msprime.sim_ancestry(
        samples=samples, demography=_demography(msprime, parameters["demography"]),
        sequence_length=length, recombination_rate=parameters["recombination_rate"],
        model=parameters["ancestry_model"], discrete_genome=parameters["discrete_genome"],
        record_migrations=False, random_seed=seeds["ancestry"])
    guard(write_headroom_bytes)
    if list(truth.samples()) != list(range(952)):
        raise ValueError("The complete original sample order changed")
    for row in rows:
        node = truth.node(row["sample_index"])
        if (node.time != row["age_generations"]
                or simulation.POPULATIONS[node.population] != row["sampling_population"]):
            raise ValueError("Original sample age/population correspondence changed")
    for index in range(0, 952, 2):
        if (truth.node(index).individual == tskit.NULL
                or truth.node(index).individual != truth.node(index + 1).individual):
            raise ValueError("Diploid pairing changed")
    targets = simulation.external_truth_intervals(truth, rows)
    if targets["native_paths_1based"].shape != (truth.num_trees, 686):
        raise ValueError("Complete native six-path truth shape changed")
    clean, mapping = simulation.strip_truth_annotations(truth)
    observed = msprime.sim_mutations(
        clean, rate=parameters["mutation_rate"], model=msprime.JC69(),
        discrete_genome=parameters["discrete_genome"], record_provenance=False,
        random_seed=seeds["mutations"])
    simulation.assert_observed_only(observed)
    manifest = []
    for row in rows:
        item = {key: row[key] for key in ("sample_index", "sample_id", "haplotype",
                "observed_cohort", "age_generations", "role", "observed_reference_class")}
        item["observed_node_id"] = int(mapping[row["sample_index"]])
        manifest.append(item)
    if [row["observed_node_id"] for row in manifest] != list(observed.samples()):
        raise ValueError("Sanitized observed sample mapping changed")
    truth_dir, obs = output / "truth", output / "observed"
    guard(16384)
    truth_dir.mkdir()
    obs.mkdir()
    guard(int(truth.nbytes) + 1048576)
    truth.dump(truth_dir / "census_ancestry.trees")
    guard(0)
    full.npz_write(truth_dir / "full_truth_intervals.npz", targets, guard)
    full.json_write(truth_dir / "target_schema.json", dict(
        parent_id=parent["parent_id"], split=parent["split"],
        native_paths_1based=[1, 2, 3, 4, 5, 6], C4_0based=["Ana", "CHG", "WHG", "EHG"],
        native_to_C4=[0, 1, 2, 3, 0, 1],
        query_sample_indices=targets["query_raw_node_ids"].tolist(),
        query_keys=[[row["sample_id"], row["haplotype"]] for row in manifest if row["role"] == "query"],
        sample_identity_includes_parent=True, inferred_node_IDs_require_external_join=True,
        truth_interval_boundaries_not_permitted_as_inference_windows=True,
        primary_endpoint="native_six_census350_paths",
        C4_is_separately_named_derived_endpoint=True), guard)
    edges = np.arange(0, length + 1, bin_bp, dtype=np.float64)
    full.npz_write(obs / "scoring_coordinates.npz",
                   {"bin_edges_bp": edges, "positions_bp": (edges[:-1] + edges[1:]) / 2}, guard)
    full.json_write(obs / "sample_manifest.json", manifest, guard)
    sample_names = [row["sample_id"] for row in manifest[::2]]
    full.text_write(obs / "sample_order.txt", "\n".join(sample_names) + "\n", guard)
    full.text_write(obs / "sample_ages.generations",
                    "\n".join(str(row["age_generations"]) for row in manifest) + "\n", guard)
    full.text_write(obs / "population_labels.txt", "sample population group sex\n" + "".join(
        f'{row["sample_id"]} {row["observed_cohort"]} {row["observed_cohort"]} NA\n'
        for row in manifest[::2]), guard)
    guard(write_headroom_bytes)
    with (obs / "genotypes.vcf").open("x") as stream:
        observed.write_vcf(stream, ploidy=2, individual_names=sample_names, contig_id="22",
                           position_transform=lambda positions: np.asarray(positions, dtype=np.int64) + 1)
    guard(0)
    positions = observed.tables.sites.position
    if not np.array_equal(positions, positions.astype(np.int64)):
        raise ValueError("Mutation SNP coordinates must remain discrete")
    full.json_write(obs / "site_coordinate_manifest.json", dict(
        build="synthetic", contig_id="22", length_bp=length,
        rows=[dict(site_id=i, tree_position_0based=int(position), VCF_position_1based=int(position)+1)
              for i, position in enumerate(positions)]), guard)
    genetic_cM = length * parameters["recombination_rate"] * 100
    full.text_write(obs / "uniform_recombination_map.txt",
                    "Position(bp)\tRate(cM/Mb)\tMap(cM)\n"
                    f'1\t{parameters["recombination_rate"]*1e8:g}\t0\n'
                    f'{length+1}\t{parameters["recombination_rate"]*1e8:g}\t{genetic_cM:g}\n', guard)
    full.json_write(obs / "observation_manifest.json", dict(
        parent_id=parent["parent_id"], split=parent["split"], config_kind=recipe["id"],
        input_kind="simulated_phased_genotypes_for_Relate", samples=952, diploids=476,
        query_haplotypes=686, pure_reference_haplotypes=266, sites=observed.num_sites,
        census_or_ancestry_annotations_present=False, cleaned_oracle_exported=False,
        scoring_bin_bp=bin_bp, length_bp=length, Relate_executed=False,
        sample_manifest="sample_manifest.json", sample_template_sha256=full.digest(rows)), guard)
    files = {str(path.relative_to(output)): {"sha256": full.sha(path), "bytes": path.stat().st_size}
             for folder in (truth_dir, obs) for path in sorted(folder.iterdir()) if path.is_file()}
    receipt = dict(
        parent_id=parent["parent_id"], split=parent["split"],
        status="observations_and_external_targets_prepared", files=files,
        sites=observed.num_sites, query_haplotypes=686, truth_intervals=truth.num_trees,
        sequence_length_bp=length, Relate_executed=False,
        recipe_id=recipe["id"], recipe_canonical_sha256=full.digest(recipe),
        parameters_canonical_sha256=full.digest(parameters),
        generator_source_ref=copy.deepcopy(recipe["generator"]["source_ref"]),
        primary_endpoint="native_six_census350_paths", paper_reproduction_claimed=False,
        material_paper_generation_conflicts=blockers, seeds=seeds)
    receipt["generation_adapter_source_ref"] = dict(
        path=str(Path(__file__).resolve().relative_to(ROOT)),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    receipt["independent_validation_extension"] = True
    full.json_write(output / "generation.json", receipt, guard)
    return receipt

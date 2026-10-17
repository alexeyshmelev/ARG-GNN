"""Inactive AP paper-scale200Mb generator using the released author demography.

Same476 diploids/ages and original native generation functions. Only physical
sequence/map/grid length changes. New seeds/splits are disclosed reconstruction,
not recovered original paper realization. ROOT supplies its actual guard/caps.
"""
from arggnn import prepare_ancestralpaths_full_panel_discovery_v1 as original
ROOT=original.ROOT
KIND="AP_paper_scale_200Mb_frozen_TreeLoom_v1"
helpers=original.helpers
roster=original.roster
json_write=original.json_write
text_write=original.text_write
npz_write=original.npz_write
sha=original.sha
digest=original.digest

def generate_parent(config, parent, output, guard):
    if config.get("kind") != KIND or config.get("simulation",{}).get("length_bp") != 200000000:
        raise ValueError("Exact fixed200Mb AP condition required")
    rows_meta=config.get("parent_registry",[])
    if parent not in rows_meta or parent.get("split") not in ("train","evaluation"):
        raise ValueError("Only the proposed3TRAIN/5evaluation registry; ROOT must authenticate and register")
    if not callable(guard) or type(config.get("native_step_headroom_bytes")) is not int or config["native_step_headroom_bytes"] <= 0:
        raise ValueError("Actual ROOT resource guard and headroom required")
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
                                sequence_length=200000000, recombination_rate=1e-8,
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
    edges = np.arange(0, 200000001, 100, dtype=np.float64)
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
        "build": "synthetic", "contig_id": "22", "length_bp": 200000000,
        "rows": [{"site_id": i, "tree_position_0based": int(p), "VCF_position_1based": int(p) + 1}
                 for i, p in enumerate(positions)]}, guard)
    text_write(obs / "uniform_recombination_map.txt",
               "Position(bp)\tRate(cM/Mb)\tMap(cM)\n1\t1\t0\n200000001\t1\t200\n", guard)
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

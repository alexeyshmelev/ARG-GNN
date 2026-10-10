"""Four-parent source-derived biological pilot; no simulation at import time."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import msprime
import numpy as np
import tskit

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
AGE_SOURCE = ".runtime/author_sources/ancestralpaths_77db3cfae100/example/mesoneo_ages_pop.txt"
AGE_SHA256 = "3cadb16abbc0a1913eadb183d324eadc26c035d61e5116cd58675f52a942b7de"
AUTHOR_COMMIT = "77db3cfae100641c35218b87e1451e14f3074666"
POPULATIONS = ("present_bronze", "Yam", "Neo", "EHG", "WHG", "NE", "BAA", "CHG", "Ana", "WA", "trunk")
SIZES = (50000, 20000, 50000, 3000, 2000, 15000, 20000, 3000, 5000, 15000, 15000)
ADMIXTURES = (
    (166, "present_bronze", ("Neo", "Yam"), (0.5, 0.5)),
    (259, "Neo", ("WHG", "Ana"), (0.25, 0.75)),
    (177, "Yam", ("CHG", "EHG"), (0.5, 0.5)),
    (169, "BAA", ("CHG", "Ana"), (0.25, 0.75)),
)
MERGES = ((600, ("WHG", "EHG"), "NE"), (800, ("CHG", "Ana"), "WA"), (1500, ("NE", "WA"), "trunk"))
# Source simulation order, with reference roles before query roles within cohort.
PANEL = (
    ("present_day", "present_bronze", 0, 32, 0),
    ("Bronze_Age", "present_bronze", 8, 8, 135),
    ("BAA", "BAA", 4, 4, 155),
    ("Neo", "Neo", 16, 8, 200),
    ("Yam", "Yam", 4, 8, 170),
    ("WHG", "WHG", 12, 0, None),
    ("EHG", "EHG", 12, 0, None),
    ("Ana", "Ana", 12, 0, None),
    ("CHG", "CHG", 12, 0, None),
)
SOURCE_CLASSES = {"Ana": 0, "CHG": 1, "WHG": 2, "EHG": 3}
PATH_TO_CLASS = np.asarray((0, 1, 2, 3, 0, 1), dtype=np.uint8)
SEED_DOMAIN = "arggnn|biological_paths_execution_pilot_v1"
CONFIG_KIND = "biological_paths_execution_pilot_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(value: str | Path) -> Path:
    p = Path(value)
    p = (ROOT / p if not p.is_absolute() else p).resolve()
    if ROOT not in p.parents:
        raise ValueError("pilot path must remain strictly inside project")
    return p


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def materialize_sample_template() -> list[dict[str, Any]]:
    """Exact280 rows from pinned observed date templates; never inspect truth."""
    p = project_path(AGE_SOURCE)
    if sha256(p) != AGE_SHA256:
        raise ValueError("pinned author example ages drift")
    ages: dict[str, list[tuple[int, int]]] = {}
    for line_number, line in enumerate(p.read_text().splitlines(), 1):
        cohort, date = line.split()
        ages.setdefault(cohort, []).append((int(date), line_number))
    rows = []
    diploid = 0
    for cohort, population, refs, queries, query_date in PANEL:
        dates = sorted(ages.get(cohort, []))
        for role, count in (("reference", refs), ("query", queries)):
            for j in range(count):
                source_line = None
                if role == "reference":
                    index = ((2 * j + 1) * len(dates)) // (2 * count)
                    date, source_line = dates[index]
                    actual_role = "pure_source_reference" if population in SOURCE_CLASSES else "mixed_reference_only"
                else:
                    date, actual_role = query_date, "query"
                for h in (0, 1):
                    rows.append({
                        "sample_index": len(rows), "diploid_index": diploid,
                        "sample_template_id": f"d{diploid:03d}", "haplotype": h,
                        "observed_cohort": cohort, "sampling_population": population,
                        "age_generations": date, "role": actual_role,
                        "observed_reference_class": SOURCE_CLASSES.get(population) if actual_role == "pure_source_reference" else None,
                        "author_age_line_1based": source_line,
                    })
                diploid += 1
    if len(rows) != 280 or diploid != 140:
        raise ValueError("source-derived panel arithmetic differs from proposal")
    return rows


def fixed_parent_registry() -> list[dict[str, Any]]:
    """Fixed domain-separated32-bit streams, independently derived per purpose."""
    registry = []
    for split in ("train", "val"):
        for i in range(2):
            seeds = {}
            digests = {}
            for purpose in ("ancestry", "mutations"):
                text = f"{SEED_DOMAIN}|master=20261010|split={split}|parent={i:03d}|purpose={purpose}"
                digest = hashlib.sha256(text.encode("ascii")).digest()
                seeds[purpose] = int.from_bytes(digest[:8], "big") % (2**32 - 1) + 1
                digests[purpose] = digest.hex()
            registry.append({"parent_id": f"bio_paths_pilot_{split}_{i:03d}",
                             "split": split, "replicate": i, "seeds": seeds,
                             "seed_domain_sha256": digests})
    values = [s for row in registry for s in row["seeds"].values()]
    if len(set(values)) != 8:
        raise ValueError("pilot stream collision")
    return registry


def metadata_seed_values(value: Any) -> set[int]:
    """Read only explicit seed-bearing JSON fields, including frozen inventories."""
    result: set[int] = set()
    def integers(v):
        if isinstance(v, int) and not isinstance(v, bool):
            result.add(v)
        elif isinstance(v, dict):
            for c in v.values():
                integers(c)
        elif isinstance(v, list):
            for c in v:
                integers(c)
    def visit(v):
        if isinstance(v, dict):
            for key, child in v.items():
                low = key.lower()
                if low == "seed" or low.endswith("_seed") or low == "seeds" or low.endswith("_seeds"):
                    integers(child)
                if low == "seed_uses" and isinstance(child, list):
                    for use in child:
                        if isinstance(use, dict):
                            integers(use.get("value"))
                visit(child)
        elif isinstance(v, list):
            for child in v:
                visit(child)
    visit(value)
    return result


def inspect_seed_metadata(paths: list[str]) -> dict[str, Any]:
    """Metadata only; no target arrays, trees or genotype files are opened."""
    records, values = [], set()
    for relative in paths:
        p = project_path(relative)
        if p.suffix != ".json":
            raise ValueError("seed registry must be explicit JSON metadata")
        content = json.loads(p.read_text())
        found = metadata_seed_values(content)
        values.update(found)
        records.append({"path": relative, "sha256": sha256(p), "distinct_seed_values": len(found)})
    candidate = {v for row in fixed_parent_registry() for v in row["seeds"].values()}
    collision = sorted(candidate.intersection(values))
    if collision:
        raise ValueError(f"pilot seeds collide with prior metadata: {collision}")
    return {"metadata_files": records, "prior_distinct_seed_values": len(values),
            "prior_seed_values_sha256": canonical_digest(sorted(values)),
            "candidate_simulation_seeds": sorted(candidate), "collision_count": 0,
            "scope": "listed config/manifest metadata plus frozen prior_inventory; no unlisted future seeds claimed",
            "arrays_or_trees_read": False}


def build_demography() -> msprime.Demography:
    d = msprime.Demography()
    for name, size in zip(POPULATIONS, SIZES):
        d.add_population(name=name, initial_size=size)
    for t, derived, ancestors, weights in ADMIXTURES:
        d.add_admixture(time=t, derived=derived, ancestral=list(ancestors), proportions=list(weights))
    for t, derived, ancestor in MERGES:
        d.add_population_split(time=t, derived=list(derived), ancestral=ancestor)
    d.add_census(time=350)
    d.sort_events()
    d.validate()
    return d


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("kind") != CONFIG_KIND or config.get("splits") != {"train": 2, "val": 2, "test": 0}:
        raise ValueError("only the four-parent development pilot is supported")
    expected = {"length_bp": 2000000, "mutation_rate": 1.25e-8, "recombination_rate": 1e-8,
                "ancestry_model": "hudson", "mutation_model": "JC69",
                "discrete_genome": True, "record_migrations": False, "census_time": 350,
                "score_bin_bp": 100}
    if config.get("simulation") != expected:
        raise ValueError("precise pilot simulation contract changed")
    if config.get("sample_template") != materialize_sample_template() or config.get("parent_registry") != fixed_parent_registry():
        raise ValueError("materialized dates/sample order or new seed registry changed")
    d = build_demography()
    if config.get("demography") != d.asdict():
        raise ValueError("fixed source-derived event model changed")
    for row in config["sample_template"]:
        retirement = {"present_bronze": 166, "Neo": 259, "Yam": 177, "BAA": 169,
                      "WHG": 600, "EHG": 600, "CHG": 800, "Ana": 800}[row["sampling_population"]]
        if not 0 <= row["age_generations"] < retirement:
            raise ValueError("sample age outside its fixed population lifetime")
        if row["role"] == "query" and row["age_generations"] >= 350:
            raise ValueError("query is not younger than census")
    audit = inspect_seed_metadata([r["path"] for r in config["seed_audit"]["metadata_files"]])
    if audit != config["seed_audit"]:
        raise ValueError("known registry metadata changed; refresh root review rather than silently resample")
    if config.get("runtime") != {"python": f"{sys.version_info.major}.{sys.version_info.minor}", "msprime": msprime.__version__,
                                 "tskit": tskit.__version__, "numpy": np.__version__}:
        raise ValueError("reviewed simulator runtime differs")
    project_path(config["output_directory"])
    return {"kind": CONFIG_KIND, "simulation_executed": False, "diploids": 140,
            "haplotypes": 280, "pure_reference_haplotypes": 96,
            "query_haplotypes": 120, "parents": 4,
            "demography_sha256": canonical_digest(d.asdict()),
            "sample_template_sha256": canonical_digest(config["sample_template"]),
            "seed_collisions": 0}


def strip_truth_annotations(ts: tskit.TreeSequence) -> tuple[tskit.TreeSequence, np.ndarray]:
    """Collapse census-only topology/boundaries before any observation mutation."""
    observed, mapping = ts.simplify(samples=list(ts.samples()), map_nodes=True,
                                    keep_unary=False, filter_sites=False,
                                    filter_populations=False, filter_individuals=False,
                                    record_provenance=False)
    if np.any(observed.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT):
        raise ValueError("branching census node survived observation construction")
    if any(mapping[int(u)] == tskit.NULL for u in ts.samples()):
        raise ValueError("sample identity lost during census removal")
    tables = observed.dump_tables()
    tables.nodes.population = np.full(observed.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.individual = np.full(observed.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.flags = tables.nodes.flags & tskit.NODE_IS_SAMPLE
    for table in (tables.nodes, tables.edges, tables.sites, tables.mutations):
        table.drop_metadata()
    tables.migrations.clear()
    tables.individuals.clear()
    tables.populations.clear()
    tables.provenances.clear()
    tables.metadata_schema = tskit.MetadataSchema(None)
    tables.metadata = b""
    result = tables.tree_sequence()
    assert_observed_only(result)
    return result, mapping


def assert_observed_only(ts: tskit.TreeSequence) -> None:
    if (ts.num_populations or ts.num_individuals or ts.num_migrations or ts.num_provenances
            or np.any(ts.tables.nodes.population != tskit.NULL)
            or np.any(ts.tables.nodes.individual != tskit.NULL)
            or np.any(ts.tables.nodes.flags != (ts.tables.nodes.flags & tskit.NODE_IS_SAMPLE))
            or ts.metadata):
        raise ValueError("truth annotation remained in oracle/genotype observation")
    for table in (ts.tables.nodes, ts.tables.edges, ts.tables.sites, ts.tables.mutations):
        if len(table.metadata):
            raise ValueError("truth metadata remained")


def external_truth_intervals(truth: tskit.TreeSequence, template: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """All raw intervals/query labels, only for the external target namespace."""
    queries = [row for row in template if row["role"] == "query"]
    ids = [row["sample_index"] for row in queries]
    boundaries, paths = [], []
    for tree in truth.trees():
        current = []
        for query in queries:
            u = query["sample_index"]
            while u != tskit.NULL and not (truth.node(u).flags & msprime.NODE_IS_CEN_EVENT):
                u = tree.parent(u)
            if u == tskit.NULL or tree.time(u) != 350:
                raise ValueError("missing/wrong native census ancestor")
            pop = truth.node(u).population
            base = {8: 1, 7: 2, 4: 3, 3: 4}.get(pop)
            if base is None:
                raise ValueError("query census ancestry outside four source populations")
            if query["observed_cohort"] == "BAA":
                if base not in (1, 2):
                    raise ValueError("BAA census outside native source paths")
                base += 4
            current.append(base)
        boundaries.append((tree.interval.left, tree.interval.right))
        paths.append(current)
    path_array = np.asarray(paths, dtype=np.uint8)
    return {"intervals_bp": np.asarray(boundaries, dtype=np.float64),
            "query_raw_node_ids": np.asarray(ids, dtype=np.int32),
            "native_paths_1based": path_array,
            "source_classes_0based": PATH_TO_CLASS[path_array - 1]}


@dataclass
class BiologicalParent:
    truth_ancestry: tskit.TreeSequence
    observed_mutated_oracle: tskit.TreeSequence
    truth_intervals: dict[str, np.ndarray]
    observed_sample_manifest: list[dict[str, Any]]
    parent_record: dict[str, Any]


def simulate_parent(config: dict[str, Any], parent: dict[str, Any]) -> BiologicalParent:
    """Root-only actual simulation entry; importing/validating never calls this."""
    template = config["sample_template"]
    samples = [msprime.SampleSet(1, ploidy=2, population=template[i]["sampling_population"],
                                time=template[i]["age_generations"])
               for i in range(0, 280, 2)]
    truth = msprime.sim_ancestry(samples=samples, demography=build_demography(),
                                sequence_length=2000000, recombination_rate=1e-8,
                                model="hudson", discrete_genome=True, record_migrations=False,
                                random_seed=parent["seeds"]["ancestry"])
    if list(truth.samples()) != list(range(280)):
        raise ValueError("msprime sample ordering differs from280-row manifest")
    for row in template:
        node = truth.node(row["sample_index"])
        if node.time != row["age_generations"] or POPULATIONS[node.population] != row["sampling_population"]:
            raise ValueError("simulated sample age/population order differs")
    for i in range(0, 280, 2):
        if truth.node(i).individual == tskit.NULL or truth.node(i).individual != truth.node(i + 1).individual:
            raise ValueError("simulation did not preserve diploid pairs")
    targets = external_truth_intervals(truth, template)
    clean, mapping = strip_truth_annotations(truth)
    # Mutate the cleaned genealogy: census unary nodes cannot alter observation boundaries.
    observed = msprime.sim_mutations(clean, rate=1.25e-8, model=msprime.JC69(),
                                    discrete_genome=True, record_provenance=False,
                                    random_seed=parent["seeds"]["mutations"])
    assert_observed_only(observed)
    actual = []
    for row in template:
        actual.append({
            "sample_index": row["sample_index"], "sample_id": parent["parent_id"] + "_" + row["sample_template_id"],
            "haplotype": row["haplotype"], "observed_node_id": int(mapping[row["sample_index"]]),
            "observed_cohort": row["observed_cohort"], "age_generations": row["age_generations"],
            "role": row["role"], "observed_reference_class": row["observed_reference_class"],
        })
    if [r["observed_node_id"] for r in actual] != list(observed.samples()):
        raise ValueError("cleaned sample/haplotype mapping differs")
    return BiologicalParent(truth, observed, targets, actual, parent)


def _json_new(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_parent(case: BiologicalParent, destination: Path) -> dict[str, Any]:
    """Truth and observation namespaces; no window/purity selection."""
    destination.mkdir()
    truth_dir, obs_dir = destination / "truth", destination / "observed"
    truth_dir.mkdir()
    obs_dir.mkdir()
    case.truth_ancestry.dump(truth_dir / "census_ancestry.trees")
    np.savez_compressed(truth_dir / "full_truth_intervals.npz", **case.truth_intervals)
    edges = np.arange(0, 2000000 + 100, 100, dtype=np.float64)
    positions = (edges[:-1] + edges[1:]) / 2
    rows = np.searchsorted(case.truth_intervals["intervals_bp"][:, 1], positions, side="right")
    np.savez_compressed(truth_dir / "targets_at_fixed_midpoints.npz",
                        positions_bp=positions,
                        query_raw_node_ids=case.truth_intervals["query_raw_node_ids"],
                        native_paths_1based=case.truth_intervals["native_paths_1based"][rows],
                        source_classes_0based=case.truth_intervals["source_classes_0based"][rows])
    # These coordinates are fixed by length/bin width, before any truth is consulted.
    np.savez_compressed(obs_dir / "scoring_coordinates.npz", bin_edges_bp=edges, positions_bp=positions)
    observed = case.observed_mutated_oracle
    observed.dump(obs_dir / "cleaned_oracle.trees")
    _json_new(obs_dir / "sample_manifest.json", case.observed_sample_manifest)
    names = [row["sample_id"] for row in case.observed_sample_manifest[::2]]
    with (obs_dir / "sample_order.txt").open("x") as stream:
        stream.write("\n".join(names) + "\n")
    with (obs_dir / "sample_ages.generations").open("x") as stream:
        stream.write("\n".join(str(row["age_generations"]) for row in case.observed_sample_manifest) + "\n")
    with (obs_dir / "population_labels.txt").open("x") as stream:
        stream.write("sample population group sex\n")
        for row in case.observed_sample_manifest[::2]:
            cohort = row["observed_cohort"]
            stream.write(f'{row["sample_id"]} {cohort} {cohort} NA\n')
    with (obs_dir / "genotypes.vcf").open("x") as stream:
        observed.write_vcf(stream, ploidy=2, individual_names=names, contig_id="22",
                           position_transform=lambda p: np.asarray(p, dtype=np.int64) + 1)
    positions_native = observed.tables.sites.position
    if not np.array_equal(positions_native, positions_native.astype(np.int64)):
        raise ValueError("discrete mutation sites are not integer coordinates")
    _json_new(obs_dir / "site_coordinate_manifest.json", {
        "vcf_position_rule": "integer zero-based mutation position+1", "build": "synthetic",
        "contig_id": "22", "length_bp": 2000000, "empirical_GRCh37_coordinate_claimed": False,
        "rows": [{"site_id": int(i), "tree_position_0based": int(p), "VCF_position_1based": int(p) + 1}
                 for i, p in enumerate(positions_native)],
    })
    with (obs_dir / "uniform_recombination_map.txt").open("x") as stream:
        stream.write("Position(bp)\tRate(cM/Mb)\tMap(cM)\n1\t1\t0\n2000001\t1\t2\n")
    _json_new(obs_dir / "observation_manifest.json", {
        "parent_id": case.parent_record["parent_id"], "input_kind": "simulated_genotypes_for_future_Relate",
        "cleaned_oracle_tree_is_not_inferred_primary_input": True,
        "sample_manifest": "sample_manifest.json", "sites": observed.num_sites,
        "samples": 280, "diploids": 140, "query_haplotypes": 120, "pure_reference_haplotypes": 96,
        "census_flags_or_populations_or_migrations_present": False, "scoring_bin_bp": 100,
        "coordinate_convention": "synthetic zero-based[0,2000000),VCF+1; no empirical-map offset",
        "Relate_executed": False,
    })
    files = {str(p.relative_to(destination)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(destination.rglob("*")) if p.is_file()}
    paths = case.truth_intervals["native_paths_1based"]
    return {**case.parent_record, "directory": str(destination.relative_to(ROOT)),
            "files": files, "status": "simulated_observations_and_external_truth_prepared",
            "truth_intervals": len(paths), "truth_switches_per_query": (paths[1:] != paths[:-1]).sum(axis=0).tolist(),
            "cleaned_oracle_trees": observed.num_trees, "sites": observed.num_sites,
            "Relate_or_biological_model_training_completed": False}


def prepare_pilot(config: dict[str, Any], *, config_sha256: str) -> dict[str, Any]:
    validate_config(config)
    output = project_path(config["output_directory"])
    output.mkdir(exist_ok=False)
    manifest = {"kind": CONFIG_KIND, "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_sha256, "status": "running", "replicates": [],
                "split_counts": {"train": 2, "val": 2, "test": 0},
                "observations": "simulated phased genotypes; primary inferred graphs pending Relate",
                "truth_is_external": True}
    current = None
    try:
        for parent in config["parent_registry"]:
            current = parent["parent_id"]
            case = simulate_parent(config, parent)
            record = save_parent(case, output / current)
            manifest["replicates"].append(record)
            print(json.dumps({"parent_id": current, "status": record["status"], "sites": record["sites"],
                              "truth_intervals": record["truth_intervals"]}), flush=True)
        manifest["status"] = "four_development_parents_prepared_Relate_and_models_pending"
    except Exception as error:
        manifest.update(status="failed_partial_output_preserved", failed_parent=current,
                        failure_type=type(error).__name__, failure_message=str(error))
        _json_new(output / "manifest.json", manifest)
        raise
    _json_new(output / "manifest.json", manifest)
    return manifest


def narrow_counterfactual_check(config: dict[str, Any]) -> dict[str, Any]:
    """One handmade280-leaf check. Does not call sim_ancestry/sim_mutations."""
    template = config["sample_template"]
    tables = tskit.TableCollection(100)
    tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
    for name in POPULATIONS:
        tables.populations.add_row(metadata={"name": name})
    for row in template:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=row["age_generations"],
                             population=POPULATIONS.index(row["sampling_population"]))
    root = tables.nodes.add_row(time=1000, population=10)
    first_census = None
    for left, right in ((0, 50), (50, 100)):
        for row in template:
            leaf = row["sample_index"]
            if row["role"] == "query":
                node = tables.nodes.add_row(time=350, flags=msprime.NODE_IS_CEN_EVENT, population=8)
                if first_census is None:
                    first_census = node
                tables.edges.add_row(left, right, node, leaf)
                tables.edges.add_row(left, right, root, node)
            else:
                tables.edges.add_row(left, right, root, leaf)
    site = tables.sites.add_row(position=10, ancestral_state="A")
    tables.mutations.add_row(site=site, node=0, derived_state="G")
    tables.sort()
    truth = tables.tree_sequence()
    original = external_truth_intervals(truth, template)
    changed_tables = truth.dump_tables()
    p = changed_tables.nodes.population.copy()
    p[first_census] = 7
    changed_tables.nodes.population = p
    changed = changed_tables.tree_sequence()
    counter = external_truth_intervals(changed, template)
    if original["native_paths_1based"].shape != (2, 120) or counter["native_paths_1based"][0, 0] != 2:
        raise AssertionError("handmade native shape/path counterfactual failed")
    clean, _ = strip_truth_annotations(truth)
    changed_clean, _ = strip_truth_annotations(changed)
    clean.tables.assert_equals(changed_clean.tables)
    if clean.num_trees != 1 or not np.array_equal(truth.genotype_matrix(), clean.genotype_matrix()):
        raise AssertionError("census-only segmentation/genotypes changed")
    if original["native_paths_1based"][0, 0] != 1 or counter["source_classes_0based"][0, 0] != 1:
        raise AssertionError("C4collapse counterfactual failed")
    return {"handmade_leaves": 280, "queries": 120, "truth_shape": [2, 120],
            "cleaned_trees": 1, "hidden_ancestry_changed_observation_unchanged": True,
            "simulation_executed": False}

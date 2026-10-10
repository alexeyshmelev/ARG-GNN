"""Pinned four-source ancient-European oracle benchmark, separate from toy pulses.

Census ancestry is an external target. Census-only topology, population IDs and
metadata are removed before constructing observed GraphSequence inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import msprime
import numpy as np
import stdpopsim
import torch
import tskit

from .simulate import (
    SPLIT_CODES, SimulatedDataset, _project_path, nearest_tmrca_probabilities,
    pack_tree_sequence, save_replicate,
)

SOURCE_NAMES = ("ANA", "CHG", "WHG", "EHG")
MODEL_ID = "AncientEurope_4A21"
MODEL_VERSION = "0.3.0"
MODEL_SOURCE = "https://github.com/popsim-consortium/stdpopsim/blob/0.3.0/stdpopsim/catalog/HomSap/demographic_models.py"


@dataclass(frozen=True)
class AncientSimulationConfig:
    sequence_length: float = 20_000.0
    recombination_rate: float = 1e-8
    mutation_rate: float = 1.25e-8
    reference_diploids_per_source: int = 3
    query_diploids: int = 4
    reference_times: tuple[float, ...] = (260., 300., 250., 250.)
    census_time: float = 350.
    grid_points: int = 128
    landmarks_per_source: int = 3
    max_graph_nodes: int = 500_000
    required_stdpopsim_version: str = MODEL_VERSION

    @property
    def num_classes(self) -> int:
        return 4

    def validate(self) -> None:
        if self.required_stdpopsim_version != MODEL_VERSION:
            raise ValueError("this benchmark is explicitly pinned to stdpopsim 0.3.0")
        if not np.isfinite(self.census_time) or not 200 < self.census_time < 600:
            raise ValueError("census must lie strictly inside the common source separation era (200,600)")
        if len(self.reference_times) != 4:
            raise ValueError("provide reference times in ANA,CHG,WHG,EHG order")
        for name, time, retirement in zip(SOURCE_NAMES, self.reference_times, (800., 800., 600., 600.)):
            if not np.isfinite(time) or not 0 < time < min(retirement, self.census_time):
                raise ValueError(f"invalid ancient reference time for {name}; require positive active-population age below census")
        if not np.isfinite(self.sequence_length) or self.sequence_length <= 0:
            raise ValueError("sequence_length must be finite and positive")
        if any(not np.isfinite(rate) or rate < 0 for rate in (self.recombination_rate, self.mutation_rate)):
            raise ValueError("rates must be finite and nonnegative")
        for name in ("reference_diploids_per_source", "query_diploids", "grid_points", "landmarks_per_source", "max_graph_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.landmarks_per_source > 2 * self.reference_diploids_per_source:
            raise ValueError("requested more landmarks than reference haplotypes")


@dataclass
class AncientDataset(SimulatedDataset):
    observed_tree_sequence: Any | None = None


def ancient_replicate_seeds(master_seed: int, split: str, replicate: int) -> dict[str, int]:
    if split not in SPLIT_CODES or master_seed < 0 or replicate < 0:
        raise ValueError("invalid split or seed")
    values = np.random.SeedSequence([master_seed, 0xAE4A21, SPLIT_CODES[split], replicate]).generate_state(2)
    return {key: int(value) % (2**32 - 1) + 1 for key, value in zip(("ancestry", "mutations"), values)}


def pinned_demography(config: AncientSimulationConfig) -> tuple[msprime.Demography, dict[str, Any]]:
    config.validate()
    if stdpopsim.__version__ != MODEL_VERSION:
        raise ValueError(f"expected stdpopsim {MODEL_VERSION}, found {stdpopsim.__version__}")
    model = stdpopsim.get_species("HomSap").get_demographic_model(MODEL_ID)
    model_spec = model.model.asdict()
    fingerprint = hashlib.sha256(json.dumps(model_spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    demography = model.model.copy()
    demography.add_census(time=config.census_time)
    demography.sort_events()
    provenance = {
        "species": "HomSap", "model_id": MODEL_ID, "stdpopsim_version": MODEL_VERSION,
        "model_source": MODEL_SOURCE, "model_demography_sha256": fingerprint,
        "unmodified_model_demography": model_spec,
        "catalog_mutation_rate": model.mutation_rate, "generation_time_years": model.generation_time,
        "citations": [citation.doi for citation in model.citations],
        "dependency_license": "GNU GPLv3+ (installed package metadata)",
        "augmentation": {"census_time": config.census_time, "population_parameters_modified": False},
    }
    return demography, provenance


def census_ancestry_at_positions(
    ts: tskit.TreeSequence, query_node_ids: Sequence[int], positions_bp: Sequence[float],
    *, source_population_to_class: Mapping[int, int], census_time: float,
) -> np.ndarray:
    positions = np.asarray(positions_bp, dtype=np.float64)
    if positions.ndim != 1 or np.any(~np.isfinite(positions)) or np.any(positions < 0) or np.any(positions >= ts.sequence_length):
        raise ValueError("census coordinates must lie in [0,sequence_length)")
    queries = [int(q) for q in query_node_ids]
    samples = set(int(u) for u in ts.samples())
    if not queries or len(queries) != len(set(queries)) or not set(queries).issubset(samples):
        raise ValueError("queries must be unique sample nodes")
    if any(ts.node(q).time >= census_time for q in queries):
        raise ValueError("query sampling dates must be younger than census")
    labels = np.full((len(positions), len(queries)), -1, dtype=np.int64)
    for p, coordinate in enumerate(positions):
        tree = ts.at(float(coordinate))
        for q, query in enumerate(queries):
            u = query
            while u != tskit.NULL and not (ts.node(u).flags & msprime.NODE_IS_CEN_EVENT):
                u = tree.parent(u)
            if u == tskit.NULL or not np.isclose(tree.time(u), census_time, rtol=0, atol=1e-9):
                raise ValueError(f"missing valid census ancestor for leaf {query} at {coordinate}")
            population = ts.node(u).population
            if population not in source_population_to_class:
                raise ValueError("census lineage is outside the four source populations")
            labels[p, q] = source_population_to_class[population]
    return labels


def remove_census_and_annotations(ts: tskit.TreeSequence) -> tuple[tskit.TreeSequence, np.ndarray]:
    """Collapse unary census nodes, then remove all truth annotations.

    Simplification may merge intervals distinguished only by census node IDs.
    Keeping those artificial boundaries would itself reveal truth segmentation.
    All sampled genotypes and sample-pair TMRCAs are preserved by simplification.
    """
    samples = list(ts.samples())
    observed, node_map = ts.simplify(
        samples=samples, map_nodes=True, keep_unary=False, filter_sites=False,
        filter_populations=False, filter_individuals=False, record_provenance=False,
    )
    if np.any(observed.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT):
        raise ValueError("a branching census node survived simplification; unsupported census artifact")
    if any(node_map[int(u)] == tskit.NULL for u in samples):
        raise ValueError("sample identity was lost when removing census nodes")
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
    if np.any(result.tables.nodes.population != tskit.NULL) or np.any(result.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT):
        raise ValueError("truth annotation survived observation construction")
    return result, node_map


def simulate_ancient_replicate(
    config: AncientSimulationConfig | None = None, *,
    master_seed: int = 31, split: str = "train", replicate: int = 0,
) -> AncientDataset:
    config = config or AncientSimulationConfig()
    demography, provenance = pinned_demography(config)
    seeds = ancient_replicate_seeds(master_seed, split, replicate)
    sample_sets = [msprime.SampleSet(config.reference_diploids_per_source, population=name, time=time, ploidy=2)
                   for name, time in zip(SOURCE_NAMES, config.reference_times)]
    # Explicit time 0 is essential: catalog Bronze default is an ancient sample.
    sample_sets.append(msprime.SampleSet(config.query_diploids, population="Bronze", time=0, ploidy=2))
    ancestry = msprime.sim_ancestry(
        samples=sample_sets, demography=demography, sequence_length=config.sequence_length,
        recombination_rate=config.recombination_rate, model="hudson", random_seed=seeds["ancestry"],
        record_migrations=False,
    )
    truth = msprime.sim_mutations(ancestry, rate=config.mutation_rate, random_seed=seeds["mutations"])
    source_to_class = {demography[name].id: i for i, name in enumerate(SOURCE_NAMES)}
    raw_references = {int(u): source_to_class[truth.node(u).population] for u in truth.samples()
                      if truth.node(u).population in source_to_class}
    raw_queries = [int(u) for u in truth.samples() if truth.node(u).population == demography["Bronze"].id]
    normalized_positions = (np.arange(config.grid_points, dtype=np.float64) + .5) / config.grid_points
    labels = census_ancestry_at_positions(
        truth, raw_queries, normalized_positions * truth.sequence_length,
        source_population_to_class=source_to_class, census_time=config.census_time,
    )
    # Full-resolution truth uses census-bearing genealogy, never cleaned model intervals.
    boundaries = np.asarray(list(truth.breakpoints()), dtype=np.float64)
    full_labels = census_ancestry_at_positions(
        truth, raw_queries, (boundaries[:-1] + boundaries[1:]) / 2,
        source_population_to_class=source_to_class, census_time=config.census_time,
    )
    true_switches = (full_labels[1:] != full_labels[:-1]).sum(axis=0).astype(int).tolist()
    observed, node_map = remove_census_and_annotations(truth)
    references = {int(node_map[u]): class_id for u, class_id in raw_references.items()}
    queries = [int(node_map[u]) for u in raw_queries]
    sequence, landmarks = pack_tree_sequence(
        observed, references, queries, normalized_positions, num_classes=4,
        landmarks_per_source=config.landmarks_per_source, max_graph_nodes=config.max_graph_nodes,
    )
    baseline = nearest_tmrca_probabilities(observed, references, queries,
                                           normalized_positions * observed.sequence_length, 4)
    sample_manifest = []
    haplotypes_seen: dict[int, int] = {}
    for raw_id in truth.samples():
        raw_id = int(raw_id)
        node = truth.node(raw_id)
        hap = haplotypes_seen.get(node.individual, 0)
        haplotypes_seen[node.individual] = hap + 1
        class_id = raw_references.get(raw_id)
        sample_manifest.append({
            "raw_node_id": raw_id, "observed_node_id": int(node_map[raw_id]),
            "sample_id": f"anc_{split}_r{replicate:03d}_i{node.individual:04d}", "haplotype": hap,
            "age_generations": node.time, "cohort_role": "query" if raw_id in raw_queries else "reference",
            "source_name": None if class_id is None else SOURCE_NAMES[class_id], "reference_class": class_id,
        })
    metadata = {
        "schema_version": 1, "benchmark": "ancient-europe-stdpopsim-0.3.0-oracle",
        "input_kind": "true_simulated_genealogy_oracle_census_removed", "ancestry_inference": "none",
        "label_method": "source_population_of_census_ancestor_at_fixed_coordinate",
        "split": split, "replicate": replicate, "master_seed": master_seed, "seeds": seeds,
        "config": asdict(config), "num_classes": 4, "class_names": list(SOURCE_NAMES),
        "demographic_provenance": provenance,
        "feature_names": [*[f"reference_{name}" for name in SOURCE_NAMES], "is_query_leaf", "is_leaf", "log_time", "log_parent_branch"],
        "feature_time_scale": 10_000., "reference_classes": {str(u): c for u, c in references.items()},
        "landmark_node_ids": landmarks, "samples": sample_manifest,
        "num_trees": observed.num_trees, "num_truth_trees": truth.num_trees, "num_sites": truth.num_sites,
        "num_raw_census_nodes": int(np.count_nonzero(truth.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT)),
        "num_observed_census_nodes": 0, "internal_population_annotations_present": False,
        "packed_nodes": int(sequence.x.shape[0]),
        "true_switch_counts_per_query": true_switches, "total_true_switches": int(sum(true_switches)),
        "grid_switch_counts_per_query": (labels[1:] != labels[:-1]).sum(axis=0).astype(int).tolist(),
        "grid_class_counts": np.bincount(labels.ravel(), minlength=4).astype(int).tolist(),
        "versions": {"stdpopsim": stdpopsim.__version__, "msprime": msprime.__version__,
                     "tskit": tskit.__version__, "torch": torch.__version__, "numpy": np.__version__},
    }
    return AncientDataset(sequence, torch.from_numpy(labels), torch.from_numpy(baseline), truth, metadata, observed)


def save_ancient_replicate(dataset: AncientDataset, directory: str | Path) -> dict[str, Any]:
    directory = _project_path(directory)
    record = save_replicate(dataset, directory)
    if dataset.observed_tree_sequence is not None:
        raw_path = directory / "observed.trees"
        dataset.observed_tree_sequence.dump(raw_path)
        with raw_path.open("rb") as source, gzip.open(directory / "observed.trees.gz", "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        raw_path.unlink()
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["files"] = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file() and p.name != "metadata.json"}
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    record["bytes"] = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    return record


def generate_ancient_dataset(
    directory: str | Path, config: AncientSimulationConfig | None = None, *,
    master_seed: int = 31, train_replicates: int = 3, val_replicates: int = 1, test_replicates: int = 2,
) -> dict[str, Any]:
    directory = _project_path(directory)
    config = config or AncientSimulationConfig()
    config.validate()
    counts = {"train": train_replicates, "val": val_replicates, "test": test_replicates}
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts.values()):
        raise ValueError("each independent replicate split must be nonempty")
    manifest = {"schema_version": 1, "benchmark": "ancient-europe-stdpopsim-0.3.0-oracle",
                "master_seed": master_seed, "config": asdict(config), "replicates": []}
    for split, count in counts.items():
        for replicate in range(count):
            data = simulate_ancient_replicate(config, master_seed=master_seed, split=split, replicate=replicate)
            record = save_ancient_replicate(data, directory / split / f"rep{replicate:03d}")
            manifest["replicates"].append(record)
            print(json.dumps({"split": split, "replicate": replicate, "trees": data.metadata["num_trees"],
                              "packed_nodes": data.metadata["packed_nodes"], "total_true_switches": data.metadata["total_true_switches"],
                              "grid_switches": sum(data.metadata["grid_switch_counts_per_query"]), "class_counts": data.metadata["grid_class_counts"],
                              "bytes": record["bytes"]}), flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--length", type=float, default=20_000.)
    parser.add_argument("--grid-points", type=int, default=128)
    parser.add_argument("--reference-diploids", type=int, default=3)
    parser.add_argument("--query-diploids", type=int, default=4)
    parser.add_argument("--train-replicates", type=int, default=3)
    parser.add_argument("--val-replicates", type=int, default=1)
    parser.add_argument("--test-replicates", type=int, default=2)
    args = parser.parse_args()
    config = AncientSimulationConfig(sequence_length=args.length, grid_points=args.grid_points,
                                     reference_diploids_per_source=args.reference_diploids, query_diploids=args.query_diploids,
                                     landmarks_per_source=min(3, 2 * args.reference_diploids))
    generate_ancient_dataset(args.output, config, master_seed=args.seed,
                            train_replicates=args.train_replicates, val_replicates=args.val_replicates,
                            test_replicates=args.test_replicates)


if __name__ == "__main__":
    main()

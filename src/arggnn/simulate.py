"""Small neutral pulse simulator for the explicitly oracle true-tree pilot.

Truth is resolved at a fixed grid from recorded admixture migration events, not
from majority ancestry in tree intervals. This pilot does not evaluate inferred
ARGs or establish applicability to ancient European ancestry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import msprime
import numpy as np
import torch
import tskit

from .data import GraphSequence, slice_genomic_window
from .resources import require_storage

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
FEATURE_TIME_SCALE = 10_000.0


@dataclass(frozen=True)
class SimulationConfig:
    sequence_length: float = 100_000.0
    recombination_rate: float = 1e-8
    mutation_rate: float = 1.25e-8
    population_size: float = 5_000.0
    split_time: float = 3_000.0
    admixture_time: float = 50.0
    proportions: tuple[float, ...] = (0.34, 0.33, 0.33)
    reference_diploids_per_source: int = 8
    query_diploids: int = 8
    grid_points: int = 128
    landmarks_per_source: int = 4
    max_graph_nodes: int = 1_000_000

    @property
    def num_classes(self) -> int:
        return len(self.proportions)

    def validate(self) -> None:
        if self.num_classes != 3:
            raise ValueError("the pilot requires exactly three source populations")
        if not all(np.isfinite(p) and p >= 0 for p in self.proportions) or not np.isclose(sum(self.proportions), 1):
            raise ValueError("source proportions must be nonnegative and sum to one")
        if not 0 < self.admixture_time < self.split_time:
            raise ValueError("require 0 < admixture_time < split_time")
        if not all(np.isfinite(v) and v > 0 for v in (self.sequence_length, self.population_size, self.split_time, self.admixture_time)):
            raise ValueError("length, size, and times must be finite and positive")
        if not all(np.isfinite(v) and v >= 0 for v in (self.recombination_rate, self.mutation_rate)):
            raise ValueError("rates must be finite and nonnegative")
        for name in ("reference_diploids_per_source", "query_diploids", "grid_points", "landmarks_per_source", "max_graph_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.landmarks_per_source > 2 * self.reference_diploids_per_source:
            raise ValueError("more landmarks requested than reference haplotypes")


@dataclass
class SimulatedDataset:
    sequence: GraphSequence
    labels: torch.Tensor                  # [P,Q], never part of sequence.x
    baseline_probabilities: torch.Tensor  # [P,Q,K], external nearest-TMRCA baseline
    tree_sequence: Any | None
    metadata: dict[str, Any]


def replicate_seeds(master_seed: int, split: str, replicate: int) -> dict[str, int]:
    if split not in SPLIT_CODES or replicate < 0 or master_seed < 0:
        raise ValueError("invalid master_seed, split, or replicate")
    states = np.random.SeedSequence([master_seed, SPLIT_CODES[split], replicate]).generate_state(2)
    return {name: int(value) % (2**32 - 1) + 1 for name, value in zip(("ancestry", "mutations"), states)}


def build_demography(config: SimulationConfig) -> msprime.Demography:
    config.validate()
    demography = msprime.Demography()
    for name in [*[f"SRC{i}" for i in range(config.num_classes)], "ADM", "ROOT"]:
        demography.add_population(name=name, initial_size=config.population_size)
    demography.add_admixture(
        time=config.admixture_time, derived="ADM",
        ancestral=[f"SRC{i}" for i in range(config.num_classes)],
        proportions=list(config.proportions),
    )
    demography.add_population_split(
        time=config.split_time,
        derived=[f"SRC{i}" for i in range(config.num_classes)], ancestral="ROOT",
    )
    demography.sort_events()
    return demography


def migration_ancestry_at_positions(
    ts: tskit.TreeSequence,
    query_node_ids: Sequence[int],
    positions: Sequence[float],
    *,
    admixed_population: int,
    source_population_to_class: Mapping[int, int],
    pulse_time: float,
) -> np.ndarray:
    """Resolve exactly one pulse migration ancestor for each coordinate/query.

    Coordinates are physical bp in [0,L). A migration record names the lineage
    node moved at the pulse and its genomic interval. Descendant membership in
    the corresponding true tree identifies which query inherited that lineage.
    All matching source transitions must agree and exactly one must exist.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 1 or np.any(~np.isfinite(pos)) or np.any(pos < 0) or np.any(pos >= ts.sequence_length):
        raise ValueError("truth coordinates must lie in [0,sequence_length)")
    queries = [int(q) for q in query_node_ids]
    if len(queries) != len(set(queries)) or not queries or any(q not in ts.samples() for q in queries):
        raise ValueError("query IDs must be unique sample leaves")
    records = [m for m in ts.migrations()
               if m.source == admixed_population and m.dest in source_population_to_class
               and np.isclose(m.time, pulse_time, rtol=0, atol=1e-9)]
    if not records:
        raise ValueError("no recorded pulse migrations; record_migrations=True is required")
    labels = np.full((len(pos), len(queries)), -1, dtype=np.int64)
    for p, coordinate in enumerate(pos):
        tree = ts.at(float(coordinate))
        active = [m for m in records if m.left <= coordinate < m.right]
        for q_index, query in enumerate(queries):
            matches = [m for m in active if tree.is_descendant(query, m.node)]
            if len(matches) != 1:
                raise ValueError(f"expected one pulse migration at {coordinate} for leaf {query}, got {len(matches)}")
            labels[p, q_index] = source_population_to_class[matches[0].dest]
    return labels


def true_ancestry_switch_counts(
    ts: tskit.TreeSequence, query_node_ids: Sequence[int], *,
    admixed_population: int, source_population_to_class: Mapping[int, int], pulse_time: float,
) -> list[int]:
    """Count ancestry transitions at all recorded pulse-migration boundaries."""
    boundaries = sorted({0.0, float(ts.sequence_length), *ts.breakpoints(), *[
        coordinate for migration in ts.migrations()
        if migration.source == admixed_population and migration.dest in source_population_to_class
        and np.isclose(migration.time, pulse_time, rtol=0, atol=1e-9)
        for coordinate in (migration.left, migration.right)
    ]})
    midpoints = (np.asarray(boundaries[:-1]) + np.asarray(boundaries[1:])) / 2
    labels = migration_ancestry_at_positions(
        ts, query_node_ids, midpoints, admixed_population=admixed_population,
        source_population_to_class=source_population_to_class, pulse_time=pulse_time,
    )
    return (labels[1:] != labels[:-1]).sum(axis=0).astype(int).tolist()


def pack_tree_sequence(
    ts: tskit.TreeSequence,
    reference_classes: Mapping[int, int],
    query_node_ids: Sequence[int],
    positions: Sequence[float],
    *,
    num_classes: int,
    landmarks_per_source: int = 4,
    max_graph_nodes: int = 1_000_000,
) -> tuple[GraphSequence, list[int]]:
    """Pack whole trees using reference labels, times, branches and landmarks.

    This function never reads internal node populations or migration records.
    Positions are normalized, and must be independent of tree segmentation.
    """
    queries = [int(q) for q in query_node_ids]
    samples = set(int(u) for u in ts.samples())
    if set(queries) & set(reference_classes):
        raise ValueError("query leaves cannot be labeled references")
    if not set(queries).issubset(samples) or not set(reference_classes).issubset(samples):
        raise ValueError("queries and references must be sample leaves")
    landmarks = []
    for population in range(num_classes):
        candidates = sorted(int(u) for u, c in reference_classes.items() if c == population)
        if len(candidates) < landmarks_per_source:
            raise ValueError("insufficient labeled reference landmarks")
        # Evenly spaced extant references, no ancestry-truth-dependent selection.
        landmarks.extend(candidates[i] for i in np.linspace(0, len(candidates) - 1, landmarks_per_source, dtype=int))
    xs, clades, node_graph, node_ids, edges, intervals = [], [], [], [], [], []
    n_packed = 0
    norm = np.log1p(FEATURE_TIME_SCALE)
    query_set = set(queries)
    for graph, tree in enumerate(ts.trees()):
        nodes = list(tree.nodes(order="postorder"))
        if n_packed + len(nodes) > max_graph_nodes:
            raise ValueError("marginal-tree expansion exceeds max_graph_nodes; reduce length/sample size")
        offset = n_packed
        index = {u: offset + i for i, u in enumerate(nodes)}
        for u in nodes:
            parent = tree.parent(u)
            is_leaf = u in samples
            feature = np.zeros(num_classes + 4, dtype=np.float32)
            if u in reference_classes:
                feature[reference_classes[u]] = 1.0
            feature[num_classes] = float(u in query_set)
            feature[num_classes + 1] = float(is_leaf)
            feature[num_classes + 2] = np.log1p(tree.time(u)) / norm
            branch = 0.0 if parent == tskit.NULL else tree.time(parent) - tree.time(u)
            feature[num_classes + 3] = np.log1p(branch) / norm
            xs.append(feature)
            clades.append([float(tree.is_descendant(landmark, u)) for landmark in landmarks])
            node_graph.append(graph)
            node_ids.append(int(u) if is_leaf else -1)
            if parent != tskit.NULL:
                edges.extend([(index[parent], index[u]), (index[u], index[parent])])
        intervals.append([tree.interval.left / ts.sequence_length, tree.interval.right / ts.sequence_length])
        n_packed += len(nodes)
    intervals_np = np.asarray(intervals, dtype=np.float64)
    normalized_positions = np.asarray(positions, dtype=np.float64)
    target_graph = np.searchsorted(intervals_np[:, 1], normalized_positions, side="right")
    sequence = GraphSequence(
        x=torch.from_numpy(np.asarray(xs, dtype=np.float32)),
        edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        node_graph=torch.tensor(node_graph, dtype=torch.long),
        node_ids=torch.tensor(node_ids, dtype=torch.long),
        clades=torch.tensor(clades, dtype=torch.float32),
        intervals=torch.from_numpy(intervals_np),
        query_node_ids=torch.tensor(queries, dtype=torch.long),
        positions=torch.from_numpy(normalized_positions),
        target_graph=torch.from_numpy(target_graph.astype(np.int64)),
    )
    sequence.validate()
    return sequence, landmarks


def nearest_tmrca_probabilities(
    ts: tskit.TreeSequence,
    reference_classes: Mapping[int, int],
    query_node_ids: Sequence[int],
    positions_bp: Sequence[float],
    num_classes: int,
) -> np.ndarray:
    """Deterministic nearest-reference voting, with all minimum-time ties shared."""
    if not reference_classes or set(reference_classes) & set(query_node_ids):
        raise ValueError("require labeled references distinct from queries")
    probabilities = np.zeros((len(positions_bp), len(query_node_ids), num_classes), dtype=np.float32)
    references = list(reference_classes)
    for p, position in enumerate(positions_bp):
        tree = ts.at(float(position))
        for qi, query in enumerate(query_node_ids):
            times = np.array([tree.tmrca(int(query), int(ref)) for ref in references])
            nearest = np.flatnonzero(np.isclose(times, times.min(), rtol=0, atol=1e-9))
            for r in nearest:
                probabilities[p, qi, reference_classes[references[r]]] += 1 / len(nearest)
    return probabilities


def simulate_replicate(
    config: SimulationConfig | None = None,
    *, master_seed: int = 13, split: str = "train", replicate: int = 0,
) -> SimulatedDataset:
    config = config or SimulationConfig()
    config.validate()
    seeds = replicate_seeds(master_seed, split, replicate)
    demography = build_demography(config)
    samples = [msprime.SampleSet(config.reference_diploids_per_source, population=f"SRC{i}", time=0, ploidy=2)
               for i in range(config.num_classes)]
    samples.append(msprime.SampleSet(config.query_diploids, population="ADM", time=0, ploidy=2))
    ancestry = msprime.sim_ancestry(
        samples=samples, demography=demography,
        sequence_length=config.sequence_length, recombination_rate=config.recombination_rate,
        random_seed=seeds["ancestry"], record_migrations=True,
        model="hudson",
    )
    ts = msprime.sim_mutations(ancestry, rate=config.mutation_rate, random_seed=seeds["mutations"])
    admixed_population = demography["ADM"].id
    source_to_class = {demography[f"SRC{i}"].id: i for i in range(config.num_classes)}
    # Sample populations are observed source-cohort labels, never internal-node features.
    reference_classes = {int(u): source_to_class[ts.node(u).population] for u in ts.samples()
                         if ts.node(u).population in source_to_class}
    queries = [int(u) for u in ts.samples() if ts.node(u).population == admixed_population]
    positions = (np.arange(config.grid_points, dtype=np.float64) + .5) / config.grid_points
    positions_bp = positions * ts.sequence_length
    labels = migration_ancestry_at_positions(
        ts, queries, positions_bp, admixed_population=admixed_population,
        source_population_to_class=source_to_class, pulse_time=config.admixture_time,
    )
    sequence, landmarks = pack_tree_sequence(
        ts, reference_classes, queries, positions, num_classes=config.num_classes,
        landmarks_per_source=config.landmarks_per_source, max_graph_nodes=config.max_graph_nodes,
    )
    baseline = nearest_tmrca_probabilities(ts, reference_classes, queries, positions_bp, config.num_classes)
    true_switches = true_ancestry_switch_counts(ts, queries, admixed_population=admixed_population,
                                              source_population_to_class=source_to_class, pulse_time=config.admixture_time)
    metadata = {
        "schema_version": 1, "benchmark": "three-source-neutral-pulse-oracle-pilot",
        "input_kind": "true_simulated_genealogy_oracle", "ancestry_inference": "none",
        "label_method": "recorded_pulse_migration_at_fixed_coordinate",
        "split": split, "replicate": replicate, "master_seed": master_seed, "seeds": seeds,
        "config": asdict(config), "num_classes": config.num_classes,
        "feature_names": [*[f"reference_SRC{i}" for i in range(config.num_classes)], "is_query_leaf", "is_leaf", "log_time", "log_parent_branch"],
        "feature_time_scale": FEATURE_TIME_SCALE,
        "landmark_node_ids": landmarks,
        "reference_classes": {str(u): int(c) for u, c in reference_classes.items()},
        "num_trees": ts.num_trees, "num_sites": ts.num_sites, "num_migrations": ts.num_migrations,
        "packed_nodes": int(sequence.x.shape[0]),
        "true_switch_counts_per_query": true_switches, "total_true_switches": int(sum(true_switches)),
        "grid_switch_counts_per_query": (labels[1:] != labels[:-1]).sum(axis=0).astype(int).tolist(),
        "expected_recombination_opportunities_per_haplotype_since_pulse": config.admixture_time * config.recombination_rate * config.sequence_length,
        "versions": {"msprime": msprime.__version__, "tskit": tskit.__version__, "torch": torch.__version__, "numpy": np.__version__},
    }
    return SimulatedDataset(sequence, torch.from_numpy(labels), torch.from_numpy(baseline), ts, metadata)



def split_fixed_windows(dataset: SimulatedDataset, windows: int) -> list[SimulatedDataset]:
    """Make fixed-coordinate in-memory tiles after the parent replicate split.

    The full truth genealogy remains an audit artifact on the parent. Every tile
    keeps global chromosome coordinates and all overlapping marginal boundaries.
    Genuine inferred coverage clips only edge tiles. Uncovered intervals and
    tiles without retained fixed scoring coordinates are never extrapolated.
    """
    if isinstance(windows, bool) or not isinstance(windows, int) or windows < 1:
        raise ValueError("windows must be a positive integer")
    bounds = np.linspace(0., 1., windows + 1)
    tiles = []
    coverage=dataset.sequence.window_bounds
    coverage_left,coverage_right=(0.,1.) if coverage is None else (float(coverage[0]),float(coverage[1]))
    for index, (global_left, global_right) in enumerate(zip(bounds[:-1], bounds[1:])):
        left,right=max(float(global_left),coverage_left),min(float(global_right),coverage_right)
        if left>=right or not bool(((dataset.sequence.positions>=left)&(dataset.sequence.positions<right)).any()):
            continue
        sequence, positions = slice_genomic_window(dataset.sequence, left, right)
        metadata = dict(dataset.metadata)
        metadata.update({"parent_replicate": dataset.metadata["replicate"], "window_index": index,
                         "window_bounds": [left, right], "global_window_bounds": [float(global_left),float(global_right)],
                         "packed_nodes": int(sequence.x.shape[0])})
        tiles.append(SimulatedDataset(sequence, dataset.labels[positions], dataset.baseline_probabilities[positions], None, metadata))
    if not tiles:
        raise ValueError("No fixed scoring coordinates inside observed tree coverage")
    return tiles


def _project_path(path: str | Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    result = Path(path).expanduser().resolve()
    if result != root and root not in result.parents:
        raise ValueError("all dataset writes must remain inside the ARG-GNN repository")
    return result


def save_replicate(dataset: SimulatedDataset, directory: str | Path) -> dict[str, Any]:
    directory = _project_path(directory)
    tensor_bytes=sum(t.numel()*t.element_size() for t in dataset.sequence.as_dict().values() if t is not None)
    tensor_bytes+=dataset.labels.numel()*dataset.labels.element_size()
    tensor_bytes+=dataset.baseline_probabilities.numel()*dataset.baseline_probabilities.element_size()
    tree_bytes=0 if dataset.tree_sequence is None else dataset.tree_sequence.nbytes
    require_storage(headroom_bytes=int(1.1*tensor_bytes+2.2*tree_bytes+10_000_000))
    directory.mkdir(parents=True, exist_ok=True)
    dataset.sequence.validate()
    if dataset.labels.shape != (dataset.sequence.positions.numel(), dataset.sequence.num_queries):
        raise ValueError("external labels do not align with fixed positions and queries")
    with gzip.open(directory / "tensors.pt.gz", "wb") as handle:
        torch.save({"sequence": dataset.sequence.as_dict(), "labels": dataset.labels,
                    "baseline_probabilities": dataset.baseline_probabilities}, handle)
    if dataset.tree_sequence is not None:
        raw = directory / "truth.trees"
        dataset.tree_sequence.dump(raw)
        with raw.open("rb") as source, gzip.open(directory / "truth.trees.gz", "wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        raw.unlink()
    metadata = dict(dataset.metadata)
    metadata["files"] = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file()}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"directory": str(directory), "metadata": str(directory / "metadata.json"),
            "tensor_file": str(directory / "tensors.pt.gz"), "split": metadata["split"],
            "replicate": metadata["replicate"], "seeds": metadata["seeds"],
            "bytes": sum(p.stat().st_size for p in directory.iterdir() if p.is_file())}


def load_replicate(directory: str | Path) -> SimulatedDataset:
    directory = Path(directory)
    with gzip.open(directory / "tensors.pt.gz", "rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    metadata = json.loads((directory / "metadata.json").read_text())
    # Training needs compact tensors only. The retained true trees are audit artifacts.
    return SimulatedDataset(GraphSequence.from_dict(payload["sequence"]), payload["labels"],
                            payload["baseline_probabilities"], None, metadata)


def generate_dataset(
    directory: str | Path, config: SimulationConfig | None = None,
    *, master_seed: int = 13, train_replicates: int = 3,
    val_replicates: int = 1, test_replicates: int = 2,
) -> dict[str, Any]:
    directory = _project_path(directory)
    config = config or SimulationConfig()
    config.validate()
    counts = {"train": train_replicates, "val": val_replicates, "test": test_replicates}
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts.values()):
        raise ValueError("each independent replicate split must be nonempty")
    manifest = {"schema_version": 1, "benchmark": "three-source-neutral-pulse-oracle-pilot",
                "master_seed": master_seed, "config": asdict(config), "replicates": []}
    for split, count in counts.items():
        for replicate in range(count):
            require_storage(headroom_bytes=1_000_000_000)
            dataset = simulate_replicate(config, master_seed=master_seed, split=split, replicate=replicate)
            record = save_replicate(dataset, directory / split / f"rep{replicate:03d}")
            manifest["replicates"].append(record)
            print(json.dumps({"split": split, "replicate": replicate, "trees": dataset.metadata["num_trees"], "packed_nodes": dataset.metadata["packed_nodes"], "total_true_switches": dataset.metadata["total_true_switches"], "grid_switches": sum(dataset.metadata["grid_switch_counts_per_query"]), "bytes": record["bytes"]}), flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--length", type=float, default=100_000)
    parser.add_argument("--grid-points", type=int, default=128)
    parser.add_argument("--reference-diploids", type=int, default=8)
    parser.add_argument("--query-diploids", type=int, default=8)
    parser.add_argument("--train-replicates", type=int, default=3)
    parser.add_argument("--val-replicates", type=int, default=1)
    parser.add_argument("--test-replicates", type=int, default=2)
    args = parser.parse_args()
    config = SimulationConfig(sequence_length=args.length, grid_points=args.grid_points,
                              reference_diploids_per_source=args.reference_diploids, query_diploids=args.query_diploids,
                              landmarks_per_source=min(4, 2 * args.reference_diploids))
    generate_dataset(args.output, config, master_seed=args.seed,
                     train_replicates=args.train_replicates, val_replicates=args.val_replicates,
                     test_replicates=args.test_replicates)


if __name__ == "__main__":
    main()

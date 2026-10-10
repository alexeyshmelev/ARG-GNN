"""Truth-coordinate and observation-separation tests for the oracle pilot."""
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch
import tskit

from arggnn.data import GraphSequence, slice_genomic_window
from arggnn.simulate import (
    SimulationConfig, load_replicate, migration_ancestry_at_positions,
    pack_tree_sequence, replicate_seeds, save_replicate, simulate_replicate,
    split_fixed_windows, true_ancestry_switch_counts,
)


def one_tree_two_ancestries():
    """Truth changes inside a single constant-topology marginal tree."""
    tables = tskit.TableCollection(100)
    for _ in range(5):
        tables.populations.add_row()
    for population in (0, 1, 2, 3):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=population)
    root = tables.nodes.add_row(time=100, population=4)
    for sample in range(4):
        tables.edges.add_row(0, 100, root, sample)
    tables.migrations.add_row(0, 50, 0, 0, 1, 50)
    tables.migrations.add_row(50, 100, 0, 0, 2, 50)
    tables.sort()
    return tables.tree_sequence()


def tiny_config(**kwargs):
    base = replace(SimulationConfig(), sequence_length=2000, recombination_rate=0,
                   reference_diploids_per_source=2, query_diploids=2,
                   grid_points=8, landmarks_per_source=2)
    return replace(base, **kwargs)


def test_labels_follow_migrations_inside_one_tree_and_half_open_boundary():
    ts = one_tree_two_ancestries()
    assert ts.num_trees == 1
    labels = migration_ancestry_at_positions(
        ts, [0], [0, 49.999, 50, 99.999], admixed_population=0,
        source_population_to_class={1: 0, 2: 1, 3: 2}, pulse_time=50,
    )
    np.testing.assert_array_equal(labels[:, 0], [0, 0, 1, 1])
    assert true_ancestry_switch_counts(ts, [0], admixed_population=0,
                                     source_population_to_class={1: 0, 2: 1, 3: 2}, pulse_time=50) == [1]
    with pytest.raises(ValueError, match="coordinates"):
        migration_ancestry_at_positions(ts, [0], [100], admixed_population=0,
                                       source_population_to_class={1: 0}, pulse_time=50)


def test_missing_or_duplicate_truth_migration_is_an_error():
    ts = one_tree_two_ancestries()
    tables = ts.dump_tables()
    tables.migrations.clear()
    with pytest.raises(ValueError, match="record_migrations"):
        migration_ancestry_at_positions(tables.tree_sequence(), [0], [25], admixed_population=0,
                                       source_population_to_class={1: 0}, pulse_time=50)
    tables = ts.dump_tables()
    tables.migrations.add_row(0, 50, 0, 0, 2, 50)
    tables.sort()
    with pytest.raises(ValueError, match="got 2"):
        migration_ancestry_at_positions(tables.tree_sequence(), [0], [25], admixed_population=0,
                                       source_population_to_class={1: 0, 2: 1}, pulse_time=50)


def test_hidden_internal_population_and_migration_labels_do_not_enter_inputs():
    ts = one_tree_two_ancestries()
    kwargs = dict(reference_classes={1: 0, 2: 1, 3: 2}, query_node_ids=[0],
                  positions=[.25, .75], num_classes=3, landmarks_per_source=1)
    original, _ = pack_tree_sequence(ts, **kwargs)
    tables = ts.dump_tables()
    population = tables.nodes.population.copy()
    population[4] = 2
    tables.nodes.population = population
    tables.migrations.clear()
    changed, _ = pack_tree_sequence(tables.tree_sequence(), **kwargs)
    for field in original.as_dict():
        if getattr(original, field) is not None:
            assert torch.equal(getattr(original, field), getattr(changed, field))
    internal = original.node_ids == -1
    query = original.node_ids == 0
    assert torch.count_nonzero(original.x[internal | query, :3]) == 0
    assert torch.equal(original.clades[internal][0], torch.ones(3))
    assert not hasattr(original, "labels")


def test_one_source_pulse_and_no_recombination_have_known_labels():
    dataset = simulate_replicate(tiny_config(proportions=(1., 0., 0.)), master_seed=91)
    assert torch.equal(dataset.labels, torch.zeros_like(dataset.labels))
    assert dataset.metadata["total_true_switches"] == 0
    assert dataset.tree_sequence.num_migrations > 0
    assert torch.allclose(dataset.baseline_probabilities.sum(-1), torch.ones_like(dataset.labels, dtype=torch.float32))
    dataset.sequence.validate()


def test_replicates_are_deterministic_and_split_seed_streams_are_disjoint():
    seeds = [replicate_seeds(13, split, index) for split in ("train", "val", "test") for index in range(5)]
    all_seeds = [seed for stream in seeds for seed in stream.values()]
    assert len(all_seeds) == len(set(all_seeds))
    first = simulate_replicate(tiny_config(), master_seed=17, split="train", replicate=1)
    second = simulate_replicate(tiny_config(), master_seed=17, split="train", replicate=1)
    for key, value in first.sequence.as_dict().items():
        if value is not None:
            assert torch.equal(value, second.sequence.as_dict()[key])
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.baseline_probabilities, second.baseline_probabilities)
    assert first.tree_sequence.tables.equals(second.tree_sequence.tables, ignore_provenance=True)


def test_windows_keep_global_coordinates_labels_and_every_tree_boundary():
    dataset = simulate_replicate(tiny_config(), master_seed=23)
    tiles = split_fixed_windows(dataset, 2)
    assert torch.equal(torch.cat([t.labels for t in tiles]), dataset.labels)
    for index, tile in enumerate(tiles):
        torch.testing.assert_close(tile.sequence.window_bounds, torch.tensor([index / 2, (index + 1) / 2], dtype=torch.float64))
        assert tile.metadata["split"] == dataset.metadata["split"]
        assert tile.metadata["parent_replicate"] == dataset.metadata["replicate"]
        assert tile.tree_sequence is None
        tile.sequence.validate()
    sequence, _ = pack_tree_sequence(one_tree_two_ancestries(), {1: 0, 2: 1, 3: 2}, [0], [.25, .75], num_classes=3, landmarks_per_source=1)
    clipped, indexes = slice_genomic_window(sequence, .5, 1.)
    assert indexes.tolist() == [1]
    assert clipped.positions.tolist() == [.75]
    assert clipped.intervals.tolist() == [[.5, 1.]]


def test_storage_roundtrip_and_path_isolation():
    dataset = simulate_replicate(tiny_config(), master_seed=41)
    root = Path(__file__).resolve().parents[1]
    scratch = root / ".runtime" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="simulate-test-") as directory:
        record = save_replicate(dataset, directory)
        loaded = load_replicate(directory)
        assert record["bytes"] > 0
        assert (Path(directory) / "truth.trees.gz").is_file()
        assert not (Path(directory) / "truth.trees").exists()
        assert torch.equal(loaded.labels, dataset.labels)
        loaded.sequence.validate()
    with pytest.raises(ValueError, match="inside"):
        save_replicate(dataset, root.parent / "outside-test")


def test_graph_contract_rejects_cross_graph_edges_and_wrong_target():
    dataset = simulate_replicate(tiny_config(), master_seed=7)
    sequence = dataset.sequence
    wrong = GraphSequence.from_dict(sequence.as_dict())
    wrong.target_graph = torch.full_like(sequence.target_graph, sequence.num_graphs)
    with pytest.raises(ValueError, match="target_graph"):
        wrong.validate()
    wrong = GraphSequence.from_dict(sequence.as_dict())
    wrong.node_ids = sequence.node_ids.clone()
    wrong.node_ids[wrong.node_ids == int(sequence.query_node_ids[0])] = -1
    with pytest.raises(ValueError, match="absent"):
        wrong.validate()


def test_recombining_ordering_pilot_truth_and_windows():
    config = tiny_config(sequence_length=1_000_000, recombination_rate=1e-8, mutation_rate=0,
                         population_size=500, admixture_time=300, split_time=2000, grid_points=128)
    dataset = simulate_replicate(config, master_seed=13, split="test", replicate=0)
    assert dataset.metadata["total_true_switches"] > 0
    assert dataset.sequence.num_graphs > 1
    tiles = split_fixed_windows(dataset, 4)
    assert torch.equal(torch.cat([tile.labels for tile in tiles]), dataset.labels)
    for tile in tiles:
        left, right = tile.sequence.window_bounds.tolist()
        expected = dataset.sequence.intervals[(dataset.sequence.intervals[:, 1] > left) & (dataset.sequence.intervals[:, 0] < right)].clone()
        expected[:, 0].clamp_(min=left)
        expected[:, 1].clamp_(max=right)
        torch.testing.assert_close(tile.sequence.intervals, expected)
        tile.sequence.validate()

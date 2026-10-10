"""Scientific and observation-separation checks for the pinned European pilot."""
from dataclasses import replace
from pathlib import Path
import tempfile

import msprime
import numpy as np
import pytest
import torch
import tskit

from arggnn.ancient_simulate import (
    AncientSimulationConfig, SOURCE_NAMES, ancient_replicate_seeds,
    census_ancestry_at_positions, pinned_demography, remove_census_and_annotations,
    save_ancient_replicate, simulate_ancient_replicate,
)
from arggnn.simulate import load_replicate, pack_tree_sequence


def tiny_config(**kwargs):
    base = replace(AncientSimulationConfig(), sequence_length=2000., recombination_rate=0,
                   reference_diploids_per_source=1, query_diploids=2, grid_points=8, landmarks_per_source=1)
    return replace(base, **kwargs)


def census_only_boundary():
    """One query changes census ancestry without changing observed topology."""
    tables = tskit.TableCollection(100)
    for _ in range(5):
        tables.populations.add_row()
    for population in range(5):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=population)
    first = tables.nodes.add_row(flags=msprime.NODE_IS_CEN_EVENT, time=350, population=1)
    second = tables.nodes.add_row(flags=msprime.NODE_IS_CEN_EVENT, time=350, population=2)
    root = tables.nodes.add_row(time=2000, population=4)
    tables.edges.add_row(0, 50, first, 0)
    tables.edges.add_row(50, 100, second, 0)
    tables.edges.add_row(0, 50, root, first)
    tables.edges.add_row(50, 100, root, second)
    for sample in range(1, 5):
        tables.edges.add_row(0, 100, root, sample)
    tables.sort()
    return tables.tree_sequence()


def test_pinned_model_and_valid_reference_dates():
    config = tiny_config()
    demography, provenance = pinned_demography(config)
    assert provenance['stdpopsim_version'] == '0.3.0'
    assert provenance['model_demography_sha256'] == '37e53cbed70dcbbf4e67cbe33226b4ed58d140d2f812e48333bf260362be4d60'
    assert demography['ANA'].initial_size == 50_000
    assert demography['CHG'].initial_size == 10_000
    assert set(event.time for event in demography.events) == {140, 180, 200, 350, 600, 800, 1500}
    for census in (200, 600, 1500):
        with pytest.raises(ValueError, match='separation era'):
            pinned_demography(replace(config, census_time=census))
    with pytest.raises(ValueError, match='reference time'):
        pinned_demography(replace(config, reference_times=(260, 800, 250, 250)))
    with pytest.raises(ValueError, match='pinned'):
        pinned_demography(replace(config, required_stdpopsim_version='latest'))


def test_census_boundary_is_external_and_collapses_from_model_topology():
    truth = census_only_boundary()
    labels = census_ancestry_at_positions(truth, [0], [0, 49.999, 50, 99.999],
                                         source_population_to_class={1: 0, 2: 1, 3: 2, 4: 3}, census_time=350)
    assert labels[:, 0].tolist() == [0, 0, 1, 1]
    assert truth.num_trees == 2
    observed, node_map = remove_census_and_annotations(truth)
    assert observed.num_trees == 1
    assert np.all(observed.tables.nodes.population == tskit.NULL)
    assert observed.num_populations == 0
    assert observed.tables.nodes.metadata.size == 0
    assert not np.any(observed.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT)
    assert not np.any(observed.tables.nodes.time == 350)
    assert observed.tables.provenances.num_rows == 0
    assert np.all(node_map[truth.samples()] >= 0)
    sequence, _ = pack_tree_sequence(observed, {int(node_map[u]): u - 1 for u in range(1, 5)},
                                    [int(node_map[0])], [.25, .75], num_classes=4, landmarks_per_source=1)
    assert sequence.target_graph.tolist() == [0, 0]
    assert not hasattr(sequence, 'labels')


def test_missing_census_or_invalid_coordinate_is_rejected():
    truth = census_only_boundary()
    with pytest.raises(ValueError, match='coordinates'):
        census_ancestry_at_positions(truth, [0], [100], source_population_to_class={1: 0, 2: 1}, census_time=350)
    observed, node_map = remove_census_and_annotations(truth)
    with pytest.raises(ValueError, match='missing valid census'):
        census_ancestry_at_positions(observed, [int(node_map[0])], [25],
                                    source_population_to_class={1: 0, 2: 1}, census_time=350)
    with pytest.raises(ValueError, match='outside'):
        census_ancestry_at_positions(truth, [0], [75], source_population_to_class={1: 0}, census_time=350)


def test_simulated_dates_census_removal_genotypes_and_tmrca_preserved():
    data = simulate_ancient_replicate(tiny_config(), master_seed=83)
    raw, observed = data.tree_sequence, data.observed_tree_sequence
    assert np.any(raw.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT)
    assert not np.any(observed.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT)
    assert np.all(observed.tables.nodes.population == tskit.NULL)
    np.testing.assert_array_equal(raw.genotype_matrix(), observed.genotype_matrix())
    mapping = {row['raw_node_id']: row['observed_node_id'] for row in data.metadata['samples']}
    for position in data.sequence.positions.numpy() * raw.sequence_length:
        raw_tree, observed_tree = raw.at(float(position)), observed.at(float(position))
        for first in raw.samples():
            for second in raw.samples():
                if first != second:
                    assert raw_tree.tmrca(int(first), int(second)) == observed_tree.tmrca(mapping[int(first)], mapping[int(second)])
    times = {name: set() for name in SOURCE_NAMES}
    for row in data.metadata['samples']:
        if row['cohort_role'] == 'query':
            assert row['age_generations'] == 0
            assert row['source_name'] is None
            assert row['reference_class'] is None
        else:
            times[row['source_name']].add(row['age_generations'])
    assert times == {'ANA': {260.}, 'CHG': {300.}, 'WHG': {250.}, 'EHG': {250.}}
    assert data.sequence.x.shape[1] == 8
    query_or_internal = (data.sequence.node_ids == -1) | torch.isin(data.sequence.node_ids, data.sequence.query_node_ids)
    assert torch.count_nonzero(data.sequence.x[query_or_internal, :4]) == 0
    assert data.labels.shape == (8, 4)
    assert set(data.labels.flatten().tolist()).issubset(set(range(4)))
    torch.testing.assert_close(data.baseline_probabilities.sum(-1), torch.ones((8, 4)))
    data.sequence.validate()


def test_hidden_census_populations_do_not_affect_observed_features():
    raw = census_only_boundary()
    observed, node_map = remove_census_and_annotations(raw)
    tables = raw.dump_tables()
    populations = tables.nodes.population.copy()
    populations[5:] = 3
    tables.nodes.population = populations
    changed, changed_map = remove_census_and_annotations(tables.tree_sequence())
    np.testing.assert_array_equal(node_map, changed_map)
    for field in ('time', 'flags', 'population'):
        np.testing.assert_array_equal(getattr(observed.tables.nodes, field), getattr(changed.tables.nodes, field))
    assert observed.tables.edges.equals(changed.tables.edges)


def test_ancient_seed_namespace_and_determinism():
    streams = [ancient_replicate_seeds(31, split, r) for split in ('train', 'val', 'test') for r in range(4)]
    flattened = [seed for stream in streams for seed in stream.values()]
    assert len(flattened) == len(set(flattened))
    first = simulate_ancient_replicate(tiny_config(), master_seed=91, split='val')
    second = simulate_ancient_replicate(tiny_config(), master_seed=91, split='val')
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.sequence.x, second.sequence.x)
    assert first.tree_sequence.tables.equals(second.tree_sequence.tables, ignore_provenance=True)


def test_ancient_archives_retain_raw_truth_and_scrubbed_observation():
    data = simulate_ancient_replicate(tiny_config(), master_seed=61)
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.runtime' / 'tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix='ancient-test-') as directory:
        record = save_ancient_replicate(data, directory)
        assert (Path(directory) / 'truth.trees.gz').is_file()
        assert (Path(directory) / 'observed.trees.gz').is_file()
        assert not (Path(directory) / 'observed.trees').exists()
        loaded = load_replicate(directory)
        assert torch.equal(data.labels, loaded.labels)
        assert loaded.metadata['num_classes'] == 4
        assert record['bytes'] > 0
        loaded.sequence.validate()

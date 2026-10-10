"""Genotype export and exact paired inference-coordinate/sample alignment."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile

import numpy as np
import pytest
import torch
import tskit

from arggnn.ancient_simulate import AncientSimulationConfig, simulate_ancient_replicate
from arggnn.relate_pipeline import (
    TOOL_ROOT, RelateConfig, collect_observed_genotypes, constant_map_rows,
    paired_inferred_dataset, prepare_relate_inputs, read_relate_sample_order,
    relate_commands, verified_relate_installation,
)


def mutation_fixture(positions=(0, 40, 99)):
    config = replace(AncientSimulationConfig(), sequence_length=100., recombination_rate=0,
                     mutation_rate=0, reference_diploids_per_source=1,
                     query_diploids=2, grid_points=16, landmarks_per_source=1)
    parent = simulate_ancient_replicate(config, master_seed=161)
    tables = parent.observed_tree_sequence.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    nodes = list(parent.observed_tree_sequence.samples())
    for index, position in enumerate(positions):
        site = tables.sites.add_row(position, 'A')
        tables.mutations.add_row(site=site, node=int(nodes[index]), derived_state='G')
    parent.observed_tree_sequence = tables.tree_sequence()
    return parent


def converted_fixture(observed, order_nodes=None):
    if order_nodes is not None:
        observed = observed.simplify(samples=order_nodes, filter_sites=False, record_provenance=False)
    tables = observed.dump_tables()
    tables.sites.position = tables.sites.position + 1
    # Match the official converter's VCF-coordinate sites and midpoint edges.
    left = tables.edges.left.copy()
    left[left > 0] += 1
    tables.edges.left = left
    tables.edges.right = tables.edges.right + 1
    tables.sequence_length = observed.sequence_length + 1
    return tables.tree_sequence()


@pytest.fixture
def scratch():
    path = TOOL_ROOT / 'test_tmp'
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path, prefix='relate-') as directory:
        yield Path(directory)


def test_export_genotypes_vcf_haps_order_and_off_by_one(scratch):
    parent = mutation_fixture()
    original_names = list(dict.fromkeys(row['sample_id'] for row in parent.metadata['samples']))
    order = original_names[::-1]
    variants = collect_observed_genotypes(parent, parent.observed_tree_sequence, sample_order=order)
    prep = prepare_relate_inputs(parent, parent.observed_tree_sequence, scratch, sample_order=order, recombination_rate=1e-8)
    assert prep['site_positions0'] == [0, 40, 99]
    assert read_relate_sample_order(prep['paths']['sample']) == order
    lines = [line.split('\t') for line in Path(prep['paths']['vcf']).read_text().splitlines() if not line.startswith('##')]
    assert lines[0][9:] == order
    assert [int(row[1]) for row in lines[1:]] == [1, 41, 100]
    reconstructed = np.array([[int(allele) for pair in row[9:] for allele in pair.split('|')] for row in lines[1:]], dtype=np.int8)
    np.testing.assert_array_equal(reconstructed, variants.genotypes)
    haps = [line.split() for line in Path(prep['paths']['haps']).read_text().splitlines()]
    assert [int(row[2]) for row in haps] == [1, 41, 100]
    np.testing.assert_array_equal(np.array([[int(value) for value in row[5:]] for row in haps]), variants.genotypes)
    ages = np.loadtxt(prep['paths']['ages'])
    np.testing.assert_array_equal(ages, [row['age_generations'] for row in variants.haplotypes])
    assert not list(scratch.glob('*.fa'))
    assert prep['unobserved_ancestral_alleles'].startswith('not supplied')


def test_constant_map_preserves_original_zero_based_origin():
    rows = constant_map_rows(1_000_000, 1e-8)
    assert rows == [(1, 1., 0.), (1_000_001, 1., 1.)]
    # Original position 12345 has VCF coordinate 12346 and cumulative distance .012345 cM.
    coordinate = 12346
    fraction = (coordinate - rows[0][0]) / (rows[1][0] - rows[0][0])
    assert fraction * rows[1][2] == pytest.approx(.012345)


def test_noninteger_site_and_missing_manifest_rejected():
    parent = mutation_fixture()
    tables = parent.observed_tree_sequence.dump_tables()
    positions = tables.sites.position.copy()
    positions[1] = 40.5
    tables.sites.position = positions
    with pytest.raises(ValueError, match='silently rounded'):
        collect_observed_genotypes(parent, tables.tree_sequence())
    original = parent.metadata['samples']
    parent.metadata['samples'] = original[:-1]
    with pytest.raises(ValueError, match='both haplotypes'):
        collect_observed_genotypes(parent, parent.observed_tree_sequence)


def test_imported_ts_queries_align_by_diploid_and_haplotype_after_reorder(scratch):
    parent = mutation_fixture()
    prep = prepare_relate_inputs(parent, parent.observed_tree_sequence, scratch, recombination_rate=1e-8)
    row_by_key = {(row['sample_id'], row['haplotype']): row for row in prep['haplotypes']}
    inferred_order = prep['sample_order'][::-1]
    node_order = [row_by_key[(name, hap)]['observed_node_id'] for name in inferred_order for hap in (0, 1)]
    converted = converted_fixture(parent.observed_tree_sequence, node_order)
    result = paired_inferred_dataset(converted, parent, prep, inferred_sample_order=inferred_order)
    assert torch.equal(result.labels, parent.labels)
    original_query_keys = [(row['sample_id'], row['haplotype']) for row in parent.metadata['samples'] if row['observed_node_id'] in parent.sequence.query_node_ids.tolist()]
    result_key_by_node = {row['observed_node_id']: (row['sample_id'], row['haplotype']) for row in result.metadata['samples']}
    assert [result_key_by_node[int(u)] for u in result.sequence.query_node_ids] == original_query_keys
    assert result.metadata['coverage_physical_bp'] == [0., 100.]
    assert result.metadata['input_kind'] == 'inferred_genealogy'
    assert result.metadata['benchmark'] == 'ancient-europe-stdpopsim-0.3.0-relate-inferred'
    assert result.metadata['parent_benchmark'] == parent.metadata['benchmark']
    assert np.all(result.tree_sequence.tables.nodes.population == tskit.NULL)
    result.sequence.validate()


def test_variant_span_masks_tails_but_keeps_exact_parent_grid(scratch):
    parent = mutation_fixture((10, 90))
    prep = prepare_relate_inputs(parent, parent.observed_tree_sequence, scratch, recombination_rate=1e-8)
    result = paired_inferred_dataset(converted_fixture(parent.observed_tree_sequence), parent, prep)
    expected = torch.nonzero((parent.sequence.positions * 100 >= 10) & (parent.sequence.positions * 100 < 91)).flatten()
    assert torch.equal(result.sequence.positions, parent.sequence.positions[expected])
    assert torch.equal(result.labels, parent.labels[expected])
    assert result.sequence.window_bounds.tolist() == [.1, .91]
    assert result.metadata['physical_coverage_fraction'] == pytest.approx(.81)
    assert result.metadata['retained_parent_grid_indexes'] == expected.tolist()
    result.sequence.validate()


def test_wrong_offset_age_and_source_oracle_features_are_detected_or_ignored(scratch):
    parent = mutation_fixture()
    prep = prepare_relate_inputs(parent, parent.observed_tree_sequence, scratch, recombination_rate=1e-8)
    with pytest.raises(ValueError, match='coordinate mapping'):
        paired_inferred_dataset(parent.observed_tree_sequence, parent, prep)
    converted = converted_fixture(parent.observed_tree_sequence)
    good = paired_inferred_dataset(converted, parent, prep)
    parent.sequence.x.fill_(123.)
    again = paired_inferred_dataset(converted, parent, prep)
    assert torch.equal(good.sequence.x, again.sequence.x)
    assert torch.equal(good.sequence.edge_index, again.sequence.edge_index)
    tables = converted.dump_tables()
    ages = tables.nodes.time.copy()
    ages[0] = 0
    tables.nodes.time = ages
    with pytest.raises(ValueError, match='sample ages'):
        paired_inferred_dataset(tables.tree_sequence(), parent, prep)


def test_command_uses_haploid_size_ordered_ages_and_confined_paths(scratch):
    parent = mutation_fixture()
    prep = prepare_relate_inputs(parent, parent.observed_tree_sequence, scratch, recombination_rate=1e-8)
    installation = {'binaries': {name: {'path': str(TOOL_ROOT / 'bin' / name)} for name in ('Relate', 'RelateFileFormats')}}
    commands = relate_commands(prep, scratch / 'run', installation, RelateConfig(seed=71))
    assert commands[0][commands[0].index('--sample_ages') + 1] == prep['paths']['ages']
    assert commands[0][commands[0].index('-N') + 1] == '30000.0'
    assert commands[0][commands[0].index('--seed') + 1] == '71'
    assert commands[1][commands[1].index('--mode') + 1] == 'ConvertToTreeSequence'


def test_verified_official_binary_provenance_when_installed():
    if not (TOOL_ROOT / 'installation.json').exists():
        pytest.skip('official Relate not installed in this project')
    manifest = verified_relate_installation()
    assert manifest['version'] == '1.2.4'
    assert manifest['authorization']['human_message'] == 'I reviewed, use it'
    assert manifest['copyright_notice'] == 'Copyright © 2018, University of Oxford'


def test_native_vcf_conversion_agrees_with_independent_haps_export(scratch):
    import subprocess
    if not (TOOL_ROOT / 'installation.json').exists():
        pytest.skip('official Relate not installed')
    installation = verified_relate_installation()
    parent = mutation_fixture()
    prep = prepare_relate_inputs(parent, parent.observed_tree_sequence, scratch, recombination_rate=1e-8)
    output_haps, output_sample = scratch / 'converted.haps', scratch / 'converted.sample'
    result = subprocess.run([installation['binaries']['RelateFileFormats']['path'], '--mode', 'ConvertFromVcf',
                             '-i', str(scratch / 'input'), '--haps', str(output_haps), '--sample', str(output_sample)],
                            cwd=scratch, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [line.split() for line in output_haps.read_text().splitlines()] == [line.split() for line in Path(prep['paths']['haps']).read_text().splitlines()]
    assert read_relate_sample_order(output_sample) == prep['sample_order']


def test_successful_cpu_fixture_exact_targets_and_native_return_codes():
    from arggnn.relate_pipeline import load_ancient_parent
    root = TOOL_ROOT / 'fixtures' / 'ancient20kb_seed31'
    run = root / 'inference_parallel'
    if not (run / 'inferred.trees').is_file():
        pytest.skip('tiny own CPU fixture has not been run')
    parent, _ = load_ancient_parent(root / 'parent')
    prep = json.loads((root / 'inputs' / 'preparation.json').read_text())
    execution = json.loads((run / 'execution.json').read_text())
    assert all(step['returncode'] == 0 for step in execution['steps'])
    result = paired_inferred_dataset(tskit.load(run / 'inferred.trees'), parent, prep)
    indexes = torch.tensor(result.metadata['retained_parent_grid_indexes'], dtype=torch.long)
    assert torch.equal(result.labels, parent.labels[indexes])
    assert torch.equal(result.sequence.positions, parent.sequence.positions[indexes])
    assert result.metadata['split'] == parent.metadata['split']
    assert result.metadata['replicate'] == parent.metadata['replicate']
    assert len(result.sequence.query_node_ids) == len(parent.sequence.query_node_ids)
    assert result.metadata['internal_population_annotations_present'] is False


def test_relate_seed_must_fit_native_signed_integer():
    RelateConfig(seed=2**31-1).validate()
    for seed in (0, -1, 2**31):
        with pytest.raises(ValueError, match="signed-31-bit"):
            RelateConfig(seed=seed).validate()

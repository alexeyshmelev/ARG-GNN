"""Explicit TEST constructor; scientific extraction inherits unchanged TRAIN code.

The original TRAIN constructor remains untouched and retains its TRAIN guard.
A separate test-only constructor in smoke_test.py validates parity on TRAIN.
"""
from collections import OrderedDict, defaultdict
import numpy as np
from research.joint_model_20261029.data.observations import ObservationSource as FrozenObservationSource
from research.redesign_20261028.data.window_data import (
    COHORTS, ReplacementTreeReader, path_for, small_json, require, uniform_coordinate_context)
from .common import TEST_PARENTS, validate_registry


def initialize_adapter(source, parent_id, binding, *, split, allowed_parents, expected_count, cache_bytes):
    """Shared strict initializer; callers provide an explicit, closed split policy."""
    require(parent_id in allowed_parents, 'Parent is outside this explicitly registered split')
    require(set(binding) == {'refs', 'tree_ref', 'panel_ref', 'coordinate_offset_bp'}, 'Observation binding whitelist')
    require(set(binding['refs']) == {'sampled_registry_ref', 'inference_ref', 'observed_vcf_ref'},
            'Only observation reference bindings enter the adapter')
    source.parent_id, source.binding, source.cache_limit = parent_id, binding, int(cache_bytes)
    source.registry = small_json(binding['refs']['sampled_registry_ref'])
    source.panel = small_json(binding['panel_ref'])
    source.records = validate_registry(source.registry, source.panel, parent_id, split, expected_count)
    require(source.registry['observation_ref'] == binding['refs']['inference_ref'], 'Registry observation binding')
    source.native_samples = np.asarray([row['observed_node_id'] for row in source.panel], dtype=np.int64)
    source.sample_id = np.asarray([f"{parent_id}:{row['sample_id']}:{row['haplotype']}" for row in source.panel])
    require(len(np.unique(source.native_samples)) == len(source.panel) and
            len(set(source.sample_id)) == len(source.panel), 'Unique native nodes and namespaced haplotypes')
    source.offset = float(binding['coordinate_offset_bp'])
    require(np.isfinite(source.offset), 'Finite original VCF coordinate offset')
    source.metadata = np.zeros((len(source.panel), 17), dtype=np.float32)
    source.reference = np.asarray([-1 if row['observed_reference_class'] is None else row['observed_reference_class']
                                   for row in source.panel], dtype=np.int64)
    source.cohort = np.asarray([COHORTS.index(row['observed_cohort']) for row in source.panel], dtype=np.int64)
    source.age = np.asarray([row['age_generations'] for row in source.panel], dtype=np.float64)
    source.query = np.asarray([row['role'] == 'query' for row in source.panel], dtype=bool)
    require((source.reference[source.query] == -1).all() and
            np.isin(source.reference[~source.query], np.arange(4)).all(), 'Query ancestry unknown; reference classes only')
    source.metadata[np.arange(len(source.panel)), source.cohort] = 1
    source.metadata[np.arange(len(source.panel)), 9 + np.where(source.reference < 0, 4, source.reference)] = 1
    source.metadata[:, 14] = np.log1p(source.age) / np.log(10001.)
    source.metadata[:, 15] = source.query
    source.metadata[:, 16] = ~source.query
    source._pid = None
    source._reader = None
    source._vcf = None
    source._cache = OrderedDict()
    source._cache_bytes = 0
    source.counters = dict(tree_requests=0, tree_cache_hits=0, trees_read=0, VCF_sites_read=0)


class TestObservationSource(FrozenObservationSource):
    def __init__(self, parent_id, binding, cache_bytes=64 * 2**20):
        initialize_adapter(self, parent_id, binding, split='test', allowed_parents=TEST_PARENTS,
                           expected_count=10000, cache_bytes=cache_bytes)


def make_site_plan(source, parent_ordinal):
    require(source.parent_id in TEST_PARENTS, 'Only canonical TEST parents may be planned')
    require(parent_ordinal == TEST_PARENTS.index(source.parent_id) + 3, 'Disjoint prospective parent seed ordinal')
    reader = ReplacementTreeReader(path_for(source.binding['tree_ref']), source.native_samples)
    groups = defaultdict(list)
    for index in sorted(source.records):
        groups[source.records[index]['native_site_id']].append(source.records[index])
    sites = []
    for sid, rows in sorted(groups.items()):
        focal = float(rows[0]['position_local_bp'])
        require(sid < len(reader.columns['sites/position']) and reader.columns['sites/position'][sid] == focal,
                'Canonical registry site equals native observed SNP')
        require(focal + source.offset == rows[0]['position_original_bp'], 'Exact original VCF coordinate offset')
        rng = np.random.default_rng(np.random.SeedSequence([20261029, parent_ordinal, sid]))
        context = uniform_coordinate_context(rng, focal, 50000., 5, reader.sequence_length)
        sites.append(dict(window_index=len(sites), parent_id=source.parent_id, parent_ordinal=parent_ordinal,
                          native_site_id=sid, position_local_bp=focal,
                          position_original_bp=rows[0]['position_original_bp'],
                          example_indices=[row['example_index'] for row in rows],
                          sample_indices=[row['sample_index'] for row in rows], context=context))
    indices = np.asarray([i for site in sites for i in site['example_indices']], dtype=np.int64)
    require(np.array_equal(np.sort(indices), np.arange(10000)), 'Every TEST target planned exactly once')
    return dict(schema='joint_model_TEST_site_plan_v1', parent_id=source.parent_id,
                sequence_length_local_bp=reader.sequence_length, rows=10000, sites=sites)

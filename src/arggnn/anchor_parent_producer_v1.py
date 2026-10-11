"""H024 original observed producer and exact eight-tile V3 assembly.

Default: standard-library source report. Explicit binding CLI modes read metadata
or qualify observed inputs on CPU. No truth reader, fitting loop or GPU entry.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/anchor_parent_producer_v1.json'
CONFIG_SHA256 = '1a5a2e5d3d1080a285828959b013c648ae27ee28afd7229a0a42f78cd4884dcb'
PARENTS = tuple(f'bio_paths_pilot_{s}_{i:03d}' for s in ('train', 'val') for i in range(2))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _input_path(ref):
    _require(isinstance(ref, dict) and 'path' in ref and 'sha256' in ref, 'Explicit file digest required')
    _require(isinstance(ref['sha256'], str) and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Exact SHA256 required')
    p = Path(ref['path']); p = p if p.is_absolute() else ROOT / p
    _require(ROOT in p.parents and '..' not in p.parts and p.resolve() == p, 'Canonical repository input required')
    _require(not any(any(word in part.lower() for word in ('truth', 'oracle', 'test')) for part in p.relative_to(ROOT).parts), 'Truth/oracle/TEST input forbidden')
    return p


def _checked(ref):
    p = _input_path(ref)
    _require(p.is_file() and _sha(p) == ref['sha256'], 'Missing or changed bound observed input: ' + str(p))
    return p


def _json(ref):
    return json.loads(_checked(ref).read_text())


def source_report_v1():
    _require(_sha(CONFIG) == CONFIG_SHA256, 'Pinned producer configuration changed')
    cfg = json.loads(CONFIG.read_text())
    _require(cfg['execution_enabled'] is False, 'Configuration is source-only')
    for name, digest in cfg['source_pins'].items():
        _require(_sha(ROOT / name) == digest, 'Original source drift: ' + name)
    review = json.loads(_checked(cfg['core_review']).read_text())
    _require(review.get('verdict') == 'PASS' and review['implementation'] == cfg['core_source'] and review['config'] == cfg['core_config'], 'Exact independently PASSed V3 required')
    return {'kind': 'anchor_parent_producer_v1_source_report', 'status': 'source_only',
            'implementation_sha256': _sha(Path(__file__)), 'config_sha256': CONFIG_SHA256,
            'core_review': cfg['core_review'], 'actual_inputs_opened': False,
            'truth_opened': False, 'fit_performed': False, 'GPU_used': False}


def _dependencies():
    source_report_v1()
    import torch
    from . import biological_candidate as frozen
    from . import anchor_parent_context_v3 as core
    from . import kron_interface as kron
    for module in (frozen, core, kron):
        _require(Path(module.__file__).resolve() == ROOT / 'src/arggnn' / (module.__name__.rsplit('.', 1)[-1] + '.py'), 'Original module origin changed')
    return torch, frozen, core, kron


def validate_metadata_records_v1(binding, prepared, observed, cfg, source_rows, derived_rows, completion):
    """Pure metadata joins, also exercised with handmade records; no array IO."""
    pid = binding['parent_id']; split = 'train' if '_train_' in pid else 'val'
    _require(pid in PARENTS, 'One original development parent required')
    entries = observed.get('replicates', [])
    _require(len(entries) == 4 and {r['parent_id'] for r in entries} == set(PARENTS), 'Original four-parent observed manifest required')
    _require(observed.get('kind') == 'biological_paths_relate_v1' and observed.get('status') == 'four_development_parents_inferred_pre_and_post_five_iterations' and observed.get('primary_condition') == 'post_five_size_iterations_primary' and observed.get('truth_or_oracle_read') is False, 'Completed original observed Relate producer required')
    _require(observed['implementation_sha256'] == cfg['relate_producer']['implementation_sha256'] and observed['config_sha256'] == cfg['relate_producer']['config_sha256'], 'Original observed producer source/config required')
    entry = next(r for r in entries if r['parent_id'] == pid)
    _require(entry.get('split') == split and entry.get('status') == 'inferred_both_conditions' and entry.get('truth_or_oracle_read') is False, 'Observed parent lineage differs')
    _require(completion == {k: v for k, v in entry.items() if k != 'completion_provenance'}, 'Original completion record differs')
    _require(prepared.get('kind') == 'biological_candidate_prepared_v1' and prepared.get('status') == 'complete_observed_only_Q184_R96_actualQ120' and prepared.get('truth_opened') is False and prepared.get('labels_in_features_or_banks') is False, 'Original observed-only preparation required')
    _require(prepared['config_sha256'] == binding['base_config']['sha256'] and prepared['observed_manifest_sha256'] == binding['observed_manifest']['sha256'] and _input_path({'path': prepared['observed_manifest_path'], 'sha256': binding['observed_manifest']['sha256']}) == _input_path(binding['observed_manifest']), 'Prepared producer/config/manifest binding differs')
    _require(prepared['source_sha256'] == cfg['source_sha256'] and prepared['geometry'] == cfg['geometry'], 'Original prepared source or geometry differs')
    parents = prepared.get('parents', [])
    _require(len(parents) == 4 and {r['parent_id'] for r in parents} == set(PARENTS), 'Complete original preparation metadata required')
    record = next(r for r in parents if r['parent_id'] == pid)
    _require(record['split'] == split and record['relate_entry'] == entry, 'Prepared parent original Relate record differs')
    tree = entry['inferred_tree']; post = entry['post_five_iterations']; coordinate = entry['coordinate_contract']
    _require(tree == binding['native_tree'] and tree['condition'] == 'post_five_size_iterations_primary', 'Explicit original native tree binding required')
    _input_path(tree)
    _require(type(binding['native_tree_count']) is int and binding['native_tree_count'] > 0 and post['trees'] == record['original_trees_preserved'] == binding['native_tree_count'], 'Original tree inventory differs')
    _require(coordinate['original_parent_interval_bp0'] == [0, 2000000] and coordinate['normalization_length_bp'] == 2000000 and coordinate['physical_coordinate_transform'] == 'physical_original = local_tree_position + coverage_left' and coordinate['uncovered_tails'] == 'excluded; no extrapolation', 'Original physical coordinate contract required')
    coverage = coordinate['coverage_original_bp0']
    _require(len(coverage) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in coverage) and 0 <= coverage[0] < coverage[1] <= 2000000 and coordinate['local_to_original_offset_bp'] == coverage[0], 'Original physical coverage/offset required')
    _require(record['coverage_original_bp0'] == coverage, 'Prepared coverage differs')
    retained = coordinate['retained_fixed100bp_indexes']; expected = [i for i in range(20000) if coverage[0] <= 50 + 100 * i < coverage[1]]
    _require(retained == expected and record['retained_fixed100bp_indexes'] == retained and len(retained) > 0, 'Complete original retained centers required')
    positions = [50. + 100. * i for i in retained]; edges = [100. * i for i in retained] + [100. * (retained[-1] + 1)]
    edges[0] = max(edges[0], coverage[0]); edges[-1] = min(edges[-1], coverage[1])
    _require(record['positions_bp'] == positions and record['bin_edges_bp'] == edges, 'Original readout grid/partial bins differ')
    _require(len(source_rows) == len(derived_rows) == 280 and [r['sample_index'] for r in source_rows] == list(range(280)), 'Original ordered 280-ID roster required')
    mapping = entry['sample_index_to_inferred_tree_node_id']
    _require(len(mapping) == 280 and [r['sample_index'] for r in mapping] == list(range(280)), 'Explicit complete inferred-ID mapping required')
    nodes = [r['inferred_tree_node_id'] for r in mapping]
    _require(all(type(x) is int and x >= 0 for x in nodes) and len(set(nodes)) == 280, 'Original distinct inferred sampled IDs required')
    for source, derived, node in zip(source_rows, derived_rows, nodes):
        expected_row = dict(source); expected_row['source_observed_node_id'] = expected_row.pop('observed_node_id', None); expected_row.update(inferred_tree_node_id=node, observed_node_id=node)
        _require(derived == expected_row, 'Derived roster changed original observed attributes')
    q = [i for i, row in enumerate(source_rows) if row['role'] == 'query']; mixed = [i for i, row in enumerate(source_rows) if row['role'] == 'mixed_reference_only']; r = [i for i, row in enumerate(source_rows) if row['role'] == 'pure_source_reference']; dynamic = [i for i in range(280) if i not in r]
    _require((len(q), len(mixed), len(r), len(dynamic)) == (120, 64, 96, 184), 'Original actual120/mixed64/Q184/R96 roles required')
    _require(all(row['observed_reference_class'] in range(4) if row['role'] == 'pure_source_reference' else row['observed_reference_class'] is None for row in source_rows), 'Truth class in query/mixed input forbidden')
    _require([sum(source_rows[i]['observed_reference_class'] == c for i in r) for c in range(4)] == [24] * 4, 'Original balanced observed source classes required')
    _require(record['dynamics_sample_indexes'] == dynamic and record['actual_query_sample_indexes'] == q and record['reference_sample_indexes'] == r and record['actual_query_output_indexes'] == [dynamic.index(i) for i in q] and record['query_node_ids'] == [nodes[i] for i in q] and record['dynamics_query_node_ids'] == [nodes[i] for i in dynamic], 'Prepared observed sample/role joins differ')
    derived_ref = entry['observed_sample_manifest']; source_ref = derived_ref['source_manifest']
    _require(source_ref == entry['original_observed_sample_manifest'] and source_ref['sha256'] == cfg['observed_roster_sha256'][pid], 'Original observed roster hash differs')
    _require(post['clean_tree'] == tree['path'] and post['clean_tree_sha256'] == tree['sha256'] and post['sample_index_to_inferred_tree_node_id'] == mapping and post['observed_sample_manifest'] == derived_ref and post['coverage_original_bp0'] == coverage and post['local_to_original_offset_bp'] == coverage[0] and post['sequence_length_local_bp'] == coverage[1] - coverage[0] and post['retained_fixed100bp_indexes'] == retained, 'Primary condition source join differs')
    tiles = record['tiles']; _require(len(tiles) == 8 and [t['tile_id'] for t in tiles] == list(range(8)), 'Eight unchanged tile records required')
    seen = []; ids = []
    for i, tile in enumerate(tiles):
        left = max(250000 * i, coverage[0]); right = min(250000 * (i + 1), coverage[1])
        indexes = [j for j, x in enumerate(positions) if left <= x < right]
        _require(left < right and len(indexes) > 0 and tile['physical_interval_bp0'] == [left, right] and tile['normalized_interval'] == [left / 2000000, right / 2000000] and tile['position_indexes'] == indexes and tile['original_grid_indexes'] == [retained[j] for j in indexes], 'Tile physical/readout support differs')
        tids = tile['original_tree_ids']; _require(tids and all(type(x) is int and 0 <= x < binding['native_tree_count'] for x in tids) and all(x < y for x, y in zip(tids, tids[1:])), 'Explicit serialized chronological original tree IDs required')
        _require(all(tile[k] is True for k in ('all_tree_nodes_retained', 'all280_leaves_retained', 'all_original_age_cache_fields_retained', 'exact_serialization_roundtrip_checked')) and (tile['query_dynamics'], tile['pure_reference_sources'], tile['supervised_outputs'], tile['mixed_latent_outputs'], tile['mixed_query_flag']) == (184, 96, 120, 64, 0), 'Full original tile bank contract required')
        _input_path(tile['bank']); _input_path(tile['whole_observed_graph']); seen.extend(indexes); ids.extend(tids)
    _require(seen == list(range(len(positions))) and set(ids) == set(range(binding['native_tree_count'])) and all(x <= y for x, y in zip(ids, ids[1:])), 'Original complete parent support/tree inventory required')
    return record, entry, {'nodes': nodes, 'query_slots': q, 'dynamics_slots': dynamic, 'reference_slots': r, 'positions': positions, 'edges': edges, 'coverage': coverage, 'retained': retained, 'source_rows': source_rows}


def read_metadata_v1(binding_ref):
    source_report_v1(); binding = _json(binding_ref); producer = json.loads(CONFIG.read_text())
    _require(set(binding) == {'kind', 'parent_id', 'base_config', 'prepared_manifest', 'observed_manifest', 'native_tree', 'native_tree_count', 'observed_only_authorized', 'cpu_headroom_bytes'}, 'Exact narrow observed producer binding required')
    _require(binding['kind'] == 'anchor_parent_producer_binding_v1' and type(binding['observed_only_authorized']) is bool and type(binding['cpu_headroom_bytes']) is int and binding['cpu_headroom_bytes'] > 0, 'Explicit root observed scope and CPU headroom required')
    _require(binding['base_config'] == producer['base_config'], 'Original biological configuration required')
    cfg = _json(binding['base_config']); prepared = _json(binding['prepared_manifest']); observed = _json(binding['observed_manifest'])
    _require(binding['parent_id'] in PARENTS, 'Original development parent required')
    entry = next(r for r in observed['replicates'] if r['parent_id'] == binding['parent_id'])
    source_ref = entry['observed_sample_manifest']['source_manifest']; derived_ref = entry['observed_sample_manifest']
    source_rows = _json(source_ref); derived_rows = _json(derived_ref); completion = _json(entry['completion_provenance'])
    record, entry, metadata = validate_metadata_records_v1(binding, prepared, observed, cfg, source_rows, derived_rows, completion)
    for ref in [entry['commands_provenance'], coordinate_ref := entry['coordinate_contract']['fixed_grid'], binding['native_tree'], *(t[k] for t in record['tiles'] for k in ('bank', 'whole_observed_graph'))]:
        _input_path(ref)  # metadata phase never opens tree, grid or tensor files
    _require(_input_path(source_ref) == ROOT / 'data/biological_paths_execution_pilot_v1' / binding['parent_id'] / 'observed/sample_manifest.json' and _input_path(coordinate_ref) == ROOT / 'data/biological_paths_execution_pilot_v1' / binding['parent_id'] / 'observed/scoring_coordinates.npz', 'Separate original source roster/grid paths required')
    return binding, cfg, record, entry, metadata


def verify_native_inventory_v1(ts, record, metadata):
    """Check IDs against original native tree iteration; never infer by matching."""
    _require(ts.num_trees == record['original_trees_preserved'] and ts.sequence_length == metadata['coverage'][1] - metadata['coverage'][0] and set(map(int, ts.samples())) == set(metadata['nodes']), 'Original native tree length/inventory/sample set differs')
    inventory = []
    for expected_id, tree in enumerate(ts.trees()):
        _require(tree.index == expected_id and tree.num_roots == 1 and set(map(int, tree.leaves())) == set(metadata['nodes']), 'Original explicit tree IDs and full leaves required')
        inventory.append((tree.index, tree.interval.left + metadata['coverage'][0], tree.interval.right + metadata['coverage'][0]))
    for tile in record['tiles']:
        left, right = tile['physical_interval_bp0']; clipped = [(i, max(a, left), min(b, right)) for i, a, b in inventory if max(a, left) < min(b, right)]
        _require([i for i, _, _ in clipped] == tile['original_tree_ids'], 'Tile IDs do not match original native serialized tree order')
    return tuple(inventory)


@dataclass(frozen=True)
class VerifiedParentInputsV1:
    loaded_tiles: tuple
    tile_records: tuple
    parent_record: dict
    namespace: str
    source_checks: tuple
    fixture: bool = False


@dataclass(frozen=True)
class ParentAssemblyV1:
    parent: object
    leaf: object
    response: object
    leaf_ids: object
    input_proof: VerifiedParentInputsV1
    geometry: str


def assemble_parent_v1(verified, geometry):
    """Keep every original occurrence and exact cached anchor, with plain input."""
    torch, frozen, core, _ = _dependencies()
    _require(isinstance(verified, VerifiedParentInputsV1) and type(verified.fixture) is bool and geometry in ('age', 'plain'), 'Explicit verified producer inputs and geometry required')
    loaded = verified.loaded_tiles; records = verified.tile_records
    _require(len(loaded) == len(records) == len(verified.source_checks) == 8 and [t['tile_id'] for t in records] == list(range(8)), 'Complete eight original tile occurrences required')
    _require(all(c.get('original_tree_ids_verified') is True and c.get('serialized_graph_reproduced') is True and c.get('plain_bank_inputs_reproduced') is True for c in verified.source_checks), 'Genuine native-source reconstruction required')
    leaf_ids = loaded[0].bank.sample_ids; contexts = []
    for prepared, record in zip(loaded, records):
        bank = prepared.bank; cache = bank.query_cache
        _require(bank.leaf_x.device.type == 'cpu' and bank.leaf_x.dtype == torch.float32 and bank.leaf_x.shape[-1] == 47 and not bank.leaf_x.requires_grad and not cache.normalized_response.requires_grad, 'Fixed original CPU float32 47-feature inputs required')
        _require(torch.equal(bank.sample_ids, leaf_ids) and torch.equal(bank.sample_ids[cache.query_indices], cache.query_ids) and torch.equal(bank.sample_ids[cache.reference_indices], cache.reference_ids), 'Exact full leaf/Q/R order required')
        _require(prepared.num_scales == 3 and prepared.layout.anchors.numel() == 32 and bank.feature_columns == tuple(c for c in range(48) if c != 7), 'Original32 anchors/three scales/47 selected fields required')
        if not verified.fixture:
            _require(bank.leaf_x.shape[1:] == (280, 47) and cache.normalized_response.shape[1:] == (280, 280) and len(cache.query_ids) == 184 and len(cache.reference_ids) == 96 and len(leaf_ids) == 280, 'Original full280/Q184/R96 required')
        original_ids = tuple(record['original_tree_ids']); _require(len(original_ids) == bank.leaf_x.shape[0], 'Explicit original tree row map required')
        fingerprints = tuple(frozen.fingerprint_sha({'parent_id': verified.parent_record['parent_id'], 'native_namespace': verified.namespace, 'original_tree_id': tree_id, 'leaf_ids': leaf_ids, 'query_ids': cache.query_ids, 'reference_ids': cache.reference_ids, 'feature_columns': bank.feature_columns, 'leaf_x': bank.leaf_x[g], 'plain_normalized_response': cache.normalized_response[g]}) for g, tree_id in enumerate(original_ids))
        contexts.append(core.TileContextV3(prepared.layout, prepared.operators[geometry], cache.query_ids, cache.reference_ids, original_ids, geometry=geometry, feature_fingerprints=fingerprints, parent_id=verified.parent_record['parent_id'], split=verified.parent_record['split'], namespace=verified.namespace))
    parent = core.stitch_context_v3(contexts, fixture=verified.fixture)
    _require(len(parent.layout.anchors) == 256 and torch.equal(parent.layout.anchors, torch.cat([x.layout.anchors for x in loaded])) and torch.equal(parent.layout.weights, torch.cat([x.layout.weights for x in loaded])), 'Exact cached256-anchor/weight union required')
    leaf = torch.cat([x.bank.leaf_x for x in loaded]); response = torch.cat([x.bank.query_cache.normalized_response for x in loaded])
    return ParentAssemblyV1(parent, leaf, response, leaf_ids.clone(), verified, geometry)


def qualify_observed_v1(binding_ref):
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Observed qualification requires explicitly empty CUDA visibility')
    binding, cfg, record, entry, metadata = read_metadata_v1(binding_ref)
    _require(binding['observed_only_authorized'] is True, 'Root observed-only admission required; metadata is not array admission')
    torch, frozen, _, kron = _dependencies()
    import numpy as np
    import tskit
    frozen.configure_cpu()
    # Lower bound only; actual full-support numerical/device profiling remains separate.
    serialized = sum(_input_path(t[k]).stat().st_size for t in record['tiles'] for k in ('bank', 'whole_observed_graph'))
    graphs = sum(len(t['original_tree_ids']) for t in record['tiles']); unique = binding['native_tree_count']
    input_bytes = graphs * (280 * 47 + 280 * 280) * 4
    estimate = serialized + 2 * input_bytes + unique * 280 * 280 * 8 + 2000000000
    _require(binding['cpu_headroom_bytes'] >= estimate, 'Declared CPU headroom is below retained/reconstruction input lower bound')
    ts = tskit.load(_checked(binding['native_tree'])); verify_native_inventory_v1(ts, record, metadata)
    _require(np.allclose(ts.tables.nodes.time[np.asarray(metadata['nodes'])], [r['age_generations'] for r in metadata['source_rows']], atol=1e-6, rtol=0), 'Original observed sample ages differ')
    grid_path = _checked(entry['coordinate_contract']['fixed_grid'])
    with np.load(grid_path, allow_pickle=False) as grid:
        _require(np.array_equal(grid['positions_bp'], np.arange(50, 2000000, 100, dtype=float)) and np.array_equal(grid['bin_edges_bp'], np.arange(0, 2000001, 100, dtype=float)), 'Original complete physical grid required')
    _checked(entry['commands_provenance'])
    parent = {'parent_id': binding['parent_id'], 'split': record['split'], 'samples': metadata['source_rows'], 'query_slots': np.asarray(metadata['query_slots']), 'dynamics_slots': np.asarray(metadata['dynamics_slots']), 'reference_slots': np.asarray(metadata['reference_slots']), 'positions': np.asarray(metadata['positions']), 'edges': np.asarray(metadata['edges']), 'observation': {'nodes': np.asarray(metadata['nodes']), 'offset': metadata['coverage'][0]}}
    loaded = []; checks = []
    for tile in record['tiles']:
        left, right = tile['physical_interval_bp0']; sequence, indexes, original_ids, landmarks = frozen.pack_observed_tile(ts, parent, left, right)
        _require(original_ids == tile['original_tree_ids'] and indexes.tolist() == tile['position_indexes'] and landmarks == tile['landmarks_observed_node_ids'], 'Original pack tree/readout/landmark identity differs')
        graph_path = _checked(tile['whole_observed_graph']); raw_graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        _require(raw_graph.get('labels_in_input') is False and frozen.fingerprint_sha(raw_graph['sequence']) == frozen.fingerprint_sha(sequence.as_dict()), 'Serialized observed graph differs from original native producer')
        del raw_graph
        bank_path = _checked(tile['bank']); saved = torch.load(bank_path, map_location='cpu', weights_only=False)
        _require(frozen.fingerprint_sha(saved) == tile['full_bank_field_fingerprint_sha256'], 'Original full age/plain cache artifact fingerprint changed')
        del saved; gc.collect()
        prepared = frozen.load_bank(tile, cfg, parent, 'cpu')
        fresh = kron.precompute_leaf_interfaces(sequence, time_column=6, parent_branch_column=7, time_scale=10000., resistance_scale=.0001, max_condition=1e12)
        fresh = kron.with_query_reference_cache(fresh, torch.as_tensor(np.asarray(metadata['nodes'])[metadata['reference_slots']], dtype=torch.long)).to('cpu', torch.float32)
        for name in ('leaf_x', 'sample_ids', 'query_node_ids', 'intervals', 'positions', 'target_graph', 'window_bounds'):
            _require(torch.equal(getattr(prepared.bank, name), getattr(fresh, name)), 'Original producer bank field differs: ' + name)
        for name in ('normalized_response', 'query_ids', 'reference_ids', 'query_indices', 'reference_indices'):
            _require(torch.equal(getattr(prepared.bank.query_cache, name), getattr(fresh.query_cache, name)), 'Original plain input/ID field differs: ' + name)
        checks.append({'tile_id': tile['tile_id'], 'original_tree_ids_verified': True, 'serialized_graph_reproduced': True, 'plain_bank_inputs_reproduced': True, 'original_cache_artifact_sha256': tile['bank']['sha256']})
        loaded.append(prepared); del sequence, fresh; gc.collect()
    namespace = binding['parent_id'] + ':' + binding['native_tree']['sha256']
    verified = VerifiedParentInputsV1(tuple(loaded), tuple(record['tiles']), record, namespace, tuple(checks))
    assemblies = {geometry: assemble_parent_v1(verified, geometry) for geometry in ('age', 'plain')}
    unique_ids, first, inverse = _dependencies()[2]._original_row_plan_v3(assemblies['age'].parent, (assemblies['age'].leaf, assemblies['age'].response))
    artifact_refs = {'base_config': binding['base_config'], 'prepared_manifest': binding['prepared_manifest'], 'observed_manifest': binding['observed_manifest'], 'native_tree': binding['native_tree'], 'source_roster': entry['original_observed_sample_manifest'], 'derived_roster': entry['observed_sample_manifest'], 'completion': entry['completion_provenance'], 'commands': entry['commands_provenance'], 'grid': entry['coordinate_contract']['fixed_grid'], 'tiles': [{'tile_id': t['tile_id'], 'bank': t['bank'], 'whole_observed_graph': t['whole_observed_graph'], 'original_tree_ids': t['original_tree_ids'], 'physical_interval_bp0': t['physical_interval_bp0']} for t in record['tiles']]}
    report = {'kind': 'anchor_parent_producer_observed_qualification_v1', 'status': 'observed_inputs_bound_no_numerical_profile_or_fit', 'parent_id': binding['parent_id'], 'binding_ref': binding_ref, 'namespace': namespace, 'source_report': source_report_v1(), 'original_tree_count': unique, 'cache_graph_occurrences': graphs, 'cached_anchor_union': 256, 'all_leaf_ids': assemblies['age'].leaf_ids.tolist(), 'query_ids': assemblies['age'].parent.query_ids.tolist(), 'reference_ids': assemblies['age'].parent.reference_ids.tolist(), 'tile_source_checks': checks, 'artifact_refs': artifact_refs, 'feature_columns': list(assemblies['age'].input_proof.loaded_tiles[0].bank.feature_columns), 'exact_anchor_union': assemblies['age'].parent.layout.anchors.tolist(), 'exact_weight_union': assemblies['age'].parent.layout.weights.tolist(), 'physical_layout_fingerprint_sha256': frozen.fingerprint_sha(assemblies['age'].parent.layout), 'target_graph_fingerprint_sha256': frozen.fingerprint_sha(assemblies['age'].parent.target_graph), 'unique_original_tree_ids': list(unique_ids), 'first_cache_graph_indexes': list(first), 'inverse_cache_graph_indexes': list(inverse), 'input_fingerprints': {'leaf': frozen.field_fingerprint(assemblies['age'].leaf), 'plain_response': frozen.field_fingerprint(assemblies['age'].response)}, 'input_lower_bound_bytes': estimate, 'declared_CPU_headroom_bytes': binding['cpu_headroom_bytes'], 'memory_is_measured_peak': False, 'target_arrays_opened': False, 'truth_opened': False, 'models_instantiated': False, 'fit_performed': False, 'GPU_used': False}
    return assemblies, report


def original_model_optimizer_v1(geometry, seed, *, dtype=None):
    """Original47-feature model and unchanged AdamW groups; caller owns admission."""
    torch, frozen, _, _ = _dependencies()
    _require(geometry in ('age', 'plain') and type(seed) is int and seed in (101, 202, 303), 'Original explicit arm/seed required')
    _require(dtype is None or dtype in (torch.float32, torch.float64), 'Qualified float32/float64 model dtype required')
    cfg = json.loads((ROOT / 'configs/biological_candidate_pilot_v1.json').read_text())
    torch.manual_seed(seed)
    model = frozen.model_factory('candidate_' + geometry + '_d2', cfg).to('cpu', dtype=torch.float32 if dtype is None else dtype)
    optimizer = torch.optim.AdamW(frozen.optimizer_groups(model, cfg), lr=cfg['training']['learning_rate'])
    return model, optimizer


def forward_parent_v1(assembly, model, *, mode, checkpoint_blocks=True):
    """Full original targets, global anchor sources and connected V3 gradients."""
    _, _, core, _ = _dependencies()
    _require(isinstance(assembly, ParentAssemblyV1) and mode in ('reset', 'carry'), 'Explicit assembly and matched mode required')
    weight = model.encoder.input_linear.weight
    context = core.parent_context_v3(assembly.parent, model, assembly.leaf.to(weight), assembly.response.to(weight), leaf_ids=assembly.leaf_ids, fixture=assembly.input_proof.fixture, mode=mode, event_block_size=64, checkpoint_blocks=checkpoint_blocks)
    return model.head(context.targets), context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(); mode.add_argument('--metadata', action='store_true'); mode.add_argument('--qualify-observed', action='store_true')
    parser.add_argument('--binding'); parser.add_argument('--expected-binding-sha256'); parser.add_argument('--expected-source-sha256'); parser.add_argument('--expected-config-sha256')
    args = parser.parse_args(argv)
    if not args.metadata and not args.qualify_observed:
        print(json.dumps(source_report_v1(), sort_keys=True)); return
    _require(args.expected_source_sha256 == _sha(Path(__file__)) and args.expected_config_sha256 == CONFIG_SHA256 and args.binding and args.expected_binding_sha256, 'Exact source/config and binding pins required')
    ref = {'path': args.binding, 'sha256': args.expected_binding_sha256}
    if args.metadata:
        binding, _, record, _, _ = read_metadata_v1(ref)
        report = {'kind': 'anchor_parent_producer_metadata_v1', 'status': 'metadata_bound_no_array_admission', 'parent_id': binding['parent_id'], 'binding_ref': ref, 'tiles': len(record['tiles']), 'native_tree_count': binding['native_tree_count'], 'tree_grid_bank_arrays_opened': False, 'truth_opened': False, 'fit_performed': False, 'GPU_used': False}
    else:
        _, report = qualify_observed_v1(ref)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

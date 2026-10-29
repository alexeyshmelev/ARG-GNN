"""TRAIN-only parity, spawned transport, frozen CPU logits, and ID failures."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from research.joint_model_20261029.data.observations import ObservationSource as FrozenObservationSource, MODEL_KEYS
from research.joint_model_20261029.model.joint_model_v1 import (
    JointAncestryModel, prepare_observations, prepared_to_state, collate_prepared)
from research.redesign_20261028.data.window_data import ROOT, require, uniform_coordinate_context
from .common import BASE, ref, write_json, validate_registry, canonical_output, json_data, now
from .observations import initialize_adapter, TestObservationSource
from .freeze import RUNTIME_FILES, DEPENDENCIES
from . import workers

TRAIN_PARENTS = tuple(f'ancestralpaths_european_train_{i:03d}' for i in range(3))


class TrainParityObservationSource(FrozenObservationSource):
    def __init__(self, parent_id, binding):
        initialize_adapter(self, parent_id, binding, split='train', allowed_parents=TRAIN_PARENTS,
                           expected_count=None, cache_bytes=64 * 2**20)


def train_worker_init(parent_id, binding):
    from multiprocessing.util import Finalize
    import setproctitle
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    setproctitle.setproctitle('AI Agent Research Data')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workers.SOURCE = TrainParityObservationSource(parent_id, binding)
    Finalize(None, workers.close_worker, exitpriority=10)


def exact_equal(left, right):
    if isinstance(left, torch.Tensor):
        require(type(right) is torch.Tensor and left.layout == right.layout and left.shape == right.shape,
                'Identical prepared tensor type, layout, and shape')
        if left.layout == torch.sparse_csr:
            exact_equal(left.crow_indices(), right.crow_indices())
            exact_equal(left.col_indices(), right.col_indices())
            exact_equal(left.values(), right.values())
        else:
            require(torch.equal(left, right), 'Bitwise prepared tensor equality')
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), 'Prepared state dictionary keys')
        for key in left:
            exact_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        require(type(left) is type(right) and len(left) == len(right), 'Prepared container structure')
        for a, b in zip(left, right):
            exact_equal(a, b)
    else:
        require(left == right, 'Prepared scalar equality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(BASE / 'smoke_result.json'))
    args = parser.parse_args()
    import setproctitle
    setproctitle.setproctitle('AI Agent Research Data')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    assessment = json.loads((ROOT / 'research/joint_model_20261029/data/SOURCE_ASSESSMENT.json').read_text())
    train_plan = json.loads((ROOT / 'research/joint_model_20261029/data/site_plan_v1.json').read_text())['sites']
    parent = TRAIN_PARENTS[0]
    original_binding = assessment['parents'][parent]
    binding = {k: original_binding[k] for k in ('tree_ref', 'panel_ref', 'coordinate_offset_bp')}
    binding['refs'] = {k: original_binding['refs'][k] for k in ('sampled_registry_ref', 'inference_ref', 'observed_vcf_ref')}
    frozen = FrozenObservationSource(parent, binding)
    adapter = TrainParityObservationSource(parent, binding)
    failures = []

    def rejected(name, operation):
        try:
            operation()
        except (ValueError, KeyError, IndexError):
            failures.append(name)
        else:
            raise AssertionError('Malformed input unexpectedly accepted: ' + name)

    rejected('production_TEST_class_rejects_TRAIN_parent', lambda: TestObservationSource(parent, binding))
    for name in ('duplicate_example_ID', 'missing_example_ID', 'wrong_sample_identity', 'unexpected_label_field', 'float_example_ID'):
        registry = copy.deepcopy(adapter.registry)
        if name == 'duplicate_example_ID':
            registry['records'][1]['example_index'] = registry['records'][0]['example_index']
        elif name == 'missing_example_ID':
            registry['records'].pop()
        elif name == 'wrong_sample_identity':
            registry['records'][0]['sample_index'] = (registry['records'][0]['sample_index'] + 1) % len(adapter.panel)
        elif name == 'unexpected_label_field':
            registry['records'][0]['label'] = 0
        else:
            registry['records'][0]['example_index'] = float(registry['records'][0]['example_index'])
        rejected(name, lambda: validate_registry(registry, adapter.panel, parent, 'train'))

    available = [site for site in train_plan if site['parent_id'] == parent]
    # Include single-cohort and many-target windows without changing registered contexts.
    selected = available[:3] + [next(site for site in available if len(site['example_indices']) >= 162)]
    selected = [dict(site, window_index=i) for i, site in enumerate(selected)]
    expected, site_receipts = [], []
    for site in selected:
        packet_old, _ = frozen.extract(site)
        packet_new, metadata = adapter.extract(site)
        require(set(packet_old) == set(packet_new) == set(MODEL_KEYS), 'Exact observation-only field set')
        for key in MODEL_KEYS:
            require(packet_old[key].dtype == packet_new[key].dtype and
                    np.array_equal(packet_old[key], packet_new[key]), 'Exact TRAIN extraction field parity: ' + key)
        expected.append(prepare_observations(packet_old))
        exact_equal(prepared_to_state(expected[-1]), prepared_to_state(prepare_observations(packet_new)))
        site_receipts.append(dict(native_site_id=site['native_site_id'], targets=len(site['example_indices']),
                                  fields_compared=len(MODEL_KEYS), unique_native_trees=metadata['unique_native_trees']))
    bad_site = copy.deepcopy(selected[0])
    bad_site['sample_indices'][0] = (bad_site['sample_indices'][0] + 1) % len(adapter.panel)
    rejected('wrong_site_sample_join', lambda: adapter.extract(bad_site))
    bad_site = copy.deepcopy(selected[0])
    bad_site['example_indices'][0] = len(adapter.records)
    rejected('missing_site_example_ID', lambda: adapter.extract(bad_site))
    actual, transport = [], []
    for index, batch, metadata in workers.prepared_stream(parent, binding, selected, initializer=train_worker_init):
        require(index == len(actual), 'Worker order matches site plan')
        exact_equal(prepared_to_state(expected[index]), prepared_to_state(batch))
        actual.append(batch)
        transport.append(metadata)
    checkpoint = ROOT / 'research/joint_model_20261029/runs/benchmark_structure_v1/final_state.pt'
    model = JointAncestryModel(width=64, layers=3, classes=6).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    with torch.inference_mode():
        expected_logits = model(collate_prepared(expected), mode='structure')
        actual_logits = model(collate_prepared(actual), mode='structure')
    require(torch.equal(expected_logits, actual_logits), 'Bitwise frozen CPU logits after spawned byte transport')
    records = {i: adapter.records[i] for i in range(4)}
    logits = np.arange(24, dtype=np.float32).reshape(4, 6) / 10
    ids = np.asarray([2, 0, 3, 1], dtype=np.int64)
    result = canonical_output(logits, ids, records, 4)
    require(np.array_equal(result['logits'], logits[np.argsort(ids)]), 'Canonical prediction reordering')
    require(np.array_equal(result['sample_indices'], [records[i]['sample_index'] for i in range(4)]), 'Canonical sample output join')
    rejected('duplicate_output_ID', lambda: canonical_output(logits, np.asarray([0, 1, 1, 3], dtype=np.int64), records, 4))
    rejected('missing_output_ID', lambda: canonical_output(logits[:3], ids[:3], records, 4))
    rejected('out_of_range_output_ID', lambda: canonical_output(logits, np.asarray([0, 1, 2, 4], dtype=np.int64), records, 4))
    rejected('noninteger_output_ID', lambda: canonical_output(logits, ids.astype(float), records, 4))
    rejected('nonfinite_logit', lambda: canonical_output(np.full((4, 6), np.nan, dtype=np.float32), ids, records, 4))
    # Exercise the same NumPy context producer that feeds production site-plan JSON.
    # The previously serialized TRAIN plan is an independent exact roundtrip oracle.
    fixture_dir = Path(args.output).with_suffix('')
    fixture_dir.mkdir(exist_ok=False)
    sequence_length = adapter._handles()[0].sequence_length
    plan_sites = []
    context_snapshots = []
    for site in selected:
        rng = np.random.default_rng(np.random.SeedSequence([20261029, 0, site['native_site_id']]))
        context = uniform_coordinate_context(rng, site['position_local_bp'], 50000., 5, sequence_length)
        for key, value in context.items():
            require(np.array_equal(np.asarray(value), np.asarray(site['context'][key])),
                    'Regenerated TRAIN context equals frozen TRAIN plan: ' + key)
        context_snapshots.append({key: np.array(value, copy=True) for key, value in context.items()})
        plan_sites.append(dict(site, context=context))
    plan_fixture = dict(schema='joint_model_TRAIN_serialization_smoke_v1', parent_id=parent,
                        rows=sum(len(site['example_indices']) for site in selected), sites=plan_sites,
                        numpy_scalar_probe=dict(integer=np.int64(7), floating=np.float64(1.25), boolean=np.bool_(True)))
    plan_ref = write_json(fixture_dir / 'site_plan.json', plan_fixture)
    restored_plan = json.loads((fixture_dir / 'site_plan.json').read_text())
    for original, restored, snapshot in zip(plan_sites, restored_plan['sites'], context_snapshots):
        require(original['example_indices'] == restored['example_indices'] and
                original['sample_indices'] == restored['sample_indices'], 'JSON preserves exact canonical row identities')
        for key in original['context']:
            require(np.array_equal(np.asarray(original['context'][key]), np.asarray(restored['context'][key])),
                    'Context metadata survives exact JSON roundtrip: ' + key)
            require(np.array_equal(np.asarray(original['context'][key]), snapshot[key]),
                    'JSON conversion does not mutate scientific arrays: ' + key)
    # Production normalizes the generated context before handing it to unchanged extraction.
    # Exercise both the in-memory normalized form and its persisted JSON roundtrip.
    for original, restored in zip(selected, restored_plan['sites']):
        normalized = dict(restored, context=json_data(next(site['context'] for site in plan_sites
                                                       if site['native_site_id'] == restored['native_site_id'])))
        baseline_packet, _ = frozen.extract(original)
        normalized_packet, _ = adapter.extract(normalized)
        roundtrip_packet, _ = adapter.extract(restored)
        for key in MODEL_KEYS:
            require(np.array_equal(baseline_packet[key], normalized_packet[key]) and
                    np.array_equal(baseline_packet[key], roundtrip_packet[key]),
                    'Regenerated normalized/JSON context preserves extraction: ' + key)
    require(restored_plan['numpy_scalar_probe'] == dict(integer=7, floating=1.25, boolean=True),
            'Nested NumPy scalar conversion')
    # Representative six-array NPZ plus complete product JSON follows the production writer convention.
    with (fixture_dir / 'predictions.npz').open('xb') as stream:
        np.savez(stream, **result)
    with np.load(fixture_dir / 'predictions.npz', allow_pickle=False) as archive:
        require(set(archive.files) == set(result), 'Prediction archive field whitelist')
        for key in result:
            require(archive[key].dtype == result[key].dtype and np.array_equal(archive[key], result[key]),
                    'Prediction output exact roundtrip: ' + key)
    product_fixture = dict(schema='joint_model_native6_prediction_v1', model_id='TRAIN_SERIALIZATION_SMOKE_ONLY',
          parent_id=parent, split='train', rows=4, native_path_names=adapter.registry['native_path_names'],
          registry_ref=binding['refs']['sampled_registry_ref'], predictions_ref=ref(fixture_dir / 'predictions.npz'),
          site_plan_ref=plan_ref, frozen_checkpoint_ref=ref(checkpoint), mode='structure',
          query_ancestry_truth_opened=False, complete=True)
    product_ref = write_json(fixture_dir / 'product.json', product_fixture)
    require(json.loads((fixture_dir / 'product.json').read_text()) == product_fixture,
            'Representative prediction product JSON exact roundtrip')
    rejected('nonfinite_JSON_metadata', lambda: write_json(fixture_dir / 'must_not_exist.json', dict(value=np.float64(np.nan))))
    require(not (fixture_dir / 'must_not_exist.json').exists(), 'Serialization failure does not leave a partial JSON artifact')
    require(not torch.cuda.is_initialized(), 'Smoke never initializes CUDA')
    receipt = dict(schema='joint_model_prediction_TRAIN_smoke_v1', passed=True, recorded_utc=now(),
                   elapsed_seconds=time.monotonic() - start, TEST_payloads_opened=False,
                   query_ancestry_truth_opened=False, CUDA_initialized=False, parent_id=parent,
                   runtime_source_refs=[ref(BASE / name) for name in RUNTIME_FILES],
                   dependency_refs=[ref(ROOT / name) for name in DEPENDENCIES], checkpoint_ref=ref(checkpoint),
                   train_site_plan_ref=ref(ROOT / 'research/joint_model_20261029/data/site_plan_v1.json'),
                   sites=site_receipts, all_observation_fields_bitwise_equal=True,
                   all_prepared_states_bitwise_equal=True, frozen_CPU_logits_bitwise_equal=True,
                   logits_rows=len(actual_logits), negative_cases_rejected=failures,
                   spawned_worker_pids=sorted({item['worker_pid'] for item in transport}),
                   workers_shutdown_cleanly=True, persistent_bulk_cache_created=False,
                   TRAIN_numpy_site_plan_roundtrip=True, TRAIN_normalized_context_extraction_bitwise_equal=True,
                   scientific_arrays_unmodified_by_JSON=True,
                   prediction_NPZ_roundtrip=True, prediction_product_roundtrip=True,
                   serialization_site_plan_ref=plan_ref, serialization_product_ref=product_ref)
    print(json.dumps(write_json(args.output, receipt)), flush=True)


if __name__ == '__main__':
    main()

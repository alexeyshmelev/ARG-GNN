"""Small authenticated metadata and strict canonical output joins."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from research.redesign_20261028.data.window_data import ROOT, CLASSES, path_for, require

BASE = ROOT / 'research/joint_evaluation_20261029/prediction'
TEST_PARENTS = tuple(f'ancestralpaths_european_test_{i:03d}' for i in range(5))
MODEL_ID = 'treeloom_joint_structure_v1'
ROW_KEYS = {'cohort', 'example_index', 'haplotype', 'native_site_id', 'position_local_bp',
            'position_original_bp', 'query_node_id', 'sample_id', 'sample_index'}


def now():
    return datetime.now(timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'All artifacts remain inside project')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def verify_ref(reference):
    actual = ref(path_for(reference))
    require(actual['sha256'] == reference['sha256'], 'Authenticated source changed: ' + reference['path'])
    if 'bytes' in reference:
        require(actual['bytes'] == reference['bytes'], 'Authenticated source size changed')
    return actual


def write_json(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE), 'Prediction writes restricted to prediction directory')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return ref(path)


def validate_registry(registry, panel, parent_id, split, expected_count=None):
    require(registry['schema'] == 'ap_paper_sampled_registry_v1', 'Original native registry schema')
    require(registry['parent_id'] == parent_id and registry['split'] == split, 'Exact registry parent and split')
    require(registry['labels_present'] is False and registry['query_truth_read'] is False,
            'Observation-only registry required')
    require(registry['native_path_names'] == list(CLASSES), 'Frozen native six-class order')
    rows = registry['records']
    count = registry['examples']
    require(type(count) is int and count > 0 and len(rows) == count, 'Exact registry row count')
    if expected_count is not None:
        require(count == expected_count, 'Exactly the registered number of targets')
    require(all(set(row) == ROW_KEYS for row in rows), 'Exact label-free registry row whitelist')
    integer_keys = ('example_index', 'sample_index', 'native_site_id', 'query_node_id', 'haplotype')
    require(all(type(row[key]) is int and row[key] >= 0 for row in rows for key in integer_keys),
            'Nonnegative integer registry IDs')
    indices = [row['example_index'] for row in rows]
    require(len(set(indices)) == count and sorted(indices) == list(range(count)),
            'No duplicate, missing, or out-of-range canonical example IDs')
    require([row['sample_index'] for row in panel] == list(range(len(panel))), 'Native panel row order')
    pairs = set()
    sites = {}
    for row in rows:
        sample = row['sample_index']
        require(sample < len(panel), 'Registry sample index within native panel')
        observed = panel[sample]
        require(observed['role'] == 'query' and row['sample_id'] == observed['sample_id'] and
                row['haplotype'] == observed['haplotype'] and row['query_node_id'] == observed['observed_node_id'] and
                row['cohort'] == observed['observed_cohort'], 'Exact canonical sample/haplotype/cohort/native-node join')
        position = (row['position_local_bp'], row['position_original_bp'])
        require(all(isinstance(x, (int, float)) and np.isfinite(x) and x >= 0 for x in position),
                'Finite nonnegative canonical coordinates')
        sid = row['native_site_id']
        require(sid not in sites or sites[sid] == position, 'One coordinate pair per native site')
        sites[sid] = position
        pair = (sid, sample)
        require(pair not in pairs, 'No duplicate canonical site/sample targets')
        pairs.add(pair)
    if split == 'test':
        require(Counter(row['cohort'] for row in rows) ==
                Counter({c: 2000 for c in ('present_day', 'Bronze_Age', 'BAA', 'Neo', 'Yam')}),
                'Registered equal TEST cohort quotas')
    return {row['example_index']: row for row in rows}


def canonical_output(logits, example_indices, records, expected_count):
    ids = np.asarray(example_indices)
    require(ids.dtype == np.int64 and ids.shape == (expected_count,), 'One int64 ID per output row')
    require(np.array_equal(np.sort(ids), np.arange(expected_count)), 'No missing or duplicate prediction IDs')
    logits = np.asarray(logits)
    require(logits.dtype == np.float32 and logits.shape == (expected_count, 6) and np.isfinite(logits).all(),
            'Finite raw float32 native-six logits')
    require(sorted(records) == list(range(expected_count)), 'Canonical registry records complete')
    order = np.argsort(ids, kind='stable')
    logits = logits[order].copy()
    # Stable raw native-six softmax; no class masking, recalibration, or cohort postprocessing.
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    probabilities = (values / values.sum(axis=1, keepdims=True)).astype(np.float32)
    require(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1., atol=2e-7, rtol=0),
            'Finite normalized native-six probabilities')
    return dict(logits=logits, probabilities=probabilities, example_indices=np.arange(expected_count, dtype=np.int64),
                sample_indices=np.asarray([records[i]['sample_index'] for i in range(expected_count)], dtype=np.int64),
                native_site_ids=np.asarray([records[i]['native_site_id'] for i in range(expected_count)], dtype=np.int64),
                positions_original_bp=np.asarray([records[i]['position_original_bp'] for i in range(expected_count)], dtype=np.float64))

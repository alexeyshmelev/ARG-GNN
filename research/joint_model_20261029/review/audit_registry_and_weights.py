"""Audit canonical TRAIN identities and target weighting without reading labels."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import hashlib
import json

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/joint_model_20261029'
assessment_path = BASE / 'data/SOURCE_ASSESSMENT.json'
assessment = json.loads(assessment_path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parents = {}
all_target_ids = set()
for parent, source in assessment['parents'].items():
    assert parent in ['ancestralpaths_european_train_000', 'ancestralpaths_european_train_001', 'ancestralpaths_european_train_002']
    registry_ref = source['refs']['sampled_registry_ref']
    registry_path = ROOT / registry_ref['path']
    panel_path = ROOT / source['panel_ref']['path']
    assert sha(registry_path) == registry_ref['sha256']
    assert sha(panel_path) == source['panel_ref']['sha256']
    registry = json.loads(registry_path.read_text())
    panel = json.loads(panel_path.read_text())
    assert registry['parent_id'] == parent and registry['split'] == 'train'
    assert registry['labels_present'] is False and registry['query_truth_read'] is False
    assert len(panel) == 952
    assert [row['sample_index'] for row in panel] == list(range(len(panel)))
    identities = [(row['sample_id'], row['haplotype']) for row in panel]
    assert len(set(identities)) == len(panel)
    assert len(set(row['sample_id'] for row in panel)) < len(panel)
    for row in panel:
        if row['role'] == 'query':
            assert row['observed_reference_class'] is None
        else:
            assert row['role'] == 'pure_source_reference'
            assert row['observed_reference_class'] in [0, 1, 2, 3]
    records = registry['records']
    assert len(records) == source['canonical_examples']
    assert sorted(row['example_index'] for row in records) == list(range(len(records)))
    grouped = defaultdict(list)
    site_positions = defaultdict(set)
    pair_keys = set()
    for row in records:
        observed = panel[row['sample_index']]
        assert observed['role'] == 'query'
        assert (row['sample_id'], row['haplotype']) == (observed['sample_id'], observed['haplotype'])
        assert row['query_node_id'] == observed['observed_node_id'] == observed['inferred_tree_node_id']
        assert row['cohort'] == observed['observed_cohort']
        assert row['position_original_bp'] == row['position_local_bp'] + source['coordinate_offset_bp']
        assert 0 <= row['position_local_bp'] < source['sequence_length_bp']
        target = (parent, row['example_index'])
        assert target not in all_target_ids
        all_target_ids.add(target)
        pair = (row['native_site_id'], row['sample_index'])
        assert pair not in pair_keys
        pair_keys.add(pair)
        grouped[row['native_site_id']].append(row)
        site_positions[row['native_site_id']].add((row['position_local_bp'], row['position_original_bp']))
    assert len(grouped) == 415 and all(len(values) == 1 for values in site_positions.values())
    assert len({next(iter(values)) for values in site_positions.values()}) == 415
    histogram = Counter(len(rows) for rows in grouped.values())
    assert {str(k): v for k, v in sorted(histogram.items())} == source['targets_per_position_histogram']
    assert dict(Counter(row['cohort'] for row in records)) == source['cohort_counts']
    parents[parent] = {
        'registry_sha256': sha(registry_path),
        'panel_sha256': sha(panel_path),
        'canonical_targets': len(records),
        'unique_sites': len(grouped),
        'maximum_targets_per_site': max(map(len, grouped.values())),
        'minimum_targets_per_site': min(map(len, grouped.values())),
        'targets_per_site_histogram': {str(k): v for k, v in sorted(histogram.items())},
        'all_registry_identity_and_coordinate_joins_pass': True,
        'query_reference_classes_all_unknown': True,
        'pure_reference_labels_only': True,
        'stable_identity': '(parent_id, sample_id, haplotype); sample_id alone is not unique',
    }

assert len(all_target_ids) == 50000
assert sum(row['unique_sites'] for row in parents.values()) == 1245

# Enumerate all two-site batches from three sites. One large site carries loss 1
# on each of its ten targets; the two single-target sites carry loss 0.
site_counts = [1, 1, 10]
site_loss_sums = [0, 0, 10]
site_count = len(site_counts)
target_count = sum(site_counts)
batch_sites = 2
batches = list(combinations(range(site_count), batch_sites))
canonical_mean = Fraction(sum(site_loss_sums), target_count)
ratio_estimates = [Fraction(sum(site_loss_sums[i] for i in batch), sum(site_counts[i] for i in batch)) for batch in batches]
weighted_estimates = [Fraction(site_count * sum(site_loss_sums[i] for i in batch), batch_sites * target_count) for batch in batches]
ratio_expectation = sum(ratio_estimates, Fraction()) / len(batches)
weighted_expectation = sum(weighted_estimates, Fraction()) / len(batches)
assert ratio_expectation != canonical_mean
assert weighted_expectation == canonical_mean
receipt = {
    'schema': 'independent_train_registry_and_weighting_audit_v1',
    'verdict': 'pass',
    'audit_source_sha256': sha(Path(__file__)),
    'source_assessment_sha256': sha(assessment_path),
    'parents': parents,
    'total_canonical_targets': len(all_target_ids),
    'total_unique_sites': 1245,
    'internal_fit': {'parents': ['ancestralpaths_european_train_000', 'ancestralpaths_european_train_001'], 'targets': 33334, 'sites': 830},
    'internal_development': {'parents': ['ancestralpaths_european_train_002'], 'targets': 16666, 'sites': 415},
    'weighting_counterexample': {
        'site_target_counts': site_counts,
        'site_target_loss_sums': site_loss_sums,
        'sites_per_uniform_batch': batch_sites,
        'true_target_mean': str(canonical_mean),
        'expected_realized_target_denominator_estimate': str(ratio_expectation),
        'expected_fixed_target_denominator_estimate': str(weighted_expectation),
        'required_formula': 'M/(S*N) times the sum of all selected target losses for a uniform S-site batch from M sites and N total targets',
        'sampling_constraint': 'Choose the fixed site group before cost-based microbatch splitting; splitting must preserve the group denominator. Adaptive stopping by size is a different sampling law.',
    },
    'limitations': [
        'This authenticates original registries and observed panel metadata, not yet the forthcoming packet/feature adapter.',
        'Stable IDs are routing metadata, not numerical model features.',
        'No label arrays, TEST payloads, GPU, inference or fitting were accessed.',
        'Large tree/VCF source checksums remain inherited from source provenance; this audit does not rescan their payloads.',
    ],
}
(BASE / 'review/REGISTRY_AND_WEIGHTING_REVIEW.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: value for key, value in receipt.items() if key not in ['parents', 'limitations']}, indent=2))

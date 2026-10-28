"""Independent export audit: all metadata and one bounded actual packet per parent."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT/'research/redesign_20261028'
EXPORT = BASE/'diagnostic_packets'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest_path = EXPORT/'manifest.json'
manifest = json.loads(manifest_path.read_text())
completion = json.loads((EXPORT/'completion.json').read_text())
protocol = json.loads((BASE/'root/pilot_protocol.json').read_text())
assert completion['complete'] is True and completion['actual_draws'] == 192
assert completion['manifest_sha256'] == digest(manifest_path)
assert manifest['protocol_sha256'] == digest(BASE/'root/pilot_protocol.json')
assert digest(manifest_path) == 'eb9ba0cff26cd1479f8752ad8a2def2f5ebde1d9c03da24f79fc38fd6720ff51'
checked = 0
for split, seed, count, code in [('fit', 20261028, 128, 0), ('development', 20261029, 64, 1)]:
    high = 33334 if split == 'fit' else 16666
    draws = np.random.default_rng(seed).integers(0, high, size=count)
    rows = manifest[split]
    assert len(rows) == count
    for index, (raw, row) in enumerate(zip(draws, rows)):
        if split == 'fit':
            parent_number, example = divmod(int(raw), 16667)
        else:
            parent_number, example = 2, int(raw)
        parent = f'ancestralpaths_european_train_{parent_number:03d}'
        assert (row['draw_index'], row['parent_id'], row['example_index']) == (index, parent, example)
        assert row['context_seed_components'] == [20261030, code, index]
        length = manifest['source_bindings'][parent]['inferred_sequence_length_bp']
        focal = row['focal_position_local_bp']
        domain = np.array([max(0., focal-50000.), min(length, focal+50000.)])
        np.testing.assert_array_equal(row['context_domain_local_bp'], domain)
        expected = np.concatenate(([focal], np.random.default_rng(np.random.SeedSequence([20261030, code, index])).uniform(*domain, size=4)))
        np.testing.assert_array_equal(row['context_coordinates_local_bp'], expected)
        np.testing.assert_array_equal(row['context_weights'], [0., .25, .25, .25, .25])
        np.testing.assert_allclose(row['context_density_per_bp'], [0.]+[1/(domain[1]-domain[0])]*4, atol=0., rtol=0.)
        checked += 1

seen = set()
samples = []
for split in ['fit', 'development']:
    for row in manifest[split]:
        parent = row['parent_id']
        if parent in seen:
            continue
        seen.add(parent)
        packet_path = EXPORT/row['packet_path']
        target_path = EXPORT/row['label_path']
        assert digest(packet_path) == row['packet_sha256']
        assert digest(target_path) == row['label_sha256']
        target = np.load(target_path, allow_pickle=False)
        assert target.dtype == np.int64 and target.shape == ()
        source_ref = manifest['source_bindings'][parent]['training_target_arrays_ref']
        source_path = ROOT/source_ref['path']
        assert digest(source_path) == source_ref['sha256']
        with np.load(source_path, allow_pickle=False) as source:
            assert int(target) == int(source['native_labels_0based'][row['example_index']])
        with np.load(packet_path, allow_pickle=False) as p:
            assert 'labels' not in p.files
            assert p['focal_example_index'].tolist() == [row['example_index']]
            assert len(p['context_coordinate_local_bp']) == 5
            np.testing.assert_array_equal(p['context_coordinate_local_bp'], row['context_coordinates_local_bp'])
            assert int(p['context_forced_focal'].sum()) == 1
            assert int(p['tree_core_mask'].sum()) == 1
            assert bool(p['tree_core_mask'][p['focal_tree_index'][0]])
            assert np.all(p['reference_class'][p['query_mask']] == -1)
            samples.append(dict(parent_id=parent, split=split, draw_index=row['draw_index'],
                example_index=row['example_index'], packet_sha256=row['packet_sha256'],
                label_matches_canonical_source=True, labels_absent_from_observation=True))
assert len(samples) == 3
receipt=dict(schema='independent_diagnostic_export_audit_v1', verdict='pass',
    manifest_sha256=digest(manifest_path), protocol_sha256=manifest['protocol_sha256'],
    completion_sha256=digest(EXPORT/'completion.json'), all_metadata_rows_checked=checked,
    actual_packets_checked=samples,
    full_readback_author_validation=str(BASE/'data/diagnostic_packet_validation.json'),
    full_readback_author_validation_sha256=digest(BASE/'data/diagnostic_packet_validation.json'),
    no_training=True, no_TEST=True,
    scope='All target/context metadata independently reconstructed; one actual packet per parent independently authenticated and joined to canonical prepared TRAIN labels. Author validator separately checks all192 files.')
(BASE/'review/EXPORT_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

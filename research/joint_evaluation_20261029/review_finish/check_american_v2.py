"""Independent TRAIN-only cache, coverage, objective and learned-result review."""
import json
import os
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from research.joint_evaluation_20261029.american.common_v2 import BASE, RESULTS, ref, read, load_npz, source_refs
from research.joint_evaluation_20261029.american.observations import ObservationSource
from research.joint_evaluation_20261029.american.cache_v2 import read_cache
from research.joint_evaluation_20261029.american.state_codec_v2 import decode
from research.joint_evaluation_20261029.american.smoke_v2 import codec_boundary_tests
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel, prepare_observations, prepared_to_state, prepared_from_state, collate_prepared


def compare(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and a.layout == b.layout
        if a.layout == torch.sparse_csr:
            compare(a.crow_indices(), b.crow_indices())
            compare(a.col_indices(), b.col_indices())
            compare(a.values(), b.values())
        else:
            assert a.detach().numpy().tobytes() == b.detach().numpy().tobytes()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    else:
        assert a == b


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads((BASE / 'protocol_v2.json').read_text())
    smoke = read(ref(BASE / 'smoke_result_v2.json'))
    assert smoke['passed'] is True and smoke['source_refs'] == source_refs()
    assert smoke['protocol_ref'] == ref(BASE / 'protocol_v2.json')
    previous = read(protocol['predecessor_protocol_ref'])
    for key in ('parents', 'model', 'model_source_ref', 'context', 'input_adaptation', 'epochs',
                'training_target_presentations', 'initialization_seed', 'batch_seed', 'optimizer',
                'private_TRAIN_bindings_ref', 'postseal_scoring_bindings_ref'):
        assert previous[key] == protocol[key]
    counts = {}
    for parent, item in protocol['parents'].items():
        if item['split'] != 'train':
            continue
        records = load_npz(item['binding']['focal_registry_ref'])
        assert len(np.unique(records['record_ids'])) == 10000
        assert len(set(zip(records['positions_bp0'], records['sample_indices']))) == 10000
        assert np.array_equal(records['first_occurrence_intervals_bp'].mean(1), records['positions_bp0'])
        coordinates, inverse = np.unique(records['positions_bp0'], return_inverse=True)
        assert np.bincount(inverse).sum() == 10000
        counts[parent] = len(coordinates)
    assert len(counts) == 20 and sum(counts.values()) == 158476
    batches = sum((w + 15) // 16 for w in counts.values())
    for windows in counts.values():
        parent_batches = (windows + 15) // 16
        sizes = [min(16, windows - start) for start in range(0, windows, 16)]
        expected = sum(Fraction(size, windows) * Fraction(windows * batches, parent_batches * size * 200000)
                       for size in sizes) / batches
        assert expected == Fraction(1, 200000)
    codec_boundary_tests()
    parent = 'argformer_american_train_000'
    source = ObservationSource(parent, protocol['parents'][parent])
    all_sites = source.plan()
    selected = [all_sites[0], all_sites[-1]]
    fresh, restored = [], []
    for site in selected:
        direct = prepare_observations(source.extract(site))
        payload = read_cache(protocol['cache_namespace'], parent, protocol['parents'][parent], site)
        assert payload is not None
        state = decode(payload)
        compare(prepared_to_state(direct), state)
        fresh.append(direct)
        restored.append(prepared_from_state(state))
    model = JointAncestryModel(**protocol['model']).eval()
    checkpoint = RESULTS / 'smoke_v2/cpu_smoke_state.pt'
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
    with torch.inference_mode():
        expected = model(collate_prepared(fresh), mode='structure')
        actual = model(collate_prepared(restored), mode='structure')
    assert torch.equal(expected, actual) and torch.isfinite(actual).all()
    source.check_stats()
    assert not torch.cuda.is_initialized()
    receipt = dict(passed=True, protocol_ref=ref(BASE / 'protocol_v2.json'), source_refs=source_refs(),
       smoke_ref=ref(BASE / 'smoke_result_v2.json'), learned_checkpoint_ref=ref(checkpoint),
       unchanged_model_inputs_splits_targets_optimizer_and_epochs=True,
       canonical_train_parents=20, canonical_train_targets=200000, coordinate_windows=158476,
       updates_per_epoch=batches, updates_13_epochs=batches * 13,
       exact_rational_expected_target_weight='1 / 200000',
       cache_vs_fresh_windows=[site['window_index'] for site in selected],
       dense_and_sparse_tensor_dtype_shape_and_bytes_equal=True,
       learned_checkpoint_logits_bitwise_equal=True, heldout_payloads_opened=False,
       truth_payloads_opened=False, CUDA_initialized=False, signals_sent=0, production_jobs_launched=0)
    destination = BASE.parent / 'review_finish/american_v2_checks.json'
    with destination.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(passed=True, receipt_ref=ref(destination), updates_per_epoch=batches,
                         checkpoint_logits_rows=len(actual), independent_train_windows=receipt['cache_vs_fresh_windows'])))


if __name__ == '__main__':
    main()

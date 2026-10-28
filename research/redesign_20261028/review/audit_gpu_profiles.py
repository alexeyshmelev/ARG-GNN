"""Independent audit of saved GPU profiles; no inference, fitting or CUDA use."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/redesign_20261028'
PROFILES = BASE / 'root/gpu_profiles'
REVIEW = BASE / 'review'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


assert not torch.cuda.is_initialized()
protocol_path = BASE / 'root/gpu_profile_protocol.json'
protocol = read(protocol_path)
prelaunch = read(REVIEW / 'GPU_PROFILE_PRELAUNCH_REVIEW.json')
learning_review = read(REVIEW / 'LEARNING_RESULT_REVIEW.json')
assert sha(protocol_path) == prelaunch['protocol_sha256']
assert sha(BASE / 'root/gpu_profile.py') == prelaunch['harness_sha256']
assert sha(BASE / 'architecture/window_model_frozen_v1.py') == protocol['v1_sha256']
assert sha(BASE / 'architecture/window_model_focal_executor_v2.py') == protocol['v2_sha256']
manifest_path = BASE / 'diagnostic_packets/manifest.json'
assert sha(manifest_path) == protocol['manifest_sha256']
manifest = read(manifest_path)
thresholds = protocol['numerical_checks']
assert protocol['warmup_repetitions'] == 2 and protocol['measured_repetitions'] == 20
assert thresholds == {
    'logits_atol': 2e-5,
    'logits_rtol': 2e-4,
    'full_gradient_relative_l2_max': 1e-4,
    'gradient_presence_identical': True,
    'meaning': 'FP32 functional checks, not bitwise optimizer trajectory parity',
}

attempt = read(BASE / 'root/gpu_profile_attempt2_receipt.json')
failure_log = ROOT / attempt['failure_log']
assert sha(failure_log) == attempt['attempt1_log_sha256']
assert "ModuleNotFoundError: No module named 'pynvml'" in failure_log.read_text()
assert (PROFILES / 'series_started').is_dir()
assert not (PROFILES / 'series_completed.txt').exists()
assert (PROFILES / 'series_started_attempt2').is_dir()
assert (PROFILES / 'series_completed_attempt2.txt').read_text().strip() == 'All eight profiles exited zero.'
assert attempt['dependency'] == 'nvidia-ml-py==12.560.30'
assert attempt['dependency_directory'] == '.runtime/vendor/nvml_profile_20261028'
vendor = ROOT / attempt['dependency_directory']
metadata = (vendor / 'nvidia_ml_py-12.560.30.dist-info/METADATA').read_text()
assert '\nName: nvidia-ml-py\n' in metadata and '\nVersion: 12.560.30\n' in metadata
assert sha(vendor / 'pynvml.py') == attempt['pynvml_source_sha256']
launcher = ROOT / attempt['new_launcher']
assert sha(launcher) == attempt['launcher_sha256']
assert 'export PYTHONPATH=.runtime/vendor/nvml_profile_20261028:src:.' in launcher.read_text()
assert attempt['harness_unchanged'] and attempt['protocol_unchanged']

selected = manifest['fit'][:max(protocol['batch_sizes'])]
labels = []
for row in selected:
    assert row['parent_id'] in ['ancestralpaths_european_train_000', 'ancestralpaths_european_train_001']
    assert sha(manifest_path.parent / row['packet_path']) == row['packet_sha256']
    label_path = manifest_path.parent / row['label_path']
    assert sha(label_path) == row['label_sha256']
    label = np.load(label_path, allow_pickle=False)
    assert label.shape == () and label.dtype == np.int64 and int(label) == row['label']
    labels.append(int(label))
labels = np.asarray(labels, dtype=np.int64)

checkpoints = {}
for arm in protocol['arms']:
    run_path = BASE / 'root/learning_runs' / arm / 'result.json'
    assert sha(run_path) == learning_review['arms'][arm]['result_sha256']
    run = read(run_path)
    assert run['updates'] == 400
    checkpoint_path = run_path.parent / 'final_state.pt'
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert all(value.device.type == 'cpu' and torch.isfinite(value).all() for value in state.values())
    digest = state_digest(state)
    assert digest == run['final_parameter_sha256']
    checkpoints[arm] = {
        'checkpoint_sha256': sha(checkpoint_path),
        'parameter_digest': digest,
        'parameter_count': sum(value.numel() for value in state.values()),
        'state_tensor_count': len(state),
    }

records = {}
arrays = {}
audits = {}
expected_names = {
    f'{arm}_{version}_b{size}'
    for arm in protocol['arms']
    for size in protocol['batch_sizes']
    for version in protocol['versions']
}
assert {path.parent.name for path in PROFILES.glob('*/result.json')} == expected_names
for name in sorted(expected_names):
    folder = PROFILES / name
    result = read(folder / 'result.json')
    arm, version, size = result['arm'], result['version'], result['batch_size']
    assert name == f'{arm}_{version}_b{size}'
    assert arm in protocol['arms'] and version in protocol['versions'] and size in protocol['batch_sizes']
    assert result['physical_gpu'] == protocol['physical_gpu'] == 3
    assert result['gpu_uuid'] == protocol['uuid'] == prelaunch['gpu_uuid']
    assert result['process_title'] == 'AI Auto Research Pipeline Test'
    assert result['torch_version'] == '2.6.0+cu124'
    assert result['allow_tf32_matmul'] is False and result['float32_matmul_precision'] == 'highest'
    assert result['protocol_sha256'] == sha(protocol_path)
    assert result['harness_sha256'] == prelaunch['harness_sha256']
    assert result['model_v1_sha256'] == protocol['v1_sha256']
    assert result['executor_v2_sha256'] == protocol['v2_sha256']
    assert result['manifest_sha256'] == protocol['manifest_sha256']
    assert result['checkpoint_sha256'] == checkpoints[arm]['checkpoint_sha256']
    assert result['target_ids'] == [[row['parent_id'], row['example_index']] for row in selected[:size]]
    assert result['optimizer_updates'] == 0 and result['accuracy_evaluated'] is False
    durations = result['forward_backward_seconds']
    assert len(durations) == protocol['measured_repetitions']
    assert np.isfinite(durations).all() and min(durations) > 0
    assert result['median_seconds'] == statistics.median(durations)
    assert result['min_seconds'] == min(durations) and result['max_seconds'] == max(durations)
    for field in ['preparation_seconds', 'transfer_seconds', 'profile_loss']:
        assert np.isfinite(result[field]) and result[field] >= 0
    assert 0 < result['allocated_peak_bytes'] <= result['reserved_peak_bytes']
    samples = result['device_activity_samples']
    assert samples
    times = [sample['monotonic'] for sample in samples]
    assert np.isfinite(times).all() and np.all(np.diff(times) > 0)
    for sample in samples:
        assert 0 <= sample['gpu_percent'] <= 100
        assert 0 <= sample['memory_controller_percent'] <= 100
        assert sample['device_memory_used'] > 0
    qualified = bool(samples) and not result['sampling_errors']
    qualified = qualified and not result['foreign_compute_pids_before'] and not result['foreign_compute_pids_after']
    qualified = qualified and not any(sample['foreign_compute_pids'] for sample in samples)
    assert qualified and result['controlled_device_activity_qualified'] is True
    with np.load(folder / 'numerics.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'logits', 'gradients', 'gradient_present'}
        current = {key: saved[key].copy() for key in saved.files}
    logits = current['logits']
    assert logits.shape == (size, 6) and logits.dtype == np.float32 and np.isfinite(logits).all()
    gradients = current['gradients']
    assert gradients.shape == (checkpoints[arm]['parameter_count'],)
    assert gradients.dtype == np.float32 and np.isfinite(gradients).all()
    presence = current['gradient_present']
    assert presence.shape == (checkpoints[arm]['state_tensor_count'],) and presence.dtype == bool
    values = logits.astype(np.float64)
    maximum = values.max(axis=1)
    cross_entropy = float(np.mean(maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1)) - values[np.arange(size), labels[:size]]))
    assert np.isclose(cross_entropy, result['profile_loss'], rtol=1e-6, atol=2e-7)
    log = json.loads((PROFILES / f'{name}.attempt2.log').read_text())
    assert all(log[key] == result[key] for key in log)
    records[name] = result
    arrays[name] = current
    audits[name] = {
        'result_sha256': sha(folder / 'result.json'),
        'numerics_sha256': sha(folder / 'numerics.npz'),
        'log_sha256': sha(PROFILES / f'{name}.attempt2.log'),
        'finite_saved_logits_and_gradients': True,
        'harness_completed_finite_checks_each_of_22_iterations': True,
        'finite_measured_durations': len(durations),
        'activity_sample_count': len(samples),
        'controlled_device_activity_qualified': qualified,
        'checkpoint_targets_and_precision_match': True,
        'independent_cross_entropy': cross_entropy,
        'cross_entropy_absolute_difference': abs(cross_entropy - result['profile_loss']),
        'median_seconds': result['median_seconds'],
        'allocated_peak_bytes': result['allocated_peak_bytes'],
        'reserved_peak_bytes': result['reserved_peak_bytes'],
    }

ordered_names = [f'{arm}_{version}_b{size}' for arm in protocol['arms'] for size in protocol['batch_sizes'] for version in protocol['versions']]
assert len({records[name]['pid'] for name in ordered_names}) == 8
for previous, following in zip(ordered_names, ordered_names[1:]):
    assert records[previous]['device_activity_samples'][-1]['monotonic'] < records[following]['device_activity_samples'][0]['monotonic']

pairs = {}
for arm in protocol['arms']:
    for size in protocol['batch_sizes']:
        name1, name2 = f'{arm}_v1_b{size}', f'{arm}_v2_b{size}'
        result1, result2 = records[name1], records[name2]
        first, second = arrays[name1], arrays[name2]
        for field in ['checkpoint_sha256', 'target_ids', 'nodes', 'tree_occurrences', 'site_occurrences', 'torch_version', 'allow_tf32_matmul', 'float32_matmul_precision']:
            assert result1[field] == result2[field]
        reference_logits = first['logits'].astype(np.float64)
        candidate_logits = second['logits'].astype(np.float64)
        logits_difference = np.abs(candidate_logits - reference_logits)
        logit_budget = thresholds['logits_atol'] + thresholds['logits_rtol'] * np.abs(reference_logits)
        logits_passed = bool(np.all(logits_difference <= logit_budget))
        reference_gradient = first['gradients'].astype(np.float64)
        candidate_gradient = second['gradients'].astype(np.float64)
        gradient_difference = candidate_gradient - reference_gradient
        gradient_norm = float(np.linalg.norm(reference_gradient))
        assert gradient_norm > 0
        gradient_relative_l2 = float(np.linalg.norm(gradient_difference) / gradient_norm)
        gradient_passed = gradient_relative_l2 <= thresholds['full_gradient_relative_l2_max']
        presence_identical = bool(np.array_equal(first['gradient_present'], second['gradient_present']))
        pair_passed = logits_passed and gradient_passed and presence_identical
        pairs[f'{arm}_b{size}'] = {
            'pass': pair_passed,
            'max_absolute_logit_difference': float(logits_difference.max()),
            'maximum_fraction_of_logit_tolerance': float(np.max(logits_difference / logit_budget)),
            'logits_threshold_passed': logits_passed,
            'full_gradient_relative_l2': gradient_relative_l2,
            'full_gradient_threshold_passed': gradient_passed,
            'reference_gradient_l2': gradient_norm,
            'max_absolute_gradient_difference': float(np.max(np.abs(gradient_difference))),
            'gradient_presence_identical': presence_identical,
            'gradient_tensors_present': int(first['gradient_present'].sum()),
            'matched_target_ids': result1['target_ids'],
            'v1_median_seconds': result1['median_seconds'],
            'v2_median_seconds': result2['median_seconds'],
            'observed_median_time_reduction_percent': 100 * (1 - result2['median_seconds'] / result1['median_seconds']),
            'observed_peak_allocated_reduction_percent': 100 * (1 - result2['allocated_peak_bytes'] / result1['allocated_peak_bytes']),
        }

total_output_bytes = sum(path.stat().st_size for path in PROFILES.rglob('*') if path.is_file())
assert total_output_bytes <= protocol['max_output_bytes']
assert not torch.cuda.is_initialized()
receipt = {
    'schema': 'independent_gpu_profile_result_review_v1',
    'reviewed_utc': datetime.now(timezone.utc).isoformat(),
    'verdict': 'pass' if all(pair['pass'] for pair in pairs.values()) else 'fail',
    'audit_source_sha256': sha(Path(__file__)),
    'protocol_sha256': sha(protocol_path),
    'harness_sha256': prelaunch['harness_sha256'],
    'prospective_thresholds': thresholds,
    'checkpoints': checkpoints,
    'profiles': audits,
    'pairs': pairs,
    'largest_absolute_logit_difference': max(pair['max_absolute_logit_difference'] for pair in pairs.values()),
    'largest_full_gradient_relative_l2': max(pair['full_gradient_relative_l2'] for pair in pairs.values()),
    'all_8_profiles_finite_and_activity_qualified': True,
    'all_4_pairs_match_checkpoints_targets_and_precision': True,
    'sequential_recorded_activity_windows': True,
    'total_profile_output_bytes': total_output_bytes,
    'dependency_correction': {
        'attempt2_receipt_sha256': sha(BASE / 'root/gpu_profile_attempt2_receipt.json'),
        'attempt1_failure_log_sha256': sha(failure_log),
        'attempt1_preserved': True,
        'attempt1_failure_before_torch_import_and_CUDA': True,
        'dependency': attempt['dependency'],
        'project_only_directory': attempt['dependency_directory'],
        'pynvml_source_sha256': attempt['pynvml_source_sha256'],
        'attempt2_launcher_sha256': sha(launcher),
        'model_harness_protocol_unchanged': True,
    },
    'limitations': [
        'Prospective FP32 functional parity passes only on the four saved GPU comparisons; this is not bitwise optimizer or trajectory parity.',
        'Earlier strict CPU one-step AdamW parameter failures remain valid and are not superseded.',
        'All 22 per-profile finite checks are established by the pinned successful harness path; only the final logits and full gradient are retained for independent array verification.',
        'One fixed-order series with repeated fixed batches does not establish streaming training speedup, order-independent performance, or sustained 80 percent useful GPU utilization.',
        'NVML activity qualification covers sampled foreign compute activity; it does not prove absence of all external device work between samples.',
        'Timing excludes optimizer updates, dataset streaming, gradient reset and finite checks; memory peaks include transfer, warmup and measured iterations.',
    ],
    'reviewer_GPU_runs': 0,
    'reviewer_CUDA_initialized': False,
    'reviewer_fitting_or_inference': False,
    'TEST_read': False,
}
(REVIEW / 'GPU_PROFILE_RESULT_REVIEW.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: receipt[key] for key in ['verdict', 'largest_absolute_logit_difference', 'largest_full_gradient_relative_l2', 'all_8_profiles_finite_and_activity_qualified', 'all_4_pairs_match_checkpoints_targets_and_precision', 'total_profile_output_bytes']}, indent=2))
for name, pair in pairs.items():
    print(name, json.dumps({key: value for key, value in pair.items() if key != 'matched_target_ids'}))
assert receipt['verdict'] == 'pass'

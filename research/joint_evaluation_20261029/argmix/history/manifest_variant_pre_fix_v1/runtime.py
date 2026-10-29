"""Frozen successor protocol, original producer waits, and separate bounded outputs."""
from datetime import datetime, timezone
from io import BytesIO
import fcntl
import json
import os
import torch
import numpy as np
from arggnn.benchmarking.paper_argmix import paper_argmix_generator as native
from .common import (ROOT, BASE, CAMPAIGN, CLASSES, MODEL_REF, RECIPE_REF, path, ref,
                     verify, read, require, json_data)

RESULTS = ROOT / 'experiments/results/joint_argmix_20261029'
CACHE = ROOT / 'experiments/cache/joint_argmix_20261029'
MODEL_ID = 'treeloom_joint_argmix_structure_v1'
RESULT_CAP = 2**30
RAM_CAP = 64 * 2**30
IDENTITY = ('record_ids', 'sample_nodes', 'sample_indices', 'positions_bp0', 'parent_id', 'class_order')
RUNTIME = ('common.py', 'observations.py', 'runtime.py', 'cache.py', 'pipeline.py', 'gpu_qualification.py', 'technical_test.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def tree_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def output_guard(required_bytes=0):
    require(0 <= required_bytes <= RESULT_CAP and tree_bytes(RESULTS) + required_bytes <= RESULT_CAP,
            'Dedicated ARGMix results1GiB cap before write')


def write_bytes(filename, payload):
    filename = path(filename)
    require(filename.is_relative_to(RESULTS), 'Only separately reserved ARGMix result outputs')
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / 'writer.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output_guard(len(payload))
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return native.file_ref(filename)


def write(filename, data):
    return write_bytes(filename, (json.dumps(json_data(data), sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def save_torch(filename, state):
    stream = BytesIO()
    torch.save(state, stream)
    return write_bytes(filename, stream.getvalue())


def save_numpy(filename, array):
    stream = BytesIO()
    np.save(stream, array, allow_pickle=False)
    return write_bytes(filename, stream.getvalue())


def require_wait(completed, expected_task=None):
    completed = path(completed)
    receipt_path = completed.parent / 'actual_wait.json'
    require(completed.is_file() and receipt_path.is_file(), 'Prerequisite producer and actual wait must exist: ' + str(completed))
    wait = read(ref(receipt_path))
    require(wait['actual_wait'] is True and wait['actual_returncode'] == 0 and wait['owned_process_group_absent'] is True,
            'Actual successful producer wait and absent owned process group')
    if 'status' in wait:
        require(wait['status'] == 'succeeded', 'Successful original producer status')
    require(path(wait['result_ref']) == completed, 'Wait binds exact prerequisite output')
    verify(wait['result_ref'])
    if expected_task is not None and 'task_id' in wait:
        require(wait['task_id'] == expected_task, 'Exact prerequisite task ID')
    return native.file_ref(completed), ref(receipt_path)


def parent_item(parent, protocol):
    require(parent in protocol['parents'], 'Parent belongs to original recipe')
    item = protocol['parents'][parent]
    observation_ref, wait_ref = require_wait(CAMPAIGN / parent / 'treeloom_observation/completed.json', f'paper_argmix_{parent}.treeloom_observation')
    observation = read(observation_ref)
    require(observation['parent_id'] == parent and observation['split'] == item['split'] and observation['private_truth_opened'] is False,
            'Original same-parent truth-free observation seal')
    return dict(split=item['split'], observation_ref=observation_ref, observation_wait_ref=wait_ref)


def train_targets(parent, protocol, source):
    require(protocol['parents'][parent]['split'] == 'train', 'Held-out labels prohibited in fitting')
    label_ref, wait_ref = require_wait(CAMPAIGN / parent / 'labels/completed.json', f'paper_argmix_{parent}.labels')
    product = read(label_ref)
    require(product['split'] == 'train' and product['parent_id'] == parent and product['examples'] == 100000 and
            product['class_order'] == list(CLASSES) and product['support_manifest_ref'] == source.seal['support_ref'] and
            product['config_ref'] == native.file_ref(path(RECIPE_REF)), 'Exact original same-support TRAIN label product')
    labels = np.fromfile(verify(product['labels_ref']), dtype=np.uint8)
    require(labels.shape == (100000,) and (labels < 4).all(), 'All original TRAIN labels')
    return labels.astype(np.int64), dict(product_ref=label_ref, actual_wait_ref=wait_ref)


def source_refs():
    codec = ROOT / 'research/joint_evaluation_20261029/american'
    sources = [ref(BASE / name) for name in RUNTIME]
    sources += [MODEL_REF, RECIPE_REF, ref(ROOT / 'research/redesign_20261028/data/window_data.py')]
    sources += [ref(codec / name) for name in ('state_codec_v2.py', 'compression_v2.py', 'weighting_v2.py', 'common.py')]
    sources += [native.file_ref(ROOT / 'src/arggnn/benchmarking/paper_argmix' / name) for name in
                ('paper_argmix_score.py', 'paper_argmix_generator.py', 'paper_argmix_features.py')]
    return sources


def frozen(filename):
    seal = read(ref(filename))
    require(seal['schema'] == 'argmix_joint_runtime_freeze_v1', 'Exact new joint ARGMix freeze')
    for source in seal['source_refs']:
        verify(source)
    protocol = read(seal['protocol_ref'])
    require(protocol['model_id'] == MODEL_ID and protocol['model'] == dict(width=64, layers=3, classes=4), 'Frozen joint architecture')
    require(protocol['epochs'] == 5 and protocol['training_target_presentations'] == 2500000 and protocol['training_targets'] == 500000,
            'Five complete TRAIN epochs matching prospectively chosen exposure')
    require(protocol['loss_weight'] == 'W_parent*B_total/(B_parent*S*N)', 'Exact corrected expected target objective')
    require(read(seal['technical_test_ref'])['passed'] is True and read(seal['observation_smoke_ref'])['passed'] is True, 'Actual TRAIN technical tests')
    return seal, protocol


def cache_admission(filename, protocol):
    require(filename is not None, 'Production cache requires a fresh actual root storage admission')
    require(path(filename) == ROOT / 'research/joint_evaluation_20261029/root/argmix_joint_production_admission_v1.json', 'Only exact root production admission path')
    admission = read(ref(filename))
    require(admission['schema'] == 'joint_argmix_cache_admission_v1' and admission['admitted'] is True and admission['root_confirmed'] is True,
            'Actual admitted root cache record; proposal is insufficient')
    require(path(admission['cache_scope']) == CACHE and path(admission['results_scope']) == RESULTS and
            admission['cache_max_bytes'] >= protocol['required_cache_allocation_bytes'] and
            admission['results_max_bytes'] == RESULT_CAP and admission['owned_RAM_max_bytes'] == RAM_CAP,
            'Sufficient exact successor scopes and limits')
    require(admission['combined_reserved_bytes'] <= 1200000000000 and admission['owner_ceiling_bytes'] == 1200000000000,
            'Actual project commitment below owner1.2TB limit')
    ledger = read(admission['actual_ledger_receipt_ref'])
    require(ledger['admitted'] is True and ledger['cache_scope'] == admission['cache_scope'] and
            ledger['cache_max_bytes'] == admission['cache_max_bytes'] and
            ledger['combined_reserved_bytes'] == admission['combined_reserved_bytes'],
            'Admission binds independently recorded actual ledger operation')
    require(admission['protocol_ref'] == ref(BASE / 'protocol_v1.json'), 'Exact prospective protocol allocation')
    return admission

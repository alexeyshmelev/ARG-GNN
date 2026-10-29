"""Successor runtime boundaries; historical v1 sources remain unchanged."""
from io import BytesIO
import json
import os
import torch
import numpy as np
from .common import (BASE, ROOT, MODEL_ID, CLASSES, IDENTITY, MODEL_SOURCE, path, ref, verify, read,
                     load_npz, require, now, json_data)

RESULTS = ROOT / 'experiments/results/joint_american_20261029'
CACHE = ROOT / 'experiments/cache/joint_american_20261029'
RESULT_CAP = 2**30
CACHE_CAP = 63 * 2**30
ALLOCATION = ROOT / 'research/joint_evaluation_20261029/root/american_additive_allocation_v1.json'
RUNTIME = ('common_v2.py', 'pipeline_v2.py', 'weighting_v2.py', 'compression_v2.py',
           'state_codec_v2.py', 'cache_v2.py', 'smoke_v2.py')


def tree_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def check_disk():
    size = tree_bytes(RESULTS)
    require(size <= RESULT_CAP, 'American results exceed1GiB reserved scope')
    return size


def output_bytes(value, payload):
    destination = path(value)
    require(destination.is_relative_to(RESULTS) or destination.parent == BASE, 'Only bounded American result or top-level small metadata outputs')
    cap = RESULT_CAP if destination.is_relative_to(RESULTS) else 16 * 2**20
    current = check_disk() if destination.is_relative_to(RESULTS) else destination.parent.stat().st_size
    require(current + len(payload) <= cap, 'Output must fit before write')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return ref(destination)


def write(value, payload):
    return output_bytes(value, (json.dumps(json_data(payload), indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def save_npz(value, arrays):
    stream = BytesIO()
    np.savez(stream, **arrays)
    return output_bytes(value, stream.getvalue())


def save_torch(value, state):
    stream = BytesIO()
    torch.save(state, stream)
    return output_bytes(value, stream.getvalue())


def source_refs():
    from .common import source_refs as original_refs
    return original_refs() + [ref(BASE / name) for name in RUNTIME]


def frozen(value):
    seal = json.loads(path(value).read_text())
    require(seal['schema'] == 'american_joint_runtime_freeze_v2', 'Successor runtime freeze required')
    for reference in seal['source_refs']:
        verify(reference)
    protocol = read(seal['protocol_ref'])
    require(protocol['model_id'] == MODEL_ID and protocol['model'] == dict(width=64, layers=3, classes=3), 'Exact frozen architecture')
    require(protocol['epochs'] == 13 and protocol['training_target_presentations'] == 2600000, 'Fixed13 complete TRAIN epochs')
    require(protocol['loss_weight'] == 'W_parent*B_total/(B_parent*S*N)', 'Correct parent-tail expected objective')
    require(read(seal['smoke_ref'])['passed'] is True, 'Successor CPU integration gate')
    allocation = read(seal['allocation_ref'])
    require(allocation['cache_max_bytes'] == CACHE_CAP and allocation['results_max_bytes'] == RESULT_CAP and
            path(allocation['cache_scope']) == CACHE and path(allocation['results_scope']) == RESULTS,
            'Exact additional64GiB root allocation')
    return seal, protocol

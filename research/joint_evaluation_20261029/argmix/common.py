"""Explicit identities and authenticated immutable source reads."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/joint_evaluation_20261029/argmix'
CAMPAIGN = ROOT / 'experiments/paper_argmix_1611_rebuild_20261024_v1'
CLASSES = ('Ana', 'CHG', 'WHG', 'EHG')
COHORTS = {'Present': 0, 'PostNeolithic': 1, 'Yam': 3, 'NF': 4,
           'WHG': 5, 'EHG': 6, 'Ana': 7, 'CHG': 8}
MODEL_REF = {'path': 'research/joint_model_20261029/model/joint_model_v1.py',
             'sha256': 'ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4'}
RECIPE_REF = {'path': 'research/benchmark_rebuild_20261024/argmix/paper_argmix_recipe.json',
              'sha256': '8e6b5edf7adea653bdae1b8a8e7009892344821afcaaa5ca0e869069aa605564'}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def path(value):
    value = value['path'] if isinstance(value, dict) else value
    result = Path(value)
    result = (ROOT / result).resolve() if not result.is_absolute() else result.resolve()
    require(result.is_relative_to(ROOT), 'Project containment')
    return result


def ref(value):
    source = path(value)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            digest.update(block)
    return {'path': str(source.relative_to(ROOT)), 'sha256': digest.hexdigest()}


def verify(reference):
    source = path(reference)
    require(ref(source)['sha256'] == reference['sha256'], 'Source hash changed: ' + str(source))
    if 'bytes' in reference:
        require(source.stat().st_size == reference['bytes'], 'Source byte size changed')
    return source


def read(reference, cap=16 * 2**20):
    require(path(reference).stat().st_size <= cap, 'Bounded metadata size')
    return json.loads(verify(reference).read_text())


def json_data(value):
    if isinstance(value, np.ndarray):
        return json_data(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_data(v) for v in value]
    return value


def write(filename, data):
    filename = path(filename)
    require(filename.is_relative_to(BASE), 'Only own compact integration reports')
    with filename.open('x') as stream:
        stream.write(json.dumps(json_data(data), indent=2, sort_keys=True, allow_nan=False) + '\n')
    return ref(filename)

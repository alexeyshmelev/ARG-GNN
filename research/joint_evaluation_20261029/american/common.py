"""Authenticated project inputs and exclusive, bounded American outputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/joint_evaluation_20261029/american'
MODEL_ID = 'treeloom_joint_american_structure_v1'
CLASSES = ('EUR', 'EAS', 'AFR')
IDENTITY = ('record_ids', 'sample_nodes', 'sample_indices', 'positions_bp0', 'parent_id', 'class_order')
MODEL_SOURCE = dict(path='research/joint_model_20261029/model/joint_model_v1.py',
                    sha256='ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4')
RUNTIME = ('__init__.py', 'common.py', 'inventory.py', 'observations.py', 'targets.py', 'pipeline.py', 'smoke.py')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def path(value):
    value = value['path'] if isinstance(value, dict) else value
    result = Path(value)
    result = (ROOT / result).resolve() if not result.is_absolute() else result.resolve()
    require(result.is_relative_to(ROOT), 'All files remain inside the project')
    return result


def ref(value):
    source = path(value)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return dict(path=str(source.relative_to(ROOT)), sha256=digest.hexdigest())


def verify(reference):
    require(ref(reference) == reference, 'Immutable source hash mismatch: ' + reference['path'])
    return path(reference)


def read(reference):
    require(path(reference).stat().st_size <= 16 * 2**20, 'Small metadata cap')
    return json.loads(verify(reference).read_text())


def json_data(value):
    if isinstance(value, np.ndarray):
        return json_data(value.tolist())
    if isinstance(value, np.generic):
        return json_data(value.item())
    if isinstance(value, dict):
        return {k: json_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_data(v) for v in value]
    return value


def write(value, payload):
    destination = path(value)
    require(destination.is_relative_to(BASE), 'American writes restricted to its separate directory')
    serialized = json.dumps(json_data(payload), indent=2, sort_keys=True, allow_nan=False) + '\n'
    with destination.open('x') as stream:
        stream.write(serialized)
    return ref(destination)


def load_npz(reference, cap=8 * 2**20):
    source = verify(reference)
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        require(len(entries) == len({item.filename for item in entries}) and
                sum(item.file_size for item in entries) <= cap, 'Bounded NPZ without duplicate members')
    with np.load(source, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def save_npz(value, arrays):
    destination = path(value)
    require(destination.is_relative_to(BASE), 'American output containment')
    with destination.open('xb') as stream:
        np.savez(stream, **arrays)
    return ref(destination)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_refs():
    dependencies = [MODEL_SOURCE, ref('research/redesign_20261028/data/window_data.py'),
                    ref('src/arggnn/american_c3_common_v1.py'), ref('src/arggnn/american_c3_evaluate_v1.py')]
    return [ref(BASE / name) for name in RUNTIME] + dependencies


def frozen(value):
    seal = json.loads(path(value).read_text())
    for reference in seal['source_refs']:
        verify(reference)
    protocol = read(seal['protocol_ref'])
    require(protocol['model_id'] == MODEL_ID and protocol['model'] == dict(width=64, layers=3, classes=3),
            'Exact frozen American architecture and separate identity')
    require(read(seal['smoke_ref'])['passed'] is True, 'Passing CPU integration gate required')
    return seal, protocol

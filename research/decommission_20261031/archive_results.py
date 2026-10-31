"""Preserve compact project evidence before the authorized server-folder deletion.

Archives are backup data, never executable transport. No simulations or training
are run. Source paths and SHA256 checksums accompany every archived file.
"""
from pathlib import Path
import collections
import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/decommission_20261031'
assert Path.cwd() == ROOT
LIMIT = 20 * 1024 * 1024
TEXT = {'.py', '.sh', '.md', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.csv', '.tsv', '.txt', '.tex', '.bib', '.svg', '.pdf', '.sty', '.bst', '.diff', '.patch', '.ipynb', '.html'}
SKIP_PARTS = {'.git', '.cache', '.tmp', '__pycache__', '.mpl-cache'}
tracked = set(subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0'))
rows = json.loads((OUT / 'compact_evidence_selection.json').read_text())
selected = {row['path']: row['bytes'] for row in rows}
ignored_scopes = ['results', 'checkpoints', 'experiments/results', 'paper']
for scope in ignored_scopes:
    names = subprocess.check_output(['git', 'ls-files', '--others', '--ignored', '--exclude-standard', '-z', '--', scope]).decode().split('\0')
    for name in names:
        p = Path(name)
        if not name or p.suffix not in TEXT or SKIP_PARTS.intersection(p.parts):
            continue
        info = p.lstat()
        if stat.S_ISREG(info.st_mode) and info.st_size <= LIMIT:
            selected[name] = info.st_size
# Keep historical queue receipts and operational source, but not environments or keys.
for directory, dirs, files in os.walk(ROOT / '.runtime/benchmarking', followlinks=False):
    dirs[:] = [d for d in dirs if d not in SKIP_PARTS and not (Path(directory) / d).is_symlink()]
    for name in files:
        p = Path(directory) / name
        if p.suffix not in TEXT or p.is_symlink():
            continue
        info = p.stat()
        if stat.S_ISREG(info.st_mode) and info.st_size <= LIMIT:
            selected[str(p.relative_to(ROOT))] = info.st_size
# Current trained models and prediction tensors are small enough to preserve.
# Input caches and full competitor optimizer checkpoints remain outside this backup.
scopes = ['experiments/results/joint*', 'research/joint_evaluation_20261029',
          'research/joint_model_20261029', 'experiments/results/paper_american_20261024',
          'experiments/paper_argmix_1611_rebuild_20261024_v1', 'results', 'checkpoints']
artifacts = []
for scope in scopes:
    names = []
    for flags in [[], ['--ignored']]:
        names += subprocess.check_output(['git', 'ls-files', '--others', *flags, '--exclude-standard', '-z', '--', scope]).decode().split('\0')
    for name in names:
        p = Path(name)
        if not name or p.suffix not in {'.pt', '.pth', '.keras', '.h5', '.npz', '.npy'}:
            continue
        if any(t in name.lower() for t in ['/cache', '/scratch', '/.cache', '/.tmp', 'fixture', '_test_scratch', 'smoke', '/labels/', '/windows/', '/graphs/', '/tensors/', '/observed/', '/truth/', '/inferred/', '/features/']):
            continue
        is_result = any(t in name.lower() for t in ['predict', '/fit/', '/fit_', '/train/', '/training/', '/scores/', '/score/', 'checkpoint', 'final_state', 'selected.pt', 'best', '/outputs/'])
        if not is_result:
            continue
        info = p.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > LIMIT:
            continue
        if name not in tracked:
            selected[name] = info.st_size
            artifacts.append({'path': name, 'bytes': info.st_size})
selected = {name: size for name, size in selected.items() if not name.startswith('research/decommission_20261031/')}
print('SELECTED', len(selected), 'BYTES', sum(selected.values()), 'MODEL_AND_PREDICTION_FILES', len(artifacts), flush=True)
(OUT / 'artifact_selection.json').write_text(json.dumps(artifacts, indent=2) + '\n')
# Split by uncompressed size so every individual archive stays below GitHub's limit.
archives = []
manifest = gzip.open(OUT / 'archive_manifest.jsonl.gz', 'wt', encoding='utf-8')
archive = None
archive_path = None
part_bytes = 0
part_files = 0
scanned = 0
secret_pattern = re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}')

def finish_part():
    global archive
    if archive is None:
        return
    archive.close()
    data = archive_path.read_bytes()
    assert len(data) < 95 * 1024 * 1024
    archives.append({'path': str(archive_path.relative_to(ROOT)), 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest(), 'files': part_files})
    print('ARCHIVE', archive_path.name, len(data), 'FILES', part_files, flush=True)
    archive = None

for name in sorted(selected):
    p = ROOT / name
    assert not p.is_symlink() and ROOT in p.resolve().parents
    data = p.read_bytes()
    if p.suffix not in {'.pdf', '.pt', '.pth', '.npz', '.npy', '.keras', '.h5'} and secret_pattern.search(data):
        raise RuntimeError('Credential marker detected; do not publish: ' + name)
    if archive is None or part_bytes + len(data) > 64 * 1024 * 1024:
        finish_part()
        archive_path = OUT / f'evidence_{len(archives) + 1:03d}.tar.gz'
        archive = tarfile.open(archive_path, 'w:gz', compresslevel=3)
        part_bytes = 0
        part_files = 0
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = int(p.stat().st_mtime)
    archive.addfile(info, io.BytesIO(data))
    manifest.write(json.dumps({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                               'archive': archive_path.name}, sort_keys=True) + '\n')
    part_bytes += len(data)
    part_files += 1
    scanned += 1
    if scanned % 2000 == 0:
        print('ARCHIVED_FILES', scanned, flush=True)
finish_part()
manifest.close()
summary = {'status': 'archives_created_pending_remote_verification', 'files': scanned,
           'uncompressed_bytes': sum(selected.values()), 'compressed_bytes': sum(row['bytes'] for row in archives),
           'archives': archives, 'model_and_prediction_files': len(artifacts),
           'scope': 'Uncommitted compact source, documentation, experiment receipts, metrics, historical queue metadata and selected small model/prediction artifacts.',
           'excluded': ['Raw simulation/input/cache payloads', 'Environment installations and private keys', 'Files larger than 20 MiB that were not already tracked', 'Large third-party training checkpoints', 'Rendering previews and transient stdout/stderr logs unless already tracked'],
           'full_directory_backup': False}
(OUT / 'archive_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('DONE', json.dumps({k: v for k, v in summary.items() if k != 'archives'}), flush=True)

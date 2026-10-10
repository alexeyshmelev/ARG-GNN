"""Versioned, integrity-bound CPU boundary banks for exact clipped ARG tiles.

No registry/runner integration. Labels and observed nearest-TMRCA probabilities
are stored outside bank files. Preparation checks every complete observed parent
before emitting any fresh cache directory. No temporary replacement or deletion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import hashlib
import io
import json
import math
import os
import platform
import stat
import warnings

import torch
from torch import Tensor
from .resources import ROOT, MAX_BYTES, WARN_BYTES, project_path
from .kron_interface import LeafInterfaceBank, precompute_leaf_interfaces, with_query_reference_cache
from .simulate import split_fixed_windows
from .train_study import load_cases

SCHEMA_VERSION = 1
DEPENDENCIES = ('boundary_dataset.py', 'kron_interface.py', 'train_study.py',
                'simulate.py', 'data.py', 'models.py', 'reference_encoder.py', 'genealogy_baselines.py')


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _expect(actual, expected, message: str) -> None:
    if actual != expected:
        raise RuntimeError(message)


def _code_hashes() -> dict[str, str]:
    return {f'src/arggnn/{name}': sha256_file(Path(__file__).with_name(name)) for name in DEPENDENCIES}


_IMPORTED_CODE_SHA256 = _code_hashes()


def scan_repository_storage(directory: Path | str = ROOT) -> int:
    """Conservative logical/allocated scan without du, symlink traversal or deletion.

    Deleted entries have no remaining project allocation and may disappear during
    a scan. Other scan failures abort. New concurrent writes remain subject to
    project-wide guards, a reserved margin and refresh before each parent write.
    Hard-linked paths may be counted repeatedly, conservatively.
    """
    directory = project_path(directory)
    pending, total = [directory], 0
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    total += max(info.st_size, getattr(info, 'st_blocks', 0)*512)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
        except FileNotFoundError:
            continue
    return total


class StorageAdmission:
    """One repository snapshot plus direct-write accounting and parent refreshes."""
    def __init__(self, *, reserve_bytes: int = 1_000_000_000):
        if not isinstance(reserve_bytes, int) or reserve_bytes < 0:
            raise ValueError('reserve_bytes must be nonnegative')
        self.reserve_bytes = reserve_bytes
        self.used_bytes = scan_repository_storage()

    def refresh(self) -> None:
        self.used_bytes = scan_repository_storage()

    def admit(self, payload_bytes: int) -> None:
        if not isinstance(payload_bytes, int) or payload_bytes < 0:
            raise ValueError('payload_bytes must be nonnegative')
        projected = self.used_bytes + payload_bytes + self.reserve_bytes
        if projected >= MAX_BYTES:
            raise RuntimeError(f'Project storage admission rejected: projected {projected} >= {MAX_BYTES}')
        if projected >= WARN_BYTES:
            warnings.warn(f'Project storage approaching limit: projected {projected} bytes', RuntimeWarning, stacklevel=2)
        available = os.statvfs(ROOT)
        free = available.f_bavail*available.f_frsize
        if payload_bytes + self.reserve_bytes >= free:
            raise RuntimeError('Filesystem free-space admission rejected')

    def write(self, path: Path, payload: bytes) -> dict:
        path = project_path(path)
        self.admit(len(payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(payload)
        self.used_bytes += max(len(payload), path.stat().st_blocks*512) + 8192
        return {'sha256': sha256_file(path), 'bytes': path.stat().st_size}


def _torch_bytes(value) -> bytes:
    handle = io.BytesIO()
    torch.save(value, handle)
    return handle.getvalue()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def _bind_source(manifest_path: Path | str, protocol_path: Path | str | None = None,
                 expected_manifest: str | None = None, expected_protocol: str | None = None) -> tuple[dict, dict]:
    path = project_path(manifest_path)
    if path.is_dir():
        path = path/'manifest.json'
    manifest_hash = sha256_file(path)
    if expected_manifest is not None:
        _expect(manifest_hash, expected_manifest, 'Source manifest integrity failure')
    manifest = json.loads(path.read_text())
    protocol_hash = manifest['protocol_sha256']
    if expected_protocol is not None:
        _expect(protocol_hash, expected_protocol, 'Source protocol binding failure')
    if protocol_path is None:
        matches = [p for p in sorted((ROOT/'configs').glob('*.json')) if sha256_file(p) == protocol_hash]
        if not matches:
            raise ValueError('Supply source_protocol_path matching the source manifest hash')
        protocol = matches[0]
    else:
        protocol = project_path(protocol_path)
    _expect(sha256_file(protocol), protocol_hash, 'Source protocol file integrity failure')
    _expect(json.loads(protocol.read_text()), manifest['protocol'], 'Source manifest protocol content mismatch')
    if manifest['protocol'].get('splits', {}).get('test', 0) != 0:
        raise ValueError('Boundary mechanism dataset permits train/validation parents only')
    identities = [(record['split'], record['replicate']) for record in manifest['replicates']]
    if len(identities) != len(set(identities)) or any(split not in {'train', 'val'} for split, _ in identities):
        raise RuntimeError('Duplicate parent identities or unsupported/test split')
    binding = {'manifest_path': str(path), 'manifest_sha256': manifest_hash,
        'protocol_path': str(protocol), 'protocol_sha256': protocol_hash,
        'protocol_content_sha256': _digest(manifest['protocol'])}
    return manifest, binding


def _parent_binding(record: dict, metadata: dict) -> dict:
    directory = project_path(record['directory'])
    fields = [('metadata', 'metadata_sha256', 'metadata.json'),
              ('tensor_file', 'tensor_sha256', 'tensors.pt.gz'),
              ('prepared_tensor_file', 'prepared_tensor_sha256', 'prepared.pt')]
    files = {}
    for path_key, hash_key, filename in fields:
        path = project_path(record[path_key])
        _expect(path, (directory/filename).resolve(), f'Parent {path_key} directory binding failure')
        _expect(sha256_file(path), record[hash_key], f'Parent {path_key} integrity failure')
        files[path_key] = str(path)
        files[hash_key] = record[hash_key]
    return {'split': record['split'], 'replicate': record['replicate'], 'seeds': record['seeds'],
        'master_seed': metadata['master_seed'], 'adapter_configuration_sha256': record['adapter_configuration_sha256'],
        'source_record_sha256': _digest(record), 'inference_seed': metadata.get('inference_seed'),
        'source_parent': record.get('source_parent'), **files}


def _grid_indices(case) -> Tensor:
    p = case.sequence.positions.numel()
    retained = case.metadata.get('retained_parent_grid_indexes', list(range(p)))
    result = torch.tensor(retained, dtype=torch.long)
    if result.shape != (p,) or bool((result < 0).any()) or torch.unique(result).numel() != p:
        raise RuntimeError('Original retained grid index binding invalid')
    parent_count = int(case.metadata.get('parent_grid_points', case.metadata.get('config', {}).get('grid_points', p)))
    if bool((result >= parent_count).any()):
        raise RuntimeError('Original retained grid index outside parent grid')
    return result


def _reference_ids(sequence, bank: LeafInterfaceBank, columns: Iterable[int]) -> Tensor:
    columns = tuple(columns)
    expected = None
    for nodes in bank.boundary_nodes:
        reference = sequence.x[nodes][:, columns]
        if not bool(((reference == 0)|(reference == 1)).all()) or bool((reference.sum(-1) > 1).any()):
            raise RuntimeError('Observed reference roles must be one-hot or zero')
        if expected is None:
            expected = reference
        elif not torch.equal(reference, expected):
            raise RuntimeError('Observed reference identity/category changed across graphs')
    refs = bank.sample_ids[expected.sum(-1) == 1]
    selected = torch.cat((bank.query_node_ids, refs))
    if selected.numel() != bank.sample_ids.numel() or torch.unique(selected).numel() != selected.numel() or not torch.equal(selected.sort().values, bank.sample_ids):
        raise RuntimeError('Observed queries/references do not exhaustively partition sample leaves')
    return refs


def _supervision(case) -> dict:
    p, q = case.sequence.positions.numel(), case.sequence.query_node_ids.numel()
    if case.labels.device.type != 'cpu' or case.labels.dtype != torch.long or case.labels.shape != (p, q):
        raise RuntimeError('External labels must be CPU long [P,Q]')
    k = int(case.metadata['num_classes'])
    if bool(((case.labels < 0)|(case.labels >= k)).any()):
        raise RuntimeError('External label outside declared class count')
    baseline = case.baseline_probabilities
    if baseline.device.type != 'cpu' or baseline.shape != (p, q, k) or not bool(torch.isfinite(baseline).all()) or bool((baseline < 0).any()) or not torch.allclose(baseline.sum(-1), torch.ones(p, q, dtype=baseline.dtype), atol=1e-6, rtol=1e-6):
        raise RuntimeError('External baseline probabilities invalid')
    return {'labels': case.labels.detach().clone(), 'baseline_probabilities': baseline.detach().clone(),
        'positions': case.sequence.positions.detach().clone(), 'query_node_ids': case.sequence.query_node_ids.detach().clone(),
        'original_grid_indices': _grid_indices(case), 'metadata': case.metadata}


def _preparation(manifest: dict, *, windows: int | None, resistance_scale: float,
                 feature_columns: Iterable[int] | None, max_condition: float, splits: Iterable[str]) -> dict:
    adapter = manifest['protocol']['reference_adapter']
    windows = manifest['protocol']['windows'] if windows is None else windows
    splits = tuple(splits)
    if isinstance(windows, bool) or not isinstance(windows, int) or windows < 1:
        raise ValueError('windows must be a positive integer')
    if not splits or len(set(splits)) != len(splits) or any(s not in {'train','val'} for s in splits):
        raise ValueError('Select unique train/validation splits only')
    if adapter.get('branch_column') is None:
        raise ValueError('Source adapter must declare the raw parent-branch column')
    if not math.isfinite(resistance_scale) or resistance_scale <= 0 or not math.isfinite(max_condition) or max_condition < 1:
        raise ValueError('Require positive finite resistance scale and max_condition >= 1')
    return {'schema_version': SCHEMA_VERSION, 'kind': 'exact_clipped_boundary_interface_dataset',
        'windows': windows, 'splits': list(splits), 'time_column': adapter['time_column'],
        'parent_branch_column': adapter['branch_column'], 'time_scale': adapter.get('time_scale',10000.),
        'reference_columns': adapter['reference_columns'], 'source_reference_adapter': adapter,
        'resistance_scale': resistance_scale, 'max_condition': max_condition,
        'feature_columns': None if feature_columns is None else list(feature_columns),
        'normalization': 'mean_boundary_diagonal', 'precompute_dtype': 'float64',
        'full_parent_admissibility_before_windowing': True, 'labels_in_bank': False}


def _bank(sequence, settings: dict) -> LeafInterfaceBank:
    return precompute_leaf_interfaces(sequence, time_column=settings['time_column'],
        parent_branch_column=settings['parent_branch_column'], time_scale=settings['time_scale'],
        resistance_scale=settings['resistance_scale'], feature_columns=settings['feature_columns'],
        max_condition=settings['max_condition'], validate=True)


@dataclass(frozen=True)
class BoundaryTile:
    bank: LeafInterfaceBank
    labels: Tensor
    metadata: dict
    parent_position_indices: Tensor
    provenance: dict


@dataclass(frozen=True)
class BoundaryCase:
    metadata: dict
    labels: Tensor
    baseline_probabilities: Tensor
    tiles: tuple[BoundaryTile, ...]
    positions: Tensor
    query_node_ids: Tensor
    parent_id: str
    provenance: dict


@dataclass(frozen=True)
class VerifiedBoundaryDataset:
    directory: Path
    manifest: dict
    manifest_sha256: str
    preparation_sha256: str
    records: tuple[dict, ...]
    parent_cases: tuple[BoundaryCase, ...]

    def cases(self, split: str) -> list[BoundaryCase]:
        result = [case for case in self.parent_cases if case.metadata['split'] == split]
        if not result:
            raise ValueError(f'No {split} boundary parents')
        return result

    def tiles(self, split: str) -> list[BoundaryTile]:
        return [tile for case in self.cases(split) for tile in case.tiles]


def prepare_boundary_dataset(source_manifest_path: Path | str, output_dir: Path | str, *,
        source_protocol_path: Path | str | None = None, windows: int | None = None,
        splits: Iterable[str] = ('train','val'), resistance_scale: float = 1e-4,
        feature_columns: Iterable[int] | None = None, max_condition: float = 1e12,
        expected_source_manifest_sha256: str | None = None,
        expected_source_protocol_sha256: str | None = None,
        progress: Callable[[dict], None] | None = None) -> VerifiedBoundaryDataset:
    output = project_path(output_dir)
    if output.exists():
        raise RuntimeError('Use a fresh versioned boundary dataset directory')
    code_snapshot = _code_hashes()
    _expect(code_snapshot, _IMPORTED_CODE_SHA256, 'Preparation implementation changed after module import. Start a fresh process')
    source, binding = _bind_source(source_manifest_path, source_protocol_path,
        expected_source_manifest_sha256, expected_source_protocol_sha256)
    settings = _preparation(source, windows=windows, splits=splits, resistance_scale=resistance_scale,
        feature_columns=feature_columns, max_condition=max_condition)
    configuration_hash = _digest(settings)
    record_map = {(r['split'], r['replicate']): r for r in source['replicates']}
    cases = [case for split in settings['splits'] for case in load_cases(source,split)]
    ready = []
    # Reject every inadmissible source tree before emitting any directory. This
    # includes graphs in unscored tiles that split_fixed_windows would omit.
    for case in cases:
        key = (case.metadata['split'], case.metadata['replicate'])
        record = record_map[key]
        _expect(case.metadata['master_seed'], source['protocol']['master_seed'], 'Parent master seed mismatch')
        parent_binding = _parent_binding(record,case.metadata)
        supervision = _supervision(case)
        parent_bank = _bank(case.sequence,settings)
        refs = _reference_ids(case.sequence,parent_bank,settings['reference_columns'])
        admission = {'graphs_checked': case.sequence.intervals.shape[0],
            'all_graphs_admissible': True, 'maximum_interior_condition': max(d.interior_condition_number for d in parent_bank.diagnostics)}
        tiles = split_fixed_windows(case,settings['windows'])
        ready.append((case,parent_binding,supervision,refs,admission,tiles))
        if progress:
            progress({'event':'parent_admissible','split':key[0],'replicate':key[1],**admission})
    _expect(_code_hashes(),code_snapshot,'Preparation implementation changed during full-parent preflight')
    storage = StorageAdmission()
    storage.admit(10_000_000)
    output.mkdir(parents=True)
    header = {'schema_version': SCHEMA_VERSION, 'source': binding, 'preparation': settings,
        'preparation_sha256': configuration_hash, 'implementation_sha256': code_snapshot,
        'environment': {'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},
        'parents':[], 'records':[]}
    storage.write(output/'preparation.json',_json_bytes({**header,'status':'incomplete'}))
    for case,parent_binding,supervision,refs,admission,tiles in ready:
        storage.refresh()
        split, replicate = case.metadata['split'],case.metadata['replicate']
        parent_id = f"{binding['manifest_sha256']}:{split}:{replicate}"
        parent_relative = Path(split)/f'rep{replicate:03d}'
        parent_hash = _digest(parent_binding)
        supervision_path = parent_relative/'supervision.pt'
        super_info = storage.write(output/supervision_path,_torch_bytes({'schema_version':SCHEMA_VERSION,
            'parent_id':parent_id,'parent_binding_sha256':parent_hash,**supervision}))
        parent_record = {'parent_id':parent_id,'binding':parent_binding,'binding_sha256':parent_hash,
            'supervision_file':str(supervision_path),'supervision_sha256':super_info['sha256'],
            'admissibility':admission,'positions':case.sequence.positions.numel(),'queries':case.sequence.query_node_ids.numel(),
            'tile_ids':[]}
        covered = torch.zeros(case.sequence.positions.numel(),dtype=torch.long)
        for tile in tiles:
            index = tile.metadata['window_index']
            left,right = tile.metadata['window_bounds']
            positions = torch.nonzero((case.sequence.positions>=left)&(case.sequence.positions<right),as_tuple=False).flatten()
            _expect(torch.equal(tile.sequence.positions,case.sequence.positions[positions]), True,'Tile coordinate alignment failure')
            _expect(torch.equal(tile.labels,case.labels[positions]),True,'Tile external-label alignment failure')
            covered[positions]+=1
            tile_bank = with_query_reference_cache(_bank(tile.sequence,settings),refs)
            relative = parent_relative/f'window{index:03d}'/'bank.pt'
            tile_id = f'{parent_id}:window{index}'
            grid = supervision['original_grid_indices'][positions]
            metadata = dict(tile.metadata)
            metadata.update(parent_id=parent_id,parent_position_indices=positions.tolist(),
                retained_parent_grid_indexes=grid.tolist())
            provenance = {'parent_id':parent_id,'tile_id':tile_id,'parent_binding_sha256':parent_hash,
                'preparation_sha256':configuration_hash,'window_index':index,'window_bounds':[left,right],
                'global_window_bounds':tile.metadata['global_window_bounds'],
                'parent_position_indices':positions.tolist(),'original_grid_indices':grid.tolist()}
            info = storage.write(output/relative,_torch_bytes({'schema_version':SCHEMA_VERSION,
                'bank':tile_bank.as_dict(),'provenance':provenance}))
            header['records'].append({'bank_file':str(relative),'bank_sha256':info['sha256'],
                'bank_bytes':info['bytes'],'metadata':metadata,**provenance})
            parent_record['tile_ids'].append(tile_id)
            event={'event':'boundary_tile_ready','split':split,'replicate':replicate,'window_index':index,'bytes':info['bytes']}
            if progress:
                progress(event)
        if not bool((covered==1).all()):
            raise RuntimeError('Clipped tiles do not cover every retained parent coordinate exactly once')
        header['parents'].append(parent_record)
    # Recheck the source snapshot before sealing a complete immutable manifest.
    _expect(sha256_file(binding['manifest_path']),binding['manifest_sha256'],'Source manifest changed during preparation')
    _expect(sha256_file(binding['protocol_path']),binding['protocol_sha256'],'Source protocol changed during preparation')
    for case,parent_binding,*_ in ready:
        _expect(_parent_binding(record_map[(case.metadata['split'],case.metadata['replicate'])],case.metadata),
            parent_binding,'Source parent changed during preparation')
    _expect(_code_hashes(),code_snapshot,'Preparation implementation changed before cache sealing')
    storage.write(output/'manifest.json',_json_bytes({**header,'status':'complete'}))
    return load_boundary_dataset(output,expected_manifest_sha256=sha256_file(output/'manifest.json'))



def _tensor_equal(actual: Tensor, expected: Tensor, message: str) -> None:
    if not isinstance(actual,Tensor) or actual.device.type != 'cpu' or actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual,expected):
        raise RuntimeError(message)


def _cached_path(directory: Path, relative: str, expected: str) -> Path:
    _expect(relative,expected,'Cached artifact directory/parent binding failure')
    path = project_path(directory/relative)
    if directory not in path.parents:
        raise RuntimeError('Cached artifact escapes its dataset directory')
    return path


def _verify_eigensystem(matrix: Tensor, values: Tensor, vectors: Tensor) -> None:
    if values.shape != matrix.shape[:-1] or vectors.shape != matrix.shape or not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(vectors).all()):
        raise RuntimeError('Cached eigensystem shape/finite check failed')
    eye = torch.eye(matrix.shape[-1],dtype=matrix.dtype)
    if not torch.allclose(vectors.transpose(-1,-2)@vectors,eye,atol=1e-8,rtol=1e-8):
        raise RuntimeError('Cached eigenbasis is not orthonormal')
    reconstructed = vectors@torch.diag_embed(values)@vectors.transpose(-1,-2)
    if not torch.allclose(reconstructed,matrix,atol=1e-8,rtol=1e-8):
        raise RuntimeError('Cached eigensystem does not reconstruct its response')


def _verify_bank(bank: LeafInterfaceBank, sequence, settings: dict) -> None:
    _expect(bank.geometry_kind,'observed_tree_leaf','Unexpected cache geometry kind')
    _expect(bank.schema_version,SCHEMA_VERSION,'Bank schema mismatch')
    _expect(bank.resistance_scale,settings['resistance_scale'],'Bank resistance scale mismatch')
    _expect(bank.time_scale,float(settings['time_scale']),'Bank time scale mismatch')
    columns = tuple(c for c in range(sequence.x.shape[1]) if c != settings['parent_branch_column']) if settings['feature_columns'] is None else tuple(settings['feature_columns'])
    _expect(bank.feature_columns,columns,'Bank feature selection mismatch')
    if settings['parent_branch_column'] in columns:
        raise RuntimeError('Raw parent-branch feature leaked into bank')
    boundaries=[]
    sample_ids=None
    for graph in range(sequence.intervals.shape[0]):
        nodes=torch.nonzero((sequence.node_graph==graph)&(sequence.node_ids>=0),as_tuple=False).flatten()
        nodes=nodes[torch.argsort(sequence.node_ids[nodes])]
        if sample_ids is None:
            sample_ids=sequence.node_ids[nodes]
        else:
            _tensor_equal(sequence.node_ids[nodes],sample_ids,'Source boundary sample identity drift')
        boundaries.append(nodes)
    boundaries=torch.stack(boundaries)
    _tensor_equal(bank.boundary_nodes,boundaries,'Bank packed-node identity mapping mismatch')
    _tensor_equal(bank.sample_ids,sample_ids,'Bank boundary sample ID mismatch')
    _tensor_equal(bank.leaf_x,sequence.x[boundaries][:,:,columns].double(),'Bank features differ from observed source leaves')
    for field in ('intervals','positions','query_node_ids','target_graph'):
        _tensor_equal(getattr(bank,field),getattr(sequence,field),f'Bank {field} differs from exact clipped source')
    bounds=getattr(sequence,'window_bounds',None)
    if (bank.window_bounds is None)!=(bounds is None):
        raise RuntimeError('Bank window bounds mismatch')
    if bounds is not None:
        _tensor_equal(bank.window_bounds,bounds,'Bank window bounds mismatch')
    g,b=boundaries.shape
    if bank.response.device.type!='cpu' or bank.response.dtype!=torch.float64 or bank.response.shape!=(g,b,b) or not bool(torch.isfinite(bank.response).all()):
        raise RuntimeError('Bank response shape/dtype/finite mismatch')
    scale=bank.response.abs().amax(dim=(-2,-1)).clamp_min(torch.finfo(torch.float64).tiny)
    tolerance=1e-8*scale
    if bool(((bank.response-bank.response.transpose(-1,-2)).abs().amax(dim=(-2,-1))>tolerance).any()) or bool((bank.response.sum(-1).abs().amax(-1)>tolerance).any()):
        raise RuntimeError('Cached response symmetry/row-sum check failed')
    if len(bank.diagnostics)!=g or any(not math.isfinite(d.interior_condition_number) or d.interior_condition_number>settings['max_condition'] for d in bank.diagnostics):
        raise RuntimeError('Cached conditioning diagnostics invalid')
    refs=_reference_ids(sequence,bank,settings['reference_columns'])
    if bank.query_cache is None:
        raise RuntimeError('Missing exact-tile query cache')
    expected=with_query_reference_cache(bank,refs).query_cache
    actual=bank.query_cache
    _expect(actual.normalization,settings['normalization'],'Cache normalization mismatch')
    for field in ('query_indices','reference_indices','query_ids','reference_ids'):
        _tensor_equal(getattr(actual,field),getattr(expected,field),f'Cached {field} mismatch')
    for field in ('normalized_response','normalization_scale','cqq','cqr','interval_weights','bag_cqq'):
        value,reference=getattr(actual,field),getattr(expected,field)
        if value.device.type!='cpu' or value.dtype!=torch.float64 or value.shape!=reference.shape or not torch.allclose(value,reference,atol=1e-9,rtol=1e-9):
            raise RuntimeError(f'Cached {field} differs from exact clipped tile geometry')
    _verify_eigensystem(actual.cqq,actual.eigenvalues,actual.eigenvectors)
    _verify_eigensystem(actual.bag_cqq,actual.bag_eigenvalues,actual.bag_eigenvectors)


def load_boundary_dataset(directory: Path | str, *, expected_manifest_sha256: str | None=None,
        expected_preparation_sha256: str | None=None,
        expected_source_manifest_sha256: str | None=None) -> VerifiedBoundaryDataset:
    directory=project_path(directory)
    path=directory/'manifest.json'
    manifest_hash=sha256_file(path)
    if expected_manifest_sha256 is not None:
        _expect(manifest_hash,expected_manifest_sha256,'Boundary manifest differs from frozen study')
    manifest=json.loads(path.read_text())
    if manifest.get('schema_version')!=SCHEMA_VERSION or manifest.get('status')!='complete':
        raise RuntimeError('Boundary dataset is incomplete or has an unsupported schema')
    settings=manifest['preparation']
    configuration_hash=_digest(settings)
    _expect(configuration_hash,manifest['preparation_sha256'],'Preparation configuration integrity failure')
    if expected_preparation_sha256 is not None:
        _expect(configuration_hash,expected_preparation_sha256,'Preparation differs from frozen study')
    _expect(_code_hashes(),_IMPORTED_CODE_SHA256,'Boundary implementation changed after module import. Start a fresh process')
    _expect(manifest['implementation_sha256'],_code_hashes(),'Boundary preparation implementation/dependency drift')
    binding=manifest['source']
    if expected_source_manifest_sha256 is not None:
        _expect(binding['manifest_sha256'],expected_source_manifest_sha256,'Unexpected boundary source dataset')
    source,actual_binding=_bind_source(binding['manifest_path'],binding['protocol_path'],
        binding['manifest_sha256'],binding['protocol_sha256'])
    _expect(actual_binding,binding,'Source manifest/protocol binding drift')
    expected_settings=_preparation(source,windows=settings['windows'],splits=settings['splits'],
        resistance_scale=settings['resistance_scale'],feature_columns=settings['feature_columns'],max_condition=settings['max_condition'])
    _expect(expected_settings,settings,'Preparation/source adapter contract drift')
    source_records={(r['split'],r['replicate']):r for r in source['replicates']}
    sources=[case for split in settings['splits'] for case in load_cases(source,split)]
    parents={record['parent_id']:record for record in manifest['parents']}
    records={record['tile_id']:record for record in manifest['records']}
    if len(parents)!=len(manifest['parents']) or len(parents)!=len(sources) or len(records)!=len(manifest['records']):
        raise RuntimeError('Duplicate or missing cached parent/tile identities')
    result=[]
    seen_records=set()
    for case in sources:
        split,replicate=case.metadata['split'],case.metadata['replicate']
        parent_id=f"{binding['manifest_sha256']}:{split}:{replicate}"
        if parent_id not in parents:
            raise RuntimeError('Missing source parent in cached dataset')
        parent=parents[parent_id]
        parent_binding=_parent_binding(source_records[(split,replicate)],case.metadata)
        _expect(parent['binding'],parent_binding,'Cached parent source binding mismatch')
        _expect(parent['binding_sha256'],_digest(parent_binding),'Cached parent binding digest mismatch')
        admission=parent['admissibility']
        if admission.get('all_graphs_admissible') is not True or admission['graphs_checked']!=case.sequence.intervals.shape[0] or not math.isfinite(admission['maximum_interior_condition']) or admission['maximum_interior_condition']>settings['max_condition']:
            raise RuntimeError('Full-parent Kron admissibility record invalid')
        expected_relative=str(Path(split)/f'rep{replicate:03d}'/'supervision.pt')
        supervision_path=_cached_path(directory,parent['supervision_file'],expected_relative)
        _expect(sha256_file(supervision_path),parent['supervision_sha256'],'External supervision integrity failure')
        supervision=torch.load(supervision_path,map_location='cpu',weights_only=True)
        _expect(supervision['schema_version'],SCHEMA_VERSION,'Supervision schema mismatch')
        _expect(supervision['parent_id'],parent_id,'Supervision parent identity mismatch')
        _expect(supervision['parent_binding_sha256'],parent['binding_sha256'],'Supervision source binding mismatch')
        expected_supervision=_supervision(case)
        for field in ('labels','baseline_probabilities','positions','query_node_ids','original_grid_indices'):
            _tensor_equal(supervision[field],expected_supervision[field],f'External {field} differs from source parent')
        _expect(supervision['metadata'],case.metadata,'External metadata differs from source parent')
        _expect(parent['positions'],case.sequence.positions.numel(),'Cached parent coordinate count mismatch')
        _expect(parent['queries'],case.sequence.query_node_ids.numel(),'Cached parent query count mismatch')
        tiles=[]
        covered=torch.zeros(case.sequence.positions.numel(),dtype=torch.long)
        expected_tile_ids=[]
        for tile in split_fixed_windows(case,settings['windows']):
            index=tile.metadata['window_index']
            tile_id=f'{parent_id}:window{index}'
            expected_tile_ids.append(tile_id)
            if tile_id not in records:
                raise RuntimeError('Missing exact clipped tile')
            record=records[tile_id]
            seen_records.add(tile_id)
            left,right=tile.metadata['window_bounds']
            positions=torch.nonzero((case.sequence.positions>=left)&(case.sequence.positions<right),as_tuple=False).flatten()
            covered[positions]+=1
            grid=expected_supervision['original_grid_indices'][positions]
            provenance={'parent_id':parent_id,'tile_id':tile_id,'parent_binding_sha256':parent['binding_sha256'],
                'preparation_sha256':configuration_hash,'window_index':index,'window_bounds':[left,right],
                'global_window_bounds':tile.metadata['global_window_bounds'],
                'parent_position_indices':positions.tolist(),'original_grid_indices':grid.tolist()}
            for field,value in provenance.items():
                _expect(record[field],value,f'Cached tile {field} mismatch')
            metadata=dict(tile.metadata)
            metadata.update(parent_id=parent_id,parent_position_indices=positions.tolist(),retained_parent_grid_indexes=grid.tolist())
            _expect(record['metadata'],metadata,'Cached tile metadata differs from exact source clip')
            expected_relative=str(Path(split)/f'rep{replicate:03d}'/f'window{index:03d}'/'bank.pt')
            bank_path=_cached_path(directory,record['bank_file'],expected_relative)
            _expect(sha256_file(bank_path),record['bank_sha256'],'Boundary bank integrity failure')
            _expect(bank_path.stat().st_size,record['bank_bytes'],'Boundary bank byte-count mismatch')
            payload=torch.load(bank_path,map_location='cpu',weights_only=True)
            if set(payload)!={'schema_version','bank','provenance'} or payload['schema_version']!=SCHEMA_VERSION:
                raise RuntimeError('Unexpected bank envelope, including possible supervision leakage')
            _expect(payload['provenance'],provenance,'Bank/tile source identity mismatch')
            bank=LeafInterfaceBank.from_dict(payload['bank'])
            _verify_bank(bank,tile.sequence,settings)
            tiles.append(BoundaryTile(bank,supervision['labels'][positions],metadata,positions,provenance))
        _expect(parent['tile_ids'],expected_tile_ids,'Parent tile ordering/identity mismatch')
        if not bool((covered==1).all()):
            raise RuntimeError('Cached tiles do not partition retained parent coordinates')
        result.append(BoundaryCase(case.metadata,supervision['labels'],supervision['baseline_probabilities'],tuple(tiles),
            supervision['positions'],supervision['query_node_ids'],parent_id,parent_binding))
    if seen_records!=set(records):
        raise RuntimeError('Extraneous/unbound cached tile records')
    return VerifiedBoundaryDataset(directory,manifest,manifest_hash,configuration_hash,
        tuple(manifest['records']),tuple(result))


def load_boundary_cases(directory: Path | str | VerifiedBoundaryDataset, split: str, **verification) -> list[BoundaryCase]:
    dataset=directory if isinstance(directory,VerifiedBoundaryDataset) else load_boundary_dataset(directory,**verification)
    if isinstance(directory,VerifiedBoundaryDataset) and verification:
        values={'expected_manifest_sha256':dataset.manifest_sha256,
                'expected_preparation_sha256':dataset.preparation_sha256,
                'expected_source_manifest_sha256':dataset.manifest['source']['manifest_sha256']}
        if set(verification)-set(values):
            raise TypeError('Unsupported verified-dataset binding argument')
        for key,value in verification.items():
            if value is not None:
                _expect(value,values[key],'Verified dataset differs from supplied frozen bindings')
    return dataset.cases(split)


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',required=True)
    parser.add_argument('--source-protocol')
    parser.add_argument('--output',required=True)
    parser.add_argument('--windows',type=int)
    parser.add_argument('--splits',nargs='+',choices=['train','val'],default=['train','val'])
    parser.add_argument('--resistance-scale',type=float,default=1e-4)
    parser.add_argument('--max-condition',type=float,default=1e12)
    parser.add_argument('--feature-columns',nargs='+',type=int)
    parser.add_argument('--expected-source-manifest-sha256')
    parser.add_argument('--expected-source-protocol-sha256')
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if args.threads<1:
        raise ValueError('threads must be positive')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(args.threads)
    dataset=prepare_boundary_dataset(args.source_manifest,args.output,source_protocol_path=args.source_protocol,
        windows=args.windows,splits=args.splits,resistance_scale=args.resistance_scale,feature_columns=args.feature_columns,
        max_condition=args.max_condition,expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_source_protocol_sha256=args.expected_source_protocol_sha256,
        progress=lambda event: print(json.dumps(event),flush=True))
    print(json.dumps({'event':'boundary_dataset_complete','directory':str(dataset.directory),
        'manifest_sha256':dataset.manifest_sha256,'preparation_sha256':dataset.preparation_sha256,
        'parents':len(dataset.parent_cases),'tiles':len(dataset.records)}),flush=True)


if __name__=='__main__':
    main()

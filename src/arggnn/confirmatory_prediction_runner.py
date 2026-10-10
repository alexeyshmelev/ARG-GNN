"""Observed-only, frozen-checkpoint ID predictions. No fitting or target readers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import zipfile
import uuid

import numpy as np
import torch

from .resources import ROOT, ALLOWED_GPUS, PROCESS_TITLE, set_research_title, require_storage
from . import confirmatory_lock as lock_api
from . import confirmatory_analysis_runner as analysis
from .data import GraphSequence, slice_genomic_window
from .reference_encoder import ReferenceFeatureAdapter
from .genealogy_baselines import _observed_trees
from .kron_interface import precompute_leaf_interfaces, with_query_reference_cache
from .age_interface import precompute_age_interfaces
from .anchor_kernel import prepare_anchor_kernel_bank
from .canonical_gru import prepare_canonical_parent, prepare_canonical_gnn_input
from .train_secondary_development import make_control
from .development_execution import (DETERMINISTIC_POLICY, configure_deterministic_development,
                                     deterministic_development_manifest)

FEATURE_NAMES = ['reference_ANA', 'reference_CHG', 'reference_WHG', 'reference_EHG',
                 'is_query_leaf', 'is_leaf', 'log_time', 'log_parent_branch']
ADAPTER = dict(reference_columns=[0, 1, 2, 3], time_column=6, branch_column=7,
               time_scale=10000, kernel_scales=[.05, .1, .2])
CLASSES = ['ANA', 'CHG', 'WHG', 'EHG']
SEEDS = [101, 202, 303]
FAMILIES = ['candidate', 'gru', 'transformer', 'reference_smoother', 'allrecord_gru',
            'allrecord_transformer', 'canonical_gru', 'argmix_full_tree', 'gst_ca_span', 'graphssm_zoh']
CONTROLS = ['gain_matched_diagonal', 'static_local', 'matched_scalar_age_full', 'noncontextual']
OBSERVED_ARRAYS = ['metadata_json', 'x', 'edge_index', 'node_graph', 'node_ids', 'clades',
                   'intervals', 'query_node_ids', 'window_bounds']
PREDICTION_ARRAYS = ['metadata_json', 'primary_logits', 'dense_logits', 'primary_positions_bp',
                     'dense_positions_bp', 'query_ids']
CODE_PATHS = {'prediction': 'src/arggnn/confirmatory_prediction_runner.py',
              'prediction_cli': 'scripts/predict_confirmatory.py'}


def _canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _keys(value, names, message):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(message)


def _decode(content):
    return analysis._decode(content)


def _pins():
    paths = {ROOT / path for path in CODE_PATHS.values()}
    paths.update(ROOT / path for path in ('AGENTS.md', 'scripts/launch_gpu.py', 'scripts/project_env.sh'))
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if name.startswith('arggnn.') and filename and Path(filename).suffix == '.py':
            p = Path(filename).resolve()
            if p.is_relative_to(ROOT): paths.add(p)
    return {str(p.relative_to(ROOT)): _sha(p.read_bytes()) for p in sorted(paths)}


_IMPORTED_PINS = None  # Captured after all definitions, including the CLI file.


def _source_check():
    for path, digest in _IMPORTED_PINS.items():
        if _sha((ROOT/path).read_bytes()) != digest:
            raise ValueError('Imported prediction source drift: '+path)


def prediction_schema_values():
    """Source-derived prospective metadata only, never an input or release artifact."""
    _source_check()
    return {'schema_version': 1, 'kind': 'prespecified_observed_prediction_artifact_schema',
        'status': 'frozen_before_test_preparation',
        'code_artifacts': {key: {'path': path, 'sha256': _sha((ROOT/path).read_bytes())}
                           for key, path in CODE_PATHS.items()},
        'observed_npz': analysis.analysis_schema_values()['observed_npz'],
        'prediction_arrays': PREDICTION_ARRAYS, 'implemented_families': FAMILIES,
        'implemented_controls': CONTROLS, 'primary_dense_forward_passes_separate': True,
        'target_readers_present': False, 'fitting_present': False}


def _safe_npz(content, names):
    if len(content) > 256*1024**2:
        raise ValueError('Observed NPZ exceeds bounded reader size')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if (len(members) != len(names) or sum(m.file_size for m in members) > 256*1024**2 or
                len({m.filename for m in members}) != len(members) or
                set(m.filename for m in members) != {name+'.npy' for name in names}):
            raise ValueError('Exact bounded NPZ member allowlist required')
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        if set(data.files) != set(names): raise ValueError('Exact observed NPZ array allowlist required')
        values = {name: data[name] for name in names}
    metadata = values.pop('metadata_json')
    if metadata.shape != () or metadata.dtype.kind != 'U':
        raise ValueError('Scalar Unicode JSON metadata required')
    return _decode(metadata.item()), values


def _grid(identity, prefix):
    grid = analysis._grid(identity[prefix+'_grid'])
    return grid


def _sequence_for_grid(sequence, identity, prefix):
    positions = torch.tensor(_grid(identity, prefix).positions_bp, dtype=torch.float64)/1000000.
    containing = ((positions[:, None] >= sequence.intervals[None, :, 0]) &
                  (positions[:, None] < sequence.intervals[None, :, 1]))
    if not bool((containing.sum(1) == 1).all()):
        raise ValueError('Every fixed grid position must have exactly one observed tree')
    result = replace(sequence, positions=positions, target_graph=containing.long().argmax(1))
    result.validate()
    return result


def validate_observed_npz(content, identity, lock_sha256):
    """Validate only safe raw observed arrays, reconstructing roles/landmarks from topology."""
    if (not isinstance(identity.get('query_ids'), list) or len(identity['query_ids']) != 64 or
        any(type(q) is not int or q < 0 for q in identity['query_ids']) or len(set(identity['query_ids'])) != 64):
        raise ValueError('Exactly64 unique original query IDs required')
    header, values = _safe_npz(content, OBSERVED_ARRAYS)
    expected = {'schema_version': 1, 'kind': 'sealed_confirmatory_observed_graph_sequence',
        'lock_sha256': lock_sha256, 'namespace': identity['namespace'], 'parent_id': identity['parent_id'],
        'feature_names': FEATURE_NAMES, 'time_scale': 10000., 'truth_free_observed_inputs': True}
    if _canonical(header) != _canonical(expected):
        raise ValueError('Exact observed input header/source/feature identity required')
    for name in ('edge_index', 'node_graph', 'node_ids', 'query_node_ids'):
        if values[name].dtype != np.dtype('int64'): raise ValueError('Observed integer arrays must be int64')
    for name in ('intervals', 'window_bounds'):
        if values[name].dtype != np.dtype('float64'): raise ValueError('Observed coverage must be float64')
    for name in ('x', 'clades'):
        if values[name].dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Raw observed features must be float32/float64')
    if values['x'].ndim != 2 or values['x'].shape[1] != 8 or values['clades'].shape != (len(values['x']), 16):
        raise ValueError('Only raw observed F8 and sixteen reference landmark indicators are accepted')
    if values['query_node_ids'].tolist() != identity['query_ids']:
        raise ValueError('Original query order differs from sealed observed identity')
    coverage = np.array(identity['primary_grid']['coverage_bp'], dtype=np.float64)/1000000.
    if not np.array_equal(values['window_bounds'], coverage):
        raise ValueError('Observed source coverage differs from fixed-grid coverage')
    seq = GraphSequence(**{key: torch.from_numpy(value.copy()) for key, value in values.items()},
                        positions=torch.empty(0, dtype=torch.float64), target_graph=torch.empty(0, dtype=torch.long))
    seq = _sequence_for_grid(seq, identity, 'primary')
    x, roles, encoded, trees = _observed_trees(seq, (0, 1, 2, 3), 6, 7, True)
    expected_clades = np.zeros((len(x), 16), dtype=np.float64)
    expected_branch = np.zeros(len(x), dtype=np.float64)
    previous_roles = None
    for tree in trees:
        leaves = {int(seq.node_ids[u]): u for u in tree.nodes if int(seq.node_ids[u]) >= 0}
        if len(leaves) != 88 or set(leaves) != set(identity['query_ids']) | {int(seq.node_ids[u]) for u in tree.reference_nodes}:
            raise ValueError('Exactly64 query and24 observed reference leaves are required')
        role_map = {leaf: tuple(roles[node]) for leaf, node in leaves.items()}
        if previous_roles is not None and role_map != previous_roles:
            raise ValueError('Stable reference/query identity changes across observed trees')
        previous_roles = role_map
        landmarks = []
        for category in range(4):
            candidates = sorted(leaf for leaf, node in leaves.items() if roles[node, category] == 1.)
            if len(candidates) != 6: raise ValueError('Require six observed references per source')
            landmarks.extend(candidates[index] for index in np.linspace(0, 5, 4, dtype=int))
        for leaf, node in leaves.items():
            is_query = leaf in identity['query_ids']
            if x[node, 4] != float(is_query) or x[node, 5] != 1.:
                raise ValueError('Observed leaf/query role features differ')
            age = 0. if is_query else [260., 300., 250., 250.][int(roles[node].argmax())]
            if not math.isclose(encoded[node], math.log1p(age)/math.log1p(10000.), abs_tol=2e-7, rel_tol=0):
                raise ValueError('Observed cohort sampling date differs')
            if len(tree.adjacency[node]) != 1: raise ValueError('Stable sample IDs must identify terminal leaves')
        for u in tree.nodes:
            if int(seq.node_ids[u]) < 0 and bool(x[u, 4:6].any()):
                raise ValueError('Internal observed nodes cannot contain leaf/query flags')
            parent = tree.parent[u]
            if parent >= 0:
                branch = math.expm1(encoded[parent]*math.log1p(10000.))-math.expm1(encoded[u]*math.log1p(10000.))
                if branch < -1e-4: raise ValueError('Observed branch age inversion')
                expected_branch[u] = math.log1p(max(0., branch))/math.log1p(10000.)
        for slot, leaf in enumerate(landmarks):
            u = leaves[leaf]
            while u >= 0:
                expected_clades[u, slot] = 1.; u = tree.parent[u]
    if not np.array_equal(values['clades'], expected_clades):
        raise ValueError('Clades must be exact observed-reference landmark descendants')
    if not np.allclose(x[:, 7], expected_branch, atol=2e-5, rtol=2e-5):
        raise ValueError('Raw branch channel differs from observed tree ages')
    return seq


@dataclass
class PredictionAdmission:
    lock: object
    capture: object
    schema: dict
    ledger: dict
    registry: dict
    identity: dict
    identity_ref: dict
    observed_ref: dict
    bundle: dict
    seed_bundle: dict
    settings: dict
    method_id: str
    seed: int
    parent: int
    ledger_ref: dict
    source_content: bytes
    content: bytes = b''

    def _state(self):
        return _canonical({'lock': self.lock.payload, 'schema': self.schema, 'ledger': self.ledger,
            'registry': self.registry, 'identity': self.identity, 'identity_ref': self.identity_ref,
            'observed_ref': self.observed_ref, 'bundle': self.bundle, 'seed_bundle': self.seed_bundle,
            'settings': self.settings, 'method': self.method_id, 'seed': self.seed, 'parent': self.parent,
            'ledger_ref': self.ledger_ref, 'observed_bytes_sha256': _sha(self.source_content)})

    def check(self, full=True):
        if self._state() != self.content: raise ValueError('In-memory prediction admission drift')
        self.lock.check(full=full); self.capture.check(); _source_check()
        if _sha(self.source_content) != self.observed_ref['sha256']:
            raise ValueError('Captured observed source bytes drift')


def admit_prediction(lock_path, expected_lock_sha256, ledger_path, expected_ledger_sha256,
                     method_id, training_seed, parent_id, *, project_root=ROOT):
    """Read sealed metadata and observed NPZ only. Target paths remain unopened."""
    _source_check()
    capture = analysis._Capture(project_root)
    frozen = lock_api.validate_confirmatory_lock(lock_path, expected_lock_sha256, project_root=project_root)
    frozen.check()
    payload = frozen.payload
    if payload.get('lock_execution_scope') != 'id_confirmation_only' or payload.get('training_seeds') != SEEDS:
        raise ValueError('Actual fixed-seed ID-only lock required')
    code = payload['code_artifacts']['prediction']
    if {key: code[key] for key in ('path', 'sha256')} not in prediction_schema_values()['code_artifacts'].values():
        raise ValueError('Frozen prediction workflow source differs')
    if payload['numerical_execution_policy'] != DETERMINISTIC_POLICY:
        raise ValueError('Frozen strict numerical policy differs')
    refs = payload['bound_references']; schema = capture.read(refs['analysis_schema'])
    if (schema != analysis.analysis_schema_values() or schema['prediction_npz']['arrays'] != PREDICTION_ARRAYS or
        schema['observed_npz']['arrays'] != OBSERVED_ARRAYS or schema['observed_npz']['feature_names'] != FEATURE_NAMES):
        raise ValueError('Exact current frozen analysis artifact schema required')
    registry = capture.read(refs['parent_seed_registry']); cohort = registry['cohorts']['id_confirmation']
    if len(cohort['parents']) != 192 or [row['parent_id'] for row in cohort['parents']] != list(range(192)):
        raise ValueError('Exact192-parent registry required')
    methods = analysis._methods(payload)
    if method_id not in methods or type(training_seed) is not int or training_seed not in SEEDS or type(parent_id) is not int:
        raise ValueError('Exact locked method, training seed and integer parent required')
    bundle = methods[method_id]; seed_bundle = analysis._seed_bundle(bundle, training_seed)
    ledger_ref = capture.ref({'path': ledger_path, 'sha256': expected_ledger_sha256})
    ledger = capture.read(ledger_ref)
    _keys(ledger, schema['admission_ledger']['fields'], 'Exact sealed admission ledger fields required')
    if (_canonical({key: ledger[key] for key in ('schema_version', 'kind', 'status', 'lock_sha256',
            'parent_seed_registry_sha256', 'cohort', 'namespace', 'target_admitted_parents', 'maximum_attempted_parents')}) !=
            _canonical({'schema_version': 1, 'kind': 'sealed_confirmatory_admission_ledger',
                'status': 'sealed_before_model_scoring', 'lock_sha256': frozen.sha256,
                'parent_seed_registry_sha256': refs['parent_seed_registry']['sha256'], 'cohort': 'id_confirmation',
                'namespace': cohort['namespace'], 'target_admitted_parents': 160, 'maximum_attempted_parents': 192})):
        raise ValueError('Ledger seal/lock/namespace/budget differs')
    attempts = ledger['attempts']; admitted = []; target_paths = set(); chosen = None
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 192:
        raise ValueError('Bounded complete admission attempt ledger required')
    for index, row in enumerate(attempts):
        _keys(row, schema['admission_ledger']['attempt_fields'], 'Exact label-blind attempt fields required')
        if (type(row['parent_id']) is not int or row['parent_id'] != index or row['label_blind'] is not True or
            row['status'] not in ('admitted', 'rejected') or not isinstance(row['reason'], str) or not row['reason']):
            raise ValueError('Contiguous registry attempt and label-blind admission reason required')
        if row['status'] == 'admitted':
            if len(admitted) == 160: raise ValueError('Cannot continue after first160 admitted parents')
            admitted.append(index)
            for name in ('observed_identity', 'primary_labels', 'census_tracts'):
                ref = capture.ref(row[name])  # Metadata only, including the two target references.
                if name != 'observed_identity': target_paths.add(ref['path'])
            if index == parent_id: chosen = row
        elif any(row[name] is not None for name in ('observed_identity', 'primary_labels', 'census_tracts')):
            raise ValueError('Rejected rows cannot carry observed/target artifacts')
    if not admitted or (len(admitted) < 160 and len(attempts) != 192) or (len(admitted) == 160 and attempts[-1]['status'] != 'admitted'):
        raise ValueError('Admission must reach160 or exhaust192 before any model scoring')
    if chosen is None: raise ValueError('Requested parent is not label-blind admitted')
    identity_ref = capture.ref(chosen['observed_identity'])
    if identity_ref['path'] in target_paths: raise ValueError('A target cannot substitute observed identity')
    identity = capture.read(identity_ref)
    _keys(identity, schema['observed_identity']['fields'], 'Exact observed identity fields required')
    reg = cohort['parents'][parent_id]
    if (_canonical({key: identity[key] for key in ('schema_version', 'kind', 'lock_sha256', 'namespace', 'parent_id',
            'parent_seeds', 'truth_free_observed_inputs')}) != _canonical({'schema_version': 1,
            'kind': 'sealed_confirmatory_observed_parent_identity', 'lock_sha256': frozen.sha256,
            'namespace': cohort['namespace'], 'parent_id': parent_id, 'truth_free_observed_inputs': True,
            'parent_seeds': {key: reg[key] for key in ('ancestry_seed', 'mutation_seed', 'inference_seed')}})):
        raise ValueError('Observed identity source/seed/lock contract differs')
    primary = _grid(identity, 'primary'); dense = _grid(identity, 'dense')
    if (primary.domain_bp != (0., 1000000.) or dense.domain_bp != primary.domain_bp or
        primary.coverage_bp != dense.coverage_bp or primary.grid_points != 128 or dense.grid_points != 4096 or
        len(primary.positions_bp) < 120 or (primary.coverage_bp[1]-primary.coverage_bp[0])/1000000. < .95):
        raise ValueError('Original128/4096 fixed coverage admission required')
    query_ids = identity['query_ids']; pairs = np.asarray(identity['diploid_pairs'])
    if (not isinstance(query_ids, list) or len(query_ids) != 64 or any(type(q) is not int or q < 0 for q in query_ids) or
        len(set(query_ids)) != 64 or pairs.shape != (32, 2) or pairs.dtype.kind not in ('i', 'u') or
        sorted(pairs.ravel().tolist()) != list(range(64))):
        raise ValueError('Exact64 query identity and32 diploid pair slots required')
    if identity['artificial_tile_edges_bp'] != [125000.*index for index in range(1, 8)]:
        raise ValueError('Exact eight fixed physical tiles required')
    sources = identity['source_artifacts']
    if not isinstance(sources, list) or len(sources) != 1: raise ValueError('Exactly one observed-only NPZ source required')
    source = capture.ref(sources[0])
    if Path(source['path']).suffix != '.npz' or source['path'] in target_paths or source['path'] == identity_ref['path']:
        raise ValueError('Observed NPZ cannot alias identity or target artifacts')
    body = capture.read(source, json_file=False)
    validate_observed_npz(body, identity, frozen.sha256)
    settings = capture.read(bundle['protocol'])
    admission = PredictionAdmission(frozen, capture, schema, ledger, registry, identity, identity_ref,
        source, bundle, seed_bundle, settings, method_id, training_seed, parent_id, ledger_ref, body)
    admission.content = admission._state(); admission.check()
    return admission


def make_frozen_model(admission):
    """Existing pure factories, CPU construction, exact schema and fixed-buffer validation."""
    family = admission.bundle['family_id']; name = admission.bundle['configuration_id']
    with torch.device('cpu'):
        model = make_control(name, admission.settings) if family in CONTROLS else lock_api._primary_model(family, name, admission.settings)
    expected = model.state_dict(); parameters = sum(p.numel() for p in model.parameters())
    if parameters != admission.bundle['choice']['parameters']:
        raise ValueError('Frozen factory capacity differs from selected bundle')
    checkpoint_ref = admission.seed_bundle['checkpoint']
    if checkpoint_ref is None:
        if family != 'reference_smoother' or parameters != 0 or expected:
            raise ValueError('Only deterministic zero-state smoother may omit checkpoint')
    else:
        content = admission.capture.read(checkpoint_ref, json_file=False)
        checkpoint = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict):
            raise ValueError('Actual frozen model checkpoint mapping required')
        state = checkpoint['model']
        if (set(state) != set(expected) or any(not isinstance(state[key], torch.Tensor) or
            state[key].shape != value.shape or state[key].dtype != value.dtype or
            not bool(torch.isfinite(state[key]).all()) for key, value in expected.items())):
            raise ValueError('Exact finite frozen state keys/shapes/dtypes required')
        for key, value in model.named_buffers():
            if not torch.equal(state[key], value): raise ValueError('Checkpoint changed a fixed factory buffer')
        model.load_state_dict(state, strict=True)
    model.eval(); model.requires_grad_(False)
    return model


def _references(seq):
    rows = seq.x[seq.node_graph == 0]; ids = seq.node_ids[seq.node_graph == 0]
    return ids[rows[:, :4].sum(-1) == 1].sort().values


def prepare_observed_tiles(sequence, identity, family, settings):
    """Pure observed preprocessing for all ten frozen primary families and controls."""
    adapter = ReferenceFeatureAdapter(8, **ADAPTER)
    canonical = family == 'canonical_gru'
    augmented = (prepare_canonical_parent(sequence, ADAPTER,
        source_provenance={'kind': 'observed_confirmatory_source', 'parent_id': identity['parent_id']}).sequence
        if canonical else adapter.precompute(sequence))
    refs = _references(augmented)
    boundary = family in ('candidate', 'gru', 'transformer', 'reference_smoother', 'allrecord_gru', 'allrecord_transformer') or family in CONTROLS
    anchors = family in ('candidate', 'gru', 'transformer', 'reference_smoother') or family in CONTROLS
    if family not in FAMILIES + CONTROLS: raise ValueError('No implemented observed adapter for selected family')
    if boundary:
        # Full-parent geometry admissibility precedes clipping, as in development.
        full = precompute_leaf_interfaces(augmented, time_column=6, parent_branch_column=7, resistance_scale=1e-4)
        with_query_reference_cache(full, refs)
    geometry = settings.get('selected_candidate_spec', {}).get('geometry') if family in CONTROLS else (
        settings.get('model_specs', {}).get(settings.get('_selected_configuration_id'), {}).get('geometry'))
    age_needed = anchors and geometry == 'age'
    result = []
    lo, hi = [float(v) for v in augmented.window_bounds]
    for index in range(8):
        left, right = max(lo, index/8), min(hi, (index+1)/8)
        if left >= right or not bool(((augmented.positions >= left) & (augmented.positions < right)).any()): continue
        tile, indices = slice_genomic_window(augmented, left, right)
        if boundary:
            bank = with_query_reference_cache(precompute_leaf_interfaces(tile, time_column=6,
                parent_branch_column=7, resistance_scale=1e-4), refs)
            if anchors:
                age = precompute_age_interfaces(tile, plain_bank=bank, reference_ids=refs,
                    time_column=6, parent_branch_column=7, resistance_scale=1e-4).operator_cache if age_needed else None
                value = prepare_anchor_kernel_bank(bank, age, num_anchors=32, num_scales=3)
            else: value = bank
        else:
            tile = replace(tile, x=tile.x.float(), clades=tile.clades.float())
            value = prepare_canonical_gnn_input(tile) if canonical else tile
        result.append((value, indices, index))
    covered = torch.cat([indices for _, indices, _ in result]) if result else torch.empty(0, dtype=torch.long)
    if not torch.equal(covered.sort().values, torch.arange(len(sequence.positions))):
        raise ValueError('Observed tiles must cover each fixed coordinate exactly once')
    return result


def _forward_grid(model, sequence, admission, device):
    family = admission.bundle['family_id']; settings = dict(admission.settings)
    settings['_selected_configuration_id'] = admission.bundle['configuration_id']
    tiles = prepare_observed_tiles(sequence, admission.identity, family, settings)
    result = None; seen = torch.zeros(len(sequence.positions), dtype=torch.bool)
    for value, indices, index in tiles:
        admission.check(full=False); deterministic_development_manifest()
        if device == 'cuda':
            import setproctitle
            if setproctitle.getproctitle() != PROCESS_TITLE: raise RuntimeError('Research process title drift')
            validate_launcher_lease()
        source = value.bank if hasattr(value, 'bank') else value.sequence if hasattr(value, 'sequence') else value
        if not torch.equal(source.positions, sequence.positions[indices]):
            raise ValueError('Prepared tile changed original fixed coordinates')
        query = source.query_node_ids
        if not torch.equal(query, sequence.query_node_ids) or bool(seen[indices].any()):
            raise ValueError('Prepared tile changed original query identity or repeated readouts')
        value = value.to(device, torch.float32) if hasattr(value, 'response') or hasattr(value, 'bank') else value.to(device)
        with torch.inference_mode(): logits = model(value).detach().cpu()
        if logits.shape != (len(indices), 64, 4) or logits.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(logits).all()):
            raise ValueError('Finite original aligned raw logits required')
        if result is None: result = torch.empty((len(sequence.positions), 64, 4), dtype=logits.dtype)
        if result.dtype != logits.dtype: raise ValueError('Raw logit dtype changed across tiles')
        result[indices] = logits; seen[indices] = True
    if result is None or not bool(seen.all()): raise ValueError('Every retained coordinate requires raw predictions')
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Prediction model entered gradient/training state')
    return result.numpy()


def validate_launcher_lease(environ=None):
    environ = os.environ if environ is None else environ
    physical = environ.get('ARGGNN_PHYSICAL_GPU')
    if physical not in {str(gpu) for gpu in ALLOWED_GPUS}: raise RuntimeError('Project launcher on physical GPU2..5 required')
    expected = ROOT/'.runtime/locks'/f'gpu_{physical}.lock'
    try:
        lease = Path(environ['ARGGNN_GPU_LEASE']).resolve(strict=True); fd = int(environ['ARGGNN_GPU_LEASE_FD'])
        actual = os.fstat(fd); on_disk = lease.stat()
    except (KeyError, ValueError, OSError) as error: raise RuntimeError('Valid inherited GPU lease required') from error
    if lease != expected.resolve() or not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (on_disk.st_dev, on_disk.st_ino):
        raise RuntimeError('Inherited GPU lease path/descriptor mismatch')
    data = _decode(lease.read_bytes())
    if (data.get('gpu') != int(physical) or not isinstance(data.get('uuid'), str) or not data['uuid'].startswith('GPU-') or
        environ.get('CUDA_VISIBLE_DEVICES') != data['uuid'] or data.get('job') != environ.get('ARGGNN_JOB_NAME')):
        raise RuntimeError('Single visible UUID and launcher lease metadata required')
    with lease.open('r+') as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(handle, fcntl.LOCK_UN); raise RuntimeError('Launcher GPU lease is not held')
    return {'physical_gpu': int(physical), 'visible_gpu_uuid': data['uuid'], 'lease_path': str(lease),
            'lease_fd': fd, 'launcher_metadata': data}


def _normalized_cuda_uuid(value):
    if not isinstance(value, str):
        raise RuntimeError("Malformed CUDA device UUID")
    text = value[4:] if value.startswith("GPU-") else value
    try:
        canonical = str(uuid.UUID(text))
    except (ValueError, AttributeError) as error:
        raise RuntimeError("Malformed CUDA device UUID") from error
    if canonical != text.lower():
        raise RuntimeError("Malformed CUDA device UUID")
    return canonical


def initialize_execution(device, runtime):
    """Title, strict policy, held lease precede every CUDA inspection/context."""
    runtime.update(device_kind=device, initialization_stage='process_title', device={'scope': 'not inspected'})
    set_research_title()
    import setproctitle
    runtime['process_title'] = setproctitle.getproctitle()
    if runtime['process_title'] != PROCESS_TITLE: raise RuntimeError('Required process title missing')
    runtime['initialization_stage'] = 'execution_policy'
    runtime['deterministic_execution_policy'] = configure_deterministic_development()
    runtime['software'] = {'python': sys.version, 'python_executable': sys.executable,
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'torch_build': torch.__config__.show()}
    runtime['initialization_stage'] = 'launcher_lease'
    if device == 'cuda':
        runtime.update(validate_launcher_lease())
        runtime['initialization_stage'] = 'visible_device_count'
        if torch.cuda.device_count() != 1: raise RuntimeError('Exactly one inherited reserved GPU required')
        runtime['initialization_stage'] = 'device_properties'
        props = torch.cuda.get_device_properties(0)
        runtime['device'] = {'logical_index': 0, 'name': props.name, 'total_memory_bytes': props.total_memory,
            'compute_capability': [props.major, props.minor], 'runtime_uuid': str(getattr(props, 'uuid', 'unavailable'))}
        if runtime['device']['runtime_uuid'] != 'unavailable':
            runtime['device']['canonical_runtime_uuid'] = _normalized_cuda_uuid(runtime['device']['runtime_uuid'])
            runtime['canonical_visible_gpu_uuid'] = _normalized_cuda_uuid(runtime['visible_gpu_uuid'])
            if runtime['device']['canonical_runtime_uuid'] != runtime['canonical_visible_gpu_uuid']:
                raise RuntimeError('CUDA UUID differs from inherited lease')
    elif device == 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('CPU validation requires CUDA_VISIBLE_DEVICES empty')
        runtime['device'] = {'name': platform.processor(), 'scope': 'CPU validation only'}
    else: raise ValueError('Device must be cpu or cuda')
    runtime['initialization_stage'] = 'complete'


def _runtime_matches_lock(admission, runtime):
    frozen = admission.lock.payload['captured_runtime']
    if runtime['device_kind'] == 'cuda':
        if any(runtime['software'].get(k) != frozen['software'].get(k) for k in ('python_executable', 'torch', 'torch_cuda', 'torch_build')):
            raise ValueError('CUDA prediction software differs from frozen numerical runtime')
        if any(runtime['device'].get(k) != frozen['device'].get(k) for k in ('name', 'compute_capability')):
            raise ValueError('CUDA prediction hardware differs from frozen numerical runtime')


def _publish(path, content, check):
    pending = None; identity = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='.prediction_pending_', delete=False) as handle:
            pending = Path(handle.name); current = os.fstat(handle.fileno())
            identity = (current.st_dev, current.st_ino)
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        check(); current = pending.lstat()
        if (current.st_dev, current.st_ino) != identity:
            raise ValueError('Owned prediction output inode changed')
        if current.st_size != len(content) or _sha(pending.read_bytes()) != _sha(content):
            raise ValueError('Owned prediction output bytes changed')
        current = pending.lstat()
        if (current.st_dev, current.st_ino) != identity:
            raise ValueError('Owned prediction output inode changed during byte verification')
        os.link(pending, path, follow_symlinks=False)
    finally:
        if pending is not None and identity is not None:
            try:
                current = pending.lstat()
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == identity:
                    pending.unlink()
    return {'path': str(path), 'sha256': _sha(content)}


def prediction_header(admission):
    return {'schema_version': 1, 'kind': 'sealed_confirmatory_raw_prediction',
        'lock_sha256': admission.lock.sha256, 'namespace': admission.identity['namespace'],
        'parent_id': admission.parent, 'method_id': admission.method_id, 'training_seed': admission.seed,
        'configuration_id': admission.bundle['configuration_id'], 'family_id': admission.bundle['family_id'],
        'protocol': admission.bundle['protocol'], 'checkpoint': admission.seed_bundle['checkpoint'],
        'observed_identity_sha256': admission.identity_ref['sha256'],
        'primary_grid_sha256': admission.identity['primary_grid']['sha256'],
        'dense_grid_sha256': admission.identity['dense_grid']['sha256'],
        'class_order': CLASSES, 'no_gradients_or_optimization': True}


def predict_confirmatory(admission, output_directory, *, device='cpu'):
    """Execute one admitted frozen method/seed/parent cell, preserving terminal failure evidence."""
    admission.check()
    output = admission.capture.path(output_directory)
    if output.exists(): raise ValueError('Fresh exclusive prediction cell output required')
    require_storage(headroom_bytes=100000000)
    output.mkdir(parents=True)
    base = {'schema_version': 1, 'kind': 'confirmatory_prediction_cell_receipt',
        'lock_sha256': admission.lock.sha256, 'admission_ledger': admission.ledger_ref,
        'cell': prediction_header(admission), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'no_gradients_or_optimization': True, 'target_artifacts_opened': False,
        'raw_primary_dense_separate_forward_passes': True, 'imported_source_sha256': _IMPORTED_PINS}
    runtime = {}; phase = 'runtime_initialization'; prediction = None
    _publish(output/'startup.json', _canonical({**base, 'status': 'started'}), admission.check)
    try:
        initialize_execution(device, runtime); _runtime_matches_lock(admission, runtime)
        admission.check(); phase = 'frozen_model_schema'; model = make_frozen_model(admission)
        admission.check(); model = model.to(device)
        phase = 'primary_observed_preprocessing_and_forward'
        sequence = validate_observed_npz(admission.source_content, admission.identity, admission.lock.sha256)
        primary = _forward_grid(model, sequence, admission, device)
        # Primary tensor is copied before constructing or forwarding any dense readout.
        primary = primary.copy(); phase = 'dense_observed_preprocessing_and_forward'
        dense_sequence = _sequence_for_grid(sequence, admission.identity, 'dense')
        dense = _forward_grid(model, dense_sequence, admission, device)
        phase = 'exclusive_raw_prediction_publication'; admission.check(); deterministic_development_manifest()
        if device == 'cuda': validate_launcher_lease()
        stream = io.BytesIO()
        np.savez(stream, metadata_json=np.array(_canonical(prediction_header(admission)).decode()),
            primary_logits=primary, dense_logits=dense,
            primary_positions_bp=np.array(admission.identity['primary_grid']['positions_bp'], dtype=np.float64),
            dense_positions_bp=np.array(admission.identity['dense_grid']['positions_bp'], dtype=np.float64),
            query_ids=np.array(admission.identity['query_ids'], dtype=np.int64))
        prediction = _publish(output/'prediction.npz', stream.getvalue(), admission.check)
        prediction['path'] = str(Path(prediction['path']).relative_to(admission.capture.root))
        receipt = {**base, 'status': 'success', 'terminal': True, 'phase': 'complete', 'runtime': runtime,
            'prediction': prediction, 'reason': '', 'captured_artifacts': admission.capture.provenance(),
            'cuda_execution_performed': device == 'cuda', 'cpu_validation_only': device == 'cpu'}
        if device == 'cuda':
            receipt['cuda_allocator_peak_bytes'] = {'allocated': torch.cuda.max_memory_allocated(),
                                                   'reserved': torch.cuda.max_memory_reserved()}
        result = _publish(output/'terminal.json', _canonical(receipt), admission.check)
        result['path'] = str(Path(result['path']).relative_to(admission.capture.root))
        return {'prediction': prediction, 'receipt': result, 'values': receipt}
    except BaseException as error:
        failure = {**base, 'status': 'failed', 'terminal': True, 'phase': phase, 'runtime': runtime,
                   'prediction': None, 'orphan_prediction': prediction, 'reason': str(error),
                   'error_type': type(error).__name__, 'cpu_validation_only': device == 'cpu'}
        # Preserve failures even when input/source verification itself has failed.
        _publish(output/'terminal.json', _canonical(failure), lambda: None)
        raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', required=True); parser.add_argument('--expected-lock-sha256', required=True)
    parser.add_argument('--admission-ledger', required=True); parser.add_argument('--expected-admission-ledger-sha256', required=True)
    parser.add_argument('--method', required=True); parser.add_argument('--training-seed', type=int, required=True)
    parser.add_argument('--parent-id', type=int, required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    admission = admit_prediction(args.lock, args.expected_lock_sha256, args.admission_ledger,
        args.expected_admission_ledger_sha256, args.method, args.training_seed, args.parent_id)
    result = predict_confirmatory(admission, args.output, device=args.device)
    print(json.dumps({'event': 'confirmatory_prediction_cell', 'status': 'success', **result['receipt'],
        'cpu_validation_only': args.device == 'cpu'}), flush=True)


_IMPORTED_PINS = _pins()

if __name__ == '__main__':
    main()

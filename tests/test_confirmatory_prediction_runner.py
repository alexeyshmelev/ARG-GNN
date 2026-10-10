"""Handmade observed trees and unit-only fake provenance admission. No final inputs."""
from dataclasses import replace
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arggnn import confirmatory_prediction_runner as pred
from arggnn.dense_tract_metrics import DenseGrid
from arggnn.genealogy_baselines import _observed_trees
from arggnn.secondary_control_bindings import controls_for


def _write(root, name, value, binary=False):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    content = value if binary else pred._canonical(value)
    path.write_bytes(content)
    return {'path': name, 'sha256': pred._sha(content)}


def _grid(n):
    return DenseGrid(str(n), (0., 1000000.), (0., 1000000.), n,
                     tuple(range(n)), tuple((np.arange(n)+.5)/n*1000000.)).identity_record()


@pytest.fixture(scope='module')
def raw():
    # Reuse only the handmade topology generator, not its targets/admission runner.
    spec = importlib.util.spec_from_file_location('handmade_prediction_test_topology', pred.ROOT/'scripts/check_development_cuda_admission.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    seq = module.handmade_observed_sequence()
    x = seq.x.clone()
    for u in range(len(x)):
        if int(seq.node_ids[u]) >= 0:
            age = 0. if x[u, 4] == 1 else [260., 300., 250., 250.][int(x[u, :4].argmax())]
            x[u, 6] = math.log1p(age)/math.log1p(10000.)
    seq = replace(seq, x=x)
    _, roles, encoded, trees = _observed_trees(seq, (0, 1, 2, 3), 6, 7, True)
    clades = np.zeros((len(x), 16))
    for tree in trees:
        leaves = {int(seq.node_ids[u]): u for u in tree.nodes if int(seq.node_ids[u]) >= 0}
        landmarks = []
        for c in range(4):
            ids = sorted(i for i, u in leaves.items() if roles[u, c] == 1)
            landmarks.extend(ids[i] for i in np.linspace(0, 5, 4, dtype=int))
        for u in tree.nodes:
            parent = tree.parent[u]
            branch = 0. if parent < 0 else math.expm1(encoded[parent]*math.log1p(10000.))-math.expm1(encoded[u]*math.log1p(10000.))
            x[u, 7] = math.log1p(branch)/math.log1p(10000.)
        for slot, leaf in enumerate(landmarks):
            u = leaves[leaf]
            while u >= 0: clades[u, slot] = 1.; u = tree.parent[u]
    return replace(seq, x=x, clades=torch.tensor(clades))


def identity(raw):
    return {'schema_version': 1, 'kind': 'sealed_confirmatory_observed_parent_identity',
        'lock_sha256': 'a'*64, 'namespace': 'synthetic-unit-only-no-real-freeze', 'parent_id': 0,
        'parent_seeds': {'ancestry_seed': 11, 'mutation_seed': 12, 'inference_seed': 13},
        'source_artifacts': [], 'query_ids': raw.query_node_ids.tolist(),
        'diploid_pairs': np.arange(64).reshape(32, 2).tolist(), 'primary_grid': _grid(128),
        'dense_grid': _grid(4096), 'artificial_tile_edges_bp': [125000.*i for i in range(1, 8)],
        'truth_free_observed_inputs': True}


def observed_bytes(raw, ident, mutate=None):
    header = {'schema_version': 1, 'kind': 'sealed_confirmatory_observed_graph_sequence',
        'lock_sha256': 'a'*64, 'namespace': ident['namespace'], 'parent_id': 0,
        'feature_names': pred.FEATURE_NAMES, 'time_scale': 10000., 'truth_free_observed_inputs': True}
    arrays = {name: getattr(raw, name).numpy().copy() for name in pred.OBSERVED_ARRAYS if name != 'metadata_json'}
    arrays['metadata_json'] = np.array(pred._canonical(header).decode())
    if mutate: mutate(arrays)
    stream = io.BytesIO(); np.savez(stream, **arrays); return stream.getvalue()


def settings(family, geometry='plain'):
    base = {'hidden_dim': 32, 'num_layers': 2, 'latent_dim': 8, 'num_anchors': 32, 'num_scales': 3,
        'initial_coupling': [5., 20., 80.], 'initial_damping': 1., 'max_attention_score_bytes': 512*1024**2,
        'models': ['unit'], 'reference_adapter': pred.ADAPTER}
    if family == 'candidate':
        base['model_specs'] = {'unit': {'geometry': geometry, 'kernel_depth': 1, 'coupling': 'full', 'rate_parameterization': 'log'}}
    elif family in ('gru', 'transformer'):
        base['model_specs'] = {'unit': {'family': family, 'hidden_dim': 32, 'dropout': 0., 'temporal_depth': 2}}
    elif family == 'reference_smoother':
        base['model_specs'] = {'unit': {'family': family, 'bandwidth_bp': 30000., 'pseudocount': .001}}
    elif family in ('allrecord_gru', 'allrecord_transformer'):
        base['model_specs'] = {'unit': {'family': family, 'hidden_dim': 32, 'dropout': 0., 'source_depth': 2, 'temporal_depth': 2}}
    else:
        base['configurations'] = [{'id': 'unit', 'hidden_dim': 32, 'dropout': 0.}]
        base['temporal_layers'] = 2; base['family'] = family
    return base


def synthetic_admission(tmp_path, raw, monkeypatch, family='reference_smoother', control=None, geometry='plain'):
    ident = identity(raw)
    observed_ref = _write(tmp_path, 'observed.npz', observed_bytes(raw, ident), True)
    ident['source_artifacts'] = [observed_ref]
    identity_ref = _write(tmp_path, 'identity.json', ident)
    selected = settings(family, geometry)
    selected_family = family
    if control:
        spec = {'geometry': geometry, 'kernel_depth': 2 if control == 'noncontextual' else 1,
                'coupling': 'full', 'rate_parameterization': 'log'}
        selected = {**selected, 'input_dim': 47, 'num_classes': 4,
                    'selected_candidate_spec': spec, 'model_specs': {'unit': controls_for(spec)[control]}}
        selected_family = control
    protocol = _write(tmp_path, 'protocol.json', selected)
    with torch.device('cpu'):
        model = pred.make_control('unit', selected) if control else pred.lock_api._primary_model(family, 'unit', selected)
    stream = io.BytesIO(); torch.save({'model': model.state_dict()}, stream)
    checkpoint = None if family == 'reference_smoother' and not control else _write(tmp_path, 'checkpoint.pt', stream.getvalue(), True)
    seed_bundles = [{'seed': seed, 'checkpoint': checkpoint} for seed in pred.SEEDS]
    bundle = {'family_id': selected_family, 'configuration_id': 'unit', 'protocol': protocol,
              'choice': {'parameters': sum(p.numel() for p in model.parameters())}, 'seed_bundles': seed_bundles}
    schema = pred.analysis.analysis_schema_values()
    schema_ref = _write(tmp_path, 'schema.json', schema)
    registry = {'training_seeds': pred.SEEDS, 'cohorts': {'id_confirmation': {'namespace': ident['namespace'],
        'parents': [{'parent_id': i, **ident['parent_seeds']} for i in range(192)]}}}
    registry_ref = _write(tmp_path, 'registry.json', registry)
    payload = {'kind': 'confirmatory_provenance_lock', 'lock_execution_scope': 'id_confirmation_only',
        'training_seeds': pred.SEEDS, 'code_artifacts': {'prediction': {**pred.prediction_schema_values()['code_artifacts']['prediction'], 'entry_point': 'main'}},
        'numerical_execution_policy': pred.DETERMINISTIC_POLICY,
        'candidate': bundle, 'strongest_required_baseline': bundle,
        'secondary_controls': {'gain_matched_diagonal': bundle, 'static_local': bundle},
        'bound_references': {'analysis_schema': schema_ref, 'parent_seed_registry': registry_ref},
        'captured_runtime': {'software': {}, 'device': {}}}
    check_calls = []
    frozen = SimpleNamespace(payload=payload, sha256='a'*64, check=lambda **kwargs: check_calls.append(kwargs))
    validator_calls = []
    def validator(*args, **kwargs): validator_calls.append((args, kwargs)); return frozen
    monkeypatch.setattr(pred.lock_api, 'validate_confirmatory_lock', validator)
    attempts = []
    for i in range(192):
        attempts.append({'parent_id': i, 'status': 'admitted' if i == 0 else 'rejected',
            'reason': 'explicit handmade unit fixture', 'label_blind': True,
            'observed_identity': identity_ref if i == 0 else None,
            'primary_labels': {'path': 'never_open_labels.npz', 'sha256': 'b'*64} if i == 0 else None,
            'census_tracts': {'path': 'never_open_census.json', 'sha256': 'c'*64} if i == 0 else None})
    ledger = {'schema_version': 1, 'kind': 'sealed_confirmatory_admission_ledger',
        'status': 'sealed_before_model_scoring', 'lock_sha256': 'a'*64,
        'parent_seed_registry_sha256': registry_ref['sha256'], 'cohort': 'id_confirmation',
        'namespace': ident['namespace'], 'target_admitted_parents': 160, 'maximum_attempted_parents': 192,
        'attempts': attempts}
    ledger_ref = _write(tmp_path, 'ledger.json', ledger)
    admission = pred.admit_prediction('synthetic_lock_not_real.json', 'a'*64,
        ledger_ref['path'], ledger_ref['sha256'], 'candidate', 101, 0, project_root=tmp_path)
    assert validator_calls and check_calls
    assert not (tmp_path/'never_open_labels.npz').exists()
    assert not (tmp_path/'never_open_census.json').exists()
    return admission


def test_raw_observed_contract_preserves_exact_query_grid(raw):
    ident = identity(raw); sequence = pred.validate_observed_npz(observed_bytes(raw, ident), ident, 'a'*64)
    assert sequence.x.shape[1] == 8
    assert sequence.query_node_ids.tolist() == ident['query_ids']
    assert np.array_equal(sequence.positions.numpy()*1000000., ident['primary_grid']['positions_bp'])


@pytest.mark.parametrize('mutation', [
    lambda a: a.update(labels=np.zeros((128, 64))),
    lambda a: a.update(truth_probabilities=np.ones((128, 64, 4))),
    lambda a: a.update(positions=np.arange(128)),
    lambda a: a.update(x=np.pad(a['x'], ((0, 0), (0, 1)))),
    lambda a: a['x'].__setitem__((0, 0), 1.),
    lambda a: a['x'].__setitem__((0, 4), 0.),
    lambda a: a['x'].__setitem__((0, 6), .5),
    lambda a: a['x'].__setitem__((0, 7), .123),
    lambda a: a['clades'].__setitem__((0, 0), 1.-a['clades'][0, 0]),
    lambda a: a.update(query_node_ids=a['query_node_ids'][::-1].copy()),
    lambda a: a.update(window_bounds=np.array([.01, 1.])),
    lambda a: a.update(node_ids=a['node_ids'].astype(np.int32)),
    lambda a: a.update(metadata_json=np.array('{"schema_version":1,"schema_version":1}')),
    lambda a: a.update(metadata_json=np.array('opaque', dtype=object)),
])
def test_truth_extra_fields_roles_and_correspondence_reject(raw, mutation):
    ident = identity(raw)
    with pytest.raises((ValueError, TypeError)):
        pred.validate_observed_npz(observed_bytes(raw, ident, mutation), ident, 'a'*64)


@pytest.mark.parametrize('family', pred.FAMILIES)
def test_all_ten_pure_factories_and_observed_adapters(tmp_path, raw, monkeypatch, family):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch, family)
    model = pred.make_frozen_model(admitted)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    pred.configure_deterministic_development()
    seq = pred.validate_observed_npz(admitted.source_content, admitted.identity, admitted.lock.sha256)
    primary = pred._forward_grid(model, seq, admitted, 'cpu').copy()
    dense = pred._forward_grid(model, pred._sequence_for_grid(seq, admitted.identity, 'dense'), admitted, 'cpu')
    replay = pred._forward_grid(model, seq, admitted, 'cpu')
    assert primary.shape == (128, 64, 4) and dense.shape == (4096, 64, 4)
    assert np.array_equal(primary, replay)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('control,geometry', [('gain_matched_diagonal', 'plain'), ('static_local', 'plain'),
                                              ('noncontextual', 'plain'), ('matched_scalar_age_full', 'age')])
def test_conditional_control_factories_and_geometry(tmp_path, raw, monkeypatch, control, geometry):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch, control=control, geometry=geometry)
    pred.configure_deterministic_development(); model = pred.make_frozen_model(admitted)
    seq = pred.validate_observed_npz(admitted.source_content, admitted.identity, admitted.lock.sha256)
    logits = pred._forward_grid(model, seq, admitted, 'cpu')
    assert logits.shape == (128, 64, 4) and np.isfinite(logits).all()


def test_age_candidate_adapter(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch, family='candidate', geometry='age')
    pred.configure_deterministic_development()
    seq = pred.validate_observed_npz(admitted.source_content, admitted.identity, admitted.lock.sha256)
    assert pred._forward_grid(pred.make_frozen_model(admitted), seq, admitted, 'cpu').shape == (128, 64, 4)


def test_complete_prediction_receipt_and_analysis_header(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch)
    monkeypatch.setattr(pred, 'require_storage', lambda **kw: 0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    result = pred.predict_confirmatory(admitted, 'cell')
    assert result['values']['status'] == 'success'
    assert result['values']['target_artifacts_opened'] is False
    assert result['values']['cpu_validation_only'] is True
    assert result['values']['cuda_execution_performed'] is False
    content = (tmp_path/result['prediction']['path']).read_bytes()
    header, arrays = pred._safe_npz(content, pred.PREDICTION_ARRAYS)
    assert header == pred.prediction_header(admitted)
    assert arrays['primary_logits'].shape == (128, 64, 4)
    assert arrays['dense_logits'].shape == (4096, 64, 4)
    assert pred._sha((tmp_path/result['receipt']['path']).read_bytes()) == result['receipt']['sha256']
    with pytest.raises(ValueError, match='Fresh exclusive'):
        pred.predict_confirmatory(admitted, 'cell')


def test_failed_runtime_preserves_partial_terminal(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch)
    monkeypatch.setattr(pred, 'require_storage', lambda **kw: 0)
    def fail(device, runtime):
        runtime['initialization_stage'] = 'launcher_lease'; raise RuntimeError('missing held lease')
    monkeypatch.setattr(pred, 'initialize_execution', fail)
    with pytest.raises(RuntimeError, match='missing held lease'): pred.predict_confirmatory(admitted, 'failed')
    value = json.loads((tmp_path/'failed/terminal.json').read_text())
    assert value['status'] == 'failed' and value['prediction'] is None
    assert value['runtime']['initialization_stage'] == 'launcher_lease'
    assert (tmp_path/'failed/startup.json').exists()
    assert not (tmp_path/'failed/prediction.npz').exists()


def test_checkpoint_schema_and_fixed_buffer_corruption(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch, family='candidate')
    ref = admitted.seed_bundle['checkpoint']; p = tmp_path/ref['path']
    checkpoint = torch.load(p, weights_only=True); key = next(iter(checkpoint['model']))
    checkpoint['model'][key] = checkpoint['model'][key].reshape(-1)[:1]
    stream = io.BytesIO(); torch.save(checkpoint, stream); p.write_bytes(stream.getvalue())
    admitted.capture.files.clear(); ref['sha256'] = pred._sha(stream.getvalue())
    with pytest.raises(ValueError, match='state keys/shapes'): pred.make_frozen_model(admitted)


def test_in_memory_and_input_byte_drift_reject(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch)
    admitted.identity['query_ids'].reverse()
    with pytest.raises(ValueError, match='In-memory'): admitted.check()


def test_captured_file_drift_reject(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch)
    p = tmp_path/admitted.observed_ref['path']; p.write_bytes(p.read_bytes()+b'drift')
    with pytest.raises(ValueError, match='drift'): admitted.check()


def test_cuda_guard_order_rejects_before_device_inspection(monkeypatch):
    events = []; runtime = {}
    import setproctitle
    monkeypatch.setattr(pred, 'set_research_title', lambda: events.append('title'))
    monkeypatch.setattr(setproctitle, 'getproctitle', lambda: pred.PROCESS_TITLE)
    monkeypatch.setattr(pred, 'configure_deterministic_development', lambda: events.append('policy') or pred.DETERMINISTIC_POLICY)
    def lease(): events.append('lease'); raise RuntimeError('no lease')
    monkeypatch.setattr(pred, 'validate_launcher_lease', lease)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('CUDA inspected before held lease'))
    with pytest.raises(RuntimeError, match='no lease'): pred.initialize_execution('cuda', runtime)
    assert events == ['title', 'policy', 'lease']
    assert runtime['initialization_stage'] == 'launcher_lease'


@pytest.mark.parametrize('mutation', [
    lambda v: v.update(status='still_generating'),
    lambda v: v.update(attempts=v['attempts'][:1]),
    lambda v: v['attempts'][0].update(label_blind=False),
    lambda v: v['attempts'][1].update(parent_id=0),
    lambda v: v['attempts'][1].update(observed_identity=v['attempts'][0]['observed_identity']),
    lambda v: v.update(lock_sha256='0'*64),
    lambda v: v.update(namespace='other-cohort'),
])
def test_sealed_ledger_rejection_before_source_or_targets(tmp_path, raw, monkeypatch, mutation):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch)
    ledger = copy.deepcopy(admitted.ledger); mutation(ledger)
    ref = _write(tmp_path, 'bad_ledger.json', ledger)
    with pytest.raises(ValueError):
        pred.admit_prediction('synthetic_lock_not_real.json', 'a'*64, ref['path'], ref['sha256'],
            'candidate', 101, 0, project_root=tmp_path)


@pytest.mark.parametrize('mutation', [
    lambda v: v.update(truth_free_observed_inputs=False),
    lambda v: v['parent_seeds'].update(inference_seed=999),
    lambda v: v.update(diploid_pairs=[[0, 0]]*32),
    lambda v: v.update(artificial_tile_edges_bp=[100000.*i for i in range(1, 8)]),
    lambda v: v.update(source_artifacts=v['source_artifacts']*2),
    lambda v: v.update(source_artifacts=[{'path': 'never_open_labels.npz', 'sha256': 'b'*64}]),
    lambda v: v.update(query_ids=v['query_ids'][::-1]),
])
def test_identity_release_contract_rejection(tmp_path, raw, monkeypatch, mutation):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch)
    ident = copy.deepcopy(admitted.identity); mutation(ident)
    identity_ref = _write(tmp_path, 'bad_identity.json', ident)
    ledger = copy.deepcopy(admitted.ledger); ledger['attempts'][0]['observed_identity'] = identity_ref
    ref = _write(tmp_path, 'bad_identity_ledger.json', ledger)
    with pytest.raises(ValueError):
        pred.admit_prediction('synthetic_lock_not_real.json', 'a'*64, ref['path'], ref['sha256'],
            'candidate', 101, 0, project_root=tmp_path)


def test_target_files_are_never_opened_even_if_present(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch)
    (tmp_path/'never_open_labels.npz').write_bytes(b'target poison bytes')
    (tmp_path/'never_open_census.json').write_bytes(b'target poison bytes')
    original = Path.open
    def guard(path, *args, **kwargs):
        if path.name in ('never_open_labels.npz', 'never_open_census.json'):
            pytest.fail('Prediction workflow opened target artifact')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    again = pred.admit_prediction('synthetic_lock_not_real.json', 'a'*64,
        admitted.ledger_ref['path'], admitted.ledger_ref['sha256'], 'candidate', 101, 0, project_root=tmp_path)
    monkeypatch.setattr(pred, 'require_storage', lambda **kw: 0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert pred.predict_confirmatory(again, 'label_guard')['values']['status'] == 'success'


def test_existing_published_output_never_overwritten(tmp_path):
    p = tmp_path/'exists'; p.write_bytes(b'original')
    with pytest.raises(FileExistsError): pred._publish(p, b'new', lambda: None)
    assert p.read_bytes() == b'original'
    assert not list(tmp_path.glob('.prediction_pending_*'))


def test_fixed_buffer_checkpoint_corruption_rejects(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch, family='candidate')
    model = pred.make_frozen_model(admitted); key, value = next(iter(model.named_buffers()))
    p = tmp_path/admitted.seed_bundle['checkpoint']['path']
    checkpoint = torch.load(p, weights_only=True); checkpoint['model'][key] = value+1
    stream = io.BytesIO(); torch.save(checkpoint, stream); p.write_bytes(stream.getvalue())
    admitted.capture.files.clear(); admitted.seed_bundle['checkpoint']['sha256'] = pred._sha(stream.getvalue())
    with pytest.raises(ValueError, match='fixed factory buffer'): pred.make_frozen_model(admitted)


def test_missing_checkpoint_not_fabricated_for_neural_family(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch, family='candidate')
    admitted.seed_bundle['checkpoint'] = None
    with pytest.raises(ValueError, match='Only deterministic'): pred.make_frozen_model(admitted)


def test_runtime_hardware_and_software_binding(tmp_path, raw, monkeypatch):
    admitted = synthetic_admission(tmp_path, raw, monkeypatch)
    frozen = admitted.lock.payload['captured_runtime']
    frozen['software'] = dict(python_executable='python', torch='torch', torch_cuda='cuda', torch_build='build')
    frozen['device'] = dict(name='frozen hardware', compute_capability=[9, 0])
    runtime = dict(device_kind='cuda', software=copy.deepcopy(frozen['software']), device=copy.deepcopy(frozen['device']))
    pred._runtime_matches_lock(admitted, runtime)
    runtime['software']['torch_build'] = 'changed'
    with pytest.raises(ValueError, match='software'): pred._runtime_matches_lock(admitted, runtime)
    runtime['software'] = frozen['software']; runtime['device']['name'] = 'different hardware'
    with pytest.raises(ValueError, match='hardware'): pred._runtime_matches_lock(admitted, runtime)

"""Replay consequential audit mutations using handmade CPU artifacts only."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arggnn import development_execution as execution
from arggnn import development_training_bindings as common
from arggnn import argmix_development_bindings as raw_binding
from arggnn import allrecord_development_bindings as all_binding
from arggnn import canonical_development_bindings as canonical_binding
from arggnn import priorart_development_bindings as prior_binding
from arggnn import train_argmix_development as raw
from arggnn import train_allrecord_development as allrecord
from arggnn import train_anchor_temporal_development as temporal
from arggnn import train_anchor_development as candidate
from test_argmix_development import tiny_parents, tiny_settings
from test_allrecord_development import one_parent_cases
from test_allrecord_boundary_temporal import model as allrecord_model
from test_development_training_bindings import linked_sources
from test_development_execution import restore_execution_flags
from test_train_anchor_development import test_tiny_cpu_fit_stores_hash_bound_raw_logit_predictions as _candidate_tiny_fit


@pytest.fixture(autouse=True)
def strict_execution(restore_execution_flags):
    torch.set_num_threads(1)
    execution.configure_deterministic_development()


@pytest.mark.parametrize('family', ['candidate', 'gru', 'transformer', 'reference_smoother'])
def test_common_rejects_changed_runtime_settings_under_unchanged_file(tmp_path, monkeypatch, family):
    completion, completion_path, _ = linked_sources(tmp_path, monkeypatch)
    settings = common.make_configs(completion, completion_path, common.sha(completion_path))[family]
    path = tmp_path / (family + '.json')
    path.write_text(json.dumps(settings))
    digest = common.sha(path)
    _, lock = common.capture_completion_sources(completion_path, common.sha(completion_path))
    common.check_live(settings, path, digest, source_snapshot=lock)
    changed = deepcopy(settings)
    if family == 'candidate':
        changed['hidden_dim'] = 16
    else:
        key = 'bandwidth_bp' if family == 'reference_smoother' else 'hidden_dim'
        changed['model_specs'][changed['models'][0]][key] = 1 if family == 'reference_smoother' else 16
    with pytest.raises(RuntimeError, match='Runtime settings differ'):
        common.check_live(changed, path, digest, source_snapshot=lock)
    assert common.sha(path) == digest


@pytest.mark.parametrize('family', ['raw', 'canonical', 'priorart'])
@pytest.mark.parametrize('drift', ['settings', 'policy'])
def test_composed_snapshot_rejects_runtime_mutation(tmp_path, family, drift):
    snapshot = {'raw': raw_binding.IntegritySnapshot,
                'canonical': canonical_binding.CanonicalIntegritySnapshot,
                'priorart': prior_binding.PriorArtSnapshot}[family]()
    path = tmp_path / 'configuration.json'
    path.write_text('{"hidden_dim":32}')
    settings = snapshot.control(path, common.sha(path))
    composed = snapshot if family == 'raw' else getattr(snapshot, 'shared' if family == 'canonical' else 'source')
    composed.bind_runtime_configuration(path, settings)
    snapshot.check(full_sources=True)
    if drift == 'settings':
        settings['hidden_dim'] = 16
    else:
        torch.use_deterministic_algorithms(False)
    with pytest.raises(RuntimeError, match='Runtime settings differ|policy drift'):
        snapshot.check(full_sources=True)


def test_captured_control_dictionary_cannot_change_under_original_hash(tmp_path):
    path = tmp_path / 'configuration.json'
    path.write_text('{"hidden_dim":32}')
    captured = all_binding.capture_control(path, common.sha(path))
    captured.check(full=True)
    captured.values['hidden_dim'] = 16
    with pytest.raises(RuntimeError, match='values differ from original bytes'):
        captured.check(full=True)


@pytest.mark.parametrize('namespace', ['test', 'confirmation', 'confirmatory', 'final_test', 'final-test'])
def test_all_development_path_guards_reject_heldout_before_open(monkeypatch, namespace):
    path = common.ROOT / 'data' / namespace / 'hypothetical.json'
    def forbidden(*args, **kwargs):
        pytest.fail('Held-out control opened before namespace rejection')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    with pytest.raises(RuntimeError, match='held-out'):
        temporal.captured_protocol(path, 'a' * 64)
    with pytest.raises(ValueError, match='Held-out'):
        raw_binding.development_path(path)


def test_temporal_effective_policy_guard_precedes_source_operation(monkeypatch):
    torch.use_deterministic_algorithms(False)
    monkeypatch.setattr(temporal, 'check_live', lambda *a, **kw: pytest.fail('Policy drift allowed source check'))
    with pytest.raises(RuntimeError, match='policy drift'):
        temporal.check_active_execution({}, None, None)


def test_raw_policy_drift_after_validation_rejects_before_checkpoint(tmp_path):
    checkpoint = tmp_path / 'selected.pt'
    configuration = {'id':'synthetic', 'hidden_dim':8, 'dropout':0., 'learning_rate':.001}
    with pytest.raises(RuntimeError, match='policy drift'):
        raw.train_configuration(configuration, {**tiny_settings(), 'epochs':2}, tiny_parents(),
            torch.device('cpu'), 101, checkpoint,
            {'protocol_sha256':'a'*64, 'implementation_sha256':raw_binding.implementation_hashes()},
            callback=lambda row: torch.use_deterministic_algorithms(False))
    assert not checkpoint.exists()


def test_allrecord_policy_drift_after_validation_rejects_before_checkpoint(tmp_path):
    case, banks = one_parent_cases()
    checkpoint = tmp_path / 'selected.pt'
    with pytest.raises(RuntimeError, match='policy drift'):
        allrecord.train_configuration(allrecord_model('allrecord_gru'),
            {'model_learning_rates':{'synthetic':.001}}, 'synthetic', 101,
            {'train':[case], 'val':[case]}, banks, checkpoint,
            {'protocol_sha256':'a'*64, 'implementation_sha256':allrecord._IMPORTED_PINS},
            integrity_check=lambda **kw: None,
            callback=lambda row: torch.use_deterministic_algorithms(False))
    assert not checkpoint.exists()


def test_allrecord_final_replay_policy_drift_rejects(tmp_path, monkeypatch):
    case, banks = one_parent_cases()
    calls = []
    def evaluation(*args):
        calls.append(None)
        if len(calls) == 10:
            torch.use_deterministic_algorithms(False)
        return [{'parent_id':case.parent_id, 'nll':1.25}], [torch.zeros(len(case.positions), 2, 4)]
    monkeypatch.setattr(allrecord, 'evaluate_allrecord', evaluation)
    with pytest.raises(RuntimeError, match='policy drift'):
        allrecord.train_configuration(allrecord_model('allrecord_gru'),
            {'model_learning_rates':{'synthetic':.001}}, 'synthetic', 101,
            {'train':[case], 'val':[case]}, banks, tmp_path/'selected.pt',
            {'protocol_sha256':'a'*64, 'implementation_sha256':allrecord._IMPORTED_PINS},
            integrity_check=lambda **kw: None)
    assert len(calls) == 10


def test_candidate_validation_policy_drift_records_failure_and_no_completion(tmp_path, monkeypatch):
    original = candidate.evaluate
    def drift(*args, **kwargs):
        torch.use_deterministic_algorithms(False)
        return original(*args, **kwargs)
    monkeypatch.setattr(candidate, 'evaluate', drift)
    with pytest.raises(RuntimeError, match='policy drift'):
        _candidate_tiny_fit(tmp_path, monkeypatch, 'none')
    failure = json.loads((tmp_path/'run/failure.json').read_text())
    assert failure['test_access'] is False
    assert not (tmp_path/'run/results.json').exists()
    assert not torch.cuda.is_initialized()

"""Synthetic CPU audit. Writes only its versioned TMPDIR and this review folder.
No chromosomes, native inference, GPU contexts, Git actions or real run changes.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import json
import os
import sys
import torch

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
TMP = Path(os.environ.get('ARGGNN_AUDIT_TMP', str(ROOT/'.runtime/tmp/development_integration_audit_v1'))).resolve()
if ROOT not in TMP.parents:
    raise ValueError('Audit temporary directory must remain inside the project')
TMP.mkdir(parents=True, exist_ok=True)
REVIEW = ROOT/'research/reviews/development_integration_audit_v1'
sys.path.insert(0, str(ROOT/'tests'))
from test_train_anchor_development import full_protocol
from test_anchor_temporal_baselines import prepared
import arggnn.train_anchor_development as development
import arggnn.train_anchor_temporal as temporal
from arggnn.confirmatory_metrics import logit_metrics


def guard_checks():
    bad = ['/tmp/absolute', '../traversal', 'some/subdir', '.hidden', 'bad\nline', 'x'*129]
    canonical = ['audit_plain_d2_lr0.001', 'audit_anchor_reference_10kb_pc0.001', 'A_1-z.2']
    reports = {}
    for family, validator, source in (
        ('development', development.checked_development_protocol, full_protocol()),
        ('temporal', temporal.checked_temporal_protocol, json.loads((ROOT/'configs/anchor_inferred_cohort64_temporal_baselines_pilot_v1.json').read_text())),
    ):
        reports[family] = {}
        for name in bad + canonical:
            p = json.loads(json.dumps(source))
            old = p['models'][0]
            p['models'][0] = name
            p['model_specs'][name] = p['model_specs'].pop(old)
            if 'model_learning_rates' in p:
                p['model_learning_rates'][name] = p['model_learning_rates'].pop(old)
            path = TMP/f'{family}_id_check.json'
            path.write_text(json.dumps(p))
            try:
                validator(path)
                accepted = True
            except RuntimeError:
                accepted = False
            assert accepted == (name in canonical), (family, name, accepted)
            reports[family][repr(name)] = accepted
    return reports


def run_temporal(*, mutate_config=False):
    directory = TMP/('temporal_drift' if mutate_config else 'temporal_roundtrip')
    directory.mkdir()
    data = directory/'data'
    data.mkdir()
    manifest = data/'manifest.json'
    manifest.write_text(json.dumps({'kind': 'synthetic_audit_bank_no_simulation'}))
    pack = prepared(anchors=32)
    groups = {}
    for split in ('train', 'val'):
        labels = torch.arange(14).reshape(7, 2).remainder(4)
        tile = SimpleNamespace(bank=pack.bank, labels=labels,
            parent_position_indices=torch.arange(7))
        case = SimpleNamespace(tiles=[tile], labels=labels,
            positions=pack.bank.positions, query_node_ids=pack.bank.query_node_ids,
            metadata={'split': split, 'replicate': 0, 'num_classes': 4},
            parent_id=f'synthetic-audit:{split}:0')
        groups[split] = [case]
    dataset = SimpleNamespace(manifest_sha256=temporal.sha(manifest), cases=lambda split: groups[split])
    p = json.loads((ROOT/'configs/anchor_inferred_cohort64_temporal_baselines_pilot_v1.json').read_text())
    names = ['anchor_gru_d2'] if mutate_config else ['anchor_gru_d2', 'anchor_temporal_transformer_d2', 'anchor_reference_10kb_pc0.001']
    p.update(hidden_dim=8, num_layers=1, latent_dim=3, epochs=1, patience=1,
        models=names, model_specs={name: p['model_specs'][name] for name in names},
        splits={'train': 1, 'val': 1, 'test': 0},
        boundary_dataset_manifest_sha256=dataset.manifest_sha256,
        stage='synthetic_audit_roundtrip', interpretation='No ancestry performance evidence')
    config = directory/'config.json'
    config.write_text(json.dumps(p))
    output = directory/'run'
    original = temporal.evaluate_temporal
    changed = False
    def evaluate_and_mutate(*args):
        nonlocal changed
        answer = original(*args)
        if mutate_config and not changed:
            changed = True
            changed_config = json.loads(config.read_text())
            changed_config['audit_mutation_after_validation'] = True
            config.write_text(json.dumps(changed_config))
        return answer
    argv = ['audit', '--config', str(config), '--data', str(data), '--output', str(output), '--device', 'cpu']
    with patch.object(sys, 'argv', argv), patch.object(temporal, 'load_boundary_dataset', return_value=dataset), patch.object(temporal, 'require_storage', return_value=0), patch.object(temporal, 'evaluate_temporal', side_effect=evaluate_and_mutate):
        temporal.main()
    result = json.loads((output/'results.json').read_text())
    fact = {'protocol_copy_sha256': temporal.sha(output/'protocol.json'),
            'results_protocol_sha256': result['protocol_sha256'],
            'current_config_sha256': temporal.sha(config), 'models': {}}
    for name, record in result['models'].items():
        payload = torch.load(record['validation_logits'], weights_only=True)
        assert payload['parent_ids'] == ['synthetic-audit:val:0']
        assert torch.equal(payload['positions'][0], groups['val'][0].positions)
        assert torch.equal(payload['query_node_ids'][0], groups['val'][0].query_node_ids)
        metrics = logit_metrics(payload['logits'][0].double().numpy(), groups['val'][0].labels.numpy())
        assert abs(metrics['nll'] - record['validation_nll']) < 1e-12
        fact['models'][name] = {'prediction_protocol_sha256': payload['protocol_sha256'],
            'saved_dtype': str(payload['logits'][0].dtype), 'raw_nll_roundtrip_matches': True}
    if mutate_config:
        assert changed
        assert fact['protocol_copy_sha256'] == fact['results_protocol_sha256']
        assert fact['models']['anchor_gru_d2']['prediction_protocol_sha256'] != fact['results_protocol_sha256']
        fact['runner_completed_despite_config_drift'] = True
    return fact


def preparation_guard_check():
    spec = importlib.util.spec_from_file_location('fresh_audit', ROOT/'scripts/prepare_fresh_development.py')
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    plan, _ = fresh.load_plan()
    plan['storage']['minimum_cpu_available_bytes'] = 1
    plan['admission'].update(min_physical_coverage_fraction=.01, min_retained_grid_points=1, max_packed_nodes=1000000000)
    config = TMP/'audit_relaxed_preparation_plan.json'
    config.write_text(json.dumps(plan))
    accepted, _ = fresh.load_plan(config)
    return {'modified_threshold_plan_admitted': True, 'minimum_cpu_available_bytes': accepted['storage']['minimum_cpu_available_bytes'], **accepted['admission']}


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = {'model_id_regression': guard_checks(),
        'temporal_three_family_roundtrip': run_temporal(),
        'temporal_config_drift_reproducer': run_temporal(mutate_config=True),
        'preparation_threshold_guard': preparation_guard_check()}
    (REVIEW/'audit_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

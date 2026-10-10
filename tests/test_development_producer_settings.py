"""Actual producer assemblers on explicit handmade completion facts, CPU only.

This is a pure control-contract fixture. Tensor/source admission is not mocked
into a genuine cohort and the declared fake raw files are never opened.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arggnn.resources import ROOT
from arggnn import development_selection as selection
from arggnn import development_selection_bindings as combined
from arggnn import development_training_bindings as common
from arggnn import argmix_development_bindings as raw
from arggnn import canonical_development_bindings as canonical
from arggnn import allrecord_development_bindings as allrecord
from arggnn import priorart_development_bindings as prior


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def factual_assembly(tmp_path):
    raw_catalog = {}
    records = []
    for split, count in (('train', 64), ('val', 16)):
        for i in range(count):
            row = {'split':split, 'replicate':i}
            for field, hash_field in (('metadata', 'metadata_sha256'), ('tensor_file', 'tensor_sha256'),
                                      ('prepared_tensor_file', 'prepared_tensor_sha256')):
                path = str(tmp_path/(split+'_'+str(i)+'_'+field+'.synthetic'))
                digest = hashlib.sha256(path.encode()).hexdigest()
                row[field], row[hash_field] = path, digest
                raw_catalog[path] = digest
            records.append(row)
    inferred = save(tmp_path/'inferred.json', {'synthetic_control_facts_only':True, 'replicates':records})
    protocol = save(tmp_path/'protocol.json', {'synthetic_control_facts_only':True})
    oracle = save(tmp_path/'oracle.json', {'synthetic_control_facts_only':True})
    boundary = save(tmp_path/'boundary.json', {'synthetic_control_facts_only':True})
    age = save(tmp_path/'age.json', {'synthetic_control_facts_only':True})
    freeze = save(tmp_path/'freeze.json', {'synthetic_control_facts_only':True})
    facts = {'oracle':oracle, 'inferred':{**inferred, 'protocol_path':protocol['path'],
        'protocol_sha256':protocol['sha256']}, 'boundary':{**boundary, 'preparation_sha256':'b'*64},
        'age':{**age, 'preparation_sha256':'c'*64}}
    completion = {'all_stages_complete':True, 'train_parents':64, 'validation_parents':16,
        'test_parents':0, 'inferred_boundary_and_age_tiles':640,
        'training_or_confirmatory_selection_frozen':False, 'stage_manifests':facts,
        'development_execution_freeze_path':freeze['path'],
        'development_execution_freeze_file_sha256':freeze['sha256']}
    receipt = save(tmp_path/'completion.json', completion)
    settings = common.make_configs(completion, receipt['path'], receipt['sha256'])
    for module, family in ((raw, 'argmix_full_tree'), (canonical, 'canonical_gru')):
        path = ROOT/module.PLAN
        plan = json.loads(path.read_bytes())
        args = (plan, path, common.sha(path), completion, receipt['path'], receipt['sha256'])
        settings[family] = module.bound_settings(*args, raw_catalog) if module is canonical else module.bound_settings(*args)
    path = ROOT/'configs/allrecord_development_search_plan_v2.json'
    settings.update(allrecord.bound_configs(json.loads(path.read_bytes()), path, common.sha(path),
        completion, receipt['path'], receipt['sha256']))
    path = ROOT/prior.PLAN
    settings.update(prior.bound_settings(json.loads(path.read_bytes()), path, common.sha(path),
        completion, receipt['path'], receipt['sha256']))
    return SimpleNamespace(settings=settings, completion=completion, receipt=receipt,
        raw_catalog=raw_catalog, capture=selection._Capture(ROOT),
        lock=SimpleNamespace(hashes=deepcopy(raw_catalog)))


def validate(f, family, settings=None):
    return combined.validate_producer_settings(family, f.settings[family] if settings is None else settings,
        completion=f.completion, completion_path=f.receipt['path'],
        completion_sha256=f.receipt['sha256'], capture=f.capture, source_snapshot=f.lock)


@pytest.mark.parametrize('family', combined.FAMILIES)
def test_exact_actual_ten_producer_assemblies_pass_without_raw_file_reads(factual_assembly, family, monkeypatch):
    f = factual_assembly
    original = Path.read_bytes
    def guarded(path):
        assert str(path) not in f.raw_catalog, 'Pure contract check opened fake source tensor/metadata'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    assert validate(f, family) == f.settings[family]


@pytest.mark.parametrize('family, mutation', [
    ('candidate', 'classes'), ('candidate', 'width'), ('candidate', 'latent'),
    ('candidate', 'anchors'), ('candidate', 'hidden_type'),
    ('gru', 'temporal_depth'), ('transformer', 'heads'),
    ('reference_smoother', 'bandwidth'), ('canonical_gru', 'classes'),
    ('argmix_full_tree', 'layers'), ('allrecord_gru', 'temporal_depth'),
    ('allrecord_transformer', 'source_depth'), ('gst_ca_span', 'input_dim'),
    ('graphssm_zoh', 'input_dim'),
])
def test_actual_producer_settings_reject_mutation(factual_assembly, family, mutation):
    f = factual_assembly
    changed = deepcopy(f.settings[family])
    if mutation == 'classes': changed['num_classes'] = 2
    elif mutation == 'width': changed['hidden_dim'] = 16
    elif mutation == 'hidden_type': changed['hidden_dim'] = 32.0
    elif mutation == 'latent': changed['latent_dim'] = 4
    elif mutation == 'anchors': changed['num_anchors'] = 16
    elif mutation == 'layers': changed['num_layers'] = 1
    elif mutation == 'input_dim': changed['input_dim'] = 47
    else:
        name = changed['models'][0]
        key = {'heads':'transformer_heads', 'bandwidth':'bandwidth_bp'}.get(mutation, mutation)
        changed['model_specs'][name][key] = 1
    with pytest.raises((ValueError, RuntimeError)):
        validate(f, family, changed)


@pytest.mark.parametrize('family, key', [
    ('candidate', 'boundary_data'), ('gru', 'age_data'),
    ('argmix_full_tree', 'search_plan'), ('canonical_gru', 'dataset_manifest'),
    ('allrecord_gru', 'boundary_data'), ('gst_ca_span', 'dataset_protocol'),
])
def test_consumed_heldout_paths_reject_before_open(factual_assembly, family, key, monkeypatch):
    f = factual_assembly
    changed = deepcopy(f.settings[family])
    changed[key] = str(ROOT/'data/final_test/hypothetical.json')
    monkeypatch.setattr(Path, 'read_bytes', lambda *a: pytest.fail('Held-out alias reached a file read'))
    with pytest.raises(ValueError, match='Held-out'):
        validate(f, family, changed)


def test_canonical_catalog_is_exact_and_verified(factual_assembly):
    f = factual_assembly
    changed = deepcopy(f.settings['canonical_gru'])
    changed['source_artifact_sha256'].pop(next(iter(f.raw_catalog)))
    with pytest.raises(ValueError, match='Complete typed producer settings'):
        validate(f, 'canonical_gru', changed)
    f.lock.hashes.pop(next(iter(f.raw_catalog)))
    with pytest.raises(ValueError, match='absent from verified completion catalog'):
        validate(f, 'canonical_gru')


def test_unknown_extra_configuration_fields_reject(factual_assembly):
    f = factual_assembly
    changed = deepcopy(f.settings['candidate'])
    changed['unreviewed_extra_budget'] = 500
    with pytest.raises(ValueError, match='Complete typed producer settings'):
        validate(f, 'candidate', changed)

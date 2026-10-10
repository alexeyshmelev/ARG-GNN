"""Independent CPU-only integrity replay with a real seed registry and fake bytes.

No simulations, native inference, real dataset loading, or GPU calls. Full source
catalog and file hashes are real. Positive full-loader contracts are stubbed.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
import sys
import time
import os
import pytest
import torch

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/reviews/integrity_fix_reaudit_v1'
TMP = ROOT / '.runtime/tmp/integrity_fix_reaudit_v1' / ('independent_' + str(time.time_ns()))
TMP.mkdir(parents=True)
import arggnn.development_training_bindings as b
import arggnn.train_canonical_gru as canonical
import arggnn.train_anchor_development as candidate
import arggnn.train_anchor_temporal_development as temporal

RESULTS = {'synthetic_only': True, 'temporary_directory': str(TMP), 'checks': []}
def record(name, **facts):
    row = {'check': name, **facts}
    RESULTS['checks'].append(row)
    print(json.dumps(row), flush=True)
def expect_rejection(fn, pattern):
    try:
        fn()
    except (RuntimeError, ValueError, TypeError, FileNotFoundError) as error:
        assert pattern in str(error), (pattern, str(error))
        return {'type': type(error).__name__, 'message': str(error)}
    raise AssertionError('Expected rejection: ' + pattern)
def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(payload, sort_keys=True).encode() if isinstance(payload, (dict, list)) else payload.encode())
    return str(path), b.sha(path)

# Use the actual installed seed helper outputs, with no registry monkeypatch.
registry = b.expected_parent_registry()
dirs = {name: TMP / name for name in ('oracle', 'inferred', 'boundary', 'age')}
for directory in dirs.values(): directory.mkdir()
op = {'splits': dict(b.SPLITS), 'master_seed': 20261101}
ip = {**op, 'inference': {'master_seed': 20261102}}
oc, och = save(dirs['oracle'] / 'protocol.json', op)
ic, ich = save(dirs['inferred'] / 'protocol.json', ip)
records = {s: [] for s in ('oracle', 'inferred')}
artifacts = {}
for row in registry:
    parent = f"rep{row['replicate']:03d}"
    od = dirs['oracle'] / row['split'] / parent
    raw = dirs['inferred'] / row['split'] / parent / 'inference/dataset'
    md = {'split': row['split'], 'replicate': row['replicate'], 'seeds': row['seeds'], 'master_seed': 20261101}
    originals = None
    for stage, directory in (('oracle', od), ('inferred', raw)):
        rec = {'split': row['split'], 'replicate': row['replicate'], 'directory': str(directory), 'seeds': row['seeds']}
        for field, digest_field, filename, value in (
            ('metadata', 'metadata_sha256', 'metadata.json', md),
            ('tensor_file', 'tensor_sha256', 'tensors.pt.gz', 'synthetic raw'),
            ('prepared_tensor_file', 'prepared_tensor_sha256', 'prepared.pt', 'synthetic prepared')):
            rec[field], rec[digest_field] = save(directory / filename, value)
        if stage == 'oracle':
            originals = rec
            _, truth_hash = save(od / 'truth.trees.gz', 'synthetic truth')
            _, observed_hash = save(od / 'observed.trees.gz', 'synthetic observed')
        else:
            rec['source_parent'] = {**originals, 'truth_archive_sha256': truth_hash, 'clean_observed_archive_sha256': observed_hash}
            rec['inference_seed'] = row['inference_seed']
            native = raw.parent
            _, rec['native_inferred_tree_sha256'] = save(native / 'inferred.trees', 'synthetic native')
            _, rec['execution_sha256'] = save(native / 'execution.json', {})
            rec['observed_tree_file'], rec['observed_tree_sha256'] = save(raw / 'observed.trees.gz', 'synthetic inferred observed')
            genotype, genotype_hash = save(native.parent / 'inputs/input.haps', 'synthetic genotype')
            _, rec['preparation_sha256'] = save(native.parent / 'inputs/preparation.json', {'paths': {'haps': genotype}, 'files_sha256': {'haps': genotype_hash}})
        records[stage].append(rec)
    if not artifacts:
        artifacts = {'oracle_metadata': originals['metadata'], 'oracle_raw': originals['tensor_file'], 'oracle_prepared': originals['prepared_tensor_file'],
                     'oracle_truth_archive': str(od / 'truth.trees.gz'), 'oracle_observed_archive': str(od / 'observed.trees.gz'),
                     'inferred_raw': rec['tensor_file'], 'inferred_prepared': rec['prepared_tensor_file'], 'genotype': genotype,
                     'native_inferred': str(raw.parent / 'inferred.trees'), 'execution': str(raw.parent / 'execution.json'),
                     'inference_preparation': str(raw.parent.parent / 'inputs/preparation.json')}

facts = {}
om = {'protocol': op, 'protocol_sha256': och, 'replicates': records['oracle']}
facts['oracle'] = dict(zip(('path', 'sha256'), save(dirs['oracle'] / 'manifest.json', om)))
im = {'protocol': ip, 'protocol_sha256': ich, 'replicates': records['inferred'], 'source_parent': {'manifest_sha256': facts['oracle']['sha256']}}
facts['inferred'] = dict(zip(('path', 'sha256'), save(dirs['inferred'] / 'manifest.json', im)))
facts['inferred']['protocol_path'] = ic
supervision, sh = save(dirs['boundary'] / 'train/rep000/supervision.pt', 'synthetic supervision')
bank, bh = save(dirs['boundary'] / 'train/rep000/window000/bank.pt', 'synthetic bank')
bm = {'status': 'complete', 'preparation_sha256': 'b' * 64, 'source': {'manifest_sha256': facts['inferred']['sha256']},
      'parents': [{'supervision_file': 'train/rep000/supervision.pt', 'supervision_sha256': sh}],
      'records': [{'bank_file': 'train/rep000/window000/bank.pt', 'bank_sha256': bh}]}
facts['boundary'] = dict(zip(('path', 'sha256'), save(dirs['boundary'] / 'manifest.json', bm)))
facts['boundary']['preparation_sha256'] = 'b' * 64
age, ah = save(dirs['age'] / 'train/rep000/window000/age_operators.pt', 'synthetic age')
am = {'status': 'complete', 'preparation_sha256': 'c' * 64, 'boundary_manifest_sha256': facts['boundary']['sha256'],
      'boundary_preparation_sha256': 'b' * 64, 'source_manifest_sha256': facts['inferred']['sha256'],
      'records': [{'operator_file': 'train/rep000/window000/age_operators.pt', 'operator_sha256': ah}]}
facts['age'] = dict(zip(('path', 'sha256'), save(dirs['age'] / 'manifest.json', am)))
facts['age']['preparation_sha256'] = 'c' * 64
plan, plan_hash = save(TMP / 'synthetic_plan.json', {'synthetic': True})
recipe, recipe_hash = save(TMP / 'synthetic_recipe.py', 'synthetic frozen recipe')
frozen = {'kind': 'fresh_development_preparation_execution_freeze', 'confirmatory_access': False,
          'not_confirmatory_test_freeze': True, 'parent_registry': registry,
          'plan': {'expected_splits': dict(b.SPLITS), 'expected_simulation_master_seed': 20261101, 'relate_master_seed': 20261102,
                   'oracle_config': oc, 'paths': {s: str(p) for s, p in dirs.items()}},
          'config_files': {oc: och, plan: plan_hash}, 'source_code_sha256': {recipe: recipe_hash}}
freeze, freeze_hash = save(TMP / 'freeze.json', frozen)
completion = {'all_stages_complete': True, 'train_parents': 64, 'validation_parents': 16, 'test_parents': 0,
              'inferred_boundary_and_age_tiles': 640, 'training_or_confirmatory_selection_frozen': False,
              'development_execution_freeze_file_sha256': freeze_hash, 'development_execution_freeze_path': freeze, 'stage_manifests': facts}
cp, ch = save(TMP / 'completion.json', completion)
artifacts.update({s + '_manifest': f['path'] for s, f in facts.items()})
artifacts.update(freeze=freeze, recipe=recipe, genotype=genotype, supervision=supervision, boundary_bank=bank, age_cache=age, oracle_protocol=oc, inferred_protocol=ic, plan=plan)
_, lock = b.capture_completion_sources(cp, ch)
record('real_seed_registry_catalog_admission', passed=True, registry_parents=len(registry), bound_artifact_count=len(lock.hashes), no_loader_called=True)

# Mutations are confined to the fixture. Re-capture fresh signatures each time.
for name, path in artifacts.items():
    _, lock = b.capture_completion_sources(cp, ch)
    p = Path(path)
    original = p.read_bytes()
    p.write_bytes(original + b' ')
    stat_error = expect_rejection(lambda: lock.check(full=False), 'artifact changed')
    lock.signatures[str(p)] = b.artifact_signature(p)
    hash_error = expect_rejection(lambda: lock.check(full=True), 'SHA256 changed')
    p.write_bytes(original)
    record('source_mutation_' + name, passed=True, stat_rejection=stat_error, full_hash_rejection=hash_error)

_, lock = b.capture_completion_sources(cp, ch)
with pytest.MonkeyPatch.context() as mp:
    mp.setattr(b, 'sha', lambda *_a: (_ for _ in ()).throw(AssertionError('Epoch source check hashed a file')))
    lock.check(full=False)
record('epoch_source_check_uses_stats_without_rehash', passed=True)

# Held-out guard spies see no file reads. These hypothetical paths are not created.
for namespace in ('confirmatory_v1', 'final_test', 'final-test', 'confirmation', 'test'):
    target = TMP / namespace / 'manifest.json'
    reads = []
    with pytest.MonkeyPatch.context() as mp:
        def trap(p):
            reads.append(str(p))
            raise AssertionError('File read before held-out rejection')
        mp.setattr(Path, 'read_bytes', trap)
        first = expect_rejection(lambda: canonical.capture_run_snapshot(TMP / 'unused.json', target), 'Held-out paths')
        second = expect_rejection(lambda: canonical.capture_run_snapshot(target, TMP / 'unused.json'), 'Held-out paths')
    record('canonical_heldout_' + namespace, passed=True, file_reads=reads, rejection=first)

canonical_config = ROOT / 'configs/canonical_inferred_cohort64_pilot_v1.json'
cs = json.loads(canonical_config.read_bytes())
config_sha = b.sha(canonical_config)
rows = []
from arggnn.canonical_gru import CanonicalTreeGNNGRU
for spec in cs['configurations']:
    count = sum(p.numel() for p in CanonicalTreeGNNGRU(cs['input_dim'], cs['num_classes'], spec['hidden_dim'], cs['num_layers'], cs['temporal_layers'], spec['dropout']).parameters())
    for seed in cs['training_seeds']:
        rows.append({'configuration_id': spec['id'], 'configuration': spec, 'training_seed': seed, 'validation_nll': 1.0, 'parameters': count,
                     'protocol_sha256': config_sha, 'dataset_manifest_sha256': cs['dataset']['manifest_sha256'],
                     'implementation_sha256': cs['implementation_sha256'], 'source_artifact_sha256': cs['dataset']['source_artifact_sha256']})
selection = canonical.select_family_configuration(rows, cs, config_sha)
record('canonical_valid_complete_synthesis', passed=True, records=len(rows), selected=selection['selected'])
for field, value, pattern in (
    ('protocol_sha256', 'f'*64, 'exact expected original config'),
    ('configuration', {**cs['configurations'][0], 'hidden_dim': 999}, 'configuration/capacity'),
    ('parameters', 1, 'configuration/capacity'),
    ('source_artifact_sha256', {'synthetic_wrong': 'f'*64}, 'raw/prepared/metadata'),
    ('implementation_sha256', {'synthetic_wrong.py': 'f'*64}, 'implementation/dependency')):
    altered = deepcopy(rows)
    for row in altered: row[field] = value
    rejection = expect_rejection(lambda: canonical.select_family_configuration(altered, cs, config_sha), pattern)
    record('canonical_wrong_common_' + field, passed=True, rejection=rejection)

# Both runners must reject incomplete completion before loader/device spies.
# Loader stubs are used only for positive 64/16 API contracts, never model fitting.
class FakeDataset:
    records = [None] * 640
    def cases(self, split):
        return [SimpleNamespace(metadata={'split': split, 'master_seed': 20261101, 'num_classes': 4}, query_node_ids=torch.arange(64))
                for _ in range(64 if split == 'train' else 16)]
settings_by_family = b.make_configs(completion, cp, ch)
for family, runner in (('candidate', candidate), ('gru', temporal)):
    settings = settings_by_family[family]
    conf, conf_hash = save(TMP / (family + '_config.json'), settings)
    events = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr('arggnn.boundary_dataset.load_boundary_dataset', lambda *_a, **_k: events.append('boundary_loader') or FakeDataset())
        mp.setattr('arggnn.age_dataset.load_age_caches', lambda *_a, **_k: events.append('age_loader') or {i: None for i in range(640)})
        admitted = b.admit_run(settings, conf, conf_hash, imported_pins=b.snapshot())
    record(family + '_real_admission_with_stubbed_full_loaders', passed=True, events=events, source_artifact_count=len(admitted[3].hashes), stubbed_full_loaders=True)
    for fault in ('missing_receipt', 'incomplete_receipt', 'dependency_mismatch', 'wrong_stage_directory'):
        modified = deepcopy(settings)
        events = []
        if fault == 'missing_receipt': modified['completion_receipt'] = str(TMP / 'absent_completion.json')
        elif fault == 'incomplete_receipt':
            comp = deepcopy(completion); comp['all_stages_complete'] = False
            modified['completion_receipt'], modified['completion_receipt_sha256'] = save(TMP / 'incomplete_completion.json', comp)
        elif fault == 'dependency_mismatch': modified['bound_implementation_sha256']['data.py'] = 'f'*64
        else:
            comp = deepcopy(completion); comp['stage_manifests']['inferred']['path'] = str(TMP / 'wrong_stage/manifest.json')
            modified['completion_receipt'], modified['completion_receipt_sha256'] = save(TMP / 'wrong_stage_completion.json', comp)
        badconf, badhash = save(TMP / (family + '_' + fault + '.json'), modified)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr('arggnn.boundary_dataset.load_boundary_dataset', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('Boundary loaded before admission')))
            mp.setattr('arggnn.age_dataset.load_age_caches', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('Age loaded before admission')))
            mp.setattr(runner, 'require_device', lambda *_a: events.append('device_query'))
            argv = ['--config', badconf, '--expected-config-sha256', badhash, '--output', str(TMP / ('never_written_' + family + fault)), '--device', 'cuda']
            if family == 'candidate':
                mp.setattr(sys, 'argv', ['runner'] + argv)
                call = runner.main
            else: call = lambda: runner.main(argv)
            try:
                call()
            except RuntimeError as error:
                rejection = {'type': type(error).__name__, 'message': str(error)}
            else: raise AssertionError('Bad completion admitted')
        record(family + '_startup_' + fault, passed=not events, device_queries=events, rejection=rejection)

# Canonical actual entry point must refuse invalid input before CUDA queries.
for device in ('cpu', 'cuda'):
    for fault in ('missing_manifest', 'heldout_data'):
        events = []
        data = TMP / ('absent_data' if fault == 'missing_manifest' else 'final_test_synthetic_unopened')
        with pytest.MonkeyPatch.context() as mp:
            def forbidden_query(*_a, **_k):
                events.append('cuda_query')
                raise AssertionError('CUDA query before canonical admission')
            mp.setattr(torch.cuda, 'is_available', forbidden_query)
            mp.setattr(torch.cuda, 'device_count', forbidden_query)
            mp.setattr(sys, 'argv', ['runner', '--config', str(canonical_config), '--data', str(data),
                                    '--output', str(TMP / 'unused_canonical_output'), '--device', device])
            try:
                canonical.main()
            except (FileNotFoundError, ValueError) as error:
                rejection = {'type': type(error).__name__, 'message': str(error)}
            else:
                raise AssertionError('Bad canonical data admitted')
        record('canonical_startup_' + device + '_' + fault, passed=not events, cuda_queries=events, rejection=rejection)

RESULTS['all_checks_passed'] = all(row.get('passed') for row in RESULTS['checks'])
RESULTS['loaded_source_sha256'] = {name: b.sha(ROOT / 'src/arggnn' / name) for name in ('development_training_bindings.py', 'train_anchor_development.py', 'train_anchor_temporal_development.py', 'train_canonical_gru.py')}
(OUT / 'replay.json').write_text(json.dumps(RESULTS, indent=2) + '\n')
assert RESULTS['all_checks_passed']

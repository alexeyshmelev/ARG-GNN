"""Pure evidence replay and prospective provenance binding, never test preparation."""
from __future__ import annotations
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import importlib.metadata
import importlib
import json
import os
import tempfile
from pathlib import Path
import sys

from . import development_selection as main
from . import development_selection_bindings as combined
from . import secondary_control_bindings as secondary
from . import confirmatory_metrics as metrics
from . import development_execution as numerical
from . import train_anchor_development as candidate_factory
from . import train_anchor_temporal_development as temporal_factory
from . import train_allrecord_development as allrecord_factory
from . import train_canonical_development as canonical_factory
from . import train_argmix_development as argmix_factory
from . import train_priorart_development as priorart_factory
from .resources import ROOT, ALLOWED_GPUS, PROCESS_TITLE

REF_FIELDS = ('protocol_json', 'protocol_markdown', 'scope_amendment', 'main_selection_plan',
              'main_selection_receipt', 'secondary_configuration', 'secondary_selection_receipt',
              'runtime_manifest', 'analysis_contract', 'parent_seed_registry', 'prior_parent_seed_registry',
              'workflow_validation', 'temperature_manifest', 'analysis_schema')
WORKFLOWS = ('custodian', 'prediction', 'joined_analysis')
SUPPORT = ('metrics', 'numerical_policy')
WORKFLOW_CHECKS = (
    'custodian_labels_sealed_from_modeling', 'prediction_without_gradients_or_optimization',
    'all_predictions_and_admission_sealed_before_label_join', 'joined_analysis_once',
    'explicit_parent_seed_alignment', 'unresolved_primary_failure_blocks_positive_confirmation',
)
PERTURBATIONS = ('reference_balanced', 'reference_imbalanced', 'reference_whg_absent',
                 'reference_age_sd25', 'reference_age_sd50', 'allele_error001', 'phase_switch1_per_mb',
                 'query_draw0', 'query_draw1', 'query_draw2', 'query_draw3')
_IMPORTED_PINS = {str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                  for module in (main, combined, secondary, metrics, numerical, candidate_factory, temporal_factory,
                                 allrecord_factory, canonical_factory, argmix_factory, priorart_factory)}
_IMPORTED_PINS[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCRIPT = ROOT/'scripts/bind_confirmatory_lock.py'
_IMPORTED_PINS[str(_SCRIPT)] = hashlib.sha256(_SCRIPT.read_bytes()).hexdigest()


def guard_lock_path(value, *, project_root=ROOT, must_exist=True):
    """Allow prospective metadata and source code, never a held-out data path."""
    root = Path(project_root).resolve(strict=True)
    if not isinstance(value, (str, Path)) or not str(value) or '\\' in str(value):
        raise ValueError('An in-project artifact path is required')
    lexical = Path(value)
    if '..' in lexical.parts:
        raise ValueError('Path traversal is forbidden')
    path = (lexical if lexical.is_absolute() else root/lexical).resolve(strict=False)
    if path == root or not path.is_relative_to(root):
        raise ValueError('Artifact escapes project root')
    relative = path.relative_to(root)
    forbidden = ('confirmatory', 'confirmation', 'final_test', 'final-test')
    # Filenames of known metadata/source artifacts may describe confirmation;
    # a held-out directory is never available, even when its file looks harmless.
    if any(part.lower() == 'test' or any(token in part.lower() for token in forbidden) for part in relative.parts[:-1]):
        raise ValueError('Held-out data paths are unavailable to the lock binder')
    name = relative.name.lower()
    if path.suffix != '.py' and any(token in name for token in forbidden):
        parent = relative.parent.as_posix()
        if parent not in ('research/protocols', 'research/registries', 'research/locks', 'configs', 'experiments/locks'):
            raise ValueError('Confirmation metadata must live in an explicit metadata location')
    if must_exist and not path.is_file():
        raise ValueError('Required lock artifact is not an existing file')
    return path


class _Capture(main._Capture):
    def path(self, value, *, must_exist=True):
        return guard_lock_path(value, project_root=self.root, must_exist=must_exist)


def _ref(capture, value):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('Exact existing path/SHA256 reference required')
    path = capture.path(value['path'])
    return {'path': str(path.relative_to(capture.root)), 'sha256': main._sha(value['sha256'])}


def _same_policy(value):
    expected = numerical.DETERMINISTIC_POLICY
    if not isinstance(value, dict) or set(value) != set(expected) or any(type(value[k]) is not type(v) or value[k] != v for k, v in expected.items()):
        raise ValueError('Exact typed strict numerical execution policy required')


def _source_stable():
    for name, expected in _IMPORTED_PINS.items():
        if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Imported lock/replay implementation drift: '+name)


def _actual_evidence_refs(capture, records):
    if not isinstance(records, list) or not records:
        raise ValueError('Captured run/checkpoint evidence provenance required')
    by_sha = {}
    for record in records:
        if not isinstance(record, dict) or not {'path', 'sha256', 'bytes'}.issubset(record):
            raise ValueError('Captured artifact provenance schema differs')
        ref = _ref(capture, {key: record[key] for key in ('path', 'sha256')})
        by_sha.setdefault(ref['sha256'], {})[ref['path']] = ref
    def one(digest):
        found = by_sha.get(main._sha(digest), {})
        if len(found) != 1:
            raise ValueError('Unique actual run/ledger artifact required')
        return next(iter(found.values()))
    return one


def replay_main(capture, plan_ref, receipt_ref):
    """Public selector replay after mandatory complete producer re-admission."""
    plan = capture.read(plan_ref); receipt = capture.read(receipt_ref)
    main.load_development_selection_plan(capture.root/plan_ref['path'], plan_ref['sha256'], project_root=capture.root)
    if set(plan['families']) != set(combined.FAMILIES) or set(plan['required_baseline_families']) != set(combined.FAMILIES)-{'candidate'}:
        raise ValueError('Complete prescribed ten-family main study required')
    if receipt.get('kind') != 'immutable_fresh_development_selection_receipt' or type(receipt.get('schema_version')) is not int or receipt['schema_version'] != 1 or receipt.get('plan_sha256') != plan_ref['sha256'] or receipt.get('matrix_resolved') is not True or receipt.get('baseline_selection_eligible') is not True or receipt.get('final_test_access') is not False or receipt.get('promotes_final_test_phase') is not False or receipt.get('required_baseline_families') != plan['required_baseline_families']:
        raise ValueError('Genuine resolved main selection receipt required')
    if not receipt.get('candidate_choice') or not receipt.get('strongest_required_baseline'):
        raise ValueError('Candidate and strongest eligible baseline required')
    one = _actual_evidence_refs(capture, receipt.get('source_artifacts'))
    cells = receipt.get('cells')
    if not isinstance(cells, list) or len(cells) != 240 or any(row.get('status') not in ('success', 'failed') for row in cells):
        raise ValueError('Every prescribed main cell must be resolved')
    runs = {}
    for row in cells:
        if row['status'] == 'success':
            key = (row['family_id'], row['run_receipt_sha256'])
            runs[key] = {'family_id': key[0], **one(key[1])}
    ledgers = {row['ledger_sha256']: one(row['ledger_sha256']) for row in receipt.get('failures_and_retry_ledger', [])}
    for row in cells:
        if row['status'] == 'failed':
            ledgers[row['ledger_sha256']] = one(row['ledger_sha256'])
    replay = main.synthesize_development_selection(capture.root/plan_ref['path'], plan_ref['sha256'],
        run_receipts=list(runs.values()), failure_ledgers=list(ledgers.values()), project_root=capture.root).payload
    left = {key: value for key, value in replay.items() if key != 'created_utc'}
    right = {key: value for key, value in receipt.items() if key != 'created_utc'}
    if json.dumps(left, sort_keys=True, allow_nan=False) != json.dumps(right, sort_keys=True, allow_nan=False):
        raise ValueError('Main selection receipt does not replay from actual captured evidence')
    for record in receipt['source_artifacts']:
        ref = {key: record[key] for key in ('path', 'sha256')}
        binary = Path(record['path']).suffix in ('.pt', '.py', '.sh')
        capture.read(ref, json_file=not binary, allow_source_reuse=binary)
    return plan, receipt


def replay_secondary(capture, config_ref, receipt_ref, main_plan_ref, main_receipt_ref, candidate_spec):
    """Re-run the strict secondary selector with recovered actual run/ledger refs."""
    settings = capture.read(config_ref)
    receipt = capture.read(receipt_ref)
    expected_controls = secondary.controls_for(candidate_spec)
    if receipt.get('kind') != 'secondary_development_attribution_selection_receipt' or type(receipt.get('schema_version')) is not int or receipt.get('schema_version') != 1:
        raise ValueError('Genuine completed secondary selection receipt required')
    if receipt.get('config_sha256') != config_ref['sha256'] or receipt.get('main_selection_plan') != main_plan_ref or receipt.get('main_selection_receipt') != main_receipt_ref:
        raise ValueError('Secondary main/config binding differs')
    if receipt.get('matrix_resolved') is not True or receipt.get('primary_selection_changed') is not False or receipt.get('final_test_access') is not False or receipt.get('authorizes_confirmatory_generation') is not False:
        raise ValueError('Incomplete or phase-changing secondary receipt')
    controls = receipt.get('controls')
    if not isinstance(controls, dict) or set(controls) != set(expected_controls) or any(row.get('matrix_resolved') is not True or row.get('choice') is None for row in controls.values()):
        raise ValueError('Every conditional secondary control requires a resolved eligible choice')
    if settings.get('controls') != list(expected_controls) or settings.get('main_selection_plan') != main_plan_ref or settings.get('main_selection_receipt') != main_receipt_ref:
        raise ValueError('Secondary configuration does not bind the selected main study')
    one = _actual_evidence_refs(capture, receipt.get('captured_artifacts'))
    cells = receipt.get('cells')
    if not isinstance(cells, list) or len(cells) != len(expected_controls)*2*3 or any(row.get('status') not in ('success', 'failed') for row in cells):
        raise ValueError('Complete two-LR/three-seed secondary matrix required')
    runs = {row['run_receipt_sha256']: one(row['run_receipt_sha256']) for row in cells if row['status'] == 'success'}
    ledgers = {row['ledger_sha256']: one(row['ledger_sha256']) for row in receipt.get('failures', [])}
    # The existing selector re-admits the factual config/source locks, CPU-loads
    # checkpoints, verifies factory state schemas and checks stored DEV NLL.
    replay = secondary.synthesize_secondary_selection(capture.root/config_ref['path'], config_ref['sha256'],
        run_receipts=list(runs.values()), failure_ledgers=list(ledgers.values()), project_root=capture.root)['values']
    if json.dumps(replay, sort_keys=True, allow_nan=False) != json.dumps(receipt, sort_keys=True, allow_nan=False):
        raise ValueError('Secondary receipt does not replay from actual captured evidence')
    for record in receipt['captured_artifacts']:
        ref = {key: record[key] for key in ('path', 'sha256')}
        binary = Path(record['path']).suffix in ('.pt', '.py', '.sh')
        capture.read(ref, json_file=not binary, allow_source_reuse=binary)
    return settings, receipt


def _primary_model(family, name, settings):
    """Existing pure constructors/factories only; never fit or load a cohort."""
    if family == 'candidate':
        return candidate_factory.make_anchor_model(name, 47, 4, settings)
    if family in ('gru', 'transformer', 'reference_smoother'):
        return temporal_factory.factory(name, settings)
    if family in ('allrecord_gru', 'allrecord_transformer'):
        return allrecord_factory.factory(name, settings)
    rows = settings.get('configurations')
    if rows is not None:
        selected = [row for row in rows if row['id'] == name]
        if len(selected) != 1:
            raise ValueError('Exactly one selected raw configuration required')
        configuration = selected[0]
    else:
        configuration = {'id': name, **settings['model_specs'][name], 'learning_rate': settings['model_learning_rates'][name]}
    if family == 'canonical_gru':
        return canonical_factory.factory(configuration, settings)
    if family == 'argmix_full_tree':
        return argmix_factory.factory(configuration, settings)
    if family in ('gst_ca_span', 'graphssm_zoh'):
        return priorart_factory.factory(configuration, settings)
    raise ValueError('Unknown prescribed primary family')


def validate_primary_bundle(capture, bundle, selection):
    """Validate primary model schema/capacity and actual replay attestations."""
    family = bundle['family_id']; name = bundle['configuration_id']
    settings = capture.read(bundle['protocol'])
    model = _primary_model(family, name, settings)
    expected = model.state_dict()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != bundle['choice']['parameters']:
        raise ValueError('Selected primary capacity differs from actual frozen factory')
    schema = {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in expected.items()}
    schema_sha = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    one = _actual_evidence_refs(capture, selection['source_artifacts'])
    for seed in bundle['seed_bundles']:
        run = capture.read(one(seed['run_receipt_sha256']))
        _same_policy(run.get(combined.POLICY_FIELDS[family]))
        body = run['models'][name]
        if seed['checkpoint'] is None:
            if parameters != 0 or family != 'reference_smoother':
                raise ValueError('Only the prescribed no-fit smoother may have no state checkpoint')
            seed['state_schema_verified'] = False
            seed['state_schema_reason'] = 'No trainable state in deterministic smoother'
            continue
        checkpoint = main._artifact(main._Capture(capture.root), seed['checkpoint']['path'], seed['checkpoint']['sha256'])
        state = checkpoint['model']
        if set(state) != set(expected) or any(state[key].shape != value.shape or state[key].dtype != value.dtype for key, value in expected.items()):
            raise ValueError('Exact selected primary checkpoint keys/shapes/dtypes required')
        for key, value in model.named_buffers():
            if key in state and not main.torch.equal(state[key], value):
                raise ValueError('Selected primary checkpoint changed a fixed factory buffer')
        model.load_state_dict(state, strict=True)
        original = body.get('original_selected_validation_nll', body.get('historical_validation_nll'))
        reloaded = body.get('reloaded_validation_nll')
        score = body.get('validation_nll')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (original, reloaded, score, body.get('checkpoint_replay_nll_delta'))):
            raise ValueError('Actual primary checkpoint replay evidence required')
        if original != reloaded or reloaded != score or body['checkpoint_replay_nll_delta'] != 0.:
            raise ValueError('Exact zero primary selected/reloaded raw-NLL delta required')
        seed['state_schema_verified'] = True
        seed['state_schema_sha256'] = schema_sha
        seed['original_selected_validation_nll'] = original
        seed['reloaded_validation_nll'] = reloaded
        seed['checkpoint_replay_nll_delta'] = 0.
    bundle['factory_parameters_verified'] = parameters


def _bundle(capture, choice, cells, config_ref, *, family=None, smoother=False):
    if not isinstance(choice, dict) or not choice.get('configuration_id'):
        raise ValueError('Eligible selected configuration required')
    chosen_family = choice['family_id'] if family is None else family
    rows = [row for row in cells if row.get('configuration_id') == choice['configuration_id'] and
            (family is not None or row.get('family_id') == chosen_family)]
    if len(rows) != 3 or {row.get('seed') for row in rows} != set(main.SEEDS) or any(row.get('status') != 'success' for row in rows):
        raise ValueError('Selected bundle must contain all three successful fixed seeds')
    seeds = []
    for row in sorted(rows, key=lambda value: value['seed']):
        checkpoint = None
        if smoother:
            if row.get('parameters') != 0 or row.get('checkpoint') is not None or row.get('checkpoint_sha256') is not None:
                raise ValueError('No-fit smoother cannot fabricate a checkpoint')
        else:
            checkpoint = _ref(capture, {'path': row.get('checkpoint'), 'sha256': row.get('checkpoint_sha256')})
            capture.read(checkpoint, json_file=False, allow_source_reuse=True)
        prediction = _ref(capture, {'path': row.get('prediction'), 'sha256': row.get('prediction_sha256')})
        capture.read(prediction, json_file=False, allow_source_reuse=True)
        seeds.append({'seed': row['seed'], 'checkpoint': checkpoint, 'development_prediction': prediction,
                      'run_receipt_sha256': row['run_receipt_sha256'], 'best_epoch': row['best_epoch'],
                      'validation_nll': row['validation_nll']})
    return {'family_id': chosen_family, 'configuration_id': choice['configuration_id'], 'protocol': config_ref,
            'choice': choice, 'seed_bundles': seeds, 'fitting': 'none; deterministic observed-reference predictor' if smoother else 'frozen development checkpoints; no refit'}


def _code(capture, workflows, support):
    if not isinstance(workflows, dict) or set(workflows) != set(WORKFLOWS) or not isinstance(support, dict) or set(support) != set(SUPPORT):
        raise ValueError('Custodian, prediction, joined-analysis and support code artifacts are mandatory')
    code = {}; forbidden_paths = set()
    for role, module in (('metrics', metrics), ('numerical_policy', numerical)):
        ref = _ref(capture, support[role]); content = capture.read(ref, json_file=False, allow_source_reuse=True)
        expected = _IMPORTED_PINS[str(Path(module.__file__).resolve())]
        if ref['sha256'] != expected:
            raise ValueError('Actual imported metric/numerical implementation must be pinned')
        forbidden_paths.add(str(capture.root/ref['path'])); code[role] = ref
    for role in WORKFLOWS:
        value = workflows[role]
        if not isinstance(value, dict) or set(value) != {'path', 'sha256', 'entry_point'}:
            raise ValueError('Workflow code requires exact path/hash/entry_point')
        ref = _ref(capture, {key: value[key] for key in ('path', 'sha256')})
        path = capture.root/ref['path']
        if path.suffix != '.py' or str(path) in forbidden_paths:
            raise ValueError('Metrics, numerical helpers or protocol prose are not workflow implementations')
        body = capture.read(ref, json_file=False, allow_source_reuse=True)
        if body is None:
            body = path.read_bytes()
        tree = ast.parse(body.decode('utf-8'), filename=ref['path'])
        entry = main._safe(value['entry_point'])
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry]
        if len(functions) != 1:
            raise ValueError('Existing top-level workflow entry point required')
        statements = [node for node in functions[0].body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
        if not statements or all(isinstance(node, (ast.Pass, ast.Raise)) or
            isinstance(node, ast.Return) and (node.value is None or isinstance(node.value, ast.Constant) and node.value.value is None) or
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis for node in statements):
            raise ValueError('Placeholder workflow cannot satisfy an executable lock')
        code[role] = ref | {'entry_point': entry}
    return code


def _positive_seed(value):
    if type(value) is not int or not 1 <= value <= 2**32-1:
        raise ValueError('Positive explicitly prespecified integer seed required')
    return value


def validate_registries(registry, prior, protocol, protocol_sha):
    if not isinstance(registry, dict) or set(registry) != {'schema_version', 'kind', 'status', 'protocol_sha256', 'training_seeds', 'analysis_seed', 'cohorts'} or type(registry['schema_version']) is not int or registry['schema_version'] != 1 or registry['kind'] != 'prespecified_parent_seed_registry' or registry['status'] != 'prespecified_before_test_preparation':
        raise ValueError('Actual prespecified parent/seed registry required')
    if registry['protocol_sha256'] != protocol_sha or registry['training_seeds'] != list(main.SEEDS) or registry['analysis_seed'] != 20261115:
        raise ValueError('Protocol/training/analysis registry binding differs')
    if not isinstance(prior, dict) or set(prior) != {'schema_version', 'kind', 'all_previous_and_development_parents_declared', 'final_test_parents', 'parents', 'source_artifacts'} or type(prior['schema_version']) is not int or prior['schema_version'] != 1 or prior['kind'] != 'prior_parent_seed_registry' or prior['all_previous_and_development_parents_declared'] is not True or type(prior['final_test_parents']) is not int or prior['final_test_parents'] != 0:
        raise ValueError('Declared prior/development seed registry with TEST0 required')
    previous = {purpose: set() for purpose in ('ancestry_seed', 'mutation_seed', 'inference_seed')}
    if not isinstance(prior['parents'], list) or not prior['parents']:
        raise ValueError('Actual prior parent seeds required')
    for row in prior['parents']:
        if not isinstance(row, dict) or set(row) != {'ancestry_seed', 'mutation_seed', 'inference_seed'}:
            raise ValueError('Prior parent seed row differs')
        for purpose, values in previous.items():
            value = row[purpose]
            if value is not None:
                values.add(_positive_seed(value))
            elif purpose != 'inference_seed':
                raise ValueError('Ancestry/mutation prior seeds cannot be absent')
    cohorts = registry['cohorts']
    if not isinstance(cohorts, dict) or set(cohorts) != {'id_confirmation', 'robustness_mother', 'demographic_ood'}:
        raise ValueError('ID, robustness mother and separate demographic registries required')
    known = protocol['separate_prespecified_analyses']['cohort']
    expected = {'id_confirmation': (192, 160, protocol['confirmation']), 'robustness_mother': (64, 64, known), 'demographic_ood': (64, 64, None)}
    namespaces = set(); used = {key: set(value) for key, value in previous.items()}
    for name, (count, target, specified) in expected.items():
        cohort = cohorts[name]
        if not isinstance(cohort, dict) or set(cohort) != {'namespace', 'ancestry_mutation_master_seed', 'relate_master_seed', 'target_admitted_parents', 'parents'}:
            raise ValueError('Prospective cohort registry schema differs')
        namespace = cohort['namespace']
        if not isinstance(namespace, str) or not namespace.strip() or namespace in namespaces:
            raise ValueError('Distinct explicit prospective namespaces required')
        namespaces.add(namespace)
        for key in ('ancestry_mutation_master_seed', 'relate_master_seed'):
            _positive_seed(cohort[key])
        if specified is not None and any(cohort[key] != specified[key] for key in ('namespace', 'ancestry_mutation_master_seed', 'relate_master_seed')):
            raise ValueError('Prespecified cohort namespace/master seeds differ')
        rows = cohort['parents']
        if type(cohort['target_admitted_parents']) is not int or cohort['target_admitted_parents'] != target or not isinstance(rows, list) or len(rows) != count or [row.get('parent_id') for row in rows] != list(range(count)):
            raise ValueError('Prespecified ordered parent attempt registry differs')
        for row in rows:
            if set(row) != {'parent_id', 'ancestry_seed', 'mutation_seed', 'inference_seed', 'perturbation_seeds'} or type(row['parent_id']) is not int:
                raise ValueError('Prospective parent seed row differs')
            for purpose, values in used.items():
                value = _positive_seed(row[purpose])
                if value in values:
                    raise ValueError('Prospective seed reuses a prior/development or another parent seed')
                values.add(value)
            perturbations = row['perturbation_seeds']
            expected_keys = set(PERTURBATIONS) if name == 'robustness_mother' else set()
            if not isinstance(perturbations, dict) or set(perturbations) != expected_keys:
                raise ValueError('All prespecified robustness/query-draw seeds required')
            for value in perturbations.values():
                _positive_seed(value)
    return {'cohort_attempts': {key: len(value['parents']) for key, value in cohorts.items()},
            'training_seeds': list(main.SEEDS), 'analysis_seed': 20261115,
            'prior_seed_coverage': 'Exactly the provided prior source registry; completeness is a reviewed declaration, not filesystem history discovery'}


def expected_analysis_schema(capture, schema):
    """Call only the known public metadata helper after its source is captured."""
    module = importlib.import_module('arggnn.confirmatory_analysis_runner')
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if schema['code_artifacts']['joined_analysis']['sha256'] != actual:
        raise ValueError('Actual joined-analysis metadata-helper source differs')
    return module.analysis_schema_values()


def validate_analysis_schema(capture, schema, code):
    if not isinstance(schema, dict) or type(schema.get('schema_version')) is not int or schema['schema_version'] != 1 or schema.get('kind') != 'prespecified_joined_analysis_artifact_schema' or schema.get('status') != 'frozen_before_test_preparation' or schema.get('analysis_phase') != 'id_primary_secondary_and_descriptive':
        raise ValueError('Actual pre-test joined-analysis artifact schema required')
    artifacts = schema.get('code_artifacts')
    if not isinstance(artifacts, dict) or set(artifacts) != {'joined_analysis', 'analysis_cli', 'metrics', 'dense_tract_metrics'}:
        raise ValueError('Joined-analysis, CLI, raw metric and dense-tract source pins required')
    for value in artifacts.values():
        ref = _ref(capture, value)
        if Path(ref['path']).suffix != '.py':
            raise ValueError('Actual analysis schema source files required')
        capture.read(ref, json_file=False, allow_source_reuse=True)
    if artifacts['metrics'] != code['metrics']:
        raise ValueError('Analysis schema metric implementation differs')
    joined_ref = {key: code['joined_analysis'][key] for key in ('path', 'sha256')}
    if joined_ref not in (artifacts['joined_analysis'], artifacts['analysis_cli']):
        raise ValueError('Actual joined-analysis workflow differs from its schema source pins')
    expected = expected_analysis_schema(capture, schema)
    if json.dumps(expected, sort_keys=True, allow_nan=False) != json.dumps(schema, sort_keys=True, allow_nan=False):
        raise ValueError('Joined-analysis schema does not match the actual public metadata helper')
    return {'implemented_phase': schema['analysis_phase'], 'remaining_coverage': schema.get('remaining_coverage')}


def validate_temperature_manifest(capture, manifest, primary, controls, selection, secondary_selection, refs, plan):
    required = {'schema_version', 'kind', 'status', 'main_selection_receipt_sha256', 'secondary_selection_receipt_sha256',
                'training_seeds', 'records', 'final_test_access'}
    if not isinstance(manifest, dict) or set(manifest) != required or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or manifest['kind'] != 'prespecified_validation_temperature_manifest' or manifest['status'] != 'frozen_before_test_preparation' or manifest['main_selection_receipt_sha256'] != refs['main_selection_receipt']['sha256'] or manifest['secondary_selection_receipt_sha256'] != refs['secondary_selection_receipt']['sha256'] or manifest['training_seeds'] != list(main.SEEDS) or manifest['final_test_access'] is not False:
        raise ValueError('Actual frozen pre-test VAL-temperature manifest required')
    bundles = dict(primary) | controls
    expected = {(method, seed['seed']): seed for method, bundle in bundles.items() for seed in bundle['seed_bundles']}
    records = manifest['records']
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError('Every selected method and fixed seed needs one actual VAL temperature')
    seen = set(); summary = []
    parent_ids = selection['ordered_parent_ids']
    for row in records:
        fields = {'method_id', 'seed', 'temperature', 'checkpoint', 'development_prediction', 'validation_parent_ids', 'fit_scope', 'fit_evidence'}
        if not isinstance(row, dict) or set(row) != fields or type(row['seed']) is not int:
            raise ValueError('Exact method/seed/temperature fitting record required')
        key = (row['method_id'], row['seed'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicated temperature method/seed')
        seen.add(key)
        temperature = row['temperature']
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Actual positive finite validation temperature required')
        bundle = expected[key]
        if row['checkpoint'] != bundle['checkpoint'] or row['development_prediction'] != bundle['development_prediction'] or row['validation_parent_ids'] != parent_ids or row['fit_scope'] != 'fresh_development_validation_only':
            raise ValueError('Temperature must fit only the exact frozen checkpoint/prediction and all sixteen VAL parents')
        fit_ref = _ref(capture, row['fit_evidence']); evidence = capture.read(fit_ref)
        required_fit = {'schema_version', 'kind', 'status', 'method_id', 'seed', 'temperature', 'checkpoint', 'development_prediction',
                        'main_selection_receipt_sha256', 'secondary_selection_receipt_sha256', 'source_binding', 'validation_parents',
                        'fit_scope', 'unchanged_model_weights', 'final_test_access', 'fit_code_artifacts', 'fit_record_artifacts'}
        expected_fit = {'schema_version': 1, 'kind': 'validation_temperature_fit_receipt', 'status': 'complete',
                        **{name: row[name] for name in ('method_id', 'seed', 'temperature', 'checkpoint', 'development_prediction', 'fit_scope')},
                        'main_selection_receipt_sha256': refs['main_selection_receipt']['sha256'],
                        'secondary_selection_receipt_sha256': refs['secondary_selection_receipt']['sha256'],
                        'source_binding': plan['source_binding'], 'validation_parents': plan['validation_parents'],
                        'unchanged_model_weights': True, 'final_test_access': False}
        if not isinstance(evidence, dict) or set(evidence) != required_fit or json.dumps({name: evidence[name] for name in expected_fit}, sort_keys=True, allow_nan=False) != json.dumps(expected_fit, sort_keys=True, allow_nan=False):
            raise ValueError('Actual VAL-only fitting/source/parent provenance and unchanged weights required')
        for name in ('fit_code_artifacts', 'fit_record_artifacts'):
            if not isinstance(evidence[name], list) or not evidence[name]:
                raise ValueError('Actual temperature fitting code and record artifacts required')
            for value in evidence[name]:
                ref = _ref(capture, value)
                if name == 'fit_code_artifacts' and Path(ref['path']).suffix != '.py':
                    raise ValueError('Temperature fitting implementation must be an actual Python source')
                capture.read(ref, json_file=Path(ref['path']).suffix == '.json', allow_source_reuse=True)
        summary.append({'method_id': row['method_id'], 'seed': row['seed'], 'temperature': temperature, 'fit_evidence': fit_ref})
    return sorted(summary, key=lambda value: (value['method_id'], value['seed']))


@dataclass(frozen=True)
class ConfirmatoryLock:
    payload: dict
    content: bytes
    sha256: str
    path: str | None


def bind_confirmatory_lock(request_path, expected_request_sha256, *, output_path=None, project_root=ROOT):
    """Replay complete development evidence and bind prospective artifacts only."""
    _source_stable()
    capture = _Capture(project_root)
    output = None if output_path is None else capture.path(output_path, must_exist=False)
    if output is not None and (output.exists() or not output.parent.is_dir()):
        raise ValueError('Fresh exclusive output with an existing in-project parent required')
    request_ref = _ref(capture, {'path': str(request_path), 'sha256': expected_request_sha256})
    request = capture.read(request_ref)
    fields = {'schema_version', 'kind', 'status', 'test_preparation_started', 'final_test_access', 'workflow_code', 'support_code', *REF_FIELDS}
    if not isinstance(request, dict) or set(request) != fields or type(request['schema_version']) is not int or request['schema_version'] != 1 or request['kind'] != 'confirmatory_lock_request' or request['status'] != 'reviewed_prespecification_before_test_preparation' or request['test_preparation_started'] is not False or request['final_test_access'] is not False:
        raise ValueError('Reviewed pre-test lock request required; proposal/hypothetical requests cannot emit a real lock')
    refs = {key: _ref(capture, request[key]) for key in REF_FIELDS}
    protocol = capture.read(refs['protocol_json'])
    markdown = capture.read(refs['protocol_markdown'], json_file=False, max_bytes=2_097_152, allow_source_reuse=True)
    if Path(refs['protocol_json']['path']).name != 'confirmatory_design_v1.json' or Path(refs['protocol_markdown']['path']).name != 'confirmatory_design_v1.md' or type(protocol.get('schema_version')) is not int or protocol.get('schema_version') != 1 or protocol.get('status') != 'reviewable_proposal_not_runner_configuration_not_executed' or not markdown:
        raise ValueError('Actual v1 proposal JSON/Markdown must remain explicitly labeled and bound')
    combined._amendment(capture.read(refs['scope_amendment']))
    captured_plan = capture.read(refs['main_selection_plan'])
    if not isinstance(captured_plan, dict) or set(captured_plan.get('families', {})) != set(combined.FAMILIES):
        raise ValueError('Complete prescribed ten-family producer plan required')
    completion_ref = captured_plan['source_artifacts']['completion']
    family_refs = [{'family_id': family, **spec['protocol']} for family, spec in captured_plan['families'].items()]
    rebound = combined.bind_development_selection(capture.root/completion_ref['path'], completion_ref['sha256'],
        scope_amendment_path=capture.root/refs['scope_amendment']['path'],
        expected_scope_amendment_sha256=refs['scope_amendment']['sha256'],
        family_configs=family_refs, output_directory=None, project_root=capture.root)
    if rebound.plan != captured_plan:
        raise ValueError('Main plan differs from complete factual producer-contract admission')
    # This strict existing verifier reconstructs all 240 cells from captured
    # runs/failure ledgers and CPU-loads real weights/logit artifacts.
    plan, selection = replay_main(capture, refs['main_selection_plan'], refs['main_selection_receipt'])
    candidate = selection['candidate_choice']; baseline = selection['strongest_required_baseline']
    candidate_config_ref = plan['families']['candidate']['protocol']
    candidate_config = capture.read(candidate_config_ref)
    candidate_spec = candidate_config['model_specs'][candidate['configuration_id']]
    controls_settings, controls = replay_secondary(capture, refs['secondary_configuration'], refs['secondary_selection_receipt'],
        refs['main_selection_plan'], refs['main_selection_receipt'], candidate_spec)
    if controls.get('source_binding') != plan['source_binding']:
        raise ValueError('Main and secondary completed source bindings differ')
    code = _code(capture, request['workflow_code'], request['support_code'])
    validation = capture.read(refs['workflow_validation'])
    if not isinstance(validation, dict) or set(validation) != {'schema_version', 'kind', 'status', 'final_test_access', 'synthetic_development_validation_only', 'code_artifacts', 'checks', 'evidence_artifacts'} or type(validation['schema_version']) is not int or validation['schema_version'] != 1 or validation['kind'] != 'confirmatory_workflow_validation_receipt' or validation['status'] != 'passed' or validation['final_test_access'] is not False or validation['synthetic_development_validation_only'] is not True or validation['code_artifacts'] != code or not isinstance(validation['checks'], dict) or set(validation['checks']) != set(WORKFLOW_CHECKS) or any(value is not True for value in validation['checks'].values()):
        raise ValueError('Actual pre-test workflow validation evidence must bind every executable/support source')
    if not isinstance(validation['evidence_artifacts'], list) or not validation['evidence_artifacts']:
        raise ValueError('Workflow validation requires actual captured evidence artifacts')
    for value in validation['evidence_artifacts']:
        ref = _ref(capture, value); capture.read(ref, json_file=False, allow_source_reuse=True)
    runtime = capture.read(refs['runtime_manifest'])
    _same_policy(runtime.get('deterministic_execution_policy'))
    if runtime.get('device_kind') != 'cuda' or type(runtime.get('physical_gpu')) is not int or runtime['physical_gpu'] not in ALLOWED_GPUS or runtime.get('process_title') != PROCESS_TITLE or not isinstance(runtime.get('device'), dict) or not isinstance(runtime.get('software'), dict):
        raise ValueError('Actual CUDA runtime/device/title evidence required; CPU smoke is insufficient')
    if not isinstance(runtime.get('visible_gpu_uuid'), str) or not runtime['visible_gpu_uuid'].startswith('GPU-') or runtime['software'].get('torch') != importlib.metadata.version('torch') or runtime['software'].get('python_executable') != sys.executable:
        raise ValueError('Captured runtime UUID/software must bind the current environment')
    for key in ('python', 'torch', 'torch_cuda', 'torch_build'):
        if not isinstance(runtime['software'].get(key), str) or not runtime['software'][key].strip():
            raise ValueError('Actual numerical software/build versions required')
    analysis = capture.read(refs['analysis_contract'])
    expected_analysis = {'schema_version': 1, 'kind': 'prespecified_confirmatory_analysis_contract',
        'status': 'prespecified_before_test_preparation', 'protocol_sha256': refs['protocol_json']['sha256'],
        'training_seeds': list(main.SEEDS), 'nll_policy': main.RAW_NLL,
        'confirmation': protocol['confirmation'], 'separate_prespecified_analyses': protocol['separate_prespecified_analyses'],
        'code_artifacts': code, 'runtime_manifest': refs['runtime_manifest'], 'final_test_access': False}
    if json.dumps(analysis, sort_keys=True, allow_nan=False) != json.dumps(expected_analysis, sort_keys=True, allow_nan=False):
        raise ValueError('Metric/analysis/multiplicity/admission/failure prespecification differs from bound protocol/code/runtime')
    analysis_schema = capture.read(refs['analysis_schema'])
    analysis_coverage = validate_analysis_schema(capture, analysis_schema, code)
    registry = capture.read(refs['parent_seed_registry']); prior = capture.read(refs['prior_parent_seed_registry'])
    registry_summary = validate_registries(registry, prior, protocol, refs['protocol_json']['sha256'])
    if not isinstance(prior['source_artifacts'], list) or not prior['source_artifacts']:
        raise ValueError('Prior seed declaration requires real source manifests')
    if plan['source_artifacts']['source_manifest'] not in prior['source_artifacts']:
        raise ValueError('Prior registry must bind the actual fresh development source manifest')
    declared = {(row['ancestry_seed'], row['mutation_seed'], row['inference_seed']) for row in prior['parents']}
    observed = set()
    for value in prior['source_artifacts']:
        source = capture.read(_ref(capture, value))
        for row in source.get('replicates', []):
            if row.get('split') not in ('train', 'val'):
                raise ValueError('Prior metadata must exclude any held-out parent')
            seeds = row.get('seeds', {})
            observed.add((_positive_seed(seeds.get('ancestry')), _positive_seed(seeds.get('mutations')),
                          None if row.get('inference_seed') is None else _positive_seed(row['inference_seed'])))
    if not observed or declared != observed:
        raise ValueError('Prior registry differs from captured source-manifest parent seeds')
    candidate_bundle = _bundle(capture, candidate, selection['cells'], candidate_config_ref)
    baseline_bundle = _bundle(capture, baseline, selection['cells'], plan['families'][baseline['family_id']]['protocol'],
                              smoother=baseline['family_id'] == 'reference_smoother')
    validate_primary_bundle(capture, candidate_bundle, selection)
    validate_primary_bundle(capture, baseline_bundle, selection)
    secondary_bundles = {control: _bundle(capture, row['choice'], controls['cells'], refs['secondary_configuration'], family=control)
                         for control, row in controls['controls'].items()}
    temperatures = validate_temperature_manifest(capture, capture.read(refs['temperature_manifest']),
        {'candidate': candidate_bundle, 'strongest_required_baseline': baseline_bundle}, secondary_bundles,
        selection, controls, refs, plan)
    capture.verify_stable(); _source_stable()
    payload = {'schema_version': 1, 'kind': 'confirmatory_provenance_lock',
        'status': 'complete_id_development_evidence_and_prospective_workflows_bound',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'project_root': str(capture.root), 'request': request_ref,
        'bound_references': refs, 'candidate': candidate_bundle, 'strongest_required_baseline': baseline_bundle,
        'secondary_controls': secondary_bundles, 'training_seeds': list(main.SEEDS), 'source_binding': plan['source_binding'],
        'main_matrix_counts': selection['matrix_counts'], 'main_failures_and_retries': selection['failures_and_retry_ledger'],
        'secondary_failures': controls['failures'], 'prespecified_registry': registry_summary,
        'analysis_coverage': analysis_coverage, 'lock_execution_scope': 'id_confirmation_only',
        'validation_temperatures': temperatures, 'code_artifacts': code, 'numerical_execution_policy': numerical.DETERMINISTIC_POLICY,
        'captured_runtime': runtime, 'analysis_environment_versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch')},
        'captured_artifacts': capture.provenance(), 'binding_implementation_sha256': {str(Path(name).relative_to(ROOT)): digest for name, digest in _IMPORTED_PINS.items()},
        'lock_implementation_sha256': _IMPORTED_PINS[str(Path(__file__).resolve())],
        'test_preparation_performed': False, 'final_test_access': False, 'generation_performed': False,
        'operational_permissions_enforced': False,
        'scope': 'Hash binding of completed development evidence and explicit prospective implementations/registries. No held-out preparation, model evaluation, joined analysis, scientific conclusion or permission enforcement performed.'}
    content = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if output is not None:
        pending = None; identity = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=output.parent, prefix='.lock_pending_', delete=False) as stream:
                pending = Path(stream.name); current = os.fstat(stream.fileno())
                identity = (current.st_dev, current.st_ino)
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            capture.verify_stable(); _source_stable()
            current = pending.lstat()
            if (current.st_dev, current.st_ino) != identity:
                raise ValueError('Owned lock output inode changed')
            # Exclusive hard-link publication never replaces an existing path.
            os.link(pending, output, follow_symlinks=False)
        finally:
            if pending is not None and identity is not None:
                try:
                    current = pending.lstat()
                except FileNotFoundError:
                    pass
                else:
                    if (current.st_dev, current.st_ino) == identity:
                        pending.unlink()
    return ConfirmatoryLock(payload, content, hashlib.sha256(content).hexdigest(), None if output is None else str(output))


@dataclass(frozen=True)
class ValidatedConfirmatoryLock:
    payload: dict
    content: bytes
    sha256: str
    path: str
    capture: object

    def check(self, full=True):
        encoded = (json.dumps(self.payload, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        if encoded != self.content:
            raise ValueError('In-memory published lock payload drift')
        _source_stable(); self.capture.verify_stable()
        if full:
            for path, value in self.capture.files.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != value['sha256']:
                    raise ValueError('Published lock captured artifact drift: '+path)
        return self.sha256


def validate_confirmatory_lock(lock_path, expected_sha256, *, project_root=ROOT):
    """Verify a reviewed published ID lock before roster/ledger/target reads.

    This checks the captured provenance, not another development search or any
    held-out artifact. Later analysis owns its sealed roster/ledger admission.
    """
    _source_stable(); capture = _Capture(project_root)
    ref = _ref(capture, {'path': str(lock_path), 'sha256': expected_sha256})
    payload = capture.read(ref)
    if not isinstance(payload, dict) or type(payload.get('schema_version')) is not int or payload['schema_version'] != 1 or payload.get('kind') != 'confirmatory_provenance_lock' or payload.get('status') != 'complete_id_development_evidence_and_prospective_workflows_bound' or payload.get('lock_execution_scope') != 'id_confirmation_only' or payload.get('project_root') != str(capture.root) or payload.get('training_seeds') != list(main.SEEDS) or any(payload.get(name) is not False for name in ('test_preparation_performed', 'final_test_access', 'generation_performed', 'operational_permissions_enforced')):
        raise ValueError('Actual published complete ID provenance lock required')
    expected_versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch')}
    if payload.get('analysis_environment_versions') != expected_versions:
        raise ValueError('Published lock analysis environment versions differ')
    software = payload.get('captured_runtime', {}).get('software', {})
    if software.get('python_executable') != sys.executable or software.get('torch') != expected_versions['torch']:
        raise ValueError('Published lock captured Python/Torch runtime differs')
    expected_sources = {str(Path(name).relative_to(ROOT)): digest for name, digest in _IMPORTED_PINS.items()}
    if payload.get('binding_implementation_sha256') != expected_sources:
        raise ValueError('Published lock binding implementation differs')
    if not isinstance(payload.get('bound_references'), dict) or set(payload['bound_references']) != set(REF_FIELDS):
        raise ValueError('Published lock lacks required prospective references')
    artifacts = payload.get('captured_artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('Published lock lacks actual captured provenance')
    for record in artifacts:
        if not isinstance(record, dict) or set(record) != {'path', 'sha256', 'bytes'} or type(record['bytes']) is not int or record['bytes'] < 0:
            raise ValueError('Published provenance record differs')
        captured_ref = _ref(capture, {key: record[key] for key in ('path', 'sha256')})
        body = capture.read(captured_ref, json_file=False, allow_source_reuse=True)
        if body is not None and len(body) != record['bytes']:
            raise ValueError('Published artifact byte count differs')
    captured = {(value['path'], value['sha256']) for value in capture.provenance()}
    for value in payload['bound_references'].values():
        normalized = _ref(capture, value)
        if (normalized['path'], normalized['sha256']) not in captured:
            raise ValueError('Published reference lacks captured bytes')
    _same_policy(payload.get('numerical_execution_policy'))
    content = (capture.root/ref['path']).read_bytes()
    result = ValidatedConfirmatoryLock(payload, content, ref['sha256'], str(capture.root/ref['path']), capture)
    result.check()
    return result

"""Independent, CPU-only final audit; held-out truth requires completed score waits.

Exit 3 means pending and guarantees no truth opened or audit product written.
This module imports no training or production scoring implementation.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import zipfile
import numpy as np

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/joint_evaluation_20261029'
REVIEW = BASE / 'review_finish'
AMERICAN = ROOT / 'experiments/results/joint_american_20261029'
ARGMIX = ROOT / 'experiments/results/joint_argmix_20261029'
NATIVE = ROOT / 'experiments/paper_argmix_1611_rebuild_20261024_v1'
IDENTITY = ('record_ids', 'sample_nodes', 'sample_indices', 'positions_bp0', 'parent_id', 'class_order')
C3 = ('EUR', 'EAS', 'AFR')
C4 = ('Ana', 'CHG', 'WHG', 'EHG')
FREEZES = {
    'american': ('american/freeze_v2.json', 'a203156e0de57b9a536d176542b84c1fea934f70f8200cb3a57a35129af060c7'),
    'argmix': ('argmix/freeze_v1.json', 'c3303268253987f305d6cb298c625ccae9a669d4785c5181f5d8f38922c66634'),
}

class Pending(Exception):
    pass


def need(ok, message):
    if not ok:
        raise ValueError(message)


def same_ref(a, b):
    return Path(a['path']).name == Path(b['path']).name and a['sha256'] == b['sha256'] and (ROOT / a['path']).resolve() == (ROOT / b['path']).resolve()


def wait_time(row, phase):
    key = phase + '_unix'
    if key in row:
        return float(row[key])
    return datetime.fromisoformat(row[phase + '_utc']).timestamp()


def near(a, b, label):
    need(math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-13), 'Metric mismatch: ' + label)


def summary(values):
    values = [float(x) for x in values]
    need(len(values) >= 2 and all(math.isfinite(x) for x in values), 'Finite independent-parent summary')
    mean = math.fsum(values) / len(values)
    return dict(mean=mean, sample_SD=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1)), parents=len(values))


def probabilities(values, n, k):
    a = np.asarray(values)
    need(a.dtype == np.float32 and a.shape == (n, k) and np.isfinite(a).all() and
         ((a >= 0) & (a <= 1)).all() and np.allclose(a.sum(1), 1., atol=1e-6, rtol=1e-5), 'Complete raw normalized probabilities')
    return a


def metrics(values, truth, classes):
    """Compute from raw arrays, without importing either production metric helper."""
    y = np.asarray(truth)
    k = len(classes)
    need(y.dtype.kind in 'iu' and y.ndim == 1 and len(y) and ((y >= 0) & (y < k)).all(), 'Valid direct ancestry labels')
    p = probabilities(values, len(y), k).astype(np.float64)
    calls = p.argmax(1)
    matrix = np.bincount(k*y + calls, minlength=k*k).reshape(k, k)
    per_class = {}
    for i, name in enumerate(classes):
        correct, actual, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        per_class[name] = dict(precision=correct/predicted if predicted else 0., recall=correct/actual if actual else 0.,
            F1=2*correct/(actual+predicted) if actual+predicted else 0., actual_count=actual, predicted_count=predicted)
    chosen = p[np.arange(len(y)), y]
    zeros = int(np.count_nonzero(chosen == 0))
    return dict(accuracy=float(np.trace(matrix)/len(y)), macro_F1=math.fsum(v['F1'] for v in per_class.values())/k,
        confusion=matrix.tolist(), per_class=per_class, Brier=float(np.mean(np.sum((p-np.eye(k)[y])**2, axis=1))),
        NLL=None if zeros else float(np.mean(-np.log(chosen))), NLL_status='positive_infinity' if zeros else 'finite',
        zero_truth_probability_count=zeros, records=len(y))


def check_metrics(actual, expected, label):
    need(actual['confusion'] == expected['confusion'], 'Confusion mismatch: ' + label)
    for key in ('accuracy', 'macro_F1', 'Brier'):
        near(actual[key], expected[key], label + ':' + key)
    need(actual['NLL_status'] == expected['NLL_status'], 'NLL infinity semantics: ' + label)
    if expected['NLL'] is None:
        need(actual['NLL'] is None, 'Infinite NLL must remain null with explicit infinity status')
    else:
        near(actual['NLL'], expected['NLL'], label + ':NLL')
    for name, values in expected['per_class'].items():
        for key, value in values.items():
            near(actual['per_class'][name][key], value, label + ':' + name + ':' + key)


def check_summary(actual, values, label, count_key='parents', mean_key='mean', sd_key='sample_SD'):
    calculated = summary(values)
    near(actual[mean_key], calculated['mean'], label + ':mean')
    near(actual[sd_key], calculated['sample_SD'], label + ':SD')
    if count_key in actual:
        need(actual[count_key] == calculated['parents'], 'Independent-parent count: ' + label)
    return calculated


def canonical(arrays, parent, classes, n):
    need(set(IDENTITY) <= set(arrays) and arrays['parent_id'].shape == () and arrays['parent_id'].item() == parent,
         'Canonical same-parent identity')
    need(arrays['class_order'].tolist() == list(classes), 'Class order must be exact')
    for key in IDENTITY[:3]:
        need(arrays[key].dtype == np.int64 and arrays[key].shape == (n,), 'Canonical integer identity: ' + key)
    need(len(np.unique(arrays['record_ids'])) == n, 'Canonical record IDs must be unique')
    need(arrays['positions_bp0'].dtype == np.float64 and arrays['positions_bp0'].shape == (n,) and
         np.isfinite(arrays['positions_bp0']).all(), 'Exact canonical coordinates')


def prediction_identity(prediction, registry, classes):
    need(set(prediction) == set(IDENTITY) | {'probabilities'}, 'Prediction must contain only canonical keys and probabilities')
    for key in IDENTITY:
        need(np.array_equal(prediction[key], registry[key]), 'Prediction identity mismatch: ' + key)
    return probabilities(prediction['probabilities'], len(registry['record_ids']), len(classes))


def american_truth(registry, truth):
    """Independent left-bound search and explicit raw-sample lookup."""
    need(set(truth) == {'intervals_bp', 'query_sample_indices', 'source_classes_0based', 'parent_id', 'class_order'}, 'Exact American truth schema')
    need(np.array_equal(truth['parent_id'], registry['parent_id']) and truth['class_order'].tolist() == list(C3), 'Same-parent C3 truth')
    intervals, samples, labels = truth['intervals_bp'], truth['query_sample_indices'], truth['source_classes_0based']
    need(intervals.dtype == np.float64 and intervals.ndim == 2 and intervals.shape[1] == 2 and len(intervals) and
         intervals[0, 0] == 0 and intervals[-1, 1] == 30000000 and np.isfinite(intervals).all() and
         (intervals[:, 1] > intervals[:, 0]).all() and np.array_equal(intervals[:-1, 1], intervals[1:, 0]), 'Complete half-open 30 Mb truth coverage')
    need(samples.dtype == np.int64 and np.array_equal(samples, np.arange(600, 1600)) and
         labels.dtype == np.uint8 and labels.shape == (len(intervals), len(samples)), 'Exact raw query-sample truth columns')
    need((labels < 3).all(), 'Direct C3 ancestry labels')
    lookup = {int(sample): i for i, sample in enumerate(samples)}
    need(all(int(sample) in lookup for sample in registry['sample_indices']), 'No reference or absent query sample')
    positions = registry['positions_bp0']
    index = np.searchsorted(intervals[:, 0], positions, side='right') - 1
    need((index >= 0).all() and (positions < intervals[index, 1]).all(), 'No extrapolation across ancestry intervals')
    columns = np.array([lookup[int(x)] for x in registry['sample_indices']])
    return labels[index, columns].astype(np.int64)


def census_truth(sequence, manifest, rows, classes=C4):
    """Follow original sample ancestry to time350 on exact physical target positions."""
    import msprime
    times, flags, populations = sequence.nodes_time, sequence.nodes_flags, sequence.nodes_population
    time = manifest['census_time']
    need(time == 350, 'Registered census350 ancestry definition')
    raw_samples = manifest['sample_index_to_raw_node_id']
    need(len(raw_samples) == 3222 and len(set(raw_samples)) == 3222, 'Complete original sample mapping')
    answer = np.empty(len(rows), dtype=np.int64)
    tree = sequence.first()
    cache, previous_tree = {}, None
    for i in sorted(range(len(rows)), key=lambda i: float(rows[i]['rebased_midpoint_bp'])):
        position = float(rows[i]['rebased_midpoint_bp'])
        need(0 <= position < sequence.sequence_length, 'No physical-coordinate extrapolation')
        tree.seek(position)
        if tree.index != previous_tree:
            cache, previous_tree = {}, tree.index
        sample = int(rows[i]['sample_index'])
        if sample not in cache:
            node = int(raw_samples[sample])
            while node >= 0 and times[node] < time:
                node = tree.parent(node)
            need(node >= 0 and times[node] == time and flags[node] & msprime.NODE_IS_CEN_EVENT, 'Exact census ancestor required')
            name = manifest['population_id_to_name'][str(int(populations[node]))]
            need(name in classes, 'Census ancestry must be one original direct founder class')
            cache[sample] = classes.index(name)
        answer[i] = cache[sample]
    return answer


class Audit:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.verified = {}
        self.truth_authorized = False
        self.truth_reads = []
        self.wait_refs = []
        self.retired_checkpoint_evidence = []

    def path(self, value):
        value = value['path'] if isinstance(value, dict) else value
        path = (self.root / value).resolve()
        need(path.is_relative_to(self.root), 'All audit inputs stay inside the project')
        return path

    def ref(self, value):
        path = self.path(value)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*2**20), b''):
                digest.update(block)
        return dict(path=str(path.relative_to(self.root)), sha256=digest.hexdigest())

    def verify(self, reference):
        path = self.path(reference)
        key = str(path)
        if key not in self.verified:
            self.verified[key] = self.ref(path)['sha256']
        need(self.verified[key] == reference['sha256'], 'Input checksum changed: ' + key)
        if 'bytes' in reference:
            need(path.stat().st_size == reference['bytes'], 'Input byte count changed: ' + key)
        return path

    def json(self, reference):
        path = self.verify(reference) if isinstance(reference, dict) else self.path(reference)
        need(path.stat().st_size <= 32*2**20, 'Bounded audit JSON input')
        return json.loads(path.read_text())

    def arrays(self, reference, *, truth=False):
        if truth:
            need(self.truth_authorized, 'Held-out truth gate is closed')
            self.truth_reads.append(reference)
        path = self.verify(reference)
        if path.suffix == '.npz':
            with zipfile.ZipFile(path) as archive:
                need(sum(item.file_size for item in archive.infolist()) <= (8*2**30 if truth else 64*2**20), 'Bounded uncompressed arrays')
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key].copy() for key in archive.files}
        need(path.stat().st_size <= 64*2**20, 'Bounded prediction array')
        return np.load(path, allow_pickle=False)

    def wait(self, filename, product=None):
        path = self.path(filename)
        if not path.exists():
            raise Pending('Missing completed wait: ' + str(path.relative_to(self.root)))
        row = self.json(path)
        need(row.get('actual_wait') is True and row.get('actual_returncode') == 0 and row.get('owned_process_group_absent') is True and
             row.get('status', 'succeeded') == 'succeeded' and row.get('products_complete', True) is True, 'Failed or incomplete genuine wait: ' + str(path))
        need(isinstance(row.get('pid'), int) and isinstance(row.get('process_start_ticks'), (str, int)), 'Wait must retain process identity')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                continue
            need(int(fields[3]) != row['pid'] or fields[0] == 'Z', 'Waited worker still has a live process session')
        if product is not None:
            need(row.get('result_ref') is not None and self.path(row['result_ref']) == self.path(product), 'Scoring wait binds wrong product')
            self.verify(row['result_ref'])
        self.wait_refs.append(self.ref(path))
        return row

    def stage_wait(self, filename, spec, plan_ref):
        row = self.wait(filename)
        need(same_ref(row['plan_ref'], plan_ref) and row['command'] == spec['command'], 'Exact registered waited command')
        need(row['products'] == [self.ref(p) for p in spec['products']], 'Exact stage products after real wait')
        return row

    def freeze(self, benchmark):
        relative, digest = FREEZES[benchmark]
        reference = dict(path='research/joint_evaluation_20261029/' + relative, sha256=digest)
        seal = self.json(reference)
        for source in seal['source_refs']:
            self.verify(source)
        protocol = self.json(seal['protocol_ref'])
        return reference, seal, protocol

    def gates(self, benchmark):
        if benchmark == 'american':
            return [self.wait(BASE/'root/finish_records_v1/american_comparison/actual_wait.json', AMERICAN/'comparison_v1/product.json')]
        return [self.wait(ARGMIX/'scores/actual_wait.json', ARGMIX/'scores/completed.json'),
                self.wait(NATIVE/'argmix_scores/actual_wait.json', NATIVE/'argmix_scores/completed.json'),
                self.wait(ARGMIX/'validation_score/actual_wait.json', ARGMIX/'validation_score/completed.json')]

    def checkpoint(self, reference, *, allow_retired=False):
        if self.path(reference).exists():
            self.verify(reference)
        else:
            need(allow_retired, 'Required final model checkpoint is missing')
            self.retired_checkpoint_evidence.append(reference)

    def memory(self):
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 32*2**30, 'Audit exceeded32GiB process RAM')

    def evidence(self):
        return dict(actual_wait_refs=self.wait_refs, verified_file_count=len(self.verified), truth_reads=self.truth_reads,
                    historical_native_checkpoint_bytes_not_available=self.retired_checkpoint_evidence,
                    cpu_only=True, production_scoring_functions_imported=False, model_or_source_changes=False,
                    training_or_GPU_launches=False, heldout_truth_opened_only_after_completed_score_waits=True)


def audit_american(a):
    gate = a.gates('american')[0]
    result = a.json(gate['result_ref'])
    need(result['schema'] == 'american_joint_native_methods_comparison_v1' and result['complete'] and
         result['class_order'] == list(C3) and result['methods'] == ['TreeLoom-Joint', 'ARGformer', 'FLARE'] and
         result['ARGformer_evaluator'] == 'reference_PCA10_nearest_class_centroid' and
         result['simulation_or_targets_modified'] is False and result['probabilities_modified'] is False, 'Exact completed American comparison')
    freeze_ref, freeze, protocol = a.freeze('american')
    plan_ref = a.ref(BASE/'root/american_continue_plan_v1.json')
    need(plan_ref['sha256'] == '2c36a4dcdaa909c5dc1f15341f67c747dacf697d9a824069969c3c077881f60b', 'Reviewed original continuation plan')
    plan = a.json(plan_ref)
    need(same_ref(plan['freeze_ref'], freeze_ref), 'Final American source freeze')
    parents = {parent: row for parent, row in protocol['parents'].items() if row['split'] != 'train'}
    train_parents = {parent for parent, row in protocol['parents'].items() if row['split'] == 'train'}
    need(Counter(row['split'] for row in parents.values()) == {'validation': 5, 'test': 10} and len(train_parents) == 20 and
         set(result['parents']) == set(parents), 'Five validation and ten TEST parents, kept separate')
    final = a.json(BASE/'root/american_continue_run_v1/scores_completed.json')
    need(same_ref(final['plan_ref'], plan_ref), 'Completed score continuation identity')
    records = BASE/'root/american_continue_run_v1'
    waited = {spec['name']: a.stage_wait(records/spec['name']/'actual_wait.json', spec, plan_ref) for spec in plan['stages']}
    need(same_ref(final['score_actual_wait_ref'], a.ref(records/'score/actual_wait.json')), 'Original own scoring wait')
    need(gate['started_unix'] > waited['score']['finished_unix'], 'Paired comparison follows own completed score')
    own_ref = result['original_new_model_score_ref']
    need(a.path(own_ref) == AMERICAN/'scores_v2/product.json', 'Original own score path')
    own = a.json(own_ref)
    need(own['all15_predictions_sealed_before_truth'] and own['complete'] and not own['probabilities_modified'] and set(own['parents']) == set(parents), 'Original complete15-parent score seal')
    native_ref = result['original_native_barrier_ref']
    need(same_ref(native_ref, final['native_prediction_barrier_ref']), 'Original native barrier identity')
    native = a.json(native_ref)
    need(native['truth_opened'] is False and same_ref(native['plan_ref'], plan_ref), 'Native all-parent seal preceded truth')
    native_by_parent = {row['parent_id']: row for row in native['parents']}
    need(len(native['parents']) == 15 and set(native_by_parent) == set(parents), 'All native parents once')
    seal = a.json(result['prediction_seal_ref'])
    need(seal['ancestry_truth_opened'] is False and same_ref(seal['freeze_ref'], freeze_ref) and set(seal['parents']) == set(parents), 'All45 predictions sealed before original comparison truth')
    fit = a.json(AMERICAN/'fit_v2/product.json')
    need(fit['complete'] and same_ref(fit['freeze_ref'], freeze_ref) and fit['epochs'] == 13 and fit['target_presentations'] == 2600000 and
         fit['heldout_observations_or_truth_opened'] is False, 'Fixed epoch13 fit; held-out isolation')
    a.checkpoint(fit['checkpoint_ref'])
    training = a.json(fit['training_plan_ref'])
    need(set(training['parents']) == train_parents and training['targets'] == 200000 and training['epochs'] == 13 and
         training['planned_updates'] == fit['updates'] and training['canonical_target_presentations'] == 2600000, 'Full TRAIN-only fixed training plan')
    total_windows = 0
    total_updates_per_epoch = 0
    for parent in sorted(train_parents):
        row = training['parents'][parent]
        expected_ref = protocol['parents'][parent]['binding']['focal_registry_ref']
        need(same_ref(row['focal_registry_ref'], expected_ref), 'Training plan binds original canonical parent')
        original = a.arrays(expected_ref); canonical(original, parent, C3, 10000)
        windows = len(np.unique(original['positions_bp0']))
        need(row['targets'] == 10000 and row['windows'] == windows, 'All original American training windows and targets')
        total_windows += windows; total_updates_per_epoch += (windows+15)//16
    need(training['windows'] == total_windows == 158476 and fit['updates'] == 13*total_updates_per_epoch == 128869,
         'Full158476-window13epoch schedule, including every parent tail')
    sealed = {}
    methods = result['methods']
    for parent, definition in parents.items():
        focal_ref = definition['binding']['focal_registry_ref']
        registry = a.arrays(focal_ref)
        canonical(registry, parent, C3, 10000)
        binding = plan['native_score_gate'][parent]
        focal = a.json(binding['focal_preparation_ref'])
        need(focal['parent_id'] == parent and same_ref(focal['focals_ref'], focal_ref) and
             same_ref(focal['observation_ref'], definition['binding']['observation_ref']) and focal['query_ancestry_truth_opened'] is False,
             'Exact original observation-only focal preparation')
        joint_ref = a.ref(AMERICAN/'predictions_v2'/parent/'product.json')
        joint = a.json(joint_ref)
        need(joint['complete'] and joint['rows'] == 10000 and joint['parent_id'] == parent and joint['split'] == definition['split'] and
             joint['class_order'] == list(C3) and joint['model_id'] == protocol['model_id'] and joint['query_ancestry_truth_opened'] is False and
             same_ref(joint['freeze_ref'], freeze_ref) and same_ref(joint['checkpoint_ref'], fit['checkpoint_ref']) and same_ref(joint['focal_registry_ref'], focal_ref), 'Joint canonical fixed-final prediction')
        need(same_ref(own['parents'][parent]['prediction_product_ref'], joint_ref), 'Own score binds same prediction')
        pred_wait = waited['predict_' + parent]
        need(pred_wait['started_unix'] > waited['fit']['finished_unix'] and pred_wait['finished_unix'] < waited['score']['started_unix'], 'Joint prediction ordering')
        original_wait = a.wait(binding['argformer_wait_path'])
        wrapper = a.json(original_wait['result_ref'])
        arg_ref = a.ref(binding['argformer_completion_path'])
        need(wrapper['scientific_success'] and same_ref(wrapper['value']['completion_ref'], arg_ref), 'Native ARGformer successful actual producer wait')
        arg = a.json(arg_ref)
        need(arg['schema'] == 'paper_american_native_completion_v1' and arg['stage'] == 'predict' and arg['variant'] == 'paper_aligned' and
             arg['parent_id'] == parent and arg['class_order'] == list(C3) and arg['records'] == 10000 and same_ref(arg['focal_registry_ref'], focal_ref) and
             arg['evaluator'] == 'reference_PCA10_nearest_class_centroid' and arg['pca_fit'] == 'all_reference_paths_only' and
             arg['probability_semantics'] == 'hard_call_one_hot_transport' and arg['query_truth_used_for_fit'] is False and arg['query_ancestry_truth_opened'] is False,
             'Disclosed reference-only PCA10 centroid evaluator')
        prep = a.json(arg['preparation_ref'])
        need(prep['parent_id'] == parent and same_ref(prep['observation_ref'], focal['observation_ref']) and
             same_ref(prep['focals_ref'], focal_ref) and prep['query_ancestry_truth_opened'] is False, 'ARGformer exact public inputs')
        executed = a.json(arg['self_spec_ref'])
        contrastive = a.json(executed['checkpoint_completion_ref'])
        need(contrastive['stage'] == 'contrastive' and contrastive['variant'] == 'paper_aligned' and contrastive['parent_id'] == parent and
             contrastive['full_budget_completed'] is True and contrastive['query_ancestry_truth_opened'] is False and
             same_ref(contrastive['preparation_ref'], arg['preparation_ref']), 'Same-parent completed native checkpoint producer')
        a.checkpoint(contrastive['checkpoint_ref'], allow_retired=True)
        a.verify(arg['reference_pca_ref'])
        flare = a.json(binding['flare_ref'])
        need(flare['status'] == 'sealed_C3_native_FLARE_calls_and_raw_posteriors' and flare['parent_id'] == parent and flare['records'] == 10000 and
             flare['class_order'] == list(C3) and same_ref(flare['focal_registry_ref'], focal_ref) and flare['query_ancestry_truth_opened'] is False and
             flare['common_focal_registry_unchanged'] and flare['native_AN1_AN2_calls_used_unchanged'] and flare['one_hot_is_native_call_transport'] and
             flare['rounded_ANP_renormalized'] is False and flare['raw_rounded_ANP_modified'] is False, 'Native FLARE hard calls preserved')
        converted = a.json(flare['conversion_ref'])
        need(converted['parent_id'] == parent and same_ref(converted['observation_ref'], focal['observation_ref']) and
             same_ref(converted['focal_registry_ref'], focal_ref) and converted['query_ancestry_truth_opened'] is False, 'FLARE shared observed inputs')
        refs = dict(zip(methods, (joint['prediction_ref'], arg['prediction_ref'], flare['prediction_ref'])))
        for method in methods:
            need(same_ref(refs[method], seal['parents'][parent][method]) and same_ref(refs[method], result['parents'][parent]['prediction_refs'][method]), 'All45 sealed references unchanged')
        entry = native_by_parent[parent]
        need(same_ref(entry['argformer'], arg_ref) and same_ref(entry['flare'], binding['flare_ref']) and
             same_ref(entry['prediction_refs'][0], refs['ARGformer']) and same_ref(entry['prediction_refs'][1], refs['FLARE']), 'Native all-parent barrier binding')
        predictions = {name: prediction_identity(a.arrays(ref), registry, C3) for name, ref in refs.items()}
        for name in ('ARGformer', 'FLARE'):
            need(((predictions[name] == 0) | (predictions[name] == 1)).all(), 'Native hard-call transport is one-hot')
        logits = a.arrays(joint['logits_ref'])
        need(np.array_equal(logits['record_ids'], registry['record_ids']), 'Joint logits canonical order')
        ex = np.exp(logits['logits'].astype(np.float64)-logits['logits'].max(1, keepdims=True))
        need(np.array_equal(predictions['TreeLoom-Joint'], (ex/ex.sum(1, keepdims=True)).astype(np.float32)), 'Joint raw softmax unchanged')
        need(result['parents'][parent]['split'] == definition['split'] and result['parents'][parent]['targets'] == 10000, 'Reported split and target count')
        sealed[parent] = (registry, predictions)
    a.truth_authorized = True
    bindings = a.json(protocol['postseal_scoring_bindings_ref'])['parents']
    need(set(bindings) == set(parents), 'Exact original postseal truth bindings')
    recalculated = {}
    for parent, (registry, predictions) in sealed.items():
        binding = bindings[parent]
        completion = a.json(binding['private_completion_path'])
        generation = a.json(binding['generation_ready_ref'])
        need(completion['parent_id'] == parent and completion['class_order'] == list(C3) and
             completion['status'] == 'completed_C3_simulation_reconstruction' and same_ref(completion['observation_ref'], generation['observation_ref']), 'Original same-parent private generator')
        truth_ref = completion['private_truth_ref']
        need(a.path(truth_ref).is_relative_to(a.path(binding['private_completion_path']).parent/'truth') and
             same_ref(truth_ref, result['parents'][parent]['truth_ref']) and same_ref(truth_ref, own['parents'][parent]['truth_ref']), 'Exact original private truth path')
        truth_arrays = a.arrays(truth_ref, truth=True)
        y = american_truth(registry, truth_arrays)
        del truth_arrays
        values = {name: metrics(p, y, C3) for name, p in predictions.items()}
        for name in methods:
            check_metrics(result['parents'][parent]['methods'][name], values[name], parent + ':' + name)
        check_metrics(own['parents'][parent]['result'], values['TreeLoom-Joint'], parent + ':original own score')
        recalculated[parent] = dict(split=parents[parent]['split'], methods=values, labels_sha256=hashlib.sha256(y.tobytes()).hexdigest())
        del y
        gc.collect(); a.memory()
        print(json.dumps(dict(checked_parent=parent, targets=10000, methods=3)), flush=True)
    summaries, paired = {}, {}
    for split in ('validation', 'test'):
        selected = [row for row in recalculated.values() if row['split'] == split]
        summaries[split], paired[split] = {}, {}
        for name in methods:
            summaries[split][name] = {}
            for key in ('accuracy', 'macro_F1', 'Brier'):
                values = [row['methods'][name][key] for row in selected]
                summaries[split][name][key] = check_summary(result['summary'][split][name][key], values, split + ':' + name + ':' + key)
                if name == 'TreeLoom-Joint':
                    check_summary(own['summary'][split][key], values, 'original own:' + split + ':' + key)
            for cls in C3:
                for key in ('precision', 'recall', 'F1'):
                    check_summary(result['summary'][split][name]['per_class'][cls][key], [row['methods'][name]['per_class'][cls][key] for row in selected], split + ':' + name + ':' + cls + ':' + key)
        for other in methods[1:]:
            paired[split][other] = {key: check_summary(result['paired_new_minus_competitor'][split][other][key],
                [row['methods']['TreeLoom-Joint'][key]-row['methods'][other][key] for row in selected], split + ':paired:' + other + ':' + key)
                for key in ('accuracy', 'macro_F1', 'Brier')}
    return dict(benchmark='american', comparison_ref=gate['result_ref'], freeze_ref=freeze_ref, prediction_products=45,
        total_heldout_targets=150000, TEST_targets=100000, validation_targets=50000, per_parent=recalculated,
        summaries=summaries, paired_new_minus_competitor=paired, caveats=[
        'Five validation parents remain separate from ten TEST parents; validation is not a substituted TEST result.',
        'ARGformer uses reference-only PCA10 nearest-class centroids from author embeddings, not native top-one retrieval.',
        'ARGformer and FLARE are evaluated as unchanged one-hot native hard-call transports. Their NLL can be positive infinity; this is not a posterior-calibration comparison.',
        'Native retired checkpoints are identified through retained authenticated producer metadata when their bytes were intentionally reclaimed; missing bytes are listed explicitly.',
        'This is a shared simulated benchmark using documented source and data reconstruction choices, not reproduction of a published numerical table.',
        'Sample SD quantifies independent-parent variation, not training-seed uncertainty or a confidence interval.'])


def argmix_support(a, reference, recipe_ref, parent, split, recipe):
    support = a.json(reference)
    need(support['schema'] == 'ARGMix_prospective1611_shared_support_v1' and support['parent_id'] == parent and support['split'] == split and
         same_ref(support['config_ref'], recipe_ref) and support['truth_or_labels_read'] is False and support['examples'] == 100000,
         'Exact original truth-free ARGMix support')
    with a.verify(support['targets_ref']).open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    need(len(rows) == 100000 and all(int(row['target_row']) == i for i, row in enumerate(rows)), 'Every original target in original order')
    need(len({(int(row['inferred_tree_index']), int(row['sample_index'])) for row in rows}) == 100000, 'Unique sampled tree/query keys')
    bridge = a.json(support['inferred_bridge_ref'])
    need(bridge['parent_id'] == parent and same_ref(bridge['config_ref'], recipe_ref), 'Original inferred bridge identity')
    roster = a.json(bridge['roster_ref'])
    expected = []
    for cohort in recipe['roster']['cohorts']:
        for individual in range(cohort['diploids']):
            for haplotype in (0, 1):
                expected.append((cohort['id']+'_'+format(individual, '04d'), haplotype, cohort['id'], cohort['query']))
    need(len(roster) == len(expected) == 3222, 'Complete original1611-diploid roster')
    for index, record in enumerate(roster):
        need((record['sample_id'], record['haplotype'], record['observed_cohort'], record['query_eligible']) == expected[index] and
             record['sample_index'] == index, 'Original semantic sample identity')
    for row in rows:
        sample = roster[int(row['sample_index'])]
        need(sample['query_eligible'] and (row['sample_id'], int(row['haplotype']), row['cohort']) ==
             (sample['sample_id'], sample['haplotype'], sample['observed_cohort']), 'Original query/cohort join')
        need(float(row['rebased_midpoint_bp']) == float(row['local_midpoint_bp']) + bridge['local_to_rebased_offset_bp'], 'Original rebased midpoint, without coordinate shifts')
    need(Counter(row['cohort'] for row in rows) == support['cohort_counts'], 'Exact original cohort counts')
    return rows, roster, bridge


NATIVE_REGISTERED_PLAN_REF = {
    'path': 'research/joint_evaluation_20261029/queue/native_argmix_cuda_retry_20261030_v1/plan_v1.json',
    'sha256': 'd6ffc2b11a4b56e820648dd769a4590e8553109377b31f6644253dcd6adb2fab',
}


def native_condition_disclosure(model, wait, registered_condition):
    labels = {
        'equation_corrected': 'ARGMix, registered equation-corrected variant (author-derived Transformer)',
        'author_key_i': 'ARGMix, author key-i implementation',
    }
    condition = model.get('condition')
    need(condition in labels and condition == registered_condition, 'Selected native condition must match the registered execution plan')
    command = wait['command']
    need(command.count('--condition') == 1 and command[command.index('--condition') + 1] == condition,
         'Selected native condition must match the genuinely waited fit command')
    return dict(condition=condition, label=labels[condition], registered_condition=registered_condition,
                published_numerical_table_reproduction=False,
                interpretation='Registered reconstructed shared-data benchmark; other numerical variants require separately labelled results.')


def audit_argmix(a):
    gates = a.gates('argmix')
    own = a.json(gates[0]['result_ref'])
    other = a.json(gates[1]['result_ref'])
    validation = a.json(gates[2]['result_ref'])
    freeze_ref, freeze, protocol = a.freeze('argmix')
    recipe_ref = protocol['recipe_ref']; recipe = a.json(recipe_ref)
    registered_plan = a.json(NATIVE_REGISTERED_PLAN_REF)
    registered_fit = next(task for task in registered_plan['tasks'] if task['id'] == 'paper_argmix_argmix.train')
    registered_command = registered_fit['argv']
    need(registered_command.count('--condition') == 1, 'One explicit registered native condition')
    registered_condition = registered_command[registered_command.index('--condition') + 1]
    expected = {parent: row['split'] for parent, row in protocol['parents'].items()}
    need(Counter(expected.values()) == {'train': 5, 'val': 1, 'iid': 3, 'misspec': 5}, 'Exact original14-parent recipe')
    test_parents = {parent for parent, split in expected.items() if split in ('iid', 'misspec')}
    val_parent = next(parent for parent, split in expected.items() if split == 'val')
    need(same_ref(own['paired_comparator_ref'], gates[1]['result_ref']), 'Exact genuinely waited comparator score')
    by_method = {}
    for name, score in [('TreeLoom-Joint', own), ('ARGMix', other)]:
        need(score['schema'] == 'paper_argmix_complete_scores_v1' and same_ref(score['config_ref'], recipe_ref), 'Original native scoring contract')
        rows = {row['parent_id']: row for row in score['results']}
        need(len(score['results']) == 8 and set(rows) == test_parents, 'All original8TEST parents')
        by_method[name] = rows
    fit_ref = a.ref(ARGMIX/'fit/completed.json'); fit = a.json(fit_ref)
    fit_wait = a.wait(ARGMIX/'fit/actual_wait.json', ARGMIX/'fit/completed.json')
    need(fit['complete'] and fit['model_id'] == protocol['model_id'] and same_ref(fit['freeze_ref'], freeze_ref) and fit['epochs'] == 5 and
         fit['target_presentations'] == 2500000 and fit['heldout_observations_or_truth_opened'] is False, 'Fixed epoch5 joint fit before held-out inputs')
    a.checkpoint(fit['checkpoint_ref'])
    training = a.json(fit['training_plan_ref'])
    need(set(training['parents']) == {parent for parent, split in expected.items() if split == 'train'} and training['epochs'] == 5 and
         training['targets'] == 500000 and training['planned_updates'] == fit['updates'] and training['canonical_target_presentations'] == 2500000,
         'Complete original five TRAIN parents and fixed exposure')
    total_windows = 0
    total_updates_per_epoch = 0
    for parent, row in training['parents'].items():
        observation = a.json(NATIVE/parent/'treeloom_observation/completed.json')
        need(same_ref(row['focal_registry_ref'], observation['registry_ref']), 'ARGMix TRAIN canonical registry binding')
        original = a.arrays(observation['registry_ref']); canonical(original, parent, C4, 100000)
        windows = len(np.unique(original['positions_bp0']))
        need(row['targets'] == 100000 and row['windows'] == windows, 'All original ARGMix training windows and targets')
        total_windows += windows; total_updates_per_epoch += (windows+15)//16
    need(training['windows'] == total_windows and fit['updates'] == 5*total_updates_per_epoch,
         'Full original ARGMix grouped-parent/tail epoch schedule')
    barriers = {}
    for name, score in [('TreeLoom-Joint', own), ('ARGMix', other)]:
        barrier = a.json(score['prediction_barrier_ref'])
        a.wait(a.path(score['prediction_barrier_ref']).parent/'actual_wait.json', a.path(score['prediction_barrier_ref']))
        need(barrier['complete'] and barrier['truth_read'] is False and same_ref(barrier['config_ref'], recipe_ref), 'Native prediction barrier identity')
        mapping = {row['parent_id']: row['prediction_ref'] for row in barrier['predictions']}
        need(len(barrier['predictions']) == 8 and set(mapping) == test_parents, 'All8 predictions before native truth')
        barriers[name] = mapping
    paired_path = NATIVE/'paired_prediction_barrier/barrier.json'
    paired = a.json(paired_path)
    a.wait(paired_path.parent/'actual_wait.json', paired_path)
    need(paired['complete'] and paired['truth_read'] is False and paired['both_methods_all_eight_predictions_complete'] and
         set(paired['parent_ids']) == test_parents and same_ref(paired['config_ref'], recipe_ref), 'Original both-method barrier')
    need({paired['first_barrier_ref']['sha256'], paired['second_barrier_ref']['sha256']} ==
         {own['prediction_barrier_ref']['sha256'], other['prediction_barrier_ref']['sha256']}, 'Exact two native barriers')
    native_model_ref = None; native_model = None
    sealed = {}
    for parent in sorted(test_parents | {val_parent}):
        split = expected[parent]
        joint_ref = a.ref(ARGMIX/'predictions'/parent/'completed.json')
        product = a.json(joint_ref)
        joint_wait = a.wait(a.path(joint_ref).parent/'actual_wait.json', a.path(joint_ref))
        need(joint_wait['task_id'] == 'joint_argmix.' + parent + '.prediction', 'Exact registered new-model prediction task')
        need(wait_time(joint_wait, 'started') > wait_time(fit_wait, 'finished') and
             wait_time(joint_wait, 'finished') < wait_time(gates[0], 'started'), 'All nine joint predictions follow fit and precede scoring')
        need(product['complete'] and product['schema'] == 'paper_argmix_predictions_v1' and product['model_id'] == protocol['model_id'] and
             product['parent_id'] == parent and product['split'] == split and product['examples'] == 100000 and product['class_order'] == list(C4) and
             product['truth_read'] is False and same_ref(product['config_ref'], recipe_ref) and same_ref(product['freeze_ref'], freeze_ref) and
             same_ref(product['checkpoint_ref'], fit['checkpoint_ref']), 'Complete original joint-model prediction identity')
        observation = a.json(product['observation_ref'])
        need(observation['parent_id'] == parent and observation['split'] == split and observation['private_truth_opened'] is False and
             observation['all_native_trees_nodes_sites_samples_retained'] and same_ref(observation['support_ref'], product['support_ref']) and
             same_ref(observation['registry_ref'], product['registry_ref']), 'Original full observation seal')
        rows, roster, bridge = argmix_support(a, product['support_ref'], recipe_ref, parent, split, recipe)
        registry = a.arrays(product['registry_ref']); canonical(registry, parent, C4, 100000)
        need(np.array_equal(registry['record_ids'], np.arange(100000)) and
             np.array_equal(registry['positions_bp0'], [float(row['rebased_midpoint_bp']) for row in rows]) and
             np.array_equal(registry['sample_indices'], [int(row['sample_index']) for row in rows]) and
             np.array_equal(registry['sample_nodes'], [roster[int(row['sample_index'])]['inferred_tree_node_id'] for row in rows]), 'All canonical row keys and physical coordinates match original support')
        probs = probabilities(a.arrays(product['probabilities_ref']), 100000, 4)
        logits = a.arrays(product['logits_ref'])
        need(logits.shape == (100000, 4) and np.isfinite(logits).all(), 'All direct joint logits')
        ex = np.exp(logits.astype(np.float64)-logits.max(1, keepdims=True))
        need(np.array_equal(probs, (ex/ex.sum(1, keepdims=True)).astype(np.float32)), 'No probability collapse or postprocessing')
        predictions = {'TreeLoom-Joint': probs}
        if parent in test_parents:
            need(same_ref(joint_ref, barriers['TreeLoom-Joint'][parent]) and same_ref(joint_ref, by_method['TreeLoom-Joint'][parent]['prediction_ref']), 'Joint score and barrier predictions agree')
            native_ref = barriers['ARGMix'][parent]
            native = a.json(native_ref)
            a.wait(a.path(native_ref).parent/'actual_wait.json', a.path(native_ref))
            need(native['schema'] == 'paper_argmix_predictions_v1' and native['parent_id'] == parent and native['split'] == split and
                 native['examples'] == 100000 and native['class_order'] == list(C4) and native['truth_read'] is False and
                 same_ref(native['config_ref'], recipe_ref) and same_ref(native['support_ref'], product['support_ref']) and
                 same_ref(native_ref, by_method['ARGMix'][parent]['prediction_ref']), 'Paired native prediction same canonical support')
            feature = a.json(native['feature_ref'])
            need(feature['parent_id'] == parent and feature['split'] == split and feature['truth_read'] is False and
                 same_ref(feature['support_ref'], product['support_ref']) and same_ref(feature['inferred_bridge_ref'], a.json(product['support_ref'])['inferred_bridge_ref']), 'Native feature provenance unchanged')
            if native_model_ref is None:
                native_model_ref = native['selected_model_ref']; native_model = a.json(native_model_ref)
                native_fit_wait = a.wait(a.path(native_model_ref).parent/'actual_wait.json', a.path(native_model_ref))
                native_disclosure = native_condition_disclosure(native_model, native_fit_wait, registered_condition)
                need(native_model['schema'] == 'paper_argmix_selected_model_v1' and same_ref(native_model['config_ref'], recipe_ref) and
                     native_model['selection'] == 'earliest highest validation example accuracy' and native_model['test_or_ood_read'] is False and
                     native_model['training_examples'] == 500000 and native_model['validation_examples'] == 100000, 'Original validation-selected native checkpoint, no TEST selection')
                a.checkpoint(native_model['checkpoint_ref'])
                a.verify(native_model['source_ref'])
                fit_inputs = a.json(native_model['fit_manifest_ref'])
                native_train = [a.json(row['features']) for row in fit_inputs['train']]
                native_val = a.json(fit_inputs['validation']['features'])
                need(len(native_train) == 5 and {row['parent_id'] for row in native_train} ==
                     {parent for parent, split in expected.items() if split == 'train'} and all(row['split'] == 'train' for row in native_train) and
                     native_val['parent_id'] == val_parent and native_val['split'] == 'val', 'Native fit used TRAIN and separate original VAL only')
                progress_path = a.path(native_model_ref).parent/'progress.jsonl'
                progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
                need(len(progress) == native_model['epochs_completed'] and [row['epoch'] for row in progress] == list(range(len(progress))), 'Complete native epoch history')
                best = max(range(len(progress)), key=lambda i: progress[i]['validation']['accuracy'])
                need(best == native_model['selected_epoch'], 'Earliest maximum validation checkpoint preserved')
                near(progress[best]['validation']['accuracy'], native_model['validation_accuracy'], 'native selected validation accuracy')
            else:
                need(same_ref(native_model_ref, native['selected_model_ref']), 'One unchanged selected native model across allTEST parents')
            predictions['ARGMix'] = probabilities(a.arrays(native['probabilities_ref']), 100000, 4)
        else:
            need(validation['schema'] == 'argmix_joint_fixed_final_validation_v1' and validation['model_id'] == protocol['model_id'] and
                 validation['selection_used'] is False and same_ref(validation['freeze_ref'], freeze_ref),
                 'Postfit validation diagnostic never selects joint checkpoint')
            need(same_ref(validation['checkpoint_ref'], fit['checkpoint_ref']) and same_ref(validation['support_ref'], product['support_ref']), 'Validation fixed checkpoint and support')
        sealed[parent] = (product, rows, predictions)
    a.truth_authorized = True
    recalculated = {}
    for parent, (product, rows, predictions) in sealed.items():
        label_path = NATIVE/parent/'labels/completed.json'
        a.wait(label_path.parent/'actual_wait.json', label_path)
        label_ref = a.ref(label_path); label = a.json(label_ref)
        need(label['schema'] == 'ARGMix_prospective1611_targets_with_external_truth_v1' and label['parent_id'] == parent and label['split'] == expected[parent] and
             label['examples'] == 100000 and label['class_order'] == list(C4) and same_ref(label['config_ref'], recipe_ref) and
             same_ref(label['support_manifest_ref'], product['support_ref']) and label['features_exported'] is False, 'Original label producer on unchanged support')
        a.truth_reads += [label['truth_manifest_ref'], label['labels_ref']]
        truth_manifest = a.json(label['truth_manifest_ref'])
        need(truth_manifest['parent_id'] == parent and truth_manifest['split'] == expected[parent] and same_ref(truth_manifest['config_ref'], recipe_ref) and
             truth_manifest['class_order'] == list(C4), 'Original same-parent census truth')
        import tskit
        truth_ref = truth_manifest['ancestry_file']; a.truth_reads.append(truth_ref)
        tree_sequence = tskit.load(a.verify(truth_ref))
        need(tree_sequence.sequence_length == recipe['map']['length_bp'], 'Original full physical truth support')
        y = census_truth(tree_sequence, truth_manifest, rows)
        del tree_sequence
        stored = np.fromfile(a.verify(label['labels_ref']), dtype=np.uint8)
        need(np.array_equal(y, stored), 'Independently rebuilt census labels differ from original labels')
        cohorts = np.array([row['cohort'] for row in rows])
        values = {}
        for method, p in predictions.items():
            metrics_row = metrics(p, y, C4)
            cohort_rows = {name: metrics(p[cohorts == name], y[cohorts == name], C4) for name in sorted(set(cohorts))}
            values[method] = dict(overall=metrics_row, cohorts=cohort_rows)
            actual = by_method[method][parent] if parent in test_parents else validation
            need(actual['examples'] == 100000 and same_ref(actual['support_ref'], product['support_ref']), 'Reported original target support')
            if parent in test_parents:
                need(actual['split'] == expected[parent] and same_ref(actual['label_ref'], label_ref), 'Reported native split and original labels')
            near(actual['accuracy'], metrics_row['accuracy'], parent + ':' + method)
            need(set(actual['per_cohort']) == set(cohort_rows), 'All original cohorts represented')
            for cohort, calculated in cohort_rows.items():
                need(actual['per_cohort'][cohort]['examples'] == calculated['records'], 'Per-cohort exact row count')
                near(actual['per_cohort'][cohort]['accuracy'], calculated['accuracy'], parent + ':' + method + ':' + cohort)
        recalculated[parent] = dict(split=expected[parent], methods=values, label_ref=label_ref, labels_sha256=hashlib.sha256(y.tobytes()).hexdigest())
        del y, stored
        gc.collect(); a.memory()
        print(json.dumps(dict(checked_parent=parent, targets=100000, methods=len(predictions), census_labels_rebuilt=True)), flush=True)
    summaries = {}
    for method, score in [('TreeLoom-Joint', own), ('ARGMix', other)]:
        summaries[method] = {}
        for split in ('iid', 'misspec'):
            values = [row['methods'][method]['overall']['accuracy'] for row in recalculated.values() if row['split'] == split]
            summaries[method][split] = check_summary(score['summaries'][split], values, method + ':' + split, mean_key='mean_accuracy', sd_key='sample_sd')
    paired = {row['parent_id']: row for row in own['paired_differences']}
    need(len(own['paired_differences']) == 8 and set(paired) == test_parents, 'All paired TEST differences')
    for parent in test_parents:
        row = recalculated[parent]
        need(paired[parent]['split'] == expected[parent], 'Paired split preserved')
        near(paired[parent]['accuracy_difference'], row['methods']['TreeLoom-Joint']['overall']['accuracy']-row['methods']['ARGMix']['overall']['accuracy'], parent + ':paired')
    return dict(benchmark='argmix', comparison_ref=gates[0]['result_ref'], comparator_ref=gates[1]['result_ref'],
        validation_ref=gates[2]['result_ref'], freeze_ref=freeze_ref, prediction_products=17, TEST_targets=800000,
        validation_targets=100000, per_parent=recalculated, summaries=summaries,
        native_selected_model_ref=native_model_ref, native_checkpoint_selection=native_model['selection'],
        native_condition_disclosure=native_disclosure, native_registered_plan_ref=NATIVE_REGISTERED_PLAN_REF,
        diagnostic_scope='Confusion, F1, Brier and NLL are new independent audit diagnostics; production ARGMix score publishes accuracy only.',
        caveats=['The native comparator is ' + native_disclosure['label'] + '. Published numerical-table reproduction is not established.',
          'Three IID and five misspecification TEST parents are separate; one validation parent is a postfit joint diagnostic.',
          'Native ARGMix retains its original earliest-highest-validation checkpoint policy; the new joint model uses fixed epoch5. Neither selects on TEST.',
          'Original 100000 targets per parent retain exact sampler order, sample identity, inferred-tree midpoint and direct census350 founder labels.',
          'The1611-diploid cohorts, dates and synthetic older-NF proxy are disclosed reconstruction choices, not recovered paper participants or an identified Iceman.',
          'Model inputs differ: full native graph/genotype context for the joint model and original compact ancestry tokens/distances for ARGMix.',
          'Sample SD measures simulated-parent variation, not training-seed uncertainty or a confidence interval.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('american', 'argmix', 'all'), default='all')
    parser.add_argument('--check-ready', action='store_true')
    parser.add_argument('--output', help='New final_results_*.json file directly inside review_finish; never overwrite')
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    selected = ['american', 'argmix'] if args.benchmark == 'all' else [args.benchmark]
    audit = Audit()
    try:
        for benchmark in selected:
            audit.gates(benchmark)
        if args.check_ready:
            print(json.dumps(dict(ready=True, benchmarks=selected, truth_opened=False, outputs_written=False)))
            return 0
        output = Path(args.output) if args.output else REVIEW/('final_results_' + args.benchmark + '_completed_v1.json')
        output = audit.path(output)
        need(output.parent == REVIEW and output.name.startswith('final_results_') and output.suffix == '.json' and not output.exists(), 'New review-owned final_results output only')
        reports = {name: (audit_american if name == 'american' else audit_argmix)(audit) for name in selected}
        receipt = dict(schema='independent_final_benchmark_audit_v1', passed=True, reviewer='benchmark_audit_oct29',
            recorded_utc=datetime.now(timezone.utc).isoformat(), source_ref=audit.ref(__file__), reports=reports, **audit.evidence())
        payload = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        need(len(payload) <= 4*2**20, 'Small bounded final audit receipt')
        with output.open('xb') as stream:
            stream.write(payload)
        print(json.dumps(dict(passed=True, output_ref=audit.ref(output), truth_payloads_opened=len(audit.truth_reads))), flush=True)
        return 0
    except Pending as error:
        need(not audit.truth_reads, 'A pending gate must never follow a private truth read')
        print(json.dumps(dict(ready=False, pending=str(error), truth_opened=False, outputs_written=False)), flush=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())

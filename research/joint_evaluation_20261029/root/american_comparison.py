"""Compare all three methods on the exact American canonical target registries.

All 45 prediction products are authenticated before opening ancestry truth.
No probability transformation, sample filtering, or checkpoint selection occurs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from arggnn.benchmarking.paper_campaign import runtime as r
from arggnn.american_c3_evaluate_v1 import metrics, verify_prediction
from research.joint_evaluation_20261029.american.common_v2 import load_npz
from research.joint_evaluation_20261029.american.targets import join
from research.joint_evaluation_20261029.root import american_continue as continuation

RESULTS = r.ROOT / 'experiments/results/joint_american_20261029'
METHODS = ('TreeLoom-Joint', 'ARGformer', 'FLARE')


def need(value, message):
    if not value:
        raise ValueError(message)


def metric_values(probabilities, truth):
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, probabilities.argmax(1)), 1)
    result = metrics(confusion)
    chosen = probabilities[np.arange(len(truth)), truth].astype(np.float64)
    result.update(Brier=float(((probabilities.astype(np.float64) - np.eye(3)[truth])**2).sum(1).mean()),
                  NLL=None if (chosen == 0).any() else float(-np.log(chosen).mean()),
                  NLL_status='positive_infinity' if (chosen == 0).any() else 'finite')
    return result


def aggregate(values):
    values = np.asarray(values, dtype=np.float64)
    need(values.size >= 2 and np.isfinite(values).all(), 'Finite independent-parent values required')
    return dict(mean=float(values.mean()), sample_SD=float(values.std(ddof=1)), parents=int(values.size))


def readiness(plan_path):
    plan_ref = r.ref(plan_path)
    plan = r.read(plan_ref)
    protocol = continuation.validate_plan(plan)
    records = r.path(plan['records_directory'])
    final_path = records / 'scores_completed.json'
    if not final_path.exists():
        return None
    final = r.read(r.ref(final_path))
    need(final['plan_ref'] == plan_ref, 'Exact completed American continuation')
    score_wait = r.read(final['score_actual_wait_ref'])
    continuation.validate_wait(score_wait, plan['stages'][-1], plan_ref)
    continuation.validate_product(plan['stages'][-1], plan, protocol)
    barrier = r.read(final['native_prediction_barrier_ref'])
    native = continuation.native_ready(plan, protocol)
    need(native is not None and barrier['plan_ref'] == plan_ref and barrier['parents'] == native
         and barrier['truth_opened'] is False, 'Authenticated immutable all-parent native barrier')
    return plan_ref, plan, protocol, final, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--american-plan', required=True)
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args()
    ready = readiness(args.american_plan)
    if args.check_ready:
        print(json.dumps(dict(ready=ready is not None, truth_opened=False, outputs_written=False)))
        return
    need(ready is not None, 'Wait for the complete American scoring chain')
    plan_ref, plan, protocol, final, native = ready
    native_by_parent = {entry['parent_id']: entry for entry in native}
    own_product_ref = r.ref(RESULTS / 'scores_v2/product.json')
    own_product = r.read(own_product_ref)
    fit = r.read(r.ref(RESULTS / 'fit_v2/product.json'))
    need(fit['complete'] and fit['freeze_ref'] == plan['freeze_ref'], 'Exact final fitted model')
    r.read(fit['training_plan_ref'])
    need(r.ref(fit['checkpoint_ref']['path']) == fit['checkpoint_ref'], 'Fixed checkpoint unchanged')
    parents = continuation.heldout(protocol)
    need(set(native_by_parent) == set(parents) == set(own_product['parents']), 'Exact 15 original parents')
    sealed = {}
    for parent in parents:
        focal_ref = protocol['parents'][parent]['binding']['focal_registry_ref']
        records = load_npz(focal_ref)
        product_ref = r.ref(RESULTS / 'predictions_v2' / parent / 'product.json')
        need(product_ref == own_product['parents'][parent]['prediction_product_ref'], 'Same prediction product as completed score')
        product = r.read(product_ref)
        need(product['model_id'] == protocol['model_id'] and product['checkpoint_ref'] == fit['checkpoint_ref'], 'Exact fixed final model identity')
        need(product['complete'] and product['rows'] == 10000 and product['parent_id'] == parent
             and product['focal_registry_ref'] == focal_ref and product['freeze_ref'] == plan['freeze_ref']
             and product['query_ancestry_truth_opened'] is False, 'Exact new-model prediction identity')
        refs = dict(zip(METHODS, [product['prediction_ref'], *native_by_parent[parent]['prediction_refs']]))
        predictions = {method: load_npz(reference) for method, reference in refs.items()}
        for prediction in predictions.values():
            verify_prediction(prediction, records)
        sealed[parent] = (records, predictions, refs)
    output = RESULTS / 'comparison_v1'
    output.mkdir(exist_ok=False)
    seal_ref = r.save(output / 'prediction_seal.json', dict(schema='american_all45_common_prediction_seal_v1',
        plan_ref=plan_ref, freeze_ref=plan['freeze_ref'], methods=list(METHODS),
        parents={parent: data[2] for parent, data in sealed.items()}, ancestry_truth_opened=False))
    rows = {}
    for parent in parents:
        records, predictions, refs = sealed[parent]
        truth_ref = own_product['parents'][parent]['truth_ref']
        truth = join(records, load_npz(truth_ref, cap=8*2**30))
        row = {method: metric_values(prediction['probabilities'], truth) for method, prediction in predictions.items()}
        need(row['TreeLoom-Joint'] == own_product['parents'][parent]['result'], 'Exact parity with completed new-model score')
        rows[parent] = dict(split=protocol['parents'][parent]['split'], methods=row, truth_ref=truth_ref,
                            prediction_refs=refs, targets=len(truth))
    summary, paired = {}, {}
    for split in ('validation', 'test'):
        selected = [row for row in rows.values() if row['split'] == split]
        summary[split] = {}
        paired[split] = {}
        for method in METHODS:
            summary[split][method] = {key: aggregate([row['methods'][method][key] for row in selected])
                                     for key in ('accuracy', 'macro_F1', 'Brier')}
            summary[split][method]['per_class'] = {
                label: {key: aggregate([row['methods'][method]['per_class'][label][key] for row in selected])
                        for key in ('precision', 'recall', 'F1')}
                for label in ('EUR', 'EAS', 'AFR')}
        for other in METHODS[1:]:
            paired[split][other] = {key: aggregate([row['methods']['TreeLoom-Joint'][key] - row['methods'][other][key]
                                                   for row in selected]) for key in ('accuracy', 'macro_F1', 'Brier')}
    result = dict(schema='american_joint_native_methods_comparison_v1', complete=True,
        methods=list(METHODS), ARGformer_evaluator='reference_PCA10_nearest_class_centroid',
        ARGformer_evaluator_semantics='paper_oriented_adaptation_of_author_embeddings', class_order=['EUR', 'EAS', 'AFR'], target_rows_per_parent=10000,
        parents=rows, summary=summary, paired_new_minus_competitor=paired, prediction_seal_ref=seal_ref,
        original_new_model_score_ref=own_product_ref, original_native_barrier_ref=final['native_prediction_barrier_ref'],
        simulation_or_targets_modified=False, probabilities_modified=False,
        uncertainty='Sample SD across independent simulated chromosomes. Five validation and ten TEST parents are separate.',
        interpretation='Primary comparison uses native accuracy and class precision, recall and F1 on identical canonical targets. '
                       'Raw probability diagnostics retain the recorded outputs. The paper-oriented ARGformer evaluator uses reference PCA with ten components and nearest class centroids, transported as one-hot hard calls. '
                       'Infinite NLL is preserved and is not evidence of inferior probability calibration. This evaluator is not labelled as native top-one retrieval.',
        publication_scope='Shared benchmark generated from the recorded paper recipe, not a numerical reproduction claim for a published table.')
    need(sum(row['targets'] for row in rows.values()) == 150000, 'All 150000 held-out records retained')
    result_ref = r.save(output / 'product.json', result)
    print(json.dumps(dict(result_ref=result_ref, summary=summary['test']), sort_keys=True))


if __name__ == '__main__':
    main()

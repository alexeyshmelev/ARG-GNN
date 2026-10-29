"""Publish saved comparison and an explicitly deduplicated operational snapshot."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from research.joint_evaluation_20261029.root.result_sections import completed_sections

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/joint_evaluation_20261029'
COMPARISON = BASE / 'scoring/comparison_v4.json'
EXPECTED = 'a1ee5f1016e8a9edfa6b03fdc49471005df1ad3453144d4a480af513f0b0bd64'


def read(path):
    return json.loads(path.read_text())


def main():
    assert hashlib.sha256(COMPARISON.read_bytes()).hexdigest() == EXPECTED
    scores = read(COMPARISON)
    original = read(ROOT / '.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json')
    effective = dict(original['tasks'])
    successor_path = BASE / 'queue/successor_records_v2/state.json'
    successor = read(successor_path) if successor_path.exists() else None
    extra_tasks = {}
    if successor is not None:
        effective.update({name: row for name, row in successor['tasks'].items() if name in effective})
        extra_tasks = {name: row for name, row in successor['tasks'].items() if name not in effective}
    overlays = {}
    for relative in ['queue/records_v1/state.json', 'queue/argmix_records_v1/state.json']:
        path = BASE / relative
        if not path.exists():
            continue
        state = read(path)
        overlays[relative] = state.get('counts', {})
        for name, row in state['tasks'].items():
            assert name in effective
            effective[name] = row
    counts = Counter(row['status'] for row in effective.values())
    unfinished = sum(value for status, value in counts.items() if status not in ('succeeded', 'superseded'))
    def optional_state(relative):
        path = BASE / relative
        return read(path) if path.exists() else None
    model_chains = {
        'american_joint': optional_state('root/american_continue_run_v1/completed.json'),
        'european_four_joint': optional_state('european_four/chain_records_v1/gpu_chain/actual_wait.json'),
        'argmix_joint': optional_state('argmix/records_v2/gpu_chain_completed.json'),
    }
    native_complete = sum(name.startswith('paper_american_argformer_') and name.endswith('_predict')
                          and not name.endswith('_treeloom_predict') and row['status'] == 'succeeded'
                          for name, row in effective.items())
    running = [{'task': name, 'pid': row.get('pid'), 'physical_gpu': row.get('physical_gpu')}
               for name, row in effective.items() if row['status'] == 'running']
    status = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'original_counts': original['counts'],
              'deduplicated_campaign_counts': dict(counts), 'unfinished_original_campaign_stages': unfinished,
              'original_pending_or_running_stages': counts['pending'] + counts['running'],
              'parallel_controller_counts': overlays, 'running': running,
              'successor_counts': successor.get('counts') if successor else None,
              'additional_successor_tasks': extra_tasks,
              'root_model_chain_receipts': model_chains,
              'successor_handoff_watch': optional_state('queue/successor_records_v2/watch_status.json'),
              'native_ARGformer_prediction_datasets_complete': native_complete,
              'native_ARGformer_prediction_datasets_total': 15,
              'new_European_joint_test_prediction_parents_complete': 5,
              'new_American_joint_pipeline': optional_state('root/american_continue_run_v1/status.json'),
              'direct_four_class_CNN': (optional_state('queue/cnn4_v3_records/state.json') or read(BASE / 'queue/cnn4_v2_records/state.json')).get('counts', {}),
              'direct_four_class_joint': optional_state('european_four/chain_records_v1/status.json') or 'Frozen40000-target direct-four pipeline independently reviewed, root launch being prepared',
              'new_ARGMix_joint_pipeline': optional_state('argmix/records_v2/state.json') or 'Original observed adapter validated, full14-parent pipeline prepared, production admission and nativeTRAINproducers pending',
              'notes': ['Parallel queues own original task IDs and are counted once',
                        'Historical failed old-TreeLoom rows remain visible',
                        'New-model integration work outside the old 252-stage campaign is not hidden in its stage count']}
    (BASE / 'root/CURRENT_STATUS.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    names = {'treeloom_joint_structure_v1': 'TreeLoom-Joint (new)',
             'ancestralpaths_historical_cnn_native6': 'AncestralPaths historical CNN (author-source port)',
             'ancestralpaths_dense_control': 'AncestralPaths Dense control',
             'treeloom': 'TreeLoom (previous architecture)',
             'gst_lai_8_native_graphsetconv': 'GST-LAI-8 (ancestry adaptation)'}
    order = list(names)
    columns = [('accuracy', 100, 2, True), ('macro_F1', 100, 2, True),
               ('Brier', 1, 4, False), ('raw_NLL', 1, 4, False), ('ECE_10_equal_width', 100, 2, False)]
    lines = ['<!-- joint-model-20261029:start -->', '## TreeLoom-Joint: held-out benchmark results, 29 October', '',
             '**The new model is trained, integrated and scored on all five European TEST chromosomes.** '
             'Its six-class accuracy is **96.33 ± 0.16%**, compared with **92.70 ± 0.35%** for the historical AncestralPaths CNN. '
             'The paired improvement is **3.63 ± 0.28 percentage points** across chromosomes. '
             'Every method is scored on the same 50,000 canonical TEST targets. Architecture and checkpoint selection used TRAIN data only.', '',
             '| Method | Accuracy (%) ↑ | Macro F1 (%) ↑ | Brier ↓ | NLL ↓ | Calibration error (%) ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for method in order:
        values = []
        for metric, scale, decimals, higher in columns:
            entry = scores['summary'][method]['overall'][metric]
            best = (max if higher else min)(scores['summary'][m]['overall'][metric]['mean'] for m in order)
            value = f"{entry['mean'] * scale:.{decimals}f} ± {entry['sample_SD'] * scale:.{decimals}f}"
            values.append('**' + value + '**' if entry['mean'] == best else value)
        lines.append('| ' + names[method] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'Values are mean ± sample SD across five independently simulated chromosomes, not across training seeds. '
              'Calibration error uses ten equal-width confidence bins. The new model improves accuracy, macro F1, Brier score and NLL on every chromosome, '
              'but its confidence calibration is worse than the CNN. This is a comparison on shared simulated data, not a claim to reproduce an original published numerical table.', '',
              '| Simulated cohort | CNN accuracy (%) | TreeLoom-Joint accuracy (%) | Gain (percentage points) |',
              '|---|---:|---:|---:|']
    for group, label in [('present_day', 'Present day'), ('Bronze_Age', 'Bronze Age'), ('BAA', 'BAA'), ('Neo', 'Neolithic'), ('Yam', 'Yamnaya')]:
        cnn = scores['summary']['ancestralpaths_historical_cnn_native6'][group]['accuracy']
        new = scores['summary']['treeloom_joint_structure_v1'][group]['accuracy']
        lines.append(f"| {label} | {100 * cnn['mean']:.2f} ± {100 * cnn['sample_SD']:.2f} | **{100 * new['mean']:.2f} ± {100 * new['sample_SD']:.2f}** | +{100 * (new['mean'] - cnn['mean']):.2f} |")
    lines += ['', '**Execution:** the full European fit completed 50 epochs and 3,900 updates in 6.00 minutes. '
              'Warm training averaged 7,600 targets/s and 88.2% sampled GPU activity. '
              'Each TEST chromosome took about 62–66 seconds for preparation and prediction, with four CPU workers. '
              'Test preparation is streamed through RAM, with no persistent bulk cache.', '',
              '**Architecture:** three joint PyTorch Geometric layers exchange messages within complete selected native trees and across trees through shared leaves and branch structure. '
              'All queries at one SNP reuse one encoding. There is no recurrent network. The general idea of graph windows has prior art, so this result does not establish a new general GNN kernel.', '',
              f'**Queue snapshot:** {counts["succeeded"]} original campaign stages completed, **{counts["pending"] + counts["running"]} pending or running** and {counts["superseded"]} explicitly superseded after counting parallel task IDs only once. '
              f'The records contain {counts["running"]} running, {counts["pending"]} pending, {counts["blocked"]} blocked and {counts["failed"]} failed stages. '
              'These are preparation, fitting, prediction and scoring stages, not that many separate experiments. '
              'The failed previous American TreeLoom and its dependent rows are retained as history.', '',
              f'- **ARGformer:** {native_complete}/15 dataset predictions complete. The remaining {15-native_complete} parents are assigned to the original GPU3 lane and the two native lanes on GPUs4 and5. Their original training schedules remain unchanged.',
              '- **FLARE:** 15/15 prediction datasets complete and authenticated.',
              '- **ARGMix:** native Relate inference and the remaining original producers continue through the native queues. The new TreeLoom pipeline covers five TRAIN, one validation and eight TEST parents. Its full cache needs admission within the existing1.2TB storage ceiling. No final ARGMix comparison is available yet.',
              '- **New American TreeLoom:** the full lossless TRAIN cache is building with 16 workers. It covers all 158,476 windows and 200,000 original target rows. Corrected partial-batch weighting, exact cache restoration and the training/prediction chain passed independent review. GPU qualification on the largest TRAIN batches precedes the fixed 13-epoch fit and all 15 held-out predictions. No American score is available yet.',
              '- **Direct four-class AncestralPaths CNN:** the corrected queue is active. An input manifest exceeded a metadata read limit on the first attempt. The repair changes that bound only, and a full real-data preparation test passed. The original CNN and its 40,000 supervised rows are unchanged. The separate direct four-class TreeLoom pipeline on those same rows is frozen and independently reviewed.', '',
              '[Detailed scores, class metrics and confusion matrices](research/joint_evaluation_20261029/scoring/comparison_v4.json) · '
              '[Independent result review](research/joint_evaluation_20261029/scoring/FINAL_RESULT_REVIEW_v1.json) · '
              '[Current queue snapshot](research/joint_evaluation_20261029/root/CURRENT_STATUS.json) · '
              '[Integration work plan](research/joint_evaluation_20261029/PLAN.md) · '
              '[Frozen model and development results](research/joint_model_20261029/PLAN.md)',
              '<!-- joint-model-20261029:end -->']
    extra, completed_artifacts = completed_sections()
    if any('/joint_american_20261029/comparison_v1/' in path for path in completed_artifacts):
        lines = [line.replace('The full lossless TRAIN cache is building with 16 workers.', 'The full lossless TRAIN cache was built with 16 workers.').replace('No American score is available yet.', 'The completed comparison is reported below.') for line in lines]
    if any('/european_four/outputs/paired_score/' in path for path in completed_artifacts):
        lines = [line.replace('The corrected queue is active.', 'The direct four-class comparison is complete and reported below.') for line in lines]
    if any('/joint_argmix_20261029/scores/' in path for path in completed_artifacts):
        lines = ['- **ARGMix:** all eight TEST parents are scored against the new TreeLoom on the same 800,000 original targets. The completed comparison is reported below.' if line.startswith('- **ARGMix:**') else line for line in lines]
    lines[-1:-1] = extra + ['']
    status['completed_comparison_artifacts'] = completed_artifacts
    (BASE / 'root/CURRENT_STATUS.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    readme = ROOT / 'README.md'
    text = readme.read_text()
    start = text.index('<!-- joint-model-20261029:start -->')
    end = text.index('<!-- joint-model-20261029:end -->') + len('<!-- joint-model-20261029:end -->')
    readme.write_text(text[:start] + '\n'.join(lines) + text[end:])
    print(json.dumps({'published': True, 'comparison_sha256': EXPECTED, 'counts': dict(counts), 'unfinished': unfinished}))


if __name__ == '__main__':
    main()

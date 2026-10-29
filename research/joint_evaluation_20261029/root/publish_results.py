"""Publish saved comparison and an explicitly deduplicated operational snapshot."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

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
    unfinished = sum(value for status, value in counts.items() if status != 'succeeded')
    native_complete = sum(name.startswith('paper_american_argformer_') and name.endswith('_predict')
                          and not name.endswith('_treeloom_predict') and row['status'] == 'succeeded'
                          for name, row in effective.items())
    running = [{'task': name, 'pid': row.get('pid'), 'physical_gpu': row.get('physical_gpu')}
               for name, row in effective.items() if row['status'] == 'running']
    status = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'original_counts': original['counts'],
              'deduplicated_campaign_counts': dict(counts), 'unfinished_original_campaign_stages': unfinished,
              'parallel_controller_counts': overlays, 'running': running,
              'native_ARGformer_prediction_datasets_complete': native_complete,
              'native_ARGformer_prediction_datasets_total': 15,
              'new_European_joint_test_prediction_parents_complete': 5,
              'new_American_joint_pipeline': 'TRAIN smoke complete, lossless cache and weighting repair unfinished',
              'new_ARGMix_joint_pipeline': 'not yet trained',
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
              f'**Queue snapshot:** {counts["succeeded"]} original campaign stages completed and **{unfinished} unfinished** after counting parallel task IDs only once. '
              f'The unfinished stages comprise {counts["running"]} running, {counts["pending"]} pending, {counts["blocked"]} blocked and {counts["failed"]} failed. '
              'These are preparation, fitting, prediction and scoring stages, not that many separate experiments. '
              'The failed previous American TreeLoom and its dependent rows are retained as history.', '',
              f'- **ARGformer:** {native_complete}/15 dataset predictions complete. The remaining seven independent parents are now in two automatic lanes assigned to GPUs 4 and 5. '
              'Parents003 and004 have completed CPU preparation and started training on GPUs4 and5. The existing parent002 run continues on GPU3.',
              '- **FLARE:** 15/15 prediction datasets complete and authenticated.',
              '- **ARGMix:** native Relate inference for TRAIN002 is running in a separate CPU queue after repairing storage admission. Later stages and the new TreeLoom comparison remain unfinished.',
              '- **New American TreeLoom:** the adapter passed its TRAIN smoke test. A lossless cache and a correction to weighting in partial batches remain unfinished before launch. The uncached route would repeat expensive CPU preparation. No American score is available yet.',
              '- **Direct four-class AncestralPaths CNN:** the data and author runtime are available. The execution adapter is being prepared.', '',
              '[Detailed scores, class metrics and confusion matrices](research/joint_evaluation_20261029/scoring/comparison_v4.json) · '
              '[Independent result review](research/joint_evaluation_20261029/scoring/FINAL_RESULT_REVIEW_v1.json) · '
              '[Current queue snapshot](research/joint_evaluation_20261029/root/CURRENT_STATUS.json) · '
              '[Integration work plan](research/joint_evaluation_20261029/PLAN.md) · '
              '[Frozen model and development results](research/joint_model_20261029/PLAN.md)',
              '<!-- joint-model-20261029:end -->']
    readme = ROOT / 'README.md'
    text = readme.read_text()
    start = text.index('<!-- joint-model-20261029:start -->')
    end = text.index('<!-- joint-model-20261029:end -->') + len('<!-- joint-model-20261029:end -->')
    readme.write_text(text[:start] + '\n'.join(lines) + text[end:])
    print(json.dumps({'published': True, 'comparison_sha256': EXPECTED, 'counts': dict(counts), 'unfinished': unfinished}))


if __name__ == '__main__':
    main()

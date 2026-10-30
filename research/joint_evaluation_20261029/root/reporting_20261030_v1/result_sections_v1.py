"""Render only completed, genuinely waited comparison products into README."""
from pathlib import Path
import hashlib
import json
from research.joint_evaluation_20261029.root.reporting_20261030_v1.effective_state_v1 import comparator_metadata

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/joint_evaluation_20261029'


def repo_path(value):
    return str((ROOT / value).resolve().relative_to(ROOT))


def read_ref(reference):
    path = (ROOT / reference['path']).resolve()
    assert path.is_relative_to(ROOT)
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == reference['sha256']
    return json.loads(payload)


def ready(wait_path):
    if not wait_path.exists():
        return None
    wait = json.loads(wait_path.read_text())
    if not (wait.get('actual_wait') is True and wait.get('actual_returncode') == 0
            and wait.get('owned_process_group_absent') is True):
        return None
    assert wait.get('products_complete', True) is True
    return wait


def table(rows, columns):
    result = ['| Method | ' + ' | '.join(label for _, label, _, _, _ in columns) + ' |',
              '|---|' + '---:|' * len(columns)]
    for name, values in rows.items():
        cells = []
        for key, label, scale, digits, higher in columns:
            score = values[key]
            best = (max if higher else min)(row[key]['mean'] for row in rows.values())
            value = f"{scale*score['mean']:.{digits}f} ± {scale*score['sample_SD']:.{digits}f}"
            cells.append('**' + value + '**' if score['mean'] == best else value)
        result.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    return result


def completed_sections():
    lines, artifacts = [], []
    wait = ready(BASE / 'root/finish_records_v1/american_comparison/actual_wait.json')
    if wait:
        reference = wait['result_ref']
        result = read_ref(reference)
        assert result['complete'] and len(result['parents']) == 15
        rows = dict(result['summary']['test'])
        rows['ARGformer (reference PCA10 centroid)'] = rows.pop('ARGformer')
        lines += ['', '### American benchmark: completed comparison', '',
                  'Ten TEST chromosomes, 10,000 identical target rows per chromosome. Values are mean ± sample SD. Validation chromosomes are reported separately in the full result.', '']
        lines += table(rows, [('accuracy', 'Accuracy (%) ↑', 100, 2, True), ('macro_F1', 'Macro F1 (%) ↑', 100, 2, True)])
        lines += ['', 'The ARGformer condition uses author embeddings with the recorded reference PCA10 centroid evaluator. It is not described as native top-one retrieval. Raw hard calls and probability diagnostics remain explicit in the full result.', '',
                  '[Full American scores, per-class metrics and paired differences](' + repo_path(reference['path']) + ')']
        artifacts.append(repo_path(reference['path']))
    wait = ready(BASE / 'european_four/chain_records_v1/paired_score/actual_wait.json')
    if wait:
        disclosure_ref = wait['result_ref']
        disclosure = read_ref(disclosure_ref)
        reference = disclosure['score_ref']
        result = read_ref(reference)
        rows = {}
        for method, groups in result['summaries'].items():
            label = 'TreeLoom-Joint' if method.startswith('treeloom_joint') else 'AncestralPaths CNN (direct four outputs)'
            rows[label] = dict(present_day=groups['present_day']['accuracy'], overall=groups['overall']['accuracy'], F1=groups['overall']['macro_F1'])
        lines += ['', '### European benchmark: completed direct four-class comparison', '',
                  'Both models are trained on the same 40,000 supervised rows and scored on the same 40,000 TEST rows. Values are mean ± sample SD across five chromosomes. Present-day accuracy is the registered primary metric.', '']
        lines += table(rows, [('present_day', 'Present-day accuracy (%) ↑', 100, 2, True), ('overall', 'Overall accuracy (%) ↑', 100, 2, True), ('F1', 'Overall macro F1 (%) ↑', 100, 2, True)])
        lines += ['', 'The models use different observed representations. TreeLoom keeps BAA query leaves as unknown context, with no BAA target labels. The CNN uses its historical four-path features. Both have four direct output classes, with no probability merging.', '',
                  '[Full direct four-class scores and cohort matrices](' + repo_path(reference['path']) + ')']
        artifacts += [repo_path(reference['path']), repo_path(disclosure_ref['path'])]
    wait = ready(ROOT / 'experiments/results/joint_argmix_20261029/scores/actual_wait.json')
    if wait:
        reference = wait['result_ref']
        result = read_ref(reference)
        other = read_ref(result['paired_comparator_ref'])
        assert len(result['results']) == len(other['results']) == 8
        rows = {}
        disclosure = comparator_metadata(other)
        for label, data in [('TreeLoom-Joint', result), (disclosure['label'], other)]:
            rows[label] = {split: dict(mean=data['summaries'][split]['mean_accuracy'], sample_SD=data['summaries'][split]['sample_sd']) for split in ('iid', 'misspec')}
        lines += ['', '### ARGMix benchmark: completed comparison', '',
                  'All 800,000 original query/tree targets are retained. Values are mean ± sample SD across three IID and five misspecification chromosomes.', '']
        lines += table(rows, [('iid', 'IID accuracy (%) ↑', 100, 2, True), ('misspec', 'Misspecification accuracy (%) ↑', 100, 2, True)])
        lines += ['', 'Comparator condition: `' + disclosure['condition'] + '`. This is a registered reconstruction on frozen shared data. It does not establish reproduction of an original published numerical table. Other named variants require their own distinct results and are not implied by this condition.', '', '[Full ARGMix scores, cohort results and paired differences](' + repo_path(reference['path']) + ')']
        artifacts += [repo_path(reference['path']), repo_path(result['paired_comparator_ref']['path'])]
    return lines, artifacts

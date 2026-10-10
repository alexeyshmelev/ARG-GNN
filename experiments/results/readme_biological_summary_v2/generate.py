from pathlib import Path
import json, hashlib, statistics, datetime
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'experiments/results/readme_biological_summary_v2'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
old=json.loads((ROOT/'experiments/results/readme_biological_summary_v1/receipt.json').read_text())
sources={}
for name,binding in old['source_summaries'].items():
    p=ROOT/binding['path']
    assert sha(p)==binding['sha256'], ('source changed',name)
    sources[name]=json.loads(p.read_text())
group='primary_present_day_64'
rows=[]
raw_bindings={}
def emit(name, a, seeds, ece, scope, params=None):
    rows.append(dict(method=name,accuracy=a['accuracy'],nll=a['nll'],brier=a['brier'],
                     ece=ece,seeds=seeds,scope=scope,parameters=params))
mix=sources['ARGMix']
a=mix['rows']
means={k:statistics.mean(row['groups'][group][k] for row in a) for k in ['accuracy','NLL','Brier','ECE']}
emit('ARGMix, current author release',dict(accuracy=means['accuracy'],nll=means['NLL'],brier=means['Brier']),[50],means['ECE'],'current author release transferred to common task, not April paper reproduction')
for key,label in [('candidate_age_d2','Our candidate, age D2'),('candidate_plain_d2','Our candidate, plain D2'),('matched_gru_d2','Matched GRU D2')]:
    c=sources['candidate']['conditions'][key]
    ece=statistics.mean(row['groups'][group]['calibration_ece'] for row in c['rows'])
    for row in c['rows']:
        p=ROOT/row['metrics_source']
        assert sha(p)==row['metrics_source_sha256']
        raw=json.loads(p.read_text())['C4_exact_support_metrics'][group]
        assert len(raw['calibration_reliability_bins'])==15
        assert raw['calibration_ece']==row['groups'][group]['calibration_ece']
        raw_bindings[row['metrics_source']]=row['metrics_source_sha256']
    emit(label,c['means'][group],[101],ece,'candidate or matched architectural control',c['parameters'])
ap=sources['AP']['groups'][group]['mean_parent']
emit('AncestralPaths, released Dense',dict(accuracy=ap['auxiliary_C4_argmax_accuracy'],nll=ap['C4_nll_from_finite_logp'],brier=ap['C4_brier_stable_probabilities']),[101,202,303],ap['C4_calibration_ece_stable_probabilities'],'auxiliary stable-logit readout, unchanged weights and argmax')
for key,label in [('gst32_primary','GST-style reconstruction, width 32'),('gst64_capacity_sensitivity','GST-style reconstruction, width 64')]:
    c=sources['GST']['conditions'][key]
    eces=[]
    for row in c['rows']:
        p=ROOT/row['metrics_source']
        assert sha(p)==row['metrics_source_sha256']
        raw=json.loads(p.read_text())['C4_exact_support_metrics'][group]
        assert len(raw['calibration_reliability_bins'])==15
        eces.append(raw['calibration_ece'])
        raw_bindings[row['metrics_source']]=row['metrics_source_sha256']
    emit(label,c['means'][group],[101],statistics.mean(eces),'independent architectural reconstruction, not author implementation',c['parameters'])
# Verify all previously reported primary accuracy/NLL cells without changing their records.
for row,original in zip(rows,sources['comparison']['rows']):
    assert row['accuracy']==original['accuracy']
    assert row['nll']==original['nll']
candidate=rows[1]
deltas=[dict(comparator=x['method'],candidate_minus_comparator_accuracy_pp=100*(candidate['accuracy']-x['accuracy']),candidate_minus_comparator_nll=candidate['nll']-x['nll'],candidate_minus_comparator_brier=candidate['brier']-x['brier'],candidate_minus_comparator_ece=candidate['ece']-x['ece']) for x in rows if x is not candidate]
summary=dict(kind='README_completed_biological_metrics_v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,primary_group=group,independent_VAL_parents=2,TEST_parents=0,aggregation='equal-parent means, AP first averaging optimizer seeds within parent',source_summaries=old['source_summaries'],raw_metric_bindings=raw_bindings,accuracy_differences=deltas,ARGformer_ranking_qualified=False,statistical_superiority_established=False,new_census_prototype_included=False)
lines=['| Method | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Optimizer seeds |','| --- | ---: | ---: | ---: | ---: | --- |']
for x in rows:
    mark='†' if x['method'].startswith('AncestralPaths') else ''
    lines.append(f"| {x['method']} | {100*x['accuracy']:.2f}% | {x['nll']:.4f}{mark} | {x['brier']:.4f}{mark} | {x['ece']:.4f}{mark} | "+', '.join(str(s) for s in x['seeds'])+' |')
lines.append('| ARGformer, author implementation | Pending qualified comparison | — | — | — | Native fits completed |')
readme=ROOT/'README.md'
assert sha(readme)==old['readme_sha256']
before=readme.read_text()
start=before.index('| Method | Accuracy ↑ |')
stop=before.index('\n\n† AncestralPaths',start)
after=before[:start]+'\n'.join(lines)+before[stop:]
after=after.replace('**Higher accuracy is better. Lower NLL and Brier are better.**','**Higher accuracy is better. Lower NLL, Brier and ECE are better.**')
needle='**Against our strongest current comparator:**'
metric_text='Metrics: accuracy is the fraction of correctly assigned query base pairs. NLL is negative log probability of the true class, in nats. Brier is the sum of squared errors across the four class probabilities. ECE measures the confidence-versus-accuracy gap in 15 equally spaced confidence bins. Scores weight exact physical query support and average parent-level values. GST32 has the lowest ECE in this pilot, despite its lower accuracy. Calibration alone does not establish better ancestry inference.\n\n'
after=after.replace(needle,metric_text+needle)
after=after.replace('† AncestralPaths NLL and Brier use','† AncestralPaths NLL, Brier and ECE use')
after=after.replace('- [Concise comparison status]', '- [Full README metrics and pairwise differences](experiments/results/readme_biological_summary_v2/summary.json).\n- [Concise comparison status]')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
readme.write_text(after)
receipt=dict(kind='main_README_biological_summary_v2',created_utc=summary['created_utc'],previous_readme_sha256=old['readme_sha256'],readme_sha256=sha(readme),summary_sha256=sha(OUT/'summary.json'),generator_sha256=sha(OUT/'generate.py'),source_summaries=old['source_summaries'],verification='hash-bound completed summaries and candidate/GST raw metric files only, previous accuracy/NLL cells unchanged, ECE arithmetic independently aggregated, no fit or prediction array access',new_results=False,completed_only=True,TEST_results=False)
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(readme_sha256=receipt['readme_sha256'],ECEs={x['method']:x['ece'] for x in rows},candidate_minus_ARGMix_accuracy_pp=deltas[0]['candidate_minus_comparator_accuracy_pp'],status='README_metrics_verified')))

from pathlib import Path
import json,hashlib,statistics,datetime
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'experiments/results/readme_biological_summary_v3'
OUT.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prev=json.loads((ROOT/'experiments/results/readme_biological_summary_v2/summary.json').read_text())
receipt=json.loads((ROOT/'experiments/results/readme_biological_summary_v2/receipt.json').read_text())
readme=ROOT/'README.md';assert sha(readme)==receipt['readme_sha256']
paths={101:ROOT/'experiments/results/biological_candidate_pilot_v1_seed101_summary/summary.json',202:ROOT/'experiments/results/biological_candidate_pilot_v1_seed202_summary/summary.json'}
pins={101:'f2213e447f8a530539dfeda4e5e7f02b5d067b8ca23618b53c1f7b475c939dd9',202:'7a7bb843794f51faf679ba59e6d2575711e83947a5233495a0bc527d6faca2a5'}
candidates={}
for seed,p in paths.items():
 assert sha(p)==pins[seed]
 candidates[seed]=json.loads(p.read_text())
 assert candidates[seed]['status']=='three_conditions_one_seed_completed'
rows=prev['rows']
condition_keys={'Our candidate, age D2':'candidate_age_d2','Our candidate, plain D2':'candidate_plain_d2','Matched GRU D2':'matched_gru_d2'}
groups=['primary_present_day_64','secondary_ancient_56','all_queries_120']
aggregates={}
for key in condition_keys.values():
 aggregates[key]={group:{metric:statistics.mean(candidates[seed]['conditions'][key]['means'][group][metric] for seed in paths) for metric in ['accuracy','nll','brier','macro_f1']} for group in groups}
 aggregates[key]['primary_present_day_64']['ece']=statistics.mean(statistics.mean(row['groups']['primary_present_day_64']['calibration_ece'] for row in candidates[seed]['conditions'][key]['rows']) for seed in paths)
for row in rows:
 if row['method'] in condition_keys:
  means=aggregates[condition_keys[row['method']]]['primary_present_day_64']
  for k in ['accuracy','nll','brier','ece']:row[k]=means[k]
  row['seeds']=[101,202]
candidate=next(x for x in rows if x['method']=='Our candidate, age D2')
deltas=[dict(comparator=x['method'],candidate_minus_comparator_accuracy_pp=100*(candidate['accuracy']-x['accuracy']),candidate_minus_comparator_nll=candidate['nll']-x['nll'],candidate_minus_comparator_brier=candidate['brier']-x['brier'],candidate_minus_comparator_ece=candidate['ece']-x['ece']) for x in rows if x is not candidate]
summary={**prev,'kind':'README_completed_biological_metrics_v3','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'candidate_completed_seeds':[101,202],'candidate_pending_seed':303,'candidate_seed_group_means':aggregates,'accuracy_differences':deltas,'aggregation':'equal-parent means within each optimizer seed, then equal completed-seed means. No prediction ensemble. Optimizer seeds are not biological replicates.','candidate_seed_summary_bindings':{str(seed):dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for seed,p in paths.items()},'original_paper_dataset_milestone_completed':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
before=readme.read_text()
after=before.replace('our candidate reaches 84.89%','our age-D2 candidate averages 84.80% over completed optimizer seeds 101/202')
after=after.replace('gains 2.83 percentage points','gains 2.74 percentage points').replace('4.86 over the GST','4.77 over the GST').replace('0.71 over the matched GRU','0.83 over the matched GRU')
after=after.replace('Present-day NLL is 0.4273','Present-day NLL is 0.4305').replace('ECE is 0.0323','ECE is 0.0317')
lines=['| Method | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Optimizer seeds |','| --- | ---: | ---: | ---: | ---: | --- |']
for x in rows:
 mark='†' if x['method'].startswith('AncestralPaths') else ''
 lines.append(f"| {x['method']} | {100*x['accuracy']:.2f}% | {x['nll']:.4f}{mark} | {x['brier']:.4f}{mark} | {x['ece']:.4f}{mark} | "+', '.join(str(s) for s in x['seeds'])+' |')
lines.append('| ARGformer, author implementation | Pending qualified comparison | — | — | — | New observed inference completed |')
start=after.index('| Method | Accuracy ↑ |');stop=after.index('\n\n† AncestralPaths',start)
after=after[:start]+'\n'.join(lines)+after[stop:]
after=after.replace('trails ARGMix by 0.66','trails ARGMix by 0.75').replace('differ by only 0.05 points','differ by only 0.01 points')
start=after.index('| Query group |');stop=after.index('\n\n### What these comparisons mean',start)
mix=json.loads((ROOT/'experiments/results/official_argmix_biological_metadata_repair_v3_summary/summary.json').read_text())
lines=['| Query group | Candidate accuracy | ARGMix accuracy | Candidate NLL | ARGMix NLL |','| --- | ---: | ---: | ---: | ---: |']
for g,label in [('secondary_ancient_56','Ancient queries, 56 haplotypes'),('all_queries_120','All queries, 120 haplotypes')]:
 a=aggregates['candidate_age_d2'][g]
 accuracy=statistics.mean(x['groups'][g]['accuracy'] for x in mix['rows'])
 nll=statistics.mean(x['groups'][g]['NLL'] for x in mix['rows'])
 lines.append(f"| {label} | {100*a['accuracy']:.2f}% | {100*accuracy:.2f}% | {a['nll']:.4f} | {nll:.4f} |")
after=after[:start]+'\n'.join(lines)+after[stop:]
after=after.replace('Candidate seeds 202/303 and seven mechanism controls are still running on physical GPUs 2/3.','Candidate seeds 101/202 are completed and averaged in the table. Seed 303 and seven mechanism controls are still running on physical GPU 3. Seed means summarize separate fits, without ensembling probabilities. All planned seeds will be reported. Optimizer repeats reuse the same two validation parents.')
after=after.replace('while the documented tsinfer/tsdate inference family is prepared.','and a separate current tsinfer/tsdate inference condition has now completed for all four development parents, preserving genotypes, sampling ages and connected shared internal IDs. Native fits and retrieval on this new condition are pending.')
after=after.replace('No final TEST results or statistical superiority claim are reported.','No final TEST results or statistical superiority claim are reported. Results average only completed fits and do not establish a final three-seed estimate.')
after=after.replace('## Evidence and research log','## Original paper data requirement\n\nFinal evaluation must use each supplied paper\'s data and task, with exact data, split, preprocessing, native method, checkpoint selection, postprocessing and scoring identities. Every change is recorded. The current development table does not fulfill this milestone. See the [paper-dataset evaluation contract](docs/paper_dataset_evaluation_contract_v1.md) for concrete acquisition routes and unresolved original data/model details.\n\n## Evidence and research log')
after=after.replace('readme_biological_summary_v2/summary.json','readme_biological_summary_v3/summary.json')
after=after.replace('- [Candidate results]', '- [Completed seed 202 results](experiments/results/biological_candidate_pilot_v1_seed202_summary/summary.json) and [four-parent observed tsinfer/tsdate qualification](experiments/results/biological_observed_tsinfer_v2_summary/summary.json).\n- [Candidate results]')
readme.write_text(after)
(OUT/'receipt.json').write_text(json.dumps(dict(kind='main_README_biological_summary_v3',created_utc=summary['created_utc'],previous_readme_sha256=receipt['readme_sha256'],readme_sha256=sha(readme),summary_sha256=sha(OUT/'summary.json'),completed_fits_only=True,heldout_TEST_results=False,probability_ensemble=False,candidate_seed303_pending=True,original_paper_dataset_milestone_completed=False),indent=2)+'\n')
print(json.dumps(dict(candidate_mean_accuracy=candidate['accuracy'],candidate_mean_nll=candidate['nll'],candidate_minus_ARGMix_accuracy_pp=deltas[0]['candidate_minus_comparator_accuracy_pp'],README_sha256=sha(readme))))

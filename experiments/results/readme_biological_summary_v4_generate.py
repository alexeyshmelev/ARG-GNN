from pathlib import Path
import json,hashlib,datetime
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
oldpath=ROOT/'experiments/results/readme_biological_summary_v3/summary.json'
x=json.loads(oldpath.read_bytes())
assert sha(oldpath)=='46a8c66d8d24da63a9de06e9cc404364bc6ebfebf0888a4b52193357aa722786'
for r in x['candidate_seed_summary_bindings'].values():assert sha(ROOT/r['path'])==r['sha256']
for p,h in x['raw_metric_bindings'].items():assert sha(ROOT/p)==h
newpath=ROOT/'experiments/results/biological_candidate_pilot_v1_seed303_summary/summary.json'
new=json.loads(newpath.read_bytes())
assert sha(newpath)=='6f3ed5af46baa6de2da45b26c4895c0096eeeb2e8480a7e4a8cf388a3f56988e'
assert new['training_seeds']==[303] and new['TEST_parents']==0
for name,condition in new['conditions'].items():
 for row in condition['rows']:
  assert sha(ROOT/row['metrics_source'])==row['metrics_source_sha256']
  x['raw_metric_bindings'][row['metrics_source']]=row['metrics_source_sha256']
 for group,values in x['candidate_seed_group_means'][name].items():
  for key in list(values):
   k='calibration_ece' if key=='ece' else key
   values[key]=(2*values[key]+condition['means'][group][k])/3
mapping={'Our candidate, age D2':'candidate_age_d2','Our candidate, plain D2':'candidate_plain_d2','Matched GRU D2':'matched_gru_d2'}
for row in x['rows']:
 if row['method'] in mapping:
  row['seeds']=[101,202,303]
  row.update({key:x['candidate_seed_group_means'][mapping[row['method']]]['primary_present_day_64'][key] for key in ['accuracy','nll','brier','ece']})
age=next(r for r in x['rows'] if r['method']=='Our candidate, age D2')
x['accuracy_differences']=[{'comparator':r['method'],'candidate_minus_comparator_accuracy_pp':100*(age['accuracy']-r['accuracy']),'candidate_minus_comparator_nll':age['nll']-r['nll'],'candidate_minus_comparator_brier':age['brier']-r['brier'],'candidate_minus_comparator_ece':age['ece']-r['ece']} for r in x['rows'] if r is not age]
x['kind']='README_completed_biological_metrics_v4'
x['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
x['candidate_completed_seeds']=[101,202,303];x['candidate_pending_seed']=None
x['candidate_seed_summary_bindings']['303']=ref(newpath)
x['previous_summary']=ref(oldpath)
x['original_paper_dataset_milestone_completed']=False
x['paper_contract_review']=ref(ROOT/'research/reviews/paper_dataset_evaluation_contract_v1_independent/revision_v3/receipt.json')
x['raw_acquisition']=ref(ROOT/'experiments/results/paper_native_raw_acquisition_v1/completion.json')
out=ROOT/'experiments/results/readme_biological_summary_v4';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(x,indent=2)+'\n')
rp=ROOT/'README.md';before=rp.read_bytes();text=before.decode()
rows={r['method']:r for r in x['rows']}
argmix=rows['ARGMix, current author release'];ap=rows['AncestralPaths, released Dense'];gst=rows['GST-style reconstruction, width 32'];gru=rows['Matched GRU D2'];plain=rows['Our candidate, plain D2']
lines=text.splitlines()
for i,line in enumerate(lines):
 if line.startswith('**Current development result:'):
  lines[i]=f"**Current development result: our age-D2 candidate averages {age['accuracy']*100:.2f}% present-day ancestry accuracy over all three planned optimizer seeds, 101/202/303. Current-release ARGMix reaches {argmix['accuracy']*100:.2f}%.** Our candidate has slightly lower negative log likelihood. Overall superiority and distinct architectural novelty remain research objectives."
 elif line.startswith('- **Accuracy gains over three controls.**'):
  lines[i]=f"- **Accuracy gains over three controls.** Our age-D2 candidate gains {(age['accuracy']-ap['accuracy'])*100:.2f} percentage points over released AncestralPaths Dense, {(age['accuracy']-gst['accuracy'])*100:.2f} over the GST-style width 32 reconstruction, and {(age['accuracy']-gru['accuracy'])*100:.2f} over the matched GRU."
 elif line.startswith('- **Better likelihood and calibration than ARGMix'):
  lines[i]=f"- **Better likelihood and calibration than ARGMix in this development run.** Present-day NLL is {age['nll']:.4f} versus {argmix['nll']:.4f}. The same 15-bin calibration ECE is {age['ece']:.4f} versus {argmix['ece']:.4f}. ARGMix has higher accuracy and a better present-day Brier score. These observations need independent confirmation."
 elif any(line.startswith('| '+name+' |') for name in mapping):
  method=next(name for name in mapping if line.startswith('| '+name+' |'));r=rows[method]
  lines[i]=f"| {method} | {r['accuracy']*100:.2f}% | {r['nll']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} | 101, 202, 303 |"
 elif line.startswith('**Against our strongest current comparator:'):
  lines[i]=f"**Against our strongest current comparator:** the age-D2 candidate trails ARGMix by {(argmix['accuracy']-age['accuracy'])*100:.2f} accuracy percentage points. Age and plain candidates differ by only {abs(age['accuracy']-plain['accuracy'])*100:.2f} points. Neither age conditioning nor overall superiority is established by this pilot."
 elif line.startswith('| Ancient queries, 56 haplotypes |'):
  v=x['candidate_seed_group_means']['candidate_age_d2']['secondary_ancient_56']
  lines[i]=f"| Ancient queries, 56 haplotypes | {v['accuracy']*100:.2f}% | 89.33% | {v['nll']:.4f} | 0.3433 |"
 elif line.startswith('| All queries, 120 haplotypes |'):
  v=x['candidate_seed_group_means']['candidate_age_d2']['all_queries_120']
  lines[i]=f"| All queries, 120 haplotypes | {v['accuracy']*100:.2f}% | 87.32% | {v['nll']:.4f} | 0.3924 |"
 elif line.startswith('These are validation results used during model development'):
  lines[i]="These are validation results used during model development and checkpoint selection. **No final TEST results or statistical superiority claim are reported.** All three planned candidate/GRU optimizer seeds, 101/202/303, are completed and included. Optimizer seeds are not independent biological replicates. Seven mechanism controls are still running on physical GPU 3. Seed means summarize separate fits, without ensembling probabilities. All optimizer repeats reuse the same two validation parents."
text='\n'.join(lines)+'\n'
text=text.replace('experiments/results/readme_biological_summary_v3/summary.json','experiments/results/readme_biological_summary_v4/summary.json')
needle='A smaller simulation, current code release or changed ancestry endpoint is reported under its own label.'
assert needle in text
text=text.replace(needle,'All 39 ancient-inclusive Wohns genealogy files cited by ARGformer and both pinned GST raw snapshots have now been acquired and checksum-verified, totaling 2.73 GB. Their original processed subsets and splits are still to be qualified. Acquisition alone does not establish a completed comparison.\n\n'+needle)
text=text.replace('- [Completed seed 202 results]','- [Completed seed 303 results](experiments/results/biological_candidate_pilot_v1_seed303_summary/summary.json), [raw acquisition receipt](experiments/results/paper_native_raw_acquisition_v1/completion.json), and [completed seed 202 results]')
rp.write_text(text)
receipt={'kind':'main_README_biological_summary_v4','created_utc':x['created_utc'],'previous_readme_sha256':hashlib.sha256(before).hexdigest(),'readme_sha256':sha(rp),'summary_sha256':sha(out/'summary.json'),'all_planned_candidate_seeds_included':True,'probability_ensemble':False,'final_TEST_scored':False,'original_paper_comparison_completed':False}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':receipt,'primary':age,'differences':x['accuracy_differences']},indent=2))

"""Postprocess only completed compact H038 diagnostic JSON."""
from pathlib import Path
import os,json,hashlib,datetime,csv,io
import numpy as np
D=Path(__file__).parent;ROOT=D.parents[2]
os.environ['MPLCONFIGDIR']=str(D/'matplotlib_cache');os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def used():return sum(p.stat().st_size for p in D.rglob('*') if p.is_file())
def save(name,x):
 b=(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();assert used()+len(b)<10_000_000
 (D/name).write_bytes(b)
j=json.loads((D/'summary.json').read_text())
models=j['models'];names={'aligned':'H038 aligned','zero_concordance':'H038 zero concordance'}
def pct(x):return f"{100*x['mean']:.2f}±{100*x['sample_sd']:.2f}%"
derived={}
for arm,r in models.items():
 c=r['pooled_count_decomposition'];n=c['true'];extra=c['extra']
 derived[arm]=dict(true_boundaries=n,predicted_boundaries=c['predicted'],extra_boundaries=extra,missed_boundaries=c['missed'],
  predicted_to_true_boundary_ratio=c['predicted']/n,
  geometric_unreachable_true_fraction=c['true_no_candidate_within_1kb']/n,
  oracle_competition_fraction=c['true_candidate_competition_gap']/n,
  observed_misses_with_candidate_count=c['missed']-c['missed_no_candidate_within_1kb'],
  observed_misses_with_candidate_fraction=(c['missed']-c['missed_no_candidate_within_1kb'])/c['missed'],
  extra_near_true_within_tolerance_fraction=c['extra_pred_near_true_within_1kb']/extra,
  positional_matches_correct_transition_fraction=c['positional_matches_same_transition']/c['matched'])
save('biological_decomposition.json',dict(derived=derived,source_ref=ref(D/'summary.json'),
 no_prediction_modification=True,no_H039_boundary_claim=True))
fields=['arm','precision_percent_mean','precision_percent_SD','recall_percent_mean','recall_percent_SD',
 'F1_percent_mean','F1_percent_SD','hold_oracle_recall_percent_mean','hold_oracle_recall_percent_SD',
 'strict_10kb_island_extra_fraction_percent_mean','strict_10kb_island_extra_fraction_percent_SD']
out=io.StringIO();w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
for arm,r in models.items():
 v=r['mean_SD_across8simulations'];s=r['short_tract_parent_summaries']['10000']['fraction_of_extra_boundaries_in_strict_false_islands']
 row={'arm':arm,'strict_10kb_island_extra_fraction_percent_mean':100*s['mean'],
  'strict_10kb_island_extra_fraction_percent_SD':100*s['sample_sd']}
 for label,key in [('precision','precision'),('recall','recall'),('F1','f1'),('hold_oracle_recall','representable_boundary_recall_upper_bound')]:
  row[label+'_percent_mean']=100*v[key]['mean'];row[label+'_percent_SD']=100*v[key]['sample_sd']
 w.writerow(row)
(D/'diagnostic_metrics.csv').write_text(out.getvalue())
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
keys=['precision','recall','f1','representable_boundary_recall_upper_bound'];x=np.arange(4)
for i,arm in enumerate(['aligned','zero_concordance']):
 v=models[arm]['mean_SD_across8simulations']
 ax.bar(x+(i-.5)*.32,[100*v[k]['mean'] for k in keys],width=.3,
  yerr=[100*v[k]['sample_sd'] for k in keys],capsize=3,label=names[arm])
ax.set_xticks(x,['Precision','Recall','F1','Hold-edge recall\noracle upper bound'])
ax.set_ylabel('Percent, mean ± sample SD across eight simulations')
ax.set_ylim(0,85);ax.set_title('Present182 boundary diagnostic, fixed 1,000 bp tolerance')
ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(D/'boundary_rates_and_geometric_ceiling.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
labels=['<100','100–500','0.5–1k','1–5k','5–10k','10–50k','50–100k','0.1–0.5M','0.5–2M'];x=np.arange(9)
series=[('True uncensored','aligned','true_uncensored_only'),('Aligned predicted internal','aligned','predicted_internal_only'),
 ('Zero predicted internal','zero_concordance','predicted_internal_only')]
for i,(name,arm,key) in enumerate(series):
 d=models[arm]['pooled_distributions'][key];y=np.asarray(d['pooled_counts'])/d['count']
 ax.bar(x+(i-1)*.25,100*y,width=.24,label=name)
ax.set_xticks(x,labels);ax.set_xlabel('Predefined tract length bin (bp)');ax.set_ylabel('Percent of interior tracts, pooled counts')
ax.set_title('Predicted tracts are more fragmented than true ancestry tracts');ax.legend()
ax.grid(axis='y',alpha=.2);fig.savefig(D/'interior_tract_length_distributions.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained');labs=j['class_order']
for row,arm in enumerate(['aligned','zero_concordance']):
 m=models[arm]['pooled_transition_matrices'];true=np.asarray(m['true'],dtype=float);pred=np.asarray(m['predicted'],dtype=float)
 matrices=[true/true.sum(),pred/pred.sum(),np.divide(np.asarray(m['matched_true']),true,out=np.full_like(true,np.nan),where=true>0)]
 for col,(a,title) in enumerate(zip(matrices,['Share of true changes','Share of predicted changes','Positional recall by true pair'])):
  ax=axes[row,col];ax.imshow(np.ma.masked_invalid(a),vmin=0,vmax=1,cmap='Blues')
  for i in range(4):
   for k in range(4):ax.text(k,i,'—' if not np.isfinite(a[i,k]) or i==k else f'{100*a[i,k]:.1f}',
    ha='center',va='center',color='white' if a[i,k]>.65 else 'black')
  ax.set_xticks(range(4),labs);ax.set_yticks(range(4),labs);ax.set_title(title+' (%)',fontsize=10)
  ax.set_xlabel('To class')
  if col==0:ax.set_ylabel(names[arm]+'\nFrom class')
fig.suptitle('C4 transition structure on original scored support')
fig.savefig(D/'class_transition_matrices.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
valid=[i*4+k for i in range(4) for k in range(4) if i!=k]
tick=[labs[i//4]+'→'+labs[i%4] for i in valid]
for ax,arm in zip(axes,['aligned','zero_concordance']):
 c=np.asarray(models[arm]['pooled_matched_transition_16x16'],dtype=float)[np.ix_(valid,valid)]
 a=np.divide(c,c.sum(1)[:,None],out=np.full_like(c,np.nan),where=c.sum(1)[:,None]>0)
 im=ax.imshow(np.ma.masked_invalid(a),vmin=0,vmax=1,cmap='Blues')
 ax.set_xticks(range(12),tick,rotation=90,fontsize=8);ax.set_yticks(range(12),tick,fontsize=8)
 ax.set_title(names[arm]);ax.set_xlabel('Predicted transition');ax.set_ylabel('True transition, among positional matches')
fig.colorbar(im,ax=axes,shrink=.75,label='Fraction among positional matches of each true transition')
fig.suptitle('Class-agnostic matches can have the wrong biological transition (truth-row fractions)')
fig.savefig(D/'matched_transition_correspondence.png',dpi=160);plt.close(fig)
a=models['aligned'];v=a['mean_SD_across8simulations'];d=derived['aligned'];short=a['short_tract_parent_summaries']
fmt=lambda x:f"{x['mean']/1000:.2f}±{x['sample_sd']/1000:.2f}"
lines=['# H038 Present182 boundary-error diagnostic','',
 'The diagnostic completed on both frozen selected H038 arms and reproduced all 16 original accuracy/boundary-score cells. '
 'All prediction seals and native readout geometry were verified before targets. No predictions or labels were modified. '
 'Values with ± are sample SDs across eight internal VAL simulations, ddof1, from one fitted model per arm.','',
 '| Frozen H038 arm | Precision | Recall | F1 | Hold-edge oracle recall ceiling |',
 '|---|---:|---:|---:|---:|']
for arm,r in models.items():
 m=r['mean_SD_across8simulations'];lines.append('| '+names[arm]+' | '+' | '.join(pct(m[k]) for k in keys)+' |')
lines+=['',
 f"**The main gap is transition recognition and tract continuity, with a substantial geometric limit.** Aligned detects only {pct(v['recall'])} of true boundaries against a {pct(v['representable_boundary_recall_upper_bound'])} class-agnostic geometric oracle ceiling. "
 f"Of 4,008 true changes,1,094({100*d['geometric_unreachable_true_fraction']:.2f}%) have no allowed hold edge within1 kb. Only 4 more are lost to one-to-one candidate competition. "
 f"Of 3,289 observed misses,2,195({100*d['observed_misses_with_candidate_fraction']:.2f}%) still have a nearby candidate. The constant hold decoder limits possible recall, but it does not explain most observed misses.",'',
 f"**False short islands are important, and sub-kilobase spikes explain little.** Strict false internal A→B→A islands≤10 kb account for {pct(short['10000']['fraction_of_extra_boundaries_in_strict_false_islands'])} of extra boundaries, with deduplicated attribution. "
 f"At≤1 kb that fraction is only {pct(short['1000']['fraction_of_extra_boundaries_in_strict_false_islands'])}. "
 f"All predicted tracts≤10 kb contain {pct(short['10000']['fraction_of_total_error_query_bp'])} of the wrong query-bp. "
 'Short-island errors matter much more for event counts than for base-pair accuracy, and they are not the whole error budget.','',
 f"**Fragmentation extends beyond tiny islands.** Aligned predicts 6,729 changes versus 4,008 true changes({d['predicted_to_true_boundary_ratio']:.2f}×). "
 f"The across-parent median interior predicted tract is {fmt(a['pooled_distributions']['predicted_internal_only']['parent_median_bp_mean_SD'])}kb, "
 f"versus {fmt(a['pooled_distributions']['true_uncensored_only']['parent_median_bp_mean_SD'])}kb for uncensored true tracts. "
 'These are means±SD of per-parent medians, not pooled medians.','',
 f"**Changing tolerance or blaming matcher competition would miss the main issue.** Only 6 of 6,010 extra predicted boundaries are within1 kb of a true boundary that was already consumed by matching. "
 f"Among positional matches,{pct(a['positional_matched_same_transition_fraction_mean_SD'])} have the correct ordered C4 transition. "
 'The original F1 ignores transition identity and can credit a nearby wrong transition, so it is slightly optimistic about biological transition fidelity. No alternative F1 or tolerance was selected.','',
 '**Useful next scientific conditions:** use independent TRAIN simulations to learn a tract-continuity or class-transition model, with explicit tests of preserving real short ancestry tracts. '
 'Study marker-aware within-hold decoding to address the approximately 27% geometric deficit, alongside training objectives that improve transition recognition where an edge already exists. '
 'Evaluate both hypotheses prospectively with the fixed scorer and a separately preserved outer evaluation. '
 'Do not apply validation-chosen smoothing or tune island cutoffs on these data.','',
 'The zero-concordance arm has the same geometric ceiling, lower precision and more strict false≤10 kb islands. '
 'This is consistent with mutation concordance reducing fragmentation; it does not establish the cause of every false transition. '
 'Inferred-genealogy error and ancestry misclassification can also contribute.','',
 'The ceiling uses all scored-bin edges where native readout_hold changes, includes100bp snapping, and is a class-agnostic oracle. '
 'It does not fit an oracle predictor, impose the model’s transition constraints, or promise attainable recall. '
 'Endpoint-censored tracts, absent boundaries and undefined metrics follow the original scorer. '
 'Pooled event counts/histograms are labeled separately from equal-parent mean±SD. '
 'These H038 diagnostics do not assign boundary performance to H039 or H040 primary-only results.','',
 'Source/pins, handmade checks, all-seal preflight, per-parent compact JSON, CSV and the transition/tract plots are preserved beside this report.']
text='\n'.join(lines)+'\n';assert used()+len(text.encode())<10_000_000;(D/'biological_conclusions.md').write_text(text)
save('postprocess_completed.json',dict(summary_ref=ref(D/'summary.json'),report_ref=ref(D/'biological_conclusions.md'),
 plots=[ref(D/n) for n in ['boundary_rates_and_geometric_ceiling.png','interior_tract_length_distributions.png',
 'class_transition_matrices.png','matched_transition_correspondence.png']],output_bytes=used(),
 GPU_used=False,new_predictions=False,H039_boundary_claim=False))
print(json.dumps({'completed':True,'output_bytes':used(),'derived':derived},indent=2))

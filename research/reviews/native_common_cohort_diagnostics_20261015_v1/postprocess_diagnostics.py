"""Compact diagnostic comparison and plots, existing JSON receipts only."""
from pathlib import Path
import json,hashlib,csv,io,os
import numpy as np
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=Path(__file__).parent
CAP=10_000_000
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def load(r):
 p=ROOT/r['path'];assert sha(p)==r['sha256'];return json.loads(p.read_text())
def jval(x):
 if isinstance(x,(np.ndarray,np.generic)):return jval(x.tolist())
 if isinstance(x,dict):return {str(k):jval(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jval(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def used():return sum(p.stat().st_size for p in D.rglob('*') if p.is_file())
def save(name,x):
 b=(json.dumps(jval(x),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 assert used()+len(b)<CAP
 (D/name).write_bytes(b)
def moments(x):
 a=np.array(x,dtype=float)
 return dict(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)),n=8,ddof=1)
baseline=load(ref(D/'native_cohort_metrics_mean_SD.json'))
native=load(ref(D/'per_parent_metrics.json'))
ours_path=ROOT/'research/reviews/cohort_reporting_20261015_v1/our_cohort_metrics_mean_SD.json'
ours_ref=ref(ours_path);ours=load(ours_ref)
models=dict(ours['models'],**baseline['models'])
expected={'primary_present_day_182','secondary_ancient_504','all_queries_686',
 'age_stratum_Bronze_Age','age_stratum_BAA','age_stratum_Neo','age_stratum_Yam'}
for k,v in models.items():assert {r['cohort'] for r in v['rows']}==expected
old_ap_ref=ref(ROOT/'experiments/results/expanded_native_AP_H038_VAL_score_v1/summary.json')
old_ap=load(old_ap_ref)
old_am_ref=ref(ROOT/'research/reviews/expanded_native_argmix_H038_score_root_v1/readonly_CPU_result_v1.json')
old_am=load(old_am_ref)['scoring_preview_result']['result']
checks=[]
for model,old in [('native_AP',old_ap),('native_ARGMix',old_am)]:
 original={c['parent_id']:c for c in old['per_parent']}
 for cell in native[model]:
  p=original[cell['parent_id']]['groups']['primary_present_day_182']
  new=cell['original_FP32_primary']
  assert p['accuracy']==new['accuracy'] and p['nll']==new['nll']
  checks.append(dict(model=model,parent_id=cell['parent_id'],exact_FP32_accuracy_match=True,exact_FP32_NLL_match=True))
def stats_nullable(x):
 if any(v is None for v in x):
  return dict(mean=None,sample_sd=None,n=8,ddof=1,
    undefined_parent_count=sum(v is None for v in x),defined_parent_count=sum(v is not None for v in x))
 return moments(x)
paired=[]
our_diag_refs={}
for arm in ['aligned','zero_concordance']:
 r=ours['models'][arm]['source_ref'];j=load(r);our_diag_refs[arm]=r
 cells=j.get('per_parent',j.get('cells'))
 assert cells is not None
 cells={x['parent_id']:x for x in cells}
 for row in models[arm]['rows']:
  cohort=row['cohort'];groups=[x['groups'][cohort] for x in cells.values()]
  row['per_class_mean_SD']=[dict(class_0based=c,class_label=baseline['class_order'][c],
    **{k:stats_nullable([g['per_class'][c][k] for g in groups]) for k in ['precision','recall','f1','span_iou']})
    for c in range(4)]
 for base in native:
  for cohort in sorted(expected):
   deltas=[100*(cells[x['parent_id']]['groups'][cohort]['accuracy']-x['groups'][cohort]['accuracy']) for x in native[base]]
   paired.append(dict(our_arm=arm,baseline=base,cohort=cohort,
        accuracy_difference_percentage_points=moments(deltas),
        estimand='equal-parent paired differences over same eight internalVAL simulations'))
comparison=dict(schema='H038_native_same_support_comparison_v1',models=models,
 native_source_ref=ref(D/'native_cohort_metrics_mean_SD.json'),our_source_ref=ours_ref,
 our_diagnostic_refs=our_diag_refs,paired_accuracy_differences=paired,
 original_primary_checks=checks,old_AP_score_ref=old_ap_ref,old_ARGMix_score_ref=old_am_ref,
 SD_definition='ddof1 over eight existing VAL simulations. One fit per method. Not optimizer-seed uncertainty.',
 scope='development results using unchanged method-specific checkpoint selection; no outer evaluation',
 class_order=['Ana','CHG','WHG','EHG'],boundary_tolerance_bp=1000,
 comparison_to_published_paper_tables=False,no_retraining_or_new_predictions=True)
for model,report in models.items():
 for row in report['rows']:
  C=np.array(row['pooled_confusion_query_bp'],dtype=float)
  truth=C.sum(1);pred=C.sum(0);diag=np.diag(C)
  row['pooled_per_class']=[dict(class_0based=c,class_label=baseline['class_order'][c],
    precision=float(diag[c]/pred[c]) if pred[c] else None,
    recall=float(diag[c]/truth[c]) if truth[c] else None,
    f1=float(2*diag[c]/(truth[c]+pred[c])) if truth[c] else None)
    for c in range(4)]
save('same_support_comparison.json',comparison)
out=io.StringIO();fields=['model','cohort','class_label','precision_mean','precision_sampleSD',
 'recall_mean','recall_sampleSD','f1_mean','f1_sampleSD','span_iou_mean','span_iou_sampleSD']
writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
for model,report in models.items():
 for row in report['rows']:
  for x in row['per_class_mean_SD']:
   values=dict(model=model,cohort=row['cohort'],class_label=x['class_label'])
   for k in ['precision','recall','f1','span_iou']:
    values[k+'_mean']=x[k]['mean'];values[k+'_sampleSD']=x[k]['sample_sd']
   writer.writerow(values)
payload=out.getvalue().encode();assert used()+len(payload)<CAP
(D/'per_class_precision_recall_F1_mean_SD.csv').write_bytes(payload)
cohorts=['primary_present_day_182','age_stratum_Bronze_Age','age_stratum_BAA','age_stratum_Neo',
 'age_stratum_Yam','secondary_ancient_504','all_queries_686']
names={'aligned':'H038 aligned','zero_concordance':'H038 zero concordance',
 'native_AP':'Native AncestralPaths','native_ARGMix':'Current native ARGMix'}
short=['Present (182)','Bronze Age (162)','BAA (14)','Neo (302)','Yam (26)','Ancient (504)','All (686)']
order=['native_AP','native_ARGMix','zero_concordance','aligned']
rows={k:{r['cohort']:r for r in v['rows']} for k,v in models.items()}
fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
for a,metric,title in zip(axes,['accuracy','boundary_f1'],['Physical-span accuracy','Boundary F1 (1,000 bp tolerance)']):
 for i,k in enumerate(order):
  yy=[rows[k][c][metric]['mean'] for c in cohorts];ss=[rows[k][c][metric]['sample_sd'] for c in cohorts]
  valid=[j for j,x in enumerate(yy) if x is not None]
  a.errorbar([100*yy[j] for j in valid],np.array(valid)+(i-1.5)*.12,
   xerr=[100*ss[j] for j in valid],fmt='o',markersize=4,capsize=2,label=names[k],alpha=.9)
 a.set_yticks(range(7),short);a.invert_yaxis();a.set_xlabel('Percent, mean ± sample SD across eight VAL simulations')
 a.set_title(title);a.grid(axis='x',alpha=.2)
axes[0].legend(fontsize=8,loc='lower left')
axes[1].text(.02,.02,'Undefined cohort boundary F1 omitted.\nOne fitted model per method. Internal VAL only.',
 transform=axes[1].transAxes,fontsize=8)
fig.savefig(D/'accuracy_and_boundary_mean_SD.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,4,figsize=(14,4),layout='constrained')
labels=['Ana','CHG','WHG','EHG']
for a,k in zip(axes,order):
 C=np.array(rows[k]['primary_present_day_182']['pooled_confusion_truth_row_normalized'],dtype=float)
 im=a.imshow(C,vmin=0,vmax=1,cmap='Blues')
 for i in range(4):
  for j in range(4):a.text(j,i,f'{100*C[i,j]:.1f}',ha='center',va='center',fontsize=9,
    color='white' if C[i,j]>.65 else 'black')
 a.set_xticks(range(4),labels);a.set_yticks(range(4),labels);a.set_xlabel('Predicted class')
 a.set_title(names[k],fontsize=10)
axes[0].set_ylabel('True class')
fig.suptitle('Present-day query-bp confusion, pooled across eight VAL simulations (%)',fontsize=12)
fig.colorbar(im,ax=axes,shrink=.7,label='Truth-row fraction')
fig.savefig(D/'present_day_confusion_matrices.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,7,figsize=(19,6),layout='constrained')
for row,k in enumerate(['native_AP','native_ARGMix']):
 for col,c in enumerate(cohorts):
  a=axes[row,col];C=np.array(rows[k][c]['pooled_confusion_truth_row_normalized'],dtype=float)
  a.imshow(np.ma.masked_invalid(C),vmin=0,vmax=1,cmap='Blues')
  for i in range(4):
   for j in range(4):a.text(j,i,'—' if not np.isfinite(C[i,j]) else f'{100*C[i,j]:.1f}',ha='center',va='center',
    fontsize=7,color='white' if C[i,j]>.65 else 'black')
  a.set_xticks(range(4),labels,rotation=45,fontsize=8);a.set_yticks(range(4),labels,fontsize=8)
  if row==0:a.set_title(short[col],fontsize=9)
  if col==0:a.set_ylabel(names[k],fontsize=10)
fig.suptitle('Native baseline confusion: truth rows, predicted columns; pooled query-bp (%)',fontsize=12)
fig.savefig(D/'native_all_cohort_confusion_matrices.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for a,k in zip(axes,['native_AP','native_ARGMix']):
 a.plot([0,1],[0,1],color='gray',ls='--',lw=1)
 for c,label in zip(['primary_present_day_182','secondary_ancient_504'],['Present','Ancient']):
  bins=rows[k][c]['pooled_calibration_bins'];b=[x for x in bins if x['query_bp']>0]
  a.plot([x['mean_confidence'] for x in b],[x['accuracy'] for x in b],marker='o',ms=3,label=label)
 a.set_xlim(0,1);a.set_ylim(0,1);a.set_xlabel('Mean raw maximum probability')
 a.set_ylabel('Physical-span accuracy');a.set_title(names[k]);a.legend();a.grid(alpha=.2)
fig.suptitle('Raw probability calibration, fifteen bins; no fitted calibration')
fig.savefig(D/'native_reliability.png',dpi=150);plt.close(fig)
assert used()<CAP
lines=['# Native common-cohort diagnostics, 15 October 2026','',
 'The CPU computation completed for both original sealed native baselines on all eight internal VAL simulations. '
 'All AP C4/native6 and ARGMix raw seals passed before any new target access. '
 'The unchanged physical-span scorer was used for all seven original H038 cohorts. '
 'Sixteen separately recomputed FP32 primary accuracy/NLL cells exactly matched the prior baseline CPU receipts.','',
 'Mean ± sample SD below is over eight simulations (ddof1), with one fit per method. It is not optimizer-seed SD or the undefined ± in the ARGMix paper. '
 'Method-specific checkpoint selection was preserved. These are development results, and outer arrays were not read.','',
 '| Cohort | Native AP accuracy (%) | Current native ARGMix (%) | H038 aligned (%) | H038 zero concordance (%) |',
 '|---|---:|---:|---:|---:|']
for c,label in zip(cohorts,short):
 vals=[]
 for k in ['native_AP','native_ARGMix','aligned','zero_concordance']:
  x=rows[k][c]['accuracy'];vals.append(f"{100*x['mean']:.2f}±{100*x['sample_sd']:.2f}")
 lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','Present-day boundary F1 at the frozen 1,000 bp tolerance:','']
for k in ['native_AP','native_ARGMix','aligned','zero_concordance']:
 x=rows[k]['primary_present_day_182']['boundary_f1']
 lines.append(f"- {names[k]}: {100*x['mean']:.2f}±{100*x['sample_sd']:.2f}%.")
lines+=['',
 'High base-pair accuracy does not imply accurate tract boundaries. Present-day boundary F1 remains low across all four methods. '
 'Yam boundary F1 is undefined in simulations with no true boundaries, so its full-eight mean/SD is left undefined; absent truth classes have undefined class recall.','',
 'Artifacts: native_cohort_metrics_mean_SD.json, per-parent compact metrics, CSV, '
 'same_support_comparison.json with paired accuracy differences and exact original-primary checks, '
 'and four PNG plots of accuracy/boundaries, confusion and reliability. '
 'No bulky per-query/tract/grid arrays were written. No retraining, new predictions, checkpoint changes, calibration, probability floor, normalization, GPU, admission, install or Git action occurred.','',
 'Recipe limitations are documented in docs/argmix_published_benchmark_audit_v1.md. '
 'Bronze_Age is only an approximate correspondence to published Post-Neolithic, and PFP remains unresolved.']
text='\n'.join(lines)+'\n';b=text.encode();assert used()+len(b)<CAP;(D/'diagnostic_report.md').write_bytes(b)
save('postprocess_completed.json',dict(comparison_ref=ref(D/'same_support_comparison.json'),
 original_FP32_primary_exact_matches=len(checks),plots=[ref(D/n) for n in ['accuracy_and_boundary_mean_SD.png',
 'present_day_confusion_matrices.png','native_all_cohort_confusion_matrices.png','native_reliability.png']],
 output_bytes=used(),GPU_used=False,new_predictions=False))
print(json.dumps({'completed':True,'output_bytes':used(),
 'present':{k:dict(accuracy=rows[k]['primary_present_day_182']['accuracy'],
 boundary_f1=rows[k]['primary_present_day_182']['boundary_f1']) for k in order},
 'paired_present':[r for r in paired if r['cohort']=='primary_present_day_182']},indent=2))

"""Aggregate genuine completed H048 metric JSON only, without prediction or truth reads."""
from pathlib import Path
import json,hashlib,statistics,math,datetime
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H048_completed_comparison_root_20261015_v1'
OUTPUT='experiments/results/H048_GPU4_grid_reset_evaluate_seed101_v1'
PACKET='research/reviews/H048_GPU45_grid_root_source_v1/admitted_evaluate'
ARMS=('graph_to_transition','graph_to_own_confidence_fixed_distance_chain')
COHORTS=('primary_present_day_182','age_stratum_Bronze_Age','age_stratum_BAA','age_stratum_Neo','age_stratum_Yam','secondary_ancient_504','all_queries_686')
METRICS=('accuracy','nll','balanced_accuracy','macro_f1','boundary_f1','boundary_precision','boundary_recall','brier','calibration_ece','dosage_mse_per_diploid_bp_class','dosage_mae_per_diploid_bp_class')
def ref(s):
 p=ROOT/s
 return dict(path=s,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(s):return json.loads((ROOT/s).read_text())
def stats(v):
 a=[float(x) for x in v if x is not None]
 return dict(mean=statistics.mean(a) if a else None,sample_SD=statistics.stdev(a) if len(a)>1 and all(math.isfinite(x) for x in a) else None,defined_parent_count=len(a),undefined_parent_count=len(v)-len(a))
def metric(g,k):
 if k.startswith('boundary_'):return g['boundaries'][k.split('_',1)[1]]
 if k=='nll' and g.get('infinite_nll_query_bp',0):return math.inf
 return g[k]
def show(x,pct=False):
 if x['mean'] is None:return 'undefined'
 k=100 if pct else 1
 return f"{x['mean']*k:.4f} ± {x['sample_SD']*k:.4f}" if x['sample_SD'] is not None else f"{x['mean']*k:.4f} (SD undefined)"
def main():
 assert Path.cwd()==ROOT
 assert not (HERE/'summary.json').exists()
 wait=read(PACKET+'/actual_wait.json');launch=read(PACKET+'/launch.json');d=read(OUTPUT+'/completed.json')
 assert ref(PACKET+'/actual_wait.json')['sha256']=='529dc28d103f187d9677e750b69d85faa3a6e863d36905a4ddf7d5004a48112d'
 assert ref(OUTPUT+'/completed.json')['sha256']=='67d14126ba4cad40c1c3aef35bcfc4cdc45dd34b9254bbc18741f599fa6e083f'
 assert wait['returncode']==0 and wait['actual_direct_launcher_wait'] is True and wait['logging_error'] is None and wait['supervisor_error'] is None
 assert wait['launcher_pid']==launch['launcher_pid'] and wait['physical_gpu']==launch['physical_gpu']==4 and wait['phase']=='evaluate'
 assert wait['completed_ref']==ref(OUTPUT+'/completed.json') and launch['input_ref']==d['input_ref']
 assert d['schema']=='H048_paired_VAL_completion_v1' and d['status']=='completed_original8VAL_paired_development_selection'
 assert d['all96_RAM_seals_before_any_VAL_truth'] is True and d['outer_truth_read'] is False and d['optimization_performed'] is False
 tr=read(d['train_completion_ref']['path']);twpath='research/reviews/H048_GPU45_grid_root_source_v1/admitted_train/actual_wait.json';tw=read(twpath)
 assert ref(d['train_completion_ref']['path'])==d['train_completion_ref'] and tw['completed_ref']==d['train_completion_ref'] and tw['returncode']==0 and tw['actual_direct_launcher_wait'] is True
 assert tr['steps']==dict.fromkeys(ARMS,960) and tr['VAL_truth_read'] is False
 arms={};support={};values={}
 for a in ARMS:
  full=d['selected_full_diagnostics'][a];selected=d['selected'][a];cells=full['cells']
  assert full['selection_revised'] is False and len(cells)==8 and {c['parent_id'] for c in cells}=={f'ap_expanded_val_{i:03d}' for i in range(8)}
  at=selected['pass_index'];assert at in (5,10,15,20,25,30) and selected['checkpoint']==tr['checkpoint_groups'][str(at)][a]
  values[a]={};support[a]={}
  for c in cells:
   assert c['split']=='val' and set(c['groups'])==set(COHORTS)
   values[a][c['parent_id']]={};support[a][c['parent_id']]={}
   for cohort in COHORTS:
    g=c['groups'][cohort]
    values[a][c['parent_id']][cohort]={k:metric(g,k) for k in METRICS}
    support[a][c['parent_id']][cohort]={k:g[k] for k in ('query_count','query_bp','denominator_query_bp','original_parent_query_bp','coverage_original_bp0','physical_coverage_fraction','represented_bin_support_bp','represented_span_bp')}
  cs={c:{k:stats([v[c][k] for v in values[a].values()]) for k in METRICS} for c in COHORTS}
  arms[a]=dict(selected_pass=at,checkpoint_ref=selected['checkpoint'],cohorts=cs,completed_extra_updates=960,cumulative_model_lineage_updates=2880,selected_model_lineage_updates=1920+32*at)
 assert support[ARMS[0]]==support[ARMS[1]]
 primary='primary_present_day_182';delta=[values[ARMS[0]][p][primary]['accuracy']-values[ARMS[1]][p][primary]['accuracy'] for p in values[ARMS[0]]]
 utility=d['registered_practical_criteria'];assert abs(statistics.mean(delta)-utility['mean_Present182_accuracy_delta'])<1e-12
 assert utility['meets_primary_utility'] is False and utility['meets_secondary_boundary_utility'] is False
 out=dict(schema='H048_completed_metric_aggregation_root_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completion_ref=ref(OUTPUT+'/completed.json'),actual_launcher_wait_ref=ref(PACKET+'/actual_wait.json'),train_completion_ref=d['train_completion_ref'],train_wait_ref=ref(twpath),arms=arms,paired_effects=utility,per_parent_metric_values=values,dataset_support_by_parent_and_cohort=support[ARMS[0]],all96_products_sealed_before_truth=True,outer_test_accessed=False,statistical_unit='eight reused VAL simulations, equal-parent mean and sample SD ddof1, one optimizer seed',training=dict(dtype='float32',optimizer='AdamW',passes_per_arm=30,completed_updates_per_arm=960,parameters_per_arm=31933,initialization='same H042 pass30',independent_reset_optimizers=True),conclusion='H048 does not meet either registered utility rule and does not improve the best H045 development accuracy. Distinct operator novelty and independent-test performance remain unproved.')
 with (HERE/'summary.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 lines=['# H048 graph-conditioned reset chain','',out['conclusion'],'','Both arms completed30FP32 passes/960updates from the sameH042 initialization. All96predictions were sealed before VALtruth. SD is across8reusedVAL simulations, one optimizer seed. This is development selection, not independent confirmation.','','| Cohort | Arm | Accuracy % ± SD | NLL ± SD | Boundary F1 % ± SD |','|---|---|---:|---:|---:|']
 for c in COHORTS:
  for a in ARMS:
   q=arms[a]['cohorts'][c];lines.append('| '+c+' | '+a+' | '+show(q['accuracy'],True)+' | '+show(q['nll'])+' | '+show(q['boundary_f1'],True)+' |')
 lines+=['',f"Transition-minus-confidence accuracy: {utility['mean_Present182_accuracy_delta']*100:+.6f}pp, paired SD {utility['paired_accuracy_sample_SD_percentagepoints']:.6f}pp. Boundary F1 delta: {utility['mean_1kb_boundary_F1_delta']*100:+.6f}pp.",'','Transition selectedpass5, own-confidence selectedpass20. Full11metrics across all7cohorts, actual scoring support and source/wait refs are in [summary.json](summary.json). No probability, checkpoint or query-truth payload was read for aggregation.','']
 (HERE/'REPORT.md').open('x').write('\n'.join(lines))
 print(json.dumps({a:{k:arms[a]['cohorts'][primary][k] for k in ('accuracy','nll','boundary_f1')} for a in ARMS}),flush=True)
if __name__=='__main__':main()

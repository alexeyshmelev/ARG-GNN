from pathlib import Path
import json,hashlib,statistics,datetime
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
seed=303
source=ROOT/f'results/biological_candidate_pilot_v1_seed{seed}/receipt.json'
raw=source.read_bytes();receipt=json.loads(raw)
assert receipt['status']=='completed_three_conditions_one_seed' and receipt['seed']==seed
assert receipt['config_sha256']=='4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1'
assert receipt['observed_manifest_sha256']=='6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28'
assert receipt['prepared_manifest_sha256']=='3c0a3320fbc6dd9d421a2acfad398961c4912cdbc49d111faf02108ee4f8596d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
conditions={}
metrics=['accuracy','nll','brier','macro_f1','calibration_ece']
for name,model in receipt['models'].items():
 assert model['status']=='completed' and model['optimizer_fit_completed']
 assert model['optimizer_steps']>0 and model['selected_maximum_parameter_delta_from_initial']>0
 rows=[]
 for binding in model['metrics']:
  p=Path(binding['path']);assert ROOT in p.parents and sha(p)==binding['sha256']
  result=json.loads(p.read_text());assert result['seed']==seed and result['model']==name
  assert result['no_calibration_fit'] and not result['mixed_latent_outputs_labeled']
  groups={}
  for group,values in result['C4_exact_support_metrics'].items():
   groups[group]={key:values[key] for key in metrics}
   groups[group]['denominator_query_bp']=values['denominator_query_bp']
   groups[group]['infinite_nll_query_bp']=values['infinite_nll_query_bp']
  rows.append(dict(parent_id=result['parent_id'],metrics_source=str(p.relative_to(ROOT)),metrics_source_sha256=binding['sha256'],groups=groups))
 assert [x['parent_id'] for x in rows]==['bio_paths_pilot_val_000','bio_paths_pilot_val_001']
 means={group:{key:statistics.mean(row['groups'][group][key] for row in rows) for key in metrics} for group in rows[0]['groups']}
 conditions[name]=dict(parameters=model['parameters'],selected_epoch=model['best_epoch'],optimizer_steps=model['optimizer_steps'],selected_maximum_parameter_delta_from_initial=model['selected_maximum_parameter_delta_from_initial'],rows=rows,means=means)
summary=dict(kind='completed_biological_candidate_development_summary_v1',status='three_conditions_one_seed_completed',hypothesis_id='H020',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_receipt=dict(path=str(source.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest()),training_seeds=[seed],independent_VAL_parents=2,TEST_parents=0,aggregation='Unweighted mean of parent metrics. Optimizer seeds are not biological replicates.',model_or_probability_changes=False,conditions=conditions,new_model_superiority_established=False,distinct_novelty_established=False)
out=ROOT/f'experiments/results/biological_candidate_pilot_v1_seed{seed}_summary'
out.mkdir(parents=True,exist_ok=False)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'summary_sha256':sha(out/'summary.json'),'primary':{name:v['means']['primary_present_day_64'] for name,v in conditions.items()}}))

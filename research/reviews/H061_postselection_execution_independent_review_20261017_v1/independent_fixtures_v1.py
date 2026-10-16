"""Independent AST/source + isolated synthetic metadata only; no project imports."""
from pathlib import Path
import ast,copy,hashlib,json,math,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SRC='research/reviews/H061_postselection_execution_source_20261017_v1'
HERE='research/reviews/H061_postselection_execution_independent_review_20261017_v1'
SEEDS=(101,202,303);ARMS=('full','no_memory')
COMPARATORS=('last_value','seasonal_lag52','ridge_alpha1000_from_prior_pilot')
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def ref(p):
 path=ROOT/p;need(path.suffix in('.py','.json','.md','.patch'),'Source/metadata only, no scientific payload')
 return dict(path=p,sha256=sha(path.read_bytes()))
def read_file(p):return json.loads((ROOT/p).read_text())
def main():
 seals=read_file(HERE+'/assigned_seals.v1.json')
 for p,v in seals.items():need(ref(p)['sha256']==v,'Exact assigned seal '+p)
 manifest=read_file(SRC+'/source_manifest.v1.json')
 for p,v in manifest['source_sha256'].items():need(ref(p)['sha256']==v,'Source closure '+p)
 binding=read_file(SRC+'/scientific_binding.v1.json')
 need(ref(binding['scientific_review_ref']['path'])==binding['scientific_review_ref'],'Exact scientific review SHA')
 need(binding['scientific_review_ref']['sha256']=='8a8b43357f1f20e56f4381eebd15b148662ebe0bedf3a773eb3e03e037e0674f','Assigned scientific review')
 for p in (SRC+'/root_contract_v1.py',SRC+'/root_driver_v1.py',SRC+'/worker_entry_v1.py',SRC+'/scientific_runner_v1.py'):
  compile((ROOT/p).read_text(),p,'exec')
 contract_source=(ROOT/SRC/'root_contract_v1.py').read_text();runner_source=(ROOT/SRC/'scientific_runner_v1.py').read_text()
 ctree=ast.parse(contract_source);rtree=ast.parse(runner_source)
 defs={n.name:n for n in ctree.body if isinstance(n,ast.FunctionDef)}
 rdefs={n.name:n for n in rtree.body if isinstance(n,ast.FunctionDef)}
 storage={};lookups=[];rawcalls=[]
 def synthetic(path,value):
  storage[path]=copy.deepcopy(value);return dict(path=path,sha256=sha(encoded(value)))
 def read(r):lookups.append(r['path']);return copy.deepcopy(storage[r['path']])
 def raw(r):rawcalls.append(r['path']);return b'SYNTHETIC_NOT_A_WEIGHT_OR_DATA_PAYLOAD'
 cfg=read_file(binding['TRAIN_scientific_config_ref']['path']);cfg['execution_enabled']=True
 directory='experiments/results/SYNTHETIC_H061_TRAIN_NOT_ACTUAL'
 fits=[];products=[];metadata_refs=[]
 for seed in SEEDS:
  for arm in ARMS:
   cp=dict(path=directory+'/selected_seed%d_%s.pt'%(seed,arm),sha256=sha(('SYNTHETIC'+str(seed)+arm).encode()))
   history=[dict(epoch=i,updates=20*i,mae_cases=(2. if i==1 else 1.),forecast_weeks=78,per_county_mae=[1.]*20,per_week_mae=[1.]*78)for i in range(1,41)]
   row=dict(seed=seed,arm=arm,selected_epoch=2,selected_validation_mae_cases=1.,epochs_completed=40,optimizer_updates=800,
    parameters={'full':25378,'no_memory':24865}[arm],test_accessed=False,selected_checkpoint_ref=cp)
   meta=dict(row,history=history,scientific_config=cfg,source_protocol_ref=binding['protocol_ref'],registration_ref=binding['registration_ref'],
    state_kind='selected_cpu_model_state_only',optimizer_state_saved=False,TEST_release_granted=False,
    training_targets=list(range(52,365)),validation_targets=list(range(365,443)))
   mr=synthetic(directory+'/selection_seed%d_%s.json'%(seed,arm),meta);metadata_refs.append(mr)
   row['selected_metadata_ref']=mr;fits.append(row)
   products.append(dict(seed=seed,arm=arm,kind='selected_checkpoint',**cp,selected_metadata_ref=mr))
 scaler=dict(fit_rows=[0,365],mean=[1.]*20,scale=[1.]*20)
 barrier=dict(status='six_TRAIN_selected_states_and_configurations_frozen',source_protocol_ref=binding['protocol_ref'],
  registration_ref=binding['registration_ref'],completion_source_ref=binding['completion_ref'],generic_core_ref=binding['generic_core_ref'],
  archive_ref=binding['archive_ref'],data_source_manifest_ref=binding['data_source_manifest_ref'],county_order=binding['counties'],TEST_accessed=False,TEST_release_granted=False,scoring_authorized=False,
  scientific_config=cfg,selected_products=products,TRAIN_only_scaler=scaler)
 br=synthetic(directory+'/six_selection_barrier.json',barrier)
 result=dict(status='completed_TreeLoom_Chickenpox_TRAIN_six_selections',seeds=list(SEEDS),arms=list(ARMS),total_optimizer_updates=4800,
  selected_count=6,validation_prediction_batches=1200,county_count=20,history_weeks=52,separate_ROOT_TEST_release_required=True,TEST_accessed=False,TEST_scored=False,test_predictions_sealed=0,
  first_scheduled_update_timed_inline=True,standalone_numerical_probe=False,selection_barrier_ref=br,fits=fits)
 rr=synthetic(directory+'/completed_TRAIN.json',result);inputref=dict(path='SYNTHETIC_INPUT_NOT_ACTUAL.json',sha256='a'*64)
 envelope=dict(status='completed_TreeLoom_Chickenpox_CPU_TRAIN_selection',root_driver_ref=binding['lifecycle_driver_ref'],input_ref=inputref,
  TEST_deferred=True,scoring_authorized=False,result_ref=rr)
 er=synthetic(directory+'/completed.json',envelope)
 wait=dict(returncode=0,actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None,completed_ref=er,input_ref=inputref)
 wr=synthetic('SYNTHETIC_actual_wait_NOT_ACTUAL.json',wait)
 accept=dict(schema='H061_TRAIN_outer_acceptance_v1',root_authorizer='root_agent',accepted=True,outer_terminal_consumed=True,outer_terminal_exit_code=0,
  source_manifest_ref=binding['TRAIN_source_manifest_ref'],source_review_ref=binding['TRAIN_wrapper_review_ref'],actual_wait_ref=wr,
  output_directory=directory,completed_ref=er,selection_barrier_ref=br)
 ar=synthetic('SYNTHETIC_TRAIN_acceptance_NOT_AUTHORITY.json',accept);q=dict(train_acceptance_ref=ar)
 storage[binding['TRAIN_scientific_config_ref']['path']]=read_file(binding['TRAIN_scientific_config_ref']['path'])
 env=dict(require=need,binding=lambda:binding,read=read,raw=raw,math=math,SEEDS=SEEDS,ARMS=ARMS,
  CAPS=read_file(SRC+'/caps.inactive.v1.json'))
 exec(compile(ast.Module(body=[defs[n]for n in('_in_output','train_selections')],type_ignores=[]),'<synthetic_selection_metadata_only>','exec'),env)
 selections=env['train_selections'](q)
 need(selections['products']==products and not rawcalls,'No payload read during metadata validation')
 env['train_selections'](q,authenticate_payloads=True)
 need(rawcalls==[p['path']for p in products],'All six fake hashes authenticated before scientific stage')
 rawcalls.clear();baseline=copy.deepcopy(storage);negative=[]
 def reject(f,label):
  try:f()
  except (ValueError,KeyError,TypeError,IndexError):negative.append(label);return
  raise AssertionError('Accepted invalid synthetic metadata '+label)
 changes=[('TRAIN_outer_unconsumed',ar,'outer_terminal_consumed',False),('TRAIN_outer_nonzero',ar,'outer_terminal_exit_code',1),
  ('TRAIN_wait_nonzero',wr,'returncode',1),('TRAIN_undrained',wr,'all_owned_processes_and_groups_drained',False),
  ('TRAIN_4799_updates',rr,'total_optimizer_updates',4799),('TRAIN_1199_validation_batches',rr,'validation_prediction_batches',1199),
  ('TRAIN_TEST_accessed',rr,'TEST_accessed',True),('TRAIN_incomplete_history',metadata_refs[0],'history',baseline[metadata_refs[0]['path']]['history'][:-1]),
  ('TRAIN_later_tie_selection',metadata_refs[0],'selected_epoch',3),('TRAIN_missing_fit',rr,'fits',fits[:-1]),
  ('TRAIN_duplicate_fit',rr,'fits',fits[:-1]+[fits[0]]),('TRAIN_bad_scaler',br,'TRAIN_only_scaler',dict(fit_rows=[0,443],mean=[1.]*20,scale=[1.]*20))]
 for label,r,k,v in changes:
  storage.clear();storage.update(copy.deepcopy(baseline));storage[r['path']][k]=v
  reject(lambda:env['train_selections'](q),label)
 storage.clear();storage.update(copy.deepcopy(baseline))
 # Read/validate fifteen synthetic product METADATA, no numerical arrays involved.
 freeze=dict(path='SYNTHETIC_frozen_barrier.json',sha256='b'*64);predbar=dict(predictions=[],chronology_predictions=[],comparator_predictions=[])
 for field,kind in(('predictions','test_prediction'),('chronology_predictions','chronology_prediction')):
  for p in products:
   identity=dict(seed=p['seed'],arm=p['arm']);path='SYNTHETIC_'+field+'_%d_%s.json'%(p['seed'],p['arm'])
   value=dict(kind=kind,**identity,freeze_barrier_ref=freeze,target_indices=list(range(443,522)),county_order=binding['counties'],
    TEST_scored=False,prediction_clipping='none',selected_checkpoint_ref={k:p[k]for k in('path','sha256')},prediction_cases='SYNTHETIC_NO_ARRAY')
   r=synthetic(path,value);predbar[field].append(dict(kind=kind,**identity,**r))
 for name in COMPARATORS:
  value=dict(kind='comparator_prediction',comparator=name,freeze_barrier_ref=freeze,target_indices=list(range(443,522)),
   county_order=binding['counties'],TEST_scored=False,prediction_clipping='none',prediction_cases='SYNTHETIC_NO_ARRAY')
  r=synthetic('SYNTHETIC_control_'+name+'.json',value);predbar['comparator_predictions'].append(dict(kind='comparator_prediction',comparator=name,**r))
 counter=[]
 def array(value,np,shape):need(value=='SYNTHETIC_NO_ARRAY'and shape==(79,20),'Full shape assertion called');counter.append(shape);return 'SYNTHETIC_RETURN_NOT_ARRAY'
 c=types.SimpleNamespace(require=need,read=read,COMPARATORS=COMPARATORS,SEEDS=SEEDS,ARMS=ARMS)
 renv=dict(c=c,_array=array)
 exec(compile(ast.Module(body=[rdefs['_load_predictions']],type_ignores=[]),'<synthetic_15product_metadata_only>','exec'),renv)
 values=renv['_load_predictions'](predbar,freeze,selections,None,binding)
 need([len(v)for v in values]==[6,6,3]and len(counter)==15,'All15complete same-support products before return')
 for field in predbar:
  invalid=copy.deepcopy(predbar);invalid[field].pop()
  reject(lambda invalid=invalid:renv['_load_predictions'](invalid,freeze,selections,None,binding),'missing_'+field)
 invalid=copy.deepcopy(predbar);invalid['chronology_predictions'][-1]=invalid['chronology_predictions'][0]
 reject(lambda:renv['_load_predictions'](invalid,freeze,selections,None,binding),'duplicate_chronology_product')
 p=predbar['predictions'][0]['path'];original=copy.deepcopy(storage[p]);storage[p]['freeze_barrier_ref']={'wrong':True}
 reject(lambda:renv['_load_predictions'](predbar,freeze,selections,None,binding),'foreign_comparator_freeze');storage[p]=original
 storage[p]['selected_checkpoint_ref']={'wrong':True}
 reject(lambda:renv['_load_predictions'](predbar,freeze,selections,None,binding),'foreign_selected_state');storage[p]=original
 storage[p]['target_indices']=list(range(443,521))
 reject(lambda:renv['_load_predictions'](predbar,freeze,selections,None,binding),'reduced_TEST_support');storage[p]=original
 storage[p]['prediction_clipping']='zero'
 reject(lambda:renv['_load_predictions'](predbar,freeze,selections,None,binding),'prediction_clipping');storage[p]=original
 def calls(node):return [ast.unparse(n.func)for n in ast.walk(node)if isinstance(n,ast.Call)]
 score=calls(rdefs['_score']);need(score.index('_load_predictions')<score.index('completion.load_test_v1')<score.index('completion.score_test_v1'),'15products before TEST decode/scorer')
 predict=calls(rdefs['_predict']);need(predict.index('_ridge')<predict.index('completion.load_test_v1')and not any('metrics' in x or 'score_test' in x for x in predict),'Frozen comparators before TEST; no prediction-stage metrics')
 need(not any(x.endswith(('.backward','.AdamW','.SGD','.manual_seed','.fit_development_pair_v1'))for n in rdefs.values()for x in calls(n)),'No core fit/optimizer/newseed')
 return dict(schema='H061_independent_source_synthetic_metadata_checks_v1',status='PASS_SOURCE_AND_SYNTHETIC_METADATA_ONLY',assigned_seals=seals,
  manifest_source_metadata_pins_authenticated=len(manifest['source_sha256']),synthetic_six_full_histories_validated=True,
  synthetic_15product_gate_validated=True,negative_metadata_fixtures=negative,phase_order_and_no_core_fit_AST_pass=True,
  actual_guard_payload_numerical_or_job_operations=0)
if __name__=='__main__':
 value=main();out=json.dumps(value,sort_keys=True,indent=2)+'\n'
 with (ROOT/HERE/'independent_fixtures.result.v1.json').open('x')as f:f.write(out)
 print(out)

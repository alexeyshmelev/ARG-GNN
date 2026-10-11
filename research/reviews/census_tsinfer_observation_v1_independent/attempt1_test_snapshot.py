"""Independent handmade CPU checks. All phase authorities/readers/budgets are explicit doubles."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch,tskit
from arggnn import census_tsinfer_observation_v1 as b
from arggnn import census_native_qualification_runner_v1 as numerical
ROOT=b.ROOT;DIR=ROOT/'research/reviews/census_tsinfer_observation_v1_independent'
C=b.load_config_v1();checks=[];details={}
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,call):
 try:call()
 except (ValueError,RuntimeError,KeyError,TypeError,FileNotFoundError,AssertionError):checks.append(name);return
 raise AssertionError('unexpected acceptance: '+name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def forbidden(*args,**kwargs):raise AssertionError('No model or CUDA operation allowed in independent handmade review')
torch.set_num_threads(1)
for name in ('is_available','device_count','current_device','init'):setattr(torch.cuda,name,forbidden)
numerical._model=forbidden

author=json.loads((ROOT/'research/reviews/census_tsinfer_observation_v1_author/receipt.json').read_text())
for path,h in author['artifact_sha256'].items():check('author_artifact_preserved_'+path,sha(ROOT/path)==h)
for path,h in C['source_sha256'].items():check('scientific_dependency_pin_'+path,sha(ROOT/path)==h)
reader=b.VerifiedObservationReaderV1()
helpers=b.metadata_inputs_v1(C,reader)
check('genuine_four_parent_metadata_lineage_join',len(helpers)==4 and len(reader.evidence)==14)
details['genuine_metadata_read']={'count':len(reader.evidence),'files':reader.evidence,'arrays_read':False}

def homemade(group_time=350.):
 t=tskit.TableCollection(2_000_000.)
 cohorts=['present_day']*64+['Bronze_Age']*16+['BAA']*8+['Neo']*16+['Yam']*16
 roles=['query']*120+['mixed_reference_only']*64+['pure_source_reference']*96
 cls=[None]*184+[i for i in range(4) for _ in range(24)]
 ages=[{'present_day':0.,'Bronze_Age':135.,'BAA':155.,'Neo':200.,'Yam':170.}[c] for c in cohorts]+[150.]*64+[176.]*24+[473.]*24+[200.]*24+[300.]*24
 cohorts+=['UNLABELED_CONTEXT']*64+[C['classes'][i] for i in range(4) for _ in range(24)]
 nodes=[]
 for i,age in enumerate(ages):
  t.nodes.add_row(time=1000.)
  nodes.append(t.nodes.add_row(time=age,flags=tskit.NODE_IS_SAMPLE))
 for graph in range(16):
  root=t.nodes.add_row(time=900.);ancestor=t.nodes.add_row(time=group_time)
  left,right=graph*125000.,(graph+1)*125000.
  t.edges.add_row(left,right,root,ancestor)
  for i,node in enumerate(nodes):t.edges.add_row(left,right,ancestor if i<184 else root,node)
 t.sort();ts=t.tree_sequence()
 original=[];derived=[];mapping=[]
 for i,(node,role,age,cl,cohort) in enumerate(zip(nodes,roles,ages,cls,cohorts)):
  r={'sample_index':i,'sample_id':f'independent_subject_{i//2}','haplotype':i%2,
   'observed_node_id':99000+i,'observed_cohort':cohort,'age_generations':age,'role':role,'observed_reference_class':cl}
  original.append(r);d=dict(r);d['source_observed_node_id']=d.pop('observed_node_id')
  d['observed_node_id']=d['inferred_tree_node_id']=node;derived.append(d)
  mapping.append({'sample_index':i,'inferred_tree_node_id':node})
 joined=b.role_join_v1(ts,original,derived,mapping,C)
 return ts,joined,original,derived,mapping

ts,roles,original,derived,mapping=homemade()
positions=np.arange(50.,2_000_000.,100.,dtype=np.float64);coverage=[0.,2_000_000.]
bindings={k:{'path':str((DIR/'handmade'/k).relative_to(ROOT)),'sha256':'a'*64}
 for k in ('native_observed_ts','serialized_whole_observed','observed_roster','observed_grid','producer_receipt')}
lineage={'parent_id':'independent_train0','split':'train','namespace':'handmade_source_not_real_ARG'}
cert,fresh=b.certify_native_tile_v1(ts,roles,positions,coverage,0,bindings,lineage,C)
check('full280_noncontiguous_native_IDs_and_slots_preserved',len(roles)==280 and [r.actual_id for r in roles]==list(range(280)) and roles[0].native_sample_node!=roles[0].actual_id)
check('Q184_R96_Q120_exact',len(cert.native_certificate.bank.query_ids)==184 and len(cert.native_certificate.bank.reference_ids)==96 and int(cert.native_certificate.bank.partitions[0].scored_query_mask.sum())==120)
check('census350_equal_MRCA_one_group',len(cert.native_certificate.bank.partitions[0].group_members)==1)
older,older_roles,*_=homemade(350.0000001)
older_cert,older_raw=b.certify_native_tile_v1(older,older_roles,positions,coverage,0,bindings,lineage,C)
check('census350_above_MRCA_184_groups',len(older_cert.native_certificate.bank.partitions[0].group_members)==184)
check('direct_float64_cut_differs_when_encoded_x48_identical',torch.equal(fresh['sequence']['x'],older_raw['sequence']['x']) and cert.certificate_sha256!=older_cert.certificate_sha256)
check('sampled_leaf_proxy_only_and_context_marker_zero',all((row[:4].sum()==(1 if node in {r.native_sample_node for r in roles if r.role=='pure_source_reference'} else 0)) and row[4]==(1 if node in {r.native_sample_node for r in roles if r.role=='query'} else 0) for node,row in zip(fresh['sequence']['node_ids'].tolist(),fresh['sequence']['x'])))
check('persistent_IDs_joins_not_feature_identity',fresh['sequence']['node_ids'].unique().numel()==281 and all(n==-1 or n in set(ts.samples()) for n in fresh['sequence']['node_ids'].tolist()))
check('original_2Mb_physical_normalization_and_F47',fresh['sequence']['window_bounds'].tolist()==[0.,.125] and cert.native_certificate.provenance['feature_columns']==tuple(list(range(7))+list(range(8,48))))
check('original_right_boundary_mapping',fresh['sequence']['target_graph'][1249].item()==0 and fresh['sequence']['target_graph'][1250].item()==1)
for field in ('query_node_ids','positions','intervals','target_graph','node_ids','node_graph','clades','window_bounds'):
 bad=deepcopy(fresh);v=bad['sequence'][field];v.reshape(-1)[0]+=1
 reject('cache_exact_field_replay_'+field,lambda bad=bad:b.certify_native_tile_v1(ts,roles,positions,coverage,0,bindings,lineage,C,cached_raw=bad))
bad=replace(cert,native_certificate=replace(cert.native_certificate,certificate_sha256='0'*64))
reject('outer_rejects_stale_inner_certificate',lambda:b.validate_tsinfer_certificate_v1(bad,cert.certificate_sha256,C))
bad=replace(cert,lineage=dict(cert.lineage,split='val'))
reject('outer_rejects_changed_parent_split',lambda:b.validate_tsinfer_certificate_v1(bad,cert.certificate_sha256,C))

# Exact admission gates use scalar fake metadata and fake runtime byte providers only.
now=datetime.now(timezone.utc);created=now.isoformat();expires=(now+timedelta(hours=1)).isoformat()
source_closure=dict(C['source_sha256'],**{b.SOURCE:sha(ROOT/b.SOURCE)})
def gate_values(phase='inspect'):
 cfgref={'path':b.CONFIG,'sha256':sha(ROOT/b.CONFIG)}
 refs={'config':cfgref,**{name:{'path':str((DIR/(name+'.json')).relative_to(ROOT)),'sha256':'b'*64} for name in b.AUTH_REFS-{'config'}}}
 a={'schema':'census_tsinfer_observation_admission_v1','status':'active','authorizer':'root_agent','job_id':'handmade_gates','phase':phase,
  'created_at_utc':created,'expires_at_utc':expires,'refs':refs,'source_sha256':source_closure,'observed_artifact_sha256':C['observed_artifact_sha256'],
  'output_directory':C['inspection_output_prefix']+'handmade_gates','inspection_manifest':None,'expected_certificates':{},
  'derived_artifact_sha256':{},'initial_state_sha256_by_dtype':{},'group_permutation':[],'query_truth_authorized':False,'fit_authorized':False}
 v={name:{} for name in b.AUTH_REFS};v['config']=C
 v['review']={'schema':'census_tsinfer_observation_independent_review_v1','review_role':'independent_reviewer','verdict':'PASS','fixture_mode':False,'actual_admission':False,'source_sha256':source_closure[b.SOURCE],'config_sha256':cfgref['sha256']}
 v['registration']={'schema':'census_tsinfer_observation_root_registration_v1','authorizer':'root_agent','status':'registered','condition':C['condition'],'hypothesis_id':'HANDMADE_ONLY','phase':phase,'job_id':a['job_id'],'query_truth_authorized':False,'fit_authorized':False}
 v['storage']={'schema':'census_tsinfer_observation_storage_v1','status':'active','job_id':a['job_id'],'output_directory':a['output_directory'],
  'reserved_new_bytes':20000000000,'used_project_bytes':1000000000,'concurrent_reserve_bytes':20000000000,'project_limit_bytes':1000000000000,'created_at_utc':created,'expires_at_utc':expires,
  'snapshot_ref':{'path':str((DIR/'snapshot.json').relative_to(ROOT)),'sha256':'c'*64}}
 v['snapshot']={'schema':'census_native_root_storage_snapshot_v1','status':'complete','returncode':0,'project_root':str(ROOT),'used_project_bytes':v['storage']['used_project_bytes'],'created_at_utc':created}
 v['launcher']={'schema':'census_tsinfer_observation_CPU_launcher_v1','job_id':a['job_id'],'python':'.runtime/env/bin/python','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','GPU_admitted':False}
 v['runtime']={'schema':'census_tsinfer_observation_CPU_runtime_v1','runtime_file_sha256':{p:'d'*64 for p in C['runtime_files']},'identity':numerical.runtime_identity_v1()}
 return a,v
class GateReader:
 def __init__(self,a,v):self.a=a;self.v=v;self.evidence={}
 def record(self,r):return r
 def read(self,r,**kw):return b'handmade provider; never actual runtime/array bytes'
 def decode_json(self,r):
  if r['path']=='handmade_admission':return self.a
  for key in b.AUTH_REFS:
   if r==self.a['refs'][key]:return self.v[key]
  if r==self.v['storage']['snapshot_ref']:return self.v['snapshot']
  raise AssertionError('unexpected decode in gate fixture')
class Budget:
 def check(self,**kw):return {'handmade_only':True}
a,v=gate_values()
with patch.object(b,'SupplementalBudgetV1',return_value=Budget()):
 check('handmade_valid_inspect_gate_accepts_only_fixture_authority_provider',b.admit_phase_v1({'path':'handmade_admission','sha256':'e'*64},'inspect',GateReader(a,v))[0]==a)
 for name,alter in [
  ('inactive',lambda a,v:a.update(status='inactive')),
  ('target_access',lambda a,v:a.update(query_truth_authorized=True)),
  ('fit_access',lambda a,v:a.update(fit_authorized=True)),
  ('expired',lambda a,v:a.update(expires_at_utc=(now-timedelta(seconds=1)).isoformat())),
  ('wrong_config_review',lambda a,v:v['review'].update(config_sha256='0'*64)),
  ('wrong_source_review',lambda a,v:v['review'].update(source_sha256='0'*64)),
  ('author_role',lambda a,v:v['review'].update(review_role='author')),
  ('fixture_review',lambda a,v:v['review'].update(fixture_mode=True)),
  ('wrong_lineage_closure',lambda a,v:a['observed_artifact_sha256'].update({'unbound': '0'*64})),
  ('registration_wrong_phase',lambda a,v:v['registration'].update(phase='numerical')),
  ('storage_snapshot_stale',lambda a,v:v['snapshot'].update(created_at_utc=(now-timedelta(hours=2)).isoformat())),
  ('runtime_missing_library',lambda a,v:v['runtime']['runtime_file_sha256'].pop(C['runtime_files'][-1])),
  ('runtime_wrong_identity',lambda a,v:v['runtime'].update(identity={'device':'cpu'})),
  ('inspect_grants_numerics',lambda a,v:a.update(expected_certificates={'anything':'0'*64}))]:
  aa,vv=gate_values();alter(aa,vv)
  reject('exact_admission_reject_'+name,lambda aa=aa,vv=vv:b.admit_phase_v1({'path':'handmade_admission','sha256':'e'*64},'inspect',GateReader(aa,vv)))
 for name,field,value in [('no_freeze','inspection_manifest',None),('missing32','expected_certificates',{}),('missing_weights','initial_state_sha256_by_dtype',{}),('bad_permutation','group_permutation',list(range(183))+[182])]:
  aa,vv=gate_values('numerical');aa.update(inspection_manifest={'path':'fake_freeze','sha256':'a'*64},expected_certificates={str(i):'a'*64 for i in range(32)},initial_state_sha256_by_dtype={'float64':'a'*64,'float32':'b'*64},group_permutation=list(range(184)))
  aa[field]=value
  reject('numerical_admission_reject_'+name,lambda aa=aa,vv=vv:b.admit_phase_v1({'path':'handmade_admission','sha256':'e'*64},'numerical',GateReader(aa,vv)))

# Run 32 genuine CPU extraction/certification cells from handmade trees. No real authority.
fc=deepcopy(C);fc['inspection_output_prefix']=str((DIR/'handmade_outputs').relative_to(ROOT))+'/'
parents=[{'parent_id':'handmade_train0','split':'train'},{'parent_id':'handmade_train1','split':'train'},{'parent_id':'handmade_val0','split':'val'},{'parent_id':'handmade_val1','split':'val'}]
hh=[(dict(p),dict(p)) for p in parents]
def loaded(h,p,c,r,tmp):
 lineage={'parent_id':p['parent_id'],'split':p['split'],'namespace':'handmade:'+p['parent_id'],
  'native_tree_ref':bindings['native_observed_ts'],'source_sha256':source_closure[b.SOURCE],
  'config_ref':{'path':b.CONFIG,'sha256':sha(ROOT/b.CONFIG)}}
 refs={'native':bindings['native_observed_ts'],'derived':bindings['observed_roster'],'grid':bindings['observed_grid']}
 return ts,roles,positions,0.,coverage,refs,lineage
pa={'job_id':'inspect32','output_directory':fc['inspection_output_prefix']+'inspect32','source_sha256':source_closure,'observed_artifact_sha256':{},'expires_at_utc':expires}
pr=b.VerifiedObservationReaderV1()
with patch.object(b,'admit_phase_v1',return_value=(pa,fc,pr,Budget())),patch.object(b,'metadata_inputs_v1',return_value=hh),patch.object(b,'_load_parent_v1',side_effect=loaded):
 result,result_ref=b.run_phase_v1({'path':'handmade_only','sha256':'f'*64},'inspect')
check('actual_handmade_32cell_inspect_all_cells_no_models',result['status']=='inspection_complete_pending_root_freeze' and len(result['cells'])==32 and not result['failed_cells'])
check('actual_handmade_64_materializations_bound',sum(len(list((DIR/'handmade_outputs'/'inspect32'/p['parent_id']).glob('*'))) for p in parents)==64)
check('inspect_does_not_claim_numerical_fit_or_predictive_qualification',all(c['status']=='inspected_pending_root_freeze' and 'qualification' not in c for c in result['cells']) and not result['fit_performed'])
check('prespecified_numerical_cells_selected_TRAIN_only',all(c['split']=='train' for c in result['prespecified_TRAIN_numerical_cells'].values()))
details['handmade_inspect32']={'receipt':result_ref,'cells':32,'materializations':64,'models':0,'real_inputs_or_authority':False}
closure={r['path']:r['sha256'] for c in result['cells'] for r in (c['raw_ref'],c['producer_ref'])}
expected={f"{c['parent_id']}:{c['tile_id']}":c['certificate_sha256'] for c in result['cells']}
na=dict(pa,job_id='incomplete_numerical',output_directory=fc['inspection_output_prefix']+'incomplete_numerical',inspection_manifest=result_ref,
 expected_certificates=expected,derived_artifact_sha256=closure,initial_state_sha256_by_dtype={'float64':'a'*64,'float32':'b'*64},group_permutation=list(range(184)))
with patch.object(b,'admit_phase_v1',return_value=(na,fc,b.VerifiedObservationReaderV1(),Budget())),patch.object(b,'metadata_inputs_v1',return_value=hh),patch.object(b,'_load_parent_v1',side_effect=loaded),patch.object(numerical,'native_oracle_checks_v1',return_value={'handmade_stub':True}),patch.object(numerical,'numerical_cell_v1',return_value={'status':'qualified','performed_controls':[],'checks':[]}):
 nr,nref=b.run_phase_v1({'path':'handmade_only','sha256':'f'*64},'numerical')
check('consequential_numerical_gate_rejects_missing_controls_and_dtypes',nr['status']=='failed_partial_preserved' and len(nr['failed_cells'])==1 and 'five-control' in nr['cells'][0]['error'])
check('numerical_gate_retains_failure_receipt',b.VerifiedObservationReaderV1().decode_json(nref)['status']=='failed_partial_preserved')
details['incomplete_numerical_gate']={'receipt':nref,'actual_numerical_models':False}
report={'schema':'census_tsinfer_independent_handmade_checks_v1','status':'passed','count':len(checks),'checks':checks,'details':details,
 'source':ref(ROOT/b.SOURCE),'config':ref(ROOT/b.CONFIG),'test_source':ref(__file__),'actual_arrays_models_GPU_or_live_ledger':False}
(DIR/'checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':'passed','count':len(checks),'source':report['source'],'config':report['config']},indent=2))

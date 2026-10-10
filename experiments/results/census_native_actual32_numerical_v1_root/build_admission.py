#!/usr/bin/env python3
"""Root freeze and separate target-free numerical admission after successful actual inspection."""
import datetime as dt,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'experiments/results/census_native_actual32_numerical_v1_root'
RESULT='experiments/results/census_native_actual32_numerical_v1_cpu';JOB='census-native-numerical-v1'
PRIOR='experiments/results/census_native_actual32_inspect_v1_root/admission.json'
EVIDENCE='experiments/results/census_native_actual32_inspect_v1_cpu/evidence.json'
def sha(path):
 h=hashlib.sha256()
 with (ROOT/path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(path):return {'path':str(Path(path).relative_to(ROOT) if Path(path).is_absolute() else path),'sha256':sha(path)}
def load(path):return json.loads((ROOT/path).read_bytes())
def save(name,value):
 p=OUT/name
 with p.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 return ref(p)
assert sha(PRIOR)=='cf4b33810045372badc0cf14ec614c801047831cd01f4f3dd85a5c7f73b4f633'
assert sha(EVIDENCE)=='2a7b01fc3476b5492df65f9b7ae1e0b759f5e88a915131ce8fd8ba2e42427430'
old=load(PRIOR);e=load(EVIDENCE);c=load(old['refs']['runner_config']['path'])
exit_receipt=load('experiments/results/census_native_actual32_inspect_v1_root/exit.json')
assert exit_receipt['exit_code']==0 and exit_receipt['watchdog_timeout'] is False
assert e['status']=='inspection_complete_pending_root_freeze' and e['phase']=='inspect'
assert e['failed_cells']==[] and len(e['cells'])==32 and len(set(e['performed_cell_coverage']))==32
assert e['source_sha256']==old['source_sha256'] and e['observed_artifact_sha256']==old['observed_artifact_sha256']
assert not any(e[k] for k in ['biological_targets_opened','fit_performed','GPU_used','geometry_adapted','fixture_mode'])
certs={f"{r['parent_id']}:{r['tile_id']}":r['certificate_sha256'] for r in e['cells']}
expected_inventory={f'{p}:{t}' for p in c['parents'] for t in range(8)}
assert set(certs)==expected_inventory
prepared=load(old['refs']['prepared_manifest']['path'])
counts={f"{p['parent_id']}:{t['tile_id']}":len(t['original_tree_ids']) for p in prepared['parents'] for t in p['tiles']}
assert all(r['status']=='inspected_pending_root_freeze' and counts[f"{r['parent_id']}:{r['tile_id']}"]==r['num_trees'] for r in e['cells'])
for path,h in old['source_sha256'].items():assert sha(path)==h
for name,r in old['refs'].items():
 if name not in ['storage','launcher']:assert sha(r['path'])==r['sha256']
summary=[]
for r in e['cells']:
 st=r['structural']
 summary.append({k:r[k] for k in ['parent_id','split','tile_id','num_trees','certificate_sha256']}|{'sample_count':st['sample_count'],'older_R_proxy_count':st['older_R_proxy_count'],'cross_query_exchange_possible_fraction':st['cross_query_exchange_possible_fraction'],'singleton_Q_fraction':st['physical_span_weighted_singleton_Q_fraction'],'maximum_retained_height':st['maximum_retained_height'],'maximum_retained_depth':st['maximum_retained_depth'],'limited_mechanism_opportunity':st['limited_mechanism_opportunity'],'event_collision_counts':r['event_collision_counts']})
freeze_ref=save('root_inspection_freeze.json',{'schema':'root_external_census_inspection_freeze_v1','inspection_ref':ref(EVIDENCE),'exit_ref':ref('experiments/results/census_native_actual32_inspect_v1_root/exit.json'),'expected_certificates':certs,'expected_cells':32,'complete_coverage':True,'source_and_original_count_lineage_rechecked':True,'model_or_label_outcomes_used_for_selection':False,'fixed_TRAIN_extrema':e['prespecified_TRAIN_numerical_cells'],'cells':summary,'scope':'Observation joins and opportunity only. No predictive or memory-performance claim.','created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
import torch
torch.set_num_threads(1)
from arggnn.census_native_qualification_runner_v1 import _model
from arggnn.census_biological_bank_v1 import fingerprint_sha256_v1
states={dtype:fingerprint_sha256_v1(_model(c,dtype).state_dict()) for dtype in c['dtypes']}
with torch.random.fork_rng(devices=[]):
 torch.manual_seed(41);permutation=torch.randperm(184).tolist()
save('initial_conditions_freeze.json',{'state_sha256_by_dtype':states,'group_permutation':permutation,'seed_model':c['seed'],'seed_permutation':41,'biological_targets_opened':False,'fixed_before_numerical_outcomes':True})
observation_path='experiments/results/root_storage_observation_v20261010T2230/receipt.json'
observation=load(observation_path);now=dt.datetime.now(dt.timezone.utc)
assert observation['exit_code']==0 and (now-dt.datetime.fromisoformat(observation['recorded_utc'])).total_seconds()<3600
used=observation['used_apparent_project_bytes']
snapshot_ref=save('storage_snapshot.json',{'schema':'census_native_root_storage_snapshot_v1','status':'complete','returncode':0,'project_root':str(ROOT),'used_project_bytes':used,'created_at_utc':observation['recorded_utc'],'measurement_ref':ref(observation_path),'measurement_is_non_atomic_traversal':True,'persistent_baseline_rebased':False})
from arggnn.gpu_pool_v1 import StorageLedger,snapshot_record
historic=snapshot_record(ROOT,str(ROOT/'experiments/results/biological_shared_pool_storage_admission_v1/snapshot.json'),'0826fcedefa40dd261999c4916b85ae11f28628e5cee9046b2e4f6bf868fa10d')
def available():
 v=os.statvfs(ROOT);return v.f_bavail*v.f_frsize
ledger=StorageLedger(ROOT,historic,130000000000,450000000000,available)
account=ledger.job('census_native_v1',{'job':JOB,'storage_bytes':1000000000,'output':str(ROOT/RESULT)})
now=dt.datetime.now(dt.timezone.utc);created=now.isoformat();expiry=(now+dt.timedelta(hours=23)).isoformat()
storage_ref=save('storage.json',{'schema':'census_native_storage_reservation_v1','status':'active','job_id':JOB,'used_project_bytes':used,'reserved_new_bytes':1000000000,'concurrent_reserve_bytes':max(0,account['projected_bytes']-used-1000000000),'snapshot_ref':snapshot_ref,'project_limit_bytes':500000000000,'output_directory':RESULT,'created_at_utc':created,'expires_at_utc':expiry,'persistent_accounting':account,'persistent_ledger_ref':ref('.runtime/gpup/storage_reservations.json'),'persistent_baseline_rebased':False,'root_reservation_key':'census_native_v1/'+JOB,'accounting_note':'No persistent reservation released. Fresh apparent observation reconciled to unchanged conservative persistent projection.'})
launcher_ref=save('launcher.json',{'schema':'census_native_CPU_launcher_v1','job_id':JOB,'python':'.runtime/env/bin/python','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','GPU_admitted':False})
a={**old,'job_id':JOB,'phase':'numerical','created_at_utc':created,'expires_at_utc':expiry,'refs':{**old['refs'],'storage':storage_ref,'launcher':launcher_ref},'output_directory':RESULT,'expected_certificates':certs,'inspection_manifest':ref(EVIDENCE),'initial_state_sha256_by_dtype':states,'group_permutation':permutation}
admission_ref=save('admission.json',a)
save('root_prelaunch.json',{'admission':admission_ref,'external_root_freeze':freeze_ref,'storage':account,'watchdog_timeout_seconds':81000,'distinct_inspection_and_numerical_job_ids':old['job_id']!=JOB,'biological_targets_authorized':False,'fitting_authorized':False,'GPU_requested':False,'actual_paper_data':False})
print(json.dumps({'admission':admission_ref,'storage':account,'performed_inspection_cells':len(e['cells']),'prespecified_TRAIN_extrema':e['prespecified_TRAIN_numerical_cells']},indent=2),flush=True)

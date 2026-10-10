#!/usr/bin/env python3
"""Root-owned metadata/runtime/resource admission for first actual H021 inspection."""
import datetime as dt,hashlib,json,os
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'experiments/results/census_native_actual32_inspect_v1_root'
RESULT='experiments/results/census_native_actual32_inspect_v1_cpu'
JOB='census-native-inspect-v1'
SOURCE='src/arggnn/census_native_qualification_runner_v1.py'
CONFIG='configs/census_native_qualification_runner_v1.json'
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
assert sha(SOURCE)=='a63c9481c934247fabe077aed8dabe02bb1cc831f7bbfe441f5974b397346140'
assert sha(CONFIG)=='b71c4d7608eb8365761131b9842636bd5ce8ee81eb9f151c6f39f51f828f5bab'
c=load(CONFIG)
source={p:sha(p) for p in c['required_source_paths']}
assert source==c['source_sha256']
source[SOURCE]=sha(SOURCE)
review='research/reviews/census_native_qualification_runner_v1_independent/runner_review.json'
assert sha(review)=='3e95a8e65bb2906d10ff5ad1b8c9e578df522deadd5c39b59b519a796b677e93'
native='data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json'
prepared='data/biological_candidate_pilot_v1/manifest.json'
assert sha(native)=='6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28'
assert sha(prepared)=='3c0a3320fbc6dd9d421a2acfad398961c4912cdbc49d111faf02108ee4f8596d'
native_data=load(native);prepared_data=load(prepared)
artifacts={}
for p in native_data['replicates']:
 for r in [p['inferred_tree'],p['observed_sample_manifest'],p['original_observed_sample_manifest'],p['completion_provenance'],p['coordinate_contract']['fixed_grid']]:
  path=str(Path(r['path']).relative_to(ROOT) if Path(r['path']).is_absolute() else r['path'])
  if path in artifacts:assert artifacts[path]==r['sha256']
  artifacts[path]=r['sha256']
for b in prepared_data['parents']:
 for t in b['tiles']:
  r=t['whole_observed_graph']
  path=str(Path(r['path']).relative_to(ROOT) if Path(r['path']).is_absolute() else r['path'])
  if path in artifacts:assert artifacts[path]==r['sha256']
  artifacts[path]=r['sha256']
decisions=load('research/experiment_decisions.json')
matches=[(h['id'],r) for h in decisions['decisions'] for r in h.get('registered_runs',[]) if r.get('id')=='census_native_actual32_inspect_v1']
assert len(matches)==1 and matches[0][0]=='H021'
save('existing_experiment_registration.json',{'hypothesis_id':matches[0][0],'registered_run':matches[0][1],'decision_ref':ref('research/experiment_decisions.json'),'actual_paper_data':False,'biological_target_access':False})
import torch,numpy,tskit,sys
torch.set_num_threads(1)
runtime={'schema':'census_native_CPU_runtime_v1','identity':{'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':str(numpy.__version__),'tskit':str(tskit.__version__),'device':'cpu','threads':1,'cuda_visible_devices':''},'runtime_file_sha256':{p:sha(p) for p in c['runtime_files']}}
runtime_ref=save('runtime.json',runtime)
feature={'schema':'census_original_feature_lineage_v1','prepared_manifest_sha256':sha(prepared),'producer_source_sha256':c['source_sha256']['src/arggnn/biological_candidate.py'],'reference_adapter_sha256':c['source_sha256']['src/arggnn/reference_encoder.py'],'feature_columns':list(range(7))+list(range(8,48)),'scaler':'identity_no_fitted_scaler','original_TRAIN_parent_ids':[p for p in c['parents'] if '_train_' in p],'labels_in_input':False}
feature_ref=save('feature_lineage.json',feature)
observation_path='experiments/results/root_storage_observation_v20261010T2230/receipt.json'
observation=load(observation_path)
now=dt.datetime.now(dt.timezone.utc)
observed_time=dt.datetime.fromisoformat(observation['recorded_utc'])
assert observation['exit_code']==0 and (now-observed_time).total_seconds()<3600
used=observation['used_apparent_project_bytes']
snapshot_ref=save('storage_snapshot.json',{'schema':'census_native_root_storage_snapshot_v1','status':'complete','returncode':0,'project_root':str(ROOT),'used_project_bytes':used,'created_at_utc':observation['recorded_utc'],'measurement_ref':ref(observation_path),'measurement_is_non_atomic_traversal':True,'persistent_baseline_rebased':False})
from arggnn.gpu_pool_v1 import StorageLedger,snapshot_record
old_snapshot=snapshot_record(ROOT,str(ROOT/'experiments/results/biological_shared_pool_storage_admission_v1/snapshot.json'),'0826fcedefa40dd261999c4916b85ae11f28628e5cee9046b2e4f6bf868fa10d')
def available():
 v=os.statvfs(ROOT);return v.f_bavail*v.f_frsize
ledger=StorageLedger(ROOT,old_snapshot,130000000000,450000000000,available)
account=ledger.job('census_native_v1',{'job':JOB,'storage_bytes':1000000000,'output':str(ROOT/RESULT)})
now=dt.datetime.now(dt.timezone.utc)
created=now.isoformat();expires=(now+dt.timedelta(hours=4)).isoformat()
concurrent=max(0,account['projected_bytes']-used-1000000000)
storage_ref=save('storage.json',{'schema':'census_native_storage_reservation_v1','status':'active','job_id':JOB,'used_project_bytes':used,'reserved_new_bytes':1000000000,'concurrent_reserve_bytes':concurrent,'snapshot_ref':snapshot_ref,'project_limit_bytes':500000000000,'output_directory':RESULT,'created_at_utc':created,'expires_at_utc':expires,'persistent_accounting':account,'persistent_ledger_ref':ref('.runtime/gpup/storage_reservations.json'),'root_reservation_key':'census_native_v1/'+JOB,'persistent_baseline_rebased':False,'accounting_note':'Concurrent allowance reconciles fresh apparent observation to unchanged conservative persistent projected total. No reservation released or baseline rebased.'})
launcher_ref=save('launcher.json',{'schema':'census_native_CPU_launcher_v1','job_id':JOB,'python':'.runtime/env/bin/python','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','GPU_admitted':False})
a={'schema':'census_native_qualification_admission_v1','status':'active','job_id':JOB,'phase':'inspect','created_at_utc':created,'expires_at_utc':expires,'refs':{'runner_config':ref(CONFIG),'runtime':runtime_ref,'storage':storage_ref,'launcher':launcher_ref,'native_manifest':ref(native),'prepared_manifest':ref(prepared),'feature_lineage':feature_ref,'runner_review':ref(review)},'source_sha256':source,'observed_artifact_sha256':artifacts,'output_directory':RESULT,'expected_certificates':{},'inspection_manifest':None,'initial_state_sha256_by_dtype':{},'group_permutation':[]}
admission_ref=save('admission.json',a)
save('root_prelaunch.json',{'admission':admission_ref,'persistent_storage':account,'observed_file_closure_count':len(artifacts),'phase':'inspect','watchdog_timeout_seconds':13800,'model_forward_authorized':False,'biological_targets_authorized':False,'fitting_authorized':False,'GPU_requested':False,'actual_paper_data':False})
print(json.dumps({'admission':admission_ref,'persistent_storage':account,'observed_file_closure_count':len(artifacts)},indent=2),flush=True)

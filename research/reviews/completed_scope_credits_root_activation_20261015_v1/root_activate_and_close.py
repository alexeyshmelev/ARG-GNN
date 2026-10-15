"""Root activation and exact completed-scope credits. No scientific execution."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
HERE=ROOT/'research/reviews/completed_scope_credits_root_activation_20261015_v1'
SOURCE=str(Path(__file__).relative_to(ROOT))
BASE='research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py'
NEW='research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py'
NEW_SHA='ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef'
REPAIR='research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json'
CORE='483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'
LAUNCH='a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'
def utc():return datetime.now(timezone.utc).isoformat()
def ref(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 assert ref(r['path'])==r
 return json.loads((ROOT/r['path']).read_text())
def save(name,value):
 p=HERE/name
 with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
 return ref(p)
def load(path,name):
 sp=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def overlay():
 assert ref(NEW)['sha256']==NEW_SHA
 m=load(NEW,'root_completed_credit_v3');m.configure_v2(ref(REPAIR),policy_ref=ref(HERE/'activation_policy.json'));return m
def activate():
 assert ref(NEW)['sha256']==NEW_SHA
 base=load(BASE,'root_activation_frozen_base');base.configure_v2(ref(REPAIR))
 with base.budget.v4.mutex(ROOT):_,_,view,bindings,_=base._state_locked(CORE,LAUNCH,fresh=True)
 m=load(NEW,'root_activation_new_overlay')
 assert not (ROOT/m.JOURNAL).exists()
 p=m.activation_policy_draft_v1(repair_authority_ref=ref(REPAIR),base_pins=base._pins(view),base_projected_bytes=view['projected_bytes'],implementation_ref=ref('scripts/project_budget_v9.py'),launcher_ref=ref('scripts/launch_reserved_gpu_v6.py'))
 p['active']=True;pr=save('activation_policy.json',p)
 save('root_prospective_admission_revocation.json',dict(root_authorizer='root_agent',recorded_utc=utc(),source_ref=ref(SOURCE),policy_ref=pr,all_future_resource_admissions_use=ref(NEW),all_future_capacity_v2_accounting_admissions_revoked=True,old_active_H040_GPU45_workers_preserved_fully_charged=True,closed_scope_all_original_resume_write_capabilities_will_be_revoked_per_certificate=True,all_original_files_and_journals_preserved=True,actual_GPU_ids=[4,5],resource_change_authorizes_scientific_execution=False))
 m.configure_v2(ref(REPAIR),policy_ref=pr);v=m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=LAUNCH)
 save('activation_receipt.json',dict(status='activated_for_future_resource_admissions',policy_ref=pr,source_ref=ref(NEW),base_projected_bytes=p['base_projected_bytes'],projected_bytes=v['projected_bytes'],base_pins=p['base_pins'],no_historical_changes=True,no_completed_scope_credit_applied=True,actual_scope_inventories=0,recorded_utc=utc()))
 agents=ROOT/'AGENTS.md';s=agents.read_text();old='New resource admissions use research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py configured with the bound root marker-repair authority.';new='New resource admissions use research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py with research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json and the bound root marker-repair authority. All future accounting admissions through capacity-v2 are revoked. Existing live GPU 4/5 workers retain their original guards and full scope charges. Completed-scope credits require exact original completion/direct-wait evidence and targeted root process/revocation certificates, without deletion or historical journal changes.';assert old in s;agents.write_text(s.replace(old,new))
 print(json.dumps(dict(activated=True,policy_ref=pr,projected_bytes=v['projected_bytes'])),flush=True)
def absent(pid):
 try:os.kill(pid,0)
 except ProcessLookupError:return dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc=utc())
 raise ValueError('Targeted PID remains present or reused: '+str(pid))
def close(which):
 m=overlay()
 with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,LAUNCH,fresh=True)
 if which=='native_argmix_preparation':
  key='H038/H038_native_argmix_prepare_v3';folder='research/reviews/expanded_full_panel_v1_root/native_argmix_preparation_v3';storage_ref=ref(folder+'/input.json');locator=['storage'];launch_ref=ref(folder+'/launch.json');wait_ref=ref(folder+'/root_wait.json');launch_pid=['pid'];wait_pid=['pid'];wait_identity=dict(returncode=0,status='complete',actual_direct_process_wait=True)
  completion=read(ref('data/expanded_native_argmix_prepared_v3/completed.json'));assert completion['TRAIN_parents']==32 and completion['VAL_parents']==8 and completion['outer_parents']==0 and len(completion['parent_products'])==40 and len(completion['commands'])==41
  assert all(c['returncode']==0 and c['worker_group_fully_exited'] is True for c in completion['commands'])
  pids=sorted({read(launch_ref)['pid'],*[pid for c in completion['commands'] for pid in c['observed_owned_pids']]})
  identity={k:completion[k] for k in ['schema','completed_utc','TRAIN_parents','VAL_parents','outer_parents','input_ref','source_ref']}
 elif which=='native_ancestralpaths':
  key='H038/H038-expanded-native-ancestralpaths-v2';folder='research/reviews/expanded_full_panel_v1_root/native_launch_helpers_v1/admitted_v1/ancestralpaths';storage_ref=ref(folder+'/launch_storage.json');locator=[];launch_ref=ref(folder+'/launch.json');wait_ref=ref(folder+'/root_wait.json');launch_pid=['pid'];wait_pid=['supervisor_pid'];wait_identity=dict(returncode=0,success=True,actual_child_waited=True)
  completion=read(ref('data/expanded_native_ancestralpaths_v2/completed.json'));assert completion['TRAIN_count']==32 and completion['VAL_count']==8 and completion['epochs']==5 and len(completion['VAL_prediction_seals'])==8 and completion['outer_included'] is False
  supervisor=read(ref('data/expanded_native_ancestralpaths_v2/supervisor/launcher_exit.json'));assert supervisor['returncode']==0 and supervisor['worker_waited'] is True and supervisor['worker_pid']==completion['worker_pid']
  pids=sorted({read(launch_ref)['pid'],completion['worker_pid']})
  identity={k:completion[k] for k in ['schema','completed_utc','TRAIN_count','VAL_count','epochs','outer_included','input_ref','source_ref']}
 elif which=='American_full_simulation':
  key='H041/American-C3-full30mb-20261015';folder='research/reviews/American_C3_root_execution_20261015_v1/admitted_full30mb_v2';storage_ref=ref(folder+'/launch_admission.json');locator=['storage'];launch_ref=ref(folder+'/launch.json');wait_ref=ref(folder+'/actual_wait.json');launch_pid=['pid'];wait_pid=['pid'];wait_identity=dict(returncode=0,actual_direct_worker_wait=True)
  completion=read(ref('experiments/results/American_C3_full30mb_20261015_v1/completed.json'));pids=[read(launch_ref)['pid']];identity=dict(status=completion['status'],parent_id=completion['parent_id'],observation_ref=completion['observation_ref'])
 else:raise ValueError(which)
 own=state[0][key];assert key not in state[1]
 completed_ref=ref(Path(own['output'])/'completed.json');assert read(completed_ref)==completion
 launch=read(launch_ref);wait=read(wait_ref);pid=launch[launch_pid[0]];assert wait[wait_pid[0]]==pid and wait['returncode']==0
 if 'completed_ref' in wait:assert wait['completed_ref']==completed_ref
 observations=[absent(p) for p in pids]
 obs=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=key,reservation_sha256=m.budget.digest(own),storage_ref=storage_ref,launch_ref=launch_ref,direct_wait_ref=wait_ref,completed_ref=completed_ref,launcher_pid=pid,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=observations,source_ref=ref(SOURCE),recorded_utc=utc(),revocation_applies_to_all_prior_input_config_storage_source_launch_and_resume_capabilities=True)
 observation_ref=save(which+'_root_observation.json',obs)
 request=dict(key=key,reservation_sha256=m.budget.digest(own),storage_ref=storage_ref,storage_locator=locator,launch_ref=launch_ref,launch_pid_locator=launch_pid,launch_identity={launch_pid[0]:pid},direct_wait_ref=wait_ref,wait_pid_locator=wait_pid,wait_exit_locator=['returncode'],wait_identity=wait_identity,completed_ref=completed_ref,completed_identity=identity,observation_ref=observation_ref)
 operation=m.operation_draft_v1('close_completed',request,expected_before=m._pins(state[2]),bindings=state[3]);operation.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=18)).isoformat())
 auth=save(which+'_close_authority.json',operation);receipt=m.apply_close_completed_v1(auth,expected_source_sha256=CORE,expected_launcher_sha256=LAUNCH);save(which+'_close_receipt.json',receipt)
 print(json.dumps(dict(closed=which,retained_bytes=receipt['retained_bytes'],credit_bytes=receipt['credit_bytes'],projected_bytes=receipt['projected_bytes'],files_deleted=0)),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');p.add_argument('--close',choices=['native_argmix_preparation','native_ancestralpaths','American_full_simulation']);a=p.parse_args()
 if a.activate:activate()
 elif a.close:close(a.close)
 else:print('Root source only')
if __name__=='__main__':main()

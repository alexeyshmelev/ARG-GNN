"""ROOT one-shot ordinary close of the genuinely completed full VAL001 conversion."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
P=ROOT/'research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v1'
P.mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def ref(rel):
 p=ROOT/rel
 return {'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(r):
 raw=(ROOT/r['path']).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==r['sha256']
 return json.loads(raw)
def save(name,value):
 p=P/name
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 return ref(str(p.relative_to(ROOT)))
selector={'path':'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json','sha256':'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}
sel=load(selector);load(sel['source_ref'])
sp=importlib.util.spec_from_file_location('_ROOT_FLARE_current_close',ROOT/sel['source_ref']['path'])
g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
g.configure_with_renewal_v1(sel['repair_authority_ref'],policy_ref=sel['policy_ref'],selector_ref=selector)
inp_ref={'path':'research/reviews/FLARE_remaining_ROOT_actual_VAL001_convert_3h_setupv4_20261017_v1/input.json','sha256':'95ca4f5bff8bd6723a0cfcfeb233e7517db10d24b443e8b70c78d5b1979492bd'}
wait_ref={'path':'research/reviews/FLARE_remaining_ROOT_actual_VAL001_convert_3h_setupv4_20261017_v1/worker_aux/actual_wait.json','sha256':'51613247e73166fbbff8e4c1e0f70130924f4e73be9ba86a0ad1ed631bf50391'}
completed_ref={'path':'data/FLARE_original_remaining_VAL001_convert_3h_setupv4_20261017_v1/completed.json','sha256':'3d8b6c3711bda0bd0224d57bd3210aabdcebdb707a26051b3b21bdd149b33366'}
terminal_ref={'path':'research/reviews/research_execution_ROOT_20261017_v22/FLARE_VAL001_conversion_actual_outer_consumed.root_v1.json','sha256':'d72ee403a0570188859bab721b0ee8d7de3bac0219d141f3b398821e4fd51e7a'}
inp=load(inp_ref);wait=load(wait_ref);completed=load(completed_ref);terminal=load(terminal_ref)
assert terminal['actual_terminal_consumed'] and terminal['actual_exit_code']==0 and terminal['tool_session_id']==91032
assert terminal['actual_wait_ref']==wait_ref and terminal['completed_ref']==completed_ref
assert wait['returncode']==0 and wait['actual_direct_worker_wait'] and wait['all_owned_processes_and_groups_drained'] and not wait['cleanup_errors'] and wait['operational_failure'] is None
assert wait['input_ref']==inp_ref and wait['completed_ref']==completed_ref
storage_ref=inp['refs']['storage'];storage=load(storage_ref);own=storage['reservation']
launch_ref=ref('research/reviews/FLARE_remaining_ROOT_actual_VAL001_convert_3h_setupv4_20261017_v1/worker_aux/launch.json')
launch=load(launch_ref)
spawn=load(ref('research/reviews/FLARE_remaining_ROOT_preparation_VAL001_convert_3h_setupv4_20261017_v1/actual_supervisor_spawn.root_v2.json'))
assert launch['pid']==wait['pid'] and launch['input_ref']==inp_ref and spawn['pid']==wait['supervisor_pid']
pids=sorted(set(wait['known_owned_pids']+[wait['supervisor_pid'],launch['pid'],spawn['pid']]))
rows=[g._absent(pid) for pid in pids]
obs=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=own['key'],reservation_sha256=g.budget.digest(own),storage_ref=storage_ref,launch_ref=launch_ref,direct_wait_ref=wait_ref,completed_ref=completed_ref,launcher_pid=launch['pid'],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=rows,actual_outer_terminal_ref=terminal_ref,original_failures_and_clocks_preserved=True)
obs_ref=save('targeted_completion_observation.root_active_v1.json',obs)
request=dict(key=own['key'],reservation_sha256=g.budget.digest(own),storage_ref=storage_ref,storage_locator=[],launch_ref=launch_ref,launch_pid_locator=['pid'],launch_identity=dict(pid=launch['pid'],input_ref=inp_ref),direct_wait_ref=wait_ref,wait_pid_locator=['pid'],wait_exit_locator=['returncode'],wait_identity=dict(pid=launch['pid'],input_ref=inp_ref,returncode=0,all_owned_processes_and_groups_drained=True,completed_ref=completed_ref),completed_ref=completed_ref,completed_identity=dict(status='completed_C3_FLARE_observation_conversion',parent_id='american_c3_val_001'),observation_ref=obs_ref)
source_sha='483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'
launcher_sha='a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'
with g.budget.v4.mutex(ROOT):
 state=g._state_locked(source_sha,launcher_sha,fresh=True)
 assert state[0][own['key']]==own and own['key'] not in state[1]
 auth=g.operation_draft_v1('close_completed',request,expected_before=g._pins(state[2]),bindings=state[3])
 auth.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
 auth_ref=save('ordinary_close_authority.root_active_v1.json',auth)
 print(json.dumps({'phase':'authority_prepared','authority_ref':auth_ref,'utc':now()}),flush=True)
receipt=g.apply_close_completed_v1(auth_ref,expected_source_sha256=source_sha,expected_launcher_sha256=launcher_sha)
receipt_ref=save('ordinary_close_receipt.root_v1.json',receipt)
print(json.dumps({'phase':'ordinary_close_completed','receipt_ref':receipt_ref,'retained_bytes':receipt['retained_bytes'],'credit_bytes':receipt['credit_bytes'],'projected_bytes':receipt['projected_bytes'],'no_scientific_score':True}),flush=True)

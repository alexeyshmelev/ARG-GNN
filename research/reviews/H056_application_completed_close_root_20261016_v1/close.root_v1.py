"""Close only the genuinely completed H056 v2 application. Root explicit execution."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json,os,sys,traceback
sys.excepthook=lambda t,v,b:traceback.print_exception(t,v,b,file=sys.stdout)
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H056_application_completed_close_root_20261016_v1'
ORIGIN=ROOT/'research/reviews/H056_three_bank_application_actual_root_20261016_v2'
OUT=ROOT/'experiments/results/H056_frozen_H045_Danish_three_chromosome_CPU_v2'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
def ref(p):
 p=Path(p)
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(name,value):
 with (HERE/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 return ref(HERE/name)
def read(p):return json.loads(Path(p).read_text())
def load(path,sha,name):
 p=ROOT/path
 assert ref(p)['sha256']==sha
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
assert sys.argv[1:]==['--execute']
tool=read(ORIGIN/'actual_tool_terminal.root_v1.json')
launch=read(ORIGIN/'actual_launch.root_v1.json')
wait=read(ORIGIN/'actual_wait.root_v1.json')
done=read(OUT/'completed.json')
storage=read(ORIGIN/'storage.json')
assert tool['actual_tool_call_id']==42438 and tool['tool_chunk_id']=='1ba168'
assert tool['actual_tool_terminal_consumed'] is True and tool['actual_tool_exit_code']==0
assert tool['worker_final_original_scope_guard_succeeded'] is True
assert tool['actual_direct_wait_ref']==ref(ORIGIN/'actual_wait.root_v1.json')
assert tool['scientific_completion_ref']==ref(OUT/'completed.json')
assert launch['launcher_identity']['pid']==3257287
assert wait['launcher_pid']==3257287 and wait['launcher_start_ticks']==launch['launcher_identity']['start_ticks']
assert wait['actual_direct_wait_observed'] is True and wait['returncode']==0
assert wait['actual_launcher_PID_absent'] is True and wait['actual_owned_group_absent'] is True
assert done['status']=='completed_descriptive_fixed_cohort_H045_application_no_accuracy'
assert done['optimization_steps']==0 and done['accuracy_available'] is False
inner=read(ROOT/tool['actual_inner_wait_ref']['path']);outer=read(ROOT/tool['actual_inner_outer_wait_ref']['path'])
assert ref(ROOT/tool['actual_inner_wait_ref']['path'])==tool['actual_inner_wait_ref']
assert ref(ROOT/tool['actual_inner_outer_wait_ref']['path'])==tool['actual_inner_outer_wait_ref']
assert inner['returncode']==0 and outer['returncode']==0
assert outer['observed_owned_groups_ESRCH'] is True
pids=[3257285,3257286,3257287,3257364]
assert sorted(x['pid'] for x in tool['observations'])==pids
observations=[]
for pid in pids:
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Original exact owned PID still present')
 observations.append(dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc=datetime.now(timezone.utc).isoformat()))
for pgid in [3257286,3257287,3257364]:
 try:os.killpg(pgid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Original owned process group still present')
renewal=load('research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py','491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468','_H056_app_close_renewal')
renewal.install_renewal_v1(dict(path='research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json',sha256='66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'))
guard=load('research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py','fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56','_H056_app_close_original_guard')
guard.configure_v2(dict(path='research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json',sha256='f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'),policy_ref=dict(path='research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json',sha256='eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'))
own=storage['reservation']
request=dict(key=own['key'],reservation_sha256=guard.budget.digest(own),storage_ref=ref(ORIGIN/'storage.json'),storage_locator=[],
 launch_ref=ref(ORIGIN/'actual_launch.root_v1.json'),launch_pid_locator=['launcher_identity','pid'],
 launch_identity={k:launch[k] for k in ['schema','launcher_identity','root_supervisor_identity','input_ref']},
 direct_wait_ref=ref(ORIGIN/'actual_wait.root_v1.json'),wait_pid_locator=['launcher_pid'],wait_exit_locator=['returncode'],
 wait_identity={k:wait[k] for k in ['schema','launcher_pid','launcher_start_ticks','actual_direct_wait_observed','returncode']},
 completed_ref=ref(OUT/'completed.json'),completed_identity={k:done[k] for k in ['schema','status','completed_utc','chromosomes','queries','optimization_steps','summary_ref']})
obs=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',
 key=own['key'],reservation_sha256=request['reservation_sha256'],storage_ref=request['storage_ref'],
 launch_ref=request['launch_ref'],direct_wait_ref=request['direct_wait_ref'],completed_ref=request['completed_ref'],
 launcher_pid=3257287,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
 completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
 external_NFS_writers_outside_observation=True,pids=pids,observations=observations,
 actual_tool_terminal_ref=ref(ORIGIN/'actual_tool_terminal.root_v1.json'),
 known_owned_groups_ESRCH=[3257286,3257287,3257364],original_successful_science_and_scope_unchanged=True)
request['observation_ref']=write('targeted_root_observation.root_active_v1.json',obs)
write('close_request.root_active_v1.json',request)
kw=dict(expected_source_sha256=storage['budget_components']['implementation']['sha256'],expected_launcher_sha256=storage['budget_components']['launcher']['sha256'])
with guard.budget.v4.mutex(ROOT):
 state=guard._state_locked(kw['expected_source_sha256'],kw['expected_launcher_sha256'],fresh=True)
 guard.assert_open(own['key'],state[0],state[1])
 assert state[0][own['key']]==own
 authority=guard.operation_draft_v1('close_completed',request,expected_before=guard._pins(state[2]),bindings=state[3])
authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat())
authority_ref=write('close_authority.root_active_v1.json',authority)
receipt=guard.apply_close_completed_v1(authority_ref,**kw)
write('close_receipt.actual_v1.json',receipt)
print(json.dumps({k:receipt[k] for k in ['key','retained_bytes','credit_bytes','projected_bytes','artifacts_deleted']}),flush=True)

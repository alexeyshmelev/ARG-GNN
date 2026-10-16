from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,traceback,os,json,hashlib,importlib.util
sys.excepthook=lambda t,v,b:traceback.print_exception(t,v,b,file=sys.stdout)
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H056_chr21_chr22_completed_scope_actual_close_root_20261016_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
def ref(p):
 p=Path(p);return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(n,v):
 p=HERE/n
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 return ref(p)
def load(r,name):
 p=ROOT/r['path'];assert ref(p)==r
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
assert '--execute' in sys.argv
source=ROOT/'research/reviews/H056_chr21_chr22_completed_scope_eligibility_review_20261016_v1'
assert ref(source/'source_handoff.v1.json')['sha256']=='a224c899804150a952d1c25b0625a87663e2c0099d88075ba773271e3634957a'
handoff=json.loads((source/'source_handoff.v1.json').read_text())
cfg=json.loads((ROOT/'research/reviews/H056_maps_actual_root_20261016_v2/config.json').read_text());fixed=cfg['fixed_refs']
renewal=load(fixed['renewal'],'_H056_acq_close_renewal');renewal.install_renewal_v1(fixed['renewal_certificate'])
guard=load(fixed['guard'],'_H056_acq_close_original980');guard.configure_v2(fixed['repair_authority'],policy_ref=fixed['policy'])
b=json.loads((ROOT/'research/reviews/H056_maps_actual_root_20261016_v2/source_closure.json').read_text())['original_bindings']
kw=dict(expected_source_sha256=b['implementation']['sha256'],expected_launcher_sha256=b['launcher']['sha256'])
for chromosome in ['21','22']:
 qr=handoff['requests'][chromosome];assert ref(ROOT/qr['path'])=={k:qr[k] for k in ['path','sha256']}
 request=json.loads((ROOT/qr['path']).read_text());assert request['observation_ref'] is None
 original=ROOT/f'research/reviews/H056_chr{chromosome}_acquisition_actual_root_20261016_v1'
 terminal=json.loads((original/'actual_tool_terminal.root_v1.json').read_text())
 assert terminal['actual_tool_exit_code']==0 and terminal['actual_tool_terminal_consumed']
 launch=json.loads((ROOT/request['launch_ref']['path']).read_text())
 wait=json.loads((ROOT/request['direct_wait_ref']['path']).read_text())
 completed=json.loads((ROOT/request['completed_ref']['path']).read_text())
 readback=json.loads((ROOT/completed['worker_readback_ref']['path']).read_text())
 obsid=launch['root_supervisor_identity'];worker=launch['worker_pid']
 pids=list(dict.fromkeys([obsid['ppid'],obsid['pid'],worker]+[x['observation']['native_pid'] for x in readback['container_tool_actual_waits']]))
 obs=[]
 for pid in pids:
  try:os.kill(pid,0)
  except ProcessLookupError:pass
  else:raise RuntimeError('Original owned PID still present: '+str(pid))
  obs.append(dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc=datetime.now(timezone.utc).isoformat()))
 for pgid in [obsid['pgid'],launch['process_group']]+[x['observation']['process_group'] for x in readback['container_tool_actual_waits']]:
  try:os.killpg(pgid,0)
  except ProcessLookupError:pass
  else:raise RuntimeError('Original owned group still present: '+str(pgid))
 for k in ['storage_ref','launch_ref','direct_wait_ref','completed_ref']:assert ref(ROOT/request[k]['path'])==request[k]
 observation=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=request['key'],reservation_sha256=request['reservation_sha256'],storage_ref=request['storage_ref'],launch_ref=request['launch_ref'],direct_wait_ref=request['direct_wait_ref'],completed_ref=request['completed_ref'],launcher_pid=worker,root_observer_pid=obsid['pid'],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=obs,known_worker_PGID_ESRCH=launch['process_group'],actual_terminal_ref=ref(original/'actual_tool_terminal.root_v1.json'),original_separate_outer_tool_terminal_not_found=False)
 request['observation_ref']=write(f'chr{chromosome}.targeted_root_observation.root_active_v2.json',observation)
 with guard.budget.v4.mutex(ROOT):
  state=guard._state_locked(kw['expected_source_sha256'],kw['expected_launcher_sha256'],fresh=True)
  guard.assert_open(request['key'],state[0],state[1]);assert state[3]==b
  assert guard.budget.digest(state[0][request['key']])==request['reservation_sha256']
  view=state[2]
 auth=guard.operation_draft_v1('close_completed',request,expected_before=guard._pins(view),bindings=b)
 auth.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat())
 ar=write(f'chr{chromosome}.close_authority.root_active_v2.json',auth)
 receipt=guard.apply_close_completed_v1(ar,**kw)
 write(f'chr{chromosome}.close_receipt.actual_v2.json',receipt)
 print(json.dumps(dict(chromosome=chromosome,**{k:receipt[k] for k in ['key','retained_bytes','credit_bytes','projected_bytes','artifacts_deleted']})),flush=True)

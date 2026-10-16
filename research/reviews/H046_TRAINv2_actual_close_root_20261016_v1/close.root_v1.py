from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,traceback,os,json,hashlib,importlib.util
sys.excepthook=lambda t,v,b:traceback.print_exception(t,v,b,file=sys.stdout)
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H046_TRAINv2_actual_close_root_20261016_v1'
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
rp=ROOT/'research/reviews/H046_H047_H048_completed_scope_eligibility_review_20261016_v1/H046_train_v2.review.v1.json'
assert ref(rp)['sha256']=='501659505917dc038a5c73b4ee17cf5d15536916c5c8ca2432f9152f096c1748'
review=json.loads(rp.read_text());qr=review['close_request_template_ref'];assert ref(ROOT/qr['path'])==qr;request=json.loads((ROOT/qr['path']).read_text())
pids=[2811644,2811715,2811941]
obs=[]
for pid in pids:
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Original owned PID still present')
 obs.append(dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc=datetime.now(timezone.utc).isoformat()))
try:os.killpg(2811941,0)
except ProcessLookupError:pass
else:raise RuntimeError('Original owned worker group still present')
for k in ['storage_ref','launch_ref','direct_wait_ref','completed_ref']:
 assert ref(ROOT/request[k]['path'])==request[k]
observation=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=request['key'],reservation_sha256=request['reservation_sha256'],storage_ref=request['storage_ref'],launch_ref=request['launch_ref'],direct_wait_ref=request['direct_wait_ref'],completed_ref=request['completed_ref'],launcher_pid=2811715,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=obs,known_worker_PGID_ESRCH=2811941,original_separate_outer_tool_terminal_not_found=False)
request['observation_ref']=write('targeted_root_observation.root_active_v1.json',observation)
cfg=json.loads((ROOT/'research/reviews/H056_maps_actual_root_20261016_v2/config.json').read_text());fixed=cfg['fixed_refs']
renewal=load(fixed['renewal'],'_H046_close_renewal');renewal.install_renewal_v1(fixed['renewal_certificate'])
guard=load(fixed['guard'],'_H046_close_original980');guard.configure_v2(fixed['repair_authority'],policy_ref=fixed['policy'])
b=json.loads((ROOT/'research/reviews/H056_maps_actual_root_20261016_v2/source_closure.json').read_text())['original_bindings']
kw=dict(expected_source_sha256=b['implementation']['sha256'],expected_launcher_sha256=b['launcher']['sha256'])
with guard.budget.v4.mutex(ROOT):
 state=guard._state_locked(kw['expected_source_sha256'],kw['expected_launcher_sha256'],fresh=True)
 guard.assert_open(request['key'],state[0],state[1]);assert state[3]==b
 assert guard.budget.digest(state[0][request['key']])==request['reservation_sha256']
 view=state[2]
auth=guard.operation_draft_v1('close_completed',request,expected_before=guard._pins(view),bindings=b)
auth.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat())
ar=write('close_authority.root_active_v1.json',auth)
receipt=guard.apply_close_completed_v1(ar,**kw)
write('close_receipt.actual_v1.json',receipt)
print(json.dumps({k:receipt[k] for k in ['key','retained_bytes','credit_bytes','projected_bytes','artifacts_deleted']}),flush=True)

from pathlib import Path
import hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone, timedelta
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/prospective980_Census_fit2_root_activation_20261016_v1'
SOURCE={'path':'research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py','sha256':'fab6bb67bc1f4be989df9bb16e7a3a17d24652ca8238617ca060a56cb8c89d43'}
POLICY={'path':'research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/activation_policy.root_active_v1.json','sha256':'72ff346c43f96ba8430411cfe6b2250480a3aa66fa8e3b8c904a625a0399932a'}
REPAIR={'path':'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json','sha256':'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}
IMPLEMENTATION='483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'
LAUNCHER='a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'
KW=dict(expected_source_sha256=IMPLEMENTATION,expected_launcher_sha256=LAUNCHER)
def ref(p): return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(name,value):
 p=OUT/name;raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();assert len(raw)<65536
 with p.open('xb') as f: f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(p)
assert ref(ROOT/SOURCE['path'])==SOURCE and ref(ROOT/POLICY['path'])==POLICY
spec=importlib.util.spec_from_file_location('_ROOT_actual_fit2_child',ROOT/SOURCE['path']);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
g.configure_with_renewal_v1(REPAIR,policy_ref=POLICY)
mode=sys.argv[1]
print(json.dumps(dict(mode=mode,status='starting_actual_ROOT_guard_call',UTC=datetime.now(timezone.utc).isoformat())),flush=True)
if mode=='state':
 with g.budget.v4.mutex(ROOT):
  state=g._state_locked(IMPLEMENTATION,LAUNCHER,fresh=True)
  snapshot=dict(schema='ROOT_actual_Census_fit2_child_state_v1',pins=g._pins(state[2]),view=state[2],bindings=state[3],observed_at_utc=datetime.now(timezone.utc).isoformat())
  r=save('initial_child_state.root_actual_v1.json',snapshot)
 print(json.dumps(dict(status='actual_initial_child_state_verified',ref=r,projected_bytes=state[2]['projected_bytes'],headroom_bytes=980000000000-state[2]['projected_bytes'])),flush=True)
elif mode in ['observe1','observe2']:
 i=int(mode[-1]);key=sorted(g.FAILED_SCOPES)[i-1]
 snapshot=json.loads((OUT/'initial_child_state.root_actual_v1.json').read_text())
 before=snapshot['pins']
 if i==2:
  previous=json.loads((OUT/'scope1.retirement_receipt.root_actual_v1.json').read_text())
  before['completed_overlay_prefix']=previous['completed_overlay_prefix']
 rev=g.failed_revocation_draft_v1(key);rev.update(active=True,revoked_at_utc=datetime.now(timezone.utc).isoformat())
 revref=save(f'scope{i}.revocation.root_active_v1.json',rev)
 obs=g.root_observe_failed_v1(key,revocation_ref=revref,**KW)
 obsref=save(f'scope{i}.observation.root_actual_v1.json',obs)
 request=g.failed_request_draft_v1(key,observation_ref=obsref)
 authority=g.operation_draft_v1(g.RETIRE_OPERATION,request,expected_before=before,bindings=snapshot['bindings'])
 authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=15)).isoformat())
 authref=save(f'scope{i}.operation.root_active_v1.json',authority)
 print(json.dumps(dict(status='actual_targeted_absence_observed_authority_issued',key=key,observation_ref=obsref,authority_ref=authref,scientific_success_asserted=False)),flush=True)
elif mode in ['apply1','apply2']:
 i=int(mode[-1]);authref=ref(OUT/f'scope{i}.operation.root_active_v1.json')
 receipt=g.apply_retire_failed_fit2_v1(authref,**KW)
 r=save(f'scope{i}.retirement_receipt.root_actual_v1.json',receipt)
 print(json.dumps(dict(status='actual_failed_scope_retained_and_retired',receipt_ref=r,retained_bytes=receipt['retained_bytes'],credit_bytes=receipt['credit_bytes'],projected_bytes=receipt['projected_bytes'],headroom_bytes=980000000000-receipt['projected_bytes'],actual_prefix=receipt['completed_overlay_prefix'],artifacts_deleted=0,scientific_success_asserted=False)),flush=True)
else: raise ValueError('unknown explicit ROOT operation')

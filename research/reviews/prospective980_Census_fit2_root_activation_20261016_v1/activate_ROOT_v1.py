from pathlib import Path
import hashlib, importlib.util, json, os
from datetime import datetime, timezone
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/prospective980_Census_fit2_root_activation_20261016_v1'
SOURCE={'path':'research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py','sha256':'fab6bb67bc1f4be989df9bb16e7a3a17d24652ca8238617ca060a56cb8c89d43'}
PARENT={'path':'research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py','sha256':'ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635'}
PARENT_POLICY={'path':'research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json','sha256':'7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582'}
RENEWAL={'path':'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py','sha256':'491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
CERT={'path':'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json','sha256':'66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
REPAIR={'path':'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json','sha256':'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}
IMPLEMENTATION={'path':'scripts/project_budget_v9.py','sha256':'483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'}
LAUNCHER={'path':'scripts/launch_reserved_gpu_v6.py','sha256':'a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'}
REVIEW={'path':'research/reviews/prospective980_Census_fit2_retention_fresh_review_20261016_v1/review_receipt.v1.json','sha256':'614c5376dbd934bdbc1e89d111ea44d97673147909e60c4af6709f8e7dbdc13f'}
def ref(p):
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 p=ROOT/r['path']; assert p.resolve()==p and p.is_file() and not p.is_symlink()
 raw=p.read_bytes(); assert hashlib.sha256(raw).hexdigest()==r['sha256']; return raw
def load(name,r):
 read(r); spec=importlib.util.spec_from_file_location(name,ROOT/r['path']); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def save(name,value):
 p=OUT/name
 raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
 assert len(raw)<65536
 with p.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
 return ref(p)
for r in [SOURCE,PARENT,PARENT_POLICY,RENEWAL,CERT,REPAIR,IMPLEMENTATION,LAUNCHER,REVIEW]: read(r)
print(json.dumps(dict(status='actual_parent_replay_starting',UTC=datetime.now(timezone.utc).isoformat())),flush=True)
r=load('_ROOT_Census_renewal',RENEWAL); r.install_renewal_v1(CERT)
parent=load('_ROOT_Census_parent',PARENT); parent.configure_v2(REPAIR,policy_ref=PARENT_POLICY)
g=load('_ROOT_Census_retention',SOURCE)
assert all(getattr(parent.budget.v4,n).__module__==parent.budget.v4.__name__ for n in ['canonical','file_bytes','mutex'])
with parent.budget.v4.mutex(ROOT):
 state=parent._state_locked(IMPLEMENTATION['sha256'],LAUNCHER['sha256'],fresh=True)
 view=state[2]; pins=parent._pins(view)
 assert view['accounting_mode']=='prospective_980gb_H020_v3' and view['limit_bytes']==980000000000 and view['hard_limit_bytes']==1000000000000
 observation=save('actual_frozen_parent_observation.root_v1.json',dict(schema='ROOT_actual_Census_fit2_parent_observation_v1',observed_at_utc=datetime.now(timezone.utc).isoformat(),source_ref=PARENT,policy_ref=PARENT_POLICY,pins=pins,view=view,source_review_ref=REVIEW,prior_turn_classification='progress',prior_pushed_commit='364ccf4345eeb208bdc3f8d83a6f42f60eff8125'))
 activation=g.root_activation_draft_v1(base_pins=pins,base_projected_bytes=view['projected_bytes']); activation['active']=True
 activation_ref=save('frozen_parent_activation.root_active_v1.json',activation)
 policy=g.activation_policy_draft_v1(repair_authority_ref=REPAIR,base_pins=pins,base_projected_bytes=view['projected_bytes'],implementation_ref=IMPLEMENTATION,launcher_ref=LAUNCHER,root_activation_ref=activation_ref); policy['active']=True
 policy_ref=save('activation_policy.root_active_v1.json',policy)
 print(json.dumps(dict(status='ROOT_actual_child_activated_parent_future_mutations_revoked',policy_ref=policy_ref,activation_ref=activation_ref,observation_ref=observation,parent_projected_bytes=view['projected_bytes'],old_parent_journal_mutated=False,capacity_recovered_bytes=0,existing_admitted_consumers_unchanged=True)),flush=True)

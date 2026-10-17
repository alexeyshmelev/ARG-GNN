from pathlib import Path
import os,json,hashlib,datetime,importlib,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/Census_observation_ROOT_activation_20261017_v1'
def ref(n):
 p=ROOT/n
 return dict(path=n,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 assert ref(r['path'])==r
 return json.loads((ROOT/r['path']).read_text())
def write(n,v):
 p=HERE/n
 with p.open('x') as f:
  json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 return ref(str(p.relative_to(ROOT)))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 assert Path.cwd()==ROOT and __import__('sys').dont_write_bytecode
 intent=json.loads((HERE/'ROOT_intent.root_active_v1.json').read_text())
 assert intent['active'] is True,'Explicit ROOT release remains pending'
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(intent['expires_at_utc'])
 terminal=read(intent['FLARE_seal_terminal_ref']);assert terminal['actual_outer_exit_code']==0 and terminal['outer_tool_terminal_consumed'] is True
 assert intent['only_physical_GPU']==0 and intent['preserve_existing_consumers'] is True
 assert not (HERE/'explicit_child_selector.root_active_v1.json').exists()
 s='research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py'
 assert ref(s)['sha256']=='72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb'
 review=ref('research/reviews/Census_observation_two_scope_retention_independent_20261017_v1/review.v1.json')
 assert review['sha256']=='9248ade781449143d58942c333204d92281001e64efc34e8295fcf3c18e433b6'
 m=importlib.import_module('research.reviews.Census_observation_two_scope_retention_source_20261017_v1.census_observation_retention_v1')
 m._manifest()
 rv=read(review);assert rv['independent'] is True and rv['verdict']=='SOURCE_PASS' and rv['blockers']==[]
 parent=importlib.import_module('research.reviews.Relate_failed64GiB_retention_successor_source_20261017_v1.relate_failed64_retention_v1')
 parent.load_selected_guard_v1(m.BASE_SELECTOR_REF)
 policy=read(m.BASE_POLICY_REF)
 impl,launcher,repair=(policy[k] for k in ('implementation_ref','launcher_ref','marker_repair_authority_ref'))
 read(m.BASE_SELECTOR_REF)
 for row in intent['additional_actual_admissions']:
  v=read(row['terminal_ref'])
  assert v['actual_outer_exit_code']==0 and v['actual_terminal_consumed'] is True
  assert v['schema']==row['expected_schema'] and v['tool_session_id']==row['expected_tool_session_id']
 assert not (ROOT/m.JOURNAL).exists()
 released=now()
 prereq=m.prerequisite_admissions_draft_v1(H058v5_terminal_ref=m.PREREQUISITE_ADMISSIONS['H058v5']['terminal_ref'],FLARE_native_terminal_ref=m.PREREQUISITE_ADMISSIONS['FLARE_VAL001_native']['terminal_ref'],released_at_utc=released)
 prereq['active']=True
 preref=write('ROOT_prerequisites_and_release.root_active_v1.json',prereq)
 print(json.dumps(dict(phase='waiting_original_mutex',released_at_utc=released,actual_credit_bytes=0)),flush=True)
 with m.budget.v4.mutex(ROOT):
  state=parent._state_locked(impl['sha256'],launcher['sha256'],fresh=True)
  assert len(state)==7
  pins=parent._pins(state[2]);projected=state[2]['projected_bytes'];selected=now()
  m._prerequisites(preref,selected)
  targets=[]
  for key in m.TARGET_KEYS:
   t=m._target(key);own=state[0][key]
   assert key not in state[1] and own['reserved_bytes']==20_000_000_000
   assert m.budget.digest(own)==t['original_reservation_sha256']
   assert sorted(str(x) for x in parent._paths(own))==sorted(t['scope_paths'])
   m._historical_stop(t)
   targets.append(dict(key=key,original_reservation_sha256=m.budget.digest(own),reserved_bytes=own['reserved_bytes'],consumer_binding_count=len(own['consumer_bindings'])))
  capref=write('actual_parent_capture.root_v1.json',dict(schema='ROOT_Census_observation_actual_parent_capture_v1',captured_utc=selected,ROOT_release_ref=preref,actual_parent_pins=pins,actual_parent_projected_bytes=projected,all_actual_parent_bindings=state[3],original_targets=targets,additional_actual_admissions_preserved=intent['additional_actual_admissions'],scientific_completion_claimed=False,actual_credit_bytes=0))
  rev=dict(schema='Census_observation_parent_future_mutation_revocation_v1',root_authorizer='root_agent',active=True,parent_source_ref=m.BASE_REF,parent_policy_ref=m.BASE_POLICY_REF,parent_selector_ref=m.BASE_SELECTOR_REF,actual_parent_pins=pins,prerequisite_admissions_ref=preref,parent_selected_at_utc=selected,all_future_parent_reserve_close_retire_mutations_permanently_cooperatively_revoked=True,existing_parent_read_check_and_fully_charged_writers_unchanged=True,no_historical_mutation=True)
  revref=write('parent_future_mutation_revocation.root_active_v1.json',rev)
  common=dict(base_pins=pins,base_projected_bytes=projected,permanent_revocation_ref=revref,source_review_ref=review,prerequisite_admissions_ref=preref,parent_selected_at_utc=selected)
  activation=m.root_activation_draft_v1(**common);activation['active']=True
  activationref=write('ROOT_activation.root_active_v1.json',activation)
  pol=m.activation_policy_draft_v1(repair_authority_ref=repair,implementation_ref=impl,launcher_ref=launcher,root_activation_ref=activationref,**common);pol['active']=True
  polref=write('activation_policy.root_active_v1.json',pol)
  lineage=dict(base_ref=m.BASE_REF,base_policy_ref=m.BASE_POLICY_REF,base_pins=pins,base_projected_bytes=projected,base_accounting_mode=m.BASE_MODE)
  sel=m.selector_draft_v1(policy_ref=polref,source_review_ref=review,repair_authority_ref=repair,frozen_parent_lineage=lineage);sel['active']=True
  selref=write('explicit_child_selector.root_active_v1.json',sel)
  pub=write('activation_publication.root_v1.json',dict(schema='ROOT_Census_observation_activation_publication_v1',utc=now(),capture_ref=capref,selector_ref=selref,policy_ref=polref,parent_future_mutation_revocation_ref=revref,ROOT_activation_ref=activationref,parent_projected_bytes=projected,actual_credit_bytes=0,child_fresh_state_verification_pending=True,existing_consumers_unchanged=True))
 print(json.dumps(dict(publication_ref=pub,selector_ref=selref,policy_ref=polref,actual_credit_bytes=0,child_verification_pending=True)),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  print(traceback.format_exc(),flush=True);raise SystemExit(1)

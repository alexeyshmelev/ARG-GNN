import os,pathlib,json,hashlib,datetime,importlib,traceback
r=pathlib.Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(r)
p=r/'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1'
def rawref(n):
 q=r/n;return dict(path=n,sha256=hashlib.sha256(q.read_bytes()).hexdigest())
def write(n,v):
 q=p/n
 with q.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 return rawref(str(q.relative_to(r)))
try:
 source='research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py'
 assert rawref(source)['sha256']=='b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164'
 review=rawref('research/reviews/Relate_failed64GiB_retention_successor_independent_review_20261017_v1/review.v1.json')
 assert review['sha256']=='1705946c29ef199fe7e359ca33656f1604dd5091262405b7fa022c82a2c15ddb'
 m=importlib.import_module('research.reviews.Relate_failed64GiB_retention_successor_source_20261017_v1.relate_failed64_retention_v1')
 m._manifest()
 sel=json.loads((r/m.BASE_SELECTOR_REF['path']).read_text());assert rawref(m.BASE_SELECTOR_REF['path'])==m.BASE_SELECTOR_REF
 selector=importlib.import_module('research.reviews.native7_H050_stopped_retention_successor_source_20261016_v1.explicit_policy_selector_v1')
 parent=selector.load_selected_guard_v1(m.BASE_SELECTOR_REF)
 policy=json.loads((r/m.BASE_POLICY_REF['path']).read_text());impl=policy['implementation_ref'];launcher=policy['launcher_ref'];repair=policy['marker_repair_authority_ref']
 assert rawref(impl['path'])==impl and rawref(launcher['path'])==launcher
 assert json.loads((r/'research/reviews/H058_CPU_ROOT_preparation_20261017_v1/actual_admission_outer_consumed.root_v1.json').read_text())['tool_exit_code']==0
 assert json.loads((r/'research/reviews/FLARE_remaining_ROOT_preparation_VAL001_convert_20261017_v1/actual_admission_subprocess_wait.root_v1.json').read_text())['returncode']==0
 assert not (r/m.JOURNAL).exists()
 print(json.dumps({'phase':'waiting_original_mutex','actual_credit':0}),flush=True)
 with m.budget.v4.mutex(r):
  print(json.dumps({'phase':'capturing_actual_parent_under_mutex'}),flush=True)
  state=parent._state_locked(impl['sha256'],launcher['sha256'],fresh=True);assert len(state)==7
  pins=parent._pins(state[2]);projected=state[2]['projected_bytes'];key=m.TARGET_KEYS[0];t=m._target(key);own=state[0][key]
  assert key not in state[1] and own['reserved_bytes']==68719476736 and len(own['consumer_bindings'])==55
  assert m.budget.digest(own)==t['original_reservation_sha256'];assert sorted(str(a) for a in parent._paths(own))==sorted(t['scope_paths'])
  m._historical_stop(t)
  capture=write('actual_parent_capture.root_v1.json',dict(schema='ROOT_Relate64_actual_parent_capture_v1',captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_parent_pins=pins,actual_parent_projected_bytes=projected,all_actual_parent_bindings=state[3],exact_failed_key=key,original_reservation_sha256=m.budget.digest(own),original_charge=own['reserved_bytes'],original_binding_count=len(own['consumer_bindings']),prerequisite_H058_H061_C3_FLARE_admissions_complete=True,scientific_success_claimed=False,credit_claimed=0))
  recheck=parent._state_locked(impl['sha256'],launcher['sha256'],fresh=True)
  assert parent._pins(recheck[2])==pins and recheck[2]['projected_bytes']==projected and recheck[3]==state[3]
  rev=json.loads((r/'research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/parent_mutation_revocation.INACTIVE.template.v1.json').read_text());rev.update(active=True,actual_parent_pins=pins)
  revref=write('parent_mutation_revocation.root_active_v1.json',rev)
  activation=m.root_activation_draft_v1(base_pins=pins,base_projected_bytes=projected,permanent_revocation_ref=revref,source_review_ref=review);activation['active']=True
  activationref=write('ROOT_activation.root_active_v1.json',activation)
  pol=m.activation_policy_draft_v1(repair_authority_ref=repair,base_pins=pins,base_projected_bytes=projected,implementation_ref=impl,launcher_ref=launcher,root_activation_ref=activationref,permanent_revocation_ref=revref,source_review_ref=review);pol['active']=True
  polref=write('activation_policy.root_active_v1.json',pol)
  lineage=dict(base_ref=m.BASE_REF,base_policy_ref=m.BASE_POLICY_REF,base_pins=pins,base_projected_bytes=projected,base_accounting_mode=m.BASE_MODE)
  cap=m.selector_draft_v1(policy_ref=polref,source_review_ref=review,repair_authority_ref=repair,frozen_parent_lineage=lineage);cap['active']=True
  capref=write('explicit_child_selector.root_active_v1.json',cap)
  receipt=write('activation_publication.root_v1.json',dict(schema='ROOT_Relate64_activation_publication_v1',root_authorizer='root_agent',published_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),capture_ref=capture,selector_ref=capref,policy_ref=polref,parent_revocation_ref=revref,ROOT_activation_ref=activationref,parent_projected_bytes=projected,activation_credit_bytes=0,child_fresh_state_verification_pending=True,all_future_parent_mutations_permanently_cooperatively_revoked=True,existing_consumers_unchanged=True))
 print(json.dumps({'activation_publication_ref':receipt,'selector_ref':capref,'policy_ref':polref,'activation_credit_bytes':0,'fresh_child_verification_required':True}),flush=True)
except Exception:
 print(traceback.format_exc(),flush=True);raise SystemExit(1)

import pathlib,json,hashlib,datetime,importlib,traceback,os
r=pathlib.Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(r);p=r/'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1'
try:
 ref={'path':'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json','sha256':'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}
 assert hashlib.sha256((r/ref['path']).read_bytes()).hexdigest()==ref['sha256']
 cap=json.loads((p/'actual_parent_capture.root_v1.json').read_text());m=importlib.import_module('research.reviews.Relate_failed64GiB_retention_successor_source_20261017_v1.relate_failed64_retention_v1');guard=m.load_selected_guard_v1(ref);pol=json.loads((p/'activation_policy.root_active_v1.json').read_text())
 with guard.budget.v4.mutex(r):
  state=guard._state_locked(pol['implementation_ref']['sha256'],pol['launcher_ref']['sha256'],fresh=True)
  assert state[2]['ancestor_projected_bytes']==cap['actual_parent_projected_bytes'] and state[2]['projected_bytes']==cap['actual_parent_projected_bytes']+guard.BOOTSTRAP_CHARGE
  assert state[2]['base_pins']==cap['actual_parent_pins'] and state[2]['completed_overlay_prefix']=={'events':0,'tail_sha256':None}
  assert state[2]['completed_credit_bytes']==0 and state[2]['new_reserved_bytes']==0
  assert all(state[3].get(k)==v for k,v in cap['all_actual_parent_bindings'].items()) and len(state[3])==len(cap['all_actual_parent_bindings'])+1
  k=guard.TARGET_KEYS[0];assert k in state[0] and k not in state[1] and state[0][k]['reserved_bytes']==68719476736
  out={'schema':'ROOT_Relate64_fresh_child_verification_v1','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selector_ref':ref,'policy_ref':{'path':str((p/'activation_policy.root_active_v1.json').relative_to(r)),'sha256':hashlib.sha256((p/'activation_policy.root_active_v1.json').read_bytes()).hexdigest()},'exact_frozen_parent_pins_preserved':True,'all_parent_guard_roles_preserved':True,'actual_guard_role_count':len(state[3]),'parent_projected_bytes':cap['actual_parent_projected_bytes'],'projected_bytes':state[2]['projected_bytes'],'activation_bootstrap_bytes':guard.BOOTSTRAP_CHARGE,'activation_credit_bytes':0,'failed_Relate_key_still_open':True,'ordinary_mutations_must_use_child':True,'existing_jobs_unchanged':True}
  q=p/'fresh_child_verification.root_v1.json'
  with q.open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(out),flush=True)
except Exception:
 print(traceback.format_exc(),flush=True);raise SystemExit(1)

"""Independent source/metadata, synthetic proof/replay and tiny no-follow helper checks."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,importlib.util,json,os,sys,tempfile
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');CAND='research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1';REVIEW='research/reviews/native7_H050_stopped_retention_independent_review_20261016_v1'
checks=[]
def good(name,ok=True):
 assert ok,name
 checks.append(dict(name=name,status='PASS_SOFTWARE_OR_SOURCE_ONLY'))
def rejects(name,fn):
 try:fn()
 except (ValueError,KeyError,AssertionError,RuntimeError):good(name);return
 raise AssertionError('Expected rejection: '+name)
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
source=(ROOT/CAND/'inert_fixtures_v1.py').read_text();tree=ast.parse(source)
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
run.body.insert(len(run.body)-1,ast.parse('globals()["_independent_locals"]=locals()').body[0]);ast.fix_missing_locations(tree)
ns={'__name__':'_independent_retention_capture'};exec(compile(tree,'<sealed_retention_fixture_only_local_capture>','exec'),ns)
original_temp=tempfile.TemporaryDirectory
def review_temp(*a,**kw):kw['dir']=ROOT/REVIEW;return original_temp(*a,**kw)
tempfile.TemporaryDirectory=review_temp
try:result=ns['run']()
finally:tempfile.TemporaryDirectory=original_temp
assert result['check_count']==49
L=ns['_independent_locals'];c=L['c'];m=L['m'];docs=L['docs']
for key,t in m['targets'].items():
 request,proof,measurement=L['fixtures'][key];own=t['original_reservation']
 wrong=copy.deepcopy(own);wrong['reserved_bytes']-=1
 rejects(key+'_original_reservation_any_cap_drift',lambda:c._stopped_proof(request,wrong))
 revref=request['revocation_ref'];saved=copy.deepcopy(docs[revref['path']]);docs[revref['path']]['unknown_historical_descendants_absent_claimed']=True
 rejects(key+'_revocation_cannot_claim_unknown_history_absent',lambda:c._stopped_proof(request,own));docs[revref['path']]=saved
 obsref=request['observation_ref'];saved=copy.deepcopy(docs[obsref['path']]);docs[obsref['path']]['group_observations']=[]
 rejects(key+'_no_known_group_omitted',lambda:c._stopped_proof(request,own));docs[obsref['path']]=saved
 recorded=copy.deepcopy(proof);recorded['scientific_success_relabelled']=True
 rejects(key+'_recorded_failure_never_relabelled_success',lambda:c._recorded_stopped_proof(recorded,proof,L['stamp']))
 recorded=copy.deepcopy(proof);recorded['post_measurement_targets']['PID_observations'][0][-1]=(datetime.fromisoformat(recorded['pre_measurement_targets']['PID_observations'][0][-1])-timedelta(seconds=1)).isoformat()
 rejects(key+'_post_measurement_absence_must_follow_pre',lambda:c._recorded_stopped_proof(recorded,proof,L['stamp']))
 good(key+'_both_original_roots_exact_and_controller_new_native_excluded',len(t['scope_paths'])==2 and not any(r['pid'] in (3349405,3752065) for r in t['known_PID_start_rows']) and not set(t['known_process_groups'])&{3349405,3752065})
# Replay cannot credit a stopped key twice even with a fresh next-prefix authority.
events=L['events'];hashes=L['hashes'];last=copy.deepcopy(events[-1]);authref=L['fx']('independent_duplicate_retirement')
auth=c.operation_draft_v1(c.RETIRE_OPERATION,last['value']['request'],expected_before=dict(base_pins=L['p']['base_pins'],completed_overlay_prefix=dict(events=len(events),tail_sha256=hashes[-1])),bindings=L['bindings'])
auth.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat());docs[authref['path']]=auth
last['authority']=authref;events.append(last);hashes.append('f'*64)
rejects('duplicate_retirement_with_new_authority_and_prefix_rejected',lambda:c._state_locked(L['p']['implementation_ref']['sha256'],L['p']['launcher_ref']['sha256']))
events.pop();hashes.pop()
good('direct_guard_roles30_all_preserved_state38_retire41_native60_C3_66blocks',result['full_role_models']['guard_state_count']==38 and result['full_role_models']['retirement_invocation_upper_count']==41 and result['full_role_models']['future_native_full_union_count']==60 and result['full_role_models']['future_C3_full_union_count']==66 and result['full_role_models']['unchanged_future_C3_fits64'] is False)
# Production helper code on fresh owned tiny roots only.
helper=load(CAND+'/retained_nodes_v1.py','_independent_real_retained_helper')
with original_temp(prefix='independent_nodes_',dir=ROOT/REVIEW) as temp:
 root=Path(temp);roots=[root/'out',root/'aux']
 for path in roots:path.mkdir()
 payload=roots[0]/'tiny.json';payload.write_text('{}')
 alias=roots[0]/'stored-link';alias.symlink_to('/SYNTHETIC_MISSING_EXTERNAL_TARGET')
 def measure():return helper.measure_retained_nodes_v1(root,roots,expected_paths=[str(p) for p in roots],socket_is_absent=lambda p:False)
 first=measure();good('absolute_stored_link_is_charged_preserved_and_never_followed',first['entries']==4 and first['per_scope'][0]['node_types']['symlink']==1 and alias.is_symlink() and first['symlink_targets_followed'] is False)
 link=roots[1]/'hardlink';os.link(payload,link)
 rejects('retained_hardlink_blocks_without_removal',measure);good('hardlink_failure_preserves_both_names',payload.exists() and link.exists());link.unlink()
 fifo=roots[1]/'fifo';os.mkfifo(fifo)
 rejects('unsupported_retained_FIFO_blocks_without_cleanup',measure);good('FIFO_failure_preserves_node',fifo.exists());fifo.unlink()
 original=helper._open_directory;calls=[0]
 def pass_churn(*a,**kw):
  calls[0]+=1
  if calls[0]==3:(roots[1]/'between_passes').write_text('{}')
  return original(*a,**kw)
 helper._open_directory=pass_churn
 try:rejects('membership_churn_between_complete_passes_blocks',measure)
 finally:helper._open_directory=original
 (roots[1]/'between_passes').unlink()
 original_leaf=helper._leaf_metadata;fired=[False]
 def final_churn(*a,**kw):
  if not fired[0]:payload.write_text('{"changed":true}');fired[0]=True
  return original_leaf(*a,**kw)
 helper._leaf_metadata=final_churn
 try:rejects('final_no_follow_leaf_revalidation_rejects_post_pass_change',measure)
 finally:helper._leaf_metadata=original_leaf
 payload.write_text('{}');measure()
 rejects('omitted_second_scope_rejected',lambda:helper.measure_retained_nodes_v1(root,roots[:1],expected_paths=[str(roots[0])],socket_is_absent=lambda p:False))
 ancestor=root/'ancestor-link';ancestor.symlink_to(roots[0],target_is_directory=True)
 rejects('symlink_ancestor_never_followed',lambda:helper.measure_retained_nodes_v1(root,[ancestor,roots[1]],expected_paths=[str(ancestor),str(roots[1])],socket_is_absent=lambda p:False))
# Independent AST comparison preserves original ordinary accounting/guard functions.
parent=ROOT/'research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py'
old={n.name:n for n in ast.parse(parent.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};new={n.name:n for n in ast.parse((ROOT/CAND/'stopped_retention_v1.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
ordinary=['charge','warning_fields','check_prefix','assert_open','_authority','_proof','_absent','_measurement','inspect_v1','apply_reserve_v1','check_storage_locked_v1','budget_check_v9','validate_storage_v9','PilotOutputV9','consumer_bindings_v1','_stamp']
good('sixteen_ordinary_accounting_guard_function_ASTs_exact_parent',all(ast.dump(old[n],include_attributes=False)==ast.dump(new[n],include_attributes=False) for n in ordinary))
manifest=json.loads((ROOT/CAND/'source_manifest.v1.json').read_text());good('source_manifest_actual_parent_pins_projection_still_null',manifest['actual_parent_pins_and_projection'] is None)
for name in ['activation_policy.INACTIVE.template.v1.json','parent_activation.INACTIVE.template.v1.json','parent_mutation_revocation.INACTIVE.template.v1.json','explicit_selector.INACTIVE.template.v1.json','target1_revocation.INACTIVE.template.v1.json','target2_revocation.INACTIVE.template.v1.json','target1_observation.INACTIVE.template.v1.json','target2_observation.INACTIVE.template.v1.json','target1_operation.INACTIVE.template.v1.json','target2_operation.INACTIVE.template.v1.json']:
 good(name+'_remains_inactive',json.loads((ROOT/CAND/name).read_text())['active'] is False)
good('live_v12_source_manifest_stays_exact_original',hashlib.sha256((ROOT/'research/reviews/native_VAL000_full_state_resume_source_20261016_v12/source_manifest.v1.json').read_bytes()).hexdigest()=='734c8f55fc6a06c20fba9b63b9d72bf733ee9e6b49261f8da26c8bb6b27ae664')
good('no_scientific_runtime_imports',not any(n in sys.modules for n in ('torch','numpy','composer','tskit')))
value=dict(status='PASS_SOFTWARE_AND_SOURCE_ONLY',independent_check_count=len(checks),checks=checks,fixture_source_capture_only=True,all_helper_operation_paths_owned_tiny_fixture_roots=True,all_probes_measurements_retirements_replay_state_synthetic=True,no_actual_guards_probes_scope_measurement_accounting_revocation_activation_admission_jobs_payload_or_Git=True)
with (ROOT/REVIEW/'independent_checks.v1.json').open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')

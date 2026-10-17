"""Independent focused v2 repair/full-route source and RAM review.
Actual new factory and genuine code authenticators run. Only scientific payload,
process/resource/clock/IO/state/ancestor interfaces are RAM doubles.
"""
from pathlib import Path
from types import FunctionType
import ast,contextlib,copy,hashlib,io,json,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
D='research/reviews/H060_global_only_full_training_source_20261017_v2'
OLD='research/reviews/H060_global_only_full_training_source_20261017_v1'
C='research/reviews/H060_global_observation_seal_source_20261017_v2'
OC='research/reviews/H060_global_observation_seal_source_20261017_v1'
OUT='research/reviews/H060_global_only_full_join_independent_20261017_v2'
SKIP={'experiments/results/census_tsinfer_observation_v1/census-tsinfer-inspect-v1/evidence.json',
'experiments/results/census_tsinfer_observation_v1/census-tsinfer-numerical-v1/evidence.json'}
def run():
 reports=[];negative=[]
 def need(ok,msg):
  if not ok:raise AssertionError(msg)
 def ck(name,ok=True):need(ok,name);reports.append(name)
 def reject(name,call):
  try:call()
  except (ValueError,AssertionError,KeyError,TypeError,AttributeError,OSError):
   negative.append(name);reports.append(name);return
  raise AssertionError('Accepted '+name)
 def raw(path):return (ROOT/path).read_bytes()
 def read(path):return json.loads(raw(path))
 def sha(path):return hashlib.sha256(raw(path)).hexdigest()
 ck('exact_full_handoff_pin',sha(D+'/source_handoff.v1.json')=='bdd01d277f993c3f2f88c687893d953c052f2f2344e51994c9668e22f6d2be06')
 ck('exact_full_manifest_pin',sha(D+'/source_manifest.v1.json')=='6aca41fdcadd2e28dd159e1627947add503d7db0bbb0692777e8749ad934181b')
 ck('blocked_prior_review_preserved',sha('research/reviews/H060_global_only_full_join_independent_20261017_v1/review.v1.json')=='dcd363a1fc2a3a220bd60553c84d0e325c8fae66f86809a86764fbc542e74a87')
 m=read(D+'/source_manifest.v1.json');oldm=read(OLD+'/source_manifest.v1.json')
 ck('all583_predecessor_source_pins_retained_exactly',len(oldm['source_sha256'])==583 and all(m['source_sha256'].get(k)==v for k,v in oldm['source_sha256'].items()))
 science=read(D+'/scientific_source_pins.v2.json')['refs'];oldscience=read(OLD+'/scientific_source_pins.v2.json')['refs']
 ck('all89_scientific_pins_retained_and_four_current_identities_added',len(science)==93 and len(oldscience)==89 and all(r in science for r in oldscience))
 checked={};excluded={}
 for path,digest in m['source_sha256'].items():
  if path in SKIP:excluded[path]=digest;continue
  p=ROOT/path
  need(p.is_file() and not p.is_symlink() and p.stat().st_size<=2**20,'Bounded frozen source/metadata '+path)
  got=hashlib.sha256(p.read_bytes()).hexdigest()
  need(got==digest,'Exact manifest source bytes '+path);checked[path]=got
 ck('624_of626_bounded_operational_pins_exact_two_historical_payload_pins_only',len(checked)==624 and set(excluded)==SKIP)
 ck('all93_scientific_refs_exact_source_bytes',all(checked.get(r['path'])==r['sha256'] for r in science))
 comp=raw(C+'/global_observation_protocol_v1.py').decode()
 ck('component_only_namespace_and_closure_attribute_correction',
  comp==raw(OC+'/global_observation_protocol_v1.py').decode().replace(OC,C).replace('actual.join.__closure__or()','actual.join.__closure__ or ()'))
 ck('runner_only_component_namespace_change',raw(D+'/warmstart_train_v2.py').decode()==raw(OLD+'/warmstart_train_v2.py').decode().replace(OC,C).replace(OC.replace('/','.'),C.replace('/','.')))
 ck('worker_only_full_route_namespace_change',raw(D+'/worker_entry_v1.py').decode()==raw(OLD+'/worker_entry_v1.py').decode().replace(OLD,D).replace(OLD.replace('/','.'),D.replace('/','.')))
 def funcs(path):return {n.name:n for n in ast.parse(raw(path)).body if isinstance(n,ast.FunctionDef)}
 oldf,newf=funcs(OLD+'/root_driver_v1.py'),funcs(D+'/root_driver_v1.py')
 ck('all_other_transaction_lifecycle_functions_AST_exact',
  set(oldf)==set(newf) and all(ast.dump(oldf[k],include_attributes=False)==ast.dump(newf[k],include_attributes=False) for k in oldf if k!='global_scientific_review'))
 oldp,newp=read(OLD+'/protocol.inactive.v1.json'),read(D+'/protocol.inactive.v1.json')
 ck('original_science_model_update_cache_checkpoint_validation_and_caps_unchanged',
  all(oldp[k]==newp[k] for k in ('training','validation','expected_parameters','mechanisms','warmstart','checkpoints','CPU_parent_cache','initialization'))
  and read(OLD+'/caps.inactive.v1.json')==read(D+'/caps.inactive.v1.json'))
 # Author tests reproduced once each with result writers disabled. The second
 # test is the author's adaptation of the prior independent RAM transport.
 def reproduce(file):
  ns={'__name__':'independent_reproduction_'+file,'__file__':str(ROOT/D/file)}
  argv=sys.argv;sys.argv=['SOURCE_RAM_REVIEW']
  try:
   with contextlib.redirect_stdout(io.StringIO()):exec(compile(raw(D+'/'+file),str(ROOT/D/file),'exec',dont_inherit=True,optimize=0),ns)
  finally:sys.argv=argv
  return ns
 f=reproduce('repair_actual_factory_RAM_checks_v2.py')
 r=reproduce('repair_request_memo_RAM_checks_v2.py')
 ck('author6_factory_and_inherited9groups10negatives_reproduced_separately',
  f['result']['assertions']==6 and f['result']['inherited_author_groups']==9 and f['result']['inherited_author_negatives']==10)
 ck('author12_request_memo_assertions4negatives_reproduced_separately',
  r['result']['assertions']==12 and r['result']['negative_fixtures']==4 and r['result']['genuine_memo2_callable_source_authentication_executed'])
 # Independently drive genuine repaired factory with original target and nested
 # actual join code. No factory, _exact_method or source reader is substituted.
 ps=f['protocol_scope'];fixture=f['ns'];a=fixture['ns'];proto=fixture['proto']
 barrier=copy.deepcopy(f['stored']['TRAIN_observation_barrier.json'])
 source_refs=fixture['b']['global_preparation_source_refs'];bindings=f['actual_bindings']
 factory=ps['make_actual_global_target_joiner_factory_v1'];validate=ps['validate_target_protocol_binding_v1']
 compiled=compile(comp,str(ROOT/C/'global_observation_protocol_v1.py'),'exec',dont_inherit=True,optimize=0)
 codes={x.co_name:x for x in compiled.co_consts if isinstance(x,types.CodeType)}
 ck('actual_factory_and_genuine_exact_method_code_objects_equal_pinned_source',
  factory.__code__==codes['make_actual_global_target_joiner_factory_v1']
  and ps['_exact_method_v1'].__code__==codes['_exact_method_v1']
  and factory.__globals__ is ps and ps['_exact_method_v1'].__globals__ is ps)
 events=[]
 made=factory(bindings,barrier=barrier,source_refs=source_refs,runtime_guard=events.append)
 validate(made.execution_binding,f['expected_original_binding'],barrier=barrier)
 ck('actual_factory_authenticates_original_methods_and_binds_new_component',
  made.execution_binding['component_source_ref']==fixture['b']['global_component_source_ref']
  and made.execution_binding['original_actual_target_binding']==f['expected_original_binding']
  and events==['before_new_global_target_protocol_import','after_new_global_target_protocol_source_join'])
 # Re-run actual runner through this exact factory. Stop at the resolver before
 # any truth file or array, and check complete same-object/durable cache barrier.
 old_factory=proto.make_actual_global_target_joiner_factory_v1
 old_validate=proto.validate_target_protocol_binding_v1;oldwarm=a['load_accepted_backbone_v1']
 proto.make_actual_global_target_joiner_factory_v1=factory
 proto.validate_target_protocol_binding_v1=validate
 a['load_accepted_backbone_v1']=lambda *args,**kwargs:({}, {},fixture['planrefs'],f['expected_original_binding'])
 resolved=[]
 class ResolverBoundary(Exception):pass
 def resolver(parent):
  need('TRAIN_observation_barrier.json'in f['stored'] and len([e for e in f['events'] if e[0]=='prepare'])==20,'All20 durable before resolver')
  resolved.append(parent);raise ResolverBoundary()
 f['reset']()
 try:
  a['run_train_v1'](fixture['cfg'],parent_plan_refs=fixture['planrefs'],artifact_acceptance_ref=fixture['sym']('independent_actual_factory'),
   writer=f['original_writer'](),device='cpu',worker_tmp_directory='RAM',resolve_train_truth=resolver,
   runtime_guard=f['original_guard'],actual_train_bindings=bindings,resume_ref=None)
 except ResolverBoundary:pass
 else:raise AssertionError('Actual runner did not reach resolver')
 finally:
  proto.make_actual_global_target_joiner_factory_v1=old_factory
  proto.validate_target_protocol_binding_v1=old_validate;a['load_accepted_backbone_v1']=oldwarm
 ck('actual_runner_factory_genuine_codeauth_typed_join_reaches_resolver_after_all20_durable',
  resolved==[proto.TRAIN[0]] and 'completed_training.json'not in f['stored'])
 # Independent adversarial source/barrier joins.
 targets=f['targets'];hooks=f['hooks']
 original_join=targets.TrainTargetJoinV1.join
 def wrong_join(self,*args,**kwargs):raise AssertionError('Wrong body must never execute')
 targets.TrainTargetJoinV1.join=wrong_join
 reject('changed_original_base_join_code_rejected_by_genuine_authenticator',
  lambda:factory(bindings,barrier=barrier,source_refs=source_refs,runtime_guard=lambda e:None))
 targets.TrainTargetJoinV1.join=original_join
 original_hookfactory=hooks.make_actual_target_joiner_factory_v1
 def wrong_actual_factory(b):
  cls=original_hookfactory(b)
  cls.join=wrong_join
  return cls
 hooks.make_actual_target_joiner_factory_v1=wrong_actual_factory
 reject('changed_actual_nested_join_rejected_by_genuine_authenticator',
  lambda:factory(bindings,barrier=barrier,source_refs=source_refs,runtime_guard=lambda e:None))
 hooks.make_actual_target_joiner_factory_v1=original_hookfactory
 bad=copy.deepcopy(barrier);bad['parents'].pop(proto.TRAIN[-1])
 reject('missing_twentieth_parent_blocks_target_factory',lambda:factory(bindings,barrier=bad,source_refs=source_refs,runtime_guard=lambda e:None))
 badrefs=copy.deepcopy(source_refs)
 next(x for x in badrefs if x['path']==C+'/global_observation_protocol_v1.py')['sha256']='0'*64
 reject('wrong_repaired_component_digest_blocks_factory',lambda:factory(bindings,barrier=barrier,source_refs=badrefs,runtime_guard=lambda e:None))
 badbinding=copy.deepcopy(made.execution_binding);badbinding['original_actual_target_binding']['target_equations_unchanged']=False
 reject('changed_nested_original_target_equations_binding_rejected',lambda:validate(badbinding,f['expected_original_binding'],barrier=barrier))
 bad=copy.deepcopy(barrier);bad['validated_cache_identities'][proto.TRAIN[0]]['tensor_sha256']='0'*64
 reject('cache_identity_mismatch_blocks_factory',lambda:factory(bindings,barrier=bad,source_refs=source_refs,runtime_guard=lambda e:None))
 # Whole scientific review explicitly covers repaired bytes, beyond predecessor.
 for name,change in (
  ('missing_component_repair_ref_cannot_inherit_old_component_PASS',lambda v:v.pop('component_repair_source_ref')),
  ('old_component_bytes_cannot_approve_new_factory',lambda v:v.update(component_repair_source_ref=read(OLD+'/scientific_binding.v1.json')['global_component_source_ref']))):
  value=copy.deepcopy(r['science_review']);change(value)
  record=r['putram']('RAM_ONLY/independent_'+name+'.json',value)
  reject(name,lambda record=record:r['w'].global_scientific_review(r['c'],record))
 # Genuine memo auth: correct pathname but altered guard code must fail before
 # any additional state lock, write or reserve.
 g=r['g'];before=(len(r['memory']),list(r['events']));original=g._state_locked
 ns={}
 exec("def _state_locked(*args,**kwargs):\n raise AssertionError('tampered state must never run')",ns)
 g._state_locked=FunctionType(ns['_state_locked'].__code__,g.__dict__,'_state_locked')
 reject('genuine_memo_auth_rejects_changed_guard_code_at_correct_path',
  lambda:r['w']._make_root_mutation_memo(g,r['q'],r['request_ref']))
 g._state_locked=original
 ck('changed_guard_constructor_no_additional_lock_or_mutation',before==(len(r['memory']),list(r['events'])))
 admitted=copy.deepcopy(r['admitted']);reservation=admitted['storage']['reservation']
 selected=r['b']['selected_child_source_ref']
 reservation['consumer_bindings']=[x for x in reservation['consumer_bindings'] if x!=selected]
 badinput=r['putram'](r['q']['packet_directory']+'/independent_missing_current_role.json',admitted)
 reject('admitted_input_missing_current_Census_source_role_rejected',lambda:r['c'].input_spec(badinput))
 ck('full_original47_roles_62direct_refs_64ceiling_preserved',
  r['answer']['consumer_binding_count']==62 and len(r['guards'])==47
  and all(x in r['admitted']['storage']['reservation']['consumer_bindings'] for x in r['guards'].values())
  and m['direct_ref_ceiling']==64)
 ck('original_two_RAM_mutexes_and_one_append_input_route',
  r['events'].count('lock')==r['events'].count('unlock')==2 and not r['locked'][0]
  and len([p for p in r['memory'] if p.startswith(g.JOURNAL+'/')])==1)
 ck('no_scientific_framework_import',not any(n in sys.modules for n in ('torch','numpy','torch_geometric','tskit')))
 return dict(schema='H060_global_only_minimal_repair_independent_full_source_RAM_checks_v2',
  status='PASS',source_only=True,independent=True,independent_assertions=len(reports),
  independent_negative_fixtures=len(negative),reports=reports,negative_cases=negative,
  operational_pins=626,operational_compact_source_metadata_pins_verified=624,
  science_pins_verified=93,all583_predecessor_pins_retained=True,
  unopened_historical_numeric_evidence_pin_map=excluded,
  separate_author_reproduction=dict(inherited_groups=9,inherited_negatives=10,actual_factory_assertions=6,
    request_memo_assertions=12,request_memo_negatives=4,independent=False,result_writers_enabled=False),
  actual_factory_and_genuine_target_code_auth_typed_join_resolver_boundary_exercised=True,
  actual_request_genuine_memo2_sourceauth_Census_state_reserve_append_input_RAM=True,
  actual_clock_process_resource_GUARD_accounting_admission_arrays_truth_Torch_GPU_or_Git_operations=False,
  scaffold_disclosure='Author source-only metadata transports and scientific interfaces reused. Additional cases independently selected. Actual v2 factory/_exact_method/source reader and actual original target join methods execute. Handwritten source-module scaffolds expose original code objects. Resolver sentinel precedes truth payload. Memo2 genuine callable authentication and original current-Census bodies run over RAM operational interfaces.',
  real_speed_fit_accuracy_parity_or_resource_capacity_established=False)
if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))

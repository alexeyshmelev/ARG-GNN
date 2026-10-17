"""Independent source/RAM fault review; author reproduction labeled separately.
All writer, tensor, target, process, lock, resource and time interfaces are RAM.
"""
from pathlib import Path
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE=ROOT/'research/reviews/H060_global_only_full_training_source_20261017_v1'
REVIEW=ROOT/'research/reviews/H060_global_only_full_join_independent_20261017_v1'
checks=[];negative=[];checked=set();skipped=[]
def ck(label,value=True):
 if not value:raise AssertionError(label)
 checks.append(label)
def raw(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 assert p.suffix in ('.py','.sh','.json','.md','.patch','.diff','.txt','.log') and p.is_file() and p.stat().st_size<=2**20,str(p)
 checked.add(str(p.relative_to(ROOT)));return p.read_bytes()
def read(path):return json.loads(raw(path))
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw(p)).hexdigest())
def funcs(path):return {n.name:n for n in ast.parse(raw(path)).body if isinstance(n,ast.FunctionDef)}
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def execute(nodes,scope):exec(compile(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]),'<independent_exact_source_AST>','exec',dont_inherit=True,optimize=0),scope)
def reject(label,call):
 try:call()
 except (AssertionError,ValueError,KeyError,TypeError,AttributeError,OSError):negative.append(label);return
 raise AssertionError('Accepted '+label)
ck('exact_requested_outer_manifest',ref(HERE/'source_manifest.v1.json')['sha256']=='c543c9a4dd6209fae3cafb67a8e2ea199afe1b62252e0c7fa8db8ab4a58494af')
ck('exact_requested_handoff',ref(HERE/'source_handoff.v1.json')['sha256']=='597fccde82ee279a597e5ad386e4f20ab3eef87844fd642dbba61bd95dbd6790')
manifest=read(HERE/'source_manifest.v1.json');scientific=read(HERE/'scientific_manifest.v1.json');audit=read(HERE/'audit_manifest.v1.json')
for label,table in [('operational',manifest['source_sha256']),('scientific',scientific['source_sha256']),('audit',audit['files_sha256'])]:
 for path,sha in table.items():
  if Path(path).stat().st_size>2**20:skipped.append(dict(path=path,sha256=sha,reason='Inherited large numerical-evidence pin retained without payload read'));continue
  ck(label+'_exact_'+path,ref(path)['sha256']==sha)
ck('all_expected583_operational_pins_and89_scientific_refs',len(manifest['source_sha256'])==583 and len(read(HERE/'scientific_source_pins.v2.json')['refs'])==89)
# Author tests are reproduced verbatim with their final stdout disabled; no output file writer exists.
author_tree=ast.parse(raw(HERE/'source_checks_v1.py'));ck('author_scaffold_has_no_direct_result_file_writer',not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in ('write_text','write_bytes')for n in ast.walk(author_tree)))
ns=dict(__name__='_separately_labeled_author_reproduction',__file__=str(HERE/'source_checks_v1.py'))
exec(compile(author_tree,str(HERE/'source_checks_v1.py'),'exec',dont_inherit=True,optimize=0),ns)
author=copy.deepcopy(ns['result']);ck('separate_author9groups10negatives_reproduced',author['checks_count']==9 and author['negative_count']==10 and author['pure_direct_union']==62)
# Original scientific functions/model kernels/cache and original outer lifecycle are exact pins.
r=funcs(HERE/'warmstart_train_v2.py');previous=funcs(ROOT/'research/reviews/H060_legacy_fast_full_training_source_20261017_v1/warmstart_train_v2.py')
for n in ('load_accepted_backbone_v1','_cache_tensors_v2','_cache_semantics_v2','_cache_charge_v2','_cache_admit_v2','_cache_get_v2','_source_snapshot_v2','_source_check_v2','_progress_v1'):ck('independent_exact_original_'+n,same(r[n],previous[n]))
text=raw(HERE/'warmstart_train_v2.py').decode();comp=raw(ROOT/ns['b']['global_component_source_ref']['path']).decode()
ck('global_preparer_calls_original_loader_once_and_global_prep_once',comp.count("state=load_observation(")==1 and comp.count('bundle,metadata=prepare_from_loaded_observation_v1(')==1)
ck('no_legacy_provider_or_legacy_block_seal_execution',all(s not in text for s in ('SealedObservationProviderV1','seal_then_prepare_one_parent_v1','providers[')))
ck('actual_original_cache_get_returns_checked_bundle_object',any(isinstance(n,ast.Return)and ast.unparse(n.value)=='bundle'for n in ast.walk(r['_cache_get_v2'])))
ck('worker_target_factory_deferred_to_runner_and_no_resume',not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='make_actual_target_joiner_factory_v1'for n in ast.walk(ast.parse(raw(HERE/'worker_entry_v1.py')))) and 'resume_ref=None' in raw(HERE/'worker_entry_v1.py').decode())
# Independent new faults use exact function bodies and the reproduced handwritten metadata transport.
a=ns['ns'];proto=ns['proto'];events=ns['events'];stored=ns['stored'];objects=ns['cache_objects'];mode=ns['mode'];mode[0]=None
original_get=a['_cache_get_v2'];original_prepare=proto.prepare_and_seal_global_parent_v1;original_factory=proto.make_actual_global_target_joiner_factory_v1
original_source=a['_source_check_v2'];original_writer=ns['Writer'];original_guard=ns['guard']
def reset():events.clear();stored.clear();objects.clear();mode[0]=None
# Failure to durably save all20 barrier must not construct factory or resolve truth.
class FailBarrierWriter(original_writer):
 def save(self,name,value,kind='json'):
  if name=='TRAIN_observation_barrier.json':events.append(('barrier_write_failed',None));raise OSError('RAM fsync failure')
  return super().save(name,value,kind)
def run_with(writer=None,guard=None,config=None,resume=None):
 return a['run_train_v1'](config or ns['cfg'],parent_plan_refs=ns['planrefs'],artifact_acceptance_ref=ns['sym']('accepted600_independent'),writer=writer or original_writer(),device='cpu',worker_tmp_directory='RAM',resolve_train_truth=lambda p:events.append(('independent_truth',p)),runtime_guard=guard or original_guard,actual_train_bindings={p:{}for p in proto.TRAIN},resume_ref=resume)
reset();reject('durable_barrier_write_failure_prevents_factory_and_truth',lambda:run_with(FailBarrierWriter()))
ck('failed_durable_barrier_never_constructs_factory_or_truth',not any(e[0]in ('factory','target','independent_truth')for e in events))
# Cache stamp corruption independent of bundle.validate (which remains successful).
def corrupt_stamp_guard(event):
 original_guard(event)
 if event=='before_all20_original_cache_integrity_check' and len([e for e in events if e[0]=='integrity'])==20:objects[proto.TRAIN[-1]].token=object()
reset();reject('changed_tensor_identity_before_barrier_rejected',lambda:run_with(guard=corrupt_stamp_guard))
ck('changed_stamp_rejected_before_barrier_factory_truth',not any(e[0]in ('factory','target','independent_truth')for e in events)and 'TRAIN_observation_barrier.json'not in stored)
# Actual witness object must match, even if a replacement passes the old generic cache check.
# This test intentionally substitutes only the checked return object's transport.
def replacement_get(cache,parent):
 value=original_get(cache,parent)
 if parent==proto.TRAIN[-1] and not any(e[0]=='factory'for e in events):return copy.copy(value)
 return value
a['_cache_get_v2']=replacement_get;reset();reject('different_bundle_object_with_equal_metadata_rejected',run_with)
ck('same_hashed_object_fault_precedes_barrier_and_truth',not any(e[0]in ('factory','independent_truth')for e in events))
a['_cache_get_v2']=original_get
# Small full-cache cap must fail rather than truncate/recompute/resolve targets.
reset();reject('cache_overflow_no_crop_or_recompute_fallback',lambda:run_with(config=dict(ns['cfg'],max_cached_parent_bytes=19)))
ck('cache_cap_failure_never_reaches_target_factory',not any(e[0]in ('factory','target','independent_truth')for e in events))
# All20 source/protocol and nested old actual binding are required by real typed gate.
reset();run_with();barrier=stored['TRAIN_observation_barrier.json'];valid=original_factory({},barrier=barrier,source_refs=ns['b']['global_preparation_source_refs'],runtime_guard=lambda e:None).execution_binding
bad=copy.deepcopy(barrier);bad['parents'].pop(proto.TRAIN[-1]);reject('nineteen_parent_barrier_rejected',lambda:proto.validate_barrier_v1(bad))
bad=copy.deepcopy(barrier);bad['parents'][proto.TRAIN[-1]]['global_identity']['tensor_sha256']='0'*64;reject('mismatched_twentieth_global_identity_rejected',lambda:proto.validate_barrier_v1(bad))
bad=copy.deepcopy(valid);bad['original_observation_protocol_or_checkpoint_identity_claimed']=True;reject('legacy_checkpoint_identity_claim_rejected',lambda:proto.validate_target_protocol_binding_v1(bad,ns['oldtarget'],barrier=barrier))
reject('heldout_truth_gate_rejected_before_resolver',lambda:proto.global_observation_gate_v1('american_c3_val_000',split='val',plan=ns['plans'][proto.TRAIN[0]],seal=ns['seals'][proto.TRAIN[0]],barrier=barrier))
# Resume rejection occurs before saved600 checkpoint load/config/model preparation.
oldwarm=a['load_accepted_backbone_v1'];calls=[];a['load_accepted_backbone_v1']=lambda *a,**k:calls.append('warmstart')
reset();reject('fresh_transfer_resume_rejected_before_warmstart',lambda:run_with(resume=ns['sym']('new-global-checkpoint')));ck('resume_rejection_does_not_load_payload',calls==[])
a['load_accepted_backbone_v1']=oldwarm
# Lifetime cache invalidation after target creation must stop before any next optimizer pair.
steps=[];oldstep=a['train_complete_parent_step_v1']
def step(arm,*args,**kw):
 steps.append(arm)
 result=oldstep(arm,*args,**kw)
 if len(steps)==1:objects[proto.TRAIN[0]].token=object()
 return result
a['train_complete_parent_step_v1']=step;reset();reject('post_barrier_cache_identity_change_stops_lifetime_reuse',run_with)
ck('cache_drift_stops_midpair_without_claimed_completed_pair',len(steps)==1 and stored['failed_training.json']['last_complete_paired_updates_per_arm']==0 and 'completed_training.json'not in stored)
a['train_complete_parent_step_v1']=oldstep
# Source drift at start must fail before first parent, all20 barrier and truth.
a['_source_check_v2']=lambda snapshot:(_ for _ in ()).throw(ValueError('RAM source drift'));reset();reject('source_identity_drift_before_preparation_stops',run_with)
ck('source_drift_never_prepares_or_resolves_truth',not any(e[0]in ('prepare','factory','target','independent_truth')for e in events));a['_source_check_v2']=original_source
# Exercise the ACTUAL new target factory, genuine original target-code
# authentication, original actual-path factory, and original base join up to the
# real supplied resolver boundary. A sentinel prevents all truth/payload IO.
import builtins,dataclasses,types,typing
protocol_scope=dict(__name__='_independent_real_global_protocol')
exec(compile(comp,str(ROOT/ns['b']['global_component_source_ref']['path']),'exec',dont_inherit=True,optimize=0),protocol_scope)
target_path='src/arggnn/american_c3_gnn_targets_v1.py';hook_path='research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/execution_hooks_prefetch_v1.py'
targets=types.ModuleType('arggnn.american_c3_gnn_targets_v1');targets.__file__=str(ROOT/target_path)
rows={p:dict(simulation_output_directory='RAM_ONLY/original_'+p)for p in proto.TRAIN}
def registry_fixture(record):
 parent=next(p for p in proto.TRAIN if ns['plans'][p]['focal_registry_ref']==record)
 return dict(parent_id=types.SimpleNamespace(item=lambda:parent),class_order=types.SimpleNamespace(tolist=lambda:list(proto.CLASSES)),record_ids=range(10000))
targets.__dict__.update(REGISTRY_REF=proto.REGISTRY_REF,require=ns['need'],authoritative_registry_v1=lambda r:copy.deepcopy(rows),registry=registry_fixture,identity_sha=lambda r:'f'*64,CLASSES=proto.CLASSES,Callable=typing.Callable)
def exact_nodes(path,names):
 tree=ast.parse(raw(path));return [n for n in tree.body if (isinstance(n,ast.ImportFrom)and n.module=='__future__')or(isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names)]
exec(compile(ast.Module(body=exact_nodes(target_path,('TrainTargetJoinV1',)),type_ignores=[]),targets.__file__,'exec',dont_inherit=True,optimize=0),targets.__dict__)
hooks=types.ModuleType('research.reviews.C3_GNN_prefetch_isolated980_root_20261015_v2.execution_hooks_prefetch_v1');hooks.__file__=str(ROOT/hook_path)
htree=ast.parse(raw(hook_path));hook_constants={}
for n in htree.body:
 if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
  try:hook_constants[n.targets[0].id]=ast.literal_eval(n.value)
  except (ValueError,TypeError):pass
actual_bindings={p:dict(inferred_observation_ref=ns['plans'][p]['observation_ref'],simulation_completion_ref=ns['sym'](p+'_sim_completion'),simulation_config_ref=ns['sym'](p+'_sim_config'))for p in proto.TRAIN}
actualmeta={p:dict(binding=actual_bindings[p],simulation_directory='RAM_ONLY/actual_'+p,private_truth_ref=ns['sym'](p+'_truth_boundary'))for p in proto.TRAIN}
def hook_import(name,globals=None,locals=None,fromlist=(),level=0):
 if name=='arggnn.american_c3_gnn_targets_v1':return targets
 return builtins.__import__(name,globals,locals,fromlist,level)
hooks.__dict__.update(hook_constants,require=ns['need'],TRAIN=proto.TRAIN,actual_train_metadata_v1=lambda b:copy.deepcopy(actualmeta),copy=copy,ref=ref,dataclasses=dataclasses,__builtins__=dict(vars(builtins),__import__=hook_import))
exec(compile(ast.Module(body=exact_nodes(hook_path,('make_actual_target_joiner_factory_v1',)),type_ignores=[]),hooks.__file__,'exec',dont_inherit=True,optimize=0),hooks.__dict__)
def component_import(name,globals=None,locals=None,fromlist=(),level=0):
 if name=='importlib':return types.SimpleNamespace(import_module=lambda module:targets if module==targets.__name__ else hooks if module==hooks.__name__ else (_ for _ in ()).throw(ValueError('Unexpected scientific import')))
 return builtins.__import__(name,globals,locals,fromlist,level)
protocol_scope['__builtins__']=dict(vars(builtins),__import__=component_import)
exec(compile(comp,str(ROOT/ns['b']['global_component_source_ref']['path']),'exec',dont_inherit=True,optimize=0),protocol_scope)
expected_original_binding=hooks.make_actual_target_joiner_factory_v1(actual_bindings).execution_binding
old_factory=proto.make_actual_global_target_joiner_factory_v1;old_validate=proto.validate_target_protocol_binding_v1;oldwarm=a['load_accepted_backbone_v1']
proto.make_actual_global_target_joiner_factory_v1=protocol_scope['make_actual_global_target_joiner_factory_v1'];proto.validate_target_protocol_binding_v1=protocol_scope['validate_target_protocol_binding_v1']
a['load_accepted_backbone_v1']=lambda *args,**kw:({}, {},ns['planrefs'],expected_original_binding)
class StopBeforeTruthPayload(Exception):pass
resolved=[]
def truth_boundary(parent):
 ck('actual_target_resolver_reached_only_after_durable_all20_barrier','TRAIN_observation_barrier.json'in stored and len([e for e in events if e[0]=='prepare'])==20)
 resolved.append(parent);raise StopBeforeTruthPayload('RAM stop before any private truth file')
def real_factory_guard(event):
 original_guard(event)
 if event=='before_new_global_target_protocol_import':ck('actual_factory_source_authentication_follows_durable_barrier','TRAIN_observation_barrier.json'in stored)
reset()
try:a['run_train_v1'](ns['cfg'],parent_plan_refs=ns['planrefs'],artifact_acceptance_ref=ns['sym']('accepted600_actual_factory_boundary'),writer=original_writer(),device='cpu',worker_tmp_directory='RAM',resolve_train_truth=truth_boundary,runtime_guard=real_factory_guard,actual_train_bindings=actual_bindings,resume_ref=None)
except AttributeError as error:
 ck('actual_new_factory_SOURCE_BLOCKER_reproduced_exact_missing_closure_attribute',"'__closure__or'" in str(error))
 factory_blocker=dict(id='H060_GLOBAL_FACTORY_CLOSURE_ATTRIBUTE',priority=1,
  path=ns['b']['global_component_source_ref']['path'],line=214,
  exact_expression='actual.join.__closure__or()',exception_type='AttributeError',message=str(error),
  effect='Actual runner fails after all20 durable barrier while constructing target factory; no TRAIN target resolver reached',
  source_ref=ns['b']['global_component_source_ref'])
except StopBeforeTruthPayload:raise AssertionError('Expected sealed component blocker disappeared')
else:raise AssertionError('Expected actual factory blocker did not reproduce')
ck('blocked_actual_factory_never_opens_truth_or_claims_training_completion',resolved==[] and 'completed_training.json'not in stored)
proto.make_actual_global_target_joiner_factory_v1=old_factory;proto.validate_target_protocol_binding_v1=old_validate;a['load_accepted_backbone_v1']=oldwarm
ck('scientific_modules_never_imported',not any(name in sys.modules for name in ('torch','numpy','torch_geometric','tskit')))
# Author result and independent assertions are intentionally separate.
result=dict(schema='H060_global_only_independent_static_and_RAM_fault_review_v1',status='SOURCE_BLOCKED',source_only=True,independent=True,blockers=[factory_blocker],actual_target_factory_reachable_before_truth=False,
 independent_assertions=len(checks),compact_source_metadata_hash_assertions=sum(x.startswith(('operational_exact_','scientific_exact_','audit_exact_'))for x in checks),material_assertions=sum(not x.startswith(('operational_exact_','scientific_exact_','audit_exact_'))for x in checks),independent_negative_fixtures=len(negative),checks=checks,negative_fixtures=negative,
 separate_author_reproduction=dict(checks_count=author['checks_count'],negative_count=author['negative_count'],result=author),
 requested_source_manifest_ref=ref(HERE/'source_manifest.v1.json'),requested_handoff_ref=ref(HERE/'source_handoff.v1.json'),
 omitted_frozen_accounting_metadata=skipped,no_actual_clock_process_resource_guard_accounting_science_or_Git_calls=True,
 source_metadata_files_read=len(checked),real_fit_speed_accuracy_or_numeric_parity_measured=False)
if '--emit-result'in sys.argv:
 with(REVIEW/'checks.final.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in ('checks','negative_fixtures','separate_author_reproduction')},sort_keys=True))

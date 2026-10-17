"""Author repair check: actual factory/code-auth/typed join reaches resolver.
All scientific modules/data are handwritten metadata doubles; a sentinel stops
before any truth payload IO. Author predecessor9groups/10negatives are separate.
"""
from pathlib import Path
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H060_global_only_full_training_source_20261017_v2'
checks=[]
def need(ok,message):
 if not ok:raise AssertionError(message)
def ck(name,value=True):need(value,name);checks.append(name)
def raw(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 need(p.is_file()and p.suffix in('.py','.json')and p.stat().st_size<=2**20,'Bounded source/metadata only');return p.read_bytes()
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw(p)).hexdigest())
ns=dict(__name__='_author_v2_source_fixture_reproduction',__file__=str(HERE/'source_checks_v1.py'))
exec(compile(raw(HERE/'source_checks_v1.py'),str(HERE/'source_checks_v1.py'),'exec',dont_inherit=True,optimize=0),ns)
ck('separate_inherited_author9groups10negatives_reproduced',ns['result']['checks_count']==9 and ns['result']['negative_count']==10)
a=ns['ns'];proto=ns['proto'];events=ns['events'];stored=ns['stored'];objects=ns['cache_objects'];mode=ns['mode'];mode[0]=None
original_guard=ns['guard'];original_writer=ns['Writer']
def reset():events.clear();stored.clear();objects.clear();mode[0]=None
comp=raw(ROOT/ns['b']['global_component_source_ref']['path']).decode()
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
except StopBeforeTruthPayload:pass
else:raise AssertionError('Actual repaired factory did not reach original resolver boundary')
ck('actual_original_target_factory_code_auth_typed_join_and_resolver_boundary_reached',resolved==[proto.TRAIN[0]] and 'completed_training.json'not in stored)
proto.make_actual_global_target_joiner_factory_v1=old_factory;proto.validate_target_protocol_binding_v1=old_validate;a['load_accepted_backbone_v1']=oldwarm

old_component='research/reviews/H060_global_observation_seal_source_20261017_v1'
new_component='research/reviews/H060_global_observation_seal_source_20261017_v2'
expected=raw(old_component+'/global_observation_protocol_v1.py').decode().replace(old_component,new_component).replace('actual.join.__closure__or()','actual.join.__closure__ or ()')
ck('component_diff_exactly_namespace_plus_closure_attribute_fix',comp==expected)
ck('no_scientific_framework_imports_or_payload_reads',not any(name in sys.modules for name in ('torch','numpy','torch_geometric','tskit')))
result=dict(schema='H060_global_only_minimal_repair_actual_factory_reachability_RAM_v2',status='PASS',source_only=True,independent=False,
 checks=checks,assertions=len(checks),inherited_author_groups=9,inherited_author_negatives=10,actual_runner_actual_new_factory_original_target_code_auth_and_typed_join_reach_resolver=True,
 stopped_before_truth_payload=True,numerical_parity_fit_speed_or_accuracy_measured=False,actual_operational_scientific_array_resource_guard_process_clock_or_Git_calls=False,
 repaired_component_source_ref=ns['b']['global_component_source_ref'],runner_ref=ref(HERE/'warmstart_train_v2.py'))
if '--emit-result'in sys.argv:
 with(HERE/'repair_factory_RAM.result.v2.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='checks'},sort_keys=True))

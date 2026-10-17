"""Independent source/RAM checks. No numerical/scientific/guard/process state execution."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace,ModuleType,FunctionType,CodeType
import ast,builtins,copy,dataclasses,difflib,hashlib,io,json,random,re,stat,sys,typing
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
D=ROOT/'research/reviews/H060_compact_barrier_independent_20261017_v1'
SRC=ROOT/'research/reviews/H060_global_only_compact_barrier_source_20261017_v1'
OLD=ROOT/'research/reviews/H060_global_only_full_training_source_20261017_v2'
PROTOCOL=ROOT/'research/reviews/H060_global_observation_seal_source_20261017_v2/global_observation_protocol_v1.py'
checks=[];negatives=[]
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def check(name,ok=True):need(ok,name);checks.append(name)
def reject(name,fn):
 try:fn()
 except (ValueError,AssertionError,KeyError,TypeError):negatives.append(name)
 else:raise AssertionError('Accepted negative '+name)
def raw(path):return Path(path).read_bytes()
def ref(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw(p)).hexdigest())
def defs(path,names,env):
 tree=ast.parse(raw(path));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',dont_inherit=True,optimize=0),env)
def module(path,name):
 env=dict(__name__=name,__file__=str(path))
 exec(compile(raw(path),str(path),'exec',dont_inherit=True,optimize=0),env)
 return env
handoff=json.loads(raw(SRC/'HANDOFF.v1.json'));check('exact_assigned_handoff',ref(SRC/'HANDOFF.v1.json')['sha256']=='7804de791f8a378522c224f496209a12768c951c1deded26cbe3ef6e58a2a405')
manifest=json.loads(raw(SRC/'MANIFEST.v1.json'));check('all15_manifest_hashes',len(manifest['files'])==15 and all(ref(r['path'])==r for r in manifest['files']))
pins=json.loads(raw(SRC/'scientific_source_pins.v2.json'))['refs'];oldpins=json.loads(raw(OLD/'scientific_source_pins.v2.json'))['refs']
check('all96_unique_source_pins_hash_exact',len(pins)==len({r['path']for r in pins})==96 and all(ref(r['path'])==r for r in pins))
check('predecessor_pins_exactly_preserved_plus_three_new',all(r in pins for r in oldpins) and len(pins)==len(oldpins)+3)
for r in pins:
 if Path(r['path']).suffix=='.py':compile(raw(ROOT/r['path']),str(ROOT/r['path']),'exec',dont_inherit=True,optimize=0)
oldtext=raw(OLD/'warmstart_train_v2.py').decode();newtext=raw(SRC/'warmstart_train_v3.py').decode()
check('literal_runner_patch_exact',''.join(difflib.unified_diff(oldtext.splitlines(True),newtext.splitlines(True),fromfile=str((OLD/'warmstart_train_v2.py').relative_to(ROOT)),tofile=str((SRC/'warmstart_train_v3.py').relative_to(ROOT))))==raw(SRC/'runner_delta.v1.patch').decode())
oldnodes={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(oldtext).body if isinstance(n,ast.FunctionDef)}
newnodes={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(newtext).body if isinstance(n,ast.FunctionDef)}
check('only_three_disclosed_function_deltas',{k for k in oldnodes if oldnodes[k]!=newnodes[k]}=={'_global_preparation_refs_v1','_signature_v1','run_train_v1'})
proto=module(PROTOCOL,'_independent_global_protocol');helper=module(SRC/'compact_barrier_v1.py','_independent_compact')
TRAIN=proto['TRAIN'];ARMS=('stable_leaf_memory','no_added_memory');sources=sorted([*proto['STATIC_SOURCE_REFS'],ref(PROTOCOL),ref(SRC/'compact_barrier_v1.py'),ref(SRC/'warmstart_train_v3.py')],key=lambda r:r['path'])
check('exact_global_preparation_source_closure',proto['checked_sources_v1'](sources)==sources)
plans={};planrefs={};seals={};trees=2500
def sym(label):return dict(path='research/reviews/RAM_ONLY_'+label+'.json',sha256=hashlib.sha256(label.encode()).hexdigest())
for i,p in enumerate(TRAIN):
 pr=sym('plan'+str(i));obs=sym('observed'+str(i));focal=sym('focal'+str(i));native=sym('native'+str(i));planrefs[p]=pr
 blocks=[dict(start_tree=k,stop_tree=k+500,nodes=800000,sites=0)for k in range(0,trees,500)]
 plan=dict(schema='american_c3_whole_tree_block_index_v1',parent_id=p,sequence_length_bp=2000000,haplotypes=1600,class_order=list(proto['CLASSES']),original_reference_counts=[200]*3,effective_reference_counts=[200]*3,focal_records=10000,all_trees_and_samples_retained=True,targets_opened=False,trees=trees,sites=0,blocks=blocks,plan_ref=pr,observation_ref=obs,focal_registry_ref=focal,config=dict(class_order=list(proto['CLASSES']),expected_haplotypes=1600,input_reference_class_mask=None,input_reference_sample_indices=None,allow_missing=False,donor_age_split=350.,max_trees_per_block=500,max_nodes_per_block=1000000,max_sites_per_block=1000000,native_preparation_ref=native))
 plans[p]=plan
 ident=dict(tensor_sha256='9'*64,tensor_bytes=123456,focal_identity_sha256='8'*64,trees=trees,sites=0,all_global_structural_recipes_before_truth=True)
 metadata=dict(schema='American_TreeLoom_full_observation_preparation_v1',parent_id=p,split='train',observation_only=True,observation_ref=obs,focal_registry_ref=focal,native_preparation_ref=native,focal_records=10000,focal_identity_sha256='8'*64,all_native_trees_nodes_sites_samples_retained=True,structural_recipes_global_before_partition=True,artificial_block_boundaries=False,private_truth_opened=False,model_executed=False,unicode_note='independent_λ',observation_metadata=dict(isolated_zero_site_holds=list(range(trees))),episode_metadata=dict(proper_clades_per_tree=list(range(trees)),native_nodes_per_tree=[1600]*trees,unary_equivalences_per_tree=[0]*trees),cut_metadata=dict(native_hold_cut_group_counts=[1]*trees,episode_starts=list(range(trees)),conditional_information_nats_per_leaf=[0.25]*trees),isolated_zero_site_holds=list(range(trees)))
 support=dict(schema='H060_observed_plan_support_without_legacy_features_v1',trees=trees,sites=0,nodes=trees*1600,blocks=len(blocks),plan_blocks_sha256=proto['json_sha'](blocks),focal_records=10000,every_focal_row_covered_exactly_once=True,original_query_sample_columns=True,all_original_block_counts_verified=True,legacy_neural_features_constructed=False)
 seals[p]=proto['make_global_seal_v1'](plan,plan_ref=pr,preparation_metadata=metadata,global_identity=ident,plan_support=support,source_refs=sources)
identities={p:seals[p]['global_identity']for p in TRAIN}
barrier=proto['make_all20_barrier_v1'](planrefs,seals,validated_cache_identities=identities);frozen=copy.deepcopy(barrier);fullsha=proto['json_sha'](barrier)
compact=helper['make_durable_barrier_v1'](barrier,full_barrier_sha256=fullsha,validate_full_barrier=proto['validate_barrier_v1'])
fullbytes=(json.dumps(barrier,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();compactbytes=(json.dumps(compact,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
check('full_original_barrier_and_every_array_unchanged',barrier==frozen)
check('streamed_digest_identical_to_existing_canonical_JSON',compact['full_original_barrier']==dict(sha256=fullsha,canonical_JSON_bytes=len(json.dumps(barrier,sort_keys=True,separators=(',',':'),allow_nan=False).encode())))
check('full_per_parent_SHA_source_refs_global_identities_and_support_retained',compact['source_refs']==sources and all(compact['parents'][p]['original_full_seal']['sha256']==proto['json_sha'](seals[p]) and compact['parents'][p]['global_identity']==identities[p] and compact['parents'][p]['plan_support']==seals[p]['plan_support']for p in TRAIN))
check('all_eight_structural_list_commitments_exact',all(len(compact['parents'][p]['structural_list_commitments'])==8 and all(x['length']==trees for x in compact['parents'][p]['structural_list_commitments'].values())for p in TRAIN))
check('distinct_compact_schema_and_fixture_bounds',compact['schema']!=proto['BARRIER_SCHEMA'] and len(fullbytes)>3145728 and len(compactbytes)<=262144)
reject('compact_not_accepted_by_original_full_barrier_validator',lambda:proto['validate_barrier_v1'](compact))
reject('wrong_full_SHA',lambda:helper['make_durable_barrier_v1'](barrier,full_barrier_sha256='f'*64,validate_full_barrier=proto['validate_barrier_v1']))
for name,mutate in (
 ('missing_original_parent',lambda b:b['parent_ids'].pop()),
 ('changed_source_identity',lambda b:b['parents'][TRAIN[-1]]['source_refs'][-1].update(sha256='0'*64)),
 ('changed_current_global_identity',lambda b:b['validated_cache_identities'][TRAIN[-1]].update(tensor_sha256='0'*64)),
 ('structural_value_changed_without_full_SHA',lambda b:b['parents'][TRAIN[-1]]['preparation_metadata']['episode_metadata']['proper_clades_per_tree'].__setitem__(-1,777777))):
 bad=copy.deepcopy(barrier);mutate(bad)
 reject(name,lambda bad=bad:helper['make_durable_barrier_v1'](bad,full_barrier_sha256=fullsha,validate_full_barrier=proto['validate_barrier_v1']))
big=copy.deepcopy(barrier)
for p in TRAIN:big['parents'][p]['plan_support']['independent_big_metadata']='x'*14000
reject('helper256KiB_bound_cannot_be_bypassed_by_valid_full_shape',lambda:helper['make_durable_barrier_v1'](big,full_barrier_sha256=proto['json_sha'](big),validate_full_barrier=proto['validate_barrier_v1']))
reject('nonfinite_canonical_JSON_denied',lambda:helper['canonical_commitment_v1'](dict(x=float('nan'))))
# Execute exact unchanged original writer with every OS/filesystem/guard operation replaced in RAM.
CAPS=json.loads(raw(ROOT/'research/reviews/H060_isolated_CPU_runtime_source_20261017_v1/caps.inactive.v1.json'))
FILES={};FDS={};opens=[];guard_writes=[];serial=[0]
class RP(PurePosixPath):
 def lstat(self):return meta(str(self))
def meta(p):return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_dev=1,st_ino=abs(hash(p)),st_size=len(FILES.get(p,b'')))
class RF(io.BytesIO):
 def __init__(self,fd):super().__init__();self.fd=fd
 def fileno(self):return self.fd
 def write(self,data):
  n=super().write(data);FILES[FDS[self.fd]]=self.getvalue();return n
class RAMOS:
 O_WRONLY=1;O_CREAT=2;O_EXCL=4;O_NOFOLLOW=8
 def open(self,path,flags,mode):
  path=str(path);need(path not in FILES,'Exclusive RAM name');serial[0]+=1;FDS[serial[0]]=path;FILES[path]=b'';opens.append(path);return serial[0]
 def fdopen(self,fd,mode):return RF(fd)
 def fsync(self,fd):pass
 def fstat(self,fd):return meta(FDS[fd])
 def statvfs(self,path):return SimpleNamespace(f_bavail=100000,f_frsize=4096)
class FakeTorch:
 float32='RAM_fp32'
 def device(self,name):return SimpleNamespace(type=name,index=None)
 def get_default_dtype(self):return self.float32
 def is_autocast_enabled(self):return False
 def save(self,value,buffer):checkpoints.append(copy.deepcopy(value));buffer.write(b'INDEPENDENT_RAM_ONLY_CHECKPOINT')
torch=FakeTorch();checkpoints=[]
wen=dict(Path=RP,ROOT=ROOT,io=io,json=json,re=re,stat=stat,hashlib=hashlib,os=RAMOS(),c=SimpleNamespace(CAPS=CAPS,require=need))
classes=[n for n in ast.parse(raw(OLD/'worker_entry_v1.py')).body if isinstance(n,ast.ClassDef) and n.name in ('CappedBuffer','BoundedWriter')]
exec(compile(ast.Module(body=classes,type_ignores=[]),str(OLD/'worker_entry_v1.py'),'exec'),wen)
events=[]
def writer(label):
 w=wen['BoundedWriter'].__new__(wen['BoundedWriter']);w.path=RP(str(D/'RAM_ONLY'/label));w.s=dict(storage='RAM_ONLY');w.used=8192;w.json_used=0;w.torch_files=0;w.names=set();w.fast=lambda event:events.append(('writer_boundary',event));w.torch=torch;w.jsonable=lambda x:x;w.g=SimpleNamespace(budget_check_v9=lambda *a,**k:guard_writes.append(k));return w
before=len(opens);reject('original_writer_rejects_large_full_without_partial_write',lambda:writer('large').save('TRAIN_observation_barrier.json',barrier));check('no_partial_RAM_write_for_oversize_full',len(opens)==before)
almost=writer('aggregate');almost.json_used=CAPS['total_JSON_bytes']-len(compactbytes)+1
reject('unchanged_aggregate3MiB_cap_applies_to_compact_write',lambda:almost.save('TRAIN_observation_barrier.json',compact));check('still_no_partial_write_after_aggregate_denial',len(opens)==before)
small=writer('compact');small.save('TRAIN_observation_barrier.json',compact)
check('unchanged_writer_actual_serialization_exactly_matches_helper_size',small.json_used==len(compactbytes))
# Actual unmodified global factory, original target-code authentication and original actual-path delegation.
targetpath=ROOT/'src/arggnn/american_c3_gnn_targets_v1.py';hookpath=ROOT/'research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/execution_hooks_prefetch_v1.py'
targets=ModuleType('arggnn.american_c3_gnn_targets_v1');targets.__file__=str(targetpath)
rows={p:dict(simulation_output_directory='RAM_ONLY/'+p)for p in TRAIN}
def registry_fixture(r):
 p=next(p for p in TRAIN if plans[p]['focal_registry_ref']==r)
 return dict(parent_id=SimpleNamespace(item=lambda:p),class_order=SimpleNamespace(tolist=lambda:list(proto['CLASSES'])),record_ids=range(10000))
targets.__dict__.update(REGISTRY_REF=proto['REGISTRY_REF'],require=need,authoritative_registry_v1=lambda r:copy.deepcopy(rows),registry=registry_fixture,identity_sha=lambda r:'8'*64,CLASSES=proto['CLASSES'],Callable=typing.Callable)
def selected(path,names):
 return [n for n in ast.parse(raw(path)).body if isinstance(n,ast.ImportFrom)and n.module=='__future__' or isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
targetcode=compile(raw(targetpath),str(targetpath),'exec',dont_inherit=True,optimize=0)
classcode=next(c for c in targetcode.co_consts if isinstance(c,CodeType)and c.co_name=='TrainTargetJoinV1')
classns={}
exec(classcode,targets.__dict__,classns)
targets.TrainTargetJoinV1=type('TrainTargetJoinV1',(),classns)
hooks=ModuleType('research.reviews.C3_GNN_prefetch_isolated980_root_20261015_v2.execution_hooks_prefetch_v1');hooks.__file__=str(hookpath);constants={}
for n in ast.parse(raw(hookpath)).body:
 if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
  try:constants[n.targets[0].id]=ast.literal_eval(n.value)
  except (ValueError,TypeError):pass
bindings={p:dict(inferred_observation_ref=plans[p]['observation_ref'],simulation_completion_ref=sym('completion'+p),simulation_config_ref=sym('config'+p))for p in TRAIN}
actualmeta={p:dict(binding=bindings[p],simulation_directory='RAM_ONLY/actual'+p,private_truth_ref=sym('STOP_BEFORE_TRUTH'+p))for p in TRAIN}
def hook_import(name,globals=None,locals=None,fromlist=(),level=0):
 if name==targets.__name__:return targets
 return builtins.__import__(name,globals,locals,fromlist,level)
hooks.__dict__.update(constants,require=need,TRAIN=TRAIN,actual_train_metadata_v1=lambda b:copy.deepcopy(actualmeta),copy=copy,ref=ref,dataclasses=dataclasses,__builtins__=dict(vars(builtins),__import__=hook_import))
hookcode=compile(raw(hookpath),str(hookpath),'exec',dont_inherit=True,optimize=0)
factorycode=next(c for c in hookcode.co_consts if isinstance(c,CodeType)and c.co_name=='make_actual_target_joiner_factory_v1')
hooks.make_actual_target_joiner_factory_v1=FunctionType(factorycode,hooks.__dict__,'make_actual_target_joiner_factory_v1')
def component_import(name,globals=None,locals=None,fromlist=(),level=0):
 if name=='importlib':return SimpleNamespace(import_module=lambda n:targets if n==targets.__name__ else hooks if n==hooks.__name__ else need(False,'Unexpected scientific import'))
 return builtins.__import__(name,globals,locals,fromlist,level)
proto['__builtins__']=dict(vars(builtins),__import__=component_import)
exec(compile(raw(PROTOCOL),str(PROTOCOL),'exec',dont_inherit=True,optimize=0),proto)
class StopBeforeTruth(Exception):pass
resolved=[]
def sentinel(p):resolved.append(p);raise StopBeforeTruth(p)
factory=proto['make_actual_global_target_joiner_factory_v1'](bindings,barrier=barrier,source_refs=sources,runtime_guard=lambda e:events.append(('actual_factory',e)))
oldbinding=hooks.make_actual_target_joiner_factory_v1(bindings).execution_binding
proto['validate_target_protocol_binding_v1'](factory.execution_binding,oldbinding,barrier=barrier)
joiner=factory(sentinel,max_truth_uncompressed_bytes=100000)
for p in TRAIN:
 try:joiner.join(p,split='train',plan=plans[p],observation_seal=seals[p])
 except StopBeforeTruth:pass
 else:raise AssertionError('Payload boundary not stopped')
check('actual_factory_code_auth_original_actual_path_and_all20_original_gates_reach_only_sentinel',resolved==list(TRAIN))
reject('VAL_membership_rejected_before_resolver',lambda:joiner.join('american_c3_val_000',split='val',plan=plans[TRAIN[0]],observation_seal=seals[TRAIN[0]]))
changedplan=copy.deepcopy(plans[TRAIN[0]]);changedplan['blocks'][0]['nodes']+=1
reject('changed_native_support_plan_rejected_before_resolver',lambda:joiner.join(TRAIN[0],split='train',plan=changedplan,observation_seal=seals[TRAIN[0]]))
check('negative_target_cases_never_reach_resolver',resolved==list(TRAIN))
# Exact future runner body with RAM cache/model/optimizer/truth providers, real helper and original pure gates.
counts={a:0 for a in ARMS};truth=[];cachechecks=[];guards=[];prepared=[];barrier_writes=[]
class Bundle:
 def __init__(self,p):self.parent_id=p
 def to(self,device):return self
class State:
 def state_dict(self):return {}
 def checkpoint_contract(self):return dict(RAM_ONLY=True)
def fake_factory(actual_bindings,*,barrier,source_refs,runtime_guard):
 need('TRAIN_observation_barrier.json' in w.names,'Durable commitment before target factory')
 need(barrier['schema']==proto['BARRIER_SCHEMA'],'Full original barrier to factory')
 proto['validate_barrier_v1'](barrier);barrier_writes.append(barrier)
 class Join:
  execution_binding=dict(factory.execution_binding)
  def __init__(self,resolve_train_truth,**kw):self.resolve=resolve_train_truth
  def join(self,p,*,split,plan,observation_seal):
   proto['global_observation_gate_v1'](p,split=split,plan=plan,seal=observation_seal,barrier=barrier)
   self.resolve(p);return SimpleNamespace(receipt=dict(parent_id=p,records=10000))
 return Join
def prepare(pr,cfg,runtime_guard,source_refs):
 p=next(p for p in TRAIN if planrefs[p]==pr);prepared.append(p);return Bundle(p),seals[p]
def admit(cache,p,b,seal,*,current_bytes,maximum_bytes,runtime_guard):cache[p]=dict(bundle=b,charge=dict(accounted_bytes=1));return current_bytes+1
def step(a,m,b,target,opt,**kw):
 counts[a]+=1
 return dict(arm=a,parent_id=b.parent_id,records=10000,whole_parent_optimizer_updates=1,block_reset_or_truncated_BPTT=False,first_real_update_checks=dict(RAM_ONLY=True),complete_parent_step_wall_seconds_including_checks=0.)
def epoch_order(e):
 values=list(TRAIN);random.Random(70002+e).shuffle(values);return values
gp=SimpleNamespace(**proto);gp.prepare_and_seal_global_parent_v1=prepare;gp.make_actual_global_target_joiner_factory_v1=fake_factory
en=dict(Path=Path,copy=copy,json=json,hashlib=hashlib,stat=stat,HERE=SRC,SOURCE=str((SRC/'warmstart_train_v3.py').relative_to(ROOT)),json_sha=proto['json_sha'],global_protocol=gp,compact_barrier=SimpleNamespace(**helper),TRAIN=TRAIN,ARMS=ARMS,FAMILY='American_TreeLoom_saved600_warmstart_matched_v1',CONDITION='accepted_C3_saved600_backbone_plus600_new_native_focal_updates_per_arm',PARAMETERS={ARMS[0]:32132,ARMS[1]:31619},REGISTRY_REF=proto['REGISTRY_REF'],SCHEMA='American_TreeLoom_saved600_warmstart_matched_v1_global_observation_training_v1',CHECKPOINT_EPOCHS=(5,10,15,20,25,30),torch=torch,time=SimpleNamespace(monotonic=lambda:0.),require=need,epoch_order_v1=epoch_order,_cpu_snapshot=lambda x:copy.deepcopy(x),_rng_state=lambda d:{})
defs(SRC/'warmstart_train_v3.py',('_source_snapshot_v2','_source_check_v2','_global_preparation_refs_v1','_signature_v1','_snapshot_v1','_progress_v1','run_train_v1'),en)
en['ref']=ref
snapshot=en['_source_snapshot_v2']();en['_source_check_v2'](snapshot);check('original_source_snapshot_and_identity_check_pass_on_all96_source_refs',len(snapshot)==96)
check('actual_runner_source_selection_includes_exact_helper_and_runner',en['_global_preparation_refs_v1']()==sources)
en.update(validate_config_v1=lambda c:None,authoritative_registry_v1=lambda:None,load_accepted_backbone_v1=lambda *a,**k:({}, {},planrefs,oldbinding),_source_snapshot_v2=lambda:('RAM_snapshot',),_source_check_v2=lambda s:cachechecks.append(s),_validated_global_cache_identities_v1=lambda *a:identities,_seed_fresh=lambda d:None,make_matched_family_v1=lambda *a,**k:({a:State()for a in ARMS},{a:State()for a in ARMS},{}),read=lambda r:next(plans[p]for p in TRAIN if planrefs[p]==r),_admit_global_parent_v1=admit,_cache_get_v2=lambda cache,p:cache[p]['bundle'],train_complete_parent_step_v1=step)
w=writer('runner');w.save('started.json',dict(status='RAM_ONLY'))
cfg=dict(execution_enabled=True,max_cached_parent_bytes=103079215104,max_truth_uncompressed_bytes=100000)
en['run_train_v1'](cfg,parent_plan_refs=planrefs,artifact_acceptance_ref='RAM_ONLY',writer=w,device='cpu',worker_tmp_directory='RAM_ONLY',resolve_train_truth=lambda p:truth.append(p),runtime_guard=lambda e:guards.append(e),actual_train_bindings=bindings,resume_ref=None)
completed=json.loads(FILES[str(w.path/'completed_training.json')]);durable=json.loads(FILES[str(w.path/'TRAIN_observation_barrier.json')])
check('exact_runner_full20_prepare_and_gate_then1200_arm_calls',prepared==list(TRAIN) and truth==list(TRAIN) and counts=={a:600 for a in ARMS})
check('all_six_actual_snapshot_checkpoint_controls_preserved',w.torch_files==6 and [x['candidate']['completed_epoch']for x in checkpoints]==[5,10,15,20,25,30] and all(en['_progress_v1'](x['trainer_state'])==e*20 for x,e in zip(checkpoints,(5,10,15,20,25,30))))
check('checkpoint_and_completion_reference_compact_file_but_full_canonical_SHA',completed['observation_seal_sha256']==fullsha==durable['full_original_barrier']['sha256'] and all(x['trainer_state']['observation_seal_sha256']==fullsha and x['trainer_state']['protocol']['observation_protocol']['durable_barrier_schema']==helper['SCHEMA'] and x['trainer_state']['observation_manifest_ref']==completed['observation_manifest_ref']for x in checkpoints))
check('all60_pre_VALtruth_and_original30_prediction_flags_unchanged',completed['all60_matched_prediction_seals_required_before_any_VAL_truth']is True and completed['total_new_prediction_seals_required']==60 and completed['existing_C3_all30_barrier_unchanged']is True and completed['either_arm_validation_truth_or_checkpoint_choice_before60_seals']is False)
check('writer_all_original_caps_remain_and_fixture_route_fits',CAPS['reservation_bytes']==33554432 and CAPS['total_JSON_bytes']==3145728 and CAPS['checkpoint_serialized_bytes']==3145728 and CAPS['output_bytes']==26214400 and CAPS['maximum_torch_files']==7 and w.json_used<=3145728 and w.used<=26214400)
check('no_scientific_frameworks_or_payloads_imported',not any(n in sys.modules for n in ('torch','numpy','torch_geometric','tskit')))
result=dict(schema='H060_compact_barrier_independent_source_RAM_checks_v1',status='PASS',checks=checks,positive_count=len(checks),negative_cases=negatives,negative_count=len(negatives),source_manifest_count=len(manifest['files']),scientific_pins=len(pins),global_preparation_refs=len(sources),fixture_only_sizes=dict(full_pretty_JSON_bytes=len(fullbytes),compact_pretty_JSON_bytes=len(compactbytes),RAM_runner_aggregate_JSON_bytes=w.json_used,RAM_runner_accounted_output_bytes=w.used),RAM_update_calls=counts,actual_factory_sentinel_resolver_parents=len(resolved),whole_ROOT_route_qualified=False,actual_operations=False,numerical_training_or_fit_claim=False,source_ref=ref(SRC/'HANDOFF.v1.json'))
with (D/'independent_checks.result.v1.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(result,sort_keys=True))

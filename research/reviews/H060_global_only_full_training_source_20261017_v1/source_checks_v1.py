"""Source/metadata/RAM fixtures. No scientific/guard imports or operational calls.
All arrays, observations, targets, models, writers, process APIs, resources,
mutexes, mutable files and accounting operations are handwritten RAM doubles.
"""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace,ModuleType
from contextlib import contextmanager
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE=ROOT/'research/reviews/H060_global_only_full_training_source_20261017_v1'
OLD=ROOT/'research/reviews/H060_legacy_fast_full_training_source_20261017_v1';ORIGINAL=ROOT/'research/reviews/H060_isolated_CPU_runtime_source_20261017_v1'
checks=[];negative=[]
def need(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def funcs(p):return {n.name:n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef)}
def execnodes(nodes,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),'<isolated-authenticated-source-AST>','exec',dont_inherit=True,optimize=0),scope)
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def reject(label,call):
 try:call()
 except (ValueError,KeyError,TypeError,AttributeError,OSError):negative.append(label);return
 raise ValueError('Negative accepted: '+label)
def js(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
SYMBOLIC={}
def sym(label):
 path='research/reviews/SOURCE_ONLY_RAM/'+label+'.json';SYMBOLIC[path]=label.encode();return dict(path=path,sha256=hashlib.sha256(label.encode()).hexdigest())
runner=funcs(HERE/'warmstart_train_v2.py');oldrunner=funcs(OLD/'warmstart_train_v2.py');common=funcs(HERE/'root_driver_v1.py')
b=read(HERE/'scientific_binding.v1.json');bindings=read(HERE/'source_bindings.v1.json');caps=read(HERE/'caps.inactive.v1.json')
for p in HERE.glob('*.py'):compile(p.read_bytes(),str(p),'exec',dont_inherit=True,optimize=0)
for n in ('load_accepted_backbone_v1','_cache_tensors_v2','_cache_semantics_v2','_cache_charge_v2','_cache_admit_v2','_cache_get_v2','_source_snapshot_v2','_source_check_v2','_progress_v1'):
 need(same(runner[n],oldrunner[n]),'Exact unchanged scientific/cache helper '+n)
need('_resume_v1'not in runner,'Old transfer resume implementation removed')
text=(HERE/'warmstart_train_v2.py').read_text()
need('SealedObservationProviderV1'not in text and 'seal_then_prepare_one_parent_v1'not in text and 'providers'not in text,'No old provider/legacy construction')
# Exact scientific update loop except obsolete provider release statement.
def update_loop(fn):return next(n for n in ast.walk(fn)if isinstance(n,ast.For)and ast.unparse(n.target)=='index'and ast.unparse(n.iter)=='range(step, 600)')
a=copy.deepcopy(update_loop(oldrunner['run_train_v1']));a.body=[n for n in a.body if ast.unparse(n)!='providers[parent].release()']
need(same(a,update_loop(runner['run_train_v1'])),'Entire original600pair/1200arm step/checkpoint/history loop')
for n in ('driver','execute_or_admit','consumer_for_admitted','append_selected_source'):
 need(same(common[n],funcs(OLD/'root_driver_v1.py')[n]),'Exact common operational function '+n)
# Worker body preserved modulo narrow factory/entry changes, exact bounded writer and serializer.
w=ast.parse((HERE/'worker_entry_v1.py').read_text());ow=ast.parse((ORIGINAL/'worker_entry_v1.py').read_text())
for name in ('CappedBuffer','BoundedWriter','jsonable_runtime_v1'):
 find=lambda t:next(n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name)
 need(same(find(w),find(ow)),'Original actual worker '+name)
old=copy.deepcopy(funcs(ORIGINAL/'worker_entry_v1.py')['worker']);new=funcs(HERE/'worker_entry_v1.py')['worker']
class WorkerNormalize(ast.NodeTransformer):
 def visit_Assign(self,n):
  if ast.unparse(n)=='factory = hooks.make_actual_target_joiner_factory_v1(ctx[\'actual_train_bindings\'])':return None
  return self.generic_visit(n)
 def visit_Call(self,n):
  n=self.generic_visit(n)
  for k in n.keywords:
   if k.arg=='train_target_joiner_factory':k.arg='actual_train_bindings';k.value=ast.parse("ctx['actual_train_bindings']",mode='eval').body
  return n
need(same(WorkerNormalize().visit(old),new),'Actual worker original gate/runtime/import/model call exact modulo factory deferral/bindings')
checks.append('source_AST_original_math_warmstart_cache_600pair_update_loop_worker_gate_runtime_writer_serializer_preserved')
need(caps==read(OLD/'caps.inactive.v1.json')==bindings['consumers']['H060']['caps'],'Exact every original cap')
for n in ('ROOT_request.INACTIVE.template.v1.json','ROOT_directive.INACTIVE.template.v1.json'):
 d=read(HERE/n);need(d['active']is False and d['root_authorizer']is None and d['created_at_utc']is None and d['expires_at_utc']is None and d['scientific_config']['execution_enabled']is False,'Inactive authority only')
need(read(HERE/'runtime_qualification.INACTIVE.template.v1.json')['status']=='inactive','No fabricated qualification')
checks.append('caps_32CPU_zeroGPU_96GiBcache_128GiBRSS_whole32MiB_and_templates_inactive')
def first_party_import_closure(entry):

    def resolve(module):
        for folder in (ROOT / 'src', ROOT):
            p = folder / (module.replace('.', '/') + '.py')
            if p.is_file():
                return p
            p = folder / module.replace('.', '/') / '__init__.py'
            if p.is_file():
                return p
        return None
    pending = [entry]
    seen = set()
    while pending:
        path = pending.pop()
        key = str(path.relative_to(ROOT))
        if key in seen:
            continue
        seen.add(key)
        module = key.removesuffix('.py')
        if module.startswith('src/'):
            module = module[4:]
        parts = module.split('/')
        package = parts[:-1]
        for at in range(1, len(package) + 1):
            p = resolve('.'.join(package[:at]))
            if p is not None and p.name == '__init__.py':
                pending.append(p)
        for n in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                prefix = (package[:len(package) - n.level + 1] if n.level else []) + ((n.module or '').split('.') if n.module else [])
                base = '.'.join(prefix)
                names = [base] + [base + '.' + a.name for a in n.names]
            for name in names:
                p = resolve(name)
                if p is not None:
                    pending.append(p)
    return seen
closure=first_party_import_closure(HERE/'warmstart_train_v2.py')
pinned={r['path']for r in read(HERE/'scientific_source_pins.v2.json')['refs']}
need(closure<=pinned,'Actual first-party import closure fully pinned: '+repr(sorted(closure-pinned)))
checks.append('complete_actual_runner_first_party_import_closure_pinned_'+str(len(closure)))
# Current source-only selector metadata functions, no guard configured/imported.
selector={};exec(compile((ROOT/bindings['selector_bridge_ref']['path']).read_bytes(),'<accepted-Census-selector-source>','exec'),selector)
c=SimpleNamespace(require=need,raw=lambda r:(ROOT/r['path']).read_bytes(),read=lambda r:read(ROOT/r['path']),ref=ref)
selector['install_v1'](c,bindings);chosen=c.bind_selector(bindings['actual_selector_metadata_ref'])
need(chosen['source_ref']==bindings['selected_child_source_ref'],'Exact current Census selector join')
reject('retired_selector_metadata',lambda:c.selected(read(OLD/'source_bindings.v1.json')['actual_selector_metadata_ref']))
reject('selector_lifetime_swap',lambda:c.bind_selector(sym('wrong-selector')))
# Genuine47roles/40refs plus original roles through actual selected pure64cap API.
fixture=read(HERE/'role_union_fixture.v1.json');guards=list(fixture['actual_guard_role_refs_by_name'].values());deps=read(ORIGINAL/'dependencies.v1.json')
need(len(guards)==47 and len({js(r)for r in guards})==40,'Current47named40distinct guard roles')
rolescope=dict(binding=lambda:b,ref=lambda p:sym(p.replace('/','_')),MANIFEST='manifest',SOURCE='contract_driver',DRIVER='contract_driver',ENTRY='entry',HANDOFF='handoff',DEPS=deps,NAMESPACE=deps['NAMESPACE'],PIDFD=deps['PIDFD'],OWNED=deps['OWNED'],SUPERVISOR=deps['SUPERVISOR'])
execnodes([funcs(ORIGINAL/'root_contract_v1.py')['direct_roles']],rolescope)
q={k:sym(k)for k in ('source_review_ref','scientific_review_ref','runtime_ref','runtime_qualification_ref','root_directive_ref','registration_ref','registration_decisions_ref','training_context_ref')};q['storage_policy_selector_ref']=bindings['actual_selector_metadata_ref']
roles=rolescope['direct_roles'](sym('request'),q,sym('config'),sym('closure'))
scope=dict(need=need,copy=copy);execnodes([common['append_selected_source']],scope);roles=scope['append_selected_source'](roles,bindings['selected_child_source_ref'],bindings['selected_child_source_ref'])
api=dict(require=need,copy=copy,_raw=lambda r:None);execnodes([funcs(ROOT/bindings['selected_child_source_ref']['path'])['consumer_bindings_v1']],api)
union=api['consumer_bindings_v1'](guards,roles);need(len(union)<=64 and all(r in union for r in guards),'All current roles retained')
reject('pure_guard_65_direct_refs',lambda:api['consumer_bindings_v1'](union,[sym('overflow'+str(i))for i in range(65-len(union))]))
checks.append('actual_Census_metadata_selector_all47roles_direct_union_'+str(len(union))+'_original64cap')
# Actual runner through all20 barrier/factory/targets and full600paired metadata loop.
protocol={};exec(compile((ROOT/b['global_component_source_ref']['path']).read_bytes(),'<typed-global-stdlib-only-source>','exec'),protocol)
proto=SimpleNamespace(**protocol);events=[];stored={};plans={};seals={};cache_objects={};mode=[None]
for at,parent in enumerate(proto.TRAIN):
 plan=dict(schema='american_c3_whole_tree_block_index_v1',parent_id=parent,sequence_length_bp=2000000,class_order=list(proto.CLASSES),haplotypes=1600,original_reference_counts=[200]*3,effective_reference_counts=[200]*3,focal_records=10000,all_trees_and_samples_retained=True,targets_opened=False,trees=1,sites=0,observation_ref=sym(parent+'_observation'),focal_registry_ref=sym(parent+'_focal'),blocks=[dict(start_tree=0,stop_tree=1,nodes=1,sites=0)],config=dict(class_order=list(proto.CLASSES),expected_haplotypes=1600,input_reference_class_mask=None,input_reference_sample_indices=None,allow_missing=False,donor_age_split=350.,max_trees_per_block=1,max_nodes_per_block=1,max_sites_per_block=1,native_preparation_ref=sym(parent+'_native')))
 plans[parent]=dict(plan,plan_ref=sym(parent+'_plan'))
 identity=dict(tensor_sha256=hashlib.sha256(parent.encode()).hexdigest(),tensor_bytes=1,focal_identity_sha256='f'*64,trees=1,sites=0,all_global_structural_recipes_before_truth=True)
 metadata=dict(schema='American_TreeLoom_full_observation_preparation_v1',parent_id=parent,split='train',observation_only=True,focal_records=10000,focal_identity_sha256='f'*64,all_native_trees_nodes_sites_samples_retained=True,structural_recipes_global_before_partition=True,artificial_block_boundaries=False,private_truth_opened=False,model_executed=False,**{k:plan[k]for k in ('observation_ref','focal_registry_ref')},native_preparation_ref=plan['config']['native_preparation_ref'])
 support=dict(schema='H060_observed_plan_support_without_legacy_features_v1',trees=1,sites=0,nodes=1,blocks=1,plan_blocks_sha256=proto.json_sha(plan['blocks']),focal_records=10000,every_focal_row_covered_exactly_once=True,original_query_sample_columns=True,all_original_block_counts_verified=True,legacy_neural_features_constructed=False)
 seals[parent]=proto.make_global_seal_v1(plan,plan_ref=plans[parent]['plan_ref'],preparation_metadata=metadata,global_identity=identity,plan_support=support,source_refs=b['global_preparation_source_refs'])
class Bundle:
 def __init__(self,parent):
  p=plans[parent];self.parent_id=parent;self.split='train';self.sequence_length_bp=2000000;self.native_holds=1;self.original_sites=0;self.focal_identity_sha256='f'*64
  self.observation_ref=p['observation_ref'];self.focal_registry_ref=p['focal_registry_ref'];self.native_preparation_ref=p['config']['native_preparation_ref'];self.cut=SimpleNamespace(group_offsets=(0,1),episode_of_hold=(0,));self.version=0;self.token=object()
 def validate(self):need(self.version==0,'Metadata bundle integrity changed');return self
 def to(self,device):return self
class Writer:
 def save(self,name,value,kind='json'):
  events.append(('save',name));stored[name]=copy.deepcopy(value);return sym(name)
oldtarget=dict(registry_ref=proto.REGISTRY_REF,target_equations_unchanged=True,source_refs=[])
planrefs={p:plans[p]['plan_ref']for p in proto.TRAIN}
def prepare(planref,*args,**kwargs):
 parent=next(p for p,r in planrefs.items()if r==planref);events.append(('prepare',parent));bundle=Bundle(parent);cache_objects[parent]=bundle;return bundle,copy.deepcopy(seals[parent])
def factory(bind,*,barrier,source_refs,runtime_guard):
 need('TRAIN_observation_barrier.json'in stored and (len([r for r in events if r[0]=='integrity'])==20 or [r[1]for r in events if r[0]=='cache_get'][:20]==list(proto.TRAIN)),'Factory before durable20verified cache barrier')
 events.append(('factory',None));proto.validate_barrier_v1(barrier)
 class Join:
  execution_binding=dict(schema=proto.TARGET_SCHEMA,registry_ref=proto.REGISTRY_REF,original_actual_target_binding=oldtarget,component_source_ref=b['global_component_source_ref'],source_refs=source_refs,global_observation_barrier_sha256=proto.json_sha(barrier),global_seal_schema=proto.SEAL_SCHEMA,target_equations_unchanged=True,actual_target_paths_unchanged=True,legacy_block_seal_fabricated=False,original_observation_protocol_or_checkpoint_identity_claimed=False)
  def __init__(self,resolver,**kw):self.resolve=resolver
  def join(self,parent,*,split,plan,observation_seal):
   proto.global_observation_gate_v1(parent,split=split,plan=plan,seal=observation_seal,barrier=barrier);self.resolve(parent);events.append(('target',parent));return SimpleNamespace(receipt={'parent_id':parent})
 return Join
proto.prepare_and_seal_global_parent_v1=prepare;proto.make_actual_global_target_joiner_factory_v1=factory
proto.checked_sources_v1=lambda rows:rows # Real metadata source pins are checked separately below.
def guard(event):
 if event=='before_all20_original_cache_integrity_check':
  events.append(('integrity',len([e for e in events if e[0]=='integrity'])))
  if mode[0]=='last_cache_drift'and len([e for e in events if e[0]=='integrity'])==20:cache_objects[proto.TRAIN[-1]].version=1
 return {}
ns=dict(require=need,copy=copy,json_sha=proto.json_sha,TRAIN=proto.TRAIN,ARMS=('stable_leaf_memory','no_added_memory'),REGISTRY_REF=proto.REGISTRY_REF,global_protocol=proto,torch=SimpleNamespace(device=lambda d:SimpleNamespace(type=d,index=None),get_default_dtype=lambda:'fp32',float32='fp32',is_autocast_enabled=lambda:False),authoritative_registry_v1=lambda:None,validate_config_v1=lambda cfg:None,load_accepted_backbone_v1=lambda *a,**k:({}, {},planrefs,oldtarget),_source_snapshot_v2=lambda:(),_source_check_v2=lambda s:None,_global_preparation_refs_v1=lambda:b['global_preparation_source_refs'],full=SimpleNamespace(memory_core=SimpleNamespace(_stamps=lambda tokens:tuple((id(t[0]),t[1])for t in tokens))),_cache_tensors_v2=lambda bundle:((bundle.token,bundle.version),),_cache_charge_v2=lambda bundle:dict(accounted_bytes=1),read=lambda r:next({k:v for k,v in p.items()if k!='plan_ref'}for p in plans.values()if p['plan_ref']==r),time=SimpleNamespace(monotonic=lambda:0.),epoch_order_v1=lambda epoch:list(proto.TRAIN),_seed_fresh=lambda d:None,make_matched_family_v1=lambda *a,**k:({a:{}for a in ('stable_leaf_memory','no_added_memory')},{a:{}for a in ('stable_leaf_memory','no_added_memory')},{}),train_complete_parent_step_v1=lambda arm,*a,**k:dict(arm=arm,first_real_update_checks={},complete_parent_step_wall_seconds_including_checks=0),_snapshot_v1=lambda *a,**k:dict(trainer_state={'additional_updates_per_arm':k['step']}),CHECKPOINT_EPOCHS=(5,10,15,20,25,30),FAMILY='family',CONDITION='condition',PARAMETERS={},SCHEMA='new_schema')
execnodes([runner[n]for n in ('_cache_semantics_v2','_cache_admit_v2','_cache_get_v2','_admit_global_parent_v1','_validated_global_cache_identities_v1','run_train_v1')],ns)
ns['_signature_v1']=lambda *a:({'observation_protocol':{'new':True}},'s'*64)
cfg=dict(max_cached_parent_bytes=96*2**30,max_truth_uncompressed_bytes=32000000)
def run():return ns['run_train_v1'](cfg,parent_plan_refs=planrefs,artifact_acceptance_ref=sym('accepted600'),writer=Writer(),device='cpu',worker_tmp_directory='RAM',resolve_train_truth=lambda p:need('TRAIN_observation_barrier.json'in stored,'Truth before barrier'),runtime_guard=guard,actual_train_bindings={p:{}for p in proto.TRAIN},resume_ref=None)
run();need(len([e for e in events if e[0]=='prepare'])==20 and len([e for e in events if e[0]=='target'])==20,'Complete original20 preparation and joins')
result=stored['completed_training.json'];need(result['additional_updates_per_arm']==600 and result['new_saved_logical_optimizer_updates_total']==1200 and result['all60_matched_prediction_seals_required_before_any_VAL_truth']is True and all(len(rows)==6 for rows in result['checkpoint_candidates'].values()),'Full600each/1200/six/checkpoints/60prediction metadata')
events.clear();stored.clear();cache_objects.clear();mode[0]='last_cache_drift';reject('last_admitted_cache_corruption_pretruth',run)
need(not any(e[0]in('factory','target')for e in events)and'TRAIN_observation_barrier.json'not in stored,'Last cache corruption cannot reach barrier/factory/targets')
reject('old_transfer_resume_early',lambda:ns['run_train_v1'](cfg,parent_plan_refs=planrefs,artifact_acceptance_ref=sym('accepted600'),writer=Writer(),device='cpu',worker_tmp_directory='RAM',resolve_train_truth=lambda p:None,runtime_guard=guard,actual_train_bindings={},resume_ref=sym('old-checkpoint')))
checks.append('actual_runner_RAM_original20cache_checks_before_durable_newbarrier_factory_targets_full600pairs_and_corrupt20th_rejected')
# Actual accepted memo2 constructor and original state/reserve public methods, handwritten operational backend.
memo=ModuleType('RAM_memo2');exec(compile((ROOT/bindings['ROOT_memo_source_ref']['path']).read_bytes(),'<accepted-memo2-stdlib-source>','exec'),memo.__dict__)
bridge={};exec(compile((ROOT/bindings['ROOT_memo_bridge_ref']['path']).read_bytes(),'<accepted-Census-memo-bridge>','exec'),bridge)
class MemoryPath(PurePosixPath):
 def exists(self):return False
 @classmethod
 def cwd(cls):return cls('/workspace-SR003.nfs2/ARG-GNN')
memroot=MemoryPath(str(ROOT));memo.ROOT=memroot;memo.Path=MemoryPath;memo.threading=SimpleNamespace(active_count=lambda:1);memo.sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=True,optimize=0))
locks=[False];op=[];mem={};mrefs={};mutation=[None]
def memory_put(p,v):
 key=str(p.relative_to(memroot));data=(json.dumps(v,sort_keys=True)+'\n').encode();mem[key]=data;mrefs[key]=dict(path=key,sha256=hashlib.sha256(data).hexdigest());return mrefs[key]
def mread(r):
 if r['path']in mem:need(hashlib.sha256(mem[r['path']]).hexdigest()==r['sha256'],'RAM exact mutable metadata SHA');return json.loads(mem[r['path']])
 return read(ROOT/r['path'])
def filebytes(root,name):
 key=str(MemoryPath(name).relative_to(memroot))if MemoryPath(name).is_absolute()else str(name)
 if key in mem:data=mem[key]
 elif key in SYMBOLIC:data=SYMBOLIC[key]
 else:data=(ROOT/key).read_bytes()
 return data,hashlib.sha256(data).hexdigest()
def canonical(root,name):
 p=MemoryPath(name);return p if p.is_absolute()else root/p
v4=ModuleType('RAM_v4');v4.__dict__.update(Path=MemoryPath,file_bytes=filebytes,canonical=canonical,parse=json.loads,integer=lambda v,*a:need(type(v)is int and v>0,'Positive metadata integer'),paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,accounting_paths=lambda r:[],encoded=lambda v:json.dumps(v,sort_keys=True).encode(),LEGACY='.runtime/RAM_legacy.json')
execnodes([funcs(ROOT/'src/arggnn/project_budget_v4.py')[n]for n in ('bound','reference','scope_spec','read_json')],v4.__dict__)
budget=ModuleType('RAM_budget');budget.__dict__.update(v4=v4,JOURNAL_REL='.runtime/RAM_journal',SOURCE_REL='scripts/project_budget_v9.py',LAUNCHER_REL='scripts/launch_reserved_gpu_v6.py')
execnodes([funcs(ROOT/'scripts/project_budget_v9.py')['successor_paths']],budget.__dict__)
orig={k:getattr(v4,k)for k in ('bound','reference','scope_spec','canonical','file_bytes')};origsuccessor=budget.successor_paths
@contextmanager
def mutex(root):
 need(not locks[0],'No caller-held/nested mutex');locks[0]=True;op.append('lock')
 try:yield
 finally:
  need(all(getattr(v4,k)is value for k,value in orig.items())and budget.successor_paths is origsuccessor,'Restore before unlock');locks[0]=False;op.append('unlock')
v4.mutex=mutex
implementation=ref(budget.SOURCE_REL);launcher=ref(budget.LAUNCHER_REL);g=ModuleType('RAM_Census_guard');state=({},set(),dict(prefix='RAM_frozen_prefix',new_reserved_bytes=0,completed_credit_bytes=0,base_pins={}),fixture['actual_guard_role_refs_by_name'],[],[],{})
def state_locked(*args,**kwargs):need(locks[0]and kwargs==dict(fresh=True),'Actual public state owns original separate lock');op.append('state');return state
def authority(record,*args,**kwargs):
 d=mread(record);need(d['active']is True,'Active RAM authority');return d
def append(operation,record,value,current):
 need(locks[0]and operation=='reserve','Original append under original lock');op.append('append');need(v4.file_bytes is orig['file_bytes'],'Raw primitive unchanged');return 'RAM_append_tail'
g.__dict__.update(budget=budget,ROOT=memroot,SOURCE=bindings['selected_child_source_ref']['path'],require=need,_state_locked=state_locked,_authority=authority,_pins=lambda v:v['prefix'],_paths=lambda value:budget.successor_paths(memroot,value),_ref=ref,_policy=lambda:dict(never_credit_keys=[],base_projected_bytes=0),charge=lambda base,reserved,credit,count:reserved,copy=copy,_append=append,warning_fields=lambda value:{},MODE='Census_observation_two_scope_retention_v1')
guardfuncs=funcs(ROOT/bindings['selected_child_source_ref']['path'])
# All call identities authenticated independently above/source pins. Here only operational
# callable-reader is doubled so constructor cannot enter a real guard or mutable file.
def authenticated_functions(module,record,names):
 need(ref(record['path'])==record,'Exact source-reader pin before operational double')
 nodes=funcs(ROOT/record['path']);return {n:(getattr(module,n,None),nodes[n])for n in names}
memo.functions_from_source=authenticated_functions
def metadata_source_bytes(record):
 need(ref(record['path'])==record,'Exact compact source SHA in source-only reader');return (ROOT/record['path']).read_bytes()
memo.source_bytes=metadata_source_bytes
# constructor + bridge helper source auth still genuine metadata reads.
methods,counts=bridge['derive_memo_methods_v1'](memo,(ROOT/bindings['ROOT_memo_source_ref']['path']).read_bytes(),bindings['selected_child_source_ref'],(),bindings['ROOT_memo_bridge_ref'])
need(counts==[1,1],'Exact source bridge two operand substitutions')
# Actual wrapper constructor will derive fresh methods with its complete control exclusions.
q=read(HERE/'ROOT_request.INACTIVE.template.v1.json');q.update(preparation_directory='research/reviews/SOURCE_ONLY_RAM/preparation',packet_directory='research/reviews/SOURCE_ONLY_RAM/packet',output_directory='experiments/results/SOURCE_ONLY_RAM',scientific_config=dict(q['scientific_config'],execution_enabled=True),created_at_utc='RAM_created',expires_at_utc='RAM_expires',cpu_ids=list(range(32)))
for k in ('root_directive_ref','training_context_ref','registration_ref','registration_decisions_ref','runtime_ref','runtime_qualification_ref','source_review_ref','scientific_review_ref'):
 q[k]=memory_put(memroot/'research/reviews/SOURCE_ONLY_RAM'/ (k+'.json'),{})
q['source_manifest_ref']=memory_put(memroot/'research/reviews/SOURCE_ONLY_RAM/integration.json',dict(source_sha256={bindings['ROOT_memo_source_ref']['path']:bindings['ROOT_memo_source_ref']['sha256'],**{r['path']:r['sha256']for r in [bindings[k]for k in ('ROOT_memo_source_ref','ROOT_memo_manifest_ref','ROOT_memo_handoff_ref','ROOT_memo_review_ref','ROOT_memo_bridge_ref')]}}))
request_ref=memory_put(memroot/q['preparation_directory']/'request.json',q)
replace_roles={js(sym('manifest')):q['source_manifest_ref'],**{js(sym(k)):q[k]for k in ('source_review_ref','scientific_review_ref','runtime_ref','runtime_qualification_ref','root_directive_ref','registration_ref','registration_decisions_ref','training_context_ref')}}
roles=[replace_roles.get(js(r),r)for r in roles]
wrapper=dict(ast=ast,bindings=lambda:bindings,read=mread,raw=lambda r:(ROOT/r['path']).read_bytes(),need=need,Path=MemoryPath,load=lambda r,n:memo if r==bindings['ROOT_memo_source_ref']else SimpleNamespace(**bridge),copy=copy)
execnodes([common['_make_root_mutation_memo'],common['_memo_admission_node']],wrapper)
constructed=[]
def construct(guard,q,r):
 result=wrapper['_make_root_mutation_memo'](guard,q,r);constructed.append(result);op.append('construct');return result
# Exact original admission AST and actual input/storage verification function, all IO and CPU APIs RAM doubles.
admission=wrapper['_memo_admission_node'](funcs(ORIGINAL/'root_driver_v1.py')['admit'],'H060')
c=SimpleNamespace(validate_request=lambda r:(mread(r),caps,{}),make_config=lambda q,c:dict(schema='RAM_config',scientific_config=q['scientific_config'],caps=c,short_tmp_directory=q['packet_directory']+'/worker_aux/tmp',training_context_ref=q['training_context_ref']),sources=lambda:mread(q['source_manifest_ref']),direct_roles=lambda *args:roles,CAPS=caps,TRAIN=tuple(proto.TRAIN),PARENT='American_C3_short20_TRAIN_TreeLoom_matched',PHASE='train_matched',POLICY_REF=chosen['policy_ref'],SELECTOR_REF=bindings['actual_selector_metadata_ref'],GUARD_REF=bindings['selected_child_source_ref'],ENTRY='RAM_entry.py',DRIVER='RAM_driver.py')
# Pure selected union and exact original draft source; no guard implementation called.
g.consumer_bindings_v1=api['consumer_bindings_v1'];dscope=dict(copy=copy,require=need,MODE=g.MODE,RETIRE_OPERATION='RAM_retire_never_called')
execnodes([guardfuncs['operation_draft_v1']],dscope);g.operation_draft_v1=dscope['operation_draft_v1']
# scope_spec validates exact metadata format and complete64cap before authority/reserve.
mem['.runtime/RAM_legacy.json']=b'{"jobs":{}}'
# Original marker/source refs must be represented in direct union; they are inherited guard roles.
source_map=mread(q['source_manifest_ref'])['source_sha256'];c.ref=lambda p:dict(path=p,sha256=source_map.get(p,'a'*64));c.read=mread;c.local=lambda p:memroot/p;c.require=need;c.current=lambda *a:None
c.validate_request=lambda r:(mread(r),caps,{})
nsinput=dict(c.__dict__,read=mread,ref=c.ref,local=c.local,require=need,sources=c.sources,validate_request=c.validate_request,make_config=c.make_config,direct_roles=c.direct_roles,_ACTIVE=None)
execnodes([funcs(ORIGINAL/'root_contract_v1.py')['input_spec']],nsinput);c.input_spec=nsinput['input_spec']
class AdmissionPath(MemoryPath):
 def mkdir(self):op.append('packet_mkdir')
def staged_put(p,v):need(bool(constructed),'Constructor must precede first staging write');op.append('put:'+p.name);return memory_put(p,v)
admit_scope=dict(c=c,local=lambda p:AdmissionPath(memroot/p),need=need,os=SimpleNamespace(sched_getaffinity=lambda p:set(range(32))),guard=lambda consumer:g,put=staged_put,ref=c.ref,ROOT=memroot,SOURCE=c.DRIVER,HERE='RAM.module',json=json,_make_root_mutation_memo=construct)
execnodes([admission],admit_scope)
receipt=admit_scope['admit'](request_ref)
need(receipt['admission_only']is True and len(constructed)==1 and op.count('lock')==op.count('unlock')==2 and op.count('append')==1,'Actual constructor/state/reserve/input whole RAM join')
need(op.index('construct')<next(i for i,e in enumerate(op)if e.startswith('put:'))and op.index('state')<op.index('append'),'Constructor before staging; original fresh state then reserve')
need(receipt['consumer_binding_count']==len(union),'Complete direct role union retained through admitted input')
checks.append('actual_wrapper_memo2_constructor_current_control_exclusions_original_two_transactions_admit_storage_input_RAM_join')
# Actual worker body invokes the real tested runner function after its genuine gate
# order. Every import/process/resource/writer API is an in-memory handwritten double.
input_record=receipt['input_ref'];admitted=mread(input_record);_,_,worker_cfg=c.input_spec(input_record)
state_locked_original=state_locked
def state_for_worker(*args,**kwargs):
 need(locks[0]and kwargs==dict(fresh=False),'Worker original role revalidation fresh=False');return state
g._state_locked=state_for_worker
verify_scope=dict(c.__dict__,ROOT=memroot,ref=ref,direct_roles=c.direct_roles,require=need)
execnodes([funcs(ORIGINAL/'root_contract_v1.py')['verify_actual_roles']],verify_scope)
c.verify_actual_roles=verify_scope['verify_actual_roles'];c.guard=lambda consumer:g;c.input_spec=lambda r:(admitted,q,worker_cfg)
c.context=lambda r:dict(parent_plan_refs=planrefs,actual_train_bindings={p:{}for p in proto.TRAIN});c.binding=lambda:b
c.raw=lambda r:(ROOT/r['path']).read_bytes();c.TARGET_HOOK=deps['TARGET_HOOK'];c.BASE_RUNTIME=deps['BASE_RUNTIME'];c.ACCEPTANCE=deps['ACCEPTANCE'];c.STATUSES={'train_matched':'completed_TreeLoom_saved600_matched_CPU_training'}
g.validate_storage_v9=lambda *a,**k:events.append(('storage_gate',None))
c.current=lambda *a:None
fake_torch=SimpleNamespace(set_num_threads=lambda n:None,set_num_interop_threads=lambda n:None,get_default_dtype=lambda:'fp32',float32='fp32',cuda=SimpleNamespace(is_initialized=lambda:False),get_num_threads=lambda:32)
real_import=__import__
def scientific_import_double(name,globals=None,locals=None,fromlist=(),level=0):
 if name=='numpy':return SimpleNamespace()
 if name=='torch':return fake_torch
 if name=='torch_geometric':return SimpleNamespace()
 if name=='arggnn.american_c3_common_v1':return SimpleNamespace(runtime_report=lambda:c.read(c.BASE_RUNTIME))
 raise ValueError('Unexpected worker import '+name)
worker_scope=dict(c=c,ROOT=memroot,Path=MemoryPath,sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=True),prefix=str(memroot/'.runtime/env')),os=SimpleNamespace(read=lambda fd,n:b'G',close=lambda fd:events.append(('gate_closed',None)),environ=dict(CUDA_VISIBLE_DEVICES='',**{key:str(memroot/worker_cfg['short_tmp_directory'])for key in ('TMPDIR','TMP','TEMP')}),sched_getaffinity=lambda pid:set(range(32)),sched_setaffinity=lambda pid,ids:events.append(('affinity',None))),time=SimpleNamespace(monotonic=lambda:0.),resource=SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda kind:SimpleNamespace(ru_maxrss=0)),BoundedWriter=lambda *a:Writer(),jsonable_runtime_v1=lambda value,*a:value,importlib=SimpleNamespace(import_module=lambda name:SimpleNamespace(actual_train_metadata_v1=lambda bindings:{p:dict(private_truth_ref=sym(p+'_truth'))for p in proto.TRAIN})if name==b['target_hook_module']else SimpleNamespace(run_train_v1=ns['run_train_v1'])),__builtins__=dict(vars(__import__('builtins')),__import__=scientific_import_double))
execnodes([funcs(HERE/'worker_entry_v1.py')['worker']],worker_scope)
original_cache_get=ns['_cache_get_v2']
def observed_cache_get(cache,parent):
 result=original_cache_get(cache,parent);events.append(('cache_get',parent));return result
ns['_cache_get_v2']=observed_cache_get
events.clear();stored.clear();cache_objects.clear();mode[0]=None
worker_scope['worker'](input_record,7)
need('completed.json'in stored and len([e for e in events if e[0]=='target'])==20,'Actual worker-to-runner full RAM join')
need(events.index(('gate_closed',None))<events.index(('storage_gate',None))<events.index(('save','started.json'))<next(i for i,e in enumerate(events)if e[0]=='prepare'),'Actual worker gate/storage/start before scientific runner')
checks.append('actual_worker_gate_role_revalidation_storage_imports_then_runner_all20barrier_600pairs_RAM_join')
# Actual whole-scientific review gate accepts only the complete new schema and refs.
review=read(HERE/'ROOT_scientific_review.INACTIVE.template.v1.json');marker=sym('review_only_RAM');payload=[dict(review,status='SOURCE_PASS',independent_review=True,blockers=[],candidate_outer_manifest_ref=sym('outer_manifest'))]
reviewc=SimpleNamespace(binding=lambda:b,read=lambda r:payload[0]if r==marker else read(ROOT/r['path']),require=need)
reviewscope=dict(metadata_ref=lambda p:sym('outer_manifest'),MANIFEST='unused');execnodes([common['global_scientific_review']],reviewscope)
reviewscope['global_scientific_review'](reviewc,marker)
for label,change in [('inactive_review',dict(status='UNREVIEWED_INACTIVE')),('old_review_schema',dict(schema='H060_legacy_fast_complete_scientific_review_v1')),('missing_new_source',dict(candidate_source_refs=[])),('wrong_component_review',dict(component_review_ref=sym('wrong_component')))]:
 base=payload[0];payload[0]=dict(base,**change);reject(label,lambda:reviewscope['global_scientific_review'](reviewc,marker));payload[0]=base
# Nested old target binding must match exactly; no spoofed old protocol equality.
barrier=stored['TRAIN_observation_barrier.json'];binding=factory({},barrier=barrier,source_refs=b['global_preparation_source_refs'],runtime_guard=lambda e:None).execution_binding
reject('mismatched_nested_original_target_binding',lambda:proto.validate_target_protocol_binding_v1(binding,dict(oldtarget,target_equations_unchanged=False),barrier=barrier))
checks.append('whole_new_scientific_review_and_nested_original_target_binding_positive_negative_gates')
# Scientific source refs and inherited source metadata only, never runtime/payload reads.
pins=read(HERE/'scientific_source_pins.v2.json')['refs']
for r in pins:need(ref(r['path'])==r,'Exact scientific closure pin '+r['path'])
need(all(r in pins for r in b['global_preparation_source_refs']),'Complete typed prepare/target closure pinned')
for key in ('ROOT_memo_source_ref','ROOT_memo_manifest_ref','ROOT_memo_handoff_ref','ROOT_memo_review_ref','ROOT_memo_bridge_ref','selector_bridge_ref'):
 need(ref(bindings[key]['path'])==bindings[key],'Exact accepted bridge evidence '+key)
for name in ('numpy','torch','torch_geometric','tskit'):need(name not in sys.modules,'Scientific import forbidden')
checks.append('all_scientific_and_accepted_component_bridge_metadata_pins_authenticate_no_scientific_imports')
result=dict(schema='H060_global_only_source_metadata_RAM_checks_v1',status='SOURCE_ONLY_CHECKS_PASS',checks=checks,negative_cases=negative,checks_count=len(checks),negative_count=len(negative),current_guard_named_roles=47,current_guard_unique_refs=40,pure_direct_union=len(union),direct_ref_ceiling=64,all20_original_cache_checks_pretruth=True,full600pairs_RAM_metadata_only=True,actual_memo2_constructor_state_reserve_admitted_input_RAM_join=True,operational_readers_mutexes_mutable_files_resources_CPUaffinity_writers_targets_models_doubled=True,scientific_imports_payloads_tensors_true_truth_actual_guards_process_resources_accounting_Git_packages=False,actual_execution_or_admission_authorized=False,real_fit_speed_or_accuracy_measured=False)
if __name__=='__main__':print(json.dumps(result,indent=2,sort_keys=True))

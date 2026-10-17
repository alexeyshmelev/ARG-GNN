"""Narrow actual entry/helper metadata routes with RAM filesystem/guard adapters only."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
from contextlib import contextmanager,redirect_stdout
import ast,copy,hashlib,importlib,io,json,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE='research/reviews/FLARE_original_VAL002004_conversion_execution_source_20261017_v1';D=ROOT/HERE
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
checks=[];calls=[];documents={};by_path={};virtual_dirs=set();actual_reads={}
def check(label,ok=True):assert ok,label;checks.append(label)
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 assert p.suffix in('.py','.json','.md','.sh','.txt','.diff','.patch')and not p.is_symlink()and p.is_file()and p.stat().st_size<=2**20
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def key(r):return r['path'],r['sha256']
def encoded(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def ram_doc(path,value):
 if not path.startswith('research/reviews/')and not path.startswith('data/'):path='research/reviews/RAM_ONLY_NOT_ACTUAL_FLARE_conversion/'+path
 r=dict(path=path,sha256=hashlib.sha256(encoded(value)).hexdigest());documents[key(r)]=copy.deepcopy(value);by_path[path]=r;return r
def read(r):
 if key(r)in documents:return copy.deepcopy(documents[key(r)])
 assert Path(r['path']).suffix=='.json'and ref(r['path'])==r;actual_reads[r['path']]=r;return json.loads((ROOT/r['path']).read_text())
def raw(r):
 if key(r)in documents:return encoded(documents[key(r)])
 assert ref(r['path'])==r;return(ROOT/r['path']).read_bytes()
def mref(p):
 p=str(p);p=str(Path(p).relative_to(ROOT))if Path(p).is_absolute()else p
 return copy.deepcopy(by_path[p])if p in by_path else ref(p)
class RamPath(PurePosixPath):
 def exists(self):return str(self)in virtual_dirs or str(self.relative_to(ROOT))in by_path
 def is_dir(self):return str(self)in virtual_dirs
 def mkdir(self,*a,**kw):
  assert not self.exists();virtual_dirs.add(str(self));calls.append('RAM_mkdir')
 def resolve(self):return self
rpath=lambda p:RamPath(str(ROOT))/(str(Path(p).relative_to(ROOT))if Path(p).is_absolute()else p)
def put(path,value,**kwargs):
 calls.append('RAM_put');return ram_doc(str(path.relative_to(ROOT)),value)
# These are the actual package modules. Their default entries must remain inert.
d=importlib.import_module(HERE.replace('/','.')+'.root_driver_v1')
h=importlib.import_module(HERE.replace('/','.')+'.prepare_ROOT_v1')
w=importlib.import_module(HERE.replace('/','.')+'.worker_entry_v1')
pc=importlib.import_module(HERE.replace('/','.')+'.prepare_context_v1')
for label,module in [('driver',d),('preparer',h),('worker',w),('context',pc)]:
 output=io.StringIO()
 with redirect_stdout(output):assert module.main([])==0
 check('actual_'+label+'_entry_no_args_inert',json.loads(output.getvalue()).get('status')in('INACTIVE_SOURCE_ONLY','SOURCE_ONLY_INACTIVE'))
# Real contract factory performs only source/JSON imports. Guard loading is prohibited here.
c=d.contract();b=d.bindings();actual_load=c.load
c.read=read;c.raw=raw;c.ref=mref;c.local=rpath;c.put=put
c.completion=lambda:h.metadata_core_v1(c,read)
check('actual_contract_factory_selects_only_original002004',c.VAL==('american_c3_val_002','american_c3_val_003','american_c3_val_004')and c.STAGES==('convert',))
check('no_guard_or_numerical_runtime_imported',not any(n.startswith(('numpy','torch','scripts.project_budget'))for n in sys.modules))
profile=b['Census_source_profile'];child=raw(profile['source_ref']);tree=ast.parse(child)
# Construct only RAM future activation/selector metadata using the exact child selector draft.
scope=dict(copy=copy,_ref=mref)
for n in tree.body:
 if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
  try:scope[n.targets[0].id]=ast.literal_eval(n.value)
  except (ValueError,TypeError):pass
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='selector_draft_v1')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact child selector draft RAM only>','exec'),scope)
parent_selector=read(b['predecessor_bindings_ref'])['actual_selector_metadata_ref'];old=read(parent_selector)
activation=ram_doc('future_activation.json',dict(schema='Census_observation_frozen_Relate64_ROOT_activation_v1',active=True,root_authorizer='root_agent',source_ref=profile['source_ref'],source_manifest_ref=profile['source_manifest_ref'],ROOT_explicitly_releases_current_selector_for_this_child=True))
lineage=dict(base_ref=scope['BASE_REF'],base_policy_ref=scope['BASE_POLICY_REF'],base_pins={'base_pins':{},'completed_overlay_prefix':{'events':0,'tail_sha256':'0'*64}},base_projected_bytes=946000000000,base_accounting_mode=scope['BASE_MODE'])
policy=ram_doc('future_policy.json',dict(schema='Census_observation_exact_retained_allowance_policy_v1',root_authorizer='root_agent',active=True,source_ref=profile['source_ref'],selector_source_ref=profile['source_ref'],source_manifest_ref=profile['source_manifest_ref'],measurement_source_ref=profile['measurement_source_ref'],source_review_ref=profile['source_review_ref'],marker_repair_authority_ref=old['repair_authority_ref'],journal=scope['JOURNAL'],base_ref=scope['BASE_REF'],base_policy_ref=scope['BASE_POLICY_REF'],base_selector_ref=scope['BASE_SELECTOR_REF'],base_accounting_mode=scope['BASE_MODE'],base_pins=lineage['base_pins'],base_projected_bytes=lineage['base_projected_bytes'],root_activation_ref=activation,scheduling_ceiling_bytes=980000000000,human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000))
sel=scope['selector_draft_v1'](policy_ref=policy,source_review_ref=profile['source_review_ref'],repair_authority_ref=old['repair_authority_ref'],frozen_parent_lineage=lineage);sel['active']=True
selector_ref=ram_doc('future_selector.json',sel)
review_ref=ram_doc('future_review.json',dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_ref=mref(d.SOURCE),source_manifest_ref=mref(d.MANIFEST),source_handoff_ref=mref(d.HANDOFF)))
now=datetime.now(timezone.utc)
slots={parent:dict(cpu_ids=list(range(i*4,i*4+4)),preparation_directory='research/reviews/FLARE_remaining_ROOT_preparation_VAL'+parent[-3:]+'_convert_RAM_only',packet_directory='research/reviews/FLARE_remaining_ROOT_actual_VAL'+parent[-3:]+'_convert_RAM_only',output_directory='data/FLARE_original_remaining_VAL'+parent[-3:]+'_convert_RAM_only')for i,parent in enumerate(c.VAL)}
authority=ram_doc('future_ROOT_instruction.json',dict(schema='ROOT_FLARE_VAL002004_conversion_metadata_preparation_v1',root_authorizer='root_agent',active=True,metadata_preparation_only=True,admission_authorized=False,launch_authorized=False,ROOT_reviewed_preparer_source=True,conversion_request_metadata_authorized=True,preparation_source_ref=mref(h.SOURCE),source_manifest_ref=mref(d.MANIFEST),source_review_ref=review_ref,storage_policy_selector_ref=selector_ref,created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(hours=8)).isoformat(),targets=slots))
# Actual prepare_v1 route, including RAM preflight, three fresh dirs, all writes and final validator.
h.read=read;h.reference=mref;h.local=rpath;h.put=put
d.contract=lambda:c
result=h.prepare_v1(authority)
check('actual_prepare_v1_complete_route_RAM_only',set(result['targets'])==set(c.VAL)and result['admission_or_launch_performed']is False)
check('only_three_preparation_directories_RAM_created',len(virtual_dirs)==3 and all(str(ROOT/slot['preparation_directory'])in virtual_dirs for slot in slots.values()))
for parent,row in result['targets'].items():
 q,caps,runtime=c.validate_request(row['request_ref'])
 expected=b['original_targets'][parent]
 check(parent+'_actual_request_and_original_context_full_support',caps['output_bytes']==expected['conversion_output_cap_bytes']and caps['reservation_bytes']==expected['whole_scope_reservation_bytes']and caps['scientific_seconds']==10800 and caps['wall_seconds']==11400 and runtime['cpu_ids']==slots[parent]['cpu_ids'])
 cfg=c.make_config(q);basis=read(mref('research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/VAL000_basis.v1.json'))['configurations']['conversion_config']
 changes={k for k in set(cfg)|set(basis)if cfg.get(k)!=basis.get(k)}
 check(parent+'_frozen_converter_settings_only_target_metadata_changes',changes<= {'registration_ref','count_manifest_ref','focal_registry_ref','max_sites_bound','observation_ref','simulation_config_ref','max_sites','output_cap_bytes'})
# Immutable source selection and admission using only the original metadata role path with a fake guard.
# The binding set is the genuine historical46name/39distinct parent metadata plus exactly one child capsule.
guard_record=read(b['predecessor_bindings_ref'])['actual_guard_metadata_ref'];base_bindings=read(guard_record)['bindings'];guard_bindings=copy.deepcopy(base_bindings);guard_bindings['Census_observation_explicit_selector_capsule']=selector_ref
v4=types.ModuleType('_RAM_original_v4');
@contextmanager
def mutex(*a,**kw):calls.append('RAM_mutex');yield
v4.mutex=mutex;v4.canonical=lambda *a:None;v4.file_bytes=lambda *a:None
for fn in(v4.mutex,v4.canonical,v4.file_bytes):fn.__module__=v4.__name__
v4.scope_spec=lambda root,reserve:[rpath(reserve['output']),*[rpath(p)for p in reserve['auxiliary_scopes']]]
budget=types.SimpleNamespace(v4=v4,SOURCE_REL='scripts/project_budget_v9.py',LAUNCHER_REL='scripts/launch_reserved_gpu_v6.py')
g=types.ModuleType('_RAM_selected_Census_guard');g.MODE=profile['accounting_mode'];g._POLICY_REF=policy;g.budget=budget;g._SELECTOR_REF=selector_ref
# Execute the exact existing64ref consumer-union function with RAM/source-authenticated refs.
ns=dict(_raw=raw,require=c.require,copy=copy)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='consumer_bindings_v1')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact unchanged consumer union>','exec'),ns);g.consumer_bindings_v1=ns['consumer_bindings_v1']
g.require=c.require;g._raw=raw
g.load_selected_guard_v1=lambda r:g if r==selector_ref else (_ for _ in()).throw(AssertionError('No alternative guard'))
g._state_locked=lambda *a,**kw:({},None,{'RAM':True},guard_bindings)
g._policy=lambda:dict(never_credit_keys=[])
g._pins=lambda value:copy.deepcopy(value)
g.operation_draft_v1=lambda operation,reserve,**kw:dict(schema='RAM_reserve_authority',operation=operation,reservation=reserve,initial_ledger_pins=kw['expected_before'])
def reserve(ar,**kw):
 value=read(ar);calls.append('RAM_apply_reserve')
 return dict(kind='concurrent_completed_scope_reserve_receipt_v1',value=copy.deepcopy(value['reservation']),initial_ledger_pins=value['initial_ledger_pins'])
g.apply_reserve_v1=reserve
g.check_storage_locked_v1=lambda *a,**kw:dict(bindings=guard_bindings)
g.budget_check_v9=lambda *a,**kw:calls.append('RAM_original_budget_check')
memo_calls=[]
memo=types.SimpleNamespace(install_consumer_v1=lambda *a,**kw:memo_calls.append(kw)or object())
c._h.load=lambda record,name:g if record==profile['source_ref']else memo if record==c.NAMESPACE else (_ for _ in()).throw(AssertionError('Unexpected guard/memo source load'))
# Complete actual admit wrapper and original admission metadata call. OS affinity is a RAM adapter.
def load(record,name):
 module=actual_load(record,name)
 if record==b['original_driver_ref']:module.os=types.SimpleNamespace(sched_getaffinity=lambda pid:set(range(12)))
 if record==b['ROOT_memo_bridge_ref']:
  def make(g,q,request,c,b,base):
   assert not hasattr(g,'_outer_memo')
   uncached=module.current_uncached_paths_v1(c,q,request,b)
   paths=[request['path'],q['ROOT_metadata_preparation_instruction_ref']['path'],q['source_manifest_ref']['path'],q['source_review_ref']['path'],q['root_directive_ref']['path'],q['registration_ref']['path'],q['context_ref']['path'],q['runtime_ref']['path']]
   assert all(p in uncached for p in paths)
   methods,counts=module.derive_memo_methods_v1(base,raw(b['ROOT_memo_source_ref']),b['selected_child_source_ref'],uncached,b['ROOT_memo_bridge_ref'])
   assert counts==[1,1]and set(methods)=={'_check','_run'}
   calls.append('RAM_memo_construct')
   class Memo:
    def read_state_v1(self,**kw):
     with mutex(ROOT):calls.append('RAM_memo_fresh_state');return g._state_locked(kw['expected_source_sha256'],kw['expected_launcher_sha256'],fresh=True)
    def reserve_v1(self,ar,**kw):
     with mutex(ROOT):calls.append('RAM_memo_separate_fresh_reserve');return g.apply_reserve_v1(ar,**kw)
   return Memo()
  module.make_memo_v1=make
 return module
c.load=load
admitted={};role_counts={}
for parent,row in result['targets'].items():
 output=io.StringIO()
 with redirect_stdout(output):assert d.main(['--admit-ref',json.dumps(row['request_ref'])])==0
 adm=json.loads(output.getvalue());admitted[parent]=adm
 s,q,cfg=c.input_spec(adm['input_ref']);role_counts[parent]=adm['consumer_binding_count']
 check(parent+'_actual_driver_main_admit_and_original_input_join_RAM_only',s['parent_id']==parent and s['storage']['accounting_mode']==g.MODE and s['storage']['storage_policy_selector_ref']==selector_ref and s['storage']['reservation']['key']==q['proposed_reservation_key']and len(s['storage']['reservation']['consumer_bindings'])<=64)
 check(parent+'_emitted_exact_successor_execution_module',adm['execution_argv'][5]==HERE.replace('/','.')+'.root_driver_v1'and adm['execution_argv'][-1]==json.dumps(adm['input_ref']))
 g2=c.guard(True);view=d.initial_storage_transaction(s,q,g2);assert c.guard(True)is g2
 check(parent+'_unchanged_initial_storage_and_full_roles_path',g2 is g and view['bindings']==guard_bindings)
 delattr(g,'_outer_memo') # Separate future consumer interpreters are RAM fixtures only.
check('existing_consumer_memo_installed_once_after_exact_child_selection',len(memo_calls)==3 and all(v['consumer_role']=='read_only_consumer'for v in memo_calls))
# Actual execute entry is routed up to the owned lifecycle with RAM clock and terminal boundary.
class RAMClock:
 failure_kind=None
 def configure(self,q):assert q['phase_budget']['scientific_seconds']==10800;calls.append('RAM_clock_configure')
 def close(self):calls.append('RAM_clock_close')
 def evidence(self):return dict(RAM_only=True)
d.PhaseClock=RAMClock
def owned(r,clock):
 s,q,cfg=c.input_spec(r);calls.append('RAM_owned_execution_boundary');return dict(status='RAM_ENTRY_ONLY_NO_SCIENCE',returncode=0,parent_id=q['parent_id'])
d.owned_execute=owned
for parent,adm in admitted.items():
 output=io.StringIO()
 with redirect_stdout(output):assert d.main(['--execute-ref',json.dumps(adm['input_ref'])])==0
 check(parent+'_actual_execute_main_dispatch_full_metadata_to_RAM_owned_boundary',json.loads(output.getvalue())['parent_id']==parent)
check('memo_constructed_three_times_with_separate_state_and_reserve_transactions',calls.count('RAM_memo_construct')==calls.count('RAM_memo_fresh_state')==calls.count('RAM_memo_separate_fresh_reserve')==3)
# The exact derived v2 caller executes two cold transactions with extra exclusions and
# final source/bridge validation; transaction/append boundaries are RAM adapters.
bridge_module=actual_load(b['ROOT_memo_bridge_ref'],'_RAM_review_exact_Census_memo_source_bridge')
base_module=actual_load(b['ROOT_memo_source_ref'],'_RAM_review_exact_memo_v2_component')
request=result['targets'][c.VAL[0]]['request_ref'];request_value=read(request)
uncached=bridge_module.current_uncached_paths_v1(c,request_value,request,b)
methods,changed=bridge_module.derive_memo_methods_v1(base_module,raw(b['ROOT_memo_source_ref']),b['selected_child_source_ref'],uncached,b['ROOT_memo_bridge_ref'])
check('exact_memo_bridge_changes_only_two_source_operands',changed==[1,1])
trace=[];transactions=[]
@contextmanager
def original_mutex(root):trace.append('lock');yield;trace.append('unlock')
class Tx:
 def __init__(self,root,v4,budget,allow,*,uncached_paths):
  self.enabled=True;self.counts={};self.original={'bound':lambda *a:read(mref(d.MANIFEST))};self.paths=uncached_paths;transactions.append(self);trace.append('cold_transaction')
 def install(self):trace.append('install')
 def finish(self):assert self.enabled;self.enabled=False;trace.append('uncached_final')
 def close(self):assert not self.enabled;trace.append('restore_before_unlock')
def original_append(*args):assert not transactions[-1].enabled;trace.append('original_append_fsync_postread');return {'RAM_reserve':True}
source_guard_tree=ast.parse(raw(profile['source_ref']));reserve_node=next(n for n in source_guard_tree.body if isinstance(n,ast.FunctionDef)and n.name=='apply_reserve_v1')
assert len(reserve_node.body)==1 and isinstance(reserve_node.body[0],ast.With)
guard=types.SimpleNamespace(_append=original_append,_state_locked=lambda *a,**kw: {'fresh':kw['fresh']})
self=types.SimpleNamespace(_check=lambda:trace.append('uncached_identity_check'),v4=types.SimpleNamespace(mutex=original_mutex),guard=guard,budget=object(),allow={},adapter_ref=b['ROOT_memo_source_ref'],manifest_ref=mref(d.MANIFEST),guard_functions={'apply_reserve_v1':(None,reserve_node)})
def critical(node,scope,append):
 assert ast.dump(node,include_attributes=False)==ast.dump(reserve_node,include_attributes=False)
 return lambda authority_ref,**kw:append('reserve',{},dict(consumer_bindings=[mref(d.MANIFEST)]),{})
methods['_run'].__globals__.update(_Transaction=Tx,critical_reserve=critical)
methods['_run'](self,'state',expected_source_sha256='1'*64,expected_launcher_sha256='2'*64)
authority_for_reserve=ram_doc('current_reserve_authority.json',{'RAM_only':True})
methods['_run'](self,'reserve',authority_for_reserve,expected_source_sha256='1'*64,expected_launcher_sha256='2'*64)
check('derived_memo_two_cold_original_mutex_transactions',len(transactions)==2 and trace.count('lock')==trace.count('unlock')==2)
check('derived_memo_current_paths_plus_authority_excluded',set(uncached)<=set(transactions[0].paths)and set(uncached)|{authority_for_reserve['path']}<=set(transactions[1].paths))
check('derived_memo_restores_before_unlock_and_original_append_after_final',trace.count('restore_before_unlock')==2 and trace.index('original_append_fsync_postread')>max(i for i,x in enumerate(trace)if x=='uncached_final'))
# Negative cases exercise actual installed source joins before any guard or write path.
def rejects(label,action):
 old_calls=len(calls)
 try:action()
 except (ValueError,TypeError,KeyError):pass
 else:raise AssertionError('Expected fail-closed: '+label)
 check(label,len(calls)==old_calls)
rejects('missing_explicit_selector_rejected',lambda:c.selected(None))
rejects('old_Relate64_selector_rejected_no_fallback',lambda:c.selected(parent_selector))
modified=dict(sel,active=False);inactive=ram_doc('inactive_selector.json',modified)
rejects('inactive_Census_selector_rejected',lambda:c.selected(inactive))
wrong=dict(sel,source_ref=old['source_ref']);mixed=ram_doc('mixed_selector.json',wrong)
rejects('mixed_Census_Relate_source_rejected',lambda:c.selected(mixed))
other=ram_doc('other_selector_same_contents.json',sel)
rejects('lifetime_selector_ref_swap_rejected',lambda:c.bind_selector(other))
qref=result['targets'][c.VAL[0]]['request_ref'];original=copy.deepcopy(documents[key(qref)])
for label,changes in [('VAL001_rerun_rejected',dict(parent_id='american_c3_val_001')),('partial_reuse_rejected',dict(reuse_partial_outputs=True)),('wrong_key_rejected',dict(proposed_reservation_key='H041/other')),('wrong_cap_rejected',dict(caps=dict(original['caps'],output_bytes=1))),('wrong_phase_budget_rejected',dict(phase_budget=dict(original['phase_budget'],scientific_seconds=3600)))]:
 documents[key(qref)]=dict(original,**changes);rejects(label,lambda:c.validate_request(qref))
documents[key(qref)]=original
old_authority=copy.deepcopy(documents[key(authority)])
modified=copy.deepcopy(old_authority);modified['targets'][c.VAL[1]]['cpu_ids']=modified['targets'][c.VAL[0]]['cpu_ids'];documents[key(authority)]=modified
rejects('overlapping_CPU4_groups_rejected',lambda:c.validate_batch_instruction_v1(authority));documents[key(authority)]=old_authority
# AST invariants cover all retained process/wait/drain/scientific worker code without running it.
oldtree=ast.parse(raw(b['predecessor_driver_ref']));newtree=ast.parse((D/'root_driver_v1.py').read_text())
oldnodes={n.name:n for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};newnodes={n.name:n for n in newtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
check('every_driver_lifecycle_function_class_AST_unchanged_except_contract_and_admit_source_bridge',all(ast.dump(n,include_attributes=False)==ast.dump(newnodes[name],include_attributes=False)for name,n in oldnodes.items()if name not in('contract','admit')))
check('whole_scientific_worker_AST_identical_to_proven_v4',ast.dump(ast.parse(raw(b['predecessor_worker_ref'])),include_attributes=False)==ast.dump(ast.parse((D/'worker_entry_v1.py').read_text()),include_attributes=False))
check('original_author_converter_exact_SHA',mref('src/arggnn/american_c3_flare_convert_v1.py')['sha256']=='6e55e295226a183548a23258e7230855842bb876b4d2a90875d8f2ad53181ff5')
check('actual_guards_payloads_numerics_and_resource_implementations_not_imported',not any(n.startswith(('numpy','torch','scripts.project_budget'))for n in sys.modules))
result=dict(schema='FLARE_original_VAL002004_actual_entry_helper_source_RAM_checks_v1',status='PASS_SOURCE_JSON_RAM_ENTRY_ROUTES_ONLY',checks=len(checks),cases=checks,actual_metadata_refs=list(actual_reads.values()),full_role_unions=role_counts,historical_parent_guard_names=len(base_bindings),historical_parent_distinct_guards=len({key(r)for r in base_bindings.values()}),Census_guard_names=len(guard_bindings),original_role_ceiling=64,actual_production_helper_admission_execute_entries_called_with_RAM_adapters=True,guard_resource_process_payload_truth_numerical_admission_launch_or_Git_operations=False,future_selector_policy_activation_and_source_review_RAM_only=True,actual_child_activation_or_available_fit_not_claimed=True)
with(D/'source_checks.result.v2.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in('cases','actual_metadata_refs')},sort_keys=True))

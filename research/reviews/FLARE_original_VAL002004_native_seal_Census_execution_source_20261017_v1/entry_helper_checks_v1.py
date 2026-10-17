"""Narrow actual entry/helper metadata routes with RAM filesystem/guard adapters only."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
from contextlib import contextmanager,redirect_stdout
import ast,builtins,copy,hashlib,importlib,io,json,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE='research/reviews/FLARE_original_VAL002004_native_seal_Census_execution_source_20261017_v1';D=ROOT/HERE
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
 if not path.startswith('research/reviews/')and not path.startswith('data/'):path='research/reviews/RAM_ONLY_NOT_ACTUAL_FLARE_native_seal/'+path
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

# Import inert source entries; no guard/process/scientific module is imported.
d=importlib.import_module(HERE.replace('/','.')+'.root_driver_v1')
h=importlib.import_module(HERE.replace('/','.')+'.prepare_ROOT_v1')
for label,module in [('driver',d),('preparer',h)]:
 output=io.StringIO()
 with redirect_stdout(output):assert module.main([])==0
 check('actual_'+label+'_no_args_inert',json.loads(output.getvalue())['status']in('SOURCE_ONLY_INACTIVE','INACTIVE_SOURCE_ONLY'))
output=io.StringIO()
with redirect_stdout(output):assert d.worker_entry_same_source_v1([])==0
check('actual_embedded_worker_no_args_inert',json.loads(output.getvalue())['status']=='SOURCE_ONLY_INACTIVE')
c=d.contract();b=d.bindings();actual_load=c.load
c.read=read;c.raw=raw;c.ref=mref;c.local=rpath;c.put=put;c.completion=lambda:h.metadata_core_v1(c,read)
check('actual_factory_native_seal_only_original002004',c.STAGES==('native','seal')and c.VAL==tuple('american_c3_val_%03d'%i for i in(2,3,4)))
check('same_authenticated_source_ref_for_driver_worker',c.DRIVER==c.ENTRY and mref(c.DRIVER)==mref(c.ENTRY))
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
# Future prerequisite success metadata is explicitly RAM-only, never an actual completion/marker claim.
def prerequisite(parent,stage,completion,inp):
 ir=ram_doc('prereq/'+parent+'/'+stage+'/input.json',inp);reserve=inp['storage']['reservation']
 wait=ram_doc('prereq/'+parent+'/'+stage+'/wait.json',dict(actual_direct_worker_wait=True,returncode=0,all_owned_processes_and_groups_drained=True,operational_failure=None,input_ref=ir,completed_ref=completion))
 outer=ram_doc('prereq/'+parent+'/'+stage+'/outer.json',dict(actual_terminal_consumed=True,actual_exit_code=0,completed_ref=completion,direct_wait_ref=wait))
 closed=ram_doc('prereq/'+parent+'/'+stage+'/ordinary_close.json',dict(kind='concurrent_completed_scope_close_receipt_v1',key=reserve['key'],artifacts_deleted=0,value=dict(future_writes_and_relaunch_revoked=True,request=dict(completed_ref=completion,direct_wait_ref=wait),reservation_sha256=hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())))
 return ram_doc('prereq/'+parent+'/'+stage+'/evidence.json',dict(schema='FLARE_remaining_stage_ROOT_terminal_evidence_v1',root_authorizer='root_agent',stage=stage,parent_id=parent,completion_ref=completion,input_ref=ir,actual_wait_ref=wait,outer_terminal_ref=outer,ordinary_close_receipt_ref=closed))
contexts={};converted={};conversion_evidence={};native={};native_evidence={}
for i,parent in enumerate(c.VAL):
 row=copy.deepcopy(b['original_targets'][parent]['target_row_for_later_ROOT_binding'])
 count=ram_doc(parent+'/count.json',dict(schema='American_C3_full30Mb_observation_count_root_v1',source_observation_ref=row['observation_ref'],num_sites=row['input_sites'],num_samples=1600,sequence_length=30000000,truth_opened=False,count_source_ref=row['plan_ref'],parent_id=parent))
 row['count_manifest_ref']=count;ctx=read(b['base_context_ref']);ctx['targets'][parent]=row
 contexts[parent]=ram_doc(parent+'/actual_prior_context.json',ctx)
 core=c.completion();core._context(contexts[parent],parent_id=parent)
 # A synthetic marker count exercises count-derived bounds; no prediction of future actual counts.
 converted[parent]=ram_doc('data/RAM_ONLY_FLARE_CONVERSION_'+parent+'/completed.json',dict(status='completed_C3_FLARE_observation_conversion',parent_id=parent,observation_ref=row['observation_ref'],focal_registry_ref=row['focal_registry_ref'],native_options=core.OPTIONS,native_source_commit=core.COMMIT,input_sites=row['input_sites'],query_ancestry_truth_opened=False,native_expected_markers=150000+i))
 prior_q=ram_doc(parent+'/convert/actual_prior_request.json',dict(stage='convert',parent_id=parent,context_ref=contexts[parent],source_manifest_ref=b['conversion_manifest_ref']))
 prior_inp=dict(schema='FLARE_remaining_CPU_admitted_input_v1',parent_id=parent,stage='convert',root_driver_ref=b['conversion_driver_ref'],main_source_ref=b['conversion_worker_ref'],source_sha256=read(b['conversion_manifest_ref'])['source_sha256'],root_stage_request_ref=prior_q,output_directory=str(Path(converted[parent]['path']).parent),storage=dict(reservation=dict(key='RAM_ONLY_CONVERSION_'+parent,output=str(ROOT/Path(converted[parent]['path']).parent),reserved_bytes=1)))
 conversion_evidence[parent]=prerequisite(parent,'convert',converted[parent],prior_inp)
def authority_for(parent,stage):
 label='VAL'+parent[-3:];stem=label+'_'+stage+'_RAM_only'
 return ram_doc(parent+'/'+stage+'/future_ROOT_instruction.json',dict(schema='ROOT_FLARE_VAL002004_native_seal_metadata_preparation_v1',root_authorizer='root_agent',active=True,metadata_preparation_only=True,admission_authorized=False,launch_authorized=False,ROOT_reviewed_preparer_source=True,one_stage_request_metadata_authorized=True,preparation_source_ref=mref(h.SOURCE),source_manifest_ref=mref(d.MANIFEST),source_review_ref=review_ref,storage_policy_selector_ref=selector_ref,created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(hours=8)).isoformat(),parent_id=parent,stage=stage,cpu_ids=list(range(4)),preparation_directory='research/reviews/FLARE_remaining_ROOT_preparation_'+stem,packet_directory='research/reviews/FLARE_remaining_ROOT_actual_'+stem,output_directory='data/FLARE_original_remaining_'+stem,context_ref=contexts[parent],conversion_ref=converted[parent],native_run_ref=None if stage=='native'else native[parent],previous_stage_evidence_ref=conversion_evidence[parent]if stage=='native'else native_evidence[parent]))
h.read=read;h.reference=mref;h.local=rpath;h.put=put;d.contract=lambda:c
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
 if record==b['original_driver_ref']:module.os=types.SimpleNamespace(sched_getaffinity=lambda pid:set(range(4)))
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

admitted={};prepared={};authorities={};role_counts={};direct_roles={}
for stage in('native','seal'):
 for parent in c.VAL:
  label=parent+':'+stage;authority=authority_for(parent,stage);authorities[label]=authority
  prep_result=h.prepare_v1(authority);prepared[label]=prep_result
  check(label+'_actual_preparer_complete_one_parent_one_stage_RAM_route',prep_result['parent_id']==parent and prep_result['stage']==stage and prep_result['context_ref']==contexts[parent]and prep_result['admission_or_launch_performed']is False)
  q,caps,runtime=c.validate_request(prep_result['request_ref']);cfg=c.make_config(q)
  check(label+'_original_CPU_RSS_science_wall_and_charge',caps['cpu_threads']==4 and caps['rss_bytes']==(24 if stage=='native'else 4)*2**30 and caps['scientific_seconds']==(14400 if stage=='native'else 3600)and caps['wall_seconds']==(15000 if stage=='native'else 4200)and caps['reservation_bytes']==cfg['output_cap_bytes']+8*2**20)
  check(label+'_original_full_support_context_and_truth_barrier',cfg['max_sites_bound']==b['original_targets'][parent]['input_sites']and cfg['external_truth_ref']is None and cfg['focal_registry_ref']==b['original_targets'][parent]['target_row_for_later_ROOT_binding']['focal_registry_ref'])
  output=io.StringIO()
  with redirect_stdout(output):assert d.main(['--admit-ref',json.dumps(prep_result['request_ref'])])==0
  adm=json.loads(output.getvalue());admitted[label]=adm
  s,q,cfg=c.input_spec(adm['input_ref']);role_counts[label]=adm['consumer_binding_count']
  direct_roles[label]=c.direct_roles(s['root_stage_request_ref'],q,s['refs']['config'],s['transitive_source_closure_ref'])
  expected=63 if stage=='native'else 64
  check(label+'_complete_source_derived_union_under_original64',adm['consumer_binding_count']==expected and all(r in s['storage']['reservation']['consumer_bindings']for r in guard_bindings.values())and all(r in s['storage']['reservation']['consumer_bindings']for r in direct_roles[label]))
  check(label+'_honest_same_file_entry_roles_deduplicate',direct_roles[label].count(mref(c.DRIVER))>=2 and {key(r)for r in direct_roles[label]}<={key(r)for r in s['storage']['reservation']['consumer_bindings']})
  check(label+'_only_original_emitted_execution_argv',adm['execution_argv'][5]==HERE.replace('/','.')+'.root_driver_v1'and adm['execution_argv'][-1]==json.dumps(adm['input_ref']))
  g2=c.guard(True);view=d.initial_storage_transaction(s,q,g2)
  check(label+'_actual_initial_storage_all_current_roles',view['bindings']==guard_bindings)
  delattr(g,'_outer_memo')
  if stage=='native':
   native[parent]=ram_doc(q['output_directory']+'/completed.json',dict(status='completed_C3_native_FLARE',parent_id=parent,focal_registry_ref=cfg['focal_registry_ref'],binary_provenance_ref=cfg['binary_provenance_ref'],conversion_ref=converted[parent],query_ancestry_truth_opened=False))
   native_evidence[parent]=prerequisite(parent,'native',native[parent],s)
check('six_separate_preparation_dirs_and_six_cold_memo_transactions',len(virtual_dirs)==12 and calls.count('RAM_memo_construct')==calls.count('RAM_memo_fresh_state')==calls.count('RAM_memo_separate_fresh_reserve')==6)
check('six_consumer_memo_installs_with_read_only_roles',len(memo_calls)==6 and all(v['consumer_role']=='read_only_consumer'for v in memo_calls))
class RAMClock:
 failure_kind=None
 def configure(self,q):assert q['phase_budget']==d.phase_budget(q['stage']);calls.append('RAM_clock_configure')
 def close(self):calls.append('RAM_clock_close')
 def evidence(self):return dict(RAM_only=True)
d.PhaseClock=RAMClock
def owned(r,clock):
 s,q,cfg=c.input_spec(r);calls.append('RAM_owned_execution_boundary');return dict(status='RAM_ENTRY_ONLY_NO_SCIENCE',returncode=0,parent_id=q['parent_id'],stage=q['stage'])
d.owned_execute=owned
for label,adm in admitted.items():
 output=io.StringIO()
 with redirect_stdout(output):assert d.main(['--execute-ref',json.dumps(adm['input_ref'])])==0
 check(label+'_actual_execute_main_to_owned_RAM_boundary',json.loads(output.getvalue())['parent_id']+':'+json.loads(output.getvalue())['stage']==label)
# Invoke actual same-source worker routing and its unchanged main, replacing only the worker body boundary.
worker_frames=[]
def capture_exec(code,scope):
 builtins.exec(code,scope)
 if scope.get('__name__')=='_FLARE_same_source_private_original_worker':
  assert scope['d']is d
  def worker(record,gate_fd,ready_fd):
   s,q,cfg=c.input_spec(record);worker_frames.append((q['parent_id'],q['stage'],gate_fd,ready_fd));return dict(RAM_only=True)
  scope['worker']=worker
d.exec=capture_exec
for label,adm in admitted.items():
 output=io.StringIO()
 with redirect_stdout(output):assert d.main(['--worker-ref',json.dumps(adm['input_ref']),'--gate-fd','7','--ready-fd','8'])==0
 check(label+'_same_authenticated_entry_dispatch_to_original_worker_main_RAM_boundary',worker_frames[-1][:2]==tuple(label.split(':')))
del d.exec
# The exact derived v2 caller executes two cold transactions with extra exclusions and
# final source/bridge validation; transaction/append boundaries are RAM adapters.
bridge_module=actual_load(b['ROOT_memo_bridge_ref'],'_RAM_review_exact_Census_memo_source_bridge')
base_module=actual_load(b['ROOT_memo_source_ref'],'_RAM_review_exact_memo_v2_component')
request=prepared[c.VAL[0]+':native']['request_ref'];request_value=read(request)
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

negative=[]
def rejects(label,action):
 before=len(calls)
 try:action()
 except (ValueError,TypeError,KeyError):pass
 else:raise AssertionError('Expected fail closed: '+label)
 check(label,len(calls)==before);negative.append(label)
rejects('missing_selector',lambda:c.selected(None));rejects('Relate64_selector_no_fallback',lambda:c.selected(parent_selector))
inactive=ram_doc('inactive_selector.json',dict(sel,active=False));rejects('inactive_Census_selector',lambda:c.selected(inactive))
mixed=ram_doc('mixed_selector.json',dict(sel,source_ref=old['source_ref']));rejects('mixed_Census_Relate_source',lambda:c.selected(mixed))
other=ram_doc('same_contents_other_selector.json',sel);rejects('lifetime_selector_swap',lambda:c.bind_selector(other))
for stage in('native','seal'):
 label=c.VAL[0]+':'+stage;qref=prepared[label]['request_ref'];original=read(qref)
 for name,change in [('VAL001',dict(parent_id='american_c3_val_001')),('convert_stage',dict(stage='convert')),('partial_reuse',dict(reuse_partial_outputs=True)),('wrong_key',dict(proposed_reservation_key='H041/wrong')),('wrong_cap',dict(caps=dict(original['caps'],output_bytes=1))),('short_science',dict(phase_budget=dict(original['phase_budget'],scientific_seconds=1)))]:
  documents[key(qref)]=dict(original,**change);rejects(stage+'_'+name,lambda:c.validate_request(qref))
 documents[key(qref)]=original
 authority=authorities[label];v=read(authority);e_ref=v['previous_stage_evidence_ref'];e=read(e_ref)
 for field,subfield,value in [('actual_wait_ref','returncode',1),('actual_wait_ref','all_owned_processes_and_groups_drained',False),('outer_terminal_ref','actual_terminal_consumed',False),('ordinary_close_receipt_ref','artifacts_deleted',1),('input_ref','parent_id','american_c3_val_001'),('input_ref','source_sha256',{})]:
  r=e[field];original_doc=read(r);documents[key(r)]=dict(original_doc,**{subfield:value})
  rejects(stage+'_prereq_'+field+'_'+subfield,lambda:c.validate_request(qref));documents[key(r)]=original_doc
 for name,change in [('old_clock',dict(expires_at_utc=(now+timedelta(hours=9)).isoformat())),('wrong_context',dict(context_ref=contexts[c.VAL[1]])),('wrong_stage',dict(stage='convert')),('wrong_parent',dict(parent_id='american_c3_val_001')),('no_explicit_metadata',dict(one_stage_request_metadata_authorized=False))]:
  documents[key(authority)]=dict(v,**change);rejects(stage+'_instruction_'+name,lambda:c.validate_request(qref))
 documents[key(authority)]=v
# 65 must fail with no omission; a real additional current role cannot be hidden.
seal_label=c.VAL[0]+':seal';extra=ram_doc('additional_current_guard.json',dict(RAM_only=True))
rejects('all65distinct_current_roles_fail_closed',lambda:g.consumer_bindings_v1(list(guard_bindings.values())+[extra],direct_roles[seal_label]))
for label,roles in direct_roles.items():
 union=g.consumer_bindings_v1(list(guard_bindings.values()),roles)
 check(label+'_all_actual_guard_names_survive_dedup',all(r in union for r in guard_bindings.values()))
# Exact original worker bytes and lifecycle ASTs, allowing only module dispatch change.
newtree=ast.parse((D/'root_driver_v1.py').read_text());conv_tree=ast.parse(raw(b['conversion_driver_ref']))
nodes=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
nn=nodes(newtree);on=nodes(conv_tree)
for name,node in on.items():
 if name in('contract','main'):continue
 candidate=copy.deepcopy(nn['_driver_main_v1']if name=='main'else nn[name])
 baseline=copy.deepcopy(node)
 if name=='owned_execute':
  old_module=b['conversion_driver_ref']['path'].rsplit('/',1)[0]
  class ModuleOnly(ast.NodeTransformer):
   def visit_Constant(self,n):
    if n.value=='.worker_entry_v1':return ast.copy_location(ast.Constant('.root_driver_v1'),n)
    return n
  baseline=ModuleOnly().visit(baseline)
 check('exact_original_driver_AST_'+name,ast.dump(candidate,include_attributes=False)==ast.dump(baseline,include_attributes=False))
original_main=copy.deepcopy(on['main']);original_main.name='_driver_main_v1'
check('original_driver_main_AST_retained',ast.dump(original_main,include_attributes=False)==ast.dump(nn['_driver_main_v1'],include_attributes=False))
check('byte_exact_original_worker_embedded',hashlib.sha256(d.EMBEDDED_ORIGINAL_WORKER_UTF8.encode()).hexdigest()==b['predecessor_worker_ref']['sha256']and d.EMBEDDED_ORIGINAL_WORKER_UTF8.encode()==raw(b['predecessor_worker_ref']))
j=actual_load(b['operational_join_ref'],'_author_AST_join')
original_stage=next(n for n in ast.parse(raw(b['original_contract_ref'])).body if isinstance(n,ast.FunctionDef)and n.name=='stage_evidence')
derived=j.derive_conversion_evidence_node_v1(original_stage)
check('conversion_predecessor_only_three_comparison_operands',len([n for n in ast.walk(derived)if isinstance(n,ast.Name)and n.id in('_CONVERSION_DRIVER_REF','_CONVERSION_ENTRY_REF','_CONVERSION_SOURCE_SHA256')])==3)
check('no_guard_resource_payload_truth_or_numerical_runtime_imported',not any(n.startswith(('numpy','torch','scripts.project_budget'))for n in sys.modules))
result=dict(schema='FLARE_original002004_Census_native_seal_source_RAM_checks_v1',status='PASS_SOURCE_RAM_ONLY',checks=len(checks),cases=checks,negative_cases=negative,full_role_unions=role_counts,historical_guard_names=len(base_bindings),historical_distinct_guard_refs=len({key(r)for r in base_bindings.values()}),Census_guard_names=len(guard_bindings),Census_distinct_guard_refs=len({key(r)for r in guard_bindings.values()}),all_six_complete_preparer_admit_input_execute_and_same_source_worker_dispatch_routes=True,original_role_ceiling=64,actual_metadata_refs=list(actual_reads.values()),all_actual_guard_roles_retained=True,no_actual_guard_state_resource_process_payload_truth_numerical_admission_activation_or_launch=True,prerequisite_completions_markers_waits_drain_close_future_selector_and_review_are_RAM_fixtures=True)
(D/'source_checks.result.v1.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in('cases','actual_metadata_refs')},sort_keys=True))

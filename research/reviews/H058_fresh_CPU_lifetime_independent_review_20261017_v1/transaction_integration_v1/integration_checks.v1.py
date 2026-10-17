"""Independent integration-only AST, metadata and handwritten memory fixtures."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,stat
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE='research/reviews/H058_transaction_admission_source_20261017_v1'
OUT=ROOT/'research/reviews/H058_fresh_CPU_lifetime_independent_review_20261017_v1/transaction_integration_v1'
OLD='research/reviews/H058_fresh_CPU_lifetime_source_20261017_v2'
S=ROOT/HERE;B=json.loads((S/'source_bindings.v1.json').read_text());M=json.loads((S/'source_manifest.v1.json').read_text())
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 st=p.lstat();need(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size<=32*2**20,'source/metadata only')
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):need(ref(r['path'])==r,'exactSHA');return json.loads((ROOT/r['path']).read_text())
def raw(r):need(ref(r['path'])==r,'exactSHA');return (ROOT/r['path']).read_bytes()
def fn(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
def isolate(nodes,ns):
 exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),'<independent-integration-AST>','exec'),ns);return ns
def equal(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def rejects(action):
 try:action()
 except (ValueError,KeyError,TypeError):return True
 return False
def fake_ref(name):return dict(path=str(name),sha256=hashlib.sha256(str(name).encode()).hexdigest())
tree=ast.parse((S/'root_driver_v1.py').read_text());prior=ast.parse((ROOT/OLD/'root_driver_v1.py').read_text())
# Compare all retained nodes, independently removing only documented integration additions.
restored=copy.deepcopy(tree)
restored.body=[n for n in restored.body if not(isinstance(n,ast.FunctionDef) and n.name in ('_make_root_mutation_memo','_memo_admission_node','_install_memo_admit'))]
for n in restored.body:
 if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):n.value.value=prior.body[0].value.value
 if isinstance(n,ast.Import):n.names=[x for x in n.names if x.name!='ast']
 if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='HERE' for x in n.targets):n.value.value=OLD
 if isinstance(n,ast.FunctionDef) and n.name=='main':n.body=[x for x in n.body if not(isinstance(x,ast.If) and isinstance(x.test,ast.Attribute) and x.test.attr=='admit_ref')]
need(equal(restored,prior),'undocumented retained-body change')
need((S/'worker_entry_v1.py').read_bytes()==(ROOT/OLD/'worker_entry_v1.py').read_bytes(),'worker changed')
newb=copy.deepcopy(B);oldb=json.loads((ROOT/OLD/'source_bindings.v1.json').read_text())
for key in ['transaction_memo','fresh_lifetime_component','schema']:newb.pop(key,None);oldb.pop(key,None)
need(newb==oldb,'unrelated binding change')
audit=json.loads((S/'audit_manifest.v1.json').read_text())
for p,h in audit['files'].items():need(ref(p)['sha256']==h,'audit')
# Every component closure and component independent-review identity is genuinely sealed.
components={}
for name,spec in [('memo',B['transaction_memo']),('fresh_lifetime',B['fresh_lifetime_component'])]:
 for k,v in spec.items():
  if k.endswith('_ref'):
   need(M['source_sha256'].get(v['path'])==v['sha256'],'componenttransitive '+k);raw(v)
 cm=read(spec['source_manifest_ref']);refs=cm.get('refs',[])+[dict(path=p,sha256=h)for p,h in cm.get('source_sha256',{}).items()]
 missing=[r for r in refs if M['source_sha256'].get(r['path'])!=r['sha256']]
 need(not missing,'component closure omission')
 rv=read(spec['independent_review_ref']);need(rv['independent']is True and rv['verdict']=='SOURCE_PASS' and rv['source_ref']==spec['source_ref'] and rv['blockers']==[],'actualcomponentpass')
 components[name]=dict(source_ref=spec['source_ref'],review_ref=spec['independent_review_ref'],component_closure_refs=len(refs),all_transitively_bound=True)
common_review_ref=dict(path='research/reviews/admission_repairs_independent_20261017_v1/whole_integration_review.v1.json',sha256='505d926a4bc37deb23891d7de3f48bb8adfb6c7477d539d6ffb2e00d7c6a0cb7')
common_review=read(common_review_ref);need(common_review['verdict']=='SOURCE_PASS' and common_review['independent']is True and common_review['blockers']==[],'accepted common integration review')
common_source_ref=common_review['source_ref'];common=ast.parse(raw(common_source_ref))
need(equal(fn(tree,'_make_root_mutation_memo'),fn(common,'_make_root_mutation_memo')),'acceptedconstructorhelper changed')
# Derive only actual AST builder in isolation. Reverse three changes and compare entire body.
builder=isolate([fn(tree,'_memo_admission_node')],dict(need=need,ast=ast,copy=copy))['_memo_admission_node']
original_tree=ast.parse(raw(B['driver_ref']));original=fn(original_tree,'admit');derived=builder(original,'H058')
need(not any(isinstance(x,ast.With) for x in ast.walk(derived)),'caller-heldmutex')
reverted=copy.deepcopy(derived);added=[i for i,n in enumerate(reverted.body)if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='_make_root_mutation_memo']
need(len(added)==1,'oneconstructor');reverted.body.pop(added[0])
statecount=reservecount=0
for i,n in enumerate(reverted.body):
 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute):
  if ast.unparse(n.value.func)=='memo.read_state_v1':reverted.body[i]=copy.deepcopy(next(x for x in original.body if isinstance(x,ast.With)));statecount+=1
  if ast.unparse(n.value.func)=='memo.reserve_v1':n.value.func=ast.Attribute(value=ast.Name(id='g',ctx=ast.Load()),attr='apply_reserve_v1',ctx=ast.Load());reservecount+=1
need(statecount==reservecount==1 and equal(reverted,original),'morethanthreeadmission substitutions')
gi=next(i for i,n in enumerate(derived.body)if isinstance(n,ast.Assign) and ast.unparse(n.value)=='guard(False)')
need(ast.unparse(derived.body[gi+1])=='memo = _make_root_mutation_memo(g, q)','constructorposition')
firstput=next(i for i,n in enumerate(derived.body)if any(isinstance(x,ast.Call) and ast.unparse(x.func)=='put' for x in ast.walk(n)))
need(gi+1<firstput,'memoafterwrite')
for mode in ('wrongconsumer','wrongname','decorated','missingguard','missingstate','missingreserve','extrawith'):
 altered=copy.deepcopy(original);consumer='H058'
 if mode=='wrongconsumer':consumer='H060'
 if mode=='wrongname':altered.name='other'
 if mode=='decorated':altered.decorator_list=[ast.Name(id='decorator',ctx=ast.Load())]
 if mode=='missingguard':altered.body=[n for n in altered.body if not(isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and ast.unparse(n.value)=='guard(False)')]
 if mode=='missingstate':altered.body=[n for n in altered.body if not isinstance(n,ast.With)]
 if mode=='missingreserve':altered.body=[n for n in altered.body if not(isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='g.apply_reserve_v1')]
 if mode=='extrawith':altered.body.append(ast.parse('with extra_lock():\n pass').body[0])
 need(rejects(lambda:builder(altered,consumer)),'builderfailclosed '+mode)
# Closure admission helper preflights all five refs/review BEFORE any memo module import.
helper=fn(tree,'_make_root_mutation_memo');helper_cases=[]
for mode in ['ok']+['missing_'+k for k in B['transaction_memo']]+['badreview']:
 spec=B['transaction_memo'];pins={r['path']:r['sha256']for r in spec.values()};events=[]
 # Deliberately include source, JSON, data and noncode noise to test allowlist extraction.
 for name in ('src/fixture.py','scripts/fixture.sh','research/reviews/fixture.py','research/reviews/fixture.json','data/fixture.py','.runtime/fixture.py','research/reviews/fixture.md'):pins[name]='a'*64
 rv=dict(verdict='SOURCE_PASS',source_only=True,source_ref=spec['source_ref'])
 if mode.startswith('missing_'):pins.pop(spec[mode[len('missing_'):]]['path'])
 if mode=='badreview':rv['verdict']='SOURCE_BLOCKED'
 q=dict(source_manifest_ref=fake_ref('fixture/manifest.json'))
 def reader(record):return dict(source_sha256=pins) if record==q['source_manifest_ref'] else rv
 def loader(record,name):
  events.append('import')
  def construct(g,**kw):
   events.append('construct');expected=sorted(p for p in pins if Path(p).suffix in ('.py','.sh') and p.split('/')[0]in ('src','scripts','research'))
   need([r['path']for r in kw['cache_refs']]==expected,'cachewhitelist');need(kw['integration_manifest_ref']==q['source_manifest_ref'] and kw['adapter_ref']==spec['source_ref'],'exactconstructorjoins');return kw
  return SimpleNamespace(RootMutationMemo=construct)
 f=isolate([helper],dict(bindings=lambda:B,read=reader,raw=lambda _:None,need=need,Path=Path,load=loader))['_make_root_mutation_memo']
 if mode=='ok':f(object(),q);need(events==['import','construct'],'success')
 else:need(rejects(lambda:f(object(),q)) and events==[],'preimportreject '+mode)
 helper_cases.append(dict(case=mode,events=events))
# Execute exact derived whole admission on in-memory path, memo and storage doubles only.
traces=[]
for fail_stage in ('constructor','state','reserve','none'):
 events=[];stored={};created=set();q=dict(packet_directory='research/reviews/H058_CPU_ROOT_REVIEW_ONLY',output_directory='experiments/results/H058_full_parent_CPU_REVIEW_ONLY',preparation_directory='research/reviews/H058_CPU_ROOT_preparation_REVIEW_ONLY',cpu_ids=list(range(160,192)),scientific_config={},expires_at_utc='2026-10-20T00:00:00+00:00',created_at_utc='2026-10-17T00:00:00+00:00')
 class FakePath:
  def __init__(self,p):self.p=str(p);self.name=self.p.rsplit('/',1)[-1]
  def exists(self):return self.p in created
  def mkdir(self):events.append('mkdir_packet');created.add(self.p)
  def __truediv__(self,v):return FakePath(self.p+'/'+str(v))
  def __str__(self):return self.p
  def __fspath__(self):return self.p
 state=({},None,{},{});caps=B['caps'];request=fake_ref('fixture/request.json');q['runtime_ref']=fake_ref('fixture/runtime.json')
 def put_base(path,value,cap=2**20):
  path=str(path);events.append('put:'+Path(path).name);stored[path]=copy.deepcopy(value);return fake_ref(path)
 c=SimpleNamespace(require=need,ref=fake_ref,ENTRY=HERE+'/worker_entry_v1.py',PHASE='train',PARENT='H058_European32_full_TRAIN_CPU',POLICY_REF=fake_ref('fixture/policy.json'),SELECTOR_REF=fake_ref('fixture/selector.json'),GUARD_REF=B['selected_child_source_ref'])
 contract=fn(tree,'contract');putnode=next(n for n in contract.body if isinstance(n,ast.FunctionDef)and n.name=='put')
 put=isolate([putnode],dict(c=c,Path=Path,SOURCE=HERE+'/root_driver_v1.py',original_put=put_base))['put']
 c.validate_request=lambda r:(events.append('validate')or(q,caps,{}))
 c.make_config=lambda *_:dict(scientific_config_ref=fake_ref('fixture/science_config.json'))
 c.sources=lambda:dict(source_sha256=M['source_sha256'])
 c.direct_roles=lambda *_:[fake_ref(HERE+'/source_manifest.v1.json')]
 def checkinput(ir):
  s=stored[ir['path']];need(s['hypothesis_id']=='H058' and s['root_driver_ref']==fake_ref(HERE+'/root_driver_v1.py') and s['main_source_ref']==fake_ref(HERE+'/worker_entry_v1.py'),'admittedjoins');events.append('input_gate')
 c.input_spec=checkinput
 def forbidden(*args,**kw):raise AssertionError('originalmutex/state/reserve should not be called')
 v4=SimpleNamespace(mutex=forbidden)
 def scope_spec(root,reserve):events.append('scope_spec');return [FakePath(reserve['output']),FakePath(reserve['auxiliary_scopes'][0])]
 v4.scope_spec=scope_spec
 g=SimpleNamespace(budget=SimpleNamespace(v4=v4,SOURCE_REL='scripts/project_budget_v9.py',LAUNCHER_REL='scripts/launch_gpu_job.py'),MODE='Relate_failed64GiB_retention_v1',_state_locked=forbidden,apply_reserve_v1=forbidden)
 g.consumer_bindings_v1=lambda guards,roles:(events.append('all_roles')or roles)
 g._policy=lambda:dict(never_credit_keys=[]);g._pins=lambda _:dict()
 def draft(*args,**kwargs):events.append('authority_draft');return {}
 g.operation_draft_v1=draft
 def guard(flag):need(flag is False,'guardFalse');events.append('guard_false');return g
 def factory(guard,q):
  events.append('memo_constructor')
  if fail_stage=='constructor':raise ValueError('fixtureconstructor')
  def getstate(**kw):
   events.append('memo_state')
   if fail_stage=='state':raise ValueError('fixturestate')
   return state
  def reserve(ar,**kw):
   events.append('memo_reserve')
   if fail_stage=='reserve':raise ValueError('fixturereserve')
   return dict(value=dict(reserved_bytes=caps['reservation_bytes']),initial_ledger_pins={})
  return SimpleNamespace(read_state_v1=getstate,reserve_v1=reserve)
 derivedfn=isolate([derived],dict(c=c,local=FakePath,need=need,os=SimpleNamespace(sched_getaffinity=lambda _:set(range(160,192))),guard=guard,_make_root_mutation_memo=factory,put=put,ref=fake_ref,ROOT=ROOT,HERE=HERE,SOURCE=HERE+'/root_driver_v1.py',json=json))['admit']
 if fail_stage=='none':
  result=derivedfn(request);need(result['execution_argv'][5]==HERE.replace('/','.')+'.root_driver_v1' and result['admission_only']is True,'emittedcurrentmodule')
  indices=[events.index(x)for x in ('memo_constructor','put:'+q['packet_directory'].rsplit('/',1)[-1]+'.scientific_config.json','memo_state','all_roles','scope_spec','authority_draft','memo_reserve','mkdir_packet','input_gate')]
  need(indices==sorted(indices),'admissionorder')
 else:
  need(rejects(lambda:derivedfn(request)),'failureexpected')
  if fail_stage=='constructor':need(not stored and not created and 'authority_draft'not in events,'prewritefailure')
  if fail_stage=='state':need('memo_state'in events and 'authority_draft'not in events and not created,'statefailpreauthority')
  if fail_stage=='reserve':need('memo_reserve'in events and not created and 'input_gate'not in events,'reservefailbeforepacketinput')
 traces.append(dict(failure_stage=fail_stage,events=events,fixture_files_created_on_disk=False))
# Verify no caller-held lock and exact component lock/teardown invariants by source AST.
memo=ast.parse(raw(B['transaction_memo']['source_ref']));cl=next(n for n in memo.body if isinstance(n,ast.ClassDef) and n.name=='RootMutationMemo');run=fn(cl,'_run')
locks=[n for n in ast.walk(run)if isinstance(n,ast.With)];need(len(locks)==1 and ast.unparse(locks[0].items[0].context_expr)=='original_mutex(ROOT)','componentlock')
cache_refs=[dict(path=p,sha256=h)for p,h in sorted(M['source_sha256'].items())if Path(p).suffix in ('.py','.sh')and p.split('/')[0]in('src','scripts','research')]
result=dict(schema='H058_transaction_integration_independent_checks_v1',source_ref=ref(S/'root_driver_v1.py'),source_manifest_ref=ref(S/'source_manifest.v1.json'),source_handoff_ref=ref(S/'source_handoff.v1.json'),audit_manifest_ref=ref(S/'audit_manifest.v1.json'),source_only=True,all_audit_refs_authenticate=True,retained_v2_AST_exact=True,worker_bytes_exact=True,prior_bindings_exact_except_documented_component_fields=True,components=components,exact_constructor_helper_matches_accepted_common_integration=True,common_integration_comparison_review_ref=common_review_ref,common_integration_comparison_source_ref=common_source_ref,three_admission_substitutions_only=True,derived_admission_has_no_mutex=True,builder_failclosed_cases=7,helper_preimport_cases=helper_cases,memory_admission_traces=traces,original_union_scope_authority_reserve_order=True,transitive_memo_no_extra_direct_role=True,explicit_code_only_cache_refs=len(cache_refs),actual_guard_or_mutex_or_state_import=False,actual_admission_resources_payload_numerics_or_Git=False)
(OUT/'integration_checks.v1.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(source_ref=result['source_ref'],integration_AST_and_memory_fixtures_pass=True,audit_refs=len(audit['files']),code_cache_refs=len(cache_refs),helper_cases=len(helper_cases),derived_admission_trace_cases=len(traces)),sort_keys=True))

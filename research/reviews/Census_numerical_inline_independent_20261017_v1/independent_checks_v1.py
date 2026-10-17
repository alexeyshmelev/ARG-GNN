"""Independent source/RAM review of numerical callback and one-shot orchestration. No actual operations."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace,ModuleType
import ast,builtins,contextlib,copy,hashlib,io,json,os,stat,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/Census_numerical_inline_ROOT_orchestration_source_20261017_v1';OUT=ROOT/'research/reviews/Census_numerical_inline_independent_20261017_v1'
checks=[];negative=[]
def ck(name,ok=True):
 if not ok:raise AssertionError(name)
 checks.append(dict(name=name,status='PASS'))
def reject(name,call):
 try:call()
 except(ValueError,TypeError,KeyError,OSError,TimeoutError,AssertionError):negative.append(name);ck(name)
 else:raise AssertionError(name+' accepted')
def rf(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def doc(p):return json.loads(p.read_bytes())
def execute(nodes,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),'independent_Census_AST_RAM','exec'),scope)
pins={'source_handoff.v1.json':'fbc6972c2151174718a3b25db0bddeacc8ccf847aad09b7441d89245f1b59398','source_manifest.v1.json':'3956ab9d5dfa023b98eea29b24d763c0d5c25fe846f8eda558c71cd3e800bf6e','ROOT_authorize_inline_retirement_v1.py':'c3f6dbb30968d0acefb28c2de36fff4f7a006cd9d0c4076ce106075eab9e1d63','orchestrate_v1.py':'70d5b5d2afa3a9252864af51edb1ac0173a7b1aa10b2283bc014fc61f963eede'}
for f,h in pins.items():ck('requested exact '+f,rf(D/f)['sha256']==h)
generic=ROOT/'research/reviews/Census_retention_inline_independent_20261017_v1/review.v1.json'
ck('exact independent generic helper SOURCE_PASS',rf(generic)['sha256']=='ab7ec25225876da84f1e061252601cb4a85df3f6f25a784d1385e70ec4b4f142'and doc(generic)['verdict']=='SOURCE_PASS')
mp=doc(D/'integration_manifest.INACTIVE.template.v1.json')['source_sha256']
aliases={r['to']['path']:r['from']['path']for r in doc(D/'staging_layout.v1.json')['byte_identical_copies']}
verified={}
for p,h in mp.items():
 actual=ROOT/aliases.get(p,p);assert actual.is_file()and not actual.is_symlink()and actual.resolve()==actual and actual.stat().st_size<=32*2**20 and actual.suffix in('.py','.json','.md','.txt','.patch','.diff')and rf(actual)['sha256']==h;verified[p]=h
ck('all218 integration pins including exact future byte aliases verify',len(mp)==218)
sm=doc(D/'source_manifest.v1.json')
for p,h in sm['authored_file_sha256'].items():assert rf(ROOT/p)['sha256']==h
ck('all authored packet pins verify')
inherited=doc(ROOT/'research/reviews/Census_retention_inline_source_20261017_v1/source_manifest.v1.json')['source_sha256']
ck('all210 original helper pins retained',len(inherited)==210 and all(mp.get(p)==h for p,h in inherited.items()))
for f in('ROOT_authorize_inline_retirement_v1.py','orchestrate_v1.py'):compile((D/f).read_bytes(),str(D/f),'exec')
ck('actual callback and entry compile inertly')
# Extract setup and RAM classes from author checker, excluding every case/campaign/writer.
at=ast.parse((D/'source_check_v1.py').read_text());run=copy.deepcopy(next(n for n in at.body if isinstance(n,ast.FunctionDef)and n.name=='run'))
first_case=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='callback_case')
setup=run.body[:first_case]
cb=next(n for n in setup if isinstance(n,ast.FunctionDef)and n.name=='callback_case')
first_call=next(i for i,n in enumerate(cb.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='failed' for t in n.targets))
cb.body=cb.body[:first_call]+ast.parse('return locals()').body
run.body=setup+ast.parse('return locals()').body
ns={'__name__':'independent_partial_fixture'}
execute([n for n in at.body if n.end_lineno<=10]+[run],ns);fix=ns['run']()
on=fix['on'];hn=fix['hn'];FS=fix['MemoryFS'];src=fix['source'];original_target=fix['target']
# Original semantic derivation is exactly the three declared additions.
gsource=(ROOT/hn['GUARD_REF']['path']).read_text();gn=next(n for n in ast.parse(gsource).body if isinstance(n,ast.FunctionDef)and n.name=='apply_retire_stopped_v1')
derived=hn['derive_inline_retirement'](gn,{},None,None,None);tree=copy.deepcopy(gn);tree.name='_inline_ROOT_exact_retirement'
call=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='_append');call.func.id='_retirement_append';call.args[:0]=[ast.Name(id='finite',ctx=ast.Load()),ast.Name(id='auth',ctx=ast.Load())]
tree.body[-1].body.insert(2,ast.parse('authority_ref = _inline_ROOT_authority(state, finite)').body[0]);tree.body[-1].body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='_retirement_transaction',ctx=ast.Load()),args=[ast.Name(id='finite',ctx=ast.Load())],keywords=[]),optional_vars=None)],body=tree.body[-1].body)]
manual={};exec(compile(ast.fix_missing_locations(ast.Module(body=[tree],type_ignores=[])),derived.__code__.co_filename,'exec'),manual)
ck('entire original retirement derivation equals independent exact3seam construction',manual[tree.name].__code__==derived.__code__)
# RAM actual callback -> actual helper authority seam -> original authority validator.
def seam_case(label='valid',fault=None,edit=None):
 z=fix['callback_case'](label,fault=fault);g=z['g'];fs=z['fs'];ctx=z['ctx'];spec=z['spec']
 if edit:edit(spec,ctx)
 def bound(root,r,as_json=False):
  data=fs.records[str(root/r['path'])]['data'];assert hashlib.sha256(data).hexdigest()==r['sha256'];return json.loads(data)if as_json else data
 def reference(root,p):
  data=fs.records[str(root/p)]['data'];return {'path':p,'sha256':hashlib.sha256(data).hexdigest()}
 g._pins=lambda v:copy.deepcopy(v);g.assert_open=lambda k,r,c:None
 original_funcs={n.name:n for n in ast.parse(gsource).body if isinstance(n,ast.FunctionDef)}
 cg={'require':lambda v,m:None if v else(_ for _ in()).throw(ValueError(m)),'_raw':lambda r:None,'copy':copy}
 execute([original_funcs['consumer_bindings_v1']],cg);g.consumer_bindings_v1=cg['consumer_bindings_v1']
 obj=hn['InlineRootRetireStoppedMemo'].__new__(hn['InlineRootRetireStoppedMemo'])
 obj.guard=g;obj.budget=g.budget;obj.v4=g.budget.v4;obj.paths=on['FRESH'];obj.key=on['KEY'];obj.instruction_ref=ctx['root_instruction_ref'];obj.revocation_ref=on['REVOCATION'];obj.callback_ref=fix['callback_ref']
 obj.manifest_ref=spec['source_manifest_ref'];obj.review_ref=spec['whole_join_review_ref'];obj.memo_review_ref=spec['memo_review_ref'];obj.helper_ref=spec['helper_ref'];obj._callback_calls=0;obj._authority_ref=None
 obj._tx=SimpleNamespace(original={'bound':bound,'reference':reference});obj._authenticate=lambda reader:spec;obj._check_callback=lambda:None
 hn['os']=fs.fake_os();obj.callback=z['cbns']['authorize_inline_ROOT_retirement_v1']
 state=({obj.key:copy.deepcopy(original_target['original_reservation'])},set(),copy.deepcopy(ctx['expected_before']),copy.deepcopy(ctx['bindings']),[],[],None)
 if label=='state_prefix_drift':state[2]={'wrong':1}
 if label=='state_role_omission':state[3].pop(next(iter(state[3])))
 if label=='full64beforefuture':g.consumer_bindings_v1=lambda *args:[{'path':'RAM/'+str(i),'sha256':'0'*64}for i in range(64)]
 r=obj._inline_ROOT_authority(state,lambda:None)
 ck('actual callback/helper seam exact original authority schema',g._authority(r,g.RETIRE_OPERATION,state[3],fresh=False)['request']['key']==on['KEY'])
 ck('sole helper attempt plus callback observation authority records',set(fs.records)=={str(ROOT/p)for p in on['FRESH'].values()}and obj._callback_calls==1)
 return obj,g,z,r,state
obj,g,z,ar,state=seam_case()
for label in('state_prefix_drift','state_role_omission','full64beforefuture'):reject('before first helper write rejects '+label,lambda label=label:seam_case(label))
for fault in('existing','short_write','name_swap','fsync'):reject('actual attempt/callback fails safely on '+fault,lambda fault=fault:seam_case(fault,fault))
# Actual new callback explicit guards, distinct from generic helper fixture design.
for label,edit in[
 ('wrong scope',lambda s,c:c.update(key='other/key')),('wrong callback prefix',lambda s,c:c.update(expected_before={'drift':1})),
 ('omitted role',lambda s,c:c['bindings'].pop(next(iter(c['bindings'])))),('authority alternate path',lambda s,c:c['fresh_metadata_paths'].update(authority='RAM/other.json')),
 ('reservation changed',lambda s,c:c['original_reservation'].update(reserved_bytes=1))]:
 zz=fix['callback_case'](label);edit(zz['spec'],zz['ctx'])
 reject('actual callback rejects '+label,lambda zz=zz:zz['cbns']['authorize_inline_ROOT_retirement_v1'](guard=zz['g'],context=zz['ctx'],finite=lambda:None))
# Exact authored instruction/review/intent DAG filled ONLY in RAM with marked synthetic prefix/roles.
ram={}
def putram(path,v):
 data=(json.dumps(v,sort_keys=True,indent=2)+'\n').encode();ram[path]=data;return {'path':path,'sha256':hashlib.sha256(data).hexdigest()}
def ramraw(r):
 data=ram[r['path']]if r['path']in ram else (ROOT/r['path']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==r['sha256'];return data
def ramdoc(r):return json.loads(ramraw(r))
manifest=doc(D/'integration_manifest.INACTIVE.template.v1.json');ram[on['MANIFEST']]=(D/'integration_manifest.INACTIVE.template.v1.json').read_bytes();ram[on['SOURCE']]=(D/'orchestrate_v1.py').read_bytes();ram[on['CALLBACK']]=src.encode()
it=doc(D/'ROOT_instruction.INACTIVE.template.v1.json');rt=doc(D/'whole_join_review.INACTIVE.template.v1.json');intent=doc(D/'ROOT_actual_operation_intent.INACTIVE.template.v1.json')
ck('inactive templates contain no genuine prefix roles clock or approval',it['active']is False and it['expected_before']is None and it['expected_guard_roles']is None and it['created_at_utc']is None and rt['independent']is False and intent['ROOT_authorizes_one_actual_original_Census_retirement']is False)
created=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();expires=(datetime.now(timezone.utc)+timedelta(seconds=300)).isoformat();roles=copy.deepcopy(fix['roles']);prefix={'RAM_ONLY_NOT_ACTUAL_PREFIX':1}
rt.update(independent=True,verdict='SOURCE_PASS',blockers=[],actual_callback_complete_source_review=True,orchestration_complete_source_review=True,expected_before=prefix,expected_guard_roles=roles);rr=putram(on['REVIEW'],rt)
intent.update(active=True,root_authorizer='root_agent',whole_join_review_ref=rr,expected_before=prefix,expected_guard_roles=roles,created_at_utc=created,expires_at_utc=expires,ROOT_authorizes_one_actual_original_Census_retirement=True);ir=putram(on['INTENT'],intent)
it.update(active=True,root_authorizer='root_agent',whole_join_review_ref=rr,expected_before=prefix,expected_guard_roles=roles,created_at_utc=created,expires_at_utc=expires);it['uncached_current_refs'][0]=ir;iref=putram(on['INSTRUCTION'],it)
on['document']=ramdoc;on['raw']=ramraw
ck('exact complete218pin controls and separate intent DAG accepts in RAM only',on['controls'](iref,ir)==it)
ht=hn['InlineRootRetireStoppedMemo'].__new__(hn['InlineRootRetireStoppedMemo'])
for attr,key in [('key','key'),('manifest_ref','source_manifest_ref'),('helper_ref','helper_ref'),('review_ref','whole_join_review_ref'),('memo_review_ref','memo_review_ref'),('callback_ref','callback_source_ref'),('callback_name','callback_name'),('cache_refs','cache_refs'),('current_refs','uncached_current_refs'),('paths','fresh_metadata_paths'),('metadata_directory','metadata_directory'),('revocation_ref','permanent_revocation_ref')]:setattr(ht,attr,copy.deepcopy(it[key]))
ht.instruction_ref=iref;fs=FS();ht.v4=SimpleNamespace(canonical=lambda root,p:fs.path(str(root/p)));ht.memo=SimpleNamespace(source_bytes=ramraw);ht.guard=SimpleNamespace(_expire=g._expire)
ck('actual helper complete authored metadata predicate joins RAM controls',ht._authenticate(lambda root,r,as_json:ramdoc(r))==it)
for label,edit in[
 ('no intent exclusion',lambda v:v.update(uncached_current_refs=[])),('46roles',lambda v:v['expected_guard_roles'].pop(next(iter(v['expected_guard_roles'])))),
 ('wrong300s',lambda v:v.update(operation_wall_seconds=301)),('wrong120s',lambda v:v.update(fresh_target_seconds=121))]:
 bad=copy.deepcopy(it);edit(bad);old=ram[on['INSTRUCTION']];br=putram(on['INSTRUCTION'],bad)
 reject('orchestration controls reject '+label,lambda br=br:on['controls'](br,ir));ram[on['INSTRUCTION']]=old
# Handwritten orchestration effects: exact run_once body, no guard or helper operation imports.
base_run=on['run_once'];outcomes=[]
for mode in('success','construct_failure','late_append_error','receipt_failure','result_failure','invocation_uncertain','failure_persist_uncertain'):
 effects={};loads=[];calls=[];paths=fs
 class P:
  def __new__(cls,p):return fs.path(p)
  @staticmethod
  def cwd():return fs.path(ROOT)
 on['Path']=P;on['__file__']=str(ROOT/on['SOURCE']);on['sys']=SimpleNamespace(flags=SimpleNamespace(optimize=0,no_user_site=1),dont_write_bytecode=True,modules={},path=[]);on['threading']=SimpleNamespace(active_count=lambda:1);on['os']=SimpleNamespace(getpid=lambda:9)
 def put(kind,v):
  if kind in effects:raise FileExistsError('RAM one-shot '+kind)
  effects[kind]=copy.deepcopy(v)
  if mode=='invocation_uncertain'and kind=='invocation':raise OSError('RAM partial invocation persistence')
  if mode in('receipt_failure','result_failure')and kind==mode.split('_')[0]:raise OSError('RAM after append persistence uncertainty')
  if mode=='failure_persist_uncertain'and kind=='failure':raise OSError('RAM failure persistence')
  return {'path':on['NAMES'][kind],'sha256':'a'*64}
 class Helper:
  counts={'RAM':1}
  def __init__(self,guard,**kw):
   calls.append('construct')
   if mode=='construct_failure':raise ValueError('RAM constructor refusal')
  def apply_retire_stopped_v1(self,**kw):
   assert kw==dict(expected_source_sha256=on['IMPLEMENTATION_SHA'],expected_launcher_sha256=on['LAUNCHER_SHA']);calls.append('apply')
   if mode in('late_append_error','failure_persist_uncertain'):raise OSError('RAM uncertain append')
   return dict(kind='RAM_ONLY_NOT_CREDIT',retained_bytes=1,credit_bytes=2,projected_bytes=3)
 def loader(record,name):
  loads.append(record)
  return SimpleNamespace(load_selected_guard_v1=lambda r:SimpleNamespace())if record==on['GUARD']else SimpleNamespace(InlineRootRetireStoppedMemo=Helper)
 on['put']=put;on['load']=loader
 failed=False
 with contextlib.redirect_stdout(io.StringIO()):
  try:answer=base_run(instruction_ref=iref,intent_ref=ir)
  except(ValueError,OSError):failed=True
 ck('one-shot orchestration material outcome '+mode,failed is(mode!='success')and calls.count('apply')<=1 and all(v.get('credit_accepted')is False for k,v in effects.items()if k!='receipt'))
 count=len(loads)
 with contextlib.redirect_stdout(io.StringIO()):reject('same invocation cannot replay '+mode,lambda:base_run(instruction_ref=iref,intent_ref=ir))
 ck('replay enters no guard/helper '+mode,len(loads)==count)
 if mode not in('success','invocation_uncertain'):ck('failure requires original journal audit '+mode,effects['failure']['append_or_late_failure_requires_original_journal_audit']is True)
 outcomes.append(dict(mode=mode,failed=failed,helper_apply_count=calls.count('apply')))
ck('no scientific frameworks imported',not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy','tskit'))for n in sys.modules))
result=dict(schema='Census_numerical_callback_orchestration_independent_RAM_checks_v1',status='PASS',independent=True,source_only=True,assertions=len(checks),negative_fixtures=len(negative),checks=checks,negative_cases=negative,orchestration_cases=outcomes,all218pins_verified=True,actual_whole_capability_supplied=False,actual_prefix_roles_or_authority_created=False,partial_author_RAM_class_setup_reused=True,author_campaign_or_writer_executed=False,actual_guard_state_accounting_mutex_process_absence_measurement_activation_GPU_Git_operations=False)
with(OUT/'checks.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in('checks','negative_cases','orchestration_cases')},sort_keys=True))


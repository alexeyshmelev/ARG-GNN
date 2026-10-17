"""Independent narrow source/AST/RAM review; no candidate module/framework/guard/resource imports."""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import ast,copy,hashlib,json,re
R=Path('/workspace-SR003.nfs2/ARG-GNN');D=R/'research/reviews/H058_lean_math_guard_design_source_20261017_v1';O=R/'research/reviews/H058_lean_math_guard_design_independent_20261017_v1'
checks=[];negatives=[]
def check(name,ok=True):assert ok,name;checks.append(name)
def denied(name,operation,types=(ValueError,RuntimeError)):
 try:operation()
 except types:check(name);negatives.append(name)
 else:raise AssertionError('Expected fail closed: '+name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),sha256=sha(p.read_bytes()))
sealed=json.loads((D/'sealed_manifest.v1.json').read_text())
for row in sealed['files']:
 raw=(D/row['path']).read_bytes();check('sealed_'+row['path'],len(raw)==row['bytes']and sha(raw)==row['sha256'])
pins=json.loads((D/'input_pins.v1.json').read_text())['refs'];drift=[]
for row in pins:
 raw=(R/row['path']).read_bytes()
 if sha(raw)==row['sha256']:check('current_original_pin_'+row['path'])
 else:
  check('historical_AGENTS_exact_prefix_preserved',row['path']=='AGENTS.md'and sha(raw[:row['bytes']])==row['sha256']);drift.append(dict(path=row['path'],historical_sha256=row['sha256'],current_sha256=sha(raw),append_only=True))
# Apply only the exact diff to RAM strings, never live paths.
patch=(D/'integration.INACTIVE.v1.patch').read_text().splitlines();candidate={};i=0
while i<len(patch):
 assert patch[i].startswith('--- a/');path=patch[i][6:];i+=1;assert patch[i]=='+++ b/'+path;i+=1
 original=(R/path).read_text().splitlines(keepends=True);out=[];cursor=0
 while i<len(patch)and not patch[i].startswith('--- a/'):
  match=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',patch[i]);assert match
  start=int(match.group(1))-1;out.extend(original[cursor:start]);cursor=start;i+=1
  while i<len(patch)and not patch[i].startswith(('@@','--- a/')):
   line=patch[i];assert line and line[0]in' +-'
   if line[0]in' -':assert original[cursor].rstrip('\n')==line[1:],(path,cursor);cursor+=1
   if line[0]in' +':out.append(line[1:]+'\n')
   i+=1
 out.extend(original[cursor:]);candidate[path]=''.join(out)
preview=json.loads((D/'integration_preview.v1.json').read_text())
for row in preview['files']:
 original=(R/row['path']).read_bytes()
 check('live_original_'+Path(row['path']).name,sha(original)==row['original_sha256'])
 check('RAM_candidate_'+Path(row['path']).name,sha(candidate[row['path']].encode())==row['prospective_sha256'])
 compile(candidate[row['path']],'<RAM candidate source>','exec')
def dump(n):return ast.dump(n,include_attributes=False)
def functions(source):return {n.name:n for n in ast.walk(ast.parse(source))if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
base=preview['base'];old={n:functions((R/base/n).read_text())for n in('context_factory_v1.py','producer_v1.py','root_driver_v1.py')}
new={n:functions(candidate[base+'/'+n])for n in old}
check('full_context_guard_unchanged',dump(old['context_factory_v1.py']['guard'])==dump(new['context_factory_v1.py']['guard']))
math=copy.deepcopy(new['context_factory_v1.py']['math_guard']);math.name='guard'
full=copy.deepcopy(old['context_factory_v1.py']['guard']);full.body=full.body[:-1]
check('math_guard_only_omits_final_historical_budget_replay',dump(math)==dump(full))
check('math_still_charged_census',ast.unparse(math.body[-1])=='charged_census_v1(spec)')
check('resource_closure_and_before_episode_unchanged',all(dump(old['context_factory_v1.py'][n])==dump(new['context_factory_v1.py'][n])for n in('resources','before_episode')))
for name,node in old['producer_v1.py'].items():
 if name in('scheduled_step_v1','counted','capture','observe'):continue
 check('producer_function_unchanged_'+name,dump(node)==dump(new['producer_v1.py'][name]))
for name in('counted','capture','observe'):
 check('scheduled_hook_callback_unchanged_'+name,dump(old['producer_v1.py'][name])==dump(new['producer_v1.py'][name]))
actual=new['producer_v1.py']['actual_step'];call=actual.body[0].value
original_call=next(n for n in ast.walk(old['producer_v1.py']['scheduled_step_v1'])if isinstance(n,ast.Call)and ast.unparse(n.func)=='h45.train_one')
check('all_five_scientific_arguments_and_both_keywords_exact',list(map(dump,call.args[:-1]))==list(map(dump,original_call.args[:-1]))and list(map(dump,call.keywords))==list(map(dump,original_call.keywords))and ast.unparse(call.args[-1])=='bound_guard')
check('both_cache_validators_retained',sum(isinstance(n,ast.Call)and ast.unparse(n.func)=='observations.validate_scheduled_cache_v1'for n in ast.walk(new['producer_v1.py']['scheduled_step_v1']))==2)
oldroot=ast.parse((R/base/'root_driver_v1.py').read_text());newroot=ast.parse(candidate[base+'/root_driver_v1.py'])
envcall="env.update(CUDA_CACHE_DISABLE='1', USE_PYTORCH_KERNEL_CACHE='0', PYTORCH_JIT='0', TORCH_COMPILE_DISABLE='1', TORCHDYNAMO_DISABLE='1')"
class RemoveEnvironment(ast.NodeTransformer):
 def visit_Expr(self,n):
  return None if isinstance(n.value,ast.Call)and ast.unparse(n.value)==envcall else self.generic_visit(n)
check('whole_driver_AST_only_cache_environment_delta',dump(oldroot)==dump(RemoveEnvironment().visit(copy.deepcopy(newroot))))
# Execute only copied helper AST under fake sys/os/fcntl/threading/torch interfaces.
# There is no imported candidate module and no actual audit hook, FD scan, resource callback or numerical work.
helper=(D/'lean_math_guard_v1.py').read_text();tree=ast.parse(helper);compile(helper,'<RAM helper syntax>','exec')
class Local:
 def __init__(self,threads):object.__setattr__(self,'threads',threads);object.__setattr__(self,'values',{})
 def __getattr__(self,n):
  try:return self.values.get(self.threads.tid,{})[n]
  except KeyError:raise AttributeError(n)
 def __setattr__(self,n,v):self.values.setdefault(self.threads.tid,{})[n]=v
class Threads:
 tid=11
 def local(self):return Local(self)
 def get_ident(self):return self.tid
class FakeSys:
 def __init__(self,accept=True):self.hooks=[];self.accept=accept;self.dont_write_bytecode=True;self.trace=self.profile=None
 def addaudithook(self,h):
  if self.accept:self.hooks.append(h)
 def audit(self,event,*args):
  for hook in self.hooks:hook(event,args)
 def gettrace(self):return self.trace
 def getprofile(self):return self.profile
class Entries:
 def __init__(self,records):self.records=records
 def __enter__(self):return [SimpleNamespace(name=str(fd))for fd in self.records]
 def __exit__(self,*args):return False
def fixture(accept=True):
 threads=Threads();system=FakeSys(accept);records={1:(2,1,1,1),2:(2,1,2,1),7:(1,5,70,2),8:(1,5,80,2),9:(2,5,90,0)}
 environment={'ARGGNN_GPU_LEASE_FD':'7','H058_LOGICAL_LOCK_FD':'8'}
 fake_os=SimpleNamespace(O_WRONLY=1,O_RDWR=2,O_APPEND=1024,O_CREAT=64,O_TRUNC=512,O_ACCMODE=3,O_RDONLY=0,environ=environment,
  fstat=lambda fd:SimpleNamespace(st_mode=records[fd][0],st_dev=records[fd][1],st_ino=records[fd][2]),scandir=lambda p:Entries(records))
 fake_fcntl=SimpleNamespace(F_GETFL=3,fcntl=lambda fd,flag:records[fd][3])
 t=SimpleNamespace(compiler=SimpleNamespace(is_compiling=lambda:False),jit=SimpleNamespace(is_scripting=lambda:False,is_tracing=lambda:False),autograd=SimpleNamespace(profiler=SimpleNamespace(_is_profiler_enabled=False)),cuda=SimpleNamespace(is_current_stream_capturing=lambda:False))
 scope=dict(contextmanager=contextmanager,os=fake_os,fcntl=fake_fcntl,stat=SimpleNamespace(S_ISREG=lambda mode:mode==1),sys=system,threading=threads)
 body=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef,ast.ClassDef))]
 exec(compile(ast.Module(body=copy.deepcopy(body),type_ignores=[]),'<copied helper AST with RAM APIs only>','exec'),scope)
 environment.update(scope['REQUIRED_ENV']);trace=[]
 def full():
  trace.append('full');system.audit('open','original_mutex_and_lease',None,2)
 def math_guard():
  trace.append('math_census_resources');system.audit('open','original_writable_lease_probe',None,2)
 gate=scope['MathEffectGateV1'](full,math_guard,t,('aligned','null'))
 return gate,scope,system,threads,records,t,trace
gate,scope,system,threads,records,t,trace=fixture()
check('RAM_constructor_hook_handshake',gate.installed)
def operation(guard):guard();guard();return 'RAM_result'
denied('lean_before_first_real_steps_rejects',lambda:gate.run_step('aligned',False,operation))
check('first_actual_aligned_two_full_checks',gate.run_step('aligned',True,operation)=='RAM_result'and trace==['full','full'])
denied('lean_before_both_arms_rejects',lambda:gate.run_step('aligned',False,operation))
check('first_actual_null_two_full_checks',gate.run_step('null',True,operation)=='RAM_result'and trace==['full']*4)
denied('duplicate_first_step_rejects',lambda:gate.run_step('aligned',True,operation))
check('lean_two_census_resource_boundaries_no_historical_replay',gate.run_step('aligned',False,operation)=='RAM_result'and trace==['full']*4+['math_census_resources']*2)
check('lean_success_clears_active_owner',gate.active is False and gate.owner is None and not getattr(gate.local,'guard',False))
def fail_operation(guard):guard();raise RuntimeError('RAM math exception')
denied('math_exception_propagates',lambda:gate.run_step('null',False,fail_operation))
check('math_exception_clears_active_owner',gate.active is False and gate.owner is None and not getattr(gate.local,'guard',False))
gate.active=True;gate.owner=threads.tid
for event in sorted(scope['MUTATION_EVENTS']):
 before=len(trace);denied('audit_mutation_'+event,lambda event=event:system.audit(event))
 check('full_guard_before_reject_'+event,trace[before:]==['full']and not getattr(gate.local,'guard',False))
for mode,flags in [('w',0),('a',0),('x',0),('r+',0),(None,1),(None,2),(None,64),(None,512),(None,1024)]:
 before=len(trace);denied('audited_write_open_'+str(mode)+'_'+str(flags),lambda:system.audit('open','RAM_only',mode,flags))
 check('write_open_escalation_first_'+str(mode)+'_'+str(flags),trace[before:]==['full'])
before=len(trace);system.audit('open','RAM_read_only','r',0);check('read_only_open_not_a_writer',len(trace)==before)
denied('file_backed_mmap_rejects',lambda:system.audit('mmap.__new__',7,4096,1,0))
before=len(trace);system.audit('mmap.__new__',-1,4096,1,0);check('anonymous_RAM_mmap_allowed',len(trace)==before)
before=len(trace);system.audit('import','late_module',None,None,None,None);check('late_import_full_check',trace[before:]==['full'])
with gate._trusted_guard():
 before=len(trace);system.audit('open','original_probe',None,2);check('exact_guard_reentrancy_bypass',len(trace)==before)
 threads.tid=22
 denied('other_thread_cannot_use_owner_guard_bypass',lambda:system.audit('open','RAM_background_writer',None,2))
 check('other_thread_guard_flag_restored',not getattr(gate.local,'guard',False))
 threads.tid=11
 check('owner_guard_flag_preserved_until_scope_exit',getattr(gate.local,'guard',False))
check('guard_bypass_restored_after_scope',not getattr(gate.local,'guard',False))
threads.tid=22;denied('other_thread_math_boundary_rejects',gate._math_boundary);threads.tid=11
records[10]=(1,5,100,1);denied('unreviewed_regular_writer_FD_escalates_rejects',gate._fds);records[10]=(1,5,100,0)
gate._fds();check('read_only_regular_FD_allowed')
records.pop(10);records[7]=(1,5,777,2);denied('lease_descriptor_identity_change_rejects',gate._fds);records[7]=(1,5,70,2)
system.environ=None
saved=scope['os'].environ['CUDA_CACHE_DISABLE'];scope['os'].environ['CUDA_CACHE_DISABLE']='0';denied('cache_control_mutation_rejects',gate._runtime);scope['os'].environ['CUDA_CACHE_DISABLE']=saved
t.autograd.profiler._is_profiler_enabled=True;denied('active_profiler_rejects',gate._runtime);t.autograd.profiler._is_profiler_enabled=False
system.trace=object();denied('active_python_trace_rejects',gate._runtime);system.trace=None
t.cuda.is_current_stream_capturing=lambda:True;denied('CUDA_graph_capture_rejects',gate._runtime);t.cuda.is_current_stream_capturing=lambda:False
gate.active=False;gate.owner=None
before=len(trace);system.audit('open','ordinary_full_boundary_outside_math','w',0);check('hook_inactive_outside_math',len(trace)==before)
denied('silently_rejected_hook_installation_fails',lambda:fixture(False))
# No source or operational edit; verify the three original live sources again after tests.
for row in preview['files']:check('final_live_hash_unchanged_'+Path(row['path']).name,sha((R/row['path']).read_bytes())==row['original_sha256'])
result=dict(schema='H058_lean_math_guard_independent_targeted_RAM_checks_v1',status='PASS_NARROW_INACTIVE_SOURCE_RAM_ONLY',checks=len(checks),negative_cases=len(negatives),cases=checks,negatives=negatives,source_manifest_ref=ref(D/'sealed_manifest.v1.json'),source_handoff_ref=ref(D/'handoff.v1.json'),source_helper_ref=ref(D/'lean_math_guard_v1.py'),historical_AGENTS_append_only_drift=drift,
 actual_candidate_module_framework_guard_resource_state_payload_numerical_or_production_calls=False,copied_helper_AST_only_under_RAM_APIs=True,universal_native_syscall_sandbox_not_required_or_claimed=True)
(O/'checks.result.v1.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in('cases','negatives')},sort_keys=True))

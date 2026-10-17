"""Independent actual new contract/request/memo2-auth/state/reserve/input reachability.
Exact source predicate bodies; all operational state, file writes, locks and clock
interfaces are handwritten RAM. Two inherited historical pins are not read.
"""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace,ModuleType,FunctionType
from contextlib import contextmanager
from datetime import datetime,timezone
import ast,collections,contextlib,copy,hashlib,io,json,os,stat,sys
REAL=Path('/workspace-SR003.nfs2/ARG-GNN');D='research/reviews/H060_global_only_full_training_source_20261017_v2';R='research/reviews/H060_global_only_full_training_source_20261017_v2'
sys.path.insert(0,str(REAL));checks=[];negative=[];memory={};dirs={str(REAL)};fds={};seq=[0];locked=[False];events=[];reads=collections.Counter()
SKIP={'experiments/results/census_tsinfer_observation_v1/census-tsinfer-inspect-v1/evidence.json','experiments/results/census_tsinfer_observation_v1/census-tsinfer-numerical-v1/evidence.json'}
manifest=json.loads((REAL/D/'source_manifest.v1.json').read_text());pins=manifest['source_sha256']
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ck(label,value=True):need(value,label);checks.append(label)
def reject(label,call):
 try:call()
 except (ValueError,AssertionError,KeyError,TypeError,AttributeError,OSError):negative.append(label);return
 raise AssertionError('Accepted '+label)
def relative(path):
 p=PurePosixPath(path);return str(p.relative_to(str(REAL)))if p.is_absolute()else str(p)
def bytesr(path):
 key=relative(path);reads[key]+=1
 if key in memory:return memory[key]
 need(key not in SKIP,'No historical numerical or accounting payload reads')
 p=REAL/key;need(p.suffix in ('.py','.sh','.json','.md','.txt','.patch','.diff','.log')and p.is_file()and p.stat().st_size<=2**20,'Only bounded source or metadata: '+key)
 return p.read_bytes()
def refr(path):return dict(path=relative(path),sha256=hashlib.sha256(bytesr(path)).hexdigest())
def rawr(record):
 # This is an explicit RAM source-hash fixture for two inherited historical pins,
 # never a claim that their real bytes were read or authenticated in this review.
 if record['path'] in SKIP:
  need(pins[record['path']]==record['sha256'],'Exact retained historical pin in RAM transport');return b'RAM omitted historical evidence'
 data=bytesr(record['path']);need(hashlib.sha256(data).hexdigest()==record['sha256'],'Exact RAM or bounded source/metadata SHA');return data
def readr(record):return json.loads(rawr(record))
def putram(path,value):
 path=relative(path);need(path not in memory,'Exclusive RAM metadata')
 memory[path]=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();return refr(path)
class MemoryPath(PurePosixPath):
 def resolve(self):return self
 @classmethod
 def cwd(cls):return cls(str(REAL))
 def exists(self):return relative(self)in memory or str(self)in dirs
 def is_file(self):return self.exists()and str(self)not in dirs
 def is_dir(self):return str(self)in dirs
 def is_symlink(self):return False
 def mkdir(self,parents=False,exist_ok=False,mode=0o700):
  need(exist_ok or not self.exists(),'RAM mkdir exclusive');dirs.add(str(self));events.append('mkdir')
 def lstat(self):
  if str(self)in dirs:data=b'';mode=stat.S_IFDIR|0o700
  else:data=bytesr(self);mode=stat.S_IFREG|0o600
  return SimpleNamespace(st_dev=1,st_ino=int(hashlib.sha256(str(self).encode()).hexdigest()[:12],16),st_mode=mode,st_nlink=1,st_size=len(data),st_mtime_ns=0,st_ctime_ns=0)
 def iterdir(self):return iter(sorted((MemoryPath(x)for x in set(dirs)|{str(MemoryPath(str(REAL))/k)for k in memory}if MemoryPath(x).parent==self),key=str))
 def read_bytes(self):return bytesr(self)
ramroot=MemoryPath(str(REAL));LOCK='.runtime/RAM_scheduler.lock';memory[LOCK]=b'RAM lock'
class RAMFile(io.BytesIO):
 def __init__(self,fd,mode):self.fd,self.mode=fd,mode;super().__init__(b''if'w'in mode else bytesr(fds[fd]))
 def fileno(self):return self.fd
 def close(self):
  if self.closed:return
  path=fds.get(self.fd)
  if 'w'in self.mode:memory[relative(path)]=self.getvalue()
  if path==ramroot/LOCK and locked[0]:
   need(all(getattr(v4,k)is val for k,val in original_primitives.items())and budget.successor_paths is original_successor,'Restore exact primitives before unlock')
   locked[0]=False;events.append('unlock')
  fds.pop(self.fd,None);super().close()
def ropen(path,flags,mode=0o600):
 path=MemoryPath(path);key=relative(path)
 if flags&os.O_CREAT:need(not(flags&os.O_EXCL)or key not in memory,'RAM O_EXCL');memory.setdefault(key,b'')
 seq[0]+=1;fds[seq[0]]=path;return seq[0]
def flock(fd,flags):need(not locked[0],'No nested mutex');locked[0]=True;events.append('lock')
ros=SimpleNamespace(**{n:getattr(os,n)for n in ('O_RDONLY','O_RDWR','O_WRONLY','O_NOFOLLOW','O_CREAT','O_EXCL','O_DIRECTORY')},open=ropen,fdopen=lambda fd,mode:RAMFile(fd,mode),fstat=lambda fd:fds[fd].lstat(),fsync=lambda fd:None,close=lambda fd:fds.pop(fd,None))
def source_module(path,names,scope):
 t=bytesr(path);tree=ast.parse(t);m=ModuleType('_independent_H060_'+Path(path).stem);m.__file__=str(REAL/path);m.__dict__.update(scope)
 nodes=[n for n in tree.body if (isinstance(n,ast.ImportFrom)and n.module=='__future__')or(isinstance(n,ast.FunctionDef)and n.name in names)]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REAL/path),'exec',dont_inherit=True,optimize=0),m.__dict__);return m
b=json.loads(bytesr(D+'/source_bindings.v1.json'));binding=json.loads(bytesr(D+'/scientific_binding.v1.json'));deps=json.loads(bytesr('research/reviews/H060_isolated_CPU_runtime_source_20261017_v1/dependencies.v1.json'))
fixture=json.loads(bytesr(D+'/role_union_fixture.v1.json'));guards=copy.deepcopy(fixture['actual_guard_role_refs_by_name']);ck('current47_named40_unique_guard_roles',len(guards)==47 and len({json.dumps(r,sort_keys=True)for r in guards.values()})==40)
# Actual original V4 primitives, including mutex and filesystem readers; os is RAM.
v4path='src/arggnn/project_budget_v4.py';v4=source_module(v4path,('canonical','file_bytes','bound','reference','scope_spec','read_json','mutex','parse','encoded','integer','paths_overlap','accounting_paths'),dict(Path=MemoryPath,os=ros,stat=stat,hashlib=hashlib,json=json,contextmanager=contextmanager,contextlib=contextlib,fcntl=SimpleNamespace(flock=flock,LOCK_EX=2),MAX_JSON=2**20,LEGACY='RAM_ONLY/legacy.json',SUPPLEMENTAL='RAM_ONLY/supplemental.json',LOCK=LOCK,SOURCE=v4path,CONFIG='configs/project_budget_v4.json',FIXED_POLICY={}))
budgetpath='scripts/project_budget_v9.py';budget=source_module(budgetpath,('successor_paths','digest'),dict(v4=v4,hashlib=hashlib,JOURNAL_REL='.runtime/RAM_successor',SOURCE_REL=budgetpath,LAUNCHER_REL='scripts/launch_reserved_gpu_v6.py'))
original_primitives={n:getattr(v4,n)for n in ('bound','reference','scope_spec','canonical','file_bytes','mutex')};original_successor=budget.successor_paths
memory['RAM_ONLY/legacy.json']=b'{"jobs":{}}';memory['RAM_ONLY/supplemental.json']=b'{}'
# Actual current Census state/reserve/append code objects with frozen RAM ancestor.
gpath=b['selected_child_source_ref']['path'];gtree=ast.parse(bytesr(gpath));constants={}
for n in gtree.body:
 if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
  try:constants[n.targets[0].id]=ast.literal_eval(n.value)
  except (ValueError,TypeError):pass
FIXED=datetime.fromisoformat('2026-10-17T12:00:00+00:00')
class FrozenDateTime(datetime):
 @classmethod
 def now(cls,tz=None):return FIXED if tz is not None else FIXED.replace(tzinfo=None)
g=source_module(gpath,('_state_locked','apply_reserve_v1','_append','_paths','_pins','_events','_authority','operation_draft_v1','_json','_raw','_ref','_expire','consumer_bindings_v1','charge','warning_fields','check_prefix','assert_open','require'),dict(constants,ROOT=ramroot,budget=budget,Path=MemoryPath,os=ros,copy=copy,json=json,hashlib=hashlib,datetime=FrozenDateTime,timezone=timezone))
g._POLICY_REF=readr(b['actual_selector_metadata_ref'])['policy_ref'];g._SELECTOR_REF=b['actual_selector_metadata_ref']
basepins=dict(RAM_frozen_prefix=True);baseview=dict(accounting_mode=g.BASE_MODE,projected_bytes=1,limit_bytes=g.CEILING_BYTES,hard_limit_bytes=g.HUMAN_LIMIT_BYTES,new_reserved_bytes=0,completed_credit_bytes=0)
def parentstate(*args,**kw):need(locked[0]and kw==dict(fresh=True),'Original fresh ancestor state under original mutex');return {},set(),dict(baseview),dict(guards),[],[],{}
g._BASE=SimpleNamespace(_state_locked=parentstate,_pins=lambda view:basepins,_paths=lambda own:budget.successor_paths(ramroot,own))
g._policy=lambda:dict(implementation_ref=refr(budget.SOURCE_REL),launcher_ref=refr(budget.LAUNCHER_REL),base_pins=basepins,base_projected_bytes=1,never_credit_keys=[])
g._bindings=lambda inherited,policy:copy.deepcopy(guards)
# Inert new wrapper module and original contract function bodies. Every scientific,
# clock, filesystem and guard interface is replaced only by RAM transport.
w=ModuleType('_independent_H060_wrapper');w.__file__=str(REAL/D/'root_driver_v1.py');exec(compile(bytesr(D+'/root_driver_v1.py'),w.__file__,'exec',dont_inherit=True,optimize=0),w.__dict__)
oldcontract=b['consumers']['H060']['contract_ref']['path'];ct=ast.parse(bytesr(oldcontract));c=source_module(oldcontract,tuple(n.name for n in ct.body if isinstance(n,ast.FunctionDef)),dict(ROOT=ramroot,Path=MemoryPath,datetime=FrozenDateTime,timezone=timezone,copy=copy,hashlib=hashlib,json=json,require=need,local=lambda p:ramroot/p,ref=refr,raw=rawr,read=readr,load=None,put=lambda p,v:putram(p,v),current=None,DEPS=deps,CAPS=b['consumers']['H060']['caps'],PHASE='train_matched',PARENT=b['consumers']['H060']['parent_id'],DIRECTIVE=json.loads(bytesr(D+'/ROOT_directive.INACTIVE.template.v1.json'))['directive'],TRAIN=tuple('american_c3_short_train_%03d'%i for i in range(20)),_ACTIVE=None))
for name in ('PIDFD','OWNED','SUPERVISOR','SELECTOR','GUARD_REF','POLICY_REF','NAMESPACE','V4','BASE_RUNTIME','TARGET_HOOK','ACCEPTANCE'):setattr(c,name,deps[name])
helpers='research/reviews/European_outer8_current_policy_successor_source_20261016_v1/prepare_outer000_v1.py';current=source_module(helpers,('current',),dict(datetime=FrozenDateTime,timezone=timezone,need=need));c.current=current.current
c._h=SimpleNamespace(guard=lambda selected,consumer:g)
selector=ModuleType('_RAM_selector');exec(compile(rawr(b['selector_bridge_ref']),str(REAL/b['selector_bridge_ref']['path']),'exec'),selector.__dict__)
# Accepted memo2 code executes its GENUINE callable authenticator, no substituted
# functions_from_source or code-auth bypass. Source bytes stay bounded real code.
memo=ModuleType('_independent_actual_memo2');memo.__file__=str(REAL/b['ROOT_memo_source_ref']['path']);exec(compile(rawr(b['ROOT_memo_source_ref']),memo.__file__,'exec',dont_inherit=True,optimize=0),memo.__dict__)
source_reader=FunctionType(memo.source_bytes.__code__,dict(memo.__dict__,ROOT=REAL),memo.source_bytes.__name__,memo.source_bytes.__defaults__,memo.source_bytes.__closure__)
memo.source_bytes=source_reader;memo.ROOT=ramroot;memo.Path=MemoryPath;memo.threading=SimpleNamespace(active_count=lambda:1);memo.sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=True,optimize=0))
bridge=ModuleType('_RAM_accepted_bridge');exec(compile(rawr(b['ROOT_memo_bridge_ref']),str(REAL/b['ROOT_memo_bridge_ref']['path']),'exec'),bridge.__dict__)
constructs=[]
def load(record,name):
 if record==b['consumers']['H060']['contract_ref']:return c
 if record==b['selector_bridge_ref']:return selector
 if record==b['ROOT_memo_source_ref']:constructs.append('memo_source');return memo
 if record==b['ROOT_memo_bridge_ref']:return bridge
 raise ValueError('Unexpected source import '+record['path'])
w.ROOT=ramroot;w.Path=MemoryPath;w.metadata_ref=refr;w.raw=rawr;w.read=readr;w.load=load;w.bindings=lambda:b
c=w.contract('H060');ck('actual_new_contract_loads_old_predicates_privately_with_Census_and_new_review',c.SOURCE==c.DRIVER==D+'/root_driver_v1.py'and c.ENTRY==D+'/worker_entry_v1.py')
# Full RAM request uses genuine old original20 context metadata, no plan/array load.
q=json.loads(bytesr(D+'/ROOT_request.INACTIVE.template.v1.json'));q.update(active=True,root_authorizer='root_agent',execution_authorized=True,created_at_utc=FIXED.isoformat(),expires_at_utc='2026-10-18T15:00:00+00:00',cpu_ids=list(range(32)),preparation_directory='research/reviews/H060_isolated_CPU_ROOT_preparation_global_only_v2_author_RAM',packet_directory='research/reviews/H060_isolated_CPU_ROOT_actual_global_only_v2_author_RAM',output_directory='experiments/results/H060_isolated_CPU_global_only_v2_author_RAM',source_manifest_ref=refr(D+'/source_manifest.v1.json'),scientific_config=dict(q['scientific_config'],execution_enabled=True))
dirs.add(str(ramroot/q['preparation_directory']))
review=json.loads(bytesr(D+'/ROOT_review.INACTIVE.template.v1.json'));review.update(independent=True,verdict='SOURCE_PASS',blockers=[],source_manifest_ref=refr(D+'/source_manifest.v1.json'),source_handoff_ref=refr(D+'/source_handoff.v1.json'))
q['source_review_ref']=putram('RAM_ONLY/operational_review.json',review)
science_review=json.loads(bytesr(D+'/ROOT_scientific_review.INACTIVE.template.v1.json'));science_review.update(status='SOURCE_PASS',independent_review=True,blockers=[],candidate_outer_manifest_ref=refr(D+'/source_manifest.v1.json'))
q['scientific_review_ref']=putram('RAM_ONLY/scientific_review.json',science_review)
registration=json.loads(bytesr(D+'/registration.INACTIVE.template.v1.json'));registration.update(root_authorizer='root_agent',status='registered_before_new_training_no_accuracy_yet',registered_utc='2026-10-17T11:59:00+00:00')
q['registration_ref']=putram('RAM_ONLY/registration.json',registration);q['registration_decisions_ref']=putram('RAM_ONLY/decisions.json',dict(decisions=[dict(id='H060')]))
memory['RAM_ONLY/ledger.md']=b'H060 RAM fixture';q['registration_ledger_ref']=refr('RAM_ONLY/ledger.md')
original=readr(binding['original_training_request_ref'])['bindings'];ctx=json.loads(bytesr(D+'/training_context.INACTIVE.template.v1.json'));ctx.update(root_authorizer='root_agent',parent_plan_refs=original['parent_plan_refs'],actual_train_bindings=original['actual_train_bindings'])
q['training_context_ref']=putram('RAM_ONLY/context.json',ctx)
runtime=json.loads(bytesr(D+'/runtime.INACTIVE.template.v1.json'));runtime.update(device='cpu',dtype='float32',CPU_threads=32,cpu_ids=q['cpu_ids'],prefix=str(ramroot/'.runtime/env'),package_report_ref=c.BASE_RUNTIME)
q['runtime_ref']=putram('RAM_ONLY/runtime.json',runtime)
qual=json.loads(bytesr(D+'/runtime_qualification.INACTIVE.template.v1.json'));qual.update(root_authorizer='root_agent',status='qualified',runtime_ref=q['runtime_ref'],source_manifest_ref=q['source_manifest_ref'],unresolved_items=[])
q['runtime_qualification_ref']=putram('RAM_ONLY/qualification.json',qual)
directive=json.loads(bytesr(D+'/ROOT_directive.INACTIVE.template.v1.json'));directive.update({k:q[k]for k in ('preparation_directory','source_manifest_ref','source_review_ref','scientific_review_ref','registration_ref','registration_decisions_ref','registration_ledger_ref','training_context_ref','runtime_ref','runtime_qualification_ref','scientific_config','caps','cpu_ids','created_at_utc','expires_at_utc')});directive.update(root_authorizer='root_agent',active=True)
q['root_directive_ref']=putram('RAM_ONLY/directive.json',directive)
request_ref=putram(q['preparation_directory']+'/ROOT_request.json',q)
ck('actual_COMPLETE_new_request_predicates_accept_RAM_metadata_with_original20_context',c.validate_request(request_ref)[0]==q)
# Actual new wrapper rejects mismatched new protocol/old scope before admission.
for label,key,value in [('new_global_protocol_required', 'new_global_observation_protocol_required',False),('no_old_live_replacement','old_scope_replay_or_live_replacement',True),('resume_flag_mandatory','transfer_resume_forbidden',False)]:
 bad=dict(q,**{key:value});rr=putram(q['preparation_directory']+'/'+label+'.json',bad);reject(label,lambda rr=rr:c.validate_request(rr))
# Reconstruct exact original driver via source AST and its reviewed three-site memo derivation.
driverpath=b['consumers']['H060']['driver_ref']['path'];dt=ast.parse(bytesr(driverpath));d=source_module(driverpath,tuple(n.name for n in dt.body if isinstance(n,ast.FunctionDef)),dict(os=SimpleNamespace(sched_getaffinity=lambda pid:set(range(32))),json=json))
for name,value in dict(c=c,ROOT=ramroot,HERE=D,SOURCE=D+'/root_driver_v1.py',PIDFD=c.PIDFD,OWNED=c.OWNED,need=need,local=c.local,ref=refr,read=readr,load=load,put=lambda path,value:putram(path,value),current=c.current,guard=c.guard).items():setattr(d,name,value)
w._install_memo_admit(d,'H060')
answer=d.admit(request_ref);admitted,q2,cfg2=c.input_spec(answer['input_ref'])
ck('actual_memo2_constructor_genuine_code_authentication_reached',constructs==['memo_source'])
ck('original_state_and_reserve_use_two_separate_non_nested_RAM_mutexes',events.count('lock')==events.count('unlock')==2 and not locked[0])
ck('actual_source_derived_append_creates_one_RAM_child_event',len([p for p in memory if p.startswith(g.JOURNAL+'/')])==1)
ck('full_current47_roles_and62_direct_union_survive_actual_request_admit_input',answer['consumer_binding_count']==62 and all(r in admitted['storage']['reservation']['consumer_bindings']for r in guards.values()))
ck('original32MiB_whole_scope_storage_input_accepted',admitted['storage']['reserved_new_bytes']==32*2**20 and admitted['root_driver_ref']==refr(D+'/root_driver_v1.py'))
ck('emitted_route_points_to_exact_new_driver',answer['execution_argv'][5]==D.replace('/','.')+'.root_driver_v1')
ck('all_memo_primitives_restored_and_generation_destroyed',all(getattr(v4,k)is val for k,val in original_primitives.items())and budget.successor_paths is original_successor and memo._ACTIVE is False)
# Genuine source authentication rejects a wrong module pathname BEFORE any next write.
oldfile=g.__file__;g.__file__=str(REAL/'research/reviews/WRONG_GUARD.py');before=len(memory)
reject('genuine_memo_constructor_rejects_wrong_guard_module_path',lambda:w._make_root_mutation_memo(g,q,request_ref));g.__file__=oldfile
ck('bad_constructor_causes_no_state_lock_or_write',len(memory)==before and events.count('lock')==2)
ck('no_scientific_framework_imports_or_actual_clock_called',not any(name in sys.modules for name in ('torch','numpy','torch_geometric','tskit')))
result=dict(schema='H060_global_only_author_v2_COMPLETE_request_memo2_auth_state_reserve_input_RAM',status='PASS',source_only=True,independent=False,assertions=len(checks),negative_fixtures=len(negative),checks=checks,negatives=negative,
 actual_complete_new_request_predicates_original20_context_and_new_reviews=True,genuine_memo2_callable_source_authentication_executed=True,actual_original_current_Census_state_reserve_append_bodies_executed=True,
 operational_IO_clock_state_parent_policy_lock_process_affinity_and_writers_all_RAM=True,current_named_guard_roles=47,full_direct_union=62,original_direct_cap=64,
 inherited_historical_pins_with_RAM_hash_transport_only=sorted(SKIP),no_actual_clock_resource_process_guard_state_accounting_reserve_launch_or_scientific_operations=True,
 manifest_ref=refr(D+'/source_manifest.v1.json'),handoff_ref=refr(D+'/source_handoff.v1.json'))
if '--emit-result'in sys.argv:
 with(REAL/R/'repair_request_memo_RAM.result.v2.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in ('checks','negatives')},sort_keys=True))

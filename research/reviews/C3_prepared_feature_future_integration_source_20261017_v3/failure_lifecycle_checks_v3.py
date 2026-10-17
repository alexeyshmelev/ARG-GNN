"""RAM-only v3 failure-resolution, lifecycle and caller-budget interfaces."""
from pathlib import Path as DiskPath
from datetime import datetime,timezone,timedelta
import ast,collections,copy,hashlib,io,json,os,stat,types
R=DiskPath('/workspace-SR003.nfs2/ARG-GNN')
D=R/'research/reviews/C3_prepared_feature_future_integration_source_20261017_v3'
OLD=R/'research/reviews/C3_prepared_feature_future_integration_source_20261017_v2'
positive=[];negative=[]
def require(ok,label):
 if not ok:raise ValueError(label)
def check(ok,label):
 if not ok:raise AssertionError(label)
 positive.append(label)
def rejected(fn,label,contains=None):
 try:fn()
 except (ValueError,KeyError,AssertionError,TypeError,FileExistsError)as error:
  if contains is not None:require(contains in str(error),'Exact expected rejection '+label)
  negative.append(label);return
 raise AssertionError('Accepted '+label)
def node(path,name):
 return next(n for n in ast.walk(ast.parse(path.read_text()))if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
def install(path,name,env):
 n=copy.deepcopy(node(path,name))
 exec(compile(ast.Module(body=[n],type_ignores=[]),'<RAM_exact_'+name+'>','exec'),env)
def ref(p):
 p=DiskPath(p);p=p if p.is_absolute()else R/p
 return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
pins=json.loads((D/'pins.v1.json').read_text())
stop=json.loads((R/pins['prior_v5_stop']['path']).read_text())
ram={}
def put(name,value):
 row=dict(path='research/reviews/C3_prepared_feature_v7_ROOT_preparation_RAMfixture/'+name+'.json',sha256=hashlib.sha256(name.encode()).hexdigest())
 ram[json.dumps(row,sort_keys=True)]=copy.deepcopy(value);return row
ram[json.dumps(pins['prior_v5_stop'],sort_keys=True)]=copy.deepcopy(stop)
stop_time=datetime.fromisoformat(stop['utc'])
proof_time=stop_time+timedelta(seconds=10)
request_time=proof_time+timedelta(seconds=5)
q=dict(source_manifest_ref=ref(D/'source_manifest.v1.json'),source_review_ref=put('review',{}),
 phase='predict_val_000',job_id='C3prepared-v7-RAMfixture',output_directory='experiments/results/C3_prepared_feature_GPU0_v7_RAMfixture',
 worker_auxiliary_scope='research/reviews/C3_prepared_feature_GPU0_ROOT_v7_RAMfixture/worker_aux',
 cell_election_ref=put('election',{}),created_at_utc=request_time.isoformat(),
 expires_at_utc=(request_time+timedelta(hours=1)).isoformat(),prior_v5_stop_ref=pins['prior_v5_stop'])
proof=dict(schema='ROOT_C3_v5_known_targets_poststop_observation_v1',root_authorizer='root_agent',
 actual_observation=True,stopped_failure_ref=pins['prior_v5_stop'],old_v5_input_ref=pins['preserved_v5_VAL000_input'],
 known_root_observations=[dict(pid=66065,expected_start=491497125,present=False),
  dict(pid=70270,expected_start=491547684,present=False),dict(pid=66030,expected_start=None,present=False)],
 known_positive_group_observations=[dict(pgid=66065,absent=True),dict(pgid=70270,absent=True)],
 only_listed_known_targets_observed=True,historical_descendant_drain_proven=False,
 storage_close_or_credit_authorized=False,utc=proof_time.isoformat())
q['prior_v5_known_targets_ref']=put('proof',proof)
resolution=dict(schema='C3_prepared_feature_v7_ROOT_prior_v5_resolution_v1',root_authorizer='root_agent',status='active',
 created_at_utc=proof_time.isoformat(),expires_at_utc=(request_time+timedelta(hours=2)).isoformat(),
 stopped_failure_ref=pins['prior_v5_stop'],known_targets_ref=q['prior_v5_known_targets_ref'],
 old_v5_input_ref=pins['preserved_v5_VAL000_input'],old_v5_source_manifest_ref=pins['preserved_v5_manifest'],
 old_v5_execution_ref=pins['preserved_v5_execution'],
 all_old_v5_future_execution_resume_retry_write_routes_revoked=True,old_v5_historical_descendant_drain='UNPROVEN',
 residual_historical_uncertainty_accepted_for_new_phase_only=True,does_not_certify_old_drain_or_cleanup_success=True,
 new_exact_phase_adoption_authorized=True,all_old_files_and_full_original_charge_preserved=True,
 storage_close_or_credit_authorized=False,old_authority_admission_or_output_reuse_authorized=False,
 state_or_output_migration_authorized=False,automatic_retry_authorized=False)
for k in ['source_manifest_ref','source_review_ref','phase','job_id','output_directory','worker_auxiliary_scope','cell_election_ref']:resolution[k]=copy.deepcopy(q[k])
q['prior_v5_resolution_ref']=put('resolution',resolution)
def active_ram(v):
 require(v.get('root_authorizer')=='root_agent'and v.get('status')=='active'and v.get('template_only')is not True,'RAM finite active ROOT identity')
def read_ram(row):return copy.deepcopy(ram[json.dumps(row,sort_keys=True)])
env=dict(require=require,pins_v1=lambda:copy.deepcopy(pins),read=read_ram,datetime=datetime,active=active_ram)
install(D/'contract_v1.py','prior_v5_failure_resolution_values_v3',env)
def join(historical=False):return env['prior_v5_failure_resolution_values_v3'](copy.deepcopy(q),historical=historical)
check(join()==(stop,proof,resolution),'Exact genuine immutable failed stop joins only explicit future RAM ROOT resolution and listed-target proof')
check(join(historical=True)==(stop,proof,resolution),'Completed fresh stage may retain its original finite resolution as history')
check(stop['original_all_owned_processes_and_groups_drained']is False and stop['storage_close_or_credit_authorized']is False,
 'Genuine outer1 stop never becomes old full drain, scientific success or released credit')
keys=dict(stop=json.dumps(pins['prior_v5_stop'],sort_keys=True),
 proof=json.dumps(q['prior_v5_known_targets_ref'],sort_keys=True),resolution=json.dumps(q['prior_v5_resolution_ref'],sort_keys=True))
for which,field,value in [
 ('stop','actual_outer_exit_code',0),('stop','actual_saved_wait_ref',{'path':'RAM/fake'}),
 ('stop','original_all_owned_processes_and_groups_drained',True),('stop','owned_output_files',['fake_prediction']),
 ('proof','actual_observation',False),
 ('proof','known_root_observations',[dict(pid=66065,expected_start=491497125,present=0),proof['known_root_observations'][1],proof['known_root_observations'][2]]),
 ('proof','known_positive_group_observations',[dict(pgid=66065,absent=1),dict(pgid=70270,absent=True)]),('proof','historical_descendant_drain_proven',True),
 ('proof','known_root_observations',proof['known_root_observations'][:-1]),
 ('proof','known_positive_group_observations',[dict(pgid=66065,absent=True)]),
 ('proof','utc',(proof_time-timedelta(seconds=1000)).isoformat()),
 ('proof','utc',(request_time+timedelta(seconds=1)).isoformat()),
 ('resolution','old_v5_historical_descendant_drain','DRAINED'),
 ('resolution','residual_historical_uncertainty_accepted_for_new_phase_only',False),
 ('resolution','all_old_v5_future_execution_resume_retry_write_routes_revoked',False),
 ('resolution','all_old_files_and_full_original_charge_preserved',False),
 ('resolution','storage_close_or_credit_authorized',True),('resolution','state_or_output_migration_authorized',True),
 ('resolution','old_authority_admission_or_output_reuse_authorized',True),
 ('resolution','automatic_retry_authorized',True),('resolution','source_manifest_ref',{'path':'RAM/wrong'}),
 ('resolution','job_id','other_job'),('resolution','output_directory','experiments/results/old'),
 ('resolution','new_exact_phase_adoption_authorized',False),('resolution','template_only',True),
 ('resolution','expires_at_utc',request_time.isoformat())]:
 key=keys[which];before=copy.deepcopy(ram[key]);ram[key][field]=value
 rejected(join,'Prior route resolution rejects '+which+'.'+field)
 ram[key]=before
before=q['prior_v5_stop_ref'];q['prior_v5_stop_ref']={'path':'RAM/fake','sha256':'0'*64}
rejected(join,'Different failed-stop identity rejected');q['prior_v5_stop_ref']=before
before=q['created_at_utc'];q['created_at_utc']=(proof_time+timedelta(seconds=301)).isoformat()
rejected(join,'Stale listed-target proof over300seconds rejected');q['created_at_utc']=before
for name in ['validate_request_v1','new_stage_values_v1']:
 n=node(D/'contract_v1.py',name)
 check(any(isinstance(z,ast.Call)and isinstance(z.func,ast.Name)and z.func.id=='prior_v5_failure_resolution_values_v3'for z in ast.walk(n)),
       name+' consumes exact tested prior-failure resolution')
# Actual lifecycle classes, RAM backend only.
limits=dict(pending=1024,discoveries=2**28,groups=65536,adopted_waits=2**28,history_rows=16)
class Backend:
 def __init__(self,initial):self.stats={r['pid']:copy.deepcopy(r)for r in initial};self.seed=copy.deepcopy(initial)
 @staticmethod
 def _same(a,b):return a is not None and b is not None and a['pid']==b['pid']and a['start']==b['start']
 def OwnedTree(self,leader,**kwargs):
  backend=self
  class Tree:
   def __init__(self):self.known={r['pid']:copy.deepcopy(r)for r in backend.seed};self.reused=set()
   def read_stat(self,pid):return copy.deepcopy(backend.stats.get(pid))
   def refresh(self):return self
  return Tree()
lenv=dict(collections=collections,hashlib=hashlib,json=json,OWNED_LIMITS_V8=limits)
for name in ['_HistoryV8','_PendingV8','OwnedLifecycleV8']:install(D/'execution_v1.py',name,lenv)
row=dict(pid=100,start=10,group=100,session=100,state='R')
b=Backend([row]);tree=lenv['OwnedLifecycleV8'](b,100)
check(tree.complete and tree.groups=={100}and 100 in tree.known,'Original positive-group identity retained')
b.stats[100]['group']=200;tree.read_stat_v9(100)
check(tree.groups=={100,200},'Every observed positive group remains retained before mutation')
killcalls=[]
class GroupOS:
 @staticmethod
 def killpg(group,sig):
  killcalls.append((group,sig));raise ProcessLookupError()
lenv.update(time=types.SimpleNamespace(monotonic=lambda:0.0),os=GroupOS)
install(D/'execution_v1.py','observe_groups_v8',lenv)
b.stats.clear();check(tree.identities_absent(),'Positive identities retire only after exact absence')
scan=lenv['observe_groups_v8'](tree,10.0)
check(scan['all_owned_groups_observed_absent']is True and set(killcalls)=={(100,0),(200,0)},
 'Positive-only exact group scan preserves ordinary full absence path')
for raw in [-1,0,True,'100',None,1.5]:
 badrow=dict(row,group=raw);b=Backend([badrow]);t=lenv['OwnedLifecycleV8'](b,100)
 check(not t.complete and 100 in t.known and not t.groups and t.invalid_groups.total>=1,
       'Invalid '+repr(raw)+' retained as known identity and permanently incomplete evidence')
 before=len(killcalls);s=lenv['observe_groups_v8'](t,10.0)
 check(len(killcalls)==before and s['all_owned_groups_observed_absent']is False,
       'Invalid '+repr(raw)+' never reaches killpg or claims full group absence')
 b.stats.clear();t.refresh()
 check(not t.complete and not t.known and not t.identities_absent(),'Invalid '+repr(raw)+' cannot regain complete status after identity retirement')
b=Backend([row]);t=lenv['OwnedLifecycleV8'](b,100);b.stats[100].update(group=-1,state='Z')
for _ in range(100):t.cleanup_read_stat_v9(100)
ev=t.evidence()
check(not t.complete and t.groups=={100}and 100 in t.known and ev['invalid_group_observation_history']['count']==100
      and len(ev['invalid_group_observation_history']['first'])==16 and len(ev['invalid_group_observation_history']['last'])==16,
 'Repeated samePID/start invalid terminal observations retain prior groups and bounded evidence without inventing origin')
t.groups.add(-1);before=len(killcalls);scan=lenv['observe_groups_v8'](t,10.0)
check(not scan['all_owned_groups_observed_absent']and all(g>0 for g,sig in killcalls[before:]),
 'Defensive corrupted group fixture also refuses nonpositive syscalls and absence claim')
# RAM filesystem for exact root metadata budget and exact waitable caller.
nodes={};fds={};contents={};nextfd=[100];opened=[];syncs=[];closed=[];race=[False]
def mk(path,kind='file',size=0,link=False):
 nodes[path]=dict(kind=kind,size=size,link=link,ino=len(nodes)+1000)
def stats(path):
 n=nodes[path]
 mode=stat.S_IFDIR|0o700 if n['kind']=='dir'else stat.S_IFREG|0o600
 if n['link']:mode=stat.S_IFLNK|0o777
 return types.SimpleNamespace(st_mode=mode,st_size=n['size'],st_nlink=2 if n['kind']=='dir'else 1,st_dev=1,st_ino=n['ino'])
class P:
 def __init__(self,path):self.key=path.key if isinstance(path,P)else str(path).strip('/')
 def __str__(self):return self.key
 def __truediv__(self,x):return P(self.key+'/'+str(x))
 def __eq__(self,x):return isinstance(x,P)and self.key==x.key
 @property
 def parent(self):return P(self.key.rsplit('/',1)[0])
 @property
 def name(self):return self.key.rsplit('/',1)[-1]
 @property
 def suffix(self):return '.'+self.name.rsplit('.',1)[-1]if'.'in self.name else''
 def exists(self):return self.key in nodes
 def is_file(self):return self.exists()and nodes[self.key]['kind']=='file'
 def is_symlink(self):return self.exists()and nodes[self.key]['link']
 def lstat(self):return stats(self.key)
 def stat(self):return stats(self.key)
 def iterdir(self):
  require(self.name!='worker_aux','No recursion into separately charged AUX subtree')
  return iter([P(k)for k in list(nodes)if k.rsplit('/',1)[0]==self.key])
 def relative_to(self,r):
  prefix=str(r).strip('/');require(self.key.startswith(prefix+'/'),'RAM root containment')
  return P(self.key[len(prefix)+1:])
 @staticmethod
 def cwd():return P('RAM')
class Stream(io.BytesIO):
 def __init__(self,fd):super().__init__();self.fd=fd;self.key=fds[fd]
 def write(self,data):
  size=super().write(data);contents[self.key]=self.getvalue();nodes[self.key]['size']=len(contents[self.key]);return size
 def fileno(self):return self.fd
 def close(self):
  if not self.closed:contents[self.key]=self.getvalue();nodes[self.key]['size']=len(contents[self.key])
  super().close()
class RamOS:
 O_RDONLY=os.O_RDONLY;O_DIRECTORY=os.O_DIRECTORY;O_NOFOLLOW=os.O_NOFOLLOW
 O_WRONLY=os.O_WRONLY;O_CREAT=os.O_CREAT;O_EXCL=os.O_EXCL
 WIFEXITED=staticmethod(os.WIFEXITED);WIFSIGNALED=staticmethod(os.WIFSIGNALED);waitstatus_to_exitcode=staticmethod(os.waitstatus_to_exitcode)
 @staticmethod
 def open(path,flags,mode=None):
  key=str(path);opened.append((key,flags,mode))
  if flags&os.O_DIRECTORY:
   require(key in nodes and nodes[key]['kind']=='dir'and not nodes[key]['link'],'RAM nofollow directory')
  else:
   require(flags==os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW and mode==0o600,'Exact caller exclusive mode')
   if key in nodes:raise FileExistsError(key)
   mk(key)
  nextfd[0]+=1;fds[nextfd[0]]=key;return nextfd[0]
 @staticmethod
 def fstat(fd):
  row=stats(fds[fd])
  if race[0]and nodes[fds[fd]]['kind']=='dir':row.st_ino+=1
  return row
 @staticmethod
 def close(fd):closed.append(fd)
 @staticmethod
 def fdopen(fd,mode):require(mode=='wb','Caller binary descriptor');return Stream(fd)
 @staticmethod
 def fsync(fd):syncs.append(fd)
 @staticmethod
 def waitpid(pid,flags):return(pid,256)
def reset():
 nodes.clear();contents.clear();fds.clear();opened.clear();closed.clear();syncs.clear();race[0]=False
 mk('RAM','dir');mk('RAM/packet','dir');mk('RAM/packet/input.json',size=100)
reset()
benv=dict(require=require,ROOT_RECORD_CAP=454656,stat=stat,os=RamOS)
install(D/'contract_v1.py','record_budget_v1',benv)
budget=benv['record_budget_v1'];directory=P('RAM/packet');aux=directory/'worker_aux'
mk(str(aux),'dir');mk(str(aux/'scientific_log.json'),size=1000000)
budget(directory,100,worker_auxiliary_scope=aux)
check(any(flags&os.O_DIRECTORY and flags&os.O_NOFOLLOW for path,flags,mode in opened),
 'Exact own AUX node accepted only through actual nofollow directory identity check')
oldenv=dict(require=require,ROOT_RECORD_CAP=454656)
install(OLD/'contract_v1.py','record_budget_v1',oldenv)
rejected(lambda:oldenv['record_budget_v1'](directory,100),'Immutable v2 reproduces packet worker_aux rejection')
rejected(lambda:budget(directory,100),'Missing explicit AUX allowance still rejects subdirectory')
mk('RAM/packet/unknown','dir')
rejected(lambda:budget(directory,100,worker_auxiliary_scope=aux),'Unknown directory remains rejected')
nodes.pop('RAM/packet/unknown')
rejected(lambda:budget(directory,100,worker_auxiliary_scope=P('RAM/else/worker_aux')),'Different-parent AUX rejected')
rejected(lambda:budget(directory,100,worker_auxiliary_scope=directory/'other'),'Different-name AUX rejected')
nodes[str(aux)]['link']=True
rejected(lambda:budget(directory,100,worker_auxiliary_scope=aux),'Symlink AUX rejected')
nodes[str(aux)]['link']=False;race[0]=True
rejected(lambda:budget(directory,100,worker_auxiliary_scope=aux),'Changed AUX inode rejected')
race[0]=False
rejected(lambda:budget(directory,-1,worker_auxiliary_scope=aux),'Negative write allowance rejected')
rejected(lambda:budget(directory,454656,worker_auxiliary_scope=aux),'Original454656 combined metadata cap retained')
reset()
for i in range(63):mk('RAM/packet/meta%02d.json'%i,size=1)
rejected(lambda:budget(directory,0),'Original count64 bound retained')
reset();mk(str(aux),'dir')
# Exact supervisor caller runs fake Popen/wait, never a real subprocess.
fixture_s=dict(worker_auxiliary_scope=str(aux))
fixture_q={}
class FakeProc:
 def __init__(self):self.pid=500;self.returncode=None
def fake_popen(command,**kwargs):
 require(kwargs.get('start_new_session')is True,'Original direct Python session behavior')
 # Existing aux may appear only after invocation in real operation, but persists at wait.
 if str(aux)not in nodes:mk(str(aux),'dir')
 return FakeProc()
c=types.SimpleNamespace(HERE=str(D.relative_to(R)))
cen=dict(require=require,local=P,record_budget_v1=budget,os=RamOS,stat=stat,json=json,hashlib=hashlib,ROOT=P('RAM'),
 datetime=datetime,timezone=timezone,c=c,sys=types.SimpleNamespace(dont_write_bytecode=True,
 flags=types.SimpleNamespace(no_user_site=True,optimize=0)),Path=P,
 validate_input_v1=lambda r:(fixture_s,fixture_q,{}, {},{}),
 subprocess=types.SimpleNamespace(Popen=fake_popen))
install(D/'execution_v1.py','_put_supervisor_record_v5',cen)
install(D/'execution_v1.py','supervise_once_v5',cen)
# Remove AUX to model exact invocation-before-executor transition.
nodes.pop(str(aux))
out=cen['supervise_once_v5']({'path':'RAM/input','sha256':'0'*64})
check(out['actual_Python_supervisor_wait']is True and out['actual_OS_wait_status']==256 and out['actual_Python_supervisor_returncode']==1,
 'Exact waitable caller retains genuine fake outer1 status instead of inferring success')
check('RAM/packet/caller_actual_supervisor_wait.v5.json'in contents and nodes[str(aux)]['kind']=='dir',
 'Exact caller persists actual supervisor wait after its validated AUX directory is created')
saved=json.loads(contents['RAM/packet/caller_actual_supervisor_wait.v5.json'])
check(saved['storage_close_or_credit_authorized']is False and saved['worker_wait_or_scientific_completion_inferred']is False,
 'Persisted caller wait remains separate from worker drain, scientific completion and credit')
check(len(syncs)>=4,'Exact caller metadata file and directory fsync paths execute against RAM descriptors')
result=dict(schema='C3_prepared_integration_v3_resolution_lifecycle_scope_RAM_checks_v1',
 status='AUTHOR_SOURCE_RAM_PASS_PENDING_FRESH_WHOLE_REVIEW',positive=len(positive),negative=len(negative),
 checks=positive,negative_cases=negative,exact_source_refs=[ref(D/'contract_v1.py'),ref(D/'execution_v1.py')],
 actual_genuine_stop_ref=pins['prior_v5_stop'],future_ROOT_resolution_and_proof_are_RAM_fixtures_only=True,
 no_active_ROOT_resolution_or_proof_created=True,unproven_historical_drain_remains_unproven=True,
 invalid_nonpositive_groups_latch_incomplete=True,practical_runtime_completion_not_guaranteed=True,
 original_scientific_arrays_opened=0,guard_imports=0,guard_calls=0,operational_module_imports=0,
 process_probes=0,actual_subprocesses=0,GPU_operations=0,Git_actions=0,admissions=0)
(D/'failure_lifecycle_checks.result.v3.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],positive=len(positive),negative=len(negative))))

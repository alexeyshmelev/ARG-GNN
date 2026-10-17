"""Full exact supervisor with RAM process/IO/lease doubles only."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,builtins,collections,contextlib,copy,hashlib,importlib,io,json,os,sys,time
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D='research/reviews/H058_GPU0_full_training_successor_source_20261017_v9';OUT='research/reviews/H058_GPU0_v9_whole_source_independent_20261017_v1';sys.path.insert(0,str(ROOT))
c=importlib.import_module(D.replace('/','.')+'.common_v1');op=importlib.import_module(D.replace('/','.')+'.operational_v1');d=op.driver_role_v1();checks=[]
def ck(n,v):
 if not v:raise AssertionError(n)
 checks.append(dict(name=n,status='PASS'))
def fn(t,n):return next(x for x in t.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name==n)
def emit(nodes,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),'independent_exact_supervisor_AST_RAM','exec'),scope)
bt=ast.parse((ROOT/c.OWNED['path']).read_text());bs={'os':SimpleNamespace(getpid=lambda:1,close=lambda fd:None),'_stat':None,'_children':None}
emit([fn(bt,n)for n in('_same','signal_identity','OwnedTree')],bs)
def row(pid,start,ppid=1100,group=1100):return dict(pid=pid,start=start,ppid=ppid,group=group,session=1100,state='R')
def case(mode):
 clock=[0.];closed=[];signals=[];records=[];table={1:row(1,1,ppid=0,group=1)};kids={1:[]};created=set();completed=set();waited=[False];inherited=[]
 def mono():clock[0]+=.001;return clock[0]
 class MP(PurePosixPath):
  @classmethod
  def cwd(cls):return MP(str(ROOT))
  def resolve(self):return self
  def exists(self):return str(self)in created or str(self)in completed
  def mkdir(self):created.add(str(self))
  def is_file(self):return str(self)in completed
 class Stream:
  def fileno(self):return 90
  def close(self):pass
 proc=SimpleNamespace(pid=1100,stdout=Stream(),returncode=None)
 s=dict(phase='train',job_id='H058-independent-RAM',output_directory='RAM_ONLY/out',worker_auxiliary_scope='RAM_ONLY/packet/worker_aux',refs={'config':{},'runtime':{}},expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=27)).isoformat())
 storage=dict(budget_components={'implementation':{'sha256':'0'*64},'launcher':{'sha256':'1'*64}})
 back=SimpleNamespace(OwnedTree=bs['OwnedTree'],_same=bs['_same'],_stat=lambda pid:table.get(pid),_children=lambda pid:kids.get(pid,[]))
 def same_tree(backend,leader):return d.OwnedLifecycleV8(backend,leader,owner=1,read_stat=back._stat,read_children=back._children)
 def waitpid(pid,flags):
  if pid==-1:raise ChildProcessError()
  assert pid==1100
  if mode=='overflow'and 1100 in table:return 0,0
  if not waited[0]:
   waited[0]=True;table.pop(1100,None);kids[1]=[];return 1100,(15 if mode=='overflow'else 256 if mode=='nonzero'else 0)
  return 0,0
 def send(fd,sig,*args):signals.append((fd,sig));table.pop(1100,None);kids[1]=[]
 def waitid(*args):
  if mode=='overflow':table[1100]=row(1100,10,ppid=1,group=1200);return None
  return SimpleNamespace(si_pid=1100)
 def mkproc(*args,**kwargs):
  inherited.extend(kwargs['pass_fds']);table[1100]=row(1100,10,ppid=1);kids[1]=[1100];completed.add(str(ROOT/s['output_directory']/'completed.json'));return proc
 def compact(spec,name,value,*args,**kwargs):records.append((name,copy.deepcopy(value)));return {'path':'RAM_ONLY/'+name,'sha256':'2'*64}
 def absent(group,sig):
  assert sig==0
  raise ProcessLookupError()
 fakeos=SimpleNamespace(getpid=lambda:1,waitid=waitid,waitpid=waitpid,P_PID=1,WEXITED=4,WNOHANG=1,WNOWAIT=0x1000000,waitstatus_to_exitcode=os.waitstatus_to_exitcode,close=closed.append,pipe=lambda:(80,81),set_blocking=lambda *a:None,write=lambda *a:None,read=lambda fd,n:b'',environ={},pathsep=os.pathsep,killpg=absent)
 back.signal_identity=lambda identity,sig,read_stat:bs['signal_identity'](identity,sig,read_stat=read_stat,open_fd=lambda p,f:91,send_fd=send,close_fd=lambda fd:None)
 direct=SimpleNamespace(attach_v1=lambda spec,q:(101,102,{'physical_lease_metadata':{'job':'C3'},'capacity_at_attachment':{}}),capacity_v1=lambda *a,**kw:{},base_v1=lambda:None,check_attachment_v1=lambda *a:None)
 importer=lambda name,*args,**kwargs:SimpleNamespace(direct_gpu0_v1=direct)if name==D.replace('/','.')else builtins.__import__(name,*args,**kwargs)
 sel=SimpleNamespace(register=lambda *a:None,select=lambda **kw:[],close=lambda:None,unregister=lambda *a:None)
 guard=SimpleNamespace(validate_storage_v9=lambda *a,**kw:None,budget_check_v9=lambda *a,**kw:None)
 ns=dict(vars(d),Path=MP,ROOT=MP(str(ROOT)),sys=SimpleNamespace(dont_write_bytecode=True,executable=str(ROOT/'.runtime/env/bin/python'),modules={'pidfd_backend_v1':SimpleNamespace(self_probe_v1=lambda:None)}),os=fakeos,time=SimpleNamespace(monotonic=mono,time=time.time,sleep=lambda delay:clock.__setitem__(0,clock[0]+delay)),signal=SimpleNamespace(SIGTERM=15,SIGINT=2,SIGKILL=9,signal=lambda *a:None),subprocess=SimpleNamespace(Popen=mkproc,PIPE=-1,STDOUT=-2),selectors=SimpleNamespace(DefaultSelector=lambda:sel,EVENT_READ=1),validate_input_v1=lambda r:(s,{}, {},storage,None),require_work_window_v2=lambda spec:None,guard_v1=lambda:guard,local=lambda path:MP(str(ROOT))/path,ref=lambda path:{'path':path,'sha256':'2'*64},charged_census_v1=lambda spec:{'RAM_ONLY/out':0},load=lambda rec,name:back if rec==c.OWNED else None,_subreaper=lambda:None,rss_identity_v1=lambda *a:0,worker_env_paths_v1=lambda spec:{'TMPDIR':str(ROOT/'RAM_ONLY/tmp')},compact_v1=compact,OwnedLifecycleV8=same_tree,__builtins__=dict(vars(builtins),__import__=importer))
 emit([fn(ast.parse(op._DRIVER_SOURCE),n)for n in('execute_v1','observe_groups_v8')],ns)
 limits=dict(d.OWNED_LIMITS_V8)
 if mode=='overflow':d.OWNED_LIMITS_V8['groups']=1
 try:
  with contextlib.redirect_stdout(io.StringIO()):answer=ns['execute_v1']({'path':'RAM_ONLY/input.json','sha256':'3'*64})
 finally:d.OWNED_LIMITS_V8.clear();d.OWNED_LIMITS_V8.update(limits)
 return answer,records,closed,signals,inherited
for mode in('success','nonzero','overflow'):
 answer,records,closed,signals,inherited=case(mode)
 ck('only own gate physical logical descriptors inherited: '+mode,inherited==[80,101,102])
 ck('own borrowed physical and logical descriptors closed: '+mode,closed[-2:]==[101,102])
 if mode=='success':ck('full actual supervisor success after direct wait EOF group absence and minimal evidence',answer['status']=='PASS'and answer['actual_direct_worker_returncode']==0 and [n for n,v in records][-3:]==['actual_wait_minimal.json','worker_log.json','actual_wait.json'])
 elif mode=='nonzero':ck('full actual supervisor fails on nonzero direct wait after honest drained evidence',answer['status']=='FAILED'and answer['actual_direct_worker_returncode']==1 and answer['no_future_stage_authorized'])
 else:ck('full actual supervisor overflow signals known identity but blocks complete drain and postdrain writes',bool(signals)and answer['status']=='FAILED_UNRESOLVED_CLEANUP'and answer['all_owned_processes_and_groups_drained']is False and not any(n.startswith('actual_wait')for n,v in records))
ck('no scientific frameworks imported',not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy'))for n in sys.modules))
result=dict(schema='H058_v9_independent_full_supervisor_RAM_checks_v1',independent=True,source_only=True,status='PASS',assertions=len(checks),checks=checks,actual_process_wait_signal_pidfd_descriptor_guard_lease_GPU_numerical_operations=False)
with(ROOT/OUT/'supervisor.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='checks'},sort_keys=True))


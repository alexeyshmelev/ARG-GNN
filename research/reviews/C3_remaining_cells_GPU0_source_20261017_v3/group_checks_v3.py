"""Source-only roster churn and full actual execute/cleanup joins; all OS/process APIs are RAM doubles."""
from pathlib import Path as RealPath
from datetime import datetime,timezone,timedelta
import ast,collections,contextlib,copy,hashlib,io,json,types,builtins
ROOT=RealPath("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/C3_remaining_cells_GPU0_source_20261017_v3"
OLD=ROOT/"research/reviews/C3_remaining_cells_GPU0_source_20261017_v1"
names=("_HistoryV8","_PendingV8","OwnedLifecycleV8","observe_groups_v8")
checks=[];negative=[]
def need(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(fn,label):
 try:fn()
 except(ValueError,PermissionError):negative.append(label);return
 raise AssertionError("Invalid accepted: "+label)
class Clock:
 def __init__(self):self.t=0.;self.base=1792238400.
 def monotonic(self):self.t+=.0001;return self.t
 def time(self):return self.base+self.t
 def sleep(self,n):self.t+=max(.0001,n)
def identity(pid,start=1,parent=100,group=None,state="S"):
 return dict(pid=pid,start=start,ppid=parent,group=pid if group is None else group,session=pid if group is None else group,state=state)
class World:
 def __init__(self):
  self.rows={1000:identity(1000)};self.child={100:[1000],1000:[]};self.signals=[];self.group_calls=[];self.group_mode="absent";self.read_error=False;self.clock=Clock()
 def read_stat(self,pid):
  if self.read_error and pid==1000:raise PermissionError("RAM inaccessible exact owned identity")
  return copy.deepcopy(self.rows.get(pid))
 def children(self,pid):return list(self.child.get(pid,[]))
 def killpg(self,pid,sig):
  self.group_calls.append((pid,self.clock.t))
  if self.group_mode=="absent":raise ProcessLookupError()
  if self.group_mode=="permission":raise PermissionError("RAM group access denied")
 def close(self,fd):pass
def backend(world):
 src=ast.parse((ROOT/"research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py").read_text())
 ns=dict(_stat=world.read_stat,_children=world.children,os=types.SimpleNamespace(getpid=lambda:100,close=world.close),
  pidfd_open_v1=lambda pid,flags:pid,pidfd_send_signal_v1=lambda fd,sig,info,flags:world.signals.append((fd,sig)))
 nodes=[n for n in src.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in("_same","signal_identity","OwnedTree")]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),"<actual-original-owned-backend-with-RAM-providers>","exec"),ns)
 ns["_stat"]=world.read_stat;ns["_children"]=world.children
 return types.SimpleNamespace(**ns)
limits=dict(pending=1024,discoveries=2**28,groups=65536,adopted_waits=2**28,history_rows=16)
def helper(world):
 src=ast.parse((D/"execution_v1.py").read_text())
 ns=dict(collections=collections,hashlib=hashlib,json=json,time=world.clock,os=types.SimpleNamespace(killpg=world.killpg),OWNED_LIMITS_V8=copy.deepcopy(limits))
 exec(compile(ast.Module(body=[n for n in src.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names],type_ignores=[]),"<actual-C3-bounded-live-helpers>","exec"),ns)
 return ns
def tracked_child():
 w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000)
 w.rows[2000]=dict(identity(2000,start=2,parent=1000,group=1000),session=1000)
 w.child[1000]=[2000];t.refresh()
 return w,b,h,t
# Material case: an owned non-session-leader child performs valid same-session setpgid.
w,b,h,t=tracked_child();w.rows[2000]["group"]=2000;t.refresh()
need(t.known[2000]["group"]==2000 and t.groups=={1000,2000},"Same child PID/start update retains old and new PGID in original session")
need(t.discovered.total==2,"Group-only transition never invents another discovered identity")
# The group cap is enforced before row/group/history mutation.
h["OWNED_LIMITS_V8"]["groups"]=2;before=copy.deepcopy(t.known[2000]);before_groups=set(t.groups);discovered=t.discovered.total
w.rows[2000]["group"]=3000
reject(t.refresh,"Same-identity new-group overflow fails before mutation")
need(t.known[2000]==before and t.groups==before_groups and t.discovered.total==discovered and not t.complete,
 "Overflow preserves old row/group/history and permanently blocks drain")
need(b.signal_identity(before,15,read_stat=t.cleanup_read_stat_v9)and w.signals==[(2000,15)]
 and t.known[2000]==before and t.groups==before_groups and not t.complete,
 "Group-overflow cleanup preserves prior bounded state and still signals exact known child pidfd")
w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000);w.read_error=True
reject(lambda:t.cleanup_read_stat_v9(1000),"Cleanup read path preserves original stat permission failure")
# A group observed by _retire cannot be lost when that child then disappears.
w,b,h,t=tracked_child();calls=[0]
def transient_group(pid):
 if pid!=2000:return w.read_stat(pid)
 calls[0]+=1
 if calls[0]==1:return dict(identity(2000,start=2,parent=1000,group=2000),session=1000)
 return None
t._original_read_stat=transient_group;w.child[1000]=[];w.rows.pop(2000);t.refresh()
need(t.groups=={1000,2000}and 2000 not in t.known,"Retirement stat observation retains new child group even before disappearance")
# A directly authenticated live() observation also retains the changed group.
w,b,h,t=tracked_child();t.refresh=lambda:t
t._original_read_stat=lambda pid:dict(w.rows[pid],group=2000)if pid==2000 else copy.deepcopy(w.rows[pid])
need(set(t.live())=={1000,2000}and t.groups=={1000,2000},"Live exact-child observation retains changed group")
# External pidfd signaling stat observations use the same authenticated group-aware path.
w,b,h,t=tracked_child();before=copy.deepcopy(t.known[2000]);w.rows[2000]["group"]=2000
need(b.signal_identity(before,15,read_stat=t.read_stat_v9)and t.groups=={1000,2000}and w.signals==[(2000,15)],
 "Original pidfd signal stat observes and retains changed exact-owned child group")
w,b,h,t=tracked_child();before=copy.deepcopy(t.known[2000]);w.rows[2000]=identity(2000,start=3,parent=999,group=3000)
need(b.signal_identity(before,15,read_stat=t.cleanup_read_stat_v9)is False and not w.signals and t.groups=={1000},
 "Reused unrelated PID never becomes an owned changed group or pidfd signal target")
# Original RSS receives the group-aware child backend without a real proc read.
rss_node=next(n for n in ast.parse((D/"execution_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="rss_identity_v1")
w,b,h,t=tracked_child();before=copy.deepcopy(t.known[2000]);w.rows[2000]["group"]=2000
rss_ns=dict(require=lambda ok,msg:None if ok else(_ for _ in()).throw(ValueError(msg)),
 Path=lambda path:types.SimpleNamespace(read_text=lambda:"32 1"),os=types.SimpleNamespace(sysconf=lambda key:4096))
exec(compile(ast.Module(body=[rss_node],type_ignores=[]),"<actual-owned-RSS-with-RAM-statm>","exec"),rss_ns)
need(rss_ns["rss_identity_v1"](types.SimpleNamespace(_stat=t.read_stat_v9,_same=b._same),before)==4096 and t.groups=={1000,2000},
 "Original child RSS stat observes and retains changed exact-owned group")
# Both groups are queried; a positive new group cannot hide behind the old group's ESRCH.
w,b,h,t=tracked_child();w.rows[2000]["group"]=2000;t.refresh();w.rows.clear();w.child[100]=[]
need(t.identities_absent(),"Exact absence retires moved child without erasing retained groups")
def changed_probe(group,sig):
 w.group_calls.append((group,w.clock.t))
 if group==2000:return None
 raise ProcessLookupError()
h["os"].killpg=changed_probe
summary=h["observe_groups_v8"](t,w.clock.monotonic()+30)
need({group for group,_ in w.group_calls}=={1000,2000}and not summary["all_owned_groups_observed_absent"],
 "Positive changed group cannot hide behind original-group ESRCH")
class RP:
 def __init__(self,p):self.p=str(p)
 def __str__(self):return self.p
 def __fspath__(self):return self.p
 def __truediv__(self,x):return RP(self.p.rstrip("/")+"/"+str(x).lstrip("/"))
 def __eq__(self,x):return self.p==str(x)
 def __hash__(self):return hash(self.p)
 def resolve(self):return self
 def exists(self):return False
 def is_file(self):return self.p.endswith("/completed.json")
 def mkdir(self,*a,**kw):pass
 def relative_to(self,x):return RP(self.p[len(str(x))+1:])
 @classmethod
 def cwd(cls):return RP(ROOT)
def execute_case(mode):
 w=World();w.rows={};w.child={100:[]};b=backend(w);captured={};pid=1000;wait_calls=[];signal_attempts=[];work_polls=[0];adopted_done=[False]
 def r(path):return dict(path=str(path),sha256="a"*64)
 s=dict(phase="matrix",job_id="RAM-C3",expires_at_utc=datetime.fromtimestamp(w.clock.base+3600,timezone.utc).isoformat(),
  output_directory="RAM/output",worker_auxiliary_scope="RAM/packet/worker_aux",refs=dict(storage=r("RAM/storage"),config=r("RAM/config"),runtime=r("RAM/runtime")))
 q=dict(capacity_policy_ref=r("RAM/policy"));storage=dict(budget_components=dict(implementation=r("RAM/impl"),launcher=r("RAM/launcher")))
 caps=dict(reservation_bytes=33554432,log_bytes=262144,max_seconds=86400,max_cpu_rss_bytes=128*2**30,
  work_cutoff_margin_seconds=300,terminal_evidence_margin_seconds=150,cleanup_seconds=30,max_checkpoint_bytes=3*2**30)
 class Guard:
  def validate_storage_v9(self,*a,**k):return {}
  def budget_check_v9(self,*a,**k):return {}
 class Stream:
  def fileno(self):return 9
  def close(self):pass
 proc=types.SimpleNamespace(pid=pid,stdout=Stream(),returncode=None)
 def popen(*a,**kw):
  need(kw["start_new_session"]is True and 10 in kw["pass_fds"],"Actual parent-owned new session launch gate "+mode)
  w.rows[pid]=identity(pid,state="S"if mode.startswith("changed-groups")else"Z");w.child[100]=[pid];w.child[pid]=[]
  if mode.startswith("changed-groups"):
   w.rows[2000]=dict(identity(2000,start=2,parent=pid,group=pid),session=pid);w.child[pid]=[2000];w.child[2000]=[]
  return proc
 def waitid(*a):
  work_polls[0]+=1
  if mode.startswith("changed-groups"):
   if work_polls[0]==1:return None
   w.rows[2000].update(group=2000,state="Z");w.rows[pid]["state"]="Z"
  return types.SimpleNamespace(si_pid=pid)
 def waitpid(target,flags):
  wait_calls.append(target)
  if target==-1:
   if mode.startswith("changed-groups")and not adopted_done[0]:
    adopted_done[0]=True;w.rows.pop(2000,None);w.child[100]=[]
    return(2000,0)
   raise ChildProcessError()
  w.rows.pop(pid,None)
  if mode.startswith("changed-groups"):
   w.rows[2000]["ppid"]=100;w.child[100]=[2000]
  else:w.child[100]=[]
  return(pid,0)
 def signal_identity(row,sig,**kwargs):
  signal_attempts.append((row["pid"],row["start"],sig))
  return True
 b.signal_identity=signal_identity
 fakeos=types.SimpleNamespace(getpid=lambda:100,environ={},pathsep=":",pipe=lambda:(10,11),write=lambda fd,data:len(data),close=lambda fd:None,set_blocking=lambda *a:None,
  waitid=waitid,waitpid=waitpid,waitstatus_to_exitcode=lambda status:0 if status==0 else 1,read=lambda fd,n:b"",killpg=w.killpg,
  P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8)
 w.group_mode="absent"
 if mode=="changed-groups-live":
  def changed_group_probe(group,sig):
   w.group_calls.append((group,w.clock.t))
   if group==2000:return None
   raise ProcessLookupError()
  w.killpg=changed_group_probe
  fakeos.killpg=changed_group_probe
 class Selector:
  def register(self,*a):pass
  def unregister(self,*a):pass
  def select(self,*a,**kw):return []
  def close(self):pass
 def compact(s,name,value,*a,**kw):captured[name]=copy.deepcopy(value);return r("RAM/"+name)
 modules={"pidfd_backend_v1":types.SimpleNamespace(self_probe_v1=lambda:None)}
 ns=dict(__name__="RAM_full_C3_execute",Path=RP,datetime=datetime,timezone=timezone,collections=collections,hashlib=hashlib,json=json,
  time=w.clock,os=fakeos,types=types,sys=types.SimpleNamespace(dont_write_bytecode=True,executable=str(ROOT/".runtime/env/bin/python"),modules=modules),
  subprocess=types.SimpleNamespace(Popen=popen,PIPE=-1,STDOUT=-2),selectors=types.SimpleNamespace(DefaultSelector=Selector,EVENT_READ=1),
  signal=types.SimpleNamespace(SIGTERM=15,SIGINT=2,SIGKILL=9,signal=lambda *a:None),OWNED_LIMITS_V8=copy.deepcopy(limits),
  ROOT=RP(ROOT),CAPS=caps,GPU_PHASES=("predict_val_000","predict_val_001","predict_val_002"),
  MINIMAL_WAIT_CAP_BYTES=65536,TERMINAL_RECEIPT_CAP_BYTES=8*2**20,
  c=types.SimpleNamespace(STATE=(s,)),DRIVER="RAM/execution.py",MANIFEST="RAM/manifest",PIDFD=r("RAM/pidfd"),OWNED=r("RAM/owned"),
  NATIVE_PYTHON_REF=dict(path=".runtime/env/bin/python3.11"),local=lambda p:RP(str(ROOT/p)if not str(p).startswith("/")else p),
  require=lambda ok,msg:None if ok else(_ for _ in()).throw(ValueError(msg)),ref=r,read=lambda r:dict(status="RAM scientific completion"),
  validate_input_v1=lambda r:(s,q,{},storage,{}),require_work_window_v2=lambda s:None,work_deadline_monotonic_v1=lambda:3000,
  guard_v1=lambda q:Guard(),prepare_master_parents_v1=lambda *a:None,worker_env_paths_v1=lambda s:{},
  charged_census_v1=lambda *a,**kw:dict(output=4096,aux=4096),compact_v1=compact,
  active_amendment_v1=lambda:None,current_amendment_ref_v1=lambda:r("RAM/input"),save=lambda *a:None,
  validate_child_completion_v1=lambda *a:None)
 def loader(r,name):
  if name=="pidfd_backend_v1":return modules["pidfd_backend_v1"]
  if "owned_lifecycle"in name:return b
  return types.SimpleNamespace()
 ns["load"]=loader
 realimport=builtins.__import__
 def imported(name,*a,**kw):
  if name.startswith("research."):return types.SimpleNamespace(check_external_prerequisites_v1=lambda q:None)
  return realimport(name,*a,**kw)
 ns["__builtins__"]=dict(vars(builtins),__import__=imported)
 nodes=[n for n in ast.parse((D/"execution_v1.py").read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names+("_RAMLog","execute_v1")]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),"<complete-actual-C3-execute-with-RAM-lower-layers>","exec"),ns)
 ns["_subreaper"]=lambda:None;ns["rss_identity_v1"]=lambda *a:4096
 with contextlib.redirect_stdout(io.StringIO())as stdout:result=ns["execute_v1"](r("RAM/input"))
 printed=[json.loads(x)for x in stdout.getvalue().splitlines()]
 facts=next(x["downstream_actual_minimal_wait_facts"]for x in printed if"downstream_actual_minimal_wait_facts"in x)
 need(wait_calls and all(x in(pid,-1)for x in wait_calls),"Actual full supervisor uses genuine exact direct/adopted wait API "+mode)
 need(facts["actual_direct_worker_wait"]is(mode!="unknown-wait"),"No fabricated direct OS wait "+mode)
 need(result["all_owned_processes_and_groups_drained"]is(mode in("normal","changed-groups-absent")),"Complete actual execute drain gate "+mode)
 if mode in("normal","changed-groups-absent"):
  need(result["returncode"]==0 and captured["actual_wait.json"]["group_absence_summary"]["all_owned_groups_observed_absent"],"Actual full execute reaches bounded own wait0 evidence")
  need(captured["actual_wait.json"]["ownership"]["retired_identity_history"]["count"]==(2 if mode.startswith("changed-groups")else 1),"Full execute retirement joins actual direct/adopted waits")
  contract_node=next(n for n in ast.parse((D/"contract_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="owned_drain_values_v2")
  contract_ns=dict(require=ns["require"],ref=r,DRIVER="RAM/execution.py",OWNED_LIMITS_V8=limits)
  exec(compile(ast.Module(body=[contract_node],type_ignores=[]),"<actual-C3-drain-contract-with-actual-RAM-supervisor-wait>","exec"),contract_ns)
  need(contract_ns["owned_drain_values_v2"](captured["actual_wait.json"])==captured["actual_wait.json"],"Actual full supervisor wait passes actual new-cell drain contract")
 else:
  need(result["returncode"]!=0 and"actual_wait.json"not in captured,"Unresolved full cleanup never publishes success wait "+mode)
 if mode.startswith("changed-groups"):need(adopted_done[0]and {group for group,_ in w.group_calls}=={1000,2000},"Full supervisor retains and probes original plus changed same-identity groups "+mode)
 if mode=="refresh-failure":need(bool(signal_attempts),"Cleanup signals known exact rows even after refresh failure")
 need(all(stamp<=30.1 for _,stamp in w.group_calls),"No actual full-execute group observation after finite cleanup "+mode)
 return dict(mode=mode,returncode=result["returncode"],drained=result["all_owned_processes_and_groups_drained"],direct_wait=facts["actual_direct_worker_wait"],group_calls=len(w.group_calls),signal_attempts=len(signal_attempts))
cases=[execute_case(mode)for mode in("normal","changed-groups-absent","changed-groups-live")]
shared_source=ROOT/"research/reviews/H058_GPU0_full_training_successor_source_20261017_v9/lifecycle_helper.v9.source.txt"
shared=ast.parse(shared_source.read_text());copied=ast.parse((D/"execution_v1.py").read_text())
for name in names:
 expected=next(n for n in shared.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 actual=next(n for n in copied.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 need(ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False),"Exact revised shared helper AST "+name)
result=dict(schema="C3_remaining_same_identity_group_source_checks_v3",verdict="AUTHOR_SOURCE_RAM_CHECKS_ONLY",
 assertions=len(checks),negatives=len(negative),checks=checks,negative_cases=negative,actual_full_execute_RAM_cases=cases,
 same_identity_group_updates_and_retirement_live_observations_checked=True,group_cap_before_mutation_checked=True,
 actual_supervisor_wait_to_new_cell_contract_join=True,new_churn_or_numerical_campaign=False,
 actual_guard_state_resource_process_GPU_scope_or_numerical_calls=False,scientific_payload_reads=False)
(D/"group_checks.result.v3.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))

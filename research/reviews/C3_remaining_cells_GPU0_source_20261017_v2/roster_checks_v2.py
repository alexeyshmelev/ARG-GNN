"""Source-only roster churn and full actual execute/cleanup joins; all OS/process APIs are RAM doubles."""
from pathlib import Path as RealPath
from datetime import datetime,timezone,timedelta
import ast,collections,contextlib,copy,hashlib,io,json,types,builtins
ROOT=RealPath("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/C3_remaining_cells_GPU0_source_20261017_v2"
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
w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000)
for i in range(50000):
 pid=2000+i%17;w.rows[pid]=identity(pid,start=100+i,parent=1000,group=1000);w.child[1000]=[pid]
 t.refresh()
 if i%1000==0:need(len(t.known)==2 and t.complete,"Bounded pending identities through churn "+str(i))
 del w.rows[pid];w.child[1000]=[];t.refresh()
w.rows.pop(1000);w.child[100]=[]
need(t.identities_absent()and t.complete,"50000 helpers exceed failed4096 cumulative roster without growing pending")
own=t.evidence()
need(own["discovered_identity_history"]["count"]==own["retired_identity_history"]["count"]==50001,"Every discovered churn identity has exact absent retirement")
need(len(json.dumps(own).encode())<16384,"Churn ownership evidence remains below16KiB")
summary=h["observe_groups_v8"](t,w.clock.monotonic()+30)
need(summary["all_owned_groups_observed_absent"]and summary["observed_group_count"]==1,"All churn groups retained and actually checked")
# Exact PID reuse outside verified ancestry never becomes an owned signal target.
w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000)
w.rows[2000]=identity(2000,start=2,parent=1000,group=1000);w.child[1000]=[2000];t.refresh();old=copy.deepcopy(t.known[2000])
w.rows[2000]=identity(2000,start=3,parent=999,group=333);w.child[1000]=[];t.refresh()
need(2000 not in t.known and t.retirement_reasons["replaced"]==1,"Only exact replaced identity retired; unrelated reused PID excluded")
need(b.signal_identity(old,15,read_stat=w.read_stat)is False and not w.signals,"Actual original pidfd signal API excludes changed PID/start")
# Same identity zombie is retained until a real wait/absence.
w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000)
w.rows[1000]["state"]="Z";need(not t.identities_absent()and 1000 in t.known and not t.live(),"Zombie does not manufacture retired/drained identity")
w.rows.clear();w.child[100]=[];need(t.identities_absent(),"Genuine post-wait absence permits zombie retirement")
# Bounds are separate and enforced before insert; no unsafe truncation.
w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000)
w.child[1000]=list(range(2000,3024))
for pid in w.child[1000]:w.rows[pid]=identity(pid,parent=1000,group=1000)
reject(t.refresh,"Current pending1024 overflow fails before insertion")
need(len(t.known)==1024 and not t.complete,"Overflow preserves bounded known rows and permanently incomplete ownership")
for field,value in(("discoveries",2),("groups",1)):
 w=World();b=backend(w);h=helper(w);h["OWNED_LIMITS_V8"][field]=value;t=h["OwnedLifecycleV8"](b,1000)
 w.child[1000]=[2000,2001]
 for pid in w.child[1000]:w.rows[pid]=identity(pid,parent=1000,group=pid if field=="groups"else 1000)
 reject(t.refresh,"Separate "+field+" bound fails closed")
 need(not t.complete,"Separate "+field+" overflow permanently incomplete")
# Permission/unknown probes preserve pending and make complete drain impossible.
w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000);w.read_error=True
reject(t.refresh,"Exact identity read permission failure")
need(1000 in t.known and not t.complete,"Refresh failure retains known signalable identity")
# Full group scan, positive reuse, permission failure and deadline exhaustion.
for mode in("absent","exists","permission"):
 w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000);t.groups=set(range(1000,1512));w.group_mode=mode
 scan=h["observe_groups_v8"](t,w.clock.monotonic()+30)
 need(scan["observed_group_count"]==512 and len(scan["history"]["first"])==len(scan["history"]["last"])==16,"Every512 retained groups observed with bounded samples "+mode)
 need(scan["all_owned_groups_observed_absent"]is(mode=="absent"),"Group absence requires every actual numeric ESRCH "+mode)
 need(len(json.dumps(scan).encode())<16384,"Group summary remains bounded "+mode)
w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000)
scan=h["observe_groups_v8"](t,w.clock.t-1)
need(not scan["all_owned_groups_observed_absent"]and not w.group_calls,"No group call after original cleanup deadline")
w=World();b=backend(w);h=helper(w);t=h["OwnedLifecycleV8"](b,1000)
def last_probe_overrun(group,sig):
 w.clock.t+=40
 raise ProcessLookupError()
h["os"].killpg=last_probe_overrun
scan=h["observe_groups_v8"](t,w.clock.monotonic()+30)
need(scan["observed_group_count"]==scan["absent_group_count"]==1 and not scan["scan_finished_within_deadline"]
 and not scan["all_owned_groups_observed_absent"],"Final group probe overrun never certifies30second drain")
hist=h["_HistoryV8"]()
for i in range(50000):hist.add(dict(pid=i,status=256 if i==40000 else 0,returncode=1 if i==40000 else 0))
need(hist.total==50000 and hist.nonzero==1 and len(hist.first)==len(hist.last)==16,"Summarized actual adopted waits retain every count and hidden nonzero")
need(len(json.dumps(hist.evidence()).encode())<8192,"50000 adopted wait summary bounded independently of lifetime")
# Execute the complete actual C3 supervisor through gate, OSwait, cleanup and terminal evidence.
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
 w=World();w.rows={};w.child={100:[]};b=backend(w);captured={};pid=1000;wait_calls=[];signal_attempts=[]
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
  w.rows[pid]=identity(pid,state="Z"if mode!="refresh-failure"else"S");w.child[100]=[pid];w.child[pid]=[]
  return proc
 def waitid(*a):
  if mode=="refresh-failure":w.read_error=True;return None
  return types.SimpleNamespace(si_pid=pid)
 def waitpid(target,flags):
  wait_calls.append(target)
  if target==-1:raise ChildProcessError()
  if mode=="unknown-wait":return(0,0)
  if mode=="refresh-failure"and wait_calls.count(pid)<3:return(0,0)
  w.read_error=False;w.rows.clear();w.child[100]=[];return(pid,0)
 def signal_identity(row,sig):
  signal_attempts.append((row["pid"],row["start"],sig))
  return True
 b.signal_identity=signal_identity
 fakeos=types.SimpleNamespace(getpid=lambda:100,environ={},pathsep=":",pipe=lambda:(10,11),write=lambda fd,data:len(data),close=lambda fd:None,set_blocking=lambda *a:None,
  waitid=waitid,waitpid=waitpid,waitstatus_to_exitcode=lambda status:0 if status==0 else 1,read=lambda fd,n:b"",killpg=w.killpg,
  P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8)
 w.group_mode="exists"if mode=="live-group"else"absent"
 class Selector:
  def register(self,*a):pass
  def unregister(self,*a):pass
  def select(self,*a,**kw):return []
  def close(self):pass
 def compact(s,name,value,*a,**kw):captured[name]=copy.deepcopy(value);return r("RAM/"+name)
 modules={"pidfd_backend_v1":types.SimpleNamespace(self_probe_v1=lambda:None)}
 ns=dict(__name__="RAM_full_C3_execute",Path=RP,datetime=datetime,timezone=timezone,collections=collections,hashlib=hashlib,json=json,
  time=w.clock,os=fakeos,sys=types.SimpleNamespace(dont_write_bytecode=True,executable=str(ROOT/".runtime/env/bin/python"),modules=modules),
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
 need(result["all_owned_processes_and_groups_drained"]is(mode=="normal"),"Complete actual execute drain gate "+mode)
 if mode=="normal":
  need(result["returncode"]==0 and captured["actual_wait.json"]["group_absence_summary"]["all_owned_groups_observed_absent"],"Actual full execute reaches bounded own wait0 evidence")
  need(captured["actual_wait.json"]["ownership"]["retired_identity_history"]["count"]==1,"Full execute retirement joins actual direct wait")
  contract_node=next(n for n in ast.parse((D/"contract_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="owned_drain_values_v2")
  contract_ns=dict(require=ns["require"],ref=r,DRIVER="RAM/execution.py",OWNED_LIMITS_V8=limits)
  exec(compile(ast.Module(body=[contract_node],type_ignores=[]),"<actual-C3-drain-contract-with-actual-RAM-supervisor-wait>","exec"),contract_ns)
  need(contract_ns["owned_drain_values_v2"](captured["actual_wait.json"])==captured["actual_wait.json"],"Actual full supervisor wait passes actual new-cell drain contract")
 else:
  need(result["returncode"]!=0 and"actual_wait.json"not in captured,"Unresolved full cleanup never publishes success wait "+mode)
 if mode=="refresh-failure":need(bool(signal_attempts),"Cleanup signals known exact rows even after refresh failure")
 need(all(stamp<=30.1 for _,stamp in w.group_calls),"No actual full-execute group observation after finite cleanup "+mode)
 return dict(mode=mode,returncode=result["returncode"],drained=result["all_owned_processes_and_groups_drained"],direct_wait=facts["actual_direct_worker_wait"],group_calls=len(w.group_calls),signal_attempts=len(signal_attempts))
cases=[execute_case(x)for x in("normal","live-group","refresh-failure","unknown-wait")]
# Normalize all four shared helper ASTs; copied helper is tracked separately at seal.
need(all(n in {x.name for x in ast.parse((D/"execution_v1.py").read_text()).body if isinstance(x,(ast.ClassDef,ast.FunctionDef))}for n in names),"Four concrete shared roster helpers in existing execution file")
shared=ast.parse((ROOT/"research/reviews/H058_GPU0_full_training_successor_source_20261017_v8/driver_role.source.txt").read_text())
copied=ast.parse((D/"execution_v1.py").read_text())
for name in names:
 expected=next(n for n in shared.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 actual=next(n for n in copied.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 need(ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False),"Exact shared H058v8 helper AST reuse "+name)
result=dict(schema="C3_remaining_bounded_live_roster_source_checks_v2",verdict="AUTHOR_SOURCE_RAM_CHECKS_ONLY",assertions=len(checks),
 negatives=len(negative),negative_cases=negative,checks=checks,high_churn_helpers=50000,actual_full_execute_RAM_cases=cases,
 original_backend_ancestry_and_pidfd_signal_functions_executed_with_RAM_providers=True,
 guard_process_resource_scope_GPU_or_numerical_operations=False,scientific_payload_reads=False)
(D/"roster_checks.result.v2.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))

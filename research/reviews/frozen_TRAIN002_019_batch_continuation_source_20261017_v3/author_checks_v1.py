"""Author AST/RAM checks for autonomous continuation. No production imports or probes."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from contextlib import contextmanager,redirect_stdout
import ast,copy,hashlib,io,json,os,stat,sys,contextlib
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3"
W=REL+"/worker_entry_v1.py";B=REL+"/batch_ROOT_v1.py";C=REL+"/close_unit_ROOT_v1.py";M=REL+"/source_manifest.v1.json"
OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
SCI="research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/scientific_condition.v5.json"
CHECKS=[]
def need(ok,message="RAM assertion"):
 if not ok:raise ValueError(message)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def doc(path):return json.loads((ROOT/path).read_text())
def fn(path,name,scope):
 text=(ROOT/path).read_text();n=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),path+"#author_RAM","exec"),scope);return scope[name]
def yes(title):CHECKS.append(title)
def rejects(title,call):
 try:call()
 except(ValueError,KeyError,TypeError):yes(title);return
 raise AssertionError("Expected reject "+title)
def original_refs():
 result={}
 for n in ast.parse((ROOT/OLD/"root_contract_v1.py").read_text()).body:
  if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=="dict":
   try:result[n.targets[0].id]={k.arg:ast.literal_eval(k.value)for k in n.value.keywords}
   except(ValueError,TypeError):pass
 return result
def fixture(parent="american_c3_train_002"):
 manifest=doc(M);rows={};refs={}
 def save(path,value):
  rows[path]=copy.deepcopy(value);refs[path]=dict(path=path,sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest());return refs[path]
 def read(rr):return copy.deepcopy(rows[rr["path"]])if rr["path"]in rows else doc(rr["path"])
 def rref(p):
  p=str(p);p=p[len(str(ROOT))+1:]if p.startswith(str(ROOT)+"/")else p
  return refs[p]if p in refs else ref(p)
 def local(p):
  p=Path(p);p=p if p.is_absolute()else ROOT/p
  need(".."not in p.parts and p.is_relative_to(ROOT));return p
 scope=dict(ROOT=ROOT,REL=REL,OLD=OLD,MANIFEST=M,SOURCE=W,WORKER=W,DRIVER=W,BRIDGE=W,SCIENCE=SCI,
  ALLOWED_PARENTS=tuple("american_c3_train_%03d"%i for i in range(2,20)),PARENT=None,PHASE=None,STAGE="simulate",
  SELECTOR=ref("research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json"),
  CHILD=manifest["close_pins"]["guard"],read=read,metadata_ref=rref,Path=Path,datetime=datetime,timezone=timezone,need=need,c=None)
 for name in("validate_batch_authority","registered_row","scientific_config_for_parent","select_parent_from_record"):
  fn(W,name,scope)
 now=datetime.now(timezone.utc);a=doc(REL+"/ROOT_batch_authority.template.inactive.v1.json")
 review=save(REL+"/RAM_only_review.json",dict(verdict="SOURCE_PASS",independent=True,blockers=[],source_manifest_ref=ref(M),
  reviewed_sources=[ref(p)for p in(W,B,C)]))
 a.update(active=True,execute_registered_simulations_and_ordinary_closes=True,ROOT_exclusive_initial_admin_admission_until_coordinator_registered=True,scope_id="RAM",
  source_review_ref=review,created_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=(now+timedelta(hours=40)).isoformat(),
  CPU_lanes=[list(range(8)),list(range(8,16))],run_directory=REL+"/ROOT_actual_batch_RAM",admission_directory=REL+"/ROOT_batch_admission_RAM")
 ar=save(REL+"/RAM_only_batch_authority.json",a)
 control=save(REL+"/RAM_batch_control_input.json",dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1",authority_ref=ar,source_manifest_ref=ref(M),storage=dict(reservation=dict(reserved_bytes=33554432))))
 ctx=SimpleNamespace(control_input_ref=control,w=SimpleNamespace(scientific_config_for_parent=scope["scientific_config_for_parent"]),
  authority=a,authority_ref=ar,manifest=manifest)
 unit=fn(B,"unit_request",dict(ROOT=ROOT,REL=REL,MANIFEST=M,datetime=datetime,timezone=timezone,timedelta=timedelta,need=need,ref=rref))
 q=unit(ctx,parent,0,100000000000,200000000000);qr=save(REL+"/RAM_only_request_"+parent+".json",q)
 active=fn(OLD+"/root_contract_v1.py","active",dict(datetime=datetime,timezone=timezone,timedelta=timedelta,require=need))
 c=SimpleNamespace(ROOT=ROOT,read=read,ref=rref,local=local,active=active,require=need,
  raw=lambda rr:need(ref(rr["path"])==rr),**{k:original_refs()[k]for k in("PIDFD","OWNED","NAMESPACE_REF","V4_REF")},
  POLICY_REF=manifest["close_pins"]["policy"])
 scope["select_parent_from_record"](qr);scope.update(c=c,_active=None,initialize=lambda:(c,None))
 for name in("source_paths","validate_request","phase_paths","direct_roles","input_spec","validate_completion"):
  fn(W,name,scope)
 return SimpleNamespace(scope=scope,rows=rows,refs=refs,read=read,ref=rref,save=save,q=q,qr=qr,a=a,ar=ar,c=c,ctx=ctx,manifest=manifest)

manifest=doc(M)
for rr in manifest["refs"]:need(ref(rr["path"])==rr)
need(len(manifest["refs"])==15);yes("all15_concrete_sources_authenticated")
old=(ROOT/"research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/worker_entry_v1.py").read_text()
new=(ROOT/W).read_text()
oldtree=ast.parse(old);newtree=ast.parse(new)
for name in("worker","_OriginalWriterCurrentGuard","local","metadata_ref","read","guard"):
 a=next(n for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 b=next(n for n in newtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 need(ast.dump(a,include_attributes=False).replace("_frozen30_train001_","_frozen30_train_continuation_")==ast.dump(b,include_attributes=False))
yes("original_worker_gate_slot_resource_Writer_metadata_and_guard_logic_unchanged")
need("d.execute="not in new);yes("original_full_supervisor_execute_and_finite_finally_not_replaced")
parents=[r["parent_id"]for r in manifest["allowed_registered_rows"]]
need(parents==["american_c3_train_%03d"%i for i in range(2,20)])
for parent in parents:
 f=fixture(parent);q,caps,rt=f.scope["validate_request"](f.qr)
 row=f.scope["registered_row"](parent);cfg=q["scientific_config"]
 original=doc(SCI)["original_full30mb_template"]
 expected=dict(original,parent_id=parent,split="train",ancestry_seed=row["ancestry_seed"],mutation_seed=row["mutation_seed"],execution_enabled=True)
 need(cfg==expected and caps["reservation_bytes"]==7000000000)
 yes("complete_request_positive_"+parent)
for parent in("american_c3_train_000","american_c3_train_001","american_c3_val_000","american_c3_test_000","american_c3_train_020"):
 f=fixture();rejects("excluded_"+parent,lambda p=parent:f.scope["registered_row"](p))
f=fixture();other=f.save(REL+"/RAM_other_parent.json",dict(f.q,parent_id="american_c3_train_003"))
rejects("parent_cannot_swap_in_one_interpreter",lambda:f.scope["select_parent_from_record"](other))
for title,mut in[
 ("batch_no_UI_substitution",lambda q:q.update(new_unit_terminal_semantics="UItool0")),
 ("batch_no_replay",lambda q:q.update(automatic_retry=True)),
 ("batch_maximum_two",lambda q:q.update(maximum_producers=3)),
 ("batch_exact_missing_train_only",lambda q:q["allowed_parents"].append("american_c3_train_001")),
 ("batch_no_unbounded_lifetime",lambda q:q.update(expires_at_utc=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat()))]:
 f=fixture();mut(f.rows[f.ar["path"]]);rejects(title,lambda:f.scope["validate_batch_authority"](f.ar))
for title,mut in[
 ("unit_no_seed_change",lambda q:q["scientific_config"].update(ancestry_seed=1)),
 ("unit_no_crop",lambda q:q["scientific_config"].update(sequence_length_bp=2000000)),
 ("unit_no_cap_reduce",lambda q:q["scientific_config"].update(output_cap_bytes=6000000000)),
 ("unit_no_quota_increase",lambda q:q["caps"].update(reservation_bytes=8000000000)),
 ("unit_no_CPU_swap",lambda q:q.update(cpu_ids=list(range(32,40)))),
 ("unit_no_directive_omission",lambda q:q.update(current_Census_original_reserve_execute_authorized=False)),
 ("unit_no_capacity_estimate",lambda q:q["actual_capacity"].update(source_only_estimate=True)),
 ("unit_no_incomplete_closure",lambda q:q["source_sha256"].pop(next(iter(q["source_sha256"])))),
 ("unit_no_native_dependency",lambda q:q.update(native_MLM_dependency=True))]:
 f=fixture();mut(f.rows[f.qr["path"]]);rejects(title,lambda:f.scope["validate_request"](f.qr))

# A genuinely complete original scope_spec uses every current role and all direct roles.
f=fixture();f.scope["validate_request"](f.qr)
roles=doc("research/reviews/Census_observation_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json")["all_actual_parent_bindings"]
roles=dict(roles,Census_observation_explicit_selector_capsule=f.scope["SELECTOR"])
for rr in roles.values():need(ref(rr["path"])==rr)
cr=f.save(f.q["packet_directory"]+"/config.json",f.q["scientific_config"])
direct=f.scope["direct_roles"](f.qr,f.q,cr,f.qr,f.q["runtime_ref"]);bindings=[]
for rr in list(roles.values())+direct:
 if rr not in bindings:bindings.append(copy.deepcopy(rr))
need(len(roles)==47 and len(bindings)<=64 and all(rr in bindings for rr in roles.values()))
yes("complete_all47_roles_and_actual_stage_binding_union_under64")
blobs={}
for rr in bindings:
 data=json.dumps(f.rows[rr["path"]],sort_keys=True).encode()if rr["path"]in f.rows else(ROOT/rr["path"]).read_bytes()
 need(hashlib.sha256(data).hexdigest()==rr["sha256"]);blobs[rr["path"]]=data
fds={};fdcounter=[100]
class P(PurePosixPath):
 def resolve(self):return self
 def lstat(self):
  data=blobs[str(self.relative_to(rrroot))]
  return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_size=len(data),st_dev=1,st_ino=int(hashlib.sha256(str(self).encode()).hexdigest()[:12],16),st_mtime_ns=1,st_ctime_ns=1)
rrroot=P(str(ROOT))
class F(io.BytesIO):
 def fileno(self):return self.fd
def ramopen(p,flags):
 need(flags==os.O_RDONLY|os.O_NOFOLLOW);p=P(p);p.lstat();fd=fdcounter[0];fdcounter[0]+=1;fds[fd]=p;return fd
def ramfdopen(fd,mode):
 need(mode=="rb");stream=F(blobs[str(fds[fd].relative_to(rrroot))]);stream.fd=fd;return stream
en=dict(Path=P,LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock",MAX_JSON=32*2**20,
 json=json,hashlib=hashlib,stat=stat,os=SimpleNamespace(open=ramopen,fdopen=ramfdopen,fstat=lambda fd:fds[fd].lstat(),O_RDONLY=os.O_RDONLY,O_NOFOLLOW=os.O_NOFOLLOW))
for name in("encoded","integer","canonical","file_bytes","parse","paths_overlap","accounting_paths","bound","scope_spec"):fn("src/arggnn/project_budget_v4.py",name,en)
res=dict(key="H041/RAMonly",output=str(rrroot/f.q["output_directory"]),auxiliary_scopes=[str(rrroot/f.q["packet_directory"]/"worker_aux")],
 reserved_bytes=7000000000,consumer_bindings=bindings)
need(len(en["scope_spec"](rrroot,res))==2);yes("full_original_scope_spec_file_bytes_all_actual_bindings_RAM")
r=copy.deepcopy(res);r["consumer_bindings"]*=2
rejects("scope_rejects_above64",lambda:en["scope_spec"](rrroot,r))
r=copy.deepcopy(res);r["consumer_bindings"][0]["path"]="not_binding.lock"
rejects("scope_rejects_lock_source_binding",lambda:en["scope_spec"](rrroot,r))
# Original event construction fits unchanged bound.
txt=(ROOT/manifest["close_pins"]["guard"]["path"]).read_text();an=next(n for n in ast.parse(txt).body if isinstance(n,ast.FunctionDef)and n.name=="_append")
ev=next(n for n in an.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="event"for t in n.targets))
eventenv=dict(state=(None,None,None,roles),events=[],hashes=[],operation="reserve",
 authority_ref=dict(path=REL+"/RAM_authority.json",sha256="0"*64),value=dict(res,root_admission=dict(path=REL+"/RAM_authority.json",sha256="0"*64)))
exec(compile(ast.Module(body=[ev],type_ignores=[]),"RAM_event","exec"),eventenv)
eventbytes=len(en["encoded"](eventenv["event"])+b"\n");need(eventbytes<65536);yes("original_full_binding_reserve_event_below64KiB")

# Real aggregate monitor callable in a complete RAM OS/process boundary.
def monitor(case="success",count=2):
 clock=[0.];rows={};reaped=set();pipes={};commands=[["RAM",str(i)]for i in range(count)];signals=[];starts=[]
 def monotonic():clock[0]+=.01;return clock[0]
 def now():clock[0]+=.01;return clock[0]
 def stat_(pid):return None if pid in reaped else rows.get(pid)
 class Tree:
  def __init__(self,pid):self.known={pid:rows[pid]}
  def refresh(self):
   self.known.update(rows)
   if case=="refresh_failure":raise ValueError("RAM refresh failure")
   return self
  def identities_absent(self):return all(pid in reaped for pid in self.known)
  def evidence(self):return dict(owned_identities=[dict(pid=p,start=r["start"])for p,r in self.known.items()],signals_use_pidfd_only=True)
 class Stream:
  def __init__(self,fd):self.fd=fd
  def fileno(self):return self.fd
  def close(self):pass
 def popen(argv,**kw):
  need(kw["start_new_session"]is True and kw["env"]["CUDA_VISIBLE_DEVICES"]=="")
  pid=100+len(starts);starts.append(pid);rows[pid]=dict(pid=pid,start=pid,group=pid,session=pid,ppid=1,state="R")
  stream=Stream(pid+100);pipes[pid+100]=[b'{"returncode":0}\n',b""]
  return SimpleNamespace(pid=pid,stdout=stream,returncode=None)
 def read_(fd,n):
  return pipes[fd].pop(0)if pipes[fd]else b""
 def waitpid(pid,flags):
  if case=="unresolved":return 0,0
  if pid in reaped:raise ChildProcessError()
  reaped.add(pid);return pid,256 if case=="nonzero"and pid==100 else 0
 def killpg(pid,sig):
  need(sig==0,"No numeric group signaling");raise ProcessLookupError()
 def children(pid):return[p for p in rows if p not in reaped]
 def signal_(row,sig):
  signals.append((row["pid"],sig))
  if case!="unresolved":reaped.add(row["pid"])
 osfake=SimpleNamespace(getpid=lambda:1,environ={},read=read_,waitpid=waitpid,WNOHANG=1,waitstatus_to_exitcode=lambda st:st>>8,
  set_blocking=lambda *a:None,killpg=killpg,sysconf=lambda key:4096)
 owned=SimpleNamespace(_children=children,OwnedTree=Tree,_stat=stat_,_same=lambda a,b:a is not None and a["start"]==b["start"],
  signal_identity=signal_)
 class PathMock:
  def __new__(cls,value):
   return SimpleNamespace(read_text=lambda:"0 0")
 env=dict(ROOT=ROOT,need=need,os=osfake,subprocess=SimpleNamespace(Popen=popen,PIPE=1,STDOUT=2),Path=PathMock,
  time=SimpleNamespace(monotonic=monotonic,time=now,sleep=lambda sec:clock.__setitem__(0,clock[0]+sec)),
  signal=SimpleNamespace(SIGTERM=15,SIGKILL=9),CANCEL=[None])
 result=fn(B,"run_group",env)(commands,owned=owned,deadline_epoch=1 if case=="unresolved"else 1000,rss_cap=100000)
 if case=="success":
  need(result["success"]and len(result["roots"])==count)
  need(all(r["actual_OS_wait_status"]==0 and r["actual_direct_wait"]is True for r in result["roots"]))
  need(result["UI_tool_terminal_claimed"]is False)
 else:need(result["success"]is False)
 return result
for case in("success","nonzero","unresolved","refresh_failure"):
 monitor(case);yes("actual_aggregate_monitor_full_RAM_"+case)
monitor("success",1);yes("actual_admin_single_child_monitor_full_RAM")
for path in(W,B,C):
 env=dict(argparse=__import__("argparse"),json=json,__doc__="RAM only",
  initialize=lambda:(_ for _ in()).throw(ValueError("unexpected initialize")),
  qualified=lambda *a:(_ for _ in()).throw(ValueError("unexpected qualify")))
 stream=io.StringIO()
 with redirect_stdout(stream):need(fn(path,"main",env)([])==0)
 need(json.loads(stream.getvalue())["status"]=="SOURCE_ONLY_INACTIVE");yes("inactive_default_"+Path(path).name)

# Complete future-unit metadata_join and admitted input with pure RAM documents.
def close_fixture(parent="american_c3_train_002"):
 f=fixture(parent);f.scope["validate_request"](f.qr);cfgref=f.save(f.q["packet_directory"]+"/config.json",f.q["scientific_config"])
 direct=f.scope["direct_roles"](f.qr,f.q,cfgref,f.qr,f.q["runtime_ref"]);union=[]
 for rr in list(roles.values())+direct:
  if rr not in union:union.append(rr)
 own=dict(key="H041/RAMunit",reserved_bytes=7000000000,output=str(ROOT/f.q["output_directory"]),
  auxiliary_scopes=[str(ROOT/f.q["packet_directory"]/"worker_aux")],consumer_bindings=union)
 storage=dict(accounting_mode="Census_observation_two_scope_retention_v1",completed_overlay_policy_ref=manifest["close_pins"]["policy"],
  budget_components=dict(consumer_adapter=manifest["close_pins"]["guard"],implementation=manifest["close_pins"]["implementation"],launcher=manifest["close_pins"]["launcher"]),
  reservation=own,reserved_new_bytes=7000000000)
 sr=f.save(f.q["packet_directory"]+"/storage.json",storage)
 inp=dict(schema="frozen_TRAIN002_019_current_Census_simulator_input_v1",parent_id=parent,stage="simulate",phase=f.q["phase"],
  GPU_authorized=False,query_truth_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False,
  root_driver_ref=ref(W),main_source_ref=ref(W),source_sha256=f.q["source_sha256"],root_stage_request_ref=f.qr,
  refs=dict(config=cfgref,storage=sr),storage=storage,runtime_ref=f.q["runtime_ref"],output_directory=f.q["output_directory"],
  worker_auxiliary_scope=f.q["packet_directory"]+"/worker_aux",transitive_source_closure_ref=f.qr,caps_ref=f.qr,caps_locator=["caps"],
  created_at_utc=f.q["created_at_utc"],expires_at_utc=f.q["expires_at_utc"])
 ir=f.save(f.q["packet_directory"]+"/input.json",inp)
 need(f.scope["input_spec"](ir)[2]==f.q["scientific_config"])
 obsref=f.save(f.q["output_directory"]+"/observed/manifest.json",dict(schema="american_c3_observed_v1",parent_id=parent,
  sequence_length_bp=30000000,counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),
  query_ancestry_truth_in_inputs=False,migrations_and_internal_population_labels_removed=True))
 cp=f.save(f.q["output_directory"]+"/completed.json",dict(status="completed_C3_simulation_reconstruction",parent_id=parent,
  sample_unit="haploid",class_order=["EUR","EAS","AFR"],sites=12345,actual_tables_bytes=100000,observation_ref=obsref))
 launch=f.save(inp["worker_auxiliary_scope"]+"/launch.json",dict(pid=100,input_ref=ir,ownership=dict(signals_use_pidfd_only=True,
  no_numeric_group_signals=True,owned_identities=[dict(pid=100,start=1)])))
 wr=f.save(inp["worker_auxiliary_scope"]+"/actual_wait.json",dict(schema="American_C3_full_original_CPU_actual_wait_v1",
  parent_id=parent,phase=f.q["phase"],input_ref=ir,pid=100,completed_ref=cp,returncode=0,actual_OS_wait_status=0,
  actual_direct_worker_wait=True,actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,
  cleanup_errors=[],operational_failure=None,adopted_waits=[],group_observations=[dict(pgid=100,absent=True)],
  known_owned_process_groups=[100],known_owned_pids=[100],supervisor_pid=200,supervisor_process_group=200))
 sw=f.save(REL+"/RAM_supervisor_wait.json",dict(actual_direct_wait=True,actual_OS_wait_status=0,actual_returncode=0,
  all_owned_processes_and_groups_drained=True,pid=200))
 outer=f.save(REL+"/RAM_OS_terminal.json",dict(schema="ROOT_batch_unit_actual_OS_terminal_v1",root_authorizer="root_agent",
  actual_supervisor_OS_terminal_consumed=True,actual_exit_code=0,UI_tool_terminal_claimed=False,batch_authority_ref=f.ar,
  actual_worker_wait_ref=wr,actual_supervisor_wait_ref=sw,completed_ref=cp,input_ref=ir,
  all_owned_processes_and_groups_drained=True,new_ancestry_score=False))
 p=dict(manifest["close_pins"],parent_id=parent,registry_row=f.scope["registered_row"](parent),batch_authority_ref=f.ar,
  producer_source=ref(W),producer_manifest=ref(M),producer_review=f.q["source_review_ref"],batch_control_input_ref=f.q["batch_control_input_ref"],input=ir,wait=wr,completed=cp,
  outer=outer,launch=launch,storage=sr,original_request=f.qr,config=cfgref,supervisor_wait=sw,reservation_key=own["key"])
 join=fn(C,"metadata_join",dict(read=f.read,raw=lambda rr:None,local=lambda p:ROOT/p,need=need))
 return f,p,join
for parent in("american_c3_train_002","american_c3_train_019"):
 f,p,join=close_fixture(parent);need(join(p)[6]==[100,200]);yes("complete_input_and_future_OS_close_join_"+parent)
for title,key,mut in[
 ("close_rejects_unconsumed_OS","outer",lambda v:v.update(actual_supervisor_OS_terminal_consumed=False)),
 ("close_rejects_fabricated_UI_terminal","outer",lambda v:v.update(UI_tool_terminal_claimed=True)),
 ("close_rejects_wrong_batch","outer",lambda v:v.update(batch_authority_ref={})),
 ("close_rejects_nonzero_supervisor","supervisor_wait",lambda v:v.update(actual_OS_wait_status=256)),
 ("close_rejects_unknown_supervisor_wait","supervisor_wait",lambda v:v.update(actual_direct_wait=False)),
 ("close_rejects_supervisor_undrained","supervisor_wait",lambda v:v.update(all_owned_processes_and_groups_drained=False)),
 ("close_rejects_worker_nonzero","wait",lambda v:v.update(returncode=1)),
 ("close_rejects_worker_undrained","wait",lambda v:v.update(all_owned_processes_and_groups_drained=False)),
 ("close_rejects_mutation_seed_drift","config",lambda v:v.update(mutation_seed=1)),
 ("close_rejects_primary_cap_drift","config",lambda v:v.update(output_cap_bytes=6000000000))]:
 f,p,join=close_fixture();mut(f.rows[p[key]["path"]]);rejects(title,lambda:join(p))
# Actual controller run_batch full high-level call graph for two waves.
# Only external operations, IO and authority boundaries are represented by RAM doubles.
def controller_flow(fail=False):
 f=fixture();events=[];saved={};own=dict(key="RAMadmin",reserved_bytes=33554432)
 inp=dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1",authority_ref=f.ar,source_manifest_ref=ref(M),storage=dict(reservation=own))
 ir=dict(path="RAM_input.json",sha256="0"*64)
 class Dir:
  def is_dir(self):return True
  def iterdir(self):return[SimpleNamespace(name="accounting_coordinator.lock")]
 def put(ctx,name,value):
  saved[name]=copy.deepcopy(value);return dict(path=name,sha256="0"*64)
 def group(ctx,commands,seconds,rss,name):
  if ".admission_"in name:
   parent=name.split(".")[0];events.append("admit:"+parent)
   result=dict(execution_started=False,input_ref=dict(path=parent,sha256="0"*64))
  elif ".simulation_"in name:
   events.append("simulate_pair")
   if fail:raise ValueError("RAM actual child failure")
   return dict(roots=[dict(parent=cmd[0],stdout="{}")for cmd in commands])
  else:events.append("close:"+name.split(".")[0]);result=dict(status="ORDINARY_CLOSE_APPLIED")
  return dict(roots=[dict(stdout=json.dumps(result))])
 g=SimpleNamespace(budget_check_v9=lambda *a,**k:None)
 g.budget=SimpleNamespace()
 env=dict(ROOT=ROOT,MANIFEST=M,WORKER=W,CLOSE=C,need=need,read=lambda rr:inp if rr==ir else f.read(rr),
  qualified=lambda rr:(SimpleNamespace(accounting_transaction=lambda rr:contextlib.nullcontext(),ALLOWED_PARENTS=tuple("american_c3_train_%03d"%i for i in range(2,6))),f.a,manifest),
  guard=lambda m:g,state=lambda g,m:({own["key"]:own},set(),dict(limit_bytes=1000000000000,projected_bytes=0)),
  local=lambda p:Dir(),put=put,original_owned=lambda m:object(),SimpleNamespace=SimpleNamespace,
  signal=SimpleNamespace(SIGINT=2,SIGTERM=15,signal=lambda *a:None),CANCEL=[None],ref=ref,
  wait_wave_capacity=lambda ctx,count:(events.append("capacity_before_admission")or(0,0,count)),available_memory=lambda:200000000000,
  unit_request=lambda ctx,parent,lane,capacity,ram:dict(parent_id=parent),group_call=group,
  argv=lambda module,flag,rr:[rr["path"],module,flag],terminal=lambda r:json.loads(r["stdout"]),
  prepare_close=lambda ctx,parent,ir,sup:dict(path=parent,sha256="0"*64),
  accept_close=lambda ctx,parent,r:(events.append("accepted:"+parent)or dict(path=parent,sha256="0"*64)),
  json=json)
 stream=io.StringIO()
 with redirect_stdout(stream):result=fn(B,"run_batch",env)(ir)
 if fail:
  need(result["returncode"]==1 and not any(e.startswith("close:")for e in events))
  need(not any("train_004"in e for e in events));need("failed_preserved.json"in saved)
 else:
  need(result["returncode"]==0 and "completed.json"in saved)
  need(events.index("accepted:american_c3_train_003")<events.index("admit:american_c3_train_004"))
  need(events.count("simulate_pair")==2)
 return events
controller_flow();yes("complete_actual_controller_two_wave_RAM_closes_before_next_admission")
controller_flow(True);yes("complete_actual_controller_failure_RAM_no_close_or_replay_or_next_wave")


# Separate32MiB admin reservation preserves full actual runtime source closure.
f=fixture();adminbindings=[]
for rr in list(roles.values())+manifest["refs"]+list(manifest["close_pins"].values())+[ref(M),f.ar,f.q["source_review_ref"],manifest["controller_runtime_ref"]]:
 if rr not in adminbindings:adminbindings.append(copy.deepcopy(rr))
need(len(adminbindings)<=64 and all(rr in adminbindings for rr in roles.values()))
for rr in adminbindings:
 data=json.dumps(f.rows[rr["path"]],sort_keys=True).encode()if rr["path"]in f.rows else(ROOT/rr["path"]).read_bytes()
 need(hashlib.sha256(data).hexdigest()==rr["sha256"]);blobs[rr["path"]]=data
adminres=dict(key="H041/RAMbatchAdmin",output=str(rrroot/f.a["run_directory"]),auxiliary_scopes=[],
 reserved_bytes=33554432,consumer_bindings=adminbindings)
need(len(en["scope_spec"](rrroot,adminres))==1);yes("complete32MiB_admin_actual_sources_and_all47roles_full_scope_spec_RAM")
# Only the existing2 pool nodes are inspected by the runtime availability source.
opened=[];closed=[]
class SlotPath:
 def __init__(self,value):self.value=value
 def lstat(self):return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_size=0,st_dev=1,st_ino=10 if self.value.endswith("0.lock")else 11)
def slot_open(p,flags):
 opened.append(p.value);return 10 if p.value.endswith("0.lock")else 11
def flock(fd,flags):
 if fd==11:raise BlockingIOError()
slot_env=dict(need=need,local=SlotPath,stat=stat,os=SimpleNamespace(open=slot_open,close=closed.append,
 O_RDWR=2,O_NOFOLLOW=1,fstat=lambda fd:SimpleNamespace(st_dev=1,st_ino=fd,st_size=0)),
 fcntl=SimpleNamespace(flock=flock,LOCK_EX=2,LOCK_NB=4))
need(fn(B,"available_original_slots",slot_env)(SimpleNamespace(manifest=manifest))==1)
need(opened==manifest["synchronization_pool"]["slot_paths"]and closed==[10,11]);yes("original_shared_slot_availability_RAM_bounded_two_nodes_and_all_fds_closed")


# Actual shared coordinator context: RAM inode/OFD/exclusion boundaries only.
def coordinator_case(case="success"):
 f=fixture();events=[];clock=[0.];attempt=[0]
 control=dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1",source_manifest_ref=ref(M),authority_ref=f.ar,
  coordinator=dict(path=f.a["run_directory"]+"/accounting_coordinator.lock",st_dev=1,st_ino=10))
 rr=dict(path="RAM_control.json",sha256="0"*64)
 class PathDouble:
  def lstat(self):
   ino=11 if case=="replaced"and attempt[0]>0 else 10
   return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_size=0,st_dev=1,st_ino=ino)
 def flock(fd,flags):
  attempt[0]+=1;events.append("probe")
  if case=="timeout"or attempt[0]==1:raise BlockingIOError()
 def monotonic():clock[0]+=1;return clock[0]
 env=dict(contextlib=contextlib,read=lambda r:control,need=need,metadata_ref=ref,MANIFEST=M,
  validate_batch_authority=lambda r:f.a,local=lambda p:PathDouble(),datetime=datetime,stat=stat,
  os=SimpleNamespace(open=lambda p,flags:77,close=lambda fd:events.append("close"),
   O_RDWR=2,O_NOFOLLOW=1,fstat=lambda fd:SimpleNamespace(st_dev=1,st_ino=10,st_size=0,st_nlink=1)),
  time=SimpleNamespace(monotonic=monotonic,time=lambda:datetime.now(timezone.utc).timestamp(),
   sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds)),
  fcntl=SimpleNamespace(flock=flock,LOCK_EX=2,LOCK_NB=4))
 context=fn(W,"accounting_transaction",env)
 def call():
  with context(rr):events.append("held_original_OFD")
 if case=="success":call();need(events==["probe","probe","held_original_OFD","close"])
 else:rejects("coordinator_rejects_"+case,call);need(events[-1]=="close"and "held_original_OFD"not in events)
 return events
coordinator_case();yes("actual_shared_coordinator_RAM_original_inode_no_follow_exclusion_and_release")
coordinator_case("timeout");coordinator_case("replaced")
for source,name,core in((W,"admit","admit_under_coordinator"),(C,"close_once","close_under_coordinator")):
 events=[]
 @contextmanager
 def coordinator(ref):
  events.append("coordinator_enter")
  try:yield
  finally:events.append("coordinator_exit")
 def operation(ref):
  need(events==["coordinator_enter"]);events.extend(["capture","authority","original_apply"]);return "ok"
 env=dict(read=lambda rr:dict(batch_control_input_ref={}),accounting_transaction=coordinator,support=lambda:None,
  **{core:operation})
 need(fn(source,name,env)({})=="ok")
 need(events==["coordinator_enter","capture","authority","original_apply","coordinator_exit"])
 yes("complete_"+name+"_transaction_wrapper_RAM_holds_coordinator_across_capture_authority_apply")

print(json.dumps(dict(schema="TRAIN002_019_author_AST_RAM_checks_v1",status="AUTHOR_PASS",count=len(CHECKS),checks=CHECKS,
 source_manifest_ref=ref(M),source_refs=[ref(p)for p in(W,B,C)],summary=dict(current_roles=len(roles),direct_roles=len(direct),
 union_unique_refs=len(bindings),admin_union_unique_refs=len(adminbindings),event_bytes=eventbytes),actual_guard_imports_or_resource_process_lock_GPU_probes=0,
 actual_admission_or_scientific_execution=False,payload_truth_reads=False),sort_keys=True,indent=2))

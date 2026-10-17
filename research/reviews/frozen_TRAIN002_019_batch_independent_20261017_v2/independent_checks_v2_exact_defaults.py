"""Independent review: compile exact source code objects without executing production modules.
Only source/compact static metadata are read; all accounting/process/lock boundaries are RAM doubles.
"""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace,FunctionType,CodeType
from datetime import datetime,timezone,timedelta
from contextlib import contextmanager,redirect_stdout
import ast,copy,hashlib,io,json,stat,contextlib,argparse
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v2"
W=REL+"/worker_entry_v1.py"; B=REL+"/batch_ROOT_v1.py"; C=REL+"/close_unit_ROOT_v1.py"; M=REL+"/source_manifest.v1.json"
OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
V5="research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5"
SCI=V5+"/scientific_condition.v5.json"
CORR="research/reviews/frozen_TRAIN002_019_batch_recipe_correction_20261017_v1"
checks=[]
def need(ok,message="independent RAM assertion"):
 if not ok:raise ValueError(message)
def check(name,ok=True):
 need(ok,name);checks.append(name)
def deny(name,call):
 try:call()
 except (ValueError,KeyError,TypeError):check(name);return
 raise AssertionError("Expected rejection: "+name)
def actualref(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def actualdoc(path):return json.loads((ROOT/path).read_text())
sources={p:(ROOT/p).read_text()for p in(W,B,C,OLD+"/root_contract_v1.py",OLD+"/root_driver_v1.py","src/arggnn/project_budget_v4.py")}
compiled={p:compile(s,p,"exec")for p,s in sources.items()}
def function(p,name,env):
 code=next(v for v in compiled[p].co_consts if isinstance(v,CodeType)and v.co_name==name)
 n=next(n for n in ast.parse(sources[p]).body if isinstance(n,ast.FunctionDef)and n.name==name)
 f=FunctionType(code,env,name,tuple(ast.literal_eval(v)for v in n.args.defaults));f.__kwdefaults__={k.arg:ast.literal_eval(v)for k,v in zip(n.args.kwonlyargs,n.args.kw_defaults)if v is not None};return f
def node(p,name):
 return next(v for v in ast.parse(sources.get(p,(ROOT/p).read_text())).body if isinstance(v,(ast.FunctionDef,ast.ClassDef))and v.name==name)
def unique(vs):
 out=[]
 for v in vs:
  if v not in out:out.append(v)
 return out
m=actualdoc(M);h=actualdoc(REL+"/HANDOFF.v2.json")
check("sealed_handoff_exact",actualref(REL+"/HANDOFF.v2.json")["sha256"]=="28e8b944e9c84c5ac91738c0fb3112be7e6eede1ce8755fad482a205b275a0e9")
check("sealed_manifest_exact",actualref(M)["sha256"]=="6742dc3b8841d7b2ebfbcdce64aaf2d362548b1d1133eaf42b82b1e1ff6a868a")
for r in m["refs"]:need(actualref(r["path"])==r)
check("all15_sealed_source_refs_authenticated",len(m["refs"])==15 and h["source_refs"]==[actualref(p)for p in(W,B,C)])
condition=actualdoc(SCI);parents=["american_c3_train_%03d"%i for i in range(2,20)]
check("manifest_exact18_original_rows",m["allowed_registered_rows"]==[r for r in condition["original_registry"]["parents"]if r["parent_id"]in parents])
oldtree=ast.parse((ROOT/V5/"worker_entry_v1.py").read_text())
for name in("worker","_OriginalWriterCurrentGuard","local","metadata_ref","read","guard"):
 old=next(n for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 check("unchanged_original_"+name,ast.dump(old,include_attributes=False).replace("_frozen30_train001_","_frozen30_train_continuation_")==ast.dump(node(W,name),include_attributes=False))
old=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name=="execute_in_admitted_context_v1")
new=copy.deepcopy(node(W,"execute_in_admitted_context_v1"))
old=copy.deepcopy(old)
# The only scientific bridge difference is the stronger selected-parent predicate.
for x in ast.walk(old):
 if isinstance(x,ast.BoolOp)and any(isinstance(t,ast.Compare)and ast.unparse(t)=='matches[0][\'parent_id\'] != \'american_c3_train_000\''for t in x.values):
  x.values=[ast.Name(id="SELECTED_PARENT_CHECK",ctx=ast.Load())if isinstance(t,ast.Compare)and ast.unparse(t)=='matches[0][\'parent_id\'] != \'american_c3_train_000\'' else t for t in x.values]
for x in ast.walk(new):
 if isinstance(x,ast.BoolOp)and any(isinstance(t,ast.Compare)and ast.unparse(t)=="matches[0]['parent_id'] == PARENT"for t in x.values):
  vals=[];done=False
  for t in x.values:
   if isinstance(t,ast.Compare)and ast.unparse(t)in("matches[0]['parent_id'] == PARENT","PARENT in ALLOWED_PARENTS"):
    if not done:vals.append(ast.Name(id="SELECTED_PARENT_CHECK",ctx=ast.Load()));done=True
   else:vals.append(t)
  x.values=vals
check("original_scientific_bridge_exact_except_stronger_parent_predicate",ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False))
check("original_supervisor_execute_not_replaced","d.execute="not in sources[W]and "d.validate_completion=validate_completion"in sources[W])
check("unchanged_original_shared_pool_pins",m["synchronization_pool"]["slot_paths"]==[OLD+"/cpu_slot_0.lock",OLD+"/cpu_slot_1.lock"]and m["maximum_producers"]==m["maximum_open_unit_reservations"]==2)
original_constants={}
for n in ast.parse(sources[OLD+"/root_contract_v1.py"]).body:
 if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=="dict":
  try:original_constants[n.targets[0].id]={k.arg:ast.literal_eval(k.value)for k in n.value.keywords}
  except (ValueError,TypeError):pass
class Repository:
 def __init__(self):self.docs={}
 def save(self,path,value):
  self.docs[str(path)]=copy.deepcopy(value);return self.ref(path)
 def ref(self,path):
  path=str(path);path=path[len(str(ROOT))+1:]if path.startswith(str(ROOT)+"/")else path
  if path not in self.docs:return actualref(path)
  return dict(path=path,sha256=hashlib.sha256(json.dumps(self.docs[path],sort_keys=True).encode()).hexdigest())
 def read(self,r):
  p=r["path"];v=copy.deepcopy(self.docs[p])if p in self.docs else actualdoc(p)
  need(self.ref(p)==r,"RAM immutable ref");return v
def fixture(parent=parents[0]):
 repo=Repository();now=datetime.now(timezone.utc)
 review=repo.save(REL+"/RAM_independent_review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],source_manifest_ref=actualref(M),reviewed_sources=[actualref(p)for p in(W,B,C)]))
 a=actualdoc(REL+"/ROOT_batch_authority.template.inactive.v1.json")
 a.update(active=True,execute_registered_simulations_and_ordinary_closes=True,ROOT_exclusive_initial_admin_admission_until_coordinator_registered=True,scope_id="INDEPENDENT_RAM",
  source_review_ref=review,created_at_utc=(now-timedelta(seconds=2)).isoformat(),expires_at_utc=(now+timedelta(hours=40)).isoformat(),
  CPU_lanes=[list(range(8)),list(range(8,16))],run_directory=REL+"/ROOT_actual_batch_INDEPENDENT_RAM",admission_directory=REL+"/ROOT_batch_admission_INDEPENDENT_RAM")
 ar=repo.save(REL+"/RAM_authority.json",a)
 control=repo.save(REL+"/RAM_control.json",dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1",authority_ref=ar,source_manifest_ref=actualref(M),storage=dict(reservation=dict(reserved_bytes=33554432))))
 e=dict(ROOT=ROOT,REL=REL,OLD=OLD,MANIFEST=M,SOURCE=W,WORKER=W,DRIVER=W,BRIDGE=W,SCIENCE=SCI,
  ALLOWED_PARENTS=tuple(parents),PARENT=None,PHASE=None,STAGE="simulate",SELECTOR=m["close_pins"]["selector"],CHILD=m["close_pins"]["guard"],
  read=repo.read,metadata_ref=repo.ref,Path=Path,datetime=datetime,timezone=timezone,need=need,c=None,_active=None)
 for name in("validate_batch_authority","registered_row","scientific_config_for_parent","select_parent_from_record"):
  e[name]=function(W,name,e)
 ctx=SimpleNamespace(control_input_ref=control,w=SimpleNamespace(scientific_config_for_parent=e["scientific_config_for_parent"]),authority=a,authority_ref=ar,manifest=m)
 unit=function(B,"unit_request",dict(REL=REL,MANIFEST=M,datetime=datetime,timezone=timezone,timedelta=timedelta,need=need,ref=repo.ref))
 q=unit(ctx,parent,0,10**12,2*10**11);qr=repo.save(REL+"/RAM_request.json",q)
 c=SimpleNamespace(ROOT=ROOT,read=repo.read,ref=repo.ref,raw=lambda r:need(repo.ref(r["path"])==r),local=lambda p:ROOT/p,require=need,
  active=function(OLD+"/root_contract_v1.py","active",dict(datetime=datetime,timezone=timezone,timedelta=timedelta,require=need)),
  POLICY_REF=m["close_pins"]["policy"],**{k:original_constants[k]for k in("PIDFD","OWNED","NAMESPACE_REF","V4_REF")})
 e["select_parent_from_record"](qr);e.update(c=c,initialize=lambda:(c,None))
 for name in("source_paths","validate_request","phase_paths","direct_roles","input_spec","validate_completion"):
  e[name]=function(W,name,e)
 return SimpleNamespace(repo=repo,env=e,c=c,q=q,qr=qr,a=a,ar=ar,control=control,ctx=ctx)
for parent in parents:
 f=fixture(parent);q,caps,rt=f.env["validate_request"](f.qr)
 check("real_request_joins_"+parent,q["scientific_config"]==dict(condition["original_full30mb_template"],parent_id=parent,split="train",ancestry_seed=21001+int(parent[-3:]),mutation_seed=1021001+int(parent[-3:]),execution_enabled=True))
for bad in("american_c3_train_000","american_c3_train_001","american_c3_train_020","american_c3_val_000","american_c3_test_000"):
 f=fixture();deny("excluded_parent_"+bad,lambda:f.env["registered_row"](bad))
for label,mut in[
 ("seed",lambda q:q["scientific_config"].update(mutation_seed=1)),
 ("length",lambda q:q["scientific_config"].update(sequence_length_bp=2000000)),
 ("primary_cap",lambda q:q["scientific_config"].update(output_cap_bytes=6000000000)),
 ("reservation",lambda q:q["caps"].update(reservation_bytes=8000000000)),
 ("CPU_lane",lambda q:q.update(cpu_ids=list(range(24,32)))),
 ("closure",lambda q:q["source_sha256"].pop(next(iter(q["source_sha256"])))),
 ("native_dependency",lambda q:q.update(native_MLM_dependency=True)),
 ("truth_access",lambda q:q.update(query_truth_access_authorized=True))]:
 f=fixture();mut(f.repo.docs[f.qr["path"]]);f.qr=f.repo.ref(f.qr["path"])
 deny("request_rejects_"+label,lambda:f.env["validate_request"](f.qr))
for label,mut in[
 ("retry",lambda a:a.update(automatic_retry=True)),("third_producer",lambda a:a.update(maximum_producers=3)),
 ("no_initial_exclusivity",lambda a:a.update(ROOT_exclusive_initial_admin_admission_until_coordinator_registered=False)),
 ("UI_terminal_substitution",lambda a:a.update(new_unit_terminal_semantics="UI0")),
 ("lifetime",lambda a:a.update(expires_at_utc=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat()))]:
 f=fixture();mut(f.repo.docs[f.ar["path"]]);f.ar=f.repo.ref(f.ar["path"])
 deny("batch_authority_rejects_"+label,lambda:f.env["validate_batch_authority"](f.ar))
# Execute exact contextmanager code with all lock/inode/time effects replaced by RAM.
def coordinator(case):
 events=[];clock=[0.];attempt=[0];now=datetime.now(timezone.utc)
 a=dict(run_directory=REL+"/ROOT_actual_batch_RAM",expires_at_utc=(now+timedelta(hours=2)).isoformat())
 control=dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1",source_manifest_ref=actualref(M),authority_ref={},
  coordinator=dict(path=a["run_directory"]+"/accounting_coordinator.lock",st_dev=1,st_ino=9))
 if case=="wrong_path":control["coordinator"]["path"]+=".other"
 class P:
  def lstat(self):
   return SimpleNamespace(st_mode=stat.S_IFLNK if case=="symlink" else stat.S_IFREG,st_nlink=2 if case=="linked"else 1,st_size=1 if case=="nonempty"else 0,st_dev=1,st_ino=10 if case=="replaced"and attempt[0]else 9)
 def flock(fd,flags):
  attempt[0]+=1;events.append("probe")
  if case=="timeout"or attempt[0]==1:raise BlockingIOError()
 def mono():clock[0]+=.5;return clock[0]
 env=dict(read=lambda r:control,need=need,metadata_ref=actualref,MANIFEST=M,validate_batch_authority=lambda r:a,local=lambda p:P(),
  stat=stat,datetime=datetime,os=SimpleNamespace(open=lambda *args:(events.append("open")or 42),close=lambda fd:events.append("close"),O_RDWR=2,O_NOFOLLOW=4,
   fstat=lambda fd:SimpleNamespace(st_dev=1,st_ino=10 if case=="fd_mismatch"else 9,st_size=0,st_nlink=1)),
  fcntl=SimpleNamespace(flock=flock,LOCK_EX=2,LOCK_NB=8),time=SimpleNamespace(monotonic=mono,time=lambda:now.timestamp(),sleep=lambda s:clock.__setitem__(0,clock[0]+s)))
 lock=contextmanager(function(W,"accounting_transaction",env))
 def call():
  with lock({}):
   events.append("capture_authority_apply")
   if case=="body_exception":raise ValueError("intentional body failure")
 if case in("success",):
  call();check("coordinator_held_across_body_then_fd_release",events==["open","probe","probe","capture_authority_apply","close"])
 elif case=="body_exception":
  deny("coordinator_body_exception_propagates",call);check("coordinator_exception_releases",events[-1]=="close")
 else:
  deny("coordinator_rejects_"+case,call);check("coordinator_no_body_"+case,"capture_authority_apply"not in events and ("open"not in events or events[-1]=="close"))
for case in("success","body_exception","wrong_path","symlink","linked","nonempty","fd_mismatch","replaced","timeout"):coordinator(case)
# Complete actual admission core and wrapper; original mutex must be released before apply
# while the separate coordinator remains held over capture, authority and apply.
f=fixture();q=f.q;events=[];held=[False];sched=[False];saved={}
@contextmanager
def coord(r):
 need(r==f.control and not held[0]);held[0]=True;events.append("coordinator_enter")
 try:yield
 finally:held[0]=False;events.append("coordinator_exit")
@contextmanager
def mutex(root):
 need(held[0]and not sched[0]);sched[0]=True;events.append("mutex_enter")
 try:yield
 finally:sched[0]=False;events.append("mutex_exit")
roles=actualdoc("research/reviews/Census_observation_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json")["all_actual_parent_bindings"]
roles=dict(roles,Census_observation_explicit_selector_capsule=m["close_pins"]["selector"])
for rr in roles.values():need(actualref(rr["path"])==rr)
def state_locked(*a,**k):
 need(held[0]and sched[0]);events.append("capture");return({},set(),dict(prefix="RAM"),roles)
def draft(op,value,**kw):
 need(held[0]and not sched[0]);events.append("draft");return dict(request=copy.deepcopy(value),expected_before=kw["expected_before"])
def apply(rr,**kw):
 need(held[0]and not sched[0]);events.append("apply");return dict(value=f.repo.read(rr)["request"],initial_ledger_pins=dict(RAM=True))
class Packet:
 def exists(self):return False
 def mkdir(self):events.append("packet_mkdir")
 def __truediv__(self,name):return ROOT/q["packet_directory"]/name
packet=Packet()
def save(directory,name,v):return f.repo.save(q["packet_directory"]+"/"+name,v)
g=SimpleNamespace(MODE="Census_observation_two_scope_retention_v1",budget=SimpleNamespace(SOURCE_REL=m["close_pins"]["implementation"]["path"],LAUNCHER_REL=m["close_pins"]["launcher"]["path"],v4=SimpleNamespace(mutex=mutex)),
 _state_locked=state_locked,_pins=lambda v:v,operation_draft_v1=draft,apply_reserve_v1=apply)
c=f.c
c.initialize=lambda:(c,None);c.validate_request=lambda r:(q,q["caps"],{});c.phase_paths=lambda p:(packet,q["output_directory"])
c.local=lambda p:SimpleNamespace(exists=lambda:False);c.original_guard=lambda:g;c.save=save;c.source_paths=lambda:tuple(r["path"]for r in m["refs"])
c.unique=unique;c.direct_roles=f.env["direct_roles"];c.make_config=lambda q,caps:q["scientific_config"];c.PARENT=parents[0];c.PHASE=q["phase"];c.STAGE="simulate";c.GUARD_REF=m["close_pins"]["guard"]
env=dict(context=SimpleNamespace(initialize=c.initialize,DRIVER=W,WORKER=W),PARENT=parents[0],accounting_transaction=coord,read=f.repo.read)
env["admit_under_coordinator"]=function(W,"admit_under_coordinator",env)
result=function(W,"admit",env)(f.qr);inp=f.repo.read(result["input_ref"]);own=inp["storage"]["reservation"]
check("actual_admit_complete_transaction_no_recursive_scheduler_mutex",events.index("capture")<events.index("mutex_exit")<events.index("draft")<events.index("apply")<events.index("coordinator_exit"))
check("admission_actual_inline_closure_caps_config_and_storage",inp["transitive_source_closure_ref"]==inp["caps_ref"]==f.qr and inp["caps_locator"]==["caps"]and inp["storage"]["reservation"]["reserved_bytes"]==7000000000)
check("unit_union_all47_roles23direct61unique",len(roles)==47 and len(f.env["direct_roles"](f.qr,q,inp["refs"]["config"],f.qr,q["runtime_ref"]))==23 and len(own["consumer_bindings"])==61 and all(r in own["consumer_bindings"]for r in roles.values()))
admin=unique(list(roles.values())+m["refs"]+list(m["close_pins"].values())+[actualref(M),f.ar,q["source_review_ref"],m["controller_runtime_ref"]])
check("admin_union_all47_roles58unique",len(admin)==58 and all(r in admin for r in roles.values()))
# Original scope_spec and bound execute; only file-byte and path filesystem edges are RAM.
class P(PurePosixPath):
 def resolve(self):return self
def ramfile(root,path):
 r=f.repo.ref(path);p=r["path"]
 raw=json.dumps(f.repo.docs[p],sort_keys=True).encode()if p in f.repo.docs else(ROOT/p).read_bytes()
 return raw,hashlib.sha256(raw).hexdigest()
e=dict(Path=P,LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock",file_bytes=ramfile,json=json)
for name in("integer","canonical","paths_overlap","accounting_paths","bound","scope_spec"):e[name]=function("src/arggnn/project_budget_v4.py",name,e)
check("original_scope_spec_entire61binding_two_scope_contract",len(e["scope_spec"](P(str(ROOT)),own))==2)
adminvalue=dict(key="H041/RAMadmin",output=str(ROOT/f.a["run_directory"]),auxiliary_scopes=[],reserved_bytes=33554432,consumer_bindings=admin)
check("original_scope_spec_actual32MiB_admin",len(e["scope_spec"](P(str(ROOT)),adminvalue))==1)
bad=copy.deepcopy(own);bad["consumer_bindings"]*=2
deny("original_scope_spec_rejects_over64",lambda:e["scope_spec"](P(str(ROOT)),bad))
bad=copy.deepcopy(own);bad["consumer_bindings"][0]["path"]="invalid.lock"
deny("original_scope_spec_rejects_lock_binding",lambda:e["scope_spec"](P(str(ROOT)),bad))
# Actual close metadata and actual controller prepare_close connected in RAM.
cp=f.repo.save(q["output_directory"]+"/completed.json",dict(status="completed_C3_simulation_reconstruction",parent_id=parents[0],sample_unit="haploid",class_order=["EUR","EAS","AFR"],sites=100,actual_tables_bytes=1000,observation_ref=f.repo.save(q["output_directory"]+"/observed/manifest.json",dict(schema="american_c3_observed_v1",parent_id=parents[0],sequence_length_bp=30000000,counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False,migrations_and_internal_population_labels_removed=True))))
launch=f.repo.save(inp["worker_auxiliary_scope"]+"/launch.json",dict(pid=101,input_ref=result["input_ref"],ownership=dict(signals_use_pidfd_only=True,no_numeric_group_signals=True,owned_identities=[dict(pid=101,start=1001)])))
wr=f.repo.save(inp["worker_auxiliary_scope"]+"/actual_wait.json",dict(schema="American_C3_full_original_CPU_actual_wait_v1",parent_id=parents[0],phase=q["phase"],input_ref=result["input_ref"],pid=101,completed_ref=cp,returncode=0,actual_OS_wait_status=0,actual_direct_worker_wait=True,actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,cleanup_errors=[],operational_failure=None,adopted_waits=[],group_observations=[dict(pgid=101,absent=True)],known_owned_process_groups=[101],known_owned_pids=[101],supervisor_pid=102,supervisor_process_group=102))
sup=dict(pid=102,actual_direct_wait=True,actual_OS_wait_status=0,actual_returncode=0,all_owned_processes_and_groups_drained=True,stdout=json.dumps(dict(status="PREPARED",returncode=0,actual_wait_ref=wr,completed_ref=cp)))
ctx=f.ctx;ctx.g=SimpleNamespace(budget=SimpleNamespace(digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()))
prepare_env=dict(read=f.repo.read,terminal=lambda s:json.loads(s["stdout"]),need=need,put=lambda ctx,name,v:f.repo.save(f.a["run_directory"]+"/"+name,v),
 datetime=datetime,timezone=timezone,timedelta=timedelta,MANIFEST=M,ref=f.repo.ref)
ca=function(B,"prepare_close",prepare_env)(ctx,parents[0],result["input_ref"],sup);ca=f.repo.read(ca)
p=dict(m["close_pins"],parent_id=parents[0],registry_row=f.env["registered_row"](parents[0]),batch_authority_ref=f.ar,batch_control_input_ref=f.control,producer_source=actualref(W),producer_manifest=actualref(M),producer_review=q["source_review_ref"],reservation_key=own["key"],**ca["evidence_refs"])
join=function(C,"metadata_join",dict(read=f.repo.read,raw=lambda r:need(f.repo.ref(r["path"])==r),local=lambda p:ROOT/p,need=need))
check("actual_prepare_close_to_actual_metadata_join_complete",join(p)[6:]==([101,102],[101,102])and ca["permanent_route_revocation_ref"]and ca["batch_control_input_ref"]==f.control)
for label,key,field,value in[
 ("worker_status","wait","actual_OS_wait_status",256),("worker_cleanup","wait","cleanup_errors",["error"]),
 ("supervisor_direct","supervisor_wait","actual_direct_wait",False),("supervisor_status","supervisor_wait","actual_OS_wait_status",256),
 ("supervisor_drain","supervisor_wait","all_owned_processes_and_groups_drained",False),("outer_UI_claim","outer","UI_tool_terminal_claimed",True),
 ("outer_unconsumed","outer","actual_supervisor_OS_terminal_consumed",False),("wrong_batch","outer","batch_authority_ref",{}),
 ("observation_length","completed","sample_unit","diploid")]:
 pp=copy.deepcopy(p);old=f.repo.docs[pp[key]["path"]];changed=copy.deepcopy(old);changed[field]=value;pp[key]=f.repo.save(pp[key]["path"]+".negative.json",changed)
 deny("actual_close_join_rejects_"+label,lambda:join(pp))
# Actual close wrapper must hold same coordinator over entire core.
events=[]
@contextmanager
def closecoord(rr):
 events.append("enter")
 try:yield
 finally:events.append("exit")
def closecore(rr):
 need(events==["enter"]);events.extend(["capture","draft","apply"]);return True
e=dict(support=lambda:None,read=lambda rr:dict(batch_control_input_ref=f.control),accounting_transaction=closecoord,close_under_coordinator=closecore)
check("actual_close_wrapper_coordinator_covers_capture_draft_apply",function(C,"close_once",e)({})and events==["enter","capture","draft","apply","exit"])
closeast=node(C,"close_under_coordinator")
withnodes=[n for n in ast.walk(closeast)if isinstance(n,ast.With)]
check("close_original_mutex_contains_fresh_capture_and_draft_but_not_apply",len(withnodes)==1 and "_state_locked"in ast.unparse(withnodes[0])and "operation_draft_v1"in ast.unparse(withnodes[0])and "apply_close_completed_v1"not in ast.unparse(withnodes[0]))
check("close_preserves_original_fresh_absence_measurement_api","g._absent(pid)"in ast.unparse(closeast)and "g.apply_close_completed_v1" in ast.unparse(closeast)and "consumer_bindings_v1" in ast.unparse(closeast))
# Exact monitor with actual original OwnedTree, all OS/time/process effects RAM only.
ownedsource=(ROOT/m["owned_ref"]["path"]).read_text()
def monitor(case):
 ticks=[0.];rows={};pipes={};exited=set();reaped=set();started=[];signals=[];waits=[];adopt=False
 def mono():ticks[0]+=.01;return ticks[0]
 def row(pid):return None if pid in reaped else rows.get(pid)
 def children(pid):return sorted(p for p,r in rows.items()if p not in reaped and r["ppid"]==pid)
 def waitpid(pid,flag):
  if case=="unknown":return(0,0)
  if pid in reaped:raise ChildProcessError()
  if len(started)<2:return(0,0)
  if pid not in rows:return(0,0)
  if pid==300 and rows[300]["ppid"]!=1:return(0,0)
  reaped.add(pid);waits.append(pid)
  for r in rows.values():
   if r["ppid"]==pid:r["ppid"]=1
  return pid,256 if(case=="nonzero"and pid==100)or(case=="adopted_nonzero"and pid==300)else 0
 class Stream:
  def __init__(self,fd):self.fd=fd
  def fileno(self):return self.fd
  def close(self):pass
 def popen(cmd,**kw):
  need(kw["start_new_session"]and kw["env"]["CUDA_VISIBLE_DEVICES"]=="");pid=100+len(started);started.append(pid)
  rows[pid]=dict(pid=pid,ppid=1,group=pid,session=pid,start=pid+1000,state="R")
  pipes[pid]=[b"x"*262145 if case=="stdout_overflow"and pid==100 else b'{"status":"RAM"}\n',b""]
  return SimpleNamespace(pid=pid,stdout=Stream(pid),returncode=None)
 def read(fd,n):
  if case=="no_eof":return None
  return pipes[fd].pop(0)if pipes[fd]else b""
 def killpg(pid,sig):
  need(sig==0)
  if case=="group_present":return None
  raise ProcessLookupError()
 osfake=SimpleNamespace(getpid=lambda:1,environ={},read=read,waitpid=waitpid,WNOHANG=1,waitstatus_to_exitcode=lambda s:s>>8,killpg=killpg,set_blocking=lambda *a:None,sysconf=lambda x:4096)
 oenv=dict(os=osfake,_stat=row,_children=children)
 for name in("_same",):oenv[name]=FunctionType(next(v for v in compile(ownedsource,"owned","exec").co_consts if isinstance(v,CodeType)and v.co_name==name),oenv,name)
 classnode=next(n for n in ast.parse(ownedsource).body if isinstance(n,ast.ClassDef)and n.name=="OwnedTree")
 exec(compile(ast.Module(body=[classnode],type_ignores=[]),"owned_class_RAM","exec"),oenv)
 def statrow(pid):
  nonlocal adopt
  if len(started)==2 and case=="adopted_nonzero"and not adopt:
   rows[300]=dict(pid=300,ppid=100,group=100,session=100,start=1300,state="R");adopt=True
  return row(pid)
 oenv["_stat"]=statrow
 # Default args were bound at class construction; wrap constructor with explicit RAM readers.
 owned=SimpleNamespace(_children=children,_stat=statrow,_same=oenv["_same"],OwnedTree=lambda pid:oenv["OwnedTree"](pid,owner=1,read_stat=statrow,read_children=children),
  signal_identity=lambda r,s:signals.append((r["pid"],s)))
 class ProcPath:
  def __new__(cls,p):return SimpleNamespace(read_text=lambda:"0 1")
 env=dict(ROOT=ROOT,need=need,os=osfake,subprocess=SimpleNamespace(Popen=popen,PIPE=1,STDOUT=2),Path=ProcPath,
  time=SimpleNamespace(monotonic=mono,time=mono,sleep=lambda s:ticks.__setitem__(0,ticks[0]+s)),
  signal=SimpleNamespace(SIGTERM=15,SIGKILL=9),CANCEL=[None])
 r=function(B,"run_group",env)([["RAM","a"],["RAM","b"]],owned=owned,deadline_epoch=185,rss_cap=10**7)
 if case=="success":
  check("actual_monitor_two_roots_true_wait_and_EOF_with_original_OwnedTree",r["success"]and waits==[100,101]and len(r["roots"][0]["ownership"]["owned_identities"])==2 and r["UI_tool_terminal_claimed"]is False)
 else:check("actual_monitor_failstop_"+case,r["success"]is False)
 check("monitor_numeric_groups_signal0_only_"+case,all(s in(15,9)for pid,s in signals))
for case in("success","nonzero","unknown","stdout_overflow","no_eof","group_present","adopted_nonzero"):monitor(case)
# Full actual batch controller across all18 parents/9waves with RAM children receipts.
def controller(fail_wave=None):
 f=fixture();events=[];saved={};adminown=dict(key="RAMadmin",reserved_bytes=33554432)
 inp=dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1",authority_ref=f.ar,source_manifest_ref=actualref(M),storage=dict(reservation=adminown))
 ir=dict(path="RAMbatchinput",sha256="0"*64);wave=[0];admitted=[]
 class Dir:
  def is_dir(self):return True
  def iterdir(self):return[SimpleNamespace(name="accounting_coordinator.lock")]
 def put(ctx,n,v):saved[n]=copy.deepcopy(v);return dict(path=n,sha256="0"*64)
 def group(ctx,commands,seconds,rss,name):
  if ".admission_"in name:
   parent=name.split(".")[0];events.append("admit:"+parent);admitted.append(parent)
   return dict(roots=[dict(stdout=json.dumps(dict(execution_started=False,input_ref=dict(path=parent,sha256="0"*64))))])
  if ".simulation_"in name:
   events.append("simulate:"+str(wave[0]));wave[0]+=1
   need(len(commands)==2)
   if wave[0]==fail_wave:raise ValueError("RAM child failure")
   return dict(roots=[dict(stdout="{}")for cmd in commands])
  parent=name.split(".")[0];events.append("close:"+parent)
  return dict(roots=[dict(stdout="{}")])
 env=dict(ROOT=ROOT,MANIFEST=M,WORKER=W,CLOSE=C,need=need,read=lambda rr:inp if rr==ir else f.repo.read(rr),
  qualified=lambda rr:(SimpleNamespace(accounting_transaction=lambda rr:contextlib.nullcontext(),ALLOWED_PARENTS=tuple(parents)),f.a,m),
  guard=lambda mm:SimpleNamespace(budget_check_v9=lambda *a,**k:None),
  state=lambda g,mm:({adminown["key"]:adminown},set(),dict(limit_bytes=10**12,projected_bytes=0)),
  local=lambda p:Dir(),put=put,original_owned=lambda mm:object(),SimpleNamespace=SimpleNamespace,
  signal=SimpleNamespace(SIGINT=2,SIGTERM=15,signal=lambda *a:None),CANCEL=[None],ref=actualref,
  wait_wave_capacity=lambda ctx,count:events.append("capacity"),available_memory=lambda:10**12,unit_request=lambda *a:{},
  group_call=group,argv=lambda module,flag,rr:[module,flag,rr["path"]],terminal=lambda r:json.loads(r["stdout"]),
  prepare_close=lambda ctx,parent,ir,sup:dict(path=parent,sha256="0"*64),
  accept_close=lambda ctx,parent,rr:(events.append("accept:"+parent)or dict(path=parent,sha256="0"*64)))
 with redirect_stdout(io.StringIO()):r=function(B,"run_batch",env)(ir)
 if fail_wave:
  check("actual_controller_failure_stops_no_next_wave_or_close",r["returncode"]==1 and len(admitted)==2 and not any(x.startswith("close:")for x in events)and "failed_preserved.json"in saved)
 else:
  check("actual_controller_all18_exact9paired_waves",r["returncode"]==0 and admitted==parents and wave[0]==9 and len(saved["completed.json"]["completed"])==18)
  check("actual_controller_closes_both_before_next_admission",all(events.index("accept:"+parents[i-1])<events.index("admit:"+parents[i])for i in range(2,18,2)))
  check("actual_controller_admin_close_external_terminal_gate",saved["completed.json"]["batch_admin_close_requires_ROOT_actual_final_terminal"]is True and saved["completed.json"]["UI_tool_terminal_claimed"]is False)
controller();controller(1)
for p in(W,B,C):
 e=dict(argparse=argparse,json=json,__doc__="RAM",initialize=lambda:need(False,"unexpected production init"),qualified=lambda *a:need(False,"unexpected production qualify"))
 with redirect_stdout(io.StringIO())as out:
  r=function(p,"main",e)([])
 check("default_inactive_"+Path(p).name,r==0 and json.loads(out.getvalue())["status"]=="SOURCE_ONLY_INACTIVE")
# Immutable recipe correction is exact replacement of two dotted module names only.
check("recipe_correction_manifest_exact",actualref(CORR+"/correction_manifest.v1.json")["sha256"]=="cc86df8cab61908b8af43bf6b9f10bb9430d863cd16a6fd2f67b586646cca1f7")
corrected=(ROOT/CORR/"ROOT_RECIPE.correction.v1.md").read_text()
check("corrected_recipe_exact",actualref(CORR+"/ROOT_RECIPE.correction.v1.md")["sha256"]=="66a613e0e24a64bff90b0635c64e7730e5b3e612aa494fd809992d7180f6b212")
check("corrected_recipe_both_actual_v2_commands",corrected.count("-m "+REL.replace("/",".")+".batch_ROOT_v1")==2 and "-m "+REL.replace("_v2","_v1").replace("/",".")+".batch_ROOT_v1"not in corrected)
check("ROOT_manual_mutation_shared_coordinator_contract","EVERY manual original accounting capture+authority+apply"in corrected and "ROOT must exclusively serialize the initial32MiB admin admission"in corrected)
print(json.dumps(dict(schema="TRAIN002_019_independent_exact_code_AST_RAM_checks_v2",status="INDEPENDENT_PASS",count=len(checks),checks=checks,
 source_manifest_ref=actualref(M),reviewed_sources=[actualref(p)for p in(W,B,C)],recipe_correction_ref=actualref(CORR+"/correction_manifest.v1.json"),
 actual_guard_imports_configurations_probes_admissions_launches_closes=0,scientific_payload_truth_checkpoint_reads_or_hashes=0,actual_model_or_optimizer_execution=False,
 semantics="Source-only independent code-object/RAM review; no numerical, current capacity, science, terminal, close or performance claim"),sort_keys=True,indent=2))

"""Independent owner4 service AST/RAM boundary tests. Never imports/calls real OS APIs."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,stat,struct
CANDIDATE="research/reviews/native_saved_MLM_successor_controller_ready_20261017_v4/empty_gpu0_controller_v1.py"
ORIGINAL="research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/empty_gpu0_controller_v1.py"
checks=[]
def need(ok,msg):
 if not ok:raise ValueError(msg)
def check(ok,msg):
 if not ok:raise AssertionError(msg)
 checks.append(msg)
def extract(path,name):
 tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
 return ast.Module(body=[node],type_ignores=[])
def rr(path):return dict(path=path,sha256=hashlib.sha256(path.encode()).hexdigest())
# Source-level executed protocol identity is anchored to original v3 code.
orig_ast=ast.parse(Path(ORIGINAL).read_text())
parent_old=Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/owner_parent_v1.py").read_bytes()
parent_new=Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v4/owner_parent_v1.py").read_bytes()
check(parent_old==parent_new,"Owner parent lifecycle byte-identical to originalv3")
source=Path(CANDIDATE).read_text();tree=ast.parse(source)
serve=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="serve_v1")
loop=next(n for n in serve.body if isinstance(n,ast.Try))
while_node=next(n for n in loop.body if isinstance(n,ast.While))
calls=[ast.unparse(n.func)for n in ast.walk(while_node)if isinstance(n,ast.Call)]
check(not any("validate_cap" in n or"guard_v1"in n or"atomic_registry"in n or"budget"in n or"inventory"in n for n in calls),
    "No source/global-accounting/registry/inventory action in steady while")
check(sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="atomic_registry"for n in ast.walk(serve))==3,
    "Only startup/final/error original writer callsites")
check('p._original_validate_cap_v3=p.validate_cap_v1'in source and 'p.validate_cap_v1=validate_cap_v1;p.serve_v1=serve_v1'in source,
    "Pinned v3 protocol and cap validator retained with explicit adapter hooks")
check('dispatch_v1'not in[n.name for n in tree.body if isinstance(n,ast.FunctionDef)],"No copied or altered wire dispatch implementation")
check("deadline=min(time.monotonic()+2."in source and "deadline-time.monotonic()"in source,"One aggregate2s peer deadline source")
check("periodic_registry_writes=False"in source and"periodic_accounting_transactions=False"in source,"Status declares fresh service without periodictransactions")
# Every fake below is in RAM and owned by this independent reviewer.
class FakeTime:
 def __init__(self,step=.001):self.mono=0.;self.step=step
 def time(self):return 100.
 def monotonic(self):self.mono+=self.step;return self.mono
class FakePath:
 def __init__(self,env,name):self.env=env;self.name=str(name)
 def __truediv__(self,part):return FakePath(self.env,self.name+"/"+part)
 def __str__(self):return self.name
 def exists(self):return self.name in self.env.existing
 @property
 def parent(self):return FakePath(self.env,self.name.rsplit("/",1)[0])
 def is_dir(self):return True
 def mkdir(self,**kw):self.env.existing.add(self.name);self.env.events.append(("mkdir",self.name))
 def lstat(self):return self.env.lease_stat()
 def read_text(self):return "999"if self.env.case=="child"else""
class Client:
 def __init__(self,env,request):
  self.env=env;self.blocks=[request]if env.case!="trickle"else[b"x"]*30;self.sent=[];self.timeouts=[]
 def __enter__(self):return self
 def __exit__(self,*a):return None
 def settimeout(self,n):self.timeouts.append(n)
 def getsockopt(self,*a):
  return struct.pack("3i",202,1001 if self.env.case=="peer_uid"else 1000,1000)
 def recv(self,n):return self.blocks.pop(0)if self.blocks else b""
 def sendall(self,packet):self.sent.append(json.loads(packet));self.env.answers.append(json.loads(packet))
class Listener:
 def __init__(self,env):self.env=env
 def bind(self,path):self.env.events.append(("bind",path))
 def listen(self,n):pass
 def setblocking(self,b):pass
 def accept(self):
  request=self.env.requests.pop(0);return Client(self.env,request),None
 def close(self):self.env.events.append(("listener_closed",))
class Environment:
 def __init__(self,case="normal"):
  self.case=case;self.events=[];self.answers=[];self.existing=set(["outparent"]);self.fdroles={};self.nextfd=10;self.held=False
  self.time=FakeTime(.3 if case=="trickle"else .001);self.lease=b"original-lease";self.peer_calls=0;self.select_calls=0
  self.requests=[b'{"v":1,"op":"status"}\n',b'{"v":1,"op":"shutdown","terminate":false}\n']if case=="normal"else[b'{"v":1,"op":"status"}\n']
  self.handlers={}
 def lease_stat(self):
  return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_dev=1,st_ino=88,st_size=len(self.lease))
 def open(self,path,flags,*a):
  fd=self.nextfd;self.nextfd+=1;self.fdroles[fd]="scheduler"if"scheduler"in str(path)else"lease"
  self.events.append(("open",self.fdroles[fd],fd));return fd
 def close(self,fd):self.events.append(("close",fd))
 def flock(self,fd,flags):
  if self.fdroles[fd]=="scheduler":return
  if not self.held:self.held=True;self.ownerfd=fd;return
  if self.case=="exclusion":return
  raise BlockingIOError()
 def proc(self,pid):
  if pid==101:return dict(pid=101,uid=1000,start_ticks=55,state="S")
  self.peer_calls+=1
  return dict(pid=pid,uid=1000,start_ticks=78 if self.case=="peer_reuse"and self.peer_calls>1 else 77,state="S")
 def select(self,*a):
  self.select_calls+=1
  if self.case=="lease_bytes"and self.select_calls==1:self.lease=b"tampered"
  if self.requests:return[a[0][0]],[],[]
  self.handlers[15](15,None);return[],[],[]
 def registry(self,path,value,g,storage):
  self.events.append(("guarded_registry",value["lifecycle"],value["updated_unix"]));self.last_registry=copy.deepcopy(value)
 def run(self):
  record=rr("ram/capability");cap=dict(pool="p",output_directory="outparent/out",created_at_utc="1970-01-01T00:00:00+00:00",
   expires_at_utc="1970-01-01T00:16:40+00:00",expires_unix=1000.,ROOT_downstream_release_path="ram/release")
  h=dict(created_at_utc=cap["created_at_utc"],expires_at_utc=cap["expires_at_utc"],
   old_physical_lease=dict(device=1,inode=88,sha256=hashlib.sha256(self.lease).hexdigest()))
  cert=dict(observed_at_utc="1970-01-01T00:00:50+00:00",known_identities=[])
  release=dict(status="active",root_authorizer="root_agent",execution_authorized=True,template_only=False,
   created_at_utc=cap["created_at_utc"],expires_at_utc=cap["expires_at_utc"],schema="ROOT_empty_GPU0_owner_downstream_release_v1",
   controller_capability_ref=record,all_authorized_borrowers_actually_drained=True,all_future_borrower_routes_revoked=True,
   genuine_wait_and_outer_refs=[rr("ram/proof")])
  def pread(fd,n,off):return self.lease
  def truncate(fd,n):self.lease=self.lease[:n]+b"\0"*max(0,n-len(self.lease))
  def pwrite(fd,data,off):self.lease=data;return len(data)
  fake_os=SimpleNamespace(read=lambda fd,n:b"G",close=self.close,environ={"CUDA_VISIBLE_DEVICES":""},
   getpid=lambda:101,getuid=lambda:1000,getpgrp=lambda:101,getsid=lambda p:101,
   open=self.open,fstat=lambda fd:self.lease_stat(),pread=pread,ftruncate=truncate,pwrite=pwrite,fsync=lambda fd:None,
   O_RDWR=2,O_NOFOLLOW=0x20000)
  fake_signal=SimpleNamespace(SIGTERM=15,SIGINT=2,signal=lambda n,f:self.handlers.setdefault(n,f))
  # fakesignal restore is allowed in RAM and never changes process signal handlers.
  ns=dict(_protocol_v4=lambda:None,need=need,os=fake_os,validate_cap_v1=lambda r:(cap,h,cert,{}),
   guard_v1=lambda *a:SimpleNamespace(budget_check_v9=lambda *a,**k:self.events.append(("startup_guard",))),
   sys=SimpleNamespace(modules={}),stamp=lambda s:__import__("datetime").datetime.fromisoformat(s).timestamp(),
   time=self.time,absent=lambda r:self.events.append(("targeted_absent",)),proc_identity=self.proc,
   local=lambda p:FakePath(self,p),Path=lambda p:FakePath(self,p),resource=SimpleNamespace(RUSAGE_SELF=0,
    getrusage=lambda r:SimpleNamespace(ru_maxrss=(3*2**30//1024)if self.case=="rss"else 1024)),
   stat=stat,hashlib=hashlib,fcntl=SimpleNamespace(flock=self.flock,LOCK_EX=2,LOCK_NB=4),signal=fake_signal,
   inventory=lambda c:self.events.append(("startup_inventory",)),encoded=lambda v:json.dumps(v,sort_keys=True).encode(),
   UUID="GPU0",socket_name=lambda c:"ram-socket",socket=SimpleNamespace(AF_UNIX=1,SOCK_STREAM=1,SOL_SOCKET=1,SO_PEERCRED=1,
    socket=lambda *a:Listener(self)),atomic_registry=self.registry,select=SimpleNamespace(select=self.select),struct=struct,
   json=json,read=lambda r:release,ref=lambda p:rr(p),raw=lambda r:self.events.append(("proof",r["path"])))
  # Original v3 dispatcher and active() run unchanged over the same RAM boundaries.
  for name in["active","dispatch_v1"]:
   exec(compile(extract(ORIGINAL,name),ORIGINAL+"#independent-RAM","exec"),ns)
  exec(compile(extract(CANDIDATE,"serve_v1"),CANDIDATE+"#independent-RAM","exec"),ns)
  try:self.returncode=ns["serve_v1"](record,9);self.error=None
  except Exception as e:self.returncode=None;self.error=type(e).__name__+": "+str(e)
  return self
normal=Environment().run()
check(normal.returncode==0 and normal.error is None,"Exact service+original dispatch RAM normal lifecycle")
check(len(normal.answers)==2 and all(v["ok"]for v in normal.answers),"Status and ROOT shutdown responses")
status_answer=normal.answers[0]["result"]
check(status_answer["lifecycle"]=="running"and status_answer["updated_unix"]==100.,"Fresh running status response")
check(status_answer["jobs"]==[]and status_answer["physical_gpu"]==0 and status_answer["empty_owner_only"]is True,"No scientific submission/child route in response")
check([e[1]for e in normal.events if e[0]=="guarded_registry"]==["running","closed"],"Exactly startup/final guarded registry writes, zero periodic writes")
check(len([e for e in normal.events if e[0]=="startup_inventory"])==1,"Only original startup inventory")
check(status_answer["lease_sha256"]==hashlib.sha256(json.dumps(status_answer["lease_metadata"],sort_keys=True).encode()).hexdigest(),
 "Status exact lease bytes/hash generation")
for case in["peer_uid","peer_reuse","trickle"]:
 env=Environment(case).run()
 check(env.returncode==0 and env.error is None,"Malformed client does not kill authenticated owner "+case)
 check(not env.answers or all(v["ok"]is False for v in env.answers),"No successful reply to invalid peer/request "+case)
 check([e[1]for e in env.events if e[0]=="guarded_registry"]==["running","closed"],"Client rejection no periodic registry "+case)
for case in["exclusion","lease_bytes","child","rss"]:
 env=Environment(case).run()
 check(env.returncode is None and env.error is not None,"Fail closed on local ownership/lease condition "+case)
 check(env.last_registry["lifecycle"]=="failed","Original guarded failed registry after "+case)
check(Environment("trickle").run().time.mono<10.,"Aggregate request deadline does not grow per-byte")
# Independently exercise new cap proof join, keeping original validator as explicit RAM predecessor boundary.
constants={}
for n in tree.body:
 if isinstance(n,ast.Assign):
  for target in n.targets:
   if isinstance(target,ast.Name)and target.id in["PROTOCOL_REF","PREDECESSOR_CAP","PREDECESSOR_ID","FAILED_QUALIFICATION"]:
    assignment=ast.Module(body=[n],type_ignores=[])
    exec(compile(assignment,CANDIDATE+"#static-constants","exec"),constants)
cap=dict(source_refs={constants["PROTOCOL_REF"]["path"]:constants["PROTOCOL_REF"]})
oldid=dict(pid=1,start_ticks=1,uid=1000)
h=dict(predecessor_owner_capability_ref=constants["PREDECESSOR_CAP"],predecessor_owner_identity=constants["PREDECESSOR_ID"],
 predecessor_termination_outcome_preserved=True,predecessor_actual_outer_exit_code=1,
 failed_TreeLoom_qualification_input_ref=constants["FAILED_QUALIFICATION"],failed_TreeLoom_qualification_worker_started=False,
 failed_TreeLoom_qualification_CUDA_context_created=False,failed_TreeLoom_qualification_namespace_replay_authorized=False,
 predecessor_owner_actual_wait_ref=rr("wait"),predecessor_owner_actual_outer_ref=rr("outer"),failed_TreeLoom_qualification_actual_outer_ref=rr("qualouter"))
wait=dict(controller_capability_ref=constants["PREDECESSOR_CAP"],controller_pid=constants["PREDECESSOR_ID"]["pid"],
 controller_start_ticks=constants["PREDECESSOR_ID"]["start_ticks"],actual_raw_wait=9,actual_direct_returncode=-9,
 all_owned_processes_drained=True,pipe_eof=True,cleanup_errors=[])
cert=dict(known_identities=[oldid,constants["PREDECESSOR_ID"]]);ram={"wait":wait,
 constants["PREDECESSOR_CAP"]["path"]:dict(handover_ref=rr("old_h")),"old_h":dict(known_root_absence_ref=rr("old_cert")),
 "old_cert":dict(known_identities=[oldid])}
protocol=SimpleNamespace(_original_validate_cap_v3=lambda *a,**k:(cap,h,cert,{}),need=need,raw=lambda r:None,read=lambda r:copy.deepcopy(ram[r["path"]]))
ns=dict(constants,_protocol_v4=lambda:protocol)
exec(compile(extract(CANDIDATE,"validate_cap_v1"),CANDIDATE+"#independent-cap-RAM","exec"),ns)
check(ns["validate_cap_v1"](rr("cap"))[0]==cap,"Exact original validator result plus new turnover joins")
for label,location,key,value in[
 ("forged graceful0",wait,"actual_direct_returncode",0),("wrongwait",wait,"actual_raw_wait",0),
 ("undrained",wait,"all_owned_processes_drained",False),("qualification replay",h,"failed_TreeLoom_qualification_namespace_replay_authorized",True),
 ("qualification GPUclaim",h,"failed_TreeLoom_qualification_CUDA_context_created",True),
 ("wrong predecessor",h,"predecessor_owner_identity",oldid)]:
 old=copy.deepcopy(location[key]);location[key]=value
 try:ns["validate_cap_v1"](rr("cap"))
 except(ValueError,KeyError):checks.append("Reject "+label)
 else:raise AssertionError("Accepted "+label)
 location[key]=old
old=cap["source_refs"];cap["source_refs"]={}
try:ns["validate_cap_v1"](rr("cap"))
except ValueError:checks.append("Reject missing unchanged v3 source pin")
else:raise AssertionError("Missing v3 pin accepted")
cap["source_refs"]=old
old=cert["known_identities"];cert["known_identities"]=[constants["PREDECESSOR_ID"]]
try:ns["validate_cap_v1"](rr("cap"))
except ValueError:checks.append("Reject dropped older known-root identity")
else:raise AssertionError("Dropped old root accepted")
cert["known_identities"]=old

# Concrete admission source structure retains one coordinator transaction around original capture/apply.
admission=Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v4/owner4_admission_ROOT_v1.py").read_text()
adast=ast.parse(admission);prepare=next(n for n in adast.body if isinstance(n,ast.FunctionDef)and n.name=="prepare_v4")
coordinator_with=next(n for n in ast.walk(prepare)if isinstance(n,ast.With)and any("accounting_transaction"in ast.unparse(x.context_expr)for x in n.items))
mutex_with=next(n for n in coordinator_with.body if isinstance(n,ast.With))
apply_calls=[n for n in ast.walk(coordinator_with)if isinstance(n,ast.Call)and ast.unparse(n.func)=="g.apply_reserve_v1"]
check(len(apply_calls)==1,"One original reserve apply in same coordinator transaction")
check(not any(isinstance(n,ast.Call)and ast.unparse(n.func)=="g.apply_reserve_v1"for n in ast.walk(mutex_with)),"Original apply outside original scheduler mutex")
check(any(isinstance(n,ast.Call)and ast.unparse(n.func)=="g.operation_draft_v1"for n in ast.walk(mutex_with)),"Original draft/capture inside unchanged mutex")
check('len(g.budget.v4.encoded(event)+b"\\n")<=65536'in admission,"Original64KiB event ceiling enforced before apply")
check("ADMIN_CAP+PARENT_CAP+65536+4096<=524288"in admission,"Bounded admin+parent records inside original512KiB event allowance")
check("no_automatic_launch=True"in admission and"genuine_builder_outer0_required_before_single_parent_launch=True"in admission,
 "Admission never launches owner and requires real builder terminal first")
check("if not entered"in admission and"READY_NO_ADMISSION_COORDINATOR_TIMEOUT"in admission,"Only before-yield coordinator timeout classified notadmitted")
check("receipt"in admission and"controller_storage_ref=storage_ref"in admission,"Capability binds genuine original reserve receipt storage")
# Exact manifest authentication is read-only source/metadata, with no dynamic imported guard.
manifest_path=Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v4/source_manifest.v1.json")
manifest=json.loads(manifest_path.read_text())
expected_manifest_sha="a9ec2e39c97da9b95c4331c9206880e02df0d098fbdaf58d4be12685bd344ed7"
check(hashlib.sha256(manifest_path.read_bytes()).hexdigest()==expected_manifest_sha,"Exact sealed owner4 manifest")
for path,sha in manifest["source_sha256"].items():
 p=Path(path);data=p.read_bytes()
 check(len(data)<=32*2**20 and hashlib.sha256(data).hexdigest()==sha,"Exact sealed source/metadata pin "+path)
check(len(manifest["source_sha256"])==247,"All247transitive source/metadata pins authenticated")
# Real inert adapter-loader joins execute actual original source definitions, no service/guard/probe.
import importlib.util,sys,contextlib,io
spec=importlib.util.spec_from_file_location("_independent_owner4_real_inert_adapter",CANDIDATE)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
with contextlib.redirect_stdout(io.StringIO())as captured:code=module.main([])
check(code==0 and json.loads(captured.getvalue())["operations"]==0,"Actual adapter default is inert")
protocol=module._protocol_v4()
check(protocol.SOURCE==module.SOURCE and protocol.MANIFEST==module.MANIFEST,"Actual original protocol globals point to reviewedv4 namespace")
check(protocol.dispatch_v1.__code__.co_filename.endswith(ORIGINAL),"Actual dispatcher executes pinned originalv3 source")
check(protocol.atomic_registry.__code__.co_filename.endswith(ORIGINAL),"Actual guarded atomic writer executes pinned originalv3 source")
check(protocol._original_validate_cap_v3.__code__.co_filename.endswith(ORIGINAL),"Actual original cap validator retained")
check(protocol.validate_cap_v1 is module.validate_cap_v1 and protocol.serve_v1 is module.serve_v1,"Actual explicit adapter hooks joined without recursion")
check(not any(str(getattr(m,"__file__","")).endswith("census_observation_retention_v1.py")for m in list(sys.modules.values())),
 "No actual current guard imported during independent review")


# Frozen downstream binding accepts the original protocol pin plus the reviewed new endpoint.
borrow_path="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/borrowed_gpu0_v1.py"
borrow_text=Path(borrow_path).read_text()
check("RPC_SECONDS=30"in borrow_text.replace(" ",""),"Original30s borrowed client deadline unchanged")
check('ended-answer["updated_unix"]<=60'in borrow_text.replace(" ",""),"Original60s response freshness unchanged")
bindingref=rr("ram/binding");ownerref=rr("ram/owner4cap")
binding=dict(schema="native_VAL000_downstream_GPU0_controller_binding_v1",source_review_passed=True,
 endpoint_source_ref=constants["PROTOCOL_REF"],physical_gpu=0,GPU_UUID="GPU0",
 additional_controller_and_other_borrower_submissions_revoked=True,controller_preserved_until_downstream_drain=True,
 independent_source_review_ref=rr("ram/review"),route="reviewed_same_protocol_successor",protected_job=None,protected_envelope_bytes=0,
 successor_capability_explicitly_ROOT_authorized=True,controller_pid=101,controller_start_ticks=55,controller_capability_ref=ownerref)
owner=dict(source_refs={constants["PROTOCOL_REF"]["path"]:constants["PROTOCOL_REF"],CANDIDATE:rr(CANDIDATE)})
binding_ram={bindingref["path"]:binding,ownerref["path"]:owner}
borrow=dict(require=need,BINDING_REF=None,read=lambda r:binding_ram[r["path"]],active=lambda v:None,
 ENDPOINT_SOURCE_REF=constants["PROTOCOL_REF"],GPU0_UUID="GPU0",raw=lambda r:None)
exec(compile(extract(borrow_path,"configure_controller_binding_v1"),borrow_path+"#independent-RAM","exec"),borrow)
check(borrow["configure_controller_binding_v1"](dict(controller_binding_ref=bindingref))==binding,
 "Unchanged frozen borrower accepts owner4cap with dual authenticated protocol/source pins")
check(borrow["CONTROLLER_PID"]==101 and borrow["C3_JOB"]is None,"Frozen borrower binds exact new owner without scientific job declaration")
science_manifest=json.loads(Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/source_manifest.v1.json").read_text())
check(len(science_manifest["executed_new_source_refs"])==9,"Original nine scientific modules remain exactly nine")
for r in science_manifest["executed_new_source_refs"]:
 check(hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()==r["sha256"],"Original frozen scientific module unchanged "+r["path"])
check(hashlib.sha256(Path(science_manifest["source_configuration_ref"]["path"]).read_bytes()).hexdigest()==science_manifest["source_configuration_ref"]["sha256"],
 "Original frozen scientific configuration unchanged")

result=dict(schema="Owner4_status_service_independent_AST_RAM_checks_v1",passed=True,check_count=len(checks),checks=checks,
 source_AST_RAM_only=True,author_fixture_reused=False,actual_guard_imported=False,actual_socket_process_lock_GPU_probes=False,
 actual_accounting=False,actual_owner_service_or_scientific_execution=False)
print(json.dumps(result,sort_keys=True,indent=2))

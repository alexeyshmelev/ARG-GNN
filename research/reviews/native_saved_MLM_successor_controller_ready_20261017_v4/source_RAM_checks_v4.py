"""Owner4 source/metadata and isolated hot-status/peer/OFD RAM checks only."""
import ast,copy,hashlib,json,math,struct,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import stat
HERE=Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v4")
def main():
 checks=[];assertions=[0]
 def check(ok,name):
  assertions[0]+=1
  if not ok:raise AssertionError(name)
  if name not in checks:checks.append(name)
 def need(ok,message):
  if not ok:raise ValueError(message)
 def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 source=(HERE/"empty_gpu0_controller_v1.py").read_text();tree=ast.parse(source)
 manifest=json.loads((HERE/"source_manifest.v1.json").read_text())
 for path,sha in manifest["source_sha256"].items():
  check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,"all247-source-and-metadata-pins-authenticated")
 check((HERE/"owner_parent_v1.py").read_bytes()==Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/owner_parent_v1.py").read_bytes(),"parent-byte-identical-pidfd-lifecycle")
 serve=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="serve_v1")
 loop=next(n for n in serve.body if isinstance(n,ast.Try)).body
 while_node=next(n for n in ast.walk(serve)if isinstance(n,ast.While)and "stopped"in ast.unparse(n.test))
 forbidden={"validate_cap_v1","guard_v1","atomic_registry","load","g.budget_check_v9","g.validate_storage_v9"}
 check(not any(isinstance(n,ast.Call)and ast.unparse(n.func)in forbidden for n in ast.walk(while_node)),"no-periodic-capwalk-registry-or-global-budget-in-service-loop")
 atomics=[n for n in ast.walk(serve)if isinstance(n,ast.Call)and ast.unparse(n.func)=="atomic_registry"]
 check(len(atomics)==3 and not any(n in list(ast.walk(while_node))for n in atomics),"exact-initial-normal-terminal-error-guarded-writes-only")
 check(ast.unparse(serve).count("absent(cert['known_identities'])")==2,"both-original-startup-known-root-absence-checks")
 hot=next(n for n in serve.body if isinstance(n,ast.FunctionDef)and n.name=="hot_check")
 status=next(n for n in serve.body if isinstance(n,ast.FunctionDef)and n.name=="status")
 check(not any(isinstance(n,ast.Call)and ast.unparse(n.func)in forbidden|{"raw","read","ref"}for n in ast.walk(hot)),"hot-clock-identity-lease-invariants-without-source-or-budget-access")
 check("hot_check()"in ast.unparse(status)and "updated_unix=time.time()"in ast.unparse(status),"fresh-status-checked-immediately-before-real-response")
 check("LOCK_EX | fcntl.LOCK_NB"in ast.unparse(hot),"independent-writable-OFD-exclusion-every-loop")
 f=dict(time=1000.,mono=0.,uid=1000,start=456,children="",rss=4096,ino=22,data=b"original lease",excluded=True,peer_reused=False)
 opened=lambda:SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_dev=11,st_ino=22,st_size=len(b"original lease"))
 class Lease:
  def lstat(self):
   s=opened();s.st_ino=f["ino"];return s
  def __str__(self):return "/ROOT/.runtime/locks/gpu_0.lock"
 class ProcText:
  def read_text(self):return f["children"]
 class OS:
  O_RDWR=2;O_NOFOLLOW=131072
  def getpid(self):return 123
  def getuid(self):return 1000
  def getpgrp(self):return 123
  def getsid(self,pid):return 123
  def fstat(self,fd):return opened()
  def pread(self,fd,size,offset):return f["data"]
  def open(self,path,flags):return 8
  def close(self,fd):check(fd==8,"probe-closes-only-independent-FD")
 def flock(fd,mode):
  if f["excluded"]:raise BlockingIOError()
 def proc(pid):
  return dict(pid=pid,uid=f["uid"],start_ticks=f["start"],state="S")
 env=dict(need=need,authority_start=900.,authority_end=9000.,identity=dict(pid=123,uid=1000,start_ticks=456,state="S"),
  os=OS(),stat=stat,hashlib=hashlib,lease=7,lease_path=Lease(),lease_sha=hashlib.sha256(b"original lease").hexdigest(),
  Path=lambda p:ProcText(),proc_identity=proc,resource=SimpleNamespace(RUSAGE_SELF=0,
   getrusage=lambda kind:SimpleNamespace(ru_maxrss=f["rss"])),
  time=SimpleNamespace(time=lambda:f["time"],monotonic=lambda:f["mono"]),
  fcntl=SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=flock),
  cap=dict(pool="owner4",created_at_utc=datetime.fromtimestamp(900,timezone.utc).isoformat()),
  record=dict(path="RAM/cap.json",sha256="1"*64),lifecycle="running",safety=None,UUID="GPU0",
  socket_name=lambda cap:"\0owner4",lease_meta=dict(gpu=0),MAX_JSON=65536)
 for n in(hot,status):exec(compile(ast.Module(body=[n],type_ignores=[]),str(HERE/"empty_gpu0_controller_v1.py"),"exec"),env)
 env["hot_check"]();check(True,"positive-full-hot-invariants")
 def bad(field,value,name):
  before=f[field];f[field]=value
  try:env["hot_check"]()
  except(ValueError,RuntimeError):check(True,name)
  else:raise AssertionError(name)
  finally:f[field]=before
 for field,value,name in[
  ("start",457,"owner-PID-reuse-rejected"),("uid",1001,"owner-UID-change-rejected"),
  ("children","777","no-owned-or-scientific-children"),("rss",2*2**30//1024+1,"2GiB-RSS-failstop"),
  ("ino",23,"lease-inode-replacement-rejected"),("data",b"changed","lease-bytes-change-rejected"),
  ("excluded",False,"lost-independent-exclusion-rejected"),("time",9000.,"work-deadline-no-extension"),
  ("time",899.,"cap-handover-start-no-backdate")]:bad(field,value,name)
 f["time"]=2000.;fresh=env["status"]()
 check(fresh["updated_unix"]==2000. and fresh["updated_unix"]-1000.>60
  and fresh["jobs"]==[]and fresh["owner_service_revision"]==4
  and fresh["periodic_registry_writes"]is False and fresh["periodic_accounting_transactions"]is False,
  "honest-aged-disk-snapshot-fresh-in-memory-RPC")
 # Execute unchanged original v3 dispatch and exact new client block in RAM.
 oldtree=ast.parse(Path(manifest["compatibility_protocol_source_ref"]["path"]).read_text())
 dispatch=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name=="dispatch_v1")
 env.update(active=lambda v:need(v.get("active")is True,"release-active"),raw=lambda r:b"RAM")
 exec(compile(ast.Module(body=[dispatch],type_ignores=[]),"v3_original_dispatch","exec"),env)
 original_dispatch=env["dispatch_v1"];dc=[0]
 def dispatch_count(*args):dc[0]+=1;return original_dispatch(*args)
 env["dispatch_v1"]=dispatch_count
 check(original_dispatch(dict(v=1,op="status"),env["record"],env["cap"],env["status"],None)[0]["updated_unix"]==2000.,
  "unchanged-v3-status-wire-protocol")
 try:original_dispatch(dict(v=1,op="submit"),env["record"],env["cap"],env["status"],None)
 except ValueError:check(True,"no-submit-worker-cancel-or-compute-route")
 else:raise AssertionError("submit")
 clientblock=next(n for n in ast.walk(serve)if isinstance(n,ast.With)and any(isinstance(i.context_expr,ast.Name)and i.context_expr.id=="client"for i in n.items))
 class Client:
  def __init__(self,trickle=False,uid=1000):self.trickle=trickle;self.uid=uid;self.timeout=2.;self.sent=None;self.reads=0
  def __enter__(self):return self
  def __exit__(self,*args):return False
  def settimeout(self,t):self.timeout=t
  def getsockopt(self,*args):return struct.pack("3i",888,self.uid,1000)
  def recv(self,n):
   self.reads+=1
   if self.trickle:
    if self.timeout<1.5:f["mono"]+=self.timeout;f["time"]+=self.timeout;raise TimeoutError("RAM socket timeout")
    f["mono"]+=1.5;f["time"]+=1.5;return b"{"
   return b'{"v":1,"op":"status"}\n'
  def sendall(self,data):self.sent=json.loads(data)
 peer_reads=[0]
 def peerproc(pid):
  if pid==123:return proc(pid)
  peer_reads[0]+=1
  return dict(pid=888,uid=1000,start_ticks=100+(1 if f["peer_reused"]and peer_reads[0]>1 else 0),state="S")
 env.update(proc_identity=peerproc,struct=struct,json=json,socket=SimpleNamespace(SOL_SOCKET=1,SO_PEERCRED=2),
  encoded=lambda v:json.dumps(v,sort_keys=True,separators=(",",":")).encode(),stopped=[False])
 code=compile(ast.Module(body=[clientblock],type_ignores=[]),"new_owner4_exact_client_block","exec")
 f["mono"]=0.;env["client"]=Client();exec(code,env)
 check(env["client"].sent["ok"]is True and env["client"].sent["result"]["updated_unix"]==f["time"],
  "fresh-same-peer-positive-client-response")
 before=dc[0];f["peer_reused"]=True;peer_reads[0]=0;env["client"]=Client();exec(code,env)
 check(env["client"].sent["ok"]is False and dc[0]==before,"peer-PID-start-reuse-fails-before-dispatch")
 f["peer_reused"]=False;peer_reads[0]=0;env["client"]=Client(uid=2000);exec(code,env)
 check(env["client"].sent["ok"]is False and dc[0]==before,"peer-UID-mismatch-fails-before-dispatch")
 f["mono"]=0.;peer_reads[0]=0;env["client"]=Client(trickle=True);exec(code,env)
 check(f["mono"]==2. and env["client"].sent is None and dc[0]==before,"aggregate2s-trickle-bound-no-dispatch-or-budget-work")
 # All old scientific source/config/module hashes remain exactly sealed.
 tm=Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/source_manifest.v1.json")
 frozen=json.loads(tm.read_text())
 check(hashlib.sha256(tm.read_bytes()).hexdigest()=="35760763a7f0608ed56cd62565ec555d4fcae7f622ea6bfb9dd5280257d20fa0","sealed-trainer-manifest-unchanged")
 for path,sha in frozen["source_sha256"].items():check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,"all85-sealed-scientific-source-metadata-pins-unchanged")
 # Concrete original53owner/64qualification union with real47role baseline.
 ar=Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_records_root01_oct17/reserve_authority.json")
 baseline=json.loads(ar.read_text())["bindings"]
 check(len(baseline)==47 and len({json.dumps(v,sort_keys=True)for v in baseline.values()})==40,"genuine47-role40-unique-baseline")
 gtree=ast.parse(Path("research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py").read_text())
 ge=dict(copy=copy,require=need,_raw=lambda r:None)
 fn=next(n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name=="consumer_bindings_v1")
 exec(compile(ast.Module(body=[fn],type_ignores=[]),"original_consumer_bindings","exec"),ge)
 synthetic=lambda p:dict(path=p,sha256=hashlib.sha256(p.encode()).hexdigest())
 owner_direct=[synthetic("RAM/owner_direct_%d.json"%i)for i in range(13)]
 qualification_direct=[synthetic("RAM/qualification_direct_%d.json"%i)for i in range(24)]
 check(len(ge["consumer_bindings_v1"](list(baseline.values()),owner_direct))==53,"owner53shape-original-binding-ceiling")
 check(len(ge["consumer_bindings_v1"](list(baseline.values()),qualification_direct))==64,"qualification64shape-original-binding-ceiling")
 check(not any(n in sys.modules for n in("torch","torch_geometric","_owner4_ROOT_current_Census_mutation")),"no-actual-guard-model-service-resource-or-accounting-operation")
 result=dict(schema="owner4_status_service_source_RAM_checks_v1",source_only=True,
  service_source_ref=ref(HERE/"empty_gpu0_controller_v1.py"),manifest_ref=ref(HERE/"source_manifest.v1.json"),
  checks_passed=len(checks),assertions_passed=assertions[0],checks=checks,
  actual_operations=0,actual_guard_import=False,actual_socket_process_lock_GPU_or_state_probe=False,
  actual_registry_or_accounting_write=False,all247_source_metadata_pins_authenticated=True,
  all85_scientific_pins_unchanged=True,original_parent_byte_identical=True,
  limits="Component/new-peer-block RAM checks and original merge shape only; independent whole-loop/cap/builder review and real ROOT startup/status/fit remain required.")
 print(json.dumps(dict(checks_passed=len(checks),assertions_passed=assertions[0]),sort_keys=True));return result
if __name__=="__main__":
 try:main()
 except BaseException:print(traceback.format_exc());raise

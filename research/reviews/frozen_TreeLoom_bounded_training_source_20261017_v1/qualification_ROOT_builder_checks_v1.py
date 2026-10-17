"""Builder source/compact-metadata and isolated full-admission-in-RAM checks only."""
import ast,copy,hashlib,importlib,json,sys,traceback
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
from pathlib import Path,PurePosixPath
from types import SimpleNamespace

def main():
 b=importlib.import_module("research.reviews.frozen_TreeLoom_bounded_training_source_20261017_v1.qualification_ROOT_builder_v1")
 c=b.c;realread=c.read;realref=c.ref;realraw=c.raw;reallocal=c.local
 checks=[]
 def check(ok,name):
  if not ok:raise AssertionError(name)
  checks.append(name)
 source=realraw(realref(b.BUILDER)).decode();tree=ast.parse(source)
 check(not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(x in ast.unparse(n)for x in("torch","Census","memo"))for n in tree.body),"stdlib-and-inert-contract-imports-only")
 prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="prepare_v1")
 text=ast.unparse(prepare)
 check(text.index("admin.mkdir")<text.index("c.load"),"attempt-consumed-before-real-guard-import")
 check("c.guard_v1("not in text and "install_consumer_v1("not in source,"mutation-never-installs-readonly-memo")
 check(text.index("scope_spec")<text.index("operation_draft_v1")<text.index("apply_reserve_v1"),"original-full-scope-before-once-apply")
 check(text.count("apply_reserve_v1(")==1,"single-apply-no-retry")
 check(not any(isinstance(n,ast.Call)and ast.unparse(n.func).split(".")[-1]in("Popen","run","system","proc_identity","socket","install_consumer_v1")for n in ast.walk(prepare)),"no-launch-process-socket-or-GPU-probe")
 template=realread(realref(c.HERE+"/ROOT_qualification_admission_request.INACTIVE.v1.json"))
 now=datetime.now(timezone.utc);q=copy.deepcopy(template)
 q.update(status="active",template_only=False,execution_authorized=True,created_at_utc=now.isoformat(),
  expires_at_utc=(now+timedelta(seconds=28860)).isoformat(),instance="ramcheck")
 request=dict(path="RAM/request.json",sha256="0"*64)
 c.read=lambda r:copy.deepcopy(q)if r==request else realread(r)
 actual=b._request(request)
 check(len(actual[3]["executed_new_source_refs"])==9,"real-all-nine-sealed-source-authentication")
 check(actual[4]["cpu_affinity"]==list(range(176,184)),"real-qualified-runtime-and-affinity-join")
 check(actual[5]["path"].endswith("/truth/c3_pulse_ancestry.npz"),"real-original-TRAIN000-handle-joined-without-payload-read")
 for field,value in [("runtime_ref",b.RUNTIME_QUALIFICATION),("controller_ready_ref",b.OWNER),("train_truth_handle_ref",b.OWNER),
  ("automatic_retry_or_resume",True),("source_review_ref",b.READY),("instance","../escape")]:
  bad=copy.deepcopy(q);bad[field]=value;c.read=lambda r:copy.deepcopy(bad)if r==request else realread(r)
  try:b._request(request)
  except (ValueError,KeyError,TypeError):check(True,"reject-request-"+field)
  else:raise AssertionError(field)
 c.read=lambda r:copy.deepcopy(q)if r==request else realread(r)
 c.active(template)if False else None
 try:c.active(template)
 except (ValueError,TypeError):check(True,"inactive-template-cannot-admit")
 else:raise AssertionError("inactive")
 owner=actual[1]
 for seconds in (True,300,28801):
  bad=dict(q,max_seconds=seconds)
  try:b._interval(bad,owner)
  except ValueError:check(True,"reject-invalid-window-"+str(seconds))
  else:raise AssertionError("seconds")
 bad=dict(q,expires_at_utc=datetime.fromtimestamp(owner["expires_unix"]+1,timezone.utc).isoformat())
 try:b._interval(bad,owner)
 except ValueError:check(True,"reject-owner-authority-overrun")
 else:raise AssertionError("owner-overrun")
 # Everything below is isolated RAM. No real guard import, state, mutex, reserve, file or process operation.
 def immutable_ref(p):
  actualp=reallocal(p)
  return dict(path=str(actualp.relative_to(c.ROOT)),sha256=hashlib.sha256(actualp.read_bytes()).hexdigest())
 def immutable_read(r):
  actualp=reallocal(r["path"]);data=actualp.read_bytes()
  if hashlib.sha256(data).hexdigest()!=r["sha256"]:raise ValueError("SHA")
  return json.loads(data)
 realref=immutable_ref;realread=immutable_read
 records={};trace=[];dirs=set();calls=dict(state=0,apply=0,scope=0,mutex=0)
 class P(PurePosixPath):
  def exists(self):return str(self)in dirs
  def mkdir(self,**kw):check(str(self)not in dirs,"exclusive-admin-mkdir");dirs.add(str(self));trace.append("mkdir")
  def is_file(self):return True
 def local(p):return P(str(c.ROOT/p)if not str(p).startswith("/")else str(p))
 def ref(p):
  path=str(p)
  if path.startswith(str(c.ROOT)+"/"):path=path[len(str(c.ROOT))+1:]
  for key,(r,value)in records.items():
   if r["path"]==path:return copy.deepcopy(r)
  return realref(path)
 def read(r):
  if r==request:return copy.deepcopy(q)
  if r["path"]in records:
   rr,value=records[r["path"]];check(rr==r,"RAM-immutable-ref-authentication");return copy.deepcopy(value)
  return realread(r)
 def save(directory,name,value):
  raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
  check(len(raw)<=65536,"bounded-record-"+name)
  path=str(directory/c.ROOT.name)if False else str(directory/name)
  path=path[len(str(c.ROOT))+1:];check(path not in records,"exclusive-record-"+name)
  r=dict(path=path,sha256=hashlib.sha256(raw).hexdigest());records[path]=(r,copy.deepcopy(value));trace.append("save:"+name)
  return r
 baseline={"role%02d"%i:dict(path="RAM/baseline_%02d.json"%(i%40),sha256=hashlib.sha256(str(i%40).encode()).hexdigest())for i in range(47)}
 baseline["role00"]=c.GUARD
 for i in range(40,47):baseline["role%02d"%i]=baseline["role%02d"%(i%40)]
 state=({},set(),dict(base_pins={"RAM":True},new_reserved_bytes=0,completed_credit_bytes=0),baseline,[],[],{})
 def raw(r):
  check(type(r)is dict and set(r)=={"path","sha256"}and not r["path"].endswith(".npz"),"source-control-only-binding")
  return b"RAM"
 def state_locked(*args,**kw):
  calls["state"]+=1;check(kw=={"fresh":True},"fresh-original-state");trace.append("state")
  return state
 @contextmanager
 def mutex(root):
  calls["mutex"]+=1;trace.append("mutex-enter")
  try:yield
  finally:trace.append("mutex-exit")
 v4=SimpleNamespace(__name__="RAM_original_v4")
 for f in [mutex]:f.__module__=v4.__name__
 def canonical(root,p):return P(p)
 canonical.__module__=v4.__name__
 def file_bytes(*a):return b"",None
 file_bytes.__module__=v4.__name__
 v4.mutex=mutex;v4.canonical=canonical;v4.file_bytes=file_bytes
 v4.encoded=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
 def extract(path,names,env):
  code=ast.parse(Path(path).read_text())
  for n in code.body:
   if isinstance(n,ast.FunctionDef)and n.name in names:exec(compile(ast.Module(body=[n],type_ignores=[]),path,"exec"),env)
  return env
 env=dict(Path=Path,canonical=canonical,paths_overlap=lambda a,d:str(a)==str(d)or str(a).startswith(str(d)+"/")or str(d).startswith(str(a)+"/"),
  accounting_paths=lambda root:[],bound=lambda root,r:raw(r),
  integer=lambda x,label,minimum:check(type(x)is int and x>=minimum,"original-integer-charge"))
 extract("src/arggnn/project_budget_v4.py",["scope_spec"],env)
 def scope(root,r):calls["scope"]+=1;trace.append("scope");return env["scope_spec"](root,r)
 v4.scope_spec=scope
 g=SimpleNamespace(MODE="Census_observation_two_scope_retention_v1",MAX_JSON=65536,META_CHARGE=524288,budget=SimpleNamespace(v4=v4))
 ge=dict(copy=copy,require=c.require,MODE=g.MODE,RETIRE_OPERATION="RAM_retire",_raw=raw)
 extract(c.GUARD["path"],["consumer_bindings_v1","operation_draft_v1"],ge)
 g.consumer_bindings_v1=ge["consumer_bindings_v1"];g.operation_draft_v1=ge["operation_draft_v1"]
 g._state_locked=state_locked;g._pins=lambda v:dict(base_pins=v["base_pins"],completed_overlay_prefix=dict(events=0,tail_sha256=None))
 g.load_selected_guard_v1=lambda r:g
 def apply(r,**kw):
  calls["apply"]+=1;trace.append("apply");auth=read(r)
  check(trace[-2]=="mutex-exit","capture-mutex-released-before-once-apply")
  check(auth["active"]is True and auth["request"]["reserved_bytes"]==64*2**20,"original64MiB-active-reserve")
  check(auth["expected_before"]==g._pins(state[2])and auth["bindings"]==baseline,"actual-full-role-and-prefix-binding")
  return dict(value=dict(auth["request"],root_admission=r),initial_ledger_pins=dict(RAM=True))
 g.apply_reserve_v1=apply
 original_validate=c.validate_input
 c.local=local;c.ref=ref;c.read=read;c.raw=raw;c.load=lambda r,name:(trace.append("guard-import")or g)
 b._request=lambda r:actual;b._save=save
 # Original contract.validate_input executes unchanged against authenticated RAM output/source controls.
 c.CURRENT_INPUT_REF=None
 result=b.prepare_v1(request)
 check(result["status"]=="QUALIFICATION_ADMITTED_ONLY"and result["automatic_launch"]is False,"full-RAM-admission-only-result")
 check(calls==dict(state=1,apply=1,scope=1,mutex=1),"one-fresh-state-full-scope-one-apply")
 check(trace.index("mkdir")<trace.index("guard-import")<trace.index("state")<trace.index("scope")<trace.index("apply"),"complete-one-shot-order")
 inputq=read(result["input_ref"]);stored=read(inputq["storage_ref"])
 check(inputq["phase"]=="qualification"and inputq["ROOT_record_directory"]==inputq["worker_auxiliary_scope"],"exact-disposable-scope-contract")
 check(inputq["train_truth_refs"]=={"american_c3_train_000":actual[5]},"exact-handle-pass-through-no-payload-read")
 check(stored["reservation"]["reserved_bytes"]==64*2**20 and len(stored["reservation"]["consumer_bindings"])==62,"47roles40unique-plus-direct62-within64")
 check(not local(inputq["output_directory"]).exists()and not local(inputq["worker_auxiliary_scope"]).exists(),"primary-and-AUX-stay-absent")
 check(result["execute_argv"][-2]=="--capability-ref"and json.loads(result["execute_argv"][-1])==result["input_ref"],"exact-parent-argv-returned-only")
 before=calls["apply"]
 try:b.prepare_v1(request)
 except ValueError:check(calls["apply"]==before,"consumed-attempt-cannot-retry")
 else:raise AssertionError("retry")
 try:g.consumer_bindings_v1(list(baseline.values()),stored["reservation"]["consumer_bindings"],
  [dict(path="RAM/overflow_%d.json"%i,sha256=str(i)*64)for i in range(3)])
 except ValueError:check(True,"original64union-ceiling-not-raised")
 else:raise AssertionError("overflow")
 check(not any(n in sys.modules for n in("torch","torch_geometric","_TreeLoom_ROOT_mutation_Census")),"no-actual-guard-or-science-import")
 resultrecord=dict(schema="ROOT_TreeLoom_qualification_builder_source_checks_v1",source_only=True,
  source_ref=realref(b.BUILDER),actual_admission=False,actual_launch=False,actual_guard_import=False,
  scientific_or_truth_payload_opened=False,checks_passed=len(set(checks)),assertions_passed=len(checks),checks=list(dict.fromkeys(checks)),
  full_admission_RAM=dict(baseline_roles=47,baseline_unique_refs=40,complete_bindings=62,
    original_scope_spec_executed_with_RAM_dependencies=True,one_reserve_apply_mock=True,
    original_contract_validate_input_executed_with_RAM_dependencies=True),
  limitations="ROOT still owns real current state/full original apply, outer0, launch, live owner/capacity, parity, whole-parent fit and waits.")
 print(json.dumps(dict(checks_passed=resultrecord["checks_passed"],assertions_passed=resultrecord["assertions_passed"],full_admission_RAM=resultrecord["full_admission_RAM"]),sort_keys=True))
 return resultrecord

if __name__=="__main__":
 try:main()
 except BaseException:print(traceback.format_exc());raise

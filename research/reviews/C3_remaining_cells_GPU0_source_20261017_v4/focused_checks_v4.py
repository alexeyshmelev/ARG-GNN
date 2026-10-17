"""Focused v4 source/RAM checks. Only actual CLI help and inert imports run."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
import ast,contextlib,copy,hashlib,importlib,io,json,os,subprocess,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/C3_remaining_cells_GPU0_source_20261017_v4"
OLD=ROOT/"research/reviews/C3_remaining_cells_GPU0_source_20261017_v3"
checks=[];negative=[];cli=[];docs={};docrefs={}
def need(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(fn,label,exc=(ValueError,KeyError,TypeError,AssertionError,FileExistsError,ChildProcessError,OSError)):
 try:fn()
 except exc:negative.append(label);return
 raise AssertionError("Accepted invalid: "+label)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def put(path,value):
 data=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
 r=dict(path=path,sha256=hashlib.sha256(data).hexdigest());docs[path]=copy.deepcopy(value);docrefs[path]=r;return r
def ram_read(r):
 v=docs[r["path"]];need(docrefs[r["path"]]==r,"RAM immutable exact ref "+r["path"]);return copy.deepcopy(v)
# Actual target modules, including the actual worker; no stubs for imports.
ns=D.relative_to(ROOT).as_posix().replace("/",".")
c=importlib.import_module(ns+".contract_v1");e=importlib.import_module(ns+".execution_v1")
w=importlib.import_module(ns+".worker_v1");direct=importlib.import_module(ns+".direct_gpu0_v1")
need("PendingPrerequisites"not in vars(w),"Actual worker does not fabricate removed missing exception")
need(callable(w.check_external_prerequisites_v1),"Actual worker prerequisite function exists")
hp=json.loads((D/"pins.v1.json").read_text())["pins"]
hd=importlib.import_module(hp["H058_DIRECT"]["path"][:-3].replace("/","."))
hc=importlib.import_module(hp["H058_COMMON"]["path"][:-3].replace("/","."))
need(callable(hd.controller_v1)and callable(hd.capacity_v1)and callable(hc.validate_native_resource_disposition_v1),"Actual current H058v9 direct/common import interface")
for fn in ("execution_v1.py","contract_v1.py","worker_v1.py","direct_gpu0_v1.py"):
 source=(D/fn).read_text();tree=ast.parse(source)
 for n in ast.walk(tree):
  if not isinstance(n,ast.ImportFrom)or not (n.module or"").startswith("research."):continue
  imported=importlib.import_module(n.module)
  for a in n.names:
   if a.name=="*":continue
   try:getattr(imported,a.name)
   except AttributeError:importlib.import_module(n.module+"."+a.name)
   need(True,"Real import target "+n.module+"."+a.name)
for module in ("execution_v1","contract_v1"):
 for args in (["--help"],[]):
  command=[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",ns+"."+module]+args
  run=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
  need(run.returncode==0 and not run.stderr,"Actual CLI "+module+" "+repr(args))
  if args:need("usage:"in run.stdout,"Real argparse help")
  else:need(json.loads(run.stdout)["status"]=="SOURCE_ONLY_INACTIVE","Real inactive default")
  cli.append(dict(command=command,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
need(not any(n in sys.modules for n in("torch","torch_geometric","numpy","tskit","psutil")),"No scientific imports")
# All actual entry actions parse against their real module; operational bodies are RAM callables.
r=dict(path="RAM/exact_input.json",sha256="1"*64);j=json.dumps(r);calls=[]
for action,target in (("--execute","execute_v1"),("--supervise-once","supervise_once_v4")):
 with patch.object(e,"set_amendment_v1",lambda x:calls.append(("amend",x))),patch.object(e,target,lambda x:dict(returncode=0)),contextlib.redirect_stdout(io.StringIO()):
  need(e.main([action,"--input-ref",j,"--amendment-ref",j])==0,"Real CLI dispatch "+action)
with patch.object(e,"set_amendment_v1",lambda x:None),patch.object(e,"worker_v1",lambda a,b,d:7),contextlib.redirect_stdout(io.StringIO()):
 need(e.main(["--worker-input-ref",j,"--gate-fd","91","--attachment-ref",j,"--amendment-ref",j])==7,"Real CLI gated worker dispatch")
with patch.object(e,"set_amendment_v1",lambda x:None):
 reject(lambda:e.main(["--execute","--input-ref",j,"--amendment-ref",json.dumps(dict(path="RAM/other",sha256="2"*64))]),"Mismatched exact CLI amendment rejected")
class Boundary(Exception):pass
# Execute reaches the actual worker import and actual prerequisite function binding.
with patch.object(e,"validate_input_v1",lambda r:({},{"RAM":True},{},{},{})),patch.object(w,"check_external_prerequisites_v1",side_effect=Boundary("actual worker boundary reached")):
 try:e.execute_v1(r)
 except Boundary:need(True,"Real execute imports actual worker before guarded operations")
 else:raise AssertionError("Boundary not reached")
# The real preparation CLI parses both metadata actions; operational bodies are RAM.
for action,target in (("--prepare-ref","prepare_v1"),("--draft-election-ref","draft_election_v1")):
 with patch.object(c,target,lambda value:dict(status="RAM_ONLY")),contextlib.redirect_stdout(io.StringIO()):
  need(c.main([action,j])==0,"Real contract CLI dispatch "+action)
# Inspect the exact source-emitted caller and gated-worker actions against the real parser.
prepare_source=ast.get_source_segment((D/"contract_v1.py").read_text(),next(n for n in ast.parse((D/"contract_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="prepare_v1"))
need('"--supervise-once"'in prepare_source and '"--execute"'not in prepare_source and '"setsid"'not in prepare_source,"Preparation emits exact new one-shot caller action")
exec_source=ast.get_source_segment((D/"execution_v1.py").read_text(),next(n for n in ast.parse((D/"execution_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="execute_v1"))
need("'--worker-input-ref'"in exec_source and "'--gate-fd'"in exec_source and "'--amendment-ref'"in exec_source,"Gated worker emits actual supported CLI arguments")
# One-shot caller: RAM Popen/waitpid and RAM immutable records.
spec=dict(worker_auxiliary_scope="research/reviews/C3_remaining_cells_GPU0_ROOT_v4_RAM/worker_aux")
def caller_case(label,status=0,wait_error=None,launch_error=None,interrupted=False,wrong_pid=False,write_error=None):
 records={};launches=[];waits=[];proc=types.SimpleNamespace(pid=456789,returncode=None);first=[interrupted]
 def record(s,name,value):
  if name in records:raise FileExistsError(name)
  if name==write_error:raise OSError("RAM evidence write uncertainty")
  records[name]=copy.deepcopy(value);return dict(path="RAM/"+name,sha256="a"*64)
 def popen(command,**kwargs):
  launches.append((command,kwargs))
  if launch_error:raise launch_error
  return proc
 def waitpid(pid,flags):
  waits.append((pid,flags))
  if first[0]:first[0]=False;raise InterruptedError()
  if wait_error:raise wait_error
  return (999 if wrong_pid else pid),status
 fake_os=types.SimpleNamespace(waitpid=waitpid,waitstatus_to_exitcode=os.waitstatus_to_exitcode,WIFEXITED=os.WIFEXITED,WIFSIGNALED=os.WIFSIGNALED)
 with patch.object(e,"validate_input_v1",lambda x:(spec,{},{},{},{})),patch.object(e,"_put_supervisor_record_v4",record),patch.object(e,"subprocess",types.SimpleNamespace(Popen=popen)),patch.object(e,"os",fake_os):
  if wait_error or launch_error or wrong_pid or write_error:
   reject(lambda:e.supervise_once_v4(r),label)
   fail=records.get("caller_supervisor_failure.v4.json")
   if fail:
    need(fail["automatic_retry"]is False and fail["scientific_success_claimed"]is False,label+" no false success/retry")
    if wait_error or launch_error or wrong_pid:need(fail["actual_Python_supervisor_wait"]is False,label+" no false exact wait")
  else:
   result=e.supervise_once_v4(r);expected=os.waitstatus_to_exitcode(status)
   need(result["actual_OS_wait_status"]==status and result["actual_Python_supervisor_returncode"]==expected,label+" actual raw/Python status")
   need(result["returncode"]==(expected if expected>=0 else 128-expected),label+" exact caller exit")
   need(result["worker_wait_or_scientific_completion_inferred"]is False and result["storage_close_or_credit_authorized"]is False,label+" independent worker/science audit retained")
   need(proc.returncode==expected,label+" Popen exact reaped status retained")
   reject(lambda:e.supervise_once_v4(r),label+" invocation cannot replay")
   need(len(launches)==1,label+" no second Popen")
  need("caller_invocation.v4.json"in records,label+" invocation consumed before launch")
  if launches:
   command,kwargs=launches[0]
   need(command[:6]==[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",ns+".execution_v1","--execute"],label+" exact direct Python argv")
   need(json.loads(command[7])==json.loads(command[9])==r and kwargs==dict(cwd=ROOT,start_new_session=True),label+" exact refs and owned Python session")
  if waits:need(all(x==(proc.pid,0)for x in waits),label+" wait exact direct child")
 return records
caller_case("zero")
caller_case("nonzero",status=9<<8)
caller_case("signal",status=15)
caller_case("interrupted then genuine wait",interrupted=True)
caller_case("Popen failure",launch_error=OSError("RAM launch failure"))
caller_case("wait unknown",wait_error=ChildProcessError("RAM no child"))
caller_case("wrong returned PID",wrong_pid=True)
caller_case("wait receipt failure",write_error="caller_actual_supervisor_wait.v4.json")
# Exact caller writer runs over RAM file descriptors/path metadata, no filesystem writes.
class RPath:
 def __init__(self,value):self.value=value
 def __truediv__(self,name):return RPath(self.value+"/"+name)
 @property
 def parent(self):return RPath(self.value.rsplit("/",1)[0])
 def __fspath__(self):return self.value
 def relative_to(self,root):return self.value[len(str(root))+1:]
 def lstat(self):return types.SimpleNamespace(st_dev=1,st_ino=22,st_mode=0o100600,st_nlink=1,st_size=len(streams[10].getvalue()))
 def __str__(self):return self.value
class Stream(io.BytesIO):
 def fileno(self):return 10
 def close(self):pass
streams={};fs_events=[];created=set();budget=[]
def fake_open(path,flags,*mode):
 fs_events.append(("open",str(path),flags,mode))
 if flags&os.O_CREAT:
  if str(path)in created:raise FileExistsError(str(path))
  created.add(str(path));streams[10]=Stream();return 10
 return 11
ram_os=types.SimpleNamespace(open=fake_open,fdopen=lambda fd,mode:streams[fd],fstat=lambda fd:types.SimpleNamespace(st_dev=1,st_ino=22,st_mode=0o100600,st_nlink=1,st_size=len(streams[fd].getvalue())),fsync=lambda fd:fs_events.append(("fsync",fd)),close=lambda fd:fs_events.append(("close",fd)))
for key in("O_WRONLY","O_CREAT","O_EXCL","O_NOFOLLOW","O_RDONLY","O_DIRECTORY"):setattr(ram_os,key,getattr(os,key))
with patch.object(e,"local",lambda p:RPath(str(ROOT)+"/"+p)),patch.object(e,"record_budget_v1",lambda directory,extra:budget.append(extra)),patch.object(e,"os",ram_os):
 wr=e._put_supervisor_record_v4(spec,"caller.json",dict(source="RAM"))
 need(budget==[len(streams[10].getvalue())+8192],"Writer calls original budget with correct two arguments and full allowance")
 need(fs_events[0][2]&os.O_EXCL and fs_events[0][2]&os.O_NOFOLLOW,"Caller evidence exclusive nofollow")
 need(("fsync",10)in fs_events and("fsync",11)in fs_events,"Caller evidence and containing directory fsync")
 reject(lambda:e._put_supervisor_record_v4(spec,"caller.json",{}),"Actual RAM writer refuses record replay")
# Current peer qualification from frozen metadata; future launch exists in RAM only.
pins=json.loads((D/"pins.v1.json").read_text());p=pins["pins"]
for key in("H058_INPUT","H058_REQUEST","H058_REVIEW","C3_V3_REVOCATION"):
 rr=p[key];need(ref(rr["path"])==rr,"Frozen current peer metadata "+key);docs[rr["path"]]=json.loads((ROOT/rr["path"]).read_text());docrefs[rr["path"]]=rr
now=datetime.now(timezone.utc);clock=dict(root_authorizer="root_agent",status="active",template_only=False,created_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=(now+timedelta(hours=1)).isoformat())
identity=dict(pid=1234567,start=9876543,group=1234567,session=1234567,ppid=1234500)
launch=put(p["H058_LAUNCH_PATH"],dict(schema="H058_fixed_full_parent_actual_child_launch_v1",input_ref=p["H058_INPUT"],pid=identity["pid"],ownership=dict(owned_identities=[identity]),borrowed_physical_inode=True,controller_inheritance_claimed=False))
qual=json.loads((D/"ROOT_peer_qualification.INACTIVE.template.v4.json").read_text());qual.update(clock,H058_launch_ref=launch,worker_identity=identity);qr=put("RAM/peer_qualification.json",qual)
peer=json.loads((D/"ROOT_peer.INACTIVE.template.v4.json").read_text());peer.update(clock,H058_launch_ref=launch,worker_identity=identity,peer_source_qualification_ref=qr);pr=put("RAM/peer.json",peer)
def peer_test(value):return c.peer_values_v1(put("RAM/test_peer.json",value))
with patch.object(c,"read",ram_read),patch.object(c,"pins_v1",lambda:pins),patch.object(c,"raw",lambda r:need(r in(p["H058_DIRECT"],p["H058_COMMON"]),"Exact current direct/common source authentication")):
 got,hq=c.peer_values_v1(pr);need(got==peer and hq==docs[p["H058_REQUEST"]["path"]],"Current qualified v9 exact interface")
 for key,value in (("H058_input_ref",pins["historical_H058_v7"]["H058_INPUT"]),("H058_launch_ref",dict(path="old/v7/launch",sha256="0"*64)),("worker_identity",dict(identity,start=1)),("H058_full_cap_plus_margin_bytes",0),("terminal_peer_absence_preserves_full30GiB",False),("peer_source_qualification_ref",None)):
  bad=copy.deepcopy(peer);bad[key]=value;reject(lambda:peer_test(bad),"Peer rejects "+key)
 for key,value in (("H058_direct_source_ref",dict(path="old/direct",sha256="0"*64)),("H058_whole_source_review_ref",dict(path="old/review",sha256="0"*64)),("worker_identity",dict(identity,start=1)),("original_C3_26_H058_30_EVAL_10_headroom_4_GiB_envelopes_preserved",False)):
  badq=copy.deepcopy(qual);badq[key]=value;bad=copy.deepcopy(peer);bad["peer_source_qualification_ref"]=put("RAM/badqual.json",badq);reject(lambda:peer_test(bad),"Qualification rejects "+key)
# Fresh registration does not authorize old input, credit or retry.
q=json.loads((D/"ROOT_predict_val_000.INACTIVE.template.v4.json").read_text());q.update(clock,source_review_ref=dict(path="RAM/v4review",sha256="b"*64),peer_launch_capsule_ref=pr)
reg=json.loads((D/"ROOT_registration.INACTIVE.template.v4.json").read_text());reg.update(clock)
for key in("source_manifest_ref","source_review_ref","phase","job_id","output_directory","worker_auxiliary_scope","cell_election_ref","peer_launch_capsule_ref"):reg[key]=copy.deepcopy(q[key])
q["registration_ref"]=put("research/reviews/C3_remaining_GPU0_v4_ROOT_preparation_RAM/registration.json",reg)
req=dict(path="research/reviews/C3_remaining_GPU0_v4_ROOT_preparation_RAM/request.json",sha256="c"*64)
with patch.object(c,"read",ram_read),patch.object(c,"pins_v1",lambda:pins):
 need(c.registration_values_v4(q,req)==reg,"Exact fresh v4 registration")
 for key,value in (("old_v3_charge_preserved_bytes",0),("old_v3_Python_supervisor_wait",0),("old_v3_authority_or_admission_reuse_authorized",True),("phase","matrix"),("automatic_retry_or_credit_authorized",True)):
  bad=copy.deepcopy(reg);bad[key]=value;badq=copy.deepcopy(q);badq["registration_ref"]=put("research/reviews/C3_remaining_GPU0_v4_ROOT_preparation_RAM/badreg.json",bad);reject(lambda:c.registration_values_v4(badq,req),"Registration rejects "+key)
 reject(lambda:c.registration_values_v4(q,dict(path="old/v3/request",sha256="0"*64)),"Old request namespace rejected")
# Source preservation and exact direct role union using original selected child's real pure function.
def normalized(path):
 return path.read_text().replace("C3_remaining_cells_GPU0_source_20261017_v4","C3_remaining_cells_GPU0_source_20261017_v3")
for file in("worker_v1.py","direct_gpu0_v1.py","__init__.py"):need((D/file).read_bytes()==(OLD/file).read_bytes(),"Byte-identical original "+file)
oldtree=ast.parse((OLD/"execution_v1.py").read_text());newtree=ast.parse(normalized(D/"execution_v1.py"))
oldnodes={n.name:n for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
newnodes={n.name:n for n in newtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for name in oldnodes:
 if name=="main":continue
 need(ast.dump(oldnodes[name],include_attributes=False)==ast.dump(newnodes[name],include_attributes=False),"Original execution/ownership AST unchanged "+name)
oldct={n.name:n for n in ast.parse((OLD/"contract_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)}
newct={n.name:n for n in ast.parse(normalized(D/"contract_v1.py")).body if isinstance(n,ast.FunctionDef)}
for name in oldct:
 if name in("peer_values_v1","validate_request_v1","prepare_v1"):continue
 need(ast.dump(oldct[name],include_attributes=False)==ast.dump(newct[name],include_attributes=False),"Original contract/science/wait AST unchanged "+name)
snapshot=json.loads((ROOT/"research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/final_original_journal_acceptance.root_v1.json").read_text())
def rolemap(v):
 if isinstance(v,dict):
  if isinstance(v.get("bindings"),dict)and len(v["bindings"])==46:return v["bindings"]
  for value in v.values():
   out=rolemap(value)
   if out:return out
 if isinstance(v,list):
  for value in v:
   out=rolemap(value)
   if out:return out
roles=copy.deepcopy(rolemap(snapshot));roles["Census_observation_explicit_selector_capsule"]=p["CENSUS_SELECTOR"]
need(len(roles)==47,"All47 historical current parent roles present")
gt=ast.parse((ROOT/p["CENSUS_GUARD"]["path"]).read_text())
fn=next(n for n in gt.body if isinstance(n,ast.FunctionDef)and n.name=="consumer_bindings_v1")
env=dict(require=c.require,copy=copy,_raw=lambda r:None);exec(compile(ast.Module(body=[fn],type_ignores=[]),"<exact_selected_child_pure_union>","exec"),env)
union=env["consumer_bindings_v1"];manifest=json.loads((D/"source_manifest.v1.json").read_text())
direct_counts={}
with patch.object(c,"read",lambda r:manifest),patch.object(c,"pins_v1",lambda:pins):
 for phase in c.PHASES:
  qd=dict(q,phase=phase,direct_attachment_authority_ref=dict(path="RAM/"+phase+"/attachment",sha256="d"*64))
  refs=c.direct_refs_v1(qd,dict(path="RAM/"+phase+"/request",sha256="e"*64),dict(path="RAM/"+phase+"/config",sha256="f"*64))
  result=union(list(roles.values()),refs);direct_counts[phase]=len(result)
  need(all(x in result for x in roles.values()),"All47 roles retained "+phase)
  need(len(result)==(62 if phase=="matrix" else 64),"Exact original union cap "+phase)
reject(lambda:union([dict(path="RAM/extra"+str(i),sha256="0"*64)for i in range(65)]),"Original65ref cap rejected")
# Manifest freshness and inactive templates, bounded metadata only.
for rr in manifest["executed_source_refs"]:
 need(ref(rr["path"])==rr,"Current executed source hash "+Path(rr["path"]).name)
need(len(manifest["executed_source_refs"])==5,"Caller embedded in existing execution role")
context=json.loads((D/"ROOT_context.v4.json").read_text())
need(sum(context["retained_count_by_parent"].values())==27 and all(v==[30]for k,v in context["pending_epochs_by_parent"].items()if k in c.VAL[:3]),"Truthful retained27 pending3 context")
for template in D.glob("*.INACTIVE.template.v4.json"):
 v=json.loads(template.read_text());need(v["template_only"]is True and v["status"]=="inactive","Inactive template "+template.name)
 if"source_manifest_ref"in v:need(v["source_manifest_ref"]==ref(D/"source_manifest.v1.json"),"Template current manifest "+template.name)
result=dict(schema="C3_v4_focused_author_source_checks_v1",verdict="AUTHOR_SOURCE_CHECKS_ONLY",positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,actual_CLI_smoke=cli,actual_target_worker_imported=True,actual_cli_help_or_inactive_only=True,direct_unions=direct_counts,parent_roles=47,parent_unique_refs=len({json.dumps(x,sort_keys=True)for x in roles.values()}),actual_execution_admission_resource_probe_or_guard_calls=False,scientific_imports_or_payload_reads=False,old_v3_unknown_Python_wait_preserved=True)
(D/"focused_checks.result.v4.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases","actual_CLI_smoke")},sort_keys=True))

"""Narrow actual-metadata/schema and CLI/caller checks; no resource operations."""
from pathlib import Path
from unittest.mock import patch
import ast,copy,contextlib,hashlib,importlib,io,json,os,subprocess,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");D=ROOT/"research/reviews/C3_remaining_cells_GPU0_source_20261017_v5";OLD=D.parent/"C3_remaining_cells_GPU0_source_20261017_v4"
checks=[];negative=[];cli=[]
def need(value,label):
 if not value:raise AssertionError(label)
 checks.append(label)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def bounded(record):
 p=ROOT/record["path"];need(p.stat().st_size<=2**20,"Bounded metadata "+p.name)
 b=p.read_bytes();need(hashlib.sha256(b).hexdigest()==record["sha256"],"Exact raw metadata "+p.name);return json.loads(b)
p=json.loads((D/"pins.v1.json").read_text())["pins"]
ns=D.relative_to(ROOT).as_posix().replace("/",".");oldns=OLD.relative_to(ROOT).as_posix().replace("/",".")
c=importlib.import_module(ns+".contract_v1");e=importlib.import_module(ns+".execution_v1");w=importlib.import_module(ns+".worker_v1");oldc=importlib.import_module(oldns+".contract_v1")
launch=bounded(p["H058_LAUNCH"]);inp=bounded(p["H058_INPUT"]);q=bounded(p["H058_REQUEST"]);attachment=bounded(p["H058_ATTACHMENT"])
peer_ref=ref("research/reviews/C3_remaining_GPU0_v4_ROOT_preparation_root01_oct17/H058_v9_peer.root_active_v1.json")
peer=bounded(peer_ref);qualification=bounded(peer["peer_source_qualification_ref"])
need(launch["input_ref"]==p["H058_INPUT"] and inp["root_stage_request_ref"]==p["H058_REQUEST"] and launch["attachment_ref"]==p["H058_ATTACHMENT"],"Actual input/request/launch/attachment exact chain")
identity=launch["ownership"]["current_owned_identities"][0]
need(identity==dict(pid=46648,start=491287607,group=46648,state="R") and launch["parent_pid"]==39747,"Actual immutable launch full identity, no invented row")
need(attachment["direct_attachment_authority_ref"]==q["direct_attachment_authority_ref"] and attachment["job_id"]==inp["job_id"]==q["job_id"] and attachment["borrowed_physical_inode"]is True,"Actual borrowed attachment/request/job lineage")
# Reproduce actual v4 preflight using genuine immutable metadata, with no capacity or prepare calls.
try:oldc.peer_values_v1(peer_ref)
except KeyError as exc:need(exc.args==("owned_identities",),"Preserved real v4 schema preflight failure reproduced")
else:raise AssertionError("V4 actual missing-field failure not reproduced")
got,hq=c.peer_values_v1(peer_ref)
need(got==peer and hq==q and peer["worker_identity"]==identity==qualification["worker_identity"],"Real v5 peer metadata join succeeds without resource doubles or schema injection")
# Current embedded summary producer from pinned actual operational source, isolated in RAM.
driver=(ROOT/p["H058_DRIVER"]["path"]).read_text();need(ref(p["H058_DRIVER"]["path"])==p["H058_DRIVER"],"Pinned actual v9 driver")
node=next(n for n in ast.parse(driver).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="_DRIVER_SOURCE"for t in n.targets))
inner=ast.literal_eval(node.value);tree=ast.parse(inner)
history=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=="_HistoryV8")
owned=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=="OwnedLifecycleV8")
evidence=next(n for n in owned.body if isinstance(n,ast.FunctionDef)and n.name=="evidence")
env=dict(collections=__import__("collections"),hashlib=hashlib,json=json,OWNED_LIMITS_V8=copy.deepcopy(launch["ownership"]["limits"]))
exec(compile(ast.Module(body=[history,evidence],type_ignores=[]),"<actual_pinned_v9_summary_RAM>","exec"),env)
discovered=env["_HistoryV8"]();discovered.add({k:identity[k]for k in("pid","start","group")})
ramtree=types.SimpleNamespace(groups={identity["group"]},complete=True,known={identity["pid"]:copy.deepcopy(identity)},discovered=discovered,retired=env["_HistoryV8"](),reused=env["_HistoryV8"](),retirement_reasons=dict(absent=0,replaced=0))
produced=env["evidence"](ramtree)
need(produced==launch["ownership"],"Actual v9 OwnedLifecycle summary producer exactly reproduces entire genuine launch ownership")
need("current_owned_identities"in produced and"owned_identities"not in produced,"Producer uses actual current summary schema exclusively")
# RAM tamper negatives exercise production binding, never provide a positive legacy fallback.
real_read=c.read
for label,mutate in [
 ("legacy ownership only",lambda v:v["ownership"].update(owned_identities=v["ownership"].pop("current_owned_identities"))),
 ("legacy field injected",lambda v:v["ownership"].update(owned_identities=[])),
 ("incomplete summary",lambda v:v["ownership"].update(complete_owned_roster_known=False)),
 ("wrong attachment ref",lambda v:v.update(attachment_ref=dict(path="wrong",sha256="0"*64)))]:
 bad=copy.deepcopy(launch);mutate(bad)
 def view(rr):return bad if rr==p["H058_LAUNCH"]else real_read(rr)
 with patch.object(c,"read",view):
  try:c.peer_values_v1(peer_ref)
  except(ValueError,KeyError,TypeError,AssertionError):negative.append(label)
  else:raise AssertionError("Accepted "+label)
badpeer=copy.deepcopy(peer);badpeer["H058_launch_ref"]=dict(p["H058_LAUNCH"],sha256="0"*64)
with patch.object(c,"read",lambda rr:badpeer if rr==peer_ref else real_read(rr)):
 try:c.peer_values_v1(peer_ref)
 except ValueError:negative.append("untrusted future launch digest")
 else:raise AssertionError("Untrusted launch ref accepted")
# Only actual help is executed; source entry and caller actions use RAM operations.
command=[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",ns+".execution_v1","--help"]
run=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
need(run.returncode==0 and not run.stderr and"--supervise-once"in run.stdout,"Actual v5 pinned-runtime CLI help and imports")
cli.append(dict(command=command,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
need(callable(w.check_external_prerequisites_v1)and"PendingPrerequisites"not in vars(w),"Actual worker prerequisite import remains correct")
r=dict(path="RAM/v5input",sha256="1"*64);j=json.dumps(r)
for action,target in(("--execute","execute_v1"),("--supervise-once","supervise_once_v5")):
 with patch.object(e,"set_amendment_v1",lambda r:None),patch.object(e,target,lambda r:dict(returncode=0)),contextlib.redirect_stdout(io.StringIO()):
  need(e.main([action,"--input-ref",j,"--amendment-ref",j])==0,"Actual v5 CLI parser dispatch "+action)
for raw in(0,9<<8):
 writes={};calls=[];proc=types.SimpleNamespace(pid=12345,returncode=None)
 def record(s,name,value):writes[name]=copy.deepcopy(value);return dict(path="RAM/"+name,sha256="2"*64)
 def popen(command,**kw):calls.append((command,kw));return proc
 fakeos=types.SimpleNamespace(waitpid=lambda pid,flags:(pid,raw),waitstatus_to_exitcode=os.waitstatus_to_exitcode,WIFEXITED=os.WIFEXITED,WIFSIGNALED=os.WIFSIGNALED)
 with patch.object(e,"validate_input_v1",lambda r:({},q,{},{},{})),patch.object(e,"_put_supervisor_record_v5",record),patch.object(e,"subprocess",types.SimpleNamespace(Popen=popen)),patch.object(e,"os",fakeos):
  result=e.supervise_once_v5(r)
 need(result["actual_OS_wait_status"]==raw and result["returncode"]==raw>>8 and result["actual_Python_supervisor_wait"]is True,"Inherited actual caller status propagation "+str(raw))
 need(calls[0][0][4]==ns+".execution_v1" and calls[0][0][5]=="--execute" and"caller_invocation.v5.json"in writes and"caller_actual_supervisor_wait.v5.json"in writes,"Exact new v5 emitted module and caller evidence namespace "+str(raw))
# Namespace-normalized AST equality proves no new science/lifecycle/guard/caller redesign.
for name in("execution_v1.py","direct_gpu0_v1.py","worker_v1.py","__init__.py"):
 old=(OLD/name).read_text();new=(D/name).read_text().replace("C3_remaining_cells_GPU0_source_20261017_v5","C3_remaining_cells_GPU0_source_20261017_v4").replace("_v5","_v4").replace("v5_","v4_").replace("-v5-","-v4-").replace("C3_v5_","C3_v4_").replace(".v5.json",".v4.json").replace("fresh v5","fresh v4").replace("Fresh v5","Fresh v4")
 need(ast.dump(ast.parse(old),include_attributes=False)==ast.dump(ast.parse(new),include_attributes=False),"Namespace-only predecessor AST "+name)
oldtree=ast.parse((OLD/"contract_v1.py").read_text());newsource=(D/"contract_v1.py").read_text().replace("C3_remaining_cells_GPU0_source_20261017_v5","C3_remaining_cells_GPU0_source_20261017_v4").replace("_v5","_v4").replace("v5_","v4_").replace("-v5-","-v4-").replace("fresh v5","fresh v4").replace("Fresh v5","Fresh v4")
newtree=ast.parse(newsource);oldmap={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)};newmap={n.name:n for n in newtree.body if isinstance(n,ast.FunctionDef)}
need(set(oldmap)==set(newmap)and all(ast.dump(oldmap[k],include_attributes=False)==ast.dump(newmap[k],include_attributes=False)for k in oldmap if k!="peer_values_v1"),"Only peer schema/attachment binding changes contract beyond namespaces")
need(not any(n in sys.modules for n in("torch","torch_geometric","numpy","tskit","psutil")),"No scientific imports or resource APIs")
result=dict(schema="C3_v5_narrow_actual_metadata_schema_CLI_caller_author_checks_v1",verdict="AUTHOR_SOURCE_CHECKS_ONLY",checks=checks,positive_count=len(checks),negative_cases=negative,negative_count=len(negative),actual_metadata_refs={k:p[k]for k in("H058_INPUT","H058_REQUEST","H058_LAUNCH","H058_ATTACHMENT","H058_DRIVER")},actual_peer_capsule_ref=peer_ref,actual_CLI_smoke=cli,actual_schema_producer_exact_join=True,v4_actual_preflight_KeyError_reproduced=True,legacy_fallback=False,actual_resource_guard_capacity_admission_process_GPU_or_Git_operations=False,scientific_payloads_read=False)
(D/"narrow_checks.result.v5.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases","actual_CLI_smoke","actual_metadata_refs")},sort_keys=True))

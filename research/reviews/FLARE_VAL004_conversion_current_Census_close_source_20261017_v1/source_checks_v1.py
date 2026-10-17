"""Focused VAL004 genuine-metadata successor checks; no guard import or real observation."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import ast,contextlib,copy,hashlib,importlib.util,io,json,sys,types
D=Path("research/reviews/FLARE_VAL004_conversion_current_Census_close_source_20261017_v1")
OLD=Path("research/reviews/FLARE_VAL002_conversion_current_Census_close_source_20261017_v1")
sp=importlib.util.spec_from_file_location("_FLARE_VAL004_close_source_check",D/"close_ROOT_v1.py")
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
checks=[];negative=[]
def need(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(fn,label):
 try:fn()
 except(ValueError,KeyError,TypeError,AssertionError):negative.append(label);return
 raise AssertionError("Accepted "+label)
p=m.pins();inp,wait,done,launch,storage,own,pids,groups=m.metadata_join(p)
need(wait["schema"]=="FLARE_phase_CPU_actual_wait_v2"and wait["pid"]==4184625 and wait["supervisor_pid"]==4183066 and pids==groups==[4183066,4184625]and launch["ownership"]["owned_identities"]==[dict(pid=4184625,start=490648256)],"Genuine VAL004 producer ownership schema and complete recorded worker/supervisor PID/group join")
for r in m.read(m.ref(str(D/"source_manifest.v1.json")))["source_refs"]:m.raw(r)
need(p["selector"]["sha256"]=="fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4"and p["guard"]["sha256"]=="72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb"and p["policy"]["sha256"]=="5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5","All manifest refs authenticated; exact current Census selector/child/policy unchanged")
need(hashlib.sha256(json.dumps(own,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()==p["reservation_sha256"]=="37b6091f9556a0a88950d9b3836e5ca115a9043d7721860419dda80f7a1f01d3"and own["reserved_bytes"]==898402844 and len(own["consumer_bindings"])==62 and own["root_admission"]["sha256"]=="e061bd3284cc93ead47d9d42bce41f3afee741d00ef227ef826bc5b61e351338","Entire genuine original reservation, admission ref,62bindings and both scopes retained by exact digest")
with contextlib.redirect_stdout(io.StringIO())as output:rc=m.main([])
need(rc==0 and json.loads(output.getvalue())==dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False),"Actual default invocation inert")
oldsource=(OLD/"close_ROOT_v1.py").read_text();source=(D/"close_ROOT_v1.py").read_text()
normal=source
for old,new in reversed([("VAL002","VAL004"),("val_002","val_004"),("18397","82340"),("899404366","898402844")]):normal=normal.replace(new,old)
a=ast.dump(ast.parse(oldsource),include_attributes=False);b=ast.dump(ast.parse(normal),include_attributes=False)
delta=json.loads((D/"normalized_AST_delta.v1.json").read_text())
need(normal==oldsource and a==b and hashlib.sha256(b.encode()).hexdigest()==delta["normalized_AST_sha256"],"Complete source and whole module AST equal to reviewed VAL002 after only four exact namespace/actual-binding substitutions")
real_read=m.read
oldinput=json.loads((OLD.parent/"FLARE_remaining_ROOT_actual_VAL002_convert_Census_root01_oct17/input.json").read_text())
for label,key,value in [
 ("VAL002 input cannot replace genuine004 input","input",oldinput),
 ("nonzero genuine004 direct wait","wait",dict(wait,returncode=1)),
 ("incomplete genuine004 owned drain","wait",dict(wait,all_owned_processes_and_groups_drained=False)),
 ("VAL002 tool18397 cannot replace consumed004 tool82340","outer",dict(real_read(p["outer"]),tool_session_id=18397,actual_tool_session_id=18397))]:
 with patch.object(m,"read",lambda r,v=value,k=key:v if r==p[k]else real_read(r)):reject(lambda:m.metadata_join(p),label)
# Compile only the original proof functions. All I/O/digest/canonical/absence boundaries below are RAM stubs.
tree=ast.parse(Path(p["guard"]["path"]).read_text());env=dict(require=m.need,datetime=datetime,timezone=timezone,Path=Path,ROOT=m.ROOT)
for name in("_pointer","_identity","_completed_identity","_proof"):
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),"<exact_current_child_"+name+"_RAM>","exec"),env)
def integer(x,label,minimum):
 if type(x)is not int or x<minimum:raise ValueError(label)
env["budget"]=types.SimpleNamespace(digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),v4=types.SimpleNamespace(encoded=lambda value:json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),integer=integer,canonical=lambda root,path:Path(path)if Path(path).is_absolute()else root/path))
obsref=dict(path="RAM/VAL004_targeted_observation.json",sha256="3"*64)
stamp=datetime.now(timezone.utc).isoformat()
rows=[dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=stamp)for pid in pids]
obs=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",key=own["key"],reservation_sha256=p["reservation_sha256"],storage_ref=p["storage"],launch_ref=p["launch"],direct_wait_ref=p["wait"],completed_ref=p["completed"],launcher_pid=launch["pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=rows)
env["_json"]=lambda r:copy.deepcopy(obs)if r==obsref else real_read(r)
env["_completed_json"]=real_read
env["_absent"]=lambda pid:copy.deepcopy(next(r for r in rows if r["pid"]==pid))
request=m.request_value(p,own,launch,obsref)
proof=env["_proof"](request,own,observe=False);applied=env["_proof"](request,own,observe=True)
need(proof["pids"]==pids and proof["completion_verified"]is True and applied["PID_observations"]==rows,"Exact current ordinary proof accepts genuine004 metadata with RAM-only absent observations in both proof branches")
bad=copy.deepcopy(request);bad["wait_identity"]["returncode"]=1
reject(lambda:env["_proof"](bad,own),"Original ordinary proof rejects changed004 direct-wait identity")
producer=Path(p["producer_driver"]["path"]).read_text();fn=next(n for n in ast.parse(producer).body if isinstance(n,ast.FunctionDef)and n.name=="owned_execute");body=ast.get_source_segment(producer,fn)
need("supervisor_pid=os.getpid()"in body and"known_owned_pids=sorted(tree.known)"in body and"FLARE_phase_CPU_actual_wait_v2"in body,"Unchanged genuine producer writes the pinned direct-wait supervisor and owned-roster fields")
need(not any(name.startswith("_ROOT_FLARE_VAL004_exact_current_Census_close")for name in sys.modules),"No actual current guard imported")
result=dict(schema="FLARE_VAL004_current_Census_close_focused_successor_author_checks_v1",verdict="AUTHOR_SOURCE_CHECKS_ONLY",positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,genuine_actual_metadata_join=True,normalized_whole_AST_equal=True,actual_current_ordinary_proof_RAM_join=True,actual_guard_import_state_lock_measurement_observation_absence_resource_admission_close_or_Git_calls=False,scientific_payloads_opened=False,predecessor_broad_campaign_repeated=False)
with(D/"source_checks.result.v1.json").open("x")as stream:stream.write(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))

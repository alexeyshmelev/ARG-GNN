"""Narrow source/actual-metadata and existing ordinary-proof RAM checks. No guard import."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import ast,contextlib,copy,hashlib,importlib.util,io,json,sys,types
D=Path("research/reviews/FLARE_VAL002_conversion_current_Census_close_source_20261017_v1")
sp=importlib.util.spec_from_file_location("_FLARE_VAL002_close_source_check",D/"close_ROOT_v1.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
checks=[];negative=[]
def need(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(fn,label):
 try:fn()
 except(ValueError,KeyError,TypeError,AssertionError):negative.append(label);return
 raise AssertionError("Accepted "+label)
p=m.pins();joined=m.metadata_join(p);inp,wait,done,launch,storage,own,pids,groups=joined
need(pids==[4167377,4168865]and groups==[4167377,4168865],"Roster derived exclusively from genuine recorded worker/supervisor wait")
need(launch["ownership"]["owned_identities"]==[dict(pid=4168865,start=490547826)],"Actual conversion launch schema, no H058 shape substitution")
need(hashlib.sha256(json.dumps(own,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()==p["reservation_sha256"],"Exact original budget encoded reservation digest")
need(own["reserved_bytes"]==899404366 and len(own["consumer_bindings"])==62,"Original reservation and62direct refs unchanged")
with contextlib.redirect_stdout(io.StringIO())as output:need(m.main([])==0,"Actual default remains inert")
need(json.loads(output.getvalue())==dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False),"No guard load or operation in default")
real_read=m.read
for label,key,mutate in [
 ("nonzero direct wait","wait",lambda x:x.update(returncode=1)),
 ("unknown direct wait","wait",lambda x:x.update(actual_direct_worker_wait=False)),
 ("incomplete drain","wait",lambda x:x.update(all_owned_processes_and_groups_drained=False)),
 ("wrong source-bound launch input","launch",lambda x:x.update(input_ref=dict(path="other",sha256="0"*64))),
 ("unconsumed outer","outer",lambda x:x.update(actual_terminal_consumed=False)),
 ("wrong completion parent","completed",lambda x:x.update(parent_id="american_c3_val_001"))]:
 value=copy.deepcopy(real_read(p[key]));mutate(value)
 with patch.object(m,"read",lambda r:value if r==p[key]else real_read(r)):reject(lambda:m.metadata_join(p),label)
# Actual current child proof only, with its I/O, digest/canonical/absence resource boundary replaced in RAM.
tree=ast.parse(Path(p["guard"]["path"]).read_text());env=dict(require=m.need,datetime=datetime,timezone=timezone,Path=Path)
for name in("_pointer","_identity","_completed_identity","_proof"):
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name);exec(compile(ast.Module(body=[fn],type_ignores=[]),"<exact_current_child_"+name+"_RAM>","exec"),env)
def integer(x,label,minimum):
 if type(x)is not int or x<minimum:raise ValueError(label)
env["budget"]=types.SimpleNamespace(digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),v4=types.SimpleNamespace(encoded=lambda value:json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),integer=integer,canonical=lambda root,path:Path(path)if Path(path).is_absolute()else root/path))
env["ROOT"]=m.ROOT
obsref=dict(path="RAM/targeted_observation.json",sha256="1"*64)
stamp=datetime.now(timezone.utc).isoformat()
rows=[dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=stamp)for pid in pids]
obs=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",key=own["key"],reservation_sha256=p["reservation_sha256"],storage_ref=p["storage"],launch_ref=p["launch"],direct_wait_ref=p["wait"],completed_ref=p["completed"],launcher_pid=launch["pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=rows)
env["_json"]=lambda r:copy.deepcopy(obs)if r==obsref else real_read(r)
env["_completed_json"]=real_read
env["_absent"]=lambda pid:copy.deepcopy(next(r for r in rows if r["pid"]==pid))
request=m.request_value(p,own,launch,obsref)
proof=env["_proof"](request,own,observe=False);applied=env["_proof"](request,own,observe=True)
need(proof["pids"]==pids and proof["completion_verified"]is True and applied["PID_observations"]==rows,"Exact current ordinary completed-proof joins actual metadata through RAM absence only")
bad=copy.deepcopy(request);bad["wait_identity"]["returncode"]=1
reject(lambda:env["_proof"](bad,own),"Original ordinary proof rejects wrong wait identity")
bad=copy.deepcopy(request);bad["reservation_sha256"]="0"*64
reject(lambda:env["_proof"](bad,own),"Original ordinary proof rejects changed reservation")
source=(D/"close_ROOT_v1.py").read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=="close_once");fnsource=ast.get_source_segment(source,fn)
need(fnsource.index('directory.mkdir()')<fnsource.index('sp.loader.exec_module(module)')<fnsource.index('module.load_selected_guard_v1')<fnsource.index('g._state_locked'),"Exclusive attempt consumed before actual guard import/configuration/state/probes")
need('g.operation_draft_v1("close_completed"'in fnsource and'g.apply_close_completed_v1('in fnsource and'apply_retire'not in source,"Only original ordinary draft/apply API, no retirement")
need('len(state[3])==47'in source and'len(complete)<=64'in source and'complete==own["consumer_bindings"]'in source,"All47roles/original64cap and original reservation bindings explicitly retained")
producer=Path(p["producer_driver"]["path"]).read_text();n=next(n for n in ast.parse(producer).body if isinstance(n,ast.FunctionDef)and n.name=="owned_execute");body=ast.get_source_segment(producer,n)
need('supervisor_pid=os.getpid()'in body and'known_owned_pids=sorted(tree.known)'in body and'FLARE_phase_CPU_actual_wait_v2'in body,"Pinned actual producer authenticates supervisor and complete known PID roster fields")
need(not any(name.startswith("_ROOT_FLARE_VAL002_exact_current_Census_close")for name in sys.modules),"No actual guard import")
result=dict(schema="FLARE_VAL002_current_Census_close_narrow_author_checks_v1",verdict="AUTHOR_SOURCE_CHECKS_ONLY",positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,genuine_actual_metadata_join=True,actual_current_ordinary_proof_RAM_join=True,guard_import_state_lock_measurement_observation_absence_resource_admission_close_or_Git_calls=False)
(D/"source_checks.result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))

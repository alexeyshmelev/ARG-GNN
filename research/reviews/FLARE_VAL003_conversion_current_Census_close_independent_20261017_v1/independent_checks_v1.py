"""Independent VAL003 successor review, source and immutable metadata only."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import ast,copy,hashlib,json,stat,argparse,contextlib,io
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SRC="research/reviews/FLARE_VAL003_conversion_current_Census_close_source_20261017_v1"
OLD="research/reviews/FLARE_VAL002_conversion_current_Census_close_source_20261017_v1"
OUT=ROOT/"research/reviews/FLARE_VAL003_conversion_current_Census_close_independent_20261017_v1"
verified={}
def raw_ref(record):
 p=ROOT/record["path"];s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=2**20
 data=p.read_bytes();a=p.lstat();key=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 assert key(s)==key(a)and len(data)==s.st_size and hashlib.sha256(data).hexdigest()==record["sha256"]
 verified[record["path"]]=record["sha256"];return data
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def doc(r):return json.loads(raw_ref(r))
handoff=doc(dict(path=SRC+"/source_handoff.v1.json",sha256="710d06dc4911e7d8d27448f4e5b9ff37733d41484bcc1d53d92af507ace0ecee"))
def refs(v):
 if type(v)is dict:
  if set(v)=={"path","sha256"}:yield v
  else:
   for x in v.values():yield from refs(x)
 elif type(v)is list:
  for x in v:yield from refs(x)
for r in refs(handoff):raw_ref(r)
manifest=doc(handoff["source_manifest_ref"])
for r in manifest["source_refs"]:raw_ref(r)
pins=doc(handoff["pins_ref"]);source=raw_ref(handoff["source_ref"]).decode();tree=ast.parse(source)
oldref=dict(path=OLD+"/close_ROOT_v1.py",sha256="73b9ed1fded392502547be6811bcbc14c25b5201abb897b8231d7d097e3ece2d")
oldsource=raw_ref(oldref).decode()
prior=doc(handoff["predecessor_independent_source_review_ref"])
assert prior["verdict"]=="SOURCE_PASS"and prior["independent"]is True and prior["blockers"]==[]and prior["source_ref"]==oldref
checks=[];negative=[]
def passed(n,ok=True):assert ok,n;checks.append(n)
def reject(n,fn):
 try:fn()
 except(ValueError,AssertionError,KeyError,TypeError):negative.append(n);return
 raise AssertionError("Accepted "+n)
normal=source
for old,new in reversed([("VAL002","VAL003"),("val_002","val_003"),("18397","37560"),("899404366","889810840")]):normal=normal.replace(new,old)
passed("entire_source_and_AST_equal_after_exact_four_substitutions",normal==oldsource and ast.dump(ast.parse(normal),include_attributes=False)==ast.dump(ast.parse(oldsource),include_attributes=False))
actual={k:doc(v)for k,v in pins.items()if type(v)is dict and k in("input","wait","completed","outer","launch","storage","original_request")}
def need(ok,m):
 if not ok:raise ValueError(m)
def local(path):
 p=Path(path);return p if p.is_absolute()else ROOT/p
table={pins[k]["path"]:v for k,v in actual.items()}
env=dict(ROOT=ROOT,Path=Path,read=lambda r:copy.deepcopy(table[r["path"]]),raw=lambda r:raw_ref(r),local=local,need=need,json=json,argparse=argparse)
for name in("metadata_join","request_value","main"):
 node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/handoff["source_ref"]["path"])+"#independent_RAM_"+name,"exec"),env)
joined=env["metadata_join"](pins)
inp,wait,done,launch,storage,own,pids,groups=joined
passed("genuine37560outer0_wait0_drain_producer_and_scientific_completion_join",pids==groups==[4175890,4180144]and launch["ownership"]["owned_identities"]==[dict(pid=4180144,start=490611464)]and actual["outer"]["tool_session_id"]==37560)
def encoded(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
passed("exact889810840B_reservation_digest62bindings_entire_packet_scope",hashlib.sha256(encoded(own)).hexdigest()==pins["reservation_sha256"]=="1c805ae3d43a4ee0024343ff3367e76286a61c4fb4887fdd932f53c05bddde8a"and own["reserved_bytes"]==889810840 and len(own["consumer_bindings"])==62 and own["auxiliary_scopes"]==[str(ROOT/inp["packet_directory"])])
selector=doc(pins["selector"]);policy=doc(pins["policy"])
passed("actual_current_Census_child_policy_selector_and_journal_bound",selector["source_ref"]==selector["selector_source_ref"]==policy["source_ref"]==pins["guard"]and selector["policy_ref"]==pins["policy"]and selector["active"]is policy["active"]is True and policy["journal"]==".runtime/storage/Census_observation_two_scope_retention_v1")
historical_path="research/reviews/Census_observation_ROOT_actual_inspect_retention_20261017_v1/fresh_capture.root_v1.json"
historical=doc(ref(historical_path))
passed("historical47roles_subset_original62bindings_no_current_state_claim",len(historical["guard_roles"])==47 and all(r in own["consumer_bindings"]for r in historical["guard_roles"].values()))
with contextlib.redirect_stdout(io.StringIO())as out:
 rc=env["main"]([])
passed("exact_default_main_inert_without_runtime_import",rc==0 and json.loads(out.getvalue())==dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False))
for label,key,field,value in [
 ("outer_tool_substitution_rejects","outer","tool_session_id",18397),
 ("nonzero_direct_wait_rejects","wait","returncode",1),
 ("incomplete_owned_drain_rejects","wait","all_owned_processes_and_groups_drained",False),
 ("reservation_cap_rebind_rejects","storage","reserved_new_bytes",899404366)]:
 old=copy.deepcopy(table[pins[key]["path"]]);table[pins[key]["path"]][field]=value
 reject(label,lambda:env["metadata_join"](pins));table[pins[key]["path"]]=old
# Original exact pure proof functions only, operational boundaries stay RAM.
guardraw=raw_ref(pins["guard"]);gtree=ast.parse(guardraw)
class FixedDatetime(datetime):
 @classmethod
 def now(cls,tz=None):return cls(2100,1,1,tzinfo=timezone.utc)
proofenv=dict(require=need,datetime=FixedDatetime,timezone=timezone,Path=Path,ROOT=ROOT)
for name in("_pointer","_identity","_completed_identity","_proof"):
 node=next(x for x in gtree.body if isinstance(x,ast.FunctionDef)and x.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/pins["guard"]["path"])+"#independent_RAM_"+name,"exec"),proofenv)
v4raw=raw_ref(ref("src/arggnn/project_budget_v4.py"));v4env=dict(json=json)
for name in("encoded","integer"):
 node=next(x for x in ast.parse(v4raw).body if isinstance(x,ast.FunctionDef)and x.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),"<original_v4_pure_"+name+">","exec"),v4env)
budget=SimpleNamespace(digest=lambda x:hashlib.sha256(v4env["encoded"](x)).hexdigest(),v4=SimpleNamespace(encoded=v4env["encoded"],integer=v4env["integer"],canonical=lambda root,path:local(path)))
obsref=dict(path="RAM/VAL003_independent_observation.json",sha256="b"*64)
rows=[dict(pid=n,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc="2001-01-01T00:00:00+00:00")for n in pids]
obs=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",key=own["key"],reservation_sha256=pins["reservation_sha256"],storage_ref=pins["storage"],launch_ref=pins["launch"],direct_wait_ref=pins["wait"],completed_ref=pins["completed"],launcher_pid=launch["pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=rows)
proofenv.update(budget=budget,_json=lambda r:copy.deepcopy(obs)if r==obsref else copy.deepcopy(table[r["path"]]),_completed_json=lambda r:copy.deepcopy(table[r["path"]]))
absent_calls=[]
proofenv["_absent"]=lambda n:(absent_calls.append(n)or copy.deepcopy(rows[pids.index(n)]))
request=env["request_value"](pins,own,launch,obsref)
proof=proofenv["_proof"](request,own,observe=False)
applied=proofenv["_proof"](request,own,observe=True)
passed("exact_unchanged_ordinary_proof_accepts_actual003_metadata_both_RAM_branches",proof["completion_verified"]is True and applied["pids"]==pids and absent_calls==pids)
for label,field,value in [
 ("wrong_reservation_digest_rejects","reservation_sha256","0"*64),
 ("wrong_completed_parent_identity_rejects","completed_identity",dict(status=done["status"],parent_id="american_c3_val_002")),
 ("wrong_direct_wait_identity_rejects","wait_identity",dict(request["wait_identity"],returncode=1))]:
 bad=copy.deepcopy(request);bad[field]=value
 reject(label,lambda bad=bad:proofenv["_proof"](bad,own,observe=False))
oldrow=copy.deepcopy(obs["observations"][0]);obs["observations"][0]["absent"]=False
reject("observed_present_PID_rejects",lambda:proofenv["_proof"](request,own,observe=False));obs["observations"][0]=oldrow
obs["observations"][0]["checked_utc"]="2200-01-01T00:00:00+00:00"
reject("future_observation_rejects",lambda:proofenv["_proof"](request,own,observe=False));obs["observations"][0]=oldrow
fns={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
close=fns["close_once"];body=ast.unparse(close)
passed("exclusive_consumption_before_guard_import_probe_state",body.index("directory.mkdir()")<body.index("spec_from_file_location")<body.index("g._absent")<body.index("g._state_locked"))
passed("one_original_ordinary_apply_and_no_retirement_or_new_guard",body.count("g.apply_close_completed_v1(")==1 and "apply_retire"not in body and "load_selected_guard_v1(p['selector'])"in body)
lock=next(n for n in close.body if isinstance(n,ast.With))
passed("capture_fresh_state_exact47role62binding_union64cap_and_current_prefix",all(x in ast.unparse(lock)for x in("fresh=True","len(state[3]) == 47","complete == own['consumer_bindings']","len(complete) <= 64","expected_before=g._pins(state[2])"))and any(isinstance(n,ast.Assign)and "g.apply_close_completed_v1"in ast.unparse(n)for n in close.body))
apply=next(n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name=="apply_close_completed_v1")
abody=ast.unparse(apply)
passed("original_apply_relocks_replays_fresh_prefix_proof_measure_once_rechecks_absence_then_append",all(x in abody for x in("with budget.v4.mutex(ROOT)","fresh=True","_pins(state[2])","observe=True","budget.v4.measure_scopes","for pid in proof['pids']","_append('close_completed'"))and abody.count("measure_scopes")==1)
passed("original_ordinary_proof_nonfuture_only_no_added120s_or300s_limit","timedelta"not in ast.unparse(next(n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name=="_proof"))and "finite("not in abody)
for r in refs(handoff):raw_ref(r)
result=dict(schema="FLARE_VAL003_ordinary_close_independent_focused_checks_v1",status="SOURCE_CHECKS_PASS",positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,verified_refs=verified,source_manifest_refs=len(manifest["source_refs"]),genuine_metadata_join=True,exact_original_proof_RAM_branches=True,entire_source_and_AST_normalized_VAL002_equal=True,predecessor_broad_campaign_repeated=False,current_state_or_open_status_claimed=False,actual_clock_or_guard_runtime_import_process_group_lock_state_measurement_resource_close_Git_calls=False,actual_run_namespace_inspected=False,actual_credit_accepted_bytes=0)
with(OUT/"checks.result.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:result[k]for k in("status","positive_count","negative_count","source_manifest_refs","actual_credit_accepted_bytes")},sort_keys=True))

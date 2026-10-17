"""Author source/RAM contracts only. No real guard, resource, process, scientific import or payload access."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
import ast,copy,contextlib,hashlib,io,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/H059_saved_outer000_acceptance_and_continuation_source_20261017_v2"
checks=[];negative=[];modules={};docs={};records={};created=set();puts=[];reserve_calls=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(call,label):
 try:call()
 except(ValueError,AssertionError,KeyError,TypeError,PermissionError,OSError):negative.append(label);return
 raise AssertionError("Accepted invalid RAM case: "+label)
def enc(v):return(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def putram(path,value,cap=2**20):
 path=str(path);path=str(PurePosixPath(path).relative_to(str(ROOT)))if path.startswith("/")else path
 raw=enc(value);check(len(raw)<=cap,"RAM compact publication bound")
 r={"path":path,"sha256":hashlib.sha256(raw).hexdigest()};docs[path]=copy.deepcopy(value);records[path]=r;puts.append(path);return r
def physical(path):
 p=ROOT/path
 check(p.suffix in(".py",".json",".md",".sh",".patch"),"Only bounded source/metadata physical read")
 check(not path.startswith("data/")or path.endswith(("/expanded_parent_completed.json","/generation.json")),"No observed data payload physical access")
 raw=p.read_bytes();check(len(raw)<=32*2**20,"Bounded source/metadata")
 r={"path":path,"sha256":hashlib.sha256(raw).hexdigest()};records[path]=r
 if p.suffix==".json":docs[path]=json.loads(raw)
 return r
def ref(path):
 path=str(path);path=str(PurePosixPath(path).relative_to(str(ROOT)))if path.startswith("/")else path
 if path not in records:
  if (ROOT/path).is_file():return physical(path)
  records[path]={"path":path,"sha256":hashlib.sha256(path.encode()).hexdigest()}
 return copy.deepcopy(records[path])
def raw(r):
 if r["path"].endswith("/experiment_ledger.snapshot.md"):
  check(physical(r["path"])==r,"Exact frozen ledger source bytes")
  return (ROOT/r["path"]).read_bytes()
 check(records.get(r["path"],r)==r,"Exact RAM bound SHA")
 return enc(docs[r["path"]])if r["path"]in docs else b"DECLARED_AUTHENTICATED_SOURCE_RAM_IO_DOUBLE"
def read(r):
 check(ref(r["path"])==r,"Exact decoded metadata ref")
 if r["path"]not in docs:physical(r["path"])
 return copy.deepcopy(docs[r["path"]])
class RP(PurePosixPath):
 def resolve(self):return self
 def is_dir(self):return str(self)in created
 def exists(self):return str(self)in created or str(self.relative_to(str(ROOT)))in docs
 def mkdir(self,**kwargs):
  check(not self.exists(),"Fresh RAM namespace");created.add(str(self))
 @classmethod
 def cwd(cls):return cls(ROOT)
class Imports:
 def import_module(self,name):
  if name==str(D.relative_to(ROOT)).replace("/",".")+".saved_outer000_v1":return types.SimpleNamespace(**saved)
  raise AssertionError("Forbidden runtime import "+name)
def module(name):
 ns={"__name__":"source_RAM_"+name};text=(D/name).read_text();exec(compile(text,"<complete_stdlib_source_RAM>","exec"),ns)
 modules[name]=ns;return ns
for p in D.glob("*.py"):ast.parse(p.read_text());check(True,"AST "+p.name)
saved=module("saved_outer000_v1.py");prediction=module("prediction_stage_v1.py")
# Metadata-only assessment of actual root report/completion/config/generation. No inferred tree or observation data.
real_metadata=saved["metadata_v1"](read)
check(real_metadata[0]["parent_id"]=="ap_expanded_outer_000","Genuine actual outer000 metadata joins")
check(saved["HISTORY"]["original_worker_OS_wait"]=="UNKNOWN"and saved["HISTORY"]["original_owned_descendant_drain"]=="UNKNOWN","Original unknown histories remain exact")
# Complete synthetic dedicated-validator acceptance, explicitly RAM_NOT_AUTHORITY only.
S=saved["HERE"];driver=ref(S+"/validation_driver_v1.py");manifest=ref(S+"/source_manifest.v1.json")
review=putram("RAM_NOT_AUTHORITY/review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],
 saved_observed_source_ref=ref(saved["SOURCE"]),validator_driver_source_ref=driver,source_manifest_ref=manifest,
 generation_source_refs=[ref(S+"/prepare_outer%03d_v1.py"%i)for i in range(1,8)],source_handoff_ref=ref(S+"/source_handoff.v1.json")))
validator_input=putram("RAM_NOT_AUTHORITY/validator_input.json",dict(schema="H059_saved_outer000_validator_admission_v1",
 source_ref=driver,operational_source_ref=driver,operational_manifest_ref=manifest))
result=putram("RAM_NOT_AUTHORITY/validator_result.json",dict(schema="H059_saved_outer000_observed_validation_result_v1",
 status="VALIDATED_SAVED_OBSERVED_ONLY",original_history=saved["HISTORY"],completion_ref=saved["COMPLETION"],
 full952_roster_686queries_and_original_clipped_grid_validated=True,private_truth_opened=False,
 original_scope_completion_or_close_or_credit_authorized=False,original_guard_monitoring_interval_recovered=False,
 known_absence_before={"complete_historical_descendant_absence_claimed":False},
 known_absence_after={"complete_historical_descendant_absence_claimed":False}))
completion=putram("RAM_NOT_AUTHORITY/validator_completion.json",dict(schema="H059_saved_outer000_validator_completion_v1",
 input_ref=validator_input,driver_source_ref=driver,validation_result_ref=result,original_history=saved["HISTORY"]))
wait=putram("RAM_NOT_AUTHORITY/validator_wait.json",dict(schema="H059_saved_outer000_validator_actual_wait_v1",
 returncode=0,actual_OS_wait_status=0,actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,
 complete_owned_roster_known=True,operational_failure=None,input_ref=validator_input,completion_ref=completion,driver_source_ref=driver))
terminal=putram("RAM_NOT_AUTHORITY/validator_terminal.json",dict(root_authorizer="root_agent",actual_tool_terminal_consumed=True,
 actual_tool_exit_code=0,actual_wait_ref=wait,input_ref=validator_input,validation_result_ref=result,completion_ref=completion))
rev=putram("RAM_NOT_AUTHORITY/old_revocation.json",dict(root_authorizer="root_agent",active=True,original_input_ref=saved["INPUT"],
 all_original_launch_resume_retry_write_routes_permanently_cooperatively_revoked=True,
 all_original_files_charges_clocks_and_failures_preserved=True,read_only_artifact_reuse_preserved=True,revoked_at_utc=(datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat()))
docs[validator_input["path"]]["root_request_ref"]=putram("RAM_NOT_AUTHORITY/acceptance_validator_request.json",dict(old_writer_revocation_ref=rev))
docs[result["path"]]["old_writer_revocation_ref"]=rev
accept=putram("RAM_NOT_AUTHORITY/accepted_observed.json",dict(schema="H059_saved_outer000_ROOT_artifact_acceptance_v1",
 active=True,root_authorizer="root_agent",accepted_for_observed_inference_only=True,original_history=saved["HISTORY"],
 completion_ref=saved["COMPLETION"],source_ref=ref(saved["SOURCE"]),source_manifest_ref=manifest,
 ordinary_close_or_credit_authorized=False,source_review_ref=review,validation_result_ref=result,validator_wait_ref=wait,
 validator_ROOT_terminal_ref=terminal,validator_input_ref=validator_input,validator_completion_ref=completion,old_writer_revocation_ref=rev))
fixture_now=datetime.now(timezone.utc)
fixture_qual=putram("RAM_NOT_AUTHORITY/acceptance_scope_qualification.json",dict(observed_at_utc=fixture_now.isoformat()))
fixture_request=docs[docs[validator_input["path"]]["root_request_ref"]["path"]]
fixture_request.update(created_at_utc=(fixture_now-timedelta(seconds=20)).isoformat(),original_readonly_scope_qualification_ref=fixture_qual)
fixture_pf=dict(schema="H059_original_readonly_scope_admission_preflight_v2",root_authorizer="root_agent",status="PASS",
 initial_admission_preflight_only=True,checked_at_utc=fixture_now.isoformat(),request_ref=docs[validator_input["path"]]["root_request_ref"],
 qualification_ref=fixture_qual,old_writer_revocation_ref=rev,qualification_observed_at_utc=fixture_now.isoformat(),
 revoked_at_utc=read(rev)["revoked_at_utc"],driver_source_ref=driver,qualification_max_age_seconds=120)
docs[validator_input["path"]]["admission_preflight_v2"]=fixture_pf
docs[result["path"]].update(admission_preflight_v2=fixture_pf,original_readonly_scope_qualification_ref=fixture_qual)
check(saved["acceptance_values_v1"](accept,read,ref)["original_history"]==saved["HISTORY"],"Actual full separate acceptance predicate passes RAM evidence chain")
# Metadata direct union and original concrete scope_spec; no actual guard import or state.
binding=read(ref("research/reviews/H059_outer000_Relate64_execution_source_20261017_v1/source_bindings.v1.json"))
guards=read(binding["guard_role_snapshot_ref"])["bindings"]
guards=copy.deepcopy(guards);guards["Census_observation_explicit_selector_capsule"]=ref("RAM_NOT_AUTHORITY/Census_selector.json")
consumer_tree=ast.parse((ROOT/"research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py").read_text())
consumer_ns={"copy":copy,"require":lambda ok,msg:check(ok,msg),"_raw":raw}
node=next(n for n in consumer_tree.body if isinstance(n,ast.FunctionDef)and n.name=="consumer_bindings_v1")
exec(compile(ast.Module(body=[node],type_ignores=[]),"<actual_Census_direct_ref_union_RAM>","exec"),consumer_ns)
def integer(v,label,minimum=0):check(type(v)is int and v>=minimum,"RAM original integer");return v
v4tree=ast.parse((ROOT/"src/arggnn/project_budget_v4.py").read_text())
scope_ns=dict(Path=RP,integer=integer,canonical=lambda root,p:RP(str(root)+"/"+str(p))if not RP(p).is_absolute()else RP(p),
 paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,accounting_paths=lambda root:[],bound=lambda root,r:raw(r))
node=next(n for n in v4tree.body if isinstance(n,ast.FunctionDef)and n.name=="scope_spec")
exec(compile(ast.Module(body=[node],type_ignores=[]),"<actual_v4_scope_spec_RAM>","exec"),scope_ns)
now=datetime.now(timezone.utc)
selected_ref=guards["Census_observation_explicit_selector_capsule"]
def run_admission(name,validator=False):
 ns=module(name)
 check(ns["HERE"]==str(D.relative_to(ROOT))and ns["SOURCE"]==str((D/name).relative_to(ROOT)),"Actual coherent current package/source identity "+name)
 ns.update(ROOT=RP(ROOT),Path=RP,read=read,raw=raw,ref=ref,put=putram,importlib=Imports())
 ns["os"]=types.SimpleNamespace(sched_getaffinity=lambda pid:set(range(8)))
 q=json.loads((D/("ROOT_saved_validator.INACTIVE.template.v1.json"if validator else "ROOT_outer"+name[13:16]+".INACTIVE.template.v1.json")).read_text())
 q.update(status="active",root_authorizer="root_agent",created_at_utc=(now-timedelta(seconds=5)).isoformat(),
 expires_at_utc=(now+timedelta(hours=24)).isoformat(),cpu_ids=[0]if validator else list(range(8)),
 source_manifest_ref=manifest,source_review_ref=review,policy_selector_ref=selected_ref,automatic_retry_or_reseed=False,prior_preparation_routes_cooperatively_revoked=True)
 q["method_freeze_ref"]=binding["frozen_method_freeze_ref"]
 rt=putram("RAM_NOT_AUTHORITY/"+name+"_runtime.json",dict(schema="European_outer"+("000"if validator else name[13:16])+"_CPU_runtime_v1",
 device="cpu",CUDA_VISIBLE_DEVICES="",interpreter=str(ROOT/".runtime/env/bin/python"),config_runtime=real_metadata[2]["runtime"],native_threads=8,cpu_ids=q["cpu_ids"]))
 q["runtime_ref"]=rt
 if validator:
  q["saved_observed_validation_authorized"]=True;q["generation_or_Relate_rerun_authorized"]=False;q["old_writer_revocation_ref"]=rev
  qr=dict(root_authorizer="root_agent",status="qualified",unresolved_items=[],source_manifest_ref=manifest,runtime_ref=rt,device="cpu",CPU_threads=1,CUDA_VISIBLE_DEVICES="")
  q["original_readonly_scope_qualification_ref"]=putram("RAM_NOT_AUTHORITY/old_scope_qualification.json",dict(root_authorizer="root_agent",status="PASS",
   original_input_ref=saved["INPUT"],original_whole_reservation_bytes=4294967296,complete_original_output_and_packet_measured=True,
   old_source_guard_caps_and_clock_unchanged=True,original_charge_retained=True,historical_monitoring_gap_recovered=False,
   observed_at_utc=now.isoformat(),old_writer_revocation_ref=rev,original_storage_ref=ref("research/reviews/European_outer000_ROOT_actual_Relate64_20261017_v1/storage.json")))
 else:
  q["generation_and_native_Relate_authorized"]=True;q["saved_outer000_acceptance_ref"]=accept
  b=read(ref(str((D/("source_bindings.outer"+name[13:16]+".v1.json")).relative_to(ROOT))))
  qr=dict(schema="European_outer"+name[13:16]+"_runtime_qualification_v1",root_authorizer="root_agent",status="qualified",runtime_ref=rt,
   operational_manifest_ref=manifest,config_ref=ns["CONFIG"],producer_ref=ns["PRODUCER"],unresolved_items=[],
   inherited_runtime_qualification_ref=b["inherited_runtime_qualification_ref"],new_source_review_ref=review,runtime_compatibility_only_no_new_numerical_probe=True)
 q["runtime_qualification_ref"]=putram("RAM_NOT_AUTHORITY/"+name+"_runtime_qualification.json",qr)
 qref=putram(q["preparation_directory"]+"/ROOT_request.json",q);created.add(str(ROOT/q["preparation_directory"]))
 polref=ref("RAM_NOT_AUTHORITY/Census_policy.json")
 chosen=dict(schema="explicit_ROOT_storage_policy_selector_v1",active=True,root_authorizer="root_agent",selector_source_ref=ns["SELECTOR"],
 source_ref=ns["SELECTOR"],source_manifest_ref=ns["CENSUS_MANIFEST"],source_review_ref=ns["CENSUS_REVIEW"],policy_ref=polref,
 measurement_source_ref=ref("RAM_NOT_AUTHORITY/measure.py"),repair_authority_ref=ref("RAM_NOT_AUTHORITY/repair.json"),
 accounting_mode="Census_observation_two_scope_retention_v1",no_fallback_discovery_upgrade_or_in_run_swap=True,future_ROOT_adapters_only=True,
 scheduling_ceiling_bytes=980000000000,human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000)
 docs[selected_ref["path"]]=chosen;docs[polref["path"]]=dict(active=True,source_ref=ns["SELECTOR"],source_manifest_ref=ns["CENSUS_MANIFEST"],source_review_ref=ns["CENSUS_REVIEW"])
 class Budget:
  SOURCE_REL="scripts/project_budget_v9.py";LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py"
  v4=types.SimpleNamespace(scope_spec=scope_ns["scope_spec"])
 @contextlib.contextmanager
 def mutex(root):yield
 Budget.v4.mutex=mutex
 view={"base_pins":{"RAM":"original_frozen"},"completed_overlay_prefix":{"events":0,"tail_sha256":None}}
 advance=[0]
 if validator:
  class ClockDateTime:
   fromisoformat=staticmethod(datetime.fromisoformat)
   @staticmethod
   def now(tz):return now+timedelta(seconds=advance[0])
  ns["datetime"]=ClockDateTime
 def reserve(ar,**kwargs):
  auth=read(ar);reserve_calls.append(auth)
  if validator:advance[0]+=900
  return dict(value=dict(auth["request"],root_admission=ar),initial_ledger_pins=view)
 g=types.SimpleNamespace(budget=Budget(),MODE=chosen["accounting_mode"],_POLICY_REF=polref,
 _state_locked=lambda *a,**k:({},set(),copy.deepcopy(view),copy.deepcopy(guards),[],[],{}),
 consumer_bindings_v1=consumer_ns["consumer_bindings_v1"],_policy=lambda:{"never_credit_keys":[]},
 operation_draft_v1=lambda op,own,**kw:dict(operation=op,request=own,**kw),_pins=lambda v:copy.deepcopy(v),apply_reserve_v1=reserve)
 ns["guard"]=lambda selected,consumer:g
 # All actual candidate request/input/admit predicates execute; only lower-layer guard/accounting IO is a RAM boundary.
 answer=ns["admit"](qref);spec,q2,cfg=ns["input_spec"](answer["input_ref"])
 check(spec["parent_id"]==ns["PARENT"]and spec["storage"]["reserved_new_bytes"]==ns["CAPS"]["reservation_bytes"],"Complete actual request/admit/input join "+name)
 check(answer["execution_argv"][-3]==ns["HERE"].replace("/",".")+"."+name[:-3],"Exact current child module argv "+name)
 check(len(spec["storage"]["reservation"]["consumer_bindings"])<=64,"Original64 ceiling actual source-derived union "+name)
 return ns,qref,answer,len(spec["storage"]["reservation"]["consumer_bindings"])
answers=[]
for i in range(1,8):answers.append(run_admission("prepare_outer%03d_v1.py"%i))
answers.append(run_admission("validation_driver_v1.py",True))

check(len(reserve_calls)==8,"Eight actual entry joins with RAM accounting only")
validator_ns,validator_qref,validator_answer,_=answers[-1]
spec,q,cfg=validator_ns["input_spec"](validator_answer["input_ref"])
preflight=spec["admission_preflight_v2"]
check(preflight["checked_at_utc"]!=validator_ns["datetime"].now(timezone.utc).isoformat(),"Reserve advanced 900s after actual initial preflight")
qualification=read(q["original_readonly_scope_qualification_ref"]);revocation=read(q["old_writer_revocation_ref"])
future=datetime.fromisoformat(preflight["checked_at_utc"])+timedelta(hours=2)
check(saved["admission_preflight_values_v2"](preflight,q,validator_qref,qualification,revocation,ref(validator_ns["SOURCE"]),now=future)==preflight,"Immutable <=120s relation survives two hours later")
def reject_pf(value,qual=qualification,revoke=revocation,at=future):
 reject(lambda:saved["admission_preflight_values_v2"](value,q,validator_qref,qual,revoke,ref(validator_ns["SOURCE"]),now=at),"Focused invalid preflight")
for key,value in[("qualification_max_age_seconds",True),("request_ref",wait),("checked_at_utc",(datetime.fromisoformat(qualification["observed_at_utc"])+timedelta(seconds=121)).isoformat()),("checked_at_utc",(future+timedelta(seconds=1)).isoformat())]:
 bad=copy.deepcopy(preflight);bad[key]=value;reject_pf(bad)
oldqual=copy.deepcopy(qualification);oldqual["observed_at_utc"]=(datetime.now(timezone.utc)-timedelta(seconds=121)).isoformat()
oldrev=copy.deepcopy(revocation);oldrev["revoked_at_utc"]=(datetime.now(timezone.utc)-timedelta(seconds=300)).isoformat()
bad=copy.deepcopy(preflight);bad.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),qualification_observed_at_utc=oldqual["observed_at_utc"],revoked_at_utc=oldrev["revoked_at_utc"]);reject_pf(bad,oldqual,oldrev,datetime.now(timezone.utc))
# Actual read-only callback, with no real resource function, scope walk or clock.
calls=[];validator_ns["time"]=types.SimpleNamespace(monotonic=lambda:100.)
validator_ns["sample_scope"]=lambda value:calls.append("scope")
validator_ns["resource"]=types.SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda who:types.SimpleNamespace(ru_maxrss=1))
for _ in range(4):validator_ns["readonly_boundary_v2"](spec,101.)
check(calls==["scope"]*4,"Each read-only chunk checks scope without accounting")
reject(lambda:validator_ns["readonly_boundary_v2"](spec,100.),"Read-only deadline expires at boundary")
# Small exact AST/source delta checks; no duplicate prediction campaign.
get=lambda text,name:next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
dump=lambda n:ast.dump(n,include_attributes=False)
oldD=ROOT/"research/reviews/H059_saved_outer000_acceptance_and_continuation_source_20261017_v1"
for i in range(1,8):
 name="prepare_outer%03d_v1.py"%i
 check((D/name).read_text().replace(str(D.relative_to(ROOT)),str(oldD.relative_to(ROOT)))==(oldD/name).read_text(),"Generation source unchanged except package identity "+str(i))
check((D/"prediction_stage_v1.py").read_text().replace(str(D.relative_to(ROOT)),str(oldD.relative_to(ROOT)))==(oldD/"prediction_stage_v1.py").read_text(),"Frozen five methods/40+8 barrier unchanged")
old_saved=(oldD/"saved_outer000_v1.py").read_text();new_saved=(D/"saved_outer000_v1.py").read_text()
for name in["stream_observed_v1","validate_saved_v1","known_absence_v1","validate_absence_values_v1","metadata_v1","native_metadata_v1"]:
 check(dump(get(old_saved,name))==dump(get(new_saved,name)),"Hashing/identity/absence/science unchanged "+name)
new_driver=(D/"validation_driver_v1.py").read_text();old_driver=(oldD/"validation_driver_v1.py").read_text()
for name in["boundary","sample_scope","rss_identity_v1"]:check(dump(get(old_driver,name))==dump(get(new_driver,name)),"Write guard/scope/lifecycle unchanged "+name)
work=get(new_driver,"worker")
call=next(n for n in ast.walk(work)if isinstance(n,ast.Call)and ast.unparse(n.func)=="saved.validate_saved_v1")
check("readonly_boundary_v2"in ast.unparse(call.args[2])and ast.unparse(call.keywords[0].value)=="CAPS['validation_work_seconds']","Actual worker supplies cheap callback and600s work")
check(validator_ns["CAPS"]["validation_work_seconds"]==600 and validator_ns["CAPS"]["cleanup_seconds"]==30 and validator_ns["CAPS"]["reservation_bytes"]==32*2**20,"Science/cleanup/charge unchanged")
write_calls=[]
validator_ns["boundary"]=lambda *a,**k:write_calls.append("full")
validator_ns["write_boundary_v2"](None,spec,deadline=101.)
check(write_calls==["full"],"Actual write wrapper retains full accounting")
write_clock=[99.]
validator_ns["time"]=types.SimpleNamespace(monotonic=lambda:write_clock[0])
def slow_write_guard(*a,**k):write_clock[0]=102.
validator_ns["boundary"]=slow_write_guard
reject(lambda:validator_ns["write_boundary_v2"](None,spec,deadline=101.),"Slow full accounting cannot silently overrun write window")
summary=dict(schema="H059_v2_focused_timing_and_eight_metadata_joins",source_only=True,assertions=len(checks),negative_fixtures=len(negative),actual_entry_joins=8,generation_union_counts=[x[3]for x in answers[:7]],validator_union_count=answers[-1][3],slow_reserve_advance_seconds=900,later_historical_relation_advance_seconds=7200,actual_guard_state_process_payload_or_numerical_calls=False)
(D/"source_check.result.v2.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2))

"""Author source/RAM contracts only. No real guard, resource, process, scientific import or payload access."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
import ast,copy,contextlib,hashlib,io,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/H059_saved_outer000_acceptance_and_continuation_source_20261017_v1"
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
check(saved["acceptance_values_v1"](accept,read,ref)["original_history"]==saved["HISTORY"],"Actual full separate acceptance predicate passes RAM evidence chain")
def mutate(r,key,value,call,label):
 old=copy.deepcopy(docs[r["path"]]);docs[r["path"]][key]=value
 try:reject(call,label)
 finally:docs[r["path"]]=old
for r,key,value in [(wait,"returncode",False),(wait,"returncode",1),(wait,"actual_direct_worker_wait",False),
 (wait,"driver_source_ref",ref(S+"/saved_outer000_v1.py")),(terminal,"actual_tool_terminal_consumed",False),
 (terminal,"actual_tool_exit_code",1),(result,"private_truth_opened",True),(accept,"ordinary_close_or_credit_authorized",True),
 (rev,"all_original_launch_resume_retry_write_routes_permanently_cooperatively_revoked",False),
 (accept,"original_history",{}),(completion,"validation_result_ref",wait)]:
 mutate(r,key,value,lambda:saved["acceptance_values_v1"](accept,read,ref),"Reject acceptance:"+key)
# Native receipt predicates with bounded RAM metadata, never actual inference/entry/arrays.
cp=real_metadata[0];infr=cp["inference_ref"];entryr=cp["observation_entry_ref"];cmd=ref("RAM_NOT_AUTHORITY/commands.json")
steps=[dict(label=l,argv=[l],command_sha256=saved["digest"]([l]),returncode=0,status="complete")for l in saved["LABELS"]]
docs[cmd["path"]]={"steps":steps}
native_tree=ref("RAM_NOT_AUTHORITY/observed_native.trees")
docs[infr["path"]]=dict(parent_id=saved["PARENT"],split="outer",status="post_five_iterations_inferred",
 truth_or_oracle_read=False,outer_command_returncodes=[0]*5,commands=cmd,inferred_tree=native_tree,
 native_status_scope="outer helper returncodes; bash-x trace retained; child statuses not independently asserted")
docs[entryr["path"]]=dict(schema="mutation_routed_full_panel_observed_entry_v1",parent_id=saved["PARENT"],split="outer",
 observations_only=True,truth_or_oracle_read=False,samples=952,query_haplotypes=686,pure_reference_haplotypes=266,
 original_native_inference_receipt=infr,refs={"native_tree":native_tree})
records[infr["path"]]=infr;records[entryr["path"]]=entryr
check(saved["native_metadata_v1"](cp,read)[2]==cmd,"Full native metadata predicate accepts five RAM outer helper receipts")
for key,value in [("outer_command_returncodes",[0]*4),("native_status_scope","all descendants0"),("truth_or_oracle_read",True)]:
 mutate(infr,key,value,lambda:saved["native_metadata_v1"](cp,read),"Reject native receipt:"+key)
old=copy.deepcopy(docs[cmd["path"]]);docs[cmd["path"]]["steps"][4]["returncode"]=False
reject(lambda:saved["native_metadata_v1"](cp,read),"Boolean helper returncode not genuine integer0");docs[cmd["path"]]=old
# Known-target observation function executes solely against numeric ESRCH IO double.
class AbsentOS:
 def kill(self,pid,sig):raise ProcessLookupError()
 def killpg(self,pid,sig):raise ProcessLookupError()
saved["os"]=AbsentOS()
check(saved["validate_absence_values_v1"](saved["known_absence_v1"]())["complete_historical_descendant_absence_claimed"]is False,"Known numeric absence preserves unknown history")
class PresentOS(AbsentOS):
 def kill(self,pid,sig):return
saved["os"]=PresentOS();reject(saved["known_absence_v1"],"Present or reused numeric PID blocks");saved["os"]=AbsentOS()
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
 def reserve(ar,**kwargs):
  auth=read(ar);reserve_calls.append(auth);return dict(value=dict(auth["request"],root_admission=ar),initial_ledger_pins=view)
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
check(len(reserve_calls)==8,"Eight RAM-only separate one-shot admissions; no loop launches")
validator_ns,validator_qref,_,_=answers[-1]
mutate(rev,"all_original_launch_resume_retry_write_routes_permanently_cooperatively_revoked",False,
 lambda:validator_ns["request"](validator_qref),"Old-writer revocation required before dedicated validator admission")
readonly_ref=read(validator_qref)["original_readonly_scope_qualification_ref"]
mutate(readonly_ref,"observed_at_utc",(now-timedelta(seconds=121)).isoformat(),
 lambda:validator_ns["request"](validator_qref),"Fresh old-scope read-only proof required before validator")
mutate(readonly_ref,"old_writer_revocation_ref",wait,
 lambda:validator_ns["request"](validator_qref),"Old-scope qualification binds exact revocation")
# Source execution/worker bodies and caps invariants; source-only no numerical execution.
old_source=(ROOT/"research/reviews/H059_outer000_Relate64_execution_source_20261017_v1/prepare_outer000_v1.py").read_text()
get=lambda text,name:next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
for i in range(1,8):
 text=(D/("prepare_outer%03d_v1.py"%i)).read_text()
 normalized=text.replace("outer%03d"%i,"outer000").replace("outer_%03d"%i,"outer_000")
 check(ast.dump(get(normalized,"worker"),include_attributes=False)==ast.dump(get(old_source,"worker"),include_attributes=False),"Entire unchanged original scientific worker "+str(i))
 check("len(tree.known)<=64"not in text and "range(64)"not in text and "[:64]"not in text,"No stale cumulative64 lifecycle operand "+str(i))
 check("OWNED_ROSTER_CAP=4096"in text and 'CAPS["cleanup_seconds"]'in text,"Finite4096 lifecycle and original30s cleanup "+str(i))
 check(answers[i-1][0]["CAPS"]==dict(reservation_bytes=4294967296,output_bytes=4000000000,auxiliary_bytes=134217728,
 cpu_threads=8,rss_bytes=34359738368,wall_seconds=43200,cleanup_seconds=30,work_reserve_seconds=300,evidence_reserve_seconds=150,log_bytes=2097152,root_packet_bytes=8388608),"Exact original whole generation cap "+str(i))
# Exact full48 product plan and canonical original namespace, no shortened readout/fits.
plan=prediction["stage_plan_v1"]("root01")
check(len(plan)==64 and sum(x["primary_product_count"]for x in plan)==40 and sum(x["raw_top5_product_count"]for x in plan)==8,"Full64 stages yield exact40+8 pretruth products")
for row in plan:
 check(row["output_directory"]=="experiments/results/European_outer8_predictions_root01/cells/"+row["method_id"]+"/"+row["parent_id"],"Canonical actual v3 output "+row["method_id"]+":"+row["stage"])
for bad in("","_bad","a"*65,"/escape"):reject(lambda bad=bad:prediction["stage_plan_v1"](bad),"Invalid original run identity")
# Actual RSS helper with IO fixtures: positive mismatches fail; disappearing rows only return0.
rss=answers[0][0]["rss_identity_v1"];rn=rss.__globals__;real_path=rn["Path"];real_os=rn["os"]
identity={"pid":7,"start":9}
class StatPath:
 def __init__(self,p):pass
 def read_text(self):return "1 3"
rn["Path"]=StatPath;rn["os"]=types.SimpleNamespace(sysconf=lambda k:4096)
def rss_case(rows):
 it=iter(rows);owned=types.SimpleNamespace(_stat=lambda pid:next(it),_same=lambda a,b:a==b)
 return rss(owned,identity)
check(rss_case([None])==0 and rss_case([identity,None])==0 and rss_case([identity,identity])==12288,"RSS disappearance and stable observation")
reject(lambda:rss_case([{"pid":7,"start":10}]),"RSS positive pre PID/start mismatch")
reject(lambda:rss_case([identity,{"pid":7,"start":10}]),"RSS positive post PID/start mismatch")
rn["Path"]=real_path;rn["os"]=real_os

# Complete later builder requests for every original stage, with RAM-only future prerequisites.
docs[review["path"]]["prediction_stage_source_ref"]=ref(S+"/prediction_stage_v1.py")
data_evidence={}
for i,parent in enumerate(prediction["PARENTS"]):
 if i==0:data_evidence[parent]={"saved_observed_acceptance_ref":accept};continue
 ns,qref,answer,count=answers[i-1];ir=answer["input_ref"]
 cr=putram("data/expanded_full_panel_v1/"+parent+"/expanded_parent_completed.json",dict(schema="expanded_full_panel_v1_parent_completion",
  parent_id=parent,split="outer",generation_completed=True,Relate_completed=True,outer_truth_evaluated=False,models_prepared=False,
  observation_entry_ref=ref("RAM_NOT_AUTHORITY/"+parent+"_entry.json")))
 wr=putram("RAM_NOT_AUTHORITY/"+parent+"_generation_wait.json",dict(returncode=0,actual_direct_worker_wait=True,actual_OS_wait_status=0,
  all_owned_processes_and_groups_drained=True,operational_failure=None,input_ref=ir,completion_ref=cr,driver_source_ref=ref(ns["SOURCE"])))
 tr=putram("RAM_NOT_AUTHORITY/"+parent+"_generation_terminal.json",dict(root_authorizer="root_agent",actual_tool_terminal_consumed=True,
  actual_tool_exit_code=0,actual_wait_ref=wr,input_ref=ir,completion_ref=cr))
 data_evidence[parent]=dict(actual_wait_ref=wr,ROOT_terminal_ref=tr,driver_source_ref=ref(ns["SOURCE"]),driver_review_ref=review,input_ref=ir,completion_ref=cr)
check(len(prediction["all_data_ready_v1"](data_evidence,read,ref))==8,"Actual full eight-parent data readiness including saved000 accepted separately")
class PredictionImports:
 def import_module(self,name):
  if name==S.replace("/",".")+".saved_outer000_v1":return types.SimpleNamespace(**saved)
  if name==S.replace("/",".")+".prepare_outer001_v1":return types.SimpleNamespace(**answers[0][0])
  if name=="research.reviews.European_outer8_preparation_source_20261016_v3.outer_adapters_v2":return api
  raise AssertionError("Forbidden actual scientific import "+name)
prediction["importlib"]=PredictionImports()
actual_science_manifest=ref(prediction["SCIENCE"]+"/source_manifest.v1.json")
adapter_ref=ref(prediction["SCIENCE"]+"/outer_adapters_v2.py")
scientific_review=putram("RAM_NOT_AUTHORITY/future_actual_adapter_review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],source_ref=adapter_ref))
inventory=read(ref(prediction["SCIENCE"]+"/frozen_method_ids.v1.json"))
qualifications={}
for method in prediction["METHODS"]:
 qualifications[method]={}
 for stage in(("observe","prepare","fit","predict")if method==prediction["METHODS"][-1]else("predict",)):
  drv=ref("RAM_NOT_AUTHORITY/future_"+method+"_"+stage+"_driver.py")
  drvr=putram("RAM_NOT_AUTHORITY/"+method+"_"+stage+"_driver_review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],source_ref=drv))
  qual=dict(root_authorizer="root_agent",status="qualified",unresolved_items=[],method_id=method,stage=stage,
   source_manifest_ref=actual_science_manifest,operational_integration_manifest_ref=manifest,exact_scientific_adapter_ref=adapter_ref,
   full952_observed_samples=952,query_haplotypes=686,private_truth_used=False,actual_driver_source_ref=drv,actual_driver_review_ref=drvr)
  if method in prediction["METHODS"][:2]:qual["full952_observed_H050_bank_and_cut_recipes_checked"]=True
  if method!=prediction["METHODS"][-1]:qual["fixed_checkpoint_ref"]=inventory["methods"][method]["checkpoint_ref"]
  else:qual.update(full952_native_tokenizer_and_transfer_qualified=True,MLM_epochs=5,MLM_logical_batch=512,MLM_seed=17,MLM_precision="amp_bf16",
   contrastive_epochs=1,contrastive_logical_batch=4096,contrastive_seed=36,contrastive_precision="fp32",reduced_batch_or_gradient_cache=False)
  qualifications[method][stage]=putram("RAM_NOT_AUTHORITY/"+method+"_"+stage+"_qualification.json",qual)
previous={}
for parent in prediction["PARENTS"]:
 method=prediction["METHODS"][-1];previous[parent]={method:{}}
 for stage in("observe","prepare","fit"):
  qual=read(qualifications[method][stage]);drv=qual["actual_driver_source_ref"]
  obj=dict(parent_id=parent,input_samples=952,query_haplotypes=686,reference_haplotypes=266,true_query_ancestry_read=False,
   MLM_optimizer_updates=100,contrastive_optimizer_updates=12)
  if stage=="fit":
   obj.update(fresh_same_ARG_fit=True,old_checkpoint_reused=False,query_truth_used=False)
   for phase,epochs,batch,seed,precision,updates in[("MLM",5,512,17,"amp_bf16",100),("contrastive",1,4096,36,"fp32",12)]:
    obj[phase]=dict(epochs=epochs,logical_batch=batch,seed=seed,precision=precision,optimizer_updates=updates,
     same_ARG_only=True,query_truth_used=False,source_batch_reduced=False,gradient_cache_used=False,
     checkpoint_ref=ref("RAM_NOT_AUTHORITY/"+parent+"/"+phase+"/composer_final.pt"))
  cr=putram("RAM_NOT_AUTHORITY/"+parent+"_"+stage+"_native_completion.json",obj)
  ir=putram("RAM_NOT_AUTHORITY/"+parent+"_"+stage+"_native_input.json",dict(parent_id=parent,method_id=method,stage=stage,separately_admitted_driver_source_ref=drv))
  wr=putram("RAM_NOT_AUTHORITY/"+parent+"_"+stage+"_native_wait.json",dict(returncode=0,actual_direct_worker_wait=True,actual_OS_wait_status=0,
   all_owned_processes_and_groups_drained=True,operational_failure=None,input_ref=ir,completion_ref=cr,driver_source_ref=drv))
  tr=putram("RAM_NOT_AUTHORITY/"+parent+"_"+stage+"_native_terminal.json",dict(root_authorizer="root_agent",actual_tool_terminal_consumed=True,
   actual_tool_exit_code=0,actual_wait_ref=wr,input_ref=ir,completion_ref=cr))
  previous[parent][method][stage]=dict(actual_wait_ref=wr,ROOT_terminal_ref=tr,driver_source_ref=drv,driver_review_ref=qual["actual_driver_review_ref"],input_ref=ir,completion_ref=cr)
intent=dict(schema="H059_later_prediction_ROOT_intent_v1",root_authorizer="root_agent",active=True,
 outer_truth_access_authorized=False,source_manifest_ref=manifest,source_review_ref=review,run_id="root01",
 expires_at_utc=(now+timedelta(hours=24)).isoformat(),policy_selector_ref=selected_ref,method_freeze_ref=binding["frozen_method_freeze_ref"],
 data_evidence_by_parent=data_evidence,actual_stage_qualification_refs=qualifications,scientific_adapter_review_ref=scientific_review,
 stage_evidence_by_parent=previous,actual_scorer_review_ref=ref("RAM_NOT_AUTHORITY/scorer_review.json"),
 actual_matrix_validator_review_ref=ref("RAM_NOT_AUTHORITY/matrix_review.json"))
intentref=putram("RAM_NOT_AUTHORITY/later_intent.json",intent)
built=[]
for row in plan:
 q=prediction["build_stage_v1"](intentref,row["parent_id"],row["method_id"],row["stage"],read,ref)
 check(q["output_directory"]==row["output_directory"]and q["outer_truth_access_authorized"]is False,"Whole actual builder request "+row["method_id"]+":"+row["stage"])
 built.append(q)
# Explicit incomplete runtime, old/reduced/native-budget and missing-parent fixtures.
argformer=prediction["METHODS"][-1];parent=prediction["PARENTS"][0]
for r,key,value in[(intentref,"method_freeze_ref",wait),(qualifications[argformer]["fit"],"full952_observed_samples",951),
 (qualifications[argformer]["fit"],"MLM_epochs",4),(qualifications[argformer]["fit"],"MLM_precision","fp32"),
 (qualifications[argformer]["fit"],"reduced_batch_or_gradient_cache",True),
 (previous[parent][argformer]["fit"]["completion_ref"],"old_checkpoint_reused",True)]:
 stage="predict"if r==previous[parent][argformer]["fit"]["completion_ref"]else"fit"
 mutate(r,key,value,lambda stage=stage:prediction["build_stage_v1"](intentref,parent,argformer,stage,read,ref),"Reject later eligibility:"+key)
missing=copy.deepcopy(data_evidence);missing.pop(prediction["PARENTS"][-1])
reject(lambda:prediction["all_data_ready_v1"](missing,read,ref),"Missing eighth biological parent blocks")
cells=[dict(parent_id=q["parent_id"],method_id=q["method_id"],run_id=q["run_id"],top5_predictions_ref=ref("RAM_NOT_AUTHORITY/top5.json"))
 for q in built if q["stage"]=="predict"]
bundle=prediction["build_matrix_bundle_v1"](intentref,cells,read,ref)
check(len(bundle["cells"])==40 and bundle["parent_ids"]==list(prediction["PARENTS"]),"Concrete canonical40+8 bundle builder")
reject(lambda:prediction["build_matrix_bundle_v1"](intentref,cells[:-1],read,ref),"Missing fortieth primary blocks")
bad=copy.deepcopy(cells)
for cell in bad:
 if cell["method_id"]==argformer:cell.pop("top5_predictions_ref",None)
reject(lambda:prediction["build_matrix_bundle_v1"](intentref,bad,read,ref),"Missing any prespecified rawtop5 blocks")
# Dispatch executes original adapter _authority and exact rebuilt request before the forbidden numerical call boundary.
validator_source=(ROOT/prediction["SCIENCE"]/"outer_prediction_validator_v2.py").read_text()
adapter_source=(ROOT/prediction["SCIENCE"]/"outer_adapters_v2.py").read_text()
class Reader:
 def __init__(self,guard):self.guard=guard
 def decode_json(self,r):self.guard();return read(r)
 def read(self,r,**kwargs):self.guard();return raw(r)
auth_env=dict(ObservedReaderV2=Reader,ref=ref,HERE=prediction["SCIENCE"],need=saved["need"],Path=Path,re=__import__("re"))
record_fn=next(n for n in ast.parse(validator_source).body if isinstance(n,ast.FunctionDef)and n.name=="record")
exec(compile(ast.Module(body=[record_fn],type_ignores=[]),"<actual_record_predicate_RAM>","exec"),auth_env)
review_fn=next(n for n in ast.parse(validator_source).body if isinstance(n,ast.FunctionDef)and n.name=="reviewed_source")
exec(compile(ast.Module(body=[review_fn],type_ignores=[]),"<actual_reviewed_source_predicate_RAM>","exec"),auth_env)
auth_node=next(n for n in ast.parse(adapter_source).body if isinstance(n,ast.FunctionDef)and n.name=="_authority")
auth_env["cell_directory"]=lambda run,method,parent:"experiments/results/European_outer8_predictions_"+run+"/cells/"+method+"/"+parent
exec(compile(ast.Module(body=[auth_node],type_ignores=[]),"<actual_v3_adapter_authority_RAM>","exec"),auth_env)
class ScientificBoundary(Exception):pass
dispatch_calls=[]
def numerical_boundary(*a,**k):dispatch_calls.append((a,k));raise ScientificBoundary("Forbidden numerical API boundary reached")
api=types.SimpleNamespace(ObservedReaderV2=Reader,ref=ref,_authority=auth_env["_authority"],
 predict_h050_outer_v2=numerical_boundary,predict_ap_outer_v2=numerical_boundary,predict_argmix_outer_v2=numerical_boundary,
 run_argformer_outer_stage_v2=numerical_boundary)
for q in built:
 qr=putram("RAM_NOT_AUTHORITY/dispatch_"+q["parent_id"]+"_"+q["method_id"]+"_"+q["stage"]+".json",q)
 try:prediction["dispatch_v1"](qr,{"config":{},"modules":{},"args":{}},guard=lambda:None,resource_caps={"RAM":True})
 except ScientificBoundary:check(True,"Exact current dispatch reached forbidden original scientific API "+q["method_id"]+":"+q["stage"])
 else:raise AssertionError("Missing numerical boundary")
check(len(dispatch_calls)==64,"All64 explicit dispatch branches gated before numerical API boundary")
print(json.dumps(dict(schema="H059_saved_outer000_continuation_author_checks_v1",status="PASS_SOURCE_AND_RAM_ONLY",independent=False,
 checks=checks,negative_cases=negative,positive_checks=len(checks),negative_fixtures=len(negative),
 full_RAM_request_admit_input_count=8,full_source_derived_role_union_counts=[x[3]for x in answers],
 generation_scientific_worker_AST_exact=True,full48_prediction_barrier_preserved=True,
 actual_guard_state_process_resource_GPU_admission_journal_payload_operations=0),indent=2,sort_keys=True))

"""Real immutable metadata and synthetic lineage negatives. No actual resource actions."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_CPU_actual_plan_lineage_integration_source_20261016_v1"
P=ROOT/HERE
sp=importlib.util.spec_from_file_location("_actual_plan_metadata_fixture",P/"acceptance_contract_v1.py")
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
checks=[]
def good(name,condition=True):
 if not condition:raise AssertionError(name)
 checks.append(dict(name=name,status="PASS"))
def reject(name,fn):
 try:fn()
 except (ValueError,KeyError,FileNotFoundError,TypeError):checks.append(dict(name=name,status="PASS_REJECTED"));return
 raise AssertionError(name)
def F(path):return dict(path=path,sha256=hashlib.sha256(("SYNTHETIC_ONLY:"+path).encode()).hexdigest())
read0=c.read
e=json.loads((P/"actual_VAL003_evidence.v1.json").read_text())
env,spec,req,wait=c.plan_producer_lineage_v1(e,"american_c3_val_003")
plan=c.plan_ready_v1(e,"american_c3_val_003")
good("real unchanged VAL003 driver/input/source/wait/outer/ordinaryclose plan passes",
 "driver_ref" not in wait and len(spec["source_sha256"])==323 and plan["haplotypes"]==1600 and plan["focal_records"]==10000)
good("real source wait dictionary never modified","driver_ref" not in read0(e["actual_wait_ref"]) and read0(e["actual_wait_ref"])==wait)
reject("legacy unmodified validator still rejects absent field",lambda:c.legacy.plan_ready_v1(e,"american_c3_val_003"))
refs=dict(completion=e["completion_ref"],wait=e["actual_wait_ref"],outer=e["outer_terminal_ref"],close=e["close_receipt_ref"],
 input=env["input_ref"],request=spec["root_stage_request_ref"],closure=spec["transitive_source_closure_ref"],
 result=env["result_ref"],plan=e["plan_ref"])
def fault(name,record,mutate,fn=lambda:c.plan_ready_v1(e,"american_c3_val_003")):
 changed=copy.deepcopy(read0(record));mutate(changed)
 c.read=lambda r,*a,**k:copy.deepcopy(changed) if r==record else read0(r,*a,**k)
 try:reject(name,fn)
 finally:c.read=read0
fault("wrong admitted input driver",refs["input"],lambda v:v.update(root_driver_ref=F("wrong.py")))
fault("missing admitted source mapping entry",refs["input"],lambda v:v["source_sha256"].pop(next(iter(v["source_sha256"]))))
fault("changed admitted source digest",refs["input"],lambda v:v["source_sha256"].update({next(iter(v["source_sha256"])):"0"*64}))
fault("wrong admitted runtime",refs["input"],lambda v:v.update(runtime_ref=F("wrong_runtime.json")))
fault("input truth capability rejects",refs["input"],lambda v:v.update(query_truth_access_authorized=True))
fault("foreign request source",refs["request"],lambda v:v.update(adapter_manifest_ref=F("unknown_source_manifest.json")))
fault("foreign request schema",refs["request"],lambda v:v.update(schema="unknown"))
fault("wrong parent request",refs["request"],lambda v:v.update(parent_id="american_c3_val_004"))
fault("wrong transitive input closure",refs["closure"],lambda v:v.update(root_stage_request_ref=F("foreign_request.json")))
fault("wait mismatched admitted input",refs["wait"],lambda v:v.update(input_ref=F("unknown_input.json")))
fault("wait schema substitution rejects",refs["wait"],lambda v:v.update(schema="unknown_wait"))
fault("manufactured driver_ref rejects even if equal",refs["wait"],lambda v:v.update(driver_ref=env["root_driver_ref"]))
fault("nonzero direct wait rejects",refs["wait"],lambda v:v.update(returncode=1))
fault("boolean zero direct wait rejects",refs["wait"],lambda v:v.update(returncode=False))
fault("unresolved direct wait rejects",refs["wait"],lambda v:v.update(actual_direct_worker_wait=False))
fault("missing owned drain rejects",refs["wait"],lambda v:v.update(all_owned_processes_and_groups_drained=False))
fault("historical nonzero outer rejects",refs["outer"],lambda v:v.update(actual_exit_code=1))
fault("unconsumed outer rejects",refs["outer"],lambda v:v.update(actual_terminal_consumed=False))
fault("wrong close key rejects",refs["close"],lambda v:v.update(key="unknown_key"))
fault("close missing future revocation rejects",refs["close"],lambda v:v["value"].update(future_writes_and_relaunch_revoked=False))
fault("close foreign input rejects",refs["close"],lambda v:v["value"]["request"]["launch_identity"].update(input_ref=F("unknown_input.json")))
fault("completion wrong driver rejects",refs["completion"],lambda v:v.update(root_driver_ref=F("unknown_driver.py")))
fault("plan incomplete cohort rejects",refs["plan"],lambda v:v.update(haplotypes=1599))
fault("plan incomplete focal support rejects",refs["plan"],lambda v:v.update(focal_records=9999))
fault("plan modified scientific config rejects",refs["plan"],lambda v:v["config"].update(max_trees_per_block=31))
bad=copy.deepcopy(e);bad["completion_ref"]=F("unrecognized_completion.json")
reject("VAL003 unknown evidence pin rejects",lambda:c.plan_ready_v1(bad,"american_c3_val_003"))
# Future VAL004 has only synthetic metadata. No future input/policy/hash is asserted.
route=read0(c.ref(c.PLAN_ROUTE_REGISTRY))["routes"]["american_c3_val_004"]
e4={k:F(path) for k,path in route["evidence_paths"].items()};e4["parent_id"]="american_c3_val_004"
input4=F(route["input_path"]);request4=F(str(Path(route["input_path"]).parent/"request.json"))
closure4=F(str(Path(route["input_path"]).parent/"source_and_input_closure.json"))
result4=F(route["output_directory"]+"/american_c3_val_004_plan_completed.json")
observed4=F("SYNTHETIC_ONLY/val004_observation.json")
focals4=F("SYNTHETIC_ONLY/val004_focals.json")
native4=F("SYNTHETIC_ONLY/val004_native_completed.json")
cfg4=copy.deepcopy(req["scientific_config"]);cfg4.update(observation_ref=observed4,native_preparation_ref=native4,focal_registry_ref=focals4)
requestdoc4=dict(schema=route["request_schema"],root_authorizer="root_agent",execution_authorized=True,
 query_truth_access_authorized=False,GNN_optimization_authorized=False,TEST_access_authorized=False,
 adapter_manifest_ref=route["source_manifest_ref"],parent_id="american_c3_val_004",stage="plan",phase="plan_val_004",
 scientific_config=cfg4,bindings=dict(parent_id="american_c3_val_004",plan_config=cfg4))
manifest4=read0(route["source_manifest_ref"]);source_map4={r["path"]:r["sha256"] for r in manifest4["refs"]}
spec4=dict(root_driver_ref=route["driver_ref"],main_source_ref=route["plan_worker_ref"],source_sha256=source_map4,
 parent_id="american_c3_val_004",phase="plan_val_004",stage="plan",output_directory=route["output_directory"],
 job_id=route["job_id"],runtime_ref=c.RUNTIME_REF,refs=dict(runtime=c.RUNTIME_REF),GPU_authorized=False,
 GNN_optimization_authorized=False,query_truth_access_authorized=False,TEST_access_authorized=False,
 root_stage_request_ref=request4,transitive_source_closure_ref=closure4)
done4=dict(input_ref=input4,root_driver_ref=route["driver_ref"],parent_id="american_c3_val_004",stage="plan_val_004",
 status="completed_C3_GNN_current980_plan_val_004",native_MLM_dependency=False,TEST_deferred=True,result_ref=result4)
wait4=dict(wait,input_ref=input4,completed_ref=e4["completion_ref"],parent_id="american_c3_val_004",phase="plan_val_004",pid=123456789)
term4=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,completed_ref=e4["completion_ref"],direct_wait_ref=e4["actual_wait_ref"])
closed4=copy.deepcopy(read0(e["close_receipt_ref"]));closed4["key"]=route["key"]
closed4["value"]["proof"].update(completed_ref=e4["completion_ref"],direct_wait_ref=e4["actual_wait_ref"])
closed4["value"]["request"].update(key=route["key"],completed_ref=e4["completion_ref"],direct_wait_ref=e4["actual_wait_ref"],
 launch_identity=dict(input_ref=input4,pid=wait4["pid"]),
 completed_identity=dict(parent_id="american_c3_val_004",status=done4["status"]))
plan4={k:copy.deepcopy(plan[k]) for k in ["schema","sequence_length_bp","haplotypes","focal_records","all_trees_and_samples_retained","targets_opened","class_order"]}
plan4.update(parent_id="american_c3_val_004",config=cfg4,observation_ref=observed4,focal_registry_ref=focals4)
db={}
def add(r,v):db[r["path"]]=copy.deepcopy(v)
for r,v in [(e4["completion_ref"],done4),(e4["actual_wait_ref"],wait4),(e4["outer_terminal_ref"],term4),
 (e4["close_receipt_ref"],closed4),(e4["plan_ref"],plan4),(input4,spec4),(request4,requestdoc4),
 (closure4,dict(source_sha256=source_map4,root_stage_request_ref=request4,bindings=requestdoc4["bindings"],native_MLM_dependency=False)),
 (result4,dict(status="completed_C3_observation_only_block_plan",parent_id="american_c3_val_004",targets_opened=False,block_index_ref=e4["plan_ref"])),
 (native4,dict(status="completed_native_C3_path_preparation",parent_id="american_c3_val_004",focals_ref=focals4,
 observation_ref=observed4,query_focal_count=10000,query_ancestry_truth_opened=False)),
 (observed4,dict(parent_id="american_c3_val_004",split="val",sequence_length_bp=30000000,
 counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False))]:add(r,v)
def synthetic_read(r,*a,**k):
 if r["path"] in db:return copy.deepcopy(db[r["path"]])
 return read0(r,*a,**k)
c.read=synthetic_read
good("synthetic exact fresh VAL004 wait shape passes without driver fabrication",c.plan_ready_v1(e4,"american_c3_val_004")==plan4)
saved=db.pop(e4["actual_wait_ref"]["path"])
reject("future VAL004 missing genuine wait blocks",lambda:c.plan_ready_v1(e4,"american_c3_val_004"))
db[e4["actual_wait_ref"]["path"]]=dict(saved,parent_id="american_c3_val_003")
reject("future VAL004 wrong-parent wait blocks",lambda:c.plan_ready_v1(e4,"american_c3_val_004"))
db[e4["actual_wait_ref"]["path"]]=saved
db[e4["completion_ref"]["path"]]["root_driver_ref"]=F("unknown_future_driver.py")
reject("future VAL004 unrecognized source driver blocks",lambda:c.plan_ready_v1(e4,"american_c3_val_004"))
db[e4["completion_ref"]["path"]]=done4
c.read=read0
# Both existing route validators stay in the original saved600 namespace.
good("original saved600 gate namespace is unchanged",c.training_ready_v1.__globals__["legacy"] is c.legacy and
 c.legacy.plan_ready_v1.__globals__["terminal0_v1"] is c.legacy.terminal0_v1)
# Exercise selectors with fake metadata/guard only, retaining the real exact source whitelist.
known=read0(c.ref(c.APPROVED_SELECTOR_SOURCES));selector=F("SYNTHETIC_ONLY/ROOT_selector.json")
poref=F("SYNTHETIC_ONLY/active_policy.json");reref=F("SYNTHETIC_ONLY/source_review.json")
choice=known["approved"][1]
pov=dict(choice,active=True,root_authorizer="root_agent",source_review_ref=reref,journal=".runtime/storage/native7_H050_stopped_retention_v1")
sel=dict(choice,active=True,root_authorizer="root_agent",selector_source_ref=c.SELECTOR_SOURCE_REF,
 repair_authority_ref=c.REPAIR_REF,policy_ref=poref,source_review_ref=reref,
 accounting_mode="native7_H050_stopped_retention_v1",journal=pov["journal"])
db={selector["path"]:sel,poref["path"]:pov,reref["path"]:{}}
ref0=c.ref
c.read=synthetic_read
c.ref=lambda p:({"path":str(p),"sha256":selector["sha256"]} if str(p)==selector["path"] else
 {"path":str(p),"sha256":poref["sha256"]} if str(p)==poref["path"] else
 {"path":str(p),"sha256":reref["sha256"]} if str(p)==reref["path"] else ref0(p))
good("explicit known child selector metadata passes without actual guard",c.select_storage_metadata_v1(selector)==sel)
c.SELECTOR_REF=None;db[selector["path"]]=copy.deepcopy(sel);db[selector["path"]]["source_ref"]["sha256"]="0"*64
reject("unknown selected source digest blocks",lambda:c.select_storage_metadata_v1(selector))
db[selector["path"]]=copy.deepcopy(sel);db[selector["path"]]["policy_ref"]["sha256"]="0"*64
reject("unknown policy digest blocks",lambda:c.select_storage_metadata_v1(selector))
db[selector["path"]]=sel;c.select_storage_metadata_v1(selector)
reject("selector swap blocks",lambda:c.select_storage_metadata_v1(F("SYNTHETIC_ONLY/other_selector.json")))
c.read=read0;c.ref=ref0
# Complete returned-state union using actual source-role function and fake stage docs.
manifest=c.manifest_v1()
reqroles=dict(caps_ref=F("SYNTHETIC_ONLY/caps.json"),training_evidence_ref=c.SAVED_ACCEPTANCE_REF,
 integration_review_ref=F("SYNTHETIC_ONLY/integration_review.json"),ROOT_directive_ref=F("SYNTHETIC_ONLY/directive.json"),
 actual_stage_evidence_ref=F("SYNTHETIC_ONLY/evidence.json"),storage_policy_selector_ref=selector)
specroles=dict(root_stage_request_ref=F("SYNTHETIC_ONLY/request.json"),refs=dict(config=F("SYNTHETIC_ONLY/config.json")))
required=c.required_direct_refs_v1(specroles,reqroles)
child=json.loads((ROOT/"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/inert_fixtures.final.v1.json").read_text())["full_role_models"]["guard_state"]
for row in child:
 if row["path"]==known["approved"][1]["source_manifest_ref"]["path"]:row.update(known["approved"][1]["source_manifest_ref"])
role_models={}
for mode,returned in [("current_native14v3",list(manifest["current_guard_roles"].values())),("future_child",child)]:
 rows={}
 for row in [*returned,*required,F(c.HANDOFF)]:
  if row["path"] in rows and rows[row["path"]]!=row:raise ValueError("Conflicting full source role")
  rows[row["path"]]=row
 good("complete builder roles preserve all current guard roles under64 "+mode,len(rows)<=64 and all(row in rows.values() for row in returned))
 def fake_guard_union(groups):
  out={}
  for group in groups:
   for rr in group:out[rr["path"]]=rr
  c.require(len(out)<=64,"Original hard64 ceiling");return list(out.values())
 fake_guard_union([returned,required,[F(c.HANDOFF)]])
 extras=[F("SYNTHETIC_ONLY/extra_"+str(i)+".json") for i in range(65-len(rows))]
 reject("65th actual returned role rejects "+mode,lambda:fake_guard_union([returned,required,[F(c.HANDOFF)],extras]))
 role_models[mode]=dict(count=len(rows),all_returned_roles_preserved=True,roles=list(rows.values()))
for phase,inputref in c.OLD_INPUT_REFS.items():
    original=c.stage_source_v1(dict(input_ref=inputref,root_driver_ref=c.LEGACY_DRIVER_REF),phase)
    good("real unchanged already-admitted81-source input lineage "+phase,
        original["source_sha256"]==c.legacy.manifest_v1()["source_sha256"] and len(original["source_sha256"])==81)
good("no scientific library imported",not any(k in sys.modules for k in ("torch","torch_geometric","numpy","tskit")))
print(json.dumps(dict(schema="C3_actual_plan_lineage_source_checks_v1",status="PASS_SOURCE_ONLY",checks=checks,
 check_count=len(checks),role_models=role_models,real_VAL003_source_count=323,
 future_VAL004_all_metadata_synthetic_only=True,actual_guard_probe_measurement_accounting_admission_job_payload_or_Git=False)))

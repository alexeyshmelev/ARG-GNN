"""Independent narrow source review for European current-policy successor; synthetic only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,contextlib,copy,difflib,hashlib,importlib.util,io,json,sys,types
R=Path("/workspace-SR003.nfs2/ARG-GNN")
C="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
B="research/reviews/European_outer8_current_policy_successor_independent_review_20261016_v1"
OLD="research/reviews/European_outer8_preparation_source_20261016_v3"
def load(path,label):
 sp=importlib.util.spec_from_file_location(label,R/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def ref(path):return dict(path=path,sha256=hashlib.sha256((R/path).read_bytes()).hexdigest())
def funcs(path):
 text=(R/path).read_text();return text,{n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def reject(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError):return
 raise AssertionError("Expected rejection")
checks=[]
def check(name,detail=None):checks.append(dict(name=name,status="PASS_SOURCE_ONLY",**({"detail":detail}if detail is not None else {})))
c=load(C+"/prepare_outer000_v1.py","_independent_European_current_source")
contract=load(C+"/source_contracts_v1.py","_independent_European_author_contracts")
author=contract.run();expected=json.loads((R/C/"source_contract_results.final.v1.json").read_text())
assert author==expected and author["check_count"]==50
with(R/B/"author_contract_reproduction.v1.json").open("x")as f:json.dump(author,f,sort_keys=True,indent=2);f.write("\n")
check("author50_source_contracts_reproduced")
# Exact published diff and inherited scientific references.
patch=""
for oldname,newname in (("prepare_outer000_v1.py","prepare_outer000_v1.py"),("check_prediction_barrier_v2.py","check_prediction_barrier_v1.py")):
 patch+="".join(difflib.unified_diff((R/OLD/oldname).read_text().splitlines(True),(R/C/newname).read_text().splitlines(True),fromfile=OLD+"/"+oldname,tofile=C+"/"+newname))
assert patch==(R/C/"fidelity_delta.v1.diff").read_text()
old=load(OLD+"/prepare_outer000_v1.py","_independent_European_old_source")
for name in ("CONFIG","PRODUCER","ORIGINAL","CAPS","PARENT","OUTPUT","JOB","OBSERVED","PRIVATE"):
 assert getattr(c,name)==getattr(old,name),name
assert c.REGISTRY==OLD+"/outer8_registry.v1.json"and c.METHODS==OLD+"/frozen_method_ids.v1.json"and c.PROTOCOL==OLD+"/protocol.v1.json"
prior=json.loads((R/"research/reviews/European_outer8_preparation_independent_review_20261016_v3/review.v3.json").read_text())
assert prior["verdict"]=="SOURCE_PASS"and prior["scientific_readiness"]is False
registry=c.read(c.ref(c.REGISTRY));methods=c.read(c.ref(c.METHODS));protocol=c.read(c.ref(c.PROTOCOL))
assert len(registry["outer_rows"])==8
# Original production math/modules have no new copies; absolute imports select exact old package.
_,newfunc=funcs(C+"/check_prediction_barrier_v1.py")
runtext=ast.unparse(newfunc["run_v3"])
assert OLD.replace("/",".")+".score_outer_v2" in runtext
assert OLD.replace("/",".")+".outer_prediction_validator_v2" in runtext
check("exact_published_delta_original_science_caps_and_scientific_imports")
for name in ("root_request.INACTIVE.template.v1.json","validation_request.INACTIVE.template.v1.json","scoring_request.INACTIVE.template.v1.json",
 "consumer_ROOT_intent.INACTIVE.template.v1.json","generation_runtime_qualification.INACTIVE.template.v1.json","registration.INACTIVE.template.v1.json","decision_entry.INACTIVE.template.v1.json"):
 value=json.loads((R/C/name).read_text())
 assert not(value.get("active")is True or value.get("status")in ("active","qualified","registered")),name
check("all_authority_registration_and_qualification_templates_inactive")
# Complete source-derived models count37/45keys but30/38unique path+SHA refs.
parent=c.read(c.ref("research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json"))["current_guard_roles"]
assert len(parent)==37 and len({json.dumps(v,sort_keys=True)for v in parent.values()})==30
_,nodes=funcs("research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py")
def fake(label):return dict(path=B+"/SYNTHETIC_"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
ns=dict(copy=copy,require=c.need,_ref=ref,_manifest=lambda:None,_raw=lambda r:b"SYNTHETIC",
 SOURCE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py",
 MEASURE_SOURCE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/retained_nodes_v1.py",
 MANIFEST="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/source_manifest.v1.json",
 SELECTOR_SOURCE=c.SELECTOR["path"],_POLICY_REF=fake("future_policy"))
exec(compile(ast.Module(body=[nodes[n]for n in ("consumer_bindings_v1","_bindings")],type_ignores=[]),"<pure_child_binding_code>","exec"),ns)
child=ns["_bindings"](parent,dict(root_activation_ref=fake("activation"),source_review_ref=fake("childreview"),permanent_revocation_ref=fake("revoke")))
assert len(child)==45 and len({json.dumps(v,sort_keys=True)for v in child.values()})==38
q={k:fake(k)for k in ("runtime_ref","method_freeze_ref","source_review_ref","policy_selector_ref","registration_ref","registration_decisions_ref",
 "registration_ledger_ref","runtime_qualification_ref","root_request_ref","entry_review_ref","bundle_ref","scorer_review_ref","actual_integration_qualification_ref")}
models=[]
for mode,returned,counts in (("current",parent,[52,53,56]),("child",child,[59,60,63])):
 for stage,count in zip(("generation","validate","score"),counts):
  required=c.generation_direct_refs_v1(q,fake("generation_root_request"))if stage=="generation"else c.consumer_direct_refs_v1(q,stage)
  union=ns["consumer_bindings_v1"](list(returned.values()),required)
  assert len(union)==count and all(r in union for r in returned.values())
  extras=[fake(mode+"_"+stage+"_extra"+str(i))for i in range(65-count)]
  reject(lambda:ns["consumer_bindings_v1"](list(returned.values()),required,extras))
  models.append(dict(policy=mode,stage=stage,returned_keys=len(returned),returned_unique_refs=30 if mode=="current"else 38,complete_unique_refs=count,all_returned_records_preserved=True))
check("complete_raw_key_and_unique_ref_models_and65_rejection",models)
# Prospective metadata contract with exact original source refs, fake actual authority facts.
docs={};reads=[]
def put(label,value):
 rr=fake(label);docs[rr["path"]]=copy.deepcopy(value);return rr
read0=c.read;raw0=c.raw
c.read=lambda rr:copy.deepcopy(docs[rr["path"]])if rr["path"]in docs else read0(rr)
c.raw=lambda rr:("SYNTHETIC_HYPOTHESIS independent registration".encode()if rr==q["registration_ledger_ref"]else raw0(rr))
created=datetime(2026,10,17,0,0,tzinfo=timezone.utc)
freeze=put("freeze",dict(frozen_at_utc=(created-timedelta(hours=2)).isoformat()))
rq=dict(q,method_freeze_ref=freeze,created_at_utc=created.isoformat(),cpu_ids=list(range(8)))
reg=dict(schema="European_outer8_prospective_registration_v1",root_authorizer="root_agent",status="registered",
 condition="European_outer8_frozen_predictor_independent_confirmation_v1",hypothesis_id="SYNTHETIC_HYPOTHESIS",linked_hypothesis_ids=["H038","H050"],
 method_freeze_ref=freeze,protocol_ref=ref(c.PROTOCOL),registry_ref=ref(c.REGISTRY),decisions_snapshot_ref=q["registration_decisions_ref"],
 ledger_snapshot_ref=q["registration_ledger_ref"],registered_at_utc=(created-timedelta(hours=1)).isoformat())
docs[q["registration_ref"]["path"]]=reg
decision={k:reg[k]for k in ("condition","linked_hypothesis_ids","method_freeze_ref","protocol_ref","registry_ref")}
decision.update(id=reg["hypothesis_id"],outer_tuning_authorized=False,repeat_reason="SYNTHETIC independent confirmation")
docs[q["registration_decisions_ref"]["path"]]=dict(decisions=[decision])
approved=c.APPROVED["native14_stopped_retention_v3"]
selector=dict(schema="explicit_ROOT_storage_policy_selector_v1",active=True,root_authorizer="root_agent",selector_source_ref=c.SELECTOR,
 accounting_mode="native14_stopped_retention_v3",source_ref=approved["source_ref"],source_manifest_ref=approved["manifest_ref"],policy_ref=approved["policy_ref"],
 measurement_source_ref=ref("research/reviews/native14_stopped_retention_successor_source_20261016_v3/retained_nodes_v1.py"),
 source_review_ref=fake("source_review"),repair_authority_ref=c.REPAIR,no_fallback_discovery_upgrade_or_in_run_swap=True,
 future_ROOT_adapters_only=True,scheduling_ceiling_bytes=980000000000,human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000)
docs[q["policy_selector_ref"]["path"]]=selector
# selection raw reads are synthetic or pinned source/metadata only.
c.raw=lambda rr:(b"SYNTHETIC_SOURCE_REVIEW"if rr==selector["source_review_ref"]else "SYNTHETIC_HYPOTHESIS independent registration".encode()if rr==q["registration_ledger_ref"]else raw0(rr))
cfg=c.read(c.CONFIG)
runtime=dict(schema="European_outer000_CPU_runtime_v1",device="cpu",CUDA_VISIBLE_DEVICES="",interpreter=str(R/".runtime/env/bin/python"),
 config_runtime=cfg["runtime"],native_threads=8,cpu_ids=list(range(8)))
qualification=dict(schema="European_outer000_runtime_qualification_v1",root_authorizer="root_agent",status="qualified",
 runtime_ref=q["runtime_ref"],operational_manifest_ref=ref(c.MANIFEST),config_ref=c.CONFIG,producer_ref=c.PRODUCER,unresolved_items=[])
docs[q["runtime_ref"]["path"]]=runtime;docs[q["runtime_qualification_ref"]["path"]]=qualification
c.prospective(rq,cfg)
for obj,key,bad in ((runtime,"native_threads",7),(runtime,"cpu_ids",list(range(7))),(runtime,"config_runtime",{}),
 (qualification,"status","inactive_template"),(qualification,"unresolved_items",["pending"]),(qualification,"operational_manifest_ref",ref(OLD+"/source_manifest.v1.json")),
 (decision,"outer_tuning_authorized",True),(reg,"registered_at_utc",(created+timedelta(seconds=1)).isoformat())):
 oldval=copy.deepcopy(obj[key]);obj[key]=bad;reject(lambda:c.prospective(rq,cfg));obj[key]=oldval
check("new_prospective_runtime_registration_and_freeze_gates_with_independent_negatives")
assert not any(name in sys.modules for name in ("torch","numpy","torch_geometric","tskit","scripts.project_budget_v9"))
result=dict(schema="European_outer8_current_policy_independent_static_and_metadata_checks_v1",status="PASS_SOURCE_ONLY",checks=checks,check_count=len(checks),
 all_runtime_registration_policy_and_guard_state_facts_synthetic=True,no_actual_guard_probe_data_model_truth_jobs_accounting_or_Git=True)
with(R/B/"independent_static_checks.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=result["status"],checks=len(checks),role_models=models)))

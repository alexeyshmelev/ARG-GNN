"""Validate independent receipt shape, exact pins/evidence, and REJECT gate."""
import hashlib,importlib.util,json,sys,datetime
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");REVIEW=ROOT/"research/reviews/native_argformer_tsinfer_fit_v2_independent"
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {"path":str(p.relative_to(ROOT)),"sha256":sha(p)}
r=json.loads((REVIEW/"review_receipt.json").read_text());template=json.loads((REVIEW/"independent_review_schema_template.json").read_text())
checks=[]
def check(name,v):
 if not v:raise AssertionError(name)
 checks.append(name)
for k in ("kind","review_role","implementation","config"):
 check("template_contract_"+k,r[k]==template[k])
check("independent_final_verdict",r["verdict"]=="REJECT" and r["template_not_review"] is False)
for k in ("actual_model_fit_executed","actual_scientific_arrays_opened","actual_GPU_inventory_context_or_live_ledger_read","authority_activation_performed"):
 check("source_only_"+k,r[k] is False)
for record in r["files"]:
 check("exact_review_file_"+record["path"],sha(ROOT/record["path"])==record["sha256"])
for name in ("implementation","config","author_source_receipt","budgetv4_independent_review"):
 record=r[name];check("exact_pin_"+name,sha(ROOT/record["path"])==record["sha256"])
h=json.loads((REVIEW/"handmade_checks.json").read_text())
check("fresh_handmade125_evidence",h["check_count"]==125 and h["released_model_instantiated"] is False and h["placeholder_model_counted_as_native_success"] is False)
check("fresh_exhaustive_retrieval_count",h["details"]["fresh_retrieval_fixture"]["exhaustive_pairs"]==1105920)
check("three_concrete_issues",len(r["issues"])==3)
spec=importlib.util.spec_from_file_location("_review_validator_exact_executor",ROOT/r["implementation"]["path"]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);c=m.policy()
for parent in c["parent_ids"]:
 for stage in m.STAGES:
  a=m.admission_bindings(c,parent,stage,"validator_fixture")
  try:m.validate_admission(c,parent,stage,"validator_fixture",a,r,{}, {})
  except ValueError as e:check("exact_REJECT_blocks_"+parent+"_"+stage,str(e)=="fresh independent source/config PASS required")
  else:raise AssertionError("REJECT must never admit execution")
result={"kind":"native_argformer_tsinfer_fit_v2_independent_review_validator_receipt","status":"passed","check_count":len(checks),"checks":checks,"review_receipt":ref(REVIEW/"review_receipt.json"),"schema_template":ref(REVIEW/"independent_review_schema_template.json"),"validator":ref(Path(__file__)),"executor_receipt_contract_validated":True,"REJECT_blocks_all12_parent_stages":True,"actual_execution_or_authority_activation":False,"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (REVIEW/"validator_receipt.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2)
print(json.dumps({"status":"passed","check_count":len(checks),"review_receipt":result["review_receipt"]},indent=2))

"""Source/metadata-only checks. Never creates a guard, admission, process or fixture."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib,io,json,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/TreeLoom_Chickenpox_CPU_execution_source_20261017_v1"
BASE="research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v4"
def fn(s,name):return next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
class Envelope(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   n.value=n.value.replace("TreeLoom_matched_CPU","TreeLoom_Chickenpox_CPU").replace("TreeLoom-matched-CPU-","TreeLoom-Chickenpox-CPU-").replace("H060/","H061/")
   n.value={"ROOT supplies32actually available CPU identities":"ROOT supplies exact CPU223 capacity",
    "American_C3_full_original_CPU_actual_wait_v1":"TreeLoom_Chickenpox_CPU_actual_wait_v1"}.get(n.value,n.value)
  return n
 def visit_keyword(self,n):
  self.generic_visit(n)
  if n.arg in("OMP_NUM_THREADS","NUMBA_NUM_THREADS"):
   assert isinstance(n.value,ast.Constant)and n.value.value=="32";n.value.value="1"
  return n
def main():
 assert sys.dont_write_bytecode and sys.flags.no_user_site
 p=ROOT/HERE
 for f in p.iterdir():
  if f.suffix==".py":compile(f.read_text(),str(f),"exec");ast.parse(f.read_text())
  elif f.suffix==".json":json.loads(f.read_text())
 modules=[importlib.import_module(HERE.replace("/",".")+"."+n)for n in("root_driver_v1","worker_entry_v1","scientific_runner_v1")]
 d,w,r=modules;c=d.c
 assert w.c is c and r.c is c and c.HERE==HERE and c.PHASE=="train_development"
 assert c._h._GUARD is None and c.SELECTOR_REF is None
 outputs=[]
 for module in(d,w):
  buf=io.StringIO()
  with contextlib.redirect_stdout(buf):assert module.main([])==0
  outputs.append(json.loads(buf.getvalue()))
  assert outputs[-1]["status"]=="SOURCE_ONLY_INACTIVE"
 b=c.binding();cfg=json.loads((p/"scientific_config.INACTIVE.template.v1.json").read_text())
 inactive_rejected=False
 try:c.validate_config(cfg)
 except Exception:inactive_rejected=True
 assert inactive_rejected
 cfg["execution_enabled"]=True;c.validate_config(cfg)
 c.context(b["data_source_manifest_ref"]);c.scientific_review(b["scientific_review_ref"])
 registration=json.loads((ROOT/b["registration_ref"]["path"]).read_text())
 c.registration(dict(registration_ref=b["registration_ref"],registration_decisions_ref=b["registration_decisions_ref"],
  created_at_utc=registration["registered_utc"]))
 old=(ROOT/BASE/"root_driver_v1.py").read_text();new=(p/"root_driver_v1.py").read_text()
 preserved=("input_spec","admit","worker_python","sample_scope","boundary","execute","main")
 for name in preserved:
  assert ast.dump(Envelope().visit(copy.deepcopy(fn(old,name))),include_attributes=False)==ast.dump(fn(new,name),include_attributes=False),name
 bridge=ast.parse((p/"scientific_runner_v1.py").read_text())
 attrs={n.attr for n in ast.walk(bridge)if isinstance(n,ast.Attribute)}
 forbidden={"load_test_v1","predict_test_v1","score_test_v1","paired_block_interval_v1",
  "fit_frozen_ridge_v1","predict_frozen_ridge_v1","classical_predictions_v1","predict_verified_legacy_gru_v1","load","Popen","apply_reserve_v1"}
 assert not attrs&forbidden,attrs&forbidden
 assert {"load_development_v1","fit_development_pair_v1"}<=attrs
 for seed in c.SEEDS:
  for arm in c.ARMS:assert re.fullmatch(r"selected_seed(101|202|303)_(full|no_memory)\.pt","selected_seed%d_%s.pt"%(seed,arm))
 assert c.CAPS["CPU_threads"]==1 and c.CAPS["CUDA_bytes"]==0
 assert c.CAPS["rss_bytes"]==4*2**30 and c.CAPS["reservation_bytes"]==32*2**20
 assert c.CAPS["wall_seconds"]==7200 and c.CAPS["scientific_seconds"]==6900
 assert not any(n in sys.modules for n in("torch","numpy","torch_geometric","scripts.project_budget_v9","arggnn.project_budget_v4"))
 assert c._h._GUARD is None and c.SELECTOR_REF is None
 return dict(schema="H061_Chickenpox_CPU_source_static_check_v1",status="PASS_SOURCE_AND_METADATA_ONLY",
  actual_inactive_module_routes_checked=True,worker_and_runner_import_exact_contract=True,
  real_H061_registration_and_decision_metadata_validated=True,stale_ledger_required=False,
  exact_existing_data_metadata_and_independent_scientific_review_validated=True,
  inactive_configuration_refused_and_exact_active_configuration_checked_in_memory=True,
  lifecycle_functions_AST_preserved=list(preserved),
  allowed_lifecycle_AST_delta="namespace/job/key/schema envelope; CPU description; OMP/NUMBA32to1",
  bridge_HAS_no_TEST_comparator_scoring_legacy_resume_or_process_call=True,
  scientific_imports_or_arrays=False,payload_read=False,numerical_model_execution=False,
  guard_factory_or_policy_mutex_called=False,admission_or_job_or_resource_process_probe=False,
  source_checks_are_not_execution_or_fit_evidence=True)
if __name__=="__main__":print(json.dumps(main(),sort_keys=True,indent=2))

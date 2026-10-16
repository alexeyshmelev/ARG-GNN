"""Standard-library-only source checks; no project imports or numerical execution."""
from pathlib import Path
import ast,hashlib,json
HERE=Path("research/reviews/H061_postselection_execution_source_20261017_v1")
OLD=Path("research/reviews/TreeLoom_Chickenpox_CPU_execution_source_20261017_v1")
EXPECTED=str(HERE).replace("/",".")
def require(value,message):
 if not value:raise AssertionError(message)
def tree(name):return ast.parse((HERE/name).read_text(),filename=str(HERE/name))
def functions(t):return {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
def calls(node):
 return [ast.unparse(x.func)for x in ast.walk(node)if isinstance(x,ast.Call)]
def main():
 files=sorted(HERE.glob("*.py"))
 for p in files:compile(p.read_text(),str(p),"exec")
 for name in("root_driver_v1.py","worker_entry_v1.py","scientific_runner_v1.py"):
  routes=[n for n in ast.walk(tree(name))if isinstance(n,ast.ImportFrom)and any(x.asname=="c"for x in n.names)]
  require(len(routes)==1 and routes[0].module==EXPECTED and routes[0].names[0].name=="root_contract_v1",
   "Exact executable contract route: "+name)
 contract=tree("root_contract_v1.py")
 literals={t.id:ast.literal_eval(n.value)for n in contract.body if isinstance(n,ast.Assign)
  and isinstance(n.value,ast.Constant)for t in n.targets if isinstance(t,ast.Name)}
 require(literals["HERE"]==str(HERE)and literals["REL"]==str(HERE),"Contract filesystem namespace")
 own=functions(contract);original=functions(ast.parse((OLD/"root_contract_v1.py").read_text()))
 for name in("binding","sources","scientific_review","context","registration"):
  require(ast.dump(own[name],include_attributes=False)==ast.dump(original[name],include_attributes=False),
   "Unchanged reviewed metadata helper: "+name)
 profiles=json.loads((HERE/"storage_profiles.v1.json").read_text())
 require(set(profiles)=={"native7_H050_stopped_retention_v1","Relate_failed64GiB_retention_v1"},
  "Only two exact known selector interfaces")
 require("_h.guard"in calls(own["guard"])and "load"not in calls(own["guard"]),
  "Unchanged original cache-before-memo helper with explicit selected module")
 require("DEPS['SELECTOR']"in ast.unparse(own["direct_roles"])or 'DEPS["SELECTOR"]'in ast.unparse(own["direct_roles"]),
  "Original direct selector source retained; new source only through returned exact capsule")
 driver=functions(tree("root_driver_v1.py"))
 require("base.execute"in calls(driver["execute"])and "c.load"in calls(driver["execute"])
  and not any(x in calls(driver["execute"])for x in("subprocess.Popen","os.fork","os.waitpid")),
  "Reuse sealed finite lifecycle, no new generic supervisor")
 admission=calls(driver["admit"])
 for before,after in(("g.budget.v4.scope_spec","g.operation_draft_v1"),
   ("g.operation_draft_v1","g.apply_reserve_v1"),("g.apply_reserve_v1","packet.mkdir")):
  require(admission.index(before)<admission.index(after),"Original admission mutation ordering")
 runner=functions(tree("scientific_runner_v1.py"))
 sites={}
 for name,node in runner.items():
  for call in calls(node):sites.setdefault(call,set()).add(name)
 for call,owners in {
  "completion.fit_frozen_ridge_v1":{"_freeze"},"completion.predict_test_v1":{"_predict"},
  "completion.predict_frozen_ridge_v1":{"_predict"},"completion.classical_predictions_v1":{"_predict"},
  "completion.score_test_v1":{"_score"},"completion.metrics_v1":{"_score"},
  "completion.paired_block_interval_v1":{"_score"},"completion.load_test_v1":{"_predict","_score"}}.items():
  require(sites.get(call)==owners,"Exact unchanged scientific seam and phase: "+call)
 forbidden=("fit_development_pair_v1","backward","AdamW","SGD","manual_seed")
 require(not any(any(x==f or x.endswith("."+f)for f in forbidden)for x in sites),
  "No new core fitting, optimizer, training or initialization seed")
 require("model.load_state_dict"in calls(runner["_state"]),"Exact selected state loading")
 state_load=[n for n in ast.walk(runner["_state"])if isinstance(n,ast.Call)and ast.unparse(n.func)=="torch.load"]
 require(len(state_load)==1 and {k.arg:ast.literal_eval(k.value)for k in state_load[0].keywords}
  =={"map_location":"cpu","weights_only":True},"Weights-only CPU retained state load")
 score_calls=calls(runner["_score"])
 require(score_calls.index("_load_predictions")<score_calls.index("completion.load_test_v1")
  <score_calls.index("completion.score_test_v1"),"All15product authentication before any TEST scoring")
 predict_calls=calls(runner["_predict"])
 require(predict_calls.index("_ridge")<predict_calls.index("completion.load_test_v1")
  and not any("metric" in x or "score_test" in x for x in predict_calls),
  "Frozen comparator state precedes TEST, no prediction-stage truth scorer")
 require(json.loads((HERE/"caps.inactive.v1.json").read_text())==json.loads((OLD/"caps.inactive.v1.json").read_text()),
  "Original finite CPU/resource caps retained")
 require((HERE/"dependencies.v1.json").read_bytes()==(OLD/"dependencies.v1.json").read_bytes(),"Exact existing lifecycle dependencies")
 binding=json.loads((HERE/"scientific_binding.v1.json").read_text())
 require(binding["runner_module"]==EXPECTED+".scientific_runner_v1"
  and binding["runner_source_ref"]["path"]==str(HERE/"scientific_runner_v1.py")
  and binding["runner_source_ref"]["sha256"]==hashlib.sha256((HERE/"scientific_runner_v1.py").read_bytes()).hexdigest(),
  "Exact emitted runner namespace and source hash")
 for stage in("freeze","predict","score"):
  q=json.loads((HERE/("ROOT_request."+stage+".INACTIVE.template.v1.json")).read_text())
  require(q["active"]is False and q["execution_authorized"]is False and q["stage"]==stage
   and q["scientific_config"]["execution_enabled"]is False
   and q["scoring_authorized"]is(stage=="score")and q["TEST_access_authorized"]is(stage in("predict","score")),
   "Inactive separate stage authority template")
 return dict(schema="H061_postselection_source_static_checks_v1",status="SOURCE_CHECKS_PASS_NOT_INDEPENDENT_REVIEW",
  project_modules_imported=False,payloads_opened=False,numerical_work=False,guard_or_resource_calls=False,
  compiled_sources=[str(p)for p in files],exact_contract_routes=True,unchanged_helper_AST=True,
  sealed_finite_lifecycle_reused=True,original_admission_ordering=True,dual_explicit_source_pinned_selector_interfaces=True,
  phase_separated_scientific_calls=True,all15authentication_before_scoring=True,
  original_caps_dependencies=True,legacy_GRU="unavailable_payload_not_accepted",
  standalone_token_probe=False,execution_authorized=False)
if __name__=="__main__":
 print(json.dumps(main(),sort_keys=True,indent=2))

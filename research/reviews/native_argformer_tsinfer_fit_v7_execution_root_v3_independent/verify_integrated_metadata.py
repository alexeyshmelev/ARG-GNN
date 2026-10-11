"""Bounded exact V3 source and saved metadata review; no runtime or process queries."""
from pathlib import Path
from copy import deepcopy
import ast,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v3_independent"
NEW=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v3"
OLD=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v2"
V2_REVIEW=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v2_independent"
CHECKS=[];HASHED={}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(p):
 raw=p.read_bytes();result={"path":str(p.relative_to(ROOT)),"sha256":sha(raw)}
 HASHED[result["path"]]={**result,"bytes":len(raw)}
 return result
def check(value,label):
 CHECKS.append(label)
 if not value:raise AssertionError(label)
def load(p):return json.loads(p.read_bytes())
def verify(reference):check(ref(ROOT/reference["path"])==reference,"hash_"+reference["path"])
pins={
 "run_four_parent_queue.py":"48925d1572d242fde668a24e668750954fcde640feb51cecddeaaa2f6631af97",
 "execution_plan.json":"0f388eaee0d9b4ff88a6dca432170a7f5a01e7f1278870c985a827b84fbda6a1",
 "source_delta.json":"8f2a8865c55c839f6a3d10fb868349434ea3e9391d311a6b0ed1d0259d7c943d"}
for name,pin in pins.items():check(ref(NEW/name)["sha256"]==pin,"exact_V3_"+name)
source=(NEW/"run_four_parent_queue.py").read_text();old_source=(OLD/"run_four_parent_queue.py").read_text()
delta=load(NEW/"source_delta.json")
for key in ("predecessor_source","new_source","predecessor_plan","new_plan"):verify(delta[key])
check(delta["new_source"]==ref(NEW/"run_four_parent_queue.py"),"delta_bound_final_runner")
check(delta["new_plan"]==ref(NEW/"execution_plan.json"),"delta_bound_final_plan")
check(delta["predecessor_source"]==ref(OLD/"run_four_parent_queue.py"),"delta_bound_V2_runner")
check(delta["predecessor_plan"]==ref(OLD/"execution_plan.json"),"delta_bound_V2_plan")
restored=source
for before,after in reversed(delta["source_diff"]):
 check(restored.count(after)==1,"unique_snapshot_replacement")
 restored=restored.replace(after,before,1)
check(restored==old_source,"exact_inverse_V3_source_returns_V2_bytes")
new_funcs={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
old_funcs={n.name:n for n in ast.parse(old_source).body if isinstance(n,ast.FunctionDef)}
check(set(new_funcs)==set(old_funcs),"unchanged_function_names")
for name,node in new_funcs.items():
 check(ast.dump(node,include_attributes=False)==ast.dump(old_funcs[name],include_attributes=False),"unchanged_function_AST_"+name)
 check(ast.get_source_segment(source,node)==ast.get_source_segment(old_source,old_funcs[name]),"unchanged_function_bytes_"+name)
check(set(new_funcs)==set(delta["unchanged_function_names"]),"delta_all_function_names_truthful")
tail='save(BASE/"queue_started.json",'
check(source[source.index(tail):]==old_source[old_source.index(tail):],"entire_stage_dispatch_tail_bytes_unchanged")
wait='while any(identity_alive(r) for r in barrier["waiters"]):'
end='\ndef admission('
check(source[source.index(wait):source.index(end)]==old_source[old_source.index(wait):old_source.index(end)],"natural_release_wait_bytes_unchanged")
check('BASE=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root"' in source,"original_BASE_metadata_and_output_authorities_preserved")
check('cfg=read_pinned({"path":str(CONFIG.relative_to(ROOT)),"sha256":CONFIG_SHA},CONFIG)' in source,"CONFIG_exact_fixed_pin_snapshot")
check('plan=read_pinned(plan_ref,plan_path)' in source and 'plan=read(plan_path)' not in source,"plan_exact_pin_snapshot")
check('review=read_pinned(review_ref,review_path)' in source and 'review=read(review_path)' not in source,"independent_review_exact_pin_snapshot")
plan=load(NEW/"execution_plan.json");old_plan=load(OLD/"execution_plan.json")
restored_plan=deepcopy(plan);restored_plan["queue_source"]=deepcopy(old_plan["queue_source"])
check(restored_plan==old_plan,"V3_plan_only_queue_source_ref_changed")
check(plan["queue_source"]==ref(NEW/"run_four_parent_queue.py"),"V3_plan_exact_final_runner")
def refs(value):
 if isinstance(value,dict):
  if set(value)=={"path","sha256"}:yield value
  else:
   for item in value.values():yield from refs(item)
 elif isinstance(value,list):
  for item in value:yield from refs(item)
plan_refs=list(refs(plan));check(len(plan_refs)==33,"exact33_plan_metadata_references")
for reference in plan_refs:verify(reference)
expected={(p,s) for p in ("bio_paths_pilot_train_000","bio_paths_pilot_train_001","bio_paths_pilot_val_000","bio_paths_pilot_val_001") for s in ("probe-memory","fit","retrieve")}
check(len(plan["stage_proposals"])==12 and {(x["parent"],x["stage"]) for x in plan["stage_proposals"]}==expected,"exact12_unique_bound_stage_pairs")
for item in plan["stage_proposals"]:
 a=load(ROOT/item["admission_proposal"]["path"]);r=load(ROOT/item["experiment_registration"]["path"])
 label=item["parent"]+"_"+item["stage"]
 check(a["experiment_registration"]==item["experiment_registration"],"pair_registration_ref_"+label)
 check(a["scope_parents"]==r["scope_parents"]==[item["parent"]] and a["stage"]==r["stage"]==item["stage"],"pair_parent_stage_"+label)
 check(a["implementation"]==r["implementation"]==plan["implementation"] and a["config"]==r["config"]==plan["config"],"pair_source_config_"+label)
 check(not a["query_truth_authorized"] and not r["query_truth_authorized"] and not a["scientific_fallback"],"pair_no_truth_fallback_"+label)
# Verify and reuse previously sealed saved metadata closure rather than query native runtime/ledgers.
prior_pins={
 "independent_review.json":"d2a52d26a1c887af6bda99da3b08bd0c9da1b2c0a0b47a17e12724b7a902c104",
 "receipt.json":"4733caf385212ba9a05dc4e4e83583e4c717ca5759e369df3473eb6bab1dbb86"}
for name,pin in prior_pins.items():check(ref(V2_REVIEW/name)["sha256"]==pin,"sealed_V2_"+name)
prior_receipt=load(V2_REVIEW/"receipt.json")
for name in ("integrated_metadata_checks.json","plan_snapshot_counterexample.json","handmade_test_results.json"):
 reference=next(x for x in prior_receipt["files"] if x["path"]==str((V2_REVIEW/name).relative_to(ROOT)))
 verify({"path":reference["path"],"sha256":reference["sha256"]})
prior_meta=load(V2_REVIEW/"integrated_metadata_checks.json")
check(prior_meta["verdict"]=="PASS" and prior_meta["assertions"]==138 and prior_meta["reference_hashes_verified"]==52,"sealed_prior_metadata_closure_verified")
check(prior_meta["queue_source"]==delta["predecessor_source"] and prior_meta["execution_plan"]==delta["predecessor_plan"],"prior_metadata_closure_exact_predecessor")
barrier_ref=next(x for x in prior_meta["verified_references"] if x["path"]==plan["release_barrier"]["path"])
check(barrier_ref["sha256"]==plan["release_barrier"]["sha256"],"preserved_barrier_ref_matches_prior_verified")
barrier_recheck=prior_meta["barrier_identity_recheck"];verify(barrier_recheck)
check(load(ROOT/barrier_recheck["path"])["barrier"]==plan["release_barrier"],"saved_root_identity_recheck_exact_preserved_barrier")
check(load(ROOT/plan["independent_source_review"]["path"])["verdict"]=="PASS","unchanged_native_scientific_independent_review_PASS")
check(plan["physical_GPU"]==2 and plan["maximum_project_physical_GPUs"]==2,"GPU2_project_two_device_bound")
result={
 "kind":"native_v7_v3_exact_integrated_source_metadata_checks","verdict":"PASS",
 "queue_source":ref(NEW/"run_four_parent_queue.py"),"execution_plan":ref(NEW/"execution_plan.json"),
 "source_delta":ref(NEW/"source_delta.json"),"barrier_identity_recheck":barrier_recheck,
 "assertions":len(CHECKS),"checks":CHECKS,"reference_hashes_verified":len(HASHED),"verified_references":list(HASHED.values()),
 "all_function_ASTs_and_function_bytes_unchanged_from_V2":True,
 "exact_three_top_level_snapshot_replacements_only":True,"plan_only_queue_source_ref_changed":True,
 "all_proposal_registration_budget_barrier_runtime_inventory_and_science_authorities_preserved":True,
 "start_finish_identity_natural_wait_and_entire_stage_dispatch_tail_bytes_unchanged":True,
 "sealed_prior_runtime_scientific_and_storage_metadata_closure_reused":True,
 "saved_inventory_JSON_hashed_only_no_native_prefix_scan":True,
 "actual_native_runtime_import_proc_ledger_data_model_GPU_budget_registration_activation_scheduling_or_Git_operations":False}
with (OUT/"integrated_metadata_checks.json").open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
print(json.dumps({"verdict":"PASS","assertions":len(CHECKS),"reference_hashes_verified":len(HASHED),"all_function_ASTs_unchanged":True,"exact_three_top_level_snapshot_replacements_only":True},indent=2))

"""Focused pure-memory path review; no actual ROOT metadata or staging."""
import ast,copy,hashlib,json,pathlib,sys,traceback
from datetime import datetime,timezone,timedelta
from pathlib import PurePosixPath
SOURCE_DIR=sys.argv[1] if len(sys.argv)>1 else "research/reviews/H060_isolated_CPU_ROOT_metadata_source_20261017_v2"
WRAPPER="research/reviews/H060_isolated_CPU_runtime_source_20261017_v1"
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
def need(ok,message):
 if not ok:raise AssertionError(message)
def enc(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def key(r):return json.dumps(r,sort_keys=True)
def main():
 folder=pathlib.Path(SOURCE_DIR);source=(folder/"assemble_root_metadata_v1.py").read_bytes()
 tree=ast.parse(source);compile(tree,str(folder/"assemble_root_metadata_v1.py"),"exec")
 ns={"__name__":"independent_in_memory_review"}
 exec(compile(tree,str(folder/"assemble_root_metadata_v1.py"),"exec"),ns)
 template_names=["caps.inactive.v1.json","scientific_config.INACTIVE.template.v1.json","runtime.INACTIVE.template.v1.json",
  "runtime_qualification.INACTIVE.template.v1.json","ROOT_request.INACTIVE.template.v1.json",
  "ROOT_directive.INACTIVE.template.v1.json","training_context.INACTIVE.template.v1.json","dependencies.v1.json"]
 templates={n:json.loads((pathlib.Path(WRAPPER)/n).read_bytes())for n in template_names}
 b=json.loads((pathlib.Path(WRAPPER)/"scientific_binding.v1.json").read_bytes())
 deps=json.loads((folder/"dependencies.v1.json").read_bytes())
 contract=(pathlib.Path(WRAPPER)/"root_contract_v1.py").read_bytes()
 profiles_path=WRAPPER+"/storage_profiles.v1.json"
 profiles_bytes=pathlib.Path(profiles_path).read_bytes();profiles=json.loads(profiles_bytes)
 refs={}
 for name in [WRAPPER+"/root_driver_v1.py",WRAPPER+"/source_manifest.v1.json",WRAPPER+"/source_handoff.v1.json",
  SOURCE_DIR+"/assemble_root_metadata_v1.py",SOURCE_DIR+"/source_manifest.v1.json",profiles_path]:
  refs[name]=dict(path=name,sha256=hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest())
 table={}
 def put(r,v):table[key(r)]=v if isinstance(v,bytes)else enc(v);return r
 def invented(name,v):
  data=v if isinstance(v,bytes)else enc(v);r=dict(path=name,sha256=hashlib.sha256(data).hexdigest());return put(r,data)
 # Fixed sealed refs are replayed against handwritten bytes solely at the raw-read seam.
 # This exercises real metadata predicates, and makes no hash-authentication claim for fixtures.
 def fraw(r):
  need(key(r)in table,"Fixture attempted unregistered read: "+str(r));return table[key(r)]
 def fref(name):
  name=str(name);return refs[name]
 class MemoryPath(PurePosixPath):
  def resolve(self):return self
 packet_calls=[]
 ns.update(Path=MemoryPath,ROOT=MemoryPath(str(ROOT)),raw=fraw,read=lambda r:json.loads(fraw(r)),
  raw_path=lambda name:contract if str(name)==WRAPPER+"/root_contract_v1.py"else (_ for _ in()).throw(AssertionError("Unplanned source read")),
  reference=fref,template=lambda n:copy.deepcopy(templates[n]),absent=lambda paths:True,
  source_packet=lambda instruction:(packet_calls.append("source_packet")or(deps,b)))
 train=tuple("american_c3_short_train_%03d"%i for i in range(20))
 dummy=dict(path="handwritten/never_materialized.json",sha256="a"*64)
 original=dict(parent_plan_refs={p:dummy for p in train},
  actual_train_bindings={p:{k:dummy for k in("simulation_completion_ref","simulation_config_ref","inferred_observation_ref")}for p in train})
 put(b["original_training_request_ref"],dict(bindings=original))
 put(deps["wrapper_review_ref"],dict(independent=True,verdict="SOURCE_PASS",blockers=[],
  source_ref=refs[WRAPPER+"/root_driver_v1.py"],source_manifest_ref=refs[WRAPPER+"/source_manifest.v1.json"],
  source_handoff_ref=refs[WRAPPER+"/source_handoff.v1.json"],runner_source_ref=b["runner_source_ref"],protocol_ref=b["protocol_ref"]))
 put(b["scientific_review_ref"],dict(status="SOURCE_PASS",independent_review=True,blockers=[],
  candidate_manifest_ref=b["source_manifest_ref"],candidate_protocol_ref=b["protocol_ref"],
  candidate_handoff_ref=b["source_handoff_ref"],candidate_scientific_pins_ref=b["scientific_source_pins_ref"],
  candidate_compatibility_protocol_ref=b["compatibility_protocol_ref"],candidate_source_refs=[b["runner_source_ref"]]))
 put(b["scientific_review_manifest_ref"],b"handwritten source review manifest")
 put(templates["dependencies.v1.json"]["BASE_RUNTIME"],b"handwritten source runtime report")
 put(refs[profiles_path],profiles_bytes)
 profile=profiles["native7_H050_stopped_retention_v1"]
 selector=copy.deepcopy(profile);selector.update(schema="explicit_ROOT_storage_policy_selector_v1",active=True,root_authorizer="root_agent",
  accounting_mode="native7_H050_stopped_retention_v1",scheduling_ceiling_bytes=980000000000,human_limit_bytes=1000000000000,
  warning_threshold_bytes=950000000000,no_fallback_discovery_upgrade_or_in_run_swap=True,future_ROOT_adapters_only=True,
  existing_adapters_runs_and_storage_records_unchanged=True)
 selector_ref=put(profile["selected_record_ref"],selector)
 for k in("selector_source_ref","source_ref","source_manifest_ref","source_review_ref","measurement_source_ref","repair_authority_ref"):
  put(selector[k],b"handwritten selector source metadata seam")
 now=datetime.now(timezone.utc);created=now.isoformat();expires=(now+timedelta(hours=25)).isoformat()
 registration=dict(schema="H060_American_TreeLoom_matched_scientific_registration_v1",root_authorizer="root_agent",hypothesis_id="H060",
  status="registered_before_new_training_no_accuracy_yet",source_protocol_ref=b["protocol_ref"],new_pretruth_barrier=60,
  original_C3_pretruth_barrier_unchanged=30,resource_admission_or_execution_granted=False,scoring_authorized=False,
  TEST_access_authorized=False,registered_utc=(now-timedelta(minutes=1)).isoformat())
 q=json.loads((folder/"ROOT_instruction.INACTIVE.template.v1.json").read_bytes())
 q.update(root_authorizer="root_agent",staging_authorized=True,issue_full_TRAIN_request=True,current_CPU_capacity_confirmed=True,
  current_runtime_settings_confirmed=True,never_used_paths_confirmed=True,cpu_ids=list(range(32)),created_at_utc=created,expires_at_utc=expires,
  assembler_source_manifest_ref=refs[SOURCE_DIR+"/source_manifest.v1.json"],
  preparation_directory="research/reviews/H060_isolated_CPU_ROOT_preparation_HANDWRITTEN_NEVER_MATERIALIZED",
  packet_directory="research/reviews/H060_isolated_CPU_ROOT_actual_HANDWRITTEN_NEVER_MATERIALIZED",
  output_directory="experiments/results/H060_isolated_CPU_HANDWRITTEN_NEVER_MATERIALIZED",
  registration_ref=invented("handwritten/registration_not_created.json",registration),
  registration_decisions_source_ref=invented("research/experiment_decisions.json",dict(decisions=[dict(id="H060")])),
  registration_ledger_source_ref=invented("research/experiment_ledger.md",b"Handwritten H060 fixture only"),
  storage_policy_selector_ref=selector_ref)
 instruction_ref=dict(path="handwritten/instruction_not_created.json",sha256="b"*64)
 baseline=ns["plan_stage"](copy.deepcopy(q),instruction_ref)
 need(len(baseline[4])==8,"Eight fully serialized files")
 request=json.loads(baseline[4][baseline[5]["path"]]);receipt=json.loads(baseline[4][baseline[6]["path"]])
 need(request["scientific_config"]["epochs"]*request["scientific_config"]["parents_per_epoch"]*len(request["scientific_config"]["arms"])==1200,"Full updates")
 need(request["created_at_utc"]==created and request["expires_at_utc"]==expires,"Fixed dates")
 need(baseline[7][5]=="--admit-ref" and "--execute-ref"not in baseline[7]and receipt["next_admission_argv"]==baseline[7],"Admission argv only")
 # Evaluate actual literal prefix clauses from the reviewed wrapper.
 ct=ast.parse(contract);fn=next(n for n in ct.body if isinstance(n,ast.FunctionDef)and n.name=="validate_request")
 prefix_nodes=[n for n in ast.walk(fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="startswith"]
 def wrapper_prefixes(value):
  return all(eval(compile(ast.Expression(node),"sealed_wrapper_prefix_clause","eval"),{},dict(q=value))for node in prefix_nodes)
 need(wrapper_prefixes(request),"Baseline request prefix clauses")
 results=[]
 for label,change in [
  ("absolute_paths",{k:str(ROOT/q[k])for k in("preparation_directory","packet_directory","output_directory")}),
  ("dot_prefixed_preparation",dict(preparation_directory="./"+q["preparation_directory"])),
  ("duplicate_separator_packet",dict(packet_directory=q["packet_directory"].replace("research/reviews/","research//reviews/"))),
  ("trailing_slash_output",dict(output_directory=q["output_directory"]+"/")),
  ("internal_dot_preparation",dict(preparation_directory=q["preparation_directory"].replace("research/reviews/","research/./reviews/"))),
  ("parent_component_output",dict(output_directory=q["output_directory"].replace("experiments/results/","experiments/results/../results/")))]:
  value=copy.deepcopy(q);value.update(change);packet_calls.clear()
  try:plan=ns["plan_stage"](value,instruction_ref)
  except ValueError as e:results.append(dict(case=label,planner_accepts=False,error=str(e),source_packet_calls=len(packet_calls)));continue
  candidate=json.loads(plan[4][plan[5]["path"]])
  results.append(dict(case=label,planner_accepts=True,wrapper_literal_prefixes_accept=wrapper_prefixes(candidate),fully_serialized_files=len(plan[4]),source_packet_calls=len(packet_calls)))
 # Planning purity is established from actual AST and zero materialization in this fixture.
 calls=[ast.unparse(n.func)for f in tree.body if isinstance(f,ast.FunctionDef)and f.name=="plan_stage"for n in ast.walk(f)if isinstance(n,ast.Call)]
 need(not any(n.endswith((".mkdir",".write",".unlink",".rmdir",".rename"))or n=="os.open"for n in calls),"Mutation-free planner")
 return dict(schema="H060_metadata_independent_focused_path_checks_v1",source_dir=SOURCE_DIR,
  source_sha256=hashlib.sha256(source).hexdigest(),canonical_relative_baseline_pass=True,actual_six_metadata_predicates_executed=True,
  baseline_eight_files=True,full1200updates=True,fixed_dates=True,admission_argv_only=True,path_cases=results,
  fixture_limit="Read/hash/source-packet/absence seams replay handwritten in-memory metadata; no SHA-authentication or current-state claim for fixtures",
  actual_ROOT_or_selector_metadata_read=False,actual_preparation_created=False,guard_resource_or_numerical_work=False)
if __name__=="__main__":
 try:print(json.dumps(main(),sort_keys=True,indent=2))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)

"""Stdlib source/metadata checks only; no project, guard or scientific imports."""
from pathlib import Path
import ast,hashlib,json,runpy
HERE=Path("research/reviews/H060_isolated_CPU_runtime_source_20261017_v1")
OLD=Path("research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v5")
H061=Path("research/reviews/H061_postselection_execution_source_20261017_v1")
def need(ok,message):
 if not ok:raise AssertionError(message)
def nodes(p):return ast.parse(p.read_text(),filename=str(p))
def funcs(t):return {x.name:x for x in t.body if isinstance(x,ast.FunctionDef)}
def main():
 for p in HERE.glob("*.py"):compile(p.read_text(),str(p),"exec")
 module=str(HERE).replace("/",".")
 for name in("root_driver_v1.py","worker_entry_v1.py"):
  routes=[x for x in ast.walk(nodes(HERE/name))if isinstance(x,ast.ImportFrom)and any(a.asname=="c"for a in x.names)]
  need(len(routes)==1 and routes[0].module==module and routes[0].names[0].name=="root_contract_v1","Actual executable contract route "+name)
 own=funcs(nodes(HERE/"root_contract_v1.py"));h=funcs(nodes(H061/"root_contract_v1.py"))
 for name in("selected","bind_selector","guard"):
  need(ast.dump(own[name],include_attributes=False)==ast.dump(h[name],include_attributes=False),"Exact independently reviewed dual selector "+name)
 need((HERE/"storage_profiles.v1.json").read_bytes()==(H061/"storage_profiles.v1.json").read_bytes(),"Same exact two-profile whitelist")
 for name in("caps.inactive.v1.json","dependencies.v1.json","scientific_config.INACTIVE.template.v1.json","ROOT_selected_operational_failstops.v1.json"):
  need((HERE/name).read_bytes()==(OLD/name).read_bytes(),"Unchanged original caps/config/dependencies "+name)
 own_driver=funcs(nodes(HERE/"root_driver_v1.py"));old_driver=funcs(nodes(OLD/"root_driver_v1.py"))
 for name in own_driver:
  a=ast.dump(own_driver[name],include_attributes=False);b=ast.dump(old_driver[name],include_attributes=False)
  need(a.replace("H060-isolated-CPU-","TreeLoom-matched-CPU-")==b,"Unchanged finite driver "+name)
 own_worker=funcs(nodes(HERE/"worker_entry_v1.py"));old_worker=funcs(nodes(OLD/"worker_entry_v1.py"))
 for name in own_worker:need(ast.dump(own_worker[name],include_attributes=False)==ast.dump(old_worker[name],include_attributes=False),"Unchanged worker/serializer "+name)
 # Authenticate the real independent scientific review through the actual gate AST.
 binding=json.loads((HERE/"scientific_binding.v1.json").read_text())
 def ref(path):
  p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 def raw(record):
  need(ref(record["path"])==record,"Real source/review metadata hash");return Path(record["path"]).read_bytes()
 scope=dict(binding=lambda:binding,require=need,read=lambda r:json.loads(raw(r)),raw=raw)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[own["scientific_review"]],type_ignores=[])),"science_gate_metadata_only","exec"),scope)
 scope["scientific_review"](binding["scientific_review_ref"])
 science_manifest=json.loads(raw(binding["source_manifest_ref"]))
 need(science_manifest["runner_source_ref"]==binding["runner_source_ref"]
  and science_manifest["protocol_ref"]==binding["protocol_ref"]
  and science_manifest["compatibility_protocol_ref"]==binding["compatibility_protocol_ref"],"Exact reviewed compatible science route")
 need(binding["runner_module"]=="research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.warmstart_train_v2",
  "Emitted actual compatible runner route")
 # Fresh prefixes prevent admitted v4/v5 packets from being replayed here.
 source=(HERE/"root_contract_v1.py").read_text()
 for prefix in("research/reviews/H060_isolated_CPU_ROOT_preparation_","research/reviews/H060_isolated_CPU_ROOT_actual_",
  "experiments/results/H060_isolated_CPU_"):need(prefix in source,"Fresh exclusive scope namespace")
 need('closure,DEPS["SELECTOR"]'in source,"Old direct selector-source role retained, successor capsule not expanded")
 return dict(schema="H060_isolated_CPU_wrapper_source_checks_v1",status="SOURCE_CHECKS_PASS_NOT_INDEPENDENT_REVIEW",
  explicit_driver_worker_contract_routes=True,compatible_runner_and_real_scientific_review_authenticated=True,
  unchanged_driver_lifecycle_AST=True,unchanged_worker_serializer_AST=True,unchanged_caps_config_dependencies=True,
  exact_reviewed_dual_selector_AST=True,all_guard_roles_and_64_limit_preserved=True,
  fresh_scope_prefixes_and_job_key=True,actual_project_or_guard_modules_imported=False,
  actual_metadata_resources_payloads_models_or_jobs_opened=False,standalone_probe=False,execution_authorized=False)
if __name__=="__main__":print(json.dumps(main(),sort_keys=True,indent=2))

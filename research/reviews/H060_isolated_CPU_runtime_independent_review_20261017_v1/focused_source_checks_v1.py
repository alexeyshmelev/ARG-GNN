from pathlib import Path
import ast,copy,hashlib,importlib,json,runpy,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/H060_isolated_CPU_runtime_source_20261017_v1"
try:
 assert hashlib.sha256((BASE/"source_manifest.v1.json").read_bytes()).hexdigest()=="dd549708d11d06a1c1583197c5f837b77544a2146badbd9c5a161f2059446b3b"
 for n,sha in json.loads((BASE/"source_manifest.v1.json").read_text())["source_sha256"].items():
  p=ROOT/n;assert p.suffix in(".py",".sh",".json",".md",".patch")and p.resolve()==p
  assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,n
 author=runpy.run_path(str(BASE/"source_check_v1.py"))["main"]()
 selector=runpy.run_path(str(BASE/"selector_metadata_check_v1.py"))["main"]()
 assert author==json.loads((BASE/"source_check.result.v1.json").read_text())
 assert selector==json.loads((BASE/"selector_metadata_check.result.v1.json").read_text())
 mod="research.reviews.H060_isolated_CPU_runtime_source_20261017_v1"
 driver=importlib.import_module(mod+".root_driver_v1")
 worker=importlib.import_module(mod+".worker_entry_v1")
 contract=importlib.import_module(mod+".root_contract_v1")
 assert driver.c is worker.c is contract
 assert contract.HERE==str(BASE.relative_to(ROOT))and driver.HERE==contract.HERE
 assert driver.SOURCE==contract.DRIVER and worker.ROOT==contract.ROOT==ROOT
 assert contract.SELECTOR_REF is None and contract._CHOSEN is None and contract._h._GUARD is None and contract._h._SELECTOR_MODULE is None
 contract.scientific_review(contract.binding()["scientific_review_ref"])
 checks=["actual_driver_worker_contract_identity_and_source_namespace","actual_science_gate_accepts_exact_sealed_compatibility_review"]
 def reject(name,fn):
  try:fn()
  except (ValueError,AssertionError):checks.append(name);return
  raise AssertionError(name)
 # Actual config predicate with in-memory defaults, no runtime/source admission.
 cfg=copy.deepcopy(contract.binding()["inactive_defaults"])
 failstops=json.loads((BASE/"ROOT_selected_operational_failstops.v1.json").read_text())

 for k in contract.binding()["operational_config_keys"]:
  if k=="execution_enabled":cfg[k]=True
  elif k=="CPU_observation_cache_bytes":cfg[k]=0
  elif k=="max_truth_uncompressed_bytes":cfg[k]=32000000
  elif k=="max_cached_parent_bytes":cfg[k]=contract.CAPS["max_cached_parent_bytes"]
  else:cfg[k]=1
 contract.validate_config(cfg);checks.append("actual_full_budget_config_predicate_accepts_exact_science_with_positive_operational_failstops")
 bad=copy.deepcopy(cfg);bad["additional_updates_per_arm"]=599
 reject("reduced599updates_rejected",lambda:contract.validate_config(bad))
 bad=copy.deepcopy(cfg);bad["max_cached_parent_bytes"]+=1
 reject("over96GiB_cache_rejected",lambda:contract.validate_config(bad))
 bad=copy.deepcopy(cfg);bad["CPU_observation_cache_bytes"]=1
 reject("block_cache_rejected",lambda:contract.validate_config(bad))
 bad=copy.deepcopy(cfg);bad["arms"]=bad["arms"][:1]
 reject("single_arm_rejected",lambda:contract.validate_config(bad))

 # Actual returned-role checker with a wholly in-memory bookkeeping stand-in.
 from types import SimpleNamespace
 from contextlib import contextmanager
 def opaque(name):return dict(path="handwritten/"+name+".json",sha256="a"*64)
 qfake={k:opaque(k)for k in("source_review_ref","scientific_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref","registration_ref","registration_decisions_ref","training_context_ref","storage_policy_selector_ref")}
 cfgref=opaque("config");closure=opaque("closure");requestref=opaque("request")
 required=contract.direct_roles(requestref,qfake,cfgref,closure)
 assert contract.DEPS["SELECTOR"]in required and qfake["storage_policy_selector_ref"]in required
 assert not any(Path(r["path"]).suffix==".md"for r in required)
 current_roles=dict(one=opaque("current_guard_one"),two=opaque("current_guard_two"))
 union=[]
 for r in list(current_roles.values())+required:
  if r not in union:union.append(r)
 @contextmanager
 def fake_mutex(root):
  assert root==ROOT
  yield
 def fake_bindings(*groups):
  result=[]
  for group in groups:
   for r in group:
    if r not in result:result.append(r)
  assert len(result)<=64
  return result
 fake=SimpleNamespace(budget=SimpleNamespace(SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py",v4=SimpleNamespace(mutex=fake_mutex)),
  _state_locked=lambda *a,**kw:(None,None,None,current_roles),consumer_bindings_v1=fake_bindings)
 sfake=dict(root_stage_request_ref=requestref,refs=dict(config=cfgref),transitive_source_closure_ref=closure,storage=dict(reservation=dict(consumer_bindings=union)))
 assert contract.verify_actual_roles(sfake,qfake,fake)==union
 checks.append("actual_verify_roles_accepts_all_direct_and_returned_roles_with_in_memory_standin")
 incomplete=copy.deepcopy(sfake);incomplete["storage"]["reservation"]["consumer_bindings"].remove(current_roles["two"])
 reject("missing_current_guard_role_rejected",lambda:contract.verify_actual_roles(incomplete,qfake,fake))
 checks.append("old_selector_source_and_new_capsule_roles_retained_no_direct_Markdown")

 assert"torch"not in sys.modules and"numpy"not in sys.modules
 guard_files=[getattr(m,"__file__","")or""for m in sys.modules.values()]
 assert not any("stopped_retention_v1.py"in n or"relate_failed64_retention_v1.py"in n or"project_budget_v4.py"in n for n in guard_files)
 print(json.dumps(dict(status="INACTIVE_METADATA_CONTRACT_CHECKS_PASS",checks=checks,
  author_source_receipt_equal=True,selector_receipt_equal=True,selector_checks=len(selector["checks"]),
  guard_modules_imported=False,guard_called=False,scientific_imports=False,resource_or_payload_or_job_actions=False),sort_keys=True))
except Exception:print(traceback.format_exc())

"""Actual inactive module routing and concrete metadata-only role validation."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib,io,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v3"
PACKAGE=HERE.replace("/",".")
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads((ROOT/p).read_text())
def funcs(t,names):return[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name in names]
def need(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 assert sys.dont_write_bytecode and sys.flags.no_user_site
 # Real metadata-only imports select the actual contract used by admission and worker.
 driver=importlib.import_module(PACKAGE+".root_driver_v1")
 worker=importlib.import_module(PACKAGE+".worker_entry_v1")
 c=driver.c
 assert worker.c is c and c.__name__==PACKAGE+".root_contract_v1"
 expected=dict(HERE=HERE,SOURCE=HERE+"/root_contract_v1.py",DRIVER=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py")
 assert all(getattr(c,k)==v for k,v in expected.items())
 assert driver.HERE==HERE and driver.SOURCE==expected["DRIVER"]
 assert c.SELECTOR_REF is None and c._h._GUARD is None and c._h._SELECTOR_MODULE is None
 routing=[]
 for name in("root_contract_v1.py","root_driver_v1.py","worker_entry_v1.py"):
  text=(ROOT/HERE/name).read_text();t=ast.parse(text)
  for n in ast.walk(t):
   if isinstance(n,ast.ImportFrom)and n.module and n.module.startswith("research.reviews.American_TreeLoom_matched_CPU_execution_source_"):
    assert n.module==PACKAGE;routing.append(dict(file=name,module=n.module))
  assert "American_TreeLoom_matched_CPU_execution_source_20261017_v1"not in text
  assert "American_TreeLoom_matched_CPU_execution_source_20261017_v2"not in text
 assert {r["file"]for r in routing}=={"root_driver_v1.py","worker_entry_v1.py"}
 inactive=[]
 for module in(driver,worker):
  captured=io.StringIO()
  with contextlib.redirect_stdout(captured):assert module.main([])==0
  value=json.loads(captured.getvalue());assert value["status"]=="SOURCE_ONLY_INACTIVE";inactive.append(value)
 assert c.SELECTOR_REF is None and c._h._GUARD is None and c._h._SELECTOR_MODULE is None
 lineage=read(HERE+"/failure_lineage.v3.json")
 for key in("actual_wait_ref","original_request_ref","preserved_reserve_authority_ref","v2_manifest_ref","v2_handoff_ref","v2_rejection_ref"):
  r=lineage[key];assert ref(r["path"])==r
 auth=read(lineage["preserved_reserve_authority_ref"]["path"]);q=read(lineage["original_request_ref"]["path"])
 packet="research/reviews/TreeLoom_matched_CPU_ROOT_actual_20261017_v1"
 # Use actual imported contract function, never a detached AST version of direct_roles.
 direct=c.direct_roles(lineage["original_request_ref"],q,ref(packet+"/config.json"),ref(packet+"/source_and_input_closure.json"))
 assert q["registration_ledger_ref"]not in direct
 child=c.GUARD_REF["path"];pure=dict(require=need,copy=copy,_raw=c.raw)
 exec(compile(ast.Module(body=funcs(ast.parse((ROOT/child).read_text()),{"consumer_bindings_v1"}),type_ignores=[]),"<exact_pure_consumer_union>","exec"),pure)
 guards=list(auth["bindings"].values());union=pure["consumer_bindings_v1"](guards,direct)
 assert len(guards)==45 and len({json.dumps(r,sort_keys=True)for r in guards})==38
 assert len(union)==61 and all(r in union for r in guards)
 assert all(Path(r["path"]).suffix in {".py",".sh",".json"}for r in union)
 assert c.ref(c.V4["path"])==c.V4
 tree=ast.parse((ROOT/c.V4["path"]).read_text())
 env=dict(Path=Path,stat=stat,os=os,hashlib=hashlib,json=json,MAX_JSON=32*1024*1024,LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock")
 names={"integer","canonical","file_bytes","parse","bound","paths_overlap","accounting_paths","scope_spec"}
 exec(compile(ast.Module(body=funcs(tree,names),type_ignores=[]),"<exact_original_readonly_scope_spec>","exec"),env)
 request=copy.deepcopy(auth["request"])
 request.update(key="H060/SOURCE_ONLY_v3_actual_route_fixture",output=str(ROOT/"experiments/results/TreeLoom_matched_CPU_full600each_20261017_v3"),
  auxiliary_scopes=[str(ROOT/"research/reviews/TreeLoom_matched_CPU_ROOT_actual_20261017_v3")],consumer_bindings=union)
 checked=env["scope_spec"](ROOT,request)
 try:env["scope_spec"](ROOT,dict(request,consumer_bindings=union+[q["registration_ledger_ref"]]))
 except ValueError as error:assert str(error)=="consumer bindings must be source/runtime authority metadata"
 else:raise AssertionError("Markdown reinsertion accepted")
 assert read(q["root_directive_ref"]["path"])["registration_ledger_ref"]==q["registration_ledger_ref"]
 assert ref(q["registration_ledger_ref"]["path"])==q["registration_ledger_ref"]
 template=read(HERE+"/ROOT_review.INACTIVE.template.v1.json")
 assert template["source_ref"]==ref(ROOT/c.DRIVER)and template["independent"]is False and template["verdict"]=="UNREVIEWED_INACTIVE"
 previous=HERE[:-1]+"2"
 for name in("caps.inactive.v1.json","dependencies.v1.json","scientific_binding.v1.json","scientific_config.INACTIVE.template.v1.json"):
  assert(ROOT/previous/name).read_bytes()==(ROOT/HERE/name).read_bytes()
 assert not any(name=="torch"or name=="numpy"or name.startswith("torch_geometric")or name=="scripts.project_budget_v9"or name=="arggnn.project_budget_v4"for name in sys.modules)
 assert c.SELECTOR_REF is None and c._h._GUARD is None and c._h._SELECTOR_MODULE is None
 return dict(schema="H060_CPU_actual_inactive_module_routing_check_v3",status="PASS_METADATA_ONLY_ACTUAL_ROUTE",
  actual_driver_module=driver.__name__,actual_worker_module=worker.__name__,actual_shared_contract=c.__name__,actual_contract_constants=expected,
  both_executable_ImportFrom_targets=routing,both_actual_default_CLIs_inactive=inactive,run_with_B_and_s=True,
  direct_roles_function_called_from_actual_imported_contract=True,complete_concrete_union_count=len(union),returned_guard_raw_roles=45,returned_guard_unique_refs=38,every_returned_guard_preserved=True,
  original_scope_schema_paths_extensions_and_ref_hashes_pass=True,Markdown_reinsertion_rejected=True,
  inactive_review_template_actual_driver_SHA_matches=True,caps_scientific_binding_config_unchanged=True,
  no_guard_module_object_configuration_policy_mutex_state_or_admission=True,no_scientific_import_model_payload_truth_checkpoint_or_numerical_probe=True,
  concrete_role_fixture_is_preserved_metadata_not_new_ROOT_request=True,future_actual_current_union_still_required=True,
  concrete_consumer_bindings=union,checked_paths=[str(p)for p in checked])
if __name__=="__main__":print(json.dumps(main(),indent=2,sort_keys=True))

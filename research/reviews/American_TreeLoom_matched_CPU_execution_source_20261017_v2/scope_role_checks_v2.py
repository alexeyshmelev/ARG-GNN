"""Exact AST-only scope/role preflight. No guard import, mutex, policy or mutation."""
from pathlib import Path
import ast,copy,hashlib,json,os,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v2"
PREV=HERE[:-1]+"1"

def reference(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads((ROOT/p).read_text())
def functions(t,names):return [n for n in t.body if isinstance(n,ast.FunctionDef)and n.name in names]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 lineage=read(HERE+"/failure_lineage.v2.json")
 for key in ("actual_wait_ref","original_request_ref","preserved_reserve_authority_ref"):
  r=lineage[key];assert reference(r["path"])==r
 auth=read(lineage["preserved_reserve_authority_ref"]["path"])
 q=read(lineage["original_request_ref"]["path"])
 v4path="src/arggnn/project_budget_v4.py"
 assert reference(v4path)["sha256"]=="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47"
 tree=ast.parse((ROOT/v4path).read_text())
 # Exact original predicates AND original bounded ordinary file/hash primitives.
 names={"integer","canonical","file_bytes","parse","bound","paths_overlap","accounting_paths","scope_spec"}
 env=dict(Path=Path,stat=stat,os=os,hashlib=hashlib,json=json,MAX_JSON=32*1024*1024,
  LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock")
 exec(compile(ast.Module(body=functions(tree,names),type_ignores=[]),"<exact_v4_readonly_scope_AST>","exec"),env)
 exact_scope=env["scope_spec"]
 try:exact_scope(ROOT,auth["request"])
 except ValueError as e:assert str(e)=="consumer bindings must be source/runtime authority metadata"
 else:raise AssertionError("Preserved actual failure did not reproduce")
 contract=ast.parse((ROOT/HERE/"root_contract_v1.py").read_text())
 binding=read(HERE+"/scientific_binding.v1.json");deps=read(HERE+"/dependencies.v1.json")
 directenv=dict(deps,binding=lambda:binding,ref=reference,HERE=HERE,SOURCE=HERE+"/root_contract_v1.py",
  DRIVER=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py",MANIFEST=HERE+"/source_manifest.v1.json",HANDOFF=HERE+"/source_handoff.v1.json")
 exec(compile(ast.Module(body=functions(contract,{"direct_roles"}),type_ignores=[]),"<exact_v2_direct_roles_AST>","exec"),directenv)
 packet="research/reviews/TreeLoom_matched_CPU_ROOT_actual_20261017_v1"
 direct=directenv["direct_roles"](lineage["original_request_ref"],q,reference(packet+"/config.json"),reference(packet+"/source_and_input_closure.json"))
 child="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py"
 pure=dict(require=require,copy=copy,_raw=lambda r:env["bound"](ROOT,r))
 exec(compile(ast.Module(body=functions(ast.parse((ROOT/child).read_text()),{"consumer_bindings_v1"}),type_ignores=[]),"<exact_pure_returned_role_union_AST>","exec"),pure)
 guards=list(auth["bindings"].values())
 union=pure["consumer_bindings_v1"](guards,direct)
 assert len(guards)==45 and len({json.dumps(r,sort_keys=True)for r in guards})==38
 assert len(union)==61 and all(r in union for r in guards)
 assert q["registration_ledger_ref"]not in union
 request=copy.deepcopy(auth["request"])
 request.update(key="H060/SOURCE_ONLY_v2_complete_role_fixture",output=str(ROOT/"experiments/results/TreeLoom_matched_CPU_full600each_20261017_v2"),
  auxiliary_scopes=[str(ROOT/"research/reviews/TreeLoom_matched_CPU_ROOT_actual_20261017_v2")],consumer_bindings=union)
 paths=exact_scope(ROOT,request)
 try:exact_scope(ROOT,dict(request,consumer_bindings=union+[q["registration_ledger_ref"]]))
 except ValueError as e:assert str(e)=="consumer bindings must be source/runtime authority metadata"
 else:raise AssertionError("Markdown extension was unexpectedly authorized")
 # Registration raw-ledger read/validation is exact old AST, while direct binding is transitive.
 def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
 prev=ast.parse((ROOT/PREV/"root_contract_v1.py").read_text())
 assert ast.dump(fn(prev,"registration"),include_attributes=False)==ast.dump(fn(contract,"registration"),include_attributes=False)
 assert read(q["root_directive_ref"]["path"])["registration_ledger_ref"]==q["registration_ledger_ref"]
 assert reference(q["registration_ledger_ref"]["path"])==q["registration_ledger_ref"]
 for name in ("caps.inactive.v1.json","dependencies.v1.json","scientific_binding.v1.json","scientific_config.INACTIVE.template.v1.json"):
  assert (ROOT/PREV/name).read_bytes()==(ROOT/HERE/name).read_bytes()
 for name in ("root_contract_v1.py","root_driver_v1.py","worker_entry_v1.py"):compile((ROOT/HERE/name).read_text(),name,"exec")
 assert (ROOT/PREV/"worker_entry_v1.py").read_text().replace(PREV,HERE)==(ROOT/HERE/"worker_entry_v1.py").read_text()
 driver=ast.parse((ROOT/HERE/"root_driver_v1.py").read_text());admit=fn(driver,"admit")
 pre=next(i for i,n in enumerate(admit.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=="scope_spec")
 draft=next(i for i,n in enumerate(admit.body)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=="operation_draft_v1")
 assert pre<draft
 return dict(schema="H060_CPU_exact_scope_role_source_checks_v2",status="PASS_SOURCE_ONLY",guard_modules_imported=False,
  guard_objects_instantiated=False,mutex_policy_state_or_accounting_API_called=False,actual_admission_or_scientific_execution=False,
  exact_scope_source_ref=reference(v4path),original62role_failure_reproduced=True,repaired_complete_union_count=len(union),
  original_guard_raw_roles=45,original_guard_unique_refs=38,every_returned_guard_preserved=True,
  exact_original_scope_schema_paths_extension_and_hash_predicates_pass=True,Markdown_reinsertion_still_rejects=True,
  registration_and_raw_ledger_validation_AST_unchanged=True,ledger_transitive_via_direct_request_and_directive=True,
  caps_scientific_binding_and_config_bytes_unchanged=True,worker_bytes_unchanged_except_successor_module_namespace=True,
  original_scope_preflight_precedes_authority_and_accounting_mutation=True,
  concrete_union_fixture_uses_preserved_failed_attempt_metadata_not_a_new_active_request=True,
  future_actual_union_must_revalidate_current_guards_and_exact_scope_at_admission=True,
  concrete_consumer_bindings=union,checked_paths=[str(p)for p in paths])
if __name__=="__main__":print(json.dumps(main(),indent=2,sort_keys=True))

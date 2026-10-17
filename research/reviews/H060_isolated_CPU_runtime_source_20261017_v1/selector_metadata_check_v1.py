"""Handwritten selector metadata seams only; no guard/project imports or operations."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,json
HERE=Path("research/reviews/H060_isolated_CPU_runtime_source_20261017_v1")
def main():
 profiles=json.loads((HERE/"storage_profiles.v1.json").read_text())
 old=profiles["native7_H050_stopped_retention_v1"];new=profiles["Relate_failed64GiB_retention_v1"]
 old_record=copy.deepcopy(old["selected_record_ref"])
 old_value=json.loads(Path(old_record["path"]).read_text())
 template=json.loads(Path("research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/explicit_selector.INACTIVE.template.v1.json").read_text())
 new_record=dict(path="handwritten/selected.json",sha256="1"*64)
 policy_record=dict(path="handwritten/policy.json",sha256="2"*64)
 activation_record=dict(path="handwritten/activation.json",sha256="3"*64)
 new_value=copy.deepcopy(template)
 new_value.update({k:new[k]for k in("selector_source_ref","source_ref","source_manifest_ref","source_review_ref","measurement_source_ref","repair_authority_ref")})
 new_value.update(active=True,policy_ref=policy_record)
 policy={k:new_value[k]for k in("source_ref","source_manifest_ref","source_review_ref","measurement_source_ref",
  "scheduling_ceiling_bytes","human_limit_bytes","warning_threshold_bytes")}
 policy.update(active=True,root_authorizer="root_agent",selector_source_ref=new["selector_source_ref"],root_activation_ref=activation_record)
 activation=dict(active=True,root_authorizer="root_agent",source_ref=new["source_ref"],source_manifest_ref=new["source_manifest_ref"])
 base_table={old_record["path"]:old_value,new_record["path"]:new_value,policy_record["path"]:policy,activation_record["path"]:activation}
 tree=ast.parse((HERE/"root_contract_v1.py").read_text())
 code=compile(ast.fix_missing_locations(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)
  and x.name in("selected","bind_selector","guard")],type_ignores=[])),"AST_selected_metadata_only","exec")
 def need(ok,message):
  if not ok:raise ValueError(message)
 def scope(table):
  count=[]
  def read(record):
   if record["path"]==str(HERE/"storage_profiles.v1.json"):return copy.deepcopy(profiles)
   return copy.deepcopy(table[record["path"]])
  helper=SimpleNamespace(guard=lambda record,consumer:count.append((record,consumer))or"handwritten_guard_sentinel")
  ns=dict(read=read,ref=lambda path:dict(path=path,sha256="4"*64),raw=lambda record:b"handwritten metadata",
   require=need,HERE=str(HERE),copy=copy,SELECTOR_REF=None,_CHOSEN=None,
   SELECTOR=None,GUARD_REF=None,POLICY_REF=None,_h=helper)
  exec(code,ns);return ns,count
 results=[]
 for label,record in(("native7",old_record),("Relate64",new_record)):
  ns,count=scope(copy.deepcopy(base_table));ns["bind_selector"](record)
  need(ns["guard"](False)=="handwritten_guard_sentinel","Original helper route stub")
  need(ns["guard"](True)=="handwritten_guard_sentinel","Consumer helper route stub")
  need(ns["_h"].SELECTOR==ns["_CHOSEN"]["selector_source_ref"]and len(count)==2,"Exact explicit module supplied to original helper")
  results.append(label+"_explicit_binding_pass")
 ns,_=scope(copy.deepcopy(base_table));ns["bind_selector"](new_record)
 try:ns["bind_selector"](old_record)
 except ValueError:results.append("in_run_selector_swap_rejected")
 else:raise AssertionError("Selector swap accepted")
 mutations=[
  ("inactive_selector",lambda t:t[new_record["path"]].update(active=False)),
  ("wrong_source",lambda t:t[new_record["path"]].update(source_ref=old["source_ref"])),
  ("wrong_review",lambda t:t[new_record["path"]].update(source_review_ref=old["source_review_ref"])),
  ("wrong_ceiling",lambda t:t[new_record["path"]].update(scheduling_ceiling_bytes=980000000001)),
  ("inactive_policy",lambda t:t[policy_record["path"]].update(active=False)),
  ("inactive_activation",lambda t:t[activation_record["path"]].update(active=False)),
  ("planning_policy_source",lambda t:t[policy_record["path"]].update(source_manifest_ref=old["source_manifest_ref"])),
  ("fallback_enabled",lambda t:t[new_record["path"]].update(no_fallback_discovery_upgrade_or_in_run_swap=False)),
 ]
 for label,mutate in mutations:
  table=copy.deepcopy(base_table);mutate(table);ns,_=scope(table)
  try:ns["bind_selector"](new_record)
  except ValueError:results.append(label+"_rejected")
  else:raise AssertionError(label+" accepted")
 return dict(schema="H061_dual_selector_handwritten_metadata_seams_v1",status="PASS",checks=results,
  real_guard_imports_or_calls=0,resource_operations=0,project_imports=0,payloads_opened=0,numerical_work=0,
  actual_activation_or_credit_certified=False,stub_helper_calls_only=True)
if __name__=="__main__":print(json.dumps(main(),indent=2,sort_keys=True))

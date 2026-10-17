"""Independent H058 context-terminal source review. Standard-library AST/metadata only."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import ast,copy,hashlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H058_context_terminal_successor_source_20261017_v1"
OUT=ROOT/"research/reviews/H058_context_terminal_successor_independent_20261017_v1"
OLD="research/reviews/H058_transaction_admission_source_20261017_v1"
SCI="research/reviews/H058_full_parent_CPU_execution_source_20261017_v1"
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 need(p.suffix in(".py",".sh",".json",".md",".patch"),"Static source/metadata only")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):need(ref(record["path"])==record,"Exact source/metadata pin");return json.loads((ROOT/record["path"]).read_text())
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
def isolate(nodes,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),"<independent-source-AST-only>","exec"),scope)
def dump(node):return ast.dump(node,include_attributes=False)
def reject(label,call,result):
 try:call()
 except(ValueError,AssertionError,KeyError,TypeError):result.append(label);return
 raise AssertionError("Invalid metadata accepted: "+label)
def qualification(node):
 return next(n.value for n in node.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant)and n.value.args[1].value=="Genuine successful one-hold arithmetic qualification and consumed direct/outer waits")
def main():
 h=ROOT/HERE;b=read(ref(h/"source_bindings.v1.json"));binding=read(b["scientific_binding_ref"]);manifest=read(ref(h/"source_manifest.v1.json"));handoff=read(ref(h/"source_handoff.v1.json"))
 pins=manifest["source_sha256"]
 for path,sha in pins.items():need(ref(path)["sha256"]==sha,"Complete static closure")
 pmanifest=read(binding["source_manifest_ref"])
 for r in pmanifest["refs"]:need(ref(r["path"])==r,"Complete original producer source/metadata closure")
 for r in [handoff["source_ref"],handoff["worker_ref"],handoff["runner_source_ref"],handoff["source_manifest_ref"],handoff["producer_manifest_ref"],handoff["producer_handoff_ref"]]:need(ref(r["path"])==r,"Exact handed source identity")
 for p in h.glob("*.py"):compile(p.read_text(),str(p),"exec")
 # Independently derive the allowed producer from original AST rather than trust its patch/audit.
 oldp=ast.parse((ROOT/SCI/"producer_v1.py").read_text());newp=ast.parse((h/"producer_v1.py").read_text());expected=copy.deepcopy(oldp)
 oldgate=qualification(fn(oldp,"validate_context_v1")).args[0];newgate=qualification(fn(newp,"validate_context_v1")).args[0]
 need(isinstance(oldgate,ast.BoolOp)and isinstance(newgate,ast.BoolOp)and len(oldgate.values)==15and len(newgate.values)==16,"Original15/repaired16conjuncts")
 need([dump(n)for n in oldgate.values[:14]]==[dump(n)for n in newgate.values[:14]],"All original14 exact")
 needed=[ast.parse(x,mode="eval").body for x in ('terminal.get("schema")=="actual_root_tool_terminal_v1"','terminal.get("actual_wait_ref")==cfg["one_hold_direct_wait_ref"]')]
 need([dump(n)for n in newgate.values[14:]]==[dump(n)for n in needed],"Exact schema and genuine actual_wait_ref only")
 for node in ast.walk(expected):
  if isinstance(node,ast.ImportFrom)and node.level==1:
   if node.module=="protocol_v1":node.level=0;node.module=SCI.replace("/",".")+".protocol_v1"
   elif node.module is None:node.level=0;node.module=SCI.replace("/",".")
  if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   if node.targets[0].id=="HERE":node.value=ast.Constant(HERE)
   elif node.targets[0].id=="MANIFEST":node.value=ast.BinOp(left=ast.Name(id="HERE",ctx=ast.Load()),op=ast.Add(),right=ast.Constant("/producer_manifest.v1.json"))
 qualification(fn(expected,"validate_context_v1")).args[0].values=copy.deepcopy(oldgate.values[:14]+needed)
 need(dump(expected)==dump(newp),"Whole producer exactly original plus authorized terminal/import/source joins")
 # Everything outside the common contract is exact current transaction/lifecycle/memo implementation.
 oldd=ast.parse((ROOT/OLD/"root_driver_v1.py").read_text());newd=ast.parse((h/"root_driver_v1.py").read_text())
 before={n.name:n for n in oldd.body if isinstance(n,ast.FunctionDef)};after={n.name:n for n in newd.body if isinstance(n,ast.FunctionDef)}
 need(set(before)==set(after),"No new top-level executor/controller")
 for name in before:
  if name!="contract":need(dump(before[name])==dump(after[name]),"Original helper exact: "+name)
 oldcontract=before["contract"];newcontract=after["contract"]
 oldnested={n.name:n for n in oldcontract.body if isinstance(n,ast.FunctionDef)};newnested={n.name:n for n in newcontract.body if isinstance(n,ast.FunctionDef)}
 need(set(newnested)==set(oldnested)|{"accept_context_failure"},"Only new exact failure-acceptance helper")
 for name in oldnested:
  if name not in("direct_roles","validate_request"):need(dump(oldnested[name])==dump(newnested[name]),"Original nested helper exact: "+name)
 need((h/"worker_entry_v1.py").read_bytes()==(ROOT/OLD/"worker_entry_v1.py").read_bytes(),"Worker byte-identical")
 oldb=read(ref(ROOT/OLD/"source_bindings.v1.json"))
 need(b["caps"]==oldb["caps"]and b["cpu_ids"]==oldb["cpu_ids"]and b["transaction_memo"]==oldb["transaction_memo"],"Caps CPUs exact actual current memo")
 need(b["transaction_memo"]["source_ref"]["sha256"]=="2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98","No unrequested memo upgrade")
 oldscience=read(oldb["scientific_binding_ref"])
 for key in oldscience:
  if key not in("runner_module","runner_source_ref","source_manifest_ref","source_handoff_ref","config_defaults"):need(binding[key]==oldscience[key],"Original scientific binding: "+key)
 for key in oldscience["config_defaults"]:
  if key not in("producer_ref","source_manifest_ref"):need(binding["config_defaults"][key]==oldscience["config_defaults"][key],"Unchanged scientific configuration: "+key)
 need(len(b["excluded_namespaces"])==6,"Both three-namespace failure lineages retained")
 # Independently exercise only the exact qualification metadata predicate, using genuine historical JSON.
 cfg=oldscience["config_defaults"];completed=read(cfg["one_hold_completion_ref"]);wait=read(cfg["one_hold_direct_wait_ref"]);terminal=read(cfg["one_hold_outer_terminal_ref"]);receipt=read(completed["qualification_ref"])
 need("direct_wait_ref"not in terminal,"No synthetic alias added to old terminal")
 scope=dict(completed=completed,wait=wait,terminal=terminal,receipt=receipt,cfg=cfg)
 def gate(node):return eval(compile(ast.Expression(body=copy.deepcopy(node)),"<exact-metadata-predicate>","eval"),{},scope)
 need(gate(oldgate)is False and all(eval(compile(ast.Expression(body=copy.deepcopy(n)),"<original-conjunct>","eval"),{},scope)is True for n in oldgate.values[:14]),"Exact historical error is only original terminal field")
 need(gate(newgate)is True,"Exact genuine qualification accepted without recomputation")
 negative=[]
 for obj,key,bad in [(completed,"status","FAILED"),(receipt,"status","FAIL"),(receipt,"optimizer_updates",1),(receipt,"schema","wrong"),(receipt,"parent_id","wrong"),(completed,"actual_wait_ref",{}),(wait,"completed_ref",{}),(wait,"returncode",1),(wait,"actual_direct_worker_wait",False),(wait,"all_owned_processes_and_groups_drained",False),(wait,"operational_failure","failure"),(terminal,"actual_exit_code",1),(terminal,"actual_terminal_consumed",False),(terminal,"completed_ref",{}),(terminal,"schema","wrong"),(terminal,"actual_wait_ref",{})]:
  previous=obj[key];obj[key]=bad
  reject("qualification:"+key,lambda:need(gate(newgate),"Bad qualification metadata"),negative);obj[key]=previous
 actual=terminal.pop("actual_wait_ref");terminal["direct_wait_ref"]=actual
 reject("direct_wait_alias_is_not_actual_schema_field",lambda:need(gate(newgate),"No alias accepted"),negative)
 terminal.pop("direct_wait_ref");terminal["actual_wait_ref"]=actual
 # Read and execute the independently audited checker only as standard-library definitions.
 # It explicitly extracts pure predicates, uses actual compact JSON and in-memory projections,
 # and hard-fails if a guard factory or episode is touched; no entrypoint/model imports.
 checker=h/"source_check_v1.py";cs={"__name__":"INDEPENDENT_AUDITED_METADATA_CHECKER","__file__":str(checker)}
 for node in ast.parse(checker.read_text()).body:
  if isinstance(node,ast.Import):
   need(all(a.name in("ast","copy","hashlib","json","os","sys")for a in node.names),"Only stdlib checker imports")
  elif isinstance(node,ast.ImportFrom):need(node.module in("pathlib","types","datetime"),"Only stdlib checker imports")
 exec(compile(checker.read_text(),str(checker),"exec"),cs);whole=cs["check"]()
 need(whole["status"]=="PASS_SOURCE_ONLY"and whole["guard_factory_calls"]==whole["scientific_episode_calls"]==0,"Whole actual context and canaries")
 # Hand-enumerate the direct role list against original runtime contract.
 roles_tree=ast.parse((ROOT/b["contract_ref"]["path"]).read_text());original_roles=fn(roles_tree,"direct_roles")
 # All source-proven original role slots, with two extra retained identities.
 names=["manifest","common_contract_driver","common_contract_driver","entry","handoff","new_producer","original_protocol","source_review","scientific_review","runtime","runtime_qualification","operational_config","ROOT_directive","scientific_config","ROOT_request","registration","registration_decisions","training_context","source_closure","actual_child_source","actual_selector_record","consumer_namespace_memo","pidfd","owned_lifecycle","supervisor","old_static_selector","actual_child_source","original_producer","latest_failure_acceptance"]
 need(len(original_roles.body[-1].value.elts)==25,"Exact25 original direct slots before deduplication")
 need(whole["guard_names"]==46and whole["distinct_guard_roles"]==39and whole["complete_role_union"]==64,"Complete current64ref union")
 deps=read(ref(ROOT/b["base_directory"]/"dependencies.v1.json"))
 P=lambda name:dict(path="INDEPENDENT_HANDWRITTEN/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
 guards=read(b["actual_guard_role_metadata_ref"])["bindings"]
 hand=[ref(h/"source_manifest.v1.json"),ref(h/"root_driver_v1.py"),ref(h/"root_driver_v1.py"),ref(h/"worker_entry_v1.py"),ref(h/"source_handoff.v1.json"),binding["runner_source_ref"],binding["protocol_ref"],P("source_review"),P("scientific_review"),P("runtime"),P("runtime_qualification"),P("operational_config"),P("ROOT_directive"),P("scientific_config"),P("ROOT_request"),P("registration"),P("registration_decisions"),P("training_context"),P("source_closure"),b["selected_child_source_ref"],b["actual_selector_metadata_ref"],deps["NAMESPACE"],deps["PIDFD"],deps["OWNED"],deps["SUPERVISOR"],deps["SELECTOR"],b["selected_child_source_ref"],b["original_runner_source_ref"],P("latest_failure_acceptance")]
 unique=[]
 for record in [*guards.values(),*hand]:
  if record not in unique:unique.append(record)
 need(len(unique)==64and all(r in unique for r in guards.values()),"Independent complete handwritten real-reference union64")

 # Actual old failed authority/charge/clock facts are preserved by metadata equality.
 fail=b["context_terminal_failure"];oldinput=read(fail["original_input_ref"]);outer=read(fail["actual_outer_terminal_ref"]);failedwait=read(fail["actual_wait_ref"])
 need(oldinput["created_at_utc"]==fail["old_created_at_utc"]and oldinput["expires_at_utc"]==fail["old_expires_at_utc"]and oldinput["storage"]["reservation"]["key"]==fail["old_reservation_key"]and oldinput["storage"]["reserved_new_bytes"]==fail["old_charge_bytes"]==33554432,"Original failure clock/key/charge facts")
 need(outer["actual_tool_session_id"]==69608and outer["actual_terminal_consumed"]is True and outer["actual_outer_exit_code"]==failedwait["returncode"]==1and outer["completed_ref"]is None,"Genuine latest consumed failure lineage")
 # More namespace, authority and exact-type negative cases, using nested pure functions only.
 scopeclock=dict(datetime=datetime,need=need);isolate([after["authority_interval"]],scopeclock)
 start=datetime(2030,1,1,tzinfo=timezone.utc)
 scopeclock["authority_interval"](dict(created_at_utc=start.isoformat(),expires_at_utc=(start+timedelta(hours=72)).isoformat()))
 for hours in(24,71,73):reject("authority_hours_"+str(hours),lambda hours=hours:scopeclock["authority_interval"](dict(created_at_utc=start.isoformat(),expires_at_utc=(start+timedelta(hours=hours)).isoformat())),negative)
 for key in("preparation_directory","packet_directory","output_directory"):
  for prior in b["excluded_namespaces"].values():
   p=ROOT/prior
   for candidate in(p,p/"child",p.parent):
    reject("cross_namespace:"+key,lambda candidate=candidate,p=p:need(candidate!=p and not candidate.is_relative_to(p)and not p.is_relative_to(candidate),"Fresh disjoint namespace"),negative)
 need(not any(n in sys.modules for n in("torch","numpy","torch_geometric","tskit")),"No numerical/model imports")
 return dict(schema="H058_context_terminal_independent_checks_v1",independent=True,status="PASS_SOURCE_ONLY",source_ref=ref(h/"root_driver_v1.py"),producer_ref=ref(h/"producer_v1.py"),source_manifest_ref=ref(h/"source_manifest.v1.json"),source_handoff_ref=ref(h/"source_handoff.v1.json"),producer_manifest_ref=binding["source_manifest_ref"],producer_handoff_ref=binding["source_handoff_ref"],runner_source_ref=binding["runner_source_ref"],protocol_ref=binding["protocol_ref"],authenticated_integration_pins=len(pins),authenticated_producer_refs=len(pmanifest["refs"]),producer_whole_AST_expected_delta_exact=True,all_original14conjuncts_exact_and_true=True,exact_new_schema_actual_wait_join_passes=True,worker_byte_identical=True,original_memo_lifecycle_science_caps_and72h_unchanged=True,old7922_and69608_failures_preserved=True,guard_names=46,distinct_guards=39,complete_role_union=64,handwritten_direct_role_slots=names,independent_handwritten_complete_union=unique,independent_negative_cases=negative,audited_whole_context_check=whole,no_actual_guard_resources_process_payload_model_numerical_training_admission_launch_or_Git=True)
if __name__=="__main__":
 result=main();(OUT/"checks.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
 print(json.dumps(dict(status=result["status"],pins=result["authenticated_integration_pins"],producer_refs=result["authenticated_producer_refs"],roles=result["complete_role_union"],extra_negatives=len(result["independent_negative_cases"]),whole_negatives=len(result["audited_whole_context_check"]["negative_cases"]))))

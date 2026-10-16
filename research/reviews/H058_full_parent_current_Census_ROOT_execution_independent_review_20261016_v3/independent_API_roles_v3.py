from pathlib import Path
import ast,copy,hashlib,json,types
R=Path("/workspace-SR003.nfs2/ARG-GNN")
B=R/"research/reviews/H058_full_parent_current_Census_ROOT_execution_source_20261016_v3"
O=R/"research/reviews/H058_full_parent_current_Census_ROOT_execution_independent_review_20261016_v3"
def run():
 plan=json.loads((B/"direct_role_plan.v3.json").read_text());m=json.loads((B/"source_manifest.v1.json").read_text())
 oldplan=json.loads((R/"research/reviews/native14_stopped_retention_successor_source_20261016_v1/direct_role_plan.v1.json").read_text())
 norm=lambda rs:{(r["path"],r["sha256"]) for r in rs}
 assert norm(plan["current_Census23_refs"])==norm(oldplan["current_Census23_refs"])
 common=ast.parse((B/"common_v1.py").read_text());driver=ast.parse((B/"root_driver_v1.py").read_text())
 constants={}
 for n in common.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
   try:constants[n.targets[0].id]=ast.literal_eval(n.value)
   except (ValueError,TypeError):pass
 def reference(path):return dict(path=path,sha256=hashlib.sha256((R/path).read_bytes()).hexdigest())
 def fake(name):return dict(path="SYNTHETIC_ONLY/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
 def require(ok,message):
  if not ok:raise ValueError(message)
 fn=next(n for n in driver.body if isinstance(n,ast.FunctionDef) and n.name=="admit_v1");needed_nodes=[]
 for n in fn.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="needed" for t in n.targets):needed_nodes.append(copy.deepcopy(n))
  if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=="needed":needed_nodes.append(copy.deepcopy(n))
  if isinstance(n,ast.If) and any(isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=="needed" for x in n.body):
   node=copy.deepcopy(n);node.body=[x for x in node.body if not isinstance(x,ast.ImportFrom)];needed_nodes.append(node)
 unique=next(n for n in common.body if isinstance(n,ast.FunctionDef) and n.name=="unique");results={}
 for phase in ("train","evaluate"):
  q=dict(phase=phase,config_ref=fake("active_scientific_config"),science_handoff_ref=m["scientific_handoff_ref"],source_review_ref=fake("independent_v3_SOURCE_PASS_review"),direct_attachment_authority_ref=fake("TRAIN_borrowed_attachment_authority"),native_noncoexistence_qualification_ref=fake("TRAIN_native_noncoexistence_qualification"))
  neededroles=dict(request_ref=fake("phase_request"),runtime=fake("phase_runtime"),closure=fake("phase_consumer_manifest"))
  cfg=dict(source_manifest_ref=m["scientific_source_manifest_ref"])
  producer=next(r for r in json.loads((R/m["scientific_source_manifest_ref"]["path"]).read_text())["refs"] if r["path"].endswith("/producer_v1.py"))
  env=dict(constants,q=q,cfg=cfg,producer=producer,direct=types.SimpleNamespace(CURRENT_CAP=reference("research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json")),**neededroles)
  env.update(MANIFEST=str(B.relative_to(R)/"source_manifest.v1.json"),DRIVER=str(B.relative_to(R)/"root_driver_v1.py"),CONTEXT=str(B.relative_to(R)/"context_factory_v1.py"),DIRECT=str(B.relative_to(R)/"direct_gpu0_v1.py"))
  records={r["path"]:r for r in plan["current_Census23_refs"]+plan["phases"][phase]["concrete_refs"]}
  for r in list(q.values())+list(neededroles.values()):
   if isinstance(r,dict) and set(r)=={"path","sha256"}:records[r["path"]]=r
  env.update(require=require,ref=lambda path:records[path])
  exec(compile(ast.Module(body=[unique,*needed_nodes],type_ignores=[]),"<actual-admit-role-assembly-with-fake-future-refs>","exec"),env)
  allrefs=env["unique"]([*plan["current_Census23_refs"],*env["needed"]])
  actual=[r for r in allrefs if not r["path"].startswith("SYNTHETIC_ONLY/")]
  assert norm(actual)==norm(plan["phases"][phase]["concrete_refs"])
  assert len(allrefs)==plan["phases"][phase]["complete_upper_union"]
  results[phase]=dict(actual_AST_derived_unique_count=len(allrefs),concrete_unique_count=len(actual),future_fake_roles=len(allrefs)-len(actual))
 guard=R/"research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py"
 gtext=guard.read_text();g=ast.parse(gtext);cls=next(n for n in g.body if isinstance(n,ast.ClassDef) and n.name=="PilotOutputV9")
 save=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="save")
 body=ast.get_source_segment(gtext,save);before="self.used+planned<=self.storage['reserved_new_bytes']";assert body.count(before)==1
 changed=ast.parse(body.replace(before,"self.used+planned<=min(self.storage['reserved_new_bytes'],256*2**20)")).body[0];new=copy.deepcopy(save)
 matches=[n for n in ast.walk(new) if isinstance(n,ast.Compare) and ast.dump(n,include_attributes=False)==ast.dump(ast.parse(before,mode="eval").body,include_attributes=False)]
 assert len(matches)==1
 matches[0].comparators[0]=ast.parse("min(self.storage['reserved_new_bytes'],256*2**20)",mode="eval").body
 assert ast.dump(new,include_attributes=False)==ast.dump(changed,include_attributes=False)
 apis={}
 for n in g.body:
  if isinstance(n,ast.FunctionDef) and n.name in ("configure_with_renewal_v1","_state_locked","operation_draft_v1","apply_reserve_v1","validate_storage_v9","budget_check_v9"):apis[n.name]=ast.unparse(n.args)
 assert set(apis)=={"configure_with_renewal_v1","_state_locked","operation_draft_v1","apply_reserve_v1","validate_storage_v9","budget_check_v9"}
 return dict(schema="independent_H058_actual_API_roles_and_writer_source_checks_v3",current23_matches_previously_independently_reconstructed_actual_guard_chain=True,role_results=results,original_direct64_ceiling=True,writer_AST_only_one_local_counter_cap_expression_changed=True,actual_current_Census_API_signatures=apis,source_only=True,no_guard_accounting_mutation_or_payload_access=True)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))

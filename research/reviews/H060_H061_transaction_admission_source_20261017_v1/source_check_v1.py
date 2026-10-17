"""Source-only integration checks. No original project module/guard/resource import."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,subprocess,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H060_H061_transaction_admission_source_20261017_v1"
OLD="research/reviews/H060_H061_child_guard_role_repair_source_20261017_v3"
def need(ok,message):
 if not ok:raise AssertionError(message)
def ref(path):
 p=ROOT/path
 return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def tree(path):return ast.parse((ROOT/path).read_bytes())
def node(path,name):
 found=[n for n in tree(path).body if isinstance(n,ast.FunctionDef)and n.name==name]
 need(len(found)==1,"one source function "+name);return copy.deepcopy(found[0])
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def isolated(nodes,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),"<isolated-source-AST>","exec"),scope)
 return scope
def fake_ref(path):
 return dict(path=str(path),sha256=hashlib.sha256(str(path).encode()).hexdigest())
def rejects(call):
 try:call()
 except(ValueError,AssertionError):return
 raise AssertionError("expected rejection")
def check():
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"ROOT -B -s")
 b=json.loads((ROOT/HERE/"source_bindings.v1.json").read_text())
 current=tree(HERE+"/root_driver_v1.py");prior=tree(OLD+"/root_driver_v1.py")
 names={"_make_root_mutation_memo","_memo_admission_node","_install_memo_admit"}
 # Inertness inspection BEFORE any actual entrypoint execution.
 for n in current.body:
  need(isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef,ast.If,ast.Expr)),"safe top-level statement")
  if isinstance(n,ast.Assign):
   need(not any(isinstance(x,ast.Call)and ast.unparse(x)!="Path('/workspace-SR003.nfs2/ARG-GNN')"for x in ast.walk(n)),"no default source/guard IO")
  if isinstance(n,ast.If):
   need(ast.unparse(n.test)=="__name__ == '__main__'","only main top-level conditional")
 main=next(n for n in current.body if isinstance(n,ast.FunctionDef)and n.name=="main")
 branch=next(n for n in main.body if isinstance(n,ast.If))
 need(any(isinstance(n,ast.Return)and isinstance(n.value,ast.Constant)and n.value.value==0 for n in branch.body),"noargs early return")
 for fn in ("main","worker","execute_or_admit"):
  compile((ROOT/HERE/"root_driver_v1.py").read_bytes(),HERE+"/root_driver_v1.py","exec")
 worker=tree(HERE+"/worker_entry_v1.py")
 need(any(isinstance(n,ast.ImportFrom)and n.level==1 and n.module is None and n.names[0].name=="root_driver_v1"for n in worker.body),"package relative worker")
 need((ROOT/HERE/"worker_entry_v1.py").read_bytes()==(ROOT/OLD/"worker_entry_v1.py").read_bytes(),"exact v3 worker")
 # Reverse own namespace/import/dispatch additions; every retained v3 AST must match.
 restored=copy.deepcopy(current);restored.body=[n for n in restored.body if not(isinstance(n,ast.FunctionDef)and n.name in names)]
 for n in restored.body:
  if isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str):
   n.value.value=prior.body[0].value.value
  if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=="HERE"for x in n.targets):n.value.value=OLD
  if isinstance(n,ast.Import):n.names=[x for x in n.names if x.name!="ast"]
  if isinstance(n,ast.FunctionDef)and n.name=="execute_or_admit":
   n.body=[x for x in n.body if not(isinstance(x,ast.If)and isinstance(x.test,ast.Name)and x.test.id=="admission"and len(x.body)==1 and isinstance(x.body[0],ast.Expr)and isinstance(x.body[0].value,ast.Call)and isinstance(x.body[0].value.func,ast.Name)and x.body[0].value.func.id=="_install_memo_admit")]
 need(same(restored,prior),"exact factory/worker/executor v3 delta")
 ns=isolated([node(HERE+"/root_driver_v1.py",name)for name in("need","append_selected_source","_memo_admission_node")],dict(ast=ast,copy=copy))
 delta={};originals={};derived={}
 for consumer,spec in b["consumers"].items():
  original=node(spec["driver_ref"]["path"],"admit");changed=ns["_memo_admission_node"](original,consumer)
  reverse=copy.deepcopy(changed);originals[consumer]=original;derived[consumer]=changed
  reverse.body=[n for n in reverse.body if not(isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=="_make_root_mutation_memo")]
  original_state=next(n for n in original.body if isinstance(n,ast.With))
  for i,n in enumerate(reverse.body):
   if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute):
    if n.value.func.attr=="read_state_v1":reverse.body[i]=copy.deepcopy(original_state)
    if n.value.func.attr=="reserve_v1":n.value.func=ast.Attribute(value=ast.Name(id="g",ctx=ast.Load()),attr="apply_reserve_v1",ctx=ast.Load())
  need(same(reverse,original),"reversed exact admit AST "+consumer)
  need(not any(isinstance(n,ast.With)for n in ast.walk(changed)),"no outer/nested lock")
  gi=next(i for i,n in enumerate(changed.body)if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=="g"for x in n.targets))
  need(ast.unparse(changed.body[gi+1])=="memo = _make_root_mutation_memo(g, q)","memo immediately after guard")
  first_put=next(i for i,n in enumerate(changed.body)if any(isinstance(x,ast.Call)and ast.unparse(x.func)in("put","c.put")for x in ast.walk(n)))
  need(gi+1<first_put,"memo preflight before first staging write")
  union=next(i for i,n in enumerate(changed.body)if "consumer_bindings_v1" in ast.unparse(n))
  scope=next(i for i,n in enumerate(changed.body)if "scope_spec" in ast.unparse(n))
  authority=next(i for i,n in enumerate(changed.body)if "operation_draft_v1" in ast.unparse(n))
  reserve=next(i for i,n in enumerate(changed.body)if "memo.reserve_v1" in ast.unparse(n))
  need(union<scope<authority<reserve,"original complete role preflight before authority/reserve")
  delta[consumer]=dict(exact_reverse_AST=True,memo_before_first_put=True,no_outer_lock=True,original_union_scope_authority_order=True)
 # Pure actual role construction and exact consumer-union predicate on handwritten refs.
 fixture=json.loads((ROOT/HERE/"handwritten_role_fixture.v1.json").read_text())
 selected=fixture["actual_selected_source_ref"];guards=list(fixture["actual_guard_role_refs_by_name"].values())
 union_scope=isolated([node(b["selected_child_source_ref"]["path"],"consumer_bindings_v1")],dict(copy=copy,require=need,_raw=lambda record:need(isinstance(record,dict)and set(record)=={"path","sha256"},"handwritten exact ref")))
 union_fn=union_scope["consumer_bindings_v1"];cases=[]
 for case in fixture["cases"]:
  consumer=case["consumer"];stage=case["stage"];spec=b["consumers"][consumer]
  deps=json.loads((ROOT/spec["base_directory"]/"dependencies.v1.json").read_text())
  binding=json.loads((ROOT/spec["scientific_binding_ref"]["path"]).read_text())
  scope={name:deps[name]for name in("NAMESPACE","PIDFD","OWNED","SUPERVISOR")}
  scope.update(ref=fake_ref,binding=lambda binding=binding:binding,MANIFEST=HERE+"/source_manifest.v1.json",SOURCE=HERE+"/root_driver_v1.py",DRIVER=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py",HANDOFF=HERE+"/source_handoff.v1.json",DEPS=deps)
  direct=isolated([node(spec["contract_ref"]["path"],"direct_roles")],scope)["direct_roles"]
  q={key:fake_ref("fixture/"+key+".json")for key in("source_review_ref","scientific_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref","registration_ref","registration_decisions_ref","training_context_ref","data_context_ref","train_acceptance_ref")}
  q["storage_policy_selector_ref"]=fixture["actual_selector_metadata_ref"]
  q["freeze_acceptance_ref"]=fake_ref("fixture/freeze.json")if stage in("predict","score")else None
  q["prediction_acceptance_ref"]=fake_ref("fixture/predict.json")if stage=="score"else None
  roles=direct(fake_ref("fixture/request.json"),q,fake_ref("fixture/config.json"),fake_ref("fixture/closure.json"))
  repaired=ns["append_selected_source"](roles,selected,selected);combined=union_fn(guards,repaired)
  need(len(combined)==case["expected_distinct_union"],"actual distinct role baseline "+consumer+stage)
  need(all(record in combined for record in guards)and selected in combined,"all current guards+selected child preserved")
  need(repaired[:len(roles)]==roles and ns["append_selected_source"](repaired,selected,selected)==repaired,"ordered idempotent role repair")
  need(b["transaction_memo"]["source_ref"]not in combined,"memo is transitive, no extra role")
  overflow=list(combined)
  while len(overflow)<65:overflow.append(fake_ref("fixture/extra_%d.json"%len(overflow)))
  rejects(lambda:union_fn(overflow))
  cases.append(dict(consumer=consumer,stage=stage,distinct_union=len(combined),all_guards_preserved=True,selected_child=True,memo_direct_role=False,role65_rejected=True))
 # Handwritten admission traces exercise the actual derived original AST.
 class FakePath:
  def __init__(self,path):self.path=str(path);self.name=self.path.rsplit("/",1)[-1]
  def exists(self):return False
  def __truediv__(self,name):return FakePath(self.path+"/"+str(name))
  def __str__(self):return self.path
 class StopFirstPut(Exception):pass
 traces=[]
 for consumer in("H060","H061"):
  for fail_memo in(True,False):
   events=[];q=dict(packet_directory="fixture/packet",output_directory="fixture/output",preparation_directory="fixture/prep",cpu_ids=[223],stage="freeze")
   guard=SimpleNamespace()
   def put(path,value):events.append("first_put");raise StopFirstPut()
   def memo_factory(g,request):
    events.append("memo_preflight")
    if fail_memo:raise ValueError("handwritten memo rejection")
    return SimpleNamespace()
   def get_guard(flag):events.append("guard_false");need(flag is False,"original guard mode");return guard
   c=SimpleNamespace(validate_request=lambda r:(q,{},{}),local=FakePath,require=need,guard=get_guard,put=put,make_config=lambda q,caps:{})
   scope=dict(c=c,need=need,local=FakePath,guard=get_guard,put=put,os=SimpleNamespace(sched_getaffinity=lambda _:set([223])),_make_root_mutation_memo=memo_factory)
   admit=isolated([derived[consumer]],scope)["admit"]
   try:admit(fake_ref("fixture/request.json"))
   except(ValueError,StopFirstPut):pass
   need(events==["guard_false","memo_preflight"]+([]if fail_memo else["first_put"]),"memo boundary before any actual put")
   traces.append(dict(consumer=consumer,memo_rejected=fail_memo,events=events))
 # Authenticate construction against handwritten manifests/reviews before any import.
 helper=node(HERE+"/root_driver_v1.py","_make_root_mutation_memo")
 helper_cases=[]
 for mode in("success","missing_helper","missing_review","review_verdict"):
  events=[];manifest={"source_sha256":{record["path"]:record["sha256"]for record in b["transaction_memo"].values()}}
  manifest["source_sha256"]["src/fixture.py"]="1"*64;manifest["source_sha256"]["research/authority.json"]="2"*64
  review={"verdict":"SOURCE_PASS","source_only":True,"source_ref":b["transaction_memo"]["source_ref"]}
  if mode=="missing_helper":manifest["source_sha256"].pop(b["transaction_memo"]["source_ref"]["path"])
  if mode=="missing_review":manifest["source_sha256"].pop(b["transaction_memo"]["independent_review_ref"]["path"])
  if mode=="review_verdict":review["verdict"]="SOURCE_BLOCK"
  request={"source_manifest_ref":fake_ref(HERE+"/source_manifest.v1.json")}
  def read_fixture(record):
   return manifest if record==request["source_manifest_ref"]else review
  def construct(g,**kwargs):
   events.append("construct")
   need(kwargs["integration_manifest_ref"]==request["source_manifest_ref"]and kwargs["adapter_ref"]==b["transaction_memo"]["source_ref"],"exact memo constructor refs")
   need(all(Path(x["path"]).suffix in(".py",".sh")for x in kwargs["cache_refs"]),"code only cache closure")
   return "handwritten memo"
  def import_fixture(record,name):
   events.append("load_memo");return SimpleNamespace(RootMutationMemo=construct)
  helper_fn=isolated([helper],dict(bindings=lambda:b,read=read_fixture,raw=lambda r:events.append("authenticate_ref"),need=need,Path=Path,load=import_fixture))["_make_root_mutation_memo"]
  if mode=="success":need(helper_fn(object(),request)=="handwritten memo","successful helper boundary")
  else:
   rejects(lambda:helper_fn(object(),request));need("load_memo"not in events and "construct"not in events,"failed component preflight before import/construction")
  helper_cases.append(dict(case=mode,constructed="construct"in events))
 # Exact original union and scope preflight still reject before authority.
 union_traces=[]
 scope_node=node("src/arggnn/project_budget_v4.py","scope_spec")
 class StopAuthority(Exception):pass
 for consumer in("H060","H061"):
  for role_count in(64,65):
   events=[];q=dict(packet_directory="fixture/packet",output_directory="fixture/output",preparation_directory="fixture/prep",cpu_ids=[223],stage="freeze")
   roles=[fake_ref("fixture/role_%d.json"%i)for i in range(role_count)]
   def put_fixture(path,value):events.append("staging_put");return fake_ref(str(path))
   def union_fixture(*groups):events.append("role_union");return union_fn(*groups)
   def integer_fixture(value,label,minimum):need(type(value)is int and value>=minimum,"bounded charge")
   actual_scope=isolated([scope_node],dict(Path=Path,integer=integer_fixture,canonical=lambda root,name:FakePath(name),paths_overlap=lambda a,b:False,accounting_paths=lambda root:[],bound=lambda *args:None))["scope_spec"]
   def scope_fixture(root,value):events.append("scope_spec");return actual_scope(root,value)
   def draft(*args,**kwargs):events.append("authority");raise StopAuthority()
   guard=SimpleNamespace(budget=SimpleNamespace(SOURCE_REL="fixture/implementation.py",LAUNCHER_REL="fixture/launcher.sh",v4=SimpleNamespace(scope_spec=scope_fixture)),consumer_bindings_v1=union_fixture,_policy=lambda:dict(never_credit_keys=[]),_pins=lambda state:{},operation_draft_v1=draft)
   def get_guard(flag):events.append("guard_false");return guard
   def state_fixture(**kwargs):events.append("memo_state");return ({},None,{}, {})
   def memo_factory(g,request):events.append("memo_preflight");return SimpleNamespace(read_state_v1=state_fixture)
   c=SimpleNamespace(ROOT=ROOT,validate_request=lambda r:(q,dict(reservation_bytes=32*2**20),{}),local=FakePath,require=need,guard=get_guard,put=put_fixture,make_config=lambda q,caps:{},sources=lambda:{"source_sha256":{}},ref=fake_ref,direct_roles=lambda *args:roles)
   scope=dict(c=c,ROOT=ROOT,need=need,local=FakePath,guard=get_guard,put=put_fixture,ref=fake_ref,os=SimpleNamespace(sched_getaffinity=lambda _:set([223])),_make_root_mutation_memo=memo_factory)
   admit=isolated([derived[consumer]],scope)["admit"]
   try:admit(fake_ref("fixture/request.json"))
   except(AssertionError,StopAuthority):pass
   need(events[:4]==["guard_false","memo_preflight","staging_put","staging_put"]and events[4:6]==["memo_state","role_union"],"unchanged staging/state/union order")
   need(events[6:]==(["scope_spec","authority"]if role_count==64 else[]),"64 accepted and65 rejected before authority")
   union_traces.append(dict(consumer=consumer,role_count=role_count,events=events,authority_created=False))
 # Asset joins audited from original literal HERE-relative reads.
 assets=[]
 for consumer,spec in b["consumers"].items():
  source=(ROOT/spec["contract_ref"]["path"]).read_text()
  for suffix in("/storage_profiles.v1.json",)+ (("/scientific_config.INACTIVE.template.v1.json",)if consumer=="H061"else()):
   need(suffix in source,"expected original asset read")
   original=spec["base_directory"]+suffix;current_asset=HERE+suffix
   need(ref(original)["sha256"]==ref(current_asset)["sha256"],"exact HERE-relative asset")
   assets.append(dict(consumer=consumer,original_ref=ref(original),integration_ref=ref(current_asset)))
  need(ref(spec["scientific_binding_ref"]["path"])==spec["scientific_binding_ref"],"original scientific binding immutable")
 need(OLD not in (ROOT/HERE/"root_driver_v1.py").read_text(),"no stale self slash namespace")
 need(OLD.replace("/",".")not in (ROOT/HERE/"worker_entry_v1.py").read_text(),"no stale dotted worker namespace")
 # Static inspection complete. Only now run the two explicitly authorized inert noargs paths.
 inactive={}
 for entry in("root_driver_v1","worker_entry_v1"):
  module=HERE.replace("/",".")+"."+entry
  code="import runpy,sys,json\nsys.argv=["+repr(module)+"]\ntry:runpy.run_module("+repr(module)+",run_name='__main__')\nexcept SystemExit as e:\n assert e.code in (None,0)\nforbidden=[n for n in sys.modules if n.startswith(('torch','numpy','arggnn','_child_source_repair','_fresh_ROOT_transaction_memo')) or 'relate_failed64_retention' in n]\nassert not forbidden, forbidden\nprint(json.dumps(dict(import_audit_pass=True,forbidden_imports=forbidden)))"
  completed=subprocess.run(["/usr/bin/python3","-B","-s","-c",code],cwd=ROOT,text=True,capture_output=True,timeout=20)
  need(completed.returncode==0,"inert noargs "+entry+" "+completed.stdout+completed.stderr)
  lines=[json.loads(line)for line in completed.stdout.splitlines()]
  need(lines[0].get("status")=="SOURCE_ONLY_INACTIVE"and lines[0].get("guard_imported")is False and lines[1]["import_audit_pass"],"actual inert entrypoint")
  inactive[module]=lines
 return dict(schema="H060_H061_transaction_admission_source_checks_v1",source_only=True,static_inertness_before_entrypoint_execution=True,exact_v3_factory_worker_executor_delta=True,admission_delta=delta,handwritten_actual_role_cases=cases,handwritten_admission_boundaries=traces,handwritten_memo_preflight_cases=helper_cases,handwritten_union_authority_boundaries=union_traces,exact_here_relative_assets=assets,worker_package_relative=True,inactive_noargs=inactive,original_guard_scientific_or_contract_modules_imported=False,actual_guard_mutex_state_resource_job_numerical_operations=False,run_authority=False)
if __name__=="__main__":
 try:print(json.dumps(check(),indent=2,sort_keys=True))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)

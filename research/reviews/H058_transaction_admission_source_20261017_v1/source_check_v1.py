"""Inactive H058 source integration checks; isolated source AST and memory fixtures."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,subprocess,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H058_transaction_admission_source_20261017_v1"
OLD="research/reviews/H058_fresh_CPU_lifetime_source_20261017_v2"
ACCEPTED="research/reviews/H060_H061_transaction_admission_source_20261017_v1"
def need(ok,message):
 if not ok:raise AssertionError(message)
def ref(path):
 p=ROOT/path;return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def tree(path):return ast.parse((ROOT/path).read_bytes())
def node(path,name):
 matches=[n for n in tree(path).body if isinstance(n,ast.FunctionDef)and n.name==name]
 need(len(matches)==1,"one source function "+name);return copy.deepcopy(matches[0])
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def isolated(nodes,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),"<isolated-AST>","exec"),scope);return scope
def fake_ref(path):return dict(path=str(path),sha256=hashlib.sha256(str(path).encode()).hexdigest())
def rejects(call):
 try:call()
 except(ValueError,AssertionError):return
 raise AssertionError("expected rejection")
def check():
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"ROOT -B -s")
 b=json.loads((ROOT/HERE/"source_bindings.v1.json").read_text())
 current=tree(HERE+"/root_driver_v1.py");prior=tree(OLD+"/root_driver_v1.py")
 # Static inertness inspection precedes authorized real noargs checks.
 for n in current.body:
  need(isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef,ast.If,ast.Expr)),"safe default top level")
  if isinstance(n,ast.Assign):
   need(not any(isinstance(x,ast.Call)and ast.unparse(x)!="Path('/workspace-SR003.nfs2/ARG-GNN')"for x in ast.walk(n)),"no top-level project IO")
  if isinstance(n,ast.If):need(ast.unparse(n.test)=="__name__ == '__main__'","main only")
 main=next(n for n in current.body if isinstance(n,ast.FunctionDef)and n.name=="main")
 branch=next(n for n in main.body if isinstance(n,ast.If))
 need(any(isinstance(n,ast.Return)and isinstance(n.value,ast.Constant)and n.value.value==0 for n in branch.body),"inactive main early return")
 worker=tree(HERE+"/worker_entry_v1.py")
 need(any(isinstance(n,ast.ImportFrom)and n.level==1 and n.module is None and n.names[0].name=="root_driver_v1"for n in worker.body),"relative sibling worker")
 need((ROOT/HERE/"worker_entry_v1.py").read_bytes()==(ROOT/OLD/"worker_entry_v1.py").read_bytes(),"byte-identical accepted worker")
 # Reverse namespace and only new helpers/dispatch: every retained v2 node is exact.
 restored=copy.deepcopy(current)
 restored.body=[n for n in restored.body if not(isinstance(n,ast.FunctionDef)and n.name in("_make_root_mutation_memo","_memo_admission_node","_install_memo_admit"))]
 for n in restored.body:
  if isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str):n.value.value=prior.body[0].value.value
  if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=="HERE"for x in n.targets):n.value.value=OLD
  if isinstance(n,ast.Import):n.names=[x for x in n.names if x.name!="ast"]
  if isinstance(n,ast.FunctionDef)and n.name=="main":
   n.body=[x for x in n.body if not(isinstance(x,ast.If)and isinstance(x.test,ast.Attribute)and x.test.attr=="admit_ref")]
 need(same(restored,prior),"all accepted v2 factory/selector/failure/namespace/science/lifecycle retained exactly")
 need(same(node(HERE+"/root_driver_v1.py","_make_root_mutation_memo"),node(ACCEPTED+"/root_driver_v1.py","_make_root_mutation_memo")),"exact already-reviewed constructor helper")
 ns=isolated([node(HERE+"/root_driver_v1.py",name)for name in("need","_memo_admission_node")],dict(ast=ast,copy=copy))
 original=node(b["driver_ref"]["path"],"admit");derived=ns["_memo_admission_node"](original,"H058")
 reversed_admit=copy.deepcopy(derived)
 reversed_admit.body=[n for n in reversed_admit.body if not(isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=="_make_root_mutation_memo")]
 original_state=next(n for n in original.body if isinstance(n,ast.With))
 for i,n in enumerate(reversed_admit.body):
  if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute):
   if n.value.func.attr=="read_state_v1":reversed_admit.body[i]=copy.deepcopy(original_state)
   if n.value.func.attr=="reserve_v1":n.value.func=ast.Attribute(value=ast.Name(id="g",ctx=ast.Load()),attr="apply_reserve_v1",ctx=ast.Load())
 need(same(reversed_admit,original),"exact three original admit substitutions")
 need(not any(isinstance(n,ast.With)for n in ast.walk(derived)),"no nested/outer mutex")
 gi=next(i for i,n in enumerate(derived.body)if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=="g"for x in n.targets))
 need(ast.unparse(derived.body[gi+1])=="memo = _make_root_mutation_memo(g, q)","constructor immediately after guard")
 first_put=next(i for i,n in enumerate(derived.body)if any(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=="put"for x in ast.walk(n)))
 need(gi+1<first_put,"preflight before original scientific_config first put")
 indices=[next(i for i,n in enumerate(derived.body)if text in ast.unparse(n))for text in("consumer_bindings_v1","scope_spec","operation_draft_v1","memo.reserve_v1")]
 need(indices==sorted(indices),"original role/scope/authority/reserve order")
 # Actual original direct roles plus actual v2 wrapper and exact union predicates.
 fixture=json.loads((ROOT/ACCEPTED/"handwritten_role_fixture.v1.json").read_text())
 guards=list(fixture["actual_guard_role_refs_by_name"].values());selected=b["selected_child_source_ref"]
 deps=json.loads((ROOT/b["base_directory"]/"dependencies.v1.json").read_text())
 binding=json.loads((ROOT/b["scientific_binding_ref"]["path"]).read_text())
 role_scope={name:deps[name]for name in("NAMESPACE","PIDFD","OWNED","SUPERVISOR")}
 role_scope.update(ref=fake_ref,binding=lambda:binding,MANIFEST=HERE+"/source_manifest.v1.json",SOURCE=HERE+"/root_driver_v1.py",DRIVER=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py",HANDOFF=HERE+"/source_handoff.v1.json",SELECTOR=selected,CAPS=b["caps"],make_config=lambda q,caps:dict(scientific_config_ref=fake_ref("fixture/science.json")))
 original_roles=isolated([node(b["contract_ref"]["path"],"direct_roles")],role_scope)["direct_roles"]
 q={key:fake_ref("fixture/"+key+".json")for key in("source_review_ref","scientific_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref","registration_ref","registration_decisions_ref","training_context_ref")}
 q["storage_policy_selector_ref"]=fixture["actual_selector_metadata_ref"]
 contract_node=node(HERE+"/root_driver_v1.py","contract")
 wrapper=next(n for n in contract_node.body if isinstance(n,ast.FunctionDef)and n.name=="direct_roles")
 c=SimpleNamespace(DEPS=deps,GUARD_REF=selected,require=need)
 repaired=isolated([wrapper],dict(c=c,original_roles=original_roles,copy=copy,selected=lambda record:dict(source_ref=selected)))["direct_roles"]
 roles=repaired(fake_ref("fixture/request.json"),q,fake_ref("fixture/config.json"),fake_ref("fixture/closure.json"))
 union=isolated([node(selected["path"],"consumer_bindings_v1")],dict(copy=copy,require=need,_raw=lambda r:need(isinstance(r,dict)and set(r)=={"path","sha256"},"exact handwritten refs")))["consumer_bindings_v1"]
 combined=union(guards,roles)
 need(len(combined)==62 and all(r in combined for r in guards)and deps["SELECTOR"]in combined and selected in combined,"all exact62 current/original roles")
 need(b["transaction_memo"]["source_ref"]not in combined,"transitive memo only")
 roles64=combined+[fake_ref("fixture/extra_%d.json"%i)for i in range(2)]
 need(len(union(roles64))==64,"64 accepted");rejects(lambda:union(roles64+[fake_ref("fixture/extra65.json")]))
 # The original accepted nine-pair loop executes on handwritten pure paths.
 validation=next(n for n in contract_node.body if isinstance(n,ast.FunctionDef)and n.name=="validate_request")
 loop=next(n for n in validation.body if isinstance(n,ast.For))
 paths={key:"fresh/"+key for key in b["excluded_namespaces"]}
 def check_paths(values):
  isolated([loop],dict(q=values,c=SimpleNamespace(local=lambda p:Path("/memory")/p,require=need),b=b))
 check_paths(paths);negative=0
 for new_key in paths:
  for old_key,old_path in b["excluded_namespaces"].items():
   for candidate in(old_path,old_path+"/child",str(Path(old_path).parent)):
    test=dict(paths);test[new_key]=candidate;rejects(lambda test=test:check_paths(test));negative+=1
 need(negative==27,"all nine old/new pairs and relations reject")
 # Exact72h, failure acceptance and hypothesis serialization remain v2 source-exact;
 # independently run their isolated guards on handwritten values.
 interval=isolated([node(HERE+"/root_driver_v1.py","authority_interval")],dict(datetime=datetime,need=need))["authority_interval"]
 created=datetime(2026,10,17,tzinfo=timezone.utc)
 interval(dict(created_at_utc=created.isoformat(),expires_at_utc=(created+timedelta(hours=72)).isoformat()))
 for hours in(25,73):rejects(lambda hours=hours:interval(dict(created_at_utc=created.isoformat(),expires_at_utc=(created+timedelta(hours=hours)).isoformat())))
 failure_node=next(n for n in contract_node.body if isinstance(n,ast.FunctionDef)and n.name=="accept_failure")
 failure=dict(schema="ROOT_H058_unstarted_failure_acceptance_v1",root_authorizer="root_agent",accepted=True,original_input_ref=b["original_input_ref"],actual_wait_ref=b["original_actual_wait_ref"],worker_log_ref=b["original_worker_log_ref"],outer_tool_session=7922,outer_tool_terminal_consumed=True,outer_exit_code=1,direct_worker_returncode=1,owned_drain=True,scientific_startup_reached=False,optimizer_updates=0,old_launch_resume_write_authority_revoked=True,old_files_key_charge_and_expiry_unchanged=True,condition=b["condition"],actual_outer_terminal_ref=fake_ref("fixture/terminal.json"))
 failure_scope=SimpleNamespace(read=lambda record:record,require=need,raw=lambda record:None)
 accept=isolated([failure_node],dict(c=failure_scope,b=b))["accept_failure"];accept(failure)
 for key in("outer_tool_terminal_consumed","owned_drain","old_launch_resume_write_authority_revoked","old_files_key_charge_and_expiry_unchanged"):
  bad=dict(failure);bad[key]=False;rejects(lambda bad=bad:accept(bad))
 put_node=next(n for n in contract_node.body if isinstance(n,ast.FunctionDef)and n.name=="put")
 serializer=isolated([put_node],dict(Path=Path,c=SimpleNamespace(require=need,ref=fake_ref),SOURCE=HERE+"/root_driver_v1.py",original_put=lambda path,value,cap:value))["put"]
 admitted=dict(schema="H058_full_parent_CPU_admitted_input_v1",root_driver_ref=fake_ref(HERE+"/root_driver_v1.py"),original_field="kept")
 need(serializer("fixture/input.json",admitted)==dict(admitted,hypothesis_id="H058")and "hypothesis_id"not in admitted,"exact H058 serialization only")
 # Constructor closure failures before helper import/construction.
 helper=node(HERE+"/root_driver_v1.py","_make_root_mutation_memo");helper_cases=[]
 for mode in("success","missing_helper","missing_review","review_verdict"):
  events=[];manifest={"source_sha256":{r["path"]:r["sha256"]for r in b["transaction_memo"].values()}}
  review={"verdict":"SOURCE_PASS","source_only":True,"source_ref":b["transaction_memo"]["source_ref"]}
  if mode=="missing_helper":manifest["source_sha256"].pop(b["transaction_memo"]["source_ref"]["path"])
  if mode=="missing_review":manifest["source_sha256"].pop(b["transaction_memo"]["independent_review_ref"]["path"])
  if mode=="review_verdict":review["verdict"]="SOURCE_BLOCK"
  request={"source_manifest_ref":fake_ref(HERE+"/source_manifest.v1.json")}
  def construct(g,**kwargs):
   events.append("construct");need(all(Path(r["path"]).suffix in(".py",".sh")for r in kwargs["cache_refs"]),"code only");return "memo"
  def load_fixture(record,name):events.append("load");return SimpleNamespace(RootMutationMemo=construct)
  fn=isolated([helper],dict(bindings=lambda:b,read=lambda record:manifest if record==request["source_manifest_ref"]else review,raw=lambda record:None,need=need,Path=Path,load=load_fixture))["_make_root_mutation_memo"]
  if mode=="success":need(fn(object(),request)=="memo","constructor success")
  else:rejects(lambda:fn(object(),request));need(not events,"no import/construct on preflight failure")
  helper_cases.append(dict(case=mode,constructed="construct"in events))
 # Actual derived admission at first write boundary, including scientific_config first.
 class FakePath:
  def __init__(self,path):self.path=str(path);self.name=self.path.rsplit("/",1)[-1]
  def exists(self):return False
  def __truediv__(self,name):return FakePath(self.path+"/"+str(name))
  def __str__(self):return self.path
 class StopPut(Exception):pass
 traces=[]
 for failed in(True,False):
  events=[];q=dict(packet_directory="fixture/packet",output_directory="fixture/output",preparation_directory="fixture/prep",cpu_ids=[160],scientific_config={})
  def get_guard(flag):events.append("guard_false");need(flag is False,"guard false");return object()
  def factory(g,q):
   events.append("memo_preflight")
   if failed:raise ValueError("handwritten failure")
   return object()
  def put_fixture(path,value):events.append("first_put");need(str(path).endswith(".scientific_config.json"),"original first put");raise StopPut()
  c=SimpleNamespace(validate_request=lambda r:(q,{},{}))
  admit=isolated([derived],dict(c=c,need=need,local=FakePath,guard=get_guard,put=put_fixture,os=SimpleNamespace(sched_getaffinity=lambda _:set([160])),_make_root_mutation_memo=factory))["admit"]
  try:admit(fake_ref("fixture/request.json"))
  except(ValueError,StopPut):pass
  need(events==["guard_false","memo_preflight"]+([]if failed else["first_put"]),"prewrite rejection")
  traces.append(dict(memo_failed=failed,events=events))
 # Asset joins: original deps/caps read before isolated HERE rebind; binding overridden;
 # selector retains its original own HERE/profile and manifest pins all of them.
 need((ROOT/HERE/"ROOT_failure_acceptance.INACTIVE.template.v1.json").read_bytes()==(ROOT/OLD/"ROOT_failure_acceptance.INACTIVE.template.v1.json").read_bytes(),"exact failure template")
 contract_text=(ROOT/b["contract_ref"]["path"]).read_text()
 need('DEPS=json.loads((ROOT/HERE/"dependencies.v1.json").read_text())'in contract_text and 'CAPS=json.loads((ROOT/HERE/"caps.inactive.v1.json").read_text())'in contract_text,"original pre-rebind asset loads")
 need('c.binding=lambda:read(b["scientific_binding_ref"])'in(ROOT/HERE/"root_driver_v1.py").read_text(),"explicit original binding")
 selector_base=str(Path(b["selector_protocol_contract_ref"]["path"]).parent)
 asset_refs=[ref(b["base_directory"]+"/"+name)for name in("dependencies.v1.json","caps.inactive.v1.json","scientific_binding.v1.json")]
 asset_refs += [ref(selector_base+"/"+name)for name in("dependencies.v1.json","caps.inactive.v1.json","storage_profiles.v1.json")]
 need(OLD not in(ROOT/HERE/"root_driver_v1.py").read_text()and OLD.replace("/",".")not in(ROOT/HERE/"worker_entry_v1.py").read_text(),"current own self namespace")
 inactive={}
 for entry in("root_driver_v1","worker_entry_v1"):
  module=HERE.replace("/",".")+"."+entry
  code="import runpy,sys,json\nsys.argv=["+repr(module)+"]\ntry:runpy.run_module("+repr(module)+",run_name='__main__')\nexcept SystemExit as e:\n assert e.code in (None,0)\nforbidden=[n for n in sys.modules if n.startswith(('torch','numpy','arggnn','_H058_fresh72h','_fresh_ROOT_transaction_memo')) or 'relate_failed64_retention' in n]\nassert not forbidden,forbidden\nprint(json.dumps(dict(import_audit_pass=True,forbidden_imports=forbidden)))"
  completed=subprocess.run(["/usr/bin/python3","-B","-s","-c",code],cwd=ROOT,text=True,capture_output=True,timeout=20)
  need(completed.returncode==0,"inert noargs "+entry+" "+completed.stdout+completed.stderr)
  lines=[json.loads(line)for line in completed.stdout.splitlines()]
  need(lines[0]["status"]=="SOURCE_ONLY_INACTIVE"and lines[0]["guard_imported"]is False and lines[1]["import_audit_pass"],"actual inactive noargs")
  inactive[module]=lines
 return dict(schema="H058_transaction_admission_source_checks_v1",source_only=True,run_authority=False,exact_accepted_v2_factory_worker_science_lifecycle_AST=True,exact_same_accepted_constructor_helper=True,exact_three_original_admit_substitutions=True,no_outer_mutex=True,memo_before_first_scientific_config_put=True,original_union_scope_authority_reserve_order=True,handwritten_distinct_role_baseline=62,all_actual_guard_and_static_selector_roles_preserved=True,no_extra_direct_memo_role=True,role64_accept65_reject=True,all_27_namespace_relation_negatives_reject=True,original72h_failure7922_revocation_and_H058_serialization=True,memo_preflight_cases=helper_cases,admission_prewrite_boundaries=traces,exact_original_asset_refs=asset_refs,relative_worker_and_failure_template_byte_identical=True,static_inertness_before_actual_noargs=True,inactive_noargs=inactive,actual_guard_state_mutex_resource_job_scientific_Git_operations=False)
if __name__=="__main__":
 try:print(json.dumps(check(),indent=2,sort_keys=True))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)

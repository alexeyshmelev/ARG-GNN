"""Independent integration source/AST + explicitly ROOT-authorized inert entries only."""
import ast,contextlib,copy,hashlib,io,json,runpy,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OWN=ROOT/"research/reviews/admission_repairs_independent_20261017_v1";relative="research/reviews/H060_H061_transaction_admission_source_20261017_v1";base=ROOT/relative;prior=ROOT/"research/reviews/H060_H061_child_guard_role_repair_source_20261017_v3";package=relative.replace("/",".")
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return json.loads(p.read_text())
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def function(p,name):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
manifest=load(base/"source_manifest.v1.json");audit=load(base/"audit_manifest.v1.json");bindings=load(base/"source_bindings.v1.json");old_bindings=load(prior/"source_bindings.v1.json")
assert bindings["consumers"]==old_bindings["consumers"]and bindings["selected_child_source_ref"]==old_bindings["selected_child_source_ref"]
for key in["source_sha256","files"]:
 mapping=manifest.get(key,{})or audit.get(key,{})
 for p,s in mapping.items():assert ref(ROOT/p)["sha256"]==s,p
for key,record in bindings["transaction_memo"].items():
 assert set(record)=={"path","sha256"}and ref(ROOT/record["path"])==record
 assert manifest["source_sha256"][record["path"]]==record["sha256"]
memo_review=load(ROOT/bindings["transaction_memo"]["independent_review_ref"]["path"])
assert memo_review["independent"]is True and memo_review["verdict"]=="SOURCE_PASS"and memo_review["source_ref"]==bindings["transaction_memo"]["source_ref"]
current=ast.parse((base/"root_driver_v1.py").read_text());original=ast.parse((prior/"root_driver_v1.py").read_text());restored=copy.deepcopy(current)
added={"_make_root_mutation_memo","_memo_admission_node","_install_memo_admit"}
restored.body=[n for n in restored.body if not(isinstance(n,ast.FunctionDef)and n.name in added)]
for n in restored.body:
 if isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str):n.value.value=original.body[0].value.value
 if isinstance(n,ast.Import):n.names=[a for a in n.names if a.name!="ast"]
 if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="HERE"for t in n.targets):n.value=ast.Constant(str(prior.relative_to(ROOT)))
 if isinstance(n,ast.FunctionDef)and n.name=="execute_or_admit":
  expected=ast.parse("if admission: _install_memo_admit(d,consumer)").body[0]
  found=[i for i,x in enumerate(n.body)if same(x,expected)];assert len(found)==1
  n.body.pop(found[0])
assert same(restored,original),"Unexpected change outside three additions/dispatch/namespace/import"
assert(base/"worker_entry_v1.py").read_bytes()==(prior/"worker_entry_v1.py").read_bytes()
for name in["storage_profiles.v1.json","scientific_config.INACTIVE.template.v1.json"]:assert(base/name).read_bytes()==(prior/name).read_bytes()
# Execute only this pure AST-to-AST transform, never any original project module.
transform=function(base/"root_driver_v1.py","_memo_admission_node");scope=dict(ast=ast,copy=copy,need=need)
exec(compile(ast.Module(body=[copy.deepcopy(transform)],type_ignores=[]),"isolated_integration_AST_transform","exec"),scope)
deltas=[];constructor_traces=[]
class FakePath:
 def __init__(self,value):self.value=str(value);self.name=self.value.rsplit("/",1)[-1]
 def exists(self):return False
 def __truediv__(self,name):return FakePath(self.value+"/"+str(name))
 def __str__(self):return self.value
for consumer,spec in bindings["consumers"].items():
 old=function(ROOT/spec["driver_ref"]["path"],"admit");derived=scope["_memo_admission_node"](old,consumer);expected=copy.deepcopy(old)
 gi=next(i for i,n in enumerate(expected.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="g"for t in n.targets))
 wi=[i for i,n in enumerate(expected.body)if isinstance(n,ast.With)];assert len(wi)==1
 expected.body[wi[0]]=ast.parse("""state = memo.read_state_v1(expected_source_sha256=implementation["sha256"], expected_launcher_sha256=launcher["sha256"])""").body[0]
 ri=next(i for i,n in enumerate(expected.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="receipt"for t in n.targets))
 assert ast.unparse(expected.body[ri].value.func)=="g.apply_reserve_v1"
 expected.body[ri].value.func=ast.Attribute(value=ast.Name(id="memo",ctx=ast.Load()),attr="reserve_v1",ctx=ast.Load())
 expected.body.insert(gi+1,ast.parse("memo = _make_root_mutation_memo(g,q)").body[0])
 assert same(derived,expected),consumer
 assert not any(isinstance(n,ast.With)for n in ast.walk(derived))
 text=ast.unparse(derived);positions=[text.index(s)for s in["_make_root_mutation_memo", "put(","memo.read_state_v1","consumer_bindings_v1","scope_spec","operation_draft_v1","memo.reserve_v1"]]
 assert positions==sorted(positions)
 deltas.append(dict(consumer=consumer,exact_independently_reconstructed_three_substitutions=True,no_caller_mutex=True,memo_before_first_put=True,union_scope_before_authority=True,executor_worker_math_changes=False))
 # The actual derived admission stops at a rejected constructor, before staging.
 events=[];q=dict(packet_directory="SOURCE_FIXTURE/packet",output_directory="SOURCE_FIXTURE/output",preparation_directory="SOURCE_FIXTURE/preparation",cpu_ids=[223],stage="freeze")
 def get_guard(flag):events.append("guard_false");assert flag is False;return object()
 def factory(g,q):events.append("memo_constructor_rejected");raise ValueError("SOURCE_FIXTURE intentional rejection")
 def put(*args,**kwargs):events.append("unexpected_put");raise AssertionError("Write before constructor preflight")
 c=SimpleNamespace(validate_request=lambda r:(q,{},{}),local=FakePath,require=need,guard=get_guard,put=put)
 fake=dict(c=c,need=need,local=FakePath,guard=get_guard,put=put,os=SimpleNamespace(sched_getaffinity=lambda _:set([223])),_make_root_mutation_memo=factory)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(derived)],type_ignores=[])),"derived_admission_on_handwritten_memory","exec"),fake)
 try:fake["admit"]({"source_fixture":True})
 except ValueError as e:assert str(e)=="SOURCE_FIXTURE intentional rejection"
 assert events==["guard_false","memo_constructor_rejected"],events
 constructor_traces.append(dict(consumer=consumer,events=events,no_staging_or_authority_write=True))
# Both requested public entrypoints are statically inert before actual noarg runs.
for p in[ROOT/"research/__init__.py",ROOT/"research/reviews/__init__.py",base/"__init__.py"]:assert not p.exists()
for p in[base/"root_driver_v1.py",base/"worker_entry_v1.py"]:
 t=ast.parse(p.read_text());main=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=="main");branch=next(n for n in main.body if isinstance(n,ast.If))
 assert len(branch.body)==2 and isinstance(branch.body[-1],ast.Return)and ast.literal_eval(branch.body[-1].value)==0
worker_tree=ast.parse((base/"worker_entry_v1.py").read_text());assert any(isinstance(n,ast.ImportFrom)and n.level==1 and n.module is None and any(a.name=="root_driver_v1"and a.asname=="d"for a in n.names)for n in worker_tree.body)
sys.path.insert(0,str(ROOT));before=set(sys.modules);entry_rows=[];argv=sys.argv[:]
try:
 for leaf in["root_driver_v1","worker_entry_v1"]:
  name=package+"."+leaf;sys.argv=[name];out=io.StringIO();code=None
  with contextlib.redirect_stdout(out):
   try:runpy.run_module(name,run_name="__main__",alter_sys=True)
   except SystemExit as e:code=e.code
  value=json.loads(out.getvalue());assert code==0 and value["status"]=="SOURCE_ONLY_INACTIVE"and value["guard_imported"]is False
  entry_rows.append(dict(module=name,exit_code=code,status=value))
finally:sys.argv=argv
loaded=set(sys.modules)-before;project=[m for m in loaded if m=="research"or m.startswith("research.")or m=="scripts"or m.startswith("scripts.")or m=="arggnn"or m.startswith("arggnn.")]
assert set(project)<=set(["research","research.reviews",package,package+".root_driver_v1"]),project
assert not any(m==p or m.startswith(p+".")for m in loaded for p in["torch","numpy","torch_geometric","arggnn","scripts"])
module=sys.modules[package+".root_driver_v1"];assert module.HERE==relative and module.SOURCE==relative+"/root_driver_v1.py"and module.ENTRY==relative+"/worker_entry_v1.py"
result=dict(schema="independent_whole_H060_H061_transaction_integration_checks_v1",source_ref=ref(base/"root_driver_v1.py"),source_manifest_ref=ref(base/"source_manifest.v1.json"),source_handoff_ref=ref(base/"source_handoff.v1.json"),all_manifest_files_authenticated=len(manifest["source_sha256"]),audit_files_authenticated=len(audit["files"]),component_consumer_science_and_caps_bindings_exact=True,memo_source_and_independent_review_transitive_pins_exact=True,whole_driver_AST_exact_v3_plus_three_added_helpers_dispatch_and_namespace=True,derived_admission_AST_deltas=deltas,constructor_rejection_before_staging_cases=constructor_traces,original_scientific_template_and_storage_profiles_exact=True,worker_exact_v3_package_relative=True,actual_inactive_entry_checks=entry_rows,actual_loaded_project_modules=sorted(project),guard_original_consumer_numerical_modules_imported=False,actual_guard_admission_mutex_process_inspection_resource_payload_numerical_Git_operations=False,blockers=[])
p=OWN/"whole_integration_checks.v1.json";assert not p.exists();p.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n");print(json.dumps(dict(result_ref=ref(p),deltas=deltas,inactive_entry_checks=entry_rows,authenticated_files=len(manifest["source_sha256"]))))

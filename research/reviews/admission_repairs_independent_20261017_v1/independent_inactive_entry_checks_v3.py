"""Narrow ROOT-authorized inactive entry check after source AST inertness audit."""
import ast,contextlib,hashlib,io,json,runpy,sys
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OWN=ROOT/"research/reviews/admission_repairs_independent_20261017_v1";relative="research/reviews/H060_H061_child_guard_role_repair_source_20261017_v3";base=ROOT/relative;old=ROOT/relative.replace("_v3","_v1");package=relative.replace("/",".")
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for p in[ROOT/"research/__init__.py",ROOT/"research/reviews/__init__.py",base/"__init__.py"]:assert not p.exists()
old_driver=ast.parse((old/"root_driver_v1.py").read_text());new_driver=ast.parse((base/"root_driver_v1.py").read_text())
for n in old_driver.body:
 if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="HERE"for t in n.targets):n.value=ast.Constant(relative)
assert ast.dump(old_driver)==ast.dump(new_driver)
old_worker=ast.parse((old/"worker_entry_v1.py").read_text());new_worker=ast.parse((base/"worker_entry_v1.py").read_text())
for n in old_worker.body:
 if isinstance(n,ast.ImportFrom)and any(a.asname=="d"for a in n.names):n.module=None;n.level=1
assert ast.dump(old_worker)==ast.dump(new_worker)
for name in["source_bindings.v1.json","storage_profiles.v1.json"]:assert(old/name).read_bytes()==(base/name).read_bytes()
config=ROOT/"research/reviews/H061_postselection_execution_source_20261017_v1/scientific_config.INACTIVE.template.v1.json"
assert config.read_bytes()==(base/config.name).read_bytes()
# Static no-arg branch has a literal return before every admission/worker call.
for p in[base/"root_driver_v1.py",base/"worker_entry_v1.py"]:
 t=ast.parse(p.read_text());main=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=="main")
 branch=next(n for n in main.body if isinstance(n,ast.If))
 assert isinstance(branch.body[-1],ast.Return)and ast.literal_eval(branch.body[-1].value)==0
 assert len(branch.body)==2 and isinstance(branch.body[0],ast.Expr)
sys.path.insert(0,str(ROOT))
before=set(sys.modules);rows=[];original_argv=sys.argv[:]
try:
 for leaf in["root_driver_v1","worker_entry_v1"]:
  name=package+"."+leaf;sys.argv=[name];out=io.StringIO();exit_code=None
  with contextlib.redirect_stdout(out):
   try:runpy.run_module(name,run_name="__main__",alter_sys=True)
   except SystemExit as error:exit_code=error.code
  status=json.loads(out.getvalue());assert exit_code==0 and status["status"]=="SOURCE_ONLY_INACTIVE"and status["guard_imported"]is False
  rows.append(dict(module=name,exit_code=exit_code,status=status))
finally:sys.argv=original_argv
loaded=set(sys.modules)-before
project=[m for m in loaded if m=="research"or m.startswith("research.")or m=="scripts"or m.startswith("scripts.")or m=="arggnn"or m.startswith("arggnn.")]
assert set(project)<=set(["research","research.reviews",package,package+".root_driver_v1"]),project
assert not any(m==x or m.startswith(x+".")for m in loaded for x in["torch","numpy","torch_geometric","arggnn","scripts"])
module=sys.modules[package+".root_driver_v1"]
assert module.HERE==relative and module.SOURCE==relative+"/root_driver_v1.py"and module.ENTRY==relative+"/worker_entry_v1.py"
result=dict(schema="independent_ROOT_authorized_inactive_entry_namespace_check_v3",source_ref=ref(base/"root_driver_v1.py"),worker_ref=ref(base/"worker_entry_v1.py"),source_manifest_ref=ref(base/"source_manifest.v1.json"),driver_AST_unchanged_except_successor_HERE=True,worker_AST_unchanged_except_package_relative_import=True,exact_original_H061_template_ref=ref(config),successor_H061_template_ref=ref(base/config.name),bindings_and_profiles_exact=True,inactive_entry_checks=rows,actual_loaded_project_modules=sorted(project),same_successor_driver_worker_identity=True,no_guards_original_consumers_or_numerical_modules_imported=True,no_operation_refs_supplied=True,actual_admission_payload_or_resource_operations=False,blockers=[])
p=OWN/"inactive_entry_and_drift.v3.json";assert not p.exists();p.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n");print(json.dumps(dict(result_ref=ref(p),checks=rows,loaded_project_modules=sorted(project))))

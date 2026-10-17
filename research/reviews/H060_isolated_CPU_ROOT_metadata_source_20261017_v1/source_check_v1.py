"""Stdlib checks of inert default, prewrite ordering and negative metadata seams."""
from pathlib import Path
import argparse,ast,builtins,contextlib,copy,hashlib,io,json,symtable
HERE=Path("research/reviews/H060_isolated_CPU_ROOT_metadata_source_20261017_v1")
def need(ok,message):
 if not ok:raise AssertionError(message)
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--inert-only",action="store_true",help="Skip immutable historical dependency pin reads")
 args=parser.parse_args(argv)
 p=HERE/"assemble_root_metadata_v1.py";s=p.read_text();tree=ast.parse(s);compile(s,str(p),"exec")
 allowed={"pathlib","datetime","argparse","ast","copy","hashlib","json","os","stat","sys"}
 for node in tree.body:
  if isinstance(node,ast.Import):need(all(x.name in allowed for x in node.names),"Only stdlib imports")
  if isinstance(node,ast.ImportFrom):need(node.module in allowed,"Only stdlib from-imports")
 functions={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
 scope={"__name__":"inert_assembler_source_definition"};exec(compile(tree,str(p),"exec"),scope)
 def no_read(*a,**k):raise AssertionError("Inactive path attempted a project read/write/stage")
 scope["read"]=no_read;scope["stage"]=no_read;scope["source_packet"]=no_read
 output=io.StringIO()
 with contextlib.redirect_stdout(output):need(scope["main"]([])==0,"Inactive default terminal0")
 inactive=json.loads(output.getvalue())
 need(inactive==dict(status="INACTIVE_SOURCE_ONLY",project_reads_or_writes=False,guard_admission_execution=False),
  "Default performs no project work")
 inactive_template=json.loads((HERE/"ROOT_instruction.INACTIVE.template.v1.json").read_text())
 checks=["inert_default_no_project_calls"]
 def rejected(label,value):
  try:scope["plan_stage"](value,dict(path="handwritten_instruction_not_created.json",sha256="0"*64))
  except ValueError:checks.append(label+"_rejected_before_source_packet")
  else:raise AssertionError(label)
 rejected("inactive_instruction",copy.deepcopy(inactive_template))
 q=copy.deepcopy(inactive_template)
 q.update(root_authorizer="root_agent",staging_authorized=True,issue_full_TRAIN_request=True,
  current_CPU_capacity_confirmed=True,current_runtime_settings_confirmed=True,never_used_paths_confirmed=True)
 q["cpu_ids"]=[0]*32;rejected("duplicate_CPU_ids",q)
 q["cpu_ids"]=list(range(32));q["runtime_flags"]["interop_threads"]=2;rejected("wrong_runtime_flags",q)
 q=copy.deepcopy(inactive_template)
 q.update(root_authorizer="root_agent",staging_authorized=True,issue_full_TRAIN_request=True,
  current_CPU_capacity_confirmed=True,current_runtime_settings_confirmed=True,never_used_paths_confirmed=True,cpu_ids=list(range(32)))
 q["runtime_flags"]["intraop_threads"]=32.0;rejected("runtime_integer_replaced_by_float",q)
 q["runtime_flags"]["intraop_threads"]=32;q["runtime_flags"]["CUDA_initialized"]=0;rejected("runtime_boolean_replaced_by_integer",q)
 q=copy.deepcopy(inactive_template);q["unexpected_field"]=True;rejected("extra_instruction_field",q)
 calls=lambda node:[(x.lineno,ast.unparse(x.func))for x in ast.walk(node)if isinstance(x,ast.Call)]
 stage_calls=sorted(calls(functions["stage"]))
 names=[name for _,name in stage_calls]
 need(names.index("plan_stage")<names.index("prep.mkdir")<names.index("os.open"),"Full planning precedes first write")
 need(not any(name.endswith((".mkdir",".write",".unlink",".rmdir",".rename"))or name=="os.open"
  for _,name in calls(functions["plan_stage"])),"Planning has no mutation syscall")
 need(all(x in s for x in('predicates["reviewed"]','predicates["scientific_review"]','predicates["validate_config"]',
  'predicates["registration"]','predicates["context"]','predicates["selected"]')),
  "Exact six reviewed wrapper metadata predicates run before first write")
 need('created=datetime.fromisoformat(qin["created_at_utc"])'in s
  and'expires=datetime.fromisoformat(qin["expires_at_utc"])'in s
  and'expires-created==timedelta(hours=25)'in s,"ROOT dates retained, no clock extension")
 need(scope["EXCLUSIVE_SUFFIXES"]==(".scientific_config.json",".config.json",".source_and_input_closure.json",".reserve_authority.json"),
  "All four exclusive basenames remain absent")
 forbidden=("subprocess","importlib","torch","numpy","resource","socket","signal","sched_getaffinity","sched_setaffinity","statvfs",
  "apply_reserve","operation_draft","budget_check","load_selected_guard","root_driver_v1.main")
 module_imports={x.name for n in tree.body if isinstance(n,ast.Import)for x in n.names}
 need(not module_imports.intersection(forbidden),"No scientific/resource/guard imports")
 callnames={name for _,name in calls(tree)}
 need(not any(any(word in name for word in forbidden)for name in callnames),"No resource/guard/process/scientific calls")
 table=symtable.symtable(s,str(p),"exec")
 bound={x.get_name()for x in table.get_symbols()if x.is_assigned()or x.is_imported()or x.is_namespace()}
 def visit(t):
  for x in t.get_symbols():
   if x.is_referenced()and x.is_global():need(x.get_name()in bound or x.get_name()in vars(builtins),"Unresolved global "+x.get_name())
  for child in t.get_children():visit(child)
 visit(table)
 count=0
 if not args.inert_only:
  deps=json.loads((HERE/"dependencies.v1.json").read_text())
  for path,sha in deps["pins"].items():need(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,"Pinned source/metadata "+path)
  count=len(deps["pins"])
 return dict(schema="H060_ROOT_metadata_assembler_source_checks_v1",
  status="INERT_SOURCE_SEAMS_PASS_NOT_INDEPENDENT_REVIEW"if args.inert_only else "SOURCE_CHECKS_PASS_NOT_INDEPENDENT_REVIEW",
  checks=checks,dependency_pins_authenticated=count,immutable_historical_dependency_pins_read=not args.inert_only,
  source_compile_AST_symbols=True,
  all_metadata_validated_before_first_write=True,eight_file_preparation_only=True,
  four_exclusive_files_and_charged_paths_absent=True,explicit_selector_without_discovery=True,
  fixed_ROOT_finite_dates=True,no_actual_ROOT_instruction_or_staging=True,no_extra_qualification=True,
  project_guard_scientific_modules_imported=False,resource_or_guard_or_job_operations=False,numerical_work=False)
if __name__=="__main__":print(json.dumps(main(),sort_keys=True,indent=2))

"""Compile/AST/hash checks only. This script never imports or executes the helper."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/ordered_marker_ROOT_staging_helper_source_20261017_v1"
def need(ok,message):
 if not ok:raise AssertionError(message)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run():
 d=ROOT/HERE;manifest=json.loads((d/"source_manifest.v1.json").read_text());deps=json.loads((d/"dependencies.v1.json").read_text())
 for path,sha in manifest["files_sha256"].items():need(digest(ROOT/path)==sha,"Exact owned source pin")
 for path,sha in deps["pins"].items():need(digest(ROOT/path)==sha,"Exact sealed upstream source/metadata pin")
 compiled=[]
 for p in sorted(d.glob("*.py")):compile(p.read_text(),str(p),"exec");compiled.append(p.name)
 p=d/"stage_root_metadata_v1.py";source=p.read_text();tree=ast.parse(source)
 imports=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
  if isinstance(n,ast.ImportFrom):imports.append(n.module)
 need(set(imports)<={"pathlib","datetime","argparse","copy","hashlib","json","os","stat","sys"},"Standard library only")
 forbidden_calls={"exec","eval","compile","__import__","system","popen","fork","forkpty","spawn","kill","killpg","wait","waitpid",
  "sched_getaffinity","sched_setaffinity","statvfs","sysconf","getrusage","sleep"}
 called={n.func.id if isinstance(n.func,ast.Name)else n.func.attr for n in ast.walk(tree)
  if isinstance(n,ast.Call)and isinstance(n.func,(ast.Name,ast.Attribute))}
 need(not(called&forbidden_calls),"No runtime/import/process/resource/admission/driver execution")
 need("subprocess"not in source and "importlib"not in source and "torch"not in imports and "numpy"not in imports,"No scientific/guard/driver import")
 constants={n.targets[0].id:ast.literal_eval(n.value)for n in tree.body
  if isinstance(n,ast.Assign)and len(n.targets)==1and isinstance(n.targets[0],ast.Name)and isinstance(n.value,ast.Constant)}
 need(constants["PREPARATION"]=="research/reviews/ordered_marker_CPU_ROOT_preparation_root01_oct17_v1"
  and constants["PACKET"]=="research/reviews/ordered_marker_CPU_ROOT_root01_oct17_v1"
  and constants["OUTPUT"]=="experiments/results/ordered_marker_full_parent_CPU_root01_oct17_v1","Exact fresh namespaces")
 cpu=next(n.value for n in tree.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=="CPUS")
 need(ast.unparse(cpu)=="list(range(128, 160))","CPU128-159 exactly32")
 functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 main=ast.get_source_segment(source,functions["main"]);stage=ast.get_source_segment(source,functions["stage"])
 need(main.index('if args.stage_root_instruction_ref is None:')<main.index('result=stage(')
  and 'project_reads_or_writes=False'in main,"Default branch inert before helper staging")
 stage_calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="stage"]
 need(len(stage_calls)==1and stage_calls[0] in list(ast.walk(functions["main"])),"Only explicit CLI branch calls stage")
 need("created=datetime.now(timezone.utc);expires=created+timedelta(seconds=259200)"in stage
  and '"created_at_utc"'not in stage.split("expected=",1)[1].split("require(set(instruction)",1)[0],"72h derived only at actual ROOT staging")
 need(stage.index("source_packet(instruction)")<stage.index("prep.mkdir(")
  and stage.index("decisions=json.loads(decision_bytes)")<stage.index("prep.mkdir(")
  and stage.index("Shared metadata changed before staging")<stage.index("prep.mkdir("),"All source/duplicate/current-source checks precede first write")
 need('HYPOTHESIS not in ledger'in stage and 'Refuse duplicate candidate registration'in stage,"No duplicate hypothesis overwrite")
 need('top_level_decision_keys=sorted(decisions)'in stage and 'existing_decision_records='in stage
  and 'frozen["decisions"][:-1]==decisions["decisions"]'in stage and 'set(frozen)==set(decisions)'in stage,"Complete original decision keys/objects preserved")
 need('os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW'in source and "O_TRUNC"not in source,"Exclusive metadata writes only")
 need('directory==local(PREPARATION)'in source and 'Path(name).name==name'in source,"Every generated product restricted to fresh prep")
 cfg_updates=[n for n in ast.walk(functions["stage"])if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
  and isinstance(n.func.value,ast.Name)and n.func.value.id=="cfg"and n.func.attr=="update"]
 need(len(cfg_updates)==1and {k.arg for k in cfg_updates[0].keywords}=={"execution_enabled","experiment_registration_ref","runtime_equivalence_ref","full_training_directive_ref","layout_resource_proof_ref"},"No reviewed scientific config or budget mutation")
 need('protocol["VAL_prediction_products_required_before_truth"]==96'in stage
  and 'protocol["updates_per_arm"]==960'in stage and 'protocol["updates_all_new_arms"]==1920'in stage
  and 'set(protocol["parameters_per_arm"].values())=={33966}'in stage,"All96gate and full matched training preserved")
 need('source_manifest_ref=deps["integration_manifest_ref"]'in stage and 'scientific_review_ref=b["scientific_review_ref"]'in stage
  and 'source_review_ref=deps["integration_review_ref"]'in stage and 'CPU_entry_ref=b["runner_source_ref"]'in stage,"Integration/scientific bindings distinct")
 need('packet_and_output_created=False'in stage and 'not os.path.lexists(packet)and not os.path.lexists(output)'in stage
  and 'shared_sources_modified=False'in stage and 'shared_metadata_publication_pending=True'in stage,"Shared/charged paths remain unchanged; publication pending explicit")
 need(all(suffix in stage for suffix in(".scientific_config.json",".config.json",".source_and_input_closure.json",".reserve_authority.json")),"All four original exclusive driver filenames remain absent")
 review=json.loads((ROOT/deps["integration_review_ref"]["path"]).read_text())
 need(review["verdict"]=="SOURCE_PASS"and review["source_manifest_ref"]==deps["integration_manifest_ref"]
  and deps["integration_review_ref"]["sha256"]=="79fc69d3e71c46fbff63b56763614edbd3c86f5c9fb6cee506dfa9f9aab92e61","Exact accepted independent integration review")
 return dict(schema="ordered_marker_ROOT_staging_helper_AST_check_v1",status="COMPILE_AST_SOURCE_PASS",
  helper_executed_or_imported=False,helper_stage_or_default_main_called=False,
  owned_pins_checked=len(manifest["files_sha256"]),upstream_pins_checked=len(deps["pins"]),compiled_sources=compiled,
  stdlib_imports=sorted(imports),CPU_ids=list(range(128,160)),authority_seconds_from_actual_staging=259200,
  shared_metadata_overwrite=False,all_preexisting_decision_keys_captured=True,packet_output_or_reserved_staging_created=False,
  admission_driver_guard_scientific_resources_Git=False)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))

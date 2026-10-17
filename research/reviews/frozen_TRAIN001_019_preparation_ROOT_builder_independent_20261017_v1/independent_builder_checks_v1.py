"""Narrow exact build-only helper source/RAM review. No builder output or operation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,argparse,contextlib,copy,hashlib,io,json,runpy,sys
REL="research/reviews/frozen_TRAIN001_019_preparation_ROOT_builder_source_20261017_v1"
PIPE="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2"
checks=[]
def check(x,label):
 if not x:raise AssertionError(label)
 checks.append(label)
def ref(path):return dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
check(ref(REL+"/HANDOFF.v1.json")["sha256"]=="fe380dc074ffd3e289ae3d86f21b6de86af05d84e1dc2b19c2102aff4dbc73f6","Exact sealed builder handoff")
m=json.loads(Path(REL+"/source_manifest.v1.json").read_text())
check(ref(REL+"/source_manifest.v1.json")["sha256"]=="f7f62d88c3be869a9128fc8ce9f41051a658f102f30cfbf25c496d5b44c47df7","Exact sealed builder manifest")
source=REL+"/build_ROOT_authority_v1.py"
check(ref(source)["sha256"]=="d4e74536bd81750d9404e6c67ac3a0e8e49930647108f34aa7eb7dada94c7a1f","Exact sealed builder source")
for key in ("ROOT_parameter_template_ref","ROOT_recipe_ref","immutable_pipeline_recipe_correction_ref","pipeline_manifest_ref","pipeline_worker_ref","author_checks_ref","author_result_ref"):
 check(ref(m[key]["path"])==m[key],"Exact referenced source/metadata "+key)
with contextlib.redirect_stdout(io.StringIO()):
 h=runpy.run_path(REL+"/author_RAM_checks_v1.py",run_name="__independent_builder_fixtures__")
ram=h["ram"];ns=h["ns"];parameter=h["param"];original=copy.deepcopy(h["p"]);source_tree=ast.parse(Path(source).read_text())
# Challenge uses the actual sealed independent review, with synthetic future reuse/capacity documents only.
reviewref=ref("research/reviews/frozen_TRAIN001_019_preparation_pipeline_independent_20261017_v2/review.preparation_pipeline.v2.json")
original["pipeline_source_review_ref"]=reviewref
ram[parameter["path"]]=copy.deepcopy(original)
reads=[];oldread=ns["read"]
def read(record):
 check(not record["path"].endswith((".npz",".npy",".pt",".trees")),"No scientific payload boundary: "+record["path"])
 reads.append(record["path"]);return oldread(record)
ns["read"]=read
values=ns["build_values"](parameter);params,directory,capsule,authority=values
check(authority["source_review_ref"]==reviewref and authority["source_manifest_ref"]==m["pipeline_manifest_ref"],"Actual independent three-source pass is carried into future authority")
check(authority["CPU_lanes"]==original["CPU_lanes"]and authority["actual_initial_capacity"]==original["actual_initial_capacity"],"ROOT supplied actual lane/capacity records preserved")
check(authority["TRAIN001_inference_predecessor_capsule_ref"]is None and capsule["refs"]==h["cap"]["refs"],"All7predecessor roles preserved, future capsule ref not invented")
check(authority["expires_at_utc"]==original["pipeline_expires_at_utc"],"Explicit finite expiry preserved within actual existing simulation authority")
def reject(label,mutate,message):
 before=copy.deepcopy(ram);p=copy.deepcopy(original);ram[parameter["path"]]=p
 try:
  mutate(p)
  try:ns["build_values"](parameter)
  except ValueError as error:check(message in str(error),"Reject "+label+" at exact boundary")
  else:raise AssertionError("Invalid builder metadata accepted: "+label)
 finally:ram.clear();ram.update(before)
now=datetime.now(timezone.utc)
reject("Future build directive",lambda p:p.update(created_at_utc=(now+timedelta(seconds=60)).isoformat()),"Fresh finite explicit ROOT")
reject("Future capacity observation",lambda p:p["actual_initial_capacity"].update(created_at_utc=(now+timedelta(seconds=60)).isoformat()),"ROOT supplied actual fresh32MiB capacity")
reject("Estimated initial capacity",lambda p:p["actual_initial_capacity"].update(source_only_estimate=True),"ROOT supplied actual fresh32MiB capacity")
reject("Boolean CPU ID",lambda p:p["CPU_lanes"][0].__setitem__(0,False),"Explicit two distinct8CPU lanes")
reject("Repeated CPU within lane",lambda p:p["CPU_lanes"][0].__setitem__(7,0),"Explicit two distinct8CPU lanes")
reject("Expiry beyond simulation authority",lambda p:p.update(pipeline_expires_at_utc=(datetime.fromisoformat(h["sim"]["expires_at_utc"])+timedelta(seconds=1)).isoformat()),"Explicit finite original work/close window")
reject("Wrong TRAIN000 original parent",lambda p:ram[h["nativeref"]["path"]].update(parent_id="american_c3_train_001"),"Original full30Mb TRAIN000")
reject("TRAIN000 observation substitution",lambda p:ram[h["nativeref"]["path"]].update(observation_ref=h["rr"]("ram/other_observation")),"Original full30Mb TRAIN000")
reject("Original001 source substitution",lambda p:ram[p["TRAIN001_inference_input_ref"]["path"]].update(main_source_ref=h["rr"]("ram/other_source")),"Actual original completed TRAIN001 inference lineage")
reject("Original simulation completion substitution",lambda p:ram[p["TRAIN000_simulation_completion_ref"]["path"]].update(status="completed_current_C3_inference"),"Actual original simulator completion metadata")
def missing_source_review(p):
 rr=h["rr"]("ram/independent/incomplete_review")
 review=json.loads(Path(reviewref["path"]).read_text());review["reviewed_sources"]=review["reviewed_sources"][:-1]
 ram[rr["path"]]=review;p["pipeline_source_review_ref"]=rr
reject("Review omits one executed pipeline source",missing_source_review,"Exact complete independent pipeline-v2 source pass")
def wrong_receipt_type(p):
 close=ram[p["TRAIN000_native_close_accepted_ref"]["path"]]
 ram[close["receipt_ref"]["path"]]["key"]=123
reject("Non-string close receipt identity",wrong_receipt_type,"ROOT genuine successful compact close acceptance")
# Exact inactive entrypoint is callable without the build function being present.
main=next(n for n in source_tree.body if isinstance(n,ast.FunctionDef)and n.name=="main")
env=dict(argparse=argparse,json=json,__doc__="RAM")
exec(compile(ast.Module(body=[main],type_ignores=[]),source+"#inert-default","exec"),env)
with contextlib.redirect_stdout(io.StringIO())as out:code=env["main"]([])
check(code==0 and json.loads(out.getvalue())["status"]=="SOURCE_ONLY_INACTIVE","Exact default entrypoint requires no build/output/import")
build=ast.unparse(next(n for n in source_tree.body if isinstance(n,ast.FunctionDef)and n.name=="build"))
check("w.predecessor_capsule" in build and "w.validate_pipeline_authority" in build and "directory.mkdir()"in build,"Actual build validates capsule/authority and reserves one-shot metadata namespace")
check(not any(term in Path(source).read_text()for term in ("apply_reserve","apply_close","accounting_transaction(","os.kill","torch","subprocess","measure_scopes","nvidia-smi")),"No accounting/probe/science execution API in builder")
check(not any(str(getattr(mod,"__file__","")).endswith("census_observation_retention_v1.py")for mod in list(sys.modules.values())),"No actual guard module imported")
# Filter repeated boundary bookkeeping from independent semantic counts.
semantic=[v for v in checks if not v.startswith("No scientific payload boundary:")]
print(json.dumps(dict(schema="TreeLoom_preparation_ROOT_builder_independent_checks_v1",passed=True,independent_checks=len(semantic),checks=semantic,reproduced_author_checks=h["result"]["check_count"],
 read_boundaries_checked=len(reads),source_ref=ref(source),source_manifest_ref=ref(REL+"/source_manifest.v1.json"),actual_builder_output=False,actual_guard_import_or_operations_probes=False,scientific_payload_reads=False),sort_keys=True,indent=2))

"""Pure build_values source/RAM tests. No builder output, guard or operation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,contextlib,copy,hashlib,io,json,runpy
REL="research/reviews/frozen_TRAIN001_019_preparation_ROOT_builder_source_20261017_v1"
PIPE="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2"
with contextlib.redirect_stdout(io.StringIO()):
 h=runpy.run_path(PIPE+"/author_RAM_checks_v1.py",run_name="__builder_fixture__")
ram=h["ram"];rr=h["rr"];cr,cap=h["fixture"]("native_paths","american_c3_train_001")
manifest=json.loads(Path(PIPE+"/source_manifest.v1.json").read_text())
controlref=dict(path="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/ROOT_batch_admission_root01_oct17/batch_input.json",sha256="dfd6ced48bf7ccfe8ebf9d58e3ffb2ab5193e8a586882d873e3bd4eb9279c092")
control=json.loads(Path(controlref["path"]).read_text());sim=json.loads(Path(control["authority_ref"]["path"]).read_text())
def aref(path):
 if str(path).startswith("ram/"):return rr(path)
 p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 if r["path"].startswith("PRIVATE_TRUTH"):raise AssertionError("Truth payload boundary")
 return copy.deepcopy(ram[r["path"]])if r["path"]in ram else json.loads(Path(r["path"]).read_text())
checks=[]
def need(ok,msg):
 if not ok:raise ValueError(msg)
def check(ok,msg):need(ok,msg);checks.append(msg)
def reject(fn,msg):
 try:fn()
 except(ValueError,KeyError,AssertionError,TypeError):checks.append(msg);return
 raise AssertionError("Invalid builder accepted: "+msg)
tree=ast.parse(Path(REL+"/build_ROOT_authority_v1.py").read_text())
ns=dict(read=read,ref=aref,need=need,datetime=datetime,timezone=timezone,timedelta=timedelta,ROOT=Path("/workspace-SR003.nfs2/ARG-GNN"),
 REL=REL,PIPE=PIPE,CONTROL=controlref,MANIFEST_REF=aref(PIPE+"/source_manifest.v1.json"),
 local=lambda p:Path("/workspace-SR003.nfs2/ARG-GNN")/p)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in["compact_close","build_values"]]
exec(compile(ast.Module(body=nodes,type_ignores=[]),REL+"/build_ROOT_authority_v1.py#RAM","exec"),ns)
review=rr("ram/builder/review");ram[review["path"]]=dict(independent=True,verdict="SOURCE_PASS",blockers=[],
 source_manifest_ref=ns["MANIFEST_REF"],reviewed_sources=[r for r in manifest["refs"]if r["path"].startswith(PIPE+"/")])
obsref=rr("ram/TRAIN000/obs");nativeref=rr("ram/TRAIN000/native");focalref=rr("ram/TRAIN000/focals")
prepref=rr("ram/TRAIN000/prep");iclose=rr("ram/TRAIN000/infer_close");nclose=rr("ram/TRAIN000/native_close")
ram[obsref["path"]]=dict(parent_id="american_c3_train_000",sequence_length_bp=30000000,
 genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG")
ram[nativeref["path"]]=dict(parent_id="american_c3_train_000",status="completed_native_C3_path_preparation",
 observation_ref=obsref,focals_ref=focalref,query_focal_count=10000,query_ancestry_truth_opened=False)
ram[prepref["path"]]=dict(parent_id="american_c3_train_000",observation_ref=obsref,native_preparation_ref=nativeref,
 focal_registry_ref=focalref,focal_identity_sha256="a"*64,observation_only=True,all_native_trees_nodes_sites_samples_retained=True)
for r in[iclose,nclose]:
 receipt=rr(r["path"]+"_receipt");ram[r["path"]]=dict(parent_id="american_c3_train_000",actual_direct_OS_wait0=True,
  all_owned_processes_and_groups_drained=True,receipt_ref=receipt);ram[receipt["path"]]=dict(key=r["path"])
sim000=rr("ram/TRAIN000/sim");sim001=rr("ram/TRAIN001/sim")
for parent,r in[("000",sim000),("001",sim001)]:
 ram[r["path"]]=dict(parent_id="american_c3_train_"+parent,status="completed_C3_simulation_reconstruction",
  private_truth_ref=rr("PRIVATE_TRUTH/"+parent))
# Derived builder paths bind the original future001predecessor RAM fixture, not synthetic payloads.
now=datetime.now(timezone.utc);expires=min(now+timedelta(hours=40),datetime.fromisoformat(sim["expires_at_utc"]))
param=rr("ram/builder/parameters")
p=dict(schema="TreeLoom_preparation_ROOT_explicit_build_parameters_v1",active=True,root_authorizer="root_agent",
 created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(seconds=600)).isoformat(),build_only=True,
 admission_or_execution_authorized=False,source_only_estimate=False,simulation_batch_control_input_ref=controlref,
 pipeline_source_review_ref=review,CPU_lanes=[list(range(8)),list(range(8,16))],
 actual_initial_capacity=dict(created_at_utc=now.isoformat(),root_authorizer="root_agent",actual_current_observation=True,
  source_only_estimate=False,admissible_new_bytes=33554432),
 scope_id="RAM_ONLY",run_directory=REL+"/ROOT_once_RAM_ONLY",TRAIN000_preparation_metadata_ref=prepref,
 TRAIN000_inference_close_accepted_ref=iclose,TRAIN000_native_close_accepted_ref=nclose,
 TRAIN001_inference_input_ref=cap["refs"]["input_ref"],TRAIN001_inference_outer_terminal_ref=cap["refs"]["outer_ref"],
 TRAIN001_inference_close_accepted_ref=cap["close_accepted_ref"],TRAIN001_inference_permanent_revocation_ref=cap["permanent_revocation_ref"],
 TRAIN000_simulation_completion_ref=sim000,TRAIN001_simulation_completion_ref=sim001,pipeline_expires_at_utc=expires.isoformat())
ram[param["path"]]=p
values=ns["build_values"](param);_,directory,capsule,a=values
check(a["active"]is True and a["source_only"]is False,"Build produces ROOT requested authority value only")
check(a["simulation_batch_control_input_ref"]==controlref,"Exact live original coordinator input")
check(a["TRAIN000_readiness_reuse"]["focal_identity_sha256"]=="a"*64,"Original000 focal identity reused")
check(a["TRAIN000_readiness_reuse"]["native_close_accepted_ref"]==nclose,"Original000 genuine close provenance")
check(capsule["refs"]==cap["refs"]and capsule["metadata"]==cap["metadata"],"Exact all7actual001 predecessor logical roles")
check(a["TRAIN001_inference_predecessor_capsule_ref"]is None,"No invented future capsule ref")
check(str(directory).endswith("/"+REL+"/ROOT_once_RAM_ONLY"),"Dedicated one-shot ROOT build scope")
check(a["expires_at_utc"]==expires.isoformat(),"ROOT supplied finite expiry preserved")
check(len(a["allowed_parents"])==19 and a["TRAIN001_inference_reuse_required"]is True,"No completed stage rerun")
check(a["query_truth_access_authorized"]is False and a["GPU_authorized"]is False,"No truth/GPU/science access introduced")
for label,mut in[
 ("inactive parameters",lambda x:x.update(active=False)),
 ("nonROOT parameters",lambda x:x.update(root_authorizer="author_agent")),
 ("runtime authority",lambda x:x.update(admission_or_execution_authorized=True)),
 ("source capacity estimate",lambda x:x.update(source_only_estimate=True)),
 ("wrong coordinator",lambda x:x.update(simulation_batch_control_input_ref=rr("ram/othercontrol"))),
 ("duplicate CPUlane",lambda x:x.update(CPU_lanes=[list(range(8)),list(range(8))])),
 ("underreserved admin",lambda x:x["actual_initial_capacity"].update(admissible_new_bytes=1)),
 ("stale capacity",lambda x:x["actual_initial_capacity"].update(created_at_utc=(now-timedelta(seconds=250)).isoformat())),
 ("scope traversal",lambda x:x.update(scope_id="../escape")),
 ("external build scope",lambda x:x.update(run_directory="elsewhere/ROOT_once_RAM_ONLY")),
 ("too short finite window",lambda x:x.update(pipeline_expires_at_utc=(now+timedelta(seconds=100)).isoformat()))]:
 changed=copy.deepcopy(p);mut(changed);ram[param["path"]]=changed;reject(lambda:ns["build_values"](param),label)
ram[param["path"]]=p
old_review=copy.deepcopy(ram[review["path"]]);ram[review["path"]]["source_manifest_ref"]=rr("ram/v1manifest")
reject(lambda:ns["build_values"](param),"Reject v1 source review");ram[review["path"]]=old_review
old_native=copy.deepcopy(ram[nativeref["path"]]);ram[nativeref["path"]]["query_focal_count"]=9999
reject(lambda:ns["build_values"](param),"No focal subset reuse");ram[nativeref["path"]]=old_native
old_close=copy.deepcopy(ram[nclose["path"]]);ram[nclose["path"]]["actual_direct_OS_wait0"]=False
reject(lambda:ns["build_values"](param),"No invented completed close");ram[nclose["path"]]=old_close
check("apply_reserve"not in Path(REL+"/build_ROOT_authority_v1.py").read_text(),"No accounting operation API")
result=dict(schema="TreeLoom_preparation_ROOT_builder_AST_RAM_checks_v1",passed=True,check_count=len(checks),checks=checks,
 source_AST_RAM_only=True,actual_builder_output=False,actual_guard_imported=False,actual_probes=False,actual_accounting=False,scientific_execution=False)
print(json.dumps(result,sort_keys=True,indent=2))

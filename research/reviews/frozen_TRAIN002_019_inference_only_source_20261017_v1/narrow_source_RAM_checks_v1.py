"""Narrow inference-only source/metadata/RAM checks; no live resource operations."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,hashlib,importlib,json,sys

HERE=Path("research/reviews/frozen_TRAIN002_019_inference_only_source_20261017_v1")
OLD=Path("research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2")
def main():
 checks=[];count=[0]
 def need(ok,message):
  if not ok:raise ValueError(message)
 def check(ok,name):
  count[0]+=1
  if not ok:raise AssertionError(name)
  if name not in checks:checks.append(name)
 def reject(call,name):
  try:call()
  except(ValueError,KeyError):check(True,name)
  else:raise AssertionError(name)
 def ref(path):
  p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 def synthetic(path):return dict(path=path,sha256=hashlib.sha256(path.encode()).hexdigest())
 texts={n:(HERE/n).read_text()for n in("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py")}
 oldtexts={n:(OLD/n).read_text()for n in texts};trees={k:ast.parse(v)for k,v in texts.items()}
 def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
 m=json.loads((HERE/"source_manifest.v1.json").read_text())
 for r in m["refs"]:check(ref(r["path"])==r,"all19-source-pins-authenticated")
 check(m["coordinator_source_ref"]["sha256"]=="2c4a976949fe147238ba775fa6aee58bbb782a9cef2c76d65945a472eae6be73","unchanged-actual-v3-coordinator-source-pin")
 check(m["condition"]["synchronization_pool"]==json.loads((OLD/"source_manifest.v1.json").read_text())["condition"]["synchronization_pool"],"exact-existing-two-slot-OFD-pool")
 original=json.loads((OLD/"source_manifest.v1.json").read_text())
 for key in("stage_caps","stage_templates","stage_runtime_refs"):
  check(m["condition"][key]=={"infer":original["condition"][key]["infer"]},"original-inference-science-runtime-caps-verbatim")
 for name in("run_group","prepare_close","accept_close","close_when_ready","simulated_candidate","unit_request","wait_wave_capacity"):
  check(ast.get_source_segment(texts["batch_ROOT_v1.py"],fn(trees["batch_ROOT_v1.py"],name))==
   ast.get_source_segment(oldtexts["batch_ROOT_v1.py"],fn(ast.parse(oldtexts["batch_ROOT_v1.py"]),name)),"original-owned-waits-close-source-and-admission-boundaries-byte-identical")
 check(texts["close_unit_ROOT_v1.py"]==oldtexts["close_unit_ROOT_v1.py"].replace(str(OLD),str(HERE)),"entire-original-close-helper-path-relocation-only")
 for name in("initialize","guard","admit","admit_under_coordinator","worker"):
  check(ast.get_source_segment(texts["worker_entry_v1.py"],fn(trees["worker_entry_v1.py"],name))==
   ast.get_source_segment(oldtexts["worker_entry_v1.py"],fn(ast.parse(oldtexts["worker_entry_v1.py"]),name)),"unchanged-original-supervisor-slot-memo-admission-worker-lifecycle")
 env=dict(need=need)
 exec(compile(ast.Module(body=[fn(trees["worker_entry_v1.py"],"stage_status")],type_ignores=[]),"RAM_stage_status","exec"),env)
 check(env["stage_status"]("infer")=="completed_current_C3_inference","original-infer-completion-status")
 reject(lambda:env["stage_status"]("native_paths"),"native-corpus-stage-rejected")
 reject(lambda:env["stage_status"]("simulate"),"no-simulation-or-substitute-stage")
 run=ast.unparse(fn(trees["batch_ROOT_v1.py"],"run_batch"))
 check("downstream_candidate("not in run and "seal_all20("not in run and "publish_parent_ready("in run
  and "range(2, 20)"not in run and "len(rows) < 18"in run,"18-simulation-frontier-parents-no-native-or-all20-dispatch")
 check(not any(isinstance(n,ast.FunctionDef)and n.name in("downstream_candidate","readiness_row","seal_all20")for n in trees["batch_ROOT_v1.py"].body),"native-chain-and-all20-publisher-absent")
 science=ast.unparse(fn(trees["worker_entry_v1.py"],"execute_in_admitted_context_v1"))
 check("STAGE == 'infer'"in science and "american_c3_observe_v3 as science"in science
  and "american_c3_paths_v1 as science"not in science and "science.execute(cfg, writer)"in science
  and "publish_focal_seal("not in science,"original-inference-execute-only-no-native-scientific-import")
 parents=["american_c3_train_%03d"%i for i in range(2,20)]
 check(m["allowed_parents"]==parents and len(parents)==18,"exact18original-TRAIN002-through019-order")
 # Changed finite authority validator against RAM control metadata.
 a=json.loads((HERE/"ROOT_authority.template.INACTIVE.v1.json").read_text())
 now=datetime.now(timezone.utc);ar=synthetic("RAM/inference_authority.json");controlref=synthetic("RAM/control.json");simref=synthetic("RAM/simulation_authority.json")
 a.update(active=True,source_only=False,root_authorizer="root_agent",created_at_utc=now.isoformat(),
  expires_at_utc=(now+timedelta(hours=12)).isoformat(),simulation_batch_control_input_ref=controlref)
 records={ar["path"]:a,controlref["path"]:dict(authority_ref=simref),simref["path"]:dict(expires_at_utc=(now+timedelta(hours=13)).isoformat())}
 def read(r):return copy.deepcopy(records[r["path"]])
 env=dict(need=need,read=read,metadata_ref=lambda p:ref(HERE/"source_manifest.v1.json"),MANIFEST=str(HERE/"source_manifest.v1.json"),
  datetime=datetime,timezone=timezone,ALLOWED_PARENTS=tuple(parents),
  coordinator_support=lambda:SimpleNamespace(validate_batch_authority=lambda r:need(r==simref,"Exact RAM simulation control")))
 exec(compile(ast.Module(body=[fn(trees["worker_entry_v1.py"],"validate_pipeline_authority")],type_ignores=[]),"RAM_authority","exec"),env)
 check(env["validate_pipeline_authority"](ar)["stages"]==["infer"],"finite-inference-only-authority-positive")
 for field,value,name in[("stages",["infer","native_paths"],"mixed-stage-authority-rejected"),
  ("native_path_corpus_preparation_authorized",True,"native-corpus-authorization-rejected"),
  ("allowed_parents",["american_c3_train_001"]+parents,"TRAIN001-replay-parent-rejected"),
  ("stage_reservation_bytes",dict(infer=16500000000),"changed-inference-reservation-rejected")]:
  old=records[ar["path"]][field];records[ar["path"]][field]=value
  reject(lambda:env["validate_pipeline_authority"](ar),name);records[ar["path"]][field]=old
 # Parent frontier publication in RAM, not an all20 training barrier.
 ri=synthetic("RAM/input.json");rd=synthetic("RAM/completed.json");ro=synthetic("RAM/observation.json");rw=synthetic("RAM/wait.json")
 rv=synthetic("RAM/accepted.json");rr=synthetic("RAM/roster.json");sc=synthetic("RAM/simcompleted.json")
 values={ri["path"]:dict(parent_id=parents[0],refs=dict(config=synthetic("RAM/config.json"))),
  rd["path"]:dict(status="completed_current_C3_inference",observation_ref=ro),
  ro["path"]:dict(parent_id=parents[0],sequence_length_bp=30000000,query_ancestry_truth_in_inputs=False,
   genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",roster_ref=rr)}
 saved={}
 def put(ctx,name,value):saved[name]=copy.deepcopy(value);return synthetic("RAM/"+name)
 env=dict(need=need,read=lambda r:copy.deepcopy(values[r["path"]]),terminal=lambda s:dict(completed_ref=rd,actual_wait_ref=rw),
  ref=lambda path:synthetic(str(path)),put=put,MANIFEST=str(HERE/"source_manifest.v1.json"),WORKER=str(HERE/"worker_entry_v1.py"))
 exec(compile(ast.Module(body=[fn(trees["batch_ROOT_v1.py"],"publish_parent_ready")],type_ignores=[]),"RAM_frontier","exec"),env)
 unit=dict(parent_id=parents[0],stage="infer",unit_key="train_002_infer",simulation_completion_ref=sc)
 env["publish_parent_ready"](SimpleNamespace(directory=Path("RAM")),unit,ri,{},rv)
 row=saved["parent_ready."+parents[0]+".json"]
 check(row["observation_ref"]==ro and row["inference_close_accepted_ref"]==rv and row["query_truth_opened"]is False
  and row["native_path_corpus_prepared"]is False and row["compact_focals_still_required"]is True
  and row["source_manifest_ref"]["path"].endswith("source_manifest.v1.json"),"incremental-authenticated-inference-frontier-exact-compact-focal-interface")
 bad=dict(unit,stage="native_paths")
 reject(lambda:env["publish_parent_ready"](SimpleNamespace(directory=Path("RAM")),bad,ri,{},rv),"frontier-does-not-claim-native-corpus")
 values[ro["path"]]["genealogy_kind"]="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG"
 reject(lambda:env["publish_parent_ready"](SimpleNamespace(directory=Path("RAM")),unit,ri,{},rv),"oracle-output-cannot-publish-inferred-ready")
 # Exact original consumer-union source and real baseline; no actual guard import.
 baseline=json.loads(Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_records_root01_oct17/reserve_authority.json").read_text())["bindings"]
 check(len(baseline)==47 and len({json.dumps(v,sort_keys=True)for v in baseline.values()})==40,"genuine47-role40-unique-baseline")
 gt=ast.parse(Path(m["close_pins"]["guard"]["path"]).read_text());ge=dict(copy=copy,require=need,_raw=lambda r:None)
 exec(compile(ast.Module(body=[fn(gt,"consumer_bindings_v1")],type_ignores=[]),"original_binding_union","exec"),ge)
 admin_direct=m["refs"]+list(m["close_pins"].values())+[ref(HERE/"source_manifest.v1.json"),synthetic("RAM/auth.json"),synthetic("RAM/review.json"),m["controller_runtime_ref"],synthetic("RAM/control_input.json")]
 check(len(ge["consumer_bindings_v1"](list(baseline.values()),admin_direct))==63,"admin-full-actual-binding-union63-under-original64ceiling")
 # Original unit direct roles are unchanged; relocation substitutes existing source refs.
 check(ast.get_source_segment(texts["worker_entry_v1.py"],fn(trees["worker_entry_v1.py"],"direct_roles"))==
  ast.get_source_segment(oldtexts["worker_entry_v1.py"],fn(ast.parse(oldtexts["worker_entry_v1.py"]),"direct_roles")),"all-original25-unit-direct-roles-preserved")
 for name in("worker_entry_v1","batch_ROOT_v1","close_unit_ROOT_v1"):
  module=importlib.import_module(str(HERE).replace("/",".")+"."+name);check(module.main([])==0,"all-three-real-defaults-inactive0")
 check(not any(n in sys.modules for n in("torch","torch_geometric","_batch_exact_current_Census")),"no-actual-guard-or-scientific-import")
 trainer=Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/source_manifest.v1.json")
 check(ref(trainer)["sha256"]=="35760763a7f0608ed56cd62565ec555d4fcae7f622ea6bfb9dd5280257d20fa0","sealed-nine-trainer-manifest-unchanged")
 for path,sha in json.loads(trainer.read_text())["source_sha256"].items():check(ref(path)["sha256"]==sha,"all85-scientific-source-metadata-pins-unchanged")
 return dict(schema="TRAIN002_019_inference_only_narrow_source_RAM_checks_v1",source_only=True,checks_passed=len(checks),
  assertions_passed=count[0],checks=checks,source_manifest_ref=ref(HERE/"source_manifest.v1.json"),
  actual_guard_resource_process_GPU_lock_state_probe_admission_launch_science_operations=0,
  actual_waits_simulations_inferences_closes_timing_and_all20_readiness_claimed=False)

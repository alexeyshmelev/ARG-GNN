"""Narrow compact CPU control checks; source/metadata and isolated RAM only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from contextlib import contextmanager
import ast,copy,hashlib,importlib,json,sys
HERE=Path("research/reviews/TreeLoom_compact_focal_CPU_continuation_source_20261017_v1")
OLD=Path("research/reviews/frozen_TRAIN002_019_inference_only_source_20261017_v1")
def main():
 checks=[];count=[0]
 def need(ok,msg):
  if not ok:raise ValueError(msg)
 def check(ok,msg):
  count[0]+=1
  if not ok:raise AssertionError(msg)
  if msg not in checks:checks.append(msg)
 def reject(call,msg):
  try:call()
  except(ValueError,KeyError):check(True,msg)
  else:raise AssertionError(msg)
 def ref(path):
  p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 def synthetic(path):return dict(path=path,sha256=hashlib.sha256(path.encode()).hexdigest())
 m=json.loads((HERE/"source_manifest.v1.json").read_text());texts={n:(HERE/n).read_text()for n in("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py","build_ROOT_authority_v1.py")}
 trees={n:ast.parse(t)for n,t in texts.items()}
 def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
 for r in m["refs"]:check(ref(r["path"])==r,"all20-exact-executed-source-metadata-pins")
 check(ref(m["compact_component_source_ref"]["path"])["sha256"]=="74ec68ceb13264b01bc432f74df375a8b229a81afd0b101b656a2eb98049d2f9","exact-sealed-compact-component-unchanged")
 check(ref("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/source_manifest.v1.json")["sha256"]=="35760763a7f0608ed56cd62565ec555d4fcae7f622ea6bfb9dd5280257d20fa0","nine-model-source-manifest-unchanged")
 for name in("run_group","prepare_close","accept_close","close_when_ready","group_call"):
  old=(OLD/"batch_ROOT_v1.py").read_text()
  check(ast.get_source_segment(texts["batch_ROOT_v1.py"],fn(trees["batch_ROOT_v1.py"],name))==ast.get_source_segment(old,fn(ast.parse(old),name)),"unchanged-original-pidfd-owned-waits-drain-close-and-acceptance")
 cap=m["condition"]["stage_caps"]["compact_focals"]
 check(cap["reservation_bytes"]==18874368==2**21+16*2**20 and cap["aux_bytes"]==16*2**20
  and cap["log_bytes"]+cap["private_work_bytes"]+11*2**20==cap["aux_bytes"],"derived18MiB-whole-cap2MiBprimary16MiBAUX-no-corpora")
 w=importlib.import_module(str(HERE).replace("/",".")+".worker_entry_v1")
 for i in range(1,20):
  cfg=w.scientific_config("american_c3_train_%03d"%i,"compact_focals",synthetic("RAM/observed%d.json"%i))
  check(cfg["path_sampling_seed"]==92001+i and cfg["output_cap_bytes"]==2097152 and cfg["query_focal_count"]==10000
   and cfg["schema"]=="TreeLoom_compact_original_focal_preparation_config_v1","all19exact-original-component-config-builder")
 reject(lambda:w.stage_status("native_paths"),"no-native-corpus-stage-route")
 reject(lambda:w.stage_status("infer"),"no-inference-replay-route-in-compact-controller")
 # Actually execute new initializer with RAM original helper objects, no original helper/guard import.
 saved={k:getattr(w,k)for k in("_load_initial","validate_pipeline_authority","PARENT","STAGE","PHASE","SLOT_REL","_selected_authority_ref","c","d")}
 authority=dict(run_directory=str(HERE)+"/ROOT_actual_pipeline_RAM")
 fake=SimpleNamespace(PIDFD=m["pidfd_ref"],OWNED=m["owned_ref"],require=need,contextmanager=contextmanager,contextlib=SimpleNamespace(contextmanager=contextmanager))
 fake.ref=lambda path:synthetic(str(path))
 fake.read=lambda r:(dict(source_ref=w.CHILD,selector_source_ref=w.CHILD,policy_ref=m["close_pins"]["policy"])if r==w.SELECTOR else m)
 fake.local=lambda p:p;fake.raw=lambda r:b"";fake.load=lambda *a:None;fake.put=lambda *a:None;fake.current=lambda *a:None
 driver=SimpleNamespace()
 w._load_initial=lambda path,sha,name:fake if path.endswith("root_contract_v1.py")else driver
 w.validate_pipeline_authority=lambda r:authority;w.PARENT="american_c3_train_001";w.STAGE="compact_focals";w.PHASE="compact_focals_train_001"
 w.SLOT_REL=authority["run_directory"]+"/producer_slots";w._selected_authority_ref=synthetic("RAM/authority.json");w.c=w.d=None
 w.initialize()
 check(w.c is fake and driver.c is fake and any(type(v)is str and v.startswith(w.SLOT_REL+"/cpu_slot_")for v in fake.cpu_slot.__wrapped__.__code__.co_consts),
  "actual-initializer-original-slot-function-binds-dedicated-charged-path")
 for k,v in saved.items():setattr(w,k,v)
 # Honest compact completion/seal joins against RAM only.
 parent="american_c3_train_001";obs=synthetic("RAM/observed.json");sealref=synthetic("RAM/focal_seal.json");focal=synthetic("RAM/focals.npz")
 seal=dict(schema="TreeLoom_original_native_focal_identity_seal_v1",parent_id=parent,observation_ref=obs,
  focal_registry_ref=focal,focal_identity_sha256="f"*64,focal_records=10000,query_truth_opened=False,
  preparation_kind="compact_original_identity_focals_without_native_corpus")
 done=dict(schema="TreeLoom_compact_original_focal_preparation_completion_v1",status=w.COMPACT_STATUS,parent_id=parent,
  observation_ref=obs,genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
  class_order=["EUR","EAS","AFR"],query_focal_count=10000,path_sampling_seed=92002,full_original_path_traversal=True,
  reference_paths_counted_in_global_record_ids=True,native_corpus_materialized=False,native_dataset_files_written=0,
  native_vocabulary_materialized=False,native_train_val_split_executed=False,sequence_to_tree_occurrence_mapping_materialized=False,
  query_ancestry_truth_opened=False,focal_seal_ref=sealref,focals_ref=focal,focal_identity_sha256="f"*64)
 oldread=w.read;w.read=lambda r:copy.deepcopy(seal)
 check(w.validate_compact_done(done,parent,obs)==done,"real-compact-only-completion-and-original-focal-seal-positive")
 for field,value,msg in[("status","completed_native_C3_path_preparation","fake-native-status-rejected"),
  ("native_corpus_materialized",True,"fake-corpus-materialization-rejected"),("native_train_val_split_executed",True,"fake-native-split-rejected"),
  ("query_ancestry_truth_opened",True,"query-truth-access-rejected"),("native_dataset_refs",{},"unused-native-corpus-reference-rejected")]:
  bad=dict(done);bad[field]=value;reject(lambda:w.validate_compact_done(bad,parent,obs),msg)
 w.read=oldread
 # Compute actual unchanged original consumer union on real47-role40unique baseline.
 baseline=json.loads(Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_records_root01_oct17/reserve_authority.json").read_text())["bindings"]
 check(len(baseline)==47 and len({json.dumps(x,sort_keys=True)for x in baseline.values()})==40,"genuine47role40unique-baseline")
 gt=ast.parse(Path(m["close_pins"]["guard"]["path"]).read_text());env=dict(copy=copy,require=need,_raw=lambda r:None)
 exec(compile(ast.Module(body=[fn(gt,"consumer_bindings_v1")],type_ignores=[]),"original_consumer_union","exec"),env)
 admin_direct=m["refs"]+list(m["close_pins"].values())+[ref(HERE/"source_manifest.v1.json"),synthetic("RAM/authority.json"),synthetic("RAM/review.json"),m["controller_runtime_ref"],synthetic("RAM/control.json")]
 admin_count=len(env["consumer_bindings_v1"](list(baseline.values()),admin_direct))
 check(admin_count==64,"exact-admin64-union-all-original-roles-no-ceiling-increase")
 c=SimpleNamespace(ref=ref,PIDFD=m["pidfd_ref"],OWNED=m["owned_ref"],NAMESPACE_REF=next(r for r in m["refs"]if r["path"].endswith("consumer_namespace_memo_v1.py")),
  V4_REF=next(r for r in m["refs"]if r["path"].endswith("project_budget_v4.py")),POLICY_REF=m["close_pins"]["policy"])
 ge=dict(need=need,c=c,SOURCE=str(HERE/"worker_entry_v1.py"),MANIFEST=str(HERE/"source_manifest.v1.json"),OLD=w.OLD,
  CORE_REF=w.CORE_REF,CORE_MANIFEST=w.CORE_MANIFEST,PATHS=w.PATHS,AUTHOR=w.AUTHOR,IDENTITY_SOURCE=w.IDENTITY_SOURCE,SELECTOR=w.SELECTOR,CHILD=w.CHILD,SIM_BATCH_SOURCE=w.SIM_BATCH_SOURCE)
 exec(compile(ast.Module(body=[fn(trees["worker_entry_v1.py"],"direct_roles")],type_ignores=[]),"actual_direct_role_expression","exec"),ge)
 qr=synthetic("RAM/unit_request.json");q=dict(source_review_ref=synthetic("RAM/review.json"),prior_capsule_ref=synthetic("RAM/prior.json"),
  batch_control_input_ref=synthetic("RAM/control.json"),pipeline_authority_ref=synthetic("RAM/authority.json"))
 direct=ge["direct_roles"](qr,q,synthetic("RAM/config.json"),qr,m["condition"]["stage_runtime_refs"]["compact_focals"])
 unit_count=len(env["consumer_bindings_v1"](list(baseline.values()),direct))
 check(len(direct)==25 and unit_count==64,"actual-unit25roles64union-preserves-all-current-roles")
 # Empty-frontier loop has no accounting validation outside advertised frontier branch.
 run=fn(trees["batch_ROOT_v1.py"],"run_batch");loop=next(x for x in ast.walk(run)if isinstance(x,ast.While)and "len(rows)"in ast.unparse(x.test))
 first=next(x for x in loop.body if isinstance(x,ast.If)and ast.unparse(x.test)=="ready_parents")
 check(any(isinstance(x,ast.Call)and ast.unparse(x.func)=="state"for x in ast.walk(first))
  and not any(isinstance(x,ast.Call)and ast.unparse(x.func)in("state","w.validate_pipeline_authority")for n in loop.body[:loop.body.index(first)]for x in ast.walk(n)),
  "absent-frontier-poll-only-cached-clock-and-known-paths-no-guard-state-source-replay")
 source_exec=ast.unparse(fn(trees["worker_entry_v1.py"],"execute_in_admitted_context_v1"))
 check("c.load(CORE_REF"in source_exec and "american_c3_paths_v1 as science"not in source_exec
  and "science.execute(cfg, writer)"in source_exec,"only-original-reviewed-compact-science-dispatch")
 for name in("worker_entry_v1","batch_ROOT_v1","close_unit_ROOT_v1","build_ROOT_authority_v1"):
  module=importlib.import_module(str(HERE).replace("/",".")+"."+name);check(module.main([])==0,"all4actual-defaults-inactive0")
 check(not any(n in sys.modules for n in("torch","torch_geometric","_batch_exact_current_Census","_actual_original_compact_focal_component")),"no-actual-guard-resource-or-scientific-import")
 return dict(schema="compact_focal_CPU_narrow_source_RAM_checks_v1",source_only=True,checks_passed=len(checks),assertions_passed=count[0],
  checks=checks,admin_union=admin_count,unit_union=unit_count,unit_direct_roles=len(direct),source_manifest_ref=ref(HERE/"source_manifest.v1.json"),
  actual_guard_resource_process_GPU_socket_lock_state_probe_admission_launch_scientific_operations=0,
  actual_fullparent_focal_equivalence_or_timing_claimed=False)

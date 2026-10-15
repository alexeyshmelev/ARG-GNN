"""Current980 GNN root contract. Metadata only; all actions require explicit root modes."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
SOURCE=REL+"/root_contract_v1.py";DRIVER=REL+"/root_driver_v1.py";ENTRY=REL+"/worker_entry_v1.py"
CANONICAL="src/arggnn/american_c3_gnn_prefetch_isolated980_exec_v2.py"
GPU_MODULE="arggnn.american_c3_gnn_prefetch_isolated980_exec_v2"
GPU_MAIN=CANONICAL;CPU_MAIN=ENTRY
BASE_HELPER=dict(path="research/reviews/C3_native_postMLM980_root_source_20261015_v1/root_native_contract_v1.py",
 sha256="aeb992d49d632003cbd971432afa607cc8e8ba2b0658f57501fcc435c8d74b69")
def _load_base():
 p=ROOT/BASE_HELPER["path"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=BASE_HELPER["sha256"]:raise ValueError("Exact reviewed common guard/metadata utilities")
 sp=importlib.util.spec_from_file_location("_C3GNN_current980_common",p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
_base=_load_base()
require,local,ref,read,save,load,active=(_base.require,_base.local,_base.ref,_base.read,_base.save,_base.load,_base.active)
CORE,BASE,GUARD_REF,POLICY_REF,REPAIR_REF,MEMO_REF,V4_REF,GPU_AUTH_REF=(_base.CORE,_base.BASE,_base.GUARD_REF,_base.POLICY_REF,_base.REPAIR_REF,_base.MEMO_REF,_base.V4_REF,_base.GPU_AUTH_REF)
original_guard,consumer_guard=_base.original_guard,_base.consumer_guard
LAUNCHER_REF=dict(path="scripts/launch_reserved_gpu_authorized_GPU45_980_v1.py",sha256="0ca3c59e49f5fd55f5a798c99a4a8a6613e71a90bcb3f89a04eb66e7555046ca")
BASE_CONFIG_REF=dict(path="configs/census_tsinfer_fit_v1.json",sha256="7c2436d5479e68beb1a8f3edb04ca0ee729a23be35ca2d7893827c0db68e3b00")
RUNTIME_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json",sha256="e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602")
HOOK_REF={'path': 'research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/execution_hooks_prefetch_v1.py', 'sha256': 'b7418267d329b0d83df45c5f5d053ffd034db0e88c794d5a59c7b59ca52900d1'}
TRAIN_SOURCE_REF={'path': 'research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/trainer_prefetch_v1.py', 'sha256': 'b2f08a3e84bd5fb65c5f3d95566ab082fb71a2dd0a826180a10aa518aa35025c'}

PREFETCH_SOURCE_PINS={'research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/prefetch_provider_isolated_v2.py': '64fdb94c4cdf16163d2443b1aea8d2afde8f389d6f6690628ec622e13c98c057', 'research/reviews/C3_GNN_prefetch_provider_20261015_v1/bounded_workers_v1.py': '66f541e7e10bf0c86f1b28b39cfa69924a31b46974548d4a1f9fbf1d0aa6c21a', 'research/reviews/C3_GNN_prefetch_provider_20261015_v1/abstract_ipc_bootstrap_v1.py': '1d8f11ff88481ae5fabb4794cdd25aeafd377d253b308cb026f020c966989ad4', 'research/reviews/C3_GNN_prefetch_provider_20261015_v1/worker_entry_integration_api_v1.py': 'aebb1f2d75277f5311dadbdbfdad98be3906dbb0a25db92cee1060ce64b52f4b', 'research/reviews/C3_GNN_prefetch_provider_20261015_v1/transport_handoff.v1.json': '8338194f068d3dcc5cedb9648e74a6cd3f01126d3ac8aa185a1ce7b68bb78495', 'research/reviews/C3_GNN_prefetch_provider_20261015_v1/source_handoff.v1.json': 'b267322dccd07f480163240ae7755fcebbeab69c5bc6b4b7080759d22666b161', 'research/reviews/C3_GNN_prefetch_provider_20261015_v1/torch_pyg_transport_receipt.v1.json': '6164ccd1975a9ffdc75acd53f3f4bfccadc1c29c4bb6e89fc56cf1a6b5c215c6', 'research/reviews/C3_GNN_prefetch_provider_20261015_v1/torch_pyg_transport_actual_wait.v1.json': '48555568895376ee270c4613342f5ee4a0bb57c5cf788eb9b127db7eb52d2b16', 'src/arggnn/american_c3_gnn_stream_train_isolated_v2.py': '09394390199ceee020e4123d25a4cfcdcee38229450abc3ddd24dde55773b912', 'src/arggnn/american_c3_gnn_dataset_isolated_v2.py': '6e4f32fcec9e6af0c6aa2c81cdbcc4ef1bf7ff83da04b233f0064b6a65c33cb2', 'src/arggnn/american_c3_gnn_observations_isolated_v2.py': 'adf1be0aab14e54b0d48c8cb577b3df541bed2ee88ac846f16bab8a75d56f781', 'research/reviews/C3_GNN_prefetch_provider_20261015_v1/prefetch_provider_v1.py': 'd4a29905b6b7d1f4daff9946557e8967ab51665a0c5648603e25f028d782b4a9'}
LEGACY_DRIVER_REF={'path': 'research/reviews/C3_GNN_current980_executable_root_20261015_v1/root_driver_v1.py', 'sha256': '95f5c8fcbf096f9f648f6ff293152cc81b5adb36586b129c80e4e6dd1326015c'}
LEGACY_REL='research/reviews/C3_GNN_current980_executable_root_20261015_v1'
FROZEN_TRAIN_CONFIG_REF={'path': 'research/reviews/C3_GNN_current980_executable_root_20261015_v1/training_config.inactive.json', 'sha256': '7c04aa9274ee091f7677287df68f74c5419d5b611a5672c69ae4095d0cdf7d26'}
PREFETCH_CAP_KEYS=frozenset(("num_workers","max_block_bytes","worker_rss_cap_bytes","aggregate_rss_cap_bytes","task_timeout_seconds","drain_timeout_seconds"))

TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
GPU_PHASES=("train",)+tuple("predict_val_%03d"%i for i in range(5))
CPU_PHASES=("matrix","score")+tuple("plan_val_%03d"%i for i in range(5))
PHASES=GPU_PHASES+CPU_PHASES
PARENT="American_C3_short20_TRAIN_original5VAL"
CONTEXT=ENTRY;RUNTIME_HELPER=ENTRY
PROC_HELPER=_base.PROC_HELPER
STATUSES={p:"completed_C3_GNN_prefetch_isolated980_"+p for p in PHASES}
def launcher_ref_v1():return LAUNCHER_REF
def phase_paths(stage):
 require(stage in PHASES,"Registered independent GNN stage")
 return ROOT/REL/("admitted_"+stage+"_v1"),"experiments/results/C3_GNN_prefetch_isolated980_"+stage+"_v1"
def verify_shim():
 target=dict(path=CANONICAL,sha256=ref(REL+"/canonical_dispatch.staged.py")["sha256"])
 require(ref(CANONICAL)==target,"Root must first integrate the exact additive canonical shim")
 return target
def hooks():
 return load(HOOK_REF,"_C3_GNN_existing_execution_hooks")
ORIGINAL_METADATA_IMPORT_SOURCE_PATHS=('src/arggnn/american_c3_gnn_dataset_v1.py', 'src/arggnn/american_c3_gnn_stream_train_v1.py', 'src/arggnn/american_c3_gnn_observations_v1.py', 'src/arggnn/american_c3_gnn_run_v1.py', 'src/arggnn/american_c3_gnn_train_v1.py')
PRIOR_PREFETCH_REL='research/reviews/C3_GNN_prefetch_current980_root_20261015_v1'
PRIOR_PREFETCH_DRIVER_REF={'path': 'research/reviews/C3_GNN_prefetch_current980_root_20261015_v1/root_driver_v1.py', 'sha256': '69a21fc2b8ef0d2a165fd6139c2348537f6920db082483dd296c23db02dbc29a'}
def source_paths():
 scientific=("american_c3_common_v1","american_c3_gnn_targets_v1","american_c3_gnn_stream_train_isolated_v2","american_c3_gnn_train_v1",
 "american_c3_gnn_model_v1","american_c3_gnn_run_isolated_v2","american_c3_gnn_dataset_isolated_v2","american_c3_gnn_observations_isolated_v2",
 "american_c3_evaluate_v1","mutation_routed_whole_tree_gnn_v1","mutation_routed_tree_observations_v1","mutation_routed_pyg_gnn_v1",
 "american_c3_gnn_predict_isolated_v2","american_c3_gnn_selection_isolated_v2","american_c3_gnn_shared30mb_isolated_v2",
 "native_argformer_exec","biological_candidate","development_execution","census_tsinfer_fit_v1","project_budget_v9_consumer","resources")
 cfg=read(BASE_CONFIG_REF)
 return tuple(dict.fromkeys((*(("src/arggnn/"+x+".py") for x in scientific),
  SOURCE,DRIVER,ENTRY,CANONICAL,HOOK_REF["path"],TRAIN_SOURCE_REF["path"],BASE_HELPER["path"],LAUNCHER_REF["path"],
  REL+"/metadata_contract_v1.py",REL+"/tracking_factory_v1.py",REL+"/persistent_provider_v1.py",REL+"/persistent_workers_v1.py",*PREFETCH_SOURCE_PINS,LEGACY_DRIVER_REF["path"],FROZEN_TRAIN_CONFIG_REF["path"],*ORIGINAL_METADATA_IMPORT_SOURCE_PATHS,PRIOR_PREFETCH_DRIVER_REF["path"],
  "scripts/launch_reserved_gpu_full_storage_v1.py","scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py",
  "scripts/project_budget_v9.py",cfg["gpu_runtime"]["nvtop_helper"],PROC_HELPER,
  "research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py",
  "research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py")))
def validate_terminal(envelope_ref,wait_ref,stage):
 packet,output=phase_paths(stage)
 require(envelope_ref["path"]==output+"/completed.json" and wait_ref["path"]==str((packet/"actual_wait.json").relative_to(ROOT)),"Exact own upstream stage paths")
 v,w=read(envelope_ref),read(wait_ref)
 require(v.get("status")==STATUSES[stage] and v.get("stage")==stage and v.get("root_driver_ref")==ref(DRIVER)
  and w.get("actual_direct_launcher_wait") is True and w.get("all_owned_processes_and_groups_drained") is True
  and w.get("returncode")==0 and w.get("operational_failure") is None and w.get("completed_ref")==envelope_ref
  and w.get("driver_ref")==ref(DRIVER),"Actual prior GNN stage completion/directwait0/drain")
 return v
def validate_preparation_wait(completed_ref,wait_ref):
 v,w=read(completed_ref),read(wait_ref)
 require(w.get("returncode")==0 and w.get("completed_ref")==completed_ref
  and (w.get("actual_direct_worker_wait") is True or w.get("actual_direct_launcher_wait") is True)
  and any(w.get(k) is True for k in ("owned_process_group_ESRCH","owned_worker_group_ESRCH","worker_process_group_drained","owned_group_drained","all_owned_processes_and_groups_drained"))
  and w.get("operational_failure") is None,"Actual preparation wait0/drain")
 return v
def validate_plan_metadata(record,preparation_ref,parent,length):
 plan=read(record,cap=33554432);prep=read(preparation_ref)
 require(plan["schema"]=="american_c3_whole_tree_block_index_v1" and plan["parent_id"]==parent
  and plan["sequence_length_bp"]==length and plan["haplotypes"]==1600 and plan["focal_records"]==10000
  and plan["class_order"]==["EUR","EAS","AFR"] and plan["all_trees_and_samples_retained"] is True
  and plan["targets_opened"] is False and plan["config"]["native_preparation_ref"]==preparation_ref
  and prep["observation_ref"]==plan["observation_ref"] and prep["focals_ref"]==plan["focal_registry_ref"]
  and prep["parent_id"]==parent and prep["query_ancestry_truth_opened"] is False,"Same genuine full observation/native focals and plan")
 require(all(plan["config"][k]==v for k,v in dict(max_trees_per_block=32,max_nodes_per_block=250000,max_sites_per_block=4096,reference_counts=[200,200,200]).items()),"Unchanged exact block recipe")
 return plan
def validate_bindings(stage,b):
 if stage=="train":
  require(set(b)=={"actual_train_bindings","parent_plan_refs","parent_plan_completion_refs","parent_plan_wait_refs"},"TRAIN-only metadata capabilities")
  meta=hooks().actual_train_metadata_v1(b["actual_train_bindings"])
  for key in ("parent_plan_refs","parent_plan_completion_refs","parent_plan_wait_refs"):require(set(b[key])==set(TRAIN),"All20 genuine short TRAIN plans")
  for parent in TRAIN:
   done=validate_preparation_wait(b["parent_plan_completion_refs"][parent],b["parent_plan_wait_refs"][parent])
   require(done["parent_id"]==parent and done["block_index_ref"]==b["parent_plan_refs"][parent]
    and done["observation_ref"]==meta[parent]["binding"]["inferred_observation_ref"],"Actual same-parent completed focal/plan")
   plan=validate_plan_metadata(b["parent_plan_refs"][parent],done["native_preparation_ref"],parent,2000000)
   require(plan["focal_registry_ref"]==done["focal_registry_ref"],"Exact completed native focal registry")
 elif stage.startswith("predict_val_"):
  require(set(b)=={"parent_id","plan_ref","native_preparation_ref","preparation_wait_ref","training","plan_stage_evidence"},"Truth-free single original VAL capability")
  parent=VAL[int(stage[-3:])];require(b["parent_id"]==parent,"Fixed original VAL stage identity")
  validate_preparation_wait(b["native_preparation_ref"],b["preparation_wait_ref"])
  validate_plan_metadata(b["plan_ref"],b["native_preparation_ref"],parent,30000000)
  validate_closed_plan_stage(b["plan_stage_evidence"],b["plan_ref"],parent)
  t=b["training"];validate_terminal(t["completion_ref"],t["actual_wait_ref"],"train")
 elif stage.startswith("plan_val_"):
  require(set(b)=={"parent_id","plan_config","native_preparation_ref","preparation_wait_ref"},"Observation-only original VAL plan")
  parent=VAL[int(stage[-3:])];require(b["parent_id"]==parent,"Original VAL")
  prep=validate_preparation_wait(b["native_preparation_ref"],b["preparation_wait_ref"])
  cfg=b["plan_config"]
  require(cfg["native_preparation_ref"]==b["native_preparation_ref"] and cfg["focal_registry_ref"]==prep["focals_ref"]
   and cfg["observation_ref"]==prep["observation_ref"] and cfg["execution_enabled"] is True,"Genuine original native preparation, no model dependency")
 elif stage=="matrix":
  require(set(b)=={"training","VAL_predictions"} and set(b["VAL_predictions"])==set(VAL),"All5 original VAL prediction completions")
  validate_terminal(b["training"]["completion_ref"],b["training"]["actual_wait_ref"],"train")
  for i,parent in enumerate(VAL):
   item=b["VAL_predictions"][parent];validate_terminal(item["completion_ref"],item["actual_wait_ref"],"predict_val_%03d"%i)
 elif stage=="score":
  require(set(b)=={"matrix","VAL_truth_capability"} and set(b["VAL_truth_capability"])==set(VAL),"Separate original5VAL truth capability; no TEST")
  validate_terminal(b["matrix"]["completion_ref"],b["matrix"]["actual_wait_ref"],"matrix")
  for parent,record in b["VAL_truth_capability"].items():
   value=read(record)
   require(value["status"]=="completed_C3_simulation_reconstruction" and value["parent_id"]==parent
    and value["private_truth_ref"]["path"]==str(Path(record["path"]).parent/"truth/c3_pulse_ancestry.npz"),"Exact original same-parent VAL truth handle metadata")
 return b
def validate_closed_plan_stage(evidence,plan_ref,parent):
 require(isinstance(evidence,dict) and set(evidence)=={"route","completion_ref","actual_wait_ref","close_receipt_ref"},"Actual closed observation-plan evidence")
 stage="plan_val_"+parent[-3:]
 require(evidence["route"] in ("current980_zero_worker","prefetch_current980","prefetch_isolated980"),"Known immutable plan route")
 legacy=evidence["route"]=="current980_zero_worker"
 prior_prefetch=evidence["route"]=="prefetch_current980"
 rel=LEGACY_REL if legacy else (PRIOR_PREFETCH_REL if prior_prefetch else REL)
 prefix="C3_GNN_current980_" if legacy else ("C3_GNN_prefetch_current980_" if prior_prefetch else "C3_GNN_prefetch_isolated980_")
 driver=LEGACY_DRIVER_REF if legacy else (PRIOR_PREFETCH_DRIVER_REF if prior_prefetch else ref(DRIVER))
 folder=rel+"/admitted_"+stage+"_v1"
 require(evidence["completion_ref"]["path"]=="experiments/results/"+prefix+stage+"_v1/completed.json"
  and evidence["actual_wait_ref"]["path"]==folder+"/actual_wait.json"
  and evidence["close_receipt_ref"]["path"]==folder+"/closed/close_receipt.json","Exact immutable observation-plan paths")
 v,w,z=(read(evidence[k]) for k in ("completion_ref","actual_wait_ref","close_receipt_ref"))
 require(v["stage"]==stage and v["status"]=="completed_"+prefix+stage and v["root_driver_ref"]==driver
  and w["completed_ref"]==evidence["completion_ref"] and w["returncode"]==0
  and w["actual_direct_launcher_wait"] is True and w["all_owned_processes_and_groups_drained"] is True
  and w["operational_failure"] is None and w["driver_ref"]==driver,"Genuine same-source plan terminal0/drain")
 result=read(v["result_ref"])
 require(result["status"]=="completed_C3_observation_only_block_plan" and result["parent_id"]==parent
  and result["block_index_ref"]==plan_ref and result["targets_opened"] is False,"Exact completed observation-only plan")
 require(z["kind"]=="concurrent_completed_scope_close_receipt_v1"
  and z["value"]["future_writes_and_relaunch_revoked"] is True
  and z["value"]["request"]["completed_ref"]==evidence["completion_ref"]
  and z["value"]["request"]["direct_wait_ref"]==evidence["actual_wait_ref"],"Root actually closed immutable plan scope")
 obs=read(z["value"]["request"]["observation_ref"])
 approval=read(obs["root_approval_ref"])
 require(obs["root_authorizer"]=="root_agent" and obs["completed_scope_all_old_original_resume_write_capabilities_revoked"] is True
  and approval["driver_ref"]==driver and approval["root_actually_consumed_supervisor_terminal_wait0"] is True
  and approval["all_prior_launch_resume_write_capabilities_revoked"] is True,"Actual root terminal consumption/revocation")
 return result

def validate_caps(req,caps,*,activated):
 stage=req["phase"]
 require(caps.get("schema")=="C3_GNN_prefetch_isolated980_operational_caps_v1"
  and caps.get("active") is activated and caps.get("phase") in (None,stage)
  and caps.get("bound_method")=="operational_fail_stop_not_measured_fit_or_timing","Actual explicit operational caps")
 for key in ("reservation_cap_bytes","phase_write_budget_bytes","temp_allowance_bytes","outer_auxiliary_cap_bytes",
  "root_packet_overhead_cap_bytes","outer_log_cap_bytes","child_log_ram_bytes","max_cpu_rss_bytes","max_cuda_reserved_bytes","operational_deadline_seconds"):
  require(type(caps.get(key)) is int and caps[key]>0,"Positive actual operational caps; inactive nulls cannot bind")
 require(caps["max_cpu_rss_bytes"]<=512000000000 and caps["max_cuda_reserved_bytes"]<=50000000000
  and caps["operational_deadline_seconds"]<=86400
  and caps["phase_write_budget_bytes"]+caps["temp_allowance_bytes"]+caps["outer_auxiliary_cap_bytes"]+caps["root_packet_overhead_cap_bytes"]<=caps["reservation_cap_bytes"],"Bounded root totals")
 for path,expected in PREFETCH_SOURCE_PINS.items():require(ref(path)["sha256"]==expected,"Frozen reviewed prefetch source/transport proof")
 if stage=="train":
  p=caps.get("observation_prefetch")
  require(isinstance(p,dict) and set(p)==PREFETCH_CAP_KEYS and type(p["num_workers"]) is int and p["num_workers"] in (2,4),"Exactly six explicit caps and2or4 workers")
  for k in ("max_block_bytes","worker_rss_cap_bytes","aggregate_rss_cap_bytes"):require(type(p[k]) is int and p[k]>0,"Positive actual byte caps")
  for k,limit in (("task_timeout_seconds",3600),("drain_timeout_seconds",30)):
   require(type(p[k]) in (int,float) and 0<p[k]<=limit,"Bounded explicit prefetch deadline")
  cfg=read(req["scientific_training_config_ref"]);base=read(FROZEN_TRAIN_CONFIG_REF)
  require(set(cfg)==set(base) and cfg["execution_enabled"] is False
   and all(cfg[k]==base[k] for k in base if k!="CPU_observation_cache_bytes"),"Exact20parent/30epoch/600update FP32 science; cache alone operational")
  cache=cfg["CPU_observation_cache_bytes"]
  require(type(cache) is int and cache>=p["max_block_bytes"] and p["worker_rss_cap_bytes"]>p["max_block_bytes"]
   and cache+(p["num_workers"]+2)*p["max_block_bytes"]+p["num_workers"]*p["worker_rss_cap_bytes"]<=p["aggregate_rss_cap_bytes"]<=caps["max_cpu_rss_bytes"],"Cache/transport/all-worker aggregate RSS bounds")
  for record in req["bindings"]["parent_plan_refs"].values():
   plan=read(record,cap=33554432)
   require(p["max_block_bytes"]>=plan["feature_tensor_byte_planning_bound_per_block"],"Block cap covers every original actual plan")
 else:require(caps.get("observation_prefetch") is None,"Prefetch belongs only to TRAIN; VAL route unchanged")
 return caps

def make_config(req,caps):
 stage=req["phase"]
 return dict(schema="C3_GNN_prefetch_isolated980_operational_config_v1",execution_enabled=True,phase=stage,
  output_cap_bytes=caps["reservation_cap_bytes"],short_tmp_directory=".runtime/tmp/c3gnnprefetchisolated980/"+stage,
  scientific_training_config_ref=req.get("scientific_training_config_ref"),
  observation_prefetch=copy.deepcopy(caps.get("observation_prefetch")),
  child_log_ram_bytes=caps["child_log_ram_bytes"])

def validate_request(record):
 req=read(record,cap=33554432);stage=req["phase"]
 require(req.get("schema")=="C3_GNN_prefetch_isolated980_request_v1" and req.get("active") is True
  and req.get("root_authorizer")=="root_agent" and stage in PHASES and req.get("native_MLM_dependency") is False,"Fresh explicit successor GNN authority")
 require(req["runtime_ref"]==RUNTIME_REF and req["base_fit_config_ref"]==BASE_CONFIG_REF
  and req["scientific_train_source_ref"]==TRAIN_SOURCE_REF and req["hook_ref"]==HOOK_REF,"Exact copied science/hooks/runtime")
 require(ref(TRAIN_SOURCE_REF["path"])==TRAIN_SOURCE_REF and ref(HOOK_REF["path"])==HOOK_REF
  and ref(LAUNCHER_REF["path"])==LAUNCHER_REF,"Actual source and unchanged GPU45 launcher pins")
 if stage in GPU_PHASES:
  verify_shim();require(req["physical_gpu"] in (4,5) and isinstance(req["GPU_UUID"],str) and req["GPU_UUID"].startswith("GPU-"),"Root-selected actual GPU45 UUID")
 else:require(req["physical_gpu"] is None and req["GPU_UUID"] is None,"CPU-only stage")
 validate_bindings(stage,req["bindings"])
 caps=read(req["caps_ref"]);validate_caps(req,caps,activated=True)
 require(caps["phase"]==stage,"Bound exact stage")
 return req,caps,read(RUNTIME_REF)

def bind(template_ref):
 req=read(template_ref,cap=33554432);stage=req["phase"]
 require(req.get("schema")=="C3_GNN_prefetch_isolated980_request_v1" and req.get("active") is False
  and req.get("root_authorizer")=="root_agent" and stage in PHASES,"Inactive explicit root successor request")
 require(req["runtime_ref"]==RUNTIME_REF and req["base_fit_config_ref"]==BASE_CONFIG_REF
  and req["scientific_train_source_ref"]==TRAIN_SOURCE_REF and req["hook_ref"]==HOOK_REF
  and req["native_MLM_dependency"] is False,"Exact copied sources/runtime")
 if stage in GPU_PHASES:verify_shim()
 validate_bindings(stage,req["bindings"])
 caps=read(req["caps_template_ref"]);validate_caps(req,caps,activated=False)
 folder=ROOT/REL;require(not (folder/("request_"+stage+".root_active_v1.json")).exists(),"No bind replay")
 caps=copy.deepcopy(caps);caps.update(active=True,phase=stage)
 caps_ref=save(folder,"caps_"+stage+".root_active_v1.json",caps)
 req.update(active=True,caps_ref=caps_ref)
 request_ref=save(folder,"request_"+stage+".root_active_v1.json",req)
 validate_request(request_ref)
 return dict(request_ref=request_ref,actual_admission=False,actual_execution=False)

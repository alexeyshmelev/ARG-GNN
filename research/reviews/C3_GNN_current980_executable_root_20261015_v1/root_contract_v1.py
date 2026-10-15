"""Current980 GNN root contract. Metadata only; all actions require explicit root modes."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_GNN_current980_executable_root_20261015_v1"
SOURCE=REL+"/root_contract_v1.py";DRIVER=REL+"/root_driver_v1.py";ENTRY=REL+"/worker_entry_v1.py"
CANONICAL="src/arggnn/american_c3_gnn_current980_exec_v1.py"
GPU_MODULE="arggnn.american_c3_gnn_current980_exec_v1"
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
HOOK_REF=dict(path="research/reviews/American_C3_PyG_execution_source_20261015_v1/execution_hooks_v1.py",sha256="6546cc4498a8bcbac4919ae08ce75b23a0ac8e5b350da25e3fa31a967286bfa1")
TRAIN_SOURCE_REF=dict(path="src/arggnn/american_c3_gnn_train_v1.py",sha256="cce52c9f1776e46c7005509544bfb3743acaf860f256e546e3ac5fc7cb518df2")
TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
GPU_PHASES=("train",)+tuple("predict_val_%03d"%i for i in range(5))
CPU_PHASES=("matrix","score")+tuple("plan_val_%03d"%i for i in range(5))
PHASES=GPU_PHASES+CPU_PHASES
PARENT="American_C3_short20_TRAIN_original5VAL"
CONTEXT=ENTRY;RUNTIME_HELPER=ENTRY
PROC_HELPER=_base.PROC_HELPER
STATUSES={p:"completed_C3_GNN_current980_"+p for p in PHASES}
def launcher_ref_v1():return LAUNCHER_REF
def phase_paths(stage):
 require(stage in PHASES,"Registered independent GNN stage")
 return ROOT/REL/("admitted_"+stage+"_v1"),"experiments/results/C3_GNN_current980_"+stage+"_v1"
def verify_shim():
 target=dict(path=CANONICAL,sha256=ref(REL+"/canonical_dispatch.staged.py")["sha256"])
 require(ref(CANONICAL)==target,"Root must first integrate the exact additive canonical shim")
 return target
def hooks():
 return load(HOOK_REF,"_C3_GNN_existing_execution_hooks")
def source_paths():
 scientific=("american_c3_common_v1","american_c3_gnn_targets_v1","american_c3_gnn_stream_train_v1","american_c3_gnn_train_v1",
 "american_c3_gnn_model_v1","american_c3_gnn_run_v1","american_c3_gnn_dataset_v1","american_c3_gnn_observations_v1",
 "american_c3_evaluate_v1","mutation_routed_whole_tree_gnn_v1","mutation_routed_tree_observations_v1","mutation_routed_pyg_gnn_v1",
 "american_c3_gnn_predict_v1","american_c3_gnn_selection_v1","american_c3_gnn_shared30mb_v1",
 "native_argformer_exec","biological_candidate","development_execution","census_tsinfer_fit_v1","project_budget_v9_consumer","resources")
 cfg=read(BASE_CONFIG_REF)
 return tuple(dict.fromkeys((*(("src/arggnn/"+x+".py") for x in scientific),
  SOURCE,DRIVER,ENTRY,CANONICAL,HOOK_REF["path"],BASE_HELPER["path"],LAUNCHER_REF["path"],
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
  require(set(b)=={"parent_id","plan_ref","native_preparation_ref","preparation_wait_ref","training"},"Truth-free single original VAL capability")
  parent=VAL[int(stage[-3:])];require(b["parent_id"]==parent,"Fixed original VAL stage identity")
  validate_preparation_wait(b["native_preparation_ref"],b["preparation_wait_ref"])
  validate_plan_metadata(b["plan_ref"],b["native_preparation_ref"],parent,30000000)
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
def make_config(req,caps):
 stage=req["phase"]
 return dict(schema="C3_GNN_current980_operational_config_v1",execution_enabled=True,phase=stage,
  output_cap_bytes=caps["reservation_cap_bytes"],short_tmp_directory=".runtime/tmp/c3gnn980/"+stage,
  scientific_training_config_ref=ref(REL+"/training_config.inactive.json"),
  child_log_ram_bytes=caps["child_log_ram_bytes"])
def validate_request(record):
 req=read(record,cap=33554432);stage=req["phase"]
 require(req.get("schema")=="C3_GNN_current980_request_v1" and req.get("active") is True
  and req.get("root_authorizer")=="root_agent" and stage in PHASES and req.get("native_MLM_dependency") is False,"Fresh explicit GNN stage authority")
 require(req["runtime_ref"]==RUNTIME_REF and req["base_fit_config_ref"]==BASE_CONFIG_REF
  and req["scientific_train_source_ref"]==TRAIN_SOURCE_REF and req["hook_ref"]==HOOK_REF,"Existing exact runtime/science sources")
 require(ref(TRAIN_SOURCE_REF["path"])==TRAIN_SOURCE_REF and ref(LAUNCHER_REF["path"])==LAUNCHER_REF,"Exact integrated science and original GPU45 launcher")
 if stage in GPU_PHASES:
  verify_shim();require(req["physical_gpu"] in (4,5) and isinstance(req["GPU_UUID"],str) and req["GPU_UUID"].startswith("GPU-"),"Actual GPU45 UUID")
 else:require(req["physical_gpu"] is None and req["GPU_UUID"] is None,"CPU stage has no GPU capability")
 validate_bindings(stage,req["bindings"])
 caps=read(req["caps_ref"])
 for key in ("reservation_cap_bytes","phase_write_budget_bytes","temp_allowance_bytes","outer_auxiliary_cap_bytes",
  "root_packet_overhead_cap_bytes","outer_log_cap_bytes","child_log_ram_bytes","max_cpu_rss_bytes","max_cuda_reserved_bytes","operational_deadline_seconds"):
  require(type(caps.get(key)) is int and caps[key]>0,"Positive actual operational caps")
 require(caps["phase"]==stage and caps["active"] is True and caps["bound_method"]=="operational_fail_stop_not_measured_fit_or_timing"
  and caps["max_cpu_rss_bytes"]==512000000000 and caps["max_cuda_reserved_bytes"]==50000000000
  and 0<caps["operational_deadline_seconds"]<=86400
  and caps["phase_write_budget_bytes"]+caps["temp_allowance_bytes"]+caps["outer_auxiliary_cap_bytes"]+caps["root_packet_overhead_cap_bytes"]<=caps["reservation_cap_bytes"],"Bounded prospective root caps")
 return req,caps,read(RUNTIME_REF)
def bind(template_ref):
 req=read(template_ref,cap=33554432);stage=req["phase"]
 require(req.get("active") is False and stage in PHASES,"Inactive reviewed request template")
 require(req["runtime_ref"]==RUNTIME_REF and req["base_fit_config_ref"]==BASE_CONFIG_REF
  and req["scientific_train_source_ref"]==TRAIN_SOURCE_REF and req["hook_ref"]==HOOK_REF
  and req["native_MLM_dependency"] is False,"Exact existing GNN science/runtime before any metadata write")
 if stage in GPU_PHASES:verify_shim()
 validate_bindings(stage,req["bindings"])
 folder=ROOT/REL;require(not (folder/("request_"+stage+".root_active_v1.json")).exists(),"No bind replay")
 caps=read(ref(REL+"/caps.inactive.json"));caps.update(active=True,phase=stage)
 caps_ref=save(folder,"caps_"+stage+".root_active_v1.json",caps)
 req.update(active=True,caps_ref=caps_ref)
 request_ref=save(folder,"request_"+stage+".root_active_v1.json",req)
 validate_request(request_ref)
 return dict(request_ref=request_ref,actual_admission=False,actual_execution=False)

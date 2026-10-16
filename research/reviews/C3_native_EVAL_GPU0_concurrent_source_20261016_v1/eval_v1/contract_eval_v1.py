"""Source-only metadata contract for one unchanged full30Mb C3 VAL prediction."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v1/eval_v1"
SOURCE=HERE+"/contract_eval_v1.py"
ENTRY=HERE+"/entry_eval_v1.py"
ADMISSION=HERE+"/root_admit_eval_v1.py"
PINS=HERE+"/source_pins.v1.json"
BOUNDS=HERE+"/resource_caps.source_only.v1.json"
MODULE="C3_original_full30Mb_EVAL"
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))
EPOCHS=(5,10,15,20,25,30)
REGISTRY_REF={"path":"research/reviews/American_C3_short_train2mb_wave_source_v1/parent_registry.inactive.json","sha256":"c75c111bc48f94d164fe9d6dfe4f8796290b03d1093aca88602049ed47f80e37"}
GPU_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
HOOK_REF={"path":"research/reviews/American_C3_PyG_execution_source_20261015_v1/execution_hooks_v1.py","sha256":"6546cc4498a8bcbac4919ae08ce75b23a0ac8e5b350da25e3fa31a967286bfa1"}
GPU_GUARD_REF={"path":"research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py","sha256":"3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69"}
CAPACITY_REF={"path":"research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py","sha256":"ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635"}
POLICY_REF={"path":"research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json","sha256":"7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582"}
REPAIR_REF={"path":"research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json","sha256":"f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"}
RENEWAL_REF={"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py","sha256":"491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468"}
RENEWAL_CERT_REF={"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json","sha256":"66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7"}
RUNTIME_REF={"path":"research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json","sha256":"e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602"}
BASE_CONFIG_REF={"path":"configs/census_tsinfer_fit_v1.json","sha256":"7c2436d5479e68beb1a8f3edb04ca0ee729a23be35ca2d7893827c0db68e3b00"}

def require(ok,message):
    if not ok:raise ValueError(message)
def local(name):
    require(isinstance(name,str) and not Path(name).is_absolute() and ".." not in Path(name).parts,"Contained project-relative path")
    p=ROOT/name
    require(p.resolve()==p and p.is_relative_to(ROOT),"Canonical path without symlink traversal")
    return p
def ref(name):
    p=local(name)
    require(p.is_file() and not p.is_symlink() and p.suffix in (".py",".sh",".json",".md") and p.stat().st_size<=32*2**20,"Bounded source/metadata only")
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and Path(record["path"]).suffix==".json" and ref(record["path"])==record,"Exact JSON metadata ref")
    return json.loads(local(record["path"]).read_bytes())
def load(record,label):
    require(Path(record["path"]).suffix==".py" and ref(record["path"])==record,"Exact reviewed source")
    sp=importlib.util.spec_from_file_location(label,local(record["path"]))
    m=importlib.util.module_from_spec(sp);sys.modules[label]=m;sp.loader.exec_module(m);return m
def unique(rows):
    d={}
    for r in rows:
        require(ref(r["path"])==r and (r["path"] not in d or d[r["path"]]==r),"Unique exact source/metadata refs")
        d[r["path"]]=r
    require(len(d)<=64,"Original64 direct-consumer cap")
    return list(d.values())
def active(value):
    now=datetime.now(timezone.utc)
    start=datetime.fromisoformat(value["created_at_utc"]);end=datetime.fromisoformat(value["expires_at_utc"])
    require(start.tzinfo and end.tzinfo and start<=now<end and 0<(end-start).total_seconds()<=86400,"Current explicit at-most24h EVAL authority")
def terminal(completion_ref,wait_ref):
    done,wait=read(completion_ref),read(wait_ref)
    require(wait.get("completed_ref")==completion_ref and wait.get("returncode")==0 and wait.get("operational_failure") is None,"Genuine exact directwait0 completion")
    require(any(wait.get(k) is True for k in ("actual_direct_launcher_wait","actual_direct_worker_wait")) and any(wait.get(k) is True for k in ("all_owned_processes_and_groups_drained","owned_worker_group_ESRCH","owned_process_group_ESRCH","owned_group_drained","worker_process_group_drained")),"Actual owned directwait and drain")
    return done
def training_v1(binding):
    require(set(binding)=={"completion_ref","actual_wait_ref","training_completion_ref","training_completion_locator"},"Truth-free genuine TRAIN metadata binding")
    done=terminal(binding["completion_ref"],binding["actual_wait_ref"])
    require(done.get("phase",done.get("stage"))=="train" and str(done.get("status","")).startswith("completed"),"Actual completed TRAIN envelope")
    value=done;keys=binding["training_completion_locator"]
    require(isinstance(keys,list) and len(keys)<=4 and all(isinstance(x,str) and x for x in keys),"Explicit bounded result-ref locator")
    for key in keys:value=value[key]
    require(value==binding["training_completion_ref"],"Inner primary completion belongs to actual TRAIN envelope")
    t=read(binding["training_completion_ref"])
    require(t.get("status")=="completed_C3_checkpointed_complete_parent_training" and t.get("schema")=="american_c3_checkpointed_complete_parent_training_v1" and t.get("registry_ref")==REGISTRY_REF and t.get("trained_parent_ids")==list(TRAIN) and t.get("train_sequence_lengths_bp")==[2000000]*20 and t.get("epochs")==30 and t.get("complete_parent_updates")==600 and t.get("focal_exposures")==6000000 and t.get("VAL_used_for_optimization") is False and t.get("H043_extension") is None and t.get("purity_filter") is False,"Genuine completed fixed primary TRAIN")
    candidates=t.get("checkpoint_candidates",[])
    require([x["epoch"] for x in candidates]==list(EPOCHS) and all(x["global_step"]==20*x["epoch"] for x in candidates) and len({x["checkpoint_ref"]["path"] for x in candidates})==6,"All six frozen scheduled candidates; checkpoint payloads untouched by metadata contract")
    return t
def parent_v1(binding):
    require(set(binding)=={"parent_id","plan_ref","native_preparation_ref","preparation_wait_ref","plan_completion_ref","plan_actual_wait_ref","plan_close_receipt_ref"},"Only original observed parent/preparation/closed-plan metadata")
    pid=binding["parent_id"];require(pid in VAL,"Fixed five original VAL parents")
    prep=terminal(binding["native_preparation_ref"],binding["preparation_wait_ref"])
    p=read(binding["plan_ref"]);cfg=p["config"]
    require(prep.get("status")=="completed_native_C3_path_preparation" and prep.get("parent_id")==pid and prep.get("query_ancestry_truth_opened") is False and prep.get("query_focal_count")==10000,"Completed same-parent native focals without query truth")
    obs=read(prep["observation_ref"])
    require(obs.get("parent_id")==pid and obs.get("split")=="val" and obs.get("sequence_length_bp")==30000000 and obs.get("query_ancestry_truth_in_inputs") is False and obs.get("class_order")==["EUR","EAS","AFR"] and "inferred_tsinfer_tsdate" in obs.get("genealogy_kind",""),"Original sanitized inferred full30Mb observation")
    require(p.get("schema")=="american_c3_whole_tree_block_index_v1" and p.get("parent_id")==pid and p.get("sequence_length_bp")==30000000 and p.get("haplotypes")==1600 and p.get("focal_records")==10000 and p.get("class_order")==["EUR","EAS","AFR"] and p.get("all_trees_and_samples_retained") is True and p.get("targets_opened") is False and p.get("genomic_context")=="two_pass_full_parent_state_carry" and cfg["native_preparation_ref"]==binding["native_preparation_ref"] and p["observation_ref"]==cfg["observation_ref"]==prep["observation_ref"] and p["focal_registry_ref"]==cfg["focal_registry_ref"]==prep["focals_ref"],"Exact original full observation/native focal/plan joins")
    require(all(cfg.get(k)==v for k,v in dict(max_nodes_per_block=250000,max_trees_per_block=32,max_sites_per_block=4096,expected_haplotypes=1600,reference_counts=[200,200,200],input_reference_class_mask=None,input_reference_sample_indices=None,allow_missing=False).items()),"Unchanged full-roster capped block recipe")
    cursor=0;sites=0
    for b in p["blocks"]:
        require(b["start_tree"]==cursor and 0<b["stop_tree"]-cursor<=32 and 0<b["nodes"]<=250000 and 0<=b["sites"]<=4096,"Complete contiguous bounded block metadata")
        cursor=b["stop_tree"];sites+=b["sites"]
    require(cursor==p["trees"] and sites==p["sites"],"All original tree/site metadata retained")
    done=terminal(binding["plan_completion_ref"],binding["plan_actual_wait_ref"])
    require(done.get("stage")=="plan_val_"+pid[-3:] and done.get("status")=="completed_C3_GNN_current980_plan_val_"+pid[-3:],"Actual observation-only plan stage")
    result=read(done["result_ref"])
    require(result.get("block_index_ref")==binding["plan_ref"] and result.get("parent_id")==pid and result.get("targets_opened") is False,"Actual plan-result ref")
    closed=read(binding["plan_close_receipt_ref"]);v=closed["value"];proof=v["proof"]
    require(closed.get("kind")=="concurrent_completed_scope_close_receipt_v1" and v.get("future_writes_and_relaunch_revoked") is True and proof.get("completion_verified") is True and proof.get("completed_ref")==binding["plan_completion_ref"] and proof.get("direct_wait_ref")==binding["plan_actual_wait_ref"],"Genuine closed plan remains immutable read-only")
    return p
def caps_v1(record):
    c=read(record)
    require(c.get("schema")=="C3_full30Mb_EVAL_caps_v1" and c.get("root_authorizer")=="root_agent" and c.get("active") is True and c.get("resource_profile_ref")==ref(BOUNDS),"Explicit actual EVAL caps")
    required=read(ref(BOUNDS))["fixed_caps"]
    require(all(c.get(k)==v for k,v in required.items()),"Exact count-derived EVAL resource profile")
    return c
def validate_request_v1(record):
    q=read(record);active(q)
    fields={"schema","status","root_authorizer","execution_authorized","GPU_authorized","physical_gpu","execution_gpu_uuid","native_MLM_dependency","main_source_ref","operational_GPU0_runtime_ref","hook_ref","runtime_ref","base_fit_config_ref","capacity_guard_ref","completed_overlay_policy_ref","operational_GPU0_guard_ref","parent","training","caps_ref","phase","parent_id","split","job_id","prior_controller_completion_ref","science_source_pins_ref","source_sha256","output_directory","packet_directory","short_tmp_directory","created_at_utc","expires_at_utc"}
    require(set(q)==fields,"Exact truth-free EVAL root request; no resolver or extra capability")
    require(q.get("schema")=="C3_full30Mb_EVAL_root_request_v1" and q.get("status")=="active" and q.get("root_authorizer")=="root_agent" and q.get("execution_authorized") is True and q.get("GPU_authorized") is True and q.get("physical_gpu")==0 and q.get("execution_gpu_uuid")==GPU_UUID and q.get("native_MLM_dependency") is False,"Fresh explicit GPU0-only EVAL root authority")
    require(q.get("main_source_ref")==ref(ENTRY) and q.get("operational_GPU0_runtime_ref")==ref(ENTRY) and q.get("hook_ref")==HOOK_REF and q.get("runtime_ref")==RUNTIME_REF and q.get("base_fit_config_ref")==BASE_CONFIG_REF and q.get("capacity_guard_ref")==CAPACITY_REF and q.get("completed_overlay_policy_ref")==POLICY_REF and q.get("operational_GPU0_guard_ref")==GPU_GUARD_REF,"Exact EVAL operational/scientific/runtime/source routes")
    p=parent_v1(q["parent"]);training_v1(q["training"]);caps=caps_v1(q["caps_ref"])
    require(q["phase"]=="eval" and q.get("parent_id")==p["parent_id"] and q.get("split")=="val" and q["job_id"]=="American-C3-full30Mb-EVAL-"+p["parent_id"]+"-v1","One fixed original parent per job")
    require(q.get("prior_controller_completion_ref") is not None,"Actual completed prior controller bound for fresh successor")
    pins=read(q["science_source_pins_ref"])
    require(q["science_source_pins_ref"]["path"]==PINS and pins["source_sha256"]==q["source_sha256"] and all(ref(k)["sha256"]==v for k,v in pins["source_sha256"].items()),"Complete exact frozen source closure before imports")
    require(q["output_directory"]=="experiments/results/C3_full30Mb_EVAL_GPU0_"+p["parent_id"]+"_v1" and q["packet_directory"]==HERE+"/admitted_"+p["parent_id"]+"_v1" and q["short_tmp_directory"]==".runtime/tmp/c3eval0/"+p["parent_id"],"Fresh exact isolated EVAL output/packet/TMP")
    return q,caps
def method_v1(input_ref):
    s=read(input_ref);q,c=validate_request_v1(s["root_stage_request_ref"]);active(s)
    require(s.get("schema")=="C3_full30Mb_EVAL_active_input_v1" and s.get("status")=="active" and s.get("root_authorizer")=="root_agent" and s.get("execution_authorized") is True and s.get("GPU_authorized") is True and s["job_id"]==q["job_id"] and s["phase"]==q["phase"]=="eval" and s["parent_id"]==q["parent_id"] and s["split"]=="val" and s["max_cpu_rss_bytes"]==c["max_cpu_rss_bytes"] and s["max_cuda_reserved_bytes"]==c["max_cuda_reserved_bytes"] and s["output_directory"]==q["output_directory"] and s["created_at_utc"]==q["created_at_utc"] and s["expires_at_utc"]==q["expires_at_utc"] and s["source_sha256"]==q["source_sha256"] and s["main_source_ref"]==s["operational_GPU0_runtime_ref"]==ref(ENTRY) and s["caps_ref"]==q["caps_ref"] and s["physical_gpu"]==0 and s["execution_gpu_uuid"]==GPU_UUID and s["completed_overlay_policy_ref"]==POLICY_REF and s["operational_GPU0_guard_ref"]==GPU_GUARD_REF and s["root_driver_ref"]==ref(ADMISSION) and s["root_auxiliary_directory"]==q["packet_directory"]+"/worker_aux" and s["short_tmp_directory"]==q["short_tmp_directory"],"Actual admitted input exact request/source/caps identity")
    storage=read(s["refs"]["storage"]);cfg=read(s["refs"]["config"]);closure=read(s["transitive_source_closure_ref"])
    require(closure["root_stage_request_ref"]==s["root_stage_request_ref"] and closure["source_sha256"]==s["source_sha256"] and closure["parent"]==q["parent"] and closure["training"]==q["training"] and closure["native_MLM_dependency"] is False,"Exact scientific and metadata closure")
    require(storage.get("kind")=="project_budget_v9_storage_record" and storage.get("active") is True and storage.get("accounting_mode")=="prospective_980gb_H020_v3" and storage.get("consumer_uses_v9_guard") is True and storage["job_id"]==s["job_id"] and storage["output_directory"]==s["output_directory"] and storage["reserved_new_bytes"]==c["reservation_cap_bytes"] and storage["expires_at_utc"]==s["expires_at_utc"] and storage["completed_overlay_policy_ref"]==POLICY_REF and storage["budget_components"]["consumer_adapter"]==CAPACITY_REF,"Actual separately charged source-bound storage")
    require(cfg==dict(schema="C3_full30Mb_EVAL_operational_config_v1",output_cap_bytes=c["reservation_cap_bytes"],phase_write_budget_bytes=c["phase_write_budget_bytes"],short_tmp_directory=q["short_tmp_directory"]),"Exact operational config")
    needed=[ref(ENTRY),ref(SOURCE),ref(ADMISSION),q["science_source_pins_ref"],q["caps_ref"],s["root_stage_request_ref"],s["refs"]["config"],RUNTIME_REF,BASE_CONFIG_REF,s["transitive_source_closure_ref"],CAPACITY_REF,POLICY_REF,REPAIR_REF,RENEWAL_REF,RENEWAL_CERT_REF,GPU_GUARD_REF,ref("src/arggnn/biological_candidate.py"),ref("src/arggnn/census_tsinfer_fit_v1.py"),ref("src/arggnn/development_execution.py")]
    require(len(storage["reservation"]["consumer_bindings"])<=64 and all(r in storage["reservation"]["consumer_bindings"] for r in needed),"All EVAL direct consumers admitted within64")
    return dict(worker_kind="C3",job_id=s["job_id"],module=MODULE,module_ref=ref(ENTRY),runtime_entry_ref=ref(ENTRY),input_ref=input_ref,storage_ref=s["refs"]["storage"],cuda_caps_ref=s["caps_ref"],cuda_cap_bytes=c["max_cuda_reserved_bytes"],cpu_rss_cap_bytes=c["max_cpu_rss_bytes"],operational_deadline_seconds=c["operational_deadline_seconds"],margin_mib=2048,peak_mib=10240,log_limit_bytes=c["outer_log_cap_bytes"],env={},args=["--execute","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]])
def validate_method_v1(method):
    require(method_v1(method["input_ref"])==method,"Exact finite EVAL method, no input/argument substitution")
    spec=read(method["input_ref"])
    return spec,read(spec["caps_ref"]),read(spec["refs"]["storage"])
def runtime_refs_v1():
    """Direct roles only; source_pins authenticates the complete indirect closure."""
    return unique([ref(ENTRY),ref(SOURCE),ref(ADMISSION),ref(PINS),ref(BOUNDS),HOOK_REF,GPU_GUARD_REF,CAPACITY_REF,POLICY_REF,REPAIR_REF,RENEWAL_REF,RENEWAL_CERT_REF,RUNTIME_REF,BASE_CONFIG_REF,ref("src/arggnn/biological_candidate.py"),ref("src/arggnn/census_tsinfer_fit_v1.py"),ref("src/arggnn/development_execution.py")])
def capacity_guard_v1():
    load(RENEWAL_REF,"_C3_EVAL_original_marker_renewal").install_renewal_v1(RENEWAL_CERT_REF)
    guard=load(CAPACITY_REF,"_C3_EVAL_actual_H020v3_capacity")
    guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    return guard

"""H050 metadata boundary and count planning. Import does not load a guard or CUDA."""
from pathlib import Path
import hashlib,json,math
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1.root_common_v1 import (
    ROOT,require,ref,read,save,unique,cap_v1,policy_env,load_guard_v1,
    GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF,CORE,BASE)
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1 import producer_v1 as worker
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.framework_reuse_v1 import authenticate_actual_cut_preparation_v1
from research.reviews.H049_order2_cut_memory_gpu_exec_source_20261015_v2 import context_factory_v2 as framework
from research.reviews.order2_cut_memory_implementation_20261015_v1 import integration_hooks_v1 as cut_hooks
HERE="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1"
SOURCE=HERE+"/bridge_common_v1.py"
DRIVER=HERE+"/root_driver_v1.py"
CONTEXT=HERE+"/context_factory_v1.py"
BINDER=HERE+"/bind_root_packet_v1.py"
INTEGRATION="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1"
REVIEWED_DRIVER_REF=dict(path="research/reviews/H049_order2_cut_memory_gpu_exec_source_20261015_v2/root_driver.py",sha256="ab755f61929b87ce80e41483429658a641b31813154220b910a7fbe8cbfc2507")
H045_CONFIG_REF=dict(path="research/reviews/H045_GPU45_episode_root_source_v1/config.json",sha256="9e778ec9b45a01d4cb233edb9586d3d44e24b998bcf191da18bbfb371ffcd0cf")
EXTRA_CAPS=("RAM_prediction_cap_bytes","episode_preparation_cap_bytes",
    "episode_python_scratch_allowance_bytes","grid_cache_cap_bytes","GPU_grid_cap_bytes",
    "grid_preparation_cap_bytes","root_auxiliary_metadata_cap_bytes")
PREP_FIELDS=framework.PREPARATION_METADATA_FIELDS_V2
TERMINAL_FIELDS=("preparation_completion_ref","preparation_actual_wait_ref")+PREP_FIELDS
TRAIN_FIELDS=("train_completion_ref","train_launch_ref","train_actual_wait_ref")
LEGACY_RECEIPT_SCHEMAS=("H049_marker_original_launch_v1","H049_marker_original_direct_wait_v1")

class MetadataReaderV1:
    root=ROOT
    @staticmethod
    def read(record):
        require(ref(record["path"])==record,"Exact pinned metadata/source bytes")
        return (ROOT/record["path"]).read_bytes()
    @staticmethod
    def decode_json(record):return json.loads(MetadataReaderV1.read(record))


def is_ref_v1(value):
    return (isinstance(value,dict) and set(value)=={"path","sha256"}
        and isinstance(value["path"],str) and isinstance(value["sha256"],str)
        and len(value["sha256"])==64 and all(x in "0123456789abcdef" for x in value["sha256"]))


def validate_config_v1(cfg,reader):
    worker.validate_protocol(cfg);worker.bind_registration(cfg,reader)
    require(cfg.get("metrics_source_config_ref")==H045_CONFIG_REF
        and cfg.get("metrics")==reader.decode_json(H045_CONFIG_REF)["metrics"],"Unchanged original H045 Present182 and seven-cohort external metric settings")
    require(all(type(cfg.get(k)) is int and cfg[k]>0 for k in EXTRA_CAPS),"Explicit H050 bridge resource caps")
    require(cfg.get("canonical_dispatcher_ref")==ref(worker.SOURCE)
        and cfg.get("root_driver_ref")==ref(DRIVER)
        and cfg.get("root_context_module_ref")==ref(CONTEXT),"Actual H050 producer/driver/context config")
    require(is_ref_v1(cfg.get("cut_preparation_registration_ref"))
        and cfg["cut_preparation_registration_ref"]!=cfg["registration_ref"],"Distinct original preparation and H050 experiment registrations")
    require(cfg.get("expected_future_preparation_output")=="experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v5"
        and cfg.get("future_completion_not_asserted") is False,"Only genuinely completed corrected v5 preparation")
    require(all(is_ref_v1(cfg.get(k)) for k in TERMINAL_FIELDS),"All actual immutable preparation terminal and admission refs")
    return cfg


def count_plan_v1(counts,phase,preparation,cfg):
    require(phase in ("train","evaluate") and counts.get("schema")=="root_H043_original40_observed_count_manifest_v1"
        and counts.get("counts_only_no_labels_or_model_metrics") is True and counts.get("outer_included") is False
        and set(counts["parents"])==set(worker.TRAIN+worker.VAL)
        and set(preparation["parents"])==set(worker.TRAIN+worker.VAL),"Exact original40 count-only H050 plan")
    parents={}
    for pid in worker.TRAIN+worker.VAL:
        row=counts["parents"][pid];p=preparation["parents"][pid]["counts"]
        n,t,h=(row[k] for k in ("native_nodes","native_holds","haplotypes"))
        require(all(type(v) is int and v>0 for v in (n,t,h)) and h==952 and p["haplotypes"]==952
            and p["queries"]==686 and p["native_holds"]==t,"Original952/686 whole-parent counts")
        episode=18*n+4*t;cut=89*t*952+952
        conservative=cut_hooks.bounds_from_counts_v1(n,t,p["sites"],p["scoring_centers"])
        marker=preparation["parents"][pid]["marker_tensor_bytes"]
        require(type(marker) is int and marker>0,"Original conservative marker metadata bound")
        parents[pid]=dict(native_nodes=n,native_holds=t,haplotypes=h,
            episode_compact_payload_upper_bytes=episode,cut_extra_tensor_upper_bytes=cut,
            complete_episode_and_cut_bundle_upper_bytes=episode+cut,
            episode_named_preparation_allowance_bytes=80*n+16*t,
            original_conservative_grid_plus_marker_upper_bytes=conservative["extra_retained_tensor_upper_bytes"]+marker,
            original_conservative_GPU_grid_cut_upper_bytes=conservative["GPU_persistent_grid_upper_bytes"],
            original_conservative_preparation_allowance_bytes=conservative["named_preparation_allowance_bytes"],
            native_FP32_probabilities_or_target_bytes=t*686*4*4)
    ids=worker.TRAIN if phase=="train" else worker.VAL
    phase_rows={p:parents[p] for p in ids}
    all192=24*sum(parents[p]["native_FP32_probabilities_or_target_bytes"] for p in worker.VAL)
    cp=(24*(cfg["checkpoint_file_cap_bytes"]+8192) if phase=="train"
        and type(cfg.get("checkpoint_file_cap_bytes")) is int and cfg["checkpoint_file_cap_bytes"]>0
        else (None if phase=="train" else 0))
    log=cfg.get("log_cap_bytes");serialized=(6*log+65536+16384 if type(log) is int and log>0 else None)
    overhead=[cfg.get(k) for k in ("fixed_metadata_cap_bytes","qualification_allowance_bytes","root_auxiliary_metadata_cap_bytes")]+[serialized]
    total=cp+sum(overhead) if cp is not None and all(type(v) is int and v>0 for v in overhead) else None
    return dict(schema="H050_native_hold_count_plan_v1",phase=phase,parents=phase_rows,parent_count=len(ids),
        arms=list(worker.ARMS),scheduled_checkpoint_files=24,VAL_probability_seals_before_truth=192,
        all192_native_hold_probability_bytes=all192,all_dense_probabilities_RAM_only=True,
        maximum_complete_episode_and_cut_bundle_upper_bytes=max(v["complete_episode_and_cut_bundle_upper_bytes"] for v in phase_rows.values()),
        maximum_episode_payload_upper_bytes=max(v["episode_compact_payload_upper_bytes"] for v in phase_rows.values()),
        maximum_episode_named_preparation_allowance_bytes=max(v["episode_named_preparation_allowance_bytes"] for v in phase_rows.values()),
        phase_native_FP32_target_bytes=sum(v["native_FP32_probabilities_or_target_bytes"] for v in phase_rows.values()),
        maximum_original_conservative_grid_plus_marker_upper_bytes=max(v["original_conservative_grid_plus_marker_upper_bytes"] for v in phase_rows.values()),
        maximum_original_conservative_GPU_grid_cut_upper_bytes=max(v["original_conservative_GPU_grid_cut_upper_bytes"] for v in phase_rows.values()),
        maximum_original_conservative_preparation_allowance_bytes=max(v["original_conservative_preparation_allowance_bytes"] for v in phase_rows.values()),
        checkpoint_files_written=24 if phase=="train" else 0,checkpoint_archive_allowance_bytes=cp,
        parameters_per_arm=worker.model.PARAMETERS,serialized_root_log_and_wait_planning_bytes=serialized,
        fixed_metadata_allowance_bytes=cfg.get("fixed_metadata_cap_bytes"),
        qualification_allowance_bytes=cfg.get("qualification_allowance_bytes"),
        root_auxiliary_metadata_allowance_bytes=cfg.get("root_auxiliary_metadata_cap_bytes"),
        total_output_scope_upper_bound_bytes=total,output_cap_bytes=cfg.get("output_cap_bytes"),
        Python_scratch_allocator_autograd_and_real_peak_unqualified=True,
        fixed_metadata_and_root_auxiliary_allowances_require_root_review=True)


def validate_plan_caps_v1(plan,cfg,reserved_bytes=None):
    require(plan["scheduled_checkpoint_files"]==24 and plan["VAL_probability_seals_before_truth"]==192
        and len(plan["arms"])==4,"H0504arm24checkpoint192seal boundary")
    require(plan["total_output_scope_upper_bound_bytes"] is not None
        and plan["total_output_scope_upper_bound_bytes"]<=cfg["output_cap_bytes"],"Complete H050 durable output reservation forecast")
    if reserved_bytes is not None:
        require(type(reserved_bytes) is int and plan["total_output_scope_upper_bound_bytes"]<=reserved_bytes<=cfg["output_cap_bytes"],"Full output reservation covers all24 files plus reviewed metadata/log/auxiliary allowances")
    require(plan["all192_native_hold_probability_bytes"]<=cfg["RAM_prediction_cap_bytes"],"All192 native-hold FP32 RAM products fit reviewed cap")
    require(plan["maximum_episode_payload_upper_bytes"]<=min(cfg["episode_payload_cap_bytes"],cfg["episode_cache_cap_bytes"])
        and plan["maximum_complete_episode_and_cut_bundle_upper_bytes"]<=min(cfg["bundle_cache_cap_bytes"],cfg["GPU_bundle_cap_bytes"]),"Complete H045episode/H049cut bounds fit all cache and device caps")
    require(plan["maximum_episode_named_preparation_allowance_bytes"]+cfg["episode_python_scratch_allowance_bytes"]<=cfg["episode_preparation_cap_bytes"],"Root-reviewed prospective episode preparation allowance")
    require(plan["maximum_original_conservative_grid_plus_marker_upper_bytes"]<=cfg["grid_cache_cap_bytes"]
        and plan["maximum_original_conservative_GPU_grid_cut_upper_bytes"]<=cfg["GPU_grid_cap_bytes"]
        and plan["maximum_original_conservative_preparation_allowance_bytes"]<=cfg["grid_preparation_cap_bytes"],"Preserved original conservative grid/cut/marker prospective caps before admission")
    if plan["phase"]=="train":require(plan["phase_native_FP32_target_bytes"]<=cfg["CPU_target_cap_bytes"],"All32 native FP32 TRAIN target masses")


def exact_sources_v1(reader,cfg):
    manifest=reader.decode_json(cfg["source_manifest_ref"])
    require(manifest.get("schema")=="H050_root_bridge_exact_source_manifest_v1"
        and manifest.get("hypothesis_id")=="H050","Actual H050 full source manifest")
    sources=manifest["source_sha256"]
    for path,sha in sources.items():reader.read(dict(path=path,sha256=sha))
    for path in (SOURCE,DRIVER,CONTEXT,BINDER,worker.SOURCE):require(sources.get(path)==ref(path)["sha256"],"Exact new H050 bridge and scientific sources")
    require(sources.get(REVIEWED_DRIVER_REF["path"])==REVIEWED_DRIVER_REF["sha256"],"Unchanged original launch supervisor source")
    framework.validate_declared_cut_preparation_sources_v2(reader,cfg,sources)
    for r in manifest.get("metadata_refs",{}).values():reader.read(r)
    return sources


def terminal_v1(cfg):return {k:cfg[k] for k in TERMINAL_FIELDS}


def preparation_spec_v1(cfg,sources,original):
    return dict(source_sha256=sources,observation_entry_refs=original["observation_entry_refs"],
        observation_cache_source_refs=original["observation_cache_source_refs"],
        refs=dict(observation_cache=original["refs"]["observation_cache"]),
        **framework.declared_cut_preparation_source_refs_v2(cfg),**terminal_v1(cfg))


def validate_train_lineage_v1(reader,spec,cfg):
    require(spec["phase"]=="evaluate","H050 evaluation TRAIN lineage")
    train=reader.decode_json(spec["train_completion_ref"]);launch=reader.decode_json(spec["train_launch_ref"])
    wait=reader.decode_json(spec["train_actual_wait_ref"])
    require(train.get("schema")=="H050_four_arm_train_completion_v1" and train.get("hypothesis_id")=="H050"
        and train.get("status")=="completed_fixed30pass_four_arm_TRAIN" and train.get("arms")==list(worker.ARMS)
        and train.get("steps")=={a:960 for a in worker.ARMS} and train.get("VAL_truth_read") is False
        and train.get("outer_truth_read") is False and train.get("common_initial_predictor_verified") is True
        and train.get("scientific_protocol")==worker.hooks.fixed_protocol_v1()
        and train.get("parameters_per_arm")==worker.model.PARAMETERS,"Complete exact3840update H050 four-arm TRAIN")
    require(wait.get("schema")==LEGACY_RECEIPT_SCHEMAS[1] and launch.get("schema")==LEGACY_RECEIPT_SCHEMAS[0]
        and wait.get("actual_direct_launcher_wait") is True and wait.get("returncode")==0
        and all(wait.get(k) is None for k in ("error","logging_error","supervisor_error"))
        and wait.get("owned_group_drained") is True and launch.get("direct_child") is True
        and wait["completed_ref"]==spec["train_completion_ref"] and wait["launch_ref"]==spec["train_launch_ref"]
        and wait["launcher_pid"]==launch["launcher_pid"] and wait["launcher_start"]==launch["launcher_start"]
        and launch["input_ref"]==train["input_ref"] and launch["phase"]==wait["phase"]=="train"
        and launch["physical_gpu"]==wait["physical_gpu"] and launch["physical_gpu"] in (4,5),"Genuine same-identity original supervisor direct TRAIN wait0/drain")
    packet=HERE+"/admitted_train/"
    require(train["input_ref"]["path"]==packet+"launch_input.json"
        and spec["train_launch_ref"]["path"]==packet+"launch.json"
        and spec["train_actual_wait_ref"]["path"]==packet+"actual_wait.json","Only fresh H050 bridge TRAIN admission")
    old=reader.decode_json(train["input_ref"]);train_cfg=reader.decode_json(train["config_ref"])
    runtime=reader.decode_json(old["refs"]["runtime"])
    require(train["config_ref"]==old["refs"]["config"] and old["schema"]=="H050_direct_readout_root_input_v1"
        and old["hypothesis_id"]=="H050" and old["arm"]=="joint_four_arm" and old["phase"]=="train"
        and old["root_authorizer"]=="root_agent" and old["status"]=="active"
        and old["physical_gpu"]==launch["physical_gpu"] and old["device"]=="cuda"
        and old["output_directory"]=="experiments/results/H050_GPU%d_direct_readout_cut_memory_train_seed101_v1"%launch["physical_gpu"]
        and spec["train_completion_ref"]["path"]==old["output_directory"]+"/completed.json"
        and old["root_driver_ref"]==ref(DRIVER) and old["root_context_module_ref"]==ref(CONTEXT)
        and old["producer_ref"]==old["canonical_dispatcher_ref"]==train["producer_ref"]==cfg["producer_ref"]==ref(worker.SOURCE)
        and old["main_source_ref"]==old["worker_cli_ref"]==old["producer_ref"]
        and runtime["job_id"]==old["job_id"] and runtime["device"]=="cuda"
        and runtime["physical_gpu"]==old["physical_gpu"] and runtime["output_directory"]==old["output_directory"],"Exact TRAIN capability/config/runtime/output/scientific producer")
    validate_config_v1(train_cfg,reader)
    require({k:v for k,v in train_cfg.items() if k not in worker.RESOURCE_CAPS}
        =={k:v for k,v in cfg.items() if k not in worker.RESOURCE_CAPS},"Only original producer-approved resource caps may differ TRAIN/EVAL")
    require(old["source_sha256"]==exact_sources_v1(reader,train_cfg)
        and old["source_sha256"]==exact_sources_v1(reader,cfg),"Same exact TRAIN/EVAL source closure")
    for k in ("registration_ref","warmstart_checkpoint_ref","warmstart_completion_ref","cut_preparation_completion_ref","cut_preparation_actual_wait_ref"):
        require(train.get(k)==cfg[k],"Same genuine TRAIN/EVAL registration, warmstart and preparation")
    require(all(old.get(k)==cfg[k] for k in TERMINAL_FIELDS),"All TRAIN preparation metadata matches EVAL")
    history=train.get("history",[]);qualification=train.get("first_step_qualification",{})
    require(len(history)==30 and all(row["pass_index"]==i and row["steps"]=={a:32*i for a in worker.ARMS}
        and row["full_parent_updates_per_arm"]==32 and set(row["parent_order"])==set(worker.TRAIN)
        and len(row["parent_order"])==32 for i,row in enumerate(history,1))
        and set(qualification)==set(worker.ARMS),"All30 genuine complete-parent passes and4 first-scheduled-step qualifications")
    for arm,row in qualification.items():
        q=row.get("full_model_qualification",{});grad=q.get("gradient_checks",{});changed=q.get("updated_parameters",{})
        require(row.get("finite_FP32") is True and row.get("part_of_continuous_960_updates") is True
            and row.get("parent_id") in worker.TRAIN and q.get("parameters")==worker.model.PARAMETERS[arm]
            and q.get("actual_optimizer_step")==1 and q.get("no_extra_updates") is True
            and q.get("warmstart_checkpoint_ref")==cfg["warmstart_checkpoint_ref"]
            and grad and set(grad)==set(changed) and all(g.get("dtype")=="torch.float32" and g.get("finite") is True for g in grad.values())
            and any(v is True for k,v in changed.items() if not k.startswith("memory.")),"Actual finiteFP32 step1/full-model body update qualification")
    groups=train["checkpoint_groups"]
    require(set(groups)=={str(p) for p in worker.PASSES} and sum(map(len,groups.values()))==24,"Exactly24 scheduled H050 checkpoints")
    for at,arms in groups.items():
        require(set(arms)==set(worker.ARMS),"All4arms each checkpoint group")
        for arm,r in arms.items():require(is_ref_v1(r) and r["path"]==old["output_directory"]+"/"+arm+".pass_%02d.pt"%int(at),"Exact generated H050 checkpoint paths; payloads verified by producer")
    return dict(train_input_ref=train["input_ref"],train_config_ref=train["config_ref"],
        train_runtime_ref=old["refs"]["runtime"],train_checkpoint_groups=groups)


def direct_required_refs_v1(cfg,spec):
    refs=[spec[k] for k in ("main_source_ref","worker_cli_ref","root_context_module_ref","root_driver_ref","canonical_dispatcher_ref","consumer_manifest_ref")]
    refs += [cfg[k] for k in ("source_manifest_ref","count_manifest_ref","registration_ref","cut_preparation_registration_ref","base_fit_config","warmstart_checkpoint_ref","warmstart_completion_ref")]
    refs += [cfg[k] for k in TERMINAL_FIELDS]+list(framework.declared_cut_preparation_source_refs_v2(cfg).values())
    refs += [H045_CONFIG_REF,GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF,
        ref("research/reviews/H049_order2_cut_memory_root_source_20261015_v1/root_common_v1.py"),
        ref("src/arggnn/project_budget_v9_consumer.py"),
        ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py"),
        ref("scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py")]
    refs += [spec["refs"][k] for k in ("config","runtime")]
    if spec["phase"]=="evaluate":
        refs += [spec[k] for k in TRAIN_FIELDS]
        refs += [r for k,r in spec["train_io_lineage"].items() if k!="train_checkpoint_groups"]
    return unique(refs)


def validate_spec_v1(reader,spec,cfg,storage=None):
    validate_config_v1(cfg,reader)
    require(spec.get("schema")=="H050_direct_readout_root_input_v1" and spec.get("hypothesis_id")=="H050"
        and spec.get("arm")=="joint_four_arm" and spec.get("root_authorizer")=="root_agent"
        and spec.get("status")=="active" and spec.get("phase") in ("train","evaluate")
        and spec.get("device")=="cuda" and spec.get("physical_gpu") in (4,5)
        and spec.get("fit_authorized") is True and spec.get("pyg_authorized") is True,"Active actual H050 GPU4/5 four-arm capability")
    require(spec["root_context_module_ref"]==cfg["root_context_module_ref"]==ref(CONTEXT)
        and spec["root_driver_ref"]==cfg["root_driver_ref"]==ref(DRIVER)
        and spec["canonical_dispatcher_ref"]==cfg["canonical_dispatcher_ref"]==ref(worker.SOURCE)
        and spec["producer_ref"]==spec["main_source_ref"]==spec["worker_cli_ref"]==cfg["producer_ref"]
        and spec["max_cpu_rss_bytes"]==cfg["max_cpu_rss_bytes"]
        and spec["max_cuda_reserved_bytes"]==cfg["max_cuda_reserved_bytes"],"Exact source/capability and reviewed CPU/CUDA limits")
    require(spec["source_sha256"]==exact_sources_v1(reader,cfg),"Exact full source closure")
    manifest=reader.decode_json(spec["consumer_manifest_ref"])
    require(manifest.get("schema")=="H050_exact_transitive_consumer_closure_v1"
        and manifest.get("hypothesis_id")=="H050" and manifest.get("phase")==spec["phase"]
        and manifest["source_sha256"]==spec["source_sha256"]
        and manifest["refs"]=={k:v for k,v in spec["refs"].items() if k!="storage"}
        and manifest["observation_entry_refs"]==spec["observation_entry_refs"]
        and manifest["observation_cache_source_refs"]==spec["observation_cache_source_refs"],"Fresh exact H050 consumer closure")
    for k,v in {**terminal_v1(cfg),**framework.declared_cut_preparation_source_refs_v2(cfg)}.items():
        require(manifest.get(k)==spec.get(k)==v,"Same immutable actual preparation references")
    if spec["phase"]=="evaluate":
        require(manifest.get("train_io_lineage")==spec.get("train_io_lineage")==validate_train_lineage_v1(reader,spec,cfg),"Exact24file genuine TRAIN capability lineage")
    else:require(all(spec.get(k) is None for k in TRAIN_FIELDS) and spec.get("train_io_lineage") is None,"Fresh TRAIN without prior completion")
    if storage is not None:
        bindings=storage["reservation"]["consumer_bindings"]
        require(len(bindings)<=64 and all(r in bindings for r in direct_required_refs_v1(cfg,spec)),"All exact direct/transitive consumers in original reservation; at most64")
    return spec

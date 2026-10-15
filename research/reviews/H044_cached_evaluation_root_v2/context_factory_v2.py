"""H044 immutable cached-evaluation-only admitted consumer context.

Root owns registration/reservation/launch and actual terminal evidence.
Memo2 is installed only here, inside the consumer.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,resource
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H044_cached_evaluation_root_v2/context_factory_v2.py"
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
LAUNCHER_REF=dict(path="scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py",sha256="d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
AUTH_REF=dict(path="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json",sha256="0231ae8edb05f8dd34ce127a0de0b97c1dd64fa461ad081e7a83bb92c558d26c")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
PREP_SOURCE="src/arggnn/prepare_marker_resolution_observations_v2.py"
WORKER_SOURCE="src/arggnn/evaluate_marker_resolution_cached_v2.py"

DRIVER_SOURCE="research/reviews/H044_cached_evaluation_root_v2/root_driver_v2.py"
PRODUCER_SOURCE="research/reviews/H044_cached_evaluation_source_v2/evaluate_cached_v2.py"
CACHE_SOURCE="src/arggnn/marker_resolution_input_cache_v2.py"
ORIGINAL_CONFIG_REF={'path': 'research/reviews/H044_marker_resolution_root_v1/train_config_root_20261015_v3.json', 'sha256': 'bd657391a65b26fdcff76b295710b4b4cbcf0650978e9f5091c65ff6e9183ec6'}
ORIGINAL_CONTEXT_REF={'path': 'research/reviews/H044_marker_resolution_root_v1/context_factory_v2.py', 'sha256': '245f9dcdff47efeacba6a3aa40377375466f9d7abe746b6bd492d021dfa0b2fe'}
ORIGINAL_DRIVER_REF={'path': 'research/reviews/H044_marker_resolution_root_v1/root_driver_v4.py', 'sha256': 'a458d8fb7606fc9d3a7aa679e1a7a0f4f7e84d85a74d7ce700a5de645d9d2da6'}
ORIGINAL_WORKER_REF={'path': 'src/arggnn/train_marker_resolution_readout_v1.py', 'sha256': '386814feb14a64310b8c51c2cfbaf681a59b0273f8be0f72ffde5a222eb08c3c'}
ORIGINAL_CORE_REF={'path': 'src/arggnn/marker_resolution_readout_v1.py', 'sha256': '27ecd30683024def2d408bf16250102b3252d0ae112687b9664fca086b141310'}
CACHE_POLICY_REF={'path': 'research/reviews/H044_cached_evaluation_source_v2/cache_policy_inactive.json', 'sha256': 'f12a48476cf618ec60b0e9f91fecff07a7482412bda8d554d31ec027620ae9ab'}
CACHE_SOURCE_REF={'path': 'src/arggnn/marker_resolution_input_cache_v2.py', 'sha256': 'a1b12a31b1890b84e9033da216ce107bb3ce6c47f687876487514f010e3b1987'}
PRODUCER_SOURCE_REF={'path': 'research/reviews/H044_cached_evaluation_source_v2/evaluate_cached_v2.py', 'sha256': '73a00a99de9f0ff75bfb49e863a6da84e1fa9e6d2e39a76913c3e7b4ceb0dde1'}
MEMORY_ENVELOPE_REF={'path': 'research/reviews/H044_cached_evaluation_root_v2/memory_envelope_v2.json', 'sha256': '036e3d932441a7db4ac1a1bf34edc16ef348235fdf298cee6419e8a9a6f47e72'}
CPU_PLANNING_CAP=256_000_000_000
OUTPUT_RESERVATION=100_000_000
VAL=tuple("ap_expanded_val_%03d"%i for i in range(8))

def require(ok,message):
    if not ok:raise ValueError(message)

def actual(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())

def module_from_ref_v1(name,ref,reader):
    reader.read(ref)
    spec=importlib.util.spec_from_file_location(name,ROOT/ref["path"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def validate_original_train_v2(reader,completion_ref,wait_ref,config_ref,cfg):
    require(config_ref==ORIGINAL_CONFIG_REF,"Original fixed H044 scientific config required")
    train=reader.decode_json(completion_ref);waited=reader.decode_json(wait_ref)
    arms=["localized","tree_pooled"];passes=[5,10,15,20,25,30]
    require(train["schema"]=="H044_marker_paired_train_completion_v1"
        and train["status"]=="completed_fixed30pass_paired_TRAIN" and train["hypothesis_id"]=="H044"
        and train["arms"]==arms and train["steps"]=={a:960 for a in arms}
        and train["config_ref"]==config_ref and train["core_ref"]==ORIGINAL_CORE_REF
        and train["VAL_truth_read"] is False and train["outer_truth_read"] is False
        and train["warmstart_checkpoint_ref"]==cfg["warmstart_checkpoint_ref"]
        and train["warmstart_completion_ref"]==cfg["warmstart_completion_ref"]
        and len(train["history"])==30 and set(train["checkpoint_groups"])=={str(p) for p in passes},
        "Original paired30pass/960-update completion required")
    require(all(row["pass_index"]==at and row["steps"]=={a:32*at for a in arms}
        and row["full_parent_updates_per_arm"]==32 for at,row in enumerate(train["history"],1))
        and all(set(train["checkpoint_groups"][str(p)])==set(arms) for p in passes),
        "Every original whole-parent pass and both checkpoint arms required")
    require(waited["schema"]=="H044_marker_original_direct_wait_v1" and waited["phase"]=="train"
        and waited["actual_direct_launcher_wait"] is True and waited["returncode"]==0
        and waited["completed_ref"]==completion_ref and waited["owned_group_drained"] is True
        and waited.get("error") is None,"Original genuine direct wait0 and complete owned drain required")
    launch=reader.decode_json(waited["launch_ref"]);original=reader.decode_json(train["input_ref"])
    require(launch["phase"]=="train" and launch["direct_child"] is True
        and launch["input_ref"]==train["input_ref"]
        and launch["launcher_pid"]==waited["launcher_pid"]
        and launch["launcher_start"]==waited["launcher_start"]
        and original["schema"]=="H044_marker_root_execution_input_v1" and original["phase"]=="train"
        and original["refs"]["config"]==config_ref
        and original["main_source_ref"]==ORIGINAL_WORKER_REF
        and original["root_context_module_ref"]==ORIGINAL_CONTEXT_REF
        and original["root_driver_ref"]==ORIGINAL_DRIVER_REF
        and completion_ref["path"]==original["output_directory"]+"/completed.json"
        and original["physical_gpu"] in (4,5)
        and launch["physical_gpu"]==waited["physical_gpu"]==original["physical_gpu"],
        "Original training launch/input/source identity and physical GPU must match direct terminal evidence")
    return train

def validate_cache_metadata_v2(reader,spec,cfg):
    require(spec["cache_policy_ref"]==CACHE_POLICY_REF and spec["cache_source_ref"]==CACHE_SOURCE_REF
        and spec["cache_producer_source_ref"]==PRODUCER_SOURCE_REF
        and spec["main_source_ref"]==actual(WORKER_SOURCE)
        and spec["root_driver_ref"]==actual(DRIVER_SOURCE),"Actual canonical entry/producer/cache/driver refs required")
    policy=reader.decode_json(spec["cache_policy_ref"])
    expected=dict(schema="H044_evaluation_immutable_input_cache_policy_v2",phase="evaluate",
        limits=dict(maximum_parents=8,maximum_feature_bytes=21_073_920_000),query_block=64,
        release_after_parent=True,targets_cached=False,learned_values_cached=False,original_feature_arithmetic=True)
    require(policy==expected,"Original immutable feature-cache policy required")
    require(spec["memory_envelope_ref"]==MEMORY_ENVELOPE_REF,"Pinned cache byte envelope required")
    memory=reader.decode_json(spec["memory_envelope_ref"])
    require(memory["schema"]=="H044_cached_evaluation_memory_envelope_v2"
        and memory["original_config_ref"]==ORIGINAL_CONFIG_REF
        and memory["CPU_planning_cap_bytes"]==spec["max_cpu_rss_bytes"]==CPU_PLANNING_CAP
        and memory["per_parent_cached_feature_upper_bytes"]==2_634_240_000
        and memory["producer_live_cached_parents"]==1
        and memory["cache_all8_hard_limit_bytes"]==policy["limits"]["maximum_feature_bytes"]
        and memory["retained_reservation_bytes"]==OUTPUT_RESERVATION
        and memory["named_planning_total_with_margin_bytes"]==sum(memory["named_allowances_bytes"].values())
        and memory["named_planning_total_with_margin_bytes"]<=CPU_PLANNING_CAP,
        "Explicit cache accounting within original256GB planning envelope required")
    return policy,memory

def validate_closure_v2(reader,spec,cfg,storage):
    require(spec["refs"]["config"]==ORIGINAL_CONFIG_REF
        and spec["refs"]["source_manifest"]==spec["source_manifest_ref"]
        and spec["refs"]["cache_policy"]==spec["cache_policy_ref"]
        and spec["refs"]["memory_envelope"]==spec["memory_envelope_ref"]
        and spec["frozen_source_refs"]==[ORIGINAL_CORE_REF,ORIGINAL_WORKER_REF],
        "Original config and exact successor ref aliases required")
    frozen=reader.decode_json(cfg["source_manifest_ref"])
    sources=reader.decode_json(spec["source_manifest_ref"])
    require(frozen["schema"]=="H044_marker_root_source_manifest_v1"
        and sources["schema"]=="H044_cached_evaluation_root_source_manifest_v2"
        and sources["frozen_source_manifest_ref"]==cfg["source_manifest_ref"]
        and sources["source_sha256"]==spec["source_sha256"]
        and all(sources["source_sha256"].get(p)==d for p,d in frozen["source_sha256"].items()),
        "Complete unchanged scientific closure plus actual successor sources required")
    for path,digest in sources["source_sha256"].items():reader.read(dict(path=path,sha256=digest))
    for path in (SOURCE,DRIVER_SOURCE,WORKER_SOURCE,PRODUCER_SOURCE,CACHE_SOURCE):
        require(sources["source_sha256"].get(path)==actual(path)["sha256"],"Actual successor source not pinned")
    closure=reader.decode_json(spec["consumer_manifest_ref"])
    require(closure["schema"]=="H044_cached_evaluation_exact_consumer_closure_v2"
        and closure["hypothesis_id"]=="H044" and closure["phase"]=="evaluate"
        and closure["source_manifest_ref"]==spec["source_manifest_ref"]
        and closure["frozen_source_manifest_ref"]==cfg["source_manifest_ref"]
        and closure["source_sha256"]==spec["source_sha256"]
        and closure["refs"]=={k:v for k,v in spec["refs"].items() if k!="storage"}
        and closure["observation_entry_refs"]==spec["observation_entry_refs"]
        and closure["observation_cache_source_refs"]==spec["observation_cache_source_refs"],
        "Exact new transitive source/data/consumer closure required")
    required=[spec["root_context_module_ref"],spec["root_driver_ref"],spec["source_manifest_ref"],
        cfg["source_manifest_ref"],spec["consumer_manifest_ref"],spec["main_source_ref"],
        spec["cache_producer_source_ref"],spec["cache_source_ref"],spec["cache_policy_ref"],
        spec["memory_envelope_ref"],ORIGINAL_WORKER_REF,ORIGINAL_CORE_REF,
        GUARD_REF,POLICY_REF,REPAIR_REF,MEMO_REF,V4_REF,LAUNCHER_REF,AUTH_REF]
    for key in ("preparation_completion_ref","preparation_actual_wait_ref","train_completion_ref","train_actual_wait_ref"):
        require(closure[key]==spec[key],"Original phase terminal evidence differs from new closure")
        required.append(spec[key])
    require(all(r in storage["reservation"]["consumer_bindings"] for r in required)
        and len(storage["reservation"]["consumer_bindings"])<=64,
        "Every actual worker/cache/producer/context/driver/manifest/terminal/authority ref directly reserved")

def completed_preparation_v1(reader,spec,cfg):
    from arggnn.prepare_marker_resolution_observations_v1 import PARENTS,COORDINATE_KEYS
    ref=spec["preparation_completion_ref"];value=reader.decode_json(ref)
    waited=reader.decode_json(spec["preparation_actual_wait_ref"])
    require(waited["actual_direct_launcher_wait"] is True and waited["returncode"]==0
        and waited["completed_ref"]==ref and waited["owned_group_drained"] is True,
        "Exact successful original CPU preparation wait and owned-group drain")
    require(value["schema"]=="H044_marker_observation_preparation_v1"
        and value["status"]=="completed_all40_observation_only_sidecars"
        and value["hypothesis_id"]=="H044" and set(value["parents"])==set(PARENTS)
        and value["coordinate_keys"]==list(COORDINATE_KEYS)
        and value["observation_cache_ref"]==spec["refs"]["observation_cache"]
        and value["entry_refs"]==spec["observation_entry_refs"]
        and value["observation_source_refs"]==spec["observation_cache_source_refs"]
        and value["observations_only"] is True and value["targets_read"] is False,
        "Exact all40 original observed preparation")
    return value

def build_context_v2(record,spec):
    from arggnn import census_tsinfer_fit_v1 as fit
    require(spec.get("schema")=="H044_cached_evaluation_root_input_v2"
        and spec.get("hypothesis_id")=="H044" and spec.get("root_authorizer")=="root_agent"
        and spec.get("status")=="active" and spec.get("phase")=="evaluate"
        and spec.get("evaluation_authorized") is True and spec.get("fit_authorized") is False,
        "Active exact root H044 phase capability")
    require(spec["root_context_module_ref"]==actual(SOURCE),"Actual context source changed")
    require(json.loads(os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])==POLICY_REF,"Exact current policy env")
    reader=fit.ConsumerReaderV1();fit._check_active_v1(spec)
    cfg=reader.decode_json(spec["refs"]["config"])
    require(cfg["schema"]=="H044_marker_execution_config_v1" and cfg["hypothesis_id"]=="H044","H044 root config")
    require(cfg["helper_refs"]["full_reservation_adapter"]==GUARD_REF
        and cfg["helper_refs"]["actual_full_storage_launcher"]==LAUNCHER_REF
        and cfg["completed_overlay_policy_ref"]==POLICY_REF
        and cfg["marker_repair_authority_ref"]==REPAIR_REF,"Current guard and unchanged authorities")
    storage=reader.decode_json(spec["refs"]["storage"])
    require(storage["completed_overlay_policy_ref"]==POLICY_REF,"Storage policy differs")
    validate_closure_v2(reader,spec,cfg,storage)
    cache_policy,memory_envelope=validate_cache_metadata_v2(reader,spec,cfg)
    validate_original_train_v2(reader,spec["train_completion_ref"],spec["train_actual_wait_ref"],spec["refs"]["config"],cfg)
    storage_guard=module_from_ref_v1("_H044_current_completed_guard",GUARD_REF,reader)
    memo=module_from_ref_v1("_H044_consumer_memo2",MEMO_REF,reader)
    reader.read(V4_REF)
    memo_handle=memo.install(storage_guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],
        expected_source_sha256=V4_REF["sha256"])
    storage_guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    source=WORKER_SOURCE
    storage_guard.validate_storage_v9(spec,cfg,storage,
        source_ref=dict(path=source,sha256=spec["source_sha256"][source]),
        config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
    from arggnn import train_anonymous_clade_flag_control_v1 as h42
    h42.science()
    output=storage_guard.PilotOutputV9(spec["output_directory"],storage,reader)
    output.save("runtime_memo_installation.json",memo_handle.receipt)
    runtime=reader.decode_json(spec["refs"]["runtime"])
    require(runtime["job_id"]==spec["job_id"],"Runtime logical job identity")
    generation=reader.decode_json(spec["refs"]["generation_config"])
    registration=reader.decode_json(spec["refs"]["registration"])
    require(registration["config_ref"]==spec["refs"]["generation_config"]
        and registration["parent_registry"]==generation["parent_registry"],"Original generation registration")
    from arggnn.prepare_marker_resolution_observations_v1 import PARENTS
    entries=spec["observation_entry_refs"]
    require(set(entries)==set(PARENTS) and spec["observation_cache_source_refs"]==cfg["observation_cache_source_refs"],
        "Original all40 observed entries and unchanged producer source refs")
    ctx=dict(root_authorizer="root_agent",admitted=True,phase=spec["phase"],arm="joint_pair",
        cfg=cfg,reader=reader,output=output,config_ref=spec["refs"]["config"],
        source_manifest_ref=spec["source_manifest_ref"],registry=generation["parent_registry"],
        entry_refs=entries,observation_cache_ref=spec["refs"]["observation_cache"],
        observation_source_refs=spec["observation_cache_source_refs"],runtime_memo_handle=memo_handle)
    from arggnn import train_marker_resolution_readout_v1 as worker
    worker.validate_protocol_v1(cfg)
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    lease=validate_lease();apply_and_validate_title();fit.require_logical_job_v1(lease,spec)
    require(spec["device"]=="cuda" and spec["physical_gpu"] in (4,5)
        and lease["physical_gpu"]==spec["physical_gpu"]
        and runtime["physical_gpu"]==spec["physical_gpu"]
        and runtime["process_title"]=="AI Auto Research Pipeline Test","Exact GPU4/5 lease/title/runtime")
    storage_guard.budget_check_v9(storage,boundary=True,required_write_bytes=cfg["qualification_allowance_bytes"])
    from arggnn.biological_candidate import configure_fit
    device,qualified_runtime,qualified_lease=configure_fit("cuda",reader.decode_json(cfg["base_fit_config"]),output.path)
    require(qualified_lease["physical_gpu"]==spec["physical_gpu"],"Qualified device is root-selected physical GPU")
    import torch,torch_geometric
    require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Pinned torch/PyG runtime")
    torch.cuda.set_per_process_memory_fraction(min(1.,spec["max_cuda_reserved_bytes"]/
        torch.cuda.get_device_properties(device).total_memory),device)
    # configure_fit writes its own bounded qualification files before workers save.
    output.used+=cfg["qualification_allowance_bytes"]
    from arggnn import train_mutation_routed_whole_tree_gnn_v1 as original
    guard=lambda:original.lightweight_resources_v1(spec,qualified_lease)
    guard()
    preparation=completed_preparation_v1(reader,spec,cfg)
    binding_metadata=reader.decode_json(spec["refs"]["target_bindings"]);bindings=binding_metadata["parents"]
    ids=worker.VAL
    require(set(bindings)==set(ids) and binding_metadata["outer_included"] is False,"Exact phase-only targets")
    for pid,b in bindings.items():
        require(b["truth_npz"]["path"]==generation["output_root"]+"/"+pid+"/truth/full_truth_intervals.npz",
            "Original same-parent truth path")
        provenance=reader.decode_json(b["generation_receipt"])
        require(provenance["parent_id"]==pid and provenance["files"]["truth/full_truth_intervals.npz"]["sha256"]
            ==b["truth_npz"]["sha256"],"Same-parent generation identity")
    ctx.update(device=device,guard=guard,bindings=bindings,preparation=preparation,
        preparation_completion_ref=spec["preparation_completion_ref"],
        preparation_actual_wait_ref=spec["preparation_actual_wait_ref"])
    if spec["phase"]=="evaluate":
        waited=reader.decode_json(spec["train_actual_wait_ref"])
        require(waited["actual_direct_launcher_wait"] is True and waited["returncode"]==0
            and waited["completed_ref"]==spec["train_completion_ref"] and waited["owned_group_drained"] is True,
            "Actual paired TRAIN wait and complete owned drain")
        ctx["train_completion_ref"]=spec["train_completion_ref"]
        ctx.update(train_actual_wait_ref=spec["train_actual_wait_ref"],cache_policy=cache_policy,
            cache_policy_ref=spec["cache_policy_ref"],memory_envelope_ref=spec["memory_envelope_ref"],
            cached_execution_ref=record)
    output.save("started.json",dict(schema="H044_marker_started_v1",hypothesis_id="H044",
        phase=spec["phase"],input_ref=record,physical_gpu=spec["physical_gpu"],
        runtime=qualified_runtime,outer_truth_read=False,same_VAL_is_development_evidence=True))
    return ctx

"""H044 admitted consumer context; CPU preparation or paired GPU4/5 work.

Root owns registration/reservation/launch and actual terminal evidence.
Memo2 is installed only here, inside the consumer.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,resource
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H044_marker_resolution_root_v1/context_factory_v1.py"
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
LAUNCHER_REF=dict(path="scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py",sha256="d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
AUTH_REF=dict(path="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json",sha256="0231ae8edb05f8dd34ce127a0de0b97c1dd64fa461ad081e7a83bb92c558d26c")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
PREP_SOURCE="src/arggnn/prepare_marker_resolution_observations_v1.py"
WORKER_SOURCE="src/arggnn/train_marker_resolution_readout_v1.py"

def require(ok,message):
    if not ok:raise ValueError(message)

def actual(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())

def module_from_ref_v1(name,ref,reader):
    reader.read(ref)
    spec=importlib.util.spec_from_file_location(name,ROOT/ref["path"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def validate_closure_v1(reader,spec,cfg,storage):
    sources=reader.decode_json(cfg["source_manifest_ref"])
    require(sources["schema"]=="H044_marker_root_source_manifest_v1"
        and sources["source_sha256"]==spec["source_sha256"],"Exact frozen H044 source closure")
    for path,digest in sources["source_sha256"].items():reader.read(dict(path=path,sha256=digest))
    closure=reader.decode_json(spec["consumer_manifest_ref"])
    require(closure["schema"]=="H044_marker_exact_consumer_closure_v1"
        and closure["hypothesis_id"]=="H044" and closure["phase"]==spec["phase"]
        and closure["source_manifest_ref"]==cfg["source_manifest_ref"]
        and closure["source_sha256"]==spec["source_sha256"]
        and closure["refs"]=={k:v for k,v in spec["refs"].items() if k!="storage"}
        and closure["observation_entry_refs"]==spec["observation_entry_refs"]
        and closure["observation_cache_source_refs"]==spec["observation_cache_source_refs"],
        "Exact transitive data/source consumer closure")
    required=[spec["root_context_module_ref"],spec["root_driver_ref"],cfg["source_manifest_ref"],
        spec["consumer_manifest_ref"],GUARD_REF,POLICY_REF,REPAIR_REF,MEMO_REF,V4_REF,
        dict(path=PREP_SOURCE if spec["phase"]=="prepare" else WORKER_SOURCE,
            sha256=spec["source_sha256"][PREP_SOURCE if spec["phase"]=="prepare" else WORKER_SOURCE])]
    if spec["phase"]!="prepare":required += [LAUNCHER_REF,AUTH_REF]
    for key in ("preparation_completion_ref","preparation_actual_wait_ref","train_completion_ref","train_actual_wait_ref"):
        if key in spec:
            require(closure[key]==spec[key],"Phase terminal evidence differs from closure")
            required.append(spec[key])
    require(all(r in storage["reservation"]["consumer_bindings"] for r in required),
        "Mandatory actual source/context/driver/guard/authority/memo/terminal refs directly reserved")
    require(len(storage["reservation"]["consumer_bindings"])<=64,"Guard direct-reference bound")
    # Guard itself enforces actual v9 marker/repair/capacity/runtime producer pins.

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

def build_context_v1(record,spec):
    from arggnn import census_tsinfer_fit_v1 as fit
    require(spec.get("schema")=="H044_marker_root_execution_input_v1"
        and spec.get("hypothesis_id")=="H044" and spec.get("root_authorizer")=="root_agent"
        and spec.get("status")=="active" and spec.get("phase") in ("prepare","train","evaluate"),
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
    validate_closure_v1(reader,spec,cfg,storage)
    storage_guard=module_from_ref_v1("_H044_current_completed_guard",GUARD_REF,reader)
    memo=module_from_ref_v1("_H044_consumer_memo2",MEMO_REF,reader)
    reader.read(V4_REF)
    memo_handle=memo.install(storage_guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],
        expected_source_sha256=V4_REF["sha256"])
    storage_guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    source=PREP_SOURCE if spec["phase"]=="prepare" else WORKER_SOURCE
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
        source_manifest_ref=cfg["source_manifest_ref"],registry=generation["parent_registry"],
        entry_refs=entries,observation_cache_ref=spec["refs"]["observation_cache"],
        observation_source_refs=spec["observation_cache_source_refs"],runtime_memo_handle=memo_handle)
    def cpu_guard():
        fit._check_active_v1(spec)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=spec["max_cpu_rss_bytes"],"Root CPU RSS cap")
    if spec["phase"]=="prepare":
        require(spec["device"]=="cpu" and spec.get("physical_gpu") is None
            and os.environ.get("CUDA_VISIBLE_DEVICES")==""
            and runtime["device"]=="cpu" and runtime.get("physical_gpu") is None,
            "CPU-only preparation with CUDA hidden")
        require("target_bindings" not in spec["refs"] and "bindings" not in spec,"No CPU target capability")
        for name in ("CPU_bank_cap_bytes","CPU_scratch_cap_bytes","marker_cache_cap_bytes"):
            require(type(cfg.get(name)) is int and cfg[name]>0,"Positive root-reviewed preparation cap: "+name)
        ctx.update(guard=cpu_guard,scratch_directory=str(output.path/"observed_scratch"),
            scratch_guard=lambda size:storage_guard.budget_check_v9(storage,boundary=True,required_write_bytes=size))
        cpu_guard()
        return ctx
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
    ids=worker.TRAIN if spec["phase"]=="train" else worker.VAL
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
    output.save("started.json",dict(schema="H044_marker_started_v1",hypothesis_id="H044",
        phase=spec["phase"],input_ref=record,physical_gpu=spec["physical_gpu"],
        runtime=qualified_runtime,outer_truth_read=False,same_VAL_is_development_evidence=True))
    return ctx

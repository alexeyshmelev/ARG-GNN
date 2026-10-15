"""H042 root flag context. Inert until root's admitted GPU4 worker calls it."""
from pathlib import Path
import hashlib,importlib.util,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H042_GPU45_frozen_flag_evaluation_source_v1/context_factory_v1.py"
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
LAUNCHER_REF=dict(path="scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py",sha256="d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
AUTH_REF=dict(path="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json",sha256="0231ae8edb05f8dd34ce127a0de0b97c1dd64fa461ad081e7a83bb92c558d26c")
PREDECESSOR_COMPLETION_REF={'path': 'experiments/results/H040_GPU45_resume_tensor_seed101_v1/completed.json', 'sha256': '3e2f656f8657edc722a3223f8a6827f7ba3e2f626f997c90966846baffe13d29'}
def require(ok,msg):
    if not ok:raise ValueError(msg)
def actual(path):
    return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def validate_closure_v1(reader,spec,cfg,storage):
    manifest=reader.decode_json(spec["consumer_manifest_ref"])
    require(manifest["schema"]=="H042_exact_transitive_consumer_closure_v1"
        and manifest["hypothesis_id"]=="H042" and manifest["phase"]==spec["phase"]
        and manifest["source_manifest_ref"]==cfg["source_manifest_ref"]
        and manifest["source_sha256"]==spec["source_sha256"]
        and manifest["refs"]=={k:v for k,v in spec["refs"].items() if k!="storage"}
        and manifest["observation_entry_refs"]==spec["observation_entry_refs"]
        and manifest["observation_cache_source_refs"]==spec["observation_cache_source_refs"],
        "Exact H042 source/data metadata transitive closure required")
    sources=reader.decode_json(cfg["source_manifest_ref"])
    require(sources["schema"]=="H042_clade_flags_frozen_source_manifest_v1"
        and sources["source_sha256"]==spec["source_sha256"],"Frozen full source manifest differs")
    for path,digest in sources["source_sha256"].items():reader.read(dict(path=path,sha256=digest))
    for r in sources["metadata_refs"].values():reader.read(r)
    require(manifest["GPU4_predecessor_launch_ref"]==spec["GPU4_predecessor_launch_ref"]
        and manifest["GPU4_predecessor_wait_ref"]==spec["GPU4_predecessor_wait_ref"],
        "GPU4 predecessor terminal evidence must remain pinned")
    launch=reader.decode_json(spec["GPU4_predecessor_launch_ref"])
    waited=reader.decode_json(spec["GPU4_predecessor_wait_ref"])
    require(launch["arm"]=="tensor" and launch["physical_gpu"]==4
        and waited["actual_direct_launcher_wait"] is True and waited["returncode"]==0
        and waited["completed_ref"]==PREDECESSOR_COMPLETION_REF
        and waited["launcher_pid"]==launch["launcher_pid"],"Successful exact H040 GPU4 tensor completion and actual wait required")
    required=[spec["root_context_module_ref"],spec["root_driver_ref"],cfg["source_manifest_ref"],
        spec["consumer_manifest_ref"],LAUNCHER_REF,GUARD_REF,POLICY_REF,AUTH_REF,REPAIR_REF]
    require(all(r in storage["reservation"]["consumer_bindings"] for r in required),
        "Actual context/driver/manifest/launcher/guard/policy authorization must be reserved")
    if spec["phase"]=="evaluate":
        require(manifest["train_completion_ref"]==spec["train_completion_ref"],"Singular same-arm train completion closure")
def build_context_v1(record,spec):
    from arggnn import H042_GPU45_frozen_flag_evaluation_v1 as successor
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    lease=validate_lease();apply_and_validate_title()
    require(lease["physical_gpu"]==spec["physical_gpu"] and spec["physical_gpu"] in (4,5),"Exact root-selected GPU4/5 evaluation lease required")
    from arggnn import train_anonymous_clade_flag_control_v1 as worker
    from arggnn import census_tsinfer_fit_v1 as fit
    from arggnn import train_mutation_routed_whole_tree_gnn_v1 as original
    require(spec["schema"]=="H042_frozen_flag_evaluation_input_v1"
        and spec["hypothesis_id"]=="H042" and spec["control_id"]==worker.CONTROL_ID
        and spec["arm"]==worker.ARM and spec["phase"]=="evaluate"
        and spec["root_authorizer"]=="root_agent" and spec["status"]=="active",
        "Distinct active H042 flag-control input required")
    require(spec["root_context_module_ref"]==actual(SOURCE),"Actual root context changed")
    require(json.loads(os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])==POLICY_REF,
        "Exact root policy env must reach launcher and worker")
    reader=fit.ConsumerReaderV1()
    fit._check_active_v1(spec);fit.require_logical_job_v1(lease,spec)
    refs=spec["refs"];cfg=reader.decode_json(refs["config"]);worker.validate_protocol(cfg)
    successor.validate_capability_v1(spec,reader,cfg)
    require(cfg["hypothesis_id"]=="H042" and cfg["base_fit_config"]["path"]=="configs/census_tsinfer_fit_v1.json",
        "H042 ID and original runtime base config required")
    require(spec["device"]=="cuda" and spec["max_cuda_reserved_bytes"]==50000000000,
        "Pinned shared FP32 CUDA cap required")
    runtime_metadata=reader.decode_json(refs["runtime"])
    require(runtime_metadata["job_id"]==spec["job_id"] and runtime_metadata["physical_gpu"]==spec["physical_gpu"]
        and runtime_metadata["process_title"]=="AI Auto Research Pipeline Test","Runtime/lease/title identity differs")
    require(cfg["helper_refs"]["full_reservation_adapter"]==GUARD_REF
        and cfg["helper_refs"]["actual_full_storage_launcher"]==LAUNCHER_REF
        and cfg["completed_overlay_policy_ref"]==POLICY_REF and cfg["marker_repair_authority_ref"]==REPAIR_REF,
        "Current exact credit-v3 guard and marker authority required")
    for r in (GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF):reader.read(r)
    storage=reader.decode_json(refs["storage"])
    require(storage["completed_overlay_policy_ref"]==POLICY_REF,"Storage policy differs from launcher/worker env")
    validate_closure_v1(reader,spec,cfg,storage)
    module_spec=importlib.util.spec_from_file_location("_H042_completed_credits_guard",ROOT/GUARD_REF["path"])
    storage_guard=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(storage_guard)
    require(spec["memo_helper_ref"]==successor.MEMO_REF,"Mandatory finalized memo helper ref required")
    reader.read(successor.MEMO_REF);reader.read(successor.V4_REF)
    memo_spec=importlib.util.spec_from_file_location("_H042_transaction_memo_v2",ROOT/successor.MEMO_REF["path"])
    memo_module=importlib.util.module_from_spec(memo_spec);memo_spec.loader.exec_module(memo_module)
    memo_handle=memo_module.install(storage_guard.budget.v4,
        expected_source_path=ROOT/successor.V4_REF["path"],expected_source_sha256=successor.V4_REF["sha256"])
    require(all(r in storage["reservation"]["consumer_bindings"] for r in (successor.MEMO_REF,successor.V4_REF,
        spec["train_completion_ref"],spec["train_launch_ref"],spec["train_actual_wait_ref"])),
        "Exact memo and actual TRAIN terminal evidence must be directly reserved")
    storage_guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    storage_guard.validate_storage_v9(spec,cfg,storage,
        source_ref=dict(path=successor.SOURCE,sha256=spec["source_sha256"][successor.SOURCE]),
        config_ref=refs["config"],runtime_ref=refs["runtime"])
    worker.science()
    output=storage_guard.PilotOutputV9(spec["output_directory"],storage,reader)
    output.save("runtime_memo_installation.json",memo_handle.receipt)
    storage_guard.budget_check_v9(storage,boundary=True,required_write_bytes=cfg["qualification_allowance_bytes"])
    from arggnn.biological_candidate import configure_fit
    device,runtime,qualified_lease=configure_fit("cuda",reader.decode_json(cfg["base_fit_config"]),output.path)
    require(qualified_lease["physical_gpu"]==spec["physical_gpu"],"Qualified device must remain root-selected GPU")
    import torch,torch_geometric
    require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Pinned torch/PyG runtime differs")
    torch.cuda.set_per_process_memory_fraction(min(1.,50000000000/torch.cuda.get_device_properties(device).total_memory),device)
    output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob("*") if p.is_file())
    require(output.used<=cfg["qualification_allowance_bytes"],"Qualification exceeds reserved bound")
    guard=lambda:original.lightweight_resources_v1(spec,qualified_lease)
    guard()
    generation=reader.decode_json(refs["generation_config"])
    registration=reader.decode_json(refs["registration"])
    require(registration["config_ref"]==refs["generation_config"] and registration["parent_registry"]==generation["parent_registry"],
        "Original generation registration differs")
    entries=spec["observation_entry_refs"];registry=generation["parent_registry"]
    require(set(entries)==set(worker.TRAIN+worker.VAL),"Original32TRAIN/eightVAL observations required")
    require(spec["observation_cache_source_refs"]==cfg["observation_cache_source_refs"],"Original observation source refs differ")
    binding_metadata=reader.decode_json(refs["target_bindings"]);bindings=binding_metadata["parents"]
    ids=worker.TRAIN if spec["phase"]=="train" else worker.VAL
    require(set(bindings)==set(ids) and binding_metadata["outer_included"] is False,"Exact phase-only targets, no outer")
    for pid,b in bindings.items():
        require(b["truth_npz"]["path"]==generation["output_root"]+"/"+pid+"/truth/full_truth_intervals.npz",
            "Same-parent truth path required")
        provenance=reader.decode_json(b["generation_receipt"])
        require(provenance["parent_id"]==pid and provenance["files"]["truth/full_truth_intervals.npz"]["sha256"]==b["truth_npz"]["sha256"],
            "Same-parent generation identity differs")
    output.save("started.json",dict(hypothesis_id="H042",control_id=worker.CONTROL_ID,phase=spec["phase"],arm=worker.ARM,
        input_ref=record,source_sha256=spec["source_sha256"],runtime=runtime,physical_gpu=4,
        training_passes=30,updates_per_arm=960,VAL_influences_optimization=False,conditioned_on_H040_VAL=False,outer_truth_read=False))
    ctx=dict(root_authorizer="root_agent",admitted=True,phase=spec["phase"],arm=worker.ARM,cfg=cfg,reader=reader,
        output=output,guard=guard,device=device,config_ref=refs["config"],registry=registry,entry_refs=entries,
        observation_cache_ref=refs["observation_cache"],observation_source_refs=spec["observation_cache_source_refs"],bindings=bindings)
    if spec["phase"]=="evaluate":ctx["train_completion_ref"]=spec["train_completion_ref"]
    ctx["runtime_memo_handle"]=memo_handle
    return ctx

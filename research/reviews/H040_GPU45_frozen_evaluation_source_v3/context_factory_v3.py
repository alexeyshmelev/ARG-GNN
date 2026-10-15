
"""Root H040 context with preserved observation data, guards and GPU leases."""
from pathlib import Path
import hashlib,json,importlib.util
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H040_GPU45_frozen_evaluation_source_v3/context_factory_v3.py"
GUARD="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py"
def require(ok,msg):
    if not ok:raise ValueError(msg)
def build_context_v3(record,spec):
    from arggnn import H040_GPU45_frozen_evaluation_v3 as worker
    from arggnn import census_tsinfer_fit_v1 as fit
    from arggnn import train_mutation_routed_whole_tree_gnn_v1 as original
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    lease=validate_lease();apply_and_validate_title()
    require(lease["physical_gpu"]==spec["physical_gpu"]==5,"Root evaluation selects GPU5 after actual event-MLP wait0")
    worker.science()
    reader=fit.ConsumerReaderV1()
    fit._check_active_v1(spec);fit.require_logical_job_v1(lease,spec)
    require(spec["root_context_module_ref"]==dict(path=SOURCE,sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()),"Actual root context source differs")
    refs=spec["refs"];cfg=reader.decode_json(refs["config"]);worker.validate_protocol(cfg)
    worker.validate_evaluation_contract_v3(spec,cfg,reader)
    for p,sha in spec["source_sha256"].items():reader.read(dict(path=p,sha256=sha))
    require(reader.decode_json(refs["runtime"])["job_id"]==spec["job_id"],"Runtime job mismatch")
    require(spec["device"]=="cuda" and spec["max_cuda_reserved_bytes"]==50000000000,"Admitted FP32 shared CUDA limit required")
    guard_ref=cfg["helper_refs"]["full_reservation_adapter"]
    require(guard_ref==worker.GUARD_REF and cfg["helper_refs"]["actual_full_storage_launcher"]==worker.LAUNCHER_REF,"Exact fixed current resource guard and launcher required")
    reader.read(guard_ref);reader.decode_json(cfg["marker_repair_authority_ref"])
    module_spec=importlib.util.spec_from_file_location("_H040_current_guard",ROOT/GUARD)
    storage_guard=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(storage_guard)
    require(spec["memo_helper_ref"]==worker.MEMO_REF and spec["source_sha256"].get(worker.MEMO_REF["path"])==worker.MEMO_REF["sha256"],
        "Mandatory actual memo helper source pin required")
    reader.read(worker.MEMO_REF);reader.read(worker.V4_REF)
    memo_spec=importlib.util.spec_from_file_location("_H040_transaction_memo_v2",ROOT/worker.MEMO_REF["path"])
    memo_module=importlib.util.module_from_spec(memo_spec);memo_spec.loader.exec_module(memo_module)
    memo_handle=memo_module.install(storage_guard.budget.v4,
        expected_source_path=ROOT/worker.V4_REF["path"],expected_source_sha256=worker.V4_REF["sha256"])
    storage_guard.configure_v2(cfg["marker_repair_authority_ref"],policy_ref=worker.POLICY_REF)
    storage=reader.decode_json(refs["storage"])
    require(spec["completed_overlay_policy_ref"]==storage["completed_overlay_policy_ref"]==worker.POLICY_REF,
        "Same exact active policy in input and storage required")
    require(spec["transitive_source_closure_ref"] in storage["reservation"]["consumer_bindings"],
        "Exact transitive source/evidence JSON must be reserved")
    require(all(r in storage["reservation"]["consumer_bindings"] for r in (worker.MEMO_REF,worker.V4_REF,spec["runtime_supersession_ref"])),
        "Memo, original v4 and supersession authority must be direct reserved bindings")
    storage_guard.validate_storage_v9(spec,cfg,storage,
        source_ref=dict(path=worker.SOURCE,sha256=spec["source_sha256"][worker.SOURCE]),
        config_ref=refs["config"],runtime_ref=refs["runtime"])
    output=storage_guard.PilotOutputV9(spec["output_directory"],storage,reader)
    output.save("runtime_memo_installation.json",memo_handle.receipt)
    storage_guard.budget_check_v9(storage,boundary=True,required_write_bytes=cfg["qualification_allowance_bytes"])
    from arggnn.biological_candidate import configure_fit
    device,runtime,lease=configure_fit("cuda",reader.decode_json(cfg["base_fit_config"]),output.path)
    import torch,torch_geometric
    require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Pinned torch/PyG runtime differs")
    torch.cuda.set_per_process_memory_fraction(min(1.,50000000000/torch.cuda.get_device_properties(device).total_memory),device)
    output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob("*") if p.is_file())
    require(output.used<=cfg["qualification_allowance_bytes"],"Qualification exceeds reserved bound")
    def guard():
        original.lightweight_resources_v1(spec,lease)
        storage_guard.budget_check_v9(storage)
    guard()
    generation=reader.decode_json(refs["generation_config"])
    registration=reader.decode_json(refs["registration"])
    require(registration["config_ref"]==refs["generation_config"] and registration["parent_registry"]==generation["parent_registry"],"Original generation registration differs")
    entries=spec["observation_entry_refs"];registry=generation["parent_registry"]
    require(set(entries)==set(worker.TRAIN+worker.VAL),"Original40 observations required")
    bindings=reader.decode_json(refs["target_bindings"])["parents"]
    ids=worker.TRAIN if spec["phase"]=="train" else worker.VAL
    require(set(bindings)==set(ids),"Exact phase-only target metadata required")
    for pid,b in bindings.items():
        require(b["truth_npz"]["path"]==generation["output_root"]+"/"+pid+"/truth/full_truth_intervals.npz","Same-parent truth path required")
        provenance=reader.decode_json(b["generation_receipt"])
        require(provenance["parent_id"]==pid and provenance["files"]["truth/full_truth_intervals.npz"]["sha256"]==b["truth_npz"]["sha256"],"Same-parent generation identity differs")
    output.save("started.json",dict(hypothesis_id="H040",phase=spec["phase"],arm=spec["arm"],input_ref=record,source_sha256=spec["source_sha256"],runtime=runtime,training_passes=30,updates_per_arm=960,VAL_influences_optimization=False,outer_truth_read=False))
    ctx=dict(root_authorizer="root_agent",admitted=True,phase=spec["phase"],arm=spec["arm"],cfg=cfg,reader=reader,
        output=output,guard=guard,device=device,config_ref=refs["config"],registry=registry,entry_refs=entries,
        observation_cache_ref=refs["observation_cache"],observation_source_refs=spec["observation_cache_source_refs"],bindings=bindings)
    ctx["train_completion_refs"]=spec["train_completion_refs"]
    ctx["train_actual_wait_refs"]=spec["train_actual_wait_refs"]
    ctx["runtime_memo_handle"]=memo_handle
    return ctx

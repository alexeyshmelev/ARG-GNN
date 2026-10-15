"""H050 root context; original storage, lease, configure_fit and runtime guards."""
import json,os
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1.bridge_common_v1 import (
    ROOT,HERE,CONTEXT,require,ref,cap_v1,worker,framework,load_guard_v1,POLICY_REF,LAUNCHER_REF,
    validate_spec_v1,count_plan_v1,validate_plan_caps_v1,authenticate_actual_cut_preparation_v1)
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1.root_common_v1 import (
    current_RSS_bytes_v1,preparation_check_v1)
from research.reviews.order2_cut_memory_implementation_20261015_v1 import integration_hooks_v1 as cut_hooks
SOURCE=CONTEXT


def build_context_v1(record,spec):
    from arggnn import census_tsinfer_fit_v1 as fit
    require(json.loads(os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])==POLICY_REF,"Original current980 policy environment")
    reader=fit.ConsumerReaderV1();fit._check_active_v1(spec)
    require(reader.decode_json(record)==spec,"Same pinned root capability bytes")
    cfg=reader.decode_json(spec["refs"]["config"]);storage=reader.decode_json(spec["refs"]["storage"])
    validate_spec_v1(reader,spec,cfg,storage)
    guard_module,memo=load_guard_v1(consumer=True)
    guard_module.validate_storage_v9(spec,cfg,storage,source_ref=spec["main_source_ref"],
        config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
    worker.h45.science()
    output=guard_module.PilotOutputV9(spec["output_directory"],storage,reader)
    runtime=reader.decode_json(spec["refs"]["runtime"])
    require(runtime["job_id"]==spec["job_id"] and runtime["device"]==spec["device"]
        and runtime["physical_gpu"]==spec["physical_gpu"]
        and runtime["output_directory"]==spec["output_directory"],"Exact actual H050 runtime")
    generation=reader.decode_json(spec["refs"]["generation_config"])
    registration=reader.decode_json(spec["refs"]["registration"])
    require(registration["config_ref"]==spec["refs"]["generation_config"]
        and registration["parent_registry"]==generation["parent_registry"],"Unchanged original generation identity")
    original=reader.decode_json(cfg["original_H038_input_ref"])
    require(all(spec["refs"][k]==original["refs"][k] for k in ("registration","generation_config","observation_cache"))
        and spec["observation_entry_refs"]==original["observation_entry_refs"]
        and spec["observation_cache_source_refs"]==original["observation_cache_source_refs"]
        and spec["observation_cache_source_refs"]==cfg["observation_cache_source_refs"]
        and set(spec["observation_entry_refs"])==set(worker.TRAIN+worker.VAL),"Exact original40 observed entries/cache/sources")
    preparation=framework.completed_marker_preparation_v1(reader,spec)
    counts=reader.decode_json(cfg["count_manifest_ref"])
    require(all(counts["parents"][p]["observation_entry_ref"]==spec["observation_entry_refs"][p]
        for p in worker.TRAIN+worker.VAL),"Genuine count metadata binds original40 observations")
    plan=count_plan_v1(counts,spec["phase"],preparation,cfg);validate_plan_caps_v1(plan,cfg,storage["reserved_new_bytes"])
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    lease=validate_lease();apply_and_validate_title();fit.require_logical_job_v1(lease,spec)
    require(spec["physical_gpu"] in (4,5) and lease["physical_gpu"]==spec["physical_gpu"]
        and runtime["process_title"]=="AI Auto Research Pipeline Test"
        and spec["max_cuda_reserved_bytes"]==cfg["max_cuda_reserved_bytes"],"Original canonical GPU4/5 lease/title/cap")
    guard_module.budget_check_v9(storage,boundary=True,required_write_bytes=cfg["qualification_allowance_bytes"])
    from arggnn.biological_candidate import configure_fit
    device,qualified_runtime,qualified_lease=configure_fit("cuda",reader.decode_json(cfg["base_fit_config"]),output.path)
    import torch,torch_geometric
    require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Pinned original Torch/PyG")
    require(qualified_lease["physical_gpu"]==spec["physical_gpu"],"Same root-selected actual physical GPU")
    torch.cuda.set_per_process_memory_fraction(min(1.,cfg["max_cuda_reserved_bytes"]/
        torch.cuda.get_device_properties(device).total_memory),device)
    output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob("*") if p.is_file())
    require(output.used<=cfg["qualification_allowance_bytes"],"Original bounded configure_fit artifacts")
    from arggnn import train_mutation_routed_whole_tree_gnn_v1 as original_worker
    def guard():original_worker.lightweight_resources_v1(spec,qualified_lease)
    guard()
    targets=reader.decode_json(spec["refs"]["target_bindings"])
    originals=reader.decode_json(original["refs"]["target_bindings"])
    ids=worker.TRAIN if spec["phase"]=="train" else worker.VAL;bindings=targets["parents"]
    require(set(bindings)==set(ids) and targets["outer_included"] is False
        and targets.get("phase")==spec["phase"]
        and targets.get("TRAIN_included") is (spec["phase"]=="train")
        and targets.get("VAL_included") is (spec["phase"]=="evaluate")
        and bindings=={p:originals["parents"][p] for p in ids},"Exact original phase-only targets; no outer")
    for pid,binding in bindings.items():
        require(binding["truth_npz"]["path"]==generation["output_root"]+"/"+pid+"/truth/full_truth_intervals.npz","Original same-parent truth metadata path")
        provenance=reader.decode_json(binding["generation_receipt"])
        require(provenance["parent_id"]==pid and provenance["files"]["truth/full_truth_intervals.npz"]["sha256"]==binding["truth_npz"]["sha256"],"Original generation receipt binds phase target metadata")
    cut_preparation=authenticate_actual_cut_preparation_v1(reader,spec,cfg)
    require(cfg["cut_preparation_completion_ref"]["path"]==cfg["expected_future_preparation_output"]+"/completed.json","Only successful actual v5 preparation completion")
    def before_episode(pid,bank,declared):
        guard();actual=worker.h45.episode_upper_bounds_v1(bank);row=counts["parents"][pid]
        require(declared==actual and actual["native_nodes"]==row["native_nodes"]
            and actual["native_holds"]==row["native_holds"] and actual["haplotypes"]==row["haplotypes"]==952,"Exact original complete-bank episode bounds")
        require(actual["compact_payload_upper_bound_bytes"]<=min(cfg["episode_payload_cap_bytes"],cfg["episode_cache_cap_bytes"]),"One whole episode fits both reviewed caps")
        estimate=actual["preparation_named_array_allowance_bytes"]+cfg["episode_python_scratch_allowance_bytes"]
        require(estimate<=cfg["episode_preparation_cap_bytes"] and current_RSS_bytes_v1()+estimate<=cfg["max_cpu_rss_bytes"],"Prospective episode array plus root-reviewed Python scratch/RSS allowance")
    def before_cut(pid,bank,receipt):
        guard();row=counts["parents"][pid]
        require(receipt.get("native_nodes")==row["native_nodes"] and receipt.get("native_holds")==row["native_holds"]
            and receipt.get("haplotypes")==952 and receipt.get("lossless_roundtrip_verified") is True
            and receipt.get("original_observed_entry_ref")==spec["observation_entry_refs"][pid],"Genuine same-bank corrected cut receipt")
        declared=cut_hooks.parent_bounds_v1(bank,range(preparation["parents"][pid]["counts"]["scoring_centers"]))
        # Preserve the original conservative whole-grid/cut/marker prospective checks.
        # H050 native-hold decoding uses less retained state; no policy is relaxed.
        preparation_check_v1(pid,bank,declared,cfg,counts,preparation)
    ctx=dict(root_authorizer="root_agent",admitted=True,hypothesis_id="H050",arm="joint_four_arm",
        phase=spec["phase"],cfg=cfg,reader=reader,output=output,config_ref=spec["refs"]["config"],
        source_manifest_ref=cfg["source_manifest_ref"],registry=generation["parent_registry"],
        entry_refs=spec["observation_entry_refs"],observation_cache_ref=spec["refs"]["observation_cache"],
        observation_source_refs=spec["observation_cache_source_refs"],preparation=preparation,
        preparation_completion_ref=spec["preparation_completion_ref"],preparation_actual_wait_ref=spec["preparation_actual_wait_ref"],
        device=device,guard=guard,physical_gpu=spec["physical_gpu"],bindings=bindings,
        cut_preparation=cut_preparation,before_episode_build=before_episode,before_cut_decode=before_cut,
        cut_preparation_completion_ref=spec["cut_preparation_completion_ref"],cut_preparation_actual_wait_ref=spec["cut_preparation_actual_wait_ref"])
    if spec["phase"]=="evaluate":ctx["train_completion_ref"]=spec["train_completion_ref"]
    output.save("started.json",dict(schema="H050_direct_readout_started_v1",hypothesis_id="H050",phase=spec["phase"],
        input_ref=record,output_plan=plan,memo_receipt=memo.receipt,parameters_per_arm=worker.model.PARAMETERS,
        training_passes=30,updates_per_arm=960,updates_total=3840,checkpoint_files=24,
        all192_evaluation_seals_before_truth=True,VAL_influences_optimization=False,outer_truth_read=False,
        launch_receipt_schemas_are_legacy_operational_formats=list(("H049_marker_original_launch_v1","H049_marker_original_direct_wait_v1"))))
    return ctx

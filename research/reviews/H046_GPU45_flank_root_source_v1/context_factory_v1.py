"""H046 future root context. No import-time guard, data or CUDA execution."""
from pathlib import Path
import json,os
from research.reviews.H046_GPU45_flank_root_source_v1.root_common_v1 import (
    ROOT,HERE,require,ref,GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF,
    RSS_CAP,CUDA_CAP,count_plan_v1,preparation_check_v1,validate_train_lineage_v1,load_guard_v1)
SOURCE=HERE+"/context_factory_v1.py"

def validate_closure_v1(reader,spec,cfg,storage):
    manifest=reader.decode_json(spec["consumer_manifest_ref"])
    require(manifest["schema"]=="H046_exact_transitive_consumer_closure_v1"
        and manifest["hypothesis_id"]=="H046" and manifest["phase"]==spec["phase"]
        and manifest["source_manifest_ref"]==cfg["source_manifest_ref"]
        and manifest["source_sha256"]==spec["source_sha256"]
        and manifest["refs"]=={k:v for k,v in spec["refs"].items() if k!="storage"}
        and manifest["observation_entry_refs"]==spec["observation_entry_refs"]
        and manifest["observation_cache_source_refs"]==spec["observation_cache_source_refs"],
        "Exact H046 transitive source/original-observation metadata closure required")
    sources=reader.decode_json(cfg["source_manifest_ref"])
    require(sources["schema"]=="H046_flank_frozen_source_manifest_v1"
        and sources["source_sha256"]==spec["source_sha256"],"Exact frozen source manifest required")
    for path,digest in sources["source_sha256"].items():reader.read(dict(path=path,sha256=digest))
    for r in sources["metadata_refs"].values():reader.read(r)
    required=[spec["main_source_ref"],spec["worker_cli_ref"],spec["root_context_module_ref"],
        spec["root_driver_ref"],cfg["source_manifest_ref"],spec["consumer_manifest_ref"],
        cfg["count_manifest_ref"],spec["preparation_completion_ref"],spec["preparation_actual_wait_ref"],LAUNCHER_REF,GUARD_REF,POLICY_REF,AUTH_REF,REPAIR_REF,MEMO_REF,V4_REF]
    require(all(r in storage["reservation"]["consumer_bindings"] for r in required),
            "Actual producer/CLI/context/driver/counts/memo/authorities must be reserved")
    require(len(storage["reservation"]["consumer_bindings"])<=64,"Guard direct-reference bound")
    for name in ("preparation_completion_ref","preparation_actual_wait_ref"):
        require(manifest[name]==spec[name] and cfg[name]==spec[name],"Exact original H044 preparation terminal closure")
    if spec["phase"]=="evaluate":
        for name in ("train_completion_ref","train_launch_ref","train_actual_wait_ref"):
            require(manifest[name]==spec[name],"Exact TRAIN completion and wait closure")
        validate_train_lineage_v1(reader,spec)

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
    require(value["artifact_format"]=="H044_all40_batched_coordinate_binding_v2","Immutable all40 batched marker preparation")
    return value

def build_context_v1(record,spec):
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    lease=validate_lease();apply_and_validate_title()
    gpu=spec["physical_gpu"]
    require(gpu in (4,5) and lease["physical_gpu"]==gpu,"Root-selected single eligible GPU4/5 only")
    from arggnn import train_continuous_clade_flank_v1 as worker
    from arggnn import census_tsinfer_fit_v1 as fit
    from arggnn import train_mutation_routed_whole_tree_gnn_v1 as original
    require(spec["schema"]=="H046_paired_flank_root_input_v1" and spec["hypothesis_id"]=="H046"
        and spec["arm"]=="joint_pair" and spec["phase"] in ("train","evaluate")
        and spec["root_authorizer"]=="root_agent" and spec["status"]=="active","Active root paired H046 input")
    require(spec["root_context_module_ref"]==ref(SOURCE),"Actual root context changed")
    require(json.loads(os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])==POLICY_REF,"Exact credit-v3 policy env")
    reader=fit.ConsumerReaderV1();fit._check_active_v1(spec);fit.require_logical_job_v1(lease,spec)
    refs=spec["refs"];cfg=reader.decode_json(refs["config"]);worker.validate_protocol(cfg)
    require(cfg["base_fit_config"]["path"]=="configs/census_tsinfer_fit_v1.json"
        and spec["device"]=="cuda" and spec["max_cpu_rss_bytes"]==RSS_CAP
        and spec["max_cuda_reserved_bytes"]==CUDA_CAP,"Original runtime base and fixed RSS/CUDA caps")
    runtime_metadata=reader.decode_json(refs["runtime"])
    require(runtime_metadata["job_id"]==spec["job_id"] and runtime_metadata["physical_gpu"]==gpu
        and runtime_metadata["process_title"]=="AI Auto Research Pipeline Test","Runtime/lease/title identity")
    require(cfg["helper_refs"]["full_reservation_adapter"]==GUARD_REF
        and cfg["helper_refs"]["actual_full_storage_launcher"]==LAUNCHER_REF
        and cfg["completed_overlay_policy_ref"]==POLICY_REF and cfg["marker_repair_authority_ref"]==REPAIR_REF
        and cfg["transaction_memo_adapter_ref"]==MEMO_REF and cfg["original_v4_ref"]==V4_REF,
        "Current unchanged guard/launcher/policy plus finalized memo2/originalv4 required")
    for r in (GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF):reader.read(r)
    storage=reader.decode_json(refs["storage"])
    require(storage["completed_overlay_policy_ref"]==POLICY_REF,"One exact storage/launcher/worker policy")
    validate_closure_v1(reader,spec,cfg,storage)
    storage_guard,memo=load_guard_v1(consumer=True)
    storage_guard.validate_storage_v9(spec,cfg,storage,source_ref=spec["main_source_ref"],
                                     config_ref=refs["config"],runtime_ref=refs["runtime"])
    worker.science()
    output=storage_guard.PilotOutputV9(spec["output_directory"],storage,reader)
    storage_guard.budget_check_v9(storage,boundary=True,required_write_bytes=cfg["qualification_allowance_bytes"])
    from arggnn.biological_candidate import configure_fit
    device,runtime,qualified_lease=configure_fit("cuda",reader.decode_json(cfg["base_fit_config"]),output.path)
    require(qualified_lease["physical_gpu"]==gpu,"Qualified leased GPU must retain root assignment")
    import torch,torch_geometric
    require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Pinned Torch/PyG")
    torch.cuda.set_per_process_memory_fraction(min(1.,CUDA_CAP/torch.cuda.get_device_properties(device).total_memory),device)
    output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob("*") if p.is_file())
    require(output.used<=cfg["qualification_allowance_bytes"],"Runtime setup exceeds fixed allowance")
    guard=lambda:original.lightweight_resources_v1(spec,qualified_lease)
    guard()
    generation=reader.decode_json(refs["generation_config"]);registration=reader.decode_json(refs["registration"])
    require(registration["config_ref"]==refs["generation_config"]
        and registration["parent_registry"]==generation["parent_registry"],"Original generation identity")
    original_input=reader.decode_json(cfg["original_H038_input_ref"])
    require(all(refs[k]==original_input["refs"][k] for k in ("registration","generation_config","observation_cache"))
        and spec["observation_entry_refs"]==original_input["observation_entry_refs"]
        and spec["observation_cache_source_refs"]==original_input["observation_cache_source_refs"],
        "Exact original H038 observation and provenance capabilities")
    entries=spec["observation_entry_refs"];registry=generation["parent_registry"]
    require(set(entries)==set(worker.TRAIN+worker.VAL)
        and spec["observation_cache_source_refs"]==cfg["observation_cache_source_refs"],"Original40 observations")
    original_targets=reader.decode_json(original_input["refs"]["target_bindings"])
    binding_metadata=reader.decode_json(refs["target_bindings"]);bindings=binding_metadata["parents"]
    ids=worker.TRAIN if spec["phase"]=="train" else worker.VAL
    require(set(bindings)==set(ids) and binding_metadata["TRAIN_included"]==(spec["phase"]=="train")
        and binding_metadata["VAL_included"]==(spec["phase"]=="evaluate")
        and binding_metadata["outer_included"] is False
        and bindings=={p:original_targets["parents"][p] for p in ids},"Exact original phase-only targets")
    for pid,b in bindings.items():
        require(b["truth_npz"]["path"]==generation["output_root"]+"/"+pid+"/truth/full_truth_intervals.npz",
                "Original same-parent target path")
        provenance=reader.decode_json(b["generation_receipt"])
        require(provenance["parent_id"]==pid
            and provenance["files"]["truth/full_truth_intervals.npz"]["sha256"]==b["truth_npz"]["sha256"],
            "Original same-parent generation identity")
    preparation=completed_preparation_v1(reader,spec,cfg)
    counts=reader.decode_json(cfg["count_manifest_ref"])
    require(all(counts["parents"][p]["observation_entry_ref"]==entries[p] for p in entries),
            "Counts must bind every exact original observation entry")
    plan=count_plan_v1(counts,spec["phase"],preparation)
    require(plan["all96_original_grid_probability_bytes"]<=cfg["RAM_prediction_cap_bytes"],"Complete96 RAM matrix planning cap")
    def before_build(pid,bank,declared):
        guard()
        proof=preparation_check_v1(pid,bank,declared,cfg,counts,preparation)
        print(json.dumps(dict(stage="H046_prebuild_bound_and_RSS",**proof)),flush=True)
    output.save("started.json",dict(hypothesis_id="H046",phase=spec["phase"],arm="joint_pair",
        input_ref=record,source_sha256=spec["source_sha256"],runtime=runtime,physical_gpu=gpu,
        training_passes=30,updates_per_arm=960,all96_evaluation_seals_before_truth=True,
        first_steps_within_continuous_training=True,output_plan=plan,memo_receipt=memo.receipt,
        VAL_influences_optimization=False,conditioned_on_H040_VAL=False,outer_truth_read=False))
    ctx=dict(root_authorizer="root_agent",admitted=True,phase=spec["phase"],arm="joint_pair",cfg=cfg,reader=reader,
        output=output,guard=guard,device=device,config_ref=refs["config"],registry=registry,entry_refs=entries,
        observation_cache_ref=refs["observation_cache"],observation_source_refs=spec["observation_cache_source_refs"],
        bindings=bindings,before_flank_build=before_build,preparation=preparation,
        preparation_completion_ref=spec["preparation_completion_ref"],preparation_actual_wait_ref=spec["preparation_actual_wait_ref"])
    if spec["phase"]=="evaluate":ctx["train_completion_ref"]=spec["train_completion_ref"]
    return ctx

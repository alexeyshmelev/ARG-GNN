"""H049 root consumer context. Import performs no guard, payload or CUDA work."""
from pathlib import Path
import ast,json,os,resource
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1.root_common_v1 import (
    ROOT,HERE,require,ref,GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF,
    cap_v1,count_plan_v1,preparation_check_v1,current_RSS_bytes_v1,
    validate_eval_train_io_v1,load_guard_v1)
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import producer_v1 as worker
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import cut_recipe_codec_v1 as codec
SOURCE="research/reviews/H049_order2_cut_memory_gpu_exec_source_20261015_v2/context_factory_v2.py"
PREP_SOURCE=None  # GPU-only; preparation producer is explicitly declared, never hardcoded.
WORKER_SOURCE=HERE+"/producer_v1.py"


PREPARATION_METADATA_FIELDS_V2=(
    "cut_preparation_completion_ref","cut_preparation_actual_wait_ref",
    "cut_preparation_admission_ref","cut_preparation_input_ref",
    "cut_preparation_config_ref","cut_preparation_runtime_ref","cut_preparation_launch_ref")


def declared_cut_preparation_source_refs_v2(cfg):
    refs=dict(cut_preparation_producer_ref=cfg.get("preparation_producer_ref"),
        cut_preparation_codec_ref=cfg.get("cut_preparation_codec_ref"),
        cut_preparation_source_manifest_ref=cfg.get("cut_preparation_source_manifest_ref"))
    require(all(isinstance(r,dict) and set(r)=={"path","sha256"}
        and isinstance(r["path"],str) and isinstance(r["sha256"],str)
        and len(r["sha256"])==64 for r in refs.values()),"Root-declared exact preparation source refs")
    return refs


def validate_declared_cut_preparation_sources_v2(reader,cfg,gpu_sources):
    refs=declared_cut_preparation_source_refs_v2(cfg)
    producer=refs["cut_preparation_producer_ref"];codec_ref=refs["cut_preparation_codec_ref"]
    manifest=reader.decode_json(refs["cut_preparation_source_manifest_ref"])
    require(manifest.get("schema")=="H049_order2_frozen_source_manifest_v1"
        and manifest.get("hypothesis_id")=="H049" and manifest.get("frozen_model_ref")==cfg["core_ref"]
        and all(gpu_sources.get(p)==s for p,s in manifest["source_sha256"].items())
        and manifest["source_sha256"].get(producer["path"])==producer["sha256"]
        and manifest["source_sha256"].get(codec_ref["path"])==codec_ref["sha256"],
        "Declared preparation closure is inside exact GPU closure and same frozen cut-plan core")
    reader.read(producer)
    repaired=reader.read(codec_ref).decode("utf-8")
    original=reader.read(ref(HERE+"/cut_recipe_codec_v1.py")).decode("utf-8")
    def function(s,name):
        n=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==name)
        return ast.get_source_segment(s,n)
    def literal(s,name):
        n=next(x for x in ast.parse(s).body if isinstance(x,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id==name for t in x.targets))
        return ast.literal_eval(n.value)
    require(function(repaired,"decode_v1")==function(original,"decode_v1")
        and function(repaired,"compact_upper_v1")==function(original,"compact_upper_v1")
        and literal(repaired,"FORMAT")==codec.FORMAT and literal(repaired,"MEMBERS")==codec.MEMBERS,
        "Exactly unchanged compact decode function, bounds and format; no scientific codec substitution")
    return refs,manifest


def authenticate_declared_cut_preparation_lineage_v2(reader,spec,cfg,completion,wait):
    refs,manifest=validate_declared_cut_preparation_sources_v2(reader,cfg,spec["source_sha256"])
    require(all(spec.get(k)==cfg.get(k) for k in PREPARATION_METADATA_FIELDS_V2)
        and all(spec.get(k)==v for k,v in refs.items()),"Exact root config/capability preparation refs")
    admitted=reader.decode_json(cfg["cut_preparation_admission_ref"])
    launch_input=reader.decode_json(cfg["cut_preparation_input_ref"])
    prep_cfg=reader.decode_json(cfg["cut_preparation_config_ref"])
    runtime=reader.decode_json(cfg["cut_preparation_runtime_ref"])
    launch=reader.decode_json(cfg["cut_preparation_launch_ref"])
    worker.validate_protocol(prep_cfg,phase="prepare")
    source_manifest=refs["cut_preparation_source_manifest_ref"];producer=refs["cut_preparation_producer_ref"]
    codec_ref=refs["cut_preparation_codec_ref"]
    driver=launch_input["root_driver_ref"];context=launch_input["root_context_module_ref"]
    require(completion["input_ref"]==cfg["cut_preparation_input_ref"]
        and completion["config_ref"]==cfg["cut_preparation_config_ref"]
        and completion["source_manifest_ref"]==prep_cfg["source_manifest_ref"]==source_manifest
        and completion.get("codec_ref")==prep_cfg.get("cut_recipe_codec_ref")==codec_ref
        and completion["producer_ref"]==prep_cfg["preparation_producer_ref"]==producer
        and type(completion.get("preparation_execution_version")) is int
        and completion["preparation_execution_version"]>=2
        and prep_cfg["core_ref"]==cfg["core_ref"] and prep_cfg["registration_ref"]==cfg["registration_ref"]
        and prep_cfg["original_H038_input_ref"]==cfg["original_H038_input_ref"],
        "Actual declared preparation producer/config/codec/closure and original registered science")
    require(launch_input.get("phase")=="prepare" and launch_input.get("device")=="cpu"
        and launch_input.get("physical_gpu") is None and launch_input.get("fit_authorized") is False
        and launch_input.get("root_authorizer")=="root_agent" and launch_input.get("status")=="active"
        and "target_bindings" not in launch_input["refs"]
        and launch_input["producer_ref"]==launch_input["main_source_ref"]==launch_input["worker_cli_ref"]==producer
        and launch_input["refs"]["config"]==cfg["cut_preparation_config_ref"]
        and launch_input["refs"]["runtime"]==cfg["cut_preparation_runtime_ref"]
        and launch_input["source_sha256"]==manifest["source_sha256"]
        and manifest["source_sha256"].get(driver["path"])==driver["sha256"]
        and manifest["source_sha256"].get(context["path"])==context["sha256"]
        and admitted.get("phase")=="prepare" and admitted["driver_ref"]==driver
        and runtime["job_id"]==launch_input["job_id"] and runtime["device"]=="cpu"
        and runtime.get("physical_gpu") is None and runtime["output_directory"]==launch_input["output_directory"],
        "Actual observation-only preparation admission, launch input, sources and runtime")
    require(wait["launch_ref"]==cfg["cut_preparation_launch_ref"]
        and wait["launcher_pid"]==launch["launcher_pid"] and wait["launcher_start"]==launch["launcher_start"]
        and launch.get("direct_child") is True and launch["input_ref"]==completion["input_ref"]
        and launch.get("phase")==wait.get("phase")=="prepare"
        and launch.get("physical_gpu") is None and wait.get("physical_gpu") is None
        and cfg["cut_preparation_completion_ref"]["path"]==launch_input["output_directory"]+"/completed.json",
        "Actual declared same-launcher successful preparation wait and completion")
    counts=reader.decode_json(cfg["count_manifest_ref"])
    for pid,parent in completion["parents"].items():
        receipt=reader.decode_json(parent["receipt_ref"]);row=counts["parents"][pid]
        require(receipt.get("schema")=="H049_compact_cut_recipe_v1" and receipt.get("format")==codec.FORMAT
            and receipt.get("codec_ref")==codec_ref and receipt.get("input_ref")==completion["input_ref"]
            and receipt.get("lossless_roundtrip_verified") is True and receipt.get("observations_only") is True
            and receipt.get("targets_read") is False and receipt.get("haplotypes")==952
            and receipt.get("original_observed_entry_ref")==spec["observation_entry_refs"][pid]
            and receipt.get("arrays_ref")==parent["arrays_ref"]
            and receipt.get("native_nodes")==row["native_nodes"]
            and receipt.get("native_holds")==row["native_holds"]
            and parent["counts"]["haplotypes"]==952,
            "Every original parent has a lossless receipt bound to the declared encoder/input/counts")
    return completion


def validate_closure_v1(reader,spec,cfg,storage):
    manifest=reader.decode_json(spec["consumer_manifest_ref"])
    require(manifest["schema"]=="H049_exact_transitive_consumer_closure_v1"
        and manifest["hypothesis_id"]=="H049" and manifest["phase"]==spec["phase"]
        and manifest["source_sha256"]==spec["source_sha256"]
        and manifest["refs"]=={k:v for k,v in spec["refs"].items() if k!="storage"}
        and manifest["observation_entry_refs"]==spec["observation_entry_refs"]
        and manifest["observation_cache_source_refs"]==spec["observation_cache_source_refs"],
        "Fresh exact H049 source/metadata/observation closure")
    sources=reader.decode_json(cfg["source_manifest_ref"])
    require(sources["schema"]=="H049_order2_frozen_source_manifest_v1"
        and sources["source_sha256"]==spec["source_sha256"],"Frozen H049 source closure")
    for path,digest in sources["source_sha256"].items():
        reader.read(dict(path=path,sha256=digest))
    for r in sources.get("metadata_refs",{}).values():reader.read(r)
    required=[spec["main_source_ref"],spec["worker_cli_ref"],spec["root_context_module_ref"],
        spec["root_driver_ref"],cfg["source_manifest_ref"],spec["consumer_manifest_ref"],
        cfg["count_manifest_ref"],cfg["registration_ref"],spec["preparation_completion_ref"],
        spec["preparation_actual_wait_ref"],GUARD_REF,POLICY_REF,AUTH_REF,REPAIR_REF,MEMO_REF,V4_REF]
    if spec["phase"]!="prepare":
        required += [LAUNCHER_REF]+[cfg[k] for k in PREPARATION_METADATA_FIELDS_V2]
        prep_refs=declared_cut_preparation_source_refs_v2(cfg);required += list(prep_refs.values())
        require(all(manifest.get(k)==spec.get(k)==v for k,v in prep_refs.items()),
            "Declared preparation sources in capability and consumer closure")
    require(all(r in storage["reservation"]["consumer_bindings"] for r in required)
        and len(storage["reservation"]["consumer_bindings"])<=64,"Exact direct and transitive reserved closure")
    for name in ("preparation_completion_ref","preparation_actual_wait_ref"):
        require(manifest[name]==spec[name]==cfg[name],"Exact immutable H044 marker preparation")
    if spec["phase"]!="prepare":
        for name in ("cut_preparation_completion_ref","cut_preparation_actual_wait_ref"):
            require(manifest[name]==spec[name]==cfg[name],"Exact H049 cut preparation terminal lineage")
    if spec["phase"]=="evaluate":
        require(manifest["train_io_lineage"]==spec["train_io_lineage"]
            ==validate_eval_train_io_v1(reader,spec),"Exact fresh full-budget TRAIN lineage")


def completed_marker_preparation_v1(reader,spec):
    from arggnn.prepare_marker_resolution_observations_v1 import PARENTS,COORDINATE_KEYS
    r=spec["preparation_completion_ref"];v=reader.decode_json(r)
    w=reader.decode_json(spec["preparation_actual_wait_ref"])
    require(w["actual_direct_launcher_wait"] is True and w["returncode"]==0
        and w["completed_ref"]==r and w["owned_group_drained"] is True,"Actual original H044 CPU wait/drain")
    require(v["schema"]=="H044_marker_observation_preparation_v1"
        and v["status"]=="completed_all40_observation_only_sidecars"
        and v["hypothesis_id"]=="H044" and set(v["parents"])==set(PARENTS)
        and v["coordinate_keys"]==list(COORDINATE_KEYS)
        and v["artifact_format"]=="H044_all40_batched_coordinate_binding_v2"
        and v["observation_cache_ref"]==spec["refs"]["observation_cache"]
        and v["entry_refs"]==spec["observation_entry_refs"]
        and v["observation_source_refs"]==spec["observation_cache_source_refs"]
        and v["observations_only"] is True and v["targets_read"] is False,
        "Unchanged original all40 marker preparation")
    return v


def completed_cut_preparation_v1(reader,spec,cfg):
    r=spec["cut_preparation_completion_ref"];v=reader.decode_json(r)
    w=reader.decode_json(spec["cut_preparation_actual_wait_ref"])
    require(w["actual_direct_launcher_wait"] is True and w["returncode"]==0
        and w["completed_ref"]==r and w["owned_group_drained"] is True
        and w.get("error") is None,"Actual successful H049 CPU direct wait/drain")
    require(v["schema"]=="H049_order2_observation_preparation_v1"
        and v["hypothesis_id"]=="H049" and v["status"]=="completed_all40_compact_cut_recipes"
        and v["format"]==codec.FORMAT and set(v["parents"])==set(worker.TRAIN+worker.VAL)
        and v["marker_preparation_completion_ref"]==spec["preparation_completion_ref"]
        and v["marker_preparation_actual_wait_ref"]==spec["preparation_actual_wait_ref"]
        and v["entry_refs"]==spec["observation_entry_refs"]
        and v["observation_source_refs"]==spec["observation_cache_source_refs"]
        and v["observation_cache_ref"]==spec["refs"]["observation_cache"]
        and v["registration_ref"]==cfg["registration_ref"]
        and v["producer_ref"]==cfg["preparation_producer_ref"]
        and v["observations_only"] is True and v["targets_read"] is False
        and v["checkpoint_payloads_read"] is False and v["CUDA_initialized"] is False
        and all(p.get("lossless_roundtrip_verified") is True for p in v["parents"].values()),
        "Complete fresh all40 compact cut preparation")
    return authenticate_declared_cut_preparation_lineage_v2(reader,spec,cfg,v,w)


def build_context_v1(record,spec):
    require(spec.get("phase") in ("train","evaluate"),"Declarative-preparation GPU context only")
    from arggnn import census_tsinfer_fit_v1 as fit
    require(spec.get("schema")=="H049_marker_grid_root_input_v1"
        and spec.get("hypothesis_id")=="H049" and spec.get("arm")=="joint_three_arm"
        and spec.get("root_authorizer")=="root_agent" and spec.get("status")=="active"
        and spec.get("phase") in ("prepare","train","evaluate"),"Active fresh H049 root capability")
    phase=spec["phase"];source=PREP_SOURCE if phase=="prepare" else WORKER_SOURCE
    require(spec["root_context_module_ref"]==ref(SOURCE)
        and spec["producer_ref"]==spec["main_source_ref"]==spec["worker_cli_ref"]==ref(source),
        "Canonical actual phase producer/context")
    require(json.loads(os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])==POLICY_REF,"Current980 environment")
    reader=fit.ConsumerReaderV1();fit._check_active_v1(spec)
    cfg=reader.decode_json(spec["refs"]["config"]);worker.validate_protocol(cfg,phase=phase)
    worker.bind_registration_v1(cfg,reader)
    require(spec["max_cpu_rss_bytes"]==cap_v1(cfg,"max_cpu_rss_bytes"),"Reviewed same CPU cap")
    storage=reader.decode_json(spec["refs"]["storage"]);validate_closure_v1(reader,spec,cfg,storage)
    guard_module,memo=load_guard_v1(consumer=True)
    guard_module.validate_storage_v9(spec,cfg,storage,source_ref=spec["main_source_ref"],
        config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
    worker.science()
    output=guard_module.PilotOutputV9(spec["output_directory"],storage,reader)
    runtime=reader.decode_json(spec["refs"]["runtime"])
    require(runtime["job_id"]==spec["job_id"] and runtime["device"]==spec["device"]
        and runtime.get("physical_gpu")==spec.get("physical_gpu"),"Exact actual phase runtime")
    generation=reader.decode_json(spec["refs"]["generation_config"])
    registration=reader.decode_json(spec["refs"]["registration"])
    require(registration["config_ref"]==spec["refs"]["generation_config"]
        and registration["parent_registry"]==generation["parent_registry"],"Original generation identity")
    original=reader.decode_json(cfg["original_H038_input_ref"])
    require(all(spec["refs"][k]==original["refs"][k] for k in ("registration","generation_config","observation_cache"))
        and spec["observation_entry_refs"]==original["observation_entry_refs"]
        and spec["observation_cache_source_refs"]==original["observation_cache_source_refs"]
        and set(spec["observation_entry_refs"])==set(worker.TRAIN+worker.VAL),"Unchanged original40 observations")
    preparation=completed_marker_preparation_v1(reader,spec)
    counts=reader.decode_json(cfg["count_manifest_ref"])
    require(all(counts["parents"][p]["observation_entry_ref"]==spec["observation_entry_refs"][p]
        for p in worker.TRAIN+worker.VAL),"Counts bind exact original observed entries")
    plan=count_plan_v1(counts,phase,preparation,cfg)
    require(plan["total_output_scope_upper_bound_bytes"] is not None
        and plan["total_output_scope_upper_bound_bytes"]<=cfg["output_cap_bytes"],"Reviewed complete output forecast")
    ctx=dict(root_authorizer="root_agent",admitted=True,phase=phase,arm="joint_three_arm",
        cfg=cfg,reader=reader,output=output,config_ref=spec["refs"]["config"],
        source_manifest_ref=cfg["source_manifest_ref"],registry=generation["parent_registry"],
        entry_refs=spec["observation_entry_refs"],observation_cache_ref=spec["refs"]["observation_cache"],
        observation_source_refs=spec["observation_cache_source_refs"],preparation=preparation,
        preparation_completion_ref=spec["preparation_completion_ref"],
        preparation_actual_wait_ref=spec["preparation_actual_wait_ref"])
    if phase=="prepare":
        require(spec["device"]=="cpu" and spec.get("physical_gpu") is None
            and os.environ.get("CUDA_VISIBLE_DEVICES")==""
            and "target_bindings" not in spec["refs"],"CPU-only observation capability without targets")
        def cpu_guard():
            fit._check_active_v1(spec)
            require(max(current_RSS_bytes_v1(),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                <=cfg["max_cpu_rss_bytes"],"Actual current and peak CPU preparation RSS cap")
        def before_cut(pid,bank):
            b=worker.hooks.parent_bounds_v1(bank,range(preparation["parents"][pid]["counts"]["scoring_centers"]))
            row=counts["parents"][pid]
            require(row["native_nodes"]==b["native_nodes"] and row["native_holds"]==b["native_holds"],
                "Exact loaded observed cut counts")
            estimate=b["additional_cut_preparation_work_estimate_bytes"]
            require(estimate<=cfg["cut_preparation_cap_bytes"]
                and current_RSS_bytes_v1()+estimate<=cfg["max_cpu_rss_bytes"],"Named prospective CPU cut preparation estimate")
        ctx.update(guard=cpu_guard,before_cut_build=before_cut);cpu_guard()
    else:
        from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
        lease=validate_lease();apply_and_validate_title();fit.require_logical_job_v1(lease,spec)
        require(spec["physical_gpu"] in (4,5) and lease["physical_gpu"]==spec["physical_gpu"]
            and runtime["process_title"]=="AI Auto Research Pipeline Test"
            and spec["max_cuda_reserved_bytes"]==cfg["max_cuda_reserved_bytes"],"Exact canonical GPU4/5 lease/title/cap")
        guard_module.budget_check_v9(storage,boundary=True,required_write_bytes=cfg["qualification_allowance_bytes"])
        from arggnn.biological_candidate import configure_fit
        device,qualified_runtime,qualified_lease=configure_fit("cuda",reader.decode_json(cfg["base_fit_config"]),output.path)
        import torch,torch_geometric
        require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Pinned Torch/PyG")
        require(qualified_lease["physical_gpu"]==spec["physical_gpu"],"Root-selected actual physical GPU")
        torch.cuda.set_per_process_memory_fraction(min(1.,cfg["max_cuda_reserved_bytes"]/
            torch.cuda.get_device_properties(device).total_memory),device)
        output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob("*") if p.is_file())
        require(output.used<=cfg["qualification_allowance_bytes"],"Bound actual configure_fit output")
        from arggnn import train_mutation_routed_whole_tree_gnn_v1 as original_worker
        guard=lambda:original_worker.lightweight_resources_v1(spec,qualified_lease)
        guard()
        target_metadata=reader.decode_json(spec["refs"]["target_bindings"])
        originals=reader.decode_json(original["refs"]["target_bindings"])
        ids=worker.TRAIN if phase=="train" else worker.VAL
        bindings=target_metadata["parents"]
        require(set(bindings)==set(ids) and target_metadata["outer_included"] is False
            and bindings=={p:originals["parents"][p] for p in ids},"Exact original phase-only target metadata")
        for pid,binding in bindings.items():
            require(binding["truth_npz"]["path"]==generation["output_root"]+"/"+pid+"/truth/full_truth_intervals.npz",
                "Original same-parent truth metadata path")
            provenance=reader.decode_json(binding["generation_receipt"])
            require(provenance["parent_id"]==pid
                and provenance["files"]["truth/full_truth_intervals.npz"]["sha256"]==binding["truth_npz"]["sha256"],
                "Original generation receipt binds same-parent target metadata")
        cut_preparation=completed_cut_preparation_v1(reader,spec,cfg)
        require(plan["all144_original_grid_probability_bytes"]<=cfg["RAM_prediction_cap_bytes"],"All144 dense RAM forecast")
        def before_build(pid,bank,declared):
            guard();preparation_check_v1(pid,bank,declared,cfg,counts,preparation)
        ctx.update(device=device,guard=guard,physical_gpu=spec["physical_gpu"],bindings=bindings,
            cut_preparation=cut_preparation,before_grid_build=before_build,
            cut_preparation_completion_ref=spec["cut_preparation_completion_ref"],
            cut_preparation_actual_wait_ref=spec["cut_preparation_actual_wait_ref"])
        if phase=="evaluate":ctx["train_completion_ref"]=spec["train_completion_ref"]
    output.save("started.json",dict(schema="H049_order2_started_v1",hypothesis_id="H049",phase=phase,
        input_ref=record,output_plan=plan,memo_receipt=memo.receipt,parameters_per_arm=32446,
        training_passes=30,updates_per_arm=960,updates_total=2880,checkpoint_files=18,
        all144_evaluation_seals_before_truth=True,VAL_influences_optimization=False,outer_truth_read=False))
    return ctx

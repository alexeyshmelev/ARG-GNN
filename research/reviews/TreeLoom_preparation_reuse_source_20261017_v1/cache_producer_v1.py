"""Actual admitted CPU scientific entry for one immutable observed cache entry.

An existing ROOT CPU supervisor/Writer/guard must supply this context with an
explicit reservation covering measured tensor bytes plus this codec's bound.
This entry itself has no reserve/admit/launch/close/retry or target capability.
"""
from __future__ import annotations
import gc
import os,sys
import resource
import time
from pathlib import Path
from arggnn.american_c3_gnn_dataset_v1 import load_observation
from arggnn.american_c3_common_v1 import require,ROOT,ref,bound,read,Writer
from . import optimized_preparation_v1 as preparation
from . import cache_codec_v1 as codec

HERE="research/reviews/TreeLoom_preparation_reuse_source_20261017_v1"


def _execute(cfg,writer,*,resource_check):
    """Whole-parent prepare -> bounded archive -> release -> exact replay proof."""
    require(cfg["schema"]=="ROOT_TreeLoom_observed_cache_producer_config_v1"
        and cfg["execution_enabled"]is True and cfg["parent_id"]in codec.f.TRAIN
        and cfg["truth_access_authorized"]is False
        and cfg["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json")
        and callable(resource_check),"Explicit admitted observation-only cache producer")
    item=cfg["parent_refs"]
    load_cfg=dict(native_preparation_ref=item["native_preparation_ref"],
        class_order=["EUR","EAS","AFR"],expected_haplotypes=1600,reference_counts=[200,200,200],
        input_reference_class_mask=None,input_reference_sample_indices=None)
    resource_check("before_full_observation_cache_prepare")
    start=time.monotonic()
    state=load_observation(item["observation_ref"],item["focal_registry_ref"],load_cfg)
    bundle,metadata=preparation.prepare_from_loaded_observation_v1(state,split="train",
        observation_ref=item["observation_ref"],focal_registry_ref=item["focal_registry_ref"],
        native_preparation_ref=item["native_preparation_ref"],max_nodes=400000000,
        max_holds=100000,max_episode_payload_bytes=8*2**30,resource_check=resource_check)
    del state
    cold=time.monotonic()-start
    key,identity=codec.identity_key_v1(cfg["parent_id"],item,cfg["source_manifest_ref"])
    measurement=codec.describe_v1(bundle,metadata,key,identity)[2]
    require(cfg["mode"]in("measure_only","serialize_and_verify"),"Explicit measurement or archive mode")
    if cfg["mode"]=="measure_only":
        return writer.json("completed.json",dict(
            schema="TreeLoom_observed_preparation_size_measurement_v1",
            status="measured_complete_observed_tensor_and_archive_bound_no_cache_claim",
            parent_id=cfg["parent_id"],parent_refs=item,key=key,identity=identity,
            source_manifest_ref=cfg["source_manifest_ref"],**measurement,
            cold_prepare_seconds=cold,actual_full_parent_measurement=True,
            sequence_length_bp=30000000,haplotypes=1600,native_focals=10000,
            worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            archive_materialized=False,roundtrip_verified=False,truth_opened=False,model_executed=False))
    require(measurement["tensor_nbytes"]<=cfg["max_tensor_bytes"]
        and measurement["serialized_upper_bound_bytes"]<=cfg["max_archive_bytes"]
        and measurement["per_entry_charge_upper_bound_bytes"]<=cfg["output_cap_bytes"],
        "Measured full payload fits explicit admitted archive/scope; no cache fit claim from source")
    resource_check("before_bounded_observed_archive")
    path=writer.prepare("observed_preparation.cache",measurement["serialized_upper_bound_bytes"])
    with path.open("xb")as stream:
        codec.write_archive_v1(stream,bundle,metadata,key,identity)
        stream.flush();__import__("os").fsync(stream.fileno())
    archive_ref=writer.finish(path,measurement["serialized_upper_bound_bytes"])
    del bundle
    gc.collect()
    resource_check("before_exact_observed_replay")
    replay_start=time.monotonic()
    verified_path=bound(archive_ref)
    with verified_path.open("rb")as stream:
        replay,replay_metadata=codec.restore_v1(stream,key,identity,
            max_tensor_bytes=measurement["tensor_nbytes"],
            max_archive_bytes=measurement["serialized_upper_bound_bytes"])
    replay_seconds=time.monotonic()-replay_start
    require(codec.encoded(replay_metadata)==codec.encoded(metadata),
        "Exact complete global recipe metadata replay")
    del replay
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    resource_check("after_exact_observed_replay")
    actual=dict(parent_id=cfg["parent_id"],actual_full_parent_measurement=True,
        preparation_source_manifest_ref=cfg["source_manifest_ref"],
        sequence_length_bp=30000000,haplotypes=1600,native_focals=10000,truth_opened=False,
        cold_prepare_seconds=cold,actual_verified_replay_seconds=replay_seconds,
        actual_measured_worker_peak_RSS_bytes=peak,
        actual_measured_replay_worker_peak_RSS_bytes=peak,
        actual_original_owned_RSS_cap_enforced=False,
        original_owned_RSS_cap_must_be_authenticated_from_actual_parent_wait=True,
        roundtrip_all_tensor_bytes_verified=True,**measurement)
    return writer.json("completed.json",dict(
        schema="TreeLoom_observed_preparation_cache_producer_completion_v1",
        status="completed_whole_observed_preparation_archive_and_exact_replay",
        parent_id=cfg["parent_id"],parent_refs=item,key=key,identity=identity,
        archive_ref=archive_ref,measurement=actual,source_manifest_ref=cfg["source_manifest_ref"],
        observation_only=True,query_truth_opened=False,model_executed=False,
        cache_producer_actual_wait_drain_ordinary_close_and_measurement_acceptance_still_required=True))


def execute_in_admitted_context_v1(config_ref,*,spec,current_guard,current_guard_source_ref,resource_check):
    """Original ROOT CPU-bridge signature, with real guard before Writer creation."""
    from research.reviews.frozen_TreeLoom_bounded_training_source_20261017_v1 import contract_v1 as contract
    cfg=read(config_ref);own_ref=ref(HERE+"/source_manifest.v1.json");own=read(own_ref)
    require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
        and sys.flags.optimize==0 and os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and spec["root_authorizer"]=="root_agent"and spec["execution_authorized"]is True
        and spec["GPU_authorized"]is False and spec["query_truth_access_authorized"]is False
        and spec["TEST_access_authorized"]is False and spec["GNN_optimization_authorized"]is False
        and spec["source_ref"]==ref(HERE+"/cache_producer_v1.py")
        and spec["config_ref"]==config_ref and cfg["source_manifest_ref"]==own_ref
        and current_guard_source_ref==contract.GUARD
        and Path(current_guard.__file__).resolve()==bound(current_guard_source_ref)
        and current_guard.MODE=="Census_observation_two_scope_retention_v1"
        and current_guard._SELECTOR_REF==spec["storage_policy_selector_ref"]
        and callable(resource_check),"Exact admitted CPU-only current-guard context; no bare producer bypass")
    contract.active(spec)
    review=read(spec["source_review_ref"])
    require(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"
        and review.get("blockers")==[]
        and (review.get("candidate_manifest_ref")==own_ref or review.get("source_manifest_ref")==own_ref),
        "Exact independent cache/preparation source pass")
    for path,sha in own["source_sha256"].items():contract.raw(dict(path=path,sha256=sha))
    runtime=read(spec["runtime_ref"])
    require(Path(runtime["prefix"]).resolve()==Path(sys.prefix).resolve()
        and Path(runtime["python"]).resolve()==Path(sys.executable).resolve()
        and runtime["CUDA_VISIBLE_DEVICES"]=="","Exact existing qualified CPU Torch runtime, no install")
    caps=read(spec["caps_ref"])
    for key in spec["caps_locator"]:caps=caps[key]
    require(type(caps["rss_bytes"])is int and 0<caps["rss_bytes"]<=384*2**30
        and type(caps["wall_seconds"])is int and 300<caps["wall_seconds"]<=7*86400,
        "Actual original owned CPU supervisor has explicit384GiB/finite-time-or-stricter caps")
    storage=spec["storage"];reservation=storage["reservation"]
    require(reservation["reserved_bytes"]==storage["reserved_new_bytes"]
        and reservation["reserved_bytes"]>=cfg["output_cap_bytes"]+spec["auxiliary_cap_bytes"]
        and reservation["output"]==str(ROOT/spec["output_directory"])
        and own_ref in reservation["consumer_bindings"]and len(reservation["consumer_bindings"])<=64,
        "Actual separate whole-cache+aux reservation with exact source binding, original64 ceiling")
    current_guard.validate_storage_v9(spec,dict(cfg,output_cap_bytes=reservation["reserved_bytes"]),storage,
        source_ref=spec["source_ref"],config_ref=config_ref,runtime_ref=spec["runtime_ref"])
    resource_check("before_actual_CPU_cache_writer")
    return _execute(cfg,Writer(spec,current_guard),resource_check=resource_check)

"""Source-pinned native attachment contract only; NO scientific runner/launch.
A future separately reviewed native per-parent adapter must use original author
loader cardinality/settings and the new pool lifetime. This module grants none.
"""
import math
from . import common_v1 as h
from . import pool_contract_v1 as c
SOURCE=h.HERE+"/native_attachment_seam_v1.py"
AUTHOR_MANIFEST={"path":"research/reviews/American_C3_native_consumer_source_v1/author_source_manifest.json","sha256":"bed570db438f2c4981ebf606460dc4ff72a97b148ba371e3873ab0141376736a"}
ORIGINAL_NATIVE_CHILD={"path":"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py","sha256":"f2770222f02979b6064a7f1ffea80953cd234114c96ddde0845372479292379e"}

def cardinality_contract_v1(record):
    v=h.read(record)
    h.require(v.get("schema")=="native_original_per_parent_MLM_cardinality_v1"and v.get("root_authorizer")=="root_agent"
        and v.get("original_author_loader_cardinality_verified") is True
        and v.get("author_source_manifest_ref")==AUTHOR_MANIFEST and v.get("full_epochs")==5
        and v.get("global_train_batch_size")==512 and v.get("device_train_microbatch_size")==512
        and v.get("precision")=="amp_bf16"and v.get("seed")==17 and v.get("num_workers")==8
        and v.get("objective_or_corpus_reduced") is False,"Exact author full native science, never reduced support")
    n=v["original_train_dataset_cardinality"];drop=v["original_author_drop_last"]
    h.require(type(n) is int and n>0 and type(drop) is bool,"Original actual parent-specific cardinality/drop_last")
    batches=n//512 if drop else (n+511)//512
    h.require(batches>0 and v["original_batches_per_epoch"]==batches
        and v["expected_full_optimizer_updates"]==5*batches,"Expected steps derive original actual loader cardinality times5epochs")
    author=h.read(AUTHOR_MANIFEST);h.require(v["original_author_loader_source_ref"]in author["refs"],"Original actual loader belongs to exact pinned author source closure");h.raw(ORIGINAL_NATIVE_CHILD);h.raw(v["original_preparation_ref"]);h.raw(v["original_cardinality_measurement_ref"]);h.raw(v["original_author_loader_source_ref"])
    h.require(v.get("parent_id") and v.get("original_actual_parent_id")==v["parent_id"],"Same actual full parent, no VAL000 budget transplant")
    return dict(parent_id=v["parent_id"],epochs=5,batches_per_epoch=batches,expected_optimizer_updates=5*batches,cardinality_ref=record)

def attachment_interface_v1(capability_ref,cardinality_ref):
    cap=c.validate_cap_v1(capability_ref);science=cardinality_contract_v1(cardinality_ref)
    native=[(n,j)for n,j in cap["jobs"].items()if j["worker_kind"]=="native"]
    h.require(len(native)==1 and native[0][1]["execution_enabled"] is False,"Current seam is explicitly not a native execution adapter")
    return dict(schema="native_postdrain_attachment_interface_v1",hook_ref=h.ref(SOURCE),pool_capability_ref=capability_ref,physical_gpu=0,gpu_uuid=c.UUID,science= science,allocator_bytes=32*2**30,margin_bytes=2*2**30,headroom_bytes=4*2**30,attachment_required="Real inherited new-owner physical FD plus authenticated per-slot logical reservation and PID/start-owned lifetime",execution_authorized=False,implemented_scientific_runner=False,original_pool_per_job_identity_ceiling=512,native_larger_roster_lifecycle_requires_separate_review=True,missing=["Complete original per-parent author preparation/runner source integration","Exact optimizer/Trainer/saved checkpoint full-budget validator using this derived expected count","Separate current storage/source/deadline/runtime/owned-lifecycle admission and independent whole join review","New inactive capability/schema/source successor enabling the native slot; this sealed capability cannot be toggled"])

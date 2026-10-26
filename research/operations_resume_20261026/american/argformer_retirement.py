
"""Retire exact new corrected scratch after real prediction and producer waits.

Historical files and reused TEST000/001 preparations are never deletion targets.
The surviving metadata chain remains sufficient for the original all45 barrier.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import stat
from arggnn.benchmarking.paper_american import protocol as p

CAMPAIGN = p.ROOT / "experiments/results/paper_american_20261024"
PROTECTED_REUSED = {"argformer_american_test_000", "argformer_american_test_001"}

RESUMED_PARENT = "argformer_american_validation_001"
RESUME_LEAF = "resume_20261026_batch119500"
PRESERVED_CHECKPOINT_BYTES = 2014910662

def producer_output(base, stage):
    native = base / stage / "native"
    return native / RESUME_LEAF if stage == "mlm" else native

def retained_nonmodel_bound():
    return 128 * 2**20 + PRESERVED_CHECKPOINT_BYTES

def preserved_checkpoint(request):
    need(request["parent_id"] == RESUMED_PARENT,
         "Only the stopped validation001 parent uses this retirement adapter")
    evidence = p.read(request["resume_evidence_ref"])
    reference = evidence["checkpoint_ref"]
    expected = CAMPAIGN / RESUMED_PARENT / "mlm/native/checkpoints/ep2-ba119500-rank0.pt"
    need(evidence["schema"] == "american_argformer_full_state_resume_20261026"
         and p.path(reference["path"]) == expected
         and expected.stat().st_size == PRESERVED_CHECKPOINT_BYTES,
         "Exact preserved interrupted checkpoint and storage charge required")
    return reference

def need(ok, message):
    if not ok:
        raise ValueError(message)

def _ordinary(value):
    raw = Path(value)
    raw = p.ROOT / raw if not raw.is_absolute() else raw
    need(".." not in raw.parts and raw.resolve() == raw
         and raw.is_relative_to(CAMPAIGN), "Exact ordinary corrected-campaign path required")
    for part in [raw, *raw.parents]:
        if part == p.ROOT:
            break
        need(not part.is_symlink(), "No symlink may appear in a retirement target path")
    need(raw.is_file() and stat.S_ISREG(raw.stat().st_mode), "Ordinary retirement payload required")
    return raw

def _hash(value, check):
    path = _ordinary(value)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16*2**20), b""):
            check("corrected_retirement_hash")
            h.update(block)
    return dict(path=str(path.relative_to(p.ROOT)), sha256=h.hexdigest())

def _wait(native):
    need(native.name == "native", "ROOT producer native subdirectory required")
    wait = p.read(p.ref(native.parent / "actual_wait.json"))
    need(wait.get("status") == "succeeded" and wait.get("actual_returncode") == 0
         and wait.get("actual_wait") is True and wait.get("owned_process_group_absent") is True,
         "Genuine successful ROOT direct wait and absent producer group required")
    return p.ref(native.parent / "actual_wait.json")

def _chain(request):
    parent = request["parent_id"]
    need(parent.startswith("argformer_american_") and "/" not in parent,
         "Explicit registered parent filename required")
    base = CAMPAIGN / parent
    prep_ref = request.get("preparation_ref") or p.ref(request["preparation_completion_path"])
    prep = p.read(prep_ref)
    records = {}
    for stage in ("mlm", "contrastive", "predict"):
        reference = p.ref(request[stage + "_completion_path"])
        need(p.path(reference["path"]) == base/stage/"native/completed.json",
             "Only new corrected-campaign producer completion paths required")
        value = p.read(reference)
        need(value.get("schema") == "paper_american_native_completion_v1"
             and value.get("parent_id") == parent and value.get("stage") == stage
             and value.get("variant") == "paper_aligned"
             and value.get("preparation_ref") == prep_ref
             and value.get("full_budget_completed") is True
             and value.get("query_ancestry_truth_opened") is False,
             "One genuine complete corrected native chain required")
        executed = p.read(value["self_spec_ref"])
        need(executed["parent_id"] == parent and executed["stage"] == stage
             and executed["variant"] == "paper_aligned"
             and p.path(executed["output_directory"]) == producer_output(base, stage),
             "Exact corrected executed-spec lineage required")
        if stage != "mlm":
            need(executed["checkpoint_completion_ref"] == records[
                "mlm" if stage == "contrastive" else "contrastive"]["reference"],
                "Exact sequential corrected checkpoint lineage required")
        records[stage] = dict(reference=reference, value=value,
                              executed_spec_ref=value["self_spec_ref"],
                              wait_ref=_wait(base/stage/"native"))
    prediction = records["predict"]["value"]
    need(prediction["records"] == 10000 and prediction["class_order"] == p.CLASSES
         and prediction["focal_registry_ref"] == prep["focals_ref"]
         and prediction["query_truth_used_for_fit"] is False
         and prediction["evaluator"] == "reference_PCA10_nearest_class_centroid",
         "Complete original focal prediction seal required before retirement")
    # Revalidate every prediction identity and probability transport before deletion.
    from arggnn.american_c3_evaluate_v1 import registry, verify_prediction
    import numpy as np
    with np.load(p.path(prediction["prediction_ref"]["path"]), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    need(p.ref(prediction["prediction_ref"]["path"]) == prediction["prediction_ref"],
         "Exact surviving prediction hash required")
    verify_prediction(arrays, registry(prep["focals_ref"]))
    need(p.ref(prediction["reference_pca_ref"]["path"]) == prediction["reference_pca_ref"],
         "Exact surviving reference PCA sufficient statistics required")
    new_prep = request.get("new_preparation_output_directory")
    if new_prep is not None:
        need(parent not in PROTECTED_REUSED and p.path(new_prep) == base/"prepare/native"
             and p.path(prep_ref["path"]) == base/"prepare/native/preparation_completed.json"
             and p.path(prep["disposable_scratch_directory"]) == base/"prepare/native/scratch",
             "Only an explicitly new corrected preparation may be retired")
        records["prepare"] = dict(reference=prep_ref, wait_ref=_wait(base/"prepare/native"))
    else:
        need(parent in PROTECTED_REUSED,
             "Missing new-preparation ownership is permitted only for the two protected reused parents")
    need(prep["parent_id"] == parent and prep["query_ancestry_truth_opened"] is False,
         "Same-parent sanitized preparation required")
    return base, prep_ref, prep, records

def _compact(torch, checkpoint_path, destination, maximum_bytes, check):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    original = saved["state"]["model"]
    need(isinstance(original, dict) and original
         and all(torch.is_tensor(value) and value.device.type == "cpu"
                 for value in original.values()), "All original saved CPU model tensors required")
    storage_bytes = {}
    for value in original.values():
        storage = value.untyped_storage()
        storage_bytes[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    need(sum(storage_bytes.values()) + 32*2**20 <= maximum_bytes,
         "Source-derived complete model retention bound exceeded")
    payload = dict(schema="paper_american_lossless_final_model_v1",
                   state=dict(model=original), quantized=False,
                   original_checkpoint_ref=_hash(checkpoint_path, check))
    with destination.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    need(destination.stat().st_size <= maximum_bytes, "Bounded complete model retention required")
    actual = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)["state"]["model"]
    need(set(actual) == set(original), "Lossless saved tensor names required")
    for key, value in original.items():
        need(actual[key].dtype == value.dtype and actual[key].shape == value.shape
             and torch.equal(actual[key], value), "Lossless exact final model tensor required: " + key)
        check("corrected_retirement_verify_lossless_model")
    del actual, saved
    return dict(model_ref=_hash(destination, check), unique_tensor_storage_bytes=sum(storage_bytes.values()),
                all_model_tensors_bitwise_preserved=True, tensor_count=len(original))

def run_task(request, *, resource_check):
    need(callable(resource_check), "ROOT resource callback required")
    need(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "Retirement is a CPU-only task")
    preserved = preserved_checkpoint(request)
    base, prep_ref, prep, records = _chain(request)
    out = p.path(request["output_directory"])
    need(out == base/"retire/native" and not out.exists(), "Fresh dedicated corrected retirement output required")
    maximum = request["max_retained_model_bytes"]
    observed = p.read(prep["observation_ref"])
    unary = p.read(observed["unary_preprocessing_ref"])
    vocabulary_upper_bound = ((unary["collapsed_num_nodes"]+4+7)//8)*8
    need(maximum == 4*(100000000+768*vocabulary_upper_bound)+32*2**20,
         "Exact current-parent node-derived complete FP32 model bound required")
    need(type(maximum) is int and 0 < maximum <= 2*2**30, "Explicit finite source-derived model storage cap required")
    out.mkdir(parents=True)
    import torch
    torch.set_num_threads(8)
    contrastive = records["contrastive"]["value"]
    compact = _compact(torch, _ordinary(contrastive["checkpoint_ref"]["path"]),
                       out/"final_model.pt", maximum, resource_check)
    # Preserve canonical source/config/vocabulary/focal/provenance records in place.
    retained = [preserved, request["resume_evidence_ref"], prep_ref, prep["vocab_ref"], prep["observation_ref"], prep["focals_ref"],
                compact["model_ref"], records["predict"]["value"]["prediction_ref"],
                records["predict"]["value"]["reference_pca_ref"]]
    for item in records.values():
        retained.append(item["reference"])
        retained.append(item["wait_ref"])
        if "executed_spec_ref" in item:
            retained.append(item["executed_spec_ref"])
    for stage in ("mlm", "contrastive"):
        retained.append(records[stage]["value"]["resolved_config_ref"])
    for reference in retained:
        need(p.ref(reference["path"]) == reference, "Every retained provenance file must survive unchanged")
    delete = {}
    if request.get("new_preparation_output_directory") is not None:
        bulk = [*prep["native_dataset_refs"].values(), *prep["full_path_id_order_refs"].values(),
                *prep["reference_label_refs"].values(), prep["sequence_to_tree_indices_ref"],
                prep["native_generated_focals_ref"]]
        scratch = base/"prepare/native/scratch"
        for reference in bulk:
            target = _ordinary(reference["path"])
            need(target.is_relative_to(scratch) and reference != prep["vocab_ref"],
                 "Only enumerated new raw corpus/sidecar data may be retired")
            need(_hash(target, resource_check) == reference, "Exact disposable corpus hash required")
            delete[str(target)] = reference
    checkpoint_metadata = {}
    for stage in ("mlm", "contrastive"):
        native = base/stage/"native"
        value = records[stage]["value"]
        executed = p.read(value["self_spec_ref"])
        need(_hash(value["checkpoint_ref"]["path"], resource_check) == value["checkpoint_ref"],
             "Exact full-budget original checkpoint required")
        checkpoint_metadata[stage] = {key: item for key, item in value.items()
                                      if key not in ("self_spec_ref",)}
        directory = producer_output(base, stage)/"checkpoints"
        for target in directory.iterdir():
            if target.is_symlink():
                # Keep symlink metadata; never follow it or delete through it.
                continue
            need(target.is_file() and target.suffix == ".pt",
                 "Only native checkpoint tensors may be retired; unknown payload blocks")
            delete[str(target)] = _hash(target, resource_check)
    inventory = []
    for value, reference in sorted(delete.items()):
        path = _ordinary(value)
        need(not any(p.path(item["path"]) == path for item in retained), "Retained records cannot be deletion targets")
        info = path.stat()
        inventory.append(dict(**reference, bytes=info.st_size, mtime_ns=info.st_mtime_ns))
    receipt = dict(schema="paper_american_corrected_retirement_inventory_v1",
                   parent_id=request["parent_id"], preparation_ref=prep_ref,
                   producers=records, retained_refs=retained, files=inventory,
                   checkpoint_counters_and_metadata=checkpoint_metadata,
                   lossless_final_model=compact, protected_reused_preparation_untouched=True,
                   source_ref=p.ref(__file__), request=request)
    total_parent_bytes = sum(path.stat().st_size for path in base.rglob("*")
                             if path.is_file() and not path.is_symlink())
    remaining_nonmodel_bytes = total_parent_bytes - sum(item["bytes"] for item in inventory) - p.path(compact["model_ref"]["path"]).stat().st_size
    need(remaining_nonmodel_bytes <= retained_nonmodel_bound()-8*2**20,
         "Surviving metadata plus preserved interrupted checkpoint exceed explicit retention bound")
    receipt["predicted_retained_nonmodel_bytes_before_receipt"] = remaining_nonmodel_bytes
    inventory_ref = p.save(receipt, out/"retirement_inventory.json")
    # The root scheduler has already reaped every producer. Repeat its exact waits.
    _chain(request)
    for item in inventory:
        resource_check("before_corrected_new_scratch_unlink")
        target = _ordinary(item["path"])
        info = target.stat()
        need(info.st_size == item["bytes"] and info.st_mtime_ns == item["mtime_ns"]
             and _hash(target, resource_check) == {"path": item["path"], "sha256": item["sha256"]},
             "Disposable identity changed after inventory")
        target.unlink()
    for reference in retained:
        need(p.ref(reference["path"]) == reference, "Retained final data/provenance changed")
    result = dict(schema="paper_american_corrected_retirement_completion_v1",
                  parent_id=request["parent_id"], inventory_ref=inventory_ref,
                  final_model_ref=compact["model_ref"], all_model_tensors_bitwise_preserved=True,
                  retired_bytes=sum(item["bytes"] for item in inventory),
                  retired_file_count=len(inventory), retained_nonmodel_cap_bytes=retained_nonmodel_bound(),
                  preserved_interrupted_checkpoint_ref=preserved,
                  preserved_interrupted_checkpoint_bytes=PRESERVED_CHECKPOINT_BYTES,
                  prediction_completion_ref=records["predict"]["reference"],
                  historical_files_deleted=False, reused_TEST000_TEST001_preparations_deleted=False,
                  original_sources_modified=False, all45_prediction_barrier_inputs_preserved=True)
    return dict(completion_ref=p.save(result, out/"retirement_completed.json"), **result)

def factory(**request):
    return dict(module=__name__, function="run_task", spec=request,
                python=str(p.NATIVE_PYTHON.relative_to(p.ROOT)), source_refs=[p.ref(__file__), p.ref(p.__file__)])

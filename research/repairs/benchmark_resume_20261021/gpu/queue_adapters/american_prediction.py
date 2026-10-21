"""Supervised American fixed-final TreeLoom fit and zero-truth prediction.

These are in-worker callables. ROOT owns the process, GPU5 lease, actual wait,
RSS/wall/storage admission and queue. Import/planning opens metadata/source only.
"""
from __future__ import annotations
from collections import OrderedDict
from . import common
from research.repairs.benchmark_recovery_20261020.training.candidate_gru import install_chunked_gru
from research.repairs.benchmark_recovery_20261020.training.execution import StagedExecutionV1
from research.repairs.benchmark_recovery_20261020.training.decoder import geometry_from_spans
from research.repairs.benchmark_recovery_20261020.runtime.binding import ReadableTreeLoom
import copy
import os
from pathlib import Path
import tempfile

from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import treeloom_fresh as fresh
from arggnn.benchmarking import treeloom_training as training
from arggnn.benchmarking import american_preparation as generation

FIT_SCHEMA = "treeloom_supervised_fixed_final_fit_v1"
PREDICTION_SCHEMA = "treeloom_fresh_heldout_prediction_v1"
IDENTITY = ("record_ids", "sample_nodes", "sample_indices", "positions_bp0", "parent_id", "class_order")


def _need(ok, message):
    if not ok:
        raise c.ContractError(message)


def _public_ref(ref):
    fresh._ref(ref)
    _need(not any(part.lower() in {"private", "truth", "oracle", "census"}
                  for part in Path(ref["path"]).parts), "Prediction accepts public observed references only")
    return ref


def requirements():
    result = training.requirements()
    result = dict(result, source_refs=[*result["source_refs"], c.file_ref(__file__),
                                      c.file_ref(generation.__file__)],
                  checkpoint_policy="exact_saved_epoch30_600updates_all49Adam_states",
                  prediction_query_truth_access=False)
    return result


def _completion_binding(registry, checkpoint_ref, training_completion_ref):
    _public_ref(checkpoint_ref)
    _public_ref(training_completion_ref)
    complete = c.read_ref(training_completion_ref)
    _need(complete.get("schema") == "treeloom_fresh_full_training_completion_v1"
          and complete.get("status") == "completed_full20parents30passes600updates"
          and complete.get("parent_registry_ref") == registry.reference
          and complete.get("checkpoint_ref") == checkpoint_ref
          and complete.get("selected_epoch") == 30
          and complete.get("complete_parent_updates") == 600
          and complete.get("completed_epochs") == 30
          and complete.get("checkpoint_selection") == "fixed_final_epoch30"
          and complete.get("final_saved_optimizer_parameter_states") == 49
          and complete.get("final_saved_optimizer_steps") == [600]
          and complete.get("VAL_truth_read") is False and complete.get("TEST_truth_read") is False,
          "Genuine same-registry final30epoch600update training completion required")
    barrier_ref = _public_ref(complete["observation_barrier_ref"])
    barrier = c.read_ref(barrier_ref)
    _need(barrier.get("parent_registry_ref") == registry.reference
          and barrier.get("train_parent_ids") == list(registry.ids("train"))
          and set(barrier.get("parents", {})) == set(registry.ids("train"))
          and barrier.get("all20_observations_sealed_before_truth") is True
          and barrier.get("private_truth_opened") is False,
          "All20 original observed TRAIN identities must precede fitting")
    return complete


def _heldout_bindings(registry, heldout_inputs):
    _need(type(heldout_inputs) is dict and heldout_inputs, "Actual prepared held-out inputs required")
    for parent, item in heldout_inputs.items():
        # Deny TRAIN, unknown IDs and extra/truth keys before any path lookup.
        split = registry.split_for(parent)
        _need(split in ("validation", "test") and type(item) is dict
              and set(item) == set(fresh.INPUT_REFS), "Only registered held-out observation/focal refs; no truth key")
        for ref in item.values():
            _public_ref(ref)
        observed = c.read_ref(item["observation_ref"])
        prepared = c.read_ref(item["native_preparation_ref"])
        _need(observed.get("parent_id") == prepared.get("parent_id") == parent
              and observed.get("split") == {"validation": "val", "test": "test"}[split]
              and observed.get("sequence_length_bp") == 30000000
              and observed.get("class_order") == prepared.get("class_order") == list(fresh.CLASSES)
              and observed.get("genealogy_kind") == prepared.get("genealogy_kind")
              and observed.get("query_ancestry_truth_in_inputs") is False
              and prepared.get("query_ancestry_truth_opened") is False
              and prepared.get("observation_ref") == item["observation_ref"]
              and prepared.get("focals_ref") == item["focal_registry_ref"],
              "Exact full held-out observation and same-parent focal binding required")


def build_prediction_spec(*, parent_registry_ref, checkpoint_ref, training_completion_ref,
                          heldout_inputs, output_directory, max_rss_bytes,
                          preparation_headroom_bytes, prediction_headroom_bytes,
                          observation_cache_bytes=0, max_output_bytes=64*2**20):
    registry = fresh.FreshParentRegistry.from_ref(parent_registry_ref)
    _completion_binding(registry, checkpoint_ref, training_completion_ref)
    _heldout_bindings(registry, heldout_inputs)
    out = c.contained_path(output_directory)
    _need(out.resolve() == out and out.is_relative_to(c.ROOT / "experiments/results")
          and not out.exists(), "Fresh dedicated held-out prediction output required")
    memory = training._memory_envelope(
        max_rss_bytes=max_rss_bytes, preparation_headroom_bytes=preparation_headroom_bytes,
        training_headroom_bytes=prediction_headroom_bytes, observation_cache_bytes=observation_cache_bytes)
    memory["prediction_headroom_bytes"] = memory.pop("training_headroom_bytes")
    _need(type(max_output_bytes) is int and max_output_bytes >= len(heldout_inputs)*2**20 + 4*2**20,
          "Bounded complete prediction arrays and receipts must fit output cap")
    return dict(schema=PREDICTION_SCHEMA, parent_registry_ref=copy.deepcopy(parent_registry_ref),
                checkpoint_ref=copy.deepcopy(checkpoint_ref),
                training_completion_ref=copy.deepcopy(training_completion_ref),
                heldout_inputs=copy.deepcopy(heldout_inputs), output_directory=str(out.relative_to(c.ROOT)),
                memory_envelope=memory, observation_cache_bytes=observation_cache_bytes,
                max_output_bytes=max_output_bytes, preparation_limits=copy.deepcopy(training.LIMITS),
                requirements=requirements(), query_ancestry_truth_opened=False,
                checkpoint_selection="fixed_final_epoch30")


def _validate_spec(spec):
    memory = spec["memory_envelope"]
    expected = build_prediction_spec(
        parent_registry_ref=spec["parent_registry_ref"], checkpoint_ref=spec["checkpoint_ref"],
        training_completion_ref=spec["training_completion_ref"], heldout_inputs=spec["heldout_inputs"],
        output_directory=spec["output_directory"], max_rss_bytes=memory["max_rss_bytes"],
        preparation_headroom_bytes=memory["preparation_headroom_bytes"],
        prediction_headroom_bytes=memory["prediction_headroom_bytes"],
        observation_cache_bytes=spec["observation_cache_bytes"], max_output_bytes=spec["max_output_bytes"])
    _need(spec == expected, "Exact source/identity-bound held-out prediction specification required")
    return fresh.FreshParentRegistry.from_ref(spec["parent_registry_ref"])


def validate_saved_checkpoint(saved, model, registry, completion):
    """Validate actual CPU tensors/counters before moving saved model to GPU."""
    import torch
    _need(isinstance(saved, dict)
          and saved.get("schema") == "treeloom_fresh_full_checkpoint_v1"
          and saved.get("epoch") == 30 and saved.get("complete_parent_updates") == 600
          and saved.get("model_seed") == 70001
          and saved.get("parent_registry_ref") == registry.reference
          and saved.get("observation_barrier_ref") == completion["observation_barrier_ref"]
          and saved.get("settings") == training.SETTINGS
          and saved.get("contract") == model.checkpoint_contract()
          and saved.get("checkpoint_selection") == "fixed_final_epoch30"
          and saved.get("VAL_truth_read") is False and saved.get("TEST_truth_read") is False,
          "Exact genuine fixed-final30epoch600update saved TreeLoom contract required")
    expected, state = model.state_dict(), saved["model_state"]
    _need(isinstance(state, dict) and set(state) == set(expected), "All exact saved model tensors required")
    for key, wanted in expected.items():
        actual = state[key]
        _need(torch.is_tensor(actual) and actual.device.type == "cpu"
              and actual.dtype == wanted.dtype == torch.float32 and actual.shape == wanted.shape
              and bool(torch.isfinite(actual).all()), "Finite exact FP32 saved model tensor: " + key)
    optimizer = saved["optimizer_state"]
    groups, states = optimizer["param_groups"], optimizer["state"]
    _need(len(groups) == 1 and len(states) == 49, "All49 saved Adam parameter states required")
    group = groups[0]
    ids = group["params"]
    _need(len(ids) == 49 and len(set(ids)) == 49 and set(states) == set(ids)
          and group["lr"] == .001 and group["weight_decay"] == .0001
          and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
          and group.get("maximize", False) is False and group.get("amsgrad", False) is False,
          "Original complete49-parameter saved AdamW state required")
    parameters = list(model.parameters())
    _need(len(parameters) == 49 and sum(p.numel() for p in parameters) == 32132,
          "Original32132-parameter49-tensor model required")
    for index, parameter in zip(ids, parameters):
        item = states[index]
        step = item["step"].item() if torch.is_tensor(item["step"]) else item["step"]
        _need(type(step) in (int, float) and step == 600, "All49 actual saved Adam counters must equal600")
        for name in ("exp_avg", "exp_avg_sq"):
            value = item[name]
            _need(torch.is_tensor(value) and value.device.type == "cpu"
                  and value.dtype == torch.float32 and value.shape == parameter.shape
                  and bool(torch.isfinite(value).all()), "Exact finite saved Adam moments required")
        _need(bool((item["exp_avg_sq"] >= 0).all()), "Nonnegative saved Adam second moments required")
    model.load_state_dict(state, strict=True)
    return model



def _read_prediction_focals(ref):
    import zipfile
    from arggnn.american_c3_evaluate_v1 import registry
    path = generation._path(_public_ref(ref)["path"])
    _need(generation._reference(path) == ref, "Exact original focal registry SHA required")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        required = {key + ".npy" for key in IDENTITY}
        allowed = required | {"first_occurrence_intervals_bp.npy"}
        _need(required <= set(names) <= allowed and len(names) == len(set(names))
              and sum(item.file_size for item in archive.infolist()) <= 32*2**20,
              "Observed focal registry cannot contain truth/label or extra arrays")
    return registry(ref)


def run_predictions(spec, *, resource_check):
    registry = _validate_spec(spec)
    torch, check, writer = common._setup_gpu(spec, resource_check)
    runtime = ReadableTreeLoom(registry)
    model = runtime.build_model()
    completion = _completion_binding(registry, spec["checkpoint_ref"], spec["training_completion_ref"])
    checkpoint_path = c.contained_path(spec["checkpoint_ref"]["path"])
    _need(checkpoint_path.stat().st_size <= 8*2**20
          and generation._reference(checkpoint_path) == spec["checkpoint_ref"], "Exact bounded final checkpoint SHA required")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    validate_saved_checkpoint(saved, model, registry, completion)
    del saved
    model = install_chunked_gru(model.to(device="cuda:0", dtype=torch.float32))
    model.eval()
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from arggnn.american_c3_gnn_targets_v1 import identity_sha
    cache, predictions = OrderedDict(), {}
    def event(name, value):
        check("TreeLoom_prediction_" + name)
    for parent in (*registry.ids("validation"), *registry.ids("test")):
        if parent not in spec["heldout_inputs"]:
            continue
        item = spec["heldout_inputs"][parent]
        split = registry.split_for(parent)
        def prepare():
            state = load_observation(item["observation_ref"], item["focal_registry_ref"],
                dict(native_preparation_ref=item["native_preparation_ref"],
                     class_order=list(fresh.CLASSES), expected_haplotypes=1600,
                     reference_counts=[200, 200, 200], input_reference_class_mask=None,
                     input_reference_sample_indices=None))
            return runtime.prepare_observation(state, split=split, resource_check=check,
                                               **item, **spec["preparation_limits"])
        bundle, metadata = training._load_cached_parent(
            parent, cache, prepare, observation_cache_bytes=spec["observation_cache_bytes"],
            memory_envelope=spec["memory_envelope"], rss_bytes=training._current_rss_bytes,
            tensor_bytes=lambda value: training._cpu_bytes(value, torch), resource_check=check, event=event)
        training._admit_parent_headroom(
            cache, memory_envelope=spec["memory_envelope"], stage="prediction",
            rss_bytes=training._current_rss_bytes, resource_check=check, event=event, keep_parent=parent)
        focals = _read_prediction_focals(item["focal_registry_ref"])
        _need(bundle.parent_id == parent and bundle.sequence_length_bp == 30000000
              and all(getattr(bundle, key) == item[key] for key in fresh.INPUT_REFS)
              and focals["parent_id"].item() == parent
              and focals["class_order"].tolist() == list(fresh.CLASSES)
              and len(focals["record_ids"]) == 10000
              and identity_sha(focals) == bundle.focal_identity_sha256
              and metadata["private_truth_opened"] is False,
              "Complete same-parent original focal/sample/class support required")
        logits = _predict_logits(runtime, model, bundle, metadata, resource_check=check)
        _need(logits.shape == (10000, 3) and logits.dtype == torch.float32
              and bool(torch.isfinite(logits).all()), "All10000 finite original FP32 logits required")
        probabilities = logits.softmax(-1).to("cpu").numpy()
        prediction_ref = writer.npz("predictions/" + parent + ".npz",
                                   **{key: focals[key] for key in IDENTITY}, probabilities=probabilities)
        receipt = dict(schema="treeloom_fresh_prediction_seal_v1", method="treeloom",
            parent_id=parent, split=split, parent_registry_ref=registry.reference,
            checkpoint_ref=spec["checkpoint_ref"], training_completion_ref=spec["training_completion_ref"],
            **item, prediction_ref=prediction_ref, focal_identity_sha256=bundle.focal_identity_sha256,
            records=10000, class_order=list(fresh.CLASSES), sequence_length_bp=30000000,
            all_native_trees_nodes_sites_samples_retained=True,
            query_ancestry_truth_opened=False, checkpoint_selection="fixed_final_epoch30")
        predictions[parent] = writer.json("seals/" + parent + ".json", receipt)
        del logits, probabilities, focals, bundle, metadata
        check("TreeLoom_heldout_parent_sealed")
    torch.cuda.synchronize(0)
    result = dict(schema="treeloom_fresh_heldout_prediction_completion_v1",
        status="completed_all_requested_heldout_prediction_seals",
        parent_registry_ref=registry.reference, checkpoint_ref=spec["checkpoint_ref"],
        training_completion_ref=spec["training_completion_ref"], prediction_seal_refs=predictions,
        query_ancestry_truth_opened=False, checkpoint_selection="fixed_final_epoch30",
        actual_worker_wait_and_owned_cleanup_still_required=True)
    completion_ref = writer.json("predictions_completed.json", result)
    _need(not writer.pending and set(predictions) == set(spec["heldout_inputs"]),
          "Every requested original held-out parent must actually finish")
    return dict(result, completion_ref=completion_ref)



def _predict_logits(runtime, model, bundle, metadata, *, resource_check):
    import torch
    split = runtime._check_bundle(model, bundle, metadata)
    _need(split in ("validation", "test") and callable(resource_check),
          "Only registered held-out original observations are accepted")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        executor = StagedExecutionV1(model, bundle, metadata, resource_check=resource_check,
                                     frozen_module=runtime.frozen)
        context = executor.context_forward()
        geometry = geometry_from_spans(bundle.bank.spans).to(device)
        output = torch.empty((bundle.focal_hold.numel(), 3), dtype=torch.float32)
        seen = 0
        for first in range(0, context.shape[1], executor.hap_tile):
            last = min(first+executor.hap_tile, context.shape[1])
            rows = torch.nonzero((bundle.focal_sample_column >= first)
                                 & (bundle.focal_sample_column < last), as_tuple=False).flatten()
            if not len(rows):
                continue
            resource_check("American_full_sequence_prediction_tile")
            tile = context[:, first:last].to(device).contiguous()
            forward, _ = model.scan_context(tile, geometry, direction="forward")
            reverse, _ = model.scan_context(tile, geometry, direction="reverse")
            holds = bundle.focal_hold[rows].to(device)
            columns = (bundle.focal_sample_column[rows]-first).to(device)
            logits = model.classifier(torch.cat((tile[holds, columns], forward[holds, columns],
                                                 reverse[holds, columns]), -1))
            output.index_copy_(0, rows, logits.to("cpu"))
            seen += len(rows)
        _need(seen == len(output) and bool(torch.isfinite(output).all()),
              "Every original held-out sampled row requires finite logits")
        return output


def prediction_factory(**kwargs):
    spec = build_prediction_spec(**kwargs)
    return common.descriptor("american_prediction", "run_predictions", spec,
                             spec["requirements"]["source_refs"])

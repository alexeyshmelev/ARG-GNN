"""Exercise the unchanged GST fit loop across synthetic update 79 of 90."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
from research.repairs.benchmark_recovery_20261025.gst import workflow as workflow
from research.repairs.benchmark_recovery_20261025.gst import checkpoint as checkpoint
from research.repairs.benchmark_recovery_20261025.gst.smoke import synthetic_state
from arggnn.benchmarking.paper_campaign import runtime as runtime

class InterruptedAtBoundary(Exception):
    pass


def exact(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right), "Tensor mismatch"
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert left == right, (left, right)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = runtime.ROOT / "research/resume_no_timeouts_20261027/gst/smoke_v1"
    root.mkdir(exist_ok=False)
    states, records = {}, {}
    for index, connected in enumerate((17, 18, 19)):
        parent = "synthetic_" + str(index)
        states[parent], records[parent] = synthetic_state(connected)
    plan = dict(train_parent_ids=list(states))
    plan_ref = runtime.save(root / "synthetic_plan.json", plan)
    inputs = {parent: dict(synthetic_only=True) for parent in states}
    labels = {parent: np.arange(len(records[parent]["record_ids"]), dtype=np.int64) % 6 for parent in states}
    proof = {parent: dict(trees=value["ts"].num_trees, sites=value["ts"].num_sites) for parent, value in states.items()}
    original_save = checkpoint.save_latest
    started = time.monotonic()
    boundary_ref = None

    def interrupt_after_save(output, payload, guard):
        latest = original_save(output, payload, guard)
        if payload["optimizer_updates"] == 79:
            raise InterruptedAtBoundary("Deliberate synthetic interruption after committed update79")
        return latest

    def fit(name, resume=None):
        return workflow.fit(plan_ref, inputs, str(root / name), device="cpu", threads=1,
            max_sites=1000, max_metadata_bytes=1024*1024,
            resource_check=lambda stage: None, execution_check=lambda: None,
            write_headroom_bytes=4*1024*1024, resume_checkpoint_ref=resume)

    with patch.object(workflow, "checked_plan", return_value=plan), \
         patch.object(workflow, "load_parents", return_value=(states, records, proof)), \
         patch.object(workflow, "_labels", return_value=labels):
        full_ref = fit("uninterrupted")
        try:
            with patch.object(checkpoint, "save_latest", side_effect=interrupt_after_save):
                fit("interrupted")
        except InterruptedAtBoundary:
            pass
        else:
            raise AssertionError("Expected exact boundary interruption")
        boundary_ref = runtime.ref(root / "interrupted/recovery_latest.pt")
        boundary = torch.load(runtime.path(boundary_ref["path"]), map_location="cpu", weights_only=True)
        assert boundary["optimizer_updates"] == 79 and boundary["progress"]["epoch"] == 27
        assert len(boundary["progress"]["completed_parents"]) == 1
        resumed_ref = fit("resumed", boundary_ref)
    full = runtime.read(full_ref)
    resumed = runtime.read(resumed_ref)
    full_saved = torch.load(runtime.path(full["checkpoint_ref"]["path"]), map_location="cpu", weights_only=True)
    resumed_saved = torch.load(runtime.path(resumed["checkpoint_ref"]["path"]), map_location="cpu", weights_only=True)
    exact(full_saved["model_state"], resumed_saved["model_state"])
    exact(full_saved["optimizer_state"], resumed_saved["optimizer_state"])
    full_latest = torch.load(root / "uninterrupted/recovery_latest.pt", map_location="cpu", weights_only=True)
    resumed_latest = torch.load(root / "resumed/recovery_latest.pt", map_location="cpu", weights_only=True)
    exact(full_latest["random_state"], resumed_latest["random_state"])
    for key in ("epoch", "completed_parents", "counts", "completed_epoch_totals"):
        exact(full_latest["progress"][key], resumed_latest["progress"][key])
    for key in ("optimizer_updates", "every_optimizer_tensor_updates", "total_native_tree_visits", "total_native_edge_visits", "total_native_site_visits", "settings", "model_spec", "parameters"):
        exact(full[key], resumed[key])
    def scientific_history(value):
        return [{key: item for key, item in row.items() if key != "elapsed_seconds"} for row in value]
    exact(scientific_history(full["history"]), scientific_history(resumed["history"]))
    assert len(full["history"]) == 90 and resumed["optimizer_updates"] == 90
    assert set(resumed["every_optimizer_tensor_updates"]) == {90}
    assert not torch.cuda.is_initialized()
    result = dict(schema="gst_actual_fit_loop_resume79_synthetic_smoke_v1", status="PASS",
        synthetic_only=True, actual_training_data_opened=False, cuda_used=False,
        unchanged_fit_source_ref=runtime.ref(workflow.__file__),
        checkpoint_source_ref=runtime.ref(checkpoint.__file__), smoke_source_ref=runtime.ref(__file__),
        boundary_ref=boundary_ref, uninterrupted_fit_ref=full_ref, resumed_fit_ref=resumed_ref,
        interrupted_at_updates=79, resumed_updates=11, total_updates=90,
        epochs=30, persistent_loader_workers=4, same_model_optimizer_and_all_rng_exact=True,
        all_scientific_history_rows_exact=True, same_epoch_parent_schedule_and_counts=True,
        all_learned_tensor_counters=90, learned_tensors=len(resumed_saved["model_state"]),
        elapsed_seconds=time.monotonic()-started,
        limitation="CPU synthetic smoke; CUDA generator metadata validated separately, no full-data CUDA equivalence claimed")
    runtime.save(root / "receipt.json", result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

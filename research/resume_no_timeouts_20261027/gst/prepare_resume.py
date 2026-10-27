"""Prepare inert GST resume evidence and repair draft; never alter the queue."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import sys
import time
import torch
from research.repairs.benchmark_recovery_20261025.gst import workflow as workflow
from research.repairs.benchmark_recovery_20261025.gst import checkpoint as checkpoint
from arggnn.benchmarking.paper_campaign import runtime as runtime

ROOT = runtime.ROOT
OUT = ROOT / "research/resume_no_timeouts_20261027/gst"
STATE = ROOT / ".runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json"
FIT_ID = "paper_gst_lai_8__all_train__fit"
IDS = [FIT_ID, "paper_gst_lai_8__all_test__predict", "paper_gst_lai_8__all_methods_all_test__score"]


def save(name, value):
    target = OUT / name
    assert not target.exists(), "Preserve previous preparation evidence"
    target.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return runtime.ref(target)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    state = json.loads(STATE.read_text())
    original_plan = runtime.read(state["plan_ref"])
    by_id = {task["id"]: task for task in original_plan["tasks"]}
    assert state["tasks"][FIT_ID]["status"] == "failed"
    assert state["tasks"][FIT_ID]["actual_wait"] and state["tasks"][FIT_ID]["owned_process_group_absent"]
    assert state["tasks"][FIT_ID]["error"] == "Wall time limit exceeded"
    for name in IDS[1:]:
        assert state["tasks"][name]["status"] == "blocked"
        assert not runtime.path(by_id[name]["output_directory"]).exists()
    affected = save("affected_queue_snapshot.json", dict(observed_unix=time.time(),
        state_path=str(STATE.relative_to(ROOT)), plan_ref=state["plan_ref"],
        plan_revision=state["plan_revision"], tasks={name:state["tasks"][name] for name in IDS},
        prior_task_definitions=[by_id[name] for name in IDS]))
    source = runtime.path(by_id[FIT_ID]["output_directory"])
    destination = OUT / "attempt3_archive"
    assert source.is_dir() and not source.is_symlink() and not destination.exists()
    rows = []
    for path in sorted(source.rglob("*")):
        metadata=path.lstat()
        assert not stat.S_ISLNK(metadata.st_mode)
        if stat.S_ISREG(metadata.st_mode):
            rows.append(dict(relative_path=str(path.relative_to(source)), size=metadata.st_size,
                allocated_bytes=metadata.st_blocks*512, device=metadata.st_dev, inode=metadata.st_ino,
                mtime_ns=metadata.st_mtime_ns, nlink=metadata.st_nlink, sha256=runtime.ref(path)["sha256"]))
        else:
            assert stat.S_ISDIR(metadata.st_mode)
    archive_plan = save("archive_plan.json", dict(schema="gst_attempt3_archive_plan_v1",
        status="prepared_not_applied", source=str(source.relative_to(ROOT)),
        destination=str(destination.relative_to(ROOT)),
        archived_bytes=sum(row["size"] for row in rows), file_count=len(rows), files=rows,
        required_action="ROOT must verify all identities and hashes, confirm the failed session is absent, then rename the entire directory exactly once. Preserve original job and log in place. Revalidate destination hashes and write final archive receipt before submitting a repair.",
        affected_queue_snapshot_ref=affected))
    checkpoint_path = source / "native/recovery_latest.pt"
    original_checkpoint_ref=runtime.ref(checkpoint_path)
    saved=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    plan=runtime.read(saved["plan_ref"])
    barrier=runtime.read(saved["observation_barrier_ref"])
    sources=workflow.repair_source_refs(python=sys.executable)
    args=by_id[FIT_ID]["call"]["factory"]["kwargs"]["arguments"]
    assert saved["train_inputs"] == args["train_inputs"]
    assert saved["plan_ref"] == args["plan_ref"]
    assert saved["class_order"] == list(workflow.sampling.NATIVE_PATHS)
    progress=checkpoint.validate_progress(saved,plan,workflow.SETTINGS,args["train_inputs"],sources,barrier["parents"])
    assert saved["optimizer_updates"] == 79
    model=workflow.GstLai8(workflow.sampling.NATIVE_PATHS)
    model.load_state_dict(saved["model_state"],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    optimizer.load_state_dict(saved["optimizer_state"])
    assert len(optimizer.state) == len(list(model.parameters())) == 80
    assert set(int(value["step"]) for value in optimizer.state.values()) == {79}
    for value in saved["model_state"].values():
        assert torch.isfinite(value).all()
    for value in optimizer.state.values():
        for key in ("exp_avg","exp_avg_sq"):
            assert torch.isfinite(value[key]).all()
    rng=saved["random_state"]
    assert rng["torch_cpu"].dtype == torch.uint8 and rng["torch_cpu"].numel() == 5056
    assert len(rng["torch_cuda"]) == 1 and rng["torch_cuda"][0].dtype == torch.uint8
    assert rng["torch_cuda"][0].numel() == 16
    assert rng["numpy"]["keys"].shape == (624,) and rng["numpy"]["kind"] == "MT19937"
    remaining=[]
    for epoch in range(progress["epoch"],31):
        order=list(plan["train_parent_ids"])
        random.Random(workflow.SETTINGS["seed"]+epoch).shuffle(order)
        if epoch == progress["epoch"]:
            order=order[len(progress["completed_parents"]):]
        for parent in order:
            remaining.append(dict(epoch=epoch,parent_id=parent,optimizer_update=80+len(remaining)))
    assert len(remaining) == 11 and remaining[-1]["optimizer_update"] == 90
    validation=save("checkpoint_validation.json",dict(schema="gst_actual79_recovery_validation_v1",status="PASS",
        checkpoint_ref=original_checkpoint_ref,checkpoint_bytes=checkpoint_path.stat().st_size,
        archived_checkpoint_path=str((destination / "native/recovery_latest.pt").relative_to(ROOT)),
        original_observation_barrier_ref=saved["observation_barrier_ref"],
        saved_source_pins_exact=True,input_bindings_exact=True,model_strict_load=True,
        optimizer_restored=True,all_optimizer_counters=79,learned_tensors=80,
        model_and_adam_tensors_finite=True,history_rows=len(saved["history"]),progress=progress,
        settings=saved["settings"],remaining_schedule=remaining,
        rng=dict(python_present=True,numpy_MT19937_keys=624,torch_CPU_bytes=5056,
                 cuda_generator_count=1,cuda_generator_bytes=16,cuda_rng_not_restored_in_cpu_inspection=True),
        actual_data_reloaded=False,observation_proof_matches_saved_barrier=True,
        launch_reauthenticates_original_observations=True,source_refs=sources,
        source_references_preserved=True,no_architecture_data_or_budget_change=True,
        note="The checkpoint retains historical observation_barrier_ref. Archive it unchanged; the resume loop reauthenticates original observations and creates a fresh barrier before loading model/optimizer. Its recovery validator compares proof rather than opening that historical output reference."))
    replacements=copy.deepcopy([by_id[name] for name in IDS])
    for task in replacements:
        task["max_seconds"]=9223372036854775807
    replacements[0]["call"]["factory"]["kwargs"]["arguments"]["resume_checkpoint_ref"]=dict(
        path=str((destination / "native/recovery_latest.pt").relative_to(ROOT)),sha256=original_checkpoint_ref["sha256"])
    replacement_ref=save("replacement_tasks.json",dict(schema="gst_resume79_replacement_tasks_v1",
        status="prepared_not_submitted",tasks=replacements))
    request=save("repair_request_draft.json",dict(schema="paper_campaign_repair_request_v1",
        request_id="gst_resume79_no_wall_limits_20261027",expected_plan_ref=state["plan_ref"],
        expected_plan_revision=state["plan_revision"],expected_attempts={name:state["tasks"][name]["attempt"] for name in IDS},
        reset_task_ids=IDS,replacement_tasks=replacements,archived_bytes=sum(row["size"] for row in rows),
        archive_manifest_ref=archive_plan,
        reason="Resume the actual durable GST update 79/90 checkpoint after the 24-hour timeout, preserving unchanged model, three TRAIN parents, schedule, AdamW state, RNG and fixed final epoch 30. max_seconds=9223372036854775807 is the explicit compatibility sentinel for effectively unlimited wall time under the existing live controller, whose validator requires a positive integer. Preserve the 1800-second DataLoader stall detector. This inert draft must bind a completed archive receipt and refreshed revision before exact repair_requests publication."))
    print(json.dumps(dict(status="PREPARED_ONLY",archive_plan_ref=archive_plan,
        checkpoint_validation_ref=validation,replacement_tasks_ref=replacement_ref,repair_request_draft_ref=request),indent=2))
    assert not torch.cuda.is_initialized()

if __name__ == "__main__":
    main()

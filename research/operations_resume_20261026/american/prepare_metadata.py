"""Authenticate stopped checkpoints and prepare inert replacement task definitions.

No GPU, scientific dataset, training loop, process launcher or queue mutation is
used. Run each mode in its corresponding original Python environment.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from arggnn.benchmarking.paper_american import protocol as p
HERE = ROOT / "research/operations_resume_20261026/american"
RUNTIME = ROOT / ".runtime/benchmarking/paper_campaign_20261024_v2"


def small_write(name, value):
    return p.save(value, HERE / name)


def stopped(task_id):
    state = json.loads((RUNTIME / "state.json").read_text())
    row = state["tasks"][task_id]
    if row["status"] != "failed" or row.get("actual_wait") is not True or row.get("owned_process_group_absent") is not True:
        raise ValueError("Paused task with actual exit and drained owned group required")
    attempt = row.get("attempt", 1)
    job_name = task_id + (".attempt" + str(attempt) if attempt > 1 else "") + ".json"
    task = json.loads((RUNTIME / "jobs" / job_name).read_text())
    snapshot = dict(task_id=task_id, row=row, original_job_ref=p.ref(RUNTIME / "jobs" / job_name),
                    original_plan_ref=state["plan_ref"], original_plan_revision=state["plan_revision"],
                    stopped_receipt_ref=p.ref("research/operations_stop_20261025/stopped.json"))
    snapshot_ref = small_write(task_id + ".stopped.json", snapshot)
    return task, snapshot_ref


def tree():
    import torch
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    task, snapshot_ref = stopped("paper_american_treeloom_fit")
    arguments = task["call"]["spec"]["arguments"]
    checkpoint = p.path(arguments["output_directory"]) / "checkpoints/latest.pt"
    reference = p.ref(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    registry = fresh.FreshParentRegistry.from_ref(arguments["parent_registry_ref"])
    model = fresh.FreshTreeLoom(registry).build_model()
    training.validate_saved_state(saved, model, registry, expected_steps=238)
    schedule = [(epoch, parent) for epoch in range(1, 31) for parent in registry.epoch_order(epoch)]
    sources = training.requirements()["source_refs"]
    evidence = dict(schema="treeloom_durable_optimizer_boundary_v2", checkpoint_ref=reference,
                    parent_registry_ref=registry.reference, complete_parent_updates=238,
                    durable_complete_parent_optimizer_boundary=True,
                    no_inflight_update_in_serialized_state=True, VAL_truth_read=False, TEST_truth_read=False,
                    source_refs=sources, original_job_ref=p.ref(RUNTIME / "jobs/paper_american_treeloom_fit.attempt3.json"),
                    stopped_snapshot_ref=snapshot_ref, observation_barrier_ref=saved["observation_barrier_ref"],
                    schedule_identity=saved["schedule_identity"], next_schedule_index=238,
                    next_parent=schedule[238][1], next_epoch=schedule[238][0], next_update=239,
                    original_last_completed_parent=saved["last_completed_parent"],
                    final_update=600, remaining_complete_parent_updates=362,
                    serialized_rng_keys=list(saved["random_state"]),
                    original_checkpoint_location_preserved=True, inspected_unix=time.time())
    evidence_ref = small_write("treeloom_evidence.json", evidence)
    args = copy.deepcopy(arguments)
    args["resume_checkpoint_ref"] = reference
    args["resume_evidence_ref"] = evidence_ref
    spec_args = dict(args, output_directory=str(p.path(args["output_directory"]) / "resume_20261026_update238"))
    spec = training.build_spec(**spec_args)
    training.validate_resume_checkpoint(saved, model, registry, spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                                 betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    model.load_state_dict(saved["model_state"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    assert len(optimizer.state) == 49 and set(training._optimizer_counts(optimizer)) == {238}
    assert {"5", "10"} == set(saved["checkpoint_event_refs"])
    task["call"] = dict(module="research.operations_resume_20261026.american.treeloom_resume",
                        function="run_task", spec=dict(stage="fit", arguments=args))
    task["source_refs"] += [p.ref(HERE / name) for name in ["__init__.py", "treeloom_resume.py"]]
    task_ref = small_write("treeloom.replacement_task.json", task)
    small_write("treeloom.qualification.json", dict(status="passed", task_ref=task_ref,
                evidence_ref=evidence_ref, checkpoint_sha256=reference["sha256"],
                full_saved_state_validated=True, exact_schedule_and_observation_barrier_validated=True,
                all49_optimizer_states_loaded_on_CPU=True, source_pins_validated=True,
                checkpoint_event_history_preserved=True, scientific_dataset_payloads_opened=False,
                GPU_used=False, training_updates_executed=0))
    print(json.dumps(dict(kind="treeloom", status="prepared", next_parent=schedule[238][1],
                         next_update=239, final_update=600, task_ref=task_ref)))


def arg():
    import torch
    from omegaconf import OmegaConf
    from research.operations_resume_20261026.american import argformer_resume as resume
    task, snapshot_ref = stopped("paper_american_argformer_american_validation_001_mlm")
    request = task["call"]["spec"]
    out = p.path(request["output_directory"])
    checkpoint = out / "checkpoints/ep2-ba119500-rank0.pt"
    if checkpoint.is_symlink() or checkpoint.stat().st_size > 3 * 2**30:
        raise ValueError("Ordinary bounded native checkpoint required")
    # Exactly one explicit full checkpoint authentication; no data-corpus reads.
    reference = p.ref(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    summary = resume.checkpoint_summary(saved)
    native_root = ROOT / ".runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages"
    source_files = [
        native_root / "composer/trainer/trainer.py", native_root / "composer/core/state.py",
        native_root / "composer/utils/checkpoint.py", native_root / "composer/utils/reproducibility.py",
        native_root / "composer/callbacks/checkpoint_saver.py",
        ROOT / "src/arggnn/benchmarking/paper_american/worker.py",
        ROOT / "src/arggnn/benchmarking/paper_american/protocol.py",
        ROOT / "src/arggnn/benchmarking/paper_american/native.py",
        ROOT / "research/reviews/American_C3_native_MLM_amendment_v12/abstract_unix_bootstrap_v1.py",
        HERE / "argformer_resume.py"]
    for f in source_files:
        ast.parse(f.read_text())
    prep_ref = p.ref(request["preparation_completion_path"])
    evidence = dict(schema="american_argformer_full_state_resume_20261026", checkpoint_ref=reference,
                    checkpoint_summary=summary, preparation_ref=prep_ref,
                    resolved_config_ref=p.ref(out / "resolved_config.yaml"), stopped_snapshot_ref=snapshot_ref,
                    source_refs=[p.ref(f) for f in source_files],
                    scheduler=saved["state"]["schedulers"]["LambdaLR"],
                    next_update=119501, remaining_updates=104670, final_update=224170,
                    next_epoch_zero_based=2, next_batch_in_epoch_zero_based=29832,
                    original_checkpoint_location_preserved=True,
                    main_process_rng_restored_by_native_composer=True,
                    deterministic_sample_order_and_saved_cursor_restored=True,
                    worker_mask_rng_not_saved=True, uninterrupted_bitwise_masks_guaranteed=False,
                    inspected_unix=time.time())
    evidence_ref = small_write("argformer_evidence.json", evidence)
    spec = p.build_spec(stage="mlm", preparation_ref=prep_ref, output_directory=str(out / resume.NEW_LEAF),
                        variant="paper_aligned", num_workers=8)
    config = resume.resume_configuration(spec, p.read(prep_ref), out / resume.NEW_LEAF, "mlm", evidence)
    old = OmegaConf.to_container(OmegaConf.load(out / "resolved_config.yaml"), resolve=True)
    new = OmegaConf.to_container(config, resolve=True)
    changed = {key: dict(before=old.get(key), after=new.get(key))
               for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)}
    assert set(changed) == {"save_folder", "load_path", "load_weights_only"}
    assert str(config.max_duration) == "5ep" and config.global_train_batch_size == 512
    assert config.train_loader.num_workers == 8 and config.train_loader.dataset.mlm_probability == .30
    malformed = dict(saved, state=dict(saved["state"], timestamp={"Timestamp":dict(saved["state"]["timestamp"]["Timestamp"], batch=119501)}))
    try:
        resume.checkpoint_summary(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("Incorrect checkpoint cursor was not rejected")
    request = dict(request, resume_evidence_ref=evidence_ref)
    task["call"] = dict(module="research.operations_resume_20261026.american.argformer_resume",
                        function="run_task", spec=request)
    task["source_refs"] += [p.ref(HERE / name) for name in ["__init__.py", "argformer_resume.py"]]
    task_ref = small_write("argformer.replacement_task.json", task)
    small_write("argformer.qualification.json",dict(status="passed",task_ref=task_ref,
                evidence_ref=evidence_ref, checkpoint_sha256=reference["sha256"], checkpoint_summary=summary,
                configuration_changes=changed, wrong_cursor_rejected=True,
                native_restore_code_reviewed=True, scientific_dataset_payloads_opened=False,
                GPU_used=False, training_updates_executed=0,
                limitation="Eight persistent masking-worker RNG states are absent; exact interrupted masks cannot be reconstructed."))
    print(json.dumps(dict(kind="argformer", status="prepared", next_update=119501,
                         final_update=224170, task_ref=task_ref)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["tree", "arg"], required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Inspection must run with CUDA_VISIBLE_DEVICES empty")
    (tree if args.kind == "tree" else arg)()

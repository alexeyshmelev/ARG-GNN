"""Continue the saved native MLM state without overwriting its interrupted attempt.

The established worker still constructs the native author model, input corpus,
optimizer, masking collator and five-epoch run. Only checkpoint loading and new
output placement differ. This module provides no process launcher.
"""
from __future__ import annotations
import importlib
from pathlib import Path
from arggnn.benchmarking.paper_american import protocol as p
from arggnn.benchmarking.paper_american import worker

HERE = "research/operations_resume_20261026/american"
PARENT = "argformer_american_validation_001"
OUTPUT = "experiments/results/paper_american_20261024/" + PARENT + "/mlm/native"
NEW_LEAF = "resume_20261026_batch119500"
BATCH = 119500
BATCH_IN_EPOCH = 29832
UPDATES_PER_EPOCH = 44834


def need(ok, message):
    if not ok:
        raise ValueError(message)


def checkpoint_summary(saved):
    state = saved["state"]
    timestamp = state["timestamp"]["Timestamp"]
    need(timestamp["epoch"] == 2 and timestamp["batch"] == BATCH
         and timestamp["batch_in_epoch"] == BATCH_IN_EPOCH
         and timestamp["sample"] == BATCH * 512, "Exact stopped MLM timestamp required")
    need(set(state["optimizers"]) == {"DecoupledAdamW"}, "Original DecoupledAdamW required")
    states = state["optimizers"]["DecoupledAdamW"]["state"]
    need(len(states) == 77 and all(float(v["step"]) == BATCH for v in states.values())
         and all({"exp_avg", "exp_avg_sq"} <= set(v) for v in states.values()),
         "All 77 saved optimizer moments and counters required")
    scheduler = state["schedulers"]["LambdaLR"]
    need(scheduler["last_epoch"] == BATCH and scheduler["_step_count"] == BATCH + 1,
         "Original saved learning-rate schedule required")
    need(len(saved["rng"]) == 1 and {"python", "numpy", "torch", "cuda"} <= set(saved["rng"][0]),
         "Saved single-rank process RNG states required")
    need(state["rank_zero_seed"] == 17 and state["dataset_state"] == {"train": None, "eval": {}},
         "Original native seed and non-streaming data contract required")
    return dict(epoch=2, batch=BATCH, batch_in_epoch=BATCH_IN_EPOCH,
                updates_per_epoch=UPDATES_PER_EPOCH, final_update=5 * UPDATES_PER_EPOCH,
                optimizer_parameter_states=77, process_rng_saved=True,
                persistent_worker_rng_saved=False)


def resume_configuration(spec, prep, out, phase, evidence):
    from omegaconf import OmegaConf
    need(phase == "mlm", "Only the interrupted MLM stage is supported")
    cfg = worker.configuration(spec, prep, out, phase)
    prior = OmegaConf.load(p.path(evidence["resolved_config_ref"]["path"]))
    original = OmegaConf.to_container(prior, resolve=True)
    candidate = OmegaConf.to_container(cfg, resolve=True)
    for key in ("save_folder",):
        original.pop(key)
        candidate.pop(key)
    need(candidate == original, "Every scientific and data-loader setting must remain identical")
    cfg.load_path = str(p.path(evidence["checkpoint_ref"]["path"]))
    cfg.load_weights_only = False
    cfg.autoresume = False
    cfg.save_overwrite = False
    return cfg


def run_task(request, *, resource_check):
    need(set(request) == {"stage", "output_directory", "preparation_completion_path",
                          "variant", "num_workers", "resume_evidence_ref"},
         "Exact reviewed continuation arguments required")
    need(request["stage"] == "mlm" and request["variant"] == "paper_aligned"
         and request["num_workers"] == 8 and request["output_directory"] == OUTPUT,
         "Original parent, stage, variant and worker count required")
    evidence = p.read(request["resume_evidence_ref"])
    need(evidence["schema"] == "american_argformer_full_state_resume_20261026"
         and evidence["checkpoint_summary"]["batch"] == BATCH, "Reviewed stopped-state evidence required")
    for record in evidence["source_refs"] + [evidence["resolved_config_ref"], evidence["checkpoint_ref"]]:
        resource_check("american_resume_authenticate")
        need(p.ref(record["path"]) == record, "Pinned checkpoint/configuration/source changed")
    original_out = p.path(OUTPUT)
    need(original_out.is_dir() and not (original_out / "completed.json").exists(),
         "Preserved interrupted native output and absent completion required")
    prep_ref = p.ref(request["preparation_completion_path"])
    need(prep_ref == evidence["preparation_ref"], "Same prepared parent corpus required")
    spec = p.build_spec(stage="mlm", preparation_ref=prep_ref,
                        output_directory=str(original_out / NEW_LEAF),
                        variant="paper_aligned", num_workers=8)
    prep, out, torch = worker._start(spec, resource_check)
    saved = torch.load(p.path(evidence["checkpoint_ref"]["path"]),
                       map_location="cpu", weights_only=False, mmap=True)
    summary = checkpoint_summary(saved)
    expected_scheduler = saved["state"]["schedulers"]["LambdaLR"]
    del saved
    cfg = resume_configuration(spec, prep, out, "mlm", evidence)
    native_module = importlib.import_module("main")
    original_trainer, original_configuration = native_module.Trainer, worker.configuration
    loaded = {}
    def checked_trainer(*args, **kwargs):
        need(kwargs.get("load_path") == cfg.load_path and kwargs.get("load_weights_only") is False,
             "Full native checkpoint loading required")
        kwargs["spin_dataloaders"] = True
        kwargs["load_strict_model_weights"] = True
        kwargs["load_ignore_keys"] = None
        trainer = original_trainer(*args, **kwargs)
        state = trainer.state
        need(int(state.timestamp.batch) == BATCH and int(state.timestamp.epoch) == 2
             and int(state.timestamp.batch_in_epoch) == BATCH_IN_EPOCH,
             "Composer must restore the exact timestamp before training")
        need(len(state.train_dataloader) == UPDATES_PER_EPOCH,
             "Original 44834 updates per epoch required")
        states = state.optimizers[0].state
        need(len(states) == 77 and all(float(v["step"]) == BATCH for v in states.values()),
             "Composer must restore all optimizer counters")
        need(state.schedulers[0].state_dict() == expected_scheduler and trainer._rng_state is not None,
             "Composer must retain saved scheduler and pending process RNG restoration")
        need(trainer.spin_dataloaders and not state.dataset_resumption.get("train", False),
             "Native deterministic sampler must skip the saved in-epoch cursor")
        loaded.update(timestamp_restored=True, optimizer_restored=True, scheduler_restored=True,
                      process_rng_restore_pending=True, spin_dataloaders=True)
        p.save(dict(schema="american_argformer_loaded_resume_state_v1", **summary, **loaded),
               out / "loaded_resume_state.json")
        return trainer
    try:
        worker.configuration = lambda *args: cfg
        native_module.Trainer = checked_trainer
        result = worker._train(spec, prep, out, torch, resource_check)
    finally:
        worker.configuration = original_configuration
        native_module.Trainer = original_trainer
    need(loaded.get("timestamp_restored") is True, "Verified native full-state restore required")
    completion = dict(schema="paper_american_native_completion_v1", stage="mlm",
                      variant="paper_aligned", parent_id=PARENT, preparation_ref=prep_ref,
                      query_ancestry_truth_opened=False, author_source_modified=False,
                      resume_evidence_ref=request["resume_evidence_ref"],
                      resumed_from_update=BATCH, preserved_interrupted_output=OUTPUT,
                      process_rng_restored_by_composer=True,
                      persistent_worker_rng_not_serialized=True,
                      uninterrupted_bitwise_mask_sequence_guaranteed=False, **result)
    completion["self_spec_ref"] = p.save(spec, out / "executed_spec.json")
    resource_check("american_resume_publish_completion")
    reference = p.save(completion, original_out / "completed.json")
    return dict(completion_ref=reference, **completion)

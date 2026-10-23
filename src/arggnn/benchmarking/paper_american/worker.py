"""Native task execution inside ROOT's separately admitted single-device worker."""
from __future__ import annotations
from contextlib import nullcontext
import importlib
import os
import pickle
import sys
import time
from . import protocol as p
from . import native

def configuration(spec, prep, out, phase):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(p.AUTHOR / ("yamls/" + ("mlm" if phase == "mlm" else "contrastive") + ".yaml"))
    root = p.path(prep["native_dataset_refs"]["mlm/train"]["path"]).parents[2]
    cfg.data_local = str(root / ("mlm" if phase == "mlm" else "contrastive"))
    cfg.run_name = spec["parent_id"] + "_" + spec["variant"] + "_" + phase
    cfg.save_folder = str(out / "checkpoints")
    cfg.load_path = None
    cfg.autoresume = False
    cfg.save_overwrite = False
    cfg.save_num_checkpoints_to_keep = 1
    cfg.loggers = {}
    cfg.device = "gpu"
    cfg.model.pretrained_model_name = str(p.ROOT / "research/reviews/C3_bert_base_configuration_root_v1")
    if phase != "mlm":
        previous = p.read(spec["checkpoint_completion_ref"])
        cfg.model.pretrained_checkpoint = str(p.path(previous["checkpoint_ref"]["path"]))
    for name in ("train_loader", "eval_loader"):
        loader = cfg[name]
        loader.num_workers = spec["num_workers"]
        loader.persistent_workers = spec["num_workers"] > 0
        loader.prefetch_factor = 2
        if spec["variant"] == "paper_aligned":
            if phase == "mlm":
                loader.dataset.skip_extant_tokens = False
                loader.dataset.mlm_probability = 0.30
            else:
                loader.skip_extant_tokens = False
    if phase != "mlm" and spec["variant"] == "paper_aligned":
        cfg.model.contrastive.temperature = 0.05
        cfg.model.contrastive.use_class_weights = True
    return cfg

def _start(spec, resource_check):
    if not callable(resource_check):
        raise ValueError("ROOT resource callback required")
    if spec["schema"] != "paper_american_native_task_v1":
        raise ValueError("Supported complete task schema required")
    if os.environ.get("ARGGNN_PHYSICAL_GPU") not in ("3", "4", "5"):
        raise ValueError("ROOT must bind one currently authorized physical GPU 3, 4 or 5")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("One explicitly admitted visible device required")
    if os.environ.get("ARGGNN_PHYSICAL_GPU") == visible:
        raise ValueError("Use the ROOT-verified device UUID rather than an ambiguous index")
    if not sys.dont_write_bytecode:
        raise ValueError("Native worker must use -B")
    if p.path(sys.executable) != p.NATIVE_PYTHON.resolve():
        raise ValueError("Use the isolated native ARGformer interpreter")
    expected = p.build_spec(stage=spec["stage"], preparation_ref=spec["preparation_ref"],
                            output_directory=spec["output_directory"], variant=spec["variant"],
                            checkpoint_completion_ref=spec["checkpoint_completion_ref"],
                            num_workers=spec["num_workers"], embedding_batch_size=spec["embedding_batch_size"])
    if spec != expected:
        raise ValueError("Task changed after construction")
    for record in spec["author_source_refs"] + spec["adapter_source_refs"] + [spec["runtime_helper_ref"]]:
        if p.ref(record["path"]) != record:
            raise ValueError("Author source identity changed")
    prep = p.read(spec["preparation_ref"])
    if prep["parent_id"] != spec["parent_id"] or prep["query_ancestry_truth_opened"] is not False:
        raise ValueError("Same-parent observation-only preparation required")
    for record in [*prep["native_dataset_refs"].values(), *prep["full_path_id_order_refs"].values(),
                   prep["vocab_ref"], prep["focals_ref"]]:
        resource_check("paper_american_authenticate_input")
        if p.ref(record["path"]) != record:
            raise ValueError("Prepared input changed")
    if spec["checkpoint_completion_ref"] is not None:
        previous = p.read(spec["checkpoint_completion_ref"])
        if p.ref(previous["checkpoint_ref"]["path"]) != previous["checkpoint_ref"]:
            raise ValueError("Predecessor checkpoint changed")
    out = p.path(spec["output_directory"])
    if out.exists():
        raise ValueError("Fresh output only; no implicit overwrite or resume")
    out.mkdir(parents=True)
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    for key in ("TMPDIR", "TMP", "TEMP", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME",
                "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        directory = out / (".tmp" if key in ("TMPDIR", "TMP", "TEMP") else ".cache/" + key.lower())
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    import tempfile
    tempfile.tempdir = str(out / ".tmp")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
                      HF_HUB_DISABLE_TELEMETRY="1")
    sys.path[:0] = [str(p.AUTHOR), str(p.ROOT)]
    os.chdir(out)
    ipc = importlib.import_module("research.reviews.American_C3_native_MLM_amendment_v12.abstract_unix_bootstrap_v1")
    ipc.install_abstract_unix_v1("paper_american")
    import torch
    resource_check("paper_american_before_cuda")
    if torch.cuda.device_count() != 1 or not visible.startswith("GPU-"):
        raise ValueError("One ROOT-verified visible device UUID required")
    memory_policy = os.environ.get("ARGGNN_GPU_MEMORY_POLICY")
    if memory_policy == "full_device":
        pass
    elif memory_policy == "explicit_allocator_cap":
        cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
        total = torch.cuda.get_device_properties(0).total_memory
        if not 0 < cap <= total:
            raise ValueError("Valid explicit allocator cap required for this selected policy")
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    else:
        raise ValueError("ROOT must explicitly select the GPU memory policy")
    return prep, out, torch

def _train(spec, prep, out, torch, check):
    from omegaconf import OmegaConf
    from composer import Callback
    phase = spec["stage"]
    cfg = configuration(spec, prep, out, phase)
    module = importlib.import_module("main" if phase == "mlm" else "sequence_contrastive")
    if not str(module.__file__).startswith(str(p.AUTHOR) + "/"):
        raise ValueError("Pinned author module origin required")
    weights = None
    if phase == "contrastive" and spec["variant"] == "paper_aligned":
        labels = p.path(cfg.train_loader.local) / cfg.train_loader.split / cfg.train_loader.labels_file
        with labels.open("rb") as stream:
            weights = native.inverse_frequency(pickle.load(stream))
    class Guard(Callback):
        def batch_end(self, state, logger):
            check("paper_american_training_batch")
        def eval_end(self, state, logger):
            check("paper_american_validation_end")
    box = {}
    original_trainer = module.Trainer
    def capture(*args, **kwargs):
        kwargs["callbacks"] = list(kwargs.get("callbacks", [])) + [Guard()]
        trainer = original_trainer(*args, **kwargs)
        box["trainer"] = trainer
        box["updates_per_epoch"] = len(trainer.state.train_dataloader)
        if phase == "contrastive" and int(cfg.device_train_microbatch_size) != 4096:
            raise ValueError("Splitting the contrastive batch changes in-batch negatives")
        return trainer
    module.Trainer = capture
    context = native.adaptations(module, cfg, phase=phase, class_weights=weights) if spec["variant"] == "paper_aligned" else nullcontext()
    OmegaConf.save(cfg, out / "resolved_config.yaml", resolve=True)
    started = time.monotonic()
    try:
        with context:
            (module.main if phase == "mlm" else module.train)(cfg)
    finally:
        module.Trainer = original_trainer
    trainer = box["trainer"]
    epochs = 5 if phase == "mlm" else 1
    updates = epochs * box["updates_per_epoch"]
    if int(trainer.state.timestamp.epoch.value) != epochs or int(trainer.state.timestamp.batch.value) != updates:
        raise ValueError("Complete original epoch and update budget required")
    from arggnn.benchmarking.american_competitor_worker import checkpoint_counters
    completed = []
    for candidate in (out / "checkpoints").glob("*.pt"):
        if candidate.is_symlink():
            continue
        check("paper_american_verify_saved_checkpoint")
        saved = torch.load(candidate, map_location="cpu", weights_only=False, mmap=True)
        timestamp = saved["state"].get("timestamp", {}).get("Timestamp", {})
        if timestamp.get("epoch") == epochs and timestamp.get("batch") == updates:
            completed.append((candidate, checkpoint_counters(saved["state"], epochs, updates)))
        del saved
    if not completed:
        destination = out / "checkpoints/final-full-budget.pt"
        trainer.save_checkpoint(str(destination), weights_only=False)
        saved = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
        completed.append((destination, checkpoint_counters(saved["state"], epochs, updates)))
        del saved
    if len(completed) != 1:
        raise ValueError("Exactly one complete final checkpoint required")
    destination, counters = completed[0]
    return dict(checkpoint_ref=p.ref(destination), resolved_config_ref=p.ref(out / "resolved_config.yaml"),
                actual_seconds=time.monotonic() - started, train_class_weights=weights,
                full_budget_completed=True, **counters)

def _predict(spec, prep, out, torch, check):
    import numpy as np
    from torch.utils.data import DataLoader
    from .evaluation import ReferencePCA, seal_predictions
    from arggnn.american_c3_native_concat_v1 import row_order
    from arggnn.american_c3_evaluate_v1 import registry, verify_prediction
    from omegaconf import OmegaConf
    previous = p.read(spec["checkpoint_completion_ref"])
    if p.ref(previous["checkpoint_ref"]["path"]) != previous["checkpoint_ref"]:
        raise ValueError("Actual final contrastive checkpoint changed")
    cfg = OmegaConf.load(p.path(previous["resolved_config_ref"]["path"]))
    module = importlib.import_module("sequence_contrastive")
    build_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    build_cfg.pretrained_checkpoint = None
    build_cfg.contrastive.use_class_weights = False
    model = module.build_contrastive_model(build_cfg)
    saved = torch.load(p.path(previous["checkpoint_ref"]["path"]), map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(saved["state"]["model"], strict=True)
    del saved
    if spec["variant"] == "paper_aligned":
        native.install_extant_head(model.base.backbone)
    model = model.cuda().eval()
    fit = ReferencePCA()
    root = p.path(prep["native_dataset_refs"]["mlm/train"]["path"]).parents[2]
    def batches(view, split):
        record = prep["full_path_id_order_refs"][view + "/" + split]
        order = row_order(record, parent_id=spec["parent_id"], reference=view == "contrastive")
        ds = module.ARGDataset(local=str(root / view), split=split, max_seq_len=128,
                               pad_sequences=True, skip_extant_tokens=spec["variant"] == "release_control")
        if len(ds) != len(order["record_ids"]):
            raise ValueError("Every path must retain its complete source identity")
        loader = DataLoader(ds, batch_size=spec["embedding_batch_size"], shuffle=False,
                            num_workers=spec["num_workers"], pin_memory=True,
                            persistent_workers=spec["num_workers"] > 0,
                            collate_fn=module.transformers.default_data_collator)
        start = 0
        with torch.inference_mode():
            for batch in loader:
                check("paper_american_stream_embedding")
                if spec["variant"] == "paper_aligned":
                    native.crop_batch(batch)
                size = len(batch["input_ids"])
                vectors = model.base.encode(**{k: v.cuda(non_blocking=True) for k, v in batch.items()}).float().cpu().numpy()
                yield vectors, {k: v[start:start + size] for k, v in order.items()
                                if k not in ("parent_id", "class_order")}
                start += size
        if start != len(ds):
            raise ValueError("Embedding traversal incomplete")
    for split in ("train", "val"):
        for vectors, order in batches("contrastive", split):
            fit.update(vectors, order["reference_classes_0based"])
    fit.finish()
    np.savez(out / "reference_pca10.npz", **fit.arrays())
    query_rows = {k: [] for k in ("record_ids", "sample_nodes", "sample_indices")}
    calls = []
    for split in ("train", "val"):
        for vectors, order in batches("query_paths", split):
            calls.append(fit.predict(vectors))
            for key in query_rows:
                query_rows[key].append(order[key])
    query_rows = {k: np.concatenate(v) for k, v in query_rows.items()}
    target = registry(prep["focals_ref"])
    predictions = seal_predictions(target, query_rows, np.concatenate(calls))
    verify_prediction(predictions, target)
    np.savez(out / "predictions.npz", **predictions)
    return dict(prediction_ref=p.ref(out / "predictions.npz"), focal_registry_ref=prep["focals_ref"],
                reference_pca_ref=p.ref(out / "reference_pca10.npz"), records=len(target["record_ids"]),
                class_order=p.CLASSES, probability_semantics="hard_call_one_hot_transport",
                evaluator="reference_PCA10_nearest_class_centroid", pca_fit="all_reference_paths_only",
                query_truth_used_for_fit=False, full_budget_completed=True)

def run(spec, *, resource_check):
    prep, out, torch = _start(spec, resource_check)
    result = _predict(spec, prep, out, torch, resource_check) if spec["stage"] == "predict" else _train(spec, prep, out, torch, resource_check)
    completion = dict(schema="paper_american_native_completion_v1", stage=spec["stage"],
                      variant=spec["variant"], parent_id=spec["parent_id"],
                      preparation_ref=spec["preparation_ref"], query_ancestry_truth_opened=False,
                      author_source_modified=False, **result)
    completion["self_spec_ref"] = p.save(spec, out / "executed_spec.json")
    return dict(completion_ref=p.save(completion, out / "completed.json"), **completion)

def run_flare(spec, *, resource_check, bounded_runner=None):
    """Use the full existing converter, original native Java route and AN1/AN2 seal."""
    from arggnn.benchmarking.american_competitors import run_flare as original
    return original(spec, resource_check=resource_check, bounded_runner=bounded_runner)


def run_task(request, *, resource_check):
    """Resolve actual dependency products only after ROOT releases this DAG node."""
    spec = p.build_spec(stage=request["stage"],
                        preparation_ref=request.get("preparation_ref") or p.ref(request["preparation_completion_path"]),
                        output_directory=request["output_directory"],
                        variant=request.get("variant", "paper_aligned"),
                        checkpoint_completion_ref=p.ref(request["checkpoint_completion_path"])
                        if request.get("checkpoint_completion_path") else None,
                        num_workers=request.get("num_workers", 8))
    return run(spec, resource_check=resource_check)

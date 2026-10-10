#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from argmix_project import write_latest_checkpoint_pointer
from config_loader import (
    get_model_config,
    get_paths_config,
    get_subgraph_merge_enabled,
    get_subgraph_merge_signature_mode,
    load_and_validate,
)
from joint_interval_model import (
    build_token_to_id_from_config,
    build_joint_model_from_config,
    compute_masked_row_accuracy,
    count_joint_predict_haplotypes,
    infer_forward_founder_holdout_fraction,
    resolve_joint_model_config,
    save_joint_model_checkpoint,
    write_joint_training_data_log,
)
from joint_support import masked_sequence_cross_entropy
from utils.logging_setup import setup_logging
from utils.reproducibility import seed_everything

GT_GRU_FORMAT_VERSION = 2
GT_GRU_METADATA_LEN = 12
GT_GRU_META_FORMAT = 0
GT_GRU_META_SPLIT = 1
GT_GRU_META_CHUNK_ROWS = 2
GT_GRU_META_CHUNK_STRIDE = 3
GT_GRU_META_PAD_ID = 7
GT_GRU_META_UNK_ID = 8
GT_GRU_META_SAMPLE_ID = 9
GT_GRU_META_ANC_ID = 10
GT_GRU_META_VOCAB_SIZE = 11
CANONICAL_SPECIAL_IDS = {
    "pad_id": 0,
    "unk_id": 1,
    "sample_id": 2,
    "anc_id": 3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train experimental joint transformer+GRU interval model")
    p.add_argument("--config", required=True, help="Backend config YAML")
    p.add_argument("--train-pt", required=True, help="Prebuilt joint train split tensor archive")
    p.add_argument("--val-pt", required=True, help="Prebuilt joint validation split tensor archive")
    p.add_argument("--checkpoint-output", default=None, help="Optional file to receive the final checkpoint path")
    return p.parse_args()


def _masked_joint_loss(
    refined_logits: torch.Tensor,
    labels: torch.Tensor,
    row_mask: torch.Tensor,
    cm_features: torch.Tensor,
    *,
    label_smoothing: float,
    loss_weighting: dict | None = None,
) -> torch.Tensor:
    mode = str((loss_weighting or {}).get("mode", "none")).strip().lower()
    if mode == "none":
        return masked_sequence_cross_entropy(
            refined_logits,
            labels,
            row_mask,
            label_smoothing=float(label_smoothing),
        )
    if mode != "sqrt_span_cm":
        raise ValueError(f"Unsupported joint loss weighting mode: {mode}")
    if refined_logits.ndim != 3:
        raise ValueError(f"refined_logits must be [B, L, C], got {tuple(refined_logits.shape)}")
    if labels.shape != refined_logits.shape[:2]:
        raise ValueError("labels shape mismatch with refined_logits")
    if row_mask.shape != refined_logits.shape[:2]:
        raise ValueError("row_mask shape mismatch with refined_logits")
    if cm_features.shape != (*refined_logits.shape[:2], 2):
        raise ValueError("cm_features must be [B, L, 2]")
    c = refined_logits.shape[-1]
    flat_labels = labels.reshape(-1)
    per_pos = F.cross_entropy(
        refined_logits.reshape(-1, c),
        flat_labels,
        ignore_index=-100,
        label_smoothing=float(label_smoothing),
        reduction="none",
    )
    valid = row_mask.reshape(-1) & (flat_labels != -100)
    if not torch.any(valid):
        return refined_logits.new_tensor(0.0)
    raw_weights = torch.sqrt(
        torch.clamp(
            cm_features[..., 0].reshape(-1).to(device=per_pos.device, dtype=per_pos.dtype)[valid],
            min=0.0,
        )
    )
    if not torch.any(raw_weights > 0):
        return per_pos[valid].mean()
    normalized_weights = raw_weights / raw_weights.mean().clamp_min(torch.finfo(raw_weights.dtype).eps)
    return (per_pos[valid] * normalized_weights).mean()


def _loss_weighting_mode(loss_weighting: dict | None) -> str:
    return str((loss_weighting or {}).get("mode", "none")).strip().lower()


def _loss_weighting_log_value(loss_weighting: dict | None) -> str:
    mode = _loss_weighting_mode(loss_weighting)
    if mode == "sqrt_span_cm":
        return "sqrt_span_cm (mean-normalized per batch)"
    return mode


def _wandb_module():
    try:
        import wandb
    except Exception:
        return None
    return wandb


def _setup_wandb(config: dict, run_name: str):
    wandb = _wandb_module()
    if wandb is None:
        return None, False
    if wandb.run is not None:
        return wandb.run, False
    wandb_cfg = config.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None, False
    run = wandb.init(
        project=wandb_cfg.get("project", "ARGMix"),
        name=run_name,
        dir=(get_paths_config(config).get("wandb_logs_dir") or None),
        config={
            "model": config.get("model", {}) or {},
            "joint_model": config.get("joint_model", {}) or {},
            "sweep": config.get("sweep", {}) or {},
        },
    )
    return run, True


def _wandb_log(metrics: dict) -> None:
    wandb = _wandb_module()
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics)


def _update_wandb_summary(values: dict) -> None:
    wandb = _wandb_module()
    if wandb is not None and wandb.run is not None:
        wandb.run.summary.update(values)


def _append_sweep_result(config: dict, row: dict) -> None:
    results_csv = (config.get("sweep", {}) or {}).get("results_csv")
    if not results_csv:
        return
    path = Path(str(results_csv))
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_name",
        "checkpoint_path",
        "best_val_loss",
        "best_epoch",
        "best_val_acc",
        "learning_rate",
        "num_refs",
        "batch_size",
        "num_epochs",
        "early_stop_patience",
        "weight_decay",
        "label_smoothing",
        "chunk_rows",
        "chunk_stride",
    ]
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def _load_gt_gru_split_pt(path: str, *, expected_split_code: int) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        import numpy as np

        with np.load(path) as npz:
            payload = {key: torch.from_numpy(np.asarray(npz[key])) for key in npz.files}
    if not isinstance(payload, dict):
        raise ValueError(f"Joint training tensor archive must load as a dict: {path}")
    if "metadata" not in payload:
        raise ValueError(f"Joint training tensor archive {path} missing keys: ['metadata']")
    meta = torch.as_tensor(payload["metadata"], dtype=torch.long).view(-1)
    format_version = int(meta[GT_GRU_META_FORMAT].item()) if meta.numel() >= 1 else -1
    if format_version != GT_GRU_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported joint training tensor archive format {format_version} in {path}; "
            "regenerate train.pt/val.pt with argmix build-training"
        )
    required = ("input_ids", "tmrca", "labels", "row_mask", "cm_features", "metadata")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Joint training tensor archive {path} missing keys: {missing}")
    if meta.numel() < GT_GRU_METADATA_LEN:
        raise ValueError(
            f"Unsupported joint training tensor archive metadata in {path}: "
            f"expected at least {GT_GRU_METADATA_LEN} fields for format {format_version}"
        )
    split_code = int(meta[GT_GRU_META_SPLIT].item())
    if split_code != int(expected_split_code):
        raise ValueError(f"Joint training tensor split code mismatch in {path}: got {split_code}")
    pad_id = int(meta[GT_GRU_META_PAD_ID].item())
    unk_id = int(meta[GT_GRU_META_UNK_ID].item())
    sample_id = int(meta[GT_GRU_META_SAMPLE_ID].item())
    anc_id = int(meta[GT_GRU_META_ANC_ID].item())
    actual_special_ids = {
        "pad_id": pad_id,
        "unk_id": unk_id,
        "sample_id": sample_id,
        "anc_id": anc_id,
    }
    if actual_special_ids != CANONICAL_SPECIAL_IDS:
        raise ValueError(
            f"Joint training tensor archive {path} has non-canonical token IDs: {actual_special_ids}; "
            f"expected {CANONICAL_SPECIAL_IDS}"
        )
    vocab_size = int(meta[GT_GRU_META_VOCAB_SIZE].item())
    if vocab_size <= 0:
        raise ValueError(f"Joint training tensor archive {path} has invalid vocab_size={vocab_size}")
    labels = torch.as_tensor(payload["labels"], dtype=torch.long)
    row_mask = torch.as_tensor(payload["row_mask"], dtype=torch.bool)
    cm_features = torch.as_tensor(payload["cm_features"], dtype=torch.float32)
    out = {
        "labels": labels,
        "row_mask": row_mask,
        "cm_features": cm_features,
        "metadata": meta,
        "pad_id": pad_id,
        "vocab_size": vocab_size,
        "format_version": format_version,
        "base_encoder": "tmrca_transformer",
    }
    input_ids = torch.as_tensor(payload["input_ids"], dtype=torch.long)
    tmrca = torch.as_tensor(payload["tmrca"], dtype=torch.int16)
    if input_ids.ndim != 3:
        raise ValueError(f"input_ids must be [N,chunk_rows,seq_len], got {tuple(input_ids.shape)}")
    if tmrca.ndim != 4:
        raise ValueError(f"tmrca must be [N,chunk_rows,seq_len,seq_len], got {tuple(tmrca.shape)}")
    if labels.shape != input_ids.shape[:2] or row_mask.shape != input_ids.shape[:2]:
        raise ValueError("labels/row_mask shape mismatch with input_ids")
    if cm_features.shape != (*input_ids.shape[:2], 2):
        raise ValueError("cm_features shape mismatch")
    if input_ids.numel():
        min_token_id = int(input_ids.min().item())
        max_token_id = int(input_ids.max().item())
        if min_token_id < 0 or max_token_id >= vocab_size:
            raise ValueError(
                f"Joint training tensor archive {path} contains token ids outside [0, {vocab_size})"
            )
    out["input_ids"] = input_ids
    out["tmrca"] = tmrca
    out["num_chunks"] = int(input_ids.shape[0])
    return out


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_and_validate(args.config)
    joint_cfg = resolve_joint_model_config(config)
    subgraph_merge_enabled = get_subgraph_merge_enabled(config)
    subgraph_merge_signature_mode = get_subgraph_merge_signature_mode(config)
    if subgraph_merge_enabled:
        logger.info(
            "Subgraph merge: enabled (signature_mode=%s)",
            subgraph_merge_signature_mode,
        )
    seed_everything(50)
    paths_cfg = get_paths_config(config)
    train_pairs = paths_cfg.get("train_pairs")
    val_pairs = paths_cfg.get("val_pairs")
    save_dir = paths_cfg.get("save_dir")
    if not train_pairs or not val_pairs or not save_dir:
        raise ValueError("paths.train_pairs, paths.val_pairs, and paths.save_dir are required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_joint_model_from_config(config, device=device)
    model_cfg = get_model_config(config)

    train_cfg = joint_cfg["train"]
    sample_seed = int(train_cfg["sample_seed"])
    train_max_subgraphs = train_cfg["max_train_subgraphs"]
    val_max_subgraphs = train_cfg["max_val_subgraphs"]
    label_smoothing = float(train_cfg["label_smoothing"])
    loss_weighting = train_cfg.get("loss_weighting", {}) or {}
    ref_aug_cfg = train_cfg.get("reference_selection_augmentation", {}) or {}
    train_ref_aug_cfg = ref_aug_cfg.get("train", {}) or {}
    val_ref_aug_cfg = ref_aug_cfg.get("val", {}) or {}
    train_ref_aug_drop_top_m = int(train_ref_aug_cfg.get("drop_top_m_nearest", 0))
    val_ref_aug_drop_top_m = int(val_ref_aug_cfg.get("drop_top_m_nearest", 0))
    expected_encoder = str(joint_cfg.get("base_encoder", "tmrca_transformer"))
    available_predict = count_joint_predict_haplotypes(config)
    founder_holdout_fraction = infer_forward_founder_holdout_fraction(config)

    logger.info(
        "Loading prebuilt joint training tensor archives: "
        "train_subgraph_cap=%s val_subgraph_cap=%s chunk_rows=%d chunk_stride=%d "
        "base_encoder=%s sample_seed=%d label_smoothing=%.3f founder_holdout_fraction=%s "
        "available_predict_haplotypes=%d "
        "reference_selection_augmentation train_drop_top_m_nearest=%d val_drop_top_m_nearest=%d "
        "loss_weighting=%s",
        "all" if train_max_subgraphs is None else train_max_subgraphs,
        "all" if val_max_subgraphs is None else val_max_subgraphs,
        int(joint_cfg["chunk_rows"]),
        int(joint_cfg["chunk_stride"]),
        expected_encoder,
        sample_seed,
        label_smoothing,
        "n/a" if founder_holdout_fraction is None else f"{float(founder_holdout_fraction):.4f}",
        int(available_predict),
        train_ref_aug_drop_top_m,
        val_ref_aug_drop_top_m,
        _loss_weighting_log_value(loss_weighting),
    )
    train_chunks = _load_gt_gru_split_pt(str(args.train_pt), expected_split_code=0)
    val_chunks = _load_gt_gru_split_pt(str(args.val_pt), expected_split_code=1)
    if expected_encoder == "tmrca_cutoff_counts":
        raise ValueError("joint_model.base_encoder=tmrca_cutoff_counts has been removed; use weighted")
    if expected_encoder not in {"tmrca_transformer", "weighted"}:
        raise ValueError(f"Unsupported joint_model.base_encoder: {expected_encoder}")
    if train_chunks["base_encoder"] != "tmrca_transformer" or val_chunks["base_encoder"] != "tmrca_transformer":
        raise ValueError(
            "Joint training tensor archive base_encoder mismatch: "
            f"config={expected_encoder} train={train_chunks['base_encoder']} val={val_chunks['base_encoder']}"
        )
    if int(train_chunks["pad_id"]) != int(val_chunks["pad_id"]):
        raise ValueError(
            f"Joint training tensor train/val pad_id mismatch: {train_chunks['pad_id']} vs {val_chunks['pad_id']}"
        )
    if int(train_chunks["vocab_size"]) != int(val_chunks["vocab_size"]):
        raise ValueError(
            "Joint training tensor train/val vocab_size mismatch: "
            f"{train_chunks['vocab_size']} vs {val_chunks['vocab_size']}"
        )
    pad_id = int(train_chunks["pad_id"])
    token_to_id = build_token_to_id_from_config(config)
    if int(train_chunks["vocab_size"]) != len(token_to_id):
        raise ValueError(
            "Joint training tensor archive vocab_size does not match model config vocabulary: "
            f"archive={train_chunks['vocab_size']} config={len(token_to_id)}"
        )
    gt_gru_archives = {
        "format_version": int(train_chunks["format_version"]),
        "base_encoder": expected_encoder,
        "train_pt": str(args.train_pt),
        "val_pt": str(args.val_pt),
        "train_metadata": [int(x) for x in train_chunks["metadata"].tolist()],
        "val_metadata": [int(x) for x in val_chunks["metadata"].tolist()],
        "pad_id": int(pad_id),
        "vocab_size": int(train_chunks["vocab_size"]),
    }
    train_sampling = {"prebuilt_pt": str(args.train_pt)}
    val_sampling = {"prebuilt_pt": str(args.val_pt)}
    train_archive_seq_len = int(train_chunks["input_ids"].shape[-1])
    val_archive_seq_len = int(val_chunks["input_ids"].shape[-1])
    expected_archive_seq_len = int(model_cfg.get("num_refs", 15)) + 1
    logger.info(
        "Loaded prebuilt joint training tensor chunks: "
        "train=%s val=%s archive_seq_len=train:%d,val:%d expected_seq_len=%d "
        "num_refs=%d base_encoder=%s base_final_layer_dim=%d gru_input_dim=%d "
        "pad_id=%d vocab_size=%d",
        tuple(train_chunks["input_ids"].shape),
        tuple(val_chunks["input_ids"].shape),
        train_archive_seq_len,
        val_archive_seq_len,
        expected_archive_seq_len,
        int(model_cfg.get("num_refs", 15)),
        expected_encoder,
        int(model.final_layer_dim),
        int(model.gru_input_dim),
        pad_id,
        int(train_chunks["vocab_size"]),
    )
    if train_archive_seq_len != expected_archive_seq_len or val_archive_seq_len != expected_archive_seq_len:
        raise ValueError(
            "Joint training tensor archive seq_len mismatch: "
            f"train={train_archive_seq_len} val={val_archive_seq_len} expected={expected_archive_seq_len}"
        )

    train_ds = TensorDataset(
        train_chunks["input_ids"],
        train_chunks["tmrca"],
        train_chunks["labels"],
        train_chunks["row_mask"],
        train_chunks["cm_features"],
    )
    val_ds = TensorDataset(
        val_chunks["input_ids"],
        val_chunks["tmrca"],
        val_chunks["labels"],
        val_chunks["row_mask"],
        val_chunks["cm_features"],
    )
    batch_size = int(joint_cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(joint_cfg["train"]["learning_rate"]),
        weight_decay=float(joint_cfg["train"]["weight_decay"]),
    )
    num_epochs = int(joint_cfg["train"]["num_epochs"])
    patience = int(joint_cfg["train"]["early_stop_patience"])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_cfg = config.get("model", {}) or {}
    run_name = (
        f"{timestamp}_joint_tmrca_gru_h{int(model_cfg.get('hidden_size', 128))}"
        f"_nh{int(model_cfg.get('num_heads', 4))}_nl{int(model_cfg.get('num_layers', 4))}"
        f"_c{int(model_cfg.get('num_classes', 2))}_chunk{int(joint_cfg['chunk_rows'])}"
    )
    sweep_run_id = str((config.get("sweep", {}) or {}).get("run_id") or "").strip()
    if sweep_run_id:
        run_name = f"{run_name}_{sweep_run_id}"
    ckpt_dir = Path(str(save_dir)) / run_name
    ckpt_path = ckpt_dir / "best_model.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb_run, finish_wandb = _setup_wandb(config, run_name)
    if wandb_run is not None:
        wandb_run.name = wandb_run.name or run_name

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    best_val_acc = float("nan")
    patience_left = patience

    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            train_correct = 0
            train_total = 0
            for batch in train_loader:
                ids_or_features, tmrca, labels, row_mask, cm_features = batch
                ids_or_features = ids_or_features.to(device, non_blocking=True)
                tmrca = tmrca.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                row_mask = row_mask.to(device, non_blocking=True)
                cm_features = cm_features.to(device, non_blocking=True)

                optimizer.zero_grad()
                _, refined_logits = model.forward_chunks(
                    ids_or_features,
                    tmrca,
                    row_mask,
                    cm_features,
                    pad_id=pad_id,
                )
                loss = _masked_joint_loss(
                    refined_logits,
                    labels,
                    row_mask,
                    cm_features,
                    label_smoothing=label_smoothing,
                    loss_weighting=loss_weighting,
                )
                correct, total = compute_masked_row_accuracy(refined_logits, labels, row_mask)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item())
                train_batches += 1
                train_correct += correct
                train_total += total

            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids_or_features, tmrca, labels, row_mask, cm_features = batch
                    ids_or_features = ids_or_features.to(device, non_blocking=True)
                    tmrca = tmrca.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    row_mask = row_mask.to(device, non_blocking=True)
                    cm_features = cm_features.to(device, non_blocking=True)
                    _, refined_logits = model.forward_chunks(
                        ids_or_features,
                        tmrca,
                        row_mask,
                        cm_features,
                        pad_id=pad_id,
                    )
                    loss = _masked_joint_loss(
                        refined_logits,
                        labels,
                        row_mask,
                        cm_features,
                        label_smoothing=label_smoothing,
                        loss_weighting=loss_weighting,
                    )
                    correct, total = compute_masked_row_accuracy(refined_logits, labels, row_mask)
                    val_loss_sum += float(loss.item())
                    val_batches += 1
                    val_correct += correct
                    val_total += total

            train_loss = train_loss_sum / max(1, train_batches)
            val_loss = val_loss_sum / max(1, val_batches)
            train_acc = (float(train_correct) / float(train_total)) if train_total else float("nan")
            val_acc = (float(val_correct) / float(val_total)) if val_total else float("nan")
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_epoch = epoch
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_left = patience
            else:
                patience_left -= 1
            logger.info(
                "Joint epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            _wandb_log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": best_val,
                    "learning_rate": float(joint_cfg["train"]["learning_rate"]),
                }
            )
            if not improved and patience_left <= 0:
                logger.info("Early stopping triggered.")
                break

        if best_state is None:
            raise RuntimeError("Joint training failed to capture a best checkpoint state")

        model.load_state_dict(best_state, strict=True)
        save_joint_model_checkpoint(
            ckpt_path,
            model=model,
            config=config,
            epoch=best_epoch,
            val_loss=best_val,
            run_name=run_name,
            gt_gru_archives=gt_gru_archives,
        )
        write_latest_checkpoint_pointer(Path(str(save_dir)), ckpt_path)
        if args.checkpoint_output:
            checkpoint_output = Path(args.checkpoint_output)
            checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_output.write_text(str(ckpt_path.resolve()) + "\n", encoding="utf-8")
        training_log_path = write_joint_training_data_log(
            ckpt_dir,
            config=config,
            train_pairs=str(train_pairs),
            val_pairs=str(val_pairs),
            available_predict=int(available_predict),
            founder_holdout_fraction=founder_holdout_fraction,
            sample_seed=sample_seed,
            train_max_subgraphs=train_max_subgraphs,
            val_max_subgraphs=val_max_subgraphs,
            train_sampling=train_sampling,
            val_sampling=val_sampling,
        )
        run_id = os.environ.get("WANDB_RUN_ID", "")
        wandb = _wandb_module()
        if wandb is not None and wandb.run is not None:
            run_id = wandb.run.id
        summary = {
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "checkpoint_path": str(ckpt_path),
            "learning_rate": float(joint_cfg["train"]["learning_rate"]),
            "num_refs": int(model_cfg.get("num_refs", 15)),
            "batch_size": int(joint_cfg["train"]["batch_size"]),
            "num_epochs": int(num_epochs),
            "early_stop_patience": int(patience),
            "weight_decay": float(joint_cfg["train"]["weight_decay"]),
            "label_smoothing": float(label_smoothing),
            "chunk_rows": int(joint_cfg["chunk_rows"]),
            "chunk_stride": int(joint_cfg["chunk_stride"]),
        }
        _update_wandb_summary(summary)
        _append_sweep_result(
            config,
            {
                **summary,
                "run_id": run_id,
                "run_name": run_name,
                "checkpoint_path": str(ckpt_path),
            },
        )
        logger.info("Wrote joint interval artifacts to %s", ckpt_dir)
        logger.info("Joint interval artifact paths: training_data=%s", str(training_log_path))
        logger.info("Saved joint prototype checkpoint to %s", ckpt_path)
        print(ckpt_path)
    finally:
        if finish_wandb:
            wandb = _wandb_module()
            if wandb is not None and wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    main()

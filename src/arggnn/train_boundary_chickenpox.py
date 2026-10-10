"""Hash-pinned validation-only county forecasting with one-sided boundary controls.

The immutable historical bank holds observed weeks [target-52,target). Following
week labels are external and rows 443 onward are masked before any downstream
geometry, scaling, bank binding, fitting or evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn

from .boundary_baselines import (BoundaryGRU, BoundaryKernelGNN, BoundaryTransformer,
                                 _interval_features)
from .boundary_chickenpox import (ChickenpoxBoundaryModel, bind_chickenpox_history,
                                  precompute_chickenpox_geometry)
from .chickenpox import ARCHIVE_SHA256, ForecastProtocol, TrainingScaler, load_chickenpox_archive
from .models import count_parameters
from .resources import ROOT, project_path, require_storage, set_research_title
from .train import set_seed

FROZEN_PROTOCOL = {"history": 52, "train_end": 365, "validation_end": 443, "observations": 522}
FROZEN_MODELS = ("boundary_full", "boundary_diagonal", "boundary_gain_matched",
                 "boundary_gain_leak", "boundary_bag", "boundary_kernel",
                 "boundary_gru", "boundary_transformer", "boundary_full_no_geography")
FROZEN_POLICY = {
    "hidden_dim": 32, "num_layers": 2, "latent_dim": 8, "epochs": 40, "patience": 8,
    "learning_rate": .001, "weight_decay": .0001, "initial_coupling": 20.,
    "initial_damping": 1., "temporal_layers": 2, "num_heads": 4, "dropout": 0.,
    "training_seed": 101,
    "selection": "minimum chronological validation MAE in raw case units",
    "training_loss": "MSE in per-county training-standardized units",
    "prediction_clipping": "none for every method", "final_test_evaluated": False,
}


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_pinned(path, expected, label):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"{label}: expected lowercase SHA256")
    path = project_path(path)
    content = path.read_bytes()
    if _sha(content) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return path, content


def _dump_json(path, value):
    path = project_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize_worker(device: str) -> torch.device:
    """Set the existing project title before making any CUDA query/context."""
    set_research_title()
    if device == "cuda":
        if os.environ.get("ARGGNN_PHYSICAL_GPU") not in {"2", "3", "4", "5"}:
            raise RuntimeError("Use the project GPU launcher")
        if not os.environ.get("ARGGNN_GPU_LEASE") or not os.environ.get("ARGGNN_GPU_LEASE_FD"):
            raise RuntimeError("Project GPU lease is required")
        lease = project_path(os.environ["ARGGNN_GPU_LEASE"])
        if os.fstat(int(os.environ["ARGGNN_GPU_LEASE_FD"])).st_ino != lease.stat().st_ino:
            raise RuntimeError("GPU lease descriptor mismatch")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one reserved project GPU")
    elif device == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU runs require explicitly empty CUDA_VISIBLE_DEVICES")
    else:
        raise ValueError("device must be cpu or cuda")
    torch.set_num_threads(1 if device == "cpu" else 4)
    return torch.device(device)


def restrict_validation_data(data, protocol):
    """Copy only permitted rows and mask the reserved tail before downstream use."""
    if (data.counts.shape != (protocol.observations, len(data.counties))
            or len(data.dates) != protocol.observations
            or not np.isfinite(data.counts[:protocol.validation_end]).all()):
        raise ValueError("Expected finite permitted observations and matching metadata")
    counts = np.full(data.counts.shape, np.nan, dtype=np.float64)
    counts[:protocol.validation_end] = data.counts[:protocol.validation_end]
    counts.setflags(write=False)
    return replace(data, counts=counts)


def _load_inputs(config, expected_config_sha256):
    config_path, config_bytes = _read_pinned(config, expected_config_sha256, "Boundary Chickenpox config")
    settings = json.loads(config_bytes)
    required = {"protocol_config", "protocol_config_sha256", "source_archive",
                "source_archive_sha256", "source_manifest", "source_manifest_sha256",
                "stage", "interpretation", "models", *FROZEN_POLICY}
    if not isinstance(settings, dict) or not required.issubset(settings):
        raise ValueError("Boundary config lacks required source/protocol/model pins")
    if settings["models"] != list(FROZEN_MODELS):
        raise ValueError("Frozen boundary model list/order differs")
    for key, value in FROZEN_POLICY.items():
        actual = settings[key]
        if actual != value or (type(value) is int and type(actual) is not int) or (
                type(value) is bool and type(actual) is not bool) or (
                type(value) is float and (type(actual) not in (int, float))):
            raise ValueError(f"Frozen boundary training policy differs: {key}")
    protocol_path, protocol_bytes = _read_pinned(
        settings["protocol_config"], settings["protocol_config_sha256"], "Common protocol")
    common = json.loads(protocol_bytes)
    if any(type(common.get(k)) is not int or common.get(k) != v for k, v in FROZEN_PROTOCOL.items()):
        raise ValueError("Common protocol differs from frozen chronological boundaries")
    for key in ("epochs", "patience", "learning_rate", "weight_decay", "training_loss",
                "selection", "prediction_clipping", "final_test_evaluated"):
        if common.get(key) != FROZEN_POLICY[key]:
            raise ValueError(f"Common protocol differs from frozen training policy: {key}")
    if settings["source_archive_sha256"] != ARCHIVE_SHA256 or common.get("source_archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("Frozen source archive pin differs")
    archive_path, archive_bytes = _read_pinned(settings["source_archive"], ARCHIVE_SHA256, "Source archive")
    manifest_path, manifest_bytes = _read_pinned(
        settings["source_manifest"], settings["source_manifest_sha256"], "Source manifest")
    manifest = json.loads(manifest_bytes)
    if (manifest.get("archive_sha256") != ARCHIVE_SHA256
            or manifest.get("archive_bytes") != len(archive_bytes)
            or manifest.get("observations") != 522 or manifest.get("license") != "CC BY 4.0"):
        raise ValueError("Source manifest disagrees with pinned archive/protocol")
    loaded = load_chickenpox_archive(archive_path)
    if loaded.archive_sha256 != ARCHIVE_SHA256 or list(loaded.counties) != manifest.get("counties"):
        raise ValueError("Loaded archive identity/county order differs")
    protocol = ForecastProtocol(**FROZEN_PROTOCOL)
    # The loader verifies all source bytes, but no final-test numeric values
    # are returned to geometry, scaler, bank binding, fitting or evaluation.
    data = restrict_validation_data(loaded, protocol)
    return {"settings": settings, "config_path": config_path, "config_bytes": config_bytes,
            "protocol_path": protocol_path, "protocol_bytes": protocol_bytes,
            "archive_path": archive_path, "manifest_path": manifest_path,
            "manifest_bytes": manifest_bytes, "protocol": protocol, "data": data}


class OneSidedBoundaryKernel(BoundaryKernelGNN):
    """Left-supported exponential interval kernel on independently encoded weeks."""
    def forward(self, bank):
        leaf, _ = self.prepare(bank)
        tau = (torch.nn.functional.softplus(self.raw_tau) + 1e-4).to(bank.intervals.dtype)
        left, right = bank.intervals.unbind(-1)
        p = bank.positions[:, None]
        stop = torch.minimum(right[None], p)
        duration = (stop-left[None]).clamp_min(0)
        valid = duration > 0
        # Positive valid widths use expm1. Inactive intervals use width one
        # only for a finite pre-mask logarithm, without injecting kernel mass.
        safe = torch.where(valid, duration, torch.ones_like(duration))
        log_mass = -(p-stop)/tau + torch.log(-torch.expm1(-safe/tau))
        weight = log_mass.masked_fill(~valid, -torch.inf).softmax(-1).to(leaf.dtype)
        context = torch.einsum("pg,gqh->pqh", weight, leaf)
        return self.head(torch.cat((leaf[bank.target_graph], context), -1))


class OneSidedBoundaryGRU(BoundaryGRU):
    """Unidirectional record GRU with chronological observed-week readout."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=False)
        self.head = nn.Linear(2*self.hidden_dim, self.num_classes)


class OneSidedBoundaryTransformer(BoundaryTransformer):
    """Causal record attention and past-only fixed-coordinate cross attention."""
    def forward(self, bank):
        local, graph = self.prepare(bank)
        tokens = self.token(torch.cat((local, graph[:, None].expand_as(local)), -1))
        tokens = tokens + self.interval_encoder(_interval_features(bank, local.dtype))[:, None]
        order = torch.argsort(bank.intervals[:, 0])
        causal = torch.ones(len(order), len(order), dtype=torch.bool, device=tokens.device).triu(1)
        memory = self.temporal(tokens[order].transpose(0, 1), mask=causal)
        p = bank.positions.to(local.dtype)
        coordinate = torch.stack((p, torch.sin(2*torch.pi*p), torch.cos(2*torch.pi*p)), -1)
        leaf = local[bank.target_graph].transpose(0, 1)
        position = self.coordinate(coordinate)[None].expand_as(leaf)
        query = self.query(torch.cat((leaf, position), -1))
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        mask = torch.arange(len(order), device=order.device)[None] > inverse[bank.target_graph][:, None]
        context, _ = self.attention(query, memory, memory, attn_mask=mask, need_weights=False)
        return self.head(torch.cat((leaf, self.norm(query+context)), -1)).transpose(0, 1).contiguous()


def make_forecaster(name, settings):
    common = {"input_dim": 1, "num_classes": 1,
              "hidden_dim": settings["hidden_dim"], "num_layers": settings["num_layers"]}
    if name == "boundary_kernel":
        return OneSidedBoundaryKernel(**common)
    if name == "boundary_gru":
        return OneSidedBoundaryGRU(**common)
    if name == "boundary_transformer":
        return OneSidedBoundaryTransformer(**common, temporal_layers=settings["temporal_layers"],
            num_heads=settings["num_heads"], dropout=settings["dropout"])
    variants = {"boundary_full": ("full", "ordered"),
                "boundary_full_no_geography": ("full", "ordered"),
                "boundary_diagonal": ("diagonal", "ordered"),
                "boundary_gain_matched": ("gain_matched_diagonal", "ordered"),
                "boundary_gain_leak": ("gain_leak_diagonal", "ordered"),
                "boundary_bag": ("full", "homogeneous_bag")}
    if name not in variants:
        raise ValueError("Unknown frozen forecast model")
    coupling, aggregation = variants[name]
    return ChickenpoxBoundaryModel(hidden_dim=settings["hidden_dim"], num_layers=settings["num_layers"],
        latent_dim=settings["latent_dim"], coupling=coupling, aggregation=aggregation,
        initial_coupling=settings["initial_coupling"], initial_damping=settings["initial_damping"],
        validate=False)


def regression_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if (prediction.ndim != 2 or prediction.shape != truth.shape or min(prediction.shape) < 1
            or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        raise ValueError("Expected finite matching nonempty [weeks,counties] predictions and truth")
    error = prediction-truth
    return {"mae_cases": float(np.abs(error).mean()), "rmse_cases": float(np.sqrt(np.square(error).mean())),
            "per_county_mae": np.abs(error).mean(0).tolist(),
            "per_county_rmse": np.sqrt(np.square(error).mean(0)).tolist(),
            "forecast_weeks": len(error)}


def evaluate(model, banks, targets, scaler, truth):
    model.eval()
    with torch.no_grad():
        standardized = np.stack([model(banks[int(t)])[0, :, 0].detach().cpu().numpy() for t in targets])
    prediction = scaler.inverse(standardized)
    return prediction, regression_metrics(prediction, truth)


def _model_manifest(model, name, settings, example):
    if isinstance(model, ChickenpoxBoundaryModel):
        record = model.configuration_manifest(example)
    else:
        record = {"kind": "independent_same_boundary_input_one_sided_forecast_control",
                  "control": name, "parameters": count_parameters(model),
                  "num_outputs": 1, "reference_nodes": 0, "input_dim": 1,
                  "hidden_dim": settings["hidden_dim"], "num_layers": settings["num_layers"],
                  "learned_numerical_id_embeddings": False, "target_label_in_bank": False}
        if name == "boundary_kernel":
            record["temporal_control"] = "normalized left-supported exponential integral, learned scalar tau"
        elif name == "boundary_gru":
            record["temporal_control"] = "unidirectional GRU in chronological record order, four observed interval features"
        else:
            record.update(temporal_control="causally masked Transformer, past-only cross attention",
                temporal_layers=settings["temporal_layers"], num_heads=settings["num_heads"],
                dropout=settings["dropout"])
    record.update(control=name, temporal_direction="one_sided",
        anchor="final observed interval midpoint, next-week target is external",
        bag_observations="all 52 available observed holds, chronology ignored" if name == "boundary_bag" else None,
        runner_adapter_sha256=_sha(Path(__file__).read_bytes()))
    record.pop("configuration_sha256", None)
    record["configuration_sha256"] = _sha(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
    return record


def _fit_model(model, banks, data, scaler, protocol, settings, checkpoint, config_sha):
    train, validation = protocol.targets("train"), protocol.targets("validation")
    target_values = scaler.transform(data.counts[train]).astype(np.float32)
    targets = {int(t): torch.as_tensor(target_values[i], device=next(model.parameters()).device)
               for i, t in enumerate(train)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    best, best_epoch, stale, history = float("inf"), -1, 0, []
    started = time.monotonic()
    for epoch in range(settings["epochs"]):
        require_storage(headroom_bytes=50_000_000)
        model.train()
        order = list(map(int, train))
        random.shuffle(order)
        losses = []
        for t in order:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(banks[t])[0, :, 0]
            loss = torch.nn.functional.mse_loss(prediction, targets[t])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite standardized forecast loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        _, metrics = evaluate(model, banks, validation, scaler, data.counts[validation])
        record = {"epoch": epoch, "train_mse_standardized": float(np.mean(losses)), **metrics,
                  "elapsed_seconds": time.monotonic()-started}
        history.append(record)
        print(json.dumps({"event": "epoch", "model": settings.get("active_model", type(model).__name__), **record}, allow_nan=False), flush=True)
        if metrics["mae_cases"] < best:
            best, best_epoch, stale = metrics["mae_cases"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mae": best,
                        "study_config_sha256": config_sha}, checkpoint)
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    chosen = torch.load(checkpoint, map_location=next(model.parameters()).device, weights_only=True)
    if chosen["study_config_sha256"] != config_sha:
        raise RuntimeError("Checkpoint frozen configuration pin differs")
    model.load_state_dict(chosen["model"], strict=True)
    prediction, metrics = evaluate(model, banks, validation, scaler, data.counts[validation])
    return prediction, {"best_epoch": best_epoch, "history": history,
                        "training_seconds": time.monotonic()-started, **metrics}


def run_study(config, output, expected_config_sha256, device="cpu", models=None):
    device = initialize_worker(device)
    output = project_path(output)
    if output.exists():
        raise RuntimeError("Use a fresh versioned output directory")
    inputs = _load_inputs(config, expected_config_sha256)
    settings, data, protocol = inputs["settings"], inputs["data"], inputs["protocol"]
    requested = list(FROZEN_MODELS if models is None else models)
    if not requested or len(set(requested)) != len(requested) or any(n not in FROZEN_MODELS for n in requested):
        raise ValueError("Choose distinct nonempty models inside frozen protocol")
    require_storage(headroom_bytes=200_000_000)
    # Masking occurred in _load_inputs before the following scaler or geometry.
    scaler = TrainingScaler.fit(data.counts[:protocol.train_end], protocol.train_end)
    train, validation = protocol.targets("train"), protocol.targets("validation")
    all_targets = np.concatenate((train, validation))
    geometries = {}
    banks = {}
    for no_geography in dict.fromkeys(n.endswith("_no_geography") for n in requested):
        geometry = precompute_chickenpox_geometry(data, protocol.history, no_geography=no_geography)
        geometry = geometry.to(device, torch.float32)
        geometries[no_geography] = geometry.configuration_manifest()
        banks[no_geography] = {int(t): bind_chickenpox_history(geometry, data, scaler, int(t))
                               for t in all_targets}
    output.mkdir(parents=True, exist_ok=False)
    (output/"study_config.json").write_bytes(inputs["config_bytes"])
    (output/"common_protocol_config.json").write_bytes(inputs["protocol_bytes"])
    (output/"source_manifest.json").write_bytes(inputs["manifest_bytes"])
    checkpoint_id = _sha(str(output.relative_to(ROOT)).encode())[:16]
    checkpoint_root = project_path(Path("checkpoints")/(output.name+"_"+checkpoint_id))
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    code_files = ("train_boundary_chickenpox.py", "boundary_chickenpox.py", "boundary_baselines.py",
                  "boundary_memory.py", "kron_interface.py", "chickenpox.py", "resources.py", "train.py")
    implementation_sha = {n: _sha((Path(__file__).parent/n).read_bytes()) for n in code_files}
    result = {"status": "running", "stage": settings["stage"], "interpretation": settings["interpretation"],
        "evaluation_split": "validation_only", "final_test_evaluated": False,
        "study_config_sha256": expected_config_sha256,
        "protocol_config_sha256": settings["protocol_config_sha256"],
        "source_archive_sha256": ARCHIVE_SHA256, "source_manifest_sha256": settings["source_manifest_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": implementation_sha, "training_seed": settings["training_seed"],
        "county_order": list(data.counties), "scaler_fit_range": [0, protocol.train_end],
        "scaler_mean": scaler.mean.tolist(), "scaler_scale": scaler.scale.tolist(),
        "train_target_range": [protocol.history, protocol.train_end],
        "validation_target_range": [protocol.train_end, protocol.validation_end],
        "reserved_numeric_rows_masked_from": protocol.validation_end,
        "history_weeks": protocol.history, "forecast_horizon_weeks": 1,
        "selection": settings["selection"], "prediction_clipping": "none",
        "predeclared_models": list(FROZEN_MODELS), "requested_models": requested,
        "geometry_configurations": {str(k): v for k, v in geometries.items()},
        "physical_gpu": os.environ.get("ARGGNN_PHYSICAL_GPU") if device.type == "cuda" else None,
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__,
                        "python": os.sys.version, "cuda": torch.version.cuda,
                        "numerical_threads": torch.get_num_threads()},
        "models": {}}
    _dump_json(output/"progress.json", result)
    started = time.monotonic()
    try:
        for name in requested:
            set_seed(settings["training_seed"])
            model = make_forecaster(name, settings).to(device)
            selected_banks = banks[name.endswith("_no_geography")]
            example = selected_banks[int(train[0])]
            model_manifest = _model_manifest(model, name, settings, example)
            checkpoint = checkpoint_root/(name+".pt")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            print(json.dumps({"event": "model_start", "model": name,
                              "parameters": count_parameters(model)}), flush=True)
            prediction, fitted = _fit_model(model, selected_banks, data, scaler, protocol,
                                            {**settings, "active_model": name}, checkpoint, expected_config_sha256)
            # Prevent completion receipts from silently describing changed code.
            if implementation_sha != {n: _sha((Path(__file__).parent/n).read_bytes()) for n in code_files}:
                raise RuntimeError("Implementation changed during active frozen training run")
            saved = checkpoint_root/(name+"_validation_predictions.npz")
            np.savez_compressed(saved, prediction=prediction, truth=data.counts[validation],
                target_indices=validation, target_dates=np.asarray([data.dates[int(t)] for t in validation]),
                counties=np.asarray(data.counties), study_config_sha256=np.asarray(expected_config_sha256))
            error = prediction-data.counts[validation]
            weekly = [{"target_index": int(t), "target_date": data.dates[int(t)],
                       "history_range": [int(t)-protocol.history, int(t)],
                       "mae_cases": float(np.abs(error[i]).mean()),
                       "rmse_cases": float(np.sqrt(np.square(error[i]).mean()))}
                      for i, t in enumerate(validation)]
            model_result = {"model": name, "parameters": count_parameters(model),
                "model_configuration": model_manifest, **fitted, "weekly_validation_metrics": weekly,
                "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint.read_bytes()),
                "validation_predictions": str(saved), "validation_predictions_sha256": _sha(saved.read_bytes()),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
            result["models"][name] = model_result
            _dump_json(output/(name+".json"), model_result)
            _dump_json(output/"progress.json", result)
            print(json.dumps({"event": "model_complete", "model": name,
                              "mae_cases": model_result["mae_cases"]}), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result["status"] = "complete"
        result["all_predeclared_models_completed_in_this_run"] = set(requested) == set(FROZEN_MODELS)
        result["total_seconds"] = time.monotonic()-started
        _dump_json(output/"results.json", result)
        _dump_json(output/"progress.json", result)
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        result["total_seconds"] = time.monotonic()-started
        _dump_json(output/"progress.json", result)
        raise
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--models", nargs="+", help="Distinct subset of the frozen model list")
    return parser


def main():
    args = make_parser().parse_args()
    run_study(args.config, args.output, args.expected_config_sha256, args.device, args.models)


if __name__ == "__main__":
    main()


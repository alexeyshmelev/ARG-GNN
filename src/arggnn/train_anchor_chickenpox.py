"""Direct fixed-anchor GNN history encoding for external next-week count forecasts.

The complete 52-week observed history is available before the forecast target.
The unchanged AnchorKernelModel uses bidirectional completed kernels and depth
inside that history. No internal earlier-anchor causality claim is made.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from .anchor_kernel import AnchorKernelModel, prepare_anchor_kernel_bank
from .boundary_chickenpox import bind_chickenpox_history, precompute_chickenpox_geometry
from .chickenpox import ARCHIVE_SHA256, ForecastProtocol, TrainingScaler, load_chickenpox_archive
from .models import count_parameters
from .resources import ROOT, project_path, require_storage
from .train import set_seed
from .train_boundary_chickenpox import (FROZEN_PROTOCOL, _dump_json, _read_pinned, _sha,
    evaluate, initialize_worker, regression_metrics, restrict_validation_data)

FROZEN_MODELS = ("anchor_full_d1", "anchor_full_d2", "anchor_diagonal_d1", "anchor_diagonal_d2",
                 "anchor_gain_matched_d1", "anchor_gain_matched_d2",
                 "anchor_scalar_d1", "anchor_scalar_d2", "anchor_full_d2_no_geography")
FROZEN_SPECS = {}
for _name in FROZEN_MODELS:
    _coupling = ("gain_matched_diagonal" if "gain_matched" in _name else
                 ("diagonal" if "diagonal" in _name else ("scalar" if "scalar" in _name else "full")))
    FROZEN_SPECS[_name] = {"geometry": "plain", "coupling": _coupling,
        "kernel_depth": 1 if _name.endswith("_d1") else 2, "rate_parameterization": "log",
        "no_geography": _name.endswith("_no_geography")}
FROZEN_POLICY = {
    "hidden_dim": 32, "num_layers": 2, "latent_dim": 8, "num_anchors": 52, "num_scales": 3,
    "initial_coupling": [5., 20., 80.], "initial_damping": [1., 1., 1.],
    "epochs": 40, "patience": 8, "learning_rate": .001, "weight_decay": .0001,
    "rate_weight_decay": 0., "training_seed": 101,
    "selection": "minimum chronological validation MAE in raw case units",
    "training_loss": "MSE in per-county training-standardized units",
    "prediction_clipping": "none for every method", "final_test_evaluated": False,
    "history_encoder": "completed bidirectional fixed-anchor kernels over all observed history",
}

CODE_FILES = ("train_anchor_chickenpox.py", "anchor_kernel.py", "boundary_chickenpox.py",
    "boundary_memory.py", "kron_interface.py", "train_boundary_chickenpox.py",
    "boundary_baselines.py", "chickenpox.py", "models.py", "resources.py", "train.py")


def _implementation_hashes():
    return {name: _sha((Path(__file__).parent/name).read_bytes()) for name in CODE_FILES}


def _load_inputs(config, expected_config_sha256):
    path, content = _read_pinned(config, expected_config_sha256, "Anchor Chickenpox config")
    settings = json.loads(content)
    required = {"protocol_config", "protocol_config_sha256", "source_archive", "source_archive_sha256",
                "source_manifest", "source_manifest_sha256", "stage", "interpretation",
                "models", "model_specs", *FROZEN_POLICY}
    if not isinstance(settings, dict) or not required.issubset(settings):
        raise ValueError("Anchor config lacks required frozen source/protocol/specification pins")
    if settings["models"] != list(FROZEN_MODELS) or settings["model_specs"] != FROZEN_SPECS:
        raise ValueError("Frozen direct-anchor model list/specifications differ")
    if any(type(spec["kernel_depth"]) is not int or type(spec["no_geography"]) is not bool
           for spec in settings["model_specs"].values()):
        raise ValueError("Frozen direct-anchor model specification types differ")
    for key, expected in FROZEN_POLICY.items():
        actual = settings[key]
        if (actual != expected or (type(expected) in (bool, int) and type(actual) is not type(expected))
                or (type(expected) is float and type(actual) not in (int, float))
                or (type(expected) is list and any(type(v) not in (int, float) for v in actual))):
            raise ValueError(f"Frozen anchor training policy differs: {key}")
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
    restricted = restrict_validation_data(loaded, protocol)
    return {"settings": settings, "config_path": path, "config_bytes": content,
            "protocol_path": protocol_path, "protocol_bytes": protocol_bytes,
            "archive_path": archive_path, "manifest_path": manifest_path,
            "manifest_bytes": manifest_bytes, "protocol": protocol, "data": restricted}


def prepare_histories(data, protocol, scaler, settings, device, *, no_geography=False):
    """Reuse one static plain anchor geometry while binding permitted histories."""
    if not np.isnan(data.counts[protocol.validation_end:]).all():
        raise ValueError("Reserved numeric rows must be masked before geometry/binding")
    geometry = precompute_chickenpox_geometry(data, protocol.history, no_geography=no_geography)
    prepared = prepare_anchor_kernel_bank(geometry.template,
        num_anchors=settings["num_anchors"], num_scales=settings["num_scales"]).to(device, torch.float32)
    # Binding shares the exact already-transferred template and query cache.
    geometry = replace(geometry, template=prepared.bank)
    targets = np.concatenate((protocol.targets("train"), protocol.targets("validation")))
    histories = {}
    for target in targets:
        bank = bind_chickenpox_history(geometry, data, scaler, int(target))
        histories[int(target)] = replace(prepared, bank=bank)
    manifest = geometry.configuration_manifest()
    manifest.update(anchor_count=settings["num_anchors"], branches=settings["num_scales"],
        branch_operators="three identical plain geography operators with learned separate rates/mixtures",
        anchor_graph_indexes=prepared.layout.anchor_graph.cpu().tolist(),
        fixed_anchor_positions=prepared.layout.anchors.cpu().tolist(),
        fixed_anchor_weights=prepared.layout.weights.cpu().tolist(),
        readout_positions=prepared.bank.positions.cpu().tolist(),
        readout_graph_indexes=prepared.bank.target_graph.cpu().tolist(),
        internal_context="completed bidirectional over fully observed history, not internally causal")
    return histories, manifest


def make_forecaster(name, settings):
    if name not in settings["model_specs"]:
        raise ValueError("Unknown direct-anchor forecast model")
    spec = {key: value for key, value in settings["model_specs"][name].items() if key != "no_geography"}
    return AnchorKernelModel(input_dim=1, num_classes=1,
        hidden_dim=settings["hidden_dim"], num_layers=settings["num_layers"],
        latent_dim=settings["latent_dim"], num_anchors=settings["num_anchors"],
        num_scales=settings["num_scales"], initial_coupling=settings["initial_coupling"],
        initial_damping=settings["initial_damping"], **spec)


def optimizer_groups(model, weight_decay):
    special = list(model.rate_and_mixture_parameters())
    identities = {id(p) for p in special}
    ordinary = [p for p in model.parameters() if id(p) not in identities]
    if len(identities) != len(special) or not identities <= {id(p) for p in model.parameters()}:
        raise RuntimeError("Invalid or duplicate rate/mixture parameter group")
    return [{"params": ordinary, "weight_decay": weight_decay},
            {"params": special, "weight_decay": 0.}]


def _gradient_norms(model):
    result = {}
    for key, attribute in (("coupling", "raw_coupling"), ("damping", "raw_damping"),
                           ("mixture", "mixture_logits")):
        gradients = [getattr(layer, attribute).grad for layer in model.layers]
        if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError(f"Missing/nonfinite anchor {key} optimization gradient")
        result[key] = float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt())
    return result


def parameter_activity(model, prepared, no_geography):
    """Disclose structural inactive blocks, without treating zero gradients as frozen."""
    total = count_parameters(model)
    refs = prepared.bank.query_cache.reference_ids.numel()
    inactive_reference = (model.hidden_dim+1)*model.latent_dim*model.kernel_depth if refs == 0 else 0
    # In full C=0 geometry, lambda multiplies the zero operator and cannot
    # affect numerator or mass. The damping and mixture remain trainable.
    inactive_neighbors = sum(layer.neighbor_linear.weight.numel() for layer in model.encoder.layers) if no_geography else 0
    inactive_coupling = model.num_scales*model.latent_dim*model.kernel_depth if (
        no_geography and model.coupling == "full") else 0
    return {"parameters": total, "structurally_inactive_reference_parameters": inactive_reference,
        "structurally_inactive_zero_geography_coupling_parameters": inactive_coupling,
        "structurally_inactive_zero_geography_neighbor_parameters": inactive_neighbors,
        "structurally_active_parameters": total-inactive_reference-inactive_coupling-inactive_neighbors,
        "activity_scope": "structural exclusions only, not effective degrees of freedom or empirical nonzero-gradient count"}


def _fit_model(model, banks, data, scaler, protocol, settings, checkpoint, config_sha):
    train, validation = protocol.targets("train"), protocol.targets("validation")
    values = scaler.transform(data.counts[train]).astype(np.float32)
    targets = {int(t): torch.as_tensor(values[i], device=next(model.parameters()).device)
               for i, t in enumerate(train)}
    optimizer = torch.optim.AdamW(optimizer_groups(model, settings["weight_decay"]),
                                  lr=settings["learning_rate"])
    initial_rates = [layer.rates.detach().clone() for layer in model.layers]
    initial_damping = [layer.damping.detach().clone() for layer in model.layers]
    best, best_epoch, stale, history = float("inf"), -1, 0, []
    started = time.monotonic()
    for epoch in range(settings["epochs"]):
        require_storage(headroom_bytes=50_000_000)
        model.train()
        order = list(map(int, train))
        random.shuffle(order)
        losses, gradients = [], {"coupling": [], "damping": [], "mixture": []}
        for target in order:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(banks[target])[0, :, 0]
            loss = torch.nn.functional.mse_loss(prediction, targets[target])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite standardized direct-anchor forecast loss")
            loss.backward()
            norms = _gradient_norms(model)
            for key in gradients:
                gradients[key].append(norms[key])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        _, metrics = evaluate(model, banks, validation, scaler, data.counts[validation])
        optimization = {
            "gradient_norm_before_clip_mean": {key: float(np.mean(items)) for key, items in gradients.items()},
            "gradient_norm_before_clip_max": {key: max(items) for key, items in gradients.items()},
            "maximum_relative_rate_change": max(float((layer.rates.detach()/initial-1).abs().max())
                for layer, initial in zip(model.layers, initial_rates)),
            "maximum_relative_damping_change": max(float((layer.damping.detach()/initial-1).abs().max())
                for layer, initial in zip(model.layers, initial_damping)),
            "current_rates": [layer.rates.detach().cpu().tolist() for layer in model.layers],
            "current_damping": [layer.damping.detach().cpu().tolist() for layer in model.layers],
            "current_mixture": [layer.mixture.detach().cpu().tolist() for layer in model.layers],
        }
        record = {"epoch": epoch, "train_mse_standardized": float(np.mean(losses)),
                  **metrics, "kernel_optimization": optimization,
                  "elapsed_seconds": time.monotonic()-started}
        history.append(record)
        print(json.dumps({"event": "epoch", "model": settings.get("active_model"),
                          **record}, allow_nan=False), flush=True)
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
        raise ValueError("Choose distinct nonempty models inside frozen direct-anchor protocol")
    require_storage(headroom_bytes=200_000_000)
    scaler = TrainingScaler.fit(data.counts[:protocol.train_end], protocol.train_end)
    train, validation = protocol.targets("train"), protocol.targets("validation")
    banks, geometries = {}, {}
    for no_geography in dict.fromkeys(settings["model_specs"][n]["no_geography"] for n in requested):
        banks[no_geography], geometries[no_geography] = prepare_histories(
            data, protocol, scaler, settings, device, no_geography=no_geography)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in (("study_config.json", inputs["config_bytes"]),
                          ("common_protocol_config.json", inputs["protocol_bytes"]),
                          ("source_manifest.json", inputs["manifest_bytes"])):
        (output/name).write_bytes(content)
    checkpoint_id = _sha(str(output.relative_to(ROOT)).encode())[:16]
    checkpoint_root = project_path(Path("checkpoints")/(output.name+"_"+checkpoint_id))
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    code_hashes = _implementation_hashes()
    result = {"status": "running", "stage": settings["stage"], "interpretation": settings["interpretation"],
        "evaluation_split": "validation_only", "final_test_evaluated": False,
        "study_config_sha256": expected_config_sha256,
        "protocol_config_sha256": settings["protocol_config_sha256"],
        "source_archive_sha256": ARCHIVE_SHA256, "source_manifest_sha256": settings["source_manifest_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": code_hashes, "training_seed": settings["training_seed"],
        "county_order": list(data.counties), "scaler_fit_range": [0, protocol.train_end],
        "scaler_mean": scaler.mean.tolist(), "scaler_scale": scaler.scale.tolist(),
        "train_target_range": [protocol.history, protocol.train_end],
        "validation_target_range": [protocol.train_end, protocol.validation_end],
        "reserved_numeric_rows_masked_from": protocol.validation_end,
        "history_weeks": protocol.history, "forecast_horizon_weeks": 1,
        "history_encoder": settings["history_encoder"],
        "internal_earlier_anchor_causal": False,
        "selection": settings["selection"], "prediction_clipping": "none",
        "rate_weight_decay": 0., "predeclared_models": list(FROZEN_MODELS), "requested_models": requested,
        "geometry_configurations": {str(k): v for k, v in geometries.items()},
        "physical_gpu": os.environ.get("ARGGNN_PHYSICAL_GPU") if device.type == "cuda" else None,
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__,
                       "python": os.sys.version, "cuda": torch.version.cuda,
                       "numerical_threads": torch.get_num_threads()}, "models": {}}
    _dump_json(output/"progress.json", result)
    started = time.monotonic()
    try:
        for name in requested:
            set_seed(settings["training_seed"])
            model = make_forecaster(name, settings).to(device)
            no_geography = settings["model_specs"][name]["no_geography"]
            selected_banks = banks[no_geography]
            example = selected_banks[int(train[0])]
            initial_configuration = model.configuration_manifest(example)
            checkpoint = checkpoint_root/(name+".pt")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            print(json.dumps({"event": "model_start", "model": name,
                              **parameter_activity(model, example, no_geography)}), flush=True)
            prediction, fitted = _fit_model(model, selected_banks, data, scaler, protocol,
                {**settings, "active_model": name}, checkpoint, expected_config_sha256)
            if code_hashes != _implementation_hashes():
                raise RuntimeError("Direct-anchor implementation changed during active frozen training")
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
            model_result = {"model": name, **parameter_activity(model, example, no_geography),
                "initial_model_configuration": initial_configuration,
                "model_configuration": model.configuration_manifest(example),
                **fitted, "weekly_validation_metrics": weekly,
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
    parser.add_argument("--models", nargs="+", help="Distinct subset of the frozen direct-anchor models")
    return parser


def main():
    args = make_parser().parse_args()
    run_study(args.config, args.output, args.expected_config_sha256, args.device, args.models)


if __name__ == "__main__":
    main()

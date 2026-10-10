"""Hash-pinned CPU GP training/validation runner, with no final-test option."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from .chickenpox import ARCHIVE_SHA256, ForecastProtocol, TrainingScaler, load_chickenpox_archive
from .graph_gp import GraphOUForecaster
from .resources import ROOT, project_path, require_storage, set_research_title

FROZEN_VARIANTS = ("nonseparable", "separable", "temporal")
FROZEN_PROTOCOL = {"history": 52, "train_end": 365, "validation_end": 443, "observations": 522}
FROZEN_OPTIMIZER = {"seed": 101, "restarts": 3, "maxiter": 100}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_pinned(path: str | Path, expected: str, label: str) -> tuple[Path, bytes]:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"{label}: expected a lowercase SHA256 digest")
    path = project_path(path)
    content = path.read_bytes()
    actual = _sha256(content)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, found {actual}")
    return path, content


def _dump_json(path: Path, value: dict) -> None:
    path = project_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_inputs(config: str | Path, expected_config_sha256: str) -> dict:
    """Verify every pinned input before any fit or output-directory creation."""
    config_path, config_bytes = _read_pinned(config, expected_config_sha256, "GP study config")
    study = json.loads(config_bytes)
    required = {
        "protocol_config", "protocol_config_sha256", "source_archive", "source_archive_sha256",
        "source_manifest", "source_manifest_sha256", "variants", "seed", "restarts", "maxiter",
    }
    if not isinstance(study, dict) or not required.issubset(study):
        raise ValueError("GP study config is missing required protocol/source/optimizer pins")
    if "stage" in study and (not isinstance(study["stage"], str) or not study["stage"].strip()):
        raise ValueError("Optional study stage must be a nonempty descriptive string")
    if study["variants"] != list(FROZEN_VARIANTS):
        raise ValueError("All three GP variants must be predeclared in the frozen order")
    if any(type(study[key]) is not int or study[key] != value
           for key, value in FROZEN_OPTIMIZER.items()):
        raise ValueError("Frozen GP optimizer requires seed=101, restarts=3, maxiter=100")
    protocol_path, protocol_bytes = _read_pinned(
        study["protocol_config"], study["protocol_config_sha256"], "Common protocol config")
    protocol_settings = json.loads(protocol_bytes)
    if any(type(protocol_settings.get(key)) is not int or protocol_settings.get(key) != value
           for key, value in FROZEN_PROTOCOL.items()):
        raise ValueError("Common protocol differs from the frozen Chickenpox boundaries")
    if protocol_settings.get("final_test_evaluated") is not False:
        raise ValueError("Common protocol must explicitly leave final test unevaluated")
    if (study["source_archive_sha256"] != ARCHIVE_SHA256
            or protocol_settings.get("source_archive_sha256") != ARCHIVE_SHA256):
        raise ValueError("Study/common protocol archive pin differs from the benchmark archive")
    archive_path, archive_bytes = _read_pinned(
        study["source_archive"], study["source_archive_sha256"], "Source archive")
    manifest_path, manifest_bytes = _read_pinned(
        study["source_manifest"], study["source_manifest_sha256"], "Source manifest")
    manifest = json.loads(manifest_bytes)
    if (manifest.get("archive_sha256") != ARCHIVE_SHA256
            or manifest.get("archive_bytes") != len(archive_bytes)
            or manifest.get("observations") != FROZEN_PROTOCOL["observations"]
            or manifest.get("license") != "CC BY 4.0"):
        raise ValueError("Source manifest metadata disagrees with the pinned archive/protocol")
    data = load_chickenpox_archive(archive_path)
    if (data.archive_sha256 != ARCHIVE_SHA256
            or data.counts.shape != (FROZEN_PROTOCOL["observations"], len(data.counties))
            or list(data.counties) != manifest.get("counties")):
        raise ValueError("Loaded source shape/county order disagrees with the source manifest")
    protocol = ForecastProtocol(**FROZEN_PROTOCOL)
    return {
        "study": study, "config_path": config_path, "config_bytes": config_bytes,
        "protocol_path": protocol_path, "protocol_bytes": protocol_bytes,
        "archive_path": archive_path, "manifest_path": manifest_path, "manifest_bytes": manifest_bytes,
        "manifest": manifest, "protocol_settings": protocol_settings, "protocol": protocol, "data": data,
    }


def regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if (prediction.ndim != 2 or prediction.shape != truth.shape or prediction.shape[0] < 1
            or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        raise ValueError("Expected finite matching validation [weeks,counties] prediction and truth")
    error = prediction - truth
    return {
        "mae_cases": float(np.abs(error).mean()),
        "rmse_cases": float(np.sqrt(np.square(error).mean())),
        "per_county_mae": np.abs(error).mean(axis=0).tolist(),
        "per_county_rmse": np.sqrt(np.square(error).mean(axis=0)).tolist(),
        "forecast_weeks": len(error),
    }


def _implementation_hashes() -> dict:
    return {name: _sha256((Path(__file__).parent / name).read_bytes()) for name in (
        "train_graph_gp.py", "graph_gp.py", "chickenpox.py", "resources.py")}


def run_study(config: str | Path, output: str | Path, expected_config_sha256: str) -> dict:
    """Execute only the frozen three-variant training/validation GP pilot."""
    set_research_title()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("GP CPU runner requires explicitly empty CUDA_VISIBLE_DEVICES")
    output = project_path(output)
    if output.exists():
        raise RuntimeError("Use a fresh versioned output directory")
    inputs = _load_inputs(config, expected_config_sha256)
    require_storage(headroom_bytes=100_000_000)
    data, protocol = inputs["data"], inputs["protocol"]
    targets = protocol.targets("validation")  # Never ask the protocol for test targets.
    scaler = TrainingScaler.fit(data.counts[:protocol.train_end], protocol.train_end)
    # Mark post-validation rows unreadable even to a accidentally changed downstream fit.
    # The pinned archive loader verifies the complete source, but final-test values are
    # not supplied to GP fitting or forecasting.
    restricted_counts = np.full(data.counts.shape, np.nan, dtype=np.float64)
    restricted_counts[:protocol.validation_end] = data.counts[:protocol.validation_end]
    restricted_data = replace(data, counts=restricted_counts)
    output.mkdir(parents=True, exist_ok=False)
    (output / "study_config.json").write_bytes(inputs["config_bytes"])
    (output / "common_protocol_config.json").write_bytes(inputs["protocol_bytes"])
    (output / "source_manifest.json").write_bytes(inputs["manifest_bytes"])
    started = time.monotonic()
    result = {
        "stage": inputs["study"].get("stage", "independent_graph_gp_fixed_topology_training_validation_pilot"),
        "status": "running",
        "study_config": str(inputs["config_path"]), "study_config_sha256": expected_config_sha256,
        "protocol_config": str(inputs["protocol_path"]),
        "protocol_config_sha256": inputs["study"]["protocol_config_sha256"],
        "source_archive": str(inputs["archive_path"]), "source_archive_sha256": ARCHIVE_SHA256,
        "source_manifest": str(inputs["manifest_path"]),
        "source_manifest_sha256": inputs["study"]["source_manifest_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": _implementation_hashes(),
        "evaluation_split": "validation_only", "final_test_evaluated": False,
        "prediction_clipping": "none", "county_order": list(data.counties),
        "history_weeks": protocol.history, "forecast_horizon_weeks": 1,
        "validation_target_indices": targets.tolist(),
        "validation_target_dates": [data.dates[int(t)] for t in targets],
        "gp_fit_raw_week_range": [0, protocol.train_end],
        "gp_fit_raw_weeks": protocol.train_end,
        "gp_fit_county_observations": protocol.train_end * len(data.counties),
        "warmup_weeks_included_in_gp_likelihood": protocol.history,
        "supervised_training_forecast_targets": protocol.train_end - protocol.history,
        "training_data_difference": (
            "GP marginal likelihood uses all 365 raw training observations per county, "
            "including 52 warmup weeks. Supervised forecasters have 313 training forecast targets. "
            "Both share the same available raw training observations."
        ),
        "optimizer_policy": dict(FROZEN_OPTIMIZER), "predeclared_variants": list(FROZEN_VARIANTS),
        "variant_selection": "none: report every predeclared variant",
        "environment": {
            "numpy": np.__version__, "scipy": scipy.__version__, "cuda_visible_devices": "",
            "python": os.sys.version, "numerical_thread_limit": 1,
        },
        "interpretation": (
            "Independent stationary spectral graph-OU GP comparator inspired by Nikitin et al., "
            "not an exact SHEK reproduction. Fixed-topology validation pilot only. "
            "No final-test, changing-topology, biological or state-of-the-art claim."
        ),
        "models": {},
    }
    _dump_json(output / "progress.json", result)
    try:
        with threadpool_limits(limits=1):
            result["environment"]["threadpools_during_run"] = threadpool_info()
            for variant in FROZEN_VARIANTS:
                require_storage(headroom_bytes=20_000_000)
                model_started = time.monotonic()
                fitted = GraphOUForecaster.fit_training(
                    restricted_data, protocol, scaler, variant=variant, **FROZEN_OPTIMIZER)
                fit_record = fitted.metadata()  # Preserve all attempts/status/bounds verbatim.
                fit_seconds = time.monotonic() - model_started
                _dump_json(output / f"graph_gp_{variant}_fit.json", fit_record)
                forecast_started = time.monotonic()
                prediction = np.stack([
                    fitted.predict(restricted_data.counts[int(t) - protocol.history:int(t)])
                    for t in targets
                ])
                forecast_seconds = time.monotonic() - forecast_started
                truth = restricted_data.counts[targets]
                metrics = regression_metrics(prediction, truth)
                prediction_path = output / f"graph_gp_{variant}_validation_predictions.npz"
                np.savez_compressed(
                    prediction_path, prediction=prediction, target_indices=targets,
                    target_dates=np.asarray([data.dates[int(t)] for t in targets]),
                    counties=np.asarray(data.counties))
                error = prediction - truth
                weekly = [
                    {"target_index": int(t), "target_date": data.dates[int(t)],
                     "history_range": [int(t) - protocol.history, int(t)],
                     "mae_cases": float(np.abs(error[i]).mean()),
                     "rmse_cases": float(np.sqrt(np.square(error[i]).mean()))}
                    for i, t in enumerate(targets)
                ]
                model_result = {
                    "model": f"graph_gp_{variant}", "variant": variant,
                    "fit": fit_record, "fit_seconds_runner": fit_seconds,
                    "forecast_seconds": forecast_seconds,
                    "total_seconds": time.monotonic() - model_started,
                    "validation_predictions": str(prediction_path),
                    "validation_predictions_sha256": _sha256(prediction_path.read_bytes()),
                    "weekly_validation_metrics": weekly, **metrics,
                }
                result["models"][f"graph_gp_{variant}"] = model_result
                _dump_json(output / f"graph_gp_{variant}.json", model_result)
                _dump_json(output / "progress.json", result)
                print(json.dumps({"event": "variant_complete", "variant": variant,
                                  "fit_seconds": fit_seconds, "forecast_seconds": forecast_seconds,
                                  **metrics}, allow_nan=False), flush=True)
        result["status"] = "complete"
        result["total_seconds"] = time.monotonic() - started
        _dump_json(output / "results.json", result)
        _dump_json(output / "progress.json", result)
        print(json.dumps({"event": "run_complete", "output": str(output)}, allow_nan=False), flush=True)
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        result["total_seconds"] = time.monotonic() - started
        _dump_json(output / "progress.json", result)
        raise
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen GP study JSON with source/protocol pins")
    parser.add_argument("--expected-config-sha256", required=True, help="SHA256 of frozen GP study JSON")
    parser.add_argument("--output", required=True, help="Fresh output directory inside the repository")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    run_study(args.config, args.output, args.expected_config_sha256)


if __name__ == "__main__":
    main()


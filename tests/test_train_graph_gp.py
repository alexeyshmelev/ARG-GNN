"""Pinned-input and causal-window runner tests, without real GP jobs or scoring.

Fits and predictions are synthetic stubs. Public files are read only to
exercise their frozen input hashes and metadata, never to compute accuracy.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from arggnn.chickenpox import ARCHIVE_NAME, ARCHIVE_SHA256, ChickenpoxData
from arggnn.resources import ROOT
import arggnn.train_graph_gp as runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return sha(path)


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    archive = ROOT / "data/raw/chickenpox" / ARCHIVE_NAME
    manifest_path = ROOT / "data/raw/chickenpox/source_manifest.json"
    common = ROOT / "configs/chickenpox_common_input_pilot_v1.json"
    if not all(p.exists() for p in [archive, manifest_path, common]):
        pytest.skip("Prepare the pinned benchmark source files first")
    manifest = json.loads(manifest_path.read_text())
    n = len(manifest["counties"])
    counts = 1. + 2. * np.arange(522)[:, None] + .1 * np.arange(n)[None, :]
    counts[443:] = np.nan  # Any final-test numeric access must fail finite checks.
    data = ChickenpoxData(
        counts, tuple(f"synthetic-week-{t}" for t in range(522)),
        tuple(manifest["counties"]), np.stack((np.arange(n), np.arange(n))), ARCHIVE_SHA256)
    monkeypatch.setattr(runner, "load_chickenpox_archive", lambda path: data)
    monkeypatch.setattr(runner, "set_research_title", lambda: None)
    monkeypatch.setattr(runner, "require_storage", lambda **kwargs: 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    study = {
        "protocol_config": str(common), "protocol_config_sha256": sha(common),
        "source_archive": str(archive), "source_archive_sha256": sha(archive),
        "source_manifest": str(manifest_path), "source_manifest_sha256": sha(manifest_path),
        "variants": ["nonseparable", "separable", "temporal"],
        "seed": 101, "restarts": 3, "maxiter": 100,
    }
    config = tmp_path / "study.json"
    pin = write_json(config, study)
    return config, pin, study, data


def install_stub(monkeypatch):
    calls, windows = [], []

    class SyntheticFit:
        def __init__(self, variant):
            self.variant = variant

        def predict(self, raw_history):
            assert raw_history.shape == (52, 20)
            assert np.isfinite(raw_history).all()
            windows.append((self.variant, raw_history.copy()))
            return raw_history[-1].copy()

        def metadata(self):
            return {
                "variant": self.variant, "hyperparameter_fit_range": [0, 365],
                "scaler_fit_range": [0, 365], "training_nll_standardized": 123.0,
                "selected_restart": 2, "selected_candidate": "optimized",
                "parameters_at_bound": ["diffusion"] if self.variant == "nonseparable" else [],
                "attempts": [
                    {"restart": 0, "success": False, "status": 1, "message": "synthetic budget limit"},
                    {"restart": 1, "success": False, "status": 2, "message": "synthetic optimizer failure"},
                    {"restart": 2, "success": True, "status": 0, "message": "synthetic convergence"},
                ],
                "final_test_evaluated": False,
            }

    class SyntheticForecaster:
        @classmethod
        def fit_training(cls, data, protocol, scaler, variant, **kwargs):
            assert kwargs == {"seed": 101, "restarts": 3, "maxiter": 100}
            assert np.isnan(data.counts[443:]).all()
            assert np.isfinite(data.counts[:443]).all()
            np.testing.assert_array_equal(scaler.mean, data.counts[:365].mean(axis=0))
            assert scaler.fit_end == 365 and protocol.history == 52
            calls.append(variant)
            return SyntheticFit(variant)

    monkeypatch.setattr(runner, "GraphOUForecaster", SyntheticForecaster)
    return calls, windows


def test_runner_preserves_pins_fit_diagnostics_and_exact_validation_windows(
        frozen_inputs, tmp_path, monkeypatch):
    config, pin, study, data = frozen_inputs
    study["stage"] = "graph_gp_common_input_train_validation_pilot_v1"
    pin = write_json(config, study)
    calls, windows = install_stub(monkeypatch)
    output = tmp_path / "fresh-output"
    result = runner.run_study(config, output, pin)
    assert calls == ["nonseparable", "separable", "temporal"]
    assert len(windows) == 3 * 78
    for variant in runner.FROZEN_VARIANTS:
        variant_windows = [w for name, w in windows if name == variant]
        for target, window in zip(range(365, 443), variant_windows, strict=True):
            np.testing.assert_array_equal(window, data.counts[target - 52:target])
        record = result["models"][f"graph_gp_{variant}"]
        assert record["mae_cases"] == pytest.approx(2.)
        assert record["rmse_cases"] == pytest.approx(2.)
        np.testing.assert_allclose(record["per_county_mae"], np.full(20, 2.))
        np.testing.assert_allclose(record["per_county_rmse"], np.full(20, 2.))
        assert len(record["weekly_validation_metrics"]) == 78
        assert record["weekly_validation_metrics"][0]["history_range"] == [313, 365]
        assert record["weekly_validation_metrics"][-1]["history_range"] == [390, 442]
        saved = Path(record["validation_predictions"])
        assert sha(saved) == record["validation_predictions_sha256"]
        with np.load(saved, allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact["target_indices"], np.arange(365, 443))
            assert artifact["prediction"].shape == (78, 20)
            np.testing.assert_array_equal(artifact["counties"], data.counties)
            assert set(artifact.files) == {"prediction", "target_indices", "target_dates", "counties"}
        fit_json = json.loads((output / f"graph_gp_{variant}_fit.json").read_text())
        assert fit_json == record["fit"]
        assert [a["success"] for a in fit_json["attempts"]] == [False, False, True]
    assert result["gp_fit_raw_weeks"] == 365
    assert result["gp_fit_county_observations"] == 7300
    assert result["warmup_weeks_included_in_gp_likelihood"] == 52
    assert result["supervised_training_forecast_targets"] == 313
    assert result["final_test_evaluated"] is False
    assert result["status"] == "complete"
    assert result["stage"] == study["stage"]
    assert result["study_config_sha256"] == pin
    assert result["protocol_config_sha256"] == study["protocol_config_sha256"]
    assert result["source_manifest_sha256"] == study["source_manifest_sha256"]
    assert (output / "study_config.json").read_bytes() == config.read_bytes()
    assert json.loads((output / "results.json").read_text()) == result
    assert json.loads((output / "progress.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("field", [
    "protocol_config_sha256", "source_manifest_sha256", "source_archive_sha256",
])
def test_source_or_protocol_pin_mismatch_prevents_all_fitting(
        frozen_inputs, tmp_path, monkeypatch, field):
    config, _, study, _ = frozen_inputs
    study[field] = "0" * 64
    pin = write_json(config, study)
    calls, _ = install_stub(monkeypatch)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="SHA256|archive pin"):
        runner.run_study(config, output, pin)
    assert calls == [] and not output.exists()


def test_study_config_pin_mismatch_prevents_output_creation(frozen_inputs, tmp_path):
    config, _, _, _ = frozen_inputs
    with pytest.raises(ValueError, match="GP study config SHA256"):
        runner.run_study(config, tmp_path / "rejected", "0" * 64)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("mutation", [
    {"variants": ["nonseparable"]}, {"variants": ["temporal", "separable", "nonseparable"]},
    {"seed": 102}, {"seed": True}, {"restarts": 1}, {"maxiter": 10},
])
def test_frozen_variant_and_optimizer_policy_cannot_be_changed(
        frozen_inputs, tmp_path, mutation):
    config, _, study, _ = frozen_inputs
    study.update(mutation)
    pin = write_json(config, study)
    with pytest.raises(ValueError, match="variant|optimizer"):
        runner.run_study(config, tmp_path / "rejected", pin)
    assert not (tmp_path / "rejected").exists()


def test_existing_output_directory_is_not_overwritten(frozen_inputs, tmp_path):
    config, pin, _, _ = frozen_inputs
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("untouched")
    with pytest.raises(RuntimeError, match="fresh"):
        runner.run_study(config, output, pin)
    assert sentinel.read_text() == "untouched"


def test_nonempty_or_unset_cuda_mask_is_rejected_before_outputs(
        frozen_inputs, tmp_path, monkeypatch):
    config, pin, _, _ = frozen_inputs
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="empty CUDA"):
        runner.run_study(config, tmp_path / "rejected", pin)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(RuntimeError, match="empty CUDA"):
        runner.run_study(config, tmp_path / "rejected", pin)


def test_input_and_output_paths_cannot_escape_repository(frozen_inputs):
    config, pin, _, _ = frozen_inputs
    with pytest.raises(ValueError, match="outside project"):
        runner.run_study(config, "/tmp/forbidden-gp-output", pin)
    with pytest.raises(ValueError, match="outside project"):
        runner._read_pinned("/tmp/forbidden-gp-config", pin, "study")


def test_manifest_county_disagreement_is_rejected(frozen_inputs, tmp_path, monkeypatch):
    config, pin, _, data = frozen_inputs
    monkeypatch.setattr(runner, "load_chickenpox_archive", lambda path: replace(
        data, counties=tuple(reversed(data.counties))))
    with pytest.raises(ValueError, match="county order"):
        runner.run_study(config, tmp_path / "rejected", pin)


def test_failed_fit_is_recorded_and_not_reported_complete(frozen_inputs, tmp_path, monkeypatch):
    config, pin, _, _ = frozen_inputs

    class FailedForecaster:
        @classmethod
        def fit_training(cls, *args, **kwargs):
            raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(runner, "GraphOUForecaster", FailedForecaster)
    output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        runner.run_study(config, output, pin)
    record = json.loads((output / "progress.json").read_text())
    assert record["status"] == "failed"
    assert record["failure"]["message"] == "synthetic fit failure"
    assert record["models"] == {}
    assert record["final_test_evaluated"] is False
    assert not (output / "results.json").exists()


def test_cli_has_no_test_device_model_or_optimizer_override():
    parser = runner.make_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {"-h", "--help", "--config", "--expected-config-sha256", "--output"}
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "frozen.json", "--output", "fresh"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "frozen.json", "--expected-config-sha256", "0" * 64,
                          "--output", "fresh", "--test"])


def test_regression_metrics_rejects_nonfinite_or_wrong_shapes():
    for prediction, truth in [(np.zeros((2, 2)), np.zeros((3, 2))),
                               (np.zeros((0, 2)), np.zeros((0, 2))),
                               (np.full((2, 2), np.nan), np.zeros((2, 2)))]:
        with pytest.raises(ValueError, match="finite matching"):
            runner.regression_metrics(prediction, truth)


@pytest.mark.parametrize("mutation", [
    {"history": 53}, {"train_end": 364}, {"validation_end": 442},
    {"observations": 521}, {"final_test_evaluated": True},
])
def test_modified_common_protocol_is_rejected_before_fit(
        frozen_inputs, tmp_path, monkeypatch, mutation):
    config, _, study, _ = frozen_inputs
    common = json.loads(Path(study["protocol_config"]).read_text())
    common.update(mutation)
    changed_protocol = tmp_path / "changed-common.json"
    study["protocol_config_sha256"] = write_json(changed_protocol, common)
    study["protocol_config"] = str(changed_protocol)
    pin = write_json(config, study)
    calls, _ = install_stub(monkeypatch)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Common protocol"):
        runner.run_study(config, output, pin)
    assert calls == [] and not output.exists()


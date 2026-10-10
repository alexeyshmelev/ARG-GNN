"""Frozen boundary runner checks on synthetic counts, never scoring the real test."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from arggnn.boundary_chickenpox import bind_chickenpox_history, precompute_chickenpox_geometry
from arggnn.chickenpox import ARCHIVE_SHA256, ChickenpoxData, ForecastProtocol, TrainingScaler
from arggnn.resources import ROOT
import arggnn.train_boundary_chickenpox as runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic(counties=20, observations=522):
    counts = 1. + 2.*np.arange(observations)[:, None] + .1*np.arange(counties)[None]
    edges = [(i, i) for i in range(counties)]
    edges += [(i, i+1) for i in range(counties-1)]
    edges += [(i+1, i) for i in range(counties-1)]
    return ChickenpoxData(counts, tuple(f"synthetic-week-{t}" for t in range(observations)),
        tuple(f"county{i}" for i in range(counties)), np.asarray(edges, dtype=np.int64).T, ARCHIVE_SHA256)


@pytest.fixture(autouse=True)
def one_cpu_thread(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runner, "require_storage", lambda **kwargs: 0)


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    original = ROOT/"configs/chickenpox_boundary_pilot_v1.json"
    if not original.exists():
        pytest.skip("Frozen boundary Chickenpox protocol is not prepared")
    settings = json.loads(original.read_text())
    # Preserve real repository path checks, but put this test's generated
    # checkpoints under its project-local temporary directory.
    native_project_path = runner.project_path
    def scoped_path(value):
        value = Path(value)
        if not value.is_absolute() and value.parts and value.parts[0] == "checkpoints":
            value = tmp_path / value
        return native_project_path(value)
    monkeypatch.setattr(runner, "project_path", scoped_path)
    manifest = json.loads((ROOT/settings["source_manifest"]).read_text())
    data = replace(synthetic(), counties=tuple(manifest["counties"]))
    monkeypatch.setattr(runner, "load_chickenpox_archive", lambda path: data)
    config = tmp_path/"config.json"
    config.write_text(json.dumps(settings, indent=2)+"\n")
    return config, sha(config), settings, data


def update_config(config, settings):
    config.write_text(json.dumps(settings, indent=2)+"\n")
    return sha(config)


def test_masked_reserved_rows_are_unavailable_and_scaler_uses_training_only():
    protocol = ForecastProtocol()
    original = synthetic()
    changed = original.counts.copy()
    changed[443:] = 1e100
    first = runner.restrict_validation_data(original, protocol)
    second = runner.restrict_validation_data(replace(original, counts=changed), protocol)
    np.testing.assert_array_equal(first.counts, second.counts)
    assert np.isnan(first.counts[443:]).all()
    assert not first.counts.flags.writeable
    assert np.isfinite(original.counts).all()  # source remains unchanged
    scaler = TrainingScaler.fit(first.counts[:365], 365)
    np.testing.assert_array_equal(scaler.mean, original.counts[:365].mean(0))
    validation_poisoned = first.counts.copy()
    validation_poisoned[365:443] = -1e100
    other = TrainingScaler.fit(validation_poisoned[:365], 365)
    np.testing.assert_array_equal(scaler.mean, other.mean)
    np.testing.assert_array_equal(scaler.scale, other.scale)


@pytest.mark.parametrize("name", runner.FROZEN_MODELS)
def test_each_bound_model_excludes_next_week_and_future_observations(name):
    data = synthetic(counties=4)
    protocol = ForecastProtocol()
    restricted = runner.restrict_validation_data(data, protocol)
    scaler = TrainingScaler.fit(restricted.counts[:365], 365)
    geometry = precompute_chickenpox_geometry(restricted,
        no_geography=name.endswith("_no_geography")).to("cpu", torch.float32)
    expected = bind_chickenpox_history(geometry, restricted, scaler, 365)
    poisoned = restricted.counts.copy()
    poisoned[365:] = np.nan
    actual = bind_chickenpox_history(geometry, replace(restricted, counts=poisoned), scaler, 365)
    torch.testing.assert_close(expected.leaf_x, actual.leaf_x, rtol=0, atol=0)
    np.testing.assert_array_equal(actual.leaf_x[..., 0].numpy(),
        scaler.transform(data.counts[313:365]).astype(np.float32))
    assert actual.target_graph.tolist() == [51]
    assert actual.positions.item() == actual.intervals[-1].mean().item()
    assert actual.query_cache.reference_ids.numel() == 0
    settings = dict(runner.FROZEN_POLICY, hidden_dim=8, num_layers=1, latent_dim=3, temporal_layers=1)
    model = runner.make_forecaster(name, settings).eval()
    with torch.no_grad():
        expected_output, actual_output = model(expected), model(actual)
    assert expected_output.shape == (1, 4, 1)
    torch.testing.assert_close(expected_output, actual_output, rtol=0, atol=0)


@pytest.mark.parametrize("name", [
    "boundary_full", "boundary_diagonal", "boundary_gain_matched", "boundary_gain_leak",
    "boundary_kernel", "boundary_gru", "boundary_transformer"])
def test_one_sided_readout_has_zero_dependency_on_later_historical_holds(name):
    data = synthetic(counties=4)
    scaler = TrainingScaler.fit(data.counts[:365], 365)
    geometry = precompute_chickenpox_geometry(data).to("cpu", torch.float32)
    bank = bind_chickenpox_history(geometry, data, scaler, 365)
    # A strict earlier anchor tests more than the actual last-week-only protocol.
    early = replace(bank, positions=bank.intervals[25].mean().reshape(1),
                    target_graph=torch.tensor([25]))
    features = early.leaf_x.clone()
    features[26:] = 1e4
    poisoned = replace(early, leaf_x=features)
    settings = dict(runner.FROZEN_POLICY, hidden_dim=8, num_layers=1, latent_dim=3, temporal_layers=1)
    model = runner.make_forecaster(name, settings).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(early), model(poisoned), rtol=1e-6, atol=1e-7)
    differentiable = replace(early, leaf_x=early.leaf_x.clone().requires_grad_())
    model(differentiable).sum().backward()
    assert torch.count_nonzero(differentiable.leaf_x.grad[26:]) == 0
    assert torch.isfinite(differentiable.leaf_x.grad[:26]).all()
    assert torch.count_nonzero(differentiable.leaf_x.grad[:26]) > 0


def test_kernel_ramp_sources_and_partial_hold_mass_have_correct_normalization():
    data = synthetic(counties=4)
    scaler = TrainingScaler.fit(data.counts[:365], 365)
    geometry = precompute_chickenpox_geometry(data).to("cpu", torch.float32)
    bank = bind_chickenpox_history(geometry, data, scaler, 365)
    model = runner.OneSidedBoundaryKernel(1, 1, hidden_dim=8, num_layers=1)
    signals = torch.arange(52, dtype=torch.float32)[:, None, None].expand(52, 4, 8)
    model.prepare = lambda b: (signals, signals.mean(1))
    # Reveal context channel only, then compare to an independent closed form
    # integral exp(-(x-s)/tau) from each observed weekly hold.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.weight[0, 8] = 1.
    actual = model(bank)[0, 0, 0].item()
    tau = float(torch.nn.functional.softplus(model.raw_tau)+1e-4)
    x = float(bank.positions[0])
    left, right = bank.intervals.numpy().T.astype(np.float64)
    stop = np.minimum(right, x)
    mass = np.where(left < x, tau*(np.exp(-(x-stop)/tau)-np.exp(-(x-left)/tau)), 0.)
    expected = np.sum(np.arange(52)*mass)/mass.sum()
    assert actual == pytest.approx(expected, rel=2e-6)
    assert mass[-1] > 0 and (stop[-1]-left[-1]) == pytest.approx((right[-1]-left[-1])/2, abs=np.finfo(np.float32).eps)


def test_ordered_modes_preserve_parameter_schema_and_no_geography_has_zero_response():
    settings = dict(runner.FROZEN_POLICY)
    names = ["boundary_full", "boundary_diagonal", "boundary_gain_matched", "boundary_gain_leak",
             "boundary_bag", "boundary_full_no_geography"]
    models = [runner.make_forecaster(n, settings) for n in names]
    schema = {key: tuple(value.shape) for key, value in models[0].state_dict().items()}
    assert all({key: tuple(value.shape) for key, value in model.state_dict().items()} == schema
               for model in models)
    assert all(not model.bidirectional for model in models)
    data = synthetic()
    geometry = precompute_chickenpox_geometry(data, no_geography=True)
    assert torch.count_nonzero(geometry.template.query_cache.cqq) == 0
    assert geometry.configuration_manifest()["no_geography"]


def test_runner_receipts_include_raw_week_county_predictions_and_exact_windows(
        frozen_inputs, tmp_path, monkeypatch):
    config, pin, settings, data = frozen_inputs
    fitting_calls, geometry_calls = [], []
    native_geometry = runner.precompute_chickenpox_geometry

    def checked_geometry(restricted, *args, **kwargs):
        assert np.isnan(restricted.counts[443:]).all()
        geometry_calls.append(kwargs["no_geography"])
        return native_geometry(restricted, *args, **kwargs)

    def synthetic_fit(model, banks, restricted, scaler, protocol, settings, checkpoint, config_sha):
        assert np.isnan(restricted.counts[443:]).all()
        assert set(banks) == set(range(52, 443))
        assert config_sha == pin
        assert scaler.fit_end == 365
        np.testing.assert_array_equal(scaler.mean, data.counts[:365].mean(0))
        for t, bank in banks.items():
            np.testing.assert_array_equal(bank.leaf_x[..., 0].cpu().numpy(),
                scaler.transform(data.counts[t-52:t]).astype(np.float32))
            assert bank.target_graph.tolist() == [51]
        fitting_calls.append(type(model).__name__)
        torch.save({"model": model.state_dict(), "study_config_sha256": config_sha}, checkpoint)
        prediction = np.stack([restricted.counts[t-1].copy() for t in range(365, 443)])
        return prediction, {"best_epoch": 3, "history": [{"epoch": 3, "mae_cases": 2.}],
                            "training_seconds": 0., **runner.regression_metrics(prediction, data.counts[365:443])}

    monkeypatch.setattr(runner, "precompute_chickenpox_geometry", checked_geometry)
    monkeypatch.setattr(runner, "_fit_model", synthetic_fit)
    requested = ["boundary_full", "boundary_full_no_geography"]
    output = tmp_path/"receipts"
    result = runner.run_study(config, output, pin, "cpu", requested)
    assert len(fitting_calls) == 2 and geometry_calls == [False, True]
    assert result["status"] == "complete"
    assert result["all_predeclared_models_completed_in_this_run"] is False
    assert result["final_test_evaluated"] is False
    assert result["reserved_numeric_rows_masked_from"] == 443
    assert result["study_config_sha256"] == pin
    assert result["source_manifest_sha256"] == settings["source_manifest_sha256"]
    assert result["protocol_config_sha256"] == settings["protocol_config_sha256"]
    assert result["implementation_sha256"]["train_boundary_chickenpox.py"] == sha(Path(runner.__file__))
    assert set(result["models"]) == set(requested)
    for name, record in result["models"].items():
        assert record["parameters"] > 0
        assert record["mae_cases"] == pytest.approx(2.)
        assert record["rmse_cases"] == pytest.approx(2.)
        np.testing.assert_allclose(record["per_county_mae"], np.full(20, 2.))
        assert len(record["weekly_validation_metrics"]) == 78
        assert record["weekly_validation_metrics"][0]["history_range"] == [313, 365]
        assert record["weekly_validation_metrics"][-1]["history_range"] == [390, 442]
        assert record["checkpoint_sha256"] == sha(record["checkpoint"])
        saved = Path(record["validation_predictions"])
        assert sha(saved) == record["validation_predictions_sha256"]
        with np.load(saved, allow_pickle=False) as artifact:
            assert artifact["prediction"].shape == artifact["truth"].shape == (78, 20)
            np.testing.assert_array_equal(artifact["target_indices"], np.arange(365, 443))
            np.testing.assert_array_equal(artifact["target_dates"], data.dates[365:443])
            np.testing.assert_array_equal(artifact["counties"], data.counties)
            assert artifact["study_config_sha256"].item() == pin
        assert record["model_configuration"]["temporal_direction"] == "one_sided"
    assert (output/"study_config.json").read_bytes() == config.read_bytes()
    assert json.loads((output/"results.json").read_text()) == result
    assert json.loads((output/"progress.json").read_text()) == result


def test_real_cpu_optimization_round_trip_on_small_synthetic_forecasting_problem(tmp_path):
    data = synthetic(counties=3, observations=58)
    protocol = ForecastProtocol(history=52, train_end=54, validation_end=56, observations=58)
    data = runner.restrict_validation_data(data, protocol)
    scaler = TrainingScaler.fit(data.counts[:54], 54)
    geometry = precompute_chickenpox_geometry(data).to("cpu", torch.float32)
    banks = {t: bind_chickenpox_history(geometry, data, scaler, t) for t in range(52, 56)}
    settings = dict(runner.FROZEN_POLICY, hidden_dim=8, num_layers=1, latent_dim=3, epochs=2)
    model = runner.make_forecaster("boundary_full", settings)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    checkpoint = tmp_path/"synthetic.pt"
    prediction, record = runner._fit_model(model, banks, data, scaler, protocol, settings, checkpoint, "a"*64)
    assert prediction.shape == (2, 3) and np.isfinite(prediction).all()
    assert any(not torch.equal(original[name], p) for name, p in model.named_parameters())
    chosen = torch.load(checkpoint, weights_only=True)
    assert chosen["epoch"] == record["best_epoch"]
    assert record["best_epoch"] == min(record["history"], key=lambda r: r["mae_cases"])["epoch"]
    assert chosen["validation_mae"] == pytest.approx(record["mae_cases"])
    assert chosen["study_config_sha256"] == "a"*64
    np.testing.assert_allclose(record["per_county_mae"], np.abs(prediction-data.counts[54:56]).mean(0))


def test_raw_unit_evaluation_preserves_negative_predictions():
    class NegativeModel(torch.nn.Module):
        def forward(self, bank):
            return torch.full((1, 2, 1), -100.)
    scaler = TrainingScaler(np.asarray([10., 20.]), np.asarray([2., 3.]), 365)
    prediction, metrics = runner.evaluate(NegativeModel(), {365: None}, np.asarray([365]),
                                           scaler, np.asarray([[1., 2.]]))
    np.testing.assert_array_equal(prediction, np.asarray([[-190., -280.]]))
    assert metrics["mae_cases"] == pytest.approx((191+282)/2)


@pytest.mark.parametrize("field", ["protocol_config_sha256", "source_archive_sha256", "source_manifest_sha256"])
def test_pin_mismatch_prevents_bank_binding_and_output(frozen_inputs, tmp_path, monkeypatch, field):
    config, _, settings, _ = frozen_inputs
    settings[field] = "0"*64
    pin = update_config(config, settings)
    def forbidden(*args, **kwargs):
        raise AssertionError("No bank after invalid pin")
    monkeypatch.setattr(runner, "precompute_chickenpox_geometry", forbidden)
    output = tmp_path/"invalid"
    with pytest.raises(ValueError, match="SHA256|archive pin"):
        runner.run_study(config, output, pin)
    assert not output.exists()


def test_config_hash_is_checked_before_outputs(frozen_inputs, tmp_path):
    config, _, _, _ = frozen_inputs
    with pytest.raises(ValueError, match="config SHA256"):
        runner.run_study(config, tmp_path/"invalid", "0"*64)


@pytest.mark.parametrize("mutation", [
    {"models": ["boundary_full"]}, {"epochs": 1}, {"patience": 2}, {"training_seed": True},
    {"selection": "maximum validation correlation"}, {"prediction_clipping": "nonnegative"},
    {"final_test_evaluated": True},
])
def test_frozen_study_training_policy_cannot_drift(frozen_inputs, tmp_path, mutation):
    config, _, settings, _ = frozen_inputs
    settings.update(mutation)
    pin = update_config(config, settings)
    with pytest.raises(ValueError, match="Frozen"):
        runner.run_study(config, tmp_path/"invalid", pin)
    assert not (tmp_path/"invalid").exists()


@pytest.mark.parametrize("mutation", [{"history": 53}, {"validation_end": 442}, {"epochs": 39},
                                      {"final_test_evaluated": True}])
def test_common_protocol_changes_rejected(frozen_inputs, tmp_path, mutation):
    config, _, settings, _ = frozen_inputs
    common = json.loads((ROOT/settings["protocol_config"]).read_text())
    common.update(mutation)
    altered = tmp_path/"common.json"
    altered.write_text(json.dumps(common)+"\n")
    settings["protocol_config"], settings["protocol_config_sha256"] = str(altered), sha(altered)
    pin = update_config(config, settings)
    with pytest.raises(ValueError, match="Common protocol"):
        runner.run_study(config, tmp_path/"invalid", pin)


def test_failure_records_never_claim_complete(frozen_inputs, tmp_path, monkeypatch):
    config, pin, _, _ = frozen_inputs
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic optimization failure")
    monkeypatch.setattr(runner, "_fit_model", failed)
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="synthetic optimization"):
        runner.run_study(config, output, pin, models=["boundary_full"])
    receipt = json.loads((output/"progress.json").read_text())
    assert receipt["status"] == "failed" and receipt["models"] == {}
    assert receipt["final_test_evaluated"] is False
    assert not (output/"results.json").exists()


def test_existing_output_and_paths_outside_repository_are_rejected(frozen_inputs, tmp_path):
    config, pin, _, _ = frozen_inputs
    output = tmp_path/"existing"
    output.mkdir()
    sentinel = output/"keep.txt"
    sentinel.write_text("untouched")
    with pytest.raises(RuntimeError, match="fresh"):
        runner.run_study(config, output, pin)
    assert sentinel.read_text() == "untouched"
    with pytest.raises(ValueError, match="outside project"):
        runner.run_study(config, "/tmp/outside-boundary-output", pin)


def test_worker_title_is_set_before_cuda_inspection(monkeypatch):
    events = []
    monkeypatch.setattr(runner, "set_research_title", lambda: events.append("title"))
    def forbidden_cuda():
        events.append("CUDA")
        raise AssertionError("Must not inspect CUDA before verified project lease")
    monkeypatch.setattr(torch.cuda, "device_count", forbidden_cuda)
    monkeypatch.delenv("ARGGNN_PHYSICAL_GPU", raising=False)
    with pytest.raises(RuntimeError, match="project GPU launcher"):
        runner.initialize_worker("cuda")
    assert events == ["title"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="empty CUDA"):
        runner.initialize_worker("cpu")
    assert events == ["title", "title"]


@pytest.mark.parametrize("models", [[], ["boundary_full", "boundary_full"], ["not_a_frozen_model"]])
def test_empty_duplicate_or_unknown_requested_models_rejected(frozen_inputs, tmp_path, models):
    config, pin, _, _ = frozen_inputs
    with pytest.raises(ValueError, match="distinct nonempty"):
        runner.run_study(config, tmp_path/"invalid", pin, models=models)


def test_cli_has_no_final_test_seed_or_optimizer_override():
    parser = runner.make_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {"-h", "--help", "--config", "--expected-config-sha256", "--output", "--device", "--models"}
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "frozen.json", "--output", "fresh"])


@pytest.mark.parametrize("prediction, truth", [(np.zeros((0, 2)), np.zeros((0, 2))),
    (np.zeros((2, 2)), np.zeros((3, 2))), (np.zeros((2, 0)), np.zeros((2, 0))),
    (np.full((2, 2), np.nan), np.zeros((2, 2)))])
def test_metrics_reject_invalid_predictions(prediction, truth):
    with pytest.raises(ValueError, match="finite matching"):
        runner.regression_metrics(prediction, truth)


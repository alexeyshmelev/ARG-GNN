"""Numerical oracle and leakage checks for the independent graph-OU GP.

All forecast accuracy tests use synthetic observations. The pinned public
archive is read only to verify county/edge ordering, without fitting or scoring.
"""
from dataclasses import replace
import csv
import io
import json
import zipfile

import numpy as np
import pytest
from scipy.linalg import expm, fractional_matrix_power

from arggnn.chickenpox import (
    ARCHIVE_NAME, REPOSITORY_ROOT, ChickenpoxData, ForecastProtocol,
    TrainingScaler, load_chickenpox_archive,
)
from arggnn.graph_gp import (
    GraphOUForecaster, GraphOUParameters, _names, _nll_gradient,
    graph_laplacian, marginal_negative_log_likelihood, modal_covariance,
    spectral_covariance,
)


def path_edges(n=3):
    edge = [(i, i) for i in range(n)]
    for i in range(n - 1):
        edge.extend([(i, i + 1), (i + 1, i)])
    return np.asarray(edge, dtype=np.int64).T


def synthetic_data():
    rng = np.random.default_rng(912)
    counts = rng.uniform(4, 20, (18, 3))
    return ChickenpoxData(counts, tuple(str(i) for i in range(18)),
                          ("A", "B", "C"), path_edges(), "synthetic")


def small_protocol():
    return ForecastProtocol(history=3, train_end=10, validation_end=14, observations=18)


def dense_covariance(times, laplacian, p, variant):
    """Independent node-space oracle using matrix powers and exponentials."""
    n = len(laplacian)
    identity = np.eye(n)
    if variant == "temporal":
        spatial = p.signal_variance * identity
    else:
        raw_spatial = fractional_matrix_power(p.spatial_shift * identity + laplacian,
                                              -p.spatial_power)
        spatial = p.signal_variance * raw_spatial / (np.trace(raw_spatial) / n)
    generator = p.decay * identity
    if variant == "nonseparable":
        generator = generator + p.diffusion * laplacian
    kernel = np.block([[expm(-generator * abs(t - u)) @ spatial
                        for u in times] for t in times])
    return kernel + p.noise_variance * np.eye(len(times) * n)


def fixed_forecaster(data, history, parameters, variant):
    laplacian = graph_laplacian(data.edge_index, len(data.counties))
    eigenvalues, basis = np.linalg.eigh(laplacian)
    eigenvalues = np.maximum(eigenvalues, 0)
    if variant == "temporal":
        eigenvalues, basis = np.zeros(len(data.counties)), np.eye(len(data.counties))
    scaler = TrainingScaler.fit(data.counts, 10)
    return GraphOUForecaster(parameters, variant, eigenvalues, basis, scaler,
                             history, data.counties, 10, {})


@pytest.mark.parametrize("variant", ["nonseparable", "separable", "temporal"])
def test_spectral_covariance_matches_node_space_oracle_and_is_psd(variant):
    laplacian = graph_laplacian(path_edges(), 3)
    p = GraphOUParameters(1.7, .13, .11, .27, .72, 1.3)
    times = np.array([0., 0., .8, 2.4, 4.])
    actual = spectral_covariance(times, laplacian, p, variant)
    expected = dense_covariance(times, laplacian, p, variant)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() >= p.noise_variance - 1e-10
    latent = spectral_covariance(times, laplacian, p, variant, observation_noise=False)
    assert np.linalg.eigvalsh(latent).min() >= -1e-10


@pytest.mark.parametrize("variant", ["nonseparable", "separable", "temporal"])
def test_filter_likelihood_matches_full_dense_gaussian(variant):
    rng = np.random.default_rng(718)
    values = rng.normal(size=(7, 3))
    laplacian = graph_laplacian(path_edges(), 3)
    p = GraphOUParameters(1.3, .08, .07, .14, .9, 1.1)
    covariance = dense_covariance(np.arange(len(values)), laplacian, p, variant)
    flattened = values.reshape(-1)
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    expected = .5 * (len(flattened) * np.log(2 * np.pi) + logdet
                     + flattened @ np.linalg.solve(covariance, flattened))
    actual = marginal_negative_log_likelihood(values, laplacian, p, variant)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("variant", ["nonseparable", "separable", "temporal"])
def test_analytic_logparameter_gradients_match_central_differences(variant):
    rng = np.random.default_rng(819)
    laplacian = graph_laplacian(path_edges(), 3)
    eig, basis = np.linalg.eigh(laplacian)
    eig = np.maximum(eig, 0)
    modes = rng.normal(size=(9, 3)) @ basis
    p = GraphOUParameters(1.3, .08, .07, .14, .9, 1.1)
    x = np.log([getattr(p, name) for name in _names(variant)])
    _, actual = _nll_gradient(x, modes, eig, variant)
    h = 1e-5
    expected = np.array([
        (_nll_gradient(x + h * direction, modes, eig, variant)[0]
         - _nll_gradient(x - h * direction, modes, eig, variant)[0]) / (2 * h)
        for direction in np.eye(len(x))
    ])
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize("variant", ["nonseparable", "separable", "temporal"])
def test_one_week_forecast_and_raw_uncertainty_match_dense_conditioning(variant):
    data = synthetic_data()
    p = GraphOUParameters(1.2, .12, .08, .19, .9, 1.4)
    history = 5
    fitted = fixed_forecaster(data, history, p, variant)
    raw = data.counts[8:8 + history]
    values = fitted.scaler.transform(raw).reshape(-1)
    covariance = dense_covariance(np.arange(history + 1),
                                   graph_laplacian(data.edge_index, 3), p, variant)
    past, cross, future = covariance[:-3, :-3], covariance[-3:, :-3], covariance[-3:, -3:]
    expected_mean = fitted.scaler.inverse(cross @ np.linalg.solve(past, values))
    expected_covariance = future - cross @ np.linalg.solve(past, cross.T)
    expected_covariance *= fitted.scaler.scale[:, None] * fitted.scaler.scale[None, :]
    actual_mean, actual_covariance = fitted.predict_distribution(raw)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual_covariance, expected_covariance, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(fitted.predict(raw), actual_mean)
    latent_mean, latent_covariance = fitted.predict_distribution(raw, observation_noise=False)
    np.testing.assert_array_equal(latent_mean, actual_mean)
    np.testing.assert_allclose(actual_covariance - latent_covariance,
                               p.noise_variance * np.diag(fitted.scaler.scale ** 2), atol=1e-11)


def test_training_fit_scaler_and_hyperparameters_ignore_all_future_observations():
    data = synthetic_data()
    protocol = small_protocol()
    changed = data.counts.copy()
    changed[protocol.train_end:] = np.nan
    first = GraphOUForecaster.fit_training(data, protocol, seed=71, restarts=2, maxiter=4)
    second = GraphOUForecaster.fit_training(replace(data, counts=changed), protocol,
                                            seed=71, restarts=2, maxiter=4)
    assert first.parameters == second.parameters
    np.testing.assert_array_equal(first.scaler.mean, second.scaler.mean)
    np.testing.assert_array_equal(first.scaler.scale, second.scaler.scale)
    assert first.fit_diagnostics["training_observations_sha256"] == second.fit_diagnostics[
        "training_observations_sha256"]
    assert first.fit_diagnostics["attempts"] == second.fit_diagnostics["attempts"]
    assert first.fit_diagnostics["training_nll_standardized"] <= min(
        a["initial_nll"] for a in first.fit_diagnostics["attempts"])
    metadata = first.metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata["hyperparameter_fit_range"] == metadata["scaler_fit_range"] == [0, 10]
    assert metadata["active_parameter_count"] == 6
    assert metadata["final_test_evaluated"] is False


def test_predictions_use_only_exact_window_and_reinitialize_every_call():
    data = synthetic_data()
    fitted = fixed_forecaster(data, 3, GraphOUParameters(), "nonseparable")
    target = 13
    expected = fitted.predict(data.counts[target - 3:target])
    changed = data.counts.copy()
    changed[:target - 3] = -1e9
    changed[target:] = 1e9
    np.testing.assert_array_equal(fitted.predict(changed[target - 3:target]), expected)
    fitted.predict(data.counts[0:3])
    np.testing.assert_array_equal(fitted.predict(data.counts[target - 3:target]), expected)
    for wrong in [data.counts[target - 4:target], data.counts[target - 2:target],
                  np.full((3, 3), np.nan), np.zeros((3, 2))]:
        with pytest.raises(ValueError, match="history"):
            fitted.predict(wrong)


def test_reject_scaler_fitted_on_future_even_when_fit_end_is_forged():
    data = synthetic_data()
    protocol = small_protocol()
    leaky = TrainingScaler.fit(data.counts, 18)
    with pytest.raises(ValueError, match="Scaler"):
        GraphOUForecaster.fit_training(data, protocol, replace(leaky, fit_end=10),
                                        restarts=1, maxiter=1)


def test_temporal_fit_is_graph_independent():
    data = synthetic_data()
    protocol = small_protocol()
    alternate = replace(data, edge_index=np.array([[0, 1, 2], [0, 1, 2]]))
    first = GraphOUForecaster.fit_training(data, protocol, variant="temporal", restarts=1, maxiter=5)
    second = GraphOUForecaster.fit_training(alternate, protocol, variant="temporal",
                                            restarts=1, maxiter=5)
    assert first.parameters == second.parameters
    np.testing.assert_array_equal(first.predict(data.counts[10:13]), second.predict(data.counts[10:13]))
    assert first.metadata()["active_parameter_count"] == 3


@pytest.mark.parametrize("variant", ["nonseparable", "separable", "temporal"])
def test_forecasting_is_equivariant_to_joint_county_and_graph_permutation(variant):
    data = synthetic_data()
    permutation = np.array([2, 0, 1])
    inverse = np.argsort(permutation)
    permuted = replace(data, counts=data.counts[:, permutation],
                       counties=tuple(data.counties[i] for i in permutation),
                       edge_index=inverse[data.edge_index])
    p = GraphOUParameters(1.3, .1, .2, .4, .8, 1.7)
    first = fixed_forecaster(data, 4, p, variant)
    second = fixed_forecaster(permuted, 4, p, variant)
    first_mean, first_covariance = first.predict_distribution(data.counts[9:13])
    second_mean, second_covariance = second.predict_distribution(permuted.counts[9:13])
    np.testing.assert_allclose(second_mean, first_mean[permutation], atol=1e-11)
    np.testing.assert_allclose(second_covariance,
                               first_covariance[np.ix_(permutation, permutation)], atol=1e-11)


def test_pinned_archive_uses_edge_id_county_order_and_selfloops_have_no_laplacian_effect():
    path = REPOSITORY_ROOT / "data/raw/chickenpox" / ARCHIVE_NAME
    if not path.exists():
        pytest.skip("Pinned archive is not present")
    data = load_chickenpox_archive(path)
    with zipfile.ZipFile(path) as archive:
        rows = list(csv.reader(io.StringIO(archive.read("hungary_chickenpox.csv").decode())))
    assert rows[0][1] == "BUDAPEST" and data.counties[0] == "BACS"
    assert data.counties[4] == "BUDAPEST"
    for county_id, name in enumerate(data.counties):
        np.testing.assert_array_equal(data.counts[:, county_id],
                                       [float(row[rows[0].index(name)]) for row in rows[1:]])
    nonloops = data.edge_index[:, data.edge_index[0] != data.edge_index[1]]
    full_laplacian = graph_laplacian(data.edge_index, 20)
    np.testing.assert_array_equal(full_laplacian, graph_laplacian(nonloops, 20))
    assert nonloops.shape[1] == 82  # 41 distinct undirected borders, 20 supplied self loops.
    assert np.linalg.eigvalsh(full_laplacian).min() > -1e-10


def test_spatial_variance_normalization_and_separability():
    eig = np.array([0., 1., 3.])
    p = GraphOUParameters(1.9, .1, .2, .4, .8, 1.7)
    q, rates = modal_covariance(eig, p, "nonseparable")
    np.testing.assert_allclose(q.mean(), p.signal_variance)
    assert np.all(np.diff(q) < 0) and np.all(np.diff(rates) > 0)
    sep_q, sep_rates = modal_covariance(eig, p, "separable")
    np.testing.assert_array_equal(sep_q, q)
    np.testing.assert_array_equal(sep_rates, np.full(3, p.decay))


@pytest.mark.parametrize("bad_edges", [np.array([[0], [1]]), np.array([[0], [3]]),
                                        np.array([[0.], [0.]])])
def test_reject_invalid_graph(bad_edges):
    with pytest.raises(ValueError):
        graph_laplacian(bad_edges, 3)


@pytest.mark.parametrize("bad_parameter", [
    GraphOUParameters(decay=0), GraphOUParameters(signal_variance=-1),
    GraphOUParameters(noise_variance=np.nan),
])
def test_reject_invalid_covariance_parameters(bad_parameter):
    with pytest.raises(ValueError):
        modal_covariance(np.arange(3.), bad_parameter)


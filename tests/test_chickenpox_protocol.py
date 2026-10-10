"""Leakage and county alignment checks. No real-data model scoring."""
from dataclasses import replace
import csv
import io
import zipfile

import numpy as np
import pytest
import torch

from arggnn.chickenpox import (
    ARCHIVE_NAME, ARCHIVE_SHA256, REPOSITORY_ROOT,
    ChickenpoxData, ForecastProtocol, LinearRidgeForecaster, TrainingScaler,
    download_chickenpox, forecast_sequence, history_permutation,
    last_value_forecast, load_chickenpox_archive, seasonal_forecast,
)


def synthetic_data(observations=18, counties=3):
    rng = np.random.default_rng(610)
    counts = rng.uniform(1, 25, (observations, counties))
    return ChickenpoxData(counts, tuple(str(i) for i in range(observations)),
                          tuple(str(i) for i in range(counties)),
                          np.vstack((np.arange(counties), np.arange(counties))),
                          "synthetic")


def test_frozen_disjoint_chronological_splits():
    protocol = ForecastProtocol()
    splits = [protocol.targets(s) for s in ("train", "validation", "test")]
    assert [len(x) for x in splits] == [313, 78, 79]
    np.testing.assert_array_equal(np.concatenate(splits), np.arange(52, 522))
    assert len(set(np.concatenate(splits))) == sum(map(len, splits))
    for targets in splits:
        for target in targets:
            history = np.arange(target - protocol.history, target)
            assert history.min() >= 0 and history.max() < target
    assert splits[1][0] == 365 and splits[2][0] == 443


@pytest.mark.parametrize("protocol", [ForecastProtocol(history=0),
    ForecastProtocol(history=365), ForecastProtocol(validation_end=522)])
def test_reject_invalid_boundaries(protocol):
    with pytest.raises(ValueError):
        protocol.targets("train")


def test_scaler_is_invariant_to_future_observations():
    data = synthetic_data()
    changed = data.counts.copy()
    changed[10:] = 1000000
    first = TrainingScaler.fit(data.counts, 10)
    second = TrainingScaler.fit(changed, 10)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_allclose(first.inverse(first.transform(data.counts)), data.counts)
    constant = TrainingScaler.fit(np.ones((5, 2)), 3)
    np.testing.assert_array_equal(constant.scale, np.ones(2))


def test_sequence_excludes_target_and_future_values():
    data = synthetic_data()
    scaler = TrainingScaler.fit(data.counts, 10)
    target, history = 13, 5
    expected = forecast_sequence(data, scaler, target, history)
    changed = data.counts.copy()
    changed[target:] = 1000000
    actual = forecast_sequence(replace(data, counts=changed), scaler, target, history)
    torch.testing.assert_close(actual.x, expected.x, rtol=0, atol=0)
    np.testing.assert_allclose(expected.x.reshape(history, 3).numpy(),
                               scaler.transform(data.counts[8:13]), rtol=1e-6, atol=1e-6)
    assert expected.x.shape == (15, 1)
    assert expected.positions.shape == (1,)
    assert expected.query_node_ids.tolist() == [0, 1, 2]
    assert expected.target_graph.item() == history - 1
    assert torch.count_nonzero(expected.clades) == 0
    assert torch.equal(expected.node_graph[expected.edge_index[0]],
                       expected.node_graph[expected.edge_index[1]])


def test_fixed_last_shuffle_changes_chronology_preserves_information():
    data = synthetic_data()
    scaler = TrainingScaler.fit(data.counts, 10)
    history = 8
    order = history_permutation(history, seed=701)
    assert order[-1] == history - 1
    assert not np.array_equal(order, np.arange(history))
    original = forecast_sequence(data, scaler, 15, history)
    shuffled = forecast_sequence(data, scaler, 15, history, order)
    original_values = original.x.reshape(history, 3)
    shuffled_values = shuffled.x.reshape(history, 3)
    torch.testing.assert_close(shuffled_values, original_values[order], rtol=0, atol=0)
    torch.testing.assert_close(shuffled_values[-1], original_values[-1], rtol=0, atol=0)
    torch.testing.assert_close(torch.sort(shuffled_values, dim=0).values,
                               torch.sort(original_values, dim=0).values, rtol=0, atol=0)
    torch.testing.assert_close(shuffled.intervals, original.intervals, rtol=0, atol=0)
    torch.testing.assert_close(shuffled.positions, original.positions, rtol=0, atol=0)


@pytest.mark.parametrize("order", [np.array([0, 0, 2]), np.array([0., 1., 2.]), np.array([0, 1])])
def test_reject_nonpermutations(order):
    data = synthetic_data()
    with pytest.raises(ValueError, match="permutation"):
        forecast_sequence(data, TrainingScaler.fit(data.counts, 10), 13, 3, order)


def test_observed_only_naive_controls():
    history = np.arange(60 * 3).reshape(60, 3)
    np.testing.assert_array_equal(last_value_forecast(history), history[-1])
    np.testing.assert_array_equal(seasonal_forecast(history), history[-52])
    with pytest.raises(ValueError):
        seasonal_forecast(history[:20])


def test_linear_control_fit_cannot_read_validation_or_test_values():
    data = synthetic_data()
    protocol = ForecastProtocol(history=3, train_end=10, validation_end=14, observations=18)
    scaler = TrainingScaler.fit(data.counts, protocol.train_end)
    expected = LinearRidgeForecaster.fit_training(data, protocol, scaler)
    changed = data.counts.copy()
    changed[protocol.train_end:] = 1000000
    actual = LinearRidgeForecaster.fit_training(replace(data, counts=changed), protocol, scaler)
    np.testing.assert_array_equal(expected.coefficients, actual.coefficients)
    np.testing.assert_array_equal(expected.feature_mean, actual.feature_mean)
    np.testing.assert_array_equal(expected.target_mean, actual.target_mean)
    assert expected.fit_targets == tuple(range(3, 10))
    prediction = expected.predict(data.counts[7:10])
    assert prediction.shape == (3,) and np.isfinite(prediction).all()


def test_dataset_writes_reject_outside_repository():
    with pytest.raises(ValueError, match="inside the repository"):
        download_chickenpox(REPOSITORY_ROOT.parent / "outside-chickenpox")


def test_pinned_archive_dates_adjacency_and_name_based_alignment():
    path = REPOSITORY_ROOT / "data/raw/chickenpox" / ARCHIVE_NAME
    if not path.exists():
        pytest.skip("Run the preparation CLI once to acquire the pinned public archive")
    data = load_chickenpox_archive(path)
    assert data.archive_sha256 == ARCHIVE_SHA256
    assert data.counts.shape == (522, 20)
    assert data.dates[0] == "2005-01-03" and data.dates[-1] == "2014-12-29"
    assert data.counties[0] == "BACS" and data.counties[4] == "BUDAPEST"
    edges = set(map(tuple, data.edge_index.T.tolist()))
    assert len(edges) == 102
    assert all((v, u) in edges for u, v in edges)
    assert sum(u == v for u, v in edges) == 20
    with zipfile.ZipFile(path) as archive:
        rows = list(csv.reader(io.StringIO(archive.read("hungary_chickenpox.csv").decode())))
    assert rows[0][1] == "BUDAPEST"
    for county_id, name in enumerate(data.counties):
        column = rows[0].index(name)
        expected = np.asarray([float(row[column]) for row in rows[1:]])
        np.testing.assert_array_equal(data.counts[:, county_id], expected)
    protocol = ForecastProtocol()
    assert data.dates[protocol.targets("train")[0]] == "2006-01-02"
    assert data.dates[protocol.targets("train")[-1]] == "2011-12-26"
    assert data.dates[protocol.targets("validation")[0]] == "2012-01-02"
    assert data.dates[protocol.targets("validation")[-1]] == "2013-06-24"
    assert data.dates[protocol.targets("test")[0]] == "2013-07-01"
    assert data.dates[protocol.targets("test")[-1]] == "2014-12-29"

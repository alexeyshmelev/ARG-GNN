"""Observed-only forecast adaptation and fixed geographic cache verification."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from arggnn.boundary_chickenpox import (ChickenpoxBoundaryModel,
    bind_chickenpox_history, precompute_chickenpox_geometry)
from arggnn.boundary_memory import BoundaryMemoryModel
from arggnn.chickenpox import (ARCHIVE_NAME, REPOSITORY_ROOT, ChickenpoxData,
    ForecastProtocol, TrainingScaler, forecast_sequence, history_permutation,
    load_chickenpox_archive)
from arggnn.kron_interface import (precompute_graph_interfaces,
                                  with_query_reference_cache)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def synthetic_data(observations=160, counties=20):
    rng = np.random.default_rng(708)
    counts = rng.uniform(1, 70, (observations, counties))
    links = [(i, i) for i in range(counties)]
    links += [(i, i+1) for i in range(counties-1)]
    links += [(i+1, i) for i in range(counties-1)]
    return ChickenpoxData(counts, tuple(str(t) for t in range(observations)),
        tuple(f"county{i}" for i in range(counties)),
        np.asarray(links, dtype=np.int64).T, "synthetic-source")


def prepared(no_geography=False):
    data = synthetic_data()
    scaler = TrainingScaler.fit(data.counts, 100)
    geometry = precompute_chickenpox_geometry(data, no_geography=no_geography)
    return data, scaler, geometry


@pytest.mark.parametrize("coupling", ["full", "diagonal", "gain_matched_diagonal", "gain_leak_diagonal"])
@pytest.mark.parametrize("aggregation", ["ordered", "homogeneous_bag", "equilibrium"])
def test_one_sided_scalar_shape_and_finite_backward(coupling, aggregation):
    data, scaler, geometry = prepared()
    bank = bind_chickenpox_history(geometry, data, scaler, 107)
    net = ChickenpoxBoundaryModel(hidden_dim=8, num_layers=1, latent_dim=3,
        coupling=coupling, aggregation=aggregation).double()
    output = net(bank)
    assert output.shape == (1, 20, 1) and torch.isfinite(output).all()
    assert net.num_outputs == net.head.out_features == 1
    assert not net.bidirectional
    assert net.encode_context(bank).backward is None
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    # R is empty, so the retained common reference projection is inactive.
    assert torch.count_nonzero(net.reference_projection.weight.grad) == 0
    assert torch.count_nonzero(net.reference_projection.bias.grad) == 0


def test_adapter_uses_exact_same_model_parameter_schema_and_computation():
    data, scaler, geometry = prepared()
    bank = bind_chickenpox_history(geometry, data, scaler, 107)
    for coupling in ("full", "diagonal", "gain_matched_diagonal", "gain_leak_diagonal"):
        adapter = ChickenpoxBoundaryModel(hidden_dim=8, num_layers=2, latent_dim=3,
            coupling=coupling).double()
        generic = BoundaryMemoryModel(1, 1, hidden_dim=8, num_layers=2, latent_dim=3,
            coupling=coupling, bidirectional=False, symmetric_readout=False).double()
        generic.load_state_dict(adapter.state_dict(), strict=True)
        torch.testing.assert_close(adapter(bank), generic(bank), rtol=0, atol=0)
        assert adapter.head.in_features == 8+3


def test_laplacian_and_common_input_match_generic_primitive():
    data, scaler, geometry = prepared()
    seq = forecast_sequence(data, scaler, 107)
    keep = seq.edge_index[0] != seq.edge_index[1]
    seq = replace(seq, edge_index=seq.edge_index[:, keep])
    reference = precompute_graph_interfaces(seq, torch.ones(seq.edge_index.shape[1]))
    reference = with_query_reference_cache(reference, torch.empty(0, dtype=torch.long))
    bank = bind_chickenpox_history(geometry, data, scaler, 107)
    torch.testing.assert_close(bank.response, reference.response, rtol=0, atol=0)
    torch.testing.assert_close(bank.query_cache.cqq, reference.query_cache.cqq, rtol=0, atol=0)
    torch.testing.assert_close(bank.leaf_x, reference.leaf_x, rtol=0, atol=0)
    torch.testing.assert_close(bank.intervals, seq.intervals, rtol=0, atol=0)
    torch.testing.assert_close(bank.positions, seq.positions, rtol=0, atol=0)
    torch.testing.assert_close(bank.target_graph, seq.target_graph, rtol=0, atol=0)
    assert bank.query_cache.cqr.shape == (52, 20, 0)
    assert bank.query_cache.reference_ids.numel() == 0
    assert bank.query_cache.query_ids.tolist() == list(range(20))
    assert bank.feature_columns == (0,) and bank.geometry_kind == "generic_graph_boundary"
    cache = bank.query_cache
    torch.testing.assert_close((cache.cqq*cache.interval_weights[:,None,None]).sum(0),
        cache.bag_cqq, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(cache.eigenvectors@torch.diag_embed(cache.eigenvalues)@
        cache.eigenvectors.transpose(-1,-2), cache.cqq, rtol=1e-12, atol=1e-12)


def test_bind_and_forward_never_recompute_fixed_geometry(monkeypatch):
    data, scaler, geometry = prepared()
    geometry = geometry.to("cpu", torch.float32)
    def forbidden(*args, **kwargs):
        raise AssertionError("Online geometry factorization is forbidden")
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(torch, "matrix_exp", forbidden)
    banks = [bind_chickenpox_history(geometry, data, scaler, t) for t in (52, 107, 108)]
    for bank in banks:
        assert bank.query_cache is geometry.template.query_cache
        assert bank.response is geometry.template.response
        assert bank.intervals is geometry.template.intervals
        assert bank.leaf_x.dtype == torch.float32
    assert not torch.equal(banks[0].leaf_x, banks[1].leaf_x)
    for coupling in ("full", "diagonal", "gain_matched_diagonal", "gain_leak_diagonal"):
        for aggregation in ("ordered", "homogeneous_bag", "equilibrium"):
            net = ChickenpoxBoundaryModel(hidden_dim=8, num_layers=1, latent_dim=3,
                coupling=coupling, aggregation=aggregation)
            assert torch.isfinite(net(banks[0])).all()


def test_binding_excludes_target_and_future_counts_with_frozen_scaler():
    data, scaler, geometry = prepared()
    target = 107
    expected = bind_chickenpox_history(geometry, data, scaler, target)
    counts = data.counts.copy()
    counts[target:] = 1e9
    actual = bind_chickenpox_history(geometry, replace(data, counts=counts), scaler, target)
    torch.testing.assert_close(actual.leaf_x, expected.leaf_x, rtol=0, atol=0)
    net = ChickenpoxBoundaryModel(hidden_dim=8, num_layers=1, latent_dim=3).double()
    torch.testing.assert_close(net(actual), net(expected), rtol=0, atol=0)
    expected_values = scaler.transform(data.counts[target-52:target]).astype(np.float32)
    np.testing.assert_array_equal(expected.leaf_x[...,0].numpy(), expected_values)


def test_fixed_last_shuffle_matches_existing_protocol_without_changing_geometry():
    data, scaler, geometry = prepared()
    order = history_permutation(52, seed=710, fix_last=True)
    original = bind_chickenpox_history(geometry, data, scaler, 107)
    shuffled = bind_chickenpox_history(geometry, data, scaler, 107, order)
    reference = forecast_sequence(data, scaler, 107, permutation=order)
    torch.testing.assert_close(shuffled.leaf_x, original.leaf_x[order], rtol=0, atol=0)
    torch.testing.assert_close(shuffled.leaf_x[-1], original.leaf_x[-1], rtol=0, atol=0)
    torch.testing.assert_close(shuffled.leaf_x.reshape(-1,1), reference.x.double(), rtol=0, atol=0)
    assert shuffled.query_cache is original.query_cache
    assert shuffled.intervals is original.intervals
    net = ChickenpoxBoundaryModel(hidden_dim=8, num_layers=1, latent_dim=3,
        aggregation="homogeneous_bag").double()
    # float32 linspace has slightly unequal widths, deliberately preserved
    # from the common protocol, so width-weighted bag is only nearly invariant.
    torch.testing.assert_close(net(shuffled), net(original), rtol=2e-6, atol=2e-7)


def test_no_geography_has_zero_response_and_all_controls_coincide():
    data, scaler, geometry = prepared(no_geography=True)
    bank = bind_chickenpox_history(geometry, data, scaler, 107)
    cache = bank.query_cache
    assert torch.count_nonzero(bank.response) == 0
    assert torch.count_nonzero(cache.cqq) == 0
    torch.testing.assert_close(cache.normalization_scale, torch.ones(52, dtype=torch.float64))
    nets = [ChickenpoxBoundaryModel(hidden_dim=8, num_layers=1, latent_dim=3,
        coupling=c).double() for c in ("full", "diagonal", "gain_matched_diagonal", "gain_leak_diagonal")]
    for net in nets[1:]:
        net.load_state_dict(nets[0].state_dict())
        torch.testing.assert_close(net(bank), nets[0](bank), rtol=0, atol=0)
    assert geometry.configuration_manifest()["no_geography"]


def test_empty_reference_is_disclosed_and_county_ids_are_not_learned_features():
    data, scaler, geometry = prepared()
    net = ChickenpoxBoundaryModel(hidden_dim=8, num_layers=1, latent_dim=3).double()
    manifest = net.configuration_manifest(bind_chickenpox_history(geometry, data, scaler, 107))
    assert manifest["num_outputs"] == 1 and manifest["bidirectional"] is False
    assert manifest["inactive_reference_parameters"] == manifest["inactive_reference_parameters_if_no_references"] == 27
    assert manifest["learned_numerical_id_embeddings"] is False
    assert manifest["reference_ids"] == []
    geometry_manifest = geometry.configuration_manifest()
    assert geometry_manifest["native_self_loop_records"] == 20
    assert geometry_manifest["native_undirected_neighbor_links"] == 19
    assert geometry_manifest["target_label_in_bank"] is False


@pytest.mark.parametrize("change", ["county", "edge", "source"])
def test_reject_mismatched_fixed_geometry(change):
    data, scaler, geometry = prepared()
    if change == "county":
        data = replace(data, counties=tuple(reversed(data.counties)))
    elif change == "edge":
        data = replace(data, edge_index=data.edge_index[:, :-2])
    else:
        data = replace(data, archive_sha256="other-source")
    with pytest.raises(ValueError, match="differs from prepared geometry"):
        bind_chickenpox_history(geometry, data, scaler, 107)


@pytest.mark.parametrize("order", [np.zeros(52, dtype=int), np.arange(52,dtype=float), np.arange(51)])
def test_reject_invalid_history_permutation(order):
    data, scaler, geometry = prepared()
    with pytest.raises(ValueError, match="permutation"):
        bind_chickenpox_history(geometry, data, scaler, 107, order)


def test_reject_incomplete_history_and_invalid_scaler():
    data, scaler, geometry = prepared()
    with pytest.raises(ValueError, match="complete observed history"):
        bind_chickenpox_history(geometry, data, scaler, 51)
    invalid = replace(scaler, scale=np.zeros_like(scaler.scale))
    with pytest.raises(ValueError, match="scaler"):
        bind_chickenpox_history(geometry, data, invalid, 107)


def test_pinned_actual_source_input_and_forward_without_scoring():
    path = REPOSITORY_ROOT/"data/raw/chickenpox"/ARCHIVE_NAME
    if not path.exists():
        pytest.skip("Pinned Chickenpox source is not prepared")
    data = load_chickenpox_archive(path)
    protocol = ForecastProtocol()
    scaler = TrainingScaler.fit(data.counts, protocol.train_end)
    geometry = precompute_chickenpox_geometry(data).to("cpu", torch.float32)
    bank = bind_chickenpox_history(geometry, data, scaler, int(protocol.targets("train")[0]))
    assert bank.leaf_x.shape == (52,20,1)
    assert bank.query_cache.cqr.shape == (52,20,0)
    metadata = geometry.configuration_manifest()
    assert metadata["native_adjacency_records"] == 102
    assert metadata["native_self_loop_records"] == 20
    assert metadata["native_undirected_neighbor_links"] == 41
    assert metadata["normalization_scale"] == pytest.approx(4.1, rel=1e-7)
    net = ChickenpoxBoundaryModel(hidden_dim=8, num_layers=1, latent_dim=3)
    output = net(bank)
    assert output.shape == (1,20,1) and torch.isfinite(output).all()
    # No labels, loss against actual targets, model fitting or raw-unit metrics.


def test_county_permutation_equivariance_including_train_scaler():
    data, scaler, geometry = prepared()
    order = np.random.default_rng(711).permutation(20)
    inverse = np.argsort(order)
    permuted = replace(data, counts=data.counts[:, order],
        counties=tuple(data.counties[i] for i in order),
        edge_index=inverse[data.edge_index])
    permuted_scaler = replace(scaler, mean=scaler.mean[order], scale=scaler.scale[order])
    permuted_geometry = precompute_chickenpox_geometry(permuted)
    original_bank = bind_chickenpox_history(geometry, data, scaler, 107)
    permuted_bank = bind_chickenpox_history(permuted_geometry, permuted, permuted_scaler, 107)
    net = ChickenpoxBoundaryModel(hidden_dim=8, num_layers=1, latent_dim=3).double()
    torch.testing.assert_close(net(permuted_bank), net(original_bank)[:,order], rtol=1e-11, atol=1e-11)

"""Split QQ/QR attribution, dense-action oracle and observed input contracts."""
from dataclasses import replace

import pytest
import torch

from arggnn.age_intervention_controls import SplitAgeAnchorKernelModel
from arggnn.anchor_kernel import AnchorKernelModel, AnchorOperators, prepare_anchor_kernel_bank
from arggnn.kron_interface import with_query_reference_cache
# Frozen test fixtures/oracle are reused read-only.
from test_anchor_kernel import cached, dense_action, prepared, sequence


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def model(intervention="source_only",coupling="full",depth=2,rate="log"):
    torch.manual_seed(220)
    return SplitAgeAnchorKernelModel(4,3,hidden_dim=8,num_layers=2,latent_dim=3,
        kernel_depth=depth,num_scales=3,geometry="age",coupling=coupling,
        num_anchors=5,initial_coupling=[.8,2.,5.],initial_damping=.6,
        rate_parameterization=rate,age_intervention=intervention).double()


@pytest.mark.parametrize("intervention",["source_only","propagation_only"])
def test_selection_aliases_both_maps_without_mutating_original_prepared_bank(intervention):
    pack,net = prepared(),model(intervention)
    plain,age = pack.operators["plain"],pack.operators["age"]
    wrapped = net.intervention_bank(pack)
    selected = wrapped.operators["age"]
    query = plain if intervention=="source_only" else age
    reference = age if intervention=="source_only" else plain
    assert selected.cqq is query.cqq
    assert selected.eigenvalues is query.eigenvalues
    assert selected.eigenvectors is query.eigenvectors
    assert selected.cqr is reference.cqr
    assert wrapped.bank is pack.bank and wrapped.layout is pack.layout
    assert wrapped.operators is not pack.operators
    assert pack.operators["age"] is age


@pytest.mark.parametrize("intervention",["source_only","propagation_only"])
@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal"])
@pytest.mark.parametrize("depth",[1,2])
def test_identical_age_plain_limiting_identity_counts_schemas_and_gradients(intervention,coupling,depth):
    pack = prepared()
    same = replace(pack,operators={**pack.operators,"age":pack.operators["plain"]})
    net = model(intervention,coupling,depth)
    baseline = AnchorKernelModel(4,3,hidden_dim=8,num_layers=2,latent_dim=3,kernel_depth=depth,
        num_scales=3,geometry="age",coupling=coupling,num_anchors=5,
        initial_coupling=[.8,2.,5.],initial_damping=.6).double()
    baseline.load_state_dict(net.state_dict(),strict=True)
    assert list(net.state_dict())==list(baseline.state_dict())
    assert sum(p.numel() for p in net.parameters())==sum(p.numel() for p in baseline.parameters())
    torch.testing.assert_close(net(same),baseline(same),rtol=0,atol=0)
    output = net(pack)
    assert output.shape==(7,3,3) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("intervention",["source_only","propagation_only"])
@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal"])
def test_chosen_reference_source_and_query_kernel_match_dense_oracle_each_layer(intervention,coupling):
    pack,net = prepared(),model(intervention,coupling)
    query_bank = pack.operators["plain"] if intervention=="source_only" else pack.operators["age"]
    reference_bank = pack.operators["age"] if intervention=="source_only" else pack.operators["plain"]
    oracle_operators = AnchorOperators(query_bank.cqq,reference_bank.cqr,query_bank.eigenvalues,query_bank.eigenvectors)
    h = net.encoder(pack.bank.leaf_x,pack.bank.query_cache.normalized_response)
    local_query = h[:,pack.bank.query_cache.query_indices]
    anchors,targets = local_query[pack.layout.anchor_graph],local_query[pack.bank.target_graph]
    reference = h[:,pack.bank.query_cache.reference_indices][pack.layout.anchor_graph]
    actual = net.encode_context(pack)
    for layer,message in zip(net.layers,actual.layers):
        qr = reference_bank.cqr[pack.layout.anchor_graph]
        source = layer.query_projection(anchors).tanh()[:,None]-(qr@layer.reference_projection(reference).tanh()[:,None])*layer.rates[None,:,None,:]
        mass = torch.ones_like(source)-qr.sum(-1)[...,None]*layer.rates[None,:,None,:]
        torch.testing.assert_close(message.sources,source,rtol=3e-11,atol=3e-11)
        torch.testing.assert_close(message.mass_sources,mass,rtol=0,atol=0)
        dense = dense_action(pack,oracle_operators,layer.rates,layer.damping,
            torch.stack((source,mass),dim=-1),coupling)
        mixed = (dense*layer.mixture[None,:,None,:,None]).sum(1)
        torch.testing.assert_close(message.numerator,mixed[...,0],rtol=3e-11,atol=3e-11)
        torch.testing.assert_close(message.mass,mixed[...,1],rtol=3e-11,atol=3e-11)
        normalized = mixed[...,0]/mixed[...,1]
        anchors = layer.update(anchors,normalized[pack.layout.anchor_output])
        targets = layer.update(targets,normalized[pack.layout.target_output])
    torch.testing.assert_close(actual.targets,targets,rtol=3e-11,atol=3e-11)
    torch.testing.assert_close(net(pack),net.head(targets),rtol=3e-11,atol=3e-11)


@pytest.mark.parametrize("intervention",["source_only","propagation_only"])
@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal"])
def test_mixed_operator_messages_remain_bounded_with_saturated_sources(intervention,coupling):
    pack,net = prepared(),model(intervention,coupling)
    with torch.no_grad():
        for layer in net.layers:
            layer.query_projection.weight.mul_(500.)
            layer.reference_projection.weight.mul_(500.)
            layer.raw_coupling.copy_(torch.tensor([[-5.,0.,5.],[2.,-2.,1.],[0.,3.,-3.]]))
            layer.mixture_logits.copy_(torch.tensor([[2.,-1.,0.],[-2.,3.,1.],[0.,-2.,2.]]))
    for message in net.encode_context(pack).layers:
        assert (message.mass>0).all()
        assert (message.anchors.abs()<=1+2e-12).all()
        assert (message.targets.abs()<=1+2e-12).all()


@pytest.mark.parametrize("intervention",["source_only","propagation_only"])
@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal"])
def test_unary_interval_and_graph_serialization_contracts(intervention,coupling):
    net = model(intervention,coupling)
    torch.testing.assert_close(net(prepared()),net(prepared(sequence(unary=True,split=True,serialize=True))),rtol=3e-11,atol=3e-11)


@pytest.mark.parametrize("intervention",["source_only","propagation_only"])
def test_readout_and_query_permutation_and_extra_readout_preserve_predictions(intervention):
    pack,net = prepared(),model(intervention)
    expected = net(pack)
    order = torch.tensor([6,2,1,5,3,0,4])
    reordered = replace(pack.bank,positions=pack.bank.positions[order],target_graph=pack.bank.target_graph[order])
    # Reuse exactly the supplied age/cache tensors while refreshing coordinates.
    coordinate_pack = prepare_anchor_kernel_bank(reordered,num_anchors=5,num_scales=3)
    coordinate_pack = replace(coordinate_pack,operators=pack.operators)
    torch.testing.assert_close(net(coordinate_pack),expected[order],rtol=3e-12,atol=3e-12)
    extra_bank = replace(pack.bank,positions=torch.cat((pack.bank.positions,torch.tensor([.123,.47,.71],dtype=torch.float64))),
        target_graph=torch.cat((pack.bank.target_graph,torch.tensor([0,1,2]))))
    extra = prepare_anchor_kernel_bank(extra_bank,num_anchors=5,num_scales=3)
    extra = replace(extra,operators=pack.operators)
    torch.testing.assert_close(net(extra)[:7],expected,rtol=3e-12,atol=3e-12)
    query_order = torch.tensor([2,0,1])
    query_bank = with_query_reference_cache(replace(pack.bank,query_node_ids=pack.bank.query_node_ids[query_order],query_cache=None),torch.tensor([40]))
    coordinate_pack = prepare_anchor_kernel_bank(query_bank,num_anchors=5,num_scales=3)
    def permuted(operators):
        return AnchorOperators(operators.cqq[:,:,query_order][:,:,:,query_order],operators.cqr[:,:,query_order],
            operators.eigenvalues,operators.eigenvectors[:,:,query_order])
    coordinate_pack = replace(coordinate_pack,operators={key:permuted(value) for key,value in pack.operators.items()})
    torch.testing.assert_close(net(coordinate_pack),expected[:,query_order],rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("intervention",["source_only","propagation_only"])
def test_observed_only_wrapper_and_no_online_geometry_or_solve(intervention,monkeypatch):
    pack,net = prepared(),model(intervention)
    expected = net(pack)
    class ForbiddenLabels:
        def __getattribute__(self,name):
            raise AssertionError("External labels cannot be accessed as features")
    object.__setattr__(pack.bank,"external_labels",ForbiddenLabels())
    def forbidden(*args,**kwargs):
        raise AssertionError("No online geometry computation or full solve")
    for name in ("eigh","eigvalsh","solve","inv"):
        monkeypatch.setattr(torch.linalg,name,forbidden)
    monkeypatch.setattr(torch,"matrix_exp",forbidden)
    torch.testing.assert_close(net(pack),expected,rtol=0,atol=0)


@pytest.mark.parametrize("intervention",["source_only","propagation_only"])
def test_log_softplus_initialization_and_intervention_manifest(intervention):
    pack = prepared()
    log,softplus = model(intervention),model(intervention,rate="softplus")
    torch.testing.assert_close(log(pack),softplus(pack),rtol=0,atol=0)
    manifest = log.configuration_manifest(pack)
    assert manifest["age_intervention"]==intervention
    assert manifest["physical_full_boundary_Kron_response"] is False
    assert manifest["reference_forcing_operator_source"]==manifest["reference_mass_operator_source"]
    assert manifest["current_rates"]==[layer.rates.detach().tolist() for layer in log.layers]


@pytest.mark.parametrize("kwargs",[{"geometry":"plain"},{"num_scales":1},{"coupling":"scalar"},{"age_intervention":"both"}])
def test_invalid_intervention_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        SplitAgeAnchorKernelModel(**kwargs)


def test_plain_only_pack_rejected():
    net = model()
    pack = prepare_anchor_kernel_bank(cached(),num_anchors=5,num_scales=3)
    with pytest.raises(ValueError,match="plain and age operators"):
        net(pack)

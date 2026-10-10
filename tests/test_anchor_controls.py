"""Frozen-schema source-feedback and static local capacity attribution checks."""
from dataclasses import replace

import pytest
from scipy.linalg import solve
import torch

from arggnn.anchor_controls import NoncontextualAnchorKernelModel, StaticLocalAnchorModel
from arggnn.anchor_kernel import AnchorKernelModel, prepare_anchor_kernel_bank
from arggnn.kron_interface import with_query_reference_cache
# Read-only reuse of frozen mathematical interface fixtures; no shared edits.
from test_anchor_kernel import cached, prepared, sequence


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def model(cls=NoncontextualAnchorKernelModel,depth=2,geometry="plain",coupling="full",rate="log",scales=3,anchors=5):
    torch.manual_seed(119)
    return cls(4,3,hidden_dim=8,num_layers=2,latent_dim=3,kernel_depth=depth,
        num_scales=scales,geometry=geometry,coupling=coupling,num_anchors=anchors,
        initial_coupling=[.8,2.,5.] if scales==3 else 2.,initial_damping=.6,
        rate_parameterization=rate).double()


@pytest.mark.parametrize("cls",[NoncontextualAnchorKernelModel,StaticLocalAnchorModel])
@pytest.mark.parametrize("depth",[1,2])
@pytest.mark.parametrize("geometry",["plain","age","matched_scalar_age"])
def test_control_counts_schemas_and_finite_gradients(cls,depth,geometry):
    pack = prepared()
    net,baseline = model(cls,depth,geometry),model(AnchorKernelModel,depth,geometry)
    assert list(net.state_dict())==list(baseline.state_dict())
    assert [(n,tuple(p.shape)) for n,p in net.named_parameters()]==[(n,tuple(p.shape)) for n,p in baseline.named_parameters()]
    assert sum(p.numel() for p in net.parameters())==sum(p.numel() for p in baseline.parameters())
    output = net(pack)
    assert output.shape==(7,3,3) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    # Retained global static rates and mixtures participate, rather than being
    # attached unused parameters, for this nondegenerate Q/R multibranch fixture.
    assert all(torch.count_nonzero(p.grad)>0 for p in net.rate_and_mixture_parameters())


@pytest.mark.parametrize("coupling",["full","scalar","diagonal","gain_matched_diagonal"])
def test_noncontextual_supports_all_frozen_kernel_controls(coupling):
    net = model(coupling=coupling)
    output = net(prepared())
    output.square().sum().backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_noncontextual_first_layer_equals_contextual_and_second_sources_ignore_feedback():
    pack = prepared()
    first,contextual = model(depth=1),model(AnchorKernelModel,depth=1)
    torch.testing.assert_close(first(pack),contextual(pack),rtol=0,atol=0)
    net,baseline = model(),model(AnchorKernelModel)
    before,base_before = net.encode_context(pack),baseline.encode_context(pack)
    with torch.no_grad():
        for current in (net,baseline):
            current.layers[0].message_projection.weight.mul_(20.)
            current.layers[0].gate.bias.add_(8.)
    after,base_after = net.encode_context(pack),baseline.encode_context(pack)
    torch.testing.assert_close(after.layers[1].sources,before.layers[1].sources,rtol=0,atol=0)
    torch.testing.assert_close(after.layers[1].mass_sources,before.layers[1].mass_sources,rtol=0,atol=0)
    assert (after.anchors-before.anchors).abs().max()>1e-4
    assert (after.targets-before.targets).abs().max()>1e-4
    assert (base_after.layers[1].sources-base_before.layers[1].sources).abs().max()>1e-4


def test_static_control_matches_dense_per_graph_solve_on_source_and_mass_each_layer():
    pack,net = prepared(),model(StaticLocalAnchorModel,geometry="age")
    operators = pack.operators["age"]
    h = net.encoder(pack.bank.leaf_x,pack.bank.query_cache.normalized_response)
    query,reference = h[:,pack.bank.query_cache.query_indices],h[:,pack.bank.query_cache.reference_indices]
    actual = net.encode_context(pack)
    for layer,message in zip(net.layers,actual.layers):
        q_signal,r_signal = layer.query_projection(query).tanh(),layer.reference_projection(reference).tanh()
        source = q_signal[:,None]-(operators.cqr@r_signal[:,None])*layer.rates[None,:,None,:]
        mass = 1-operators.cqr.sum(-1)[...,None]*layer.rates[None,:,None,:]
        numerator_branches,mass_branches = torch.empty_like(source),torch.empty_like(mass)
        for g in range(len(source)):
            for s in range(3):
                for channel in range(3):
                    k = layer.rates[s,channel]*operators.cqq[g,s]+layer.damping[s,channel]*torch.eye(3,dtype=source.dtype)
                    numerator_branches[g,s,:,channel] = torch.as_tensor(solve(k.detach().numpy(),source[g,s,:,channel].detach().numpy(),assume_a="pos"))
                    mass_branches[g,s,:,channel] = torch.as_tensor(solve(k.detach().numpy(),mass[g,s,:,channel].detach().numpy(),assume_a="pos"))
        numerator = (numerator_branches*layer.mixture[None,:,None,:]).sum(1)
        denominator = (mass_branches*layer.mixture[None,:,None,:]).sum(1)
        normalized = numerator/denominator
        torch.testing.assert_close(message.sources,source,rtol=3e-12,atol=3e-12)
        torch.testing.assert_close(message.mass_sources,mass,rtol=3e-12,atol=3e-12)
        torch.testing.assert_close(message.numerator,numerator,rtol=3e-12,atol=3e-12)
        torch.testing.assert_close(message.mass,denominator,rtol=3e-12,atol=3e-12)
        torch.testing.assert_close(message.message,normalized,rtol=3e-12,atol=3e-12)
        query = layer.update(query,normalized)
    torch.testing.assert_close(actual.local,query,rtol=3e-12,atol=3e-12)
    torch.testing.assert_close(net(pack),net.head(query[pack.bank.target_graph]),rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("geometry",["plain","age","matched_scalar_age"])
def test_static_local_has_no_context_from_other_graphs_or_within_graph_coordinate(geometry):
    pack,net = prepared(),model(StaticLocalAnchorModel,geometry=geometry)
    changed_features = pack.bank.leaf_x.clone()
    changed_features[1,:,1] += 3.
    changed = replace(pack,bank=replace(pack.bank,leaf_x=changed_features))
    original,actual = net(pack),net(changed)
    unaffected = pack.bank.target_graph!=1
    torch.testing.assert_close(actual[unaffected],original[unaffected],rtol=0,atol=0)
    assert (actual[~unaffected]-original[~unaffected]).abs().max()>1e-4
    # Every coordinate whose target_graph is the same has identical static output.
    torch.testing.assert_close(original[0],original[1],rtol=0,atol=0)
    torch.testing.assert_close(original[1],original[2],rtol=0,atol=0)
    context = net.encode_context(pack)
    changed_context = net.encode_context(changed)
    torch.testing.assert_close(context.layers[1].sources[[0,2]],changed_context.layers[1].sources[[0,2]],rtol=0,atol=0)


@pytest.mark.parametrize("cls",[NoncontextualAnchorKernelModel,StaticLocalAnchorModel])
@pytest.mark.parametrize("geometry",["plain","age","matched_scalar_age"])
def test_unary_interval_and_serialization_contracts(cls,geometry):
    net = model(cls,geometry=geometry)
    original,changed = prepared(),prepared(sequence(unary=True,split=True,serialize=True))
    torch.testing.assert_close(net(original),net(changed),rtol=3e-11,atol=3e-11)


@pytest.mark.parametrize("cls",[NoncontextualAnchorKernelModel,StaticLocalAnchorModel])
def test_query_order_and_readout_set_permutation(cls):
    bank = cached()
    net = model(cls,scales=1)
    original = prepare_anchor_kernel_bank(bank,num_anchors=5)
    expected = net(original)
    query_order = torch.tensor([2,0,1])
    reordered = with_query_reference_cache(replace(bank,query_node_ids=bank.query_node_ids[query_order],query_cache=None),torch.tensor([40]))
    torch.testing.assert_close(net(prepare_anchor_kernel_bank(reordered,num_anchors=5)),expected[:,query_order],rtol=3e-12,atol=3e-12)
    positions = torch.tensor([6,2,1,5,3,0,4])
    reordered = replace(bank,positions=bank.positions[positions],target_graph=bank.target_graph[positions])
    torch.testing.assert_close(net(prepare_anchor_kernel_bank(reordered,num_anchors=5)),expected[positions],rtol=3e-12,atol=3e-12)
    extra = replace(bank,positions=torch.cat((bank.positions,torch.tensor([.123,.47,.71],dtype=torch.float64))),
        target_graph=torch.cat((bank.target_graph,torch.tensor([0,1,2]))))
    torch.testing.assert_close(net(prepare_anchor_kernel_bank(extra,num_anchors=5))[:7],expected,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("cls",[NoncontextualAnchorKernelModel,StaticLocalAnchorModel])
def test_poison_external_labels_and_no_online_geometry_computation(cls,monkeypatch):
    pack = prepared()
    net = model(cls)
    expected = net(pack)
    class ForbiddenLabels:
        def __getattribute__(self,name):
            raise AssertionError("External labels cannot be consulted as model input")
    object.__setattr__(pack.bank,"external_labels",ForbiddenLabels())
    def forbidden(*args,**kwargs):
        raise AssertionError("No online inverse/solve/eigh/matrix_exp")
    for name in ("eigh","eigvalsh","solve","inv"):
        monkeypatch.setattr(torch.linalg,name,forbidden)
    monkeypatch.setattr(torch,"matrix_exp",forbidden)
    torch.testing.assert_close(net(pack),expected,rtol=0,atol=0)


@pytest.mark.parametrize("cls",[NoncontextualAnchorKernelModel,StaticLocalAnchorModel])
def test_log_softplus_match_and_bounded_normalized_messages(cls):
    pack = prepared()
    log,softplus = model(cls),model(cls,rate="softplus")
    torch.testing.assert_close(log(pack),softplus(pack),rtol=0,atol=0)
    with torch.no_grad():
        for layer in log.layers:
            layer.query_projection.weight.mul_(500.)
            layer.reference_projection.weight.mul_(500.)
    context = log.encode_context(pack)
    for message in context.layers:
        values = message.message if cls==StaticLocalAnchorModel else message.targets
        assert (values.abs()<=1+2e-12).all()


def test_static_ignores_anchor_grid_and_rejects_unsupported_coupling():
    bank = cached()
    net = model(StaticLocalAnchorModel,scales=1,anchors=32)
    first = prepare_anchor_kernel_bank(bank,num_anchors=1)
    second = prepare_anchor_kernel_bank(bank,num_anchors=32)
    torch.testing.assert_close(net(first),net(second),rtol=0,atol=0)
    with pytest.raises(ValueError,match="coupling='full'"):
        model(StaticLocalAnchorModel,coupling="diagonal")


@pytest.mark.parametrize("cls",[NoncontextualAnchorKernelModel,StaticLocalAnchorModel])
def test_empty_references_keep_common_schema_and_gradients(cls):
    bank = cached()
    bank = with_query_reference_cache(replace(bank,query_node_ids=bank.sample_ids,query_cache=None),torch.empty(0,dtype=torch.long))
    pack = prepare_anchor_kernel_bank(bank,num_anchors=5)
    net = model(cls,scales=1)
    output = net(pack)
    assert output.shape==(7,4,3) and torch.isfinite(output).all()
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(torch.count_nonzero(layer.reference_projection.weight.grad)==0 for layer in net.layers)
    assert net.configuration_manifest(pack)["inactive_reference_parameters"]==54


@pytest.mark.parametrize("cls",[NoncontextualAnchorKernelModel,StaticLocalAnchorModel])
def test_control_manifest_overrides_interpretation_and_preserves_current_rates(cls):
    pack,net = prepared(),model(cls)
    manifest = net.configuration_manifest(pack)
    assert manifest["parameters"]==sum(p.numel() for p in net.parameters())
    assert manifest["current_rates"]==[layer.rates.detach().tolist() for layer in net.layers]
    assert manifest["controls_implementation_sha256"]!=manifest["implementation_sha256"]
    if cls==StaticLocalAnchorModel:
        assert manifest["anchor_coordinates_used"] is False
        assert manifest["underlying_symmetric_block_kernel_is_PSD"] is None
        assert manifest["static_unormalized_equilibrium_operator_is_PSD"] is True
    else:
        assert manifest["anchor_source_feedback"] is False


@pytest.mark.parametrize("cls",[NoncontextualAnchorKernelModel,StaticLocalAnchorModel])
def test_node_storage_and_numerical_id_relabeling_equivariance(cls):
    from arggnn.kron_interface import precompute_leaf_interfaces
    seq = sequence()
    perm = torch.arange(len(seq.x)-1,-1,-1)
    inverse = torch.argsort(perm)
    original_ids = seq.node_ids[perm]
    ids = original_ids.clone()
    for old,new in {10:900,20:70,30:300,40:110}.items():
        ids[original_ids==old]=new
    changed = replace(seq,x=seq.x[perm],node_graph=seq.node_graph[perm],node_ids=ids,
        edge_index=inverse[seq.edge_index],clades=seq.clades[perm],query_node_ids=torch.tensor([900,70,300]))
    bank = precompute_leaf_interfaces(changed,time_column=3,parent_branch_column=4)
    bank = with_query_reference_cache(bank,torch.tensor([110]))
    net = model(cls,scales=1)
    expected = net(prepare_anchor_kernel_bank(cached(seq),num_anchors=5))
    actual = net(prepare_anchor_kernel_bank(bank,num_anchors=5))
    torch.testing.assert_close(actual,expected,rtol=3e-12,atol=3e-12)

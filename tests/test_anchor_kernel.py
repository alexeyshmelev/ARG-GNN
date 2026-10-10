"""Dense noncommuting kernel oracle and fixed-interface representation contracts."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
from scipy.linalg import expm, solve
import torch

from arggnn.anchor_kernel import (AnchorKernelModel, AnchorOperators,
    anchor_kernel_action, gain_matched_anchor_sources, prepare_anchor_kernel_bank)
from arggnn.data import GraphSequence
from arggnn.kron_interface import precompute_leaf_interfaces, with_query_reference_cache


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def tree(root_age, signal, unary=False):
    ages = [0., .2, .5, .8, root_age]
    edges = [(i,4) for i in range(4)]
    if unary:
        ages.append(1.)
        edges = [(0,5),(5,4),*[(i,4) for i in range(1,4)]]
    age = torch.tensor(ages,dtype=torch.float64)
    branch = torch.zeros_like(age)
    for u,v in edges:
        branch[u] = age[v]-age[u]
    x = torch.zeros(len(age),5,dtype=torch.float64)
    x[3,0] = 1. # observed reference role
    x[:4,1] = torch.tensor([signal,1-signal,.4,.7],dtype=torch.float64)
    x[:4,2] = 1. # observed leaf flag
    x[:,3] = torch.log1p(age)/math.log1p(10000.)
    x[:,4] = torch.log1p(branch)/math.log1p(10000.)
    edge = torch.tensor(edges,dtype=torch.long).T
    edge = torch.cat((edge,edge.flip(0)),dim=1)
    ids = torch.tensor([10,20,30,40]+[-1]*(len(age)-4))
    return x,edge,ids


def sequence(unary=False,split=False,serialize=False):
    graphs = [tree(1.8,.2,unary),tree(4.,.9),tree(2.3,.4)]
    intervals = [[0.,.35],[.35,.65],[.65,1.]]
    positions = torch.tensor([0.,.08,.25,.35,.57,.88,.999],dtype=torch.float64)
    target = torch.tensor([0,0,0,1,1,2,2])
    if split:
        graphs = [graphs[0],graphs[0],*graphs[1:]]
        intervals = [[0.,.17],[.17,.35],[.35,.65],[.65,1.]]
        target = torch.tensor([0,0,1,2,2,3,3])
    if serialize:
        order = torch.arange(len(graphs)-1,-1,-1)
        inverse = torch.argsort(order)
        graphs = [graphs[i] for i in order]
        intervals = [intervals[i] for i in order]
        target = inverse[target]
    xs,edges,ids,node_graph = [],[],[],[]
    offset = 0
    for graph,(x,edge,identity) in enumerate(graphs):
        xs.append(x);edges.append(edge+offset);ids.append(identity)
        node_graph.extend([graph]*len(x));offset += len(x)
    seq = GraphSequence(torch.cat(xs),torch.cat(edges,dim=1),torch.tensor(node_graph),
        torch.cat(ids),torch.zeros(offset,1,dtype=torch.float64),
        torch.tensor(intervals,dtype=torch.float64),torch.tensor([10,20,30]),positions,target,
        torch.tensor([0.,1.],dtype=torch.float64))
    seq.validate()
    return seq


def cached(seq=None):
    bank = precompute_leaf_interfaces(seq or sequence(),time_column=3,parent_branch_column=4)
    return with_query_reference_cache(bank,torch.tensor([40]))


def synthetic_age_cache(bank):
    """Positive conductance fixtures, not a claim these are physical age warps."""
    cache = bank.query_cache
    base = cache.normalized_response
    b = base.shape[-1]
    multiplier = torch.ones(b,b,dtype=base.dtype)
    multiplier[0,1] = multiplier[1,0] = .35
    multiplier[0,3] = multiplier[3,0] = .6
    adjacency = -base+torch.diag_embed(base.diagonal(dim1=-2,dim2=-1))
    responses = [base]
    for scale in (.8,.4):
        a = adjacency*multiplier*scale
        responses.append(torch.diag_embed(a.sum(-1))-a)
    response = torch.stack(responses,dim=1)
    q,r = cache.query_indices,cache.reference_indices
    cqq,cqr = response[:,:,q[:,None],q[None,:]],response[:,:,q[:,None],r[None,:]]
    eig,vectors = torch.linalg.eigh(cqq)
    ratio = response.diagonal(dim1=-2,dim2=-1).mean(-1)
    scalar_cqq = cache.cqq[:,None]*ratio[:,:,None,None]
    scalar_cqr = cache.cqr[:,None]*ratio[:,:,None,None]
    return SimpleNamespace(cqq=cqq,cqr=cqr,eigenvalues=eig,eigenvectors=vectors,
        scalar_cqq=scalar_cqq,scalar_cqr=scalar_cqr,
        scalar_eigenvalues=cache.eigenvalues[:,None]*ratio[:,:,None],
        scalar_eigenvectors=cache.eigenvectors[:,None].expand(-1,3,-1,-1),
        query_ids=cache.query_ids,reference_ids=cache.reference_ids,
        query_indices=q,reference_indices=r,normalization_scale=cache.normalization_scale,
        normalization="shared_plain_mean_boundary_diagonal")


def prepared(seq=None,anchors=5,scales=3):
    bank = cached(seq)
    age = synthetic_age_cache(bank) if scales == 3 else None
    return prepare_anchor_kernel_bank(bank,age,num_anchors=anchors,num_scales=scales)


def model(depth=2,geometry="plain",coupling="full",anchors=5,scales=3,rate="log"):
    torch.manual_seed(918)
    return AnchorKernelModel(4,3,hidden_dim=8,num_layers=2,latent_dim=3,
        kernel_depth=depth,num_scales=scales,geometry=geometry,coupling=coupling,
        num_anchors=anchors,initial_coupling=[.8,2.,5.] if scales==3 else 2.,
        initial_damping=.6,rate_parameterization=rate).double()


def dense_propagator(bank,operators,rates,damping,x,y,scale,channel,coupling="full"):
    """Chronological SciPy dense expm oracle independent of spectral actions."""
    q = operators.cqq.shape[-1]
    result = torch.eye(q,dtype=rates.dtype)
    for graph in torch.argsort(bank.intervals[:,0]).tolist():
        left,right = bank.intervals[graph].tolist()
        duration = max(0.,min(x,right)-max(y,left))
        if duration:
            if coupling=="full":
                cqq = operators.cqq[graph,scale]
            elif coupling=="scalar":
                cqq = torch.eye(q,dtype=rates.dtype)
            else:
                cqq = torch.diag(operators.cqq[graph,scale].diagonal())
            generator = rates[scale,channel]*cqq+damping[scale,channel]*torch.eye(q,dtype=rates.dtype)
            transition = torch.as_tensor(expm((-duration*generator).detach().numpy()),dtype=rates.dtype)
            result = transition@result
    return result


def dense_block(bank,operators,rates,damping,x,y,scale,channel,coupling="full"):
    if x >= y:
        return dense_propagator(bank,operators,rates,damping,x,y,scale,channel,coupling)
    return dense_propagator(bank,operators,rates,damping,y,x,scale,channel,coupling).T


def dense_action(pack,operators,rates,damping,sources,coupling="full"):
    layout = pack.layout
    coordinates = [float(x) for x,output in zip(layout.coordinates,layout.output_at_event) if output>=0]
    _,scales,q,channels,values = sources.shape
    if coupling=="gain_matched_diagonal":
        transformed = torch.empty_like(sources)
        for a,graph in enumerate(layout.anchor_graph.tolist()):
            for s in range(scales):
                for h in range(channels):
                    k = rates[s,h]*operators.cqq[graph,s]+damping[s,h]*torch.eye(q,dtype=sources.dtype)
                    equilibrium = torch.as_tensor(solve(k.detach().numpy(),sources[a,s,:,h].detach().numpy(),assume_a="pos"),dtype=sources.dtype)
                    transformed[a,s,:,h] = k.diagonal()[:,None]*equilibrium
        sources = transformed
    output = []
    for x in coordinates:
        branch = []
        for s in range(scales):
            channel = []
            for h in range(channels):
                value = torch.zeros(q,values,dtype=sources.dtype)
                for a,y in enumerate(layout.anchors.tolist()):
                    value += layout.weights[a]*dense_block(pack.bank,operators,rates,damping,x,y,s,h,coupling)@sources[a,s,:,h]
                channel.append(value)
            branch.append(torch.stack(channel,dim=1))
        output.append(torch.stack(branch))
    return torch.stack(output)


@pytest.mark.parametrize("coupling",["full","scalar","diagonal","gain_matched_diagonal"])
def test_scan_matches_dense_noncommuting_kernel_oracle(coupling):
    pack = prepared(anchors=4)
    operators = pack.operators["age"]
    # The fixture must expose reverse-order errors hidden by commuting holds.
    a,b = operators.cqq[:2,0]
    assert torch.linalg.matrix_norm(a@b-b@a) > 1e-4
    rates = torch.tensor([[.7,2.],[1.4,.9],[3.,1.]],dtype=torch.float64,requires_grad=True)
    damping = torch.tensor([[.5,.4],[.2,.6],[.8,.3]],dtype=torch.float64,requires_grad=True)
    sources = torch.randn(4,3,3,2,2,dtype=torch.float64,requires_grad=True)
    actual = anchor_kernel_action(pack.layout,operators,sources,rates,damping,coupling=coupling)
    expected = dense_action(pack,operators,rates,damping,sources,coupling)
    torch.testing.assert_close(actual,expected,rtol=3e-12,atol=3e-12)
    actual.square().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (sources,rates,damping))


def test_complete_block_kernel_and_global_nonnegative_mixture_are_psd():
    pack = prepared(anchors=4)
    operators = pack.operators["age"]
    rates = torch.tensor([[.7],[1.4],[3.]],dtype=torch.float64)
    damping = torch.full_like(rates,.4)
    coordinates = [0.,.12,.35,.58,.94]
    blocks = []
    for s in range(3):
        gram = torch.cat([torch.cat([dense_block(pack.bank,operators,rates,damping,x,y,s,0)
            for y in coordinates],dim=1) for x in coordinates],dim=0)
        torch.testing.assert_close(gram,gram.T,rtol=0,atol=1e-12)
        assert torch.linalg.eigvalsh(gram).min() > -1e-11
        blocks.append(gram)
    mixed = sum(w*gram for w,gram in zip((.2,.3,.5),blocks))
    assert torch.linalg.eigvalsh(mixed).min() > -1e-11
    row_normalized = mixed/mixed.sum(-1,keepdim=True)
    assert (row_normalized-row_normalized.T).abs().max() > 1e-3


def test_anchor_self_impulse_is_counted_once_and_reference_mass_is_included():
    pack = prepared(anchors=4,scales=1)
    operators = pack.operators["plain"]
    rates,damping = torch.full((1,1),2.,dtype=torch.float64),torch.full((1,1),.5,dtype=torch.float64)
    source = torch.zeros(4,1,3,1,1,dtype=torch.float64)
    source[2] = torch.tensor([1.,2.,3.],dtype=torch.float64).reshape(1,3,1,1)
    result = anchor_kernel_action(pack.layout,operators,source,rates,damping)
    torch.testing.assert_close(result[pack.layout.anchor_output[2]],pack.layout.weights[2]*source[2],rtol=0,atol=1e-14)
    mass = 1-operators.cqr[pack.layout.anchor_graph].sum(-1)[...,None]*rates[None,:,None,:]
    assert (mass>=1).all() and (mass>1).any()
    result = anchor_kernel_action(pack.layout,operators,torch.stack((mass,mass),dim=-1),rates,damping)
    torch.testing.assert_close(result[...,0]/result[...,1],torch.ones_like(result[...,0]),rtol=0,atol=0)
    assert (result[pack.layout.anchor_output,...,1] >= pack.layout.weights[:,None,None,None]-1e-14).all()


@pytest.mark.parametrize("depth",[1,2])
@pytest.mark.parametrize("geometry",["plain","age","matched_scalar_age"])
@pytest.mark.parametrize("coupling",["full","scalar","diagonal","gain_matched_diagonal"])
def test_all_matched_controls_shape_and_finite_parameter_gradients(depth,geometry,coupling):
    pack = prepared()
    net = model(depth,geometry,coupling)
    output = net(pack)
    assert output.shape==(7,3,3) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    context = net.encode_context(pack)
    assert len(context.layers)==depth and context.anchors.shape==(5,3,8)
    for layer,message in zip(net.layers,context.layers):
        assert (message.mass>0).all()
        expected_mass = 1-pack.operators[geometry].cqr[pack.layout.anchor_graph].sum(-1)[...,None]*layer.rates[None,:,None,:]
        torch.testing.assert_close(message.mass_sources,expected_mass,rtol=0,atol=0)


def test_model_messages_match_dense_numerator_and_ones_source_denominator():
    pack,net = prepared(anchors=4),model(anchors=4,geometry="age")
    context = net.encode_context(pack)
    for layer,message in zip(net.layers,context.layers):
        bundled = torch.stack((message.sources,message.mass_sources),dim=-1)
        oracle = dense_action(pack,pack.operators["age"],layer.rates,layer.damping,bundled)
        mixed = (oracle*layer.mixture[None,:,None,:,None]).sum(1)
        torch.testing.assert_close(message.numerator,mixed[...,0],rtol=3e-12,atol=3e-12)
        torch.testing.assert_close(message.mass,mixed[...,1],rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("geometry",["plain","age","matched_scalar_age"])
@pytest.mark.parametrize("coupling",["full","scalar","diagonal","gain_matched_diagonal"])
def test_fixed_anchor_subdivision_unary_and_graph_serialization_contracts(geometry,coupling):
    base = prepared()
    changed = prepared(sequence(unary=True,split=True,serialize=True))
    net = model(geometry=geometry,coupling=coupling)
    expected,actual = net(base),net(changed)
    torch.testing.assert_close(actual,expected,rtol=2e-11,atol=2e-11)
    torch.testing.assert_close(base.layout.anchors,changed.layout.anchors,rtol=0,atol=0)
    torch.testing.assert_close(base.layout.weights,changed.layout.weights,rtol=0,atol=0)
    assert base.bank.feature_columns==changed.bank.feature_columns==(0,1,2,3)


def test_extra_query_readouts_and_position_permutation_do_not_change_other_predictions():
    bank = cached()
    net = model(scales=1)
    original = prepare_anchor_kernel_bank(bank,num_anchors=5)
    expected = net(original)
    order = torch.tensor([6,2,1,5,3,0,4])
    permuted = replace(bank,positions=bank.positions[order],target_graph=bank.target_graph[order])
    actual = net(prepare_anchor_kernel_bank(permuted,num_anchors=5))
    torch.testing.assert_close(actual,expected[order],rtol=3e-12,atol=3e-12)
    added = replace(bank,positions=torch.cat((bank.positions,torch.tensor([.123,.47,.71],dtype=torch.float64))),
        target_graph=torch.cat((bank.target_graph,torch.tensor([0,1,2]))))
    actual = net(prepare_anchor_kernel_bank(added,num_anchors=5))
    torch.testing.assert_close(actual[:7],expected,rtol=3e-12,atol=3e-12)
    duplicated = replace(bank,positions=torch.cat((bank.positions,bank.positions[[2]])),
        target_graph=torch.cat((bank.target_graph,bank.target_graph[[2]])))
    actual = net(prepare_anchor_kernel_bank(duplicated,num_anchors=5))
    torch.testing.assert_close(actual[:7],expected,rtol=3e-12,atol=3e-12)
    torch.testing.assert_close(actual[-1],expected[2],rtol=0,atol=0)


def test_query_identity_order_equivariance():
    bank = cached()
    order = torch.tensor([2,0,1])
    changed = with_query_reference_cache(replace(bank,query_node_ids=bank.query_node_ids[order],query_cache=None),torch.tensor([40]))
    net = model(scales=1)
    expected = net(prepare_anchor_kernel_bank(bank,num_anchors=5))
    actual = net(prepare_anchor_kernel_bank(changed,num_anchors=5))
    torch.testing.assert_close(actual,expected[:,order],rtol=3e-12,atol=3e-12)


def test_node_storage_and_numerical_identity_relabeling_are_invariant():
    seq = sequence()
    # Reverse packed storage while all structural and correspondence fields move.
    perm = torch.arange(len(seq.x)-1,-1,-1)
    inverse = torch.argsort(perm)
    relabeled = {10:900,20:70,30:300,40:110}
    ids = seq.node_ids[perm].clone()
    for old,new in relabeled.items():
        ids[seq.node_ids[perm]==old] = new
    changed = replace(seq,x=seq.x[perm],node_graph=seq.node_graph[perm],node_ids=ids,
        edge_index=inverse[seq.edge_index],clades=seq.clades[perm],query_node_ids=torch.tensor([900,70,300]))
    bank = precompute_leaf_interfaces(changed,time_column=3,parent_branch_column=4)
    bank = with_query_reference_cache(bank,torch.tensor([110]))
    net = model(scales=1)
    expected = net(prepare_anchor_kernel_bank(cached(seq),num_anchors=5))
    actual = net(prepare_anchor_kernel_bank(bank,num_anchors=5))
    torch.testing.assert_close(actual,expected,rtol=3e-12,atol=3e-12)
    assert not any(isinstance(module,torch.nn.Embedding) for module in net.modules())


@pytest.mark.parametrize("coupling",["full","scalar","diagonal","gain_matched_diagonal"])
def test_log_and_softplus_have_identical_initial_rates_outputs_and_parameter_counts(coupling):
    pack = prepared()
    log,softplus = model(rate="log",coupling=coupling),model(rate="softplus",coupling=coupling)
    assert list(log.state_dict())==list(softplus.state_dict())
    assert sum(p.numel() for p in log.parameters())==sum(p.numel() for p in softplus.parameters())
    for a,b in zip(log.layers,softplus.layers):
        torch.testing.assert_close(a.rates,b.rates,rtol=0,atol=0)
        torch.testing.assert_close(a.damping,b.damping,rtol=0,atol=0)
    torch.testing.assert_close(log(pack),softplus(pack),rtol=0,atol=0)
    softplus(pack).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in softplus.parameters())
    assert len(list(log.rate_and_mixture_parameters()))==3*log.kernel_depth


def test_default32_anchor_grid_and_plain_only_cache_need_no_eigh_online(monkeypatch):
    bank = cached()
    def forbidden(*args,**kwargs):
        raise AssertionError("No online eigensolve or dense matrix exponential")
    monkeypatch.setattr(torch.linalg,"eigh",forbidden)
    monkeypatch.setattr(torch.linalg,"eigvalsh",forbidden)
    monkeypatch.setattr(torch,"matrix_exp",forbidden)
    pack = prepare_anchor_kernel_bank(bank)
    expected = (torch.arange(32,dtype=torch.float64)+.5)/32
    torch.testing.assert_close(pack.layout.anchors,expected,rtol=0,atol=0)
    net = model(anchors=32,scales=1)
    assert torch.isfinite(net(pack)).all()
    assert pack.operators.keys()=={"plain"}
    with pytest.raises(ValueError,match="geometry differs"):
        model(anchors=32,scales=1,geometry="age")(pack)


def test_empty_references_keep_schema_and_zero_response_is_valid():
    bank = cached()
    all_ids = bank.sample_ids.clone()
    bank = with_query_reference_cache(replace(bank,query_node_ids=all_ids,query_cache=None),torch.empty(0,dtype=torch.long))
    bank = replace(bank,response=torch.zeros_like(bank.response),query_cache=None)
    bank = with_query_reference_cache(bank,torch.empty(0,dtype=torch.long))
    pack = prepare_anchor_kernel_bank(bank,num_anchors=5)
    net = model(scales=1)
    output = net(pack)
    assert output.shape==(7,4,3) and torch.isfinite(output).all()
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(torch.count_nonzero(layer.reference_projection.weight.grad)==0 for layer in net.layers)
    assert net.configuration_manifest(pack)["inactive_reference_parameters"]==2*9*3


def test_nonanchor_row_mass_underflow_is_rejected_without_a_floor():
    pack = prepared(anchors=1,scales=1)
    net = model(anchors=1,scales=1)
    with torch.no_grad():
        for layer in net.layers:
            layer.raw_damping.fill_(30.)
    with pytest.raises(ValueError,match="mass is nonpositive"):
        net(pack)


def test_nonfinite_rate_overflow_is_rejected_without_clipping():
    pack,net = prepared(scales=1),model(scales=1)
    with torch.no_grad():
        net.layers[0].raw_coupling.fill_(1000.)
    with pytest.raises(ValueError,match="finite"):
        net(pack)


@pytest.mark.parametrize("error",["normalization","ids","scale","plainfirst"])
def test_reject_unmatched_age_cache_contract(error):
    bank = cached()
    age = synthetic_age_cache(bank)
    if error=="normalization": age.normalization="independent_scale"
    elif error=="ids": age.query_ids=age.query_ids.flip(0)
    elif error=="scale": age.normalization_scale=age.normalization_scale*2
    else: age.cqq=age.cqq.flip(1)
    with pytest.raises(ValueError):
        prepare_anchor_kernel_bank(bank,age,num_anchors=5)


def test_scalar_scan_matches_analytic_exponential_without_spatial_transfer():
    pack = prepared(anchors=4)
    operators = pack.operators["age"]
    rates = torch.tensor([[.7,2.],[1.4,.9],[3.,1.]],dtype=torch.float64)
    damping = torch.full_like(rates,.4)
    sources = torch.randn(4,3,3,2,1,dtype=torch.float64)
    sources[:,:,1:] = 0
    actual = anchor_kernel_action(pack.layout,operators,sources,rates,damping,coupling="scalar")
    coordinates = [float(x) for x,o in zip(pack.layout.coordinates,pack.layout.output_at_event) if o>=0]
    expected = torch.stack([sum(pack.layout.weights[a]*torch.exp(-abs(x-y)*(rates+damping))[:,None,:,None]*sources[a]
        for a,y in enumerate(pack.layout.anchors.tolist())) for x in coordinates])
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    assert torch.count_nonzero(actual[:,:,1:])==0


def test_geometry_and_coupling_controls_have_identical_parameter_schemas():
    nets = [model(geometry=g,coupling=c) for g in ("plain","age","matched_scalar_age") for c in ("full","scalar","diagonal","gain_matched_diagonal")]
    keys = [(name,tuple(value.shape)) for name,value in nets[0].named_parameters()]
    for net in nets[1:]:
        assert [(name,tuple(value.shape)) for name,value in net.named_parameters()]==keys
        assert sum(p.numel() for p in net.parameters())==sum(p.numel() for p in nets[0].parameters())


@pytest.mark.parametrize("geometry",["plain","age","matched_scalar_age"])
def test_real_age_cache_adapter_and_two_subdivision_contracts(geometry,monkeypatch):
    from arggnn.age_interface import precompute_age_interfaces
    seq,changed = sequence(),sequence(unary=True,split=True,serialize=True)
    def age_pack(source):
        bank = cached(source)
        age = precompute_age_interfaces(source,plain_bank=bank,time_column=3,
            parent_branch_column=4,resistance_scale=1.)
        return prepare_anchor_kernel_bank(age.plain_bank,age.operator_cache,num_anchors=5)
    base,refined = age_pack(seq),age_pack(changed)
    def forbidden(*args,**kwargs):
        raise AssertionError("Online eigensolve after physical cache preparation")
    monkeypatch.setattr(torch.linalg,"eigh",forbidden)
    monkeypatch.setattr(torch.linalg,"eigvalsh",forbidden)
    monkeypatch.setattr(torch,"matrix_exp",forbidden)
    net = model(geometry=geometry)
    torch.testing.assert_close(net(base),net(refined),rtol=3e-11,atol=3e-11)


def test_default47_feature_schema_and_exact32_anchor_model():
    seq = sequence()
    padded = torch.cat((seq.x,torch.zeros(len(seq.x),43,dtype=seq.x.dtype)),dim=-1)
    bank = cached(replace(seq,x=padded))
    assert bank.leaf_x.shape[-1]==47 and 4 not in bank.feature_columns
    pack = prepare_anchor_kernel_bank(bank).to("cpu",torch.float32)
    net = AnchorKernelModel()
    output = net(pack)
    assert output.shape==(7,3,4) and torch.isfinite(output).all()
    manifest = net.configuration_manifest(pack)
    assert manifest["input_dim"]==47 and manifest["num_anchors"]==32
    assert manifest["kernel_depth"]==2 and manifest["num_scales"]==1
    assert manifest["row_normalized_operator_is_PSD_Gram"] is False


@pytest.mark.parametrize("geometry",["plain","age","matched_scalar_age"])
@pytest.mark.parametrize("coupling",["full","scalar","diagonal","gain_matched_diagonal"])
def test_bounded_normalized_messages_with_saturated_sources_and_varied_rates(geometry,coupling):
    pack,net = prepared(),model(geometry=geometry,coupling=coupling)
    with torch.no_grad():
        for layer in net.layers:
            # Diverse node/anchor signs approach the tanh endpoints. Rate and
            # mixture changes preserve the convex source-weight interpretation.
            layer.query_projection.weight.mul_(500.)
            layer.reference_projection.weight.mul_(500.)
            layer.raw_coupling.copy_(torch.tensor([[-5.,0.,5.],[2.,-2.,1.],[0.,3.,-3.]]))
            layer.raw_damping.copy_(torch.tensor([[-2.,1.,0.],[0.,-1.,2.],[1.,0.,-2.]]))
            layer.mixture_logits.copy_(torch.tensor([[2.,-1.,0.],[-2.,3.,1.],[0.,-2.,2.]]))
    context = net.encode_context(pack)
    assert any((message.sources.abs()>1).any() for message in context.layers)
    for message in context.layers:
        assert (message.anchors.abs()<=1+2e-12).all()
        assert (message.targets.abs()<=1+2e-12).all()


def test_gain_matching_preserves_homogeneous_equilibrium_but_not_self_impulse():
    pack = prepared(anchors=4)
    operators = pack.operators["age"]
    rates = torch.tensor([[.7,2.],[1.4,.9],[3.,1.]],dtype=torch.float64)
    damping = torch.full_like(rates,.4)
    source = torch.zeros(4,3,3,2,2,dtype=torch.float64)
    source[2,...,0] = torch.tensor([1.,-.5,.2],dtype=torch.float64)[None,:,None]
    source[2,...,1] = 1
    transformed = gain_matched_anchor_sources(operators,pack.layout.anchor_graph,source,rates,damping)
    for a,graph in enumerate(pack.layout.anchor_graph.tolist()):
        for s in range(3):
            for h in range(2):
                k = rates[s,h]*operators.cqq[graph,s]+damping[s,h]*torch.eye(3,dtype=source.dtype)
                full_equilibrium = torch.linalg.solve(k,source[a,s,:,h])
                independent_equilibrium = transformed[a,s,:,h]/k.diagonal()[:,None]
                torch.testing.assert_close(independent_equilibrium,full_equilibrium,rtol=3e-12,atol=3e-12)
    gain = anchor_kernel_action(pack.layout,operators,source,rates,damping,coupling="gain_matched_diagonal")
    full = anchor_kernel_action(pack.layout,operators,source,rates,damping,coupling="full")
    at_self = pack.layout.anchor_output[2]
    torch.testing.assert_close(gain[at_self],pack.layout.weights[2]*transformed[2],rtol=0,atol=1e-14)
    assert (gain[at_self]-full[at_self]).abs().max()>.01
    # Independent propagation blocks contain no prior-state cross-query path.
    block = dense_block(pack.bank,operators,rates,damping,.88,.08,0,0,"gain_matched_diagonal")
    assert torch.count_nonzero(block-torch.diag(block.diagonal()))==0
    # The source transform itself deliberately retains instantaneous mixing.
    assert torch.count_nonzero(transformed[2,0,:,0,0]-source[2,0,:,0,0])>0


def test_raw_diagonal_has_no_cross_query_source_propagation():
    pack = prepared(anchors=4)
    rates = torch.ones(3,2,dtype=torch.float64)
    damping = torch.ones_like(rates)*.4
    source = torch.randn(4,3,3,2,1,dtype=torch.float64)
    source[:,:,1:]=0
    result = anchor_kernel_action(pack.layout,pack.operators["age"],source,rates,damping,coupling="diagonal")
    assert torch.count_nonzero(result[:,:,1:])==0


def test_manifest_records_current_rate_damping_mixture_and_raw_parameters():
    net = model()
    with torch.no_grad():
        net.layers[0].raw_coupling.fill_(.3)
        net.layers[0].raw_damping.fill_(-.2)
        net.layers[0].mixture_logits[0].fill_(2.)
    manifest = net.configuration_manifest()
    for field,attribute in (("current_rates","rates"),("current_damping","damping"),
            ("current_mixture","mixture"),("raw_coupling","raw_coupling"),
            ("raw_damping","raw_damping"),("mixture_logits","mixture_logits")):
        assert manifest[field]==[getattr(layer,attribute).detach().tolist() for layer in net.layers]
    assert manifest["current_rates"][0]!=manifest["current_rates"][1]

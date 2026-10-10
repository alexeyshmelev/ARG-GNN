"""Exact coupled memory, invariant full inputs and explicit control contracts."""
from dataclasses import replace
import math

import pytest
import torch
from torch import nn

from arggnn.boundary_memory import (BoundaryMemoryModel, BoundarySourceBank,
                                   diagonal_affine_action, spectral_affine_action, spectral_equilibrium,
                                   gain_matched_diagonal_forcing, query_grounding_leakage)
from arggnn.data import GraphSequence
from arggnn.kron_interface import (precompute_graph_interfaces, precompute_leaf_interfaces,
                                  with_query_reference_cache)


def tree(root_age=2., signal=.2, unary=False, cap=False):
    times=[0., .2, .4, root_age]
    edges=[(0, 3), (1, 3), (2, 3)]
    if unary:
        times.append(.7)
        edges=[(0, 4), (4, 3), (1, 3), (2, 3)]
    if cap:
        index=len(times);times.append(root_age+1.5);edges.append((3, index))
    ages=torch.tensor(times,dtype=torch.float64)
    branch=torch.zeros(len(times),dtype=torch.float64)
    for u,v in edges: branch[u]=ages[v]-ages[u]
    x=torch.zeros(len(times),5,dtype=torch.float64)
    x[2,0]=1. # observed pure-reference role
    x[:3,1]=torch.tensor([signal, 1-signal, .7],dtype=torch.float64)
    x[:3,2]=1. # observed leaf flag
    x[:,3]=torch.log1p(ages)/math.log1p(10000.)
    x[:,4]=torch.log1p(branch)/math.log1p(10000.)
    edge=torch.tensor(edges,dtype=torch.long).T
    return x,torch.cat((edge,edge.flip(0)),dim=1),torch.tensor([10,20,30]+[-1]*(len(times)-3))


def sequence(unary=False,cap=False,refine=False):
    graphs=[tree(2.,.2,unary,cap),tree(4.,.9),tree(3.,.4)]
    intervals=[[0.,.4],[.4,.7],[.7,1.]]
    target=[0,0,1,2]
    if refine:
        graphs=[graphs[0],graphs[0],*graphs[1:]]
        intervals=[[0.,.15],[.15,.4],[.4,.7],[.7,1.]]
        target=[0,1,2,3]
    xs=[];edges=[];ids=[];membership=[];offset=0
    for g,(x,e,node_ids) in enumerate(graphs):
        xs.append(x);edges.append(e+offset);ids.append(node_ids)
        membership.extend([g]*len(x));offset+=len(x)
    result=GraphSequence(torch.cat(xs),torch.cat(edges,dim=1),torch.tensor(membership),torch.cat(ids),
        torch.zeros(offset,1,dtype=torch.float64),torch.tensor(intervals,dtype=torch.float64),
        torch.tensor([10,20]),torch.tensor([.1,.3,.6,.9],dtype=torch.float64),torch.tensor(target),
        torch.tensor([0.,1.],dtype=torch.float64))
    result.validate()
    return result


def cached(seq):
    bank=precompute_leaf_interfaces(seq,time_column=3,parent_branch_column=4)
    return with_query_reference_cache(bank,torch.tensor([30]))


def model(coupling="full",aggregation="ordered",bidirectional=True,num_classes=3,latent=3):
    torch.manual_seed(41)
    return BoundaryMemoryModel(4,num_classes,hidden_dim=8,num_layers=2,latent_dim=latent,
        coupling=coupling,aggregation=aggregation,bidirectional=bidirectional).double()


@pytest.fixture(autouse=True)
def cpu_threads(): torch.set_num_threads(1)


@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal","gain_leak_diagonal"])
@pytest.mark.parametrize("aggregation",["ordered","homogeneous_bag","equilibrium"])
def test_shape_and_finite_training_gradients(coupling,aggregation):
    bank=cached(sequence())
    net=model(coupling,aggregation)
    output=net(bank)
    assert output.shape==(4,2,3) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert not any(isinstance(layer,nn.Embedding) for layer in net.modules())


@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal","gain_leak_diagonal"])
@pytest.mark.parametrize("aggregation",["ordered","homogeneous_bag","equilibrium"])
def test_dual_unary_and_interval_subdivision_with_full_invariant_input(coupling,aggregation):
    base=cached(sequence())
    split=cached(sequence(unary=True,cap=True,refine=True))
    assert 4 not in split.feature_columns
    torch.testing.assert_close(base.leaf_x[0],split.leaf_x[0],rtol=0,atol=0)
    torch.testing.assert_close(base.query_cache.normalized_response[0],split.query_cache.normalized_response[0],rtol=1e-12,atol=1e-12)
    net=model(coupling,aggregation)
    torch.testing.assert_close(net(base),net(split),rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize("aggregation",["ordered","homogeneous_bag","equilibrium"])
@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal","gain_leak_diagonal"])
def test_node_identity_storage_and_query_order_invariance(aggregation,coupling):
    seq=sequence(unary=True,cap=True)
    net=model(coupling=coupling,aggregation=aggregation)
    base=net(cached(seq))
    permutation=torch.randperm(len(seq.x),generator=torch.Generator().manual_seed(11))
    inverse=torch.empty_like(permutation);inverse[permutation]=torch.arange(len(permutation))
    nodes=replace(seq,x=seq.x[permutation],edge_index=inverse[seq.edge_index],node_graph=seq.node_graph[permutation],
        node_ids=seq.node_ids[permutation],clades=seq.clades[permutation])
    torch.testing.assert_close(base,net(cached(nodes)),rtol=1e-10,atol=1e-10)
    mapping={10:300,20:100,30:200} # changes numeric sorting, not only a common offset
    renamed=replace(seq,node_ids=torch.tensor([mapping.get(int(x),-1) for x in seq.node_ids]),query_node_ids=torch.tensor([300,100]))
    renamed_bank=with_query_reference_cache(precompute_leaf_interfaces(renamed,time_column=3,parent_branch_column=4),torch.tensor([200]))
    torch.testing.assert_close(base,net(renamed_bank),rtol=1e-10,atol=1e-10)
    order=torch.tensor([2,0,1]);remap=torch.empty_like(order);remap[order]=torch.arange(3)
    serialized=replace(seq,node_graph=remap[seq.node_graph],intervals=seq.intervals[order],target_graph=remap[seq.target_graph])
    torch.testing.assert_close(base,net(cached(serialized)),rtol=1e-10,atol=1e-10)
    queries=replace(seq,query_node_ids=seq.query_node_ids.flip(0))
    torch.testing.assert_close(base.flip(1),net(cached(queries)),rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize("aggregation",["ordered","homogeneous_bag","equilibrium"])
def test_positive_reverse_scan_and_symmetric_reflection_readout(aggregation):
    seq=sequence()
    reflected=replace(seq,intervals=torch.stack((1-seq.intervals[:,1],1-seq.intervals[:,0]),-1),positions=1-seq.positions)
    net=model(aggregation=aggregation)
    torch.testing.assert_close(net(cached(seq)),net(cached(reflected)),rtol=1e-10,atol=1e-10)


def test_exact_spectral_action_matches_dense_affine_oracle_and_semigroup():
    c=torch.tensor([[1.2,-.4],[-.4,.9]],dtype=torch.float64)
    values,vectors=torch.linalg.eigh(c)
    state=torch.tensor([[.2,-.3],[.7,.4]],dtype=torch.float64)
    forcing=torch.tensor([[.4,.8],[-.2,.1]],dtype=torch.float64,requires_grad=True)
    rate=torch.tensor([.7,2.],dtype=torch.float64,requires_grad=True)
    damping=torch.tensor([.1,.4],dtype=torch.float64,requires_grad=True)
    whole=spectral_affine_action(values,vectors,state,forcing,rate,damping,.6)
    columns=[]
    for h in range(2):
        a=-rate[h]*c-damping[h]*torch.eye(2,dtype=torch.float64)
        upper=torch.cat((a,forcing[:,h,None]),1)
        generator=torch.cat((upper,torch.zeros(1,3,dtype=torch.float64)),0)
        result=torch.matrix_exp(.6*generator)@torch.cat((state[:,h],torch.ones(1,dtype=torch.float64)))
        columns.append(result[:2])
    torch.testing.assert_close(whole,torch.stack(columns,1),rtol=1e-12,atol=1e-12)
    first=spectral_affine_action(values,vectors,state,forcing,rate,damping,.2)
    split=spectral_affine_action(values,vectors,first,forcing,rate,damping,.4)
    torch.testing.assert_close(whole,split,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(spectral_affine_action(values,vectors,state,forcing,rate,damping,0.),state,rtol=0,atol=0)
    whole.square().sum().backward()
    for p in (forcing,rate,damping): assert p.grad is not None and torch.isfinite(p.grad).all()
    with pytest.raises(ValueError,match="nonnegative"):
        spectral_affine_action(values,vectors,state,forcing,rate,damping,-.1)


def geographic(q=2,first_zero=False):
    gen=torch.Generator().manual_seed(516)
    x=torch.randn(2*q,4,generator=gen,dtype=torch.float64)
    pairs=[]
    for g in range(2):
        if g==0 and first_zero: continue
        links=[(g*q+i,g*q+(i+1)%q) for i in range(q)] if q>2 else [(g*q,g*q+1)]
        pairs+=links+[(v,u) for u,v in links]
    edge=torch.tensor(pairs,dtype=torch.long).T.contiguous()
    seq=GraphSequence(x,edge,torch.arange(2).repeat_interleave(q),torch.arange(q).repeat(2),
        torch.zeros(2*q,1,dtype=torch.float64),torch.tensor([[0.,.5],[.5,1.]],dtype=torch.float64),
        torch.arange(q),torch.tensor([.2,.8],dtype=torch.float64),torch.tensor([0,1]),
        torch.tensor([0.,1.],dtype=torch.float64))
    bank=precompute_graph_interfaces(seq,torch.ones(edge.shape[1],dtype=torch.float64))
    return with_query_reference_cache(bank,torch.empty(0,dtype=torch.long))


def test_other_query_memory_reaches_target_across_time_only_when_coupled():
    bank=geographic(first_zero=True)
    net=model(bidirectional=False,num_classes=1,latent=1)
    diagonal=model(coupling="diagonal",bidirectional=False,num_classes=1,latent=1)
    diagonal.load_state_dict(net.state_dict())
    forcing=torch.zeros(2,2,1,dtype=torch.float64);forcing[0,1,0]=1.
    independent=BoundarySourceBank(torch.zeros(2,2,8,dtype=torch.float64),forcing,torch.empty(2,0,1,dtype=torch.float64),forcing)
    net.source_bank=lambda unused: independent
    diagonal.source_bank=lambda unused: independent
    full=net.encode_context(bank).forward
    isolated=diagonal.encode_context(bank).forward
    assert full[0,0,0]==0. and isolated[0,0,0]==0.
    assert full[1,0,0]>0. and isolated[1,0,0]==0.
    assert full[1,1,0]>0.


def test_learned_past_evidence_gradient_crosses_queries_with_future_forcing_fixed():
    bank=geographic(first_zero=True)
    full=model(bidirectional=False,num_classes=1,latent=1)
    diagonal=model(coupling="diagonal",bidirectional=False,num_classes=1,latent=1)
    with torch.no_grad():
        full.head.weight.zero_();full.head.weight[0,8]=1.;full.head.bias.zero_()
    diagonal.load_state_dict(full.state_dict())
    gradients=[]
    for net in (full,diagonal):
        features=bank.leaf_x.clone().requires_grad_(True)
        net(replace(bank,leaf_x=features))[1,0,0].backward()
        gradients.append(features.grad[0,1].abs().sum())
    assert gradients[0]>1e-8 and gradients[1]==0.


def test_full_diagonal_controls_keep_identical_local_sources_and_forcing():
    bank=cached(sequence())
    full,diagonal=model(),model(coupling="diagonal")
    diagonal.load_state_dict(full.state_dict())
    a,b=full.source_bank(bank),diagonal.source_bank(bank)
    torch.testing.assert_close(a.local,b.local,rtol=0,atol=0)
    torch.testing.assert_close(a.forcing,b.forcing,rtol=0,atol=0)
    assert sum(p.numel() for p in full.parameters())==sum(p.numel() for p in diagonal.parameters())
    assert (full(bank)-diagonal(bank)).abs().max()>1e-7
    assert a.query_signal.abs().max()<=1. and a.reference_signal.abs().max()<=1.


def test_bag_control_discards_arrangement_but_ordered_flow_keeps_it():
    seq=sequence();graph=seq.node_graph.clone()
    graph[seq.node_graph==1]=2;graph[seq.node_graph==2]=1
    moved=replace(seq,node_graph=graph)
    before,after=cached(seq),cached(moved)
    bag=model(aggregation="homogeneous_bag")
    torch.testing.assert_close(bag(before)[:2],bag(after)[:2],rtol=1e-10,atol=1e-10)
    ordered=model()
    assert (ordered(before)[:2]-ordered(after)[:2]).abs().max()>1e-8


def test_one_sided_flow_does_not_read_later_source_evidence():
    seq=sequence();changed=seq.x.clone()
    changed[seq.node_graph==2,1]+=3.
    net=model(bidirectional=False)
    torch.testing.assert_close(net(cached(seq))[:3],net(cached(replace(seq,x=changed)))[:3],rtol=0,atol=0)


@pytest.mark.parametrize("q",[8,16])
def test_prototype_query_counts_and_scalar_geographic_output(q):
    bank=geographic(q)
    net=model(bidirectional=False,num_classes=1,latent=4)
    result=net(bank)
    assert result.shape==(2,q,1) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert bank.query_cache.reference_indices.numel()==0


def test_static_equilibrium_is_stationary_and_has_no_unused_reverse_head():
    bank=cached(sequence())
    net=model(aggregation="equilibrium")
    context=net.encode_context(bank)
    sources=net.source_bank(bank);target=bank.target_graph
    moved=spectral_affine_action(bank.query_cache.eigenvalues[target],bank.query_cache.eigenvectors[target],
        context.forward,sources.forcing[target],net.rates,net.damping,.7)
    torch.testing.assert_close(context.forward,moved,rtol=1e-12,atol=1e-12)
    assert context.backward is None and net.head.in_features==8+3


@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal","gain_leak_diagonal"])
def test_no_online_eigh_or_augmented_exp_and_float32_coordinate_precision(monkeypatch,coupling):
    bank=cached(sequence()).to('cpu',torch.float32)
    assert bank.positions.dtype==torch.float64
    net=model(coupling=coupling).float()
    def forbidden(*args,**kwargs): raise AssertionError("online dense operation")
    monkeypatch.setattr(torch.linalg,'eigh',forbidden)
    monkeypatch.setattr(torch,'matrix_exp',forbidden)
    output=net(bank)
    assert output.shape==(4,2,3) and torch.isfinite(output).all()
    manifest=net.configuration_manifest(bank)
    assert manifest['online_eigensystem_or_augmented_matrix_exp'] is False
    assert manifest['coordinate_dtype']=='torch.float64'


def test_invalid_cache_dtype_and_numerical_instability_rejected():
    raw=precompute_leaf_interfaces(sequence(),time_column=3,parent_branch_column=4)
    with pytest.raises(ValueError,match="cache"):
        model()(raw)
    with pytest.raises(ValueError,match="device/dtype"):
        model().float()(cached(sequence()))
    state=torch.ones(2,1,dtype=torch.float64)
    with pytest.raises(ValueError,match="dissipative"):
        diagonal_affine_action(torch.tensor([-1.,1.],dtype=torch.float64),state,state,
            torch.ones(1,dtype=torch.float64),torch.tensor([.1],dtype=torch.float64),.2)


def test_physical_unit_cache_is_not_silently_accepted_as_normalized():
    raw=precompute_leaf_interfaces(sequence(),time_column=3,parent_branch_column=4)
    physical=with_query_reference_cache(raw,torch.tensor([30]),normalization="none")
    with pytest.raises(ValueError,match="mean-diagonal"):
        model()(physical)


@pytest.mark.parametrize("coupling",["gain_matched_diagonal","gain_leak_diagonal"])
def test_gain_controls_preserve_full_equilibrium_and_remove_initial_state_transfer(coupling):
    c=torch.tensor([[1.2,-.4],[-.4,.9]],dtype=torch.float64)
    values,vectors=torch.linalg.eigh(c)
    rates=torch.tensor([.7,2.],dtype=torch.float64)
    damping=torch.tensor([.1,.4],dtype=torch.float64)
    forcing=torch.tensor([[.4,.8],[-.2,.1]],dtype=torch.float64,requires_grad=True)
    initial=torch.tensor([[.2,-.3],[.7,.4]],dtype=torch.float64,requires_grad=True)
    diagonal=c.diagonal() if coupling=="gain_matched_diagonal" else query_grounding_leakage(c)
    matched=gain_matched_diagonal_forcing(diagonal,values,vectors,forcing,rates,damping)
    equilibrium=spectral_equilibrium(values,vectors,forcing,rates,damping)
    torch.testing.assert_close(matched/(diagonal[:,None]*rates+damping),equilibrium,rtol=1e-12,atol=1e-12)
    evolved=diagonal_affine_action(diagonal,initial,matched,rates,damping,.6)
    columns=[]
    for h in range(2):
        k=rates[h]*c+damping[h]*torch.eye(2,dtype=torch.float64)
        dense_e=torch.linalg.solve(k,forcing[:,h])
        d=rates[h]*diagonal+damping[h]
        upper=torch.cat((-torch.diag(d),(d*dense_e)[:,None]),1)
        generator=torch.cat((upper,torch.zeros(1,3,dtype=torch.float64)),0)
        oracle=torch.matrix_exp(.6*generator)@torch.cat((initial[:,h],torch.ones(1,dtype=torch.float64)))
        columns.append(oracle[:2])
    torch.testing.assert_close(evolved,torch.stack(columns,1),rtol=1e-12,atol=1e-12)
    derivative=torch.autograd.grad(evolved[0,0],initial,retain_graph=True)[0]
    assert derivative[1].abs().sum()==0. and derivative[0,0]>0.
    # Instantaneous forcing is still mixed across queries by the full equilibrium.
    source_derivative=torch.autograd.grad(matched[0,0],forcing)[0]
    assert source_derivative[1,0].abs()>1e-8


@pytest.mark.parametrize("coupling",["gain_matched_diagonal","gain_leak_diagonal"])
def test_gain_control_static_and_bag_equilibrium_match_full_model(coupling):
    bank=cached(sequence())
    full=model(aggregation="equilibrium")
    matched=model(coupling=coupling,aggregation="equilibrium")
    matched.load_state_dict(full.state_dict())
    torch.testing.assert_close(full(bank),matched(bank),rtol=1e-12,atol=1e-12)
    full=model(aggregation="homogeneous_bag")
    matched=model(coupling=coupling,aggregation="homogeneous_bag")
    matched.load_state_dict(full.state_dict())
    sources=full.source_bank(bank);cache=bank.query_cache
    f=torch.einsum("g,gqh->qh",cache.interval_weights,sources.forcing)
    e=spectral_equilibrium(cache.bag_eigenvalues,cache.bag_eigenvectors,f,full.rates,full.damping)
    diagonal=cache.bag_cqq.diagonal() if coupling=="gain_matched_diagonal" else query_grounding_leakage(cache.bag_cqq)
    matched_force=gain_matched_diagonal_forcing(diagonal,cache.bag_eigenvalues,cache.bag_eigenvectors,f,full.rates,full.damping)
    torch.testing.assert_close(matched_force/(diagonal[:,None]*full.rates+full.damping),e,rtol=1e-12,atol=1e-12)
    assert sum(p.numel() for p in full.parameters())==sum(p.numel() for p in matched.parameters())
    torch.testing.assert_close(full.source_bank(bank).forcing,matched.source_bank(bank).forcing,rtol=0,atol=0)


@pytest.mark.parametrize("coupling",["gain_matched_diagonal","gain_leak_diagonal"])
def test_gain_control_memory_cannot_receive_another_query_past_state(coupling):
    bank=geographic(first_zero=True)
    full=model(bidirectional=False,num_classes=1,latent=1)
    matched=model(coupling=coupling,bidirectional=False,num_classes=1,latent=1)
    matched.load_state_dict(full.state_dict())
    f=torch.zeros(2,2,1,dtype=torch.float64);f[0,1,0]=1.
    sources=BoundarySourceBank(torch.zeros(2,2,8,dtype=torch.float64),f,torch.empty(2,0,1,dtype=torch.float64),f)
    full.source_bank=lambda unused:sources
    matched.source_bank=lambda unused:sources
    assert full.encode_context(bank).forward[1,0,0]>0.
    assert matched.encode_context(bank).forward[1,0,0]==0.


def test_leakage_roundoff_is_bounded_and_material_negative_rows_rejected():
    eps=torch.finfo(torch.float64).eps
    roundoff=torch.tensor([[1.,-(1.+eps)],[-(1.+eps),1.]],dtype=torch.float64)
    torch.testing.assert_close(query_grounding_leakage(roundoff),torch.zeros(2,dtype=torch.float64),rtol=0,atol=0)
    invalid=torch.tensor([[1.,-1.1],[-1.1,1.5]],dtype=torch.float64)
    with pytest.raises(ValueError,match="materially negative"):
        query_grounding_leakage(invalid)


def test_position_order_is_preserved_for_parent_scatter():
    bank=cached(sequence());net=model()
    order=torch.tensor([3,0,2,1])
    permuted=replace(bank,positions=bank.positions[order],target_graph=bank.target_graph[order])
    torch.testing.assert_close(net(bank)[order],net(permuted),rtol=1e-12,atol=1e-12)


def test_half_open_breakpoint_selects_right_local_graph_and_limits_reflection():
    seq=sequence()
    at=replace(seq,positions=torch.tensor([.4],dtype=torch.float64),target_graph=torch.tensor([1]))
    bank=cached(at);net=model()
    context=net.encode_context(bank)
    torch.testing.assert_close(context.local[0],net.source_bank(bank).local[1],rtol=0,atol=0)
    invalid=replace(bank,target_graph=torch.tensor([0]))
    with pytest.raises(ValueError,match="contain fixed coordinate"):
        net(invalid)
    reflected=replace(at,intervals=torch.stack((1-at.intervals[:,1],1-at.intervals[:,0]),-1),
                      positions=1-at.positions,target_graph=torch.tensor([0]))
    reverse=net.encode_context(cached(reflected))
    torch.testing.assert_close(context.forward,reverse.backward,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(context.backward,reverse.forward,rtol=1e-10,atol=1e-10)
    assert (context.local-reverse.local).abs().max()>1e-8
    assert (net(bank)-net(cached(reflected))).abs().max()>1e-8


@pytest.mark.parametrize("coupling",["full","diagonal","gain_matched_diagonal","gain_leak_diagonal"])
def test_batched_scan_matches_public_validated_actions(coupling):
    bank=cached(sequence());net=model(coupling=coupling)
    source=net.source_bank(bank);cache=bank.query_cache
    if coupling=="full": values,vectors=cache.eigenvalues,cache.eigenvectors
    else:
        values=query_grounding_leakage(cache.cqq) if coupling=="gain_leak_diagonal" else cache.cqq.diagonal(dim1=-2,dim2=-1)
        vectors=None
    forcing=source.forcing
    if coupling in {"gain_matched_diagonal","gain_leak_diagonal"}:
        forcing=gain_matched_diagonal_forcing(values,cache.eigenvalues,cache.eigenvectors,forcing,net.rates,net.damping)
    def action(g,state,duration):
        basis=None if vectors is None else vectors[g]
        return net._action(values[g],basis,state,forcing[g],duration)
    order=torch.argsort(bank.intervals[:,0]).tolist()
    left=[];state=torch.zeros(2,3,dtype=torch.float64)
    for g in order:
        left.append(state);state=action(g,state,bank.intervals[g,1]-bank.intervals[g,0])
    right=[state]*len(order);state=torch.zeros(2,3,dtype=torch.float64)
    for rank in range(len(order)-1,-1,-1):
        g=order[rank];right[rank]=state;state=action(g,state,bank.intervals[g,1]-bank.intervals[g,0])
    inverse=torch.empty(len(order),dtype=torch.long);inverse[torch.tensor(order)]=torch.arange(len(order))
    target=bank.target_graph;rank=inverse[target]
    reference_forward=net._action(values[target],None if vectors is None else vectors[target],torch.stack(left)[rank],forcing[target],bank.positions-bank.intervals[target,0])
    reference_backward=net._action(values[target],None if vectors is None else vectors[target],torch.stack(right)[rank],forcing[target],bank.intervals[target,1]-bank.positions)
    actual=net.encode_context(bank)
    torch.testing.assert_close(actual.forward,reference_forward,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(actual.backward,reference_backward,rtol=1e-12,atol=1e-12)


def test_scan_validation_counts_do_not_scale_with_hold_count(monkeypatch):
    import arggnn.boundary_memory as memory
    original_decay,original_factors=memory._decay,memory._factors
    counts={"decay":0,"factors":0}
    def decay(*args,**kwargs): counts["decay"]+=1;return original_decay(*args,**kwargs)
    def factors(*args,**kwargs): counts["factors"]+=1;return original_factors(*args,**kwargs)
    monkeypatch.setattr(memory,"_decay",decay);monkeypatch.setattr(memory,"_factors",factors)
    net=model();net(cached(sequence()));first=dict(counts)
    counts.update(decay=0,factors=0)
    net(cached(sequence(refine=True)))
    assert first==counts=={"decay":1,"factors":3}


def test_prepared_hold_action_has_no_tensor_boolean_validation(monkeypatch):
    from arggnn.boundary_memory import _PreparedHoldBank
    bank=cached(sequence());net=model();cache=bank.query_cache;source=net.source_bank(bank)
    decay=cache.eigenvalues[:,:,None]*net.rates+net.damping
    prepared=_PreparedHoldBank.prepare(decay,cache.eigenvectors,source.forcing,bank.intervals[:,1]-bank.intervals[:,0])
    def forbidden(*args,**kwargs): raise AssertionError("tensor boolean inside recurrence")
    monkeypatch.setattr(torch.Tensor,"__bool__",forbidden)
    output=prepared.full_action(0,torch.zeros(2,3,dtype=torch.float64))
    assert output.shape==(2,3)

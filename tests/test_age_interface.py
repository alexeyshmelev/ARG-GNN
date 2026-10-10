"""Contracts for fixed observed age operators, independent of supervision."""
from dataclasses import FrozenInstanceError, replace
import io
import math
from types import SimpleNamespace

import pytest
import torch

from arggnn.age_interface import (
    AgeInterfaceBank, AgeInterfaceCache, age_potential, age_potential_difference,
    observed_age_laplacian, precompute_age_interfaces,
)
from arggnn.kron_interface import (
    kron_reduce, precompute_leaf_interfaces, with_query_reference_cache,
)

torch.set_num_threads(2)


def observed_example(unary=False):
    # The plain bank has 47 of the same 48 observed columns. Column 7 changes
    # after unary insertion and must never enter the persistent leaf features.
    ages = torch.tensor([0.,10.,260.,300.,500.,900.,2800.] + ([180.] if unary else []),dtype=torch.float64)
    edges = torch.tensor([[0,1,2,3,4,5],[4,4,5,5,6,6]],dtype=torch.long)
    if unary:
        edges = torch.tensor([[0,7,1,2,3,4,5],[7,4,4,5,5,6,6]],dtype=torch.long)
    x = torch.zeros(ages.numel(),48,dtype=torch.float64)
    x[2,0],x[3,1] = 1.,1.
    x[:2,4] = 1.
    x[:,6] = torch.log1p(ages)/math.log1p(10000.)
    branch = torch.zeros_like(ages)
    branch[edges[0]] = ages[edges[1]]-ages[edges[0]]
    x[:,7] = torch.log1p(branch)/math.log1p(10000.)
    x[:4,8:] = torch.arange(160,dtype=torch.float64).reshape(4,40)/100.
    edges = torch.cat((edges,edges.flip(0)),dim=1)
    return SimpleNamespace(x=x,edge_index=edges,node_graph=torch.zeros(ages.numel(),dtype=torch.long),
        node_ids=torch.tensor([10,20,30,40,-1,-1,-1]+([-1] if unary else [])),
        clades=torch.zeros(ages.numel(),2,dtype=torch.float64),
        intervals=torch.tensor([[0.,1.]],dtype=torch.float64),
        query_node_ids=torch.tensor([20,10]),positions=torch.tensor([.125,.75],dtype=torch.float64),
        target_graph=torch.zeros(2,dtype=torch.long),window_bounds=torch.tensor([0.,1.],dtype=torch.float64))


def age_bank(seq=None,**kwargs):
    return precompute_age_interfaces(observed_example() if seq is None else seq,reference_ids=torch.tensor([40,30]),**kwargs)


def split_intervals(seq):
    n=seq.x.shape[0]
    return SimpleNamespace(x=seq.x.repeat(3,1),edge_index=torch.cat([seq.edge_index+i*n for i in range(3)],dim=1),
        node_graph=torch.arange(3).repeat_interleave(n),node_ids=seq.node_ids.repeat(3),
        clades=seq.clades.repeat(3,1),intervals=torch.tensor([[0.,.125],[.125,.7],[.7,1.]],dtype=torch.float64),
        query_node_ids=seq.query_node_ids.clone(),positions=seq.positions.clone(),
        target_graph=torch.tensor([1,2]),window_bounds=seq.window_bounds.clone())


@pytest.mark.parametrize('scale',[350.,1400.])
def test_potential_endpoint_identity_telescoping_and_bounded_resistance(scale):
    child=torch.tensor([0.,10.,260.,350.,1e6],dtype=torch.float64)
    parent=child+torch.tensor([1.,200.,330.,1800.,.001],dtype=torch.float64)
    middle=child+.37*(parent-child)
    stable=age_potential_difference(parent,child,scale)
    direct=age_potential(parent,scale)-age_potential(child,scale)
    torch.testing.assert_close(stable,direct,rtol=2e-7,atol=1e-9)
    split=age_potential_difference(parent,middle,scale)+age_potential_difference(middle,child,scale)
    torch.testing.assert_close(stable,split,rtol=2e-13,atol=2e-13)
    duration=parent-child
    assert torch.all(stable>=duration*(1-1e-13))
    assert torch.all(stable<=4*duration*(1+1e-13))
    torch.testing.assert_close(age_potential_difference(parent,child,scale,strength=0),duration,rtol=0,atol=0)


def test_stable_potential_difference_for_close_old_ages_and_scaled_atan2():
    child=torch.tensor([1e12,1e150,0.],dtype=torch.float64)
    parent=torch.tensor([1e12+.125,1e150*(1+1e-12),1e200],dtype=torch.float64)
    difference=age_potential_difference(parent,child,350.)
    torch.testing.assert_close(difference[:2],4*(parent-child)[:2],rtol=1e-14,atol=1e-14)
    assert torch.isfinite(difference).all()
    assert difference[2]==4*parent[2]


@pytest.mark.parametrize('scale',[350.,1400.])
def test_exact_unary_subdivision_preserves_every_raw_response_and_leaf_input(scale):
    before=age_bank(age_scales=(scale,))
    after=age_bank(observed_example(True),age_scales=(scale,))
    assert before.plain_bank.leaf_x.shape==(1,4,47)
    assert 7 not in before.plain_bank.feature_columns
    assert not torch.equal(observed_example().x[0,7],observed_example(True).x[0,7])
    torch.testing.assert_close(before.plain_bank.leaf_x,after.plain_bank.leaf_x,rtol=0,atol=0)
    for name in ('raw_response','normalized_response','cqq','cqr','scalar_cqq','scalar_cqr'):
        torch.testing.assert_close(getattr(before.operator_cache,name),getattr(after.operator_cache,name),rtol=2e-12,atol=2e-12)


def test_node_permutation_and_query_reference_identity_are_preserved():
    seq=observed_example(True)
    permutation=torch.tensor([7,3,1,6,0,2,5,4])
    inverse=torch.empty_like(permutation); inverse[permutation]=torch.arange(len(permutation))
    permuted=SimpleNamespace(**{name:value.clone() for name,value in vars(seq).items()})
    for name in ('x','node_ids','node_graph','clades'):
        setattr(permuted,name,getattr(seq,name)[permutation])
    permuted.edge_index=inverse[seq.edge_index]
    first,second=age_bank(seq),age_bank(permuted)
    torch.testing.assert_close(first.operator_cache.raw_response,second.operator_cache.raw_response,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(second.operator_cache.query_ids,torch.tensor([20,10]),rtol=0,atol=0)
    torch.testing.assert_close(second.operator_cache.reference_ids,torch.tensor([40,30]),rtol=0,atol=0)
    assert second.operator_cache.query_indices.tolist()==[1,0]
    assert second.operator_cache.reference_indices.tolist()==[3,2]
    torch.testing.assert_close(second.plain_bank.leaf_x,first.plain_bank.leaf_x,rtol=0,atol=0)


def test_identical_interval_subdivision_preserves_operator_and_physical_average():
    whole=age_bank()
    split=age_bank(split_intervals(observed_example()))
    cache=split.operator_cache
    torch.testing.assert_close(cache.interval_weights,torch.tensor([.125,.575,.3],dtype=torch.float64),rtol=0,atol=1e-16)
    for name in ('raw_response','normalized_response','cqq','cqr','scalar_cqq','scalar_cqr'):
        original=getattr(whole.operator_cache,name)[0]
        repeated=getattr(cache,name)
        torch.testing.assert_close(repeated,original.expand_as(repeated),rtol=0,atol=0)
        average=(repeated*cache.interval_weights.reshape(3,*([1]*(repeated.ndim-1)))).sum(0)
        torch.testing.assert_close(average,original,rtol=1e-15,atol=1e-15)
    for name in ('positions','query_node_ids','window_bounds'):
        torch.testing.assert_close(getattr(split.plain_bank,name),getattr(whole.plain_bank,name),rtol=0,atol=0)


def test_common_plain_normalization_scalar_mean_match_and_fixed_eigensystems():
    bank=age_bank()
    cache=bank.operator_cache; plain=bank.plain_bank.query_cache
    assert cache.operator_scales==(0.,350.,1400.)
    assert cache.operator_labels==('plain','age350','age1400')
    assert cache.normalization=='shared_plain_mean_boundary_diagonal'
    torch.testing.assert_close(cache.raw_response[:,0],bank.plain_bank.response,rtol=0,atol=0)
    torch.testing.assert_close(cache.normalization_scale,plain.normalization_scale,rtol=0,atol=0)
    torch.testing.assert_close(cache.normalized_response,cache.raw_response/plain.normalization_scale[:,None,None,None],rtol=0,atol=0)
    ratios=cache.normalized_response.diagonal(dim1=-2,dim2=-1).mean(-1)
    torch.testing.assert_close(cache.mean_diagonal_ratios,ratios,rtol=1e-15,atol=0)
    assert (ratios[:,1:]<1).all()
    # Scalar controls have every age operator's full-B mean diagonal and the
    # same plain relative geometry. They are not separately normalized to one.
    scalar_full=plain.normalized_response[:,None]*ratios[:,:,None,None]
    torch.testing.assert_close(scalar_full.diagonal(dim1=-2,dim2=-1).mean(-1),ratios,rtol=1e-15,atol=0)
    torch.testing.assert_close(cache.scalar_cqq,plain.cqq[:,None]*cache.mean_diagonal_ratios[:,:,None,None],rtol=0,atol=0)
    torch.testing.assert_close(cache.scalar_cqr,plain.cqr[:,None]*cache.mean_diagonal_ratios[:,:,None,None],rtol=0,atol=0)
    torch.testing.assert_close(cache.scalar_eigenvectors,plain.eigenvectors[:,None].expand_as(cache.scalar_eigenvectors),rtol=0,atol=0)
    for prefix in ('','scalar_'):
        values=getattr(cache,prefix+'eigenvalues'); vectors=getattr(cache,prefix+'eigenvectors')
        reconstructed=(vectors*values.unsqueeze(-2))@vectors.transpose(-1,-2)
        torch.testing.assert_close(reconstructed,getattr(cache,prefix+'cqq'),rtol=2e-12,atol=2e-12)
        assert torch.isfinite(values).all() and (values>=-1e-12).all()
    assert cache.spectral_reconstruction_relative.max()<1e-12
    assert cache.spectral_orthogonality_max.max()<1e-12
    assert float((cache.cqq[:,1]-cache.scalar_cqq[:,1]).abs().max())>.01
    assert len(cache.diagnostics)==1 and len(cache.diagnostics[0])==3
    raw=cache.raw_response[0]
    for operator in raw[1:]:
        assert torch.linalg.eigvalsh((operator-.25*raw[0])/cache.normalization_scale[0]).min()>=-1e-12
        assert torch.linalg.eigvalsh((raw[0]-operator)/cache.normalization_scale[0]).min()>=-1e-12


@pytest.mark.parametrize('seed',[0,11,29])
def test_loewner_quarter_plain_to_plain_bounds_on_random_observed_stars(seed):
    generator=torch.Generator().manual_seed(seed)
    n=12; ages=torch.cat((torch.rand(n,generator=generator,dtype=torch.float64)*300,torch.tensor([2000.],dtype=torch.float64)))
    edges=torch.stack((torch.arange(n),torch.full((n,),n)))
    boundary=torch.arange(n)
    plain=kron_reduce(observed_age_laplacian(ages,edges,age_scale=350.,strength=0),boundary).response
    for scale in (350.,1400.):
        age=kron_reduce(observed_age_laplacian(ages,edges,age_scale=scale),boundary).response
        sigma=plain.diagonal().mean()
        assert float(torch.linalg.eigvalsh((age-.25*plain)/sigma).min())>=-1e-12
        assert float(torch.linalg.eigvalsh((plain-age)/sigma).min())>=-1e-12


def test_cpu_transfer_metadata_immutability_and_weights_only_serialization():
    bank=age_bank(split_intervals(observed_example()))
    moved=bank.to('cpu',dtype=torch.float32)
    assert moved.plain_bank.leaf_x.dtype==torch.float32
    assert moved.operator_cache.raw_response.dtype==torch.float32
    assert moved.operator_cache.eigenvectors.dtype==torch.float32
    assert moved.operator_cache.query_ids.dtype==torch.long
    for name in ('intervals','positions','window_bounds'):
        assert getattr(moved.plain_bank,name).dtype==torch.float64
        torch.testing.assert_close(getattr(moved.plain_bank,name),getattr(bank.plain_bank,name),rtol=0,atol=0)
    payload=io.BytesIO(); torch.save(bank.as_dict(),payload); payload.seek(0)
    restored=AgeInterfaceBank.from_dict(torch.load(payload,weights_only=True))
    for name in ('raw_response','scalar_cqr','eigenvalues','boundary_nodes'):
        torch.testing.assert_close(getattr(restored.operator_cache,name),getattr(bank.operator_cache,name),rtol=0,atol=0)
    assert restored.operator_cache.diagnostics==bank.operator_cache.diagnostics
    payload=io.BytesIO(); torch.save(bank.operator_cache.as_dict(),payload); payload.seek(0)
    restored_cache=AgeInterfaceCache.from_dict(torch.load(payload,weights_only=True))
    torch.testing.assert_close(restored_cache.cqq,bank.operator_cache.cqq,rtol=0,atol=0)
    with pytest.raises(FrozenInstanceError): bank.operator_cache=None


def test_supplied_plain_bank_reuse_and_source_mismatch_rejection():
    seq=observed_example()
    plain=with_query_reference_cache(precompute_leaf_interfaces(seq,time_column=6,parent_branch_column=7,resistance_scale=1e-4),torch.tensor([40,30]))
    age=age_bank(seq,plain_bank=plain)
    assert age.plain_bank is plain
    # Feature, geometry, order and physical normalization changes are distinct
    # admission failures even when raw tensor shape remains compatible.
    changes=[replace(plain,leaf_x=plain.leaf_x+1),replace(plain,response=plain.response*1.1),
             replace(plain,positions=plain.positions+.01),replace(plain,time_scale=9999.),
             replace(plain,query_cache=replace(plain.query_cache,normalization='none')),
             replace(plain,query_cache=replace(plain.query_cache,eigenvalues=plain.query_cache.eigenvalues*2))]
    for changed in changes:
        with pytest.raises(ValueError): age_bank(seq,plain_bank=changed)
    with pytest.raises(ValueError,match='reference order'):
        precompute_age_interfaces(seq,plain_bank=plain,reference_ids=torch.tensor([30,40]))


def test_s1_plain_zero_geometry_and_no_supervision_dependence():
    seq=observed_example(); single=age_bank(seq,age_scales=())
    assert single.operator_cache.num_scales==1
    torch.testing.assert_close(single.operator_cache.cqq[:,0],single.plain_bank.query_cache.cqq,rtol=0,atol=0)
    seq.labels=torch.full((2,2),1000)
    seq.baseline_probabilities=torch.randn(2,2,4)
    unrelated=age_bank(seq,age_scales=())
    torch.testing.assert_close(single.operator_cache.raw_response,unrelated.operator_cache.raw_response,rtol=0,atol=0)
    # A connected tree with only one sampled boundary has analytically zero
    # response. All operators use the explicit scale-one zero convention.
    seq.x=seq.x[[0,4]].clone(); seq.node_ids=torch.tensor([10,-1]); seq.node_graph=torch.zeros(2,dtype=torch.long)
    seq.edge_index=torch.tensor([[0,1],[1,0]]); seq.clades=torch.zeros(2,2,dtype=torch.float64)
    seq.query_node_ids=torch.tensor([10])
    zero=precompute_age_interfaces(seq)
    torch.testing.assert_close(zero.operator_cache.raw_response,torch.zeros(1,3,1,1,dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(zero.operator_cache.normalization_scale,torch.ones(1,dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(zero.operator_cache.mean_diagonal_ratios,torch.ones(1,3,dtype=torch.float64),rtol=0,atol=0)
    assert zero.operator_cache.cqr.shape==(1,3,1,0)


def test_invalid_physical_inputs_and_schema_are_rejected_without_flooring():
    edge=torch.tensor([[0],[1]])
    for ages in (torch.tensor([0.,0.]),torch.tensor([1.,1.]),torch.tensor([-1.,2.]),torch.tensor([0.,float('inf')])):
        with pytest.raises(ValueError): observed_age_laplacian(ages,edge,age_scale=350.)
    for scale in (0.,-1.,float('nan')):
        with pytest.raises(ValueError): age_potential(torch.ones(1),scale)
    with pytest.raises(ValueError): age_potential_difference(torch.ones(1),torch.tensor([2.]),350.)
    with pytest.raises(ValueError): age_potential(torch.ones(1),350.,strength=-1.)
    with pytest.raises(ValueError): age_bank(age_scales=(350.,350.))
    with pytest.raises(ValueError): age_bank(feature_columns=[0,7])
    with pytest.raises(ValueError): age_bank(resistance_scale=0.)
    bank=age_bank(); cache=bank.operator_cache.as_dict()
    for changed in ({**cache,'schema_version':2},{**cache,'normalization':'per_age_mean_diagonal'},
                    {**cache,'cqq':cache['cqq'][:,:2]},{**cache,'query_ids':torch.tensor([10,20])}):
        with pytest.raises(ValueError): AgeInterfaceCache.from_dict(changed)
    with pytest.raises(ValueError): AgeInterfaceBank.from_dict({'schema_version':1})

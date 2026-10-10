"""Same-source temporal comparators, fixed anchors, and independent readouts."""
from dataclasses import replace
import io
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from arggnn.anchor_kernel import AnchorKernelModel,prepare_anchor_kernel_bank
from arggnn.anchor_temporal_baselines import (
    AnchorGRUModel,AnchorTemporalTransformerModel,ObservedReferenceCoordinateSmoother,
    interpolate_anchor_values,
)
from arggnn.data import GraphSequence
from arggnn.kron_interface import precompute_leaf_interfaces,with_query_reference_cache
from arggnn.reference_encoder import ReferenceFeatureAdapter

torch.set_num_threads(1)


def seq(unary=False,split=False,serialize=False):
    records=[]; intervals=[[0.,.31],[.31,.72],[.72,1.]]
    for graph,delta in enumerate((0.,150.,400.)):
        ages=[0.,10.,260.,300.,250.,250.,500.+delta,900.+delta,1200.+delta,1800.+delta,2800.+delta]
        pairs=[(0,6),(1,6),(2,7),(3,7),(4,8),(5,8),(6,9),(7,9),(9,10),(8,10)]
        if unary and graph==0:
            ages.append(180.); pairs=[p for p in pairs if p!=(0,6)]+[(0,11),(11,6)]
        ages=torch.tensor(ages,dtype=torch.float64)
        x=torch.zeros(len(ages),8,dtype=torch.float64)
        x[torch.arange(2,6),torch.arange(4)]=1.; x[:2,4]=1.; x[:6,5]=1.
        x[:,6]=torch.log1p(ages)/math.log1p(10000.)
        branch=torch.zeros_like(ages)
        for child,parent in pairs: branch[child]=ages[parent]-ages[child]
        x[:,7]=torch.log1p(branch)/math.log1p(10000.)
        edge=torch.tensor(pairs,dtype=torch.long).T
        records.append((x,torch.cat((edge,edge.flip(0)),1),torch.tensor([10,20,30,40,50,60]+[-1]*(len(ages)-6))))
    if split:
        records=[records[0],records[0],*records[1:]]
        intervals=[[0.,.17],[.17,.31],[.31,.72],[.72,1.]]
    if serialize:
        records=list(reversed(records)); intervals=list(reversed(intervals))
    xs=[]; edges=[]; ids=[]; graphs=[]; offset=0
    for graph,(x,edge,identity) in enumerate(records):
        xs.append(x); edges.append(edge+offset);ids.append(identity);graphs.extend([graph]*len(x));offset+=len(x)
    positions=torch.tensor([0.,.08,.22,.31,.57,.9,.999],dtype=torch.float64)
    intervals=torch.tensor(intervals,dtype=torch.float64)
    target=((positions[:,None]>=intervals[None,:,0])&(positions[:,None]<intervals[None,:,1])).long().argmax(1)
    raw=GraphSequence(torch.cat(xs),torch.cat(edges,1),torch.tensor(graphs),torch.cat(ids),
        torch.zeros(offset,1,dtype=torch.float64),intervals,torch.tensor([20,10]),positions,target,
        torch.tensor([0.,1.],dtype=torch.float64))
    return ReferenceFeatureAdapter(8,reference_columns=(0,1,2,3),time_column=6,branch_column=7).precompute(raw)


def prepared(sequence=None,anchors=5):
    sequence=seq() if sequence is None else sequence
    bank=precompute_leaf_interfaces(sequence,time_column=6,parent_branch_column=7,resistance_scale=1e-4)
    bank=with_query_reference_cache(bank,torch.tensor([30,40,50,60]))
    return prepare_anchor_kernel_bank(bank,num_anchors=anchors,num_scales=3)


def model(family='gru',anchors=5,dropout=0.):
    torch.manual_seed(918)
    cls=AnchorGRUModel if family=='gru' else AnchorTemporalTransformerModel
    return cls(hidden_dim=8,latent_dim=3,num_anchors=anchors,transformer_heads=2,dropout=dropout).double()


def readouts(pack,positions):
    intervals=pack.bank.intervals
    mapping=((positions[:,None]>=intervals[None,:,0])&(positions[:,None]<intervals[None,:,1])).long().argmax(1)
    bank=replace(pack.bank,positions=positions,target_graph=mapping)
    return prepare_anchor_kernel_bank(bank,num_anchors=len(pack.layout.anchors),num_scales=3)


@pytest.mark.parametrize('family',['gru','transformer'])
def test_same_plain_encoder_and_exact_first_candidate_point_sources(family):
    pack=prepared()
    baseline=model(family)
    candidate=AnchorKernelModel(47,4,hidden_dim=8,latent_dim=3,num_anchors=5,num_scales=3,
        initial_coupling=[5.,20.,80.]).double()
    baseline.encoder.load_state_dict(candidate.encoder.state_dict())
    for name in ('query_projection','reference_projection','message_projection','gate','norm'):
        getattr(baseline.layers[0],name).load_state_dict(getattr(candidate.layers[0],name).state_dict())
    baseline.layers[0].raw_coupling.data.copy_(candidate.layers[0].raw_coupling.data)
    baseline.layers[0].mixture_logits.data.copy_(candidate.layers[0].mixture_logits.data)
    h=candidate.encoder(pack.bank.leaf_x,pack.bank.query_cache.normalized_response)
    q=h[:,pack.bank.query_cache.query_indices][pack.layout.anchor_graph]
    r=h[:,pack.bank.query_cache.reference_indices][pack.layout.anchor_graph]
    expected=candidate.layers[0].messages(pack,q,r,'plain','full')
    actual=baseline.layers[0].conditioned_sources(pack,q,r)
    torch.testing.assert_close(actual.sources,expected.sources,rtol=0,atol=0)
    torch.testing.assert_close(actual.mass_sources,expected.mass_sources,rtol=0,atol=0)
    assert actual.normalized.abs().max()<=1+1e-12


@pytest.mark.parametrize('family',['gru','transformer'])
def test_forward_shape_all_parameter_gradients_and_feature_reachability(family):
    pack=prepared(); m=model(family)
    features=pack.bank.leaf_x.clone().requires_grad_()
    pack=replace(pack,bank=replace(pack.bank,leaf_x=features))
    output=m(pack)
    assert output.shape==(7,2,4)
    output.square().mean().backward()
    for name,parameter in m.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all(),name
        assert parameter.grad.abs().max()>0,name
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad[:,pack.bank.query_cache.query_indices].abs().max()>0
    assert features.grad[:,pack.bank.query_cache.reference_indices].abs().max()>0


@pytest.mark.parametrize('family',['gru','transformer'])
def test_readout_addition_duplication_permutation_never_changes_sources_or_existing_predictions(family):
    pack=prepared(); m=model(family).eval()
    original=m(pack)
    positions=torch.cat((pack.bank.positions,torch.tensor([.15,.42,.73,.22],dtype=torch.float64)))
    expanded=readouts(pack,positions)
    torch.testing.assert_close(m(expanded)[:7],original,rtol=2e-12,atol=2e-12)
    a,b=m.encode_anchors(pack),m.encode_anchors(expanded)
    torch.testing.assert_close(a.anchors,b.anchors,rtol=0,atol=0)
    for first,second in zip(a.layers,b.layers):
        torch.testing.assert_close(first[0].sources,second[0].sources,rtol=0,atol=0)
        torch.testing.assert_close(first[1],second[1],rtol=0,atol=0)
    permutation=torch.tensor([6,1,3,0,5,2,4])
    reordered=readouts(pack,pack.bank.positions[permutation])
    torch.testing.assert_close(m(reordered),original[permutation],rtol=2e-12,atol=2e-12)
    duplicate=readouts(pack,pack.bank.positions[[2,2,2]])
    torch.testing.assert_close(m(duplicate),original[2].expand(3,-1,-1),rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('family',['gru','transformer'])
def test_training_anchor_dropout_bank_independent_of_readout_set(family):
    pack=prepared(); changed=readouts(pack,torch.tensor([.03,.2,.4,.6,.8,.9],dtype=torch.float64))
    m=model(family,dropout=.1).train()
    torch.manual_seed(2); original=m.encode_anchors(pack)
    torch.manual_seed(2); expanded=m.encode_anchors(changed)
    torch.testing.assert_close(original.anchors,expanded.anchors,rtol=0,atol=0)
    for first,second in zip(original.layers,expanded.layers):
        torch.testing.assert_close(first[0].sources,second[0].sources,rtol=0,atol=0)


@pytest.mark.parametrize('family',['gru','transformer'])
def test_unary_interval_and_serialization_contract(family):
    m=model(family).eval()
    base=m(prepared())
    for sequence in (seq(unary=True),seq(split=True),seq(serialize=True),seq(unary=True,split=True,serialize=True)):
        torch.testing.assert_close(m(prepared(sequence)),base,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize('family',['gru','transformer'])
def test_query_permutation_stable_ID_rename_and_boundary_storage_alignment(family):
    pack=prepared(); m=model(family).eval(); baseline=m(pack)
    original=seq()
    original.query_node_ids=original.query_node_ids.flip(0)
    changed=prepared(original)
    torch.testing.assert_close(m(changed),baseline.flip(1),rtol=3e-12,atol=3e-12)
    # Numerical leaf IDs remain correspondence keys and are never embedded.
    original=seq(); before_ids=original.node_ids.clone(); mapping={10:1010,20:220,30:130,40:140,50:150,60:160}
    for old,new in mapping.items(): original.node_ids[before_ids==old]=new
    original.query_node_ids=torch.tensor([mapping[int(q)] for q in original.query_node_ids])
    renamed=precompute_leaf_interfaces(original,time_column=6,parent_branch_column=7,resistance_scale=1e-4)
    renamed=with_query_reference_cache(renamed,torch.tensor([mapping[q] for q in (30,40,50,60)]))
    renamed=prepare_anchor_kernel_bank(renamed,num_anchors=5,num_scales=3)
    torch.testing.assert_close(m(renamed),baseline,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize('family',['gru','transformer'])
def test_external_annotations_unused_and_reference_input_counterfactual_matters(family):
    pack=prepared(); m=model(family).eval(); expected=m(pack)
    class Guard:
        def __init__(self,base): self.base=base
        def __getattr__(self,name):
            if name in ('labels','baseline_probabilities','population','truth','metadata'):
                raise AssertionError('Forbidden annotation access')
            return getattr(self.base,name)
    guarded=SimpleNamespace(bank=Guard(pack.bank),layout=pack.layout,operators=pack.operators,num_scales=pack.num_scales)
    torch.testing.assert_close(m(guarded),expected,rtol=0,atol=0)
    leaf_x=pack.bank.leaf_x.clone()
    leaf_x[:,pack.bank.query_cache.reference_indices,0]+=1
    altered=replace(pack,bank=replace(pack.bank,leaf_x=leaf_x))
    assert (m(altered)-expected).abs().max()>1e-8


@pytest.mark.parametrize('family',['gru','transformer'])
def test_no_online_matrix_solve_or_eigendecompose_and_state_dict_roundtrip(family):
    pack=prepared(); m=model(family).eval()
    with patch('torch.linalg.eigh',side_effect=AssertionError),patch('torch.linalg.solve',side_effect=AssertionError),patch('torch.linalg.inv',side_effect=AssertionError),patch('torch.matrix_exp',side_effect=AssertionError):
        output=m(pack)
    payload=io.BytesIO();torch.save(m.state_dict(),payload);payload.seek(0)
    second=model(family).eval();second.load_state_dict(torch.load(payload,weights_only=True))
    torch.testing.assert_close(second(pack),output,rtol=0,atol=0)
    manifest=m.configuration_manifest(pack)
    assert manifest['labels_in_input'] is False and manifest['numeric_ID_features'] is False
    assert manifest['parameters']==sum(p.numel() for p in m.parameters())


def test_physical_interpolation_known_values_endpoint_extrapolation():
    anchors=torch.tensor([.2,.5,.8],dtype=torch.float64)
    values=torch.tensor([2.,5.,8.],dtype=torch.float64)[:,None,None]
    points=torch.tensor([0.,.2,.35,.5,.65,.8,1.],dtype=torch.float64)
    expected=torch.tensor([2.,2.,3.5,5.,6.5,8.,8.],dtype=torch.float64)[:,None,None]
    torch.testing.assert_close(interpolate_anchor_values(anchors,values,points),expected,rtol=1e-15,atol=1e-15)


def test_reference_smoother_known_coordinate_units_readout_independence_and_no_fitting():
    pack=prepared()
    m=ObservedReferenceCoordinateSmoother(num_anchors=5,bandwidth_bp=30000.).eval()
    output=m(pack)
    assert output.shape==(7,2,4) and torch.isfinite(output).all()
    torch.testing.assert_close(output.exp().sum(-1),torch.ones(7,2,dtype=torch.float64),rtol=1e-12,atol=1e-12)
    assert not list(m.parameters())
    points=torch.cat((pack.bank.positions,torch.tensor([.12,.7],dtype=torch.float64)))
    torch.testing.assert_close(m(readouts(pack,points))[:7],output,rtol=1e-15,atol=1e-15)
    evidence=m.anchor_evidence(pack)
    x,y=pack.bank.positions,pack.layout.anchors
    weights=(-((x[:,None]-y[None]).abs()/.03)+pack.layout.weights.log()[None]).softmax(-1)
    score=torch.einsum('pa,aqk->pqk',weights,evidence)+.001
    expected=score.log()-score.sum(-1,keepdim=True).log()
    torch.testing.assert_close(output,expected,rtol=0,atol=0)
    assert m.configuration_manifest()['source_encoder'].startswith('none')


@pytest.mark.parametrize('family',['gru','transformer'])
def test_invalid_schema_feature_dtype_and_nonfinite_rates_rejected(family):
    pack=prepared(); m=model(family)
    with pytest.raises(ValueError): m(replace(pack,num_scales=1))
    with pytest.raises(ValueError): m(replace(pack,bank=pack.bank.to('cpu',torch.float32)))
    m.layers[0].raw_coupling.data.fill_(10000.)
    with pytest.raises(ValueError): m(pack)


def test_constructor_and_smoother_observed_role_rejections():
    with pytest.raises(ValueError): AnchorTemporalTransformerModel(hidden_dim=31,transformer_heads=4)
    with pytest.raises(ValueError): AnchorGRUModel(dropout=1.)
    with pytest.raises(ValueError): ObservedReferenceCoordinateSmoother(bandwidth_bp=0.)
    pack=prepared(); features=pack.bank.leaf_x.clone();features[:,pack.bank.query_cache.reference_indices,0]=.5
    with pytest.raises(ValueError): ObservedReferenceCoordinateSmoother(num_anchors=5)(replace(pack,bank=replace(pack.bank,leaf_x=features)))


def test_transformer_joint_attention_score_bound_rejects_before_large_allocation():
    pack=prepared()
    m=AnchorTemporalTransformerModel(hidden_dim=8,latent_dim=3,num_anchors=5,transformer_heads=2,max_attention_score_bytes=1).double()
    with pytest.raises(ValueError,match='dense-score estimate'): m(pack)

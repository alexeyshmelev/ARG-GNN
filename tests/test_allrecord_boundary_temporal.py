"""Synthetic all-hold reachability and independent fixed-coordinate readouts."""
from dataclasses import replace
import io
from unittest.mock import patch

import pytest
import torch

from test_anchor_temporal_baselines import seq
from arggnn.allrecord_boundary_temporal import (
    AllRecordBoundaryGRU,AllRecordBoundaryTransformer,interpolate_record_states,
    FEATURE_COLUMNS,INPUT_SCOPE,AGGREGATION,interval_features,
)
from arggnn.boundary_memory import BoundaryResponseGNN
from arggnn.kron_interface import precompute_leaf_interfaces,with_query_reference_cache,LeafInterfaceBank

torch.set_num_threads(1)
FAMILIES=('allrecord_gru','allrecord_transformer')


def bank(sequence=None):
    sequence=seq() if sequence is None else sequence
    value=precompute_leaf_interfaces(sequence,time_column=6,parent_branch_column=7,resistance_scale=1e-4)
    return with_query_reference_cache(value,torch.tensor([30,40,50,60]))


def model(family,dropout=0.):
    torch.manual_seed(701)
    cls=AllRecordBoundaryGRU if family=='allrecord_gru' else AllRecordBoundaryTransformer
    return cls(hidden_dim=8,transformer_heads=2,dropout=dropout).double()


def readouts(value,points):
    mapping=((points[:,None]>=value.intervals[None,:,0])&(points[:,None]<value.intervals[None,:,1])).long().argmax(1)
    return replace(value,positions=points,target_graph=mapping)


@pytest.mark.parametrize('family',FAMILIES)
def test_same_depth2_encoder_all_record_tokens_and_finite_parameter_gradients(family):
    value=bank();net=model(family);features=value.leaf_x.clone().requires_grad_();value=replace(value,leaf_x=features)
    expected=BoundaryResponseGNN(47,8,2).double();expected.load_state_dict(net.encoder.state_dict())
    source=net.encode_records(value)
    torch.testing.assert_close(source.local_query,expected(features,value.query_cache.normalized_response)[:,value.query_cache.query_indices],rtol=0,atol=0)
    assert source.tokens.shape==(3,2,8) and source.memory.shape[0]==len(value.intervals)
    output=net(value);assert output.shape==(7,2,4) and torch.isfinite(output).all()
    output.square().mean().backward()
    for name,parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
        assert parameter.grad.abs().max()>0,name
    assert torch.isfinite(features.grad).all()
    for hold in range(3):assert features.grad[hold].abs().max()>0
    assert features.grad[:,value.query_cache.reference_indices].abs().max()>0
    manifest=net.configuration_manifest(value)
    assert manifest['source_GNN_depth']==2 and manifest['temporal_depth']==2
    assert manifest['input_scope']==INPUT_SCOPE and manifest['cross_tree_aggregation']==AGGREGATION[family]
    assert manifest['parameters']==sum(p.numel() for p in net.parameters())
    assert manifest['observed_records']==3 and manifest['readout_created_source_tokens'] is False
    if family=='allrecord_gru':assert net.temporal.num_layers==2
    else:assert len(net.temporal.layers)==2


@pytest.mark.parametrize('family',FAMILIES)
def test_intervening_hold_without_target_is_a_live_temporal_source(family):
    original=readouts(bank(),torch.tensor([.04,.96],dtype=torch.float64));net=model(family).eval()
    assert set(original.target_graph.tolist())=={0,2}
    features=original.leaf_x.clone().requires_grad_();value=replace(original,leaf_x=features)
    output=net(value);output[0].square().sum().backward()
    assert features.grad[1].abs().max()>1e-10
    changed=features.detach().clone();changed[1,value.query_cache.query_indices,8]+=.4
    altered=replace(original,leaf_x=changed)
    assert (net(altered)-net(original)).abs().max()>1e-7
    torch.testing.assert_close(net.encode_records(original).local_query[[0,2]],net.encode_records(altered).local_query[[0,2]],rtol=0,atol=0)


@pytest.mark.parametrize('family',FAMILIES)
def test_every_source_hold_can_change_fixed_predictions(family):
    value=readouts(bank(),torch.tensor([.04,.96],dtype=torch.float64));net=model(family).eval();expected=net(value)
    for hold in range(3):
        features=value.leaf_x.clone();features[hold,value.query_cache.query_indices,8]+=.5
        altered=replace(value,leaf_x=features)
        assert (net(altered)-expected).abs().max()>1e-8,hold


@pytest.mark.parametrize('family',FAMILIES)
def test_added_reordered_and_duplicate_readouts_leave_existing_predictions_fixed(family):
    value=bank();net=model(family).eval();expected=net(value)
    expanded=readouts(value,torch.cat((value.positions,torch.tensor([.19,.5,.79,.22],dtype=torch.float64))))
    torch.testing.assert_close(net(expanded)[:len(value.positions)],expected,rtol=3e-12,atol=3e-12)
    original_context,expanded_context=net.encode_records(value),net.encode_records(expanded)
    torch.testing.assert_close(original_context.tokens,expanded_context.tokens,rtol=0,atol=0)
    torch.testing.assert_close(original_context.memory,expanded_context.memory,rtol=0,atol=0)
    permutation=torch.tensor([6,3,0,4,1,5,2])
    torch.testing.assert_close(net(readouts(value,value.positions[permutation])),expected[permutation],rtol=3e-12,atol=3e-12)
    torch.testing.assert_close(net(readouts(value,value.positions[[2,2,2]])),expected[2].expand(3,-1,-1),rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize('family',FAMILIES)
def test_training_dropout_source_bank_and_readout_outputs_ignore_readout_set(family):
    value=bank();expanded=readouts(value,torch.cat((value.positions,torch.tensor([.1,.45],dtype=torch.float64))))
    net=model(family,.1).train()
    torch.manual_seed(93);a=net.encode_records(value)
    torch.manual_seed(93);b=net.encode_records(expanded)
    torch.testing.assert_close(a.tokens,b.tokens,rtol=0,atol=0);torch.testing.assert_close(a.memory,b.memory,rtol=0,atol=0)
    torch.testing.assert_close(net.decode(value,a),net.decode(expanded,b)[:7],rtol=3e-12,atol=3e-12)
    class SourceOnly:
        def __getattr__(self,name):
            if name in ('positions','target_graph'):raise AssertionError('Readout-derived source access')
            return getattr(value,name)
    torch.manual_seed(93);guarded=net.encode_records(SourceOnly())
    torch.testing.assert_close(guarded.memory,a.memory,rtol=0,atol=0)


@pytest.mark.parametrize('family',FAMILIES)
def test_query_identity_numeric_rename_serialization_and_unary_contract(family):
    value=bank();net=model(family).eval();expected=net(value)
    changed=seq();changed.query_node_ids=changed.query_node_ids.flip(0)
    torch.testing.assert_close(net(bank(changed)),expected.flip(1),rtol=3e-12,atol=3e-12)
    torch.testing.assert_close(net(bank(seq(serialize=True))),expected,rtol=3e-12,atol=3e-12)
    torch.testing.assert_close(net(bank(seq(unary=True))),expected,rtol=3e-12,atol=3e-12)
    raw=seq();before=raw.node_ids.clone();mapping={10:510,20:220,30:730,40:940,50:150,60:860}
    for old,new in mapping.items():raw.node_ids[before==old]=new
    raw.query_node_ids=torch.tensor([mapping[int(q)] for q in raw.query_node_ids])
    renamed=precompute_leaf_interfaces(raw,time_column=6,parent_branch_column=7,resistance_scale=1e-4)
    renamed=with_query_reference_cache(renamed,torch.tensor([mapping[r] for r in (30,40,50,60)]))
    torch.testing.assert_close(net(renamed),expected,rtol=3e-12,atol=3e-12)
    assert not any(isinstance(layer,torch.nn.Embedding) for layer in net.modules())


@pytest.mark.parametrize('family',FAMILIES)
def test_standard_record_subdivision_is_explicitly_not_an_invariance_claim(family):
    net=model(family).eval()
    assert (net(bank(seq(split=True)))-net(bank())).abs().max()>1e-8
    assert net.configuration_manifest()['record_subdivision_invariant'] is False


@pytest.mark.parametrize('family',FAMILIES)
def test_source_label_counterfactual_and_observed_reference_roles(family):
    value=bank();net=model(family).eval();expected=net(value)
    class AnnotationGuard:
        def __getattr__(self,name):
            if name in ('labels','truth','population','metadata','baseline_probabilities'):
                raise AssertionError('Forbidden annotation access')
            return getattr(value,name)
    torch.testing.assert_close(net(AnnotationGuard()),expected,rtol=0,atol=0)
    features=value.leaf_x.clone();reference=value.query_cache.reference_indices[0]
    features[:,reference,:4]=torch.tensor([0.,1.,0.,0.],dtype=torch.float64)
    altered=replace(value,leaf_x=features)
    assert (net(altered)-expected).abs().max()>1e-8
    context=net.encode_records(value)
    assert (context.reference_role_evidence.sum(-1)<=1+1e-12).all()
    assert context.reference_role_evidence.min()>=0


@pytest.mark.parametrize('family',FAMILIES)
def test_weights_and_bank_serialization_and_no_online_matrix_operations(family):
    value=bank();net=model(family).eval()
    with patch('torch.linalg.eigh',side_effect=AssertionError),patch('torch.linalg.solve',side_effect=AssertionError),patch('torch.linalg.inv',side_effect=AssertionError),patch('torch.matrix_exp',side_effect=AssertionError):
        expected=net(value)
    stream=io.BytesIO();torch.save({'model':net.state_dict(),'bank':value.as_dict()},stream);stream.seek(0)
    payload=torch.load(stream,weights_only=True)
    restored=model(family).eval();restored.load_state_dict(payload['model'])
    restored_bank=LeafInterfaceBank.from_dict(payload['bank'])
    torch.testing.assert_close(restored(restored_bank),expected,rtol=0,atol=0)


@pytest.mark.parametrize('family',FAMILIES)
def test_invalid_source_and_target_contracts_rejected(family):
    value=bank();net=model(family)
    with pytest.raises(ValueError):net(replace(value,feature_columns=tuple(range(47))))
    with pytest.raises(ValueError):net(value.to('cpu',torch.float32))
    features=value.leaf_x.clone();features[1,0,8]=float('nan')
    with pytest.raises(ValueError):net(replace(value,leaf_x=features))
    features=value.leaf_x.clone();features[:,value.query_cache.reference_indices[0],0]=.5
    with pytest.raises(ValueError,match='one-hot'):net(replace(value,leaf_x=features))
    with pytest.raises(ValueError,match='declared observed hold'):net(replace(value,target_graph=torch.zeros_like(value.target_graph)))
    intervals=value.intervals.clone();intervals[1,0]=.1
    with pytest.raises(ValueError,match='nonoverlapping'):net(replace(value,intervals=intervals))


def test_attention_bound_and_constructor_dimensions():
    with pytest.raises(ValueError):AllRecordBoundaryGRU(temporal_depth=1)
    with pytest.raises(ValueError):AllRecordBoundaryGRU(source_depth=1)
    with pytest.raises(ValueError):AllRecordBoundaryGRU(input_dim=48)
    with pytest.raises(ValueError):AllRecordBoundaryGRU(dropout=True)
    with pytest.raises(ValueError):AllRecordBoundaryTransformer(hidden_dim=31)
    with pytest.raises(ValueError,match='dense-score estimate'):
        AllRecordBoundaryTransformer(hidden_dim=8,transformer_heads=2,max_attention_score_bytes=1).double()(bank())


def test_physical_record_interpolation_known_values():
    points=torch.tensor([.1,.4,.9],dtype=torch.float64)
    values=torch.tensor([1.,4.,9.],dtype=torch.float64)[:,None,None]
    readout=torch.tensor([0.,.1,.25,.4,.65,.9,1.],dtype=torch.float64)
    expected=torch.tensor([1.,1.,2.5,4.,6.5,9.,9.],dtype=torch.float64)[:,None,None]
    torch.testing.assert_close(interpolate_record_states(points,values,readout),expected,rtol=1e-15,atol=1e-15)


def test_interval_features_keep_global_coordinates_and_form_width_before_float32_cast():
    intervals=torch.tensor([[.5,.625],[.625,.8]],dtype=torch.float64)
    expected=torch.tensor([[.5,.625,.5625,.125],[.625,.8,.7125,.175]],dtype=torch.float64)
    torch.testing.assert_close(interval_features(intervals,torch.float64),expected,rtol=0,atol=2e-16)
    # Subtracting after a low-precision cast would erase this positive width.
    large=torch.tensor([[100000000.,100000000.001]],dtype=torch.float64)
    assert float(interval_features(large,torch.float32)[0,3])>0
    assert float((large.float()[0,1]-large.float()[0,0]))==0


@pytest.mark.parametrize('family',FAMILIES)
def test_intervening_plain_operator_counterfactual_changes_temporal_predictions(family):
    original=readouts(bank(),torch.tensor([.04,.96],dtype=torch.float64));net=model(family).eval()
    cache=original.query_cache;raw=original.response.clone();hold=1
    incidence=torch.zeros(raw.shape[-1],dtype=raw.dtype)
    incidence[cache.query_indices[0]]=1.;incidence[cache.reference_indices[0]]=-1.
    raw[hold]+=cache.normalization_scale[hold]*.1*torch.outer(incidence,incidence)
    scale=raw.diagonal(dim1=-2,dim2=-1).mean(-1);normal=raw/scale[:,None,None]
    cqq=normal[:,cache.query_indices][:,:,cache.query_indices]
    cqr=normal[:,cache.query_indices][:,:,cache.reference_indices]
    # Preparation may solve cached bases. Model forward performs no such solve.
    eigenvalues,eigenvectors=torch.linalg.eigh(cqq)
    altered=replace(original,response=raw,query_cache=replace(cache,normalized_response=normal,
        normalization_scale=scale,cqq=cqq,cqr=cqr,eigenvalues=eigenvalues,eigenvectors=eigenvectors))
    assert (net(altered)-net(original)).abs().max()>1e-8
    torch.testing.assert_close(altered.leaf_x,original.leaf_x,rtol=0,atol=0)

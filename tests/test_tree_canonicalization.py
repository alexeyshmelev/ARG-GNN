"""Observed-only canonicalization and ordinary-model comparator contracts."""
from dataclasses import replace
import io
import math

import pytest
import torch

from arggnn.data import GraphSequence
from arggnn.models import IndependentGNN, GNNGRU
from arggnn.reference_encoder import ReferenceFeatureAdapter
from arggnn.tree_canonicalization import CanonicalizedObservedTrees, canonicalize_observed_trees

torch.set_num_threads(2)


def example(interior=False,root=False,dtype=torch.float64):
    ages=[0.,10.,260.,300.,250.,250.,500.,900.,1200.,1800.,2800.]
    pairs=[(0,6),(1,6),(2,7),(3,7),(4,8),(5,8),(6,9),(7,9),(9,10),(8,10)]
    if interior:
        u=len(ages); ages.append(180.)
        pairs=[pair for pair in pairs if pair!=(0,6)]+[(0,u),(u,6)]
    if root:
        for time in (3200.,4000.):
            oldroot=max(range(len(ages)),key=lambda u:ages[u])
            u=len(ages); ages.append(time); pairs.append((oldroot,u))
    ages=torch.tensor(ages,dtype=torch.float64)
    x=torch.zeros(len(ages),8,dtype=dtype)
    x[torch.arange(2,6),torch.arange(4)]=1
    x[:2,4]=1; x[:6,5]=1
    x[:,6]=(torch.log1p(ages)/math.log1p(10000.)).to(dtype)
    branch=torch.zeros_like(ages)
    for child,parent in pairs: branch[child]=ages[parent]-ages[child]
    x[:,7]=(torch.log1p(branch)/math.log1p(10000.)).to(dtype)
    edges=torch.tensor(pairs,dtype=torch.long).T.contiguous()
    clades=torch.zeros(len(ages),4,dtype=dtype)
    clades[torch.arange(2,6),torch.arange(4)]=1
    for u in torch.argsort(ages).tolist():
        for child,parent in pairs:
            if child==u: clades[parent]=torch.maximum(clades[parent],clades[child])
    return GraphSequence(x=x,edge_index=torch.cat((edges,edges.flip(0)),dim=1),
        node_graph=torch.zeros(len(ages),dtype=torch.long),
        node_ids=torch.tensor([10,20,30,40,50,60]+[-1]*(len(ages)-6)),clades=clades,
        intervals=torch.tensor([[0.,1.]],dtype=torch.float64),query_node_ids=torch.tensor([20,10]),
        positions=torch.tensor([.05,.2,.8],dtype=torch.float64),target_graph=torch.zeros(3,dtype=torch.long),
        window_bounds=torch.tensor([0.,1.],dtype=torch.float64))


def clone(seq):
    return GraphSequence(**{name:None if value is None else value.clone() for name,value in seq.as_dict().items()})


def duplicate_intervals(seq):
    n=seq.x.shape[0]
    return GraphSequence(x=seq.x.repeat(3,1),edge_index=torch.cat([seq.edge_index+i*n for i in range(3)],dim=1),
        node_graph=torch.arange(3).repeat_interleave(n),node_ids=seq.node_ids.repeat(3),clades=seq.clades.repeat(3,1),
        intervals=torch.tensor([[0.,.1],[.1,.5],[.5,1.]],dtype=torch.float64),
        query_node_ids=seq.query_node_ids.clone(),positions=seq.positions.clone(),target_graph=torch.tensor([0,1,2]),
        window_bounds=seq.window_bounds.clone())


def permute_nodes(seq,seed=13):
    generator=torch.Generator().manual_seed(seed)
    permutation=torch.randperm(len(seq.x),generator=generator)
    inverse=torch.empty_like(permutation); inverse[permutation]=torch.arange(len(permutation))
    result=clone(seq)
    for name in ('x','node_ids','node_graph','clades'): setattr(result,name,getattr(seq,name)[permutation])
    result.edge_index=inverse[seq.edge_index]
    return result


def assert_same_sequence(first,second):
    for name in first.as_dict():
        a,b=getattr(first,name),getattr(second,name)
        if a is None or b is None: assert a is b
        else: torch.testing.assert_close(a,b,rtol=0,atol=0)


def adapter(seq):
    return ReferenceFeatureAdapter(8,reference_columns=(0,1,2,3),time_column=6,branch_column=7).precompute(seq)


@pytest.mark.parametrize('interior,root',[(True,False),(False,True),(True,True)])
def test_unary_interior_and_dangling_root_chains_canonicalize_exactly(interior,root):
    original=canonicalize_observed_trees(example())
    expanded=canonicalize_observed_trees(example(interior,root))
    assert_same_sequence(original.sequence,expanded.sequence)
    assert expanded.provenance['removed_unary_interior_nodes']==int(interior)
    assert expanded.provenance['removed_unary_root_nodes']==2*int(root)
    assert expanded.provenance['truth_fields_used']==[]
    assert expanded.sequence.x.shape==(11,8)
    assert 'evidence beyond invariance' in expanded.provenance['operator_note']
    assert_same_sequence(adapter(original.sequence),adapter(expanded.sequence))
    # Existing sampled features except branch remain intact, and requested query
    # order does not inherit the canonical sorted-leaf row ordering.
    for sample in range(10,61,10):
        before=example(interior,root); row=(before.node_ids==sample).nonzero().item()
        out=(expanded.sequence.node_ids==sample).nonzero().item()
        torch.testing.assert_close(expanded.sequence.x[out,:7],before.x[row,:7],rtol=0,atol=0)
    assert expanded.sequence.query_node_ids.tolist()==[20,10]


def test_branch_recomputed_from_decoded_endpoint_ages_and_stale_summaries_dropped():
    seq=example(True,True,dtype=torch.float32)
    seq.x[:,7]=.999  # finite stale branch fields are replaced, including root
    seq.x=torch.cat((seq.x,torch.randn(len(seq.x),40,dtype=seq.x.dtype)),dim=1)
    result=canonicalize_observed_trees(seq)
    assert result.sequence.x.shape[1]==8
    assert result.provenance['discarded_appended_feature_count']==40
    ages=torch.expm1(result.sequence.x[:,6].double()*math.log1p(10000.))
    parent={}
    for u,v in result.sequence.edge_index.T.tolist():
        if ages[v]>ages[u]: parent[u]=v
    for u in range(len(result.sequence.x)):
        expected=0. if u not in parent else float(torch.log1p(ages[parent[u]]-ages[u])/math.log1p(10000.))
        assert result.sequence.x[u,7].item()==torch.tensor(expected,dtype=torch.float32).item()
    assert adapter(result.sequence).x.shape[1]==48
    assert result.sequence.intervals.dtype==torch.float64
    torch.testing.assert_close(result.sequence.positions,seq.positions,rtol=0,atol=0)


def test_anonymous_row_relabeling_edge_order_and_direction_serialize_identically():
    seq=example(True,True)
    changed=permute_nodes(seq)
    changed.edge_index=changed.edge_index[:,torch.arange(changed.edge_index.shape[1]-1,-1,-1)]
    # Anonymous IDs are deliberately the shared sentinel -1. Any external node
    # table numbering is expressed only by a permutation of packed rows.
    assert (changed.node_ids[changed.node_ids<0]==-1).all()
    before=canonicalize_observed_trees(seq)
    after=canonicalize_observed_trees(changed)
    assert_same_sequence(before.sequence,after.sequence)
    assert before.sequence.node_ids[:6].tolist()==[10,20,30,40,50,60]
    assert before.provenance['learned_numeric_id_features'] is False


def test_interval_duplication_optional_exact_merge_preserves_scoring_mapping():
    original=canonicalize_observed_trees(example(),merge_identical_adjacent=True)
    duplicated=duplicate_intervals(example(True,True))
    separate=canonicalize_observed_trees(duplicated)
    assert separate.sequence.num_graphs==3
    assert separate.source_graph_indices==((0,),(1,),(2,))
    assert separate.source_graph_to_canonical.tolist()==[0,1,2]
    merged=canonicalize_observed_trees(duplicated,merge_identical_adjacent=True)
    assert_same_sequence(original.sequence,merged.sequence)
    assert merged.source_graph_indices==((0,1,2),)
    assert merged.source_graph_to_canonical.tolist()==[0,0,0]
    assert merged.provenance['merged_duplicate_graph_records']==2
    assert merged.sequence.target_graph.tolist()==[0,0,0]
    assert_same_sequence(adapter(original.sequence),adapter(merged.sequence))


def test_chronological_graph_reordering_and_clipped_coordinate_precision():
    seq=duplicate_intervals(example())
    seq.intervals=torch.tensor([[.2,.3],[.3,.6],[.6,.9]],dtype=torch.float64)
    seq.window_bounds=torch.tensor([.2,.9],dtype=torch.float64)
    seq.positions=torch.tensor([.21,.55,.899],dtype=torch.float64)
    # Change serialized graph IDs without changing physical trees.
    perm=torch.tensor([2,0,1]); inv=torch.empty_like(perm); inv[perm]=torch.arange(3)
    seq.intervals=seq.intervals[perm]
    seq.node_graph=inv[seq.node_graph]; seq.target_graph=inv[seq.target_graph]
    separate=canonicalize_observed_trees(seq)
    torch.testing.assert_close(separate.sequence.intervals,torch.tensor([[.2,.3],[.3,.6],[.6,.9]],dtype=torch.float64),rtol=0,atol=0)
    assert separate.source_graph_indices==((1,),(2,),(0,))
    assert separate.source_graph_to_canonical.tolist()==[2,0,1]
    merged=canonicalize_observed_trees(seq,merge_identical_adjacent=True)
    torch.testing.assert_close(merged.sequence.intervals,torch.tensor([[.2,.9]],dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(merged.sequence.positions,seq.positions,rtol=0,atol=0)
    torch.testing.assert_close(merged.sequence.window_bounds,seq.window_bounds,rtol=0,atol=0)


@pytest.mark.parametrize('kind',['age','topology','feature','clades'])
def test_genuine_observed_difference_is_not_merged(kind):
    seq=duplicate_intervals(example()); n=11
    if kind=='age': seq.x[n+10,6]=math.log1p(3000.)/math.log1p(10000.)
    elif kind=='feature': seq.x[n+2,0]=0.; seq.x[n+2,1]=1.
    elif kind=='clades': seq.clades[n+9,3]=1.-seq.clades[n+9,3]
    else:
        # Swap sampled query 1 and reference 2 between existing branches.
        edge=seq.edge_index
        first=(edge==n+1); second=(edge==n+2)
        edge[first]=n+2; edge[second]=n+1
    result=canonicalize_observed_trees(seq,merge_identical_adjacent=True)
    assert result.sequence.num_graphs==3
    assert result.source_graph_indices==((0,),(1,),(2,))


@pytest.mark.parametrize('model_class',[IndependentGNN,GNNGRU])
def test_ordinary_models_after_canonicalization_satisfy_both_fixture_contracts(model_class):
    first=adapter(canonicalize_observed_trees(example(),merge_identical_adjacent=True).sequence)
    second=adapter(canonicalize_observed_trees(permute_nodes(duplicate_intervals(example(True,True))),merge_identical_adjacent=True).sequence)
    torch.manual_seed(41)
    model=model_class(48,4,hidden_dim=12,num_layers=2).double().eval()
    with torch.no_grad():
        before=model(first); after=model(second)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert torch.isfinite(before).all()
    # Correspondence IDs can be renamed consistently without becoming learned
    # scalar features. Query output order is carried by query_node_ids.
    renamed=clone(first); mapping={10:90,20:80,30:70,40:60,50:50,60:40}
    for old,new in mapping.items(): renamed.node_ids[first.node_ids==old]=new
    renamed.query_node_ids=torch.tensor([mapping[int(q)] for q in first.query_node_ids])
    # first is augmented, so strip/recompute the same observed summaries again.
    renamed=adapter(canonicalize_observed_trees(renamed,merge_identical_adjacent=True).sequence)
    with torch.no_grad(): prediction=model(renamed)
    torch.testing.assert_close(before,prediction,rtol=2e-12,atol=2e-12)


def test_idempotence_mapping_provenance_and_weights_only_roundtrip():
    source_provenance={'manifest_sha256':'a'*64,'split':'train','parent_id':'rep000'}
    first=canonicalize_observed_trees(example(True,True),source_provenance=source_provenance)
    second=canonicalize_observed_trees(first.sequence)
    assert_same_sequence(first.sequence,second.sequence)
    source_provenance['split']='changed'
    assert first.provenance['source_provenance']['split']=='train'
    for out,row in enumerate(first.source_node_indices.tolist()):
        original=example(True,True)
        assert int(first.sequence.node_ids[out])==int(original.node_ids[row])
        torch.testing.assert_close(first.sequence.x[out,:7],original.x[row,:7],rtol=0,atol=0)
    payload=io.BytesIO(); torch.save(first.as_dict(),payload); payload.seek(0)
    restored=CanonicalizedObservedTrees.from_dict(torch.load(payload,weights_only=True))
    assert_same_sequence(first.sequence,restored.sequence)
    assert restored.provenance==first.provenance
    with pytest.raises(ValueError): CanonicalizedObservedTrees.from_dict({**first.as_dict(),'schema_version':2})


def test_external_labels_population_truth_and_metadata_never_read():
    class ObservedOnly:
        def __init__(self,seq): self.seq=seq
        def __getattr__(self,name):
            if name in ('labels','population','populations','truth','tree_sequence','metadata'):
                raise AssertionError('Forbidden external annotation access')
            return getattr(self.seq,name)
    baseline=canonicalize_observed_trees(example(),merge_identical_adjacent=True)
    forbidden=canonicalize_observed_trees(ObservedOnly(example(True,True)),merge_identical_adjacent=True)
    assert_same_sequence(baseline.sequence,forbidden.sequence)
    seq=example(True,True)
    seq.labels=torch.full((3,2),-999); seq.population_truth=torch.randn(50)
    changed=canonicalize_observed_trees(seq,merge_identical_adjacent=True)
    assert_same_sequence(baseline.sequence,changed.sequence)


@pytest.mark.parametrize('kind',['equal_age','negative_age','overflow_age','disconnected','cycle','self_edge','sampled_internal','anonymous_tip','duplicate_sample','branch_negative','interval_gap','bad_target'])
def test_admissibility_checks_reject_invalid_observed_inputs(kind):
    seq=example()
    if kind=='equal_age': seq.x[6,6]=seq.x[0,6]
    elif kind=='negative_age': seq.x[0,6]=-1.
    elif kind=='overflow_age': seq.x[10,6]=1000.
    elif kind=='disconnected':
        seq.edge_index=seq.edge_index[:,~((seq.edge_index[0]==8)&(seq.edge_index[1]==10))&~((seq.edge_index[0]==10)&(seq.edge_index[1]==8))]
    elif kind=='cycle': seq.edge_index=torch.cat((seq.edge_index,torch.tensor([[6,7],[7,6]])),dim=1)
    elif kind=='self_edge': seq.edge_index=torch.cat((seq.edge_index,torch.tensor([[0],[0]])),dim=1)
    elif kind=='sampled_internal': seq.node_ids[6]=123
    elif kind=='anonymous_tip': seq.node_ids[5]=-1
    elif kind=='duplicate_sample': seq.node_ids[3]=seq.node_ids[2]
    elif kind=='branch_negative': seq.x[0,7]=-1.
    elif kind=='interval_gap':
        seq=duplicate_intervals(seq); seq.intervals[1,0]+=.0001
    else: seq.target_graph[0]=100
    with pytest.raises(ValueError): canonicalize_observed_trees(seq)


def test_single_sample_unary_chain_has_a_valid_canonical_sample_root():
    ages=torch.tensor([0.,10.,100.],dtype=torch.float64)
    x=torch.zeros(3,8,dtype=torch.float64); x[0,4:6]=1
    x[:,6]=torch.log1p(ages)/math.log1p(10000.)
    x[:,7]=torch.log1p(torch.tensor([10.,90.,0.],dtype=torch.float64))/math.log1p(10000.)
    seq=GraphSequence(x=x,edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
        node_graph=torch.zeros(3,dtype=torch.long),node_ids=torch.tensor([42,-1,-1]),
        clades=torch.zeros(3,1,dtype=torch.float64),intervals=torch.tensor([[0.,1.]],dtype=torch.float64),
        query_node_ids=torch.tensor([42]),positions=torch.tensor([.5],dtype=torch.float64),
        target_graph=torch.tensor([0]),window_bounds=None)
    result=canonicalize_observed_trees(seq)
    assert result.sequence.node_ids.tolist()==[42]
    assert result.sequence.edge_index.shape==(2,0)
    assert result.sequence.x[0,7]==0
    assert result.provenance['removed_unary_root_nodes']==2


def test_invalid_prefix_and_physical_parameter_declarations():
    for options in ({'base_feature_count':9},{'base_feature_count':0},
                    {'time_column':7},{'parent_branch_column':8},
                    {'time_scale':0.},{'merge_identical_adjacent':1},
                    {'source_provenance':{'value':float('nan')}}):
        with pytest.raises(ValueError): canonicalize_observed_trees(example(),**options)
    seq=example(); seq.x=seq.x[:,0]
    with pytest.raises(ValueError): canonicalize_observed_trees(seq)

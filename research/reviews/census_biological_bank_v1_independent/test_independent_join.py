"""Fresh handmade CPU-only native oracles; no observed project artifact readers."""
from dataclasses import replace
from copy import deepcopy
import hashlib
import math
import os
import numpy as np
import pytest
import torch
import tskit
from arggnn.biological_candidate import pack_observed_tile
from arggnn.census_biological_bank_v1 import (
    ObservedSampleRoleV1, prepare_census_biological_bank_v1,
    forward_certified_census_biological_v1, fingerprint_sha256_v1,
    validate_census_biological_certificate_v1, load_census_biological_config_v1,
    collect_census_biological_support_v1, select_census_numerical_tiles_v1, _content)
from arggnn.census_tree_encoder_v1 import CensusTreeGNNV1, CensusWholeTreeEncoderV1

@pytest.fixture(autouse=True)
def cpu_only():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)


def handcrafted(unary=True, merger=350., q_age=349.9, width=1100., relabel=False):
    tables=tskit.TableCollection(sequence_length=width)
    tables.nodes.add_row(time=2000.)  # permanently unrelated, before sample IDs
    ages=[0.,100.,q_age,80.,200.,330.,350.,351.,0.,700.]
    nodes=[tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=t) for t in ages]
    ids=[71,4,908,25,444,2,603,9,114,88]
    if relabel: ids=[x+100000 for x in ids]
    roster=[ObservedSampleRoleV1(ids[i],n,'query' if i<3 else 'mixed_reference_only' if i<6 else 'pure_source_reference',ages[i],None if i<6 else i-6) for i,n in enumerate(nodes)]
    bounds=[0.,150.,220.,220.1,width] if width==1100. else [0.,width]
    for graph,(left,right) in enumerate(zip(bounds[:-1],bounds[1:])):
        def join(children,time):
            node=tables.nodes.add_row(time=time)
            for child in children:tables.edges.add_row(left,right,node,child)
            return node
        if graph in (0,3):
            one=nodes[0]
            if unary and graph==0:one=join([join([one],30.)],30.000001)
            a=join([one,nodes[1]],merger);b=join([nodes[2],nodes[3]],400.);c=join([nodes[4],nodes[5]],340.)
        elif graph==1:
            a=join([nodes[0],nodes[2]],350.0000001);b=join([nodes[1],nodes[4]],300.);c=join([nodes[3],nodes[5]],340.)
        else:
            a=join([nodes[0],nodes[4]],220.);b=join([nodes[1],nodes[2]],max(349.99,q_age+0.01));c=join([nodes[3],nodes[5]],349.)
        qroot=join([join([a,b],500.),c],600.)
        rroot=join([join(nodes[6:8],800.),join(nodes[8:10],801.)],850.)
        join([qroot,rroot],900.)
    tables.sort()
    return tables.tree_sequence(),roster


def cfg():
    config=load_census_biological_config_v1()
    config.update(fixture_mode=True,role_counts={'query':3,'mixed_reference_only':3,'pure_source_reference':4},references_per_class=1)
    return config


def encode(ts,roster,left=250000.,right=None):
    if right is None:right=left+ts.sequence_length
    grid=np.arange(50.,2000000.,100.,dtype=np.float64)
    parent={'observation':{'nodes':np.array([r.native_sample_node for r in roster]),'offset':left},
      'dynamics_slots':np.array([i for i,r in enumerate(roster) if r.role!='pure_source_reference']),
      'query_slots':np.array([i for i,r in enumerate(roster) if r.role=='query']),
      'reference_slots':np.array([i for i,r in enumerate(roster) if r.role=='pure_source_reference']),
      'samples':[{'observed_reference_class':r.observed_reference_class} for r in roster],'positions':grid}
    sequence,indexes,_,_=pack_observed_tile(ts,parent,left,right)
    return {'sequence':sequence.as_dict(),'labels_in_input':False},torch.from_numpy(grid[indexes]),torch.from_numpy(indexes)


def prepare(ts=None,roster=None,raw=None,left=250000.,right=None,positions=None,indexes=None,bindings=None):
    if ts is None:ts,roster=handcrafted()
    if right is None:right=left+ts.sequence_length
    original,p,g=encode(ts,roster,left,right)
    if raw is None:raw=original
    if positions is None:positions=p
    if indexes is None:indexes=g
    if bindings is None:bindings={k:{'path':'independent_handmade/'+k,'sha256':hashlib.sha256(('synthetic_'+k).encode()).hexdigest()} for k in ('native_observed_ts','serialized_whole_observed','observed_roster','observed_grid','producer_receipt')}
    return prepare_census_biological_bank_v1(ts,raw,roster,left_bp=left,right_bp=right,offset_bp=left,
        positions_bp=positions,original_grid_indexes=indexes,source_bindings=bindings,parent_id='independent_train',split='train',tile_id=1,config=cfg())


def native_groups(tree,roster):
    queries=[r for r in roster if r.role!='pure_source_reference']
    remaining=set(range(len(queries)));groups=[]
    while remaining:
        first=min(remaining)
        members={i for i in remaining if tree.time(tree.mrca(queries[first].native_sample_node,queries[i].native_sample_node))<=350.}
        groups.append(frozenset(queries[i].actual_id for i in members));remaining-=members
    return set(groups)


def ancestor_union(tree,samples):
    union=set()
    for sample in samples:
        n=sample
        while n!=-1:union.add(n);n=tree.parent(n)
    return union


def test_fresh_native_donor_cut_edge_row_grid_and_f47_oracles():
    ts,roster=handcrafted();raw,_,_=encode(ts,roster);cert=prepare(ts,roster,raw)
    assert cert.provenance['original_tree_ids']==[0,1,2,3]
    assert not (cert.bank.target_graph==2).any() and not (cert.bank.layout.anchor_graph==2).any()
    assert cert.bank.target_graph[1].item()==1
    assert len(cert.inputs)==4 and cert.bank.donor_weights.shape==(4,3,6,4)
    q=[r.native_sample_node for r in roster if r.role!='pure_source_reference'];refs=[r.native_sample_node for r in roster if r.role=='pure_source_reference']
    for graph,tree in enumerate(ts.trees()):
        packed=list(tree.nodes(order='postorder'));source=cert.provenance['source_rows_by_tree'][graph]
        assert cert.provenance['native_vertex_by_source_row'][graph].tolist()==packed
        assert set(packed)==ancestor_union(tree,q+refs)
        assert cert.provenance['excluded_unrelated_nodes_by_tree'][graph]==ts.num_nodes-len(packed)
        local={n:i for i,n in enumerate(packed)}
        assert cert.native_trees[graph].parent.tolist()==[-1 if tree.parent(n)==-1 else local[tree.parent(n)] for n in packed]
        assert cert.native_trees[graph].node_times.tolist()==[tree.time(n) for n in packed]
        assert set(cert.bank.partitions[graph].group_members)==native_groups(tree,roster)
        oracle=np.array([[tree.time(tree.mrca(a,b)) for b in refs] for a in q],dtype=np.float64)
        logits=-np.log1p(oracle)[None]/math.log1p(10000.)/np.array([.05,.1,.2])[:,None,None]
        oracle=np.exp(logits-logits.max(-1,keepdims=True));oracle/=oracle.sum(-1,keepdims=True)
        np.testing.assert_allclose(cert.bank.donor_weights[graph].numpy(),oracle,rtol=2e-15,atol=2e-16)
        for i,r in enumerate(roster):
            assert cert.provenance['sample_joins_by_tree'][graph]['source_rows'][i].item()==source[local[r.native_sample_node]].item()
        retained=cert.inputs[graph].native_vertex_index.tolist()
        cols=[i for i in range(48) if i!=7]
        expected=raw['sequence']['x'][source[[local[n] for n in retained]]][:,cols]
        torch.testing.assert_close(cert.inputs[graph].features,expected,rtol=0,atol=0)
        for i,n in enumerate(retained):
            parent=cert.inputs[graph].parent[i].item()
            duration=0. if parent<0 else tree.time(retained[parent])-tree.time(n)
            assert cert.inputs[graph].branch_duration[i].item()==duration
    assert len(cert.bank.partitions[0].older_reference_ids)==3


def test_independent_enumeration_of_x48_reference_summary_semantics():
    ts,roster=handcrafted();cert=prepare(ts,roster)
    refs=[r.native_sample_node for r in roster if r.role=='pure_source_reference']
    for graph,tree in enumerate(ts.trees()):
        packed=cert.provenance['native_vertex_by_source_row'][graph].tolist();rows=cert.provenance['source_rows_by_tree'][graph]
        encoded={n:float(np.float32(math.log1p(tree.time(n))/math.log1p(10000.))) for n in packed}
        for i,n in enumerate(packed):
            times=np.array([encoded[tree.mrca(n,r)] for r in refs]);minimum=times.min()
            votes=(times==minimum).astype(float);votes/=votes.sum()
            evidence=np.exp(-times[:,None]/np.array([.05,.1,.2])[None]);evidence/=evidence.sum(0,keepdims=True)
            expected=[]
            for j,r in enumerate(refs):expected.extend([float(tree.is_descendant(r,n)),.25,times[j],times[j],0.,1.,votes[j],*evidence[j]])
            np.testing.assert_allclose(cert.stored_sequence.x[rows[i],8:].numpy(),np.array(expected),rtol=5e-7,atol=2e-7)


def test_identical_float32_x48_different_native_census_oracle():
    ts,roster=handcrafted(merger=350.);before=prepare(ts,roster)
    ts,roster=handcrafted(merger=350.0000001);after=prepare(ts,roster)
    torch.testing.assert_close(before.stored_sequence.x,after.stored_sequence.x,rtol=0,atol=0)
    assert set(before.bank.partitions[0].group_members)!=set(after.bank.partitions[0].group_members)
    assert before.certificate_sha256!=after.certificate_sha256


def test_unary_additive_native_endpoints_and_retained_features():
    with_unary=prepare();ts,roster=handcrafted(unary=False);plain=prepare(ts,roster)
    assert len(with_unary.native_trees[0].parent)-len(with_unary.inputs[0].parent)==2
    # The two explicitly inserted node-table rows shift interior native IDs by2,
    # and postorder may change retained row order. Align the known recipe IDs.
    lookup={n:i for i,n in enumerate(with_unary.inputs[0].native_vertex_index.tolist())}
    order=torch.tensor([lookup[n if n<=10 else n+2] for n in plain.inputs[0].native_vertex_index.tolist()])
    torch.testing.assert_close(with_unary.inputs[0].features[order],plain.inputs[0].features,rtol=0,atol=0)
    torch.testing.assert_close(with_unary.inputs[0].branch_duration[order],plain.inputs[0].branch_duration,rtol=0,atol=0)
    torch.testing.assert_close(with_unary.bank.donor_weights,plain.bank.donor_weights,rtol=0,atol=0)
    ts,roster=handcrafted();raw,_,_=encode(ts,roster)
    unaryrow=next(i for i,n in enumerate(ts.first().nodes(order='postorder')) if ts.first().num_children(n)==1)
    raw['sequence']['x'][unaryrow,20]+=.3
    with pytest.raises(ValueError,match='parity failed: x'):prepare(ts,roster,raw)


@pytest.mark.parametrize('field',['x','edge_index','node_graph','node_ids','clades','intervals','query_node_ids','positions','target_graph','window_bounds'])
def test_independent_negative_all_serialized_fields(field):
    ts,roster=handcrafted();raw,_,_=encode(ts,roster);raw=deepcopy(raw);v=raw['sequence'][field]
    if field=='edge_index':v[:,0]=v[:,1]
    elif field=='target_graph':v[0]=1
    elif v.is_floating_point():v.reshape(-1)[0]+=.005
    else:v.reshape(-1)[0]+=1
    with pytest.raises(ValueError):prepare(ts,roster,raw)


@pytest.mark.parametrize('mutator',['missing_digest','wrong_digest','donor','R_time','parent','F47','forest_edge','scored_role','source_hash','native_map','grid'])
def test_stale_or_absent_certificate_rejected_before_model_call(mutator):
    cert=prepare();expected=cert.certificate_sha256
    if mutator=='missing_digest':expected=None
    elif mutator=='wrong_digest':expected='a'*64
    elif mutator=='donor':cert.bank.donor_weights[0,0,0,0]+=.01
    elif mutator=='R_time':cert.native_trees[0].node_times[cert.native_trees[0].sample_nodes[-1]]+=.01
    elif mutator=='parent':cert.native_trees[0].parent[0]=-1
    elif mutator=='F47':cert.inputs[0].features[0,0]+=.01
    elif mutator=='forest_edge':cert.forest.edge_time[0,0]+=.01
    elif mutator=='scored_role':cert.bank.partitions[0].scored_query_mask[0]=False
    elif mutator=='source_hash':cert.provenance['source_bindings']['observed_roster']['sha256']='f'*64
    elif mutator=='native_map':cert.provenance['native_vertex_by_source_row'][0][0]+=1
    else:cert.provenance['positions_bp'][0]+=.1
    called=[]
    def sentinel(*args,**kwargs):called.append(True);raise AssertionError('model called')
    with pytest.raises(ValueError,match='Changed/stale'):forward_certified_census_biological_v1(sentinel,cert,expected_certificate_sha256=expected)
    assert not called


def test_coherent_census_equal_context_still_rejected():
    ts,roster=handcrafted(q_age=350.)
    with pytest.raises(ValueError,match='All Q'):prepare(ts,roster)
    ts,roster=handcrafted();roster=list(roster);roster[1]=replace(roster[1],actual_id=roster[0].actual_id)
    with pytest.raises(ValueError,match='roster differs'):prepare(ts,roster)


def test_signed_zero_byte_parity_boundary_is_recorded():
    ts,roster=handcrafted();raw,_,_=encode(ts,roster);changed=deepcopy(raw)
    index=next((i,j) for i in range(raw['sequence']['x'].shape[0]) for j in range(48) if raw['sequence']['x'][i,j].item()==0. and j!=7)
    changed['sequence']['x'][index]=-0.
    assert torch.equal(changed['sequence']['x'],raw['sequence']['x'])
    assert changed['sequence']['x'].numpy().tobytes()!=raw['sequence']['x'].numpy().tobytes()
    original=prepare(ts,roster,raw);accepted=prepare(ts,roster,changed)
    assert original.certificate_sha256!=accepted.certificate_sha256


def test_physical_equal_anchor_readout_ieee_mismatch_rejects():
    ts,roster=handcrafted(width=3200.);raw,positions,indexes=encode(ts,roster)
    frozen=.125+(torch.arange(32,dtype=torch.float64)+.5)*((253200./2000000.-.125)/32.)
    assert (frozen!=positions/2000000.).any()
    with pytest.raises(ValueError,match='Physical anchor collision'):prepare(ts,roster,raw,positions=positions,indexes=indexes)


def test_source_record_content_provenance_boundary_and_digest_resealing():
    cert=prepare();bad=deepcopy(cert);external=cert.certificate_sha256
    bad.provenance['source_bindings']['native_observed_ts']['sha256']='f'*64
    bad=replace(bad,certificate_sha256=fingerprint_sha256_v1(_content(bad)))
    with pytest.raises(ValueError,match='Changed/stale'):validate_census_biological_certificate_v1(bad,external)
    validate_census_biological_certificate_v1(bad,bad.certificate_sha256)


def model():
    torch.manual_seed(317)
    return CensusTreeGNNV1(5,2,4,1,3,initial_exchange=(.7,1.1,1.8),initial_damping=.5).double()


def test_actual_id_relabel_invariance_and_all_five_same_weight_controls():
    cert=prepare();ts,roster=handcrafted(relabel=True);relabeled=prepare(ts,roster);gnn=model()
    for control in ('proposed','eta0','permuted_groups','raw_diagonal','local'):
        kwargs={'group_permutation':torch.tensor([5,2,0,1,3,4])} if control=='permuted_groups' else {}
        a=forward_certified_census_biological_v1(gnn,cert,expected_certificate_sha256=cert.certificate_sha256,control=control,**kwargs)
        b=forward_certified_census_biological_v1(gnn,relabeled,expected_certificate_sha256=relabeled.certificate_sha256,control=control,**kwargs)
        torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0)
        assert a.scored_query_ids.tolist()==[71,4,908] and b.scored_query_ids.tolist()==[100071,100004,100908]
        grads=torch.autograd.grad(a.logits.square().sum(),tuple(gnn.parameters()),allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_fresh_forest_output_vjp_and_whole_gnn_directional_derivative():
    cert=prepare();torch.manual_seed(719);encoder=CensusWholeTreeEncoderV1(5).double()
    singles=[encoder(item) for item in cert.inputs];forest=encoder(cert.forest)
    for a,b in zip(singles,forest):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    parameters=tuple(encoder.parameters());grads=[]
    for values in (singles,forest):grads.append(torch.autograd.grad(sum((i+1)*x.sin().sum() for i,x in enumerate(values)),parameters))
    for a,b in zip(*grads):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    gnn=model();pars=tuple(gnn.parameters());out=forward_certified_census_biological_v1(gnn,cert,expected_certificate_sha256=cert.certificate_sha256)
    objective=out.logits.sin().sum();grad=torch.autograd.grad(objective,pars)
    directions=[torch.randn_like(p)/math.sqrt(sum(x.numel() for x in pars)) for p in pars]
    analytic=sum((g*d).sum() for g,d in zip(grad,directions)).item();eps=1e-5
    def shifted(sign):
        with torch.no_grad():
            for p,d in zip(pars,directions):p.add_(sign*eps*d)
        value=forward_certified_census_biological_v1(gnn,cert,expected_certificate_sha256=cert.certificate_sha256).logits.sin().sum().item()
        with torch.no_grad():
            for p,d in zip(pars,directions):p.add_(-sign*eps*d)
        return value
    finite=(shifted(1)-shifted(-1))/(2*eps)
    assert math.isclose(analytic,finite,rel_tol=2e-6,abs_tol=2e-8)


def test_structural_selection_ties_and_native_span_weighted_support_oracle():
    rows=[{'parent_id':f'{split}{parent}','split':split,'tile_id':tile,'num_trees':1+(tile%3)} for split in ('train','val') for parent in (0,1) for tile in range(8)]
    a=select_census_numerical_tiles_v1(rows);b=select_census_numerical_tiles_v1(list(reversed(rows)))
    assert a==b and a['minimum']['parent_id']=='train0' and a['minimum']['tile_id']==0
    assert a['maximum']['parent_id']=='train0' and a['maximum']['tile_id']==2
    with pytest.raises(ValueError,match='no outcomes'):select_census_numerical_tiles_v1([dict(r,nll=.5) for r in rows])
    ts,roster=handcrafted();cert=prepare(ts,roster)
    support=collect_census_biological_support_v1([cert],[cert.certificate_sha256],expected_tiles=1)['tiles'][0]
    weighted={};singleton=0.
    for tree in ts.trees():
        groups=native_groups(tree,roster);span=tree.interval.right-tree.interval.left
        for group in groups:
            weighted[str(len(group))]=weighted.get(str(len(group)),0.)+span*len(group)/6/1100.
            if len(group)==1:singleton+=span/1100./6
    assert support['physical_span_weighted_query_group_size']==pytest.approx(weighted,rel=1e-14,abs=1e-14)
    assert support['physical_span_weighted_singleton_Q_fraction']==pytest.approx(singleton,rel=1e-14,abs=1e-14)


def test_native_vertex_table_relabel_changes_postorder_without_learned_ids():
    ts,roster=handcrafted();original=prepare(ts,roster)
    tables=ts.dump_tables();order=np.arange(ts.num_nodes)[::-1];inverse=np.argsort(order)
    tables.nodes.set_columns(flags=tables.nodes.flags[order],time=tables.nodes.time[order])
    tables.edges.parent=inverse[tables.edges.parent].astype(np.int32);tables.edges.child=inverse[tables.edges.child].astype(np.int32)
    tables.sort();changed=tables.tree_sequence()
    changed_roster=[replace(r,native_sample_node=int(inverse[r.native_sample_node])) for r in roster]
    relabeled=prepare(changed,changed_roster)
    assert original.provenance['native_vertex_by_source_row'][0].tolist()!=relabeled.provenance['native_vertex_by_source_row'][0].tolist()
    assert original.certificate_sha256!=relabeled.certificate_sha256
    gnn=model()
    for control in ('proposed','eta0','permuted_groups','raw_diagonal','local'):
        kwargs={'group_permutation':torch.tensor([5,2,0,1,3,4])} if control=='permuted_groups' else {}
        a=forward_certified_census_biological_v1(gnn,original,expected_certificate_sha256=original.certificate_sha256,control=control,**kwargs)
        b=forward_certified_census_biological_v1(gnn,relabeled,expected_certificate_sha256=relabeled.certificate_sha256,control=control,**kwargs)
        torch.testing.assert_close(a.logits,b.logits,rtol=1e-10,atol=1e-12)


def test_native_sample_set_turnover_and_right_hold_span_oracle():
    ts,roster=handcrafted();cert=prepare(ts,roster)
    groups=[native_groups(tree,roster) for tree in ts.trees()]
    lookup=[{query:group for group in part for query in group} for part in groups]
    queries=[r.actual_id for r in roster if r.role!='pure_source_reference']
    rhos=[[len(lookup[g-1][q]&lookup[g][q])/len(lookup[g-1][q]|lookup[g][q]) for q in queries] for g in range(1,len(groups))]
    layout=cert.bank.layout
    for event in range(1,layout.num_segments):
        before,after=layout.graph_at_segment[event-1:event+1]
        wanted=[1.]*len(queries) if before==after else rhos[after-1]
        torch.testing.assert_close(cert.bank.boundary_rho[event],torch.tensor(wanted,dtype=torch.float64),rtol=0,atol=0)
    weighted={}
    for graph,tree in enumerate(ts.trees()):
        if graph:
            for rho in rhos[graph-1]:weighted[str(rho)]=weighted.get(str(rho),0.)+(tree.interval.right-tree.interval.left)/len(queries)
    total=sum(weighted.values());weighted={k:v/total for k,v in weighted.items()}
    row=collect_census_biological_support_v1([cert],[cert.certificate_sha256],expected_tiles=1)['tiles'][0]
    assert row['right_hold_span_weighted_boundary_rho']==pytest.approx(weighted,rel=1e-14,abs=1e-14)

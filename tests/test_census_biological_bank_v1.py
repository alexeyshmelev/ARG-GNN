"""Handmade TS/serialization join qualification only; no actual arrays or fit."""
from dataclasses import replace
import hashlib, json, math, os
import numpy as np
import pytest
import torch
import tskit
from arggnn.biological_candidate import pack_observed_tile
from arggnn.census_biological_bank_v1 import (ObservedSampleRoleV1,prepare_census_biological_bank_v1,validate_census_biological_certificate_v1,
    forward_certified_census_biological_v1,load_census_biological_config_v1,collect_census_biological_support_v1,select_census_numerical_tiles_v1)
from arggnn.census_tree_encoder_v1 import CensusTreeGNNV1,CensusWholeTreeEncoderV1

@pytest.fixture(autouse=True)
def handmade_cpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)


def bindings():
    return {name:{'path':'handmade/'+name,'sha256':hashlib.sha256(name.encode()).hexdigest()} for name in ('native_observed_ts','serialized_whole_observed','observed_roster','observed_grid','producer_receipt')}


def fixture(full=False,unary=False):
    tables=tskit.TableCollection(sequence_length=1000.)
    count=280 if full else 8
    ages=[100.]*120+[200.]*64+[400.+(i%24)*3 for i in range(96)] if full else [0.,100.,200.,300.,400.,500.,550.,600.]
    for age in ages:tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age)
    if full:
        active=list(range(count))
        while len(active)>1:
            later=[]
            for start in range(0,len(active),2):
                pair=active[start:start+2]
                if len(pair)==1:later+=pair;continue
                ancestor=tables.nodes.add_row(time=max(tables.nodes[x].time for x in pair)+100.)
                for child in pair:tables.edges.add_row(0.,1000.,ancestor,child)
                later.append(ancestor)
            active=later
        tables.nodes.add_row(time=2000.)  # unrelated stored marginal node
    else:
        times=[300.,330.,500.,700.,750.,850.,1000.,330.,340.,500.,700.,750.,850.,1000.]
        for age in times:tables.nodes.add_row(time=age)
        first={0:8,1:8,2:9,3:9,8:10,9:10,4:11,5:11,6:12,7:12,11:13,12:13,10:14,13:14}
        if unary:
            a=tables.nodes.add_row(time=200.);b=tables.nodes.add_row(time=200.000001)
            first[0]=a;first[a]=b;first[b]=8
        second={0:15,2:15,1:16,3:16,15:17,16:17,4:18,5:18,6:19,7:19,18:20,19:20,17:21,20:21}
        for left,right,parents in ((0.,500.,first),(500.,1000.,second)):
            for child,parent in parents.items():tables.edges.add_row(left,right,parent,child)
    tables.sort();ts=tables.tree_sequence()
    scored=120 if full else 2;context=64 if full else 2
    roster=[ObservedSampleRoleV1(1000+i,i,'query' if i<scored else 'mixed_reference_only' if i<scored+context else 'pure_source_reference',age,None if i<scored+context else (i-scored-context)//24 if full else i-4) for i,age in enumerate(ages)]
    return ts,roster


def serialized(ts,roster,left=0.,right=1000.,offset=0.):
    grid=np.arange(50.,max(right,1000.),100.)
    parent={'observation':{'nodes':np.array([row.native_sample_node for row in roster]),'offset':offset},
            'dynamics_slots':np.array([i for i,row in enumerate(roster) if row.role!='pure_source_reference']),
            'query_slots':np.array([i for i,row in enumerate(roster) if row.role=='query']),
            'reference_slots':np.array([i for i,row in enumerate(roster) if row.role=='pure_source_reference']),
            'samples':[{'observed_reference_class':row.observed_reference_class} for row in roster], 'positions':grid}
    sequence,indexes,tree_ids,landmarks=pack_observed_tile(ts,parent,left,right)
    return {'sequence':sequence.as_dict(),'labels_in_input':False},torch.tensor(grid[indexes],dtype=torch.float64),torch.tensor(indexes),tree_ids,landmarks


def prepare(full=False,unary=False,*,ts=None,roster=None,observation=None):
    if ts is None:ts,roster=fixture(full,unary)
    raw,positions,indexes,_,_=serialized(ts,roster)
    if observation is not None:raw=observation
    config=load_census_biological_config_v1()
    if not full:config.update(fixture_mode=True,role_counts={'query':2,'mixed_reference_only':2,'pure_source_reference':4},references_per_class=1)
    return prepare_census_biological_bank_v1(ts,raw,roster,left_bp=0.,right_bp=1000.,offset_bp=0.,positions_bp=positions,
        original_grid_indexes=indexes,source_bindings=bindings(),parent_id='handmade_train',split='train',tile_id=0,config=config)


def test_pinned_full280_roster_exact_native_all_tree_and_F47_join():
    cert=prepare(full=True)
    assert cert.bank.query_ids.shape==(184,) and cert.bank.reference_ids.shape==(96,)
    assert cert.bank.partitions[0].scored_query_mask.sum()==120
    assert len(cert.native_trees[0].parent)==559 and len(cert.inputs[0].parent)==559
    assert cert.provenance['excluded_unrelated_nodes_by_tree']==[1]
    assert cert.provenance['full_x48_and_all_GraphSequence_fields_exact_parity'] is True
    assert cert.bank.donor_weights.shape==(1,3,184,96) and len(cert.bank.partitions[0].older_reference_ids)==96
    validate_census_biological_certificate_v1(cert,cert.certificate_sha256)


def test_anonymous_interiors_join_by_producer_rows_not_equal_age_or_clade():
    cert=prepare(unary=True)
    native=cert.provenance['native_vertex_by_source_row'][0].tolist()
    assert 22 in native and 23 in native and native.index(22)!=native.index(23)
    assert cert.native_trees[0].node_times[native.index(22)].item()==200.
    assert cert.native_trees[0].node_times[native.index(23)].item()==200.000001
    assert cert.stored_sequence.x[native.index(22),6]==cert.stored_sequence.x[native.index(23),6]
    assert len(cert.native_trees[0].parent)==17 and len(cert.inputs[0].parent)==15
    assert 22 not in cert.inputs[0].native_vertex_index.tolist() and 23 not in cert.inputs[0].native_vertex_index.tolist()
    plain=prepare()
    lookup={node:row for row,node in enumerate(cert.inputs[0].native_vertex_index.tolist())}
    order=torch.tensor([lookup[node] for node in plain.inputs[0].native_vertex_index.tolist()])
    torch.testing.assert_close(cert.inputs[0].features[order],plain.inputs[0].features,rtol=0,atol=0)
    torch.testing.assert_close(cert.inputs[0].branch_duration[order],plain.inputs[0].branch_duration,rtol=0,atol=0)
    assert cert.bank.partitions[0].group_members==plain.bank.partitions[0].group_members


def test_direct_float64_dates_survive_identical_stored_float32_age_semantics():
    ts,roster=fixture();old=prepare(ts=ts,roster=roster)
    tables=ts.dump_tables();times=tables.nodes.time.copy();times[14]+=1e-7
    tables.nodes.time=times;tables.sort();changed=tables.tree_sequence();new=prepare(ts=changed,roster=roster)
    torch.testing.assert_close(old.stored_sequence.x,new.stored_sequence.x,rtol=0,atol=0)
    location=new.provenance['native_vertex_by_source_row'][0].tolist().index(14)
    assert new.native_trees[0].node_times.dtype==torch.float64
    assert new.native_trees[0].node_times[location].item()==1000.0000001
    assert new.certificate_sha256!=old.certificate_sha256


@pytest.mark.parametrize('field',['x','edge_index','node_ids','clades','intervals','query_node_ids','positions','target_graph'])
def test_any_serialized_feature_topology_landmark_identity_coordinate_change_rejects(field):
    ts,roster=fixture();raw,*_=serialized(ts,roster)
    raw={'labels_in_input':False,'sequence':{key:(value.clone() if isinstance(value,torch.Tensor) else value) for key,value in raw['sequence'].items()}}
    value=raw['sequence'][field]
    if field=='edge_index':value[:,0]=value[:,1]
    elif field=='query_node_ids':value[0]=value[1]
    elif field=='target_graph':value[0]=1-value[0]
    elif value.is_floating_point():value.flatten()[0]+=.001
    else:value.flatten()[0]+=1
    with pytest.raises(ValueError):prepare(ts=ts,roster=roster,observation=raw)


def test_observation_and_role_whitelists_and_strict_context_dates():
    ts,roster=fixture();raw,*_=serialized(ts,roster)
    for forbidden in ('targets','truth','sampling_population'):
        bad=dict(raw);bad[forbidden]=torch.zeros(1)
        with pytest.raises(ValueError,match='whitelist'):prepare(ts=ts,roster=roster,observation=bad)
    changed=list(roster);changed[2]=replace(changed[2],observed_reference_class=0)
    with pytest.raises(ValueError,match='All Q'):prepare(ts=ts,roster=changed)
    changed=list(roster);changed[2]=replace(changed[2],age_generations=350.)
    with pytest.raises(ValueError,match='native sample date'):prepare(ts=ts,roster=changed)
    with pytest.raises(ValueError,match='role counts'):prepare(full=True,ts=ts,roster=roster)


@pytest.mark.parametrize('which',['R_date','donor','F47','forest','source','grid','cut'])
def test_external_certificate_rejects_all_stale_or_mutated_join_inputs(which):
    cert=prepare();digest=cert.certificate_sha256
    if which=='R_date':cert.native_trees[0].node_times[cert.native_trees[0].sample_nodes[-1]]+=1.
    elif which=='donor':cert.bank.donor_weights[0,0,0,0]+=.1
    elif which=='F47':cert.inputs[0].features[0,8]+=.1
    elif which=='forest':cert.forest.parent[0]=0
    elif which=='source':cert.provenance['source_bindings']['native_observed_ts']['sha256']='f'*64
    elif which=='grid':cert.provenance['original_grid_indexes'][0]+=1
    else:cert.bank.group_index[0,0]=3
    with pytest.raises(ValueError,match='Changed/stale'):validate_census_biological_certificate_v1(cert,digest)


@pytest.mark.parametrize('control',['proposed','eta0','permuted_groups','raw_diagonal','local'])
def test_certified_same_weight_shared_GNN_controls_finite_VJP_and_one_unit_AdamW_step(control):
    cert=prepare();torch.manual_seed(41)
    model=CensusTreeGNNV1(7,3,4,2,3,initial_exchange=(.7,1.1,1.8),initial_damping=.5).double()
    kwargs={'group_permutation':torch.tensor([2,3,1,0])} if control=='permuted_groups' else {}
    output=forward_certified_census_biological_v1(model,cert,expected_certificate_sha256=cert.certificate_sha256,control=control,**kwargs)
    assert output.logits.shape==(10,2,4) and output.scored_query_ids.tolist()==[1000,1001]
    assert torch.isfinite(output.logits).all()
    weight=torch.linspace(-.4,1.,output.logits.numel(),dtype=torch.float64).reshape_as(output.logits)
    (output.logits*weight).sum().backward()
    for name,parameter in model.named_parameters():
        inactive=(name.endswith('raw_eta') and control in {'eta0','local'}) or (name.endswith('raw_damping') and control=='local')
        if inactive:assert parameter.grad is None
        else:assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    assert torch.isfinite(norm)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    optimizer.step()  # single synthetic unit step, no labels, fitting loop or real inputs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    validate_census_biological_certificate_v1(cert,cert.certificate_sha256)


def test_prespecified_TRAIN_min_max_selection_all32_structural_tiles_and_no_outcomes():
    metadata=[{'parent_id':f'{split}_{parent}','split':split,'tile_id':tile,'num_trees':(tile+parent)%8+1} for split in ('train','val') for parent in range(2) for tile in range(8)]
    selected=select_census_numerical_tiles_v1(metadata)
    assert selected['minimum']=={'parent_id':'train_0','split':'train','tile_id':0,'num_trees':1}
    assert selected['maximum']=={'parent_id':'train_0','split':'train','tile_id':7,'num_trees':8}
    bad=[dict(row,accuracy=1.) for row in metadata]
    with pytest.raises(ValueError,match='no outcomes'):select_census_numerical_tiles_v1(bad)
    cert=prepare();summary=collect_census_biological_support_v1([cert],[cert.certificate_sha256],expected_tiles=1)
    row=summary['tiles'][0]
    assert row['sample_count']==8 and row['num_trees']==2 and row['older_R_proxy_count']==4
    assert abs(sum(row['physical_span_weighted_query_group_size'].values())-1)<1e-12
    assert abs(sum(row['right_hold_span_weighted_boundary_rho'].values())-1)<1e-12
    assert 0<=row['cross_query_exchange_possible_fraction']<=1
    assert row['forest_float32_state_copy_budget_bytes']>0


def test_joined_single_tree_and_forest_encoder_output_parameter_VJP_parity():
    cert=prepare();torch.manual_seed(521);encoder=CensusWholeTreeEncoderV1(7).double()
    separate=[encoder(item) for item in cert.inputs];batched=encoder(cert.forest)
    for left,right in zip(separate,batched):torch.testing.assert_close(left,right,rtol=1e-10,atol=1e-12)
    gradients=[torch.autograd.grad(sum(value.square().sum() for value in values),tuple(encoder.parameters()),retain_graph=True) for values in (separate,batched)]
    for left,right in zip(*gradients):torch.testing.assert_close(left,right,rtol=1e-8,atol=1e-10)



def test_native_census_cut_changes_when_stored_float32_ages_remain_identical():
    ts,roster=fixture()
    def dated(age):
        tables=ts.dump_tables();times=tables.nodes.time.copy();times[8]=age
        tables.nodes.time=times;tables.sort();return tables.tree_sequence()
    before=prepare(ts=dated(350.),roster=roster);after=prepare(ts=dated(350.0000001),roster=roster)
    torch.testing.assert_close(before.stored_sequence.x,after.stored_sequence.x,rtol=0,atol=0)
    assert before.bank.partitions[0].group_members!=after.bank.partitions[0].group_members
    assert before.certificate_sha256!=after.certificate_sha256



def test_original_grid_cannot_be_thinned_and_joined_source_mass_is_bounded():
    ts,roster=fixture();raw,positions,indexes,_,_=serialized(ts,roster)
    cfg=load_census_biological_config_v1();cfg.update(fixture_mode=True,role_counts={'query':2,'mixed_reference_only':2,'pure_source_reference':4},references_per_class=1)
    with pytest.raises(ValueError,match='All original centers'):
        prepare_census_biological_bank_v1(ts,raw,roster,left_bp=0.,right_bp=1000.,offset_bp=0.,positions_bp=positions[1:],original_grid_indexes=indexes[1:],source_bindings=bindings(),parent_id='train',split='train',tile_id=0,config=cfg)
    cert=prepare();torch.manual_seed(12)
    model=CensusTreeGNNV1(7,3,4,2,3,initial_exchange=(.7,1.1,1.8),initial_damping=.5).double()
    from arggnn.census_tree_encoder_v1 import encode_whole_tree_contexts_v1
    q,r=encode_whole_tree_contexts_v1(model.encoder,cert.inputs,cert.bank.query_ids,cert.bank.reference_ids)
    anchor=cert.bank.layout.anchor_graph
    for control in ('proposed','eta0','raw_diagonal'):
        messages=model.layers[0].messages(cert.bank,q[anchor],r[anchor],control=control)
        assert (messages.mass>0).all() and torch.isfinite(messages.mass).all()
        assert messages.targets.abs().max()<=1.+1e-12 and messages.anchors.abs().max()<=1.+1e-12



def test_handmade_stored_serialization_roundtrip_and_direct_tskit_donor_oracle(tmp_path):
    ts,roster=fixture();raw,*_=serialized(ts,roster)
    artifact=tmp_path/'handmade.whole_observed.pt';torch.save(raw,artifact)
    restored=torch.load(artifact,map_location='cpu',weights_only=True)
    cert=prepare(ts=ts,roster=roster,observation=restored)
    queries=[row.native_sample_node for row in roster if row.role!='pure_source_reference']
    refs=[row.native_sample_node for row in roster if row.role=='pure_source_reference']
    sigma=torch.tensor([.05,.1,.2],dtype=torch.float64)
    for graph,tree in enumerate(ts.trees()):
        times=torch.tensor([[tree.time(tree.mrca(q,r)) for r in refs] for q in queries],dtype=torch.float64)
        expected=torch.softmax(-torch.log1p(times)[None]/math.log1p(10000.)/sigma[:,None,None],-1)
        torch.testing.assert_close(cert.bank.donor_weights[graph],expected,rtol=0,atol=0)
    assert cert.provenance['original_tree_ids']==[0,1]
    assert cert.provenance['landmark_native_nodes']==[4]*4+[5]*4+[6]*4+[7]*4


def test_physical_anchor_readout_collision_with_unequal_rounding_fails_without_epsilon():
    ts,roster=fixture();tables=ts.dump_tables();tables.sequence_length=3200.
    left=tables.edges.left.copy();right=tables.edges.right.copy()
    tables.edges.left=np.where(left==0.,0.,1600.)
    tables.edges.right=np.where(right==500.,1600.,3200.)
    tables.sort();ts=tables.tree_sequence()
    left_bp,right_bp,offset=100000.,103200.,100000.
    raw,positions,indexes,_,_=serialized(ts,roster,left_bp,right_bp,offset)
    frozen=left_bp/2000000.+(torch.arange(32,dtype=torch.float64)+.5)*((right_bp/2000000.-left_bp/2000000.)/32)
    assert not torch.equal(frozen,positions/2000000.)
    cfg=load_census_biological_config_v1();cfg.update(fixture_mode=True,role_counts={'query':2,'mixed_reference_only':2,'pure_source_reference':4},references_per_class=1)
    with pytest.raises(ValueError,match='Physical anchor collision'):
        prepare_census_biological_bank_v1(ts,raw,roster,left_bp=left_bp,right_bp=right_bp,offset_bp=offset,
            positions_bp=positions,original_grid_indexes=indexes,source_bindings=bindings(),parent_id='handmade_train',split='train',tile_id=0,config=cfg)

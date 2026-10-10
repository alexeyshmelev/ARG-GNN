"""Independent handmade CPU tests. Mock admissions are fixture data only."""
from pathlib import Path
from copy import deepcopy
from dataclasses import fields
from datetime import datetime, timezone, timedelta
import importlib.util
import io,json,math,hashlib
import numpy as np
import pytest
import torch
import tskit
import arggnn.census_native_qualification_runner_v1 as runner
from arggnn.census_biological_bank_v1 import fingerprint_sha256_v1

ROOT=runner.ROOT
OWN=ROOT/'research/reviews/census_native_qualification_runner_v1_independent'
spec=importlib.util.spec_from_file_location('immutable_runner_source_tests',ROOT/'tests/test_census_native_qualification_runner_v1.py')
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)

@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)

def independent_ts():
    tables=tskit.TableCollection(sequence_length=2000000.)
    for age in (10.,60.,240.,300.,350.,450.,520.,620.):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age)
    for hold,(left,right) in enumerate(zip((0.,312550.,625050.,1125050.),(312550.,625050.,1125050.,2000000.))):
        def node(age):return tables.nodes.add_row(time=age)
        if hold==2:
            a,b=node(320.),node(340.);c=node(350.);d=node(700.);e=node(760.);f=node(820.);root=node(1100.)
            parents={0:a,1:a,2:b,3:b,a:c,b:c,c:d,4:d,5:e,6:e,e:f,7:f,d:root,f:root}
        else:
            pairs=((0,1),(2,3)) if hold!=1 else ((0,2),(1,3))
            a,b=node(320. if hold==0 else 330. if hold==1 else 360.),node(340. if hold==0 else 345. if hold==1 else 380.)
            c,d,e,f,root=node(700.),node(760.),node(820.),node(940.),node(1100.)
            r0,r1=(4,5) if hold!=1 else (5,4)
            parents={pairs[0][0]:a,pairs[0][1]:a,pairs[1][0]:b,pairs[1][1]:b,a:c,r0:c,b:d,r1:d,6:e,7:e,c:f,d:f,f:root,e:root}
            if hold==0:
                unary=node(400.);parents[b]=unary;parents[unary]=d
        for child,parent in parents.items():tables.edges.add_row(left,right,parent,child)
    for age in (2000.,2200.,2400.):tables.nodes.add_row(time=age)
    tables.sort();return tables.tree_sequence()

def envelope(tmp_path):
    old=author.handmade_ts;author.handmade_ts=independent_ts
    try:config,store,active,ref,put,finish,ts=author.fixture_store(tmp_path)
    finally:author.handmade_ts=old
    native=json.loads(store[active['refs']['native_manifest']['path']]);prepared=json.loads(store[active['refs']['prepared_manifest']['path']])
    for parent,bank in zip(native['replicates'],prepared['parents']):
        parent['post_five_iterations']['trees']=4
        completion=put(parent['completion_provenance']['path'],{k:v for k,v in parent.items() if k!='completion_provenance'})
        parent['completion_provenance']=completion;bank['relate_entry']=deepcopy(parent)
        active['observed_artifact_sha256'][completion['path']]=completion['sha256']
    author.rebind_metadata(store,active,put,native,prepared)
    return config,store,active,finish(active),put,finish,ts

def inspect_case(tmp_path):
    config,store,active,ref,put,finish,ts=envelope(tmp_path)
    result,record=runner.run_qualification_phase_v1(ref,phase='inspect',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__))
    assert result['status']=='inspection_complete_pending_root_freeze',result
    return config,store,active,put,finish,ts,result,record

def numerical_admission(active,put,config,result,tmp_path):
    active=deepcopy(active)
    active.update(job_id='independent-mock-numerical',phase='numerical',output_directory=str((tmp_path/'numerical').relative_to(ROOT)),inspection_manifest=put('fixtures/independent_frozen_inspection.json',result),expected_certificates={f"{c['parent_id']}:{c['tile_id']}":c['certificate_sha256'] for c in result['cells']},group_permutation=[3,1,0,2],initial_state_sha256_by_dtype={d:fingerprint_sha256_v1(runner._model(config,d).state_dict()) for d in config['dtypes']})
    return active

def probe_function(calls):
    def probe(cert,expected,config,permutation,states):
        assert cert.certificate_sha256==expected
        calls.append((cert.provenance['parent_id'],cert.provenance['tile_id']))
        wanted=[[d,c] for d in config['dtypes'] for c in runner.CONTROLS]
        return {'status':'qualified','performed_controls':wanted,'checks':[{'independent_phase_probe':True} for _ in wanted]}
    return probe

def load_cert(config,store,active,ts,parent_index=0,tile_index=1):
    reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__)
    paired=runner.read_manifests_v1(active,config,reader);parent,bank=paired[parent_index]
    loaded=runner.load_parent_v1(parent,bank,active,config,reader,tmp_directory=OWN/'tmp',native_decoder=lambda data:ts)
    cert,collision,lineage=runner.prepare_tile_v1(parent,bank,bank['tiles'][tile_index],loaded,active,config,reader)
    return cert,loaded,collision,lineage

def test_own_four_tree_full32_handoff_oracles_and_independent_f47(tmp_path):
    config,store,active,put,finish,ts,inspected,record=inspect_case(tmp_path)
    assert len(inspected['cells'])==32 and len(set(inspected['performed_cell_coverage']))==32
    assert all(c['structural']['older_R_proxy_count']==4 for c in inspected['cells'])
    assert all(min(c['structural']['excluded_unrelated_nodes'])>=3 for c in inspected['cells'])
    assert sum(c['num_trees'] for c in inspected['cells'])==44
    assert sum(c['event_collision_counts']['boundary_readout'] for c in inspected['cells'])==12
    calls=[];numerical=numerical_admission(active,put,config,inspected,tmp_path)
    qualified,_=runner.run_qualification_phase_v1(finish(numerical),phase='numerical',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__),numerical_probe=probe_function(calls))
    assert qualified['status']=='numerical_complete_no_fitting_or_predictive_claim',qualified
    assert calls[:2]==[('bio_paths_pilot_train_000',0),('bio_paths_pilot_train_000',1)] and len(set(calls))==32
    cert,loaded,collision,lineage=load_cert(config,store,active,ts)
    # Own direct native samples/cuts/donors and exact F47 source-row projection.
    assert runner.native_oracle_checks_v1(cert,loaded)['direct_native_donor_max_abs_error']==0.
    for graph,(tree_id,input) in enumerate(zip(cert.provenance['original_tree_ids'],cert.inputs)):
        tree=ts.at_index(tree_id);rows=cert.provenance['source_rows_by_tree'][graph]
        source=cert.provenance['native_vertex_by_source_row'][graph].tolist()
        union=set()
        for sample in ts.samples():
            node=int(sample)
            while node!=-1:union.add(node);node=tree.parent(node)
        assert set(source)==union
        assert set(input.native_vertex_index.tolist())<=union
        for k,node in enumerate(input.native_vertex_index.tolist()):
            row=int(rows[source.index(node)])
            assert torch.equal(input.features[k],cert.stored_sequence.x[row,list(range(7))+list(range(8,48))])
        q=[r.native_sample_node for r in loaded[1] if r.role!='pure_source_reference'];r=[x.native_sample_node for x in loaded[1] if x.role=='pure_source_reference']
        m=np.array([[tree.time(tree.mrca(a,b)) for b in r] for a in q])
        logits=-np.log1p(m)[None]/math.log1p(10000.)/np.array([.05,.1,.2])[:,None,None]
        exp=np.exp(logits-logits.max(axis=-1,keepdims=True));oracle=exp/exp.sum(axis=-1,keepdims=True)
        np.testing.assert_allclose(cert.bank.donor_weights[graph].numpy(),oracle,rtol=1e-12,atol=1e-14)
    summary=runner.structural_cell_v1(cert,hidden_dim=config['model']['hidden_dim'])
    assert summary['physical_span_weighted_query_group_size']=={'2':1.0}
    assert summary['right_hold_span_weighted_boundary_rho']=={str(1./3.):1.0}
    assert len(cert.native_trees[0].parent)==16 and len(cert.inputs[0].parent)==15
    assert cert.stored_sequence.positions[cert.stored_sequence.target_graph==1][0].item()==312550./2000000.
    (OWN/'own_four_tree_inspection_metrics.json').write_text(json.dumps({'fixture_only':True,'native_nodes':ts.num_nodes,'native_trees':ts.num_trees,'native_samples':ts.num_samples,'inspected_cells':len(inspected['cells']),'tree_holds':sum(c['num_trees'] for c in inspected['cells']),'boundary_readout_collisions':sum(c['event_collision_counts']['boundary_readout'] for c in inspected['cells']),'TRAIN_first_calls':calls[:2],'target_free_callback_cells':len(calls),'real_cell_summary':summary,'real_cell_collision':collision},indent=2)+'\n')

def test_own_four_tree_real_numerics_all_controls_dtypes(tmp_path):
    config,store,active,ref,put,finish,ts=envelope(tmp_path)
    cert,loaded,*_=load_cert(config,store,active,ts)
    states={d:fingerprint_sha256_v1(runner._model(config,d).state_dict()) for d in config['dtypes']}
    result=runner.numerical_cell_v1(cert,cert.certificate_sha256,config,torch.tensor([3,1,0,2]),states)
    assert len(result['checks'])==10 and all(c['positive_mass_min']>0 and c['bounded_source_max_abs']<=1.+config['tolerances'][c['dtype']]['output_atol'] and c['finite_disposable_AdamW'] for c in result['checks'])
    (OWN/'own_real_numerical_metrics.json').write_text(json.dumps(result,indent=2)+'\n')

@pytest.mark.parametrize('kind',['last_parent_array','last_parent_native','compiled_runtime','last_source','config_mutation','fixture_review_for_actual'])
def test_global_preflight_and_injected_decoder_guard(kind,tmp_path):
    config,store,active,ref,put,finish,ts=envelope(tmp_path)
    if kind=='last_parent_array':path=[p for p in active['observed_artifact_sha256'] if p.endswith('.whole_observed.pt')][-1];store[path]+=b'byte drift'
    elif kind=='last_parent_native':path=[p for p in active['observed_artifact_sha256'] if p.endswith('.trees')][-1];store[path]+=b'byte drift'
    elif kind=='compiled_runtime':store[config['runtime_files'][-1]]+=b'byte drift'
    elif kind=='last_source':store[config['required_source_paths'][-1]]+=b'byte drift'
    elif kind=='config_mutation':config['directional_epsilon']=1.;active['refs']['runner_config']=put('fixtures/altered_config.json',config);ref=finish(active)
    else:
        config['fixture_mode']=False;active['refs']['runner_config']=put('fixtures/altered_config.json',config);ref=finish(active)
    calls=[]
    def sentinel(data):calls.append(True);raise AssertionError('decoder reached')
    try:result,_=runner.run_qualification_phase_v1(ref,phase='inspect',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__),native_decoder=sentinel,observation_decoder=sentinel)
    except ValueError:pass
    else:assert result['status'].startswith('failed') and not result['performed_cell_coverage']
    assert not calls

@pytest.mark.parametrize('kind',['date','subject_pairing','derived_role','mapped_sample','labels','feature_marker','feature_columns','TRAIN_lineage','extra_target_path'])
def test_original_identity_feature_and_target_lineage_rejects(kind,tmp_path):
    config,store,active,ref,put,finish,ts=envelope(tmp_path)
    if kind in ('labels','feature_marker'):
        path=[p for p in active['observed_artifact_sha256'] if p.endswith('.whole_observed.pt')][0]
        raw=torch.load(io.BytesIO(store[path]),map_location='cpu',weights_only=True)
        if kind=='labels':raw['labels_in_input']=True
        else:raw['sequence']['x'][0,9]+=.1
        data=io.BytesIO();torch.save(raw,data);updated=put(path,data.getvalue());active['observed_artifact_sha256'][path]=updated['sha256']
        native=json.loads(store[active['refs']['native_manifest']['path']]);prepared=json.loads(store[active['refs']['prepared_manifest']['path']]);prepared['parents'][0]['tiles'][0]['whole_observed_graph']=updated
        author.rebind_metadata(store,active,put,native,prepared)
    elif kind in ('feature_columns','TRAIN_lineage'):
        feature=json.loads(store[active['refs']['feature_lineage']['path']])
        feature['feature_columns']=[*range(47)] if kind=='feature_columns' else feature['feature_columns']
        if kind=='TRAIN_lineage':feature['original_TRAIN_parent_ids']=config['parents'][2:]
        active['refs']['feature_lineage']=put('fixtures/changed_feature.json',feature)
    elif kind=='extra_target_path':active['observed_artifact_sha256']['fixtures/targets/hidden.npy']='a'*64
    else:
        native=json.loads(store[active['refs']['native_manifest']['path']]);prepared=json.loads(store[active['refs']['prepared_manifest']['path']])
        parent=native['replicates'][0];bank=prepared['parents'][0]
        original=json.loads(store[parent['original_observed_sample_manifest']['path']]);derived=json.loads(store[parent['observed_sample_manifest']['path']])
        if kind=='date':original[0]['age_generations']+=1.;derived[0]['age_generations']+=1.
        elif kind=='derived_role':derived[0]['role']='mixed_reference_only'
        elif kind=='mapped_sample':parent['sample_index_to_inferred_tree_node_id'][0]['inferred_tree_node_id']=1
        else:
            # Exercise actual-mode diploid pairing directly with entirely handmade8 rows.
            config['fixture_mode']=False
            reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__)
            with pytest.raises(ValueError,match='diploid subject'):runner.load_parent_v1(parent,bank,active,config,reader,tmp_directory=OWN/'tmp',native_decoder=lambda data:ts)
            return
        source=put(parent['original_observed_sample_manifest']['path'],original);parent['original_observed_sample_manifest']=source
        current=parent['observed_sample_manifest'];new=dict(put(current['path'],derived),source_manifest=source,rows=8);parent['observed_sample_manifest']=new;parent['post_five_iterations']['observed_sample_manifest']=new;parent['post_five_iterations']['sample_index_to_inferred_tree_node_id']=parent['sample_index_to_inferred_tree_node_id']
        completion=put(parent['completion_provenance']['path'],{k:v for k,v in parent.items() if k!='completion_provenance'});parent['completion_provenance']=completion;bank['relate_entry']=deepcopy(parent)
        for record in (source,new,completion):active['observed_artifact_sha256'][record['path']]=record['sha256']
        author.rebind_metadata(store,active,put,native,prepared)
    result,_=runner.run_qualification_phase_v1(finish(active),phase='inspect',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__))
    assert result['status'].startswith('failed') and result['failed_cells'] or result['status']=='failed_partial_preserved'

@pytest.mark.parametrize('kind',['lineage_drift','count_drift','structural_drift','collision_drift','missing_coverage','bad_permutation','same_directory','partial_probe'])
def test_numerical_freeze_guards(kind,tmp_path):
    config,store,active,put,finish,ts,inspection,_=inspect_case(tmp_path)
    numerical=numerical_admission(active,put,config,inspection,tmp_path)
    if kind in ('lineage_drift','structural_drift','collision_drift','count_drift','missing_coverage'):
        inspection=deepcopy(inspection)
        if kind=='lineage_drift':inspection['cells'][0]['decoded_lineage']['target_columns'].reverse()
        elif kind=='structural_drift':inspection['cells'][0]['structural']['sample_count']+=1
        elif kind=='collision_drift':inspection['cells'][0]['event_collision_counts']['boundary_readout']+=1
        elif kind=='count_drift':inspection['cells'][0]['num_trees']+=1
        else:inspection['cells'].pop()
        numerical['inspection_manifest']=put('fixtures/tampered_inspection.json',inspection)
    elif kind=='bad_permutation':numerical['group_permutation']=[0,0,1,2]
    elif kind=='same_directory':numerical['output_directory']=active['output_directory']
    calls=[]
    def partial(cert,expected,config,permutation,states):calls.append(True);return {'status':'qualified','performed_controls':[['float64','proposed']],'checks':[{}]}
    if kind=='same_directory':
        with pytest.raises(ValueError,match='exclusive'):runner.run_qualification_phase_v1(finish(numerical),phase='numerical',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__),numerical_probe=probe_function(calls))
    else:
        result,_=runner.run_qualification_phase_v1(finish(numerical),phase='numerical',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__),numerical_probe=partial if kind=='partial_probe' else probe_function(calls))
        assert result['status']=='failed_partial_preserved' and not result['performed_cell_coverage']
        assert len(calls)==(1 if kind=='partial_probe' else 0)

@pytest.mark.parametrize('kind',['compiled_identity','reservation_concurrent','snapshot_age','review_unqualified','source_missing','future_admission','long_admission','wrong_thread_env'])
def test_resource_runtime_review_admission_rejects(kind,tmp_path,monkeypatch):
    config,store,active,ref,put,finish,ts=envelope(tmp_path)
    if kind=='compiled_identity':
        value=json.loads(store[active['refs']['runtime']['path']]);value['identity']['torch']='0';active['refs']['runtime']=put('fixtures/changed_runtime.json',value)
    elif kind in ('reservation_concurrent','snapshot_age'):
        storage=json.loads(store[active['refs']['storage']['path']])
        if kind=='reservation_concurrent':storage['concurrent_reserve_bytes']=450_000_000_000
        else:
            snapshot=json.loads(store[storage['snapshot_ref']['path']]);snapshot['created_at_utc']=(datetime.now(timezone.utc)-timedelta(hours=2)).isoformat();storage['snapshot_ref']=put('fixtures/old_snapshot.json',snapshot)
        active['refs']['storage']=put('fixtures/changed_storage.json',storage)
    elif kind=='review_unqualified':
        review=json.loads(store[active['refs']['runner_review']['path']]);full=json.loads(store[review['review_receipt_ref']['path']]);full['status']='BLOCKED';review['review_receipt_ref']=put('fixtures/blocked_full_review.json',full);active['refs']['runner_review']=put('fixtures/blocked_review.json',review)
    elif kind=='source_missing':active['source_sha256'].pop('src/arggnn/census_native_qualification_runner_v1.py')
    elif kind=='future_admission':active['created_at_utc']=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    elif kind=='long_admission':active['expires_at_utc']=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
    else:monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError):runner.admit_runner_v1(put('fixtures/changed_admission.json',active),phase='inspect',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__))


def test_admission_inspection_cannot_self_supply_numerical_freeze(tmp_path):
    config,store,active,ref,put,finish,ts=envelope(tmp_path);active['expected_certificates']={'x':'f'*64}
    with pytest.raises(ValueError,match='Inspection cannot grant'):runner.admit_runner_v1(finish(active),phase='inspect',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__))


def test_default_model_state_fingerprints_independently_match_author():
    config=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text())
    states={d:fingerprint_sha256_v1(runner._model(config,d).state_dict()) for d in config['dtypes']}
    receipt=json.loads((ROOT/'research/reviews/census_native_qualification_runner_v1_author/receipt.json').read_text())
    assert states==receipt['default_shipped_model_initial_state_sha256_by_dtype']
    (OWN/'default_initial_state_sha256_by_dtype.json').write_text(json.dumps(states,indent=2,sort_keys=True)+'\n')

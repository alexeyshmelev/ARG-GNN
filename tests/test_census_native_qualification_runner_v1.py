"""Handmade admitted-byte stores only. Fake review/job receipts are fixture data."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from pathlib import Path
from dataclasses import replace
import hashlib
import io
import json
import os
import numpy as np
import pytest
import torch
import tskit
from arggnn.biological_candidate import pack_observed_tile
from arggnn.census_native_qualification_runner_v1 import (
    ROOT,CONTROLS,VerifiedReaderV1,BoundedEvidenceOutputV1,runtime_identity_v1,
    admit_runner_v1,run_qualification_phase_v1,numerical_cell_v1,_model,
    digest_bytes_v1,json_bytes_v1,event_collision_counts_v1)
from arggnn.census_biological_bank_v1 import fingerprint_sha256_v1

@pytest.fixture(autouse=True)
def cpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)


def handmade_ts():
    tables=tskit.TableCollection(sequence_length=2000000.)
    for age in [0.,100.,200.,300.,400.,500.,550.,600.]:tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age)
    tables.nodes.add_row(time=300.);tables.nodes.add_row(time=330.);tables.nodes.add_row(time=700.)
    tables.nodes.add_row(time=750.);tables.nodes.add_row(time=850.);tables.nodes.add_row(time=900.);tables.nodes.add_row(time=1000.)
    tables.nodes.add_row(time=330.);tables.nodes.add_row(time=340.);tables.nodes.add_row(time=700.)
    tables.nodes.add_row(time=750.);tables.nodes.add_row(time=850.);tables.nodes.add_row(time=900.);tables.nodes.add_row(time=1000.)
    tables.nodes.add_row(time=3000.)  # unused global node
    first={0:8,1:8,2:9,3:9,8:10,4:10,9:11,5:11,6:12,7:12,10:13,11:13,13:14,12:14}
    second={0:15,2:15,1:16,3:16,15:17,5:17,16:18,4:18,6:19,7:19,17:20,18:20,20:21,19:21}
    for left,right,parents in ((0.,875050.,first),(875050.,2000000.,second)):
        for child,parent in parents.items():tables.edges.add_row(left,right,parent,child)
    tables.sort();return tables.tree_sequence()


def fixture_store(tmp_path):
    config=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text())
    config.update(fixture_mode=True,role_counts={'query':2,'mixed_reference_only':2,'pure_source_reference':4},references_per_class=1)
    config['model']=dict(config['model'],hidden_dim=5,latent_dim=2,num_layers=1)
    store={};refs={}
    def put(path,data):
        if not isinstance(data,bytes):data=json_bytes_v1(data)
        store[path]=data;return {'path':path,'sha256':digest_bytes_v1(data)}
    config_ref=put('fixtures/runner_config.json',config)
    source={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in config['required_source_paths']}
    own='src/arggnn/census_native_qualification_runner_v1.py';source[own]=hashlib.sha256((ROOT/own).read_bytes()).hexdigest()
    for path in source:store[path]=(ROOT/path).read_bytes()
    runtime_files={path:put(path,b'handmade runtime file '+path.encode())['sha256'] for path in config['runtime_files']}
    runtime_ref=put('fixtures/runtime.json',{'schema':'census_native_CPU_runtime_v1','identity':runtime_identity_v1(),'runtime_file_sha256':runtime_files})
    full_review_ref=put('fixtures/mock_full_review.json',{'status':'qualified_fixture_only','fixture_mode':True,'artifact_sha256':{'src/arggnn/census_native_qualification_runner_v1.py':source[own],'configs/census_native_qualification_runner_v1.json':config_ref['sha256']}})
    review_ref=put('fixtures/mock_independent_review.json',{'schema':'census_native_qualification_review_binding_v1','status':'qualified_source_handmade_CPU_only','runner_source_sha256':source[own],'runner_config_sha256':config_ref['sha256'],'actual_admission':False,'review_receipt_ref':full_review_ref})
    ts=handmade_ts();treefile=tmp_path/'only_handmade_native.trees';ts.dump(treefile);treebytes=treefile.read_bytes()
    positions=np.arange(50.,2000000.,100.);buffer=io.BytesIO();np.savez(buffer,positions_bp=positions,bin_edges_bp=np.arange(0.,2000001.,100.));gridbytes=buffer.getvalue()
    native={'kind':'biological_paths_relate_v1','status':'four_development_parents_inferred_pre_and_post_five_iterations','primary_condition':'post_five_size_iterations_primary','truth_or_oracle_read':False,'implementation_sha256':config['source_sha256']['scripts/infer_biological_paths_relate_v1.py'],'config_sha256':config['source_sha256']['configs/biological_paths_relate_v1.json'],'replicates':[]}
    prepared={'kind':'biological_candidate_prepared_v1','status':'complete_observed_only_Q184_R96_actualQ120','truth_opened':False,'labels_in_features_or_banks':False,'config_sha256':config['source_sha256']['configs/biological_candidate_pilot_v1.json'],'source_sha256':config['source_sha256'],'parents':[]}
    artifacts={}
    for parent_id in config['parents']:
        split='train' if '_train_' in parent_id else 'val'
        original=[dict(sample_index=i,sample_id=f'toy_{i}',haplotype=i%2,observed_node_id=i,observed_cohort=config['classes'][i-4] if i>=4 else 'present_day',age_generations=ts.node(i).time,role='query' if i<2 else 'mixed_reference_only' if i<4 else 'pure_source_reference',observed_reference_class=None if i<4 else i-4) for i in range(8)]
        source_ref=put(f'fixtures/{parent_id}/source_roster.json',original)
        derived=[dict(r,source_observed_node_id=r['observed_node_id'],inferred_tree_node_id=r['observed_node_id']) for r in original]
        derived_ref=dict(put(f'fixtures/{parent_id}/derived_roster.json',derived),source_manifest=source_ref,rows=8)
        tree_ref=dict(put(f'fixtures/{parent_id}/observed.trees',treebytes),condition='post_five_size_iterations_primary')
        grid_ref=dict(put(f'fixtures/{parent_id}/grid.npz',gridbytes),total=20000)
        coordinate={'normalization_length_bp':2000000,'original_parent_interval_bp0':[0,2000000],'physical_coordinate_transform':'physical_original = local_tree_position + coverage_left','uncovered_tails':'excluded; no extrapolation','coverage_original_bp0':[0.,2000000.],'local_to_original_offset_bp':0.,'retained_fixed100bp_indexes':list(range(20000)),'fixed_grid':grid_ref}
        parent={'parent_id':parent_id,'split':split,'status':'inferred_both_conditions','truth_or_oracle_read':False,'inferred_tree':tree_ref,'observed_sample_manifest':derived_ref,'original_observed_sample_manifest':source_ref,'coordinate_contract':coordinate,'sample_index_to_inferred_tree_node_id':[{'sample_index':i,'inferred_tree_node_id':i} for i in range(8)]}
        parent['post_five_iterations']={'clean_tree':tree_ref['path'],'clean_tree_sha256':tree_ref['sha256'],'sample_index_to_inferred_tree_node_id':parent['sample_index_to_inferred_tree_node_id'],'observed_sample_manifest':derived_ref,'coverage_original_bp0':[0.,2000000.],'local_to_original_offset_bp':0.,'sequence_length_local_bp':2000000.,'retained_fixed100bp_indexes':list(range(20000)),'trees':2}
        completion=put(f'fixtures/{parent_id}/completion.json',parent);parent['completion_provenance']=completion;native['replicates'].append(parent)
        bank={'parent_id':parent_id,'split':split,'relate_entry':deepcopy(parent),'dynamics_sample_indexes':[0,1,2,3],'actual_query_sample_indexes':[0,1],'reference_sample_indexes':[4,5,6,7],'actual_query_output_indexes':[0,1],'query_node_ids':[0,1],'dynamics_query_node_ids':[0,1,2,3],'coverage_original_bp0':[0.,2000000.],'retained_fixed100bp_indexes':list(range(20000)),'positions_bp':positions.tolist(),'tiles':[]}
        pack_parent={'observation':{'nodes':np.arange(8),'offset':0.},'dynamics_slots':np.arange(4),'query_slots':np.arange(2),'reference_slots':np.arange(4,8),'samples':original,'positions':positions}
        for tile in range(8):
            left,right=tile*250000.,(tile+1)*250000.;seq,indexes,tree_ids,landmarks=pack_observed_tile(ts,pack_parent,left,right)
            output=io.BytesIO();torch.save({'sequence':seq.as_dict(),'labels_in_input':False},output)
            record=put(f'fixtures/{parent_id}/tile_{tile}.whole_observed.pt',output.getvalue())
            bank['tiles'].append({'tile_id':tile,'physical_interval_bp0':[left,right],'normalized_interval':[left/2000000.,right/2000000.],'original_grid_indexes':indexes.tolist(),'position_indexes':indexes.tolist(),'original_tree_ids':tree_ids,'landmarks_observed_node_ids':landmarks,'all_tree_nodes_retained':True,'all280_leaves_retained':True,'mixed_query_flag':0,'query_dynamics':4,'pure_reference_sources':4,'supervised_outputs':2,'mixed_latent_outputs':2,'whole_observed_graph':record})
            artifacts[record['path']]=record['sha256']
        for record in (source_ref,derived_ref,tree_ref,grid_ref,completion):artifacts[record['path']]=record['sha256']
        prepared['parents'].append(bank)
    native_ref=put('fixtures/native_manifest.json',native);prepared['observed_manifest_sha256']=native_ref['sha256'];prepared_ref=put('fixtures/prepared_manifest.json',prepared)
    feature={'schema':'census_original_feature_lineage_v1','prepared_manifest_sha256':prepared_ref['sha256'],'producer_source_sha256':config['source_sha256']['src/arggnn/biological_candidate.py'],'reference_adapter_sha256':config['source_sha256']['src/arggnn/reference_encoder.py'],'feature_columns':list(range(7))+list(range(8,48)),'scaler':'identity_no_fitted_scaler','original_TRAIN_parent_ids':[p for p in config['parents'] if '_train_' in p],'labels_in_input':False}
    feature_ref=put('fixtures/feature_lineage.json',feature)
    admission={'schema':'census_native_qualification_admission_v1','status':'active','job_id':'handmade-inspection','phase':'inspect','created_at_utc':datetime.now(timezone.utc).isoformat(),'expires_at_utc':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),'refs':{'runner_config':config_ref,'runtime':runtime_ref,'runner_review':review_ref,'native_manifest':native_ref,'prepared_manifest':prepared_ref,'feature_lineage':feature_ref},'source_sha256':source,'observed_artifact_sha256':artifacts,'output_directory':str((tmp_path/'inspect').relative_to(ROOT)),'expected_certificates':{},'inspection_manifest':None,'initial_state_sha256_by_dtype':{},'group_permutation':[]}
    def finish(admission):
        snapshot=put(f"fixtures/{admission['job_id']}_snapshot.json",{'schema':'census_native_root_storage_snapshot_v1','status':'complete','returncode':0,'project_root':str(ROOT),'used_project_bytes':123,'created_at_utc':datetime.now(timezone.utc).isoformat()})
        admission['refs']['storage']=put(f"fixtures/{admission['job_id']}_storage.json",{'schema':'census_native_storage_reservation_v1','status':'active','job_id':admission['job_id'],'used_project_bytes':123,'reserved_new_bytes':1000000000,'concurrent_reserve_bytes':0,'snapshot_ref':snapshot,'project_limit_bytes':500000000000,'output_directory':admission['output_directory'],'created_at_utc':admission['created_at_utc'],'expires_at_utc':admission['expires_at_utc']})
        admission['refs']['launcher']=put(f"fixtures/{admission['job_id']}_launcher.json",{'schema':'census_native_CPU_launcher_v1','job_id':admission['job_id'],'python':'.runtime/env/bin/python','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','GPU_admitted':False})
        return put(f"fixtures/{admission['job_id']}_admission.json",admission)
    ref=finish(admission)
    return config,store,admission,ref,put,finish,ts


def test_complete_two_phase_handmade_32_tile_handoff_and_coverage(tmp_path):
    config,store,admission,ref,put,finish,ts=fixture_store(tmp_path)
    result,record=run_qualification_phase_v1(ref,phase='inspect',reader=VerifiedReaderV1(read_bytes=store.__getitem__))
    assert result['status']=='inspection_complete_pending_root_freeze',result.get('error')
    assert len(result['cells'])==32 and len(result['performed_cell_coverage'])==32
    assert sum(c['event_collision_counts']['boundary_readout'] for c in result['cells'])==4
    assert all(c['structural']['excluded_unrelated_nodes'] for c in result['cells'])
    assert result['prespecified_TRAIN_numerical_cells']['maximum']['tile_id']==3
    frozen=put('fixtures/root_frozen_inspection.json',result)
    active=deepcopy(admission);active.update(job_id='handmade-numerical',phase='numerical',output_directory=str((tmp_path/'numerical').relative_to(ROOT)),inspection_manifest=frozen,expected_certificates={f"{c['parent_id']}:{c['tile_id']}":c['certificate_sha256'] for c in result['cells']},group_permutation=[2,3,1,0],initial_state_sha256_by_dtype={dtype:fingerprint_sha256_v1(_model(config,dtype).state_dict()) for dtype in config['dtypes']})
    calls=[]
    def probe(cert,expected,config,permutation,states):
        assert cert.certificate_sha256==expected;calls.append((cert.provenance['parent_id'],cert.provenance['tile_id']))
        wanted=[[d,c] for d in config['dtypes'] for c in CONTROLS]
        return {'status':'qualified','performed_controls':wanted,'checks':[{'fixture_probe':True} for _ in wanted]}
    out,_=run_qualification_phase_v1(finish(active),phase='numerical',reader=VerifiedReaderV1(read_bytes=store.__getitem__),numerical_probe=probe)
    assert out['status']=='numerical_complete_no_fitting_or_predictive_claim',out.get('error')
    assert len(calls)==32 and len(set(calls))==32
    assert calls[:2]==[('bio_paths_pilot_train_000',0),('bio_paths_pilot_train_000',3)]


def test_hash_mismatch_never_reaches_native_decoder_and_all_attempts_preserved(tmp_path):
    _,store,admission,ref,*_=fixture_store(tmp_path)
    path=next(p for p in admission['observed_artifact_sha256'] if p.endswith('.whole_observed.pt'));store[path]+=b'changed'
    calls=[]
    def decode(data):calls.append(data);raise AssertionError('decoded before byte verification')
    out,_=run_qualification_phase_v1(ref,phase='inspect',reader=VerifiedReaderV1(read_bytes=store.__getitem__),native_decoder=decode)
    assert out['status']=='failed_partial_preserved' and out['performed_cell_coverage']==[]
    assert calls==[] and out['cells']==[]


@pytest.mark.parametrize('change',['expired','missing_source','runtime','storage','scaler','truth','review','missing_tile'])
def test_phase_admission_and_lineage_fail_closed(change,tmp_path):
    _,store,active,ref,put,finish,_=fixture_store(tmp_path)
    if change=='expired':active['expires_at_utc']='2020-01-01T00:00:00+00:00'
    elif change=='missing_source':active['source_sha256'].pop(next(iter(active['source_sha256'])))
    elif change=='runtime':store[active['refs']['runtime']['path']]+=b'bad'
    elif change=='storage':active['output_directory']='fixtures/different_output'
    elif change=='review':store[active['refs']['runner_review']['path']]+=b'bad'
    elif change=='scaler':
        value=json.loads(store[active['refs']['feature_lineage']['path']]);value['scaler']='fit_on_VAL';active['refs']['feature_lineage']=put('fixtures/wrong_scaler.json',value)
    elif change=='truth':
        value=json.loads(store[active['refs']['native_manifest']['path']]);value['truth_or_oracle_read']=True;active['refs']['native_manifest']=put('fixtures/truth_manifest.json',value)
    else:
        value=json.loads(store[active['refs']['prepared_manifest']['path']]);value['parents'][0]['tiles'].pop();active['refs']['prepared_manifest']=put('fixtures/missing_tile_manifest.json',value)
    if change in ('expired','missing_source','runtime','storage','review'):
        # Storage mismatch uses its already bound reservation, not a regenerated one.
        if change=='storage':ref=put('fixtures/changed_active.json',active)
        else:ref=finish(active)
        with pytest.raises(ValueError):run_qualification_phase_v1(ref,phase='inspect',reader=VerifiedReaderV1(read_bytes=store.__getitem__))
    else:
        out,_=run_qualification_phase_v1(finish(active),phase='inspect',reader=VerifiedReaderV1(read_bytes=store.__getitem__))
        assert out['status']=='failed_partial_preserved' and out['performed_cell_coverage']==[]


def test_project_paths_and_bounded_exclusive_output(tmp_path):
    reader=VerifiedReaderV1(read_bytes=lambda p:b'a')
    for path in ('../outside','/tmp/not_project','.'):
        with pytest.raises(ValueError):reader.relative(path)
    output=BoundedEvidenceOutputV1(tmp_path/'small',limit_bytes=2)
    with pytest.raises(ValueError,match='cap'):output.write('proof.json',{'a':1})
    with pytest.raises(ValueError,match='exclusive'):BoundedEvidenceOutputV1(tmp_path/'small')


def test_actual_small_handmade_numerical_engine_five_controls_float64_float32():
    import importlib.util
    spec=importlib.util.spec_from_file_location('sealed_handmade_adapter_tests',ROOT/'tests/test_census_biological_bank_v1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ts=handmade_ts();tables=ts.dump_tables();tables.sequence_length=1000.
    tables.edges.left=np.where(tables.edges.left==0.,0.,500.);tables.edges.right=np.where(tables.edges.right==875050.,500.,1000.);tables.sort()
    _,roster=module.fixture();cert=module.prepare(ts=tables.tree_sequence(),roster=roster)
    assert cert.bank.donor_weights[0,0,0].max()-cert.bank.donor_weights[0,0,0].min()>.05
    config=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text());config['model']=dict(config['model'],hidden_dim=5,latent_dim=2,num_layers=1)
    states={d:fingerprint_sha256_v1(_model(config,d).state_dict()) for d in config['dtypes']}
    result=numerical_cell_v1(cert,cert.certificate_sha256,config,torch.tensor([2,3,1,0]),states)
    assert len(result['performed_controls'])==10 and len(result['checks'])==10
    assert all(r['finite_disposable_AdamW'] and r['positive_mass_min']>0 for r in result['checks'])


def rebind_metadata(store,active,put,native,prepared):
    native_ref=put('fixtures/rebound_native.json',native);prepared['observed_manifest_sha256']=native_ref['sha256']
    prepared_ref=put('fixtures/rebound_prepared.json',prepared)
    feature=json.loads(store[active['refs']['feature_lineage']['path']]);feature['prepared_manifest_sha256']=prepared_ref['sha256']
    active['refs'].update(native_manifest=native_ref,prepared_manifest=prepared_ref,feature_lineage=put('fixtures/rebound_feature.json',feature))


@pytest.mark.parametrize('change',['target_columns','grid_indexes','post_condition','unadmitted_extra'])
def test_original_order_and_condition_metadata_rejected(change,tmp_path):
    _,store,active,ref,put,finish,_=fixture_store(tmp_path)
    native=json.loads(store[active['refs']['native_manifest']['path']]);prepared=json.loads(store[active['refs']['prepared_manifest']['path']])
    if change=='target_columns':prepared['parents'][0]['actual_query_output_indexes']=[1,0]
    elif change=='grid_indexes':prepared['parents'][0]['tiles'][0]['original_grid_indexes'].reverse()
    elif change=='post_condition':native['replicates'][0]['inferred_tree']['condition']='pre_size_iterations';prepared['parents'][0]['relate_entry']=native['replicates'][0]
    else:active['observed_artifact_sha256']['fixtures/truth_target.npy']='f'*64
    rebind_metadata(store,active,put,native,prepared)
    out,_=run_qualification_phase_v1(finish(active),phase='inspect',reader=VerifiedReaderV1(read_bytes=store.__getitem__))
    assert out['status'].startswith('failed')
    assert not out['biological_targets_opened'] and not out['fit_performed']


def test_source_roster_extra_truth_field_rejected_before_native_decode(tmp_path):
    _,store,active,_,put,finish,_=fixture_store(tmp_path)
    native=json.loads(store[active['refs']['native_manifest']['path']]);prepared=json.loads(store[active['refs']['prepared_manifest']['path']])
    for parent,bank in zip(native['replicates'],prepared['parents']):
        old=parent['original_observed_sample_manifest'];rows=json.loads(store[old['path']]);rows[0]['truth_ancestry']=.5
        new=put(old['path'],rows);parent['original_observed_sample_manifest']=new
        parent['observed_sample_manifest']['source_manifest']=new
        parent['post_five_iterations']['observed_sample_manifest']=parent['observed_sample_manifest']
        completion=put(parent['completion_provenance']['path'],{k:v for k,v in parent.items() if k!='completion_provenance'})
        parent['completion_provenance']=completion;bank['relate_entry']=deepcopy(parent)
        active['observed_artifact_sha256'][new['path']]=new['sha256'];active['observed_artifact_sha256'][completion['path']]=completion['sha256']
    rebind_metadata(store,active,put,native,prepared)
    calls=[]
    def decode(data):calls.append(True);raise AssertionError('truth roster decoded')
    out,_=run_qualification_phase_v1(finish(active),phase='inspect',reader=VerifiedReaderV1(read_bytes=store.__getitem__),native_decoder=decode)
    assert out['status']=='failed_complete_inventory_preserved' and len(out['failed_cells'])==32 and not calls


def test_numerical_missing_or_stale_frozen_digest_never_calls_probe(tmp_path):
    config,store,active,ref,put,finish,_=fixture_store(tmp_path)
    inspected,_=run_qualification_phase_v1(ref,phase='inspect',reader=VerifiedReaderV1(read_bytes=store.__getitem__))
    assert inspected['status']=='inspection_complete_pending_root_freeze'
    active.update(job_id='blocked-numerical',phase='numerical',output_directory=str((tmp_path/'blocked').relative_to(ROOT)),inspection_manifest=put('fixtures/frozen_inspection.json',inspected),expected_certificates={f"{c['parent_id']}:{c['tile_id']}":c['certificate_sha256'] for c in inspected['cells']},group_permutation=[2,3,1,0],initial_state_sha256_by_dtype={d:fingerprint_sha256_v1(_model(config,d).state_dict()) for d in config['dtypes']})
    active['expected_certificates'][next(iter(active['expected_certificates']))]='f'*64
    calls=[]
    def probe(*args):calls.append(True);raise AssertionError('stale probe called')
    out,_=run_qualification_phase_v1(finish(active),phase='numerical',reader=VerifiedReaderV1(read_bytes=store.__getitem__),numerical_probe=probe)
    assert out['status']=='failed_partial_preserved' and not calls and not out['performed_cell_coverage']


def test_wrong_initial_state_and_mutated_donor_fail_before_unit_forward():
    import importlib.util
    spec=importlib.util.spec_from_file_location('sealed_handmade_runner_negative',ROOT/'tests/test_census_biological_bank_v1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text());config['model']=dict(config['model'],hidden_dim=5,latent_dim=2,num_layers=1)
    cert=module.prepare();states={d:fingerprint_sha256_v1(_model(config,d).state_dict()) for d in config['dtypes']};wrong=dict(states,float64='f'*64)
    with pytest.raises(ValueError,match='initial model state'):numerical_cell_v1(cert,cert.certificate_sha256,config,torch.tensor([2,3,1,0]),wrong)
    digest=cert.certificate_sha256;cert.bank.donor_weights[0,0,0,0]+=.1
    with pytest.raises(ValueError,match='Changed/stale'):numerical_cell_v1(cert,digest,config,torch.tensor([2,3,1,0]),states)


def test_default_filesystem_reader_checks_handmade_bytes_and_rejects_symlink(tmp_path):
    path=tmp_path/'handmade_reader.bin';data=b'only synthetic fixture bytes';path.write_bytes(data)
    record={'path':str(path.relative_to(ROOT)),'sha256':digest_bytes_v1(data)}
    reader=VerifiedReaderV1();assert reader.read(record)==data
    path.write_bytes(data+b'changed')
    with pytest.raises(ValueError,match='source bytes changed'):reader.read(record)
    link=tmp_path/'handmade_reader_link.bin';link.symlink_to(path)
    with pytest.raises(ValueError,match='Invalid'):reader.read({'path':str(link.relative_to(ROOT)),'sha256':digest_bytes_v1(path.read_bytes())})


def test_full280_single_tree_numeric_qualification_has_structurally_inactive_eta():
    import importlib.util
    spec=importlib.util.spec_from_file_location('sealed_full_panel_runner_fixture',ROOT/'tests/test_census_biological_bank_v1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cert=module.prepare(full=True)
    assert len(cert.native_trees)==1 and len(cert.bank.query_ids)==184 and len(cert.bank.reference_ids)==96
    config=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text());config['model']=dict(config['model'],hidden_dim=5,latent_dim=2,num_layers=1)
    states={d:fingerprint_sha256_v1(_model(config,d).state_dict()) for d in config['dtypes']}
    result=numerical_cell_v1(cert,cert.certificate_sha256,config,torch.arange(183,-1,-1),states)
    assert len(result['checks'])==10 and all(c['finite_disposable_AdamW'] for c in result['checks'])
    for dtype in ('float64','float32'):
        rows={c['control']:c for c in result['checks'] if c['dtype']==dtype}
        assert rows['proposed']['active_VJP_tensors']==rows['eta0']['active_VJP_tensors']

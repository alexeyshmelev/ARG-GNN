"""Integrity, exact clipped-tile caching and external-supervision contracts."""
from pathlib import Path
import gzip
import json
import math
from unittest.mock import patch

import pytest
import torch
from arggnn.data import GraphSequence
from arggnn.reference_encoder import ReferenceFeatureAdapter
from arggnn.kron_interface import precompute_leaf_interfaces, with_query_reference_cache
from arggnn.train_study import load_cases
from arggnn.boundary_dataset import (prepare_boundary_dataset,load_boundary_dataset,
    load_boundary_cases,sha256_file,StorageAdmission,scan_repository_storage)


def fixture_source(root: Path, *, clipped=True, unsorted=False, zero_unscored=False):
    adapter_options={'reference_columns':[0,1],'time_column':4,'branch_column':5,
                     'time_scale':10000.,'kernel_scales':[.05,.1,.2]}
    settings={'master_seed':321,'splits':{'train':1,'val':1,'test':0},
              'windows':2,'reference_adapter':adapter_options}
    root.mkdir(parents=True)
    protocol=root/'source_protocol.json'
    protocol.write_text(json.dumps(settings,indent=2)+'\n')
    records=[]
    for split in ('train','val'):
        xs,edges,groups,ids=[],[],[],[]
        offset=0
        for graph,root_time in enumerate((2.,3.,5.)):
            root_time+=.5*(split=='val')
            ages=torch.tensor([.6,0.,.4,.2,root_time],dtype=torch.float64)
            local_ids=torch.tensor([30,10,20,11,-1])
            local_edges=torch.tensor([[0,1,2,3],[4,4,4,4]])
            if zero_unscored and graph==0 and split=='train':
                ages=torch.cat((ages,ages[:1]))
                local_ids=torch.cat((local_ids,torch.tensor([-1])))
                local_edges=torch.tensor([[0,5,1,2,3],[5,4,4,4,4]])
            n=len(ages)
            x=torch.zeros(n,6,dtype=torch.float64)
            x[2,0]=1.;x[0,1]=1.;x[[1,3],2]=1.;x[:4,3]=1.
            x[:,4]=torch.log1p(ages)/math.log1p(10000.)
            branches=torch.zeros(n,dtype=torch.float64)
            for u,v in local_edges.T.tolist():
                branches[u]=ages[v]-ages[u]
            x[:,5]=torch.log1p(branches)/math.log1p(10000.)
            xs.append(x);ids.append(local_ids);groups.extend([graph]*n)
            bidirected=torch.cat((local_edges,local_edges.flip(0)),dim=1)
            edges.append(bidirected+offset);offset+=n
        if clipped:
            boundaries=[.1,.2 if zero_unscored else .3,.6,.9]
            original=torch.tensor([5,6,10] if zero_unscored else [2,5,6,10])
            positions=(original.double()+.5)/12
        else:
            boundaries=[0.,.3,.6,1.]
            original=torch.arange(4)
            positions=(original.double()+.5)/4
        if unsorted:
            permutation=torch.tensor([2,0,3,1])
            positions=positions[permutation];original=original[permutation]
        intervals=torch.tensor(list(zip(boundaries[:-1],boundaries[1:])),dtype=torch.float64)
        target=((positions[:,None]>=intervals[None,:,0])&(positions[:,None]<intervals[None,:,1])).long().argmax(-1)
        sequence=GraphSequence(torch.cat(xs),torch.cat(edges,dim=1),torch.tensor(groups),torch.cat(ids),
            torch.zeros(offset,2,dtype=torch.float64),intervals,torch.tensor([11,10]),positions,target,
            torch.tensor([boundaries[0],boundaries[-1]],dtype=torch.float64))
        sequence.validate()
        pcount=positions.numel()
        row=torch.arange(pcount)
        labels=torch.stack(((row//2)%2,row%2),dim=1)
        baseline=torch.full((pcount,2,2),.5,dtype=torch.float64)
        seeds={'ancestry':37+(split=='val'),'mutations':71+(split=='val')}
        feature_names=['reference_SRC0','reference_SRC1','is_query_leaf','is_leaf','log_time','log_parent_branch']
        metadata={'split':split,'replicate':0,'seeds':seeds,'master_seed':321,'num_classes':2,
            'feature_names':feature_names,'config':{'grid_points':12 if clipped else 4},
            'input_kind':'synthetic_observed_fixture','reference_classes':{'20':0,'30':1}}
        if clipped:
            metadata.update(parent_grid_points=12,retained_parent_grid_indexes=original.tolist())
        directory=root/split/'rep000';directory.mkdir(parents=True)
        raw=directory/'tensors.pt.gz'
        with gzip.open(raw,'wb') as handle:
            torch.save({'sequence':sequence.as_dict(),'labels':labels,'baseline_probabilities':baseline},handle)
        meta=directory/'metadata.json';meta.write_text(json.dumps(metadata,indent=2)+'\n')
        adapter=ReferenceFeatureAdapter(6,**adapter_options)
        prepared=adapter.precompute(sequence)
        provenance=adapter.manifest_provenance(feature_names,sha256_file(raw))
        cached=directory/'prepared.pt';torch.save({'sequence':prepared.as_dict(),'adapter':provenance},cached)
        records.append({'directory':str(directory),'metadata':str(meta),'metadata_sha256':sha256_file(meta),
            'tensor_file':str(raw),'tensor_sha256':sha256_file(raw),'prepared_tensor_file':str(cached),
            'prepared_tensor_sha256':sha256_file(cached),'adapter_configuration_sha256':provenance['configuration_sha256'],
            'split':split,'replicate':0,'seeds':seeds})
    manifest=root/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':2,'protocol':settings,
        'protocol_sha256':sha256_file(protocol),'replicates':records},indent=2)+'\n')
    return manifest,protocol


@pytest.fixture(autouse=True)
def cpu_and_bounded_storage():
    torch.set_num_threads(1)
    # Fixtures are tiny. Production uses a real recursive storage scan.
    with patch('arggnn.boundary_dataset.scan_repository_storage',return_value=0):
        yield


def prepare(root: Path,**source_options):
    manifest,protocol=fixture_source(root/'source',**source_options)
    dataset=prepare_boundary_dataset(manifest,root/'cache',source_protocol_path=protocol,
        expected_source_manifest_sha256=sha256_file(manifest),expected_source_protocol_sha256=sha256_file(protocol))
    return dataset,manifest,protocol


def mutate_bank(dataset, mutation):
    manifest_path=dataset.directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    record=manifest['records'][0]
    bank_path=dataset.directory/record['bank_file']
    payload=torch.load(bank_path,map_location='cpu',weights_only=True)
    mutation(payload)
    torch.save(payload,bank_path)
    record.update(bank_sha256=sha256_file(bank_path),bank_bytes=bank_path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


def test_oracle_roundtrip_parent_grid_and_labels_outside_bank(tmp_path):
    dataset,_,_=prepare(tmp_path,clipped=False)
    train=load_boundary_cases(dataset,'train',expected_manifest_sha256=dataset.manifest_sha256)[0]
    assert len(train.tiles)==2 and train.labels.shape==(4,2)
    assert train.metadata['num_classes']==2
    assert torch.equal(train.query_node_ids,torch.tensor([11,10]))
    for tile in train.tiles:
        assert torch.equal(tile.labels,train.labels[tile.parent_position_indices])
        assert torch.equal(tile.bank.positions,train.positions[tile.parent_position_indices])
        assert tile.bank.leaf_x.shape[-1]==25
        assert 5 not in tile.bank.feature_columns
    record=dataset.records[0]
    envelope=torch.load(dataset.directory/record['bank_file'],weights_only=True)
    assert set(envelope)=={'schema_version','bank','provenance'}
    assert 'labels' not in envelope['bank'] and 'baseline_probabilities' not in envelope['bank']
    loaded=load_boundary_dataset(dataset.directory,expected_manifest_sha256=dataset.manifest_sha256,
        expected_preparation_sha256=dataset.preparation_sha256)
    assert len(loaded.parent_cases)==2 and len(loaded.records)==4


def test_clipped_inferred_tile_bag_and_original_grid_indices(tmp_path):
    dataset,_,_=prepare(tmp_path)
    case=dataset.cases('train')[0]
    first,second=case.tiles
    assert first.metadata['window_bounds']==[.1,.5]
    assert second.metadata['window_bounds']==[.5,.9]
    assert first.metadata['global_window_bounds']==[0.,.5]
    assert second.metadata['global_window_bounds']==[.5,1.]
    torch.testing.assert_close(first.bank.query_cache.interval_weights,torch.tensor([.5,.5],dtype=torch.float64))
    torch.testing.assert_close(second.bank.query_cache.interval_weights,torch.tensor([.25,.75],dtype=torch.float64))
    assert first.metadata['retained_parent_grid_indexes']==[2,5]
    assert second.metadata['retained_parent_grid_indexes']==[6,10]
    for tile in case.tiles:
        cache=tile.bank.query_cache
        torch.testing.assert_close(cache.bag_cqq,(cache.cqq*cache.interval_weights[:,None,None]).sum(0))


def test_parent_scatter_preserves_nonmonotone_source_position_order(tmp_path):
    dataset,_,_=prepare(tmp_path,unsorted=True)
    case=dataset.cases('train')[0]
    reconstructed=torch.empty_like(case.labels)
    covered=torch.zeros(len(case.positions),dtype=torch.long)
    for tile in case.tiles:
        reconstructed[tile.parent_position_indices]=tile.labels
        covered[tile.parent_position_indices]+=1
    assert torch.equal(reconstructed,case.labels) and bool((covered==1).all())
    assert not bool((case.positions[1:]>case.positions[:-1]).all())


def test_fresh_output_and_frozen_manifest_config_hashes(tmp_path):
    dataset,manifest,protocol=prepare(tmp_path)
    with pytest.raises(RuntimeError,match='fresh'):
        prepare_boundary_dataset(manifest,dataset.directory,source_protocol_path=protocol)
    with pytest.raises(RuntimeError,match='frozen study'):
        load_boundary_dataset(dataset.directory,expected_manifest_sha256='0'*64)
    with pytest.raises(RuntimeError,match='frozen study'):
        load_boundary_dataset(dataset.directory,expected_preparation_sha256='0'*64)
    with pytest.raises(RuntimeError,match='Source manifest'):
        prepare_boundary_dataset(manifest,tmp_path/'never_created',source_protocol_path=protocol,
            expected_source_manifest_sha256='0'*64)
    assert not (tmp_path/'never_created').exists()


def test_source_prepared_tensor_tamper_rejected(tmp_path):
    dataset,manifest,_=prepare(tmp_path)
    record=json.loads(manifest.read_text())['replicates'][0]
    with Path(record['prepared_tensor_file']).open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(RuntimeError,match='Prepared tensor integrity'):
        load_boundary_dataset(dataset.directory)


def test_cached_bank_checksum_and_directory_binding(tmp_path):
    dataset,_,_=prepare(tmp_path)
    path=dataset.directory/dataset.records[0]['bank_file']
    with path.open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(RuntimeError,match='bank integrity'):
        load_boundary_dataset(dataset.directory)


def test_rehashed_feature_leak_rejected_against_observed_leaves(tmp_path):
    dataset,_,_=prepare(tmp_path)
    mutate_bank(dataset,lambda payload:payload['bank']['leaf_x'].add_(1.))
    with pytest.raises(RuntimeError,match='observed source leaves'):
        load_boundary_dataset(dataset.directory)


def test_rehashed_wrong_full_parent_bag_rejected(tmp_path):
    dataset,manifest,_=prepare(tmp_path)
    source=json.loads(manifest.read_text())
    case=load_cases(source,'train')[0]
    bank=precompute_leaf_interfaces(case.sequence,time_column=4,parent_branch_column=5,resistance_scale=1e-4)
    full=with_query_reference_cache(bank,torch.tensor([20,30])).query_cache
    mutate_bank(dataset,lambda payload:payload['bank']['query_cache'].update(
        bag_cqq=full.bag_cqq,bag_eigenvalues=full.bag_eigenvalues,bag_eigenvectors=full.bag_eigenvectors))
    with pytest.raises(RuntimeError,match='Cached bag_cqq'):
        load_boundary_dataset(dataset.directory)


def test_unscored_zero_branch_is_rejected_before_any_cache_directory(tmp_path):
    manifest,protocol=fixture_source(tmp_path/'source',zero_unscored=True)
    # The existing integrity loader permits this equal-age observed edge.
    assert len(load_cases(json.loads(manifest.read_text()),'train'))==1
    with pytest.raises(ValueError):
        prepare_boundary_dataset(manifest,tmp_path/'cache',source_protocol_path=protocol,windows=4)
    assert not (tmp_path/'cache').exists()


def test_incomplete_dataset_and_path_escape_rejected(tmp_path):
    incomplete=tmp_path/'incomplete';incomplete.mkdir()
    (incomplete/'manifest.json').write_text(json.dumps({'schema_version':1,'status':'incomplete'}))
    with pytest.raises(RuntimeError,match='incomplete'):
        load_boundary_dataset(incomplete)
    dataset,_,_=prepare(tmp_path/'complete')
    path=dataset.directory/'manifest.json';manifest=json.loads(path.read_text())
    manifest['records'][0]['bank_file']='../other_parent/bank.pt'
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='directory/parent binding'):
        load_boundary_dataset(dataset.directory)


def test_storage_warning_and_hard_admission(tmp_path):
    guard=StorageAdmission(reserve_bytes=0)
    guard.used_bytes=450_000_000_000
    with pytest.warns(RuntimeWarning,match='approaching'):
        guard.admit(1)
    guard.used_bytes=500_000_000_000-10
    with pytest.raises(RuntimeError,match='storage admission'):
        guard.write(tmp_path/'must_not_exist.bin',b'x'*20)
    assert not (tmp_path/'must_not_exist.bin').exists()


def test_actual_small_scanner_and_outside_project_path_rejection(tmp_path):
    (tmp_path/'small.bin').write_bytes(b'x'*123)
    assert scan_repository_storage(tmp_path)>=123
    # Project path confinement is enforced before any outside write.
    with pytest.raises(ValueError,match='outside project'):
        prepare_boundary_dataset('/tmp/nonexistent_source', '/tmp/nonexistent_cache')

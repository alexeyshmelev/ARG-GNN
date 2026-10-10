"""Independent handmade scalar-metadata CPU probes. No actual manifests or CUDA imports."""
import ast
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import pytest
from arggnn import anchor_compact_cuda_bootstrap_v2 as boot

PROJECT=Path('/workspace-SR003.nfs2/ARG-GNN')
PINS=dict(boot.METADATA_MANIFEST_SHA256)
MIB=1048576
CAP=67108864

@pytest.fixture
def handmade(tmp_path,monkeypatch):
    monkeypatch.setattr(boot,'ROOT',tmp_path)
    pins=dict(PINS)
    monkeypatch.setattr(boot,'METADATA_MANIFEST_SHA256',pins)
    def write(relative,payload,pin=False):
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(payload)
        record={'path':relative,'sha256':hashlib.sha256(payload).hexdigest()}
        if pin: pins[relative]=record['sha256']
        return path,record
    return tmp_path,pins,write

def scalar_bytes(size):
    # Exactly sized valid JSON with one handmade scalar, no arrays.
    return b'{"handmade_scalar":"'+b'x'*(size-22)+b'"}'

@pytest.mark.parametrize('relative',list(PINS))
@pytest.mark.parametrize('size',[MIB+1,1254321])
def test_complete_hash_precedes_decoder_on_both_exact_paths(handmade,monkeypatch,relative,size):
    root,pins,write=handmade
    payload=scalar_bytes(size)
    assert len(payload)==size
    _,record=write(relative,payload,True)
    events=[]
    sha,loads=hashlib.sha256,json.loads
    def complete(data):
        assert data==payload
        events.append(('hash',len(data)))
        return sha(data)
    def decode(data):
        assert events==[('hash',size)]
        events.append(('decode',len(data)))
        return loads(data)
    monkeypatch.setattr(boot.hashlib,'sha256',complete)
    monkeypatch.setattr(boot.json,'loads',decode)
    assert boot.read_bound(record)=={'handmade_scalar':'x'*(size-22)}
    assert events==[('hash',size),('decode',size)]

@pytest.mark.parametrize('relative',list(PINS))
def test_wrong_declared_pin_never_opens_hashes_or_decodes(handmade,relative):
    root,pins,write=handmade
    path,record=write(relative,scalar_bytes(MIB+1),True)
    record['sha256']='f'*64
    with mock.patch.object(Path,'open',side_effect=AssertionError('read forbidden')),mock.patch.object(boot.hashlib,'sha256') as digest,mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError,match='Pinned metadata manifest SHA256 differs'):
            boot.read_bound(record)
    digest.assert_not_called();decoder.assert_not_called()

@pytest.mark.parametrize('relative',list(PINS))
def test_changed_whole_content_hash_rejects_no_decoder(handmade,monkeypatch,relative):
    root,pins,write=handmade
    payload=scalar_bytes(MIB+17)
    path,record=write(relative,payload,True)
    changed=payload[:-4]+b'y'+payload[-3:]
    path.write_bytes(changed)
    hashed=[];sha=hashlib.sha256
    monkeypatch.setattr(boot.hashlib,'sha256',lambda data:hashed.append(data) or sha(data))
    with mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError,match='Source/permit/receipt SHA256 differs') as error:boot.read_bound(record)
    assert 'size limit' not in str(error.value)
    assert hashed==[changed];decoder.assert_not_called()

@pytest.mark.parametrize('relative',list(PINS))
def test_64mib_plus_one_rejects_before_allocation_hash_decode(handmade,monkeypatch,relative):
    root,pins,write=handmade
    path,record=write(relative,b'handmade scalar',True)
    with path.open('r+b') as f:f.truncate(CAP+1)
    realopen=Path.open;reads=[]
    class NoRead:
        def __init__(self,f):self.f=f
        def __enter__(self):return self
        def __exit__(self,*args):self.f.close()
        def fileno(self):return self.f.fileno()
        def read(self,n):reads.append(n);raise AssertionError('oversize read forbidden')
    monkeypatch.setattr(Path,'open',lambda p,*a,**kw:NoRead(realopen(p,*a,**kw)))
    with mock.patch.object(boot.hashlib,'sha256') as digest,mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError,match='Bound file exceeds size limit 67108864 bytes') as error:boot.read_bound(record)
    assert 'SHA256' not in str(error.value) and reads==[]
    digest.assert_not_called();decoder.assert_not_called()

@pytest.mark.parametrize('relative',list(PINS))
def test_metadata_growth_after_stat_rejected_before_hash_or_decode(handmade,monkeypatch,relative):
    root,pins,write=handmade
    path,record=write(relative,b'handmade',True)
    with path.open('r+b') as f:f.truncate(CAP+1)
    monkeypatch.setattr(boot.os,'fstat',lambda fd:SimpleNamespace(st_size=1))
    with mock.patch.object(boot.hashlib,'sha256') as digest,mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError,match='size limit 67108864 bytes'):boot.read_bound(record)
    digest.assert_not_called();decoder.assert_not_called()

@pytest.mark.parametrize('relative',['src/arggnn/source.py','research/handmade_permit.json','research/reviews/handmade/review_receipt.json','configs/handmade.json','data/manifest.json','data/biological_candidate_pilot_v1/renamed.json','data/biological_candidate_pilot_v1_other/manifest.json','data/biological_paths_execution_pilot_v1/other/manifest.json'])
def test_other_names_sources_permits_reviews_configs_keep_default_cap(handmade,relative):
    root,pins,write=handmade
    _,record=write(relative,scalar_bytes(MIB+1))
    with mock.patch.object(boot.hashlib,'sha256') as digest,mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError,match='size limit 1048576 bytes'):boot.read_bound(record)
    digest.assert_not_called();decoder.assert_not_called()

@pytest.mark.parametrize('relative',list(PINS))
def test_unlisted_hardlink_alias_has_default_cap(handmade,relative):
    root,pins,write=handmade
    path,record=write(relative,scalar_bytes(MIB+1),True)
    alias=root/'renamed_manifest.json';alias.hardlink_to(path)
    with mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError,match='size limit 1048576 bytes'):boot.read_bound(dict(record,path=alias.name))
    decoder.assert_not_called()

@pytest.mark.parametrize('kind',['traversal','symlink','directory','outside'])
def test_noncanonical_refs_block_before_decode(handmade,kind):
    root,pins,write=handmade
    path,record=write(next(iter(PINS)),b'{"handmade":true}',True)
    if kind=='traversal':record['path']='data/../'+record['path']
    elif kind=='symlink':
        alias=root/'alias.json';alias.symlink_to(path);record['path']=alias.name
    elif kind=='directory':record['path']='data'
    else:record['path']='/outside_handmade_manifest.json'
    with mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError):boot.read_bound(record)
    decoder.assert_not_called()

@pytest.mark.parametrize('mutation',[{'sha256':True},{'sha256':'F'*64},{'sha256':'f'*63},{'path':1},{'extra':True}])
def test_record_exact_schema_rejects_before_io(handmade,mutation):
    root,pins,write=handmade
    _,record=write(next(iter(PINS)),b'{"handmade":true}',True);record.update(mutation)
    with mock.patch.object(Path,'open',side_effect=AssertionError('IO forbidden')),mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError,match='Exact hash-bound record'):boot.read_bound(record)
    decoder.assert_not_called()

def test_exact_64mib_boundary_accepts_complete_handmade_json(handmade):
    root,pins,write=handmade
    payload=scalar_bytes(CAP)
    _,record=write(next(iter(PINS)),payload,True)
    assert boot.bound_bytes(record)==payload

def test_exact_1mib_default_accepts_json_and_bad_hash_is_distinct(handmade):
    root,pins,write=handmade
    payload=scalar_bytes(MIB)
    _,record=write('research/default.json',payload)
    assert boot.read_bound(record)=={'handmade_scalar':'x'*(MIB-22)}
    record['sha256']='0'*64
    with mock.patch.object(boot.json,'loads') as decoder:
        with pytest.raises(ValueError,match='SHA256 differs') as error:boot.read_bound(record)
    assert 'size limit' not in str(error.value);decoder.assert_not_called()

@pytest.mark.parametrize('which',['prepared_manifest','observed_manifest'])
@pytest.mark.parametrize('failure',['oversize','changed_hash','declared_pin','changed_name','changed_ref'])
def test_adapter_negative_metadata_blocks_run_before_context(handmade,monkeypatch,which,failure):
    root,pins,write=handmade
    candidate_bytes=b'{"handmade_candidate_config":true}'
    _,candidate=write('configs/handmade_candidate.json',candidate_bytes)
    paths=list(PINS)
    observed={'truth_or_oracle_read':False,'handmade_scalar':'x'*(MIB+11)}
    observed_payload=json.dumps(observed).encode()
    observed_path,observed_record=write(paths[1],observed_payload,True)
    prepared={'config_sha256':candidate['sha256'],'observed_manifest_sha256':observed_record['sha256'],'source_sha256':{'handmade.py':'a'*64},'handmade_scalar':'x'*(MIB+11)}
    prepared_payload=json.dumps(prepared).encode()
    prepared_path,prepared_record=write(paths[0],prepared_payload,True)
    cfg={'prepared_manifest':prepared_record,'observed_manifest':observed_record,'original_candidate_config':candidate,'original_preparation_source_sha256':prepared['source_sha256']}
    bad=cfg[which]; badpath=prepared_path if which=='prepared_manifest' else observed_path
    badpayload=prepared_payload if which=='prepared_manifest' else observed_payload
    if failure=='oversize':
        with badpath.open('r+b') as f:f.truncate(CAP+1)
        match='size limit 67108864 bytes'
    elif failure=='changed_hash':
        badpath.write_bytes(badpayload+b' ');match='Source/permit/receipt SHA256 differs'
    elif failure=='declared_pin':bad['sha256']='b'*64;match='Pinned metadata manifest SHA256 differs'
    elif failure=='changed_name':
        renamed=badpath.with_name('renamed.json');renamed.write_bytes(badpayload)
        bad['path']=renamed.relative_to(root).as_posix();match='size limit 1048576 bytes'
    else:
        renamed=root/'data/other_handmade_manifest.json';renamed.write_bytes(badpayload)
        bad['path']=renamed.relative_to(root).as_posix();match='size limit 1048576 bytes'
    _,config_record=write('configs/handmade_adapter.json',json.dumps(cfg).encode())
    monkeypatch.setattr(boot,'ADAPTER_CONFIG',config_record)
    events=[]
    monkeypatch.setattr(boot,'prepare_scope',lambda output:events.append('scope'))
    monkeypatch.setattr(boot.guard,'apply_and_validate_title',lambda:events.append('title') or boot.guard.PROCESS_TITLE)
    torch=SimpleNamespace(__version__='fake',version=SimpleNamespace(cuda='fake'),cuda=SimpleNamespace(is_initialized=lambda:False))
    policy=SimpleNamespace(DETERMINISTIC_POLICY={'handmade':True},configure_deterministic_development=lambda:{'handmade':True})
    sealed=SimpleNamespace(select_train_tiles_v1=mock.Mock(side_effect=AssertionError('selection forbidden after negative guard')))
    runner=SimpleNamespace(sealed=sealed,check_owned_storage=mock.Mock(),main=mock.Mock())
    monkeypatch.setattr(boot,'load_cuda_modules',lambda:(torch,policy,runner))
    monkeypatch.setattr(boot,'initialize_context',mock.Mock(side_effect=AssertionError('CUDA forbidden')))
    monkeypatch.setattr(boot,'recheck_inputs',mock.Mock())
    admission={'permit':{'runtime_expected':{'torch':'fake','torch_cuda':'fake'},'adapter_admission_fields':{}},'output':root/'no_scientific_output'}
    with pytest.raises(ValueError,match=match):boot.run_bootstrap(admission)
    assert events==['scope','title']
    boot.initialize_context.assert_not_called();runner.main.assert_not_called();sealed.select_train_tiles_v1.assert_not_called()
    assert not admission['output'].exists()

def test_source_constants_exact_match_root_failure_and_adapter():
    adapter=json.loads((PROJECT/'configs/anchor_compact_actual_bank_v1.json').read_text())
    evidence=json.loads((PROJECT/'experiments/results/anchor_compact_actual_qualification_v1_root/metadata_bound_failure.json').read_text())
    expected={adapter[k]['path']:adapter[k]['sha256'] for k in ('prepared_manifest','observed_manifest')}
    assert PINS==expected=={x['path']:x['actual_sha256'] for x in evidence['manifest_checks']}
    assert [x['bytes'] for x in evidence['manifest_checks']]==[34087964,25043899]
    assert all(MIB<x['bytes']<CAP and x['same_as_admitted_original'] for x in evidence['manifest_checks'])
    assert evidence['CUDA_context_reached'] is False and evidence['scientific_bank_arrays_opened'] is False
    assert set(PINS)=={'data/biological_candidate_pilot_v1/manifest.json','data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json'}
    assert boot.DEFAULT_BOUND_FILE_LIMIT_BYTES==MIB and boot.METADATA_MANIFEST_LIMIT_BYTES==CAP

def test_review_process_has_no_torch_or_scientific_import():
    assert 'torch' not in sys.modules
    assert 'arggnn.anchor_compact_actual_bank_v1' not in sys.modules
    assert 'arggnn.anchor_compact_qualification_runner_v1' not in sys.modules

"""Handmade full280/47-field CPU banks only. No actual arrays or GPU calls."""
from copy import deepcopy
from dataclasses import replace
import hashlib,json,os,random
from pathlib import Path
import numpy as np
import pytest
import torch
import tskit
from arggnn import biological_bank_cache_v1 as cachemod
from arggnn import biological_candidate as frozen
from arggnn import biological_mechanism_controls_v1 as controls
from arggnn.age_interface import precompute_age_interfaces
from arggnn.anchor_kernel import prepare_anchor_kernel_bank,AnchorKernelBank

@pytest.fixture(autouse=True)
def CPU():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(1)


def write_json(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True)+'\n');return {'path':str(path.relative_to(frozen.ROOT)),'sha256':frozen.sha(path)}


@pytest.fixture(scope='module')
def handmade(tmp_path_factory):
    torch.set_num_threads(1);directory=tmp_path_factory.mktemp('full280')
    tables=tskit.TableCollection(sequence_length=2000.)
    ages=[0.]*64+[130.]*16+[135.]*16+[136.]*8+[155.]*8+[170.]*32+[200.]*16+[165.]*8+[170.]*16+[300.+4*(i//2) for i in range(96)]
    for age in ages:tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age)
    for number,age in enumerate((1000.,1200.,900.,1500.)):
        root=tables.nodes.add_row(time=age)
        for child in range(280):tables.edges.add_row(number*500.,(number+1)*500.,root,child)
    tables.sort();ts=tables.tree_sequence()
    query=np.r_[np.arange(64),np.arange(80,96),np.arange(104,112),np.arange(144,160),np.arange(168,184)]
    rows=[{'observed_reference_class':None if i<184 else (i-184)//24} for i in range(280)]
    parent={'parent_id':'handmade_train_000','split':'train','observation':{'nodes':np.arange(280),'offset':0.},'dynamics_slots':np.arange(184),'reference_slots':np.arange(184,280),'query_slots':query,'positions':np.arange(50.,2000.,100.),'samples':rows,'output_indexes':query}
    tiles=[]
    for number in range(2):
        seq,indexes,tree_ids,landmarks=frozen.pack_observed_tile(ts,parent,number*1000.,(number+1)*1000.)
        age=precompute_age_interfaces(seq,time_column=6,parent_branch_column=7,time_scale=10000.,resistance_scale=.0001,age_scales=(350.,1400.),strength=3.,reference_ids=torch.arange(184,280),max_condition=1e12)
        prepared=prepare_anchor_kernel_bank(age.plain_bank,age.operator_cache,num_anchors=32,num_scales=3)
        bankfile=directory/f'tile{number}.bank.pt';bundle={'schema':'biological_candidate_full_age_bank_v1','age_bank':age.as_dict(),'fixed_prepared':prepared};torch.save(bundle,bankfile)
        tiles.append({'tile_id':number,'bank':{'path':str(bankfile.relative_to(frozen.ROOT)),'sha256':frozen.sha(bankfile)},'position_indexes':indexes.tolist(),'exact_serialization_roundtrip_checked':True,'all_original_age_cache_fields_retained':True,'full_bank_field_fingerprint_sha256':frozen.fingerprint_sha(bundle),'original_tree_ids':tree_ids})
    cfg=json.loads((frozen.ROOT/'configs/biological_mechanism_controls_v1.json').read_text());configref=write_json(directory/'config.json',cfg)
    manifest={'status':'complete_observed_only_Q184_R96_actualQ120','truth_opened':False,'parents':[{'parent_id':parent['parent_id'],'tiles':tiles}]};preparedref=write_json(directory/'prepared.json',manifest);observedref=write_json(directory/'observed.json',{'handmade':True})
    return directory,parent,tiles,cfg,preparedref,configref,observedref


def make_cache(fixture,CPU_bytes=1000000000,GPU_bytes=1000000000):
    _,_,_,_,prep,cfg,obs=fixture
    return cachemod.BiologicalBankCacheV1(prep,cfg,obs,cpu_budget_bytes=CPU_bytes,gpu_budget_bytes=GPU_bytes)


def rng():return cachemod._rng_v1('cpu')


@pytest.mark.parametrize('name',controls.CONTROL_NAMES)
def test_all_seven_full_default_models_complete_field_logit_VJP_AdamW_RNG_parity(handmade,name):
    _,parent,tiles,cfg,*_=handmade;cache=make_cache(handmade);tile=tiles[0]
    baseline=cachemod.BASE_LOAD_BANK(tile,cfg,parent,'cpu');before_rng=rng();cached=cache(tile,cfg,parent,'cpu');again=cache(tile,cfg,parent,torch.device('cpu'))
    assert rng()==before_rng and cached is again and frozen.fingerprint_sha(cached)==frozen.fingerprint_sha(baseline)
    assert cached.bank.leaf_x.shape==(2,280,47) and cached.bank.query_node_ids.numel()==184 and cached.bank.query_cache.reference_ids.numel()==96
    assert cached.bank.positions.dtype==cached.bank.intervals.dtype==cached.layout.coordinates.dtype==cached.layout.segment_duration.dtype==torch.float64
    torch.manual_seed(181);first=controls.model_factory(name,cfg);second=deepcopy(first);outputs=[];gradients=[];states=[]
    bank_fingerprint=frozen.fingerprint_sha(cached)
    for model,bank in ((first,baseline),(second,cached)):
        optimizer=torch.optim.AdamW(frozen.optimizer_groups(model,cfg),lr=cfg['training']['learning_rate'])
        saved_rng=rng();logits=model(bank).index_select(1,torch.from_numpy(parent['output_indexes']))
        weights=torch.linspace(-.7,1.1,logits.numel(),dtype=logits.dtype).reshape_as(logits);(logits*weights).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        outputs.append(logits.detach().clone());gradients.append([p.grad.detach().clone() for p in model.parameters()]);assert rng()==saved_rng
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();states.append(optimizer.state_dict())
    assert torch.equal(*outputs) and all(torch.equal(a,b) for a,b in zip(*gradients))
    assert frozen.fingerprint_sha(first.state_dict())==frozen.fingerprint_sha(second.state_dict()) and frozen.fingerprint_sha(states[0])==frozen.fingerprint_sha(states[1])
    assert frozen.fingerprint_sha(cached)==bank_fingerprint
    assert cache.stats['cold_original_loads']==1 and cache.stats['CPU_hits']==1 and cache.stats['bank_sha_verified_bytes']==2*Path(frozen.checked(tile['bank'])).stat().st_size
    cache.close()


def test_every_nested_original_field_and_shared_stride_is_present_and_unchanged(handmade):
    _,parent,tiles,cfg,*_=handmade;cache=make_cache(handmade);bank=cache(tiles[0],cfg,parent,'cpu')
    original=cachemod.BASE_LOAD_BANK(tiles[0],cfg,parent,'cpu');assert frozen.field_fingerprint(bank)==frozen.field_fingerprint(original)
    assert set(bank.operators)=={'plain','age','matched_scalar_age'}
    assert tuple(bank.bank.feature_columns)==tuple(i for i in range(48) if i!=7)
    assert not any(t.requires_grad for t in cachemod._tensors(bank));cache.close()


def test_CPU_LRU_evicts_whole_entries_and_bypasses_oversize_without_changing_values(handmade):
    _,parent,tiles,cfg,*_=handmade;first=cachemod.BASE_LOAD_BANK(tiles[0],cfg,parent,'cpu');size=cachemod.resident_bytes_v1(first)
    second=cachemod.BASE_LOAD_BANK(tiles[1],cfg,parent,'cpu');size2=cachemod.resident_bytes_v1(second);cache=make_cache(handmade,max(size,size2),0)
    for tile in (tiles[0],tiles[1],tiles[0]):
        bank=cache(tile,cfg,parent,'cpu');assert frozen.fingerprint_sha(bank)==frozen.fingerprint_sha(cachemod.BASE_LOAD_BANK(tile,cfg,parent,'cpu'))
        assert cache.cpu_used<=cache.cpu_budget and len(cache.cpu)==1
    assert cache.stats['cold_original_loads']==3 and cache.stats['CPU_evictions']==2;cache.close()
    small=make_cache(handmade,1,0);small(tiles[0],cfg,parent,'cpu');small(tiles[0],cfg,parent,'cpu')
    assert small.cpu_used==0 and small.stats['cold_original_loads']==2 and small.stats['CPU_oversized_bypass']==2;small.close()


def test_GPU_LRU_bookkeeping_with_explicit_mock_transfer_uses_no_GPU(handmade,monkeypatch):
    _,parent,tiles,cfg,*_=handmade;original_to=AnchorKernelBank.to;transfers=[]
    def mocked_to(bank,device,dtype=None):
        target=torch.device(device)
        if target.type=='cuda':transfers.append(str(target));return original_to(bank,'cpu',dtype)
        return original_to(bank,device,dtype)
    monkeypatch.setattr(AnchorKernelBank,'to',mocked_to)
    one=cachemod.BASE_LOAD_BANK(tiles[0],cfg,parent,'cpu');two=cachemod.BASE_LOAD_BANK(tiles[1],cfg,parent,'cpu')
    limit=max(cachemod.transfer_upper_bytes_v1(one),cachemod.transfer_upper_bytes_v1(two))
    cache=make_cache(handmade,GPU_bytes=limit)
    cache(tiles[0],cfg,parent,'cuda');cache(tiles[0],cfg,parent,'cuda:0');assert len(transfers)==1 and cache.stats['GPU_hits']==1
    cache(tiles[1],cfg,parent,'cuda:0');assert cache.gpu_used<=limit
    cache(tiles[0],cfg,parent,'cuda:0');assert cache.stats['GPU_evictions']>=1 and len(transfers)==3
    cache.close()
    tiny=make_cache(handmade,GPU_bytes=1);tiny(tiles[0],cfg,parent,'cuda:0');tiny(tiles[0],cfg,parent,'cuda:0')
    assert not tiny.gpu and tiny.stats['GPU_oversized_bypass']==2;tiny.close()


@pytest.mark.parametrize('bad',['tile','parent','roles','grid','dtype','device','config'])
def test_no_substitution_across_tile_parent_roles_grid_device_dtype_or_config(handmade,bad):
    _,parent,tiles,cfg,*_=handmade;cache=make_cache(handmade);cache(tiles[0],cfg,parent,'cpu')
    p=deepcopy(parent);tile=deepcopy(tiles[0]);config=deepcopy(cfg);device='cpu';dtype=torch.float32
    if bad=='tile':tile['tile_id']=1
    elif bad=='parent':p['parent_id']='wrong_parent'
    elif bad=='roles':p['dynamics_slots']=p['dynamics_slots'][::-1].copy()
    elif bad=='grid':p['positions'][0]+=.001
    elif bad=='dtype':dtype=torch.float64
    elif bad=='device':device='cuda:1'
    elif bad=='config':config['training']['epochs']+=1
    with pytest.raises(ValueError):cache(tile,config,p,device,dtype=dtype)
    cache.close()


def test_changed_bank_BYTES_fail_even_after_restored_mtime_and_before_cache_hit(tmp_path,handmade):
    directory,parent,tiles,cfg,prep,cfgref,obs=handmade;tile=deepcopy(tiles[0]);source=frozen.checked(tile['bank']);copy=tmp_path/'copied.bank.pt';copy.write_bytes(source.read_bytes());tile['bank']={'path':str(copy.relative_to(frozen.ROOT)),'sha256':frozen.sha(copy)}
    manifest={'status':'complete_observed_only_Q184_R96_actualQ120','truth_opened':False,'parents':[{'parent_id':parent['parent_id'],'tiles':[tile]}]};manifestref=write_json(tmp_path/'prepared.json',manifest)
    cache=cachemod.BiologicalBankCacheV1(manifestref,cfgref,obs,cpu_budget_bytes=1000000000,gpu_budget_bytes=0);cache(tile,cfg,parent,'cpu');stat=copy.stat()
    with copy.open('r+b') as stream:stream.seek(-20,2);value=stream.read(1);stream.seek(-1,1);stream.write(bytes([value[0]^1]))
    os.utime(copy,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(ValueError,match='changed'):cache(tile,cfg,parent,'cpu')
    cache.close()


def test_source_or_manifest_identity_change_fails_without_native_decode(tmp_path,handmade):
    _,parent,tiles,cfg,prep,cfgref,_=handmade;obs=write_json(tmp_path/'observed.json',{'handmade':True});cache=cachemod.BiologicalBankCacheV1(prep,cfgref,obs,cpu_budget_bytes=1000000000,gpu_budget_bytes=0)
    path=frozen.path(obs['path']);path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError):cache(tiles[0],cfg,parent,'cpu')
    assert cache.stats['cold_original_loads']==0;cache.close()


def test_known_tensor_mutation_fails_and_scoped_aliases_restore_on_exception(handmade):
    _,parent,tiles,cfg,*_=handmade;cache=make_cache(handmade)
    try:
        with cachemod.installed_cache_v1(cache):
            assert controls.load_bank is cache and frozen.load_bank is cache
            bank=controls.load_bank(tiles[0],cfg,parent,'cpu');bank.bank.leaf_x.add_(1.)
            with pytest.raises(ValueError,match='mutated'):cache(tiles[0],cfg,parent,'cpu')
            raise RuntimeError('handmade failure')
    except ValueError as error:assert 'mutated' in str(error)
    assert controls.load_bank is cachemod.BASE_LOAD_BANK and frozen.load_bank is cachemod.BASE_LOAD_BANK


def test_scope_covers_both_TRAIN_and_VAL_loader_aliases_and_restores_normal_errors(handmade):
    _,parent,tiles,cfg,*_=handmade;cache=make_cache(handmade)
    with pytest.raises(RuntimeError,match='intentional'):
        with cachemod.installed_cache_v1(cache):
            a=controls.load_bank(tiles[0],cfg,parent,'cpu');b=frozen.load_bank(tiles[0],cfg,parent,'cpu');assert a is b
            raise RuntimeError('intentional')
    assert cache.closed and controls.load_bank is cachemod.BASE_LOAD_BANK and frozen.load_bank is cachemod.BASE_LOAD_BANK


def test_transfer_upper_bound_handles_expanded_aliases_and_integer_coordinate_fields(handmade):
    _,parent,tiles,cfg,*_=handmade;bank=cachemod.BASE_LOAD_BANK(tiles[0],cfg,parent,'cpu')
    assert cachemod.transfer_upper_bytes_v1(bank)>=cachemod.resident_bytes_v1(bank)>0
    expanded=torch.ones(1,2,dtype=torch.float32).expand(9,2)
    assert expanded.stride()[0]==0 and cachemod.transfer_upper_bytes_v1([expanded,expanded])==2*expanded.numel()*expanded.element_size()
    assert cachemod.resident_bytes_v1([expanded,expanded])==expanded.untyped_storage().nbytes()
    assert bank.bank.positions.dtype==torch.float64 and bank.layout.anchor_graph.dtype==torch.int64


def test_mutated_bank_eviction_fails_and_close_still_releases_all_entries(handmade):
    _,parent,tiles,cfg,*_=handmade;size=cachemod.resident_bytes_v1(cachemod.BASE_LOAD_BANK(tiles[0],cfg,parent,'cpu'))
    cache=make_cache(handmade,size,0);bank=cache(tiles[0],cfg,parent,'cpu');bank.bank.leaf_x.add_(1.)
    with pytest.raises(ValueError,match='mutated'):cache(tiles[1],cfg,parent,'cpu')
    with pytest.raises(ValueError,match='mutated'):cache.close()
    assert cache.closed and not cache.cpu and not cache.gpu and cache.cpu_used==cache.gpu_used==0


def test_root_phase_and_memory_expiry_stops_cache_before_new_file_decode(handmade):
    _,parent,tiles,cfg,*_=handmade;_,_,_,_,prep,cref,obs=handmade;calls=[]
    def expired():calls.append(True);raise ValueError('root memory reservation expired')
    cache=cachemod.BiologicalBankCacheV1(prep,cref,obs,cpu_budget_bytes=1000000000,gpu_budget_bytes=0,active_guard=expired)
    with pytest.raises(ValueError,match='expired'):cache(tiles[0],cfg,parent,'cpu')
    assert len(calls)==1 and cache.stats['cold_original_loads']==0;cache.close()


def test_original_runner_entry_preserves_exact_scientific_argv_and_restores_aliases(handmade,monkeypatch,tmp_path):
    _,parent,tiles,cfg,prep,cref,obs=handmade;cache=make_cache(handmade);output=str((tmp_path/'fresh_output').relative_to(frozen.ROOT))
    admission={'seed':202,'device':'cpu','output_directory':output,'refs':{'original_config':cref,'prepared_manifest':prep,'observed_manifest':obs}}
    monkeypatch.setattr(cachemod,'admit_cache_v1',lambda *_args,**_kw:(cache,admission,{'original_source_sha256':cfg['source_sha256']}))
    argv=['--train','--device','cpu','--seed','202','--output',output,'--config',cref['path'],'--expected-config-sha256',cref['sha256'],'--observed-manifest',obs['path'],'--expected-observed-manifest-sha256',obs['sha256'],'--prepared-manifest',prep['path'],'--expected-prepared-manifest-sha256',prep['sha256'],'--expected-base-config-sha256',controls.BASE_CONFIG_SHA256,'--storage-snapshot','handmade_original_snapshot','--expected-storage-snapshot-sha256','a'*64]
    original=list(__import__('sys').argv);seen=[]
    def original_main():
        seen.append(list(__import__('sys').argv)[1:]);assert frozen.load_bank is cache and controls.load_bank is cache
        (frozen.ROOT/output).mkdir();bank=controls.load_bank(tiles[0],cfg,parent,'cpu');assert bank.bank.query_node_ids.numel()==184
        raise SystemExit(0)
    monkeypatch.setattr(controls,'main',original_main)
    with pytest.raises(SystemExit) as stopped:cachemod.run_original_mechanisms_with_cache_v1({'path':'handmade','sha256':'a'*64},argv)
    assert stopped.value.code==0 and seen==[argv] and __import__('sys').argv==original
    assert controls.load_bank is cachemod.BASE_LOAD_BANK and frozen.load_bank is cachemod.BASE_LOAD_BANK
    record=json.loads((frozen.ROOT/output/'bank_cache_execution_v1.json').read_text());assert record['original_science_argv']==argv and record['math_or_science_changed'] is False


def test_shipped_caps_are_explicit64GB_CPU32GB_GPU_and_class_enforces_own_bound(handmade):
    cfg=json.loads((frozen.ROOT/'configs/biological_bank_cache_v1.json').read_text())
    assert cfg['CPU_budget_bytes']==64000000000 and cfg['GPU_budget_bytes']==32000000000
    cache=make_cache(handmade,CPU_bytes=cfg['CPU_budget_bytes'],GPU_bytes=cfg['GPU_budget_bytes'])
    _,parent,tiles,science,*_=handmade
    for tile in tiles:cache(tile,science,parent,'cpu')
    assert len(cache.cpu)==2 and cache.cpu_used<=64000000000 and cache.gpu_used==0
    cache.close()


def test_target_free_profile_entry_executes_all_seven_models_on_handmade_banks(handmade,monkeypatch,tmp_path):
    _,parent,tiles,cfg,prep,cref,obs=handmade;cache=make_cache(handmade);output=str((tmp_path/'profile').relative_to(frozen.ROOT))
    admission={'device':'cpu','output_directory':output,'profile_cell':{'parent_id':parent['parent_id'],'tile_id':0},'refs':{'cache_config':{'path':'handmade_config','sha256':'a'*64},'original_config':cref,'prepared_manifest':prep,'observed_manifest':obs},'source_sha256':{'src/arggnn/biological_bank_cache_v1.py':'b'*64}}
    monkeypatch.setattr(cachemod,'admit_cache_v1',lambda *_args,**_kw:(cache,admission,{}))
    monkeypatch.setattr(controls,'load_protocol',lambda *_:(cfg,cfg,None))
    monkeypatch.setattr(frozen,'observed_contract',lambda *_:[parent])
    monkeypatch.setattr(controls,'validate_prepared_records',lambda *_:{parent['parent_id']:{'tiles':tiles}})
    monkeypatch.setattr(frozen,'configure_fit',lambda *_:(torch.device('cpu'),{'handmade_CPU_only':True},None))
    result=cachemod.profile_one_tile_v1({'path':'handmade_admission','sha256':'c'*64})
    assert result['status']=='one_tile_parity_and_loading_profile_complete_no_speed_claim'
    assert result['target_arrays_opened'] is False and result['scientific_fit_performed'] is False
    assert len(result['control_checks'])==7 and result['all_fields_logits_VJP_optimizer_RNG_parity'] is True
    assert len(result['loading_seconds']['original'])==len(result['loading_seconds']['cache'])==4
    assert (frozen.ROOT/output/'one_tile_profile.json').is_file() and cache.closed

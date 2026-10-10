"""Independent extension source/handmade CPU probes. No live controller or worker."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import pytest
from arggnn import gpu_pool_v1 as pool
PROJECT=Path('/workspace-SR003.nfs2/ARG-GNN')
spec=importlib.util.spec_from_file_location('independent_pool_research_v3',PROJECT/'scripts/gpu_pool_research_v3.py')
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)

@pytest.fixture
def source_fixture(tmp_path):
    cfg=json.loads((PROJECT/ext.CONFIG).read_text())
    records=[cfg['base_controller'],*cfg['unchanged_entrypoint_and_guards'].keys(),cfg['extension_worker']['source'],cfg['extension_worker']['config'],cfg['extension_worker']['independent_review'],cfg['bootstrap_worker']['source'],cfg['bootstrap_worker']['config'],ext.CONFIG]
    for item in records:
        relative=item if isinstance(item,str) else item['path']
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((PROJECT/relative).read_bytes())
    return tmp_path,cfg

def rebind_cfg(root,cfg,monkeypatch):
    raw=json.dumps(cfg,sort_keys=True).encode();(root/ext.CONFIG).write_bytes(raw)
    monkeypatch.setattr(ext,'CONFIG_SHA256',hashlib.sha256(raw).hexdigest())

@pytest.mark.parametrize('field,value',[
    ('kind','other'),('status','active_root_admission'),('project_root','/wrong'),('actual_execution_admitted',True),('root_bootstrap_and_fresh_admission_required',False),('allowed_modules',[]),('allowed_modules',sorted(pool.MODULES|{ext.BOOTSTRAP})),('allowed_modules',sorted(ext.MODULES|{'arggnn.arbitrary'})),('direct_runner_submission_allowed',True)])
def test_config_shape_and_allowlist_reject_with_rebound_handmade_config(source_fixture,monkeypatch,field,value):
    root,cfg=source_fixture;cfg[field]=value;rebind_cfg(root,cfg,monkeypatch)
    with mock.patch.object(pool,'Controller') as controller,mock.patch.object(pool,'request') as ipc:
        with pytest.raises(ValueError):ext.verify_bindings(root)
    controller.assert_not_called();ipc.assert_not_called()

@pytest.mark.parametrize('which,field,value',[
    ('extension_worker','module','arggnn.arbitrary'),('bootstrap_worker','module','arggnn.anchor_compact_cuda_bootstrap_v1'),('bootstrap_worker','independent_review_required_before_execution',False)])
def test_worker_identity_or_review_requirement_cannot_change(source_fixture,monkeypatch,which,field,value):
    root,cfg=source_fixture;cfg[which][field]=value;rebind_cfg(root,cfg,monkeypatch)
    with pytest.raises(ValueError):ext.verify_bindings(root)

@pytest.mark.parametrize('which',['source','config'])
def test_bootstrap_changed_hash_or_ref_rejects(source_fixture,monkeypatch,which):
    root,cfg=source_fixture;record=cfg['bootstrap_worker'][which]
    record['sha256']='0'*64;rebind_cfg(root,cfg,monkeypatch)
    with pytest.raises(ValueError,match='SHA256 differs'):ext.verify_bindings(root)

@pytest.mark.parametrize('which',['source','config'])
def test_bootstrap_ref_to_other_bytes_cannot_reuse_pin(source_fixture,monkeypatch,which):
    root,cfg=source_fixture;record=cfg['bootstrap_worker'][which]
    oldpath=root/record['path'];newpath=root/'renamed_bootstrap';newpath.write_bytes(oldpath.read_bytes()+b'changed')
    record['path']=newpath.name;rebind_cfg(root,cfg,monkeypatch)
    with pytest.raises(ValueError,match='SHA256 differs'):ext.verify_bindings(root)

@pytest.mark.parametrize('role',['base','bootstrap_source','bootstrap_config','review'])
def test_every_extension_bound_role_retains_1mib_guard(source_fixture,role):
    root,cfg=source_fixture
    records={'base':cfg['base_controller'],'bootstrap_source':cfg['bootstrap_worker']['source'],'bootstrap_config':cfg['bootstrap_worker']['config'],'review':cfg['extension_worker']['independent_review']}
    path=root/records[role]['path']
    with path.open('r+b') as f:f.truncate(1048577)
    with mock.patch.object(pool,'Controller') as controller,mock.patch.object(pool,'request') as ipc:
        with pytest.raises(ValueError,match='SHA256 differs'):ext.verify_bindings(root)
    controller.assert_not_called();ipc.assert_not_called()

@pytest.mark.parametrize('role',['bootstrap_source','bootstrap_config'])
def test_symlink_bootstrap_refs_reject(source_fixture,monkeypatch,role):
    root,cfg=source_fixture;record=cfg['bootstrap_worker']['source' if role=='bootstrap_source' else 'config']
    path=root/record['path'];alias=root/'alias_bootstrap';alias.symlink_to(path)
    record['path']=alias.name;rebind_cfg(root,cfg,monkeypatch)
    with pytest.raises(ValueError,match='without symlinks'):ext.verify_bindings(root)

def submit(module):
    return ['submit','--pool','handmade','--job','handmade','--model','H020_compact_qualification','--seed','101','--output','research/handmade_absent','--peak-mib','12288','--storage-bytes','1001048576','--policy','reject','--log-limit-bytes','1048576',module,'--config','handmade','--permit','inactive']

@pytest.mark.parametrize('module',[ext.WORKER,'arggnn.anchor_compact_cuda_bootstrap_v1','arggnn.arbitrary'])
def test_nonbootstrap_additions_reject_before_ipc(monkeypatch,module):
    monkeypatch.setattr(ext,'verify_bindings',lambda:None)
    with mock.patch.object(pool,'request') as ipc,mock.patch.object(pool,'Controller') as controller:
        with pytest.raises((ValueError,SystemExit)):ext.main(submit(module))
    ipc.assert_not_called();controller.assert_not_called()

def test_only_binding_failure_precedes_submission(monkeypatch):
    monkeypatch.setattr(ext,'verify_bindings',mock.Mock(side_effect=ValueError('handmade binding failure')))
    with mock.patch.object(pool,'request') as ipc:
        with pytest.raises(ValueError,match='handmade binding failure'):ext.main(submit(ext.BOOTSTRAP))
    ipc.assert_not_called()

@pytest.mark.parametrize('command',['status','wait','shutdown'])
def test_maintenance_delegates_despite_source_guard_failure(monkeypatch,command):
    monkeypatch.setattr(ext,'verify_bindings',mock.Mock(side_effect=AssertionError('should not run')))
    with mock.patch.object(pool,'main',return_value=29) as base:
        assert ext.main([command,'--pool','handmade'])==29
    base.assert_called_once_with([command,'--pool','handmade']);ext.verify_bindings.assert_not_called()

def test_exact_names_pins_and_base_allowlist_remain_fixed(source_fixture):
    root,cfg=source_fixture;original=pool.MODULES
    assert ext.verify_bindings(root)==cfg
    assert pool.MODULES is original
    assert ext.MODULES==original|{'arggnn.anchor_compact_cuda_bootstrap_v2','arggnn.anchor_compact_qualification_runner_v1'}
    assert cfg['bootstrap_worker']['source']['sha256']=='d6f1504901b926d45a59664fcc9d211892eb087586aca1a53d34256e05db1fe5'
    assert cfg['bootstrap_worker']['config']['sha256']=='bf94468e5b8d613e027c1d46fbb7991a672870d5a6f0159ec9511af1d8285dac'
    assert cfg['actual_execution_admitted'] is False

def test_no_torch_or_scientific_module_imported():
    assert 'torch' not in sys.modules
    assert 'arggnn.anchor_compact_actual_bank_v1' not in sys.modules
    assert 'arggnn.anchor_compact_qualification_runner_v1' not in sys.modules

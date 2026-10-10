"""Independent extension/IPC review with isolated handmade CPU objects."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import pytest
from arggnn import gpu_pool_v1 as pool
PROJECT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=PROJECT/"research/reviews/gpu_pool_research_v2_independent"
spec=importlib.util.spec_from_file_location("independent_pool_scaffold",OWN/"test_focused_replay_v1.py")
scaffold=importlib.util.module_from_spec(spec);spec.loader.exec_module(scaffold)
extension=scaffold.extension
bound_fixture=scaffold.bound_fixture

def change_config(root,cfg,monkeypatch):
    data=(json.dumps(cfg,sort_keys=True)+"\n").encode()
    (root/extension.CONFIG).write_bytes(data)
    monkeypatch.setattr(extension,"CONFIG_SHA256",hashlib.sha256(data).hexdigest())

@pytest.mark.parametrize("key,value",[
    ("kind","other"),("status","active"),("project_root","/other"),
    ("actual_execution_admitted",0),("root_bootstrap_and_fresh_admission_required",1),
    ("allowed_modules",sorted(pool.MODULES)),("direct_runner_submission_allowed",0)
])
def test_independent_exact_extension_config(bound_fixture,key,value,monkeypatch):
    root,cfg=bound_fixture;cfg[key]=value;change_config(root,cfg,monkeypatch)
    with pytest.raises(ValueError):extension.verify_bindings(root)

@pytest.mark.parametrize("which,key,value",[
    ("extension_worker","module","arggnn.arbitrary"),
    ("bootstrap_worker","module",extension.WORKER),
    ("bootstrap_worker","independent_review_required_before_execution",1)
])
def test_independent_added_module_fields(bound_fixture,which,key,value,monkeypatch):
    root,cfg=bound_fixture;cfg[which][key]=value;change_config(root,cfg,monkeypatch)
    with pytest.raises(ValueError):extension.verify_bindings(root)

@pytest.mark.parametrize("key,value",[
    ("verdict","BLOCKED"),("runner_sha256","0"*64),("protocol_sha256","0"*64),
    ("actual_execution_admitted",True),("actual_execution_admitted",0)
])
def test_independent_runner_review_content(bound_fixture,key,value,monkeypatch):
    root,cfg=bound_fixture;record=cfg["extension_worker"]["independent_review"];path=root/record["path"]
    review=json.loads(path.read_text());review[key]=value;data=(json.dumps(review)+"\n").encode();path.write_bytes(data)
    record["sha256"]=hashlib.sha256(data).hexdigest();change_config(root,cfg,monkeypatch)
    with pytest.raises(ValueError):extension.verify_bindings(root)

@pytest.mark.parametrize("which",["missing","symlink","oversize"])
def test_independent_bound_regular_canonical_bounded_sources(bound_fixture,which):
    root,cfg=bound_fixture;path=root/cfg["bootstrap_worker"]["source"]["path"]
    if which=="missing":path.unlink()
    elif which=="symlink":
        target=path.with_name("handmade_copy.py");target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target)
    else:path.write_bytes(b"x"*(1024*1024+1))
    with pytest.raises(ValueError):extension.verify_bindings(root)

def test_independent_noncanonical_injected_root_rejects(bound_fixture):
    root,cfg=bound_fixture;link=root/"linked";link.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError):extension.verify_bindings(link)

@pytest.mark.parametrize("which",["extension","base"])
def test_independent_canonical_entrypoint_gates_before_ipc(which,monkeypatch):
    target=extension if which=="extension" else pool
    monkeypatch.setattr(target,"__file__",str(PROJECT/"research/reviews/other.py"))
    with mock.patch.object(pool,"request") as ipc,mock.patch.object(pool,"Controller") as ctrl:
        with pytest.raises(ValueError):extension.main(scaffold.submit_argv())
    ipc.assert_not_called();ctrl.assert_not_called()

@pytest.mark.parametrize("command",[scaffold.start_argv(),scaffold.submit_argv()])
def test_independent_source_failure_prevents_controller_and_ipc(command,monkeypatch):
    monkeypatch.setattr(extension,"verify_bindings",mock.Mock(side_effect=ValueError("frozen bytes drift")))
    with mock.patch.object(pool,"request") as ipc,mock.patch.object(pool,"Controller") as ctrl:
        with pytest.raises(ValueError):extension.main(command)
    ipc.assert_not_called();ctrl.assert_not_called()

@pytest.mark.parametrize("command",[
    ["status","--pool","p"],["wait","--pool","p","--job","j"],
    ["shutdown","--pool","p"],["shutdown","--pool","p","--terminate"]
])
def test_independent_maintenance_remains_available_on_hash_drift(command,monkeypatch):
    monkeypatch.setattr(extension,"verify_bindings",mock.Mock(side_effect=ValueError("drift")))
    with mock.patch.object(pool,"main",return_value=7) as delegated:
        assert extension.main(command)==7
    delegated.assert_called_once_with(command)

@pytest.mark.parametrize("module",[extension.WORKER,"arggnn.arbitrary","arggnn.anchor_compact_cuda_bootstrap_v2"])
def test_independent_cli_direct_arbitrary_module_rejects_without_ipc(module):
    with mock.patch.object(pool,"request") as ipc,mock.patch.object(pool,"Controller") as ctrl:
        with pytest.raises((ValueError,SystemExit)):extension.main(scaffold.submit_argv(module))
    ipc.assert_not_called();ctrl.assert_not_called()

@pytest.mark.parametrize("key,value",[
    ("job","../outside"),("job",""),("model",""),("seed",-1),("seed",1.0),
    ("peak_mib",0),("peak_mib",0.5),("storage_bytes",0),("log_limit_bytes",0),
    ("log_limit_bytes",1001048577),("args",["x"]*1025),("args",[False]),("module",True),
    ("output","outputs/missing/new")
])
def test_independent_unchanged_spec_types_shape_paths(tmp_path,key,value):
    ctrl=scaffold.bare_controller(tmp_path);spec=scaffold.worker_spec(tmp_path);spec[key]=value
    with pytest.raises((ValueError,TypeError)):ctrl._validate_spec(spec)
    assert ctrl.jobs=={} and not (tmp_path/"outputs/new").exists()

@pytest.mark.parametrize("mode",["duplicate","existing_output","symlink_output","too_large_peak","unknown_field"])
def test_independent_spec_admission_cases(tmp_path,mode):
    ctrl=scaffold.bare_controller(tmp_path);spec=scaffold.worker_spec(tmp_path)
    if mode=="duplicate":ctrl.jobs={"j":{"source_only":True}}
    elif mode=="existing_output":Path(spec["output"]).mkdir()
    elif mode=="symlink_output":Path(spec["output"]).symlink_to(tmp_path/"outputs/target")
    elif mode=="too_large_peak":spec["peak_mib"]=81921
    else:spec["unreviewed"]="x"
    with pytest.raises(ValueError):ctrl._validate_spec(spec)

@pytest.mark.parametrize("request",[
    {"v":1.0,"op":"submit","spec":{}},{"v":1,"op":"submit","spec":{},"extra":1},
    {"v":1,"op":"submit"},{"v":1,"op":"exec","spec":{}},
    {"v":1,"op":"shutdown","terminate":False,"extra":1},
])
def test_independent_dispatch_exact_v1(request,tmp_path):
    ctrl=scaffold.bare_controller(tmp_path)
    with mock.patch.object(ctrl,"submit") as submit,mock.patch.object(ctrl,"shutdown",create=True) as shutdown:
        with pytest.raises(ValueError):ctrl.dispatch(request)
    submit.assert_not_called();shutdown.assert_not_called()

def test_observed_raw_ipc_runner_allowlist_is_admission_only(tmp_path):
    """Preserve the exact weaker boundary: raw IPC accepts its spec, never runs here."""
    ctrl=scaffold.bare_controller(tmp_path);spec=scaffold.worker_spec(tmp_path);spec["module"]=extension.WORKER
    result=ctrl._validate_spec(spec)
    assert result["module"]==extension.WORKER and not Path(result["output"]).exists()
    assert ctrl.jobs=={}

def test_direct_runner_source_body_blocks_before_driver_context_or_nvtop():
    """Invoke the frozen observer with handmade precontext dependencies and no live reads."""
    tree=ast.parse((PROJECT/"src/arggnn/anchor_compact_qualification_runner_v1.py").read_text())
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="observe_live"))
    node.body=[n for n in node.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    class HandmadePath:
        def __init__(self,value):assert value=="/proc/self/cmdline"
        def read_bytes(self):return b"AI Auto Research Pipeline Test\0"
    def require(value,message):
        if not value:raise ValueError(message)
    driver=mock.Mock();nvtop=mock.Mock()
    scope=dict(validate_lease=lambda:{"source_only_fixture":True},Path=HandmadePath,
        sealed=SimpleNamespace(PROCESS_TITLE="AI Auto Research Pipeline Test"),require=require,
        deterministic_development_manifest=lambda:{"strict":True},
        torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)),
        ctypes=SimpleNamespace(CDLL=driver),subprocess=SimpleNamespace(run=nvtop))
    exec(compile(ast.Module(body=[node],type_ignores=[]),"frozen_observe_live_body","exec"),scope)
    with pytest.raises(ValueError,match="own CUDA context first"):scope["observe_live"]({})
    driver.assert_not_called();nvtop.assert_not_called()

def test_original_scheduler_allowlist_object_stays_unchanged(bound_fixture):
    root,cfg=bound_fixture;original=pool.MODULES
    assert extension.verify_bindings(root)==cfg
    assert pool.MODULES is original and len(original)==6
    assert extension.MODULES==original|{extension.WORKER,extension.BOOTSTRAP}
    source=ast.parse((PROJECT/"scripts/gpu_pool_research_v2.py").read_text())
    assignments=[n for n in ast.walk(source) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign))]
    assert not any(isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=="pool" for n in assignments for target in (n.targets if isinstance(n,ast.Assign) else [n.target]))

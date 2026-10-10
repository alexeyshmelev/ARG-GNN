"""Source-only/fake-context tests for separate reduced handmade qualification."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/biological_paths_tsinfer_native_failure_v1"
spec=importlib.util.spec_from_file_location("handmade_qualification_v2",OUT/"qualify_handmade_rescaling_v2.py")
h=importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
def cfg():return json.loads((OUT/"qualification_v2.json").read_text())

def test_source_only_config_reuses_frozen_pipeline_with_one_explicit_delta(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split(".")[0] in {"tsdate","tsinfer"}:
            raise AssertionError("native inference module imported")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    base=h.load_producer()
    biological,effective,prior=h.validate_config(base,cfg())
    assert biological["dating"]["rescaling_intervals"]==1000
    assert effective["dating"]["rescaling_intervals"]==1
    old=copy.deepcopy(biological)
    new=copy.deepcopy(effective)
    old["dating"]["rescaling_intervals"]=1
    old["qualification_root"]=cfg()["qualification_root"]
    old["storage_accounting"]["qualification_launch_log"]=cfg()["qualification_launch_log"]
    assert old==new
    assert base.handmade_assets()["identity_support_original_bp0"]==[10,61]
    assert len(base.handmade_assets()["sites"])==6

@pytest.mark.parametrize("field,value",[
    ("qualification_dating",{}),("actual_observed_inference_authorized",True),
    ("silent_retry_or_parameter_fallback",True),("certification","same1000"),
    ("include_preserved_v3_qualification_scopes_in_1GB_local_cap",False)])
def test_config_mutations_require_new_source_condition(field,value):
    c=cfg();c[field]=value
    with pytest.raises(ValueError,match="exact qualification"):
        h.validate_config(h.load_producer(),c)

def test_missing_root_hashes_fail_before_runtime_or_native_access(monkeypatch):
    base=h.load_producer()
    monkeypatch.setattr(base,"runtime_inventory",lambda *args:(_ for _ in()).throw(AssertionError("runtime visited")))
    c=cfg();biological,effective,prior=h.validate_config(base,c)
    args=SimpleNamespace(expected_config_sha256=None,expected_source_sha256=None,
                         admission=None,expected_admission_sha256=None)
    with pytest.raises(ValueError,match="exact root"):
        h.admission(base,c,effective,prior,args,OUT/"qualification_v2.json")

def test_infer_mode_is_not_an_entrypoint(monkeypatch):
    monkeypatch.setattr(sys,"argv",["qualification","--mode","infer"])
    monkeypatch.setattr(h,"load_producer",lambda:(_ for _ in()).throw(AssertionError("producer accessed")))
    with pytest.raises(SystemExit) as error:
        h.main()
    assert error.value.code==2

def prior_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(h,"ROOT",tmp_path)
    tree=tmp_path/"oldtree";tree.mkdir()
    (tree/"tiny.trees").write_text("hand-authored fake bytes, not an array")
    log=tmp_path/"old.log";log.write_text("handmade log")
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prior={"scope_paths":["oldtree","old.log"],"files":{
        "oldtree/tiny.trees":{"sha256":sha(tree/"tiny.trees"),"bytes":(tree/"tiny.trees").stat().st_size},
        "old.log":{"sha256":sha(log),"bytes":log.stat().st_size}}}
    base=SimpleNamespace(confined=lambda p:tmp_path/p,sha=sha)
    return base,prior,tree,log

def test_preserved_scope_hashes_are_checked_readonly(tmp_path,monkeypatch):
    base,prior,tree,log=prior_fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in [tree/"tiny.trees",log]}
    assert h.verify_prior_scopes(base,prior)
    assert all(p.read_bytes()==v for p,v in before.items())

@pytest.mark.parametrize("change",["changed","added","symlink"])
def test_changed_or_extended_previous_scope_is_not_adopted(tmp_path,monkeypatch,change):
    base,prior,tree,log=prior_fixture(tmp_path,monkeypatch)
    if change=="changed":log.write_text("changed")
    elif change=="added":(tree/"added").write_text("added")
    else:(tree/"dangling").symlink_to(tree/"missing")
    with pytest.raises(ValueError):
        h.verify_prior_scopes(base,prior)

def storage_fixture(monkeypatch,new_bytes=100,prior_bytes=200):
    c=cfg();prior={"scope_paths":["oldtree","oldlog"]}
    values={"new":new_bytes,"oldtree":prior_bytes,"oldlog":0}
    class FakeStorage:
        def __init__(self,effective,receipt,runtime):
            self.policy={"qualify_local_write_cap_bytes":1000000000,
                         "qualify_reserved_bytes":8000000000}
            self.installation_charge=7000000000
            self.calls=0
            self.check()
        def check(self):
            self.calls+=1
            return {"local_current_bytes":values["new"],"local_high_water_bytes":values["new"],
                    "projected_bytes":439875052544}
    base=SimpleNamespace(StorageAccounting=FakeStorage,
         ref=lambda p:{"path":str(p),"sha256":"handmade"},
         confined=lambda p:p,local_scope_bytes=lambda p:values[p])
    receipt={"harness_config":base.ref(h.ROOT/h.DEFAULT_CONFIG),
             "harness_implementation":base.ref(Path(h.__file__)),
             "previous_failed_scope_pins":c["previous_failed_scope_pins"],
             "qualification_write_scopes":[c["qualification_root"],c["qualification_launch_log"],*prior["scope_paths"]]}
    context=h.storage_context(base,c,{},receipt,prior)
    return context,values,base,c,receipt,prior

def test_storage_subclass_delegates_shared_guard_and_combines_prior_scope(monkeypatch):
    context,values,*_=storage_fixture(monkeypatch)
    assert context.calls==1
    result=context.check()
    assert context.calls==2
    assert result["local_current_bytes"]==300 and result["preserved_v3_local_bytes"]==200
    assert result["projected_bytes"]==439875052544
    values["new"]=50
    assert context.check()["local_high_water_bytes"]==300

def test_preserved_plus_new_local_cap_cannot_double_spend_slot(monkeypatch):
    context,values,*_=storage_fixture(monkeypatch,900000000,100000000)
    assert context.check()["local_current_bytes"]==1000000000
    values["new"]+=1
    with pytest.raises(ValueError,match="combined preserved"):
        context.check()

def test_storage_receipt_cannot_omit_prior_scope(monkeypatch):
    context,values,base,c,r,prior=storage_fixture(monkeypatch)
    r["qualification_write_scopes"]=r["qualification_write_scopes"][:2]
    with pytest.raises(ValueError,match="all current/previous"):
        h.storage_context(base,c,{},r,prior)

def test_pipeline_date_bindings_are_never_reassigned():
    tree=ast.parse(Path(h.__file__).read_text())
    writes=[ast.unparse(t) for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))
            for t in (n.targets if isinstance(n,ast.Assign) else [n.target])]
    assert not any(w in {"base.pipeline","base.qualification_summary","base.handmade_assets","tsdate.date"} for w in writes)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=="base.pipeline"]
    assert len(calls)==1
    assert [ast.unparse(x) for x in calls[0].args]==["effective","case","target","context"]

def test_separate_manifest_kind_cannot_certify_original_infer_contract():
    assert cfg()["kind"]!="biological_paths_tsinfer_v1_cpu_qualification"
    assert cfg()["certification"]=="reduced_single_interval_rescaling_only_not_exact1000_interval_biological_qualification"
    assert cfg()["actual_observed_inference_authorized"] is False

def test_old_v3_root_admission_cannot_authorize_reduced_v2(monkeypatch):
    base=h.load_producer()
    c=cfg();biological,effective,prior=h.validate_config(base,c)
    path=ROOT/"research/reviews/biological_paths_tsinfer_v1_root_admission_v3/root_cpu_admission.json"
    args=SimpleNamespace(expected_config_sha256=base.sha(OUT/"qualification_v2.json"),
                         expected_source_sha256=base.sha(Path(h.__file__)),
                         admission=str(path),expected_admission_sha256=base.sha(path))
    monkeypatch.setattr(base,"runtime_inventory",lambda *args:(_ for _ in()).throw(AssertionError("runtime visited")))
    with pytest.raises(ValueError,match="reduced handmade-only"):
        h.admission(base,c,effective,prior,args,OUT/"qualification_v2.json")

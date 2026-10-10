"""Source/handmade/fake-ledger checks; no observed arrays or native pipeline."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
AUDIT=ROOT/"research/reviews/biological_paths_tsinfer_observed_v2_author"
spec=importlib.util.spec_from_file_location("observed_v2_owned",ROOT/"scripts/infer_biological_paths_tsinfer_observed_v2.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def config():return json.loads((ROOT/"configs/biological_paths_tsinfer_observed_v2.json").read_text())

def test_source_only_import_and_file_whitelist_guard(monkeypatch):
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.split(".")[0] in {"tsinfer","tsdate"}:
            raise AssertionError("native module imported")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",blocked)
    original_open=Path.open
    def open_guard(path,*args,**kwargs):
        if path.suffix in {".vcf",".npz",".trees"}:
            raise AssertionError("actual array opened")
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",open_guard)
    base=h.load_producer();biological,lock=h.validate_config(base,config())
    assert biological["dating"]["rescaling_intervals"]==1000
    assert set(lock["parents"])==set(base.PARENTS)
    assert config()["phases"]["canary"]==["bio_paths_pilot_train_000"]
    assert len(config()["input_files"])==9

@pytest.mark.parametrize("field,value",[
    ("scientific_recipe",{"rescaling_intervals":1}),
    ("truth_or_oracle_inputs_authorized",True),
    ("heldout_test_or_publication_predictions_authorized",True),
    ("silent_parameter_fallback",True),
    ("phases",{"canary":["bio_paths_pilot_val_000"],"remaining":[]}),
    ("input_files",["truth.trees"]),
])
def test_config_drift_cannot_expand_or_change_science(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError,match="exact observed"):
        h.validate_config(h.load_producer(),c)

def test_exact_contract_distinguishes_bool_and_integer():
    assert not h.exact({"intervals":True},{"intervals":1})
    assert not h.exact({"authorized":0},{"authorized":False})

def test_no_root_hash_optin_fails_before_runtime_and_inputs(monkeypatch):
    base=h.load_producer();c=config();b,lock=h.validate_config(base,c)
    monkeypatch.setattr(base,"runtime_inventory",lambda *args:(_ for _ in()).throw(AssertionError("runtime visited")))
    args=SimpleNamespace(expected_config_sha256=None,expected_source_sha256=None,
                         admission=None,expected_admission_sha256=None,mode="canary")
    with pytest.raises(ValueError,match="exact root"):
        h.admission(base,c,b,lock,args,ROOT/h.DEFAULT_CONFIG)

def root_admission(base,c,mode):
    return {"kind":"biological_paths_tsinfer_observed_v2_root_admission","authorizer":"root_agent",
        "mode":mode,"condition":h.CONDITION,"config":base.ref(ROOT/h.DEFAULT_CONFIG),
        "implementation":base.ref(Path(h.__file__)),"producer":c["producer"],
        "biological_config":c["biological_config"],"runtime":c["runtime"],
        "observed_asset_pins":c["observed_asset_pins"],"scope_parents":c["phases"][mode],
        "rescaling_intervals":1000,"reduced_smoke_outcome":c["reduced_smoke_outcome"],
        "reduced_smoke_only":True,"truth_or_oracle_inputs_authorized":False,
        "heldout_test_or_publication_predictions_authorized":False,
        "observed_development_genotypes_authorized":True,"silent_parameter_fallback":False,
        "experiment_registration":c["reduced_smoke_outcome"]}

def test_remaining_never_accepts_reduced1_smoke_as1000_canary(tmp_path,monkeypatch):
    base=h.load_producer();c=config();b,lock=h.validate_config(base,c)
    a=root_admission(base,c,"remaining")
    a["canary_adequacy"]=c["reduced_smoke_outcome"]
    a["canary_manifest"]=c["reduced_smoke_outcome"]
    p=tmp_path/"fake_root_admission.json";p.write_text(json.dumps(a))
    args=SimpleNamespace(expected_config_sha256=base.sha(ROOT/h.DEFAULT_CONFIG),
        expected_source_sha256=base.sha(Path(h.__file__)),admission=str(p),
        expected_admission_sha256=base.sha(p),mode="remaining")
    monkeypatch.setattr(base,"runtime_inventory",lambda *args:(_ for _ in()).throw(AssertionError("runtime visited")))
    with pytest.raises(ValueError,match="same-source/runtime1000"):
        h.admission(base,c,b,lock,args,ROOT/h.DEFAULT_CONFIG)

def storage_fixture(tmp_path,monkeypatch):
    base=h.load_producer();c=config();b,_=h.validate_config(base,c)
    pool_spec=importlib.util.spec_from_file_location("observed_v2_fake_pool",ROOT/"src/arggnn/gpu_pool_v1.py")
    pool=importlib.util.module_from_spec(pool_spec);pool_spec.loader.exec_module(pool)
    monkeypatch.setattr(h,"ROOT",tmp_path);monkeypatch.setattr(base,"ROOT",tmp_path)
    source=tmp_path/"script.py";source.write_text("handmade source bytes")
    monkeypatch.setattr(h,"__file__",str(source))
    cfg=tmp_path/h.DEFAULT_CONFIG;cfg.parent.mkdir();cfg.write_text("{}")
    monkeypatch.setattr(base,"storage_module",lambda biological:pool)
    monkeypatch.setattr(h.os,"statvfs",lambda root:SimpleNamespace(f_bavail=10**15,f_frsize=1))
    snapshot=tmp_path/"snapshot.json"
    snapshot.write_text(json.dumps({"schema":"biological_candidate_storage_snapshot_v1","status":"complete","returncode":0,
        "project_root":str(tmp_path),"command":["du","-s","-B1",str(tmp_path)],
        "used_bytes":167336640512,"completed_utc":"2026-10-10T17:31:06.897981+00:00"}))
    c["storage"]["snapshot"]={"path":"snapshot.json","sha256":base.sha(snapshot)}
    p=c["storage"];prefix=tmp_path/b["runtime_prefix"];prefix.mkdir(parents=True)
    ledger_path=tmp_path/p["ledger_path"];ledger_path.parent.mkdir(parents=True)
    jobs={p["reservation_key"]:{"reserved_bytes":20000000000,"output":str(tmp_path/c["output_root"])},
          p["readonly_runtime_reservation_key"]:{"reserved_bytes":8000000000,"output":str(prefix)},
          "candidate202":{"reserved_bytes":29281048576,"output":str(tmp_path/"candidate202")},
          "GST101":{"reserved_bytes":2000000000,"output":str(tmp_path/"GST101")},
          "candidate303":{"reserved_bytes":29281048576,"output":str(tmp_path/"candidate303")},
          "mechanisms101":{"reserved_bytes":52321048576,"output":str(tmp_path/"mechanisms101")}}
    ledger={"schema":"gpu_pool_storage_ledger_v1","root":str(tmp_path),"snapshot_sha256":p["snapshot"]["sha256"],
        "baseline_bytes":167336640512,"external_reserve_bytes":130000000000,"limit_bytes":450000000000,
        "controllers":{"gpu2":{"reserved_bytes":67108864},"gpu3":{"reserved_bytes":67108864}},"jobs":jobs}
    ledger_path.write_text(json.dumps(ledger))
    receipt={"kind":"biological_paths_tsinfer_observed_v2_storage_admission","mode":"canary","project_root":str(tmp_path),
        "config":base.ref(cfg),"implementation":base.ref(source),"producer":c["producer"],"runtime":c["runtime"],
        "snapshot":p["snapshot"],"reservation_key":p["reservation_key"],"reserved_bytes":20000000000,
        "local_write_cap_bytes":20000000000,"warning_bytes":450000000000,"hard_limit_bytes":500000000000,
        "write_scopes":[c["output_root"],*c["phase_launch_logs"].values()],
        "readonly_runtime_reservation_key":p["readonly_runtime_reservation_key"],
        "readonly_runtime_reserved_bytes":8000000000,"ledger_sha256_at_admission":base.sha(ledger_path)}
    return base,c,b,receipt,ledger_path,ledger

def test_all_jobs_controllers_external_and20GB_are_readonly_accounted(tmp_path,monkeypatch):
    base,c,b,r,path,state=storage_fixture(tmp_path,monkeypatch)
    before=path.read_bytes();context=h.storage_context(base,c,b,r,"canary")
    result=base.storage_check(b,context)  # frozen pipeline protocol accepts subclass
    assert result["projected_bytes"]==438354003968
    assert result["persistent_reservations"]==6
    assert result["readonly_runtime_slot_retained"]
    assert path.read_bytes()==before

@pytest.mark.parametrize("slot,value",[("observed",8000000000),("runtime",0)])
def test_reservation_cannot_borrow_runtime_or_change_slot(tmp_path,monkeypatch,slot,value):
    base,c,b,r,path,state=storage_fixture(tmp_path,monkeypatch)
    key=c["storage"]["reservation_key" if slot=="observed" else "readonly_runtime_reservation_key"]
    state["jobs"][key]["reserved_bytes"]=value
    path.write_text(json.dumps(state));r["ledger_sha256_at_admission"]=base.sha(path)
    with pytest.raises(ValueError,match="reservation"):
        h.storage_context(base,c,b,r,"canary")

def test_stale_initial_receipt_rejected(tmp_path,monkeypatch):
    base,c,b,r,path,state=storage_fixture(tmp_path,monkeypatch)
    r["ledger_sha256_at_admission"]="0"*64
    with pytest.raises(ValueError,match="renew root"):
        h.storage_context(base,c,b,r,"canary")

@pytest.mark.parametrize("field,value",[("baseline_bytes",1),("external_reserve_bytes",0),
    ("snapshot_sha256","0"*64),("limit_bytes",500000000000)])
def test_no_silent_rebase(tmp_path,monkeypatch,field,value):
    base,c,b,r,path,state=storage_fixture(tmp_path,monkeypatch)
    state[field]=value;path.write_text(json.dumps(state));r["ledger_sha256_at_admission"]=base.sha(path)
    with pytest.raises(ValueError,match="rebase"):
        h.storage_context(base,c,b,r,"canary")

def test_exact450GB_and_free_space_overcommit_rejected(tmp_path,monkeypatch):
    base,c,b,r,path,state=storage_fixture(tmp_path,monkeypatch)
    context=h.storage_context(base,c,b,r,"canary")
    state["jobs"]["extra"]={"reserved_bytes":11645996032,"output":str(tmp_path/"extra")}
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match="ceiling"):context.check()

def test_filesystem_headroom_is_checked(tmp_path,monkeypatch):
    base,c,b,r,path,state=storage_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(h.os,"statvfs",lambda root:SimpleNamespace(f_bavail=1,f_frsize=1))
    with pytest.raises(ValueError,match="free-space"):h.storage_context(base,c,b,r,"canary")

def test_canary_remaining_JIT_temp_and_both_logs_share_one_cap(tmp_path,monkeypatch):
    base,c,b,r,path,state=storage_fixture(tmp_path,monkeypatch)
    context=h.storage_context(base,c,b,r,"canary")
    for suffix in ["canary/a.trees","remaining/b.trees","canary/write_scopes/numba_cache_dir/jit.bin",
                   "remaining/write_scopes/tmpdir/transient"]:
        p=tmp_path/c["output_root"]/suffix;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"x"*5000)
    for path in c["phase_launch_logs"].values():
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"log"*3000)
    before=context.check();assert before["local_current_bytes"]>=38000
    (tmp_path/c["output_root"]/"remaining/write_scopes/tmpdir/transient").unlink()
    after=context.check()
    assert after["local_current_bytes"]<before["local_current_bytes"]
    assert after["local_high_water_bytes"]==before["local_high_water_bytes"]
    monkeypatch.setattr(base,"local_scope_bytes",lambda p:20000000001 if p==tmp_path/c["output_root"] else 0)
    with pytest.raises(ValueError,match="20GB local cap"):context.check()

def test_storage_receipt_must_bind_both_phase_logs(tmp_path,monkeypatch):
    base,c,b,r,path,state=storage_fixture(tmp_path,monkeypatch)
    r["write_scopes"]=r["write_scopes"][:-1]
    with pytest.raises(ValueError,match="all scopes"):h.storage_context(base,c,b,r,"canary")

def metadata_fixture(tmp_path,monkeypatch):
    base=h.load_producer();c=config()
    monkeypatch.setattr(h,"ROOT",tmp_path);monkeypatch.setattr(base,"ROOT",tmp_path)
    src=tmp_path/"script.py";src.write_text("handmade source");monkeypatch.setattr(h,"__file__",str(src))
    cfg=tmp_path/h.DEFAULT_CONFIG;cfg.parent.mkdir();cfg.write_text("{}")
    c["output_root"]="handmade_observed"
    target=tmp_path/c["output_root"]/"canary"
    work=target/"bio_paths_pilot_train_000";work.mkdir(parents=True)
    tree=work/"observed.trees";tree.write_text("fake primary bytes, not a tree sequence")
    roster=work/"observed_sample_manifest.json";roster.write_text("[]")
    hashes={name:"0"*64 for name in c["input_files"]}
    pin={"observed_files_sha256":hashes,"comparison_coverage_original_bp0":[10,61],
         "comparison_retained_fixed100bp_indexes":[0]}
    lock={"parents":{"bio_paths_pilot_train_000":pin}}
    verification={"exact_observed_genotypes_and_ages":True,"strict_native_times":True,
        "source_labels_passed_to_inference":False,"truth_or_oracle_tree_read":False,
        "all_original_sites_preserved":6,"identity_support_original_bp0":[10,61]}
    (work/"native_verification.json").write_text(json.dumps(verification))
    names=["modern_inferred","modern_preprocessed","modern_dated","modern_site_ages","ancient_site_bounds",
        "ancestors_generated","historical_proxy_ancestors","ancestors_matched","all_samples_matched","all_preprocessed","all_dated"]
    (work/"stages.json").write_text(json.dumps([{"stage":x,"status":"complete"} for x in names]))
    native={"parent_id":"bio_paths_pilot_train_000","status":"qualified_observed_cpu_inference",
        "input_observed_sha256":hashes,"native_node_ids_and_times_preserved_during_sanitation":True,
        "no_input_variant_filtering":True,"truth_or_oracle_read":False,"native_full_sites":6,
        "inferred_tree":base.ref(tree),"observed_sample_manifest":base.ref(roster),
        "coordinate_contract":{"coverage_original_bp0":[10,61],"retained_fixed100bp_indexes":[0]},
        "connected_identity":{"internal_reuse_occurrences":1,"uncovered_interval_inside_support_admitted":False}}
    completion=work/"inference.json";completion.write_text(json.dumps(native))
    native["completion_provenance"]=base.ref(completion)
    adequacy=h.parent_adequacy(base,c,lock,native)
    row={"parent_id":native["parent_id"],"frozen_helper_result":native,"adequacy":adequacy}
    manifest={"kind":"biological_paths_tsinfer_observed_v2_canary","status":"passed_train000_observed_canary1000",
        "mode":"canary","condition":h.CONDITION,"config":base.ref(cfg),"implementation":base.ref(src),
        "producer":c["producer"],"biological_config":c["biological_config"],"runtime":c["runtime"],
        "observed_asset_pins":c["observed_asset_pins"],"scope_parents":c["phases"]["canary"],
        "rescaling_intervals":1000,"reduced_smoke_only":True,"native_paper_reproduction_claimed":False,
        "truth_or_oracle_read":False,"heldout_test_or_publication_predictions":False,"replicates":[row]}
    manifest_path=target/"manifest.json";manifest_path.write_text(json.dumps(manifest))
    review={"kind":"biological_paths_tsinfer_observed_v2_root_canary_adequacy","authorizer":"root_agent",
        "status":"passed_source_input_genotype_date_support_identity_canary1000",
        "canary_manifest":base.ref(manifest_path),"config":base.ref(cfg),"implementation":base.ref(src),
        "runtime":c["runtime"],"producer":c["producer"],"biological_config":c["biological_config"],
        "observed_asset_pins":c["observed_asset_pins"],"biological1000_observed_canary":True,
        "reduced_smoke_is_same1000_qualification":False}
    review_path=tmp_path/"adequacy.json";review_path.write_text(json.dumps(review))
    admission={"canary_manifest":base.ref(manifest_path),"canary_adequacy":base.ref(review_path)}
    return base,c,lock,native,manifest,review,manifest_path,review_path,admission

def test_canary_gate_accepts_bound_handmade_metadata_only(tmp_path,monkeypatch):
    base,c,lock,*rest=metadata_fixture(tmp_path,monkeypatch)
    assert h.canary_gate(base,c,lock,rest[-1])["rescaling_intervals"]==1000

@pytest.mark.parametrize("field,value",[("rescaling_intervals",1),("status","failed_partial_preserved"),
    ("scope_parents",["bio_paths_pilot_val_000"]),("runtime",{"path":"other","sha256":"0"*64})])
def test_canary_manifest_drift_blocks_remaining(tmp_path,monkeypatch,field,value):
    base,c,lock,native,m,review,mp,rp,a=metadata_fixture(tmp_path,monkeypatch)
    m[field]=value;mp.write_text(json.dumps(m));a["canary_manifest"]=base.ref(mp)
    review["canary_manifest"]=a["canary_manifest"];rp.write_text(json.dumps(review));a["canary_adequacy"]=base.ref(rp)
    with pytest.raises(ValueError,match="same1000"):h.canary_gate(base,c,lock,a)

def test_native_completion_metadata_cannot_be_rewritten_in_outer_manifest(tmp_path,monkeypatch):
    base,c,lock,native,*_=metadata_fixture(tmp_path,monkeypatch)
    native["native_full_sites"]=7
    with pytest.raises(ValueError,match="preserved frozen-helper"):
        h.parent_adequacy(base,c,lock,native)

def test_primary_canary_artifact_bytes_are_rechecked(tmp_path,monkeypatch):
    base,c,lock,native,m,review,mp,rp,a=metadata_fixture(tmp_path,monkeypatch)
    (tmp_path/native["inferred_tree"]["path"]).write_text("changed")
    with pytest.raises(ValueError,match="artifact bytes changed"):h.canary_gate(base,c,lock,a)

def test_native_failure_is_not_converted_into_parameter_retry(tmp_path,monkeypatch):
    base=h.load_producer();c=config();b,lock=h.validate_config(base,c)
    calls=[]
    def fail(biological,parent,target,pin,context):
        calls.append(biological["dating"]["rescaling_intervals"])
        raise AssertionError("Use fewer rescaling intervals")
    fake=SimpleNamespace(infer_parent=fail)
    context=SimpleNamespace(check=lambda:None)
    with pytest.raises(AssertionError,match="fewer rescaling"):
        h.run_parent(fake,c,b,lock,"bio_paths_pilot_train_000",tmp_path,context)
    assert calls==[1000] and b["dating"]["rescaling_intervals"]==1000

def test_direct_frozen_helper_calls_no_global_rebinding():
    tree=ast.parse((ROOT/"scripts/infer_biological_paths_tsinfer_observed_v2.py").read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=="base.infer_parent"]
    assert len(calls)==1
    assert [ast.unparse(a) for a in calls[0].args]==["biological","parent","target","lock['parents'][parent]","context"]
    writes=[ast.unparse(t) for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))
            for t in (n.targets if isinstance(n,ast.Assign) else [n.target])]
    assert not any(w in {"base.infer_parent","base.pipeline","base.CONDITION","tsdate.date"} for w in writes)
    assert not any("project_bytes" in ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call))

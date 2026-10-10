"""Independent observed harness metadata/storage probes; no real inputs."""
import copy, importlib.util, json
from pathlib import Path
import pytest
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
spec=importlib.util.spec_from_file_location("observed_author_helpers",ROOT/"research/reviews/biological_paths_tsinfer_observed_v2_author/test_observed_v2_source_only.py")
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
h=t.h


def update_manifest(base,native,m,review,mp,rp,a,work):
    cp=work/"inference.json"
    cp.write_text(json.dumps({k:v for k,v in native.items() if k!="completion_provenance"}))
    native["completion_provenance"]=base.ref(cp)
    m["replicates"][0]["frozen_helper_result"]=native
    return cp


def reseal(base,m,review,mp,rp,a):
    mp.write_text(json.dumps(m));a["canary_manifest"]=base.ref(mp)
    review["canary_manifest"]=a["canary_manifest"]
    rp.write_text(json.dumps(review));a["canary_adequacy"]=base.ref(rp)


def test_counterexample_other_parent_native_evidence_accepted_under_TRAIN000_outer_row(tmp_path,monkeypatch):
    base,c,lock,native,m,review,mp,rp,a=t.metadata_fixture(tmp_path,monkeypatch)
    other="bio_paths_pilot_train_001"
    pin=copy.deepcopy(lock["parents"]["bio_paths_pilot_train_000"])
    pin["observed_files_sha256"]={key:"1"*64 for key in pin["observed_files_sha256"]}
    pin["comparison_coverage_original_bp0"]=[20,80]
    lock["parents"][other]=pin
    native["parent_id"]=other
    native["input_observed_sha256"]=pin["observed_files_sha256"]
    native["coordinate_contract"]["coverage_original_bp0"]=[20,80]
    work=mp.parent/"bio_paths_pilot_train_000"
    verification=json.loads((work/"native_verification.json").read_text())
    verification["identity_support_original_bp0"]=[20,80]
    (work/"native_verification.json").write_text(json.dumps(verification))
    update_manifest(base,native,m,review,mp,rp,a,work)
    m["replicates"][0]["adequacy"]=h.parent_adequacy(base,c,lock,native)
    reseal(base,m,review,mp,rp,a)
    accepted=h.canary_gate(base,c,lock,a)
    assert accepted["replicates"][0]["parent_id"]=="bio_paths_pilot_train_000"
    assert accepted["replicates"][0]["frozen_helper_result"]["parent_id"]==other


def test_counterexample_canary_native_scope_can_be_outside_reserved_output_tree(tmp_path,monkeypatch):
    base,c,lock,native,m,review,mp,rp,a=t.metadata_fixture(tmp_path,monkeypatch)
    original=mp.parent/"bio_paths_pilot_train_000"
    outside=tmp_path/"outside_20GB_observed_scope"/"bio_paths_pilot_train_000"
    outside.parent.mkdir()
    original.rename(outside)
    native["inferred_tree"]=base.ref(outside/"observed.trees")
    native["observed_sample_manifest"]=base.ref(outside/"observed_sample_manifest.json")
    update_manifest(base,native,m,review,mp,rp,a,outside)
    m["replicates"][0]["adequacy"]=h.parent_adequacy(base,c,lock,native)
    reseal(base,m,review,mp,rp,a)
    accepted=h.canary_gate(base,c,lock,a)
    completion=base.checked_ref(accepted["replicates"][0]["frozen_helper_result"]["completion_provenance"])
    assert not completion.is_relative_to(tmp_path/c["output_root"])


@pytest.mark.parametrize("change",["missing20GB","different20GBoutput","different8GBoutput"])
def test_missing_or_misdirected_slots_fail_before_native_access(tmp_path,monkeypatch,change):
    base,c,b,r,path,state=t.storage_fixture(tmp_path,monkeypatch)
    p=c["storage"]
    if change=="missing20GB":state["jobs"].pop(p["reservation_key"])
    elif change=="different20GBoutput":state["jobs"][p["reservation_key"]]["output"]=str(tmp_path/"wrong_output")
    else:state["jobs"][p["readonly_runtime_reservation_key"]]["output"]=str(tmp_path/"wrong_runtime")
    path.write_text(json.dumps(state));r["ledger_sha256_at_admission"]=base.sha(path)
    with pytest.raises(ValueError,match="reservation"):
        h.storage_context(base,c,b,r,"canary")


def test_additional_shared_job_and_controller_counted_after_admission(tmp_path,monkeypatch):
    base,c,b,r,path,state=t.storage_fixture(tmp_path,monkeypatch)
    context=h.storage_context(base,c,b,r,"canary")
    before=context.check()["projected_bytes"]
    state["jobs"]["new_logical_job"]={"reserved_bytes":1000,"output":str(tmp_path/"extra")}
    state["controllers"]["new_controller"]={"reserved_bytes":2000}
    path.write_text(json.dumps(state));unchanged=path.read_bytes()
    assert context.check()["projected_bytes"]==before+3000
    assert path.read_bytes()==unchanged


def test_remaining_context_recounts_prior_canary_in_shared20GB_cap(tmp_path,monkeypatch):
    base,c,b,r,path,state=t.storage_fixture(tmp_path,monkeypatch)
    r["mode"]="remaining"
    cap=20000000000
    values={str(tmp_path/c["output_root"]):cap-1,
            str(tmp_path/c["phase_launch_logs"]["canary"]):0,
            str(tmp_path/c["phase_launch_logs"]["remaining"]):0}
    monkeypatch.setattr(base,"local_scope_bytes",lambda p:values[str(p)])
    context=h.storage_context(base,c,b,r,"remaining")
    assert context.check()["local_current_bytes"]==cap-1
    values[str(tmp_path/c["phase_launch_logs"]["canary"])]=2
    with pytest.raises(ValueError,match="combined canary/remaining"):
        context.check()

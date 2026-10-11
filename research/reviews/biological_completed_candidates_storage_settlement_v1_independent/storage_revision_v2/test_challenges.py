"""Independent challenges using author handmade fixture builder under our owned temp roots."""
from pathlib import Path
import copy, importlib.util, json, os
import pytest
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
AUTHOR=ROOT/"research/reviews/biological_completed_candidates_storage_settlement_v1_author/storage_revision_v2"
spec=importlib.util.spec_from_file_location("independent_author_fixture",AUTHOR/"test_settlement.py")
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
h=a.h

def test_exact_two_delta_inverse_and_arbitrary_other_fields(tmp_path,monkeypatch):
    root,c,prior,module,authority,ledger,state=a.fixture(tmp_path,monkeypatch)
    state["extra_opaque_ledger_field"]={"nested":[1,True,None,"preserved"]}
    state["jobs"]["active_job"]["additional_field"]="retained"
    before=copy.deepcopy(state);after=h.proposal(c,state,module)
    assert state==before
    for t in c["targets"]:
        assert after["jobs"][t["key"]]["reserved_bytes"]==15000000000
        after["jobs"][t["key"]]["reserved_bytes"]=29281048576
    assert h.exact(after,before)
    assert sum(t["old_bytes"]-t["new_bytes"] for t in c["targets"])==28562097152

@pytest.mark.parametrize("change",["extra_target_field","boolean_charge","float_charge","wrong_root_output","missing_key"])
def test_proposal_rejects_nonexact_slots(tmp_path,monkeypatch,change):
    root,c,prior,module,authority,ledger,state=a.fixture(tmp_path,monkeypatch);t=c["targets"][0]
    slot=state["jobs"][t["key"]]
    if change=="extra_target_field":slot["other"]=1
    elif change=="boolean_charge":slot["reserved_bytes"]=True
    elif change=="float_charge":slot["reserved_bytes"]=float(t["old_bytes"])
    elif change=="wrong_root_output":slot["output"]=str(root/"unrelated")
    else:del state["jobs"][t["key"]]
    with pytest.raises(ValueError,match="exact original"):h.proposal(c,state,module)

@pytest.mark.parametrize("change",["symlink_output","symlink_intermediate","fifo","hardlink_log","symlink_log_parent"])
def test_retained_namespace_cases_fail_closed(tmp_path,monkeypatch,change):
    root,c,prior,module,authority,ledger,state=a.fixture(tmp_path,monkeypatch);t=c["targets"][0]
    out=root/t["output"];log=root/t["retained_pool_log"]
    if change=="symlink_output":
        moved=out.with_name("moved");out.rename(moved);out.symlink_to(moved,target_is_directory=True)
    elif change=="symlink_intermediate":
        nested=out/"nested";moved=out/"moved";nested.rename(moved);nested.symlink_to(moved,target_is_directory=True)
    elif change=="fifo":os.mkfifo(out/"pipe")
    elif change=="hardlink_log":os.link(log,root/"linked_log")
    else:
        parent=log.parent;moved=parent.with_name("moved_jobs");parent.rename(moved);parent.symlink_to(moved,target_is_directory=True)
    with pytest.raises(ValueError):h.full_inventory(root,t)

@pytest.mark.parametrize("change",["missing","symlink","directory"])
def test_mutex_no_creation_or_wrong_namespace(tmp_path,monkeypatch,change):
    root,c,prior,module,authority,ledger,state=a.fixture(tmp_path,monkeypatch);p=root/c["scheduler_lock"]
    p.unlink()
    if change=="symlink":
        target=root/"synthetic_lock";target.touch();p.symlink_to(target)
    elif change=="directory":p.mkdir()
    with pytest.raises((ValueError,FileNotFoundError)):
        with h.existing_mutex(root,c):raise AssertionError("entered unsafe mutex")
    if change=="missing":assert not p.exists()

@pytest.mark.parametrize("field,value",[("pid",True),("start_ticks",0),("start_ticks",1.0)])
def test_saved_identity_shape_rejected(tmp_path,monkeypatch,field,value):
    root,c,prior,module,authority,ledger,state=a.fixture(tmp_path,monkeypatch);t=c["targets"][0]
    p=root/t["registry"];registry=json.loads(p.read_text());job=registry["jobs"][0]
    job["process_identities"][1][field]=value;prior[t["key"]]["terminal_job"]={k:copy.deepcopy(v) for k,v in job.items() if k not in {"admission","submitted_unix"}};a.dump(p,registry)
    with pytest.raises(ValueError,match="positive exact saved"):h.terminal(root,c,t,prior[t["key"]],module)

def test_source_recheck_failure_keeps_fake_ledger_and_evidence(tmp_path,monkeypatch):
    f=a.fixture(tmp_path,monkeypatch);root,c,prior,module,authority,ledger,state=f
    inspection=a.inspect_fixture(f);admission,review=a.apply_parts(f,inspection);before=ledger.read_bytes()
    def fail():raise ValueError("independent synthetic final source drift")
    with pytest.raises(ValueError,match="final source drift"):
        h.run(root,c,"apply",admission,prior,module,inspection,review,fail)
    assert ledger.read_bytes()==before
    receipt=json.loads((root/c["apply_dir"]/"failure_receipt.json").read_text())
    assert receipt["ledger_mutation_attempted"] is False and receipt["atomic_json_completed"] is False

def test_wrong_review_config_and_old_ledger_pin_fail_before_apply(tmp_path,monkeypatch):
    f=a.fixture(tmp_path,monkeypatch);root,c,prior,module,authority,ledger,state=f
    inspection=a.inspect_fixture(f);admission,review=a.apply_parts(f,inspection);before=ledger.read_bytes()
    for field in ["config","expected_before_ledger_sha256","prior_inventory"]:
        bad=copy.deepcopy(review)
        if isinstance(bad[field],dict):bad[field]["sha256"]="0"*64
        else:bad[field]="0"*64
        with pytest.raises(ValueError,match="fresh exact independent"):
            h.validate_authority(root,c,"apply",admission,inspection,bad)
    assert ledger.read_bytes()==before

def test_bound_metadata_hash_and_duplicate_keys_rejected(tmp_path):
    p=tmp_path/"metadata.json";p.write_text("{\"x\":1}\n");ref=a.ref(tmp_path,p);p.write_text("{\"x\":2}\n")
    with pytest.raises(ValueError,match="bound metadata bytes changed"):h.bound_json(tmp_path,ref)
    p.write_text("{\"x\":1,\"x\":2}\n")
    with pytest.raises(ValueError,match="duplicate JSON"):h.bound_json(tmp_path,a.ref(tmp_path,p))

def test_default_exact_policy_and_contract_no_unknown_writer_claim():
    c=h.policy();assert h.exact(c,h.FIXED_CONFIG)
    assert c["unknown_writer_absence_claimed"] is False
    assert "do not prove arbitrary unknown writers absent" in c["operator_frozen_output_policy"]
    assert c["baseline_bytes"]==167336640512 and c["external_reserve_bytes"]==130000000000

def test_canonical_mutex_excludes_other_descriptor_and_preserves_lock_bytes(tmp_path,monkeypatch):
    import fcntl
    root,c,prior,module,authority,ledger,state=a.fixture(tmp_path,monkeypatch)
    p=root/c["scheduler_lock"];p.write_bytes(b"immutable synthetic mutex bytes");before=p.read_bytes()
    with module.lock_file(root,"scheduler.lock") as other:
        with h.existing_mutex(root,c):
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(other,fcntl.LOCK_UN)
    assert p.read_bytes()==before

def test_finished_sealed_timestamp_must_stay_ordered(tmp_path,monkeypatch):
    root,c,prior,module,authority,ledger,state=a.fixture(tmp_path,monkeypatch);t=c["targets"][0]
    p=root/t["registry"];current=json.loads(p.read_text())["jobs"][0];sealed=copy.deepcopy(prior[t["key"]]["terminal_job"])
    sealed["finished_unix"]=sealed["started_unix"]-1;current["finished_unix"]=sealed["finished_unix"]
    with pytest.raises(ValueError,match="timestamp order"):h.terminal_projection(c,t,current,sealed)

def test_valid_typed_extras_preserved_exactly_and_not_dropped(tmp_path,monkeypatch):
    root,c,prior,module,authority,ledger,state=a.fixture(tmp_path,monkeypatch);t=c["targets"][0]
    current=json.loads((root/t["registry"]).read_text())["jobs"][0];before=copy.deepcopy(current)
    projected,extras=h.terminal_projection(c,t,current,prior[t["key"]]["terminal_job"])
    assert current==before and projected==prior[t["key"]]["terminal_job"]
    assert extras=={k:current[k] for k in ["admission","submitted_unix"]}
    extras["admission"]["observed"]["utilization"]=99
    assert current==before

def test_postreplacement_uncertainty_is_preserved_without_rollback(tmp_path,monkeypatch):
    f=a.fixture(tmp_path,monkeypatch);root,c,prior,module,authority,ledger,state=f
    inspection=a.inspect_fixture(f);admission,review=a.apply_parts(f,inspection)
    original=module.atomic_json
    def write_then_error(path,value):
        original(path,value)
        raise OSError("synthetic postreplacement uncertainty")
    monkeypatch.setattr(module,"atomic_json",write_then_error)
    with pytest.raises(OSError,match="postreplacement uncertainty"):
        h.run(root,c,"apply",admission,prior,module,inspection,review)
    receipt=json.loads((root/c["apply_dir"]/"failure_receipt.json").read_text())
    assert receipt["ledger_mutation_attempted"] is True and receipt["atomic_json_completed"] is False and receipt["no_automatic_rollback"] is True
    after=json.loads(ledger.read_text())
    for t in c["targets"]:assert after["jobs"][t["key"]]["reserved_bytes"]==15000000000
    assert (root/c["apply_dir"]/"preapply_audit.json").is_file() and (root/c["apply_dir"]/"ledger_before.json").is_file()

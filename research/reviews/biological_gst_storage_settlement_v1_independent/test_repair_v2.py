"""Settlement v2 repair checks using fake metadata and inspect-only mode."""
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
P=ROOT/"research/reviews/biological_gst_completed_storage_settlement_v1/storage_revision_v2/audit_and_settle.py"
spec=importlib.util.spec_from_file_location("settlement_v2_review",P)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def fixture(tmp_path,monkeypatch):
    out=tmp_path/"product";out.mkdir()
    (out/"receipt.json").write_text('{"status":"completed_two_GST_conditions_one_seed","models":{"GST32":{"status":"completed"},"GST64":{"status":"completed"}}}')
    (out/"weights.bin").write_bytes(b"handmade weights")
    log=tmp_path/".runtime/gpup/biological_gpu2_v1/jobs/biological-gst-seed101.log";log.parent.mkdir(parents=True);log.write_text("handmade log")
    audit=tmp_path/"audit";audit.mkdir()
    monkeypatch.setattr(s,"ROOT",tmp_path);monkeypatch.setattr(s,"OUTPUT",out);monkeypatch.setattr(s,"AUDIT",audit);monkeypatch.setattr(s,"NEW",100000)
    pool_spec=importlib.util.spec_from_file_location("settlement_v2_fake_pool",ROOT/"src/arggnn/gpu_pool_v1.py")
    pool=importlib.util.module_from_spec(pool_spec);pool_spec.loader.exec_module(pool)
    import arggnn
    monkeypatch.setattr(arggnn,"gpu_pool_v1",pool,raising=False);monkeypatch.setitem(sys.modules,"arggnn.gpu_pool_v1",pool)
    snap=tmp_path/"experiments/results/biological_shared_pool_storage_admission_v1/snapshot.json";snap.parent.mkdir(parents=True)
    snap.write_text(json.dumps({"schema":"biological_candidate_storage_snapshot_v1","status":"complete","returncode":0,"project_root":str(tmp_path),"command":["du","-s","-B1",str(tmp_path)],"used_bytes":167336640512,"completed_utc":"2026-10-10T17:31:06.897981+00:00"}))
    monkeypatch.setattr(s,"SNAPSHOT_SHA",s.sha(snap))
    state={"schema":"gpu_pool_storage_ledger_v1","root":str(tmp_path),"snapshot_sha256":s.SNAPSHOT_SHA,"baseline_bytes":167336640512,"external_reserve_bytes":130000000000,"limit_bytes":450000000000,"controllers":{"live_pool":{"reserved_bytes":1234}},"jobs":{s.KEY:{"output":str(out),"reserved_bytes":s.OLD},"live_other":{"output":str(tmp_path/"other"),"reserved_bytes":1000000}}}
    lp=tmp_path/".runtime/gpup/storage_reservations.json";lp.parent.mkdir(parents=True,exist_ok=True);lp.write_text(json.dumps(state));monkeypatch.setattr(s,"EXPECTED_LEDGER",s.sha(lp))
    measured,files=s.inventory()
    prior=tmp_path/"prior_inspection.json";prior.write_text(json.dumps({"applied":False,"retained_charge_bytes":s.NEW,"retained_conservative_measured_bytes":measured,"inventory_files":files}))
    monkeypatch.setattr(s,"INSPECTED",prior);monkeypatch.setattr(s,"INSPECTED_SHA",s.sha(prior))
    monkeypatch.setattr(s.os,"statvfs",lambda path:SimpleNamespace(f_bavail=10**15,f_frsize=1))
    monkeypatch.setattr(s,"snapshot_job",lambda module:{"logical_job":"biological-gst-seed101","state":"succeeded","exit_code":0})
    monkeypatch.setattr(sys,"argv",["inspect-only"])
    return out,audit,lp,state,pool,snap


def test_walk_errors_now_fail_closed(tmp_path,monkeypatch):
    out,*_=fixture(tmp_path,monkeypatch)
    def inaccessible(path,*,followlinks,onerror):
        onerror(PermissionError("handmade unreadable subtree"))
        return []
    monkeypatch.setattr(s.os,"walk",inaccessible)
    with pytest.raises(PermissionError,match="unreadable subtree"):s.inventory()


def test_two_inventories_are_locked_and_no_fake_or_other_ledger_entry_is_mutated(tmp_path,monkeypatch):
    out,audit,lp,state,pool,snap=fixture(tmp_path,monkeypatch)
    locked=[];original=s.fcntl.flock
    def flock(fd,mode):
        original(fd,mode)
        if mode==s.fcntl.LOCK_EX:locked.append(True)
    monkeypatch.setattr(s.fcntl,"flock",flock)
    inv=s.inventory;calls=[]
    def observed():
        assert locked
        calls.append(True);return inv()
    monkeypatch.setattr(s,"inventory",observed)
    before=lp.read_bytes();s.main()
    r=json.loads((audit/"inspection.json").read_text())
    assert len(calls)==2 and not r["applied"] and lp.read_bytes()==before
    assert r["after_account"]["projected_bytes"]==r["before_account"]["projected_bytes"]-(s.OLD-s.NEW)
    assert r["before_account"]["persistent_reservations"]==r["after_account"]["persistent_reservations"]==2


def test_scope_change_since_frozen_inspection_fails_before_any_reduction(tmp_path,monkeypatch):
    out,audit,lp,*_=fixture(tmp_path,monkeypatch)
    (out/"new_file").write_text("handmade addition")
    before=lp.read_bytes()
    with pytest.raises(ValueError,match="frozen root inspection"):s.main()
    assert lp.read_bytes()==before and not (audit/"inspection.json").exists()


def test_scope_change_between_two_locked_inventories_fails(tmp_path,monkeypatch):
    out,audit,lp,*_=fixture(tmp_path,monkeypatch)
    inv=s.inventory;calls=[]
    def changing():
        result=inv();calls.append(True)
        if len(calls)==1:(out/"weights.bin").write_text("different handmade content")
        return result
    monkeypatch.setattr(s,"inventory",changing)
    before=lp.read_bytes()
    with pytest.raises(ValueError,match="changed before reduction"):s.main()
    assert lp.read_bytes()==before and len(calls)==2


def test_worker_reappearing_after_inventory_blocks_settlement(tmp_path,monkeypatch):
    out,audit,lp,*_=fixture(tmp_path,monkeypatch)
    calls=[]
    def terminal(module):
        calls.append(True)
        if len(calls)==3:raise ValueError("A known GST worker/descendant remains present")
        return {"state":"succeeded"}
    monkeypatch.setattr(s,"snapshot_job",terminal)
    before=lp.read_bytes()
    with pytest.raises(ValueError,match="remains present"):s.main()
    assert len(calls)==3 and lp.read_bytes()==before


@pytest.mark.parametrize("job",["biological-gst-seed101","new_job_same_finished_output"])
def test_persistent_retained_key_output_cannot_be_resubmitted_or_reexpanded(tmp_path,monkeypatch,job):
    out,audit,lp,state,pool,snap=fixture(tmp_path,monkeypatch)
    state["jobs"][s.KEY]["reserved_bytes"]=s.NEW
    lp.write_text(json.dumps(state))
    snapshot=pool.snapshot_record(tmp_path,str(snap),s.SNAPSHOT_SHA)
    ledger=pool.StorageLedger(tmp_path,snapshot,130000000000,450000000000,lambda:10**15)
    before=lp.read_bytes()
    with pytest.raises(ValueError,match="already reserved"):
        ledger.job("biological_gpu2_v1",{"job":job,"output":str(out),"storage_bytes":s.OLD})
    assert lp.read_bytes()==before

"""Concrete settlement blockers; handmade files and inspect-only mode."""
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE=ROOT/"research/reviews/biological_gst_completed_storage_settlement_v1/audit_and_settle.py"
spec=importlib.util.spec_from_file_location("settlement_review",SOURCE)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def scope(tmp_path,monkeypatch):
    output=tmp_path/"product";output.mkdir()
    receipt={"status":"completed_two_GST_conditions_one_seed","models":{"GST32":{"status":"completed"},"GST64":{"status":"completed"}}}
    (output/"receipt.json").write_text(json.dumps(receipt))
    (output/"weights.bin").write_bytes(b"handmade bytes")
    log=tmp_path/".runtime/gpup/biological_gpu2_v1/jobs/biological-gst-seed101.log"
    log.parent.mkdir(parents=True);log.write_text("handmade worker log")
    audit=tmp_path/"audit";audit.mkdir()
    monkeypatch.setattr(s,"ROOT",tmp_path);monkeypatch.setattr(s,"OUTPUT",output);monkeypatch.setattr(s,"AUDIT",audit)
    monkeypatch.setattr(s,"NEW",100000)
    return output,log,audit


def test_gap_probe_unreadable_subtree_is_silently_omitted(tmp_path,monkeypatch):
    output,log,audit=scope(tmp_path,monkeypatch)
    hidden=output/"unreadable";hidden.mkdir()
    (hidden/"retained.bin").write_bytes(b"x"*200000)
    walk=s.os.walk
    def omitted(path,followlinks=False):
        # Mimic os.walk's default behavior on a PermissionError: silently skip
        # an inaccessible child unless caller supplied onerror.
        for base,dirs,files in walk(path,followlinks=followlinks):
            if Path(base)==hidden:
                continue
            yield base,dirs,files
    monkeypatch.setattr(s.os,"walk",omitted)
    charged,records=s.inventory()
    assert charged<s.NEW
    assert "unreadable/retained.bin" not in records
    assert (hidden/"retained.bin").stat().st_size>s.NEW


def test_gap_probe_output_can_grow_after_inventory_before_inspect_commit(tmp_path,monkeypatch):
    output,log,audit=scope(tmp_path,monkeypatch)
    pool_spec=importlib.util.spec_from_file_location("settlement_fake_pool_source",ROOT/"src/arggnn/gpu_pool_v1.py")
    pool=importlib.util.module_from_spec(pool_spec);pool_spec.loader.exec_module(pool)
    import arggnn
    monkeypatch.setattr(arggnn,"gpu_pool_v1",pool,raising=False)
    monkeypatch.setitem(sys.modules,"arggnn.gpu_pool_v1",pool)
    snapshot_path=tmp_path/"experiments/results/biological_shared_pool_storage_admission_v1/snapshot.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot={"schema":"biological_candidate_storage_snapshot_v1","status":"complete","returncode":0,
              "project_root":str(tmp_path),"command":["du","-s","-B1",str(tmp_path)],
              "used_bytes":167336640512,"completed_utc":"2026-10-10T17:31:06.897981+00:00"}
    snapshot_path.write_text(json.dumps(snapshot))
    snapshot_sha=s.sha(snapshot_path);monkeypatch.setattr(s,"SNAPSHOT_SHA",snapshot_sha)
    state={"schema":"gpu_pool_storage_ledger_v1","root":str(tmp_path),"snapshot_sha256":snapshot_sha,
           "baseline_bytes":snapshot["used_bytes"],"external_reserve_bytes":130000000000,
           "limit_bytes":450000000000,"controllers":{},
           "jobs":{s.KEY:{"output":str(output),"reserved_bytes":s.OLD},
                   "live_other":{"output":str(tmp_path/"other"),"reserved_bytes":1000000}}}
    lp=tmp_path/".runtime/gpup/storage_reservations.json";lp.parent.mkdir(parents=True,exist_ok=True);lp.write_text(json.dumps(state))
    monkeypatch.setattr(s,"EXPECTED_LEDGER",s.sha(lp))
    monkeypatch.setattr(s.os,"statvfs",lambda path:SimpleNamespace(f_bavail=10**15,f_frsize=1))
    calls=[]
    def terminal(module):
        calls.append(1)
        if len(calls)==2:
            # Happens after complete inventory, at its purported final terminal check.
            (output/"weights.bin").write_bytes(b"new handmade output"*12000)
        return {"logical_job":"biological-gst-seed101","state":"succeeded","exit_code":0}
    monkeypatch.setattr(s,"snapshot_job",terminal)
    monkeypatch.setattr(sys,"argv",["inspect-only"])
    before=lp.read_bytes()
    s.main()  # No --apply. It may only write fake inspection metadata.
    inspection=json.loads((audit/"inspection.json").read_text())
    assert not inspection["applied"] and lp.read_bytes()==before
    assert inspection["retained_conservative_measured_bytes"]<s.NEW
    assert (output/"weights.bin").stat().st_size>s.NEW
    assert inspection["inventory_files"]["weights.bin"]["sha256"]!=s.sha(output/"weights.bin")


@pytest.mark.parametrize("present", [False,True])
def test_absent_pid_is_accepted_but_same_identity_present_is_rejected(tmp_path,monkeypatch,present):
    output,log,audit=scope(tmp_path,monkeypatch)
    registry_path=tmp_path/".runtime/gpup/biological_gpu2_v1/registry.json"
    registry_path.parent.mkdir(parents=True,exist_ok=True)
    job={"logical_job":"biological-gst-seed101","state":"succeeded","exit_code":0,
         "requested":{"module":"arggnn.biological_gst_control_v1","output":str(output),"storage_bytes":s.OLD},
         "process_identities":[{"pid":123,"start_ticks":456}],"pid":123,"leader_start_ticks":456,"module_source_sha256":"handmade"}
    registry_path.write_text(json.dumps({"safety_error":None,"jobs":[job]}))
    def identity(pid):
        if present:return {"start_ticks":456}
        raise FileNotFoundError("handmade absent PID")
    module=SimpleNamespace(proc_identity=identity)
    if present:
        with pytest.raises(ValueError,match="remains present"):s.snapshot_job(module)
    else:
        assert s.snapshot_job(module)["state"]=="succeeded"

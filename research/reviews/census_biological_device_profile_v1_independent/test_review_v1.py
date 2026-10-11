"""Independent handmade CPU checks for the source-only profile review.

All admission objects are in-memory mocks. No project observations/targets,
CUDA calls, runtime installations, or live accounting reads occur.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from arggnn import census_biological_device_profile_v1 as profile
from arggnn import census_native_qualification_runner_v1 as runner
from arggnn import census_biological_fit_v1 as fit
from arggnn import project_budget_v4 as budget
from arggnn import census_biological_bank_v1 as bank
from arggnn.census_tree_encoder_v1 import CensusTreeGNNV1

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
AUTHOR_FIXTURE = ROOT / "research/reviews/census_biological_fit_v1_author/test_fit_v1.py"

@pytest.fixture(autouse=True)
def handmade_CPU_only():
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    torch.set_num_threads(1)

def ref(role):
    return {"path": "handmade/" + role + ".json", "sha256": "a"*64}

@pytest.fixture
def gate(monkeypatch):
    now = datetime.now(timezone.utc)
    cfg = json.loads(profile.CONFIG.read_text())
    qcfg = json.loads((ROOT / "configs/census_native_qualification_runner_v1.json").read_text())
    refs = {role: ref(role) for role in profile.REFS}
    sources = dict(cfg["source_sha256"])
    sources[str(profile.SOURCE.relative_to(ROOT))] = profile.sha(profile.SOURCE)
    admission = {
        "schema": "census_biological_device_profile_admission_v1", "status": "active",
        "root_authorizer": "root_agent", "job_id": "handmade_profile",
        "created_at_utc": (now-timedelta(minutes=1)).isoformat(),
        "expires_at_utc": (now+timedelta(hours=1)).isoformat(),
        "device": "cuda", "output_directory": "handmade/profile_output",
        "refs": refs, "source_sha256": sources, "selected_cells": []
    }
    runtime_files = {path: "b"*64 for path in qcfg["runtime_files"]}
    numerical_runtime = {"schema":"census_native_CPU_runtime_v1",
        "identity":runner.runtime_identity_v1(), "runtime_file_sha256":runtime_files}
    native = {"refs":{"runner_config":ref("runner_config"),
        "runtime":ref("numerical_runtime")}, "source_sha256":{},
        "job_id":"handmade_numerical"}
    runtime = {"schema":"census_biological_device_profile_runtime_v1",
        "python":str(ROOT/".runtime/env/bin/python"),
        "python_version":sys.version.split()[0], "torch":str(torch.__version__),
        "numpy":str(np.__version__),
        "tskit":str(__import__("tskit").__version__),
        "numerical_identity":runner.runtime_identity_v1(), "runtime_file_sha256":runtime_files}
    memory = {"schema":"census_biological_device_profile_memory_v1",
        "job_id":admission["job_id"], "status":"active",
        "GPU_total_reservation_mib":cfg["GPU_total_reservation_mib"],
        "CPU_total_reservation_bytes":cfg["CPU_total_reservation_bytes"],
        "external_watchdog":True, "created_at_utc":admission["created_at_utc"],
        "expires_at_utc":admission["expires_at_utc"]}
    reservation = {"key":admission["job_id"],
        "output":str(ROOT/admission["output_directory"]),
        "reserved_bytes":cfg["output_reservation_bytes"],
        "consumer_bindings":[{"path":str(profile.SOURCE.relative_to(ROOT)),
            "sha256":sources[str(profile.SOURCE.relative_to(ROOT))]},
            {"path":str(profile.CONFIG.relative_to(ROOT)),
             "sha256":refs["config"]["sha256"]}]}
    review = {"status":"qualified_source_handmade_CPU_only",
        "actual_admission":False,"fixture_mode":False,
        "artifact_sha256":{str(profile.SOURCE.relative_to(ROOT)):
            sources[str(profile.SOURCE.relative_to(ROOT))],
            str(profile.CONFIG.relative_to(ROOT)):refs["config"]["sha256"]}}
    objects = {ref("admission")["path"]:admission,
        refs["config"]["path"]:cfg,refs["review"]["path"]:review,
        refs["runtime"]["path"]:runtime,refs["memory"]["path"]:memory,
        refs["reservation"]["path"]:reservation,
        native["refs"]["runner_config"]["path"]:qcfg,
        native["refs"]["runtime"]["path"]:numerical_runtime}
    for role in ("numerical_evidence","numerical_admission","inspection_evidence",
                 "inspection_freeze","numerical_terminal"):
        objects[refs[role]["path"]] = {"handmade":role}
    reads = []
    class MockReader:
        def decode_json(self, record):
            reads.append(record["path"])
            return deepcopy(objects[record["path"]])
        def read(self, record, **kwargs):
            reads.append(record["path"])
            return b"handmade_mock_bytes"
        def relative(self, value):
            return str(Path(value).relative_to(ROOT)) if Path(value).is_absolute() else value
    monkeypatch.setattr(runner, "VerifiedReaderV1", MockReader)
    monkeypatch.setattr(fit, "require_terminal_numerical_v1", lambda *args:native)
    monkeypatch.setattr(budget, "check_reservation", lambda *args,**kwargs:{"handmade":True})
    return objects, admission, runtime, memory, reservation, review, reads

def test_mock_admission_baseline_all_checks(gate):
    objects, admission, *rest = gate
    accepted = profile.admit_profile(ref("admission"))
    assert accepted[0]["job_id"] == admission["job_id"]
    assert not any("data/" in path for path in gate[-1])

@pytest.mark.parametrize("change", [
    "expired", "long_lifetime", "root", "device", "extra_field", "source",
    "config", "author_review", "fixture_review", "review_hash",
    "model", "tolerance", "recorded_python", "runtime_hashes",
    "reservation_output", "reservation_bytes", "reservation_binding",
    "memory_job", "memory_expired", "memory_GPU", "memory_watchdog"])
def test_mock_admission_rejects_changed_authority(gate, change):
    objects, admission, runtime, memory, reservation, review, reads = gate
    now=datetime.now(timezone.utc)
    cfg=objects[admission["refs"]["config"]["path"]]
    if change=="expired":admission["expires_at_utc"]=(now-timedelta(seconds=1)).isoformat()
    elif change=="long_lifetime":admission["expires_at_utc"]=(now+timedelta(days=2)).isoformat()
    elif change=="root":admission["root_authorizer"]="author"
    elif change=="device":admission["device"]="cpu"
    elif change=="extra_field":admission["unaudited"]=True
    elif change=="source":admission["source_sha256"]={}
    elif change=="config":cfg["kind"]="other"
    elif change=="author_review":review["status"]="authored_pending_review"
    elif change=="fixture_review":review["fixture_mode"]=True
    elif change=="review_hash":review["artifact_sha256"][str(profile.SOURCE.relative_to(ROOT))]="c"*64
    elif change=="model":cfg["model"]["hidden_dim"]=16
    elif change=="tolerance":cfg["tolerances"]["output_rtol"]=.1
    elif change=="recorded_python":runtime["python"]="/unqualified/bin/python"
    elif change=="runtime_hashes":runtime["runtime_file_sha256"]={}
    elif change=="reservation_output":reservation["output"]=str(ROOT/"handmade/other")
    elif change=="reservation_bytes":reservation["reserved_bytes"]=1
    elif change=="reservation_binding":reservation["consumer_bindings"]=[]
    elif change=="memory_job":memory["job_id"]="other"
    elif change=="memory_expired":memory["expires_at_utc"]=(now-timedelta(seconds=1)).isoformat()
    elif change=="memory_GPU":memory["GPU_total_reservation_mib"]=1
    elif change=="memory_watchdog":memory["external_watchdog"]=False
    with pytest.raises(ValueError):
        profile.admit_profile(ref("admission"))

def test_actual_interpreter_must_match_qualified_receipt(gate, monkeypatch):
    monkeypatch.setattr(sys,"executable","/unqualified/bin/python")
    with pytest.raises(ValueError,match="interpreter|Python|runtime"):
        profile.admit_profile(ref("admission"))

def test_live_torch_version_must_match_qualified_receipt(gate, monkeypatch):
    monkeypatch.setattr(torch,"__version__","unqualified_torch")
    with pytest.raises(ValueError,match="runtime|Runtime|identity"):
        profile.admit_profile(ref("admission"))

def test_handmade_single_tree_all_controls_and_inactive_eta():
    spec=importlib.util.spec_from_file_location("independent_profile_handmade_source",AUTHOR_FIXTURE)
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    ts, rows, roster = fixtures.native_fixture()
    tables=ts.dump_tables()
    edges=tables.edges
    keep=np.asarray(edges.left)==0.
    edges.set_columns(left=np.asarray(edges.left)[keep],
        right=np.full(int(keep.sum()),ts.sequence_length),
        parent=np.asarray(edges.parent)[keep],child=np.asarray(edges.child)[keep])
    tables.sort();single=tables.tree_sequence()
    assert single.num_trees==1
    cert=fixtures.certificate(single,rows,roster,"handmade_train","train")
    cfg=json.loads((ROOT/"configs/census_native_qualification_runner_v1.json").read_text())
    torch.manual_seed(975);model=CensusTreeGNNV1(**cfg["model"]).float()
    initial=bank.fingerprint_sha256_v1(model.state_dict())
    out=profile.compare_certified_tile_v1(cert,cert.certificate_sha256,model,
        torch.arange(184),cfg,"cpu",lambda:None)
    assert [x["control"] for x in out] == list(profile.CONTROLS)
    assert all(x["output_max_absolute_difference"]==0 for x in out)
    assert all(x["VJP_max_absolute_difference"]==0 for x in out)
    assert all(x["scored_query_count"]==120 and x["output_shape"][-2:]==[120,4] for x in out)
    assert bank.fingerprint_sha256_v1(model.state_dict())==initial


@pytest.mark.parametrize("change", ["status","created_before","expires_after","CPU",
    "future","expired","watchdog","job"])
def test_memory_helper_rejects_lifetime_or_job_drift(gate, change):
    objects,admission,runtime,memory,reservation,review,reads=gate
    now=datetime.now(timezone.utc)
    if change=="status":memory["status"]="released"
    elif change=="created_before":memory["created_at_utc"]=(now-timedelta(days=1)).isoformat()
    elif change=="expires_after":memory["expires_at_utc"]=(now+timedelta(days=1)).isoformat()
    elif change=="CPU":memory["CPU_total_reservation_bytes"]=1
    elif change=="future":memory["created_at_utc"]=(now+timedelta(minutes=1)).isoformat()
    elif change=="expired":memory["expires_at_utc"]=(now-timedelta(seconds=1)).isoformat()
    elif change=="watchdog":memory["external_watchdog"]=False
    elif change=="job":memory["job_id"]="unreserved_job"
    cfg=objects[admission["refs"]["config"]["path"]]
    with pytest.raises(ValueError,match="memory reservation"):
        profile.require_memory_v1(memory,admission,cfg)

@pytest.mark.parametrize("mode",["exclusive","shared","logical_mismatch",
    "pool_mismatch","missing"])
def test_actual_lease_logical_job_binding(monkeypatch,mode):
    admission={"job_id":"handmade_profile"}
    pool="handmade_profile" if mode=="exclusive" else "handmade_pool"
    lease={"launcher_metadata":{"job":pool}}
    monkeypatch.delenv("ARGGNN_LOGICAL_JOB_NAME",raising=False)
    monkeypatch.setenv("ARGGNN_JOB_NAME",pool)
    if mode=="shared":monkeypatch.setenv("ARGGNN_LOGICAL_JOB_NAME","handmade_profile")
    elif mode=="logical_mismatch":monkeypatch.setenv("ARGGNN_LOGICAL_JOB_NAME","other")
    elif mode=="pool_mismatch":
        monkeypatch.setenv("ARGGNN_LOGICAL_JOB_NAME","handmade_profile")
        monkeypatch.setenv("ARGGNN_JOB_NAME","wrong_pool")
    elif mode=="missing":monkeypatch.delenv("ARGGNN_JOB_NAME")
    if mode in {"exclusive","shared"}:
        profile.require_logical_job_v1(lease,admission)
    else:
        with pytest.raises(ValueError,match="logical job"):
            profile.require_logical_job_v1(lease,admission)

@pytest.fixture
def run_gate(gate,monkeypatch,tmp_path):
    import arggnn.biological_candidate as candidate
    objects,admission,runtime,memory,reservation,review,reads=gate
    admitted=profile.admit_profile(ref("admission"))
    cfg,native,qcfg,reader,reservation=admitted[1:]
    parent={"parent_id":"handmade_train","split":"train"}
    tile={"tile_id":0,"original_tree_ids":[0,1]}
    prepared={"tiles":[tile]}
    admission["selected_cells"]=[{"parent_id":"handmade_train","tile_id":0,"original_tree_count":2}]
    native.update(initial_state_sha256_by_dtype={"float32":"d"*64},
        expected_certificates={"handmade_train:0":"e"*64},group_permutation=list(range(184)))
    monkeypatch.setattr(profile,"admit_profile",lambda record:
        (admission,cfg,native,qcfg,reader,reservation))
    manifest_calls=[]
    def manifests(*args):
        manifest_calls.append(1)
        return [(parent,prepared)]
    monkeypatch.setattr(runner,"read_manifests_v1",manifests)
    monkeypatch.setattr(runner,"load_parent_v1",lambda *args,**kwargs:"handmade_loaded")
    monkeypatch.setattr(runner,"prepare_tile_v1",lambda *args:("handmade_cert",None,None))
    class Initial:
        def state_dict(self):return {}
    monkeypatch.setattr(runner,"_model",lambda *args:Initial())
    monkeypatch.setattr(bank,"fingerprint_sha256_v1",lambda *args:"d"*64)
    captures=[]
    class Output:
        def __init__(self,*args,**kwargs):self.path=tmp_path
        def write(self,name,result):captures.append(deepcopy(result))
    monkeypatch.setattr(runner,"BoundedEvidenceOutputV1",Output)
    lease={"launcher_metadata":{"job":"handmade_profile"}}
    monkeypatch.setenv("ARGGNN_JOB_NAME","handmade_profile")
    monkeypatch.delenv("ARGGNN_LOGICAL_JOB_NAME",raising=False)
    device_configuration=("cpu",{"handmade_CPU":True},lease)
    monkeypatch.setattr(candidate,"configure_fit",lambda *args:device_configuration)
    monkeypatch.setattr(candidate,"check_fit_lease",lambda *args:None)
    checks=[{"control":control,
        "selected_device":{"peak_CUDA_reserved_bytes":1},
        "handmade_control_flow_mock":True} for control in profile.CONTROLS]
    monkeypatch.setattr(profile,"compare_certified_tile_v1",lambda *args:deepcopy(checks))
    return {"memory":memory,"cfg":cfg,"captures":captures,
        "manifest_calls":manifest_calls,"candidate":candidate,
        "device_configuration":device_configuration,"checks":checks}

def test_run_resource_boundary_stops_expired_memory_and_preserves_failure(run_gate,monkeypatch):
    def expire(*args):
        resource_check=args[-1]
        resource_check()
        run_gate["memory"]["expires_at_utc"]=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        resource_check()
        raise AssertionError("Expired memory unexpectedly continued")
    monkeypatch.setattr(profile,"compare_certified_tile_v1",expire)
    with pytest.raises(ValueError,match="memory reservation"):
        profile.run_profile(ref("admission"))
    result=run_gate["captures"][-1]
    assert result["status"]=="failed_partial_preserved"
    assert result["error_type"]=="ValueError"
    assert result["query_truth_accessed"] is False
    assert result["scientific_fit_performed"] is False

def test_expiry_during_qualification_stops_before_observation_reads(run_gate,monkeypatch):
    def expire(*args):
        run_gate["memory"]["expires_at_utc"]=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        return run_gate["device_configuration"]
    monkeypatch.setattr(run_gate["candidate"],"configure_fit",expire)
    with pytest.raises(ValueError,match="memory reservation"):
        profile.run_profile(ref("admission"))
    assert run_gate["manifest_calls"]==[]
    assert run_gate["captures"][-1]["status"]=="failed_partial_preserved"

def test_peak_rejection_retains_completed_measurements(run_gate,monkeypatch):
    too_large=run_gate["cfg"]["GPU_total_reservation_mib"]*1024*1024+1
    measured=deepcopy(run_gate["checks"])
    measured[0]["selected_device"]["peak_CUDA_reserved_bytes"]=too_large
    monkeypatch.setattr(profile,"compare_certified_tile_v1",lambda *args:deepcopy(measured))
    with pytest.raises(ValueError,match="allocator peak"):
        profile.run_profile(ref("admission"))
    result=run_gate["captures"][-1]
    assert result["status"]=="failed_partial_preserved"
    assert result["selected_cells"][0]["checks"]==measured
    assert result["selected_cells"][0]["checks"][0]["selected_device"]["peak_CUDA_reserved_bytes"]==too_large

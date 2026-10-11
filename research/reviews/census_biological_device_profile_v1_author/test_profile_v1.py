import importlib.util
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import os
from copy import deepcopy
import json

import pytest
import torch
from arggnn import census_biological_device_profile_v1 as profile
from arggnn import census_biological_bank_v1 as bankmod
from arggnn.census_tree_encoder_v1 import CensusTreeGNNV1

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
spec = importlib.util.spec_from_file_location("profile_handmade_fixture_source",
    ROOT / "research/reviews/census_biological_fit_v1_author/test_fit_v1.py")
fixture_source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_source)

@pytest.fixture(scope="module")
def handmade():
    torch.set_num_threads(1)
    ts, rows, roster = fixture_source.native_fixture()
    cert = fixture_source.certificate(ts, rows, roster, "handmade_train", "train")
    cfg = json.loads((ROOT / "configs/census_native_qualification_runner_v1.json").read_text())
    torch.manual_seed(41)
    model = CensusTreeGNNV1(**cfg["model"]).float()
    return cert, model, cfg

def test_default_source_report_never_imports_torch_or_opens_actual_data():
    code = "import sys,json;from arggnn import census_biological_device_profile_v1 as p;v=p.source_report();assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules;assert not v['CUDA_accessed'];print(json.dumps(v))"
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"}
    r=subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)["models_created"] is False
    with pytest.raises(ValueError,match="Fresh root admission"):
        profile.main(["--execute"])

def test_full_default_two_tree_five_control_disposable_CPU_execution(handmade):
    cert,model,cfg=handmade
    initial=bankmod.fingerprint_sha256_v1(model.state_dict())
    calls=[]
    rows=profile.compare_certified_tile_v1(cert,cert.certificate_sha256,model,torch.arange(184),cfg,"cpu",lambda:calls.append(1))
    assert [r["control"] for r in rows]==list(profile.CONTROLS)
    assert all(r["CPU_float32_selected_device_float32_output_VJP_parity"] for r in rows)
    assert all(r["output_max_absolute_difference"]==0 and r["VJP_max_absolute_difference"]==0 for r in rows)
    assert all(r["output_shape"][-2:]==[120,4] and r["finite_disposable_AdamW"] for r in rows)
    assert all(r["selected_device"]["peak_CUDA_allocated_bytes"] is None for r in rows)
    assert all(r["optimizer_step_parity_claimed"] is False for r in rows)
    assert len(calls)==15
    assert bankmod.fingerprint_sha256_v1(model.state_dict())==initial
    bankmod.validate_census_biological_certificate_v1(cert,cert.certificate_sha256)

def test_detects_changed_device_output_with_fixed_tolerance(handmade,monkeypatch):
    cert,model,cfg=handmade
    original=bankmod.forward_certified_census_biological_v1
    calls=[]
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);calls.append(1)
        return replace(result,logits=result.logits+.1) if len(calls)%2==0 else result
    monkeypatch.setattr(bankmod,"forward_certified_census_biological_v1",changed)
    with pytest.raises(AssertionError):
        profile.compare_certified_tile_v1(cert,cert.certificate_sha256,model,torch.arange(184),cfg,"cpu",lambda:None)

def test_detects_changed_VJP_without_changing_output(handmade,monkeypatch):
    cert,model,cfg=handmade
    original=bankmod.forward_certified_census_biological_v1
    calls=[]
    def changed(model,*args,**kwargs):
        result=original(model,*args,**kwargs);calls.append(1)
        if len(calls)%2==0:
            scalar=next(model.parameters()).flatten()[0]
            return replace(result,logits=result.logits+(scalar-scalar.detach())*10.)
        return result
    monkeypatch.setattr(bankmod,"forward_certified_census_biological_v1",changed)
    with pytest.raises(AssertionError):
        profile.compare_certified_tile_v1(cert,cert.certificate_sha256,model,torch.arange(184),cfg,"cpu",lambda:None)

def test_live_runtime_receipt_cannot_substitute_for_wrong_interpreter(monkeypatch):
    import sys,numpy as np,tskit
    identity={"python":sys.version.split()[0],"torch":str(torch.__version__),"numpy":str(np.__version__),"tskit":str(tskit.__version__)}
    assert profile.require_live_runtime_identity_v1({"identity":identity})==identity
    with monkeypatch.context() as patch:
        patch.setattr(sys,"executable","/not-the-project-python")
        with pytest.raises(ValueError,match="Actual main"):
            profile.require_live_runtime_identity_v1({"identity":identity})
    for key in identity:
        changed=dict(identity);changed[key]="wrong"
        with pytest.raises(ValueError,match="Live numerically"):
            profile.require_live_runtime_identity_v1({"identity":changed})

def test_memory_expiry_and_job_binding():
    from datetime import datetime,timezone,timedelta
    now=datetime.now(timezone.utc)
    admission={"job_id":"handmade-profile","created_at_utc":(now-timedelta(seconds=20)).isoformat(),"expires_at_utc":(now+timedelta(seconds=60)).isoformat()}
    cfg={"GPU_total_reservation_mib":16384,"CPU_total_reservation_bytes":64000000000}
    memory={"schema":"census_biological_device_profile_memory_v1","job_id":admission["job_id"],"status":"active",**cfg,"external_watchdog":True,"created_at_utc":(now-timedelta(seconds=10)).isoformat(),"expires_at_utc":(now+timedelta(seconds=30)).isoformat()}
    profile.require_memory_v1(memory,admission,cfg)
    for key,value in [("job_id","other"),("status","expired"),("GPU_total_reservation_mib",1),("CPU_total_reservation_bytes",1),("external_watchdog",False),("created_at_utc",(now-timedelta(seconds=30)).isoformat()),("expires_at_utc",(now-timedelta(seconds=1)).isoformat()),("expires_at_utc",(now+timedelta(seconds=90)).isoformat())]:
        changed=dict(memory);changed[key]=value
        with pytest.raises(ValueError):
            profile.require_memory_v1(changed,admission,cfg)

def test_exclusive_and_shared_logical_job_binding(monkeypatch):
    admission={"job_id":"handmade-profile"}
    monkeypatch.setenv("ARGGNN_JOB_NAME","exclusive-job")
    monkeypatch.delenv("ARGGNN_LOGICAL_JOB_NAME",raising=False)
    with pytest.raises(ValueError):
        profile.require_logical_job_v1({"launcher_metadata":{"job":"exclusive-job"}},admission)
    monkeypatch.setenv("ARGGNN_JOB_NAME","handmade-profile")
    profile.require_logical_job_v1({"launcher_metadata":{"job":"handmade-profile"}},admission)
    monkeypatch.setenv("ARGGNN_JOB_NAME","shared-pool")
    monkeypatch.setenv("ARGGNN_LOGICAL_JOB_NAME","handmade-profile")
    profile.require_logical_job_v1({"launcher_metadata":{"job":"shared-pool"}},admission)
    monkeypatch.setenv("ARGGNN_LOGICAL_JOB_NAME","other-logical")
    with pytest.raises(ValueError):
        profile.require_logical_job_v1({"launcher_metadata":{"job":"shared-pool"}},admission)

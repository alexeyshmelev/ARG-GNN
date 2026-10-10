"""Inject the actual Torch/nvidia UUID encodings without creating CUDA contexts."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from arggnn import confirmatory_prediction_runner as prediction

path=Path(__file__).resolve().parents[1]/"scripts/check_development_cuda_admission.py"
spec=importlib.util.spec_from_file_location("uuid_admission_probe",path)
admission=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=admission
spec.loader.exec_module(admission)
ACTUAL="df11de7f-7a15-ebe2-085d-9c7d907f3786"
OTHER="976a5cd3-df12-2535-fa51-e3ea063688ba"

@pytest.mark.parametrize("module",[admission,prediction],ids=["admission","prediction"])
@pytest.mark.parametrize("runtime_uuid",[ACTUAL,"GPU-"+ACTUAL,ACTUAL.upper()])
def test_real_uuid_encodings_match_reserved_device_without_cuda(module,runtime_uuid,monkeypatch):
    initialize(module,runtime_uuid,monkeypatch,success=True)

@pytest.mark.parametrize("module",[admission,prediction],ids=["admission","prediction"])
@pytest.mark.parametrize("runtime_uuid",[OTHER,"GPU-"+OTHER,"not-a-device-uuid","GPU-"+ACTUAL+"extra","{"+ACTUAL+"}"])
def test_wrong_or_malformed_uuid_rejects_before_later_startup(module,runtime_uuid,monkeypatch):
    initialize(module,runtime_uuid,monkeypatch,success=False)

def initialize(module,runtime_uuid,monkeypatch,*,success):
    import setproctitle
    assert not torch.cuda.is_initialized()
    events=[]
    monkeypatch.setattr(module,"set_research_title",lambda:events.append("title"))
    monkeypatch.setattr(setproctitle,"getproctitle",lambda:module.PROCESS_TITLE)
    monkeypatch.setattr(module,"configure_deterministic_development",lambda:events.append("policy") or {"strict":True})
    monkeypatch.setattr(module,"validate_launcher_lease",lambda:events.append("lease") or {"visible_gpu_uuid":"GPU-"+ACTUAL,"physical_gpu":3})
    monkeypatch.setattr(torch.cuda,"device_count",lambda:events.append("count") or 1)
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda *_:events.append("properties") or SimpleNamespace(name="mock A100",total_memory=1024,major=8,minor=0,uuid=runtime_uuid))
    original_run=module.subprocess.run
    def run(cmd,*a,**kw):
        if cmd[0]=="nvidia-smi":
            events.append("driver")
            return SimpleNamespace(stdout="mock driver")
        return original_run(cmd,*a,**kw)
    monkeypatch.setattr(module.subprocess,"run",run)
    receipt={}
    if success:
        module.initialize_execution("cuda",receipt)
        assert receipt["device"]["runtime_uuid"]==runtime_uuid
        assert receipt["device"]["canonical_runtime_uuid"]==ACTUAL
        assert receipt["canonical_visible_gpu_uuid"]==ACTUAL
        assert receipt["initialization_stage"]=="complete"
        assert events[:5]==["title","policy","lease","count","properties"]
    else:
        with pytest.raises(RuntimeError,match="UUID"):
            module.initialize_execution("cuda",receipt)
        assert receipt["initialization_stage"]=="device_properties"
        assert events==["title","policy","lease","count","properties"]
    assert not torch.cuda.is_initialized()

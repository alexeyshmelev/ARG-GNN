"""Temporal development device admission ordering without CUDA or fitting."""
import pytest

from arggnn import train_anchor_temporal_development as runner
from arggnn.development_training_bindings import sha
from test_development_training_bindings import configs,write_config


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_missing_completion_rejected_before_device_or_cuda_queries(tmp_path,monkeypatch,device):
    settings=configs(tmp_path)["gru"]
    path=write_config(tmp_path,settings)
    (tmp_path/"completion.json").unlink()
    def forbidden(*args,**kwargs):
        pytest.fail("Device/CUDA queried before completed-source admission")
    monkeypatch.setattr(runner,"require_device",forbidden)
    for name in ("device_count","is_available","init","_lazy_init"):
        monkeypatch.setattr(runner.torch.cuda,name,forbidden)
    with pytest.raises(RuntimeError,match="missing|Missing|absent|Absent"):
        runner.main(["--config",str(path),"--expected-config-sha256",sha(path),
            "--output",str(tmp_path/"run"),"--device",device])
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_device_check_follows_successful_completed_source_admission(tmp_path,monkeypatch,device):
    settings=configs(tmp_path)["gru"]
    path=write_config(tmp_path,settings)
    events=[]
    def admitted(*args,**kwargs):
        events.append("completed_source_admission")
        return {},object(),{},object()
    class DeviceReached(RuntimeError):pass
    def device_check(requested):
        assert requested==device
        events.append("device_check")
        raise DeviceReached
    monkeypatch.setattr(runner,"admit_run",admitted)
    monkeypatch.setattr(runner,"require_device",device_check)
    with pytest.raises(DeviceReached):
        runner.main(["--config",str(path),"--expected-config-sha256",sha(path),
            "--output",str(tmp_path/"run"),"--device",device])
    assert events==["completed_source_admission","device_check"]
    assert not (tmp_path/"run").exists()

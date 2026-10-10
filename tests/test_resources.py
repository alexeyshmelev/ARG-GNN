from pathlib import Path
import pytest
from arggnn.resources import ROOT, project_path, eligible_free

def test_project_path_rejects_escape():
    with pytest.raises(ValueError):
        project_path("../elsewhere/checkpoint.pt")
    with pytest.raises(ValueError):
        project_path("/tmp/checkpoint.pt")
    assert project_path("checkpoints/run.pt") == ROOT / "checkpoints/run.pt"

@pytest.mark.parametrize("idx", [0, 1, 6])
def test_reserved_gpus_are_never_eligible(idx):
    assert not eligible_free(dict(index=idx, has_compute_process=False,
                                 memory_used_mib=0, utilization_percent=0))

def test_compute_process_prevents_gpu_assignment():
    assert not eligible_free(dict(index=2, has_compute_process=True,
                                 memory_used_mib=0, utilization_percent=0))
    assert eligible_free(dict(index=2, has_compute_process=False,
                              memory_used_mib=0, utilization_percent=0))

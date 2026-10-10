"""CPU-only numerical policy and pre-context ordering checks."""
import copy
import os
import pytest
import torch
from arggnn.development_execution import (
    DETERMINISTIC_POLICY, configure_deterministic_development,
    deterministic_development_manifest,
)


@pytest.fixture(autouse=True)
def restore_execution_flags(monkeypatch):
    old = (torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
           torch.get_float32_matmul_precision(),
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    yield
    torch.use_deterministic_algorithms(old[0], warn_only=old[1])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:4]
    torch.set_float32_matmul_precision(old[4])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[5:7]


def test_real_cpu_configuration_and_manifest_without_cuda_context():
    assert not torch.cuda.is_initialized()
    expected = copy.deepcopy(DETERMINISTIC_POLICY)
    actual = configure_deterministic_development()
    assert actual == expected and not torch.cuda.is_initialized()
    actual["scope"] = "changed caller copy"
    assert deterministic_development_manifest() == expected


def test_existing_cuda_context_rejects_before_environment_mutation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="precede CUDA initialization"):
        configure_deterministic_development()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_later_policy_drift_rejects():
    configure_deterministic_development()
    torch.use_deterministic_algorithms(False)
    with pytest.raises(RuntimeError, match="policy drift"):
        deterministic_development_manifest()

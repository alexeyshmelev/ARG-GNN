"""Strict numerical execution policy for fresh development only."""
from __future__ import annotations
from copy import deepcopy
import os
import torch

DETERMINISTIC_POLICY = {
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "warn_only": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "configure_before_cuda_initialization": True,
    "scope": "fixed runtime and hardware, no cross-platform bitwise guarantee",
}


def configure_deterministic_development():
    """Apply after completed-source admission and before any CUDA context."""
    if torch.cuda.is_initialized():
        raise RuntimeError("Fresh development policy must precede CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return deterministic_development_manifest()


def deterministic_development_manifest():
    result = deepcopy(DETERMINISTIC_POLICY)
    actual = {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if any(actual[key] != result[key] for key in actual):
        raise RuntimeError("Fresh development deterministic execution policy drift")
    return result

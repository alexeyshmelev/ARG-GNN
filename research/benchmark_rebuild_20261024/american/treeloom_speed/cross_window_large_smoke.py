"""Small FP32 check of full cross-window backward carry, with no CUDA or training."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

root = Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path.insert(0, str(root))
torch.set_num_threads(1)
module_name = "research.benchmark_rebuild_20261024.american.treeloom_speed"
new = importlib.import_module(module_name + ".stable_leaf_history_v2")
spec = importlib.util.spec_from_file_location("serial_window_reference",
    root / "research/reviews/TreeLoom_GPU_history_source_20261017_v1/stable_leaf_history_v1.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
torch.manual_seed(841)
cases = []
for reverse in (False, True):
    n, h, rank = 2049, 16, 8
    forcing = torch.randn(n * 9, rank, requires_grad=True)
    groups = torch.arange(n)[:, None] * 9 + torch.arange(h)[None, :] % 9
    theta = torch.tensor(-2., requires_grad=True)
    spans = 10. + 1500. * torch.rand(n)
    ell = 100. + 99900. * theta.sigmoid()
    rho = torch.exp(-spans / ell)
    source = -torch.expm1(-spans / ell)
    offsets = tuple(i * 9 for i in range(n + 1))
    profile = old.NullProfileV1()
    args = (forcing, rho, source, groups, offsets, reverse, 1024, torch.device("cpu"), lambda label: None, profile)
    a = old._Direction.apply(*args)
    b = new._Direction.apply(*args)
    gradient = torch.randn_like(a)
    ga = torch.autograd.grad((a * gradient).sum(), (forcing, theta), retain_graph=True)
    gb = torch.autograd.grad((b * gradient).sum(), (forcing, theta))
    assert torch.allclose(a, b, atol=4e-6, rtol=4e-5)
    assert torch.allclose(ga[0], gb[0], atol=6e-5, rtol=4e-4)
    assert torch.allclose(ga[1], gb[1], atol=2e-3, rtol=5e-4)
    cases.append(dict(reverse=reverse, windows=3, output_max_error=float((a-b).abs().max()),
                      forcing_VJP_max_error=float((ga[0]-gb[0]).abs().max()),
                      theta_VJP_absolute_error=float((ga[1]-gb[1]).abs())))
result = dict(schema="TreeLoom_parallel_history_cross_window_CPU_v1", passed=True,
              FP32_only=True, no_CUDA=True, no_training=True,
              global_boundary_VJP_preserved=True, cases=cases)
with Path(__file__).with_name("cross_window_large_smoke.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))

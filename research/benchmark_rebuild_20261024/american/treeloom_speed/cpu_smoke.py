"""Small CPU output and VJP checks for the parallel history arithmetic schedule."""
import importlib.util
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root = Path("/workspace-SR003.nfs2/ARG-GNN")
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

serial = module("serial_history", root / "research/reviews/TreeLoom_GPU_history_source_20261017_v1/stable_leaf_history_v1.py")
parallel = module("parallel_history", Path(__file__).with_name("parallel_history_window.py"))
torch.manual_seed(20261024)
cases = []
for reverse in (False, True):
    for with_boundary in (False, True):
        n, h, rank = 128, 16, 8
        forcing = torch.randn(n * 9, rank, requires_grad=True)
        # Repeated group memberships exercise the gather/scatter adjoint.
        groups = torch.arange(n)[:, None] * 9 + torch.arange(h)[None, :] % 9
        rho = (.50 + .499 * torch.rand(n)).requires_grad_()
        source = (.001 + .2 * torch.rand(n)).requires_grad_()
        incoming = torch.randn(h, rank, requires_grad=True) if with_boundary else None
        arguments = (forcing, rho, source) + ((incoming,) if with_boundary else ())
        a, a_last = serial._window(forcing, rho, source, groups, incoming, reverse)
        b, b_last = parallel.parallel_window(forcing, rho, source, groups, incoming, reverse)
        gradient = torch.randn_like(a)
        terminal_gradient = torch.randn_like(a_last)
        ga = torch.autograd.grad((a * gradient).sum() + (a_last * terminal_gradient).sum(),
                                 arguments, retain_graph=True)
        gb = torch.autograd.grad((b * gradient).sum() + (b_last * terminal_gradient).sum(), arguments)
        assert torch.allclose(a, b, atol=3e-6, rtol=2e-5)
        for x, y in zip(ga, gb):
            assert torch.allclose(x, y, atol=4e-5, rtol=3e-4)
        cases.append(dict(reverse=reverse, incoming_boundary=with_boundary,
                          output_max_error=float((a-b).abs().max()),
                          VJP_max_errors=[float((x-y).abs().max()) for x,y in zip(ga,gb)]))
out = dict(schema="TreeLoom_parallel_history_CPU_smoke_v1", passed=True,
           CPU_only=True, no_training_or_CUDA=True, cases=cases,
           time_steps=128, affine_prefix_levels=7,
           numerical_change="FP32 arithmetic reassociation only",
           full_GPU_speed_not_measured=True)
destination = Path(__file__).with_name("cpu_smoke.json")
with destination.open("x") as stream:
    json.dump(out, stream, indent=2)
    stream.write("\n")
print(json.dumps(out))

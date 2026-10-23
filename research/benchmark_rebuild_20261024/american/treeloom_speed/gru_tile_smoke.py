"""Small FP32 check that larger GRU execution tiles preserve independent haplotypes."""
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.manual_seed(335)
module = torch.nn.GRU(35, 32)
sequence = torch.randn(17, 129, 35, requires_grad=True)
probe = torch.randn(17, 129, 32)
def tiled(width):
    return torch.cat([module(x)[0] for x in sequence.split(width, dim=1)], dim=1)
old, new = tiled(8), tiled(64)
parameters = tuple(module.parameters()) + (sequence,)
ga = torch.autograd.grad((old * probe).mean(), parameters)
gb = torch.autograd.grad((new * probe).mean(), parameters)
assert torch.allclose(old, new, atol=2e-6, rtol=2e-5)
for a, b in zip(ga, gb):
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-4)
result = dict(schema="TreeLoom_GRU_tile_FP32_CPU_smoke_v1", passed=True,
              CPU_only=True, old_haplotype_tile=8, new_haplotype_tile=64,
              partial_last_tile_checked=True, sequence_length=17, haplotypes=129,
              output_max_error=float(abs(old-new).max()),
              parameter_and_input_VJP_max_errors=[float(abs(a-b).max()) for a,b in zip(ga,gb)],
              no_training_or_CUDA=True)
with Path(__file__).with_name("gru_tile_smoke.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))

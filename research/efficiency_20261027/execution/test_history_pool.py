"""CPU pool and complete recurrence/VJP parity, including window boundaries."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from research.efficiency_20261027.execution.history_pool import windowed_cut_mean,stable_leaf_history
from research.benchmark_rebuild_20261024.american.treeloom_speed.stable_leaf_history_v2 import stable_leaf_history_v2
from research.benchmark_rebuild_20261024.american.treeloom_speed.static_model_v2 import AmericanTreeLoomC3FullV1

OUT=Path("research/efficiency_20261027/execution")


def plan(holds,haplotypes,seed):
    generator=torch.Generator().manual_seed(seed)
    rows=[];counts=[];offsets=[0]
    for hold in range(holds):
        # Every group nonempty, arbitrary recurrent leaf membership, varying
        # counts, and groups that meet exactly at a pooling window boundary.
        number=1+(hold%min(haplotypes,17))
        local=torch.arange(haplotypes)%number
        local=local[torch.randperm(haplotypes,generator=generator)]
        rows.append(local+offsets[-1])
        counts.append(torch.bincount(local,minlength=number).float())
        offsets.append(offsets[-1]+number)
    return SimpleNamespace(leaf_groups=torch.stack(rows),group_counts=torch.cat(counts),
        group_offsets=tuple(offsets),native_holds=holds,haplotypes=haplotypes)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(27)
    model=AmericanTreeLoomC3FullV1()
    rows=[]
    for holds,haplotypes,window in ((1,1,1),(7,19,1),(7,19,3),(7,19,7),(1025,17,1024)):
        cut=plan(holds,haplotypes,34)
        raw=[torch.randn(holds,haplotypes,8) for _ in range(2)]
        outputs=[];grads=[]
        weight=torch.randn(len(cut.group_counts),8)
        for candidate in (False,True):
            left,right=[x.clone().requires_grad_(True) for x in raw]
            if candidate:
                value=windowed_cut_mean(model.memory,left,right,cut,device="cpu",holds_per_window=window)
            else:
                value=model.memory._mean((.5*(left+right)).flatten(0,1),cut.leaf_groups.flatten(),cut.group_counts,len(cut.group_counts))
            (value*weight).sum().backward()
            outputs.append(value.detach());grads.append((left.grad,right.grad))
        torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
        for a,b in zip(*grads):torch.testing.assert_close(a,b,atol=0,rtol=0)
        rows.append(dict(kind="pool",holds=holds,haplotypes=haplotypes,window=window,
                         output_error=0.,gradient_error=0.))
    for holds,window in ((1,1),(9,1),(9,4),(9,9)):
        cut=plan(holds,1600,45)
        raw=torch.randn(len(cut.group_counts),8)
        spans=torch.linspace(1.,19000.,holds)
        weight=torch.randn(len(cut.group_counts),8)
        outputs=[];grads=[]
        for function in (stable_leaf_history_v2,stable_leaf_history):
            forcing=raw.clone().requires_grad_(True)
            theta=model.memory.ell_theta.detach().clone().requires_grad_(True)
            result=function(model.memory,forcing,theta,spans,cut,device="cpu",holds_per_window=window)
            (result*weight).sum().backward()
            outputs.append(result.detach());grads.append((forcing.grad,theta.grad))
        torch.testing.assert_close(outputs[0],outputs[1],atol=1e-6,rtol=1e-5)
        for a,b in zip(*grads):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        rows.append(dict(kind="connected_bidirectional_recurrence",holds=holds,haplotypes=1600,window=window,
            output_error=float((outputs[0]-outputs[1]).abs().max()),
            forcing_gradient_error=float((grads[0][0]-grads[1][0]).abs().max()),
            theta_gradient_error=float((grads[0][1]-grads[1][1]).abs().max())))
    assert not torch.cuda.is_initialized()
    receipt=dict(schema="windowed_history_pool_CPU_v1",status="pass",cases=rows,
        CPU_threads=1,CUDA_initialized=False,scope="CPU pooling and connected original recurrence parity only; GPU scatter rounding,throughput,full-parent peaks unmeasured.",
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (OUT/"history_pool.py",OUT/"test_history_pool.py")])
    (OUT/"history_cpu_results.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()

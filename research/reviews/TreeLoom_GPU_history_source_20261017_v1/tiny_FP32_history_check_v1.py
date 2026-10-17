"""One tiny FP32 forward/forcing/ell VJP comparison; no optimizer or file reads."""
from types import SimpleNamespace
import torch
from research.reviews.order2_cut_memory_implementation_20261015_v1.order2_cut_memory_v1 import CutMemoryV1
from .stable_leaf_history_v1 import stable_leaf_history_v1

def run_tiny_v1(*,device="cpu",resource_check=lambda label:None,profile=None):
    resource_check("tiny_FP32_history_check_begin")
    generator=torch.Generator(device="cpu").manual_seed(70019)
    t,h,r=9,1600,8;counts=[1,3,5,7,2,9,4,6,8]
    offsets=[0];groups=[]
    for n in counts:
        row=torch.arange(h,dtype=torch.long)%n
        row=row[torch.randperm(h,generator=generator)]
        groups.append(row+offsets[-1]);offsets.append(offsets[-1]+n)
    groups=torch.stack(groups)
    sizes=torch.bincount(groups.flatten(),minlength=offsets[-1]).float()
    plan=SimpleNamespace(native_holds=t,haplotypes=h,leaf_groups=groups,group_counts=sizes,
        group_offsets=tuple(offsets),episode_of_hold=(0,)*t,
        reverse_cells=None,forward_cells=None,reverse_current_group=None,forward_current_group=None,
        reverse_old_group=None,forward_old_group=None,reverse_counts=None,forward_counts=None)
    # Includes a coefficient underflow, unequal physical spans and every
    # first/last window boundary in both directions.
    spans=torch.tensor([1.,17.,430.,5000.,5000000.,2.,100.,9000.,37.],dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(70019);memory=CutMemoryV1(arm="stable_leaf_memory",checkpoint_memory=False)
    f=torch.randn((offsets[-1],r),generator=generator,dtype=torch.float32,requires_grad=True)
    theta=memory.ell_theta
    y=memory._history(f,spans,plan)
    weights=torch.randn(y.shape,generator=generator,dtype=torch.float32)
    objective=(y*weights).sum()+.03125*y.square().sum()
    expected=torch.autograd.grad(objective,(f,theta))
    candidate_f=f.detach().clone().requires_grad_(True)
    candidate_theta=theta.detach().clone().requires_grad_(True)
    value=stable_leaf_history_v1(memory,candidate_f,candidate_theta,spans,plan,
        device=device,holds_per_window=3,resource_check=resource_check,profile=profile)
    got=torch.autograd.grad((value*weights).sum()+.03125*value.square().sum(),
                            (candidate_f,candidate_theta))
    labels=("forward","forcing_gradient","ell_theta_gradient")
    checks={}
    for label,actual,reference in zip(labels,(value,*got),(y,*expected)):
        passed=bool(torch.isfinite(actual).all())and torch.allclose(actual,reference,rtol=3e-5,atol=2e-6)
        checks[label]=dict(passed=bool(passed),max_absolute_error=float((actual-reference).abs().max()),
                           reference_max_abs=float(reference.abs().max()),rtol=3e-5,atol=2e-6)
        if not passed:raise ValueError("Tiny FP32 original history comparison failed: "+label)
    if not all(bool((g!=0).any())for g in got):raise ValueError("Both learned history inputs have nonzero VJPs")
    resource_check("tiny_FP32_history_check_end")
    return dict(schema="TreeLoom_tiny_FP32_history_forward_backward_v1",passed=True,device=str(device),
        holds=t,haplotypes=h,rank=r,groups=offsets[-1],window_holds=3,checks=checks,
        cross_window_reverse_VJP=True,coefficient_underflow_included=True,
        optimizer_updates=0,scientific_data_reads=0,bitwise_GPU_parity_claimed=False)
if __name__=="__main__":
    import json
    print(json.dumps(run_tiny_v1(device="cpu"),sort_keys=True))

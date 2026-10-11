"""Two handmade donor components test detached-gauge VJPs with empty family."""
import json
from pathlib import Path
import torch
from arggnn.reference_dependence_gnn_v1 import merge_moment_v1, transport_moment_v1, normalize_moment_v1

torch.set_num_threads(1)
def local(cls, weight):
    source=torch.full((1,2,4),-torch.inf,dtype=torch.float32)
    source[0,0,cls]=torch.tensor(weight).log()
    q=torch.where(torch.isfinite(source).any(-1),source.max(-1).values,torch.zeros(1,2))
    m=source-q[...,None]
    return q,m,m,m[..., :,None]+m[...,None,:]
checks=[]
for offset in (0.,80.):
    for independent in (False,True):
        cost=torch.tensor([offset+1.2,offset+1.7],dtype=torch.float32,requires_grad=True)
        a=transport_moment_v1(local(0,.4),torch.stack((cost[0],cost[0]))[None])
        b=transport_moment_v1(local(1,.6),torch.stack((cost[1],cost[1]))[None])
        p,c,mask=normalize_moment_v1(merge_moment_v1(a,b,independent=independent))
        pi=torch.softmax(torch.tensor([.4,.6]).log()-cost,0)
        k=torch.eye(2) if independent else torch.stack((torch.stack((cost.new_ones(()),(-cost.sum()).exp())),torch.stack(((-cost.sum()).exp(),cost.new_ones(())))))
        expected=pi[:,None]*k*pi[None,:]
        assert torch.allclose(p[0,0,:2],pi,atol=2e-6,rtol=0)
        assert torch.allclose(c[0,0,:2,:2],expected,atol=2e-6,rtol=0)
        assert mask.tolist()==[[True,False]] and (c[0,1]==0).all()
        weights=torch.tensor([[.3,.4],[.4,1.1]])
        observed_grad=torch.autograd.grad((c[0,0,:2,:2]*weights).sum(),cost,retain_graph=True)[0]
        expected_grad=torch.autograd.grad((expected*weights).sum(),cost)[0]
        assert torch.isfinite(observed_grad).all()
        assert torch.allclose(observed_grad,expected_grad,atol=4e-6,rtol=2e-5)
        checks.append(dict(offset=offset,independent=independent,max_C_error=float((c[0,0,:2,:2]-expected).abs().max().detach()),max_VJP_error=float((observed_grad-expected_grad).abs().max())))
result=dict(scope="two_donor_handmade_FP32_only",actual_data_read=False,GPU_created=False,optimizer_steps=0,checks=checks)
Path("research/reviews/reference_dependence_gnn_v1_independent/independent_gauge_handcase.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

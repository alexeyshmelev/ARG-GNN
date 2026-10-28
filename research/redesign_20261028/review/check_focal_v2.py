"""Label-free FP32 numerical executor review. No ancestry fitting or source edits."""
from pathlib import Path
import hashlib
import json
import torch
from research.redesign_20261028.architecture import window_model_frozen_v1 as old
from research.redesign_20261028.architecture import window_model_focal_executor_v2 as new
from research.redesign_20261028.architecture.test_focal_executor_v2 import (
    select_focals,toy,edgeless_focal_with_edged_sources,four_real_packets)

ROOT=Path('research/redesign_20261028')
torch.set_num_threads(2)
THRESHOLD=1e-5

def synthetic_loss(y):
    coefficient=torch.linspace(-.7,1.1,y.numel()).reshape_as(y)
    return 10*(y*coefficient).sum()+y.square().sum()

def models(seed=711):
    torch.manual_seed(seed)
    a=old.WindowTreeLoom(width=32);b=new.WindowTreeLoom(width=32)
    b.load_state_dict(a.state_dict())
    oa=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.01,eps=1e-8)
    ob=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.01,eps=1e-8)
    return a,b,oa,ob

def step(arrays,mode,state):
    a,b,oa,ob=state
    source=old.prepare_observations(arrays);plan=new.prepare_executor(source)
    oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
    y=a(source,mode=mode);z=b(plan,mode=mode)
    loss=synthetic_loss(y);other_loss=synthetic_loss(z)
    loss.backward();other_loss.backward()
    ga=[];gb=[];none=[]
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None),name
        if p.grad is None:none.append(name)
        else:
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
            ga.append(p.grad.flatten());gb.append(q.grad.flatten())
    ga=torch.cat(ga);gb=torch.cat(gb)
    grad_relative=float((ga-gb).norm()/ga.norm().clamp_min(1))
    forward_scaled=float((y-z).abs().max()/(1+y.abs().max()))
    torch.nn.utils.clip_grad_norm_(a.parameters(),.5,error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(b.parameters(),.5,error_if_nonfinite=True)
    oa.step();ob.step()
    va=torch.cat([p.detach().flatten() for p in a.parameters()])
    vb=torch.cat([p.detach().flatten() for p in b.parameters()])
    with torch.no_grad():
        yy=a(source,mode=mode);zz=b(plan,mode=mode)
        post_scaled=float((yy-zz).abs().max()/(1+yy.abs().max()))
        loss_delta=float((synthetic_loss(yy)-synthetic_loss(zz)).abs())
        loss_scaled=loss_delta/(1+abs(float(synthetic_loss(yy))))
    for p,q in zip(a.parameters(),b.parameters()):
        assert oa.state[p].keys()==ob.state[q].keys()
        if 'step' in oa.state[p]:assert float(oa.state[p]['step'])==float(ob.state[q]['step'])
    return dict(forward_scaled_max_error=forward_scaled,full_VJP_relative_L2_error=grad_relative,
        updated_logits_scaled_max_error=post_scaled,updated_probe_loss_scaled_error=loss_scaled,
        updated_probe_loss_absolute_error=loss_delta,
        updated_parameter_max_abs_error=float((va-vb).abs().max()),
        updated_parameter_relative_L2_error=float((va-vb).norm()/va.norm()),
        gradient_connectivity_identical=True,optimizer_step_counters_identical=True,
        none_gradient_parameters=none,
        practical_function_check=all(v<=THRESHOLD for v in [forward_scaled,grad_relative,post_scaled,loss_scaled]))

four,_=four_real_packets()
cases={}
for name,arrays,mode in [('synthetic_interval_subset_leaf_only',select_focals(toy(),[1,1,2]),'leaf_only'),
    ('four_real_structure',four,'structure'),('four_real_leaf_only',four,'leaf_only'),('four_real_no_cross',four,'no_cross')]:
    cases[name]=step(arrays,mode,models())
state=models(911)
sequence=[]
for name,arrays in [('edged',select_focals(toy(),[1])),('edgeless_with_edged_sources',edgeless_focal_with_edged_sources()),
                    ('edged_again',select_focals(toy(),[1]))]:
    row=step(arrays,'no_cross',state);row['case']=name;sequence.append(row)
assert all(row['practical_function_check'] for row in list(cases.values())+sequence)
receipt=dict(schema='independent_focal_executor_v2_numerics_v1',
    candidate_sha256=hashlib.sha256((ROOT/'architecture/window_model_focal_executor_v2.py').read_bytes()).hexdigest(),
    reference_sha256=hashlib.sha256((ROOT/'architecture/window_model_frozen_v1.py').read_bytes()).hexdigest(),
    precision='FP32 learned operations, inherited FP64 coordinate kernels',cases=cases,connectivity_sequence=sequence,
    numerical_budget=THRESHOLD,
    budget_rationale='A 1e-5 scale-normalized functional/VJP budget is approximately84 FP32 machine epsilons; used for accumulated dense/scatter arithmetic, independently of AdamW coordinate thresholds. This is an engineering tolerance, not a mathematical error bound.',
    verdict='Qualified functional FP32 equivalence for reviewed CPU cases; strict optimizer-parameter equivalence is not established.',
    limitations=['Original strict parameter failures are retained; tiny-gradient AdamW amplification is real.',
        'Three-step connectivity check exercises existing momentum before and after an edgeless focal batch.',
        'No bitwise optimizer equivalence or long-training-trajectory equivalence is claimed.',
        'No GPU or ancestry-fit qualification is supplied.'],labels_used=False,ancestry_fitting=False)
(ROOT/'review/FOCAL_V2_NUMERICS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

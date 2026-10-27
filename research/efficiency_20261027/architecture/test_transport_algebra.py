"""Small synthetic correctness checks only. No performance or ancestry claims."""
import json
from pathlib import Path
import torch
from transport_algebra import parent_jumps, tree_transport, explicit_tree_kernel, normalized_tree_transport, affine_scan, distance_scan

torch.set_num_threads(1)
torch.manual_seed(9027)
rows=[]


def compare(name, actual, expected, tolerance=2e-5):
    error=float((actual-expected).abs().max())
    scale=float(expected.abs().max())
    passed=error<=tolerance*(1+scale)
    rows.append(dict(check=name,max_abs_error=error,reference_max_abs=scale,tolerance=tolerance,passed=passed))
    assert passed, rows[-1]


for name, raw in [
    ('chain',[-1]+list(range(12))),
    ('binary',[-1,0,0,1,1,2,2,3,3,4,4,5,5,6,6]),
    ('polytomy',[-1]+[0]*11),
    ('forest',[-1,0,0,-1,3,4,4,-1]),
]:
    parent=torch.tensor(raw)
    n=len(raw)
    gate_logits=torch.randn(n,5,requires_grad=True)
    values=torch.randn(n,5,requires_grad=True)
    gates=torch.sigmoid(gate_logits)
    actual=tree_transport(parent,gates,values)
    kernel=explicit_tree_kernel(parent,gates)
    expected=torch.einsum('uvd,vd->ud',kernel,values)
    compare(name+'_all_pairs',actual,expected)
    probe=torch.randn_like(actual)
    actual_grads=torch.autograd.grad((actual*probe).sum(),(values,gate_logits),retain_graph=True)
    reference_grads=torch.autograd.grad((expected*probe).sum(),(values,gate_logits),retain_graph=True)
    for target,a,b in zip(['value','gate'],actual_grads,reference_grads):
        compare(name+'_'+target+'_gradient',a,b)
    compare(name+'_normalization',normalized_tree_transport(parent,gates,values),expected/kernel.sum(1))
    permutation=torch.randperm(n)
    inverse=torch.argsort(permutation)
    renamed=torch.where(parent[permutation]>=0,inverse[parent[permutation].clamp_min(0)],-torch.ones(n,dtype=torch.long))
    compare(name+'_ID_equivariance',tree_transport(renamed,gates[permutation],values[permutation]),actual[permutation])

# A distant leaf reaches its cousin in one operator; no shallow hop cutoff.
parent=torch.tensor([-1,0,0,1,1,2,2])
gates=torch.full((7,1),0.8)
values=torch.zeros(7,1);values[3]=1
transport=tree_transport(parent,gates,values)
compare('distant_leaf_path',transport[6],torch.tensor([0.8**4]))

# Width-aware exact affine composition versus sequential recurrence and gradients.
values=torch.randn(37,4,6,requires_grad=True)
widths=torch.rand(37)*1500+1
log_rate=torch.randn(6,requires_grad=True)
rate=torch.nn.functional.softplus(log_rate)/5000
initial=torch.randn(4,6,requires_grad=True)
actual=distance_scan(values,widths,rate,initial)
state=initial;sequential=[]
for value,width in zip(values,widths):
    a=torch.exp(-width*rate)
    state=a*state-torch.expm1(-width*rate)*value
    sequential.append(state)
expected=torch.stack(sequential)
compare('genomic_scan_values',actual,expected)
probe=torch.randn_like(actual)
a=torch.autograd.grad((actual*probe).sum(),(values,log_rate,initial),retain_graph=True)
b=torch.autograd.grad((expected*probe).sum(),(values,log_rate,initial),retain_graph=True)
for target,x,y in zip(['forcing','rate','initial'],a,b):compare('genomic_scan_'+target+'_gradient',x,y)
first=distance_scan(values[:13],widths[:13],rate,initial)
second=distance_scan(values[13:],widths[13:],rate,first[-1])
compare('scan_no_chunk_reset',torch.cat((first,second)),actual)
constant=torch.randn(1,4,6)
whole=distance_scan(constant,torch.tensor([1000.]),rate,initial)[-1]
split=distance_scan(constant.expand(3,-1,-1),torch.tensor([200.,300.,500.]),rate,initial)[-1]
compare('constant_forcing_subdivision',split,whole)
try:
    parent_jumps(torch.tensor([1,0]))
except ValueError:
    rows.append(dict(check='cycle_rejected',passed=True))
else:raise AssertionError('Cycle accepted')
receipt=dict(schema='TreeLoom_transport_algebra_CPU_v1',umbrella_hypothesis='treeloom_efficiency_redesign_20261027',checks=rows,all_passed=all(x['passed'] for x in rows),torch_version=torch.__version__,device='cpu',threads=1,synthetic_only=True,observed_data_opened=False,ancestry_truth_opened=False,model_training=False,performance_measured=False,CUDA_initialized=False,limitations=['Reference pointer jumping is not a fused GPU implementation.','No ancestry accuracy, utilization, throughput, or full-parent memory claim.','Candidate removes nonlinear mixing at every tree depth, so it is not an exact TreeLoom replacement.'])
Path(__file__).with_name('algebra_test_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

"""Only hand-derived CPU log-field, empty-mask and gradient checks."""
from pathlib import Path
from dataclasses import replace
import json
import torch
from arggnn.persistence_trace_kernel_v1 import MetricTreeHoldV1
from arggnn.reference_logcavity_gnn_v1 import (
    ReferenceLogCavityGNNV1, prepare_reference_logcavity_bank_v1,
    unit_class_fields_v1, masked_segment_logsumexp_v1, masked_logaddexp_v1, ARMS)

torch.set_num_threads(1);torch.manual_seed(101)
def tree(parent,times=None):
    if times is None:times=[0.,0.,200.,220.,450.,470.,500.,700.,1000.]
    return MetricTreeHoldV1(torch.tensor(parent),torch.tensor(times,dtype=torch.float64),
        torch.arange(len(parent)),"toy-native",torch.arange(6),torch.arange(6))
trees=[tree([6,7,6,7,7,8,8,8,-1]),tree([7,7,6,7,7,8,8,8,-1]),
       tree([6,7,6,7,7,8,8,8,-1])]
interval=torch.tensor([[0.,1000.],[1000.,2000.],[2000.,3000.]],dtype=torch.float64)
def prepare(ts,ints,x):
    return prepare_reference_logcavity_bank_v1(ts,ints,x,torch.tensor([0,1]),
        torch.tensor([2,3,4,5]),torch.tensor([0,1,1,2]),sample_identity_namespace="toy-samples")
bank=prepare(trees,interval,torch.tensor([100.,1100.,2100.],dtype=torch.float64))
model=ReferenceLogCavityGNNV1()
baseline=torch.randn(3,2,4)
checks={}
for arm in ARMS:
    assert torch.equal(model(bank,baseline,control=arm),baseline)
checks["zero_head_exact_frozen_skip_all_arms"]=True
# Hand-computed reference mixture for query0: young-reference MRCAs500 and1000.
up,out=model.log_cavities(bank)
full=masked_logaddexp_v1(up,out)
fields,support=unit_class_fields_v1(full)
row=int(bank.forest.query_nodes[0,0])
coef=torch.nn.functional.softplus(model.raw_cost)[0]
phi=lambda age:coef[0]*((age-350)/350)+coef[1]*torch.log1p(torch.tensor((age-350)/350))
expected=torch.softmax(torch.stack((-2*phi(500.),-2*phi(1000.))),0)
assert torch.allclose(fields[row,:2],expected,atol=2e-6,rtol=0)
assert (fields[:,4]==0).all() # No older ANA source, pseudocount or floor.
assert torch.allclose(fields.reshape(-1,2,4).sum(-1)[support],torch.ones_like(fields.reshape(-1,2,4).sum(-1)[support]),atol=2e-6,rtol=0)
checks["hand_derived_young_reference_kernel_fields"]=fields[row,:4].detach().tolist()
checks["absent_older_ANA_exact_zero_and_nonempty_families_unit_sum"]=True
# Empty supports must have finite zero gradients, including segmented empty sums.
empty=torch.full((3,8),-torch.inf,requires_grad=True)
zero,mask=unit_class_fields_v1(masked_segment_logsumexp_v1(empty,torch.tensor([0,1,1]),3))
zero.sum().backward()
assert not mask.any() and torch.isfinite(empty.grad).all() and (empty.grad==0).all()
checks["all_empty_support_safe_zero_VJP"]=True
# Child exclusion is exact in both the log cavity and learned outside signal.
source_node=int(bank.forest.forward_frames[0].nodes[2])
change=bank.log_source.clone();change[source_node,0]+=.7
changed=replace(bank,log_source=change)
up2,out2=model.log_cavities(changed)
assert torch.allclose(out[source_node],out2[source_node],atol=1e-6,rtol=0)
_,learned_out=model.encode_units(bank,up,out)
_,learned_out2=model.encode_units(changed,up2,out2)
assert torch.allclose(learned_out[source_node],learned_out2[source_node],atol=1e-6,rtol=0)
checks["subtree_excluded_log_and_learned_outside"]=True
# Source-gradient masking is safe even though absent classes are -inf leaves.
differentiable=replace(bank,log_source=bank.log_source.clone().requires_grad_())
a,b=model.log_cavities(differentiable)
outside_fields,_=unit_class_fields_v1(b)
outside_fields[source_node,6].backward()
assert torch.isfinite(differentiable.log_source.grad).all()
assert differentiable.log_source.grad[source_node,0]==0
checks["subtree_excluded_source_gradient_and_empty_class_VJP"]=True
with torch.no_grad():
    model.context_head.weight.copy_(torch.randn_like(model.context_head.weight)*.03)
    model.kernel_head.weight.copy_(torch.randn_like(model.kernel_head.weight)*.03)
for arm in ARMS:
    model.zero_grad(set_to_none=True)
    value=model(bank,baseline,control=arm)
    value.square().sum().backward()
    grad=model.raw_cost.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0,(arm,grad)
    names=("kernel_head.weight",) if arm=="logcavity_only" else (
        "reference_projection.weight","up_value.weight","down_value.weight","decoder_base.weight")
    norms={name:float(dict(model.named_parameters())[name].grad.norm()) for name in names}
    assert all(x>0 for x in norms.values()),(arm,norms)
    if arm not in ("static","logcavity_only"):
        assert model.value.weight.grad.norm()>0 and model.key.weight.grad.norm()>0
        if arm=="carrier_gru":assert model.carrier_gru.weight_hh.grad.norm()>0
    checks[arm+"_finite_nonzero_gradients"]=dict(cost_norm=float(grad.norm()),**norms)
# No reference evidence is exactly zero residual, with nonzero heads.
no_refs=replace(bank,log_source=torch.full_like(bank.log_source,-torch.inf))
for arm in ARMS:
    assert torch.equal(model(no_refs,baseline,control=arm),baseline)
checks["empty_reference_bank_exact_zero_residual_all_arms"]=True
# Kernel-only serial subdivision: insert an unlabeled750 node on500->1000.
original=prepare(trees[:1],interval[:1],torch.tensor([100.],dtype=torch.float64))
subdivided=prepare([tree([6,7,6,7,7,8,9,8,-1,8],
    [0.,0.,200.,220.,450.,470.,500.,700.,1000.,750.])],interval[:1],
    torch.tensor([100.],dtype=torch.float64))
def read_fields(b):
    a,z=model.log_cavities(b)
    f,_=unit_class_fields_v1(masked_logaddexp_v1(a,z))
    return f[b.forest.query_nodes]
assert torch.allclose(read_fields(original),read_fields(subdivided),atol=2e-6,rtol=0)
checks["additive_potential_kernel_serial_subdivision_invariance"]=True
gnn_original=model(original,baseline[:1],control="static")
gnn_subdivided=model(subdivided,baseline[:1],control="static")
checks["nonlinear_GNN_subdivision_difference_expected"]=float((gnn_original-gnn_subdivided).detach().norm())
# Large costs would underflow raw affinity masses, but unit fields remain finite.
with torch.no_grad():model.raw_cost.fill_(1000.)
a,z=model.log_cavities(bank)
extreme,mask=unit_class_fields_v1(masked_logaddexp_v1(a,z))
assert torch.isfinite(extreme).all()
assert torch.allclose(extreme.reshape(-1,2,4).sum(-1)[mask],torch.ones_like(extreme.reshape(-1,2,4).sum(-1)[mask]),atol=2e-6,rtol=0)
model.zero_grad(set_to_none=True)
extreme.square().sum().backward()
assert torch.isfinite(model.raw_cost.grad).all()
checks["raw_mass_underflow_regime_unit_fields_and_VJP_finite"]=True
checks["all_below350_edge_cost_basis_exact_zero"]=bool((bank.edge_basis[
    bank.forest.geometry[:,0] < torch.log1p(torch.tensor(350.))/torch.log1p(torch.tensor(10000.))] == 0).all())
assert checks["all_below350_edge_cost_basis_exact_zero"]
checks["parameter_counts"]={arm:model.parameter_counts(arm) for arm in ARMS}
checks["geometry"]=bank.metadata
p=Path("research/reviews/reference_logcavity_gnn_v1_author/toy_checks.json")
p.write_text(json.dumps(checks,indent=2)+"\n")
print(json.dumps({k:v for k,v in checks.items() if k!="geometry"},indent=2))

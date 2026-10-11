"""Hand-derived CPU signal-flow/gradient checks, not a data/model fit."""
from pathlib import Path
import json
import torch
from arggnn.persistence_trace_kernel_v1 import MetricTreeHoldV1
from arggnn.edge_innovation_gnn_v1 import EdgeInnovationGNNV1, prepare_edge_innovation_bank_v1, ARMS

torch.set_num_threads(1)
torch.manual_seed(101)
def tree(parent):
    return MetricTreeHoldV1(torch.tensor(parent), torch.tensor([0.,0.,200.,220.,450.,470.,500.,700.,1000.],dtype=torch.float64),
        torch.arange(9), "toy-native", torch.arange(6), torch.arange(6))
trees=[tree([6,7,6,7,7,8,8,8,-1]),tree([7,7,6,7,7,8,8,8,-1]),tree([6,7,6,7,7,8,8,8,-1])]
bank=prepare_edge_innovation_bank_v1(trees,
    torch.tensor([[0.,1000.],[1000.,2000.],[2000.,3000.]],dtype=torch.float64),
    torch.tensor([100.,1100.,2100.],dtype=torch.float64), torch.tensor([0,1]),
    torch.tensor([2,3,4,5]),torch.tensor([0,1,0,1]),sample_identity_namespace="toy-samples")
model=EdgeInnovationGNNV1()
baseline=torch.randn(3,2,4)
checks={}
for arm in ARMS:
    value=model(bank,baseline,control=arm)
    checks[arm+"_zero_head_skip"]=bool(torch.equal(value,baseline))
    assert checks[arm+"_zero_head_skip"]
with torch.no_grad():
    model.context_head.weight.copy_(torch.randn_like(model.context_head.weight)*.03)
for arm in ARMS:
    model.zero_grad(set_to_none=True)
    result=model(bank,baseline,control=arm)
    (result.square().sum()).backward()
    names=("reference_projection.weight","up_value.weight","down_value.weight","decoder_base.weight")
    norms={name:float(dict(model.named_parameters())[name].grad.norm()) for name in names}
    assert all(x>0 for x in norms.values()), (arm,norms)
    if arm != "static":
        for name in ("value.weight","key.weight","retention.weight","raw_ell"):
            grad=dict(model.named_parameters())[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0,(arm,name,grad)
        if arm=="carrier_gru":
            assert model.carrier_gru.weight_hh.grad.norm()>0
    checks[arm+"_finite_gradient_norms"]=norms
# Zero reference contrasts must not generate any residual, even with a nonzero head.
from dataclasses import replace
empty=replace(bank,source=torch.zeros_like(bank.source))
for arm in ARMS:
    assert torch.equal(model(empty,baseline,control=arm),baseline)
checks["zero_reference_contrasts_zero_all_arm_residual"]=True
# Query0 has no source on its native sample subtree but changes outside affinity.
u,o=model.encode(bank)
first,second=bank.forward_frames[:2]
# Carrier ordering retains original native child order; query0 is first.
a,b=int(first.nodes[0]),int(second.nodes[0])
assert torch.equal(u[a],torch.zeros_like(u[a]))
assert (o[b]-o[a]).norm()>0
checks["unlabeled_query_carrier_outside_delta_nonzero"]=float((o[b]-o[a]).norm())
assert bank.metadata["older_proxy_placements_at_actual_sampling_nodes"]==6
assert bank.metadata["native_vertices"]==27
checks["older_proxy_actual_nodes_and_all_native_vertices_retained"]=True
# Every forward/reverse frame at a tile endpoint starts fresh.
assert (bank.forward_frames[0].previous_index==-1).all()
assert (bank.reverse_frames[0].previous_index==-1).all()
checks["bidirectional_tile_endpoint_reset"]=True
checks["parameter_counts"]={arm:model.parameter_counts(arm) for arm in ARMS}
checks["geometry"]=bank.metadata

# Cavity exclusion: changing a reference value under an edge cannot change its
# outside state. Geometry/support gates remain fixed for this signal-flow check.
cavity_node=int(bank.forward_frames[0].nodes[2])
changed_source=bank.source.clone()
changed_source[cavity_node]*=-1
u_changed,o_changed=model.encode(replace(bank,source=changed_source))
assert torch.allclose(o_changed[cavity_node],o[cavity_node],atol=1e-7,rtol=0)
assert not torch.allclose(u_changed[cavity_node],u[cavity_node],atol=1e-7,rtol=0)
checks["subtree_excluding_cavity_signal_flow"]=True
# A memory read on one ancestral edge reaches a query on another current branch.
zero_read=u.new_zeros(len(u),4)
one_read=zero_read.clone()
one_read[int(bank.forward_frames[0].nodes[6]),0]=1
plain=model.decode(bank,u,o,zero_read,zero_read)
injected=model.decode(bank,u,o,one_read,zero_read)
other_query=int(bank.query_nodes[0,1])
assert (injected[other_query]-plain[other_query]).norm()>0
checks["ancestral_edge_read_reaches_other_query_before_pooling"]=float(
    (injected[other_query]-plain[other_query]).norm())
# Deliberately equal age/duration/support bins make a real internal-history sham.
sham_tree=MetricTreeHoldV1(torch.tensor([6,7,8,8,8,8,8,8,-1]),
    torch.tensor([0.,0.,200.,220.,450.,470.,500.,520.,1000.],dtype=torch.float64),
    torch.arange(9),"toy-sham",torch.arange(6),torch.arange(6))
sham_bank=prepare_edge_innovation_bank_v1([sham_tree]*3,
    torch.tensor([[0.,1000.],[1000.,2000.],[2000.,3000.]],dtype=torch.float64),
    torch.tensor([100.,1100.,2100.],dtype=torch.float64),torch.tensor([0,1]),
    torch.tensor([2,3,4,5]),torch.tensor([0,1,0,1]),sample_identity_namespace="toy-sham-samples")
assert sham_bank.metadata["sham_changed_lookups"]>0
for frame in sham_bank.forward_frames[1:]+sham_bank.reverse_frames[1:]:
    assert torch.equal(frame.previous_index[:6],frame.sham_previous_index[:6])
    assert not torch.equal(frame.previous_index[6:],frame.sham_previous_index[6:])
checks["internal_sham_changed_sample_histories_fixed"]=sham_bank.metadata["sham_changed_lookup_fraction"]
# Native ancestor6 disappears completely and later returns. Its history resets.
global_parent={0:8,1:7,2:8,3:7,4:7,5:8,7:8,8:-1}
native=[0,1,2,3,4,5,7,8]
local={node:rank for rank,node in enumerate(native)}
absent=MetricTreeHoldV1(torch.tensor([-1 if global_parent[node]<0 else local[global_parent[node]] for node in native]),
    trees[0].node_times[native],torch.tensor(native),"toy-native",torch.arange(6),torch.arange(6))
absence_bank=prepare_edge_innovation_bank_v1([trees[0],absent,trees[0]],
    torch.tensor([[0.,1000.],[1000.,2000.],[2000.,3000.]],dtype=torch.float64),
    torch.tensor([100.,1100.,2100.],dtype=torch.float64),torch.tensor([0,1]),
    torch.tensor([2,3,4,5]),torch.tensor([0,1,0,1]),sample_identity_namespace="toy-samples")
assert int(absence_bank.forward_frames[2].previous_index[6])==-1
assert int(absence_bank.reverse_frames[2].previous_index[6])==-1
checks["absence_reappearance_no_history_resurrection"]=True

print(json.dumps(checks,indent=2))
Path("research/reviews/edge_innovation_gnn_v1_author/toy_checks.json").write_text(json.dumps(checks,indent=2)+"\n")

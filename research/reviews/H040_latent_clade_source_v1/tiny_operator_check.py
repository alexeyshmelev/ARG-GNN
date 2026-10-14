"""One handmade H040 operator fixture. CPU only; no data/cache/training."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import torch
from arggnn.anonymous_clade_witness_v1 import (
 ForestLCAV1,choose_landmarks_v1,candidate_triples_v1,_block_states,AnonymousCladeEventsV1)
from arggnn.anonymous_clade_gnn_v1 import AnonymousCladeTensorGNNV1
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(101)
# Two five-leaf trees: old(0,1)|2 -> new(0,2)|1. Internal5 and12 are anonymous.
parent=np.asarray([5,5,6,6,6,6,-1,12,13,12,13,13,13,-1],dtype=np.int64)
down=(np.asarray([2,3,4,5,8,10,11,12]),np.asarray([0,1,7,9]))
samples=np.asarray([[0,1,2,3,4],[7,8,9,10,11]],dtype=np.int64)
keys=tuple(("toy_%d"%i,0) for i in range(5))
lm=choose_landmarks_v1(keys,"toy_fixture");triples=candidate_triples_v1(keys,lm)
state,receiver=_block_states(ForestLCAV1(parent,down),samples,lm,triples)
changed=(state[0]<3)&(state[1]<3)&(state[0]!=state[1]);ids=np.flatnonzero(changed)
assert ids.size==1
j=int(ids[0]);assert tuple(triples[j])==(0,1,2) and tuple(state[:,j])==(0,1) and tuple(receiver[:,j])==(5,12)
events=AnonymousCladeEventsV1(torch.tensor([0],dtype=torch.int32),torch.tensor([5],dtype=torch.int32),
 torch.tensor([12],dtype=torch.int32),torch.tensor([0],dtype=torch.int32),torch.tensor([1.]),2,5,14)
empty=replace(events,**{n:getattr(events,n)[:0] for n in ("boundary_index","old_node","new_node","shared_sample","weight")})
bank=SimpleNamespace(parent=torch.from_numpy(parent),sample_index=torch.from_numpy(samples),
 upward_levels=(torch.tensor([0,1,2,3,4,7,8,9,10,11]),torch.tensor([5,12]),torch.tensor([6,13])),
 upward_children=(torch.empty(0,dtype=torch.long),torch.tensor([0,1,7,9]),torch.tensor([2,3,4,5,8,10,11,12])),
 upward_parent_rank=(torch.empty(0,dtype=torch.long),torch.tensor([0,0,1,1]),torch.tensor([0,0,0,0,1,1,1,1])),
 downward_levels=tuple(torch.from_numpy(x) for x in down))
model=AnonymousCladeTensorGNNV1(event_block_size=1);op=model.clade_coupling
assert sum(p.numel() for p in op.parameters())==1024 and sum(p.numel() for p in model.parameters())==30788
base=(torch.arange(14*32,dtype=torch.float32).reshape(14,32)/59).sin()
ug=torch.full_like(base,.6);dg=torch.full_like(base,.4)
degree=torch.bincount(bank.parent[bank.parent>=0],minlength=14).float().clamp_min(1)
with torch.no_grad():
 upward,outside=model.sweep_tree_state_v1(bank,base,ug,dg,degree)
 leaf=model.leaf_context_v1(bank,upward,outside)
 injection=op(upward,leaf,events)
 assert torch.equal(injection,torch.zeros_like(injection))
 up2,out2=model.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
 assert torch.equal(model.leaf_context_v1(bank,up2,out2),leaf)
# Linear loss proves zeroO sends the first new-core gradient to O, not U/V/W.
upward=upward.detach().requires_grad_();leaf=leaf.detach().requires_grad_()
pattern=torch.arange(14*32,dtype=torch.float32).reshape(14,32)/31
((base+op(upward,leaf,events))*pattern).sum().backward()
assert op.O.weight.grad is not None and bool(op.O.weight.grad.abs().sum()>0)
assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and not bool(p.grad.count_nonzero())
 for p in (op.U.weight,op.V.weight,op.W.weight))
model.zero_grad(set_to_none=True)
with torch.no_grad():op.O.weight.copy_(torch.linspace(-.3,.4,256).reshape(32,8))
node_order=torch.tensor([13,2,8,0,5,11,3,7,1,12,4,10,6,9]);node_inverse=torch.argsort(node_order)
sample_order=torch.tensor([4,1,3,0,2]);sample_inverse=torch.argsort(sample_order)
reversed_events=replace(events,old_node=events.new_node,new_node=events.old_node)
node_events=replace(events,old_node=node_inverse[events.old_node.long()],new_node=node_inverse[events.new_node.long()])
sample_events=replace(events,shared_sample=sample_inverse[events.shared_sample.long()])
with torch.no_grad():
 errors={}
 for arm in ("tensor","event_mlp","pairwise"):
  rank=op.rank_residual(upward,leaf,events,interaction=arm)
  errors[arm+"_reversal"]=float((op.rank_residual(upward,leaf.flip(0),reversed_events,interaction=arm)-rank).abs().max())
  errors[arm+"_node_covariance"]=float((op.rank_residual(upward[node_order],leaf,node_events,interaction=arm)-rank[node_order]).abs().max())
  errors[arm+"_sample_covariance"]=float((op.rank_residual(upward,leaf[:,sample_order],sample_events,interaction=arm)-rank).abs().max())
  assert torch.equal(op(upward,leaf,empty,interaction=arm),torch.zeros(14,32))
  equal=upward.clone();equal[12]=equal[5]
  assert torch.equal(op.rank_residual(equal,leaf,events,interaction=arm),torch.zeros(14,8))
  assert float(rank.sum(0).abs().max())<1e-6
 assert max(errors.values())<1e-6
 # Full shared PyG sweep is storage-node covariant too.
 pnew=bank.parent[node_order].clone();mask=pnew>=0;pnew[mask]=node_inverse[pnew[mask]]
 remapped=SimpleNamespace(parent=pnew,sample_index=node_inverse[bank.sample_index],
  upward_levels=tuple(node_inverse[x] for x in bank.upward_levels),
  upward_children=tuple(node_inverse[x] for x in bank.upward_children),
  upward_parent_rank=bank.upward_parent_rank,downward_levels=tuple(node_inverse[x] for x in bank.downward_levels))
 pu,po=model.sweep_tree_state_v1(remapped,base[node_order],ug[node_order],dg[node_order],degree[node_order])
 ou,oo=model.sweep_tree_state_v1(bank,base,ug,dg,degree)
 errors["shared_PyG_sweep_node_covariance"]=float(torch.maximum((pu-ou[node_order]).abs().max(),(po-oo[node_order]).abs().max()))
 assert errors["shared_PyG_sweep_node_covariance"]<1e-6
permkeys=tuple(keys[i] for i in sample_order.tolist())
plm=choose_landmarks_v1(permkeys,"toy_fixture")
assert [permkeys[i] for i in plm]==[keys[i] for i in lm]
assert np.array_equal(candidate_triples_v1(permkeys,plm),sample_inverse[torch.from_numpy(triples).long()].numpy())
gradients={}
for arm in ("tensor","event_mlp","pairwise"):
 up=upward.detach().clone().requires_grad_();context=leaf.detach().clone().requires_grad_()
 model.zero_grad(set_to_none=True)
 (op(up,context,events,interaction=arm)*pattern).sum().backward()
 gradients[arm]=bool(torch.isfinite(up.grad).all() and torch.isfinite(context.grad).all() and
  all(p.grad is not None and torch.isfinite(p.grad).all() for p in op.parameters()))
 assert gradients[arm] and bool(up.grad[5].abs().sum()>0) and bool(up.grad[12].abs().sum()>0) and bool(context.grad.abs().sum()>0)
print(json.dumps(dict(status="passed_single_handmade_H040_CPU_fixture",eligible_triples=10,changed_events=1,
 old_receiver=5,new_receiver=12,shared_sample=0,rank=8,added_weights=1024,total_model_weights=30788,
 zeroO_original_shared_sweep_identity=True,no_event_and_equal_clade_zero=True,
 first_new_gradient_only_O=True,errors=errors,finite_connected_gradients=gradients,
 actual_data_read=False,GPU_used=False,optimizer_created=False,optimizer_updates=0),sort_keys=True))

"""Tiny handmade CPU FP32 dense-oracle checks; no actual parent payloads."""
from pathlib import Path
from dataclasses import replace
import json
import torch
from arggnn.persistence_trace_kernel_v1 import MetricTreeHoldV1
from arggnn.reference_logcavity_gnn_v1 import ReferenceLogCavityGNNV1,unit_class_fields_v1,masked_logaddexp_v1
from arggnn.reference_dependence_gnn_v1 import (
 ReferenceDependenceGNNV1,prepare_reference_dependence_bank_v1,normalize_moment_v1,
 merge_moment_v1,SOURCE_CONTROLS,ARMS)

torch.set_num_threads(1);torch.manual_seed(101)
parent=[6,7,9,7,7,8,8,8,-1,6,9]
ages=[0.,0.,200.,220.,450.,470.,500.,700.,1000.,250.,210.]
samples=[0,1,2,3,4,5,10]
refs=[2,3,4,5,6];classes=[0,1,1,2,0]
def tree(p=parent,t=ages):
 return MetricTreeHoldV1(torch.tensor(p),torch.tensor(t,dtype=torch.float64),
  torch.arange(len(p)),"handmade-native",torch.tensor(samples),torch.arange(len(samples)))
def prepare(ts):
 return prepare_reference_dependence_bank_v1([ts],torch.tensor([[0.,1000.]],dtype=torch.float64),
  torch.tensor([100.,200.],dtype=torch.float64),torch.tensor([0,1]),torch.tensor(refs),
  torch.tensor(classes),sample_identity_namespace="handmade-samples")
bank=prepare(tree());model=ReferenceDependenceGNNV1();baseline=torch.randn(2,2,4)
checks={}
for arm in SOURCE_CONTROLS:
 assert torch.equal(model(bank,baseline,control=arm),baseline)
checks['zero_head_exact_frozen_skip_all_controls']=True
# Dense oracle uses only small FP32 source/kernel matrices; float64 dates are geometry only.
augparent=bank.forest.parent.tolist()
def placement(ts):
 p,t=ts.parent.tolist(),ts.node_times.tolist()
 for node,ancestor in enumerate(p.copy()):
  if ancestor>=0 and t[node]<350<t[ancestor]:
   p.append(ancestor);t.append(350.);p[node]=len(p)-1
 placed=[]
 counts=bank.reference.metadata
 for identity,cls in zip(refs,classes):
  node=samples[identity];family=int(t[node]>350)
  if not family:
   while t[node]<350:node=p[node]
  count=counts['older_class_counts' if family else 'younger_class_counts'][cls]
  placed.append((node,family,cls,1/count))
 return placed
placed=placement(tree())
def ancestors(node):
 out=[]
 while node>=0:out.append(node);node=augparent[node]
 return out
def distance(a,b,cost):
 aa=ancestors(a);bb=ancestors(b)
 common=next((x for x in aa if x in bb),None)
 if common is None:return cost.new_tensor(float('inf'))
 edges=aa[:aa.index(common)]+bb[:bb.index(common)]
 return cost[torch.tensor(edges,dtype=torch.long)].sum() if edges else cost.new_zeros(())
def dense(node,family,subset,independent,cost):
 donors=[r for r in placed if r[1]==family and (
  subset=='full' or ((node in ancestors(r[0]))==(subset=='up')))]
 if not donors:return cost.new_zeros(4),cost.new_zeros(4,4)
 loga=torch.stack([cost.new_tensor(r[3]).log()-distance(node,r[0],cost[:,family]) for r in donors])
 pi=torch.softmax(loga,0)
 b=torch.stack([torch.nn.functional.one_hot(torch.tensor(r[2]),4).float()*pi[i] for i,r in enumerate(donors)])
 k=torch.eye(len(donors),dtype=torch.float32) if independent else torch.stack([
  torch.stack([(-distance(r[0],s[0],cost[:,family])).exp() for s in donors]) for r in donors])
 assert b.dtype==torch.float32 and k.dtype==torch.float32
 return b.sum(0),b.T@k@b
cost=(bank.reference.edge_basis*torch.nn.functional.softplus(model.raw_cost)[None]).sum(-1)
lu,lo=model.log_cavities(bank.reference)
max_error=0.
for independent in (False,True):
 up,out=model.moment_cavities(bank,independent=independent)
 full=merge_moment_v1(up,out,independent=independent)
 for label,moment,log in [('up',up,lu),('out',out,lo),('full',full,masked_logaddexp_v1(lu,lo))]:
  pm,c,mask=normalize_moment_v1(moment)
  hp,hm=unit_class_fields_v1(log)
  assert torch.allclose(pm.reshape(-1,8),hp,atol=3e-6,rtol=0)
  assert torch.equal(mask,hm)
  for node in range(len(augparent)):
   for family in range(2):
    pexpected,cexpected=dense(node,family,label,independent,cost)
    error=float((c[node,family]-cexpected).detach().abs().max())
    max_error=max(max_error,error)
    assert torch.allclose(pm[node,family],pexpected,atol=3e-6,rtol=0),(independent,label,node,family)
    assert torch.allclose(c[node,family],cexpected,atol=5e-6,rtol=0),(independent,label,node,family,error)
    assert torch.linalg.eigvalsh(c[node,family].detach()).min()>-2e-6
  assert (c>=0).all() and (c.sum((-1,-2))<=1.00001).all()
  assert (c[:,1,0,:]==0).all() and (c[:,1,:,0]==0).all()
checks['dense_FP32_full_subtree_outside_oracle_max_abs_error']=max_error
checks['symmetry_PSD_nonnegative_unit_bound_absent_older_ANA']=True
# Distinct same-class donors share one census source: tree includes both cross terms.
coincident=placed[0][0]
assert placed[0][0]==placed[-1][0]
a=model.local_moment(bank,independent=False);b=model.local_moment(bank,independent=True)
_,ca,_=normalize_moment_v1(a);_,cb,_=normalize_moment_v1(b)
assert ca[coincident,0,0,0]==1 and cb[coincident,0,0,0]==.5
checks['coincident_local_tree_outer1_vs_independent_self_sum_half']=True
# Cost gradient parity also audits the detached-gauge transport compensation.
for independent in (False,True):
 model.zero_grad(set_to_none=True)
 up,out=model.moment_cavities(bank,independent=independent)
 _,c,_=normalize_moment_v1(merge_moment_v1(up,out,independent=independent))
 weights=torch.arange(c.numel(),dtype=torch.float32).reshape_as(c)/c.numel()
 g1=torch.autograd.grad((c*weights).sum(),model.raw_cost)[0]
 cost=(bank.reference.edge_basis*torch.nn.functional.softplus(model.raw_cost)[None]).sum(-1)
 oracle=torch.stack([torch.stack([dense(node,family,'full',independent,cost)[1] for family in range(2)])
  for node in range(len(augparent))])
 g2=torch.autograd.grad((oracle*weights).sum(),model.raw_cost)[0]
 assert torch.isfinite(g1).all() and torch.allclose(g1,g2,atol=8e-6,rtol=2e-4),(g1,g2)
checks['relative_moment_cost_VJP_matches_dense_FP32_oracle']=True
# Exact p calculation remains the immutable H030 path in every condition.
h030=ReferenceLogCavityGNNV1();h030.load_state_dict({k:v for k,v in model.state_dict().items() if k in h030.state_dict()})
with torch.no_grad():
 model.context_head.weight.copy_(torch.randn_like(model.context_head.weight)*.03)
 h030.context_head.weight.copy_(model.context_head.weight)
assert torch.equal(model(bank,baseline,control='first_moment_only'),h030(bank.reference,baseline,control='static'))
for arm in ARMS:
 assert torch.equal(model(bank,baseline,control=arm),h030(bank.reference,baseline,control='static'))
checks['first_moment_control_exact_H030_static_and_zero_Gram_projection_parity']=True
with torch.no_grad():
 for name,p in model.named_parameters():
  if name.startswith('gram_'):p.copy_(torch.randn_like(p)*.05)
for arm in ARMS:
 model.zero_grad(set_to_none=True)
 value=model(bank,baseline,control=arm)
 value.square().sum().backward()
 names=['raw_cost','reference_projection.weight','up_value.weight','down_value.weight',
  'decoder_base.weight','gram_up_gate.weight','gram_down_gate.weight',
  'gram_decoder_up_gate.weight','gram_decoder_down_gate.weight']
 norms={name:float(dict(model.named_parameters())[name].grad.norm()) for name in names}
 assert all(x>0 for x in norms.values()),(arm,norms)
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 checks[arm+'_finite_nonzero_learned_gradients']=norms
# Excluded-subtree evidence does not enter its outside Gram or learned state.
source=bank.reference.log_source.clone();source[coincident,0]+=.3
changed=replace(bank,reference=replace(bank.reference,log_source=source))
up,out=model.moment_cavities(bank);up2,out2=model.moment_cavities(changed)
assert torch.allclose(normalize_moment_v1(out)[1][coincident],normalize_moment_v1(out2)[1][coincident],atol=1e-6,rtol=0)
a,z=model.log_cavities(bank.reference);a2,z2=model.log_cavities(changed.reference)
gu,go,gf=model.grams(bank,a,z,'tree_gram');gu2,go2,gf2=model.grams(changed,a2,z2,'tree_gram')
_,outside=model.encode_dependence(bank,a,z,gu,go)
_,outside2=model.encode_dependence(changed,a2,z2,gu2,go2)
assert torch.allclose(outside[coincident],outside2[coincident],atol=1e-6,rtol=0)
checks['source_subtree_excluded_outside_Gram_and_learned_value']=True
empty=replace(bank,reference=replace(bank.reference,log_source=torch.full_like(bank.reference.log_source,-torch.inf)),
 log_self_source=torch.full_like(bank.log_self_source,-torch.inf))
for arm in SOURCE_CONTROLS:assert torch.equal(model(empty,baseline,control=arm),baseline)
for independent in (False,True):
 model.zero_grad(set_to_none=True)
 u,o=model.moment_cavities(empty,independent=independent)
 p,c,mask=normalize_moment_v1(merge_moment_v1(u,o,independent=independent))
 assert not mask.any() and (p==0).all() and (c==0).all()
 c.sum().backward();assert torch.isfinite(model.raw_cost.grad).all() and (model.raw_cost.grad==0).all()
checks['empty_reference_exact_zero_residual_and_finite_zero_moment_VJP']=True
# Insert a source-free 750-generation node on the 500->1000 edge.
sp=parent.copy();sp[6]=len(sp);sp.append(8)
sub=prepare(tree(sp,ages+[750.]))
for independent in (False,True):
 u,o=model.moment_cavities(bank,independent=independent)
 u2,o2=model.moment_cavities(sub,independent=independent)
 p,c,_=normalize_moment_v1(merge_moment_v1(u,o,independent=independent))
 p2,c2,_=normalize_moment_v1(merge_moment_v1(u2,o2,independent=independent))
 rows=bank.forest.query_nodes;rows2=sub.forest.query_nodes
 assert torch.allclose(p[rows],p2[rows2],atol=3e-6,rtol=0)
 assert torch.allclose(c[rows],c2[rows2],atol=5e-6,rtol=0)
checks['kernel_first_and_Gram_serial_subdivision_parity']=True
checks['nonlinear_GNN_subdivision_difference_expected']=float((model(bank,baseline,control='tree_gram')-
 model(sub,baseline,control='tree_gram')).detach().norm())
with torch.no_grad():model.raw_cost.fill_(1000.)
for independent in (False,True):
 model.zero_grad(set_to_none=True)
 u,o=model.moment_cavities(bank,independent=independent)
 p,c,mask=normalize_moment_v1(merge_moment_v1(u,o,independent=independent))
 assert torch.isfinite(p).all() and torch.isfinite(c).all()
 assert torch.allclose(p.sum(-1)[mask],torch.ones_like(p.sum(-1)[mask]),atol=3e-6,rtol=0)
 assert (c.sum((-1,-2))<=1.00001).all()
 c.square().sum().backward();assert torch.isfinite(model.raw_cost.grad).all()
checks['raw_mass_underflow_regime_finite_unit_Gram_and_cost_VJP']=True
checks['parameter_counts']={arm:model.parameter_counts(arm) for arm in SOURCE_CONTROLS}
checks['geometry']=bank.metadata
Path('research/reviews/reference_dependence_gnn_v1_author/toy_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k!='geometry'},indent=2))

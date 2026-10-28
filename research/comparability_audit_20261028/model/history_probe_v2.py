"""Synthetic CPU history parity using scored checkpoint weights; no optimizer or real observations."""
from pathlib import Path
import hashlib
import json
import torch
from torch.nn import functional as F
from research.reviews.TreeLoom_GPU_history_source_20261017_v1 import stable_leaf_history_v1 as old
from research.benchmark_rebuild_20261024.american.treeloom_speed import stable_leaf_history_v2 as new
root=Path('/workspace-SR003.nfs2/ARG-GNN')
checkpoint=Path('experiments/results/benchmark_campaign_20261018_workflow/release_20261022_full_vram_v2/other/AP__all_train__treeloom_fit/native/checkpoints/epoch_30.pt')
checkpoint_sha256=hashlib.sha256((root/checkpoint).read_bytes()).hexdigest()
assert checkpoint_sha256=='c06fc82db27d21cb7933da07d4247692c18b5c96df9cb67cb0bff71453e3f5e3'
state=torch.load(root/checkpoint,map_location='cpu',weights_only=True)['model_state']
torch.manual_seed(1028)
t,h,r,g=1031,12,8,7
forcing=F.linear(torch.randn(t*g,32),state['memory.U.weight'])
groups=torch.arange(t)[:,None]*g+(torch.arange(h)[None,:]+torch.arange(t)[:,None])%g
offsets=tuple(range(0,t*g+1,g))
counts=torch.bincount(groups.flatten(),minlength=t*g).float()[:,None]
spans=torch.exp(torch.linspace(-2,8,t))
theta=state['memory.ell_theta']
ell=100.+(100000.-100.)*theta.sigmoid()
rho=torch.exp(-spans/ell)
source=-torch.expm1(-spans/ell)
result={}
left,right=[],[]
with torch.no_grad():
 for reverse in (False,True):
  a=old._Direction.apply(forcing,rho,source,groups,offsets,reverse,128,torch.device('cpu'),lambda label:None,old.NullProfileV1())
  b=new._Direction.apply(forcing,rho,source,groups,offsets,reverse,1024,torch.device('cpu'),lambda label:None,new.NullProfileV1())
  left.append(a);right.append(b)
  diff=(a-b).abs()
  result[str(reverse)]=dict(max_abs=float(diff.max()),mean_abs=float(diff.mean()),reference_max_abs=float(a.abs().max()))
  assert float(diff.max())<2e-5
 def project(directions):
  leaves=.5*(directions[0]+directions[1])
  pooled=forcing.new_zeros((t*g,r)).index_add(0,groups.flatten(),leaves.flatten(0,1))/counts
  return F.linear(pooled-forcing,state['memory.O.weight'])
 delta=(project(left)-project(right)).abs()
assert not torch.cuda.is_initialized()
assert hashlib.sha256((root/checkpoint).read_bytes()).hexdigest()==checkpoint_sha256
result=dict(schema='synthetic_history_forward_parity_with_scored_weights_v2',holds=t,haplotypes=h,rank=r,groups_per_hold=g,
 checkpoint_ref=dict(path=str(checkpoint),sha256=checkpoint_sha256),ell_theta=float(theta),ell=float(ell),
 directions=result,projected_memory_delta_max_abs=float(delta.max()),projected_memory_delta_mean_abs=float(delta.mean()),
 cuda_initialized=False,full_parent_forward_test=False,optimizer_steps=0,
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p=root/'research/comparability_audit_20261028/model/history_probe.v2.json'
with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))

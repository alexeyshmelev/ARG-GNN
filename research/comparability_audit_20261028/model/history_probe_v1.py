"""Synthetic CPU forward parity probe; no model, optimizer or real observations."""
from pathlib import Path
import hashlib
import json
import torch
from research.reviews.TreeLoom_GPU_history_source_20261017_v1 import stable_leaf_history_v1 as old
from research.benchmark_rebuild_20261024.american.treeloom_speed import stable_leaf_history_v2 as new
root=Path('/workspace-SR003.nfs2/ARG-GNN')
torch.manual_seed(1028)
t,h,r,g=1031,12,8,7
forcing=torch.randn(t*g,r)
groups=torch.arange(t)[:,None]*g+(torch.arange(h)[None,:]+torch.arange(t)[:,None])%g
offsets=tuple(range(0,t*g+1,g))
spans=torch.exp(torch.linspace(-2,8,t))
ell=torch.tensor(55000.)
rho=torch.exp(-spans/ell)
source=-torch.expm1(-spans/ell)
result={}
with torch.no_grad():
 for reverse in (False,True):
  a=old._Direction.apply(forcing,rho,source,groups,offsets,reverse,128,torch.device('cpu'),lambda label:None,old.NullProfileV1())
  b=new._Direction.apply(forcing,rho,source,groups,offsets,reverse,1024,torch.device('cpu'),lambda label:None,new.NullProfileV1())
  diff=(a-b).abs()
  result[str(reverse)]=dict(max_abs=float(diff.max()),mean_abs=float(diff.mean()),reference_max_abs=float(a.abs().max()))
  assert float(diff.max())<2e-5
assert not torch.cuda.is_initialized()
result=dict(schema='synthetic_history_forward_parity_v1',holds=t,haplotypes=h,rank=r,groups_per_hold=g,
 directions=result,cuda_initialized=False,full_parent_forward_test=False,
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p=root/'research/comparability_audit_20261028/model/history_probe.v1.json'
with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))

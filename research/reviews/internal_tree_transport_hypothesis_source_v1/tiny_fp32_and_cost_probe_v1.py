"""One synthetic FP32 operator check and one full-length32-query CPU scan probe."""
from pathlib import Path
import copy,hashlib,itertools,json,resource,time,sys
import torch
from torch_geometric.nn import MessagePassing
from arggnn.marker_resolution_readout_v1 import MarkerEvidenceBankV1,MarkerResolutionReadoutV1
from arggnn.marker_grid_reset_chain_gnn_v1 import (
    ARMS,MarkerGridResetChainGNNV1,disjoint_bin_features_v1,
    reset_log_transition_v1,reset_chain_log_marginals_v1,
)
OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(101)
receipt={'scope':'synthetic_only_no_scientific_payload_no_GPU_no_optimizer_updates',
         'dtype':'float32','torch':torch.__version__,'threads':torch.get_num_threads()}
p,b=5,2
emission=torch.randn(p,b,4,dtype=torch.float32,requires_grad=True)
log_rate=torch.full((p-1,b),-8.,dtype=torch.float32,requires_grad=True)
beta=torch.tensor([.1,-.2,.3],dtype=torch.float32,requires_grad=True)
log_pi=torch.cat((beta,beta.new_zeros(1))).log_softmax(-1)
delta=torch.tensor([100.,100.,100.,100.],dtype=torch.float32)
rate=log_rate.exp()
transition=reset_log_transition_v1(rate,delta,log_pi)
generator=rate[0,0]*(log_pi.exp()[None].expand(4,4)-torch.eye(4,dtype=torch.float32))
analytic_error=(transition[0,0].exp()-torch.matrix_exp(delta[0]*generator)).abs().max().item()
result=reset_chain_log_marginals_v1(emission,rate,delta,log_pi)
paths=torch.tensor(list(itertools.product(range(4),repeat=p)),dtype=torch.long)
weights=log_pi[paths[:,0]][:,None].expand(-1,b)
for j in range(p):
    weights=weights+emission[j].T.index_select(0,paths[:,j])
    if j:
        weights=weights+transition[j-1][:,paths[:,j-1],paths[:,j]].T
posterior=weights.softmax(0)
rows=[]
for j in range(p):
    row=posterior.new_zeros((4,b)).index_add(0,paths[:,j],posterior)
    rows.append(row.T)
oracle=torch.stack(rows).log()
error=(result-oracle).abs().max().item()
loss=-result[...,0].mean()
actual_grad=torch.autograd.grad(loss,(emission,log_rate,beta),retain_graph=True)
expected_grad=torch.autograd.grad(-oracle[...,0].mean(),(emission,log_rate,beta))
grad_error=max((a-e).abs().max().item() for a,e in zip(actual_grad,expected_grad))
assert analytic_error<1e-6 and error<5e-6 and grad_error<5e-6
assert all(torch.isfinite(g).all() for g in actual_grad)
receipt['tiny_chain']={'positions':p,'queries':b,'enumerated_paths':len(paths),
    'analytic_transition_max_abs':analytic_error,'log_marginal_max_abs':error,
    'gradient_max_abs':grad_error,'row_sum_max_abs':(transition.exp().sum(-1)-1).abs().max().item()}

# Small categorical observation fixture, with exact interior-edge membership.
genotype=torch.tensor([[0,0,1,1,0,1],[0,1,1,0,1,0],[0,1,0,1,-1,0],
                       [1,0,0,1,1,1],[1,1,0,0,0,0],[0,1,1,0,1,1]],dtype=torch.int8)
positions=torch.tensor([0,99,100,199,200,499],dtype=torch.long)
centers=torch.arange(50,500,100,dtype=torch.long)
holds=torch.tensor([0,0,1,2,2],dtype=torch.long)
markers=MarkerEvidenceBankV1(genotype,torch.tensor([4,5]),positions,centers,
    torch.full((5,),100.,dtype=torch.float32),holds,
    torch.randn(6,4,4,dtype=torch.float32),torch.ones(6,4,dtype=torch.bool),
    torch.empty((5,3),dtype=torch.long),torch.empty((5,3),dtype=torch.long),
    holds,3,3,'synthetic_only')
features=disjoint_bin_features_v1(markers,torch.arange(0,501,100,dtype=torch.long))
expected_count=torch.tensor([[2,2],[1,2],[1,1],[0,0],[1,1]],dtype=torch.float32)
assert torch.allclose(features[...,4].expm1(),expected_count,atol=2e-6)
assert torch.equal(features[3],torch.zeros_like(features[3]))
model=MarkerGridResetChainGNNV1(checkpoint_chain=False)
other=copy.deepcopy(model);other.arm=ARMS[1]
assert sum(v.numel() for v in model.parameters())==sum(v.numel() for v in other.parameters())==31933
assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
assert MarkerGridResetChainGNNV1.backbone_state_v1 is MarkerResolutionReadoutV1.backbone_state_v1
assert sum(isinstance(m,MessagePassing) for m in model.backbone.modules())==2
base=torch.randn((3,2,4),dtype=torch.float32,requires_grad=True)
context=torch.randn((3,2,32),dtype=torch.float32,requires_grad=True)
left=torch.tensor([0,0,1,2]);right=torch.tensor([0,1,2,2]);edge=torch.arange(4)
args=(base,context,features,holds,delta,left,right,edge)
a=model.decode_block_v1(*args);z=other.decode_block_v1(*args)
assert torch.equal(a,z)
chunks=torch.cat([model.decode_block_v1(base[:,q:q+1],context[:,q:q+1],features[:,q:q+1],
    holds,delta,left,right,edge) for q in range(2)],1)
assert torch.allclose(a,chunks,atol=3e-6,rtol=3e-6)
(-a[...,1].mean()).backward()
assert base.grad is not None and torch.isfinite(base.grad).all() and base.grad.abs().sum()>0
assert model.rate_head[-1].weight.grad is not None and model.rate_head[-1].weight.grad.abs().sum()>0
receipt['tiny_model']={'parameters_per_arm':31933,'bit_identical_initial_pair':True,
    'actual_inherited_PyG_message_modules':2,'exact_inherited_H042_state_method':True,
    'query_chunk_max_abs':(a-chunks).abs().max().item(),'SNP_bin_count_max_abs':
    (features[...,4].expm1()-expected_count).abs().max().item(),
    'finite_nonzero_initial_rate_output_gradient':True,
    'full_tree_forward_executed':False,'actual_H042_checkpoint_loaded':False}
receipt['source_sha256']=hashlib.sha256(Path(__import__('arggnn.marker_grid_reset_chain_gnn_v1',fromlist=['SOURCE']).__file__).read_bytes()).hexdigest()
(OUT/'tiny_fp32_receipt_final_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
if "--tiny-only" in sys.argv:
    raise SystemExit(0)

# One bounded, full-length CPU head scan. Synthetic emissions and rates only.
p,b=20000,32
e=torch.randn((p,b,4),dtype=torch.float32,requires_grad=True)
r=torch.full((p-1,b),2e-5,dtype=torch.float32,requires_grad=True)
d=torch.full((p-1,),100.,dtype=torch.float32)
lp=torch.full((4,),-1.38629436112,dtype=torch.float32)
start=time.perf_counter()
y=reset_chain_log_marginals_v1(e,r,d,lp)
forward=time.perf_counter()-start
start=time.perf_counter()
(-y[...,0].mean()).backward()
backward=time.perf_counter()-start
assert torch.isfinite(y).all() and torch.isfinite(e.grad).all() and torch.isfinite(r.grad).all()
cost={'scope':'one_synthetic_CPU_scan_no_scientific_payload_no_backbone_no_optimizer',
      'positions':p,'queries':b,'dtype':'float32','threads':torch.get_num_threads(),
      'forward_seconds':forward,'backward_seconds':backward,
      'maximum_RSS_bytes_including_imports_tiny_and_probe':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      'finite_forward_and_gradients':True,'GPU_performance_or_whole_model_fit_qualified':False,
      'source_sha256':hashlib.sha256(Path(__import__('arggnn.marker_grid_reset_chain_gnn_v1',fromlist=['SOURCE']).__file__).read_bytes()).hexdigest()}
(OUT/'bounded_CPU_scan_cost_v1.json').write_text(json.dumps(cost,indent=2)+'\n')
print(json.dumps(cost,indent=2),flush=True)

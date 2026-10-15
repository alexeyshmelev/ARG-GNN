"""One handmade CPU implementation check, not a fit or equivalence campaign."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import math
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"src"), str(HERE)]
from order2_cut_memory_v1 import (
    ARMS, CutMemoryV1, Order2CutMarkerGridV1, prepare_cut_plan_v1,
    make_matched_family_v1, H042_PASS30_REF, H042_COMPLETION_REF, MODEL)
from arggnn.marker_grid_reset_chain_gnn_v1 import MarkerGridResetChainGNNV1
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from arggnn.anonymous_clade_flag_control_v1 import build_clade_flags_v1

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(101)
H,T = 952,7
parents=[];times=[];sample_rows=[];children=[]
age=torch.zeros(H);age[-1]=360.
for at in range(T):
    off=len(parents);sample_rows.append(list(range(off,off+H)))
    p=[-1]*(H+5);tm=age.tolist()+[100.,100.,100.,200. if at in (1,2,4) else 500.,800.]
    pair0,pair1,rest,four,root=range(H,H+5)
    groups=((0,2),(1,3)) if at==5 else ((0,1),(2,3))
    for u,group in zip((pair0,pair1),groups):
        for leaf in group:p[leaf]=u
    for leaf in range(4,H-1):p[leaf]=rest
    p[H-1]=root;p[pair0]=four;p[pair1]=four;p[rest]=root;p[four]=root
    parents.extend([v+off if v>=0 else -1 for v in p]);times.extend(tm)
parent=torch.tensor(parents,dtype=torch.long);N=len(parents)
kids=[[] for _ in range(N)]
for child,p in enumerate(parents):
    if p>=0:kids[p].append(child)
height=[0]*N
for u in range(N):
    if kids[u]:height[u]=1+max(height[c] for c in kids[u])
levels=[];level_children=[];ranks=[]
for ht in range(max(height)+1):
    ns=[u for u,v in enumerate(height) if v==ht];levels.append(torch.tensor(ns))
    cs=[];rs=[]
    for rank,u in enumerate(ns):
        cs.extend(kids[u]);rs.extend([rank]*len(kids[u]))
    level_children.append(torch.tensor(cs,dtype=torch.long));ranks.append(torch.tensor(rs,dtype=torch.long))
depth=[0]*N
for u in range(N-1,-1,-1):
    if parents[u]>=0:depth[u]=depth[parents[u]]+1
down=tuple(torch.tensor([u for u,d in enumerate(depth) if d==at],dtype=torch.long)
           for at in range(1,max(depth)+1))
query=torch.tensor([1,3]+list(range(4,688)),dtype=torch.long)
ref=torch.full((H,),-1,dtype=torch.long)
ri=[i for i in range(H) if i not in set(query.tolist())]
for j,i in enumerate(ri):ref[i]=j%4
sample_index=torch.tensor(sample_rows,dtype=torch.long)
holds=torch.arange(T).repeat_interleave(2)
bank=MutationRoutedWholeTreeInputsV1(
    parent=parent,node_time=torch.tensor(times,dtype=torch.float32),sample_index=sample_index,
    upward_levels=tuple(levels),upward_children=tuple(level_children),upward_parent_rank=tuple(ranks),
    downward_levels=down,edge_geometry=torch.zeros(N,4),
    branch_support_true=torch.zeros(N,5),branch_support_sham=torch.zeros(N,5),
    genotype=(torch.arange(T)[:,None]+torch.arange(H)[None,:]).remainder(4).to(torch.int8),
    log_donor_allele_frequency=torch.full((T,4,8),-math.log(4.)),donor_call_mask=torch.ones(T,8,dtype=torch.bool),
    panel_allele_frequency=torch.full((T,4),.25),site_entropy=torch.ones(T),site_hold=torch.arange(T),
    spans=torch.full((T,),200.),readout_hold=holds,query_column=query,
    reference_class=ref,sample_age=age)
cut,meta=prepare_cut_plan_v1(bank)
assert cut.episode_of_hold==(0,1,1,2,3,4,5)
assert meta["above_horizon_actual_samples"]==1 and meta["cut_episodes"]==6
assert bool((cut.group_counts>0).all())
# Same fixture: an exact scalar field exercises both scientific contrasts.
# A,B,B,A gives nonzero conditional dependence; later B,C,A loses old within-C history.
forcing=torch.zeros(len(cut.representatives),8)
forcing[0,0]=1.
direction={}
for arm in ARMS:
    op=CutMemoryV1(arm=arm,checkpoint_memory=False)
    with torch.no_grad():direction[arm]=op._direction(forcing,bank.spans,cut,False)
d3=torch.exp(-bank.spans[1:4].sum()/5000.)
d6=torch.exp(-bank.spans[1:7].sum()/5000.)
assert meta["conditional_information_nats_per_leaf"][0]>0
assert torch.allclose(direction[ARMS[0]][3,torch.tensor([0,2]),0],torch.stack((d3,d3*0)),atol=2e-7,rtol=1e-6)
assert torch.allclose(direction[ARMS[1]][3,torch.tensor([0,2]),0],torch.stack((d3*.5,d3*.5)),atol=2e-7,rtol=1e-6)
assert torch.allclose(direction[ARMS[0]][6,torch.tensor([0,2]),0],torch.stack((d6*.5,d6*.5)),atol=2e-7,rtol=1e-6)
assert torch.allclose(direction[ARMS[2]][6,torch.tensor([0,2]),0],torch.stack((d6,d6*0)),atol=2e-7,rtol=1e-6)
flags,_=build_clade_flags_v1(bank)
pairs,inverse=torch.unique_consecutive(torch.stack((holds[:-1],holds[1:]),-1),dim=0,return_inverse=True)
feature=torch.zeros(2*T,686,5);feature[...,4]=1.
feature[::2,:,0]=.2;feature[1::2,:,1]=-.1
grid=SimpleNamespace(features_cpu=feature,grid_hold=holds,delta_bp=torch.full((2*T-1,),100.),
                     pair_left=pairs[:,0],pair_right=pairs[:,1],edge_pair=inverse)
# Synthetic H042-shaped state only. No checkpoint bytes or authenticated scientific result.
synthetic=MarkerGridResetChainGNNV1()
payload=dict(schema="anonymous_clade_flag_checkpoint_v1",hypothesis_id="H042",arm="exact_clade_flags",
             control_id="H040_exact_clade_flag_control_v1",pass_index=30,additional_optimizer_steps=960,
             parameter_count=30788,seed=101,teacher_used=False,model_kwargs=MODEL,
             model_state=synthetic.backbone.state_dict())
family,opts,_=make_matched_family_v1(payload,checkpoint_ref=H042_PASS30_REF,
                                    completion_ref=H042_COMPLETION_REF,
                                    checkpoint_chain=True,checkpoint_memory=True)
assert all(sum(p.numel() for p in m.parameters())==32446 for m in family.values())
assert all(not opt.state for opt in opts.values())
assert all(torch.equal(v,family[ARMS[0]].state_dict()[k])
           for m in family.values() for k,v in m.state_dict().items())
initial={}
with torch.no_grad():
    for a,m in family.items():
        m.eval()
        state=m._backbone_state(bank,flags,cut)
        initial[a]=m._decode_state(state,bank,grid)
        assert initial[a].shape==(2*T,686,4) and bool(torch.isfinite(initial[a]).all())
        assert torch.allclose(initial[a].logsumexp(-1),torch.zeros(2*T,686),atol=2e-5,rtol=0)
assert all(torch.equal(initial[a],initial[ARMS[0]]) for a in ARMS)
# Open only the new zero-output head by a deterministic synthetic assignment.
# One backward traverses the real whole-tree/PyG/GRU/grid-chain path. No optimizer.step.
for m in family.values():
    with torch.no_grad():m.memory.O.weight.copy_(torch.linspace(-.03,.04,256).reshape(32,8))
chosen=family[ARMS[0]]
chosen.train()
value=chosen._decode_state(chosen._backbone_state(bank,flags,cut),bank,grid)
loss=-value[...,0].mean()
loss.backward()
gradient={
    "U_finite_connected": chosen.memory.U.weight.grad is not None
        and bool(torch.isfinite(chosen.memory.U.weight.grad).all())
        and bool(chosen.memory.U.weight.grad.abs().sum()>0),
    "ell_finite_connected": chosen.memory.ell_theta.grad is not None
        and bool(torch.isfinite(chosen.memory.ell_theta.grad))
        and bool(chosen.memory.ell_theta.grad.abs()>0),
    "O_finite_connected": chosen.memory.O.weight.grad is not None
        and bool(torch.isfinite(chosen.memory.O.weight.grad).all())
        and bool(chosen.memory.O.weight.grad.abs().sum()>0),
    "all_existing_gradients_finite": all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                       for p in chosen.parameters())}
assert all(gradient.values()),gradient
assert all(not opt.state for opt in opts.values())
# Immutable source version binding fails even for a zero-valued in-place write.
with torch.no_grad():bank.spans.add_(0.)
try:cut.validate(bank)
except ValueError:mutation_rejected=True
else:raise AssertionError("Source version mutation must be rejected")
receipt=dict(schema="order2_single_CPU_implementation_check_v1",status="passed",
             fixture="one handmade7hold952leaf686query forest;7observed synthetic SNPs;6distinctcutepisodes;one360generationactualsample",
             synthetic_only=True,private_arrays_or_checkpoint_files_opened=False,
             model_parameters_per_arm=32446,memory_parameters_per_arm=513,
             initial_three_arm_grid_identity=True,grid_shape=[2*T,686,4],
             original_grid_decoder_method_reused=True,PyG_complete_tree_sweeps=True,
             real_CPU_backward_calls=1,optimizer_updates=0,GPU_used=False,
             connected_gradient_checks=gradient,source_version_mutation_rejected=mutation_rejected,
             nonzero_triple_weight_vs_independence_witness=True,
             later_pair_pooling_vs_leaf_witness=True,
             cut_metadata=meta,
             limitations=["no actual bank/current checkpoint qualification",
                          "no original20000bin runtime/memory qualification",
                          "no fit/accuracy/ancestry claim",
                          "bound production bundle transfer remains root integration"])
(HERE/"CPU_check_receipt_v1.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))

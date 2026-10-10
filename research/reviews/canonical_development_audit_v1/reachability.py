"""Isolated CPU counterfactual of a tree hit by no source anchor/readout."""
from pathlib import Path
from dataclasses import replace
import json,sys,math
import torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');sys.path.insert(0,str(ROOT/'tests'))
from test_tree_canonicalization import example,duplicate_intervals
from arggnn.reference_encoder import ReferenceFeatureAdapter
from arggnn.kron_interface import precompute_leaf_interfaces,with_query_reference_cache
from arggnn.anchor_kernel import prepare_anchor_kernel_bank,AnchorKernelModel
from arggnn.anchor_temporal_baselines import AnchorGRUModel,AnchorTemporalTransformerModel,ObservedReferenceCoordinateSmoother
from arggnn.canonical_gru import prepare_canonical_parent,prepare_canonical_gnn_input,CanonicalTreeGNNGRU
from arggnn.argformer_control import ARGMixTreeControl,WholeTreeGraphTransformerControl
ADAPTER={'reference_columns':(0,1,2,3),'time_column':6,'branch_column':7,'time_scale':10000.,'kernel_scales':(.05,.1,.2)}
torch.set_num_threads(1)
seq=duplicate_intervals(example())
seq.intervals=torch.tensor([[0.,.45],[.45,.55],[.55,1.]],dtype=torch.float64)
seq.positions=torch.tensor([.1,.9],dtype=torch.float64);seq.target_graph=torch.tensor([0,2])
changed=replace(seq,x=seq.x.clone())
n=len(example().x)
# Genuine middle tree age difference, and recomputed raw branch fields.
changed.x[n+6,6]=math.log1p(700.)/math.log1p(10000.)
ages=torch.expm1(changed.x[:,6]*math.log1p(10000.));branch=torch.zeros_like(ages)
for u,v in changed.edge_index.T.tolist():
    if ages[v]>ages[u]:branch[u]=ages[v]-ages[u]
changed.x[n:2*n,7]=(torch.log1p(branch)/math.log1p(10000.))[n:2*n]
def augment(s):return ReferenceFeatureAdapter(8,**ADAPTER).precompute(s)
def bank(s):
    b=precompute_leaf_interfaces(augment(s),time_column=6,parent_branch_column=7,resistance_scale=1e-4)
    return prepare_anchor_kernel_bank(with_query_reference_cache(b,torch.tensor([30,40,50,60])),num_anchors=2,num_scales=3)
a,b=bank(seq),bank(changed)
assert a.layout.anchor_graph.tolist()==[0,2] and a.bank.target_graph.tolist()==[0,2]
results={'intervening_tree_interval':[.45,.55],'anchor_positions':a.layout.anchors.tolist(),'readouts':seq.positions.tolist(),'models':{}}
for cls in (AnchorGRUModel,AnchorTemporalTransformerModel):
    torch.manual_seed(928)
    model=cls(47,4,hidden_dim=8,num_layers=2,latent_dim=3,num_anchors=2,transformer_heads=2).double().eval()
    delta=float((model(a)-model(b)).abs().max())
    assert delta==0
    results['models'][cls.__name__]={'maximum_logit_change':delta}
model=ObservedReferenceCoordinateSmoother(num_anchors=2)
delta=float((model(a)-model(b)).abs().max());assert delta==0
results['models']['ObservedReferenceCoordinateSmoother']={'maximum_logit_change':delta}
torch.manual_seed(928)
model=AnchorKernelModel(47,4,hidden_dim=8,num_layers=2,latent_dim=3,num_anchors=2,num_scales=3,initial_coupling=[5.,20.,80.]).double().eval()
delta=float((model(a)-model(b)).abs().max());assert delta>1e-9
results['models']['AnchorKernelModel']={'maximum_logit_change':delta}
ca,cb=prepare_canonical_parent(seq,ADAPTER),prepare_canonical_parent(changed,ADAPTER)
# Exact duplicates collapse in original, distinct middle tree remains in changed.
# To separate token-count from age effects, use a distinct baseline middle tree too.
base=replace(seq,x=seq.x.clone());base.x[n+6,6]=math.log1p(600.)/math.log1p(10000.)
ages=torch.expm1(base.x[:,6]*math.log1p(10000.));branch=torch.zeros_like(ages)
for u,v in base.edge_index.T.tolist():
    if ages[v]>ages[u]:branch[u]=ages[v]-ages[u]
base.x[n:2*n,7]=(torch.log1p(branch)/math.log1p(10000.))[n:2*n]
ca=prepare_canonical_parent(base,ADAPTER)
assert ca.sequence.num_graphs==cb.sequence.num_graphs==3
torch.manual_seed(928)
model=CanonicalTreeGNNGRU(48,4,hidden_dim=8).double().eval()
delta=float((model(prepare_canonical_gnn_input(ca.sequence))-model(prepare_canonical_gnn_input(cb.sequence))).abs().max());assert delta>1e-9
results['models']['CanonicalTreeGNNGRU']={'maximum_logit_change':delta,'canonical_trees_both':3}
for cls in (ARGMixTreeControl,WholeTreeGraphTransformerControl):
    torch.manual_seed(928)
    model=cls(48,4,hidden_dim=8,num_heads=2,num_layers=2,dropout=0.).double().eval()
    delta=float((model(augment(seq))-model(augment(changed))).abs().max());assert delta==0
    results['models'][cls.__name__]={'maximum_logit_change':delta,'cross_tree_aggregation':'none'}
(ROOT/'research/reviews/canonical_development_audit_v1/reachability.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

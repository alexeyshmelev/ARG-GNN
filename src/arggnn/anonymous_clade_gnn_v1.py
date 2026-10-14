"""H040 anonymous two-tree clade incidence and latent rerouting. Inactive model API."""
from __future__ import annotations
from dataclasses import fields
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
from .mutation_routed_whole_tree_gnn_v1 import _FLOAT_FIELDS,_LEVEL_FIELDS
from .anonymous_clade_witness_v1 import AnonymousCladeEventsV1,EVENT_INDEX_FIELDS
SOURCE="src/arggnn/anonymous_clade_gnn_v1.py"
MAIN_ARMS=("tensor","event_mlp")
OPTIONAL_ARMS=("pairwise",)
NEW_KEYS={"clade_coupling.U.weight","clade_coupling.V.weight","clade_coupling.W.weight","clade_coupling.O.weight"}


def validate_events_v1(events,upward,leaf_context):
 if not isinstance(events,AnonymousCladeEventsV1):raise TypeError("Verified explicit observation-only incidence required")
 if (upward.dtype!=torch.float32 or leaf_context.dtype!=torch.float32 or upward.ndim!=2
  or leaf_context.ndim!=3 or upward.shape[1]!=32 or leaf_context.shape[-1]!=32
  or upward.shape[0]!=events.native_nodes or leaf_context.shape[:2]!=(events.native_holds,events.haplotypes)
  or upward.device!=leaf_context.device):
  raise ValueError("Complete FP32 native node/sample contexts and event dimensions must agree")
 e=events.weight.numel()
 for name in EVENT_INDEX_FIELDS:
  value=getattr(events,name)
  if value.ndim!=1 or value.numel()!=e or value.dtype not in (torch.int32,torch.int64) or value.device!=upward.device or value.requires_grad:
   raise ValueError("Read-only bounded event index vectors on the bank device required")
 if events.weight.shape!=(e,) or events.weight.dtype!=torch.float32 or events.weight.device!=upward.device or events.weight.requires_grad:
  raise ValueError("Read-only observed FP32 degree weights required")
 # Value bounds and winning-LCA proofs are preparation/read-time checks, not GPU scalar scans per update.


class RankEightCladeCouplingV1(nn.Module):
 """Established antisymmetric trilinear core; new candidate object is incidence/rerouting."""
 def __init__(self,*,event_block_size=32768,checkpoint_event_blocks=True):
  super().__init__()
  if type(event_block_size) is not int or event_block_size<1:raise ValueError("Positive bounded event block required")
  self.U=nn.Linear(32,8,bias=False);self.V=nn.Linear(32,8,bias=False);self.W=nn.Linear(32,8,bias=False)
  self.O=nn.Linear(8,32,bias=False);nn.init.zeros_(self.O.weight)
  self.event_block_size=event_block_size;self.checkpoint_event_blocks=bool(checkpoint_event_blocks)

 @staticmethod
 def _event_block(pu,pv,pw,boundary,old_node,new_node,shared,weight,interaction):
  boundary,old_node,new_node,shared=(v.long() for v in (boundary,old_node,new_node,shared))
  a=.5*(pu[boundary,shared]+pu[boundary+1,shared])
  vb,vc=pv[old_node],pv[new_node];wb,wc=pw[old_node],pw[new_node]
  if interaction=="tensor":z=a*(vb*wc-vc*wb)
  elif interaction=="event_mlp":z=F.gelu(a+vb+wc)-F.gelu(a+vc+wb)
  elif interaction=="pairwise":z=a*((vb+wb)-(vc+wc))+vb*wc-vc*wb
  else:raise ValueError("Prospectively fixed tensor/event_mlp or optional pairwise arm required")
  return z*weight[:,None]

 def rank_residual(self,upward,leaf_context,events,*,interaction="tensor"):
  validate_events_v1(events,upward,leaf_context)
  if interaction not in MAIN_ARMS+OPTIONAL_ARMS:raise ValueError("Registered latent interaction required")
  if any(p.dtype!=torch.float32 or p.device!=upward.device for p in self.parameters()):
   raise ValueError("FP32 coupling parameters on complete observed bank device required")
  residual=upward.new_zeros(events.native_nodes,8)
  if not events.weight.numel():return residual
  # Project each native node/sample once; no HxH matrix or repeated full-state projection per event.
  pu,pv,pw=self.U(leaf_context),self.V(upward),self.W(upward)
  for start in range(0,events.weight.numel(),self.event_block_size):
   part=slice(start,min(events.weight.numel(),start+self.event_block_size))
   args=(pu,pv,pw,events.boundary_index[part],events.old_node[part],
    events.new_node[part],events.shared_sample[part],events.weight[part],interaction)
   if self.checkpoint_event_blocks and self.training and torch.is_grad_enabled():
    message=checkpoint(self._event_block,*args,use_reentrant=False)
   else:message=self._event_block(*args)
   # Only incident rank8 vectors cross the two anonymous trees. Endpoint ids are local indexes.
   residual.index_add_(0,events.old_node[part].long(),-message)
   residual.index_add_(0,events.new_node[part].long(),message)
  return residual

 def forward(self,upward,leaf_context,events,*,interaction="tensor"):
  return self.O(self.rank_residual(upward,leaf_context,events,interaction=interaction))


class AnonymousCladeTensorGNNV1(MutationRoutedFullPanelPyGGNNV1):
 """Original29764-weight body plus1024 weights, two fresh shared full-tree PyG sweeps."""
 def __init__(self,hidden_dim=32,*,expected_haplotypes=952,site_chunk_size=256,checkpoint_sites=True,
  event_block_size=32768,checkpoint_event_blocks=True):
  if hidden_dim!=32 or expected_haplotypes!=952:raise ValueError("Frozen width32/all952 H038 body required")
  super().__init__(hidden_dim=hidden_dim,expected_haplotypes=expected_haplotypes,
   site_chunk_size=site_chunk_size,checkpoint_sites=checkpoint_sites)
  self.clade_coupling=RankEightCladeCouplingV1(event_block_size=event_block_size,
   checkpoint_event_blocks=checkpoint_event_blocks)

 def prepare_tree_base_v1(self,bank,*,control="aligned"):
  """Unchanged H038 site/leaf/node inputs and gates, computed only once."""
  if control!="aligned":raise ValueError("H040 compares latent interactions over the same aligned observations")
  t,h=bank.sample_index.shape;n,d=len(bank.parent),self.hidden_dim
  site_leaf=self.encode_sites(bank);labels=bank.reference_class
  onehot=F.one_hot(labels.clamp_min(0),4).to(site_leaf.dtype)*(labels>=0)[:,None]
  unknown=(labels<0).to(site_leaf.dtype)[:,None];age=(bank.sample_age.log1p()/math.log(10001.))[:,None]
  observed_leaf=torch.cat((onehot,unknown,age),-1)[None].expand(t,h,6)
  leaf_base=self.leaf_projection(torch.cat((site_leaf,observed_leaf),-1))
  sampled=bank.node_time.new_zeros(n);sampled.index_fill_(0,bank.sample_index.flatten(),1.)
  root=(bank.parent<0).to(sampled.dtype);node_age=bank.node_time.log1p()/math.log(10001.)
  base=self.node_projection(torch.stack((node_age,root,sampled),-1))
  base=base.index_add(0,bank.sample_index.flatten(),leaf_base.reshape(-1,d))
  raw=bank.branch_support_true
  support=torch.cat((raw[:,:3],raw[:,3:4].log1p()/math.log(100001.),raw[:,4:5]),-1)
  gate_input=torch.cat((bank.edge_geometry,support),-1)
  ug,dg=self.up_gate(gate_input).sigmoid(),self.down_gate(gate_input).sigmoid()
  nonroot=bank.parent>=0;degree=torch.bincount(bank.parent[nonroot],minlength=n).to(base.dtype).clamp_min(1)
  return base,ug,dg,degree

 def sweep_tree_state_v1(self,bank,base,ug,dg,degree):
  """Fresh buffers, actual inherited PyG messages and original recipient exclusion."""
  n,d=base.shape;upward=base.new_zeros(n,d);message_sum=base.new_zeros(n,d);child_message=base.new_zeros(n,d)
  for nodes,children,ranks in zip(bank.upward_levels,bank.upward_children,bank.upward_parent_rank):
   if children.numel():
    aggregate,message=self.pyg_upward(upward[children],ug[children],
     degree[bank.parent[children]],ranks,len(nodes),self.up_message.weight)
    child_message.index_copy_(0,children,message)
   else:aggregate=base.new_zeros(len(nodes),d)
   message_sum.index_copy_(0,nodes,aggregate)
   value=self.up_norm(base[nodes]+self.up_update(torch.cat((base[nodes],aggregate),-1)).tanh())
   upward.index_copy_(0,nodes,value)
  outside=base.new_zeros(n,d)
  for nodes in bank.downward_levels:
   parents=bank.parent[nodes]
   parent_context=outside[parents]+base[parents]+message_sum[parents]
   value=self.pyg_downward(parent_context,child_message[nodes],dg[nodes],self.down_message.weight)
   outside.index_copy_(0,nodes,self.down_norm(self.down_update(value).tanh()))
  return upward,outside

 def leaf_context_v1(self,bank,upward,outside):
  return self.leaf_context(torch.cat((upward[bank.sample_index],outside[bank.sample_index]),-1))

 def encode_trees(self,bank,*,boundary,control="aligned",interaction="tensor"):
  base,ug,dg,degree=self.prepare_tree_base_v1(bank,control=control)
  upward,outside=self.sweep_tree_state_v1(bank,base,ug,dg,degree)
  context=self.leaf_context_v1(bank,upward,outside)
  injection=self.clade_coupling(upward,context,boundary,interaction=interaction)
  second_upward,second_outside=self.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
  return self.leaf_context_v1(bank,second_upward,second_outside)

 def forward_holds(self,bank,*,boundary,control="aligned",interaction="tensor"):
  """All952 contexts then the unchanged complete native-hold bidirectional GRU/head."""
  parameter=next(self.parameters());t,h=bank.sample_index.shape
  if parameter.dtype!=torch.float32 or h!=952 or bank.query_column.numel()!=686 or not t or bank.parent.device!=parameter.device:
   raise ValueError("Complete native952 bank and FP32 model device required")
  for name in _FLOAT_FIELDS:
   value=getattr(bank,name)
   if value.dtype!=torch.float32 or value.device!=parameter.device:raise ValueError("Original floating observations/device differ")
  for f in fields(bank):
   value=getattr(bank,f.name);values=value if f.name in _LEVEL_FIELDS else (value,)
   if any(v.device!=parameter.device for v in values):raise ValueError("Complete observation bank device differs")
  context=self.encode_trees(bank,boundary=boundary,control=control,interaction=interaction)
  span=bank.spans.log1p()/math.log(2000001.)
  prev=torch.cat((bank.spans.new_zeros(1),.5*(bank.spans[1:]+bank.spans[:-1])))
  nxt=torch.cat((.5*(bank.spans[1:]+bank.spans[:-1]),bank.spans.new_zeros(1)))
  geometry=torch.stack((span,prev.log1p()/math.log(2000001.),nxt.log1p()/math.log(2000001.)),-1)
  sequence=torch.cat((context,geometry[:,None].expand(t,h,3)),-1)
  history,_=self.genomic_gru(sequence)
  return self.classifier(torch.cat((context,history),-1))

 def forward_train_holds(self,bank,*,boundary,control="aligned",interaction="tensor"):
  return self.forward_holds(bank,boundary=boundary,control=control,interaction=interaction).index_select(1,bank.query_column)

 def forward(self,bank,*,boundary,control="aligned",interaction="tensor"):
  return self.forward_train_holds(bank,boundary=boundary,control=control,interaction=interaction).index_select(0,bank.readout_hold)


def load_h038_warmstart_v1(model,payload):
 """Root supplies the actual selected aligned model payload; no predictions/truth loader."""
 if (not isinstance(model,AnonymousCladeTensorGNNV1) or payload.get("arm")!="aligned"
  or payload.get("seed")!=101 or payload.get("teacher_used") is not False):
  raise ValueError("Same actual selected alignedH038/seed101/teacher-free payload required")
 state=payload["model_state"]
 if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 for v in state.values()):
  raise ValueError("Original FP32 learned backbone weights required")
 result=model.load_state_dict(state,strict=False)
 if set(result.missing_keys)!=NEW_KEYS or result.unexpected_keys:
  raise ValueError("Only four new bias-free rank8 projections may be missing")
 if sum(p.numel() for p in model.parameters())!=30788:raise ValueError("29764+1024 learned weights required")
 if bool(model.clade_coupling.O.weight.detach().count_nonzero()):raise ValueError("ZeroO initial identity required")
 return dict(base_parameter_count=29764,added_parameter_count=1024,total_parameter_count=30788,
  output_projection_initialized_zero=True,first_update_new_core_projection_gradients_zero_until_O_moves=True,
  optimizer_policy="reset_moments_and_step_identically_for_registered_arms")


def build_matched_family_v1(payload,model_kwargs,*,device,execution_guard,include_pairwise=False):
 """Root-bound matched initial states and fresh original AdamW, no target readers."""
 if not callable(execution_guard):raise ValueError("Root resource/execution guard required")
 if model_kwargs!=dict(hidden_dim=32,expected_haplotypes=952,site_chunk_size=256,checkpoint_sites=True):
  raise ValueError("Exact original width32/all952 H038 model settings required")
 execution_guard();torch.manual_seed(101)
 prototype=AnonymousCladeTensorGNNV1(**model_kwargs);proof=load_h038_warmstart_v1(prototype,payload)
 initial={k:v.detach().cpu().clone() for k,v in prototype.state_dict().items()};del prototype
 arms=MAIN_ARMS+OPTIONAL_ARMS if include_pairwise else MAIN_ARMS
 family={}
 for arm in arms:
  execution_guard();model=AnonymousCladeTensorGNNV1(**model_kwargs)
  model.load_state_dict(initial,strict=True);family[arm]=model.to(device=device,dtype=torch.float32)
 optimizers={arm:torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
  for arm,model in family.items()}
 return family,optimizers,dict(proof,arms=list(arms),seed=101,
  same_backbone_and_all_four_projection_initial_weights=True,optimizer_moments_and_step_reset_for_all=True)


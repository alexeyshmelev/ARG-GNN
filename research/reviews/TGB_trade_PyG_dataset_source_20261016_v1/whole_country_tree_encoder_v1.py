"""Generic full-tree Torch/PyG encoder; no temporal scan, head, labels or fit.

Country witness IDs are returned as metadata and are never embedded. All leaf
profiles, flags, staleness and retained internal hierarchy enter message passing.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch_geometric.nn.aggr import SumAggregation
from .trade_tree_core_v1 import require

@dataclass(frozen=True)
class FullTreeScheduleV1:
    node_count:int
    up_nodes:tuple
    up_children:tuple
    up_parent_rank:tuple
    down_nodes:tuple
    down_parents:tuple

    def to(self,device):
        move=lambda values:tuple(x.to(device) for x in values)
        return FullTreeScheduleV1(self.node_count,move(self.up_nodes),move(self.up_children),
            move(self.up_parent_rank),move(self.down_nodes),move(self.down_parents))

def make_full_tree_schedule_v1(graph):
    """Prepare once on CPU after PyG Batch collation, before device transfer."""
    require(graph.edge_index.device.type=="cpu" and graph.upward_level.device.type=="cpu"
        and graph.downward_level.device.type=="cpu","Prepare batch schedule on CPU")
    n=graph.num_nodes;child,parent=graph.edge_index
    require(child.numel()==n-graph.root_node_index.numel(),"One edge per nonroot across complete trees")
    parent_of=torch.full((n,),-1,dtype=torch.long);parent_of[child]=parent
    up_nodes=[];up_children=[];up_rank=[]
    for level in range(int(graph.upward_level.max())+1):
        nodes=(graph.upward_level==level).nonzero(as_tuple=False).flatten()
        edges=graph.upward_level.index_select(0,parent)==level
        children=child[edges];parents=parent[edges]
        rank=torch.full((n,),-1,dtype=torch.long);rank[nodes]=torch.arange(nodes.numel())
        require(bool((graph.upward_level[children]<level).all()),"Every upward child is already computed")
        up_nodes.append(nodes);up_children.append(children);up_rank.append(rank[parents])
    down_nodes=[];down_parents=[]
    for level in range(int(graph.downward_level.max())+1):
        nodes=(graph.downward_level==level).nonzero(as_tuple=False).flatten()
        parents=parent_of[nodes]
        require(level==0 or bool((parents>=0).all()),"Only roots lack a parent")
        require(level==0 or bool((graph.downward_level[parents]==level-1).all()),"Every downward parent is already computed")
        down_nodes.append(nodes);down_parents.append(parents)
    return FullTreeScheduleV1(n,tuple(up_nodes),tuple(up_children),tuple(up_rank),
        tuple(down_nodes),tuple(down_parents))

@dataclass(frozen=True)
class CountryEncodingV1:
    node_state:torch.Tensor
    leaf_state:torch.Tensor
    country_witness_index:torch.Tensor
    leaf_graph_index:torch.Tensor

class WholeCountryTreeEncoderV1(nn.Module):
    """Shared per-node maps and degree-normalized full-tree up/down aggregation."""
    def __init__(self,*,num_destinations=255,hidden_channels=64):
        super().__init__()
        require(type(num_destinations) is int and num_destinations>=2 and
            type(hidden_channels) is int and hidden_channels>0,"Explicit encoder dimensions")
        self.num_destinations=num_destinations;self.hidden_channels=hidden_channels
        mlp=lambda before:nn.Sequential(nn.Linear(before,hidden_channels),nn.SiLU(),
            nn.Linear(hidden_channels,hidden_channels))
        self.leaf_projection=mlp(num_destinations+3)
        self.structure_projection=mlp(4)
        self.up_update=mlp(2*hidden_channels)
        self.down_update=mlp(3*hidden_channels)
        self.output_projection=mlp(2*hidden_channels)
        self.sum_aggregation=SumAggregation()

    def forward(self,graph,*,schedule):
        require(isinstance(schedule,FullTreeScheduleV1) and schedule.node_count==graph.num_nodes,
            "Schedule for this complete graph/batch")
        require(graph.leaf_profile.ndim==2 and graph.leaf_profile.shape[1]==self.num_destinations,
            "Fixed destination feature axis")
        dtype=self.leaf_projection[0].weight.dtype
        profile=graph.leaf_profile.to(dtype=dtype)
        leaf_features=torch.cat((profile,graph.leaf_missing[:,None].to(dtype),
            graph.leaf_zero_mass[:,None].to(dtype),
            torch.log1p(graph.leaf_staleness_years[:,None].to(dtype))),dim=1)
        structure=torch.stack((graph.node_merge_height.to(dtype),
            torch.log1p(graph.node_leaf_count.to(dtype))/math.log1p(self.num_destinations),
            graph.node_child_count.to(dtype)/self.num_destinations,
            graph.node_is_leaf.to(dtype)),dim=1)
        seed=self.structure_projection(structure).index_add(0,graph.leaf_node_index,
            self.leaf_projection(leaf_features))
        up=seed.new_zeros(seed.shape)
        for nodes,children,rank in zip(schedule.up_nodes,schedule.up_children,schedule.up_parent_rank):
            aggregated=self.sum_aggregation(up.index_select(0,children),index=rank,dim_size=nodes.numel())
            degree=graph.node_child_count.index_select(0,nodes).clamp_min(1).to(dtype)[:,None]
            value=self.up_update(torch.cat((seed.index_select(0,nodes),aggregated/degree),dim=1))
            up=up.index_copy(0,nodes,value)
        down=seed.new_zeros(seed.shape)
        for level,(nodes,parents) in enumerate(zip(schedule.down_nodes,schedule.down_parents)):
            if level==0:
                parent_down=seed.new_zeros((nodes.numel(),self.hidden_channels));parent_up=parent_down
            else:
                parent_down=down.index_select(0,parents);parent_up=up.index_select(0,parents)
            value=self.down_update(torch.cat((up.index_select(0,nodes),parent_down,parent_up),dim=1))
            down=down.index_copy(0,nodes,value)
        state=self.output_projection(torch.cat((up,down),dim=1))
        batch=getattr(graph,"batch",None)
        leaf_batch=(torch.zeros(graph.leaf_node_index.numel(),dtype=torch.long,device=state.device)
            if batch is None else batch.index_select(0,graph.leaf_node_index))
        return CountryEncodingV1(state,state.index_select(0,graph.leaf_node_index),
            graph.country_witness_index,leaf_batch)

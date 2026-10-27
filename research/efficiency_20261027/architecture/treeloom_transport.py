"""TreeLoom-Transport: isolated changed-model candidate, never a resume path.

Preserves all supplied native nodes, edges, SNPs, sampled identities, reference
roles and coordinates. Full-tree path-product transport replaces per-depth
nonlinear cavity updates. Two nonlinear rounds interleave observed-allele tree
reasoning with sample-aligned, distance-aware associative genomic scans.

Topology is observation-only. Parent node IDs route computation, not embeddings.
No loader, target resolver, optimizer, checkpoint writer, or GPU launcher exists
in this module. The reference pointer-jumping schedule is not a speed claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from research.scalability_20261024.categorical_site.categorical_site import categorical_encode_sites


@dataclass(frozen=True)
class TransportPlan:
    parent: torch.Tensor
    jumps: tuple
    node_flags: torch.Tensor

    def to(self, device):
        return TransportPlan(self.parent.to(device),tuple(tuple(x.to(device) for x in stage) for stage in self.jumps),self.node_flags.to(device))


def prepare_transport_plan(parent, node_flags):
    """One CPU topology preparation, outside every differentiable forward."""
    if parent.device.type != 'cpu' or parent.dtype != torch.long:
        raise ValueError('CPU int64 topology required')
    n=parent.numel()
    if not n or node_flags.shape!=(n,2) or node_flags.device.type!='cpu':
        raise ValueError('Complete original node flags required')
    if bool(((parent < -1)|(parent >= n)).any()):
        raise ValueError('Parent index outside this forest')
    p=parent.clone();stages=[]
    for _ in range(n.bit_length()+1):
        ids=(p>=0).nonzero(as_tuple=True)[0]
        if not len(ids):
            return TransportPlan(parent,tuple(stages),node_flags.float())
        edge_index=torch.stack((ids,p[ids]))
        stages.append((p,ids,edge_index))
        p=torch.where(p>=0,p[p.clamp_min(0)],-torch.ones_like(p))
    raise ValueError('Cycle in parent routing')


class DiagonalTransport(MessagePassing):
    """Genuine PyG aggregation on every precomputed ancestor-jump edge."""
    def __init__(self):
        super().__init__(aggr='add',node_dim=0)

    def message(self,x_j,decay):
        return x_j*decay

    def kernel(self,values,decay,plan):
        up=values;a=decay;n=len(values)
        for parent,ids,edges in plan.jumps:
            up=up+self.propagate(edges,x=up,decay=a[ids],size=(n,n))
            a=a*a[parent.clamp_min(0)]*(parent>=0)[:,None]
        parent=plan.parent;mask=(parent>=0)[:,None]
        down=decay*(up[parent.clamp_min(0)]-decay*up)*mask
        a=decay
        for parent,ids,edges in plan.jumps:
            mask=(parent>=0)[:,None]
            down=down+a*down[parent.clamp_min(0)]*mask
            a=a*a[parent.clamp_min(0)]*mask
        return up+down

    def forward(self,values,decay,plan):
        both=torch.cat((values,torch.ones_like(values)),dim=-1)
        filtered=self.kernel(both,torch.cat((decay,decay),dim=-1),plan)
        numerator,denominator=filtered.chunk(2,dim=-1)
        return numerator/denominator


def affine_scan(alpha,beta,initial=None):
    """No state-dependent coefficient: exact associative composition is valid."""
    a,b=alpha,beta;offset=1
    while offset<len(beta):
        a0,b0=a,b
        a=torch.cat((a0[:offset],a0[offset:]*a0[:-offset]),dim=0)
        b=torch.cat((b0[:offset],b0[offset:]+a0[offset:]*b0[:-offset]),dim=0)
        offset*=2
    return b if initial is None else b+a*initial


class TransportRound(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.values=nn.Linear(d,d)
        self.tree_update=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d))
        self.tree_norm=nn.LayerNorm(d)
        self.force=nn.Linear(d+3,d)
        self.rate=nn.Linear(d+3,d)
        # Multiple genomic scales are an initialization, never truth-derived.
        scales=torch.logspace(math.log10(100.),math.log10(100000.),d)
        self.log_rate=nn.Parameter(-scales.log())
        self.merge=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,d))
        self.leaf_norm=nn.LayerNorm(d)
        self.transport=DiagonalTransport()

    def forward(self,nodes,decay,plan,samples,spans,geometry):
        filtered=self.transport(self.values(nodes),decay,plan)
        nodes=self.tree_norm(nodes+self.tree_update(torch.cat((nodes,filtered),dim=-1)))
        leaf=nodes[samples]
        features=torch.cat((leaf,geometry[:,None,:].expand(-1,leaf.shape[1],-1)),dim=-1)
        rate=torch.exp(self.log_rate+self.rate(features).tanh())
        exponent=-spans[:,None,None]*rate
        alpha=torch.exp(exponent)
        beta=-torch.expm1(exponent)*self.force(features).tanh()
        left=affine_scan(alpha,beta)
        right=affine_scan(alpha.flip(0),beta.flip(0)).flip(0)
        history=self.leaf_norm(leaf+self.merge(torch.cat((leaf,left,right),dim=-1)))
        # Persistent sample columns define the sole genomic identity route.
        delta=(history-leaf).reshape(-1,leaf.shape[-1])
        nodes=nodes.index_add(0,samples.reshape(-1),delta)
        return nodes,leaf,history


class TreeLoomTransport(nn.Module):
    def __init__(self,hidden_dim=32,num_classes=3,rounds=2,site_chunk_size=256):
        super().__init__()
        self.hidden_dim=hidden_dim;self.num_classes=num_classes
        self.site_chunk_size=site_chunk_size;self.donor_channels=2*num_classes
        d=hidden_dim
        self.site_mlp=nn.Sequential(nn.Linear(2*self.donor_channels+3,d),nn.GELU(),nn.Linear(d,d),nn.Tanh())
        self.site_pool_projection=nn.Sequential(nn.Linear(2*d+3,d),nn.Tanh())
        self.leaf_projection=nn.Sequential(nn.Linear(d+num_classes+2,d),nn.Tanh())
        self.node_projection=nn.Sequential(nn.Linear(5,d),nn.Tanh())
        self.edge_rate=nn.Linear(9,d)
        genealogy_scales=torch.logspace(math.log10(50.),math.log10(50000.),d)
        self.log_tree_rate=nn.Parameter(-genealogy_scales.log())
        self.rounds=nn.ModuleList(TransportRound(d) for _ in range(rounds))
        self.classifier=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,num_classes))

    def forward_all_samples(self,bank,plan):
        site_leaf=categorical_encode_sites(self,bank)
        t,h=bank.sample_index.shape;n=len(bank.parent)
        labels=bank.reference_class
        reference=F.one_hot(labels.clamp_min(0),self.num_classes).to(site_leaf.dtype)*(labels>=0)[:,None]
        role=torch.cat((reference,(labels<0).to(site_leaf.dtype)[:,None],(bank.sample_age.log1p()/math.log(10001.))[:,None]),dim=-1)
        leaf=self.leaf_projection(torch.cat((site_leaf,role[None].expand(t,h,-1)),dim=-1))
        sampled=bank.node_time.new_zeros(n).index_fill(0,bank.sample_index.reshape(-1),1.)
        node_features=torch.cat((torch.stack((bank.node_time.log1p()/math.log(10001.),(bank.parent<0).to(sampled.dtype),sampled),dim=-1),plan.node_flags),dim=-1)
        nodes=self.node_projection(node_features).index_add(0,bank.sample_index.reshape(-1),leaf.reshape(-1,self.hidden_dim))
        raw=bank.branch_support_true
        support=torch.cat((raw[:,:3],raw[:,3:4].log1p()/math.log(100001.),raw[:,4:5]),dim=-1)
        edge_features=torch.cat((bank.edge_geometry,support),dim=-1)
        # The stored duration feature was computed from positive float64 native
        # endpoint differences. Do not subtract potentially tied float32 clocks.
        duration=torch.expm1(bank.edge_geometry[:,2]*math.log(10001.))
        decay=torch.exp(-duration[:,None]*torch.exp(self.log_tree_rate+self.edge_rate(edge_features).tanh()))
        previous=torch.cat((bank.spans.new_zeros(1),.5*(bank.spans[1:]+bank.spans[:-1])))
        following=torch.cat((.5*(bank.spans[1:]+bank.spans[:-1]),bank.spans.new_zeros(1)))
        geometry=torch.stack((bank.spans.log1p(),previous.log1p(),following.log1p()),dim=-1)/math.log(2000001.)
        for layer in self.rounds:
            nodes,tree_context,history=layer(nodes,decay,plan,bank.sample_index,bank.spans,geometry)
        return self.classifier(torch.cat((tree_context,history),dim=-1))

    def forward(self,bank,plan):
        """All native intervals and actual query columns: [T,Q,C]."""
        return self.forward_all_samples(bank,plan).index_select(1,bank.query_column)

    def focal_logits(self,bank,plan,focal_hold,focal_sample_column):
        """Original frozen registry uses full-panel sample columns, not query rank."""
        return self.forward_all_samples(bank,plan)[focal_hold,focal_sample_column]

    def contract(self):
        return dict(family='TreeLoom-Transport-research-v1',hidden_dim=self.hidden_dim,num_classes=self.num_classes,rounds=len(self.rounds),parameters=sum(p.numel() for p in self.parameters()),equivalent_to_frozen_TreeLoom=False,full_tree_filter='normalized_channelwise_path_product_with_PyG_jump_aggregation',genomic_operator='two_directional_input_selective_affine_distance_scan',early_allele_input=True,all_supplied_nodes_edges_sites_samples_retained=True,native_hold_readout=True,internal_node_ID_embeddings=False,trained=False)

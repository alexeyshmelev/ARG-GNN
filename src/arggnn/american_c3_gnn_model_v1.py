"""Fresh generic-class whole-tree PyG classifier and exact directional GRU scans.

Graph equations are copied unchanged from the existing PyG cavity encoder,
except C-dependent observed-leaf width. All parameters initialize afresh.
"""
from __future__ import annotations
import math
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .american_c3_common_v1 import require,CLASSES
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeGNNV1
from .mutation_routed_pyg_gnn_v1 import _GatedUpwardMessagePassingV1,_ExcludedCavityMessagePassingV1


class AmericanC3WholeTreeGNNV1(nn.Module):
    def __init__(self,hidden_dim=32,*,class_order=CLASSES,expected_haplotypes=None,
            donor_cohorts=2,site_chunk_size=256,checkpoint_sites=True):
        super().__init__()
        self.class_order=tuple(class_order);self.num_classes=len(self.class_order)
        require(self.num_classes>=2 and len(set(self.class_order))==self.num_classes
            and all(isinstance(x,str) and x for x in self.class_order),'Explicit generic class order required')
        require(all(type(x) is int and x>0 for x in (hidden_dim,donor_cohorts,site_chunk_size))
            and donor_cohorts in (1,2) and (expected_haplotypes is None or expected_haplotypes>0),'Positive explicit model cardinalities required')
        self.hidden_dim=hidden_dim;self.expected_haplotypes=expected_haplotypes
        self.donor_cohorts=donor_cohorts;self.donor_channels=self.num_classes*donor_cohorts
        self.site_chunk_size=site_chunk_size;self.checkpoint_sites=checkpoint_sites
        d=hidden_dim
        self.site_mlp=nn.Sequential(nn.Linear(2*self.donor_channels+3,d),nn.GELU(),nn.Linear(d,d),nn.Tanh())
        self.site_pool_projection=nn.Sequential(nn.Linear(2*d+3,d),nn.Tanh())
        self.leaf_projection=nn.Sequential(nn.Linear(d+self.num_classes+2,d),nn.Tanh())
        self.node_projection=nn.Sequential(nn.Linear(3,d),nn.Tanh())
        self.up_gate=nn.Linear(9,d);self.down_gate=nn.Linear(9,d)
        self.up_message=nn.Linear(d,d,bias=False);self.up_update=nn.Linear(2*d,d);self.up_norm=nn.LayerNorm(d)
        self.down_message=nn.Linear(d,d,bias=False);self.down_update=nn.Linear(d,d);self.down_norm=nn.LayerNorm(d)
        self.leaf_context=nn.Sequential(nn.Linear(2*d,d),nn.Tanh())
        # Two unidirectional modules make exact bidirectional carry explicit.
        # They are learned jointly from fresh initialization, never copied from C4.
        self.forward_gru=nn.GRU(d+3,d,num_layers=1,bidirectional=False)
        self.reverse_gru=nn.GRU(d+3,d,num_layers=1,bidirectional=False)
        self.classifier=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,self.num_classes))
        self.pyg_upward=_GatedUpwardMessagePassingV1();self.pyg_downward=_ExcludedCavityMessagePassingV1()

    # Original mean/max/count/empty/span pooling equations, now using our generic chunk.
    encode_sites=MutationRoutedWholeTreeGNNV1.encode_sites

    def _site_chunk(self,genotype,donor,donor_mask,panel,entropy,inverse,count_holds):
        h,d=genotype.shape[1],self.hidden_dim;channels=self.donor_channels
        called=genotype>=0;chosen=genotype.long().clamp_min(0)
        logfreq=donor.gather(1,chosen[...,None].expand(-1,-1,channels))
        available=donor_mask[:,None,:].expand(-1,h,-1)&called[...,None]
        logfreq=torch.where(available,logfreq,torch.zeros_like(logfreq))
        panel_chosen=panel.gather(1,chosen);panel_log=torch.where(called,panel_chosen,torch.ones_like(panel_chosen)).log()
        ent=entropy[:,None].expand(-1,h)
        features=torch.cat((logfreq,available.to(logfreq.dtype),panel_log[...,None],ent[...,None],called[...,None].to(logfreq.dtype)),-1)
        features=torch.where(called[...,None],features,torch.zeros_like(features));encoded=self.site_mlp(features)
        index=(inverse[:,None]*h+torch.arange(h,device=genotype.device)[None]).flatten()
        values=torch.where(called[...,None],encoded,torch.zeros_like(encoded)).reshape(-1,d)
        sums=values.new_zeros(count_holds*h,d).index_add(0,index,values)
        counts=values.new_zeros(count_holds*h,1).index_add(0,index,called.to(values.dtype).reshape(-1,1))
        max_values=torch.where(called[...,None],encoded,torch.full_like(encoded,-torch.inf)).reshape(-1,d)
        maxima=values.new_full((count_holds*h,d),-torch.inf).scatter_reduce(0,index[:,None].expand_as(max_values),max_values,reduce='amax',include_self=True)
        maxima=torch.where(counts>0,maxima,torch.zeros_like(maxima))
        return sums.reshape(count_holds,h,d),maxima.reshape(count_holds,h,d),counts.reshape(count_holds,h,1)

    def encode_trees(self, bank: MutationRoutedWholeTreeInputsV1, *, control):
        t, h = bank.sample_index.shape
        n, d = len(bank.parent), self.hidden_dim
        site_leaf = self.encode_sites(bank)
        labels = bank.reference_class
        onehot = F.one_hot(labels.clamp_min(0), self.num_classes).to(site_leaf.dtype)
        onehot = onehot * (labels >= 0)[:, None]
        unknown = (labels < 0).to(site_leaf.dtype)[:, None]
        age = (bank.sample_age.log1p()/math.log(10001.))[:, None]
        observed_leaf = torch.cat((onehot, unknown, age), -1)[None].expand(t, h, self.num_classes+2)
        leaf_base = self.leaf_projection(torch.cat((site_leaf, observed_leaf), -1))
        sampled = bank.node_time.new_zeros(n)
        sampled.index_fill_(0, bank.sample_index.flatten(), 1.)
        root = (bank.parent < 0).to(sampled.dtype)
        node_age = bank.node_time.log1p()/math.log(10001.)
        base = self.node_projection(torch.stack((node_age, root, sampled), -1))
        # This is unique sampled-leaf feature injection, not graph messaging.
        base = base.index_add(0, bank.sample_index.flatten(), leaf_base.reshape(-1, d))
        raw = bank.branch_support_true if control == "aligned" else bank.branch_support_sham
        support = torch.cat((raw[:, :3], raw[:, 3:4].log1p()/math.log(100001.), raw[:, 4:5]), -1)
        gate_input = torch.cat((bank.edge_geometry, support), -1)
        ug, dg = self.up_gate(gate_input).sigmoid(), self.down_gate(gate_input).sigmoid()
        nonroot = bank.parent >= 0
        degree = torch.bincount(bank.parent[nonroot], minlength=n).to(base.dtype).clamp_min(1)
        upward = base.new_zeros(n, d)
        message_sum = base.new_zeros(n, d)
        child_message = base.new_zeros(n, d)
        for nodes, children, ranks in zip(
                bank.upward_levels, bank.upward_children, bank.upward_parent_rank):
            if children.numel():
                aggregate, message = self.pyg_upward(
                    upward[children], ug[children],
                    degree[bank.parent[children]], ranks, len(nodes),
                    self.up_message.weight,
                )
                child_message.index_copy_(0, children, message)
            else:
                aggregate = base.new_zeros(len(nodes), d)
            message_sum.index_copy_(0, nodes, aggregate)
            value = self.up_norm(
                base[nodes] +
                self.up_update(torch.cat((base[nodes], aggregate), -1)).tanh()
            )
            upward.index_copy_(0, nodes, value)
        outside = base.new_zeros(n, d)
        for nodes in bank.downward_levels:
            parents = bank.parent[nodes]
            parent_context = outside[parents] + base[parents] + message_sum[parents]
            value = self.pyg_downward(
                parent_context, child_message[nodes], dg[nodes],
                self.down_message.weight,
            )
            value = self.down_norm(self.down_update(value).tanh())
            outside.index_copy_(0, nodes, value)
        return self.leaf_context(torch.cat((
            upward[bank.sample_index], outside[bank.sample_index],
        ), -1))

    def encode_bank(self,bank):
        t,h=bank.sample_index.shape;p=next(self.parameters())
        require(p.dtype==torch.float32 and bank.parent.device==p.device and t>0
            and (self.expected_haplotypes is None or h==self.expected_haplotypes)
            and bank.log_donor_allele_frequency.shape[-1]==self.donor_channels,
            'Exact fresh model/bank dtype, device, class channels and full sample cardinality required')
        return self.encode_trees(bank,control='aligned')

    def scan_context(self,context,geometry,*,direction,hidden=None):
        require(direction in ('forward','reverse') and context.ndim==3
            and geometry.shape==(context.shape[0],3),'Declared direction and complete global-neighbor geometry required')
        sequence=torch.cat((context,geometry[:,None,:].expand(-1,context.shape[1],-1)),-1)
        if direction=='reverse':sequence=sequence.flip(0)
        module=self.forward_gru if direction=='forward' else self.reverse_gru
        history,state=module(sequence,hidden)
        return (history.flip(0) if direction=='reverse' else history),state

    def classify_context(self,context,forward,reverse):
        return self.classifier(torch.cat((context,forward,reverse),-1))

    def forward_complete(self,block):
        """Full recurrence only when a complete independent parent fits one block."""
        require(block.complete_parent is True,'No implicit recurrent reset at a training block boundary')
        context=self.encode_bank(block.to_model_inputs())
        forward,_=self.scan_context(context,block.genomic_geometry,direction='forward')
        reverse,_=self.scan_context(context,block.genomic_geometry,direction='reverse')
        return self.classify_context(context,forward,reverse)

    def forward(self,block):
        logits=self.forward_complete(block)
        return logits[block.focal_hold,block.focal_sample_column]

    def checkpoint_contract(self):
        return dict(schema='american_c3_fresh_whole_tree_GNN_checkpoint_v1',class_order=list(self.class_order),
            hidden_dim=self.hidden_dim,donor_cohorts=self.donor_cohorts,donor_channels=self.donor_channels,
            expected_haplotypes=self.expected_haplotypes,initialization='fresh_no_C4_loading',
            site_feature_width=2*self.donor_channels+3,observed_leaf_width=self.num_classes+2,
            recurrent_contract='one_layer_bidirectional_equations_as_two_directional_GRUs',
            tree_backbone='unchanged_PyG_gated_upward_recipient_excluded_downward_cavity',
            whole_parent_training_required=True)

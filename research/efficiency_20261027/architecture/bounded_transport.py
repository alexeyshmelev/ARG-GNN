"""Checkpointed whole-parent TreeLoom-Transport execution.

Native trees are partitioned only for computation. Complete stable-sample
histories live on CPU, and both affine directions carry across every block.
No data reader, target resolver, GPU admission, optimizer or launcher is here.
"""
from functools import partial
from types import SimpleNamespace
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from research.scalability_20261024.integration.executors import _IntegratedLayoutAndSites
from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import geometry_from_spans
from .treeloom_transport import prepare_transport_plan,affine_scan


def need(condition,message):
    if not condition:raise ValueError(message)


class BoundedTransport:
    # The qualified implementation preserves original global256 site chunks,
    # categorical encoding, pooling order and all original full-panel alleles.
    _sites=_IntegratedLayoutAndSites._sites

    def __init__(self,model,bank,node_flags,native_nodes_per_tree,*,holds_per_block=16,
                 haplotypes_per_tile=16,scan_holds_per_block=256,resource_check=lambda stage:None):
        need(model.hidden_dim==32 and model.num_classes==3 and len(model.rounds)==2 and model.site_chunk_size==256,
             'This qualified executor implements exactly the d32 two-round candidate')
        need(bank.parent.device.type=='cpu' and bank.sample_index.shape[1]==1600,
             'Complete original CPU observation bank and all1600 samples required')
        need(node_flags.shape==(len(bank.parent),2) and node_flags.device.type=='cpu',
             'Original globally prepared local clade flags required')
        need(1<=holds_per_block<=128 and 1<=haplotypes_per_tile<=128 and 1<=scan_holds_per_block<=1024,
             'Explicit bounded compute partitions required')
        self.model,self.bank,self.tick=model,bank,resource_check
        self.node_flags=node_flags
        self.device=next(model.parameters()).device
        self.hap_tile=haplotypes_per_tile;self.scan_block=scan_holds_per_block
        self.geometry=geometry_from_spans(bank.spans)
        counts=tuple(int(x) for x in native_nodes_per_tree)
        need(len(counts)==len(bank.spans) and all(x>=1600 for x in counts) and sum(counts)==len(bank.parent),
             'Exact complete native node partition required')
        offsets=np.r_[0,np.cumsum(counts,dtype=np.int64)]
        self.blocks=[]
        for a in range(0,len(counts),holds_per_block):
            b=min(a+holds_per_block,len(counts));lo,hi=int(offsets[a]),int(offsets[b])
            p=bank.parent[lo:hi].clone();nonroot=p>=0
            need(bool(((p[nonroot]>=lo)&(p[nonroot]<hi)).all()) and
                 bool(((bank.sample_index[a:b]>=lo)&(bank.sample_index[a:b]<hi)).all()),
                 'No node, edge or sample may cross a native compute partition')
            self.blocks.append((a,b,lo,hi))
        self.tick('transport_complete_CPU_layout_ready')

    def _cp(self,fn,*args):
        if not torch.is_grad_enabled():return fn(*args)
        return checkpoint(fn,*args,use_reentrant=False,preserve_rng_state=True)

    def _block(self,z):
        a,b,lo,hi=z;bank=self.bank
        # Keep only one block's jump indexes, even on200M-node parents.
        # This unoptimized reference rebuilds topology during recomputation.
        parent=bank.parent[lo:hi].clone();parent[parent>=0]-=lo
        plan=prepare_transport_plan(parent,self.node_flags[lo:hi])
        return SimpleNamespace(parent=plan.parent.to(self.device),
            node_time=bank.node_time[lo:hi].to(self.device),
            sample_index=(bank.sample_index[a:b]-lo).to(self.device),
            edge_geometry=bank.edge_geometry[lo:hi].to(self.device),
            branch_support_true=bank.branch_support_true[lo:hi].to(self.device),
            reference_class=bank.reference_class.to(self.device),sample_age=bank.sample_age.to(self.device)),plan.to(self.device)

    def _initial(self,z):
        bank,plan=self._block(z);m=self.model;site_leaf=self._sites(z[0],z[1])
        t,h=bank.sample_index.shape;n=len(bank.parent)
        labels=bank.reference_class
        reference=F.one_hot(labels.clamp_min(0),m.num_classes).to(site_leaf.dtype)*(labels>=0)[:,None]
        role=torch.cat((reference,(labels<0).to(site_leaf.dtype)[:,None],
                        (bank.sample_age.log1p()/math.log(10001.))[:,None]),dim=-1)
        leaf=m.leaf_projection(torch.cat((site_leaf,role[None].expand(t,h,-1)),dim=-1))
        sampled=bank.node_time.new_zeros(n).index_fill(0,bank.sample_index.reshape(-1),1.)
        features=torch.cat((torch.stack((bank.node_time.log1p()/math.log(10001.),
                            (bank.parent<0).to(sampled.dtype),sampled),dim=-1),plan.node_flags),dim=-1)
        nodes=m.node_projection(features).index_add(0,bank.sample_index.reshape(-1),leaf.reshape(-1,m.hidden_dim))
        raw=bank.branch_support_true
        support=torch.cat((raw[:,:3],raw[:,3:4].log1p()/math.log(100001.),raw[:,4:5]),dim=-1)
        edge_features=torch.cat((bank.edge_geometry,support),dim=-1)
        duration=torch.expm1(bank.edge_geometry[:,2]*math.log(10001.))
        decay=torch.exp(-duration[:,None]*torch.exp(m.log_tree_rate+m.edge_rate(edge_features).tanh()))
        return bank,plan,nodes,decay

    def _tree(self,layer,nodes,decay,plan):
        filtered=layer.transport(layer.values(nodes),decay,plan)
        return layer.tree_norm(nodes+layer.tree_update(torch.cat((nodes,filtered),dim=-1)))

    def _first(self,z):
        self.tick('transport_first_tree_block')
        bank,plan,nodes,decay=self._initial(z)
        nodes=self._tree(self.model.rounds[0],nodes,decay,plan)
        result=nodes[bank.sample_index].to('cpu')
        self.tick('transport_first_tree_block_complete')
        return result

    def _second(self,z,history):
        self.tick('transport_second_tree_block')
        bank,plan,nodes,decay=self._initial(z)
        nodes=self._tree(self.model.rounds[0],nodes,decay,plan)
        leaf=nodes[bank.sample_index]
        delta=(history.to(self.device)-leaf).reshape(-1,self.model.hidden_dim)
        nodes=nodes.index_add(0,bank.sample_index.reshape(-1),delta)
        nodes=self._tree(self.model.rounds[1],nodes,decay,plan)
        result=nodes[bank.sample_index].to('cpu')
        self.tick('transport_second_tree_block_complete')
        return result

    def _scan_piece(self,layer,a,b,reverse,leaf,carry):
        self.tick('transport_affine_scan_block')
        leaf=leaf.to(self.device);geometry=self.geometry[a:b].to(self.device)
        features=torch.cat((leaf,geometry[:,None,:].expand(-1,leaf.shape[1],-1)),dim=-1)
        rate=torch.exp(layer.log_rate+layer.rate(features).tanh())
        exponent=-self.bank.spans[a:b,None,None].to(self.device)*rate
        alpha=torch.exp(exponent);beta=-torch.expm1(exponent)*layer.force(features).tanh()
        if reverse:alpha,beta=alpha.flip(0),beta.flip(0)
        values=affine_scan(alpha,beta,initial=carry.to(self.device))
        if reverse:values=values.flip(0)
        result=values.to('cpu')
        self.tick('transport_affine_scan_block_complete')
        return result

    def _merge_piece(self,layer,leaf,left,right):
        self.tick('transport_history_merge_block')
        leaf=leaf.to(self.device)
        value=layer.leaf_norm(leaf+layer.merge(torch.cat((leaf,left.to(self.device),right.to(self.device)),dim=-1)))
        return value.to('cpu')

    def _history_tile(self,layer,context):
        t,h,d=context.shape
        chunks=[(a,min(a+self.scan_block,t)) for a in range(0,t,self.scan_block)]
        left=[];carry=context.new_zeros((h,d))
        for a,b in chunks:
            value=self._cp(partial(self._scan_piece,layer,a,b,False),context[a:b],carry)
            left.append(value);carry=value[-1]
        right=[None]*len(chunks);carry=context.new_zeros((h,d))
        for i in range(len(chunks)-1,-1,-1):
            a,b=chunks[i]
            value=self._cp(partial(self._scan_piece,layer,a,b,True),context[a:b],carry)
            right[i]=value;carry=value[0]
        return torch.cat([self._cp(partial(self._merge_piece,layer),context[a:b],l,r)
                          for (a,b),l,r in zip(chunks,left,right)],dim=0)

    def context(self):
        first=torch.cat([self._cp(partial(self._first,z)) for z in self.blocks],dim=0)
        histories=[self._history_tile(self.model.rounds[0],first[:,a:a+self.hap_tile])
                   for a in range(0,first.shape[1],self.hap_tile)]
        history=torch.cat(histories,dim=1)
        second=torch.cat([self._cp(partial(self._second,z),history[z[0]:z[1]]) for z in self.blocks],dim=0)
        return second

    def _classify(self,tree,history,hold,column):
        return self.model.classifier(torch.cat((tree[hold,column],history[hold,column]),dim=-1).to(self.device))

    def forward_focals(self,focal_hold,focal_sample_column):
        """Every supplied focal once; tree support still includes all1600 samples."""
        need(focal_hold.device.type=='cpu' and focal_hold.dtype==torch.long and
             focal_sample_column.device.type=='cpu' and focal_sample_column.dtype==torch.long and
             focal_hold.shape==focal_sample_column.shape and len(focal_hold)>0,
             'Complete CPU focal coordinates required')
        need(bool(((focal_hold>=0)&(focal_hold<len(self.bank.spans))).all()) and
             bool(((focal_sample_column>=0)&(focal_sample_column<1600)).all()),'Bounded original focal indexes required')
        context=self.context();result=torch.zeros((len(focal_hold),3),device=self.device)
        seen=0
        for first in range(0,1600,self.hap_tile):
            last=min(first+self.hap_tile,1600)
            rows=((focal_sample_column>=first)&(focal_sample_column<last)).nonzero(as_tuple=True)[0]
            if not len(rows):continue
            tree=context[:,first:last]
            history=self._history_tile(self.model.rounds[1],tree)
            logits=self._cp(self._classify,tree,history,focal_hold[rows],focal_sample_column[rows]-first)
            result=result.index_copy(0,rows.to(self.device),logits);seen+=len(rows)
        need(seen==len(focal_hold),'Every original focal contributes exactly once')
        self.tick('transport_complete_parent_logits')
        return result

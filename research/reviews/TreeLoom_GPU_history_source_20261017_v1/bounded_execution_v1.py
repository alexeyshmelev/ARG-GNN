"""Exact frozen TreeLoom equations with CPU observations and recomputed CUDA blocks.
No file I/O, admission, launch, state probe or checkpoint reader in this module.
"""
from types import SimpleNamespace
from functools import partial,wraps
from .stable_leaf_history_v1 import stable_leaf_history_v1,NullProfileV1

def profiled(fn):
    @wraps(fn)
    def call(self,*args,**kwargs):
        with self.profile.stage(fn.__name__):return fn(self,*args,**kwargs)
    return call

import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as frozen

def need(ok,message):
    if not ok:raise ValueError(message)

def _cp(fn,*args):
    return checkpoint(fn,*args,use_reentrant=False,preserve_rng_state=True)

class BoundedExecutionV1:
    """Hold blocks are execution partitions; all structural/history recipes are global."""
    def __init__(self,model,bundle,metadata,*,holds_per_block=16,haplotypes_per_gru=8,resource_check=lambda label:None,history_holds_per_window=128,profile=None):
        need(type(model)is frozen.AmericanTreeLoomC3FullV1 and type(bundle)is frozen.AmericanTreeLoomWholeParentV1,
             "Exact original frozen model and complete parent")
        bundle.validate()
        need(bundle.bank.parent.device.type=="cpu"and all(x.device.type=="cpu"for x in frozen._bank_tensors(bundle.bank))
             and all(x.device.type=="cpu"for x in bundle.cut.tensors()),"Full bank and recipes remain on CPU")
        need(type(holds_per_block)is int and 1<=holds_per_block<=32 and type(haplotypes_per_gru)is int
             and 1<=haplotypes_per_gru<=16 and callable(resource_check),"Bounded execution partitions")
        self.model,self.bundle,self.bank=model,bundle,bundle.bank
        self.device=next(model.parameters()).device
        need(self.device.type=="cuda"and self.device.index in(None,0),"Only logical cuda0 bound to physical GPU0")
        self.tick=resource_check;self.hap_tile=haplotypes_per_gru
        self.profile=profile or NullProfileV1();self.history_window=history_holds_per_window
        counts=metadata["episode_metadata"]["native_nodes_per_tree"]
        need(len(counts)==bundle.native_holds and all(type(n)is int and n>=1600 for n in counts)
             and sum(counts)==bundle.bank.parent.numel(),"Exact globally prepared per-native-hold node partition")
        offsets=np.concatenate(([0],np.cumsum(counts,dtype=np.int64)))
        parent=self.bank.parent.numpy();sample=self.bank.sample_index.numpy();n=len(parent)
        height=np.empty(n,dtype=np.int16);depth=np.zeros(n,dtype=np.int16)
        for k,rows in enumerate(self.bank.upward_levels):
            need(k<32767,"Bounded native depth");height[rows.numpy()]=k
        for k,rows in enumerate(self.bank.downward_levels,start=1):
            need(k<32767,"Bounded native depth");depth[rows.numpy()]=k
        self.layout=[]
        member_hold=bundle.episodes.member_hold.numpy()
        need(not len(member_hold)or np.all(member_hold[1:]>=member_hold[:-1]),"Original episode membership hold order")
        for a in range(0,len(counts),holds_per_block):
            b=min(a+holds_per_block,len(counts));lo,hi=int(offsets[a]),int(offsets[b])
            p=parent[lo:hi].copy();nonroot=p>=0
            need(np.all((p[nonroot]>=lo)&(p[nonroot]<hi))and np.all((sample[a:b]>=lo)&(sample[a:b]<hi)),
                 "Every real node and sample belongs to its original native hold block")
            p[nonroot]-=lo
            ht,dp=height[lo:hi],depth[lo:hi]
            levels=tuple(np.flatnonzero(ht==k)for k in range(int(ht.max())+1))
            rank=np.empty(hi-lo,dtype=np.int64)
            for rows in levels:rank[rows]=np.arange(len(rows))
            child=np.flatnonzero(nonroot)
            children=tuple(child[ht[p[child]]==k]for k in range(len(levels)))
            ranks=tuple(rank[p[x]]for x in children)
            downward=tuple(np.flatnonzero(dp==k)for k in range(1,int(dp.max())+1))
            ml,mr=map(int,np.searchsorted(member_hold,[a,b]))
            self.layout.append((a,b,lo,hi,p,levels,children,ranks,downward,ml,mr))
        del height,depth

    def _block(self,z):
        a,b,lo,hi,p,levels,children,ranks,downward,_,_=z;bank=self.bank
        def tensor(x,dtype=torch.long):return torch.as_tensor(x,dtype=dtype,device=self.device)
        return SimpleNamespace(parent=tensor(p),node_time=bank.node_time[lo:hi].to(self.device),
            sample_index=(bank.sample_index[a:b]-lo).to(self.device),
            upward_levels=tuple(tensor(x)for x in levels),upward_children=tuple(tensor(x)for x in children),
            upward_parent_rank=tuple(tensor(x)for x in ranks),downward_levels=tuple(tensor(x)for x in downward),
            edge_geometry=bank.edge_geometry[lo:hi].to(self.device),
            branch_support_true=bank.branch_support_true[lo:hi].to(self.device),
            reference_class=bank.reference_class.to(self.device),sample_age=bank.sample_age.to(self.device),
            spans=bank.spans[a:b].to(self.device))

    def _sites(self,a,b):
        """Original global256-site chunk boundaries, amax ties and reduction order."""
        bank=self.bank;m=self.model;h=1600;d=32
        site_hold=bank.site_hold.numpy();left,right=map(int,np.searchsorted(site_hold,[a,b]))
        hs,ss,mm,cc=[],[],[],[]
        for start in range((left//256)*256,right,256):
            end=min(start+256,len(site_hold))
            global_hold=bank.site_hold[start:end]
            holds,inverse=torch.unique_consecutive(global_hold,return_inverse=True)
            selected=torch.nonzero((holds>=a)&(holds<b),as_tuple=False).flatten()
            if not len(selected):continue
            args=tuple(x[start:end].to(self.device)for x in (bank.genotype,bank.log_donor_allele_frequency,
                bank.donor_call_mask,bank.panel_allele_frequency,bank.site_entropy))
            sums,maxima,count=m._site_chunk(*args,inverse.to(self.device),len(holds))
            ix=selected.to(self.device)
            hs.append((holds[selected]-a).to(self.device))
            ss.append(sums.index_select(0,ix));mm.append(maxima.index_select(0,ix));cc.append(count.index_select(0,ix))
        sums=bank.spans.new_zeros((b-a,h,d),device=self.device)
        maxima=bank.spans.new_full((b-a,h,d),-torch.inf,device=self.device)
        counts=bank.spans.new_zeros((b-a,h,1),device=self.device)
        if hs:
            index=torch.cat(hs);sums=sums.index_add(0,index,torch.cat(ss));counts=counts.index_add(0,index,torch.cat(cc))
            chunk_max=torch.cat(mm);chunk_count=torch.cat(cc)
            chunk_max=torch.where(chunk_count>0,chunk_max,torch.full_like(chunk_max,-torch.inf))
            maxima=maxima.scatter_reduce(0,index[:,None,None].expand_as(chunk_max),chunk_max,reduce="amax",include_self=True)
        present=counts>0;means=sums/counts.clamp_min(1)
        maxima=torch.where(present,maxima,torch.zeros_like(maxima))
        logcount=counts.log1p()/math.log(100001.);empty=(~present).to(sums.dtype)
        span=(bank.spans[a:b].to(self.device).log1p()/math.log(2000001.))[:,None,None].expand(b-a,h,1)
        return m.site_pool_projection(torch.cat((means,maxima,logcount,empty,span),-1))

    def _base(self,bank,site_leaf):
        # Exact frozen prepare_tree_base_v1 body with supplied bounded site encoding.
        m=self.model;t,h=bank.sample_index.shape;n,d=len(bank.parent),32
        labels=bank.reference_class;onehot=F.one_hot(labels.clamp_min(0),3).to(site_leaf.dtype)
        onehot=onehot*(labels>=0)[:,None];unknown=(labels<0).to(site_leaf.dtype)[:,None]
        age=(bank.sample_age.log1p()/math.log(10001.))[:,None]
        observed_leaf=torch.cat((onehot,unknown,age),-1)[None].expand(t,h,5)
        leaf_base=m.leaf_projection(torch.cat((site_leaf,observed_leaf),-1))
        sampled=bank.node_time.new_zeros(n);sampled.index_fill_(0,bank.sample_index.flatten(),1.)
        root=(bank.parent<0).to(sampled.dtype);node_age=bank.node_time.log1p()/math.log(10001.)
        base=m.node_projection(torch.stack((node_age,root,sampled),-1))
        base=base.index_add(0,bank.sample_index.flatten(),leaf_base.reshape(-1,d))
        raw=bank.branch_support_true
        support=torch.cat((raw[:,:3],raw[:,3:4].log1p()/math.log(100001.),raw[:,4:5]),-1)
        gate_input=torch.cat((bank.edge_geometry,support),-1)
        ug,dg=m.up_gate(gate_input).sigmoid(),m.down_gate(gate_input).sigmoid()
        nonroot=bank.parent>=0
        degree=torch.bincount(bank.parent[nonroot],minlength=n).to(base.dtype).clamp_min(1)
        return base,ug,dg,degree

    @profiled
    def _first(self,z):
        self.tick("bounded_first_sweep");bank=self._block(z);base,ug,dg,degree=self._base(bank,self._sites(z[0],z[1]))
        up,out=self.model.sweep_tree_state_v1(bank,base,ug,dg,degree)
        return up.to("cpu"),out.to("cpu")

    @profiled
    def _forcing(self,z,up,out):
        a,b,lo=z[:3];cut=self.bundle.cut
        first,last=cut.group_offsets[a],cut.group_offsets[b]
        ix=(cut.representatives[first:last]-lo)
        value=up.index_select(0,ix).to(self.device)+out.index_select(0,ix).to(self.device)
        return self.model.memory.U(value).to("cpu")

    def _finite_own_episode(self,upward):
        """Prove omitted 0*foreign is finite; preserve own_state exact algebra."""
        episodes=self.bundle.episodes;m=self.model.episode_coupling
        with torch.no_grad():
            weights=episodes.hold_span.index_select(0,episodes.member_hold.long())[:,None].to(self.device)
            index=episodes.member_episode.long().to(self.device)
            totals=m.aggregation(weights,index=index,dim_size=episodes.episode_size.numel(),dim=0)
            eligible=episodes.episode_size.to(self.device).index_select(0,index)>=2
            denominator=totals.index_select(0,index)-weights
            minimum=float(denominator[eligible].min())if bool(eligible.any())else 1.
            maximum=max(float(x.abs().max())for x in upward)
            need(math.isfinite(maximum)and minimum>0 and math.isfinite(minimum)
                 and 4*maximum*float(self.bank.spans.double().sum())/minimum<torch.finfo(torch.float32).max,
                 "Original span-weighted foreign context provably finite before own_state0 shortcut")
        return eligible.to("cpu")

    @profiled
    def _second(self,z,up,value,eligible):
        self.tick("bounded_second_sweep");a,b,lo,hi,_,_,_,_,_,ml,mr=z
        bank=self._block(z);base,ug,dg,degree=self._base(bank,self._sites(a,b))
        up=up.to(self.device);m=self.model
        channels=self.bundle.episodes.flags.node_flags[lo:hi].to(self.device,dtype=up.dtype)
        common=m.clade_coupling.U(up)
        flag=m.clade_coupling.O(channels[:,0:1]*common*m.clade_coupling.V(up)
            -channels[:,1:2]*common*m.clade_coupling.W(up))
        member_node=(self.bundle.episodes.member_node[ml:mr].long()-lo).to(self.device)
        member=up.index_select(0,member_node);ep=m.episode_coupling
        residual=ep.O(ep.U(member)*ep.V(member)+ep.W(member))
        residual=residual*eligible[ml:mr,None].to(self.device,dtype=torch.float32)
        episode=up.new_zeros(up.shape).index_add(0,member_node,residual)
        injection=flag+episode
        cut=self.bundle.cut;first,last=cut.group_offsets[a],cut.group_offsets[b]
        representatives=(cut.representatives[first:last]-lo).to(self.device)
        delta=m.memory.O(value.to(self.device))
        injection=injection.index_add(0,representatives,delta)
        second_up,second_out=m.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
        return m.leaf_context_v1(bank,second_up,second_out).to("cpu")

    @profiled
    def _decode(self,context,geometry,focal_hold,focal_column):
        self.tick("bounded_full_length_GRU")
        context=context.to(self.device);geometry=geometry.to(self.device)
        forward,_=self.model.scan_context(context,geometry,direction="forward")
        reverse,_=self.model.scan_context(context,geometry,direction="reverse")
        if len(focal_hold):
            hold=focal_hold.to(self.device);column=focal_column.to(self.device)
            return self.model.classifier(torch.cat((context[hold,column],forward[hold,column],reverse[hold,column]),-1)),forward.sum()*0.+reverse.sum()*0.
        return context.new_empty((0,3)),forward.sum()*0.+reverse.sum()*0.

    @profiled
    def forward(self):
        self.tick("bounded_complete_parent_begin")
        pairs=[_cp(partial(self._first,z))for z in self.layout]
        up=tuple(x[0]for x in pairs);out=tuple(x[1]for x in pairs)
        eligible=self._finite_own_episode(up)
        forces=[_cp(partial(self._forcing,z),u,o)for z,u,o in zip(self.layout,up,out)]
        force=torch.cat(forces);del forces,out,pairs
        ell=self.model.memory.ell_theta.to("cpu")
        value=stable_leaf_history_v1(self.model.memory,force,ell,self.bank.spans,self.bundle.cut,
            device=self.device,holds_per_window=self.history_window,resource_check=self.tick,profile=self.profile)
        del force
        sizes=[self.bundle.cut.group_offsets[z[1]]-self.bundle.cut.group_offsets[z[0]]for z in self.layout]
        values=value.split(sizes,dim=0)
        contexts=[_cp(partial(self._second,z),u,v,eligible)for z,u,v in zip(self.layout,up,values)]
        context=torch.cat(contexts,dim=0);del contexts,up,values,value
        spans=self.bank.spans;span=spans.log1p()/math.log(2000001.)
        prev=torch.cat((spans.new_zeros(1),.5*(spans[1:]+spans[:-1])))
        nxt=torch.cat((.5*(spans[1:]+spans[:-1]),spans.new_zeros(1)))
        geometry=torch.stack((span,prev.log1p()/math.log(2000001.),nxt.log1p()/math.log(2000001.)),-1)
        result=torch.empty((10000,3),device=self.device);zero=result.new_zeros(())
        for tile,block in enumerate(context.split(self.hap_tile,dim=1)):
            first=tile*self.hap_tile;last=first+block.shape[1]
            columns=self.bundle.focal_sample_column
            rows=torch.nonzero((columns>=first)&(columns<last),as_tuple=False).flatten()
            logits,dummy=_cp(self._decode,block,geometry,self.bundle.focal_hold[rows],columns[rows]-first)
            result=result.index_copy(0,rows.to(self.device),logits);zero=zero+dummy
        self.tick("bounded_complete_parent_logits")
        return result+zero

def train_step_v1(model,bundle,targets,optimizer,metadata,*,resource_check,holds_per_block=16,haplotypes_per_gru=8,history_holds_per_window=128,profile=None):
    need(type(model)is frozen.AmericanTreeLoomC3FullV1 and type(bundle)is frozen.AmericanTreeLoomWholeParentV1
         and bundle.split=="train"and bundle.parent_id in frozen.TRAIN,"Original complete TRAIN parent")
    need(isinstance(targets,frozen.TrainFocalTargetVectorV1)and targets.parent_id==bundle.parent_id
         and targets.focal_registry_ref==bundle.focal_registry_ref and targets.focal_identity_sha256==bundle.focal_identity_sha256,
         "Same original independently sealed10000 TRAIN focal targets")
    need(type(optimizer)is torch.optim.AdamW and len(optimizer.param_groups)==1,"Original AdamW")
    group=optimizer.param_groups[0];params=list(model.parameters())
    need(group["lr"]==.001 and group["weight_decay"]==.0001 and group["betas"]==(.9,.999)and group["eps"]==1e-8
         and group["maximize"]is False and group["amsgrad"]is False and len(params)==49
         and len(group["params"])==49 and {id(p)for p in group["params"]}=={id(p)for p in params}
         and all(p.requires_grad and p.dtype==torch.float32 and p.device.type=="cuda"for p in params),"All original49FP32 parameters")
    need(targets.labels_cpu.device.type=="cpu"and targets.labels_cpu.dtype==torch.long and targets.labels_cpu.shape==(10000,)
         and bool(((targets.labels_cpu>=0)&(targets.labels_cpu<3)).all()),"All10000 original TRAIN labels")
    bundle.validate();model.train();optimizer.zero_grad(set_to_none=True)
    profile=profile or NullProfileV1()
    executor=BoundedExecutionV1(model,bundle,metadata,holds_per_block=holds_per_block,
        haplotypes_per_gru=haplotypes_per_gru,resource_check=resource_check,history_holds_per_window=history_holds_per_window,profile=profile)
    with torch.autocast(device_type="cuda",enabled=False):
        logits=executor.forward()
        loss=F.cross_entropy(logits,targets.labels_cpu.to(logits.device),reduction="mean")
    need(bool(torch.isfinite(loss)),"Finite original all-focal mean CE")
    resource_check("bounded_complete_parent_backward")
    with profile.stage("complete_parent_backward"):loss.backward()
    grads=[p.grad for p in params]
    need(all(g is not None and bool(torch.isfinite(g).all())for g in grads)
         and any(bool((g!=0).any())for g in grads),"All49finite connected gradients and nonzero global gradient")
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    need(bool(torch.isfinite(norm)),"Finite original clip1norm")
    resource_check("bounded_complete_parent_optimizer")
    with profile.stage("complete_parent_optimizer"):optimizer.step()
    need(all(bool(torch.isfinite(p).all())for p in params),"Finite updated original49 tensors")
    resource_check("bounded_complete_parent_end")
    return dict(parent_id=bundle.parent_id,native_holds=bundle.native_holds,original_sites=bundle.original_sites,
        native_focals=10000,haplotypes=1600,sequence_length_bp=30000000,loss=float(loss.detach()),
        gradient_norm=float(norm),one_complete_parent_optimizer_update=True,
        detached_history=False,artificial_block_boundaries=False,ancestry_accuracy=None,
        execution="CPU_observations_recomputed_CUDA_PyG_bounded_GPU_memory_original_CPU_pool_full_length_GRU_tiles")

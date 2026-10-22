"""Ordinary European executor with explicit panel dimensions and full context.
Only layout grouping and the decoder boundary change; scientific inputs, model
math, first/second sweeps and full history are preserved. No launcher or loader.
Original SHA256: 8198f589aa7e7ad6b26cb530992ed44be2481ecde2c6ef0a7c98d396408de7e4.
"""
from types import SimpleNamespace
from .. import gpu_binding
from functools import partial,wraps
from research.reviews.TreeLoom_GPU_history_source_20261017_v1.stable_leaf_history_v1 import stable_leaf_history_v1, NullProfileV1
from research.repairs.benchmark_recovery_20261020.training.layout import block_indices

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
from . import european_observations as observed
from .european_observations import implementation as frozen_default
from research.repairs.benchmark_recovery_20261020.training.decoder import geometry_from_spans

def need(ok,message):
    if not ok:raise ValueError(message)

def _cp(fn,*args):
    return checkpoint(fn,*args,use_reentrant=False,preserve_rng_state=True)

class EuropeanStagedExecutionV1:
    """Hold blocks are execution partitions; all structural/history recipes are global."""
    def __init__(self,model,bundle,metadata,*,holds_per_block=128,haplotypes_per_gru=32,resource_check=lambda label:None,history_holds_per_window=128,profile=None,frozen_module=None):
        frozen = frozen_module if frozen_module is not None else frozen_default
        need(type(model)is frozen.FrozenEuropeanTreeLoomV1 and type(bundle)is observed.EuropeanWholeParent,
             "Exact original frozen model and complete parent")
        bundle.validate()
        need(bundle.bank.parent.device.type=="cpu"and all(x.device.type=="cpu"for x in frozen._bank_tensors(bundle.bank))
             and all(x.device.type=="cpu"for x in bundle.cut.tensors()),"Full bank and recipes remain on CPU")
        need(type(holds_per_block)is int and 1<=holds_per_block<=256 and type(haplotypes_per_gru)is int
             and 1<=haplotypes_per_gru<=128 and callable(resource_check),"Bounded execution partitions")
        self.model,self.bundle,self.bank=model,bundle,bundle.bank
        self.device=next(model.parameters()).device
        need(self.device.type in ("cpu","cuda") and self.device.index in(None,0),
             "Only CPU synthetic tests or admitted logical cuda0 are supported")
        self.tick=resource_check;self.hap_tile=haplotypes_per_gru
        self.profile=profile or NullProfileV1();self.history_window=history_holds_per_window
        counts=metadata["episode_metadata"]["native_nodes_per_tree"]
        need(len(counts)==bundle.native_holds and all(type(n)is int and n>=bundle.bank.sample_index.shape[1] for n in counts)
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
            levels,children,ranks,downward=block_indices(p,ht,dp)
            ml,mr=map(int,np.searchsorted(member_hold,[a,b]))
            self.layout.append((a,b,lo,hi,p,levels,children,ranks,downward,ml,mr))
            self.tick("layout_block_complete")
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
        bank=self.bank;m=self.model;h=bank.sample_index.shape[1];d=32
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
        labels=bank.reference_class;onehot=F.one_hot(labels.clamp_min(0),4).to(site_leaf.dtype)
        onehot=onehot*(labels>=0)[:,None];unknown=(labels<0).to(site_leaf.dtype)[:,None]
        age=(bank.sample_age.log1p()/math.log(10001.))[:,None]
        observed_leaf=torch.cat((onehot,unknown,age),-1)[None].expand(t,h,6)
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
    def context_forward(self):
        self.tick("bounded_complete_parent_begin")
        pairs=[_cp(partial(self._first,z))for z in self.layout]
        up=tuple(x[0]for x in pairs);out=tuple(x[1]for x in pairs)
        eligible=self._finite_own_episode(up)
        forces=[_cp(partial(self._forcing,z),u,o)for z,u,o in zip(self.layout,up,out)]
        force=torch.cat(forces);del forces,out,pairs
        ell=self.model.memory.ell_theta.to("cpu")
        value=observed._general_history(self.model.memory,force,ell,self.bank.spans,self.bundle.cut,
            device=self.device,holds_per_window=self.history_window,resource_check=self.tick,profile=self.profile)
        del force
        sizes=[self.bundle.cut.group_offsets[z[1]]-self.bundle.cut.group_offsets[z[0]]for z in self.layout]
        values=value.split(sizes,dim=0)
        contexts=[_cp(partial(self._second,z),u,v,eligible)for z,u,v in zip(self.layout,up,values)]
        context=torch.cat(contexts,dim=0);del contexts,up,values,value
        self.tick("bounded_complete_parent_context")
        return context


def _model_binding(model, bundle, metadata, resource_check):
    need(type(model) is frozen_default.FrozenEuropeanTreeLoomV1
         and type(bundle) is observed.EuropeanWholeParent
         and model.condition == bundle.original.condition
         and callable(resource_check), "Exact European model and full observed parent required")
    import os
    parameter = next(model.parameters())
    gpu_binding.validate_env()
    need(parameter.device.type == "cuda" and parameter.device.index in (None, 0)
         and all(p.dtype == torch.float32 for p in model.parameters()),
         "A supervised assigned physical GPU FP32 model is required")
    need(metadata.get("parent_id") == bundle.parent_id
         and metadata.get("sampled_registry_ref") == bundle.sampled_registry_ref
         and metadata.get("focal_identity_sha256") == bundle.focal_identity_sha256
         and metadata.get("private_truth_opened") is False,
         "Exact immutable observation and sampled support binding is required")
    bundle.validate()
    resource_check("European_TreeLoom_complete_parent")


def _tile_logits(model, tile, geometry, holds, columns):
    sequence = torch.cat((tile, geometry[:, None].expand(-1, tile.shape[1], 3)), -1)
    history, _ = model.genomic_gru(sequence)
    return model.classifier(torch.cat((tile[holds, columns], history[holds, columns]), -1))


def predict_logits(model, bundle, metadata, *, resource_check):
    _model_binding(model, bundle, metadata, resource_check)
    need(bundle.split in ("val_prediction", "evaluation_prediction", "iid_prediction"),
         "Only held-out observation bundles are accepted for prediction")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        executor = EuropeanStagedExecutionV1(model, bundle, metadata, resource_check=resource_check)
        context = executor.context_forward()
        geometry = geometry_from_spans(bundle.bank.spans).to(device)
        output = torch.empty((bundle.focal_hold.numel(), len(model.output_classes)), dtype=torch.float32)
        seen = 0
        for first in range(0, context.shape[1], executor.hap_tile):
            last = min(first+executor.hap_tile, context.shape[1])
            rows = torch.nonzero((bundle.focal_sample_column >= first)
                                 & (bundle.focal_sample_column < last), as_tuple=False).flatten()
            if not len(rows):
                continue
            resource_check("European_full_sequence_prediction_tile")
            logits = _tile_logits(model, context[:, first:last].to(device).contiguous(), geometry,
                bundle.focal_hold[rows].to(device), (bundle.focal_sample_column[rows]-first).to(device))
            output.index_copy_(0, rows, logits.to("cpu"))
            seen += len(rows)
        need(seen == len(output) and bool(torch.isfinite(output).all()),
             "Every original sampled row must receive finite original logits")
        return output


def decode_parent_backward(model, context, geometry, focal_hold, focal_column, labels,
                           *, tile_haplotypes=32, resource_check=lambda label: None):
    """Accumulate the exact sampled mean CE and return its complete context VJP."""
    need(context.device.type == "cpu" and context.dtype == torch.float32 and context.ndim == 3,
         "Complete FP32 CPU context is required")
    need(labels.dtype == torch.long and labels.device.type == "cpu" and labels.numel() > 0
         and focal_hold.shape == focal_column.shape == labels.shape,
         "Complete aligned CPU labels and focal coordinates are required")
    need(type(tile_haplotypes) is int and 1 <= tile_haplotypes <= 128
         and geometry.shape == (context.shape[0], 3), "Complete geometry and bounded tile required")
    need(bool(((focal_hold >= 0) & (focal_hold < len(context))).all())
         and bool(((focal_column >= 0) & (focal_column < context.shape[1])).all()),
         "Original focal coordinates must lie within the complete context")
    device = next(model.parameters()).device
    gradient = torch.zeros_like(context, memory_format=torch.contiguous_format)
    loss_total = torch.zeros((), device=device, dtype=torch.float32)
    geometry = geometry.to(device)
    seen = 0
    for first in range(0, context.shape[1], tile_haplotypes):
        last = min(first+tile_haplotypes, context.shape[1])
        rows = torch.nonzero((focal_column >= first) & (focal_column < last), as_tuple=False).flatten()
        if not len(rows):
            continue
        resource_check("European_decoder_tile_forward")
        tile = context[:, first:last].detach().to(device).contiguous().requires_grad_(True)
        logits = _tile_logits(model, tile, geometry, focal_hold[rows].to(device),
                             (focal_column[rows]-first).to(device))
        loss = F.cross_entropy(logits, labels[rows].to(device), reduction="sum")/len(labels)
        need(bool(torch.isfinite(loss)), "Finite original sampled mean CE is required")
        resource_check("European_decoder_tile_backward")
        loss.backward()
        need(tile.grad is not None and bool(torch.isfinite(tile.grad).all()),
             "Finite connected complete decoder context VJP is required")
        gradient[:, first:last].copy_(tile.grad.detach().to("cpu"))
        loss_total.add_(loss.detach())
        seen += len(rows)
        del tile, logits, loss
    need(seen == len(labels), "Every sampled row, including repeated draws, contributes once")
    return loss_total, gradient


def train_sampled_step(model, bundle, targets, optimizer, metadata, *, resource_check):
    _model_binding(model, bundle, metadata, resource_check)
    from arggnn.benchmarking.targets import SampledFocalTargets
    need(bundle.split == "train" and isinstance(targets, SampledFocalTargets)
         and targets.class_order == tuple(model.output_classes)
         and type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1,
         "Original sampled TRAIN targets and AdamW are required")
    targets.validate_binding(parent_id=bundle.parent_id, split="train",
                             sampled_registry_ref=bundle.sampled_registry_ref)
    rows = torch.tensor(targets.readout_rows, dtype=torch.long)
    columns = torch.tensor(targets.query_columns, dtype=torch.long)
    need(torch.equal(rows, bundle.focal_hold)
         and bool(((columns >= 0) & (columns < bundle.bank.query_column.numel())).all())
         and torch.equal(bundle.bank.query_column[columns], bundle.focal_sample_column),
         "All registered sampled query and hold draws must remain unchanged")
    group, parameters = optimizer.param_groups[0], list(model.parameters())
    need(group["lr"] == .001 and group["weight_decay"] == .0001
         and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
         and group.get("maximize", False) is False and group.get("amsgrad", False) is False
         and len(group["params"]) == len(parameters)
         and {id(p) for p in group["params"]} == {id(p) for p in parameters},
         "Exact original optimizer settings and parameter identities are required")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    executor = EuropeanStagedExecutionV1(model, bundle, metadata, resource_check=resource_check)
    with torch.autocast(device_type="cuda", enabled=False):
        context = executor.context_forward()
        geometry = geometry_from_spans(bundle.bank.spans)
        loss, gradient = decode_parent_backward(model, context, geometry, bundle.focal_hold,
            bundle.focal_sample_column, torch.tensor(targets.labels, dtype=torch.long),
            tile_haplotypes=executor.hap_tile, resource_check=resource_check)
        resource_check("European_complete_context_backward")
        context.backward(gradient)
    need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
         "All original parameters require connected finite gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    resource_check("European_complete_parent_optimizer")
    optimizer.step()
    need(all(bool(torch.isfinite(p).all()) for p in parameters), "Finite original optimizer update required")
    return dict(parent_id=bundle.parent_id, one_complete_parent_optimizer_update=True,
        sampled_records=len(targets.labels), loss=float(loss.detach()), gradient_norm=float(norm),
        private_truth_opened_by_observation_or_executor=False, ancestry_accuracy=None,
        all_native_trees_nodes_sites_samples_retained=True, detached_history=False,
        artificial_block_boundaries=False)

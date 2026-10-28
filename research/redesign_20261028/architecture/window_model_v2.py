"""New finite-context TreeLoom prototype. No training, CUDA or source mutation.

Two complete PyG cavity sweeps surround bounded cut/leaf communication.
Labels and sample/internal identifier values are absent from model observations.
The initial implementation uses complete consecutive intervals and exact bp
kernel integrals. Sampled-coordinate semantics require an explicit adapter.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing

OBS_KEYS = (
    'parent','node_time','node_tree_index','leaf_node_index','leaf_features',
    'tree_interval_local_bp','tree_window_index','tree_core_mask',
    'focal_tree_index','focal_sample_index','focal_position_local_bp',
    'focal_genotypes','focal_allele_count','reference_class','sample_age',
    'observed_cohort','query_mask','site_tree_index','site_position_local_bp',
    'site_genotypes','site_allele_count')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_observations(path):
    """Whitelist reads: the NPZ labels member is never accessed."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in OBS_KEYS}


@dataclass(frozen=True)
class Cut:
    representatives: torch.Tensor
    groups: torch.Tensor
    counts: torch.Tensor
    valid_time: torch.Tensor
    root_saturated: torch.Tensor

    def to(self, device):
        return Cut(*(getattr(self,f.name).to(device) for f in fields(self)))


@dataclass(frozen=True)
class Batch:
    parent: torch.Tensor
    node_time: torch.Tensor
    node_tree: torch.Tensor
    leaves: torch.Tensor
    metadata: torch.Tensor
    intervals: torch.Tensor
    tree_window: torch.Tensor
    core: torch.Tensor
    reference: torch.Tensor
    cohort: torch.Tensor
    query: torch.Tensor
    site_tree: torch.Tensor
    site_genotype: torch.Tensor
    site_alleles: torch.Tensor
    focal_tree: torch.Tensor
    focal_sample: torch.Tensor
    focal_genotype: torch.Tensor
    focal_alleles: torch.Tensor
    focal_position: torch.Tensor
    up_levels: tuple
    up_children: tuple
    up_ranks: tuple
    down_levels: tuple
    cuts: tuple

    @property
    def core_focal_mask(self):
        return self.core[self.focal_tree]

    def to(self, device):
        values = {}
        for f in fields(self):
            value = getattr(self,f.name)
            if f.name == 'cuts':
                values[f.name] = tuple(x.to(device) for x in value)
            elif isinstance(value,tuple):
                values[f.name] = tuple(x.to(device) for x in value)
            else:
                values[f.name] = value.to(device)
        return Batch(**values)


def prepare_observations(arrays, cut_times=(100.,350.,1000.)):
    """CPU preparation from a strict observation-only dictionary.

    Ancient samples older than a cut are singleton computational carriers with
    valid_time=False, not ancestors claimed to exist at the younger time.
    All terminal nodes must be declared samples. Sampled interiors are rejected.
    """
    require(set(arrays)==set(OBS_KEYS), 'Exact observation whitelist required; no labels/extra fields')
    a={k:np.asarray(v) for k,v in arrays.items()}
    p=a['parent'].astype(np.int64); n=len(p)
    time=a['node_time'].astype(np.float64)
    leaves=a['leaf_node_index'].astype(np.int64)
    require(leaves.ndim==2 and len(leaves)>0, 'Nonempty [trees,samples] leaf map')
    t,h=leaves.shape
    nt=a['node_tree_index'].astype(np.int64)
    intervals=a['tree_interval_local_bp'].astype(np.float64)
    tw=a['tree_window_index'].astype(np.int64)
    require(p.shape==(n,) and time.shape==(n,) and nt.shape==(n,), 'Native node vectors')
    require(n>0 and np.isfinite(time).all() and (time>=0).all(), 'Finite nonnegative observed time')
    require(((p>=-1)&(p<n)).all() and ((nt>=0)&(nt<t)).all(), 'Native routing bounds')
    require(((leaves>=0)&(leaves<n)).all() and np.unique(leaves).size==t*h, 'Unique sample occurrences')
    require(np.array_equal(nt[leaves],np.broadcast_to(np.arange(t)[:,None],(t,h))), 'Samples belong to complete trees')
    require(intervals.shape==(t,2) and np.isfinite(intervals).all() and (intervals[:,1]>intervals[:,0]).all(), 'Strictly positive finite tree spans')
    require(tw.shape==(t,) and (tw>=0).all(), 'Nonnegative window ownership')
    require(np.array_equal(np.unique(tw),np.arange(tw.max()+1)), 'Contiguous window labels')
    w=int(tw.max())+1
    for win in range(w):
        ix=np.flatnonzero(tw==win)
        require(np.all(intervals[ix[1:],0]>=intervals[ix[:-1],1]), 'Ordered nonoverlapping intervals within each window')
    children=[[] for _ in range(n)]
    for child,parent in enumerate(p):
        if parent>=0:
            require(parent!=child and nt[parent]==nt[child] and time[parent]>=time[child], 'Within-tree acyclic monotone edge')
            children[parent].append(child)
    degree=np.array([len(x) for x in children],dtype=np.int64)
    require(set(np.flatnonzero(degree==0))==set(leaves.ravel()), 'All and only terminal nodes are declared sampled leaves')
    remaining=degree.copy(); height=np.zeros(n,dtype=np.int64)
    queue=list(np.flatnonzero(remaining==0)); at=0
    while at<len(queue):
        node=queue[at];at+=1;parent=p[node]
        if parent>=0:
            height[parent]=max(height[parent],height[node]+1)
            remaining[parent]-=1
            if remaining[parent]==0:queue.append(int(parent))
    require(len(queue)==n, 'Cycle in native parent forest')
    depth=np.zeros(n,dtype=np.int64)
    for node in reversed(queue):
        if p[node]>=0:depth[node]=depth[p[node]]+1
    levels=[];child_levels=[];ranks=[]
    for lev in range(int(height.max())+1):
        nodes=np.flatnonzero(height==lev);rank={int(v):j for j,v in enumerate(nodes)}
        child=np.array([c for v in nodes for c in children[v]],dtype=np.int64)
        levels.append(nodes);child_levels.append(child)
        ranks.append(np.array([rank[int(p[c])] for c in child],dtype=np.int64))
    down=[np.flatnonzero(depth==lev) for lev in range(1,int(depth.max())+1)]
    def sample_field(name, last=None):
        value=a[name]
        shape=(h,) if last is None else (h,last)
        if value.shape==shape:
            # A packet may contain several windows from one shared observed panel.
            value=np.broadcast_to(value,(w,)+shape)
        require(value.shape==((w,h) if last is None else (w,h,last)), 'Sample metadata shape: '+name)
        return value
    meta=sample_field('leaf_features',17).astype(np.float32)
    ref=sample_field('reference_class').astype(np.int64)
    cohort=sample_field('observed_cohort').astype(np.int64)
    ages=sample_field('sample_age').astype(np.float64)
    query=sample_field('query_mask').astype(bool)
    require(np.isfinite(meta).all() and np.isfinite(ages).all(), 'Finite sample metadata')
    require(((ref>=-1)&(ref<4)).all() and ((cohort>=0)&(cohort<9)).all(), 'Observed reference/cohort categories')
    require((ref[query]==-1).all(), 'Query ancestry must remain unknown')
    require(np.allclose(time[leaves],ages[tw],atol=1e-6,rtol=1e-6), 'Persistent sample ages')
    cuts=[]
    for horizon in cut_times:
        require(math.isfinite(horizon) and horizon>=0, 'Nonnegative finite cut time')
        groups=np.empty((t,h),dtype=np.int64);reps=[];counts=[];valid=[];saturated=[]
        for ti,row in enumerate(leaves):
            lookup={}
            for si,leaf in enumerate(row):
                node=int(leaf)
                if time[node]<=horizon:
                    while p[node]>=0 and time[p[node]]<=horizon:node=int(p[node])
                if node not in lookup:
                    lookup[node]=len(reps);reps.append(node);counts.append(0)
                    valid.append(time[node]<=horizon)
                    saturated.append(p[node]<0 and time[node]<=horizon)
                group=lookup[node];groups[ti,si]=group;counts[group]+=1
        cuts.append(Cut(torch.tensor(reps),torch.tensor(groups),torch.tensor(counts,dtype=torch.float32),
                        torch.tensor(valid,dtype=torch.float32),torch.tensor(saturated,dtype=torch.float32)))
    st=a['site_tree_index'].astype(np.int64);sg=a['site_genotypes'].astype(np.int64)
    ft=a['focal_tree_index'].astype(np.int64);fs=a['focal_sample_index'].astype(np.int64)
    fg=a['focal_genotypes'].astype(np.int64);fp=a['focal_position_local_bp'].astype(np.float64)
    require(st.ndim==1 and ((st>=0)&(st<t)).all() and sg.shape==(len(st),h), 'Observed SNP routing')
    require(ft.ndim==1 and fs.shape==ft.shape and fp.shape==ft.shape and fg.shape==(len(ft),h), 'Focal observation shapes')
    require(((ft>=0)&(ft<t)).all() and ((fs>=0)&(fs<h)).all(), 'Focal bounds')
    require(((fp>=intervals[ft,0])&(fp<intervals[ft,1])).all(), 'Focal coordinate belongs to full tree')
    require(query[tw[ft],fs].all(), 'Focal outputs are actual queries')
    for genotype,alleles in ((sg,a['site_allele_count']),(fg,a['focal_allele_count'])):
        require(alleles.shape==(len(genotype),) and ((alleles>=1)&(alleles<=4)).all(), 'One to four observed alleles')
        require(((genotype>=-1)&(genotype<alleles[:,None])).all(), 'Categorical called or missing genotype')
    core=a['tree_core_mask'].astype(bool)
    require(core.shape==(t,), 'Per-tree core mask')
    tensor=lambda x,dtype=None:torch.as_tensor(np.array(x,copy=True),dtype=dtype)
    return Batch(tensor(p),tensor(time,torch.float32),tensor(nt),tensor(leaves),tensor(meta),tensor(intervals),
                 tensor(tw),tensor(core),tensor(ref),tensor(cohort),tensor(query),tensor(st),tensor(sg),
                 tensor(a['site_allele_count'],torch.long),tensor(ft),tensor(fs),tensor(fg),
                 tensor(a['focal_allele_count'],torch.long),tensor(fp),tuple(map(tensor,levels)),
                 tuple(map(tensor,child_levels)),tuple(map(tensor,ranks)),tuple(map(tensor,down)),tuple(cuts))


def collate_observations(packets):
    """Collate independent equal-panel windows, rebuilding only bounded routes."""
    require(len(packets)>0, 'At least one packet')
    pieces={k:[] for k in OBS_KEYS};no=to=wo=0
    sample_names={'leaf_features','reference_class','sample_age','observed_cohort','query_mask'}
    for packet in packets:
        require(set(packet)==set(OBS_KEYS), 'Whitelisted packet')
        p={k:np.array(v,copy=True) for k,v in packet.items()}
        nw=int(p['tree_window_index'].max())+1
        for key in sample_names:
            single_ndim=2 if key=='leaf_features' else 1
            if p[key].ndim==single_ndim:
                p[key]=np.broadcast_to(p[key],(nw,)+p[key].shape).copy()
        p['parent']=np.where(p['parent']<0,-1,p['parent']+no)
        p['leaf_node_index']+=no
        for key in ('node_tree_index','site_tree_index','focal_tree_index'):p[key]+=to
        p['tree_window_index']+=wo
        for key in OBS_KEYS:pieces[key].append(p[key])
        no+=len(p['parent']);to+=len(p['tree_window_index']);wo+=nw
    return {k:np.concatenate(v,axis=0) for k,v in pieces.items()}


class EdgeSum(MessagePassing):
    """Actual PyG sparse aggregation of learned edge messages."""
    def __init__(self):
        super().__init__(aggr='add',node_dim=0)

    def forward(self, values, destinations, target_count):
        source=torch.arange(len(destinations),device=destinations.device)
        return self.propagate(torch.stack((source,destinations)),x=(values,None),
                              size=(len(destinations),target_count))

    def message(self,x_j):
        return x_j


def mean_to_cut(values, cut):
    result=values.new_zeros((len(cut.counts),values.shape[-1]))
    result=result.index_add(0,cut.groups.reshape(-1),values.reshape(-1,values.shape[-1]))
    return result/cut.counts[:,None]


def interval_kernel(intervals, window, ell):
    """Exact double integral of exp(-|x-y|/ell) for disjoint intervals.

    Float64 evaluation avoids cancellation for short intervals. This operator
    is additive under source splitting, not a full-model invariance promise.
    """
    bounds=intervals.double();scale=ell.double()
    width=bounds[:,1]-bounds[:,0];ratio=width/scale
    mass=-torch.expm1(-ratio)
    gap=(torch.maximum(bounds[:,None,0],bounds[None,:,0])-
         torch.minimum(bounds[:,None,1],bounds[None,:,1])).clamp_min(0)
    kernel=scale.square()*mass[:,None]*mass[None,:]*torch.exp(-gap/scale)
    ordinary=2*scale.square()*(ratio+torch.expm1(-ratio))
    series=scale.square()*(ratio.square()-ratio.pow(3)/3+ratio.pow(4)/12-ratio.pow(5)/60)
    diagonal=torch.where(ratio<1e-3,series,ordinary)
    kernel=kernel-torch.diag_embed(torch.diagonal(kernel))+torch.diag_embed(diagonal)
    return kernel*(window[:,None]==window[None,:])


def temporal_mean(values, kernel, window):
    """Window-local dense reference, avoiding cross-example arithmetic."""
    output=torch.zeros_like(values)
    for win in torch.unique(window):
        index=torch.nonzero(window==win,as_tuple=False).flatten()
        block=kernel.index_select(0,index).index_select(1,index)
        weights=(block/block.sum(-1,keepdim=True)).to(values.dtype)
        result=torch.einsum('tu,uhd->thd',weights,values.index_select(0,index))
        output=output.index_copy(0,index,result)
    return output


class WindowTreeLoom(nn.Module):
    """Complete-tree finite-context model, independent of historical checkpoints."""
    def __init__(self,width=32,classes=6,scales=3):
        super().__init__();self.width=width;self.classes=classes;self.scales=scales
        d=width
        self.site_mlp=nn.Sequential(nn.Linear(29,d),nn.GELU(),nn.Linear(d,d),nn.Tanh())
        self.site_pool=nn.Sequential(nn.Linear(2*d+2,d),nn.Tanh())
        self.leaf_projection=nn.Sequential(nn.Linear(d+17,d),nn.Tanh())
        self.node_projection=nn.Sequential(nn.Linear(3,d),nn.Tanh())
        self.up_gate=nn.Linear(3,d);self.down_gate=nn.Linear(3,d)
        self.up_message=nn.Linear(d,d,bias=False);self.down_message=nn.Linear(d,d,bias=False)
        self.up_update=nn.Linear(2*d,d);self.down_update=nn.Linear(d,d)
        self.up_norm=nn.LayerNorm(d);self.down_norm=nn.LayerNorm(d)
        self.leaf_context=nn.Sequential(nn.Linear(2*d,d),nn.Tanh())
        self.cut_project=nn.ModuleList(nn.Linear(d+2,d) for _ in range(scales))
        self.cut_output=nn.ModuleList(nn.Linear(d,d,bias=False) for _ in range(scales))
        self.leaf_output=nn.Linear(d,d,bias=False)
        self.residual_gate=nn.Parameter(torch.full((scales+1,),-3.))
        self.log_lengths=nn.Parameter(torch.log(torch.tensor([2000.,10000.,50000.][:scales]+[10000.])))
        require(scales in (1,2,3), 'Prototype supports one to three cuts')
        self.classifier=nn.Sequential(nn.Linear(2*d+17,d),nn.GELU(),nn.Linear(d,classes))
        self.aggregate=EdgeSum()

    def encode_snps(self,genotype,allele_count,windows,batch):
        j,h=genotype.shape;d=self.width
        if not j:return self.node_projection[0].weight.new_zeros((0,h,d))
        called=genotype>=0;safe=genotype.clamp_min(0)
        alleles=F.one_hot(safe,4).to(self.log_lengths.dtype)*called[:,:,None]
        ref=F.one_hot(batch.reference.clamp_min(0),4).to(alleles.dtype)*(batch.reference>=0)[:,:,None]
        cohorts=F.one_hot(batch.cohort,9).to(alleles.dtype)
        membership=torch.cat((ref,cohorts),-1).index_select(0,windows)
        counts=torch.einsum('jha,jhg->jga',alleles,membership)
        total=counts.sum(-1,keepdim=True)
        frequency=(counts+.5)/(total+.5*allele_count[:,None,None])
        available=(total>0).to(alleles.dtype)
        logfreq=torch.where(available.bool(),frequency.log(),torch.zeros_like(frequency)).transpose(1,2)
        mask=available.transpose(1,2).expand(-1,4,-1)
        panel=alleles.sum(1);panel=panel/panel.sum(-1,keepdim=True).clamp_min(1)
        panel_log=panel.clamp_min(1e-12).log()
        entropy=-(panel*panel_log).sum(-1)
        features=torch.cat((logfreq,mask,panel_log[:,:,None],entropy[:,None,None].expand(-1,4,1),
                            torch.ones((j,4,1),device=genotype.device)),dim=-1)
        encoded=self.site_mlp(features)
        values=encoded.gather(1,safe[:,:,None].expand(j,h,d))
        return values*called[:,:,None]

    def bases(self,batch):
        t,h=batch.leaves.shape;d=self.width;n=len(batch.parent)
        values=self.encode_snps(batch.site_genotype,batch.site_alleles,batch.tree_window[batch.site_tree],batch)
        called=(batch.site_genotype>=0).to(values.dtype)
        sums=values.new_zeros((t,h,d)).index_add(0,batch.site_tree,values)
        counts=values.new_zeros((t,h,1)).index_add(0,batch.site_tree,called[:,:,None])
        maxima=values.new_full((t,h,d),-torch.inf)
        if len(values):
            selected=torch.where(called[:,:,None].bool(),values,torch.full_like(values,-torch.inf))
            maxima=maxima.scatter_reduce(0,batch.site_tree[:,None,None].expand_as(selected),selected,reduce='amax',include_self=True)
        maxima=torch.where(counts>0,maxima,torch.zeros_like(maxima))
        pooled=self.site_pool(torch.cat((sums/counts.clamp_min(1),maxima,counts.log1p()/10,(counts==0).to(values.dtype)),-1))
        leaf=self.leaf_projection(torch.cat((pooled,batch.metadata[batch.tree_window]),-1))
        sampled=values.new_zeros(n).index_fill(0,batch.leaves.reshape(-1),1)
        age=batch.node_time.log1p()/math.log(10001.)
        base=self.node_projection(torch.stack((age,(batch.parent<0).to(age.dtype),sampled),-1))
        base=base.index_add(0,batch.leaves.reshape(-1),leaf.reshape(-1,d))
        parent_age=batch.node_time[batch.parent.clamp_min(0)]
        duration=(parent_age-batch.node_time).clamp_min(0).log1p()/math.log(10001.)
        geometry=torch.stack((age,parent_age.log1p()/math.log(10001.),duration),-1)
        gates=(self.up_gate(geometry).sigmoid(),self.down_gate(geometry).sigmoid())
        degree=torch.bincount(batch.parent[batch.parent>=0],minlength=n).to(base.dtype).clamp_min(1)
        return base,gates,degree

    def sweep(self,batch,base,gates,degree):
        n,d=base.shape;up=torch.zeros_like(base);total=torch.zeros_like(base);child_message=torch.zeros_like(base)
        for nodes,children,ranks in zip(batch.up_levels,batch.up_children,batch.up_ranks):
            if len(children):
                msg=self.up_message(up[children])*gates[0][children]/degree[batch.parent[children],None]
                aggregate=self.aggregate(msg,ranks,len(nodes))
                child_message=child_message.index_copy(0,children,msg)
            else:aggregate=base.new_zeros((len(nodes),d))
            total=total.index_copy(0,nodes,aggregate)
            value=self.up_norm(base[nodes]+self.up_update(torch.cat((base[nodes],aggregate),-1)).tanh())
            up=up.index_copy(0,nodes,value)
        outside=torch.zeros_like(base)
        for nodes in batch.down_levels:
            p=batch.parent[nodes]
            cavity=outside[p]+base[p]+total[p]-child_message[nodes]
            message=self.down_message(cavity)*gates[1][nodes]
            routed=self.aggregate(message,torch.arange(len(nodes),device=nodes.device),len(nodes))
            outside=outside.index_copy(0,nodes,self.down_norm(self.down_update(routed).tanh()))
        leaf=self.leaf_context(torch.cat((up[batch.leaves],outside[batch.leaves]),-1))
        return up,outside,leaf

    def forward(self,batch,*,mode='structure'):
        require(type(batch) is Batch, 'Prepared observation-only Batch required')
        require(mode in ('structure','leaf_only','no_cross'), 'Declared matched communication control')
        require(len(batch.cuts)==self.scales, 'Model and observed cut count agree')
        base,gates,degree=self.bases(batch)
        up,outside,leaf=self.sweep(batch,base,gates,degree)
        injection=torch.zeros_like(base)
        for si,cut in enumerate(batch.cuts):
            z=self.cut_project[si](torch.cat((up[cut.representatives]+outside[cut.representatives],
                                            cut.valid_time[:,None],cut.root_saturated[:,None]),-1))
            kernel=interval_kernel(batch.intervals,batch.tree_window,self.log_lengths[si].exp())
            lifted=z[cut.groups]
            transported=temporal_mean(lifted,kernel,batch.tree_window)
            delta=mean_to_cut(transported,cut)-z
            if mode!='structure':delta=delta*0
            residual=self.cut_output[si](delta)*self.residual_gate[si].sigmoid()/self.scales
            injection=injection.index_add(0,cut.representatives,residual)
        kernel=interval_kernel(batch.intervals,batch.tree_window,self.log_lengths[-1].exp())
        delta=temporal_mean(leaf,kernel,batch.tree_window)-leaf
        if mode=='no_cross':delta=delta*0
        residual=self.leaf_output(delta)*self.residual_gate[-1].sigmoid()
        injection=injection.index_add(0,batch.leaves.reshape(-1),residual.reshape(-1,self.width))
        _,_,second=self.sweep(batch,base+injection,gates,degree)
        focal=self.encode_snps(batch.focal_genotype,batch.focal_alleles,batch.tree_window[batch.focal_tree],batch)
        row=torch.arange(len(batch.focal_tree),device=base.device)
        context=second[batch.focal_tree,batch.focal_sample]
        snp=focal[row,batch.focal_sample]
        metadata=batch.metadata[batch.tree_window[batch.focal_tree],batch.focal_sample]
        return self.classifier(torch.cat((context,snp,metadata),-1))

    def core_logits(self,batch,*,mode='structure'):
        return self(batch,mode=mode)[batch.core_focal_mask]

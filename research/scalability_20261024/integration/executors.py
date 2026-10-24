"""Explicit original/integrated executor selection, without source monkeypatches.

Only immutable layout construction and the row-independent SNP MLP execution
are replaced. Inherited first/second sweeps, forcing, non-reentrant checkpoints,
stable-leaf history, decoder inputs and whole-history autograd stay unchanged.
"""
import math
from types import SimpleNamespace
import numpy as np
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed import (
    staged_execution_v2 as original_staged,
    bounded_execution_v2 as original_bounded)
from research.scalability_20261024.categorical_site.categorical_site import (
    categorical_site_chunk)
from .cache import LayoutCache


class _IntegratedLayoutAndSites:
    def _initialize(self,model,bundle,metadata,*,layout_cache,holds_per_block,
                    haplotypes_per_gru,resource_check,
                    history_holds_per_window,profile,frozen_module):
        frozen = frozen_module
        need = original_staged.need
        need(type(model) is frozen.AmericanTreeLoomC3FullV1
             and type(bundle) is frozen.AmericanTreeLoomWholeParentV1,
             "Exact original model and authenticated complete bundle required")
        bundle.validate()
        need(bundle.bank.parent.device.type=="cpu"
             and all(x.device.type=="cpu" for x in frozen._bank_tensors(bundle.bank))
             and all(x.device.type=="cpu" for x in bundle.cut.tensors()),
             "Original complete observation bank and recipes remain CPU")
        need(type(holds_per_block) is int and 1<=holds_per_block<=256
             and type(haplotypes_per_gru) is int and 1<=haplotypes_per_gru<=128
             and callable(resource_check),"Original bounded execution partitions")
        self.model,self.bundle,self.bank = model,bundle,bundle.bank
        self.device = next(model.parameters()).device
        need(self.device.type=="cuda" and self.device.index in (None,0),
             "Only admitted logical cuda0; production GPU guard unchanged")
        import os
        need(os.environ.get("ARGGNN_GPU_MEMORY_POLICY")=="full_device",
             "Original explicit full-device permission required")
        need(isinstance(layout_cache,LayoutCache),
             "Integrated executor requires explicitly budgeted LayoutCache")
        self.tick,self.hap_tile = resource_check,haplotypes_per_gru
        self.profile = profile or original_staged.NullProfileV1()
        self.history_window = history_holds_per_window
        counts = metadata["episode_metadata"]["native_nodes_per_tree"]
        need(len(counts)==bundle.native_holds
             and all(type(n) is int and n>=1600 for n in counts)
             and sum(counts)==self.bank.parent.numel(),
             "Original complete native per-hold node partition required")
        self._layout_lease = layout_cache.acquire(bundle,metadata,
            holds_per_block=holds_per_block,resource_check=resource_check)
        self.layout = self._layout_lease.layout

    def release_layout(self):
        """Call only after all forward/backward recomputation has completed."""
        lease = getattr(self,"_layout_lease",None)
        self.layout = ()
        self._layout_lease = None
        if lease is not None:
            lease.release()

    def __del__(self):
        self.release_layout()

    def _block(self,z):
        # Immutable NumPy index buffers require an explicit transfer copy.
        # Original code also requested torch.long on a different CUDA device.
        a,b,lo,hi,p,levels,children,ranks,downward,_,_ = z
        bank = self.bank
        def tensor(x,dtype=torch.long):
            return torch.tensor(x,dtype=dtype,device=self.device)
        return SimpleNamespace(parent=tensor(p),
            node_time=bank.node_time[lo:hi].to(self.device),
            sample_index=(bank.sample_index[a:b]-lo).to(self.device),
            upward_levels=tuple(tensor(x) for x in levels),
            upward_children=tuple(tensor(x) for x in children),
            upward_parent_rank=tuple(tensor(x) for x in ranks),
            downward_levels=tuple(tensor(x) for x in downward),
            edge_geometry=bank.edge_geometry[lo:hi].to(self.device),
            branch_support_true=bank.branch_support_true[lo:hi].to(self.device),
            reference_class=bank.reference_class.to(self.device),
            sample_age=bank.sample_age.to(self.device),
            spans=bank.spans[a:b].to(self.device))

    def _sites(self,a,b):
        """Original global256 chunks, selection, pooling order and projection."""
        bank,m = self.bank,self.model
        h,d = 1600,32
        site_hold = bank.site_hold.numpy()
        left,right = map(int,np.searchsorted(site_hold,[a,b]))
        hs,ss,mm,cc = [],[],[],[]
        for start in range((left//256)*256,right,256):
            end = min(start+256,len(site_hold))
            global_hold = bank.site_hold[start:end]
            holds,inverse = torch.unique_consecutive(global_hold,return_inverse=True)
            selected = torch.nonzero((holds>=a)&(holds<b),as_tuple=False).flatten()
            if not len(selected):
                continue
            args = tuple(x[start:end].to(self.device) for x in (
                bank.genotype,bank.log_donor_allele_frequency,
                bank.donor_call_mask,bank.panel_allele_frequency,bank.site_entropy))
            sums,maxima,count = categorical_site_chunk(
                m,*args,inverse.to(self.device),len(holds))
            ix = selected.to(self.device)
            hs.append((holds[selected]-a).to(self.device))
            ss.append(sums.index_select(0,ix))
            mm.append(maxima.index_select(0,ix))
            cc.append(count.index_select(0,ix))
        sums = bank.spans.new_zeros((b-a,h,d),device=self.device)
        maxima = bank.spans.new_full((b-a,h,d),-torch.inf,device=self.device)
        counts = bank.spans.new_zeros((b-a,h,1),device=self.device)
        if hs:
            index = torch.cat(hs)
            sums = sums.index_add(0,index,torch.cat(ss))
            counts = counts.index_add(0,index,torch.cat(cc))
            chunk_max,chunk_count = torch.cat(mm),torch.cat(cc)
            chunk_max = torch.where(chunk_count>0,chunk_max,
                                    torch.full_like(chunk_max,-torch.inf))
            maxima = maxima.scatter_reduce(0,
                index[:,None,None].expand_as(chunk_max),chunk_max,
                reduce="amax",include_self=True)
        present = counts>0
        means = sums/counts.clamp_min(1)
        maxima = torch.where(present,maxima,torch.zeros_like(maxima))
        logcount = counts.log1p()/math.log(100001.)
        empty = (~present).to(sums.dtype)
        span = (bank.spans[a:b].to(self.device).log1p()/math.log(2000001.))[:,None,None].expand(b-a,h,1)
        return m.site_pool_projection(torch.cat(
            (means,maxima,logcount,empty,span),-1))


class IntegratedStagedExecution(_IntegratedLayoutAndSites,
                                original_staged.StagedExecutionV1):
    def __init__(self,model,bundle,metadata,*,layout_cache,
                 holds_per_block=128,haplotypes_per_gru=64,
                 resource_check=lambda label:None,history_holds_per_window=1024,
                 profile=None,frozen_module=None):
        self._initialize(model,bundle,metadata,layout_cache=layout_cache,
            holds_per_block=holds_per_block,haplotypes_per_gru=haplotypes_per_gru,
            resource_check=resource_check,history_holds_per_window=history_holds_per_window,
            profile=profile,frozen_module=frozen_module or original_staged.frozen_default)


class IntegratedBoundedExecution(_IntegratedLayoutAndSites,
                                 original_bounded.BoundedExecutionV1):
    def __init__(self,model,bundle,metadata,*,layout_cache,
                 holds_per_block=64,haplotypes_per_gru=64,
                 resource_check=lambda label:None,history_holds_per_window=1024,
                 profile=None):
        self._initialize(model,bundle,metadata,layout_cache=layout_cache,
            holds_per_block=holds_per_block,haplotypes_per_gru=haplotypes_per_gru,
            resource_check=resource_check,history_holds_per_window=history_holds_per_window,
            profile=profile,frozen_module=original_bounded.frozen)


def build_executor(kind,model,bundle,metadata,*,execution="staged",
                   layout_cache=None,frozen_module=None,**kwargs):
    """Clean profiler/runtime selection; production constructors retain guards.

    A reviewed diagnostic slice may supply its own explicit frozen_module
    namespace to the staged route, as the original staged constructor supports.
    That route does not claim complete-parent scientific equivalence.
    """
    if kind not in ("original","integrated"):
        raise ValueError("Explicit original or integrated executor required")
    if execution=="staged":
        kwargs["frozen_module"] = frozen_module
        cls = (original_staged.StagedExecutionV1 if kind=="original"
               else IntegratedStagedExecution)
    elif execution=="bounded":
        if frozen_module is not None:
            raise ValueError("Bounded prediction retains the original frozen namespace")
        cls = (original_bounded.BoundedExecutionV1 if kind=="original"
               else IntegratedBoundedExecution)
    else:
        raise ValueError("Explicit staged or bounded execution route required")
    if kind=="integrated":
        kwargs["layout_cache"] = layout_cache
    return cls(model,bundle,metadata,**kwargs)

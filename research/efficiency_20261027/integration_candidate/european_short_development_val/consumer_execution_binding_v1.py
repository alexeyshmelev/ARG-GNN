"""Isolated exact-type constructor; all inherited numerical methods unchanged.

This module provides no launch or training path. Predictions remain pending
independent source review, weights/registry sealing and ROOT admission.
"""
from types import SimpleNamespace
import numpy as np
import torch
from research.repairs.benchmark_multigpu_20261022.scientific import gpu_binding
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_execution as original
from research.repairs.benchmark_recovery_20261020.training.layout import block_indices
from research.reviews.TreeLoom_GPU_history_source_20261017_v1.stable_leaf_history_v1 import NullProfileV1
from . import observations_v1 as observed
from .cohort_model_binding_v1 import ShortDevelopmentCohortTreeLoom, CONDITION, PANEL_BINDING, need
from .cohort_observations_v1 import ShortDevelopmentCohortParent


class ShortDevelopmentStagedExecution(original.EuropeanStagedExecutionV1):
    def __init__(self,model,bundle,metadata,*,holds_per_block=128,haplotypes_per_gru=32,resource_check=lambda label:None,history_holds_per_window=128,profile=None,frozen_module=None):
        need(frozen_module is not None, "Explicit isolated short model module required")
        frozen = frozen_module
        need(type(model)is frozen.FrozenEuropeanTreeLoomV1 and type(bundle)is observed.ShortDevelopmentWholeParent,
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


class ShortDevelopmentCohortStagedExecution(ShortDevelopmentStagedExecution):
    def __init__(self, model, bundle, metadata, **kwargs):
        need(type(model) is ShortDevelopmentCohortTreeLoom and type(bundle) is ShortDevelopmentCohortParent,
             'Explicit successor model and all-panel cohort bundle')
        bundle.validate()
        need(model.scientific_condition == metadata.get('scientific_condition') == CONDITION,
             'Distinct scientific condition on model and prepared observations')
        self.cohort_parent = bundle
        # The original executor already exposes this typed module injection.
        # No monkeypatch, inherited global mutation or compatibility alias.
        module = SimpleNamespace(FrozenEuropeanTreeLoomV1=ShortDevelopmentCohortTreeLoom,
                                 _bank_tensors=observed.implementation._bank_tensors)
        super().__init__(model, bundle.original, metadata, frozen_module=module, **kwargs)
        self.cohort_device = bundle.observed_cohort.to(self.device)

    def _base(self, bank, site_leaf):
        return self.model.base_from_sites(bank, site_leaf, self.cohort_device)


def binding(model, bundle, metadata, resource_check):
    need(type(model) is ShortDevelopmentCohortTreeLoom and type(bundle) is ShortDevelopmentCohortParent
         and model.scientific_condition == bundle.scientific_condition == CONDITION
         and model.condition == bundle.original.original.condition == PANEL_BINDING
         and callable(resource_check), 'Exact new condition and unchanged physical panel')
    gpu_binding.validate_env()
    parameter = next(model.parameters())
    need(parameter.device.type == 'cuda' and parameter.device.index in (None, 0)
         and all(p.dtype == torch.float32 for p in model.parameters()),
         'Supervised assigned physical GPU and FP32 required for real data')
    need(metadata.get('scientific_condition') == CONDITION
         and metadata.get('parent_id') == bundle.parent_id
         and metadata.get('sampled_registry_ref') == bundle.sampled_registry_ref
         and metadata.get('focal_identity_sha256') == bundle.focal_identity_sha256
         and metadata.get('observed_cohort_manifest_ref') == bundle.sample_manifest_ref
         and metadata.get('private_truth_opened') is False,
         'Exact immutable observation, cohort and registered focal support binding')
    bundle.validate()
    resource_check('European_cohort_complete_parent')


def predict_validation(model,bundle,metadata,*,resource_check):
    binding(model,bundle,metadata,resource_check)
    need(bundle.split=='val_prediction', 'Only independently registered VAL observations')
    model.eval();device=next(model.parameters()).device
    with torch.no_grad():
        executor=ShortDevelopmentCohortStagedExecution(model,bundle,metadata,resource_check=resource_check)
        context=executor.context_forward();geometry=original.geometry_from_spans(bundle.bank.spans).to(device)
        output=torch.empty((bundle.focal_hold.numel(),6),dtype=torch.float32)
        seen=0
        for first in range(0,context.shape[1],executor.hap_tile):
            last=min(first+executor.hap_tile,context.shape[1])
            rows=torch.nonzero((bundle.focal_sample_column>=first)&(bundle.focal_sample_column<last),as_tuple=False).flatten()
            if not len(rows):continue
            resource_check('European_cohort_full_sequence_validation_tile')
            logits=original._tile_logits(model,context[:,first:last].to(device).contiguous(),geometry,
                bundle.focal_hold[rows].to(device),(bundle.focal_sample_column[rows]-first).to(device))
            output.index_copy_(0,rows,logits.cpu());seen+=len(rows)
        need(seen==len(output) and bool(torch.isfinite(output).all()), 'All independent VAL rows predicted')
        return output

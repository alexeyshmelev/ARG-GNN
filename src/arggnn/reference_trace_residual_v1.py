"""H027 reference-driven residual over the established persistence operator.

Only observed R proxy labels supply signal and normalization mass. Frozen query
unaries are the output skip only. This is an information-route hypothesis, not
a novel operator/GNN claim. No floors or invented query/reference classes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .persistence_trace_kernel_v1 import (
    MetricTreeHoldV1, PersistenceTraceBankV1, _tree, solve_screened_v1,
)

ARMS = ("sample_only", "genuine_ancestor", "sham_ancestor")


@dataclass(frozen=True)
class ReferenceTraceBankV1:
    operator: PersistenceTraceBankV1
    reference_nodes: Tensor
    source_mass: Tensor
    source_signal: Tensor
    age_groups: tuple
    metadata: dict

    def to(self, device):
        return replace(self, operator=self.operator.to(device),
            reference_nodes=self.reference_nodes.to(device),
            source_mass=self.source_mass.to(device), source_signal=self.source_signal.to(device),
            age_groups=tuple(group.to(device) for group in self.age_groups))


def prepare_reference_trace_bank_v1(
        operator: PersistenceTraceBankV1, trees: Sequence[MetricTreeHoldV1], intervals_bp: Tensor,
        query_ids: Tensor, reference_ids: Tensor, reference_classes: Tensor, *,
        census: float = 350.) -> ReferenceTraceBankV1:
    """Join actual R IDs to native carrier rows plus original branch-midpoint offsets."""
    arrays=(intervals_bp,query_ids,reference_ids,reference_classes)
    if (not trees or any(x.device.type!="cpu" or x.requires_grad for x in arrays)
            or intervals_bp.dtype!=torch.float64 or intervals_bp.shape!=(len(trees),2)
            or any(x.dtype!=torch.long or x.ndim!=1 for x in arrays[1:])
            or reference_classes.shape!=reference_ids.shape or not len(query_ids) or not len(reference_ids)
            or query_ids.unique().numel()!=len(query_ids) or reference_ids.unique().numel()!=len(reference_ids)
            or bool(torch.isin(query_ids,reference_ids).any())
            or bool(((reference_classes<0)|(reference_classes>3)).any())
            or not bool(torch.isfinite(intervals_bp).all())
            or bool((intervals_bp[:,1]<=intervals_bp[:,0]).any())
            or not torch.equal(intervals_bp[:-1,1],intervals_bp[1:,0])
            or operator.spatial.device.type!="cpu" or operator.spatial.dtype!=torch.float32):
        raise ValueError("Fixed original CPU geometry and distinct observed Q/R sources required")
    rows, weights, signals, expected_query_nodes=[],[],[],[]
    offset=0
    query_ages=reference_ages=None
    class_counts=None
    source_labels=torch.nn.functional.one_hot(reference_classes,4).float()
    scale=float(operator.metadata["genomic_scale_bp"])
    for tree,interval in zip(trees,intervals_bp):
        samples,branches,count=_tree(tree)
        if count!=tree.parent.numel()+int((tree.parent>=0).sum()):
            raise ValueError("Carrier must retain original vertices and one original-branch midpoint")
        if any(identity not in samples for identity in (*query_ids.tolist(),*reference_ids.tolist())):
            raise ValueError("Every hold requires complete actual Q/R sample joins")
        qnodes=[samples[identity] for identity in query_ids.tolist()]
        rnodes=[samples[identity] for identity in reference_ids.tolist()]
        qa,ra=tree.node_times[qnodes],tree.node_times[rnodes]
        if query_ages is None:
            query_ages,reference_ages=qa.clone(),ra.clone()
        elif not torch.equal(qa,query_ages) or not torch.equal(ra,reference_ages):
            raise ValueError("Persistent observed sample dates changed")
        young=ra<census
        if not bool(young.any()) or not bool((~young).any()):
            raise ValueError("Both observed reference age cohorts required")
        mass=torch.zeros((len(reference_ids),8),dtype=torch.float32)
        signal=torch.zeros_like(mass)
        counts=[]
        width=float(interval[1]-interval[0])/scale
        for cohort,selected in enumerate((young,~young)):
            labels=reference_classes[selected]
            count_by_class=torch.bincount(labels,minlength=4)
            present=int((count_by_class>0).sum())
            # The ones source is cohort mass, not a class label. An absent ANA
            # older class has exactly zero numerator despite positive old mass.
            balance=1./(present*count_by_class[labels].float())
            mass[selected,cohort*4:cohort*4+4]=width*balance[:,None]
            signal[selected,cohort*4:cohort*4+4]=source_labels[selected]
            counts.append(count_by_class.tolist())
        if class_counts is None:
            class_counts=counts
        elif class_counts!=counts:
            raise ValueError("Fixed observed reference cohort classes changed")
        rows.extend(offset+node for node in rnodes)
        expected_query_nodes.extend(offset+node for node in qnodes)
        weights.append(mass);signals.append(signal)
        offset+=count
    expected=torch.tensor(expected_query_nodes,dtype=torch.long)
    if (offset!=operator.num_states or not torch.equal(operator.source_nodes,expected)
            or operator.readout_left.shape[1]!=len(query_ids)
            or operator.metadata["original_vertices"]!=sum(tree.parent.numel() for tree in trees)
            or operator.metadata["branch_midpoints"]!=sum(int((tree.parent>=0).sum()) for tree in trees)):
        raise ValueError("Existing H025 operator carrier/source order differs from genuine observed trees")
    groups=tuple((query_ages==age).nonzero().flatten() for age in torch.unique(query_ages))
    metadata={
        "kind":"reference_driven_established_persistence_operator",
        "source":"only_observed_reference_proxy_labels;no_query_unary_propagation",
        "reference_count":len(reference_ids),"younger_reference_count":int((reference_ages<census).sum()),
        "older_reference_count":int((reference_ages>=census).sum()),
        "younger_class_counts":class_counts[0],"older_class_counts":class_counts[1],
        "actual_reference_ids":reference_ids.tolist(),"reference_classes":reference_classes.tolist(),
        "reference_ages":reference_ages.tolist(),
        "actual_query_ids":query_ids.tolist(),
        "query_age_groups":{str(float(age)):int((query_ages==age).sum()) for age in torch.unique(query_ages)},
        "carrier_rows":"cumulative_original_Nnative_plus_original_Nbranches+actual_sample_join_local_row",
        "source_weights":"physical_hold_width/genomic_scale_bp*inverse_class_count/number_observed_classes_per_cohort",
        "normalizer":"matching_reference_cohort_ones_same_weights_same_channel_operator_and_readout",
        "older_labels_are_observed_proxies_not_ancestral_truth":True,
        "centering":"full_actual_Q_same_exact_observed_age_at_each_original_readout;unlabeled_transduction",
        "operator":operator.metadata,
    }
    return ReferenceTraceBankV1(operator,torch.tensor(rows,dtype=torch.long),
        torch.cat(weights),torch.cat(signals),groups,metadata)


class ReferenceTraceResidualV1(nn.Module):
    def __init__(self, *, initial_spatial_rate:float=18., initial_temporal_rate:float=1.,
                 max_rate:float=20., solver_rtol:float=1e-6, solver_max_iterations:int=128):
        super().__init__()
        self.min_rate=1e-4
        if (not self.min_rate<initial_spatial_rate<max_rate
                or not self.min_rate<initial_temporal_rate<max_rate):
            raise ValueError("Initial rates must lie strictly within the fixed positive rate range")
        self.max_rate=max_rate
        self.rtol,self.iterations=solver_rtol,solver_max_iterations
        raw=lambda rate:math.log((rate-self.min_rate)/(max_rate-rate))
        self.raw_spatial=nn.Parameter(torch.full((8,),raw(initial_spatial_rate),dtype=torch.float32))
        self.raw_temporal=nn.Parameter(torch.full((8,),raw(initial_temporal_rate),dtype=torch.float32))
        self.context_head=nn.Linear(8,4,bias=False,dtype=torch.float32)
        nn.init.zeros_(self.context_head.weight)
        self.last_mass_min=None

    def forward(self, bank:ReferenceTraceBankV1, frozen_baseline_logits:Tensor, *, control:str):
        if control not in ARMS:
            raise ValueError("Unknown matched reference-trace control")
        operator=bank.operator
        if (frozen_baseline_logits.shape!=(*operator.readout_left.shape,4)
                or frozen_baseline_logits.dtype!=torch.float32
                or frozen_baseline_logits.device!=self.raw_spatial.device
                or bank.source_mass.device!=self.raw_spatial.device
                or not bool(torch.isfinite(frozen_baseline_logits).all())):
            raise ValueError("Aligned finite frozen FP32 baseline and reference-only bank required")
        numerator=bank.source_signal*bank.source_mass
        rhs=bank.source_mass.new_zeros((operator.num_states,16))
        rhs.index_add_(0,bank.reference_nodes,torch.cat((numerator,bank.source_mass),1))
        spatial=(self.min_rate+(self.max_rate-self.min_rate)*self.raw_spatial.sigmoid()).repeat(2)
        temporal_rate=(self.min_rate+(self.max_rate-self.min_rate)*self.raw_temporal.sigmoid()).repeat(2)
        temporal=operator.sample
        if control!="sample_only":
            temporal=(temporal+getattr(operator,control)).coalesce()
        state=solve_screened_v1(rhs,spatial,temporal_rate,operator.spatial,temporal,
            max_iterations=self.iterations,rtol=self.rtol,atol=0.)
        fraction=operator.readout_right_weight[...,None]
        readout=state[operator.readout_left]*(1-fraction)+state[operator.readout_right]*fraction
        values,mass=readout.split(8,-1)
        if not bool(torch.isfinite(readout).all()) or bool((mass<=0).any()):
            raise RuntimeError("Reference-only FP32 readout mass is not finite-positive; no floor: "+
                f"min={float(mass.detach().min())},nonpositive={int((mass<=0).sum())}")
        self.last_mass_min=float(mass.detach().min())
        context=values/mass
        context=context.clone()
        for group in bank.age_groups:
            context[:,group]-=context[:,group].mean(1,keepdim=True)
        if not bool(torch.isfinite(context).all()):
            raise RuntimeError("Nonfinite reference-normalized centered graph context")
        return frozen_baseline_logits.detach()+self.context_head(context)

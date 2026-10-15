"""Prepare immutable causal annual graphs, cut-change plans and device tensors."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
import torch
from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.whole_country_tree_encoder_v1 import make_full_tree_schedule_v1
from research.reviews.H051_conditional_innovation_design_20261016_v1.conditional_innovation_v1 import make_plan_v1

def require(ok,message):
    if not ok:raise ValueError(message)

def canonical_cut_v1(membership):
    """Anonymous cut labels by first country occurrence; no internal-ID matching."""
    require(membership.device.type=="cpu" and membership.ndim==1,"CPU country-aligned membership")
    labels={};out=[]
    for value in membership.tolist():
        if value not in labels:labels[value]=len(labels)
        out.append(labels[value])
    return torch.tensor(out,dtype=torch.long)

@dataclass(frozen=True)
class PreparedYearV1:
    year:int
    graph:object
    schedule:object
    event_plan:object
    cut_changed:bool

    def to(self,device):
        return PreparedYearV1(self.year,self.graph.clone().to(device),self.schedule.to(device),
            None if self.event_plan is None else self.event_plan.to(device),self.cut_changed)

@dataclass(frozen=True)
class PreparedSequenceV1:
    steps:tuple
    catalogue_sha256:str
    observation_digests:tuple
    builder_sha256:str

    def to(self,device):
        # Transfer once BEFORE repeated epochs; graph/schedule/plan stay resident.
        return PreparedSequenceV1(tuple(step.to(device) for step in self.steps),
            self.catalogue_sha256,self.observation_digests,self.builder_sha256)

def prepare_sequence_v1(dataset,*,builder_sha256):
    """No labels. Every graph independently uses complete years < its forecast."""
    require(tuple(dataset.forecast_years)==tuple(range(1987,2017)),"Full audited forecast sequence")
    steps=[];digests=[];a=b=None;digest_a=digest_b=None;catalogue=None
    for index,year in enumerate(dataset.forecast_years):
        graph=dataset[index]
        require(graph.leaf_profile.shape==(255,255) and graph.country_witness_index.tolist()==list(range(255))
            and graph.observation_year.tolist()==[year-1],"Complete causal country graph")
        if catalogue is None:catalogue=graph.catalogue_sha256
        require(graph.catalogue_sha256==catalogue,"One country witness catalogue")
        c=canonical_cut_v1(graph.cut_node_index);plan=None;changed=False
        if b is None:a=b=c;digest_a=digest_b=graph.observation_sha256
        elif not torch.equal(b,c):
            source=dict(annual_observations=[digest_a,digest_b,graph.observation_sha256],
                catalogue_sha256=catalogue,forecast_year=year)
            digest=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            provenance=dict(schema="H051_observation_only_cut_source_v1",observed_trees_sha256=digest,
                builder_sha256=builder_sha256,targets_or_outcomes_used=False,identity_namespace_sha256=None)
            plan=make_plan_v1(a,b,c,provenance=provenance);changed=True
            a,b=b,c;digest_a,digest_b=digest_b,graph.observation_sha256
        else:
            # Hold: retain the last two DISTINCT cut episodes and current forcing.
            digest_b=graph.observation_sha256
        steps.append(PreparedYearV1(year,graph,make_full_tree_schedule_v1(graph),plan,changed))
        digests.append(graph.observation_sha256)
    return PreparedSequenceV1(tuple(steps),catalogue,tuple(digests),builder_sha256)

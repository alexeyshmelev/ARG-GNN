"""Separate target capabilities and exact H052 support/metric; no label reader."""
from __future__ import annotations
from dataclasses import dataclass
import torch

def require(ok,message):
    if not ok:raise ValueError(message)

@dataclass(frozen=True)
class TargetYearV1:
    year:int
    values:torch.Tensor
    source_present:torch.Tensor

def validate_target_v1(target,year):
    require(isinstance(target,TargetYearV1) and target.year==year,"Requested target year")
    value,present=target.values,target.source_present
    require(value.device.type=="cpu" and present.device.type=="cpu" and value.shape==(255,255)
        and value.dtype==torch.float64 and not value.requires_grad and value.grad_fn is None and present.shape==(255,) and present.dtype==torch.bool,"Full CPU official country/destination support")
    require(bool(torch.isfinite(value).all()) and bool((value>=0).all()) and not bool(value[~present].any())
        and bool(present.any()) and bool((value[present].sum(1)>0).all()),"Exact official queried-source support")
    return target

def prepare_train_targets_v1(loader,device):
    # No VAL/TEST target callback is accepted by the trainer.
    out={}
    for year in range(1987,2010):
        t=validate_target_v1(loader(year),year);value=t.values.to(device=device,dtype=torch.float32)
        mask=t.source_present.to(device=device);normalized=value[mask]/value[mask].sum(1,keepdim=True)
        out[year]=(normalized,mask)
    return out

def equal_year_soft_cross_entropy_v1(logits,target):
    # All 255 logits exist before source selection; each queried nation equal weight.
    normalized,mask=target
    return -(normalized*torch.log_softmax(logits[mask],dim=1)).sum(1).mean()

def score_year_v1(scores,target):
    import sklearn
    from sklearn.metrics import ndcg_score
    require(sklearn.__version__=="1.9.1","Exact H052 sklearn metric version")
    validate_target_v1(target,target.year)
    require(scores.device.type=="cpu" and scores.shape==(255,255) and bool(torch.isfinite(scores).all()),
        "Previously sealed all-country scores")
    present=target.source_present
    score=float(ndcg_score(target.values[present].numpy(),scores[present].numpy(),k=10,ignore_ties=False))
    return dict(target_year=target.year,queried_countries=int(present.sum()),ndcg_at10=score,
        latest_allowed_edge_year=target.year-1)

def summarize_years_v1(rows):
    import numpy as np
    values=np.asarray([r["ndcg_at10"] for r in rows],dtype=np.float64)
    return dict(mean_ndcg_at10=float(values.mean()),sample_SD_over_years=float(values.std(ddof=1)),
        year_count=len(rows),years=[r["target_year"] for r in rows],
        queried_country_years=sum(r["queried_countries"] for r in rows),
        uncertainty="Descriptive serially dependent year SD; not optimizer seed uncertainty")

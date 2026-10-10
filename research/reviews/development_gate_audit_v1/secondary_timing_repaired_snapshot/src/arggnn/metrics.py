"""Pilot metrics on the declared fixed grid, with parent replicate summaries."""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor

def classification_metrics(probabilities: Tensor, labels: Tensor) -> dict:
    p = probabilities.detach().cpu().double().reshape(-1, probabilities.shape[-1])
    y = labels.detach().cpu().long().reshape(-1)
    k = p.shape[1]
    if p.shape[0] != y.numel() or not y.numel():
        raise ValueError("Nonempty aligned probabilities and labels required")
    if not torch.isfinite(p).all() or (p < 0).any() or (y < 0).any() or (y >= k).any():
        raise ValueError("Invalid probabilities or labels")
    if not torch.allclose(p.sum(-1),torch.ones(len(p),dtype=p.dtype),atol=1e-5,rtol=0):
        raise ValueError("Probabilities must sum to one")
    pred = p.argmax(-1)
    confusion = torch.bincount(y*k+pred,minlength=k*k).reshape(k,k).double()
    precision = confusion.diag()/confusion.sum(0).clamp_min(1)
    recall = confusion.diag()/confusion.sum(1).clamp_min(1)
    f1 = 2*precision*recall/(precision+recall).clamp_min(1e-15)
    onehot = torch.nn.functional.one_hot(y,k).double()
    return {
        "accuracy_uniform_grid":float((pred==y).double().mean()),
        "nll_uniform_grid":float(-p[torch.arange(len(y)),y].clamp_min(1e-12).log().mean()),
        "brier_uniform_grid":float((p-onehot).square().sum(-1).mean()),
        "macro_f1_all_classes":float(f1.mean()),
        "confusion":confusion.long().tolist(),"per_class_precision":precision.tolist(),
        "per_class_recall":recall.tolist(),"observations":len(y),
    }

def replicate_summary(metrics: list[dict], seed: int = 0) -> dict:
    values = np.array([m["accuracy_uniform_grid"] for m in metrics],dtype=float)
    if not len(values):
        raise ValueError("Need independent replicate metrics")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
    return {"independent_replicates":len(values),"mean_accuracy":float(values.mean()),
            "std_accuracy":float(values.std(ddof=1)) if len(values)>1 else None,
            "bootstrap95_accuracy":np.quantile(means,[.025,.975]).tolist(),
            "bootstrap_unit":"independent simulated parent chromosome"}

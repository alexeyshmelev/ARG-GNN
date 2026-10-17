"""Sampled-example supervision, separate from observed inputs and bp weighting."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import chain

from .data import ImmutableRef


@dataclass(frozen=True)
class SampledFocalTargets:
    """One label per sampled training example, preserving repeated draws.

    Rows refer to the frozen observed readout grid, columns to its query panel.
    TRAIN sampling may use the native method's declared class quotas. Query
    labels never define graph structure or evaluation support. The authorized
    join binds these examples to one TRAIN parent and its sampled registry.
    """
    readout_rows: tuple[int, ...]
    query_columns: tuple[int, ...]
    labels: tuple[int, ...]
    class_order: tuple[str, ...]
    parent_id: str
    split: str
    sampled_registry_ref: ImmutableRef

    def __post_init__(self):
        if not isinstance(self.parent_id, str) or not self.parent_id or self.split != "train":
            raise ValueError("Sampled training labels must belong to a declared TRAIN parent")
        object.__setattr__(self, "sampled_registry_ref", ImmutableRef.from_value(self.sampled_registry_ref))
        for field in ("readout_rows", "query_columns", "labels", "class_order"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        n, c = len(self.labels), len(self.class_order)
        if not n or c < 2 or len(set(self.class_order)) != c:
            raise ValueError("Nonempty sampled examples and explicit endpoint classes required")
        if len(self.readout_rows) != n or len(self.query_columns) != n:
            raise ValueError("Every sampled example needs its own coordinate, query and label")
        if any(type(x) is not int or x < 0 for x in chain(self.readout_rows, self.query_columns)):
            raise ValueError("Observed readout and query indices must be nonnegative integers")
        if any(type(x) is not int or not 0 <= x < c for x in self.labels):
            raise ValueError("Labels must belong to the declared ancestry endpoint")


    def validate_binding(self, *, parent_id: str, split: str, sampled_registry_ref) -> None:
        if (self.parent_id != parent_id or self.split != split
                or self.sampled_registry_ref != ImmutableRef.from_value(sampled_registry_ref)):
            raise ValueError("Sampled labels must join their exact TRAIN parent and sampled registry")


def sampled_focal_loss(logits, targets: SampledFocalTargets):
    """Mean cross-entropy per sampled example, with no physical-span weights.

    Index only the sampled predictions, avoiding a dense tree×query×class target
    tensor. Repeated training draws retain their intended sample weight.
    """
    import torch
    from torch.nn import functional as F
    if (not isinstance(targets, SampledFocalTargets) or logits.ndim != 3
            or logits.shape[2] != len(targets.class_order)
            or max(targets.readout_rows) >= logits.shape[0]
            or max(targets.query_columns) >= logits.shape[1]):
        raise ValueError("Sampled targets must join the complete observed readout and query panel")
    rows = torch.tensor(targets.readout_rows, dtype=torch.long, device=logits.device)
    columns = torch.tensor(targets.query_columns, dtype=torch.long, device=logits.device)
    labels = torch.tensor(targets.labels, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits[rows, columns], labels, reduction="mean")

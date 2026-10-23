"""Stream exact full-parent decoder gradients without recurrent recomputation.

The caller owns the complete differentiable CPU context. This module detaches
only the decoder boundary, saves its complete VJP, then reconnects that VJP in
one upstream backward call. No optimizer update occurs in this module.
"""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F


def need(ok, message):
    if not ok:
        raise ValueError(message)


def geometry_from_spans(spans):
    span = spans.log1p() / math.log(2000001.)
    previous = torch.cat((spans.new_zeros(1), .5 * (spans[1:] + spans[:-1])))
    following = torch.cat((.5 * (spans[1:] + spans[:-1]), spans.new_zeros(1)))
    return torch.stack((span, previous.log1p() / math.log(2000001.),
                        following.log1p() / math.log(2000001.)), -1)


def decode_parent_backward(model, context, geometry, focal_hold, focal_column,
                           labels, *, tile_haplotypes=32, profile=None,
                           resource_check=lambda label: None):
    """Return CE and a CPU context VJP; accumulate decoder parameter gradients.

    All focal rows enter exactly once, and each tile keeps the full sequence.
    Empty focal tiles have exactly zero decoder/context gradient and need no GRU
    call. Their observed input has already contributed to the complete trunk.
    Accumulation changes FP32 reduction order, not the mathematical objective.
    """
    need(context.ndim == 3 and context.device.type == "cpu"
         and context.dtype == torch.float32, "Complete FP32 CPU context required")
    need(type(tile_haplotypes) is int and 1 <= tile_haplotypes <= 128,
         "Explicit bounded decoder tile size required")
    total = int(labels.numel())
    need(total > 0 and labels.dtype == torch.long and labels.device.type == "cpu"
         and focal_hold.shape == focal_column.shape == labels.shape,
         "Aligned complete CPU focal labels and coordinates required")
    need(geometry.shape == (context.shape[0], 3), "Global genomic geometry required")
    need(bool(((focal_hold >= 0) & (focal_hold < context.shape[0])).all())
         and bool(((focal_column >= 0) & (focal_column < context.shape[1])).all()),
         "Every focal coordinate must belong to the complete parent")
    device = next(model.parameters()).device
    stage = profile.stage if profile is not None else lambda label: nullcontext()
    gradient = torch.zeros_like(context, memory_format=torch.contiguous_format)
    loss_total = torch.zeros((), device=device, dtype=torch.float32)
    seen, completed, zero_tiles = 0, 0, 0
    geometry_device = geometry.to(device)
    for first in range(0, context.shape[1], tile_haplotypes):
        last = min(first + tile_haplotypes, context.shape[1])
        rows = torch.nonzero((focal_column >= first) & (focal_column < last),
                             as_tuple=False).flatten()
        resource_check("decoder_tile_begin")
        if not len(rows):
            zero_tiles += 1
            if profile is not None:
                profile.mark("decoder_zero_focal_tile", first=first, last=last,
                             completed_focals=seen, total_focals=total)
            continue
        with stage("decoder_tile_forward"):
            tile = context[:, first:last].detach().to(device).contiguous().requires_grad_(True)
            forward, _ = model.scan_context(tile, geometry_device, direction="forward")
            reverse, _ = model.scan_context(tile, geometry_device, direction="reverse")
            hold = focal_hold[rows].to(device)
            column = (focal_column[rows] - first).to(device)
            logits = model.classifier(torch.cat((tile[hold, column], forward[hold, column],
                                                 reverse[hold, column]), -1))
            tile_loss = F.cross_entropy(logits, labels[rows].to(device), reduction="sum") / total
        need(bool(torch.isfinite(tile_loss)), "Finite complete-parent decoder loss required")
        with stage("decoder_tile_backward"):
            tile_loss.backward()
            need(tile.grad is not None and bool(torch.isfinite(tile.grad).all()),
                 "Finite connected decoder context gradient required")
            gradient[:, first:last].copy_(tile.grad.detach().to("cpu"))
            loss_total.add_(tile_loss.detach())
        seen += len(rows)
        completed += 1
        if profile is not None:
            profile.mark("decoder_tile_complete", first=first, last=last,
                         completed_focals=seen, total_focals=total,
                         completed_decoder_tiles=completed, zero_focal_tiles=zero_tiles)
        del tile, forward, reverse, logits, tile_loss
    need(seen == total, "Every native focal contributes once before any optimizer step")
    return loss_total, gradient, dict(completed_decoder_tiles=completed,
        zero_focal_tiles=zero_tiles, decoder_haplotypes_per_tile=tile_haplotypes,
        native_focals=seen, full_sequence_per_tile=True,
        decoder_recomputed_during_backward=False)

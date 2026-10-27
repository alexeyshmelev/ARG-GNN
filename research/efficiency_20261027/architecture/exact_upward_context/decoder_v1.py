"""Original resident decoder with caller-preallocated VJP; public executor retains CUDA constructor guards."""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need

def decode_model_device(model,context,gradient,geometry,focal_hold,focal_column,
                                    labels,*,tile_haplotypes=32,profile=None,
                                    resource_check=lambda label:None):
    """Original tiled full-history decoder and VJP, held on the model GPU.

    The full differentiable context remains owned by the caller. Each decoder
    tile detaches at this boundary exactly as the original decoder does, and
    returns the complete VJP for one upstream backward. Empty target tiles
    contribute zero gradient. No history is reset and no target is omitted.
    """
    device=next(model.parameters()).device
    need(context.ndim==3 and context.device==device
         and context.dtype==torch.float32,"Complete FP32 model-device context required")
    need(type(tile_haplotypes)is int and 1<=tile_haplotypes<=128,
         "Original bounded decoder tile size required")
    total=int(labels.numel())
    need(total>0 and labels.dtype==torch.long and labels.device.type=="cpu"
         and focal_hold.shape==focal_column.shape==labels.shape,
         "Aligned complete original CPU focal labels and coordinates required")
    need(geometry.shape==(context.shape[0],3),"Global genomic geometry required")
    need(bool(((focal_hold>=0)&(focal_hold<context.shape[0])).all())
         and bool(((focal_column>=0)&(focal_column<context.shape[1])).all()),
         "Every focal coordinate belongs to the complete parent")
    stage=profile.stage if profile is not None else lambda label:nullcontext()
    resource_check("resident_context_decoder_begin")
    need(gradient.shape==context.shape and gradient.device==device and gradient.dtype==torch.float32
         and gradient.is_contiguous() and not gradient.requires_grad
         and gradient.untyped_storage().data_ptr()!=context.untyped_storage().data_ptr(),
         "Separate complete preallocated VJP on model device required")
    loss_total=torch.zeros((),device=device,dtype=torch.float32)
    seen,completed,zero_tiles=0,0,0
    geometry_device=geometry.to(device)
    for first in range(0,context.shape[1],tile_haplotypes):
        last=min(first+tile_haplotypes,context.shape[1])
        rows=torch.nonzero((focal_column>=first)&(focal_column<last),as_tuple=False).flatten()
        resource_check("resident_context_decoder_tile_begin")
        if not len(rows):
            zero_tiles+=1
            if profile is not None:
                profile.mark("decoder_zero_focal_tile",first=first,last=last,
                             completed_focals=seen,total_focals=total)
            continue
        with stage("decoder_tile_forward"):
            tile=context[:,first:last].detach().contiguous().requires_grad_(True)
            forward,_=model.scan_context(tile,geometry_device,direction="forward")
            reverse,_=model.scan_context(tile,geometry_device,direction="reverse")
            hold=focal_hold[rows].to(device)
            column=(focal_column[rows]-first).to(device)
            logits=model.classifier(torch.cat((tile[hold,column],forward[hold,column],
                                               reverse[hold,column]),-1))
            tile_loss=F.cross_entropy(logits,labels[rows].to(device),reduction="sum")/total
        need(bool(torch.isfinite(tile_loss)),"Finite complete-parent decoder loss required")
        with stage("decoder_tile_backward"):
            tile_loss.backward()
            need(tile.grad is not None and bool(torch.isfinite(tile.grad).all()),
                 "Finite connected decoder context gradient required")
            gradient[:,first:last].copy_(tile.grad.detach())
            loss_total.add_(tile_loss.detach())
        seen+=len(rows)
        completed+=1
        if profile is not None:
            profile.mark("decoder_tile_complete",first=first,last=last,
                         completed_focals=seen,total_focals=total,
                         completed_decoder_tiles=completed,zero_focal_tiles=zero_tiles)
        del tile,forward,reverse,logits,tile_loss
    need(seen==total,"Every native focal contributes once before any optimizer step")
    return loss_total,gradient,dict(completed_decoder_tiles=completed,
        zero_focal_tiles=zero_tiles,decoder_haplotypes_per_tile=tile_haplotypes,
        native_focals=seen,full_sequence_per_tile=True,
        decoder_boundary_replayed=False,recurrent_chunks_checkpointed=True,
        context_residency=str(device),VJP_residency=str(device),complete_VJP_preallocated=True)

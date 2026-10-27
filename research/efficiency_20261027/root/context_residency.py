"""Exact context-only GPU residency for isolated paired profiles.

First-sweep upward states, forcing and full genomic history remain in the
original CPU staging path. Only second-sweep leaf context and its decoder VJP
remain on CUDA. This changes storage placement, not model equations or support.
No production binding, optimizer, checkpoint writer or launcher is provided.
"""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from research.scalability_20261024.integration.executors import IntegratedStagedExecution
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import profiled, need


class ResidentContextExecution(IntegratedStagedExecution):
    @profiled
    def _second(self, z, up, value, eligible):
        """Original second-sweep body with its final CPU transfer removed."""
        self.tick("resident_context_second_sweep")
        a,b,lo,hi,_,_,_,_,_,ml,mr=z
        bank=self._block(z)
        base,ug,dg,degree=self._base(bank,self._sites(a,b))
        up=up.to(self.device)
        m=self.model
        channels=self.bundle.episodes.flags.node_flags[lo:hi].to(self.device,dtype=up.dtype)
        common=m.clade_coupling.U(up)
        flag=m.clade_coupling.O(channels[:,0:1]*common*m.clade_coupling.V(up)
            -channels[:,1:2]*common*m.clade_coupling.W(up))
        member_node=(self.bundle.episodes.member_node[ml:mr].long()-lo).to(self.device)
        member=up.index_select(0,member_node)
        ep=m.episode_coupling
        residual=ep.O(ep.U(member)*ep.V(member)+ep.W(member))
        residual=residual*eligible[ml:mr,None].to(self.device,dtype=torch.float32)
        episode=up.new_zeros(up.shape).index_add(0,member_node,residual)
        injection=flag+episode
        cut=self.bundle.cut
        first,last=cut.group_offsets[a],cut.group_offsets[b]
        representatives=(cut.representatives[first:last]-lo).to(self.device)
        delta=m.memory.O(value.to(self.device))
        injection=injection.index_add(0,representatives,delta)
        second_up,second_out=m.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
        return m.leaf_context_v1(bank,second_up,second_out)


def build_executor(model,bundle,metadata,**kwargs):
    """Explicit opt-in and unchanged original constructor/bank checks."""
    return ResidentContextExecution(model,bundle,metadata,**kwargs)


def decode_resident_parent_backward(model,context,geometry,focal_hold,focal_column,
                                    labels,*,tile_haplotypes=32,profile=None,
                                    resource_check=lambda label:None):
    """Original tiled full-history decoder and VJP, held on the model GPU.

    The full differentiable context remains owned by the caller. Each decoder
    tile detaches at this boundary exactly as the original decoder does, and
    returns the complete VJP for one upstream backward. Empty target tiles
    contribute zero gradient. No history is reset and no target is omitted.
    """
    device=next(model.parameters()).device
    need(context.ndim==3 and context.device==device and device.type=="cuda"
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
    gradient=torch.zeros_like(context,memory_format=torch.contiguous_format)
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
        context_residency="CUDA",VJP_residency="CUDA")

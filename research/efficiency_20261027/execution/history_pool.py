"""Exact windowed cut-group mean for the original stable-leaf recurrence.

The recurrence, boundary states and all reverse/forward dependencies remain the
original implementation. Only the parameter-free CPU pooling stage is tiled and
run on the selected device. No production binding or launch occurs here.
"""
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed.stable_leaf_history_v2 import (
    _Direction, NullProfileV1, need)


class _WindowedCutMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, groups, counts, offsets, device, window, memory, tick, profile):
        need(left.device.type == right.device.type == groups.device.type == counts.device.type == "cpu",
             "Unchanged CPU directional history and cut recipes required")
        need(left.shape == right.shape and left.ndim == 3 and groups.shape == left.shape[:2],
             "Aligned complete directional histories and persistent sample groups")
        need(offsets[0] == 0 and offsets[-1] == len(counts) and len(offsets) == left.shape[0]+1,
             "Complete disjoint hold cut groups")
        need(type(window) is int and 1 <= window <= 1024, "Bounded full-sequence pooling windows")
        ctx.save_for_backward(groups, counts)
        ctx.offsets,ctx.device,ctx.window = offsets,torch.device(device),window
        ctx.tick,ctx.profile,ctx.shape = tick,profile,left.shape
        result=left.new_empty((len(counts),left.shape[-1]))
        with profile.stage("history_windowed_device_pool_forward"):
            for a in range(0,len(groups),window):
                b=min(a+window,len(groups));lo,hi=offsets[a],offsets[b]
                tick("history_windowed_pool_forward")
                # Preserve the original .5*(left+right) before the mean. Mean
                # groups are disjoint across holds, so windowing changes no
                # contributing rows or their order within an original group.
                leaf=.5*(left[a:b].to(device)+right[a:b].to(device))
                local=(groups[a:b]-lo).to(device)
                pooled=memory._mean(leaf.flatten(0,1),local.flatten(),counts[lo:hi].to(device),hi-lo)
                result[lo:hi].copy_(pooled.to("cpu"))
        return result

    @staticmethod
    def backward(ctx, gradient):
        groups,counts=ctx.saved_tensors
        grad_history=gradient.new_empty(ctx.shape)
        with ctx.profile.stage("history_windowed_device_pool_backward"):
            for a in range(0,len(groups),ctx.window):
                b=min(a+ctx.window,len(groups));lo,hi=ctx.offsets[a],ctx.offsets[b]
                ctx.tick("history_windowed_pool_backward")
                group_gradient=gradient[lo:hi].to(ctx.device)/counts[lo:hi].to(ctx.device).clamp_min(1.)[:,None]
                local=(groups[a:b]-lo).to(ctx.device)
                block=.5*group_gradient.index_select(0,local.flatten()).view(b-a,ctx.shape[1],ctx.shape[2])
                grad_history[a:b].copy_(block.to("cpu"))
        # Both directional VJPs are identical and read-only consumers. Returning
        # one shared buffer avoids allocating two identical chromosome tensors.
        return grad_history,grad_history,None,None,None,None,None,None,None,None


def windowed_cut_mean(memory,left,right,plan,*,device,holds_per_window=1024,
                      resource_check=lambda label:None,profile=None):
    return _WindowedCutMean.apply(left,right,plan.leaf_groups,plan.group_counts,
        tuple(plan.group_offsets),torch.device(device),holds_per_window,memory,
        resource_check,profile or NullProfileV1())


def stable_leaf_history(memory,forcing,theta,spans,plan,*,device,
                        holds_per_window=1024,resource_check=lambda label:None,profile=None):
    """Original recurrence with windowed pooling; no detach or sequence reset."""
    need(memory.arm=="stable_leaf_memory","Only the frozen stable-leaf arm")
    need(forcing.device.type==theta.device.type==spans.device.type=="cpu"
         and forcing.dtype==theta.dtype==spans.dtype==torch.float32,
         "Original FP32 CPU history interface")
    need(type(holds_per_window)is int and 1<=holds_per_window<=1024,
         "At most1024holds per bounded history window")
    offsets=tuple(plan.group_offsets)
    need(len(offsets)==plan.native_holds+1 and offsets[0]==0
         and offsets[-1]==len(forcing)
         and all(0<b-a<=plan.haplotypes for a,b in zip(offsets,offsets[1:]))
         and plan.leaf_groups.shape==(plan.native_holds,plan.haplotypes)
         and plan.haplotypes==1600 and forcing.shape[1]==8,
         "Exact complete disjoint per-hold cut partition and original1600x8history")
    profile=profile or NullProfileV1()
    def direction(reverse):
        ell=100.+(100000.-100.)*theta.sigmoid()
        rho=torch.exp(-spans/ell);source=-torch.expm1(-spans/ell)
        return _Direction.apply(forcing,rho,source,plan.leaf_groups,offsets,reverse,
                                holds_per_window,torch.device(device),resource_check,profile)
    left=direction(False);right=direction(True)
    read=windowed_cut_mean(memory,left,right,plan,device=device,holds_per_window=holds_per_window,
                           resource_check=resource_check,profile=profile)
    return read-forcing

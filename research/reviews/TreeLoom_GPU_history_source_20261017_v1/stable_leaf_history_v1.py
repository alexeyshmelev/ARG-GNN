"""Same frozen stable-leaf recurrence with bounded device VJP and CPU pooling.

No imports of project loaders, model constructors, resource APIs or launchers.
The whole forcing and readout stay on CPU. A disjoint cut-group slice is
transferred once per window; its gradient is written once into the CPU result.
"""
import torch

def need(ok,message):
    if not ok:raise ValueError(message)

def _window(forcing,rho,source,groups,boundary,reverse):
    n,h=groups.shape;r=forcing.shape[1]
    # One gather backward for the local group slice, rather than T scatters
    # into the complete parent's forcing tensor.
    leaves=forcing.index_select(0,groups.reshape(-1)).reshape(n,h,r)
    leaf_rows=leaves.unbind(0);rho_rows=rho.unbind(0);source_rows=source.unbind(0)
    state=boundary;result=[None]*n
    for j in (range(n-1,-1,-1)if reverse else range(n)):
        force=leaf_rows[j]
        state=force if state is None else rho_rows[j]*state+source_rows[j]*force
        result[j]=state
    return torch.stack(result,0),state

class _Direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,forcing,rho,source,groups,offsets,reverse,window,device,tick,profile):
        t,h=groups.shape;r=forcing.shape[1]
        ctx.save_for_backward(forcing,rho,source,groups)
        ctx.offsets,ctx.reverse,ctx.device=offsets,reverse,device
        ctx.tick,ctx.profile=tick,profile
        windows=[(a,min(t,a+window))for a in range(0,t,window)]
        if reverse:windows.reverse()
        ctx.windows=windows;ctx.boundaries=[]
        result=forcing.new_empty((t,h,r));state=None
        with profile.stage("history_reverse_forward"if reverse else"history_forward_forward"):
            for a,b in windows:
                tick("bounded_GPU_history_forward_window")
                ctx.boundaries.append(None if state is None else state.to("cpu"))
                lo,hi=offsets[a],offsets[b]
                local=forcing[lo:hi].to(device)
                index=(groups[a:b]-lo).to(device)
                out,state=_window(local,rho[a:b].to(device),source[a:b].to(device),
                                  index,state,reverse)
                result[a:b].copy_(out.to("cpu"))
        return result

    @staticmethod
    def backward(ctx,gradient):
        forcing,rho,source,groups=ctx.saved_tensors
        grad_force=torch.zeros_like(forcing)
        grad_rho=torch.zeros_like(rho);grad_source=torch.zeros_like(source)
        carry=None;device=ctx.device
        with ctx.profile.stage("history_reverse_backward"if ctx.reverse else"history_forward_backward"):
            for k in range(len(ctx.windows)-1,-1,-1):
                a,b=ctx.windows[k];lo,hi=ctx.offsets[a],ctx.offsets[b]
                ctx.tick("bounded_GPU_history_backward_window")
                with torch.enable_grad():
                    f=forcing[lo:hi].detach().to(device).requires_grad_(True)
                    p=rho[a:b].detach().to(device).requires_grad_(True)
                    s=source[a:b].detach().to(device).requires_grad_(True)
                    boundary=ctx.boundaries[k]
                    incoming=None if boundary is None else boundary.to(device).detach().requires_grad_(True)
                    index=(groups[a:b]-lo).to(device)
                    out,_=_window(f,p,s,index,incoming,ctx.reverse)
                    g=gradient[a:b].to(device).clone()
                    # The next window's incoming state is this window's last
                    # traversal state. Add its VJP before local reverse-mode.
                    if carry is not None:g[0 if ctx.reverse else -1].add_(carry)
                    inputs=(f,p,s)if incoming is None else(f,p,s,incoming)
                    vjp=torch.autograd.grad(out,inputs,g,allow_unused=True)
                grad_force[lo:hi].copy_(vjp[0].to("cpu"))
                if vjp[1]is not None:grad_rho[a:b].copy_(vjp[1].to("cpu"))
                if vjp[2]is not None:grad_source[a:b].copy_(vjp[2].to("cpu"))
                carry=None if incoming is None else vjp[3]
        return grad_force,grad_rho,grad_source,None,None,None,None,None,None,None

class NullProfileV1:
    class _Stage:
        def __enter__(self):return self
        def __exit__(self,*args):return False
    def stage(self,label):return self._Stage()

def stable_leaf_history_v1(memory,forcing,theta,spans,plan,*,device,
                           holds_per_window=128,resource_check=lambda label:None,profile=None):
    """Original U/O remain outside. No detach of trainable forcing or ell."""
    need(memory.arm=="stable_leaf_memory","Only the frozen stable-leaf arm")
    need(forcing.device.type==theta.device.type==spans.device.type=="cpu"
         and forcing.dtype==theta.dtype==spans.dtype==torch.float32,
         "Original FP32 CPU history interface")
    need(type(holds_per_window)is int and 1<=holds_per_window<=128,
         "At most128holds per bounded history window")
    offsets=tuple(plan.group_offsets)
    need(len(offsets)==plan.native_holds+1 and offsets[0]==0
         and offsets[-1]==len(forcing)
         and all(0<b-a<=plan.haplotypes for a,b in zip(offsets,offsets[1:]))
         and plan.leaf_groups.shape==(plan.native_holds,plan.haplotypes)
         and plan.haplotypes==1600 and forcing.shape[1]==8,
         "Exact complete disjoint per-hold cut-group partition; bounded <=1600groups/hold")
    profile=profile or NullProfileV1()
    def direction(reverse):
        # Recompute separately, as in the original two _direction calls.
        ell=100.+(100000.-100.)*theta.sigmoid()
        rho=torch.exp(-spans/ell);source=-torch.expm1(-spans/ell)
        return _Direction.apply(forcing,rho,source,plan.leaf_groups,offsets,reverse,
                                holds_per_window,torch.device(device),resource_check,profile)
    left=direction(False);right=direction(True)
    with profile.stage("history_original_CPU_pool"):
        leaf=.5*(left+right)
        read=memory._mean(leaf.flatten(0,1),plan.leaf_groups.flatten(),
                          plan.group_counts,forcing.shape[0])
        return read-forcing

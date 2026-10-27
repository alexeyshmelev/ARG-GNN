"""Prepared first/forcing forward fusion with the unchanged explicit backward.

No production registration, launcher or GPU execution. Execution owns later
integration. Full CPU up/out/forcing/readout boundaries are retained unchanged.
"""
import math
import torch
from research.efficiency_20261027.execution.explicit_vjp import _ExplicitTrunk, ExplicitVJPExecution
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need
from research.benchmark_rebuild_20261024.american.treeloom_speed.stable_leaf_history_v2 import stable_leaf_history_v2


def first_with_forcing(executor,z):
    """Same first sweep; select on its device before lossless CPU boundary copies."""
    need(not torch.is_grad_enabled(),'Fusion is restricted to the numerical custom forward')
    executor.tick('bounded_first_sweep')
    bank=executor._block(z)
    base,ug,dg,degree=executor._base(bank,executor._sites(z[0],z[1]))
    up,out=executor.model.sweep_tree_state_v1(bank,base,ug,dg,degree)
    cut=executor.bundle.cut;a,b,lo=z[:3]
    first,last=cut.group_offsets[a],cut.group_offsets[b]
    index=(cut.representatives[first:last]-lo).to(executor.device)
    selected=up.index_select(0,index)
    # Same binary FP32 sum as the original forcing; selected is a fresh gather
    # and aliases neither first-sweep boundary. No parameter/boundary mutation.
    selected.add_(out.index_select(0,index))
    forcing=executor.model.memory.U(selected)
    minimum,maximum=torch.aminmax(up)
    absmax=torch.maximum(-minimum,maximum)
    return up.to('cpu'),out.to('cpu'),forcing.to('cpu'),absmax


def finite_own_episode_from_max(executor,maximum):
    """Original eligibility/proof with one already reduced upward abs maximum."""
    need(not torch.is_grad_enabled() and maximum.shape==() and maximum.device.type=='cpu',
         'One numerical CPU scalar from all device block maxima')
    episodes=executor.bundle.episodes;m=executor.model.episode_coupling
    weights=episodes.hold_span.index_select(0,episodes.member_hold.long())[:,None].to(executor.device)
    index=episodes.member_episode.long().to(executor.device)
    totals=m.aggregation(weights,index=index,dim_size=episodes.episode_size.numel(),dim=0)
    eligible=episodes.episode_size.to(executor.device).index_select(0,index)>=2
    denominator=totals.index_select(0,index)-weights
    minimum=float(denominator[eligible].min()) if bool(eligible.any()) else 1.
    maximum=float(maximum)
    need(math.isfinite(maximum) and minimum>0 and math.isfinite(minimum)
         and 4*maximum*float(executor.bank.spans.double().sum())/minimum<torch.finfo(torch.float32).max,
         'Original span-weighted foreign context provably finite before own_state0 shortcut')
    return eligible.to('cpu')


class _ForwardFusionTrunk(torch.autograd.Function):
    # Exactly the existing function object, same saved order and ctx contract.
    backward=staticmethod(_ExplicitTrunk.backward)

    @staticmethod
    def forward(ctx,executor,*parameters):
        model,bank,cut=executor.model,executor.bank,executor.bundle.cut
        need(tuple(id(p) for p in parameters)==tuple(id(p) for p in model.parameters()),
             'Original parameter identity/order required')
        need(not torch.is_grad_enabled(),'No per-block forward autograd graph')
        executor.tick('explicit_full_parent_forward_begin')
        with executor.profile.stage('explicit_fused_first_forcing_forward'):
            blocks=[first_with_forcing(executor,z) for z in executor.layout]
            upward=tuple(block[0] for block in blocks)
            outside=tuple(block[1] for block in blocks)
            forcing=torch.cat([block[2] for block in blocks])
            maxima=torch.stack([block[3] for block in blocks])
            maximum=maxima.amax().to('cpu')
            del blocks,maxima
            eligible=finite_own_episode_from_max(executor,maximum)
        with executor.profile.stage('explicit_history_forward'):
            theta=model.memory.ell_theta.detach().to('cpu')
            readout=stable_leaf_history_v2(model.memory,forcing,theta,bank.spans,cut,
                device=executor.device,holds_per_window=executor.history_window,
                resource_check=executor.tick,profile=executor.profile)
        with executor.profile.stage('explicit_second_blocks_forward'):
            contexts=[]
            for z,up in zip(executor.layout,upward):
                first,last=cut.group_offsets[z[0]],cut.group_offsets[z[1]]
                contexts.append(executor._second(z,up,readout[first:last],eligible))
            context=torch.cat(contexts,dim=0)
            del contexts
        need(context.device.type=='cpu' and context.dtype==torch.float32,
             'Original complete FP32 CPU context interface')
        need(all(t.device.type=='cpu' and not t.requires_grad
                 for t in (forcing,readout,eligible,*upward,*outside)),
             'Original immutable CPU numerical boundaries retained')
        ctx.executor=executor;ctx.block_count=len(upward);ctx.parameter_count=len(parameters)
        ctx.parameter_ids=tuple(id(p) for p in parameters)
        ctx.save_for_backward(forcing,readout,eligible,*upward,*outside,*parameters)
        ctx.set_materialize_grads(False)
        groups=cut.group_offsets[-1]
        executor.explicit_receipt=dict(forward_blocks=len(upward),second_backward_blocks=0,
            forcing_backward_blocks=0,first_backward_blocks=0,full_history_backward_calls=0,
            retained_CPU_boundary_bytes=sum(t.numel()*t.element_size() for t in (forcing,readout,eligible,*upward,*outside)),
            parameters_returned_through_autograd=True,parameter_grad_mutated_inside=False,
            learned_states_reused_across_updates=False,history_gradient_truncated=False,
            backward_block_order='reverse native block order',CUDA_boundary_allocated_max_bytes=0,
            forward_first_forcing_fused=True,original_backward_function_identity=True,
            forward_selected_H2D_float_bytes_avoided=2*groups*32*4,
            forward_new_H2D_index_bytes=groups*8,
            upward_absmax_device_reduction=True,forward_upward_CPU_abs_scan_bytes_avoided=bank.parent.numel()*32*4,
            whole_CPU_upward_outside_boundaries_preserved=True)
        return context


class ForwardFusedExecution(ExplicitVJPExecution):
    """Unregistered opt-in candidate; inherits all original constructor guards."""
    def context_forward(self):
        with self.profile.stage('explicit_fused_context_forward'):
            return _ForwardFusionTrunk.apply(self,*tuple(self.model.parameters()))

"""European-only explicit VJP and device boundary fusion; no launch or registration.

The original European 128-hold layout, site reduction, six-class decoder and
952-leaf stable history remain. This is a separate ordinary-source adaptation
of the reviewed boundary algorithm, not an American initializer or monkeypatch.
"""
import math
import torch
from research.efficiency_20261027.architecture.european_cohort_candidate.execution import CohortStagedExecution
from research.efficiency_20261027.architecture.european_cohort_candidate.model import need
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_observations import _general_history


def first_device(executor, block):
    executor.tick('European_boundary_first_sweep')
    bank = executor._block(block)
    base, up_gate, down_gate, degree = executor._base(bank, executor._sites(block[0], block[1]))
    return executor.model.sweep_tree_state_v1(bank, base, up_gate, down_gate, degree)


def forcing_device(executor, block, upward, outside):
    cut = executor.bundle.cut
    first, last = cut.group_offsets[block[0]], cut.group_offsets[block[1]]
    index = (cut.representatives[first:last] - block[2]).to(executor.device)
    return executor.model.memory.U(upward.index_select(0, index) + outside.index_select(0, index))


def finite_own_episode(executor, maximum):
    episodes, module = executor.bundle.episodes, executor.model.episode_coupling
    weights = episodes.hold_span.index_select(0, episodes.member_hold.long())[:, None].to(executor.device)
    index = episodes.member_episode.long().to(executor.device)
    totals = module.aggregation(weights, index=index, dim_size=episodes.episode_size.numel(), dim=0)
    eligible = episodes.episode_size.to(executor.device).index_select(0, index) >= 2
    denominator = totals.index_select(0, index) - weights
    minimum = float(denominator[eligible].min()) if bool(eligible.any()) else 1.
    maximum = float(maximum)
    need(math.isfinite(maximum) and minimum > 0 and math.isfinite(minimum)
         and 4 * maximum * float(executor.bank.spans.double().sum()) / minimum < torch.finfo(torch.float32).max,
         'Original finite foreign-context proof before own-state shortcut')
    return eligible.cpu()


def history(executor, forcing, theta):
    return _general_history(executor.model.memory, forcing, theta, executor.bank.spans,
        executor.bundle.cut, device=executor.device, holds_per_window=executor.history_window,
        resource_check=executor.tick, profile=executor.profile)


class _EuropeanBoundaryTrunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, executor, *parameters):
        need(tuple(id(p) for p in parameters) == tuple(id(p) for p in executor.model.parameters()),
             'Original49 parameter identities and order')
        need(not torch.is_grad_enabled() and executor.bank.sample_index.shape[1] == 952
             and executor.bundle.cut.haplotypes == 952, 'European952 complete numerical boundary')
        upward, pieces, maxima = [], [], []
        with executor.profile.stage('European_boundary_first_forcing'):
            for block in executor.layout:
                up, outside = first_device(executor, block)
                pieces.append(forcing_device(executor, block, up, outside).cpu())
                upward.append(up.cpu())
                lower, upper = torch.aminmax(up)
                maxima.append(torch.maximum(-lower, upper))
                del up, outside
            forcing = torch.cat(pieces)
            eligible = finite_own_episode(executor, torch.stack(maxima).amax().cpu())
            del pieces, maxima
        with executor.profile.stage('European_boundary_full_history'):
            readout = history(executor, forcing, executor.model.memory.ell_theta.detach().cpu())
        with executor.profile.stage('European_boundary_second'):
            contexts = []
            for block, up in zip(executor.layout, upward):
                first, last = executor.bundle.cut.group_offsets[block[0]], executor.bundle.cut.group_offsets[block[1]]
                contexts.append(executor._second(block, up, readout[first:last], eligible))
            context = torch.cat(contexts)
            del contexts
        need(context.shape == (executor.bundle.cut.native_holds, 952, 32)
             and context.device.type == 'cpu' and context.dtype == torch.float32,
             'All original European holds and samples in FP32 CPU context')
        need(all(value.device.type == 'cpu' and not value.requires_grad
                 for value in (forcing, readout, eligible, *upward)), 'Immutable numerical CPU boundaries')
        ctx.executor, ctx.block_count = executor, len(upward)
        ctx.parameter_ids = tuple(id(p) for p in parameters)
        ctx.save_for_backward(forcing, readout, eligible, *upward, *parameters)
        ctx.set_materialize_grads(False)
        executor.explicit_receipt = dict(forward_blocks=len(upward), second_backward_blocks=0,
            forcing_backward_blocks=0, first_backward_blocks=0, full_history_backward_calls=0,
            retained_CPU_boundary_bytes=sum(v.numel()*v.element_size() for v in (forcing,readout,eligible,*upward)),
            outside_CPU_boundary_retained=False, parameter_grad_mutated_inside=False,
            history_gradient_truncated=False, learned_states_reused_across_updates=False,
            sample_columns=952, full_context_on_CPU=True)
        return context

    @staticmethod
    def backward(ctx, gradient):
        need(not torch.is_grad_enabled(), 'First-order training VJP')
        executor, count = ctx.executor, ctx.block_count
        saved = ctx.saved_tensors
        forcing, readout, eligible = saved[:3]
        upward, saved_parameters = saved[3:3+count], saved[3+count:]
        parameters = tuple(executor.model.parameters())
        need(tuple(id(p) for p in parameters) == ctx.parameter_ids and len(saved_parameters) == len(parameters) == 49,
             'Unchanged parameter identities through complete backward')
        need(gradient is not None and gradient.device.type == 'cpu' and gradient.dtype == torch.float32
             and gradient.shape == (executor.bundle.cut.native_holds, 952, 32), 'Complete European context VJP')
        parameter_vjp, direct_up_vjp = [None]*len(parameters), [None]*count
        readout_vjp = torch.empty_like(readout)
        cut = executor.bundle.cut
        reverse = range(count-1, -1, -1)
        def accumulate(values):
            for index, value in enumerate(values):
                if value is not None:
                    if parameter_vjp[index] is None:
                        parameter_vjp[index] = value.detach()
                    else:
                        parameter_vjp[index].add_(value.detach())
        with executor.profile.stage('European_boundary_second_backward'):
            for index in reverse:
                block = executor.layout[index]
                first, last = cut.group_offsets[block[0]], cut.group_offsets[block[1]]
                executor.tick('European_boundary_second_backward')
                with torch.enable_grad():
                    up_leaf = upward[index].detach().requires_grad_(True)
                    read_leaf = readout[first:last].detach().requires_grad_(True)
                    output = executor._second(block, up_leaf, read_leaf, eligible)
                    values = torch.autograd.grad(output, (up_leaf,read_leaf,*parameters), gradient[block[0]:block[1]],
                                                 allow_unused=True, create_graph=False)
                need(values[0] is not None and values[1] is not None, 'Both second-sweep numerical boundaries connected')
                direct_up_vjp[index] = values[0].detach()
                readout_vjp[first:last].copy_(values[1])
                accumulate(values[2:])
                executor.explicit_receipt['second_backward_blocks'] += 1
                del up_leaf, read_leaf, output, values
        with executor.profile.stage('European_boundary_history_backward'):
            executor.tick('European_boundary_complete_history_backward')
            with torch.enable_grad():
                force_leaf = forcing.detach().requires_grad_(True)
                theta_leaf = executor.model.memory.ell_theta.detach().cpu().requires_grad_(True)
                result = history(executor, force_leaf, theta_leaf)
                force_vjp, theta_vjp = torch.autograd.grad(result, (force_leaf,theta_leaf), readout_vjp, create_graph=False)
            theta_index = next(i for i,p in enumerate(parameters) if p is executor.model.memory.ell_theta)
            values = [None]*len(parameters)
            values[theta_index] = theta_vjp.to(executor.device)
            accumulate(values)
            executor.explicit_receipt['full_history_backward_calls'] += 1
            del force_leaf, theta_leaf, result, readout_vjp, theta_vjp, values
        with executor.profile.stage('European_boundary_first_forcing_backward'):
            for index in reverse:
                block = executor.layout[index]
                first, last = cut.group_offsets[block[0]], cut.group_offsets[block[1]]
                executor.tick('European_boundary_first_forcing_backward')
                with torch.enable_grad():
                    first_up, first_out = first_device(executor, block)
                    up_leaf = first_up.detach().requires_grad_(True)
                    out_leaf = first_out.detach().requires_grad_(True)
                    force = forcing_device(executor, block, up_leaf, out_leaf)
                    values = torch.autograd.grad(force, (up_leaf,out_leaf,*parameters), force_vjp[first:last].to(executor.device),
                                                 allow_unused=True, create_graph=False)
                need(values[0] is not None and values[1] is not None, 'Both forcing derivatives connected')
                total_up = direct_up_vjp[index].to(executor.device)
                total_up.add_(values[0])
                total_out = values[1].detach()
                accumulate(values[2:])
                executor.explicit_receipt['forcing_backward_blocks'] += 1
                del up_leaf, out_leaf, force, values
                with torch.enable_grad():
                    values = torch.autograd.grad((first_up,first_out), parameters, (total_up,total_out),
                                                 allow_unused=True, create_graph=False)
                accumulate(values)
                executor.explicit_receipt['first_backward_blocks'] += 1
                direct_up_vjp[index] = None
                del first_up, first_out, total_up, total_out, values
        need(all(executor.explicit_receipt[key] == count for key in
                 ('second_backward_blocks','forcing_backward_blocks','first_backward_blocks'))
             and executor.explicit_receipt['full_history_backward_calls'] == 1, 'Every block and complete history differentiated')
        executor.explicit_receipt['connected_trunk_parameters'] = sum(value is not None for value in parameter_vjp)
        return (None,*parameter_vjp)


class EuropeanBoundaryExecution(CohortStagedExecution):
    def __init__(self, model, bundle, metadata, **kwargs):
        need(kwargs.get('holds_per_block',128) == 128 and kwargs.get('history_holds_per_window',128) == 128,
             'Fixed original European128 hold partitions and history window')
        super().__init__(model, bundle, metadata, **kwargs)

    def context_forward(self):
        return _EuropeanBoundaryTrunk.apply(self, *tuple(self.model.parameters()))

"""Separate upward/direct-VJP residency candidate over qualified boundary fusion.

CPU context, forcing, readout and complete history stay unchanged. Upward
blocks remain on the compute device; direct upward gradients grow blockwise
and are consumed in place. No full-bank concatenation or duplicate GPU bank.
No launch, optimizer mutation, production binding or GPU qualification.
"""
import torch
from research.efficiency_20261027.execution.explicit_vjp import ExplicitVJPExecution
from research.efficiency_20261027.architecture.exact_forward_fusion.forward_fusion import finite_own_episode_from_max
from research.efficiency_20261027.architecture.exact_boundary_fusion.boundary_fusion import first_device, forcing_device
from .memory_v1 import residency_plan, ResidencyLedger, admit_device, DEFAULT_SCRATCH_BYTES
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need
from research.benchmark_rebuild_20261024.american.treeloom_speed.stable_leaf_history_v2 import stable_leaf_history_v2


class _UpwardResidentTrunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, executor, *parameters):
        model, bank, cut = executor.model, executor.bank, executor.bundle.cut
        need(tuple(id(p) for p in parameters) == tuple(id(p) for p in model.parameters()),
             'Original parameter identity/order required')
        need(not torch.is_grad_enabled(), 'No per-block forward autograd graphs')
        plan = residency_plan(executor)
        ledger = ResidencyLedger(plan)
        admission = executor.admit_residency(plan, 'before_forward', ledger.live_bytes, ledger.future_bytes)
        executor.tick('explicit_full_parent_forward_begin')
        with executor.profile.stage('boundary_fused_first_forcing_forward'):
            upward, pieces, maxima = [], [], []
            for z in executor.layout:
                up, out = first_device(executor, z)
                pieces.append(forcing_device(executor, z, up, out).to('cpu'))
                need(up.device == executor.device and up.dtype == torch.float32
                     and up.shape == (z[3]-z[2], 32), 'Exact native device upward block')
                upward.append(up.detach())
                ledger.add_upward(up)
                minimum, maximum = torch.aminmax(up)
                maxima.append(torch.maximum(-minimum, maximum))
                del up, out
                executor.admit_residency(plan, 'first_block_released', ledger.live_bytes, ledger.future_bytes)
            upward = tuple(upward)
            ledger.require_upward_complete()
            forcing = torch.cat(pieces)
            maximum = torch.stack(maxima).amax().to('cpu')
            del pieces, maxima
            eligible = finite_own_episode_from_max(executor, maximum)
        with executor.profile.stage('explicit_history_forward'):
            theta = model.memory.ell_theta.detach().to('cpu')
            readout = stable_leaf_history_v2(model.memory, forcing, theta, bank.spans, cut,
                device=executor.device, holds_per_window=executor.history_window,
                resource_check=executor.tick, profile=executor.profile)
        with executor.profile.stage('explicit_second_blocks_forward'):
            contexts = []
            for z, up in zip(executor.layout, upward):
                first, last = cut.group_offsets[z[0]], cut.group_offsets[z[1]]
                contexts.append(executor._second(z, up, readout[first:last], eligible))
            context = torch.cat(contexts, dim=0)
            del contexts
        need(context.device.type == 'cpu' and context.dtype == torch.float32,
             'Original complete FP32 CPU context interface')
        need(all(t.device.type == 'cpu' and not t.requires_grad
                 for t in (forcing, readout, eligible)),
             'Forcing, readout and eligibility retain the CPU interface')
        need(all(t.device == executor.device and not t.requires_grad for t in upward),
             'Only detached upward blocks retained on compute device')
        ctx.residency_ledger = ledger
        ctx.residency_plan = plan
        ctx.executor = executor
        ctx.block_count = len(upward)
        ctx.parameter_count = len(parameters)
        ctx.parameter_ids = tuple(id(p) for p in parameters)
        ctx.save_for_backward(forcing, readout, eligible, *upward, *parameters)
        ctx.set_materialize_grads(False)
        executor.explicit_receipt = dict(
            forward_blocks=len(upward), second_backward_blocks=0,
            forcing_backward_blocks=0, first_backward_blocks=0,
            full_history_backward_calls=0,
            retained_CPU_boundary_bytes=sum(t.numel()*t.element_size()
                for t in (forcing, readout, eligible)),
            retained_device_upward_bytes=ledger.upward_bytes,
            residency_plan=plan, residency_admission=admission,
            residency_lifetime=ledger.snapshot(),
            outside_CPU_boundary_retained=False,
            parameters_returned_through_autograd=True,
            parameter_grad_mutated_inside=False,
            learned_states_reused_across_updates=False,
            history_gradient_truncated=False,
            backward_block_order='reverse native block order',
            parameter_contribution_order='all second blocks reversed, complete-history theta, then forcing followed by first for each reversed block',
            CUDA_boundary_allocated_max_bytes=0,
            first_forcing_forward_device_fused=True,
            first_forcing_backward_device_fused=True,
            direct_up_CPU_retained_until_history_VJP=False,
            direct_up_device_retained_until_history_VJP=True,
            whole_GPU_bank_concatenation=False,
            final_context_residency="CPU", full_history_truncated=False,
            removed_block_construction_calls=0)
        return context

    @staticmethod
    def backward(ctx, gradient):
        need(not torch.is_grad_enabled(), 'First-order training VJPs only')
        executor = ctx.executor
        ledger, plan = ctx.residency_ledger, ctx.residency_plan
        saved = ctx.saved_tensors
        count = ctx.block_count
        forcing, readout, eligible = saved[:3]
        upward = saved[3:3+count]
        saved_parameters = saved[3+count:]
        parameters = tuple(executor.model.parameters())
        need(len(saved_parameters) == len(parameters) == ctx.parameter_count
             and tuple(id(p) for p in parameters) == ctx.parameter_ids,
             'Unchanged original parameter identity/order at backward')
        need(gradient is not None and gradient.device.type == 'cpu'
             and gradient.shape == (executor.bundle.cut.native_holds, 1600, 32),
             'Complete original context VJP required')
        cut, model = executor.bundle.cut, executor.model
        parameter_vjp = [None] * len(parameters)
        direct_up_vjp = [None] * count
        readout_vjp = torch.empty_like(readout)
        reverse = range(count-1, -1, -1)

        def accumulate(values):
            for index, value in enumerate(values):
                if value is None:
                    continue
                value = value.detach()
                if parameter_vjp[index] is None:
                    parameter_vjp[index] = value
                else:
                    parameter_vjp[index].add_(value)

        def boundary(stage):
            executor.tick(stage)
            executor.admit_residency(plan, stage, ledger.live_bytes, ledger.future_bytes)
            executor.explicit_receipt['residency_lifetime'] = ledger.snapshot()
            if executor.device.type == 'cuda':
                executor.explicit_receipt['CUDA_boundary_allocated_max_bytes'] = max(
                    executor.explicit_receipt['CUDA_boundary_allocated_max_bytes'],
                    torch.cuda.memory_allocated(executor.device))

        with executor.profile.stage('explicit_second_blocks_backward'):
            for index in reverse:
                z = executor.layout[index]
                first, last = cut.group_offsets[z[0]], cut.group_offsets[z[1]]
                boundary('explicit_second_block_backward_begin')
                with torch.enable_grad():
                    up_leaf = upward[index].detach().requires_grad_(True)
                    read_leaf = readout[first:last].detach().requires_grad_(True)
                    output = executor._second(z, up_leaf, read_leaf, eligible)
                    values = torch.autograd.grad(output, (up_leaf, read_leaf, *parameters),
                        gradient[z[0]:z[1]], allow_unused=True, create_graph=False)
                need(values[0] is not None and values[1] is not None,
                     'Both direct upward and readout derivatives remain connected')
                need(values[0].device == executor.device
                     and values[0].dtype == torch.float32
                     and values[0].shape == upward[index].shape,
                     "Direct upward VJP remains on its original compute device")
                direct_up_vjp[index] = values[0].detach()
                ledger.add_direct(direct_up_vjp[index])
                readout_vjp[first:last].copy_(values[1])
                accumulate(values[2:])
                executor.explicit_receipt['second_backward_blocks'] += 1
                del up_leaf, read_leaf, output, values
                boundary('explicit_second_block_backward_released')

        ledger.require_both_complete()
        with executor.profile.stage('explicit_complete_history_backward'):
            boundary('explicit_complete_history_backward_begin')
            with torch.enable_grad():
                force_leaf = forcing.detach().requires_grad_(True)
                theta_leaf = model.memory.ell_theta.detach().to('cpu').requires_grad_(True)
                history_readout = stable_leaf_history_v2(model.memory, force_leaf, theta_leaf,
                    executor.bank.spans, cut, device=executor.device,
                    holds_per_window=executor.history_window,
                    resource_check=executor.tick, profile=executor.profile)
                force_vjp, theta_vjp = torch.autograd.grad(history_readout,
                    (force_leaf, theta_leaf), readout_vjp, create_graph=False)
            theta_index = next(i for i, p in enumerate(parameters) if p is model.memory.ell_theta)
            theta_contribution = [None] * len(parameters)
            theta_contribution[theta_index] = theta_vjp.to(executor.device)
            accumulate(theta_contribution)
            executor.explicit_receipt['full_history_backward_calls'] += 1
            del force_leaf, theta_leaf, history_readout, readout_vjp, theta_vjp, theta_contribution
            boundary('explicit_complete_history_backward_released')

        with executor.profile.stage('boundary_fused_forcing_and_first_backward'):
            for index in reverse:
                z = executor.layout[index]
                first, last = cut.group_offsets[z[0]], cut.group_offsets[z[1]]
                boundary('boundary_fused_first_recompute_begin')
                with torch.enable_grad():
                    first_up, first_out = first_device(executor, z)
                    # Detached device leaves keep U's parameter VJP separate.
                    up_leaf = first_up.detach().requires_grad_(True)
                    out_leaf = first_out.detach().requires_grad_(True)
                    local_force = forcing_device(executor, z, up_leaf, out_leaf)
                    values = torch.autograd.grad(local_force, (up_leaf, out_leaf, *parameters),
                        force_vjp[first:last].to(executor.device),
                        allow_unused=True, create_graph=False)
                need(values[0] is not None and values[1] is not None,
                     'Both first-sweep forcing derivatives remain connected')
                total_up = direct_up_vjp[index]
                need(total_up.device == executor.device, "No direct-upward gradient transfer")
                total_up.add_(values[0])
                total_out = values[1].detach()
                # Same ordering as original: U contribution, then first sweep.
                accumulate(values[2:])
                executor.explicit_receipt['forcing_backward_blocks'] += 1
                del up_leaf, out_leaf, local_force, values
                boundary('boundary_fused_first_VJP_begin')
                with torch.enable_grad():
                    values = torch.autograd.grad((first_up, first_out), parameters,
                        (total_up, total_out), allow_unused=True, create_graph=False)
                accumulate(values)
                executor.explicit_receipt['first_backward_blocks'] += 1
                ledger.release_direct(direct_up_vjp[index])
                direct_up_vjp[index] = None
                del first_up, first_out, total_up, total_out, values
                boundary('boundary_fused_first_block_released')
        need(all(executor.explicit_receipt[key] == count for key in
            ('second_backward_blocks', 'forcing_backward_blocks', 'first_backward_blocks'))
            and executor.explicit_receipt['full_history_backward_calls'] == 1,
            'Every native tree block and complete history differentiated exactly once')
        ledger.require_direct_released()
        executor.explicit_receipt['residency_lifetime'] = ledger.snapshot()
        executor.explicit_receipt['connected_trunk_parameters'] = sum(g is not None for g in parameter_vjp)
        return (None, *parameter_vjp)


class UpwardResidentExecution(ExplicitVJPExecution):
    """Unregistered opt-in candidate; inherited initialization guards unchanged."""
    resident_scratch_bytes = DEFAULT_SCRATCH_BYTES

    def admit_residency(self, plan, stage, live_bank_bytes, future_bank_bytes):
        return admit_device(self, plan, stage, live_bank_bytes, future_bank_bytes)

    def context_forward(self):
        with self.profile.stage('boundary_fused_context_forward'):
            return _UpwardResidentTrunk.apply(self, *tuple(self.model.parameters()))

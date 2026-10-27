"""Exact explicit trunk VJP with final context on the model device.

The original integrated equations and complete bidirectional history are reused.
CPU first-state/history boundaries replace the global copy graph; final leaf
context and its externally supplied VJP remain on the model device.
The custom backward returns parameter VJPs; it never mutates parameter.grad,
clips, steps an optimizer, truncates history or caches learned state for reuse.
"""
import torch
from research.efficiency_20261027.root.context_residency import ResidentContextExecution
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need
from research.benchmark_rebuild_20261024.american.treeloom_speed.stable_leaf_history_v2 import stable_leaf_history_v2


class _ExplicitResidentTrunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, executor, *parameters):
        model, bank, cut = executor.model, executor.bank, executor.bundle.cut
        need(tuple(id(p) for p in parameters) == tuple(id(p) for p in model.parameters()),
             "Original parameter identity/order required")
        need(not torch.is_grad_enabled(), "Custom trunk forward must not retain per-block autograd graphs")
        executor.tick("explicit_full_parent_forward_begin")
        with executor.profile.stage("explicit_first_blocks_forward"):
            pairs = [executor._first(z) for z in executor.layout]
            upward = tuple(pair[0] for pair in pairs)
            outside = tuple(pair[1] for pair in pairs)
            del pairs
            eligible = executor._finite_own_episode(upward)
        with executor.profile.stage("explicit_forcing_forward"):
            pieces = [executor._forcing(z, up, out)
                for z, up, out in zip(executor.layout, upward, outside)]
            forcing = torch.cat(pieces)
            del pieces
        with executor.profile.stage("explicit_history_forward"):
            theta = model.memory.ell_theta.detach().to("cpu")
            readout = stable_leaf_history_v2(model.memory, forcing, theta, bank.spans, cut,
                device=executor.device, holds_per_window=executor.history_window,
                resource_check=executor.tick, profile=executor.profile)
        with executor.profile.stage("explicit_second_blocks_forward"):
            contexts = []
            for z, up in zip(executor.layout, upward):
                first, last = cut.group_offsets[z[0]], cut.group_offsets[z[1]]
                contexts.append(executor._second(z, up, readout[first:last], eligible))
            context = torch.cat(contexts, dim=0)
            del contexts
        need(context.device == executor.device and context.dtype == torch.float32,
             "Complete FP32 context on the original model device")
        need(all(t.device.type == "cpu" and not t.requires_grad
                 for t in (forcing, readout, eligible, *upward, *outside)),
             "Only immutable CPU numerical boundaries retained by the trunk")
        ctx.executor = executor
        ctx.block_count = len(upward)
        ctx.parameter_count = len(parameters)
        ctx.parameter_ids = tuple(id(p) for p in parameters)
        ctx.save_for_backward(forcing, readout, eligible, *upward, *outside, *parameters)
        ctx.set_materialize_grads(False)
        executor.explicit_receipt = dict(forward_blocks=len(upward), second_backward_blocks=0,
            forcing_backward_blocks=0, first_backward_blocks=0, full_history_backward_calls=0,
            retained_CPU_boundary_bytes=sum(t.numel()*t.element_size()
                for t in (forcing, readout, eligible, *upward, *outside)),
            context_residency=str(context.device),
            complete_context_bytes=context.numel()*context.element_size(),
            parameters_returned_through_autograd=True, parameter_grad_mutated_inside=False,
            learned_states_reused_across_updates=False, history_gradient_truncated=False,
            backward_block_order="reverse native block order",
            CUDA_boundary_allocated_max_bytes=0)
        return context

    @staticmethod
    def backward(ctx, gradient):
        need(not torch.is_grad_enabled(), "This qualification implements first-order training VJPs only")
        executor = ctx.executor
        saved = ctx.saved_tensors
        count = ctx.block_count
        forcing, readout, eligible = saved[:3]
        upward = saved[3:3+count]
        outside = saved[3+count:3+2*count]
        saved_parameters = saved[3+2*count:]
        parameters = tuple(executor.model.parameters())
        need(len(saved_parameters) == len(parameters) == ctx.parameter_count
             and tuple(id(p) for p in parameters) == ctx.parameter_ids,
             "Unchanged original parameter identity/order at backward")
        need(gradient is not None and gradient.device == executor.device
             and gradient.shape == (executor.bundle.cut.native_holds, 1600, 32),
             "Complete original context VJP required")
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
            if executor.device.type == "cuda":
                executor.explicit_receipt["CUDA_boundary_allocated_max_bytes"] = max(
                    executor.explicit_receipt["CUDA_boundary_allocated_max_bytes"],
                    torch.cuda.memory_allocated(executor.device))

        with executor.profile.stage("explicit_second_blocks_backward"):
            for index in reverse:
                z = executor.layout[index]
                first, last = cut.group_offsets[z[0]], cut.group_offsets[z[1]]
                boundary("explicit_second_block_backward_begin")
                with torch.enable_grad():
                    up_leaf = upward[index].detach().requires_grad_(True)
                    read_leaf = readout[first:last].detach().requires_grad_(True)
                    output = executor._second(z, up_leaf, read_leaf, eligible)
                    values = torch.autograd.grad(output, (up_leaf, read_leaf, *parameters),
                        gradient[z[0]:z[1]], allow_unused=True, create_graph=False)
                need(values[0] is not None and values[1] is not None,
                     "Both direct upward and readout derivatives must remain connected")
                direct_up_vjp[index] = values[0].detach()
                readout_vjp[first:last].copy_(values[1])
                accumulate(values[2:])
                executor.explicit_receipt["second_backward_blocks"] += 1
                del up_leaf, read_leaf, output, values
                boundary("explicit_second_block_backward_released")

        with executor.profile.stage("explicit_complete_history_backward"):
            boundary("explicit_complete_history_backward_begin")
            with torch.enable_grad():
                force_leaf = forcing.detach().requires_grad_(True)
                theta_leaf = model.memory.ell_theta.detach().to("cpu").requires_grad_(True)
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
            executor.explicit_receipt["full_history_backward_calls"] += 1
            del force_leaf, theta_leaf, history_readout, readout_vjp, theta_vjp, theta_contribution
            boundary("explicit_complete_history_backward_released")

        with executor.profile.stage("explicit_forcing_and_first_blocks_backward"):
            for index in reverse:
                z = executor.layout[index]
                first, last = cut.group_offsets[z[0]], cut.group_offsets[z[1]]
                boundary("explicit_forcing_block_backward_begin")
                with torch.enable_grad():
                    up_leaf = upward[index].detach().requires_grad_(True)
                    out_leaf = outside[index].detach().requires_grad_(True)
                    local_force = executor._forcing(z, up_leaf, out_leaf)
                    values = torch.autograd.grad(local_force, (up_leaf, out_leaf, *parameters),
                        force_vjp[first:last], allow_unused=True, create_graph=False)
                need(values[0] is not None and values[1] is not None,
                     "Both first-sweep output forcing derivatives remain connected")
                total_up = direct_up_vjp[index]
                total_up.add_(values[0])
                total_out = values[1].detach()
                accumulate(values[2:])
                executor.explicit_receipt["forcing_backward_blocks"] += 1
                del up_leaf, out_leaf, local_force, values
                boundary("explicit_first_block_backward_begin")
                with torch.enable_grad():
                    first_up, first_out = executor._first(z)
                    values = torch.autograd.grad((first_up, first_out), parameters,
                        (total_up, total_out), allow_unused=True, create_graph=False)
                accumulate(values)
                executor.explicit_receipt["first_backward_blocks"] += 1
                direct_up_vjp[index] = None
                del first_up, first_out, total_up, total_out, values
                boundary("explicit_first_block_backward_released")
        need(all(executor.explicit_receipt[key] == count for key in
            ("second_backward_blocks", "forcing_backward_blocks", "first_backward_blocks"))
            and executor.explicit_receipt["full_history_backward_calls"] == 1,
            "Every native tree block and complete history differentiated exactly once")
        executor.explicit_receipt["connected_trunk_parameters"] = sum(g is not None for g in parameter_vjp)
        return (None, *parameter_vjp)


class ExplicitResidentVJPExecution(ResidentContextExecution):
    def context_forward(self):
        with self.profile.stage("explicit_context_forward"):
            return _ExplicitResidentTrunk.apply(self, *tuple(self.model.parameters()))

    def explicit_vjp_snapshot(self):
        return dict(getattr(self, "explicit_receipt", {}))


def build_executor(model, bundle, metadata, **kwargs):
    """Standalone opt-in; original resident/integrated CUDA constructor guards remain."""
    return ExplicitResidentVJPExecution(model, bundle, metadata, **kwargs)

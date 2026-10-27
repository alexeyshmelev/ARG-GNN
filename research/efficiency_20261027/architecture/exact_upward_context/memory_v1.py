"""Joint released-upward plus preallocated context/VJP capacity, without CUDA allocation."""
import torch
from research.efficiency_20261027.architecture.exact_upward_release.memory_v1 import residency_plan as release_plan
from research.efficiency_20261027.architecture.exact_upward_residency.memory_v1 import capacity_check,DEFAULT_SCRATCH_BYTES
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need


def residency_plan(executor):
    plan=release_plan(executor)
    context=plan['native_holds']*1600*32*4
    block_context=max(z[1]-z[0] for z in executor.layout)*1600*32*4
    tile=context*executor.hap_tile//1600
    return dict(plan,complete_context_bytes=context,complete_VJP_bytes=context,
        largest_second_context_block_bytes=block_context,decoder_contiguous_tile_bytes=tile,
        decoder_tile_and_tile_gradient_bytes=2*tile,
        final_context_preallocated=True,full_context_concatenation=False,
        initial_additional_capacity_bytes=plan['initial_additional_capacity_bytes']+2*context,
        context_forcing_readout_device='context=model_device;forcing_and_history=CPU',
        bounded_second_output_and_decoder_tile_workspace_counted_inside_original16GiB_scratch=True)


def future_context_bytes(plan,context_ready,VJP_ready):
    need(type(context_ready) is bool and type(VJP_ready) is bool and (not VJP_ready or context_ready),
         'VJP cannot precede its complete preallocated context')
    return (2-int(context_ready)-int(VJP_ready))*plan['complete_context_bytes']


def admit_device(executor,plan,stage,live,future):
    need(executor.device.type=='cuda' and executor.device.index in (None,0),'Actual admitted cuda0 required')
    executor.tick('joint_context_residency_capacity_'+stage)
    need(type(live) is int and type(future) is int and 0<=live<=plan['maximum_simultaneous_bank_bytes']
         and 0<=future<=plan['maximum_simultaneous_bank_bytes']-live,'Exact released-upward live/future counters')
    contexts=future_context_bytes(plan,executor._context_ready,executor._VJP_ready)
    free,total=torch.cuda.mem_get_info(executor.device)
    allocated=torch.cuda.memory_allocated(executor.device)
    reserved=torch.cuda.memory_reserved(executor.device)
    required=future+contexts+plan['scratch_reserve_bytes']
    need(allocated+required<=executor.resident_soft_bytes,
         'Current allocation plus all future bank/context/VJP and16GiB scratch fit ROOT soft budget')
    result=capacity_check(free_bytes=free,total_bytes=total,allocated_bytes=allocated,reserved_bytes=reserved,
        future_bank_bytes=future+contexts,scratch_bytes=plan['scratch_reserve_bytes'])
    return dict(result,stage=stage,live_bank_bytes=live,future_bank_bytes=future,
        future_context_VJP_bytes=contexts,context_buffer_ready=executor._context_ready,VJP_buffer_ready=executor._VJP_ready,
        final_context_preallocated=True,soft_budget_bytes=executor.resident_soft_bytes,GPU_capacity_checked=True)

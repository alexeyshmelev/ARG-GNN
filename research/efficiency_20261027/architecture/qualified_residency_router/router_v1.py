"""Choose an already-qualified CPU-context executor before any learned forward.

The complete CPU layout is acquired first. Selection then uses actual prepared
counts and live CUDA counters. Both routes retain the original full-parent
equations and schedule. There is no OOM catch, partial-update retry, CUDA cache
flush, observation disk cache, or GPU-context route.
"""
GIB = 2**30
SCRATCH_BYTES = 16 * GIB
MAX_SOFT_BYTES = 74 * GIB
RESIDENT = 'upward_direct_resident_CPUcontext512'
EXPLICIT = 'explicit_CPUcontext512'
FACTORIES = {
    RESIDENT: 'research.efficiency_20261027.architecture.exact_upward_residency.large_resident_executor_v1:build_executor',
    EXPLICIT: 'research.efficiency_20261027.execution.large_block_executors:build_executor',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def select_capacity(*, native_nodes, native_holds, free_bytes, total_bytes,
                    allocated_bytes, reserved_bytes, resident_soft_bytes,
                    resident_scratch_bytes=SCRATCH_BYTES):
    """Pure integer admission. A failed resident screen selects explicit CPU."""
    values = (native_nodes, native_holds, free_bytes, total_bytes,
              allocated_bytes, reserved_bytes, resident_soft_bytes,
              resident_scratch_bytes)
    need(all(type(value) is int and value >= 0 for value in values),
         'Capacity values must be nonnegative Python integers, never booleans')
    need(native_holds > 0 and native_nodes >= 1600 * native_holds,
         'Positive complete original1600-sample native counts required')
    need(0 < resident_soft_bytes <= MAX_SOFT_BYTES
         and resident_scratch_bytes == SCRATCH_BYTES,
         'Explicit soft budget at most74GiB and unchanged16GiB scratch required')
    need(total_bytes > 0 and free_bytes <= total_bytes
         and allocated_bytes <= reserved_bytes <= total_bytes
         and free_bytes + reserved_bytes <= total_bytes,
         'Coherent live CUDA free/total and allocator counters required')
    available = free_bytes + reserved_bytes - allocated_bytes
    baseline_soft = allocated_bytes + resident_scratch_bytes
    need(baseline_soft <= resident_soft_bytes
         and resident_scratch_bytes <= available,
         'Even explicitCPU512 scratch does not fit; reject before either forward')
    upward = native_nodes * 32 * 4
    additional = 2 * upward + resident_scratch_bytes
    soft_margin = resident_soft_bytes - allocated_bytes - additional
    capacity_margin = available - additional
    failures = []
    if soft_margin < 0:
        failures.append('resident_soft_budget')
    if capacity_margin < 0:
        failures.append('resident_physical_capacity')
    route = EXPLICIT if failures else RESIDENT
    return dict(schema='TreeLoom_pre_forward_CPUcontext_capacity_route_v1',
        selected_route=route, selected_factory=FACTORIES[route],
        selection_reason=failures or ['resident_fits_actual_counts_and_live_capacity'],
        native_nodes=native_nodes, native_holds=native_holds, haplotypes=1600,
        holds_per_block=512, haplotypes_per_gru=64, history_holds_per_window=1024,
        width=32, element_bytes=4, upward_bank_bytes=upward,
        resident_two_bank_bytes=2*upward, scratch_reserve_bytes=resident_scratch_bytes,
        current_allocated_bytes=allocated_bytes, current_reserved_bytes=reserved_bytes,
        driver_free_bytes=free_bytes, device_total_bytes=total_bytes,
        allocator_cached_unallocated_bytes=reserved_bytes-allocated_bytes,
        available_for_new_allocations_bytes=available, soft_budget_bytes=resident_soft_bytes,
        resident_required_additional_bytes=additional,
        resident_soft_margin_bytes=soft_margin,
        resident_capacity_margin_bytes=capacity_margin,
        explicit_soft_margin_bytes=resident_soft_bytes-baseline_soft,
        explicit_capacity_margin_bytes=available-resident_scratch_bytes,
        selection_before_forward=True, full_parent_unchanged=True,
        context_residency='CPU', no_mid_update_fallback=True,
        allocation_screen_is_not_reserved_peak_proof=True,
        cached_bytes_are_not_contiguity_guarantee=True)


def actual_counts(bundle, metadata):
    """Read existing CPU shape and count metadata, never scientific values."""
    need(bundle.bank.parent.device.type == 'cpu', 'Complete original bank remains CPU')
    nodes = bundle.bank.parent.numel()
    holds = bundle.native_holds
    counts = metadata['episode_metadata']['native_nodes_per_tree']
    need(type(holds) is int and holds > 0 and bundle.cut.native_holds == holds
         and bundle.cut.haplotypes == 1600
         and isinstance(counts, (list, tuple)) and len(counts) == holds
         and all(type(value) is int and value >= 1600 for value in counts)
         and sum(counts) == nodes,
         'Actual complete native per-hold counts must match the prepared CPU bank')
    return nodes, holds


def device_snapshot(device):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return dict(free_bytes=free, total_bytes=total,
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device))


def build_executor(model, bundle, metadata, *, resident_soft_bytes,
                   resident_scratch_bytes=SCRATCH_BYTES, **kwargs):
    """Compatible executor factory with a required caller-declared soft budget.

    The caller records executor.capacity_route_receipt in its update result.
    Existing external worker/family/GPU admission remains mandatory. This
    factory chooses storage execution only and never mutates optimizer state.
    """
    import torch
    from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
    from research.efficiency_20261027.execution.large_block_executors import build_executor as explicit_factory
    from research.efficiency_20261027.architecture.exact_upward_residency.large_resident_executor_v1 import build_executor as resident_factory

    need(type(resident_soft_bytes) is int and 0 < resident_soft_bytes <= MAX_SOFT_BYTES
         and type(resident_scratch_bytes) is int and resident_scratch_bytes == SCRATCH_BYTES,
         'Explicit soft budget at most74GiB and original16GiB scratch required')
    expected = dict(holds_per_block=512, haplotypes_per_gru=64, history_holds_per_window=1024)
    for key, value in expected.items():
        need(type(kwargs.get(key, value)) is int and kwargs.get(key, value) == value,
             'Both routes require the qualified512/64/1024 partitions')
        kwargs[key] = value
    need('resident_context' not in kwargs,
         'Both qualified routes retain CPU context; no alternate context flag')
    parameters = tuple(model.parameters())
    need(parameters and all(p.dtype == torch.float32 and p.device == parameters[0].device
         for p in parameters) and parameters[0].device.type == 'cuda'
         and parameters[0].device.index in (None, 0),
         'Original FP32 model on the admitted logical cuda0 required')
    nodes, holds = actual_counts(bundle, metadata)
    cache = kwargs.get('layout_cache')
    need(isinstance(cache, LargeBlockLayoutCache), 'Qualified actual-RSS governed large layout cache required')
    check = kwargs.get('resource_check', lambda label: None)
    need(callable(check), 'Existing resource admission callback required')
    # Hold a temporary layout lease so the selected factory's acquire is a fast
    # bind. The potentially costly CPU layout build precedes the live snapshot.
    lease = cache.acquire(bundle, metadata, holds_per_block=512, resource_check=check)
    try:
        check('qualified_capacity_route_before_forward')
        receipt = select_capacity(native_nodes=nodes, native_holds=holds,
            resident_soft_bytes=resident_soft_bytes,
            resident_scratch_bytes=resident_scratch_bytes,
            **device_snapshot(parameters[0].device))
        if receipt['selected_route'] == RESIDENT:
            executor = resident_factory(model, bundle, metadata,
                resident_scratch_bytes=resident_scratch_bytes, **kwargs)
        else:
            executor = explicit_factory(model, bundle, metadata, resident_context=False, **kwargs)
        executor.capacity_route_receipt = dict(receipt, parent_id=bundle.parent_id,
            CPU_layout_prepared_before_live_snapshot=True,
            existing_executor_capacity_checks_preserved=True)
        return executor
    finally:
        lease.release()

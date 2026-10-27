"""Count-based resident-bank capacity admission and scalar lifetime accounting.

This is an allocator-capacity screen, not a measured total CUDA peak proof.
The fixed 16GiB scratch floor is additional to both exact FP32 node banks.
No cache flush, allocator configuration change, GPU allocation or fallback.
"""
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need

DEFAULT_SCRATCH_BYTES = 16 * 2**30
WIDTH = 32
ELEMENT_BYTES = 4


def capacity_check(*, free_bytes, total_bytes, allocated_bytes, reserved_bytes,
                   future_bank_bytes, scratch_bytes):
    values = (free_bytes, total_bytes, allocated_bytes, reserved_bytes,
              future_bank_bytes, scratch_bytes)
    need(all(type(x) is int and x >= 0 for x in values)
         and 0 < total_bytes and free_bytes <= total_bytes
         and allocated_bytes <= reserved_bytes <= total_bytes
         and free_bytes + reserved_bytes <= total_bytes,
         'Coherent actual CUDA free/total and allocator byte counters')
    need(scratch_bytes >= DEFAULT_SCRATCH_BYTES,
         'Original explicit sixteen-GiB scratch floor cannot be reduced')
    reusable = reserved_bytes - allocated_bytes
    available = free_bytes + reusable
    required = future_bank_bytes + scratch_bytes
    need(required <= available,
         'Exact resident banks plus scratch do not fit actual CUDA capacity; no partial-bank fallback')
    return dict(driver_free_bytes=free_bytes, device_total_bytes=total_bytes,
        current_allocated_bytes=allocated_bytes, current_reserved_bytes=reserved_bytes,
        allocator_cached_unallocated_bytes=reusable,
        available_for_new_allocations_bytes=available,
        future_bank_bytes=future_bank_bytes, scratch_reserve_bytes=scratch_bytes,
        required_additional_bytes=required, capacity_margin_bytes=available-required,
        cached_bytes_are_capacity_not_contiguity_guarantee=True)


def residency_plan(executor):
    need(executor.bank.parent.device.type == 'cpu', 'Original observation bank remains CPU')
    parameters = tuple(executor.model.parameters())
    need(parameters and all(p.device == executor.device and p.dtype == torch.float32
                            for p in parameters), 'Original FP32 parameter/device contract')
    nodes = executor.bank.parent.numel()
    holds = executor.bundle.cut.native_holds
    need(type(nodes) is int and nodes > 0 and type(holds) is int and holds > 0,
         'Complete positive native node and hold counts')
    next_node = next_hold = 0
    block_nodes = []
    for z in executor.layout:
        a, b, lo, hi = z[:4]
        need(all(type(v) is int for v in (a, b, lo, hi))
             and a == next_hold and lo == next_node and a < b <= holds and lo < hi <= nodes,
             'Contiguous complete native layout partition without duplicates')
        next_hold, next_node = b, hi
        block_nodes.append(hi-lo)
    need(next_node == nodes and next_hold == holds and block_nodes,
         'Resident plan covers every actual native node and hold exactly once')
    scratch = executor.resident_scratch_bytes
    need(type(scratch) is int and scratch >= DEFAULT_SCRATCH_BYTES,
         'Finite explicit scratch allowance with unchanged sixteen-GiB floor')
    bank_bytes = nodes * WIDTH * ELEMENT_BYTES
    return dict(native_nodes=nodes, native_holds=holds, blocks=len(block_nodes),
        largest_block_nodes=max(block_nodes), width=WIDTH, element_bytes=ELEMENT_BYTES,
        upward_bank_bytes=bank_bytes, direct_upward_gradient_bank_bytes=bank_bytes,
        maximum_simultaneous_bank_bytes=2*bank_bytes,
        scratch_reserve_bytes=scratch, initial_additional_capacity_bytes=2*bank_bytes+scratch,
        upward_lifetime='grows during first forward; saved by autograd through backward return',
        direct_gradient_lifetime='grows during reversed second VJPs; full across complete history VJP; released blockwise after first VJP',
        context_forcing_readout_device='cpu', whole_bank_concatenation=False,
        measured_CUDA_peak_proof=False)


def admit_device(executor, plan, stage, live_bank_bytes, future_bank_bytes):
    need(executor.device.type == 'cuda' and executor.device.index in (None, 0),
         'Residency admission requires the originally admitted logical cuda0')
    executor.tick('upward_residency_capacity_'+stage)
    need(type(live_bank_bytes) is int and type(future_bank_bytes) is int
         and 0 <= live_bank_bytes <= plan['maximum_simultaneous_bank_bytes']
         and 0 <= future_bank_bytes <= plan['maximum_simultaneous_bank_bytes']-live_bank_bytes,
         'Exact nonoverlapping resident bank lifetime counters')
    free, total = torch.cuda.mem_get_info(executor.device)
    result = capacity_check(free_bytes=free, total_bytes=total,
        allocated_bytes=torch.cuda.memory_allocated(executor.device),
        reserved_bytes=torch.cuda.memory_reserved(executor.device),
        future_bank_bytes=future_bank_bytes, scratch_bytes=plan['scratch_reserve_bytes'])
    return dict(result, stage=stage, live_bank_bytes=live_bank_bytes,
        GPU_capacity_checked=True, no_allocator_flush=True)


class ResidencyLedger:
    """Only integers/storage addresses; never an additional tensor owner."""
    def __init__(self, plan):
        self.bank_bytes = plan['upward_bank_bytes']
        self.upward_bytes = self.direct_bytes = self.direct_produced_bytes = 0
        self.peak_bytes = 0
        self.upward_storages = set()
        self.direct_storages = set()

    @property
    def live_bytes(self):
        return self.upward_bytes + self.direct_bytes

    @property
    def future_bytes(self):
        return 2*self.bank_bytes - self.upward_bytes - self.direct_produced_bytes

    def _storage(self, tensor):
        amount = tensor.numel()*tensor.element_size()
        need(tensor.dtype == torch.float32 and tensor.ndim == 2 and tensor.shape[1] == WIDTH
             and tensor.shape[0] > 0 and not tensor.requires_grad
             and tensor.is_contiguous() and tensor.storage_offset() == 0
             and tensor.untyped_storage().nbytes() == amount,
             'Each resident block owns exactly its FP32 payload without a hidden backing bank')
        key = (tensor.device.type, tensor.device.index, tensor.untyped_storage().data_ptr())
        return amount, key

    def add_upward(self, tensor):
        amount, key = self._storage(tensor)
        need(key not in self.upward_storages and key not in self.direct_storages,
             'Distinct private upward block storage required')
        need(self.upward_bytes+amount <= self.bank_bytes, 'No duplicate upward bank allocation')
        self.upward_storages.add(key)
        self.upward_bytes += amount
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)

    def add_direct(self, tensor):
        amount, key = self._storage(tensor)
        need(self.upward_bytes == self.bank_bytes
             and key not in self.upward_storages and key not in self.direct_storages,
             'Direct gradient storage must be separate from immutable saved upward blocks')
        need(self.direct_produced_bytes+amount <= self.bank_bytes, 'No duplicate direct-gradient bank')
        self.direct_storages.add(key)
        self.direct_bytes += amount
        self.direct_produced_bytes += amount
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)

    def release_direct(self, tensor):
        amount, key = self._storage(tensor)
        need(key in self.direct_storages and self.direct_bytes >= amount,
             'Release each owned direct-gradient block once')
        self.direct_storages.remove(key)
        self.direct_bytes -= amount

    def require_upward_complete(self):
        need(self.upward_bytes == self.bank_bytes, 'Complete actual-node upward bank required')

    def require_both_complete(self):
        self.require_upward_complete()
        need(self.direct_bytes == self.direct_produced_bytes == self.bank_bytes
             and self.future_bytes == 0, 'Both complete banks span the full history VJP')

    def require_direct_released(self):
        need(self.direct_bytes == 0 and not self.direct_storages
             and self.direct_produced_bytes == self.bank_bytes and self.future_bytes == 0,
             'Every direct-gradient block released after its final first-sweep VJP')

    def snapshot(self):
        return dict(upward_bytes=self.upward_bytes, direct_gradient_bytes=self.direct_bytes,
            direct_gradient_produced_bytes=self.direct_produced_bytes,
            live_bank_bytes=self.live_bytes, future_bank_bytes=self.future_bytes,
            peak_bank_bytes=self.peak_bytes, stores_tensor_references=False,
            saved_upward_release='autograd releases saved tensors after backward returns')

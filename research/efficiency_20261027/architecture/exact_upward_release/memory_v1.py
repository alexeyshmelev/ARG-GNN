"""One-bank replacement lifetime, with original16GiB scratch and exact native counts."""
import torch
from research.efficiency_20261027.architecture.exact_upward_residency.memory_v1 import (
    residency_plan as original_plan, admit_device, DEFAULT_SCRATCH_BYTES)
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need


def residency_plan(executor):
    plan=original_plan(executor)
    block=plan['largest_block_nodes']*plan['width']*plan['element_bytes']
    maximum=plan['upward_bank_bytes']+block
    return dict(plan,maximum_simultaneous_bank_bytes=maximum,largest_block_bytes=block,
        initial_additional_capacity_bytes=maximum+plan['scratch_reserve_bytes'],
        upward_lifetime='Private single-use blocks released after their final second VJP and all local aliases die',
        direct_gradient_lifetime='Replaces upward block by block; complete direct bank spans full history; released after each first VJP',
        saved_tensor_hooks_supported=False,repeated_backward_supported=False)


def storage(tensor):
    amount=tensor.numel()*tensor.element_size()
    backing=tensor.untyped_storage()
    need(tensor.dtype==torch.float32 and tensor.ndim==2 and tensor.shape[1]==32
         and tensor.shape[0]>0 and not tensor.requires_grad and tensor.grad_fn is None
         and tensor._base is None and tensor.is_contiguous() and tensor.storage_offset()==0
         and backing.nbytes()==amount,'Private complete detached FP32 block storage required')
    return amount,(tensor.device.type,tensor.device.index,backing.data_ptr())


class ReplacementLedger:
    """Scalar active-storage accounting; retired addresses may be reused, never live ones."""
    def __init__(self,plan):
        self.bank=plan['upward_bank_bytes'];self.bound=plan['maximum_simultaneous_bank_bytes']
        self.count=plan['blocks'];self.upward={};self.direct={};self.live_keys=set()
        self.upward_produced=0;self.direct_produced=0;self.peak=0;self.phase='forward'
        self.upward_releases=0;self.direct_releases=0;self.storage_generations=0
    @property
    def live_bytes(self):return sum(x[0] for x in self.upward.values())+sum(x[0] for x in self.direct.values())
    @property
    def future_bytes(self):return max(0,self.bound-self.live_bytes) if self.phase in ('forward','second') else 0
    def add_upward(self,index,tensor):
        amount,key=storage(tensor)
        need(self.phase=='forward' and index==len(self.upward) and key not in self.live_keys
             and self.upward_produced+amount<=self.bank,'Distinct complete forward block order')
        self.upward[index]=(amount,key);self.live_keys.add(key);self.upward_produced+=amount
        self.storage_generations+=1;self.peak=max(self.peak,self.live_bytes)
    def require_upward_complete(self):
        need(len(self.upward)==self.count and self.upward_produced==self.bank,'Complete actual upward bank')
    def begin_second(self):
        self.require_upward_complete();need(self.phase=='forward','Single-use second VJP phase');self.phase='second'
    def add_direct(self,index,tensor):
        amount,key=storage(tensor)
        need(self.phase=='second' and index==self.count-1-len(self.direct)
             and index in self.upward and self.upward[index][0]==amount and key not in self.live_keys,
             'Exact reverse replacement; direct gradients cannot alias any live saved block')
        self.direct[index]=(amount,key);self.live_keys.add(key);self.direct_produced+=amount;self.storage_generations+=1
        self.peak=max(self.peak,self.live_bytes)
        need(self.live_bytes<=self.bound,'One-bank plus largest-block payload bound')
    def release_upward(self,index):
        need(self.phase=='second' and index in self.upward and index in self.direct,'Release only after exact second VJP')
        _,key=self.upward.pop(index);self.live_keys.remove(key);self.upward_releases+=1
    def require_direct_complete(self):
        need(not self.upward and len(self.direct)==self.count and self.direct_produced==self.bank
             and self.upward_releases==self.count,'All original blocks released; complete direct bank only')
        self.phase='history'
    def begin_first(self):
        need(self.phase=='history','Full history must precede first VJPs');self.phase='first'
    def release_direct(self,index,tensor):
        amount,key=storage(tensor)
        need(self.phase=='first' and self.direct.get(index)==(amount,key)
             and index==self.count-1-self.direct_releases,'Each direct block released once in original reverse order')
        self.direct.pop(index);self.live_keys.remove(key);self.direct_releases+=1
    def require_released(self):
        need(not self.upward and not self.direct and not self.live_keys and self.direct_releases==self.count,
             'All resident block storage released after final VJPs')
        self.phase='finished'
    def abort(self):
        self.upward.clear();self.direct.clear();self.live_keys.clear();self.phase='failed'
    def snapshot(self):
        return dict(upward_bytes=sum(x[0] for x in self.upward.values()),
            direct_gradient_bytes=sum(x[0] for x in self.direct.values()),direct_gradient_produced_bytes=self.direct_produced,
            live_bank_bytes=self.live_bytes,future_bank_bytes=self.future_bytes,peak_bank_bytes=self.peak,
            upward_blocks_released=self.upward_releases,direct_blocks_released=self.direct_releases,
            storage_generations=self.storage_generations,phase=self.phase,stores_tensor_references=False,
            saved_upward_release='After each second VJP, before complete history backward',
            pointer_reuse_allowed_only_after_previous_storage_release=True)

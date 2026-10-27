"""Opt-in one-parent immutable block inputs; no learned state or context residency.

The original _block returns these same values/dtypes/orders on every call.
This pack copies them once and retains them through one explicit trunk backward.
Bindings deliberately do not survive bank restoration or parameter updates:
the next complete parent gets a fresh pack. No LRU, launcher or production hook.
"""
from types import SimpleNamespace
import threading
import weakref
import numpy as np
import torch


def need(ok,message):
    if not ok:raise ValueError(message)


def tensor_stamp(value):
    need(isinstance(value,torch.Tensor),'Exact source tensor required')
    return (id(value),value.data_ptr(),value._version,tuple(value.shape),
        tuple(value.stride()),str(value.dtype),str(value.device),value.requires_grad)


def source_stamp(bank):
    names=('parent','node_time','sample_index','edge_geometry','branch_support_true',
           'reference_class','sample_age','spans')
    tensors=tuple(getattr(bank,name) for name in names)+tuple(bank.upward_levels)+tuple(bank.downward_levels)
    need(all(t.device.type=='cpu' and not t.requires_grad and t.grad_fn is None for t in tensors),
         'Immutable original CPU observation tensors required')
    return tuple(tensor_stamp(t) for t in tensors)


def layout_stamp(layout):
    result=[]
    for z in layout:
        need(type(z) is tuple and len(z)==11,'Exact immutable eleven-field executor block required')
        arrays=(z[4],*z[5],*z[6],*z[7],*z[8])
        stamps=[]
        for value in arrays:
            need(isinstance(value,np.ndarray) and value.dtype in (np.dtype('int32'),np.dtype('int64'))
                 and value.ndim==1 and value.flags.c_contiguous and not value.flags.writeable,
                 'Readonly contiguous exact integer layout arrays required')
            backing=value
            while isinstance(backing,np.ndarray):
                need(not backing.flags.writeable,'Readonly layout backing required')
                backing=backing.base
            need(isinstance(backing,bytes) or isinstance(backing,memoryview) and backing.readonly,
                 'Layout must be backed by immutable bytes, never a mutable external array')
            stamps.append((id(value),value.ctypes.data,value.shape,value.dtype.str))
        result.append((id(z),z[:4],z[9:11],tuple(stamps)))
    return tuple(result)


def tensors(block):
    return (block.parent,block.node_time,block.sample_index,*block.upward_levels,
        *block.upward_children,*block.upward_parent_rank,*block.downward_levels,
        block.edge_geometry,block.branch_support_true,block.reference_class,block.sample_age,block.spans)


def block_bytes(bank,z):
    a,b,lo,hi=z[:4]
    indices=sum(value.size*8 for value in (z[4],*z[5],*z[6],*z[7],*z[8]))
    values=(bank.node_time[lo:hi],bank.sample_index[a:b],bank.edge_geometry[lo:hi],
        bank.branch_support_true[lo:hi],bank.reference_class,bank.sample_age,bank.spans[a:b])
    return int(indices+sum(value.numel()*value.element_size() for value in values))


def admission_bounds(*,remaining_payload_bytes,scratch_reserve_bytes,allocated_bytes,reserved_bytes,
                     physical_free_bytes,soft_budget_bytes):
    values=(remaining_payload_bytes,scratch_reserve_bytes,allocated_bytes,reserved_bytes,
            physical_free_bytes,soft_budget_bytes)
    need(all(type(value) is int and value>=0 for value in values)
         and scratch_reserve_bytes>0 and soft_budget_bytes>0 and reserved_bytes>=allocated_bytes,
         'Exact finite nonnegative allocator and free-memory measurements required')
    additional=remaining_payload_bytes+scratch_reserve_bytes
    need(reserved_bytes+additional<=soft_budget_bytes,
         'Current reserved memory plus remaining pack and complete scratch reserve exceed soft budget')
    need(additional<=physical_free_bytes,
         'Physical free memory must cover remaining pack and complete scratch reserve')
    return dict(remaining_payload_bytes=remaining_payload_bytes,scratch_reserve_bytes=scratch_reserve_bytes,
        allocated_bytes=allocated_bytes,reserved_bytes=reserved_bytes,physical_free_bytes=physical_free_bytes,
        soft_budget_bytes=soft_budget_bytes,required_additional_bytes=additional,
        arithmetic_is_not_peak_guarantee=True)


class CUDAPackAdmission:
    """Conservative actual allocator/free-memory checks; never empties the allocator."""
    def __init__(self,*,device,soft_budget_bytes,scratch_reserve_bytes,resource_check):
        self.device=torch.device(device)
        need(self.device.type=='cuda' and self.device.index in (None,0)
             and callable(resource_check),'Admitted logical cuda0 and existing resource callback required')
        need(type(soft_budget_bytes) is int and 0<soft_budget_bytes<=64*2**30
             and type(scratch_reserve_bytes) is int and scratch_reserve_bytes>=16*2**30,
             'At most64GiB soft budget and at least16GiB independent complete scratch reserve required')
        self.soft=soft_budget_bytes
        self.scratch=scratch_reserve_bytes
        self.check=resource_check
        self.stream=torch.cuda.current_stream(self.device).cuda_stream
        self.records=[]

    def __call__(self,remaining_payload_bytes,stage):
        self.check(stage)
        need(torch.cuda.current_stream(self.device).cuda_stream==self.stream,
             'One explicit original CUDA stream required for pack lifetime')
        free,total=torch.cuda.mem_get_info(self.device)
        record=admission_bounds(remaining_payload_bytes=int(remaining_payload_bytes),
            scratch_reserve_bytes=self.scratch,allocated_bytes=torch.cuda.memory_allocated(self.device),
            reserved_bytes=torch.cuda.memory_reserved(self.device),physical_free_bytes=int(free),
            soft_budget_bytes=self.soft)
        self.records.append(dict(stage=stage,physical_total_bytes=int(total),**record))
        return record


class BlockInputPack:
    """Finite strong device storage plus a weak exact live-bank binding.

    All original input tensors remain CPU. Mutation/replacement, different
    provenance/layout/thread/stream, premature release and second-update reuse
    fail closed. A failed execution can explicitly abort, invalidating its graph.
    """
    def __init__(self,bank,layout,*,device,provenance,max_payload_bytes,admit,CPU_test_only=False):
        self.device=torch.device(device)
        need(self.device.type=='cuda' and self.device.index in (None,0)
             or self.device.type=='cpu' and CPU_test_only is True,
             'CUDA pack or explicitly labeled CPU-only qualification required')
        need(type(provenance) is str and bool(provenance) and callable(admit)
             and type(max_payload_bytes) is int and max_payload_bytes>0,'Explicit provenance and finite admission required')
        self.provenance=provenance
        self.layout=tuple(layout)
        need(bool(self.layout) and len({id(z) for z in self.layout})==len(self.layout),'Nonempty unique original block tuples required')
        self.bank=weakref.ref(bank)
        self.source_stamp=source_stamp(bank)
        self.layout_stamp=layout_stamp(self.layout)
        self.layout_id=id(self.layout)
        self.owner_thread=threading.get_ident()
        self.stream=torch.cuda.current_stream(self.device).cuda_stream if self.device.type=='cuda' else None
        self.block_count=len(self.layout)
        self.positions={id(z):index for index,z in enumerate(self.layout)}
        self.block_bytes=tuple(block_bytes(bank,z) for z in self.layout)
        self.total_bytes=sum(self.block_bytes)
        need(self.total_bytes<=max_payload_bytes,'Entire exact pack exceeds explicitly admitted payload cap')
        self.max_payload_bytes=max_payload_bytes
        self.admit=admit
        self.entries={}
        self.payload_bytes=0
        self.peak_payload_bytes=0
        self.copies=0
        self.hits=0
        self.phase='bound'
        self.admit(self.total_bytes,'hot_block_pack_before_first_allocation')

    def _binding(self,bank,provenance):
        need(self.phase not in ('closed','aborted'),'Closed or aborted pack cannot be reused')
        need(threading.get_ident()==self.owner_thread,'One original pack owner thread required')
        if self.device.type=='cuda':
            need(torch.cuda.current_stream(self.device).cuda_stream==self.stream,
                 'Pack cannot move to an unrecorded CUDA stream')
        need(bank is self.bank() and provenance==self.provenance
             and source_stamp(bank)==self.source_stamp and id(self.layout)==self.layout_id,
             'Original immutable bank, provenance and layout binding must remain exact')

    def begin_forward(self):
        need(self.phase=='bound','Pack is single-update; create a fresh pack for another parent update')
        self._binding(self.bank(),self.provenance)
        self.phase='forward_and_backward'

    def backward_complete(self):
        need(self.phase=='forward_and_backward','One original successful explicit backward required')
        self._binding(self.bank(),self.provenance)
        self.phase='backward_complete'

    def block(self,bank,z,*,provenance):
        self._binding(bank,provenance)
        need(self.phase in ('bound','forward_and_backward'),'Block input use outside the original complete update')
        index=self.positions.get(id(z))
        need(index is not None and z is self.layout[index] and layout_stamp((z,))[0]==self.layout_stamp[index],
             'Exact admitted immutable partition tuple required')
        if index in self.entries:
            block,stamp=self.entries[index]
            need(tuple(tensor_stamp(t) for t in tensors(block))==stamp,
                 'Retained input tensor was mutated or replaced')
            self.hits+=1
            return block
        self.admit(self.total_bytes-self.payload_bytes,'hot_block_pack_before_block_copy')
        a,b,lo,hi,p,levels,children,ranks,downward,_,_=z
        def index_tensor(value):
            return torch.tensor(value,dtype=torch.long,device=self.device)
        def copy_tensor(value):
            return value.to(device=self.device,copy=True)
        try:
            block=SimpleNamespace(parent=index_tensor(p),
                node_time=copy_tensor(bank.node_time[lo:hi]),
                sample_index=copy_tensor(bank.sample_index[a:b]-lo),
                upward_levels=tuple(index_tensor(x) for x in levels),
                upward_children=tuple(index_tensor(x) for x in children),
                upward_parent_rank=tuple(index_tensor(x) for x in ranks),
                downward_levels=tuple(index_tensor(x) for x in downward),
                edge_geometry=copy_tensor(bank.edge_geometry[lo:hi]),
                branch_support_true=copy_tensor(bank.branch_support_true[lo:hi]),
                reference_class=copy_tensor(bank.reference_class),
                sample_age=copy_tensor(bank.sample_age),spans=copy_tensor(bank.spans[a:b]))
            values=tensors(block)
            need(all(not value.requires_grad and value.grad_fn is None for value in values),
                 'Only observed and structural values may enter pack')
            actual=sum(value.numel()*value.element_size() for value in values)
            need(actual==self.block_bytes[index],'Exact dtype/shape payload accounting required')
            self._binding(bank,provenance)
            self.entries[index]=(block,tuple(tensor_stamp(t) for t in values))
            self.payload_bytes+=actual
            self.peak_payload_bytes=max(self.peak_payload_bytes,self.payload_bytes)
            self.copies+=1
            return block
        except BaseException:
            self.abort()
            raise

    def close(self):
        need(self.phase in ('bound','backward_complete','closed'),'Retain the pack through complete trunk backward; abort a failed graph explicitly')
        self.entries.clear()
        self.layout=()
        self.positions.clear()
        self.payload_bytes=0
        self.phase='closed'

    def abort(self):
        self.entries.clear()
        self.layout=()
        self.positions.clear()
        self.payload_bytes=0
        self.phase='aborted'

    def snapshot(self):
        return dict(phase=self.phase,blocks=self.block_count,block_copies=self.copies,cache_hits=self.hits,
            total_payload_bytes=self.total_bytes,retained_payload_bytes=self.payload_bytes,
            peak_payload_bytes=self.peak_payload_bytes,max_payload_bytes=self.max_payload_bytes,
            source_device='cpu',pack_device=str(self.device),learned_values_cached=False,
            CPU_context_boundary_preserved=True,restored_bank_reuse=False,
            cross_update_reuse=False,one_owner_thread_and_stream=True)

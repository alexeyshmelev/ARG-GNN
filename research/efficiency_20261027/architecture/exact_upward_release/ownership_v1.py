"""Context-owned blocks, never exported through executor state or saved_tensors."""
import weakref
import torch
from .memory_v1 import storage
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need


class PrivateBlocks:
    def __init__(self,device):
        self.device=device;self.blocks=[];self.stamps=[];self.retired=[];self.state='forward'
        self.stream=torch.cuda.current_stream(device).cuda_stream if device.type=='cuda' else None
    def add(self,tensor):
        amount,key=storage(tensor)
        need(self.state=='forward' and all(stamp[1]!=key for stamp in self.stamps),'Distinct private block ownership')
        need(tensor.device==self.device,'Original compute device')
        self.blocks.append(tensor);self.stamps.append((amount,key,tensor._version,tuple(tensor.shape)))
    def get(self,index):
        tensor=self.blocks[index]
        need(tensor is not None,'Consumed upward block cannot be read again')
        amount,key=storage(tensor)
        need((amount,key,tensor._version,tuple(tensor.shape))==self.stamps[index],
             'Private upward tensor shape/storage/version changed')
        return tensor
    def begin(self):
        need(self.state=='forward','Private upward context permits exactly one backward')
        if self.stream is not None:
            need(torch.cuda.current_stream(self.device).cuda_stream==self.stream,'Same autograd CUDA stream lifetime')
        for index in range(len(self.blocks)):self.get(index)
        self.state='backward'
    def release(self,index):
        need(self.state=='backward','Release only during this single backward')
        tensor=self.get(index)
        probes=(weakref.ref(tensor),weakref.ref(tensor.untyped_storage()))
        self.blocks[index]=None
        self.retired.append(probes)
        del tensor
        need(all(probe() is None for probe in probes),'Private upward tensor/storage still has an unexpected live alias')
    def require_released(self):
        need(self.state=='backward' and all(x is None for x in self.blocks)
             and all(probe() is None for probes in self.retired for probe in probes),'Every private upward block released')
    def finish(self):
        self.require_released();self.state='finished'
    def abort(self):
        self.blocks.clear();self.stamps.clear();self.state='failed'

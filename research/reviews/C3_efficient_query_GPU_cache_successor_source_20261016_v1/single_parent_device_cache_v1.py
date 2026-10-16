"""Owned single-parent device cache. Source-only, no numerical imports.
Only immutable raw observations survive within one complete-parent update.
"""
import time

def require(ok,message):
    if not ok:raise ValueError(message)

def tensors(value,torch):
    if torch.is_tensor(value):yield value
    elif isinstance(value,(tuple,list)):
        for item in value:yield from tensors(item,torch)

class SingleParentDeviceFeatureCacheV1:
    def __init__(self,provider,*,torch,device,fields,max_bytes,max_block_bytes,check):
        require(callable(check) and type(max_bytes)is int and max_bytes>0
                and type(max_block_bytes)is int and max_block_bytes>0,"Explicit actual device cache bounds")
        self.provider,self.torch,self.device,self.fields,self.check=provider,torch,device,tuple(fields),check
        self.max_bytes,self.max_block_bytes=max_bytes,max_block_bytes
        self.blocks=[];self.metadata=[];self.versions=[];self.used=0
        self.host_gets=0;self.copy_seconds=0.;self.hash_get_seconds=0.;self.closed=False
    def _signature(self,block):
        return tuple((id(t),t._version,t.dtype,tuple(t.shape),tuple(t.stride()),t.device)
                     for key in self.fields for t in tensors(block[key],self.torch))
    def prepare(self):
        require(not self.blocks and not self.closed,"One fresh single-parent cache")
        torch=self.torch
        for index in range(len(self.provider)):
            self.check("efficient_raw_device_cache_"+str(index))
            start=time.perf_counter();cpu=self.provider.get(index)
            self.hash_get_seconds+=time.perf_counter()-start;self.host_gets+=1
            values=[t for key in self.fields for t in tensors(cpu[key],torch)]
            require(values and all(t.device.type=="cpu" and not t.requires_grad for t in values),
                    "Original authenticated nonlearned CPU banks only")
            size=sum(t.numel()*t.element_size() for t in values)
            require(0<size<=self.max_block_bytes and self.used+size<=self.max_bytes,
                    "Prospective whole-parent device feature cache cap")
            meta={k:cpu[k].clone() for k in
                  ("focal_rows","focal_hold","focal_sample_column","query_column","reference_class")}
            start=time.perf_counter();bank=cpu.clone().to(self.device)
            self.copy_seconds+=time.perf_counter()-start
            require(all(t.device==self.device and not t.requires_grad
                        for key in self.fields for t in tensors(bank[key],torch)),
                    "Exact explicit device, categorical/FP32 source tensor copy")
            # Count every logical tensor separately, conservatively including aliases.
            actual=sum(t.numel()*t.element_size() for key in self.fields for t in tensors(bank[key],torch))
            require(actual==size,"Shape/dtype preserving raw bank transfer")
            self.blocks.append(bank);self.metadata.append(meta)
            self.versions.append(self._signature(bank));self.used+=actual
            del cpu,values,bank,meta
        require(len(self.blocks)==len(self.provider),"Every complete physical block cached, no fallback")
    def get(self,index):
        require(not self.closed and type(index)is int and 0<=index<len(self.blocks),"Owned cached block")
        bank=self.blocks[index]
        require(self._signature(bank)==self.versions[index],"Raw device bank tensor identity/version unchanged")
        return bank
    def unchanged(self,index):
        self.get(index)
    def diagnostics(self):
        return dict(host_provider_gets=self.host_gets,raw_device_transfers=self.host_gets,
                    retained_logical_device_feature_bytes=self.used,active_parents=1,
                    hash_get_seconds=self.hash_get_seconds,copy_call_seconds=self.copy_seconds,
                    copy_timing_is_host_call_elapsed_not_separate_CUDA_event_measurement=True,
                    GPU_content_rehashed_each_use=False,immutable_source_copy_and_tensor_versions_checked=True)
    def clear(self):
        self.blocks.clear();self.metadata.clear();self.versions.clear();self.used=0;self.closed=True

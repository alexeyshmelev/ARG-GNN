
"""One-thread prepare/compact prefetch without expanded-bank IPC or eviction."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import threading
from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import (
    pack_bundle,restore_bundle)

def need(ok,message):
    if not ok:raise ValueError(message)

class MemoryGovernor:
    """Reserve complete CPU phases before work, with actual RSS rechecks."""
    def __init__(self,*,maximum_bytes,rss_bytes,resource_check,reserve_bytes):
        self.maximum=maximum_bytes;self.rss=rss_bytes;self.check=resource_check
        self.reserve=reserve_bytes;self.active={};self.condition=threading.Condition()
        self.closed=False

    @contextmanager
    def phase(self,label,required_bytes=None):
        amount=self.reserve if required_bytes is None else required_bytes
        token=object()
        with self.condition:
            while True:
                need(not self.closed,"Prefetch governor closed")
                self.check("pipelined_admit_"+label)
                current=self.rss()
                if current+amount+sum(self.active.values())<=self.maximum:
                    self.active[token]=amount
                    break
                if not self.active:
                    gc.collect()
                    need(self.rss()+amount<=self.maximum,
                         "Complete-parent RAM reserve unavailable; no eviction or reduced workload")
                self.condition.wait(timeout=1.)
        try:
            yield
        finally:
            with self.condition:
                self.active.pop(token,None)
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed=True;self.condition.notify_all()

class PipelinedCompactParents:
    """Every parent prepares once; one lookahead, complete compact records in RAM."""
    def __init__(self,*,parent_bounds,input_refs,external_arrays,max_cache_bytes,
                 max_rss_bytes,phase_reserve_bytes,rss_bytes,resource_check,event,
                 prepare,pack=pack_bundle,restore=restore_bundle):
        need(sum(parent_bounds.values())<=max_cache_bytes,"All20 compact parent bounds must fit upfront")
        need(max_cache_bytes+phase_reserve_bytes<max_rss_bytes,"Original full-parent working reserve required")
        self.bounds=dict(parent_bounds);self.inputs=input_refs;self.external=external_arrays
        self.maximum=max_cache_bytes;self.check=resource_check;self.event=event;self.prepare=prepare
        self.pack=pack;self.restore=restore;self.records={};self.preparations={};self.restorations={}
        self.charged=0;self.lock=threading.RLock();self.closed=False
        self.governor=MemoryGovernor(maximum_bytes=max_rss_bytes,rss_bytes=rss_bytes,
            resource_check=resource_check,reserve_bytes=phase_reserve_bytes)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="TreeLoom_observation_prefetch")
        self.future=None;self.future_parent=None

    def _producer_check(self,stage):
        need(not self.closed,"Cooperative CPU producer stop")
        self.check(stage)

    def _produce(self,parent):
        with self.governor.phase("prepare_and_lossless_pack"):
            with self.lock:
                need(parent not in self.preparations and parent not in self.records,
                     "No repeated complete-parent preparation")
                need(self.charged+self.bounds[parent]<=self.maximum,"Compact record admission must fit")
                self.preparations[parent]=1
            bundle,metadata=self.prepare(parent,resource_check=self._producer_check)
            record=self.pack(bundle,metadata,input_refs=self.inputs[parent],
                external_arrays=self.external.get(parent),resource_check=self._producer_check)
            need(record["charged_bytes"]<=self.bounds[parent],"Actual compact record exceeds authenticated bound")
            del bundle,metadata
            gc.collect()
            with self.lock:
                self.records[parent]=record
                self.charged+=record["charged_bytes"]
                parents=len(self.records);charged=self.charged
            self.event("compact_parent_ready",dict(parent_id=parent,parents=parents,
                compact_charged_bytes=record["charged_bytes"],pool_charged_bytes=charged,
                expanded_bank_sent_over_IPC=False,disk_cache_bytes=0))
            return parent

    def prefetch(self,parent):
        need(parent in self.bounds and not self.closed,"Only registered TRAIN parent prefetch")
        with self.lock:
            if self.future is not None and self.future.done():
                self.future.result();self.future=None;self.future_parent=None
            if parent in self.records:return False
            if self.future is not None:
                need(self.future_parent==parent,"Only one lookahead parent may be in flight")
                return False
            self.future_parent=parent
            self.future=self.executor.submit(self._produce,parent)
        self.event("observation_prefetch_started",dict(parent_id=parent,CPU_workers=1))
        return True

    def load(self,parent):
        with self.lock:
            present=parent in self.records
        if not present:
            self.prefetch(parent)
            future=self.future
            need(future is not None and self.future_parent==parent,"Exact needed parent future required")
            need(future.result()==parent,"Prepared future must retain its parent")
        with self.lock:
            if self.future_parent==parent:
                need(self.future is not None and self.future.done(),"Needed future must finish")
                self.future.result();self.future=None;self.future_parent=None
            record=self.records[parent]
        with self.governor.phase("exact_restore",record["read_peak_upper_bound_bytes"]):
            bundle,metadata=self.restore(record,input_refs=self.inputs[parent],resource_check=self.check)
        with self.lock:
            self.restorations[parent]=self.restorations.get(parent,0)+1
        return bundle,metadata

    def prefetch_next(self,parents):
        with self.lock:
            if self.future is not None:
                if not self.future.done():return False
                self.future.result();self.future=None;self.future_parent=None
            parent=next((p for p in parents if p not in self.records),None)
            if parent is None:return False
        return self.prefetch(parent)

    def training_phase(self):
        return self.governor.phase("complete_parent_training")

    def receipt(self):
        with self.lock:
            return dict(schema="TreeLoom_lazy_compact_prefetch_pool_v1",parents=len(self.records),
                compact_charged_bytes=self.charged,cap_bytes=self.maximum,
                preparations=dict(self.preparations),restorations=dict(self.restorations),
                CPU_prefetch_workers=1,lookahead_parents=1,cold_rebuilds=0,
                evictions=0,disk_cache_bytes=0,expanded_tensor_IPC=False)

    def close(self):
        self.closed=True
        self.governor.close()
        self.executor.shutdown(wait=True,cancel_futures=True)

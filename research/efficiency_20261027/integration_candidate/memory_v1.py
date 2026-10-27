"""Actual-RSS admission shared by the compact pool and immutable layout LRU.

Retained-byte accounting is never treated as released physical memory. Every
phase is admitted against actual RSS plus all outstanding phase reservations.
Only unleased immutable plans can be evicted. There is no workload fallback.
"""
from contextlib import contextmanager
import gc
import threading
from research.scalability_20261024.integration.cache import LayoutCache, _identity
from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.pool import (
    PipelinedCompactParents, need)

LAYOUT_CACHE_BYTES = 80*2**30


class LayoutMemoryGovernor:
    def __init__(self, *, maximum_bytes, rss_bytes, resource_check, reserve_bytes):
        need(type(maximum_bytes) is int and maximum_bytes > 0
             and type(reserve_bytes) is int and reserve_bytes > 0,
             'Explicit complete-parent memory envelope required')
        self.maximum=maximum_bytes
        self.rss=rss_bytes
        self.check=resource_check
        self.reserve=reserve_bytes
        self.condition=threading.Condition(threading.RLock())
        self.active={}
        self.owners={}
        self.closed=False
        self.cache=None
        self.eviction_checks=0
        self.admissions=0
        self.waits=0
        self.peak_reserved=0

    @contextmanager
    def phase(self, label, required_bytes=None):
        amount=self.reserve if required_bytes is None else required_bytes
        need(type(amount) is int and amount >= 0, 'Finite nonnegative phase reserve required')
        token=object()
        owner=threading.get_ident()
        with self.condition:
            while True:
                need(not self.closed, 'Memory governor closed')
                self.check('layout_governor_admit_'+label)
                required=amount+sum(self.active.values())
                if self.rss()+required <= self.maximum:
                    self.active[token]=amount
                    self.owners[token]=owner
                    self.admissions+=1
                    self.peak_reserved=max(self.peak_reserved,sum(self.active.values()))
                    break
                self.eviction_checks+=1
                if self.cache is not None and self.cache.evict_one_unleased():
                    gc.collect()
                    # Re-read real RSS; allocator retention may defeat eviction.
                    continue
                gc.collect()
                if self.rss()+required <= self.maximum:
                    continue
                other_owners=any(value != owner for value in self.owners.values())
                need(other_owners,
                     'Actual RSS plus complete working reserves unavailable after layout eviction; '
                     'no reduced parent and no wait on own nested training reservation')
                self.waits+=1
                self.condition.wait(timeout=1.)
        try:
            yield
        finally:
            with self.condition:
                self.active.pop(token,None)
                self.owners.pop(token,None)
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed=True
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(maximum_bytes=self.maximum,actual_RSS_bytes=self.rss(),
                active_reserved_bytes=sum(self.active.values()),active_phases=len(self.active),
                admissions=self.admissions,waits=self.waits,eviction_checks=self.eviction_checks,
                peak_reserved_bytes=self.peak_reserved,actual_RSS_rechecked_after_eviction=True,
                own_thread_nested_wait_forbidden=True)


class _GovernedLayoutMixin:
    def __init__(self, governor, *, max_bytes=LAYOUT_CACHE_BYTES):
        need(governor.cache is None, 'One shared layout cache per pool governor')
        self.governor=governor
        self._working_allowance=0
        self._protected_keys=set()
        super().__init__(max_bytes=max_bytes,admit=self._admit)
        governor.cache=self
        self.phase_evictions=0

    def _admit(self, amount, stage):
        need(type(amount) is int and 0 <= amount <= self._working_allowance,
             'Layout allocation must fit the explicit active working reservation at '+stage)
        self.governor.check(stage)
        need(self.governor.rss() <= self.governor.maximum, 'Actual layout RSS cap exceeded')

    def acquire(self,bundle,metadata,*,holds_per_block,resource_check=lambda label:None):
        # This lock also protects plan dictionaries from background phase eviction.
        with self.governor.condition:
            key=_identity(bundle,metadata,holds_per_block)
            counts=metadata['episode_metadata']['native_nodes_per_tree']
            if key in self.entries:
                # Content bind plus a finite lease/control allowance.
                amount=64*len(counts)+16*2**20
            else:
                _,amount=self.estimate(bundle.bank,counts,holds_per_block)
                amount+=16*2**20
            self._protected_keys.add(key)
            try:
                with self.governor.phase('layout_build_or_bind',int(amount)):
                    self._working_allowance=int(amount)
                    try:
                        return super().acquire(bundle,metadata,holds_per_block=holds_per_block,
                                               resource_check=resource_check)
                    finally:
                        self._working_allowance=0
            finally:
                self._protected_keys.discard(key)

    def evict_one_unleased(self):
        with self.governor.condition:
            for key,entry in list(self.entries.items()):
                if entry['leases'] or key in self._protected_keys:
                    continue
                del self.entries[key]
                self.stats['evictions']+=1
                self.phase_evictions+=1
                self._refresh()
                return True
            return False

    def _release(self,key,charge):
        with self.governor.condition:
            super()._release(key,charge)
            self.governor.condition.notify_all()

    def _expire(self,key,bank_id):
        with self.governor.condition:
            super()._expire(key,bank_id)

    def clear(self):
        with self.governor.condition:
            super().clear()

    def snapshot(self):
        with self.governor.condition:
            return dict(super().snapshot(),phase_admission_evictions=self.phase_evictions,
                        actual_memory_admission=self.governor.snapshot())


class LayoutAwareParents(PipelinedCompactParents):
    def __init__(self, *, layout_governor, **kwargs):
        super().__init__(**kwargs)
        need(layout_governor.maximum == kwargs['max_rss_bytes']
             and layout_governor.reserve == kwargs['phase_reserve_bytes'],
             'Pool and layout reservations must share the original memory envelope')
        self.governor.close()
        self.governor=layout_governor


class GovernedLayoutCache(_GovernedLayoutMixin, LayoutCache):
    pass


class GovernedLargeLayoutCache(_GovernedLayoutMixin, LargeBlockLayoutCache):
    pass

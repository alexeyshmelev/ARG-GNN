"""Finite immutable-index LRU for an explicitly budgeted successor runtime.

No observation tensor is strongly retained. Cold build/restore admission remains
part of the caller's declared host-memory reserve; this cache does not increase
an observation-cache cap or bypass a resource guard.
"""
from collections import OrderedDict
from dataclasses import fields, is_dataclass
import json
import sys
import weakref
import numpy as np
import torch
from research.scalability_20261024.execution.layout_plan import (
    LayoutPlan, BoundLayout, _stamp)


def _charge(value, seen=None):
    """Retained Python/array bytes; weak-binding callback gets finite padding."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.base is not None:
            total += _charge(value.base, seen)
    elif isinstance(value, dict):
        total += sum(_charge(k, seen)+_charge(v, seen) for k,v in value.items())
    elif isinstance(value, (tuple,list)):
        total += sum(_charge(v, seen) for v in value)
    elif is_dataclass(value):
        total += sum(_charge(getattr(value,f.name),seen) for f in fields(value))
    elif isinstance(value, weakref.ReferenceType):
        total += 512
    return total


def _tensors(bank, member_hold):
    return (bank.parent,bank.sample_index,*bank.upward_levels,
            *bank.downward_levels,member_hold)


def _identity(bundle, metadata, holds_per_block):
    refs = []
    for name in ("observation_ref","focal_registry_ref","native_preparation_ref"):
        value = getattr(bundle,name,None)
        if value is None:
            value = metadata.get(name)
        if value is not None:
            refs.append((name,json.dumps(value,sort_keys=True,separators=(",",":"))))
    if not refs:
        raise ValueError("Explicit observed provenance is required for layout reuse")
    scope = json.dumps(metadata.get("layout_cache_scope","complete_parent"),
                       sort_keys=True,separators=(",",":"))
    bank = bundle.bank
    return (str(bundle.parent_id),tuple(refs),scope,int(holds_per_block),
            int(bundle.native_holds),int(bank.parent.numel()),
            tuple(bank.sample_index.shape))


class LayoutLease:
    """Pins a charged plan until all executor recomputation has completed."""
    __slots__ = ("cache","key","layout","charge")
    def __init__(self,cache,key,layout,charge):
        self.cache,self.key,self.layout,self.charge = cache,key,layout,charge

    def release(self):
        cache = self.cache
        if cache is not None:
            self.layout = ()
            self.cache = None
            cache._release(self.key,self.charge)

    def __del__(self):
        self.release()


class LayoutCache:
    """Bounded LRU; plans survive restores, bindings expire with the CPU bank.

    max_bytes is an explicit additional host reserve, including plan/binding
    metadata. admit(bytes,stage) may reject cold working memory before allocation.
    External admission must account for existing cache residency as well.
    No automatic fallback silently changes the configured execution mode.
    """
    def __init__(self, *, max_bytes, admit=None, max_bindings_per_plan=1):
        if (type(max_bytes) is not int or max_bytes <= 8192
                or type(max_bindings_per_plan) is not int
                or max_bindings_per_plan < 1):
            raise ValueError("Explicit finite layout-cache byte/binding budgets required")
        if admit is not None and not callable(admit):
            raise ValueError("Readable caller admission callback required")
        self.max_bytes = max_bytes
        self.admit = admit
        self.max_bindings_per_plan = max_bindings_per_plan
        self.entries = OrderedDict()
        self.retained_bytes = 4096
        self.peak_retained_bytes = 4096
        self.stats = dict(builds=0,restored_content_binds=0,fast_hits=0,evictions=0,
                          expired_bindings=0,rejected_mutations=0)

    def _refresh(self):
        self.retained_bytes = 4096+sum(e["base_charge"]+
            sum(x["charge"] for x in e["bindings"].values())+e["lease_charge"]
            for e in self.entries.values())
        self.peak_retained_bytes = max(self.peak_retained_bytes,self.retained_bytes)

    def _evict_for(self, needed, keep=None):
        self._refresh()
        for key in list(self.entries):
            if self.retained_bytes+needed <= self.max_bytes:
                break
            if key == keep or self.entries[key]["leases"]:
                continue
            del self.entries[key]
            self.stats["evictions"] += 1
            self._refresh()
        if self.retained_bytes+needed > self.max_bytes:
            raise MemoryError("Layout cache declared host reserve would be exceeded")

    def _lease(self,key,layout):
        # Only tuple views/control are new: immutable array backing is already
        # charged in the non-evictable cache entry.
        charge = 512+sys.getsizeof(layout)+sum(sys.getsizeof(z) for z in layout)
        self._evict_for(charge,keep=key)
        if self.admit is not None:
            self.admit(charge,"layout_executor_lease")
        entry = self.entries[key]
        entry["leases"] += 1
        entry["lease_charge"] += charge
        self._refresh()
        return LayoutLease(self,key,layout,charge)

    def _release(self,key,charge):
        entry = self.entries.get(key)
        if entry is None or entry["leases"] <= 0:
            raise RuntimeError("A leased layout entry must remain resident")
        entry["leases"] -= 1
        entry["lease_charge"] -= charge
        self._refresh()

    def _expire(self,key,bank_id):
        entry = self.entries.get(key)
        if entry is not None:
            if entry["bindings"].pop(bank_id,None) is not None:
                self.stats["expired_bindings"] += 1
                self._refresh()

    @staticmethod
    def estimate(bank,counts,holds_per_block):
        """Finite operational cold-build reserve, not a proven allocator peak.

        Includes eventual int32 indices, full height/depth/assigned arrays and
        a generous128 bytes per largest-block node for simultaneous int64
        parent/groups/ranks, sorting scratch and immutable conversion. Caller
        headroom/admission and measured RSS guards remain required.
        """
        n = bank.parent.numel()
        blocks = (len(counts)+holds_per_block-1)//holds_per_block
        arrays = blocks*(1+3*len(bank.upward_levels)+len(bank.downward_levels))
        retained = 20*n+512*arrays+64*len(counts)+16384
        largest = max(sum(counts[i:i+holds_per_block])
                      for i in range(0,len(counts),holds_per_block))
        working = retained+5*n+128*largest
        return int(retained),int(working)

    def acquire(self,bundle,metadata,*,holds_per_block,resource_check=lambda label:None):
        bank = bundle.bank
        counts = metadata["episode_metadata"]["native_nodes_per_tree"]
        member_hold = bundle.episodes.member_hold
        if not all(isinstance(x,torch.Tensor) and x.device.type=="cpu"
                   for x in _tensors(bank,member_hold)):
            raise ValueError("Fast cache requires original immutable CPU tensors")
        key = _identity(bundle,metadata,holds_per_block)
        entry = self.entries.get(key)
        bank_id = id(bank)
        if entry is not None:
            self.entries.move_to_end(key)
            existing = entry["bindings"].get(bank_id)
            if existing is not None and existing["bank_ref"]() is bank:
                try:
                    layout = existing["bound"].assert_binding(bank,counts,member_hold)
                except ValueError:
                    self.stats["rejected_mutations"] += 1
                    raise
                self.stats["fast_hits"] += 1
                resource_check("layout_cache_fast_hit")
                return self._lease(key,layout)
        if entry is None:
            retained,working = self.estimate(bank,counts,holds_per_block)
            self._evict_for(retained)
            if self.admit is not None:
                self.admit(working,"layout_cold_build")
            resource_check("layout_cache_before_cold_build")
            plan = LayoutPlan.build(bank,counts,member_hold,
                                    holds_per_block=holds_per_block)
            entry = dict(plan=plan,bindings=OrderedDict(),leases=0,lease_charge=0,
                         base_charge=_charge(plan)+_charge(key)+512)
            # A new plan was just derived from these exact immutable tensors:
            # no second full topology hash is required for its initial binding.
            bound = BoundLayout(plan,_stamp(bank,counts,member_hold))
            self.stats["builds"] += 1
        else:
            if self.admit is not None:
                self.admit(64*len(counts)+16384,"layout_restored_binding")
            resource_check("layout_cache_before_restored_content_bind")
            bound = entry["plan"].bind(bank,counts,member_hold)
            self.stats["restored_content_binds"] += 1
        # Binding metadata has only weak bank ownership and scalar tensor stamps.
        owner = weakref.ref(self)
        def expired(_):
            cache = owner()
            if cache is not None:
                cache._expire(key,bank_id)
        bank_ref = weakref.ref(bank,expired)
        binding = dict(bank_ref=bank_ref,bound=bound,
                       charge=_charge((bound.observation_stamp,bank_ref))+512)
        while len(entry["bindings"]) >= self.max_bindings_per_plan:
            entry["bindings"].popitem(last=False)
        self._refresh()
        # Never remove an active entry even transiently during admission.
        if key in self.entries:
            self._evict_for(binding["charge"],keep=key)
        else:
            needed = entry["base_charge"]+binding["charge"]
            self._evict_for(needed)
        entry["bindings"][bank_id] = binding
        self.entries[key] = entry
        self._refresh()
        resource_check("layout_cache_bound")
        return self._lease(key,bound.assert_binding(bank,counts,member_hold))

    def clear(self):
        if any(e["leases"] for e in self.entries.values()):
            raise RuntimeError("Release active executor layouts before clearing cache")
        self.entries.clear()
        self._refresh()

    def snapshot(self):
        self._refresh()
        return dict(max_bytes=self.max_bytes,retained_bytes=self.retained_bytes,
                    peak_retained_bytes=self.peak_retained_bytes,
                    cached_plans=len(self.entries),
                    active_executor_leases=sum(e["leases"] for e in self.entries.values()),
                    live_bank_bindings=sum(len(e["bindings"]) for e in self.entries.values()),
                    max_bindings_per_plan=self.max_bindings_per_plan,
                    admission_callback_present=self.admit is not None,**self.stats)

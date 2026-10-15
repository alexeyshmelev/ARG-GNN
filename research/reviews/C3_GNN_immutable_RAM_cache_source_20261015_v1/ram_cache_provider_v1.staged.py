"""Source-only bounded immutable CPU features; original scientific code is reused."""
from __future__ import annotations
import copy, resource, time
from arggnn.american_c3_common_v1 import require, ref
from research.reviews.C3_GNN_prefetch_isolated980_root_20261015_v2.persistent_provider_v1 import (
    PersistentSealedObservationProviderV1, PersistentObservationProviderFactoryV1, TRAIN)
from research.reviews.C3_GNN_prefetch_isolated980_root_20261015_v2.prefetch_provider_isolated_v2 import (
    _buffer_feature_identity)

SOURCE="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/ram_cache_provider_v1.py"
GiB=2**30
CACHE_CAPS=dict(per_parent_logical_bytes=2*GiB,aggregate_logical_bytes=40*GiB,
    main_RSS_cap_bytes=88*GiB,main_noncache_planning_allowance_bytes=48*GiB,
    main_block_duplicate_planning_allowance_bytes=GiB,
    aggregate_RSS_cap_bytes=120*GiB,root_RSS_cap_bytes=128*GiB,
    worker_count=2,worker_RSS_cap_bytes=8*GiB)

def tensor_byte_bound_v1(plan):
    """Tensor payload upper bound from the exact complete observed plan, no data."""
    require(plan["haplotypes"]==1600 and plan["focal_records"]==10000
        and plan["class_order"]==["EUR","EAS","AFR"],"Exact C3 observation cardinalities")
    n=sum(b["nodes"] for b in plan["blocks"]);t=plan["trees"];s=plan["sites"];b=len(plan["blocks"])
    # Node tensors <=100N: parent8,time4,level/index arrays<=32,geometry16,support40.
    # samples8TH; sites int8 genotype SH + donor96/mask6/panel16/entropy4/hold8.
    # hold arrays24T; query/reference/date27200 per block; three focal indexes24F.
    return 100*n+8*t*1600+(1600+130)*s+24*t+27200*b+24*10000

class _RAMBudgetV1:
    def __init__(self,caps):
        require(caps==CACHE_CAPS,"Exact reviewed RAM ownership and headroom caps")
        self.caps=copy.deepcopy(caps);self.entries={};self.by_parent={};self.used=0;self.peak=0
    def reserve(self,parent,index,size):
        key=(parent,index);require(parent in TRAIN and key not in self.entries and type(size)is int and size>0,"One complete immutable bank per physical block")
        used=self.by_parent.get(parent,0)
        require(used+size<=self.caps["per_parent_logical_bytes"] and self.used+size<=self.caps["aggregate_logical_bytes"],"Explicit per-parent and all20 logical CPU cache caps")
        self.entries[key]=size;self.by_parent[parent]=used+size;self.used+=size;self.peak=max(self.peak,self.used)
    def release_parent(self,parent):
        for key in [k for k in self.entries if k[0]==parent]:self.used-=self.entries.pop(key)
        self.by_parent.pop(parent,None)
        require(self.used>=0,"Exact retained RAM ownership accounting")
    def snapshot(self):
        return dict(logical_bytes=self.used,peak_logical_bytes=self.peak,parents=dict(self.by_parent),entries=len(self.entries),caps=copy.deepcopy(self.caps))

class ImmutableRAMObservationProviderV1(PersistentSealedObservationProviderV1):
    def __init__(self,*args,ram_budget,**kwargs):
        require(kwargs.get("cache_bytes")==CACHE_CAPS["per_parent_logical_bytes"],"Root selected exact2GiB per-parent immutable feature cache")
        self.ram_budget=ram_budget;self._sealing=False;self._sealed_cache_complete=False;self._cache_poisoned=False
        super().__init__(*args,**kwargs)
        self.tensor_byte_bound=tensor_byte_bound_v1(self.plan)
        require(self.tensor_byte_bound<=self.cache_bytes,"Whole original parent fits without eviction or spill")
    def _remember(self,index,block,digest,size):
        require(self._sealing and index not in self.cache and len(digest)==64
            and size<=self.options["max_block_bytes"] and self.used+size<=self.tensor_byte_bound,"Only original checked seal features enter RAM")
        self.ram_budget.reserve(self.plan["parent_id"],index,size)
        try:self.cache[index]=(block,digest,size);self.used+=size
        except BaseException:
            self.ram_budget.release_parent(self.plan["parent_id"]);self.cache.clear();self.used=0;raise
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=CACHE_CAPS["main_RSS_cap_bytes"],"Cache plus every main-process allocation fits88GiB")
    def _read(self,index):
        require(self._sealing and not self._cache_poisoned,"No CPU feature rebuild after the required seal")
        block,digest,size=super()._read(index)
        self._remember(index,block,digest,size)
        return block,digest,size
    def seal(self,*,resource_check):
        require(not self._sealing and not self.cache and self.seal_record is None and not self._cache_poisoned,"One required full observation seal")
        self._sealing=True
        try:
            value=super().seal(resource_check=resource_check)
            require(set(self.cache)==set(range(len(self))) and self.used<=self.tensor_byte_bound
                and self.ram_budget.by_parent.get(self.plan["parent_id"])==self.used,"Every sealed complete block retained before any truth")
            self._sealed_cache_complete=True
            return value
        except BaseException:
            self._release(failed=True);raise
        finally:self._sealing=False
    def get(self,index):
        try:
            require(type(index)is int and 0<=index<len(self) and self._sealed_cache_complete
                and not self._sealing and not self._cache_poisoned and self.seal_record is not None
                and index in self.cache,"Only complete required-seal banks; no fallback or eviction")
            started=time.perf_counter();block,digest,size=self.cache[index]
            actual,actual_size=_buffer_feature_identity(block)
            self.counters["cache_hash_seconds"]+=time.perf_counter()-started
            require(actual==digest==self.seal_record["block_feature_sha256"][index] and actual_size==size
                and block.plan_ref==self.plan["plan_ref"] and block.parent_id==self.plan["parent_id"]
                and block.block_index==index and block.start_tree==self.plan["blocks"][index]["start_tree"]
                and block.stop_tree==self.plan["blocks"][index]["stop_tree"],"Full cached tensor content, original identity and seal digest on every reuse")
            self.counters["cache_hits"]+=1;self.cache.move_to_end(index)
            # Bind the same authenticated worker context even for a cache-only parent.
            # No prefetch dispatch is useful because every physical block is retained.
            self._pool()
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=CACHE_CAPS["main_RSS_cap_bytes"],"Main-process complete RAM/graph/runtime cap")
            return block
        except BaseException:
            self._release(failed=True);raise
    def clear_retained_cache(self):
        self.ram_budget.release_parent(self.plan["parent_id"]);self.cache.clear();self.used=0;self._sealed_cache_complete=False
    def _release(self,*,failed):
        try:
            row=self.manager.release(self,failed=failed)
            if row is not None:self.parent_releases.append(row)
        except BaseException:
            self._cache_poisoned=True;raise
        finally:
            self.engine=None;self.last_index=None;self.direction=1
            self.dataset._state=None;self.dataset._pid=None
            if failed:self._cache_poisoned=True
            if self._cache_poisoned:self.clear_retained_cache()
    def diagnostics(self):
        value=super().diagnostics()
        value["immutable_RAM_feature_cache"]=dict(retained_bytes=self.used,tensor_byte_bound=self.tensor_byte_bound,
            fully_sealed=self._sealed_cache_complete,no_feature_rebuild_after_seal=True,
            full_digest_each_reuse=True,model_inputs_cloned_by_unchanged_stream_consumer=True,
            logical_cache_budget=self.ram_budget.snapshot())
        return value

class ImmutableRAMObservationProviderFactoryV1(PersistentObservationProviderFactoryV1):
    def __init__(self,*,operational_caps,immutable_cache_caps,resource_check):
        require(immutable_cache_caps==CACHE_CAPS and operational_caps["num_workers"]==2
            and operational_caps["worker_rss_cap_bytes"]==8*GiB
            and operational_caps["max_block_bytes"]==128*2**20
            and operational_caps["aggregate_rss_cap_bytes"]==120*GiB,"Exact2workers/8GiB and complete reviewed RAM bounds")
        self.ram_budget=_RAMBudgetV1(immutable_cache_caps);self.parent_tensor_bounds={}
        super().__init__(operational_caps=operational_caps,resource_check=resource_check)
        self.execution_binding["source_refs"].append(ref(SOURCE))
        self.execution_binding["immutable_RAM_feature_cache"]=dict(caps=copy.deepcopy(CACHE_CAPS),
            parent_tensor_byte_bounds=self.parent_tensor_bounds,required_seal_populates_cache=True,
            full_digest_every_get=True,CPU_features_only_no_learned_context_cache=True,
            no_disk_spill=True,all620_worker_context_releases_preserved=True)
    def __call__(self,plan_ref,*,parent_id,worker_tmp_directory,cache_bytes):
        require(parent_id not in self.providers,"One original immutable RAM provider per parent")
        value=ImmutableRAMObservationProviderV1(plan_ref,parent_id=parent_id,worker_tmp_directory=worker_tmp_directory,
            cache_bytes=cache_bytes,persistent_manager=self.manager,resource_check=self.resource_check,
            ram_budget=self.ram_budget,**self.options)
        self.providers[parent_id]=value;self.parent_tensor_bounds[parent_id]=value.tensor_byte_bound
        require(sum(self.parent_tensor_bounds.values())<=CACHE_CAPS["aggregate_logical_bytes"],"All registered complete features fit logical RAM cap")
        self.execution_binding["registered_parent_plan_refs"][parent_id]=copy.deepcopy(plan_ref)
        return value
    def finalize(self,*,training_failed):
        retained_before=self.ram_budget.snapshot()
        try:
            # Original factory enforces20seal+600step releases and actual worker joins.
            result=super().finalize(training_failed=training_failed)
            result["immutable_RAM_feature_cache_before_final_clear"]=retained_before
            return result
        finally:
            for provider in self.providers.values():provider.clear_retained_cache()
            require(self.ram_budget.used==0 and not self.ram_budget.entries,"Every retained CPU bank released after final worker drain")

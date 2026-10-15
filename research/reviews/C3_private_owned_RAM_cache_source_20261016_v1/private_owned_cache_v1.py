"""Additive private-owned C3 CPU cache; old admitted provider remains frozen."""
from __future__ import annotations
from pathlib import Path
import copy,hashlib,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_private_owned_RAM_cache_source_20261016_v1"
SOURCE=REL+"/private_owned_cache_v1.py"
BASE="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/ram_cache_provider_v1.py"
BASE_SHA="7de3ce76df415f20056bdba0ef2b4a2fe48ea1938b270fa0fa4384719ac23936"
if hashlib.sha256((ROOT/BASE).read_bytes()).hexdigest()!=BASE_SHA:raise ValueError("Exact original failed-bound RAM cache source")
from research.reviews.C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3 import ram_cache_provider_v1 as original
from research.reviews.C3_private_owned_RAM_cache_source_20261016_v1.private_owned_tensors_v1 import clone_block_private_v1
from arggnn.american_c3_common_v1 import require,ref
from arggnn.american_c3_gnn_dataset_isolated_v2 import BANK_FIELDS,BLOCK_TENSORS
import torch
require(Path(original.__file__).resolve()==ROOT/BASE,"Exact original base module identity")
CACHE_CAPS=copy.deepcopy(original.CACHE_CAPS)

class PrivateOwnedRAMObservationProviderV1(original.ImmutableRAMObservationProviderV1):
    def __init__(self,*args,**kwargs):
        self.private_storage_records={};self.private_clone_seconds=0.
        super().__init__(*args,**kwargs)
    def _remember(self,index,block,digest,size):
        require(self._sealing and index not in self.cache and len(digest)==64 and type(size) is int and 0<size<=self.options["max_block_bytes"]
            and self.used+size<=self.tensor_byte_bound,"Check original cache envelope before private allocation")
        started=time.perf_counter()
        owned,record=clone_block_private_v1(block,tensor_fields=BANK_FIELDS+BLOCK_TENSORS,expected_digest=digest,
            expected_size=size,identity=original._buffer_feature_identity,torch_api=torch)
        self.private_clone_seconds+=time.perf_counter()-started
        # Original cap/accounting/commit/RSS validation remains the final authority.
        super()._remember(index,owned,digest,size)
        self.private_storage_records[index]=record
    def _read(self,index):
        incoming,digest,size=super()._read(index)
        # Base dynamic _remember already installed the owned block. Return it to
        # the required seal too, so seal's NumPy focal view cannot retain IPC storage.
        owned,cached_digest,cached_size=self.cache[index]
        require(cached_digest==digest and cached_size==size,"Exact original sealed return identity")
        return owned,digest,size
    def clear_retained_cache(self):
        try:super().clear_retained_cache()
        finally:self.private_storage_records.clear()
    def diagnostics(self):
        out=super().diagnostics()
        out["private_owned_CPU_feature_storage"]=dict(clone_on_receipt=True,retained_IPC_storage_aliases=False,
            tensor_shape_stride_storage_offset_preserved=True,full_original_digest_after_clone=True,
            private_storage_bytes=sum(x["private_storage_bytes"] for x in self.private_storage_records.values()),
            private_storage_count=sum(x["private_storage_count"] for x in self.private_storage_records.values()),
            received_shared_storage_count=sum(x["incoming_shared_storage_count"] for x in self.private_storage_records.values()),
            clone_seconds=self.private_clone_seconds,records=len(self.private_storage_records))
        return out

class PrivateOwnedRAMObservationProviderFactoryV1(original.ImmutableRAMObservationProviderFactoryV1):
    def __init__(self,*,operational_caps,immutable_cache_caps,resource_check):
        super().__init__(operational_caps=operational_caps,immutable_cache_caps=immutable_cache_caps,resource_check=resource_check)
        self.execution_binding["source_refs"].extend([ref(SOURCE),ref(REL+"/private_owned_tensors_v1.py")])
        self.execution_binding["private_owned_CPU_feature_cache"]=dict(clone_all_incoming_tensor_storages=True,
            preserve_private_view_alias_graph=True,hash_equals_original_before_commit=True,
            unchanged_cache_caps=copy.deepcopy(CACHE_CAPS),no_worker_count_or_science_change=True,
            new_GPU_route_authorized=False,future_experiment_GPU=0)
    def __call__(self,plan_ref,*,parent_id,worker_tmp_directory,cache_bytes):
        require(parent_id not in self.providers,"One original immutable provider per parent")
        value=PrivateOwnedRAMObservationProviderV1(plan_ref,parent_id=parent_id,worker_tmp_directory=worker_tmp_directory,
            cache_bytes=cache_bytes,persistent_manager=self.manager,resource_check=self.resource_check,
            ram_budget=self.ram_budget,**self.options)
        self.providers[parent_id]=value;self.parent_tensor_bounds[parent_id]=value.tensor_byte_bound
        require(sum(self.parent_tensor_bounds.values())<=CACHE_CAPS["aggregate_logical_bytes"],"All20 complete original bank caps")
        self.execution_binding["registered_parent_plan_refs"][parent_id]=copy.deepcopy(plan_ref)
        return value

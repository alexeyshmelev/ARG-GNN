"""Bind tracking source while delegating the separately reviewed persistent factory."""
import copy
from arggnn.american_c3_common_v1 import ref
SOURCE="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/tracking_factory_v1.py"

class TrackingObservationProviderFactoryV1:
    def __init__(self,*,operational_caps,immutable_cache_caps,resource_check):
        from .ram_cache_provider_v1 import ImmutableRAMObservationProviderFactoryV1
        self.base=ImmutableRAMObservationProviderFactoryV1(operational_caps=operational_caps,immutable_cache_caps=immutable_cache_caps,resource_check=resource_check)
    @property
    def execution_binding(self):
        value=copy.deepcopy(self.base.execution_binding);value["source_refs"].append(ref(SOURCE))
        value["all_provider_drain_manifest_required"]=True
        return value
    def __call__(self,plan_ref,*,parent_id,worker_tmp_directory,cache_bytes):
        return self.base(plan_ref,parent_id=parent_id,worker_tmp_directory=worker_tmp_directory,cache_bytes=cache_bytes)
    def finalize(self,*,training_failed):
        return self.base.finalize(training_failed=training_failed)

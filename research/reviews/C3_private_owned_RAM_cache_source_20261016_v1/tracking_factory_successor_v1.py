"""Fresh-root factory hook only; no old root/route/source mutation."""
import copy
from arggnn.american_c3_common_v1 import ref
SOURCE="research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/tracking_factory_successor_v1.py"
class TrackingObservationProviderFactoryV1:
    def __init__(self,*,operational_caps,immutable_cache_caps,resource_check):
        from .private_owned_cache_v1 import PrivateOwnedRAMObservationProviderFactoryV1
        self.base=PrivateOwnedRAMObservationProviderFactoryV1(operational_caps=operational_caps,immutable_cache_caps=immutable_cache_caps,resource_check=resource_check)
    @property
    def execution_binding(self):
        out=copy.deepcopy(self.base.execution_binding);out["source_refs"].append(ref(SOURCE));out["all_provider_drain_manifest_required"]=True
        return out
    def __call__(self,plan_ref,*,parent_id,worker_tmp_directory,cache_bytes):
        return self.base(plan_ref,parent_id=parent_id,worker_tmp_directory=worker_tmp_directory,cache_bytes=cache_bytes)
    def finalize(self,*,training_failed):return self.base.finalize(training_failed=training_failed)

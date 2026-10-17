"""Single-parent eager replay or exact optimized preparation, never a resident LRU."""
import time,json
from arggnn.american_c3_gnn_dataset_v1 import load_observation
from research.reviews.frozen_TreeLoom_bounded_training_source_20261017_v1 import contract_v1 as c
from . import optimized_preparation_v1 as preparation
from . import cache_codec_v1 as codec
from . import reuse_contract_v1 as contract


class ParentLoaderV1:
    def __init__(self,q,storage,*,resource_check):
        self.q=q;self.tick=resource_check
        self.plan=contract.validate_input_extension(q,storage)
        self.manifest_ref=c.ref(contract.MANIFEST)
        self.reported=set()
        self.last_stage=None

    def stage_tick(self,label):
        self.tick(label)
        if label!=self.last_stage:
            print(json.dumps(dict(event="observed_preparation_stage",stage=label)),flush=True)
            self.last_stage=label

    def load(self,parent_id,item,cfg):
        self.tick("before_observed_preparation_or_exact_replay")
        start=time.monotonic()
        self.last_stage=None
        selected=()if self.plan is None else self.plan["selected_parent_ids"]
        key,identity=codec.identity_key_v1(parent_id,item,self.manifest_ref)
        if parent_id in selected:
            entry=self.plan["entries"][parent_id]
            c.require(entry["key"]==key and entry["identity"]==identity,
                "Original observed/focal/source cache identity before file authentication")
            # bound authenticates the entire immutable archive. No caller bypass.
            from arggnn.american_c3_common_v1 import bound
            self.stage_tick("verify_immutable_observed_archive")
            path=bound(entry["archive_ref"])
            with path.open("rb")as stream:
                bundle,metadata=codec.restore_v1(stream,key,identity,
                    max_tensor_bytes=entry["tensor_nbytes"],max_archive_bytes=entry["max_archive_bytes"])
            mode="exact_immutable_observed_archive_replay"
        else:
            state=load_observation(item["observation_ref"],item["focal_registry_ref"],cfg)
            bundle,metadata=preparation.prepare_from_loaded_observation_v1(state,split="train",
                observation_ref=item["observation_ref"],focal_registry_ref=item["focal_registry_ref"],
                native_preparation_ref=item["native_preparation_ref"],max_nodes=400000000,
                max_holds=100000,max_episode_payload_bytes=8*2**30,resource_check=self.stage_tick)
            del state
            mode="cache_cold_exact_optimized_whole_observation_prepare"
        if parent_id not in self.reported:
            descriptor=(dict(tensor_nbytes=entry["tensor_nbytes"],
                serialized_upper_bound_bytes=entry["max_archive_bytes"])
                if parent_id in selected else dict(
                    tensor_nbytes=sum(t.numel()*t.element_size()for t in
                        (*codec.f._bank_tensors(bundle.bank),*codec.f._episode_tensors(bundle.episodes),
                         *bundle.cut.tensors(),bundle.focal_hold,bundle.focal_sample_column)),
                    archive_materialized=False,serialization_size_measurement_available=False))
            c.put(self.q,"observed_preparation_measurement."+parent_id+".json",dict(
                schema="TreeLoom_actual_worker_observed_preparation_measurement_v1",
                parent_id=parent_id,mode=mode,elapsed_seconds=time.monotonic()-start,
                source_manifest_ref=self.manifest_ref,key=key,**descriptor,
                targets_opened=False,model_executed_for_measurement=False,
                qualification_cache_cold=self.q["phase"]=="qualification"))
            self.reported.add(parent_id)
        self.tick("after_observed_preparation_or_exact_replay")
        return bundle,metadata

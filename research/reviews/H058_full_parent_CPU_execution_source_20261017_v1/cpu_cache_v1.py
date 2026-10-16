"""CPU preparation reuse only. No model equations, file reader, launcher or authority.

The original observed-only H045 builder/authentication is called once per parent.
After sealing, only the active parent's immutable tensor versions are checked.
All packed payloads are built once with the unchanged original F1 binder.
"""
from __future__ import annotations
import json
from arggnn import train_ordered_clade_episode_v1 as h45
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.observation_binding_v3 import bank_stamps,episode_stamps
from research.reviews.biological_operator_next_fresh_20261016_v1.packed_csr_v2.same_marker_packed_csr_v2 import stamp
from .protocol_v1 import TRAIN,VAL,ARMS,require


def _keys(value):
    return json.dumps(value,separators=(",",":"))


def cache_preflight_v1(ctx,ids):
    require(tuple(ids)in(TRAIN,VAL),"Exact original full phase only")
    cfg=ctx["cfg"];rows=ctx["reader"].decode_json(cfg["source_count_ref"])["parents"]
    selected=[rows[pid]for pid in ids]
    forecast=dict(parents=len(ids),bank_bytes_upper=sum(r["CPU_GPU_original_bank_tensor_bytes_upper"]for r in selected),
        episode_bytes_upper=sum(r["original_H045_episode_tensor_upper_bytes"]for r in selected),
        packed_numeric_bytes_upper=2*sum(r["one_packed_layout_numeric_bytes"]for r in selected),
        packed_tensor_objects_upper=2*sum(r["one_packed_layout_Tensor_objects"]for r in selected),
        target_bytes=0 if tuple(ids)==VAL else sum(r["native_TRAIN_target_FP32_bytes"]for r in selected),
        Python_metadata_allowance_bytes=cfg["CPU_packed_python_allowance_bytes"])
    forecast["prepared_cache_bytes_upper_with_allowance"]=sum(forecast[k]for k in
        ("bank_bytes_upper","episode_bytes_upper","packed_numeric_bytes_upper","target_bytes","Python_metadata_allowance_bytes"))
    require(forecast["bank_bytes_upper"]<=cfg["CPU_bank_cap_bytes"]
        and forecast["episode_bytes_upper"]<=cfg["episode_cache_cap_bytes"]
        and forecast["packed_numeric_bytes_upper"]<=cfg["CPU_packed_cache_cap_bytes"]
        and forecast["packed_tensor_objects_upper"]<=cfg["CPU_packed_cache_max_tensor_objects"]
        and forecast["target_bytes"]<=cfg["CPU_target_cap_bytes"]
        and forecast["prepared_cache_bytes_upper_with_allowance"]<=cfg["CPU_prepared_cache_cap_bytes"],
        "All complete phase caches/counts inside explicit bounds; no eviction, crop, spill or fallback")
    return forecast


class FrozenEpisodeCacheV1(h45.EpisodeCache):
    """Keep the original episode builder/proofs; no repeated whole-bank byte hashes."""
    def __init__(self,cfg,guard,before_build):
        super().__init__(cfg,guard,before_build)
        self.frozen=False;self.identities={}

    def get(self,pid,bank,keys):
        self.guard()
        if self.frozen:
            self.validate_active_v1(pid,bank,keys)
            self.hits+=1
            return self.values[pid]
        previous=set(self.values)
        value=super().get(pid,bank,keys)
        require(previous<=set(self.values),"Complete CPU episode cache may never evict/rebuild")
        self.identities[pid]=(id(bank),bank_stamps(bank),id(value),episode_stamps(value),_keys(keys))
        return value

    def validate_active_v1(self,pid,bank,keys):
        require(self.frozen and pid in self.values and pid in self.identities,"Only sealed original CPU episodes")
        expected=self.identities[pid];value=self.values[pid]
        require((id(bank),bank_stamps(bank),id(value),episode_stamps(value),_keys(keys))==expected,
            "Active parent bank/episode/keys changed; stop without rebuilding or rehash fallback")

    def freeze_v1(self,banks,keys,ids):
        require(not self.frozen and set(self.values)==set(ids)==set(self.identities)
            and self.builds==len(ids),"All original phase episodes built exactly once before truth")
        self.frozen=True
        for pid in ids:self.validate_active_v1(pid,banks[pid],keys[pid])


class FrozenPackedCacheV1:
    """One immutable payload per phase parent/arm, no tensor copies on lookup.

    The model retains its original per-hold F1 binding checks. This cache adds only
    active bank/episode/global-scatter version checks around the scheduled step.
    It does not rewalk all packed tensors or all parents at each optimizer update.
    """
    def __init__(self,ctx,observed,ids,forecast):
        self.ctx=ctx;self.observed=observed;self.ids=tuple(ids);self.forecast=forecast
        self.values={};self.proofs={};self.numeric_bytes=0;self.tensor_objects=0
        self.sealed=False;self.active=None;self.build_seconds=0.

    def add_v1(self,pid,arm,payload,proof):
        require(not self.sealed and pid in self.ids and arm in ARMS and(pid,arm)not in self.values,
            "Exactly one original packed payload per parent/arm")
        self.numeric_bytes+=payload.route_cpu.numeric_tensor_bytes
        self.tensor_objects+=payload.route_cpu.tensor_object_count
        cfg=self.ctx["cfg"]
        require(self.numeric_bytes<=cfg["CPU_packed_cache_cap_bytes"]
            and self.tensor_objects<=cfg["CPU_packed_cache_max_tensor_objects"],
            "Complete packed-cache cap; no eviction or reconstruction fallback")
        payload.binding.validate(self.observed["banks"][pid],payload.episodes,payload.route_cpu)
        self.values[(pid,arm)]=payload;self.proofs[(pid,arm)]=dict(proof)
        self.build_seconds+=float(proof["seconds"])

    def seal_v1(self):
        required={(pid,arm)for pid in self.ids for arm in ARMS}
        require(not self.sealed and set(self.values)==required
            and self.numeric_bytes<=self.forecast["packed_numeric_bytes_upper"]
            and self.tensor_objects<=self.forecast["packed_tensor_objects_upper"],
            "All original parents/both routes fully cached before phase truth")
        # One complete phase check here, never a complete phase rewalk per update.
        for pid in self.ids:
            self.observed["episodes"].validate_active_v1(pid,self.observed["banks"][pid],self.observed["keys"][pid])
        self.sealed=True

    def validate_active_v1(self,pid,bank,episode):
        require(self.sealed and pid in self.ids and bank is self.observed["banks"][pid]
            and episode is self.observed["episodes"].values[pid],"Exact active cached CPU objects")
        self.observed["episodes"].validate_active_v1(pid,bank,self.observed["keys"][pid])

    def get_v1(self,pid,arm,bank,episode):
        self.validate_active_v1(pid,bank,episode)
        require((pid,arm)in self.values,"No new route construction after truth")
        payload=self.values[(pid,arm)]
        require(payload.episodes is episode,"Exact immutable episode payload")
        self.active=(pid,arm,id(bank),id(episode),id(payload),id(payload.route_cpu),stamp(payload.route_cpu.global_query_node))
        proof=dict(self.proofs[(pid,arm)],CPU_cache_lookup=True,CPU_payload_same_object=True,
            cached_preparation_seconds=self.proofs[(pid,arm)]["seconds"],seconds=0.,
            CPU_route_cache_parents=len(self.ids),route_built_once_before_phase_truth=True)
        return payload,proof

    def validate_scheduled_v1(self,bank,payload):
        require(self.active is not None,"Scheduled update must use active original cached route")
        pid,arm,bank_id,episode_id,payload_id,route_id,scatter=self.active
        self.validate_active_v1(pid,bank,payload.episodes)
        require(id(bank)==bank_id and id(payload.episodes)==episode_id and id(payload)==payload_id
            and id(payload.route_cpu)==route_id and stamp(payload.route_cpu.global_query_node)==scatter
            and self.values[(pid,arm)]is payload,"Scheduled step changed original cached payload")

    def receipt_v1(self):
        require(self.sealed,"Complete CPU cache before receipt")
        return dict(schema="H058_complete_CPU_preparation_cache_v1",parents=list(self.ids),arms=list(ARMS),
            parent_arm_payloads=len(self.values),forecast=self.forecast,
            actual_packed_numeric_bytes=self.numeric_bytes,actual_packed_tensor_objects=self.tensor_objects,
            episode_builds=self.observed["episodes"].builds,episode_bytes=self.observed["episodes"].bytes,
            packed_build_seconds=self.build_seconds,all_preparation_before_phase_truth=True,
            numeric_model_operations_or_optimizer_updates=0,original_builders_and_F1_binding_unchanged=True,
            active_parent_version_checks_only_between_updates=True,full_cache_rewalk_per_update=False,
            spill_eviction_rebuild_or_reduced_support=False,measured_full_training_fit=False,
            mutation_contract="Exclusive owned observation tensors; no external NumPy/storage aliases may mutate inputs. Torch storage/shape/stride/version and object identity are checked. Source/closure ownership must enforce the no-foreign-writer rule.")
